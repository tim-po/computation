"""One-command deployment tool for distributed column-parallel inference.

Handles the full workflow:
  1. Split checkpoint into shards
  2. Upload code + shards to remote GPU machines via SSH
  3. Install dependencies on remote machines
  4. Launch workers on remote machines
  5. Launch coordinator locally (or on a designated machine)

Usage:
    # Define your cluster in a simple YAML/JSON config, then:
    python -m distributed.deploy \
        --model h100_col8_drop25 \
        --checkpoint results/h100_col8_drop25_best.pt \
        --cluster cluster.json \
        --prompt "The future of artificial intelligence" \
        --generate --max-tokens 200

    # Or manually specify workers:
    python -m distributed.deploy \
        --model h100_col8_drop25 \
        --checkpoint results/h100_col8_drop25_best.pt \
        --workers "user@gpu1:22,user@gpu2:22,user@gpu3:22" \
        --prompt "Hello world" --generate

Cluster config (cluster.json):
{
    "coordinator": {
        "device": "cuda"
    },
    "workers": [
        {"host": "user@gpu1.example.com", "port": 22, "device": "cuda"},
        {"host": "user@gpu2.example.com", "port": 22, "device": "cuda"},
        {"host": "user@gpu3.example.com", "port": 22, "device": "cuda"}
    ]
}
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import shutil
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).parent.parent
REMOTE_WORKDIR = "/tmp/column_transformer"


@dataclass
class WorkerNode:
    ssh_target: str   # user@host
    ssh_port: int     # SSH port
    column_idx: int   # which column this worker runs
    device: str       # cuda, cpu, etc.
    pid: int | None = None  # remote PID once started


def run_ssh(ssh_target: str, ssh_port: int, cmd: str, check: bool = True,
            capture: bool = False, timeout: int = 120) -> subprocess.CompletedProcess:
    """Run a command on a remote machine via SSH."""
    ssh_cmd = [
        "ssh", "-o", "StrictHostKeyChecking=no",
        "-o", "ConnectTimeout=10",
        "-p", str(ssh_port),
        ssh_target,
        cmd,
    ]
    return subprocess.run(
        ssh_cmd, check=check, capture_output=capture,
        text=True, timeout=timeout,
    )


def run_scp(ssh_target: str, ssh_port: int, local_path: str, remote_path: str,
            recursive: bool = False) -> None:
    """Copy files to remote machine via SCP."""
    cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-P", str(ssh_port)]
    if recursive:
        cmd.append("-r")
    cmd.extend([local_path, f"{ssh_target}:{remote_path}"])
    subprocess.run(cmd, check=True, timeout=300)


def parse_workers_string(workers_str: str) -> list[dict]:
    """Parse 'user@host1:port,user@host2:port' into worker configs."""
    workers = []
    for entry in workers_str.split(","):
        entry = entry.strip()
        if ":" in entry:
            host_part, port = entry.rsplit(":", 1)
            port = int(port)
        else:
            host_part = entry
            port = 22
        workers.append({"host": host_part, "port": port, "device": "cuda"})
    return workers


def split_checkpoint(args, n_columns: int, tmpdir: str) -> str:
    """Split checkpoint into shards."""
    shard_dir = os.path.join(tmpdir, "shards")
    print(f"\n[1/5] Splitting checkpoint into {n_columns} shards...")
    result = subprocess.run(
        [sys.executable, "-m", "distributed.checkpoint_split",
         "--checkpoint", args.checkpoint,
         "--model", args.model,
         "--output-dir", shard_dir],
        cwd=str(PROJECT_ROOT),
        check=True, capture_output=True, text=True,
    )
    print(result.stdout)
    return shard_dir


def prepare_code_bundle(tmpdir: str) -> str:
    """Create a minimal code bundle for remote machines."""
    bundle_dir = os.path.join(tmpdir, "bundle")
    os.makedirs(bundle_dir)

    # Copy only what workers need
    dirs_to_copy = ["column_transformer", "distributed"]
    for d in dirs_to_copy:
        src = PROJECT_ROOT / d
        dst = os.path.join(bundle_dir, d)
        shutil.copytree(str(src), dst)

    # Create a minimal setup script
    setup_script = """#!/bin/bash
set -e
cd /tmp/column_transformer
pip install torch --quiet 2>/dev/null || true
echo "Setup complete"
"""
    with open(os.path.join(bundle_dir, "setup.sh"), "w") as f:
        f.write(setup_script)

    return bundle_dir


def deploy_to_worker(
    node: WorkerNode, bundle_dir: str, shard_dir: str
) -> None:
    """Upload code + shard to a single worker node."""
    print(f"  Deploying to {node.ssh_target} (column {node.column_idx})...")

    # Create remote directory
    run_ssh(node.ssh_target, node.ssh_port,
            f"mkdir -p {REMOTE_WORKDIR}/shards")

    # Upload code bundle
    run_scp(node.ssh_target, node.ssh_port,
            bundle_dir + "/.", REMOTE_WORKDIR, recursive=True)

    # Upload worker shard
    shard_file = os.path.join(shard_dir, f"worker_{node.column_idx}.pt")
    run_scp(node.ssh_target, node.ssh_port,
            shard_file, f"{REMOTE_WORKDIR}/shards/")

    # Run setup
    run_ssh(node.ssh_target, node.ssh_port,
            f"bash {REMOTE_WORKDIR}/setup.sh", check=False)

    print(f"  ✓ {node.ssh_target} ready")


def start_remote_worker(
    node: WorkerNode, coordinator_ip: str, coordinator_port: int, model_name: str
) -> None:
    """Start worker process on remote machine (in background)."""
    cmd = (
        f"cd {REMOTE_WORKDIR} && "
        f"nohup python -m distributed.run_worker "
        f"--model {model_name} "
        f"--shard shards/worker_{node.column_idx}.pt "
        f"--column {node.column_idx} "
        f"--coordinator {coordinator_ip}:{coordinator_port} "
        f"--device {node.device} "
        f"> /tmp/worker_{node.column_idx}.log 2>&1 & "
        f"echo $!"
    )
    result = run_ssh(node.ssh_target, node.ssh_port, cmd, capture=True)
    pid = result.stdout.strip()
    node.pid = int(pid) if pid.isdigit() else None
    print(f"  Started worker {node.column_idx} on {node.ssh_target} (PID: {pid})")


def stop_remote_worker(node: WorkerNode) -> None:
    """Stop worker process on remote machine."""
    if node.pid:
        run_ssh(node.ssh_target, node.ssh_port,
                f"kill {node.pid} 2>/dev/null || true", check=False)
    # Also kill by name as fallback
    run_ssh(node.ssh_target, node.ssh_port,
            "pkill -f 'distributed.run_worker' 2>/dev/null || true", check=False)


def get_local_ip() -> str:
    """Get this machine's IP address that remote machines can reach."""
    import socket
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "127.0.0.1"
    finally:
        s.close()


def main():
    parser = argparse.ArgumentParser(
        description="Deploy and run distributed column-parallel inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with 3 Vast.ai GPUs:
  python -m distributed.deploy \\
      --model h100_col8_drop25 \\
      --checkpoint model.pt \\
      --workers "root@gpu1.vast.ai:12345,root@gpu2.vast.ai:12346,root@gpu3.vast.ai:12347" \\
      --generate --prompt "The meaning of life"

  # Using a cluster config file:
  python -m distributed.deploy \\
      --model h100_col8_comp64 \\
      --checkpoint model.pt \\
      --cluster cluster.json \\
      --generate --max-tokens 200
        """,
    )

    # Model & checkpoint
    parser.add_argument("--model", required=True, help="Model config name from EXPERIMENTS")
    parser.add_argument("--checkpoint", required=True, help="Path to full model checkpoint")

    # Cluster definition (one of these is required)
    cluster_group = parser.add_mutually_exclusive_group(required=True)
    cluster_group.add_argument("--workers", type=str,
                               help="Comma-separated: user@host:port,user@host2:port,...")
    cluster_group.add_argument("--cluster", type=str,
                               help="Path to cluster config JSON")

    # Coordinator settings
    parser.add_argument("--port", type=int, default=9000,
                        help="Coordinator port (default: 9000)")
    parser.add_argument("--coordinator-device", default="auto",
                        help="Device for coordinator (default: auto)")
    parser.add_argument("--coordinator-ip", default=None,
                        help="Coordinator IP (auto-detected if not set)")

    # Inference settings
    parser.add_argument("--prompt", type=str, default=None)
    parser.add_argument("--generate", action="store_true")
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--n-forward", type=int, default=5)

    # Deployment options
    parser.add_argument("--skip-deploy", action="store_true",
                        help="Skip uploading code/shards (already deployed)")
    parser.add_argument("--cleanup", action="store_true",
                        help="Clean up remote workdir after running")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without doing it")

    args = parser.parse_args()

    # Load cluster config
    if args.cluster:
        with open(args.cluster) as f:
            cluster = json.load(f)
        worker_configs = cluster["workers"]
    else:
        worker_configs = parse_workers_string(args.workers)

    # Build worker nodes (assign column indices 1..N)
    nodes = []
    for i, wc in enumerate(worker_configs):
        nodes.append(WorkerNode(
            ssh_target=wc["host"],
            ssh_port=wc.get("port", 22),
            column_idx=i + 1,  # coordinator is column 0
            device=wc.get("device", "cuda"),
        ))

    coordinator_ip = args.coordinator_ip or get_local_ip()

    print(f"═══════════════════════════════════════════════════")
    print(f"  Column-Parallel Distributed Inference Deploy")
    print(f"═══════════════════════════════════════════════════")
    print(f"  Model:       {args.model}")
    print(f"  Checkpoint:  {args.checkpoint}")
    print(f"  Coordinator: {coordinator_ip}:{args.port} (column 0)")
    print(f"  Workers:     {len(nodes)}")
    for n in nodes:
        print(f"    Column {n.column_idx}: {n.ssh_target}:{n.ssh_port} ({n.device})")
    print(f"  Total columns: {len(nodes) + 1}")
    print(f"═══════════════════════════════════════════════════")

    if args.dry_run:
        print("\n[DRY RUN] Would deploy and start inference. Exiting.")
        return

    with tempfile.TemporaryDirectory() as tmpdir:
        if not args.skip_deploy:
            # Step 1: Split checkpoint
            from column_transformer.config import EXPERIMENTS, ColumnConfigV2
            config = EXPERIMENTS[args.model]
            shard_dir = split_checkpoint(args, config.n_columns, tmpdir)

            # Step 2: Prepare code bundle
            print(f"\n[2/5] Preparing code bundle...")
            bundle_dir = prepare_code_bundle(tmpdir)
            print(f"  Bundle ready: {bundle_dir}")

            # Step 3: Deploy to workers
            print(f"\n[3/5] Deploying to {len(nodes)} workers...")
            for node in nodes:
                deploy_to_worker(node, bundle_dir, shard_dir)
        else:
            print("\n[1-3/5] Skipping deploy (--skip-deploy)")

        # Step 4: Start remote workers
        print(f"\n[4/5] Starting remote workers...")
        for node in nodes:
            start_remote_worker(node, coordinator_ip, args.port, args.model)

        # Give workers a moment to start
        print("  Waiting 3s for workers to initialize...")
        time.sleep(3)

        # Step 5: Start coordinator (runs locally, blocks until done)
        print(f"\n[5/5] Starting coordinator...")
        coord_cmd = [
            sys.executable, "-m", "distributed.run_coordinator",
            "--model", args.model,
            "--checkpoint", args.checkpoint,
            "--port", str(args.port),
            "--device", args.coordinator_device,
            "--wait-for", str(len(nodes)),
            "--wait-timeout", "60",
        ]
        if args.prompt:
            coord_cmd.extend(["--prompt", args.prompt])
        if args.generate:
            coord_cmd.append("--generate")
            coord_cmd.extend(["--max-tokens", str(args.max_tokens)])
            coord_cmd.extend(["--temperature", str(args.temperature)])
        else:
            coord_cmd.extend(["--n-forward", str(args.n_forward)])

        try:
            subprocess.run(coord_cmd, cwd=str(PROJECT_ROOT), check=True)
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
        finally:
            # Stop remote workers
            print("\nStopping remote workers...")
            for node in nodes:
                stop_remote_worker(node)
                print(f"  Stopped worker {node.column_idx} on {node.ssh_target}")

            if args.cleanup:
                print("Cleaning up remote workdirs...")
                for node in nodes:
                    run_ssh(node.ssh_target, node.ssh_port,
                            f"rm -rf {REMOTE_WORKDIR}", check=False)

    print("\nDone!")


if __name__ == "__main__":
    main()
