"""Local multi-process test for distributed inference correctness.

Spawns coordinator + N-1 workers on localhost and verifies that
distributed output matches single-process model output.

Usage:
    python -m distributed.test_local --model v2_trunk3_xattn2

For quick testing without a trained checkpoint, uses random weights.
For testing with a real checkpoint:
    python -m distributed.test_local --model h100_col8_drop25 --checkpoint results/h100_col8_drop25_best.pt
"""

import argparse
import asyncio
import logging
import os
import sys
import tempfile
import time
import torch

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from column_transformer.config import EXPERIMENTS, ColumnConfigV2
from column_transformer.model_column_v2 import ColumnTransformerV2
from distributed.checkpoint_split import split_checkpoint
from distributed.coordinator import DistributedCoordinator
from distributed.worker import DistributedWorker
from distributed.protocol import MsgType, send_msg, recv_msg

logger = logging.getLogger(__name__)


def create_test_checkpoint(config: ColumnConfigV2, tmpdir: str) -> str:
    """Create a model with random weights and save checkpoint."""
    model = ColumnTransformerV2(config)
    path = os.path.join(tmpdir, "test_model.pt")
    torch.save(model.state_dict(), path)
    return path


def single_process_forward(
    checkpoint_path: str, config: ColumnConfigV2, input_ids: torch.Tensor,
    active_columns: list[int] | None = None,
) -> torch.Tensor:
    """Run the model in single-process mode for reference output."""
    model = ColumnTransformerV2(config)
    sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
    model.load_state_dict(sd)
    model.eval()
    with torch.no_grad():
        return model(input_ids, active_columns=active_columns)


async def run_distributed_forward(
    checkpoint_path: str,
    shard_dir: str,
    config: ColumnConfigV2,
    input_ids: torch.Tensor,
    worker_columns: list[int],
    port: int = 9123,
) -> torch.Tensor:
    """Run distributed forward pass with coordinator + workers on localhost."""

    coordinator = DistributedCoordinator(
        config=config,
        checkpoint_path=checkpoint_path,
        host="127.0.0.1",
        port=port,
        device="cpu",
        coordinator_column=0,
        merge_timeout=30.0,
    )
    await coordinator.start_server()

    # Start workers as async tasks
    worker_tasks = []
    for col_idx in worker_columns:
        shard_path = os.path.join(shard_dir, f"worker_{col_idx}.pt")
        worker = DistributedWorker(
            column_idx=col_idx,
            config=config,
            shard_path=shard_path,
            coordinator_host="127.0.0.1",
            coordinator_port=port,
            device="cpu",
        )
        task = asyncio.create_task(worker.connect_and_run(n_forward=1))
        worker_tasks.append(task)

    # Give workers time to connect
    await coordinator.wait_for_workers(min_workers=len(worker_columns), timeout=10.0)

    # Run forward pass
    logits = await coordinator.forward(input_ids)

    # Shutdown
    await coordinator.shutdown_workers()

    # Wait for workers to finish
    for task in worker_tasks:
        try:
            await asyncio.wait_for(task, timeout=5.0)
        except (asyncio.TimeoutError, Exception):
            task.cancel()

    return logits


async def test_full_columns(config, checkpoint_path, shard_dir, port):
    """Test: all columns active, distributed output should match single-process."""
    print("\n=== Test 1: Full columns (all active) ===")

    input_ids = torch.randint(0, config.vocab_size, (1, 32))
    worker_columns = list(range(1, config.n_columns))
    all_columns = list(range(config.n_columns))

    # Reference
    ref_logits = single_process_forward(checkpoint_path, config, input_ids,
                                         active_columns=all_columns)

    # Distributed
    dist_logits = await run_distributed_forward(
        checkpoint_path, shard_dir, config, input_ids, worker_columns, port
    )

    # Compare
    max_diff = (ref_logits - dist_logits).abs().max().item()
    mean_diff = (ref_logits - dist_logits).abs().mean().item()
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    if max_diff < 1e-3:
        print("  PASSED (within tolerance)")
        return True
    else:
        print("  FAILED (difference too large)")
        return False


async def test_partial_columns(config, checkpoint_path, shard_dir, port):
    """Test: subset of columns, should match single-process with same active_columns."""
    print("\n=== Test 2: Partial columns (fault tolerance) ===")

    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    # Use only half the columns
    n_use = max(2, config.n_columns // 2)
    active = list(range(n_use))
    worker_columns = active[1:]  # coordinator handles column 0

    # Reference: single-process with same active columns
    ref_logits = single_process_forward(checkpoint_path, config, input_ids,
                                         active_columns=active)

    # Distributed
    dist_logits = await run_distributed_forward(
        checkpoint_path, shard_dir, config, input_ids, worker_columns, port + 1
    )

    max_diff = (ref_logits - dist_logits).abs().max().item()
    mean_diff = (ref_logits - dist_logits).abs().mean().item()
    print(f"  Active columns: {active}")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    if max_diff < 1e-3:
        print("  PASSED (within tolerance)")
        return True
    else:
        print("  FAILED (difference too large)")
        return False


async def test_coordinator_only(config, checkpoint_path, port):
    """Test: coordinator with no workers (single column)."""
    print("\n=== Test 3: Coordinator only (no workers) ===")

    input_ids = torch.randint(0, config.vocab_size, (1, 32))

    # Reference: single column
    ref_logits = single_process_forward(checkpoint_path, config, input_ids,
                                         active_columns=[0])

    # Distributed: coordinator only, no workers
    coordinator = DistributedCoordinator(
        config=config,
        checkpoint_path=checkpoint_path,
        host="127.0.0.1",
        port=port + 2,
        device="cpu",
        coordinator_column=0,
    )
    await coordinator.start_server()
    dist_logits = await coordinator.forward(input_ids)
    await coordinator.shutdown_workers()

    max_diff = (ref_logits - dist_logits).abs().max().item()
    mean_diff = (ref_logits - dist_logits).abs().mean().item()
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    if max_diff < 1e-3:
        print("  PASSED (within tolerance)")
        return True
    else:
        print("  FAILED (difference too large)")
        return False


async def run_tests(args):
    config = EXPERIMENTS[args.model]
    assert isinstance(config, ColumnConfigV2)

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create or use checkpoint
        if args.checkpoint:
            checkpoint_path = args.checkpoint
        else:
            print(f"Creating test model ({args.model}) with random weights...")
            checkpoint_path = create_test_checkpoint(config, tmpdir)

        # Split checkpoint
        shard_dir = os.path.join(tmpdir, "shards")
        os.makedirs(shard_dir)
        print(f"Splitting checkpoint into shards...")
        sd = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        sd = {k.replace("_orig_mod.", ""): v for k, v in sd.items()}
        coord_shard, worker_shards = split_checkpoint(sd, config)

        # Save shards
        torch.save(coord_shard, os.path.join(shard_dir, "coordinator.pt"))
        worker_col_indices = [i for i in range(config.n_columns) if i != 0]
        for col_idx, shard in zip(worker_col_indices, worker_shards):
            torch.save(shard, os.path.join(shard_dir, f"worker_{col_idx}.pt"))
        print(f"  {len(worker_shards) + 1} shards created")

        # Run tests
        results = []
        port = args.port

        results.append(await test_full_columns(config, checkpoint_path, shard_dir, port))
        results.append(await test_partial_columns(config, checkpoint_path, shard_dir, port))
        results.append(await test_coordinator_only(config, checkpoint_path, port))

        # Summary
        print(f"\n{'='*50}")
        passed = sum(results)
        total = len(results)
        print(f"Results: {passed}/{total} tests passed")
        if passed == total:
            print("All tests passed!")
        else:
            print("Some tests FAILED")
            sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Local distributed inference tests")
    parser.add_argument("--model", default="v2_trunk3_xattn2",
                        help="Model config name (default: v2_trunk3_xattn2 for quick testing)")
    parser.add_argument("--checkpoint", default=None,
                        help="Checkpoint path (omit for random weights)")
    parser.add_argument("--port", type=int, default=9123,
                        help="Base port for test servers")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(name)s %(levelname)s: %(message)s",
    )

    print(f"Testing distributed inference with model: {args.model}")
    asyncio.run(run_tests(args))


if __name__ == "__main__":
    main()
