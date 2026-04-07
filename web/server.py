"""Web platform for distributed column-parallel transformer inference.

Provides:
- GPU worker management (SSH-based connect/disconnect/status)
- Model management (list/load/unload models)
- Inference API (generate text, benchmark)
- Metrics dashboard (compare models, track performance)
- Training launch (on high-end GPUs via SSH)
- Remote GPU monitoring (nvidia-smi)

Run:
    pip install fastapi uvicorn websockets aiofiles
    python -m web.server --port 8000
"""

import asyncio
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse
from pydantic import BaseModel

# Add project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from column_transformer.config import EXPERIMENTS, ColumnConfigV2, DenseConfig

logger = logging.getLogger(__name__)

app = FastAPI(title="Column Transformer Platform", version="0.2.0")

# ============================================================
# State
# ============================================================

@dataclass
class RemoteGPU:
    """A remote GPU machine connected via SSH."""
    id: str                     # unique identifier
    ssh_target: str             # user@host
    ssh_port: int               # SSH port
    gpu_name: str = ""          # e.g., "RTX 4080 SUPER"
    gpu_memory_mb: int = 0
    gpu_memory_free_mb: int = 0
    gpu_utilization: float = 0.0
    role: str = "idle"          # idle, coordinator, worker
    column_idx: int = -1        # assigned column (-1 = unassigned)
    status: str = "connected"   # connected, disconnected, error, busy
    worker_pid: int = 0
    last_seen: float = 0.0
    error_msg: str = ""

@dataclass
class ModelInfo:
    name: str
    config_name: str
    checkpoint_path: str
    params: int = 0
    status: str = "unloaded"
    device: str = "cpu"

@dataclass
class InferenceResult:
    model: str
    prompt: str
    output: str
    tokens_generated: int
    total_time_s: float
    tokens_per_sec: float
    active_columns: int
    total_columns: int
    timestamp: float = 0.0

@dataclass
class BenchmarkResult:
    model: str
    active_columns: int
    total_columns: int
    avg_latency_ms: float
    tokens_per_sec: float
    n_forward: int
    seq_len: int
    timestamp: float = 0.0

@dataclass
class TrainingJob:
    id: str
    model_config: str
    dataset: str
    steps: int
    status: str = "pending"     # pending, running, completed, failed
    gpu_id: str = ""
    ssh_target: str = ""
    ssh_port: int = 22
    remote_pid: int = 0
    current_step: int = 0
    current_loss: float = 0.0
    best_val_loss: float = float('inf')
    started_at: float = 0.0
    finished_at: float = 0.0
    error_msg: str = ""
    log_tail: str = ""

@dataclass
class ExperimentRecord:
    """Historical experiment result for comparison."""
    name: str
    model_config: str
    params: int
    perplexity: float
    val_loss: float
    dataset: str
    steps: int
    col_drop_prob: float = 0.0
    n_columns: int = 0
    comm_rank: int = 0
    timestamp: float = 0.0
    notes: str = ""


class AppState:
    def __init__(self):
        self.coordinator = None
        self.remote_gpus: dict[str, RemoteGPU] = {}
        self.models: dict[str, ModelInfo] = {}
        self.loaded_model: Optional[str] = None
        self.inference_history: list[InferenceResult] = []
        self.benchmark_history: list[BenchmarkResult] = []
        self.training_jobs: dict[str, TrainingJob] = {}
        self.experiments: list[ExperimentRecord] = []
        self.coordinator_port: int = 9000
        self.coordinator_device: str = "cpu"
        self.websocket_clients: list[WebSocket] = []

        # Load saved experiment data
        self._load_experiments()

    def _load_experiments(self):
        """Load known experiment results for comparison dashboard."""
        self.experiments = [
            ExperimentRecord("Dense 60M", "dense", 60_000_000, 30.31, 3.411, "WikiText-103", 10000),
            ExperimentRecord("V2 trunk3 60M", "v2_trunk3_xattn2", 60_000_000, 31.12, 3.437, "WikiText-103", 10000,
                             n_columns=4),
            ExperimentRecord("V2 trunk3 merge1 60M", "v2_trunk3_xattn1", 60_000_000, 30.94, 3.432, "WikiText-103", 10000,
                             n_columns=4),
            ExperimentRecord("H100 Dense 350M", "h100_dense", 350_000_000, 18.5, 2.918, "WikiText-103", 10000),
            ExperimentRecord("H100 Col8 drop0 350M", "h100_col8_drop0", 350_000_000, 18.6, 2.923, "WikiText-103", 10000,
                             n_columns=8, col_drop_prob=0.0),
            ExperimentRecord("H100 Col8 drop25 350M", "h100_col8_drop25", 350_000_000, 18.43, 2.914, "WikiText-103", 10000,
                             n_columns=8, col_drop_prob=0.25,
                             notes="Best 350M result — dropout regularizes"),
            ExperimentRecord("H100 Col8 drop25 350M (retrained)", "h100_col8_drop25", 231_911_424, 21.83, 3.083, "WikiText-103", 20000,
                             n_columns=8, col_drop_prob=0.25,
                             notes="Retrained 20K steps on A100"),
            ExperimentRecord("1B drop25", "h100_1b_col8_drop25", 1_000_000_000, 20.21, 3.006, "FineWeb-Edu", 20000,
                             n_columns=8, col_drop_prob=0.25, comm_rank=0),
            ExperimentRecord("1B comp64", "h100_1b_col8_comp64", 1_000_000_000, 20.21, 3.006, "FineWeb-Edu", 20000,
                             n_columns=8, col_drop_prob=0.25, comm_rank=64,
                             notes="8x compression, identical to uncompressed"),
        ]

    async def broadcast(self, event: str, data: dict):
        """Broadcast event to all connected websocket clients."""
        msg = json.dumps({"event": event, "data": data})
        disconnected = []
        for ws in self.websocket_clients:
            try:
                await ws.send_text(msg)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            self.websocket_clients.remove(ws)


state = AppState()

# ============================================================
# API Models
# ============================================================

class AddGPURequest(BaseModel):
    ssh_target: str         # user@host
    ssh_port: int = 22

class AssignRoleRequest(BaseModel):
    gpu_id: str
    role: str               # coordinator, worker, idle
    column_idx: int = -1

class LoadModelRequest(BaseModel):
    config_name: str
    checkpoint_path: str
    device: str = "auto"
    coordinator_port: int = 9000

class GenerateRequest(BaseModel):
    prompt: str
    max_tokens: int = 100
    temperature: float = 0.8

class BenchmarkRequest(BaseModel):
    seq_len: int = 128
    n_forward: int = 10

class LaunchTrainingRequest(BaseModel):
    gpu_id: str
    config_name: str
    dataset: str = "wikitext"
    steps: int = 10000
    batch_size: int = 8
    learning_rate: float = 3e-4

class StartDistributedRequest(BaseModel):
    coordinator_gpu_id: str
    worker_gpu_ids: list[str]
    config_name: str
    checkpoint_path: str
    coordinator_port: int = 8384

# ============================================================
# SSH Helpers
# ============================================================

def _filter_ssh_banner(text: str) -> str:
    """Remove SSH welcome banners from output."""
    lines = text.split("\n")
    filtered = []
    for line in lines:
        stripped = line.strip()
        if any(stripped.startswith(b) for b in [
            "Welcome to", "Have fun", "If authentication fails",
            "double check your ssh key",
        ]):
            continue
        if stripped:
            filtered.append(line)
    return "\n".join(filtered).strip()


async def ssh_run(ssh_target: str, ssh_port: int, cmd: str,
                  timeout: int = 30) -> tuple[bool, str]:
    """Run command on remote machine via SSH. Returns (success, output)."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", "ConnectTimeout=10",
            "-p", str(ssh_port),
            ssh_target, cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await asyncio.wait_for(
            proc.communicate(), timeout=timeout
        )
        output = _filter_ssh_banner(stdout.decode() + stderr.decode())
        return proc.returncode == 0, output
    except asyncio.TimeoutError:
        return False, "SSH command timed out"
    except Exception as e:
        return False, str(e)


async def get_gpu_info(ssh_target: str, ssh_port: int) -> dict:
    """Get GPU info from remote machine."""
    ok, output = await ssh_run(
        ssh_target, ssh_port,
        "nvidia-smi --query-gpu=name,memory.total,memory.free,utilization.gpu "
        "--format=csv,noheader,nounits 2>/dev/null || echo 'NO_GPU'"
    )
    if not ok or "NO_GPU" in output:
        return {"gpu_name": "No GPU", "gpu_memory_mb": 0,
                "gpu_memory_free_mb": 0, "gpu_utilization": 0.0}

    # Filter out SSH banners and other noise — find the CSV line
    csv_line = None
    for line in output.split("\n"):
        line = line.strip()
        # CSV line from nvidia-smi has commas and starts with GPU name
        if "," in line and not line.startswith("Welcome") and not line.startswith("Have fun"):
            csv_line = line
            break

    if csv_line:
        parts = csv_line.split(",")
        if len(parts) >= 4:
            try:
                return {
                    "gpu_name": parts[0].strip(),
                    "gpu_memory_mb": int(parts[1].strip()),
                    "gpu_memory_free_mb": int(parts[2].strip()),
                    "gpu_utilization": float(parts[3].strip()),
                }
            except (ValueError, IndexError):
                pass

    return {"gpu_name": output.split("\n")[0].strip()[:50], "gpu_memory_mb": 0,
            "gpu_memory_free_mb": 0, "gpu_utilization": 0.0}


# ============================================================
# Routes: System
# ============================================================

@app.get("/")
async def root():
    return FileResponse(Path(__file__).parent / "static" / "index.html")

@app.get("/api/status")
async def get_status():
    active_gpus = [g for g in state.remote_gpus.values() if g.status == "connected"]
    coordinators = [g for g in active_gpus if g.role == "coordinator"]
    workers = [g for g in active_gpus if g.role == "worker"]
    return {
        "loaded_model": state.loaded_model,
        "n_gpus": len(active_gpus),
        "n_coordinators": len(coordinators),
        "n_workers": len(workers),
        "coordinator_active": state.coordinator is not None,
        "active_training_jobs": len([j for j in state.training_jobs.values()
                                      if j.status == "running"]),
    }

# ============================================================
# Routes: GPU Management
# ============================================================

@app.post("/api/gpus/add")
async def add_gpu(req: AddGPURequest):
    """Add a remote GPU via SSH."""
    gpu_id = f"{req.ssh_target}:{req.ssh_port}"

    # Test SSH connection
    ok, output = await ssh_run(req.ssh_target, req.ssh_port, "echo OK", timeout=15)
    if not ok:
        raise HTTPException(400, f"Cannot connect via SSH: {output}")

    # Get GPU info
    gpu_info = await get_gpu_info(req.ssh_target, req.ssh_port)

    gpu = RemoteGPU(
        id=gpu_id,
        ssh_target=req.ssh_target,
        ssh_port=req.ssh_port,
        gpu_name=gpu_info["gpu_name"],
        gpu_memory_mb=gpu_info["gpu_memory_mb"],
        gpu_memory_free_mb=gpu_info["gpu_memory_free_mb"],
        gpu_utilization=gpu_info["gpu_utilization"],
        status="connected",
        last_seen=time.time(),
    )
    state.remote_gpus[gpu_id] = gpu

    await state.broadcast("gpu_added", {
        "id": gpu_id, "gpu_name": gpu.gpu_name,
        "memory_mb": gpu.gpu_memory_mb,
    })
    return asdict(gpu)


@app.delete("/api/gpus/{gpu_id:path}")
async def remove_gpu(gpu_id: str):
    """Remove a GPU from the cluster."""
    if gpu_id not in state.remote_gpus:
        raise HTTPException(404, "GPU not found")
    del state.remote_gpus[gpu_id]
    await state.broadcast("gpu_removed", {"id": gpu_id})
    return {"status": "ok"}


@app.get("/api/gpus")
async def list_gpus():
    """List all registered GPUs."""
    return [asdict(g) for g in state.remote_gpus.values()]


@app.post("/api/gpus/refresh")
async def refresh_all_gpus():
    """Refresh GPU info for all registered machines."""
    tasks = []
    for gpu in state.remote_gpus.values():
        tasks.append(_refresh_gpu(gpu))
    await asyncio.gather(*tasks, return_exceptions=True)
    return [asdict(g) for g in state.remote_gpus.values()]


async def _refresh_gpu(gpu: RemoteGPU):
    """Refresh GPU status for a single machine."""
    try:
        info = await get_gpu_info(gpu.ssh_target, gpu.ssh_port)
        gpu.gpu_name = info["gpu_name"]
        gpu.gpu_memory_mb = info["gpu_memory_mb"]
        gpu.gpu_memory_free_mb = info["gpu_memory_free_mb"]
        gpu.gpu_utilization = info["gpu_utilization"]
        gpu.status = "connected"
        gpu.last_seen = time.time()
    except Exception as e:
        gpu.status = "error"
        gpu.error_msg = str(e)


@app.get("/api/gpus/{gpu_id:path}/nvidia-smi")
async def gpu_nvidia_smi(gpu_id: str):
    """Get full nvidia-smi output from a GPU."""
    gpu = state.remote_gpus.get(gpu_id)
    if not gpu:
        raise HTTPException(404, "GPU not found")
    ok, output = await ssh_run(gpu.ssh_target, gpu.ssh_port, "nvidia-smi")
    return {"output": output, "ok": ok}


@app.post("/api/gpus/{gpu_id:path}/assign")
async def assign_gpu_role(gpu_id: str, req: AssignRoleRequest):
    """Assign a role to a GPU (coordinator, worker, idle)."""
    gpu = state.remote_gpus.get(gpu_id)
    if not gpu:
        raise HTTPException(404, "GPU not found")
    gpu.role = req.role
    gpu.column_idx = req.column_idx
    await state.broadcast("gpu_role_changed", {"id": gpu_id, "role": req.role,
                                                "column_idx": req.column_idx})
    return asdict(gpu)


# ============================================================
# Routes: Models
# ============================================================

@app.get("/api/models/available")
async def list_available_models():
    """List all model configs from EXPERIMENTS."""
    models = []
    for name, config in EXPERIMENTS.items():
        info = {"name": name, "type": type(config).__name__}
        if isinstance(config, ColumnConfigV2):
            info.update({
                "n_columns": config.n_columns,
                "d_model": config.d_model,
                "d_col": config.d_col,
                "n_trunk_layers": config.n_trunk_layers,
                "n_col_layers": config.n_col_layers,
                "merge_every": config.merge_every,
                "col_drop_prob": config.col_drop_prob,
                "comm_rank": config.comm_rank,
                "quant_comm": config.quant_comm,
                "max_seq_len": config.max_seq_len,
                "params_estimate": config.param_estimate(),
            })
        elif isinstance(config, DenseConfig):
            info.update({
                "d_model": config.d_model,
                "n_layers": config.n_layers,
                "max_seq_len": config.max_seq_len,
                "params_estimate": config.param_estimate(),
            })
        models.append(info)
    return models

@app.get("/api/models/loaded")
async def get_loaded_model():
    if state.loaded_model is None:
        return {"model": None}
    return {
        "model": state.loaded_model,
        "info": asdict(state.models[state.loaded_model]),
    }

@app.post("/api/models/load")
async def load_model(req: LoadModelRequest):
    """Load a model checkpoint and start coordinator."""
    import torch
    from distributed.coordinator import DistributedCoordinator

    config = EXPERIMENTS.get(req.config_name)
    if config is None:
        raise HTTPException(404, f"Unknown model config: {req.config_name}")
    if not isinstance(config, ColumnConfigV2):
        raise HTTPException(400, "Only ColumnConfigV2 models supported")
    if not os.path.exists(req.checkpoint_path):
        raise HTTPException(404, f"Checkpoint not found: {req.checkpoint_path}")

    # Unload existing
    if state.coordinator is not None:
        await state.coordinator.shutdown_workers()
        state.coordinator = None
        state.loaded_model = None

    device = req.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    try:
        coordinator = DistributedCoordinator(
            config=config,
            checkpoint_path=req.checkpoint_path,
            host="0.0.0.0",
            port=req.coordinator_port,
            device=device,
        )
        await coordinator.start_server()
        state.coordinator = coordinator
        state.coordinator_port = req.coordinator_port
        state.coordinator_device = device

        model_info = ModelInfo(
            name=req.config_name,
            config_name=req.config_name,
            checkpoint_path=req.checkpoint_path,
            params=sum(p.numel() for p in coordinator.model.parameters()),
            status="ready",
            device=device,
        )
        state.models[req.config_name] = model_info
        state.loaded_model = req.config_name

        await state.broadcast("model_loaded", {
            "model": req.config_name, "params": model_info.params,
        })
        return {"status": "ok", "model": req.config_name, "params": model_info.params}
    except Exception as e:
        logger.exception("Failed to load model")
        raise HTTPException(500, str(e))

@app.post("/api/models/unload")
async def unload_model():
    if state.coordinator:
        await state.coordinator.shutdown_workers()
        state.coordinator = None
    state.loaded_model = None
    await state.broadcast("model_unloaded", {})
    return {"status": "ok"}

# ============================================================
# Routes: Distributed Inference
# ============================================================

@app.post("/api/distributed/start")
async def start_distributed(req: StartDistributedRequest):
    """Start distributed inference across GPUs.

    1. Start coordinator on the designated GPU
    2. Start workers on worker GPUs
    """
    coord_gpu = state.remote_gpus.get(req.coordinator_gpu_id)
    if not coord_gpu:
        raise HTTPException(404, f"Coordinator GPU not found: {req.coordinator_gpu_id}")

    worker_gpus = []
    for wid in req.worker_gpu_ids:
        wgpu = state.remote_gpus.get(wid)
        if not wgpu:
            raise HTTPException(404, f"Worker GPU not found: {wid}")
        worker_gpus.append(wgpu)

    config = EXPERIMENTS.get(req.config_name)
    if not config:
        raise HTTPException(404, f"Model config not found: {req.config_name}")

    # Start coordinator
    coord_cmd = (
        f"cd /tmp/column_transformer && "
        f"export HF_HOME=/tmp/hf_home && "
        f"nohup /venv/main/bin/python -m distributed.run_coordinator "
        f"--model {req.config_name} "
        f"--checkpoint {req.checkpoint_path} "
        f"--port {req.coordinator_port} "
        f"--device cuda "
        f"--wait-for {len(worker_gpus)} "
        f"--wait-timeout 180 "
        f"> /tmp/coordinator.log 2>&1 & echo $!"
    )
    ok, output = await ssh_run(coord_gpu.ssh_target, coord_gpu.ssh_port, coord_cmd)
    if ok and output.strip().isdigit():
        coord_gpu.role = "coordinator"
        coord_gpu.column_idx = 0
        coord_gpu.worker_pid = int(output.strip())
        coord_gpu.status = "busy"

    # Determine external coordinator address
    # For Vast.ai, we need the public IP + mapped port
    coord_host = coord_gpu.ssh_target.split("@")[-1] if "@" in coord_gpu.ssh_target else coord_gpu.ssh_target

    # Get the Vast.ai port mapping
    ok2, port_output = await ssh_run(
        coord_gpu.ssh_target, coord_gpu.ssh_port,
        f"echo $VAST_TCP_PORT_{req.coordinator_port}"
    )
    external_port = int(port_output.strip()) if ok2 and port_output.strip().isdigit() else req.coordinator_port

    # Wait for coordinator to start
    await asyncio.sleep(5)

    # Start workers
    for i, wgpu in enumerate(worker_gpus):
        col_idx = i + 1
        worker_cmd = (
            f"cd /tmp/column_transformer && "
            f"nohup /venv/main/bin/python -m distributed.run_worker "
            f"--model {req.config_name} "
            f"--shard worker_{col_idx}.pt "
            f"--column {col_idx} "
            f"--coordinator {coord_host}:{external_port} "
            f"--device cuda "
            f"> /tmp/worker_{col_idx}.log 2>&1 & echo $!"
        )
        ok, output = await ssh_run(wgpu.ssh_target, wgpu.ssh_port, worker_cmd)
        if ok and output.strip().isdigit():
            wgpu.role = "worker"
            wgpu.column_idx = col_idx
            wgpu.worker_pid = int(output.strip())
            wgpu.status = "busy"

    await state.broadcast("distributed_started", {
        "coordinator": req.coordinator_gpu_id,
        "workers": req.worker_gpu_ids,
        "model": req.config_name,
    })
    return {
        "status": "ok",
        "coordinator": req.coordinator_gpu_id,
        "workers": req.worker_gpu_ids,
        "external_port": external_port,
    }


@app.post("/api/distributed/stop")
async def stop_distributed():
    """Stop all distributed inference processes."""
    for gpu in state.remote_gpus.values():
        if gpu.role in ("coordinator", "worker") and gpu.worker_pid > 0:
            await ssh_run(gpu.ssh_target, gpu.ssh_port,
                          f"kill {gpu.worker_pid} 2>/dev/null; "
                          f"pkill -f 'distributed.run_' 2>/dev/null || true")
            gpu.role = "idle"
            gpu.column_idx = -1
            gpu.worker_pid = 0
            gpu.status = "connected"

    await state.broadcast("distributed_stopped", {})
    return {"status": "ok"}


# ============================================================
# Routes: Workers (for local coordinator)
# ============================================================

@app.get("/api/workers")
async def list_workers():
    if state.coordinator is None:
        return {"workers": [], "active_columns": []}
    return {
        "workers": [asdict(w) for w in getattr(state.coordinator, 'workers', {}).values()
                     if hasattr(w, '__dict__')],
        "active_columns": state.coordinator.active_columns,
        "n_active": state.coordinator.n_active,
    }

@app.get("/api/workers/wait")
async def wait_for_workers(min_workers: int = 1, timeout: float = 60):
    if state.coordinator is None:
        raise HTTPException(400, "No model loaded")
    await state.coordinator.wait_for_workers(min_workers, timeout)
    return {
        "n_workers": len(state.coordinator.workers),
        "active_columns": state.coordinator.active_columns,
    }

# ============================================================
# Routes: Inference
# ============================================================

@app.post("/api/inference/generate")
async def generate_text(req: GenerateRequest):
    """Generate text using the loaded model."""
    import torch

    if state.coordinator is None:
        raise HTTPException(400, "No model loaded")

    config = EXPERIMENTS[state.loaded_model]
    device = state.coordinator_device

    try:
        from transformers import GPT2Tokenizer
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    except Exception:
        raise HTTPException(500, "Could not load GPT2 tokenizer")

    input_ids = tokenizer.encode(req.prompt, return_tensors="pt").to(device)
    generated = input_ids.clone()
    t0 = time.time()

    for step in range(req.max_tokens):
        context = generated[:, -config.max_seq_len:]
        logits = await state.coordinator.forward(context)
        next_logits = logits[:, -1, :]

        if req.temperature > 0:
            next_logits = next_logits / req.temperature
            probs = torch.softmax(next_logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
        else:
            next_token = next_logits.argmax(dim=-1, keepdim=True)

        generated = torch.cat([generated, next_token], dim=-1)

    dt = time.time() - t0
    output_text = tokenizer.decode(generated[0, input_ids.shape[1]:])

    result = InferenceResult(
        model=state.loaded_model,
        prompt=req.prompt,
        output=output_text,
        tokens_generated=req.max_tokens,
        total_time_s=dt,
        tokens_per_sec=req.max_tokens / dt,
        active_columns=state.coordinator.n_active,
        total_columns=config.n_columns,
        timestamp=time.time(),
    )
    state.inference_history.append(result)
    await state.broadcast("inference_complete", asdict(result))
    return asdict(result)

# ============================================================
# Routes: Benchmarks
# ============================================================

@app.post("/api/benchmark/run")
async def run_benchmark(req: BenchmarkRequest):
    import torch

    if state.coordinator is None:
        raise HTTPException(400, "No model loaded")

    config = EXPERIMENTS[state.loaded_model]
    device = state.coordinator_device

    input_ids = torch.randint(0, config.vocab_size, (1, req.seq_len), device=device)

    times = []
    for i in range(req.n_forward):
        t0 = time.time()
        await state.coordinator.forward(input_ids)
        dt = time.time() - t0
        times.append(dt)

    if len(times) > 1:
        avg_time = sum(times[1:]) / len(times[1:])
    else:
        avg_time = times[0]

    result = BenchmarkResult(
        model=state.loaded_model,
        active_columns=state.coordinator.n_active,
        total_columns=config.n_columns,
        avg_latency_ms=avg_time * 1000,
        tokens_per_sec=req.seq_len / avg_time,
        n_forward=req.n_forward,
        seq_len=req.seq_len,
        timestamp=time.time(),
    )
    state.benchmark_history.append(result)
    await state.broadcast("benchmark_complete", asdict(result))
    return asdict(result)

# ============================================================
# Routes: Training
# ============================================================

@app.post("/api/training/launch")
async def launch_training(req: LaunchTrainingRequest):
    """Launch training on a remote GPU via SSH."""
    gpu = state.remote_gpus.get(req.gpu_id)
    if not gpu:
        raise HTTPException(404, "GPU not found")

    job_id = f"train_{int(time.time())}_{req.config_name}"

    # Build training command
    train_cmd = (
        f"cd /tmp/column_transformer && "
        f"export HF_HOME=/tmp/hf_home && "
        f"nohup /venv/main/bin/python run_experiment.py "
        f"--model {req.config_name} "
        f"--dataset {req.dataset} "
        f"--steps {req.steps} "
        f"--batch-size {req.batch_size} "
        f"--lr {req.learning_rate} "
        f"--device cuda "
        f"--compile "
        f"> /tmp/training_{job_id}.log 2>&1 & echo $!"
    )

    ok, output = await ssh_run(gpu.ssh_target, gpu.ssh_port, train_cmd, timeout=30)
    if not ok:
        raise HTTPException(500, f"Failed to launch training: {output}")

    pid = int(output.strip()) if output.strip().isdigit() else 0

    job = TrainingJob(
        id=job_id,
        model_config=req.config_name,
        dataset=req.dataset,
        steps=req.steps,
        status="running",
        gpu_id=req.gpu_id,
        ssh_target=gpu.ssh_target,
        ssh_port=gpu.ssh_port,
        remote_pid=pid,
        started_at=time.time(),
    )
    state.training_jobs[job_id] = job
    gpu.role = "training"
    gpu.status = "busy"

    await state.broadcast("training_started", {"job_id": job_id, "model": req.config_name})
    return asdict(job)


@app.get("/api/training/jobs")
async def list_training_jobs():
    return [asdict(j) for j in state.training_jobs.values()]


@app.get("/api/training/{job_id}/status")
async def get_training_status(job_id: str):
    """Get training job status with latest log output."""
    job = state.training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")

    # Check if process is still running
    ok, output = await ssh_run(
        job.ssh_target, job.ssh_port,
        f"kill -0 {job.remote_pid} 2>/dev/null && echo RUNNING || echo STOPPED",
        timeout=10,
    )
    if "STOPPED" in output:
        job.status = "completed"
        job.finished_at = time.time()

    # Get recent log output
    ok, log_output = await ssh_run(
        job.ssh_target, job.ssh_port,
        f"tail -20 /tmp/training_{job_id}.log 2>/dev/null",
        timeout=10,
    )
    if ok:
        job.log_tail = log_output

        # Parse step and loss from log
        for line in log_output.split("\n"):
            if "step" in line.lower() and "loss" in line.lower():
                import re
                step_match = re.search(r'step\s+(\d+)', line, re.I)
                loss_match = re.search(r'loss\s+([\d.]+)', line, re.I)
                if step_match:
                    job.current_step = int(step_match.group(1))
                if loss_match:
                    job.current_loss = float(loss_match.group(1))
            if "val_loss" in line.lower():
                import re
                val_match = re.search(r'val_loss\s+([\d.]+)', line, re.I)
                if val_match:
                    val_loss = float(val_match.group(1))
                    job.best_val_loss = min(job.best_val_loss, val_loss)

    return asdict(job)


@app.post("/api/training/{job_id}/stop")
async def stop_training(job_id: str):
    """Stop a training job."""
    job = state.training_jobs.get(job_id)
    if not job:
        raise HTTPException(404, "Training job not found")

    await ssh_run(job.ssh_target, job.ssh_port,
                  f"kill {job.remote_pid} 2>/dev/null || true")
    job.status = "stopped"
    job.finished_at = time.time()

    # Reset GPU status
    gpu = state.remote_gpus.get(job.gpu_id)
    if gpu:
        gpu.role = "idle"
        gpu.status = "connected"

    await state.broadcast("training_stopped", {"job_id": job_id})
    return asdict(job)


# ============================================================
# Routes: Metrics & Experiments
# ============================================================

@app.get("/api/metrics/history")
async def get_metrics_history():
    return {
        "inference": [asdict(r) for r in state.inference_history],
        "benchmarks": [asdict(r) for r in state.benchmark_history],
    }

@app.get("/api/metrics/comparison")
async def get_model_comparison():
    """Compare performance across benchmark runs."""
    comparisons = {}
    for b in state.benchmark_history:
        key = f"{b.model}_{b.active_columns}col"
        if key not in comparisons:
            comparisons[key] = {
                "model": b.model, "active_columns": b.active_columns,
                "total_columns": b.total_columns, "runs": [],
            }
        comparisons[key]["runs"].append({
            "avg_latency_ms": b.avg_latency_ms,
            "tokens_per_sec": b.tokens_per_sec,
            "seq_len": b.seq_len,
            "timestamp": b.timestamp,
        })

    for key, data in comparisons.items():
        runs = data["runs"]
        data["avg_latency_ms"] = sum(r["avg_latency_ms"] for r in runs) / len(runs)
        data["avg_tokens_per_sec"] = sum(r["tokens_per_sec"] for r in runs) / len(runs)
        data["n_runs"] = len(runs)

    return list(comparisons.values())

@app.get("/api/experiments")
async def get_experiments():
    """Get all known experiment results for the comparison dashboard."""
    return [asdict(e) for e in state.experiments]

@app.get("/api/experiments/degradation")
async def get_degradation_data():
    """Get graceful degradation data (perplexity vs active columns)."""
    return {
        "model": "h100_col8_drop25",
        "total_columns": 8,
        "data": [
            {"active_columns": 8, "fraction": 1.0, "perplexity": 18.43, "vs_full_pct": 0.0},
            {"active_columns": 7, "fraction": 0.875, "perplexity": 18.67, "vs_full_pct": 1.3},
            {"active_columns": 6, "fraction": 0.75, "perplexity": 19.12, "vs_full_pct": 3.7},
            {"active_columns": 5, "fraction": 0.625, "perplexity": 19.51, "vs_full_pct": 5.9},
            {"active_columns": 4, "fraction": 0.5, "perplexity": 20.27, "vs_full_pct": 10.0},
            {"active_columns": 3, "fraction": 0.375, "perplexity": 22.19, "vs_full_pct": 20.4},
            {"active_columns": 2, "fraction": 0.25, "perplexity": 26.83, "vs_full_pct": 45.6},
            {"active_columns": 1, "fraction": 0.125, "perplexity": 36.52, "vs_full_pct": 98.2},
        ],
    }

@app.get("/api/experiments/bandwidth")
async def get_bandwidth_data():
    """Get bandwidth analysis data."""
    return {
        "model_scale": "1B",
        "n_columns": 8,
        "n_merge_points": 4,
        "variants": [
            {"name": "Full (uncompressed)", "per_merge_mb": 33.6, "total_mb": 134.2,
             "per_token_kb": 131, "compression": "1x"},
            {"name": "Rank-64 compressed", "per_merge_mb": 4.2, "total_mb": 16.8,
             "per_token_kb": 16.4, "compression": "8x"},
            {"name": "Rank-64 + int8", "per_merge_mb": 2.1, "total_mb": 8.4,
             "per_token_kb": 8.2, "compression": "16x"},
        ],
        "latency_by_interconnect": [
            {"interconnect": "NVLink (900 GB/s)", "full_ms": 0.15, "rank64_ms": 0.02, "rank64_int8_ms": 0.01},
            {"interconnect": "PCIe 5.0 (64 GB/s)", "full_ms": 2.10, "rank64_ms": 0.26, "rank64_int8_ms": 0.13},
            {"interconnect": "InfiniBand HDR", "full_ms": 5.37, "rank64_ms": 0.67, "rank64_int8_ms": 0.34},
            {"interconnect": "100GbE", "full_ms": 10.74, "rank64_ms": 1.34, "rank64_int8_ms": 0.67},
        ],
    }


# ============================================================
# Routes: Remote Log Tailing
# ============================================================

@app.get("/api/gpus/{gpu_id:path}/logs/{log_type}")
async def get_gpu_logs(gpu_id: str, log_type: str, lines: int = 50):
    """Get logs from a remote GPU (coordinator or worker)."""
    gpu = state.remote_gpus.get(gpu_id)
    if not gpu:
        raise HTTPException(404, "GPU not found")

    if log_type == "coordinator":
        log_file = "/tmp/coordinator.log"
    elif log_type.startswith("worker"):
        col = gpu.column_idx if gpu.column_idx >= 0 else 1
        log_file = f"/tmp/worker_{col}.log"
    else:
        log_file = f"/tmp/{log_type}.log"

    ok, output = await ssh_run(gpu.ssh_target, gpu.ssh_port,
                                f"tail -{lines} {log_file} 2>/dev/null || echo 'No log found'")
    return {"log": output, "ok": ok}


# ============================================================
# WebSocket: Live updates
# ============================================================

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    state.websocket_clients.append(ws)
    try:
        while True:
            data = await ws.receive_text()
            if data == "ping":
                await ws.send_text(json.dumps({"event": "pong"}))
    except WebSocketDisconnect:
        if ws in state.websocket_clients:
            state.websocket_clients.remove(ws)

# ============================================================
# Static files
# ============================================================

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# ============================================================
# CLI
# ============================================================

def main():
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="Column Transformer Web Platform")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--reload", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(
        "web.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )

if __name__ == "__main__":
    main()
