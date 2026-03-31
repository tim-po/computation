#!/usr/bin/env python3
"""Simulate distributed inference and analyze communication bandwidth.

Hooks into CrossColumnAttention merge points to inject network latency
and measure the bytes that would be transferred between GPUs. Runs on
a single GPU — no actual distributed setup required.

Usage:
    python eval_distributed.py \
        --model h100_1b_col8 \
        --checkpoint checkpoints/h100_1b_col8_best.pt \
        --latencies 0,10,50,100 \
        --batch-sizes 1,8,32 \
        --bf16

    # Static bandwidth analysis only (no checkpoint needed):
    python eval_distributed.py --model h100_1b_col8 --analyze-only
"""

import argparse
import json
import math
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from column_transformer.config import EXPERIMENTS, ColumnConfigV2
from column_transformer.model_column_v2 import ColumnTransformerV2
from column_transformer.data import load_wikitext
from column_transformer.train import get_device, count_parameters


class DistributedSimWrapper(nn.Module):
    """Wraps CrossColumnAttention to simulate distributed communication.

    At each merge point in real distributed inference:
      1. Each GPU computes its column's K and V locally
      2. K/V are gathered to a coordinator (or all-reduced)
      3. Coordinator averages K/V and broadcasts back
      4. Each GPU computes Q locally and runs attention

    This wrapper simulates steps 2-3 by:
      - Measuring the tensor sizes that would be communicated
      - Injecting time.sleep() to simulate network round-trip latency

    For compressed merge layers, the communicated dimension is comm_rank
    instead of d_col. With quant_comm, dtype_bytes is halved (int8).
    """

    def __init__(self, merge_layer, n_columns: int, d_col: int, latency_ms: float = 0.0,
                 comm_rank: int = 0, quant_comm: bool = False):
        super().__init__()
        self.merge_layer = merge_layer
        self.n_columns = n_columns
        self.d_col = d_col
        self.comm_dim = comm_rank if comm_rank > 0 else d_col
        self.quant_comm = quant_comm
        self.latency_s = latency_ms / 1000.0

        # Accumulated stats
        self.comm_bytes_total = 0
        self.comm_time_total = 0.0
        self.n_calls = 0

    def forward(self, columns, col_mask=None):
        B, T, _ = columns[0].shape

        # Count active columns
        if col_mask is not None:
            n_active = col_mask.sum().item()
        else:
            n_active = self.n_columns

        # Determine dtype size (int8 quantized comm uses 1 byte)
        dtype_bytes = 2 if columns[0].dtype in (torch.float16, torch.bfloat16) else 4
        if self.quant_comm:
            dtype_bytes = 1

        # Communication: uses comm_dim (which is comm_rank for compressed, d_col for full)
        bytes_per_tensor = B * T * self.comm_dim * dtype_bytes
        gather_bytes = n_active * 2 * bytes_per_tensor   # K + V from each column
        broadcast_bytes = n_active * 2 * bytes_per_tensor  # averaged K + V back
        comm_bytes = gather_bytes + broadcast_bytes
        self.comm_bytes_total += comm_bytes

        # Simulate latency
        if self.latency_s > 0:
            time.sleep(self.latency_s)
            self.comm_time_total += self.latency_s

        self.n_calls += 1

        return self.merge_layer(columns, col_mask)

    def reset_stats(self):
        self.comm_bytes_total = 0
        self.comm_time_total = 0.0
        self.n_calls = 0


def wrap_merge_layers(model, config, latency_ms: float) -> list[DistributedSimWrapper]:
    """Replace CrossColumnAttention layers in model.merge_layers with DistributedSimWrapper."""
    wrappers = []
    for key in list(model.merge_layers.keys()):
        original = model.merge_layers[key]
        wrapper = DistributedSimWrapper(
            original, config.n_columns, config.d_col, latency_ms,
            comm_rank=config.comm_rank, quant_comm=config.quant_comm,
        )
        model.merge_layers[key] = wrapper
        wrappers.append(wrapper)
    return wrappers


def unwrap_merge_layers(model, wrappers):
    """Restore original merge layers."""
    for key in list(model.merge_layers.keys()):
        layer = model.merge_layers[key]
        if isinstance(layer, DistributedSimWrapper):
            model.merge_layers[key] = layer.merge_layer


@torch.no_grad()
def run_inference_benchmark(
    model, val_loader, device, max_batches: int = 50, use_bf16: bool = False,
) -> tuple[float, int]:
    """Run inference and return (elapsed_seconds, total_tokens)."""
    model.eval()
    total_tokens = 0
    t_start = time.time()

    for i, (input_ids, targets) in enumerate(val_loader):
        if i >= max_batches:
            break
        input_ids = input_ids.to(device)
        with torch.amp.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
            _ = model(input_ids)
        total_tokens += input_ids.numel()

    # Sync GPU
    if device.type == "cuda":
        torch.cuda.synchronize()

    elapsed = time.time() - t_start
    return elapsed, total_tokens


def analyze_bandwidth(config, batch_sizes=None, dtype_bytes: int = 2):
    """Static analysis of communication requirements for distributed inference."""
    if batch_sizes is None:
        batch_sizes = [1, 8, 32, 64]

    n_merges = config.n_col_layers // config.merge_every if config.merge_every > 0 else 0
    T = config.max_seq_len
    comm_dim = config.comm_rank if config.comm_rank > 0 else config.d_col
    eff_dtype_bytes = 1 if config.quant_comm else dtype_bytes

    interconnects = {
        "NVLink (900 GB/s)": 900e9,
        "PCIe 5.0 (64 GB/s)": 64e9,
        "InfiniBand HDR (25 GB/s)": 25e9,
        "100GbE (12.5 GB/s)": 12.5e9,
    }

    print(f"\n{'='*70}")
    print("BANDWIDTH ANALYSIS")
    print(f"{'='*70}")
    print(f"  Model: {config.n_columns} columns, d_col={config.d_col}, "
          f"{n_merges} merge points, seq_len={T}")
    if config.comm_rank > 0:
        print(f"  Compression: rank={config.comm_rank} (d_col={config.d_col} -> {config.comm_rank}, "
              f"{config.d_col / config.comm_rank:.0f}x reduction)")
    if config.quant_comm:
        print(f"  Quantization: int8 (additional 2x reduction)")
    comm_label = f"comm_dim={comm_dim}, {eff_dtype_bytes}B/element"
    print(f"  Effective: {comm_label}")

    results = {}

    for B in batch_sizes:
        # Per merge: gather K,V from all columns + broadcast back
        gather_bytes = config.n_columns * 2 * B * T * comm_dim * eff_dtype_bytes
        broadcast_bytes = config.n_columns * 2 * B * T * comm_dim * eff_dtype_bytes
        per_merge = gather_bytes + broadcast_bytes
        total_per_forward = per_merge * n_merges
        per_token = total_per_forward / (B * T) if B * T > 0 else 0

        print(f"\n  Batch size {B}:")
        print(f"    Per merge point: {per_merge / 1e6:.2f} MB")
        print(f"    Total per forward ({n_merges} merges): {total_per_forward / 1e6:.2f} MB")
        print(f"    Per token: {per_token / 1e3:.2f} KB")

        merge_times = {}
        for name, bw in interconnects.items():
            merge_time_ms = (per_merge / bw) * 1000
            total_time_ms = merge_time_ms * n_merges
            status = "OK" if merge_time_ms < 1.0 else "marginal" if merge_time_ms < 5.0 else "BOTTLENECK"
            print(f"      {name}: {merge_time_ms:.2f}ms/merge, "
                  f"{total_time_ms:.2f}ms total — {status}")
            merge_times[name] = {"merge_ms": merge_time_ms, "total_ms": total_time_ms}

        results[B] = {
            "per_merge_mb": per_merge / 1e6,
            "total_mb": total_per_forward / 1e6,
            "per_token_kb": per_token / 1e3,
            "interconnects": merge_times,
        }

    return results


def main():
    parser = argparse.ArgumentParser(description="Distributed Inference Simulator")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name from EXPERIMENTS (e.g. h100_1b_col8)")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to model checkpoint (not needed with --analyze-only)")
    parser.add_argument("--latencies", type=str, default="0,10,50,100",
                        help="Comma-separated latencies in ms to simulate")
    parser.add_argument("--batch-sizes", type=str, default="1,8,32",
                        help="Comma-separated batch sizes to test")
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length (default: from config)")
    parser.add_argument("--max-batches", type=int, default=50,
                        help="Max batches per benchmark run")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16")
    parser.add_argument("--analyze-only", action="store_true",
                        help="Only run static bandwidth analysis (no checkpoint needed)")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load config
    config = EXPERIMENTS[args.model]
    assert isinstance(config, ColumnConfigV2), "Distributed sim only works with V2/V3 models"

    seq_len = args.seq_len or config.max_seq_len
    latencies = [float(x) for x in args.latencies.split(",")]
    batch_sizes = [int(x) for x in args.batch_sizes.split(",")]

    n_merges = config.n_col_layers // config.merge_every if config.merge_every > 0 else 0

    print(f"{'='*70}")
    print(f"DISTRIBUTED INFERENCE SIMULATION: {args.model}")
    print(f"{'='*70}")
    print(f"  Columns: {config.n_columns}, d_col: {config.d_col}")
    print(f"  Merge points per forward: {n_merges}")
    print(f"  Latencies to test: {latencies} ms")
    print(f"  Batch sizes to test: {batch_sizes}")

    # Static bandwidth analysis (always run)
    bw_results = analyze_bandwidth(
        config, batch_sizes=batch_sizes,
        dtype_bytes=2 if args.bf16 else 4,
    )

    if args.analyze_only:
        # Save and exit
        json_path = os.path.join(args.results_dir, f"bandwidth_{args.model}.json")
        with open(json_path, "w") as f:
            json.dump({"model": args.model, "bandwidth": bw_results}, f, indent=2, default=str)
        print(f"\nSaved bandwidth analysis to {json_path}")
        return

    # Load model for inference benchmark
    assert args.checkpoint is not None, "--checkpoint required for inference benchmark"

    device = get_device()
    model = ColumnTransformerV2(config)
    state_dict = torch.load(args.checkpoint, map_location=device, weights_only=True)
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    print(f"\n  Parameters: {count_parameters(model):,}")
    print(f"  Device: {device}")

    # Run inference benchmarks
    all_results = {}

    for batch_size in batch_sizes:
        _, val_loader = load_wikitext(seq_len=seq_len, batch_size=batch_size)

        print(f"\n{'─'*50}")
        print(f"Batch size: {batch_size}")
        print(f"{'─'*50}")

        batch_results = {}

        for latency_ms in latencies:
            # Wrap merge layers with latency injection
            wrappers = wrap_merge_layers(model, config, latency_ms)

            # Warmup
            run_inference_benchmark(model, val_loader, device, max_batches=3, use_bf16=args.bf16)
            for w in wrappers:
                w.reset_stats()

            # Benchmark
            elapsed, total_tokens = run_inference_benchmark(
                model, val_loader, device, max_batches=args.max_batches, use_bf16=args.bf16,
            )

            tokens_per_sec = total_tokens / elapsed if elapsed > 0 else 0
            total_comm_bytes = sum(w.comm_bytes_total for w in wrappers)
            total_comm_time = sum(w.comm_time_total for w in wrappers)
            total_calls = sum(w.n_calls for w in wrappers)
            bytes_per_token = total_comm_bytes / total_tokens if total_tokens > 0 else 0

            batch_results[latency_ms] = {
                "tokens_per_sec": tokens_per_sec,
                "elapsed_sec": elapsed,
                "total_tokens": total_tokens,
                "total_comm_mb": total_comm_bytes / 1e6,
                "comm_time_sec": total_comm_time,
                "bytes_per_token": bytes_per_token,
                "merge_calls": total_calls,
            }

            # Compute slowdown relative to 0ms latency
            if latency_ms == 0:
                baseline_tps = tokens_per_sec
            slowdown = ((baseline_tps - tokens_per_sec) / baseline_tps * 100) if baseline_tps > 0 else 0

            print(
                f"  Latency {latency_ms:>5.0f}ms: "
                f"{tokens_per_sec:>8.1f} tok/s "
                f"({slowdown:>+5.1f}%) | "
                f"comm: {total_comm_bytes / 1e6:.1f} MB total, "
                f"{bytes_per_token / 1e3:.1f} KB/tok"
            )

            # Unwrap for next iteration
            unwrap_merge_layers(model, wrappers)

        all_results[batch_size] = batch_results

    # Plot: tokens/sec vs latency for each batch size
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    for batch_size in batch_sizes:
        br = all_results[batch_size]
        lats = sorted(br.keys())
        tps = [br[l]["tokens_per_sec"] for l in lats]
        ax1.plot(lats, tps, "o-", linewidth=2, markersize=8, label=f"batch={batch_size}")

    ax1.set_xlabel("Simulated Latency (ms)", fontsize=12)
    ax1.set_ylabel("Tokens/sec", fontsize=12)
    ax1.set_title(f"Inference Throughput vs Network Latency\n{args.model}", fontsize=13)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: relative slowdown
    for batch_size in batch_sizes:
        br = all_results[batch_size]
        lats = sorted(br.keys())
        baseline = br[0]["tokens_per_sec"] if 0 in br else br[lats[0]]["tokens_per_sec"]
        slowdowns = [(1 - br[l]["tokens_per_sec"] / baseline) * 100 for l in lats]
        ax2.plot(lats, slowdowns, "o-", linewidth=2, markersize=8, label=f"batch={batch_size}")

    ax2.set_xlabel("Simulated Latency (ms)", fontsize=12)
    ax2.set_ylabel("Slowdown (%)", fontsize=12)
    ax2.set_title("Relative Throughput Degradation", fontsize=13)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, f"distributed_{args.model}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved plot to {plot_path}")

    # Save raw results
    json_path = os.path.join(args.results_dir, f"distributed_{args.model}.json")
    save_data = {
        "model": args.model,
        "config": {
            "n_columns": config.n_columns,
            "d_col": config.d_col,
            "n_merges": n_merges,
            "seq_len": seq_len,
        },
        "bandwidth": bw_results,
        "inference": {str(k): v for k, v in all_results.items()},
    }
    with open(json_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
