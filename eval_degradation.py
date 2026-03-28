#!/usr/bin/env python3
"""Evaluate how a column-dropout model degrades as columns are removed.

Loads a trained V2/V3 checkpoint and evaluates perplexity with every possible
number of active columns (1 to N). For each count, averages over multiple
random column subsets to reduce variance.

This simulates distributed inference: what happens when only K of N GPUs are
available?

Usage:
    # After training h100_col8_drop25:
    python eval_degradation.py \
        --model h100_col8_drop25 \
        --checkpoint checkpoints/h100_col8_drop25_best.pt \
        --samples-per-k 10

Output:
    - Table: columns_active → perplexity
    - Plot: degradation curve (saved to results/)
"""

import argparse
import itertools
import json
import math
import os
import random

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

from column_transformer.config import EXPERIMENTS, ColumnConfigV2
from column_transformer.model_column_v2 import ColumnTransformerV2
from column_transformer.data import load_wikitext
from column_transformer.train import get_device, count_parameters


@torch.no_grad()
def evaluate_with_columns(
    model: nn.Module,
    val_loader,
    device: torch.device,
    active_columns: list[int],
    max_batches: int = 100,
    use_bf16: bool = False,
) -> tuple[float, float]:
    """Evaluate model using only the specified active columns."""
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss = 0.0
    n_batches = 0

    for input_ids, targets in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        with torch.amp.autocast("cuda", enabled=use_bf16, dtype=torch.bfloat16):
            logits = model(input_ids, active_columns=active_columns)
        loss = loss_fn(logits.float().view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))
    return avg_loss, perplexity


def get_column_subsets(n_columns: int, k: int, n_samples: int) -> list[list[int]]:
    """Get column subsets of size k to evaluate.

    If all C(n,k) combinations fit within n_samples, use all of them.
    Otherwise, sample randomly.
    """
    all_combos = list(itertools.combinations(range(n_columns), k))
    if len(all_combos) <= n_samples:
        return [list(c) for c in all_combos]
    else:
        return [list(c) for c in random.sample(all_combos, n_samples)]


def main():
    parser = argparse.ArgumentParser(description="Column Degradation Evaluation")
    parser.add_argument("--model", type=str, required=True,
                        help="Model name from EXPERIMENTS (e.g. h100_col8_drop25)")
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to model checkpoint")
    parser.add_argument("--samples-per-k", type=int, default=10,
                        help="Number of random column subsets to average per K")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seq-len", type=int, default=None,
                        help="Sequence length (default: from config)")
    parser.add_argument("--max-batches", type=int, default=100,
                        help="Max validation batches per evaluation")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 for evaluation")
    parser.add_argument("--results-dir", type=str, default="results")
    args = parser.parse_args()

    os.makedirs(args.results_dir, exist_ok=True)

    # Load config and model
    config = EXPERIMENTS[args.model]
    assert isinstance(config, ColumnConfigV2), "Degradation eval only works with V2/V3 models"

    seq_len = args.seq_len or config.max_seq_len
    n_columns = config.n_columns

    print(f"Loading model: {args.model}")
    print(f"  Columns: {n_columns}")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Samples per K: {args.samples_per_k}")

    device = get_device()
    model = ColumnTransformerV2(config)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device, weights_only=True))
    model = model.to(device)
    model.eval()
    print(f"  Parameters: {count_parameters(model):,}")

    # Load validation data
    _, val_loader = load_wikitext(seq_len=seq_len, batch_size=args.batch_size)

    # Evaluate with all columns (baseline)
    print(f"\n{'='*70}")
    print(f"DEGRADATION EVALUATION: {args.model}")
    print(f"{'='*70}")

    results = []

    for k in range(1, n_columns + 1):
        subsets = get_column_subsets(n_columns, k, args.samples_per_k)
        losses = []
        ppls = []

        for subset in subsets:
            loss, ppl = evaluate_with_columns(model, val_loader, device, subset, args.max_batches, args.bf16)
            losses.append(loss)
            ppls.append(ppl)

        avg_loss = sum(losses) / len(losses)
        avg_ppl = sum(ppls) / len(ppls)
        min_ppl = min(ppls)
        max_ppl = max(ppls)

        results.append({
            "k": k,
            "n_subsets": len(subsets),
            "avg_loss": avg_loss,
            "avg_ppl": avg_ppl,
            "min_ppl": min_ppl,
            "max_ppl": max_ppl,
        })

        print(
            f"  {k}/{n_columns} columns | "
            f"avg_ppl {avg_ppl:>8.2f} | "
            f"range [{min_ppl:.2f}, {max_ppl:.2f}] | "
            f"{len(subsets)} subsets evaluated"
        )

    print(f"{'='*70}")

    # Compute relative degradation
    full_ppl = results[-1]["avg_ppl"]
    print(f"\nRelative to full model (PPL {full_ppl:.2f}):")
    for r in results:
        ratio = r["avg_ppl"] / full_ppl
        bar = "█" * int(ratio * 20)
        print(f"  {r['k']}/{n_columns} cols: {ratio:.2f}x  {bar}")

    # Plot degradation curve
    ks = [r["k"] for r in results]
    avg_ppls = [r["avg_ppl"] for r in results]
    min_ppls = [r["min_ppl"] for r in results]
    max_ppls = [r["max_ppl"] for r in results]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: absolute perplexity
    ax1.plot(ks, avg_ppls, "o-", color="tab:blue", linewidth=2, markersize=8, label="avg PPL")
    ax1.fill_between(ks, min_ppls, max_ppls, alpha=0.2, color="tab:blue", label="min/max range")
    ax1.set_xlabel("Active Columns", fontsize=12)
    ax1.set_ylabel("Perplexity", fontsize=12)
    ax1.set_title(f"Degradation Curve: {args.model}", fontsize=14)
    ax1.set_xticks(ks)
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Right: relative degradation
    ratios = [r["avg_ppl"] / full_ppl for r in results]
    colors = ["tab:red" if r > 2.0 else "tab:orange" if r > 1.5 else "tab:green" for r in ratios]
    ax2.bar(ks, ratios, color=colors, edgecolor="black", linewidth=0.5)
    ax2.axhline(y=1.0, color="black", linestyle="--", alpha=0.5, label="full model")
    ax2.axhline(y=1.5, color="tab:orange", linestyle="--", alpha=0.5, label="1.5x threshold")
    ax2.axhline(y=2.0, color="tab:red", linestyle="--", alpha=0.5, label="2x threshold")
    ax2.set_xlabel("Active Columns", fontsize=12)
    ax2.set_ylabel("PPL Ratio (vs full model)", fontsize=12)
    ax2.set_title("Relative Quality Degradation", fontsize=14)
    ax2.set_xticks(ks)
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plot_path = os.path.join(args.results_dir, f"degradation_{args.model}.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved degradation plot to {plot_path}")

    # Save raw results
    json_path = os.path.join(args.results_dir, f"degradation_{args.model}.json")
    with open(json_path, "w") as f:
        json.dump({"model": args.model, "n_columns": n_columns, "results": results}, f, indent=2)
    print(f"Saved results to {json_path}")


if __name__ == "__main__":
    main()
