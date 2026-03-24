"""Evaluation: compute perplexity on a validation set."""

import math
import torch
import torch.nn as nn


@torch.no_grad()
def evaluate(
    model: nn.Module,
    val_loader,
    device: torch.device,
    loss_fn: nn.Module | None = None,
    max_batches: int = 100,
) -> tuple[float, float]:
    """Returns (avg_loss, perplexity) on validation set."""
    if loss_fn is None:
        loss_fn = nn.CrossEntropyLoss()

    model.eval()
    total_loss = 0.0
    n_batches = 0

    for input_ids, targets in val_loader:
        if n_batches >= max_batches:
            break
        input_ids = input_ids.to(device)
        targets = targets.to(device)

        logits = model(input_ids)
        loss = loss_fn(logits.view(-1, logits.size(-1)), targets.view(-1))
        total_loss += loss.item()
        n_batches += 1

    avg_loss = total_loss / max(n_batches, 1)
    perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
    return avg_loss, perplexity


def print_comparison(results: list[dict]):
    """Print a comparison table of all experiment results."""
    print("\n" + "=" * 80)
    print("EXPERIMENT RESULTS")
    print("=" * 80)
    print(f"{'Model':<25} {'Params':>10} {'Best Val Loss':>15} {'Best Val PPL':>15} {'Time (min)':>12}")
    print("-" * 80)

    for r in sorted(results, key=lambda x: x["best_val_loss"]):
        best_ppl = math.exp(min(r["best_val_loss"], 20))
        print(
            f"{r['model_name']:<25} "
            f"{r['params']:>10,} "
            f"{r['best_val_loss']:>15.4f} "
            f"{best_ppl:>15.2f} "
            f"{r['total_time']/60:>12.1f}"
        )
    print("=" * 80)
