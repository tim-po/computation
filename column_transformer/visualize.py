"""Visualization: training curves and comparison charts."""

import os
import math
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt


COLORS = {
    "dense": "#2196F3",
    "column_merge_2": "#4CAF50",
    "column_merge_4": "#FF9800",
    "column_no_merge": "#F44336",
}


def smooth(values, weight=0.9):
    """Exponential moving average for smoother curves."""
    smoothed = []
    last = values[0] if values else 0
    for v in values:
        last = weight * last + (1 - weight) * v
        smoothed.append(last)
    return smoothed


def plot_training_curves(all_results: list[dict], save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # --- Training loss ---
    ax = axes[0]
    for r in all_results:
        name = r["model_name"]
        steps = [s for s, _ in r["train_losses"]]
        losses = [l for _, l in r["train_losses"]]
        smoothed = smooth(losses)
        color = COLORS.get(name, "#999999")
        ax.plot(steps, smoothed, label=name, color=color, linewidth=1.5)
    ax.set_xlabel("Step")
    ax.set_ylabel("Training Loss")
    ax.set_title("Training Loss (smoothed)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # --- Validation perplexity ---
    ax = axes[1]
    for r in all_results:
        name = r["model_name"]
        steps = [s for s, _ in r["val_perplexities"]]
        ppls = [p for _, p in r["val_perplexities"]]
        color = COLORS.get(name, "#999999")
        ax.plot(steps, ppls, label=name, color=color, linewidth=2, marker="o", markersize=4)
    ax.set_xlabel("Step")
    ax.set_ylabel("Perplexity")
    ax.set_title("Validation Perplexity")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "training_curves.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved training curves to {path}")


def plot_final_comparison(all_results: list[dict], save_dir: str = "results"):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    names = [r["model_name"] for r in all_results]
    colors = [COLORS.get(n, "#999999") for n in names]

    # --- Final perplexity bar chart ---
    ax = axes[0]
    ppls = [math.exp(min(r["best_val_loss"], 20)) for r in all_results]
    bars = ax.bar(names, ppls, color=colors)
    ax.set_ylabel("Best Validation Perplexity")
    ax.set_title("Final Perplexity Comparison")
    for bar, ppl in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(),
                f"{ppl:.1f}", ha="center", va="bottom", fontsize=10)
    ax.tick_params(axis="x", rotation=15)

    # --- Parameter count vs perplexity scatter ---
    ax = axes[1]
    params = [r["params"] / 1e6 for r in all_results]
    for i, r in enumerate(all_results):
        ax.scatter(params[i], ppls[i], color=colors[i], s=100, zorder=5)
        ax.annotate(r["model_name"], (params[i], ppls[i]),
                    textcoords="offset points", xytext=(5, 5), fontsize=9)
    ax.set_xlabel("Parameters (M)")
    ax.set_ylabel("Best Validation Perplexity")
    ax.set_title("Parameter Efficiency")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    path = os.path.join(save_dir, "final_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Saved final comparison to {path}")
