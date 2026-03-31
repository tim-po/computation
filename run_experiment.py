#!/usr/bin/env python3
"""Main experiment: train all column-parallel variants and compare against dense baseline."""

import argparse
import json
import os

from column_transformer.config import DenseConfig, ColumnConfig, ColumnConfigV2, EXPERIMENTS
from column_transformer.model_dense import DenseTransformer
from column_transformer.model_column import ColumnTransformer
from column_transformer.model_column_v2 import ColumnTransformerV2
from column_transformer.model_column_v2_fast import ColumnTransformerV2Fast
from column_transformer.data import load_wikitext
from column_transformer.train import train, count_parameters
from column_transformer.evaluate import print_comparison
from column_transformer.visualize import plot_training_curves, plot_final_comparison

# Global flag set by --fast CLI arg
_USE_FAST = False


def build_model(name: str, config):
    if isinstance(config, DenseConfig):
        return DenseTransformer(config)
    elif isinstance(config, ColumnConfigV2):
        if _USE_FAST:
            return ColumnTransformerV2Fast(config)
        return ColumnTransformerV2(config)
    else:
        return ColumnTransformer(config)


def main():
    parser = argparse.ArgumentParser(description="Column-Parallel Transformer Experiment")
    parser.add_argument("--max-steps", type=int, default=20000, help="Training steps per model")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length")
    parser.add_argument("--lr", type=float, default=3e-4, help="Peak learning rate")
    parser.add_argument("--eval-every", type=int, default=500, help="Evaluate every N steps")
    parser.add_argument("--models", nargs="+", default=None,
                        help="Which models to train (default: all). "
                             "Options: dense, column_merge_2, column_merge_4, column_no_merge")
    parser.add_argument("--grad-accum", type=int, default=1,
                        help="Gradient accumulation steps (effective batch = batch-size * grad-accum)")
    parser.add_argument("--bf16", action="store_true", help="Use bfloat16 mixed precision")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile")
    parser.add_argument("--fast", action="store_true",
                        help="Use vectorized (fast) column model — 3-5x faster on GPU")
    parser.add_argument("--dataset", type=str, default="wikitext",
                        choices=["wikitext", "fineweb-edu"],
                        help="Dataset to use (default: wikitext)")
    parser.add_argument("--results-dir", type=str, default="results", help="Output directory")
    args = parser.parse_args()

    global _USE_FAST
    _USE_FAST = args.fast

    os.makedirs(args.results_dir, exist_ok=True)
    os.makedirs("checkpoints", exist_ok=True)

    # Select which models to train
    model_names = args.models or list(EXPERIMENTS.keys())

    # Print experiment overview
    print("=" * 60)
    print("COLUMN-PARALLEL TRANSFORMER EXPERIMENT")
    print("=" * 60)
    for name in model_names:
        config = EXPERIMENTS[name]
        model = build_model(name, config)
        print(f"  {name}: {count_parameters(model):,} params")
        del model
    eff_batch = args.batch_size * args.grad_accum
    print(f"  Training: {args.max_steps} steps, batch {args.batch_size}×{args.grad_accum}={eff_batch}, seq {args.seq_len}")
    print("=" * 60)

    # Load data once (shared by all models)
    if args.dataset == "fineweb-edu":
        from column_transformer.data import load_fineweb_edu
        train_loader, val_loader = load_fineweb_edu(
            seq_len=args.seq_len, batch_size=args.batch_size
        )
    else:
        train_loader, val_loader = load_wikitext(
            seq_len=args.seq_len, batch_size=args.batch_size
        )

    # Train each model
    all_results = []
    for name in model_names:
        config = EXPERIMENTS[name]
        model = build_model(name, config)
        result = train(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            max_steps=args.max_steps,
            lr=args.lr,
            eval_every=args.eval_every,
            model_name=name,
            save_dir="checkpoints",
            grad_accum_steps=args.grad_accum,
            use_bf16=args.bf16,
            use_compile=getattr(args, 'compile', False),
        )
        all_results.append(result)
        # Free GPU memory between models
        del model
        import torch, gc
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()

    # Print comparison
    print_comparison(all_results)

    # Generate plots
    plot_training_curves(all_results, save_dir=args.results_dir)
    plot_final_comparison(all_results, save_dir=args.results_dir)

    # Save raw results
    serializable = []
    for r in all_results:
        serializable.append({
            "model_name": r["model_name"],
            "params": r["params"],
            "best_val_loss": r["best_val_loss"],
            "total_time": r["total_time"],
            "val_perplexities": r["val_perplexities"],
        })
    with open(os.path.join(args.results_dir, "results.json"), "w") as f:
        json.dump(serializable, f, indent=2)
    print(f"\nResults saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
