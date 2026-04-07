"""Split a full ColumnTransformerV2 checkpoint into coordinator + worker shards.

Usage:
    python -m distributed.checkpoint_split \
        --checkpoint results_h100/h100_col8_drop25_best.pt \
        --model h100_col8_drop25 \
        --output-dir shards/
"""

import argparse
import re
import os
import torch

from column_transformer.config import EXPERIMENTS, ColumnConfigV2


def split_checkpoint(
    state_dict: dict[str, torch.Tensor],
    config: ColumnConfigV2,
    coordinator_column: int = 0,
) -> tuple[dict, list[dict]]:
    """Split state_dict into coordinator shard + N-1 worker shards.

    Coordinator gets: trunk, embeddings, column 0, ALL merge layers, output.
    Worker i gets: column i layers, per-column merge weights for i, shared decompress.

    Returns:
        (coordinator_shard, [worker_1_shard, worker_2_shard, ...])
        Worker list has N-1 entries (one per non-coordinator column).
    """
    coord_shard = {}
    worker_shards = {i: {} for i in range(config.n_columns) if i != coordinator_column}
    worker_col_indices = sorted(worker_shards.keys())

    # Merge layer keys where we need to extract per-column weights
    merge_keys = set()
    if config.merge_every > 0:
        for layer_idx in range(config.merge_every, config.n_col_layers + 1, config.merge_every):
            merge_keys.add(str(layer_idx))

    for key, value in state_dict.items():
        # Strip torch.compile prefix
        clean_key = key.replace("_orig_mod.", "")

        # --- Trunk, embedding, output (coordinator only) ---
        if any(clean_key.startswith(p) for p in [
            "tok_emb.", "drop.", "trunk.", "out_proj.", "norm.", "lm_head.",
        ]):
            coord_shard[clean_key] = value
            continue

        # --- Column input projections ---
        m = re.match(r"col_in_projs\.(\d+)\.(.*)", clean_key)
        if m:
            col_idx = int(m.group(1))
            if col_idx == coordinator_column:
                coord_shard[clean_key] = value
            elif col_idx in worker_shards:
                worker_shards[col_idx][clean_key] = value
            continue

        # --- Per-column transformer blocks ---
        m = re.match(r"columns\.(\d+)\.(.*)", clean_key)
        if m:
            col_idx = int(m.group(1))
            if col_idx == coordinator_column:
                coord_shard[clean_key] = value
            elif col_idx in worker_shards:
                worker_shards[col_idx][clean_key] = value
            continue

        # --- Merge layers: split per-column vs shared ---
        m = re.match(r"merge_layers\.(\w+)\.(.*)", clean_key)
        if m:
            merge_key = m.group(1)
            rest = m.group(2)

            # Coordinator always gets full merge layers (for orchestration)
            coord_shard[clean_key] = value

            # Check if this is a per-column weight (has column index)
            col_m = re.match(r"(norms_kv|w_ks|w_vs|k_compress|v_compress|norms_q|w_qs|w_os)\.(\d+)\.(.*)", rest)
            if col_m:
                col_idx = int(col_m.group(2))
                if col_idx in worker_shards:
                    worker_shards[col_idx][clean_key] = value
            else:
                # Shared weights (k_decompress, v_decompress, etc.) — all workers need these
                for col_idx in worker_col_indices:
                    worker_shards[col_idx][clean_key] = value
            continue

        # --- Column dropout (not needed for inference, but include in coordinator) ---
        if clean_key.startswith("col_dropout."):
            coord_shard[clean_key] = value
            continue

        # --- RoPE buffers ---
        if clean_key in ("trunk_rope", "col_rope"):
            coord_shard[clean_key] = value
            # Workers need col_rope
            if clean_key == "col_rope":
                for col_idx in worker_col_indices:
                    worker_shards[col_idx][clean_key] = value
            continue

        # Catch-all: goes to coordinator
        coord_shard[clean_key] = value

    worker_list = [worker_shards[i] for i in worker_col_indices]
    return coord_shard, worker_list


def save_shards(
    checkpoint_path: str,
    config: ColumnConfigV2,
    output_dir: str,
    coordinator_column: int = 0,
) -> None:
    """Load checkpoint, split, and save shards to disk."""
    os.makedirs(output_dir, exist_ok=True)

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    # Strip torch.compile prefix
    state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    coord_shard, worker_shards = split_checkpoint(state_dict, config, coordinator_column)

    coord_path = os.path.join(output_dir, "coordinator.pt")
    torch.save(coord_shard, coord_path)
    coord_mb = sum(v.numel() * v.element_size() for v in coord_shard.values()) / 1e6
    print(f"  Coordinator shard: {coord_path} ({len(coord_shard)} tensors, {coord_mb:.1f} MB)")

    worker_col_indices = [i for i in range(config.n_columns) if i != coordinator_column]
    for idx, (col_idx, shard) in enumerate(zip(worker_col_indices, worker_shards)):
        worker_path = os.path.join(output_dir, f"worker_{col_idx}.pt")
        torch.save(shard, worker_path)
        worker_mb = sum(v.numel() * v.element_size() for v in shard.values()) / 1e6
        print(f"  Worker {col_idx} shard: {worker_path} ({len(shard)} tensors, {worker_mb:.1f} MB)")

    # Verify: all params accounted for
    total_original = sum(v.numel() for v in state_dict.values())
    total_coord = sum(v.numel() for v in coord_shard.values())
    total_workers = sum(sum(v.numel() for v in s.values()) for s in worker_shards)
    print(f"\n  Original params: {total_original:,}")
    print(f"  Coordinator params: {total_coord:,}")
    print(f"  Worker params (total): {total_workers:,}")
    # Note: some params are duplicated (shared decompress in all workers + coordinator)


def main():
    parser = argparse.ArgumentParser(description="Split checkpoint into distributed shards")
    parser.add_argument("--checkpoint", required=True, help="Path to full model checkpoint")
    parser.add_argument("--model", required=True, help="Model config name from EXPERIMENTS")
    parser.add_argument("--output-dir", required=True, help="Output directory for shards")
    parser.add_argument("--coordinator-column", type=int, default=0,
                        help="Which column the coordinator runs (default: 0)")
    args = parser.parse_args()

    config = EXPERIMENTS[args.model]
    assert isinstance(config, ColumnConfigV2), f"{args.model} is not a ColumnConfigV2"

    print(f"Splitting checkpoint for {args.model} ({config.n_columns} columns)...")
    save_shards(args.checkpoint, config, args.output_dir, args.coordinator_column)
    print("Done!")


if __name__ == "__main__":
    main()
