"""Distributed worker: runs a single column of the column-parallel transformer.

Each worker loads only its column's weights and communicates with the
coordinator at merge points via compressed K/V exchange.
"""

import asyncio
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from column_transformer.config import ColumnConfigV2
from column_transformer.model_dense import TransformerBlock, precompute_rope
from .protocol import MsgType, send_msg, recv_msg, send_tensor_pair, recv_tensor_pair

logger = logging.getLogger(__name__)


class ColumnWorkerModel(nn.Module):
    """Holds only the weights needed for a single column.

    This is NOT a full model — it's a collection of modules that the worker
    loads from a shard checkpoint and uses step-by-step during the distributed
    forward pass.
    """

    def __init__(self, config: ColumnConfigV2, column_idx: int):
        super().__init__()
        self.config = config
        self.column_idx = column_idx

        # Column input projection
        self.col_in_proj = nn.Linear(config.d_model, config.d_col)

        # Column transformer blocks
        self.layers = nn.ModuleList([
            TransformerBlock(config.d_col, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_col_layers)
        ])

        # RoPE for column attention
        col_rope = precompute_rope(config.head_dim, config.max_seq_len)
        self.register_buffer("col_rope", col_rope, persistent=False)

        # Merge layer components (per-column + shared decompress)
        self.merge_norms_kv = nn.ModuleDict()
        self.merge_w_ks = nn.ModuleDict()
        self.merge_w_vs = nn.ModuleDict()
        self.merge_norms_q = nn.ModuleDict()
        self.merge_w_qs = nn.ModuleDict()
        self.merge_w_os = nn.ModuleDict()

        # Compression (only if comm_rank > 0)
        self.merge_k_compress = nn.ModuleDict()
        self.merge_v_compress = nn.ModuleDict()
        self.merge_k_decompress = nn.ModuleDict()
        self.merge_v_decompress = nn.ModuleDict()

        self.merge_keys = []
        if config.merge_every > 0:
            for layer_idx in range(config.merge_every, config.n_col_layers + 1, config.merge_every):
                mk = str(layer_idx)
                self.merge_keys.append(mk)

                self.merge_norms_kv[mk] = nn.LayerNorm(config.d_col)
                self.merge_w_ks[mk] = nn.Linear(config.d_col, config.d_col, bias=False)
                self.merge_w_vs[mk] = nn.Linear(config.d_col, config.d_col, bias=False)
                self.merge_norms_q[mk] = nn.LayerNorm(config.d_col)
                self.merge_w_qs[mk] = nn.Linear(config.d_col, config.d_col, bias=False)
                self.merge_w_os[mk] = nn.Linear(config.d_col, config.d_col, bias=False)

                if config.comm_rank > 0:
                    self.merge_k_compress[mk] = nn.Linear(config.d_col, config.comm_rank, bias=False)
                    self.merge_v_compress[mk] = nn.Linear(config.d_col, config.comm_rank, bias=False)
                    self.merge_k_decompress[mk] = nn.Linear(config.comm_rank, config.d_col, bias=False)
                    self.merge_v_decompress[mk] = nn.Linear(config.comm_rank, config.d_col, bias=False)

    def load_from_shard(self, shard: dict[str, torch.Tensor]) -> None:
        """Load weights from a worker shard checkpoint."""
        ci = self.column_idx
        mapping = {}

        # Column input projection
        for suffix in ["weight", "bias"]:
            src = f"col_in_projs.{ci}.{suffix}"
            dst = f"col_in_proj.{suffix}"
            if src in shard:
                mapping[dst] = shard[src]

        # Column transformer blocks
        for key, value in shard.items():
            # columns.{col_idx}.{layer_idx}.{rest}
            prefix = f"columns.{ci}."
            if key.startswith(prefix):
                rest = key[len(prefix):]
                mapping[f"layers.{rest}"] = value

        # Merge layers
        for key, value in shard.items():
            if not key.startswith("merge_layers."):
                continue
            # merge_layers.{merge_key}.{component}.{col_idx}.{param}
            parts = key.split(".")
            merge_key = parts[1]
            component = parts[2]

            # Per-column components
            if component in ("norms_kv", "w_ks", "w_vs", "norms_q", "w_qs", "w_os",
                             "k_compress", "v_compress"):
                if len(parts) >= 4 and parts[3] == str(ci):
                    param = ".".join(parts[4:])
                    local_component = f"merge_{component}"
                    mapping[f"{local_component}.{merge_key}.{param}"] = value

            # Shared decompress
            elif component in ("k_decompress", "v_decompress"):
                param = ".".join(parts[3:])
                local_component = f"merge_{component}"
                mapping[f"{local_component}.{merge_key}.{param}"] = value

        # col_rope buffer
        if "col_rope" in shard:
            mapping["col_rope"] = shard["col_rope"]

        self.load_state_dict(mapping, strict=False)

    def compute_compressed_kv(
        self, col_state: torch.Tensor, merge_key: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute compressed K/V for sending to coordinator at a merge point."""
        normed = self.merge_norms_kv[merge_key](col_state)
        k_full = self.merge_w_ks[merge_key](normed)
        v_full = self.merge_w_vs[merge_key](normed)

        if self.config.comm_rank > 0:
            k_c = self.merge_k_compress[merge_key](k_full)
            v_c = self.merge_v_compress[merge_key](v_full)
            return k_c, v_c
        else:
            return k_full, v_full

    def apply_merged_kv(
        self, col_state: torch.Tensor, k_avg: torch.Tensor, v_avg: torch.Tensor, merge_key: str
    ) -> torch.Tensor:
        """Apply attention using averaged K/V received from coordinator."""
        config = self.config

        # Decompress if needed
        if config.comm_rank > 0:
            k = self.merge_k_decompress[merge_key](k_avg)
            v = self.merge_v_decompress[merge_key](v_avg)
        else:
            k = k_avg
            v = v_avg

        n_cross_heads = config.n_cross_heads
        head_dim = config.d_col // n_cross_heads
        B, T, _ = col_state.shape

        k = k.view(B, T, n_cross_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_cross_heads, head_dim).transpose(1, 2)

        # Local Q projection + attention
        q = self.merge_w_qs[merge_key](self.merge_norms_q[merge_key](col_state))
        q = q.view(B, T, n_cross_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, config.d_col)

        return col_state + self.merge_w_os[merge_key](attn_out)


class DistributedWorker:
    """Worker process that runs a single column and communicates with coordinator."""

    def __init__(
        self,
        column_idx: int,
        config: ColumnConfigV2,
        shard_path: str,
        coordinator_host: str = "localhost",
        coordinator_port: int = 9000,
        device: str = "cpu",
    ):
        self.column_idx = column_idx
        self.config = config
        self.device = device
        self.coordinator_host = coordinator_host
        self.coordinator_port = coordinator_port

        # Build and load model
        self.model = ColumnWorkerModel(config, column_idx)
        shard = torch.load(shard_path, map_location="cpu", weights_only=True)
        self.model.load_from_shard(shard)
        self.model.to(device)
        self.model.eval()

        logger.info(f"Worker {column_idx}: loaded shard, "
                     f"{sum(p.numel() for p in self.model.parameters()):,} params on {device}")

    async def run_once(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Run one forward pass: receive col_state, process layers, exchange at merges."""
        # Receive initial column state from coordinator
        msg_type, col_state, _ = await recv_msg(reader, self.device)
        assert msg_type == MsgType.COL_STATE, f"Expected COL_STATE, got {msg_type}"
        logger.debug(f"Worker {self.column_idx}: received col_state {col_state.shape}")

        merge_key_set = set(self.model.merge_keys)

        with torch.no_grad():
            for layer_idx in range(self.config.n_col_layers):
                # Run column layer
                col_state = self.model.layers[layer_idx](col_state, self.model.col_rope)

                # Check for merge point
                merge_key = str(layer_idx + 1)
                if merge_key in merge_key_set:
                    # Compute compressed K/V
                    k_c, v_c = self.model.compute_compressed_kv(col_state, merge_key)

                    # Send to coordinator
                    await send_tensor_pair(writer, MsgType.COMPRESSED_KV, k_c, v_c, layer_idx + 1)

                    # Receive averaged K/V
                    msg_type, k_avg, v_avg, _ = await recv_tensor_pair(reader, self.device)
                    assert msg_type == MsgType.AVERAGED_KV, f"Expected AVERAGED_KV, got {msg_type}"

                    # Apply attention with averaged K/V
                    col_state = self.model.apply_merged_kv(col_state, k_avg, v_avg, merge_key)

            # Send final column state
            await send_msg(writer, MsgType.FINAL_STATE, col_state)
            logger.debug(f"Worker {self.column_idx}: sent final state")

    async def connect_and_run(self, n_forward: int = 1) -> None:
        """Connect to coordinator and run forward passes."""
        reader, writer = await asyncio.open_connection(
            self.coordinator_host, self.coordinator_port
        )
        logger.info(f"Worker {self.column_idx}: connected to "
                     f"{self.coordinator_host}:{self.coordinator_port}")

        # Register
        register_data = torch.tensor([self.column_idx], dtype=torch.int32)
        await send_msg(writer, MsgType.REGISTER, register_data)

        msg_type, _, _ = await recv_msg(reader, self.device)
        assert msg_type == MsgType.ACK, f"Expected ACK, got {msg_type}"
        logger.info(f"Worker {self.column_idx}: registered and acknowledged")

        try:
            for i in range(n_forward):
                # Wait for READY signal
                msg_type, _, _ = await recv_msg(reader, self.device)
                if msg_type == MsgType.SHUTDOWN:
                    logger.info(f"Worker {self.column_idx}: received shutdown")
                    break
                assert msg_type == MsgType.READY, f"Expected READY, got {msg_type}"
                await self.run_once(reader, writer)
        finally:
            writer.close()
            await writer.wait_closed()
