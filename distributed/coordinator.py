"""Distributed coordinator: runs trunk + column 0 + orchestrates merges.

The coordinator:
1. Loads the full model (or coordinator shard)
2. Runs the shared trunk on input tokens
3. Projects to column states, sends each worker its column state
4. At merge points: gathers compressed K/V from workers, averages, broadcasts back
5. Collects final column states, runs output projection to produce logits
"""

import asyncio
import logging
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from column_transformer.config import ColumnConfigV2
from column_transformer.model_column_v2 import ColumnTransformerV2
from column_transformer.model_dense import precompute_rope
from .protocol import MsgType, send_msg, recv_msg, send_tensor_pair, recv_tensor_pair

logger = logging.getLogger(__name__)


class DistributedCoordinator:
    """Coordinator for distributed column-parallel inference."""

    def __init__(
        self,
        config: ColumnConfigV2,
        checkpoint_path: str,
        host: str = "0.0.0.0",
        port: int = 9000,
        device: str = "cpu",
        coordinator_column: int = 0,
        merge_timeout: float = 30.0,
    ):
        self.config = config
        self.host = host
        self.port = port
        self.device = device
        self.coordinator_column = coordinator_column
        self.merge_timeout = merge_timeout

        # Load full model
        self.model = ColumnTransformerV2(config)
        state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}
        self.model.load_state_dict(state_dict)
        self.model.to(device)
        self.model.eval()

        params = sum(p.numel() for p in self.model.parameters())
        logger.info(f"Coordinator: loaded model ({params:,} params) on {device}")

        # Connected workers: column_idx -> (reader, writer)
        self.workers: dict[int, tuple[asyncio.StreamReader, asyncio.StreamWriter]] = {}
        self._server = None

        # Compute merge point keys
        self.merge_keys = set()
        if config.merge_every > 0:
            for layer_idx in range(config.merge_every, config.n_col_layers + 1, config.merge_every):
                self.merge_keys.add(str(layer_idx))

    @property
    def active_columns(self) -> list[int]:
        """Currently active column indices (coordinator + connected workers)."""
        cols = [self.coordinator_column] + sorted(self.workers.keys())
        return cols

    @property
    def n_active(self) -> int:
        return len(self.active_columns)

    async def start_server(self) -> None:
        """Start TCP server and wait for workers to connect."""
        self._server = await asyncio.start_server(
            self._handle_worker_connection, self.host, self.port
        )
        addr = self._server.sockets[0].getsockname()
        logger.info(f"Coordinator: listening on {addr[0]}:{addr[1]}")

    async def _handle_worker_connection(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        """Handle a new worker connection (registration only)."""
        msg_type, data, _ = await recv_msg(reader, "cpu")
        assert msg_type == MsgType.REGISTER, f"Expected REGISTER, got {msg_type}"
        column_idx = int(data[0].item())

        if column_idx == self.coordinator_column:
            logger.warning(f"Worker tried to register as coordinator column {column_idx}, rejecting")
            writer.close()
            return

        self.workers[column_idx] = (reader, writer)
        await send_msg(writer, MsgType.ACK)
        logger.info(f"Coordinator: worker registered for column {column_idx} "
                     f"(active: {self.active_columns})")

    async def wait_for_workers(self, min_workers: int = 1, timeout: float = 60.0) -> None:
        """Wait until at least min_workers have connected."""
        start = time.time()
        while len(self.workers) < min_workers:
            if time.time() - start > timeout:
                logger.warning(f"Timeout waiting for workers. "
                               f"Got {len(self.workers)}/{min_workers}")
                break
            await asyncio.sleep(0.1)
        logger.info(f"Coordinator: {len(self.workers)} workers connected, "
                     f"active columns: {self.active_columns}")

    def _compute_coordinator_compressed_kv(
        self, col_state: torch.Tensor, merge_key: str
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute compressed K/V for the coordinator's own column."""
        ci = self.coordinator_column
        merge_layer = self.model.merge_layers[merge_key]

        normed = merge_layer.norms_kv[ci](col_state)
        k_full = merge_layer.w_ks[ci](normed)
        v_full = merge_layer.w_vs[ci](normed)

        if self.config.comm_rank > 0:
            k_c = merge_layer.k_compress[ci](k_full)
            v_c = merge_layer.v_compress[ci](v_full)
            return k_c, v_c
        else:
            return k_full, v_full

    def _apply_merged_kv(
        self, col_state: torch.Tensor, k_avg: torch.Tensor, v_avg: torch.Tensor, merge_key: str
    ) -> torch.Tensor:
        """Apply attention with averaged K/V for coordinator's column."""
        ci = self.coordinator_column
        merge_layer = self.model.merge_layers[merge_key]

        if self.config.comm_rank > 0:
            k = merge_layer.k_decompress(k_avg)
            v = merge_layer.v_decompress(v_avg)
        else:
            k = k_avg
            v = v_avg

        n_cross_heads = self.config.n_cross_heads
        head_dim = self.config.d_col // n_cross_heads
        B, T, _ = col_state.shape

        k = k.view(B, T, n_cross_heads, head_dim).transpose(1, 2)
        v = v.view(B, T, n_cross_heads, head_dim).transpose(1, 2)

        q = merge_layer.w_qs[ci](merge_layer.norms_q[ci](col_state))
        q = q.view(B, T, n_cross_heads, head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.config.d_col)

        return col_state + merge_layer.w_os[ci](attn_out)

    async def _gather_worker_kv(
        self, merge_key: str
    ) -> dict[int, tuple[torch.Tensor, torch.Tensor]]:
        """Gather compressed K/V from all connected workers."""
        results = {}
        disconnected = []

        async def recv_from_worker(col_idx, reader):
            try:
                msg_type, k_c, v_c, layer_idx = await asyncio.wait_for(
                    recv_tensor_pair(reader, self.device),
                    timeout=self.merge_timeout,
                )
                assert msg_type == MsgType.COMPRESSED_KV
                results[col_idx] = (k_c, v_c)
            except (asyncio.TimeoutError, ConnectionError, AssertionError) as e:
                logger.warning(f"Worker {col_idx} failed at merge {merge_key}: {e}")
                disconnected.append(col_idx)

        tasks = [
            recv_from_worker(col_idx, reader)
            for col_idx, (reader, writer) in self.workers.items()
        ]
        await asyncio.gather(*tasks)

        # Remove disconnected workers
        for col_idx in disconnected:
            logger.warning(f"Removing disconnected worker {col_idx}")
            _, writer = self.workers.pop(col_idx)
            writer.close()

        return results

    async def _broadcast_averaged_kv(
        self, k_avg: torch.Tensor, v_avg: torch.Tensor, merge_key: str
    ) -> None:
        """Broadcast averaged K/V to all connected workers."""
        disconnected = []

        async def send_to_worker(col_idx, writer):
            try:
                await send_tensor_pair(writer, MsgType.AVERAGED_KV, k_avg, v_avg,
                                       int(merge_key))
            except (ConnectionError, BrokenPipeError) as e:
                logger.warning(f"Failed to send to worker {col_idx}: {e}")
                disconnected.append(col_idx)

        tasks = [
            send_to_worker(col_idx, writer)
            for col_idx, (reader, writer) in self.workers.items()
        ]
        await asyncio.gather(*tasks)

        for col_idx in disconnected:
            self.workers.pop(col_idx, None)

    async def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        """Run distributed forward pass.

        Args:
            input_ids: [B, T] token ids

        Returns:
            logits: [B, T, vocab_size]
        """
        model = self.model
        config = self.config
        ci = self.coordinator_column

        with torch.no_grad():
            # === Shared trunk (coordinator only) ===
            x = model.drop(model.tok_emb(input_ids))
            for block in model.trunk:
                x = block(x, model.trunk_rope)

            # === Project to columns ===
            col_states_all = [proj(x) for proj in model.col_in_projs]

            # === Apply scaling for active columns ===
            n_active = self.n_active
            scale = config.n_columns / n_active

            # Send column states to workers
            signal_tasks = []
            for col_idx, (reader, writer) in self.workers.items():
                signal_tasks.append(send_msg(writer, MsgType.READY))
            await asyncio.gather(*signal_tasks)

            send_tasks = []
            for col_idx, (reader, writer) in self.workers.items():
                scaled_state = col_states_all[col_idx] * scale
                send_tasks.append(send_msg(writer, MsgType.COL_STATE, scaled_state))
            await asyncio.gather(*send_tasks)

            # Coordinator's own column state
            coord_state = col_states_all[ci] * scale

            # === Column layers with merge points ===
            for layer_idx in range(config.n_col_layers):
                # Coordinator runs its own column layer
                coord_state = model.columns[ci][layer_idx](coord_state, model.col_rope)
                # Workers run their layers in parallel (no communication needed here)

                merge_key = str(layer_idx + 1)
                if merge_key in self.merge_keys:
                    # === Merge point: gather, average, broadcast ===
                    t0 = time.time()

                    # Coordinator computes its own compressed K/V
                    coord_k, coord_v = self._compute_coordinator_compressed_kv(
                        coord_state, merge_key
                    )

                    # Gather from workers
                    worker_kvs = await self._gather_worker_kv(merge_key)

                    # Average all K/V (coordinator + workers)
                    n_active_now = 1 + len(worker_kvs)  # coordinator + responding workers
                    k_sum = coord_k.clone()
                    v_sum = coord_v.clone()
                    for col_idx, (k_c, v_c) in worker_kvs.items():
                        k_sum = k_sum + k_c
                        v_sum = v_sum + v_c
                    k_avg = k_sum / n_active_now
                    v_avg = v_sum / n_active_now

                    # Broadcast averaged K/V to workers
                    await self._broadcast_averaged_kv(k_avg, v_avg, merge_key)

                    # Apply merge to coordinator's column
                    coord_state = self._apply_merged_kv(coord_state, k_avg, v_avg, merge_key)

                    dt = time.time() - t0
                    logger.debug(f"Merge {merge_key}: {dt*1000:.1f}ms "
                                 f"({n_active_now} columns)")

            # === Collect final states from workers ===
            final_states = {ci: coord_state}

            disconnected = []
            for col_idx, (reader, writer) in self.workers.items():
                try:
                    msg_type, state, _ = await asyncio.wait_for(
                        recv_msg(reader, self.device),
                        timeout=self.merge_timeout,
                    )
                    assert msg_type == MsgType.FINAL_STATE
                    final_states[col_idx] = state
                except (asyncio.TimeoutError, ConnectionError, AssertionError) as e:
                    logger.warning(f"Failed to get final state from worker {col_idx}: {e}")
                    disconnected.append(col_idx)

            for col_idx in disconnected:
                _, writer = self.workers.pop(col_idx)
                writer.close()

            # === Combine and output ===
            # Build full column state list (zeros for missing columns)
            B, T, d_col = coord_state.shape
            all_states = []
            for i in range(config.n_columns):
                if i in final_states:
                    all_states.append(final_states[i])
                else:
                    all_states.append(torch.zeros(B, T, d_col, device=self.device,
                                                   dtype=coord_state.dtype))

            combined = torch.cat(all_states, dim=-1)  # [B, T, total_col_dim]
            combined = model.out_proj(combined)
            combined = model.norm(combined)
            logits = model.lm_head(combined)

            return logits

    async def shutdown_workers(self) -> None:
        """Send shutdown signal to all workers and close connections."""
        for col_idx, (reader, writer) in self.workers.items():
            try:
                await send_msg(writer, MsgType.SHUTDOWN)
                writer.close()
            except Exception:
                pass
        self.workers.clear()
        if self._server:
            self._server.close()
