"""
TP-Surgical simulator: воспроизводит условия 2-узлового деплоя на одном GPU.

Принцип:
  1. Модель запускается полностью на одном GPU (единственный доступный).
  2. Хуки зануляют col_1 головы у каждого full_attention слоя —
     симуляция того, что Worker 0 видит только col_0.
  3. На каждой CCA-точке хук вставляет time.sleep(RTT) —
     симуляция AllReduce/CCA-sync между двумя узлами.
  4. Пропускная способность учитывается как transfer_time = data_bytes / bw.

На выходе: реальный tok/s (compute + simulated_comm), breakdown по компонентам.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────────────────────────────────────
# Конфиг сети
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class NetworkConfig:
    topology: str           # lan | wan | nvlink | custom
    latency_one_way_ms: float
    bandwidth_gbps: float

    @property
    def rtt_ms(self) -> float:
        return self.latency_one_way_ms * 2

    def transfer_ms(self, n_bytes: int) -> float:
        """Время передачи данных в мс (без латентности)."""
        bw_bytes_per_ms = self.bandwidth_gbps * 1e9 / 8 / 1000
        return n_bytes / bw_bytes_per_ms

    def sync_ms(self, n_bytes: int) -> float:
        """Полная стоимость одного CCA sync: RTT + transfer."""
        return self.rtt_ms + self.transfer_ms(n_bytes)


# ─────────────────────────────────────────────────────────────────────────────
# Сборщик статистики
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepStats:
    compute_ms: float = 0.0
    comm_ms: float = 0.0
    cca_hits: int = 0
    bytes_transferred: int = 0

    @property
    def total_ms(self) -> float:
        return self.compute_ms + self.comm_ms

    @property
    def comm_ratio(self) -> float:
        if self.total_ms == 0:
            return 0.0
        return self.comm_ms / self.total_ms


# ─────────────────────────────────────────────────────────────────────────────
# Simulator
# ─────────────────────────────────────────────────────────────────────────────

class TPSurgicalSim:
    """
    Контекст-менеджер: устанавливает хуки на модель для симуляции TP-Surgical.

    Для каждого full_attention слоя: нулит col_1 головы (o_proj input).
    На каждой CCA-точке: спит RTT + transfer, собирает статистику.
    """

    def __init__(
        self,
        model: nn.Module,
        net: NetworkConfig,
        full_attn_layers: list[int],
        cca_layers: list[int],
        n_gpu: int = 2,
        device: str = "cpu",
    ):
        self.model = model
        self.net = net
        self.full_attn_layers = set(full_attn_layers)
        self.cca_layers = set(cca_layers)
        self.n_gpu = n_gpu
        self.device = device
        self._hooks: list = []
        self.stats: Optional[StepStats] = None

    # ── hook logic ────────────────────────────────────────────────────────────

    def _make_mask_hook(self, col_size: int):
        def hook(module, args):
            inp = args[0]
            m = torch.ones_like(inp)
            m[..., col_size:] = 0.0
            return (inp * m,)
        return hook

    def _make_cca_hook(self, layer_idx: int):
        net = self.net
        stats_ref = self

        def hook(module, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            n_bytes = h.numel() * h.element_size() * 2  # × 2: bidirectional
            comm_ms = net.sync_ms(n_bytes)

            t0 = time.perf_counter()
            time.sleep(comm_ms / 1000)
            actual_ms = (time.perf_counter() - t0) * 1000

            if stats_ref.stats is not None:
                stats_ref.stats.comm_ms += actual_ms
                stats_ref.stats.cca_hits += 1
                stats_ref.stats.bytes_transferred += n_bytes

            return out
        return hook

    # ── context manager ───────────────────────────────────────────────────────

    def _install(self):
        cfg = self.model.config
        n_heads = cfg.num_attention_heads
        head_dim = cfg.head_dim
        col_size = (n_heads // self.n_gpu) * head_dim

        for l_idx, layer in enumerate(self.model.model.layers):
            if l_idx in self.full_attn_layers:
                if hasattr(layer, "self_attn") and hasattr(layer.self_attn, "o_proj"):
                    self._hooks.append(
                        layer.self_attn.o_proj.register_forward_pre_hook(
                            self._make_mask_hook(col_size)
                        )
                    )
            if l_idx in self.cca_layers:
                self._hooks.append(
                    layer.register_forward_hook(self._make_cca_hook(l_idx))
                )

    def _remove(self):
        for h in self._hooks:
            h.remove()
        self._hooks.clear()

    def __enter__(self):
        self.stats = StepStats()
        self._install()
        return self

    def __exit__(self, *_):
        self._remove()

    # ── single decode step ────────────────────────────────────────────────────

    @torch.no_grad()
    def decode_step(self, model, input_ids: torch.Tensor) -> tuple[torch.Tensor, StepStats]:
        """Один шаг декодирования с измерением compute и comm."""
        self.stats = StepStats()

        # Измеряем только compute (без comm sleep)
        t_start = time.perf_counter()
        self._install()
        logits = model(input_ids=input_ids).logits
        if self.device == "cuda":
            torch.cuda.synchronize()
        elapsed_ms = (time.perf_counter() - t_start) * 1000
        self._remove()

        # compute = total - comm
        self.stats.compute_ms = elapsed_ms - self.stats.comm_ms
        return logits, self.stats
