"""Merge mechanisms for combining column outputs."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LinearMerge(nn.Module):
    """Simple linear merge: concat columns -> linear -> GELU -> split back."""

    def __init__(self, n_columns: int, d_col: int):
        super().__init__()
        total_dim = n_columns * d_col
        self.n_columns = n_columns
        self.d_col = d_col
        self.norm = nn.LayerNorm(total_dim)
        self.linear = nn.Linear(total_dim, total_dim)

    def forward(self, columns: list[torch.Tensor]) -> list[torch.Tensor]:
        # columns: list of [batch, seq, d_col]
        merged = torch.cat(columns, dim=-1)  # [batch, seq, n_columns * d_col]
        merged = self.linear(F.gelu(self.norm(merged)))
        return list(merged.chunk(self.n_columns, dim=-1))


class LowRankMerge(nn.Module):
    """Low-rank merge: concat -> project down -> project up -> split.

    Simulates limited-bandwidth communication between columns.
    The rank parameter controls the communication bottleneck.
    """

    def __init__(self, n_columns: int, d_col: int, rank: int = 64):
        super().__init__()
        total_dim = n_columns * d_col
        self.n_columns = n_columns
        self.d_col = d_col
        self.norm = nn.LayerNorm(total_dim)
        self.down = nn.Linear(total_dim, rank, bias=False)
        self.up = nn.Linear(rank, total_dim, bias=False)

    def forward(self, columns: list[torch.Tensor]) -> list[torch.Tensor]:
        merged = torch.cat(columns, dim=-1)
        compressed = F.gelu(self.down(self.norm(merged)))
        expanded = self.up(compressed)
        return list(expanded.chunk(self.n_columns, dim=-1))


class CrossColumnAttention(nn.Module):
    """Cross-column attention merge: columns attend to each other's representations.

    Each column has its own K/V projection. K/V from active columns are averaged
    to form shared keys/values. This design supports column dropout: any subset
    of columns can participate without dimension mismatches.

    At inference on distributed hardware, each GPU computes its column's K/V
    locally, then they aggregate — matching this architecture exactly.
    """

    def __init__(self, n_columns: int, d_col: int, n_cross_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_columns = n_columns
        self.d_col = d_col
        self.n_cross_heads = n_cross_heads
        self.head_dim = d_col // n_cross_heads
        assert d_col % n_cross_heads == 0, "d_col must be divisible by n_cross_heads"

        # Per-column K/V projections (supports any column subset)
        self.norms_kv = nn.ModuleList([nn.LayerNorm(d_col) for _ in range(n_columns)])
        self.w_ks = nn.ModuleList([nn.Linear(d_col, d_col, bias=False) for _ in range(n_columns)])
        self.w_vs = nn.ModuleList([nn.Linear(d_col, d_col, bias=False) for _ in range(n_columns)])

        # Per-column Q and output projections
        self.norms_q = nn.ModuleList([nn.LayerNorm(d_col) for _ in range(n_columns)])
        self.w_qs = nn.ModuleList([nn.Linear(d_col, d_col, bias=False) for _ in range(n_columns)])
        self.w_os = nn.ModuleList([nn.Linear(d_col, d_col, bias=False) for _ in range(n_columns)])

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self, columns: list[torch.Tensor], col_mask: torch.Tensor | None = None
    ) -> list[torch.Tensor]:
        """
        Args:
            columns: list of n_columns x [B, T, d_col]
            col_mask: optional bool tensor [n_columns]. True = active, False = dropped.
                      If None, all columns are active.
        """
        B, T, _ = columns[0].shape

        if col_mask is None:
            col_mask = torch.ones(self.n_columns, dtype=torch.bool, device=columns[0].device)

        active_indices = col_mask.nonzero(as_tuple=True)[0]
        n_active = len(active_indices)

        # Aggregate K/V from active columns (average)
        k_sum = torch.zeros(B, T, self.d_col, device=columns[0].device, dtype=columns[0].dtype)
        v_sum = torch.zeros_like(k_sum)
        for i in active_indices:
            normed = self.norms_kv[i](columns[i])
            k_sum = k_sum + self.w_ks[i](normed)
            v_sum = v_sum + self.w_vs[i](normed)
        k = (k_sum / n_active).view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)
        v = (v_sum / n_active).view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)

        # Each active column queries the aggregated K/V
        outputs = []
        for i in range(self.n_columns):
            if not col_mask[i]:
                # Dropped column: pass through unchanged
                outputs.append(columns[i])
                continue

            q = self.w_qs[i](self.norms_q[i](columns[i]))
            q = q.view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)

            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_col)
            outputs.append(columns[i] + self.resid_dropout(self.w_os[i](attn_out)))

        return outputs


class ColumnDropout(nn.Module):
    """Drops entire columns during training to build fault tolerance.

    During training: randomly masks columns, scales survivors by N/n_active
    (inverted dropout). Guarantees at least min_active columns survive.

    During eval/inference: all columns active, no scaling.

    The same mask is returned to be used consistently throughout the forward pass.
    """

    def __init__(self, n_columns: int, drop_prob: float = 0.2, min_active: int = 1):
        super().__init__()
        self.n_columns = n_columns
        self.drop_prob = drop_prob
        self.min_active = min(min_active, n_columns)

    def forward(
        self, columns: list[torch.Tensor]
    ) -> tuple[list[torch.Tensor], torch.Tensor]:
        device = columns[0].device

        if not self.training or self.drop_prob == 0.0:
            mask = torch.ones(self.n_columns, dtype=torch.bool, device=device)
            return columns, mask

        # Sample which columns to keep
        mask = torch.rand(self.n_columns, device=device) >= self.drop_prob

        # Guarantee minimum active columns
        if mask.sum() < self.min_active:
            inactive = (~mask).nonzero(as_tuple=True)[0]
            perm = inactive[torch.randperm(len(inactive), device=device)]
            need = self.min_active - mask.sum().item()
            mask[perm[:need]] = True

        n_active = mask.sum().float()
        scale = self.n_columns / n_active

        # Scale active columns, zero dropped ones
        result = []
        for i, col in enumerate(columns):
            if mask[i]:
                result.append(col * scale)
            else:
                result.append(torch.zeros_like(col))

        return result, mask
