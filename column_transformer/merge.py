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

    Each column's tokens serve as queries, and ALL columns' tokens serve as
    keys/values. This allows rich information flow between columns while
    maintaining per-column identity via residual connections.

    Uses n_cross_heads attention heads over the concatenated column space.
    """

    def __init__(self, n_columns: int, d_col: int, n_cross_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_columns = n_columns
        self.d_col = d_col
        self.n_cross_heads = n_cross_heads
        self.head_dim = d_col // n_cross_heads
        assert d_col % n_cross_heads == 0, "d_col must be divisible by n_cross_heads"

        total_dim = n_columns * d_col

        # Shared K/V projections over concatenated columns
        self.norm_kv = nn.LayerNorm(total_dim)
        self.w_k = nn.Linear(total_dim, d_col, bias=False)  # project to d_col
        self.w_v = nn.Linear(total_dim, d_col, bias=False)

        # Per-column Q projections and output projections
        self.norms_q = nn.ModuleList([nn.LayerNorm(d_col) for _ in range(n_columns)])
        self.w_qs = nn.ModuleList([nn.Linear(d_col, d_col, bias=False) for _ in range(n_columns)])
        self.w_os = nn.ModuleList([nn.Linear(d_col, d_col, bias=False) for _ in range(n_columns)])

        self.attn_dropout = nn.Dropout(dropout)
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, columns: list[torch.Tensor]) -> list[torch.Tensor]:
        # columns: list of n_columns x [B, T, d_col]
        B, T, _ = columns[0].shape

        # Shared keys and values from ALL columns
        concat = torch.cat(columns, dim=-1)  # [B, T, n_columns * d_col]
        kv_input = self.norm_kv(concat)
        k = self.w_k(kv_input).view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)
        v = self.w_v(kv_input).view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)

        # Each column queries the shared K/V
        outputs = []
        for i in range(self.n_columns):
            q = self.w_qs[i](self.norms_q[i](columns[i]))
            q = q.view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)

            attn_out = F.scaled_dot_product_attention(
                q, k, v, is_causal=True,
                dropout_p=self.attn_dropout.p if self.training else 0.0,
            )
            attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, self.d_col)
            # Residual connection: column + cross-attention output
            outputs.append(columns[i] + self.resid_dropout(self.w_os[i](attn_out)))

        return outputs
