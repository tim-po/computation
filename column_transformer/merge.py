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
