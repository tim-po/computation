"""Column-parallel transformer v2: shared trunk + cross-column attention merges.

Supports column dropout for fault-tolerant distributed inference.
When col_drop_prob > 0, randomly drops entire columns during training so the
model learns to produce good outputs from any subset of columns.
"""

import torch
import torch.nn as nn

from .config import ColumnConfigV2
from .model_dense import TransformerBlock, RMSNorm, precompute_rope
from .merge import CrossColumnAttention, ColumnDropout


class ColumnTransformerV2(nn.Module):
    """Column transformer with:
    1. Shared trunk layers before splitting into columns
    2. Cross-column attention at merge points (per-column K/V, supports masking)
    3. Optional column dropout for fault-tolerant inference
    """

    def __init__(self, config: ColumnConfigV2):
        super().__init__()
        self.config = config

        # Shared token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # === Shared trunk: full d_model transformer blocks ===
        self.trunk = nn.ModuleList([
            TransformerBlock(config.d_model, config.trunk_n_heads, config.trunk_d_ff, config.dropout)
            for _ in range(config.n_trunk_layers)
        ])

        # RoPE for trunk (uses trunk head_dim)
        trunk_head_dim = config.d_model // config.trunk_n_heads
        trunk_rope = precompute_rope(trunk_head_dim, config.max_seq_len)
        self.register_buffer("trunk_rope", trunk_rope, persistent=False)

        # === Column split: project from d_model to per-column d_col ===
        self.col_in_projs = nn.ModuleList([
            nn.Linear(config.d_model, config.d_col) for _ in range(config.n_columns)
        ])

        # === Column dropout (only active when col_drop_prob > 0) ===
        self.col_dropout = ColumnDropout(
            n_columns=config.n_columns,
            drop_prob=config.col_drop_prob,
            min_active=config.min_active_columns,
        )

        # === Per-column transformer blocks ===
        self.columns = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(config.d_col, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_col_layers)
            ])
            for _ in range(config.n_columns)
        ])

        # RoPE for columns (uses column head_dim)
        col_rope = precompute_rope(config.head_dim, config.max_seq_len)
        self.register_buffer("col_rope", col_rope, persistent=False)

        # === Cross-column attention at merge points ===
        self.merge_layers = nn.ModuleDict()
        if config.merge_every > 0:
            for layer_idx in range(config.merge_every, config.n_col_layers + 1, config.merge_every):
                self.merge_layers[str(layer_idx)] = CrossColumnAttention(
                    config.n_columns, config.d_col,
                    n_cross_heads=config.n_cross_heads,
                    dropout=config.dropout,
                )

        # === Output: concat all columns -> project to d_model -> LM head ===
        self.out_proj = nn.Linear(config.total_col_dim, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight  # weight tying

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        active_columns: list[int] | None = None,
    ) -> torch.Tensor:
        """
        Args:
            input_ids: [B, T] token ids
            active_columns: optional list of column indices to use at inference.
                          If None, uses all columns (training) or all columns (eval).
                          Use this to simulate distributed inference with N < n_columns GPUs.
        """
        # === Shared trunk ===
        x = self.drop(self.tok_emb(input_ids))  # [B, T, d_model]
        for block in self.trunk:
            x = block(x, self.trunk_rope)

        # === Split into columns ===
        col_states = [proj(x) for proj in self.col_in_projs]  # list of [B, T, d_col]

        # === Apply column dropout (training) or manual column selection (inference) ===
        if active_columns is not None:
            # Manual column selection for distributed inference simulation
            col_mask = torch.zeros(self.config.n_columns, dtype=torch.bool, device=x.device)
            for idx in active_columns:
                col_mask[idx] = True
            n_active = len(active_columns)
            scale = self.config.n_columns / n_active
            col_states = [
                col * scale if col_mask[i] else torch.zeros_like(col)
                for i, col in enumerate(col_states)
            ]
        else:
            # Training: random column dropout; eval: all columns
            col_states, col_mask = self.col_dropout(col_states)

        # === Column layers with cross-column attention merges ===
        for layer_idx in range(self.config.n_col_layers):
            # Only compute active columns
            for col_idx in range(self.config.n_columns):
                if col_mask[col_idx]:
                    col_states[col_idx] = self.columns[col_idx][layer_idx](
                        col_states[col_idx], self.col_rope
                    )

            merge_key = str(layer_idx + 1)
            if merge_key in self.merge_layers:
                col_states = self.merge_layers[merge_key](col_states, col_mask)

        # === Combine and output ===
        combined = torch.cat(col_states, dim=-1)  # [B, T, total_col_dim]
        combined = self.out_proj(combined)         # [B, T, d_model]
        combined = self.norm(combined)
        return self.lm_head(combined)
