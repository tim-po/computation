"""Column-parallel transformer: N independent columns with periodic merge layers."""

import torch
import torch.nn as nn

from .config import ColumnConfig
from .model_dense import TransformerBlock, RMSNorm, precompute_rope
from .merge import LinearMerge


class ColumnTransformer(nn.Module):
    def __init__(self, config: ColumnConfig):
        super().__init__()
        self.config = config

        # Shared token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # Per-column input projections: d_model -> d_col
        self.col_in_projs = nn.ModuleList([
            nn.Linear(config.d_model, config.d_col) for _ in range(config.n_columns)
        ])

        # Per-column transformer blocks
        # columns[i][j] = column i, layer j
        self.columns = nn.ModuleList([
            nn.ModuleList([
                TransformerBlock(config.d_col, config.n_heads, config.d_ff, config.dropout)
                for _ in range(config.n_layers)
            ])
            for _ in range(config.n_columns)
        ])

        # Merge layers (placed after every merge_every layers)
        self.merge_layers = nn.ModuleDict()
        if config.merge_every > 0:
            for layer_idx in range(config.merge_every, config.n_layers + 1, config.merge_every):
                self.merge_layers[str(layer_idx)] = LinearMerge(
                    config.n_columns, config.d_col
                )

        # Output: concat all columns -> project back to d_model -> tied lm_head
        self.out_proj = nn.Linear(config.total_col_dim, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying with shared embedding
        self.lm_head.weight = self.tok_emb.weight

        # RoPE frequencies for column head_dim
        rope_freqs = precompute_rope(config.head_dim, config.max_seq_len)
        self.register_buffer("rope_freqs", rope_freqs, persistent=False)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        # Shared embedding
        emb = self.drop(self.tok_emb(input_ids))  # [B, T, d_model]

        # Project to each column's dimension
        col_states = [proj(emb) for proj in self.col_in_projs]  # list of [B, T, d_col]

        # Run layers with periodic merging
        for layer_idx in range(self.config.n_layers):
            # Each column processes independently
            for col_idx in range(self.config.n_columns):
                col_states[col_idx] = self.columns[col_idx][layer_idx](
                    col_states[col_idx], self.rope_freqs
                )

            # Merge after this layer?
            merge_key = str(layer_idx + 1)
            if merge_key in self.merge_layers:
                col_states = self.merge_layers[merge_key](col_states)

        # Concatenate all columns -> project to d_model -> vocab
        combined = torch.cat(col_states, dim=-1)  # [B, T, total_col_dim]
        combined = self.out_proj(combined)         # [B, T, d_model]
        combined = self.norm(combined)
        return self.lm_head(combined)
