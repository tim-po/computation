from dataclasses import dataclass


@dataclass
class DenseConfig:
    vocab_size: int = 50257
    d_model: int = 512
    n_layers: int = 8
    n_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 512
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        return self.d_model // self.n_heads

    def param_estimate(self) -> int:
        emb = self.vocab_size * self.d_model
        attn = 4 * self.d_model * self.d_model * self.n_layers
        ffn = 3 * self.d_model * self.d_ff * self.n_layers  # SwiGLU has 3 matrices
        return emb + attn + ffn


@dataclass
class ColumnConfig:
    vocab_size: int = 50257
    d_model: int = 512          # shared embedding dim
    n_columns: int = 4
    d_col: int = 256            # per-column hidden dim
    n_layers: int = 8           # layers per column
    n_heads: int = 4            # heads per column
    d_ff: int = 1024            # FFN dim per column
    max_seq_len: int = 512
    merge_every: int = 4        # 0 = no merge (ensemble baseline)
    dropout: float = 0.1

    @property
    def head_dim(self) -> int:
        return self.d_col // self.n_heads

    @property
    def total_col_dim(self) -> int:
        return self.n_columns * self.d_col

    def param_estimate(self) -> int:
        emb = self.vocab_size * self.d_model
        proj = self.n_columns * self.d_model * self.d_col * 2  # in + out
        col_attn = 4 * self.d_col * self.d_col * self.n_layers * self.n_columns
        col_ffn = 3 * self.d_col * self.d_ff * self.n_layers * self.n_columns
        n_merges = self.n_layers // self.merge_every if self.merge_every > 0 else 0
        merge = n_merges * self.total_col_dim * self.total_col_dim
        out_head = self.total_col_dim * self.vocab_size
        return emb + proj + col_attn + col_ffn + merge + out_head


# Preset experiment configs
EXPERIMENTS = {
    "dense": DenseConfig(),
    "column_merge_2": ColumnConfig(merge_every=2),
    "column_merge_4": ColumnConfig(merge_every=4),
    "column_no_merge": ColumnConfig(merge_every=0),
}
