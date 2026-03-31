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


@dataclass
class ColumnConfigV2:
    """V2 column config: shared trunk layers + cross-column attention merges.
    Supports column dropout for fault-tolerant distributed inference."""
    vocab_size: int = 50257
    d_model: int = 512          # shared embedding & trunk dim
    n_trunk_layers: int = 3     # shared trunk layers (full d_model width)
    trunk_n_heads: int = 8      # attention heads in trunk
    trunk_d_ff: int = 2048      # FFN dim in trunk
    n_columns: int = 4          # columns after trunk
    d_col: int = 256            # per-column hidden dim
    n_col_layers: int = 5       # column layers after trunk (total = trunk + col = 8)
    n_heads: int = 4            # heads per column
    d_ff: int = 1024            # FFN dim per column
    n_cross_heads: int = 4      # heads in cross-column attention
    max_seq_len: int = 512
    merge_every: int = 2        # cross-attn merge frequency within column layers
    dropout: float = 0.1
    col_drop_prob: float = 0.0  # column dropout probability (0 = disabled)
    min_active_columns: int = 1 # minimum columns kept alive during dropout
    comm_rank: int = 0          # compressed cross-attn rank (0 = full, no compression)
    quant_comm: bool = False    # simulate int8 quantization on compressed K/V

    @property
    def head_dim(self) -> int:
        return self.d_col // self.n_heads

    @property
    def total_col_dim(self) -> int:
        return self.n_columns * self.d_col

    @property
    def total_layers(self) -> int:
        return self.n_trunk_layers + self.n_col_layers

    def param_estimate(self) -> int:
        emb = self.vocab_size * self.d_model
        # Trunk params
        trunk_attn = 4 * self.d_model * self.d_model * self.n_trunk_layers
        trunk_ffn = 3 * self.d_model * self.trunk_d_ff * self.n_trunk_layers
        # Column projections
        proj = self.n_columns * self.d_model * self.d_col  # in only (out is concat->proj)
        out_proj = self.total_col_dim * self.d_model
        # Column params
        col_attn = 4 * self.d_col * self.d_col * self.n_col_layers * self.n_columns
        col_ffn = 3 * self.d_col * self.d_ff * self.n_col_layers * self.n_columns
        # Cross-column attention merges
        n_merges = self.n_col_layers // self.merge_every if self.merge_every > 0 else 0
        # Each merge: shared K,V projections + per-column Q,O projections
        merge_shared = n_merges * 2 * self.total_col_dim * self.d_col  # K, V
        merge_percol = n_merges * self.n_columns * 2 * self.d_col * self.d_col  # Q, O per col
        return emb + trunk_attn + trunk_ffn + proj + out_proj + col_attn + col_ffn + merge_shared + merge_percol


# Preset experiment configs
EXPERIMENTS = {
    "dense": DenseConfig(),
    "column_merge_2": ColumnConfig(merge_every=2),
    "column_merge_4": ColumnConfig(merge_every=4),
    "column_no_merge": ColumnConfig(merge_every=0),
    # V2: shared trunk + cross-column attention
    "v2_trunk3_xattn2": ColumnConfigV2(
        n_trunk_layers=3, n_col_layers=5, merge_every=2,
    ),
    "v2_trunk2_xattn2": ColumnConfigV2(
        n_trunk_layers=2, n_col_layers=6, merge_every=2,
    ),
    "v2_trunk3_xattn1": ColumnConfigV2(
        n_trunk_layers=3, n_col_layers=5, merge_every=1,
    ),
    # V3: column dropout for fault-tolerant distributed inference (small scale)
    "v3_drop20": ColumnConfigV2(
        n_trunk_layers=3, n_col_layers=5, merge_every=2,
        col_drop_prob=0.2, min_active_columns=2,
    ),
    "v3_drop40": ColumnConfigV2(
        n_trunk_layers=3, n_col_layers=5, merge_every=2,
        col_drop_prob=0.4, min_active_columns=1,
    ),
    # =====================================================================
    # H100 SCALE: 8 columns, ~350M params
    # Target: train on H100, inference on consumer GPUs (1-8 columns)
    # =====================================================================
    "h100_dense": DenseConfig(
        d_model=1024, n_layers=16, n_heads=16, d_ff=4096,
        max_seq_len=1024, dropout=0.1,
    ),
    "h100_col8_drop0": ColumnConfigV2(
        d_model=1024,
        n_trunk_layers=4, trunk_n_heads=16, trunk_d_ff=4096,
        n_columns=8, d_col=256, n_col_layers=12,
        n_heads=4, d_ff=1024, n_cross_heads=4,
        max_seq_len=1024, merge_every=3,
        col_drop_prob=0.0, min_active_columns=8,
    ),
    "h100_col8_drop25": ColumnConfigV2(
        d_model=1024,
        n_trunk_layers=4, trunk_n_heads=16, trunk_d_ff=4096,
        n_columns=8, d_col=256, n_col_layers=12,
        n_heads=4, d_ff=1024, n_cross_heads=4,
        max_seq_len=1024, merge_every=3,
        col_drop_prob=0.25, min_active_columns=2,
    ),
    "h100_col8_drop50": ColumnConfigV2(
        d_model=1024,
        n_trunk_layers=4, trunk_n_heads=16, trunk_d_ff=4096,
        n_columns=8, d_col=256, n_col_layers=12,
        n_heads=4, d_ff=1024, n_cross_heads=4,
        max_seq_len=1024, merge_every=3,
        col_drop_prob=0.5, min_active_columns=1,
    ),
    # =====================================================================
    # H100 SCALE: 350M with compressed cross-attention (bandwidth reduction)
    # =====================================================================
    "h100_col8_comp64": ColumnConfigV2(
        d_model=1024,
        n_trunk_layers=4, trunk_n_heads=16, trunk_d_ff=4096,
        n_columns=8, d_col=256, n_col_layers=12,
        n_heads=4, d_ff=1024, n_cross_heads=4,
        max_seq_len=1024, merge_every=3,
        col_drop_prob=0.25, min_active_columns=2,
        comm_rank=64,
    ),
    "h100_col8_comp32": ColumnConfigV2(
        d_model=1024,
        n_trunk_layers=4, trunk_n_heads=16, trunk_d_ff=4096,
        n_columns=8, d_col=256, n_col_layers=12,
        n_heads=4, d_ff=1024, n_cross_heads=4,
        max_seq_len=1024, merge_every=3,
        col_drop_prob=0.25, min_active_columns=2,
        comm_rank=32,
    ),
    "h100_col8_comp64_q8": ColumnConfigV2(
        d_model=1024,
        n_trunk_layers=4, trunk_n_heads=16, trunk_d_ff=4096,
        n_columns=8, d_col=256, n_col_layers=12,
        n_heads=4, d_ff=1024, n_cross_heads=4,
        max_seq_len=1024, merge_every=3,
        col_drop_prob=0.25, min_active_columns=2,
        comm_rank=64, quant_comm=True,
    ),
    # =====================================================================
    # H100 SCALE: 1B params — validation before multi-GPU training
    # Dataset: FineWeb-Edu (streaming) for diversity
    # =====================================================================
    "h100_1b_dense": DenseConfig(
        d_model=2048, n_layers=22, n_heads=32, d_ff=4096,
        max_seq_len=1024, dropout=0.1,
    ),
    "h100_1b_col8": ColumnConfigV2(
        d_model=2048,
        n_trunk_layers=6, trunk_n_heads=32, trunk_d_ff=8192,
        n_columns=8, d_col=512, n_col_layers=16,
        n_heads=8, d_ff=2048, n_cross_heads=8,
        max_seq_len=1024, merge_every=4,
        col_drop_prob=0.0, min_active_columns=8,
    ),
    "h100_1b_col8_drop25": ColumnConfigV2(
        d_model=2048,
        n_trunk_layers=6, trunk_n_heads=32, trunk_d_ff=8192,
        n_columns=8, d_col=512, n_col_layers=16,
        n_heads=8, d_ff=2048, n_cross_heads=8,
        max_seq_len=1024, merge_every=4,
        col_drop_prob=0.25, min_active_columns=2,
    ),
    "h100_1b_col8_comp64": ColumnConfigV2(
        d_model=2048,
        n_trunk_layers=6, trunk_n_heads=32, trunk_d_ff=8192,
        n_columns=8, d_col=512, n_col_layers=16,
        n_heads=8, d_ff=2048, n_cross_heads=8,
        max_seq_len=1024, merge_every=4,
        col_drop_prob=0.25, min_active_columns=2,
        comm_rank=64,
    ),
    "h100_1b_col8_comp64_q8": ColumnConfigV2(
        d_model=2048,
        n_trunk_layers=6, trunk_n_heads=32, trunk_d_ff=8192,
        n_columns=8, d_col=512, n_col_layers=16,
        n_heads=8, d_ff=2048, n_cross_heads=8,
        max_seq_len=1024, merge_every=4,
        col_drop_prob=0.25, min_active_columns=2,
        comm_rank=64, quant_comm=True,
    ),
}
