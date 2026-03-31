"""Column-parallel transformer v2 (FAST): vectorized column operations.

Functionally identical to model_column_v2.py but replaces Python for-loops
over columns with batched tensor operations (torch.bmm). This eliminates
hundreds of small sequential GPU kernel launches, improving utilization
from ~17% to ~60-80% on H100.

Key idea: instead of N separate nn.Linear modules (one per column), store
all column weights in a single [N, d_out, d_in] tensor and use torch.bmm
to process all columns in one kernel launch.

Supports loading checkpoints from the original (slow) model via
convert_checkpoint_to_fast().
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ColumnConfigV2
from .model_dense import RMSNorm, precompute_rope, apply_rope
from .merge import ColumnDropout


# ============================================================================
# Batched linear: replaces N separate nn.Linear with one bmm
# ============================================================================

class BatchedLinear(nn.Module):
    """N parallel linear layers computed as a single batched matmul.

    Replaces: [nn.Linear(d_in, d_out) for _ in range(N)]
    With:     one torch.bmm call

    Weight shape: [N, d_out, d_in]  (transposed for bmm)
    Input shape:  [N, B*T, d_in]
    Output shape: [N, B*T, d_out]
    """

    def __init__(self, n_parallel: int, d_in: int, d_out: int, bias: bool = False):
        super().__init__()
        self.n_parallel = n_parallel
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.randn(n_parallel, d_out, d_in) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(n_parallel, 1, d_out))
        else:
            self.bias = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, *, d_in] -> [N, *, d_out]
        out = torch.bmm(x, self.weight.transpose(-1, -2))
        if self.bias is not None:
            out = out + self.bias
        return out


class BatchedRMSNorm(nn.Module):
    """N parallel RMSNorms with separate weight vectors.

    Weight shape: [N, 1, 1, dim]
    Input shape:  [N, B, T, dim]
    """

    def __init__(self, n_parallel: int, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_parallel, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, B, T, dim]
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).type_as(x) * self.weight


class BatchedLayerNorm(nn.Module):
    """N parallel LayerNorms with separate weight and bias vectors.

    Used for cross-column attention norms (the original uses nn.LayerNorm).
    Weight shape: [N, 1, 1, dim]
    Bias shape:   [N, 1, 1, dim]
    Input shape:  [N, B, T, dim]
    """

    def __init__(self, n_parallel: int, dim: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(n_parallel, 1, 1, dim))
        self.bias = nn.Parameter(torch.zeros(n_parallel, 1, 1, dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, B, T, dim]
        mean = x.float().mean(-1, keepdim=True)
        var = x.float().var(-1, keepdim=True, unbiased=False)
        normed = (x.float() - mean) / (var + self.eps).sqrt()
        return (normed * self.weight + self.bias).type_as(x)


# ============================================================================
# Batched attention: all columns' Q/K/V/O in single bmm calls
# ============================================================================

class BatchedColumnAttention(nn.Module):
    """Self-attention for all N columns in parallel using batched matmuls."""

    def __init__(self, n_columns: int, d_col: int, n_heads: int, dropout: float = 0.0):
        super().__init__()
        self.n_columns = n_columns
        self.d_col = d_col
        self.n_heads = n_heads
        self.head_dim = d_col // n_heads

        self.w_q = BatchedLinear(n_columns, d_col, d_col)
        self.w_k = BatchedLinear(n_columns, d_col, d_col)
        self.w_v = BatchedLinear(n_columns, d_col, d_col)
        self.w_o = BatchedLinear(n_columns, d_col, d_col)

        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, B, T, d_col] — all columns stacked
            rope_freqs: precomputed RoPE frequencies
        Returns: [N, B, T, d_col]
        """
        N, B, T, _ = x.shape

        # Reshape for bmm: [N, B*T, d_col]
        x_flat = x.reshape(N, B * T, self.d_col)

        q = self.w_q(x_flat).view(N, B, T, self.n_heads, self.head_dim)
        k = self.w_k(x_flat).view(N, B, T, self.n_heads, self.head_dim)
        v = self.w_v(x_flat).view(N, B, T, self.n_heads, self.head_dim)

        # Merge N and B for attention: [N*B, n_heads, T, head_dim]
        q = q.reshape(N * B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.reshape(N * B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.reshape(N * B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Apply RoPE
        q = apply_rope(q, rope_freqs)
        k = apply_rope(k, rope_freqs)

        # Attention (N*B batch — all columns + batch in one SDPA call)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )

        # Reshape back: [N*B, n_heads, T, head_dim] -> [N, B*T, d_col]
        out = out.transpose(1, 2).contiguous().view(N, B * T, self.d_col)
        out = self.w_o(out)
        return self.resid_dropout(out.view(N, B, T, self.d_col))


class BatchedSwiGLU(nn.Module):
    """SwiGLU FFN for all N columns in parallel using batched matmuls."""

    def __init__(self, n_columns: int, d_col: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.n_columns = n_columns
        self.w_gate = BatchedLinear(n_columns, d_col, d_ff)
        self.w_up = BatchedLinear(n_columns, d_col, d_ff)
        self.w_down = BatchedLinear(n_columns, d_ff, d_col)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [N, B, T, d_col]
        N, B, T, d = x.shape
        x_flat = x.reshape(N, B * T, d)
        out = self.w_down(F.silu(self.w_gate(x_flat)) * self.w_up(x_flat))
        return self.dropout(out.view(N, B, T, -1))


class BatchedColumnBlock(nn.Module):
    """One transformer block for all columns: batched attention + batched FFN."""

    def __init__(self, n_columns: int, d_col: int, n_heads: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.norm1 = BatchedRMSNorm(n_columns, d_col)
        self.attn = BatchedColumnAttention(n_columns, d_col, n_heads, dropout)
        self.norm2 = BatchedRMSNorm(n_columns, d_col)
        self.ffn = BatchedSwiGLU(n_columns, d_col, d_ff, dropout)

    def forward(self, x: torch.Tensor, rope_freqs: torch.Tensor) -> torch.Tensor:
        # x: [N, B, T, d_col]
        x = x + self.attn(self.norm1(x), rope_freqs)
        x = x + self.ffn(self.norm2(x))
        return x


# ============================================================================
# Batched cross-column attention (merge layers)
# ============================================================================

class BatchedCrossColumnAttention(nn.Module):
    """Cross-column attention with batched K/V/Q/O projections.

    Still needs to aggregate K/V across columns (the "communication" step),
    but all per-column projections are now batched.
    """

    def __init__(self, n_columns: int, d_col: int, n_cross_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_columns = n_columns
        self.d_col = d_col
        self.n_cross_heads = n_cross_heads
        self.head_dim = d_col // n_cross_heads

        # Batched projections: [N, d_col, d_col] each
        self.norm_kv = BatchedLayerNorm(n_columns, d_col)
        self.w_k = BatchedLinear(n_columns, d_col, d_col)
        self.w_v = BatchedLinear(n_columns, d_col, d_col)

        self.norm_q = BatchedLayerNorm(n_columns, d_col)
        self.w_q = BatchedLinear(n_columns, d_col, d_col)
        self.w_o = BatchedLinear(n_columns, d_col, d_col)

        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(
        self, col_states: torch.Tensor, col_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        """
        Args:
            col_states: [N, B, T, d_col]
            col_mask: [N] bool tensor, True = active
        Returns: [N, B, T, d_col]
        """
        N, B, T, _ = col_states.shape

        if col_mask is None:
            col_mask = torch.ones(N, dtype=torch.bool, device=col_states.device)

        mask_f = col_mask.float().view(N, 1, 1, 1)  # [N, 1, 1, 1] for broadcasting
        n_active = col_mask.float().sum().clamp(min=1.0)

        # Batched K/V projection: [N, B, T, d_col] -> [N, B*T, d_col]
        normed_kv = self.norm_kv(col_states)
        kv_flat = normed_kv.reshape(N, B * T, self.d_col)
        k_all = self.w_k(kv_flat).view(N, B, T, self.d_col) * mask_f  # [N, B, T, d_col]
        v_all = self.w_v(kv_flat).view(N, B, T, self.d_col) * mask_f

        # Aggregate K/V across columns (this is the "communication" step)
        k = k_all.sum(dim=0) / n_active  # [B, T, d_col]
        v = v_all.sum(dim=0) / n_active
        k = k.view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)

        # Batched Q projection
        normed_q = self.norm_q(col_states)
        q_flat = normed_q.reshape(N, B * T, self.d_col)
        q_all = self.w_q(q_flat).view(N, B, T, self.n_cross_heads, self.head_dim)

        # Expand shared K/V for all columns: [N*B, n_heads, T, head_dim]
        k_expanded = k.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N * B, self.n_cross_heads, T, self.head_dim)
        v_expanded = v.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N * B, self.n_cross_heads, T, self.head_dim)
        q_batched = q_all.reshape(N * B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)

        # Single SDPA call for all columns
        attn_out = F.scaled_dot_product_attention(
            q_batched, k_expanded, v_expanded, is_causal=True,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )

        # Reshape back: [N*B, n_heads, T, head_dim] -> [N, B*T, d_col]
        attn_out = attn_out.transpose(1, 2).contiguous().view(N, B * T, self.d_col)
        attn_out = self.w_o(attn_out).view(N, B, T, self.d_col)

        # Residual + blend (active columns get merged, dropped pass through)
        merged = col_states + self.resid_dropout(attn_out)
        mask_bool = col_mask.view(N, 1, 1, 1).expand_as(merged)
        return torch.where(mask_bool, merged, col_states)


class BatchedCrossColumnAttentionCompressed(nn.Module):
    """Compressed cross-column attention with batched projections.

    Same as BatchedCrossColumnAttention but compresses K/V to comm_rank
    before aggregation, reducing communication bandwidth.
    """

    def __init__(
        self, n_columns: int, d_col: int, n_cross_heads: int = 4,
        comm_rank: int = 64, quant_comm: bool = False, dropout: float = 0.1,
    ):
        super().__init__()
        self.n_columns = n_columns
        self.d_col = d_col
        self.n_cross_heads = n_cross_heads
        self.head_dim = d_col // n_cross_heads
        self.comm_rank = comm_rank
        self.quant_comm = quant_comm

        # Batched K/V projection + compression
        self.norm_kv = BatchedLayerNorm(n_columns, d_col)
        self.w_k = BatchedLinear(n_columns, d_col, d_col)
        self.w_v = BatchedLinear(n_columns, d_col, d_col)
        self.k_compress = BatchedLinear(n_columns, d_col, comm_rank)
        self.v_compress = BatchedLinear(n_columns, d_col, comm_rank)

        # Shared decompression
        self.k_decompress = nn.Linear(comm_rank, d_col, bias=False)
        self.v_decompress = nn.Linear(comm_rank, d_col, bias=False)

        # Batched Q/O projection
        self.norm_q = BatchedLayerNorm(n_columns, d_col)
        self.w_q = BatchedLinear(n_columns, d_col, d_col)
        self.w_o = BatchedLinear(n_columns, d_col, d_col)

        self.attn_dropout_p = dropout
        self.resid_dropout = nn.Dropout(dropout)

    @staticmethod
    def _simulate_int8_quantize(x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            absmax = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = 127.0 / absmax
        quantized = (x * scale).round() / scale
        return quantized

    def forward(
        self, col_states: torch.Tensor, col_mask: torch.Tensor | None = None
    ) -> torch.Tensor:
        N, B, T, _ = col_states.shape

        if col_mask is None:
            col_mask = torch.ones(N, dtype=torch.bool, device=col_states.device)

        mask_f = col_mask.float().view(N, 1, 1, 1)
        n_active = col_mask.float().sum().clamp(min=1.0)

        # Batched K/V + compression
        normed_kv = self.norm_kv(col_states)
        kv_flat = normed_kv.reshape(N, B * T, self.d_col)
        k_full = self.w_k(kv_flat)  # [N, B*T, d_col]
        v_full = self.w_v(kv_flat)
        k_c = self.k_compress(k_full).view(N, B, T, self.comm_rank) * mask_f  # [N, B, T, rank]
        v_c = self.v_compress(v_full).view(N, B, T, self.comm_rank) * mask_f

        # Quantization simulation
        if self.quant_comm:
            k_c_flat = k_c.reshape(N * B * T, self.comm_rank)
            v_c_flat = v_c.reshape(N * B * T, self.comm_rank)
            if self.training:
                k_c_flat = k_c_flat + (self._simulate_int8_quantize(k_c_flat) - k_c_flat).detach()
                v_c_flat = v_c_flat + (self._simulate_int8_quantize(v_c_flat) - v_c_flat).detach()
            else:
                k_c_flat = self._simulate_int8_quantize(k_c_flat)
                v_c_flat = self._simulate_int8_quantize(v_c_flat)
            k_c = k_c_flat.view(N, B, T, self.comm_rank)
            v_c = v_c_flat.view(N, B, T, self.comm_rank)

        # Aggregate compressed K/V and decompress
        k_avg = k_c.sum(dim=0) / n_active  # [B, T, rank]
        v_avg = v_c.sum(dim=0) / n_active
        k = self.k_decompress(k_avg)  # [B, T, d_col]
        v = self.v_decompress(v_avg)
        k = k.view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)

        # Batched Q
        normed_q = self.norm_q(col_states)
        q_flat = normed_q.reshape(N, B * T, self.d_col)
        q_all = self.w_q(q_flat).view(N, B, T, self.n_cross_heads, self.head_dim)

        # Expand K/V, single SDPA
        k_exp = k.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N * B, self.n_cross_heads, T, self.head_dim)
        v_exp = v.unsqueeze(0).expand(N, -1, -1, -1, -1).reshape(N * B, self.n_cross_heads, T, self.head_dim)
        q_bat = q_all.reshape(N * B, T, self.n_cross_heads, self.head_dim).transpose(1, 2)

        attn_out = F.scaled_dot_product_attention(
            q_bat, k_exp, v_exp, is_causal=True,
            dropout_p=self.attn_dropout_p if self.training else 0.0,
        )

        attn_out = attn_out.transpose(1, 2).contiguous().view(N, B * T, self.d_col)
        attn_out = self.w_o(attn_out).view(N, B, T, self.d_col)

        merged = col_states + self.resid_dropout(attn_out)
        mask_bool = col_mask.view(N, 1, 1, 1).expand_as(merged)
        return torch.where(mask_bool, merged, col_states)


# ============================================================================
# Main model
# ============================================================================

class ColumnTransformerV2Fast(nn.Module):
    """Fast column transformer: vectorized column operations.

    Functionally identical to ColumnTransformerV2 but ~3-5x faster on GPU
    due to batched matmuls replacing Python for-loops.
    """

    def __init__(self, config: ColumnConfigV2):
        super().__init__()
        self.config = config
        N = config.n_columns

        # Shared token embedding
        self.tok_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.drop = nn.Dropout(config.dropout)

        # === Shared trunk: standard transformer blocks ===
        from .model_dense import TransformerBlock
        self.trunk = nn.ModuleList([
            TransformerBlock(config.d_model, config.trunk_n_heads, config.trunk_d_ff, config.dropout)
            for _ in range(config.n_trunk_layers)
        ])

        trunk_head_dim = config.d_model // config.trunk_n_heads
        trunk_rope = precompute_rope(trunk_head_dim, config.max_seq_len)
        self.register_buffer("trunk_rope", trunk_rope, persistent=False)

        # === Column split: batched projection d_model -> d_col ===
        self.col_in_proj = BatchedLinear(N, config.d_model, config.d_col, bias=True)

        # === Column dropout ===
        self.col_dropout = ColumnDropout(
            n_columns=N,
            drop_prob=config.col_drop_prob,
            min_active=config.min_active_columns,
        )

        # === Batched column transformer blocks ===
        self.col_blocks = nn.ModuleList([
            BatchedColumnBlock(N, config.d_col, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_col_layers)
        ])

        col_rope = precompute_rope(config.head_dim, config.max_seq_len)
        self.register_buffer("col_rope", col_rope, persistent=False)

        # === Cross-column attention at merge points ===
        self.merge_layers = nn.ModuleDict()
        if config.merge_every > 0:
            for layer_idx in range(config.merge_every, config.n_col_layers + 1, config.merge_every):
                if config.comm_rank > 0:
                    self.merge_layers[str(layer_idx)] = BatchedCrossColumnAttentionCompressed(
                        N, config.d_col,
                        n_cross_heads=config.n_cross_heads,
                        comm_rank=config.comm_rank,
                        quant_comm=config.quant_comm,
                        dropout=config.dropout,
                    )
                else:
                    self.merge_layers[str(layer_idx)] = BatchedCrossColumnAttention(
                        N, config.d_col,
                        n_cross_heads=config.n_cross_heads,
                        dropout=config.dropout,
                    )

        # === Output ===
        self.out_proj = nn.Linear(config.total_col_dim, config.d_model, bias=False)
        self.norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = self.tok_emb.weight

        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, BatchedLinear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        active_columns: list[int] | None = None,
    ) -> torch.Tensor:
        N = self.config.n_columns

        # === Shared trunk ===
        x = self.drop(self.tok_emb(input_ids))  # [B, T, d_model]
        for block in self.trunk:
            x = block(x, self.trunk_rope)

        B, T, _ = x.shape

        # === Split into columns (batched projection) ===
        # Expand x for all columns: [N, B*T, d_model]
        x_expanded = x.reshape(1, B * T, self.config.d_model).expand(N, -1, -1)
        col_states = self.col_in_proj(x_expanded).view(N, B, T, self.config.d_col)

        # === Column dropout / manual selection ===
        if active_columns is not None:
            col_mask = torch.zeros(N, dtype=torch.bool, device=x.device)
            for idx in active_columns:
                col_mask[idx] = True
            n_active = len(active_columns)
            scale = N / n_active
            mask_f = col_mask.float().view(N, 1, 1, 1) * scale
            col_states = col_states * mask_f
        else:
            # ColumnDropout expects list, returns list + mask — adapt to tensor API
            col_list = [col_states[i] for i in range(N)]
            col_list, col_mask = self.col_dropout(col_list)
            col_states = torch.stack(col_list, dim=0)

        # === Column blocks with merges ===
        for layer_idx in range(self.config.n_col_layers):
            col_states = self.col_blocks[layer_idx](col_states, self.col_rope)

            merge_key = str(layer_idx + 1)
            if merge_key in self.merge_layers:
                col_states = self.merge_layers[merge_key](col_states, col_mask)

        # === Combine and output ===
        # col_states: [N, B, T, d_col] -> [B, T, N * d_col]
        combined = col_states.permute(1, 2, 0, 3).contiguous().view(B, T, -1)
        combined = self.out_proj(combined)
        combined = self.norm(combined)
        return self.lm_head(combined)


# ============================================================================
# Checkpoint conversion: slow model -> fast model
# ============================================================================

def convert_checkpoint_to_fast(slow_state_dict: dict, config: ColumnConfigV2) -> dict:
    """Convert a ColumnTransformerV2 checkpoint to ColumnTransformerV2Fast format.

    The key transformation: N separate nn.Linear weight matrices get stacked
    into single BatchedLinear weight tensors.

    Example:
        slow: columns.0.0.attn.w_q.weight  [d_col, d_col]
              columns.1.0.attn.w_q.weight  [d_col, d_col]
              ...
              columns.7.0.attn.w_q.weight  [d_col, d_col]
        fast: col_blocks.0.attn.w_q.weight [8, d_col, d_col]
    """
    N = config.n_columns
    fast = {}

    # Copy trunk, embedding, output layers unchanged
    for k, v in slow_state_dict.items():
        # Strip torch.compile prefix if present
        k = k.replace("_orig_mod.", "")

        if k.startswith("trunk.") or k.startswith("tok_emb.") or k.startswith("drop.") \
                or k.startswith("out_proj.") or k.startswith("norm.") or k.startswith("lm_head."):
            fast[k] = v
            continue

        # Column input projections: col_in_projs.{i}.weight -> col_in_proj.weight[i]
        if k.startswith("col_in_projs."):
            parts = k.split(".")
            col_idx = int(parts[1])
            param_name = parts[2]  # weight or bias
            target_key = f"col_in_proj.{param_name}"
            if target_key not in fast:
                shape = list(v.shape)
                fast[target_key] = torch.zeros(N, *shape, dtype=v.dtype)
                if param_name == "bias":
                    fast[target_key] = fast[target_key].unsqueeze(1)  # [N, 1, d_col]
            if param_name == "bias":
                fast[target_key][col_idx, 0] = v
            else:
                fast[target_key][col_idx] = v
            continue

        # Column blocks: columns.{col_idx}.{layer_idx}.{...} -> col_blocks.{layer_idx}.{...}.weight[col_idx]
        if k.startswith("columns."):
            parts = k.split(".")
            col_idx = int(parts[1])
            layer_idx = int(parts[2])
            rest = ".".join(parts[3:])

            # Map old structure to new
            # Old: columns.{col}.{layer}.attn.w_q.weight
            # New: col_blocks.{layer}.attn.w_q.weight[col]
            # Old: columns.{col}.{layer}.norm1.weight
            # New: col_blocks.{layer}.norm1.weight[col]
            target_key = f"col_blocks.{layer_idx}.{rest}"

            if target_key not in fast:
                # RMSNorm weights: [N, 1, 1, dim]
                if "norm1.weight" in rest or "norm2.weight" in rest:
                    fast[target_key] = torch.zeros(N, 1, 1, v.shape[0], dtype=v.dtype)
                # BatchedLinear weights: [N, d_out, d_in]
                elif "weight" in rest and len(v.shape) == 2:
                    fast[target_key] = torch.zeros(N, v.shape[0], v.shape[1], dtype=v.dtype)
                elif "bias" in rest and len(v.shape) == 1:
                    fast[target_key] = torch.zeros(N, 1, v.shape[0], dtype=v.dtype)
                else:
                    fast[target_key] = torch.zeros(N, *v.shape, dtype=v.dtype)

            if "norm1.weight" in rest or "norm2.weight" in rest:
                fast[target_key][col_idx, 0, 0] = v
            elif "bias" in rest and len(v.shape) == 1:
                fast[target_key][col_idx, 0] = v
            else:
                fast[target_key][col_idx] = v
            continue

        # Merge layers: already indexed by layer string, need to convert per-column -> batched
        if k.startswith("merge_layers."):
            parts = k.split(".")
            layer_key = parts[1]  # e.g. "4"
            rest = ".".join(parts[2:])

            # Per-column ModuleLists: norms_kv.{i}.weight, w_ks.{i}.weight, etc.
            # -> batched: norm_kv.weight[i], w_k.weight[i], etc.
            per_col_mappings = {
                "norms_kv": "norm_kv",
                "norms_q": "norm_q",
                "w_ks": "w_k",
                "w_vs": "w_v",
                "w_qs": "w_q",
                "w_os": "w_o",
                "k_compress": "k_compress",
                "v_compress": "v_compress",
            }

            matched = False
            for old_prefix, new_prefix in per_col_mappings.items():
                if rest.startswith(f"{old_prefix}."):
                    sub_parts = rest.split(".")
                    col_idx = int(sub_parts[1])
                    param_name = sub_parts[2]  # weight or bias
                    target_key = f"merge_layers.{layer_key}.{new_prefix}.{param_name}"

                    if target_key not in fast:
                        if "norm" in new_prefix:
                            # LayerNorm weight/bias: [N, 1, 1, dim]
                            fast[target_key] = torch.zeros(N, 1, 1, v.shape[0], dtype=v.dtype)
                            if param_name == "weight":
                                fast[target_key].fill_(1.0)  # LayerNorm weight init
                        elif len(v.shape) == 2:
                            fast[target_key] = torch.zeros(N, v.shape[0], v.shape[1], dtype=v.dtype)
                        else:
                            fast[target_key] = torch.zeros(N, *v.shape, dtype=v.dtype)

                    if "norm" in new_prefix:
                        fast[target_key][col_idx, 0, 0] = v
                    else:
                        fast[target_key][col_idx] = v
                    matched = True
                    break

            if not matched:
                # Shared layers (attn_dropout, resid_dropout, k_decompress, v_decompress)
                target_key = f"merge_layers.{layer_key}.{rest}"
                fast[target_key] = v
            continue

        # Column dropout has no parameters, skip
        if k.startswith("col_dropout."):
            fast[k] = v
            continue

        print(f"  WARNING: unmapped key: {k}")

    return fast
