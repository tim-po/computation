#!/bin/bash
# =============================================================================
# H100 Experiment: Column-Parallel Transformers with Column Dropout
# =============================================================================
# Run this on the Vast.ai H100 NVL instance.
#
# Experiment plan:
#   1. Train dense baseline (~350M params, 16 layers, d=1024)
#   2. Train column model WITHOUT dropout (quality ceiling for columns)
#   3. Train column model with 25% dropout (sweet spot?)
#   4. Train column model with 50% dropout (aggressive — max fault tolerance)
#   5. Run degradation eval on all column models
#
# Expected time: ~4-6 hours total on H100
# Expected cost: ~$12-18 at ~$3/hr
# =============================================================================

set -e

echo "============================================"
echo "H100 EXPERIMENT: Column Dropout at Scale"
echo "============================================"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo "VRAM: $(nvidia-smi --query-gpu=memory.total --format=csv,noheader 2>/dev/null || echo 'unknown')"
echo ""

# Install deps if needed
pip install -q datasets transformers tensorboard matplotlib

# --- Phase 1: Training ---
echo ""
echo "=== PHASE 1: Training all models ==="
echo ""

# Train dense baseline + all 3 column variants
# bf16 + compile for ~5-6x speedup on H100
# batch 32 × 2 accum = effective batch 64
python run_experiment.py \
    --max-steps 30000 \
    --batch-size 32 \
    --grad-accum 2 \
    --seq-len 1024 \
    --lr 3e-4 \
    --eval-every 1000 \
    --bf16 \
    --compile \
    --models h100_dense h100_col8_drop0 h100_col8_drop25 h100_col8_drop50

# --- Phase 2: Degradation evaluation ---
echo ""
echo "=== PHASE 2: Degradation evaluation ==="
echo ""

for model in h100_col8_drop0 h100_col8_drop25 h100_col8_drop50; do
    echo "--- Evaluating ${model} ---"
    python eval_degradation.py \
        --model ${model} \
        --checkpoint checkpoints/${model}_best.pt \
        --samples-per-k 15 \
        --batch-size 64 \
        --max-batches 200
    echo ""
done

echo "============================================"
echo "EXPERIMENT COMPLETE"
echo "Results in: results/"
echo "  - training_curves.png"
echo "  - final_comparison.png"
echo "  - degradation_*.png"
echo "  - degradation_*.json"
echo "============================================"
