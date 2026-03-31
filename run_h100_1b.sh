#!/bin/bash
# 1B Scale Validation on Single H100
# Estimated cost: ~$50-100 (1-2 days at ~$3/hr)
#
# This validates the column architecture at 1B scale before
# committing to expensive 8x H100 multi-GPU training.
set -euo pipefail

RESULTS_DIR="results_1b"
mkdir -p "$RESULTS_DIR" checkpoints

echo "============================================================"
echo "1B SCALE VALIDATION EXPERIMENT"
echo "============================================================"
echo "Phase 1: Train dense + column models on FineWeb-Edu"
echo "Phase 2: Degradation eval on column models"
echo "Phase 3: Distributed inference simulation"
echo "============================================================"

# ---------------------------------------------------------------
# Phase 1: Training (~50K steps each, ~6-12 hours per model)
# ---------------------------------------------------------------
# Use batch_size=8 with grad_accum=16 for effective batch of 128
# Seq_len=1024 for all 1B models

echo ""
echo "=== PHASE 1: Training ==="

# 1a) Dense baseline
echo "[1/5] Training h100_1b_dense..."
python run_experiment.py \
    --models h100_1b_dense \
    --dataset fineweb-edu \
    --max-steps 50000 \
    --batch-size 8 \
    --grad-accum 16 \
    --seq-len 1024 \
    --lr 1e-4 \
    --eval-every 1000 \
    --bf16 \
    --compile \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/train_dense.log"

# 1b) Column model (no dropout)
echo "[2/5] Training h100_1b_col8..."
python run_experiment.py \
    --models h100_1b_col8 \
    --dataset fineweb-edu \
    --max-steps 50000 \
    --batch-size 8 \
    --grad-accum 16 \
    --seq-len 1024 \
    --lr 1e-4 \
    --eval-every 1000 \
    --bf16 \
    --compile \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/train_col8.log"

# 1c) Column model with dropout
echo "[3/5] Training h100_1b_col8_drop25..."
python run_experiment.py \
    --models h100_1b_col8_drop25 \
    --dataset fineweb-edu \
    --max-steps 50000 \
    --batch-size 8 \
    --grad-accum 16 \
    --seq-len 1024 \
    --lr 1e-4 \
    --eval-every 1000 \
    --bf16 \
    --compile \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/train_col8_drop25.log"

# 1d) Compressed cross-attention (rank-64) — bandwidth-efficient variant
echo "[4/5] Training h100_1b_col8_comp64..."
python run_experiment.py \
    --models h100_1b_col8_comp64 \
    --dataset fineweb-edu \
    --max-steps 50000 \
    --batch-size 8 \
    --grad-accum 16 \
    --seq-len 1024 \
    --lr 1e-4 \
    --eval-every 1000 \
    --bf16 \
    --compile \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/train_col8_comp64.log"

# 1e) Compressed + int8 quantized communication
echo "[5/5] Training h100_1b_col8_comp64_q8..."
python run_experiment.py \
    --models h100_1b_col8_comp64_q8 \
    --dataset fineweb-edu \
    --max-steps 50000 \
    --batch-size 8 \
    --grad-accum 16 \
    --seq-len 1024 \
    --lr 1e-4 \
    --eval-every 1000 \
    --bf16 \
    --compile \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/train_col8_comp64_q8.log"

# ---------------------------------------------------------------
# Phase 2: Degradation evaluation
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 2: Degradation Evaluation ==="

echo "[1/4] Evaluating h100_1b_col8 degradation..."
python eval_degradation.py \
    --model h100_1b_col8 \
    --checkpoint checkpoints/h100_1b_col8_best.pt \
    --samples-per-k 10 \
    --seq-len 1024 \
    --bf16 \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/degradation_col8.log"

echo "[2/4] Evaluating h100_1b_col8_drop25 degradation..."
python eval_degradation.py \
    --model h100_1b_col8_drop25 \
    --checkpoint checkpoints/h100_1b_col8_drop25_best.pt \
    --samples-per-k 10 \
    --seq-len 1024 \
    --bf16 \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/degradation_col8_drop25.log"

echo "[3/4] Evaluating h100_1b_col8_comp64 degradation..."
python eval_degradation.py \
    --model h100_1b_col8_comp64 \
    --checkpoint checkpoints/h100_1b_col8_comp64_best.pt \
    --samples-per-k 10 \
    --seq-len 1024 \
    --bf16 \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/degradation_col8_comp64.log"

echo "[4/4] Evaluating h100_1b_col8_comp64_q8 degradation..."
python eval_degradation.py \
    --model h100_1b_col8_comp64_q8 \
    --checkpoint checkpoints/h100_1b_col8_comp64_q8_best.pt \
    --samples-per-k 10 \
    --seq-len 1024 \
    --bf16 \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/degradation_col8_comp64_q8.log"

# ---------------------------------------------------------------
# Phase 3: Distributed inference simulation
# ---------------------------------------------------------------
echo ""
echo "=== PHASE 3: Distributed Inference Simulation ==="

echo "[1/4] Bandwidth analysis: full vs compressed (static)..."
for model in h100_1b_col8 h100_1b_col8_comp64 h100_1b_col8_comp64_q8; do
    python eval_distributed.py \
        --model "$model" \
        --analyze-only \
        --batch-sizes 1,8,32,64 \
        --bf16 \
        --results-dir "$RESULTS_DIR" \
        2>&1 | tee "$RESULTS_DIR/bandwidth_${model}.log"
done

echo "[2/4] Inference benchmark: full (no compression)..."
python eval_distributed.py \
    --model h100_1b_col8_drop25 \
    --checkpoint checkpoints/h100_1b_col8_drop25_best.pt \
    --latencies 0,5,10,25,50,100 \
    --batch-sizes 1,8,32 \
    --seq-len 1024 \
    --bf16 \
    --max-batches 50 \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/distributed_full.log"

echo "[3/4] Inference benchmark: compressed rank-64..."
python eval_distributed.py \
    --model h100_1b_col8_comp64 \
    --checkpoint checkpoints/h100_1b_col8_comp64_best.pt \
    --latencies 0,5,10,25,50,100 \
    --batch-sizes 1,8,32 \
    --seq-len 1024 \
    --bf16 \
    --max-batches 50 \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/distributed_comp64.log"

echo "[4/4] Inference benchmark: compressed rank-64 + int8..."
python eval_distributed.py \
    --model h100_1b_col8_comp64_q8 \
    --checkpoint checkpoints/h100_1b_col8_comp64_q8_best.pt \
    --latencies 0,5,10,25,50,100 \
    --batch-sizes 1,8,32 \
    --seq-len 1024 \
    --bf16 \
    --max-batches 50 \
    --results-dir "$RESULTS_DIR" \
    2>&1 | tee "$RESULTS_DIR/distributed_comp64_q8.log"

echo ""
echo "============================================================"
echo "EXPERIMENT COMPLETE"
echo "============================================================"
echo "Results in $RESULTS_DIR/"
echo "  - Training logs and checkpoints"
echo "  - Degradation curves (1-8 columns)"
echo "  - Distributed inference: throughput vs latency"
echo "  - Bandwidth requirements per interconnect"
echo "============================================================"
