#!/bin/bash
# Full downstream task evaluation: Baseline vs H2O vs H2O+ at various budgets
#
# Matches H2O paper methodology: 5-shot log-likelihood evaluation
# Tasks: BoolQ, ARC-Easy (HellaSwag needs larger max-seq-len, run separately)
#
# Usage: bash experiments/benchmarks/run_full_eval.sh

set -euo pipefail

MODEL="models/llama3.2-1b"
N_QUESTIONS=50
MAX_SEQ_LEN=1024
RESULTS_DIR="experiments/benchmarks/results"
SCRIPT="experiments/benchmarks/run_eval.py"

mkdir -p "$RESULTS_DIR"

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Downstream Task Evaluation (H2O Paper Methodology)"
echo "  Model: $MODEL | Questions: $N_QUESTIONS | Max Seq: $MAX_SEQ_LEN"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
START_TIME=$(date +%s)

# ── Phase 1: Baseline (no eviction) ──
echo "▶ Phase 1: Baseline (full attention, no eviction)"
python3 "$SCRIPT" \
    --model "$MODEL" \
    --tasks boolq,arc_easy \
    --policies none \
    --n-questions "$N_QUESTIONS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --output "$RESULTS_DIR/baseline_none.json"

# ── Phase 2: H2O with budget 256 ──
echo ""
echo "▶ Phase 2: H2O (budget=256)"
python3 "$SCRIPT" \
    --model "$MODEL" \
    --tasks boolq,arc_easy \
    --policies h2o \
    --kv-budget 256 \
    --n-questions "$N_QUESTIONS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --output "$RESULTS_DIR/h2o_b256.json"

# ── Phase 3: H2O+ with budget 256 ──
echo ""
echo "▶ Phase 3: H2O+ (budget=256)"
python3 "$SCRIPT" \
    --model "$MODEL" \
    --tasks boolq,arc_easy \
    --policies h2o_plus \
    --kv-budget 256 \
    --n-questions "$N_QUESTIONS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --output "$RESULTS_DIR/h2o_plus_b256.json"

# ── Phase 4: Sliding window with budget 256 ──
echo ""
echo "▶ Phase 4: Sliding Window (budget=256)"
python3 "$SCRIPT" \
    --model "$MODEL" \
    --tasks boolq,arc_easy \
    --policies sliding \
    --kv-budget 256 \
    --n-questions "$N_QUESTIONS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --output "$RESULTS_DIR/sliding_b256.json"

# ── Phase 5: H2O with budget 128 (aggressive) ──
echo ""
echo "▶ Phase 5: H2O (budget=128, aggressive)"
python3 "$SCRIPT" \
    --model "$MODEL" \
    --tasks boolq,arc_easy \
    --policies h2o \
    --kv-budget 128 \
    --n-questions "$N_QUESTIONS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --output "$RESULTS_DIR/h2o_b128.json"

# ── Phase 6: H2O+ with budget 128 (aggressive) ──
echo ""
echo "▶ Phase 6: H2O+ (budget=128, aggressive)"
python3 "$SCRIPT" \
    --model "$MODEL" \
    --tasks boolq,arc_easy \
    --policies h2o_plus \
    --kv-budget 128 \
    --n-questions "$N_QUESTIONS" \
    --max-seq-len "$MAX_SEQ_LEN" \
    --output "$RESULTS_DIR/h2o_plus_b128.json"

# ── Summary ──
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Evaluation Complete (${MINUTES}m)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Results:"
for f in "$RESULTS_DIR"/*.json; do
    echo "    $f"
done
echo ""

# Generate comparison report
python3 -c "
import json, os, glob

results_dir = '$RESULTS_DIR'
files = sorted(glob.glob(os.path.join(results_dir, '*.json')))

print()
print('┌─────────────────────────────────────────────────────────────────────┐')
print('│  Task           Policy       Budget    Accuracy    Time(s)         │')
print('├─────────────────────────────────────────────────────────────────────┤')

for f in files:
    with open(f) as fh:
        data = json.load(fh)
    for key, r in data.items():
        budget = str(r['kv_budget']) if r['kv_budget'] > 0 else 'full'
        print(f\"│  {r['task']:<15} {r['policy']:<12} {budget:<8}  {r['accuracy']:>6.1%}     {r.get('wall_time_s',0):>7.0f}s  │\")

print('└─────────────────────────────────────────────────────────────────────┘')
"
