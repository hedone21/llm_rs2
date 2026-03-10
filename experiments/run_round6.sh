#!/bin/bash
# Round 6: H2O 품질 저하 원인 규명 — 단일 가설 격리 실험
#
# Step 1: recent_window 격리 (HH 제거, keep_ratio=0.0)
#   V-01: rw=128, V-02: rw=133, V-03: rw=192, V-04: rw=256
#
# Step 2: HH 효과 격리 (recent_window=128 고정)
#   V-05: kr=0.2, V-06: kr=0.8
#
# Controls (already available):
#   B-512        — no eviction (EMR=1.000)
#   M-C-256-sl   — sliding (EMR=0.687)
#   M-C-256-h2o  — H2O kr=0.5/rw=128/d=0.1 (EMR=0.593)

set -euo pipefail

BINARY="./target/release/generate"
MODEL="/home/go/Workspace/models"
RESULTS="experiments/results"
CONFIGS="experiments/configs"
PROMPT="The history of artificial intelligence began in antiquity, with myths, stories and rumors of"

mkdir -p "$RESULTS"

H2O_COMMON="--h2o-decay 0.1"

run_experiment() {
    local name="$1"
    local kr="$2"
    local rw="$3"

    echo "━━━ Running: $name (kr=$kr, rw=$rw) ━━━"

    $BINARY -m "$MODEL" \
        -p "$PROMPT" \
        -n 512 \
        --greedy \
        --eviction-policy h2o \
        --h2o-keep-ratio "$kr" \
        --h2o-recent-window "$rw" \
        $H2O_COMMON \
        --experiment-schedule "$CONFIGS/memory_critical_256.json" \
        --experiment-output "$RESULTS/${name}.jsonl" \
        --experiment-sample-interval 1 \
        --experiment-logits-topk 10 \
        2>&1 | grep -E '^\[Experiment\]|^\[Resilience\]|^\[CacheManager\]|^H2O'

    echo ""
}

echo "========================================"
echo "  Round 6: H2O Root Cause Isolation"
echo "========================================"
echo ""

# ── Step 1: recent_window 격리 (HH=0) ──
echo "── Step 1: recent_window isolation (keep_ratio=0.0, no HH) ──"
echo ""

run_experiment "V-01" 0.0 128   # pure recent 128
run_experiment "V-02" 0.0 133   # match Sliding token count
run_experiment "V-03" 0.0 192   # expanded recent
run_experiment "V-04" 0.0 256   # large recent

# ── Step 2: HH 효과 격리 (rw=128 고정) ──
echo "── Step 2: HH effect isolation (recent_window=128 fixed) ──"
echo ""

run_experiment "V-05" 0.2 128   # HH small
run_experiment "V-06" 0.8 128   # HH large

echo "========================================"
echo "  Round 6 Complete"
echo "========================================"
echo ""
echo "Results in: $RESULTS/"
echo ""
echo "Next: run compare.py for each V-* against B-512 baseline"
