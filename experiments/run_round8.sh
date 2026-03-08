#!/bin/bash
# Round 8: 대규모 시퀀스에서 H2O vs Sliding 비교
#
# 시퀀스 길이: 2048, 4096, 8192, 16384, 32768
# 각 길이에서 50% 지점에 Memory Critical 신호 주입
# 비교: Baseline (no eviction) vs H2O (new) vs Sliding
#
# H2O 설정: kr=0.5, decay=0.0, tracked_layers=0 (all)
# Sliding: eviction_window=65536 (auto-eviction 비활성, signal-only)

set -euo pipefail

BINARY="./target/release/generate"
MODEL="/home/go/Workspace/models"
RESULTS="experiments/results"
CONFIGS="experiments/configs"
PROMPT="The history of artificial intelligence began in antiquity, with myths, stories and rumors of"

mkdir -p "$RESULTS"

run_baseline() {
    local name="$1"
    local tokens="$2"
    local max_seq="$3"

    echo "━━━ Baseline: $name (tokens=$tokens) ━━━"

    $BINARY -m "$MODEL" \
        -p "$PROMPT" \
        -n "$tokens" \
        --greedy \
        --max-seq-len "$max_seq" \
        --eviction-policy none \
        --experiment-schedule "$CONFIGS/baseline.json" \
        --experiment-output "$RESULTS/${name}.jsonl" \
        --experiment-sample-interval 10 \
        --experiment-logits-topk 10 \
        2>&1 | grep -E '^\[Experiment\]|^\[Profile\].*End'

    echo ""
}

run_h2o() {
    local name="$1"
    local tokens="$2"
    local max_seq="$3"
    local inject_pos="$4"

    echo "━━━ H2O: $name (tokens=$tokens, inject@$inject_pos) ━━━"

    $BINARY -m "$MODEL" \
        -p "$PROMPT" \
        -n "$tokens" \
        --greedy \
        --max-seq-len "$max_seq" \
        --eviction-policy h2o \
        --h2o-keep-ratio 0.5 \
        --h2o-decay 0.0 \
        --h2o-tracked-layers 0 \
        --experiment-schedule "$CONFIGS/memory_critical_${inject_pos}.json" \
        --experiment-output "$RESULTS/${name}.jsonl" \
        --experiment-sample-interval 10 \
        --experiment-logits-topk 10 \
        2>&1 | grep -E '^\[Experiment\]|^\[Resilience\]'

    echo ""
}

run_sliding() {
    local name="$1"
    local tokens="$2"
    local max_seq="$3"
    local inject_pos="$4"

    echo "━━━ Sliding: $name (tokens=$tokens, inject@$inject_pos) ━━━"

    $BINARY -m "$MODEL" \
        -p "$PROMPT" \
        -n "$tokens" \
        --greedy \
        --max-seq-len "$max_seq" \
        --eviction-policy sliding \
        --eviction-window "$max_seq" \
        --experiment-schedule "$CONFIGS/memory_critical_${inject_pos}.json" \
        --experiment-output "$RESULTS/${name}.jsonl" \
        --experiment-sample-interval 10 \
        --experiment-logits-topk 10 \
        2>&1 | grep -E '^\[Experiment\]|^\[Resilience\]'

    echo ""
}

echo "========================================"
echo "  Round 8: Long-Sequence H2O vs Sliding"
echo "========================================"
echo ""

START_TIME=$(date +%s)

# ── 2048 tokens (inject@1024) ──
echo "══ 2048 tokens ══"
# B-2048 already exists, skip
run_h2o     "L-2k-h2o" 2048 4096 1024
run_sliding "L-2k-sl"  2048 4096 1024

# ── 4096 tokens (inject@2048) ──
echo "══ 4096 tokens ══"
run_baseline "B-4096"    4096 8192
run_h2o      "L-4k-h2o" 4096 8192 2048
run_sliding  "L-4k-sl"  4096 8192 2048

# ── 8192 tokens (inject@4096) ──
echo "══ 8192 tokens ══"
run_baseline "B-8192"    8192 16384
run_h2o      "L-8k-h2o" 8192 16384 4096
run_sliding  "L-8k-sl"  8192 16384 4096

# ── 16384 tokens (inject@8192) ──
echo "══ 16384 tokens ══"
run_baseline "B-16384"    16384 32768
run_h2o      "L-16k-h2o" 16384 32768 8192
run_sliding  "L-16k-sl"  16384 32768 8192

# ── 32768 tokens (inject@16384) ──
echo "══ 32768 tokens ══"
run_baseline "B-32768"    32768 65536
run_h2o      "L-32k-h2o" 32768 65536 16384
run_sliding  "L-32k-sl"  32768 65536 16384

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
echo "========================================"
echo "  Round 8 Complete (${MINUTES} min)"
echo "========================================"
