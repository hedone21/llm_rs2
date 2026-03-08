#!/bin/bash
# Round 9: Phase 2 — Prompt diversity validation
#
# Purpose: Verify that the non-linear EMR pattern is a measurement artifact
# caused by repetitive text + phase alignment, NOT a real quality issue.
#
# Strategy: Use 3 different prompts and compare EMR patterns.
# If the non-linearity pattern changes with different prompts,
# it confirms the artifact hypothesis.
#
# Prompts:
#   P1 (original): "The history of artificial intelligence began in antiquity..."
#   P2 (technical): "Chapter 1: Introduction to Computer Networks..."
#   P3 (narrative): "Once upon a time in a small village near the mountains..."

set -euo pipefail

BINARY="./target/release/generate"
MODEL="/home/go/Workspace/models"
RESULTS="experiments/results"
CONFIGS="experiments/configs"

mkdir -p "$RESULTS"

# Prompts
P1="The history of artificial intelligence began in antiquity, with myths, stories and rumors of"
P2="Chapter 1: Introduction to Computer Networks. A computer network is a collection of interconnected devices that"
P3="Once upon a time in a small village near the mountains, there lived an old man who had three sons. The eldest son was"

run_baseline() {
    local name="$1"
    local tokens="$2"
    local max_seq="$3"
    local prompt="$4"

    echo "━━━ Baseline: $name (tokens=$tokens) ━━━"

    $BINARY -m "$MODEL" \
        -p "$prompt" \
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

run_sliding() {
    local name="$1"
    local tokens="$2"
    local max_seq="$3"
    local inject_pos="$4"
    local prompt="$5"

    echo "━━━ Sliding: $name (tokens=$tokens, inject@$inject_pos) ━━━"

    $BINARY -m "$MODEL" \
        -p "$prompt" \
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
echo "  Round 9: Prompt Diversity Validation"
echo "========================================"
echo ""

START_TIME=$(date +%s)

# ── Quick repetition check: 512 tokens with each prompt ──
echo "══ Phase 2a: Quick repetition check (512 tokens) ══"
run_baseline "V-P1-512" 512 1024 "$P1"
run_baseline "V-P2-512" 512 1024 "$P2"
run_baseline "V-P3-512" 512 1024 "$P3"

# ── Prompt 2: 2048, 4096 tokens with baseline + sliding ──
echo "══ Phase 2b: Prompt 2 — 2K and 4K ══"
run_baseline "V-P2-B2k" 2048 4096 "$P2"
run_sliding  "V-P2-2k"  2048 4096 1024 "$P2"

run_baseline "V-P2-B4k" 4096 8192 "$P2"
run_sliding  "V-P2-4k"  4096 8192 2048 "$P2"

# ── Prompt 3: 2048, 4096 tokens with baseline + sliding ──
echo "══ Phase 2c: Prompt 3 — 2K and 4K ══"
run_baseline "V-P3-B2k" 2048 4096 "$P3"
run_sliding  "V-P3-2k"  2048 4096 1024 "$P3"

run_baseline "V-P3-B4k" 4096 8192 "$P3"
run_sliding  "V-P3-4k"  4096 8192 2048 "$P3"

END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
echo "========================================"
echo "  Round 9 Complete (${MINUTES} min)"
echo "========================================"
