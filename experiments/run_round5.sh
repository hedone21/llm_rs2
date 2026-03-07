#!/bin/bash
# Round 5: Compound Conditions (8 runs + optional B-2048 baseline)

set -euo pipefail

BINARY="./target/release/generate"
MODEL="/home/go/Workspace/models"
RESULTS="experiments/results"
CONFIGS="experiments/configs"
PROMPT="The history of artificial intelligence began in antiquity, with myths, stories and rumors of"

mkdir -p "$RESULTS"

run_experiment() {
    local name="$1"
    local tokens="$2"
    local eviction="$3"
    local schedule="$4"
    local extra_args="${5:-}"

    echo "━━━ Running: $name (${tokens}tok, eviction=${eviction}) ━━━"

    $BINARY -m "$MODEL" \
        -p "$PROMPT" \
        -n "$tokens" \
        --greedy \
        --eviction-policy "$eviction" \
        --experiment-schedule "$CONFIGS/$schedule" \
        --experiment-output "$RESULTS/${name}.jsonl" \
        --experiment-sample-interval 1 \
        --experiment-logits-topk 10 \
        $extra_args \
        2>&1 | grep -E '^\[Experiment\]|^\[Resilience\]'

    echo ""
}

echo "========================================"
echo "  Round 5: Compound Conditions"
echo "========================================"
echo ""

H2O_ARGS="--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"

# ── B-2048 baseline (required for X-05, X-06, X-07) ──

if [ ! -f "$RESULTS/B-2048.jsonl" ]; then
    echo "── B-2048 baseline not found, running... ──"
    echo ""
    run_experiment "B-2048" 2048 "none" "baseline.json"
else
    echo "── B-2048 baseline already exists, skipping ──"
    echo ""
fi

# ── 512-token compound experiments ──

echo "── Compound experiments (512 tokens) ──"
echo ""

run_experiment "X-01" 512 "h2o" "x01_thermal_memory_256.json"  "$H2O_ARGS"
run_experiment "X-02" 512 "h2o" "x02_chain_128_256_384.json"   "$H2O_ARGS"
run_experiment "X-03" 512 "h2o" "x03_repeated_eviction.json"   "$H2O_ARGS --eviction-window 256"
run_experiment "X-04" 512 "h2o" "x04_signal_storm.json"        "$H2O_ARGS"

# ── 2048-token long sequence experiments ──

echo "── Long sequence experiments (2048 tokens) ──"
echo ""

run_experiment "X-05" 2048 "h2o" "x05_memory_512_long.json"    "$H2O_ARGS"
run_experiment "X-06" 2048 "h2o" "x06_memory_1024_long.json"   "$H2O_ARGS"
run_experiment "X-07" 2048 "h2o" "x07_repeated_3x_long.json"   "$H2O_ARGS"

# ── 1024-token compound experiment ──

echo "── Compound experiment (1024 tokens) ──"
echo ""

run_experiment "X-08" 1024 "h2o" "x08_thermal_256_memory_512.json" "$H2O_ARGS"

echo "========================================"
echo "  Round 5 Complete"
echo "========================================"
echo ""
echo "Results in: $RESULTS/"
ls -la "$RESULTS/"
