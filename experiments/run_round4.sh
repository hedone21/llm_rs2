#!/bin/bash
# Round 4: H2O Parameter Sweep (9 runs)

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
echo "  Round 4: H2O Parameter Sweep"
echo "========================================"
echo ""

# All use memory_critical_256.json, 512 tokens, h2o eviction
# Vary: keep_ratio (0.3, 0.5, 0.7), recent_window (64, 128), decay (0.05, 0.1)

echo "── keep_ratio=0.3 ──"
echo ""

run_experiment "H-01" 512 "h2o" "memory_critical_256.json" "--h2o-keep-ratio 0.3 --h2o-recent-window 64 --h2o-decay 0.1"
run_experiment "H-02" 512 "h2o" "memory_critical_256.json" "--h2o-keep-ratio 0.3 --h2o-recent-window 128 --h2o-decay 0.1"
run_experiment "H-03" 512 "h2o" "memory_critical_256.json" "--h2o-keep-ratio 0.3 --h2o-recent-window 128 --h2o-decay 0.05"

echo "── keep_ratio=0.5 ──"
echo ""

run_experiment "H-04" 512 "h2o" "memory_critical_256.json" "--h2o-keep-ratio 0.5 --h2o-recent-window 64 --h2o-decay 0.1"
run_experiment "H-05" 512 "h2o" "memory_critical_256.json" "--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"
run_experiment "H-06" 512 "h2o" "memory_critical_256.json" "--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.05"

echo "── keep_ratio=0.7 ──"
echo ""

run_experiment "H-07" 512 "h2o" "memory_critical_256.json" "--h2o-keep-ratio 0.7 --h2o-recent-window 64 --h2o-decay 0.1"
run_experiment "H-08" 512 "h2o" "memory_critical_256.json" "--h2o-keep-ratio 0.7 --h2o-recent-window 128 --h2o-decay 0.1"
run_experiment "H-09" 512 "h2o" "memory_critical_256.json" "--h2o-keep-ratio 0.7 --h2o-recent-window 128 --h2o-decay 0.05"

echo "========================================"
echo "  Round 4 Complete"
echo "========================================"
echo ""
echo "Results in: $RESULTS/"
ls -la "$RESULTS/"
