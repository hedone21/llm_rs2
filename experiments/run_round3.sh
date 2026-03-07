#!/bin/bash
# Round 3: Injection Position Variable (8 runs)

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
echo "  Round 3: Injection Position Variable"
echo "========================================"
echo ""

H2O_ARGS="--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"

# ── Quality experiments (512 tokens, h2o, eviction at different positions) ──

echo "── Quality: eviction position sweep (512 tokens) ──"
echo ""

run_experiment "P-128"  512 "h2o" "memory_critical_128.json" "$H2O_ARGS"
run_experiment "P-256"  512 "h2o" "memory_critical_256.json" "$H2O_ARGS"
run_experiment "P-384"  512 "h2o" "memory_critical_384.json" "$H2O_ARGS"
run_experiment "P-448"  512 "h2o" "memory_critical_448.json" "$H2O_ARGS"

# ── Memory experiments (1024 tokens, h2o, eviction at different positions) ──

echo "── Memory: eviction position sweep (1024 tokens) ──"
echo ""

run_experiment "RP-256"  1024 "h2o" "memory_critical_256.json" "$H2O_ARGS"
run_experiment "RP-512"  1024 "h2o" "memory_critical_512.json" "$H2O_ARGS"
run_experiment "RP-768"  1024 "h2o" "memory_critical_768.json" "$H2O_ARGS"
run_experiment "RP-896"  1024 "h2o" "memory_critical_896.json" "$H2O_ARGS"

echo "========================================"
echo "  Round 3 Complete"
echo "========================================"
echo ""
echo "Results in: $RESULTS/"
ls -la "$RESULTS/"
