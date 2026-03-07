#!/bin/bash
# Round 1: Baseline experiments (5 runs)
# Runs each baseline twice to verify greedy reproducibility

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
    local extra_args="${4:-}"
    local schedule="${5:-$CONFIGS/baseline.json}"

    echo "━━━ Running: $name (${tokens} tokens, eviction=${eviction}) ━━━"

    $BINARY -m "$MODEL" \
        -p "$PROMPT" \
        -n "$tokens" \
        --greedy \
        --eviction-policy "$eviction" \
        --experiment-schedule "$schedule" \
        --experiment-output "$RESULTS/${name}.jsonl" \
        --experiment-sample-interval 1 \
        --experiment-logits-topk 10 \
        $extra_args \
        2>&1 | grep -E '^\[|^Done|^TTFT|^Avg'

    echo ""
}

echo "========================================"
echo "  Round 1: Baseline Experiments"
echo "========================================"
echo ""

# B-128: Speed baseline (128 tokens, no eviction)
run_experiment "B-128" 128 "none"

# B-512: Quality baseline (512 tokens, no eviction)
run_experiment "B-512" 512 "none"

# B-1024: Memory baseline (1024 tokens, no eviction)
run_experiment "B-1024" 1024 "none"

# B-512-sliding: Sliding window overhead (512 tokens)
run_experiment "B-512-sliding" 512 "sliding" "--eviction-window 1024"

# B-512-h2o: H2O overhead (512 tokens)
run_experiment "B-512-h2o" 512 "h2o" "--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"

echo "========================================"
echo "  Reproducibility check (B-128 x2)"
echo "========================================"
echo ""

# Run B-128 again to verify greedy reproducibility
run_experiment "B-128-check" 128 "none"

echo "========================================"
echo "  Round 1 Complete"
echo "========================================"
echo ""
echo "Results in: $RESULTS/"
ls -la "$RESULTS/"
