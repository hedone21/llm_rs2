#!/bin/bash
# Round 2: Single signal experiments (14 runs)

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
echo "  Round 2: Single Signal Experiments"
echo "========================================"
echo ""

# ── Speed experiments (128 tokens, no eviction) ──

echo "── Speed experiments (128 tokens) ──"
echo ""

run_experiment "T-W-32"     128 "none" "thermal_warning_32.json"
run_experiment "T-C-32"     128 "none" "thermal_critical_32.json"
run_experiment "T-CR-32-96" 128 "none" "thermal_crit_32_recover_96.json"
run_experiment "C-32"       128 "none" "compute_cpu_32.json"
run_experiment "E-32"       128 "none" "energy_emergency_32.json"

# ── Quality experiments (512 tokens, with eviction) ──

echo "── Quality experiments (512 tokens) ──"
echo ""

run_experiment "M-W-256-sl"    512 "sliding" "memory_warning_256.json"    "--eviction-window 1024"
run_experiment "M-C-256-sl"    512 "sliding" "memory_critical_256.json"   "--eviction-window 1024"
run_experiment "M-C-256-h2o"   512 "h2o"     "memory_critical_256.json"   "--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"
run_experiment "M-CR-256-384"  512 "h2o"     "memory_crit_256_recover_384.json" "--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"

# ── Memory experiments (1024 tokens, with eviction) ──

echo "── Memory experiments (1024 tokens) ──"
echo ""

run_experiment "R-C-512-sl"     1024 "sliding" "memory_critical_512.json"   "--eviction-window 1024"
run_experiment "R-C-512-h2o"    1024 "h2o"     "memory_critical_512.json"   "--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"
run_experiment "R-C-256-h2o"    1024 "h2o"     "memory_critical_256.json"   "--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"
run_experiment "R-C-768-h2o"    1024 "h2o"     "memory_critical_768.json"   "--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"
run_experiment "R-CR-512-768"   1024 "h2o"     "memory_crit_512_recover_768.json" "--h2o-keep-ratio 0.5 --h2o-recent-window 128 --h2o-decay 0.1"

echo "========================================"
echo "  Round 2 Complete"
echo "========================================"
echo ""
echo "Results in: $RESULTS/"
ls -la "$RESULTS/"
