#!/bin/bash
# Round 13: Run H2O+ experiments on Android device via adb
set -euo pipefail

DEVICE_BIN="/data/local/tmp/llm_rs2/generate"
MODEL="/data/local/tmp/models/llama3.2-1b"
DEVICE_RESULTS="/data/local/tmp/llm_rs2/results/round13"
DEVICE_CONFIGS="/data/local/tmp/llm_rs2/configs"
LOCAL_RESULTS="experiments/results/round13"

mkdir -p "$LOCAL_RESULTS"

SCHEDULE="$DEVICE_CONFIGS/memory_critical_1024.json"
TOKENS=2048
MAX_SEQ=4096
EVICT_RATIO="0.20"

PPL01="It was a bright cold day in April, and the clocks were striking thirteen. The street was deserted except for a few figures hurrying along the pavement, their coat collars turned up against the wind that swept down from the grey towers."
PPL03="A hash table is a data structure that implements an associative array, mapping keys to values. It uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found. Ideally, the hash function will assign each key to a unique bucket."

TOTAL=0
DONE=0
SKIPPED=0
START_TIME=$(date +%s)

run_on_device() {
    local name="$1"
    local prompt="$2"
    local policy="$3"
    local keep_ratio="$4"
    local decay="$5"

    TOTAL=$((TOTAL + 1))
    local local_out="$LOCAL_RESULTS/${name}.jsonl"
    local device_out="$DEVICE_RESULTS/${name}.jsonl"

    if [[ -f "$local_out" ]]; then
        echo "  SKIP $name (already exists)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    echo "  RUN  $name (policy=$policy, kr=$keep_ratio, decay=$decay)"

    adb shell "$DEVICE_BIN -m $MODEL \
        -p '$prompt' \
        -n $TOKENS --max-seq-len $MAX_SEQ --greedy \
        --eviction-policy $policy \
        --h2o-keep-ratio $keep_ratio --h2o-decay $decay \
        --experiment-schedule $SCHEDULE \
        --experiment-eviction-ratio $EVICT_RATIO \
        --experiment-output $device_out \
        --experiment-logits-topk 10 --experiment-sample-interval 10" \
        2>&1 | grep -E '^\[Experiment\]|^\[Resilience\]|Error|tok/s' | head -5

    # Pull result back
    adb pull "$device_out" "$local_out" 2>/dev/null && DONE=$((DONE + 1)) || echo "  WARN: failed to pull $name"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 13: H2O+ Experiments on Device"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Ensure results dir on device
adb shell "mkdir -p $DEVICE_RESULTS"

# ─── Phase 1: 4-Way (h2o_plus kr=0.5 only, rest reused) ───
echo ""
echo "── Phase 1: H2O+ kr=0.5 ──"
run_on_device "H2OP-50-PPL01" "$PPL01" "h2o_plus" "0.5" "0.0"
run_on_device "H2OP-50-PPL03" "$PPL03" "h2o_plus" "0.5" "0.0"

# ─── Phase 2: H2O+ Keep Ratio Sweep ───
echo ""
echo "── Phase 2: H2O+ Keep Ratio Sweep ──"
for kr in 0.0 0.1 0.2 0.3 0.7 0.9; do
    kr_label=$(echo "$kr" | awk '{printf "%02.0f", $1*100}')
    run_on_device "H2OP-${kr_label}-PPL01" "$PPL01" "h2o_plus" "$kr" "0.0"
    run_on_device "H2OP-${kr_label}-PPL03" "$PPL03" "h2o_plus" "$kr" "0.0"
done

# ─── Summary ───
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 13 Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:   $TOTAL"
echo "  Done:    $DONE"
echo "  Skipped: $SKIPPED"
echo "  Time:    ${MINUTES}m ${SECONDS_REM}s"
echo "  Results: $LOCAL_RESULTS/"
