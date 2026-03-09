#!/bin/bash
# Round 14: H2O/H2O+ with Real Attention Scores (score bug fixed)
# Re-validates Round 12/13 conclusions with actual attention score accumulation.
set -euo pipefail

DEVICE_BIN="/data/local/tmp/generate"
MODEL="/data/local/tmp/models/llama3.2-1b"
DEVICE_RESULTS="/data/local/tmp/round14"
LOCAL_RESULTS="experiments/results/round14"
SCHEDULE="/data/local/tmp/llm_rs2/configs/memory_critical_1024.json"

mkdir -p "$LOCAL_RESULTS"

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
    shift 5
    local extra_args=("$@")

    TOTAL=$((TOTAL + 1))
    local local_out="$LOCAL_RESULTS/${name}.jsonl"
    local device_out="$DEVICE_RESULTS/${name}.jsonl"

    if [[ -f "$local_out" ]]; then
        echo "  SKIP $name (already exists)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    echo "  RUN  $name (policy=$policy, kr=$keep_ratio)"

    local cmd="$DEVICE_BIN -m $MODEL"
    cmd+=" -p '$prompt'"
    cmd+=" -n $TOKENS --max-seq-len $MAX_SEQ --greedy"
    cmd+=" --eviction-policy $policy"

    if [[ "$policy" == "h2o" || "$policy" == "h2o_plus" ]]; then
        cmd+=" --h2o-keep-ratio $keep_ratio --h2o-decay $decay"
    fi

    if [[ "$policy" != "none" ]]; then
        cmd+=" --experiment-schedule $SCHEDULE"
        cmd+=" --experiment-eviction-ratio $EVICT_RATIO"
    fi

    cmd+=" --experiment-output $device_out"
    cmd+=" --experiment-logits-topk 10 --experiment-sample-interval 10"

    for arg in "${extra_args[@]}"; do
        cmd+=" $arg"
    done

    adb shell "$cmd" 2>&1 | grep -E '^\[Experiment\]|^\[Resilience\]|Error|tok/s' | head -5

    adb pull "$device_out" "$local_out" 2>/dev/null && DONE=$((DONE + 1)) || echo "  WARN: failed to pull $name"
}

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 14: H2O/H2O+ with Real Scores"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Score bug fixed: compute_attention_scores()"
echo ""

adb shell "mkdir -p $DEVICE_RESULTS"

# ─── Phase 1: Baselines ───
echo "── Phase 1: Baselines ──"
echo "  (Reusing Round 13 baselines — not score-dependent)"
# Copy Round 13 baselines if they exist
for f in BASE-PPL01 BASE-PPL03 SL-PPL01 SL-PPL03; do
    src="experiments/results/round13/${f}.jsonl"
    dst="$LOCAL_RESULTS/${f}.jsonl"
    if [[ -f "$src" && ! -f "$dst" ]]; then
        cp "$src" "$dst"
        echo "  COPY $f from Round 13"
    fi
done

# ─── Phase 2: H2O Keep Ratio Sweep ───
echo ""
echo "── Phase 2: H2O Keep Ratio Sweep (with real scores) ──"
for kr in 0.0 0.1 0.2 0.3 0.5 0.7 0.9; do
    kr_label=$(echo "$kr" | awk '{printf "%02.0f", $1*100}')
    for ppl_id in PPL01 PPL03; do
        if [[ "$ppl_id" == "PPL01" ]]; then
            prompt="$PPL01"
        else
            prompt="$PPL03"
        fi
        run_on_device "H2O-${kr_label}-${ppl_id}" "$prompt" "h2o" "$kr" "0.0"
    done
done

# ─── Phase 3: H2O+ Keep Ratio Sweep ───
echo ""
echo "── Phase 3: H2O+ Keep Ratio Sweep (with real scores) ──"
for kr in 0.0 0.1 0.2 0.3 0.5 0.7 0.9; do
    kr_label=$(echo "$kr" | awk '{printf "%02.0f", $1*100}')
    for ppl_id in PPL01 PPL03; do
        if [[ "$ppl_id" == "PPL01" ]]; then
            prompt="$PPL01"
        else
            prompt="$PPL03"
        fi
        run_on_device "H2OP-${kr_label}-${ppl_id}" "$prompt" "h2o_plus" "$kr" "0.0"
    done
done

# ─── Summary ───
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 14 Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:   $TOTAL"
echo "  Done:    $DONE"
echo "  Skipped: $SKIPPED"
echo "  Time:    ${MINUTES}m ${SECONDS_REM}s"
echo "  Results: $LOCAL_RESULTS/"
echo ""
echo "Next: python experiments/analysis/round14_analyze.py"
