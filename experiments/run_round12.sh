#!/bin/bash
# Round 12: H2O Root Cause Analysis
#
# Hypotheses:
#   H1: Budget split (keep_ratio) is the primary cause
#   H3: Decay parameter neutralizes heavy hitter selection
#   H5: Heavy hitters are fundamentally valueless for autoregressive generation
#
# Phases:
#   1) Keep Ratio Sweep — 7 ratios × 2 prompts = 14 experiments
#   3) Decay Sweep — 5 decays × 2 prompts = 10 experiments
#   4) Sliding Window Reduction — 1 config × 2 prompts = 2 experiments
#
# Total: 26 new experiments. Round 11 results reused for baselines.
#
# References: experiments/PLAN_round12.md

set -euo pipefail

BINARY="./target/release/generate"
MODEL="/home/go/Workspace/models"
RESULTS="experiments/results/round12"
CONFIGS="experiments/configs"

mkdir -p "$RESULTS"

# ============================================================
#  Counters
# ============================================================
TOTAL=0
DONE=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

run_generate() {
    local name="$1"
    local prompt="$2"
    local tokens="$3"
    local max_seq="$4"
    local eviction="$5"    # none | sliding | h2o
    local schedule="$6"    # config json path or "none"
    local evict_ratio="$7" # override ratio or "none"
    shift 7
    local extra_args=("$@")

    TOTAL=$((TOTAL + 1))
    local outfile="$RESULTS/${name}.jsonl"

    if [[ -f "$outfile" ]]; then
        echo "  SKIP $name (already exists)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local cmd=("$BINARY" -m "$MODEL" -p "$prompt" -n "$tokens"
        --max-seq-len "$max_seq" --greedy
        --experiment-output "$outfile"
        --experiment-sample-interval 10
        --experiment-logits-topk 10)

    case "$eviction" in
        none)
            cmd+=(--eviction-policy none)
            ;;
        sliding)
            cmd+=(--eviction-policy sliding --eviction-window "$max_seq")
            ;;
        h2o)
            # Only set policy; keep_ratio and decay come from extra_args
            cmd+=(--eviction-policy h2o)
            ;;
    esac

    if [[ "$schedule" != "none" ]]; then
        cmd+=(--experiment-schedule "$schedule")
    fi

    if [[ "$evict_ratio" != "none" ]]; then
        cmd+=(--experiment-eviction-ratio "$evict_ratio")
    fi

    if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "  RUN  $name ($eviction, ${tokens}tok)"

    if "${cmd[@]}" 2>&1 | grep -E '^\[Experiment\]|^\[Resilience\]|Error' | head -5; then
        DONE=$((DONE + 1))
    else
        DONE=$((DONE + 1))
    fi
}

# ============================================================
#  Prompts (PPL01: Literary, PPL03: Technical)
# ============================================================
PPL01="It was a bright cold day in April, and the clocks were striking thirteen. The street was deserted except for a few figures hurrying along the pavement, their coat collars turned up against the wind that swept down from the grey towers."
PPL03="A hash table is a data structure that implements an associative array, mapping keys to values. It uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found. Ideally, the hash function will assign each key to a unique bucket."

declare -A PPL_PROMPTS
PPL_PROMPTS[PPL01]="$PPL01"
PPL_PROMPTS[PPL03]="$PPL03"

# ============================================================
#  Common parameters
# ============================================================
TOKENS=2048
MAX_SEQ=4096
SCHEDULE="$CONFIGS/memory_critical_1024.json"
EVICT_RATIO="0.20"  # 80% eviction

# ============================================================
#  Phase 1: Keep Ratio Sweep (H1 — Budget Split)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 12 Phase 1: Keep Ratio Sweep (H1)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Varying h2o-keep-ratio [0.0 .. 1.0], fixed decay=0.1"
echo "  KR-50 (keep_ratio=0.5) reused from Round 11"
echo ""

# KR-50 is already in Round 11 as PPL0x-2048-h2o.jsonl — skip
for kr in 0.0 0.1 0.2 0.3 0.7 0.9 1.0; do
    # Label: "00", "10", "20", "30", "70", "90", "100"
    kr_label=$(echo "$kr" | awk '{printf "%02.0f", $1*100}')
    # Fix "100" for 1.0 (awk gives "100" already)

    echo "── KR-${kr_label} (keep_ratio=${kr}) ──"

    for ppl_id in PPL01 PPL03; do
        prompt="${PPL_PROMPTS[$ppl_id]}"
        run_generate "KR-${kr_label}-${ppl_id}" "$prompt" "$TOKENS" "$MAX_SEQ" \
            h2o "$SCHEDULE" "$EVICT_RATIO" \
            --h2o-keep-ratio "$kr" --h2o-decay 0.1
    done
done

# ============================================================
#  Phase 3: Decay Parameter Sweep (H3)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 12 Phase 3: Decay Sweep (H3)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Varying h2o-decay [0.0 .. 0.9], fixed keep_ratio=0.5"
echo "  D-010 (decay=0.10) reused from Round 11"
echo ""

# D-010 is already in Round 11 as PPL0x-2048-h2o.jsonl — skip
for decay in 0.00 0.01 0.05 0.50 0.90; do
    # Label: "000", "001", "005", "050", "090"
    d_label=$(echo "$decay" | awk '{printf "%03.0f", $1*100}')

    echo "── D-${d_label} (decay=${decay}) ──"

    for ppl_id in PPL01 PPL03; do
        prompt="${PPL_PROMPTS[$ppl_id]}"
        run_generate "D-${d_label}-${ppl_id}" "$prompt" "$TOKENS" "$MAX_SEQ" \
            h2o "$SCHEDULE" "$EVICT_RATIO" \
            --h2o-keep-ratio 0.5 --h2o-decay "$decay"
    done
done

# ============================================================
#  Phase 4: Sliding Window Reduction (H5)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 12 Phase 4: Sliding Reduction (H5)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Sliding with eviction_ratio=0.118 → ~86 recent tokens"
echo "  Same recent window as H2O (keep_ratio=0.5)"
echo ""

# SL-086: Sliding with ratio 0.118 → keeps ~126 tokens (prefix 40 + recent 86)
for ppl_id in PPL01 PPL03; do
    prompt="${PPL_PROMPTS[$ppl_id]}"
    run_generate "SL-086-${ppl_id}" "$prompt" "$TOKENS" "$MAX_SEQ" \
        sliding "$SCHEDULE" "0.118"
done

# ============================================================
#  Summary
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$(( END_TIME - START_TIME ))
MINUTES=$(( ELAPSED / 60 ))
SECONDS_REM=$(( ELAPSED % 60 ))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 12 Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:   $TOTAL experiments"
echo "  Done:    $DONE"
echo "  Skipped: $SKIPPED"
echo "  Failed:  $FAILED"
echo "  Time:    ${MINUTES}m ${SECONDS_REM}s"
echo ""
echo "Results in: $RESULTS/"
echo "Round 11 baselines in: experiments/results/round11/"
echo "Next: python experiments/analysis/round12_analyze.py"
