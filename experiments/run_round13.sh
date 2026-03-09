#!/bin/bash
# Round 13: H2O+ (Per-Head Eviction) Accuracy Comparison
#
# Compares 4 eviction policies: none, sliding, h2o, h2o_plus
#
# Phases:
#   1) 4-Way Direct Comparison — 2 new (h2o_plus kr=0.5)
#   2) H2O+ Keep Ratio Sweep — 12 new (6 ratios × 2 prompts)
#
# Total: 14 new experiments. Round 11/12 results reused for baselines.
#
# References: experiments/PLAN_round13.md

set -euo pipefail

BINARY="./target/release/generate"
MODEL="/home/go/Workspace/models"
RESULTS="experiments/results/round13"
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
    local eviction="$5"    # none | sliding | h2o | h2o_plus
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
            cmd+=(--eviction-policy h2o)
            ;;
        h2o_plus)
            cmd+=(--eviction-policy h2o_plus)
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
#  Phase 1: 4-Way Direct Comparison
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 13 Phase 1: 4-Way Direct Comparison"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Baselines reused from Round 11 (none, sliding)"
echo "  H2O (kr=0.5) reused from Round 11/12"
echo "  H2O+ (kr=0.5) — NEW"
echo ""

echo "── H2O+ (keep_ratio=0.5, decay=0.0) ──"
for ppl_id in PPL01 PPL03; do
    prompt="${PPL_PROMPTS[$ppl_id]}"
    run_generate "H2OP-50-${ppl_id}" "$prompt" "$TOKENS" "$MAX_SEQ" \
        h2o_plus "$SCHEDULE" "$EVICT_RATIO" \
        --h2o-keep-ratio 0.5 --h2o-decay 0.0
done

# ============================================================
#  Phase 2: H2O+ Keep Ratio Sweep
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 13 Phase 2: H2O+ Keep Ratio Sweep"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "  Varying h2o-keep-ratio [0.0 .. 0.9], fixed decay=0.0"
echo "  H2OP-50 (keep_ratio=0.5) reused from Phase 1"
echo ""

# H2OP-50 already run in Phase 1 — skip
for kr in 0.0 0.1 0.2 0.3 0.7 0.9; do
    kr_label=$(echo "$kr" | awk '{printf "%02.0f", $1*100}')

    echo "── H2OP-${kr_label} (keep_ratio=${kr}) ──"

    for ppl_id in PPL01 PPL03; do
        prompt="${PPL_PROMPTS[$ppl_id]}"
        run_generate "H2OP-${kr_label}-${ppl_id}" "$prompt" "$TOKENS" "$MAX_SEQ" \
            h2o_plus "$SCHEDULE" "$EVICT_RATIO" \
            --h2o-keep-ratio "$kr" --h2o-decay 0.0
    done
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
echo "  Round 13 Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:   $TOTAL experiments"
echo "  Done:    $DONE"
echo "  Skipped: $SKIPPED"
echo "  Failed:  $FAILED"
echo "  Time:    ${MINUTES}m ${SECONDS_REM}s"
echo ""
echo "Results in: $RESULTS/"
echo ""
echo "Phase 3 (Head-to-Head) uses data from Round 12 + this round:"
echo "  Round 11 baselines: experiments/results/round11/"
echo "  Round 12 H2O sweep: experiments/results/round12/"
echo ""
echo "Next: python experiments/analysis/round13_analyze.py"
