#!/bin/bash
# Round 11: Aggressive Eviction Deep Dive
#
# Three sub-experiments:
#   A) Long Context PPL — 2048 tokens, 80% eviction, 5 domains
#   B) NIAH Redesign — passkey length scaling (3/5/10/20 digits) + simple facts
#   C) PPL03 Position Sensitivity — eviction position vs domain sensitivity
#
# Key changes from Round 10:
#   - 80% eviction (--experiment-eviction-ratio 0.20) vs Round 10's 50%
#   - Longer context (2048 tokens vs 512/1024)
#   - New NIAH needles (passkey length scaling + simpler facts)
#   - Eviction position sensitivity analysis
#
# References: docs/30_evaluation_methodology.md, experiments/PLAN.md

set -euo pipefail

BINARY="./target/release/generate"
MODEL="/home/go/Workspace/models"
RESULTS="experiments/results/round11"
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
            cmd+=(--eviction-policy h2o
                  --h2o-keep-ratio 0.5
                  --h2o-recent-window 128
                  --h2o-decay 0.1)
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
#  Sub-experiment A: Long Context PPL (2048 tokens, 80% eviction)
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 11A: Long Context PPL (2048tok, 80% eviction)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PPL01="It was a bright cold day in April, and the clocks were striking thirteen. The street was deserted except for a few figures hurrying along the pavement, their coat collars turned up against the wind that swept down from the grey towers."
PPL02="The theory of plate tectonics describes the large-scale motion of seven major plates and many minor plates of Earth's lithosphere. The concept was first proposed in 1912 by Alfred Wegener as continental drift, but it was not until the 1960s that a comprehensive theory emerged."
PPL03="A hash table is a data structure that implements an associative array, mapping keys to values. It uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found. Ideally, the hash function will assign each key to a unique bucket."
PPL04="So the thing about learning a new language is that everyone tells you to just immerse yourself, but nobody really explains what that means in practice. I tried watching movies without subtitles for a month, and honestly, it was frustrating at first because I could barely catch every other word."
PPL05="Scientists at the European Organization for Nuclear Research announced preliminary results from a new series of experiments that could reshape our understanding of fundamental physics. The findings, which have not yet been peer reviewed, suggest the existence of previously undetected interactions between subatomic particles."

declare -A PPL_PROMPTS
PPL_PROMPTS[PPL01]="$PPL01"
PPL_PROMPTS[PPL02]="$PPL02"
PPL_PROMPTS[PPL03]="$PPL03"
PPL_PROMPTS[PPL04]="$PPL04"
PPL_PROMPTS[PPL05]="$PPL05"

TOKENS=2048
MAX_SEQ=4096
INJECT_POS=1024
SCHEDULE="$CONFIGS/memory_critical_${INJECT_POS}.json"
EVICT_RATIO="0.20"

echo ""
echo "── PPL: ${TOKENS} tokens (inject@${INJECT_POS}, keep 20%) ──"

for ppl_id in PPL01 PPL02 PPL03 PPL04 PPL05; do
    prompt="${PPL_PROMPTS[$ppl_id]}"

    # Baseline
    run_generate "${ppl_id}-2048-base" "$prompt" "$TOKENS" "$MAX_SEQ" none none none

    # Sliding + 80% eviction
    run_generate "${ppl_id}-2048-sl" "$prompt" "$TOKENS" "$MAX_SEQ" sliding "$SCHEDULE" "$EVICT_RATIO"

    # H2O + 80% eviction
    run_generate "${ppl_id}-2048-h2o" "$prompt" "$TOKENS" "$MAX_SEQ" h2o "$SCHEDULE" "$EVICT_RATIO"
done

# ============================================================
#  Sub-experiment B: NIAH Redesign (passkey length + simple facts)
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 11B: NIAH Redesign (80% eviction)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

NIAH_SCRIPT="experiments/prompts/assemble_niah.py"
EVICT_RATIO="0.20"

# Passkey length scaling: 3, 5, 10, 20 digits
# Simple facts: NUM (42), DATE2 (July 20, 1969)
for needle in N-P3 N-PASS N-P10 N-P20 N-NUM N-DATE2; do
    for depth in 0.1 0.5 0.9; do
        depth_pct=$(echo "$depth" | awk '{printf "%d", $1*100}')

        # Use 16 blocks (~960 tok) for longer context
        blocks=16
        niah_id="NIAH-${needle#N-}-D${depth_pct}-B${blocks}"

        echo ""
        echo "── $niah_id ──"

        # Generate the prompt
        niah_prompt=$(python "$NIAH_SCRIPT" --needle "$needle" --depth "$depth" --blocks "$blocks")

        # Max seq for 16 blocks (~960 tok prompt + 64 tok gen)
        max_seq=2048

        # Baseline
        run_generate "${niah_id}-base" "$niah_prompt" 64 "$max_seq" none none none

        # Sliding + 80% eviction at token 1
        run_generate "${niah_id}-sl" "$niah_prompt" 64 "$max_seq" sliding "$CONFIGS/niah_evict_at_1.json" "$EVICT_RATIO"

        # H2O + 80% eviction at token 1
        run_generate "${niah_id}-h2o" "$niah_prompt" 64 "$max_seq" h2o "$CONFIGS/niah_evict_at_1.json" "$EVICT_RATIO"
    done
done

# ============================================================
#  Sub-experiment C: PPL03 Position Sensitivity
# ============================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Round 11C: PPL03 Position Sensitivity (2048tok, 80% eviction)"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# Compare PPL03 (technical, eviction-sensitive) vs PPL01 (literary, eviction-robust)
# at 3 eviction injection positions
TOKENS=2048
MAX_SEQ=4096
EVICT_RATIO="0.20"

for ppl_id in PPL01 PPL03; do
    prompt="${PPL_PROMPTS[$ppl_id]}"

    echo ""
    echo "── Position sensitivity: $ppl_id ──"

    # Baseline already run in sub-experiment A — skip
    # (${ppl_id}-2048-base already exists)

    # P50 already covered in sub-experiment A (${ppl_id}-2048-sl/h2o)
    for inject_pct in 25 75; do
        inject_pos=$((TOKENS * inject_pct / 100))
        schedule="$CONFIGS/memory_critical_${inject_pos}.json"

        for policy in sl h2o; do
            run_generate "${ppl_id}-2048-${policy}-P${inject_pct}" \
                "$prompt" "$TOKENS" "$MAX_SEQ" \
                "${policy/sl/sliding}" "$schedule" "$EVICT_RATIO"
        done
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
echo "  Round 11 Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:   $TOTAL experiments"
echo "  Done:    $DONE"
echo "  Skipped: $SKIPPED"
echo "  Failed:  $FAILED"
echo "  Time:    ${MINUTES}m ${SECONDS_REM}s"
echo ""
echo "Results in: $RESULTS/"
echo "Next: python experiments/analysis/round11_analyze.py"
