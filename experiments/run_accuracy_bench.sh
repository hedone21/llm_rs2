#!/bin/bash
# Accuracy Benchmark: Base vs Sliding Window vs H2O
#
# Tests eviction policy accuracy on both Llama 3.2 1B and 3B models.
# Runs on host (CPU) with greedy sampling for reproducibility.
#
# Metrics: EMR, FDT, ROUGE-L, BLEU-4, Top-K Overlap
# Analysis via: experiments/analysis/compare.py

set -euo pipefail

BINARY="./target/release/generate"
RESULTS="experiments/results/accuracy_bench"
CONFIGS="experiments/configs"
ANALYSIS="experiments/analysis"

mkdir -p "$RESULTS"

# ============================================================
#  Counters
# ============================================================
TOTAL=0
DONE=0
FAILED=0
SKIPPED=0
START_TIME=$(date +%s)

# ============================================================
#  Models
# ============================================================
declare -A MODELS
MODELS[1B]="models/llama3.2-1b"
MODELS[3B]="models/llama3.2-3b"

# ============================================================
#  Prompts
# ============================================================
PPL01="It was a bright cold day in April, and the clocks were striking thirteen. The street was deserted except for a few figures hurrying along the pavement, their coat collars turned up against the wind that swept down from the grey towers."
PPL03="A hash table is a data structure that implements an associative array, mapping keys to values. It uses a hash function to compute an index into an array of buckets or slots, from which the desired value can be found. Ideally, the hash function will assign each key to a unique bucket."

declare -A PPL_PROMPTS
PPL_PROMPTS[PPL01]="$PPL01"
PPL_PROMPTS[PPL03]="$PPL03"

# ============================================================
#  Parameters
# ============================================================
TOKENS=256
MAX_SEQ=512
SCHEDULE="$CONFIGS/memory_critical_128.json"
EVICT_RATIO="0.50"

# ============================================================
#  Runner
# ============================================================
run_generate() {
    local name="$1"
    local model="$2"
    local prompt="$3"
    local eviction="$4"  # none | sliding | h2o | h2o_plus
    shift 4
    local extra_args=("$@")

    TOTAL=$((TOTAL + 1))
    local outfile="$RESULTS/${name}.jsonl"

    if [[ -f "$outfile" ]]; then
        echo "  SKIP $name (already exists)"
        SKIPPED=$((SKIPPED + 1))
        return
    fi

    local cmd=("$BINARY" -m "$model" -p "$prompt" -n "$TOKENS"
        --max-seq-len "$MAX_SEQ" --greedy
        --experiment-output "$outfile"
        --experiment-sample-interval 10
        --experiment-logits-topk 10)

    case "$eviction" in
        none)
            cmd+=(--eviction-policy none)
            ;;
        sliding)
            cmd+=(--eviction-policy sliding --eviction-window "$MAX_SEQ")
            cmd+=(--experiment-schedule "$SCHEDULE")
            cmd+=(--experiment-eviction-ratio "$EVICT_RATIO")
            ;;
        h2o)
            cmd+=(--eviction-policy h2o)
            cmd+=(--experiment-schedule "$SCHEDULE")
            cmd+=(--experiment-eviction-ratio "$EVICT_RATIO")
            cmd+=(--h2o-keep-ratio 0.5 --h2o-decay 0.0)
            ;;
        h2o_plus)
            cmd+=(--eviction-policy h2o_plus)
            cmd+=(--experiment-schedule "$SCHEDULE")
            cmd+=(--experiment-eviction-ratio "$EVICT_RATIO")
            cmd+=(--h2o-keep-ratio 0.5 --h2o-decay 0.0)
            ;;
    esac

    if [[ ${#extra_args[@]} -gt 0 ]]; then
        cmd+=("${extra_args[@]}")
    fi

    echo "  RUN  $name ($eviction, model=$model)"
    local run_start=$(date +%s)

    if "${cmd[@]}" > /dev/null 2>&1; then
        local run_end=$(date +%s)
        local elapsed=$((run_end - run_start))
        echo "  DONE $name (${elapsed}s)"
        DONE=$((DONE + 1))
    else
        echo "  FAIL $name"
        FAILED=$((FAILED + 1))
    fi
}

# ============================================================
#  Run all experiments
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Accuracy Benchmark: Base vs Sliding vs H2O vs H2O+"
echo "  Models: 1B, 3B | Tokens: $TOKENS | Max: $MAX_SEQ"
echo "  Eviction at: decode 128 | Ratio: $EVICT_RATIO"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

for model_id in 1B 3B; do
    model_path="${MODELS[$model_id]}"

    if [[ ! -d "$model_path" ]]; then
        echo "  WARN: Model not found: $model_path — skipping $model_id"
        continue
    fi

    echo "── Model: $model_id ($model_path) ──"

    for ppl_id in PPL01 PPL03; do
        prompt="${PPL_PROMPTS[$ppl_id]}"

        # 1. Baseline (no eviction)
        run_generate "${model_id}-BASE-${ppl_id}" "$model_path" "$prompt" none

        # 2. Sliding Window
        run_generate "${model_id}-SLIDE-${ppl_id}" "$model_path" "$prompt" sliding

        # 3. H2O
        run_generate "${model_id}-H2O-${ppl_id}" "$model_path" "$prompt" h2o

        # 4. H2O+
        run_generate "${model_id}-H2OP-${ppl_id}" "$model_path" "$prompt" h2o_plus
    done

    echo ""
done

# ============================================================
#  Analysis
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Analysis"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

REPORT_FILE="$RESULTS/accuracy_report.md"
echo "# Accuracy Benchmark: Base vs Sliding vs H2O" > "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "Generated: $(date -Iseconds)" >> "$REPORT_FILE"
echo "" >> "$REPORT_FILE"
echo "| Model | Prompt | Policy | EMR | FDT | ROUGE-L | BLEU-4 | Top-K Overlap | Evictions |" >> "$REPORT_FILE"
echo "|-------|--------|--------|-----|-----|---------|--------|---------------|-----------|" >> "$REPORT_FILE"

for model_id in 1B 3B; do
    for ppl_id in PPL01 PPL03; do
        baseline="$RESULTS/${model_id}-BASE-${ppl_id}.jsonl"

        if [[ ! -f "$baseline" ]]; then
            continue
        fi

        # Baseline row (self-comparison → perfect scores)
        echo "| $model_id | $ppl_id | Base | 1.000 | N/A | 1.000 | 1.000 | 1.000 | 0 |" >> "$REPORT_FILE"

        for policy in SLIDE H2O H2OP; do
            exp_file="$RESULTS/${model_id}-${policy}-${ppl_id}.jsonl"

            if [[ ! -f "$exp_file" ]]; then
                echo "  WARN: Missing $exp_file"
                continue
            fi

            echo "  COMPARE: ${model_id}-${policy}-${ppl_id} vs baseline"
            report_out=$(python3 "$ANALYSIS/compare.py" \
                --baseline "$baseline" \
                --experiment "$exp_file" 2>/dev/null || echo "ANALYSIS_ERROR")

            if [[ "$report_out" == "ANALYSIS_ERROR" ]]; then
                echo "  WARN: Analysis failed for ${model_id}-${policy}-${ppl_id}"
                echo "| $model_id | $ppl_id | $policy | ERR | ERR | ERR | ERR | ERR | ERR |" >> "$REPORT_FILE"
                continue
            fi

            # Extract metrics from compare.py output
            emr=$(echo "$report_out" | grep "Exact Match Rate:" | awk '{print $4}')
            fdt=$(echo "$report_out" | grep "First Divergent Token:" | awk '{print $4}')
            rouge=$(echo "$report_out" | grep "ROUGE-L F1:" | awk '{print $3}')
            bleu=$(echo "$report_out" | grep "BLEU-4:" | awk '{print $2}')
            topk=$(echo "$report_out" | grep "Top-K Overlap (avg):" | awk '{print $4}')
            evictions=$(echo "$report_out" | grep "Evictions:" | awk '{print $2}')

            echo "| $model_id | $ppl_id | $policy | $emr | $fdt | $rouge | $bleu | $topk | $evictions |" >> "$REPORT_FILE"
        done
    done
done

echo "" >> "$REPORT_FILE"
echo "## Parameters" >> "$REPORT_FILE"
echo "- Decode tokens: $TOKENS" >> "$REPORT_FILE"
echo "- Max seq len: $MAX_SEQ" >> "$REPORT_FILE"
echo "- Eviction trigger: decode token 128 (memory_critical)" >> "$REPORT_FILE"
echo "- Eviction ratio: $EVICT_RATIO (keep 50%)" >> "$REPORT_FILE"
echo "- H2O: keep_ratio=0.5, decay=0.0, time-normalized (default)" >> "$REPORT_FILE"
echo "- Sampling: greedy (temperature=0)" >> "$REPORT_FILE"
echo "- Backend: CPU (host)" >> "$REPORT_FILE"

# ============================================================
#  Summary
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
MINUTES=$((ELAPSED / 60))
SECONDS_REM=$((ELAPSED % 60))

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Benchmark Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:   $TOTAL experiments"
echo "  Done:    $DONE"
echo "  Skipped: $SKIPPED"
echo "  Failed:  $FAILED"
echo "  Time:    ${MINUTES}m ${SECONDS_REM}s"
echo ""
echo "  Results: $RESULTS/"
echo "  Report:  $REPORT_FILE"
echo ""
cat "$REPORT_FILE"
