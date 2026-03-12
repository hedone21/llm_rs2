#!/bin/bash
# Downstream Task Accuracy v2: H2O Paper Reproduction
#
# H2O paper tasks (COPA, PiQA, Winogrande, OpenBookQA, RTE, MathQA)
# Policies: baseline, sliding, h2o, h2o_plus
# Budget: ratio-based (20%, 50%, 80% of prompt length)
# KV cache: f16 (no quantization)
# Models: 1B + 3B
# No sleep between experiments.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
BENCHMARKS_DIR="$SCRIPT_DIR/benchmarks"
RESULTS_DIR="$BENCHMARKS_DIR/results/v2"
EVAL_SCRIPT="$BENCHMARKS_DIR/run_eval.py"
PREPARE_SCRIPT="$BENCHMARKS_DIR/prepare_datasets.py"

mkdir -p "$RESULTS_DIR"

TASKS="copa,piqa,winogrande,openbookqa,rte,mathqa"
POLICIES="none,sliding,h2o,h2o_plus"
RATIOS=(0.2 0.5 0.8)
KV_TYPE="f16"
MAX_SEQ=2048
N_QUESTIONS=100

# ============================================================
#  Step 1: Build
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 1: Building release binary"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
cd "$PROJECT_ROOT"
cargo build --release -p llm_rs2 --bin generate 2>&1 | tail -5
echo ""

# ============================================================
#  Step 2: Prepare datasets
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 2: Preparing H2O benchmark datasets"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
python3 "$PREPARE_SCRIPT" --tasks h2o --n-questions "$N_QUESTIONS"
echo ""

# ============================================================
#  Step 3: Run evaluations
# ============================================================
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Step 3: Running evaluations"
echo "  Tasks: $TASKS"
echo "  Policies: $POLICIES"
echo "  Budget ratios: ${RATIOS[*]}"
echo "  KV type: $KV_TYPE"
echo "  Models: 1B, 3B"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

START_TIME=$(date +%s)
TOTAL=0
DONE=0
FAILED=0

declare -A MODELS
MODELS[1B]="models/llama3.2-1b"
MODELS[3B]="models/llama3.2-3b"

for model_id in 1B 3B; do
    model_path="${MODELS[$model_id]}"

    if [[ ! -d "$model_path" ]]; then
        echo "  WARN: Model not found: $model_path — skipping $model_id"
        continue
    fi

    echo "══ Model: $model_id ($model_path) ══"

    # ── Baseline (no eviction, full KV cache) ──
    echo "  ── Baseline (full KV) ──"
    TOTAL=$((TOTAL + 1))
    out="$RESULTS_DIR/${model_id}_baseline.json"
    if [[ -f "$out" ]]; then
        echo "  SKIP baseline (exists)"
    elif python3 "$EVAL_SCRIPT" \
            --model "$model_path" \
            --tasks "$TASKS" \
            --policies none \
            --kv-type "$KV_TYPE" \
            --max-seq-len "$MAX_SEQ" \
            --n-questions "$N_QUESTIONS" \
            --output "$out" 2>&1; then
        echo "  DONE baseline"
        DONE=$((DONE + 1))
    else
        echo "  FAIL baseline"
        FAILED=$((FAILED + 1))
    fi

    # ── Eviction policies at each budget ratio ──
    for ratio in "${RATIOS[@]}"; do
        ratio_pct=$(echo "$ratio" | awk '{printf "%d", $1*100}')
        echo ""
        echo "  ── Budget ratio: ${ratio_pct}% ──"

        for policy in sliding h2o h2o_plus; do
            TOTAL=$((TOTAL + 1))
            out="$RESULTS_DIR/${model_id}_${policy}_r${ratio_pct}.json"

            if [[ -f "$out" ]]; then
                echo "  SKIP ${policy} r${ratio_pct}% (exists)"
                continue
            fi

            echo "  RUN  ${model_id} / ${policy} / r${ratio_pct}%"
            run_start=$(date +%s)

            if python3 "$EVAL_SCRIPT" \
                    --model "$model_path" \
                    --tasks "$TASKS" \
                    --policies "$policy" \
                    --kv-budget-ratio "$ratio" \
                    --kv-type "$KV_TYPE" \
                    --max-seq-len "$MAX_SEQ" \
                    --n-questions "$N_QUESTIONS" \
                    --output "$out" 2>&1; then
                run_end=$(date +%s)
                elapsed=$((run_end - run_start))
                echo "  DONE ${policy} r${ratio_pct}% (${elapsed}s)"
                DONE=$((DONE + 1))
            else
                echo "  FAIL ${policy} r${ratio_pct}%"
                FAILED=$((FAILED + 1))
            fi
        done
    done

    echo ""
done

# ============================================================
#  Step 4: Generate summary
# ============================================================
END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))
HOURS=$((ELAPSED / 3600))
MINUTES=$(( (ELAPSED % 3600) / 60 ))

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Downstream v2 Complete"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total:   $TOTAL runs"
echo "  Done:    $DONE"
echo "  Failed:  $FAILED"
echo "  Time:    ${HOURS}h ${MINUTES}m"
echo "  Results: $RESULTS_DIR/"
echo ""

# Generate consolidated summary from all result files
python3 -c "
import json, glob, os

results_dir = '$RESULTS_DIR'
files = sorted(glob.glob(os.path.join(results_dir, '*.json')))

if not files:
    print('No result files found.')
    exit()

# Collect all results
rows = []
for f in files:
    with open(f) as fh:
        data = json.load(fh)
    for key, r in data.items():
        rows.append(r)

# Print H2O-style summary table
tasks = ['copa', 'piqa', 'winogrande', 'openbookqa', 'rte', 'mathqa']
task_short = {'copa': 'COPA', 'piqa': 'PiQA', 'winogrande': 'Wino', 'openbookqa': 'OBQA', 'rte': 'RTE', 'mathqa': 'MathQA'}

models = sorted(set(os.path.basename(r['model']) for r in rows))

for model in models:
    print(f'\n=== {model} ===')
    header = f\"{'Policy':<12} {'Budget':<8}\"
    for t in tasks:
        header += f' {task_short.get(t,t):>7}'
    header += f' {\"Avg\":>7}'
    print(header)
    print('-' * len(header))

    model_rows = [r for r in rows if os.path.basename(r['model']) == model]

    # Group by (policy, budget_ratio)
    configs = set()
    for r in model_rows:
        ratio = r.get('kv_budget_ratio', 0)
        configs.add((r['policy'], ratio))

    for policy, ratio in sorted(configs, key=lambda x: (x[0], x[1])):
        budget_str = 'full' if ratio == 0 else f'{int(ratio*100)}%'
        line = f'{policy:<12} {budget_str:<8}'
        accs = []
        for t in tasks:
            match = [r for r in model_rows if r['task'] == t and r['policy'] == policy and r.get('kv_budget_ratio', 0) == ratio]
            if match:
                acc = match[0]['accuracy']
                accs.append(acc)
                line += f' {acc:>6.1%}'
            else:
                line += f' {\"---\":>7}'
        if accs:
            avg = sum(accs) / len(accs)
            line += f' {avg:>6.1%}'
        print(line)
"
