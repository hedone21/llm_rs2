#!/bin/bash
# Proxy Validation PPL Sweep
# Usage: ./run_sweep.sh [1b|3b]
set -e

MODEL_SIZE=${1:-1b}
GENERATE=./target/release/generate
TEXT=experiments/proxy_validation/texts/eval_text.txt
OUTDIR=experiments/proxy_validation/results/${MODEL_SIZE}
mkdir -p "$OUTDIR"

case "$MODEL_SIZE" in
  1b) MODEL=models/llama3.2-1b; MAX_SEQ=2048 ;;
  3b) MODEL=models/llama3.2-3b; MAX_SEQ=2048 ;;
  *)  echo "Usage: $0 [1b|3b]"; exit 1 ;;
esac

echo "=== Proxy Validation: $MODEL_SIZE ==="
echo "Model: $MODEL"
echo "Text: $TEXT"
echo "Output: $OUTDIR"
echo ""

# Phase 1: Baseline (no eviction)
echo "--- Phase 1: Baseline ---"
$GENERATE --model-path $MODEL --ppl $TEXT \
  --backend cpu --kv-type f32 --temperature 0 \
  --eviction-policy none --kv-layout head --max-seq-len $MAX_SEQ \
  > "$OUTDIR/baseline.json" 2>"$OUTDIR/baseline.log"
echo "Baseline done: $(python3 -c "import json; d=json.load(open('$OUTDIR/baseline.json')); print(f'PPL={d[\"ppl\"]:.4f}, tokens={d[\"token_count\"]}')")"

# Phase 2a: Sliding Window sweep
echo ""
echo "--- Phase 2a: Sliding Window ---"
for budget in 1600 1200 900 700 500 350; do
  echo -n "  budget=$budget: "
  $GENERATE --model-path $MODEL --ppl $TEXT \
    --backend cpu --kv-type f32 --temperature 0 \
    --eviction-policy sliding --eviction-window $budget \
    --kv-budget $budget --kv-layout head --max-seq-len $MAX_SEQ \
    > "$OUTDIR/sliding_${budget}.json" 2>"$OUTDIR/sliding_${budget}.log"
  python3 -c "import json; d=json.load(open('$OUTDIR/sliding_${budget}.json')); print(f'PPL={d[\"ppl\"]:.4f}, evictions={d[\"eviction_count\"]}, proxy_avg={sum(m[\"raw_value\"] for m in d[\"proxy_metrics\"])/max(len(d[\"proxy_metrics\"]),1):.4f}')"
done

# Phase 2b: H2O sweep
echo ""
echo "--- Phase 2b: H2O ---"
for budget in 1600 1200 900 700 500 350; do
  echo -n "  budget=$budget: "
  $GENERATE --model-path $MODEL --ppl $TEXT \
    --backend cpu --kv-type f32 --temperature 0 \
    --eviction-policy h2o --h2o-keep-ratio 0.5 \
    --protected-prefix 4 --kv-budget $budget \
    --kv-layout head --max-seq-len $MAX_SEQ \
    > "$OUTDIR/h2o_${budget}.json" 2>"$OUTDIR/h2o_${budget}.log"
  python3 -c "import json; d=json.load(open('$OUTDIR/h2o_${budget}.json')); print(f'PPL={d[\"ppl\"]:.4f}, evictions={d[\"eviction_count\"]}, proxy_avg={sum(m[\"raw_value\"] for m in d[\"proxy_metrics\"])/max(len(d[\"proxy_metrics\"]),1):.4f}')"
done

echo ""
echo "=== Done: $MODEL_SIZE ==="
