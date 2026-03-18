#!/bin/bash
# Proxy Validation: Additional eviction policies (D2O, Streaming, H2O+)
# Usage: ./run_sweep_extra.sh [1b|3b]
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

echo "=== Extra Policies: $MODEL_SIZE ==="

# Phase 3a: D2O sweep
echo "--- Phase 3a: D2O ---"
for budget in 1600 1200 900 700 500 350; do
  echo -n "  budget=$budget: "
  $GENERATE --model-path $MODEL --ppl $TEXT \
    --backend cpu --kv-type f32 --temperature 0 \
    --eviction-policy d2o --d2o-keep-ratio 0.75 \
    --protected-prefix 4 --kv-budget $budget \
    --kv-layout head --max-seq-len $MAX_SEQ \
    > "$OUTDIR/d2o_${budget}.json" 2>"$OUTDIR/d2o_${budget}.log"
  python3 -c "import json; d=json.load(open('$OUTDIR/d2o_${budget}.json')); print(f'PPL={d[\"ppl\"]:.4f}, evictions={d[\"eviction_count\"]}, proxy_avg={sum(m[\"raw_value\"] for m in d[\"proxy_metrics\"])/max(len(d[\"proxy_metrics\"]),1):.4f}')"
done

# Phase 3b: Streaming (sink + sliding)
echo ""
echo "--- Phase 3b: Streaming ---"
for budget in 1600 1200 900 700 500 350; do
  echo -n "  budget=$budget: "
  $GENERATE --model-path $MODEL --ppl $TEXT \
    --backend cpu --kv-type f32 --temperature 0 \
    --eviction-policy streaming --sink-size 4 \
    --eviction-window $budget --kv-budget $budget \
    --kv-layout head --max-seq-len $MAX_SEQ \
    > "$OUTDIR/streaming_${budget}.json" 2>"$OUTDIR/streaming_${budget}.log"
  python3 -c "import json; d=json.load(open('$OUTDIR/streaming_${budget}.json')); print(f'PPL={d[\"ppl\"]:.4f}, evictions={d[\"eviction_count\"]}, proxy_avg={sum(m[\"raw_value\"] for m in d[\"proxy_metrics\"])/max(len(d[\"proxy_metrics\"]),1):.4f}')"
done

# Phase 3c: H2O+ (per-head GQA-aware)
echo ""
echo "--- Phase 3c: H2O+ ---"
for budget in 1600 1200 900 700 500 350; do
  echo -n "  budget=$budget: "
  $GENERATE --model-path $MODEL --ppl $TEXT \
    --backend cpu --kv-type f32 --temperature 0 \
    --eviction-policy h2o_plus --h2o-keep-ratio 0.5 \
    --protected-prefix 4 --kv-budget $budget \
    --kv-layout head --max-seq-len $MAX_SEQ \
    > "$OUTDIR/h2o_plus_${budget}.json" 2>"$OUTDIR/h2o_plus_${budget}.log"
  python3 -c "import json; d=json.load(open('$OUTDIR/h2o_plus_${budget}.json')); print(f'PPL={d[\"ppl\"]:.4f}, evictions={d[\"eviction_count\"]}, proxy_avg={sum(m[\"raw_value\"] for m in d[\"proxy_metrics\"])/max(len(d[\"proxy_metrics\"]),1):.4f}')"
done

echo ""
echo "=== Done: $MODEL_SIZE ==="
