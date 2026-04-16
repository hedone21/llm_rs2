#!/usr/bin/env bash
# Gemma 3 4B focused bench using llama-cli-new (gemma3 support)
set -euo pipefail

OUTDIR=".agent/research/2026-04-16_cross_model_gpu_bench"
NGEN=32
COOLDOWN=60

read_max_thermal() {
  adb shell 'for z in 1 10 28 30; do cat /sys/class/thermal/thermal_zone$z/temp 2>/dev/null; done' \
    | tr -d '\r' | sort -nr | head -1
}

log_thermal() {
  echo "=== THERMAL $1 $(date +%H:%M:%S) === max=$(read_max_thermal)mC" >> "$OUTDIR/thermal_log.txt"
}

run_llama_new() {
  local tag="$1" prompt="$2" ctx="$3"
  local out="$OUTDIR/llama_${tag}.log"
  log_thermal "pre-llama_$tag"
  sleep $COOLDOWN
  log_thermal "llama_${tag} START"
  echo "[$(date +%H:%M:%S)] RUN llama_$tag ctx=$ctx"
  adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp timeout 600 ./llama-cli-new \
      -m gemma3-4b-gguf/model.gguf -f $prompt -n $NGEN --temp 0 -ngl 99 --single-turn --no-warmup -c $ctx 2>&1 < /dev/null" \
    | tr -d '\r' > "$out"
  log_thermal "llama_${tag} END"
  grep -E "Prompt:|Generation:|t/s|llama_perf" "$out" | tail -5
}

run_llmrs() {
  local tag="$1" prompt="$2" ctx="$3"
  local out="$OUTDIR/llmrs_${tag}.log"
  log_thermal "pre-llmrs_$tag"
  sleep $COOLDOWN
  log_thermal "llmrs_${tag} START"
  echo "[$(date +%H:%M:%S)] RUN llmrs_$tag ctx=$ctx"
  adb shell "cd /data/local/tmp && timeout 600 ./generate -m gemma3-4b-gguf/model.gguf --prompt-file $prompt \
      --num-tokens $NGEN --temperature 0 --backend opencl --max-seq-len $ctx 2>&1 < /dev/null" \
    | tr -d '\r' > "$out"
  log_thermal "llmrs_${tag} END"
  grep -E "Decode|Prefill|TTFT" "$out" | tail -5
}

echo "======== Gemma 3 4B Q4_0 (llama-cli-new) ========"
for tag in 256:320 1024:1088 2048:2112 6k:4544; do
  T="${tag%:*}"; C="${tag#*:}"
  run_llama_new "4b_$T" "prompts_v2/prompt_${T}.txt" "$C" || echo "  [warn] llama_4b_$T failed"
  run_llmrs "4b_$T" "prompts_v2/prompt_${T}.txt" "$C" || echo "  [warn] llmrs_4b_$T failed"
done

echo "[$(date +%H:%M:%S)] DONE"
