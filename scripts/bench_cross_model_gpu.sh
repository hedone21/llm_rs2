#!/usr/bin/env bash
# Cross-model GPU bench: llm.rs vs llama.cpp (-ngl 99)
# Models: llama3.2-3b, gemma3-4b (Q4_0)
# Contexts: 256, 1024, 2048, 6k
# 90s cooldown between runs
set -euo pipefail

OUTDIR="${OUTDIR:-.agent/research/2026-04-16_cross_model_gpu_bench}"
mkdir -p "$OUTDIR"
NGEN="${NGEN:-32}"
COOLDOWN="${COOLDOWN:-90}"

read_max_thermal() {
  adb shell 'for z in 1 10 28 30; do cat /sys/class/thermal/thermal_zone$z/temp 2>/dev/null; done' \
    | tr -d '\r' | sort -nr | head -1
}

log_thermal() {
  echo "=== THERMAL $1 $(date +%H:%M:%S) === max=$(read_max_thermal)mC" >> "$OUTDIR/thermal_log.txt"
}

cooldown() {
  echo "[cooldown] ${COOLDOWN}s..."
  sleep "$COOLDOWN"
}

run_llama() {
  local tag="$1" model="$2" prompt="$3" ctx="$4"
  local out="$OUTDIR/llama_${tag}.log"
  log_thermal "pre-llama_$tag"
  cooldown
  log_thermal "llama_${tag} START"
  echo "[$(date +%H:%M:%S)] RUN llama_$tag model=$model ctx=$ctx"
  adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-cli-orig \
      -m $model -f $prompt -n $NGEN --temp 0 -ngl 99 --single-turn --no-warmup -c $ctx 2>&1 < /dev/null" \
    | tr -d '\r' > "$out"
  log_thermal "llama_${tag} END"
  grep "llama_perf_context_print" "$out" | tail -5
}

run_llmrs() {
  local tag="$1" model_dir="$2" prompt="$3" ctx="$4"
  local out="$OUTDIR/llmrs_${tag}.log"
  log_thermal "pre-llmrs_$tag"
  cooldown
  log_thermal "llmrs_${tag} START"
  echo "[$(date +%H:%M:%S)] RUN llmrs_$tag model=$model_dir ctx=$ctx"
  adb shell "cd /data/local/tmp && timeout 300 ./generate -m $model_dir/model.gguf --prompt-file $prompt \
      --num-tokens $NGEN --temperature 0 --backend opencl --max-seq-len $ctx 2>&1 < /dev/null" \
    | tr -d '\r' > "$out"
  log_thermal "llmrs_${tag} END"
  grep -E "Decode|Prefill|Prompt|tok/s" "$out" | tail -10
}

: > "$OUTDIR/thermal_log.txt"
echo "# Cross-model GPU bench — $(date)" > "$OUTDIR/thermal_log.txt"

# === Llama 3.2 3B ===
echo "======== Llama 3.2 3B Q4_0 ========"
for tag in 256:320 1024:1088 2048:2112 6k:4544; do
  T="${tag%:*}"; C="${tag#*:}"
  run_llama "3b_$T" "llama3.2-3b-gguf/model.gguf" "prompts_v2/prompt_${T}.txt" "$C"
  run_llmrs "3b_$T" "llama3.2-3b-gguf" "prompts_v2/prompt_${T}.txt" "$C"
done

# === Gemma 3 4B ===
echo "======== Gemma 3 4B Q4_0 ========"
for tag in 256:320 1024:1088 2048:2112 6k:4544; do
  T="${tag%:*}"; C="${tag#*:}"
  run_llama "4b_$T" "gemma3-4b-gguf/model.gguf" "prompts_v2/prompt_${T}.txt" "$C"
  run_llmrs "4b_$T" "gemma3-4b-gguf" "prompts_v2/prompt_${T}.txt" "$C"
done

echo "[$(date +%H:%M:%S)] DONE"
