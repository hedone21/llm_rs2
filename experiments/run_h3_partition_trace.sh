#!/bin/bash
# H3: tensor partition internal timing trace.
# Runs a single decode at ratio=0.875 with LLMRS_PARTITION_TRACE=1.
# Prints per-layer (read_buf, cpu_matmul, gpu_wait, merge) averages every
# 280 layers (≈10 tokens at 28 layers), allowing us to tell whether the
# CPU or GPU is the bottleneck in the partition FFN path.
#
# Requires: new binary built with the instrumentation patch deployed to
# device. Device must be cooled (< 42°C max CPU zone) before running.
#
# Usage:
#   bash experiments/run_h3_partition_trace.sh
set -euo pipefail

DEVICE=galaxy_s25
MODEL=/data/local/tmp/models/qwen2.5-1.5b
OUT=/data/local/tmp/tp_results
LOCAL_OUT=experiments/results/tensor_partition
LOG_FILE=${LOCAL_OUT}/h3_trace.log
RATIO=0.875
PREFILL=1024
N=128

mkdir -p "${LOCAL_OUT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee "${LOG_FILE}"
echo "  H3: Tensor Partition Internal Trace" | tee -a "${LOG_FILE}"
echo "  Ratio:   ${RATIO}  Prefill: ${PREFILL}  N: ${N}" | tee -a "${LOG_FILE}"
echo "  Start:   $(date '+%Y-%m-%dT%H:%M:%S')" | tee -a "${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"

# Deploy new binary (assumes caller already built)
echo "[setup] deploying instrumented binary..." | tee -a "${LOG_FILE}"
adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate 2>&1 | tee -a "${LOG_FILE}"
adb shell "chmod +x /data/local/tmp/generate"
adb shell "mkdir -p ${OUT}"

# Warmup (kernel cache + throwaway data, no trace)
echo "" | tee -a "${LOG_FILE}"
echo "[warmup] r=${RATIO} p=${PREFILL} (trace off)" | tee -a "${LOG_FILE}"
adb shell "cd /data/local/tmp && ./generate \
  --model-path ${MODEL} -b opencl --threads 6 \
  --prompt-file prompts/prefill_${PREFILL}.txt \
  --tensor-partition ${RATIO} -n 32 --ignore-eos \
  --experiment-output ${OUT}/h3_warmup.jsonl \
  --experiment-sample-interval 100 \
  --experiment-logits-topk 0" 2>&1 | tail -20 | tee -a "${LOG_FILE}" || true

# Cool down briefly
echo "" | tee -a "${LOG_FILE}"
echo "[cooldown] 30s..." | tee -a "${LOG_FILE}"
sleep 30

# Real trace run
echo "" | tee -a "${LOG_FILE}"
echo "[run] r=${RATIO} p=${PREFILL} n=${N} with LLMRS_PARTITION_TRACE=1" | tee -a "${LOG_FILE}"
adb shell "cd /data/local/tmp && LLMRS_PARTITION_TRACE=1 ./generate \
  --model-path ${MODEL} -b opencl --threads 6 \
  --prompt-file prompts/prefill_${PREFILL}.txt \
  --tensor-partition ${RATIO} -n ${N} --ignore-eos \
  --experiment-output ${OUT}/h3_trace.jsonl \
  --experiment-sample-interval 100 \
  --experiment-logits-topk 0" 2>&1 | tee -a "${LOG_FILE}"

echo "" | tee -a "${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"
echo "  H3 Trace Complete — ${LOG_FILE}" | tee -a "${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"
echo ""
echo "Extract [partition-trace] lines:"
echo "  grep '^\\[partition-trace\\]' ${LOG_FILE}"
