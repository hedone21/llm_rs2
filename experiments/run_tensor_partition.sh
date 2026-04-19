#!/bin/bash
# Tensor Partition benchmark — Galaxy S25 / Qwen 2.5-1.5B Q4_0 / OpenCL
#
# Measures avg_tbt_ms across ratio ∈ {1.0, 0.875, 0.75} × prefill ∈ {128, 1024, 4096}
# 2 reps per cell = 18 runs total (+ 1 warmup).
#
# Usage:
#   bash experiments/run_tensor_partition.sh
#
# Results are pulled to experiments/results/tensor_partition/r{R}_p{P}_run{N}.jsonl
# Analysis: python experiments/analysis/tensor_partition_report.py
#
# NOTE: Do NOT add --profile — it adds ~54 ms/token sync overhead (CLAUDE.md).
set -euo pipefail

DEVICE=galaxy_s25
MODEL=/data/local/tmp/models/qwen2.5-1.5b
OUT=/data/local/tmp/tp_results
LOCAL_OUT=experiments/results/tensor_partition
LOG_FILE=${LOCAL_OUT}/run.log

RATIOS=(1.0 0.875 0.75)   # GPU-only first → warm baseline
PREFILL=(128 1024 4096)
REPS=2

# ── Thermal cooldown policy ──────────────────────────────────────────────────
THERMAL_ZONE=/sys/class/thermal/thermal_zone0/temp
THERMAL_TARGET_MC=42000   # 42°C — idle ~40°C + 2°C margin (docs/31_perf_comparison_llama_cpp.md)
THERMAL_MAX_WAIT=300      # abort if not cool after 5 min
COOLDOWN_MIN_SHORT=30     # prefill ≤ 1024: min 30s cooldown
COOLDOWN_MIN_LONG=90      # prefill = 4096: min 90s (bench_cross_model_gpu.sh convention)

wait_for_cool() {
  local min_wait=$1
  echo "[cooldown] sleeping ${min_wait}s (minimum)" | tee -a "${LOG_FILE}"
  sleep "${min_wait}"
  local waited=${min_wait}
  while (( waited < THERMAL_MAX_WAIT )); do
    local t
    t=$(adb shell cat "${THERMAL_ZONE}" | tr -d '\r')
    echo "[cooldown] t=${t} mC (target<${THERMAL_TARGET_MC}), waited=${waited}s" | tee -a "${LOG_FILE}"
    if (( t < THERMAL_TARGET_MC )); then
      echo "[cooldown] OK" | tee -a "${LOG_FILE}"
      return 0
    fi
    sleep 10
    waited=$((waited + 10))
  done
  echo "[cooldown] ABORT: did not reach target in ${THERMAL_MAX_WAIT}s" | tee -a "${LOG_FILE}"
  exit 1
}

# ── adb_run: execute single generate run on device ───────────────────────────
# Globals used: MODEL, OUT, r, p, rep
adb_run() {
  local tag="r${r}_p${p}_run${rep}"
  local device_out="${OUT}/${tag}.jsonl"
  echo "" | tee -a "${LOG_FILE}"
  echo "[run] ${tag} — ratio=${r}, prefill=${p}, rep=${rep}" | tee -a "${LOG_FILE}"
  echo "[run] start $(date '+%Y-%m-%dT%H:%M:%S')" | tee -a "${LOG_FILE}"

  adb shell "cd /data/local/tmp && ./generate \
    --model ${MODEL} -b opencl --threads 6 \
    --prompt-file prompts/prefill_${p}.txt \
    --tensor-partition ${r} -n 128 \
    --experiment-output ${device_out} \
    --experiment-sample-interval 10 \
    --experiment-logits-topk 0" 2>&1 | tee -a "${LOG_FILE}"

  echo "[run] pulling ${tag}.jsonl" | tee -a "${LOG_FILE}"
  adb pull "${device_out}" "${LOCAL_OUT}/${tag}.jsonl" 2>&1 | tee -a "${LOG_FILE}"
}

# ── Warmup adb_run (result discarded) ────────────────────────────────────────
adb_run_warmup() {
  echo "" | tee -a "${LOG_FILE}"
  echo "[warmup] ratio=1.0, prefill=128 — filling OpenCL JIT cache" | tee -a "${LOG_FILE}"
  adb shell "cd /data/local/tmp && ./generate \
    --model ${MODEL} -b opencl --threads 6 \
    --prompt-file prompts/prefill_128.txt \
    --tensor-partition 1.0 -n 128 \
    --experiment-output ${OUT}/warmup.jsonl \
    --experiment-sample-interval 10 \
    --experiment-logits-topk 0" 2>&1 | tee -a "${LOG_FILE}" || true
}

# ── Main ─────────────────────────────────────────────────────────────────────
mkdir -p "${LOCAL_OUT}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee "${LOG_FILE}"
echo "  Tensor Partition Benchmark — Galaxy S25" | tee -a "${LOG_FILE}"
echo "  Device:  ${DEVICE}" | tee -a "${LOG_FILE}"
echo "  Model:   ${MODEL}" | tee -a "${LOG_FILE}"
echo "  Ratios:  ${RATIOS[*]}" | tee -a "${LOG_FILE}"
echo "  Prefill: ${PREFILL[*]}" | tee -a "${LOG_FILE}"
echo "  Reps:    ${REPS}" | tee -a "${LOG_FILE}"
echo "  Start:   $(date '+%Y-%m-%dT%H:%M:%S')" | tee -a "${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"

# ── 1. Build + deploy (binary push, run --help to trigger adb push only) ─────
echo "" | tee -a "${LOG_FILE}"
echo "[setup] Building and deploying binary to ${DEVICE}..." | tee -a "${LOG_FILE}"
python scripts/run_device.py -d "${DEVICE}" generate --help >/dev/null
echo "[setup] Binary deployed." | tee -a "${LOG_FILE}"

# ── 2. Push prompt files to device ───────────────────────────────────────────
echo "[setup] Pushing prompt files..." | tee -a "${LOG_FILE}"
adb shell "mkdir -p /data/local/tmp/prompts"
for p_len in "${PREFILL[@]}"; do
  adb push "experiments/prompts/prefill_${p_len}.txt" \
    "/data/local/tmp/prompts/prefill_${p_len}.txt" 2>&1 | tee -a "${LOG_FILE}"
done

# ── 3. Create output dir on device ───────────────────────────────────────────
adb shell "mkdir -p ${OUT}"

# ── 4. Warmup run ─────────────────────────────────────────────────────────────
adb_run_warmup
wait_for_cool ${COOLDOWN_MIN_SHORT}

# ── 5. Main sweep: ratio × prefill × rep ─────────────────────────────────────
for r in "${RATIOS[@]}"; do
  for p in "${PREFILL[@]}"; do
    for rep in $(seq 1 "${REPS}"); do
      adb_run

      # Cooldown before next run — scale by prefill length
      if (( p >= 4096 )); then
        wait_for_cool ${COOLDOWN_MIN_LONG}
      else
        wait_for_cool ${COOLDOWN_MIN_SHORT}
      fi
    done
  done
done

# ── 6. Summary ────────────────────────────────────────────────────────────────
echo "" | tee -a "${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"
echo "  Tensor Partition Benchmark Complete" | tee -a "${LOG_FILE}"
echo "  End:     $(date '+%Y-%m-%dT%H:%M:%S')" | tee -a "${LOG_FILE}"
echo "  Results: ${LOCAL_OUT}/" | tee -a "${LOG_FILE}"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━" | tee -a "${LOG_FILE}"
echo "" | tee -a "${LOG_FILE}"
echo "Next: python experiments/analysis/tensor_partition_report.py" | tee -a "${LOG_FILE}"
