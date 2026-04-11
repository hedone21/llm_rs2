#!/usr/bin/env bash
# Measures the TWO missing legs of the Qwen 4-way comparison:
#   1. llm.rs CPU
#   2. llama.cpp OpenCL GPU (via llama-cli-new --single-turn)
#
# The other two legs (llm.rs GPU, llama.cpp CPU) are reused from
# results/data/flash_attn_decode/thermal/c15a_qwen_strict_isolation.txt
# which was measured with the same strict protocol in Task 5.
#
# Same strict thermal methodology: 5-min rest, zombie preflight, zone monitoring.
# Total runtime: ~75 min (2 backends × 2 combos × 3 runs × 5-min rest).

set -euo pipefail
export LC_ALL=C

command -v adb >/dev/null 2>&1 || { echo "adb not on PATH" >&2; exit 2; }
adb get-state 1>/dev/null 2>&1 || { echo "no device connected" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="$REPO_ROOT/results/data/flash_attn_decode/thermal"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/qwen_missing_legs_strict.txt"

REST_SEC="${REST_SEC:-300}"
INTER_PHASE_REST="${INTER_PHASE_REST:-300}"
THERMAL_THRESHOLD_MC="${THERMAL_THRESHOLD_MC:-50000}"
N_RUNS="${N_RUNS:-3}"

zombie_check() {
  local zombies
  zombies=$(adb shell 'ps -A 2>/dev/null | grep -E "(llama-cli|llama-cli-new|llama-cli-orig|llama-bench|generate_master|generate[^a-z0-9])" | grep -v grep || true' \
    | tr -d '\r')
  if [ -n "$zombies" ]; then
    echo "[missing] ERROR: stale processes on device:" >&2
    echo "$zombies" >&2
    exit 3
  fi
}
zombie_check
echo "[missing] zombie check OK"

read_zones() {
  adb shell 'for z in 1 10 28 30; do cat /sys/class/thermal/thermal_zone$z/temp 2>/dev/null; done' \
    | tr -d '\r' | tr '\n' ' '
}

read_max_thermal() {
  read_zones | tr ' ' '\n' | grep -v '^$' | sort -nr | head -1
}

strict_rest() {
  local label="$1"
  local start_ts=$(date +%s)
  echo "  [rest] $label — ${REST_SEC}s..."
  local remain=$REST_SEC
  while [ $remain -gt 0 ]; do
    if [ $remain -gt 60 ]; then
      sleep 60
      remain=$((remain - 60))
      local t=$(read_max_thermal 2>/dev/null || echo "?")
      local elapsed=$(( $(date +%s) - start_ts ))
      echo "    elapsed=${elapsed}s max=${t}mC remaining=${remain}s"
    else
      sleep "$remain"
      remain=0
    fi
  done
  local t=$(read_max_thermal)
  if [ "$t" -gt "$THERMAL_THRESHOLD_MC" ]; then
    echo "  [rest] still ${t}mC above ${THERMAL_THRESHOLD_MC}mC — extra 60s"
    sleep 60
    t=$(read_max_thermal)
  fi
  echo "  [rest] done: max=${t}mC after $(( $(date +%s) - start_ts ))s"
}

# llm.rs CPU
run_llm_cpu() {
  local label="$1" promptarg="$2" ntok="$3"
  local zones_start zones_end out decode
  zones_start=$(read_zones)
  out=$(adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b ${promptarg} -n ${ntok} -b cpu 2>&1" \
    | tr -d '\r')
  zones_end=$(read_zones)
  decode=$(echo "$out" | awk '/^Decode:/ { gsub(/[^0-9.]/, "", $2); print $2; exit }')
  echo "    llm.rs CPU ${label}: ${decode}ms/tok [${zones_start}] -> [${zones_end}]" >&2
  echo "$decode"
}

# llama.cpp OpenCL GPU via llama-cli-new
run_llc_gpu() {
  local label="$1" promptarg="$2" ntok="$3"
  local zones_start zones_end out tps decode
  zones_start=$(read_zones)
  out=$(adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-cli-new -m Qwen2.5-1.5B-Instruct-f16.gguf ${promptarg} -n ${ntok} --temp 0 -ngl 99 --single-turn --no-warmup 2>&1 < /dev/null" \
    | tr -d '\r')
  zones_end=$(read_zones)
  # Parse "[ Prompt: 81.7 t/s | Generation: 20.2 t/s ]"
  tps=$(echo "$out" | awk '
    /Generation:/ {
      if (match($0, /Generation:[[:space:]]*[0-9]+\.[0-9]+[[:space:]]*t\/s/)) {
        s = substr($0, RSTART, RLENGTH);
        sub(/Generation:[[:space:]]*/, "", s);
        sub(/[[:space:]]*t\/s/, "", s);
        print s;
        exit;
      }
    }')
  decode=""
  if [ -n "$tps" ]; then
    decode=$(awk "BEGIN { printf \"%.2f\", 1000.0 / $tps }")
  fi
  echo "    llama.cpp GPU ${label}: ${decode}ms/tok [${zones_start}] -> [${zones_end}]" >&2
  echo "$decode"
}

median_of() {
  sort -n | awk '
    { a[NR] = $1 }
    END {
      if (NR == 0) { print "0" }
      else if (NR % 2 == 1) { print a[(NR+1)/2] }
      else { printf "%.2f\n", (a[NR/2] + a[NR/2+1]) / 2.0 }
    }'
}

# Same combos as existing strict bench
SHORT_LLM_PROMPT='--prompt "The quick brown fox jumps over"'
SHORT_LLC_PROMPT='-p "The quick brown fox jumps over"'
SHORT_NTOK=64

LONG_LLM_PROMPT='--prompt-file /data/local/tmp/long_prompt.txt'
LONG_LLC_PROMPT='-f /data/local/tmp/long_prompt.txt'
LONG_NTOK=128

{
  echo "# $(date) — Qwen missing legs (llm.rs CPU + llama.cpp GPU)"
  echo "# Device: Samsung Galaxy S25 (Adreno 830)"
  echo "# Reuses c15a_qwen_strict_isolation.txt for llm.rs GPU and llama.cpp CPU"
  echo "# Rest=${REST_SEC}s, inter-phase=${INTER_PHASE_REST}s, threshold=${THERMAL_THRESHOLD_MC}mC, runs=${N_RUNS}"
  echo "# Short: decode=${SHORT_NTOK}"
  echo "# Long:  decode=${LONG_NTOK}"
  echo
} > "$OUTFILE"

llc_gpu_short=(); llc_gpu_long=()
llm_cpu_short=(); llm_cpu_long=()

# -----------------------------------------------------------------
# Phase A: llama.cpp GPU (via llama-cli-new)
# -----------------------------------------------------------------
echo
echo "============================================================"
echo "Phase A: llama.cpp OpenCL GPU"
echo "============================================================"
{ echo; echo "=== Phase A: llama.cpp OpenCL GPU ==="; } >> "$OUTFILE"

for run in $(seq 1 "$N_RUNS"); do
  strict_rest "A-short-r${run}"
  v=$(run_llc_gpu "short-r${run}" "$SHORT_LLC_PROMPT" "$SHORT_NTOK")
  llc_gpu_short+=("$v")
  echo "  llc_gpu short r=${run}: ${v}ms" >> "$OUTFILE"

  strict_rest "A-long-r${run}"
  v=$(run_llc_gpu "long-r${run}" "$LONG_LLC_PROMPT" "$LONG_NTOK")
  llc_gpu_long+=("$v")
  echo "  llc_gpu long  r=${run}: ${v}ms" >> "$OUTFILE"
done

echo
echo "Inter-phase rest..."
{ echo; echo "--- inter-phase ---"; } >> "$OUTFILE"
REST_SEC="$INTER_PHASE_REST" strict_rest "A->B"

# -----------------------------------------------------------------
# Phase B: llm.rs CPU
# -----------------------------------------------------------------
echo
echo "============================================================"
echo "Phase B: llm.rs CPU"
echo "============================================================"
{ echo; echo "=== Phase B: llm.rs CPU ==="; } >> "$OUTFILE"

for run in $(seq 1 "$N_RUNS"); do
  strict_rest "B-short-r${run}"
  v=$(run_llm_cpu "short-r${run}" "$SHORT_LLM_PROMPT" "$SHORT_NTOK")
  llm_cpu_short+=("$v")
  echo "  llm_cpu short r=${run}: ${v}ms" >> "$OUTFILE"

  strict_rest "B-long-r${run}"
  v=$(run_llm_cpu "long-r${run}" "$LONG_LLM_PROMPT" "$LONG_NTOK")
  llm_cpu_long+=("$v")
  echo "  llm_cpu long  r=${run}: ${v}ms" >> "$OUTFILE"
done

# Medians
llc_gpu_short_med=$(printf "%s\n" "${llc_gpu_short[@]}" | median_of)
llc_gpu_long_med=$(printf "%s\n" "${llc_gpu_long[@]}" | median_of)
llm_cpu_short_med=$(printf "%s\n" "${llm_cpu_short[@]}" | median_of)
llm_cpu_long_med=$(printf "%s\n" "${llm_cpu_long[@]}" | median_of)

{
  echo
  echo "=== Medians (missing legs) ==="
  echo "  llama.cpp GPU short: ${llc_gpu_short_med} ms (runs: ${llc_gpu_short[*]})"
  echo "  llama.cpp GPU long:  ${llc_gpu_long_med} ms (runs: ${llc_gpu_long[*]})"
  echo "  llm.rs    CPU short: ${llm_cpu_short_med} ms (runs: ${llm_cpu_short[*]})"
  echo "  llm.rs    CPU long:  ${llm_cpu_long_med} ms (runs: ${llm_cpu_long[*]})"
} | tee -a "$OUTFILE"

echo
echo "[missing] Done. Results in $OUTFILE"
echo "[missing] Reuse existing data from c15a_qwen_strict_isolation.txt for 4-way table:"
echo "          llm.rs    GPU short: 58.11 ms  long: 69.19 ms"
echo "          llama.cpp CPU short: 50.20 ms  long: 54.20 ms"
