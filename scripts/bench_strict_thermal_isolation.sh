#!/usr/bin/env bash
# Strict thermal isolation benchmark for Qwen 2.5-1.5B on Adreno 830.
#
# Stricter than bench_thermal_controlled.sh:
#   1. 5-minute MINIMUM rest between every run (no interleaving CPU/GPU)
#   2. CPU (llama.cpp) and GPU (llm.rs) batches are completely separated —
#      all CPU runs first, then a long inter-phase rest, then all GPU runs.
#      This eliminates cross-contamination from recent opposite-backend heat.
#   3. Monitors 4 thermal zones (cpu+gpu multi-sample) for a fuller picture.
#
# Total runtime: ~80 minutes for 3 runs × 2 combos × 2 backends × 5-min rest.

set -euo pipefail
export LC_ALL=C

command -v adb >/dev/null 2>&1 || { echo "adb not on PATH" >&2; exit 2; }
adb get-state 1>/dev/null 2>&1 || { echo "no device connected" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="$REPO_ROOT/results/data/flash_attn_decode/thermal"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/qwen_strict_isolation.txt"

REST_SEC="${REST_SEC:-300}"        # 5 min after every run
INTER_PHASE_REST="${INTER_PHASE_REST:-300}"  # 5 min between CPU and GPU phases
THERMAL_THRESHOLD_MC="${THERMAL_THRESHOLD_MC:-50000}"
N_RUNS="${N_RUNS:-3}"

# Zombie preflight
zombie_check() {
  local zombies
  zombies=$(adb shell 'ps -A 2>/dev/null | grep -E "(llama-cli|llama-cli-new|generate_master|generate[^a-z0-9])" | grep -v grep || true' \
    | tr -d '\r')
  if [ -n "$zombies" ]; then
    echo "[strict] ERROR: stale processes on device:" >&2
    echo "$zombies" >&2
    exit 3
  fi
}
zombie_check
echo "[strict] zombie check OK"

# Sample 4 thermal zones: 2 CPU clusters + 2 GPU sub-zones
# zone1=cpu-0-0-0, zone10=cpu-0-4-1, zone28=gpuss-5, zone30=gpuss-7
read_zones() {
  adb shell 'for z in 1 10 28 30; do cat /sys/class/thermal/thermal_zone$z/temp 2>/dev/null; done' \
    | tr -d '\r' | tr '\n' ' '
}

read_max_thermal() {
  read_zones | tr ' ' '\n' | grep -v '^$' | sort -nr | head -1
}

# Wait REST_SEC minimum, then optionally until thermal cools below threshold.
# Reports every 60s.
strict_rest() {
  local label="$1"
  local start_ts=$(date +%s)
  echo "  [rest] $label — waiting ${REST_SEC}s..."
  local remain=$REST_SEC
  while [ $remain -gt 0 ]; do
    if [ $remain -gt 60 ]; then
      sleep 60
      remain=$((remain - 60))
      local t=$(read_max_thermal 2>/dev/null || echo "?")
      local elapsed=$(( $(date +%s) - start_ts ))
      echo "    elapsed=${elapsed}s  max_temp=${t}mC  remaining=${remain}s"
    else
      sleep "$remain"
      remain=0
    fi
  done
  # After fixed rest, ensure thermal threshold is met
  local t=$(read_max_thermal)
  if [ "$t" -gt "$THERMAL_THRESHOLD_MC" ]; then
    echo "  [rest] max=${t}mC still above ${THERMAL_THRESHOLD_MC}mC after ${REST_SEC}s — waiting extra 60s..."
    sleep 60
    t=$(read_max_thermal)
  fi
  local final_elapsed=$(( $(date +%s) - start_ts ))
  echo "  [rest] done: max=${t}mC after ${final_elapsed}s"
}

# Run llama.cpp llama-bench. Returns ms/tok on stdout.
run_llama_cpp() {
  local label="$1" depth="$2" ntok="$3"
  local zones_start zones_end out tps
  zones_start=$(read_zones)
  out=$(adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-bench -m Qwen2.5-1.5B-Instruct-f16.gguf -p 0 -n ${ntok} -d ${depth} -r 1 -t 8 2>&1" \
    | tr -d '\r')
  zones_end=$(read_zones)
  tps=$(echo "$out" | awk '
    /tg/ {
      if (match($0, /\|[[:space:]]*[0-9]+\.[0-9]+[[:space:]]*\xc2\xb1/)) {
        s = substr($0, RSTART+1, RLENGTH-3);
        gsub(/[[:space:]]/, "", s);
        print s;
        exit;
      }
    }')
  local decode=""
  if [ -n "$tps" ]; then
    decode=$(awk "BEGIN { printf \"%.2f\", 1000.0 / $tps }")
  fi
  echo "    llama.cpp ${label}: ${decode}ms/tok zones_start=[${zones_start}] zones_end=[${zones_end}]" >&2
  echo "$decode"
}

# Run llm.rs generate. Returns ms/tok on stdout.
run_llm_rs() {
  local label="$1" promptarg="$2" ntok="$3"
  local zones_start zones_end out decode
  zones_start=$(read_zones)
  out=$(adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b ${promptarg} -n ${ntok} -b opencl 2>&1" \
    | tr -d '\r')
  zones_end=$(read_zones)
  decode=$(echo "$out" | awk '/^Decode:/ { gsub(/[^0-9.]/, "", $2); print $2; exit }')
  echo "    llm.rs ${label}: ${decode}ms/tok zones_start=[${zones_start}] zones_end=[${zones_end}]" >&2
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

# Config
SHORT_PROMPT='--prompt "The quick brown fox jumps over"'
SHORT_DEPTH=7
SHORT_NTOK=64

LONG_PROMPT='--prompt-file /data/local/tmp/long_prompt.txt'
LONG_DEPTH=720
LONG_NTOK=128

{
  echo "# $(date) — Qwen strict thermal isolation (5 min rest, CPU/GPU batched)"
  echo "# Device: Samsung Galaxy S25 (Adreno 830)"
  echo "# Zones monitored: cpu-0-0-0 (zone1), cpu-0-4-1 (zone10), gpuss-5 (zone28), gpuss-7 (zone30)"
  echo "# Rest between runs: ${REST_SEC}s (min) + threshold ${THERMAL_THRESHOLD_MC}mC"
  echo "# Inter-phase rest (CPU→GPU): ${INTER_PHASE_REST}s"
  echo "# Runs per combo: ${N_RUNS}"
  echo "# Short combo: prefill=${SHORT_DEPTH}, decode=${SHORT_NTOK}"
  echo "# Long combo:  prefill=${LONG_DEPTH}, decode=${LONG_NTOK}"
  echo
} > "$OUTFILE"

# -----------------------------------------------------------------
# Phase 1: llama.cpp CPU batch
# -----------------------------------------------------------------
echo
echo "============================================================"
echo "Phase 1: llama.cpp CPU batch"
echo "============================================================"
{ echo; echo "=== Phase 1: llama.cpp CPU ==="; } >> "$OUTFILE"

cpu_short=()
cpu_long=()

for run in $(seq 1 "$N_RUNS"); do
  strict_rest "cpu-short-run${run}"
  v=$(run_llama_cpp "cpu-short-run${run}" "$SHORT_DEPTH" "$SHORT_NTOK")
  cpu_short+=("$v")
  echo "  cpu short run=${run}: ${v}ms" >> "$OUTFILE"

  strict_rest "cpu-long-run${run}"
  v=$(run_llama_cpp "cpu-long-run${run}" "$LONG_DEPTH" "$LONG_NTOK")
  cpu_long+=("$v")
  echo "  cpu long  run=${run}: ${v}ms" >> "$OUTFILE"
done

cpu_short_med=$(printf "%s\n" "${cpu_short[@]}" | median_of)
cpu_long_med=$(printf "%s\n" "${cpu_long[@]}" | median_of)

# -----------------------------------------------------------------
# Inter-phase rest
# -----------------------------------------------------------------
echo
echo "Inter-phase rest (${INTER_PHASE_REST}s)..."
{ echo; echo "--- inter-phase rest ${INTER_PHASE_REST}s ---"; } >> "$OUTFILE"
REST_SEC="$INTER_PHASE_REST" strict_rest "inter-phase"

# -----------------------------------------------------------------
# Phase 2: llm.rs OpenCL GPU batch
# -----------------------------------------------------------------
echo
echo "============================================================"
echo "Phase 2: llm.rs OpenCL GPU batch"
echo "============================================================"
{ echo; echo "=== Phase 2: llm.rs OpenCL GPU ==="; } >> "$OUTFILE"

gpu_short=()
gpu_long=()

for run in $(seq 1 "$N_RUNS"); do
  strict_rest "gpu-short-run${run}"
  v=$(run_llm_rs "gpu-short-run${run}" "$SHORT_PROMPT" "$SHORT_NTOK")
  gpu_short+=("$v")
  echo "  gpu short run=${run}: ${v}ms" >> "$OUTFILE"

  strict_rest "gpu-long-run${run}"
  v=$(run_llm_rs "gpu-long-run${run}" "$LONG_PROMPT" "$LONG_NTOK")
  gpu_long+=("$v")
  echo "  gpu long  run=${run}: ${v}ms" >> "$OUTFILE"
done

gpu_short_med=$(printf "%s\n" "${gpu_short[@]}" | median_of)
gpu_long_med=$(printf "%s\n" "${gpu_long[@]}" | median_of)

# -----------------------------------------------------------------
# Summary
# -----------------------------------------------------------------
short_ratio=$(awk "BEGIN { if ($cpu_short_med > 0) { printf \"%+.1f%%\", ($gpu_short_med - $cpu_short_med) / $cpu_short_med * 100 } }")
long_ratio=$(awk "BEGIN { if ($cpu_long_med > 0) { printf \"%+.1f%%\", ($gpu_long_med - $cpu_long_med) / $cpu_long_med * 100 } }")

{
  echo
  echo "=== Medians across $N_RUNS runs ==="
  echo "  Combo A short(pf=$SHORT_DEPTH,dc=$SHORT_NTOK):"
  echo "    llama.cpp CPU : ${cpu_short_med} ms/tok  (runs: ${cpu_short[*]})"
  echo "    llm.rs    GPU : ${gpu_short_med} ms/tok  (runs: ${gpu_short[*]})"
  echo "    GPU vs CPU    : ${short_ratio}"
  echo
  echo "  Combo B long (pf=$LONG_DEPTH,dc=$LONG_NTOK):"
  echo "    llama.cpp CPU : ${cpu_long_med} ms/tok  (runs: ${cpu_long[*]})"
  echo "    llm.rs    GPU : ${gpu_long_med} ms/tok  (runs: ${gpu_long[*]})"
  echo "    GPU vs CPU    : ${long_ratio}"
} | tee -a "$OUTFILE"

echo
echo "[strict] Done. Results in $OUTFILE"
