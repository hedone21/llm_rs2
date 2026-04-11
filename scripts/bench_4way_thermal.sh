#!/usr/bin/env bash
# 4-way apples-to-apples Qwen benchmark on Adreno 830 / Snapdragon 8 Elite.
#
# Measures:
#   - llm.rs OpenCL GPU
#   - llm.rs CPU
#   - llama.cpp OpenCL GPU (via llama-cli-new --single-turn)
#   - llama.cpp CPU (via llama-bench)
#
# Apples-to-apples: CPU↔CPU gap and GPU↔GPU gap, not cross-backend.
#
# Strict thermal isolation: 5-min rest between every run, all 4 backends
# batched as separate phases to avoid cross-contamination.
# Total runtime: ~150 minutes (4 phases × 2 combos × 3 runs × 5-min rest).

set -euo pipefail
export LC_ALL=C

command -v adb >/dev/null 2>&1 || { echo "adb not on PATH" >&2; exit 2; }
adb get-state 1>/dev/null 2>&1 || { echo "no device connected" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="$REPO_ROOT/results/data/flash_attn_decode/thermal"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/qwen_4way_strict.txt"

REST_SEC="${REST_SEC:-300}"
INTER_PHASE_REST="${INTER_PHASE_REST:-300}"
THERMAL_THRESHOLD_MC="${THERMAL_THRESHOLD_MC:-50000}"
N_RUNS="${N_RUNS:-3}"

# Zombie preflight
zombie_check() {
  local zombies
  zombies=$(adb shell 'ps -A 2>/dev/null | grep -E "(llama-cli|llama-cli-new|llama-cli-orig|generate_master|generate[^a-z0-9])" | grep -v grep || true' \
    | tr -d '\r')
  if [ -n "$zombies" ]; then
    echo "[4way] ERROR: stale processes on device:" >&2
    echo "$zombies" >&2
    exit 3
  fi
}
zombie_check
echo "[4way] zombie check OK"

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
  local t=$(read_max_thermal)
  if [ "$t" -gt "$THERMAL_THRESHOLD_MC" ]; then
    echo "  [rest] max=${t}mC still above ${THERMAL_THRESHOLD_MC}mC — extra 60s"
    sleep 60
    t=$(read_max_thermal)
  fi
  local final_elapsed=$(( $(date +%s) - start_ts ))
  echo "  [rest] done: max=${t}mC after ${final_elapsed}s"
}

# -----------------------------------------------------------------
# Backend runners (each returns ms/tok on stdout)
# -----------------------------------------------------------------

# llm.rs GPU (OpenCL)
run_llm_gpu() {
  local label="$1" promptarg="$2" ntok="$3"
  local zones_start zones_end out decode
  zones_start=$(read_zones)
  out=$(adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b ${promptarg} -n ${ntok} -b opencl 2>&1" \
    | tr -d '\r')
  zones_end=$(read_zones)
  decode=$(echo "$out" | awk '/^Decode:/ { gsub(/[^0-9.]/, "", $2); print $2; exit }')
  echo "    llm.rs GPU ${label}: ${decode}ms/tok [${zones_start}] -> [${zones_end}]" >&2
  echo "$decode"
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

# llama.cpp CPU (via llama-bench)
run_llc_cpu() {
  local label="$1" depth="$2" ntok="$3"
  local zones_start zones_end out tps decode
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
  decode=""
  if [ -n "$tps" ]; then
    decode=$(awk "BEGIN { printf \"%.2f\", 1000.0 / $tps }")
  fi
  echo "    llama.cpp CPU ${label}: ${decode}ms/tok [${zones_start}] -> [${zones_end}]" >&2
  echo "$decode"
}

# llama.cpp GPU (via llama-cli-new --single-turn, parses "Generation: X t/s")
run_llc_gpu() {
  local label="$1" promptarg="$2" ntok="$3"
  local zones_start zones_end out tps decode
  zones_start=$(read_zones)
  # promptarg for llama-cli-new: either "-p \"...\"" or "-f /data/local/tmp/file"
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

# Test combos
SHORT_LLM_PROMPT='--prompt "The quick brown fox jumps over"'
SHORT_LLC_PROMPT='-p "The quick brown fox jumps over"'
SHORT_DEPTH=7
SHORT_NTOK=64

LONG_LLM_PROMPT='--prompt-file /data/local/tmp/long_prompt.txt'
LONG_LLC_PROMPT='-f /data/local/tmp/long_prompt.txt'
LONG_DEPTH=720
LONG_NTOK=128

{
  echo "# $(date) — 4-way Qwen strict isolation"
  echo "# Device: Samsung Galaxy S25 (Adreno 830, Snapdragon 8 Elite)"
  echo "# Rest: ${REST_SEC}s / Inter-phase: ${INTER_PHASE_REST}s / Threshold: ${THERMAL_THRESHOLD_MC}mC"
  echo "# Runs per combo: ${N_RUNS}"
  echo "# Short: prefill=${SHORT_DEPTH}, decode=${SHORT_NTOK}"
  echo "# Long:  prefill=${LONG_DEPTH}, decode=${LONG_NTOK}"
  echo
} > "$OUTFILE"

declare -a llm_cpu_short llm_cpu_long llm_gpu_short llm_gpu_long
declare -a llc_cpu_short llc_cpu_long llc_gpu_short llc_gpu_long
llm_cpu_short=(); llm_cpu_long=(); llm_gpu_short=(); llm_gpu_long=()
llc_cpu_short=(); llc_cpu_long=(); llc_gpu_short=(); llc_gpu_long=()

# -----------------------------------------------------------------
# Phase 1: llama.cpp CPU
# -----------------------------------------------------------------
echo
echo "============================================================"
echo "Phase 1: llama.cpp CPU"
echo "============================================================"
{ echo; echo "=== Phase 1: llama.cpp CPU ==="; } >> "$OUTFILE"

for run in $(seq 1 "$N_RUNS"); do
  strict_rest "p1-short-r${run}"
  v=$(run_llc_cpu "short-r${run}" "$SHORT_DEPTH" "$SHORT_NTOK")
  llc_cpu_short+=("$v")
  echo "  llc_cpu short r=${run}: ${v}ms" >> "$OUTFILE"

  strict_rest "p1-long-r${run}"
  v=$(run_llc_cpu "long-r${run}" "$LONG_DEPTH" "$LONG_NTOK")
  llc_cpu_long+=("$v")
  echo "  llc_cpu long  r=${run}: ${v}ms" >> "$OUTFILE"
done

echo
echo "Inter-phase rest..."
{ echo; echo "--- inter-phase ---"; } >> "$OUTFILE"
REST_SEC="$INTER_PHASE_REST" strict_rest "1->2"

# -----------------------------------------------------------------
# Phase 2: llama.cpp GPU (OpenCL via llama-cli-new)
# -----------------------------------------------------------------
echo
echo "============================================================"
echo "Phase 2: llama.cpp GPU (OpenCL)"
echo "============================================================"
{ echo; echo "=== Phase 2: llama.cpp GPU ==="; } >> "$OUTFILE"

for run in $(seq 1 "$N_RUNS"); do
  strict_rest "p2-short-r${run}"
  v=$(run_llc_gpu "short-r${run}" "$SHORT_LLC_PROMPT" "$SHORT_NTOK")
  llc_gpu_short+=("$v")
  echo "  llc_gpu short r=${run}: ${v}ms" >> "$OUTFILE"

  strict_rest "p2-long-r${run}"
  v=$(run_llc_gpu "long-r${run}" "$LONG_LLC_PROMPT" "$LONG_NTOK")
  llc_gpu_long+=("$v")
  echo "  llc_gpu long  r=${run}: ${v}ms" >> "$OUTFILE"
done

echo
echo "Inter-phase rest..."
{ echo; echo "--- inter-phase ---"; } >> "$OUTFILE"
REST_SEC="$INTER_PHASE_REST" strict_rest "2->3"

# -----------------------------------------------------------------
# Phase 3: llm.rs CPU
# -----------------------------------------------------------------
echo
echo "============================================================"
echo "Phase 3: llm.rs CPU"
echo "============================================================"
{ echo; echo "=== Phase 3: llm.rs CPU ==="; } >> "$OUTFILE"

for run in $(seq 1 "$N_RUNS"); do
  strict_rest "p3-short-r${run}"
  v=$(run_llm_cpu "short-r${run}" "$SHORT_LLM_PROMPT" "$SHORT_NTOK")
  llm_cpu_short+=("$v")
  echo "  llm_cpu short r=${run}: ${v}ms" >> "$OUTFILE"

  strict_rest "p3-long-r${run}"
  v=$(run_llm_cpu "long-r${run}" "$LONG_LLM_PROMPT" "$LONG_NTOK")
  llm_cpu_long+=("$v")
  echo "  llm_cpu long  r=${run}: ${v}ms" >> "$OUTFILE"
done

echo
echo "Inter-phase rest..."
{ echo; echo "--- inter-phase ---"; } >> "$OUTFILE"
REST_SEC="$INTER_PHASE_REST" strict_rest "3->4"

# -----------------------------------------------------------------
# Phase 4: llm.rs GPU
# -----------------------------------------------------------------
echo
echo "============================================================"
echo "Phase 4: llm.rs GPU (OpenCL)"
echo "============================================================"
{ echo; echo "=== Phase 4: llm.rs GPU ==="; } >> "$OUTFILE"

for run in $(seq 1 "$N_RUNS"); do
  strict_rest "p4-short-r${run}"
  v=$(run_llm_gpu "short-r${run}" "$SHORT_LLM_PROMPT" "$SHORT_NTOK")
  llm_gpu_short+=("$v")
  echo "  llm_gpu short r=${run}: ${v}ms" >> "$OUTFILE"

  strict_rest "p4-long-r${run}"
  v=$(run_llm_gpu "long-r${run}" "$LONG_LLM_PROMPT" "$LONG_NTOK")
  llm_gpu_long+=("$v")
  echo "  llm_gpu long  r=${run}: ${v}ms" >> "$OUTFILE"
done

# -----------------------------------------------------------------
# Summary
# -----------------------------------------------------------------
llc_cpu_short_med=$(printf "%s\n" "${llc_cpu_short[@]}" | median_of)
llc_cpu_long_med=$(printf "%s\n" "${llc_cpu_long[@]}" | median_of)
llc_gpu_short_med=$(printf "%s\n" "${llc_gpu_short[@]}" | median_of)
llc_gpu_long_med=$(printf "%s\n" "${llc_gpu_long[@]}" | median_of)
llm_cpu_short_med=$(printf "%s\n" "${llm_cpu_short[@]}" | median_of)
llm_cpu_long_med=$(printf "%s\n" "${llm_cpu_long[@]}" | median_of)
llm_gpu_short_med=$(printf "%s\n" "${llm_gpu_short[@]}" | median_of)
llm_gpu_long_med=$(printf "%s\n" "${llm_gpu_long[@]}" | median_of)

cpu_short_ratio=$(awk "BEGIN { if ($llc_cpu_short_med > 0) { printf \"%+.1f%%\", ($llm_cpu_short_med - $llc_cpu_short_med) / $llc_cpu_short_med * 100 } }")
cpu_long_ratio=$(awk "BEGIN { if ($llc_cpu_long_med > 0) { printf \"%+.1f%%\", ($llm_cpu_long_med - $llc_cpu_long_med) / $llc_cpu_long_med * 100 } }")
gpu_short_ratio=$(awk "BEGIN { if ($llc_gpu_short_med > 0) { printf \"%+.1f%%\", ($llm_gpu_short_med - $llc_gpu_short_med) / $llc_gpu_short_med * 100 } }")
gpu_long_ratio=$(awk "BEGIN { if ($llc_gpu_long_med > 0) { printf \"%+.1f%%\", ($llm_gpu_long_med - $llc_gpu_long_med) / $llc_gpu_long_med * 100 } }")

{
  echo
  echo "=== Medians across ${N_RUNS} runs ==="
  echo
  echo "  CPU comparison (llm.rs CPU vs llama.cpp CPU)"
  echo "    Short: llm.rs=${llm_cpu_short_med} ms  llama.cpp=${llc_cpu_short_med} ms  llm.rs is ${cpu_short_ratio}"
  echo "    Long : llm.rs=${llm_cpu_long_med} ms  llama.cpp=${llc_cpu_long_med} ms  llm.rs is ${cpu_long_ratio}"
  echo
  echo "  GPU comparison (llm.rs OpenCL vs llama.cpp OpenCL)"
  echo "    Short: llm.rs=${llm_gpu_short_med} ms  llama.cpp=${llc_gpu_short_med} ms  llm.rs is ${gpu_short_ratio}"
  echo "    Long : llm.rs=${llm_gpu_long_med} ms  llama.cpp=${llc_gpu_long_med} ms  llm.rs is ${gpu_long_ratio}"
  echo
  echo "  Per-run values:"
  echo "    llm.rs CPU short:    ${llm_cpu_short[*]}"
  echo "    llm.rs CPU long:     ${llm_cpu_long[*]}"
  echo "    llm.rs GPU short:    ${llm_gpu_short[*]}"
  echo "    llm.rs GPU long:     ${llm_gpu_long[*]}"
  echo "    llama.cpp CPU short: ${llc_cpu_short[*]}"
  echo "    llama.cpp CPU long:  ${llc_cpu_long[*]}"
  echo "    llama.cpp GPU short: ${llc_gpu_short[*]}"
  echo "    llama.cpp GPU long:  ${llc_gpu_long[*]}"
} | tee -a "$OUTFILE"

echo
echo "[4way] Done. Results in $OUTFILE"
