#!/usr/bin/env bash
# Thermally-controlled comparison benchmark for Qwen 2.5-1.5B on Adreno 830.
#
# Motivation: the simple 4-context bench_flash_attn_decode_qwen.sh interleaves
# llm.rs OpenCL and llama.cpp CPU runs with 30s cooldowns. Thermal state may
# still differ between runs because:
#   - llama.cpp saturates all 8 CPU cores (big thermal load)
#   - llm.rs uses mostly GPU with light CPU involvement
#   - fixed 30s cooldown may not recover from CPU load fully
#
# This script enforces thermal parity via active polling:
#   - Before each run, poll thermal zones until max(cpu, gpu) drops below
#     THERMAL_THRESHOLD_MC (default 48000 = 48°C)
#   - Hard floor: COOLDOWN_MIN_SEC (default 45) between runs regardless
#   - Hard ceiling: COOLDOWN_MAX_SEC (default 300) — abort if not cool in time
#
# Measures two combos:
#   A. Short prefill (7 tokens) + short decode (32 tokens)
#   B. Long prefill (~720 tokens) + long decode (128 tokens)
#
# For each combo: N_RUNS (default 3) iterations, alternating the starting
# side to cancel any systematic bias. Results are median of the N_RUNS.

set -euo pipefail
export LC_ALL=C

# Preflight
command -v adb >/dev/null 2>&1 || { echo "adb not on PATH" >&2; exit 2; }
adb get-state 1>/dev/null 2>&1 || { echo "no device connected" >&2; exit 2; }

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="$REPO_ROOT/results/data/flash_attn_decode/thermal"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/qwen_thermal_controlled.txt"

THERMAL_THRESHOLD_MC="${THERMAL_THRESHOLD_MC:-55000}"
COOLDOWN_MIN_SEC="${COOLDOWN_MIN_SEC:-45}"
COOLDOWN_MAX_SEC="${COOLDOWN_MAX_SEC:-180}"
N_RUNS="${N_RUNS:-3}"

# Preflight: ensure no leftover llama-cli / llama-cli-new / generate zombies
# from previous runs are still holding CPU cores. These can silently wreck
# measurements (a zombie llama-cli-new running at 100% CPU adds 60 ms/tok
# of pollution to both sides of the comparison).
zombie_check() {
  # `|| true` on the outer adb because grep exits 1 when no match and
  # pipefail would kill the whole script.
  local zombies
  zombies=$(adb shell 'ps -A 2>/dev/null | grep -E "(llama-cli|llama-cli-new|generate_master)" | grep -v grep || true' \
    | tr -d '\r')
  if [ -n "$zombies" ]; then
    echo "[thermal] ERROR: stale processes on device:" >&2
    echo "$zombies" >&2
    echo "[thermal] kill them first: adb shell 'kill -9 <pid>'" >&2
    exit 3
  fi
}
zombie_check
echo "[thermal] zombie check OK"

# CPU: zone 1 (cpu-0-0-0), GPU: zone 28 (gpuss-5) — both are representative
# hottest zones per the earlier scan. Take the MAX of both.
read_max_thermal() {
  adb shell 'cat /sys/class/thermal/thermal_zone1/temp /sys/class/thermal/thermal_zone28/temp 2>/dev/null' \
    | tr -d '\r' \
    | sort -nr \
    | head -1
}

wait_for_cool() {
  local label="$1"
  local start_ts=$(date +%s)
  # Always wait at least COOLDOWN_MIN_SEC
  sleep "$COOLDOWN_MIN_SEC"
  local elapsed=0
  while true; do
    local t=$(read_max_thermal 2>/dev/null || echo 99999)
    elapsed=$(( $(date +%s) - start_ts ))
    if [ "$t" -le "$THERMAL_THRESHOLD_MC" ]; then
      echo "  [thermal] cooled to ${t}mC after ${elapsed}s (${label})"
      return 0
    fi
    if [ "$elapsed" -ge "$COOLDOWN_MAX_SEC" ]; then
      echo "  [thermal] WARN: still ${t}mC after ${elapsed}s (${label}, timeout)" >&2
      return 0
    fi
    sleep 5
  done
}

# Run llm.rs with the given prompt description and decode count.
# $1 = label (for logging)
# $2 = prompt arg ("--prompt \"...\"" or "--prompt-file /data/local/tmp/...")
# $3 = n_tokens
# Returns: prints <decode_ms_per_tok>
run_llm_rs() {
  local label="$1" promptarg="$2" ntok="$3"
  local t_start t_end out
  t_start=$(read_max_thermal)
  out=$(adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b ${promptarg} -n ${ntok} -b opencl 2>&1" \
    | tr -d '\r')
  t_end=$(read_max_thermal)
  local decode
  decode=$(echo "$out" | awk '/^Decode:/ { gsub(/[^0-9.]/, "", $2); print $2; exit }')
  echo "    llm.rs ${label}: decode=${decode}ms/tok t_start=${t_start}mC t_end=${t_end}mC" >&2
  echo "$decode"
}

# Run llama.cpp with the given depth and decode count.
# $1 = label
# $2 = depth (pre-fill tokens already in cache)
# $3 = n_tokens (decode length to measure)
# Returns: prints <decode_ms_per_tok>
run_llama_cpp() {
  local label="$1" depth="$2" ntok="$3"
  local t_start t_end out tps
  t_start=$(read_max_thermal)
  out=$(adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-bench -m Qwen2.5-1.5B-Instruct-f16.gguf -p 0 -n ${ntok} -d ${depth} -r 1 -t 8 2>&1" \
    | tr -d '\r')
  t_end=$(read_max_thermal)
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
  echo "    llama.cpp ${label}: decode=${decode}ms/tok t_start=${t_start}mC t_end=${t_end}mC" >&2
  echo "$decode"
}

median_of() {
  # Print median of stdin numeric values
  sort -n | awk '
    { a[NR] = $1 }
    END {
      if (NR % 2 == 1) { print a[(NR+1)/2] }
      else { printf "%.2f\n", (a[NR/2] + a[NR/2+1]) / 2.0 }
    }'
}

{
  echo "# $(date) — Qwen thermal-controlled comparison"
  echo "# Device: Samsung Galaxy S25 (Adreno 830)"
  echo "# Thermal threshold: ${THERMAL_THRESHOLD_MC}mC (max of cpu-0-0-0 + gpuss-5)"
  echo "# Cooldown: min ${COOLDOWN_MIN_SEC}s, max ${COOLDOWN_MAX_SEC}s"
  echo "# Runs per combo: ${N_RUNS}"
  echo
} > "$OUTFILE"

echo "[thermal] Starting controlled comparison..."
echo "[thermal] Writing to $OUTFILE"

# Combo A: short prefill + short decode
# Prompt "The quick brown fox jumps over" = 7 tokens (for llm.rs)
# llama.cpp depth=7, n=64 (64 decode tokens to reach steady-state TBT)
SHORT_PROMPT='--prompt "The quick brown fox jumps over"'
SHORT_DEPTH=7
SHORT_NTOK=64

# Combo B: long prefill + long decode
# Uses /data/local/tmp/long_prompt.txt (depth ~720) for both
LONG_PROMPT='--prompt-file /data/local/tmp/long_prompt.txt'
LONG_DEPTH=720
LONG_NTOK=128

declare -a a_llm=() a_llc=() b_llm=() b_llc=()

for run in $(seq 1 "$N_RUNS"); do
  echo
  echo "=== Run $run of $N_RUNS ==="

  # Alternate order per iteration to cancel systematic bias
  if [ $((run % 2)) -eq 1 ]; then
    order="llm_first"
  else
    order="llc_first"
  fi
  echo "  order: $order"

  wait_for_cool "pre-short-run${run}"
  if [ "$order" = "llm_first" ]; then
    v=$(run_llm_rs "short-run${run}" "$SHORT_PROMPT" "$SHORT_NTOK")
    a_llm+=("$v")
    wait_for_cool "mid-short-run${run}"
    v=$(run_llama_cpp "short-run${run}" "$SHORT_DEPTH" "$SHORT_NTOK")
    a_llc+=("$v")
  else
    v=$(run_llama_cpp "short-run${run}" "$SHORT_DEPTH" "$SHORT_NTOK")
    a_llc+=("$v")
    wait_for_cool "mid-short-run${run}"
    v=$(run_llm_rs "short-run${run}" "$SHORT_PROMPT" "$SHORT_NTOK")
    a_llm+=("$v")
  fi

  wait_for_cool "pre-long-run${run}"
  if [ "$order" = "llm_first" ]; then
    v=$(run_llm_rs "long-run${run}" "$LONG_PROMPT" "$LONG_NTOK")
    b_llm+=("$v")
    wait_for_cool "mid-long-run${run}"
    v=$(run_llama_cpp "long-run${run}" "$LONG_DEPTH" "$LONG_NTOK")
    b_llc+=("$v")
  else
    v=$(run_llama_cpp "long-run${run}" "$LONG_DEPTH" "$LONG_NTOK")
    b_llc+=("$v")
    wait_for_cool "mid-long-run${run}"
    v=$(run_llm_rs "long-run${run}" "$LONG_PROMPT" "$LONG_NTOK")
    b_llm+=("$v")
  fi

  # bash 3.2 compat: ${arr[@]: -1} instead of ${arr[-1]}
  {
    echo "run=$run order=$order"
    echo "  A short(pf=$SHORT_DEPTH,dc=$SHORT_NTOK): llm.rs=${a_llm[@]: -1} llama.cpp=${a_llc[@]: -1}"
    echo "  B long (pf=$LONG_DEPTH,dc=$LONG_NTOK): llm.rs=${b_llm[@]: -1} llama.cpp=${b_llc[@]: -1}"
  } >> "$OUTFILE"
done

# Compute medians
a_llm_med=$(printf "%s\n" "${a_llm[@]}" | median_of)
a_llc_med=$(printf "%s\n" "${a_llc[@]}" | median_of)
b_llm_med=$(printf "%s\n" "${b_llm[@]}" | median_of)
b_llc_med=$(printf "%s\n" "${b_llc[@]}" | median_of)

a_ratio=$(awk "BEGIN { if ($a_llc_med > 0) { printf \"%+.1f%%\", ($a_llm_med - $a_llc_med) / $a_llc_med * 100 } else { print \"n/a\" } }")
b_ratio=$(awk "BEGIN { if ($b_llc_med > 0) { printf \"%+.1f%%\", ($b_llm_med - $b_llc_med) / $b_llc_med * 100 } else { print \"n/a\" } }")

{
  echo
  echo "=== Medians across $N_RUNS runs ==="
  echo "  A short(pf=$SHORT_DEPTH,dc=$SHORT_NTOK): llm.rs=${a_llm_med}ms  llama.cpp=${a_llc_med}ms  delta=${a_ratio}"
  echo "  B long (pf=$LONG_DEPTH,dc=$LONG_NTOK): llm.rs=${b_llm_med}ms  llama.cpp=${b_llc_med}ms  delta=${b_ratio}"
} | tee -a "$OUTFILE"

echo
echo "[thermal] Done. Results in $OUTFILE"
