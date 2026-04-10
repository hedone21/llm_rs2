#!/usr/bin/env bash
# Flash attention decode benchmark vs llama.cpp CPU (Adreno 830).
#
# Runs a 4-context-length sweep of llm.rs and llama.cpp on the connected
# device and asserts llm.rs TBT <= llama.cpp TBT * (1 + TOLERANCE_PCT/100).
#
# Requirements on device:
#   /data/local/tmp/generate (our binary)
#   /data/local/tmp/models/llama3.2-1b
#   /data/local/tmp/{p100,p300,p600,long_prompt}.txt
#   /data/local/tmp/Llama-3.2-1B-Instruct-f16.gguf
#   /data/local/tmp/llama-bench + libllama.so + libggml*.so
#
# Exit code 0 on success; 1 on any per-context failure.

set -euo pipefail

# Force byte-level locale so awk can match the multibyte ± (U+00B1) used in
# llama-bench's markdown output without "multibyte conversion failure".
export LC_ALL=C

# Preflight: require adb and a connected device.
command -v adb >/dev/null 2>&1 || {
  echo "[bench] adb not on PATH" >&2
  exit 2
}
adb get-state 1>/dev/null 2>&1 || {
  echo "[bench] no device connected (adb get-state failed)" >&2
  exit 2
}

# Resolve OUTDIR relative to the script's location (not the shell's cwd)
# so the script works from any working directory.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="$REPO_ROOT/results/data/flash_attn_decode"
mkdir -p "$OUTDIR"

# Tolerances: llm.rs TBT <= llama.cpp TBT * (1 + TOLERANCE_PCT/100)
TOLERANCE_PCT="${TOLERANCE_PCT:-5}"
# Cooldown between adb runs. Empirically 15s was not enough on a warm
# device — raise to 30s to avoid thermal drift in long_prompt. Override
# with COOLDOWN_SEC=<n> for faster local iteration.
COOLDOWN_SEC="${COOLDOWN_SEC:-30}"

tbt_llm_rs() {
  local prompt_file="$1"
  adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/llama3.2-1b --prompt-file /data/local/tmp/${prompt_file} -n 128 -b opencl 2>&1" \
    | awk '/^Decode:/ { gsub(/[^0-9.]/, "", $2); print $2; exit }'
}

tbt_llama_cpp() {
  local depth="$1"
  # llama-bench prints a markdown table; the t/s field is "<number> ± <stddev>".
  # ± is UTF-8 0xC2 0xB1 — match at the byte level and run awk under LC_ALL=C.
  local tps
  tps=$(adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-bench -m Llama-3.2-1B-Instruct-f16.gguf -p 0 -n 128 -d ${depth} -r 1 -t 8 2>&1" \
    | awk '
        /tg128/ && /llama 1B F16/ {
          if (match($0, /\|[[:space:]]*[0-9]+\.[0-9]+[[:space:]]*\xc2\xb1/)) {
            s = substr($0, RSTART+1, RLENGTH-3);
            gsub(/[[:space:]]/, "", s);
            print s;
            exit;
          }
        }')
  if [ -z "$tps" ]; then
    echo "ERROR" >&2
    return 1
  fi
  awk "BEGIN { printf \"%.2f\", 1000.0 / $tps }"
}

# Parallel arrays (associative arrays require bash 4+, macOS ships bash 3.2).
PROMPTS=(p100.txt p300.txt p600.txt long_prompt.txt)
DEPTHS=(170 374 676 720)

{
  echo "# $(date) — flash attention decode benchmark vs llama.cpp CPU"
  echo "# Device: Samsung Galaxy S25 (Adreno 830)"
  echo "# Tolerance: llm.rs TBT <= llama.cpp TBT * 1.0${TOLERANCE_PCT}"
  echo
} > "$OUTDIR/after-task4.txt"

echo "[bench] Starting 4-context sweep..."

fail=0
for i in 0 1 2 3; do
  f="${PROMPTS[$i]}"
  depth="${DEPTHS[$i]}"
  echo "[bench] $f (approx seq ${depth})..."

  ours=$(tbt_llm_rs "$f") || ours=""
  if [ -z "$ours" ]; then
    echo "[FAIL] $f: failed to parse llm.rs TBT" | tee -a "$OUTDIR/after-task4.txt"
    fail=1
    continue
  fi
  sleep "$COOLDOWN_SEC"

  theirs=$(tbt_llama_cpp "$depth") || theirs=""
  if [ -z "$theirs" ]; then
    echo "[FAIL] $f: failed to parse llama.cpp TBT" | tee -a "$OUTDIR/after-task4.txt"
    fail=1
    sleep "$COOLDOWN_SEC"
    continue
  fi
  sleep "$COOLDOWN_SEC"

  limit=$(awk "BEGIN { printf \"%.2f\", $theirs * (100 + $TOLERANCE_PCT) / 100 }")
  cmp=$(awk "BEGIN { print ($ours > $limit) ? 1 : 0 }")

  line="$f depth=$depth llm_rs=${ours}ms llama_cpp=${theirs}ms limit=${limit}ms"
  if [ "$cmp" = "1" ]; then
    echo "[FAIL] $line" | tee -a "$OUTDIR/after-task4.txt"
    fail=1
  else
    echo "[ OK ] $line" | tee -a "$OUTDIR/after-task4.txt"
  fi
done

echo
echo "[bench] Results saved to $OUTDIR/after-task4.txt"
exit $fail
