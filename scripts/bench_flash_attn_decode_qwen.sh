#!/usr/bin/env bash
# Flash attention decode benchmark for Qwen 2.5-1.5B (head_dim=128) on
# Adreno 830, vs llama.cpp CPU tg128 for the same GGUF.
#
# Keep in sync with scripts/bench_flash_attn_decode.sh. Only the model
# path / GGUF file / expected baseline differ.
#
# Requirements on device:
#   /data/local/tmp/generate (our binary)
#   /data/local/tmp/models/qwen2.5-1.5b
#   /data/local/tmp/{p100,p300,p600,long_prompt}.txt
#   /data/local/tmp/Qwen2.5-1.5B-Instruct-f16.gguf
#   /data/local/tmp/llama-bench + libllama.so + libggml*.so
#
# Exit code 0 on success; 1 on any per-context failure.
# If the Qwen GGUF is missing, the script records llm.rs TBT only and
# exits 0 with a warning (no regression gate in that mode).

set -euo pipefail

# Force byte-level locale for multibyte awk matches (llama-bench's ±).
export LC_ALL=C

# Preflight: require adb and a connected device.
command -v adb >/dev/null 2>&1 || {
  echo "[bench-qwen] adb not on PATH" >&2
  exit 2
}
adb get-state 1>/dev/null 2>&1 || {
  echo "[bench-qwen] no device connected (adb get-state failed)" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="$REPO_ROOT/results/data/flash_attn_decode"
mkdir -p "$OUTDIR"

TOLERANCE_PCT="${TOLERANCE_PCT:-5}"
COOLDOWN_SEC="${COOLDOWN_SEC:-30}"

# Detect whether the Qwen GGUF is present for llama-bench comparison.
HAS_LLAMA_BENCH=1
if ! adb shell 'ls /data/local/tmp/Qwen2.5-1.5B-Instruct-f16.gguf' >/dev/null 2>&1; then
  HAS_LLAMA_BENCH=0
  echo "[bench-qwen] WARNING: Qwen GGUF not on device; recording llm.rs numbers only"
fi

tbt_llm_rs() {
  local prompt_file="$1"
  adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b --prompt-file /data/local/tmp/${prompt_file} -n 128 -b opencl 2>&1" \
    | awk '/^Decode:/ { gsub(/[^0-9.]/, "", $2); print $2; exit }'
}

tbt_llama_cpp() {
  local depth="$1"
  local tps
  tps=$(adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-bench -m Qwen2.5-1.5B-Instruct-f16.gguf -p 0 -n 128 -d ${depth} -r 1 -t 8 2>&1" \
    | awk '
        /tg128/ {
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

PROMPTS=(p100.txt p300.txt p600.txt long_prompt.txt)
DEPTHS=(170 374 676 720)

{
  echo "# $(date) — Qwen flash attention decode benchmark vs llama.cpp CPU"
  echo "# Device: Samsung Galaxy S25 (Adreno 830)"
  echo "# Model: Qwen 2.5-1.5B (head_dim=128)"
  echo "# Tolerance: llm.rs TBT <= llama.cpp TBT * 1.0${TOLERANCE_PCT}"
  if [ "$HAS_LLAMA_BENCH" -eq 0 ]; then
    echo "# NOTE: Qwen GGUF missing — recording llm.rs only"
  fi
  echo
} > "$OUTDIR/qwen_after_c1.txt"

echo "[bench-qwen] Starting 4-context sweep..."

fail=0
for i in 0 1 2 3; do
  f="${PROMPTS[$i]}"
  depth="${DEPTHS[$i]}"
  echo "[bench-qwen] $f (approx seq ${depth})..."

  ours=$(tbt_llm_rs "$f") || ours=""
  if [ -z "$ours" ]; then
    echo "[FAIL] $f: failed to parse llm.rs TBT" | tee -a "$OUTDIR/qwen_after_c1.txt"
    fail=1
    continue
  fi
  sleep "$COOLDOWN_SEC"

  if [ "$HAS_LLAMA_BENCH" -eq 0 ]; then
    echo "[ INFO ] $f depth=$depth llm_rs=${ours}ms (no llama.cpp baseline)" | tee -a "$OUTDIR/qwen_after_c1.txt"
    continue
  fi

  theirs=$(tbt_llama_cpp "$depth") || theirs=""
  if [ -z "$theirs" ]; then
    echo "[FAIL] $f: failed to parse llama.cpp TBT" | tee -a "$OUTDIR/qwen_after_c1.txt"
    fail=1
    sleep "$COOLDOWN_SEC"
    continue
  fi
  sleep "$COOLDOWN_SEC"

  limit=$(awk "BEGIN { printf \"%.2f\", $theirs * (100 + $TOLERANCE_PCT) / 100 }")
  cmp=$(awk "BEGIN { print ($ours > $limit) ? 1 : 0 }")

  line="$f depth=$depth llm_rs=${ours}ms llama_cpp=${theirs}ms limit=${limit}ms"
  if [ "$cmp" = "1" ]; then
    echo "[FAIL] $line" | tee -a "$OUTDIR/qwen_after_c1.txt"
    fail=1
  else
    echo "[ OK ] $line" | tee -a "$OUTDIR/qwen_after_c1.txt"
  fi
done

echo
echo "[bench-qwen] Results saved to $OUTDIR/qwen_after_c1.txt"
exit $fail
