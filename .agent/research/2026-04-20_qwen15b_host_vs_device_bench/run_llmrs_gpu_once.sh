#!/bin/bash
# Usage: run_llmrs_gpu_once.sh <model_path_on_device> <out_log_path>
set -e
SERIAL="R3CY408S5SB"
MODEL="$1"
OUT="$2"
PORT=38701
HERE="/home/go/Workspace/llm_rs2/.agent/research/2026-04-20_qwen15b_host_vs_device_bench"

# Kill stragglers (force)
adb -s "$SERIAL" shell 'pkill -9 llm_manager 2>/dev/null; pkill -9 generate 2>/dev/null; sleep 1; true' >/dev/null 2>&1 || true
adb -s "$SERIAL" shell 'rm -f /data/local/tmp/llm_manager.log'

# Manager: TCP transport (Android 14 SELinux blocks unix socket on /data/local/tmp)
adb -s "$SERIAL" shell "RUST_LOG=info /data/local/tmp/llm_manager --policy-script /data/local/tmp/noop_policy.lua --transport tcp:127.0.0.1:$PORT > /data/local/tmp/llm_manager.log 2>&1 &"

sleep 2
{
    echo "=== manager pgrep ==="
    adb -s "$SERIAL" shell 'pgrep -a llm_manager'
    echo "=== manager log (pre-connect) ==="
    adb -s "$SERIAL" shell 'cat /data/local/tmp/llm_manager.log 2>&1'
    echo "=== temp snapshot before run ==="
} >> "$OUT"
bash "$HERE/snapshot_temps.sh" "$SERIAL" >> "$OUT"

echo "=== llm.rs generate (-b opencl) ===" >> "$OUT"
adb -s "$SERIAL" shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate --model-path $MODEL --prompt-file /data/local/tmp/prompt_128.txt -b opencl -n 128 --temperature 0 --greedy --ignore-eos --threads 6 --enable-resilience --resilience-transport tcp:127.0.0.1:$PORT 2>&1" >> "$OUT"

echo "=== temp snapshot after run ===" >> "$OUT"
bash "$HERE/snapshot_temps.sh" "$SERIAL" >> "$OUT"

{
    echo "=== manager log (post-run) ==="
    adb -s "$SERIAL" shell 'cat /data/local/tmp/llm_manager.log 2>&1'
} >> "$OUT"

adb -s "$SERIAL" shell 'pkill -9 llm_manager 2>/dev/null; pkill -9 generate 2>/dev/null; sleep 1; true' >/dev/null 2>&1 || true
echo "=== done ===" >> "$OUT"
