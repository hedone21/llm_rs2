#!/bin/bash
# Validate if cold-start of GEMM kernel inflates prefill timer.
# Tests 3 warmup levels on the deployed binary.
set -e
SERIAL="R3CY408S5SB"
MODEL="/data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf"
PORT=38701
OUTDIR="/home/go/Workspace/llm_rs2/.agent/research/2026-04-20_qwen15b_q4_0_gpu_optim/warmup_check"
HERE="/home/go/Workspace/llm_rs2/.agent/research/2026-04-20_qwen15b_host_vs_device_bench"

run_one() {
    local WTOK="$1"
    local OUT="$2"
    adb -s "$SERIAL" shell 'pkill -9 llm_manager 2>/dev/null; pkill -9 generate 2>/dev/null; sleep 1; true' >/dev/null 2>&1 || true
    adb -s "$SERIAL" shell 'rm -f /data/local/tmp/llm_manager.log'
    adb -s "$SERIAL" shell "RUST_LOG=info /data/local/tmp/llm_manager --policy-script /data/local/tmp/noop_policy.lua --transport tcp:127.0.0.1:$PORT > /data/local/tmp/llm_manager.log 2>&1 &"
    sleep 2
    {
        echo "=== WARMUP_TOKENS=$WTOK ==="
        echo "=== temp snapshot before ==="
    } >> "$OUT"
    bash "$HERE/snapshot_temps.sh" "$SERIAL" >> "$OUT"
    echo "=== llm.rs generate (-b opencl, LLMRS_WARMUP_TOKENS=$WTOK) ===" >> "$OUT"
    adb -s "$SERIAL" shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp LLMRS_WARMUP_TOKENS=$WTOK ./generate --model-path $MODEL --prompt-file /data/local/tmp/prompt_128.txt -b opencl -n 128 --temperature 0 --greedy --ignore-eos --threads 6 --enable-resilience --resilience-transport tcp:127.0.0.1:$PORT 2>&1" >> "$OUT"
    echo "=== temp snapshot after ===" >> "$OUT"
    bash "$HERE/snapshot_temps.sh" "$SERIAL" >> "$OUT"
    adb -s "$SERIAL" shell 'pkill -9 llm_manager 2>/dev/null; pkill -9 generate 2>/dev/null; true' >/dev/null 2>&1 || true
}

# Warmup 1 (baseline, same as prior measurement)
run_one 1 "$OUTDIR/warmup_1.log"
# Thermal gate between runs
bash "$HERE/thermal_gate.sh" "$SERIAL" >> "$OUTDIR/thermal.log" 2>&1
# Warmup 128 (prefill path fully warmed)
run_one 128 "$OUTDIR/warmup_128.log"

echo "DONE"
