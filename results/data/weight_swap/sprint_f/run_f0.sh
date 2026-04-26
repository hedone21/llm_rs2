#!/bin/bash
# Sprint F-0: lm_head Q4_0 quantize TBT recovery measurement
# 3 conditions × N=3 runs each, 128 tokens, async mode (no --profile).
# Galaxy S25, threads=6, V10 thermal isolation.

set -u
DEVICE=R3CY408S5SB
RESULTS_DIR=/data/local/tmp/sprint_f
adb -s "$DEVICE" shell "mkdir -p $RESULTS_DIR"

run_one() {
    local label="$1"
    local condition="$2"
    local cmd="$3"
    echo "=== $label ==="
    for i in 1 2 3; do
        echo "--- run $i ---"
        adb -s "$DEVICE" shell "cd /data/local/tmp && $cmd" 2>&1 | \
            grep -E "Decode\(excl|Decode:|Avg TBT|Quantized lm_head|weight_swap: force"
        # cool-down between runs
        sleep 5
    done
    echo
}

COMMON="--backend opencl --threads 6 --temperature 0.0 --num-tokens 128 --protected-prefix 4 --prompt 'The capital of France is'"

# Condition 1: Q4 baseline (lm_head naturally Q4_0)
run_one "Q4 baseline" "q4_baseline" \
    "taskset f0 ./generate --model-path /data/local/tmp/Llama-3.2-1B-Instruct-q4_0.gguf $COMMON"

# Condition 2: Mixed C-2 baseline (F16 primary + AUF secondary, ratio=1.0, lm_head=F16)
run_one "Mixed C-2 baseline (lm_head=F16)" "mixed_c2_f16_lmhead" \
    "taskset f0 ./generate --model-path /data/local/tmp/Llama-3.2-1B-Instruct-f16.gguf --secondary-gguf /data/local/tmp/Llama-3.2-1B-Instruct.auf --force-swap-ratio 1.0 $COMMON"

# Condition 3: Mixed C-2 + lm_head force Q4_0
run_one "Mixed C-2 + lm_head Q4_0" "mixed_c2_q4_lmhead" \
    "taskset f0 ./generate --model-path /data/local/tmp/Llama-3.2-1B-Instruct-f16.gguf --secondary-gguf /data/local/tmp/Llama-3.2-1B-Instruct.auf --force-swap-ratio 1.0 --quantize-lm-head q4_0 $COMMON"
