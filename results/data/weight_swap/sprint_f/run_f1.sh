#!/bin/bash
# Sprint F-1: production-ready validation. With --quantize-lm-head=auto (default),
# Mixed AUF ratio=1.0 should match the F-0 results (lm_head Q4_0 path).
# Reference: Q4 baseline ~16.4 ms/tok, F-0 mixed+q4_0 ~14.8 ms/tok.

set -u
DEVICE=R3CY408S5SB

run_one() {
    local label="$1"
    local cmd="$2"
    echo "=== $label ==="
    for i in 1 2 3; do
        echo "--- run $i ---"
        adb -s "$DEVICE" shell "cd /data/local/tmp && $cmd" 2>&1 | \
            grep -E "Decode\(excl|Decode:|Avg TBT|Quantized lm_head|weight_swap: force"
        sleep 5
    done
    echo
}

COMMON="--backend opencl --threads 6 --temperature 0.0 --num-tokens 128 --protected-prefix 4 --prompt 'The capital of France is'"

# Q4 baseline (auto = no-op, already Q4_0)
run_one "Q4 baseline (auto)" \
    "taskset f0 ./generate --model-path /data/local/tmp/Llama-3.2-1B-Instruct-q4_0.gguf $COMMON"

# Mixed AUF ratio=1.0 (auto = quantize triggered)
run_one "Mixed AUF ratio=1.0 (auto)" \
    "taskset f0 ./generate --model-path /data/local/tmp/Llama-3.2-1B-Instruct-f16.gguf --secondary-gguf /data/local/tmp/Llama-3.2-1B-Instruct.auf --force-swap-ratio 1.0 $COMMON"

# Mixed AUF ratio=0.5 (intermediate, auto = quantize)
run_one "Mixed AUF ratio=0.5 (auto)" \
    "taskset f0 ./generate --model-path /data/local/tmp/Llama-3.2-1B-Instruct-f16.gguf --secondary-gguf /data/local/tmp/Llama-3.2-1B-Instruct.auf --force-swap-ratio 0.5 $COMMON"

# Regression test: --quantize-lm-head=none on Mixed should reproduce the gap
run_one "Mixed AUF ratio=1.0 + --quantize-lm-head=none (regression)" \
    "taskset f0 ./generate --model-path /data/local/tmp/Llama-3.2-1B-Instruct-f16.gguf --secondary-gguf /data/local/tmp/Llama-3.2-1B-Instruct.auf --force-swap-ratio 1.0 --quantize-lm-head none $COMMON"
