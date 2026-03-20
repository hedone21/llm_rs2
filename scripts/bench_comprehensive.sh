#!/bin/bash
# Comprehensive benchmark: llm.rs vs llama.cpp, CPU vs GPU
# Memory polling + speed measurement with 3-minute cooling between tests
set -euo pipefail

PROMPT='Explain the process of photosynthesis in detail, including the light-dependent reactions and the Calvin cycle. Describe how plants convert carbon dioxide and water into glucose and oxygen using sunlight energy. Also discuss the role of chlorophyll and other pigments in capturing light energy, the electron transport chain in the thylakoid membrane, and how ATP and NADPH are produced and used in the carbon fixation reactions.'
N_TOKENS=128
COOLDOWN=180  # 3 minutes
POLL_INTERVAL=0.2
OUTDIR="/tmp/bench_results"
mkdir -p "$OUTDIR"

wait_cool() {
    local target_temp=${1:-35000}
    echo "  Cooling down (target: ${target_temp}m°C, max ${COOLDOWN}s)..."
    local elapsed=0
    while [ $elapsed -lt $COOLDOWN ]; do
        local temp=$(adb shell "cat /sys/class/thermal/thermal_zone1/temp" 2>/dev/null)
        if [ "$temp" -le "$target_temp" ] 2>/dev/null && [ $elapsed -ge 60 ]; then
            echo "  Cooled to ${temp}m°C after ${elapsed}s"
            return
        fi
        sleep 10
        elapsed=$((elapsed + 10))
        if [ $((elapsed % 30)) -eq 0 ]; then
            echo "    ${elapsed}s elapsed, temp=${temp}m°C"
        fi
    done
    local temp=$(adb shell "cat /sys/class/thermal/thermal_zone1/temp" 2>/dev/null)
    echo "  Cooldown complete (${temp}m°C)"
}

# Run a command on device with memory polling
# Usage: run_with_memmon <tag> <cmd>
run_with_memmon() {
    local tag="$1"
    shift
    local cmd="$*"
    local memfile="$OUTDIR/${tag}_mem.csv"
    local outfile="$OUTDIR/${tag}_out.txt"

    local temp_before=$(adb shell "cat /sys/class/thermal/thermal_zone1/temp" 2>/dev/null)
    echo "  Start temp: ${temp_before}m°C"

    # Start the command in background on device, capture PID
    adb shell "cd /data/local/tmp && nohup sh -c '${cmd}' > /data/local/tmp/_bench_out.txt 2>&1 & echo \$!" > /tmp/_bench_pid.txt
    local pid=$(cat /tmp/_bench_pid.txt | tr -d '[:space:]')
    echo "  PID: $pid"

    # Poll memory
    echo "time_s,rss_kb,vmpeak_kb" > "$memfile"
    local start_time=$(date +%s%N)
    local alive=1
    while [ $alive -eq 1 ]; do
        # Check if process still exists
        local check=$(adb shell "kill -0 $pid 2>/dev/null && echo alive || echo dead" | tr -d '[:space:]')
        if [ "$check" != "alive" ]; then
            alive=0
            break
        fi
        # Read memory
        local mem=$(adb shell "cat /proc/$pid/status 2>/dev/null | grep -E 'VmRSS|VmPeak'" 2>/dev/null || echo "")
        if [ -n "$mem" ]; then
            local rss=$(echo "$mem" | grep VmRSS | awk '{print $2}')
            local peak=$(echo "$mem" | grep VmPeak | awk '{print $2}')
            local now=$(date +%s%N)
            local elapsed_s=$(echo "scale=3; ($now - $start_time) / 1000000000" | bc)
            echo "${elapsed_s},${rss:-0},${peak:-0}" >> "$memfile"
        fi
        sleep $POLL_INTERVAL
    done

    # Wait a moment for output file to flush
    sleep 1
    adb pull /data/local/tmp/_bench_out.txt "$outfile" 2>/dev/null || true

    local temp_after=$(adb shell "cat /sys/class/thermal/thermal_zone1/temp" 2>/dev/null)
    echo "  End temp: ${temp_after}m°C"
    echo "  Output: $outfile"
    echo "  Memory: $memfile"
}

echo "=============================================="
echo " Comprehensive Benchmark: llm.rs vs llama.cpp"
echo " Tokens: $N_TOKENS, Cooldown: ${COOLDOWN}s"
echo "=============================================="
echo ""

# === Test 1: llm.rs CPU ===
echo "[1/4] llm.rs CPU F16"
wait_cool
run_with_memmon "llmrs_cpu" \
    "LD_LIBRARY_PATH=/data/local/tmp ./generate --model-path /data/local/tmp/models/llama3.2-1b --prompt '${PROMPT}' -n ${N_TOKENS} -b cpu --weight-dtype f16"
echo ""

# === Test 2: llm.rs GPU ===
echo "[2/4] llm.rs GPU (OpenCL)"
wait_cool
run_with_memmon "llmrs_gpu" \
    "LD_LIBRARY_PATH=/data/local/tmp ./generate --model-path /data/local/tmp/models/llama3.2-1b --prompt '${PROMPT}' -n ${N_TOKENS} -b opencl --weight-dtype f16"
echo ""

# === Test 3: llama.cpp CPU ===
echo "[3/4] llama.cpp CPU"
wait_cool
run_with_memmon "llamacpp_cpu" \
    "./llama-cli-orig -p '${PROMPT}' -n ${N_TOKENS} -st --temp 0 -no-cnv -m Llama-3.2-1B-Instruct-f16.gguf"
echo ""

# === Test 4: llama.cpp GPU ===
echo "[4/4] llama.cpp GPU (ngl 99)"
wait_cool
run_with_memmon "llamacpp_gpu" \
    "./llama-cli-orig -p '${PROMPT}' -n ${N_TOKENS} -st --temp 0 -no-cnv -ngl 99 -m Llama-3.2-1B-Instruct-f16.gguf"
echo ""

echo "=============================================="
echo " All tests complete. Results in: $OUTDIR"
echo "=============================================="
