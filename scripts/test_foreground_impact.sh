#!/bin/bash
# Test inference performance with various foreground apps

GENERATE_BIN="/data/local/tmp/generate"
MODEL_PATH="/data/local/tmp/llm_rs2/models/llama3.2-1b"
PROMPT_FILE="/data/local/tmp/llm_rs2/eval/short_len.txt"
STRESS_SCRIPT=".agent/skills/testing/scripts/stress_test_adb.py"

# Test scenarios
declare -A SCENARIOS=(
    ["idle"]=""
    ["gpu_youtube"]="https://www.youtube.com/watch?v=LXb3EKWsInQ"
    ["gpu_maps"]="com.google.android.apps.maps"
    ["cpu_chrome"]="com.android.chrome"
    ["memory_heavy"]="com.android.chrome,com.google.android.apps.maps,com.google.android.youtube"
)

BACKEND="cpu"  # or "opencl"
NUM_TOKENS=128

echo "=== Foreground App Impact Test ==="
echo "Backend: $BACKEND"
echo "Tokens: $NUM_TOKENS"
echo ""

for scenario in "${!SCENARIOS[@]}"; do
    echo "--- Scenario: $scenario ---"
    apps="${SCENARIOS[$scenario]}"
    
    if [ -z "$apps" ]; then
        # Idle test - no foreground app switching
        echo "Running in IDLE mode (no foreground apps)..."
        device_cmd="$GENERATE_BIN --model-path $MODEL_PATH --prompt-file $PROMPT_FILE --num-tokens $NUM_TOKENS -b $BACKEND"
        
        python3 scripts/android_profile.py \
            --cmd "$device_cmd" \
            --output-dir "results/data" \
            --suffix "_idle"
    else
        # Stress test with foreground app switching
        echo "Running with foreground apps: $apps"
        device_cmd="$GENERATE_BIN --model-path $MODEL_PATH --prompt-file $PROMPT_FILE --num-tokens $NUM_TOKENS -b $BACKEND"
        
        python3 "$STRESS_SCRIPT" \
            --cmd "$device_cmd" \
            --duration 60 \
            --switch-interval 10 \
            --background-apps "$apps"
        
        # Pull output log
        adb pull /data/local/tmp/stress_output.log "results/data/stress_${scenario}_$(date +%Y%m%d_%H%M%S).log"
    fi
    
    echo ""
    sleep 5  # Cool down between tests
done

echo "=== Test Complete ==="
echo "Check results/data/ for results"
