#!/bin/bash
adb -s R3CY408S4HN shell "cd /data/local/tmp && export LD_LIBRARY_PATH=/data/local/tmp:\$LD_LIBRARY_PATH && export LLMRS_PER_TOKEN_MS=1 && timeout 600 ./generate \
  --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
  --secondary-gguf /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0-aos.auf \
  --secondary-layout aos \
  --force-swap-ratio 0.9 \
  --swap-intra-forward \
  --ignore-eos \
  --threads 6 \
  --backend opencl \
  -p 'The quick brown fox jumps' -n 100; echo END_EXIT=\$?" > /tmp/swap_v2_measurements_liswap4/v2_diag.log 2>&1
echo "v2 diag done"
