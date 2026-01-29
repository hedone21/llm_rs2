#!/bin/bash
set -e

# Usage: ./auto_profile.sh --cmd "..." --output-name "my_profile_run"
# Example: ./auto_profile.sh --cmd "/data/local/tmp/generate ..." --output-name "cpu_short"

# Defaults
CMD=""
OUTPUT_NAME=""

# Parse Args
while [[ $# -gt 0 ]]; do
  case $1 in
    --cmd)
      CMD="$2"
      shift 2
      ;;
    --output-name)
      OUTPUT_NAME="$2"
      shift 2
      ;;
    *)
      echo "Unknown argument: $1"
      exit 1
      ;;
  esac
done

if [ -z "$CMD" ] || [ -z "$OUTPUT_NAME" ]; then
    echo "Usage: $0 --cmd '<android_command>' --output-name '<run_identifier>'"
    exit 1
fi

echo "=========================================="
echo "🚀 Starting Automated Profiling Pipeline"
echo "ID: $OUTPUT_NAME"
echo "CMD: $CMD"
echo "=========================================="

# 1. Run Profiling
echo ""
echo "[1/3] Running On-Device Profiling..."
# Note: we use existing android_profile.py
# Using a temp output dir initially or direct to benchmarks/data?
# Best practice: direct to benchmarks/data as per GUIDE
python3 scripts/android_profile.py --cmd "$CMD" --output-dir benchmarks/data

# Find the generated file. android_profile.py generates a timestamped filename.
# We need to find the most recent one in benchmarks/data
LATEST_JSON=$(ls -t benchmarks/data/*.json | head -n 1)
echo "Captured Profile: $LATEST_JSON"

# 2. Visualize
echo ""
echo "[2/3] Generating Visualization..."
# Construct output PNG path
BASENAME=$(basename "$LATEST_JSON" .json)
PNG_PATH="benchmarks/plots/${BASENAME}.png"

python3 scripts/visualize_profile.py "$LATEST_JSON" --output "$PNG_PATH"
echo "Saved Plot: $PNG_PATH"

# 3. Update Summary
echo ""
echo "[3/3] Updating Benchmark README..."
python3 scripts/update_benchmark_summary.py

echo "=========================================="
echo "✅ Profiling Pipeline Complete!"
echo "Check benchmarks/README.md for the new entry."
