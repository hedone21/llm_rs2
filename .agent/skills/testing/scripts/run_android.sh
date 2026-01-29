#!/bin/bash
set -e

# Usage: ./run_android.sh <binary_name> [args...]
# Example: ./run_android.sh generate --model ...

BIN_NAME=$1
shift
ARGS="$@"

# 0. Check if android.source is available (optional safety check)
if [ ! -f "android.source" ]; then
    echo "Error: android.source not found in root workspace"
    exit 1
fi

# 1. Setup Environment
echo "[1/4] Setting up environment..."
source android.source

# 2. Build
echo "[2/4] Building '$BIN_NAME' for Android..."
cargo build --target aarch64-linux-android --release --bin "$BIN_NAME"

# 3. Push
echo "[3/4] Pushing to device..."
LOCAL_PATH="target/aarch64-linux-android/release/$BIN_NAME"
REMOTE_PATH="/data/local/tmp/$BIN_NAME"
adb push "$LOCAL_PATH" "$REMOTE_PATH"
adb shell "chmod +x $REMOTE_PATH"

# 4. Run
echo "[4/4] Executing on device..."
echo "Command: $REMOTE_PATH $ARGS"
echo "----------------------------------------"
adb shell "LD_LIBRARY_PATH=/data/local/tmp $REMOTE_PATH $ARGS"
