#!/bin/bash
set -e

# Usage: ./run_remote_linux.sh <user@host> <binary_name> [args...]
# Example: ./run_remote_linux.sh pi@192.168.1.100 generate --model ...

# Default target architecture
TARGET_ARCH="aarch64-unknown-linux-gnu"

REMOTE_TARGET=$1
BIN_NAME=$2
shift 2
ARGS="$@"

if [ -z "$REMOTE_TARGET" ] || [ -z "$BIN_NAME" ]; then
    echo "Usage: $0 <user@host> <binary_name> [args...]"
    exit 1
fi

# 1. Build
echo "[1/4] Building '$BIN_NAME' for $TARGET_ARCH..."
cargo build --target $TARGET_ARCH --release --bin "$BIN_NAME"

# 2. Setup Remote Directory
echo "[2/4] Setting up remote directory on $REMOTE_TARGET..."
REMOTE_DIR="/tmp/llm_rs2_tests"
ssh $REMOTE_TARGET "mkdir -p $REMOTE_DIR"

# 3. Transfer
echo "[3/4] Transferring binary to $REMOTE_TARGET:$REMOTE_DIR..."
LOCAL_PATH="target/$TARGET_ARCH/release/$BIN_NAME"
scp "$LOCAL_PATH" "$REMOTE_TARGET:$REMOTE_DIR/$BIN_NAME"
ssh $REMOTE_TARGET "chmod +x $REMOTE_DIR/$BIN_NAME"

# 4. Run
echo "[4/4] Executing on remote device..."
echo "Command: $REMOTE_DIR/$BIN_NAME $ARGS"
echo "----------------------------------------"
ssh $REMOTE_TARGET "LD_LIBRARY_PATH=$REMOTE_DIR $REMOTE_DIR/$BIN_NAME $ARGS"
