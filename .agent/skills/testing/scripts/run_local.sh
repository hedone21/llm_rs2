#!/bin/bash
set -e

# Usage: ./run_local.sh <binary_name> [args...]
# Example: ./run_local.sh generate --model ...

BIN_NAME=$1
shift
ARGS="$@"

if [ -z "$BIN_NAME" ]; then
    echo "Usage: $0 <binary_name> [args...]"
    exit 1
fi

echo "[1/2] Building '$BIN_NAME' for local host (release)..."
cargo build --release --bin "$BIN_NAME"

LOCAL_PATH="target/release/$BIN_NAME"

echo "[2/2] Executing locally..."
echo "Command: $LOCAL_PATH $ARGS"
echo "----------------------------------------"
$LOCAL_PATH $ARGS
