#!/bin/bash
set -e

echo "🔍 Running Sanity Check..."

# 1. Format Check
echo "[1/2] Checking Formatting..."
if cargo fmt --all -- --check; then
    echo "✅ Format OK"
else
    echo "❌ Format Issues Found. Retrieving details..."
    cargo fmt --all
    echo "⚠️  Auto-formatting applied. Please verify changes."
fi

# 2. Lint Check (Clippy)
echo ""
echo "[2/2] Running Linter (Clippy)..."
# We run clippy and capture output, but don't fail immediately on warnings
# to allow the user to see them.
cargo clippy --workspace -- -D warnings || true

echo ""
echo "📝 Sanity Check Complete."
echo "If you see clippy errors above, please fix them before committing."
