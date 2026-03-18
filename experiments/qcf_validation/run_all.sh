#!/usr/bin/env bash
# QCF Validation — Full Experiment Runner
#
# Runs all 4 benchmark phases and unified analysis.
# Estimated time: ~3.5 hours on host CPU.
#
# Usage:
#   ./run_all.sh              # Run everything
#   ./run_all.sh --phase 1    # Run only Phase 1 (PPL)
#   ./run_all.sh --phase 2    # Run only Phase 2 (NIAH)
#   ./run_all.sh --phase 3    # Run only Phase 3 (QA)
#   ./run_all.sh --phase 4    # Run only Phase 4 (MMLU)
#   ./run_all.sh --phase 5    # Run only Phase 5 (Analysis)
#   ./run_all.sh --skip-mmlu  # Skip MMLU (saves ~90 min)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
SCRIPTS="$SCRIPT_DIR/scripts"

cd "$PROJECT_ROOT"

# Parse arguments
PHASE=""
SKIP_MMLU=false
for arg in "$@"; do
    case "$arg" in
        --phase) shift; PHASE="${1:-}"; shift || true ;;
        --skip-mmlu) SKIP_MMLU=true ;;
        [1-5]) PHASE="$arg" ;;
    esac
done

# Ensure release binary is built
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  QCF Validation Experiment Suite"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

if [ ! -f target/release/generate ]; then
    echo "[build] Compiling release binary..."
    cargo build --release -p llm_rs2 --bin generate
fi

if [ ! -d models/llama3.2-1b ]; then
    echo "[ERROR] Model not found: models/llama3.2-1b"
    echo "  Download: huggingface-cli download meta-llama/Llama-3.2-1B --local-dir models/llama3.2-1b"
    exit 1
fi

START_TIME=$(date +%s)

run_phase() {
    local phase=$1
    local name=$2
    local cmd=$3
    echo ""
    echo "┌─────────────────────────────────────────────┐"
    echo "│  Phase $phase: $name"
    echo "└─────────────────────────────────────────────┘"
    local t0=$(date +%s)
    eval "$cmd"
    local t1=$(date +%s)
    echo "[Phase $phase] Finished in $((t1 - t0))s"
}

# Phase 1: PPL
if [ -z "$PHASE" ] || [ "$PHASE" = "1" ]; then
    run_phase 1 "Perplexity Sweep" \
        "python3 $SCRIPTS/run_ppl.py"
fi

# Phase 2: NIAH
if [ -z "$PHASE" ] || [ "$PHASE" = "2" ]; then
    run_phase 2 "Needle-in-a-Haystack" \
        "python3 $SCRIPTS/run_niah.py"
fi

# Phase 3: QA
if [ -z "$PHASE" ] || [ "$PHASE" = "3" ]; then
    run_phase 3 "Document QA" \
        "python3 $SCRIPTS/run_qa.py"
fi

# Phase 4: MMLU
if [ -z "$PHASE" ] || [ "$PHASE" = "4" ]; then
    if [ "$SKIP_MMLU" = true ]; then
        echo ""
        echo "[Phase 4] MMLU skipped (--skip-mmlu)"
    else
        # Download MMLU data if needed
        if [ ! -d "$SCRIPT_DIR/data/mmlu" ]; then
            echo "[Phase 4] Downloading MMLU data..."
            python3 "$SCRIPTS/run_mmlu.py" --download
        fi
        run_phase 4 "MMLU (Many-Shot ICL)" \
            "python3 $SCRIPTS/run_mmlu.py"
    fi
fi

# Phase 5: Analysis
if [ -z "$PHASE" ] || [ "$PHASE" = "5" ]; then
    run_phase 5 "Unified Analysis" \
        "python3 $SCRIPTS/analyze.py"
fi

END_TIME=$(date +%s)
TOTAL=$((END_TIME - START_TIME))
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  Total time: ${TOTAL}s ($((TOTAL / 60))m $((TOTAL % 60))s)"
echo "  Results: $SCRIPT_DIR/results/"
echo "  Plots:   $SCRIPT_DIR/plots/"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

notify-send "llm.rs" "QCF validation complete (${TOTAL}s)" 2>/dev/null || true
