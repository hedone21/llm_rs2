#!/bin/bash
# Round 7: 개선된 H2O (per-layer MAX + 50:50 budget) 검증
#
# 목적: 논문 핵심 매커니즘 정렬 후 H2O 품질 변화 측정
#
# 변경 사항:
#   1. accumulate_layer: SUM → per-layer MAX
#   2. budget split: min_keep+16 → keep_ratio로 HH:Recent 비율 제어 (기본 50:50)
#   3. tracked_layers: 3 → 0 (전체 레이어)
#   4. decay: 0.1 → 0.0 (decay 없음)
#
# 비교 대상 (기존 데이터):
#   B-512        — no eviction (EMR=1.000)
#   M-C-256-sl   — sliding (EMR=0.687)
#   M-C-256-h2o  — old H2O kr=0.5/rw=128/d=0.1 (EMR=0.593)
#   H-01~H-09   — old H2O parameter sweep (EMR=0.503~0.593)

set -euo pipefail

BINARY="./target/release/generate"
MODEL="/home/go/Workspace/models"
RESULTS="experiments/results"
CONFIGS="experiments/configs"
PROMPT="The history of artificial intelligence began in antiquity, with myths, stories and rumors of"

mkdir -p "$RESULTS"

run_experiment() {
    local name="$1"
    local kr="$2"
    local tokens="$3"
    local schedule="$4"

    echo "━━━ Running: $name (kr=$kr, tokens=$tokens, schedule=$schedule) ━━━"

    $BINARY -m "$MODEL" \
        -p "$PROMPT" \
        -n "$tokens" \
        --greedy \
        --eviction-policy h2o \
        --h2o-keep-ratio "$kr" \
        --h2o-decay 0.0 \
        --h2o-tracked-layers 0 \
        --experiment-schedule "$CONFIGS/${schedule}.json" \
        --experiment-output "$RESULTS/${name}.jsonl" \
        --experiment-sample-interval 1 \
        --experiment-logits-topk 10 \
        2>&1 | grep -E '^\[Experiment\]|^\[Resilience\]|^\[CacheManager\]|^H2O'

    echo ""
}

echo "========================================"
echo "  Round 7: Improved H2O Verification"
echo "========================================"
echo ""

# ── Group A: 기본 비교 (kr=0.5, Memory Critical@256, 512 tokens) ──
# 직접 비교: M-C-256-h2o (old EMR=0.593), M-C-256-sl (EMR=0.687)
echo "── Group A: Default comparison (kr=0.5 @ 256) ──"
run_experiment "N-01" 0.5 512 "memory_critical_256"

# ── Group B: keep_ratio 변수 (kr=0.3, 0.5, 0.7) ──
# 기존: H-01/H-04/H-07 모두 동일 EMR → keep_ratio 무효과
# 개선 후: keep_ratio가 HH:Recent 비율을 직접 결정 → 효과 있어야 함
echo "── Group B: keep_ratio variation ──"
run_experiment "N-02" 0.3 512 "memory_critical_256"
run_experiment "N-03" 0.7 512 "memory_critical_256"

# ── Group C: 주입 위치 변수 ──
# 비교: P-128(EMR=1.000), P-384(EMR=0.885), P-448(EMR=0.890)
echo "── Group C: Injection position variation ──"
run_experiment "N-04" 0.5 512 "memory_critical_128"
run_experiment "N-05" 0.5 512 "memory_critical_384"
run_experiment "N-06" 0.5 512 "memory_critical_448"

# ── Group D: 1024 토큰 비교 ──
# 비교: R-C-512-h2o(EMR=0.546), RP-256(EMR=0.297)
echo "── Group D: 1024 tokens ──"
run_experiment "N-07" 0.5 1024 "memory_critical_512"
run_experiment "N-08" 0.5 1024 "memory_critical_256"

echo "========================================"
echo "  Round 7 Complete"
echo "========================================"
echo ""
echo "Results in: $RESULTS/"
echo ""
echo "Next: run analysis"
echo "  python experiments/analysis/round_report.py --round 7"
