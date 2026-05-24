#!/bin/bash
# Feature matrix build check (§13.8-B B-5).
#
# 다중 feature 조합 cargo build 통과 검증. 매번 sanity_check 시 돌리기엔 시간
# 부담이 커서 별도 스크립트로 분리. CI 및 release 직전 권장.
#
# Usage: ./.agent/skills/developing/scripts/feature_matrix.sh [bin_name]
#   bin_name: 측정 대상 binary (default: argus_cli)

set -e

BIN="${1:-argus_cli}"
echo "🔧 Feature Matrix Build Check (bin: $BIN)"
echo ""

# Matrix entries — 한 줄당 "label|cargo args" 형식.
MATRIX=(
  "default                  |"
  "--no-default-features    |--no-default-features"
  "+opencl                  |--no-default-features --features opencl"
  "+opencl,resilience       |--no-default-features --features opencl,resilience"
  "+cuda (PC)               |--no-default-features --features cuda"
  "+cuda-embedded (Jetson)  |--no-default-features --features cuda-embedded"
  "+qnn                     |--no-default-features --features qnn"
  "+vulkan                  |--no-default-features --features vulkan"
  "+resilience              |--no-default-features --features resilience"
)

PASS=0
FAIL=0
FAILED_LABELS=()

for entry in "${MATRIX[@]}"; do
  label="${entry%|*}"
  args="${entry#*|}"
  # trim
  label=$(echo "$label" | sed 's/[[:space:]]*$//')
  args=$(echo "$args" | sed 's/^[[:space:]]*//')

  printf "[%-26s] " "$label"
  if cargo build --release --bin "$BIN" $args 2>/dev/null 1>/dev/null; then
    echo "✅ PASS"
    PASS=$((PASS + 1))
  else
    echo "❌ FAIL"
    FAIL=$((FAIL + 1))
    FAILED_LABELS+=("$label")
  fi
done

echo ""
echo "Summary: $PASS PASS, $FAIL FAIL"
if [ $FAIL -gt 0 ]; then
  echo ""
  echo "Failed feature combinations:"
  for l in "${FAILED_LABELS[@]}"; do
    echo "  - $l"
  done
  echo ""
  echo "Re-run individually for diagnostic, e.g.:"
  echo "  cargo build --release --bin $BIN <args>"
  exit 1
fi
echo "✅ All feature combinations build."
