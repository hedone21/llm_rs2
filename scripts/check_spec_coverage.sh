#!/usr/bin/env bash
# scripts/check_spec_coverage.sh
# 사용법: ./scripts/check_spec_coverage.sh
# 종료코드: 0 = 전수 커버, 1 = 미커버 INV 존재

set -euo pipefail

# 1. spec/에서 모든 INV-ID 추출
SPEC_INVS=$(grep -roE 'INV-[0-9]+' spec/41-invariants.md \
  | sed 's/.*://' | sort -u)

# 2. tests/spec/에서 참조된 INV-ID 추출
if [ -d "engine/tests/spec" ] || [ -d "manager/tests/spec" ]; then
  TEST_INVS=$(grep -roE 'inv_[0-9]+' engine/tests/spec/ manager/tests/spec/ 2>/dev/null \
    | sed 's/.*inv_/INV-/' | sort -u)
else
  TEST_INVS=""
fi

# 3. spec에는 있지만 tests에 없는 INV 찾기
MISSING=$(comm -23 <(echo "$SPEC_INVS") <(echo "$TEST_INVS"))

if [ -n "$MISSING" ]; then
  MISSING_COUNT=$(echo "$MISSING" | wc -l | tr -d ' ')
  TOTAL_COUNT=$(echo "$SPEC_INVS" | wc -l | tr -d ' ')
  echo "WARNING: $MISSING_COUNT / $TOTAL_COUNT INV가 테스트에 미대응:"
  echo "$MISSING"
  exit 1
fi

echo "OK: 모든 INV가 테스트에 대응됨."
exit 0
