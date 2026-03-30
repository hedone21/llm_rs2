#!/usr/bin/env bash
# scripts/check_spec_coverage.sh
# Spec → Arch → Test 3계층 파이프라인 검증
# 누락이 없으면 아무것도 출력하지 않는다.
# 종료코드: 0 = 전수 커버, 1 = 누락 존재

set -euo pipefail

ERRORS=0

# ─── static 전용 INV (테스트 제외 대상) ───
STATIC_INVS="INV-001 INV-002 INV-010 INV-011 INV-012 INV-013 INV-018 INV-027 INV-028 INV-045 INV-051 INV-060 INV-061 INV-063 INV-065 INV-080 INV-084"

# ─── 1. Spec → Arch: arch/ 파일이 spec/과 1:1 대응하는지 ───
ARCH_MISSING=""
for spec_file in spec/[0-4]*.md; do
  base=$(basename "$spec_file")
  if [ ! -f "arch/$base" ]; then
    ARCH_MISSING="${ARCH_MISSING}  ${base}\n"
  fi
done

if [ -n "$ARCH_MISSING" ]; then
  COUNT=$(echo -e "$ARCH_MISSING" | grep -c '[^ ]' || true)
  echo "[Spec→Arch] arch/ 파일 누락 ${COUNT}건:"
  echo -e "$ARCH_MISSING"
  ERRORS=1
fi

# ─── 2. INV → Test: INV-ID가 tests/spec/에 대응되는지 (static 제외) ───
SPEC_INVS=$(grep -oE 'INV-[0-9]+' spec/41-invariants.md | sort -u)

TEST_INVS=""
if [ -d "engine/tests/spec" ] || [ -d "manager/tests/spec" ]; then
  TEST_INVS=$(grep -roE 'inv_?[0-9]+' engine/tests/spec/ manager/tests/spec/ 2>/dev/null \
    | sed 's/.*inv_\?/INV-/' | sort -u || true)
fi

INV_TEST_MISSING=""
for inv in $SPEC_INVS; do
  # static 전용이면 스킵
  echo "$STATIC_INVS" | grep -qwF "$inv" && continue
  # 테스트에 있는지 확인
  if ! echo "$TEST_INVS" | grep -qwF "$inv"; then
    INV_TEST_MISSING="${INV_TEST_MISSING}  ${inv}\n"
  fi
done

if [ -n "$INV_TEST_MISSING" ]; then
  COUNT=$(echo -e "$INV_TEST_MISSING" | grep -c '[^ ]' || true)
  echo "[INV→Test] 누락 ${COUNT}건:"
  echo -e "$INV_TEST_MISSING"
  ERRORS=1
fi

exit $ERRORS
