#!/usr/bin/env bash
# scripts/check_spec_coverage.sh
# Spec → Arch → Test 3계층 파이프라인 검증
# 종료코드: 0 = 전수 커버, 1 = 누락 존재

set -euo pipefail

ERRORS=0

# ─── static 전용 INV (테스트 제외 대상) ───
STATIC_INVS="INV-001 INV-002 INV-010 INV-011 INV-012 INV-013 INV-018 INV-027 INV-028 INV-045 INV-051 INV-060 INV-061 INV-063 INV-065 INV-080 INV-084"

# ═══════════════════════════════════════════════════
# [1] Spec → Arch: arch/ 파일이 spec/과 1:1 대응하는지
# ═══════════════════════════════════════════════════
ARCH_MISSING=""
for spec_file in spec/[0-4]*.md; do
  base=$(basename "$spec_file")
  if [ ! -f "arch/$base" ]; then
    ARCH_MISSING="${ARCH_MISSING}  ${base}\n"
  fi
done

if [ -n "$ARCH_MISSING" ]; then
  COUNT=$(echo -e "$ARCH_MISSING" | grep -c '[^ ]' || true)
  echo "[1/Spec→Arch] arch/ 파일 누락 ${COUNT}건:"
  echo -e "$ARCH_MISSING"
  ERRORS=1
fi

# ═══════════════════════════════════════════════════
# [2] INV → Test: INV-ID가 tests/spec/에 대응되는지
# ═══════════════════════════════════════════════════
SPEC_INVS=$(grep -oE 'INV-[0-9]+' spec/41-invariants.md | sort -u)

TEST_INVS=""
if [ -d "engine/tests/spec" ] || [ -d "manager/tests/spec" ]; then
  TEST_INVS=$(grep -roE 'inv_?[0-9]+' engine/tests/spec/ manager/tests/spec/ 2>/dev/null \
    | sed 's/.*inv_\{0,1\}/INV-/' | sort -u || true)
fi

INV_TEST_MISSING=""
for inv in $SPEC_INVS; do
  echo "$STATIC_INVS" | grep -qwF "$inv" && continue
  if ! echo "$TEST_INVS" | grep -qwF "$inv"; then
    INV_TEST_MISSING="${INV_TEST_MISSING}  ${inv}\n"
  fi
done

if [ -n "$INV_TEST_MISSING" ]; then
  COUNT=$(echo -e "$INV_TEST_MISSING" | grep -c '[^ ]' || true)
  echo "[2/INV→Test] 누락 ${COUNT}건:"
  echo -e "$INV_TEST_MISSING"
  ERRORS=1
fi

# ═══════════════════════════════════════════════════
# [3] Static INV 자동 검증
# ═══════════════════════════════════════════════════
echo ""
echo "[3/Static INV] 자동 검증"
if [ -x "scripts/check_static_invs.sh" ]; then
  if ! scripts/check_static_invs.sh; then
    echo "[3/Static INV] ❌ 위반 발견"
    ERRORS=1
  else
    echo "[3/Static INV] ✅ 전수 통과"
  fi
else
  echo "[3/Static INV] ⚠️  scripts/check_static_invs.sh 미발견, 스킵"
fi

# ═══════════════════════════════════════════════════
# [4] 비-INV 요구사항 추적성
# ═══════════════════════════════════════════════════
echo ""
echo "[4/요구사항 추적성] spec/ PREFIX-NNN → tests/ 참조 분석"

# spec/에서 모든 PREFIX-NNN ID 추출 (INV 제외)
ALL_IDS_FILE=$(mktemp)
grep -ohE '\[(SYS|PROTO|MSG|SEQ|MGR|MGR-ALG|MGR-DAT|ENG|ENG-ST|ENG-ALG|ENG-DAT|CROSS)-[0-9]+\]' spec/[0-4]*.md 2>/dev/null \
  | tr -d '[]' | sort -u > "$ALL_IDS_FILE" || true

# tests/ 및 src/ 코드에서 참조되는 ID 수집
TEST_IDS_FILE=$(mktemp)
{
  grep -rohE '(SYS|PROTO|MSG|SEQ|MGR|MGR-ALG|MGR-DAT|ENG|ENG-ST|ENG-ALG|ENG-DAT|CROSS)-[0-9]+' \
    engine/tests/ manager/tests/ engine/src/ manager/src/ 2>/dev/null || true
} | sort -u > "$TEST_IDS_FILE"

# 접두사별 집계 (임시 파일 기반, bash 3.x 호환)
PREFIXES=$(sed 's/-[0-9]*$//' "$ALL_IDS_FILE" | sort -u)
printf "  %-12s | %5s | %5s | %5s | %s\n" "접두사" "전체" "참조됨" "미참조" "커버리지"
printf "  %-12s-|------:|------:|------:|----------\n" "------------"
TOTAL_ALL=0
TOTAL_COV=0
for prefix in $PREFIXES; do
  total=$(grep -c "^${prefix}-" "$ALL_IDS_FILE" || true)
  covered=0
  while IFS= read -r id; do
    if grep -qwF "$id" "$TEST_IDS_FILE"; then
      covered=$((covered + 1))
    fi
  done < <(grep "^${prefix}-" "$ALL_IDS_FILE")
  missing=$((total - covered))
  if [ "$total" -gt 0 ]; then
    pct=$((covered * 100 / total))
  else
    pct=0
  fi
  printf "  %-12s | %5d | %5d | %5d | %3d%%\n" "$prefix" "$total" "$covered" "$missing" "$pct"
  TOTAL_ALL=$((TOTAL_ALL + total))
  TOTAL_COV=$((TOTAL_COV + covered))
done
if [ "$TOTAL_ALL" -gt 0 ]; then
  TOTAL_PCT=$((TOTAL_COV * 100 / TOTAL_ALL))
else
  TOTAL_PCT=0
fi
printf "  %-12s | %5d | %5d | %5d | %3d%%\n" "합계" "$TOTAL_ALL" "$TOTAL_COV" "$((TOTAL_ALL - TOTAL_COV))" "$TOTAL_PCT"
rm -f "$ALL_IDS_FILE" "$TEST_IDS_FILE"

# ═══════════════════════════════════════════════════
# [5] INV 커버리지 통합 통계
# ═══════════════════════════════════════════════════
echo ""
echo "[5/INV 통합 통계]"
INV_TOTAL=$(echo "$SPEC_INVS" | wc -w | tr -d ' ')
STATIC_COUNT=$(echo "$STATIC_INVS" | wc -w | tr -d ' ')
TESTABLE=$((INV_TOTAL - STATIC_COUNT))
TESTED=$(echo "$TEST_INVS" | wc -w | tr -d ' ')
# TESTED는 test 파일에서 추출된 것이므로 TESTABLE보다 클 수 있음 (restatement 등)
if [ "$TESTED" -gt "$TESTABLE" ]; then
  TESTED=$TESTABLE
fi
if [ "$TESTABLE" -gt 0 ]; then
  INV_PCT=$((TESTED * 100 / TESTABLE))
else
  INV_PCT=0
fi
echo "  전체 INV: ${INV_TOTAL}"
echo "  Static 전용: ${STATIC_COUNT}"
echo "  테스트 대상: ${TESTABLE}"
echo "  테스트 구현: ${TESTED} (${INV_PCT}%)"

# ═══════════════════════════════════════════════════
# [6] #[ignore] 감지 + 빈 테스트 파일 감지
# ═══════════════════════════════════════════════════
echo ""
echo "[6/테스트 품질]"

# #[ignore] 감지
IGNORED=$(grep -rn '#\[ignore\]' engine/tests/spec/ manager/tests/spec/ 2>/dev/null || true)
if [ -n "$IGNORED" ]; then
  echo "  ⚠️  #[ignore] 테스트 발견:"
  echo "$IGNORED" | sed 's/^/    /'
else
  echo "  ✅ #[ignore] 테스트 없음"
fi

# assert 없는 테스트 파일 감지
EMPTY_TESTS=""
for f in engine/tests/spec/test_*.rs manager/tests/spec/test_*.rs; do
  [ -f "$f" ] || continue
  if ! grep -q 'assert' "$f"; then
    EMPTY_TESTS="${EMPTY_TESTS}  ${f}\n"
  fi
done
if [ -n "$EMPTY_TESTS" ]; then
  echo "  ⚠️  assert 없는 테스트 파일:"
  echo -e "$EMPTY_TESTS" | sed 's/^/    /'
  ERRORS=1
else
  echo "  ✅ 모든 테스트 파일에 assert 존재"
fi

# ─── 최종 결과 ───
echo ""
if [ "$ERRORS" -eq 0 ]; then
  echo "=== 전수 통과 ==="
else
  echo "=== 누락/위반 발견 (exit 1) ==="
fi

exit $ERRORS
