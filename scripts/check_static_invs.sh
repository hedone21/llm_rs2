#!/usr/bin/env bash
# scripts/check_static_invs.sh
# Static INV 자동 검증 (17개 static-only INV)
# 자동 검증 7개 + 수동 체크리스트 10개
# 종료코드: 0 = 자동 검증 전수 통과, 1 = 위반 존재

set -euo pipefail

PASS=0
FAIL=0
MANUAL=0

pass()   { echo "  ✅ $1"; PASS=$((PASS+1)); }
fail()   { echo "  ❌ $1"; FAIL=$((FAIL+1)); }
manual() { echo "  🔍 $1"; MANUAL=$((MANUAL+1)); }

echo "=== Static INV 자동 검증 ==="

# ── INV-001/010: Engine ↛ Manager 직접 코드 의존 금지 ──
echo ""
echo "[INV-001/010] Engine-Manager 직접 코드 의존 금지"
if grep -q 'llm_manager' engine/Cargo.toml 2>/dev/null; then
  fail "engine/Cargo.toml이 llm_manager에 의존"
else
  pass "engine → manager 의존 없음"
fi
if grep -q 'llm_rs2' manager/Cargo.toml 2>/dev/null; then
  fail "manager/Cargo.toml이 llm_rs2에 의존"
else
  pass "manager → engine 의존 없음"
fi

# ── INV-011: Shared ↛ Engine/Manager 의존 금지 ──
echo ""
echo "[INV-011] Shared는 Engine/Manager에 의존 금지"
if grep -qE 'llm_rs2|llm_manager' shared/Cargo.toml 2>/dev/null; then
  fail "shared/Cargo.toml이 engine 또는 manager에 의존"
else
  pass "shared → engine/manager 의존 없음"
fi

# ── INV-002: NEON 코드는 #[cfg(target_arch = "aarch64")] 게이트 내부 ──
echo ""
echo "[INV-002] NEON SIMD는 ARM64에서만 활성화"
# engine/src/backend/cpu/mod.rs에서 neon 모듈이 cfg 게이트 내부인지 확인
if grep -B1 'mod neon' engine/src/backend/cpu/mod.rs 2>/dev/null | grep -q 'cfg(target_arch = "aarch64")'; then
  pass "neon 모듈이 aarch64 cfg 게이트 내부"
else
  fail "neon 모듈에 aarch64 cfg 게이트 없음"
fi
# std::arch::aarch64 사용이 모두 cfg 게이트 내부인지 확인
# 각 파일에서 use std::arch::aarch64 행의 위 5줄 내에 #[cfg(target_arch 가 있어야 함
NEON_LEAK=""
while IFS= read -r f; do
  # neon.rs는 mod.rs의 모듈 레벨 cfg 게이트로 보호됨 — 스킵
  basename "$f" | grep -q '^neon' && continue
  # 각 사용 위치의 위 5줄 내에 인라인 cfg 게이트가 있어야 함
  while IFS=: read -r lineno _; do
    start=$((lineno > 5 ? lineno - 5 : 1))
    if ! sed -n "${start},${lineno}p" "$f" | grep -q 'cfg(target_arch'; then
      NEON_LEAK="${NEON_LEAK}  ${f}:${lineno}\n"
    fi
  done < <(grep -n 'std::arch::aarch64' "$f")
done < <(grep -rl 'std::arch::aarch64' engine/src/ 2>/dev/null || true)
if [ -z "$NEON_LEAK" ]; then
  pass "aarch64 intrinsic이 모두 cfg 게이트 내부"
else
  fail "aarch64 intrinsic이 cfg 게이트 밖에 존재:\n$NEON_LEAK"
fi

# ── INV-065: Backend trait = Send + Sync ──
echo ""
echo "[INV-065] Backend trait은 Send + Sync"
if grep -q 'pub trait Backend: Send + Sync' engine/src/core/backend.rs 2>/dev/null; then
  pass "Backend trait에 Send + Sync bound 존재"
else
  fail "Backend trait에 Send + Sync bound 없음"
fi

# ── INV-080: async 런타임 사용 금지 ──
echo ""
echo "[INV-080] async 런타임 사용 금지"
ASYNC_DEPS=$(grep -rlE '^tokio|^async-std' engine/Cargo.toml manager/Cargo.toml shared/Cargo.toml 2>/dev/null || true)
if [ -n "$ASYNC_DEPS" ]; then
  fail "async 런타임 의존성 발견: $ASYNC_DEPS"
else
  pass "Cargo.toml에 tokio/async-std 없음"
fi
ASYNC_FN=$(grep -rn 'async fn' engine/src/ manager/src/ shared/src/ --include='*.rs' 2>/dev/null || true)
if [ -n "$ASYNC_FN" ]; then
  fail "async fn 발견:\n$ASYNC_FN"
else
  pass "소스 코드에 async fn 없음"
fi

# ── INV-028: shared/ 새 필드에 #[serde(default)] 필수 ──
echo ""
echo "[INV-028] shared/ Capability 필드에 #[serde(default)]"
# Capability struct의 필드들이 #[serde(default)]를 갖는지 확인
# shared/src/lib.rs에서 Capability 구조체 내부를 검사
if [ -f shared/src/lib.rs ]; then
  # Capability struct 존재 여부 확인
  if grep -q 'struct Capability' shared/src/lib.rs; then
    # Capability 내부 pub 필드 중 #[serde(default)] 없는 것 찾기
    IN_CAP=0
    MISSING_DEFAULT=0
    while IFS= read -r line; do
      if echo "$line" | grep -q 'struct Capability'; then
        IN_CAP=1
        continue
      fi
      if [ "$IN_CAP" -eq 1 ]; then
        # struct 종료 감지
        echo "$line" | grep -q '^}' && IN_CAP=0 && continue
        # pub 필드인데 이전 줄에 serde(default)가 없으면 체크
        if echo "$line" | grep -qE '^\s+pub \w+:'; then
          MISSING_DEFAULT=$((MISSING_DEFAULT+1))
        fi
        if echo "$line" | grep -q 'serde(default)'; then
          MISSING_DEFAULT=$((MISSING_DEFAULT-1))
        fi
      fi
    done < shared/src/lib.rs
    if [ "$MISSING_DEFAULT" -le 0 ]; then
      pass "Capability 필드에 serde(default) 존재"
    else
      fail "Capability 필드 ${MISSING_DEFAULT}개에 serde(default) 누락"
    fi
  else
    pass "Capability struct 없음 (검사 불필요)"
  fi
else
  fail "shared/src/lib.rs 파일 없음"
fi

# ── 수동 검토 체크리스트 (10개 INV) ──
echo ""
echo "=== 수동 검토 필요 항목 ==="
manual "INV-012: Backend trait 우회 직접 호출 금지 (코드 리뷰)"
manual "INV-013: Monitor 스레드 장애 미전파, 독립 OS 스레드 (아키텍처 검토)"
manual "INV-018: 추론 루프(Prefill/Decode) 단일 스레드 (아키텍처 검토)"
manual "INV-027: Shared serde 어노테이션 변경 = 프로토콜 버전 변경 (코드 리뷰)"
manual "INV-045: primary_domain 매핑 고정 (코드 검토)"
manual "INV-051: 동시 적용 시 전체 relief 귀속, 개별 분리 불가 (설계 한계)"
manual "INV-060: CommandExecutor.poll() 토큰당 최대 1회 호출 (코드 구조)"
manual "INV-061: ExecutionPlan 생성 즉시 소비, 1회성 (코드 구조)"
manual "INV-063: MessageLoop 스레드는 Transport 유일 소유자 (ownership)"
manual "INV-084: ActionSelector stateless, predict 읽기 전용 (코드 구조)"

# ── 요약 ──
echo ""
echo "=== 결과 ==="
echo "  자동 통과: $PASS"
echo "  자동 실패: $FAIL"
echo "  수동 검토: $MANUAL"

if [ "$FAIL" -gt 0 ]; then
  exit 1
fi
exit 0
