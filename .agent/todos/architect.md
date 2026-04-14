# Architect TODO

> **역할**: 시스템 설계, 트레이트/인터페이스 정의, 모듈 간 의존성 관리, 기술 결정
> **소유 영역**: `engine/src/core/`, `shared/`, Cargo workspace 구조, `ARCHITECTURE.md`

---

## [P1] Manager Clock trait 도입 설계 (Phase 5 관측 학습 차단 해소)
- **Status**: DESIGN-READY (2026-04-14)
- **Sprint**: active
- **Dependencies**: 없음 (spec/arch 문서 작성 완료)
- **Description**: `manager/src/lua_policy.rs`, `pipeline.rs`, `supervisory.rs`의 `Instant::now()`/`elapsed()` 의존을 `Arc<dyn Clock>` 주입으로 대체. 시뮬레이터 `VirtualClock`이 Policy 내부 시간 판정을 지배하도록 함.
- **Design Doc**: `arch/clock_abstraction.md` (권장안: 옵션 B, LogicalInstant newtype, 5단계 PR 분할)
- **Acceptance Criteria**:
  - PR 1: Clock trait + SystemClock + LogicalInstant 도입 (무해)
  - PR 2: LuaPolicy 전환 + relief 학습 시뮬 재현
  - PR 3: Simulator VirtualClockHandle 어댑터
  - PR 4: HierarchicalPolicy + Supervisory 전환
  - INV-093(Clock monotonicity) spec 추가
- **Next Steps**: Implementer에게 PR 1 구현 의뢰

## [P3] 디바이스-모델 호환성 프로파일 설계
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: 향후 다중 모델/디바이스 확장 시 자동 설정 결정을 위한 프로파일 시스템 설계
- **Acceptance Criteria**: 프로파일 스키마, 메모리 산출 공식 문서
- **Notes**: 당장 구현 불필요, 디바이스/모델 조합이 늘어날 때 착수
