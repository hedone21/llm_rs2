# Architecture Decision Records (ADR)

본 디렉토리는 llm_rs2 프로젝트의 정식 아키텍처 결정 (Architecture Decision Record) 을 기록한다. 일반 설계 문서 (`docs/*.md` 와 `arch/*.md`) 와 다음 점에서 차이:

| 측면 | ADR (`docs/adr/`) | 일반 설계 (`arch/`, `docs/`) |
|---|---|---|
| 목적 | 명시 정책 / 가정 / 패러다임의 **반전** 또는 **확립** 의 정당화 영구 기록 | 컴포넌트 구조 / 인터페이스 / 흐름 기술 |
| 변경 | append-only (Status 만 갱신: Accepted → Deprecated / Superseded by ADR-NNNN) | 가변 (코드 변경에 따라 갱신) |
| 트리거 | 본 프로젝트 명시 코드 정책 / 코멘트 / spec 의 반전 | 새 컴포넌트 / 기능 추가 |
| Granularity | 1 결정 = 1 file | 1 sprint / 1 컴포넌트 = 1 file |

## ID 컨벤션

- 4-digit zero-padded number (`0001`, `0002`, ...).
- 충돌 방지를 위해 새 ADR 작성 전 `ls docs/adr/` 로 다음 번호 확인.
- 파일명 패턴: `NNNN-<kebab-case-title>.md`

## 작성 의무

ADR 는 다음 결정 유형에서 작성 의무:
- 본 프로젝트의 명시 정책 / 주석 / 코드 컨벤션 반전 (예: ADR-0001 = kv_cache_ops.rs:53 정책 반전)
- 핵심 trait / abstraction paradigm 의 변경 (Generic ↔ Trait object, RCU ↔ Lock, sync ↔ async 등)
- 5년 시야의 cross-cutting 결정 (binary 분할, IPC 프로토콜 버전, 외부 API 컨벤션 등)
- Spec 의 핵심 INV 폐기 / 추가 (예: INV-KVBUNDLE-SYNC 폐기 → (β) sync 모델 자동 처리)

다음 결정은 ADR 작성 의무 **없음** (일반 arch/ 또는 docs/ 에 기록):
- 컴포넌트 내부 리팩토링
- 새 stage / handler / 기능 추가
- 성능 최적화

## 인덱스

| ID | 제목 | Status | Date |
|---|---|---|---|
| [ADR-0001](0001-kv-dispatch-paradigm.md) | KV Cache Dispatch Paradigm — Generic Monomorphization → Trait Object Transition | Accepted | 2026-05-28 |
| [ADR-0002](0002-pressure-scalar-lossy-unification.md) | Pressure 스칼라 — lossy 단일화 (graded 입력 융합 ⊥ mode 출력 분리) | Accepted | 2026-06-02 |
| [ADR-0003](0003-extension-mechanism-static-crates.md) | 확장 메커니즘 — 정적 링크 technique crate + 자동 등록, 런타임 `.so` 보류 | Accepted | 2026-06-05 |
| [ADR-0004](0004-kvcachestage-plan-returning-trait.md) | `KVCacheStage` — 단일 plan-returning trait (per-head keep + 가중 merge 통합) | Accepted | 2026-06-05 |
| [ADR-0005](0005-format-backend-capability-plugin-unification.md) | Format · Backend Capability 확장 — plugin 패턴 통일 (descriptor + backend-owned kernel + 3축 평행 registry, crate→`.so` phasing) | Accepted | 2026-06-06 |
| [ADR-0006](0006-weight-stage-plan-returning-unification.md) | weight 축 stage 통일 — plan-returning `WeightStage` (KVCacheStage 형제; 결정 ⊥ 변형(엔진 독점) ⊥ pacing 분리, precision⊥dispatch, `WEIGHT_STAGES` 평행 registry, Seam B = Phase β 의존) | Accepted | 2026-06-07 |
| [ADR-0007](0007-opaque-dtype-kv-format-unlock.md) | opaque-dtype — descriptor-운반 KV format 의 `DType`-우회 해금 (zero-compile `.so` 북극성, GATE-B host) | Accepted | 2026-06-08 |
| [ADR-0008](0008-opaque-kv-production-integration.md) | opaque KV format production 통합 — `KVCache` 흡수 + `is_q4`→descriptor-keyed (실추론 grow/eviction/D2O) | Accepted | 2026-06-08 |
| [ADR-0009](0009-gate-c-stage-dlopen-plugin.md) | GATE-C — Stage 축 `.so` cdylib dlopen plugin 승격 (북극성 zero-compile install, ADR-0007 D6 해결) | Accepted (D2/D6 부분 supersede by 0010) | 2026-06-09 |
| [ADR-0010](0010-gate-c-multi-vtable-bundle-abi.md) | GATE-C 멀티-vtable bundle ABI — 한 `.so` 다수 capability + cross-axis open-once dispatch (ADR-0009 D2 supersede + production 배선 해결) | Proposed | 2026-06-09 |
| [ADR-0011](0011-kv-read-plan-surface.md) | KV read-plan 표면 — `KVReadStage` plan-returning trait ("무엇을 읽을지" 4번째 plugin 표면; Quest 선택 읽기 + KVSwap prefetch 통합, format `attention_into_selected` capability opt-in, **구현 보류=리팩토링 머지 후**) | Proposed | 2026-06-12 |
| [ADR-0012](0012-session-prefix-cache-snapshot.md) | 세션 KV persistence (prefix cache) Tier 1 — format snapshot/restore capability + 세션 prefix 저장/복원 | Proposed | 2026-06-12 |
