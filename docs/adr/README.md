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
