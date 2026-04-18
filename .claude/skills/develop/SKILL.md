---
name: develop
description: llm.rs 개발 파이프라인 오케스트레이터. 기능 구현, 버그 수정, 성능 최적화, 리팩토링, 조사, 계획 수립 등 모든 개발 작업을 에이전트 파이프라인으로 조율한다. '구현해줘', '추가해줘', '수정해줘', '버그', '최적화해줘', '리팩토링해줘', '조사해줘', '계획 세워줘', 'TODO', '성능 개선' 등 개발 관련 작업 요청 시 반드시 이 스킬을 사용. 단순 코드 질문이나 설명 요청은 트리거하지 않는다.
allowed-tools: Agent, Bash, Read, Glob, Grep, Edit, Write
---

# Development Orchestrator

llm.rs 개발 작업을 에이전트 파이프라인으로 오케스트레이션한다. 메인 세션이 오케스트레이터 역할을 하며, 각 에이전트를 순차/병렬로 호출한다.

## 워크플로우 분류

사용자 요청을 분석하여 아래 워크플로우 중 하나를 선택한다:

| 타입 | 트리거 예시 | 파이프라인 |
|------|-----------|-----------|
| **feature** | 새 기능, 구현, 추가 | PM → Architect → Implementer → Tester |
| **fix** | 버그, 수정, 에러, 패닉 | Implementer(분석+수정) → Tester |
| **optimize** | 성능, 최적화, 속도, 메모리 | Tester(profile) → Implementer → Tester(검증) |
| **research** | 조사, 논문, 기법 분석 | Researcher → (보고) |
| **refactor** | 리팩토링, 구조 개선, 정리 | Architect → Implementer → Tester |
| **plan** | 계획, TODO, 우선순위, 현황 | PM → (보고) |

분류가 모호하면 사용자에게 확인한다.

## 에이전트 호출 공통 규칙

- `subagent_type`: 에이전트 정의 이름 사용 (pm, architect, implementer, senior-implementer, tester, researcher)
- 이전 에이전트 결과를 다음 에이전트의 prompt에 포함하여 전달
- 독립적인 에이전트는 `run_in_background: true`로 병렬 실행 가능

### Implementer 선택 기준

| 조건 | 에이전트 | 모델 |
|------|---------|------|
| `.cl` 커널, NEON/SIMD, GPU 백엔드, 성능 커널, 복잡한 알고리즘 (D2O, KIVI, QCF) | **senior-implementer** | `model: "opus"` |
| 일반 Rust 구현, 프로토콜 연결, CLI, Manager 크레이트, 테스트, spec 테스트 | **implementer** | `model: "sonnet"` |
| 판단 불확실 시 → senior-implementer 사용 | **senior-implementer** | `model: "opus"` |

구현 작업에 두 영역이 혼합된 경우, 고급 부분은 senior-implementer, 나머지는 implementer에게 병렬 위임한다.

## Spec Triage (모든 워크플로우 공통)

모든 워크플로우 시작 전, 요청이 Spec 변경을 수반하는지 판정한다.

### 판정 기준

| Spec 관련 (O) — tests/spec/ 필수 | Spec 무관 (X) |
|-----------------------------------|---------------|
| 새 트레이트/인터페이스 정의 | 코드 리팩토링 (동작 불변) |
| FSM 상태/전이 추가·변경 | 버그 수정 (spec 동작 불변) |
| 프로토콜 메시지 추가 | 성능 최적화 |
| 값 범위/제약 조건 변경 (INV) | config 키 추가 (arch만) |
| 새 시스템 요구사항 | |
| **기존 미구현 spec ID 구현** (가장 흔한 케이스) | |

### Spec 변경이 필요한 경우

Architect에게 spec-manage 스킬 사용을 명시하여 다음을 요청한다:

1. **Spec Triage 보고**: 어떤 spec/ 파일의 어떤 ID가 영향받는지
2. **Spec 갱신**: 새 ID 추가 또는 기존 ID 수정 (spec/ 파일 + 41-invariants.md)
3. **Arch 갱신**: 대응 arch/ 파일에 구현 매핑 추가 (arch/ 존재 시)
4. **테스트 요구사항**: 추가된 INV에 대해 Implementer가 작성할 tests/spec/ 테스트 명세

이 결과를 이후 Phase의 Implementer/Tester에게 전달한다.

## 파이프라인 상세

### feature — 기능 구현

가장 완전한 파이프라인. 새 기능이나 주요 변경 시 사용.

**Phase 0: Spec Triage** (오케스트레이터)
- 요청이 새 요구사항/불변식/인터페이스를 수반하는지 판정
- Spec 변경 필요 시 → Architect에게 spec 갱신 요청 (Phase 2에 통합)

**Phase 1: 계획** (PM)
- 작업 분해, 영향 범위 예측, TODO 생성
- prompt에 포함: 사용자 요청 원문

**Phase 2: 설계 + Spec** (Architect)
- 트레이트/모듈 구조, 의존성, 영향 범위 설계
- Spec 변경 필요 시: spec/ 갱신, arch/ 매핑, tests/spec/ 테스트 명세 작성
- prompt에 포함: 사용자 요청 + PM 계획 + Spec Triage 결과
- 필요 시 Researcher를 Phase 2와 병렬로 호출하여 관련 기법 조사

**Phase 3: 구현** (Implementer)
- 코드 작성, 유닛 테스트 (`#[cfg(test)]`), sanity check (fmt + clippy + test)
- **Spec ID가 관련된 작업 (Spec Triage O)**: 반드시 `tests/spec/` 테스트를 작성해야 한다
  - `{crate}/tests/spec/test_{prefix}_{nnn}.rs` 파일 생성
  - `{crate}/tests/spec.rs` harness에 모듈 등록
  - 테스트 함수명에 spec ID 포함 (예: `test_seq_095_...`)
  - `#[cfg(test)]` inline 테스트는 **spec 추적성에 포함되지 않음** — 반드시 `tests/spec/` 필수
- prompt에 포함: Architect 설계안 + Spec 테스트 명세 + (있으면) Researcher 조사 결과
- Implementer에게 sanity check 실행을 명시적으로 요청

**Phase 4: 검증** (Tester)
- 호스트 유닛 테스트 전체 실행, 회귀 확인
- **Spec ID가 관련된 작업**: `/sanity-check --spec` 실행 필수 (spec 테스트 + 커버리지 검증)
  - `cargo test --test spec` (engine + manager + shared)
  - `scripts/check_spec_coverage.sh` — 3계층 추적성 검증
  - 커버리지 스크립트에서 새 spec ID가 누락되면 Implementer에 tests/spec/ 추가 요청
- prompt에 포함: 변경된 파일/모듈 목록 + Spec ID 목록

**Phase 5: 완료**
- 결과 종합 + 커밋 + 알림

### fix — 버그 수정

빠른 피드백 루프. 설계가 불필요한 버그 수정.

**Phase 1: 분석+수정** (Implementer)
- 증상 분석, 원인 파악, 최소 변경으로 수정, sanity check
- prompt에 포함: 에러 메시지, 재현 조건

**Phase 2: 검증** (Tester)
- 수정 검증 + 회귀 테스트
- 실패 시 → Implementer에 실패 내용 전달하여 재수정 (1회 재시도)

### optimize — 성능 최적화

측정→분석→개선→재측정 사이클.

**Phase 1: 프로파일링** (Tester)
- 현재 성능 기준선 측정
- 디바이스 미연결 시 호스트 벤치마크로 대체

**Phase 2: 최적화** (Implementer)
- 프로파일 결과 기반 병목 해소
- prompt에 포함: 프로파일 결과, 병목 지점

**Phase 3: 재검증** (Tester)
- 최적화 전후 성능 비교, 회귀 확인

### research — 기술 조사

단일 에이전트 워크플로우.

**Phase 1: 조사** (Researcher)
- 주제 검색, 논문 분석, 프로젝트 매핑
- 결과를 사용자에게 보고

구현이 필요하면 사용자 확인 후 feature 또는 refactor 워크플로우로 전환.

### refactor — 리팩토링

구조 변경에 초점. 기능 추가 없이 구조만 개선.

**Phase 0: Spec Triage** (오케스트레이터)
- 리팩토링이 인터페이스/트레이트 시그니처를 변경하는지 판정
- 트레이트 시그니처 변경 시 → Architect에게 spec 검토 요청

**Phase 1: 설계 + Spec** (Architect)
- 현재 구조 분석, 개선안 설계, 리스크 분석
- Spec 영향 시: spec/ 갱신 (보통 arch/만 갱신 — 동작 불변이므로)
- prompt에 포함: 대상 모듈/파일, 개선 방향, Spec Triage 결과

**Phase 2: 구현** (Implementer)
- 설계안 대로 구조 변경, 기존 테스트 유지, sanity check

**Phase 3: 검증** (Tester)
- 전체 회귀 테스트 — 리팩토링은 동작 변경이 없어야 함

### plan — 계획/TODO

단일 에이전트 워크플로우.

**Phase 1** (PM)
- 현황 분석, 작업 분해, 우선순위 조정, TODO 파일 업데이트

## 에러 핸들링

| 상황 | 대응 |
|------|------|
| **sanity check 실패** | Implementer에게 오류 내용과 함께 수정 재요청. 재실패 시 사용자에게 보고하고 중단 |
| **테스트 실패** | 실패 내용을 Implementer에 전달하여 수정 요청. 재실패 시 사용자에게 보고 |
| **설계 충돌/모호성** | 사용자에게 선택지 제시, 결정 대기 |
| **디바이스 미연결** | 호스트 테스트만 수행, 디바이스 테스트 스킵을 명시적으로 알림 |
| **에이전트 실패** | 1회 재시도 후 재실패 시 해당 Phase 결과 없이 진행 (보고에 누락 명시) |

## 도메인 제약 (llm.rs)

이 프로젝트의 핵심 제약을 모든 에이전트에 전달한다:

- `.cl` 커널 파일: 명시적 지시 없이 수정 금지
- KV cache 변경 시: HeadMajor 레이아웃 가정 유지
- Backend 트레이트 변경 시: CPU/OpenCL 양쪽 구현 필수
- Android 크로스 컴파일: `run_device.py`가 `hosts.toml`로 NDK env 자동 주입 (최초 1회 `bootstrap-host` 필요)
- 성능 코드: ARM64 NEON/dotprod, Adreno GPU 특성 고려
- 커밋: Conventional Commits 형식 (`type(scope): subject`)

## 완료 프로토콜

모든 워크플로우 완료 후:

1. **결과 요약**: 변경 파일, 테스트 결과, 주의사항을 간결히 보고
2. **Spec 관련 작업 시**: `scripts/check_spec_coverage.sh` 실행하여 3계층 추적성 확인. 누락 시 Implementer에 tests/spec/ 추가 요청
3. **커밋**: Implementer가 코드를 변경한 경우, Conventional Commits 형식으로 커밋
4. **알림**: `notify-send "llm.rs" "{워크플로우}: {요약}"`

## Phase 건너뛰기

작업 규모가 작으면 전체 파이프라인을 돌릴 필요가 없다:

| 규모 | 기준 | 생략 가능 Phase |
|------|------|----------------|
| **소규모** | 단일 함수/메서드 변경 | PM, Architect (→ Implementer → Tester) |
| **중규모** | 단일 모듈 내 변경 | PM (→ Architect → Implementer → Tester) |
| **대규모** | 다중 모듈/트레이트 변경 | 전체 파이프라인 실행 |

사용자가 명시적으로 "빠르게" 또는 "간단히"를 요청하면 소규모로 처리.

## 테스트 시나리오

### 정상 흐름: feature
> "KV cache에 LRU eviction policy를 추가해줘"
1. PM: 작업 분해 (EvictionPolicy 구현, CLI 옵션, 테스트)
2. Architect: LruPolicy 설계, eviction/mod.rs에 추가, 기존 트레이트 호환성
3. Implementer: 구현 + tests + sanity check 통과
4. Tester: cargo test 전체 통과, 회귀 없음
5. 커밋 `feat(eviction): add LRU eviction policy` + 알림

### 에러 흐름: fix with retry
> "H2O eviction에서 인덱스 패닉이 발생해"
1. Implementer: 원인 분석 → off-by-one 수정 + sanity check
2. Tester: 관련 테스트 실패 발견 → Implementer에 전달
3. Implementer: 엣지 케이스 추가 수정 + sanity check 통과
4. Tester: 전체 통과
5. 커밋 `fix(eviction): fix H2O index panic on boundary` + 알림

### 소규모 fast path
> "sampling.rs의 temperature 기본값을 0.8로 바꿔줘"
1. (PM, Architect 생략)
2. Implementer: 기본값 변경 + 테스트 업데이트 + sanity check
3. Tester: 회귀 확인
4. 커밋 `fix(sampling): change default temperature to 0.8` + 알림
