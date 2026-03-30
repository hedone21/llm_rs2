---
name: spec-manage
description: Spec/Arch/Test 3계층 문서를 관리한다. 새 요구사항(INV, PREFIX-NNN) 추가, 기존 요구사항 수정/폐기, arch/ 구현 매핑 갱신, tests/spec/ 테스트 요구 생성. Architect 에이전트가 spec 변경 작업 시 반드시 이 스킬을 사용. 'spec 추가', 'INV 추가', '불변식', '요구사항', 'arch 갱신', '3계층 동기화' 등의 요청 시 트리거.
allowed-tools: Read, Glob, Grep, Edit
---

# Spec Management

llm.rs의 **Spec(불변) → Arch(가변) → Test(검증)** 3계층 문서를 관리한다.

이 스킬이 Spec 관리의 유일한 참조점이다.

## 3계층 구조

```
spec/       ← WHAT: 불변식, 요구사항, FSM 전이 (구현 독립적)
arch/       ← HOW: 코드 경로, config 키, struct 매핑 (코드 종속적)
tests/spec/ ← VERIFY: INV-NNN별 자동 검증 테스트 (Rust 코드)
```

단일 ID(`INV-031`)가 3계층을 관통하여 정의→구현→검증을 추적한다.

## 1. Spec 변경 유형 판정

요청을 받으면 먼저 변경 유형을 분류한다:

| 분류 | 정의 | 예시 | 필요 작업 |
|------|------|------|----------|
| **Normative 변경** | `[PREFIX-NNN]` 의미 변경/추가/삭제 | 새 INV, 요구사항 수정 | spec/ + arch/ + tests/spec/ 동시 갱신 |
| **Non-normative 변경** | Rationale, 예시, 오타 수정 | 섹션 7 보강 | spec/만 수정. 다른 계층 불필요 |
| **Arch-only 변경** | 구현 상세만 변경 | 파일 경로 변경, config 키 추가 | arch/만 수정 |

## 2. 새 요구사항 ID 추가 절차

### 2.1 ID 접두사 선택

| 접두사 | 파일 | 대상 |
|--------|------|------|
| `SYS` | 00-overview, 01-architecture | 시스템 목표, 크레이트 구조 |
| `PROTO` | 10-protocol | 와이어 포맷, 트랜스포트 |
| `MSG` | 11-protocol-messages | 메시지별 필드 |
| `SEQ` | 12-protocol-sequences | 상호작용 시퀀스 |
| `MGR` | 20-manager, 21-manager-state | Manager 요구사항 |
| `MGR-ALG` | 22-manager-algorithms | PI, Supervisory 알고리즘 |
| `MGR-DAT` | 23-manager-data | 설정 스키마, 센서 |
| `ENG` | 30-engine | Engine 요구사항 |
| `ENG-ST` | 31-engine-state | Engine 상태 머신 |
| `ENG-ALG` | 32-engine-algorithms | KV 캐시, eviction, 양자화 |
| `ENG-DAT` | 33-engine-data | 텐서 레이아웃, 캐시 포맷 |
| `CROSS` | 40-cross-cutting | 에러, 로깅, 타이밍 |
| `INV` | 41-invariants (수집) | 불변식 카탈로그 |

### 2.2 ID 할당 순서

```bash
# 1. 해당 접두사의 마지막 번호 확인
grep -oE '\[ENG-ALG-[0-9]+\]' spec/32-engine-algorithms.md | sort -t'-' -k3 -n | tail -5

# 2. +1로 새 번호 할당

# 3. 충돌 검사
grep -r 'ENG-ALG-096' spec/
# 결과가 없으면 안전

# 4. spec/ 파일에 ID 추가
# 5. INV인 경우 41-invariants.md 카탈로그에 행 추가
```

### 2.3 ID 불변 규칙

- ID는 절대 변경하지 않는다. 의미가 바뀌면 DEPRECATED + 새 ID 할당
- ID 번호는 재사용하지 않는다
- spec/에 코드 경로를 넣지 않는다 (arch/의 역할)
- arch/에 불변식을 넣지 않는다 (spec/의 역할)

## 3. Spec 파일 구조

모든 spec/ 파일의 필수 섹션:

```markdown
# [제목]

> **TL;DR**: 3-5줄 요약

## 1. Purpose and Scope
## 2. Definitions
## 3. Specification       ← normative. RFC 2119 키워드 + 요구사항 ID.
## 4. Alternative Behavior ← (선택적)
## 5. Constraints          ← (선택적) 적합성 기준
## 6. Examples             ← non-normative
## 7. Rationale (non-normative)
```

요구사항 형식: `[PREFIX-NNN] 설명 (MUST/SHOULD/MAY)`

## 4. INV 추가 시 필수 동기화

INV를 추가하면 반드시 4가지를 동시 처리한다:

1. **해당 spec/ 파일**에 `[INV-NNN]` 정의 추가
2. **`spec/41-invariants.md`** 카탈로그 테이블에 행 추가
   - 카테고리: Safety / Correctness / Performance / Compatibility
   - 검증 방법: static / runtime / test
3. **arch/ 파일** (존재 시): 구현 위치 기술
4. **오케스트레이터에 보고**: tests/spec/ 테스트 작성이 필요함을 Implementer 요청으로 전달

## 5. Arch 문서 규칙

### 5.1 기본 원칙

- arch/ 파일명 = spec/ 파일명 (예: `arch/22-manager-algorithms.md`)
- spec/ 없는 arch/ 파일은 존재 불가
- arch/에 독자적 요구사항 ID를 만들지 않는다 (ID 원천은 항상 spec/)
- **줄번호 매핑 금지** — 함수/모듈/컴포넌트 단위로만 매핑

### 5.2 arch/ 파일 구조: 컴포넌트 중심

arch/ 파일은 ID-코드 1:1 테이블이 아니라, **컴포넌트(모듈/구조체/트레이트) 단위**로 구성한다. 각 컴포넌트 섹션에 다음을 포함한다:

| 항목 | 내용 | 필수 |
|------|------|------|
| **설계 결정** | spec 요구사항을 구현하기 위한 전략과 근거 | O |
| **인터페이스** | pub 함수 시그니처, 전제조건(pre), 후조건(post) | O |
| **처리 흐름** | Mermaid 다이어그램 (flowchart, sequence, state) | 복잡한 로직일 때 |
| **예외 처리** | 에러 경로, fallback 전략, 에러 타입 | 에러가 있을 때 |
| **코드-스펙 차이** | 스펙과 구현이 다른 부분 + 그 이유 | 차이가 있을 때 |
| **Config/CLI** | config 키, CLI 플래그, 기본값 | 있을 때 |

### 5.3 arch/ 파일 템플릿

```markdown
# [제목] -- Architecture

> spec/XX-YYYY.md의 구현 설계.

## [컴포넌트명] (관련 spec ID: PREFIX-NNN~NNN)

### 설계 결정
(spec이 요구하는 WHAT을 HOW로 실현하는 전략. 왜 이 방식을 선택했는지.)

### 인터페이스
\```rust
impl Component {
    pub fn method(&self, input: Type) -> Result<Output>
    // 전제: ...
    // 후조건: ... (INV-NNN)
}
\```

### 처리 흐름
\```mermaid
flowchart TD
    A[입력] --> B{판정}
    B -->|조건1| C[처리1]
    B -->|조건2| D[처리2]
\```

### 예외 처리
- 입력 X가 범위 밖이면 → ...
- 의존 서비스 실패 시 → ...

### 코드-스펙 차이
| 스펙 | 코드 | 이유 |
|------|------|------|

### Config / CLI
| 키/플래그 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|
```

## 6. INV 폐기 절차

```markdown
<!-- 41-invariants.md에서 -->
| INV-030 | ~~22-manager-algorithms~~ | ~~설명~~ | ~~카테고리~~ | ~~검증~~ | **DEPRECATED v2.1** |
```

- 행을 삭제하지 않는다. 취소선 + DEPRECATED 표기
- 번호를 재사용하지 않는다

## 7. ID 할당 범위 상세

### INV 범위별 다음 가용 번호

| INV 범위 | 원본 파일 | 다음 가용 번호 |
|----------|----------|-------------|
| INV-001~006 | 00-overview | INV-007 |
| INV-010~018 | 01-architecture | INV-019 |
| INV-020~028 | 10/11-protocol | INV-029 |
| INV-030~051 | 22-manager-algorithms | INV-052 |
| INV-060~065 | 30-engine | INV-066 |
| INV-070~076 | 31-engine-state | INV-077 |
| INV-080~085 | 41-invariants (cross-cutting) | INV-086 |

### Constraint (CON-*) 범위

| 범위 | 원본 파일 |
|------|----------|
| CON-010~012 | 10-protocol |
| CON-020~022 | 11-protocol-messages |
| CON-030~033 | 12-protocol-sequences |
| CON-040~042 | 40-cross-cutting |

### Manager/Engine Constraint 범위

| 접두사 | 범위 | 원본 파일 |
|--------|------|----------|
| MGR-C01~C02 | 20-manager | Manager 개요 제약 |
| MGR-C03~C05 | 21-manager-state | Manager 상태 제약 |
| MGR-C06~C10 | 22-manager-algorithms | Manager 알고리즘 제약 |
| MGR-DAT-C01~C02 | 23-manager-data | Manager 데이터 제약 |
| ENG-ALG-C01~C08 | 32-engine-algorithms | Engine 알고리즘 제약 |
| ENG-DAT-C01~C07 | 33-engine-data | Engine 데이터 제약 |

## 8. COVERAGE.md 형식

`spec/COVERAGE.md`의 상태 기호:

| 기호 | 의미 |
|------|------|
| ✅ | 테스트 존재 + 통과 |
| ❌ | 테스트 존재 + 실패 |
| ⬜ | 테스트 미작성 |
| 🔶 | 자동 테스트 불가 (static 전용). 사유 명시. |

## 9. 일관성 검사

Spec 변경 완료 후 실행:

```bash
# ID 충돌 검사
grep -roE '\[[A-Z]+-[A-Z]*-?[0-9]+[a-z]?\]' spec/*.md | sed 's/.*://' | sort | uniq -d

# 3계층 + INV 커버리지 + 비-INV 추적성 통합 검사
scripts/check_spec_coverage.sh
```

## 10. 산출물 체크리스트

Spec 작업 완료 후 확인:

- [ ] 새 ID에 충돌 없음 (`grep` 확인)
- [ ] INV 추가 시 41-invariants.md 동기화됨
- [ ] spec/에 코드 경로가 없음 (WHAT만)
- [ ] arch/에 불변식이 없음 (HOW만)
- [ ] Normative 변경 시 arch/ + tests/spec/ 요구 사항 명시됨
- [ ] ID 형식: `[PREFIX-NNN]` + RFC 2119 키워드 (MUST/SHOULD/MAY)

## FAQ

**Q: spec에 있는데 코드에 없는 것은?** — 구현 의무. spec/이 source of truth. 즉시 구현이 어려우면 arch/에 `TODO: 미구현` 표기.

**Q: arch/ 없이 테스트 작성 가능한가?** — spec/만으로 가능. spec/은 WHAT을 정의하므로 입출력 조건으로 테스트 작성 가능. arch/는 편의 참조.

**Q: INV-ID를 변경하면?** — **절대 변경 금지.** 의미가 바뀌면 DEPRECATED + 새 ID 할당. 번호 재사용 금지.
