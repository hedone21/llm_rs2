---
name: spec-manage
description: Spec/Arch/Test 3계층 문서를 관리한다. 새 요구사항(INV, PREFIX-NNN) 추가, 기존 요구사항 수정/폐기, arch/ 구현 매핑 갱신, tests/spec/ 테스트 요구 생성. Architect 에이전트가 spec 변경 작업 시 반드시 이 스킬을 사용. 'spec 추가', 'INV 추가', '불변식', '요구사항', 'arch 갱신', '3계층 동기화' 등의 요청 시 트리거.
allowed-tools: Read, Glob, Grep, Edit
---

# Spec Management

llm.rs의 **Spec(불변) → Arch(가변) → Test(검증)** 3계층 문서를 관리한다.

상세 규칙은 `spec/CONTRIBUTING.md`에 정의되어 있다. 이 스킬은 핵심 절차를 요약하고 실행 가이드를 제공한다.

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

### 5.1 spec/과 1:1 대응

- arch/ 파일명 = spec/ 파일명 (예: `arch/22-manager-algorithms.md`)
- spec/ 없는 arch/ 파일은 존재 불가
- arch/에 독자적 요구사항 ID를 만들지 않는다 (ID 원천은 항상 spec/)

### 5.2 arch/ 파일 내용

| 항목 | 예시 |
|------|------|
| 코드 파일 경로 | `manager/src/policy/pi_controller.rs` |
| 구조체/함수 매핑 | `INV-031 → PiController::update():87` |
| config 키와 기본값 | `policy.pi.integral_clamp = 2.0` |
| CLI 플래그 | `--resilience-config <path>` |

### 5.3 arch/ 파일 템플릿

```markdown
# [제목] -- Architecture

> spec/XX-YYYY.md의 구현 상세.

## 코드 매핑

| spec/ ID | 코드 위치 | 구현 방법 | 비고 |
|----------|----------|----------|------|

## Config

| config 키 | 타입 | 기본값 | spec/ 근거 |
|-----------|------|--------|-----------|

## CLI

| 플래그 | 설명 | spec/ 근거 |
|--------|------|-----------|
```

## 6. INV 폐기 절차

```markdown
<!-- 41-invariants.md에서 -->
| INV-030 | ~~22-manager-algorithms~~ | ~~설명~~ | ~~카테고리~~ | ~~검증~~ | **DEPRECATED v2.1** |
```

- 행을 삭제하지 않는다. 취소선 + DEPRECATED 표기
- 번호를 재사용하지 않는다

## 7. 일관성 검사

Spec 변경 완료 후 실행:

```bash
# ID 충돌 검사
grep -roE '\[[A-Z]+-[A-Z]*-?[0-9]+[a-z]?\]' spec/*.md | sed 's/.*://' | sort | uniq -d

# INV 테스트 커버리지 (tests/spec/ 존재 시)
# spec/CONTRIBUTING.md 섹션 9.1 참조
```

## 8. 산출물 체크리스트

Spec 작업 완료 후 확인:

- [ ] 새 ID에 충돌 없음 (`grep` 확인)
- [ ] INV 추가 시 41-invariants.md 동기화됨
- [ ] spec/에 코드 경로가 없음 (WHAT만)
- [ ] arch/에 불변식이 없음 (HOW만)
- [ ] Normative 변경 시 arch/ + tests/spec/ 요구 사항 명시됨
- [ ] ID 형식: `[PREFIX-NNN]` + RFC 2119 키워드 (MUST/SHOULD/MAY)
