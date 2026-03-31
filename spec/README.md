# llm_rs2 Specification

이 디렉토리는 llm_rs2 시스템의 정식 스펙(specification)을 포함한다.
이 스펙만으로 시스템을 독립 재구현하고 검증할 수 있는 수준을 목표로 한다.

## 문서 컨벤션

### RFC 2119 키워드

이 스펙에서 사용하는 키워드는 [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119)를 따른다:

- **MUST** / **MUST NOT**: 절대적 요구사항. 위반하면 비적합(non-conformant) 구현이다.
- **SHOULD** / **SHOULD NOT**: 정당한 사유가 있으면 위반 가능하나, 사유를 이해하고 결정해야 한다.
- **MAY**: 완전히 선택적이다.

### 요구사항 ID 체계

모든 요구사항에 고유 ID를 부여한다: `[PREFIX-NNN]`

| 접두사 | 범위 | 대상 |
|--------|------|------|
| `SYS` | 001–099 | 시스템 전체 (00, 01) |
| `PROTO` | 010–099 | 프로토콜 (10, 11, 12) |
| `MGR` | 010–099 | Manager 개요/상태 (20, 21) |
| `MGR-ALG` | 010–099 | Manager 알고리즘 (22) |
| `MGR-DAT` | 010–099 | Manager 데이터 (23) |
| `ENG` | 010–099 | Engine 개요/상태 (30, 31) |
| `ENG-ALG` | 010–099 | Engine 알고리즘 (32) |
| `ENG-DAT` | 010–099 | Engine 데이터 (33) |
| `XC` | 010–099 | 횡단관심사 (40) |
| `INV` | 001–999 | 불변식 (41) — 전체 수집 |
| `TOOL` | 010–099 | 테스트 도구 (50) |
| `CON` | 001–999 | 적합성 기준 (90) |
| `TRC` | 001–999 | 추적성 (91) |

### 스펙 레벨

| 대상 | 수준 |
|------|------|
| **프로토콜** (10–12) | 완전 구체적: 필드명, 타입, 범위, JSON 예시 |
| **알고리즘** | 의사코드 + 불변식 + trace. 변수명/타입 자유 |
| **상태 머신** | 전이 테이블 + 조건. 빠짐없는 열거 |
| **데이터** | 의미와 관계. struct 레이아웃 자유 |

### 파일 내부 구조

모든 스펙 파일은 다음 구조를 따른다:

```markdown
# [제목]

> **TL;DR**: 3-5줄 요약

## 1. Purpose and Scope
## 2. Definitions
## 3. Specification          ← RFC 2119 키워드, 요구사항 ID
## 4. Alternative Behavior   ← 선택적
## 5. Constraints            ← 선택적
## 6. Examples
## 7. Rationale (non-normative)
```

## 디렉토리 구조

```
spec/
├── README.md                    ← 이 파일
├── 00-overview.md               # 시스템 컨텍스트, 목표, 용어집
├── 01-architecture.md           # 2-컴포넌트 분해, IPC 토폴로지
│
├── 10-protocol.md               # IPC 와이어 포맷, 프레이밍
├── 11-protocol-messages.md      # 메시지별 필드 정의
├── 12-protocol-sequences.md     # 정규 상호작용 시퀀스
│
├── 20-manager.md                # Manager 개요 (HLR)
├── 21-manager-state.md          # Manager 상태 머신
├── 22-manager-algorithms.md     # PI 제어, Action Selection, Relief
├── 23-manager-data.md           # 설정 스키마, 센서 인터페이스
│
├── 30-engine.md                 # Engine 개요 (HLR)
├── 31-engine-state.md           # Engine 상태 머신
├── 32-engine-algorithms.md      # KV 캐시 연산, eviction, 양자화
├── 33-engine-data.md            # 텐서 레이아웃, 캐시 포맷
│
├── 40-cross-cutting.md          # 에러 처리, 로깅, 타이밍 규약
├── 41-invariants.md             # 시스템 전체 불변식 수집
│
├── 50-test-tools.md             # mock_engine, mock_manager 테스트 도구
│
├── 90-conformance.md            # 적합성 기준 (MUST → 테스트 assertion)
└── 91-traceability.md           # HLR → LLR → spec → test 매핑
```

### 그룹핑 규칙

- `0x`: 시스템 수준
- `1x`: 프로토콜 (Manager ↔ Engine 계약)
- `2x`: Manager 컴포넌트
- `3x`: Engine 컴포넌트
- `4x`: 횡단관심사
- `5x`: 테스트 도구
- `9x`: 검증

## 읽기 순서

1. `00-overview.md` → 용어와 시스템 맥락 이해
2. `01-architecture.md` → 컴포넌트 분해와 책임 경계
3. `10–12` → 프로토콜 (양쪽의 계약)
4. `20–23` 또는 `30–33` → 구현할 컴포넌트의 스펙
5. `40–41` → 횡단관심사와 불변식
6. `90–91` → 검증 기준

Manager와 Engine은 독립적으로 읽을 수 있다. 프로토콜을 먼저 읽으면 된다.

## 이 스펙과 다른 문서의 관계

| 문서 | 역할 | 관계 |
|------|------|------|
| `docs/` | 기존 구현 가이드 (대체 대상) | 이 스펙이 docs/를 대체한다 |
| `ARCHITECTURE.md` | 고수준 아키텍처 개요 | 이 스펙이 상세 레벨을 담당 |
| `pact2026/plan/` | 논문 설계/실험 계획 | 이 스펙이 구현을 명세 |
