# Review — `<Plan/Design/Decision 제목>`

**스코프**: Plan 사전 리뷰 | Design 리뷰 | Decision 재검토 | 빠른 sanity-check
**대상**: `<파일/모듈/sprint/spec ID>`

---

## 1. TL;DR

- 추천: `<옵션 / 방향>` (Status: Proposed | Accepted | Conditional | Superseded)
- 가장 큰 리스크: `<요약>` (RPN <NN>)
- 게이트: `<통과 조건 수치 또는 조건>`

## 2. Context — 현재 문제 / 기대 이득

### 2.1 현재 문제

`<무엇이 깨졌나 / 무엇이 부족한가. spec ID / invariant ID 인용. 가능하면 정량(수치/baseline/violation 개수), 정성적 표현이 더 적절하면 정성으로>`

### 2.2 기대 이득

`<무엇이 풀리고 무엇이 가능해지나. 가능하면 정량 게이트(baseline 감소, TBT 변화, 후속 unblock 수), 부적절하면 정성>`

### 2.3 왜 지금

`<시점 정당화 — 선행 작업 종료 / 다른 트랙 의존성 / 컨텍스트 휘발 우려 / deadline. 1~2줄>`

### 2.4 왜 이 방식 (한 줄)

`<추천안의 핵심 한 줄 직관. 자세한 비교는 §4 Alternatives에 위임>`

## 3. Proposed Change (조건부 — 구조/trait 변경 시)

> **발동 조건**: Design 스코프 + 구조·trait 변경 → 강제. Plan 스코프 + 구조 변경 → 권장.
> 변경 없으면 §3 전체 생략.

### 3a. UML (구조 변경, ASCII)

`<ASCII 다이어그램. 변경 유형별:>`
`<  - trait 계층/impl: 박스 + tree (▲ │ ├── └──)>`
`<  - dispatch/sequence: 화살표 (──▶ ◀───) + lane>`
`<  - 모듈 의존: tree (├── └──)>`
`<  - state machine: 상태 박스 + 화살표>`
`<가능하면 Before / After 두 블록>`

```
Backend (trait)
├─ matmul()
├─ <existing method>()
├─ <new method>()  [NEW] default: <self / None / no-op>
└─ ...
     △
     │ impl
     ├── CpuBackend       (override <method> = <impl>)
     ├── OpenCLBackend    (override <method>)
     │       └─owns Arc─▶ CpuBackend
     ├── CudaBackend      (override <method>)
     └── QnnBackend       (all defaults)
```

### 3b. Trait Diff (주요 trait 변경)

**Before / After signature**:

| Method | Before | After | Default impl |
|---|---|---|---|
| `<method>` | `<sig 또는 —>` | `<sig>` | `<self / None / no-op / unimplemented!()>` |

**Default impl 정책**: `<self 반환 vs None vs unimplemented!() 비교 + 선택 근거>`

**구현체별 override 매트릭스**:

| Backend | `<method1>` | `<method2>` | `<method3>` |
|---|---|---|---|
| CpuBackend | `<override / default>` | … | … |
| OpenCLBackend | `<override / default>` | … | … |
| CudaBackend | `<override / default>` | … | … |
| Qnn* | `<override / default>` | … | … |

**ISP 누적 + 호출지 패턴**:
- ISP 누적: trait method N→N+<K>. 신규 backend 추가 비용 `<0줄 / N줄>`.
- 호출지 패턴 변경: `<before pattern>` → `<after pattern>`. 영향 사이트 `<N>`건.

## 4. Alternatives Considered

| 후보 | 핵심 메커니즘 | Pros | Cons | 기각 사유 |
|------|---------------|------|------|-----------|
| (a) `<이름>` | `<요약>` | `<>` | `<>` | (선택) |
| (b) `<이름>` | `<요약>` | `<>` | `<>` | `<>` |
| (c) status quo (변경 없음) | — | `<>` | `<>` | `<>` |

## 5. Risks (목록 형식 — 표 금지)

정성 서술 (RPN < 100):
- [경미] `<설명>`
- [경미] `<설명>`

정량 (RPN ≥ 100):
- R-1 [RPN <NN>] `<설명>`
    Severity <N> (`<이유>`) × Occurrence <N> (`<이유>`) × Detection <N> (`<이유>`)
    게이트: `<통과 조건>` → 초과 시 `<완화책>`
- R-2 [블로커 RPN ≥ 200] `<설명>` — **작업 차단**
    S <N> × O <N> × D <N>
    게이트: `<premortem 필수 + 통과 조건>`

척도:
| 축 | 1점 | 5점 | 10점 |
|---|---|---|---|
| Severity | log 한 줄 | TBT +10% | PACT2026 차단 / crash |
| Occurrence | 검증 환경만 | 1B 모델 가끔 | 모든 backend 항상 |
| Detection | host test | S25 device test | logcat·production만 |

## 6. Tradeoffs (ATAM)

- **Sensitivity points** (작은 변경이 큰 영향):
  - `<예: DK=128 per-thread 32 float4 임계>`
- **Tradeoff points** (품질 희생):
  - `<예: TBT 개선 vs 메모리 spike>`

## 7. Devil's Advocate Pass

- 내 결론이 틀렸다면 가장 가능성 높은 이유: `<>`.
- 이 결론이 맞으려면 `<>` 가정이 참이어야 한다. 이 가정의 약점: `<>`.

## 8. Recommendation

**선택**: `<단일안 + 핵심 한 줄>`
**Status**: Conditional — `<게이트>` 통과 시 Accepted
**근거**: `<3줄 이내>`
**관련 spec/arch**: `<INV-XXX / §13.5 Resolution Log V-?? / arch/<file>.md>`

## 9. Open Questions

- [Architect 답변 필요, `<phase 진입 전>`] `<질문>`
- [구현 중 해소 가능] `<질문>`
- [사용자 결정] `<질문>`

---

## Appendix (응답 끝, 짧게)

### A. Reviewed Items

- 읽은 파일·라인: `<path>:<L>-<L>` × 5건 정도 (결정에 영향 준 것만)
- 인용한 실측: `<수치>` × 2~3건
- 검토했지만 기각: `<후보명 — 1줄 기각 사유>` × 2건
- 확인한 invariant·spec: `<INV-XXX / §13.8-X>` — 영향 / 무영향

### B. Coverage (Design 스코프만 강제, Plan은 한 줄 OK)

- **Design**: L1~L5 layering 일치 — `<O/X + 사유>`
- **Functionality**: prefill / decode 영향 — `<범위>`
- **Tests**: host cargo test + S25 + Jetson — `<커버리지>`
- **Backend Coverage**: NEON / Adreno / CUDA — `<회귀 가능 영역>`

---

## 사용 팁

- **스코프별 본문 부분집합**:
  - Plan 사전 리뷰: 1·2·4·5·7·8·9 (+ §3 옵션)
  - Design 리뷰: 전 항목 (1·2·3·4·5·6·7·8·9) + Appendix A·B
  - Decision 재검토: 1·2·4·7·8
  - 빠른 sanity-check: 1·2·5·8
- **자기점검**: §2 Context / §4 Alternatives(≥2 + status quo) / §7 DA Pass 3개는 강제. §5 Risks RPN ≥ 100은 게이트 강제. 구조 변경 있으면 §3a UML (ASCII), trait 변경 있으면 §3b Trait Diff.
- **5 Whys**: 표면 추론 의심 시 / 사용자가 "근본 원인" 요청 시에만 추가.
- **UML은 ASCII** (리뷰 응답 인라인). docs/arch/ 정식 문서는 Mermaid.
