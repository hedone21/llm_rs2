---
name: review
description: 구현 전 Plan/Action/Decision 사전 리뷰와 Design/Architecture 리뷰를 일관된 골격으로 수행한다. 본문은 결론 직결 섹션(Context → Proposed Change → Alternatives → Risks → Recommendation)을 앞에 두고, 메타정보(Reviewed Items / Coverage)는 Appendix로 후순위 배치한다. '리뷰', 'review', '검토', '재검토', '리스크 확인', '설계 관점 문제 확인', '옵션 비교', '계획 검토', '의사결정 재검토' 같은 요청 시 반드시 이 스킬을 사용. 사용자가 "리스크는 없는지 한번 더 확인", "옵션 N 비교", "이 방향 진행 전 검토", "왜 이게 더 나은지", "단점은?" 등을 묻는 모든 경우에 트리거. 코드 작성 후 SOLID 분석은 design-review를 쓰고 본 스킬은 구현 전 사전 리뷰·아키텍처 결정에 집중한다. 단순 사실 확인이나 "방금 한 작업 보여줘" 같은 verification에는 트리거하지 않는다.
allowed-tools: Read, Glob, Grep, Bash, WebSearch, WebFetch
---

# Review Skill — Plan/Action 사전 리뷰 + Design 리뷰

구현 전 작업 계획·설계 결정·트레이드오프에 대해 **결론 직결 섹션을 앞에 두는 골격**으로 응답한다. 외부 베스트 프랙티스 (ADR / Rust RFC / Premortem / Google eng-practices / ATAM / FMEA / Anthropic Extended Thinking / Devil's Advocate + 5 Whys) + 본 프로젝트 case study 분석에서 도출된 7원칙을 운영 형태로 압축했다.

조사 원본: `papers/review_research_report.md` (또는 jobs cache).

## 왜 필요한가 — 3대 불만 + 4대 보강

본 스킬이 직접 해소하는 사용자 불만:

1. **구체적 설명 부족** — 에이전트가 검토·고려한 내용이 사용자에게 전달 안 됨 (속으로는 더 봤을 텐데 결론만 짧게 옴)
2. **리스크 표면화 부족** — 낙관 편향. landmine, 안 가본 길, 반증 가능성, 실패 시나리오 누락
3. **구조/일관성 부족** — 리뷰마다 다른 포맷. 같은 항목이 어느 땐 있고 어느 땐 없음

2026-05-23 골격 개정에서 보강된 4대 요청:

4. **결론 직결 섹션이 앞에** — 메타정보(읽은 파일·테스트 커버리지)는 결론에 영향이 작은데 앞에 오면 응답 가독성을 깬다. Appendix로 분리.
5. **서두에 "현재 문제 + 기대 이득" 정량 설명** — TL;DR은 결론, Context는 problem statement. 둘은 다른 역할.
6. **구조 변경 시 UML 강제** (Design 스코프) — 모듈/trait 관계는 텍스트로 설명해도 잘 안 보임. Mermaid 1개가 텍스트 20줄을 절약.
7. **주요 trait 변경 시 별도 리뷰** — signature/default impl/ISP 누적은 응답에 명시 안 하면 재요청을 부른다.

각 섹션이 어느 불만에 답하는지 본문에 명시.

## 적용 범위

- **메인 세션 전용**. architect/researcher/PM/implementer/tester 서브에이전트는 자체 응답 포맷 유지.
- 본 스킬을 호출한 메인 세션 응답이 골격을 따른다.
- 서브에이전트에게 위임할 때는 위임 prompt에 "review 스킬 골격으로 응답" 요구 가능 (선택).

## 트리거

### 반드시 트리거

- "리뷰해줘 / review / 검토해줘 / 재검토 / 한번 더 확인"
- "옵션 N에 대해 목적/작업/변경점/리스크"
- "리스크나 설계 관점 문제 없는지"
- "왜 (a) 대신 (b)인가" / "단점은?" / "트레이드오프"
- "이 방향 진행 전" / "구현 전" / "사전 점검"
- "의사결정 재검토" / "이 결정 다시 봐"

### 트리거하지 않음

- 단순 사실 확인 ("이 파일 어디 있어?", "이 함수 뭐 하는 거야?")
- 단순 verification ("방금 한 작업 보여줘", "테스트 통과했어?")
- 구현 완료 후 SOLID 코드 분석 → `design-review` 스킬
- handoff 작성 → `handoff-doc` 스킬

## 응답 골격 — 본문 9섹션 + Appendix

`assets/review_template.md`에 채워야 할 빈 템플릿. SKILL 본문에서는 각 섹션의 작성 가이드만 다룬다.

```
본문 (결론 직결 순서)
 1. TL;DR (3줄)
 2. Context — 현재 문제 / 기대 이득 / 왜 지금 / 왜 이 방식  ★강제
 3. Proposed Change                                     ★조건부
       3a. UML (구조 변경 시, ASCII)
       3b. Trait Diff (주요 trait 변경 시)
 4. Alternatives Considered (≥ 2 + status quo)
 5. Risks (정성 서술 + RPN ≥ 100 점수, 목록 형식)
 6. Tradeoffs (Sensitivity / Tradeoff points)
 7. Devil's Advocate Pass (1단락)
 8. Recommendation (단일안 + Status)
 9. Open Questions (ownership 명시)

Appendix (응답 끝, 짧게)
 A. Reviewed Items (파일·라인 5건 + spec ID 3건 + 기각 후보 2건 정도)
 B. Coverage — Design / Functionality / Tests / Backend (3~4줄)
```

**§2 Context는 모든 스코프에서 강제**. 결론이 어디서 왔는지 사용자가 첫 화면에서 이해해야 한다.

> **2026-05-23 개정**: Premortem 섹션 제거. 본 프로젝트 실측에서 Premortem이 Risks의 narrative 확장에 머무르고 새 시나리오 발굴 0건 + DA Pass와 결론 부정 각도가 가까워 중복으로 작동. failure narrative가 유용한 경우 Risks 항목의 sub-bullet으로 흡수 가능 (강제 X). Gary Klein 원본의 가치(심리적 안전성, 시간 시퀀스 발굴)는 인정하나 본 프로젝트 스타일에서는 비효율.

### 스코프별 본문 섹션 선택

모든 리뷰에 본문 9섹션 전부를 강제하면 응답이 비대해진다. 스코프별로 권장 부분집합:

| 스코프 | 본문 섹션 | §3 (UML/Trait Diff) | Appendix |
|---|---|---|---|
| **Plan/Action 사전 리뷰** | 1·2·4·5·7·8·9 | **옵션** (구조 변경 있으면 권장, 강제 X — Architect 단계에서 만들 수 있음) | A 짧게 |
| **Design/Architecture 리뷰** | 전 항목 (1~9) | **강제** (구조·trait 변경 발견 시 반드시 작성) | A + B |
| **Decision 재검토** (R4 식) | 1·2·4·7·8 | 옵션 | A 짧게 |
| **빠른 sanity-check** | 1·2·5·8 | 생략 | 생략 |

스코프가 모호하면 사용자에게 "Plan 사전 리뷰로 진행하면 본문 7섹션 / Design 리뷰로 진행하면 본문 9섹션 + §3 강제. 어느 쪽?"으로 묻는다.

## 섹션별 작성 가이드

### 1. TL;DR (3줄 이내)

핵심 결론 3줄. 추천안 + 가장 큰 리스크 + 게이트 한 줄. **Context의 요약이 아니라 결론**.

```
- 추천: 옵션 (a) Backend trait + default impl 4 method (Status: Conditional)
- 가장 큰 리스크: cpu_kernels 함수 포인터 hot path 회귀 (RPN 144)
- 게이트: S25 Qwen 2.5 1.5B Q4_0 TBT Δ ≤ +3% 통과 시 Accepted
```

### 2. Context — 현재 문제 / 기대 이득 (강제)

**3대 불만 1번 + 4대 보강 5번** 직접 해소. TL;DR이 "결론"이라면 Context는 "왜 이 결정이 필요한가". 자세하게, 명확하게, 쉽게 작성한다.

**표현 방식**: 가능하면 정량(수치/baseline/violation 개수)이 검증성과 비교 가능성 면에서 유리하지만, 정성적 표현이 더 적절한 경우(예: "기존 코드 스타일과 충돌", "사용자 mental model이 깨짐", "downstream 트랙의 옵션 폭이 좁아짐")는 정성으로 작성해도 된다. **억지 수치보다 명확한 정성이 낫다**.

4 sub-block 모두 포함:

#### 2.1 현재 문제

- **무엇이 깨졌나** 또는 **무엇이 부족한가**를 spec ID / invariant ID / 사용 시나리오로 표시
- 가능하면 정량 (baseline 건수, violation 개수, 회귀 수치, 미커버 시나리오 N개)
- 정량이 부적절하면 정성 (예: "현재 호출지가 backend impl을 직접 알아야 하므로 신규 backend 추가 시 5개 모듈 손대야 함")
- 1~3줄

예:
```
INV-LAYER-003 위반 150건 (전체 baseline 206건의 73%). 그중 73건이 L3-inference → L1
backend 직접 의존 패턴(V-13/V-17/V-20 클러스터). spec/41-invariants.md INV-LAYER-003
본문은 "L3 코드는 backend 추상화 trait만 import 가능"이라 명시 — 현재는 모든 GPU
fallback 호출지에서 `if let Some(opencl) = backend.as_any().downcast_ref::<OpenCLBackend>()`
같은 downcast 패턴이 강제되어 spec 본문이 사실상 의도대로 동작하지 않는다.
```

#### 2.2 기대 이득

- 단순 "깨끗해짐"이 아니라 **무엇이 풀리고 무엇이 가능해지나**
- 가능하면 정량 게이트 (baseline 감소, TBT 변화, LOC 변화, 후속 작업 unblock 수)
- 정량이 부적절하면 정성 (예: "신규 backend 추가가 1 파일 변경으로 가능해짐", "RFC 응답 일관성 확보로 다음 라운드의 비교 비용 ↓")
- 1~3줄

예:
```
Backend trait 4 method 추가 (cpu_companion / cpu_kernels / as_opencl_secondary /
yield_after_layer)로 71건 downcast 사이트가 trait method 호출로 치환 가능 →
baseline 206 → 192 (-14). 더 중요한 효과: B-3 sub-sprint (V-13/V-17/V-20 본격 치환)
의 prerequisite가 해소되어 후속 -71 unblock. 누적 B sprint -90 → -161 가능 (paper
post-cleanup 트랙 default 메인 작업).
```

#### 2.3 왜 지금

- 시점 정당화 — 선행 작업 종료, 다른 트랙 의존성, deadline, 컨텍스트 휘발 우려
- 1~2줄

예:
```
B-5b Phase 0/1이 직전 worktree에 종료되어 architect 결정 라운드(R1/R2/R4/R8)가
이미 산출된 상태. 컨텍스트가 휘발되면 Phase 0 재실행 비용이 작업 자체보다 클 수 있다.
```

#### 2.4 왜 이 방식 (한 줄)

- 추천안의 핵심 한 줄. **자세한 비교는 §4 Alternatives에 위임**.
- "왜 이 방식인지"의 1차 직관

예:
```
default impl 4 method는 ISP 부채(method 누적)와 vtable dispatch 비용을 받는 대신,
호출지 영향 0 + 신규 backend 추가 시 자동 no-op fallback이라는 가장 큰 장점.
```

### 3. Proposed Change — UML + Trait Diff (조건부)

**조건**: 구조 변경(trait 추가/제거, struct field 변경, 모듈 경계 이동) 또는 주요 trait signature 변경이 있는 경우.

- Design/Architecture 리뷰 스코프 → **강제**
- Plan 사전 리뷰 스코프 → **권장** (Architect 단계에서 작성 가능, 메인 세션이 안 그릴 수도)
- Decision 재검토 → 옵션
- 빠른 sanity-check → 생략

#### 3a. UML (구조 변경 있을 때)

**ASCII 다이어그램 사용** — 리뷰 응답은 Claude Code 콘솔에 직접 렌더되므로 Mermaid는 매번 수동 변환 부담이 있다. ASCII로 작성하면 그대로 읽힌다.

> **분기 규칙**: 리뷰 응답 인라인 = **ASCII**. `docs/`, `arch/` 정식 문서 = **Mermaid** (메모리 `feedback_mermaid_diagrams.md` 준수 — 적용 범위는 "Architect가 docs/ 문서에 추가할 때"). 둘은 컨텍스트가 달라 모순 아님.

변경 유형별 권장 표현:

| 변경 유형 | ASCII 패턴 | 예 |
|---|---|---|
| Trait 계층 / impl 관계 | 박스 또는 tree (`▲ │ ├── └──`) | Backend trait + impl |
| Dispatch / sequence | 화살표 (`──▶` `◀───`) + lane | cpu_fallback path |
| 모듈 의존 그래프 | tree (`├── └──`) | L1~L5 layer |
| State machine | 상태 박스 + 화살표 | LayerSlot Cold→Loading→Loaded |

다이어그램은 **Before / After 두 장**을 가능하면 함께 (변경 가시화). 한 장만 가능하면 After만.

trait 계층 예 (After):

```
Backend (trait)
├─ matmul()
├─ flash_attention()
├─ ... 57 method ...
├─ cpu_companion()       [NEW] default: self           → &dyn Backend
├─ cpu_kernels()         [NEW] default: None           → Option<&CpuKernelSet>
├─ as_opencl_secondary() [NEW] default: None (cfg=opencl)
└─ yield_after_layer()   [NEW] default: no-op
     △
     │ impl
     ├── CpuBackend           (override cpu_kernels = Some(&CPU_KERNEL_SET))
     ├── OpenCLBackend        (override cpu_companion + yield_after_layer + as_opencl_secondary)
     │       └─owns Arc─▶ CpuBackend
     ├── CudaEmbeddedBackend  (override cpu_companion)
     │       └─owns Arc─▶ CpuBackend
     └── QnnOppkgBackend      (all defaults — OpenCLBackend ownership 경유)
```

sequence 예:

```
Before — downcast 패턴 (backend별 분기):
  L3 caller ──▶ backend.as_any().downcast_ref::<CudaBackend>()
            ◀── Some(cuda)
  L3 caller ──▶ cuda.cpu_fallback().matmul(...)
  ※ 5 backend별 5개 분기 필요

After — trait method 통일:
  L3 caller ──▶ backend.cpu_companion()        // &dyn Backend trait method
            ◀── &dyn Backend (CpuBackend)
  L3 caller ──▶ cpu.matmul(...)
  ※ backend 무관 단일 경로
```

너무 큰 다이어그램은 zoom 부분만 (전체 모듈 그래프 대신 영향 sub-tree).

#### 3b. Trait Diff (주요 trait 변경 있을 때)

trait 변경은 호출지·구현체 양쪽에 파급되므로 **별도 sub-block으로 명시 강제**. 4 sub-block:

**1) Before / After signature 표**:

| Method | Before | After | Default impl |
|---|---|---|---|
| `cpu_companion(&self)` | — | `&dyn Backend` | `self` |
| `cpu_kernels(&self)` | — | `Option<&'static CpuKernelSet>` | `None` |
| `as_opencl_secondary(&self)` | — | `Option<&dyn OpenClSecondary>` (cfg opencl) | `None` |
| `yield_after_layer(&self, layer, decode)` | — | `()` | no-op |

**2) Default impl 정책**:

- `self` 반환 vs `None` vs `unimplemented!()` 비교
- 본 case는 모두 no-op/None — 신규 backend가 1줄도 안 적어도 컴파일 통과
- 단 hot path(yield_after_layer)는 default no-op이라도 vtable dispatch 발생 — `#[inline]` 힌트 검토

**3) 구현체별 override 매트릭스**:

| Backend | cpu_companion | cpu_kernels | as_opencl_secondary | yield_after_layer |
|---|---|---|---|---|
| CpuBackend | default(`self`) | `Some(&CPU_KERNEL_SET)` | default(`None`) | default(no-op) |
| OpenCLBackend | **override** (`&self.cpu_companion`) | default(`None`) | **override** (`Some(self)`) | **override** (gpu_yield 로직) |
| CudaBackend | **override** (`&self.cpu_companion`) | default | default | default |
| Qnn* | default | default | default | default |

각 backend의 init 변경 (struct field 추가 등)도 1줄씩.

**4) ISP 누적 + 호출지 패턴 영향**:

- ISP 누적: trait method가 N→N+4. 신규 backend 추가 비용은 default impl 덕에 추가 0줄. 단 trait 가독성·grep 비용은 늘어남.
- 호출지 변경 패턴: `if let Some(opencl) = ...downcast` → `backend.as_opencl_secondary().unwrap_or_else(...)`. 정확 사이트 수 N건 명시.

### 4. Alternatives Considered (≥ 2 + status quo)

3대 불만 2번 (리스크 표면화) + 1번 (구체 설명) 양쪽에 답. Rust RFC "Rationale and alternatives" 강제.

```
| 후보 | 핵심 메커니즘 | Pros | Cons | 기각 사유 |
|------|---------------|------|------|-----------|
| (a) | … | … | … | (선택) |
| (b) | … | … | … | … |
| (c) status quo | 변경 없음 | … | … | … |
```

**status quo (c)는 항상 포함**. "안 하는 것"의 비용도 평가해야 추천의 정당화가 완성된다.

**최소 2개 + status quo = 총 3개**. 단일안만 제시하는 추천은 본 스킬 위반.

본 세션 B-5b R4 case에서 (a)/(b)/(c) 3 후보만 제시했는데, 사용자 우려 제기 후 (b') 변종이 떠올랐다. 첫 응답에서 가능한 변종까지 4 후보로 확장하면 재요청을 줄일 수 있다.

### 5. Risks (FMEA RPN — 완화 규칙)

**사용자 결정 반영**: 모든 리스크에 RPN을 강제하면 응답이 비대해진다. **RPN ≥ 100인 리스크만** S × O × D 점수 명시. 나머지는 정성 서술 1~2줄.

**형식은 목록 (표 금지)** — 표 형식은 컬럼이 5~6개로 늘어나 RPN 점수·게이트·완화책이 좁은 셀에 끼어 가독성이 오히려 떨어진다. 목록은 RPN 계산식과 게이트를 indent로 inline 표시 가능해 한눈에 보인다. (2026-05-23 사용자 피드백 반영.)

본 프로젝트 척도:

| 축 | 1점 | 5점 | 10점 |
|---|---|---|---|
| Severity | log 한 줄 변경 | TBT +10% | PACT2026 측정 차단 / production crash |
| Occurrence | 검증 환경에서만 | 1B 모델에서 가끔 | 모든 backend 항상 |
| Detection | host cargo test로 잡힘 | S25 device test 필요 | logcat·production에서만 |

**임계값**:
- RPN ≥ 200 — **작업 차단**, 게이트 통과 필수
- RPN 100~200 — 추가 검증 게이트 명시
- RPN < 100 — 정성 서술만, best-effort

작성 형식 (목록 — 표가 아닌):

```
정성 서술 (RPN < 100):
- [경미] CLI flag 명칭 일관성 (정성)
- [경미] DATA_CONSUMER_PATTERNS 9 vs 예상 11 갭

정량 (RPN ≥ 100):
- R-1 [RPN 144] cpu_kernels 함수 포인터 hot path 회귀
    Severity 6 (TBT regression 가능) × Occurrence 4 (모든 layer) × Detection 6 (host 부분 잡힘, S25 필요)
    게이트: S25 Qwen 2.5 1.5B Q4_0 TBT Δ ≤ +3% 통과 → 초과 시 매크로 fallback
- R-2 [RPN 105] cpu_kernels() 미주입 backend 'expect' panic — robustness 회귀
    S 7 × O 3 × D 5
    게이트: spec test에 non-None assertion 추가 + init 단계 검증
- R-3 [블로커 RPN 240] SeqMajor silent garbage 패턴 재현
    S 8 × O 5 × D 6
    작업 차단 — 호출지 전수 사전 검증 필수
```

각 RPN ≥ 100 항목은 (1) ID + 한줄 + 점수 / (2) 계산식 indent / (3) 게이트 indent 3블록. 표보다 indent 목록이 한눈에 들어온다.

**왜 sparse RPN인가**: 사소 리스크까지 점수화하면 작성 부담 + 응답 가독성 저하. 중요 리스크만 정량화하면 가중치 차이가 명확해진다.

### 6. Tradeoffs — Sensitivity / Tradeoff points (ATAM)

| 종류 | 의미 | 본 프로젝트 예 |
|---|---|---|
| Sensitivity | 작은 변경이 큰 품질 영향 | FlashAttn DK=128 per-thread 32 float4 임계 |
| Tradeoff | 한 품질을 위해 다른 품질 희생 | sub_group_reduce 직관 vs Adreno 33-55% 느림 |

추천이 어느 임계 근처에 있는지, 어느 품질을 희생하는지 명시.

### 7. Devil's Advocate Pass (필수 강제 1단락)

3대 불만 2번 (리스크 표면화) 직격. CIA Tradecraft Primer의 단일 안 반박 패턴.

응답 마지막에 별도 섹션 1단락. 두 문장 템플릿:

- "내 결론이 틀렸다면 가장 가능성 높은 이유는 ___."
- "이 결론이 맞으려면 ___ 가정이 참이어야 한다. 이 가정의 약점은 ___."

본 세션 B-5b R1 case의 DA pass 예:

```
Devil's Advocate — default impl 추가가 아니라 enum capability flag
패턴이 더 맞을 수 있다. trait method를 추가하면 ISP 누적 부채가 본
sprint 외로 빠지는데, capability enum이면 새 backend 추가 시 enum
variant 1개로 끝나 ISP 부채 자체가 발생하지 않는다. 이 결론이
맞으려면 향후 새 backend (Vulkan / WebGPU)가 추가될 가능성이
낮다는 가정이 필요한데, EuroSys 2027 사후 리팩토링 트랙에서
Vulkan PoC를 이미 검토했으므로 가정의 약점이 명확하다.
```

### 8. Recommendation (단일안)

**사용자 결정 반영**: 별도 ADR 라이프사이클 도입 안 함. 본 프로젝트 spec/ 안에서 흡수.

단일안 + Status 한 줄:

| Status | 의미 |
|---|---|
| Proposed | 추천이지만 (a)/(b)도 합리적. Architect/PM 합의 필요 |
| Accepted | 후보 비교 끝, Implementer 진입 가능 |
| Conditional | 정량 게이트 통과 시에만 Accepted |
| Superseded by [spec ID] | 폐기 — 신규 결정으로 대체 |

작성 예:

```
Recommendation: 후보 (a) Backend trait + default impl 4 method
Status: Conditional — S25 microbench TBT Δ ≤ +3% 통과 시 Accepted
근거: §13.8-F~J 정책 인플레이션 회피 + 호출지 영향 0 + 작업량 최소
관련 spec: INV-LAYER-001~007 무변경, §13.5 Resolution Log V-?? 갱신
```

### 9. Open Questions (미결 + ownership)

Rust RFC "Unresolved questions" 강제. **결정 전 해소 vs 구현 중 해소 분리** + 누가 답하는지 명시.

```
Open Questions
- [Architect 답변 필요, Phase 2 진입 전] SecondaryStore trait method
  set이 호출지 전수조사 결과로 충분한가?
- [구현 중 해소 가능] cpu_kernels 함수 포인터의 inline hint 필요성
- [사용자 결정] B-5b Phase 2 진입 시 worktree 재사용 vs 신규?
```

ownership 없는 질문은 "누군가 답할 것" 함정. 명시 강제.

## Appendix 작성 가이드

본문 결론을 뒷받침하는 메타정보. **짧게 유지** — Reviewed Items는 파일·라인 5건 + spec ID 3건 + 기각 후보 2건 정도, Coverage는 3~4줄.

### A. Reviewed Items

- **읽은 파일·라인**: 결정에 영향을 준 핵심 5건만 (`engine/src/backend.rs:1-1268`)
- **인용한 실측 데이터**: 정량 근거 2~3건 (`baseline 206건 = INV-LAYER-003 150건 + ...`)
- **검토했지만 기각한 후보**: 1줄씩 (`(d) status quo — sunk-cost 우려`)
- **확인한 invariant·spec**: 본 결정 무영향 또는 영향 영역 (`INV-LAYER-001~007 영향 없음`)

전부 표시 X — 결론 정당화에 필요한 것만.

### B. Coverage — Design / Functionality / Tests / Backend

본 프로젝트 맥락 보강. 3~4줄:

- **Design**: L1~L5 layering 일치 — O/X + 사유
- **Functionality**: prefill / decode 영향 범위
- **Tests**: host cargo test + S25 device + Jetson CUDA — 회귀 커버 영역
- **Backend Coverage**: NEON / Adreno / CUDA — 회귀 가능 영역

Design 리뷰 스코프만 강제. Plan 리뷰는 한 줄 요약 OK.

## 본 프로젝트 특화 운영

### 5 Whys는 선택

사용자 결정 반영: 5 Whys 깊이 검증은 응답 길이 부담이 크므로 강제 아닌 선택. 다음 경우에만 사용:
- 표면 추론에 의심이 들 때 ("trait 추가가 깔끔하므로" 같은 약한 근거)
- 사용자가 "더 깊이"·"근본 원인"·"진짜 이유"를 직접 물을 때

본 세션 B-5b R4 case에서 "왜 §13.8-K 신설을 피해야 하는가?"의 5 Whys는 1단에서 멈췄다. 5단까지 갔으면 "본문 무결성 보존"이 진짜 이유라는 결론에 도달 가능. 깊이가 필요한 결정에서는 응답에 포함.

### 인용 출처

리뷰 응답에 외부 출처를 직접 인용하지는 않는다 (응답 비대화). 단 본 스킬 자체의 근거가 궁금한 사용자에게는 `papers/review_research_report.md`를 안내.

### ASCII 다이어그램 작성 팁

리뷰 응답 인라인은 ASCII 사용. 본 프로젝트 권장 패턴:

- **trait + impl tree**: `▲ │ ├── └──` 박스 + tree. method 옆에 `[NEW]` `default: <값>` annotation으로 정보 압축.
- **dispatch sequence**: `──▶` `◀───` 화살표 + lane. Before/After 두 블록 권장.
- **모듈 의존 graph**: tree (`├── └──`)가 graph보다 ASCII 친화. 화살표는 `→ depends on` 의미.
- **state machine**: 상태 박스 (`[Cold]`) + `──▶` 전이.

너무 큰 다이어그램은 분할. zoom in: 영향 sub-tree만 그린다.

> **docs/, arch/ 정식 문서는 Mermaid** (메모리 `feedback_mermaid_diagrams.md`). Architect가 작성하는 docs와 메인 세션이 작성하는 리뷰 응답은 다른 컨텍스트라 모순 아님.

## 자기점검 체크리스트

응답 발송 전 본인이 답한다. 하나라도 No면 보강:

### 본문 강제
- [ ] **§2 Context의 현재 문제 + 기대 이득이 명확히 적혀 있는가?** (정량 권장, 정성도 OK — 억지 수치보다 명확한 정성이 낫다) — 4대 보강 5번
- [ ] **구조 변경이 있고 Design 스코프이면 §3a UML이 있는가?** (ASCII) — 4대 보강 6번
- [ ] **주요 trait 변경이 있으면 §3b Before/After 표 + 구현체 영향 매트릭스가 있는가?** — 4대 보강 7번
- [ ] **§4 Alternatives에 최소 2 후보 + status quo가 있는가?** (3대 불만 2번)
- [ ] **§5 Risks가 목록 형식인가?** (표 금지) — 가독성. RPN ≥ 100 항목 점수와 게이트가 indent로 inline 표시되어 있는가? (3대 불만 2번)
- [ ] **§7 Devil's Advocate Pass 1단락이 있는가?** (3대 불만 2번 — 강제)
- [ ] **§8 Recommendation에 단일안 + Status가 명시되어 있는가?** (3대 불만 3번)
- [ ] **§9 Open Questions에 ownership이 명시되어 있는가?** (3대 불만 3번)

### 골격
- [ ] **본문 결론 직결 섹션(2~9)이 Appendix보다 앞에 있는가?** (4대 보강 4번)
- [ ] **Appendix(Reviewed Items / Coverage)가 짧게 유지되었는가?** (5건 + 3~4줄)
- [ ] **스코프에 맞는 섹션 부분집합을 선택했는가?** (모든 리뷰에 본문 9섹션 강제 X)
- [ ] **UML이 ASCII인가?** (Mermaid 사용 X — Claude Code 콘솔 직접 렌더)

## 트리거 보조 메모

본 스킬은 키워드 매칭만으로 트리거하기보다 **사용자 의도**를 본다:

- 단순 "확인해줘" — verification → 트리거 X
- "(a) vs (b) 어느 쪽이 나아?" — 비교 → 트리거 O
- "이 방향 진행 전에 리스크 한번 더" — 사전 리뷰 → 트리거 O
- "지금까지 한 거 정리해줘" — handoff → handoff-doc 스킬
- "코드 SOLID로 봐줘" — post-impl → design-review 스킬

## 구조 참고

`.claude/skills/review/assets/review_template.md`에 채울 빈 템플릿이 있다.

가벼운 리뷰는 본문 4섹션만 (1·2·5·8), Plan 리뷰는 7섹션, Design 리뷰는 전 9섹션 + §3 강제. 템플릿을 그대로 복사하지 말고 스코프에 맞게 자르고 사용한다.
