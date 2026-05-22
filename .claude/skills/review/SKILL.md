---
name: review
description: 구현 전 Plan/Action/Decision 사전 리뷰와 Design/Architecture 리뷰를 일관된 10섹션 골격으로 수행한다. '리뷰', 'review', '검토', '재검토', '리스크 확인', '설계 관점 문제 확인', '옵션 비교', '계획 검토', '의사결정 재검토' 같은 요청 시 반드시 이 스킬을 사용. 사용자가 "리스크는 없는지 한번 더 확인", "옵션 N 비교", "이 방향 진행 전 검토", "왜 이게 더 나은지", "단점은?" 등을 묻는 모든 경우에 트리거. 코드 작성 후 SOLID 분석은 design-review를 쓰고 본 스킬은 구현 전 사전 리뷰·아키텍처 결정에 집중한다. 단순 사실 확인이나 "방금 한 작업 보여줘" 같은 verification에는 트리거하지 않는다.
allowed-tools: Read, Glob, Grep, Bash, WebSearch, WebFetch
---

# Review Skill — Plan/Action 사전 리뷰 + Design 리뷰

구현 전 작업 계획·설계 결정·트레이드오프에 대해 **고정 10섹션 골격**으로 응답한다. 외부 베스트 프랙티스 (ADR / Rust RFC / Premortem / Google eng-practices / ATAM / FMEA / Anthropic Extended Thinking / Devil's Advocate + 5 Whys) + 본 프로젝트 B-5b case study 분석에서 도출된 7원칙을 운영 형태로 압축했다.

조사 원본: `papers/review_research_report.md` (또는 jobs cache).

## 왜 필요한가 — 3대 불만

본 스킬이 직접 해소하는 사용자 불만:

1. **구체적 설명 부족** — 에이전트가 검토·고려한 내용이 사용자에게 전달 안 됨 (속으로는 더 봤을 텐데 결론만 짧게 옴)
2. **리스크 표면화 부족** — 낙관 편향. landmine, 안 가본 길, 반증 가능성, 실패 시나리오 누락
3. **구조/일관성 부족** — 리뷰마다 다른 포맷. 같은 항목이 어느 땐 있고 어느 땐 없음

각 섹션이 어느 불만에 답하는지 본문에 명시.

## 적용 범위

- **메인 세션 전용**. architect/researcher/PM/implementer/tester 서브에이전트는 자체 응답 포맷 유지.
- 본 스킬을 호출한 메인 세션 응답이 10섹션 골격을 따른다.
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

## 응답 10섹션 골격

`assets/review_template.md`에 채워야 할 빈 템플릿. SKILL 본문에서는 각 섹션의 작성 가이드만 다룬다.

```
1. TL;DR (3줄)
2. Reviewed Items
3. Design / Functionality / Tests / Backend Coverage
4. Tradeoffs (Sensitivity / Tradeoff points)
5. Risks (정성 서술 + RPN ≥ 100 점수)
6. Alternatives Considered (≥ 2 + status quo)
7. Premortem
8. Devil's Advocate Pass
9. Recommendation
10. Open Questions
```

### 스코프별 섹션 선택

모든 리뷰에 10섹션 전부를 강제하면 응답이 비대해진다. 스코프별로 권장 부분집합:

| 스코프 | 권장 섹션 | 이유 |
|---|---|---|
| **Plan/Action 사전 리뷰** | 1·2·5·6·7·9·10 | 후보안과 리스크가 핵심. 구현 detail (3·4)은 약함 |
| **Design/Architecture 리뷰** | 전 항목 (1~10) | Tradeoffs·Sensitivity가 정수 |
| **Decision 재검토** (R4 식) | 1·2·6·8·9 | 결정 한 점에 집중. 후보 비교 + DA가 핵심 |
| **빠른 sanity-check** | 1·2·5·9 | 짧은 응답이 더 가치 있을 때 |

스코프가 모호하면 사용자에게 "Plan 사전 리뷰로 진행하면 1·2·5·6·7·9·10 7섹션 / Design 리뷰로 진행하면 10섹션 전체. 어느 쪽?"으로 묻는다.

## 섹션별 작성 가이드

### 1. TL;DR (3줄 이내)

핵심 결론 3줄. 추천안 + 가장 큰 리스크 + 게이트 한 줄.

```
- 추천: 옵션 (a) Backend trait + default impl 4 method (Status: Proposed)
- 가장 큰 리스크: cpu_kernels 함수 포인터 hot path 회귀 (RPN 168)
- 게이트: S25 Qwen 2.5 1.5B Q4_0 TBT ±3% 통과 시 Accepted
```

### 2. Reviewed Items (필수 강제)

3대 불만 1번 (구체 설명)에 직접 답하는 섹션. **결론에 들어가지 않은 것 포함**:

- 읽은 파일·라인 범위: `engine/src/backend.rs:1-1268` (57 method)
- 인용한 실측 데이터: `Round 14, kr=0.5 EMR=0.506`
- 검토했지만 기각한 후보안 (1줄 이유)
- 본 프로젝트 invariant 확인 결과: `INV-LAYER-001~007 영향 없음`
- 관련 기존 결정·spec ID: `§13.8-F~J 누적 정책 확인`

**원칙**: 속으로 본 것을 응답 표면에 노출한다. Anthropic Extended Thinking "show your work" 패턴.

### 3. Design / Functionality / Tests / Backend Coverage

Google eng-practices "What to look for"의 핵심 4축. 본 프로젝트 맥락 보강:

- **Design**: L1~L5 layering (INV-LAYER-001~007)과 일치하는가?
- **Functionality**: prefill / decode 양쪽 경로 영향?
- **Tests**: host cargo test + S25 device test + Jetson CUDA 3-way 커버리지?
- **Backend Coverage**: CPU NEON / OpenCL Adreno / CUDA Jetson — 어느 backend에서 회귀 가능?

Plan 리뷰면 약식, Design 리뷰면 상세.

### 4. Tradeoffs — Sensitivity / Tradeoff points (ATAM)

| 종류 | 의미 | 본 프로젝트 예 |
|---|---|---|
| Sensitivity | 작은 변경이 큰 품질 영향 | FlashAttn DK=128 per-thread 32 float4 임계 |
| Tradeoff | 한 품질을 위해 다른 품질 희생 | sub_group_reduce 직관 vs Adreno 33-55% 느림 |

추천이 어느 임계 근처에 있는지, 어느 품질을 희생하는지 명시.

### 5. Risks (FMEA RPN — 완화 규칙)

**사용자 결정 반영**: 모든 리스크에 RPN을 강제하면 응답이 비대해진다. **RPN ≥ 100인 리스크만** S × O × D 점수 명시. 나머지는 정성 서술 1~2줄.

본 프로젝트 척도:

| 축 | 1점 | 5점 | 10점 |
|---|---|---|---|
| Severity | log 한 줄 변경 | TBT +10% | PACT2026 측정 차단 / production crash |
| Occurrence | 검증 환경에서만 | 1B 모델에서 가끔 | 모든 backend 항상 |
| Detection | host cargo test로 잡힘 | S25 device test 필요 | logcat·production에서만 |

**임계값**:
- RPN ≥ 200 — **작업 차단**, premortem 필수, 게이트 통과 필수
- RPN 100~200 — 추가 검증 게이트 명시
- RPN < 100 — 정성 서술만, best-effort

작성 형식:

```
Risks
- [경미] CLI flag 명칭 일관성 (RPN < 100, 정성)
- [중간 RPN 168] cpu_kernels 함수 포인터 hot path 회귀
    - S 7 (TBT regression 가능) × O 4 (모든 layer) × D 6 (host 부분 잡힘, S25 필요)
    - 게이트: S25 Qwen 2.5 1.5B Q4_0 TBT ±3% 통과
- [블로커 RPN 324] SeqMajor silent garbage 패턴 재현 risk
    - S 9 × O 4 × D 9
    - 작업 차단 — premortem 필수
```

**왜 sparse RPN인가**: 사소 리스크까지 점수화하면 작성 부담 + 응답 가독성 저하. 중요 리스크만 정량화하면 가중치 차이가 명확해진다.

### 6. Alternatives Considered (≥ 2 + status quo)

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

### 7. Premortem (자유 서술, 강제 없음)

**사용자 결정 반영**: 체크리스트 baseline 제거. 자유 서술 1단락.

"이 작업이 실패했다고 가정하라. 실패의 부고를 쓴다." 가능성 토론이 아니라 사후 분석 톤 (Gary Klein, HBR 2007).

본 프로젝트 표준 실패 모드는 참고만 (강제 아님):
- 1B 모델에서 효과 < 8B의 1/3
- Adreno register spill / SLM 한계
- 3-backend 중 1개만 silent garbage (Det 9점 위험)
- async/sync 비결정 race가 PACT2026 측정 직전 발견
- AUF/legacy/swap 3 path 중 1개만 정확성 깨짐

자유 서술 예:

```
Premortem — 이 작업이 3주 후 PACT2026 측정 직전에 실패한다면:
가장 가능성 높은 시나리오는 Adreno OpenCL backend에서 cpu_kernels 함수
포인터 indirection이 register pressure를 +2 늘리고, DK=128 flash attn이
spill 임계에 부딪혀 GPU TBT가 +8% 증가한 채 발견되는 것이다. host
cargo test는 통과하지만 S25 device test에서야 회귀가 잡힌다.
```

### 8. Devil's Advocate Pass (필수 강제 1단락)

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

### 9. Recommendation (단일안)

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
Status: Conditional — S25 microbench TBT ±3% 통과 시 Accepted
근거: §13.8-F~J 정책 인플레이션 회피 + 호출지 영향 0 + 작업량 최소
관련 spec: INV-LAYER-001~007 무변경, §13.5 Resolution Log V-?? 갱신
```

### 10. Open Questions (미결 + ownership)

Rust RFC "Unresolved questions" 강제. **결정 전 해소 vs 구현 중 해소 분리** + 누가 답하는지 명시.

```
Open Questions
- [Architect 답변 필요, Phase 2 진입 전] SecondaryStore trait method
  set이 호출지 전수조사 결과로 충분한가?
- [구현 중 해소 가능] cpu_kernels 함수 포인터의 inline hint 필요성
- [사용자 결정] B-5b Phase 2 진입 시 worktree 재사용 vs 신규?
```

ownership 없는 질문은 "누군가 답할 것" 함정. 명시 강제.

## 본 프로젝트 특화 운영

### 5 Whys는 선택

사용자 결정 반영: 5 Whys 깊이 검증은 응답 길이 부담이 크므로 강제 아닌 선택. 다음 경우에만 사용:
- 표면 추론에 의심이 들 때 ("trait 추가가 깔끔하므로" 같은 약한 근거)
- 사용자가 "더 깊이"·"근본 원인"·"진짜 이유"를 직접 물을 때

본 세션 B-5b R4 case에서 "왜 §13.8-K 신설을 피해야 하는가?"의 5 Whys는 1단에서 멈췄다. 5단까지 갔으면 "본문 무결성 보존"이 진짜 이유라는 결론에 도달 가능. 깊이가 필요한 결정에서는 응답에 포함.

### 인용 출처

리뷰 응답에 외부 출처를 직접 인용하지는 않는다 (응답 비대화). 단 본 스킬 자체의 근거가 궁금한 사용자에게는 `papers/review_research_report.md`를 안내.

## 자기점검 체크리스트

응답 발송 전 본인이 답한다. 하나라도 No면 보강:

- [ ] **Reviewed Items에 결론에 안 들어간 후보·실측·기각 가설을 1줄씩 적었는가?** (3대 불만 1번)
- [ ] **Alternatives에 최소 2 후보 + status quo가 있는가?** (3대 불만 2번)
- [ ] **Risks에 RPN ≥ 100 항목 점수와 게이트가 명시되어 있는가?** (3대 불만 2번)
- [ ] **Premortem 1단락이 있는가?** (3대 불만 2번 — 강제)
- [ ] **Devil's Advocate Pass 1단락이 있는가?** (3대 불만 2번 — 강제)
- [ ] **Recommendation에 단일안 + Status가 명시되어 있는가?** (3대 불만 3번)
- [ ] **Open Questions에 ownership이 명시되어 있는가?** (3대 불만 3번)
- [ ] **스코프에 맞는 섹션 부분집합을 선택했는가?** (모든 리뷰에 10섹션 강제 X)

## 트리거 보조 메모

본 스킬은 키워드 매칭만으로 트리거하기보다 **사용자 의도**를 본다:

- 단순 "확인해줘" — verification → 트리거 X
- "(a) vs (b) 어느 쪽이 나아?" — 비교 → 트리거 O
- "이 방향 진행 전에 리스크 한번 더" — 사전 리뷰 → 트리거 O
- "지금까지 한 거 정리해줘" — handoff → handoff-doc 스킬
- "코드 SOLID로 봐줘" — post-impl → design-review 스킬

## 구조 참고

`.claude/skills/review/assets/review_template.md`에 채울 빈 템플릿이 있다.

가벼운 리뷰는 4섹션만 (1·2·5·9), Plan 리뷰는 7섹션, Design 리뷰는 전 10섹션. 템플릿을 그대로 복사하지 말고 스코프에 맞게 자르고 사용한다.
