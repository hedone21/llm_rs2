# Sprint: KV 캐시 관리 확장성 로드맵 — 측정 스프린트 (항목 0 + 항목 1 게이트)

**작성**: 2026-06-12 (PM)
**상위 트랙**: `[트랙] KV 캐시 관리 확장성 로드맵` (`backlog.md` L865~942, 2026-06-10 등록)
**착수 순서 (사용자 결정 2026-06-10)**: **0(검증 선행)** → 1–3 → 4 → 5. 본 스프린트 = 그 첫 단계(항목 0) + 항목 1 게이트 실험(D3 합류).
**평가 방법론**: `docs/30_evaluation_methodology.md` (EMR / PPL / Top-K), 프롬프트: `experiments/prompts/`
**진입 문장**: "KV roadmap 측정 스프린트 — 측정 종결(P4 판정 기록 완료). 후속 = 항목 3+4 인프라 착수(D4 확정 경로)"
**Status**: ✅ **완료 — 측정 종결 (2026-06-12)**. P1~P4 전 Phase GREEN. 게이트 판정 4종 확정(R-KV 보류 / A2SF 보류 / head 분산 개봉 후보 / Demote RED), backlog 로드맵 트랙 갱신 반영.
**종합 리포트**: `experiments/kv_roadmap_item0/rerun/REPORT.md` (P3 재측정, HEAD `c702ff83`)
**설계서(게이트 임계 SSOT, 변경 금지)**: `arch/kv_roadmap_item0_measurement.md` §5

---

## TL;DR

조사된 KV 관리 기법(2024–2026, ~70개)의 효익은 대부분 7–8B+/long-context 검증이고, Round 14–15에서 1B의 누적 score 차별화는 무가치로 판명됐다. 따라서 후속 확장(어휘 확장 항목 1–3, read-plan 항목 4) **착수 전에 1B 실측으로 게이트**를 통과시킨다. 본 스프린트는 **확장 0개로 가능한 측정 4종**(하네스 셋업 1회 공유, host 단독)을 묶어 수행한다:

1. **R-KV** (NeurIPS'25 2505.24133) — cosine redundancy + importance joint eviction 프로토타입 stage. sliding 대비 우열 판정.
2. **A2SF** (2407.20485) — score accumulator forgetting factor. 1B BOS 지배(Round 15: BOS=3002 vs prompt avg 3.3) 완화 가설 검증, sliding/H2O 대비.
3. **head importance 분산 실측** — 항목 6(per-head 가변 budget) 개봉 게이트. 1B 16층×8 kv_head의 head별 attention concentration 분산이 8B급(Ada-KV 전제)인지.
4. **Demote 모사 게이트** (항목 1) — F16 캐시 선택 토큰 Q4/Q2 왕복 양자화 모사가 sliding eviction을 PPL/EMR에서 이기는가. RED면 항목 1 전체 보류.

**핵심**: 측정 4종은 하네스 인프라를 공유(셋업 1회)하지만, 각 게이트는 독립 판정한다. 기법 게이트(1, 2)가 RED여도 인프라 항목(3 QueryStats, 4 read-plan ADR)은 **1B 승패와 무관하게 진행**한다 (D4 결정).

---

## 목표

- 트랙 항목 0의 3종 실측 리포트 산출 → 항목 1·2 게이트 판정 + 항목 6 개봉/보류 판정.
- 트랙 항목 1의 Demote 모사 게이트 실측 → 항목 1 전체 GO/보류 판정.
- 후속(항목 1–4) 착수 여부를 데이터로 확정 (추측 금지 — Round 14–15 교훈).

---

## 사용자 결정 4건 (2026-06-12 확정)

| ID | 결정 | 근거 |
|----|------|------|
| **D1** | 측정 모델 = **Llama 3.2 1B** | Round 14–15 기준선 연속성. 항목 0-3 "16층×8 kv_head" 전제와 일치. |
| **D2** | 측정 환경 = **host 단독** | 품질 지표(EMR/PPL/Top-K)는 디바이스 무관. 부수효과: backlog `[P2] argus-eval functional smoke`의 **Linux 런타임 검증이 자연 소화**됨 (해당 항목에 cross-ref 기재). |
| **D3** | 스프린트 범위 = **측정 4종 묶음** | 항목 0의 3종 + 항목 1 게이트 실험(Demote 모사). 하네스 셋업 1회 공유로 비용 절감. |
| **D4** | 게이트 RED 시 분기 = **(a) 항목별 게이트 존중** | RED인 기법 항목(1, 2)은 보류 기록. **인프라 항목 3(QueryStats)·4(read-plan ADR)는 1B 승패와 무관하게 진행** — 가치 근거 = 8B 토대 + prefill TTFT + 논문 기여. |

---

## Phase 분해

| Phase | Owner | 산출물 | 의존성 | 상태 |
|---|---|---|---|---|
| **P1** 측정 설계서 | Architect | 4종 측정 설계서(판정 임계 제안 + 하네스 매핑 + 구현 seam) | 없음 | ✅ DONE (`arch/kv_roadmap_item0_measurement.md`) |
| **P2a** R-KV 프로토타입 stage | Senior Implementer | cosine redundancy + importance joint eviction stage (D2O `dequant_k` cosine 인프라 재사용) | P1 | ✅ DONE |
| **P2b** A2SF forgetting factor accumulator | Senior Implementer | score accumulator forgetting factor (**기본값 off + 기존 경로 bit-identical**) | P1 | ✅ DONE |
| **P2c** head 분산 instrumentation | Implementer | per-layer·kv_head attention concentration 분산 덤프 (host) | P1 | ✅ DONE (단일 layer 해상도 한계 — 아래 P3 결과 §측정 3) |
| **P2d** Demote 모사 스크립트 | Implementer | F16 선택 토큰 Q4/Q2 왕복 양자화 모사 + sliding 비교 (host eval) | P1 | ✅ DONE |
| **P3** 측정 실행 (host) | Tester | raw 측정 + aggregated 리포트 (EMR/PPL/Top-K) | P2a~d | ✅ DONE — 1차 측정→수정 라운드 7커밋→재측정 (아래 §P3 결과) |
| **P4** 판정 + 보고 | PM (+ Architect 판정 입력) | 항목 0·1 게이트 판정 기록 + backlog 갱신 | P3 | ✅ DONE (2026-06-12, 아래 §P4 판정) |

병렬 가능성:
- **P2a/P2b/P2c/P2d 4 트랙은 P1 완료 후 동시 진행 가능** (Senior 2 트랙 + Implementer 2 트랙 분담).
- P2d(Demote 모사)는 엔진 코드 무수정 host 스크립트라 가장 독립적 — P1 설계서의 모사 정의만 고정되면 즉시 착수 가능.
- P2b(A2SF)는 격리 제약(아래 ⚠) 때문에 P2a~d 중 가장 신중하게 진행.

---

## Phase 상세

### [P1] 측정 설계서 (Architect)
- **Status**: TODO / **Sprint**: current / **Dependencies**: 없음
- **Owner**: Architect
- **Description**: 측정 전 과학적 설계를 고정한다. PM은 측정의 **과학적 임계값을 결정하지 않는다** — 전부 Architect 위임.
  - **판정 임계 제안** (4종 각각): "sliding 대비 우열"의 정량 기준(예: PPL 비율 임계, EMR Δ 임계, head 분산의 "8B급" 판정 기준). 측정 전 고정 필수 (사후 임계 조정 = 결과 오염).
  - **하네스 매핑**: 4종이 공유할 host 하네스의 단일 셋업 정의. `argus_eval` bin의 어느 모드(`ppl` / `experiment` / `dump-importance`)에 매핑되는지, 프롬프트 선택(긴 프롬프트 ≥300 tok — Known Bug 5 eviction floor 회피, `experiments/prompts/prefill_512.txt` 등), `--protected-prefix 4` / `--eviction-window 2048` 등 Known Bug 1·2 회피 플래그.
  - **4종 각각의 구현 seam**: R-KV(어느 stage 트레이트에 어떤 인터페이스로), A2SF(accumulator의 어느 지점에 forgetting factor를, off 기본 격리 방법), head 분산(forward 어느 hook에서 캡처), Demote 모사(엔진 코드 무수정 host 스크립트 vs 엔진 경유 — 모사이므로 host 스크립트 권장 여부 판정).
- **Acceptance Criteria**: 측정 설계서 1건(4종 판정 임계 수치 고정 + 하네스 매핑 + 구현 seam 4개 명시). P2a~d 착수 차단 해제 게이트.
- **Notes**: 설계서는 `arch/` 또는 `docs/`에 Architect가 배치(PM 범위 밖). 임계값은 **측정 전 고정** — P3 실행 후 변경 금지.

### [P2a] R-KV 프로토타입 stage (Senior Implementer)
- **Status**: TODO / **Sprint**: current / **Dependencies**: P1
- **Owner**: Senior Implementer (cosine 계열 = score/redundancy 인프라 접촉)
- **Description**: R-KV(cosine redundancy + importance joint eviction)를 현 `KVCacheStage`로 프로토타입(새 TensorKind 0개). D2O의 `dequant_k` cosine 인프라 재사용. 2048ctx에서 redundant 토큰 비율 실측 포함.
- **Acceptance Criteria**: R-KV stage가 sliding/H2O와 동일 하네스에서 측정 가능 + lib/clippy 무회귀 + redundant 비율 덤프.
- **Notes**: R-KV는 reasoning long-CoT 대상 — 2048ctx에서 redundant 비율이 낮으면 보류가 정당한 결론(항목 0 Notes).

### [P2b] A2SF forgetting factor accumulator (Senior Implementer) ⚠ 격리 필수
- **Status**: TODO / **Sprint**: current / **Dependencies**: P1
- **Owner**: Senior Implementer (score accumulator 접촉)
- **Description**: score accumulator에 forgetting factor 추가(plan 사후 적용 불가, 엔진측 소형 수정) 후 sliding/H2O 대비. 1B BOS 지배 완화 가설 검증.
- **⚠ 격리 제약 (필수)**: forgetting factor는 **기본값 off + off일 때 기존 경로 bit-identical**. 근거 = `backlog.md` L1112 항목(`[P2] QCF_kv 정규화 비대칭 + estimator 우회 — 설계 라운드 2026-06-12`)이 **같은 score accumulator를 만지는 중**(`AttentionScoreAccumulator` / `qcf_kv.rs` / `qcf_runtime.rs`). 두 작업이 같은 코드 표면을 동시에 건드리면 회귀 추적 불가 → A2SF는 별도 플래그 게이팅으로 완전 격리, off 경로는 기존 동작 보존 회귀 테스트 필수.
- **Acceptance Criteria**: forgetting factor on일 때만 동작 + off일 때 기존 score accumulator 출력 bit-identical(회귀 테스트) + 1B BOS 지배 완화 측정값 + lib/clippy 무회귀.
- **Notes**: L1112 설계 라운드와의 머지 충돌 위험을 P4에서 PM이 추적. 두 작업의 commit 순서/격리 상태를 handoff에 명시.

### [P2c] head 분산 instrumentation (Implementer)
- **Status**: TODO / **Sprint**: current / **Dependencies**: P1
- **Owner**: Implementer (하네스/instrumentation 계열)
- **Description**: 1B 16층×8 kv_head에서 head별 attention concentration 분산을 forward 경로에서 캡처·덤프(host). 항목 6(per-head 가변 budget) 개봉 게이트의 입력.
- **Acceptance Criteria**: per-layer·kv_head 분산 덤프(CSV 등) + host 측정 가능 + 기존 forward 경로 무영향(instrumentation off 시).
- **Notes**: 측정 전용 — production 경로 변경 금지. P1 설계서가 캡처 hook 지점을 고정.

### [P2d] Demote 모사 스크립트 (Implementer)
- **Status**: TODO / **Sprint**: current / **Dependencies**: P1
- **Owner**: Implementer (host 스크립트 / CLI 계열)
- **Description**: 트랙 항목 1의 **게이트 실험**. F16 캐시의 선택 토큰을 Q4/Q2 왕복 양자화해 "content-aware 강등이 sliding eviction을 품질(PPL/EMR)에서 이기는가"만 선검증. 엔진 코드 무수정 host eval 모사 (P1 설계서가 모사 방식 최종 확정).
- **Acceptance Criteria**: 동일 하네스에서 Demote 모사 vs sliding의 PPL/EMR 비교 산출. GO/RED 판정 가능한 측정값.
- **Notes**: RED면 트랙 항목 1 **전체 보류**(Round 14–15와 동일 위험 = 1B score 신뢰성). 항목 1 구현(`Demote` op / `DemoteSpec`)은 본 스프린트 범위 밖 — 게이트만.

### [P3] 측정 실행 (Tester, host)
- **Status**: TODO / **Sprint**: current / **Dependencies**: P2a, P2b, P2c, P2d
- **Owner**: Tester
- **Description**: 4종을 공유 host 하네스에서 실행. EMR / PPL / Top-K 산출(`docs/30_evaluation_methodology.md` 준수). 긴 프롬프트(≥300 tok) + Known Bug 1·2·5 회피 플래그(P1 설계서 지정).
- **Acceptance Criteria**: 4종 raw 측정 + aggregated 리포트. R-KV/A2SF는 sliding 대비 표, head 분산은 per-head 분포, Demote 모사는 PPL/EMR 비교 표.
- **Notes**: host 단독(D2). 부수효과로 `argus_eval` Linux 런타임 검증 일부 자연 소화.

### [P4] 판정 + 보고 (PM)
- **Status**: TODO / **Sprint**: current / **Dependencies**: P3
- **Owner**: PM (Architect 판정 입력 — P1 임계 대조)
- **Description**: P3 측정값을 P1 설계서의 고정 임계와 대조하여 게이트 판정. backlog 갱신.
  - 항목 0: R-KV/A2SF sliding 대비 우열 판정, head 분산 → 항목 6 개봉/보류.
  - 항목 1: Demote 모사 GO/RED → 항목 1 Dependencies 갱신.
  - D4 적용: 기법 항목(1, 2) RED 시 보류 기록. 인프라 항목 3·4는 승패 무관 진행 확인.
- **Acceptance Criteria**: 트랙 항목 0 Status를 측정 결과 기반으로 갱신(DONE 또는 보류 기록) + 항목 1·6 Dependencies/개봉 판정 기록 + 후속 스프린트(어휘 확장 1–3 / read-plan 4) 착수 권고. handoff 문서 작성.
- **Notes**: PM은 판정 기록·backlog 갱신만 — 설계 결정·구현·실행은 각 역할.

---

## 담당 에이전트 매핑 (요약)

| Phase | 에이전트 | 사유 |
|---|---|---|
| P1 | Architect | 측정 과학 설계(임계/seam) |
| P2a R-KV | Senior Implementer | cosine/redundancy score 인프라 접촉 |
| P2b A2SF | Senior Implementer | score accumulator 접촉(격리 주의) |
| P2c head 분산 | Implementer | instrumentation/덤프 |
| P2d Demote 모사 | Implementer | host 스크립트/CLI |
| P3 측정 | Tester | host 실행/분석 |
| P4 판정 | PM | 게이트 판정·backlog 갱신 |

---

## P3 측정 결과 + P4 판정 (2026-06-12 — 측정 종결)

> **종합 리포트(SSOT)**: `experiments/kv_roadmap_item0/rerun/REPORT.md` (정독). 게이트 임계 = `arch/kv_roadmap_item0_measurement.md` §5 고정값(측정 전 동결, 변경 금지). 환경: HEAD `c702ff83`, llama3.2-1b-f16.gguf, KV f32, CPU backend, greedy. 5도메인 프롬프트(304~327 tok), `--protected-prefix 4`, `--kv-budget 200`, `--ppl-prefill-tokens 150`.

### 측정 경위 — 1차 → 수정 라운드 → 재측정

1. **1차 측정 (2026-06-12)**: P2 구현 4종이 **전부 e2e 배선 누락**으로 판명 (오측정 산출). 대표 증상 = A2SF 1차 스모크가 "decay 0.8 → BOS ratio 증가(11.34→17.11)"로 보고됐으나, 이는 미배선 산출물의 허상.
2. **수정 라운드 — 7커밋**: `449cf55f` `6987d8ef` `2ccb5825` `4baea4e0` `dc79755f` `c702ff83` `1182fa0c` (e2e 배선 완결).
3. **재측정**: 위 HEAD에서 4종 재실행 → 본 §의 수치로 **확정**. A2SF 1차 신호는 **반전**(BOS 증가→감소 63.1%)되어 재측정값이 정본.

### ⚠ 프로토콜 편차 2건 (판정 무효화 없음 — 명시 기록)

- **① prefill 전체 고정 → prefill=150 통일로 대체**: 지시 "prefill을 프롬프트 전체 길이로 고정"은 PPL 모드 구조와 충돌. PPL 모드는 reference 텍스트(=프롬프트)를 prefill+decode로 분할하므로, prefill=전체이면 decode=0 → eviction 미발동(전 도메인 `n_evictions=0` 실측). eviction 측정 자체가 불가. → 차선으로 prefill을 도메인-무관 고정값 150으로 통일(가변 절단 제거 + 전 도메인 동일 + decode 154~177 step으로 eviction 1건 정상 발동). "절단 오염 제거"의 실질은 달성.
- **② EMR 미산출 → PPL ratio로 대체**: EMR은 현 하네스가 미산출. 게이트의 "EMR Δ" 조건은 PPL ratio로 대체 판정. **보류 판정엔 영향 없음** — PPL만으로 이미 GO 미달(R-KV PPL ratio 0.9982 동률, A2SF 2/5 win로 +임계 미달).

### 측정값 요약 (5도메인 평균)

| 측정 | 핵심 수치 | 게이트(설계서 §5 고정) | **판정** | 1차 신호 |
|---|---|---|---|---|
| **1. R-KV** | redundant fraction **0.9964** 포화(τ=0.5 변별력 없음 — 1B K벡터 본질 밀집, MPC 0.49) / rkv≈h2o 퇴화(PPL 차 <0.5%) / vs sliding PPL ratio **0.9982** 동률(2/5 근소 우세) | fraction<15%→보류, 2단 EMRΔ≥+3%p AND PPL≤sliding→GO | **보류** | ✅ 일치 |
| **2. A2SF** | BOS ratio **63.1% 감소**(d0→d8, forgetting 기계적 동작) / vs sliding PPL **2/5 승**(decay=0과 동일, 개선 없음) / HH Jaccard 0.792 | BOS≥30%감소 AND EMRΔ≥+2%p→GO | **보류** (BOS 충족, 품질 미충족) | ❌ 반전 |
| **3. head 분산** | max/min C_h **63.7배**, 0-수렴 head 배제한 2nd-min 기준도 **~9.8배** ≥ 5배. sparse(~0.25)/dispersed(~0.004) head 명확 공존 | <2배 보류 / ≥5배 개봉 후보 / 2~5배 약신호 | **개봉 후보** | ✅ degenerate 해소 |
| **4. Demote** | 실모델 PPL **5/5 RED**(demote ratio 1.22~6.04× 전부 sliding보다 나쁨). NMSE 보조 신호(demote 0.145 < sliding 0.75)는 demote 우세였으나 실추론 PPL이 정본(K 2차 효과가 NMSE에 미포착 — 설계서 §2.4-D) | demote PPL < sliding(Q4)→GO | **RED** | ✅ 일치 |

### P4 판정 (D4 적용)

- **R-KV (항목 0 中)**: **보류**. 1단 fraction 포화로 2단 진입했으나, 2단에서 sliding과 동률 → redundancy-aware가 1B/2048에서 무가치. R-KV는 importance-only(H2O)로 퇴화 확정(설계서 §3 예측대로).
- **A2SF (항목 0 中)**: **보류**. BOS 지배 완화는 확인(forgetting 실동작)되나 PPL 개선 없음(설계서 §2.2-D 예측대로 "완화돼도 sliding 못 이기면 무가치"). → 기법 항목으로 **D4(a) 게이트 존중 = 보류**.
- **head 분산 (항목 0 中 → 항목 6 개봉 게이트)**: **개봉 후보**. per-head budget 차등 가치 신호. **단 단일 layer 해상도 한계**(`last_step_head_attn()`이 최종 디코더 layer 1개만 반환 — API 제약, 설계서 §2.3-B 의도된 단순화). 16층 전체 분산은 미측정 → 항목 6 개봉 트리거 "분산 大 판정"은 **부분 충족**.
- **Demote (항목 1 게이트)**: **RED**. 실모델 PPL 5/5 RED → 논문의 "4×@4bit > 1×@16bit"가 1B/2048/비-reasoning에서 미재현. **항목 1 전체 보류**(D4(a) 게이트 존중). 재개 조건 = 8B/long-context 온보딩 또는 retrieval 태스크 실수요.

### 부수 관찰 — qcf_sum ↔ PPL 역전 (QCF_kv 설계 라운드 보강 실측)

측정 중 `qcf_sum`이 PPL과 역전 관찰: sliding 0.60 vs h2o 0.018인데 실 PPL은 sliding 우세. 이는 진행 중인 **"QCF_kv 정규화 비대칭" 설계 라운드**(`backlog.md` L1112, 2026-06-12 등록)가 적발한 수식 포화 결함을 **독립 경로(품질 eval)에서 보강하는 실측**. 해당 backlog 항목에 cross-ref 한 줄 추가(P4 산출).

### D4(b) 인프라 항목 비차단 확인

- **항목 3 (QueryStats TensorKind)·항목 4 (read-plan ADR)**: 기법 게이트 RED와 **무관하게 진행**(D4 결정). 본 스프린트 측정 결과는 두 항목의 착수 여부에 영향을 주지 않음 — backlog에 "차기 착수 대상"으로 표기(P4 산출).

### 머지 충돌 추적 (P2b A2SF ↔ L1112 QCF_kv 설계 라운드)

- P2b A2SF는 score accumulator를 만지는 작업으로 L1112 QCF_kv 설계 라운드와 같은 코드 표면(`AttentionScoreAccumulator`/`qcf_kv.rs`/`qcf_runtime.rs`)을 접촉. 격리 제약(기본값 off + off 시 bit-identical)은 측정 완료 시점까지 유지됨. **A2SF는 보류 판정**이므로 production 경로로 승격되지 않음 → 두 작업 간 머지 충돌 리스크는 본 스프린트 종결과 함께 해소. L1112 설계 라운드는 독립 진행(사용자 공동 검토 대기).

---

## 제약 (필독)

1. **A2SF accumulator 격리** (P2b ⚠): 기본값 off + off 시 기존 경로 bit-identical. `backlog.md` L1112 QCF_kv 설계 라운드가 같은 accumulator를 동시 작업 중 — 완전 격리 필수.
2. **측정 임계는 Architect가 측정 전 고정** (P1): PM은 과학적 임계값을 결정하지 않는다. P3 실행 후 임계 변경 금지(결과 오염).
3. **host 단독** (D2): 품질 지표는 디바이스 무관. device 게이트 불요.
4. **인프라 항목 비차단** (D4): 항목 3·4는 1B 게이트 RED와 무관하게 진행.
5. **항목 1 구현은 범위 밖**: 본 스프린트는 게이트(모사)만. `Demote` op / `DemoteSpec` 실제 구현은 게이트 GREEN 후 별도 등록.
6. **Known Bug 회피**: 긴 프롬프트(≥300 tok, Bug 5) + `--protected-prefix 4`(Bug 1) + `--eviction-window 2048`(Bug 2) — P1 설계서가 정확한 플래그 고정.
