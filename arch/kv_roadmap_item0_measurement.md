# KV 캐시 확장성 로드맵 — 항목 0 측정 스프린트 설계서

> **성격**: 측정 전용 설계서. production 동작을 바꾸지 않는다(아래 §6 Spec Triage 참조).
> **상위 스프린트**: `.agent/todos/sprint_kv_roadmap_item0_2026_06_12.md` (P1 산출물)
> **상위 트랙**: `backlog.md` L865~942 (KV 캐시 관리 확장성 로드맵)
> **평가 방법론**: `docs/30_evaluation_methodology.md` (EMR / PPL / Top-K)
> **작성**: 2026-06-12 (Architect, P1) / **Researcher 입력**: 2026-06-12 구현 명세

---

## 0. 요약 (Executive Overview)

조사된 KV 관리 기법(2024–2026)의 효익은 대부분 7–8B+/long-context 검증이고, Round 14–15에서 1B의 누적 score 차별화는 무가치로 판명됐다(BOS=3002.7 vs prompt avg 3.3 지배). 따라서 후속 확장(어휘 확장 항목 1–3, read-plan 항목 4) 착수 전에 **확장 0개로 가능한 측정 4종을 1B host 실측으로 게이트**한다.

| # | 측정 | 논문 | 게이트 대상 | 무조건/조건부 |
|---|------|------|------------|---------------|
| 1 | **R-KV** | arXiv 2505.24133 (NeurIPS'25) | sliding 대비 redundancy-aware eviction 우열 | **조건부 2단** (§3.1) |
| 2 | **A2SF** | arXiv 2407.20485 | forgetting factor의 1B BOS 지배 완화 | 무조건 |
| 3 | **head importance 분산** | arXiv 2407.11550 (Ada-KV, 전제 검증) | 항목 6(per-head budget) 개봉/보류 | 무조건 |
| 4 | **Demote 모사** | arXiv 2412.12706 (EMNLP'25) | 항목 1(`Demote` op) 전체 GO/RED | 무조건 |

**핵심 설계 원칙 3개**:
1. **임계 고정** — 4종 판정 임계(§5 표)는 P3 측정 전에 본 문서로 확정한다. P3 실행 후 변경 금지(사후 조정 = 결과 오염).
2. **최소 신규 표면** — 측정용 프로토타입은 production CLI/IPC/spec을 오염시키지 않는다. R-KV는 `KVCacheStage` 1개 추가(기존 dispatch에 등록만), A2SF는 기존 `decay` 필드 재사용(신규 코드 ≈0), head 분산은 측정 전용 덤프 hook, Demote는 host 스크립트 + 순수 함수 재사용.
3. **A2SF 격리** — A2SF는 `AttentionScoreAccumulator`를 만지는데, 같은 accumulator를 `backlog.md` L1112 QCF_kv 설계 라운드가 동시 작업 중이다. A2SF는 `decay` 기본값 0.0을 유지하고 off일 때 기존 경로 bit-identical이어야 한다(§4 P2b 제약).

---

## 1. 공통 측정 셋업 (4종 공유, 셋업 1회)

### 1.1 모델·환경 (사용자 결정 D1·D2)

| 항목 | 값 | 근거 |
|------|-----|------|
| 모델 | Llama 3.2 1B (16층 × 8 kv_head × head_dim 64) | D1 — Round 14–15 연속성, 항목 0-3 전제와 일치 |
| max_seq_len | 2048 | Round 14–15 기준선 |
| 환경 | host 단독 (CPU backend) | D2 — 품질 지표(EMR/PPL/Top-K)는 디바이스 무관 |
| KV type | **F32** | V-norm/score readback 정확성 필요(QCF_kv·R-KV redundancy 모두 dequant 없는 정본 K/V 요구). Q4_0은 Demote 모사(측정 4)의 *대상*이지 측정 *기반*이 아님 |
| KV layout | HeadMajor | production 고정 (MEMORY: SeqMajor는 Plan 경로 silent garbage) |
| 샘플링 | greedy (temperature 0) | 결정성 — EMR/Top-K 비교의 전제 |

### 1.2 프롬프트 (Known Bug 회피 필수)

`experiments/prompts/benchmark_prompts.json`의 PPL-01~05 (5개 도메인: Literary/Encyclopedic/Technical/Conversational/News)를 **연장**해 사용한다. 단 프롬프트 길이가 핵심 제약이다:

| 제약 | 값 | 근거 (MEMORY Known Bugs) |
|------|-----|--------------------------|
| 프롬프트 길이 | **≥ 300 tok** (목표 ~400–512) | Bug 5: 짧은(≤40 tok) 프롬프트는 min_kv_cache floor 미달로 eviction no-op. e2e eviction 검증은 긴 프롬프트 필수 |
| `--protected-prefix` | **4** | Bug 1: 기본값 `input_ids.len()` → 프롬프트 전체 보호 → score-based eviction이 생성 토큰만 제거 → 무의미 |
| `--eviction-window` | **2048** | Bug 2: sliding `should_evict()`가 prefill 중 자동 eviction → H2O/D2O와 불공정 비교. window=2048로 자동 eviction 억제 후 신호 주입으로만 eviction 발동 |
| 생성 길이 | 512 (1024 보조) | Round 10 매트릭스 정합 |

> **프롬프트 길이 확보**: 기존 PPL-01~05가 ~30–50 tok이라 부족하다. `experiments/prompts/`에 ≥300 tok 연장본(PPL-01L~05L 등)을 P3 셋업 시 준비하거나, NIAH filler_blocks(F-01~F-08) 조합으로 길이를 채운다. 이 준비는 P3(Tester) 셋업 항목이며, 본 설계서는 길이 하한만 고정한다.

### 1.3 베이스라인 (모든 측정 공통)

| 베이스라인 | 역할 |
|-----------|------|
| **Full KV** (eviction 없음) | 상한선 (PPL/EMR 분모) |
| **Sliding window** | 1차 비교 대상 — Round 14–15에서 1B 최적 eviction으로 판명 |
| **H2O** (kr=0.5) | 2차 비교 대상 (score-based 대조군) |

`docs/30_evaluation_methodology.md` §3 메트릭 정의(EMR / Top-K Overlap / Entropy Ratio / PPL ratio)를 그대로 사용한다. 신규 메트릭은 측정별 게이트 지표(MPC/redundant fraction, BOS ratio, concentration C_h, demote-vs-sliding PPL)만 추가한다.

---

## 2. 측정 4종 프로토콜

각 측정은 **입력 → 절차 → 산출 지표 → 판정 임계(고정)** 4단으로 기술한다. 임계 확정 표는 §5에 모은다(본 절에서는 근거를 기술).

### 2.1 측정 1: R-KV (cosine redundancy + importance joint eviction)

**기법 요지** (arXiv 2505.24133 Eq.5-6):
- redundancy R: K(key) pairwise cosine N×N 행렬의 row-mean → softmax 정규화 (토큰 t의 "다른 토큰과 얼마나 중복인가").
- importance I: 최근 α=8 query window attention 평균 (SnapKV식 max-pool).
- fusion Z = λ·I − (1−λ)·R, **λ=0.1** (redundancy 지배). per-head single-shot top-k 선택.

**GQA 주의**: redundancy/importance 모두 KV-head 단위(8개)로 측정·선택한다. K는 KV-head별로 1세트만 존재(GQA), R-KV의 N×N도 KV-head별 행렬이다.

#### 2.1-A 입력
- 프롬프트: §1.2 (≥300 tok PPL-01~05L), `--protected-prefix 4`, `--eviction-window 2048`.
- eviction 발동: 생성 시작 직후 신호 주입(experiment schedule) → target_len을 budget(예: before_len × keep_ratio)으로.
- keep_ratio: 0.5 (Round 15 H2O 비교 정합).

#### 2.1-B 절차 (2단 구조 — §3 조건부 게이트와 결합)
1. **1단 (저비용 덤프)**: prefill-end + eviction 시점에 KV-head별 K pairwise cosine을 계산해 두 게이트 지표를 덤프한다 — (a) **MPC** = mean pairwise K-cosine (전체 N×N 비대각 평균), (b) **redundant fraction** = "nearest-neighbour cosine > τ=0.5인 토큰 비율". 이 1단은 R-KV stage의 redundancy 계산부만 호출하므로 full eviction 프로토타입 없이 측정 가능.
2. **게이트 분기 (§3)**: redundant fraction이 §5 임계 미달이면 R-KV는 **보류 판정**으로 종결(2단 생략). 임계 이상이면 2단 진입.
3. **2단 (full 프로토타입 + EMR)**: R-KV stage를 sliding/H2O와 동일 하네스에 등록해 동일 budget·동일 evicted 토큰 수로 EMR/Top-K/PPL ratio를 비교.

#### 2.1-C 산출 지표
- MPC (스칼라, KV-head·layer별 분포), redundant fraction (%).
- (2단 시) sliding 대비 EMR Δ, Top-K Overlap, PPL ratio.

#### 2.1-D 판정 임계 근거
R-KV는 8B/reasoning 16K~32K CoT 대상이라 redundancy가 지배적인 환경을 전제한다. 1B/2048/비reasoning에서는 redundant token이 적을 것으로 예상된다(Researcher 입력). redundant fraction이 낮으면 "redundancy-aware의 전제 자체가 1B에 부재" → 보류가 과학적으로 정당한 결론이며, EMR 비교까지 갈 필요가 없다. → §5 임계 확정.

---

### 2.2 측정 2: A2SF (score accumulator forgetting factor)

**기법 요지** (arXiv 2407.20485):
- 재귀 누적 A_n = α·A_{n-1} + S_n. 우리 `attention_scores.rs`의 `decay` 필드 + `begin_step()`(L137-145)의 `factor = 1.0 - decay` 감쇠가 이미 동형이다 → **A2SF α = 1 − decay**.
- 논문 최적 α=[0.1, 0.3] → **decay=[0.7, 0.9] 스윕**.

**핵심 가설과 사전 위험**: Round 15의 BOS 지배(BOS=3002.7 vs prompt avg 3.3) 완화가 가설. 단 **논문 자체가 "forgetting은 sink(BOS) 선택을 못 막는다"고 명시**한다(Researcher 입력). 가설 기각 가능성이 높으나, 측정으로 확정한다(추측 금지 — Round 14–15 교훈).

#### 2.2-A 입력
- 프롬프트: §1.2. score-based eviction(H2O 경로) 활성 필수(`--eviction-policy h2o` 계열, accumulator active).
- 스윕: decay ∈ {0.0 (기준), 0.7, 0.8, 0.9}. 0.0은 **기존 경로 bit-identical** 대조군.

#### 2.2-B 절차
1. decay 각 값으로 동일 프롬프트·동일 budget eviction 실행.
2. eviction 시점에 `importance_scores()` 덤프 → BOS 위치 score vs non-BOS 평균 score 추출.
3. HH(heavy hitter) 집합 = score 상위 top-k 위치. decay=0.0과 decay=0.8의 HH 집합 Jaccard 유사도.
4. 각 decay에서 EMR/PPL ratio vs sliding.

#### 2.2-C 산출 지표
- **BOS/non-BOS ratio** (decay 0.0 vs 0.8) — 완화 여부의 직접 지표.
- HH 집합 Jaccard (decay 0.0 vs 0.8) — forgetting이 선택 집합을 실제로 바꾸는가.
- EMR Δ, PPL ratio vs sliding.

#### 2.2-D 판정 임계 근거
"완화"의 정량 기준 = BOS/non-BOS ratio가 decay 도입으로 유의하게 줄고(§5 임계), 그 결과 EMR이 sliding을 이기는가. 논문이 sink 미해결을 명시했으므로, ratio가 거의 안 줄거나 줄어도 EMR이 sliding 이하면 RED(보류). → §5 임계 확정.

---

### 2.3 측정 3: head importance 분산 (Ada-KV 전제 검증)

**기법 요지** (arXiv 2407.11550 전제): per-head budget 차등이 가치 있으려면 head별 attention concentration 분산이 커야 한다. 논문(Mistral-7B)은 sparse head(상위 5% 토큰이 ~5% 질량 차지 안 됨) vs dispersed head ≈ 10배 차이.

#### 2.3-A 입력
- 프롬프트: §1.2. eviction 불필요 — forward 경로에서 attention 분포만 캡처.
- 캡처 지점: `attention_scores.rs:366 last_step_head_attn()` (proper softmax, head별 sum≈1.0 — **1순위**). 보조로 `accumulate_layer_gqa()` L198의 KV-head 평균 경로.

#### 2.3-B 절차
1. decode 매 step의 마지막 tracked layer per-KV-head attention(`last_step_head_attn`, layout `[n_kv_heads * max_seq_len]`)을 측정 전용 hook으로 캡처.
2. KV-head h별 **concentration C_h** = "상위 5% 토큰이 차지하는 attention 질량 합".
3. 분산 판정 = **max C_h / min C_h** (across 8 head × 16 layer). step 평균(또는 후반 step) 사용.

#### 2.3-C 산출 지표
- per-(layer, kv_head) concentration C_h 매트릭스(CSV, 16×8).
- max/min ratio (스칼라, 항목 6 게이트의 직접 입력).

#### 2.3-D 판정 임계 근거
Ada-KV 전제(8B급 ~10배)가 1B에서 성립하는가. Round 14에서 H2O+(per-head) Δ=+0.011 무효 전례 → 분산이 작게 나올 것으로 예상(Researcher 입력). max/min이 작으면 per-head 차등 budget의 실익이 없어 항목 6 보류, 크면 개봉 후보. → §5 임계 확정.

---

### 2.4 측정 4: Demote 모사 게이트 (항목 1 GO/RED)

**기법 요지** (arXiv 2412.12706, EMNLP'25 — **1B 직접 검증된 유일 논문, scale 안정**): 동일 메모리 예산에서 "4× 토큰 @4bit > 1× 토큰 @16bit". 즉 evict 대신 저정밀 강등(quantized pruning)이 순수 eviction을 이긴다. 단 **2-bit는 1B에서도 급락 — Q4_0 우선, Q2는 극한 예산만**. reasoning/code 태스크는 full-precision 우세(태스크 의존).

#### 2.4-A 입력
- 프롬프트: §1.2.
- 비교 설계 (동일 메모리 예산):
  - (a) **sliding**: N개 토큰 F16 유지(나머지 evict).
  - (b) **demote**: 2N~4N개 토큰 — sliding 창 안 F16 + 창 밖 후보를 Q4_0 (또는 일부 Q2)으로 demote. 메모리 = (a)와 동일하도록 demote 토큰 수 조정 (F16 16B/elem vs Q4 ~4.5B/elem → 약 4× 토큰).
- demote 후보 기준: **1차 = sliding 창 밖 전부 demote (단순)**. 2차(보조) = importance 하위 X% demote.

#### 2.4-B 절차 (host 스크립트, 엔진 코드 무수정)
1. F16 KV에서 demote 후보를 in-place quant → dequant 왕복 — `quant_qcf.rs:15 compute_nmse_block()` + `BlockKVQ4`/`BlockQ2_0` 재사용.
2. **축 정합 필수**: K는 per-channel, V는 per-token 양자화 축. (KIVI 컨벤션 — 잘못된 축은 NMSE 과대.)
3. 왕복 후 KV로 forward 계속 → PPL/EMR 측정.
4. (a) sliding vs (b) demote를 동일 메모리 예산에서 PPL ratio·EMR 비교.

#### 2.4-C 산출 지표
- demote vs sliding의 PPL ratio, EMR (동일 메모리 예산).
- (보조) Q4 vs Q2 demote의 품질 격차 — 논문의 "2-bit 1B 급락" 재현 여부.

#### 2.4-D 모사의 한계 (명시 필수)
quant→dequant 왕복은 실제 mixed-precision 저장과 **품질상 수치 등가**이나, **K 손실의 차기 step 2차 효과(attention weight 교란)는 미포착**한다 (`quant_qcf.rs:170` 주석과 동일 한계 — "K quantization's primary effect is on attention weights, a second-order effect ignored"). 따라서 본 게이트는 **V 손실 + K 직접 손실의 1차 효과**만 본다. demote가 1차에서도 sliding에 지면 RED는 확정적(2차 효과는 손실을 더하지 더 좋게 못 함). demote가 1차에서 이기면 GO이되, 실제 op 구현 시 2차 효과 재검증을 항목 1 Acceptance에 명시(예고, §6).

#### 2.4-E 판정 임계 근거
논문 결과(4× @4bit > 1× @16bit)가 1B/2048/비reasoning에서 재현되는가. demote PPL ratio가 sliding보다 낮으면(품질 우세) GO. → §5 임계 확정.

---

## 3. 측정 순서의 조건부 구조 (판정)

**질문**: R-KV는 (i) MPC/redundant-fraction 저비용 덤프 → 임계 미달이면 (ii) full 프로토타입+EMR을 생략하고 보류하는 2단 구조가 합리적인가?

**판정: 합리적이다. R-KV만 조건부 2단으로 구조화하고, A2SF·head 분산·demote는 무조건 실행한다.**

근거:
1. **R-KV의 게이트 지표는 EMR과 인과적으로 선행한다**. redundant fraction이 낮다는 것은 "중복 토큰이 거의 없다" = redundancy-aware eviction이 sliding 대비 추가로 제거할 중복이 부재하다는 뜻이다. 이 경우 R-KV는 redundancy 항(λ=0.1 가중이라 지배항)이 무의미해져 importance-only(≈ H2O)로 퇴화한다. Round 15가 이미 H2O ≤ sliding을 입증했으므로 EMR 측정은 정보 가치가 없다(예측 가능한 RED). → 2단 생략이 정당.
2. **비용 비대칭**. 1단(redundancy 덤프)은 R-KV stage의 cosine 계산부만 호출(N×N row-mean, eviction 미발동)이라 저비용. 2단(full 프로토타입 stage 등록 + 동일-budget EMR 매트릭스)은 sliding/H2O 3-way 비교라 고비용. 게이트 지표가 결과를 강하게 예측하므로 조건부 분기의 ROI가 높다.
3. **다른 3종은 조건부 분기 부재**. A2SF는 decay 스윕 자체가 측정(중간 게이트 없음), head 분산은 단일 패스 덤프(분기 불요), demote는 1차 효과 비교가 최종 판정(중간 게이트 없음). → 무조건 실행이 단순하고 정직.

**구현 함의** (§4와 연결): R-KV의 1단/2단은 동일 stage 코드의 두 진입점이다. 1단은 stage의 redundancy 계산부만 노출하는 측정 hook(또는 stage의 plan() 결과를 evict 미적용으로 덤프), 2단은 stage를 dispatch에 등록한 full eviction. P2a는 둘 다 구현하되, **P3 실행은 1단 먼저 → 게이트 → 2단**의 순서를 따른다.

---

## 4. 하네스 매핑 + 구현 seam (P2 구현 task 명세)

> **최소 신규 표면 원칙**: 측정 프로토타입은 production CLI/IPC/spec을 오염시키지 않는다. 아래 각 task는 "어느 기존 경로에 무엇을 추가하는가 + 완료 게이트"를 명시한다.

### 4.0 공유 하네스 = `argus_eval` bin

4종 모두 `argus_eval`(`engine/src/bin/argus_eval.rs`)의 기존 모드에 매핑된다. 신규 bin은 만들지 않는다.

| 모드 | 진입 flag | 본 스프린트 용도 |
|------|-----------|------------------|
| `ppl` | `--ppl <text>` + `ppl_*` | R-KV/A2SF/demote의 PPL·EMR·Top-K 측정 정본 경로 (`session/ppl/runner.rs`의 eviction hook에 QCF/score 추출 이미 배선됨 — L1010~1089) |
| `dump importance` | `--dump-importance` | head 분산 측정의 1차 후보 (`session/dump_importance.rs` — prefill importance JSON 덤프). 단 head별 concentration은 별도 hook 필요(§4.3) |
| `experiment` | `--experiment-schedule` | eviction 신호 주입(memory_critical) 스케줄로 eviction 시점 제어 (`ScheduleCommandSource`) |

`ppl` + `experiment` 조합이 핵심 측정 경로다: experiment schedule로 eviction을 정확한 시점에 발동시키고, ppl runner의 eviction hook이 EMR/QCF/score를 산출한다.

### 4.1 [P2a] R-KV 프로토타입 stage — Senior Implementer

**구현 seam**: `KVCacheStage` 트레이트 신규 구현체 1개. D2O가 이미 같은 트레이트로 `D2OStage`를 구현한 선례를 그대로 따른다.

- **재사용**: `engine/src/kv/d2o_handler.rs`의 (a) `dequantize_k()`(L514, pub(crate)) 또는 `StageCtx::dequant_k` reader, (b) `cosine_similarity()`(L495). D2O는 1:N argmax(evict↔retain nearest), R-KV는 **N×N row-mean** — 집계 루프만 신규.
- **신규 stage**: `engine/src/kv/rkv_stage.rs` (no-mod.rs 스타일, `kv.rs`에 `mod rkv_stage;` 추가). `impl KVCacheStage`:
  - `plan(ctx: &dyn StageCtx) -> Option<KVCachePlan>`: ctx에서 `current_pos`/`target_len`/`importance`/`n_kv_heads`/`head_dim`/`dequant_k` 사용. KV-head별 (1) K N×N cosine row-mean → softmax = R, (2) importance I (ctx.importance, 최근 α window), (3) Z = λ·I − (1−λ)·R top-k → `KeepSpec::LayerWide(keep)` (per-head 차등 keep은 §6 항목 6 영역이라 본 프로토타입은 layer-wide 근사 — head별 Z 계산 후 union/평균으로 단일 keep 산출, 측정 충분).
  - **λ=0.1, τ=0.5, α=8 상수**: stage 내부 const 또는 `RkvConfig`(측정 전용, CLI 노출 안 함).
- **1단 측정 hook**: redundancy 계산 결과(MPC, redundant fraction)를 stderr/CSV로 덤프하는 진입점. plan() 내부에서 계산하는 R 행렬을 측정 모드에서 덤프(예: env var `ARGUS_RKV_DUMP` 또는 측정 전용 메서드). **production dispatch에 등록하되 측정 schedule에서만 활성**.
- **dispatch 등록**: 정책 표면은 clap subcommand enum `EvictionCmd`(`engine/src/session/cli/eviction.rs`)다. R-KV를 `EvictionCmd::Rkv` variant + `policy_name() => "rkv"`로 추가하되, **CAOTE 선례**(L68 `#[cfg(feature = "caote")]`)를 그대로 따라 **`#[cfg(feature = "rkv")]` 측정 feature 게이트**로 격리한다 — feature 미설치 시 subcommand 부재 = production 표면 불변. stage 등록은 `engine/src/kv/eviction/stage_registry.rs`의 stage 빌더에서 동일 feature 게이트로 분기. **이렇게 하면 §6 Spec Triage의 "production 정책 목록 불변"이 빌드 타임에 보장된다.**
- **완료 게이트**: (1) `cargo test -p llm_rs2` 무회귀, (2) clippy clean, (3) R-KV stage가 sliding/H2O와 동일 ppl 하네스에서 실행 가능, (4) MPC/redundant fraction 덤프 동작, (5) **순수 함수 단위 테스트**: N×N row-mean + softmax R 계산이 알려진 입력에서 정확(예: 동일 벡터 N개 → R 균등, 직교 벡터 → R 낮음).

### 4.2 [P2b] A2SF forgetting factor — Senior Implementer ⚠ 격리 필수

**구현 seam**: **신규 코드 ≈ 0**. `AttentionScoreAccumulator`(`engine/src/inference/attention_scores.rs`)의 `decay` 필드 + `begin_step()` 감쇠가 이미 A2SF α=1−decay와 동형이다. 필요한 것은 **스윕 하네스(decay를 CLI/schedule로 주입)** 뿐이다.

- **격리 제약 (절대)**: `decay` 기본값 **0.0 유지**. `begin_step()` L137의 `if self.decay > 0.0` 가드가 이미 존재 → decay=0.0이면 감쇠 루프 진입 안 함 = 기존 경로 **bit-identical**. 이 가드를 건드리지 않는다.
  - **근거**: `backlog.md` L1112 QCF_kv 설계 라운드가 같은 `AttentionScoreAccumulator`/`qcf_kv.rs`/`qcf_runtime.rs`를 동시 작업 중. A2SF가 accumulator 로직을 바꾸면 두 작업의 회귀 추적 불가. A2SF는 **decay 값 주입 경로만** 추가하고 누적 로직은 손대지 않는다.
- **신규 표면 = decay 주입 경로 1개**: `decay`는 `AttentionScoreAccumulator::new(...)` 5번째 인자로 이미 받는다. 이를 채우는 CLI/config 값이 있는지 확인 → 없으면 측정 전용 flag(`--score-decay <f32>`, 기본 0.0) 1개 추가. **off(0.0)일 때 기존 동작 보존**이 회귀 테스트 게이트.
- **완료 게이트**: (1) `--score-decay 0.0`일 때 `importance_scores()` 출력이 flag 도입 전과 bit-identical(회귀 테스트 — 기존 `test_accumulate_single_layer` 등이 decay=0.0이므로 자동 커버, 추가로 decay 미지정 경로 = 0.0 단언), (2) `--score-decay 0.8`일 때만 감쇠 동작(단위 테스트: 2-step 누적에서 factor=0.2 적용 확인), (3) BOS/non-BOS ratio 측정 가능, (4) clippy/test 무회귀.

> **commit 격리 보고**: A2SF commit과 L1112 QCF_kv commit의 순서/격리 상태를 handoff에 명시(스프린트 마스터 P2b Notes). 두 작업이 같은 파일을 만지면 PM이 P4에서 머지 충돌 추적.

### 4.3 [P2c] head 분산 instrumentation — Implementer

**구현 seam**: 측정 전용 캡처 hook. **production forward 경로 무영향**(off 시).

- **추출점**: `AttentionScoreAccumulator::last_step_head_attn()`(L366) — proper softmax per-KV-head attention(`[n_kv_heads * max_seq_len]`, head별 sum≈1.0). 이미 pub. 이 값이 GQA mode active(`n_kv_heads > 0`)일 때만 존재 → `argus_eval`이 GQA mode로 accumulator 구성(`eval_setup.rs:279`가 이미 GQA required 처리).
- **신규 표면 = 덤프 hook 1개**: decode loop(ppl runner 또는 dump 모드)에서 매 step `last_step_head_attn()`을 읽어 per-(layer, kv_head) concentration C_h(상위 5% 질량)를 누적·CSV 출력. ppl runner의 eviction hook이 이미 `last_step_head_attn()`을 호출(L1043)하므로 동일 패턴 재사용.
  - 측정 전용 flag(`--dump-head-concentration <path>`) 또는 기존 `--dump-importance` 확장 중 택1. **권장 = `--dump-importance` 출력에 per-head concentration 섹션 추가**(신규 flag 0). `dump_importance.rs`의 JSON에 `head_concentration` 필드 추가.
- **완료 게이트**: (1) per-(layer, kv_head) C_h CSV/JSON 덤프, (2) max/min ratio 산출, (3) instrumentation off(flag 미지정) 시 forward 경로 무영향(기존 dump 출력 불변), (4) test/clippy 무회귀, (5) 단위 테스트: 알려진 attention 분포(예: 한 토큰에 질량 집중 vs 균등)에서 C_h가 1.0 vs ~0.05로 정확.

### 4.4 [P2d] Demote 모사 스크립트 — Implementer

**구현 seam**: **엔진 코드 무수정 host 스크립트** (Researcher·스프린트 마스터 권장). 엔진 경유 강등은 항목 1 실구현이라 범위 밖.

- **재사용 (라이브러리 함수)**: `engine/src/qcf/quant_qcf.rs`의 `compute_nmse_block()`(L15) + `BlockKVQ4`/`BlockQ2_0`의 `quantize`/`dequantize`(`engine/src/quant.rs`). 이들은 순수 함수라 host 스크립트(또는 측정 전용 통합 테스트/bin)에서 직접 호출.
- **모사 경로 2안 중 택1**:
  - **(권장) Rust 측정 전용 통합 테스트 또는 소형 측정 bin**: ppl 하네스로 sliding 실행 → KV 덤프 → demote 후보를 quant/dequant 왕복 → 왕복 KV로 forward 재개 → PPL 비교. 엔진 production 경로 무수정, 측정 전용 코드는 `tests/` 또는 측정 bin에 격리.
  - **(대안) 순수 품질 모사**: PPL 두 번(sliding budget vs demote budget) 측정 + NMSE 왕복 오차를 별도 스크립트로 계산해 정합. 단 forward 재개 없는 NMSE-only는 EMR을 못 줌 → ppl 하네스 경유가 정확.
- **축 정합** (§2.4-B): K=per-channel, V=per-token 양자화 — `BlockKVQ4`가 어느 축 가정인지 확인 후 정합(quant_qcf의 K/V 분리 처리 참조).
- **완료 게이트**: (1) 동일 메모리 예산에서 demote vs sliding PPL/EMR 비교 산출, (2) Q4/Q2 격차 측정(논문 2-bit 급락 재현 여부), (3) **production 엔진 코드 무수정 확인**(diff가 측정 스크립트/테스트에 국한), (4) 모사 한계(§2.4-D K 2차 효과 미포착)를 리포트에 명시.

### 4.5 P2 task 분해 요약표

| Task | 담당 | 신규 표면 | 재사용 정본 | 완료 게이트 핵심 |
|------|------|-----------|-------------|-------------------|
| **P2a** R-KV | Senior | `KVCacheStage` 구현체 1 + `EvictionCmd::Rkv` (`#[cfg(feature="rkv")]` 게이트, CAOTE 선례) + 1단 덤프 hook | `dequantize_k`/`cosine_similarity`(d2o_handler), `KVCachePlan` executor | row-mean R 단위 테스트 + sliding 하네스 실행 + redundant fraction 덤프 |
| **P2b** A2SF | Senior | decay 주입 flag 1 (`--score-decay`, 기본 0.0) — **누적 로직 무수정** | `decay` 필드 + `begin_step()` 가드 (이미 동형) | decay=0.0 bit-identical 회귀 테스트 + decay>0 감쇠 단위 테스트 |
| **P2c** head 분산 | 일반 | 덤프 hook 1 (`--dump-importance` 확장 권장) | `last_step_head_attn()`(이미 pub) | C_h CSV + off 시 무영향 + 분포 단위 테스트 |
| **P2d** Demote | 일반 | host 스크립트/측정 bin (엔진 무수정) | `compute_nmse_block`/`BlockKVQ4`/`BlockQ2_0` | demote vs sliding PPL/EMR + 엔진 diff 0 확인 |

---

## 5. 판정 임계 확정 표 (P3 측정 전 고정 — 변경 금지)

> **이 표의 모든 임계는 P3 실행 전에 본 문서로 확정된다. P3 측정 후 임계를 조정하면 결과 오염이다.** Researcher 제안 임계를 검토·확정했으며, 변경한 항목은 "Architect 조정" 열에 근거를 명시한다.

| # | 측정 | 게이트 지표 | **확정 임계** | 판정 | Researcher 제안 | Architect 조정 |
|---|------|------------|---------------|------|-----------------|----------------|
| 1 | R-KV (1단) | redundant fraction (NN cosine > τ=0.5 비율) | **< 15% → 보류** (2단 생략) | fraction < 15%: 보류. ≥ 15%: 2단 진입 | fraction < ~15% → 보류 정당 | **채택**. 8B reasoning 전제와의 거리를 redundancy 부재로 정량화. τ=0.5는 D2O cosine 정합 |
| 1 | R-KV (2단) | sliding 대비 EMR Δ | **EMR Δ ≥ +3%p AND PPL ratio ≤ sliding → GO** | 둘 다 충족: GO. 아니면 보류 | (2단 일반 우열) | **신설**. "우열"의 정량 기준 명시. +3%p는 Round 15 sliding-vs-H2O 격차(3×) 대비 보수적 유의 하한 |
| 2 | A2SF | BOS/non-BOS score ratio (decay 0.0 → 0.8) | **ratio가 ≥ 30% 감소 AND EMR Δ ≥ +2%p vs sliding → GO** | 둘 다 충족: GO. 아니면 보류 | ratio + Jaccard + EMR (정성) | **정량화**. 논문이 sink 미해결 명시 → 30% 감소를 "유의 완화" 하한으로, EMR이 최종 판정자(완화돼도 sliding 못 이기면 무가치) |
| 3 | head 분산 | max C_h / min C_h (8 head × 16 layer) | **< 2배 → 항목 6 보류. ≥ 5배 → 개봉 후보. 2~5배 → 약한 신호(PM 판단)** | 3구간 | < ~2배 보류 / > ~5배 개봉 | **채택**. 2~5배 중간 구간을 명시(Ada-KV 8B ~10배와 Round 14 무효 사이) |
| 4 | Demote | demote PPL ratio vs sliding (동일 메모리 예산) | **demote PPL ratio < sliding PPL ratio (Q4) → GO. 아니면 RED(항목 1 전체 보류)** | demote가 품질 우세면 GO | demote가 sliding 이기면 GO | **명시**. Q4 우선 판정(논문: Q2 1B 급락). Q2는 보조 측정(극한 예산 참고용, 게이트 판정 불포함) |

**판정 종합 (P4 입력)**:
- 측정 1 R-KV: 1단 보류 / 2단 GO / 2단 보류 중 하나.
- 측정 2 A2SF: GO(항목 2 추진) / 보류.
- 측정 3 head 분산: 항목 6 보류 / 약한 신호 / 개봉 후보.
- 측정 4 Demote: GO(항목 1 추진) / RED(항목 1 전체 보류).
- **D4 적용**: 기법 항목(1, 2) RED여도 인프라 항목(3 QueryStats, 4 read-plan ADR)은 1B 승패 무관 진행. 측정 3·4 결과는 항목 6·1에만 게이팅하고 인프라 항목엔 비차단.

---

## 6. Spec Triage

> `/spec-manage` 스킬의 변경 영향 판정 절차 적용.

### 6.1 기본 가설 검증: "이 스프린트는 spec 변경이 필요 없다"

본 스프린트는 **측정 전용**이며 production 동작을 바꾸지 않는다. 4종 각각을 `/spec-manage` 판정 기준표에 대조:

| 측정 | 변경 유형 | Spec 영향 | 판정 근거 |
|------|-----------|-----------|-----------|
| **R-KV** | 새 트레이트 구현? | **X (arch만)** | `KVCacheStage`는 **기존 트레이트**(ADR-0004). R-KV는 그 구현체 1개를 추가하고 **측정 opt-in 정책**으로 등록할 뿐. 새 인터페이스/불변식 추가 아님. production 기본 정책 목록 불변. 코드 경로 추가 = arch 매핑 갱신 대상 |
| **A2SF** | 값 범위/동작 변경? | **X** | `decay` 필드는 **기존 필드**(`AttentionScoreAccumulator::new` 5번째 인자). 기본값 0.0 유지 + off 시 bit-identical → **동작 추가 아님**. CLI flag 1개는 측정 전용 modifier(plan/protocol 무변경) |
| **head 분산** | 측정 instrumentation | **X** | forward 무영향(off 시) 덤프 hook. 관측만 추가, 동작 무변경 |
| **Demote 모사** | host 스크립트 | **X** | 엔진 코드 무수정. 순수 함수 재사용. spec 무관 |

**결론: 이 스프린트는 spec 변경이 필요 없다. 기본 가설 성립.** arch 매핑은 본 설계서(`kv_roadmap_item0_measurement.md`)가 담당하며, `spec/`·`tests/spec/`·`41-invariants.md` 변경 없음.

### 6.2 측정 전용 CLI flag의 spec 무관성 (명시)

`--score-decay`(P2b), `--dump-head-concentration` 또는 `--dump-importance` 확장(P2c)은 측정 전용 modifier다. spec/30-engine.md의 CLI 표면(production 플래그)은 변경하지 않는다. 측정 후 항목 1·2 실구현으로 갈 때 이 flag들의 production 승격 여부는 그때 별도 Triage한다.

> **주의**: R-KV 측정 정책(`EvictionCmd::Rkv`)은 **`#[cfg(feature = "rkv")]` 측정 feature 게이트**로 격리한다(CAOTE 선례 L68). production 빌드(feature 미설치)에서는 subcommand가 부재하므로 production `EvictionPolicy`/`spec` 카탈로그에 노출되지 않는다 — 빌드 타임 격리가 spec 무관성을 보장한다. 측정 후 GO 시 production 승격은 별도 Triage. feature 게이트 enum variant 추가는 **arch 매핑 갱신**(코드 경로)이지 spec 변경이 아니다.

### 6.3 후속 spec 영향 "예고" (측정 결과가 구현으로 이어질 때)

측정 결과가 항목 1~3·6 구현으로 이어지면 spec 영향이 발생한다. **본 스프린트에서는 예고만** 기재(실제 ID 할당은 해당 항목 착수 시).

| 후속 항목 | 게이트 | 예상 spec 영향 (착수 시) |
|-----------|--------|--------------------------|
| 항목 1 `Demote` op | 측정 4 GO | 새 IR 어휘 — `KVCachePlan.demotes`, `DemoteSpec`, `KVCacheFormat::demote`. ADR-0004 amendment + `PlanAbi` 가산 필드 + 새 INV(demote 미지원 format은 fail-fast). **§2.4-D 명시 한계**(K 2차 효과) 재검증을 Acceptance에 포함 |
| 항목 2 A2SF forgetting (production화) | 측정 2 GO | `decay`의 production 승격 → spec/32 score accumulator 알고리즘 기술 + 기본값 결정. L1112 QCF_kv 라운드와 동일 accumulator라 **조율 필수** |
| 항목 6 per-head budget | 측정 3 개봉 후보 | `KeepSpec::PerHead` 엔진 불변식(new_pos 회계, ragged stride) — 대규모. 트리거 개봉 시 별도 ADR |

---

## 7. 산출물 위치

### 7.1 설계서 (본 문서)

`arch/kv_roadmap_item0_measurement.md` — arch/ 독립 설계 문서 컨벤션(spec 1:1 대응 아닌 feature/measurement 설계)을 따른다. `arch/README.md` 독립 설계 문서 카탈로그에 등록 완료. spec 대응 ID 없음(§6 측정 전용).

### 7.2 측정 결과 디렉토리 (P3 산출물)

기존 `experiments/` 라운드 컨벤션(`experiments/reports/roundNN_report.md` + `experiments/<topic>/PLAN.md`+`REPORT.md`)을 따라 신규 디렉토리를 제안한다:

```
experiments/kv_roadmap_item0/
├── PLAN.md                  # 본 설계서 §1~5 요약 + P3 실행 매트릭스 (Tester 셋업 시)
├── REPORT.md                # 4종 aggregated 판정 (P4 입력)
├── prompts/                 # ≥300 tok 연장 프롬프트 (PPL-01L~05L 등, §1.2)
├── rkv/
│   ├── redundancy_dump.csv  # 1단: MPC, redundant fraction (per layer·kv_head)
│   └── emr_compare.csv      # 2단(조건부): sliding/H2O/rkv EMR·Top-K·PPL
├── a2sf/
│   ├── bos_ratio.csv        # decay 0.0/0.7/0.8/0.9별 BOS/non-BOS ratio + Jaccard
│   └── emr_compare.csv      # decay별 EMR·PPL vs sliding
├── head_variance/
│   └── concentration.csv    # per-(layer, kv_head) C_h 16×8 + max/min ratio
└── demote/
    └── ppl_compare.csv      # sliding vs demote(Q4/Q2) PPL·EMR (동일 메모리 예산)
```

- `experiments/qcf_validation/`, `experiments/proxy_validation/`(PLAN+REPORT 쌍) 컨벤션과 정합.
- raw 측정 jsonl은 각 하위 디렉토리에 둔다(round 보고서 패턴).
- **PLAN.md/REPORT.md 작성 시점**: PLAN은 P3 셋업(Tester), REPORT는 P3 측정 후 → P4 판정(PM + Architect 임계 대조). 본 설계서가 PLAN의 과학적 골격(§1~5)을 고정하므로, Tester는 실행 매트릭스(프롬프트 ID × decay/budget grid)만 채운다.

---

## 8. 변경 이력

| 날짜 | 변경 |
|------|------|
| 2026-06-12 | 초판 (Architect, P1). 측정 4종 프로토콜 + 판정 임계 고정(§5) + 하네스 매핑/구현 seam(§4) + Spec Triage(§6, spec 무관 결론) + 산출물 위치(§7). Researcher 2026-06-12 구현 명세 입력. |
