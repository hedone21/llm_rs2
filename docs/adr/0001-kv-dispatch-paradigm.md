# ADR-0001: KV Cache Dispatch Paradigm — Generic Monomorphization → Trait Object Transition

> **Status**: Accepted
> **Date**: 2026-05-28
> **Decision-makers**: Architect + user (KvBundle/WeightBundle grill 종결)
> **Selected**: 갈래 2 — Trait object (`Arc<dyn KVCacheFormat>`)
> **Supersedes**: `engine/src/kv_cache_ops.rs:53` 의 명시 정책 ("Generic monomorphization preserves zero overhead")
> **Related**: `arch/pipeline_stage_design.md` §3.5, §10 (본 grill 결정 #9), spec `INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC` / `INV-KVCACHELAYER-PAIRED-KERNEL` / `INV-STAGE-LAYER-HANDLE`
>
> **명칭 정리 (2026-05-30, grill-with-docs)**: `/CONTEXT.md` 확정에 따라 저장 형태(noun)의 타입명 `KVCacheLayer`/`StandardLayer`/`KIVILayer`(및 `WeightLayer`) → `*Format`, 본문의 "storage/KV paradigm"(저장 형태 의미) → `Format` 으로 정리. 단 **제목·§5 의 "dispatch paradigm"**(Generic ↔ Trait object 라는 *접근법*)은 저장 형태 Format 과 다른 축이라 유지하며, 본 ADR 의 **결정(Generic → Trait object)도 불변**. §6 게이트의 "5 KV 구성"(Sliding/H2O/D2O/KIVI/SnapKV)은 Format(KIVI)·Stage(Sliding/H2O/D2O) 혼재라 'Format' 대신 중립어 '구성'으로 둠. 코드 trait 명 `KVCacheOps` 는 Phase α-K(Generic→dyn) 구현 시 `KVCacheFormat` 으로 동행 rename. INV ID(`INV-KVCACHELAYER-*` / `INV-STAGE-LAYER-HANDLE`)는 추적용 안정 키로 유지.

---

## 1. Context

본 프로젝트는 KV cache 추상화를 `engine/src/kv_cache_ops.rs:55` 의 `KVCacheOps` trait 로 제공한다. 동 파일의 line 53 주석은 다음을 명시한다:

> "Generic monomorphization (`<C: KVCacheOps>`) is used instead of `dyn Trait` to preserve contiguous slice access (`&mut [C]`) and zero runtime overhead."

이 정책은 다음 forward path 패턴에 강결합:
- `LlamaModel::forward(...)` → `<C: KVCacheOps>` generic parameter 전파
- `LlamaLayer::forward(...)` → `kv_cache: &mut C` (concrete type generic)
- `attention_gen<C: KVCacheOps>(...)` → SIMD/GPU dispatch 가 concrete type 에 monomorphize
- `CacheManager` / `EvictionPolicy` / `D2OHandler` / `CachePressurePipeline` 등 ~10 component 모두 generic parameter 전파

KV cache 구현체로 `KVCache` (standard F32/F16/Q4_0) 와 `KiviCache` (KIVI Q2 + residual buffer) 두 개가 공존하나, 두 cache 가 한 모델 인스턴스 안에서 동시에 사용된 적은 없다 (model build 시 한 타입으로 monomorphize).

본 grill (2026-05-28, KvBundle/WeightBundle grill) 에서 다음 결정이 누적되어 본 ADR 작성 필요성이 발생했다:
- **Q27 (Mixed storage 허용)**: Layer 별 다른 KV storage Format 허용 (layer 0 = KIVI Q4, layer 1 = TurboQuant, ...)
- **Q31 ((γ) Layer handle 전환)**: PipelineStage 가 `Arc<dyn KVCacheFormat>` 를 register 시점 보관 — Stage 별 책임 layer 만 보유 (god ctx 회피)
- 5년 시야: 새 KV Format (SnapKV / D2O / TurboQuant / KIVI variants / Sparse / KvOffload 등) 추가 frequency 가 증가 추세 — 매 Format 추가 마다 generic instantiation explosion 발생

기존 Generic monomorphization 정책은 (a) mixed storage 차단, (b) `Arc<dyn KVCacheFormat>` 의 Stage 객체 보관 패턴 차단 (Generic generic parameter 가 trait object 로 들어갈 수 없음), (c) 매 Format 추가마다 forward path 전체의 instantiation 증가. 본 ADR 은 이 trade-off 를 재평가하여 정식 paradigm 전환을 결정한다.

---

## 2. Decision

**KV cache dispatch 를 Generic monomorphization (`<C: KVCacheOps>`) 에서 Trait object (`Arc<dyn KVCacheFormat>`) 로 전환한다**.

상세:
- `KVCacheOps` trait (현 ~15 method) → `KVCacheFormat` trait (5 method: idx / current_pos / capacity / view / 3 mutation + apply_storage). 자세한 시그니처는 `arch/pipeline_stage_design.md` §3.5.
  - **[갱신 주 2026-06-02, R5]** 본 method 집합은 ADR 작성(2026-05-28) *이후* 진화했다 — `view`/`KVCacheView` **삭제**(v2 §4.1 연혁), `apply_storage` **폐기**(결정 #15), `attention_into` **추가**. 현재는 **7 method** = geometry 3(idx/current_pos/capacity) + mutation 3(write_kv/write_kv_batch/compact) + attention 1(attention_into). ADR 본문은 작성 시점 기록으로 보존하며(역사적 결정), 현재 시그니처의 SSOT 는 `arch/pipeline_stage_design_v2.md` §4.1 다.
- Forward path 의 `<C: KVCacheOps>` generic parameter → `&[Arc<dyn KVCacheFormat>]` slice.
- `LlamaLayer::forward` / `attention_gen` / `CacheManager` / `EvictionPolicy` / `D2OHandler` / `CachePressurePipeline` 모두 trait object 기반으로 마이그레이션.
- KVCacheOps 의 15 method 중 read-only 일부 → KVCacheView sub-trait (KVCacheFormat::view 반환).
- mutation method 는 layer-self (`&self` via interior mutability — RCU/Mutex/Atomic). `LayerSlot::rcu_weights` (`engine/src/models/weights/slot.rs:158`) 패턴의 자연 확장.

---

## 3. Rationale

### 3.1 Q27 Mixed storage 허용

본 grill 결정 #4: layer 별 다른 KV storage Format 허용 (KIVI per-layer mix / D2O variance allocation / SnapKV / 향후 Sparse 등). 이는 paper-aligned use case (D2O Eq.10/11 layer-level variance allocation, KIVI partial layer Q2, ...) 를 직접 지원하기 위한 핵심 요구사항.

Generic monomorphization 은 model build 시점에 한 concrete type 으로 고정되므로 layer 별 다른 Format 불가. Trait object 만이 이를 가능케 함.

### 3.2 Q31 (γ) Layer handle 모델

본 grill 결정 #8: PipelineStage 가 register 시점에 `Arc<dyn KVCacheFormat>` 를 보관. Stage 가 자기 책임 layer 만 보유 (god ctx 회피).

Generic parameter (`<C: KVCacheOps>`) 는 trait object 로 들어갈 수 없으므로, Stage 가 layer handle 을 보관하려면 trait object 가 필수. (γ) 모델 채택과 본 결정은 직접 결합.

### 3.3 5년 시야 Format frequency

본 프로젝트 git history (2026-03 ~ 2026-05) 에서 KV Format 추가 frequency:
- 2026-03: D2O (paper alignment)
- 2026-04: KIVI Q2 (residual buffer pattern)
- 2026-05: SnapKV (compress handler stub)
- 2026-05: Weight Swap precision swap (Phase 6.5)

매 Format 추가마다 generic instantiation 증가 — `LlamaModel<KVCache>` / `LlamaModel<KiviCache>` / `LlamaModel<...>` ... binary size 증가 + 컴파일 시간 증가. Trait object 는 Format 추가가 binary size 0 (impl 만 추가).

### 3.4 LayerSlot/RCU 자연 정합

본 프로젝트 `LayerSlot::rcu_weights` (slot.rs:158) 는 이미 `&self` mutation via RCU semantics. KVCacheFormat trait 도 동일 패턴 (interior mutability via `&self`) — 새 정신 도입 cost 0.

### 3.5 Mixed storage 요구사항의 PipelineStage 정합

본 grill PipelineStage hook 패턴 (2026-05-27 결정) 에서 EvictionStage / SwapDispatchStage / KvMergeStage 등이 layer 단위로 책임 분배 — Stage 가 자기 책임 layer 의 storage Format 을 추상화 통해 추론 — Stage 가 storage-agnostic (INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC). 이는 trait object 의 storage-agnostic dispatch 와 직접 매칭.

---

## 4. Consequences

### 4.1 Positive

- **Mixed storage 가능**: layer 별 다른 KV storage Format (Q27 #4 직접 지원)
- **새 Format 추가 cost 감소**: impl 만 추가, 기존 forward path 변경 0. 매 Format 마다 ~20 file refactor 회피.
- **PipelineStage 패턴 자연 정합**: Stage 객체 안 `Arc<dyn KVCacheFormat>` 보관 가능 — (γ) 모델의 핵심 인프라.
- **Stage storage-agnostic**: stage 가 dtype / codebook / rotation matrix / sparse pattern 모름. OCP 강화.
- **Binary size 감소**: generic instantiation 폭증 회피 (5 Format × 모델 = 5 binary blob 회피)

### 4.2 Negative

- **Vtable cost (R-G1 RPN 144)**: Layer-step KV write 가 매 `Arc<dyn>::call` vtable lookup 1회 추가. Modern CPU ~1-3 ns/call. Layer-step (n_layers × decode_step) frequency × cost = ~0.1-0.3 ms/token. S25 TBT 32 ms/tok 의 1% 미만 예상이나 측정 게이트 (Δ ≤ +3%) 통과 필수. **[갱신 주 2026-06-04] 이 vtable 비용은 *dyn trait object flip 가정* 에서만 성립한다** — production hot plan path 의 (3p) 마이그레이션은 **④-a concrete-handle**(static dispatch, vtable 0)로 확정되어 본 vtable 비용이 적용되지 않는다(§8.3/§6.5 [갱신 주 2026-06-04]). (3p) 의 실제 perf 비용 = ④-a getter 의 layer당 Mutex lock.
- **Refactor scope (R-G4 RPN 168)**: CacheManager / EvictionPolicy / D2OHandler / CachePressurePipeline / Forward path ~20 file. Phase α-K (4-6주 high risk).
- **Bit-identical 검증 (R-G5 RPN 168)**: KVCacheOps 15 method → KVCacheFormat 5 method 통합 (compact merges/keep atomic) 으로 D2O / SnapKV / KIVI 의 출력 비트 정확성 회귀 risk. Phase α-K 종료 게이트 (S25 Qwen2.5-1.5B Q4_0 32 token bit-identical) 필수.
- **kv_cache_ops.rs:53 명시 정책 반전**: 미래 explorer 가 "왜 trait object 인가" 의문 가질 수 있음. 본 ADR 이 영구 기록 — kv_cache_ops.rs:53 주석에 ADR-0001 참조 추가 필요 (Phase α-K 구현 단계).
- **Contiguous slice access 손실**: `&mut [C]` (generic) → `&[Arc<dyn KVCacheFormat>]` (trait object). interior mutability 로 `&self` mutation 이라 `&mut` 불필요하나, slice 의 cache locality 일부 손실 가능 — 측정 통해 검증.

### 4.3 Phase α-K 진입 조건

- Phase α-W 종료 + 본 ADR Accepted 상태
- KVCacheOps 15 method → KVCacheFormat 5 method 매핑 표 작성 (R-G2 mitigation)
- CacheManager / EvictionPolicy / D2OHandler / CachePressurePipeline 마이그레이션 plan (R-G4 mitigation)

---

## 5. Alternatives Considered

본 grill 2026-05-28 의 KV dispatch paradigm 갈래 4종 평가 (`arch/pipeline_stage_design.md` §14.1 Q32 갈래 비교).

### 5.1 갈래 1 — Generic 유지 + Mixed storage 포기 (REJECTED)

**형태**: 현 `KVCacheOps` generic 유지. Layer 별 다른 Format 차단.

**거부 사유**:
- Q27 (mixed storage 허용) 정면 충돌. paper-aligned use case (KIVI per-layer mix, D2O variance allocation) 직접 차단.
- 본 grill 결정 #8 (γ Layer handle 모델) 와도 충돌 — generic parameter 가 trait object 안에 들어갈 수 없음.
- 5년 시야 Format frequency 증가 시 generic instantiation explosion.

### 5.2 갈래 2 — Trait object (`Arc<dyn KVCacheFormat>`) (**ACCEPTED**)

**형태**: 본 ADR 의 결정. KVCacheOps generic → KVCacheFormat trait object.

**채택 사유**: §3 Rationale 참조.

### 5.3 갈래 3 — Hybrid (Read = Generic, Write = Trait object) (REJECTED)

**형태**: Read-only path (forward, attention) 는 Generic 유지 (zero overhead 보존). Write path (mutation, eviction) 는 trait object.

**거부 사유**:
- API surface 복잡도 폭증 — caller 가 read 와 write 의 dispatch 방식이 다른 것을 관리.
- Q25 (primitive only) semantic 깨짐 — primitive method 가 read/write 분리되어야 함.
- 본 grill 의 단순성 정신과 충돌 — hybrid 가 본 프로젝트의 PipelineStage 단일 hook 패턴 정신과 어긋남.

### 5.4 갈래 4 — Static enum dispatch (`enum KVCacheFormatVariant { Standard, KIVI, Sparse, ... }`) (REJECTED)

**형태**: Trait object 대신 enum variant 로 static dispatch. 새 Format = 새 enum variant.

**거부 사유**:
- 새 Format 추가 = enum variant 추가 = **OCP 위반**. 본 grill 의 hook 패턴 (PipelineStage) 이 OCP 정신 채택했으므로 KV dispatch 도 동일 정신 — Trait object 가 OCP 정합.
- Variant 추가 시 모든 caller 의 `match` exhaustive 업데이트 필요 — 본 grill 의 stage 객체 캡슐화 정신과 충돌.
- Variant 수가 Format 수로 한계 — extension 없는 closed type system.

---

## 6. Validation Gate

본 ADR 의 결정이 정합한지 검증하기 위한 Phase α-K 종료 게이트:

### 6.1 Functional gate

- **S25 Qwen2.5-1.5B Q4_0 32-token bit-identical**: baseline (HEAD `master`, Generic dispatch) vs Phase α-K branch (Trait object) token id sequence 100% 일치 (모든 KV 구성 — Sliding / H2O / D2O / KIVI / SnapKV).
- 측정: `python scripts/run_device.py -d s25 generate --backend opencl --opencl-rpcmem --model-path qwen2.5-1.5b-q40.auf --prompt "..." --max-tokens 32 --seed 42` × 5 KV 구성.

### 6.2 Performance gate

- **Decode TBT Δ ≤ ±3%**: tok0 inclusive avg_tbt, n ≥ 5 median (`feedback_tbt_metric_tok0_inclusive.md` 부합). INV-147 noise 3배.
- 임계 초과 시 R-G1 vtable cost 직접 검증 — flamegraph profile 로 KVCacheFormat trait dispatch overhead 측정.

### 6.3 Mixed storage gate

- **Mixed storage layer 0 Q4 + layer 1 F16 test PASS**: KIVI per-layer mix 시나리오. Q27 결정 직접 검증.
- 임계: forward 정확성 회귀 0건 + layer 별 dtype assertion PASS.

### 6.4 Refactor scope gate

- **R-G2 매핑 표 완성**: KVCacheOps 15 method 전수 → KVCacheFormat 5 method + KVCacheView sub-trait + 폐기 분류.
- **R-G4 마이그레이션 plan 완성**: CacheManager / EvictionPolicy / D2OHandler / CachePressurePipeline 흡수 경로 명시.

### 6.5 ADR revoke 조건

다음 중 1+ 발생 시 본 ADR Accepted → Rejected 전환, Phase α-K 종료, 갈래 1 (Generic 유지 + Mixed storage 포기) 후퇴:
- 6.1 Functional gate fail (bit-identical 회귀)
- 6.2 Performance gate fail (Δ > +3%) **AND** vtable cost 실측 검증으로 root cause 확인
- 6.3 Mixed storage gate fail
- KVCacheOps → KVCacheFormat 매핑 표에서 semantic loss 5+ method (R-G2)

> **[갱신 주 2026-06-03, cold-path 반영]** 6.2 performance revoke 판정의 무게는 substep 별로 다르다(§8.3 type-flip census 정정). **substep (3)(forward_gen fallback flip)의 perf 게이트 실패는 revoke trigger 강도가 낮다** — forward_gen 은 cold/fallback tier 라 production TBT 미영향이고, (3) device 게이트는 `--no-gpu-plan` 강제로 측정하는 cold path 라 production perf 를 대표하지 않는다. (3) 의 perf 회귀는 fallback path 한정 정보로 기록하되 단독으로 ADR revoke 를 발동하지 않는다(functional bit-identical 회귀는 cold/hot 무관하게 6.1 발동). **production perf revoke 의 무게는 substep (3p)(plan-flip, layer-tier hot)에 있다** — (3p) 의 6.2 게이트 fail(Δ > +3%) **AND** perf cost 실측 확인이 본 ADR revoke 의 정식 perf trigger 다.

> **[갱신 주 2026-06-04, (3p) = ④-a concrete-handle(vtable 0) 정정]** 위 "vtable cost 실측 확인"은 (3p) 를 dyn trait object flip 으로 가정한 표현이라 부정확하다. §8.3 [갱신 주 2026-06-04] 가 소스로 (3p) = **④-a concrete-handle**(static dispatch, vtable 0)임을 확정했다 → (3p) 의 perf revoke 판정에서 **실측 대상은 vtable 이 아니라 ④-a 가 추가하는 layer당 Mutex lock**(`StandardFormat` getter 의 `inner.lock()`)이다. §4.2 의 vtable 비용(0.1~0.3 ms/tok)은 dyn-flip 가정 하에서만 유효하며 ④-a 에는 적용되지 않는다. 따라서 (3p) 6.2 게이트 fail(Δ > +3%) **AND** lock-cost 실측 확인이 정정된 perf revoke trigger 다.

---

## 7. References

- `arch/pipeline_stage_design.md` §3.5 (KVCacheFormat 시그니처), §10 (Phase α-W → ADR-0001 → Phase α-K), §14.1 Q32 (갈래 비교)
- `spec/41-invariants.md` §3.28 (INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC / INV-KVCACHELAYER-PAIRED-KERNEL / INV-STAGE-LAYER-HANDLE)
- `engine/src/kv_cache_ops.rs:53` — Generic monomorphization 명시 정책 (본 ADR 에서 반전)
- `engine/src/models/weights/slot.rs:158` — LayerSlot::rcu_weights 패턴 (본 ADR (γ) interior mutability 자연 확장 근거)
- `.agent/todos/handoff_kv_weight_grill_2026_05_28.md` — KvBundle/WeightBundle grill 종결 handoff
- `feedback_tbt_metric_tok0_inclusive.md` — TBT 측정 메트릭 기준

---

## 8. Implementation Notes (Phase α-K 진입 시점)

본 ADR 의 implementation 은 Phase α-W 종료 후 Phase α-K 에서 진행. 다음을 준수:

1. **kv_cache_ops.rs:53 주석 갱신**: "Generic monomorphization ..." 정책을 ADR-0001 참조로 대체. 정확한 표현은 Phase α-K 시작 commit 에서 결정.
2. **KVCacheOps trait 보존 기간**: Phase α-K 중에는 KVCacheOps + KVCacheFormat 둘 다 존재 (점진적 마이그레이션). KVCacheOps trait 의 폐기는 Phase α-K 종료 + bit-identical 검증 후 별 commit.
3. **마이그레이션 순서** (R-G4 plan 후속):
   - (1) KVCacheFormat trait 정의 + KVCacheView sub-trait + StandardFormat / KIVIFormat impl
   - (2) CacheManager → Stage 분해 (EvictionStage / KvMergeStage / SwapDispatchStage)
   - (3) Forward **fallback** path generic → trait object (`forward_gen` / `attention_into`) — **cold/fallback tier** (아래 [갱신 주 2026-06-03] 참조)
   - (3p) **plan path** generic → **④-a concrete-handle** (`backend/opencl/plan.rs::execute<C: KVCacheOps>` → `Arc<StandardFormat>`/`Arc<KIVIFormat>` static dispatch) — **production hot layer-tier crux**. (4) B-1 차단자를 해소(선결 조건). **dyn trait object flip 이 아니다** — 아래 [갱신 주 2026-06-04] 참조.
   - (4) bit-identical 검증 → KVCacheOps trait 폐기 commit — **(3p)/④-a 는 (4) 의 B-1 차단자만 해소**(plan path 의 `KVCacheOps` generic bound). **(3p) 가 (4) 의 *필요조건이나 충분조건 아님*** — KVCacheOps 폐기 차단자 = 5 cluster(B-1~B-5). **B-5(legacy) 는 사용자 결정(2026-06-04, legacy disposable)으로 ✅해소 → "완전 폐기" 달성 가능**(B-1~B-4 + legacy 폐기/이주 다-substep). 상세 — 아래 [갱신 주 2026-06-04] + §8.3 cross-ref.
   - **[갱신 주 2026-06-02, R3]** §6 게이트는 본 4-step 의 *end-only* 였다 — R3 grill 에서 **substep 별 중간 게이트**로 확장. branch-by-abstraction(item 2 의 KVCacheOps∥KVCacheFormat 공존)이 매 substep 을 runnable 로 만들어 중간 검증을 가능케 한다. 게이트 강도는 substep 이 바꾸는 dispatch tier 에 맞춘다((1)/(4) 무변이나 (4)는 parallel path 제거라 device 측정 / (2) step-tier / (3) layer-tier=perf 위험). avg_tbt 는 (2)/(3)/(4) 전부 측정(가정 말고 실측). **중간 게이트 SSOT = `arch/pipeline_stage_design_v2.md` §9.1** (본 ADR §6 은 종료 게이트 정의로 유지).
   - **[갱신 주 2026-06-03, type-flip ripple census `wf_c2e4bf13-9e3` — 소스 직접 검증]** 위 [R3] 의 "(3) layer-tier=perf 위험" 규정이 **부정확**함이 census 로 드러나 정정한다. production GPU decode hot path = **plan path**(`backend/opencl/plan.rs::execute`)이고, `forward_gen`(→ `forward_into`)은 plan invalidation / build 실패 / `--no-gpu-plan` 시에만 도는 **cold/fallback tier** 다(`session/forward/model_forward.rs::step` 이 매 decode step `execute_plan` 먼저 시도 → `Ok(true)` 면 즉시 return). 따라서 (1) **(3) forward_gen flip 은 production-hot crux 가 아니다** — flip 으로 추가되는 `Arc<dyn KVCacheFormat>::attention_into` vtable + `StandardFormat` interior-mutability lock 은 §4.1(v2) cold/hot 분리의 **cold path** 라 production TBT 에 닿지 않고 `INV-HOTPATH-DISPATCH` 위반이 아니다(§5.2 R-G1 이 명시 허용한 측정-게이트 vtable; §8 표의 layer-tier dyn 금지는 *production hot* 한정). (3) device 게이트는 `--no-gpu-plan` 강제 없이는 vacuous(production 이 plan path 만 타므로 flip 코드 미실행)하며, 강제해도 측정 대상이 production 이 거의 안 타는 cold path 라 production perf 회귀를 직접 증명 못 한다 — (3) 게이트 역할 = *fallback path 정확성(bit-identical) + fallback 자체 perf 회귀 부재* 한정. (2) **진짜 layer-tier perf crux = (3p) plan-flip** — plan path 가 production hot 인데 `plan.rs::execute<C: KVCacheOps>` 는 여전히 `<C: KVCacheOps>` generic 이고 attention 을 `AttentionVariant` enum static dispatch(`attention_into` 호출 0건)로 처리한다. 이 plan path 를 trait flip 하는 것이 production crux 이며, census 이전엔 어느 substep 에도 없어 **(3p) 로 신설**. (3) cut-point 정정(branch-by-abstraction): `ModelForward` 의 단일 `kv_caches: Vec<KVCache>` 필드를 prefill/execute_plan/fallback 이 공유하므로 in-place flip 불가 → **신 entry `forward_into_fmt` + 신 args struct** 로 fallback 분기 후 단일 호출처 전환. 세부 (3a) trait gap host-additive(누락 method 를 base trait 에 추가 금지 — `INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC` — impl 내부 흡수) → (3b) `attention_into`+`write_kv` GPU scatter 흡수 → (3c) `forward_into_fmt` wiring + decode fallback 단일 호출처 전환(★device `--no-gpu-plan` 강제) → (3d) prefill flip + plan 평가. 정정 상세 SSOT = `arch/pipeline_stage_design_v2.md` §9.1.

   - **[갱신 주 2026-06-03, (3c) 2-증분 분리 — eviction census 소스 직접 검증]** 위 (3c) 가 "decode fallback 단일 호출처 전환(forward flip) + eviction 발화 device 게이트" 를 한 증분으로 묶었으나, census 가 **eviction 발화와 fmt flip 이 동시 성립 불가**함을 잡았다: eviction dispatch chain(`CacheManager::force_evict` → `execute_dispatch(&mut [KVCache])` → `run_policy_eviction` → `for cache in caches.iter_mut() { policy.evict(cache,...) }`)이 contiguous `&mut [KVCache]` slice 에 강결합하는데, fmt 활성 시 캐시는 `Vec<Arc<StandardFormat>>`(각 `KVCache` 를 by-value 소유)라 그 slice 를 만들 수 없다(본 ADR §4.2 핵심 충돌점). 해소 = eviction 을 §4.2 interior-mutability(eviction 도 `KVCacheFormat::compact(keep, merges)` `&self` 경유)로 옮긴다. 따라서 (3c) 를 **(3c-fwd: forward flip, NoEviction happy-path 한정) + (3c-evict: eviction→compact flip)** 두 증분으로 분리한다. **(3c-fwd) ✅ 완료 (`c2b05aff`, S25 PASS)**. **(3c-evict)** = `EvictionPolicy` 가 keep-list(+merges) 를 산출해 `fmt.compact()` 를 호출하는 신 write-path 경로(R-G4 write-path 마이그레이션 — `CacheManager`/`EvictionPolicy`). bit-identical 범위 = **Sliding · H2O · StreamingLLM · NoEviction**(소스 등가 검증). **deferred**: per-head **H2O+**(layer-wide keep-list 표현 불가, `compact_keep_positions_for_head` 사용) → (3c-evict-perhead); 가중 merge **D2O**(Eq.11 가중 vs `apply_merges` 균등평균 불일치 + `CachePressureHandler`) → (3c-evict-d2o). (3c-evict) device 게이트는 eviction **subcommand 필수**(`eviction sliding` 등 — `--eviction-target-ratio` 는 일반 필드라 `is_standard_happy_path` 를 탈출 못 함) + `--no-gpu-plan` 강제 + F16/Q4_0/F32-device-only carve-out. 정정 상세 SSOT = `arch/pipeline_stage_design_v2.md` §9.1-EVICT.

   - **[갱신 주 2026-06-04, (3p)/(4) ④-a viability cut — wf `wf_00283de8-f1a` 소스 직접 재확인]** 위 (3p)/(4) 표기 2건을 정정한다(SSOT = `arch/pipeline_stage_design_v2.md` §9.1 의 동일 정정 cross-ref).
     - **정정 1 — (3p) = dyn trait object flip 이 아니라 ④-a concrete-handle(vtable 0).** `plan.rs::execute<C: KVCacheOps>`(:1257) 가 C 에 닿는 표면 = **6 스칼라 getter**(capacity/current_pos/res_pos/q2_tokens) + **`advance_pos` 1회**(:1828) 뿐이고, attention 은 `AttentionVariant` enum static dispatch(`attention_into` 호출 0건)다 → (3p) 의 설계 경로는 v2 §4.1(연혁 ④/R4)이 명시한 **④-a concrete-handle**(`Arc<StandardFormat>`/`Arc<KIVIFormat>`, static dispatch, vtable 0)이다. 귀결: **본 ADR §4.2 의 vtable 비용(0.1~0.3 ms/tok)은 dyn-flip 가정에서만 성립** — ④-a 는 vtable 0(§6.5 [갱신 주 2026-06-04] 참조). (3p) perf 측정 대상은 vtable 이 아니라 **getter+advance 의 layer당 Mutex lock**(`StandardFormat` 의 `inner.lock().unwrap()` 패턴). 현 production plan path 는 generic monomorphization(`execute::<KVCache>`)이라 이미 vtable 0 + lock 없음 = perf-optimal → (3p)/④-a 는 perf neutral-or-slightly-worse(cleanup 목적, gain 아님). lock 비용은 end-to-end avg_tbt device 게이트가 직접 측정(격리 microbench 폐기 — vtable 0 이라 측정 대상 부재).
     - **정정 2 — (3p)은 (4)의 필요조건이나 충분조건 아님(§8.3:208 "마지막 generic 소비자" 부정확).** 전수 census 결과 `KVCacheOps` 폐기((4)) 컴파일 차단자 = 5 cluster: **B-1** plan `execute<C>`(:1257) ← (3p)/④-a 가 해소 / **B-2** `forward_into<C>`(transformer.rs:1491) + 레이어 chain(`forward_gen`/`forward_prefill`/`update_kv_cache`/Args) ← full-surface 라 ④-a 로 못 벗김, fmt fork 는 decode-only·게이트 OFF 라 prefill+비-decode forward flip 선결 / **B-3** offload(`PrefetchableCache: KVCacheOps` supertrait kv_cache.rs:16 + `forward_into_offload<C>` transformer.rs:2906 + `preload_erased<C>` preload_pool.rs:177, production hot 아님 → `PrefetchableCache` 를 `KVCacheOps` 비의존 재정의해 분리 권장) / **B-4** eval `run_eval_ll_generic<C>`(eval_loop.rs:45) + `StepHook<C>`/`CacheSnapshot<C>`(KVCache/KiviCache 런타임 다형성) / **B-5 legacy 충돌 → ✅해소** `legacy/generate.rs`(현 device-gate bin)가 `forward_into`/`execute_plan`/`execute_plan_for_kivi`/`forward_into_offload`/`run_eval_ll_generic` + `use KVCacheOps` 전부 호출하나, **사용자 결정(2026-06-04): legacy 는 참고용 보존·호환성 불필요 → 마이그레이션이 깨면 폐기**. frozen 가정 폐기로 B-5 는 hard blocker 아님(부수효과: device-gate 를 `argus_cli` 로 이주 선결). 따라서 **(4) 완전 폐기 달성 가능** — 남은 판단점은 B-5 가 아니라 **(3p) hot-path Mutex lock 비용 + cleanup ROI**(옵션 A vs BC, SSOT §9.1 "⚠️ α-K (3p)/(4) 방향" 블록 참조).
