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

- **Vtable cost (R-G1 RPN 144)**: Layer-step KV write 가 매 `Arc<dyn>::call` vtable lookup 1회 추가. Modern CPU ~1-3 ns/call. Layer-step (n_layers × decode_step) frequency × cost = ~0.1-0.3 ms/token. S25 TBT 32 ms/tok 의 1% 미만 예상이나 측정 게이트 (Δ ≤ +3%) 통과 필수.
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
   - (3) Forward path generic → trait object (LlamaLayer / LlamaModel)
   - (4) bit-identical 검증 → KVCacheOps trait 폐기 commit
