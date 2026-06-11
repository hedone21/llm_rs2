# ADR-0006: weight 축 stage 통일 — plan-returning `WeightStage` (KVCacheStage 형제; 결정 ⊥ 변형 ⊥ pacing 분리)

> **Status**: Accepted
> **Date**: 2026-06-07
> **Decision-makers**: 사용자 + 메인 세션 (precision/weight plugin 통일 grill 세션; 적대적 검증 워크플로 6-agent 대조 후속)
> **Selected**: KV 의 plan-returning plugin 패턴(ADR-0004)을 weight 축(precision swap / layer skip / tensor partition)에 **거울 복제**한다. plugin 은 `WeightStage::plan()` 한 함수로 **결정만** 내고(Seam A), 변형은 **엔진 executor 가 concrete-handle 로 독점**하며(Seam B, ADR-0004 D1 거울), pacing 은 executor 소유. **precision(format 축)은 dispatch-mode(`LayerDispatch`)에 싣지 않고 분리 운반**(직교 보존). 등록은 `WEIGHT_STAGES` 평행 linkme registry(ADR-0005 D6 확장). **현 상태는 greenfield**(legacy 삭제 + AB-6 미배선)이고 Seam B 소비는 **Phase β 의존**.
> **Related**: ADR-0003(정적 crate + linkme + force-link), ADR-0004(`KVCacheStage` plan-returning, **D1 변형=엔진 독점**), ADR-0005(D6 3축 평행 registry, §8 hotpath tier, §3.3.1 단일 병합 금지), `arch/pipeline_stage_design_v2.md` §4.2(WeightFormat) / §5(PipelineStage) / §8(INV-HOTPATH-DISPATCH), `/CONTEXT.md`(3축 직교 + weight swap = stage+format 합성 / partition = format×hardware 곱)

---

## 1. Context

궁극 목표: **Stage·Format·Backend(capability) 를 모두 plugin(.so)으로**, 중간 단계로 crate 분리. ADR-0003/0004 가 **KV 캐시 stage 축**(`KVCacheStage` plan-returning + `crates/technique-api`)을 완성했고, ADR-0005 가 Format·Backend capability 의 plugin 통일 인터페이스를 확정했다. 본 ADR 은 ADR-0005 §6 가 deferred 한 **weight 축**(precision swap·layer skip·tensor partition) 을 같은 plan-returning 궤도에 올린다.

**발단**: precision(weight) 세계는 KV 와 동형 3축 — weight Stage(어느 layer 를 어느 precision/dispatch 로) ⊥ weight Format(`WeightFormat`, 이미 α-W-5 존재) ⊥ Backend(공유 matmul dtype-dispatch). 그러나 KV evict 는 plan-returning plugin(`KVCacheStage`)을 거치는데, weight 결정(`WeightSwapDecider`)은 엔진 내부에 박혀 plugin 패턴을 안 거치는 **비대칭**이 남아 있었다. 본 ADR 이 그 비대칭을 해소한다.

**검증 전제(적대적 6-agent 대조 후 정정된 출발 사실)**:
- `WeightFormat` trait 은 **이미 존재**(Phase α-W-5) — `engine/src/format/weight_format.rs:21`, `impl WeightFormat for LayerSlot`(`models/weights/slot.rs:188`), live 소비(`models/transformer.rs:1010` `slot.apply_dispatch(LayerDispatch::Full, hw)`, :1017 Partition). 본 ADR 은 신설이 아니라 **기존 WeightFormat 위에 결정층(WeightStage)을 추가**한다.
- weight swap 의 **manager(EngineCommand) live 경로는 현 master 에 부재** — legacy `engine/legacy/generate.rs`(`dispatch_swap_weights` 의 manager 소비자) 가 commit `d5ed71d2` 로 비가역 삭제됐고, 신규 `session/decode_loop.rs` 는 `NoOpSwapStage`(defaults.rs:25)만, `take_pending_swap_weights`(executor.rs:258)는 production 소비자 0, `argus_cli.rs:82` 는 swap 을 명시 reject. 즉 본 작업은 migration 이 아니라 **greenfield wiring**(AB-6 pending).
- `pipeline.rs::PipelineStage`(:165)는 **design-only scaffolding** — `impl PipelineStage` 0개, `PipelineRegistry` struct 부재. Seam B 소비는 **Phase β(decode loop 재작성) 선행**.

**충돌 불변·제약**:
- **3축 직교**(CONTEXT.md): stage ⊥ format ⊥ hardware. weight swap = "stage 로드 + format precision 변경" **합성**(CONTEXT.md:25), partition = "format × hardware" **곱**(CONTEXT.md:69). 이 분해를 역행 금지.
- **ADR-0004 D1**: 변형(buffer mutation)은 plugin 이 아니라 **엔진이 plan 을 실행해 독점**.
- **§8 INV-HOTPATH-DISPATCH**: layer-tier(N×/token, production hot) = 정적만, `dyn` 금지. step/boundary = `Box<dyn>` OK.
- **ADR-0005 §3.3.1**: 단일 통합 registry 병합 금지.

---

## 2. Decision

### D1 — `WeightStage` = plan-returning trait + `WEIGHT_STAGES` 평행 registry. 빌트인은 엔진측 stateless 어댑터.
KV 의 관용구를 거울 복제한다(`KVCacheStage` lib.rs:214 / `KV_CACHE_STAGES` lib.rs:257 / `find_stage` lib.rs:260):
```rust
pub trait WeightStage: Send + Sync {
    fn name(&self) -> &str;
    fn plan(&self, ctx: &dyn WeightStageCtx) -> Option<WeightDispatchPlan>;
}
pub struct WeightStageReg { pub name: &'static str, pub make: fn(WeightStageParams) -> Box<dyn WeightStage> }
#[distributed_slice] pub static WEIGHT_STAGES: [WeightStageReg] = [..];
pub fn find_weight_stage(name: &str) -> Option<&'static WeightStageReg>;
```
- 빌트인 = `WeightSwapDeciderAsStage`(엔진측 어댑터, `EvictionPolicyAsStage` stage_registry.rs:51 거울). **단, 소유 박스 거울이 아님**: `WeightSwapDecider<'a>`(decider.rs:92)는 `Option<&'a dyn ImportanceLookup>` + `Option<&'a dyn QuantNoiseAccess>` + `currently_swapped: &'a [usize]` 를 **borrow** 하는 stateless struct 라 `Box<dyn>`('static) 소유 불가. 따라서 어댑터는 **config 만 보유**(algorithm/allow_boundary)하고, `plan(ctx)` 호출마다 ctx borrow 로 `WeightSwapDecider` 를 **stack 즉석 생성**해 `decide(ratio_max)`(decider.rs:119, **`select` 아님**)를 위임한다.
- `WEIGHT_STAGES` 는 stage 축의 4번째 평행 linkme registry — ADR-0005 D6 의 **확장**(병합 아님, §3.3.1 무위배). 빌트인이 엔진 내장(같은 crate)이라 dead-crate elision 무관이나, **외부 weight technique crate** 는 ADR-0003 §4 force-link(`use crate as _`) + startup self-test(`ensure_builtin_weight_stages_registered` 류, fat-LTO `--gc-sections` 대비) 대상.

### D2 — `WeightDispatchPlan` = 결정만. dispatch-mode ⊥ precision (직교). pacing 은 executor 소유.
```rust
pub struct WeightDispatchPlan { pub per_layer: Vec<LayerDirective> }   // step/boundary-tier, Rust-native (repr(C) 불요, KVCachePlan 거울)
pub struct LayerDirective {
    pub layer: usize,
    pub dispatch: LayerDispatch,         // Full/Skip/Partition — mode (stage/hardware 축). Full 은 unit 유지.
    pub precision: Option<TensorDtype>,  // swap target — format 축. None=현 dtype 유지. (Skip/Partition 은 None)
}
```
- **precision 을 `LayerDispatch` enum 에 넣지 않는다**(KeepSpec ⊥ WeightedMerge 거울). CONTEXT.md:25/69 가 분해한 직교(weight swap=stage+format 합성, partition=format×hardware 곱)를 보존 — precision 은 실제로 `swap_weights(new_dtype)`(slot.rs:141)라는 별개 primitive 이고 `SliceSpec.format` 은 partition 의 per-slice precision(format×hardware 곱의 format 좌표). `LayerDispatch::Full` 은 **unit variant 유지** → 기존 소비자 2곳(slot.rs:199 match, transformer.rs:1010) **무회귀**.
- **pacing 은 plan 에 안 담는다** — `IncrementalSwapPlan`(incremental_plan.rs:25, `drain_chunk`/`set_per_tick`/`is_done`)·`DynamicKController`(dynamic_k.rs)·`ProbingKController`(probing_k.rs)는 관측 TBT 로 per-tick K 를 정하는 **executor 소유 메커니즘**. plugin 결정과 분리(decider 와 별개 모듈 — 검증 confirmed).

### D3 — 변형 = 엔진 executor 독점 (ADR-0004 D1 거울). concrete-handle 은 executor 의 도구.
`WeightStage::plan` 은 `WeightDispatchPlan` 만 반환(읽기/결정). 변형은 **엔진 Seam-B executor** 가 수행:
- executor 가 `Arc<WeightSwapModelRef>`(weight_swap_handler.rs:33 — `Arc<Vec<Arc<LayerSlot>>>` + `secondary_mmap` + `ratio_generation` + `config` + `backend`) + `memory` + `release_worker` + dispatcher 를 register 시점 보유(`Arc<LayerSlot>` **단일이 아님** — `SwapExecutor::execute_on_slots` swap_executor.rs:498 가 5-arg 요구).
- executor 가 `LayerSlot::apply_dispatch`(slot.rs:188) / `SwapExecutor::execute_on_slots`(stateless, 매 호출 new OK) 를 호출해 변형 독점.
- §4.2 의 "precision swap = concrete-handle Stage 가 직접 수행"(weight_format.rs:28)은 **그 concrete-handle 을 쥔 Stage = Seam-B executor(엔진)** 로 정합 — `Arc<LayerSlot>` 은 **변형 도구**지 결정자(plugin Stage)가 아니다. `WeightFormat` base trait 표면은 **무증설 보존**(view() 없음 / apply_storage() 없음 — §4.2 INV 유지).

### D4 — `WeightStageCtx` = 읽기 추상. budget 은 엔진 해소 절대값. Importance↔QuantNoise 비대칭 명시.
```rust
pub trait WeightStageCtx {
    fn n_layers(&self) -> usize;
    fn budget(&self) -> usize;                            // 엔진이 ratio→count + currently_swapped 차감 + boundary 보호까지 해소한 절대 count (KV target_len 거울, lib.rs:95)
    fn pressure(&self) -> u8;                             // graded 압력 0–100 (자율 정책용)
    fn current_format(&self, layer: usize) -> TensorDtype;// 이미 swap 됐나 (LayerSlot::current_dtype 원천)
    fn layer_metric(&self, kind: LayerMetricKind) -> Option<&[f32]>;  // OCP (tensor(kind) 거울)
    fn importance(&self) -> Option<&[f32]> { self.layer_metric(LayerMetricKind::Importance) }   // sugar
    fn quant_noise(&self) -> Option<&[f32]> { self.layer_metric(LayerMetricKind::QuantNoise) }  // sugar
}
#[repr(u32)] pub enum LayerMetricKind { Importance, QuantNoise }
```
- **budget 은 엔진이 해소한 절대 count** — `decide(ratio_max)` 내부 ratio→count(decider.rs:131) + `currently_swapped` 차감 + protected boundary(layer0/last, decider.rs:146)를 엔진이 흡수해 plugin 엔 `needed` 절대값만 노출(KV `target_len` 거울). idempotency·boundary 보호가 plugin 으로 새지 않음.
- **layer_metric 비대칭 명시** — `QuantNoise` 는 `QuantNoiseAccess::as_slice()`(runtime_resources_access.rs)로 **깨끗한 &[f32]** zero-copy 노출. `Importance` 는 `ImportanceTable::entries() -> &[ImportanceEntry]{layer_id, sublayer, importance}`(layer_importance.rs:122)라 flat 아님 → 엔진 impl 이 **SubLayer::Full 투영 + layer_id→dense reorder + NaN→absent** 계약으로 평탄화(production `decide()` 가 쓰는 바로 그 투영, decider.rs:200). sublayer 세분(Attention/Mlp) 노출은 deferred(필요 시 enum variant 추가).
- **QuantNoise 노출 목적 한정** — ε 는 현 production 랭킹에 **미사용**(decider.rs:190: Spearman ρ(imp×ε, imp)=0.998), NaN 후보 배제(INV-127) + U5 ablation(ε-aware 알고리즘) 전용. ctx 1급 노출하되 "랭킹 기여 0, 필터/ablation 용"을 계약에 명시(over-engineering 방지).

### D5 — lifecycle = trigger 종류로 결정. OneShot 은 multi-tick drain 후 Consumed. `&mut` 상태는 Mutex.
- **lifecycle 은 stage 종류가 아니라 trigger 종류**(SSOT §5.4): command-driven = `OneShot`, pressure-driven = `Persistent`. swap 은 `EngineCommand::SwapWeights`(command-only)라 **OneShot**, evict 은 graded Pressure 로도 와서 Persistent — 단 `kv.evict_h2o`(command-driven)면 evict 도 OneShot 가능("evict=항상 Persistent" 부정확).
- **OneShot multi-tick**: 증분 drain 동안 매 `WeightMutate` tick(구 `PreSwap` — **정오 2026-06-11**, 아래) `drain_chunk` 후 `StageOutcome::Continue`(GC 아님, pipeline.rs:115), `is_done()` 되는 tick 에 `Consumed` 반환 → GC. 현 live retire(`Option::take()`) 자리 = `Consumed` 자리. "1회 실행 후 GC"(pipeline.rs:98)는 "1 plan = 1 OneShot 생명, 여러 tick Continue 후 종료 시 Consumed"로 ADR 이 의미 고정.

  > **정오 (2026-06-11, AB-6)**: 본 ADR 작성 시점(2026-06-07)의 "매 `PreSwap` tick" 표기는 β-7 driver 배선 + AB-6 결정으로 **"매 `WeightMutate` tick"** 으로 정정된다. 사유: (1) β-7 census 에서 v1 swap-before 인라인 슬롯이 `PostEviction` phase 로 수렴(decode_loop.rs:338 주석)함이 확정됐고 `PreSwap`/`PostSwapBefore`/`PostSwapAfter` 3 variant 는 driver 발화 0·구독자 0 dead variant 였다. (2) AB-6 사용자 확정 B안(기발화 phase 구독)으로 WeightSwapStage 의 Incremental drain 은 구 `PostEviction`(→ `WeightMutate` rename, `arch/pipeline_stage_design_v2.md` §5.6.5) tick 에 발화한다. 따라서 ADR-0006 D5 의 OneShot multi-tick drain phase = **`WeightMutate`**(삭제된 `PreSwap` 아님). 대응 SSOT = `pipeline_stage_design_v2.md` §5.6.1(dispatch tier D5) / §5.6.5(rename).
- **`on_phase(&self)` + `&mut` 상태**: `IncrementalSwapPlan`/`DynamicKController` 는 `&mut self` API 라 `Mutex` 래핑 강제(`Send+Sync` trait bound 상 `RefCell` 불가). 선례 실증: `IntraForwardSwapHook`(`Mutex<plan>`, intra_forward_swap.rs:126), `PhaseAwareSwapDispatcher`(`Mutex<queue>`+atomics). 단일스레드 추론이라 contention 0.

### D6 — 3-Seam 분리. Seam A 는 지금, Seam B 는 Phase β 의존.
```
Seam C (resilience 경계, EngineCommand) — 이미 통일됨, 단 canonical name 정규화 필요
Seam A (plugin 결정, technique-api, WEIGHT_STAGES) — OCP, 지금 추가 가능
Seam B (엔진 PipelineStage executor, plan 소비+증분+변형) — design-only scaffold, Phase β 선행
```
- **Seam A ⊥ Seam B**: plugin 결정(`WeightStage`)과 루프 dispatch(`PipelineStage` on_phase)는 별개 객체. Seam-B executor 가 plan 소비. 합치면 plugin 이 concrete-handle·pacing·loop 메커니즘에 결합 → OCP 파괴.
- **시점 분리(dependency gate)**: `WeightStage`/`WEIGHT_STAGES`(Seam A)는 α-W 에서 추가 가능하나, Seam B 소비(`on_phase` 구동)는 **`PipelineRegistry` 구현(현 미존재) + decode loop 배선(Phase β) 선행**. 그 전까지 weight swap 은 동작 안 함 — AB-6 가 이 배선.
- **Seam C canonical name 정규화**: wire serde tag(`kv.evict_h2o`)·registry name(`h2o`)·sim harness metric(`kv_evict_d2o`)이 **3중 불일치**이고 `SwapWeights`/`SetPartitionRatio` 는 serde rename 부재. EngineCommand variant → **canonical stage name(= registry name)** 정규화 표를 엔진이 소유(예: `SwapWeights`→`"swap"`, `LayerSkip`→`"skip"`, `SetPartitionRatio`→`"partition"`). "find_weight_stage(name) 직매핑"은 이 정규화 레이어 경유.

### D7 — dtype 2층(api=TensorDtype 표면 / executor=DType). Phase 3 = Q4_0-only INV. DeviceTarget api mirror.
- "api=TensorDtype 단일화"는 **부정확** — `TensorDtype`(lib.rs:56, 3-variant F32/F16/Q4_0)은 `DType`(buffer.rs:25, 7-variant)·`DtypeTag`(shared, Q8_0 포함 4)을 cover 못함. **2층으로 명시**: step-tier plugin 표면 = `TensorDtype`, executor 내부 = `DType`. **현 Phase 3 = Q4_0-only**(SwapWeights 가 Q4_0 만 executable, INV-126)를 INV 로 박고, 표현 불가 dtype 은 silent 아닌 **loud-fail**(`anyhow::bail`). `TensorDtype` 에 Q8_0/BF16/Q4_1 추가는 deferred.
- **`SliceSpec` api 이동**: `hardware: DeviceTarget`(engine `hardware.rs` L2)를 technique-api(linkme-만-의존)가 참조 불가 → **api-side `DeviceTarget` mirror**(`#[repr(u32)]`) + engine↔api `From`/`TryFrom` + variant drift 방지 exhaustive-match(컴파일 게이트). `SliceSpec.format: DType → TensorDtype` 도 동일 lossy 경고.

---

## 3. Consequences

- **weight 결정이 plan-returning plugin 표면을 얻음** — KV 와 완전 대칭(Seam A/B/C, plan⊥변형⊥pacing). 미래 weight 기법(SWIFT 변형·새 precision 정책)은 `WeightStage` plugin 으로 OCP 확장(단 §4 OCP 제약 참조).
- **변형 소유권 일관** — KV·weight 모두 "plugin 결정 + 엔진 executor 변형"(D1 거울). `WeightFormat` 표면 무증설 유지(§4.2 INV 보존).
- **직교 보존** — precision(format) ⊥ dispatch-mode(stage/hardware)가 plan 구조에 반영(`LayerDirective.precision` 분리). `LayerDispatch::Full` unit 유지로 기존 소비자 무회귀.
- **§8 정합** — `WeightDispatchPlan` 은 boundary-tier 산물, forward weight 소비는 정적(`partition_ctx: Option` 분기 transformer.rs:1185 + `load_weights()`)이라 dyn 0. plan path 에 새 경계 call 미삽입.
- **확장 비용** — 외부 weight 기법 = crate + dep 1줄 + force-link 1줄(KV 와 동일), 단 Seam B 가 Phase β 후 배선됨.

## 4. Alternatives considered

- **단일 generic `Stage<Plan>` trait**(KV/weight 통합) — **기각**: object-safety(상이 Plan) + ADR-0005 D6(평행 registry) 위반.
- **precision 을 `LayerDispatch::Full{format}` 에 fold** — **기각**(적대적 C1): CONTEXT.md:25/69 직교 역행, 어휘 곱 폭발, `LayerDispatch::Full` 소비자 2곳 동행 수정. → D2 의 분리 채택.
- **변형을 concrete-handle Stage(plugin)가 직접**(§4.2 원안 그대로) — **기각**(적대적 C2): ADR-0004 D1(변형=엔진 독점)과 모순. → D3 의 executor 독점 채택.
- **`api=TensorDtype` 완전 단일화** — **기각**: weight precision 어휘(q8_0/bf16/q4_1) 손실. → D7 의 2층 + Q4_0-only INV.
- **status quo**(weight 결정 = 엔진 내부 `WeightSwapDecider` 직배선) — plugin 성 0, KV 와 비대칭.

## 5. Risks / Landmines

| 위험 | 심각도 | 완화 |
|---|---|---|
| **weight 순수 plugin OCP 미실증(instance 0)** — KV 는 `caote`(순수 plugin) 실증, weight 는 `WeightSwapDeciderAsStage`(builtin wrapping)만. decider 가 engine-internal trait 3종(`ImportanceLookup`/`QuantNoiseAccess`/`SubLayer`) 결합 | 高 | **builtin-first 로 정직히 범위**. 순수 plugin 경로는 `WeightStageCtx`(D4)만 읽는 caote-equivalent weight crate 1개를 host 테스트로 실증하기 전까지 **가설**로 명시. 실증 불가 시 "OCP 는 KV 한정" |
| **Seam B 기반 부재(Phase β 의존)** — `impl PipelineStage` 0, `PipelineRegistry` 부재. Seam A 만으론 weight swap 미동작 | 高 | D6 dependency-gate: Seam A 추가 ↔ Seam B 배선(AB-6 + Phase β) 시점 분리 명시 |
| **greenfield, not migration** — legacy `dispatch_swap_weights` manager 경로 삭제(`d5ed71d2`), 신규 `NoOpSwapStage`+`take_pending_swap_weights` 소비자 0 | 中 | baseline = legacy frozen(S25 device gate)로 회귀 검증. 호스트 GPU 부재로 검증은 S25/Jetson 한정 |
| **RestoreDefaults swap 역전 = 신규 메커니즘** — `SwapExecutor` 단방향(F16→Q4_0), F16 recall 코드 전무. skip→0/partition 역전과 난이도 질적 차이 | 中 | "swap reversal = 신규 구현(F16 recall/dual-resident + 이미 Consumed 된 OneShot 의 역방향 재기동)"을 **별도 결정 항목**으로 분리 |
| **§8 LTO-숨김 회귀** — crate 경계가 hot call 을 인라인해 숨김(ADR-0005 L2) | 中 | weight 는 plan path 격리가 KV 보다 명확(정적 `partition_ctx`)하나 무측정 → §8 게이트(plan-path bit-identical + S25 TBT Δ≤+3%) 강제 |
| **apply_dispatch 현 구현 한계** — 2-fixed positional(specs[0]=GPU/specs[1]=CPU, slot.rs:212), partition format 변환 미구현 leaf(slot.rs:226 hard-assert), Skip 미배선(slot.rs:276 bail) | 中 | `WeightStage` 가 현 impl 능력(same-dtype 2-way) 초과 plan 을 내면 **런타임 loud-fail**(KV PerHead executor bail stage_registry.rs:103 거울). N-way/per-slice format 변환/Skip 은 Deferred |
| **companion resolve 부재 = loud-fail(graceful fallback 아님)** | 低 | item-4 의 hardware = 클래스 intent, `apply_dispatch(d, &Hardware)` 가 단독 resolve. **부재 시 loud-fail**(slot.rs:217, design v2 §3.5 정합) — "graceful fallback" 표현 폐기 |

## 6. Deferred / 안 가본 길

- **Seam B 실배선**(`PipelineRegistry`/`PipelineDispatcher` 구현 + decode loop 재작성) — Phase β. AB-6(weight-swap SwapStage glue) + AB-4(tensor partition 동적 enable) **둘 다 완료**.
  - **AB-4 결정 (2026-06-11, 사용자 확정 b-1)**: tensor partition 의 동적 enable 은 **`"partition"` `WeightStage` 빌트인(D1 plan-returning 결정층)을 거치지 않는다** — 엔진 직결 OneShot `PartitionStage`(PipelineStage, `arch/pipeline_stage_design_v2.md §5.5`). 근거: SetPartitionRatio 의 결정이 *항등*(directive ratio → 그대로 slot fan-out, method-drop 같은 정책 변환 없음)이라 plan() 결정층이 vacuous + persistent 공유 이득 없음(setup-1회). EvictionStage 의 method-drop OneShot 선례 동형. Seam C 정규화표(§4 line 94)의 `SetPartitionRatio`→`"partition"` 매핑은 **미사용 잔존**(WeightStage 경로 미진입). **재검토 trigger = per-layer 알고리즘**(layer 별 다른 ratio / importance-driven split)이 등장하면 그때 `"partition"` WeightStage 결정층을 도입한다(KV evict 가 plan-returning 인 것과 대칭 회복).
  - **AB-6 결정 (2026-06-11, 사용자 확정)**: weight precision swap 의 동적 enable 도 **`"swap"` `WeightStage` 빌트인(D1 plan-returning 결정층)을 거치지 않는다** — 엔진 직결 OneShot `WeightSwapStage`(PipelineStage, `arch/pipeline_stage_design_v2.md §5.6`). 근거: 현 production swap trigger 정본 = `EngineSwapRuntime::handle_swap_weights`(swap_runtime.rs:137, orphan)가 이미 decider·QCF·4-mode commit 을 한 함수에 결합 수행 → 그대로 Stage 본문 이전이 최소 변경. SetPartitionRatio 와 달리 swap 은 decider(`WeightSwapDecider::decide`)가 *비항등*(importance-aware layer 선택)이나, 그 decider 가 engine-internal(`ImportanceLookup`/`QuantNoiseAccess` borrow stateless)이라 plugin 분리(D1)는 ADR-0006 §5 Risk "weight 순수 plugin OCP 미실증"에 걸려 builtin-first 가 정직하다. **재검토 trigger = per-layer 알고리즘**(layer 별 다른 precision / importance-driven swap geometry)이 등장하면 그때 `"swap"` WeightStage 결정층을 도입한다. Seam C 정규화표(§4 line 94)의 `SwapWeights`→`"swap"` 매핑은 partition 과 동형으로 **미사용 잔존**(WeightStage 경로 미진입).
- **weight 순수 plugin 실증** — `WeightStageCtx` 만 읽는 외부 weight technique crate(caote-equivalent).
- **`WeightStageParams` 필드 확정** — `StageParams`(KV 5필드, lib.rs:230) 의 weight 거울(ratio/target_dtype/skip_ratio/algorithm/allow_boundary). per-technique opaque params vs 공용 struct 비대화(KV d2o 와 동일 미결, lib.rs:225) 동반 결정.
- **swap reversal**(F16 recall) + **RestoreDefaults 합성 역전**(partition + precision 동시) 메커니즘. **(2026-06-11 재확인) Deferred 유지** — Track-AB seam 통합(hook 실배선 §5.9.2)이 *전방향* swap dispatch(F16→Q4_0)를 완성하나, *역방향*(Q4_0→F16 recall)은 `SwapExecutor` 단방향(ADR-0006 §5 Risk "RestoreDefaults swap 역전 = 신규 메커니즘") + 이미 `Consumed` 된 OneShot 의 역기동이 필요해 본 통합 범위 밖이다. AB-6 `submit_swap` 도 RestoreDefaults swap 복원에 미대응(`pipeline_stage_design_v2.md §5.6.4` "swap 복원 no-op").
- **`TensorDtype` 확장**(Q8_0/BF16/Q4_1) — Phase 3 Q4_0-only 해제 시점.
- **partition per-slice format 변환 · N-way(3+) · Skip dispatch** — 현 leaf 미구현(slot.rs:226/276).
- **`decide_dry_run`**(QCF 곡선, ENG-ALG-218) 의 plan-returning 표면 대응.
- **v1 `session/traits.rs::SwapStage`(stateful &mut)** deprecated/공존 정책 — `on_phase(&self)` + LayerSlot interior-mutability 로 흡수.
- ~~**4 swap mode**(Incremental/IntraForward/PhaseAware/LayerImmediate, swap_runtime.rs) 의 단일 `WeightStage` 흡수 vs mode 별 executor 다형성.~~ **AB-6 해소 (2026-06-11)**: `WeightSwapStage`(Seam B 엔진 직결 OneShot)가 4-mode 를 흡수하되, **Incremental 만 Stage 가 multi-tick drain**(step-tier, D5)하고 **IntraForward/LayerImmediate/PhaseAware 는 layer/op-tier 서브시스템**(`LayerBoundaryHook`/`PhaseHook`, INV-147/149/150 — `INV-HOTPATH-DISPATCH` 로 PipelineStage 격상 불가)이라 Stage 는 **hook 설치(commit)만** 담당한다. 상세: `pipeline_stage_design_v2.md §5.6.3`.
- ~~**IntraForward/LayerImmediate hook 의 forward slot 실배선**(AB-6 commit 시점 `let _hook` drop, weight_swap.rs:217-238 — "forward slot 배선은 device 게이트" 미룸).~~ **해소 (2026-06-11, Track-AB seam 통합 설계)**: `WeightSwapStage::commit` 의 IntraForward/LayerImmediate arm 이 hook 을 drop 하지 않고 **공유 cell**(`Arc<Mutex<Option<Arc<dyn LayerBoundaryHook>>>>`, assembly 생성 → WeightSwapStage·ModelForward Arc 양분)에 설치하고, ModelForward 가 매 step `cell.lock()`→forward args `layer_boundary_hook` 슬롯 주입한다(`TransformerModelForwardArgs` 신규 슬롯 + `forward_into` layer loop hook 호출 배선 동반 — production decode struct 에 현재 슬롯 부재). hook lifetime = plan finalize(INV-150) 후 cell 클리어. 상세: `pipeline_stage_design_v2.md §5.9.2`(§5.6.3 표 "commit(설치)만" 의 *설치 메커니즘*). PhaseAware 는 `op_trace::set_phase_hook` global singleton 으로 이미 배선(cell 무관). **본 cell 패턴은 Track A(argus_bench score accumulator 배선, `pipeline_stage_design_v2.md §5.9.1`)와 공유 관용구** — ModelForward 의 β-7 `None`-고정 슬롯(model_forward.rs:7-12)을 사후 주입하는 단일 메커니즘(assembly 생성 `Arc<Mutex<T>>` + ModelForward step-tier lock-read). **단, swap reversal 은 본 해소 범위 밖**(아래 별도 Deferred 항목 유지).
