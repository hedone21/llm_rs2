# Handoff: weight 축 stage 통일 설계 확정(ADR-0006) → crate 단계 구현

**작성**: 2026-06-07
**HEAD**: (이 커밋) `docs(adr): ADR-0006 weight 축 plan-returning WeightStage 통일`
**브랜치**: master
**작성자**: 메인 세션 (precision/weight plugin 통일 grill + 적대적 검증 6-agent 세션)

**다음 세션 진입 문장**: **"ADR-0006 crate 단계 구현 — WEIGHT_STAGES registry + WeightStage/WeightDispatchPlan/WeightStageCtx 신설부터 (Seam A 한정, Seam B 는 Phase β)"**

---

## TL;DR

KV 의 plan-returning plugin 패턴(ADR-0004)을 **weight 축**(precision swap/skip/partition)에 거울 복제하는 설계를 grill 6문답 + 적대적 6-agent 검증으로 확정(→ ADR-0006 Accepted). 핵심: plugin 은 `WeightStage::plan()` 으로 **결정만**(Seam A) / 변형은 **엔진 executor 가 concrete-handle 로 독점**(Seam B, ADR-0004 D1 거울) / pacing 은 executor 소유 / **precision(format)은 dispatch-mode 와 분리**(직교). **멈춘 이유**: 설계·인터페이스 확정 완료, 코드 변경 0. 구현은 별도 세션. **단 Seam B(실동작)는 Phase β(decode loop 재작성) + AB-6 선행 — Seam A 만으론 weight swap 미동작.**

---

## 진행 상태

### 확정 결정 (ADR-0006, grill + 검증 락)

| # | 결정 |
|---|---|
| D1 | `WeightStage` plan-returning trait + `WEIGHT_STAGES` 평행 registry. 빌트인 `WeightSwapDeciderAsStage` = **stateless 어댑터**(decider 가 borrow-struct`<'a>` 라 plan 마다 ctx borrow 로 즉석 생성, `decide(ratio_max)` 위임 — `select` 아님) |
| D2 | `WeightDispatchPlan{per_layer: Vec<LayerDirective{layer, dispatch: LayerDispatch, precision: Option<TensorDtype>}>}`. **precision ⊥ dispatch**(LayerDispatch 에 안 넣음, `Full` unit 유지 → 소비자 무회귀). pacing = executor 소유 |
| D3 | **변형 = 엔진 executor 독점**(ADR-0004 D1 거울). concrete-handle(`Arc<WeightSwapModelRef>` — LayerSlot 단일 아님) = executor 도구. `WeightFormat` 표면 무증설(§4.2 INV 보존) |
| D4 | `WeightStageCtx`: budget=엔진 해소 **절대 count**(ratio→count+swapped 차감+boundary, KV target_len 거울). `layer_metric(kind)` — QuantNoise=깨끗한 &[f32], **Importance=SubLayer::Full 투영 필요**(entries() flat 아님). ε=필터/ablation 전용(랭킹 미사용) |
| D5 | lifecycle = **trigger 종류**(command=OneShot, pressure=Persistent). OneShot multi-tick(Continue 반복→is_done 시 Consumed). `&mut` 상태=Mutex(on_phase &self) |
| D6 | 3-Seam 분리. Seam A(지금) ⊥ Seam B(Phase β 의존) ⊥ Seam C(EngineCommand, **canonical name 정규화 필요**). |
| D7 | dtype **2층**(api=TensorDtype/executor=DType), Phase 3 Q4_0-only INV + loud-fail. DeviceTarget **api mirror** + From/TryFrom + drift 게이트 |

### 적대적 검증이 잡은 교정 (반영 완료)

| 항목 | 정정 |
|---|---|
| WeightFormat 신설 (거짓) | **이미 존재**(α-W-5, slot.rs:188 impl + transformer.rs:1010 live). "위에 결정층 추가"로 재서술 |
| LayerDispatch::Full 소비자 0 (거짓) | **소비자 2**(slot.rs:199, transformer.rs:1010). D2 가 Full unit 유지로 회피 |
| precision in Full{format} (직교 위반) | **R1 역전** — 별도 `precision` 필드(D2) |
| 변형 주체 모순(D1 vs §4.2) | **R2 확정** — 엔진 executor 독점(D3) |
| dispatch_swap_weights = manager 배선 (거짓) | legacy 삭제(`d5ed71d2`). **greenfield**(AB-6) |
| RestoreDefaults swap 역전 (코드 전무) | swap reversal = **신규 메커니즘**(deferred) |
| graceful fallback (거짓) | **loud-fail**(slot.rs:217). item-4 표현 폐기 |

---

## 다음 작업 (crate 단계 구현, 순서대로 — Seam A 한정)

1. **`technique-api` 에 weight 타입 추가** — `WeightStage`/`WeightDispatchPlan`/`LayerDirective`/`WeightStageCtx`/`LayerMetricKind`/`WeightStageReg`/`WEIGHT_STAGES`/`find_weight_stage`/`WeightStageParams` (Weight* prefix, KVCacheStage 동형). + `DeviceTarget` api mirror(repr(u32)). → 검증: `cargo build` + dyn-safe + `find_weight_stage`/`registered_weight_names` 단위 테스트 + release fat-LTO linkme 생존(ADR-0003 §4)
2. **`LayerDispatch`/`SliceSpec` engine→api 이동** — `DType→TensorDtype`(lossy, Q4_0-only INV)·`DeviceTarget→api mirror`. `Full` unit 유지. → 검증: 기존 소비자 2곳(slot.rs:199, transformer.rs:1010) 무회귀(기계적 import 갱신만), clippy clean
3. **`WeightSwapDeciderAsStage` 엔진측 어댑터** — `decide(ratio_max)` 위임, ctx borrow 로 decider 즉석 생성. `#[distributed_slice(WEIGHT_STAGES)]` 등록("swap") + `ensure_builtin_weight_stages_registered` self-test → 검증: builtin 등록 + plan() 이 `WeightSwapDecider::decide` 와 동일 selected_layers 산출
4. **`WeightStageCtx` 엔진 impl**(`&TransformerModel` 위로) — budget 해소·`current_format`·`layer_metric`(QuantNoise=as_slice, Importance=SubLayer::Full 투영) → 검증: importance/noise 투영이 decider 직접 입력과 bit-identical
5. **(Phase β 의존) Seam B 배선** — `PipelineRegistry` 구현 + decode loop → `WeightSwapStage`(OneShot) on_phase → `execute_weight_plan`(IncrementalSwapPlan/SwapExecutor). **AB-6 + Phase β 선행, 본 crate 단계 밖.**

### 위임 prompt (선택)

> **에이전트**: `architect`(spec/arch 정착) → `senior-implementer`(trait/registry/어댑터 구현)
> **권한**: `crates/technique-api/`, `engine/src/format/weight_format.rs`, `engine/src/pressure/weights/`, `engine/src/pressure/eviction/stage_registry.rs`(거울 참조)
> **첫 명령**: "ADR-0006 D1 — WeightStage + WEIGHT_STAGES registry 를 KVCacheStage 동형으로 technique-api 에 신설. Seam A 한정, Seam B 는 Phase β."

---

## Landmines / 미해결 / 안 가본 길

- **Seam B 기반 부재(최대 함정)** — `impl PipelineStage` 0개, `PipelineRegistry` struct 부재(design-only scaffold). **Seam A 만 추가하면 weight swap 은 여전히 미동작**(현 live = `NoOpSwapStage`). 실동작은 Phase β decode loop 재작성 + AB-6 선행. ADR-0006 D6 dependency-gate 준수.
- **weight 순수 plugin OCP 미실증(instance 0)** — KV `caote` 같은 순수 plugin 이 weight 엔 없음. decider 가 engine-internal trait 3종(`ImportanceLookup`/`QuantNoiseAccess`/`SubLayer`) 결합. **builtin-first 로 정직히 범위**. 순수 plugin 은 `WeightStageCtx` 만 읽는 caote-equivalent crate 실증 전까지 가설.
- **Importance flat 화 손실** — `ImportanceTable::entries()`(`&[ImportanceEntry]`)는 SubLayer 차원 보유. `layer_metric(Importance)->&[f32]` 는 SubLayer::Full 투영(production decide() 가 쓰는 그 투영) — sublayer 세분 노출은 deferred.
- **dtype 어휘 천장** — TensorDtype 3-variant < DType 7 < (q8_0/bf16/q4_1). Phase 3 Q4_0-only 라 무영향이나 확장 시 천장. loud-fail INV 로 silent 손실 방지.
- **§8 LTO-숨김** — weight plan path 격리가 KV 보다 명확(정적 `partition_ctx`)하나 무측정. crate 단계에서 §8 게이트(bit-identical + S25 TBT Δ≤+3%) 검증.
- **apply_dispatch 현 한계** — 2-fixed positional(GPU/CPU), partition format 변환 미구현 leaf(slot.rs:226), Skip 미배선(slot.rs:276 bail). plan 이 초과 능력 요구 시 loud-fail.
- **호스트 GPU 부재** — weight swap/partition 검증은 S25/Jetson device 한정. baseline = legacy frozen.

---

## 참조

- **SSOT**: `docs/adr/0006-weight-stage-plan-returning-unification.md` (본 grill 7결정 + 검증 교정 정본)
- 선행: ADR-0003(정적 crate+linkme+force-link), ADR-0004(KVCacheStage plan-returning, **D1 변형=엔진 독점**), ADR-0005(D6 3축 평행 registry, §8 hotpath, §6 weight 축 deferred 여지)
- arch: `arch/pipeline_stage_design_v2.md` §4.2(WeightFormat)/§5(PipelineStage)/§8(INV-HOTPATH). `/CONTEXT.md`(weight swap=stage+format 합성 / partition=format×hardware 곱)
- 현 코드 앵커: `crates/technique-api/src/lib.rs`(KVCacheStage:214/StageCtx:87/KVCachePlan:200/registry:257), `engine/src/format/weight_format.rs`(WeightFormat:21/LayerDispatch:36/SliceSpec:47/apply_dispatch:25), `engine/src/models/weights/slot.rs`(impl WeightFormat:188/Full match:199/swap_weights:141/2-fixed:212/format-assert:226/Skip-bail:276), `engine/src/pressure/weights/decider.rs`(decide:119/compute_qcf_weight_swap:258/_internal:269/boundary:146), `engine/src/pressure/weights/{incremental_plan.rs:25,dynamic_k.rs,swap_executor.rs:498}`, `engine/src/pressure/weight_swap_handler.rs`(WeightSwapModelRef:33/execute_swap:90), `engine/src/pipeline.rs`(PipelineStage:165/StageLifecycle:94/LifecyclePhase:127), `shared/src/lib.rs`(EngineCommand: SwapWeights:209/LayerSkip:199/SetPartitionRatio:254, DtypeTag:12), `engine/src/pressure/eviction/stage_registry.rs`(EvictionPolicyAsStage:51/execute_kv_plan:88/self-test:325 — 거울 모델)
- AB-6(weight-swap 8종 SwapStage glue) / AB-4(tensor partition 동적 enable) = Seam B 실배선 task

---

## 미커밋 / 별개

- 이 세션은 ADR + README + handoff 만 추가(코드 변경 0). 검증 워크플로 산출은 `/tmp/.../tasks/w7xebuitn.output`(영속 아님).
- ADR-0005 세션의 미커밋 실험 아티팩트(`engine/microbench/score_readback.rs`, `/tmp/llm_rs2_poc_hook/`)는 여전히 보존/폐기 미결정 — 본 ADR 과 무관.
