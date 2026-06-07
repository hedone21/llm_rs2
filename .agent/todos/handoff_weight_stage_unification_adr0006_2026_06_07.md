# Handoff: weight 축 stage 통일 설계 확정(ADR-0006) → crate 단계 구현

**작성**: 2026-06-07
**HEAD**: (이 커밋) `docs(adr): ADR-0006 weight 축 plan-returning WeightStage 통일`
**브랜치**: master
**작성자**: 메인 세션 (precision/weight plugin 통일 grill + 적대적 검증 6-agent 세션)

**다음 세션 진입 문장**: **"ADR-0006 MW-B — technique-api 에 WeightStage/WeightDispatchPlan/LayerDirective/WeightStageCtx + WEIGHT_STAGES registry 신설 (MW-A ✅ `e07da198`, Seam B 는 Phase β)"**

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

## 다음 작업 (crate 단계 구현 — Seam A 한정)

### 진행 현황 (2026-06-07 갱신)
- **MW-A ✅ DONE** (commit `e07da198`): `LayerDispatch`/`PartitionShare`/`DeviceTarget`(mirror)를 technique-api 로 이동. **의존 순서 교정** — `WeightDispatchPlan`(MW-B)이 `LayerDispatch`를 참조하므로 이 이동을 weight-타입 신설(MW-B)보다 **먼저** 함.
  - **설계 결정(option A, 사용자 승인)**: `SliceSpec.format`(per-slice 저장 dtype)은 plugin 표면에 넣지 **않는다**. 근거: format 은 plugin 결정이 아니라 executor 가 weight dtype 에서 파생(split byte layout 원천 `bytes_per_row`)하고, plugin 표면 규칙 `TensorDtype`(3종)로 좁히면 현 7-dtype partition 중 **Q4_1/Q8_0/BF16/U8 이 회귀**한다. → `SliceSpec` 폐기, api `PartitionShare{share, hardware}` 신설, format 은 executor-내부 전체 `DType` 유지, 기존 format 동등 assert 제거. (**handoff 원안 step 2 "SliceSpec engine→api 이동(format:TensorDtype)" 을 이 결정으로 대체.**) ADR-0006 D2/D7 직교 보존 + 회귀 0. 직교성 자체는 format 을 plugin 에 둬도 유지(다른 필드/trait)되나, 선택은 *회귀 방지* 논거로 갈렸음.
  - 게이트: build/clippy --workspace clean, lib 1238/0(무회귀; 1 flaky=`test_prune_prefix_calls_release_unused_pages` RSS 임계, 격리 통과), DeviceTarget drift round-trip + apply_dispatch Full/Partition 테스트 통과.

### 남은 MW (Seam A)
- **MW-B ✅ DONE** (commit `691064f0`): technique-api 에 weight stage 타입 신설 (KVCacheStage 동형) — `WeightStage`/`WeightDispatchPlan`/`LayerDirective`(precision⊥dispatch)/`WeightStageCtx`/`LayerMetricKind`/`WeightStageReg`/`WEIGHT_STAGES`/`find_weight_stage`/`registered_weight_names`/`WeightStageParams`. 게이트: technique-api 5/0(weight Dummy 등록·find·sugar + dyn-safe) + clippy --workspace clean. (release linkme 생존 = MW-C builtin+self-test 게이트.)
- **MW-C ⏸ BLOCKED (설계 결정 대기)** — **`WeightSwapDeciderAsStage` 어댑터**가 막혔다. api `WeightStageCtx`(D4: flat `layer_metric` &[f32] + `budget`=선해소 count)가 기존 `WeightSwapDecider`(`&dyn ImportanceLookup` entries-based + `&dyn QuantNoiseAccess` + `decide(ratio)` 내부 재해소)와 **3겹 불일치**: importance flat↔entries, noise flat↔as_slice(경미), budget-as-count ↔ decide의 ratio-내부해소(이중 해소 위험). **D1("ctx borrow 로 decider 생성") ↔ D4(flat ctx) 내부 충돌** + ADR이 명시한 "weight OCP 미실증(decider=engine-internal trait 3종 결합)"의 실체. 결정 fork: **(A) builtin 이 api ctx 소비(shim importance→ImportanceLookup/noise→QuantNoiseAccess + budget→ratio) → 표면 검증, decider 무변경 / (B) builtin 이 api ctx 우회(concrete ctx downcast/model refs) → ADR builtin-first 부합, 표면 미실증 유지 / (C) decider 를 flat &[f32]+count 로 리팩터 → 표면 자연 입력화, but 12 call site(9 test+3 prod) 침습**. 사용자 결정 대기.
- **MW-D** — **`WeightStageCtx` 엔진 impl**(`&TransformerModel` 위로) — budget 해소·`current_format`·`layer_metric`(QuantNoise=as_slice, Importance=SubLayer::Full 투영) → 검증: importance/noise 투영이 decider 직접 입력과 bit-identical
- **(Phase β 의존) Seam B 배선** — `PipelineRegistry` 구현 + decode loop → `WeightSwapStage`(OneShot) on_phase → `execute_weight_plan`(IncrementalSwapPlan/SwapExecutor). **AB-6 + Phase β 선행, 본 crate 단계 밖.**

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
