# llm_rs2 Architecture Documents

이 디렉토리는 spec/의 구현 상세를 기술한다. spec/과 1:1 대응하는 파일 구조를 따른다.

## 관계

```
spec/  → 불변. WHAT (무엇을 보장해야 하는가)
arch/  → 가변. HOW (코드에서 어떻게 구현하는가)  ← 이 디렉토리
```

## 파일 구조

| arch/ 파일 | 대응 spec/ | 내용 |
|-----------|-----------|------|
| `00-overview.md` | `spec/00-overview.md` | 크레이트 구조, 빌드 프로파일, Feature Gate |
| `01-architecture.md` | `spec/01-architecture.md` | 서브시스템-모듈 매핑, Transport 구현체, **§6 Layered Architecture Mapping** (SYS-100~105, INV-LAYER-001~005 구현 매핑) |
| `10-protocol.md` | `spec/10-protocol.md` | Wire format 구현, MAX_PAYLOAD_SIZE |
| `11-protocol-messages.md` | `spec/11-protocol-messages.md` | 메시지 타입 serde 매핑 |
| `12-protocol-sequences.md` | `spec/12-protocol-sequences.md` | 시퀀스 구현 진입점 |
| `20-manager.md` | `spec/20-manager.md` | Monitor/Policy/Emitter 코드 매핑 |
| `21-manager-state.md` | `spec/21-manager-state.md` | FSM 코드 위치, ThresholdEvaluator |
| `22-manager-algorithms.md` | `spec/22-manager-algorithms.md` | PI Controller, Supervisory, ActionSelector |
| `23-manager-data.md` | `spec/23-manager-data.md` | TOML 스키마, Config 키 |
| `30-engine.md` | `spec/30-engine.md` | 서브시스템 모듈, CLI 플래그, Transport |
| `31-engine-state.md` | `spec/31-engine-state.md` | FSM 코드, ExecutionPlan, EngineCommand |
| `32-engine-algorithms.md` | `spec/32-engine-algorithms.md` | Eviction, KIVI, QCF, Pipeline |
| `33-engine-data.md` | `spec/33-engine-data.md` | Trait 구현체, Buffer, DType |
| `40-cross-cutting.md` | `spec/40-cross-cutting.md` | Fail-safety, 로깅, 에러 전파 |
| `41-invariants.md` | `spec/41-invariants.md` | 65개 INV 구현 위치 마스터 테이블 |
| `inference_pipeline.md` §13 | (CLI 표면, spec ID 없음) | argus-eval bin (γ-3) 설계 — eval/ppl/dump/experiment 진입 + ScheduleCommandSource seam (✅ 구현 완료 2026-06-11, commit `33d5bc8f`+`11c8721f`+`2e53cf44`). 별도 `argus_eval_bin.md` 미신설 — §13 이 단일 설계 원본 |
| `50-test-tools.md` | `spec/50-test-tools.md` | mock_engine, mock_manager 구현 매핑 |

## 독립 설계 문서

spec/ 대응이 아닌 독립적인 feature 설계 문서:

| 파일 | 내용 |
|------|------|
| `tensor_partition.md` | CPU-GPU Cooperative Inference (Option B) — Tensor Partition 설계 |
| `plan_partition_integration.md` | Plan × Partition 협업 — GPU plan path가 partition 활성에서도 작동 |
| `cpu_flash_decoding.md` | CPU Attention KV-Split 병렬화 (Step 2 — `attention_gen_f16_neon`) 설계 |
| `clock_abstraction.md` | Manager `Clock` trait 추상화 (테스트 용이성, 시뮬레이터 시간 주입) |
| `action_constraints.md` | Manager `ConstraintRegistry` 설계 — EngineCommand 조합 제약의 다층 방어 (Lua + Rust pipeline) |
| `weight_swap.md` | Dynamic Weight Swap — ArcSwap snapshot, ratio_generation, Phase 1~3.6 통합 |
| `auf_format.md` | AUF (Argus Unified Format) v0.1 — self-contained 가중치 자산 (Phase 3.7b) |
| `rpcmem_allocator.md` | RpcmemAllocator — `libcdsprpc.so` 단일 책임 모듈 (Sprint 2a Phase 2, ENG-RPCMEM-010~013) |
| `opencl_backend.md` | OpenCLBackend `--opencl-rpcmem` wire-up + OpenCLMemory alloc_kv 분기 + RpcmemKvBuffer (Sprint 2a Phase 2, ENG-RPCMEM-020~024) |
| `precision_swap.md` | RpcmemSecondaryStore allocator routing (fn-pointer → `Arc<RpcmemAllocator>`, 2-tier backend lookup) (Sprint 2a Phase 2, ENG-RPCMEM-030~033) |
| `inference_pipeline.md` | **v1** — DecodeLoop SOLID 분해 + 빌더 (Phase 4-2/4-3/4-4-2.3 7-trait, 2026-05-16~25). Phase β 시점에 v2로 재작성 예정 (`pipeline_stage_design.md` 기준). |
| `pipeline_stage_design.md` | **v3 메인** — DecodeLoop 단일 `PipelineStage` trait + `LifecyclePhase` enum + entry point별 `PipelineRegistry` 패턴 (2026-05-27 23 라운드 grill, **2026-05-28 본 grill 갱신**: KvBundle/WeightBundle trait 폐기 + KVCacheLayer/WeightLayer trait + ctx 5→3 field, **post-grill review 2026-05-28** 1차: §2 다이어그램 화살표 정정 + concrete stages 위치 L4 session/ → L3 cross-cutting `engine/src/stages/{kv,weight,system}/` 이동 + §5.4 sub-structure + §13.4 후속 결정점 4건, **post-grill review 2026-05-28** 2차: 후속 결정 13 (PipelineDispatcher trait 유지) + 후속 결정 14 (BackendExtensions trait 폐기, ctx 3 → 2 field) + §13.5 sub-grill 3건 추가 (R5 #10/#11/#12), **2026-05-28~29 본 sub-grill 갱신 (3차)**: **§0 Executive Overview 신설** (외부 explorer 진입점) + 결정 #15 (3-tier Stage 패턴 Tier 1/2/3, StorageSpec / apply_storage / as_any 폐기) + 결정 #16 (Stage cardinality 자유 1/N/0) + 결정 #17 (Score 도메인 별 sprint, hot/cold path asymmetry intentional) + 결정 #18 (`KVCacheView::dtype()` 폐기) + **갈래 B 메타 결정 (Boundary 명시)** — PipelineStage 적용 범위 = KV/Weight state mutation 한정, cross-cutting 은 자기 패턴 인정 + §13.6 본 sub-grill 매트릭스 (결정 4 + 갈래 B + 미해결 Q-#1-3~#12 + 발견 모순 10건 + llm.npu / mllm-NPU 정합도 부록)). spec 대응: `spec/41-invariants.md` §3.28 (INV-DECODE-STAGE-001/004~007 + INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC + INV-KVCACHELAYER-PAIRED-KERNEL + INV-STAGE-LAYER-HANDLE; INV-DECODE-STAGE-002/003 폐기; INV-DECODE-STAGE-006/INV-STAGE-LAYER-HANDLE/INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC 본문 갱신; INV-LAYER-006 본문 갱신 (PipelineDispatcher 위치 L4 finalize); **INV-STAGE-MODULE-LOCATION 후보** — Phase α-W 등록). KV dispatch 정식 결정: `docs/adr/0001-kv-dispatch-paradigm.md`. |

> **QNN OpPackage cdylib (M1, 2026-05-09)**: 별도 arch 파일 없이 `arch/30-engine.md` §16에 매핑. cdylib(`crates/qnn_oppkg/`)는 engine/manager/shared와 cargo dependency edge를 형성하지 않는 외부 산출물(INV-151). spec 대응: `spec/30-engine.md` 부록 A (ENG-QNN-010~C04, INV-151~155).

> **Sprint 2a Phase 2 — RpcmemAllocator 분리 (2026-05-26)**: `--backend qnn_oppkg` 의 production-critical 가속 원천인 `libcdsprpc.so` 의존을 backend-agnostic 한 단일 책임 모듈로 격리. `--opencl-rpcmem` flag 활성 시 OpenCLBackend 가 KV cache + precision swap secondary 양쪽에서 동일 `Arc<RpcmemAllocator>` 를 사용. `libQnnGpu.so` / `libqnn_oppkg.so` dlopen 은 Sprint 2b 에서 backend 와 함께 제거. spec 대응: `spec/30-engine.md` 부록 E (ENG-RPCMEM-010~C04, INV-RPCMEM-001~008).

> **DecodeLoop v3 Hook Pattern (2026-05-27)**: v2 7-trait 설계의 잔여 문제(R-2 EvictionStage 시그니처 부족, R-5 ResilienceStage 신설 결정점, God Ctx, Manager IPC wiring 격차, 신규 책임 추가 비용)를 단일 `PipelineStage` trait + `LifecyclePhase` enum + entry point별 `PipelineRegistry` 패턴으로 재설계. 23 라운드 grill 결정. arch 대응: `pipeline_stage_design.md`. spec 대응: `spec/41-invariants.md` §3.28 (INV-DECODE-STAGE-001~007).

> **KvBundle/WeightBundle grill 종결 (2026-05-28)**: 2026-05-27 의 KvBundle 8 method / WeightBundle 10 method 시그니처 재검토. 12 결정 누적 — (1) KvBundle / WeightBundle trait 폐기, (2) KVCacheLayer / WeightLayer trait + interior mutability 채택 ((γ) 모델, LayerSlot::rcu_weights 자연 확장), (3) StageContext 5 field → 3 field 축소, (4) KV dispatch Generic → Trait object 전환 (`docs/adr/0001-kv-dispatch-paradigm.md` 정식 결정), (5) Sprint 분리: Phase α-W (2-3주) → ADR-0001 → Phase α-K (4-6주). 본 grill 후속 결정 12 건 + 신규 INV 3건 (PRIMITIVE-AGNOSTIC / PAIRED-KERNEL / STAGE-LAYER-HANDLE) + 폐기 INV 2건 (INV-DECODE-STAGE-002 KVBUNDLE-CONSISTENCY / 003 KVBUNDLE-SYNC). 진입: `.agent/todos/handoff_kv_weight_grill_2026_05_28.md`. 총 작업 기간 8~13주 → 12~19주.

> **post-grill review 2026-05-28 — 다이어그램 정정 + concrete stages 위치 이동**: 본 grill 종결 후 사용자 추가 review 결과 2건 정정 + 1건 위치 이동 + 4건 후속 결정점 분리. (1) `arch/pipeline_stage_design.md` §2 Mermaid 다이어그램 화살표 방향 정정 (`bext --> be` / `kvl -.-> kvcache` 2건 방향 반대 → impl 이 trait 을 implements 로 수정 + 화살표 marker convention 헤더 주석 명시), (2) **concrete PipelineStage 구현체 위치 L4 `session/` → L3 cross-cutting `engine/src/stages/{kv,weight,system}/`** (`core/` cross-cutting 패턴 정합 + 외부 기여자 discoverability + session/ god module 방지) — `PipelineRegistry` 는 L4 `session/` 유지 (entry point 별 build 객체), (3) §5.4 sub-structure 신설 (kv/ vs weight/ vs system/ 분류 규약), (4) §13.4 후속 결정점 4건: BackendExtensions 재설계 sub-grill (leaky abstraction `as_opencl_secondary()`, §13.8-O 위반 신호, Phase α-W 종료 후 sub-grill round) / stages/mod.rs 신규 stage 가이드 doc (Phase α-W detail) / system/ 명명 검토 (system vs misc vs dispatch) / **INV-STAGE-MODULE-LOCATION 후보 등록** (즉시 추가 X, Phase α-W stages/ 신설 commit 동시 등록 — 현 코드에 stages/ 부재로 vacuous truth 회피). 진입 handoff R5 #7~#9 + R6.

> **post-grill review 2026-05-28 2차 — 본 grill 후속 결정 13/14 누적**: 위 1차 review (다이어그램 정정 + stages 위치 이동) 위에 누적된 2차 review. (1) **결정 13 — `PipelineDispatcher` trait 유지**: 단일 impl 우려에도 trait 유지 확정. 근거: (a) deletion test — trait 폐기 시 DecodeLoop 이 concrete `Arc<PipelineRegistry>` 결합 → INV-LAYER-006 위반, (b) 본 프로젝트 mock 패턴 (mock_engine/mock_manager 정착) → 미래 impl ≥ 2 보장 (NoopDispatcher / TestMockDispatcher), (c) vtable cost ~3 μs/sec (noise 이하). 위치 (L2 vs L4) 만 별 sub-grill (handoff R5 #10). (2) **결정 14 — `BackendExtensions` trait 폐기**: §13.8-L Backend trait capability provider 패턴 (`backend.gpu_score_acc()`, `backend.as_kivi_attention()`) 중복 추상화 + leaky abstraction (`as_opencl_secondary()` backend variant 명명 leak). (γ) 정신 일관 — Stage 는 mechanism 모름, Layer impl 이 backend ref 보유 + capability 내부 호출. **`StageContext` 3 → 2 field** (`step` / `profiler` 만, `backend_ext` 폐기). KIVI 예시 교정 (`as_kivi_attention()` / `gpu_score_acc()` / `kivi_gather_update()` — 이전 `secondary_store_handle()` 부정확). Backend trait Stage 2 sprint `as_opencl_secondary()` (engine/src/backend.rs:1232-1255) 명명 정합 충돌 + Layer impl backend ref 보유 패턴 (full `Arc<dyn Backend>` vs ISP-split sub-trait) 2건 sub-grill 분리 (handoff R5 #11/#12). spec: `spec/41-invariants.md` §3.28 INV-DECODE-STAGE-006 본문 갱신 (3 → 2 field). 진입 handoff R5 #10~#12 + R6.

> **본 sub-grill 2026-05-28~29 — KVCacheLayer/WeightLayer 시그니처 detail 3차 review (4 결정 + 갈래 B 메타 결정 + §0 Executive Overview 신설)**: 본 grill (2026-05-28) 12 결정 + 후속 결정 13/14 위에 누적된 sub-grill (2026-05-28~29). KVCacheLayer / WeightLayer trait 시그니처 detail finalize 과정에서 발견 모순 10건 (apply_storage 의 mixed dtype 불가 / StorageSpec enum variant OCP 위반 / KVCacheView::dtype() mixed paradigm 답 모호 / EvictionStage score_accumulator hot/cold path 혼재 / OneShotKvQuantStage downcast 필요 / TierMoveStage cross-paradigm OCP / 임의 1-layer 가정 / D2O K read paradigm leak / "single pattern fits all" / KVCacheLayer::as_any() type safety 약화) 식별. 4 결정 + 갈래 B 메타 결정으로 해소: **결정 #15** (3-tier Stage 패턴 — Tier 1 Primitive-only `Arc<dyn>` / Tier 2 Paradigm-specific `Arc<ConcreteLayer>` downcast 0 / Tier 3 Cross-paradigm `Arc<dyn CapabilityTrait>` Stage 측 정의; 부산물 — `StorageSpec` trait 폐기, `apply_storage(spec)` method 폐기, `KVCacheLayer::as_any()` 부재), **결정 #16** (Stage cardinality 자유 1/N/0 layer; 임의 1-layer 가정 폐기), **결정 #17** (Score 도메인 별 sprint — EvictionHook 1:1 wrap + score_accumulator concrete type 보존, hot/cold path asymmetry intentional, 별 sprint refactor 갈래 4/7/2 추천 + 1/3/5/8 pre-rejected), **결정 #18** (`KVCacheView::dtype()` 폐기 — dtype 사용처 5건 모두 backend / layer 내부, Mixed paradigm 의미 모호). **메타 결정 갈래 B (Boundary 명시)**: PipelineStage 적용 범위 = KV/Weight state mutation 도메인 한정. Cross-cutting (score collection / dispatch / cross-paradigm policy / Backend capability) = 자기 패턴 인정. 발견된 asymmetry 는 도메인 본질의 정직한 표현 — intentional. **사용자 요청 — §0 Executive Overview 신설**: 외부 explorer 진입점, standalone 으로 본 sprint 전체 그림 파악 가능 (본 sprint 미션 1단락 + 18 결정 한 줄씩 + 갈래 B + 다이어그램 + 적용 범위 boundary 표 + 다음 sprint 분기 + §13.6 미해결 매트릭스 링크). arch: `pipeline_stage_design.md` §0 신설 + §3.5/3.6/5.2/5.3/5.4/13.1/16/16.3 갱신 + §13.6 본 sub-grill 매트릭스 신설 (결정 4 + 갈래 B + 미해결 Q-#1-3~#12 9건 + 발견 모순 10건 + llm.npu / mllm-NPU ASPLOS 2025 정합도 부록 — 본 grill 구조 forward-compat 보장, 흡수 안 함). spec: `spec/41-invariants.md` §3.28 변경 요약에 본 sub-grill 5 결정 추가 + INV-LAYER-006/INV-STAGE-LAYER-HANDLE/INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC 본문 갱신 (PipelineDispatcher 위치 L4 finalize / cardinality 자유 / Tier 1 한정 + dtype 노출 X). 진입: `.agent/todos/handoff_kv_weight_subgrill_2026_05_29.md` (신규).

## 규칙

- arch/ 파일에 독자적 요구사항 ID를 만들지 않는다. **ID 원천은 항상 spec/**.
- 코드 변경 시 관련 arch/ 파일을 갱신한다.
- 상세 규칙: `.claude/skills/spec-manage/SKILL.md` 참조.
