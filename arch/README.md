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
| `pipeline_stage_design.md` | **v3 메인** — DecodeLoop 단일 `PipelineStage` trait + `LifecyclePhase` enum + entry point별 `PipelineRegistry` 패턴 (2026-05-27 23 라운드 grill, **2026-05-28 본 grill 갱신**: KvBundle/WeightBundle trait 폐기 + KVCacheLayer/WeightLayer trait + ctx 5→3 field). spec 대응: `spec/41-invariants.md` §3.28 (INV-DECODE-STAGE-001/004~007 + INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC + INV-KVCACHELAYER-PAIRED-KERNEL + INV-STAGE-LAYER-HANDLE; INV-DECODE-STAGE-002/003 폐기). KV dispatch 정식 결정: `docs/adr/0001-kv-dispatch-paradigm.md`. |

> **QNN OpPackage cdylib (M1, 2026-05-09)**: 별도 arch 파일 없이 `arch/30-engine.md` §16에 매핑. cdylib(`crates/qnn_oppkg/`)는 engine/manager/shared와 cargo dependency edge를 형성하지 않는 외부 산출물(INV-151). spec 대응: `spec/30-engine.md` 부록 A (ENG-QNN-010~C04, INV-151~155).

> **Sprint 2a Phase 2 — RpcmemAllocator 분리 (2026-05-26)**: `--backend qnn_oppkg` 의 production-critical 가속 원천인 `libcdsprpc.so` 의존을 backend-agnostic 한 단일 책임 모듈로 격리. `--opencl-rpcmem` flag 활성 시 OpenCLBackend 가 KV cache + precision swap secondary 양쪽에서 동일 `Arc<RpcmemAllocator>` 를 사용. `libQnnGpu.so` / `libqnn_oppkg.so` dlopen 은 Sprint 2b 에서 backend 와 함께 제거. spec 대응: `spec/30-engine.md` 부록 E (ENG-RPCMEM-010~C04, INV-RPCMEM-001~008).

> **DecodeLoop v3 Hook Pattern (2026-05-27)**: v2 7-trait 설계의 잔여 문제(R-2 EvictionStage 시그니처 부족, R-5 ResilienceStage 신설 결정점, God Ctx, Manager IPC wiring 격차, 신규 책임 추가 비용)를 단일 `PipelineStage` trait + `LifecyclePhase` enum + entry point별 `PipelineRegistry` 패턴으로 재설계. 23 라운드 grill 결정. arch 대응: `pipeline_stage_design.md`. spec 대응: `spec/41-invariants.md` §3.28 (INV-DECODE-STAGE-001~007).

> **KvBundle/WeightBundle grill 종결 (2026-05-28)**: 2026-05-27 의 KvBundle 8 method / WeightBundle 10 method 시그니처 재검토. 12 결정 누적 — (1) KvBundle / WeightBundle trait 폐기, (2) KVCacheLayer / WeightLayer trait + interior mutability 채택 ((γ) 모델, LayerSlot::rcu_weights 자연 확장), (3) StageContext 5 field → 3 field 축소, (4) KV dispatch Generic → Trait object 전환 (`docs/adr/0001-kv-dispatch-paradigm.md` 정식 결정), (5) Sprint 분리: Phase α-W (2-3주) → ADR-0001 → Phase α-K (4-6주). 본 grill 후속 결정 12 건 + 신규 INV 3건 (PRIMITIVE-AGNOSTIC / PAIRED-KERNEL / STAGE-LAYER-HANDLE) + 폐기 INV 2건 (INV-DECODE-STAGE-002 KVBUNDLE-CONSISTENCY / 003 KVBUNDLE-SYNC). 진입: `.agent/todos/handoff_kv_weight_grill_2026_05_28.md`. 총 작업 기간 8~13주 → 12~19주.

## 규칙

- arch/ 파일에 독자적 요구사항 ID를 만들지 않는다. **ID 원천은 항상 spec/**.
- 코드 변경 시 관련 arch/ 파일을 갱신한다.
- 상세 규칙: `.claude/skills/spec-manage/SKILL.md` 참조.
