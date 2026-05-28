# Backlog — 미배정 작업

> 역할이 배정되지 않은 작업 대기열. PM이 우선순위 판단 후 역할별 TODO로 이동.

---

## [ACTIVE Sprint] Qwen 2.5-1.5B Full Microbench Matrix (2026-05-28 진입)

- **Master TODO**: `.agent/todos/sprint_microbench_full_matrix_2026_05_28.md`
- **Entry handoff (P0 Architect)**: `.agent/todos/handoff_microbench_full_matrix_phaseP0_2026_05_28.md`
- **목표**: 13 op × 7 backend × 2 dtype = **182 cell** paper main evidence 통합 매트릭스 측정 (Galaxy S25 단일 디바이스)
- **선행 sprint**: μ-Q1 4-cell (`handoff_qnn_microbench_phase_e_complete_2026_05_26.md`, M3/M4/M6b/M7 GREEN)
- **Phase 분해**: P0 (Architect 디자인) → P1{a,b,c,d} 병렬 (Implementer/Senior) → P2 (Implementer, P0 후 시작) → P3 (Implementer) → P4 (Tester, 8~12 device hour)
- **확정 사항**:
  - Op 13개 (Tier A+B+D, Qwen 실사용): `MUL_MAT`, `RMS_NORM`, `ROPE`, `FLASH_ATTN_EXT`, `GET_ROWS`, `SILU`, `MUL`, `ADD`, `SOFT_MAX`, `SCALE`, `CPY`, `SET_ROWS`, `SWIGLU`
  - Backend 7개: ARM64Neon CPU / OpenCL GPU / ExecuTorch HTP NPU / Ours-NPU HTP FastRPC / L.cpp.CPU / L.cpp.GPU (Adreno) / L.cpp.HTP0 (Hexagon)
  - Dtype 2개: F16, Q4_0 (ExecuTorch Q4 = `use_8a4w` W4A8 단일)
  - Shape: Qwen 2.5-1.5B actual (hidden=1536, n_heads=12, n_kv=2, head_dim=128, FFN=8960, vocab=151936, n_layers=28)
- **Tier-A 우선순위 (hot path 8 op)**: MUL_MAT, RMS_NORM, ROPE, FLASH_ATTN_EXT, GET_ROWS, SILU, MUL, ADD

---

## [P2] LLama 3.2 1B 동일 shape 매트릭스 재측정 (Qwen Full Microbench Matrix sprint 완료 후)

- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: `sprint_microbench_full_matrix_2026_05_28` 종결 + P4 보고서 GREEN
- **Description**:
  - Qwen 2.5-1.5B Full Microbench Matrix sprint 완료 후, 동일 driver (`scripts/microbench_qnn_matrix.py` 확장본) + 동일 protocol로 LLama 3.2 1B 매트릭스 재측정
  - shape 매핑만 변경 (dim=2048, n_heads_q=32, n_kv_heads=8, head_dim=64, FFN=8192, vocab=128256, n_layers=16)
  - Q4_0 dtype은 동일, F16 dtype도 동일
  - 13 op × 7 backend × 2 dtype = 182 cell (Qwen과 동일 구조)
- **Acceptance Criteria**:
  - LLama 3.2 1B 매트릭스 `papers/eurosys2027/_workspace/experiment/microbench_full_matrix_llama321b_<DATE>/report.md`
  - Qwen 결과와 nominal shape 차이 분석 (head_dim=64 vs 128, FFN=8192 vs 8960 등이 latency에 미치는 영향)
- **Notes**: paper figure에서 2 model 비교 가능. driver 확장 0 (재사용).

## [P3] Tier-E 11 op 측정 (Qwen 미사용 op, 본 sprint 제외)

- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: `sprint_microbench_full_matrix_2026_05_28` driver (P3 확장) 완료
- **Description**:
  - Qwen 2.5-1.5B에서 사용되지 않는 11 op (Tier-E) 측정
  - 후보: `OUT_PROD`, `DIAG_MASK_INF`, `ARGMAX`, `ARGSORT`, `LEAKY_RELU`, `GELU`, `TANH`, `SQR`, `SQRT`, `SUM_ROWS`, `MEAN` (사전 조사 필요)
  - paper main evidence 아닌 supplementary table 용도
- **Acceptance Criteria**:
  - 11 op × 7 backend × 2 dtype = 154 cell 측정 완료 (UNSUPPORTED는 skip)
  - supplementary appendix table 생성
- **Notes**: paper main figure에 직접 인용 없음, reviewer rebuttal용. priority 낮음.

---

## [P2] typed lifecycle hook 확장 (h-1) — 별 sprint, 2026-05-27 등록
- **배경**: events sprint(2026-05-27)에서 사용자 결정 — sink가 fire-and-forget 디버깅 채널(비즈니스 영향 0)이라는 결론에 도달했으나, **비즈니스 동작(KV cache 관리, swap trigger)을 typed lifecycle hook으로 격리**하는 트랙은 별 sprint로 분리. 본 sprint는 events trait 인프라 제거(외과적)만 처리.
- **방향**: `LayerBoundaryHook` (Sprint C에서 L2 격상 `engine/src/layer_boundary_hook.rs`)이 이미 precedent. 추가로 typed hook을 다음 단위로 확장 가설:
  - `PressureHook` (신규) — 매 step pressure 체크 → eviction trigger (현재 `cache_manager.maybe_evict()` 직접 호출 패턴 격리)
  - `KvCacheHook` (신규) — KV append/grow/evict 직접 호출 추상화
  - `PrefetchHook` (신규) — 다음 layer/token weight 사전 로드
  - 기존 `LayerBoundaryHook` + `Sampler` + `StopChecker` + `ModelForward` + `DecodeWorkspace` + `Logits` (Phase 4-2/4-3 6 trait)와 일관 패턴
- **이득**: inference loop이 KV cache / swap_executor / cache_manager 모듈 직접 import 0으로 격리. 신규 도메인(speculative decoding, prefetch) 추가 시 inference loop 무변경.
- **위험**: silent drop이 비즈니스 영향 (eviction 안 일어남 → OOM, swap 안 일어남 → 정확도 저하). typed hook 패턴이면 컴파일 타임에 등록 누락 차단. event bus(callback registry)는 비추 — silent drop 위험.
- **블로커**: 본 트랙은 inference loop 전체 재설계 (Phase 4-4 후속) — 1~2주 범위. design round (Architect + 사용자) 필수.
- **연관**: `[P2] generate 바이너리 분할 + Manager 통합` (line ~71) — 함께 처리하면 자연. handoff `arch/inference_pipeline.md` precedent (Phase 4-2 4-3 6 trait + DecodeLoopBuilder typestate).

---

## [P3] §13.8-L hot path sub-trait 격상 — 잔여 7건 (2026-05-26 P2→P3 강등, 리뷰 결과)
- **Status**: TODO (P3) — 정책 RESOLVED + 1차 sprint(`b94e55ee`)로 강제 trigger 사유 해소. 잔여 marker는 정적, 새 추가 없으면 status quo로 충분.
- **강등 사유 (리뷰 결정 2026-05-26)**: §13.8-L 정책 자체가 RESOLVED (`ARCHITECTURE.md` line 2120 `§L … RESOLVED (2026-05-24, S-C5 확장)`) + S-L-1/2/3 1차 sprint 종결 (`b94e55ee`, hot path marker 9→7). 잔여 7건의 baseline 효과 0 + transitive drag(PlanExecutor 8+ method + nested 3 trait + OpenCL queue trait 노출)이 3~5일로 ROI 최저. 동시 backlog의 generate 분할 / transformer.rs ctor 본질 해소 트랙이 더 ROI 큼.
- **잔여 marker 7건 (handoff_inv_layer_L_hot_path_subtrait_2026_05_24.md §다음 작업 D)**:
  - **PlanExecutor 군 (3건)**: `models/transformer.rs:1479` (forward_into) / `:2492` (execute_plan) / `:2739` (KIVI plan execute). `FullKernelPlan` struct가 `OpenCLBackend` concrete를 직접 받음 — trait 추상화 시 nested trait 3개(Queue/GpuScoreAcc/Program) 동반 노출 필요. 가장 큰 산.
  - **flash_attention NEON fallback (1건)**: `layers/attention.rs:232`. 함수 안 backend downcast 분석 필요 (격상 가능성 미평가).
  - **kivi raw queue (3건)**: `pressure/kivi_cache.rs:1811` (flush_residual_gpu) / `:2108` (assemble_view_gpu) / 기타. `get_cl_mem` + raw `ocl::core::enqueue_write_buffer` 직접 호출 — trait method 추출 시 OpenCL queue trait surface 노출 → backend abstraction 약화.
- **재발동 트리거 조건** (자동 P2 승격):
  - 새 GPU backend(Vulkan/WebGPU 등)가 production 진지 검토 → PlanExecutor 한정 sprint 발동
  - hot path marker 7 → 10+ inflation
  - paper figure / spec 가독성 사유로 사용자 명시 우선순위 부여
- **연관**: 본 항목과 `[P3] KiviCache hot path downcast resolve` 항목은 통합 처리 가능 (잔여 위치 일부 중복).
- **참고**: `papers/eurosys2027/` 또는 conversation log의 §13.8-L 리뷰(2026-05-26, review 스킬 산출물)에 옵션 비교 + 리스크 + DA Pass + 권고 보존.

## [RESOLVED] §13.8-O cross-L3 vocabulary trait inversion — 2026-05-26 종결
- **Status**: RESOLVED — ARCHITECTURE.md `§O — Cross-L3 domain vocabulary zone: RESOLVED (2026-05-24, S-C3; Sprint C 2026-05-26 갱신)` (line ~2130).
- **선행 (역사)**: 2026-05-24 등록 시점에 D-1 sprint S-C3 후 marker 우회 + 본질 격상 backlog로 분리. 이후 3 갈래 본질 해소가 별 sprint 시리즈로 진행되어 RESOLVED.
- **3 갈래 해소 상태**:
  - **WeightSwapDispatch trait** (3건) — **RESOLVED (Sprint B + B-fixup + Sprint C, 2026-05-26)**:
    - ModelConfig 부분: `6dcba548` + `d78d3956` — `engine/src/model_config.rs` L2 직속 격상 + `from_gguf_metadata` → `models/loader/gguf.rs::parse_model_config` 이전.
    - Weight swap 부분: `5c698d79` — orchestrator 10 파일 (`swap_executor`/`async_swap`/`phase_aware_swap`/`intra_forward_swap`/`decider`/`incremental_plan`/`dynamic_k`/`probing_k`/`noise_table`/`release_worker`) `models/weights/` → `pressure/weights/` git mv + `LayerBoundaryHook` trait L2 격상(`engine/src/layer_boundary_hook.rs`). `weight_swap_handler.rs:22-25` LAYER-EXEMPT marker 2건 자연 해소.
  - **PrefetchAccess + PreloadPool L2 격상** (3건) — **PARTIAL RESOLVED**:
    - PreloadAccess trait inversion 적용 (`engine/src/pressure/offload/preload_pool.rs::pub trait PreloadAccess`, Sprint A `a9dcb5be` 시기).
    - 잔여: `models/transformer.rs:2807`의 `PrefetchController` import marker(`§13.8-O offload-path trait bound + PrefetchController (offload 분리 backlog)`). preload_pool/prefetch의 L2 또는 inference 도메인 격상 + `forward_into_offload` 분리는 별 sprint backlog (아래 [P2] offload 분리 항목 참조).
  - **KvCacheView trait** (3건) — **RESOLVED (`45bfd16f` B-5b-1b)**: `KVCacheOps` trait L2 격상 (`engine/src/kv_cache_ops.rs`), 외부 import path 일괄 재지향.
- **잔존 marker 의미**: 본 RESOLVED는 *위반 zone 정책*의 RESOLVED — register 갱신과 동기화된 정합 marker가 잔존. Sprint D(2026-05-26)로 transformer.rs ctor 호출 부분 추가 해소. 본질 해소(struct field trait inversion 등)는 별 sprint:
  - [RESOLVED Sprint D 2026-05-26] transformer.rs ctor 호출 위계 어긋남 본질 해소 (`arch/weights_pressure_split.md §7.4`) — `setup_runtime_resources` helper + `compute_quant_noise` 이전 완료. marker 순감 2건 (loader 2 + transformer helper 2 제거, use 위계 정합 2 추가). 잔여 field 정의 2건은 별 sprint(`RuntimeResourcesAccess` trait inversion).
  - [P2] SecondaryStore trait inversion (`arch/weights_pressure_split.md §7.5`) — V-09 `SecondaryMmapBytes` 패턴 확장.
  - [P2] observability events trait inversion (`arch/weights_pressure_split.md §7.1`) — Sprint C 잔여.
  - [P2] op_trace `DdrPhase`/`PhaseHook` L2 격상 (`arch/weights_pressure_split.md §7.2`).
  - [P2] offload 분리 (preload_pool/prefetch L2 격상 + `forward_into_offload` 분리) — 갈래 2 잔여.
  - [P3] `RuntimeResourcesAccess` trait inversion — `TransformerModel::quant_noise`/`release_worker` field를 `Arc<dyn ...>` trait object로 추상화 → field marker 2건 자연 해소. Sprint D 후속.
- **참고 register**:
  - `ARCHITECTURE.md` §13.8-O register V-24 (line 1755) — RESOLVED 상세 + register 갱신 이력 (line 1811).
  - `arch/weights_pressure_split.md` Sprint C design doc + §7 별 sprint backlog 5건.

---

## [DONE] swap 기본 모드를 `--swap-intra-forward`로 — 2026-05-25 종결
- **Status**: DONE (옵션 (b) `--swap` shorthand 채택, 2026-05-25)
- **선택 옵션**: (b) CLI 신규 `--swap [intra-forward|incremental|phase-aware|layer-immediate]`. flag 단독 시 default = IntraForward (LISWAP-4 production winner). 기존 4 flag는 deprecated + stderr 1회 경고 후 그대로 동작.
- **구현**: `engine/src/session/cli/mod.rs` — `SwapMode` enum + `parse_swap_mode` + `swap: Option<SwapMode>` field + `Args::normalize_swap_shorthand()`. caller (`argus_cli.rs`, `legacy/generate.rs`)에서 `Args::parse()` 직후 normalize 호출. Manager-driven path는 별도 (qcf_runtime.rs sync path 등 backlog P3 잔여).
- **검증**: cargo test --lib 1194 PASS, spec inv_layer 8 PASS, clippy --lib + main bins clean. S25 Adreno OpenCL Qwen2.5-1.5b Q4_0 32 토큰 baseline vs --swap intra-forward generation **bit-identical** (Paris... 동일).
- **잔여 (별 sprint)**: (a) Manager-driven swap path (executor.rs:549~) 자동 IntraForward 분기 — Manager 정책 layer 영향 큰 작업으로 분리. argus_cli v0 reject 함수의 swap 4 flag 차단 (v1-3 sprint 대상).

---

## [PARTIAL → CANCELLED] Phase 4-4-2.3 — decode_fallback 추출 — 2026-05-21 부분 완료 + 잔여 취소
- **Status**: 3a/3c/3b **완료** (master `02cb7106`). 3d/3e는 **취소** — 사용자 결정 (2026-05-21).
- **결정 사유**: generate.rs를 더 줄이지 않고 legacy로 보존, 새로운 다수 바이너리로 기능 분할하는 방향 전환. 잔여 3d/3e/4-4-2.4 추출은 새 바이너리 설계 안에서 자연 해소.
- **완료된 sub-sprint**:
  - 3a decode prologue (655 LOC, `session::decode_fallback::prologue`, commit `9313670b`)
  - 3c eviction trigger (200 LOC, `session::decode_fallback::eviction_trigger`, commit `bcb221e2`)
  - 3b swap dispatcher (452 LOC, `session::decode_fallback::swap_dispatch`, commit `02cb7106`)
- **취소된 sub-sprint**:
  - 3d Resilience checkpoint (~852 LOC)
  - 3e decode loop assembler (~447 LOC)
- **참고**: 분해 설계 `arch/decode_fallback_decomposition.md` (503 LOC, commit `391aa3f7`)은 향후 새 바이너리 분할 시 참고용으로 보존.
- **후속**: `[P2] generate 바이너리 분할 + Manager 통합` 항목으로 흡수.

## [CANCELLED] Phase 4-4-2.4 — post-process 추출 (~206 LOC) — 2026-05-21 취소
- **Status**: CANCELLED — 사용자 결정 (2026-05-21). Phase 4-4-2.3 잔여 취소와 함께 폐기.
- **사유**: generate.rs legacy 보존 방향 전환. 새 바이너리 분할 작업에서 자연 흡수.

## [P2] generate 바이너리 분할 + Manager 통합 — 2026-05-21 등록
- **Status**: TODO (사용자 결정 대기 — 분할 단위 + 진입 시점)
- **결정 (2026-05-21)**: 현 `engine/src/bin/generate.rs` (master `02cb7106`, 4,953 LOC)를 **legacy로 보존**하고, 새로운 다수 바이너리로 기능 분할. Manager IPC 통합도 새 바이너리에서 다룸.
- **배경**:
  - Phase 4-4-2.3 5 sub-sub-sprint 중 3a/3c/3b 추출 완료 — `session::decode_fallback::{prologue,eviction_trigger,swap_dispatch}` 모듈 자산 확보.
  - 단일 generate를 ≤400 LOC까지 줄이는 직선 추출 경로 대신, 기능별로 분할된 새 바이너리들이 공유 모듈을 사용하는 구조로 전환.
- **방향 (TBD)**: 추후 설계 라운드 필요. 잠정 후보 (사용자 결정 대기):
  - `gen-cli` — 단일 추론 (기본 모드, manager-free)
  - `gen-chat` — REPL 대화 모드
  - `gen-experiment` — 실험 측정 모드 (experiment_writer / schedule)
  - `gen-resilience` — Manager-integrated 모드 (CommandExecutor + Resilience checkpoint + 12+ plan dispatch)
- **선행**: 없음 (현 master에서 즉시 시작 가능)
- **영향 범위**:
  - `engine/src/bin/` — 신규 바이너리 N개 추가, legacy `generate` 보존
  - `engine/src/session/` — 공유 모듈 확장 (현재 init/cli/prefill/decode_fallback 활용)
  - `manager/` — IPC protocol + mock 시나리오 (gen-resilience 한정)
  - `Cargo.toml` — bin entry 추가
  - `scripts/run_device.py` — 다수 바이너리 배포 지원
- **재사용 가능 자산** (이미 분리됨):
  - `engine/src/session/init.rs` (Phase 4-1)
  - `engine/src/session/cli.rs` (Phase 4-1)
  - `engine/src/session/prefill.rs` (Phase 4-4-2.1)
  - `engine/src/session/decode_fallback/prologue.rs` (Sprint 3a)
  - `engine/src/session/decode_fallback/eviction_trigger.rs` (Sprint 3c)
  - `engine/src/session/decode_fallback/swap_dispatch.rs` (Sprint 3b)
  - `arch/inference_pipeline.md` (Phase 4 6 trait 설계)
  - `arch/decode_fallback_decomposition.md` (잔여 영역 참고용)
- **게이트 (예상)**: 각 신규 바이너리 bit-identical (legacy generate 동일 모드 대비) + avg_tbt 회귀 0
- **담당 권장**: Architect (설계) → Implementer (바이너리별 cut/paste) → Tester (디바이스 게이트)
- **상세 설계**: 진입 결정 시 별도 handoff 작성

---

## [RESOLVED] llm_rs2 lib clippy 회귀 — 2026-05-21 종결
- **Status**: RESOLVED — commit `0e406abc` (2026-05-21)
- **결과**: `cargo clippy -p llm_rs2 --lib -- -D warnings` 34건 → 0건. 게이트 PASS.
- **Mechanical 30건**: doc_list_item indent (12), needless_return (3), collapsible_if (3), needless_range_loop (1, cross-array index 본질적이라 함수 단위 #[allow]), useless_format (1), derivable_impls (1), is_multiple_of (1), let_chain (1), let_and_return (1), question_mark (2).
- **Design 4건**: `#[allow(...)]` silence (구조적 변경 필요) — not_unsafe_ptr_arg_deref (opencl::alloc_alias_weight_buffer), type_complexity (swap_executor::execute_on_slots, layer_importance::build_with_raws), large_enum_variant (ChatKvMode), too_many_arguments (run_qcf_warmup_workflow). 구조적 정리는 별도 sprint에서 검토.

## [RESOLVED — model issue, not code regression] Qwen2.5-1.5b chat 모드 garbage 출력 — 2026-05-19 확정

**원래 가설 4 확정**: `qwen2.5-1.5b` 변형이 **base 모델** (Instruct 아님). ChatML markers (`<|im_start|>`/`<|im_end|>`) 학습 없음.

**결정적 증거**:
- llama.cpp `llama-simple` (검증된 reference impl)에서 같은 q4_0 GGUF + ChatML prompt → 동일 garbage `撙\n撙\n撙...`
- raw prompt "The capital of France is" → 정상 "Paris. Paris is a very big city..."
- 즉 우리 코드 회귀 아님. **모델 학습 부족**.

**Ablation (S25 baseline binary)**:
- chat off → 정상 (E)
- greedy off (sampling on) → 정상 (D, noise로 일부 탈출)
- rep_penalty/system_prompt 단독 영향 없음

**해결책**: `Qwen2.5-1.5B-Instruct` variant 사용 (HuggingFace `Qwen/Qwen2.5-1.5B-Instruct`). **2026-05-19 S25 검증 PASS**: 같은 baseline binary + 같은 OpenCL path + 모델만 교체 → chat 1턴 "Paris is the capital of France." 정상 응답. non-chat sanity 30.36 ms/tok. GGUF: `models/qwen2.5-1.5b-instruct/qwen2.5-1.5b-instruct-q4_0.gguf` (1154.8 MB).

**config 차이 (base vs Instruct)**:
- base: `eos_token_id: 151643` (`<|endoftext|>`)
- Instruct (예상): `eos_token_id: 151645` (`<|im_end|>`)

**원래 가설 1/2/3 (template/encoding/KV sync)**: 반증.
- tokenizer.encode `add_special=False`에도 ChatML markers 정확히 single token (151644/151645) 확인 (호스트 python check)
- Phase 4-5-f chat repl v2 ↔ baseline chat repl 코드 path 동치 (PHB게이트에서 garbage 동등 재현)
- 4-5-g (`c1a4b481`)가 multi-turn KV pos 보존 fix 완료

**HEAD**: master `619dd655`


## [P0] M3.4 RED — pos baked architectural blocker — 2026-05-10
- **Status**: 사용자 architectural decision 대기
- **Handoff**: `.agent/todos/handoff_qnn_oppkg_m3_4_red_pos_baked_20260510.md`
- **상세**: `papers/eurosys2027/_workspace/experiment/m3_4_passgate.md`
- **HEAD**: `90617cc` (M3.4 14-node body + device gate RED, push됨)
- **요약**: graphFinalize 28× ~1.36s GREEN (예상 ~33s 대비 24× 빠름), prefill segfault. 후속 분석 결과 root cause는 M2.H builder가 pos를 `QNN_PARAMTYPE_SCALAR`로 graph build 시점 hardcoded — multi-token decode 불가능. 옵션: D-D (M2 ops 수정 +1.5주) / D-E (scope 약화 +0.5주). 사용자 결정 필요.

## [CLOSED] LISWAP-5 v2 — drop (2026-05-10, fair comparison + LISWAP-6 측정 후)
- **Status**: DROP (LISWAP-5 design 폐기 확정)
- **이유**: LISWAP-6 (DMA-BUF alias) 측정 결과 phase-aware는 alias 환경에서도 per-tick=25 대비 13~53× 손해. 분산 자체가 비효율적 — sub-chunking은 chunk 수 증가로 더 악화. 상세: `swap_overhead_liswap5_v1_postmortem.md` §7.7
- **확정 production path**: `--backend qnn_oppkg --swap-incremental-per-tick 25` (LISWAP-1 + LISWAP-6 alias 자동 활성)

## [P2] LISWAP-6 cleanup segfault — 2026-05-10
- **Status**: TODO (production 영향 없음, 측정 영향 없음 — 측정 깨끗하게 끝나려면 fix)
- **상세**: swap mode runs (`--swap-incremental-per-tick`, `--swap-phase-aware`) 에서 generation 정상 완료 후 process exit 시 SIGSEGV. baseline (no `--secondary-gguf`) 미발생. 모든 swap mode + `--backend qnn_oppkg` 조합에서 재현.
- **추정 원인**: `RpcmemAliasBuffer` Drop ordering 또는 `cl_mem` release ↔ `rpcmem_free` race. OpenCL queue/context teardown 순서.
- **참고 파일**: `engine/src/buffer/rpcmem_alias_buffer.rs`, `engine/src/models/weights/rpcmem_secondary.rs::RpcmemLayerRegion::Drop`
- **fix 방안 후보**: (a) explicit drop sequence (cl_mem all release → rpcmem_free), (b) reference count guard, (c) backend teardown 시 rpcmem region 명시 release

## [P3] KiviCache hot path downcast resolve — 2026-05-24 등록 (backend extension sprint 후속) / 2026-05-26 P2→P3 강등
- **Status**: TODO (P3) — §13.8-L 강등과 동기 (잔여 위치 일부 중복, 통합 처리 가능).
- **선행**: `.agent/todos/handoff_backend_extension_2026_05_24.md` (HEAD `5be6c0d7`)
- **상세 (위치 정정 2026-05-26)**: 원래 `kivi_cache.rs:1559/1842/2108` 3건으로 등록되었으나 **`:1559`는 S-L-3 sub-sprint(`dde575b9`)에서 KiviAttentionBackend trait 격상으로 이미 해소**. 잔여는 2건: `:1842` (flush_residual_gpu, raw `ocl::core::enqueue_write_buffer` 호출) / `:2108` (assemble_view_gpu, 동일 패턴). 두 위치 모두 OpenCL queue handle을 trait surface로 노출해야 하므로 §13.8-L 잔여 7건 중 kivi raw queue 군과 본질 동일 sprint.
- **제약**: `OpenCLBackend`는 Clone 미구현 + `Arc<dyn Backend>` → `Arc<OpenCLBackend>` Rust 기본 API 변환 불가. 단순 패턴 통일(`get_extension`)은 본 sprint 정책("hot path 보존")과 mismatch.
- **fix 방안 후보**:
  - (B) `OpenCLBackend` self-ref Arc — `Arc::new_cyclic` + `self_weak: Weak<Self>` field 추가, `OpenCLBackend::new()` signature를 `-> Result<Arc<Self>>`로 변경. KiviCache는 `Option<Arc<OpenCLBackend>>` 보유. 영향 범위 큼 (호출지 5+곳 마이그레이션). cuda/qnn backend도 같은 패턴 따라야 일관성. 추정 1~2일.
  - (B') KiviCache 한정 unsafe raw pointer 보존 — `gpu_opencl_ptr: Option<*const OpenCLBackend>` + invariant guard (Arc<dyn Backend> alive). `unsafe impl Send/Sync` 필요. 추정 0.5일.
  - (E) 새 trait `KiviGpuOps` inversion (RpoenCL impl) — signature 변경. 추정 1일.
- **실측 비용**: B-5b Phase 2 vtable Δ -0.231% / -0.018% 측정으로 string compare 비용도 ≈ 0 추정. **production 영향 매우 작음** (KIVI 모드 한정).
- **강등 사유 (2026-05-26 리뷰 결정)**: 위 `[P3] §13.8-L hot path sub-trait 격상` 강등과 같은 논리 — 1차 sub-trait sprint로 trigger 사유 해소, 잔여는 정적, transitive drag 큼, production 영향 미미.

## [DROP — 2026-05-24] CpuBackend 생성 책임 통일 (DI 강화) — 2026-05-24 등록 / 2026-05-24 종결
- **Status**: 부분 정리 완료 (3B Minimal, master `7df6c0b6`) — 나머지 작업은 ROI 0으로 평가, DROP 후보
- **2026-05-24 추가 측정 결과**: L4(`session/`) → L1(`backend::cpu`) import는 `scripts/layer_lint.py` invariant 미정의 영역 (INV-LAYER-001~005 모두 L4 source 규칙 없음). 본 entry의 "INV-LAYER-003 ~7~10건 해소" 추정은 잘못된 사전 측정이었음. 실제로는 trait signature 변경 옵션 (B/Hammered) 진행해도 baseline 0건 효과.
- **완료된 부분 (3B Minimal, commit `7df6c0b6`)**:
  - `session/batch/runner.rs:202/364` → `cpu_backend_arc.clone()` (alloc 2건 제거)
  - `session/qcf_runtime.rs:202/237/284/295/326` → `cpu_backend.clone()` (alloc 5건 제거 + `cpu_back: Arc<dyn Backend>` 중간 변수 2건 폐기)
  - `runner.rs` / `qcf_runtime.rs`의 `use crate::backend::cpu::CpuBackend;` import 2건 제거
  - 정성적 효과: 단일 Arc 공유로 메모리 footprint ↓ + production DI 강화
- **2026-05-24 S-2.5 sprint에서 GPU backend bootstrap 3건 추가 정리 (`9b8bcddd`)**:
  - `engine/src/backend/cpu/mod.rs::cpu_singleton()` (`OnceLock<Arc<dyn Backend>>`) free fn 추가.
  - cuda_embedded:530 / cuda_pc:328 / opencl:1540 의 `Arc::new(CpuBackend::new())` → `cpu_singleton()`. allocator + feature detect 3회 → 1회.
  - baseline 효과 0 (cross-backend import 패턴 동일) — 사용자 결정 옵션 B로 의식적 진행.
- **잔여 (의도적 보류)**:
  - `qcf_runtime.rs:881` (debug dump fallback) 1건 — 함수가 cpu_backend 미수신, signature 변경 비용 > ROI
  - `pressure/kivi_cache.rs:358` (`KiviCache::new()` default `Arc::new(CpuBackend::new())`) — 호출지 전원 test scope
  - test scope 40+ 호출지 — 의도된 test 패턴 (각 test 독립 객체)
- **DROP 사유**: production callsite (session/, GPU backend bootstrap 3건) 정리 완료. 잔여 callsite는 본질적으로 단독 진행 ROI 미만 (test scope 또는 sig 변경 cost > value).

## [P3] qnn_oppkg_poc clippy not_unsafe_ptr_arg_deref 15 errors — 2026-05-10 발견
- **Status**: TODO (M2 baseline부터 누적, M3.0 무관)
- **상세**: `cargo clippy --workspace --features opencl --tests -- -D warnings`에서 `crates/qnn_oppkg_poc/src/lib.rs:725` 근방 raw pointer deref 함수에 `unsafe` 누락. rust 1.93 신규 lint. M1 회귀 안전망 crate이라 P3 우선순위. M2가 main 진입한 이상 PoC는 read-only — 손대지 않거나 일괄 `#[allow(clippy::not_unsafe_ptr_arg_deref)]`로 silence.

## [P3] backend::opencl::* host test 24개 device-required fail — 2026-05-10 발견
- **Status**: TODO (호스트 측정 환경 한계)
- **상세**: `cargo test --workspace --features opencl --tests`에서 host에 OpenCL device 없을 때 24 fail (gpu_buffer_shift, kv_scatter_batch, noshuffle, plan tests). Galaxy S25 디바이스 빌드에선 정상. 호스트 회귀 게이트에선 본 모듈 제외 권장 — sanity-check skill에 `--exclude-tests backend::opencl` 패턴 추가 검토.

## [P2] Adreno noshuffle GEMV cross-run tuning (Phase 4-4.9/10 Path B) — 2026-05-18 등록 / 2026-05-18 갱신
- **Status**: TODO (Senior Implementer 위임 대기). Phase 4-4.10에서 default를 AOS로 invert하여 production은 회귀 없이 동작. Path B는 noshuffle SOA의 메모리 절약(≈702.8 MiB)을 회수하는 게 목표 (default를 다시 SOA로 되돌릴 수 있게).
- **재현 방법**: `LLMRS_ENABLE_NOSHUFFLE_SOA=1`로 SOA path 명시적 활성화 → G7' n=5 측정 → 4-4.7 baseline 32.06 ms 대비 Δ ≤ 5% 합격 시 default 재invert 후보.
- **상세**: 회귀 origin은 `kernel_gemv_noshuffle_q4_0` (plan path가 `make_q4_0_noshuffle_matmul_step`으로 직접 dispatch). Adreno 830에서 standard Q4_0 GEMV(`kernel_mul_mat_q4_0_f32`) 대비 m==1 디코드에서 ~4 ms/tok 느림 (4-4.10 measurement: 32.06 vs 36.44 median). 회수 후보 변형: LWS/SIMD 폭, image1d_buffer_t vs r32ui buffer, sub_group_reduce vs SLM tree-reduce — `feedback_adreno_subgroup_reduce.md` 원칙 준수. `feedback_cl_modification.md` 허용.
- **측정 지표**: G7' Δ ≤ 5% (4-4.7 post 32.06 ms baseline). G6' bit-identical 32 토큰.
- **참고 파일**:
  - `engine/kernels/` 하위 `kernel_gemv_noshuffle_q4_0` 정의 위치 (grep 필요)
  - `engine/src/backend/opencl/plan.rs::make_q4_0_noshuffle_matmul_step` (plan dispatch)
  - `engine/src/backend/opencl/mod.rs::matmul_q4_0_noshuffle` (non-plan fallback)
  - `papers/eurosys2027/_workspace/experiment/phase4_4_10_device_2026_05_18/measurement.md` (default invert 측정)
  - `papers/eurosys2027/_workspace/experiment/phase4_4_9_device_2026_05_18/measurement.md` (Path A env gate 측정)
- **참고 handoff**: `.agent/todos/handoff_phase4_4_entry_2026_05_17.md` Phase 4-4.10 종결.

---

# QNN-GPU OpPackage Migration — M2 (Layer-level Graph) — 2026-05-09 신규

> **상세 plan**: `.agent/todos/feat_qnn_oppkg_m2.md`
> M1 (production OpPackage crate, 5 ops) 완료. M2는 Qwen 1 layer (12-15 op)을 단일 OpPackage graph로 wrap.
> 5 신규 op (RoPE, DeqQ40, MatMulQ40F32, KvScatter, FlashAttn) + SiluMul OOP refactor + Layer graph builder + TBT 측정.
> Pass-gate: 1 layer accuracy max_abs_err < 1e-2, TBT ≤ baseline × 1.10, graphFinalize ≤ 200 ms, production code 변경 0.
> 추정: 18~22일 (병렬 가정, FlashAttn 디버깅 buffer 포함).

## [P0] M2.A — Layer op sequence 분석 + spec 갱신
- **Status**: TODO (Architect 위임 대기)
- **Sprint**: current
- **담당 권장**: Architect
- **상세**: `.agent/todos/feat_qnn_oppkg_m2.md` §1 M2.A

## [P0] M2.B/M2.C — RoPE / DeqQ40 op wrap (병렬 가능)
- **Status**: TODO
- **Sprint**: current
- **담당 권장**: Implementer (sonnet)
- **Dependencies**: M2.A
- **상세**: `.agent/todos/feat_qnn_oppkg_m2.md` §1 M2.B, M2.C

## [P0] M2.D — CustomMatMulQ40F32 op wrap (production hot path)
- **Status**: TODO
- **Sprint**: current
- **담당 권장**: Senior Implementer (Adreno + Q4_0 block)
- **Dependencies**: M2.C, M2.A
- **상세**: `.agent/todos/feat_qnn_oppkg_m2.md` §1 M2.D

## [P1] M2.E/M2.F — KvScatter / FlashAttn op wrap
- **Status**: TODO
- **Sprint**: current
- **담당 권장**: Implementer (sonnet) for E, Senior Implementer for F (online softmax + 32-float4 register)
- **Dependencies**: M2.A, M2.E (F의 의존)
- **상세**: `.agent/todos/feat_qnn_oppkg_m2.md` §1 M2.E, M2.F (위험 HIGH: FlashAttn)

## [P0] M2.G — SiluMul OOP refactor 결정 + 적용
- **Status**: TODO (Architect 옵션 결정 대기)
- **Sprint**: current
- **담당 권장**: Architect (옵션 결정) → Senior Implementer or Implementer (옵션별 적용)
- **결정 필요 항목**: `.agent/todos/feat_qnn_oppkg_m2.md` §4 (3 옵션 trade-off + escalate 질문 4건)

## [P1] M2.H — Layer graph builder
- **Status**: TODO
- **Sprint**: current
- **담당 권장**: Senior Implementer
- **Dependencies**: M2.B, M2.C, M2.D, M2.E, M2.F, M2.G individual GREEN 후
- **상세**: `.agent/todos/feat_qnn_oppkg_m2.md` §1 M2.H

## [P1] M2.I — Layer-level TBT 측정
- **Status**: TODO
- **Sprint**: current
- **담당 권장**: Tester
- **Dependencies**: M2.H GREEN
- **상세**: `.agent/todos/feat_qnn_oppkg_m2.md` §1 M2.I (성능 게이트, fail 시 §6 fallback)

## [P2] M2.J — Spec ID 추가 + 추적성 검증
- **Status**: TODO
- **Sprint**: current
- **담당 권장**: Architect (spec/INV) → Tester (spec test)
- **Dependencies**: M2.B~H 구현 완료, M2.I PASS
- **상세**: `.agent/todos/feat_qnn_oppkg_m2.md` §1 M2.J

---

# Weight Swap Overhead 감축 — 2026-05-07 신규 / 2026-05-21 post-paper 재분류

> 측정 보고: `/home/go/Workspace/papers/eurosys2027/_workspace/experiment/swap_overhead_s25.md`
> Galaxy S25 단발성 stall 1564.6 ms → 목표 ~70 ms (95.5% 감축).
> 6개 finding (A~F) + 보조 1개 (eager prefault). spec-manage가 부여할 ID 컨벤션: `WSWAP-6-A` ~ `WSWAP-6-F`, `WSWAP-6-PREFAULT`.
> **2026-05-21 갱신**: EuroSys 2027 paper 사이클 종결. paper deadline 압박 없음. 우선순위는 기술 부채 정리 가치로만 재판정 필요 (현재 P0/P1 표기는 paper-driven 잔여, 사후 컨텍스트에서 재평가 대상).
> 의존성: Finding E(stage label rename)는 다른 모든 작업 후 batch rename으로 처리.

## [P0] WSWAP-6-A: Fused SOA convert kernel (.cl `cvt_q4_0_noshuffle` 6 round-trip → 1 dispatch)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (independent)
- **Spec ID**: WSWAP-6-A (spec-manage가 부여 예정, cross-link)
- **Description**: |
  현재 AOS Q4_0 → Adreno SOA 재변환이 **layer당 6 GPU round-trip**을 거치며 swap stall의 48.5% (758.3 ms)를 차지.
  fused single-dispatch kernel `cvt_q4_0_noshuffle`로 교체하여 dispatch 오버헤드 + intermediate buffer 왕복 제거.
- **영향 파일** (예상 LOC 200-300):
  - `engine/kernels/cvt_q4_0_noshuffle.cl` (신규, ~150 LOC) — fused convert kernel
  - `engine/src/backend/opencl/weight_swap.rs` 또는 SOA convert path (수정, ~50 LOC) — dispatch 1회로 변경
  - `engine/kernels/` 기존 SOA convert 6-pass kernel은 deprecate 표시 후 추후 제거
- **검증 방법**:
  - GGUF spec test (AOS path 영향 없음 확인): `cargo test --workspace -- spec_weight_swap`
  - Galaxy S25 swap stall 실측: ratio=0.9 25 layers swap에서 `soa_reconvert` stage가 758 → ~100 ms 이하
  - 정확성: top-5 overlap > 99% vs 현행 SOA 출력 (token-by-token)
- **절감 추정**: 500-650 ms (전체 1564.6 ms의 32-42%)
- **위험**: low — AOS path는 영향받지 않음. SOA path만 변경.
- **담당 권장**: Senior Implementer (`.cl` 커널 + Adreno 최적화)
- **작성일**: 2026-05-07

---

## [P0] WSWAP-6-C: Primary cl_mem release를 critical path에서 제거 (mpsc + bg worker)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (independent)
- **Spec ID**: WSWAP-6-C
- **Description**: |
  현재 swap 시점에서 primary F16 cl_mem release(`madvise_ms` 단계로 잘못 명명됨, 실제로는 cl_mem release)가 **동기적**으로 critical path에 포함되어 173 ms 소비.
  mpsc channel + 별도 worker thread로 release를 비동기화하여 swap stall에서 제거.
- **영향 파일** (예상 LOC 100-150):
  - `engine/src/backend/opencl/weight_swap.rs` 또는 swap path (수정, ~80 LOC) — release call → mpsc::Sender::send
  - `engine/src/backend/opencl/mod.rs` 또는 신규 `release_worker.rs` (~50 LOC) — mpsc::Receiver + drop loop thread
- **검증 방법**:
  - swap stall에서 madvise_ms(현 명명) stage 비용 173 → ~0 ms (release deferred)
  - 메모리 회수: swap 후 일정 시간(<1s) 내 primary cl_mem이 실제 release되어 PSS 감소 확인 (procrank)
  - Crash 안전성: bg thread가 swap 중 panic 시 main thread 진행 보장 테스트
- **절감 추정**: 173 ms (전체 11.1%)
- **위험**: low — release 자체는 background, swap 정확성에 영향 없음
- **담당 권장**: Implementer (Rust 동기화 패턴)
- **작성일**: 2026-05-07

---

## [P1] WSWAP-6-F: `enqueue_write_buffer(blocking=true)` → async + 1회 finish (`alloc_and_upload_soa_buffers`)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (independent, A와 병행 가능)
- **Spec ID**: WSWAP-6-F
- **Description**: |
  현재 `alloc_and_upload_soa_buffers`에서 layer마다 `enqueue_write_buffer(blocking=true)`로 host→GPU 전송이 **layer 수만큼 직렬화**됨. blocking=false로 enqueue 모두 완료 후 한 번 `clFinish`/`synchronize()`로 합치면 driver pipelining 활성화.
- **영향 파일** (예상 LOC 50-80):
  - `engine/src/backend/opencl/weight_swap.rs` 또는 SOA upload path (수정, ~30 LOC) — write_buffer 호출 blocking 플래그 변경
  - 마지막 일괄 sync 1회 추가
- **검증 방법**:
  - SOA upload phase wall-clock 100-150 ms 감소 (S25 25 layers 기준)
  - 정확성: 모든 layer weight가 GPU에 정상 도착 (top-5 overlap > 99%)
  - thread safety: write enqueue 순서 보존 검증 (in-order queue 사용 가정 확인)
- **절감 추정**: 100-150 ms
- **위험**: low — async write는 OpenCL spec 표준 동작. 단, queue가 out-of-order면 검토 필요.
- **담당 권장**: Implementer (OpenCL API)
- **작성일**: 2026-05-07

---

## [P1] WSWAP-6-B: AOS path heap copy 제거 (`SharedBuffer::from_vec(data.to_vec())` → BorrowedBuffer)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (independent)
- **Spec ID**: WSWAP-6-B
- **Description**: |
  AOS swap path에서 mmap 영역의 `&[u8]` slice를 `data.to_vec()`로 heap에 복사하여 `SharedBuffer::from_vec`에 전달 → 1.2 GB 모델 기준 **80-100 ms heap allocation + memcpy**.
  `BorrowedBuffer` 또는 직접 `copy_weight_from(mmap_slice)` 경로로 변경하여 zero-copy.
- **영향 파일** (예상 LOC 80-120):
  - `engine/src/loader/auf.rs` 또는 secondary AOS load path (수정, ~50 LOC)
  - `engine/src/core/buffer.rs` 또는 `SharedBuffer` 관련 (수정, ~30 LOC, 필요 시 BorrowedBuffer 도입)
- **검증 방법**:
  - AOS swap path stall에서 heap copy 비용 80-100 ms 감소
  - mmap lifetime 보장: `LayerSlot` 또는 `SecondaryMmap`이 swap 완료 후 release되지 않도록 ownership 검증
  - spec_weight_swap AOS variant 테스트 PASS
- **절감 추정**: 80-100 ms
- **위험**: medium — buffer lifetime 관리. mmap이 cl_mem보다 먼저 drop되면 GPU read-after-free
- **담당 권장**: Implementer (Rust ownership 신중)
- **작성일**: 2026-05-07

---

## [P2] WSWAP-6-D: Prefault 범위를 target_layers byte range로 축소 (현재 28 layer 전체)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: WSWAP-6-PREFAULT 결정 후 (eager prefault 적용 여부에 따라 prefault 코드 위치가 달라짐)
- **Spec ID**: WSWAP-6-D
- **Description**: |
  현재 prefault(`MADV_WILLNEED` + page touch)가 **28 layer 전체**에 대해 수행. swap 대상은 일반적으로 ratio=0.9에서 25 layer. 비-target layer 3개의 prefault는 낭비.
  `--swap-ratio`로 결정된 target_layers의 byte range만 prefault.
- **영향 파일** (예상 LOC 30-50):
  - `engine/src/loader/auf.rs` 또는 prefault path (수정, ~30 LOC) — target_layers iter로 range 계산
  - `WeightSwapDecider` 또는 plan path에서 target byte range 전달
- **검증 방법**:
  - prefault stage 비용 328 → ~290 ms (약 40 ms 감소, 25/28 비율)
  - non-target layer 3개의 PSS 증가 없음 (procrank 비교)
- **절감 추정**: 40 ms
- **위험**: low — 범위 축소 단순 변경
- **담당 권장**: Implementer
- **작성일**: 2026-05-07

---

## [P2] WSWAP-6-E: Stage label rename `madvise_ms` → `primary_release_ms` (engine + shared IPC + manager)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: WSWAP-6-A, B, C, D, F 모두 완료 후 (rename batch)
- **Spec ID**: WSWAP-6-E
- **Description**: |
  현재 `madvise_ms` stage 라벨은 실제 측정 내용(primary cl_mem release)과 의미 불일치. 측정 보고서·매니저 trace·spec 모두에 잘못된 의미가 전파됨.
  cross-crate (engine + shared IPC + manager) 일괄 rename.
- **영향 파일** (예상 LOC 50-80):
  - `engine/src/backend/opencl/weight_swap.rs` 또는 trace 라벨 정의 (수정, ~10 LOC)
  - `shared/src/lib.rs` IPC 메시지 schema (수정, ~10 LOC) — back-compat 필요 시 양쪽 alias 허용 후 단계적 제거
  - `manager/src/` trace 파서/lua context (~20 LOC)
  - `policy_*.lua` (해당 라벨 사용 시, ~10 LOC)
  - `docs/`, spec 테스트 (~20 LOC)
- **검증 방법**:
  - manager trace_level=2 출력에서 `primary_release_ms` 라벨 확인
  - shared IPC schema test PASS
  - policy_default.lua / policy_s25_unified.lua가 새 라벨로 정상 동작
- **절감 추정**: 0 ms (정합성/가독성)
- **위험**: low — cross-crate change, IPC back-compat 1버전 유지 권장
- **담당 권장**: Implementer (cross-crate rename)
- **작성일**: 2026-05-07

---

## [P1] WSWAP-6-PREFAULT: Eager prefault at startup (doc 3.1) — Finding A/C와 결합 시 stall ≤70 ms 달성 핵심
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (independent, A/C와 병행)
- **Spec ID**: WSWAP-6-PREFAULT
- **Description**: |
  AUF mmap이 `--secondary-gguf` 등록 시점에 수행되지만 page는 swap fire 시점에야 fault-in 발생.
  모델 로딩 직후 `madvise(MADV_WILLNEED)` + manual page touch (또는 `mlock`) → swap 시점의 prefault·page-fault 비용 0.
  Finding A(soa kernel)+C(release async)와 결합 시 1564.6 → ~70 ms 도달 가능.
- **영향 파일** (예상 LOC 50-80):
  - `engine/src/loader/auf.rs` 또는 secondary mmap 등록 path (수정, ~40 LOC)
  - `engine/src/bin/generate.rs` CLI 옵션 `--eager-prefault` (선택적, ~10 LOC)
- **검증 방법**:
  - swap stall에서 prefault stage 328 → ~0 ms
  - 모델 로딩 시간 +500 ms 이내 (one-time cost)
  - PSS 증가 1.2 GB (AUF 전체 commit) — S25 12 GB RAM에서 허용
  - foreground memory pressure 시 동작 확인 (PSI 또는 매니저 신호)
- **절감 추정**: ~328 ms (prefault 단독). Finding A/C와 결합 시 stall ~70 ms 달성에 필수
- **위험**: low — page commit 증가하나 12 GB RAM 모델에 부담 작음. contention 환경에선 PSI 가드 추가 검토
- **담당 권장**: Implementer
- **작성일**: 2026-05-07
- **Note**: Finding A가 doc 3.2(SOA secondary AUF 회귀 패치, ★★)를 대체하므로 doc 3.2는 별도 backlog 등록하지 않음. doc 3.1은 본 항목으로 등록.

---

## [P2] QCF 명명 컨벤션 정리 — `QCF_kv` / `QCF_weight` 2-tier rename
- **Status**: RESOLVED (2026-05-19, Sprint 2 완료)
- **Sprint**: completed
- **Dependencies**: 없음
- **Description**: 현재 `unified_qcf`(KV 액션 5종)와 `compute_qcf_swap`(weight)이 이름상 같은 QCF지만 측정 공간이 다르다(`‖ΔO‖/‖O‖` vs `Σ imp×ε / Σ`). 통일이 불가하다고 판단되어 **이름으로 패밀리를 분리**.
- **적용 결과 (2026-05-19, Sprint 2 commit)**:
  - `core/qcf/unified_qcf.rs` → `core/qcf/qcf_kv.rs` rename
  - `UnifiedQcfParams` → `QcfKvParams`, `compute_unified_qcf` → `compute_qcf_kv`
  - `ImportanceTable::compute_qcf` → `compute_qcf_weight` (skip 패밀리)
  - `decider::compute_qcf_swap` → `compute_qcf_weight_swap` (swap 패밀리, internal helper 포함)
  - `docs/qcf_taxonomy.md` 갱신 (파일 경로 + 함수명 모두 신규명 반영)
  - `engine/tests/spec/inv_layer_baseline.json` 의 unified_qcf.rs → qcf_kv.rs, compute_qcf_swap → compute_qcf_weight_swap 동기화
  - 호출처 일괄 sed: `eval/eviction_hook.rs`, `eval/eval_loop.rs`, `bin/generate.rs`, `session/{ppl/runner,qcf_runtime}.rs`, `models/weights/{decider,mod}.rs`, `profile/quality_metrics.rs`, spec test 5개 파일
- **Deferred (별 backlog)**:
  - ~~`shared::QcfEstimate.estimates` HashMap key — IPC 메시지라 manager 동시 갱신 필요~~ — **2026-05-21 완료** (commit `3d4e1a09`, action name + IPC wire format 모두 dot prefix 통일).
  - `DegradationEstimator` default curves key (`eviction`, `sliding`, `kivi`, `swift` — `engine/src/qcf/estimator.rs:71-75`) → `kv.eviction` / `kv.sliding` / `kv.kivi` / `weight.skip` — 추상 카테고리 명명, 다른 의미 layer라 별도 정리. 본 sprint 범위 밖.
- **검증**: `cargo build --release -p llm_rs2` PASS, lib 160 qcf-related test PASS, spec 660/663 PASS (3 fail = 사전 device-required).
- **작성일**: 2026-04-27
- **종결일**: 2026-05-19 (taxonomy rename) / 2026-05-21 (IPC dot prefix 후속)

---

## [P1] QCF_kv 측정의 layer-0 단일 proxy → 모든 layer aggregate
- **Status**: CANCELLED (2026-04-27, layer 0 proxy 유지로 결정)
- **Sprint**: backlog
- **Dependencies**: —
- **Description**: |
  KV eviction 4종(`kv_evict_sliding`, `kv_evict_h2o`, `kv_evict_streaming`, `kv_merge_d2o`)이 `compute_qcf_estimates`(`engine/src/bin/generate.rs:6458`)에서 **layer 0의 KV cache 1개만** 측정하여 액션 대표값으로 보고한다(`let cache = &ctx.kv_caches[0];`, line 6485).

  ### CANCELLED 사유 (2026-04-27)
  - 본 layer 0 측정은 **ad-hoc 경량 proxy**로 의도된 설계임을 확인. dry-run "estimate" 명명대로 매니저 정책 입력값으로서의 **상대 ordering**이 핵심이고, layer 0 proxy로도 그 ordering이 보존됨이 실측에서 검증되었음.
  - 모든 layer aggregate 시 `compute_unified_qcf` 비용 ×N_layer (1B 16-layer = 16배), signal path 응답 지연 영향 — proxy 정확도 향상 대비 비용이 정당화되지 않음.
  - 따라서 코드 수정은 진행하지 않고 layer 0 proxy를 유지. 본 backlog 항목은 CANCELLED로 보존하여 결정 추적성 확보.

  ### 후속 작업 (수정 대신 문서화)
  - `docs/qcf_taxonomy.md` §2.3에 "Layer 차원 처리" 행 추가 — KV eviction 4종 = layer 0 ad-hoc proxy(의도된 단순화), KIVI = 모든 layer 평균, swap/skip = ImportanceTable 전체 sum.
  - Figure 1 캡션 또는 본문에 "(*) KV eviction 4종은 layer 0 proxy로 측정 — ad-hoc 경량 proxy, ordering 보존 실효성 검증됨" 각주.
- **작성일**: 2026-04-27
- **종결일**: 2026-04-27

---

## [P0] Weight Swap — Layer-Level Mixed Precision & Dynamic Swap
- **Status**: TODO (Architect 판단 완료, Phase 분해 완료, 구현 대기)
- **Sprint**: current
- **Dependencies**: 없음 (Phase A 즉시 착수 가능)
- **Description**: 메모리 극한 환경(Android 모바일)에서 layer별 dtype 혼용(F16/Q4_0) 및 동적 weight swap을 통해 PSS 감소. GGUF 두 벌 기반 설계 확정(커스텀 포맷 기각). Phase A(정적 mixed precision) → Phase B(동적 pressure-driven swap) → Phase C(커스텀 포맷, 연기) 순차 진행.
- **Acceptance Criteria**:
  - Phase A: Galaxy S25에서 PSS 100–150 MB 감소, tok/s 열화 < 5%, top-5 overlap > 95%
  - Phase B: swap latency < 50 ms/layer, 동적 PSS 감소 확인, ROUGE-L > 0.8 vs F16-only
- **상세 계획**: `.agent/todos/feat_weight_swap.md`
- **이번 스프린트 최우선**: (1) WSWAP-A1 GGUF 두 벌 생성, (2) WSWAP-A2 `LoadConfig::per_layer_dtype` spec+구조체, (3) WSWAP-A3 GgufLoader 오버레이
- **담당 권장**: Architect(spec) + Implementer(로더/CLI) + Senior Implementer(Phase B 리팩토링) + Tester(실측)
- **측정 환경**: Galaxy S25 / Snapdragon 8 Elite / Llama 3.2 1B (F16+Q4_0 GGUF) / 6 threads
- **작성일**: 2026-04-24

---

## [P0] Long context CPU attention 최적화 — 4K에서 llama.cpp 대비 35% 수준
- **Status**: TODO (설계+측정 완료, 구현 대기)
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: 4K context에서 llm_rs decode가 10.6 tok/s로 llama.cpp 30.5 tok/s의 35% 수준. Short context(~20)는 75%. 원인: standard 3-pass attention + head-parallel이 GQA 6:1에서 KV 중복 읽기(6배) + L2 thrash 유발. DRAM-bound가 되어 context 확장 시 급격 열화.
- **Acceptance Criteria**:
  - 4K context decode: 25+ tok/s (llama.cpp 대비 80% 이상)
  - Short context 회귀 없음 (22 tok/s 이상 유지)
  - 정확도 유지 (F16 NMSE < 1e-4, top-k match > 99%)
- **상세 계획**: `.agent/todos/long_context_attention_optimization.md`
- **구현 단계**: (1) Online Softmax (Step 1, 낮은 난이도) → (2) Flash Decoding KV split (Step 2, 중간 난이도, 메인 효과) → (3) CPU Flash Attention for prefill (Step 3, prefill O(n²) 해결)
- **주 수정 파일**: `engine/src/backend/cpu/neon.rs:235 attention_gen_f16_neon`
- **담당 권장**: senior-implementer (NEON + numerical algorithm)
- **측정 환경**: Galaxy S25 / Snapdragon 8 Elite / Qwen2.5 1.5B Q4_0 (`qwen2.5-1.5b-q4_0-v2.gguf`) / 6 threads
- **측정일**: 2026-04-13

---

## [P3] 다중 모델 사이즈 검증 테스트 매트릭스
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 다중 디바이스 포팅 완료
- **Description**: Llama 3.2 1B/3B 및 향후 7B/8B 모델에 대한 디바이스별 테스트 매트릭스 정의
- **Acceptance Criteria**: 매트릭스 문서, 디바이스별 최대 지원 모델 크기 명시
- **Notes**: 실제 테스트는 디바이스 확보 후 진행

## [P2] NVIDIA GPU OpenCL 추론 정확성 문제
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  NVIDIA RTX 3090 Ti에서 OpenCL 백엔드로 추론 시 garbage 출력 발생.
  개별 커널(rms_norm, F16 matmul, softmax, half 읽기)은 pyopencl 단위 테스트에서 정확하나,
  전체 추론 파이프라인에서 garbage 발생. Q4 weight + F32 KV cache에서도 동일 → F16 커널 무관.

  ### 조사된 사항 (2026-03-24)
  - fallback 커널 컴파일: F32, Q4_0, Simple Ops, F16 모두 nosub 컴파일 성공
  - PoCL CPU OpenCL: 정상 추론 (subgroup 지원 → 원본 커널 사용)
  - 개별 커널 정확성: rms_norm, matmul_f16 모두 pyopencl 테스트 통과
  - `CL_MEM_ALLOC_HOST_PTR` (UnifiedBuffer): NVIDIA discrete GPU에서의 동작 미검증
  - `unified_buffer::test_map_write_unmap_cycle`: 호스트에서 panic 발생 (기존 이슈)

  ### 의심 원인 (우선순위순)
  1. UnifiedBuffer + CL_MEM_ALLOC_HOST_PTR의 NVIDIA 호환성 (버퍼 동기화/매핑)
  2. 커널 간 데이터 전달 시 GPU↔Host 메모리 일관성 문제
  3. nosub 커널 내 미세 인자 불일치 (dispatch parameter vs kernel expectation)
- **Acceptance Criteria**: NVIDIA GPU에서 CPU 백엔드와 동일한 coherent 텍스트 생성
- **Notes**: |
  - 환경: NVIDIA RTX 3090 Ti, OpenCL 3.0 CUDA, cl_khr_subgroups 미지원
  - F16 nosub fallback 커널은 구현 완료 (17b2763)
  - 디버깅 접근: UnifiedBuffer를 비활성화(use_zero_copy=false)하여 discrete GPU용 버퍼 할당으로 전환 테스트 권장

## [P2] Gemma 3 1B NVIDIA GPU 추론 실패
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  Gemma 3 1B이 NVIDIA RTX 3090 Ti에서 `<unused6241>` 토큰만 생성.
  CPU에서는 정상 동작. Llama/Qwen은 NVIDIA에서도 정상.

  Gemma 3 특이사항:
  - head_dim=256 (Llama=64, Qwen=128)
  - `kernel_attn_gen_half`에서 `float out_local[256]` → 256 registers/thread
  - NVIDIA register limit (255) 초과 → spill to local memory → 가능한 정확성 문제
  - sliding_window=512 (로컬 어텐션)
  - gelu_pytorch_tanh 활성화

  회귀 테스트 baseline에서 확인됨 (735ba71).
- **Acceptance Criteria**: Gemma 3 1B NVIDIA GPU에서 coherent 텍스트 생성
- **Notes**: regression_test.py 3/3 FAIL (nvidia), 2/3 PASS (cpu)

## [P1] Manager ↔ Engine 프로토콜 이슈 (E2E 테스트에서 발견)
- **Status**: DONE
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  2026-03-24 E2E 테스트(Manager + mock_engine, Unix socket)에서 발견된 이슈 목록.

  ### 1. Relief Model cold-start 문제
  relief model이 없으면 모든 예측이 `ReliefVector::zero()` → ActionSelector가 액션을 선택 불가.
  **수정 방향**: ActionSelector에서 observation_count==0인 액션에 대해 domain 기반 default relief를 반환하는 fallback 추가.

  ### 2. `~` 경로 미확장
  `ReliefModelConfig::default()`의 `storage_dir: "~/.llm_rs/models"`가 셸 확장 안 됨.
  **수정 방향**: `dirs::home_dir()` 또는 `std::env::var("HOME")` 기반 절대 경로로 확장.

  ### 3. main config `[policy]` 섹션 미사용
  `Config`에 `policy: Option<PolicyConfig>` 필드가 있으나, `load_policy_config()`는 `--policy-config` CLI 플래그만 읽음. main config의 `[policy.*]` 섹션이 무시됨.
  **수정 방향**: `--policy-config` 미지정 시 `config.policy`를 fallback으로 사용.

  ### 4. 단방향 소켓 (Manager가 Engine 메시지를 읽지 않음)
  `UnixSocketEmitter`는 write-only. Engine이 보내는 Capability/Heartbeat/Response를 Manager가 수신하지 않음.
  프로토콜 스펙(docs/37)은 양방향이나 구현은 단방향.
  **수정 방향**: UnixSocketEmitter를 양방향 `UnixSocketTransport`로 리팩토링, reader 스레드 추가, Heartbeat → Pipeline의 `engine_state` 갱신.

- **Acceptance Criteria**: |
  1. Seed model 없이도 cold-start에서 directive 생성 가능
  2. `~` 경로가 올바르게 확장됨
  3. main config의 `[policy]` 섹션이 인식됨
  4. Manager가 Engine Heartbeat를 수신하여 pipeline engine_state 갱신
- **Notes**: |
  - 이슈 #1, #2, #3은 독립적으로 수정 가능 (각각 소규모)
  - 이슈 #4는 아키텍처 변경 (UnixSocketEmitter → 양방향 transport). 설계 검토 후 진행
  - E2E 테스트 커맨드: `manager --transport unix:<sock> --policy-config <toml>` + `mock_engine --socket <sock>`
  - emit_initial 프로토콜 불일치는 수정 완료 (7895824)

---

## [P2] policy_default.lua — action 계열 반복(연속 관측 실패 후 교체) 방지 논의 필요
- **Status**: TODO (논의 필요, 구현 방향 미결)
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: |
  external injection처럼 외부 압박이 지속되는 상황에서 kv_quant_dynamic을 반복 발동해도
  memory가 실제로 줄지 않으면 EWMA relief가 감소한다.
  relief argmax가 kv_evict_h2o로 교체되고, 그것도 관측 실패하면 다시 교체 → action 순환 발생.

  관측된 사례 (2026-04-15 시뮬레이션):
    t=1s  KvQuantDynamic(8) → obs#1 actual≈0 → EWMA 0.500→0.437
    t=4s  KvQuantDynamic(4) → obs#2 actual≈0 → EWMA 0.437→0.383
    t=7s  KvEvictH2o(0.5)   ← h2o prior(0.400) > quant learned(0.383)
    t=10s KvQuantDynamic(4) ← h2o obs#1 후 quant가 다시 승리

  논의 필요 사항:
    A. active guard 확장: observation window(3s) 외에 "최근 N번 관측이 모두 낮으면
       동일 계열 재선택 쿨다운" 추가 (ctx.history 활용 가능)
    B. 계열(category) 개념 도입: kv_evict_*, kv_quant_* 묶어 동일 계열 내 교체 억제
    C. 외부 압박 감지: memory가 지속 상승 중이면 relief 관측을 신뢰하지 않음 (p.memory
       slope from ctx.history 로 판단)
    D. 현행 유지: 실제 relief가 낮은 action을 자연스럽게 교체하는 것이 올바른 동작.
       production에서 injection은 없으므로 real relief signal이 정확할 것.

  주의: D 옵션이 맞을 수도 있음. 실기(Galaxy S25) 테스트 전에 변경 여부 결정 권장.
- **Acceptance Criteria**: 논의 후 방향 결정, policy_default.lua 또는 lua_policy.rs 수정
- **Notes**: |
  - 관련 파일: manager/scripts/policy_default.lua (is_active 가드, observation window)
  - ctx.history (ring buffer, 최근 10 tick) 이미 Lua에 노출됨 → slope 계산 가능
  - EwmaReliefTable α=0.875: 관측 2회 후 prior 대비 ~23% 감소 (external pressure 상황)

---

# Spec-Implementation Divergence (2026-03-31 조사)

> spec/에 정의되어 있지만 코드에 구현되지 않은 항목. 우선순위 순.

## [RESOLVED] QcfEstimate 메시지 + RequestQcf 커맨드 구현 — 2026-05-21 종결
- **Status**: RESOLVED — 코드 검증 결과 이미 구현됨 (sprint 2026-05-21에서 stale 확인).
- **Notes**: `shared/src/lib.rs:229,453` (EngineCommand::RequestQcf, EngineMessage::QcfEstimate), `engine/src/resilience/executor.rs:531,1346` (handler 구현). Spec `32-engine-algorithms.md:656`에서도 "구현 완료" 명시.

## [RESOLVED] Manager 페이로드 크기 가드 추가 — 2026-05-21 종결 (PROTO-012)
- **Status**: RESOLVED — commit `1e6e80f6` (2026-05-21)
- **결과**: `manager/src/channel/{unix_socket,tcp}.rs::read_engine_message()`에 MAX_PAYLOAD_SIZE=64KB 검증 추가. 초과 시 anyhow::bail!로 에러 + 연결 유지.

## [CANCELLED] Heartbeat/Response 타임아웃 구현 — 2026-05-21 종결 (spec과 backlog 충돌)
- **Status**: CANCELLED — spec이 "현재 적용하지 않음" 정의와 backlog 충돌. 2026-05-21 stale 확인.
- **Notes**: `spec/12-protocol-sequences.md:371,375` SEQ-087/088: "Manager는 특정 Heartbeat 주기를 가정하지 않는다" + "현재 Response 타임아웃을 적용하지 않는다" *(MUST)*. spec이 미적용을 정상 동작으로 정의. 구현 필요성 자체가 spec과 모순. 향후 spec 갱신이 선행돼야 backlog 재등록 가능.

## [P2] KvStreaming 커맨드 정상 구현
- **Status**: DONE (2026-03-31)
- **Sprint**: backlog
- **Notes**: cc0b9ce — EngineCommand::KvStreaming → StreamingLLMPolicy 연결 완료

## [P2] KvMergeD2o 액션 추가
- **Status**: DONE (2026-03-31)
- **Sprint**: backlog
- **Notes**: ffce391 — Pipeline 재활용 설계, D2OHandler 수정 0줄

## [P3] MergeHandler 정상 구현
- **Status**: CANCELLED (2026-03-31)
- **Notes**: D2OHandler가 cosine merge를 이미 수행. 기능 중복으로 stub 삭제 (7742543)

## [P3] SparseHandler 정상 구현
- **Status**: CANCELLED (2026-03-31)
- **Notes**: 1B+2048ctx 타겟에서 실익 없음. stub 삭제 (7742543)

## [P3] EnergyConstraint 스펙-코드 Divergence 해소
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: |
  MGR-ALG-015: 스펙은 raw battery_pct → 연속 pressure (m = clamp(1-pct/100, 0, 1) * 0.5).
  코드는 Level enum → 4단계 이산값 (Normal=0.0, Warning=0.55, Critical=0.80, Emergency=1.0).
  기능적으로 동작하지만 스펙과 다름.
- **Acceptance Criteria**: 스펙 수식대로 연속 변환하거나, 스펙을 현재 구현에 맞게 갱신
- **Notes**: 스펙 갱신이 더 현실적일 수 있음. Architect 판단 필요.

---

## [P3] ThermalCollector zone 패턴 매칭 auto-discovery
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 없음
- **Description**: 현재 `zone_types`는 exact match. substring/keyword 매칭으로 확장하면 다양한 장치 커버 가능
- **Acceptance Criteria**: contains 기반 패턴 매칭, 기존 exact match와 공존, 테스트 추가
- **Notes**: 필요성 미확정. 실제 다중 장치 배포 시점에 재평가

## [P1] Qwen CPU decode gap 해소 — matmul 외 원인 조사 필요
- **Status**: TODO (재정의됨)
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  Qwen 2.5-1.5B CPU decode가 llama.cpp CPU 대비 +14-15% 느린 gap이 남아 있음.

  ### 이미 시도한 것 (모두 실측 효과 없음)
  1. **Native F16 FMA 전환** (`a9cd3cc`, 2026-04-11): FMLAL → FMLA .8H inline asm.
     Short −0.5% / long +0.3% (mean) — noise 범위. Commit 유지됨 (cleanup 가치).
     분석: `results/data/flash_attn_decode/thermal/FMA_ANALYSIS.md`
  2. **vfmaq_f16 intrinsic 포팅** (branch `feat/f16-intrinsic-gemv`, 2026-04-11,
     revert): nightly toolchain + `stdarch_neon_f16`. Disassembly 상 main loop
     36→30 instructions, load-to-use 거리 2→3-5로 명확히 개선. 그러나 실측 short
     동등, long 데이터 부족, **prefill +15-20% regression**. Net negative → revert.
     분석: `results/data/flash_attn_decode/thermal/INTRINSIC_EXPERIMENT.md`

  ### 학습된 것
  - **Kernel-level instruction scheduling 최적화는 실측에 반영되지 않음**: S25에서
    FMA GEMV는 이미 memory subsystem ceiling에 가까움. Disassembly 개선이 runtime
    개선을 보장하지 않는다.
  - **Nightly toolchain 전환은 숨은 cost 있음**: 포팅 대상이 아닌 prefill 경로에서도
    regression 관찰됨. LLVM/codegen 차이가 전체 binary에 영향.
  - 과거 S24 `b25bc19` 교훈 재확인: "inner loop optimizations (multi-row, prefetch,
    stride) have no effect because the bottleneck is DRAM bandwidth".

  ### 이제 필요한 것: 진짜 병목 찾기
  Kernel 최적화 루트는 exhausted. 다음 접근:

  1. **Per-op 프로덕션 프로파일링** (--profile 없이). `simpleperf`, `perf`, 또는
     수동 timestamp로 token당 어디에 시간이 쓰이는지 정확히 측정. matmul_ffn 외
     candidate: RMSNorm, attention softmax, sampling, thread dispatch, SpinPool
     overhead. **이 정보 없이는 추가 최적화가 다 hunch-driven이다.**
  2. **Thread pool dispatch overhead 측정**: SpinPool 자체의 per-chunk cost.
     llama.cpp threadpool과 어느 정도 차이 나는지.
  3. **Chunk size A/B** (llama.cpp 64 rows/chunk vs 우리 140 rows/chunk). 1줄 변경,
     빠른 실험 가치 있음.
  4. **Big.LITTLE affinity**: Long decode가 bi-modal (±5.8% spread)인 이유가
     Oryon Phoenix L/M 스케줄링 jitter일 가능성. Gap 축소보다 variance 축소.
  5. **Single-asm super-block** (stable-friendly 최후 옵션): 4 rows 모두 한 asm
     블록 안에서 explicit interleaving. 학습 결과(1) 기반으로 ROI 낮아 보이지만
     theoretical latency-hiding 경로를 완전히 소진할 마지막 카드.

  **(1)이 blocker** — 병목이 어디인지 모른 채로 (2)-(5) 시도는 또 다른 neutral
  실험이 될 위험.
- **Acceptance Criteria**: |
  - 먼저 (1) 프로파일링으로 per-op breakdown 확보 → 보고
  - 그 다음 가장 큰 op을 타겟으로 실측 최적화
  - 최종 목표: CPU decode short ≤ llama.cpp + 5%, long ≤ llama.cpp + 5%
  - V10 strict thermal isolation 프로토콜로 검증
- **Notes**: |
  - **시작점은 kernel이 아니라 measurement**. hunch에 기반한 kernel 변경은 금지.
  - Quality/Correctness는 `--greedy` byte-identical test로 보호
  - branch `feat/f16-intrinsic-gemv` 유지 (미래 참고용, merge 안 됨)
  - Device backup `/data/local/tmp/generate.fma-asm.backup` 유지
