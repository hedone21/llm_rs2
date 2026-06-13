# Backlog — 미배정 작업

> 역할이 배정되지 않은 작업 대기열. PM이 우선순위 판단 후 역할별 TODO로 이동.

---

## [ACTIVE Sprint — 사용자 주도 트랙, Backlog Burndown 범위 밖] Qwen 2.5-1.5B Full Microbench Matrix (2026-05-28 진입)

> **2026-06-12 주석 (Backlog Burndown triage)**: 위임 결정 A1(paper/측정 트랙 제외)에 따라 본 sprint는 **사용자 주도 트랙**으로 Backlog Burndown 자율 실행 범위 밖. Status/우선순위 미변경 — 사용자가 직접 운영. burndown 트랙이 건드리지 않음.

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

## [RESOLVED Roadmap] Phase γ 종결 — `pressure/`→`kv/`·`weight/` rename + sweep + bin화 + orphan 처분 (γ-1~γ-4, 2026-06-10~11)

- **Status**: RESOLVED — γ-1~γ-4 전 substep 종결 (2026-06-10~11).
- **SSOT (γ 정의 출처)**: `arch/pipeline_stage_design_v2.md` §9 "Phase γ 재정의"(G2-(iii) 승인, 2026-06-10). 직전 γ 정의("legacy generate.rs 잔여 마이그레이션 + PACT2026 PoC")는 α-K BC 결정(2026-06-04, §9.1 BC)에서 흡수 + β-7 v1 표면 삭제(2026-06-10)로 빈 껍데기가 되어 재정의.
- **선행 완료**: **Phase α-K BC 완주**(2026-06-05 — `KVCacheOps` trait 완전 폐기 + legacy generate.rs 폐기) + **Phase β 전 substep 종결**(β-1~β-7, 2026-06-10, HEAD `4abab582` — decode loop rewrite + v1 trait 표면 삭제).
- **γ 4 substep (전부 완료)** (arch §9 재정의):
  - **(γ-1) ✅** `kv/`·`weight/` rename — `pressure/`→`kv/`, `pressure/weights/`→`weight/`(commit `7fe1fe8b`). 파생 후속 3건 별도 등록(아래 "γ-1 파생 후속" 그룹, spec L3 도메인 재정의 RESOLVED).
  - **(γ-2) ✅** nested `mod.rs` 38개 sweep — no-`mod.rs` 모던 path 스타일(§2.1 규칙 C / CLAUDE.md 컨벤션).
  - **(γ-3) ✅** argus-eval bin화 (2026-06-11, `33d5bc8f`/`11c8721f`) — eval 진입 경로를 `argus_eval` bin 으로 부활(살림). argus-chat bin화는 별 sprint 잔여(아래 generate 분할 항목 참조).
  - **(γ-4) ✅** batch orphan 처분 (2026-06-11, `2e53cf44`) — `run_prompt_batch`/batch runner ~1170 LOC 순수 삭제(죽은 포장지). 아래 RESOLVED 항목 참조.
- **ACTIVE (2026-06-11 진입)**: **AB-2/4/6 Stage 모델 재개** (G1 결정 — 과도기 LoopControl 필드 이전 포함). 착수 순서 **AB-4 → AB-6 → AB-2** (사용자 확정). 진입 SSOT = `handoff_ab246_stage_entry_2026_06_11.md` (census verdict + 게이트 + landmine 9건 — D8 재확인은 census 로 완료: re-export 보존, 저위험).
- **이전 Master roadmap (α-K BC, 완료)**: `.agent/todos/roadmap_alpha_k_bc_completion_2026_06_04.md`.

---

## [RESOLVED] host lib 테스트 위생 — γ-3 결정적 테스트 버그 2건 + POCL-first 호스트 OpenCL 테스트 환경 실패

- **Status**: **RESOLVED 전체 (2026-06-12)** — (a)+(b)+(c) 완료. AC 충족: 필터 없는 `cargo test -p llm_rs2 --lib` **1410/1410 PASS, 0 FAIL** (skip 가드 발화 = unified UMA 왕복 1건) + `check_spec_coverage.sh` 정상 완주([6]까지 전 구간 실행, exit 1 = 정당한 커버리지 갭 45건 보고). (발견 2026-06-11, AB-4 device 게이트의 Linux host 이름 단위 대조에서 — pre-AB-4 `ecd07549` worktree 대조로 전원 사전존재 입증, AB-4 회귀 0건)
- **Description**:
  - **(a) γ-3 결정적 테스트 버그 2건** — **✅ 수정 (2026-06-12)**: 둘 다 테스트 fixture 버그(production 무관). roundtrip 은 fixture JSON 을 snake_case(`throttle`/`suspend`)로, protected_prefix 는 clap 입력 `h2o-plus`(kebab) ↔ canonical `h2o_plus`(`policy_name()`) 분리로 수정 → 비-OpenCL lib **결정적 실패 0** (잔여 간헐 FAIL = `kv_cache` RSS flaky 2건, 병렬 교란 — 격리 `--test-threads 1` PASS, 별개 기지 사항). (γ-3a `33d5bc8f` / γ-3b `11c8721f` 도입분):
    - `session::experiment::schedule_source::tests::experiment_schedule_parse_roundtrip` — fixture 가 PascalCase `Throttle`, serde 는 snake_case(`throttle`) 기대.
    - `session::eval_setup::tests::protected_prefix_score_based_defaults_to_4` — clap subcommand 는 `h2o-plus`(kebab), 테스트는 `h2o_plus` 전달.
  - **(b) POCL-first 호스트 OpenCL 테스트 환경 실패 ~25종** — **✅ 수정 (2026-06-12)**, 2026-06-12 재진단으로 실체가 2갈래로 분해됨:
    - **플랫폼 선택**: `OpenCLBackend::new_with_profile_events`(production)와 `create_test_queue`(unified 테스트 헬퍼)가 첫 플랫폼(POCL CPU)을 잡던 것을 **요청 device type(기본 GPU)을 가진 플랫폼 우선 스캔**으로 수정 (스캔 Err="해당 없음" 처리로 POCL clGetDeviceIDs 병렬 간헐 Err에도 견딤). 이 호스트(POCL+NVIDIA)에선 NVIDIA RTX 3090 Ti 가 선택돼 backend::opencl 35종이 **실 GPU에서 전부 PASS**. 단일 플랫폼 디바이스(Adreno/Jetson)는 기존 선택과 동일(무영향).
    - **unified stale 테스트 4종**: SIGABRT 의 진범은 환경이 아니라 **구 "starts mapped" 시맨틱을 가정한 stale 테스트** — `new()`는 의도적으로 unmapped 시작(doc 명시)인데 테스트가 `as_mut_ptr()`(null) 직접 쓰기 → non-unwinding panic → SIGABRT 스위트 전멸(플랫폼 무관). 4종 전부 현 시맨틱으로 재작성 + `test_map_write_unmap_cycle` 왕복 검증은 **UMA(host_unified_memory) skip 가드** 추가 — discrete GPU 는 mapping 이 staging 사본 + 현 `unmap()`이 실제 clEnqueueUnmapMemObject 미수행이라 write 커밋 보장 없음(UnifiedBuffer 설계 타겟=ARM UMA, production 무영향이라 unmap 시맨틱 자체는 미수정).
  - **(c) `scripts/check_spec_coverage.sh` line 90 octal 버그** — **✅ 수정 (2026-06-12)**: L78(LAYER)/L90(RPCMEM) 두 곳 `printf '%03d' "$((10#$n))"` 으로 base-10 강제. `set -e` 하에서 008 파싱 실패 → 즉사하던 것이 [6]까지 전 구간 완주로 복구(45건 INV 커버리지 갭 보고는 정당한 잔여 — 별개 사안).
- **Acceptance Criteria**: Linux host(POCL+NVIDIA)에서 `cargo test -p llm_rs2 --lib` 0 FAIL (skip 가드 발화 허용) + `check_spec_coverage.sh` 정상 완주.

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
- **★전제 stale 확인 + 범위 재산정 (2026-06-13 census)**: backlog 의 "inference loop 전체 재설계, 3 hook 신설, 1~2주" 가정은 **대부분 stale**. β 재작성(2026-06-10) + AB-1~6 cutover 가 `LifecyclePhase`(8-phase) + `PipelineStage` dispatch 로 backlog 가 원한 "typed hook 격리 + 컴파일 타임 등록 누락 차단" 을 이미 정식화. 제안 3 hook 은 기존과 중복 — **PressureHook = EvictionStage Persistent 모드(band edge-trigger)가 이미 수행 / KvCacheHook = evict 는 EvictionStage 대체, append·grow 는 Forward 내부 캡슐화(직접 결합 0) / PrefetchHook = preload_pool 의 PreloadAccess(L2) 별 경로**. "구조 격리" ~90% 달성, "신규 도메인 토대" 는 LifecyclePhase enum 가산+Stage 등록으로 충족(별도 hook 표면 불요). **실질 잔여 = decode_loop (a.6) offload/recall 인라인 1곳(~110 LOC)**.
- **거취 확정 (2026-06-13 사용자 결정)**: 3 hook 신설(ROI 음수·중복) 기각, **OffloadStage(AB-3) 1개 외과적 신설로 진행** — 아래 RESOLVED.
- **★RESOLVED (2026-06-13, AB-3 OffloadStage 구현 완료 `448771e6`)**: Architect 설계(spec triage = 신규 INV 0, §3.28 AB-3 단락 + arch §5.10) → 구현 → S25 검증. **`OffloadStage`**(`engine/src/stages/kv/offload.rs`, EvictionStage 거울, 양방향 `OffloadDirection{Offload,Recall}`, transient OneShot, KvMutate phase) 신설 + decode_loop (a.6) offload/recall 인라인 결합 **제거**(h-1 잔여 유일 결합점 소거) + deprecated `LoopControl.offload_ratio/recall_offload` + `Forward::try_offload/try_recall` + `CommandDispatcher::cache_manager()` getter 삭제(`try_evict` 는 chat 사용→보존) + `reconcile_kv_pos` 가드 `<`→`!=` 일반화(감소=eviction/offload byte-identical·on_kv_prune 감소 시에만, 증가=recall 신규 흡수). **게이트 전부 GREEN**: host lib **1444/0** + manager 226/0 + spec 722/0 + clippy clean + coverage 갭 0 / **S25 verify** `direct_cmd_kvoffload`·`kvoffload_restore` f16/q4 **4종 PASS** + OffloadStage 로그 verbatim 실증(`KvOffload: ratio=0.50, 728 tokens swapped` → `Recalled 728 tokens from swap`, 대칭) / **α-K frozen 3-dtype byte-identical** + tbt f16 +1.0%/f32 +0.3%/q4 −0.1%(n=5 간격 median). 결론: h-1 "구조 격리" 목적 = OffloadStage 1개로 **잔여 0 완성**(census 발견대로 1~2주 아닌 1일). "신규 도메인 토대"는 LifecyclePhase enum 가산+Stage 등록 메커니즘이 이미 충족 — speculative decoding 실착수 시 그 경로 사용.
- (구) **거취 (2026-06-12 T5 D-3 = B)**: 별 sprint 분리 — census 로 범위가 OffloadStage 1개로 축소되어 2026-06-13 착수.
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
- **Status**: TODO (P3) — 정책 RESOLVED + 1차 sprint(`b94e55ee`)로 강제 trigger 사유 해소. 잔여 marker는 정적, 새 추가 없으면 status quo로 충분. **거취 확정 (2026-06-12, Backlog Burndown T1-10)**: status quo 유지 — 아래 재발동 트리거 충족 시에만 재개(트리거 3종은 본문 "재발동 트리거 조건" 참조).
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

## [P2] generate 바이너리 분할 + Manager 통합 — 2026-05-21 등록 / 2026-06-11 부분 해소
- **Status**: **RESOLVED — 분할 완성 + 잔여 chat bin화는 별 sprint 이월 (2026-06-13, Backlog Burndown T6, C1 처분)** — ① 원 항목의 "다수 바이너리 분할" 의도는 argus 패밀리 3종(`argus_cli`/`argus_bench`/`argus_eval`)으로 완성. ② "Manager IPC 통합"(원 gen-resilience 의도)은 argus_bench 가 담당으로 정착 — verify 매트릭스 32종 + T5 recall e2e 로 실증(directive 전 경로 동작). argus_cli 는 happy path 전용이라 swap runtime 미구성(의도된 분업 — T5 기록). ③ **잔여 argus-chat bin화는 설계 SSOT 가 이미 별 sprint 로 이월** (`arch/pipeline_stage_design_v2.md` §9 "chat bin화 잔여 — eval 표면과 분리되어 별 sprint 로 이월" + §5.7.7 AB-2 검증 수단으로서의 chat bin화는 사용자 기각 2026-06-11) — burndown 의 본 항목은 SSOT 처분을 따라 종결, 착수 시점은 사용자 추후 결정(h-1 D-3 과 동일 패턴).
- **부분 해소 (2026-06-11)**: 본 항목의 "다수 바이너리 분할" 의도는 **argus 패밀리 분할로 사실상 완성** — `argus_cli`(단일 추론) / `argus_bench`(측정) / `argus_eval`(eval, γ-3 `33d5bc8f`/`11c8721f`) 3종 정착. **잔여 = `argus-chat` bin화 만** (SSOT `arch/pipeline_stage_design_v2.md` §9 별 sprint 목록에 기등재) + Manager IPC 통합(원 항목 설계 의도).
- **⚠️ STALE 전제 (2026-06-10 갱신)**: 아래 "현 `engine/src/bin/generate.rs` … legacy로 보존" 전제는 **이미 무효**. α-K BC(2026-06-05)에서 legacy generate.rs 가 폐기되어 현 `engine/src/bin/` 에 generate.rs 부재 — 현 bin 8종 = `argus_cli`/`argus_bench`/`argus_eval`/`auf_tool`/`signal_injector`/`test_backend`/`test_model`/`test_q4_soa_byte_equal`(argus_eval γ-3 추가). (항목 자체는 잔여 argus-chat bin화 + Manager IPC 통합 설계 의도 보존을 위해 미삭제.)
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
- **Status**: **범위 밖 재분류 (2026-06-12, Backlog Burndown T1-14 거취)** — 재현 조건이 swap mode + `--backend qnn_oppkg` 조합 한정인데 qnn_oppkg 는 Sprint 2b(2026-05-26)에서 production 제거됨(M3/M4/M5 microbench 보존 전용). A2(성능/QNN 보존 트랙) 동류로 burndown 범위 밖. **재발동 트리거**: qnn_oppkg production 부활, 또는 OpenCL swap mode 에서 동일 cleanup segfault 재현 시 P2 복귀.
- **상세**: swap mode runs (`--swap-incremental-per-tick`, `--swap-phase-aware`) 에서 generation 정상 완료 후 process exit 시 SIGSEGV. baseline (no `--secondary-gguf`) 미발생. 모든 swap mode + `--backend qnn_oppkg` 조합에서 재현.
- **추정 원인**: `RpcmemAliasBuffer` Drop ordering 또는 `cl_mem` release ↔ `rpcmem_free` race. OpenCL queue/context teardown 순서.
- **참고 파일**: `engine/src/buffer/rpcmem_alias_buffer.rs`, `engine/src/models/weights/rpcmem_secondary.rs::RpcmemLayerRegion::Drop`
- **fix 방안 후보**: (a) explicit drop sequence (cl_mem all release → rpcmem_free), (b) reference count guard, (c) backend teardown 시 rpcmem region 명시 release

## [P3] KiviCache hot path downcast resolve — 2026-05-24 등록 (backend extension sprint 후속) / 2026-05-26 P2→P3 강등
- **Status**: TODO (P3) — §13.8-L 강등과 동기 (잔여 위치 일부 중복, 통합 처리 가능). **거취 확정 (2026-06-12, Backlog Burndown T1-11)**: status quo 유지 — production 영향 ≈0 실측(vtable Δ -0.231%) + transitive drag 큼. 재발동 트리거는 §13.8-L 항목과 동일(새 GPU backend production 검토 / marker inflation / 사용자 명시 우선순위).
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
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T1-15)** — `cargo test -p llm_rs2 --lib backend::opencl` 실행: **35 passed; 0 failed** (host NVIDIA RTX 3090 Ti 실 GPU). `7daa7e69` GPU-우선 플랫폼 스캔 수정으로 해소 확인.
- **2026-06-12 triage 단서**: L38 항목(host lib 테스트 위생 RESOLVED) 본문에 따르면 OpenCL 플랫폼 GPU-우선 스캔 수정(`7daa7e69`)으로 호스트 NVIDIA(RTX 3090 Ti)에서 `backend::opencl` 35종이 실 GPU에서 전부 PASS, lib 1410/0 달성으로 보고됨. 본 24-fail 항목은 그 수정으로 해소됐을 가능성이 높으나 **코드/실행 검증 미완** — 단정 금지. T1 위생 트랙에서 `cargo test -p llm_rs2 --lib backend::opencl` 1회 실행으로 0 FAIL 확인 후 RESOLVED 처분 예정.
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

## [RESOLVED] QCF 명명 컨벤션 정리 — `QCF_kv` / `QCF_weight` 2-tier rename
- **Status**: RESOLVED (전체 종결 2026-06-10 — taxonomy rename + IPC dot prefix + estimator curves key 모두 적용 확인)
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
- **Deferred 잔여 — 전부 RESOLVED (2026-06-10 실측 확인)**:
  - ~~`shared::QcfEstimate.estimates` HashMap key — IPC 메시지라 manager 동시 갱신 필요~~ — **2026-05-21 완료** (commit `3d4e1a09`, action name + IPC wire format 모두 dot prefix 통일).
  - ~~`DegradationEstimator` default curves key (`eviction`, `sliding`, `kivi`, `swift`) → `kv.eviction` / `kv.sliding` / `kv.kivi` / `weight.skip`~~ — **완료 (2026-06-10 실측 확인)**: `engine/src/qcf/estimator.rs:72-75` 의 `with_defaults` curves key 가 이미 `kv.eviction`/`kv.sliding`/`kv.kivi`/`weight.skip` dot prefix 로 적용되어 있음 (직접 read 검증). 마지막 Deferred 잔여 해소 → 본 항목 전체 RESOLVED 종결.
- **검증**: `cargo build --release -p llm_rs2` PASS, lib 160 qcf-related test PASS, spec 660/663 PASS (3 fail = 사전 device-required). + estimator curves key 적용 직접 확인 (2026-06-10).
- **작성일**: 2026-04-27
- **종결일**: 2026-05-19 (taxonomy rename) / 2026-05-21 (IPC dot prefix 후속) / **2026-06-10 (estimator curves key — 전체 종결)**

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
- **Status**: **보류 (2026-06-12, Backlog Burndown T1-12)** — 다중 디바이스 확보 의존 + B1=NO(8B 온보딩 없음)로 매트릭스 상한도 미확정. 디바이스 확보 시 재개.
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
- **Status**: **RESOLVED — 현행 유지 확정 (2026-06-12, Backlog Burndown T5 사용자 결정 D-2 = D)** — 순환은 external injection(인위 압박 유지 → actual relief 0 → EWMA 연쇄 하락) 전제에서만 발생하고 production 엔 injection 이 없어 relief 신호가 정확함. 낮은 relief action 의 자연 교체가 올바른 동작. **재발동 예약**: 실측 순환이 관측되면 **(C) memory slope 감지**(ctx.history 기존 노출 재활용, 순수 Lua 1~2일, 최소 침습)가 우선 후보 — 설계안은 burndown T5 Architect 보고(2026-06-12) 참조. 부수 기록: T2 timeout 폴백 decide 는 stale-QCF 구간 관측 빈도를 약간 높여 EWMA 감소를 가속할 수 있으나 순환 구조 자체는 만들지 않음(추가 조치 불요).
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

## [P3] QuantizeHandler stub 제거 + ENG-ALG-092 spec 개정
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T1-9)** — (a) `QuantizeHandler` struct + never-registered NoOp `CachePressureHandler` impl 삭제 (b) `target_bits_for_pressure` free fn 강등(spec MUST 보존 진입점, 기대값 None/8/4/2 불변) (c) ENG-ALG-092 표 정정(`spec/32-engine-algorithms.md` §3.9.2 — "6종" 제거, 등록 3종 + free-fn 항목 분리, `arch/32` 동기) (d) spec test 호출형 갱신. 검증: lib quantize 18/0 + test_action_pool 11/0 + spec eng_alg_060_092 19/0 + clippy clean + `QuantizeHandler` 잔존 참조 grep 0.

## [P3] MGR-DAT-015 데이터 모델 divergence — Level 의 SystemSignal 포함 여부 — 2026-06-12 등록 (T1-7 부수 발견)
- **Status**: **RESOLVED (2026-06-13, `0dc120ad`)** — (a) 채택: spec 을 현 구현(Level 항상 포함)에 정합, 코드 불변. divergence 가 §3.1 전반(MGR-DAT-010/011/012/013/015)에 퍼져 있어 일괄 정정 — SystemSignal 4 variant 전부 `level:Level` 보유(shared/lib.rs) 반영. spec test 무영향(이미 level 채워 생성), COVERAGE 무관(기존 명세 의미 정정).
- **Sprint**: backlog
- **Dependencies**: 없음
- **출처**: Backlog Burndown T1-7(EnergyConstraint divergence 해소) 중 Architect 발견
- **Description**: `spec/23-manager-data.md` MGR-DAT-015 는 "Level 은 D-Bus 전송 경로 전용, 내부 SystemSignal 에 미포함"으로 명세하나, 실제 `SystemSignal::EnergyConstraint`(및 다른 3개 variant)는 `level` 필드를 가짐. MGR-ALG-015 수식보다 큰 데이터 모델 divergence 라 T1-7 범위(C5 보수적 디폴트)에서 의도적으로 제외.
- **Acceptance Criteria**: 택1 — (a) MGR-DAT-015 를 "Level 은 SystemSignal 에 항상 포함, D-Bus 가 ThresholdEvaluator 로 재산출"로 명세 정정(코드 불변), 또는 (b) MGR-DAT-015 재설계. 방향이 갈리면 spec 의미 변경이라 사용자 질문 대상(C5).
- **Notes**: 담당 권장 Architect(/spec-manage).
- **Notes**: `QuantizeHandler::handle()`은 어느 pipeline에도 등록 안 되는 NoOp stub (위 MergeHandler/SparseHandler 동류, 단 아직 미삭제). 단순 삭제 불가 사유 — `target_bits_for_pressure`(PressureLevel→KIVI bits 매핑)가 ENG-ALG-092 *(MUST)* 로 명세 + `test_eng_alg_060_092.rs`/`test_action_pool.rs`가 검증(production 호출지는 0). 묶음 작업: (a) QuantizeHandler struct 삭제, (b) `target_bits_for_pressure`를 free fn 강등 또는 함께 정리, (c) ENG-ALG-092 표 정정 — 제목 "6종"인데 표엔 4개만 나열(Compress/Merge/Sparse 잔재 stale) + QuantizeHandler "완료"인데 never-registered NoOp(spec-impl divergence), (d) spec test 갱신. spec ID 걸려 Architect/spec-manage 라운드 필요. **선행 완료**: 2026-05-29 B-a — `ActionResult` dead variant `Quantized`/`Recalled` 삭제(producer/consumer 0, 영수증 채널의 빈 칸).

## [P3] EnergyConstraint 스펙-코드 Divergence 해소
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T1-7, C5 보수적 디폴트 = spec 갱신·코드 불변)** — divergence 가 수식뿐 아니라 데이터 모델 서술 전반(SystemSignal 에 `battery_pct` 필드 부재 — EnergyMonitor 가 Level 변환 후 폐기)에 걸쳐 있어 4개 spec 위치 정합화: `spec/22-manager-algorithms.md` MGR-ALG-014 표 + MGR-ALG-015 전면 재작성("Raw"→"Level Processing") / `spec/20-manager.md` MGR-029·MGR-020 / `spec/23-manager-data.md` MGR-DAT-014 / `arch/22-manager-algorithms.md` §1.6·§10. 기존 spec test 2건은 이미 Level 기반 검증이라 무영향. **부수 발견 → 신규 항목 등록**: MGR-DAT-015 데이터 모델 divergence(아래 항목 참조).
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
- **Status**: **현행 유지 확정 (2026-06-12, Backlog Burndown T1-8, C5 보수적 디폴트)** — 필요성 미확정(Notes) 상태에서 매칭 거동 변경은 회피. 재평가 트리거: 실제 다중 장치 배포 시점(신규 디바이스의 zone_type 이 exact match 로 안 잡히는 실사례 발생 시).
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

---

## [RESOLVED] Format 명명 통일 — `KVCacheLayer`/`WeightLayer` → `KVCacheFormat`/`WeightFormat`
- **Status**: RESOLVED (2026-06-12, Backlog Burndown triage) — 문서 prose는 2026-05-30 전부 완료, **유일 잔여였던 코드 rename(`KVCacheOps`)은 처분 불요**: α-K BC(2026-06-05, MEMORY 인덱스 [project-backend-axis] + L120 "KVCacheOps trait 완전 폐기")에서 `KVCacheOps` trait 자체가 폐기되어 rename 대상 심볼이 부재. "Format"으로 통일할 코드 노출 0 → 사실상 종결.
- **Sprint**: backlog
- **Dependencies**: 없음 (문서 rename은 독립). 코드 rename만 ADR-0001/Phase α-K 동행.
- **Description**: "Layer"가 두 의미로 충돌 — ① transformer layer(`LlamaLayer`/`TransformerLayer`/`LayerSlot`/`layer_idx` 등 코드 다수) ② 저장 형태(설계 `KVCacheLayer`/`WeightLayer`, 코드엔 grep 0건). 특히 weight/precision swap 도메인에 `LayerSlot`(①)과 설계 `WeightLayer`(②)가 공존 예정이라 혼동. grill(2026-05-30) 결과 **저장 형태(noun)는 `Format`으로 통일**. "Layer"는 transformer layer 전용으로 보존.
- **범위 (문서 — 코드 0)**:
  - ~~`arch/pipeline_stage_design_v2.md`~~ **완료 (2026-05-30)**: 타입명(`KVCacheLayer→KVCacheFormat` 등) + 축명(`storage/policy 축`→`Format/Stage 축`) + generic "paradigm"→"Format" 일괄. handle 3종 예시 `Arc<dyn KVCacheFormat>` 포함.
  - ~~`spec/41-invariants.md`~~ **완료 (2026-05-30)**: 현행 normative INV 본문(`INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`/`PAIRED-KERNEL`/`INV-STAGE-LAYER-HANDLE` + 교차참조 INV-121/123/149/150 + INV-DECODE-STAGE-001) prose의 `*Layer`/`paradigm` → `*Format`. **INV ID(`INV-KVCACHELAYER-*` / `INV-STAGE-LAYER-HANDLE`)는 안정 키로 유지** (2026-05-30 결정 — ID는 추적용 opaque key). → `tests/spec/` 영향 없음 (ID 불변 + 해당 INV는 source-grep 검증). **§3.28 변경요약·폐기 로그(changelog)는 동결** — 당시 용어 보존. v1-pointer 원본열 `§3.5 (KVCacheLayer)`도 frozen v1 doc 가리키므로 유지.
  - ~~`arch/backend_conformance_harness.md`~~ **완료 (2026-05-30)**: `paradigm-agnostic`→`Format-agnostic`, `④(KVCacheLayer)`→`④(KVCacheFormat)`.
  - ~~`docs/adr/0001-kv-dispatch-paradigm.md`~~ **완료 (2026-05-30)**: 타입명(`KVCacheLayer`/`StandardLayer`/`KIVILayer`→`*Format`) + storage-sense `paradigm`→`Format` + 명칭정리 amendment 노트. **유지**: 제목·§5 "dispatch paradigm"(Generic↔dyn *접근법*, ≠저장형태), 파일명, `KVCacheOps`(코드명), §6 게이트 "5 KV 구성"(Format·Stage 혼재라 중립어).
  - **편집 불요 (전부 changelog)**: `arch/inference_pipeline.md`(상단 `>` 연혁 + 문서 자체 v1-legacy 동결) · `arch/README.md`(catalog cell + `>` 연혁) — 현행 normative prose 부재 확인 (2026-05-30).
  - **동결** `arch/pipeline_stage_design.md` (v1, 124곳): 결정-이력 아카이브 → 당시 명칭 보존, 손대지 않음.
- **코드 rename (ADR-0001 진입 시 흡수)**: `KVCacheOps` → `KVCacheFormat` (현 trait 명). 선행 불필요 — Generic→dyn 전환 sprint와 동행.
- **이미 반영**: `CONTEXT.md` (Format 용어 + 두 축 Format/Stage + Flagged ambiguities) + `arch/pipeline_stage_design_v2.md` (2026-05-30).
- **담당 권장**: 코드 rename은 Phase α-K (Senior Implementer / Implementer)
- **작성일**: 2026-05-30 / **갱신**: 2026-05-30 (문서 prose 전부 완료 — spec INV 본문 + backend_conformance + adr/0001. inference_pipeline/README는 전부 changelog라 편집 불요. 잔여 = 코드 `KVCacheOps`만)

---

## [트랙] KV 캐시 관리 확장성 로드맵 — 연구 동향 스트레스 테스트 후속 (2026-06-10 등록)

> **출처**: 2026-06-10 설계 견고성 검토 — 최신 연구(2024–2026) ~70개 기법을 4갈래 병렬 조사(merge / eviction·budget / 표현 변경 / 패러다임·시스템)하여 3축 플러그인 설계(Stage ⊥ Format ⊥ Backend-capability)에 대조. 판정 등급: **A**(현 플러그인으로 가능, ~17) / **B**(어휘 확장·단순 op 추가, ~22) / **C**(복잡한 추가·설계 변경, ~30 — 이 중 ~12는 모델 아키텍처·학습·모달리티라 어느 엔진이든 코어 작업인 "범위 밖").
> **결론**: plan-returning 골격과 3축 직교는 견고 — 식별된 어떤 갭도 "stage 직접 쓰기 권한"으로는 풀리지 않음(입력 부재/어휘 부족/구조 부재가 원인). 갭은 IR 어휘 3건(항목 1–3) + 표면 2건(항목 4–5) + 트리거 보류 3군집(항목 6–8)으로 수렴. 항목 1–5 수행 시 다룰 수 있는 영역(57개) 기준 표현 커버리지 **~40% → ~80%**, 잔여는 전부 개봉 조건이 명시된 의도적 보류.
> **착수 순서 (사용자 결정 2026-06-10)**: 0(검증 선행) → 1–3(어휘 확장) → 4(read-plan ADR) → 5(persistence — **결이 달라 마지막**). 6–9는 트리거 개봉.
> **진행 (2026-06-12)**: 0 RESOLVED(4종 게이트 종결) · 1 보류(Demote RED) · 3 RESOLVED(QueryStats 구현 완료) · **4 RESOLVED(ADR-0011 확정, 구현=4-impl)**. 항목 3+4 = "KV 구조 확정" 게이트 도달.
> **★분기 취소 (2026-06-12 저녁 사용자 결정)**: 별도 워크트리 대형 리팩토링을 하지 않기로 확정 — **항목 2·4-impl·5 의 "리팩토링 머지 후" 동결 해제**(즉시 착수 가능). `worktree_split_hygiene_2026_06_12.md` 효력 종료(②게이트 3항은 일반 회귀 게이트로 존속, verify 는 QCF_kv 라운드 종결로 **30/30**).

### [RESOLVED 2026-06-12] 0. 검증 선행 — 확장 0개로 가능한 1B 실측 (4종: 3 + 항목 1 게이트)
- **Status**: **RESOLVED (2026-06-12)** / **Sprint**: completed / **Dependencies**: 없음
- **판정 요약 (4종)**: **R-KV 보류**(redundant fraction 0.9964 포화·rkv≈h2o 퇴화·vs sliding PPL ratio 0.9982 동률) / **A2SF 보류**(BOS ratio 63.1% 감소로 forgetting 동작 확인되나 PPL 2/5 win로 sliding 미달) / **head 분산 개봉 후보**(max/min 63.7배·2nd-min 9.8배 ≥ 5배 — 단 최종 layer 1개 해상도 한계) / **Demote RED**(실모델 PPL 5/5 RED, ratio 1.22~6.04× — 항목 1 게이트 = 전체 보류).
- **D4 적용**: 기법 항목(R-KV·A2SF·Demote=항목1)은 게이트 RED/보류 존중. 인프라 항목 3·4는 1B 승패 무관 진행(아래 각 항목 표기).
- **종합 리포트**: `experiments/kv_roadmap_item0/rerun/REPORT.md` (P3 재측정, HEAD `c702ff83`) / **게이트 임계 SSOT**: `arch/kv_roadmap_item0_measurement.md` §5 / **스프린트 기록**: `.agent/todos/sprint_kv_roadmap_item0_2026_06_12.md`.
- **측정 경위**: 1차 측정에서 P2 구현 4종 e2e 배선 누락 발견 → 수정 라운드 7커밋(`449cf55f`..`1182fa0c`, HEAD `c702ff83`) → 재측정 확정. 프로토콜 편차 2건: ① prefill 전체 고정이 PPL 모드와 충돌(decode=0→eviction 미발동)하여 prefill=150 통일로 대체, ② EMR 미산출이라 PPL ratio로 대체(보류 판정 불변). 상세 = 스프린트 파일 §P3 결과.
- **부수 실측**: qcf_sum ↔ PPL 역전(sliding 0.60 vs h2o 0.018인데 PPL은 sliding 우세) — L1112 "QCF_kv 정규화 비대칭" 설계 라운드의 수식 포화 결함을 독립 경로에서 보강(해당 항목에 cross-ref 추가됨).
- **원 Description (아카이브)**: 후속 확장의 선결 게이트 — 조사 기법 효익은 전부 7–8B+/long-context 검증(Round 14–15 교훈: 1B에서 누적 score 차별화 무가치)이라 1B 실측 없이 착수 금지. R-KV(arXiv 2505.24133) / A2SF(arXiv 2407.20485) / head 분산(항목 6 개봉 게이트) + Demote 모사(항목 1 게이트, D3 합류).

### [보류 2026-06-12] 1. plan IR 어휘 확장 ① — `Demote` op ("보존하되 저정밀로" 제3 상태)
- **Status**: **보류 (게이트 RED, 2026-06-12)** / **Sprint**: backlog / **Dependencies**: 게이트 실험(아래) — **RED 확정**
- **게이트 판정 (2026-06-12, 항목 0 스프린트 P2d/P3)**: Demote 모사 실모델 PPL **5/5 도메인 RED** (demote PPL ratio 1.22~6.04× — 전부 sliding보다 나쁨). 즉 **현 1B 타겟에서 demote가 sliding을 이기지 못함** — 논문의 "4×@4bit > 1×@16bit"가 1B/2048/비-reasoning에서 미재현. NMSE 보조 신호(demote 0.145 < sliding 0.75)는 demote 우세였으나 실추론 PPL이 정본(K 2차 효과가 NMSE에 미포착). 리포트: `experiments/kv_roadmap_item0/rerun/REPORT.md` §측정 4.
- **재개 조건**: 8B/long-context 온보딩 또는 retrieval 태스크 실수요. 재개 시 동일 모사 게이트 GREEN 선행 필수(Round 14–15 위험 = 1B score 신뢰성).
- **게이트 실험 (구현 전 모사)**: host eval에서 F16 캐시의 선택 토큰을 Q4/Q2 왕복 양자화해 "content-aware 강등이 sliding eviction을 품질(PPL/EMR)에서 이기는가"만 선검증. Round 14–15와 동일 위험(1B score 신뢰성)이므로 RED면 본 항목 전체 보류. → **2026-06-12 RED 확정 → 보류.**
- **Description**: 현 IR은 keep(보존) 아니면 소멸(evict/merge)의 이분법 — 문헌은 "evict 후보를 버리는 대신 저정밀 강등"(quantized pruning)이 순수 eviction을 일관되게 이기는 쪽으로 정착(arXiv 2412.12706). KIVI 합성 패턴이 못 덮는 이유 = 강등 경계가 시간 규칙이 아니라 importance 기반 **stage의 결정**이라, 결정(stage)→실행(format) 간 IR 채널이 미싱 링크.
  - `KVCachePlan`에 `demotes: Vec<DemoteSpec { tokens, level: u8 }>` 추가 (level 해석은 format 책임)
  - `KVCacheFormat::demote(tokens, level)` optional 메서드 — 미지원 format은 명시적 Err(fail-fast)
  - executor 순서: demote → merge → compact. `PlanAbi`에 배열 1개 추가(가산적 — 기존 `.so` 무수정 호환)
  - 혼합 정밀 저장 + 혼합 dequant 커널은 demote 지원 **format plugin 내부 책임** (KIVI가 한 캐시에 Q2 packed + F32 residual 두 영역을 운용하는 선례)
- **해금**: MiKV(2402.18096), ZipCache(NeurIPS'24 2405.14256), QAQ, Don't-Waste-Bits(2604.04722 — 온디바이스 직격), MixKVQ·PM-KVQ 부분.
- **코드 접점**: `crates/technique-api/src/lib.rs`(`KVCachePlan`/`PlanAbi`), `engine/src/format/kv_cache_format.rs`, plan executor(`engine/src/pressure/eviction/stage_registry.rs` 경로), 신규 mixed-precision format crate.
- **Acceptance Criteria**: ADR-0004 amendment + `compact_parity` demote 케이스 확장 + 게이트 실험 GREEN + lib/clippy 무회귀 + 기존 .so dlopen 호환 확인.

### [보류 2026-06-12] 2. plan IR 어휘 확장 ② — K/V 비대칭 merge 가중치
- **Status**: **보류 (2026-06-12 사용자 결정 B1=NO, Backlog Burndown triage)** / **Sprint**: backlog / **Dependencies**: 없음 (1과 독립)
- **보류 근거 (B1=NO)**: AC가 요구하는 1B WeightedKV ablation(8B 결과 무비판 이식 금지 = V 정보 균등 분포 가정의 1B 성립 여부 검증)이 **검증 불가** — (a) 8B/long-context 온보딩 NO(B1) 확정 + (b) 항목 0 결과로 1B score 계열 전반 무가치 재확인(R-KV·A2SF 보류, Round 14–15 연속)이라 1B ablation 회의적. 따라서 검증 게이트를 통과할 경로 없음.
- **재개 트리거**: **8B/long-context 모델 온보딩** — 온보딩 시 V 이질성 가정 우선순위 재평가 + WeightedKV ablation 환경 확보. (트리거 충족 전 착수 금지.)
- **Description**: `WeightedMerge`는 K/V 동일 가중치 강제 — 2025–26 문헌이 "K=스펙트럼 집중(균질)이라 적극 merge, V=분산(이질)이라 보수적"으로 수렴(순수 merge 신규 기법의 44%가 이 가정에 차단). `from: Vec<(pos, w)>` → `(pos, w_k, w_v)` + `into_weight` 분리(최소안: `apply_to: Both|KeyOnly|ValueOnly` 1필드). `apply_merges` K/V 루프 가중 분리(F32/F16/Q4_0 — `scatter_reduce_q4` 의미 포함). `MergeAbi`/`FromPairAbi` 필드 추가(가산적).
- **해금**: WeightedKV(ICASSP'25 2503.01330 — K discard + V만 merge), KVSlimmer(2603.00907 — 2026 이론 정당화), KeepKV 부분(2504.09936 — K log-스케일은 plugin이 가중치 계산을 끝내 plan에 굽는 방식으로 흡수, 실행은 선형 유지), EMS 부분.
- **Acceptance Criteria**: 기존 동일-가중 경로 bit-identical(d2o_stage_eq 무회귀) + 비대칭 단위 테스트 + ABI 호환. 1B 검증: WeightedKV ablation(8B 결과 무비판 이식 금지 — V 정보 균등 분포 가정이 1B에서 성립하는지).
- **참고 (2026-06-12, 항목 0 결과)**: 변동 없음(자체 1B ablation 게이트 유지)이나, 항목 0에서 **1B score 계열 전반이 무가치로 재확인**(R-KV·A2SF 보류, Round 14–15 연속) → 1B ablation은 회의적 전망. 8B 온보딩 시 우선순위 재평가 권장.

### [P2] 3. StageCtx 어휘 확장 ③ — `QueryStats` TensorKind
- **Status**: **RESOLVED (2026-06-12, KV 로드맵 항목 3+4 스프린트 — `783bcadd`+`a98cd679`)** — `TensorKind::QueryStats`(disc 4 가산) + `QueryStatsAccumulator`(Welford online, GQA Q32→kv8 평균) 구현. 캡처 = score-active 폴백 경로 한정(production decode happy path 분기 1회). 게이트: e2e non_empty 128행 + greedy byte-identical + S25 α-K frozen 3-dtype byte-identical. 항목 4(read-plan)의 신호 공급원으로 충족 — T4 Quest 가 소비. (구 헤더 "진행 중" stale, 2026-06-13 정정.) / **Sprint**: completed / **Dependencies**: 없음
- **스프린트 (2026-06-12)**: `.agent/todos/sprint_kv_roadmap_item34_2026_06_12.md` (P1 설계 → P2 구현 → P3 검증). 진입 = `handoff_kv_roadmap_item34_entry_2026_06_12.md`. 완료 게이트 = host 통계 테스트 + 기존 `tensor()` 소비자 무회귀 + α-K frozen 3-dtype byte-identical + **실모델 e2e 1회**(항목 0 미배선 허상 교훈). Q 캡처는 기본 off / score-active 시만(hot path 비용 0). accumulator commit 격리(L1112 QCF_kv 라운드와 동시 작업 주의).
- **D4 비차단 확인 (2026-06-12)**: 인프라 항목이라 항목 0 기법 게이트 RED와 **무관하게 진행**. 항목 4의 신호 공급원이므로 4보다 선행 권장 → 항목 3+4가 항목 0 종결 후 차기 착수 1순위.
- **Description**: prefill-end 압축 패러다임(SnapKV류)의 신호인 windowed attention 행렬은 flash attention이 materialize하지 않음(우리/문헌 공통 제약) — 2025 frontier(Expected Attention, arXiv 2510.00636)는 행렬 대신 **query 분포 통계로 미래 attention을 closed-form 추정**. `TensorKind::QueryStats`(per layer·kv_head Q running mean/var) 추가 + forward 경로 Q 캡처 1지점(`AttentionScoreAccumulator` 패턴 재사용). ADR-0004 §7이 예고한 자리("query_state(Quest) 캡처 미배선 → 후속 PR").
- **해금**: Expected Attention, LU-KV·MixKVQ 부분. 항목 4(read-plan)의 Quest류 page 선택 신호로 재사용(시너지).
- **Acceptance Criteria**: TensorKind variant + 엔진 누적 배선 + host 통계 정확성 테스트 + 기존 기법(`tensor()` 소비자) 무영향.

### [RESOLVED 2026-06-12] 4. read-plan 표면 ADR — "무엇을 읽을지"의 4번째 plugin 표면
- **Status**: **RESOLVED (ADR-0011 확정, 2026-06-12)** — ADR 작성이 본 항목의 전부, **구현 코드 0줄**(병렬 리팩토링 의미적 충돌 예방). 구현은 신규 항목 "4-impl"(아래)로 재등록 / **Sprint**: completed / **Dependencies**: 항목 3 선행(P3 완료 후 착수 — 충족)
- **★ADR-0011 확정 (2026-06-12, Architect P4)**: `docs/adr/0011-kv-read-plan-surface.md` (Status=Proposed, 구현 보류). 표면 = `KVReadStage::read_plan(ctx) -> Option<KVReadPlan{granularity, select}>`(`KVCacheStage`/`WeightStage` 거울 = 4번째 plan-returning 형제, `KV_READ_STAGES` 평행 registry). 6 설계 결정 고정 — D2 `KVReadPlan{ReadGranularity(Token/Page), select:ascending Vec<usize>}`(new_pos 없음=비변형) / D3 발화=layer i 진입 전, plan 이중 해석(Quest 마스크 ⊥ KVSwap prefetch)은 소비자-주도 / D4 format `attention_into_selected`=capability opt-in(미지원=full read 폴백, soft constraint=근사 가속이지 정확성 계약 아님) / D5 page 메타(K min/max) 유지=read stage 자신이 `tensor(Key)`+`tensor(QueryStats)`로 incremental(코어 무수정, 항목 3 시너지) / §6 read⊥cache plan 직교(read 비변형이라 race 없음, eviction 후 read stage 메타 재구축 책임) / §11 C-ABI forward-compat(AbiStageCtx 재사용 + ADR-0010 capability 슬롯). 대안 6종 기각(status quo / Quest-format 매장=M×N재발 / 2표면 분리 / 정확성 계약화 / 코어 page 메타 / step-tier 1회). 최대 리스크 R1 layer-tier hot-path(RPN 378) → read stage 부재 시 분기 1회(INV-147 정신)+TBT Δ≤+3% 게이트(구현 시). Spec Triage=spec 무관(arch-only, 신규 INV 불요).
- **원 스프린트 (아카이브)**: `.agent/todos/sprint_kv_roadmap_item34_2026_06_12.md` P4. 항목 3 구현 + 본 ADR 확정 = "KV 구조 확정" 게이트 → 직후 사용자 별도 워크트리 대형 리팩토링 병렬 시작. 완료 게이트(충족) = ADR 신규 1건(대안 ≥2 + status quo + grill 통과) + 구현 단계 분해 backlog 재등록(아래 4-impl) + **구현 코드 0줄**.
- **해금 (ADR §1, ~9건)**: Quest(ICML'24 2406.10774), InfLLM, HISA, BLASST, shadowAttn(2508.16703 — 모바일 NPU prefill 전용), InfiniGen(OSDI'24), KVSwap(2511.11907) prefetch.
- **실익 정직 평가 (ADR §10)**: 1B/2048 decode는 memory-bound라 **순이득 기대 낮음**(항목 0 1B 무가치 교훈과 동결) — 가치는 prefill TTFT + 8B/디스크 offload 토대 + 논문 기여("4번째 표면도 동일 plan-returning 문법으로 가산" 확장성 실증). **과대 포장 금지 못박음**: 1B 성능 주장은 8B 측정 전까지 하지 않는다.

### [P2] 4-impl. read-plan 표면 구현 — `KVReadStage` 배선 (ADR-0011 산출)
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T4 — S1~S6 전부 구현)** — Amendment A1(`ec3c6a38`, dispatch 고정: per-step 캡처 + None 분기 1회 + &dyn layer 당 1 call, R1 378→108) → S1+S3(`f97e7597` — 4번째 linkme registry + SelectiveRead capability/StandardFormat gather 구현) → S2(seam — `as_selective_read()` capability-handle 라우팅 + plan 검증 + full read 폴백) → S4+S5(`0d3eb94a` — Quest crate: page K min/max Mutex incremental + eviction 후 자기 재구축 + 상한 내적 top-k + sink/recent 보존 + CLI `--read-stage`) → S6(prefetch priority hint 보강 채널, 배선 GREEN only). **게이트 전부 충족**: α-K frozen 3-dtype byte-identical(S2 직후 + 최종 2회 확인) + tbt Δ f16 +0.9%/f32 +0.2%/q4 −0.1%(n=5 간격 median) + 폴백 bit-identical + dlopen gate 2종 PASS + lib 1432/0 + spec 706/0 + clippy clean + `arch/kv_read_plan.md` 매핑 신설. **한계 기록**: production decode 는 StageCtx 에 query/QueryStats 미공급이라 Quest 가 K-magnitude proxy 강하 — 긴 prompt 품질 발산 실측(예측된 근사, ADR §10 — 1B 성능/품질 주장 없음, 평가는 8B/QueryStats 공급 후속). / **Sprint**: completed / **Dependencies**: ADR-0011 확정(✅) + 항목 3(✅).
- **★착수 게이트 (2026-06-12 사용자 결정)**: ADR-선행 = 미래 표면을 문서 제약으로 고정해 병렬 리팩토링과 의미적 충돌 예방. **리팩토링이 `attention_into` 주변을 재설계하므로 머지 전 구현 착수 금지**(ADR §8 Premortem #3 — read plan→capability→폴백 의미 계약을 ADR 이 지키되, 구현은 리팩토링 결과 위에 올림). 같은 이유 항목 2(K/V 비대칭 merge)·5(persistence)도 머지 후.
- **구현 단계 분해 (ADR-0011 D1~D5 기준, 머지 후 재-triage 필요)**:
  - **S1 technique-api 표면**: `KVReadStage` trait + `KVReadPlan{ReadGranularity(#[repr(u32)] Token/Page), select:Vec<usize>}` + `KVReadStageReg`/`KV_READ_STAGES`(linkme) + `find_read_stage` + `ensure_builtin_read_stages_registered` self-test. `ReadStageCtx` = 기존 `StageCtx` 재사용(ADR §11 — 신설 불요). 검증: 빌드 GREEN + registry 등록.
  - **S2 엔진 executor seam (★hot-path 핵심)**: layer 경계에서 `read_plan` 호출 + plan 라우팅(활성 format `SelectiveRead` → selective / offload → prefetch 큐 / 미지원 → full read 폴백). **read stage 부재 시 분기 1회**(`Option::is_none()`, INV-147 정신). hot-path dispatch 전략(정적 vs dyn)은 **별도 amendment 로 고정**(ADR R1 RPN 378). 검증: α-K frozen byte-identical(read stage 부재) + TBT Δ≤+3%(ADR-0005 §8 게이트).
  - **S3 format capability**: `SelectiveRead{attention_into_selected(q, ..., select, granularity, scores)}` capability-handle. 첫 구현 = Standard format. 미지원 자동 full read 폴백(코드 0). 검증: 폴백 경로 bit-identical(full read).
  - **S4 page 메타 유지**: read stage `&self` Mutex 에 page K min/max 보유 + `read_plan` 마다 `tensor(Key)`/`tensor(QueryStats)`(항목 3) incremental 갱신. eviction 후 메타 재구축(ADR §6 — page-id 안정성 R3 구현 시 확정: page-aligned eviction or content-derived page-id). 검증: eviction 후 read plan 정합.
  - **S5 첫 빌트인 = Quest**: host 테스트(value-aware 실행) → CLI `--read-stage quest` opt-in(ADR-0004 §8 CAOTE 배선 패턴). 폴백 시 stderr 1회 경고(ADR §8 Premortem #2). 검증: PPL/EMR 근사 일치(정확 일치 아님 — 근사 가속) + 폴백 경고.
  - **S6 offload prefetch 정합**: 기존 `OffloadKVCache`(SeqMajor) + `--max-prefetch-depth` 와 read plan 을 prefetch *목록 공급원*으로 연결(KVSwap). 8B/offload 측정 — **1B 성능 주장 금지**(ADR §10).
- **검증 게이트 (전체)**: α-K frozen byte-identical(read stage 부재 happy path) + 폴백 bit-identical(미지원 format) + read 활성 근사 일치+품질 메트릭 + TBT Δ≤+3% + lib N/0 + clippy --workspace clean + 기존 .so dlopen 호환(ADR §11). **arch 컴포넌트 매핑 신설**(executor seam + capability + page-meta 흐름 — ADR-0004 §10 M-Q 의 `arch/kv_query_stats.md` 타이밍 규율).
- **참고**: ADR-0011 이 표면 SSOT. 머지 후 리팩토링이 바꾼 `attention_into` 시그니처 위에서 S2/S3 를 재-triage(ADR §8 #3 — 의미 계약은 ADR 이 보존, 구현 디테일만 이동).

### [P3] 5. 세션 KV persistence (prefix cache) — ★사용자 결정: 결이 달라 마지막 수행
- **Status**: **Tier 1 RESOLVED (2026-06-12, Backlog Burndown T3)** — 설계 `b207d17b`(ADR-0012, ENG-080~085/ENG-DAT-110/INV-189~191) + 구현 `ebc59fcf`(SnapshotRestore capability + ARGUSKV1 v2 + CLI 2-flag + happy path 배선). **AC 전부 충족**: ① S25 동일 prompt 2회차 TTFT **5085.83ms → 0.01ms**(medium_qa 918tok prefill skip, 스냅샷 26.9MB) ② 복원 후 greedy 생성 == fresh prefill — host 3 kv-type BIT-EXACT + S25 OpenCL(read/write_buffer coherent) 생성 텍스트 동일 ③ 무효화 3케이스+α 테스트(INV-190 7종). **핵심 교훈 2건**: (a) KV snapshot 이 byte-perfect 여도 마지막 logits 의 GEMM 배치 경로(M=n batch vs M=1 forward_gen) ±ulp 차이가 greedy 를 가름 → **last-token logits 를 스냅샷에 포함**(v2, llama.cpp state_save 동형)이 bit-exact 의 원천 (b) restore 는 prefill 의 `sampler.observe_token` 부수효과도 건너뛰므로 sampler history 재구성 필수(repetition_penalty=1.1 기본이라 greedy 에서도 발현). partial restore 는 잔여 prefill 의 M 차이로 bit-exact 비보장(수치 동등 — INV-191 명문화). frozen 무회귀: 3-dtype byte-identical + tbt Δ≤+0.1%. **Tier 2(턴 종료 누적)/Tier 3(임의 chunk, CacheBlend)는 미착수 — 필요 시 별도 등록.**
- **Description**: KV 캐시 수명이 프로세스 내 한 세션 — 새 대화/프로세스 재시작/대화 전환마다 동일 prefix(system prompt + 히스토리)를 전부 재prefill. 업계 동일물: llama.cpp `--prompt-cache`, vLLM automatic prefix caching.
  - **Tier 1** (본 항목): system prompt prefix 저장/복원 — ① format snapshot capability(`snapshot(range)→bytes` / `restore(bytes, at_pos)`, capability-handle 신설 요건 충족: 소비자 = prefix cache + resilience Suspend) ② 스냅샷 파일(헤더: model_hash·format_id·tokenizer_hash·token_ids — 하나라도 불일치 시 폐기) ③ 세션 API `save_prefix`/`try_restore_prefix`(token-id 단위 prefix 일치 검사가 정확성의 전부) ④ "prefill 직후·eviction 전 상태만 저장"으로 position 매핑 문제 회피
  - **정확성 근거**: KV는 prefix 토큰의 순수 함수(causal) + RoPE는 절대 위치로 K에 baked — prefix는 항상 위치 0부터라 자연 정합
  - **Tier 2**(턴 종료 히스토리 누적 저장 — 프로세스 재시작 복원) / **Tier 3**(임의 chunk 재사용, CacheBlend/EPIC — position 재정합 필요)는 Tier 1 후 별도 등록
- **효익**: 1B 기준 500tok prefix = Q4_0 ~4.6MB / f16 ~16MB → 디스크 복원 수십 ms vs prefill 재계산 수백 ms~수 초. C 군집 중 1B 현 타겟에서 즉시 체감(TTFT) 나오는 유일 항목.
- **Acceptance Criteria**: 동일 prompt 2회차 TTFT 단축 실측(S25) + 복원 후 greedy token-id가 fresh prefill과 일치 + 무효화 3케이스(model/format/tokenizer 변경) 테스트.

### [P3·트리거] 6. per-head 가변 budget 회계 — Ada-KV/HeadKV/LAVa/EMS/LU-KV 군집
- **개봉 트리거**: 항목 0-3(1B head importance 분산 실측)에서 분산 大 판정 **그리고** head-adaptive 기법 실수요(논문 비교군 포함).
- **트리거 상태 (2026-06-12, 항목 0 결과)**: **부분 충족** — "분산 大 판정"은 ✅ (max/min C_h 평균 63.7배, 0-수렴 head 배제한 2nd-min 기준도 ~9.8배 ≥ 게이트 5배 / sparse~0.25·dispersed~0.004 head 명확 공존, `experiments/kv_roadmap_item0/rerun/REPORT.md` §측정 3). **단 단일 layer 해상도 한계** — `last_step_head_attn()`이 최종 디코더 layer 1개만 반환(API 제약, 설계서 §2.3-B 의도된 단순화), 16층 전체 분산 미측정. **잔여 개봉 트리거 2건**: (a) head-adaptive 실수요(논문 비교군), (b) 다층 해상도 확인(소규모 보강 측정 — 분산이 단일 layer 아티팩트가 아님을 확정). 두 잔여 충족 전 착수 금지(§Notes ARM64 변길이 per-head 커널 재설계 비용 위험).
- **Description**: `KeepSpec::PerHead`는 타입상 이미 ragged 표현 가능 — 차단은 어휘가 아니라 **엔진 불변식**: `new_pos = keep.len()` 도출, HeadMajor 균일 stride, decode loop 단일 pos 회계, ragged attention 커널(전용 format으로 격리 가능하나 pos 회계는 엔진 잔존). 잔여 C 중 가장 뼈아픈 군집 — head 차등 budget이 2025–26 SOTA의 핵심 차별점(Ada-KV NeurIPS'25 2407.11550, HeadKV ICLR'25 2410.19258, LAVa 2509.09754, EMS 2412.08521, LU-KV 2602.08585).
- **Notes**: ARM64/Adreno에서 변길이 per-head KV 커널 재설계 비용이 효익을 압도할 위험 — 게이트 통과 전 착수 금지.

### [P3·트리거] 7. cross-layer cache group — xKV/MiniCache/KVSharer/CommonKV 군집
- **개봉 트리거**: 8B+ 모델 온보딩(또는 4K+ ctx 상시화).
- **Description**: layer별 독립 버퍼 + per-layer plan 호출 구조에 레이어 간 조정 채널이 0 — cross-layer cache group(공유 basis 버퍼 + group-level plan 경로) 신설은 ADR급 설계 변경. 대상 전부 training-free post-hoc 가능(xKV 2503.18893 SVD, MiniCache 2405.14366 SLERP, KVSharer, CommonKV)인데 **구조가 차단하는 유일 부류**(인접 layer KV cosine 유사도 0.72–0.87 근거 탄탄).
- **Notes**: 1B/2048에선 KV 절대 크기가 작아 실익 미미(KVSwap 평가와 동일 논리) + 복원 GEMM 오버헤드가 절감 상쇄 위험.

### [P3·트리거] 8. windowed attention TensorKind — SnapKV/PyramidKV/CAKE 군집
- **개봉 트리거**: 항목 3(QueryStats) 경로가 prefill-end 압축 요구를 실측에서 못 덮을 때, 또는 SnapKV류 직접 재현 실수요(논문 비교군 등).
- **Description**: `TensorKind::WindowedAttn`(최근 w query × 전체 key) + prefill-end 캡처. flash attention이라 행렬 materialize가 비쌈 — prefill 마지막 w 토큰 한정 보조 경로(non-flash 또는 부분 재계산) 필요. SnapKV(NeurIPS'24 2404.14469)·PyramidKV(2406.02069)·CAKE(ICLR'25 2503.12491) 해금. layer 차등 budget 자체는 기존 메커니즘(D2O layer allocator)으로 이미 가능.
- **Notes**: 기법 "그대로"의 재현용 — 목적(미래 무용 토큰의 prefill-end 식별)만이면 항목 3이 같은 목적의 2025 frontier 대체 경로.

### [DROP 2026-06-12] 9. 멀티세션 paging — PagedAttention/FlexGen류
- **Status**: DROP (폐기 — 2026-06-12 사용자 결정 B2=NO, Backlog Burndown triage)
- **폐기 근거 (B2=NO)**: "동시 멀티 세션(대화 전환, 에이전트 병행)을 1급으로 지원하는가?" = **NO**. 따라서 allocator paging + 세션 스케줄(ADR급 설계)은 불필요 — 해당 수요는 **항목 5(세션 KV persistence, save/restore 전환)가 단일 캐시 + 디스크 스왑으로 커버**. PM 로드맵 질문에 답이 나옴.
- **재개 트리거**: 제품 방향 전환(멀티세션 1급 지원으로 전환)이 명시되면 재등록 — 현 시점 영구 폐기 아니라 제품 결정 변경 시 부활 가능.
- **원 성격 (아카이브)**: 기법 백로그가 아니라 PM 로드맵 질문. YES → allocator paging + 세션 스케줄 설계(ADR급). NO → 항목 5의 save/restore 전환으로 충분(단일 캐시 + 디스크 스왑).

> **범위 밖 확정(영구 보류, 어느 엔진이든 코어 작업)**: 모델 아키텍처 전환(TransMLA/X-EcoMLA/Mamba류 — "지원할 입력 모델 종류" 문제로 별도), 학습 필요(MatryoshkaKV/MoQAE/NSA/MoBA/CLA/YOCO), 멀티모달(FlowMM/AIM). 상세 판정표(~70개 기법 A/B/C + 근거)는 2026-06-10 세션 기록 — 필요 시 `.agent/research/`로 보존.

---

# Phase β 이관 — 4건 (2026-06-10 등록)

> Phase β(decode loop rewrite, β-1~β-7) 진행 중 발견·이월된 잔여. 출처 = `.agent/todos/roadmap_beta_decode_loop_rewrite_2026_06_10.md`. β-1~6 은 **전부 비차단 사전존재 이슈**, γ-4 는 β-7 G3 ripple 격리용 의도적 이월.

## [P3] test_backend 하네스 이슈 — MatMulTransposed 16 FAIL + MatMulSlice 8 FAIL + RoPE 2 ERROR — 2026-06-10 등록
- **Status**: **RESOLVED (2026-06-13, `55200e8f`)** — T1-5 RoPE(`d7a123b9`) + 잔여 전부 해소. **근본 원인 = 하네스 readback 결함**: S25 production OpenCL 버퍼는 `UnifiedBuffer`(ALLOC_HOST_PTR)인데 하네스가 `OpenCLBuffer` 만 downcast → **조용히 0 벡터 반환** → MatMul `diff=|dot-0|` FAIL, Softmax `diff=|ref-0|` 일부 FAIL (CPU/NEON 은 전 op PASS 라 reference 정확성 교차 입증). UMA-coherent readback(`cl_buffer().read().enq()`) 경로 추가 + 미지원 타입 명시 ERROR(가짜 검증 차단) + Softmax numerically-stable reference 추가(구 `ref_sum=0` 가짜 검증 교체) + 멱등성 결함(softmax^10) 수정. **S25 `--backends opencl` 16+8 FAIL → 51/51 PASS**(MatMulTransposed/Slice/Softmax/RMSNorm/RoPE/KIVI, error ≤7e-6). production .cl/backend 무변경 — 하네스만. host NVIDIA 의 Q4_0 GEMV 6 FAIL 은 `cl_khr_subgroups` 미지원 환경 한계(S25 Adreno 8/8 PASS 로 production 무관 확정).
- **Sprint**: backlog
- **Dependencies**: 없음 (비차단)
- **출처**: roadmap β-7 완료 기록 (`roadmap_beta_decode_loop_rewrite_2026_06_10.md:126`)
- **Description**: S25 실측에서 `test_backend` 하네스가 MatMulTransposed 16건 FAIL + MatMulSlice 8건 FAIL + RoPE 2건 ERROR. **β 무관 사전존재 확정** — pre-β commit `6e52ac50` worktree 빌드로 동일 패턴 재현. production matmul 정상은 β-7 device sig 15/15 MATCH 가 증명(하네스 issue ≠ production 회귀).
- **Acceptance Criteria**: 하네스 reference 산출 또는 비교 tolerance/shape 가정 정정으로 5 op(또는 해당 op) PASS 복구. production matmul/RoPE 무영향 확인(sig MATCH 유지).
- **Notes**: 비차단. production 정확성은 device sig 가 ground truth.

## [P3] INV-LAYER-001/002/003 만성 FAIL — `inv_layer_baseline.json` 재동결 — 2026-06-10 등록
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T1-4)** — 등록 시점 위반 동수(8/3/12)가 stale 이라 현 실측으로 재동결: baseline 7건 → 30건 (INV-001 +8 / INV-002 +3 / INV-003 +14, 전부 pre-existing 아키텍처 의존 — 코드 위반 아님). `cargo test -p llm_rs2 --test spec inv_layer` **8 passed; 0 failed**.
- **Sprint**: backlog
- **Dependencies**: 없음 (비차단)
- **출처**: roadmap β-2 완료 기록 (`roadmap_beta_decode_loop_rewrite_2026_06_10.md:65`)
- **Description**: spec suite 에서 INV-LAYER-001/002/003 만성 FAIL. 원인 = `inv_layer_baseline.json` 미갱신 (위반 동수 8/3/12). **β 무관 pre-existing 입증** — β-1 worktree 대조로 동일 위반 동수 확인. 코드 위반이 아니라 baseline 재동결만 필요한 별도 chore.
- **Acceptance Criteria**: `engine/tests/spec/inv_layer_baseline.json` 의 001/002/003 baseline 을 현 위반 동수(8/3/12)로 재동결 → 3건 spec test PASS 복구.
- **Notes**: 비차단. baseline-only chore (코드 무변경).

## [P3] scripts/check_spec_coverage.sh 버그 — INV-DECODE-STAGE ID 추출 부재 (잔여 1건) — 2026-06-10 등록 / 2026-06-12 축소
- **Status**: **RESOLVED (2026-06-12 T1-6 추출 + 2026-06-13 `2b0caa63` 갭 해소)** — T1-6 에서 INV-DECODE-STAGE 추출 추가(노출된 커버리지 갭 6건). **2026-06-13 갭 0 종결**: 001(orphan phase/Fine variant) + 002/003(폐기 INV tombstone) 신규 테스트 + `004_007.rs`→`004_005_006_007.rs` 확장 rename + 스크립트 파싱 수정(`stage_NNN_MMM` 다중 숫자 토큰 추출 — 구 `grep 'stage_[0-9]+'` 는 첫 숫자만). **INV-DECODE-STAGE 누락 6→0**(전체 51→45). spec 727/0.
- **Sprint**: backlog
- **Dependencies**: 없음 (비차단)
- **출처**: roadmap β-1 완료 기록 (`roadmap_beta_decode_loop_rewrite_2026_06_10.md:51`) + `handoff_beta_1_complete_beta_2_entry_2026_06_10.md:52`
- **2026-06-12 축소 근거**: (1) octal 해석 버그는 `7daa7e69`에서 L78(LAYER)/L90(RPCMEM) 두 곳 `printf '%03d' "$((10#$n))"` base-10 강제로 **해소** (L38 항목 본문 (c) 수정 기록 일치 — [6]까지 전 구간 완주 복구). 잔여 = (2)뿐.
- **Description**: `scripts/check_spec_coverage.sh` 잔여 버그 1건:
  - **(2) INV-DECODE-STAGE 시리즈 ID 추출 로직 부재** — 해당 시리즈가 coverage 추출에서 누락.
- **Acceptance Criteria**: INV-DECODE-STAGE 시리즈 ID 추출 추가 → coverage 정확 산출 (신규 갭 0 확인).
- **Notes**: 비차단. 실제 line 번호는 착수 시 grep 재확인.

## [RESOLVED] γ-4: eval/batch orphan 처분 — `run_eval_ll` + `run_prompt_batch` — 2026-06-10 등록 / 2026-06-11 종결
- **Status**: RESOLVED (2026-06-11) — 처분 방향 합의 = **"기능은 살리고 죽은 포장지만 삭제"**. eval = argus-eval bin 으로 부활(살림, γ-3) / batch = ~1170 LOC 순수 삭제(γ-4).
- **Sprint**: completed
- **Dependencies**: **γ-3(argus-eval/chat bin화)와 처분 방향 조율 선결** — eval 을 bin 으로 살릴지 orphan 삭제할지 결정 후 착수
- **출처**: roadmap β-7 완료 기록 (`roadmap_beta_decode_loop_rewrite_2026_06_10.md:126`) + 후속 섹션 (:144) + `arch/pipeline_stage_design_v2.md` §9 γ-4
- **Description**: `run_eval_ll`(`session/eval/runner.rs:16`) + `run_prompt_batch`(`session/batch/runner.rs:34`) 외부 호출처 0 (β-7 grep 확인). β-7 G3 삭제에서 ripple 격리를 위해 **의도적으로 이월**. Phase γ 의 γ-4 항목에 해당.
- **Acceptance Criteria**: γ-3 방향 확정 후 — (a) eval 을 bin 으로 살리면 entry point 정착 + orphan 해소, 또는 (b) orphan 삭제 시 두 fn + 종속 dead code grep census 후 동반 삭제 + 컴파일 GREEN.
- **적용 결과 (2026-06-11)**:
  - **처분 방향 합의**: "기능은 살리고 죽은 포장지만 삭제" — eval 은 실사용 기능이라 살리고(bin 부활), batch 는 죽은 포장지라 삭제.
  - **eval (살림, γ-3)**: `run_eval_ll` 경로를 `argus_eval` bin 으로 부활 (commit `33d5bc8f`/`11c8721f`).
  - **batch (삭제, γ-4)**: `run_prompt_batch` + batch runner ~1170 LOC 순수 삭제 (commit `2e53cf44`).
- **Notes**: γ-3 과 처분 방향(살림 vs 삭제)이 상호 의존 — γ-3 선행/동행 권장. 담당: Architect(방향 결정) → Implementer(처분). 파생 후속(argus-eval smoke / batch 삭제 2차 census 등)은 아래 "γ-3/γ-4 파생 후속" 그룹 참조.

---

# γ-1 파생 후속 (2026-06-10 등록)

> γ-1(commit `7fe1fe8b`, `pressure/`→`kv/` + `pressure/weights/`→`weight/` rename) 후 발견된 문서 잔여 3건. 단순 anchor 치환으로 끝나지 않는(spec 의미 변경 / 미실현 우산 구조 / 실위치 미반영) 항목만 분리. **출처**: γ-1 문서 anchor 정오 작업(2026-06-10)의 Architect 보존 판단 보고. **참고**: `layer_lint.py` LAYER_RULES 는 γ-1 commit A 에서 이미 `kv`/`weight` 로 갱신 완료(본 그룹 범위 밖).

## [RESOLVED] spec L3 도메인 재정의 — INV-LAYER-003 + §6 도메인 표 + INV-RPCMEM 열거 stale — 2026-06-10 등록 / 2026-06-11 종결
- **Status**: RESOLVED (2026-06-11) — spec 4-도메인 {`kv/`, `weight/`, `inference/`, `qcf/`} 개정 + 동반 코드 이동 완료. 커밋 `08114da8`(refactor) + `65c867ad`(docs).
- **Sprint**: completed
- **Dependencies**: 없음
- **출처**: γ-1 문서 anchor 정오 작업(2026-06-10) Architect 보존 판단 보고
- **Description**: γ-1 rename 후 spec 의 L3 도메인 열거가 stale (실제 = {`kv/`, `weight/`, `inference/`, `qcf/`}):
  - `spec/41-invariants.md` INV-LAYER-003 의 "L3 도메인 = {`inference/`, `pressure/`, `qcf/`}" 열거
  - `spec/01-architecture.md` §6 L3 도메인 표
  - INV-RPCMEM 의 "L3(`models/`, `pressure/`, `inference/`)" 열거
  - **단순 anchor 정오가 아니라 spec 의미 변경** — `pressure/` 단일 도메인이 `kv/`+`weight/` 2 도메인으로 분리된 것을 spec 불변식 정의에 반영해야 함. `/spec-manage` 작업 필요.
- **Acceptance Criteria**: 3 spec 위치의 L3 도메인 열거가 실제 모듈 구조({`kv/`, `weight/`, `inference/`, `qcf/`})와 정합 + spec test/coverage GREEN.
- **적용 결과 (2026-06-11)**:
  - spec 개정: `spec/01-architecture.md` SYS-100/101/104 + `spec/41-invariants.md` INV-LAYER-001/002/003 + ENG-RPCMEM-C04 — 4-도메인 {`kv/`, `weight/`, `inference/`, `qcf/`} 반영.
  - 동반 코드: `weight_swap_handler` → `weight/` 이동 + `ActionResult` §13.8-G L2 격상(`engine/src/action_result.rs` 신설) + `layer_lint` 라벨 L3-kv/L3-weight 분리.
  - 커밋: `08114da8`(refactor) + `65c867ad`(docs).
  - 검증: Tester 독립 검증 PASS (lint 위반 전후 file:line 집합 일치, 신규 위반 0).
- **Notes**: `layer_lint.py` LAYER_RULES 는 γ-1 commit A 에서 이미 갱신됨(본 항목은 spec 문서 측 + 동반 코드 이동). 담당: Architect(/spec-manage).

## [P3] arch 미실현 우산 구조 재작성 — 잔여 `ARCHITECTURE.md` §13.4/§13.6 — 2026-06-10 등록 / 2026-06-11 부분 해소
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T1-2)** — §13.4(정정 노트 + KV/Weight 도메인 행 실 구조 정정) + §13.2 다이어그램 정정 노트 + §13.6(진입점 표 `kv/eviction/`·`kv/*_handler.rs`·`weight/*.rs` 정정, dead `EventSink`/`CacheEvent` 참조 정오) 완료. 부수: §13.8-O 카탈로그 경로 stale(`pressure/weights/`→`weight/`) + γ-1 후속 PENDING 표기 2곳(`08114da8` 으로 기완료)도 동라운드에서 정정.
- **Sprint**: backlog
- **Dependencies**: 없음
- **출처**: γ-1 문서 anchor 정오 작업(2026-06-10) Architect 보존 판단 보고
- **Description**: 다음 arch 위치가 **미실현 계획 구조**를 기재하고 있어 γ-1 단순 치환으로 정합 불가(치환하면 `kv/policy/` 로 여전히 stale):
  - ~~`arch/01-architecture.md` §6.3 의 `pressure/policy/`·`pressure/state/` 우산 서브구조~~ — **완료 (`65c867ad`)**: 실 코드 평면 구조(`kv/` + `weight/`)로 재작성.
  - **[잔여]** `ARCHITECTURE.md` §13.4 확장 가이드 / §13.6 마이그레이션 표·다이어그램
  - 실제는 flat `kv/eviction/` 구조 — 계획했던 우산(policy/state) 층이 실현되지 않음.
- **Acceptance Criteria**: 잔여 `ARCHITECTURE.md` §13.4/§13.6 을 **실제 flat `kv/eviction/` 구조 기준으로 재작성** (미실현 우산 서술 제거 또는 "계획" 명시).
- **Notes**: 단순 rename 치환 금지(stale 잔존). 구조 기술 재작성 필요. §6.3 은 2026-06-11 `65c867ad` 에서 해소됨. 담당 권장: Architect.

## [P3] Sprint C 미반영 weight swap 위치 표기 — Precision Swap 다이어그램 + qcf_taxonomy.md — 2026-06-10 등록
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T1-3)** — ARCHITECTURE.md Precision Swap 다이어그램 라벨(`L3 Inference — models/weights/` → `L3 Weight — weight/`) + Key Components 표 + qcf_taxonomy.md 경로 9곳 `engine/src/weight/` 정합(Sprint C+γ-1 누적 반영). 부수: qcf_taxonomy.md 의 `engine/src/core/qcf/` stale 경로 8곳도 `engine/src/qcf/` 로 정정(2026-05-29 core/ 접두어 제거 반영).
- **Sprint**: backlog
- **Dependencies**: 없음
- **출처**: γ-1 문서 anchor 정오 작업(2026-06-10) Architect 보존 판단 보고
- **Description**: 다음 위치가 weight swap 오케스트레이션을 `models/weights/` 로 표기 — Sprint C(`pressure/weights/`)도 γ-1(`weight/`)도 미반영(2단계 stale):
  - `ARCHITECTURE.md` Precision Swap 다이어그램의 "L3 Inference — `models/weights/` orchestrator" 라벨
  - `docs/qcf_taxonomy.md` weight swap 오케스트레이션 경로 표기
- **Acceptance Criteria**: 두 위치 모두 실위치 `engine/src/weight/` 로 정합.
- **Notes**: Sprint C(`models/weights/`→`pressure/weights/` git mv, `5c698d79`) 와 γ-1(`pressure/weights/`→`weight/`) 두 이동을 누적 반영해야 함. 담당 권장: Architect.

---

# γ-3/γ-4 파생 후속 (2026-06-11 등록)

> γ-3(argus-eval bin화, `33d5bc8f`/`11c8721f`) + γ-4(batch orphan 삭제, `2e53cf44`) 후 발견된 잔여. macOS OpenCL 링크 한계로 런타임 미검증분 + batch 삭제 파생 orphan census + doc stale. **출처**: γ-3/γ-4 구현 후속(2026-06-11) — Tester 후속 목록 + Architect census 보고.

## [P2] argus-eval functional smoke (디바이스/Linux 런타임 검증) — 2026-06-11 등록
- **Status**: **RESOLVED (2026-06-13, Backlog Burndown T6)** — Linux 호스트(CPU + NVIDIA OpenCL)에서 5개 모드 E2E **전부 exit 0 + 유한값**: ll(3항목 choice_nlls) / ppl(ppl=55.25, CPU≡OpenCL 7e-5 차) / dump-importance(16 layer) / qcf-dump modifier(qcf.json 16키) / experiment(17줄 jsonl). caps/swap 보존 = `[Backend] CPU primary, GPU secondary (SwitchHw ready)` 정상. 전체 `cargo test -p llm_rs2` **2334 passed / 0 failed** + spec 722/0 + flash_attn_decode_dk128 1/0·prefill_dk256 3/0. **비차단 관찰**: CPU 실행 시 GPU secondary 마이그레이션 단계의 `.cl` 컴파일 진단 카운트("N errors generated" 본문 없이)가 stderr 노이즈 — 결과 정합성 무영향(전 모드 exit 0), 향후 verbose gate 검토 여지만 기록.
- **Sprint**: backlog
- **Dependencies**: 없음 (디바이스/Linux 환경 필요)
- **출처**: γ-3 구현 후속 — Tester 후속 목록 6건
- **Description**: `argus_eval` bin 은 macOS OpenCL 링크 한계로 **런타임 미검증** (빌드/컴파일만 확인). 디바이스(S25) 또는 Linux 에서 E2E 런타임 검증 필요.
- **Acceptance Criteria** (Tester 후속 6건):
  - 5개 모드 E2E: `ll` / `ppl` / `dump-importance` / `qcf-dump` modifier / `experiment`.
  - legacy generate 등가 — `caps` / `swap_algorithm` 보존 런타임 확인.
  - `cargo test -p llm_rs2` 전체 PASS.
  - spec suite 런타임 PASS.
  - `flash_attn_decode_dk128` + `prefill_dk256` 2 test 타깃 PASS.
- **Notes**: macOS OpenCL 링크 한계가 런타임 검증의 유일 블로커 — 환경만 확보되면 즉시 가능. 담당 권장: Tester.

## [P3] experiments/*.sh argus_eval 이주 — 2026-06-11 등록
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T1-13)** — `run_accuracy_bench.sh` + `run_round1~13.sh` 14개 파일의 `BINARY=` 1줄 교체(`argus_eval`). flag 표면 호환이라 flag 무수정. `run_round13_device.sh`/`run_round14_device.sh` 는 `BINARY=` 패턴 부재(run_device.py 경유)라 비대상. 검증: `argus_eval --help` 정상 (host 모델 부재로 추론 실행 검증은 미수행 — 디바이스 라운드에서 1회 확인 가능).
- **Sprint**: backlog
- **Dependencies**: 없음
- **출처**: γ-3 구현 후속
- **Description**: `experiments/` 의 `run_accuracy_bench.sh` / `run_round*.sh` 가 **삭제된 generate 기준** binary 호출이라 stale. `argus_eval` 로 binary 교체 필요.
- **Acceptance Criteria**: 해당 스크립트의 binary 호출을 `argus_eval` 로 교체 + 실행 검증. **flag 표면은 호환** (argus_eval 이 flag-based dispatch 채택한 이유 = 기존 스크립트 호환) — flag 수정 불필요, binary 이름만 교체 + 동작 확인.
- **Notes**: flag-based dispatch 덕에 이주 비용 최소. 담당 권장: Implementer → Tester(실행 검증).

## [P3] session/warmup.rs::run_warmup orphan 거취 결정 — 2026-06-11 등록
- **Status**: **RESOLVED — 삭제 (2026-06-13, `59737375`)** — 거취 = (b) orphan 삭제. 근거: argus_bench(측정 bin)가 warmup/DVFS ramp-up 없이 운용 → 측정 품질 의존 미입증 + 호출자 0 으로 장기 존속 = 측정 필수 아니었던 방증. `warmup.rs` 모듈 + `session.rs` mod 선언 삭제. orphan import 자동 해소(`forward_fmt_roundtrip` 등 의존 심볼은 타처 사용 보존). 컴파일 GREEN.
- **Sprint**: backlog
- **Dependencies**: 없음
- **출처**: γ-4 구현 후속 — orphan census
- **Description**: `session/warmup.rs::run_warmup` 호출자 0 (legacy DVFS ramp-up warmup). 거취 결정 필요.
- **Acceptance Criteria**: 둘 중 택1 — (a) `argus_bench` 가 채택(측정 품질 위해 DVFS ramp-up warmup 유효) + 호출 배선, 또는 (b) orphan 삭제 + 컴파일 GREEN.
- **Notes**: 측정 품질(DVFS ramp-up) 효용 vs dead code 비용 트레이드오프. 담당 권장: Architect(거취 결정) → Implementer(처분).

## [P3] CommandExecutor legacy 채널 2차 orphan census — 2026-06-11 등록
- **Status**: **RESOLVED — census 완료 + 부분 삭제 (2026-06-13, `59737375`)** — 심볼별 소비자 grep census 결과: ① **`LoopControl.prefill_{chunk,yield,cpu_chunk}` 3필드만 진짜 dead write**(set/clear만·DecodeLoop run() read 0 — prefill 정책은 CLI Args 경로 전용, doc 주석 "batch/runner.rs 소비" stale) → 삭제 + SetPrefillPolicy arm no-op 화. ② **`CommandExecutor::poll`/`apply_command`/`ExecutionPlan`/`EvictPlan`/`StreamingParams` 5종은 프로덕션 소비자 0 이나 테스트 30+ 파일(beta4/test_resilience/spec)이 v1 reference oracle 로 의존** → 단순 orphan 아님, **삭제 보류**(의존 테스트 마이그레이션 선행 필요 — census 범위 밖). ③ heartbeat/EngineReport 경로 보존. `CommandExecutor` struct 자체는 `resilience_init.rs` 생성 → production live(orphan 아님).
- **Sprint**: backlog
- **Dependencies**: 없음
- **출처**: γ-4 구현 후속 — batch 삭제 파생 census
- **Description**: batch 삭제(`2e53cf44`)로 CommandExecutor legacy 채널의 소비자가 0 이 된 심볼 후보: `poll` / `apply_command` / `ExecutionPlan` / `EvictPlan` / `StreamingParams` + `LoopControl.prefill_*` 3필드. **단 `heartbeat`/`EngineReport` 는 잔류** — 일괄 삭제 불가. 정밀 census 후 삭제 범위 확정 필요.
- **Acceptance Criteria**: 후보 심볼별 소비자 grep census → 진짜 orphan 만 삭제 (heartbeat/EngineReport 잔존 경로 보존) + 컴파일/clippy GREEN.
- **Notes**: heartbeat 경로가 일부 심볼을 살리고 있어 일괄 삭제 위험 — 심볼 단위 census 필수. 담당 권장: Architect(census) → Implementer(삭제).

## [P3] test_inv_layer_005.rs doc 헤더 stale — 2026-06-11 등록
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T1-1)** — doc 헤더를 `L5_PRODUCTION_BINS` 기준(argus_cli/argus_bench/argus_eval)으로 정정, generate.rs 폐기 커밋(d5ed71d2) 사유 명시. 주석 only — 코드 동작 무변경.
- **Sprint**: backlog
- **Dependencies**: 없음
- **출처**: γ-3/γ-4 구현 후속
- **Description**: `test_inv_layer_005.rs` 주석 헤더가 "generate.rs 한정" 잔존 — 실제 lint 동작은 `L5_PRODUCTION_BINS` 로 기확장됨(commit `33d5bc8f`). 주석만 stale.
- **Acceptance Criteria**: doc 헤더 주석을 `L5_PRODUCTION_BINS` 기준 동작으로 정정 (코드 동작 무변경 — 주석 only).
- **Notes**: lint 동작은 정상(33d5bc8f). 주석 정오만. 담당 권장: Implementer.

## [P2] QCF estimate 역방향 IPC v2 재배선 — 2026-06-11 등록
- **Status**: **RESOLVED (2026-06-11 당일)** — 설계 `bf6230e8`(arch v2 §5.8) + 구현 `226d154b`(dispatcher 직결 `compute_and_send_qcf` + report_tx 주입 + `LoopControl.request_qcf` 삭제 + v1 lift-and-shift→qcf_runtime.rs) + device-only KV read-back fallback `e267bd50`. S25 실증: thermal f16 GREEN, manager `Engine QCF estimate: N actions` 수신 확인.
- **Sprint**: backlog
- **Dependencies**: 없음
- **출처**: AB-5 S25 verify 매트릭스 — `signal_thermal_critical_throttle` f16/q4 FAIL의 직접 원인 (Tester triage 2026-06-11)
- **Description**: β-7 DecodeLoop 재작성이 v1 report slot(IPC 송출: capability/qcf/swap_report)을 제거한 뒤, capability(connect 시)·swap_report(AB-6 §5.6.6 report_tx)는 재배선됐으나 **QcfEstimate 송출만 미재배선 잔여**. `CommandDispatcher`(:309)가 `request_qcf` 플래그만 set하고 소비처 부재 → QCF-기반 LuaPolicy의 2단계 결정(ThermalAlert→RequestQcf→QcfEstimate→throttle)이 manager 측 `QCF estimate timeout 1.0s`로 끊김. v1은 `compute_qcf_estimates`→`executor.send_qcf_estimate` 경로가 live였음(legacy generate L4373-4388, d5ed71d2^).
- **Acceptance Criteria**: v2 경로에서 RequestQcf 수신 시 `compute_qcf_estimates` 산출을 `QcfEstimate` IPC로 응답 (AB-6 report_tx 선례 동형 배선) + `signal_thermal_critical_throttle` f16/q4 S25 verify GREEN.
- **Notes**: 직접 directive 경로(Suspend/Offload/Quant/TargetTbt/Partition/Swap)는 전부 정상 — 영향 범위는 QCF 2단계 정책 경로 한정. verify 매트릭스의 known-fail 2건이 이 항목의 회귀 게이트 역할. 담당 권장: Architect(배선 설계) → Implementer.

## [P2] QCF 2-step 핸드셰이크 신호 유실 — timeout 폴백 부재 + late estimate 미캐시 (2026-06-12 등록)
- **Status**: **RESOLVED (2026-06-12, Backlog Burndown T2)** — AC ①~④ 전부 충족 (`a1d43634` + verify 시나리오 커밋). ① `check_qcf_timeout` timeout 시 pending_signal 로 무-QCF 폴백 decide(`decide_and_build_directive` 직호출 — RequestQcf 재발행 구조적 차단) ② `complete_qcf_selection` cache 갱신을 early-return 앞으로(late estimate 도 캐시 반영, INV-117 보존) ③ SEQ-098 개정 + SEQ-098a 신설(spec/12 §3.10) + 회귀 테스트 2종 ④ 신규 시나리오 `signal_memory_critical_prefill`(prefill 0.8s 주입) S25 f16/q4 **PASS** — manager.stdout 에 "QCF estimate timeout (1.0s) — falling back to no-QCF decide" + "QCF timeout fallback directive seq=2" 발화 입증(구 거동 = directive 0건 유실). 결함은 LuaPolicy 한정(HierarchicalPolicy 기존부터 무결 — spec 테스트 사각지대였음). verify 매트릭스 **32/32**(기존 30 무회귀 + 신규 2) + α-K frozen 3-dtype byte-identical·tbt Δ≤+0.9%. sim 스냅샷 4종 갱신(폴백 발화로 relief 관측 분포 미세 변동 — 정당 파생).
- **Sprint**: backlog
- **Dependencies**: 없음 (QCF_kv 설계 라운드에서 적발, 라운드 범위 밖 처분)
- **출처**: 2026-06-12 QCF_kv 라운드 P5 — `signal_memory_critical` medium_qa 재정렬 중 S25 실측: 신호가 prefill 중 도착하면 엔진 QcfEstimate 응답이 chunk-경계 poll 간격으로 ~3s 지연 → manager `QCF_TIMEOUT_SECS=1.0` 초과.
- **결함 2층** (`manager/src/lua_policy.rs`):
  1. **timeout 폴백 부재** (`check_qcf_timeout` :893): pending 만 clear 하고 `qcf_pending_signal` 로 무-QCF decide 를 실행하지 않음 → **압박 신호가 통째로 유실** (directive 0건). "stale 이라도 없는 것보다 낫다"는 cache 보존 주석의 정신이 signal 에는 미적용.
  2. **late estimate 미캐시** (`complete_qcf_selection` :871): `qcf_pending_at.take()?` 가 cache 갱신보다 앞에 있어 늦게 도착한 estimate 는 qcf_cache 에도 반영 안 됨 (지식 폐기).
- **Acceptance Criteria**: ① timeout 시 pending_signal 로 decide 실행(무-QCF, INV-117 cache-miss=0 의미 유지) ② late estimate 는 pending 여부와 무관하게 cache 반영 ③ spec SEQ-098(timeout 거동) 개정 동반 + `manager/tests/spec/test_seq_095_098.rs` 갱신 ④ prefill-중-주입 변형 시나리오로 S25 검증.
- **Notes**: 현 verify 매트릭스는 decode-구간 주입(`signal_memory_critical` delay 9.0s)으로 우회 중 — 본 항목은 prefill-구간 주입의 liveness 복원. spec 거동 변경이라 Architect 선행 필수.

## [P3] weight swap 역전 (RestoreDefaults F16 recall) — 2026-06-11 등록 (ADR-0006 §6 Deferred 잔존분)
- **Status**: **RESOLVED (2026-06-13, Backlog Burndown T5 — 사용자 결정 D-1 = B 명시 트리거 recall, 구현 완료)** — 설계+spec `afeec446`(MSG-043 `RecallWeights{ratio}` / SEQ-065~067 / ENG-ALG-240·241 / INV-192~195 / INV-126 방향별 완화 / ENG-DAT-097 단방향 가정 개정) + 구현 `fa06b824`(open_secondary_f16_for_recall — ReverseSwapRejected 우회·SOA/F16부재 거부 유지, run_layer_recall — SwapExecutor target=F16 재사용, WeightRecallStage OneShot — lazy-once F16 view 캐싱 + loud no-op 5종, dispatcher arm — **RestoreDefaults 는 평시 no-op 유지**). **게이트**: host lib 1438/0 + spec 722/0(INV-192~195 16/16) + verify 32/32 무회귀 + α-K frozen 3-dtype byte-identical / **e2e GREEN(host cpu + S25 cpu)**: mock_manager 시나리오로 SwapWeights 1.0(26 layers F16→Q4_0 Incremental) → RecallWeights 1.0 → `[WeightRecall] Done: recalled=26`(S25 latency 1515ms) → 완주+정상 텍스트. **기록**: ① multi-dtype AUF 신규 빌드 필요했음 — 기존 디바이스 AUF 자산들은 embed 가 Q4_0-only entry 라 F16 primary 로드 불가(`gather: unsupported src dtype Q4_0`), `auf_tool build --dtypes f16,q4_0`(F16 GGUF 소스)로 재빌드한 `qwen2.5-1.5b-md2.auf` 는 정상 ② argus_cli(standard happy path)는 swap runtime 미구성이라 Swap/Recall directive 수신 시 의도된 no-op(사전존재 — swap 은 experiment/argus_bench 경로 전용 구성) ③ manager 자동 트리거(품질 메트릭 기반)는 후속 — 이번 범위 = 메커니즘 + 수동/시나리오 발화.
- **Sprint**: backlog
- **Dependencies**: 없음 (단, 착수 시 ADR-0006 §5 Risk "RestoreDefaults swap 역전 = 신규 메커니즘" 정독)
- **출처**: ADR-0006 §6 Deferred. 2026-06-11 "ADR-0006 Deferred 진행" 세션에서 hook 실배선(§5.9.2)은 해소했으나 swap reversal은 **의도적으로 범위 제외** — `SwapExecutor` 단방향(F16→Q4_0, INV-126 Q4_0-only validate swap_runtime.rs) + F16 recall 코드 전무 + report `from/to_dtype` 하드코딩으로, 역전은 신규 메커니즘(F16 recall/dual-resident + Consumed된 OneShot의 역방향 재기동) 설계가 필요하다.
- **Acceptance Criteria**: RestoreDefaults 수신 시 swap된 layer가 F16으로 복원 + 복원 후 출력이 swap-전 happy baseline과 일치 (S25 device 게이트) + partition+precision 합성 역전(ADR-0006 §6) 처분 결정.
- **Notes**: 현 production에서 RestoreDefaults는 swap에 한해 no-op(§5.6.4 — 의도된 동작, partition/quant guard clear와 비대칭). 우선순위 P3 근거: production winner(Incremental swap)는 압력 해소 후에도 Q4 유지가 메모리 이득 관점에서 보통 바람직 — 역전의 실수요(품질 회복 시나리오)가 구체화되면 P2 승격.

## [RESOLVED 2026-06-12] QCF_kv 정규화 비대칭 + estimator 우회 + manager floor 재설정 — 설계 라운드
- **Status**: **RESOLVED (2026-06-12 라운드 완주)** — AC ①~⑤ 전부 충족. 결정: ① estimator = **B raw 직송 합법화**(ENG-ALG-050 step 4 개정 `eab908ab`) ② QCF_FLOOR = **A 재설정**(S25 66샘플 실측 → 0.10/0.25/0.50/huge, policy v2.5.0 `1a8ee375`). 수식 정규화 `753fb257`(0.985 포화 → 0.03~0.39 변별 + d2o 재보정 + ENG-ALG-051 spec 테스트 3종) + 키 정렬 `ddc2c2fb`/`d6fcf376`. **라운드 중 추가 적발·해소 2건**: ④ heartbeat available_actions 결함(`09a82ad9` — eviction_policy 미전파로 capability 12→3 덮어쓰기, kv 후보 전멸) ⑤ 시나리오 물리 불가 파라미터(`7397512c` — MIN_EVICT_TOKENS·emergency 압력·prefill 주입). **게이트: `signal_memory_critical` f16/q4 GREEN + verify 매트릭스 30/30 봉인(20260612_165122) + α-K frozen 3-dtype byte-identical·tbt Δ≤+0.9%**. 잔여 신규 등록: "QCF 2-step 핸드셰이크 신호 유실" 항목(아래). 스프린트: `sprint_qcf_kv_design_round_2026_06_12.md`. (구 Status: 라운드 실행 중. Architect 사전 판정: 3층 정합성 결함이라 폐쇄 수정 불가 — 실제로는 5층이었음)
- **Sprint**: backlog
- **Dependencies**: 없음 (본 항목의 Architect 분석이 안건 SSOT — 2026-06-12 세션)
- **출처**: 2026-06-12 score accumulator 배선 세션 — `signal_memory_critical` known-fail의 최종 잔여 원인으로 적발. S25 실측: V-readback 수정(`4444bdc8`) 후 QCF_kv가 0.985로 포화 → policy quality floor(critical=0.90)가 모든 kv.evict 기각.
- **3층 결함 (Architect 분석)**:
  1. **수식 비대칭**: `qcf_kv.rs:217~232` o_before = raw Σα·V vs o_after = (α/Σα) 정규화 — 스케일 불일치로 QCF 포화. **spec ENG-ALG-051(spec/32 L591~600)에 동일 비대칭이 명문화**돼 있고 spec 불변식 `QCF ∈ [0,1]`(L630)과 자기모순. 올바른 형태는 유일(o_before에도 α/Σ_all α 정규화 — o_after의 잔존 Σα 재정규화는 이미 정확). 수정 시 QCF 0.08~0.22 정상화 S25 실측됨.
  2. **estimator 우회**: `qcf_runtime.rs:981~986`이 raw QCF를 DegradationEstimator 환산 없이 직접 IPC 송출 — spec ENG-ALG-050 step 4(L688) 위반. raw 전송+manager측 환산 vs 엔진측 ΔPPL 환산 택일이 **floor 의미를 좌우하는 미결 설계 결정** (estimator 자체는 명목 linear(1.0) 기본값 — 보정 의존 없음).
  3. **manager floor 정렬**: `policy_default.lua:69~73` QCF_FLOOR 4임계(0.30/0.60/0.90/huge) + V_Q=0.5(L53)가 현 포화 스케일에 정렬 — 수식 수정 시 재설정 + verify 매트릭스 재검증 필수.
- **동반 재보정**: `test_d2o_less_than_h2o`(qcf_kv.rs:1156~1240) — 계약은 순서 관계(D2O<H2O)로 정규화-중립이나 테스트 데이터(`make_v_data` t-단조 V, head_dim=4)가 정규화 공간에서 순서가 뒤집힘 → 실분포 근사 데이터(head_dim≥32, 비단조 norm)로 교체 + 측정 확인. paper 절대값 노출 없음(grep 전수).
- **2026-06-12 검토 라운드 추가 발견 (estimator 실태)**:
  - DegradationEstimator(`engine/src/qcf/estimator.rs` — piecewise-linear + EMA 보정 + JSON 캘리브레이션 로드)는 **완성돼 있으나 live 경로에 미장착** — qcf_runtime 어디에서도 인스턴스화/호출 0. "완성된 부품 미장착"이 spec 위반의 실체.
  - `estimate()`는 곡선 미등록 액션에 `linear(1.0)` 항등 fallback(:137~141), `with_defaults` 기본 곡선도 전 액션 항등 → **장착해도 당장 수치 변화 0 (전환 무위험)**. 결정의 본질은 IPC 값의 단위 선언 + 향후 캘리브레이션 경로.
  - **곡선 키 불일치 잠복**: `with_defaults` 키(`kv.eviction`/`kv.sliding`/`kv.kivi`/`weight.skip`) ≠ IPC estimates 키(`kv.evict_h2o`/`kv.evict_sliding`/`kv.merge_d2o`/`kv.quant_dynamic`/`weight.skip`). 항등 fallback이라 현재 무해하나 **캘리브레이션 곡선 등록 시 silent 미적용** (fixture relief silent-0 동일 클래스). manager `context.rs:173`의 `kv_streaming` 언더스코어 표기도 감사 대상. → 방향 결정과 무관하게 **키 전수 감사를 라운드에 포함**.
  - 포화 메커니즘 실측 보강: ‖o_before‖=201 vs ‖o_after‖=2.9 (S25 qwen2.5-1.5b, pos=1045, 누적 Σα≈12) → QCF ≈ 1−2.9/201 ≈ 0.985. 분자를 스케일 차이가 지배해 액션/eviction 폭 무관 동일값.
  - floor 부수 효과 확인: 수식 수정만으로 `signal_memory_critical` 게이트는 GREEN 예상(0.08~0.22 < critical 0.90)이나, 현행 QCF_FLOOR(normal 0.30 포함)는 신규 스케일에서 **영구 미발동 dead config**가 됨 — 재설정은 게이트가 아니라 의미 복원 차원. `weight.skip`(QCF_weight, 다른 측정 공간) 분포와 함께 재산정해야 함.
- **사용자 결정 2건 (✅ 확정 2026-06-12: 1 = B raw 직송 합법화 / 2 = A floor 재설정 포함)**:
  1. **estimator 우회 해소 방향**: (A) 엔진측 ΔPPL 환산 — estimator 장착, spec 무변경, 가족 간 비교 합법화, 향후 캘리브레이션 시 manager 무변경 (권장) vs (B) raw 직송 합법화 — 코드 무변경, spec ENG-ALG-050 step 4 개정, 가족 간 raw 비교 고착. **floor의 단위가 이 결정을 따름** (1번 확정 후 2번 값 산정이 한 번에 끝남).
  2. **QCF_FLOOR 재설정**: (A) 이번 라운드 포함 — 수정 후 S25 분포 실측(양 가족) → 4임계+V_Q 재산정 → policy_default.lua 반영 → 매트릭스 30 재검증 봉인 (권장; directive 변동 리스크 있음) vs (B) 현행 유지 — dead config 잔존(V_Q soft 페널티는 계속 작동), 분포 실측 후일 이중 작업.
- **Acceptance Criteria**: ① 수식+spec(ENG-ALG-051)+docs(qcf_taxonomy §2.1/§2.2.1) 3소스 동기화, ② estimator 우회 방향 결정+구현, ③ floor 재설정, ④ d2o 테스트 재보정, ⑤ `signal_memory_critical` f16/q4 S25 verify GREEN (이 시나리오가 회귀 게이트).
- **독립 보강 실측 (2026-06-12, KV roadmap 항목 0 스프린트)**: 품질 eval 경로에서 `qcf_sum` ↔ PPL **역전** 관찰 — sliding 0.60 vs h2o 0.018인데 실 PPL은 sliding 우세. 본 항목의 수식 포화 결함(o_before 비정규화 → QCF 0.985 포화)을 IPC/verify와 무관한 독립 경로에서 재확인. 리포트: `experiments/kv_roadmap_item0/rerun/REPORT.md` 부수 관찰 + `.agent/todos/sprint_kv_roadmap_item0_2026_06_12.md` §부수 관찰.
- **참고**: v1(4월)의 동 시나리오 PASS는 fixture relief 키 깨짐(전부 0)에 의존한 우연이었음이 git 포렌식으로 확정 — v1도 동일 수식 결함 보유.

## [RESOLVED(배선)/이관] argus_bench AttentionScoreAccumulator 배선 (AB-1 잔여) — 2026-06-11 등록 / 2026-06-12 처분
- **Status**: **배선 RESOLVED (2026-06-12)** — AC 중 "signal_memory_critical GREEN"만 미달. S25 트레이스로 score 체인 전 구간(누적 ws.scores.sum=12.0 → dispatcher 전달 n=2048/nonzero=132 → QCF estimate h2o/d2o 포함) 정상 입증 후, 잔여 원인이 별개 결함(QCF_kv 수식 포화)으로 재진단되어 위 "QCF_kv 정규화 비대칭" 설계 라운드로 이관.
- **완료분**: ① accumulator 생성·forward 장착(`3b241d11`, arch v2 §5.9.1 공유 cell — eviction-시 ScoreContext 공급+reset 포함), ② capability 송출 동적화(`274472b6` — kv.evict_* 포함 12 actions S25 실측), ③ fixture relief 키 정정(`b65690af`), ④ rpcmem stale V-readback 강제(`4444bdc8` — GPU KV는 read_buffer 필수, as_ptr=stale cache). happy-path 무회귀: α-K frozen 3-dtype 비트 일치 + TBT Δ≤+0.53%, 전체 매트릭스 신규 회귀 0.
