# Backlog — 미배정 작업

> 역할이 배정되지 않은 작업 대기열. PM이 우선순위 판단 후 역할별 TODO로 이동.

---

## [P2] llm_rs2 lib clippy 회귀 — doc_lazy_continuation 29건 + unsafe pointer 1건 — 2026-05-18 등록 (Phase 4-D 게이트 도중 발견)
- **Status**: TODO (Phase 4-D scope 외, 게이트 통과 위해 일시 우회)
- **증상**: `cargo clippy -p llm_rs2 --lib -- -D warnings` 실패. 30개 deny 처리됨.
- **위치**:
  - `engine/src/core/qcf/layer_importance.rs:310, 394`
  - `engine/src/models/weights/async_swap.rs:253`
  - `engine/src/models/weights/noise_table.rs:627, 750, 754, 784, 939, 954, 962, 966`
  - `engine/src/models/weights/phase_aware_swap.rs:599`
  - `engine/src/models/weights/swap_executor.rs:652, 661, 675`
  - `engine/src/session/chat/repl.rs:156`
  - `engine/src/session/cli.rs:852, 853, 854, 855, 899, 900, 903, 904, 905`
  - 1건 `this public function might dereference a raw pointer but is not marked unsafe`
- **원인**: 최근 PR들이 doc 추가 시 lazy_continuation 회피 미적용. clippy gate가 통과 안 되는 master 상태에서 새 PR 머지가 누적됨.
- **수정**: `///`의 multi-line continuation을 indent 4-space로 정렬. unsafe pointer는 `unsafe fn` marking 또는 `&self` 변경.
- **영향**: 4-D 가 sanity gate 일부를 `cargo check --bins`로 좁힘. 본 backlog 해결 후 `cargo clippy --workspace --bins -- -D warnings` 게이트 복원 가능.

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

# Weight Swap Overhead 감축 (EuroSys 2027 critical path) — 2026-05-07 신규

> 측정 보고: `/home/go/Workspace/papers/eurosys2027/_workspace/experiment/swap_overhead_s25.md`
> Galaxy S25 단발성 stall 1564.6 ms → 목표 ~70 ms (95.5% 감축).
> 6개 finding (A~F) + 보조 1개 (eager prefault). spec-manage가 부여할 ID 컨벤션: `WSWAP-6-A` ~ `WSWAP-6-F`, `WSWAP-6-PREFAULT`.
> EuroSys 2027 paper critical path이므로 기존 backlog의 [P2] QCF rename / [P0] Weight Swap Phase B 스프린트 항목보다 **Sprint 우선권 상위**에 배치 (P0/P1).
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
- **Status**: TODO (결정만 완료, 코드 미적용)
- **Sprint**: backlog
- **Dependencies**: 없음 (rename 단순 작업이지만 IPC schema 영향 검토 필요)
- **Description**: 현재 `unified_qcf`(KV 액션 5종)와 `compute_qcf_swap`(weight)이 이름상 같은 QCF지만 측정 공간이 다르다(`‖ΔO‖/‖O‖` vs `Σ imp×ε / Σ`). 통일이 불가하다고 판단되어 **이름으로 패밀리를 분리**. CLAUDE.md "QCF 명명 컨벤션" 섹션에 결정 기록됨 (2026-04-27).
- **결정 사항**:
  - **QCF_kv**: sliding/H2O/streaming eviction, KIVI quant, D2O merge (구 `unified_qcf`)
  - **QCF_weight**: weight swap (F16→Q4), layer skip (SWIFT)
  - 두 패밀리는 직접 비교 불가, cross-action 비교는 `DegradationEstimator` ΔPPL 환산 경유
- **Acceptance Criteria**:
  - `core/qcf/unified_qcf.rs` → `kv_qcf.rs`로 rename
  - 함수/타입 rename: `compute_unified_qcf` → `compute_kv_qcf`, `UnifiedQcfParams` → `KvQcfParams`, `QcfActionType` → `KvQcfAction`
  - swap/skip 함수에 `weight_` 접두 추가: `compute_weight_swap_qcf`, `compute_weight_skip_qcf`
  - `DegradationEstimator` action key 컨벤션 정리: `kv.evict_h2o`, `kv.quant_kivi_q4`, `weight.swap_q4`, `weight.skip` 등 `family.action` 2단계
  - `shared/src/lib.rs::QcfEstimate.estimates` HashMap key prefix 통일 (구조체 이름·필드는 IPC 호환성 위해 유지)
  - `docs/USAGE.md`, `docs/layer_swap_qcf_measurement.md`, `arch/`, spec 테스트의 용어 갱신
  - 기존 spec 테스트 PASS 유지 (rename만이라 동작 변경 없음)
- **담당 권장**: Architect(spec key 정규화) + Implementer(rename + 테스트)
- **Notes**:
  - layer skip 자체의 QCF는 이미 `layer_importance.rs::compute_qcf`에 존재. weight 패밀리로 묶기.
  - 별도 후속 검토(별 backlog): swap의 ε_i를 `‖W_F16·x − W_Q4·x‖ / ‖W_F16·x‖`로 재정의하면 QCF_kv와 같은 분자/분모 모양이 됨 → cross-family raw 비교 가능. 본 rename과는 독립.
- **작성일**: 2026-04-27

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

## [P1] QcfEstimate 메시지 + RequestQcf 커맨드 구현
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  MSG-014: EngineMessage에 QcfEstimate variant 없음.
  MSG-036b: EngineCommand에 RequestQcf variant 없음.
  Manager가 Critical 모드에서 QCF 비용 기반 액션 선택을 못 함.

  필요 구현:
  1. shared/src/lib.rs — QcfEstimate 구조체 + EngineMessage variant 추가
  2. shared/src/lib.rs — EngineCommand::RequestQcf variant 추가
  3. Engine — QCF 계산 후 QcfEstimate 전송 로직
  4. Manager — Critical 진입 시 RequestQcf Directive 발행 + 1초 타임아웃 수신
- **Acceptance Criteria**: Manager가 Critical 모드에서 RequestQcf → QcfEstimate 수신 → Lossy 액션 cost 반영
- **Notes**: 프로토콜 레벨 변경이므로 Architect spec 검토 필요. SEQ-090~098 참조.

## [P2] Manager 페이로드 크기 가드 추가
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  PROTO-012: Engine측은 64KB MAX_PAYLOAD 검증 구현 완료.
  Manager측(unix_socket.rs, tcp.rs)의 read_engine_message()에 페이로드 크기 검증 없음.
  악의적/버그 Engine이 거대 페이로드를 보내면 OOM 위험.
- **Acceptance Criteria**: Manager가 64KB 초과 메시지를 거부하고 연결 유지
- **Notes**: 소규모 변경. manager/src/channel/unix_socket.rs:311, tcp.rs:299.

## [P2] Heartbeat/Response 타임아웃 구현
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 없음
- **Description**: |
  SEQ-087: Manager가 Engine Heartbeat 부재를 감지하지 못함 (권장 3초).
  SEQ-088: Directive 후 Response 무한 대기 가능 (권장 500ms).
  Engine 장애 시 Manager가 대응 불가.
- **Acceptance Criteria**: Heartbeat 3초 미수신 시 Disconnected 전이. Response 500ms 초과 시 타임아웃 처리.
- **Notes**: 타이밍 상수는 Config로 설정 가능하게.

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
