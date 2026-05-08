# Handoff: Adreno OpenCL Weight Swap Overhead — Exhaustive Negative Exploration

**작성일**: 2026-05-08 (extended session)
**작성자 세션**: 9~10 swap track 측정 후 user가 새 세션에서 다른 접근 시도 위해 정리 요청
**기준 commit**: `0c8951e` (LISWAP-4 v1 commit) + 미커밋 변경 ~10개 ablation 트랙

---

## 0. 한 줄 요약

Galaxy S25 Adreno 830에서 **weight swap H2D를 forward GPU compute와 overlap시키는 9개 SW 트랙을 모두 측정**, 모두 negative 또는 nuanced. driver-internal command queue 직렬화가 본질이며, OpenCL 표준/벤더 확장 어떤 mechanism으로도 우회 불가능.

핵심 finding 두 가지:
1. **Per-token TBT inflation은 op 안이 아니라 op 사이 "gap"에서 발생** (driver queue scheduling)
2. **Long decode (n≥1000)에서 모든 swap mechanism 동일 wall-clock 수렴** — production에선 TTFT 우선 결정

---

## 1. 측정 9 트랙 결과 표

| 트랙 | 환경 | Decode TBT vs sync | 정확성 | 결론 |
|------|------|-------------------:|:------:|------|
| sync (Phase 6.5) | baseline | 26.7 ms (ref) | 5/5 | **production main** |
| LISWAP-1 (per-tick=1) | `--swap-incremental-per-tick=1` | +5~12 ms | 5/5 | TTFT-우호 trade-off |
| LISWAP-2 (async dispatch) | `--swap-async-dispatch` | 0% | 5/5 | multi-queue serialize |
| LISWAP-3 (zero-copy ALLOC_HOST_PTR pool) | `--swap-zero-copy --swap-pool-slots=N` | -3.6% saving (noise) | 5/5 | minimal effect |
| LISWAP-4 v1 (intra-forward, dummy event) | `--swap-intra-forward` | -202% (broken) | 5/5 | dummy event × OpenCL safety fall-through |
| LISWAP-4 v2 (real cl_event via `Arc<GpuEvent>`) | + wait gate fix | -88% | 5/5 | hook이 dead code였음 (plan path 우회) |
| LISWAP-4 v3 (plan path 우회) | + `gpu_plan` build guard | -65% | 5/5 | swap이 token 0 안에 직렬 합산 |
| LISWAP-1+3 (combo) | LISWAP-1 + zero-copy | -11% | 5/5 | ALLOC_HOST_PTR Map/Unmap 비용 |
| LISWAP-1+3 v2 (skip clFinish) | + `LLMRS_HOST_PTR_SKIP_FINISH=1` | -8% than v1 | 5/5 | unmap 누적, 더 나쁨 |
| **PreStage v2 (BG thread)** | `--swap-pre-stage` | **5/5 SIGABRT** | crash | Adreno driver thread-unsafe (`libCB`) |
| **DMA-BUF heap zero-copy** | `LLMRS_OPENCL_DMABUF_HEAP=1` | identical to LISWAP-3, garbage (no sync) | garbage 0/3 | DMA-BUF cache coherency 미보장, 비용 동일 |
| **OoO queue** | `LLMRS_OPENCL_OOO_QUEUE=1` | 0~-43% (worse) | 5/5 | driver가 OoO 무시 또는 보수 적용 |
| **Multi-context** | `LLMRS_OPENCL_SWAP_CONTEXT=1` | nuanced (n=200 +30%, n=1000 ≈0%) | 5/5 | per-tick swap -7%지만 forward 인플레 동일, **변동 큼** |

### 부가 결과 (decode 길이 sweep, n=5)

| n_tokens | sync | LISWAP-1 | multictx | 결론 |
|----------|-----:|---------:|---------:|------|
| 50 | 1.95s | 1.99s (+1.9%) | 2.05s (+5.2%) | sync 미세 우세 |
| 200 | 7.78s | 8.83s (+13.5%) | 9.73s (+25%) | sync 우세 |
| 500 | 23.7s | 25.7s (+8.5%) | 24.9s (+5.2%) | sync 우세 |
| **1000** | **61.6s** | **61.4s (-0.2%)** | **61.8s (+0.3%)** | **모두 동일** |

→ **Long decode에서 swap mechanism 선택은 wall-clock에 영향 없음**. TTFT만 차이 (-200~300ms로 incremental 우세).

---

## 2. 진짜 Root Cause (op-level profiling으로 확정)

`LLMRS_FORWARD_GEN_OP_TRACE=async/sync` op-level 측정 결과:

- async mode: trace coverage 22% → 8% (LISWAP-1 swap-active 시), gap +14.5 ms
- sync mode: forced synchronize가 op에 wait 흡수, gap 0
- Per-op time: **단일 op의 GPU kernel exec time 증가가 아님**
- `matmul_ffn_gate_up`만 +114 us/call (sync mode), 이는 driver의 implicit fence wait가 그 op에 흡수된 결과

**Adreno GPU command processor는 FIFO single-issue. cl_mem object identity, queue 분리, sync 호출 명시 여부와 무관하게 commands를 받은 순서로 직렬 실행.** SW layer에서 우회 불가능.

### Multi-context의 흥미로운 부수 finding

- per-tick swap 자체: -7% (multi-context가 swap 작업을 빠르게 함)
- forward 인플레: 동일 (multi-context도 forward는 swap 완료를 wait)
- **driver는 per-context로 schedule하지만 cross-cl_mem dependency는 device-wide enforce**

`DMABUF_SYNC` 추가 시 multi-context 모드 -14% 빠름. driver의 implicit cache management 우회 가능성. 단 n=3 σ=7-8로 신뢰구간 큼.

---

## 3. 미커밋 working tree 변경 (정리 권장 plan)

### 3.1 REMOVE (dead code / debug only)

#### `engine/src/bin/generate.rs`
- **`--swap-pre-stage` CLI flag** + 관련 BG thread spawn 코드 (PreStage v2 → 5/5 SIGABRT)
- **DIAG 로그**: `[IntraForwardSwap-DIAG] token=N pending_layers=...` (debug only, production noise)
- 관련 state 변수: `pre_stage_pending_layers`, `pre_stage_bg_handle`, `pre_stage_spawn_at`

#### `engine/src/backend/opencl/mod.rs`
- `[DMABUF-DEBUG]` 진단 로그 (`fill_dmabuf_via_swap_queue` 안의 `LLMRS_DMABUF_DEBUG` block) — multi-context 디버그 완료 후 remove

### 3.2 KEEP (env-gated, default OFF, paper ablation 재현용)

#### LISWAP-4 wait gate fix (이미 v1 commit `0c8951e`에 있음)
- `Arc<GpuEvent>` shared between dispatcher + forward thread
- `engine/src/models/weights/intra_forward_swap.rs::arm_pending(idx, event: Arc<GpuEvent>)`
- `engine/src/models/weights/async_swap.rs::SwapCommitJob.write_event: Arc<GpuEvent>`

#### Plan path bypass guard (필수)
- `engine/src/bin/generate.rs:5121` `gpu_plan` build에 `&& !args.swap_intra_forward`
- LISWAP-4 hook 작동 위해 필수

#### DMA-BUF heap infrastructure
- `engine/src/backend/opencl/mod.rs::alloc_dmabuf_heap_buffer()` (single-context)
- env: `LLMRS_OPENCL_DMABUF_HEAP=1`

#### Multi-context infrastructure
- `engine/src/backend/opencl/mod.rs::swap_context_or_init()`, `swap_queue_or_init()`, `alloc_dmabuf_with_swap_context()`, `fill_dmabuf_via_swap_queue()`
- `engine/src/backend/opencl/host_ptr_pool.rs`: `HostPtrPoolEntry::swap_ctx_mem`, `HostPtrPoolGuard::swap_ctx_mem()`, `HostPtrPoolGuard::dmabuf_fd()`, `HostPtrPoolGuard::dmabuf_host_ptr()`
- `engine/src/models/weights/swap_executor.rs::try_pool_materialise()` 3-way 분기 (multi-ctx / single-ctx / ALLOC_HOST_PTR fallback)
- env: `LLMRS_OPENCL_SWAP_CONTEXT=1`, `LLMRS_DMABUF_SYNC=1`

#### OoO queue env-gate
- `engine/src/backend/opencl/mod.rs:680` 부근 main queue + transfer queue
- env: `LLMRS_OPENCL_OOO_QUEUE=1`

#### Skip-finish env-gate
- `engine/src/backend/opencl/mod.rs::fill_host_ptr_buffer()` 끝
- env: `LLMRS_HOST_PTR_SKIP_FINISH=1`

#### Probe binaries (paper reproducibility)
- `engine/src/bin/probe_cl_extensions.rs` — Adreno OpenCL extension list dump
- `engine/src/bin/probe_qcom_alloc.rs` — QCOM allocation_type 매트릭스 + DMA-BUF heap test

### 3.3 사용자 결정

- `engine/src/bin/stage0_alignment_check.rs` (prior session 미커밋): 진단 도구. paper future work 시 사용 가능. KEEP 권장.

---

## 4. 미커밋 측정 산출물

### 신규 측정 리포트 (10개, 모두 paper supplementary 가치 있음)

```
papers/eurosys2027/_workspace/experiment/
├── swap_overhead_pre_stage_feasibility.md  (PreStage v1 negative)
├── swap_overhead_pre_stage_v2_bg.md         (PreStage v2 crash)
├── swap_overhead_liswap4_v3.md              (LISWAP-4 plan path 우회, swap이 token 0에 직렬 합산)
├── swap_overhead_liswap1_plus_3.md          (LISWAP-1+3 combo -11%)
├── swap_overhead_liswap1_plus_3_v2.md       (skip-finish 더 나쁨)
├── swap_overhead_op_profiling.md            (op-level root cause: gap)
├── swap_overhead_dmabuf_heap.md             (DMA-BUF coherency 미보장, garbage)
├── swap_overhead_ooo_queue.md               (OoO queue 무시 or 보수 적용)
├── swap_overhead_multictx.md                (multi-context bug v1)
├── swap_overhead_multictx_v2.md             (multi-context bug v2)
└── swap_overhead_multictx_v3.md             (multi-context fixed, nuanced)
```

각각의 raw 로그도 함께 보존:
- `liswap4_v3_raw/`, `liswap1_plus_3_raw/`, `liswap1_plus_3_v2_raw/`, `dmabuf_heap_raw/`, `swap_overhead_ooo_queue_raw/`, `multictx_raw/`, `multictx_v2_raw/`, `multictx_v3_raw/`

### 부분 완료 — n_tokens sweep
`/tmp/swap_decode_sweep/` (60/75 runs 완료, n=2000 진행 중 stop).
`/tmp/swap_decode_sweep/results.csv`에 CSV 형태로 정리됨.

### 보존 위치
이전 세션 부산물 백업: `/tmp/liswap4_artifacts_backup_20260508_151335/` (paper에서 인용 시 필요할 수 있음)

---

## 5. 다음 세션 추천 — 다른 접근 방법

### 5.1 사용자 reference 기반 — heteroinfer / npu.llm

이들은 heterogeneous compute (NPU/DSP/GPU coordination) 활용. Adreno OpenCL 만으로는 한계 도달. 가능한 방향:

#### A. Hexagon DSP offload (Snapdragon 전용)
- weight prep을 Hexagon DSP에서 처리 (별개 hardware)
- GPU와 진짜 병렬 실행 가능
- Qualcomm Hexagon SDK 통합 필요 (multi-day work)
- 본 paper 범위 외, future work

#### B. CPU forward during swap window
- swap 중에는 forward를 CPU 백엔드로 switch
- GPU가 swap만 점유 → 자원 경쟁 없음
- 본 코드베이스에 이미 인프라 있음 (`--switch-threshold`, `--resilience-prealloc-switch`, `model.map_weights_for_cpu`)
- 즉시 측정 가능 (~1d)
- CPU forward는 ~5x 느리지만 stall 회피, TTFT 개선
- 가장 실용적 다음 시도

#### C. Vulkan compute backend
- 다른 driver path. driver-internal command processor 다를 가능성
- 백엔드 신규 구현 필요 (multi-week)
- 본 paper 범위 외

#### D. QNN / SNPE (Qualcomm AI Stack) 통합
- Native Qualcomm runtime. NPU/DSP/GPU 통합 schedule
- 본 paper 범위 변경 (OpenCL 외)

### 5.2 Paper 메시지 update 권장

이전: "All 9 swap paths fail negatively"

수정안 (더 정직):
1. **Per-token TBT는 모든 path에서 비슷한 +5~12 ms 인플레** (driver queue serialization이 본질, op-level profiling 확정)
2. **TTFT는 incremental(LISWAP-1) path가 200-300ms 우세** (sync swap의 stall 회피)
3. **Long decode (n≥1000)에서 모든 path 동일 wall-clock** — production 선택은 TTFT 우선
4. **Multi-context는 per-operation isolation 부분 효과 (-7% swap)지만 forward 인플레 동일** — driver는 cl_mem dependency를 device-wide enforce
5. **벤더 확장(DMA-BUF, KHR external_memory, cl_qcom_*)도 동일한 driver-level 직렬화 받음** — software 우회 구조적 불가능
6. **Future work**: heterogeneous (Hexagon DSP offload, NPU 병렬), CPU forward during swap, Vulkan compute backend

### 5.3 Production 권고 (확정)

```
Sync mode (Phase 6.5):
  - 290 ms single-shot stall
  - TTFT inflated by full swap cost
  - Best total wall-clock for short decodes
  - 기본 권장

LISWAP-1 (per-tick=1):
  - 25 ticks × +22 ms inflation, total wall-clock ~+10%
  - TTFT 200-300 ms 빠름 (decode 시작 즉시)
  - Long decode (1000+ tokens)에서 sync와 wall-clock 동일
  - 응답성 중시 시나리오 권장

기타 (LISWAP-2/3/4, multi-ctx, DMA-BUF, OoO): ablation 재현용 env-gated, default OFF
```

---

## 6. 다음 세션 시작 절차

```bash
cd /home/go/Workspace/llm_rs2

# 1. 현재 상태 파악
git log --oneline -5
# HEAD = 0c8951e (LISWAP-4 v1) — 미커밋 변경 ~10개 트랙

# 2. handoff 읽기
cat .agent/todos/handoff_swap_overhead_negative_2026-05-08.md

# 3. 사용자 결정 분기:
#    (a) 코드 정리 진행 — REMOVE 항목 revert + KEEP 항목 commit
#    (b) 새 접근 시작 — Hexagon DSP / CPU forward during swap / Vulkan / QNN
#    (c) 현재 상태 유지하고 paper drafting

# 4. 측정 결과 사용 — paper supplementary 작성 시
ls papers/eurosys2027/_workspace/experiment/swap_overhead_*.md
```

---

## 7. Critical Files

### 핵심 변경 파일 (KEEP)
- `engine/src/backend/opencl/mod.rs` — DMA-BUF heap, multi-context, OoO env-gate, skip-finish env-gate
- `engine/src/backend/opencl/host_ptr_pool.rs` — DMA-BUF + multi-ctx slot
- `engine/src/models/weights/swap_executor.rs` — 3-way 분기 (multi-ctx / DMA-BUF / ALLOC_HOST_PTR)
- `engine/src/models/weights/intra_forward_swap.rs` — `Arc<GpuEvent>` wait gate fix
- `engine/src/models/weights/async_swap.rs` — `SwapCommitJob.write_event: Arc<GpuEvent>`
- `engine/tests/spec/test_async_swap_executor.rs`, `test_inv_149_wait_gate_ordering.rs` — Arc<GpuEvent> 호환

### 정리 대상 파일 (REMOVE 부분)
- `engine/src/bin/generate.rs` — `--swap-pre-stage` flag, BG thread, DIAG log

### 진단 도구 (KEEP)
- `engine/src/bin/probe_cl_extensions.rs`
- `engine/src/bin/probe_qcom_alloc.rs`

---

## 8. 미해결 / 결정 대기

1. **코드 정리 방침** — 사용자 결정 (이 문서 §3.1 / §3.2 참고)
2. **Paper drafting timing** — 9 트랙 + decode sweep으로 충분히 단단한 negative + nuanced result. 즉시 시작 가능
3. **다음 시도 트랙 선택** — 사용자 의향 (heteroinfer/npu.llm이 가리키는 방향: B (CPU forward) 또는 C (Vulkan) 또는 D (QNN))
4. **Hexagon DSP / Vulkan / QNN 작업 commit** — 새 paper scope 결정

---

## 9. Memory file pointers

새 세션 자동 인지를 위해 auto-memory entry 등록 권장:

```
경로: /home/go/.claude/projects/-home-go-Workspace-llm-rs2/memory/
신규 파일: project_swap_overhead_exhaustive_negative_20260508.md
내용: 9 트랙 negative + multi-context nuanced + Adreno driver-level serialization 본질 + 다음 시도 후보 (Hexagon/CPU forward/Vulkan)
```

본 handoff와 별개로 새 세션 시작 시 인덱싱되도록 작성.
