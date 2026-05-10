# Phase-aware Async Swap (LISWAP-5) — Implementation Plan

**Date**: 2026-05-10
**선행**:
- Phase R `qnn_phase_r_summary.md` §5 (1.04× of max GREEN microbench)
- 9-track negative `swap_overhead_negative_2026-05-08.md` (driver FIFO 미회피)
- 본 세션 measurement `swap_overhead_phase_predictability_2026_05_10.md` (CV < 2%)

---

## 0. 전략

production decode forward의 op-level wall-clock이 deterministic (CV < 2%)임을 활용하여, **op_trace 경계에서 phase를 식별하고 cache-fit phase에만 swap chunk을 enqueue**하는 dispatcher 구현. DDR contention이 본질적 문제이므로 driver-level 우회가 아닌 **trigger 시점 통제**로 driver FIFO와 공존.

핵심 가설: **chunk이 cache-fit window(~980 us per layer) 안에서 시작하면 다음 matmul 시작 전까지 H2D가 완료된다** (Phase R Scenario B 1.04×).

---

## 1. 작업 분해

### B-2.1 — `OpKind::ddr_phase()` classifier (✅ 본 세션 commit 예정)
- `engine/src/profile/op_trace.rs`에 `DdrPhase { Heavy, CacheFit, Medium }` enum + `OpKind::ddr_phase() const fn`
- 회귀 위험: LOW (compile-time const)

### B-2.2 — `phase_aware_swap.rs` skeleton + dispatcher trait
- `engine/src/models/weights/phase_aware_swap.rs` 신규
- 구성:
  ```rust
  pub trait PhaseHook: Send + Sync {
      fn on_op_complete(&self, kind: OpKind);
  }
  pub struct PhaseAwareSwapDispatcher {
      chunk_queue: Mutex<VecDeque<SwapChunk>>,
      in_flight: Mutex<Option<Arc<GpuEvent>>>,
      chunk_size_bytes: usize,
      executor: Arc<SwapExecutor>,
      secondary: Arc<SecondaryMmap>,
  }
  ```
- 동작:
  - `on_op_complete(CacheFit)`: in_flight 없으면 next chunk pop → `enqueue_write_async` → in_flight = event
  - `on_op_complete(Heavy)`: 진입 직전 wait — 별도 trait `on_op_start(Heavy)` 또는 `tr_start!(MatmulQkv)`로 분리 필요
- 추정: 1.5d (Implementer)

### B-2.3 — `tr_start!`/`tr_record!` 매크로 확장
- 기존 `tr_start!()` → `tr_start!(OpKind)` 시그니처 변경 (11 callsite)
- `op_trace::start(kind)` + `op_trace::record(t, kind)` 둘 다 phase hook 콜백 호출
- zero-overhead (PHASE_HOOK이 None이면 inline check 후 return)
- 회귀 위험: LOW (변경된 callsite는 기존과 동일 동작 유지)
- 추정: 0.5d (Implementer)

### B-2.4 — Layer weight chunk 분할 + dispatch loop
- `SwapExecutor::split_layer_weights_into_chunks(layer_idx, chunk_size_bytes) -> Vec<SwapChunk>`
- 각 chunk = (cl_mem dst, host src ptr, byte range)
- 25 layer × ceil(36 MB / 4 MB) = 25 × 9 = 225 chunk per swap plan
- chunk_queue에 push 순서: layer 1 chunk 1..9, layer 2 chunk 1..9, ...
- 추정: 1d (Senior Implementer — async H2D + cl_event 관리)

### B-2.5 — CLI + dispatcher wire-up
- `--swap-phase-aware` flag (bool)
- `--swap-phase-aware-chunk-mb=4` (default)
- generate.rs decode loop: pressure signal → PhaseAwareSwapDispatcher 생성 + PHASE_HOOK 등록
- mutually exclusive with `--swap-incremental-per-tick > 0` and `--swap-intra-forward`
- 추정: 0.5d (Implementer)

### B-3 — 검증 + 측정 (이미 등록된 task)
- correctness: top-5 overlap > 99%
- hide ratio measurement: in_flight cl_event timestamps vs forward op timestamps
- TBT regression: ≤ baseline + 5%
- 25-layer swap user-perceived stall: ≤ 50 ms
- 추정: 1d (Tester)

---

## 2. Total estimate
**~5 dev-day** (병렬 가정 시 wall ~3d)

---

## 3. 기존 시스템과의 관계

| Component | 영향 |
|---|---|
| `IntraForwardSwapHook` (LISWAP-4 v3) | 기능 중복 — `--swap-phase-aware`와 mutually exclusive. LISWAP-4는 v3에서 -65% 회귀, deprecated 후보 |
| `IncrementalSwapPlan` (LISWAP-1) | 호환 — phase-aware는 swap commit lifecycle 동일 (per-layer 단위 ArcSwap) |
| `AsyncSwapDispatcher` | 재사용 — chunk-level enqueue + cl_event 관리는 이미 인프라 있음 |
| `SwapExecutor::execute_on_slots` | 분할 시점 변경 — 전체 layer batch가 아닌 chunk batch 단위 호출 |
| Spec ID | 신규: ENG-ALG-239~ (phase classifier), ENG-ALG-240~ (chunk dispatcher), INV-151~ (driver FIFO 가정 + chunk timing safety) |

---

## 4. Risk

| Risk | 평가 | 완화 |
|---|---|---|
| chunk_size 4 MB가 production에서 cache-fit window 초과 | LOW | 측정으로 σ < 4% 확인됨, 30% margin |
| Adreno driver가 host-pinned write를 GPU compute와 정말 overlap하는지 | MEDIUM | Phase R에서 1.04× 확인, 단 Adreno UMA + DMA-BUF interop 환경에서 재확인 필요 |
| op_trace boundary가 GPU 실제 실행과 lag | MEDIUM | sync mode는 lag 없음, async mode는 enqueue overhead만 — chunk은 wall-clock에 align되므로 OK |
| LISWAP-4 v3 dead code 정리 부담 | LOW | env-gated로 보존 가능, default off |

---

## 5. Pass-gate (production-ready)

```
✓ correctness: top-5 overlap > 99% vs --swap-incremental-per-tick=0 (single-shot)
✓ TBT regression: ≤ baseline + 5% (n=5)
✓ hide ratio: ≥ 80% (Phase R는 96%, production은 보수적으로 80% 목표)
✓ 25-layer swap user-perceived stall: ≤ 50 ms
✓ no cargo test regression
✓ no clippy regression (warnings as errors)
```

---

## 6. 다음 세션 entry point

```bash
# 본 세션 산출물 확인
git log --oneline -5
cat papers/eurosys2027/_workspace/experiment/swap_overhead_phase_predictability_2026_05_10.md

# B-2.2 시작
# 1. phase_aware_swap.rs skeleton
# 2. tr_start! kind 확장 (11 callsite)
# 3. Implementer 위임 가능 (sonnet) — 단 chunk dispatcher cl_event는 Senior Implementer
```

---

**End of plan**
