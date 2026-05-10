# Handoff: LISWAP-5 Phase-aware Async Swap — B-2.2 ~ B-3 진행

**Date**: 2026-05-10
**HEAD**: `2015597` (B-1 + B-2.1 commit)
**선행 plan**: `.agent/todos/feat_phase_aware_swap_2026_05_10.md`
**선행 measurement**: `papers/eurosys2027/_workspace/experiment/swap_overhead_phase_predictability_2026_05_10.md`
**Device**: Galaxy S25 (R3CY408S5SB, Adreno 830 + Hexagon V79)

---

## 0. 한 줄 요약

production decode op timing이 1.2% CV로 결정론적임을 확인. 4 MB chunk이 cache-fit window 980 us에 30% margin으로 들어감. 9-track 음성 결과와 직교한 trigger-시점-통제 전략으로 LISWAP-5 구현 진행. B-2.2 (dispatcher skeleton)부터 B-3 (production 검증)까지 작업.

---

## 1. 완료된 것 (이전 세션)

### 1.1 B-1 — Production op-level breakdown 측정 ✅
- 5-run × (28 layer × 31 token) = 4,340 op-sample
- cache-fit per layer: **984 us, CV 1.2%**
- ddr-heavy per layer: 1706 us, CV 1.1%
- 모든 op CV < 4%, matmul_ffn_gate_up은 σ=0
- 결과: `papers/eurosys2027/_workspace/experiment/swap_overhead_phase_predictability_2026_05_10.md`

### 1.2 B-2.1 — `OpKind::ddr_phase()` classifier ✅
- `engine/src/profile/op_trace.rs`에 추가 (commit `2015597`)
- `DdrPhase { Heavy, CacheFit, Medium }` enum + const fn 매핑
- 회귀 위험 LOW (compile-time, no runtime cost)

### 1.3 산출 commit
```
2015597 feat(profile): OpKind::ddr_phase() classifier + phase-aware swap plan
```

---

## 2. 다음 작업 (B-2.2 ~ B-3)

### B-2.2 — `phase_aware_swap.rs` skeleton + `PhaseHook` trait (1.5d)

**신규 파일**: `engine/src/models/weights/phase_aware_swap.rs`

```rust
//! LISWAP-5 — Phase-aware Async Weight Swap
//!
//! 전략: op_trace boundary에서 OpKind::ddr_phase()를 검사하여 cache-fit
//! phase 진입 시 chunk H2D를 enqueue, ddr-heavy phase 진입 직전 wait.
//! Phase R Scenario B (1.04× of max) precondition 충족 (CV 1.2%).

use crate::profile::op_trace::{DdrPhase, OpKind};
use crate::core::backend::{Backend, GpuEvent};
use crate::models::weights::async_swap::{AsyncSwapDispatcher, SwapCommitJob};
use crate::models::weights::secondary_mmap::SecondaryMmap;
use crate::models::weights::slot::LayerSlot;
use crate::models::weights::swap_executor::SwapExecutor;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// op_trace에서 호출되는 phase boundary hook.
/// `tr_start!`/`tr_record!` 매크로 확장으로 자동 호출됨.
pub trait PhaseHook: Send + Sync {
    /// op 시작 직전 호출. ddr-heavy 진입이면 in-flight chunk 완료 대기.
    fn on_op_start(&self, kind: OpKind);
    /// op 끝난 직후 호출. cache-fit 끝났으면 다음 chunk dispatch.
    fn on_op_end(&self, kind: OpKind);
}

/// 4 MB chunk 단위 layer weight 분할.
pub struct WeightChunk {
    pub layer_idx: usize,
    pub byte_offset: usize,
    pub byte_len: usize,  // typically 4 MB except last chunk
}

pub struct PhaseAwareSwapDispatcher {
    chunk_queue: Mutex<VecDeque<WeightChunk>>,
    in_flight: Mutex<Option<Arc<GpuEvent>>>,
    chunk_size_bytes: usize,
    layer_slots: Vec<Arc<LayerSlot>>,
    secondary: Arc<SecondaryMmap>,
    backend: Arc<dyn Backend>,
    dispatcher: Arc<AsyncSwapDispatcher>,
    // ... see intra_forward_swap.rs:60-120 for full pattern
}

impl PhaseAwareSwapDispatcher {
    pub fn new(...) -> Arc<Self>;
    pub fn commit_plan(&self, target_layers: &[usize]);  // 전체를 chunk으로 분할
    fn wait_pending(&self);                              // in_flight 대기
    fn try_dispatch_chunk(&self) -> Result<()>;          // 다음 chunk enqueue
    pub fn finalize(&self);                              // drain + ratio bump
}

impl PhaseHook for PhaseAwareSwapDispatcher {
    fn on_op_start(&self, kind: OpKind) {
        if matches!(kind.ddr_phase(), DdrPhase::Heavy) {
            self.wait_pending();
        }
    }
    fn on_op_end(&self, kind: OpKind) {
        if matches!(kind.ddr_phase(), DdrPhase::CacheFit) {
            let _ = self.try_dispatch_chunk();
        }
    }
}
```

**참고 patterns** (재사용/모방 대상):
- `engine/src/models/weights/intra_forward_swap.rs:415-510` — `dispatch_layer` 메서드: secondary mmap에서 weight 읽고 `enqueue_write_async` 호출 후 `SwapCommitJob` submit. **chunk 버전은 layer 전체 대신 chunk 1개만 호출**.
- `engine/src/models/weights/swap_executor.rs:388-518` — `execute_on_slots` 본체: prefault → for-loop layer → write → SwapCommitJob.
- `engine/src/models/weights/async_swap.rs:141` — `submit_commit` API.

**Chunk 분할 전략 (보수적 v1)**:
- chunk_size = 4 MB
- layer 1개 = ~36 MB (Q4_0) → 9 chunk
- 25 layer × 9 chunk = 225 chunk in queue
- **단순화**: 한 chunk 완료 후 다음 chunk → 동일 layer를 모두 swap한 뒤에야 ratio_generation bump (intra_forward_swap.rs와 동일 ArcSwap 패턴 유지).
- 실제 layer slot replacement는 **layer 전체 chunk이 끝났을 때만**. partial layer는 dual buffer로 staging.

**Open question**: chunk-level cl_mem 어떻게 alloc? 두 가지 옵션:
- **Option A**: layer 전체 cl_mem alloc 후 chunk 단위 partial write_buffer offset. ArcSwap 교체는 마지막 chunk 후.
- **Option B**: chunk 단위 별도 cl_mem (sub-buffer). complex, 안 권장.

→ **Option A 권장**. `enqueue_write_buffer`의 `offset` 인자 사용. 이미 OpenCL spec 보장.

### B-2.3 — `tr_start!`/`tr_record!` 매크로 kind 확장 (0.5d)

**현재**:
```rust
let tr = tr_start!();
... matmul ...
tr_record!(tr, MatmulQkv);
```

**변경 후**:
```rust
let tr = tr_start!(MatmulQkv);   // ← kind 인자 추가, on_op_start 호출
... matmul ...
tr_record!(tr, MatmulQkv);       // ← 기존과 동일, on_op_end 호출
```

**구현**:
- `engine/src/profile/op_trace.rs`:
  ```rust
  static PHASE_HOOK: OnceLock<Arc<dyn PhaseHook>> = OnceLock::new();

  pub fn set_phase_hook(hook: Arc<dyn PhaseHook>) {
      let _ = PHASE_HOOK.set(hook);
  }

  #[inline]
  pub fn start_op(kind: OpKind) -> Option<Instant> {
      if let Some(hook) = PHASE_HOOK.get() {
          hook.on_op_start(kind);
      }
      start()
  }

  // record() 끝부분에 추가:
  if let Some(hook) = PHASE_HOOK.get() {
      hook.on_op_end(op);
  }
  ```
- `engine/src/layers/transformer_layer/forward_gen.rs`:
  - `tr_start!()` → `tr_start!($kind:ident)` 변경 (11 callsite at lines 202/305/355/419/1114/1124/1161/1519/1533/1543/1568에서 호출되는 자리들)
  - `tr_start!()` 호출자 위치는 `let tr = tr_start!();` 직전. tr_record!의 kind를 그대로 사용.

**주의**: `tr_start!()`는 forward_gen.rs에서 11번 호출됨. 각 호출자마다 그 다음 줄 또는 가까이에 `tr_record!(tr, KIND)`가 옴. 매크로 변경 시 모든 callsite를 함께 갱신.

**Zero-overhead 보장**:
- `PHASE_HOOK.get()` = atomic load (~2 ns)
- hook 미설정 시 즉시 return
- 11 callsite × 2 op ends/op = 22 calls per layer × 28 layer × 30 token = ~18,000 atomic loads/run = 36 us total → noise floor 이하

### B-2.4 — Layer weight chunk 분할 + async H2D dispatcher (1d, **Senior Implementer 권장**)

**핵심 작업**:
1. `SwapExecutor`에 chunk 단위 build 메서드 추가:
   ```rust
   pub fn build_chunk_async(
       &self,
       secondary: &SecondaryMmap,
       slot: &LayerSlot,
       layer_idx: usize,
       byte_offset: usize,
       byte_len: usize,
   ) -> Result<(ChunkBuildResult, GpuEvent)>;
   ```
2. `PhaseAwareSwapDispatcher::try_dispatch_chunk` 안에서 호출.
3. **누적 staging**: 같은 layer의 chunk 결과를 `LayerSlot`의 staging cl_mem에 누적. 마지막 chunk 완료 시 `slot.swap_weights` (ArcSwap atomic).
4. cl_event 관리: 각 chunk마다 `Arc<GpuEvent>` 보관, `wait_pending`은 가장 최근 event 대기.

**관련 file**:
- `engine/src/models/weights/swap_executor.rs:388-518` — `execute_on_slots` 패턴 참고
- `engine/src/models/weights/slot.rs` — `LayerSlot::swap_weights` ArcSwap commit
- `engine/src/core/backend.rs::Backend::enqueue_write_async` — async H2D entry

**왜 Senior Implementer**:
- cl_event lifetime + thread-safety (worker thread + forward thread)
- offset 기반 partial write_buffer (Adreno UMA semantic 주의)
- ArcSwap commit ordering (INV-150 보존)

### B-2.5 — CLI flag + decode loop wire-up (0.5d)

**generate.rs 변경**:
```rust
// Args 구조체에 추가:
#[arg(long, default_value_t = false)]
swap_phase_aware: bool,

#[arg(long, default_value_t = 4)]
swap_phase_aware_chunk_mb: usize,
```

**Mutually exclusive validation** (generate.rs:1011 근처에 추가):
```rust
let swap_modes = [
    args.swap_incremental_per_tick > 0,
    args.swap_intra_forward,
    args.swap_phase_aware,
];
if swap_modes.iter().filter(|x| **x).count() > 1 {
    anyhow::bail!("--swap-incremental-per-tick / --swap-intra-forward / --swap-phase-aware are mutually exclusive");
}
```

**Decode loop hook 등록** (generate.rs:5675 근처, intra_forward 옆):
```rust
if args.swap_phase_aware && pressure_signal_received {
    let dispatcher = PhaseAwareSwapDispatcher::new(...);
    dispatcher.commit_plan(&target_layers);
    crate::profile::op_trace::set_phase_hook(dispatcher.clone());
    phase_aware_swap_dispatcher = Some(dispatcher);
}
// finalize 시점:
if let Some(disp) = phase_aware_swap_dispatcher.take() {
    disp.finalize();
}
```

### B-3 — Production 검증 + 측정 (1d, Tester)

**측정 시나리오** (Galaxy S25, Qwen2.5-1.5B Q4_0):
1. **Correctness gate**:
   ```bash
   ./generate ... --backend opencl --force-swap-ratio 0.9 --swap-phase-aware
   ```
   vs single-shot baseline. top-5 overlap > 99%, token sequence 비교.

2. **TBT regression**:
   ```bash
   # n=5 run, decode 32 token
   ./generate ... --swap-phase-aware --temperature 0
   ```
   기준: baseline (--swap-incremental-per-tick=0) Decode TBT ± 5%.

3. **Hide ratio 측정**:
   - dispatcher 안에 `total_h2d_us` (chunk write_buffer wall-clock 합) 누적
   - `total_forward_us` (forward 시작~끝 wall-clock 합) 누적
   - **hide ratio = 1 - (TBT_with_swap - TBT_baseline) / total_h2d_us**
   - 목표: ≥ 80%

4. **User-perceived stall**:
   - 25-layer swap 시 frame budget(33 ms) 초과 token 수
   - 목표: 0 (모든 token < 33 ms)

5. **Long decode wall-clock** (n=1000):
   - sync vs LISWAP-1 vs LISWAP-5 비교
   - 9-track 결과대로면 모든 path 수렴 — LISWAP-5가 최소한 타이는 쳐야

**측정 보고**: `papers/eurosys2027/_workspace/experiment/swap_overhead_liswap5_v1.md`

---

## 3. Risk + 완화

| Risk | 평가 | 완화 |
|---|---|---|
| Adreno driver가 host-pinned partial write를 GPU compute와 정말 overlap하는지 | MEDIUM | Phase R 1.04× 확인됨. **B-2.4 Senior Impl이 먼저 microbench로 chunk write+matmul concurrent 측정** 권장 (1d 내). 음성이면 plan re-scope. |
| chunk_size 4 MB가 production에서 cache-fit window 초과 | LOW | 측정 σ=12 us, 30% margin. 처음엔 2 MB로 보수 시작 가능 (`--swap-phase-aware-chunk-mb=2`). |
| 11 callsite 매크로 변경 시 컴파일 회귀 | LOW | 모든 호출이 같은 파일 — sed로 일괄 변환 가능. |
| LISWAP-4 v3와 코드 중복 / 충돌 | LOW | mutually exclusive flag로 차단. LISWAP-4는 deprecated 후보지만 본 작업에서 제거하지 않음. |
| ArcSwap commit ordering — partial chunk 중 forward가 layer 진입하면? | MEDIUM | **layer당 모든 chunk 완료 전엔 ArcSwap swap 금지**. dispatcher는 staging cl_mem에 누적, 마지막 chunk 후 atomic commit. (intra_forward_swap의 INV-149 패턴 모방) |

---

## 4. Verification gates

```
B-2.2 PASS: cargo build OK + skeleton compile (no runtime test)
B-2.3 PASS: tr_start!/tr_record! 변경 후 op_trace=sync 로 1 run, 결과 동일
B-2.4 PASS: microbench (chunk 4 MB write + matmul concurrent) wall-clock 측정 → C3/max ≤ 1.10×
B-2.5 PASS: --swap-phase-aware flag 인식, mutually exclusive validation 동작
B-3 PASS:
  - top-5 overlap > 99%
  - TBT regression ≤ baseline + 5%
  - hide ratio ≥ 80%
  - user-perceived stall ≤ 33 ms (frame budget)
  - cargo test --workspace 회귀 0
  - cargo clippy -D warnings 회귀 0
```

---

## 5. 다음 세션 시작 절차

```bash
cd /Users/li/Workspace/llm_rs2

# 1. 상태 확인
git log --oneline -5    # 2015597이 HEAD
cat .agent/todos/handoff_phase_aware_swap_b22_2026_05_10.md  # 본 문서
cat .agent/todos/feat_phase_aware_swap_2026_05_10.md         # 5-step plan
cat papers/eurosys2027/_workspace/experiment/swap_overhead_phase_predictability_2026_05_10.md

# 2. 디바이스 확인
adb devices    # R3CY408S5SB

# 3. B-2.2 시작
# 신규 파일: engine/src/models/weights/phase_aware_swap.rs
# 참고 파일:
#   - engine/src/profile/op_trace.rs (DdrPhase + OpKind::ddr_phase())
#   - engine/src/models/weights/intra_forward_swap.rs (LISWAP-4 v3 패턴)
#   - engine/src/models/weights/swap_executor.rs::execute_on_slots
#   - engine/src/models/weights/async_swap.rs::submit_commit

# 4. 위임 권장:
# B-2.2, B-2.3, B-2.5: Implementer (sonnet)
# B-2.4: Senior Implementer (opus, cl_event + Adreno UMA)
# B-3:   Tester (opus, deploy-test skill)
```

---

## 6. 메모리 (다음 세션이 알아야 할 것)

본 작업과 직접 관련:
- `project_phase_aware_swap_viable.md` — 본 세션에서 추가. CV 1.2% precondition 측정 결과.
- `project_qnn_oppkg_dual_buffer_kv.md` — Adreno DMA-BUF interop 환경. chunk H2D efficiency에 영향.

직접 관련 X (참조용):
- `project_partition_a1_async_read_failed.md` — Adreno async overlap에 대한 부정 사례 (KV partition). LISWAP-5와는 직교 (KV vs weight).
- `feedback_byte_equal_alone_is_not_correctness.md` — top-5 overlap 검증 시 항상 semantic ground truth.

---

## 7. 사용자 결정 사항 (필요 시)

| 질문 | 옵션 |
|---|---|
| LISWAP-4 v3 코드 처분 | A. 보존 (env-gated, default off) / B. 본 작업 후 제거 |
| Spec ID 발급 | Architect 위임해서 ENG-ALG-239~ / INV-151~ 할당 |
| Backend 범위 | A. opencl만 / B. qnn_oppkg도 (DMA-BUF interop으로 더 빠를 가능성) |

기본값: **A 보존 / 본 세션 plan 그대로 / opencl만 (qnn_oppkg는 B-3 통과 후 별도 검토)**.

---

**End of Handoff**

self-contained: 다음 세션은 본 문서 + plan doc + measurement 보고만으로 시작 가능.
