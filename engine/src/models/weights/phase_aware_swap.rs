//! LISWAP-5 — Phase-aware Async Weight Swap (skeleton, B-2.2).
//!
//! 전략: production decode forward의 op-level wall-clock이 deterministic
//! (CV 1.2%)이라는 측정 결과 (`papers/.../swap_overhead_phase_predictability_2026_05_10.md`)를
//! 활용하여 `op_trace::PhaseHook`를 통해 op boundary에서 phase를 검사하고:
//!
//! - `DdrPhase::CacheFit` 끝 → 다음 chunk H2D enqueue (Phase R Scenario B 1.04× of max)
//! - `DdrPhase::Heavy` 시작 직전 → in-flight chunk 완료 대기 (driver FIFO 공존)
//!
//! 9-track 음성 결과와 직교: trigger 시점 통제로 driver-internal command
//! processor FIFO를 우회하지 않고 공존.
//!
//! Spec 참고: ENG-ALG-239~ (B-3 검증 후 Architect 발급 예정).
//! 본 파일은 **skeleton (B-2.2)** — 실제 chunk dispatch 로직은 B-2.4에서 구현.

use crate::core::backend::{Backend, GpuEvent};
use crate::core::buffer::DType;
use crate::models::weights::async_swap::AsyncSwapDispatcher;
use crate::models::weights::secondary_mmap::SecondaryMmap;
use crate::models::weights::slot::LayerSlot;
use crate::profile::op_trace::{DdrPhase, OpKind, PhaseHook};
use anyhow::Result;
use std::collections::VecDeque;
use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

/// Layer weight를 chunk 단위로 분할한 단위. B-2.4에서 SwapExecutor가
/// `byte_offset..byte_offset+byte_len` 영역만 staging cl_mem으로 H2D.
#[derive(Clone, Debug)]
pub struct WeightChunk {
    pub layer_idx: usize,
    pub byte_offset: usize,
    pub byte_len: usize,
    /// Layer 내 chunk seq (0-based). 마지막 chunk(`is_last_in_layer=true`)
    /// 완료 시점에 LayerSlot::swap_weights ArcSwap commit.
    pub chunk_seq: usize,
    pub is_last_in_layer: bool,
}

/// Phase-aware async swap dispatcher (skeleton). PhaseHook impl로 op boundary
/// 호출을 받아 chunk을 cache-fit window에만 dispatch.
///
/// **B-2.2 skeleton 상태**: 필드 + 시그니처만. 실제 chunk dispatch (cl_event,
/// async H2D, ArcSwap commit ordering) 은 B-2.4에서 senior-implementer가 구현.
pub struct PhaseAwareSwapDispatcher {
    /// 분할된 chunk 큐. layer 1 chunk 0..N → layer 2 chunk 0..N → ... 순서.
    chunk_queue: Mutex<VecDeque<WeightChunk>>,
    /// 가장 최근에 enqueue한 chunk의 cl_event. ddr-heavy 진입 시 wait.
    in_flight: Mutex<Option<Arc<GpuEvent>>>,
    /// chunk 1개 크기 (default 4 MB, CV 1.2% 측정에서 980us cache-fit window의 30% margin).
    chunk_size_bytes: usize,
    /// Layer slot 참조 (chunk staging cl_mem alloc + ArcSwap commit 대상).
    layer_slots: Vec<Arc<LayerSlot>>,
    /// Secondary weight source (mmap-backed).
    secondary: Arc<SecondaryMmap>,
    /// GPU backend (host-pinned write_buffer + cl_event).
    backend: Arc<dyn Backend>,
    /// AsyncSwapDispatcher worker — 실제 ArcSwap commit은 worker thread.
    dispatcher: Arc<AsyncSwapDispatcher>,
    /// Target dtype (Q4_0 등).
    #[allow(dead_code)]
    target_dtype: DType,
    /// finalize() 호출 후 true. 이후 dispatch 시도는 noop.
    finalized: AtomicBool,
}

impl PhaseAwareSwapDispatcher {
    /// 생성자 — chunk_size_bytes는 보통 4 MB (`chunk_size_mb * 1_048_576`).
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        chunk_size_bytes: usize,
        layer_slots: Vec<Arc<LayerSlot>>,
        secondary: Arc<SecondaryMmap>,
        backend: Arc<dyn Backend>,
        dispatcher: Arc<AsyncSwapDispatcher>,
        target_dtype: DType,
    ) -> Arc<Self> {
        Arc::new(Self {
            chunk_queue: Mutex::new(VecDeque::new()),
            in_flight: Mutex::new(None),
            chunk_size_bytes,
            layer_slots,
            secondary,
            backend,
            dispatcher,
            target_dtype,
            finalized: AtomicBool::new(false),
        })
    }

    /// Plan commit — `target_layers` 각각을 `chunk_size_bytes` 단위로 분할하여
    /// chunk_queue에 push. **B-2.4 todo**: 실제 layer byte-size 계산 + chunk 생성.
    pub fn commit_plan(&self, target_layers: &[usize]) {
        // TODO B-2.4:
        // for &layer_idx in target_layers {
        //     let total_bytes = self.layer_byte_size(layer_idx);
        //     let n_chunks = (total_bytes + self.chunk_size_bytes - 1) / self.chunk_size_bytes;
        //     for seq in 0..n_chunks {
        //         let offset = seq * self.chunk_size_bytes;
        //         let len = (total_bytes - offset).min(self.chunk_size_bytes);
        //         queue.push_back(WeightChunk {
        //             layer_idx, byte_offset: offset, byte_len: len,
        //             chunk_seq: seq, is_last_in_layer: seq + 1 == n_chunks,
        //         });
        //     }
        // }
        let _ = target_layers;
        let _ = &self.layer_slots;
        let _ = &self.secondary;
        let _ = &self.dispatcher;
        let _ = &self.backend;
    }

    /// in_flight chunk H2D 완료 대기. ddr-heavy phase 진입 직전 호출.
    /// **B-2.4 todo**: `wait_event_blocking` 호출 후 in_flight clear.
    fn wait_pending(&self) {
        // TODO B-2.4:
        // if let Ok(mut guard) = self.in_flight.lock() {
        //     if let Some(ev) = guard.take() {
        //         let _ = self.backend.wait_event_blocking(&ev);
        //     }
        // }
    }

    /// 다음 chunk pop → secondary mmap에서 staging cl_mem으로 enqueue_write_async →
    /// in_flight = event. **B-2.4 todo**.
    fn try_dispatch_chunk(&self) -> Result<()> {
        // TODO B-2.4:
        // 1. in_flight.is_some() 이면 skip (한 chunk in-flight 정책)
        // 2. chunk_queue.pop_front() → None이면 done
        // 3. SwapExecutor::build_chunk_async(secondary, slot, layer_idx, offset, len)
        // 4. enqueue_write_async → cl_event 받기 → in_flight = Arc<GpuEvent>
        // 5. is_last_in_layer 면 SwapCommitJob submit (ArcSwap commit)
        Ok(())
    }

    /// Plan 종료 — 남은 chunk drain + synchronize + ratio_generation +1.
    /// `IntraForwardSwapHook::finalize` (intra_forward_swap.rs:550 부근) 패턴 모방.
    pub fn finalize(&self) -> Result<()> {
        self.finalized
            .store(true, std::sync::atomic::Ordering::Release);
        // TODO B-2.4:
        // 1. chunk_queue 남은 chunk drain (최대 노력)
        // 2. wait_pending()
        // 3. backend.synchronize()
        // 4. release_worker drain
        // 5. ArcSwap ratio_generation bump (현재 활성 layer 무효화)
        Ok(())
    }

    /// 진단 — 남은 chunk 수.
    pub fn remaining_chunks(&self) -> usize {
        self.chunk_queue.lock().map(|q| q.len()).unwrap_or(0)
    }
}

impl PhaseHook for PhaseAwareSwapDispatcher {
    #[inline]
    fn on_op_start(&self, kind: OpKind) {
        if self.finalized.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        if matches!(kind.ddr_phase(), DdrPhase::Heavy) {
            self.wait_pending();
        }
    }

    #[inline]
    fn on_op_end(&self, kind: OpKind) {
        if self.finalized.load(std::sync::atomic::Ordering::Acquire) {
            return;
        }
        if matches!(kind.ddr_phase(), DdrPhase::CacheFit) {
            let _ = self.try_dispatch_chunk();
        }
    }
}
