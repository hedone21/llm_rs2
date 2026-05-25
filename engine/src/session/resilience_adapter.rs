//! [`ResilienceAdapter`] — [`CommandExecutor`]를 3개 session trait에 연결하는 어댑터.
//!
//! `CommandExecutor`는 `poll` / `send_capability` / `on_token_generated` 메서드를
//! 각각 갖지만, session pipeline은 3개의 별도 trait slot
//! ([`CommandSource`], [`EngineReport`], [`TokenTickSink`])을 요구한다.
//!
//! [`ResilienceAdapter`]는 단일 `CommandExecutor` 인스턴스를 소유하고
//! 3개 trait을 모두 구현함으로써 이 두 인터페이스를 연결한다.
//! `DecodeLoopBuilder::with_resilience` 내에서 `Arc<Mutex<Self>>`로 감싸
//! 3개 slot에 각각 newtype wrapper를 통해 주입된다.

use std::sync::{Arc, Mutex};

use anyhow::Result;
use llm_shared::{EngineCapability, QcfEstimate, WeightSwapReport};

use crate::resilience::{CommandExecutor, ExecutionPlan, KVSnapshot};
use crate::session::traits::{CommandSource, EngineReport, StepCtx, TokenTickSink};

/// [`CommandExecutor`]를 session 3-trait으로 연결하는 어댑터.
pub struct ResilienceAdapter {
    executor: CommandExecutor,
}

impl ResilienceAdapter {
    pub fn new(executor: CommandExecutor) -> Self {
        Self { executor }
    }

    /// 직접 접근이 필요한 caller (capability send 전 `set_has_secondary` 등)를 위해
    /// mutable ref를 노출한다.
    pub fn executor_mut(&mut self) -> &mut CommandExecutor {
        &mut self.executor
    }
}

impl CommandSource for ResilienceAdapter {
    fn poll(&mut self, _ctx: &StepCtx, kv_snap: &KVSnapshot) -> Result<ExecutionPlan> {
        Ok(self.executor.poll(kv_snap))
    }
}

impl EngineReport for ResilienceAdapter {
    fn send_capability(&mut self, cap: EngineCapability) {
        self.executor.send_capability(cap);
    }
    fn send_qcf_estimate(&mut self, qcf: QcfEstimate) {
        self.executor.send_qcf_estimate(qcf);
    }
    fn send_swap_report(&mut self, report: WeightSwapReport) {
        self.executor.send_weight_swap_report(report);
    }
}

impl TokenTickSink for ResilienceAdapter {
    fn on_token_generated(&mut self, _ctx: &StepCtx) {
        self.executor.on_token_generated();
    }
}

// ── Arc<Mutex<ResilienceAdapter>> 기반 3개 newtype wrapper ──
//
// `DecodeLoopBuilder::with_resilience`가 단일 ResilienceAdapter 인스턴스를
// Arc<Mutex<…>>로 감싸고, 3개 슬롯에 각각 newtype wrapper를 주입하는 방식이다.
// 이를 통해 한 인스턴스로 3개 trait slot 동시 충족.
//
// per-token 호출 빈도: poll 1회 + on_token_generated 1회 → Mutex contention 무시 가능.

/// `Arc<Mutex<ResilienceAdapter>>` 를 `CommandSource` 로 노출하는 newtype.
pub(crate) struct CmdSrcWrapper(pub Arc<Mutex<ResilienceAdapter>>);

impl CommandSource for CmdSrcWrapper {
    fn poll(&mut self, ctx: &StepCtx, kv_snap: &KVSnapshot) -> Result<ExecutionPlan> {
        self.0
            .lock()
            .expect("resilience mutex poisoned")
            .poll(ctx, kv_snap)
    }
}

/// `Arc<Mutex<ResilienceAdapter>>` 를 `EngineReport` 로 노출하는 newtype.
pub(crate) struct ReportWrapper(pub Arc<Mutex<ResilienceAdapter>>);

impl EngineReport for ReportWrapper {
    fn send_capability(&mut self, cap: EngineCapability) {
        self.0
            .lock()
            .expect("resilience mutex poisoned")
            .send_capability(cap);
    }
    fn send_qcf_estimate(&mut self, qcf: QcfEstimate) {
        self.0
            .lock()
            .expect("resilience mutex poisoned")
            .send_qcf_estimate(qcf);
    }
    fn send_swap_report(&mut self, report: WeightSwapReport) {
        self.0
            .lock()
            .expect("resilience mutex poisoned")
            .send_swap_report(report);
    }
}

/// `Arc<Mutex<ResilienceAdapter>>` 를 `TokenTickSink` 로 노출하는 newtype.
pub(crate) struct TickWrapper(pub Arc<Mutex<ResilienceAdapter>>);

impl TokenTickSink for TickWrapper {
    fn on_token_generated(&mut self, ctx: &StepCtx) {
        self.0
            .lock()
            .expect("resilience mutex poisoned")
            .on_token_generated(ctx);
    }
}
