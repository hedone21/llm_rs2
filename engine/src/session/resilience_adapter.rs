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
use llm_shared::{EngineCapability, EngineCommand, QcfEstimate, WeightSwapReport};

use crate::format::KVCacheFormat;
use crate::resilience::{CommandExecutor, KVSnapshot};
use crate::session::traits::{CommandSource, EngineReport, StepCtx, TokenTickSink};

/// [`CommandExecutor`]를 session 3-trait으로 연결하는 어댑터 (β-4: ManagerCommandSource 역할).
///
/// **β-4 (매핑 문서 4부 채택안 (가))**: `CommandSource::poll` 이 pure(`Vec<EngineCommand>` 반환)로
/// retarget 되면서, heartbeat 송출은 이 source 내부에 잔존한다 — poll 직전에
/// `send_heartbeat_if_due` 를 호출한다. heartbeat payload(`KVSnapshot`)의 `kv_snap` 운반은 poll
/// 인자 대신 **held-handle**(register 시점 주입받은 layer-0 `Arc<dyn KVCacheFormat>`)에서
/// `current_pos`/`capacity` 를 query 해 자체 구성한다(§5.2.1 (가) god-ctx 회피 정신).
pub struct ResilienceAdapter {
    executor: CommandExecutor,
    /// β-4: heartbeat payload 의 kv_cache_tokens/capacity 를 query 할 held-handle.
    /// `None` 이면 partial snapshot(pos=0) — set_kv_handle 미주입 경로.
    kv_handle: Option<Arc<dyn KVCacheFormat>>,
}

impl ResilienceAdapter {
    pub fn new(executor: CommandExecutor) -> Self {
        Self {
            executor,
            kv_handle: None,
        }
    }

    /// β-4: heartbeat snapshot query 용 held-handle 주입(§5.2.1 (가) — kv_pos_handle 과 동일 패턴).
    pub fn set_kv_handle(&mut self, handle: Arc<dyn KVCacheFormat>) {
        self.kv_handle = Some(handle);
    }

    /// 직접 접근이 필요한 caller (capability send 전 `set_has_secondary` 등)를 위해
    /// mutable ref를 노출한다.
    pub fn executor_mut(&mut self) -> &mut CommandExecutor {
        &mut self.executor
    }

    /// held-handle 에서 heartbeat payload 용 `KVSnapshot` 을 구성한다 (§5.2.1 (가) query).
    ///
    /// v1 `build_kv_snapshot`(decode_loop.rs)과 동일 partial — `current_pos`/`capacity` 만 채우고
    /// total_bytes 등은 placeholder(v1 도 0). handle 미주입 시 default(pos=0).
    fn build_kv_snapshot(&self) -> KVSnapshot {
        match &self.kv_handle {
            Some(h) => KVSnapshot {
                total_tokens: h.current_pos(),
                capacity: h.capacity(),
                ..KVSnapshot::default()
            },
            None => KVSnapshot::default(),
        }
    }
}

impl CommandSource for ResilienceAdapter {
    fn poll(&mut self) -> Result<Vec<EngineCommand>> {
        // β-4: pure 화 후에도 heartbeat 송출 잔존 (매핑 문서 4부 채택안 (가)).
        // drain 직전 interval 체크 + 송출. kv_snap 은 held-handle query 로 자체 구성.
        let kv_snap = self.build_kv_snapshot();
        self.executor.send_heartbeat_if_due(&kv_snap);
        Ok(self.executor.drain_commands())
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
    fn poll(&mut self) -> Result<Vec<EngineCommand>> {
        self.0.lock().expect("resilience mutex poisoned").poll()
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
