//! MSG-068, MGR-DAT-076 (Phase 2): Engine process 자신의 GPU 사용률 측정.
//!
//! Phase 1에서는 placeholder(항상 0.0)였으나 Phase 2에서 OpenCL queue
//! profiling(`CL_QUEUE_PROFILING_ENABLE`) 기반 실측으로 확장되었다.
//!
//! # 개요
//!
//! OpenCL 백엔드가 `CL_QUEUE_PROFILING_ENABLE`로 command queue를 구성하면
//! 각 커널 dispatch에 대해 `CL_PROFILING_COMMAND_{START,END}` 이벤트 정보가
//! 기록된다. Engine은 decode loop에서 주기적으로 이 이벤트를 drain하여
//! `(end - start)` ns를 누적한다. `GpuSelfMeter::sample(wall_elapsed)`는
//! 이 누적값을 wall-clock elapsed로 나누어 `[0.0, 1.0]` 사용률을 반환한다.
//!
//! # 정책
//!
//! - **Opt-in only**: Adreno에서 profiling 활성화 시 ~54 ms/token 오버헤드.
//!   CLI 플래그(`--heartbeat-gpu-profile`)로 명시적으로 켜야 한다.
//! - **Clamp**: 반환값은 `[0.0, 1.0]`로 clamp된다 (INV-091).
//! - **Fallback**: 측정 실패/meter 미주입 시 0.0 (INV-092, Heartbeat 송출을
//!   절대 차단하지 않음).
//! - **Warm-up**: 첫 샘플은 이전 기준점이 없으므로 0.0 반환.
//! - **비침투**: LuaPolicy `ctx.engine.gpu_pct`로만 노출. Pressure6D /
//!   EwmaReliefTable 계산에는 여전히 섞이지 않는다.

use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Duration;

/// Engine process 자가 GPU 사용률 측정기 추상화.
///
/// `sample(wall_elapsed)`는 가장 최근 샘플과 현재 사이의 GPU busy 시간을
/// wall-clock 경과로 나눈 `[0.0, 1.0]` 범위 값을 반환한다. 구현은 측정 실패
/// 시 0.0 fallback을 보장해야 한다 (INV-092).
pub trait GpuSelfMeter: Send + Sync {
    /// 현재 사용률 샘플을 반환한다.
    ///
    /// * `wall_elapsed` — 직전 샘플 이후 실제 경과 시간. 호출자(executor)가
    ///   측정 주기를 관리한다. 0 또는 음수 등 비정상 값이 들어오면 0.0 반환.
    ///
    /// 반환값은 `[0.0, 1.0]`로 clamp된다 (INV-091). 측정 실패 시 0.0 (INV-092).
    fn sample(&self, wall_elapsed: Duration) -> f64;
}

/// 항상 0.0을 반환하는 no-op meter. 테스트 및 CPU-only 구성에서 사용한다.
#[derive(Debug, Default, Clone, Copy)]
pub struct NoOpGpuMeter;

impl GpuSelfMeter for NoOpGpuMeter {
    fn sample(&self, _wall_elapsed: Duration) -> f64 {
        0.0
    }
}

/// OpenCL queue profiling 기반 GPU 사용률 누적기.
///
/// 백엔드 측 `flush_and_aggregate_profile()`이 매 decode step 후
/// `record_busy_ns()`로 누적한 값을 `sample()`이 drain하여 wall-clock으로
/// 정규화한다. 샘플링 시 누적값은 0으로 reset된다.
#[derive(Debug, Default)]
pub struct OpenClEventGpuMeter {
    /// 누적 GPU busy nanoseconds. drain-on-sample 방식(snapshot + reset).
    busy_ns: AtomicU64,
}

impl OpenClEventGpuMeter {
    /// 새 meter를 만든다. 누적값은 0에서 시작한다.
    pub fn new() -> Self {
        Self {
            busy_ns: AtomicU64::new(0),
        }
    }

    /// 백엔드가 flush 시점에 호출하는 누적 엔트리. `delta_ns`는 단일 kernel
    /// event의 `(end - start)` 값이며 여러 호출에 걸쳐 saturating add 된다.
    pub fn record_busy_ns(&self, delta_ns: u64) {
        // fetch_add는 wrap-around 하지만 실제 GPU 누적이 u64를 overflow할
        // 가능성은 없으므로(584년 @ 1 GHz) 단순 fetch_add로 충분.
        self.busy_ns.fetch_add(delta_ns, Ordering::Relaxed);
    }

    /// 테스트용 현재 누적값을 (reset 없이) 읽는다.
    #[cfg(test)]
    pub(crate) fn peek_busy_ns(&self) -> u64 {
        self.busy_ns.load(Ordering::Relaxed)
    }
}

impl GpuSelfMeter for OpenClEventGpuMeter {
    fn sample(&self, wall_elapsed: Duration) -> f64 {
        // snapshot + reset (drain semantics)
        let busy_ns = self.busy_ns.swap(0, Ordering::Relaxed);
        let wall_ns = wall_elapsed.as_nanos();
        if wall_ns == 0 {
            return 0.0;
        }
        let ratio = busy_ns as f64 / wall_ns as f64;
        clamp_unit(ratio)
    }
}

impl GpuSelfMeter for Arc<OpenClEventGpuMeter> {
    fn sample(&self, wall_elapsed: Duration) -> f64 {
        (**self).sample(wall_elapsed)
    }
}

fn clamp_unit(x: f64) -> f64 {
    if !x.is_finite() {
        0.0
    } else {
        x.clamp(0.0, 1.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn noop_meter_always_returns_zero() {
        // SPEC: MSG-068 (Phase 2 default), INV-092
        let m = NoOpGpuMeter;
        assert_eq!(m.sample(Duration::from_millis(0)), 0.0);
        assert_eq!(m.sample(Duration::from_millis(1000)), 0.0);
        assert_eq!(m.sample(Duration::from_secs(3600)), 0.0);
    }

    #[test]
    fn opencl_meter_first_sample_is_zero() {
        // SPEC: MSG-068 warm-up, INV-092
        let m = OpenClEventGpuMeter::new();
        assert_eq!(m.sample(Duration::from_millis(1000)), 0.0);
    }

    #[test]
    fn opencl_meter_records_and_drains() {
        // SPEC: MSG-068
        let m = OpenClEventGpuMeter::new();
        // 500ms busy out of 1000ms wall = 0.5
        m.record_busy_ns(500_000_000);
        let v = m.sample(Duration::from_millis(1000));
        assert!((v - 0.5).abs() < 1e-9, "expected 0.5, got {v}");
        // drain: 두 번째 샘플은 0.0
        assert_eq!(m.sample(Duration::from_millis(1000)), 0.0);
    }

    #[test]
    fn opencl_meter_clamps_over_one() {
        // SPEC: INV-091
        let m = OpenClEventGpuMeter::new();
        // busy > wall_elapsed (theoretically impossible, but must clamp)
        m.record_busy_ns(2_000_000_000);
        let v = m.sample(Duration::from_millis(1000));
        assert_eq!(v, 1.0);
    }

    #[test]
    fn opencl_meter_zero_wall_returns_zero() {
        // SPEC: INV-092 (degenerate input → 0.0 fallback)
        let m = OpenClEventGpuMeter::new();
        m.record_busy_ns(500_000_000);
        assert_eq!(m.sample(Duration::from_nanos(0)), 0.0);
    }

    #[test]
    fn opencl_meter_accumulates_multiple_records() {
        // SPEC: MSG-068
        let m = OpenClEventGpuMeter::new();
        m.record_busy_ns(100_000_000);
        m.record_busy_ns(150_000_000);
        m.record_busy_ns(50_000_000);
        assert_eq!(m.peek_busy_ns(), 300_000_000);
        let v = m.sample(Duration::from_millis(1000));
        assert!((v - 0.3).abs() < 1e-9, "expected 0.3, got {v}");
    }

    #[test]
    fn arc_wrapper_forwards_to_inner() {
        // 테스트: Arc<OpenClEventGpuMeter>도 GpuSelfMeter 구현.
        let m = Arc::new(OpenClEventGpuMeter::new());
        m.record_busy_ns(250_000_000);
        let v = GpuSelfMeter::sample(&m, Duration::from_millis(1000));
        assert!((v - 0.25).abs() < 1e-9);
    }

    #[test]
    fn arc_dyn_trait_object_works() {
        // executor가 저장하는 형태(Arc<dyn GpuSelfMeter>)가 동작.
        let inner = Arc::new(OpenClEventGpuMeter::new());
        inner.record_busy_ns(750_000_000);
        let dyn_meter: Arc<dyn GpuSelfMeter> = inner.clone();
        let v = dyn_meter.sample(Duration::from_millis(1000));
        assert!((v - 0.75).abs() < 1e-9);
    }

    #[test]
    fn nan_inf_safety() {
        // clamp_unit 안전망: NaN/Inf는 0.0으로.
        assert_eq!(clamp_unit(f64::NAN), 0.0);
        assert_eq!(clamp_unit(f64::INFINITY), 0.0);
        assert_eq!(clamp_unit(f64::NEG_INFINITY), 0.0);
        assert_eq!(clamp_unit(-0.5), 0.0);
        assert_eq!(clamp_unit(1.5), 1.0);
        assert_eq!(clamp_unit(0.42), 0.42);
    }
}
