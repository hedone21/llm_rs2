//! L2 instrument helpers — §13.8-H instrument macro helper 정책.
//!
//! 매크로 정의는 L2, expansion 내부에서 cross-cutting concrete 참조.
//! cfg gate(`profile` feature)로 production 빌드 시 zero-cost 컴파일 제거.

/// Pattern A 매크로 — Timer + QCF counter RAII 측정.
///
/// 사용: `let _t = qcf_timer!(NLL);` → cfg=on이면 Timer::start(&NLL),
/// cfg=off이면 unit `()` (no-op).
///
/// `$counter`는 `observability::profile::quality_metrics` 모듈 내 static 식별자
/// (NLL, QCF_KV_UNIFIED, QCF_KV_DRYRUN, QCF_WEIGHT_SWAP, QCF_LAYER_SKIP, DECODE_TOTAL).
#[macro_export]
macro_rules! qcf_timer {
    ($counter:ident) => {{
        #[cfg(feature = "profile")]
        let _g = $crate::observability::profile::quality_metrics::Timer::start(
            &$crate::observability::profile::quality_metrics::$counter,
        );
        #[cfg(not(feature = "profile"))]
        let _g = ();
        _g
    }};
}

/// Pattern B 매크로 — op_trace span 시작 (`Option<Instant>` 반환).
///
/// 사용: `let tok = op_start!();` → cfg=on이면 `op_trace::start()`,
/// cfg=off이면 `None`. embedding/FinalNorm/LmHead 등 OpKind 인자가
/// 불필요한 (phase hook 미사용) 호출 지점에서 사용.
#[macro_export]
macro_rules! op_start {
    () => {{
        #[cfg(feature = "profile")]
        let _tok = $crate::observability::profile::op_trace::start();
        #[cfg(not(feature = "profile"))]
        let _tok: Option<std::time::Instant> = None;
        _tok
    }};
}

/// Pattern B 매크로 — OpKind와 함께 시작 (phase hook 활성 경로).
///
/// `op_trace::start_op(OpKind::$kind)`를 호출하여 PhaseHook이 등록된 경우
/// `on_op_start(kind)`를 fire. forward_gen 11개 op 측정에 사용.
#[macro_export]
macro_rules! op_start_kind {
    ($kind:ident) => {{
        #[cfg(feature = "profile")]
        let _tok =
            $crate::observability::profile::op_trace::start_op($crate::op_kind::OpKind::$kind);
        #[cfg(not(feature = "profile"))]
        let _tok: Option<std::time::Instant> = None;
        _tok
    }};
}

/// Pattern B 매크로 — span 종료 + 라벨링.
///
/// `$tok`: `op_start!` 또는 `op_start_kind!` 반환값.
/// `$kind`: `OpKind` variant (식별자).
/// `$backend`: `&Arc<dyn Backend>` (sync 호출용).
/// `$is_gpu`: bool (sync mode에서만 의미 있음).
#[macro_export]
macro_rules! op_record {
    ($tok:expr, $kind:ident, $backend:expr, $is_gpu:expr) => {{
        #[cfg(feature = "profile")]
        $crate::observability::profile::op_trace::record(
            $tok,
            $crate::op_kind::OpKind::$kind,
            $backend,
            $is_gpu,
        );
        #[cfg(not(feature = "profile"))]
        {
            // unused 경고 방지 (cfg=off 빌드에서도 평가 부수효과 보존).
            let _ = ($tok, $backend, $is_gpu);
        }
    }};
}

/// Pattern B 매크로 — forward call 카운터 증가.
///
/// `op_trace::note_forward_call()`를 호출. 단일 decode token당 1회.
#[macro_export]
macro_rules! op_note_forward_call {
    () => {{
        #[cfg(feature = "profile")]
        $crate::observability::profile::op_trace::note_forward_call();
    }};
}
