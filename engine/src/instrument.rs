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
