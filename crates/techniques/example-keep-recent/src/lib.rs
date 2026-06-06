//! 예제 technique crate — "폴더만 추가 = 엔진 코어 수정 0" (ADR-0003) 검증 + 기여자 템플릿(M5).
//!
//! 본 crate 는 [`technique_api`] 에만 의존해 [`KVCacheStage`] 를 구현하고 [`KV_CACHE_STAGES`] 슬라이스에
//! `#[distributed_slice]` 로 자기를 등록한다. 엔진 타입(`KVCache`/`Backend`)을 일절 참조하지 않는다 —
//! stage 축에 새 멤버를 더하는 비용이 다른 축(format/hardware) 코드를 0 만큼 건드림(가산 확장).
//!
//! **알고리즘**: 최근 `target_len` 토큰만 유지(sliding 의 prefix=0 변형). [`StageCtx`] 의
//! `current_pos`/`target_len` 만 읽는 순수 계산이며, 버퍼 변형은 엔진 executor 가 반환된 plan 을
//! `compact` 로 실행한다(plan-returning, ADR-0004 D1). CLI 선택: `--eviction-policy example_keep_recent`.

use linkme::distributed_slice;
use technique_api::{
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageCtx, StageParams,
};

/// 최근 `target_len` 토큰만 유지하는 stage.
struct KeepRecent;

impl KVCacheStage for KeepRecent {
    fn name(&self) -> &str {
        "example_keep_recent"
    }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let current = ctx.current_pos();
        let target = ctx.target_len();
        if current <= target {
            return None; // 축소 불필요 — no-op
        }
        let keep: Vec<usize> = (current - target..current).collect(); // ascending
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
            merges: Vec::new(),
        })
    }
}

/// 등록 — 엔진은 construction 시 `find_stage("example_keep_recent")` 로 이 항목을 찾는다.
#[distributed_slice(KV_CACHE_STAGES)]
static EXAMPLE_KEEP_RECENT: KVCacheStageReg = KVCacheStageReg {
    name: "example_keep_recent",
    make: |_params: StageParams| Box::new(KeepRecent),
};

#[cfg(test)]
mod tests {
    use super::*;
    use technique_api::find_stage;

    struct Ctx {
        cur: usize,
        tgt: usize,
    }
    impl StageCtx for Ctx {
        fn current_pos(&self) -> usize {
            self.cur
        }
        fn target_len(&self) -> usize {
            self.tgt
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn importance(&self) -> Option<&[f32]> {
            None
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            1
        }
        // tensor()만 구현 — head_score/dequant_* 등은 technique-api default sugar(None→trivial).
        fn tensor(
            &self,
            _kind: technique_api::TensorKind,
        ) -> Option<&dyn technique_api::TensorHandle> {
            None
        }
    }

    fn params() -> StageParams {
        StageParams {
            eviction_window: 0,
            protected_prefix: 0,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        }
    }

    #[test]
    fn registers_into_slice() {
        let reg =
            find_stage("example_keep_recent").expect("예제 stage 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "example_keep_recent");
    }

    #[test]
    fn plan_keeps_recent_window() {
        let stage = (find_stage("example_keep_recent").unwrap().make)(params());
        assert_eq!(stage.name(), "example_keep_recent");
        let plan = stage.plan(&Ctx { cur: 100, tgt: 30 }).expect("plan Some");
        match plan.keep {
            KeepSpec::LayerWide(k) => assert_eq!(k, (70..100).collect::<Vec<_>>()),
            KeepSpec::PerHead(_) => panic!("LayerWide 여야 한다"),
        }
        assert!(plan.merges.is_empty());
        // current <= target → no-op(None).
        assert!(stage.plan(&Ctx { cur: 20, tgt: 30 }).is_none());
    }
}
