//! 빌트인 eviction 정책을 technique-api `KVCacheStage` 표면으로 노출하는 어댑터 + linkme 등록.
//!
//! ADR-0004 M2-B②a: stage 축 레지스트리([`KV_CACHE_STAGES`])에 빌트인 LayerWide 정책 3종
//! (sliding/streaming/h2o)을 등록한다. 각 정책은 기존 [`EvictionPolicy::plan_keep`]
//! (`compact_parity` 가 in-place `evict*` 와 bit-identical 임을 증명)을 [`KVCacheStage::plan`]
//! 으로 위임하는 [`EvictionPolicyAsStage`] 어댑터로 감싼다.
//!
//! 본 단계(②a)는 **등록만** — 프로덕션 소비(match arm 교체 + plan executor)는 ②b. 그래서 등록은
//! 되어 있으나 아직 `find_stage` 로 구동되지 않는다(unwired). 등록 누락(linkme fat-LTO `--gc-sections`
//! silent drop)은 ②b 의 startup self-test 가 fail-fast 로 잡는다.
//!
//! **제외**: h2o_plus(per-head, `plan_keep`→`None`)는 head_score source(F5) 미완으로 단계 ⑤ deferred,
//! d2o(`EvictionPolicy` 아님, 가중 merge)는 M4, no_eviction("none")은 happy-path 라 match 밖.

use linkme::distributed_slice;
use technique_api::{
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageCtx, StageParams,
    WeightedMerge,
};

use super::{EvictionPolicy, H2OPolicy, SlidingWindowPolicy, StreamingLLMPolicy};

/// 기존 [`EvictionPolicy`](in-place `evict*` + `plan_keep`)를 plan-returning [`KVCacheStage`] 로 노출.
///
/// [`KVCacheStage::plan`] 은 [`EvictionPolicy::plan_keep`](layer-wide keep + 균등 merge)을
/// [`KVCachePlan`](`KeepSpec::LayerWide` + [`WeightedMerge`])으로 매핑한다. `plan_keep` 이 `None`
/// (per-head 등 단일 layer-wide keep 으로 표현 불가)이면 `None` 을 전파한다. 버퍼 변형은 엔진
/// executor 가 실행한다(ADR-0004 D1).
pub struct EvictionPolicyAsStage {
    inner: Box<dyn EvictionPolicy>,
}

impl EvictionPolicyAsStage {
    /// 주어진 정책을 stage 표면으로 감싼다.
    pub fn new(inner: Box<dyn EvictionPolicy>) -> Self {
        Self { inner }
    }
}

impl KVCacheStage for EvictionPolicyAsStage {
    fn name(&self) -> &str {
        self.inner.name()
    }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let (keep, merges) =
            self.inner
                .plan_keep(ctx.current_pos(), ctx.target_len(), ctx.importance())?;
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
            merges: merges.into_iter().map(uniform_to_weighted).collect(),
        })
    }
}

/// 균등 `format::Merge` → [`WeightedMerge`] 매핑. 현 빌트인 3정책은 모두 빈 merge 를 내므로 실질적으로
/// 빈 Vec→빈 Vec 이나, 균등 가중치 의미(`into` 포함 N개 동일 가중, Σ=1)를 보존한다 — 현 `apply_merges`
/// 의 uniform 거동과 정합. d2o 의 Eq.11 가중 merge 는 이 경로가 아니라 M4 에서 직접 산출한다.
fn uniform_to_weighted(m: crate::format::Merge) -> WeightedMerge {
    let n = (1 + m.from.len()) as f32; // into + from 토큰 수
    let w = 1.0 / n;
    WeightedMerge {
        into: m.into,
        into_weight: w,
        from: m.from.into_iter().map(|p| (p, w)).collect(),
    }
}

#[distributed_slice(KV_CACHE_STAGES)]
static SLIDING_STAGE: KVCacheStageReg = KVCacheStageReg {
    name: "sliding",
    make: |p: StageParams| {
        Box::new(EvictionPolicyAsStage::new(Box::new(
            SlidingWindowPolicy::new(p.eviction_window, p.protected_prefix),
        )))
    },
};

#[distributed_slice(KV_CACHE_STAGES)]
static STREAMING_STAGE: KVCacheStageReg = KVCacheStageReg {
    name: "streaming",
    make: |p: StageParams| {
        Box::new(EvictionPolicyAsStage::new(Box::new(
            StreamingLLMPolicy::new(p.sink_size, p.streaming_window),
        )))
    },
};

#[distributed_slice(KV_CACHE_STAGES)]
static H2O_STAGE: KVCacheStageReg = KVCacheStageReg {
    name: "h2o",
    make: |p: StageParams| {
        Box::new(EvictionPolicyAsStage::new(Box::new(H2OPolicy::new(
            p.keep_ratio,
            p.protected_prefix,
        ))))
    },
};

#[cfg(test)]
mod tests {
    use super::*;
    use technique_api::{find_stage, registered_names};

    /// 최소 StageCtx 스텁 — LayerWide 정책이 읽는 current_pos/target_len/importance 만 의미가 있고
    /// per-head/dequant accessor 는 trivial(이 단계 미사용).
    struct TestCtx {
        current_pos: usize,
        target_len: usize,
        importance: Option<Vec<f32>>,
    }
    impl StageCtx for TestCtx {
        fn current_pos(&self) -> usize {
            self.current_pos
        }
        fn target_len(&self) -> usize {
            self.target_len
        }
        fn layer_idx(&self) -> usize {
            0
        }
        fn importance(&self) -> Option<&[f32]> {
            self.importance.as_deref()
        }
        fn n_kv_heads(&self) -> usize {
            1
        }
        fn head_dim(&self) -> usize {
            1
        }
        fn head_score(&self, _kv_head: usize, _pos: usize) -> f32 {
            0.0
        }
        fn has_head_scores(&self) -> bool {
            false
        }
        fn dequant_k(&self, _pos: usize, _head: usize, _out: &mut [f32]) {}
    }

    #[test]
    fn builtins_registered() {
        // linkme 가 엔진의 등록을 슬라이스로 모으는지 (fat-LTO 생존은 ②b release self-test).
        let names = registered_names();
        for n in ["sliding", "streaming", "h2o"] {
            assert!(
                names.contains(&n),
                "'{n}' 등록 누락 (linkme distributed_slice)"
            );
        }
    }

    #[test]
    fn adapter_plan_matches_plan_keep_sliding() {
        // 어댑터 plan() 의 LayerWide keep 이 원본 plan_keep keep 과 동일한지 (faithful, score-free).
        let params = StageParams {
            eviction_window: 8,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 4,
            streaming_window: 8,
        };
        let reg = find_stage("sliding").expect("sliding 등록");
        let stage = (reg.make)(params);
        let ctx = TestCtx {
            current_pos: 200,
            target_len: 100,
            importance: None,
        };
        let plan = stage.plan(&ctx).expect("sliding plan Some");
        let direct = SlidingWindowPolicy::new(8, 4)
            .plan_keep(200, 100, None)
            .expect("direct plan_keep Some");
        match plan.keep {
            KeepSpec::LayerWide(keep) => {
                assert_eq!(keep, direct.0, "어댑터 keep == plan_keep keep")
            }
            KeepSpec::PerHead(_) => panic!("sliding 은 LayerWide 여야 한다"),
        }
        assert!(plan.merges.is_empty(), "sliding 은 merge 없음");
    }

    #[test]
    fn adapter_plan_matches_plan_keep_h2o_scored() {
        // score-based(H2O) 경로도 importance 를 ctx 로 받아 plan_keep 과 동일 keep 을 내는지.
        let params = StageParams {
            eviction_window: 8,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 4,
            streaming_window: 8,
        };
        let mut imp = vec![0.01f32; 200];
        imp[10] = 9.0;
        imp[20] = 8.0;
        let reg = find_stage("h2o").expect("h2o 등록");
        let stage = (reg.make)(params);
        let ctx = TestCtx {
            current_pos: 200,
            target_len: 100,
            importance: Some(imp.clone()),
        };
        let plan = stage.plan(&ctx).expect("h2o plan Some");
        let direct = H2OPolicy::new(0.5, 4)
            .plan_keep(200, 100, Some(&imp))
            .expect("direct plan_keep Some");
        match plan.keep {
            KeepSpec::LayerWide(keep) => {
                assert_eq!(keep, direct.0, "어댑터 keep == plan_keep keep")
            }
            KeepSpec::PerHead(_) => panic!("h2o 는 LayerWide 여야 한다"),
        }
    }
}
