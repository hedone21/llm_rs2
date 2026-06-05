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

use anyhow::Result;
use linkme::distributed_slice;
use technique_api::{
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageCtx, StageParams,
    WeightedMerge,
};

use super::{EvictionPolicy, H2OPolicy, SlidingWindowPolicy, StreamingLLMPolicy};
use crate::pressure::kv_cache::KVCache;

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

// ── ②b: KVCachePlan executor + StageBackedPolicy 역어댑터 (World B) ──────────────

/// [`KVCacheStage`] 가 산출한 [`KVCachePlan`] 을 `&mut KVCache` 에 적용한다(ADR-0004 D1 — 변형은
/// 엔진 독점). `StandardFormat::compact` 의 빈-merge 경로와 동일: `compact_keep_positions(keep, 0)` +
/// `set_current_pos(keep.len())`. compact_parity 가 이 경로 ≡ in-place `evict*` 를 4정책×3dtype 에서
/// 증명하므로, plan keep 이 `plan_keep` keep 과 같으면(②a 어댑터 faithful) 버퍼 bit-identical 무회귀.
fn execute_kv_plan(cache: &mut KVCache, plan: &KVCachePlan) -> Result<()> {
    match &plan.keep {
        KeepSpec::LayerWide(keep) => {
            if !plan.merges.is_empty() {
                // (M4-b) 가중 merge 를 compact 이전 좌표계에서 in-place 적용(scatter_reduce 와
                // bit-identical, F32/F16/Q4_0). ADR-0004 §4(M4 정정) — Q4_0 merge 활성.
                crate::pressure::standard_format::apply_weighted_merges(cache, &plan.merges);
            }
            cache.compact_keep_positions(keep, 0)?;
            cache.set_current_pos(keep.len());
            Ok(())
        }
        KeepSpec::PerHead(_) => {
            // per-head executor = 단계 ⑤(h2o_plus, head_score source 미완) deferred.
            // 현 빌트인 3정책은 PerHead 미생산이라 도달 불가.
            anyhow::bail!("per-head executor not implemented (단계 ⑤ deferred)")
        }
    }
}

/// `&KVCache`(+ budget + scores) 위로 구현한 [`StageCtx`] — 정적 단계의 읽기 borrow(ADR-0004 D5).
///
/// ②b 는 LayerWide 정책(sliding/streaming/h2o)만 구동하므로 `current_pos`/`target_len`/`importance`
/// 만 실질 사용된다. per-head(`head_score`/`has_head_scores`)·raw-K(`dequant_k`)·`layer_idx` 는
/// head_importance forward(F5/⑤)·d2o(M4)에서 채운다 — 현재는 미plumb 안전 기본값.
struct KVStageCtx<'a> {
    cache: &'a KVCache,
    target_len: usize,
    importance: Option<&'a [f32]>,
}

impl StageCtx for KVStageCtx<'_> {
    fn current_pos(&self) -> usize {
        self.cache.current_pos
    }
    fn target_len(&self) -> usize {
        self.target_len
    }
    fn layer_idx(&self) -> usize {
        0 // per-layer(d2o) = M4
    }
    fn importance(&self) -> Option<&[f32]> {
        self.importance
    }
    fn n_kv_heads(&self) -> usize {
        self.cache.kv_heads()
    }
    fn head_dim(&self) -> usize {
        self.cache.head_dim()
    }
    fn head_score(&self, _kv_head: usize, _pos: usize) -> f32 {
        0.0 // head_importance forward = F5/⑤
    }
    fn has_head_scores(&self) -> bool {
        false // ②b 미plumb
    }
    fn dequant_k(&self, _pos: usize, _head: usize, _out: &mut [f32]) {
        // raw-K 읽기 = d2o(M4) 에서 채움. ②b LayerWide 정책 미사용.
    }
}

/// [`KVCacheStage`](plan-returning)를 레거시 [`EvictionPolicy`](in-place)로 노출하는 역어댑터(ADR-0004).
///
/// 프로덕션 eviction 경로(`run_policy_eviction` → `evict*`)는 구조 불변으로 두되, 내부에서 stage 의
/// plan 을 [`execute_kv_plan`] 으로 실행한다 — 즉 sliding/streaming/h2o 의 evict 가 in-place(World A)
/// 에서 plan→compact(World B)로 바뀐다. compact_parity 가 등가성을 보장(무회귀).
pub struct StageBackedPolicy {
    stage: Box<dyn KVCacheStage>,
}

impl StageBackedPolicy {
    /// 주어진 stage 를 `EvictionPolicy` 표면으로 감싼다.
    pub fn new(stage: Box<dyn KVCacheStage>) -> Self {
        Self { stage }
    }

    /// 읽기 ctx 로 plan 산출(immutable borrow) → borrow 종료 후 executor 가 `&mut` 로 실행.
    fn run(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: Option<&[f32]>,
    ) -> Result<()> {
        let plan = {
            let ctx = KVStageCtx {
                cache: &*cache,
                target_len,
                importance,
            };
            self.stage.plan(&ctx)
        };
        if let Some(plan) = plan {
            execute_kv_plan(cache, &plan)?;
        }
        Ok(())
    }
}

impl EvictionPolicy for StageBackedPolicy {
    fn should_evict(&self, _cache: &KVCache, _mem_available: usize) -> bool {
        // WHEN(트리거)은 엔진 소유(ADR-0004 D6) — `run_policy_eviction` 의 target_len/MIN_EVICT
        // 가드가 결정한다. 프로덕션 미호출(should_evict 의미는 구체 정책 테스트에서 검증). 엔진 위임.
        true
    }

    fn evict(&self, cache: &mut KVCache, target_len: usize) -> Result<()> {
        self.run(cache, target_len, None)
    }

    fn evict_with_scores(
        &self,
        cache: &mut KVCache,
        target_len: usize,
        importance: &[f32],
    ) -> Result<()> {
        self.run(cache, target_len, Some(importance))
    }

    fn name(&self) -> &str {
        self.stage.name()
    }
}

/// 빌트인 LayerWide 기법(sliding/streaming/h2o)이 `KV_CACHE_STAGES` 에 등록됐는지 단언한다 — eviction
/// CacheManager build 진입 시 1회 호출(ADR-0003 §4). fat-LTO `--gc-sections` 가 linkme 등록을 silent
/// drop 하면 누락 기법에 대해 `Err` 로 fail-fast 한다(release 에서 정책 이름 미해석 → 조용한 폴백 방지).
pub fn ensure_builtin_stages_registered() -> Result<()> {
    for name in ["sliding", "streaming", "h2o"] {
        if technique_api::find_stage(name).is_none() {
            anyhow::bail!(
                "내장 KVCacheStage '{name}' 미등록 — linkme fat-LTO --gc-sections silent drop 의심\
                 (ADR-0003 §4). stage_registry 의 #[distributed_slice] 등록이 링크되지 않음."
            );
        }
    }
    Ok(())
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
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::{Buffer, DType};
    use crate::memory::host::shared::SharedBuffer;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    use std::sync::Arc;
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

    // ADR-0003 cross-crate linkme 실증 결과(M3): **dev-dep 선언만으로는 부족**하다. Rust 는 미참조
    // 의존 rlib 을 링크에서 제외하므로 `#[distributed_slice]` 등록이 누락된다(실측 — forcing 없으면
    // find_stage None). 따라서 technique crate 의 등록을 활성화하려면 의존 1줄에 더해 **force-link
    // 참조 1줄**(`use <crate> as _;`)이 designated 지점에 필요하다. 즉 확장 비용 = dep 1줄 + force-link
    // 1줄(둘 다 기계적, 기존 로직 수정 0 → OCP 유지). 상세: ADR-0003 §4 (M3 정정).
    use example_keep_recent as _;

    #[test]
    fn example_technique_crate_visible_to_engine() {
        // force-link(위 `use ... as _`) 가 걸린 상태에서 별도 technique crate 의 등록이 엔진 뷰의
        // KV_CACHE_STAGES 에 나타나는가 — "폴더 추가 + dep 1줄 + force-link 1줄 = 기법 추가" 검증.
        assert!(
            find_stage("example_keep_recent").is_some(),
            "force-link 후 예제 technique crate 등록이 엔진에서 보여야 한다"
        );
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

    // ── StageBackedPolicy parity: World B(plan→compact) ≡ in-place evict(World A) ──
    // ②a 어댑터 faithful + compact_parity(plan_keep→compact ≡ in-place) 의 합성을, 프로덕션
    // 메커니즘 전체(find_stage→make→StageBackedPolicy→KVStageCtx→plan→execute_kv_plan)로 직접 확인.
    const PHD: usize = 32; // head_dim = QK4_0 → Q4_0 위치당 1 block
    const PMAX: usize = 128;

    fn pbytes(dt: DType) -> usize {
        match dt {
            DType::F32 => PHD * 4,
            DType::F16 => PHD * 2,
            DType::Q4_0 => {
                (PHD / crate::quant::QK4_0) * std::mem::size_of::<crate::quant::BlockQ4_0>()
            }
            o => panic!("unsupported dtype {o:?}"),
        }
    }

    /// 위치 p 의 모든 byte = (p+1) (K), +128 (V) — distinct 라 잘못된 keep 은 byte 비교로 잡힘.
    fn mk(dt: DType, n: usize) -> KVCache {
        let bpp = pbytes(dt);
        let kb = Arc::new(SharedBuffer::new(PMAX * bpp, dt));
        let vb = Arc::new(SharedBuffer::new(PMAX * bpp, dt));
        unsafe {
            let (kp, vp) = (kb.as_mut_ptr(), vb.as_mut_ptr());
            for p in 0..n {
                let byte = (p + 1) as u8;
                for b in 0..bpp {
                    *kp.add(p * bpp + b) = byte;
                    *vp.add(p * bpp + b) = byte.wrapping_add(128);
                }
            }
        }
        let be = Arc::new(CpuBackend::new());
        let sh = Shape::new(vec![1, PMAX, 1, PHD]);
        let mut c = KVCache::new(
            Tensor::new(sh.clone(), kb, be.clone()),
            Tensor::new(sh, vb, be),
            PMAX,
        );
        c.current_pos = n;
        c
    }

    fn region(c: &KVCache) -> (Vec<u8>, Vec<u8>) {
        let nb = c.current_pos * pbytes(c.k_buffer.dtype());
        unsafe {
            (
                std::slice::from_raw_parts(c.k_buffer.as_mut_ptr() as *const u8, nb).to_vec(),
                std::slice::from_raw_parts(c.v_buffer.as_mut_ptr() as *const u8, nb).to_vec(),
            )
        }
    }

    fn sb_params() -> StageParams {
        StageParams {
            eviction_window: 10,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 4,
            streaming_window: 6,
        }
    }

    #[test]
    fn stage_backed_evict_parity_sliding() {
        for dt in [DType::F32, DType::F16, DType::Q4_0] {
            let mut a = mk(dt, 40);
            SlidingWindowPolicy::new(10, 4).evict(&mut a, 20).unwrap();
            let mut b = mk(dt, 40);
            let stage = (find_stage("sliding").unwrap().make)(sb_params());
            StageBackedPolicy::new(stage).evict(&mut b, 20).unwrap();
            assert_eq!(a.current_pos, b.current_pos, "sliding[{dt:?}] current_pos");
            assert_eq!(region(&a), region(&b), "sliding[{dt:?}] valid-region byte");
        }
    }

    #[test]
    fn stage_backed_evict_parity_h2o_scored() {
        let imp: Vec<f32> = (0..PMAX).map(|i| (PMAX - i) as f32).collect();
        for dt in [DType::F32, DType::F16, DType::Q4_0] {
            let mut a = mk(dt, 40);
            H2OPolicy::new(0.5, 4)
                .evict_with_scores(&mut a, 20, &imp)
                .unwrap();
            let mut b = mk(dt, 40);
            let stage = (find_stage("h2o").unwrap().make)(sb_params());
            StageBackedPolicy::new(stage)
                .evict_with_scores(&mut b, 20, &imp)
                .unwrap();
            assert_eq!(a.current_pos, b.current_pos, "h2o[{dt:?}] current_pos");
            assert_eq!(region(&a), region(&b), "h2o[{dt:?}] valid-region byte");
        }
    }
}
