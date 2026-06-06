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
    TensorDtype, TensorHandle, TensorKind, TensorShape, WeightedMerge,
};

use super::{EvictionPolicy, H2OPolicy, SlidingWindowPolicy, StreamingLLMPolicy};
use crate::buffer::DType;
use crate::pressure::d2o_handler::{D2OConfig, D2OStage, dequantize_k, dequantize_v};
use crate::pressure::kv_cache::KVCache;

// ADR-0004 §8: CAOTE production 활성화. feature `caote` ON 시 caote crate 를 force-link 한다 —
// dep 선언만으로는 미참조 rlib 이 링크 제외돼 `#[distributed_slice]` 등록이 누락되기 때문(ADR-0003 §4
// M3 실측). 이 1줄이 production 바이너리에서 `find_stage("caote")` 를 가시화한다(session score_based
// 경유 value-aware 동작). feature OFF = 미링크 → `--eviction-policy caote` 는 unknown 으로 graceful fail.
#[cfg(feature = "caote")]
use caote as _;

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
///
/// pub(crate): M4-c d2o 동등성 테스트가 D2OStage plan 을 실행해 D2OHandler 와 비교하는 데 쓴다.
pub(crate) fn execute_kv_plan(cache: &mut KVCache, plan: &KVCachePlan) -> Result<()> {
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

/// 엔진 `DType` → technique-api `TensorDtype` 매핑(핸들 진단용; 읽기 산출은 항상 f32).
fn map_dtype(dt: DType) -> TensorDtype {
    match dt {
        DType::F16 => TensorDtype::F16,
        DType::Q4_0 => TensorDtype::Q4_0,
        _ => TensorDtype::F32,
    }
}

/// `tensor(Key)` 핸들 — raw K 를 `dequantize_k` 정본으로 읽는다(D2OHandler 와 bit-identical).
struct KeyHandle<'a> {
    cache: &'a KVCache,
}
impl TensorHandle for KeyHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.cache.current_pos(),
            cols: self.cache.head_dim(),
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        map_dtype(self.cache.k_buffer.dtype())
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        dequantize_k(self.cache, row, kv_head, self.cache.head_dim(), out);
    }
}

/// `tensor(Value)` 핸들 — raw V 를 `dequantize_v` 정본으로 읽는다(CAOTE 의 v_i).
struct ValueHandle<'a> {
    cache: &'a KVCache,
}
impl TensorHandle for ValueHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.cache.current_pos(),
            cols: self.cache.head_dim(),
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        map_dtype(self.cache.v_buffer.dtype())
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        dequantize_v(self.cache, row, kv_head, self.cache.head_dim(), out);
    }
}

/// `tensor(Scores)`/`tensor(AttnWeights)` 핸들 — per-(kv_head,pos) f32 스칼라.
/// 원천 레이아웃 `[n_kv_heads * max_seq]` row-major(accumulator stride=max_seq).
struct ScalarHandle<'a> {
    data: &'a [f32],
    rows: usize,
    max_seq: usize,
}
impl TensorHandle for ScalarHandle<'_> {
    fn shape(&self) -> TensorShape {
        TensorShape {
            rows: self.rows,
            cols: 1,
            per_head: true,
        }
    }
    fn dtype(&self) -> TensorDtype {
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        out[0] = self
            .data
            .get(kv_head * self.max_seq + row)
            .copied()
            .unwrap_or(0.0);
    }
}

/// `&KVCache`(+ budget + scores) 위로 구현한 [`StageCtx`] (ADR-0004 D5, M-A 통합).
///
/// 모든 텐서/스코어 읽기는 [`StageCtx::tensor`] 단일 경로로 흐른다: Key/Value 핸들은 항상,
/// Scores/AttnWeights 는 `new()` 에 슬라이스가 공급될 때만 `Some`. flat `importance()` 만 zero-copy 직접
/// 노출(D1 예외). builtin LayerWide(sliding/streaming/h2o) + d2o(tensor(Key))는 production 에서 구동,
/// Scores/AttnWeights 공급은 현재 host 테스트(CAOTE) 경로 — production eviction-hook threading 은 CLI
/// 배선(D-3 deferred)과 함께 후속.
pub(crate) struct KVStageCtx<'a> {
    cache: &'a KVCache,
    target_len: usize,
    importance: Option<&'a [f32]>,
    key_handle: KeyHandle<'a>,
    value_handle: ValueHandle<'a>,
    scores_handle: Option<ScalarHandle<'a>>,
    attn_handle: Option<ScalarHandle<'a>>,
}

impl<'a> KVStageCtx<'a> {
    /// 엔진 eviction 경로(+ d2o 동등성/CAOTE host 테스트)가 `&KVCache` 위로 ctx 를 만든다.
    /// `head_scores`/`last_attn`: per-(kv_head,pos) `[n_kv_heads*max_seq]`. `None`=미공급(`tensor()`→None).
    pub(crate) fn new(
        cache: &'a KVCache,
        target_len: usize,
        importance: Option<&'a [f32]>,
        head_scores: Option<&'a [f32]>,
        last_attn: Option<&'a [f32]>,
    ) -> Self {
        let rows = cache.current_pos();
        let max_seq = cache.max_seq_len;
        Self {
            cache,
            target_len,
            importance,
            key_handle: KeyHandle { cache },
            value_handle: ValueHandle { cache },
            scores_handle: head_scores.map(|data| ScalarHandle {
                data,
                rows,
                max_seq,
            }),
            attn_handle: last_attn.map(|data| ScalarHandle {
                data,
                rows,
                max_seq,
            }),
        }
    }
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
    /// 단일 텐서 접근 — Key/Value 항상, Scores/AttnWeights 는 공급 시. dequant_k/v·head_score·
    /// attn_weight 등 sugar 는 technique-api default 가 이 위에 얹힌다.
    fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
        match kind {
            TensorKind::Key => Some(&self.key_handle),
            TensorKind::Value => Some(&self.value_handle),
            TensorKind::Scores => self.scores_handle.as_ref().map(|h| h as &dyn TensorHandle),
            TensorKind::AttnWeights => self.attn_handle.as_ref().map(|h| h as &dyn TensorHandle),
        }
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
            let ctx = KVStageCtx::new(cache, target_len, importance, None, None);
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

/// d2o(M4-c) — `D2OStage`(plan-returning, 가중 merge + EMA). non-alloc 기본 D2OConfig: StageParams
/// 에 d2o 전용 필드(ema_beta/merge_e/use_layer_allocation)가 없어 protected_prefix/keep_ratio 만 매핑
/// 하고 나머지는 D2OConfig::default()(non-alloc, ema_beta=0.7, merge_e=0.1). **production d2o 는
/// 여전히 if-branch(session.rs:604·build_bench_loop.rs:72)=D2OHandler 가 처리**(layer-alloc 지원 +
/// 비권장 정책) — 본 등록은 proven-equivalent(non-alloc) available 표면.
#[distributed_slice(KV_CACHE_STAGES)]
static D2O_STAGE: KVCacheStageReg = KVCacheStageReg {
    name: "d2o",
    make: |p: StageParams| {
        Box::new(D2OStage::new(D2OConfig {
            protected_prefix: p.protected_prefix,
            keep_ratio: p.keep_ratio,
            ..D2OConfig::default()
        }))
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
        // LayerWide 정책만 구동 → 텐서 미공급(None). head_score/dequant_* 는 default sugar(None→trivial).
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
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
    fn d2o_stage_registered() {
        // (M4-c) D2OStage 가 "d2o" 로 KV_CACHE_STAGES 에 등록됐는지 — find_stage 해석 + make 가능.
        // production 은 if-branch(D2OHandler) 가 가로채므로 이 등록은 proven-equivalent available
        // 표면(release fat-LTO 에서도 생존해야). make 로 D2OStage 인스턴스 생성 가능 확인.
        let reg = find_stage("d2o").expect("d2o stage 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "d2o");
        let params = StageParams {
            eviction_window: 0,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 0,
            streaming_window: 0,
        };
        let stage = (reg.make)(params);
        assert_eq!(stage.name(), "d2o");
    }

    // ADR-0003 cross-crate linkme 실증 결과(M3): **dev-dep 선언만으로는 부족**하다. Rust 는 미참조
    // 의존 rlib 을 링크에서 제외하므로 `#[distributed_slice]` 등록이 누락된다(실측 — forcing 없으면
    // find_stage None). 따라서 technique crate 의 등록을 활성화하려면 의존 1줄에 더해 **force-link
    // 참조 1줄**(`use <crate> as _;`)이 designated 지점에 필요하다. 즉 확장 비용 = dep 1줄 + force-link
    // 1줄(둘 다 기계적, 기존 로직 수정 0 → OCP 유지). 상세: ADR-0003 §4 (M3 정정).
    use example_keep_recent as _;
    // CAOTE 의 force-link 는 production(module-level `#[cfg(feature = "caote")] use caote as _`)
    // 가 담당한다 — `--features caote` 테스트 시 그 cfg 가 활성이라 별도 test-only force-link 불필요.

    #[test]
    fn example_technique_crate_visible_to_engine() {
        // force-link(위 `use ... as _`) 가 걸린 상태에서 별도 technique crate 의 등록이 엔진 뷰의
        // KV_CACHE_STAGES 에 나타나는가 — "폴더 추가 + dep 1줄 + force-link 1줄 = 기법 추가" 검증.
        assert!(
            find_stage("example_keep_recent").is_some(),
            "force-link 후 예제 technique crate 등록이 엔진에서 보여야 한다"
        );
    }

    #[cfg(feature = "caote")]
    #[test]
    fn caote_stage_visible_and_value_aware_executes() {
        // (M-F) CAOTE crate 의 cross-crate 등록 + KVStageCtx(V 공급)로 value-aware plan 산출 →
        // execute_kv_plan 실행. mk() 가 토큰별 distinct V 를 채우므로 criticality(‖v_i−o_h‖)가 V 에
        // 의존 → 기법이 [`StageCtx::tensor`]`(Value)` 로 V 를 직접 읽어 자체 metric 을 계산함을 증명.
        let reg = find_stage("caote").expect("caote 등록이 엔진에서 보여야 한다");
        let stage = (reg.make)(StageParams {
            eviction_window: 0,
            protected_prefix: 0,
            keep_ratio: 0.0,
            sink_size: 0,
            streaming_window: 0,
        });
        let mut c = mk(DType::F32, 8); // kv_heads=1, head_dim=PHD, V distinct per pos, current_pos=8
        let imp = vec![1.0f32; 8]; // 균일 가중 → criticality 는 V 가 결정
        let plan = {
            let ctx = KVStageCtx::new(&c, 4, Some(&imp), None, None);
            assert!(
                ctx.tensor(TensorKind::Value).is_some(),
                "KVStageCtx 는 Value 핸들을 항상 공급"
            );
            stage.plan(&ctx).expect("plan Some")
        };
        match &plan.keep {
            KeepSpec::LayerWide(k) => {
                assert_eq!(k.len(), 4, "target_len=4 만큼 유지");
                assert!(k.windows(2).all(|w| w[0] < w[1]), "ascending keep");
                assert!(k.iter().all(|&p| p < 8), "유효 위치");
            }
            KeepSpec::PerHead(_) => panic!("v1 CAOTE 는 LayerWide"),
        }
        assert!(plan.merges.is_empty());
        execute_kv_plan(&mut c, &plan).unwrap();
        assert_eq!(c.current_pos(), 4, "executor 가 keep.len() 로 compact");
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

    #[test]
    fn kvstagectx_dequant_k_reads_f32() {
        // (M-D) dequant_k sugar(→ tensor(Key) → KeyHandle → d2o_handler::dequantize_k)로 raw K(F32) 읽기.
        // 완전 통합 후에도 기존 dequant_k 시그니처·결과가 보존됨을 확인.
        let mut c = mk(DType::F32, 8);
        let off = c.offset(5, 0);
        {
            let k = c.k_buffer.as_mut_slice::<f32>();
            for d in 0..PHD {
                k[off + d] = (d as f32) * 0.5 + 1.0;
            }
        }
        let ctx = KVStageCtx::new(&c, 0, None, None, None);
        let mut out = vec![0.0f32; PHD];
        ctx.dequant_k(5, 0, &mut out);
        for d in 0..PHD {
            assert_eq!(out[d], (d as f32) * 0.5 + 1.0, "dequant_k F32 d={d}");
        }
        // tensor(Key) 핸들 shape/dtype 계약.
        let kh = ctx.tensor(TensorKind::Key).expect("Key handle 항상 존재");
        assert_eq!(kh.shape().cols, PHD);
        assert!(kh.shape().per_head);
        assert_eq!(kh.dtype(), TensorDtype::F32);
    }

    #[test]
    fn kvstagectx_dequant_v_reads_f32() {
        // (M-C/M-D) dequant_v sugar(→ tensor(Value) → ValueHandle → dequantize_v)로 raw V(F32) 읽기.
        let mut c = mk(DType::F32, 8);
        let off = c.offset(5, 0);
        {
            let v = c.v_buffer.as_mut_slice::<f32>();
            for d in 0..PHD {
                v[off + d] = (d as f32) * 0.25 - 2.0;
            }
        }
        let ctx = KVStageCtx::new(&c, 0, None, None, None);
        let mut out = vec![0.0f32; PHD];
        ctx.dequant_v(5, 0, &mut out);
        for d in 0..PHD {
            assert_eq!(out[d], (d as f32) * 0.25 - 2.0, "dequant_v F32 d={d}");
        }
    }

    #[test]
    fn kvstagectx_scores_and_attn_handles() {
        // (M-D) Scores/AttnWeights 핸들 — 공급 시 per-(kv_head,pos) 스칼라 읽기, 미공급 시 None.
        let c = mk(DType::F32, 4); // kv_heads=1
        let max_seq = c.max_seq_len;
        let scores: Vec<f32> = (0..max_seq).map(|p| p as f32 + 0.5).collect();
        let attn: Vec<f32> = (0..max_seq).map(|p| p as f32 * 10.0).collect();
        let ctx = KVStageCtx::new(&c, 0, None, Some(&scores), Some(&attn));
        assert!(ctx.has_head_scores());
        assert!(ctx.has_attn_weights());
        assert_eq!(ctx.head_score(0, 3), 3.5);
        assert_eq!(ctx.attn_weight(0, 2), 20.0);
        // 미공급 ctx → None / trivial.
        let bare = KVStageCtx::new(&c, 0, None, None, None);
        assert!(!bare.has_head_scores());
        assert!(!bare.has_attn_weights());
        assert_eq!(bare.head_score(0, 3), 0.0);
        assert!(bare.tensor(TensorKind::Scores).is_none());
    }
}
