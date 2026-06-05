//! technique-api — 확장 기법(stage 축)이 **엔진 코어 수정 0** 으로 자기를 등록하는 가산 표면.
//!
//! ADR-0003: 확장 메커니즘 = 정적 링크 technique crate + linkme 자동 등록. 각 기법은 별도 crate
//! (`crates/techniques/<name>/`)에서 본 crate 에만 의존해 [`KVCacheStage`] 를 구현하고
//! [`KV_CACHE_STAGES`] 슬라이스에 `#[distributed_slice]` 로 자기를 제출한다. 엔진은 construction
//! 시 그 슬라이스를 읽어 기법을 고른다 (closed match arm 제거 → OCP).
//!
//! ADR-0004: stage 축 확장 기법(eviction/merge)은 **단일 plan-returning trait [`KVCacheStage`]**
//! 로 통일된다 (엔진측 저장-표현 trait `KVCacheFormat` 의 형제). 기법은 [`StageCtx`] 를 *읽고* [`KVCachePlan`]
//! (보존 토큰 + 가중 merge 계획)을 반환할 뿐, 버퍼를 직접 변형하지 않는다 — 변형은 엔진이 plan 을
//! `compact` 로 실행해 독점한다(D1). 상태(d2o EMA 등)는 plugin struct 가 `&self` + interior-mutability
//! 로 보유한다(D4); ctx 로 thread 하지 않는다.
//!
//! 의존 방향: `engine → technique-api ← technique crate` (단방향, 순환 없음). 그래서 본 crate 는
//! 엔진 타입(`KVCache`/`Backend`)을 **참조하지 않는다** — 기법이 읽어야 하는 캐시 상태는 본 crate 가
//! 정의하는 읽기 추상 [`StageCtx`] 로 노출하고, 엔진이 그것을 `&KVCache` 위로 구현한다(D5). 정적
//! 단계엔 borrow, 미래 `.so` C-ABI 단계엔 동일 추상이 C accessor/flat 스냅샷으로 교체 — forward-compatible.

use linkme::distributed_slice;

/// 기법이 캐시를 읽는 추상(ADR-0004 D5). 엔진이 `&KVCache`(+ scores/budget) 위로 구현한다.
///
/// **dyn-safe 필수**: `plan(&self, ctx: &dyn StageCtx)` 가 trait object 로 ctx 를 받으므로, 모든
/// 메서드는 제네릭 파라미터 / `Self` by value / 연관 타입 / `impl Trait` 인자가 없어야 한다. 따라서
/// 원시 K 읽기 같은 dtype-분기 접근도 슬라이스 반환이 아니라 `out: &mut [f32]` out-param 으로 노출한다.
///
/// 호출 간 누적 상태(d2o EMA 등)는 여기로 thread 하지 않는다 — plugin struct 가 보유한다(D4).
/// accessor 집합은 sliding/streaming/h2o/no_eviction/h2o_plus/d2o 6개 기법의 실제 요구를 합집합한
/// 최소 완전 집합이다(설계 검증 workflow `wf_21533739`).
pub trait StageCtx {
    /// 현재 유효 토큰 수. 전 기법이 keep/prune budget 산출의 출발점으로 읽는다.
    /// 엔진 impl 원천: `KVCache::current_pos()`.
    fn current_pos(&self) -> usize;

    /// 해소된 budget — 보존할 절대 토큰 수. ratio→len 환산은 엔진 책임(`EvictionHandler`)이므로
    /// plugin 은 환산된 값만 읽는다. score-free 또는 head-relative budget 기법(no_eviction/h2o_plus)은
    /// 호출하지 않을 수 있다.
    fn target_len(&self) -> usize;

    /// 이 plan 호출이 처리하는 layer index (d2o per-layer 예산/protect 판정). 엔진이 layer 순회 중
    /// 주입하므로 ctx 는 단일-layer 관점을 유지한다.
    fn layer_idx(&self) -> usize;

    /// flat per-token importance score. `Some` → score-based(h2o heavy-hitter, d2o token rank),
    /// `None` → score-free(sliding/streaming). 위치별 인덱스 접근 전용(`imp.get(pos)`). 반환 슬라이스
    /// 의 borrow 는 ctx 수명에 묶여 dyn-safe.
    fn importance(&self) -> Option<&[f32]>;

    /// KV head 수. h2o_plus per-head 루프 상한 + [`KeepSpec::PerHead`] outer Vec 길이, d2o 의
    /// `layer_dim = n_kv_heads * head_dim` 산출. 엔진 impl 원천: `KVCache::kv_heads()`.
    fn n_kv_heads(&self) -> usize;

    /// head 당 차원. d2o 의 K vector 길이 / cosine 차원 / dequant 버퍼 크기 결정.
    /// 엔진 impl 원천: `KVCache::head_dim()`.
    fn head_dim(&self) -> usize;

    /// per-head importance score (h2o_plus). row-major `[n_kv_heads * max_seq]` 레이아웃의 stride 는
    /// 엔진이 내부화하므로 plugin 은 평탄한 `(kv_head, pos) → f32` 로만 읽는다(레이아웃 결합 해소).
    fn head_score(&self, kv_head: usize, pos: usize) -> f32;

    /// per-head score 가 존재하는가. `false` 면 h2o_plus 가 [`KeepSpec::LayerWide`] 로 degenerate
    /// (score-free/flat 폴백 복원). dyn-safe.
    fn has_head_scores(&self) -> bool;

    /// 원시 K(`pos`, `head`)를 dtype-무관 f32 로 `out` 에 채운다(d2o cosine-nearest 매칭용).
    /// `out.len() == head_dim` 계약. dtype 분기(F32/F16/Q4_0)는 엔진 impl 내부에서 흡수한다.
    /// out-param 방식 = dyn-safe(제네릭 아님) + Q4_0 dequant 임시버퍼 수명/Vec 할당 회피.
    fn dequant_k(&self, pos: usize, head: usize, out: &mut [f32]);
}

/// 가중 병합 지시(ADR-0004 D2/D3). evicted 토큰들(`from`)을 retained 토큰(`into`) 한 자리에 가중
/// 합산한다. `Σ from.1 + into_weight ≈ 1` (magnitude 보존, d2o Eq.11 가중치).
///
/// `into`/`from` 위치는 compact 적용 직전(pre-compact)의 논리 좌표다. 가중치는 plan 에 baked 되어
/// 엔진 executor(`apply_merges`)가 그대로 사용한다(현 uniform 병합 대체). merge-free 정책은 빈 Vec.
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMerge {
    /// 병합 대상 retained 토큰의 위치 (가중 합이 누적될 자리).
    pub into: usize,
    /// `into` 자신의 가중치 (d2o Eq.11 의 `w_c`).
    pub into_weight: f32,
    /// 병합될 evicted 토큰들의 `(위치, 가중치)`.
    pub from: Vec<(usize, f32)>,
}

/// 보존 토큰의 모양 — 배타 enum(ADR-0004 D2).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeepSpec {
    /// sliding/h2o/streaming/no_eviction/d2o. prefix 포함 **ascending**.
    LayerWide(Vec<usize>),
    /// h2o_plus. `[n_kv_heads][keep]`, 각 ascending·길이 동일(엔진이 assert).
    PerHead(Vec<Vec<usize>>),
}

/// 기법이 산출하는 계획. `keep`(배타) ⊥ `merges`(직교). `new_pos` 는 싣지 않는다 — 엔진이
/// `keep.len()` 으로 도출한다([`KeepSpec::PerHead`] 는 전 head 길이 동일 가정).
#[derive(Clone, Debug, PartialEq)]
pub struct KVCachePlan {
    /// 보존 토큰 모양.
    pub keep: KeepSpec,
    /// 가중 병합 지시 (없으면 빈 Vec).
    pub merges: Vec<WeightedMerge>,
}

/// stage 축 확장 기법 표면(ADR-0004) — 상주 토큰을 조절한다 (엔진측 저장-표현 trait `KVCacheFormat`
/// 의 형제).
///
/// plan-returning, self-mutating 아님(D1): 버퍼를 직접 만지지 않고 [`KVCachePlan`] 을 반환하면 엔진이
/// 이를 `KVCacheFormat::compact` 로 실행한다. 그래서 본 trait 은 `&mut KVCache` 같은 엔진 타입을 받지
/// 않고 [`StageCtx`] 읽기 추상만 받는다 — C-ABI 미래(`cdylib` 승격)와 정합하며, 기법 crate 가 엔진
/// 내부에 결합하지 않게 한다.
pub trait KVCacheStage: Send + Sync {
    /// 기법 이름 (CLI `--eviction-policy <name>` 와 매칭, 로깅용). 슬라이스 내 유일해야 한다.
    fn name(&self) -> &str;

    /// 보존/병합 계획 산출. `None` = 미적용(no-op). ctx 읽기 + impl 상태(Mutex)로 계산한다.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan>;
}

/// 기법 인스턴스 생성에 필요한 공통 파라미터. 엔진이 CLI args 를 본 struct 로 매핑해 넘긴다
/// (technique-api 가 엔진 args 타입에 의존하지 않도록 평탄한 값만 싣는다).
///
/// NOTE(ADR-0004 open question): d2o 전용 파라미터(ema_beta/merge_e/use_layer_allocation 등)는
/// "공용 struct 비대화 vs per-technique opaque params" 결정이 미해결이라 **여기에 싣지 않는다** —
/// d2o 마이그레이션(M4) 진입 시 확정한다. 현 4개 빌트인(sliding/streaming/h2o/no_eviction)은 아래
/// 5필드로 충분하다.
#[derive(Clone, Copy, Debug)]
pub struct StageParams {
    /// sliding window 크기 (최근 유지 토큰 수).
    pub eviction_window: usize,
    /// 앞에서 보호할 prefix 길이 (BOS/시스템 프롬프트 등).
    pub protected_prefix: usize,
    /// heavy-hitter 유지 비율 (H2O 계열).
    pub keep_ratio: f32,
    /// streaming sink(attention sink) 크기.
    pub sink_size: usize,
    /// streaming window 크기 (0 이면 엔진이 기본값 유도).
    pub streaming_window: usize,
}

/// 한 stage 기법의 등록 항목. technique crate 가
/// `#[distributed_slice(KV_CACHE_STAGES)] static FOO: KVCacheStageReg = ...` 로 제출한다.
pub struct KVCacheStageReg {
    /// CLI `--eviction-policy` 이름. 슬라이스 내 유일해야 한다.
    pub name: &'static str,
    /// 파라미터로부터 기법 인스턴스를 만드는 팩토리.
    pub make: fn(StageParams) -> Box<dyn KVCacheStage>,
}

/// 전역 등록 슬라이스 — 링크된 모든 technique crate 의 등록이 **링크 타임**에 모인다.
///
/// fat-LTO + `--gc-sections` 가 미참조 섹션을 silent drop 할 수 있다(ADR-0003 §4) — 엔진은 release
/// 빌드에서 기대 기법이 모두 등록됐는지 startup self-test 로 단언해 fail-fast 한다.
#[distributed_slice]
pub static KV_CACHE_STAGES: [KVCacheStageReg] = [..];

/// 이름으로 등록된 기법을 찾는다 (엔진 construction 시 사용).
pub fn find_stage(name: &str) -> Option<&'static KVCacheStageReg> {
    KV_CACHE_STAGES.iter().find(|r| r.name == name)
}

/// 등록된 모든 기법 이름 (self-test / 진단용).
pub fn registered_names() -> Vec<&'static str> {
    KV_CACHE_STAGES.iter().map(|r| r.name).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 등록·조회 round-trip 검증용 no-op 기법.
    struct Dummy;
    impl KVCacheStage for Dummy {
        fn name(&self) -> &str {
            "dummy"
        }
        fn plan(&self, _ctx: &dyn StageCtx) -> Option<KVCachePlan> {
            None
        }
    }

    /// `plan` 호출 경로를 닫기 위한 최소 ctx 스텁 (모든 accessor trivial).
    struct DummyCtx;
    impl StageCtx for DummyCtx {
        fn current_pos(&self) -> usize {
            0
        }
        fn target_len(&self) -> usize {
            0
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
        fn head_score(&self, _kv_head: usize, _pos: usize) -> f32 {
            0.0
        }
        fn has_head_scores(&self) -> bool {
            false
        }
        fn dequant_k(&self, _pos: usize, _head: usize, _out: &mut [f32]) {}
    }

    #[distributed_slice(KV_CACHE_STAGES)]
    static DUMMY_REG: KVCacheStageReg = KVCacheStageReg {
        name: "dummy",
        make: |_params| Box::new(Dummy),
    };

    #[test]
    fn dummy_registers_into_slice() {
        // linkme 가 같은 crate 의 등록을 슬라이스로 모으는지 확인.
        let reg = find_stage("dummy").expect("dummy 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "dummy");
        let params = StageParams {
            eviction_window: 8,
            protected_prefix: 4,
            keep_ratio: 0.5,
            sink_size: 4,
            streaming_window: 0,
        };
        let stage = (reg.make)(params);
        assert_eq!(stage.name(), "dummy");
        assert!(stage.plan(&DummyCtx).is_none());
    }

    #[test]
    fn registered_names_contains_dummy() {
        assert!(registered_names().contains(&"dummy"));
    }
}
