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

/// 엔진이 노출하는 named 캐시 텐서(ADR-0004 M-A 통합). 변형(보존/병합)은 plan 으로만, 읽기는 본 enum
/// 으로 통일한다. **OCP**: 미래 입력(Query/PageBounds 등)은 variant 추가 1줄 + 엔진 impl 1곳 — `StageCtx`
/// 메서드 신설 불필요. read dispatch 비용은 additive accessor 와 동등(PoC: host/ARM ±0~1%).
/// `#[repr(u32)]`: 미래 `.so` C-ABI 에서 fieldless enum 을 u32 discriminant 로 그대로 건넨다(ADR §7).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum TensorKind {
    /// raw K. row=(pos,head), cols=head_dim. dtype 분기(F32/F16/Q4_0)는 핸들 내부 흡수.
    Key,
    /// raw V. CAOTE/VATP 의 v_i. Key 와 동일 (pos,head) 좌표계.
    Value,
    /// 직전 decode step 의 per-(kv_head,pos) attention weight(last layer). CAOTE 의 a_i. cols=1, per_head.
    /// 원천: `AttentionScoreAccumulator::last_step_head_attn`(CPU overwrite / GPU=head_importance proxy).
    /// **주의**: last layer·last step 근사 — windowed/per-layer 정확값이 아니다(`has_attn_weights` 게이트).
    AttnWeights,
    /// per-(kv_head,pos) 누적 head importance(h2o_plus). cols=1, per_head.
    /// flat per-token importance 는 본 핸들이 아니라 [`StageCtx::importance`] 로 zero-copy 직접 노출(D1 예외).
    Scores,
}

/// dtype-무관 텐서 형태(POD). 미래 FFI 경계를 그대로 건널 수 있는 평탄 필드만(`#[repr(C)]`-able).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct TensorShape {
    /// 유효 row 수(보통 `current_pos`).
    pub rows: usize,
    /// row 당 f32 원소 수(Key/Value=head_dim, AttnWeights/Scores=1).
    pub cols: usize,
    /// row 가 per-kv-head 로 분리되는가(현 4 kind 전부 true; layer-wide flat 은 `importance()` 별도 경로).
    pub per_head: bool,
}

/// 핸들의 저장 dtype(진단/버퍼 사이징용). 읽기 산출은 항상 f32.
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum TensorDtype {
    F32,
    F16,
    Q4_0,
}

/// 한 named 텐서의 **읽기 전용** 핸들(ADR-0004 D5, M-A 통합). dyn-safe: 제네릭 / `Self` by value /
/// 연관타입 / `impl Trait` 인자 없음. 산출은 항상 dtype-무관 f32 out-param(슬라이스 반환 금지 —
/// `dequant_k` out-param 규약 계승). 미래 `.so` 단계에선 (opaque ptr + read 함수포인터 + POD shape)로 환원.
pub trait TensorHandle {
    /// POD 형태. 미래 FFI 단계에선 동일 struct 가 그대로 건너간다.
    fn shape(&self) -> TensorShape;
    /// 저장 dtype(읽기 산출 f32 와 무관, 진단용).
    fn dtype(&self) -> TensorDtype;
    /// `(row, kv_head)` 행을 f32 로 `out` 에 채운다. `out.len() == shape().cols` 계약.
    /// `per_head=false` 텐서는 `kv_head` 무시. dtype 분기는 impl 내부 흡수.
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]);
}

/// 기법이 캐시를 읽는 추상(ADR-0004 D5). 엔진이 `&KVCache`(+ scores/budget) 위로 구현한다.
///
/// **dyn-safe 필수**: `plan(&self, ctx: &dyn StageCtx)` 가 trait object 로 ctx 를 받으므로, 모든
/// 메서드는 제네릭 파라미터 / `Self` by value / 연관 타입 / `impl Trait` 인자가 없어야 한다. 따라서
/// 원시 K 읽기 같은 dtype-분기 접근도 슬라이스 반환이 아니라 `out: &mut [f32]` out-param 으로 노출한다.
///
/// 호출 간 누적 상태(d2o EMA 등)는 여기로 thread 하지 않는다 — plugin struct 가 보유한다(D4).
///
/// **읽기 통합(ADR-0004 M-A, D1)**: 모든 텐서/스코어 읽기는 단일 [`StageCtx::tensor`] 메커니즘으로
/// 흐른다. `dequant_k`/`dequant_v`/`head_score`/`has_head_scores`/`attn_weight`/`has_attn_weights` 는
/// `tensor()` 위 default sugar 다 — 엔진은 `tensor()` 1개만 구현하면 된다. flat `importance()` 만 zero-copy
/// 직접 노출(scalar 를 per-element read_row 로 돌리면 H2O 랭킹 경로가 순손해라 예외).
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

    /// ★ **단일 텐서 접근 메커니즘**(D1 통합). 해당 `kind` 가 이 호출에 가용하면 핸들, 아니면 `None`
    /// (score-free 정책은 `tensor(Scores)==None`, value-unaware/attn-unaware 엔진은 해당 kind `None`).
    /// 반환 핸들 borrow 는 ctx 수명에 묶여 dyn-safe. 아래 sugar 들이 전부 이 위에 얹힌다.
    fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle>;

    // ── 이하 default sugar (전부 `tensor()` 위임). 엔진은 override 불필요 ──

    /// 원시 K(`pos`,`head`)를 f32 로 `out` 에 채운다(d2o cosine-nearest). `tensor(Key)` 위 sugar.
    /// `out.len() == head_dim` 계약. kind 미가용 시 no-op(out 불변).
    fn dequant_k(&self, pos: usize, head: usize, out: &mut [f32]) {
        if let Some(h) = self.tensor(TensorKind::Key) {
            h.read_row(pos, head, out);
        }
    }

    /// 원시 V(`pos`,`head`)를 f32 로 `out` 에 채운다(CAOTE/VATP 의 v_i). `tensor(Value)` 위 sugar.
    /// `out.len() == head_dim` 계약. kind 미가용 시 no-op.
    fn dequant_v(&self, pos: usize, head: usize, out: &mut [f32]) {
        if let Some(h) = self.tensor(TensorKind::Value) {
            h.read_row(pos, head, out);
        }
    }

    /// per-head 누적 importance(h2o_plus). 평탄한 `(kv_head, pos) → f32`. `tensor(Scores)` 위 sugar.
    fn head_score(&self, kv_head: usize, pos: usize) -> f32 {
        match self.tensor(TensorKind::Scores) {
            Some(h) => {
                let mut o = [0.0f32];
                h.read_row(pos, kv_head, &mut o);
                o[0]
            }
            None => 0.0,
        }
    }

    /// per-head score 존재 여부. `false` 면 h2o_plus 가 [`KeepSpec::LayerWide`] 로 degenerate.
    fn has_head_scores(&self) -> bool {
        self.tensor(TensorKind::Scores).is_some()
    }

    /// `(kv_head, pos)` 의 직전 decode step per-head attention weight(CAOTE 의 a_i). `tensor(AttnWeights)`
    /// 위 sugar. `has_attn_weights()==false` 면 의미 없음(0.0) — CAOTE 는 `importance()` 폴백 권장.
    fn attn_weight(&self, kv_head: usize, pos: usize) -> f32 {
        match self.tensor(TensorKind::AttnWeights) {
            Some(h) => {
                let mut o = [0.0f32];
                h.read_row(pos, kv_head, &mut o);
                o[0]
            }
            None => 0.0,
        }
    }

    /// attn_weight 가 채워졌는가(직전 step last-layer per-head attn 트래킹 여부). `false` 면 폴백.
    fn has_attn_weights(&self) -> bool {
        self.tensor(TensorKind::AttnWeights).is_some()
    }
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

// ── weight 축 dispatch 타입 (ADR-0006 MW-A) ──
//
// KV 의 plan-returning 과 동형으로 weight stage plugin 의 dispatch 결정을 표현하는 표면 타입.
// (`WeightStage`/`WeightDispatchPlan`/`WeightStageCtx` 본체는 MW-B 신설; 본 단계는 dispatch 모드 타입만.)

/// 연산 위치 축(hardware) plugin 표면 mirror. 엔진 `hardware::DeviceTarget` 와 1:1 (engine 측
/// `From` 양방향 + drift 게이트). api crate 가 engine 에 의존하지 않도록 별도 정의한다.
/// `#[repr(u32)]` 은 미래 `.so` C-ABI 가 discriminant 직접 전달하도록.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceTarget {
    Cpu,
    Gpu,
    Npu,
}

/// weight layer 의 dispatch 모드. plugin 이 결정하는 **stage/hardware 축 모드**다.
/// precision(format 축)은 본 enum 이 아니라 `LayerDirective.precision`(MW-B)으로 분리한다(R1 직교).
#[derive(Debug, Clone)]
pub enum LayerDispatch {
    /// 1-slice dense fast-path (slice 기계 우회).
    Full,
    /// 0-slice (layer skip; 실행 배선은 Phase β).
    Skip,
    /// N-slice composite, share 합 ≈ 1.0.
    Partition(Vec<PartitionShare>),
}

/// partition 슬라이스 1개의 **plugin-결정 좌표** = (share, hardware).
///
/// per-slice 저장 format(precision)은 **plugin 결정이 아니라 executor 가 weight dtype 에서 파생**하므로
/// 본 표면에서 제외한다(ADR-0006 D2/D7; ADR-0004 §9 importance 비통합과 동형의 "결정 vs 파생" 분리).
/// 근거: split 의 byte layout 은 weight tensor 의 실제 dtype 에서 나오고(엔진 `bytes_per_row`), 기존
/// `SliceSpec.format` 은 그것과의 동등 assert 뿐이었다. 표면을 좁은 `TensorDtype`(3종)로 좁히면 현
/// 7-dtype partition 중 Q4_1/Q8_0/BF16/U8 이 회귀하므로, format 은 executor-내부(전체 `DType`)로 둔다.
#[derive(Debug, Clone)]
pub struct PartitionShare {
    /// 이 슬라이스가 weight 의 몇 비율 (out_dim 축).
    pub share: f32,
    /// 이 슬라이스를 resolve 할 hardware 위치.
    pub hardware: DeviceTarget,
}

// ── weight stage plugin (ADR-0006 MW-B, KVCacheStage 동형) ──

/// weight stage 가 읽는 per-layer 메트릭 종류. `WeightStageCtx::layer_metric` 의 kind 인자
/// (KV `TensorKind` 거울). `#[repr(u32)]` 은 미래 `.so` C-ABI discriminant 직접 전달용.
#[repr(u32)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerMetricKind {
    /// per-layer importance (swap 랭킹 key 의 한 축). 엔진 impl 은 ImportanceTable 의
    /// `SubLayer::Full` 투영을 평탄화해 제공한다(entries() 의 sublayer 차원 환원).
    Importance,
    /// per-layer quantization noise ε (swap 랭킹 key 의 ε 축).
    QuantNoise,
}

/// weight stage plugin 이 읽는 읽기 전용 컨텍스트 (KV `StageCtx` 거울, dyn-safe).
/// 엔진이 `&TransformerModel` 위로 구현(MW-D). 변형 권한 없음 — plugin 은 읽고 plan 만 낸다(D1/D3).
pub trait WeightStageCtx {
    /// 디코더 레이어 총 수.
    fn n_layers(&self) -> usize;
    /// 엔진이 해소한 swap budget = **절대 레이어 수**(ratio→count + currently_swapped 차감 +
    /// boundary 보호까지 엔진이 처리; KV `target_len` 거울).
    fn budget(&self) -> usize;
    /// graded 메모리 압력 0–100 (pressure-driven stage 용).
    fn pressure(&self) -> u8;
    /// 해당 레이어의 현재 저장 dtype.
    fn current_format(&self, layer: usize) -> TensorDtype;
    /// ★ per-layer 메트릭 단일 접근 (KV `tensor(kind)` 거울, OCP). kind 미가용 시 `None`.
    /// 반환 슬라이스 borrow 는 ctx 수명에 묶인다(dyn-safe). 길이 = `n_layers()`.
    fn layer_metric(&self, kind: LayerMetricKind) -> Option<&[f32]>;

    // ── default sugar (전부 `layer_metric` 위임). 엔진은 override 불필요 ──

    /// per-layer importance. `layer_metric(Importance)` 위 sugar.
    fn importance(&self) -> Option<&[f32]> {
        self.layer_metric(LayerMetricKind::Importance)
    }
    /// per-layer quantization noise. `layer_metric(QuantNoise)` 위 sugar.
    fn quant_noise(&self) -> Option<&[f32]> {
        self.layer_metric(LayerMetricKind::QuantNoise)
    }
}

/// 한 레이어에 대한 dispatch 지시 (D2). dispatch(stage/hardware 축) ⊥ precision(format 축, R1).
#[derive(Debug, Clone)]
pub struct LayerDirective {
    /// 대상 디코더 레이어 인덱스.
    pub layer: usize,
    /// dispatch 모드(Full / Skip / Partition).
    pub dispatch: LayerDispatch,
    /// precision swap 목표 dtype. `None` = 현 dtype 유지. dispatch 와 직교(R1).
    pub precision: Option<TensorDtype>,
}

/// weight stage 의 plan 산출물 (KV `KVCachePlan` 거울). 결정만 담는 Rust-native 데이터
/// (step/boundary-tier, repr(C) 불요). 변형은 엔진 executor 가 수행(D3).
#[derive(Debug, Clone, Default)]
pub struct WeightDispatchPlan {
    /// 레이어별 지시. 비어 있으면 no-op.
    pub per_layer: Vec<LayerDirective>,
}

/// weight 축 plan-returning 기법 trait (KV `KVCacheStage` 거울).
pub trait WeightStage: Send + Sync {
    /// 기법 이름 (canonical stage name; 슬라이스 내 유일).
    fn name(&self) -> &str;
    /// ctx 를 읽어 dispatch plan 을 낸다. `None` = no-op(미적용).
    fn plan(&self, ctx: &dyn WeightStageCtx) -> Option<WeightDispatchPlan>;
}

/// weight stage 의 CLI-파생 정적 구성 (KV `StageParams` 거울).
///
/// 현재는 swap builtin 의 정적 knob 만 담는다. 런타임 값(swap ratio)은 params 가 아니라
/// `WeightStageCtx::budget`(command-driven)에서 온다. per-builtin 추가 필드는 MW-C 배선 시
/// 확장(KV `StageParams` 의 4-field-vs-opaque open question 과 동형).
#[derive(Debug, Clone, Copy)]
pub struct WeightStageParams {
    /// 경계 레이어(0, 마지막)도 swap 대상에 포함 (연구/ablation; production 기본 false).
    pub allow_boundary_layers: bool,
}

/// 한 weight stage 기법의 등록 항목 (KV `KVCacheStageReg` 거울).
pub struct WeightStageReg {
    /// canonical stage name (resilience `EngineCommand` → name 정규화 표와 일치, Seam C).
    pub name: &'static str,
    /// 파라미터로부터 기법 인스턴스를 만드는 팩토리.
    pub make: fn(WeightStageParams) -> Box<dyn WeightStage>,
}

/// 전역 weight stage 등록 슬라이스 — stage 축의 **4번째 평행 registry**(ADR-0005 D6 확장, 병합 아님).
/// linkme 가 링크 타임에 모은다; fat-LTO `--gc-sections` 대비 startup self-test 는 엔진 측(MW-C).
#[distributed_slice]
pub static WEIGHT_STAGES: [WeightStageReg] = [..];

/// 이름으로 등록된 weight stage 를 찾는다 (엔진 construction 시 사용).
pub fn find_weight_stage(name: &str) -> Option<&'static WeightStageReg> {
    WEIGHT_STAGES.iter().find(|r| r.name == name)
}

/// 등록된 모든 weight stage 이름 (self-test / 진단용).
pub fn registered_weight_names() -> Vec<&'static str> {
    WEIGHT_STAGES.iter().map(|r| r.name).collect()
}

// ── Format 축 plugin registry (ADR-0005 D6, KVCacheStage 동형) ──

/// 양자 블록의 scale 저장 방식 (block-quant family 어휘, ADR-0005 D5).
///
/// `#[repr(u32)]`: 미래 `.so` C-ABI 가 fieldless discriminant 를 그대로 건넨다(L1).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ScaleLayout {
    /// scale 없음 (f32/f16 raw).
    None,
    /// 블록당 단일 f16 scale (q4_0/q8_0).
    PerBlockF16,
    /// 블록당 f16 scale + f16 min (q4_1).
    PerBlockF16WithMin,
}

/// 양자 블록의 비트 패킹 방식 (block-quant family 어휘, ADR-0005 D5).
#[repr(u32)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Packing {
    /// 연속 raw (f32/f16).
    Dense,
    /// nibble(4-bit) 패킹 (q4_0/q4_1).
    Nibble,
    /// byte(8-bit) 패킹 (q8_0).
    Byte,
}

/// layer-tier 경계 POD — format plugin 의 **실제 기여**(ADR-0005 D3/L1).
///
/// block-quant family 어휘만 담는다(`block_elems`/`bits`/`scale_layout`/`packing`):
/// q4_0/q4_1/q8_0/q5 등은 이 descriptor 로 generic floor(dequant→f32 matmul, M-F3)가
/// 구동된다. mxfp4 shared-exponent·codebook·sparse 는 floor 밖 → backend 특화 opt-in escape(D5).
///
/// `#[repr(C)]`: 미래 `.so` C-ABI 경계를 그대로 건너는 평탄 POD(L1 게이트 — 지금 repr(C) 로
/// 두어 `.so` 때 reshape 강제 회피).
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct KVLayoutDesc {
    /// 한 양자 블록의 원소 수 (q4_0/q8_0 = 32). raw(f32/f16) 포맷은 1.
    pub block_elems: u32,
    /// 원소당 비트 수 (q4_0 = 4, q8_0 = 8, f16 = 16, f32 = 32).
    pub bits: u8,
    /// scale 저장 방식.
    pub scale_layout: ScaleLayout,
    /// 비트 패킹 방식.
    pub packing: Packing,
}

/// Format 축 plugin trait — 저장 layout 을 기술한다(ADR-0005 D3).
///
/// layer-tier COMPUTE(`write_kv`/`attention_into`)는 본 trait 에 없다 — 그것은 hardware 축의
/// M×N 커널 셀로 backend 가 소유(D4). format plugin 은 순수 descriptor 다(2-method: name+layout).
///
/// NOTE(phasing, S4-2 2026-06-07): step-tier `compact` 은 본 trait 에 **추가하지 않는다** —
/// ADR-0004 D1 으로 superseded. keep/merge 결정은 stage 축(`KVCacheStage::plan → KVCachePlan`)이,
/// 변형은 엔진 executor `execute_kv_plan` 이 독점한다(결정=plugin, 변형=엔진). compact 을 format 축
/// 으로 끌어오면 stage⊥format 직교를 결정 층위에서 흐리고 `Merge`(engine)를 api 로 누출한다. 따라서
/// `KVFormat` 의 layer-tier 기여는 `layout()` descriptor read 뿐(M-F2, L1 repr(C) 경계).
pub trait KVFormat: Send + Sync {
    /// canonical format 이름 (예: "q4_0"/"f16"/"f32"). 슬라이스 내 유일.
    fn name(&self) -> &str;

    /// 이 포맷의 저장 layout descriptor (엔진 generic reader 가 hot-path 에서 읽음, D3).
    fn layout(&self) -> KVLayoutDesc;
}

/// 한 format 기법의 등록 항목 (KV `KVCacheStageReg` 거울, ADR-0005 D6).
pub struct KVFormatReg {
    /// canonical format 이름. 슬라이스 내 유일.
    pub name: &'static str,
    /// format 인스턴스 팩토리.
    pub make: fn() -> Box<dyn KVFormat>,
}

/// 전역 format 등록 슬라이스 — 3축 평행 registry 의 하나(ADR-0005 D6).
///
/// fat-LTO `--gc-sections` silent drop 리스크는 실제 builtin 등록이 생기는 시점(M-F3)에
/// 엔진 startup self-test 로 게이트한다(ADR-0003 §4 동형).
#[distributed_slice]
pub static KV_FORMATS: [KVFormatReg] = [..];

/// 이름으로 등록된 format 을 찾는다 (엔진 construction 시 사용).
pub fn find_kv_format(name: &str) -> Option<&'static KVFormatReg> {
    KV_FORMATS.iter().find(|r| r.name == name)
}

/// 등록된 모든 format 이름 (self-test / 진단용).
pub fn registered_kv_format_names() -> Vec<&'static str> {
    KV_FORMATS.iter().map(|r| r.name).collect()
}

// ── Backend capability 축 plugin registry (ADR-0005 D4/D6) ──

/// Backend capability plugin trait — backend 소유 커널 위의 특화 opt-in 능력(ADR-0005 D4/D5).
///
/// 골격만(D6): GpuFold 등 첫 instance(step5, 본 crate 단계 밖)가 메서드를 확정한다. backend 는
/// generic floor(descriptor 구동 dequant→f32)를 항상 제공하고 hot 경로만 본 capability 로 특화.
pub trait BackendCapability: Send + Sync {
    /// canonical capability 이름 (예: "gpu_fold"). 슬라이스 내 유일.
    fn name(&self) -> &str;
}

/// 한 backend capability 의 등록 항목 (KV `KVCacheStageReg` 거울, ADR-0005 D6).
pub struct BackendCapReg {
    /// canonical capability 이름. 슬라이스 내 유일.
    pub name: &'static str,
    /// capability 인스턴스 팩토리.
    pub make: fn() -> Box<dyn BackendCapability>,
}

/// 전역 backend capability 등록 슬라이스 — 3축 평행 registry 의 하나(ADR-0005 D6).
#[distributed_slice]
pub static BACKEND_CAPABILITIES: [BackendCapReg] = [..];

/// 이름으로 등록된 capability 를 찾는다.
pub fn find_backend_capability(name: &str) -> Option<&'static BackendCapReg> {
    BACKEND_CAPABILITIES.iter().find(|r| r.name == name)
}

/// 등록된 모든 capability 이름 (self-test / 진단용).
pub fn registered_backend_capability_names() -> Vec<&'static str> {
    BACKEND_CAPABILITIES.iter().map(|r| r.name).collect()
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
        // tensor()만 구현 — head_score/has_head_scores/dequant_k/dequant_v/attn_weight/
        // has_attn_weights 는 default sugar(전부 None → trivial)로 충족.
        fn tensor(&self, _kind: TensorKind) -> Option<&dyn TensorHandle> {
            None
        }
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

    // ── weight stage (MW-B) ──

    /// weight 축 등록·조회 round-trip 검증용 no-op 기법.
    struct DummyWeight;
    impl WeightStage for DummyWeight {
        fn name(&self) -> &str {
            "dummy_weight"
        }
        fn plan(&self, _ctx: &dyn WeightStageCtx) -> Option<WeightDispatchPlan> {
            None
        }
    }

    /// `plan` 경로를 닫는 최소 weight ctx 스텁. `layer_metric` 만 구현 — importance/quant_noise 는
    /// default sugar(전부 None → trivial)로 충족.
    struct DummyWeightCtx;
    impl WeightStageCtx for DummyWeightCtx {
        fn n_layers(&self) -> usize {
            0
        }
        fn budget(&self) -> usize {
            0
        }
        fn pressure(&self) -> u8 {
            0
        }
        fn current_format(&self, _layer: usize) -> TensorDtype {
            TensorDtype::F32
        }
        fn layer_metric(&self, _kind: LayerMetricKind) -> Option<&[f32]> {
            None
        }
    }

    #[distributed_slice(WEIGHT_STAGES)]
    static DUMMY_WEIGHT_REG: WeightStageReg = WeightStageReg {
        name: "dummy_weight",
        make: |_params| Box::new(DummyWeight),
    };

    #[test]
    fn dummy_weight_registers_into_slice() {
        let reg =
            find_weight_stage("dummy_weight").expect("dummy_weight 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "dummy_weight");
        let stage = (reg.make)(WeightStageParams {
            allow_boundary_layers: false,
        });
        assert_eq!(stage.name(), "dummy_weight");
        assert!(stage.plan(&DummyWeightCtx).is_none());
    }

    #[test]
    fn registered_weight_names_contains_dummy() {
        assert!(registered_weight_names().contains(&"dummy_weight"));
    }

    /// `WeightStageCtx` default sugar(importance/quant_noise)가 `layer_metric` 위에서 동작.
    #[test]
    fn weight_ctx_sugar_delegates_to_layer_metric() {
        let ctx = DummyWeightCtx;
        assert!(ctx.importance().is_none());
        assert!(ctx.quant_noise().is_none());
    }

    // ── Format 축 registry (ADR-0005 M-F1) ──

    /// format 등록·조회 round-trip 검증용 no-op format (raw f16 descriptor).
    struct DummyFormat;
    impl KVFormat for DummyFormat {
        fn name(&self) -> &str {
            "dummy_format"
        }
        fn layout(&self) -> KVLayoutDesc {
            KVLayoutDesc {
                block_elems: 1,
                bits: 16,
                scale_layout: ScaleLayout::None,
                packing: Packing::Dense,
            }
        }
    }

    #[distributed_slice(KV_FORMATS)]
    static DUMMY_FORMAT_REG: KVFormatReg = KVFormatReg {
        name: "dummy_format",
        make: || Box::new(DummyFormat),
    };

    #[test]
    fn dummy_format_registers_into_slice() {
        let reg =
            find_kv_format("dummy_format").expect("dummy_format 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "dummy_format");
        let f = (reg.make)();
        assert_eq!(f.name(), "dummy_format");
        // M-F2: layer-tier descriptor read (repr(C) KVLayoutDesc).
        let d = f.layout();
        assert_eq!(d.block_elems, 1);
        assert_eq!(d.bits, 16);
        assert_eq!(d.scale_layout, ScaleLayout::None);
        assert_eq!(d.packing, Packing::Dense);
    }

    #[test]
    fn registered_kv_format_names_contains_dummy() {
        assert!(registered_kv_format_names().contains(&"dummy_format"));
    }

    // ── Backend capability 축 registry (ADR-0005 M-F1) ──

    /// capability 등록·조회 round-trip 검증용 no-op capability.
    struct DummyCap;
    impl BackendCapability for DummyCap {
        fn name(&self) -> &str {
            "dummy_cap"
        }
    }

    #[distributed_slice(BACKEND_CAPABILITIES)]
    static DUMMY_CAP_REG: BackendCapReg = BackendCapReg {
        name: "dummy_cap",
        make: || Box::new(DummyCap),
    };

    #[test]
    fn dummy_cap_registers_into_slice() {
        let reg =
            find_backend_capability("dummy_cap").expect("dummy_cap 등록이 슬라이스에 있어야 한다");
        assert_eq!(reg.name, "dummy_cap");
        assert_eq!((reg.make)().name(), "dummy_cap");
    }

    #[test]
    fn registered_backend_capability_names_contains_dummy() {
        assert!(registered_backend_capability_names().contains(&"dummy_cap"));
    }
}
