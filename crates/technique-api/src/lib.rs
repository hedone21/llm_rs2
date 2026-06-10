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

use core::ffi::{c_char, c_void};

/// `register_kv_stage!` 매크로가 plugin 크레이트에서 `distributed_slice` 어트리뷰트를 경로로 참조할 수
/// 있도록 linkme 의 proc-macro 를 재노출한다(plugin 이 linkme 를 직접 dep 하지 않아도 됨, ADR-0009 D2).
/// 본 crate 내부 등록(`#[distributed_slice]`)도 이 import 를 그대로 쓴다. (crate 자체가 아니라 매크로를
/// 직접 재노출해야 proc-macro 어트리뷰트 경로 resolve 가 된다.)
pub use linkme::distributed_slice;

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
#[repr(C)] // GATE-C(ADR-0009 D2): `.so` C-ABI 가 POD 로 그대로 값 전달(make thunk 인자).
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

// ════════════════════════════════════════════════════════════════════════════
// GATE-C — Stage 축 `.so` cdylib dlopen plugin C-ABI (ADR-0009)
// ════════════════════════════════════════════════════════════════════════════
//
// 정적 등록(`KV_CACHE_STAGES` + `find_stage`)은 그대로 두고(D3 가산), `.so` plugin 이
// 동일 `KVCacheStage` 를 C-ABI 로 노출하는 표면을 추가한다. trait object(`&dyn StageCtx`,
// `Box<dyn KVCacheStage>`)는 C-ABI 불안정이라 fn-ptr 테이블([`StageCtxAbi`]) + opaque
// handle 로 평탄화한다(D2). [`AbiStageCtx`] 어댑터가 `StageCtxAbi` 위에 `StageCtx` 를 다시
// 구현해, plugin 저자는 정적/동적 무관하게 동일한 `impl KVCacheStage` 코드를 쓴다.

/// `register_kv_stages_v2` 봉투([`StageExportAbi`])의 ABI 버전. host 가 mismatch 시 로드 거부(ADR-0010 E1).
pub const KV_STAGE_ABI_VERSION: u32 = 2;

/// [`PluginVTableAbi::plan`] 반환 코드: 정상(`out_plan` 채워짐).
pub const KV_PLAN_OK: i32 = 0;
/// [`PluginVTableAbi::plan`] 반환 코드: no-op(`None` — eviction 미적용).
pub const KV_PLAN_NOOP: i32 = 1;
// 음수 = plugin 의 깨끗한 논리 오류(host 는 로그 후 no-op 처리; panic 아님 — ADR-0009 D1).

/// [`StageCtx`] 의 C-ABI 평탄화(D2). host 가 concrete ctx(`ctx`) 위로 fn-ptr 들을 채워 plugin 에
/// 넘기고, plugin 은 fn-ptr 만 호출한다. 모든 fn-ptr 는 raw 포인터를 deref 하므로 `unsafe`.
///
/// stack-local(per-plan-call)로 만들어 ptr 로 전달 — `static` 아님 → `Sync` 불요.
#[repr(C)]
pub struct StageCtxAbi {
    /// host 의 concrete `&dyn StageCtx` 구현을 가리키는 opaque thin 포인터(아래 fn-ptr 의 첫 인자).
    pub ctx: *const c_void,
    /// [`StageCtx::current_pos`].
    pub current_pos: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::target_len`].
    pub target_len: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::layer_idx`].
    pub layer_idx: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::n_kv_heads`].
    pub n_kv_heads: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::head_dim`].
    pub head_dim: unsafe extern "C" fn(*const c_void) -> usize,
    /// [`StageCtx::importance`]. `Some` → `true` + `out_ptr`/`out_len` 채움(borrow 는 ctx 수명),
    /// `None` → `false`.
    pub importance:
        unsafe extern "C" fn(*const c_void, out_ptr: *mut *const f32, out_len: *mut usize) -> bool,
    /// [`TensorHandle::read_row`] 평탄화. `kind`([`TensorKind`] u32) 가용 + 읽힘 → `true`(`out` 채움,
    /// `out_len == shape().cols` 계약 — host 가 검증), 미가용 → `false`.
    pub tensor_read_row: unsafe extern "C" fn(
        *const c_void,
        kind: u32,
        row: usize,
        kv_head: usize,
        out: *mut f32,
        out_len: usize,
    ) -> bool,
    /// [`TensorHandle::shape`] 평탄화. `kind` 가용 → `true`(`out` 채움), 미가용 → `false`.
    pub tensor_shape: unsafe extern "C" fn(*const c_void, kind: u32, out: *mut TensorShape) -> bool,
}

/// [`KVCachePlan`] 의 C-ABI 평탄화(D5). plugin-arena 가 소유하는 안정 버퍼를 (ptr+len)으로 노출;
/// host 가 즉시 복사 후 [`PluginVTableAbi::plan_free`]`(owner)` 로 plugin 이 자기 arena 를 회수한다
/// ("각자 자기 것 free" — cross-allocator UB 차단).
#[repr(C)]
pub struct PlanAbi {
    /// `0` = [`KeepSpec::LayerWide`], `1` = [`KeepSpec::PerHead`](v1 예약 — host 가 bail).
    pub keep_kind: u32,
    /// 보존 위치 ascending. LayerWide=전체, PerHead=전 head concat.
    pub keep_ptr: *const usize,
    /// `keep_ptr` 길이.
    pub keep_len: usize,
    /// PerHead 전용: head 별 keep 길이(LayerWide=null).
    pub keep_outer_lens: *const usize,
    /// `keep_outer_lens` 길이(= n_kv_heads, LayerWide=0).
    pub keep_outer_count: usize,
    /// 가중 merge 배열(없으면 len 0).
    pub merges_ptr: *const MergeAbi,
    /// `merges_ptr` 길이.
    pub merges_len: usize,
    /// plugin-arena 소유 핸들 → `plan_free(owner)`. host 는 절대 직접 free 금지.
    pub owner: *mut c_void,
}

impl PlanAbi {
    /// out-param 초기값(전부 null/0). host 가 `&mut` 로 plugin 에 넘긴다.
    pub fn zeroed() -> Self {
        PlanAbi {
            keep_kind: 0,
            keep_ptr: core::ptr::null(),
            keep_len: 0,
            keep_outer_lens: core::ptr::null(),
            keep_outer_count: 0,
            merges_ptr: core::ptr::null(),
            merges_len: 0,
            owner: core::ptr::null_mut(),
        }
    }
}

/// [`WeightedMerge`] 의 C-ABI 평탄화(D5). `from` 은 별도 [`FromPairAbi`] 배열로.
#[repr(C)]
pub struct MergeAbi {
    /// [`WeightedMerge::into`].
    pub into: usize,
    /// [`WeightedMerge::into_weight`].
    pub into_weight: f32,
    /// `(pos, weight)` 배열(plugin-arena 소유).
    pub from_ptr: *const FromPairAbi,
    /// `from_ptr` 길이.
    pub from_len: usize,
}

/// `(pos, weight)` 쌍의 C-ABI POD(D5).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct FromPairAbi {
    /// evicted 토큰 위치.
    pub pos: usize,
    /// 병합 가중치.
    pub weight: f32,
}

/// 한 stage 의 C-ABI vtable(D2). v2(ADR-0010)에서 plugin 은 이 vtable 들을 [`PLUGIN_KV_STAGE_VTABLES`]
/// 슬라이스에 누적하고 `register_kv_stages_v2()` 가 [`StageExportAbi`] 봉투로 묶어 노출한다(한 `.so` 다수
/// stage 허용). vtable 은 plugin `.so` 의 `static` 이라 프로세스 수명 내내 유효.
#[repr(C)]
pub struct PluginVTableAbi {
    /// null-종단 canonical 이름(CLI `--eviction-policy` 매칭). plugin `.so` 의 `'static` str.
    /// (ABI 게이트는 봉투 [`StageExportAbi::abi_version`] 가 담당 — vtable 당 버전 필드 없음, ADR-0010 E1.)
    pub name: *const c_char,
    /// `StageParams` → opaque plugin 인스턴스 핸들.
    pub make: unsafe extern "C" fn(*const StageParams) -> *mut c_void,
    /// 핸들 + ctx → plan. 반환 [`KV_PLAN_OK`]/[`KV_PLAN_NOOP`]/음수(err). `out_plan` 채움.
    pub plan: unsafe extern "C" fn(*mut c_void, *const StageCtxAbi, *mut PlanAbi) -> i32,
    /// `plan` 이 채운 [`PlanAbi::owner`] 를 회수(host 가 plan 복사 직후 호출).
    pub plan_free: unsafe extern "C" fn(owner: *mut c_void),
    /// plugin 인스턴스 핸들 해제(host 가 stage drop 시 호출).
    pub drop: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: vtable 은 불변이고 `name` 은 plugin `.so` 의 `'static` str 을 가리킨다. fn-ptr 는 본래
// Send+Sync. 따라서 스레드 간 공유 안전 — plugin 의 distributed_slice element static 선언에 필요(ADR-0010 E1).
unsafe impl Sync for PluginVTableAbi {}

/// stage 축 봉투(ADR-0010 E1) — 한 plugin `.so` 의 stage vtable 들을 한 번에 신고. `register_kv_stages_v2()`
/// 가 **by-value** 로 반환(sret >16B; `count`/`vtables` 는 슬라이스에서 런타임 도출이라 const static 불가).
/// `vtables` 는 [`PLUGIN_KV_STAGE_VTABLES`] base(`.so` static) → `.so` 수명 동안 유효, `count==0` 가능(빈 축).
#[repr(C)]
pub struct StageExportAbi {
    /// [`KV_STAGE_ABI_VERSION`]. host 가 mismatch 시 `.so` 거부(.so 당 ABI 1개).
    pub abi_version: u32,
    /// `vtables` 가 가리키는 연속 배열 길이.
    pub count: usize,
    /// `count` 개의 [`PluginVTableAbi`] 연속 배열(`.so` static). 로더는 `vtables.add(i)` 로 원소 접근.
    pub vtables: *const PluginVTableAbi,
}

/// plugin `.so` 내부에서 stage vtable 들을 누적하는 슬라이스(ADR-0010 E2). **선언은 technique-api 1곳**
/// (linkme section 명이 선언 static 이름으로 결정 — plugin 측 선언은 cross-crate 기여를 깸). `register_kv_stage!`
/// 가 plugin-cdylib 게이트 하에 const-block 격리 static 으로 기여(다회 호출 = 다수 stage). 정적 빌드에선 빈 채
/// 무해(엔진은 `KV_CACHE_STAGES` 만 읽음).
#[distributed_slice]
pub static PLUGIN_KV_STAGE_VTABLES: [PluginVTableAbi] = [..];

/// [`StageCtxAbi`] 의 한 [`TensorKind`] 를 [`TensorHandle`] 로 노출하는 어댑터([`AbiStageCtx`] 내부).
/// `abi` borrow 는 어댑터 수명에 묶인다.
pub struct AbiTensorHandle {
    abi: *const StageCtxAbi,
    kind: u32,
    shape: TensorShape,
}

impl TensorHandle for AbiTensorHandle {
    fn shape(&self) -> TensorShape {
        self.shape
    }
    fn dtype(&self) -> TensorDtype {
        // ABI 읽기 산출은 항상 f32(read_row 가 f32 out). 저장 dtype 은 v1 C-ABI 가 운반하지 않는다
        // (진단용 필드 — 필요 시 abi_version 2 에서 tensor_dtype fn-ptr 추가).
        TensorDtype::F32
    }
    fn read_row(&self, row: usize, kv_head: usize, out: &mut [f32]) {
        // SAFETY: `abi` 는 AbiStageCtx 수명 동안 유효(생성 시 계약). out_len 계약(== cols)은 호출자 책임.
        unsafe {
            let a = &*self.abi;
            (a.tensor_read_row)(a.ctx, self.kind, row, kv_head, out.as_mut_ptr(), out.len());
        }
    }
}

/// [`StageCtxAbi`](C fn-ptr 테이블) 위에 [`StageCtx`] 를 다시 구현하는 어댑터(D2 "write-once,
/// link-either-way"). plugin 의 plan thunk 가 host 의 `StageCtxAbi` 를 이걸로 감싸 동일 `impl
/// KVCacheStage::plan(&dyn StageCtx)` 를 호출한다.
pub struct AbiStageCtx {
    abi: *const StageCtxAbi,
    // TensorKind(repr u32: Key=0/Value=1/AttnWeights=2/Scores=3) 인덱스. 생성 시 shape 프로빙.
    handles: [Option<AbiTensorHandle>; 4],
}

impl AbiStageCtx {
    /// # Safety
    /// `abi` 는 유효한 [`StageCtxAbi`] 를 가리켜야 하고, 그 `ctx` + 모든 fn-ptr 은 이 어댑터 수명
    /// 동안 살아 있어야 한다(host 가 plan 호출 동안 보장).
    pub unsafe fn new(abi: *const StageCtxAbi) -> Self {
        let a = unsafe { &*abi };
        let kinds = [
            TensorKind::Key,
            TensorKind::Value,
            TensorKind::AttnWeights,
            TensorKind::Scores,
        ];
        let mut handles: [Option<AbiTensorHandle>; 4] = [None, None, None, None];
        for kind in kinds {
            let mut shape = TensorShape {
                rows: 0,
                cols: 0,
                per_head: false,
            };
            // SAFETY: new 의 계약상 fn-ptr·ctx 유효.
            let ok = unsafe { (a.tensor_shape)(a.ctx, kind as u32, &mut shape) };
            if ok {
                handles[kind as u32 as usize] = Some(AbiTensorHandle {
                    abi,
                    kind: kind as u32,
                    shape,
                });
            }
        }
        AbiStageCtx { abi, handles }
    }
}

impl StageCtx for AbiStageCtx {
    fn current_pos(&self) -> usize {
        // SAFETY: new 계약.
        unsafe {
            let a = &*self.abi;
            (a.current_pos)(a.ctx)
        }
    }
    fn target_len(&self) -> usize {
        unsafe {
            let a = &*self.abi;
            (a.target_len)(a.ctx)
        }
    }
    fn layer_idx(&self) -> usize {
        unsafe {
            let a = &*self.abi;
            (a.layer_idx)(a.ctx)
        }
    }
    fn n_kv_heads(&self) -> usize {
        unsafe {
            let a = &*self.abi;
            (a.n_kv_heads)(a.ctx)
        }
    }
    fn head_dim(&self) -> usize {
        unsafe {
            let a = &*self.abi;
            (a.head_dim)(a.ctx)
        }
    }
    fn importance(&self) -> Option<&[f32]> {
        // SAFETY: new 계약. 반환 슬라이스 borrow 는 &self 에 묶여, host 가 보장하는 ctx 수명 내 유효.
        unsafe {
            let a = &*self.abi;
            let mut ptr: *const f32 = core::ptr::null();
            let mut len: usize = 0;
            if (a.importance)(a.ctx, &mut ptr, &mut len) && !ptr.is_null() {
                Some(core::slice::from_raw_parts(ptr, len))
            } else {
                None
            }
        }
    }
    fn tensor(&self, kind: TensorKind) -> Option<&dyn TensorHandle> {
        self.handles[kind as u32 as usize]
            .as_ref()
            .map(|h| h as &dyn TensorHandle)
    }
}

/// plugin-arena: plan thunk 가 [`KVCachePlan`] 을 평탄화해 안정 버퍼에 보관(D5). [`Self::into_abi`]
/// 가 leak 한 Box 를 host 가 [`Self::free`] 로 회수한다. self-referential([`MergeAbi::from_ptr`] 가
/// `from_storage` 를 가리킴)이나 Vec heap 버퍼는 이동하지 않아 안전.
pub struct PlanArena {
    keep: Vec<usize>,
    keep_outer_lens: Vec<usize>,
    keep_kind: u32,
    // `merges[i].from_ptr` 가 가리키는 backing 버퍼. raw ptr 로만 참조되므로(직접 read 0)
    // 컴파일러가 못 보지만, drop 시 ptr 가 dangling 되지 않도록 arena 가 소유해야 한다.
    #[allow(dead_code)]
    from_storage: Vec<Vec<FromPairAbi>>,
    merges: Vec<MergeAbi>,
}

impl PlanArena {
    fn from_plan(plan: KVCachePlan) -> Self {
        let (keep_kind, keep, keep_outer_lens) = match plan.keep {
            KeepSpec::LayerWide(k) => (0u32, k, Vec::new()),
            KeepSpec::PerHead(heads) => {
                let lens: Vec<usize> = heads.iter().map(|h| h.len()).collect();
                let flat: Vec<usize> = heads.into_iter().flatten().collect();
                (1u32, flat, lens)
            }
        };
        // from_storage 를 먼저 채워 안정 주소를 확보한 뒤 merges 가 그것을 가리킨다.
        let mut from_storage: Vec<Vec<FromPairAbi>> = Vec::with_capacity(plan.merges.len());
        for m in &plan.merges {
            from_storage.push(
                m.from
                    .iter()
                    .map(|&(pos, weight)| FromPairAbi { pos, weight })
                    .collect(),
            );
        }
        let merges: Vec<MergeAbi> = plan
            .merges
            .iter()
            .enumerate()
            .map(|(i, m)| MergeAbi {
                into: m.into,
                into_weight: m.into_weight,
                from_ptr: from_storage[i].as_ptr(),
                from_len: from_storage[i].len(),
            })
            .collect();
        PlanArena {
            keep,
            keep_outer_lens,
            keep_kind,
            from_storage,
            merges,
        }
    }

    /// plan 을 평탄화·leak 하고 그것을 가리키는 [`PlanAbi`] 를 만든다. host 가 [`Self::free`]`(owner)`
    /// 로 해제할 때까지 유효.
    pub fn into_abi(plan: KVCachePlan) -> PlanAbi {
        let arena = Box::new(Self::from_plan(plan));
        let raw = Box::into_raw(arena);
        // SAFETY: 방금 leak 한 유효 포인터.
        let a = unsafe { &*raw };
        PlanAbi {
            keep_kind: a.keep_kind,
            keep_ptr: a.keep.as_ptr(),
            keep_len: a.keep.len(),
            keep_outer_lens: if a.keep_outer_lens.is_empty() {
                core::ptr::null()
            } else {
                a.keep_outer_lens.as_ptr()
            },
            keep_outer_count: a.keep_outer_lens.len(),
            merges_ptr: a.merges.as_ptr(),
            merges_len: a.merges.len(),
            owner: raw as *mut c_void,
        }
    }

    /// # Safety
    /// `owner` 는 [`Self::into_abi`] 가 만든 `PlanArena` 포인터여야 하고, 정확히 1회만 호출해야 한다.
    pub unsafe fn free(owner: *mut c_void) {
        if !owner.is_null() {
            drop(unsafe { Box::from_raw(owner as *mut PlanArena) });
        }
    }
}

/// stage plugin 을 정적(rlib→linkme) · 동적(cdylib→C-ABI) 양쪽에 등록하는 dual-wiring 매크로(D2).
///
/// `$make` 는 기존 [`KVCacheStageReg::make`] 와 동일한 `fn(StageParams) -> Box<dyn KVCacheStage>`
/// (closure 가능). 동적 C-ABI export(`register_kv_stage_v1`)는 `plugin-cdylib` feature 로 게이트해,
/// 정적 force-link 시 `#[no_mangle]` 심볼 충돌을 원천 차단한다(`.so` 빌드만 `--features plugin-cdylib`).
///
/// 한 plugin 크레이트(`.so`)에서 **여러 번 호출 가능**(다수 stage — ADR-0010 E2). 기여 static 은 모두
/// 익명 `const _: () = {}` 스코프에 격리돼 invocation 간 충돌하지 않는다(linkme static element 는 ident
/// rename 안 함 → 스코프 격리가 유일 회피책). `.so` 엔트리(`register_kv_stages_v2`)는 별도 [`export_plugin!`]
/// 가 `.so` 당 1회 emit 한다.
///
/// ```ignore
/// technique_api::register_kv_stage!("example_keep_recent", |_p| Box::new(KeepRecent));
/// technique_api::export_plugin!();   // .so 당 1회
/// ```
#[macro_export]
macro_rules! register_kv_stage {
    ($name:literal, $make:expr) => {
        // ── 정적 경로 (rlib → linkme distributed_slice). const-block 격리 = 다회 호출 허용(E2). ──
        const _: () = {
            #[$crate::distributed_slice($crate::KV_CACHE_STAGES)]
            static __REG: $crate::KVCacheStageReg = $crate::KVCacheStageReg {
                name: $name,
                make: $make,
            };
        };

        // ── 동적 경로 (cdylib → PLUGIN_KV_STAGE_VTABLES 기여). plugin-cdylib 게이트로 정적 빌드 시 미emit. ──
        // entry(register_kv_stages_v2)는 export_plugin! 가 emit; 여기선 vtable 만 슬라이스에 기여(E2).
        #[cfg(feature = "plugin-cdylib")]
        const _: () = {
            // 핸들 = Box<Box<dyn KVCacheStage>>(thin ptr). make/plan/drop 이 이 표현을 공유.
            type __Handle = ::std::boxed::Box<dyn $crate::KVCacheStage>;

            unsafe extern "C" fn __make(p: *const $crate::StageParams) -> *mut ::core::ffi::c_void {
                // SAFETY: host 가 유효한 StageParams 포인터 전달(D2). StageParams 는 Copy POD.
                // ADR-0009 #116: $make(Rust-ABI fn)는 여기 내부 호출 전용 — extern "C" 직접 캐스팅 금지.
                let params = unsafe { *p };
                let make_fn: fn($crate::StageParams) -> __Handle = $make;
                let stage: __Handle = make_fn(params);
                ::std::boxed::Box::into_raw(::std::boxed::Box::new(stage))
                    as *mut ::core::ffi::c_void
            }

            unsafe extern "C" fn __plan(
                h: *mut ::core::ffi::c_void,
                ctx: *const $crate::StageCtxAbi,
                out: *mut $crate::PlanAbi,
            ) -> i32 {
                // SAFETY: h 는 __make 가 만든 Box<Box<dyn>>, ctx 는 host 가 채운 유효 StageCtxAbi(D2).
                let stage: &dyn $crate::KVCacheStage = unsafe { &**(h as *const __Handle) };
                let abi_ctx = unsafe { $crate::AbiStageCtx::new(ctx) };
                match stage.plan(&abi_ctx) {
                    ::core::option::Option::None => $crate::KV_PLAN_NOOP,
                    ::core::option::Option::Some(plan) => {
                        let abi = $crate::PlanArena::into_abi(plan);
                        // SAFETY: out 는 host 가 준 유효 &mut PlanAbi.
                        unsafe {
                            *out = abi;
                        }
                        $crate::KV_PLAN_OK
                    }
                }
            }

            unsafe extern "C" fn __plan_free(owner: *mut ::core::ffi::c_void) {
                // SAFETY: owner 는 into_abi 가 만든 PlanArena, host 가 1회만 호출(D5).
                unsafe { $crate::PlanArena::free(owner) };
            }

            unsafe extern "C" fn __drop(h: *mut ::core::ffi::c_void) {
                // SAFETY: h 는 __make 가 만든 Box<Box<dyn>>, host 가 1회만 호출.
                drop(unsafe { ::std::boxed::Box::from_raw(h as *mut __Handle) });
            }

            // vtable 을 PLUGIN_KV_STAGE_VTABLES 에 기여(const-block 격리 = 다회 호출 시 누적). entry 아님.
            #[$crate::distributed_slice($crate::PLUGIN_KV_STAGE_VTABLES)]
            static __VTABLE: $crate::PluginVTableAbi = $crate::PluginVTableAbi {
                name: ::core::concat!($name, "\0").as_ptr() as *const ::core::ffi::c_char,
                make: __make,
                plan: __plan,
                plan_free: __plan_free,
                drop: __drop,
            };
        };
    };
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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
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

impl KVLayoutDesc {
    /// block-quant 한 블록의 raw 바이트 (scale + packed quants). raw(`Dense`)는 블록 개념이
    /// 없어 `None` (원소-단위 회계는 [`Self::bytes_for_elems`]).
    ///
    /// ADR-0007 D2: byte-회계 단일 진실원천 — engine `dequant_via_descriptor` 의 인라인 공식
    /// (구 `dtype_layout.rs`)과 `OpaqueBuffer` alloc 이 이 메서드를 공유한다.
    pub fn block_bytes(&self) -> Option<usize> {
        let quant_bytes = match self.packing {
            Packing::Dense => return None,
            Packing::Nibble => self.block_elems as usize / 2,
            Packing::Byte => self.block_elems as usize,
        };
        let scale_bytes = match self.scale_layout {
            ScaleLayout::None => 0,
            ScaleLayout::PerBlockF16 => 2,
            ScaleLayout::PerBlockF16WithMin => 4,
        };
        Some(scale_bytes + quant_bytes)
    }

    /// `numel` 원소를 이 layout 으로 저장하는 총 바이트.
    ///
    /// raw(`Dense`) = `numel * (bits/8)` (f32=4, f16/bf16=2), block-quant =
    /// `(numel / block_elems) * block_bytes`. block-quant 에서 `numel` 이 `block_elems`
    /// 배수가 아니면 `None`(부분 블록 불가).
    pub fn bytes_for_elems(&self, numel: usize) -> Option<usize> {
        match self.block_bytes() {
            None => Some(numel * (self.bits as usize / 8)),
            Some(block_bytes) => {
                let be = self.block_elems as usize;
                if be == 0 || !numel.is_multiple_of(be) {
                    return None;
                }
                Some((numel / be) * block_bytes)
            }
        }
    }
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

// ════════════════════════════════════════════════════════════════════════════
// GATE-C v2 — Format 축 `.so` cdylib dlopen plugin C-ABI (ADR-0009 D4)
// ════════════════════════════════════════════════════════════════════════════
//
// Stage 축(위 GATE-C)과 동형이나 **압도적으로 단순**하다: [`KVFormat`] 은 콜백 0(ctx 도 plan 도
// 없는 순수 descriptor — `name`+`layout`)이라 [`StageCtxAbi`]/[`PlanAbi`]/[`PlanArena`] 가 전부
// 불필요. vtable 은 `make`(opaque 핸들) + `layout`([`KVLayoutDesc`] POD by-value) + `drop` 만 운반.
// `KVLayoutDesc`/`ScaleLayout`/`Packing` 은 이미 `#[repr(C)]`/`#[repr(u32)]` 라 fn-ptr 반환으로
// 그대로 값 전달된다(reshape 0). plugin 은 단일 `register_kv_format_v1() -> *const FormatVTableAbi`
// 만 export(D6 landmine: stage/format/backend register 심볼 분리 — 통합 금지).

/// `register_kv_formats_v2` 봉투([`FormatExportAbi`])의 ABI 버전. host 가 mismatch 시 로드 거부(ADR-0010 E1).
pub const KV_FORMAT_ABI_VERSION: u32 = 2;

/// [`KVFormat`] 의 C-ABI 평탄화(D4). plugin 이 export 하는 단일 vtable. `name` 은 registry 매칭용
/// 정적 식별자(`'static`), `layout` 은 핸들 인스턴스의 [`KVLayoutDesc`] 를 POD by-value 로 반환한다.
///
/// Stage 의 `plan`/`plan_free`(arena 마샬링)에 대응하는 것이 없다 — descriptor 는 stack POD 라
/// cross-allocator 경계가 발생하지 않는다("각자 자기 것 free" 는 핸들 lifecycle(`make`/`drop`)에만
/// 적용).
#[repr(C)]
pub struct FormatVTableAbi {
    /// null-종단 canonical 이름(`--kv-format`/registry 매칭). plugin `.so` 의 `'static` str.
    /// (ABI 게이트는 봉투 [`FormatExportAbi::abi_version`] 가 담당 — vtable 당 버전 필드 없음, ADR-0010 E1.)
    pub name: *const c_char,
    /// format 인스턴스 생성 → opaque 핸들. host 가 `make_format` 시 호출.
    pub make: unsafe extern "C" fn() -> *mut c_void,
    /// 핸들 → [`KVLayoutDesc`](POD by-value). 엔진 generic floor 가 읽는 그 descriptor(D3).
    pub layout: unsafe extern "C" fn(*mut c_void) -> KVLayoutDesc,
    /// format 인스턴스 핸들 해제(host 가 format drop 시 호출).
    pub drop: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: vtable 은 불변이고 `name` 은 plugin `.so` 의 `'static` str 을 가리킨다. fn-ptr 는 본래
// Send+Sync. plugin 의 distributed_slice element static 선언에 필요(ADR-0010 E1).
unsafe impl Sync for FormatVTableAbi {}

/// format 축 봉투(ADR-0010 E1) — 한 plugin `.so` 의 format vtable 들을 한 번에 신고. `register_kv_formats_v2()`
/// 가 **by-value** 로 반환(sret >16B; `count`/`vtables` 는 슬라이스에서 런타임 도출이라 const static 불가).
/// `vtables` 는 [`PLUGIN_KV_FORMAT_VTABLES`] base(`.so` static) → `.so` 수명 동안 유효, `count==0` 가능(빈 축).
#[repr(C)]
pub struct FormatExportAbi {
    /// [`KV_FORMAT_ABI_VERSION`]. host 가 mismatch 시 `.so` 거부(.so 당 ABI 1개).
    pub abi_version: u32,
    /// `vtables` 가 가리키는 연속 배열 길이.
    pub count: usize,
    /// `count` 개의 [`FormatVTableAbi`] 연속 배열(`.so` static). 로더는 `vtables.add(i)` 로 원소 접근.
    pub vtables: *const FormatVTableAbi,
}

/// plugin `.so` 내부에서 format vtable 들을 누적하는 슬라이스(ADR-0010 E2). **선언은 technique-api 1곳**
/// (linkme section 명이 선언 static 이름으로 결정 — plugin 측 선언은 cross-crate 기여를 깸). `register_kv_format!`
/// 가 plugin-cdylib 게이트 하에 const-block 격리 static 으로 기여(다회 호출 = 다수 format). 정적 빌드에선 빈
/// 채 무해(엔진은 `KV_FORMATS` 만 읽음).
#[distributed_slice]
pub static PLUGIN_KV_FORMAT_VTABLES: [FormatVTableAbi] = [..];

/// format plugin 을 정적(rlib→linkme) · 동적(cdylib→C-ABI) 양쪽에 등록하는 dual-wiring 매크로(D4).
///
/// `$make` 는 기존 [`KVFormatReg::make`] 와 동일한 `fn() -> Box<dyn KVFormat>`(closure 가능). 동적
/// C-ABI export(`register_kv_format_v1`)는 `plugin-cdylib` feature 로 게이트해, 정적 force-link 시
/// `#[no_mangle]` 심볼 충돌을 원천 차단한다(`.so` 빌드만 `--features plugin-cdylib`). [`register_kv_stage!`]
/// 의 format 축 짝.
///
/// 한 plugin 크레이트(`.so`)에서 **여러 번 호출 가능**(다수 format = quant 패밀리 — ADR-0010 E2). 기여
/// static 은 모두 익명 `const _: () = {}` 스코프에 격리. `.so` 엔트리(`register_kv_formats_v2`)는 별도
/// [`export_plugin!`] 가 `.so` 당 1회 emit. [`register_kv_stage!`] 의 format 축 짝.
///
/// ```ignore
/// technique_api::register_kv_format!("nf4",  || Box::new(Nf4));
/// technique_api::register_kv_format!("awq4", || Box::new(Awq4));   // 한 .so 에 여러 format
/// technique_api::export_plugin!();   // .so 당 1회
/// ```
#[macro_export]
macro_rules! register_kv_format {
    ($name:literal, $make:expr) => {
        // ── 정적 경로 (rlib → linkme distributed_slice). const-block 격리 = 다회 호출 허용(E2). ──
        const _: () = {
            #[$crate::distributed_slice($crate::KV_FORMATS)]
            static __REG: $crate::KVFormatReg = $crate::KVFormatReg {
                name: $name,
                make: $make,
            };
        };

        // ── 동적 경로 (cdylib → PLUGIN_KV_FORMAT_VTABLES 기여). plugin-cdylib 게이트로 정적 빌드 시 미emit. ──
        // entry(register_kv_formats_v2)는 export_plugin! 가 emit; 여기선 vtable 만 슬라이스에 기여(E2).
        #[cfg(feature = "plugin-cdylib")]
        const _: () = {
            // 핸들 = Box<Box<dyn KVFormat>>(thin ptr). make/layout/drop 이 이 표현을 공유.
            type __Handle = ::std::boxed::Box<dyn $crate::KVFormat>;

            unsafe extern "C" fn __make() -> *mut ::core::ffi::c_void {
                // ADR-0009 #116: $make(Rust-ABI fn)는 여기 내부 호출 전용 — extern "C" 직접 캐스팅 금지.
                let make_fn: fn() -> __Handle = $make;
                let fmt: __Handle = make_fn();
                ::std::boxed::Box::into_raw(::std::boxed::Box::new(fmt)) as *mut ::core::ffi::c_void
            }

            unsafe extern "C" fn __layout(h: *mut ::core::ffi::c_void) -> $crate::KVLayoutDesc {
                // SAFETY: h 는 __make 가 만든 Box<Box<dyn>>. layout()은 POD 반환(arena 불요).
                let fmt: &dyn $crate::KVFormat = unsafe { &**(h as *const __Handle) };
                fmt.layout()
            }

            unsafe extern "C" fn __drop(h: *mut ::core::ffi::c_void) {
                // SAFETY: h 는 __make 가 만든 Box<Box<dyn>>, host 가 1회만 호출.
                drop(unsafe { ::std::boxed::Box::from_raw(h as *mut __Handle) });
            }

            // vtable 을 PLUGIN_KV_FORMAT_VTABLES 에 기여(const-block 격리 = 다회 호출 시 누적). entry 아님.
            #[$crate::distributed_slice($crate::PLUGIN_KV_FORMAT_VTABLES)]
            static __VTABLE: $crate::FormatVTableAbi = $crate::FormatVTableAbi {
                name: ::core::concat!($name, "\0").as_ptr() as *const ::core::ffi::c_char,
                make: __make,
                layout: __layout,
                drop: __drop,
            };
        };
    };
}

/// `.so` 당 **1회** 호출 — 이 plugin 의 per-axis 엔트리 심볼을 emit 한다(ADR-0010 E2). plugin-cdylib 게이트:
/// 정적 force-link 빌드에선 미emit(다수 force-link plugin 의 entry 충돌 차단). `register_kv_*!` 가 누적한
/// [`PLUGIN_KV_STAGE_VTABLES`]/[`PLUGIN_KV_FORMAT_VTABLES`] 슬라이스를 by-value 봉투로 반환한다.
///
/// **3축 분리 심볼 불변(ADR-0009 #118)**: `register_kv_stages_v2` ⊥ `register_kv_formats_v2` ⊥
/// `register_backend_caps_v2` 분리 entry + 분리 슬라이스 — 통합 심볼/통합 registry 아님(작성자 편의로
/// 한 번에 emit 할 뿐). backend 축(3번째)은 D8 구현으로 추가됨.
///
/// 기여 0 인 축은 `count==0` 봉투(빈 distributed_slice — ELF `__start==__stop`, 안전).
///
/// ```ignore
/// technique_api::register_kv_format!("nf4", || Box::new(Nf4));
/// technique_api::export_plugin!();
/// ```
#[macro_export]
macro_rules! export_plugin {
    () => {
        #[cfg(feature = "plugin-cdylib")]
        const _: () = {
            /// stage 봉투 entry — `PLUGIN_KV_STAGE_VTABLES` 를 by-value 로 반환(sret).
            #[unsafe(no_mangle)] // Rust 2024: no_mangle 은 unsafe attribute.
            pub extern "C" fn register_kv_stages_v2() -> $crate::StageExportAbi {
                // .len()/.as_ptr() 는 linkme Deref(static_slice) 경유 런타임 평가 — 봉투는 호출 시점 산출.
                $crate::StageExportAbi {
                    abi_version: $crate::KV_STAGE_ABI_VERSION,
                    count: $crate::PLUGIN_KV_STAGE_VTABLES.len(),
                    vtables: $crate::PLUGIN_KV_STAGE_VTABLES.as_ptr(),
                }
            }

            /// format 봉투 entry — `PLUGIN_KV_FORMAT_VTABLES` 를 by-value 로 반환(sret).
            #[unsafe(no_mangle)]
            pub extern "C" fn register_kv_formats_v2() -> $crate::FormatExportAbi {
                $crate::FormatExportAbi {
                    abi_version: $crate::KV_FORMAT_ABI_VERSION,
                    count: $crate::PLUGIN_KV_FORMAT_VTABLES.len(),
                    vtables: $crate::PLUGIN_KV_FORMAT_VTABLES.as_ptr(),
                }
            }

            /// backend-cap 봉투 entry(3축, D8) — `PLUGIN_BACKEND_CAP_VTABLES` 를 by-value 로 반환(sret).
            #[unsafe(no_mangle)]
            pub extern "C" fn register_backend_caps_v2() -> $crate::BackendCapExportAbi {
                $crate::BackendCapExportAbi {
                    abi_version: $crate::BACKEND_CAP_ABI_VERSION,
                    count: $crate::PLUGIN_BACKEND_CAP_VTABLES.len(),
                    vtables: $crate::PLUGIN_BACKEND_CAP_VTABLES.as_ptr(),
                }
            }
        };
    };
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

// ── Backend capability 축 — ATTENTION(KIVI) 카테고리 동적 C-ABI (design D2/D7/D8, ADR-0010 봉투) ──
//
// D8(single-trait): canonical [`KiviAttentionBackend`] 를 technique-api 가 소유한다. 엔진 정적 OpenCL
// impl · host dlopen 어댑터 · plugin `.so` 가 **모두 이 1벌**을 구현(Stage `KVCacheStage` 동형). 시그니처는
// `&Tensor` 가 아니라 ABI struct(`KiviAttnArgs`/`KiviGatherArgs`, cl_mem `*mut c_void`)라 plugin 이 엔진
// 타입을 비참조(독립). 위 정적 `BACKEND_CAPABILITIES`(name 키)는 그대로 — fat-LTO 이름 생존 smoke 용.

/// `register_backend_caps_v2` 봉투([`BackendCapExportAbi`])의 ABI 버전. host mismatch 시 `.so` 거부(ADR-0010 E1).
pub const BACKEND_CAP_ABI_VERSION: u32 = 1;

/// capability 카테고리 태그 — ATTENTION(KIVI fused dequant+attention). [`BackendCapVTableAbi::category`].
/// host 카테고리 다리(`match`)가 이 값으로 `vtable` 포인터를 카테고리별 테이블([`KiviAttnVTable`])로 캐스팅(D7).
pub const BACKEND_CAP_CATEGORY_ATTENTION: u32 = 1;

/// KIVI capability 인스턴스 생성 인자(D4). host 가 빌려준 GPU context/device + build 옵션으로 plugin 이
/// 커널을 **1회** 빌드해 opaque 핸들을 만든다. bare-C 핸들만(C4) — `ocl` 래퍼 타입 금지.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KiviMakeArgs {
    /// `cl_context` raw 핸들(host backend 소유, borrow-for-make).
    pub cl_ctx: *mut c_void,
    /// `cl_device_id` raw 핸들.
    pub device: *mut c_void,
    /// null-종단 OpenCL build 옵션(host `build_cl_opts(device)` 결과 — Adreno 일관성, C7). null 가능.
    pub build_opts: *const c_char,
}

/// KIVI fused dequant+attention 호출 인자(D6). 모든 GPU 자원은 **borrow-for-call**(C5 retain 금지).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KiviAttnArgs {
    /// `cl_command_queue` raw 핸들(host 가 빌려줌).
    pub cl_queue: *mut c_void,
    pub q_mem: *mut c_void,
    pub qk_mem: *mut c_void,
    pub qv_mem: *mut c_void,
    pub res_k_mem: *mut c_void,
    pub res_v_mem: *mut c_void,
    pub out_mem: *mut c_void,
    /// CPU score readback 버퍼(optional; null=score 없음). `scores_len` 개 f32.
    pub scores_out: *mut f32,
    pub scores_len: usize,
    pub num_heads_q: usize,
    pub num_heads_kv: usize,
    pub head_dim: usize,
    pub q_tokens: usize,
    pub res_tokens: usize,
    pub res_cap: usize,
    pub scale: f32,
    pub bits: u8,
}

/// KIVI residual gather-update 호출 인자(D6). 2-mem(input/residual) + 스칼라 5.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct KiviGatherArgs {
    pub cl_queue: *mut c_void,
    pub input_mem: *mut c_void,
    pub residual_mem: *mut c_void,
    pub kv_heads: usize,
    pub res_cap: usize,
    pub head_dim: usize,
    pub seq_len: usize,
    pub res_pos: usize,
}

/// ATTENTION 카테고리 canonical capability trait(D8 single-trait). technique-api 소유 → 엔진 정적 impl ·
/// host dlopen 어댑터 · plugin `.so` 가 **모두 이 1벌**을 구현. `&Tensor` 대신 ABI struct 를 받아 plugin
/// 독립(엔진 타입 비참조). 반환 `i32`(C3 panic=abort: 0=OK · 음수=err — vtable fn-ptr 는 panic 금지).
pub trait KiviAttentionBackend: Send + Sync {
    /// `bits`(2/4/8) fused KIVI attention 커널 보유 여부.
    fn has_kivi_attn_kernel(&self, bits: u8) -> bool;
    /// sub-group 미지원 device(Adreno nosub) 여부 — 커널 변형 선택용.
    fn is_nosub_device(&self) -> bool;
    /// fused dequant+attention. cl_mem 은 [`KiviAttnArgs`] 안, borrow-for-call(C5).
    fn attention_gen_kivi(&self, args: &KiviAttnArgs) -> i32;
    /// residual ring gather-update(K/V quant 직전).
    fn kivi_gather_update(&self, args: &KiviGatherArgs) -> i32;
}

/// 정적(force-link) KIVI ATTENTION capability 등록 항목 — Stage [`KVCacheStageReg`] 의 backend 축 짝(D8).
/// `make` 는 host 가 GPU context 를 가졌을 때만 호출(`KiviMakeArgs`); fat-LTO 생존 smoke 는 이름만 확인.
pub struct KiviAttentionReg {
    /// canonical capability 이름. 슬라이스 내 유일.
    pub name: &'static str,
    /// capability 인스턴스 팩토리(host GPU context 로 커널 1회 빌드, D4).
    pub make: fn(&KiviMakeArgs) -> Box<dyn KiviAttentionBackend>,
}

/// 전역 KIVI ATTENTION capability 정적 등록 슬라이스(linkme). `register_kivi_attention_plugin!` 가 기여.
/// 동적 dlopen 경로([`PLUGIN_BACKEND_CAP_VTABLES`])와 분리 — source-agnostic 조회는 host 가 합친다(D3 거울).
#[distributed_slice]
pub static KIVI_ATTENTION_REGS: [KiviAttentionReg] = [..];

/// 이름으로 정적 등록된 KIVI ATTENTION capability 를 찾는다.
pub fn find_kivi_attention(name: &str) -> Option<&'static KiviAttentionReg> {
    KIVI_ATTENTION_REGS.iter().find(|r| r.name == name)
}

/// 정적 등록된 모든 KIVI ATTENTION capability 이름(fat-LTO 생존 smoke / 진단용).
pub fn registered_kivi_attention_names() -> Vec<&'static str> {
    KIVI_ATTENTION_REGS.iter().map(|r| r.name).collect()
}

/// ATTENTION 카테고리 C-ABI vtable(D7) — [`BackendCapVTableAbi::vtable`] 가 category==ATTENTION 일 때
/// 가리키는 테이블. make/drop 도 여기(make 인자가 카테고리별 [`KiviMakeArgs`] 라 공통 헤더에 못 둠).
#[repr(C)]
pub struct KiviAttnVTable {
    /// [`KiviMakeArgs`] → opaque plugin 핸들(커널 1회 빌드, D4). host `make` 시 호출.
    pub make: unsafe extern "C" fn(*const KiviMakeArgs) -> *mut c_void,
    /// 핸들 + bits → 커널 보유 bool.
    pub has_kivi_attn_kernel: unsafe extern "C" fn(*mut c_void, u8) -> bool,
    /// 핸들 → nosub device bool.
    pub is_nosub_device: unsafe extern "C" fn(*mut c_void) -> bool,
    /// 핸들 + [`KiviAttnArgs`] → i32(0=OK · 음수=err). 매 토큰 hot-path.
    pub attention_gen_kivi: unsafe extern "C" fn(*mut c_void, *const KiviAttnArgs) -> i32,
    /// 핸들 + [`KiviGatherArgs`] → i32. residual gather-update.
    pub kivi_gather_update: unsafe extern "C" fn(*mut c_void, *const KiviGatherArgs) -> i32,
    /// 핸들 해제(host 가 capability drop 시 1회).
    pub drop: unsafe extern "C" fn(*mut c_void),
}

// SAFETY: vtable 불변 + fn-ptr 는 본래 Send+Sync. distributed_slice element static 선언에 필요(ADR-0010 E1).
unsafe impl Sync for KiviAttnVTable {}

/// backend-cap 축 엔트리(D7 태그드 포인터) — 얇은 `{name, category, vtable}`. 실제 함수는 category 별
/// 테이블([`KiviAttnVTable`] 등)에. host 가 `category` 로 `vtable` 를 캐스팅(category 다리).
#[repr(C)]
pub struct BackendCapVTableAbi {
    /// null-종단 canonical 이름(registry 매칭). plugin `.so` 의 `'static` str.
    /// (ABI 게이트는 봉투 [`BackendCapExportAbi::abi_version`] 가 담당 — vtable 당 버전 필드 없음.)
    pub name: *const c_char,
    /// 카테고리 태그([`BACKEND_CAP_CATEGORY_ATTENTION`] 등). host `match` 키.
    pub category: u32,
    /// category 별 `#[repr(C)]` 테이블 포인터(예: `*const KiviAttnVTable`). host 가 `category` 로 캐스팅.
    pub vtable: *const c_void,
}

// SAFETY: 불변 + name/vtable 은 `.so` 의 `'static`. distributed_slice element static 에 필요(ADR-0010 E1).
unsafe impl Sync for BackendCapVTableAbi {}

/// backend-cap 축 봉투(ADR-0010 E1) — 한 `.so` 의 capability vtable 들을 한 번에 신고. `register_backend_caps_v2()`
/// 가 **by-value** 반환(sret). `vtables` 는 [`PLUGIN_BACKEND_CAP_VTABLES`] base → `.so` 수명 유효, `count==0` 가능.
#[repr(C)]
pub struct BackendCapExportAbi {
    /// [`BACKEND_CAP_ABI_VERSION`]. host mismatch 시 `.so` 거부(.so 당 ABI 1개).
    pub abi_version: u32,
    /// `vtables` 연속 배열 길이.
    pub count: usize,
    /// `count` 개의 [`BackendCapVTableAbi`] 연속 배열(`.so` static). 로더는 `vtables.add(i)`.
    pub vtables: *const BackendCapVTableAbi,
}

/// plugin `.so` 내부에서 backend-cap vtable 들을 누적하는 슬라이스(ADR-0010 E2). **선언은 technique-api 1곳**.
/// `register_kivi_attention_plugin!` 가 plugin-cdylib 게이트 하에 기여. 정적 빌드에선 빈 채 무해.
#[distributed_slice]
pub static PLUGIN_BACKEND_CAP_VTABLES: [BackendCapVTableAbi] = [..];

/// KIVI ATTENTION capability plugin 을 정적(rlib→linkme 이름 생존) · 동적(cdylib→C-ABI vtable) 양쪽에
/// 등록하는 dual-wiring 매크로(D8). `$make` = `fn(&KiviMakeArgs) -> Box<dyn KiviAttentionBackend>`(closure 가능).
///
/// **정적 경로**: `$make` 를 [`KIVI_ATTENTION_REGS`] 슬라이스에 기여(force-link 시 이름 생존 — fat-LTO 생존
/// smoke 용, `registered_kivi_attention_names()` 가 이름 확인). **동적 경로**(plugin-cdylib): `$make`/trait 메서드를
/// C thunk 로 래핑해 [`KiviAttnVTable`] + 봉투 엔트리를 [`PLUGIN_BACKEND_CAP_VTABLES`] 에 기여. `.so` 엔트리
/// (`register_backend_caps_v2`)는 [`export_plugin!`] 가 emit. 한 `.so` 에서 다회 호출 가능(다수 capability).
#[macro_export]
macro_rules! register_kivi_attention_plugin {
    ($name:literal, $make:expr) => {
        // ── 정적 경로 (rlib → linkme KIVI_ATTENTION_REGS, force-link 시 이름 생존). 비게이트(양 빌드 공통). ──
        // `$make` 를 live distributed_slice static 에 저장 → 정적 조회 인프라 + feature-OFF 빌드에서도
        // `$make`/연관 타입이 reachable(Stage `register_kv_stage!` 동형, 미사용 경고 없음).
        const _: () = {
            #[$crate::distributed_slice($crate::KIVI_ATTENTION_REGS)]
            static __REG: $crate::KiviAttentionReg = $crate::KiviAttentionReg {
                name: $name,
                make: $make,
            };
        };

        // ── 동적 경로 (cdylib → KiviAttnVTable + 봉투 엔트리). plugin-cdylib 게이트로 정적 빌드 시 미emit. ──
        #[cfg(feature = "plugin-cdylib")]
        const _: () = {
            // 핸들 = Box<Box<dyn KiviAttentionBackend>>(thin ptr). 모든 thunk 가 이 표현 공유.
            type __Handle = ::std::boxed::Box<dyn $crate::KiviAttentionBackend>;

            unsafe extern "C" fn __make(
                p: *const $crate::KiviMakeArgs,
            ) -> *mut ::core::ffi::c_void {
                // SAFETY: host 가 유효한 KiviMakeArgs 포인터 전달(D4). KiviMakeArgs 는 Copy POD.
                let args: &$crate::KiviMakeArgs = unsafe { &*p };
                let make_fn: fn(&$crate::KiviMakeArgs) -> __Handle = $make;
                let be: __Handle = make_fn(args);
                ::std::boxed::Box::into_raw(::std::boxed::Box::new(be)) as *mut ::core::ffi::c_void
            }

            unsafe extern "C" fn __has(h: *mut ::core::ffi::c_void, bits: u8) -> bool {
                // SAFETY: h 는 __make 가 만든 Box<Box<dyn>>.
                let be: &dyn $crate::KiviAttentionBackend = unsafe { &**(h as *const __Handle) };
                be.has_kivi_attn_kernel(bits)
            }

            unsafe extern "C" fn __nosub(h: *mut ::core::ffi::c_void) -> bool {
                // SAFETY: 위와 동일.
                let be: &dyn $crate::KiviAttentionBackend = unsafe { &**(h as *const __Handle) };
                be.is_nosub_device()
            }

            unsafe extern "C" fn __attn(
                h: *mut ::core::ffi::c_void,
                a: *const $crate::KiviAttnArgs,
            ) -> i32 {
                // SAFETY: h 는 __make 가 만든 Box<Box<dyn>>, a 는 host 가 채운 유효 KiviAttnArgs(C5).
                let be: &dyn $crate::KiviAttentionBackend = unsafe { &**(h as *const __Handle) };
                be.attention_gen_kivi(unsafe { &*a })
            }

            unsafe extern "C" fn __gather(
                h: *mut ::core::ffi::c_void,
                a: *const $crate::KiviGatherArgs,
            ) -> i32 {
                // SAFETY: 위와 동일.
                let be: &dyn $crate::KiviAttentionBackend = unsafe { &**(h as *const __Handle) };
                be.kivi_gather_update(unsafe { &*a })
            }

            unsafe extern "C" fn __drop(h: *mut ::core::ffi::c_void) {
                // SAFETY: h 는 __make 가 만든 Box<Box<dyn>>, host 가 1회만 호출.
                drop(unsafe { ::std::boxed::Box::from_raw(h as *mut __Handle) });
            }

            static __VTABLE: $crate::KiviAttnVTable = $crate::KiviAttnVTable {
                make: __make,
                has_kivi_attn_kernel: __has,
                is_nosub_device: __nosub,
                attention_gen_kivi: __attn,
                kivi_gather_update: __gather,
                drop: __drop,
            };

            // 봉투 엔트리를 PLUGIN_BACKEND_CAP_VTABLES 에 기여(const-block 격리 = 다회 호출 누적). entry 아님.
            #[$crate::distributed_slice($crate::PLUGIN_BACKEND_CAP_VTABLES)]
            static __ENTRY: $crate::BackendCapVTableAbi = $crate::BackendCapVTableAbi {
                name: ::core::concat!($name, "\0").as_ptr() as *const ::core::ffi::c_char,
                category: $crate::BACKEND_CAP_CATEGORY_ATTENTION,
                vtable: &__VTABLE as *const $crate::KiviAttnVTable as *const ::core::ffi::c_void,
            };
        };
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    /// ADR-0007 G1: `KVLayoutDesc` byte-회계가 engine block 구조체 크기와 일치.
    /// engine 의존 0(technique-api 격리)이라 literal 로 검증 — engine 측에서
    /// `size_of::<Block*>()` cross-check(dtype_layout.rs).
    #[test]
    fn kv_layout_desc_byte_accounting() {
        let q4_0 = KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Nibble,
        };
        assert_eq!(q4_0.block_bytes(), Some(18)); // == size_of::<BlockQ4_0>()
        assert_eq!(q4_0.bytes_for_elems(32), Some(18));
        assert_eq!(q4_0.bytes_for_elems(64), Some(36));
        assert_eq!(q4_0.bytes_for_elems(31), None); // 부분 블록 불가

        let q4_1 = KVLayoutDesc {
            block_elems: 32,
            bits: 4,
            scale_layout: ScaleLayout::PerBlockF16WithMin,
            packing: Packing::Nibble,
        };
        assert_eq!(q4_1.block_bytes(), Some(20)); // == size_of::<BlockQ4_1>()
        assert_eq!(q4_1.bytes_for_elems(32), Some(20));

        let q8_0 = KVLayoutDesc {
            block_elems: 32,
            bits: 8,
            scale_layout: ScaleLayout::PerBlockF16,
            packing: Packing::Byte,
        };
        assert_eq!(q8_0.block_bytes(), Some(34)); // == size_of::<BlockQ8_0>()
        assert_eq!(q8_0.bytes_for_elems(32), Some(34));

        // raw(Dense): 블록 개념 없음, 원소당 bits/8.
        let f32 = KVLayoutDesc {
            block_elems: 1,
            bits: 32,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        };
        assert_eq!(f32.block_bytes(), None);
        assert_eq!(f32.bytes_for_elems(10), Some(40));
        let f16 = KVLayoutDesc {
            block_elems: 1,
            bits: 16,
            scale_layout: ScaleLayout::None,
            packing: Packing::Dense,
        };
        assert_eq!(f16.bytes_for_elems(10), Some(20));
    }

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

    // ── GATE-C C-ABI round-trip (ADR-0009 C1, `.so` 없이 in-process 검증) ──

    /// host concrete ctx 모의 — StageCtxAbi 의 fn-ptr 들이 이 위로 동작.
    struct HostCtx {
        cur: usize,
        tgt: usize,
        imp: Vec<f32>,
        key: Vec<f32>, // 비면 tensor(Key) Some(1 row × len), 빈면 None
    }

    unsafe extern "C" fn h_current_pos(c: *const c_void) -> usize {
        unsafe { (*(c as *const HostCtx)).cur }
    }
    unsafe extern "C" fn h_target_len(c: *const c_void) -> usize {
        unsafe { (*(c as *const HostCtx)).tgt }
    }
    unsafe extern "C" fn h_layer_idx(_c: *const c_void) -> usize {
        0
    }
    unsafe extern "C" fn h_n_kv_heads(_c: *const c_void) -> usize {
        1
    }
    unsafe extern "C" fn h_head_dim(c: *const c_void) -> usize {
        unsafe { (*(c as *const HostCtx)).key.len().max(1) }
    }
    unsafe extern "C" fn h_importance(
        c: *const c_void,
        out_ptr: *mut *const f32,
        out_len: *mut usize,
    ) -> bool {
        let h = unsafe { &*(c as *const HostCtx) };
        if h.imp.is_empty() {
            return false;
        }
        unsafe {
            *out_ptr = h.imp.as_ptr();
            *out_len = h.imp.len();
        }
        true
    }
    unsafe extern "C" fn h_tensor_shape(
        c: *const c_void,
        kind: u32,
        out: *mut TensorShape,
    ) -> bool {
        let h = unsafe { &*(c as *const HostCtx) };
        if kind == TensorKind::Key as u32 && !h.key.is_empty() {
            unsafe {
                *out = TensorShape {
                    rows: 1,
                    cols: h.key.len(),
                    per_head: true,
                };
            }
            true
        } else {
            false
        }
    }
    unsafe extern "C" fn h_tensor_read_row(
        c: *const c_void,
        kind: u32,
        _row: usize,
        _kv_head: usize,
        out: *mut f32,
        out_len: usize,
    ) -> bool {
        let h = unsafe { &*(c as *const HostCtx) };
        if kind == TensorKind::Key as u32 && out_len == h.key.len() {
            unsafe { core::ptr::copy_nonoverlapping(h.key.as_ptr(), out, out_len) };
            true
        } else {
            false
        }
    }

    fn make_abi(host: &HostCtx) -> StageCtxAbi {
        StageCtxAbi {
            ctx: host as *const HostCtx as *const c_void,
            current_pos: h_current_pos,
            target_len: h_target_len,
            layer_idx: h_layer_idx,
            n_kv_heads: h_n_kv_heads,
            head_dim: h_head_dim,
            importance: h_importance,
            tensor_read_row: h_tensor_read_row,
            tensor_shape: h_tensor_shape,
        }
    }

    #[test]
    fn abi_stage_ctx_reproduces_scalars_and_importance() {
        let host = HostCtx {
            cur: 100,
            tgt: 30,
            imp: vec![1.0, 2.0, 3.0],
            key: vec![],
        };
        let abi = make_abi(&host);
        // SAFETY: abi(및 host)는 ctx 수명 동안 살아 있다.
        let ctx = unsafe { AbiStageCtx::new(&abi) };
        assert_eq!(ctx.current_pos(), 100);
        assert_eq!(ctx.target_len(), 30);
        assert_eq!(ctx.layer_idx(), 0);
        assert_eq!(ctx.n_kv_heads(), 1);
        assert_eq!(ctx.importance(), Some(&[1.0f32, 2.0, 3.0][..]));
        // key 빈 → tensor(Key)/Value/Scores 모두 None(default sugar 도 trivial).
        assert!(ctx.tensor(TensorKind::Key).is_none());
        assert!(!ctx.has_head_scores());
    }

    #[test]
    fn abi_stage_ctx_reproduces_tensor_read() {
        let host = HostCtx {
            cur: 10,
            tgt: 5,
            imp: vec![],
            key: vec![1.5, 2.5, 3.5, 4.5],
        };
        let abi = make_abi(&host);
        let ctx = unsafe { AbiStageCtx::new(&abi) };
        assert!(ctx.importance().is_none());
        let kh = ctx.tensor(TensorKind::Key).expect("Key 가용");
        assert_eq!(kh.shape().cols, 4);
        let mut out = [0.0f32; 4];
        // dequant_k 는 tensor(Key) 위 default sugar — fn-ptr 를 타고 host key 를 채운다.
        ctx.dequant_k(0, 0, &mut out);
        assert_eq!(out, [1.5, 2.5, 3.5, 4.5]);
    }

    #[test]
    fn plan_arena_layerwide_round_trip() {
        let plan = KVCachePlan {
            keep: KeepSpec::LayerWide(vec![70, 71, 72]),
            merges: Vec::new(),
        };
        let abi = PlanArena::into_abi(plan.clone());
        assert_eq!(abi.keep_kind, 0);
        // SAFETY: into_abi 가 leak 한 유효 arena 를 free 전까지 읽는다.
        let keep = unsafe { core::slice::from_raw_parts(abi.keep_ptr, abi.keep_len) };
        assert_eq!(keep, &[70usize, 71, 72]);
        assert_eq!(abi.merges_len, 0);
        assert!(abi.keep_outer_lens.is_null());
        unsafe { PlanArena::free(abi.owner) };
    }

    #[test]
    fn plan_arena_with_merges_round_trip() {
        let plan = KVCachePlan {
            keep: KeepSpec::LayerWide(vec![0, 1]),
            merges: vec![WeightedMerge {
                into: 0,
                into_weight: 0.6,
                from: vec![(5, 0.2), (6, 0.2)],
            }],
        };
        let abi = PlanArena::into_abi(plan.clone());
        // host 측 reconstruct 미러(C2 가 동일 로직 수행).
        let keep = unsafe { core::slice::from_raw_parts(abi.keep_ptr, abi.keep_len) }.to_vec();
        let merges_abi = unsafe { core::slice::from_raw_parts(abi.merges_ptr, abi.merges_len) };
        let merges: Vec<WeightedMerge> = merges_abi
            .iter()
            .map(|m| {
                let from = unsafe { core::slice::from_raw_parts(m.from_ptr, m.from_len) };
                WeightedMerge {
                    into: m.into,
                    into_weight: m.into_weight,
                    from: from.iter().map(|p| (p.pos, p.weight)).collect(),
                }
            })
            .collect();
        let reconstructed = KVCachePlan {
            keep: KeepSpec::LayerWide(keep),
            merges,
        };
        assert_eq!(reconstructed, plan);
        unsafe { PlanArena::free(abi.owner) };
    }

    #[test]
    fn plan_arena_perhead_flattens_with_outer_lens() {
        let plan = KVCachePlan {
            keep: KeepSpec::PerHead(vec![vec![1, 2], vec![3, 4, 5]]),
            merges: Vec::new(),
        };
        let abi = PlanArena::into_abi(plan);
        assert_eq!(abi.keep_kind, 1);
        let keep = unsafe { core::slice::from_raw_parts(abi.keep_ptr, abi.keep_len) };
        assert_eq!(keep, &[1usize, 2, 3, 4, 5]); // 전 head concat
        let lens =
            unsafe { core::slice::from_raw_parts(abi.keep_outer_lens, abi.keep_outer_count) };
        assert_eq!(lens, &[2usize, 3]);
        unsafe { PlanArena::free(abi.owner) };
    }

    // ── GATE-C v2 Format C-ABI round-trip (ADR-0009 CF1, `.so` 없이 in-process 검증) ──

    /// 테스트용 format — q4_0-like descriptor. 매크로가 emit 하는 것과 동형의 핸들 lifecycle
    /// (make/layout/drop) thunk 를 노출해 [`KVLayoutDesc`] POD by-value 전달의 무손실을 검증한다.
    struct RtFormat;
    impl KVFormat for RtFormat {
        fn name(&self) -> &str {
            "rt_format"
        }
        fn layout(&self) -> KVLayoutDesc {
            KVLayoutDesc {
                block_elems: 32,
                bits: 4,
                scale_layout: ScaleLayout::PerBlockF16,
                packing: Packing::Nibble,
            }
        }
    }

    type RtHandle = Box<dyn KVFormat>;
    unsafe extern "C" fn rt_make() -> *mut c_void {
        Box::into_raw(Box::new(Box::new(RtFormat) as RtHandle)) as *mut c_void
    }
    unsafe extern "C" fn rt_layout(h: *mut c_void) -> KVLayoutDesc {
        // SAFETY: h 는 rt_make 가 만든 Box<Box<dyn KVFormat>>.
        let fmt: &dyn KVFormat = unsafe { &**(h as *const RtHandle) };
        fmt.layout()
    }
    unsafe extern "C" fn rt_drop(h: *mut c_void) {
        // SAFETY: h 는 rt_make 가 만든 Box<Box<dyn>>, 1회만 호출.
        drop(unsafe { Box::from_raw(h as *mut RtHandle) });
    }

    #[test]
    fn format_vtable_layout_pod_round_trip() {
        // 매크로의 cdylib 경로(plugin-cdylib)와 동형의 vtable 을 수동 구성해 ABI 경로를 feature 없이 검증.
        let vtable = FormatVTableAbi {
            name: b"rt_format\0".as_ptr() as *const c_char,
            make: rt_make,
            layout: rt_layout,
            drop: rt_drop,
        };
        // make → layout → drop: KVLayoutDesc 가 extern "C" 경계를 값으로 무손실 왕복.
        let handle = unsafe { (vtable.make)() };
        assert!(!handle.is_null(), "make 가 non-null 핸들 반환");
        let desc = unsafe { (vtable.layout)(handle) };
        assert_eq!(
            desc,
            RtFormat.layout(),
            "KVLayoutDesc 가 extern \"C\" by-value 로 무손실 전달"
        );
        // name(null-종단 'static) 회수.
        let name = unsafe { core::ffi::CStr::from_ptr(vtable.name) }
            .to_str()
            .unwrap();
        assert_eq!(name, "rt_format");
        unsafe { (vtable.drop)(handle) };
    }

    // ── ADR-0010 V1: 봉투(ExportAbi) by-value sret round-trip + 다회 호출 누적 ──

    // stage 봉투 round-trip 용 stub vtable (호출 안 함 — 마샬링만 검증).
    unsafe extern "C" fn st_make(_p: *const StageParams) -> *mut c_void {
        ::core::ptr::null_mut()
    }
    unsafe extern "C" fn st_plan(_h: *mut c_void, _c: *const StageCtxAbi, _o: *mut PlanAbi) -> i32 {
        KV_PLAN_NOOP
    }
    unsafe extern "C" fn st_plan_free(_o: *mut c_void) {}
    unsafe extern "C" fn st_drop(_h: *mut c_void) {}

    static FMT_EXPORT_VTS: [FormatVTableAbi; 2] = [
        FormatVTableAbi {
            name: b"rt_env_a\0".as_ptr() as *const c_char,
            make: rt_make,
            layout: rt_layout,
            drop: rt_drop,
        },
        FormatVTableAbi {
            name: b"rt_env_b\0".as_ptr() as *const c_char,
            make: rt_make,
            layout: rt_layout,
            drop: rt_drop,
        },
    ];

    static STAGE_EXPORT_VTS: [PluginVTableAbi; 2] = [
        PluginVTableAbi {
            name: b"rt_st_a\0".as_ptr() as *const c_char,
            make: st_make,
            plan: st_plan,
            plan_free: st_plan_free,
            drop: st_drop,
        },
        PluginVTableAbi {
            name: b"rt_st_b\0".as_ptr() as *const c_char,
            make: st_make,
            plan: st_plan,
            plan_free: st_plan_free,
            drop: st_drop,
        },
    ];

    // export_plugin! 의 register_kv_formats_v2 와 동형 — 봉투를 by-value(sret >16B) 로 반환.
    extern "C" fn mk_format_export() -> FormatExportAbi {
        FormatExportAbi {
            abi_version: KV_FORMAT_ABI_VERSION,
            count: FMT_EXPORT_VTS.len(),
            vtables: FMT_EXPORT_VTS.as_ptr(),
        }
    }
    extern "C" fn mk_stage_export() -> StageExportAbi {
        StageExportAbi {
            abi_version: KV_STAGE_ABI_VERSION,
            count: STAGE_EXPORT_VTS.len(),
            vtables: STAGE_EXPORT_VTS.as_ptr(),
        }
    }

    /// FormatExportAbi 가 extern "C" by-value(sret) 로 무손실 왕복하고, 로더가 `vtables.add(i)` 로
    /// .so static 배열 원소를 (봉투 스택 폐기 후에도) 정상 접근함을 검증.
    #[test]
    fn format_export_abi_by_value_sret_round_trip() {
        let env = mk_format_export();
        assert_eq!(env.abi_version, KV_FORMAT_ABI_VERSION);
        assert_eq!(env.count, 2);
        for (i, expect) in ["rt_env_a", "rt_env_b"].iter().enumerate() {
            // SAFETY: vtables 는 FMT_EXPORT_VTS('static 배열) base, i < count.
            let vt = unsafe { &*env.vtables.add(i) };
            let name = unsafe { core::ffi::CStr::from_ptr(vt.name) }
                .to_str()
                .unwrap();
            assert_eq!(&name, expect);
        }
    }

    /// StageExportAbi 동형 round-trip (양축 대칭).
    #[test]
    fn stage_export_abi_by_value_sret_round_trip() {
        let env = mk_stage_export();
        assert_eq!(env.abi_version, KV_STAGE_ABI_VERSION);
        assert_eq!(env.count, 2);
        for (i, expect) in ["rt_st_a", "rt_st_b"].iter().enumerate() {
            // SAFETY: vtables 는 STAGE_EXPORT_VTS('static 배열) base, i < count.
            let vt = unsafe { &*env.vtables.add(i) };
            let name = unsafe { core::ffi::CStr::from_ptr(vt.name) }
                .to_str()
                .unwrap();
            assert_eq!(&name, expect);
        }
    }

    // 다회 호출 — 한 crate 에서 register_kv_format! 2회(v1 단일심볼 ABI 에선 __REGISTER_KV_FORMAT_REG
    // E0428 충돌로 불가했음). const-block 격리로 비로소 가능(ADR-0010 E2).
    crate::register_kv_format!("v1_mc_a", || Box::new(DummyFormat));
    crate::register_kv_format!("v1_mc_b", || Box::new(DummyFormat));

    /// 정적 경로: 2회 호출이 모두 KV_FORMATS 에 등록(find_kv_format 가시화).
    #[test]
    fn register_kv_format_multicall_static() {
        assert!(find_kv_format("v1_mc_a").is_some(), "v1_mc_a 정적 등록");
        assert!(find_kv_format("v1_mc_b").is_some(), "v1_mc_b 정적 등록");
    }

    /// 동적 경로(plugin-cdylib): 2회 호출이 모두 PLUGIN_KV_FORMAT_VTABLES 에 누적.
    #[cfg(feature = "plugin-cdylib")]
    #[test]
    fn register_kv_format_multicall_dynamic() {
        let names: Vec<&str> = PLUGIN_KV_FORMAT_VTABLES
            .iter()
            .map(|vt| {
                unsafe { core::ffi::CStr::from_ptr(vt.name) }
                    .to_str()
                    .unwrap()
            })
            .collect();
        assert!(names.contains(&"v1_mc_a"), "v1_mc_a 동적 누적: {names:?}");
        assert!(names.contains(&"v1_mc_b"), "v1_mc_b 동적 누적: {names:?}");
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
