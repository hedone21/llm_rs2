# KVCacheStage 기법 추가 가이드 (stage 축 확장)

KV 캐시의 **상주 토큰을 조절하는** 새 기법(eviction / 가중 merge 등)을 **엔진 코어 수정 0** 으로
추가하는 절차다. ADR-0003(확장 메커니즘 = 정적 link technique crate + linkme 자동 등록) +
ADR-0004(`KVCacheStage` plan-returning trait)의 구현이며, 동작하는 템플릿은
`crates/techniques/example-keep-recent/` 다.

> **3축 직교(CONTEXT.md)**: stage(상주 토큰 조절) ⊥ format(정밀도/레이아웃) ⊥ hardware(연산 위치).
> 본 가이드는 **stage 축**에만 멤버를 더한다 — format/hardware 코드는 한 줄도 건드리지 않는다.

---

## 1. Hook — 무엇을 구현하나

기법은 `technique-api` crate 의 [`KVCacheStage`] trait **하나만** 구현한다
(`crates/technique-api/src/lib.rs`):

```rust
pub trait KVCacheStage: Send + Sync {
    /// CLI `--eviction-policy <name>` 와 매칭, 로깅용. 레지스트리 내 유일.
    fn name(&self) -> &str;

    /// 보존/병합 계획 산출. None = 미적용(no-op). ctx 읽기 + impl 상태(Mutex)로 계산.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan>;
}
```

**핵심 원칙(ADR-0004 D1)**: `plan` 은 캐시를 *읽고* **계획만 반환**한다. 버퍼를 직접 변형하지 않는다 —
변형(merge 적용 + compaction)은 엔진이 독점 실행한다. 이 plan-returning 모델이 (a) 기법이 버퍼를
손상시키지 못하게 하고, (b) 미래 `.so` C-ABI 경계를 최소화한다.

기법은 엔진 타입(`KVCache`/`Backend`)을 참조하지 않는다. 캐시 상태는 `technique-api` 의 읽기 추상
[`StageCtx`] 로만 읽는다(엔진이 `&KVCache` 위로 구현):

| accessor | 용도 |
|---|---|
| `fn current_pos(&self) -> usize` | 현재 유효 토큰 수(모든 기법의 budget 출발점) |
| `fn target_len(&self) -> usize` | 보존할 절대 토큰 수(ratio→len 환산은 엔진 책임) |
| `fn layer_idx(&self) -> usize` | 현재 layer index(per-layer 동작용) |
| `fn importance(&self) -> Option<&[f32]>` | flat per-token score. Some=score-based, None=score-free |
| `fn n_kv_heads(&self) -> usize` | KV head 수 |
| `fn head_dim(&self) -> usize` | head 당 차원 |
| `fn head_score(&self, kv_head, pos) -> f32` | per-head score(per-head 기법용) |
| `fn has_head_scores(&self) -> bool` | per-head score 존재 여부 |
| `fn dequant_k(&self, pos, head, out: &mut [f32])` | raw K 를 dtype-무관 f32 로 읽기(`out.len()==head_dim`) |

상태(예: d2o 의 EMA τ)는 `&self` + interior-mutability(`Mutex`)로 기법 struct 가 보유한다(D4).
ctx 로 thread 하지 않는다.

---

## 2. 반환 타입 — `KVCachePlan`

```rust
pub struct KVCachePlan {
    pub keep: KeepSpec,            // 보존 토큰 모양(배타)
    pub merges: Vec<WeightedMerge>,// 가중 병합 지시(직교, 없으면 빈 Vec)
}

pub enum KeepSpec {
    LayerWide(Vec<usize>),     // sliding/h2o/streaming/no_eviction/d2o. prefix 포함 ascending
    PerHead(Vec<Vec<usize>>),  // h2o+. [n_kv_heads][keep], 각 ascending·길이 동일
}

pub struct WeightedMerge {
    pub into: usize,             // 병합 대상 retained 위치
    pub into_weight: f32,        // into 자신의 가중치(d2o Eq.11 w_c)
    pub from: Vec<(usize, f32)>, // (병합될 evicted 위치, 가중치). Σw + into_weight ≈ 1
}
```

`keep`/`merges` 위치는 compact 적용 직전(pre-compact)의 논리 좌표다. 엔진 executor 매핑:

- `LayerWide` + 빈 merges → `compact_keep_positions`(keep 만 앞으로 당김).
- `LayerWide` + merges → `apply_weighted_merges`(가중 in-place 병합, F32/F16/Q4_0) → `compact`.
- `PerHead`(merge 유무 무관) → **현재 executor 미배선 → `bail!`**(단계 ⑤ deferred). `KVCache::compact_keep_positions_for_head` primitive 는 존재하나 `execute_kv_plan`(stage_registry.rs)이 아직 호출하지 않는다. h2o+ per-head 활성화는 head_importance session forward(F5)와 함께 ⑤에서 배선된다. → **신규 기법은 `KeepSpec::PerHead` 를 반환하면 현재 런타임 `bail!` 된다**(LayerWide 만 실행 가능).

`new_pos` 는 plan 에 싣지 않는다 — 엔진이 `keep.len()` 으로 도출한다.

---

## 3. 등록 — crate + linkme + force-link 1줄

### 3-1. technique crate 신설

`crates/techniques/<name>/` 폴더를 만든다. workspace 의 `members` glob `crates/techniques/*`
(루트 `Cargo.toml`)이 자동으로 편입한다.

`crates/techniques/<name>/Cargo.toml`:
```toml
[package]
name = "<name>"
version = "0.1.0"
edition = "2024"

[dependencies]
technique-api = { path = "../../technique-api" }   # 기법은 이 crate 에만 의존
linkme = "0.3"
```

`crates/techniques/<name>/src/lib.rs` — `KVCacheStage` 구현 + 등록:
```rust
use linkme::distributed_slice;
use technique_api::{
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageCtx, StageParams,
};

struct MyStage { /* config 필드(StageParams 로부터) */ }

impl KVCacheStage for MyStage {
    fn name(&self) -> &str { "my_stage" }
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        // ctx 를 읽어 계획 산출. 예: 최근 target_len 토큰 유지(example-keep-recent 참조).
        let (cur, tgt) = (ctx.current_pos(), ctx.target_len());
        if cur <= tgt { return None; }
        Some(KVCachePlan {
            keep: KeepSpec::LayerWide((cur - tgt..cur).collect()),
            merges: Vec::new(),
        })
    }
}

#[distributed_slice(KV_CACHE_STAGES)]
static MY_STAGE: KVCacheStageReg = KVCacheStageReg {
    name: "my_stage",
    make: |_p: StageParams| Box::new(MyStage { /* ... */ }),
};
```

`KVCacheStageReg` 는 `{ name: &'static str, make: fn(StageParams) -> Box<dyn KVCacheStage> }`.
`StageParams` 는 엔진이 CLI args 를 매핑한 평탄 값(`eviction_window`/`protected_prefix`/`keep_ratio`/
`sink_size`/`streaming_window`).

### 3-2. ★ 바이너리에 링크 — force-link 1줄 (필수)

> **주의(M3 실측, ADR-0003 §4 정정)**: Cargo 의존성 **1줄만으로는 부족**하다. Rust 는 **참조되지 않는
> 의존 rlib 을 링크에서 제외**하므로(dead-crate elision), 기법 crate 의 `#[distributed_slice]` 등록이
> 최종 바이너리에 포함되지 않는다(실측: `find_stage` 가 `None`).

활성화하려면 (1) 엔진(또는 bin)의 `Cargo.toml` 에 의존 1줄, (2) designated 지점에 **force-link 참조
1줄**을 추가한다:

```toml
# engine/Cargo.toml (또는 bin 의 manifest)
my-stage = { path = "../crates/techniques/my-stage" }
```
```rust
// 엔진의 technique-link 모듈(또는 bin 진입부) 한 곳에
use my_stage as _;   // 미참조 crate 를 링크에 강제 포함 → distributed_slice 등록 활성화
```

이 둘은 모두 **기계적 설정성 라인**이라 기존 *로직* 을 수정하지 않는다 — OCP 유지. 즉 기법 추가 비용 =
**폴더 + dep 1줄 + force-link 1줄**.

### 3-3. production 활성화 — feature install + 선택 + (value-aware 시) score 공급

위 3-1/3-2 는 "등록"까지다. 사용자가 **CLI 로 선택**하고 production 추론에서 **실제로 동작**하게 하려면
세 가지를 더한다(정본 예 = CAOTE, ADR-0004 §8):

1. **feature install(권장)** — dep 를 optional 로 두고 feature 로 opt-in 한다. 연구용 기법을 무조건
   코어 의존으로 묶지 않아 default 빌드가 깨끗하고, feature = "plugin 설치" 단위가 된다.
   ```toml
   # engine/Cargo.toml
   my-stage = { path = "../crates/techniques/my-stage", optional = true }
   [features]
   my_stage = ["dep:my-stage"]
   ```
   force-link 도 `#[cfg(feature = "my_stage")] use my_stage as _;`. feature OFF = 미링크 =
   `find_stage` None → 아래 선택 seam 이 "unknown policy" 로 graceful fail.

2. **선택 seam = 무수정** — 정책 선택부(`session/chat/session.rs`, `session/assembly/build_bench_loop.rs`)는
   이미 `name => find_stage(name) → StageBackedPolicy` generic fallback 이라 **건드릴 필요 없다**(OCP).
   CLI 표현만 `EvictionCmd` 에 variant 추가(튜닝 파라미터 없으면 unit variant) + `policy_name()` 한 줄.
   feature-gate(`#[cfg(feature = "my_stage")]`)하면 미설치 시 subcommand 자체가 사라진다.

3. **value-aware 면 score 공급** — `StageCtx::tensor(Value)` 는 cache 만으로 항상 노출되지만, 가중치
   (importance) 가 필요하면 decode 경로가 `force_evict_with_scores` 로 흘리도록 `score_based` 집합에 이름을
   추가한다(미공급 시 importance=None → degenerate). attention-weight(`last_attn`) 까지 쓰는 정밀화는
   trait/cache_manager threading 이 필요(ADR-0004 §8 Tier 2, deferred).

> **현 한계(ADR-0004 §8 Landmine)**: value-aware eviction 을 E2E 실행하는 shipping 바이너리는 argus-*
> 마이그레이션 중이라 아직 없다(chat session 추출본 미배선, argus_bench score-free, legacy 동결). 본 레시피는
> "배선"을 완성하나 live E2E 는 그 전환에 종속.

---

## 4. 동작 예제

`crates/techniques/example-keep-recent/` 가 실제로 컴파일·등록되는 최소 템플릿이다("최근 N 토큰 유지",
sliding 의 prefix=0 변형). `technique-api` 에만 의존하고 `StageCtx` 의 `current_pos`/`target_len` 만
읽는 순수 계산이다. 새 기법은 이 crate 를 복사해 시작하면 된다.

엔진은 이 crate 를 **dev-dependency** 로만 링크하고(`engine/Cargo.toml`), `stage_registry.rs` 테스트가
`use example_keep_recent as _;` + `find_stage("example_keep_recent")` 로 cross-crate 등록을 검증한다.

---

## 5. 게이트 — 추가 후 확인

1. **컴파일**: `cargo build -p llm_rs2`(엔진), `cargo build -p <name>`(crate 단독 — technique-api 만
   의존하는지).
2. **등록**: `technique_api::find_stage("<name>")` 가 `Some` 인지(cross-crate). 예제 검증 =
   `engine/src/pressure/eviction/stage_registry.rs` 의 `example_technique_crate_visible_to_engine`,
   `d2o_stage_registered` 테스트.
3. **동작 정확성**(plan→compact 가 의도대로): `compact_parity` 패턴 — plan 적용 결과가 기준 거동과
   bit-identical 인지 host unit test. 가중 merge 기법은 `apply_weighted_merges` ≡ 기준 merge 를
   확인(`d2o_handler.rs` 의 `d2o_stage_eq_handler_*` 참조).
4. **release linkme 생존**(fat-LTO `--gc-sections`): `cargo test --release -p llm_rs2 --lib
   stage_registry` — `#[distributed_slice]` 등록이 release 빌드에서 silent drop 되지 않는지. 엔진은
   startup self-test(`ensure_builtin_stages_registered`)로도 fail-fast.

---

## 6. 참조

- ADR-0003 — 확장 메커니즘(정적 crate + linkme, `.so` 보류). §4 D4 에 force-link 정정.
- ADR-0004 — `KVCacheStage` plan-returning trait 설계(D1~D6, executor 매핑, Q4_0 merge 정정).
- `crates/technique-api/src/lib.rs` — trait/타입 정본.
- `engine/src/pressure/eviction/stage_registry.rs` — 빌트인 등록 + executor(`execute_kv_plan`) +
  역어댑터(`StageBackedPolicy`).
- `engine/src/pressure/standard_format.rs` — `apply_weighted_merges`(가중 merge 적용).
