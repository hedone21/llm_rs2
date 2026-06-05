# KVCacheStage 기법 API 레퍼런스 (외부 기여자용)

> 대상 독자: `crates/techniques/<name>/` 에 KV 캐시 **stage 축**(eviction / 가중 merge) 기법을 새로 추가하려는 외부 기여자.
> 이 문서는 각 타입·trait·함수의 **정확한 시그니처와 계약**(언제 `None`/0/false/빈 Vec, 불변식, dyn-safety, panic 조건, borrow/lifetime, 호출 시점·상태 누적, 엔진이 plan 을 소비하는 방식)을 코드에서 그대로 풀어쓴 상세 레퍼런스다.
> 단계별 추가 절차(폴더·Cargo·게이트)는 how-to 가이드 [`docs/50_adding_kvcache_stage.md`](50_adding_kvcache_stage.md) 를 보라 — 본 문서는 그 가이드를 **API 레벨에서 보완**하며 중복을 피한다.
>
> 정본 위치: 타입/trait = `crates/technique-api/src/lib.rs`, 엔진 측 구현/executor/등록 = `engine/src/pressure/eviction/stage_registry.rs`, 가중 merge 산술 = `engine/src/pressure/standard_format.rs`. 설계 근거 = [ADR-0003](adr/0003-extension-mechanism-static-crates.md)(확장 메커니즘) + [ADR-0004](adr/0004-kvcachestage-plan-returning-trait.md)(plan-returning trait).

## 개요 — stage 축과 plan-returning 모델

KV 캐시 관리는 세 축이 직교한다(`CONTEXT.md`): **stage**(상주 토큰 조절: eviction/merge) ⊥ **format**(정밀도/레이아웃) ⊥ **hardware**(연산 위치). 본 문서가 다루는 **stage 축**의 확장 표면은 단일 trait `KVCacheStage` 하나다(ADR-0004). 기법은 별도 technique crate(`crates/techniques/<name>/`)에서 `technique-api` crate **에만** 의존해 이 trait 을 구현하고, `#[distributed_slice(KV_CACHE_STAGES)]` 로 자기를 전역 슬라이스에 등록한다 — **엔진 코어 수정 0** 으로 가산 확장된다(ADR-0003).

핵심 설계는 **plan-returning, self-mutating 아님**(ADR-0004 D1)이다: 기법의 `plan(ctx)` 은 캐시 상태를 읽기 추상 `StageCtx` 로 *읽기만* 하고 "어느 토큰을 보존하고(`keep`) 어떻게 가중 병합할지(`merges`)" 의 *계획*(`KVCachePlan`)만 반환한다. **버퍼 변형은 엔진 executor 가 독점**한다(`execute_kv_plan`). 이 모델은 (a) 기법이 버퍼를 손상시키지 못하게 막고, (b) 미래의 `.so`/C-ABI 경계를 최소화한다(`technique-api` 는 엔진 타입 `KVCache`/`Backend` 를 일절 참조하지 않는 단방향 의존).

의존 방향: `engine → technique-api ← technique crate` (단방향, 순환 없음).

---

## 1. `KVCacheStage` trait

기법이 구현하는 **유일한 핵심 trait**. 정의: `crates/technique-api/src/lib.rs:112-118`.

```rust
pub trait KVCacheStage: Send + Sync {
    /// 기법 이름 (CLI `--eviction-policy <name>` 와 매칭, 로깅용). 슬라이스 내 유일해야 한다.
    fn name(&self) -> &str;

    /// 보존/병합 계획 산출. `None` = 미적용(no-op). ctx 읽기 + impl 상태(Mutex)로 계산한다.
    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan>;
}
```

### `fn name(&self) -> &str`

- 기법의 고유 이름. CLI `--eviction-policy <name>` 와 문자열 매칭되고 로깅에 쓰인다.
- **불변식 — 슬라이스 내 유일.** `find_stage(name)` 가 `KV_CACHE_STAGES.iter().find(...)` 로 **첫 매치만** 반환하므로(`lib.rs:158-160`), 중복 이름은 silent shadowing 을 일으킨다. 등록 시점에 강제되는 unique 검사는 **없다** — 기여자 책임.
- **반환 타입 `&str`** — 보통 `&'static str` 리터럴(예: `"example_keep_recent"`). `&self` 에 묶인 borrow 라도 무방하나, 호출마다 동일 값을 안정적으로 반환해야 한다(누적 상태에 따라 이름이 바뀌면 안 됨).
- 등록 항목 `KVCacheStageReg.name` 과 인스턴스의 `name()` 일치는 **컨벤션**이다(코드 강제 없음). 빌트인은 일치시킨다(`stage_registry.rs:233/243/252/268`).

### `fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan>`

기법의 핵심. 버퍼를 직접 변형하지 않고 *계획*만 반환한다.

- **`&self` 수신자 (ADR-0004 D1·D4)** — `&mut self` 가 아니다. 이유: (1) trait object(`Box<dyn KVCacheStage>`)로 호출되고, (2) `Send + Sync` bound 와 함께 공유 참조로 병렬 안전하게 호출 가능해야 하기 때문. 호출 간 누적 상태(d2o EMA threshold τ 등)는 `ctx` 로 thread 하지 않고 **impl struct 가 interior-mutability(`Mutex`)로 보유**한다(D4). 예: `D2OStage { config, state: Mutex<D2OState> }` (`d2o_handler.rs:423-426`), `plan` 내부에서 `let mut state = self.state.lock().unwrap();` 로 잠가 갱신(`d2o_handler.rs:447`).
  - `Send + Sync` 때문에 평범한 `Cell`/`RefCell` 은 `Sync` 가 아니라 **컴파일 거부**된다. `Mutex`/`RwLock`/`AtomicXxx` 를 써야 한다.
- **`ctx: &dyn StageCtx` (D5 dyn-safety)** — 캐시 읽기 추상을 trait object 로 받는다. 기법은 `ctx.current_pos()`, `ctx.target_len()`, `ctx.importance()`, `ctx.dequant_k(...)` 등으로만 캐시를 읽으며 `&KVCache` 같은 엔진 타입을 절대 직접 받지 않는다(§2 참조).
- **`Option<KVCachePlan>` 반환 계약:**
  - `Some(plan)` → 엔진 executor 가 `plan` 을 소비해 버퍼를 변형한다(§4).
  - `None` → **no-op(미적용)**. executor 가 호출되지 않는다. 가장 흔한 케이스는 "축소 불필요". 표준 패턴:
    - `current_pos() <= target_len()` 일 때(예제: `example-keep-recent/src/lib.rs:27-29`).
    - d2o 처럼 내부 `compute_d2o_plan` 이 `None`(`current <= keep` 또는 evict 대상 없음, `d2o_handler.rs:218,247`)을 내면 `?` 로 전파(`d2o_handler.rs:452-462`).
    - 어댑터 `EvictionPolicyAsStage::plan` 은 내부 `plan_keep` 이 `None`(per-head 등 layer-wide 로 표현 불가)이면 `?` 로 전파(`stage_registry.rs:48-51`).
  - **불변식 — `Some` 인데 사실상 no-op 인 plan 을 만들지 말 것.** 특히 **빈 `KeepSpec::LayerWide(vec![])` 는 no-op 가 아니라 "전부 삭제"** 다: executor 가 `set_current_pos(keep.len())` = `set_current_pos(0)` 을 실행해 캐시를 완전히 비운다(§4). 축소 불필요 시에는 `None` 을 반환하라.
- **panic 금지** — `plan` 은 엔진 forward 경로에서 호출된다. `Mutex::lock().unwrap()` 의 poison-panic 은 빌트인이 모두 쓰는 관용 패턴이라 허용되나, 그 외 직접 panic 은 금지한다.

### 호출 시점 · 상태 누적

엔진은 `KVCacheStage` 를 직접 호출하지 않는다. 역어댑터 `StageBackedPolicy`(`stage_registry.rs:163-214`)가 레거시 `EvictionPolicy` 표면(`evict`/`evict_with_scores`)으로 노출하며, 그 내부 `run()`(`stage_registry.rs:174-188`)이 다음 2-phase 로 동작한다:

1. `KVStageCtx::new(cache, target_len, importance)` 로 **immutable** ctx 생성,
2. `self.stage.plan(&ctx)` 로 plan 획득 (이 블록이 끝나면 `&KVCache` borrow 종료),
3. `if let Some(plan) = plan { execute_kv_plan(cache, &plan)?; }` 로 `&mut KVCache` 변형 실행.

```rust
// stage_registry.rs:180-187
let plan = {
    let ctx = KVStageCtx::new(cache, target_len, importance);
    self.stage.plan(&ctx)
};
if let Some(plan) = plan {
    execute_kv_plan(cache, &plan)?;
}
```

이 read(borrow)→plan→`&mut` execute 2-phase 분리가 borrow checker 를 만족시키는 핵심이며, plan 단계에서 `&KVCache` 만 잡으므로 `plan` 구현이 ctx 의 immutable accessor 만 쓰도록 강제된다.

- **per-cache 호출** — eviction 은 layer(=KVCache)마다 한 번씩 일어난다. 단, 빌트인 경로의 `KVStageCtx::layer_idx()` 는 항상 `0` 을 반환한다(미plumb, §2-3).
- **상태 누적** — `plan` 이 `&self` 이므로 동일 인스턴스가 재사용되고, `Mutex` 안의 상태(d2o EMA τ, 카운터)가 호출 간 **누적**된다. 동일 config·cache·call 시퀀스에서 `D2OStage` 와 production `D2OHandler` 가 EMA 갱신까지 bit-identical 하도록 `compute_d2o_plan` 을 공유한다.

---

## 2. `StageCtx` trait — 캐시 읽기 추상 (9개 accessor)

기법이 KV 캐시 상태를 **읽기 전용**으로 보는 추상. 정의: `crates/technique-api/src/lib.rs:30-69`. 엔진 impl: `KVStageCtx<'a>` (`stage_registry.rs:105-156`, `&KVCache` 위로 구현).

### dyn-safety 제약 (전 메서드 공통)

`KVCacheStage::plan(&self, ctx: &dyn StageCtx)` 가 ctx 를 **trait object** 로 받으므로, `StageCtx` 의 모든 메서드는 object-safe 여야 한다(`lib.rs:23-25`):

- 제네릭 타입 파라미터 금지
- `Self` by value(`self` 소비) 금지 — 전부 `&self`
- 연관 타입(associated type) 금지
- `impl Trait` 인자 금지

이 제약 때문에 원시 K 읽기 같은 dtype-분기 접근도 **슬라이스 반환이 아니라 `out: &mut [f32]` out-param** 으로 노출된다(`dequant_k`). 누적 상태는 ctx 로 thread 하지 않는다 — `StageCtx` 는 순수 읽기 표면이며, 상태는 plugin struct 가 `Mutex` 로 보유한다(D4).

### accessor 요약표

| accessor | 시그니처 | plumb 상태 | 의미 |
|---|---|---|---|
| `current_pos` | `fn current_pos(&self) -> usize` | ✅ 실사용 | 현재 유효 토큰 수 (budget 출발점) |
| `target_len` | `fn target_len(&self) -> usize` | ✅ 실사용 | 보존할 **절대 토큰 수**(ratio→len 환산은 엔진 책임) |
| `layer_idx` | `fn layer_idx(&self) -> usize` | ⚠️ 미plumb(항상 0) | 현재 layer index (per-layer 동작용) |
| `importance` | `fn importance(&self) -> Option<&[f32]>` | ✅ 실사용 | flat per-token score. `Some`=score-based, `None`=score-free |
| `n_kv_heads` | `fn n_kv_heads(&self) -> usize` | ✅ 실사용 | KV head 수 |
| `head_dim` | `fn head_dim(&self) -> usize` | ✅ 실사용 | head 당 차원 |
| `head_score` | `fn head_score(&self, kv_head: usize, pos: usize) -> f32` | ⚠️ 미plumb(항상 0.0) | per-head score (h2o_plus 용) |
| `has_head_scores` | `fn has_head_scores(&self) -> bool` | ⚠️ 미plumb(항상 false) | per-head score 존재 여부 |
| `dequant_k` | `fn dequant_k(&self, pos: usize, head: usize, out: &mut [f32])` | ✅ 실사용 | raw K 를 dtype-무관 f32 로 `out` 에 채움 |

> **현 프로덕션 KVCacheStage 경로에서 실사용되는 accessor 는 6개**(`current_pos`/`target_len`/`importance`/`n_kv_heads`/`head_dim`/`dequant_k`)다. `layer_idx`/`head_score`/`has_head_scores` 3개는 미plumb 상수이므로(아래), **신규 기법은 이 3개에 의존하지 말 것.**

### 2-1. `current_pos`

- **의미**: 현재 유효 토큰 수. keep/prune budget 산출의 출발점.
- **계약**: 음수 없음(`usize`). 의미 있는 축소는 `current_pos > target_len` 일 때만 가능 — `<=` 면 보통 `None`(no-op) 을 반환한다.
- **엔진 원천**(`stage_registry.rs:127-129`): `self.cache.current_pos` 필드 직접 읽기(public `KVCache::current_pos()` 와 동일 값).

### 2-2. `target_len`

- **의미**: 해소된 budget — 보존할 **절대 토큰 수**(ratio 가 아니라 len). ratio→len 환산은 엔진(`EvictionHandler`/`run_policy_eviction`) 책임이므로 plugin 은 환산된 값만 읽는다.
- **계약**: score-free/head-relative budget 기법(no_eviction/h2o_plus)은 호출하지 않을 수 있다. `target_len == 0` 인 호출 경로가 존재하므로(예: `dequant_k` 단위 테스트의 `KVStageCtx { target_len: 0 }`, `stage_registry.rs:541`) 0 을 "budget 미지정/측정-only" 로 다룰 수 있다 — 0 의 해석은 기법 책임이다(코드에 명시적 의미 부여 없음).
- **엔진 원천**(`stage_registry.rs:130-132`): `KVStageCtx::new` 생성자 인자 `self.target_len` 을 그대로 반환. 원천 체인: `run_policy_eviction` 의 `target_len` → `current_pos − target_len < MIN_EVICT_TOKENS` 이면 엔진이 eviction 자체를 skip 한다.

### 2-3. `layer_idx` — ⚠️ 미plumb(항상 0)

- **의미**: 이 plan 호출이 처리하는 layer index (d2o per-layer 예산/protect 판정용).
- **현 상태**(`stage_registry.rs:133-135`): 항상 `0` 을 반환한다(주석: "per-layer(d2o) = M4"). per-layer 변별(d2o layer-allocation)은 여전히 production if-branch `D2OHandler` 가 처리하며, 레지스트리 경유 `D2OStage` 는 non-layer-alloc config 한정이다. **기법이 `layer_idx` 에 의존하면 현재는 단일-layer(0)로 degenerate 된다.**

### 2-4. `importance`

- **의미**: flat per-token importance score 슬라이스. 위치별 인덱스 접근 전용(`imp.get(pos)`).
- **계약(`None` 의미)**:
  - `Some(slice)` → score-based 기법(h2o heavy-hitter, d2o token rank).
  - `None` → **score-free** 기법(sliding/streaming/no_eviction).
  - **빈 슬라이스 주의**: `Some(&[])`(길이 0 이지만 Some)도 가능하다. D2OStage 는 `ctx.importance().unwrap_or(&[])` 로 None→빈 슬라이스 정규화(`d2o_handler.rs:446`). score 부재를 정확히 구분하려면 `Some(&[])` 와 `None` 을 둘 다 빈 score 로 다루는 게 안전하다.
- **borrow/lifetime**: 반환 슬라이스의 borrow 는 ctx 수명(`&self`)에 묶인다. plan 반환 전까지만 빌려 쓸 수 있다.
- **엔진 원천**(`stage_registry.rs:136-138`): `KVStageCtx::new` 의 `importance: Option<&'a [f32]>` 인자 그대로 반환. `evict`(score 없음)→`None`, `evict_with_scores`→`Some(importance)` 로 분기(`stage_registry.rs:198-209`). 원천 체인: `run_policy_eviction` 의 `ScoreContext` → flat importance → `StageBackedPolicy::run`.

### 2-5. `n_kv_heads`

- **의미**: KV head 수. h2o_plus per-head 루프 상한 + `KeepSpec::PerHead` outer Vec 길이, d2o 의 `layer_dim = n_kv_heads * head_dim` 산출에 쓴다.
- **계약**: `KeepSpec::PerHead(Vec<Vec<usize>>)` 를 반환하는 기법은 outer Vec 길이를 정확히 `n_kv_heads` 로 맞춰야 한다(§3). Llama 3.2 1B 기준 8.
- **엔진 원천**(`stage_registry.rs:139-141`): `self.cache.kv_heads()`.

### 2-6. `head_dim`

- **의미**: head 당 차원. d2o 의 K vector 길이 / cosine 차원 / dequant 버퍼 크기를 결정한다.
- **계약**: `dequant_k` 의 `out` 슬라이스 길이가 정확히 이 값이어야 한다(`out.len() == head_dim`). Llama 3.2 1B 기준 64.
- **엔진 원천**(`stage_registry.rs:142-144`): `self.cache.head_dim()`.

### 2-7. `head_score` — ⚠️ 미plumb(항상 0.0)

- **의미**: per-head importance score(h2o_plus 용). row-major `[n_kv_heads * max_seq]` stride 를 엔진이 내부화하므로 plugin 은 평탄한 `(kv_head, pos) → f32` 로만 읽는다.
- **계약**: `has_head_scores()` 가 `false` 면 이 메서드 반환값은 **무의미**하다 — 반드시 `has_head_scores()` 로 가드한 뒤 호출하라.
- **현 상태**(`stage_registry.rs:145-147`): 인자를 무시하고 항상 `0.0` 을 반환한다(주석: "head_importance forward = F5/⑤"). per-head score source 가 미완이라 단계 ⑤ deferred, **h2o_plus 는 stage 레지스트리에서 제외**되어 있다(`stage_registry.rs:12`).

### 2-8. `has_head_scores` — ⚠️ 미plumb(항상 false)

- **의미**: per-head score 가 존재하는가.
- **계약**: `false` 면 per-head 기법은 `KeepSpec::LayerWide` 로 degenerate 해야 한다(score-free/flat 폴백). `head_score()` 호출 전 가드 조건.
- **현 상태**(`stage_registry.rs:148-150`): 항상 `false`(주석: "②b 미plumb"). 현 프로덕션 경로에서는 per-head 분기가 항상 LayerWide 폴백으로 떨어진다.

### 2-9. `dequant_k`

```rust
fn dequant_k(&self, pos: usize, head: usize, out: &mut [f32]);
```

- **의미**: 원시 K(`pos`, `head`)를 dtype-무관 f32 로 `out` 에 채운다(d2o cosine-nearest 매칭용). dtype 분기(F32/F16/Q4_0)는 엔진 impl 내부에서 흡수한다.
- **계약(out-param)**: 호출자는 `out.len() == head_dim` 인 버퍼를 제공해야 한다. 엔진 impl 은 `out[..head_dim]` 범위를 채운다. `out` 이 `head_dim` 보다 짧으면 인덱싱에서 **panic 가능**. 반환값 없음(`()`).
- **out-param 이유**: ① dyn-safe(제네릭 아님) ② Q4_0 dequant 임시버퍼 수명/Vec 할당 회피.
- **엔진 원천**(`stage_registry.rs:151-155`): `crate::pressure::d2o_handler::dequantize_k(self.cache, pos, head, out.len(), out)` 위임. dtype 별 처리(`d2o_handler.rs:514-547`):
  - `F32`: `out[..head_dim].copy_from_slice(&k[off..off+head_dim])`
  - `F16`: `k[off+d].to_f32()` per element
  - `Q4_0`: block 단위 dequant(`q4_block_offset` + per-block `BlockQ4_0::dequantize`, `blocks_per_pos = head_dim / QK4_0`)

  D2OHandler 와 bit-identical. 단위 테스트(`stage_registry.rs:529-549`)가 F32 경로 값 일치를 검증한다.

---

## 3. 반환 타입 — `KVCachePlan` / `KeepSpec` / `WeightedMerge`

`plan` 이 `Some(...)` 으로 반환하는 산출물. 정의: `crates/technique-api/src/lib.rs:76-103`. 이들은 **값 타입**이라 dyn-safety 제약을 받지 않는다.

### `KVCachePlan`

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct KVCachePlan {
    /// 보존 토큰 모양.
    pub keep: KeepSpec,
    /// 가중 병합 지시 (없으면 빈 Vec).
    pub merges: Vec<WeightedMerge>,
}
```

- `keep`(어느 토큰을 남기나 — 배타) ⊥ `merges`(남는 토큰에 evicted 를 합산할지 — 직교)는 **서로 독립**이다. merge-free 기법은 `merges: Vec::new()`.
- **`new_pos` 필드 부재** — "병합 후 새 토큰 수" 를 plan 에 싣지 않는다. 엔진 executor 가 `keep.len()` 으로 도출한다(`set_current_pos(keep.len())`, `stage_registry.rs:89`). `PerHead` 의 경우 "전 head 의 keep 길이 동일" 가정 위에서 도출하므로 그 불변식을 지켜야 한다.

### `KeepSpec`

```rust
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeepSpec {
    /// sliding/h2o/streaming/no_eviction/d2o. prefix 포함 ascending.
    LayerWide(Vec<usize>),
    /// h2o_plus. [n_kv_heads][keep], 각 ascending·길이 동일(엔진이 assert).
    PerHead(Vec<Vec<usize>>),
}
```

(`KVCachePlan`/`WeightedMerge` 는 `#[derive(Clone, Debug, PartialEq)]`, `KeepSpec` 만 추가로 `Eq` 를 derive 함에 유의.)

#### `LayerWide(Vec<usize>)`

모든 KV head 에 **동일한** 보존 위치 집합을 적용. 현행 빌트인 기법 전체(sliding/h2o/streaming/no_eviction/d2o)가 이 변형을 쓴다. 불변식(엔진이 가정 → 기법이 보장):

- **ascending(오름차순) 정렬 필수.** `compact_keep_positions`(`kv_cache.rs`)는 인접 위치를 연속 batch 로 묶어 한 번에 shift 하는데, ascending 이 아니면 batch 경계 판정이 깨져 **silent garbage**(panic 아님)가 된다. 예제는 `(current - target..current).collect()` 로 자연히 ascending.
- **prefix 를 명시적으로 포함시켜야 한다.** "protected prefix(BOS/시스템 프롬프트)" 는 자동 보존되지 않는다 — 보존하려면 keep Vec 에 그 위치(`0..protected_prefix`)를 직접 넣어야 한다.
- 위치는 **pre-compact 논리 좌표**(compaction 적용 직전 현재 캐시 위치).

#### `PerHead(Vec<Vec<usize>>)`

head 마다 다른 보존 집합 — `[n_kv_heads][keep]`. h2o_plus 용으로 예약. 불변식:

- outer Vec 길이 == `ctx.n_kv_heads()`.
- inner Vec 각각 ascending 정렬.
- **모든 inner Vec 길이 동일**(엔진이 `keep.len()` 으로 `new_pos` 도출하기 때문, `lib.rs:92` "엔진이 assert").

> **⚠️ 미구현 경고 — 현재 `PerHead` 를 반환하면 안 된다.** `execute_kv_plan` 의 `KeepSpec::PerHead(_)` arm 은 `anyhow::bail!("per-head executor not implemented (단계 ⑤ deferred)")` 로 즉시 에러를 낸다(`stage_registry.rs:92-96`). per-head executor 는 head_score source(F5) 미완으로 단계 ⑤ deferred. **외부 기여자가 `PerHead` plan 을 반환하면 eviction 호출 경로가 `Err` 로 실패한다(panic 아님, `Result` Err).** 현 단계 production-ready 표면은 `LayerWide` 한정이다.

### `WeightedMerge`

```rust
#[derive(Clone, Debug, PartialEq)]
pub struct WeightedMerge {
    /// 병합 대상 retained 토큰의 위치 (가중 합이 누적될 자리).
    pub into: usize,
    /// `into` 자신의 가중치 (d2o Eq.11 의 w_c).
    pub into_weight: f32,
    /// 병합될 evicted 토큰들의 (위치, 가중치).
    pub from: Vec<(usize, f32)>,
}
```

evicted 토큰들(`from`)의 K/V 벡터를 retained 토큰(`into`) 한 자리에 가중 합산한다(d2o merge compensation). 필드 의미:

- **`into`** — 가중 합이 누적될 retained 토큰 위치. 반드시 `keep` 에 포함된 보존 토큰이어야 한다.
- **`into_weight`** — `into` 자신의 가중치(d2o Eq.11 의 `w_c`).
- **`from`** — 합산될 evicted 토큰들의 `(위치, 가중치)` 쌍. 각 위치는 `keep` 에 **포함되지 않은**(evicted) 토큰이어야 한다.

불변식(기법이 보장 — 엔진은 assert 하지 않음):

- **`Σ from.1 + into_weight ≈ 1`**(magnitude 보존, d2o Eq.11). 강제되지 않으며 위반해도 패닉하지 않으나, 1 에서 벗어나면 병합 토큰의 norm 이 왜곡되어 attention 분포가 흐트러진다. d2o 의 `compute_eq11_weights`(`d2o_handler.rs:657-668`)가 `w_c + Σ w_e = 1` 을 산출하는 것이 정본 패턴이다.
- **위치는 pre-compact 논리 좌표.** `apply_weighted_merges` 는 compaction *이전* 좌표계에서 원위치를 읽으므로 `keep` 과 동일 좌표계여야 한다.
- **`from` ∩ retained == ∅**(evicted 는 retained 와 비중첩), merge 간에도 비중첩. 이 덕에 엔진의 in-place 적용이 안전하다(코드는 assert 하지 않음 — 기여자 책임).
- **빈 `from` 은 skip** — `apply_weighted_merges` 가 `if m.from.is_empty() { continue; }` 로 건너뛴다(`standard_format.rs:588`).

---

## 4. 엔진 executor 의미 — plan 을 어떻게 소비하나

기법은 버퍼를 만지지 않는다(ADR-0004 D1). 변형 진입점은 `execute_kv_plan` 이며, 가중 merge 산술은 `apply_weighted_merges` 가 담당한다. 두 함수 모두 `pub(crate)` 라 외부 technique crate 에서 직접 호출 불가 — 기여자는 자신의 plan 이 *어떻게 해석되는지*만 알면 된다.

### `execute_kv_plan`

```rust
// engine/src/pressure/eviction/stage_registry.rs:80
pub(crate) fn execute_kv_plan(cache: &mut KVCache, plan: &KVCachePlan) -> Result<()>
```

`StageBackedPolicy::run` 안에서 `plan` 이 `Some` 일 때만 호출된다(`None` → executor 미호출 = no-op). 소비 분기:

| `plan.keep` | `plan.merges` | 엔진 동작 |
|---|---|---|
| `LayerWide(keep)` | 빈 Vec | `compact_keep_positions(keep, 0)` → `set_current_pos(keep.len())` |
| `LayerWide(keep)` | 비어있지 않음 | `apply_weighted_merges(cache, &plan.merges)` → `compact_keep_positions(keep, 0)` → `set_current_pos(keep.len())` |
| `PerHead(_)` | (merges 무관) | `anyhow::bail!("per-head executor not implemented (단계 ⑤ deferred)")` — **현재 미구현** |

소스 그대로(`stage_registry.rs:81-97`):

```rust
match &plan.keep {
    KeepSpec::LayerWide(keep) => {
        if !plan.merges.is_empty() {
            crate::pressure::standard_format::apply_weighted_merges(cache, &plan.merges);
        }
        cache.compact_keep_positions(keep, 0)?;
        cache.set_current_pos(keep.len());
        Ok(())
    }
    KeepSpec::PerHead(_) => {
        anyhow::bail!("per-head executor not implemented (단계 ⑤ deferred)")
    }
}
```

**실행 순서 불변식 — merge 는 반드시 compact 이전.** compact 가 토큰을 앞으로 당기면 위치 인덱스가 바뀌므로, merge 를 compact 뒤에 적용하면 잘못된 자리를 가리킨다. 따라서 `WeightedMerge::into`/`from` 와 `KeepSpec` 의 위치는 **동일한 pre-compact 좌표계**(현 `current_pos` 기준)를 공유해야 한다.

`LayerWide(keep)` 계약 (기여자가 보장):
- ascending 정렬 필수(위반 시 silent garbage).
- `keep` 위치는 `[0, current_pos)` 범위 유효 토큰(prefix 포함 가능).
- **빈 `keep` → 캐시 0 토큰 리셋**: `compact_keep_positions` 가 빈 keep 에 즉시 `Ok(())` 를 내지만 이어지는 `set_current_pos(0)` 으로 캐시가 비워진다(high_water 도 0). no-op 의도면 `plan()` 에서 `None` 을 반환하라.
- compact 는 `write_start=0` 고정(retained 를 맨 앞부터 연속 재배치). 동일 위치(src==dst)는 `shift_positions` 가 skip 하므로 손상 없음.

### `apply_weighted_merges`

```rust
// engine/src/pressure/standard_format.rs:576
pub(crate) fn apply_weighted_merges(cache: &mut KVCache, merges: &[WeightedMerge])
```

- 반환값 없음(`()`) — 실패 모드 없음(에러 없이 in-place 적용). `merges.is_empty()` 면 즉시 return, 개별 `WeightedMerge` 도 `m.from.is_empty()` 면 skip.
- **가중합 공식** — 각 `WeightedMerge m` 마다 **모든 KV head `h`** 에 걸쳐 `into` 토큰 한 자리를 head_dim 차원 `d` 별로 덮어쓴다:

  ```
  out[d] = into_weight · into[d] + Σ_i  from_w[i] · from[i][d]
  ```

  소스(F32, `merge_row_weighted_f32`, `standard_format.rs:664-670`):

  ```rust
  for d in 0..head_dim {
      let mut acc = into_w * buf[into_off + d];
      for (idx, &fo) in from_offs.iter().enumerate() {
          acc += from_w[idx] * buf[fo + d];
      }
      buf[into_off + d] = acc;
  }
  ```

  `into` 항을 먼저, `from` 은 `m.from` Vec 의 list 순서대로 누적한다.
- **dtype 디스패치(K/V 독립)** — K 버퍼(`cache.k_buffer.dtype()`)와 V 버퍼(`cache.v_buffer.dtype()`)를 각각 별도 match 로 디스패치한다(`standard_format.rs:597,625`). 지원:
  - `F32` → `merge_row_weighted_f32`(위 공식 직접).
  - `F16` → `merge_row_weighted_f16`(각 항 `to_f32()` 승격 → f32 누적 → `f16::from_f32(acc)` 재양자화).
  - `Q4_0` → `merge_row_weighted_q4`(`from` 블록 dequant → `into` 블록 dequant 후 `*= into_w` → `+= from_w·from` → `BlockQ4_0::quantize` 재양자화; 위치는 `q4_block_offset`, `blocks_per_pos = head_dim / QK4_0`).
  - 그 외 → `_ => {}`(skip).

> **주의 — `apply_weighted_merges` 는 Q4_0 merge 를 활성한다**(ADR-0004 §4 M4 정정). 이는 `KVCacheFormat::compact` 가 쓰는 별도 함수 `apply_merges`(`standard_format.rs`, **균등 가중**, F32/F16 만, Q4_0 skip)와 **다르다**. plan executor 경로(`execute_kv_plan`)는 항상 `apply_weighted_merges` 를 쓰므로 Q4_0 가중 merge 가 적용된다. 두 함수를 혼동하지 말 것.

d2o 의 `scatter_reduce_merge_layer_wide` 와 bit-identical 임이 M4-b 에서 증명됐다.

---

## 5. 등록 API — 가산 확장

기법은 trait 구현 + 전역 슬라이스 등록 2단계로 끝난다. 정의: `crates/technique-api/src/lib.rs:143-165`.

### `KVCacheStageReg` — 등록 항목

```rust
pub struct KVCacheStageReg {
    pub name: &'static str,
    pub make: fn(StageParams) -> Box<dyn KVCacheStage>,
}
```

- **`name: &'static str`** — CLI `--eviction-policy <name>` 키이자 로깅 이름. **슬라이스 내 유일**해야 한다(중복 시 `find_stage` 가 첫 매치만 반환 → silent shadow, 코드상 중복 검출 assert 없음). `KVCacheStage::name()` 과의 일치 강제도 없다(컨벤션).
- **`make: fn(StageParams) -> Box<dyn KVCacheStage>`** — 일반 `fn` 포인터(환경 캡처 불가, non-capturing closure 만 coercion). 항상 `Box<dyn KVCacheStage>`(heap, dyn-dispatch)를 반환하므로 `KVCacheStage` 는 object-safe 여야 한다.

### `KV_CACHE_STAGES` — 전역 등록 슬라이스

```rust
#[distributed_slice]
pub static KV_CACHE_STAGES: [KVCacheStageReg] = [..];
```

linkme `distributed_slice`. 링크된 **모든** technique crate(및 엔진 빌트인)의 등록이 **링크 타임**에 수집된다 — 런타임 생성자가 아닌 순수 링커 데이터라 `panic="abort"`/Android 에서 안전(ADR-0003 §3). 주의:

- 항목 순서는 **링크 순서 의존**(미정의에 가깝다). 기법 선택은 순서가 아니라 `name` 으로만 한다.
- fat-LTO + `--gc-sections` 가 미참조 섹션을 silent drop 할 수 있다(ADR-0003 §4) — 엔진은 build 진입 시 `ensure_builtin_stages_registered()`(`stage_registry.rs:219-229`)로 빌트인 3종(`sliding`/`streaming`/`h2o`) 등록을 단언해 fail-fast 한다.

### `StageParams` — 팩토리 입력 (5필드)

```rust
#[derive(Clone, Copy, Debug)]
pub struct StageParams {
    pub eviction_window: usize,    // sliding window 크기 (최근 유지 토큰 수)
    pub protected_prefix: usize,   // 앞에서 보호할 prefix 길이 (BOS/시스템 프롬프트)
    pub keep_ratio: f32,           // heavy-hitter 유지 비율 (H2O 계열)
    pub sink_size: usize,          // streaming sink(attention sink) 크기
    pub streaming_window: usize,   // streaming window 크기 (0 이면 엔진이 기본값 유도)
}
```

엔진이 CLI args 를 이 평탄 struct 로 매핑해 `make` 에 넘긴다(`technique-api` 가 엔진 args 타입에 의존하지 않도록 원시 스칼라만 싣는다). `Copy` 이므로 값으로 전달된다. 빌트인 매핑 원천(`session.rs`):

- `eviction_window` → `args.eviction_window` (sliding 의 `SlidingWindowPolicy::new(p.eviction_window, p.protected_prefix)` 첫 인자, `stage_registry.rs:236`).
- `protected_prefix` → `--protected-prefix` 해소값. sliding/h2o/d2o 가 사용.
- `keep_ratio` → `args.h2o_keep_ratio`. `H2OPolicy::new(p.keep_ratio, p.protected_prefix)` 와 `D2OConfig{keep_ratio: p.keep_ratio, ...}`.
- `sink_size` → `args.sink_size`.
- `streaming_window` → 0 이면 기본값 유도이나, **유도는 `make` 안이 아니라 caller(엔진)** 에서 일어난다(`args.streaming_window>0 ? args.streaming_window : ...`). stage 도달 시엔 이미 baked 된 양수값이다.

> **d2o 전용 파라미터 부재(의도적 미plumb)**: `ema_beta`/`merge_e`/`use_layer_allocation` 등은 "공용 struct 비대화 vs per-technique opaque params" 결정이 미해결이라 `StageParams` 에 싣지 않는다(`lib.rs:123-126`, ADR-0004 open question). 결과로 `D2O_STAGE.make` 는 `protected_prefix`/`keep_ratio` 만 매핑하고 나머지를 `D2OConfig::default()`(non-alloc, ema_beta=0.7, merge_e=0.1)로 채운다(`stage_registry.rs:270-276`). **따라서 레지스트리 경유 d2o 는 layer-allocation 을 못 쓴다** — production d2o 는 레지스트리가 아니라 if-branch `D2OHandler` 가 처리한다. 새 기법이 고유 파라미터를 요구하면 현재로선 `make` 안에서 default 로 채우거나, API 가 opaque params 를 받도록 확장될 때까지 대기해야 한다.

### 조회 함수

```rust
pub fn find_stage(name: &str) -> Option<&'static KVCacheStageReg> {
    KV_CACHE_STAGES.iter().find(|r| r.name == name)
}

pub fn registered_names() -> Vec<&'static str> {
    KV_CACHE_STAGES.iter().map(|r| r.name).collect()
}
```

- **`find_stage(name)`** — 이름으로 첫 매치 등록 반환. **`None` 조건**: (1) 그 이름 미등록, (2) 등록됐으나 fat-LTO/`--gc-sections` strip, (3) technique crate 가 dep 으로만 선언되고 **force-link 참조가 없어** dead-crate elision 됨(아래). 반환 참조는 `'static`. miss 시 엔진은 `anyhow!("Unknown eviction policy ...")` 로 bail.
- **`registered_names()`** — 등록 이름 전부를 매 호출 새 `Vec` 으로 수집(할당 발생). self-test/진단 전용, hot path 아님. 순서는 링크 순서 의존.

### force-link — 별도 crate 등록의 필수 1줄

**M3 실측(ADR-0003 §4 D4 정정)**: Rust 는 **미참조 의존 rlib 을 링크에서 제외**한다(dead-crate elision). 따라서 technique crate 를 Cargo dep 으로만 선언하면 그 crate 의 `#[distributed_slice]` 등록이 바이너리에 포함되지 않아 `find_stage` 가 `None` 을 반환한다(실측). 활성화하려면 designated 지점에 **force-link 참조 1줄**이 필요하다:

```rust
use my_stage as _;   // 미참조 crate 를 링크에 강제 포함 → distributed_slice 등록 활성화
```

확장 비용 총합 = **폴더 + Cargo dep 1줄 + force-link 1줄**(둘 다 기계적 설정성 라인, 기존 로직 0 edit → OCP 유지).

> **주의 — 예제 crate 는 production 미링크.** `example-keep-recent` 의 force-link `use example_keep_recent as _;` 는 현재 `stage_registry.rs:360` 의 **`#[cfg(test)] mod tests`** 안에만 있고, 엔진의 **dev-dependency** 다(`engine/Cargo.toml`, "프로덕션 미링크(dev-only)"). 따라서 production 바이너리에서 `find_stage("example_keep_recent")` 는 `None` 이 정상이며, 예제는 메커니즘 검증·기여자 템플릿 전용이다. 기여자가 production 에 기법을 노출하려면 (a) `[dependencies]` 에 dep 선언, (b) production 코드 경로(엔진 technique-link 모듈 또는 bin)에 `use <crate> as _;` 를 둬야 한다. 빌트인 3종(sliding/streaming/h2o)·d2o 는 엔진 crate **내부** 모듈(`stage_registry.rs`)에서 직접 등록하므로 cross-crate force-link 가 불필요하다(엔진 자기 자신은 항상 링크됨).

### 엔진 소비 전체 흐름 (등록된 기법이 어떻게 구동되나)

1. `find_stage(name)` → `&KVCacheStageReg`(miss 면 bail).
2. caller 가 `StageParams` 를 CLI args 로부터 baked(streaming_window 유도 포함).
3. `(reg.make)(params)` → `Box<dyn KVCacheStage>`.
4. `StageBackedPolicy::new(stage)` 로 레거시 `EvictionPolicy` 표면으로 감싸 `CacheManager` 에 주입.
5. eviction 트리거 시 `StageBackedPolicy::run` 이 `KVStageCtx` 로 immutable borrow → `stage.plan(&ctx)` → plan `Some` 이면 borrow 종료 후 `execute_kv_plan(&mut cache, &plan)` 으로 **엔진이 독점 변형**(§4).

---

## 6. 예제

### (a) `example-keep-recent` — 최소 LayerWide, score-free 템플릿

`crates/techniques/example-keep-recent/src/lib.rs`. `technique-api` 에만 의존하고 엔진 타입을 일절 참조하지 않는다. "최근 `target_len` 토큰만 유지"(sliding 의 prefix=0 변형). 새 기법의 출발 골격으로 복사한다.

```rust
use linkme::distributed_slice;
use technique_api::{
    KV_CACHE_STAGES, KVCachePlan, KVCacheStage, KVCacheStageReg, KeepSpec, StageCtx, StageParams,
};

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

#[distributed_slice(KV_CACHE_STAGES)]
static EXAMPLE_KEEP_RECENT: KVCacheStageReg = KVCacheStageReg {
    name: "example_keep_recent",
    make: |_params: StageParams| Box::new(KeepRecent),
};
```

관찰 포인트:
- **쓰는 ctx accessor 는 `current_pos`/`target_len` 둘뿐** — score-free 라 `importance`/`head_score`/`dequant_k` 미호출. `make` 가 `StageParams` 를 `_params` 로 무시(상태 없는 unit struct).
- `current <= target` 이면 `None`(no-op) — 빈 keep 을 반환하면 캐시가 비워지므로(§4) 반드시 `None` 으로 표현.
- `keep = (current - target..current).collect()` 은 range collect 라 자동 ascending(`KeepSpec::LayerWide` 불변식 충족).
- config 필드를 쓰려면 `make: |p: StageParams| Box::new(MyStage { window: p.eviction_window, ... })` 로 평탄 값을 끌어다 쓴다.

### (b) `D2OStage` — 가중 merge + EMA(impl 상태) + `dequant_k` 사용

`engine/src/pressure/d2o_handler.rs:423-490`. score-based + stateful 기법의 레퍼런스. (1) `Mutex<D2OState>` 로 EMA threshold τ 를 호출 간 누적(D4), (2) 가중 `WeightedMerge` 산출(Eq.11), (3) cosine-nearest 매칭을 위해 `ctx.dequant_k` 로 raw K 읽기.

```rust
pub struct D2OStage {
    config: D2OConfig,
    state: Mutex<D2OState>,
}

impl KVCacheStage for D2OStage {
    fn name(&self) -> &str { "d2o" }

    fn plan(&self, ctx: &dyn StageCtx) -> Option<KVCachePlan> {
        let kv_heads = ctx.n_kv_heads();
        let head_dim = ctx.head_dim();
        let importance = ctx.importance().unwrap_or(&[]);   // None → 빈 슬라이스 정규화
        let mut state = self.state.lock().unwrap();          // 누적 상태는 impl Mutex (D4)

        // D2OHandler::evict_and_merge 와 공유 — K 는 ctx.dequant_k reader 클로저로 전달.
        let (retain_all, passing, matches) = compute_d2o_plan(
            &|p, h, o| ctx.dequant_k(p, h, o),
            &self.config, &mut state,
            ctx.current_pos(), ctx.target_len(),
            importance, kv_heads, head_dim, true,
        )?;                                                  // None → no-op → plan None

        let merges: Vec<WeightedMerge> = if passing.is_empty() {
            Vec::new()
        } else {
            group_by_retain(&passing, &matches)
                .iter()
                .map(|(retain, evicted_list)| {
                    let (w_c, w_e) = compute_eq11_weights(evicted_list, self.config.merge_e);
                    WeightedMerge {
                        into: *retain,
                        into_weight: w_c,
                        from: evicted_list.iter().zip(w_e.iter())
                            .map(|(&(ep, _), &w)| (ep, w)).collect(),
                    }
                })
                .collect()
        };

        Some(KVCachePlan { keep: KeepSpec::LayerWide(retain_all), merges })
    }
}
```

본문 요지:
1. **ctx 읽기**: `n_kv_heads`(K vector 차원 `layer_dim = n_kv_heads * head_dim`), `head_dim`(dequant 버퍼 크기), `importance`(token rank — `unwrap_or(&[])` 정규화), `current_pos`, `target_len`, `dequant_k`(closure 로 전달). `layer_idx`/`head_score`/`has_head_scores` 는 미사용.
2. **상태 lock**: `self.state.lock().unwrap()` 로 `D2OState`(EMA τ + initialized + 통계)에 `&mut` 획득 — D4 실물.
3. **계획 위임**: `compute_d2o_plan(reader, config, state, current, target, importance, kv_heads, head_dim, merge_enabled=true)`. production `D2OHandler::evict_and_merge` 와 **동일 함수 공유** → bit-identical by construction. reader 인자가 핵심 — D2OStage 는 `ctx.dequant_k` 클로저, D2OHandler 는 `dequantize_k(cache, ...)` 를 넘긴다. `compute_d2o_plan` 이 `None`(`current <= keep`(=`target_len.max(prefix+2)`) 또는 evict 대상 없음)을 내면 `?` 로 plan `None` 전파.
   - 내부 단계(`d2o_handler.rs:204-305`): Step1 H2O 3-partition(prefix + heavy-hitter + recent) → Step2 evicted 마다 retain 집합에서 layer-wide K cosine 최근접 1개 argmax(reader 로 K 읽음) → Step3 EMA τ 갱신(첫 호출 = mean(sim), 이후 `β·max + (1-β)·τ`) → Step4 `sim >= τ` 인 evicted 만 passing.
4. **merges 산출**: `group_by_retain` 으로 passing 을 nearest retain 별로 묶고, 그룹마다 `compute_eq11_weights(evicted_list, merge_e)` 로 `(w_c, w_e)` 산출. Eq.11(`d2o_handler.rs:657-668`): `D = Σ exp(u_i) + e`, `w_c = e/D`, `w_i = exp(u_i)/D`, 합 = 1; `u_i = sim` 은 exp 전 `[-10, 10]` clamp.

**엔진 소비**: 반환 plan 은 `execute_kv_plan` 의 LayerWide+(non-empty merges) 경로로 들어가 `apply_weighted_merges`(pre-compact 좌표, K/V 독립 dtype 디스패치) → `compact_keep_positions` 순으로 처리된다(§4).

> **⚠️ production unwired.** `D2OStage` 는 `"d2o"` 로 `KV_CACHE_STAGES` 에 등록되어 있으나(`stage_registry.rs:267-277`), **production d2o 경로는 여전히 if-branch `D2OHandler` 가 가로챈다**. 이유: `StageBackedPolicy`/`run_policy_eviction` 경로는 per-cache uniform target 만 주므로 layer-alloc/protected-layer 를 표현하지 못한다. D2OStage 등록은 **proven-equivalent(non-alloc) available 표면**이다. 또 `D2OStage` 는 별도 crate 가 아니라 엔진 내부(`engine/src/pressure/`)에 있다 — 별도 crate 로 빼려면 `ctx.dequant_k` reader 클로저 + Eq.11 가중 로직만 가져가면 된다.

### 대비표 (복붙 선택 가이드)

| 항목 | example-keep-recent | D2OStage |
|---|---|---|
| keep 모양 | `KeepSpec::LayerWide` | `KeepSpec::LayerWide` |
| 상태 | 없음(unit struct) | `Mutex<D2OState>`(EMA τ, D4) |
| score | score-free(`importance` 미사용) | score-based(`importance` 사용) |
| raw K | 미사용 | `ctx.dequant_k`(cosine-nearest) |
| merges | 빈 Vec | Eq.11 가중 `WeightedMerge` |
| `None` 조건 | `current <= target` | `compute_d2o_plan` 이 `None`(`?` 전파) |
| make config | `StageParams` 무시 | `protected_prefix`/`keep_ratio` 매핑 |
| 복붙 추천 | score-free / merge-free 신규 기법 | 가중 merge·stateful 기법 |

---

## 7. 구현 불변식 체크리스트

기법 작성 시 다음을 모두 만족하는지 확인하라:

- [ ] `KVCacheStage` 의 `name()` 이 `KV_CACHE_STAGES` 내에서 **유일**하다(중복은 silent shadow, 강제 없음).
- [ ] `name()` 반환값과 `KVCacheStageReg.name` 이 동일 문자열이다(컨벤션).
- [ ] 누적 상태는 `&self` + `Mutex`/`RwLock`/`Atomic`(`Cell`/`RefCell` 불가 — `Sync` 위반).
- [ ] `plan` 이 `ctx` 의 dyn-safe accessor 만 사용한다(엔진 타입 직접 참조 0).
- [ ] **축소 불필요는 `None` 으로 반환**한다(빈 `KeepSpec::LayerWide(vec![])` 는 캐시를 비운다 — no-op 아님).
- [ ] `KeepSpec::LayerWide` 의 keep Vec 이 **ascending** 정렬이다(위반 시 silent garbage).
- [ ] 보존하려는 prefix(BOS 등)를 keep Vec 에 **명시적으로** 포함시켰다.
- [ ] `KeepSpec::PerHead` 를 **반환하지 않는다**(executor 미구현 — `bail!` Err). 현 production-ready 는 LayerWide 한정.
- [ ] `head_score`/`has_head_scores`/`layer_idx` 에 의존하지 않는다(미plumb 상수: 0.0/false/0).
- [ ] `WeightedMerge` 의 위치(`into`/`from`)가 keep 과 **동일 pre-compact 좌표계**다.
- [ ] `WeightedMerge` 의 `Σ from.1 + into_weight ≈ 1`(강제 없음, 위반 시 norm 왜곡).
- [ ] `from` ∩ retained == ∅, merge 간 비중첩(in-place 안전성, 강제 없음).
- [ ] `dequant_k(pos, head, out)` 호출 시 `out.len() == head_dim`(짧으면 panic 가능).
- [ ] `plan` 내부에서 직접 panic 하지 않는다(forward 경로).
- [ ] (별도 crate) Cargo dep 1줄 + designated 지점 **force-link `use <crate> as _;` 1줄** 추가(dead-crate elision 방어).

---

## 8. 참조

- [ADR-0003](adr/0003-extension-mechanism-static-crates.md) — 확장 메커니즘(정적 link crate + linkme, `.so` 보류). §4 D4 에 force-link 정정(M3 실측).
- [ADR-0004](adr/0004-kvcachestage-plan-returning-trait.md) — `KVCacheStage` plan-returning trait 설계(D1~D6, executor 매핑, Q4_0 merge 정정).
- [docs/50_adding_kvcache_stage.md](50_adding_kvcache_stage.md) — 단계별 추가 절차(폴더·Cargo·게이트) how-to. 본 문서는 그 API 보완본.
- `crates/technique-api/src/lib.rs` — trait/타입 정본(`KVCacheStage`/`StageCtx`/`KVCachePlan`/`KeepSpec`/`WeightedMerge`/`StageParams`/`KVCacheStageReg`/`KV_CACHE_STAGES`/`find_stage`/`registered_names`).
- `engine/src/pressure/eviction/stage_registry.rs` — 엔진 측 `StageCtx` impl(`KVStageCtx`), executor(`execute_kv_plan`), 역어댑터(`StageBackedPolicy`), `EvictionPolicyAsStage`, 빌트인 등록, `ensure_builtin_stages_registered`.
- `engine/src/pressure/standard_format.rs` — `apply_weighted_merges`(가중 merge 산술, F32/F16/Q4_0).
- `engine/src/pressure/d2o_handler.rs` — `D2OStage`, `compute_d2o_plan`, `compute_eq11_weights`, `dequantize_k`(`StageCtx::dequant_k` 정본).
- `crates/techniques/example-keep-recent/src/lib.rs` — 최소 기여자 템플릿.
- `CONTEXT.md` — 도메인 용어(3축 직교: stage ⊥ format ⊥ hardware).
