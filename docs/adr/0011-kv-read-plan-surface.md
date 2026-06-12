# ADR-0011: KV read-plan 표면 — `KVReadStage` plan-returning trait ("무엇을 읽을지"의 4번째 plugin 표면)

> **Status**: Proposed (구현 보류 — 리팩토링 머지 후 착수)
> **Date**: 2026-06-12
> **Decision-makers**: 사용자 + Architect (KV 로드맵 항목 4, `.agent/todos/sprint_kv_roadmap_item34_2026_06_12.md` §P4)
> **Selected**: `attention_into`의 "캐시 전체 읽기" 가정을 깨는 **read-plan 표면**을 신설한다. KV 의 plan-returning plugin 패턴(ADR-0004)을 *읽기 결정*에 거울 복제한 `KVReadStage::read_plan(ctx) -> Option<KVReadPlan>`(`granularity` + `select`). format 은 `attention_into_selected` **capability opt-in**(미지원 format 은 full read 폴백 → 합성성 보존). page 메타데이터(K min/max)는 read stage 가 `tensor(Key)`/`tensor(QueryStats)`로 자기 상태 incremental 갱신(코어 무수정). 표면 하나로 Quest 류 선택 읽기 + InfiniGen/KVSwap 류 prefetch 두 군집(~9건)을 동시 해소한다.
> **★범위 제약 (사용자 결정 2026-06-12, U2)**: 본 ADR 은 **설계 고정까지 — 구현 코드 0줄**. ADR 확정 = "KV 구조 확정" 게이트 → 직후 사용자가 별도 워크트리에서 대형 리팩토링을 병렬 시작한다. ADR-선행의 목적 = 미래 표면을 *문서 제약*으로 고정해 병렬 리팩토링과의 의미적 충돌을 예방하는 것. 구현 단계 분해는 backlog 재등록으로만 산출(§9).
> **Related**: ADR-0003(정적 crate + linkme + force-link), ADR-0004(`KVCacheStage` plan-returning, **D1 변형=엔진 독점** / §7 M-A TensorHandle / §10 M-Q QueryStats), ADR-0005(D5 generic floor + 특화 opt-in / D6 3축 평행 registry / §8 INV-HOTPATH-DISPATCH), ADR-0006(weight 축 거울 복제 = 세 번째 plan-returning 형제), `/CONTEXT.md`(stage=verb ⊥ format=noun, 3축 직교), `engine/src/format/kv_cache_format.rs`(`attention_into` 시그니처), `engine/src/kv/eviction/stage_registry.rs`(`KVStageCtx`/`execute_kv_plan`), `spec/41-invariants.md` §3.28(INV-DECODE-STAGE / INV-HOTPATH-DISPATCH)

---

## 1. Context

3축 plugin 골격(stage ⊥ format ⊥ backend)에서 **stage 축은 두 결정 형제**를 이미 가진다 — KV 의 `KVCacheStage`(ADR-0004, "무엇을 *상주시킬지*" = keep/merge)와 weight 의 `WeightStage`(ADR-0006, "어느 layer 를 어느 precision/dispatch 로"). 둘 다 동일 문법(plan-returning: plugin 이 *결정*만, 엔진이 *변형*을 독점)을 따른다.

그러나 **"무엇을 *읽을지*"는 plugin 표면이 없다.** 현 읽기 경로(`KVCacheFormat::attention_into(q, backend, out, dims, scores)`, `format/kv_cache_format.rs:94`)는 **캐시 전체를 읽는다는 것을 암묵 전제**하고 selection 인자가 없다. 이 한 가정이 2024–2026 연구의 두 군집을 통째로 차단한다:

- **선택적 읽기 (Quest 류)**: query 와 무관한 KV page 를 attention 에서 건너뛰면 decode 의 attention FLOPs/메모리 대역폭을 줄인다 — Quest(ICML'24, arXiv 2406.10774)는 page 별 K min/max 로 query 와의 상한 내적을 추정해 top-k page 만 읽는다. InfLLM·HISA·BLASST·shadowAttn(2508.16703, 모바일 NPU prefill 전용)이 같은 부류. 현 표면은 "전부 읽기"라 *읽기 범위 축소 결정을 표현할 어휘가 없다*.
- **예측 prefetch (InfiniGen/KVSwap 류)**: layer i 의 attention 이 *어느 KV 를 읽을지*를 layer i−1 에서 미리 알면, 디스크/호스트로 offload 된 KV 를 layer i 도착 전에 prefetch 해 stall 을 숨긴다 — InfiniGen(OSDI'24), KVSwap(2511.11907) prefetch. 현 표면은 "읽기 직전에 통째로 읽기"라 *예측 채널이 없다*.

**핵심 통찰** (backlog L913·L915): 이 두 군집은 **동일 산출물**을 요구한다 — *layer i 에서 읽을 KV 토큰/page 의 집합*. Quest 에겐 그것이 **읽기 마스크**(이번 layer 에서 건너뛸 것), KVSwap 에겐 그것이 **다음 layer 의 prefetch 목록**(미리 끌어올 것)이다. layer i−1 의 read plan 을 layer i 의 prefetch 목록으로 *재해석*하면 표면 하나가 두 use case 를 동시에 덮는다.

**왜 format 축에 매장하면 안 되는가** (backlog L916): "Quest 를 KIVI-format 안에 박는다"는 즉시 M×N 재발이다 — Quest-on-KIVI / Quest-on-Standard / Quest-on-opaque 마다 신규 format 이 생긴다(selection 로직 × N format = 폭발). selection 은 *format 과 직교한 결정*(어느 표현이든 "이 page 들만 읽어라"는 같은 의미)이므로, **표면 신설이 정공**이고 매장은 직교 위반이다. ADR-0003 가 없앤 closed match-arm 재도입과 같은 안티패턴.

**충돌 불변·제약**:
- **3축 직교**(CONTEXT.md): stage ⊥ format ⊥ hardware. read-plan 은 stage 축 신표면(verb — "어떻게 읽을지 조절") ⊥ format(noun). 새 표면이 format 을 알면 안 된다.
- **ADR-0004 D1**: 결정은 plugin, 변형(여기선 *읽기 실행*)은 엔진. read stage 는 plan 만 반환하고 attention 커널 dispatch 는 엔진/format 이 한다.
- **ADR-0005 §8 INV-HOTPATH-DISPATCH**: read-plan 은 **layer-tier**(N×/token)에서 *소비*된다 — attention 이 매 layer 마다 일어남. 여기가 가장 뜨거운 경계라 `.so`/`dyn` 비용에 극도로 민감하다(KVCacheStage/WeightStage 는 step-tier 라 이 압박이 약했다).
- **ADR-0005 D5 generic floor**: 미지원 format 은 full read 폴백(exact, 느릴 뿐) — capability opt-in 과 동형.

---

## 2. Decision

### D1 — `KVReadStage` = plan-returning trait, read stage 의 4번째 평행 registry. (ADR-0004/0006 거울)

KV/weight 의 plan-returning 관용구를 *읽기 결정*에 거울 복제한다:
```rust
// crates/technique-api/src/lib.rs (개념 — 구현은 리팩토링 머지 후)
pub trait KVReadStage: Send + Sync {
    fn name(&self) -> &str;
    /// layer i 진입 전, layer i−1 의 결과(scores/QueryStats)를 기반으로 "layer i 에서 읽을 KV"를 결정.
    /// None = 이번 layer 는 full read(=현행 동작). plugin 은 자기 상태(page 메타)를 ctx 로 incremental 갱신.
    fn read_plan(&self, ctx: &dyn ReadStageCtx) -> Option<KVReadPlan>;
}
pub struct KVReadStageReg { pub name: &'static str, pub make: fn(ReadStageParams) -> Box<dyn KVReadStage> }
#[distributed_slice] pub static KV_READ_STAGES: [KVReadStageReg] = [..];
pub fn find_read_stage(name: &str) -> Option<&'static KVReadStageReg>;
```
- `KV_READ_STAGES` 는 stage 축의 **4번째 평행 linkme registry**(`KV_CACHE_STAGES`/`WEIGHT_STAGES` 형제) — ADR-0005 D6 의 *확장*(병합 아님, §3.3.1 무위배). 외부 read technique crate 는 ADR-0003 §4 force-link(`use crate as _`) + startup self-test(`ensure_builtin_read_stages_registered` 류) 대상.
- **빌트인 0개로 시작 가능** — read stage 부재 시 엔진은 항상 full read(현행 100% 보존, D5). 첫 빌트인은 Quest(host 테스트 → CLI opt-in, ADR-0004 §8 CAOTE 배선 패턴과 동형).

### D2 — `KVReadPlan` = `{ granularity, select }`. select 는 ascending 위치/page id, granularity 가 단위를 정함.

```rust
pub enum ReadGranularity { Token, Page { page_size: u32 } }   // #[repr(u32)] — layer-tier 경계 POD 후보
pub struct KVReadPlan {
    pub granularity: ReadGranularity,
    pub select: Vec<usize>,   // ascending. Token 단위면 토큰 pos, Page 단위면 page index.
}
```
- **`granularity` 의미**: `Token` = `select` 가 KV 토큰 위치(pos)의 부분집합(fine-grained, Quest 의 token-level 변형). `Page { page_size }` = `select` 가 page index 의 부분집합이고 각 page 가 `page_size` 토큰을 묶음(coarse, Quest 정본·InfiniGen prefetch 단위). page 가 기본 단위인 이유: (a) 연구 주류(Quest/InfiniGen)가 page 단위라 메타데이터(K min/max) 회계가 page 당 O(1), (b) prefetch 는 contiguous chunk 가 DMA 효율적, (c) per-token mask 는 attention 커널 분기 비용이 큼. `Token` 은 fine 변형 여지로 enum 에 포함(타입 breaking 0).
- **`select` 표현 = ascending `Vec<usize>`** — `KeepSpec::LayerWide`(ADR-0004 D2)와 동형. bitmask 가 아니라 explicit index 리스트인 이유: sparse selection(Quest 가 1024 page 중 32 page 선택)에서 메모리 효율적이고, prefetch 목록으로 그대로 재사용 가능하며, 엔진 executor 가 ascending 가정으로 검증·실행 단순화(ADR-0004 executor 와 같은 계약).
- **`new_pos` 없음** — read plan 은 캐시를 *변형하지 않는다*(eviction 과 결정적 차이). 캐시 상태는 그대로, 이번 layer 의 attention *읽기 범위*만 좁힌다. 따라서 position 재배열·current_pos 변경이 없다(§6 직교성 근거).

### D3 — 발화 시점 = **layer i 진입 전**, 입력 = layer i−1 결과. plan 의 이중 해석(마스크 ⊥ prefetch)은 *소비자*가 정한다.

read stage 는 layer 경계에서 발화한다 — layer i 의 attention 직전에 `read_plan(ctx)`을 호출해 "layer i 에서 읽을 것"을 받는다. 이 plan 은 **두 소비자가 다르게 쓴다**:
- **읽기 마스크 소비자 (Quest)**: layer i 의 `attention_into_selected(q, plan.select, ...)`에 직접 전달 → 그 layer 의 attention 이 select 된 KV 만 읽음.
- **prefetch 소비자 (KVSwap)**: layer i 의 plan 을 layer i 도착 *전*에 미리 산출(layer i−1 결과로) → offload backend 가 select 된 KV 를 미리 끌어옴. 즉 같은 plan 이 "이번에 읽을 것"이자 "다음에 끌어올 것".
- **엔진은 plan 의 의미를 모른다** — 엔진은 plan 을 (a) 활성 format 이 `attention_into_selected` 를 지원하면 거기로 전달, (b) offload format 이면 prefetch 큐로 전달, (c) 둘 다 아니면 full read 폴백. 어느 해석인지는 *format 의 capability* 가 정하지 plan 자체에 라벨이 없다(소비자-주도 — backlog L915 통찰의 타입 구현).

### D4 — format `attention_into_selected` = **capability opt-in**. 미지원 = full read 폴백 (ADR-0005 D5 거울).

`KVCacheFormat` base trait 에 selection 을 강제 추가하지 *않는다*(6-method 불변). 대신 **선택 capability**를 opt-in 으로:
```rust
// 개념 — capability-handle 패턴(INV-STAGE-LAYER-HANDLE 3-form 중 capability-handle)
pub trait SelectiveRead {
    fn attention_into_selected(
        &self, q: &Tensor, backend: &dyn Backend, out: &mut Tensor,
        dims: AttnDims, select: &[usize], granularity: ReadGranularity,
        scores: Option<&mut [f32]>,
    ) -> Result<()>;
}
```
- **미지원 format 폴백** = `read_plan` 이 plan 을 반환해도 활성 format 이 `SelectiveRead` 를 구현 안 하면 엔진이 plan 을 **무시하고 `attention_into`(full read)** 를 호출한다 → 출력 정확성 보존(plan 은 *근사 가속*이지 *정확성 계약*이 아님 — Quest 가 page 를 빠뜨려도 "틀린 답"이 아니라 "근사 답"). 이것이 ADR-0005 D5 의 "floor 는 exact, 특화는 opt-in"과 동형이고 합성성(stage × format 임의 조합)을 보존한다.
- **왜 base trait 에 안 넣나**: `attention_into` 에 `select: Option<&[usize]>` 를 추가하면 모든 format(Standard/KIVI/opaque)이 selection 분기를 구현해야 하고, selection 을 안 쓰는 format 도 인자를 받아야 한다(ISP 위반). capability 분리는 selection-aware format 만 비용을 지불한다. INV-STAGE-LAYER-HANDLE 의 capability-handle form 정합.

### D5 — page 메타데이터(K min/max) 유지 주체 = **read stage 자신** (코어 무수정). `tensor(Key)`/`tensor(QueryStats)`로 incremental.

Quest 의 page 선택은 page 별 K 의 min/max(채널별)를 유지해 query 와의 상한 내적을 추정해야 한다. 이 page 메타를 **누가 유지하는가**가 핵심 설계 쟁점이다. 채택안:
- **read stage 가 자기 `&self` 상태(Mutex)에 page 메타를 보유**하고, `read_plan(ctx)` 호출마다 ctx 의 `tensor(Key)`(ADR-0004 §7 M-A)로 새로 쓰인 KV 를 읽어 incremental 갱신한다(D2O EMA 가 impl Mutex 에 상태를 두는 ADR-0004 D4 와 동형). 코어(`KVCache`)는 page 메타를 *모른다* — 무수정.
- **항목 3 시너지**: Quest 의 query-aware 변형(Expected Attention 결합)은 `tensor(QueryStats)`(ADR-0004 §10 M-Q, per-(layer,kv_head) Q running mean/var)로 query 분포까지 읽어 page 선택 신호를 강화한다. QueryStats 표면은 **무수정 재사용** — read stage 가 소비자로 추가될 뿐(MEMORY 항목 3 산출물이 항목 4 의 신호 공급원).
- **대안(코어가 page 메타 유지)은 §5 에서 기각** — 코어가 Quest-전용 상태를 갖는 것은 plugin 화 정신 위배(ADR-0003 가 없앤 closed-knowledge 재도입).

### D6 — 네이밍·도메인. `KVReadStage` = stage 축(verb) 신표면. read plan ⊥ cache plan ⊥ dispatch plan.

`KVCacheFormat`(저장 noun) 의 형제가 아니라 `KVCacheStage`(상주 토큰 조절 verb)·`WeightStage`(weight 조절 verb) 의 **형제**다 — read-plan 은 "어떻게 읽을지 *조절*"하는 stage 축 동작. CONTEXT.md 의 stage(verb) ⊥ format(noun) 분류 정합. 부속: `KVReadPlan`/`KVReadStageReg`/`KV_READ_STAGES`/`ReadStageParams`/`ReadStageCtx`. `KVReadPlan`(읽기 범위) ⊥ `KVCachePlan`(keep/merge) ⊥ `WeightDispatchPlan`(layer dispatch) — 세 plan 은 직교 산출물(같은 step 에서 다 발화 가능, §6).

---

## 3. Rationale

- **plan-returning 이 네 번째 표면에서도 정합** — read 결정도 "plugin 이 indices 만 방출, 엔진이 실행" 형태라 ADR-0004 D1(변형=엔진 독점)의 정신을 *읽기 실행*에 그대로 적용한다. 논문 기여: **4번째 표면이 동일 plan-returning 문법으로 가산**됨을 실증(확장성 = 새 stage 축 동작이 같은 골격에 무비용 추가). 이것이 backlog L919 의 "논문 기여" 항목.
- **soft constraint 가 합성성의 열쇠** — read plan 은 *정확성 계약이 아니라 근사 가속 힌트*다(D4). 그래서 미지원 format 폴백이 silent-wrong 이 아니라 단지 "가속 안 됨". eviction plan(틀리면 캐시 손상)과 결정적으로 다른 안전 등급이고, 이 차이가 capability opt-in 을 안전하게 만든다.
- **page 메타를 stage 가 들면 코어가 Quest 를 모른다** — ADR-0004 §7 TensorHandle 이 이미 K/Q 읽기 어휘를 깔아놨으므로(D5), read stage 는 그 위에서 자기 상태를 incremental 갱신만 한다. 코어 KVCache 는 page 개념이 없고 영원히 모른다 → plugin 화 정신 충실.
- **마스크 ⊥ prefetch 의 통일이 표면 수를 절약** — 두 use case 에 표면 2개(read-mask trait + prefetch trait)를 만드는 대신 1개로 수렴(D3). 이는 우연이 아니라 둘이 "layer i 의 읽기 집합"이라는 같은 정보를 다르게 *소비*할 뿐이라는 도메인 사실의 반영.

## 4. Consequences

- **technique-api**: `KVReadStage`/`KVReadPlan`/`ReadGranularity`/`KVReadStageReg`/`KV_READ_STAGES`/`find_read_stage`/`ReadStageCtx`(또는 기존 `StageCtx` 재사용) 신설. `ReadGranularity` 는 layer-tier 경계 후보라 `#[repr(u32)]`(ADR-0005 D2/L1).
- **엔진**: (a) layer 경계에서 `read_plan` 호출 + plan 을 활성 format 의 `SelectiveRead` 또는 offload prefetch 큐로 라우팅하는 executor seam(layer-tier — INV-HOTPATH-DISPATCH 정합 설계 필수, §7 리스크), (b) `KVStageCtx`(또는 신규 `ReadStageCtx` 어댑터)가 read stage 에 `tensor(Key)`/`tensor(QueryStats)` 공급, (c) `KV_READ_STAGES` startup self-test.
- **format**: `SelectiveRead` capability 를 selection-aware format(첫 대상 = Standard, 후에 offload)이 opt-in 구현. 미지원은 full read 폴백(자동, 코드 0).
- **offload**: 기존 `OffloadKVCache`(spec/01 L152, SeqMajor 전용 prefetch 파이프라인)와 정합 — read plan 이 그 prefetch depth(`--max-prefetch-depth`, spec/33 L560)의 *목록 공급원*이 된다(현재는 순차 depth, read plan 이 *예측 목록*으로 대체/보강).
- **spec**: §Spec Triage 결론 = spec 무관(arch-only). 신규 INV 불요 — read-plan 소비는 INV-HOTPATH-DISPATCH(layer-tier dispatch 규율)의 *적용 대상*이고 INV-DECODE-STAGE 시리즈(stage 발화 phase)의 read 변형이나, 구현 시점(리팩토링 머지 후)에 arch 갱신으로 흡수(§Spec Triage).
- **검증**: read plan 활성 시 출력이 full read 와 *근사 일치*(정확 일치 아님 — 근사 가속) + plan 미지원 format 에서 full read 와 *bit-identical*(폴백 정확성) + α-K frozen happy path(read stage 부재 시) byte-identical.

## 5. Alternatives Considered

- **(status quo) `attention_into` full read 유지, read-plan 표면 없음** — **기각**: Quest/InfiniGen/KVSwap(~9건)이 통째로 차단. decode attention 의 메모리 대역폭 절감(8B/long-ctx 핵심)·디스크 offload prefetch(KVSwap)가 원천 불가. 3축 골격의 "확장은 가산"이 *읽기 결정*에서만 깨진 비대칭.
- **(A) Quest 를 전용 format 에 매장**(`QuestKIVIFormat` 등) — **기각**: format 축 안에서 M×N 재발(Quest × N format = 신규 format 폭발, §1). selection 이 format 과 직교한데 매장하면 직교 위반 + ADR-0003 가 없앤 closed-knowledge 재도입. backlog L916 의 "표면 신설이 정공" 근거.
- **(B) read-mask trait ⊥ prefetch trait 2표면 분리** — **기각**: 둘이 "layer i 읽기 집합"이라는 동일 정보를 다르게 소비할 뿐(§3). 2표면은 표면 수 낭비 + read stage 가 마스크용·prefetch용 둘 다 구현해야 하는 중복. D3 의 소비자-주도 단일 plan 이 더 단순.
- **(C) `KVReadPlan` 을 정확성 계약으로**(format 이 반드시 select 만 읽도록 강제, 폴백 금지) — **기각**: capability opt-in 합성성 파괴(미지원 format = 컴파일/런타임 에러 → stage × format 임의 조합 불가). read 가 *근사 가속*이라는 본질(D4) 무시. ADR-0005 D5 의 "floor 는 exact" 정신 위배.
- **(D) page 메타(K min/max)를 코어 `KVCache` 가 유지** — **기각**: 코어가 Quest-전용 상태를 가짐 = plugin 화 정신 위배(코어가 특정 기법을 앎). ADR-0004 §7 TensorHandle 이 이미 K 읽기 어휘를 깔아 read stage 가 자기 상태로 incremental 가능(D5). 코어는 page 를 영원히 모르는 게 옳다.
- **(E) read plan 을 step-tier 에서 1회 산출**(layer 마다 호출 안 함, decode step 시작에 전 layer plan 일괄) — **기각**: layer 마다 query 가 같다는 가정 필요(decode 는 layer 마다 hidden state 가 다름 → query 가 다름). Quest 의 layer-별 page 선택을 표현 못 함. **단** layer-tier 호출의 hot-path 비용이 §7 최대 리스크 — D3 는 layer-tier 호출을 받되 INV-HOTPATH-DISPATCH 정합 설계(정적 dispatch / read stage 부재 시 분기 1회)로 비용을 통제(이 ADR 은 *표면*을 고정하고 hot-path 구현 전략은 구현 단계로 위임).

## 6. read plan ⊥ cache plan (KVCacheStage) 직교성 — 같은 step 동시 발화 규칙

같은 decode step 에서 read plan(이번 layer 읽기 범위)·cache plan(eviction/merge)·weight plan(dispatch) 셋이 다 발화할 수 있다. 정합 규칙:

- **read plan 은 캐시를 변형하지 않는다**(D2 `new_pos` 없음) — 따라서 read plan 과 cache plan 은 *서로의 입력을 안 망친다*. read 는 "이번 layer 에 어느 KV 를 attention 에 넣을지"(읽기만), evict 는 "어느 KV 를 캐시에서 영구 제거할지"(변형). read 가 select 한 토큰을 evict 가 같은 step 에 지우는 race 는?
- **순서 규칙: read plan(layer i attention) → cache plan(eviction)** — eviction 은 step-tier(forward 후·capacity 도달 시), read 는 layer-tier(attention 직전). 한 step 안에서 read 는 *현재 캐시 상태*를 보고 select 하고, eviction 은 *그 step 의 forward 가 끝난 뒤* 발화한다(INV-DECODE-STAGE-005 의 `PreEviction`/`PostEviction` phase = forward 후). 즉 read 가 가리킨 토큰이 같은 step 에 evict 돼도 그 토큰은 *이미 이번 layer attention 에 쓰인 뒤* 제거된다 → race 없음(read 가 stale select 를 쓰지 않음).
- **read stage 의 page 메타 ↔ eviction 의 위치 재배열 정합**: eviction 이 토큰을 compact 하면(위치 앞당김) read stage 의 page 메타가 stale 해진다. 규칙 = **read stage 가 자기 메타를 eviction 후 재구축**(다음 `read_plan` 호출에서 `tensor(Key)` 로 현재 캐시를 incremental 재반영) 또는 eviction 발화를 신호로 메타 reset(`AttentionScoreAccumulator::reset` 패턴, INV §606 "eviction 후 score reset" 정합). 코어는 read stage 의 메타를 모르므로 *재구축 책임은 read stage 자신*(D5). 이 규칙은 구현 단계에서 page-id 안정성(eviction 이 page 경계를 깨는가)을 확정해야 하는 미해결 디테일이고(§7 R4), ADR 은 "read stage 가 eviction 후 메타 재구축 책임"이라는 *제약*만 고정한다.

## 7. Risks / Landmines

> RPN = Severity(1–10) × Occurrence(1–10) × Detection(1–10, 높을수록 *탐지 어려움*). RPN ≥ 100 = 완화책 필수.

| ID | 리스크 | S | O | D | RPN | 완화 |
|----|--------|---|---|---|-----|------|
| **R1** | **layer-tier hot-path 회귀** — read_plan 호출 + plan 라우팅이 attention(N×/token, 가장 뜨거운 경계)에 비용을 더함. ADR-0005 §8 L2(fat-LTO 가 crate 단계에서 cross-crate call 인라인 → `.so` 때 회귀 은닉)가 read 경로에서 가장 위험(KVCacheStage 는 step-tier 라 약했던 압박이 여기선 직격) | 9 | 6 | 7 | **378** | read stage **부재 시 분기 1회**(INV-147 hook=None 정신 — `Option<&dyn KVReadStage>::is_none()` 1회로 full read 직행, plan path 무비용). 활성 시에도 page-granularity 가 호출 빈도를 낮춤(layer 당 1회, page 당 아님). 구현 단계 게이트 = α-K frozen byte-identical(read stage 부재) + TBT Δ≤+3%(ADR-0005 §8 게이트). hot-path dispatch 전략(정적 vs dyn)은 구현 ADR-amendment 로 별도 고정 — *본 ADR 은 표면만, dispatch 메커니즘은 구현 시점 결정*. |
| **R2** | **항목 4 구현 유혹** — ADR 작성 중 "바로 구현하면 빠르다"로 U2 위반 → 병렬 리팩토링과 의미적 충돌 | 8 | 5 | 3 | **120** | §9 backlog 재등록에 "리팩토링 머지 후 착수" 명시. 본 ADR 산출물 = 문서 0줄 코드. ADR-선행 목적(미래 표면을 문서 제약으로 고정)을 §Status·§9 에 재강조. 자기 점검(§11)에 "코드 0줄 확인". |
| **R3** | **page-id 안정성 미정** — eviction 이 page 경계를 깨면(부분 page 제거) read stage page 메타가 좌표를 잃음 | 6 | 5 | 6 | 180 | §6 규칙 = read stage 가 eviction 후 메타 재구축(코어 무수정). 구현 단계에서 "eviction 이 page-aligned 인가 / read stage 가 매 호출 재계산하는가"를 확정(본 ADR 은 *재구축 책임이 read stage* 라는 제약만 고정). page-id 를 절대 위치 아닌 *content-derived* 로 두면 compaction 무관(구현 옵션). |
| **R4** | **근사 가속이 정확성 회귀로 오인** — Quest 가 중요 page 를 빠뜨려 품질 저하 시 "버그"로 추적될 위험(soft constraint 인데 hard 로 오해) | 5 | 5 | 5 | 125 | D4 명문화(read plan = 근사 힌트, 정확성 계약 아님). 검증 = 폴백 경로 bit-identical(정확성은 full read 가 보증) + read 활성은 *근사 일치 + 품질 메트릭*(PPL/EMR)로 별도 평가. CLI opt-in(기본 off)이라 happy path 무영향. |
| **R5** | **표면이 미래 read 기법을 못 담음** — granularity 2종(Token/Page)이 hierarchical/그래프 기반 selection 미수용 | 4 | 4 | 6 | 96 | `ReadGranularity` 가 enum(가산적 — 새 variant 추가 타입 breaking 0, ADR-0004 KeepSpec 동형). `select: Vec<usize>` 가 충분히 일반적(token/page id 둘 다 표현). 트립와이어 = 첫 hierarchical 기법 등장 시 amendment(§7 deferred 트립와이어 선례). |

## 8. Premortem — "이 설계가 1년 뒤 실패해 있다면 왜인가"

가상 부검(2027-06). 실패 시나리오 후보:

1. **"layer-tier 비용이 절감을 먹었다"** — read_plan 라우팅 오버헤드(layer 당 plan 조회 + select 검증 + capability 분기)가 1B/2048 의 selection 절감(이미 memory-bound 라 작음)을 초과해, read stage 활성이 full read 보다 *느렸다*. → 완화: 실익 정직 평가(§10)가 이미 "1B 는 제한적"이라 못박음. 가치는 8B/long-ctx/offload 이고 1B 은 토대일 뿐. 1B 에서 느린 것은 실패가 아니라 *예측된 무이득*. 실패 조건은 "8B/offload 에서도 못 이김"이고 그건 구현 측정으로만 확인(ADR 범위 밖).
2. **"capability opt-in 폴백이 silent 품질 절벽"** — 사용자가 `--read-stage quest` 를 켰는데 활성 format 이 `SelectiveRead` 미지원이라 조용히 full read 로 폴백 → "Quest 켰는데 왜 안 빨라지지?" 혼란. → 완화: 폴백 시 stderr 1회 경고(`--secondary-gguf` deprecation 경고 패턴, CLAUDE.md) + `find_read_stage` self-test 가 capability 매칭 보고. 정확성은 안전(폴백 = 정답), UX 만 개선 대상.
3. **"리팩토링이 attention_into 시그니처를 바꿔 capability 가 안 맞았다"** — 병렬 리팩토링이 `KVCacheFormat` trait 을 재설계해 `SelectiveRead` capability 의 전제(`attention_into` 가 q→out)가 깨짐. → **이것이 ADR-선행의 정확한 목표** — 본 ADR 이 "read 결정은 plan-returning + capability opt-in + 폴백" 제약을 *문서로 고정*해 리팩토링이 이 의미를 보존하도록 강제. 리팩토링이 시그니처를 바꿔도 "read plan → capability → 폴백" 의미 계약은 ADR 이 지킨다(구현 디테일만 이동).
4. **"page 메타 stale 로 garbage 출력"** — eviction 후 read stage 메타 재구축을 빠뜨려 stale page 를 select → 엉뚱한 KV attention. → 완화: §6 재구축 규칙 + D4 폴백(메타가 틀려도 full read 로 안전 강하 가능하게 설계). 단 이건 구현 버그 영역이고 ADR 은 "재구축 책임 read stage" 제약만 고정.

**부검 결론**: 가장 그럴듯한 실패 = #1(1B 무이득을 실패로 오인). 이는 §10 실익 정직 평가가 선제 방어한다 — ADR 단계에서 "1B 무이득은 예측됨, 가치는 8B/offload/논문"을 못박아 *기대 관리*로 차단.

## 9. Devil's Advocate — 설계 자신에 대한 최강 반론

**반론: "read-plan 표면은 YAGNI 다. 1B 타겟에서 Quest/InfiniGen 이 안 먹히고(memory-bound), 8B 는 아직 온보딩 안 됐고, KVSwap 은 1B/2048 에서 실익 미미(MEMORY 명시)다. 즉 *지금 소비자가 0*인 표면을 미리 만드는 것 = ADR-0003 가 경계한 추측성 추상화 아닌가?"**

**응답**:
1. **이 ADR 은 표면을 *만들지 않는다 — 고정만 한다*.** U2 제약상 구현 0줄. 따라서 "지금 추상화를 짓는" 비용은 0이고, 산출물은 *미래 제약 문서*다. YAGNI 의 대상(짓는 비용)이 발생하지 않는다.
2. **ADR-선행의 트리거가 정당하다** — 사용자가 *대형 리팩토링을 병렬 시작*한다(스프린트 게이트). 리팩토링이 `attention_into` 주변을 재설계할 때 read-plan 의 미래 의미(plan-returning + capability + 폴백)를 모르면, 나중에 read-plan 을 끼울 때 리팩토링 결과와 충돌한다. ADR 이 *지금* 의미를 고정하는 것은 추측성 코드가 아니라 *조정 비용 선지불*(병렬 작업의 머지 충돌 예방).
3. **그래도 "표면 모양이 틀릴" 위험은 남는다** — 소비자 0 상태에서 고정한 `KVReadPlan{granularity, select}` 가 실제 Quest 구현 때 안 맞을 수 있다. → 이에 대한 방어: (a) 표면을 *최소*로(plan 2필드, capability 1메서드) 잡아 틀릴 표면적을 줄임, (b) 모든 선택을 enum/Vec 가산적으로(타입 breaking 0) 두어 amendment 로 수정 가능, (c) 실제 Quest/InfiniGen 논문의 산출물(page top-k list)이 정확히 `{granularity:Page, select:Vec}` 라 frontier 정렬됨. *완전 무위험은 아니나*, 고정 안 하는 비용(병렬 충돌)이 고정하는 위험(amendment)보다 크다는 판단.
4. **소비자 0 이 영구가 아니다** — 8B 온보딩이 로드맵에 있고(MEMORY·backlog), 그때 read-plan 이 *이미 합의된 표면*이면 즉시 착수 가능. ADR-0004/0006 도 일부 소비자가 deferred 인 채 표면을 먼저 고정한 선례(WeightStage = AB-6 미배선, CAOTE = argus-chat 종속).

**잔여 인정**: 반론의 핵심(표면 모양 불확실성)은 완전히 0 이 되지 않는다. 본 ADR 은 이를 §7 R5(트립와이어) + 가산적 타입 설계로 *관리*할 뿐 *제거*하지 못한다. 이것이 "Proposed(구현 보류)" 상태로 두고 첫 실제 구현 시 amendment 가능성을 열어두는 이유다.

## 10. 실익 정직 평가 (과대 포장 금지)

backlog L919 의 정직 평가를 ADR 에 못박는다:

- **1B/2048 decode = 제한적** — decode 는 memory-bound(matmul 이 아니라 KV 읽기 대역폭이 병목)라, Quest 의 attention FLOPs 절감은 *이미 작은 compute* 를 더 줄일 뿐이고 메모리 대역폭(진짜 병목)은 select 해도 page 단위 random access 가 오히려 비효율일 수 있다. 1B 에서 **순이득 기대 낮음** — 항목 0 의 1B 교훈(Round 14–15: 1B 누적 score 무가치)과 같은 결.
- **가치 = 세 갈래**:
  1. **prefill TTFT** — prefill 은 compute-bound 라 selection 절감이 직접 TTFT 단축(shadowAttn 이 모바일 prefill 전용인 이유).
  2. **8B+/디스크 offload 토대** — 8B/long-ctx 에서 KV 가 메모리/디스크에 안 들어갈 때 selective read + prefetch 가 *가능성 자체의 조건*(KVSwap). 1B 실익 미미는 MEMORY 명시 — 가치는 큰 모델.
  3. **논문 기여** — 4번째 표면이 *동일 plan-returning 문법으로 가산*됨을 실증(3축 골격의 확장성 = 새 stage 축 동작이 무비용 추가). 이것이 정량 성능과 독립한 *구조적* 기여.
- **과대 포장 금지 못박음**: "Quest 가 1B 를 빠르게 한다"는 주장 **금지**. 본 표면의 1차 정당화는 *확장성/토대*이지 1B 성능이 아니다. 1B 성능 주장은 8B 측정 전까지 하지 않는다.

## 11. C-ABI forward-compat (미래 `.so` read-stage plugin)

ADR-0004 §7 M-A(TensorHandle C-ABI) / §10 MQ-5(AbiStageCtx 가산 확장) 선례를 따른다 — read-plan 표면도 처음부터 `.so` 전환 가능하게 설계:
- **`KVReadStage::read_plan`** → `extern "C" read_plan(ctx_handle, out_plan: *mut KVReadPlanAbi) -> i32`. plugin 경계는 step/layer-boundary tier(layer 당 1회 ≪ op 당)라 ADR-0005 D2 의 "step/boundary = Rust-native, `.so` 시 기계적 shim" 대상.
- **`KVReadPlan`** → `#[repr(C)] KVReadPlanAbi { granularity: u32, select_ptr: *const usize, select_len: usize }`. `ReadGranularity` 는 `#[repr(u32)]`(Token=0, Page=1; page_size 는 별 필드). ADR-0004 KVCachePlan 의 `Vec` → ptr+len 마샬링과 동형.
- **`ReadStageCtx`** = ADR-0004 §10 MQ-5 의 `AbiStageCtx`(`tensor(kind)` fn-ptr 테이블) **재사용** — read stage 가 `tensor(Key)`/`tensor(QueryStats)` 를 읽는 경로가 이미 C-ABI 가산 확장된 표면. read stage 전용 ctx 신설 불요(기존 StageCtx 가 충분).
- **`SelectiveRead` capability** → ADR-0010 멀티-vtable bundle ABI 의 capability 슬롯으로 등록(한 `.so` 가 read-stage + selective-format capability 를 동시 운반 가능). GATE-C v2 ABI 버전 bump 불요 — TensorKind 가산이 fn-ptr 시그니처 불변인 것과 동형(read-plan 은 새 *축 registry* 추가라 기존 stage/weight/format registry ABI 불변).

## 12. References

- ADR-0003(정적 crate + linkme + force-link), ADR-0004(`KVCacheStage` plan-returning + §7 M-A TensorHandle + §10 M-Q QueryStats — read-plan 의 K/Q 읽기 어휘 + 신호 공급원), ADR-0005(D5 generic floor + 특화 opt-in + D6 3축 registry + §8 INV-HOTPATH-DISPATCH — read-plan 의 capability opt-in/registry/hot-path 정합 선례), ADR-0006(weight 축 거울 = 세 번째 plan-returning 형제, read-plan = 네 번째)
- ADR-0010(멀티-vtable bundle ABI — `.so` capability 슬롯)
- `engine/src/format/kv_cache_format.rs`(`attention_into` 시그니처 — read-plan 이 selective 변형을 capability 로 추가), `engine/src/kv/eviction/stage_registry.rs`(`KVStageCtx`/`execute_kv_plan`/`KV_CACHE_STAGES` — read registry 거울 대상)
- `/CONTEXT.md`(stage=verb ⊥ format=noun, 3축 직교), `spec/41-invariants.md` §3.28(INV-DECODE-STAGE / INV-HOTPATH-DISPATCH / INV-147)
- backlog `.agent/todos/backlog.md` 항목 4(L909~920) — Description·핵심 통찰·설계 쟁점·해금(~9건)·실익 정직 평가
- 스프린트 `.agent/todos/sprint_kv_roadmap_item34_2026_06_12.md` §P4

### Spec Triage 결론 (arch-only, spec 변경 없음)

`/spec-manage` triage 판정 = **spec 무관**:
- **새 어휘는 plugin 계약** — `KVReadStage`/`KVReadPlan`/`attention_into_selected` 는 technique-api plugin 어휘 + 엔진 impl 의 계약이지 `spec/` 요구사항이 아니다. grep 확인: `read_plan`/`KVReadStage`/`attention_into_selected` 는 `spec/` 0건, `attention_into` 도 `spec/` 0건(ADR-0004/0005 도 spec 무관으로 처리한 선례 — TensorKind/QueryStats 가 spec 에 없음).
- **기존 INV 의미 불변** — read-plan 은 (a) off 기본(read stage 부재 = full read = 현행 100% 보존), (b) 폴백 정확성(미지원 format = full read), (c) 근사 가속(정확성 계약 아님)이라 *동작 추가가 아니다*. INV-DECODE-STAGE 시리즈(stage phase)·INV-HOTPATH-DISPATCH(layer-tier dispatch)는 read-plan 구현 시 *적용 대상*이나 그 INV 들의 **의미를 바꾸지 않는다**.
- **신규 INV 불요** — ADR-0004 §7/§10 amendment 가 INV 를 추가하지 않은 선례. read-plan 의 hot-path 무회귀는 기존 **INV-147**(hook=None hot-path 비용 = `Option::is_some` 1회)의 정신을 재사용(R1 완화). 새 INV 는 *구현 시점*(리팩토링 머지 후)에 layer-tier read dispatch 가 정적으로 flip 될 때 INV-HOTPATH-DISPATCH normative 행으로 흡수 가능(현재는 vacuous — 구현 없음).
- **arch 매핑은 구현 시점** — 본 ADR 이 SSOT. 구현 단계(§9 backlog 재등록 항목)가 착수될 때 `arch/` 에 컴포넌트 매핑(executor seam + capability + page-meta 흐름)을 추가한다. 현재는 ADR 단독(코드 0줄이라 매핑할 구현 부재 — ADR-0004 §10 M-Q 가 `arch/kv_query_stats.md` 를 구현과 함께 신설한 것과 같은 타이밍 규율).
