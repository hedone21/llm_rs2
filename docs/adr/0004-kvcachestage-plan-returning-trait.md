# ADR-0004: `KVCacheStage` — 단일 plan-returning trait (per-head keep + 가중 merge 통합)

> **Status**: Accepted
> **Date**: 2026-06-05
> **Decision-makers**: 사용자 + Architect (grill-with-docs 세션 "EvictionPlan 재설계", /loop M2 STOP 후)
> **Selected**: stage 축 확장 기법(eviction/merge)을 **단일 plan-returning trait `KVCacheStage`** 로 통일. 입력 = 캐시 읽기 접근 + scores + budget, 상태는 impl, 반환 `KVCachePlan`(keep + 가중 merge), 버퍼 변형은 엔진이 실행.
> **Supersedes (부분)**: ADR-0003 §D2 의 "planning 표면이 h2o+/d2o 를 덮는다 (낙관적)" — 본 ADR 이 *어떻게* 덮는지 확정하며 §D2 의 미명세를 닫는다.
> **Related**: ADR-0003(확장 메커니즘 = 정적 crate + linkme), `/CONTEXT.md`("KVCacheStage"/"KVCachePlan" 항목), `arch/pipeline_stage_design_v2.md`, `engine/src/pressure/eviction/compact_parity.rs`

---

## 1. Context

ADR-0003 은 stage 축 기법을 별도 crate 가 구현하는 "planning 표면"으로 플러그인화하기로 했다(§D2). 그러나 그 표면(M1 의 `EvictionPlan::plan_keep → (Vec<usize>, Vec<Merge>)`)이 실제로 **h2o+·d2o 를 못 덮음**을 /loop M2 가 실측했다 (workflow `wf_a9f025a7`):

- **h2o+** 는 head 마다 다른 토큰을 유지(per-head) → 단일 layer-wide `Vec<usize>` 로 표현 불가 → `plan_keep` 가 `None` 반환.
- **d2o** 는 `EvictionPolicy` 가 아니라 가중 scatter-merge handler. `Merge{into, from}` 에 가중치가 없고(현 `apply_merges` 는 uniform), cosine-nearest 매칭에 raw K 가 필요하며, EMA 임계가 호출 간 stateful 이다.
- 가중 merge 적용 코드(`StandardFormat::apply_merges`)·per-head 압축(`compact_keep_positions_for_head`) primitive 는 이미 존재. parity 게이트(`compact_parity.rs`)가 4 정책(sliding/streaming/h2o/no_eviction)×3 dtype 에서 plan→compact ≡ in-place evict 를 이미 증명.

사용자는 (B) "표면을 per-head + 가중 merge 까지 확장" 을 선택했다.

## 2. Decision

**stage 축 확장 기법을 단일 plan-returning trait `KVCacheStage` 로 통일한다.**

**(D1) plan-returning, self-mutating 아님.** plugin 은 캐시를 *읽고* **계획**(`KVCachePlan`)을 반환할 뿐, 버퍼를 직접 변형하지 않는다. 변형은 엔진이 plan 을 `compact` 로 실행해 독점한다. "plan-returning vs self-mutating" 이 핵심 축이며 — 이것이 (a) 캐시 손상 방지, (b) 미래 `.so` C-ABI 표면 최소화(ADR-0003)의 근거다. **순수성은 요구하지 않는다** — stateful impl(아래 D4)을 허용한다.

**(D2) `KVCachePlan` 타입** (keep 은 배타 enum, merge 는 직교 필드):
```rust
struct KVCachePlan { keep: KeepSpec, merges: Vec<WeightedMerge> }
enum KeepSpec {
    LayerWide(Vec<usize>),       // sliding·h2o·streaming·no_eviction·d2o (ascending)
    PerHead(Vec<Vec<usize>>),    // h2o+ : [n_kv_heads][keep], 각 ascending·길이 동일
}
struct WeightedMerge { into: usize, into_weight: f32, from: Vec<(usize, f32)> } // Σw + into_weight ≈ 1
```
executor 매핑: `LayerWide`+merges → `apply_merges`(가중) → `compact_keep_positions`. `PerHead` → head 별 `compact_keep_positions_for_head`. `PerHead`+merges(=d2o+h2o+ 융합) → **당분간 `bail!`**(promotion-trigger). `new_pos` 는 plan 에 없음 — 엔진이 `keep.len()` 으로 도출(PerHead 전 head 길이 동일 assert).

**(D3) `Merge` → `WeightedMerge` 통합.** merge 타입은 하나. 가중치는 plan 에 baked 되고 `apply_merges` 가 그 가중치를 쓴다(현 uniform 대체). merge-free 정책은 빈 `Vec`.

**(D4) 상태는 impl 에.** d2o EMA(τ) 같은 호출 간 상태는 `&self` + interior-mutability(Mutex)로 plugin struct 가 보유한다 — D2OHandler 의 현 방식. ctx 로 thread 하지 않는다(type-erased 상태 blob/제네릭은 `dyn`-object trait 에 치명적).

**(D5) 입력은 "캐시 읽기 접근" 추상(`StageCtx`), `&KVCache` 직접 아님.** trait 은 `technique-api` 에 살고 거기서 엔진 타입 `KVCache` 를 참조할 수 없다(단방향 의존). 그래서 `technique-api` 가 정의하는 읽기 추상 `StageCtx`(geometry + scores + dequant K 읽기 accessor)를 받고, 엔진이 그것을 `&KVCache` 위로 구현한다. 정적 단계엔 borrow, 미래 `.so` 단계엔 동일 추상이 C accessor/flat 스냅샷으로 교체 — forward-compatible.

**(D6) 네이밍.** `KVCacheFormat`(저장 표현) 의 형제로 `KVCacheStage`(상주 토큰 조절). "Eviction" 접두는 merge/per-head 를 포괄 못 해 폐기. 부속: `KVCachePlan`/`KVCacheStageReg`/`KV_CACHE_STAGES`/`StageParams`. 세션 `EvictionStage`(dormant) deprecated — WHAT 은 `KVCacheStage`, WHEN 은 엔진 소유. legacy in-place `EvictionPolicy` 는 마이그레이션 중 공존 후 phase-out.

## 3. Rationale

- **plan-returning 이 C-ABI/안전의 근거**: plugin 이 indices+weights 만 방출하고 엔진이 검증·실행 → plugin 이 버퍼 손상 불가, KVCache 내부 결합 최소(ADR-0003 의 self-mutating accessor table 회피).
- **"순수성" 기각이 정직함**: 캐싱/리플레이는 이 엔진에 use case 없음(YAGNI). stateful 객체도 C-ABI 경계를 멀쩡히 넘음. 따라서 d2o EMA 를 impl 상태로 두는 게 옳고, 그래야 d2o 가 같은 trait 에 들어온다.
- **keep=enum / merge=필드 가 직교성 충실**: keep 모양(layer-wide vs per-head)은 진짜 배타 → enum. merge 는 keep 과 독립 capability → 필드. 융합(per-head+merge)은 타입상 이미 표현 가능(executor 만 나중에 구현, 타입 breaking 0).
- **기존 primitive 재사용**: `apply_merges`·`compact_keep_positions(_for_head)` 가 이미 있고 `compact_parity` 가 등가성 게이트를 제공.

## 4. Consequences

- M1 `technique-api` 의 `EvictionPlan`(trait)/`Merge`/`EvictionPolicyReg`/`EVICTION_POLICIES`/`PolicyParams` → `KVCacheStage`/`WeightedMerge`+`KVCachePlan`+`KeepSpec`/`KVCacheStageReg`/`KV_CACHE_STAGES`/`StageParams` 로 rename + 재구성. `StageCtx` 읽기 추상 신설.
- 엔진: `StageCtx` 를 `&KVCache` 위로 구현, `KVCachePlan` executor(LayerWide/PerHead 분기 + 가중 `apply_merges`), `KV_CACHE_STAGES` 레지스트리로 match arm(session.rs:621·init.rs) 제거, startup self-test.
- d2o: `KVCacheStage` 로 재구현(plan 에 가중치+nearest 산출, EMA 는 impl Mutex, K 는 StageCtx 로 읽기). 기존 D2OHandler 경로와 등가성 테스트 선행(미확립 시 STOP — ADR-0003 M4 게이트).
- head_importance session forward 배선 추가(현재 flat 만).
- **검증**: `compact_parity` 를 가중 merge·per-head 케이스로 확장. ~~Q4_0 merge 는 비활성 유지(현 정책)~~ — plan 은 dtype-agnostic, executor 가 dtype 분기.

> **(정정 — M4 사용자 결정 2026-06-05)**: "Q4_0 merge 비활성 유지"는 폐기한다. 현 `D2OHandler` 는 이미 `scatter_reduce_q4`(d2o_handler.rs:585)로 **Q4_0 에서 가중 merge 를 수행**하므로, d2o 를 plan→executor 로 옮기며 executor 의 `apply_merges`(현 Q4 스킵)를 그대로 쓰면 Q4_0 merge 가 silently drop 되어 **paper Eq.11 정렬 회귀**다. 따라서 executor 의 `apply_merges` 를 (a) `WeightedMerge` 가중치 사용 + (b) **Q4_0 지원**(기존 `scatter_reduce_q4` 의미 이식)으로 확장한다. plan 표면은 여전히 dtype-agnostic(`WeightedMerge` 는 위치+가중치만 운반); dtype 분기는 executor 가 흡수 — 본 ADR 의 "executor 가 dtype 분기" 원칙과 일관. 동등성 게이트: 새 d2o-KVCacheStage(plan→확장 executor)가 기존 `D2OHandler` 와 F32/F16/Q4_0 모두에서 bit-identical(미확립 시 STOP).

## 5. Alternatives Considered

- **(가-원안) D2O 를 그대로 통합 trait에 — K snapshot 복사** (REFINED): "스냅샷 복사"는 정적 단계에 불필요(읽기 borrow 로 충분) → D5 로 정련. 스냅샷은 `.so` 단계에서만.
- **(나) 2 표면(planning ⊥ handler)** (REJECTED): d2o 를 handler 로 분리. 단일 표면의 단순함을 잃고, "순수성" 논거가 약해 분리 근거가 부족 — D4(상태 impl)로 d2o 가 plan-returning 에 들어오므로 불필요.
- **Plan 타입 B(top-level enum)/C(전부 per-head)** (REJECTED): B 는 per-head+merge 를 배타로 못박아 직교성 위배 + 융합 시 breaking. C 는 layer-wide 정책이 head 수만큼 복제 — 과설계. → D2(struct + KeepSpec enum) 채택.
- **상태를 ctx 로 thread** (REJECTED): `&mut dyn Any`(downcast 취약·엔진이 미지 상태 소유) / 제네릭(`dyn`-object 깨짐, 레지스트리 불가). → D4(impl interior-mut).

## 6. References

- ADR-0003 §D2 (본 ADR 이 정정/구체화)
- `/CONTEXT.md` — "KVCacheStage"/"KVCachePlan"/"EvictionPolicy(legacy)" + Flagged ambiguities("stage 축 vs PipelineStage vs KVCacheStage")
- `engine/src/pressure/eviction/compact_parity.rs` (등가성 게이트), `engine/src/pressure/eviction/h2o_plus.rs`(per-head), `engine/src/pressure/d2o_handler.rs`(가중 merge/EMA/nearest)
- `engine/src/pressure/kv_cache.rs` (`compact_keep_positions`/`_for_head`), `engine/src/pressure/standard_format.rs` (`apply_merges`)
- workflow `wf_a9f025a7` (4축 surface map), `.agent/todos/adr0003_impl_progress.md`

## 7. Amendment — M-A: TensorHandle 범용 읽기 표면 (2026-06-06 결정)

**맥락**: 본 ADR 의 `StageCtx` 는 6개 기존 기법(sliding/streaming/h2o/no_eviction/h2o_plus/d2o) 요구의
합집합인 최소 accessor 집합이었다. 그러나 **value-dependent 기법(CAOTE: criticality = `a_i·‖v_i−o_h‖`)**
은 V(value) 벡터를 읽어야 하는데, 기존 표면은 `dequant_k`(K)만 노출하고 V accessor 가 없어 plugin 안에서
metric 을 계산할 수 없었다(plugin = metric *선택자*에 그침). "zero-compile plugin 으로 임의 기법" 목표에
대한 구조적 갭.

**결정**: 읽기를 **단일 메커니즘 `tensor(kind) -> Option<&dyn TensorHandle>`** 로 통일한다.
- `TensorKind = { Key, Value, AttnWeights, Scores }`. `TensorHandle { shape()->TensorShape(POD);
  dtype()->TensorDtype; read_row(row, kv_head, out:&mut[f32]) }` (dyn-safe, out-param).
- 기존 `dequant_k`·`head_score`·`has_head_scores` + 신규 `dequant_v`·`attn_weight`·`has_attn_weights` 는
  전부 `tensor()` 위 **default sugar**. 엔진은 `tensor()` 1개만 구현(KVStageCtx). technique crate 는 무수정.
- **flat `importance()` 만 zero-copy 직접 노출(D1 통합 예외)** — scalar 를 per-element `read_row` 로 돌리면
  H2O scalar 랭킹 경로가 유일하게 순손해라 제외.
- **AttnWeights = `AttentionScoreAccumulator::last_step_head_attn`** (last layer·last step; CPU overwrite /
  GPU=head_importance proxy). `has_attn_weights()` 게이트 — last-layer 근사임을 명시(windowed/per-layer 정확값
  아님). CAOTE 는 가용 시 `attn_weight`, else `importance` 폴백.
- v1 CAOTE 는 `KeepSpec::LayerWide` 만 산출(head reduce 는 plugin 내부). per-head CAOTE 는 PerHead executor
  (단계 ⑤)와 함께.

**근거(PoC + 판정단)**: TensorHandle 의 추가 indirection(`read_row` per-element vtable)이 additive accessor
대비 비용이 큰가? — host x86 + **ARM(S25 Oryon)** microbench 실측: handle vs additive = **±0~1%**(F16 ±0%,
Q4_0 ±1%). 둘 다 per-element vtable 1회로 구조적 등가, TensorHandle 은 `tensor()` 핸들 조회만 head당 1회(O(H))
추가. dyn-vs-direct gap(ARM Q4_0 +63%)은 `plan(&dyn StageCtx)` plugin 경계 비용 그 자체로 두 안 공통·기존
기법이 이미 지불. → 성능은 차별 요소 아님 → 장기 OCP(미래 입력=variant 1개)로 TensorHandle 채택.

**Alternatives Rejected (panel)**: ① additive granular accessors(perf 동등이나 메서드 단조증가) ② CacheView
단일 borrowed struct(trait→struct 교체 = OCP/C-ABI 위반) ③ capability negotiation(materialize 타이밍 미해결
+ no-op default silent-wrong) ④ engine-precompute(plugin=metric 선택자 → ADR-0003 가 없앤 closed match-arm
재도입, 북극성 위배). 상세: 설계 workflow `design-general-stagectx`(wf_1dda0f82).

**C-ABI forward-compat**: `&dyn TensorHandle` → `{ opaque ptr + extern "C" read_row fn-ptr + #[repr(C)] POD
shape }`. `TensorKind→u32`, out-param = FFI out-buffer. snapshot/capability bit 불요. (코드 정합 — 리뷰
`wf_b5d13ff1` 반영: `TensorShape #[repr(C)]`, `TensorKind`/`TensorDtype #[repr(u32)]` 부여로 주장과 구현 일치.)

**Scope / deferred(후속)**:
- production eviction-hook 의 head_scores/last_attn → StageBackedPolicy threading 은 **CAOTE CLI 배선과 함께**
  후속(현 builtins 는 None 으로 충분; CAOTE 는 host 테스트로 증명). `EvictionPolicy::evict_with_head_scores`
  확장 필요.
- **windowed RawAttn 패밀리(SnapKV/Scissorhands/Ada-KV)** 는 엔진이 per-query attention 윈도우를 보존하지
  않아 미해금 — interface 모양과 직교한 별도 엔진 작업. TensorKind 에 variant 추가로 흡수 가능.
- `query_state`(Quest): decode-step Q 캡처 미배선 → drop(후속 PR).
- TensorKind→TensorHandle fold 임계(현 accessor 다수): RawAttn×Window 첫 기법 등장 시 트립와이어.

**구현/검증(M-B~M-G, `.agent/todos/tensorhandle_impl_progress.md`)**: technique-api 표면 + 엔진 핸들
(KeyHandle/ValueHandle/ScalarHandle) + `dequantize_v`(dequant_k 의 v_buffer 미러, bit-identity F32/F16/Q4_0
테스트) + CAOTE crate(`crates/techniques/caote`, technique-api 만 의존, dev-dep + force-link, host
value-aware 실행 테스트). 게이트: compact_parity·d2o_stage_eq_handler_* 무회귀 + lib 1238/0 + clippy
--workspace clean + release linkme 생존.
