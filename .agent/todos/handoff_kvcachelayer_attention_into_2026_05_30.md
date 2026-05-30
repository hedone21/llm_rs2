# Handoff — ④ KVCacheLayer KIVI creep 제거 (attention_into) grill 정리

> **일자**: 2026-05-30
> **진입 문장**: **"④-a 확정 완료 — arch §4.1(base trait 7, attention_into) + spec PAIRED-KERNEL + CONTEXT.md(Format) 반영(2026-05-30). 다음 = ④-b / Phase α-K (ADR-0001 Generic→dyn 동행)."**
> **트리거**: `/grill-me` — 1차 review 후보 ④ "KVCacheLayer migration 이 base trait KIVI no-op creep 떨궈내야" (handoff_kv_weight_grill_2026_05_28.md 의 재개 진입점 1순위).
> **선행 문서**:
> - `arch/pipeline_stage_design_v2.md §4.1` (KVCacheLayer base trait — 현재 6-method 선언, 본 grill 이 7 로 정정 제안)
> - `arch/pipeline_stage_design_v2.md §3.4` (concrete-handle 패턴 — D2O 예시, 본 grill 의 plan concrete-handle 근거)
> - `docs/adr/0001-kv-dispatch-paradigm.md` (Generic → Trait object 전환 = Phase α-K)
> - `.agent/todos/handoff_kv_weight_grill_2026_05_28.md` (④′ 종결까지. 본 grill 은 그 다음 후보)
> **상태**: **④-a 확정 + 문서 반영 완료** (arch §4.1 / spec / CONTEXT.md / backlog). 코드 변경 0 — 구현은 Phase α-K (ADR-0001 Generic→dyn 전환 동행).

---

## 0. 한 줄 요약

④ = KVCacheLayer base trait 의 **KIVI no-op creep 제거**. 결론 = attention dispatch 책임을 layer 로 위임(`attention_into`)해서 forward_gen 의 "너 KIVI냐?" paradigm sniff 를 제거. **④-a**(지금 확정 가능, base trait 7-method)와 **④-b**(plan enum 평탄화, Phase α-K 연기)로 분리. 런타임 KIVI 활성화 stress test 를 base trait 7 무변경으로 통과. **2026-05-30 ④-a 확정 + arch §4.1 / spec / CONTEXT.md 반영 완료.**

---

## 1. 문제 배경 (왜 ④ 인가 — 의도)

### creep 의 실체
- `KVCacheOps` trait (`engine/src/kv_cache_ops.rs`) 은 현재 **21 method**. 그 중 **9개가 KIVI-specific no-op creep**:
  - `get_kivi_raw_buffers` (default `None`) — **이름에 paradigm "kivi" 박힘**. forward_gen 이 raw Q2 buffer 얻는 우회로.
  - `res_pos`/`q2_tokens`/`res_cap`/`needs_flush`/`flush_if_needed` (default `0`/`false`) — "KIVI Plan support" 주석. plan.rs 가 소비.
  - `needs_attn_scores`/`set_attn_scores` (default `false`/no-op) — AWQE score 수집. **#17 Score domain refactor 별 sprint — ④ 범위 밖.**
- §4.1 이상형 = **6-method** base trait (geometry 3: `idx`/`current_pos`/`capacity` + mutation 3: `write_kv`/`write_kv_batch`/`compact`), storage-format-agnostic. 현실은 21. 이 갭 검증이 ④.

### creep 발생 원인
KIVI(Q2 packed + F32 residual)가 표준 KVCache(F32/F16/Q4_0 연속)와 **storage 구조 자체가 다름**. 하나의 generic trait(`<C: KVCacheOps>` monomorphization)으로 양쪽을 추상화하며 표준엔 의미 없는 KIVI 동작이 default no-op 으로 스며듦.

### 왜 지금 (ADR-0001 창문)
현재는 generic 이라 creep 이 monomorphize 시 죽은 코드로 사라져 "봉인". 그러나 ADR-0001 (Generic `<C>` → `Arc<dyn KVCacheLayer>` trait object, = Phase α-K) 전환 시 **base trait 에 선언된 method 만 dyn 호출 가능** → creep 9개가 base trait 에 **영구화**. 새 paradigm(TurboQuant 등) 추가 시 "내겐 의미 없는 `get_kivi_raw_buffers` 를 어떻게 구현?" 을 마주침. **migration 전(=지금)에 떨궈내야 함.**

### north star 연결
KV paradigm 은 활발한 확장 지점(Sliding/H2O/D2O/KIVI/SnapKV/미래 Turbo). 새 paradigm = 새 impl + paired kernel, **base trait·forward_gen 무변경**이어야 함. 현재는 새 paradigm 추가 시 (a) trait method 추가(전 impl 영향) (b) forward_gen 에 sniff 분기 추가 → north star 위반. ④ 가 이를 해소.

---

## 2. 핵심 코드 발견

1. **`get_kivi_raw_buffers` 는 forward_gen 의 paradigm sniff** (`forward_gen.rs:417-461`):
   `if let Some(raw) = kv_cache.get_kivi_raw_buffers()` 의 `Some`/`None` 이 사실상 "이 layer 가 KIVI냐?" 런타임 판별. `INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`("base-trait-handle Stage 는 paradigm 모름") 정면 위반.

2. **통일 read 인터페이스는 이미 존재** = `get_view() -> (Tensor, Tensor)` (`kivi_cache.rs:2316`, 표준 `kv_cache.rs:1028` 과 시그니처 동일). KIVI 가 내부에서 Q2→F32 dequant + residual concat. **`get_kivi_raw_buffers` 는 순수하게 NVIDIA native fused 성능 우회로** — Adreno(production 주 타겟)는 `is_nosub_device()` 가 false 라 어차피 `get_view` dequant 경로 (forward_gen.rs:420-424 주석). 즉 raw 노출의 성능 이득은 **NVIDIA 한정**, 실측 수치는 코드에 없음(주석 기반).

3. **attention dispatch 2 경로**:
   - **forward_gen** (interpreted, Rust 코드가 매 op 분기) — `attention_into`(A) 가 고칠 대상.
   - **plan.rs** (compiled, op graph 미리 빌드) — `AttentionVariant` enum(`Standard`/`StandardFlash`/`KiviAssembled`/`KiviNative`/`HybridKvSplit`, plan.rs:168) 으로 KIVI 를 안다.

4. **plan 라이프사이클**: `transformer.rs` 가 decode 진입 시 **lazy 1회 빌드**(`build_full_plan`:2470 / `build_kivi_full_plan`:2745) → `gpu_plan: Option<FullKernelPlan>` session 캐시(`prologue.rs:76`) → **매 토큰 execute 재사용**(`execute_plan`:2520, hot). `PlanInvalidated`(capacity 변경 / weight swap `ratio_generation` INV-129) 시 재빌드. **build=cold, execute=hot.**

5. **plan paradigm 분기 3중** (OCP 위반 — 사용자 지적):
   - `transformer.rs`: `build_full_plan` vs `build_kivi_full_plan` (어느 빌더)
   - `transformer.rs`: `execute_plan(&mut [KVCache])`:2526 vs kivi execute(`&[KiviCache]`):2773 (concrete 타입 강결합)
   - `plan.rs`: `AttentionVariant`/`KvUpdateVariant` enum + execute match(1378-1518) + `build_kivi_layer_plan`(4604, "표준 plan 만든 뒤 KV update/attention step 갈아끼움")

6. **QuantizeHandler 는 dead stub** (`pressure/quantize_handler.rs`):
   - `handle()` 무조건 `NoOp`(51,58). pipeline 에 **미등록**(re-export `pub use` 만, `add_handler` 0건). `target_bits_for_pressure()`(PressureLevel→8/4/2 bits 매핑) production 호출 **0건**(전부 자기 파일 주석+테스트).
   - **현재 KIVI 는 시작 시 CLI 정적 설정** (`--kv-kivi-bits` default 2, `kv_mode.rs:18`). 런타임 bit 전환도 paradigm 전환도 **미구현**.
   - 즉 QuantizeHandler = "런타임 KV quantization 담당하기로 이름만 잡아둔 빈 의자".

---

## 3. 도달한 결론 (미확정 — 재논의 대상)

### Q1: attention dispatch 책임 = **(A) KVCacheLayer base trait 에 `attention_into` 추가**
- `forward_gen` 은 `kv.attention_into(q, backend, out, dims, scores)` 한 줄 → **paradigm 무지**.
- `KIVILayer::attention_into` 내부에서 디바이스 분기(NVIDIA native `attention_gen_kivi` `self.qk_buf` 직접 / Adreno `self.dequant_view()` + flash). `get_kivi_raw_buffers` 의 raw 구조체 노출 **증발**(자기 필드 자기가 읽음).
- `StandardLayer::attention_into` 내부에서 dtype 분기(현재 forward_gen.rs:476-491 의 `is_q4_gpu`/`use_typed_attn` 흡수).
- attention 은 모든 KV paradigm 보편 연산(D2O raw-K-read 1-consumer 가설적 seam 과 달리) → base trait 에 두면 deletion-test 통과. `INV-KVCACHELAYER-PAIRED-KERNEL` 을 코드로 실체화.
- 기각: (B) concrete-handle 분기 — Mixed storage 에서 OCP 재발. (C) 별도 capability trait — attention 보편이라 opt-in capability 취지 안 맞음, (A)와 실질 동일 + trait 하나 더.
- **capability 검토 결론** (사용자가 "capability 쓰면 되잖아" 제기): capability 는 paradigm 을 **격리만 하지 제거 안 함**(`KiviAttentionBackend` trait 이 이미 이름·시그니처에 KIVI/Q2/residual 박고 있음, backend.rs:128). raw-노출 capability(`KiviRawAccess`)는 creep 을 capability 로 옮길 뿐 + forward_gen 이 여전히 두 capability 엮음. 추상-attention(`attention_into`)이 paradigm 을 진짜 숨김.

### ④-a / ④-b 분리 (재검토로 도출 — scope 정직화)
| | **④-a** (확정 방향) | **④-b** (friction-triggered 연기) |
|---|---|---|
| 내용 | `attention_into` + plan 이 `res_pos` 등을 **concrete-handle**(`KiviNative{step, layer:Arc<KIVILayer>}`)로 읽기. enum **유지**. | enum 평탄화(`Vec<KernelStep>`) + layer `build_*_steps` 위임 + 동적 arg accessor |
| base trait | **7 method** ✓ | 9~10 (god trait, front-door cap ~7 초과) |
| SRP | layer = 실행 위임만 (자연) | layer = compute graph build 책임 (긴장) |
| 해결 | forward_gen creep + base trait creep 전부 | plan.rs 3중 분기 → 위임 |
| 비용 | — | ADR-0001(Generic→dyn) 얽힘, Phase α-K 본작업 |

- **base trait 최종 7**: geometry 3 + mutation 3 + `attention_into` 1. KIVI creep 0.
- **④-a 가 base trait creep 을 완전 제거하는 근거**: plan 이 generic `<C: KVCacheOps>` 라 `cache.res_pos()` 를 base trait method 로 부르는 게 creep 의 직접 원인. ④-a 는 plan KIVI variant 가 `Arc<KIVILayer>` concrete-handle 보유 → `layer.res_pos`(inherent)로 읽음 → base trait 에서 `res_pos`/`q2_tokens`/`get_kivi_raw_buffers` 제거. **enum 평탄화(④-b) 없이도 base trait 7 달성.**
- **④-b 를 연기하는 근거**: plan 은 성능 opt-in 경로. 새 paradigm 은 `attention_into`(forward_gen)로 **즉시 동작**(plan 없으면 fallback, 느리지만 정확). plan 빠른 경로는 최적화라 별 축. plan OCP 위반은 "새 paradigm 을 빠른 경로로 올릴 때만" friction → 둘째 paradigm 올 때 추상화(KISS). ②′ Backend long-tail 을 friction-triggered 로 미룬 것과 동일 논리.

---

## 4. stress test (전부 통과 — ④-a 견고함 확인)

### 4.1 런타임 KIVI 활성화 (사용자 요구: "런타임에 kivi 가 활성화 될 수 있어야 해")
- **현재 구조적으로 불가능**: `ModelForward.kv_caches: Vec<KVCache>` concrete 하드코딩(model_forward.rs:46). 표준/KIVI 가 컴파일 타임 분리 경로(`Vec<KVCache>` vs `&[KiviCache]`). QuantizeHandler dead. → 이게 ADR-0001 의 존재 이유.
- **④-a 는 base trait 7 을 안 키움**: 마이그레이션(F32 K/V → Q2)·layer swap·plan 재빌드 전부 base trait **밖**(handler/slot/plan). forward 는 `attention_into` 로 무지 유지.
- **plan 은 "런타임 직접 수정"이 아니라 "invalidation → 전체 재빌드"** 로 paradigm 전환 흡수 — weight swap 의 `ratio_generation` 패턴 재사용. concrete-handle 은 재빌드 시 새 KIVILayer capture. 전환 토큰은 forward_gen fallback(attention_into 라 paradigm 무지로 정확).
- **데이터 마이그레이션 비용 + plan 재빌드 비용 = 사용자가 명시적으로 수용** ("동적 전환의 당연한 비용"). 따라서 "점진 vs 일괄 전환" deep dive 는 불필요.
- 런타임 활성화 = ④-a(전제·입구) + ADR-0001(dyn) + QuantizeHandler 실체화 + KVLayerSlot RCU + paradigm generation. **④-a 만 "지금 확정", 나머지 전부 Phase α-K.**

### 4.2 마이그레이션 책임 귀속 = **(A)**
"현재 layer → KIVI 변환"을 누가 아는가:
- **(A) 추천**: `QuantizeHandler`(실체화 시) 가 glue — source = `StandardLayer::get_view()`/`read_k_layer_wide`(concrete-handle, §3.4 D2O 패턴) read + target = `KIVILayer::from_kv()` 생성자. **base trait 7 유지.** 변환 지식이 pressure 도메인(④′ 도메인 귀속 기준 정합). source 거의 항상 Standard 고정 → 1×M 조합, friction-triggered 추상화.
- 기각: (B) `KIVILayer::from_source(&dyn KVCacheLayer)` — source 전체 K/V 를 base trait read 로 노출 → §4.1 이 막은 generic content read 부활. (C) `StandardLayer::to_kivi()` — source 가 KIVI 앎 = creep 부활.

### 4.3 두 축 직교성 (사용자 질문: "외부 사용자가 KiviLayer 만드는 게 맞아? eviction 은 Stage 만 아닌가?")
- **비대칭 아니라 직교 두 축**:
  - **storage 축 (KVCacheLayer impl)**: "데이터를 어떤 형태로 저장·attention". Standard/KIVI/Turbo. 새 멤버 = **Layer impl**(write_kv/attention_into 형태가 다름).
  - **policy 축 (EvictionStage)**: "이미 저장된 데이터에서 어느 토큰 버림/병합". Sliding/H2O/D2O. 새 멤버 = **Stage**(`compact(keep,merges)` primitive 만 호출, 형태 무변경).
- **판단 기준**: base trait mutation primitive(`write_kv`/`compact`)로 표현되면 **Stage**, 저장 형태 자체가 다르면 **Layer impl**. (KIVI: compact 로 "F32→Q2" 표현 불가 → Layer impl. Sliding: compact 로 완전 표현 → Stage.)
- **EvictionPolicy 코드 확인**(eviction/mod.rs:15-18): `should_evict(&KVCache)` + `evict(&mut KVCache, target_len)` — 토큰 수만 줄임, 형태 무변경. 단 현재 concrete `KVCache` 강결합 → ADR-0001 후 `&mut dyn KVCacheLayer` + base trait `compact` 만 호출해야 진짜 직교(④-a 가 전제).
- **외부 기여자 대부분은 Stage**(정책 연구가 흔함). storage paradigm 추가(KIVI/Turbo)는 드문 프레임워크 작업. handoff #8("신규 stage 추가 가이드 doc") 첫 문단에 두 축 + 판단 기준 박아야.
- M storage × N policy = **M+N 코드**(조합 폭발 X) = north star. `attention_into` = storage 축 인터페이스, Stage(policy)와 직교. ④-a 가 "왜 attention_into 가 base trait 이고 Stage 아닌가" 정당화.

---

## 5. 미확정 / 재논의 진입점

1. ~~**④-a 확정 여부**~~ — **확정 완료 (2026-05-30)**: grill 재논의에서 verb/noun·외부 확장 양축 개방·cold-hot 분리로 견고함 재확인 후 확정.
2. ~~**§4.1 정정 내용**~~ — **반영 완료 (2026-05-30)**. 적용된 내용:
   - 역할: "state mutation primitive base trait" → "**KV state 책임 base trait** (geometry 3 + mutation 3 + attention 1 = 7 method)"
   - `attention_into` 추가, `INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC` **유지**(호출자는 q→out 만 보므로 storage-format 무지)
   - `INV-KVCACHELAYER-PAIRED-KERNEL`: `attention_into` 가 "impl 과 paired kernel 매핑" 실체화
   - 두 축 직교성 명문화(storage Layer impl ⊥ policy Stage)
   - 연혁 blockquote 에 본 grill (2026-05-30) 기록 (소섹션 구조 컨벤션 = `feedback_arch_doc_subsection_structure.md`: 역할 먼저, 연혁 맨 아래)
3. **`attention_into` 정확한 시그니처** = #12 (Phase α-W/α-K impl detail). `dims: AttnDims`(n_heads_q/n_heads_kv/head_dim/cache_len) + `scores: Option<&mut [f32]>` + `&self` vs `&mut self`(γ interior mutability).
4. **score 도메인**(`needs_attn_scores`/`set_attn_scores`) = #17 Score domain refactor 별 sprint. **④ 범위 밖**(명시).

---

## 6. 변경 파일

**2026-05-30 ④-a 확정 + 용어 정리 반영** (코드 0, 문서만):
- `arch/pipeline_stage_design_v2.md §4.1`: 역할 재정의(state 책임 = geometry 3 + mutation 3 + attention 1 = 7 method) + `attention_into` trait 추가 + "왜 attention 이 Layer(noun)" 단락(두 축 직교성/verb-noun/cold-hot 분리) + 연혁 blockquote(2026-05-30 ④). `INV-KVCACHELAYER-PAIRED-KERNEL` 본문에 `attention_into` 실체화 1줄.
- `spec/41-invariants.md`: `INV-KVCACHELAYER-PAIRED-KERNEL` 에 `attention_into` 실체화 1줄 (normative 강화 아님 — arch 정합).
- `CONTEXT.md` (신규): KV/weight 용어집 — 저장 형태=Format(noun) ⊥ 관리 동작=Stage(verb), "Layer"는 transformer layer 전용.
- `.agent/todos/backlog.md`: [P2] Format 명명 통일 (`KVCacheLayer`/`WeightLayer` → `KVCacheFormat`/`WeightFormat`, 문서 전체 일괄 rename).
- memory `reference_kv_weight_glossary.md` + MEMORY.md.

---

## 7. ④-b 및 후속 (Phase α-K 등록)

본 grill 이 Phase α-K 로 미룬 항목 (전부 ADR-0001 Generic→dyn 전환과 동행):
- **④-b**: plan `AttentionVariant`/`KvUpdateVariant` enum 평탄화(`Vec<KernelStep>`) + layer `build_*_steps` 위임 + 동적 arg accessor(closure vs 작은 data enum — execute hot path 비용). plan 3중 분기 → 위임.
- **QuantizeHandler 실체화**: dead stub → "압력 신호 → StandardLayer K/V read → KIVILayer 마이그레이션 → layer swap" (마이그레이션 귀속 A).
- **KVLayerSlot RCU**: weight `LayerSlot::rcu_weights` 패턴 확장. 런타임 layer 교체.
- **paradigm_generation counter**: weight `ratio_generation` 패턴. KV paradigm 전환 시 plan invalidation 트리거.
- **EvictionPolicy dyn 화**: `&mut KVCache` → `&mut dyn KVCacheLayer`, base trait `compact` 만 호출 → policy 축이 모든 storage 에 직교 적용.

---

## 자기점검 (handoff-doc)

| 항목 | 확인 |
|---|---|
| 진입 문장 | "④ grill 재논의 — ④-a 확정 직전, §4.1 정정 대기" |
| 문제 배경(의도) | §1 (creep 실체 + 발생 원인 + ADR-0001 창문 + north star) |
| 도달 결론 + 미확정 구분 | §3(결론) / §5(미확정) 분리 명시 |
| stress test | §4 (런타임 활성화 + 마이그레이션 귀속 + 두 축) 전부 통과 |
| 변경 파일 | §6 (없음 — 설계만) |
| 후속 등록 | §7 (Phase α-K 5항목) |
| 자기완결성 | cold read 가능 (코드 경로·라인·근거 인용) |
