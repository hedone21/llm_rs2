# 설계 cut: α-K BC Step 3 — (3p) ④-a plan hot-path flip (concrete-handle)

**작성**: 2026-06-04 (메인 세션 오케스트레이터)
**설계+적대검증**: workflow `wf_2be25cb8-bc9` (3 design lens[architect: form / wiring·perf / scope·split] → 3 adversarial verify[architect: completeness / perf·정확성 / scope·host-device]). **V1=needs-revision (build_plan 누락 blocking, 아래 §3 흡수 완료)**, V2/V3=confirmed-with-notes.
**SSOT**: `arch/pipeline_stage_design_v2.md` §9.1 line 730/742~746/758~761 + §4.1(연혁 ④/R4). **ADR**: `docs/adr/0001-kv-dispatch-paradigm.md` §8.3(정정1/2) + §6.5.
**선행**: Step 1 ✅(B-2 prefill `2bf5c500` + B-4 eval `1e4f20fe`) · Step 2 ✅(B-3 offload `936d0c99`). roadmap `roadmap_alpha_k_bc_completion_2026_06_04.md` Step 3.

> **본 문서는 설계 spec — 구현·게이트는 device 라운드(별도 세션).** Step 3 = production GPU decode hot path flip 이라 acceptance(5 KV × 32-tok bit-identical + avg_tbt Δ≤+3%)가 **device 전용**(S25 OpenCL + Jetson CUDA). host 세션(GPU 부재)에선 plan path 미발화 → 본 설계는 device 세션이 구현하는 device-ready spec.

---

## 결정 요약

**④-a 형태 = plan-local 최소 trait `PlanCacheHandle` + concrete-handle monomorphize (vtable 0).** dyn trait object flip 아님. 핵심:

```rust
// 위치: engine/src/backend/opencl/plan/cache_handle.rs (no-mod.rs, plan 모듈 소유, KVCacheOps 비의존)
pub(crate) trait PlanCacheHandle {
    fn plan_geometry(&self) -> PlanGeometry;      // 레이어 진입부 1 lock 으로 4 getter 묶음
    fn plan_advance(&self, n: usize);             // post_ffn 후 &self + interior-mut lock
    fn plan_kv_bufs(&self, f: &mut dyn FnMut(&KVCache));  // build_plan buffer-read seam (§3, V1)
}
pub(crate) struct PlanGeometry { current_pos, capacity, res_pos, q2_tokens }
```

- **execute 본문 DRY 보존**: `execute<C: KVCacheOps>(&mut [C])` → 신 production 진입점 `execute_fmt<H: PlanCacheHandle>(&self, backend, start_pos, &[H])` 로 generic body 그대로 monomorphize. 본문 600 LOC 복제 0. legacy `execute<C: KVCacheOps>` 는 **Step 5 까지 co-exist**(rename 으로 symbol 충돌 회피 — 아래 §4).
- **concrete-handle**: production 은 `&[Arc<StandardFormat>]` 전달 → `execute_fmt::<StandardFormat>` monomorphize → **vtable 0** (static dispatch). `Arc<dyn>` 아님.
- **perf = neutral-or-slightly-worse** (cleanup, gain 아님): 현 production `execute::<KVCache>` = lock 0 + vtable 0 = perf-optimal. ④-a 는 vtable 0 유지하나 getter/advance 의 layer당 Mutex lock 추가. bundled-getter 로 **2 lock/layer**(plan_geometry 1 + plan_advance 1) 최소화. S25 16 layer × 2 = 32 lock/tok, uncontended std::sync::Mutex(~수십 ns) ⇒ TBT 대비 <0.01% 예상 — **단 device 실측 필수**(Adreno/Jetson futex 실비용 미측정).

**범위 — production Standard 단독. KVCacheOps 폐기 아님.**
| 항목 | Step 3 처리 | 근거 |
|---|---|---|
| B-1 production plan path (Standard) | ✅ flip (execute_fmt + build_plan_fmt + wiring) | production hot = StandardFormat 단독 |
| `execute<C: KVCacheOps>` 본체 | **유지(Step 5 삭제)** | legacy KIVI(`execute_plan_for_kivi`) + legacy KVCache(`generate.rs:2024`) co-exist |
| KIVI plan (`execute_plan_for_kivi`) | **defer(legacy 전용)** | `engine/src` production 호출 0건 (grep), `generate.rs:4287` 단독 |
| ④-b (AttentionVariant 평탄화) | **defer(friction-triggered)** | attention 은 enum static, C 미접촉(아래 §5) |
| KVCacheOps 완전 삭제 (B-2/B-4/legacy 잔존) | **Step 5** | Step 3 = B-1 단독 해소 (과장 금지 — Step 2 V3 정신) |

---

## 핵심 변경 (flip 표면 — 4개)

### 1. `plan.rs::execute` 본문 (C-접촉 6 지점)
현 C 소비 표면(census + V1/V2 재확인, K/V 데이터 접근 0):
- `:1286` entry check `cache.capacity()`/`cache.current_pos()` (PlanInvalidated)
- `:1290~1296` `current_pos`(×2: cache_seq_len+write_pos)·`capacity`·`res_pos`(rp)·`q2_tokens`(q2t)·rt=rp 6 i32 로컬 1회 추출
- `:1828` `cache.advance_pos(1)` (post_ffn 후, 레이어 끝)
- `:1851` `kv_caches[0].current_pos()` (gpu_score end_step, **advance-후** post-advance pos, 루프 밖 1회)

flip: `cache=&mut kv_caches[i]` → `handle=&handles[i]`. entry check + 6 i32 추출을 **`let g = handle.plan_geometry()` 1회**(단일 lock 스냅샷)로 통합 — 현 체크/추출이 별 statement 라 1 lock 결과를 둘 다에 재사용하도록 미세 재배치(동작 bit-identical). `:1828` → `handle.plan_advance(1)`. `:1851` → `handles[0].plan_geometry().current_pos`.

### 2. StandardFormat / KIVIFormat — 신규 inherent (base trait 무변)
`KVCacheFormat` 7 method 에 추가 **금지**(`INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`, §4.1 R4 ④ KIVI creep). concrete inherent 로만:
- `StandardFormat`: `plan_geometry()` = `with_cache(|c| PlanGeometry{ current_pos:c.current_pos(), capacity:c.capacity(), res_pos:0, q2_tokens:0 })` (standard 는 residual/q2 부재 → 0). `plan_advance(n)` = `with_cache_mut(|c| c.advance_pos(n))` (기존 seam :62 재사용). `plan_kv_bufs(f)` = read seam(§3).
- `KIVIFormat`: 동일 패턴, res_pos/q2_tokens = KiviCache 실값 위임. **단 KIVI plan = legacy 전용 → Step 3 production 범위 밖, Step 5/별 device round defer**(V2 minor: KiviCache `current_pos=q2_tokens+res_pos` 파생이라 plan_geometry 가 반드시 단일 lock 스냅샷이어야 일관 — production 무관, KIVI 흡수 시 INV 명문화).
- **PlanGeometry 단일 lock 스냅샷 계약**을 doc 에 명시(V2 minor fix).

### 3. ★`build_plan` 동반 flip — V1 적대검증이 잡은 누락 (소스 확정)
**3 design 에이전트 모두 누락. V1(완전성)이 blocking 으로 적발 → 메인 세션 소스 확정**:
- `build_plan`(transformer.rs:2344)이 plan **생성** 시 KVCache `pub` field 직접 접근:
  - `:2577` `k_cache: cl!(kv_caches[i].k_buffer)` / `:2578` `v_cache: cl!(kv_caches[i].v_buffer)` (모든 plan, GPU cl_mem 을 KvBufs 로 pre-bind)
  - `:2760~2761` hybrid_attn: `c.k_buffer.buffer().as_ptr()` / `c.v_buffer.buffer().as_ptr()` (host ptr 추출)
- `KVCache.k_buffer`/`v_buffer` = `pub`(kv_cache.rs:42~43). StandardFormat 은 `Mutex<StandardFormatInner{cache}>`(standard_format.rs:38~41) 로 감싸므로 **`.k_buffer` 직접 도달 불가** → lock guard 통과 seam 필수.
- **fix**: build_plan 도 fmt 핸들 기반 진입점(`build_plan_fmt(&[Arc<StandardFormat>])`)으로 동반 flip + StandardFormat `plan_kv_bufs`(또는 `with_cache` read closure)로 cl_mem/ptr 추출. build_plan 은 decode 첫 step lazy 1회 → **lock 비용 perf 무영향**, 단 시그니처/seam 변경 작업량은 ④-a 범위. (try_build_plan model_forward.rs:207~213 도 동반.)

### 4. ModelForward::step wiring — fmt/plan 상호배타 해소 (3p 본질)
현재(model_forward.rs): fmt ON(:459~487) → `forward_into_fmt`(dyn 폴백) **early return**, plan path(:489~528) 우회. **즉 "fmt 간다 = 느린 dyn 폴백 간다".** prefill `ensure_fmt_wrapped`(:247~256)가 `mem::take(kv_caches)` → `fmt_caches: Vec<Arc<StandardFormat>>` move ⇒ fmt active 시 `kv_caches` 빈 Vec → plan 의 `&mut Vec<KVCache>`(:508) 무효.
- **flip**: fmt ON 분기를 재배선 — `fmt_caches` 의 `&[Arc<StandardFormat>]` 를 `execute_plan_fmt`(execute_fmt 래퍼)에 먼저 전달 → `Ok(true)` 면 logits 반환 / `Ok(false)`(PlanInvalidated)·build 실패 시에만 `forward_into_fmt`(dyn) fallback. plan 이 `&self` 핸들만 쓰므로(getter/advance interior-mut) 현 `gpu_plan.take()` borrow-juggling(:500) 제거 가능(미세 단순화 — 별 cleanup 분리 가능).
- **단일 물리 캐시 공유**: wrap 후 1벌 캐시가 `Arc<StandardFormat>` interior-mut 로 존재 → plan(`&[Arc]` slice borrow)·forward_into_fmt(`Arc::clone`)이 동일 캐시 공유, dual-ownership 부재(ADR §4.2).
- **transformer.rs::execute_plan**(:2842) 시그니처 `&mut [KVCache]` → fmt 핸들. legacy `execute_plan` 호출처(generate.rs:2024)는 KVCache concrete 유지 → **신 메서드 분기(execute_plan_fmt)** 권고, legacy 무변.

---

## 게이트 (acceptance = device 전용)

| 게이트 | 값 | 매체 |
|---|---|---|
| 기능 bit-identical | 5 KV 구성(Sliding/H2O/D2O/KIVI/SnapKV) × 32-tok token-id 완전일치 (frozen baseline=legacy_generate) | **device** (plan GPU-only) |
| perf | avg_tbt **Δ≤+3%** (n≥5 median, tok0-inclusive) | **device** (S25 OpenCL `--opencl-rpcmem` + Jetson CUDA) |
| surprise tripwire | fmt OFF=plan 무변 (generic monomorph execute::<KVCache> 미접촉) → avg_tbt Δ≈0 | device |
| host (보조) | build + clippy(`--workspace -D warnings`) + fmt clean + scaffold unit test(getter 위임) + 비-plan 경로 무회귀 | host |

- **회귀 시 (3p)만 revert**, Step 1·2 cold cluster 정리 유지(전역 원칙 5). avg_tbt Δ>+3% **AND** lock-cost 실측 = perf revoke trigger(ADR §6.5). 격리 microbench 폐기(vtable 0 = 측정 대상 부재).
- host 검증 불가 근거: `try_build_plan`(model_forward.rs:195) `backend.name()!="OpenCL"` 시 None → plan GPU-only. host(CpuBackend)는 plan 미빌드 → forward_into_fmt/forward_into 폴백만 실행 → ④-a 코드 경로 미발화.

---

## 확정 결정 (open question 해소)

1. **(3d) plan-eval = flip 확정** (선결 충족). SSOT line 761 BC 결정이 "③ (3p) ④-a hot-path flip" 으로 답. ①-b(`2bf5c500` S25 device PASS)로 prefill-fmt 정합 실증. **(3p) 의 실작업 = step() fmt/plan 상호배타 해소**(§4). World A eviction 발화 seam(F1)은 (3c-evict) device 게이트 트랙 — **Step 3 hard 선결 아님**(decode_loop try_evict 호출 0건, NoOpEvictionStage → eviction unwired 로도 decode happy-path bit-identical 게이트 성립).
2. **④-b defer 확정**: AttentionVariant 5 variant(plan.rs:1378~1532) 모두 dispatch_step 에 **i32 스칼라만** 전달, attention_into·C·K/V 데이터 0 접근 → ④-a(getter+advance)와 데이터 결합 부재 → ④-b 불요. 묶으면 hot-path 커널 라우팅 재구조화가 ④-a lock 과 무관한 perf 위험 유입 + revert 격리 불가(SRP). friction 미발동.
3. **host scaffold = device 라운드 동행**(독립 land 안 함): inherent getter(메서드 3×2 impl) + PlanCacheHandle trait + read seam 은 additive·unwired 라 host-landable 하나 LOC 작고 plan GPU-only 라 의미있는 독립 checkpoint 아님 + unwired orphan = `dead_code`(-D warnings) 위험 + (3p) revert 격리는 전체를 1 device 증분으로 둘 때 최선 → **device 세션이 scaffold+flip+wiring+게이트를 한 묶음으로 구현**(Agent C + V3 합의).

---

## Landmines / 미해결 (R6)

- **★build_plan 동반 flip(§3)** — 설계 3 에이전트가 놓쳐 V1 이 잡은 갭. execute flip 단독 아님 — build_plan_fmt + StandardFormat buffer-read seam 까지 device 범위. 누락 시 fmt 핸들에서 `.k_buffer` 접근 컴파일 불가.
- **symbol 충돌**(§4): `execute<H>` + `execute<C: KVCacheOps>` 동명 generic 공존 불가(Rust overlapping) → 신 production = `execute_fmt`, legacy `execute` 유지(Step 5 삭제 시 `execute_fmt`→`execute` rename). 동일하게 execute_plan/build_plan 도 `_fmt` 분기.
- **KIVI 일관성**(V2 minor): KiviCache `current_pos=q2_tokens+res_pos` 파생 → plan_geometry 단일 lock 스냅샷 필수. production 무관(KIVI=legacy)이나 KIVI 흡수 시 INV.
- **try_evict census 정정**(V3 minor): "decode_loop try_evict 0건"은 **World A DecodeLoop 한정** 정확. 전역으론 chat/session.rs:195/287 호출(단 chat=fmt_eligible=false → fmt/plan 상호배타 → production 범위 밖). ④-a 가 try_evict 회계 미파손.
- **Step 3 ≠ (4)**(과장 경계, Step 2 V3 정신): B-1 단독 해소. B-2 full-surface(prefill 포함)·B-4 eval·legacy 잔존. KVCacheOps 삭제 = Step 5.
- **cargo authoritative**: subagent/IDE 진단 불신. host build/clippy/fmt 메인 세션 재검증. RSS 테스트 병렬 flaky → 단독 재확인. opencl 테스트 host fail = GPU 부재(비-회귀).
- **커밋 금지 untracked**: `arch/pipeline/`·`.antigravitycli/`·`.claude/scheduled_tasks.lock`·`papers/.../microbench_*`·`.agent/todos/handoff_microbench_*`. 명시 파일만 add. push 사용자 요청 시.

---

## 자기점검
- ④-a 형태 결정? ✓ PlanCacheHandle 최소 trait + concrete-handle monomorphize(vtable 0) + bundled-getter 2 lock/layer
- flip 표면 전수? ✓ execute(6 지점) + StandardFormat/KIVIFormat inherent + **build_plan(V1 갭)** + ModelForward wiring(fmt/plan 통합)
- 적대검증 흡수? ✓ V1 build_plan blocking → §3 / V2 KIVI 스냅샷·perf confirmed / V3 scope·host-device confirmed
- host vs device? ✓ host=설계(완료)+scaffold(device 동행) / device=구현+5KV×32tok bit-identical+avg_tbt Δ≤+3%
- 과장 경계? ✓ Step 3=B-1 단독, KVCacheOps 삭제=Step 5
