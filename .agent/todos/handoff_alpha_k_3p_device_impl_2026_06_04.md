# Handoff: α-K BC Step 3 ((3p) ④-a) 설계 완료 → device 구현·게이트

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `89272861` (+ 본 설계 docs 커밋 예정)
**브랜치**: `master` (origin push 미실행 — 사용자 요청 시)
**다음 세션 진입 문장**: **"BC Step 3 device 구현"** (device-capable 세션: S25 OpenCL + Jetson CUDA 필요)

> 설계 SSOT = **`design_alpha_k_3p_cut_2026_06_04.md`**(workflow `wf_2be25cb8-bc9`, 3 design + 3 verify). roadmap = `roadmap_alpha_k_bc_completion_2026_06_04.md` Step 3. arch SSOT = `arch/pipeline_stage_design_v2.md` §9.1 line 742~746/761 + §4.1 연혁 ④. ADR §8.3 정정1/2. 트랙 = [[project-pipeline-alpha-k]].

---

## TL;DR
α-K BC **Step 3 ((3p) ④-a plan hot-path flip) 설계 완료** — host 세션. ④-a 형태(plan-local `PlanCacheHandle` 최소 trait + concrete-handle monomorphize, vtable 0), flip 표면 4개, perf 분석, ④-b defer, (3d) plan-eval=flip 확정을 `design_alpha_k_3p_cut` 에 박았다. **적대검증이 구현 전에 갭 1건 적발**(V1: `build_plan` 의 KVCache buffer 직접 접근 누락 → flip 표면에 흡수). **다음 = device 구현**: 설계 spec 대로 코드 flip + **5 KV × 32-tok bit-identical + avg_tbt Δ≤+3%** device 게이트. **왜 host 에서 멈췄나**: plan path = GPU-only(`try_build_plan` `backend.name()!="OpenCL"` None) → host(GPU 부재)에선 ④-a 코드 미발화 → 기능·perf acceptance 가 **device 전용**. host 는 설계까지가 책임 한계(unverifiable hot-path flip 커밋 금지).

---

## 진행 상태
| 증분 | 상태 | 게이트 |
|---|---|---|
| Step 1 (B-2 prefill + B-4 eval cold) | ✅ | `2941edca` 등 host bit-identical |
| Step 2 (B-3 offload 분리) | ✅ | `936d0c99` host bit-identical |
| **Step 3 설계** | ✅ **완료** | `design_alpha_k_3p_cut_2026_06_04.md` + workflow `wf_2be25cb8-bc9`(V1 needs-revision→흡수, V2/V3 confirmed) |
| **Step 3 device 구현** | ★**다음** | **device-gate(full)**: 5 KV × 32-tok bit-identical + avg_tbt Δ≤+3% (S25+Jetson) |
| Step 4·5 | TODO | — |

---

## 다음 작업 (Step 3 device 구현) — `design_alpha_k_3p_cut` §핵심 변경 4개
1. **PlanCacheHandle trait 신설**(`plan/cache_handle.rs`): `plan_geometry()`(4 getter 1 lock 스냅샷) + `plan_advance(&self,n)` + `plan_kv_bufs(read seam)`. → 검증: host build/clippy/fmt clean + StandardFormat/KIVIFormat unit test(getter 위임).
2. **StandardFormat/KIVIFormat inherent + impl** (base trait 무변, `INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC`). Standard res_pos/q2_tokens=0. → host unit test.
3. **`execute_fmt<H: PlanCacheHandle>`**(generic body monomorphize, legacy `execute<C>` 는 rename 후 Step 5 까지 co-exist) + **★`build_plan_fmt`**(V1 갭 — `:2577/2760` KVCache pub buffer → StandardFormat read seam) + **ModelForward wiring**(fmt ON 분기가 plan-on-fmt-handle 먼저 시도 → forward_into_fmt fallback; 현 :459~528 상호배타 해소). → 검증: **device** 5 KV × 32-tok bit-identical + avg_tbt Δ≤+3%.
- **권장 역할**: Senior Implementer(plan.rs hot flip — GPU/lock 민감) → Tester(device-gate full + avg_tbt 실측).
- **회귀 시 (3p)만 revert**, Step 1·2 cold cluster 유지(전역 원칙 5). avg_tbt Δ>+3% AND lock-cost 실측 = perf revoke(ADR §6.5).

---

## Landmines / 미해결 (R6) — 상세 `design_alpha_k_3p_cut` §Landmines
- **★build_plan 동반 flip**: execute flip 단독 아님. build_plan_fmt + StandardFormat buffer-read seam 까지 device 범위(V1 적대검증 적발, 소스 확정 `:2344/2577/2578/2760`).
- **symbol 충돌**: `execute<H>`+`execute<C: KVCacheOps>` 동명 generic 공존 불가 → 신 production=`execute_fmt`/`execute_plan_fmt`/`build_plan_fmt`, legacy 동명 유지(Step 5 삭제 시 `_fmt`→canonical rename).
- **KIVI plan = legacy 전용**(`engine/src` 호출 0, `generate.rs:4287`) → Step 3 production 범위 밖. KIVIFormat plan_geometry 단일 lock 스냅샷(current_pos=q2_tokens+res_pos 파생) INV 는 KIVI 흡수 시.
- **Step 3 ≠ KVCacheOps 삭제**(과장 경계): B-1 단독 해소. B-2 full-surface·B-4·legacy 잔존. 삭제=Step 5.
- **fmt OFF=plan 무변 tripwire**: device avg_tbt Δ≈0 확인(generic monomorph 미접촉).
- **cargo authoritative** / 커밋 금지 untracked(`arch/pipeline/`·`.antigravitycli/`·microbench_* 등, 명시 파일만 add) / push 사용자 요청 시.

---

## 자기점검
- 진입 문장 한 줄? ✓ "BC Step 3 device 구현"(device-capable 세션)
- 왜 멈췄나? ✓ plan GPU-only → acceptance device 전용, host 설계 한계(unverifiable flip 커밋 금지)
- 최대 landmine? ✓ build_plan 동반 flip(V1 갭) + symbol 충돌 + KIVI legacy 범위
- 게이트 수치/명령? ✓ device 5 KV × 32-tok bit-identical + avg_tbt Δ≤+3% (S25 `opencl --opencl-rpcmem` + Jetson CUDA, n≥5 median tok0-inclusive)
- 길이 적정? ✓ 상세는 `design_alpha_k_3p_cut_2026_06_04.md`
