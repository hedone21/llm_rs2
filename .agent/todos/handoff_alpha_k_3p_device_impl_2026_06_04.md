# Handoff: α-K BC Step 3 ((3p) ④-a) 코드+host게이트 완료 → device 게이트(=acceptance)

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `659b130a` 설계 + (본 갱신 = 구현 커밋 예정)
**브랜치**: `master` (origin push 미실행 — 사용자 요청 시)
**다음 세션 진입 문장**: **"BC Step 3 device 게이트"** (device-capable 세션: S25 OpenCL + Jetson CUDA 필요 — 코드 완성, **게이트만** 잔여)

> 설계 SSOT = **`design_alpha_k_3p_cut_2026_06_04.md`**(workflow `wf_2be25cb8-bc9`, 3 design + 3 verify, §구현 정정 포함). roadmap = `roadmap_alpha_k_bc_completion_2026_06_04.md` Step 3. arch SSOT = `arch/pipeline_stage_design_v2.md` §9.1 line 742~746/761 + §4.1 연혁 ④. ADR §8.3 정정1/2. 트랙 = [[project-pipeline-alpha-k]].

---

## TL;DR
α-K BC **Step 3 ((3p) ④-a plan hot-path flip) 코드 작성 + host 게이트 완료** — host 세션. 설계(`design_alpha_k_3p_cut`, 적대검증이 build_plan 갭 적발) 후, **additive + `LLMRS_KV_FMT` 게이트 OFF**(production-default fmt-OFF byte-불변, (3c-fwd) 선례)로 4 표면 구현: `execute_fmt`(execute<C> copy-fork) + `StandardFormat::plan_geometry/plan_advance/plan_lock` + `build_plan_fmt`/`execute_plan_fmt` + ModelForward wiring(fmt-ON plan-on-handle→dyn 폴백, fmt-OFF 무변). 메인 세션이 **3 함수 기계적 diff = 문서화된 변경뿐**(byte-identical), 보호 함수 0 deletion, host 게이트 전부 GREEN 재검증. **다음 = device 게이트만**(코드 추가 없음): `LLMRS_KV_FMT=1` 로 5 KV × 32-tok **bit-identical** + **avg_tbt Δ≤+3%**. **왜 host 에서 멈췄나**: plan path=GPU-only(`try_build_plan` 비-OpenCL None) → host 에서 ④-a 미발화(fmt-ON 도 build_plan_fmt None→forward_into_fmt 폴백) → 기능·perf acceptance 가 **device 전용**. unverifiable hot-path 라 게이트 OFF 로 두어 production-default 안전(device 게이트 PASS 후 ON/기본화).

---

## 진행 상태
| 증분 | 상태 | 게이트 |
|---|---|---|
| Step 1 (B-2 prefill + B-4 eval cold) | ✅ | `2941edca` 등 host bit-identical |
| Step 2 (B-3 offload 분리) | ✅ | `936d0c99` host bit-identical |
| Step 3 설계 | ✅ | `659b130a` + workflow `wf_2be25cb8-bc9`(V1→흡수, V2/V3 confirmed) |
| **Step 3 구현 + host 게이트** | ✅ **완료** | 코드 commit(본 handoff 동행) — execute_fmt↔execute / build_plan_fmt↔build_plan / execute_plan_fmt↔execute_plan **기계적 diff=문서화 변경뿐**, 보호 함수 0 deletion, clippy/fmt clean, standard_format 16/16, 전체 lib 1235 pass/23 fail(전부 GPU부재) |
| **Step 3 device 게이트** | ★**다음** | **device-gate(full)**: 5 KV × 32-tok bit-identical + avg_tbt Δ≤+3% (S25 `opencl --opencl-rpcmem` + Jetson CUDA, `LLMRS_KV_FMT=1`, n≥5 median tok0-inclusive) |
| Step 4·5 | TODO | — |

---

## 다음 작업 (Step 3 device 게이트 — **코드 추가 없음, 게이트만**)
1. **device 빌드 + 배포** (S25 OpenCL, Jetson CUDA). 코드는 master 에 이미 있음(`execute_fmt`/`build_plan_fmt`/wiring, 게이트 OFF default).
2. **bit-identical**: `LLMRS_KV_FMT=1 legacy_generate -b opencl --opencl-rpcmem --greedy -n 32` (5 KV: Sliding/H2O/D2O/KIVI/SnapKV) 출력이 **OFF(=execute<C>)** 출력과 **token-id 완전 일치**. 단, plan path 발화 확인(`LLMRS_PLAN_TRACE` 로 execute_plan_fmt ok 카운트>0). reference baseline = legacy 출력.
   - **★주의(게이트 발화)**: fmt-ON 시 plan 이 발화하려면 (a) build_plan_fmt 가 None 반환 안 해야(GPU+F16/지원 dtype), (b) `--no-gpu-plan` **미사용**(plan 켜야). MIN_EVICT_TOKENS 등 eviction 발화 조건은 (3c-evict) 트랙(eviction unwired 라도 decode bit-identical 게이트 성립).
3. **avg_tbt Δ≤+3%**: ON vs OFF avg_tbt(n≥5 median, tok0-inclusive, cool-state). 회귀 +3% 초과 시 root-cause=StandardFormat getter layer당 Mutex lock(2/layer, vtable 아님). thermal 지배 주의(cool-state first-run).
- **권장 역할**: Tester (device-gate full + avg_tbt 실측). 코드 변경 불필요(PASS 시); 회귀 시 **(3p)만 revert**(execute_fmt/build_plan_fmt/wiring 제거, Step 1·2 cold cluster 유지, 전역 원칙 5).
- **★device 에서 특히 검증할 의미 변경 1건**: geometry **단일 lock 스냅샷**(`plan_geometry()` 1회)이 execute<C> 의 per-getter 호출과 bit-identical 인지(host 분석상 루프 본문 첫 mutation 이전 스냅샷이라 동일값, 단 device 실측으로 확정). 나머지는 순수 mechanical substitution.

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
- 진입 문장 한 줄? ✓ "BC Step 3 device 게이트"(device-capable 세션 — 코드 완성, 게이트만)
- 왜 멈췄나? ✓ 코드+host게이트 완료, acceptance(bit-identical+avg_tbt)=plan GPU-only 라 device 전용. 게이트 OFF default 라 production-default 안전(unverifiable flip 미발화).
- 최대 landmine? ✓ geometry 단일 lock 스냅샷 의미동등(device 확정) + 게이트 발화 조건(`LLMRS_KV_FMT=1` + `--no-gpu-plan` 미사용) + 회귀 시 (3p)만 revert
- 게이트 수치/명령? ✓ device 5 KV × 32-tok bit-identical(ON vs OFF) + avg_tbt Δ≤+3% (S25 `opencl --opencl-rpcmem` + Jetson CUDA, `LLMRS_KV_FMT=1`, n≥5 median tok0-inclusive)
- 길이 적정? ✓ 상세는 `design_alpha_k_3p_cut_2026_06_04.md`(§구현 정정 포함)
