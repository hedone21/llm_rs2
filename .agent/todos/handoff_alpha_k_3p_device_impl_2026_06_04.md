# Handoff: α-K BC Step 3 ((3p) ④-a) — device 게이트 ✅ COMPLETE → Step 4

**작성**: 2026-06-04 (메인 세션) / **갱신**: 2026-06-04 (device 게이트 세션, S25 실측)
**HEAD**: `ae9fc460` fix(non-opencl 빌드 복구) ← `004147bf` 구현 ← `659b130a` 설계
**브랜치**: `master` — **`ae9fc460` push 미실행** (사용자 승인 대기; ⚠️ origin/master 는 `004147bf` 에서 non-opencl 빌드 깨진 상태)
**다음 세션 진입 문장**: **"BC Step 5"** (legacy 폐기 + B-2 OLD-chain 잔여 migrate + KVCacheOps 삭제). **Step 3·4 모두 ✅ COMPLETE** (아래).

---

## ★ Step 3 device 게이트 결과 (2026-06-04 S25 실측) — PASS

**S25 Adreno OpenCL (`-b opencl --opencl-rpcmem`) = (3p) ④-a flip 의 완전·충분한 acceptance. PASS.**

| 게이트 | 결과 |
|---|---|
| **발화 확인** | F16 weight 모델(qwen2.5-1.5b-f16.gguf)에서 `LLMRS_KV_FMT=1` → "build_plan SUCCESS" + "KV_FMT ON" → `execute_plan_fmt` 발화 확정. KV dtype f16/f32/q4 전부 plan build SUCCESS. |
| **bit-identical** | f16/f32/q4 KV 3 dtype 전부 **ON(execute_plan_fmt) ≡ OFF(execute<C>)** 32-tok 생성 텍스트 완전 일치. post-fix(`ae9fc460`) 재확인도 3/3 일치. |
| **avg_tbt Δ≤+3%** | f16 KV ON/OFF 인터리브 n=7: ON median 59.61 / OFF 59.47 → **Δ +0.24%** (mean +0.53%). StandardFormat getter Mutex lock(56/tok)=무시 가능. vtable-free concrete-handle 설계 검증. |

### ★ 게이트 spec 정정 2건 (handoff 원안 오류 — 코드 ground-truth 로 정정)
1. **"5 KV: Sliding/H2O/D2O/KIVI/SnapKV" = 오류.** `is_standard_happy_path` 가 `eviction_policy()=="none"` 를 요구(`build_standard_loop.rs:60`) → eviction 켜면 happy path 이탈 → fallback decode loop(generic `execute_plan`, `LLMRS_KV_FMT` 무시) → ON≡OFF 가 vacuous → flip 미검증. **올바른 게이트 = no-eviction standard happy path × KV dtype(f16/f32/q4).** 발화엔 **F16 weight 모델 필수**(q4_0 weight → build_plan None(SOA q_img/qkv_bias) → dyn fallback, flip 미발화).
2. **"Jetson CUDA" = flip 미적용.** `execute_plan_fmt`/`build_plan_fmt`/`execute_fmt` + step() plan 분기 전부 `#[cfg(feature="opencl")]`; CUDA 백엔드(cuda_embedded/cuda_pc)엔 plan 경로 부재. Jetson(cuda-embedded, opencl 없음) 빌드는 flip 코드 통째 컴파일 제외 → fmt-ON 도 `forward_into_fmt`(dyn, =(3c-fwd)) 로 빠짐. **(3p) flip acceptance = S25 OpenCL 단독.**

### ★ Step 3 회귀 적발 + fix (`ae9fc460`)
- 발화 분석 중 발견: `004147bf` 가 `standard_format.rs::plan_geometry()` 에서 opencl-게이트 타입 `crate::backend::opencl::plan::PlanGeometry` 를 cfg 게이트 없이 참조 → **non-opencl 빌드(Jetson cuda-embedded 등) E0433 컴파일 실패**.
- fix: plan seam 3 메서드(plan_geometry/plan_advance/plan_lock) + unit test 3개 `#[cfg(feature="opencl")]` 게이트. opencl 빌드 no-op(16/16 PASS, S25 post-fix bit-identical 3/3). non-opencl `cargo check` PASS(구 FAIL).

### 기존 이슈 플래그 (out of scope — flip 무관)
- **f32 / q4 KV + F16 weight 모델 = degenerate 출력** ("Paris" 뒤 garbage/non-rendering). **OFF(untouched execute<C>)에도 동일** → flip 회귀 아님, 기존 production 이슈. f16 KV 만 coherent. 별도 트랙.

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
| **Step 3 device 게이트** | ✅ **PASS** | **S25 OpenCL** = 완전 acceptance. bit-identical f16/f32/q4 KV 3/3 (ON≡OFF) + avg_tbt Δ +0.24% median(n=7). 게이트 spec 2건 정정(eviction→KV dtype, Jetson N/A). 회귀 fix `ae9fc460`+`26e77908`. |
| **Step 4 (device-gate → argus_cli)** | ✅ **PASS** | argus_cli CLI 갭 0(Args+happy-path 공유). S25: argus OFF≡legacy OFF 3/3(baseline 연속성) + argus ON≡OFF 3/3(fmt 게이트) + execute_plan_fmt 발화 확인 + avg_tbt Δ+0.10%. 게이트 명령=roadmap Step 4 §canonical. |
| Step 5 | ★**다음** | legacy 폐기 + B-2 OLD-chain 잔여(forward_into_offload/run_chunked_prefill) fmt 이주 + KVCacheOps trait 삭제. ★argus 는 happy-path 전용 → 비-happy 모드 family bin 이주/drop 결정 선결. |

---

## 다음 작업 (Step 5 — legacy 폐기 + KVCacheOps trait 삭제)
> Step 3(flip 정확/perf) + Step 4(argus_cli 게이트 매체 등가) PASS 로 BC 의 cold/hot flip + 게이트 이주가 모두 완료. 남은 것은 OLD path(`KVCacheOps`) 소비자 0 만들고 trait 삭제(roadmap Step 5 SSOT).
1. **비-happy 모드 거취 결정** (Architect/사용자): argus_cli 는 happy-path 전용(eviction/KIVI/offload/swap/profile/batch reject). legacy 폐기 전 이 모드들을 argus-chat/argus-eval/argus-bench family bin 으로 이주할지, 일부 drop 할지 결정. **이게 legacy 폐기의 실질 선결**(device-gate 매체는 Step 4 로 이미 이전).
2. **B-2 OLD-chain 잔여 2 소비자 fmt 이주** (roadmap Step 5 §★): `forward_into_offload`(OffloadForward, chat/session.rs:547) + `run_chunked_prefill`(prefill.rs profiler/variance). 둘 다 OLD `forward_gen<C>`/`forward_prefill<C>` + `impl KVCacheOps for OffloadKVCache` 소비 → Option B(`forward_gen_fmt`+`OffloadKVCache: KVCacheFormat` interior-mut+preload aliasing 재설계, **device GPU 재검증 필수**).
3. **`KVCacheOps` trait 삭제**: `grep -r KVCacheOps engine/` 0건 확인 후 trait + generic bound + use 삭제 → `KVCacheFormat` rename 동행(ADR-0001). **device-gate full = 진짜 최종 perf**(parallel path 제거 후 monomorphization 드러남).
- **권장 역할**: Architect(비-happy 모드 거취 + KVCacheOps 잔존 census) → Senior Implementer(offload fmt 이주 — GPU/aliasing 민감) → Tester(device-gate 최종).
- **★Step 3/4 회귀 시 처방**: (3p)만 revert(`execute_fmt`/`build_plan_fmt`/`execute_plan_fmt`/wiring + standard_format plan seam 제거), Step 1·2 cold cluster 유지. fix `ae9fc460`(plan seam cfg)+`26e77908`(map_weights cfg-free)는 빌드 복구라 유지.

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
