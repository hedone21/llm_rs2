# Handoff: α-K BC ①-b 완료 → ①-c (eval flip)

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `2bf5c500 feat(kv): Phase α-K BC ①-b — forward_into_fmt multi-token prefill flip (host)`
**브랜치**: `master` (origin push 미실행 — 사용자 요청 시)
**다음 세션 진입 문장**: **"BC ①-c 진행"**

> roadmap = `.agent/todos/roadmap_alpha_k_bc_completion_2026_06_04.md`(Step 1, ①-b ✅/①-c 다음). 설계+적대검증 = `.agent/todos/design_alpha_k_1b_cut_2026_06_04.md`(workflow wfceex20u). SSOT = `arch/pipeline_stage_design_v2.md` §9.1-BC1'. 트랙 = [[project-pipeline-alpha-k]].

---

## TL;DR
α-K BC Step 1 의 **①-b(forward_into prefill flip) 완료** — host + **S25 device 게이트 3 dtype bit-identical** PASS. 다음 = **①-c(B-4 eval flip)**: `run_eval_ll_generic<C: KVCacheOps>` + `StepHook<C>`/`CacheSnapshot<C>` 의 KVCache/KiviCache 런타임 다형성을 fmt-cache/enum 으로 전환(forward_into flip 이후 thin follow-on). **왜 멈췄나**: ①-b 가 device 게이트까지 종결된 clean checkpoint, ①-c 는 eval 다형성(KiviCache 포함) 설계가 필요한 별도 증분.

---

## 진행 상태 (검증 게이트)
| 증분 | 상태 | commit | 게이트 |
|---|---|---|---|
| ①-a phantom / C3 host | ✅ | `2e6b50fb` | host |
| **①-b prefill flip** | ✅ **host+device** | `2bf5c500` | host(build+test 13/13+회귀0+clippy) + **S25**: F16/Q4_0/F32 3 dtype `--no-gpu-plan` bit-identical + avg_tbt Δ≤+3% |
| ①-c eval flip | ★다음 | — | host + legacy EvalOutput JSON bit-identical |
| ①-d 비-decode 잔여(warmup/qcf/batch/ppl) | 대기 | — | host 회귀 0 |

①-b device 실측(Qwen2.5-1.5B, prompt 15 tok, n=32 greedy): F16(rpcmem) TTFT Δ+0.17%/TBT −1.9% · Q4_0(rpcmem) Δ−3.1%/+0.15% · F32(device-only) +1.4%/+0.6%. 전부 first token 49689 + 텍스트 일치.

---

## 다음 작업 (①-c)
1. **Architect 설계** → `run_eval_ll_generic<C: KVCacheOps>`(eval_loop.rs:45, forward_into 7 call site)의 `C` 다형성 census. **핵심 난점**: eval 은 KVCache **와** KiviCache 둘 다 런타임 다형(`StepHook<KVCache>` eviction_hook.rs:217 / `StepHook<KiviCache>` kivi_hook.rs:137 / `CacheSnapshot<C>`). ①-b 의 `forward_into_fmt` 는 prefill+decode 둘 다 fmt 지원하나, eval 의 multi-token prefill(`run_full_prefill`/`run_chunked_prefill` eval_loop.rs:748/826)을 fmt entry 로 전환 + StepHook/snapshot 의 C 를 trait object 또는 enum 으로. KiviCache → `KIVIFormat`(이미 존재) 경유 가능한지 확인.
2. **Implementer 구현** → host build/test/fmt/clippy.
3. **검증 게이트** → host + legacy `run_eval_ll` EvalOutput JSON(ppl/nll) bit-identical, KVCache·KiviCache 각 1회.

---

## Landmines / 미해결 (R6)
- **①-b additive-fork 중복**: `StandardFormat::prefill_attention`(standard_format.rs) + `forward_prefill_fmt.rs` 는 `forward_prefill`(forward.rs:259-585)을 미러한 **중복 코드**. forward_prefill 무수정(byte-불변). 중복은 **Step 5(forward_prefill<C> 삭제)에서 자연 해소** — 그때까지 두 경로 divergence 주의(둘 중 하나 수정 시 다른 쪽 동기화 필요).
- **plan+Q4_0 KV 는 별개 garbage**: ①-b 게이트 중 gate-off **plan** 경로(--no-gpu-plan 없이)가 Q4_0 KV 에서 garbage 출력 확인(`Jupiter::~10-private...`). 이는 **선재 plan+Q4 이슈**(내 변경 무관) — 게이트는 `forward_into`(--no-gpu-plan) 기준으로 비교해야 정확. ①-c/Step3 에서 plan+Q4 만나면 이 사실 상기.
- **F32 host-mapped decode 는 NOT bit-identical**(3c 캐비엇): F32+rpcmem 의 **decode** 는 forward_gen inline-NEON vs attention_gen 누산차로 불일치. ①-b 는 F32 를 **device-only**(`--backend opencl` no rpcmem)로 게이트해 우회. prefill F32 자체는 bit-identical(flash 경로 공유). eval F32 게이트도 device-only 또는 F16/Q4 사용 권장.
- **Jetson CUDA prefill 미측정**: optional follow-on(같은 `flash_attention_prefill` trait method). S25 가 GPU-flash+CPU-fallback+e2e 커버. Step 3(hot flip) 전 1회 확인 권장.
- **cargo authoritative**: subagent 자기보고/IDE 진단 불신, 메인이 `cargo build/test/clippy` 직접 재검증.
- **커밋 금지 untracked**: `.antigravitycli/`·`.claude/scheduled_tasks.lock`·`papers/.../microbench_*`·`.agent/todos/handoff_microbench_*`·`arch/pipeline/`. 명시 파일만 add(`git add -A` 금지).

---

## 자기점검
- 진입 문장 한 줄? ✓ "BC ①-c 진행"
- 왜 멈췄나? ✓ ①-b device 게이트 종결 clean checkpoint, ①-c=eval 다형성(KiviCache) 별도 설계 증분
- 최대 landmine? ✓ eval 의 KVCache+KiviCache 이중 다형성(StepHook<C>) + additive-fork 중복(Step5 해소)
- 검증 게이트 수치/명령? ✓ host(13/13 등) + device(3 dtype bit-identical, --no-gpu-plan, 명령 roadmap line 33-36)
- 길이 적정? ✓ 상세는 roadmap/design 링크
