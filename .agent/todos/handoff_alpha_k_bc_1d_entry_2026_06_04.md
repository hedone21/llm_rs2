# Handoff: α-K BC ①-c 완료 → ①-d (B-2 비-decode 잔여 flip)

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `1e4f20fe feat(kv): Phase α-K BC ①-c — eval flip to forward_into_fmt (KVCacheOps 바운드 제거)`
**브랜치**: `master` (origin push 미실행 — 사용자 요청 시)
**다음 세션 진입 문장**: **"BC ①-d 진행"**

> roadmap = `roadmap_alpha_k_bc_completion_2026_06_04.md`(Step 1: ①-a/b/c ✅, ①-d 다음). 설계+적대검증 = `design_alpha_k_1c_cut_2026_06_04.md`(workflow `wdrcgtqwz`). SSOT = `arch/pipeline_stage_design_v2.md` §9.1-BC1'(line 794-795). 트랙 = [[project-pipeline-alpha-k]].

---

## TL;DR
α-K BC Step 1 의 **①-c(eval flip) 완료** — eval 의 `KVCacheOps` 바운드 제거, host 게이트 PASS. 다음 = **①-d(B-2 비-decode 잔여)**: `forward_into<C: KVCacheOps>` 의 잔존 비-decode 소비자(warmup / qcf_runtime / batch/runner / ppl/runner)를 fmt entry(`forward_into_fmt`)로 전환. **왜 멈췄나**: ①-c 가 host 게이트까지 종결된 clean checkpoint, ①-d 는 별도 호출처군(grep 으로 census 먼저).

---

## 진행 상태 (검증 게이트)
| 증분 | 상태 | commit | 게이트 |
|---|---|---|---|
| ①-a phantom / C3 | ✅ | `2e6b50fb` | host |
| ①-b prefill flip | ✅ host+device | `2bf5c500` | host + S25 3 dtype bit-identical |
| **①-c eval flip** | ✅ **host** | `1e4f20fe` | host(build+test+fmt+clippy) + legacy `--eval-ll` BEFORE/AFTER |
| ①-d 비-decode 잔여 | ★다음 | — | host + 회귀 0 |

①-c host 게이트 실측(legacy `--eval-ll`, CPU, BEFORE=forward_into vs AFTER=forward_into_fmt):
- **KVCache F16 basic**(prefill+choice decode): EvalOutput JSON **bit-identical**.
- **KVCache F16 H2O**(probe+score-feed): **bit-identical** (logits 동일 ⇒ ws.scores 동일 ⇒ 누적 score 동일 ⇒ eviction 선택 동일).
- **KIVI Q2 short / long+AWQE+flush**: flush_count/q2_tokens/res_pos/predicted **정수회계 완전일치**(Verify 3 AWQE fix 확인). nll Δ~1e-6~2e-4 = ★2 carve-out.

---

## 다음 작업 (①-d)
1. **census**: `grep -rn "forward_into\b" engine/src/` 로 `forward_into<C: KVCacheOps>` 잔존 호출처 census. 예상 = `session/qcf_runtime`(warmup/swap), `session/batch/runner`, `session/ppl/runner`(run_ppl/run_kivi_ppl), 그 외 warmup. **decode/prefill 단일 토큰·배치**는 ①-b/①-c 패턴(fmt round-trip 또는 ModelForward)으로 전환.
2. **forward_into_fmt 충분성 확인**: 각 호출처가 넘기는 args(variance_collector/profiler/layer_boundary_hook/prefill_workspace)가 `forward_into_fmt` 에 없으면 — eval 처럼 추가 필요한지 vs 해당 호출처가 실제 None 인지 census. (①-c 는 score/skip/importance 만 추가; variance/profiler/layer_hook/prefill_ws 는 eval 이 None 이라 미추가.)
3. **검증**: host build + `cargo test` + 해당 경로 출력 bit-identical(가능 시). ppl 은 EvalOutput/PPL 수치 일치.

---

## Landmines / 미해결 (R6)
- **★2 carve-out (KIVI 포함)**: `KiviCache::get_view`=F32 → CPU host 에서 forward_gen 은 inline-flash(`forward_gen.rs:554+`, `use_typed_attn=false`), `attention_into`(StandardFormat/KIVIFormat)는 attention_gen 위임 → host NOT bit-identical(~1e-6, FP 누산순서). F16/Q4_0/device-only 는 bit-identical. **①-d 의 ppl/batch 게이트도 F16/Q4_0 KV 권장**(F32 host 제외).
- **`forward_into_fmt` additive-fork 중복 증가**: ①-c 가 score-feed/importance/end_step 미러를 `forward_into_fmt` 에 복사 → forward_into 중복본 커짐. 미러 코드에 `forward_into:NNNN 미러` 주석 有. **Step 5(forward_into/forward_gen/forward_prefill<C> 삭제)에서 dedup**.
- **`use KVCacheOps` 잔여**: `fmt_bridge.rs::EvalCacheKind for KiviCache`(cur_pos/needs_scores 가 `total_tokens`/`awqe_enabled` private 접근 불가로 trait 경유). `grep KVCacheOps engine/src/session/eval/`≠0 — ①-c 의도된 잔여(Step 5 inherent 화로 정리). ①-d 도 동류 잔여 허용(수용 기준=`forward_into<C>` 호출 0 + 바운드 0).
- **eval eviction 미발화**(host 게이트): ratio/budget eval 에서 `evicted=0`(min_kv_cache/budget 로직) — score-feed→eviction→nll 체인은 logits bit-identical 로 간접 증명(forward 동일 ⇒ score 동일 ⇒ 선택 동일). ①-d 무관.
- **cargo authoritative**: subagent/IDE 진단 불신, 메인이 `cargo build/test/clippy` 직접 재검증.
- **커밋 금지 untracked**: `.antigravitycli/`·`.claude/scheduled_tasks.lock`·`papers/.../microbench_*`·`.agent/todos/handoff_microbench_*`·`arch/pipeline/`. 명시 파일만 add(`git add -A` 금지).

---

## 자기점검
- 진입 문장 한 줄? ✓ "BC ①-d 진행"
- 왜 멈췄나? ✓ ①-c host 게이트 종결 clean checkpoint, ①-d=별도 호출처군 census 필요
- 최대 landmine? ✓ ★2 carve-out(KIVI 포함) + forward_into_fmt 중복(Step5 dedup) + use KVCacheOps 잔여
- 검증 게이트 수치/명령? ✓ host(44+3 test, BEFORE/AFTER JSON), ①-d=grep census + bit-identical
- 길이 적정? ✓ 상세는 roadmap/design 링크
