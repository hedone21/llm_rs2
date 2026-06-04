# Handoff: α-K BC ①-d 완료 → ①-e (run_kivi_ppl fmt 전환 = KIVIFormat prefill arm 신설)

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `84bed97e feat(kv): Phase α-K BC ①-d — B-2 비-decode forward_into→forward_into_fmt flip (10 site)` (impl; 뒤이은 docs 커밋이 roadmap+본 handoff 동반)
**브랜치**: `master` (origin push 미실행 — 사용자 요청 시)
**다음 세션 진입 문장**: **"BC ①-e 진행"**

> roadmap = `roadmap_alpha_k_bc_completion_2026_06_04.md`(Step 1: ①-a/b/c/d ✅, ①-e 다음). 설계+적대검증 = `design_alpha_k_1d_cut_2026_06_04.md`(workflow `w12qx2ybg`). SSOT = `arch/pipeline_stage_design_v2.md` §9.1-BC1'. 트랙 = [[project-pipeline-alpha-k]].

---

## TL;DR
α-K BC Step 1 의 **①-d(B-2 비-decode 10 site) 완료** — warmup·qcf·batch·run_ppl(KVCache)·dump_importance 의 `forward_into<C>` → `forward_into_fmt`, host 게이트 PASS. **forward_into_fmt 에 workspace=None decode fallthrough 신설**(발산 A). 다음 = **①-e**: run_kivi_ppl(KIVI 2 site) fmt 전환. **왜 멈췄나**: ①-d 가 host 게이트까지 종결된 clean checkpoint, ①-e 는 **KIVIFormat multi-token prefill arm 신설**이라는 별도 feature 선결이 필요(①-d 게이트가 갭 발견).

---

## 진행 상태 (검증 게이트)
| 증분 | 상태 | commit | 게이트 |
|---|---|---|---|
| ①-b prefill flip | ✅ host+device | `2bf5c500` | host + S25 3 dtype |
| ①-c eval flip | ✅ host | `1e4f20fe` | host eval-ll |
| **①-d 비-decode 10 site** | ✅ **host** | `84bed97e` | host(build+test+fmt+clippy) + legacy bit-identical |
| ①-e run_kivi_ppl fmt | ★다음 | — | KIVIFormat prefill arm 선결 + host + (device 권장) |

①-d host 게이트 실측(CPU qwen2.5-1.5b-q4_0, BEFORE=pre-①-d vs AFTER):
- build + clippy `--workspace -D warnings` + fmt clean. lib **1241 pass / 13 fail(전부 `backend::opencl` GPU 부재 pre-existing) / 비-opencl 회귀 0** / 변경모듈 74 pass.
- **batch** 출력 텍스트 bit-identical. **ppl(KVCache)** NLL=173.1049 bit-identical. **dump_importance** table IDENTICAL. **ppl(KIVI)** defer→forward_into 라 bit-identical by construction.
- **warmup seq_len=1 fallthrough 런타임 PASS**: `eviction sliding --window 1024`로 run_chunked_prefill→run_warmup 강제 → `[WARMUP] tokens=1` 발화 + 출력 "Paris..." + 패닉 없음.

---

## 다음 작업 (①-e)
**목표**: run_kivi_ppl(ppl/runner.rs:367 prefill + 448 decode)을 forward_into_fmt 로 전환. **선결 = KIVIFormat multi-token prefill arm 신설**.
1. **KIVIFormat::attention_into 에 prefill arm 추가**(`engine/src/pressure/kivi_format.rs:95`): `seq_len = q.shape().dims()[1]; if seq_len > 1 { … prefill_attention(…) }`. StandardFormat::attention_into:327(`prefill_attention` 호출) 미러하되 **KiviCache 특이성 처리**:
   - `KiviCache::get_view` 는 **compact view**(`[1, total, kv_heads, head_dim]`, total=current_pos) — StandardFormat(capacity=max)과 다름. prefill_attention 의 `capacity`/`q_start_pos` 인자를 compact view 기준으로(`capacity`=view stride, layout=SeqMajor bits2/4/8 CPU). `cache.layout()` = SeqMajor(bits≠16)/HeadMajor(bits16 GPU)로 분기.
   - GPU native 경로(attention_native, bits16 HeadMajor)도 multi-token 지원 필요 여부 census — 안 되면 prefill 은 F32 dequant view fallback 강제.
2. **run_kivi_ppl 전환**(367/448): `KiviCache::forward_fmt_roundtrip` + `cache_self_need_scores = kv_caches.first().is_some_and(|c| c.needs_scores())` 선계산 주입(①-c 미러, AWQE 보존). import `TransformerModelForwardArgs` 제거.
3. **검증**: host `--ppl <text> --kv-mode kivi` BEFORE/AFTER — flush 정수회계 + nll Δ≤2e-4(★2 carve-out). multi-token prefill 이므로 **panic 재발 없음** 우선 확인. AWQE on(긴 텍스트+flush) case 도 커버 권장. device(S25 KIVI) 권장.
4. prefill.rs(run_chunked_prefill, profiler+variance_collector)도 ①-e 또는 Step5 후보(legacy-only).

---

## Landmines / 미해결 (R6)
- **★KIVIFormat prefill arm 부재(①-d 게이트 발견, ①-e 의 본질)**: `KIVIFormat::attention_into`(kivi_format.rs:95-173)는 `attention_gen`(single-query decode)만 호출 — multi-token prefill arm 없음. forward_prefill_fmt 가 KIVI multi-token prefill query 넘기면 **panic**(`x86.rs:228` slice index). eval KIVI 가 안 걸린 이유 = eval 은 KIVI+AWQE 시 token-by-token prefill(eval_loop.rs:641-658). **①-e 핵심 = 이 arm 신설**. 적대검증(V1/V3)이 forward_prefill_fmt bit-identity(헤더)만 보고 KIVIFormat arm 부재를 놓침 → **gate 가 잡음**. 교훈: fmt fork 의 attention arm 존재를 cache 종류별로 확인.
- **forward_into_fmt fallthrough(발산 A)**: decode 분기에 `workspace.is_none()` → forward_prefill_fmt(degenerate seq_len=1) 추가됨(transformer.rs:2137). production 호출처(model_forward/eval decode=workspace Some) 미발화 검증 완료. 이 분기는 **forward_gen(inline) 아닌 forward_prefill(flash)** 시맨틱이라 ★2 carve-out(F32-host) 무관(prefill=flash 는 F32 도 bit-identical).
- **★2 carve-out(KIVI 포함)**: KiviCache get_view=F32 → CPU host decode 는 NOT bit-identical(~1e-6). ①-e KIVI 게이트는 nll Δ≤2e-4 허용(①-c 확인). F32-host 배제.
- **`forward_into_fmt` additive-fork 중복**: Step 5(forward_into/forward_gen/forward_prefill<C> 삭제)에서 dedup.
- **cargo authoritative**: subagent/IDE(rust-analyzer) 진단 불신, 메인이 cargo build/test/clippy 직접 재검증(이번에도 stale E0422 무시·cargo clean 확인).
- **커밋 금지 untracked**: `.antigravitycli/`·`.claude/scheduled_tasks.lock`·`papers/.../microbench_*`·`.agent/todos/handoff_microbench_*`·`arch/pipeline/`. 명시 파일만 add(`git add -A` 금지). push 는 사용자 요청 시.

---

## 자기점검
- 진입 문장 한 줄? ✓ "BC ①-e 진행"
- 왜 멈췄나? ✓ ①-d host 게이트 종결 clean checkpoint, ①-e = KIVIFormat prefill arm 신설(별도 feature) 선결
- 최대 landmine? ✓ KIVIFormat multi-token prefill arm 부재(①-d 게이트 발견)
- 검증 게이트 수치/명령? ✓ host(1241 pass/회귀0, bit-identical 4종 + fallthrough 런타임), ①-e=`--ppl --kv-mode kivi` BEFORE/AFTER
- 길이 적정? ✓ 상세는 roadmap/design 링크
