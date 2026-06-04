# Handoff: α-K BC Step 1 완료 → Step 2 (B-3 offload — `PrefetchableCache` KVCacheOps 비의존화)

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `2941edca feat(kv): Phase α-K BC ①-e — KIVIFormat prefill arm + run_kivi_ppl forward_into_fmt flip`
**브랜치**: `master` (origin push 미실행 — 사용자 요청 시. 현 working tree: roadmap 1 파일 + 본 handoff 미커밋 → docs 커밋 예정)
**다음 세션 진입 문장**: **"BC Step 2 진행"**

> roadmap = `roadmap_alpha_k_bc_completion_2026_06_04.md`(Step 1 ✅ COMPLETE, Step 2 다음). 설계+적대검증 = workflow `wej192c82`(①-e). SSOT = `arch/pipeline_stage_design_v2.md` §9.1-BC1'. ADR = `docs/adr/0001-kv-dispatch-paradigm.md` §8.3(5 cluster census). 트랙 = [[project-pipeline-alpha-k]].

---

## TL;DR
α-K BC **Step 1(B-2 forward chain + B-4 eval cold-path flip) 전체 완료** — ①-a/b/c/d/e 종결. ①-e 가 ①-d 게이트 발견 갭(KIVIFormat multi-token prefill arm 부재)을 메우고 run_kivi_ppl 의 마지막 `forward_into<C>` 2 site 를 fmt 전환, host 완전 bit-identical PASS. 다음 = **Step 2(B-3 offload)**: `PrefetchableCache: KVCacheOps` supertrait 제거 + `forward_into_offload<C>` / `preload_erased<C>` generic 소비 전환. **왜 멈췄나**: Step 1 이 host 게이트까지 종결된 clean checkpoint, Step 2 는 offload cluster(별도 차단자)라 새 설계 라운드 필요.

---

## 진행 상태 (검증 게이트)
| 증분 | 상태 | commit | 게이트 |
|---|---|---|---|
| ①-b prefill flip | ✅ host+device | `2bf5c500` | host + S25 3 dtype |
| ①-c eval flip | ✅ host | `1e4f20fe` | host eval-ll |
| ①-d 비-decode 10 site | ✅ host | `84bed97e` | host bit-identical |
| **①-e run_kivi_ppl + KIVIFormat prefill arm** | ✅ **host** | `2941edca` | host bit-identical(아래) |
| **Step 1 전체** | ✅ **COMPLETE** | — | B-2+B-4 cold flip 종결 |
| Step 2 B-3 offload | ★다음 | — | host(supertrait grep 0) + device `--kv-offload` crash-free |

①-e host 게이트 실측(CPU qwen2.5-1.5b-q4_0, prefill_512.txt=456 tok, `--kv-mode kivi --ppl`, BEFORE=forward_into vs AFTER=forward_into_fmt):
- build + clippy `--workspace -D warnings` + fmt clean. kivi_format 6/standard_format 12/fmt_bridge 3 pass. **비-opencl·비-RSS 회귀 0**(opencl/memory::opencl 실패=GPU 부재, pressure::kv_cache RSS 2건=병렬 flakiness, 단독 단일스레드 통과 확인).
- **완전 bit-identical**: NLL=56.9390 / count=455 / PPL=1.1333 / Q2_tokens=384 / res_pos=72 / flush_count=6 / final_cache_pos=456 / total_nll(전정밀) / flush qcf_metrics 전부 일치. **panic 0**(prefill arm 신설로 ①-d 갭 해소). ★2 carve-out 미발동(prefill=flash exact, single-shot=decode 미진입).
- 신규 단위테스트 `test_attention_into_prefill_causal_uniform`: seq=4<res_cap(32) Q2 flush 미발생 → residual F32 exact → causal mean(0..=r)=r/2 **bit-exact** PASS.

---

## 다음 작업 (Step 2 — B-3 offload)
**목표**: `PrefetchableCache` 를 `KVCacheOps` 비의존으로 재정의하여 (4) 컴파일 차단자 중 offload cluster 해소. **production hot 아님**(opt-in `--kv-offload`, plan path 미사용).
1. **census**: `PrefetchableCache`(`engine/src/pressure/kv_cache.rs:16` 부근, supertrait `: KVCacheOps`)가 KVCacheOps 의 **어느 method 를 실제로 쓰는지** 전수 — current_pos/capacity/get_view 등. supertrait → 독립 trait 또는 `KVCacheFormat` 위임으로 재정의 형태 결정.
2. **`forward_into_offload<C>`**(`transformer.rs:2906`) + **`preload_erased<C>` fn-ptr**(`preload_pool.rs:177`)의 generic 소비를 trait object/비-generic 으로 전환. fn-ptr 의 generic erasure 방식이 설계 포인트.
3. **검증**: host build+test + `grep "PrefetchableCache" engine/` 로 supertrait bound 제거 확인. device `--kv-offload` K-sweep crash-free + sane output(offload 는 hot 아님 → bit-identical 보다 정확성+안정성 우선; 가능하면 legacy offload 출력과 bit-identical). hot path 무변 확인.
- **권장 역할**: Architect(`PrefetchableCache` 재정의 census + 형태) → Implementer(구현+host) → Tester(`--kv-offload` device).
- **대안 경로**: Step 2 와 Step 3((3p) hot crux)는 둘 다 Step 1 에만 의존 → 독립. offload 분리가 어려우면 Step 3((3p) plan flip) 먼저 진입 가능하나, **Step 3 는 유일한 perf 위험 지점**(hot crux)이라 cold cluster(Step 2)를 먼저 비우는 것이 위험 순서상 권장(roadmap 전역 원칙 3·5).

---

## Landmines / 미해결 (R6)
- **KIVIFormat GPU prefill 은 host 미검증**: ①-e 의 prefill arm 은 host CPU(new_gpu 가 `backend.name()!=OpenCL`이면 CPU-mode 반환, `kivi_cache.rs:449-451`)에서만 게이트 통과. **GPU 경로(bits=16 HeadMajor / bits 2/4/8 assembled view)는 device 검증 미실시** — prefill_attention 이 `kv_layout`/`kv_capacity` 로 분기하므로 구조적으로 forward_prefill GPU 경로 미러이나, S25 KIVI device 게이트는 Step 4/5 또는 후속에서 권장(현 run_kivi_ppl host CPU 측정만).
- **run_kivi_ppl decode site = unreachable**: `prefill_len = eval_tokens.min(max_seq_len)` 이고 `eval_tokens = total.min(max_seq_len)` → 항상 `prefill_len==eval_tokens` → decode 루프 `prefill_len..eval_tokens-1` 빈 범위. ①-e 의 decode 전환(runner.rs:456)은 correct-by-construction + ①-c 커버지만 **이 binary 제어흐름상 미실행**. forward_into<C> 소비 제거 목적의 전환(정당).
- **`prefill_attention` 공유**: 이제 StandardFormat + KIVIFormat 양쪽이 `pub(crate)` free fn 재사용. Step 5(forward_prefill<C> 삭제) 시 additive-fork 중복 dedup 대상.
- **prefill.rs 잔여**(run_chunked_prefill, profiler+variance_collector)는 아직 `forward_prefill<C>` 직접 소비 — Step 5 또는 별도 증분에서 정리(B-2 잔여).
- **cargo authoritative**: subagent/IDE(rust-analyzer) 진단 불신, 메인이 cargo build/test/clippy 직접 재검증. RSS 테스트(`test_release_unused_pages_rss_reduction`/`test_prune_prefix_calls_release_unused_pages`)는 **병렬 실행 시 flaky** — 단독 단일스레드로 재확인.
- **커밋 금지 untracked**: `.antigravitycli/`·`.claude/scheduled_tasks.lock`·`papers/.../microbench_*`·`.agent/todos/handoff_microbench_*`·`arch/pipeline/`. 명시 파일만 add(`git add -A` 금지). push 는 사용자 요청 시.

---

## 자기점검
- 진입 문장 한 줄? ✓ "BC Step 2 진행"
- 왜 멈췄나? ✓ Step 1 host 게이트 종결 clean checkpoint, Step 2 = offload cluster 별도 설계 라운드 필요
- 최대 landmine? ✓ KIVIFormat GPU prefill host 미검증(device defer) + run_kivi_ppl decode unreachable
- 검증 게이트 수치/명령? ✓ host(완전 bit-identical NLL=56.9390/flush_count=6, 회귀 0), Step 2=`grep PrefetchableCache` supertrait 0 + `--kv-offload` device crash-free
- 길이 적정? ✓ 상세는 roadmap/ADR/workflow 링크
