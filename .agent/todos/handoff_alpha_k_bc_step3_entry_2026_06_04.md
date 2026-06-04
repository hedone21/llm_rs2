# Handoff: α-K BC Step 2 완료 → Step 3 ((3p) ④-a plan hot-path flip)

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `936d0c99 feat(kv): Phase α-K BC Step 2 — B-3 offload 분리 (PrefetchableCache KVCacheOps 비의존)` (+ 본 handoff 포함 docs 커밋 예정)
**브랜치**: `master` (origin push 미실행 — 사용자 요청 시)
**다음 세션 진입 문장**: **"BC Step 3 진행"**

> roadmap = `roadmap_alpha_k_bc_completion_2026_06_04.md`(Step 1·2 ✅, Step 3 ★다음). 적대검증 = workflow `w06swlxi9`(Step 2, 3 lens). SSOT = `arch/pipeline_stage_design_v2.md` §9.1 "⚠️ plan-flip" + line 742~748. ADR = `docs/adr/0001-kv-dispatch-paradigm.md` §8.3(B-1) + §6.5. 트랙 = [[project-pipeline-alpha-k]].

---

## TL;DR
α-K BC **Step 2(B-3 offload 분리) 완료** — Option A(supertrait bound 제거안). `PrefetchableCache: KVCacheOps` supertrait 제거 + `forward_into_offload<C>`/`preload_erased::<C>` → concrete `OffloadKVCache` monomorphize. offload **cluster-specific** KVCacheOps 결합을 (4) 차단자에서 분리. host gate bit-identical PASS. **다음 = Step 3((3p) ④-a)**: `plan.rs::execute<C: KVCacheOps>`(B-1)를 `Arc<StandardFormat>`/`Arc<KIVIFormat>` concrete-handle 로 flip — **BC 완주의 유일한 perf 위험 지점(hot crux)**. **왜 멈췄나**: Step 2 가 host 게이트까지 종결된 clean checkpoint, Step 3 는 production hot path 라 device avg_tbt 게이트(S25+Jetson) + (3d) plan 평가 선결이 필요한 별도 설계 라운드.

---

## 진행 상태 (검증 게이트)
| 증분 | 상태 | commit | 게이트 |
|---|---|---|---|
| Step 1 전체 (B-2 prefill + B-4 eval cold flip) | ✅ COMPLETE | `2941edca` 등 | host bit-identical |
| **Step 2 (B-3 offload 분리, Option A)** | ✅ **COMPLETE** | `936d0c99` | host bit-identical(아래) |
| Step 3 ((3p) ④-a plan hot flip) | ★**다음** | — | **device-gate(full)**: 5 KV 구성 × 32-tok bit-identical + **avg_tbt Δ≤+3%** (S25+Jetson) |
| Step 4 (device-gate → argus_cli) | TODO | — | argus_cli == legacy bit-identical |
| Step 5 (legacy 폐기 + B-2 OLD-chain 잔여 + KVCacheOps 삭제) | TODO | — | `grep KVCacheOps` 0 |

Step 2 host 게이트 실측(qwen2.5-1.5b-f16, `--kv-mode offload --kv-offload-storage raw --kv-type f16`, CPU, greedy n=32, BEFORE=generic vs AFTER=concrete):
- build + clippy `--workspace -D warnings` + fmt clean. offload 58 + preload 14 + base-vs-offload accuracy test pass. full lib **1229 pass / 26 fail**(24 opencl=GPU부재 panic `opencl.rs:678` + 2 RSS=병렬 flaky 단독 PASS, **둘 다 비-회귀**).
- **생성텍스트 bit-identical**: BEFORE/AFTER md5 `568a03e963bd021503d7fdd350cdd3a9` 동일(timing 라인만 wall-clock 노이즈). crash-free + sane("...Paris. The French are known for their love of food and wine...").
- acceptance: `grep "PrefetchableCache:"` supertrait 0 / `forward_into_offload<` generic 0.
- 적대검증 `w06swlxi9` 3 lens: 완전성 `confirmed-with-notes`(코드 blocking 0) / behavior-neutral `confirmed`(monomorphization semantics-preserving, blocking 0) / 잔여갭-scope `confirmed-with-notes`(framing 정밀화 요구 → roadmap 반영 완료).

---

## 다음 작업 (Step 3 — (3p) ④-a plan hot-path flip)
**목표**: `backend/opencl/plan.rs::execute<C: KVCacheOps>`(:1257, B-1 차단자)를 **④-a concrete-handle**(`Arc<StandardFormat>`/`Arc<KIVIFormat>`, static dispatch, **vtable 0**)로 flip. **production GPU decode hot path** — BC 완주의 유일한 perf 위험.
1. **(3d) plan 평가 선결**(진입 전 확정): `handoff_alpha_k_3d_entry_2026_06_03.md` §"다음 작업 (3d)" 3번(plan 평가 = (3p) 분기 결정)을 먼저 확정. prefill flip(①-b) 후 plan 평가 정합 확인.
2. **④-a 도입**: C 가 plan 에서 닿는 표면 = **6 스칼라 getter**(capacity :1286/1293 / current_pos :1286/1290/1291/1851 / res_pos :1294 / q2_tokens :1295) + **`advance_pos` 1회**(:1828). K/V 데이터 접근 0, attention 은 `AttentionVariant` enum static(`attention_into` 호출 0). → `Arc<dyn>` trait object 가 **아니라** concrete-handle.
3. **perf 측정 대상 = `StandardFormat` getter/advance 의 layer당 Mutex lock**(vtable 아님 — ④-a vtable 0). 현 plan path 는 generic monomorph(vtable 0 + lock 없음 = perf-optimal)라 ④-a 는 neutral-or-slightly-worse(cleanup, gain 아님).
- **권장 역할**: Architect (④-a concrete-handle 형태 + ④-b `AttentionVariant` 평탄화 묶음 여부 friction-triggered) → **Senior Implementer** (plan.rs hot flip, GPU/lock 민감) → Tester (device-gate full + avg_tbt 실측).
- **검증 게이트**: **device-gate(full)** 5 KV 구성 × 32-tok **bit-identical** + **avg_tbt Δ≤+3%** (S25 OpenCL `--opencl-rpcmem` + Jetson CUDA, n≥5 median tok0-inclusive). **회귀 시 (3p)만 revert** + Step 1·2 cold cluster 정리 유지(전역 원칙 5).

---

## Landmines / 미해결 (R6)
- **★Step 2 잔여 = B-2 OLD-chain 삭제 prerequisite (Step 5 선결, ⚠️Step 3 와 무관)**: Option A 후에도 `forward_into_offload` 본체가 **공유 OLD B-2 layer chain**(`layer.forward<OffloadKVCache>` → `forward_gen<C>`/`forward_prefill<C>`) 소비 + `impl KVCacheOps for OffloadKVCache`(offload.rs:263) 유지. **설계 종착 아니라 강제 유지** — 최종 KVCacheOps 삭제(Step 5) 전 `forward_into_offload` 의 **fmt 이주(=Option B: `forward_gen_fmt` + `OffloadKVCache: KVCacheFormat` interior-mut + preload pool aliasing 재설계, device GPU 재검증 필수)** 필요. `forward_into_offload` 는 `forward_into<C>` caller 가 아닌 **별개 forward** 라 ①-a~①-d 미접촉 → roadmap Step 5 의 "B-2 OLD-chain 잔여" 에 `run_chunked_prefill`(prefill.rs)과 함께 명시 추가됨.
- **Option A 범위 정밀(과장 금지)**: Step 2 = **B-3 cluster 분리(5 차단자 중 1개)** — offload-specific 결합만 제거, **(4) 자체는 미진전**(B-1/B-2/B-4/B-5 잔존). V3 적대검증이 "(4) 진전" 프레이밍을 경계 → roadmap Step 2 "수행 결과" 에 정밀 반영.
- **Step 3 = 유일한 perf 위험(hot crux)**: cold cluster(Step 1·2)를 먼저 비운 이유 = 위험 순서(전역 원칙 3·5). Step 3 회귀 시 (3p)만 격리 revert.
- **preload_erased generic 유지**: roadmap "비-generic 전환" 제안과 다르나, standalone PrefetchableCache bound 라 KVCacheOps 비의존(차단자 아님) + type-erasure 본래 목적 + Step 5 forward-compatible. PrefetchableCache trait standalone 보존(삭제 아님 — `재정의`).
- **cargo authoritative**: subagent/IDE(rust-analyzer) 진단 불신(이번에도 mid-edit E0277 가 cargo build 에선 clean). RSS 테스트 병렬 flaky → 단독 단일스레드 재확인.
- **커밋 금지 untracked**: `.antigravitycli/`·`.claude/scheduled_tasks.lock`·`papers/.../microbench_*`·`.agent/todos/handoff_microbench_*`·`arch/pipeline/`. 명시 파일만 add(`git add -A` 금지). push 는 사용자 요청 시.

---

## 자기점검
- 진입 문장 한 줄? ✓ "BC Step 3 진행"
- 왜 멈췄나? ✓ Step 2 host 게이트 종결 clean checkpoint, Step 3 = production hot crux 별도 설계 라운드((3d) plan 평가 선결 + device avg_tbt 게이트)
- 최대 landmine? ✓ Step 2 잔여(forward_into_offload fmt 이주 = Step 5 선결, device 재검증) + Step 3 = 유일 perf 위험
- 검증 게이트 수치/명령? ✓ Step 2 host(bit-identical md5 동일, 회귀 0), Step 3=device-gate full 5 KV × 32-tok bit-identical + avg_tbt Δ≤+3%
- 길이 적정? ✓ 상세는 roadmap/SSOT/ADR/workflow 링크
