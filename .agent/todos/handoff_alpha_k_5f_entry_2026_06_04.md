# Handoff: α-K BC Step 5-E ✅ (host) → Step 5-F (legacy+trait 삭제, 비가역 device)

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `cbb5f376` refactor(pressure): Phase α-K BC Step 5-E — KVCacheOps 본문 inherent 이전 + 소비자 rewire
**브랜치**: `master` — **push 미실행**(사용자 승인 대기)
**다음 세션 진입 문장**: **"BC Step 5-F"** (legacy 폐기 + KVCacheOps trait 삭제 — ★비가역, device 게이트 필수)

---

## TL;DR
α-K BC **Step 5-E(inherent rewire) host 완결**. KVCacheOps trait 본문을 3 cache(KVCache/KiviCache/OffloadKVCache)의 inherent fn 으로 **이전(MOVE)** + trait 은 inherent 위임 + fmt/비-forward/test 의 `KVCacheOps::` 호출과 import 를 inherent 직호출로 **rewire**. **additive·trait 미삭제**. 적대검증(3 fidelity lens): 3 cache 본문 전부 원본 trait(fd246887)와 **byte-identical**, recursion 무. **왜 멈췄나**: 5-E 종결, 5-F 는 별개 증분이며 **비가역(코드 삭제) + device 게이트 + 선결 2건(offload 매체·verify 재배선)**이라 사용자 go-ahead + device 준비 후 진입.

---

## 진행 상태 (Step 5)
| 증분 | 상태 | commit |
|---|---|---|
| 5-A 거취 결정 | ✅ | (문서) A1 |
| 5-C KiviForward fmt | ✅ host | `641dc932` |
| 5-D run_chunked_prefill | ✅ collapse | dies-with-legacy(5-F 흡수) |
| 5-B offload fmt | ✅ host+S25 device | `7a22bb63`+`bb8a200f` |
| **5-E inherent rewire** | ✅ **host** | **`cbb5f376`** |
| **5-F legacy+trait 삭제** | ★**다음** | 비가역 device |

### 5-E 게이트 결과 (검증 가능)
- build: opencl(lib+tests+bins) + non-opencl(lib) + workspace(legacy/manager/shared) **clean**.
- fmt `--all --check` clean / clippy `--workspace -D warnings` clean.
- **grep §6.2**: rewire 대상 14파일 `KVCacheOps::` 호출 **0건**.
- lib test **1241 PASS**; 실패 = opencl 21(GPU 부재)·RSS madvise 2(병렬압박 flakiness, 격리 PASS)·INV-LAYER 2(`htp_fastrpc/opencl/plan.rs` 무관 파일 사전 baseline staleness — `git diff` 로 내 변경 아님 확정).
- 적대검증 `wf_70537a28` 3 fidelity lens: KVCache 7 / KiviCache 12 / OffloadKVCache 8 inherent 본문 = 원본 trait **byte-identical**(update/get_view `diff` IDENTICAL). rewire-neutrality·completeness lens 는 StructuredOutput 미호출+synth 세션한도로 미완 → **cargo 게이트로 대체 검증**(이미 PASS).

---

## 다음 작업 (5-F — 설계 SSOT `design_alpha_k_step5_2026_06_04.md` §6.1/§6.3)
**5-F = 비가역 cutover (격리 단일 commit, 회귀 시 `git revert` 1회로 5-E 복귀)**:
- **F1**: `legacy_generate` bin + `engine/legacy/generate.rs` 삭제 (Cargo.toml bin 선언 포함).
- **F2**: OLD generic chain 삭제 — `forward_into<C>`/`forward<C>`/`forward_gen<C>`/`forward_prefill<C>`(+`forward_into_offload`/`forward_into_offload_fmt` 의 OLD 본체) / `execute<C>`(plan.rs) / `TransformerModelForwardArgs<'_,C>` / `impl KVCacheOps for {KVCache,KiviCache,OffloadKVCache}` 3개 + `LLMRS_KV_FMT`/`LLMRS_OFFLOAD_FMT` 게이트 상수화.
- **F3**: `kv_cache_ops.rs` 의 `KVCacheOps` trait 정의 삭제 + `kv_cache.rs:8-10` 재export 블록 삭제 (KVLayout/KiviRawBuffers 정의는 잔존 — §7-(4) 파일명 유지 권고).
- **F4**: (선택) `KVCacheFormat` rename — **별도 chore commit**(mechanical sed, 삭제 회귀 분석 오염 방지). host build/test 만.
- **★5-F 선결 2건 (진입 전 확보)**: ① **offload device 매체**(BL-1) — 5-A 결정 = **legacy offload 경로를 5-F 에서 부분 제외**(argus-bench 없이 legacy offload 한시 보존 후 별도 제거). ② **verify.py 재배선**(HB-3) — `verify.py:99` 의 stale `generate` bin → argus-bench/eval 로 재배선 or verify 폐기 결정.
- **device 게이트(full)**: 5 KV × 32-tok bit-identical(ON≡OFF) + avg_tbt Δ≤+3% (S25 OpenCL + Jetson CUDA, **frozen baseline = 5-F 이전 캡처 필수** — legacy 가 삭제되므로). offload bit-identical 은 5-B device 게이트가 이미 증명(host 미발화 경로).

---

## Landmines / 미해결 (R6)
- **★비가역(코드 삭제)**: 5-F 는 동작 변경이 아니라 parallel path 제거. bit-identical 은 5-E 까지 fmt 경로가 이미 증명 — 5-F device 게이트의 유일 위험 = **monomorphization perf 노출(avg_tbt)**. 회귀 시 격리 commit revert.
- **★5-E defer 잔여 (trait-only, inherent 미신설)**: KVCache/KiviCache 의 `set_current_pos/advance_pos/get_buffers_mut/res_pos/q2_tokens/res_cap/needs_flush/flush_if_needed`(KVCache 는 set_current_pos/advance_pos/get_buffers_mut/ensure_capacity 는 inherent 有, res_pos 등은 trait default 미사용) + **OffloadKVCache `memory_usage_bytes`/`set_current_pos`**. 생존 fmt 소비자 미호출 — **`offload.rs` test 모듈 + `legacy:4044/4645` 만 trait method-syntax 로 사용**. 5-F 트랙: legacy 는 F1 으로 소멸 / **offload test `cache.memory_usage_bytes()`는 OffloadKVCache inherent `memory_usage_bytes`(1줄=`self.store.storage_size()`) 추가 후 offload test `use KVCacheOps` 제거** 필요(미처리 시 trait 삭제로 컴파일 break). res_pos/q2_tokens 등은 호출처 0(또는 pub 필드 직접접근)이라 trait 와 함께 소멸.
- **★get_view 이름 비대칭**: KVCache inherent = **`view(&mut)`**(2-arg `get_view(:534)` 충돌 회피), KiviCache/OffloadKVCache inherent = **`get_view(&mut)`**(동명). 5-F 에서 KVCache 의 OLD `get_view`(0-arg trait) 호출처가 더 나오면 `.view()` 로(2-arg inherent 와 arity 충돌 주의).
- **INV-LAYER 001/003 spec test**: `htp_fastrpc.rs`/`opencl.rs`/`opencl/plan.rs` 7건 = 사전 baseline staleness(KV 리팩토링 무관). 5-F 와 별개 — baseline JSON 갱신 or 무시.
- **RSS madvise test 2종**: 병렬 메모리압박 flakiness(`--test-threads=1` 격리 PASS) — 비-회귀.
- **cargo authoritative** / 커밋 금지 untracked(`.antigravitycli`·`scheduled_tasks.lock`·`microbench_*`·`arch/pipeline/`) / push 사용자 요청 시.

---

## 자기점검
- 진입 문장 한 줄? ✓ "BC Step 5-F"
- 왜 멈췄나? ✓ 5-E host 종결, 5-F 는 비가역+device+선결 2건이라 별도 진입
- 최대 landmine? ✓ 비가역 cutover + 5-E defer 잔여(offload memory_usage_bytes inherent 미신설)
- 게이트 수치/명령? ✓ 5-E: grep §6.2 0건 + test 1241 PASS + 적대검증 byte-identical / 5-F: 5 KV×32-tok bit-identical + Δ≤+3%
- 길이 적정? ✓ 상세 = `design_alpha_k_step5_2026_06_04.md` §6 + roadmap Step 5
