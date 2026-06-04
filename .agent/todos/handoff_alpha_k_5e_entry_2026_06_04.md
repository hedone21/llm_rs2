# Handoff: α-K BC Step 5-B ✅ (host+S25 device) → Step 5-E

**작성**: 2026-06-04 (메인 세션, device 게이트 세션)
**HEAD**: `bb8a200f` perf(offload cast scratch 영속화) ← `7a22bb63` feat(5-B host) ← `641dc932`(5-C)
**브랜치**: `master` — **push 미실행**(사용자 승인 대기)
**다음 세션 진입 문장**: **"BC Step 5-E"** (inherent rewire — additive, host, trait 미삭제)

---

## TL;DR
α-K BC **Step 5-B(offload `forward_into_offload` → `KVCacheFormat` fmt 이주) host 구현 + S25 device 게이트 완료**. `OffloadFormat`(`Mutex<OffloadKVCache>` interior-mut wrapper) + `forward_into_offload_fmt`(copy-fork + DrainGuard) + `LLMRS_OFFLOAD_FMT` 게이트(transient wrap, ①-c 선례). **decode arm bit-identical**(host CPU + S25 OpenCL GPU) + **avg_tbt Δ+0.58%**(S25, n=6 median). 적대검증(4 lens) blocking 1(UAF) + warning(logits overflow) 적발·해소. additive + OFF default → production byte-불변. **왜 멈췄나**: 5-B 종결, 다음 증분(5-E)은 별개 작업(inherent rewire). Jetson defer(unreachable), OLD 경로 삭제는 5-F.

---

## 진행 상태 (Step 5)
| 증분 | 상태 | 비고 |
|---|---|---|
| 5-A 거취 결정 | ✅ | A1(session 보존) + 5-C 선착수 + offload 매체=legacy 부분제외 + verify 재배선 |
| 5-C KiviForward fmt | ✅ host `641dc932` | ①-e 상속 bit-identical |
| 5-D run_chunked_prefill | ✅ collapse | legacy 단독 호출 → dies-with-legacy(5-F 흡수) |
| **5-B offload fmt** | ✅ **host + S25 device** `7a22bb63`+`bb8a200f` | 아래 게이트 결과 |
| **5-E inherent rewire** | ★**다음** | additive, host, trait 미삭제. 설계 §3/§6.1 E1~E5 |
| 5-F legacy+trait 삭제 | 대기 | 비가역 device, 5-D 흡수, OLD offload-chain 삭제 |

### 5-B 게이트 결과 (검증 가능)
- **host**: clippy `--workspace -D warnings`/fmt/non-opencl/offload 62 test clean. decode arm `legacy run_offload`(qwen2.5-1.5b-q4_0 `--kv-mode offload --kv-offload-storage raw --kv-type f16 -n 40`) **ON≡OFF md5 `6aae1595` + coherent**("Gustave Eiffel..."), 게이트 발화 로그 확인(vacuous 아님).
- **S25 device**: `opencl --opencl-rpcmem` f16 offload n=32 **ON≡OFF md5 `46e8e0b7`** + **avg_tbt Δ +0.58% median(n=6)**(OFF 53.93 / ON 54.25 ms) ≪ +3% + **BL-2 Mutex poisoning 무발생**(GPU get_view alloc under guard, rc=0). 발화 로그 확인.
- 게이트 명령(S25): `LD_LIBRARY_PATH=/vendor/lib64:/system/lib64 [LLMRS_OFFLOAD_FMT=1] ./legacy_generate -b opencl --opencl-rpcmem --model-path models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf --tokenizer-path .../tokenizer.json --kv-mode offload --kv-offload-storage raw --kv-type f16 --greedy -n 32 -p "..."`.

---

## 다음 작업 (5-E — 설계 SSOT `design_alpha_k_step5_2026_06_04.md` §3/§6.1)
**5-E = inherent 전환 + rewire (★additive, trait 미삭제, host, device 불요)**:
- E1 KVCache inherent / E2 KiviCache pub fn / E3 fmt 위임 rewire(`standard_format`/`kivi_format`/`fmt_bridge` 의 `KVCacheOps::` 호출 → inner inherent) / E4 잡소비(`swap_handler:302` KVCache·`batch/runner:726`·test 별도파일) / **E5 `offload.rs:701 compare_views` 단형화 + `offload.rs:1558`·`kivi_cache.rs:2929` inline test**.
- **★5-B 잔여 반영**: `offload_format.rs` 의 `KVCacheOps::current_pos`/`get_view`(path X) + `OffloadFormat`이 쓰는 inherent(preload/release/retain/update/capacity/kv_dtype 등)도 5-E rewire 대상에 포함. `OffloadKVCache.cast_k/cast_v`(pub(crate)) 는 inherent write 경로라 무관.
- **게이트**: host build + `cargo test --workspace` + fmt/clippy clean + **grep 게이트 §6.2**(주석·문자열 제외 `KVCacheOps::` 호출 0건). device 불요(동작 불변).
- **권장 역할**: Architect(inherent census — get_view 이름충돌 R3 `view` 강제, KiviCache 조건부 R2 byte-identical) → Implementer.

---

## Landmines / 미해결 (R6)
- **★offload prefill 사전 존재 버그(5-B 범위 밖)**: `forward_into_offload`(preload pool)를 prefill 에 쓰면 빈-캐시 `preload()` 가 `preloaded=true` 설정(offload.rs:166) → `get_view` 가 store 로드 skip(offload.rs:408) → **zero K/V → garbage**. `OffloadForward`(chat)는 OLD 부터 이 경로라 broken+비결정(표준 chat 은 정상). legacy `run_offload` 는 prefill=`forward_into`(정상)라 회피 → fmt 게이트도 **decode arm 한정**. 5-F 에서 chat 살릴 때 OffloadForward prefill 을 forward_into 류로 교체 or preload 버그 수정 필요(별도 트랙). forward_into_offload_fmt prefill arm 은 OffloadForward/발산A 전용, causal 정확성은 단위테스트 격리 검증.
- **F32+host-mapped carve-out(W-2)**: offload `--kv-type f32` 게이트 ON 은 OLD inline-NEON(forward_gen.rs:554+) vs fmt `attention_gen` 으로 NOT bit-identical(StandardFormat 동일 carve-out). device(null-ptr)는 양쪽 attention_gen 이라 OK. host F32-CPU·UMA non-null 만 회피. 게이트는 F16 한정.
- **Jetson defer**: unreachable(hostname 미해석). offload GPU 경로는 CUDA 에도 존재(get_view gpu_backend) → Jetson 재연결 시 F16 offload 재검증 권장(선택, S25 가 1차 acceptance).
- **DrainGuard 불변식**: `forward_into_offload_fmt` 가 모든 반환 경로(에러·패닉)에서 preload worker 완료 보장 → caller `unwrap_caches` 의 `try_unwrap` 항상 성공(`expect` 는 방어적 assertion, 발화 불가). 5-E 가 `forward_into_offload_fmt` 본문 건드리면 이 가드 보존 필수.
- **cargo authoritative** / 커밋 금지 untracked(`.antigravitycli`·`scheduled_tasks.lock`·`microbench_*`·`arch/pipeline/`) / push 사용자 요청 시.

---

## 자기점검
- 진입 문장 한 줄? ✓ "BC Step 5-E"
- 왜 멈췄나? ✓ 5-B 종결(host+device PASS), 5-E 는 별개 증분(inherent rewire, trait 미삭제)
- 최대 landmine? ✓ offload prefill 사전 버그(decode-only 게이트 근거) + DrainGuard 불변식
- 게이트 수치/명령? ✓ S25 avg_tbt Δ+0.58% + md5 46e8e0b7 / 5-E grep 게이트 §6.2
- 길이 적정? ✓ 상세 = `design_alpha_k_step5_2026_06_04.md` + roadmap Step 5
