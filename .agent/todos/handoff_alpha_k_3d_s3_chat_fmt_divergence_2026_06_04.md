# Handoff: α-K BC (3d) S3 — chat 비결정성 ✅ 수정 → #2 fmt start_pos>0 prefill 발산 (진짜 blocker)

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `1eae6c46` feat(chat): (3d) S3 fmt_eligible flip ← `9d1f16f7` fix(chat): --greedy RNG 비결정 수정 ← `9bfa6db8`
**브랜치**: `master` — **push 미실행**(origin 대비 ahead 3, 사용자 승인 대기)
**다음 세션 진입 문장**: **"BC (3d) S3 #2 — forward_prefill_fmt start_pos>0 발산 root-cause"**

---

## TL;DR
(3d) S3 진입 → chat fmt_eligible flip 적용 후 host 게이트 시도 중 **chat 비결정성** 발견. 광범위 조사 끝에 **두 개의 독립 버그**로 판명:
1. **✅ 수정·커밋(`9d1f16f7`)**: chat `--greedy` 가 first-token 샘플링에 temperature=0 미전파 → RNG 첫 토큰 비결정 (사용자 요청 deliverable, 완료).
2. **★ 미해결(`1eae6c46`에 #2로 명시)**: chat 결정성 회복 후 노출된 **진짜 (3d) S3 blocker** — `forward_prefill_fmt` 가 **start_pos>0 multi-token prefill** 에서 `forward_prefill` 과 미세 수치 발산. **왜 멈췄나**: #1 은 완료, #2 는 fresh 깊은 수치 조사(turn-1 start_pos=0 bit-identical, turn-2 start_pos>0 발산 — 이유 불명, Q/K/V/flash args 모두 동일해야 함).

---

## 진행 상태
| 항목 | 상태 | commit / 근거 |
|---|---|---|
| chat `--greedy` 비결정 root-cause | ✅ | generate.rs:480 `--greedy`→temp=0 override 누락(canonical=session/init.rs:168) |
| 위 수정 | ✅ host 검증 | `9d1f16f7` — chat --greedy **8/8 결정적**(이전 1/6~ flaky), build both+clippy clean, sampling test 6 pass |
| (3d) S3 fmt_eligible flip | ✅ 커밋(env-gated, inert) | `1eae6c46` — production env OFF byte-불변 |
| **#2 fmt start_pos>0 prefill 발산** | ★ **다음** | host 게이트로 확정, root-cause 필요 |
| S4 S25 device γ-sanity | 대기 | #2 선결 |

---

## ★ #2 — 진짜 (3d) S3 blocker (정밀 특성화 완료, 결정적 재현)
**증상**: chat `--greedy` fmt-ON(`LLMRS_KV_FMT=1`) vs fmt-OFF, F16 KV, CPU:
- **turn-1 (prefill start_pos=0 + 전체 decode): bit-identical** (logit_sum 완전 일치, rows 1-8).
- **turn-2 (prefill start_pos>0 multi-token): 발산** (row 9 = turn-2 prefill: OFF sum=-329937.41125 / ON sum=-328538.10382, **같은 argmax=198** 이지만 ~0.4% logit 차 → K/V 누적되어 turn-2 decode 2~3 토큰째 argmax flip → 출력 발산).
- **5 정책(sliding/h2o/streaming/h2o_plus/d2o) 전부 동일 발산**, eviction 회계는 동일(eviction turn-4 발화, 발산은 turn-2 선행).

**왜 버그인가(carve-out 아님)**: host CPU 에서 fmt(`prefill_attention`=standard_format.rs:583)와 OLD(`forward_prefill`=transformer_layer/forward.rs:41) **둘 다 동일 `flash_attention_forward_strided`** 를 탄다(GPU 미dispatch). q_start_pos=start_pos, cache_seq_len, strides, window 모두 동일하게 전달됨(코드 추적 확인). Q(RoPE@start_pos)·K·V(동일 KV 버퍼, 동일 dequant)도 동일해야 함. **그런데 start_pos=0 은 일치하고 start_pos>0 만 발산** → 추적상 설명 안 되는 미세 수치차. structural 아님(coherent 출력, 같은 prefill argmax).

**다음 조사(진입 즉시)**: turn-2 prefill 의 attention 입력(Q/K/V 슬라이스, q_start_pos, cache_seq_len)을 `forward_prefill_fmt`(transformer_layer/forward_prefill_fmt.rs:140-156)와 `forward_prefill`(forward.rs:251-535) 양쪽에서 **직접 덤프 비교**(이제 결정적이라 계측이 마스킹 안 함). 의심: (a) fmt wrap/unwrap(mem::take)이 KV 내용을 미세 변경, (b) write_kv_batch vs cast+update 의 F16 라운딩 차(turn-1 [0..15] 은 일치했으나 [15..23] decode-write 경로 차이 가능), (c) prefill_attention 의 dequant 범위/stride 가 start_pos>0 에서 미세 차. **재현 명령**:
```
INPUT=$'What is its population?\n/exit\n'
printf '%s' "$INPUT" | LLMRS_KV_FMT=1 RAYON_NUM_THREADS=1 ./target/release/legacy_generate --chat -b cpu \
  -m models/qwen2.5-1.5b-instruct/qwen2.5-1.5b-instruct-q4_0.gguf --kv-type f16 --max-seq-len 256 \
  -n 10 --greedy --protected-prefix 4 --prompt "What is the capital of France?" eviction none
# vs 동일 (LLMRS_KV_FMT 미설정) → turn-2 출력 비교
```
(CPU-only 빌드: `--no-default-features --features profile`)

---

## 조사 방법론 / Landmines (R6) — 같은 함정 반복 금지
- **★ chat 비결정성(#1)이 #2 를 가렸다**: 조사 초기 "turn-2 발산"을 fmt 버그로 오인 → 실은 RNG(--greedy 미전파)로 매 run noise. **#1 수정 후에야 #2 가 결정적 재현됨.** 결정성 먼저 확보가 필수였음.
- **메모리 버그 아님 (전부 배제)**: ASan(UAF/OOB) clean, TSan(data race) clean+비결정 지속, zeroing global allocator 무효, MALLOC_PERTURB flaky(고친 게 아니라 timing 확률만 변경). **모든 CPU 버퍼 = `SharedBuffer::new`(memory/host/shared.rs:19) `vec![0u8]` zero-init** → uninit read 원천 불가. 이 막대한 도구 조사는 #1 이 RNG 임을 모를 때 헛수고였음 — **비결정 의심 시 먼저 `--temperature 0` 으로 RNG 경로부터 배제할 것**.
- **계측이 #1 을 마스킹**: in-process eprintln(LOGIT_DBG/X_DBG/CHAT_DBG) 추가 시 #1 비결정이 사라짐(heisenbug). #2 는 결정적이라 계측 충실 — #2 조사엔 자유롭게 계측 가능.
- **happy-path 는 #2 무관**: `-p` 단일 prompt(start_pos=0 단일 prefill)는 start_pos>0 multi-token prefill 을 안 거침 → ①-b/Step3/4 가 #2 를 못 잡은 이유. chat turn-2 가 유일 host 재현.
- **valgrind 미설치**(sudo password 불가), **MSan 비현실적**(C deps + 버퍼 zero라 어차피 clean). nightly ASan/TSan 은 사용 가능(`-Zsanitizer`, `--target x86_64-unknown-linux-gnu`).
- fmt_eligible flip 은 **env-gated**(production OFF=inert). #2 미해결 동안 chat fmt 는 `LLMRS_KV_FMT=1` 로만 발동.
- cargo authoritative(rust-analyzer inactive-code/stale 경고 무시) / 커밋 금지 untracked(`.antigravitycli`·`scheduled_tasks`·`microbench_*`·`arch/pipeline/`) / push 사용자 요청 시.

---

## #2 이후 (3d) 종결 경로
- #2 root-cause+fix → forward_prefill_fmt start_pos>0 bit-identical 확보.
- (3d) S3 host 게이트 재실행: chat --greedy fmt-ON≡OFF 5 정책 멀티턴+eviction **텍스트 bit-identical**(이제 결정적이라 유효).
- (3d) S4 device(S25): γ-sanity(crash-free/pos↓/evicted>0/sane/turn-2 append 좌표 정합).
- 이후 5-F(legacy+trait 삭제) — `handoff_alpha_k_5f_entry_2026_06_04.md`.

---

## 자기점검
- 진입 문장? ✓ "BC (3d) S3 #2 — forward_prefill_fmt start_pos>0 발산 root-cause"
- 왜 멈췄나? ✓ #1(chat 비결정) 완료, #2 는 fresh 깊은 수치 조사
- 최대 landmine? ✓ #1 이 #2 를 가렸음 / 메모리 버그 전부 배제(시간 낭비 반복 금지) / 비결정은 RNG 먼저 의심
- 게이트 수치? ✓ #1: chat --greedy 8/8 결정 / #2: turn-2 row9 OFF -329937.41 vs ON -328538.10(argmax 동일)
- 길이? ✓ 상세 = design_alpha_k_3d_chat_fmt + 본 handoff
