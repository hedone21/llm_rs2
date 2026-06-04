# Handoff: α-K BC (3d) S1+S2 ✅ (additive) → S3 (fmt_eligible flip) + 5-F

**작성**: 2026-06-04 (메인 세션)
**HEAD**: `928a95ee` feat(pressure): (3d) S1+S2 UER eviction seam ← `b56f5e09`(5-E docs) ← `cbb5f376`(5-E)
**브랜치**: `master` — **push 미실행**(origin 대비 ahead 3)
**다음 세션 진입 문장**: **"BC (3d) S3"** (chat fmt_eligible flip + env-gate 해소 + S25 device 게이트)

---

## TL;DR
α-K BC **5-E 완결**(KVCacheOps 본문 inherent 이전) + **(3d) S1+S2 완결**(chat-fmt eviction UER seam, additive·env-gated·production-dead). 5-F(legacy+trait 삭제) 전체 census + 사용자 4 결정 확정. **(3d) 설계 확정**(Approach B=Unwrap-Evict-Rewrap, 워크플로 만장일치). **왜 멈췄나**: S3(fmt_eligible flip)이 **production fmt-default flip 과 엮임**(env 게이트 공유 + host F32-CPU carve-out) → device 게이트 + host-test 기대값 갱신 동반이라 신중 진입 필요.

---

## 사용자 결정 (확정, 재질문 불요)
1. **5-F 범위 = Full 폐기**: legacy bin 완전 삭제 + OLD chain + KVCacheOps trait 삭제 + fmt-only. offload 최종 증명=5-B S25 게이트. verify.py=argus 재배선/drop.
2. **device 게이트 = S25 단독**(galaxy_s25, adb R3CY408S5SB 연결됨).
3. **chat = fmt 이주**(drop 아님, A1 보존) → (3d) 완성 필요.
4. (3d) **설계부터** → 완료(`design_alpha_k_3d_chat_fmt_2026_06_04.md`).

---

## 진행 상태
| 증분 | 상태 | commit |
|---|---|---|
| 5-E inherent rewire | ✅ host | `cbb5f376` |
| (3d) 설계 | ✅ | `design_alpha_k_3d_chat_fmt_2026_06_04.md` (wf_ba60a394-33f) |
| **(3d) S1** take_inner/put_inner | ✅ host additive | `928a95ee` |
| **(3d) S2** try_evict UER 분기 | ✅ host (env-gated, dead) | `928a95ee` |
| **(3d) S3** fmt_eligible flip | ★**다음** | env-gate 해소 + host chat 게이트 |
| **(3d) S4** S25 eviction-fmt device | 대기 | γ sanity 게이트 |
| 5-F flip+삭제 | 대기 | 비가역, S25 device |

---

## ★다음 = (3d) S3 — 핵심 entanglement (반드시 인지)
**S3 = chat builder `fmt_eligible: false→true`(session.rs:439)** 인데, chat fmt 발화엔 `fmt_eligible=true` **AND** env 게이트(`standard_format_gate_enabled()`=`LLMRS_KV_FMT`, model_forward.rs:674 기본 OFF) **둘 다** 필요(ensure_fmt_wrapped:262 `!fmt_eligible || !gate`). 즉 chat fmt-only(trait 삭제 가능)하려면 ensure_fmt_wrapped 의 **env-gate 의존을 제거**해야 함.

**그런데 env 게이트는 happy-path(build_standard_loop, fmt_eligible=true)와 공유** → 제거하면 happy-path 도 fmt-only 화 = **production fmt-default flip**(Step 4 미룬 "기본화"). 이건:
- **host F32-CPU carve-out(W-2)**: fmt standard F32 on CPU = attention_gen vs OLD inline-NEON → NOT bit-identical. host 표준 F32-CPU 기대값 test 는 fmt-truth 로 갱신 필요(legitimate). device(null-ptr)는 양쪽 attention_gen 이라 무관 → device 게이트가 acceptance.
- **device 게이트 필수**: S25 happy-path(argus_cli) 5 KV × 32-tok fmt-only ≡ frozen OLD baseline(F16 weight) + avg_tbt Δ≤+3%.

**S3 선택지**(설계 단계 권고):
- (a) env-gate 제거 + happy/chat 동시 fmt-only flip(= 5-F fmt-flip 과 병합) — host carve-out test 갱신 + S25 device 게이트. 가장 일관.
- (b) chat 만 always-fmt(별도 flag, happy 는 env-gated 유지) — 복잡, 임시. 비권장.

**권고**: (a) — S3 를 사실상 5-F 의 fmt-flip 으로 통합. S2 의 UER 분기가 이때 발화.

---

## S3 이후 (3d 종결 → 5-F)
- **(3d) S3 host 게이트**: chat 멀티턴 fmt eviction(sliding/h2o/d2o) ON≡OFF pos/evicted_total/logits. ★chat spec test(test_chat_session_multi_turn)는 `new_for_test`(build_chat_standard 우회)라 fmt 미발화 → fmt-eligible ModelForward 빌드하는 test 추가 or device 가 1차 acceptance.
- **(3d) S4 device(S25)**: γ sanity — eviction succeeded/pos↓/evicted_total>0/sane logits/unwrap-rewrap panic 0/turn-2 prefill append 좌표 정합(W3). 6 정책 × 3 KV × multi-turn(≥3턴, max_seq 축소).
- **5-F**: (3d) 후 chat off-OLD → forward_into<C>/forward_into_offload/execute<C> + impl×3 + trait + legacy bin + run_chunked_prefill + parity/inv_122 test + probe microbench 삭제. handoff_alpha_k_5f_entry_2026_06_04.md 의 삭제 표면 census 참조.

---

## Landmines / 미해결 (R6)
- **★S3 env-gate entanglement**(위) — S3 는 chat 만이 아니라 production fmt-default flip. host carve-out test 갱신 + S25 device 게이트 동반.
- **UER placeholder(잔여위험 1)**: take_inner 의 0-size placeholder 는 put_inner 까지 transient. evict panic-unwind 시 placeholder 잔존 가능(클로저 캡처로 `?` 전파는 rewrap 이후 — 완화). evict 는 실용상 panic-free.
- **W1 layer idx 정렬**: D2O cross-layer 가 fmt_caches enumerate 순서==layer idx 의존. spec test 로 고정 권장.
- **H2O+ flat-only carve-out**: chat try_evict 가 head_scores 미전달 → H2O+ = flat-score H2O. Round15 worthless → 영향 0.
- **device γ sanity ≠ bit-identical**: World-split 타이밍(F3)으로 eviction bit-identical 불가. host compact_parity=1차, device=sanity. A/C 접근의 logits/byte 게이트는 기각됨.
- **cargo authoritative** / 커밋 금지 untracked(`.antigravitycli`·`scheduled_tasks.lock`·`microbench_*`·`arch/pipeline/`) / push 사용자 요청 시.

---

## 자기점검
- 진입 문장? ✓ "BC (3d) S3"
- 왜 멈췄나? ✓ S3 = production fmt-default flip 엮임(env-gate 공유 + host carve-out + device 게이트)
- 최대 landmine? ✓ S3 entanglement(chat-only 아님)
- 게이트 수치? ✓ (3d) S1+S2: standard_format 16/chat 17 PASS·clippy clean / S4 device: S25 γ sanity 6정책×3KV×멀티턴
- 길이? ✓ 상세 = design_alpha_k_3d_chat_fmt + handoff_alpha_k_5f_entry + roadmap Step5
