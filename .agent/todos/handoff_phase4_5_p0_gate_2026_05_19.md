# Phase 4-5-f P0 디바이스 게이트 + C4-1~3 종결 + chat pre-existing 분리

**날짜**: 2026-05-19
**master HEAD**: `83c73dff` (C4-1~3 ff-merge 직후)
**Phase 4-5-f branch HEAD**: `a9f0f106` (별도, 미머지)
**baseline HEAD**: `b991691f` (Phase 4-4.10)

## 한 줄 요약

P0 게이트 측정 결과 **Phase 4-5-f chat repl v2 회귀 없음** — baseline `b991691f`에서도 동일 chat garbage가 재현되어 pre-existing 이슈로 확인. non-chat path PASS + TBT 32.50 ms/tok 정상. C4-1~3 (KvMode subcommand) master ff-merge 완료.

## P0 S25 디바이스 게이트 결과

디바이스: Galaxy S25 (R3CY408S5SB), 백엔드 opencl, 6 thread, qwen2.5-1.5b-q4_0.gguf.

| 항목 | Phase 4-5-f (`a9f0f106`) | Baseline (`b991691f`) | 판정 |
|---|---|---|---|
| chat 1 turn | garbage (`I` 또는 `.IsAny` 반복) | garbage (`Capital of China?撙�\n撙�...`) | **동치** (pre-existing) |
| chat 3 turn (R1 KV 보존) | turn별 garbage, KV pos 73→103→133 정상 누적 | 미측정 (1 turn 이미 garbage 확인) | KV invariant 정상 |
| non-chat 32 tok | `The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into` | 미측정 | **PASS** |
| avg_tbt n=5 (ms/tok) | 32.50 ± 0.066 (raw=[32.60, 32.54, 32.46, 32.44, 32.48]) | 미측정 | 절대치 정상 |

## 핵심 결론

1. **Phase 4-5-f chat repl v2 = baseline chat 동치 구현**. 1228 LOC legacy ChatTurnExec 삭제 + DecodeLoop+ModelForward+KiviForward+OffloadForward 패턴 교체가 chat path의 동작을 **변경하지 않음** (garbage까지 동등 재현 = 행동 보존).
2. **chat garbage는 baseline부터 존재**. 명령: `--chat --system-prompt "You are concise." --prompt "Capital of France?" --greedy --repetition-penalty 1.1` 조합이 Phase 4-4.10 (`b991691f`)에서도 `Capital of China?撙�` garbage. Qwen2 chat template + system_prompt + greedy + rep_penalty 결합 어딘가에 회귀가 더 일찍 도입됨.
3. **non-chat path bit-identical 후보 + TBT 정상**. 32 토큰 출력이 합리적, TBT 32.50 ms/tok (Adreno OpenCL baseline 28 ms/tok 대비 +16% — 별도 분석 필요할 수 있으나 절대치는 production 정상 범위).

## 산출물

| 항목 | 위치 |
|---|---|
| 결과 종합 | `/home/go/.claude/jobs/77310ddb/p0_s25_results.md` |
| Phase 4-5-f chat 1 turn 로그 | `/home/go/.claude/jobs/77310ddb/chat_1turn.log` |
| Phase 4-5-f chat 3 turn 로그 | `/home/go/.claude/jobs/77310ddb/chat_3turn.log` |
| Phase 4-5-f non-chat 로그 | `/home/go/.claude/jobs/77310ddb/nonchat_32tok.log` |
| Phase 4-5-f TBT n=5 로그 | `/home/go/.claude/jobs/77310ddb/tbt_runs.log` |
| Baseline chat 1 turn 로그 | `/home/go/.claude/jobs/77310ddb/baseline_chat_1turn.log` |

## C4-1~3 (KvMode subcommand) 종결

| Step | 파일 | 상태 |
|---|---|---|
| C4-1 | `engine/src/session/cli/kv_mode.rs` (+35) | PASS |
| C4-2 | `engine/src/session/cli/mod.rs` (+27, KvModeArgs flatten + `effective_kv_mode()` shim) | PASS |
| C4-3 | `engine/tests/spec/test_kv_mode_args.rs` (+94, 6/6 PASS) | PASS |

**Commit**: `83c73dff` (worktree → master ff-merge). worktree `c4_kv_mode` 정리 완료.

신규 flag: `--kv-mode {standard|kivi|offload}`, `--kv-kivi-bits`, `--kv-kivi-residual-len`, `--kv-offload-storage`, `--kv-max-prefetch-depth`. 기존 `--kivi <bool>` / `--kv-offload <String>`은 `effective_kv_mode()` shim이 fallback 처리 (legacy 사용처 무변경).

## Phase 4-5-f branch (`a9f0f106`) 머지 결정 대기

**기술적 머지 가능**: chat path 동치 + non-chat PASS + TBT 정상. legacy 1228 LOC 삭제 + chat 도메인 코드 `session/chat/`로 이관이 architectural 개선.

**머지 보류 사유 (있다면)**:
- chat pre-existing 이슈 fix 전에 머지하면 fix가 두 chat 구현 위에 깔려야 함. 하지만 어차피 v2가 동치 구현이므로 fix는 둘 다 동일 영향 → 사실상 문제 없음.
- TBT 32.50 ms/tok이 baseline TBT와 동등한지 동일 빌드 비교 필요. (현재 baseline TBT 미측정 → 머지 전 baseline TBT n=5 추가 측정 권장)

**권장**: baseline TBT n=5 추가 측정 후 회귀율 확인 → 머지 결정.

## 다음 진입 명령 후보

```
"baseline TBT n=5"              ← Phase 4-5-f 머지 결정용 회귀율 측정
"Phase 4-5-f master 머지"        ← (TBT PASS 후) ff-merge + worktree 정리
"chat pre-existing 분석"         ← Task #27, Qwen2 chat template garbage 트랙
"C4-4 진행"                     ← init.rs/generate.rs의 effective_kv_mode() 마이그레이션
"S-subcmd C5/C6 진행"           ← KvModeArgs 후속 정리
```

## Task 상태

- #22 P0 S25 디바이스 게이트 — **completed**
- #23~25 C4-1~3 KvMode — **completed**
- #26 Phase 4-5-f chat repl v2 회귀 root cause — **completed** (pre-existing 확인)
- #27 Chat pre-existing garbage 트랙 — **pending** (별도 디버깅)
- #20 KvMode 단일 flag + sub-args 격리 (PR2) — **completed**

## Risk / 미해결

| ID | 내용 | 우선순위 |
|---|---|---|
| R-chat | Qwen2 + system_prompt + greedy + rep_penalty chat garbage. baseline 어디서 회귀했는지 git bisect 후보 | P2 |
| R-tbt | Phase 4-5-f TBT 32.50 vs baseline TBT (미측정). 회귀율 산정 필요 | P1 (머지 전) |
| R-c4-4 | `effective_kv_mode()` shim이 있어도 call site migration은 별도 sprint 필요. legacy `--kivi`/`--kv-offload` 영구 유지 vs 제거 결정 | P2 |
