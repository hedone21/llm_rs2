# Handoff: Phase 4-5 chat 전면 재작성 종결 → 다음 진입점

**작성**: 2026-05-18
**HEAD**: `e5e9155a` (master, Phase 4-5-g hotfix + chat garbage backlog 등록 직후)
**다음 세션 진입 후보** (사용자 선택):
- (A) "Phase 4 #4 잔여 + #5 진입" — main() eval/ppl/batch 추출 + L1/L2 경계
- (B) "Qwen2 chat garbage 격리 진행"
- (C) "Weight Swap Layer-Level Mixed Precision Phase A 진입"
- (D) "Long context CPU attention 진입"
- (F) M3.4 D-D/D-E 결정 (사용자 직접)

---

## TL;DR — Phase 4-5 종결 결과

| Gate | 결과 |
|---|---|
| **G6' bit-identical 32 tok (non-chat)** | **PASS** — Phase 4-5+hotfix `e5e9155a` 출력이 baseline `b991691f`과 32 토큰 완전 동일 ("The capital of France is Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into") |
| **G7' avg_tbt n=5 (non-chat)** | **PASS Δ -0.28%** (baseline median 32.22 ms / Phase 4-5+hotfix median 32.13 ms, 미세 개선) |
| **G1 /stats 라인 동치** | PASS (`ChatSession::stats_line` 3 variant 포맷 보존, spec test) |
| **G2 multi-turn 2nd-turn bit-identical** | PASS (`test_chat_repl_v2_multi_turn` + Phase 4-5-g hotfix 후 보장) |
| **G3 /reset 동작 동치** | PASS (`ChatSession::reset` 3단 atomic) |
| **G4 chat-specific eviction** | PASS (`try_evict` + 90% threshold) |
| **G5 grep gates** | PASS (`ChatTurnExec` 0건 / `core::chat_ipc` 0건) |
| **Chat E2E smoke (S25 Qwen2.5-1.5b Q4_0)** | ⚠️ baseline에서도 garbage 재현 → Phase 4-5 책임 아닌 별도 P1 backlog 등록 |

## 핵심 변경

| Commit | 내용 |
|---|---|
| `6ac7f752` 4-5-a | KiviForward + OffloadForward 도입 (ModelForward 패턴 복제, plan path 제외) |
| `80baf155` (style) | cargo fmt 누락분 |
| `dbd97466` 4-5-c | StopCondition trait + `DecodeLoop::run_until_stop` (finalize 미호출) + `reset_pos` + `forward_mut` |
| `b5ff653f` 4-5-b | `core/chat_ipc.rs` → `session/chat_ipc.rs` 이관 (V-11 해소, shim 유지) |
| `85d1ca5d` 4-5-d | ChatSession + 3 KV-mode builder + Forward::reset_kv (multi-turn KV invariant). `try_evict` (R3 미완) + multi-turn prefill (R1 미완) 두 건 4-5-e로 deferred |
| `dcce2ae5` 4-5-e | run_chat_repl_v2 (`session/chat/repl.rs`) + 4-5-d deferred 일부 해결 (force_evict completion). multi-turn prefill은 여전히 미해결 |
| `a9f0f106` 4-5-f | legacy ChatTurnExec/run_chat_repl/3 impl/3 wrapper 1,228 LOC 삭제 + main dispatch v2 교체 + core/chat_ipc.rs shim 삭제 |
| `c1a4b481` 4-5-g | **hotfix: Forward::prefill에 start_pos 인자 추가** — 디바이스 게이트 발견 회귀(첫 turn조차 garbage)의 진짜 root cause 해결. ModelForward/KiviForward/OffloadForward 3종 + DecodeLoop::prefill + 4 mock + 1 호출자 시그니처 갱신. |
| `e5e9155a` (backlog) | Qwen2.5-1.5b chat baseline garbage P1 등록 (Phase 4-5 책임 아님 격리 결과) |

## main() LOC delta

11,032 → 9,860 (**-1,172 net**). chat 분기 1,228 LOC 삭제 + 신규 dispatch ~90 LOC.

eval-ll / ppl / batch 분기는 본 sprint 범위 외로 main()에 잔존. Phase 4 #4 잔여 작업으로 별도 sprint 필요.

## 디바이스 측정 데이터

`$CLAUDE_JOB_DIR/phase4_5_device/` (job-local):
- `chat_1turn_smoke.log` — Phase 4-5 chat 첫 시도 (garbage `ЂЂЂ...`)
- `chat_1turn_hotfix.log` — Phase 4-5+hotfix chat (garbage `İlçe ` 반복, baseline도 동일이라 격리)
- `chat_baseline.log` — baseline b991691f chat (garbage `撙afournis` — 같은 패턴 다른 토큰)
- `g6_baseline.log` / `g6_phase4_5.log` — non-chat 32 tok bit-identical "Paris. It has a population..."
- `g7_baseline_n5.log` — baseline TBT [32.22, 32.32, 32.30, 32.07, 32.21] median 32.22
- `g7_phase4_5_n5.log` — Phase 4-5+hotfix TBT [32.29, 31.99, 31.75, 32.13, 32.22] median 32.13

## Phase 4 sub-phase 진행도

| Sub-phase | 상태 | 결과 |
|---|---|---|
| 4-1 외곽 추출 | ✅ `f637722e` | `session/init.rs` (1,030 LOC), `session/cli.rs` |
| 4-2 trait + Builder | ✅ `584496b7` | 6 trait + DecodeLoopBuilder + INV-LAYER-006/007 |
| 4-3 ModelForward + microbench | ✅ host + S25 | Δ ≤ 2.29% bit-identical |
| 4-4 main() 조립자화 (standard happy path) | ✅ a/b/d (c skip) | DecodeLoop+ModelForward 진입 |
| 4-4.5~10 paradigm/equivalence/plan/noshuffle | ✅ | G6'/G7' PASS Δ 0.00% (default AOS invert) |
| **4-5 chat 전면 재작성** | ✅ **6 sub-step + hotfix** | G1~G5 PASS, ChatTurnExec 0건, main() -1,172 LOC, multi-turn KV invariant 보장 |
| **4-X eval-ll / ppl / batch 분기 추출** | ⏳ **다음 진입 후보 A** | main() 9,860 → ~4,500 LOC 목표 |

## 다음 진입 후보 상세

### (A) Phase 4 #4 잔여 + #5 진입 — main() eval/ppl/batch 추출

- **목표**: main()에 잔존하는 3개 분기를 `session/eval/`, `session/ppl/`, `session/batch/` 모듈로 분리. Phase 4 task #5 (L1/L2 경계 정리) 진입 가능.
- **추정**: main() 9,860 → ~4,500 LOC (chat과 유사 패턴, 각 분기 ~1,500 LOC 추정).
- **검증 게이트**: 각 분기별 e2e (`--eval-likelihood`, `--ppl-dataset`, `--batch-prompts`) bit-identical + cargo test 회귀 0.
- **risk**: chat 작업과 패턴 유사. R1~R5 매핑은 chat 대비 낮음 (multi-turn KV 없음).
- **선행 작업**: 없음 (Phase 4-5 완료로 모든 의존 해소).

### (B) Qwen2.5-1.5b chat garbage 격리 (P1 backlog)

- **목표**: chat garbage root cause 격리. Phase 4-5 hotfix는 multi-turn KV는 보장하므로, 본 garbage는 다른 layer 문제.
- **검증 방법** (backlog.md):
  1. Llama-3.2-1B chat 모드 같은 prompt 측정 — 정상이면 Qwen2 chat template 이슈 확정
  2. Qwen2.5-1.5b non-chat 모드에서 `"You are concise.\nCapital of France?"` 직접 prefill — 정상이면 chat REPL의 template 적용 단계 issue
- **추정**: 측정 + 분석 2~3시간. fix은 별도 sprint.
- **선행 작업**: 없음. Qwen 모델 + Llama 모델 디바이스 배포 상태 확인.

### (C) Weight Swap Layer-Level Mixed Precision Phase A

- **목표**: 정적 F16/Q4_0 layer mix로 PSS 100~150 MB 감소.
- **상세**: backlog.md `[P0] Weight Swap — Layer-Level Mixed Precision & Dynamic Swap`
- **추정**: Phase A만 2~3주. Phase B/C는 후속.
- **선행 작업**: Architect spec 시작.

### (D) Long context CPU attention

- **목표**: 4K context decode 35% → 80% (NEON GQA fused kernel).
- **상세**: backlog.md `[P0] Long context CPU attention 최적화`
- **추정**: 측정+설계 완료, 구현 1주.
- **선행 작업**: Senior Implementer NEON 위임.

### (F) M3.4 D-D/D-E 결정 (사용자)

- backlog.md `[P0] M3.4 RED — pos baked architectural blocker`
- D-D: M2 ops 수정 +1.5주 / D-E: scope 약화 +0.5주
- 본인 진행 불가, 사용자 결정만.

## 권장 (본인 판단)

**A + B 두 트랙 병렬**:
- A는 Phase 4-5의 자연스러운 연속 (layered arch 외부 공개 직전 정합성).
- B는 본 sprint에서 발견한 P1 회귀로 빠른 디바이스 격리 (fix 별도). 책임 분담 명확화.

C/D는 sprint 단위가 크고 우선순위 사용자 결정 필요.

## 측정 절차 (재현용)

```bash
# 워크트리 진입 (격리)
cd /home/go/Workspace/llm_rs2/.claude/worktrees/phase4_5_chat_rewrite

# 디바이스 빌드 + 배포
python scripts/run_device.py -d galaxy_s25 --skip-exec generate

# G6' non-chat bit-identical (Phase 4-5 vs baseline)
adb -s R3CY408S5SB shell '</dev/null cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate \
    --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512 --greedy --repetition-penalty 1.1'

# G7' avg_tbt n=5
adb -s R3CY408S5SB shell 'for i in 1 2 3 4 5; do </dev/null cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./generate \
    --model-path /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512 --greedy --repetition-penalty 1.1 \
    2>&1 | grep "Avg TBT"; done'

# Chat baseline 격리용 baseline worktree
# /home/go/Workspace/llm_rs2-baseline (b991691f detached HEAD, third_party/libs 수동 setup됨)
cd /home/go/Workspace/llm_rs2-baseline && python scripts/run_device.py -d galaxy_s25 --skip-exec generate
```

## 환경 / 규칙 (불변)

- 언어: 한국어 (CLAUDE.md 시스템 지시)
- 자동 commit + `notify-send`
- GGUF 우선, `.cl` 커널은 성능 최적화 시만 수정
- 신규 test는 `engine/tests/spec/` 하위
- TBT metric은 avg_tbt (tok0 inclusive)
- Adreno 벤치 = Galaxy S25 6T
- baseline 비교는 detached HEAD checkout 또는 별도 worktree (현재 `/home/go/Workspace/llm_rs2-baseline` 활성)

## 미해결 (다음 sprint 후속)

1. **Qwen2.5-1.5b chat garbage** (P1, backlog 등록) — 격리 후 fix는 별도
2. **Path B Adreno noshuffle GEMV** (P2, backlog) — Phase 4-4.10 cleanup
3. **eval-ll/ppl/batch main() 분기** — Phase 4 #4 잔여 (다음 진입 후보 A)
4. **Phase 4 #5~#8** (L1/L2, L3 도메인, cross-cutting, /simplify) — Phase 4 마무리

## 참조 문서

- `arch/inference_pipeline.md` §9~§11 — Phase 4-5 sub-step 분해 + R1~R5
- `ARCHITECTURE.md` §13 — Layered architecture
- `spec/41-invariants.md` §3.26 — INV-LAYER-001~007
- `engine/src/session/chat/{mod,session,repl,stop_condition}.rs` — Phase 4-5 산출물
- `engine/src/session/forward/{model,kivi,offload}_forward.rs` — Forward trait 3 구현체
- `engine/tests/spec/test_chat_session_multi_turn.rs` (17 test) + `test_chat_repl_v2_multi_turn.rs` (G2 bit-identical)
- `.agent/todos/backlog.md` — P0~P3 미배정 작업 (40개)
