# R-tbt + R-chat 분석 종결 (Phase 4-5-f 통합 검증 완료)

**날짜**: 2026-05-19
**master HEAD**: `619dd655` (Phase 4-5 sprint 전체 + C4-1~3 + handoff 머지 완료)

## 한 줄 요약

R-tbt 회귀는 **측정 메트릭 혼용 착오** (회귀 없음 +0.40%), R-chat garbage는 **base 모델 학습 부족** (llama.cpp도 동일 garbage 재현으로 code 무관 확정). Phase 4-5 sprint 전체가 이미 master에 머지된 상태 확인.

## R-tbt: 회귀 없음 확정 (Task #30)

| 메트릭 (n=5) | baseline (`b991691f`) | Phase 4-5-f (`a9f0f106`) | Δ |
|---|---|---|---|
| Decode (rest_tbt, tok0 제외) | 30.41 ± 0.13 | 30.54 | +0.43% |
| **Avg TBT (tok0 포함)** | **32.37** | **32.50 ± 0.07** | **+0.40%** |

이전 보고 `-6.42%`의 정체: ablation 보고서가 `baseline rest_tbt 30.41` vs `Phase 4-5-f Avg TBT 32.50`을 교차 비교. `feedback_tbt_metric_tok0_inclusive.md` 원칙 위반.

**코드 path 검증**: `step_ctx() 7회/step`, `Box<dyn> vtable 6회`, `try_build_plan`, `forward_into` fallback, `NoOpEvictionStage/SwapStage`, `RepetitionPenaltySampler` 모두 baseline ↔ Phase 4-5-f **byte-level 동일**. Phase 4-5-f 신규 overhead 0건.

## R-chat: base 모델 학습 부족 확정 (Task #27)

**Ablation (S25)**:
- chat off → 정상 ("Capital of France? No, it's not Paris. It is the city of Bordeaux...")
- greedy off (sampling) → 정상 (noise로 garbage 탈출)
- rep_penalty/system_prompt 단독 → garbage 유지 (영향 없음)

**호스트 llama.cpp 결정적 검증** (Reference impl):
- raw "The capital of France is" → 정상 "Paris. Paris is a very big city. It has about 2.2 million people..."
- ChatML system+user prompt → garbage `撙\n撙\n撙\n...` (baseline S25와 동일 패턴)
- ChatML user-only prompt → garbage `Capital of France?撙�\n撙\n撙...`

**Root cause 확정**: `models/qwen2.5-1.5b/` 디렉토리의 모델이 **base variant** (NOT Instruct):
- 디렉토리명 `qwen2.5-1.5b` (-Instruct 접미 없음)
- `eos_token_id: 151643` = `<|endoftext|>` (Instruct는 보통 151645 `<|im_end|>`)
- `do_sample: false` (base 기본값)
- `architectures: Qwen2ForCausalLM` (base 시그너처)

ChatML markers (`<|im_start|>`/`<|im_end|>`)에 대한 응답 distribution 학습 안 됨 → greedy argmax가 비정상 BPE 토큰 `撙` (U+649A) 고정. Sampling 켜면 noise로 탈출.

**해결책**: `Qwen2.5-1.5B-Instruct` 다운로드 (HuggingFace `Qwen/Qwen2.5-1.5B-Instruct`). Task #29에서 검증 진행 중.

backlog 갱신: `.agent/todos/backlog.md`의 `[P1]` 항목을 `[RESOLVED — model issue]`로 격상 + llama.cpp 증거 추가.

## Phase 4-5 sprint master 머지 상태

`git merge-base master 75edb358 = 75edb358` 확인 → **Phase 4-5 sprint 전체가 이미 master에 머지됨**:
- 4-5-a (KiviForward+OffloadForward) `6ac7f752`
- 4-5-b (chat_ipc 이관) `b5ff653f`
- 4-5-c (StopCondition + run_until_stop) `dbd97466`
- 4-5-d (ChatSession) `85d1ca5d`
- 4-5-e (run_chat_repl_v2) `dcce2ae5`
- 4-5-f (legacy ChatTurnExec 1228 LOC 삭제) `a9f0f106`
- **4-5-g (Forward::prefill start_pos 추가, multi-turn KV 보존) `c1a4b481`**
- handoff `75edb358`, backlog `e5e9155a`

P0 게이트 측정값이 사실상 master 코드를 평가한 셈. 추가 머지 작업 불필요.

## 산출물

| 항목 | 위치 |
|---|---|
| TBT 회귀 분석 (implementer 결과) | task notification 안에 인라인 |
| R-chat ablation (tester 결과) | `/home/go/.claude/jobs/77310ddb/r_tbt_r_chat_ablation.md` |
| llama.cpp 호스트 검증 | (본 turn 인라인 명령 결과) |
| backlog 갱신 | `.agent/todos/backlog.md` (P1 → RESOLVED) |

## Task 상태

- #27 R-chat root cause — **completed** (base 모델 확정)
- #28 R-tbt + ablation 측정 — **completed**
- #29 Instruct 다운로드 + 검증 — **pending** (background tester `adea1fad`)
- #30 TBT 회귀 분석 — **completed** (회귀 없음, 측정 착오)

## 다음 진입 명령 후보

```
"Instruct 결과 대기"          ← Task #29 background tester 완료 알림 후 종합 갱신
"Qwen2 base 모델 default 교체" ← scripts/run_device.py 또는 generate.rs default를 Instruct로
"C4-4 진행"                   ← effective_kv_mode() shim의 call site migration (S-subcmd 다음 단계)
"S-subcmd C5/C6/C9/C10"        ← KvModeArgs 후속 정리
```
