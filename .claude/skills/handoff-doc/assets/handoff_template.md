# Handoff: <PHASE/SPRINT 이름> → <다음 작업>

**작성**: YYYY-MM-DD
**HEAD**: `<sha> <commit subject>`
**브랜치 / Worktree**: `<branch>` (`<worktree path if applicable>`)
**작성자**: <agent name or human>

**다음 세션 진입 문장**: "<한 줄, 그대로 입력 가능한 명령>"

---

## TL;DR

<3~5줄. 무엇이 끝났고 / 무엇이 다음이고 / 왜 멈췄는지>

---

## 진행 상태

### Task

| ID | 상태 | 작업 | Commit |
|---|---|---|---|
| #1 | ✅ completed | <작업 요약> | `<sha>` |
| **#2** | **⏳ 이번 세션 진입 대상** | <작업 요약> | - |
| #3 | ⏳ blocked-by #2 | <작업 요약> | - |

### 측정 / 검증 결과

| 항목 | 값 | 비고 |
|---|---|---|
| `cargo test -p llm_rs2 --lib` | <PASS / FAIL count> | 회귀 N |
| <metric e.g. Avg TBT> | <num + 단위> | baseline 대비 ±N% |
| <device gate e.g. S25 32-token bit-identical> | <PASS/FAIL> | <조건> |

---

## 다음 작업 (다음 세션이 그대로 사용 가능)

### 액션

1. **<X 구현>** → 검증: `<Y 명령>` 통과 / `<수치>` 달성
2. **<X 구현>** → 검증: `<Y 명령>` 통과 / `<수치>` 달성
3. **<X 구현>** → 검증: `<Y 명령>` 통과 / `<수치>` 달성

### 위임 prompt (선택)

> **에이전트**: `<implementer | senior-implementer | architect | tester | researcher | pm>`
> **모델**: `<sonnet | opus>`
> **권한**: 수정 가능 경로 — `<engine/src/..., tests/spec/..., ...>`

```
<프롬프트 본문. Agent tool에 그대로 복붙 가능한 형태.>
<목표·제약·검증 게이트를 모두 포함.>
```

---

## Landmines / 미해결 / 안 가본 길

- **<항목 1>**: <설명 + 왜 위험한지>
- **시도했지만 실패한 방향**: <가설> → <반증 데이터/측정값>
- **결정 대기**: <항목 + 누구의 결정 / 어떤 정보를 기다리는지>
- **이 길은 가지 말 것**: <패턴 + 이유>

---

## 참고 링크

- 상위 plan / spec: `<path>`
- 관련 memory 항목: `[[name]]`
- 이전 handoff: `.agent/todos/handoff_<prev>.md`
- 관련 commit / PR: `<sha>` / `#<num>`
