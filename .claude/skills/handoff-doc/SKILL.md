---
name: handoff-doc
description: 세션 또는 단계 종료 시 다음 세션·다른 에이전트·사람이 이어받을 수 있는 handoff 문서를 일관된 포맷으로 작성한다. '인수인계', 'handoff', '다음 세션', '세션 종료', '컨텍스트 정리', 'phase 완료', 'sprint 완료', '작업 일시 중단', '메모리 추가' 등의 요청 시 반드시 이 스킬을 사용. 단계가 끝났는데 진입점이 모호하면 사용자가 명시 요청하지 않아도 능동적으로 제안한다. 단순 커밋·PR 생성·일반 문서 수정에는 트리거하지 않는다.
allowed-tools: Read, Write, Edit, Bash, Glob, Grep
---

# Handoff Document Skill

세션·단계 종료 시점에 **다음 세션이 0에서 시작해도 첫 명령을 칠 수 있게 하는** handoff 문서를 작성한다. 본 프로젝트의 `.agent/todos/handoff_*.md` 컨벤션과 외부 베스트 프랙티스(Context Dump Fallacy 회피, Selective Context Passing, 검증 가능한 진입점)를 합성한다.

## 왜 필요한가

- 다음 세션의 컨텍스트는 0에서 시작한다. 진입 문장 한 줄이 없으면 "어디서부터?"에 토큰을 낭비한다.
- 풀 트랜스크립트를 그대로 넘기면 노이즈가 늘어 다운스트림 reasoning이 저하된다 (Context Dump Fallacy — typed context 200~500 토큰이 raw 5K~20K 토큰보다 의사결정 품질이 높다).
- 작성 시점이 늦으면 직전 분기·landmine·암묵 결정이 휘발한다. 단계 종료마다 갱신해야 한다.

## 출력물

### 기본 — `.agent/todos/handoff_<topic-snake-case>_<YYYY_MM_DD>.md`

다음 세션이 즉시 참조하는 진입 문서. 예시 (실제 파일):
- `handoff_phase4_4_entry_2026_05_17.md`
- `handoff_liswap5_crash_fix_2026_05_10.md`
- `handoff_b_3a_auf_device_correctness_2026_05_20.md`

### 장기 보존 — `memory/project_<topic>.md` + `MEMORY.md` 라인

다음 한 세션이 아니라 향후 여러 세션이 참조할 가치가 있을 때만 추가:
- 메모리 파일: 압축본(프론트매터 + 1~2단락 요약 + `.agent/todos/handoff_*.md` 링크)
- `MEMORY.md`: `- [Title](project_<topic>.md) — one-line hook` 한 줄

단기 인계만 필요하면 `MEMORY.md`는 건드리지 않는다. (MEMORY.md는 매 대화 강제 로드되므로 무게를 늘리지 말 것)

## 필수 6요소 (R1~R6)

본문은 아래 6개를 반드시 포함한다. 빠지면 다음 세션이 그 부분을 재구성하느라 토큰을 낭비한다.

### R1. 헤더 메타

| 항목 | 수집 명령 |
|---|---|
| 작성일 (YYYY-MM-DD) | `date +%Y-%m-%d` |
| HEAD SHA + commit subject | `git log -1 --oneline` |
| 브랜치 / worktree | `git rev-parse --abbrev-ref HEAD`, `git worktree list` |
| 작성자 (에이전트명 또는 사람) | 본인 |

### R2. 진입 문장 — 가장 중요

다음 세션이 **첫 입력으로 그대로 칠 수 있는 한 줄**.

| | 예 |
|---|---|
| 좋음 | `"B-3a 진행"` · `"Phase 4-4 진행"` · `"LISWAP-5 risk 검증부터"` |
| 나쁨 | `"이어서 작업해줘"` · `"다음 단계 진행"` · `"TODO에 적힌 거 해줘"` |

**이유**: 진입 문장은 검색 키워드이자 첫 명령이다. 모호하면 다음 세션이 "어디서 멈췄지?" 탐색에 들어간다. handoff 작성 시 가장 먼저 결정하면 나머지가 자연스럽게 따라온다.

### R3. TL;DR (3~5줄)

한 단락에 세 가지를 담는다:
1. 무엇이 끝났는가
2. 다음이 무엇인가
3. 왜 멈췄는가 (시간 / 결정 대기 / 막힌 이슈 / 다음 세션에 위임)

### R4. 진행 상태 / 측정 결과

검증 가능한 형태로 — 정성 문장 금지.

- 통과한 테스트 게이트: `cargo test -p llm_rs2 --lib`: 1159 PASS, 회귀 0
- 실측 수치 (단위 포함): `TBT 14.66 ms/tok = Q4 baseline -10.1%`
- 완료 task의 commit SHA
- 권장: task ID × 상태 × 작업 × commit 표

### R5. 다음 액션

- **"X 구현 → Y로 검증"** 형식. 검증 게이트(통과 조건)를 수치/명령으로 명시.
- 위임 대상 에이전트가 정해졌다면 prompt 초안까지 포함 — 다음 세션이 Agent tool에 그대로 붙여넣기 가능.
- 최대 3~5개. 그 이상이면 분해 부족 신호 → 더 쪼개거나 별도 plan으로 옮긴다.

### R6. Landmines / 미해결 / 안 가본 길

다음 세션이 같은 실수를 반복하지 않도록:
- 깨지기 쉬운 가정 (예: "post-swap CPU forward는 baseline부터 garbage — 별도 issue로 분리됨")
- 시도했지만 실패한 방향 + 반증 데이터 (가설을 기각한 측정값까지)
- 결정 대기 항목 (누구에게 묻고 있나)
- "이 길은 가지 마라" 경고

웹 베스트 프랙티스에서 가장 강조된 "document dump 안티패턴"의 반대 — landmine을 명시해야 다음 세션이 같은 함정에 빠지지 않는다.

## 금지 (N1~N4)

| | 안티패턴 | 이유 |
|---|---|---|
| N1 | 풀 트랜스크립트 / 전체 로그 dump | Context Dump Fallacy — 노이즈 증가, reasoning 저하 |
| N2 | 코드 본문 복붙 | `engine/src/foo.rs:123-145` 포인터로 충분. 코드는 변하지만 handoff는 frozen → stale 위험 |
| N3 | 정성 문장만 ("잘 됨", "정리 완료") | 수치/게이트 없는 완료 주장은 다음 세션에서 재확인을 강제한다 |
| N4 | 막판 일괄 작성 | 단계 종료 시점마다 갱신하지 않으면 직전 분기·암묵 결정이 휘발한다 |

## 자기점검 체크리스트

작성 완료 전 본인이 답한다. 하나라도 No면 그 항목을 보강:

- [ ] 진입 문장 한 줄만으로 다음 세션이 첫 명령을 칠 수 있는가?
- [ ] "왜 멈췄는가"가 명시되어 있는가? ("TODO" 단독 ❌)
- [ ] 가장 큰 landmine이 표면화되어 있는가?
- [ ] 검증 게이트가 수치 또는 실행 가능한 명령으로 표현되어 있는가?
- [ ] 본문 길이가 적정한가? (~500 토큰 권장. 넘는 상세는 외부 파일/링크로 분리)

## 작성 절차

### 1. 메타 수집

```bash
git log -1 --oneline
git rev-parse --abbrev-ref HEAD
git status --short
git worktree list | grep -F "$(pwd)" || true
```

### 2. 파일명 결정 + 템플릿 복사

```bash
TOPIC="<topic-snake-case>"        # 예: liswap5_risk, phase4_4_entry
DATE=$(date +%Y_%m_%d)
TARGET=".agent/todos/handoff_${TOPIC}_${DATE}.md"
cp .claude/skills/handoff-doc/assets/handoff_template.md "$TARGET"
```

### 3. R1~R6 채우기

순서 권장: **R2 진입 문장 → R3 TL;DR → R5 다음 액션 → R4 측정 → R6 landmines → R1 메타**.
진입 문장을 먼저 확정하면 나머지가 자연스럽게 정렬된다.

### 4. (선택) MEMORY.md 업데이트

장기 보존 가치가 있을 때만:
1. `memory/project_<topic>.md` 생성 (프론트매터: `name`, `description`, `type: project`)
2. 본문 = 1~2단락 요약 + `.agent/todos/handoff_*.md` 링크
3. `MEMORY.md` 적절한 섹션에 한 줄 추가

판단 기준: "후속 세션 3개 이상이 같은 정보를 다시 찾아볼 가치가 있나?" → Yes면 메모리, No면 handoff만.

### 5. 자기점검 통과 후 커밋

CLAUDE.md "완료 시 자동 커밋" 규칙. 커밋 메시지 예시:

```
docs(handoff): <topic> 진입 문서 작성

다음 세션 진입 문장: "<R2>"
```

### 6. 사용자에게 알림

```bash
notify-send "llm.rs" "handoff_<topic> 작성 완료"
```

## 구조 참고

`.agent/todos/handoff_layered_arch_step1_complete_2026_05_16.md`가 실측 좋은 예. 골격:

```
# Handoff: <Phase/Sprint> → <다음 작업>

**작성**: <YYYY-MM-DD>
**HEAD**: `<sha> <subject>`
**다음 세션 진입 문장**: "<R2>"

---

## TL;DR
<R3>

---

## 진행 상태
<R4: task table + 측정 수치>

---

## 다음 작업
<R5: 액션 + 검증 게이트 + 위임 prompt>

---

## Landmines / 미해결
<R6>
```

## 트리거 보조 메모

- "인수인계" / "handoff" / "다음 세션" / "세션 종료"
- "phase 완료" / "sprint 완료" / "작업 일시 중단" / "컨텍스트 정리"
- "메모리 추가" / "memory 기록" (단, project 타입에 한함)

단계 완료가 감지되었는데 사용자가 명시 요청을 하지 않았다면 능동적으로 "handoff 작성할까요?" 제안한다. 단순 커밋·PR·일반 문서 수정에는 트리거하지 않는다.
