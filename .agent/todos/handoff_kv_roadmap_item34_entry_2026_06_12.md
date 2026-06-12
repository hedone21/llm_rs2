# Handoff: KV 로드맵 항목 3+4 진입 — QueryStats 구현 + read-plan ADR (워크트리 분리 게이트)

**작성**: 2026-06-12
**스프린트 종결 (2026-06-12)**: P1~P4 ✅ + P5 분리 위생 ②③ ✅(PM) — 분리 위생 문서 = `worktree_split_hygiene_2026_06_12.md`. 잔여 = P5 ① origin 푸시(메인 세션).
**HEAD**: `757ca2a8 docs(kv-roadmap): 항목 0 측정 스프린트 종결 …` (+ 본 handoff 커밋)
**브랜치**: master (worktree 없음, **origin 미푸시 누적** — QCF 세션분 + 측정 스프린트분, 푸시는 사용자 지시 대기)
**다음 세션 진입 문장**: **"KV 로드맵 항목 3 진행 — QueryStats TensorKind 구현부터"** (항목 4 ADR은 3 완료 후 같은 흐름에서)

---

## TL;DR

항목 0 측정 스프린트 종결(판정: R-KV 보류/A2SF 보류/head 분산 개봉 후보/Demote RED — 상세는 `handoff_kv_roadmap_item0_2026_06_12.md`) 후, 사용자 확정으로 차기 = **항목 3(QueryStats TensorKind) → 항목 4(read-plan ADR)**. **새 결정(2026-06-12 대화)**: 사용자는 이 둘이 끝나는 시점을 "KV 캐시 구조 확정" 게이트로 삼아 **별도 워크트리에서 대형 리팩토링을 병렬 시작할 계획** — 따라서 **항목 4는 ADR 작성까지만**(구현 착수 금지, 리팩토링 머지 후 별도 항목으로 재개)이고, 완료 직후 분리 위생 3종을 수행한다. 멈춘 이유 = 세션 종료(다음 세션 위임).

## 진행 상태

| 완료 | 게이트 | 기록 |
|---|---|---|
| 항목 0 측정 스프린트 (4종 판정) | `experiments/kv_roadmap_item0/rerun/REPORT.md` | backlog 트랙 갱신 + `handoff_kv_roadmap_item0_2026_06_12.md` |
| QCF_kv 설계 라운드 | **공동 검토 대기** (사용자 결정 2건: estimator 방향 + floor 재설정) | backlog L1112 항목 = 안건 SSOT |

## 스프린트 마스터 (PM 수립 2026-06-12)

P1~P5 단계 분해 + AC + 리스크 = `.agent/todos/sprint_kv_roadmap_item34_2026_06_12.md`. backlog 항목 3·4 Status = "진행 중 (Sprint 2026-06-12)". 아래 "다음 작업"의 상세 게이트는 스프린트 파일이 SSOT.

## 다음 작업 (순서 고정)

1. **항목 3: `TensorKind::QueryStats` 구현** — per layer·kv_head Q running mean/var + forward 경로 Q 캡처 1지점(`AttentionScoreAccumulator` 패턴 재사용, ADR-0004 §7이 예고한 자리).
   - AC: TensorKind variant + 엔진 누적 배선 + host 통계 정확성 테스트 + 기존 `tensor()` 소비자 무영향 + **실모델 e2e 1회**(항목 0 교훈 — 단위 게이트만으로 완료 선언 금지).
   - Spec triage: technique-api 어휘 추가라 spec 영향 판정 선행 (Architect).
2. **항목 4: read-plan 표면 ADR** — `KVReadStage::read_plan(ctx) → KVReadPlan{granularity, select}`. 대안 비교 + grill 통과 + **구현 단계 분해를 backlog 재등록까지만**. 설계 쟁점: page 메타데이터(K min/max) 유지 주체(1차안 = read stage가 `tensor(Key)`로 자기 상태 갱신, 코어 무수정).
3. **분리 위생 3종** (항목 3+4 완료 직후): ① origin 푸시(공통 베이스 앵커), ② 공통 회귀 게이트 명문화(host lib + α-K frozen 3-dtype byte-identical + S25 verify 매트릭스 28/30 — 양 브랜치 머지 판정 기준), ③ rename 상호 배제 규칙 합의(한쪽만 `git mv`).

## Landmines / 미해결

- **항목 4 구현 착수 금지 — ADR까지가 항목**: ADR-선행의 목적이 병렬 리팩토링과의 **의미적 충돌 예방**(미래 표면을 문서 제약으로 고정)이다. 구현은 리팩토링 머지 후 별도 등록 항목으로.
- **항목 3은 forward 경로 접촉**: Q 캡처가 hot path에 비용을 더하면 안 됨 — 기본 off/score-active 시만 (v1 need_scores 교훈 + score-active 시 plan path 우회 시맨틱 인지). 완료 게이트에 **α-K frozen 재검증**(happy path 무회귀) 포함할 것.
- **accumulator 영역 동시 작업 주의**: QCF_kv 설계 라운드(공동 검토 대기)가 같은 `attention_scores.rs`/`qcf_runtime.rs`를 만질 예정 — 항목 3 캡처 배선은 commit 격리 + 누적 로직 무수정 원칙 유지.
- **동결 권고 반영됨**: 항목 2(1B ablation 회의적)·항목 5(persistence)는 리팩토링 머지 후로 — 리팩토링 기간 중 KV 표면(plan ABI·format trait) 무변경 보장이 목적 (backlog 해당 항목에 표기).
- **origin 미푸시 누적**: 분리 위생 ①이 해소 — 그 전까지 로컬 유일본임을 인지.
- 측정 인프라 신규 표면(rkv feature, `--score-decay`, `--dump-a2sf`, C_h 덤프)은 전부 기본 off — 리팩토링이 정리 대상으로 오인하지 말 것 (측정 재현용 보존).

## 자기점검

- 진입 문장 한 줄로 첫 명령 가능? ✓ ("항목 3 진행 — QueryStats부터")
- 왜 멈췄나? ✓ (세션 종료 — 차기 작업·순서·금지선까지 합의 완료)
- 최대 landmine 표면화? ✓ (항목 4 구현 금지선 + forward hot path 비용 + accumulator 격리)
- 게이트가 수치/명령? ✓ (α-K frozen byte-identical, verify 28/30, e2e 1회 포함 AC)
- 길이 적정? ✓ (측정 상세는 item0 handoff/REPORT.md로 위임)
