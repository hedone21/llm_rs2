# Sprint: QCF_kv 설계 라운드 — 수식 비대칭 + estimator B안 + floor 재설정 (2026-06-12)

**안건 SSOT**: `backlog.md` "[P2] QCF_kv 정규화 비대칭 + estimator 우회 + manager floor 재설정"
**진입**: `handoff_qcf_kv_design_round_entry_2026_06_12.md`
**결정 (2026-06-12 사용자 확정)**:
- ① estimator 우회 해소 = **B. raw 직송 합법화** — 엔진 코드 무변경, spec ENG-ALG-050 step 4 개정. IPC 단위 = ENG-ALG-051 raw 상대 섭동 [0,1].
- ② QCF_FLOOR = **A. 이번 라운드 재설정 포함** — 수식 수정 후 S25 양 가족 분포 실측 → 4임계 + V_Q 재산정 (raw 단위).

PM 단계는 생략 — 계획이 backlog AC ①~⑤로 이미 완결돼 있어 본 문서가 backlog AC의 실행 순서 전개임.

## 단계

- [ ] **P1 설계 (Architect)**: spec ENG-ALG-051 수식 개정(o_before 정규화 + KIVI 항 정합) + ENG-ALG-050 step 4 B안 개정 + ENG-ALG-060 위상 재정의 + `docs/qcf_taxonomy.md` §2.1/§2.2.1 동기화 + **키 전수 감사 보고**(엔진 송출↔manager 소비 전 구간) + d2o 테스트 재보정 명세 + floor 재산정 방법론
  - 검증: spec/docs 동기화 diff + 키 매핑 표 + Senior Implementer에게 전달 가능한 구현 지시(파일:라인)
- [ ] **P2 구현 (Senior Implementer)**: `qcf_kv.rs` o_before 정규화 + KIVI 항 정합 + `test_d2o_less_than_h2o` 재보정(실분포 근사) + 키 감사 최소 수정
  - 검증: host lib 0 FAIL + fmt + clippy. `attention_scores.rs`/`qcf_runtime.rs` 접촉 시 commit 격리(QueryStats 배선과 분리)
- [ ] **P3 호스트 게이트 (Tester)**: lib 전수(1410+ 기준) + spec 테스트(`cargo test --test spec`) + `scripts/check_spec_coverage.sh`
- [ ] **P4 S25 분포 측정 + floor 재산정**: 양 가족(QCF_kv·QCF_weight) 분포 실측 → 4임계 + V_Q 산정(근거 수치 기록) → `policy_default.lua` 반영
- [ ] **P5 회귀 게이트**: `signal_memory_critical` f16/q4 S25 verify GREEN → 전체 매트릭스 **30/30** 봉인
- [ ] **P6 종결**: backlog RESOLVED + CLAUDE.md QCF 섹션 문구 정합(B안 반영) + 메모리/handoff 갱신 + α-K frozen 3-dtype 무회귀 확인(결정론 라인 diff)

## 게이트 요약

- 수식 수정 후 QCF 기대값: 0.985 포화 → **0.08~0.22** (S25 기실측)
- happy path 무회귀: α-K frozen 3-dtype 결정론 라인 diff + tbt Δ≤+3% (`worktree_split_hygiene_2026_06_12.md` ②(b))
- 최종: verify 매트릭스 30/30 (현 28/30 + known-fail 2 해소)

## Landmines

- **키 불일치 = silent-0 클래스** (v1 가짜 PASS 전례) — 감사 없이 키 rename 금지. 엔진 IPC 키 변경은 manager+fixture 동시 영향이라 최소 수정 원칙.
- 워크트리 분기 R1: **git mv 금지** (신규 파일만 허용).
- layer-0 단일 proxy → 전 layer aggregate는 별개 backlog [P1] — 본 라운드 범위 아님 (scope creep 금지).
- estimator(`estimator.rs`)는 B안에서 live 미장착 유지 — 삭제 금지(향후 캘리브레이션 도구 보존), spec 위상만 재정의.
