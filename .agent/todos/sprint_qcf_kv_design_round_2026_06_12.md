# Sprint: QCF_kv 설계 라운드 — 수식 비대칭 + estimator B안 + floor 재설정 (2026-06-12)

**안건 SSOT**: `backlog.md` "[P2] QCF_kv 정규화 비대칭 + estimator 우회 + manager floor 재설정"
**진입**: `handoff_qcf_kv_design_round_entry_2026_06_12.md`
**결정 (2026-06-12 사용자 확정)**:
- ① estimator 우회 해소 = **B. raw 직송 합법화** — 엔진 코드 무변경, spec ENG-ALG-050 step 4 개정. IPC 단위 = ENG-ALG-051 raw 상대 섭동 [0,1].
- ② QCF_FLOOR = **A. 이번 라운드 재설정 포함** — 수식 수정 후 S25 양 가족 분포 실측 → 4임계 + V_Q 재산정 (raw 단위).

PM 단계는 생략 — 계획이 backlog AC ①~⑤로 이미 완결돼 있어 본 문서가 backlog AC의 실행 순서 전개임.

## 단계

- [x] **P1 설계 (Architect)** ✅ `eab908ab`: spec ENG-ALG-051 수식 개정 + ENG-ALG-050 step 4 B안 + ENG-ALG-060 위상 재정의 + qcf_taxonomy 동기화 + 키 전수 감사(불일치 2건: streaming 표기 혼재 silent-0 실결함 + estimator dead 키) + d2o 재보정 명세 + floor 방법론
- [x] **P2 구현 (Senior Implementer)** ✅ `753fb257`(수식+d2o 재보정+spec 테스트 3종) + `ddc2c2fb`(streaming 키 정렬) + `d6fcf376`(estimator 키 정렬)
  - 검증: lib 1410/0, spec 683/4(사전존재), manager 0 FAIL, β regression PASS. 항등→QCF=0, 포화 시나리오 0.085(구 0.985)
- [x] **P3 호스트 게이트** ✅: lib 1410/0 + spec 테스트 + coverage 스크립트(신규 누락 0, 기존 갭 45 불변)
- [x] **P4 S25 분포 측정 + floor 재산정** ✅ `5454e564`+`1a8ee375`: 66샘플(h2o 0.031~0.168/d2o 0.086~0.243/sliding 0.261~0.389, 포화 0) → QCF_FLOOR 0.10/0.25/0.50/huge + V_Q 0.5 유지(근거: REPORT.md). weight.skip live 미산출 확인(단일 가족 산정 정당)
- [x] **P5 회귀 게이트 (1차)** ✅: `signal_memory_critical` f16/q4 S25 **PASS** — 추가 결함 2건 적발·해소: ④ heartbeat available_actions 결함(`09a82ad9`) ⑤ 시나리오 물리적 통과 불가 파라미터(`7397512c` — MIN_EVICT_TOKENS/emergency 압력/prefill 주입 타임아웃/accuracy 임계). QCF 핸드셰이크 신호 유실은 backlog 신규 등록
- [x] **P5 봉인** ✅: 전체 매트릭스 **30/30 PASS** (`verify/results/20260612_165122_galaxy_s25_f16_q4/`) — 구 known-fail 2건 해소, 신규 회귀 0
- [x] **P6 종결** ✅: backlog RESOLVED + CLAUDE.md QCF 섹션 정합 + α-K frozen 3-dtype **byte-identical** + tbt median f16 54.34(Δ+0.2%)/f32 54.53(Δ+0.9%)/q4 53.45(Δ−0.6%) 전부 ≤+3% (초기 초과분은 매트릭스 직후 열누적 — 쿨다운 후 재측정으로 해소 확인)

## 종결 (2026-06-12)

**Status: ✅ 라운드 완주** — verify 매트릭스 30/30 봉인. 잔여 1건(QCF 핸드셰이크 신호 유실)은 backlog 신규 [P2]로 이관 (spec SEQ-098 개정 동반이라 별도 라운드).

## 게이트 요약

- 수식 수정 후 QCF 기대값: 0.985 포화 → **0.08~0.22** (S25 기실측)
- happy path 무회귀: α-K frozen 3-dtype 결정론 라인 diff + tbt Δ≤+3% (`worktree_split_hygiene_2026_06_12.md` ②(b))
- 최종: verify 매트릭스 30/30 (현 28/30 + known-fail 2 해소)

## Landmines

- **키 불일치 = silent-0 클래스** (v1 가짜 PASS 전례) — 감사 없이 키 rename 금지. 엔진 IPC 키 변경은 manager+fixture 동시 영향이라 최소 수정 원칙.
- 워크트리 분기 R1: **git mv 금지** (신규 파일만 허용).
- layer-0 단일 proxy → 전 layer aggregate는 별개 backlog [P1] — 본 라운드 범위 아님 (scope creep 금지).
- estimator(`estimator.rs`)는 B안에서 live 미장착 유지 — 삭제 금지(향후 캘리브레이션 도구 보존), spec 위상만 재정의.
