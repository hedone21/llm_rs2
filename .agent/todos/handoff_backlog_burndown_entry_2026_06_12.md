# Handoff: Backlog Burndown 진입 — 잔여 백로그 전체 위임 실행 (사용자 컨펌 완료)

**작성**: 2026-06-12
**HEAD**: `c9930991 docs(pm): Backlog Burndown 마스터 플랜` (+ 본 handoff 커밋, origin push 포함)
**브랜치**: master
**다음 세션 진입 문장**: **"Backlog Burndown 진행"** — 첫 액션 = T1(위생 일괄) 착수, 질문 없이 바로 실행

---

## TL;DR

사용자가 잔여 백로그 전체 처리를 위임(2026-06-12)하고 마스터 플랜을 컨펌("다음 세션에서 goal로 진행"). 멈춘 이유 = 세션 경계(플랜 확정까지 완료, 실행은 다음 세션). **플랜 SSOT = `sprint_backlog_burndown_2026_06_12.md`** — 위임 결정 표(A/B/C) + 처분 7건 + 트랙 T1~T6 + 결정점 예고 D-1~D-3 전부 수록. 진입 즉시 T1부터 순차 실행하며, 트랙 종결마다 origin 푸시 + 트랙 보고.

## 위임 운영 규칙 (사용자 확정 — 재질문 금지)

- **범위 제외**: paper/측정 트랙(A1) · 5월 성능 잔존군 M2/M3.4/WSWAP-6/mixed-precision/long-context/GEMV/decode-gap(A2) · repo 밖 OSS FC(A4)
- **방향**: 8B 온보딩 없음(B1=NO — 1B 성능 주장 금지 유지) · 멀티세션 1급 지원 없음(B2=NO — 항목 9 DROP 처분 완료)
- **운영**: stale 재량 처분+기록+일괄보고(C1) / **트랙 종결마다 origin 자동 푸시**(C2) / 디바이스 제약 없음·게이트 측정은 쿨다운 규율(C3) / 트랙 단위 보고(C4) / **막힌 결정 = 보수적 디폴트(동작 불변)+기록, 비가역(삭제·spec 의미 분기·성능 회귀 수용)만 질문**(C5)
- **B3**: T5(weight swap 역전 + action 반복 방지 + h-1 거취)는 설계안 도출 후 D-1~D-3 **일괄 질문** — 그 전까지 질문 없이 진행

## 진행 상태

- re-triage 완료: 즉시 처분 7건(backlog Status 갱신 완료 — Format 명명 RESOLVED, 항목 9 DROP, 항목 2 보류 등)
- 실행 트랙: **T1 위생 15항목(host-only) → T2 QCF 핸드셰이크 신호 유실(spec SEQ-098, S25) → T3 KV persistence Tier 1(TTFT 게이트) → T4 read-plan S1~S6(ADR-0011) → T5 설계+결정(D 질문) → T6 잔여 P2**
- 직전 종결 트랙: QCF_kv 설계 라운드 — verify 매트릭스 **30/30** + α-K frozen GREEN ([[project-qcf-kv-design-round]] / `handoff_qcf_kv_round_complete_2026_06_12.md`)

## 다음 액션 (T1 — 위생 일괄, host-only)

`sprint_backlog_burndown_2026_06_12.md` (b) T1 표의 15항목. 요지:
1. 즉답형(Implementer 일괄 위임 가능): 1-1 doc stale / 1-4 INV-LAYER baseline 재동결 / 1-5 test_backend 하네스 / 1-6 coverage ID 추출 / 1-13 argus_eval 이주
2. Architect 선행(spec 동반): 1-7 EnergyConstraint(디폴트=spec 갱신·코드 불변) / 1-9 QuantizeHandler stub+ENG-ALG-092
3. 거취 1줄(디폴트=status quo+재발동 트리거 명시): 1-10 §13.8-L / 1-11 KiviCache downcast / 1-14 LISWAP-6(범위 밖 재분류 후보)
4. 검증 스텝: 1-15 `cargo test -p llm_rs2 --lib backend::opencl` 1회 → 0 FAIL이면 L282 RESOLVED 처분
5. 문서: 1-2 arch §13.4/§13.6 / 1-3 Precision Swap 다이어그램 (Architect)
- 게이트: lib 1410+/0 + manager 0 FAIL + fmt + clippy + (spec 동반 시) spec 테스트/coverage 신규 갭 0 → **T1 종결 시 origin 푸시 + 보고**

## Landmines / 미해결

- **공통 게이트 정본**: host lib 0 FAIL / α-K frozen 3-dtype(절차 = `worktree_split_hygiene_2026_06_12.md` ②(b), baseline 보존본 `.agent/measurements/frozen_baseline_alpha_k_5f/`) / S25 verify **30/30**(`python verify/verify.py --device galaxy_s25 --model f16,q4`). frozen tbt는 **연속 부하 직후 측정 금지**(열누적 +3~4% 허상 — 쿨다운 ~3분).
- **S25 시리얼 명시**: 디바이스 2대 연결 가능성 — `adb -s R3CY408S5SB`.
- T4-S2 hot-path dispatch 전략은 ADR-0011 R1(RPN 378) — Architect amendment로 고정, TBT Δ>+3% 회귀 시에만 사용자 질문.
- T6 h-1 디폴트 = 별 sprint 분리(D-3) — burndown에서는 설계 라운드 예약 표기만.
- backlog 항목은 **헤더만 보고 살아있다 단정 금지**(Status 본문 확인 — layer aggregate 오기재 전례).
- manager merge는 non-empty heartbeat available_actions를 capability보다 우선 — 액션 추가 시 `compute_available_actions` 입력 전파 확인.

## 자기점검

- 진입 문장 한 줄로 첫 명령 가능? ✓ ("Backlog Burndown 진행" → T1 즉시 착수)
- 왜 멈췄나? ✓ (플랜 컨펌까지 완료, 실행은 다음 세션 — 사용자 지시)
- 최대 landmine 표면화? ✓ (재질문 금지 규칙 + 열누적 + 헤더 단정 금지)
- 게이트가 수치/명령? ✓ (트랙별 게이트는 플랜 SSOT에 명시)
