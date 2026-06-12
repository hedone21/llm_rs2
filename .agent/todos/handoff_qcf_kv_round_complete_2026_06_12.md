# Handoff: QCF_kv 설계 라운드 완주 — verify 매트릭스 30/30 봉인

**작성**: 2026-06-12
**HEAD**: `7397512c fix(verify): signal_memory_critical 재정렬` + 종결 docs 커밋 (브랜치 master, origin 미푸시 — 푸시는 사용자 확인 대기)
**다음 세션 진입 문장 (후보 2, 사용자 택1)**: **"QCF 핸드셰이크 신호 유실 진행"** (backlog 신규 P2, Architect 선행) 또는 **"QCF_kv layer aggregate 진행"** (backlog P1 — layer-0 단일 proxy → 전 layer, R3 예외라 메인 가능)

---

## TL;DR

QCF_kv 설계 라운드 완주 — 당초 3층 진단이 실측 중 **5층 결함 체인**으로 확장됐고 전부 해소. 결정 2건(estimator=**B raw 직송**, floor=**A 재설정**) 반영 + verify 매트릭스 **30/30 PASS 봉인**(구 28/30+known-fail 2) + α-K frozen 3-dtype byte-identical. 멈춘 이유 = 라운드 AC 전부 충족(종결). 상세 = 메모리 `project_qcf_kv_design_round.md` + `sprint_qcf_kv_design_round_2026_06_12.md`.

## 진행 상태 (커밋 9 + 게이트)

| 작업 | 커밋 | 게이트 |
|---|---|---|
| P1 spec 3소스 동기화 (ENG-ALG-050 B안/051 수식/060 위상) | `eab908ab` | 키 전수 감사 표 포함 |
| P2 수식 정규화 + d2o 재보정 + spec 테스트 3종 | `753fb257` | 0.985 포화→0.03~0.39, lib 1410/0 |
| P2 streaming 키 silent-0 수정 | `ddc2c2fb` | manager 0 FAIL |
| P2 estimator 곡선 키 IPC 정렬 | `d6fcf376` | 동작 무변(live 미장착) |
| P4 mock_manager 값 출력 (측정 도구) | `5454e564` | — |
| P4 QCF_FLOOR 재산정 (policy v2.5.0) | `1a8ee375` | S25 66샘플 분위수 근거 |
| P5 heartbeat available_actions 결함 수정 | `09a82ad9` | 호스트 재현 테스트 + manager 224/0 |
| P5 시나리오 물리 불가 재정렬 | `7397512c` | signal_memory_critical f16/q4 PASS |
| P6 종결 docs (backlog/sprint/AGENTS.md/handoff) | (본 커밋) | — |

- **verify 매트릭스**: **30/30 PASS** (`verify/results/20260612_165122_galaxy_s25_f16_q4/`)
- **α-K frozen**: 3-dtype 결정론 라인 byte-identical + tbt median f16 54.34(Δ+0.2%)/f32 54.53(Δ+0.9%)/q4 53.45(Δ−0.6%)

## 다음 액션 후보

1. **QCF 핸드셰이크 신호 유실** (backlog 신규 P2): timeout 시 무-QCF decide 폴백 + late estimate 캐시 반영 → spec SEQ-098 개정 동반(Architect 선행) → prefill-중-주입 변형 시나리오 S25 검증.
2. **QCF_kv layer-0 → 전 layer aggregate** (backlog P1): 측정 대표성 — QCF 영역이라 워크트리 분기 예외(R3) 메인 진행 가능.
3. origin 푸시 (워크트리 분기 공통 베이스 앵커 — hygiene 문서 ①) — 사용자 확인 후.

## Landmines / 미해결

- **manager merge는 non-empty heartbeat available_actions를 capability보다 우선** — heartbeat 산출 입력(eviction_policy 등)이 빠지면 capability를 동적화해도 silent 강등. 새 액션 추가 시 `compute_available_actions` 입력 전파 확인 필수.
- **tbt 게이트는 연속 부하 직후 측정 금지** — 매트릭스 직후 +3~4% 열누적 허상(쿨다운 ~3분 후 정상 복귀 실측).
- floor 분포는 **dry-run keep_ratio=0.5 고정** 기준 — keep_ratio 동적화 시 재실측.
- **QCF_weight(weight.skip)는 live 미산출**(dispatcher `importance_table=None`) → qcf_cache miss=0으로 floor 투명. weight.skip QCF 배선 시 cross-family floor 재검토.
- 디버그 기법: 정책 의사결정 불일치는 policy lua에 임시 `print()`(Z/avail/candidates/score) 주입이 결정타 — manager.stdout 아티팩트로 수집.

## 자기점검

- 진입 문장 한 줄로 첫 명령 가능? ✓ (후보 2 — 둘 다 backlog 항목으로 추적 가능)
- 왜 멈췄나? ✓ (라운드 AC 전부 충족 = 종결)
- 최대 landmine 표면화? ✓ (heartbeat merge 우선순위 + 열누적)
- 게이트가 수치/명령? ✓ (30/30 결과 경로 + frozen Δ 수치)
