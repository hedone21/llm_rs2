# Handoff: QCF_kv 설계 라운드 진입 (사용자 확정 — 차기 세션)

**작성**: 2026-06-12
**HEAD**: `7daa7e69 fix(test): host lib 테스트 위생 (b)(c) …` (origin/master 동기화 — 본 handoff 커밋 포함 푸시됨)
**브랜치**: master
**다음 세션 진입 문장**: **"QCF_kv 설계 라운드 진행"** — 첫 액션 = 사용자 결정 2건 수령 후 라운드 실행

---

## TL;DR

KV 로드맵 항목 3+4 종결("KV 구조 확정" 게이트, `handoff_kv_roadmap_item34_entry_2026_06_12.md`) + 분리 위생 3종 + host lib 테스트 위생 전체(P2-chore (a)(b)(c)) 완료 후, **사용자가 차기 = QCF_kv 설계 라운드로 확정**. 멈춘 이유 = 세션 종료(라운드 실행은 결정 2건 수령부터). 안건 SSOT = `backlog.md`의 **"[P2] QCF_kv 정규화 비대칭 + estimator 우회 + manager floor 재설정 — 설계 라운드"** 항목 (3층 결함 분석 + 2026-06-12 검토 라운드 추가 발견 4건 전부 수록).

## 첫 액션 — 사용자 결정 2건 (라운드 실행의 선행 조건)

1. **estimator 우회 해소 방향**: (A) 엔진측 ΔPPL 환산 — estimator 장착, spec 무변경, 항등 fallback이라 전환 무위험 (**권장**) vs (B) raw 직송 합법화 — spec ENG-ALG-050 step 4 개정. floor의 단위가 이 결정을 따름.
2. **QCF_FLOOR 재설정**: (A) 이번 라운드 포함 — 수식 수정 후 S25 분포 실측(양 가족) → 4임계+V_Q 재산정 → policy_default.lua 반영 (**권장**) vs (B) 현행 유지 — dead config 잔존.

## 라운드 범위 (backlog 항목 AC — 결정 후 실행)

① 수식 수정(o_before에 α/Σ_all α 정규화) + spec ENG-ALG-051 + docs/qcf_taxonomy §2.1/§2.2.1 **3소스 동기화** ② estimator 방향 구현 ③ floor 재설정 ④ `test_d2o_less_than_h2o` 재보정(실분포 근사 데이터로 교체) ⑤ **`signal_memory_critical` f16/q4 S25 verify GREEN** = 회귀 게이트 → 완료 시 verify 매트릭스 **30/30** 봉인.

## Landmines / 미해결

- **곡선 키 불일치 잠복**: `estimator.rs::with_defaults` 키(`kv.eviction`/`kv.sliding`/…) ≠ IPC estimates 키(`kv.evict_h2o`/`kv.evict_sliding`/…) — 항등 fallback이라 현재 무해하나 캘리브레이션 등록 시 silent 미적용. **방향 결정과 무관하게 키 전수 감사를 라운드에 포함** (manager `context.rs` `kv_streaming` 언더스코어 표기 포함).
- **워크트리 분기와의 관계**: QCF_kv 영역(`qcf_kv.rs`/estimator/policy lua)은 KV 표면 밖 — **메인 진행 가능 명시 예외**(`worktree_split_hygiene_2026_06_12.md` R3). 단 `attention_scores.rs`/`qcf_runtime.rs` 접촉 시 기존 QueryStats 배선(항목 3, 신규 코드 격리됨)과 commit 격리 유지.
- **포화 메커니즘 실측**: ‖o_before‖=201 vs ‖o_after‖=2.9 (S25, pos=1045, Σα≈12) → QCF≈0.985 고정. 수정 후 기대 0.08~0.22 (S25 실측 있음). 항목 0 스프린트의 qcf_sum↔PPL 역전이 독립 보강 증거.
- **v1 PASS는 우연**: 4월 signal_memory PASS는 fixture relief 키 깨짐(silent 0) 의존 — "예전엔 됐는데"로 회귀 추적하지 말 것.
- 상세 컨텍스트 메모리: [[project-score-wiring-qcf-round]] (QCF 라운드 SSOT), [[project-kv-roadmap-item34]] (분기 머지 규칙).

## 오늘 세션 누적 (2026-06-12, 전부 origin 푸시)

| 작업 | 커밋 | 게이트 |
|---|---|---|
| 항목 3 QueryStats 구현 | `783bcadd`+`a98cd679` | e2e non_empty + token-id byte-identical + S25 α-K frozen 3-dtype Δ 음수 |
| 항목 4 ADR-0011 (설계만) | `70729062` | grill 통과 + 구현 코드 0줄 |
| 분리 위생 3종 | `dd56a647` | 머지 게이트 명문화 + frozen 재앵커 + origin 푸시 |
| 테스트 위생 (a) | `b9775f7d` | 결정적 실패 2건 해소 |
| 테스트 위생 (b)(c) | `7daa7e69` | **lib 1410/1410 PASS 필터 없음** + coverage 스크립트 완주 |

## 자기점검

- 진입 문장 한 줄로 첫 명령 가능? ✓ ("QCF_kv 설계 라운드 진행")
- 왜 멈췄나? ✓ (세션 종료 — 결정 2건이 라운드 실행의 선행 조건)
- 최대 landmine 표면화? ✓ (키 불일치 잠복 + v1 PASS 우연성)
- 게이트가 수치/명령? ✓ (signal_memory_critical S25 GREEN → verify 30/30)
- 길이 적정? ✓ (상세는 backlog 항목 + 메모리 2건으로 위임)
