# Handoff: KV 로드맵 항목 0 측정 스프린트 종결 → 항목 3+4 (인프라) 착수

**작성**: 2026-06-12
**HEAD**: `c702ff83` + 본 종결 docs 커밋 (P2 구현 4커밋 `b5250bda`/`c9e1f961`/`c41865e9`/`bdf744a8` + e2e 수정 7커밋 `449cf55f`/`6987d8ef`/`2ccb5825`/`4baea4e0`/`dc79755f`/`c702ff83`/`1182fa0c`)
**브랜치**: master (origin 미푸시 — QCF 세션분 포함 누적, 푸시는 사용자 지시 대기)
**다음 세션 진입 문장**: **"KV 로드맵 항목 3+4 진행 — QueryStats TensorKind 구현 → read-plan ADR"** (대안: "항목 6 다층 해상도 보강 측정" / "QCF_kv 설계 라운드 공동 검토")

---

## TL;DR

로드맵 항목 0(확장 0개로 가능한 1B 실측 4종 — 사용자 결정 D1~D4 반영)을 완주했다. 1차 측정에서 P2 구현 전부가 **단위 게이트는 통과했으나 e2e 측정 경로 미배선**임이 드러나 수정 라운드(7커밋) 후 재측정으로 게이트 판정을 확정: **R-KV 보류 / A2SF 보류 / head 분산 개봉 후보(63.7배) / Demote RED**. 멈춘 이유 = 측정 스프린트 종결 — 다음은 D4 확정 경로인 인프라 항목 3(QueryStats)+4(read-plan ADR).

## 진행 상태 (게이트 판정 — `experiments/kv_roadmap_item0/rerun/REPORT.md` 정본)

| 측정 | 핵심 수치 (5도메인) | 판정 |
|---|---|---|
| R-KV | redundant fraction 0.9964 포화(τ=0.5 변별력 없음, MPC 0.49) · R-KV≈h2o 퇴화 · vs sliding PPL 0.9982 동률 | **보류** |
| A2SF | BOS ratio 63.1% 감소(기계적 동작 확인) · 그러나 vs sliding 2/5 승 = 품질 개선 0 | **보류** |
| head 분산 | max/min C_h 63.7배 (2nd-min 기준 ~9.8배 ≥ 5배 게이트) — sparse/dispersed head 공존 | **개봉 후보** (항목 6 트리거 부분 충족) |
| Demote | 실모델 PPL 5/5 도메인 RED (ratio 1.22~6.04×) | **RED** (항목 1 보류) |

설계서 = `arch/kv_roadmap_item0_measurement.md` (임계 고정값 포함). 스프린트 상세 = `.agent/todos/sprint_kv_roadmap_item0_2026_06_12.md`. backlog 트랙(L865~) 항목 0/1/2/3/4/6 처분 갱신 완료.

## 다음 작업 (우선순위 — PM 권고)

1. **항목 3 (QueryStats TensorKind) → 항목 4 (read-plan ADR)** — D4 확정 경로(1B 게이트 결과와 무관). 항목 3이 4의 신호 공급원이라 이 순서. 검증: TensorKind variant + 누적 배선 + host 통계 정확성 테스트 / ADR grill 통과.
2. 항목 6 다층 해상도 보강 측정 (소규모) — 현 63.7배는 최종 layer 1개 해상도. 단독으로는 항목 6 개봉 불가(실수요 트리거 잔여).
3. QCF_kv 설계 라운드 — 사용자 공동 검토 대기 (이번 측정의 qcf_sum↔PPL 역전이 보강 증거로 추가됨).

## Landmines / 미해결

- **단위 게이트 ≠ e2e 측정 게이트**: P2 구현 4종 전부가 단위 테스트 GREEN인데 e2e 측정 불가였다 (R-KV는 eval_setup legacy match 미배선, head 분산은 `workspace: None`으로 score 미공급, A2SF는 지표 덤프 부재, Demote는 PPL 경로 부재). **측정용 구현의 완료 게이트에는 "실모델 e2e 1회"를 반드시 포함할 것.**
- **NMSE는 demote를 오도한다**: NMSE(1차 왕복 손실)는 demote 우세, 실추론 PPL은 5/5 RED — K 손실의 attention 누적 전파(2차 효과)가 NMSE에 안 잡힌다. **품질 판정의 정본은 실추론 PPL.**
- **PPL 모드에서 prefill 전체 고정은 eviction과 양립 불가** (decode=0 → eviction 미발동). `--ppl-prefill-tokens`로 고정값(이번엔 150) 통일이 정답.
- **RkvStats의 layer 필드는 plan() 호출 순번** — `% 16` 역산 필요 (`KVStageCtx.layer_idx()` 0 고정 제약).
- **rkv는 기본 off feature** — 측정 시 `--features rkv` + `ARGUS_RKV_DUMP=1`. production 표면 불변.
- **head 분산 63.7배는 단일 layer 증거** — 항목 6 개봉 논거로 쓰기 전에 다층 확인 필수.
- 측정 인프라 신규 플래그: `--score-decay`(A2SF α=1−decay), `--dump-a2sf <path>`, `--dump-importance`(C_h 확장), demote 실모델 테스트는 `demote_measure.rs` env 게이트.

## 자기점검

- 진입 문장 한 줄로 첫 명령 가능? ✓ ("항목 3+4 진행")
- 왜 멈췄나? ✓ (측정 스프린트 종결, 다음 경로는 D4로 기확정)
- 최대 landmine 표면화? ✓ (단위≠e2e 게이트 교훈 + NMSE 오도)
- 게이트가 수치/명령? ✓ (4종 판정 표 + REPORT.md 경로)
- 길이 적정? ✓ (상세는 REPORT.md/설계서/스프린트 파일로 위임)
