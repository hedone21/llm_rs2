# QCF_kv 분포 실측 — floor 재산정 근거 (S25, 2026-06-12)

**목적**: QCF_kv 수식 정규화(`753fb257`, ENG-ALG-051 개정) 후 raw QCF 분포를 실측하여
`policy_default.lua` QCF_FLOOR 4임계 + V_Q를 재산정 (사용자 결정 2 = A, 2026-06-12).
**장비**: Galaxy S25 (R3CY408S5SB), `argus_bench -b opencl --opencl-rpcmem`, 6T, KV f16.
**HEAD**: `d6fcf376` (P2 구현 3커밋 직후) + mock_manager 시나리오 값 출력 패치.

## 측정 절차

mock_manager TCP 시나리오 모드로 decode 구간에 RequestQcf 6회 프로브
(`qcf_probe_{long,short}.json` — long: 6500ms + 1200ms×5, short: 1500ms + 1000ms×5).
엔진 플래그 = verify `signal_memory_critical` baseline 정합:
`--greedy -n 128 --threads 6 --max-seq-len 1536 --kv-type f16 --protected-prefix 4
--min-kv-cache 8 eviction h2o` + `--resilience-transport tcp:127.0.0.1:9899`.

dry-run 파라미터 (qcf_runtime 고정): keep_ratio=0.5 (캐시 절반 eviction 시뮬레이션),
protected_prefix=4. h2o/d2o 키는 score accumulator 활성(`eviction h2o`) 시 산출.

| 구성 | 프롬프트 | 프로브 수 | raw |
|---|---|---|---|
| f16 long | medium_qa (918 tok) | 6/6 | `mm_qcf_f16_long.txt` |
| q4 long | medium_qa | 6/6 | `mm_qcf_q4_long.txt` |
| f16 short | short_smoke (~28 tok, verify 게이트 레짐) | 6/6 | `mm_qcf_f16_short.txt` |
| q4 short | short_smoke | 4/6 (decode 조기 종료) | `mm_qcf_q4_short.txt` |

## 분포 (n=66, 포화 0건 — 수정 전 0.985 고정과 대비)

| action | n | min | p50 | p90 | max | 레짐 구조 |
|---|---|---|---|---|---|---|
| kv.evict_h2o | 22 | 0.0308 | 0.0376 | 0.1427 | 0.1677 | 이봉: long ~0.033 / short ~0.14 |
| kv.merge_d2o | 22 | 0.0861 | 0.1635 | 0.2188 | 0.2433 | 단봉 |
| kv.evict_sliding | 22 | 0.2610 | 0.3029 | 0.3451 | 0.3893 | 단봉, short가 +0.03 높음 |
| pooled | 66 | 0.0308 | 0.1648 | 0.3210 | 0.3893 | |

순서 구조 h2o < d2o < sliding — heavy-hitter 보존 < merge 보상 < 무차별 절반삭제.
액션·레짐·dtype별 변별이 살아 있음 (수식 수정의 의미 복원 실증).

**weight.skip / layer_swap**: live 경로 미산출 확인 (dispatcher `QcfEstimateContext.importance_table=None`
→ estimates 3키만). QCF_weight 가족은 qcf_cache miss → cost 0 (INV-117) → floor 투명.
단일 가족(QCF_kv) 기준 산정이 현 시점 정확. weight.skip 배선 시 cross-family 재검토
(가족별 floor 분리는 별도 backlog 후보).

## 산정 결과 (`policy_default.lua` v2.5.0)

| 레벨 | 구 (포화 스케일) | 신 | 근거 |
|---|---|---|---|
| normal | 0.30 | **0.10** | 준무손실만 — h2o long(0.031~0.039) 통과, 그 외 차단 |
| warning | 0.60 | **0.25** | 보상형 허용 — h2o 22/22 + d2o 21/22, blind sliding(≥0.261) 차단 |
| critical | 0.90 | **0.50** | 전 표준 eviction 통과 (max 0.389 + 28% 여유) — sliding 단독 키 구성에서도 발행 보장 |
| emergency | huge | huge | 불변 (lossy 반드시 허용) |

**V_Q = 0.5 유지** (재산정 결론): penalty 대역 0.5×[0.03,0.39]=[0.015,0.195] ≈ resource
term(Z_mem×relief_mem≈0.09~0.16) 동급. 가족 내 변별 Δ(h2o↔sliding)=0.5×(0.303−0.038)≈0.13
> relief Δ(0.4 vs 0.3)×Z≈0.03 → QCF가 가족 내 액션 선택을 실질 주도. 구 스케일에서는
0.5×0.985=0.49 균일 오프셋(변별 0)이었음.

## 주의 (후속 세션용)

- 분포는 **dry-run keep_ratio=0.5 고정** 기준 — keep_ratio가 동적이 되면 재실측 필요.
- QCF>1 가능성: 비상관(직교) V에서는 정규화 후에도 1 초과 가능 (spec "near-convex" 전제).
  실 LLM 분포에서는 ≤0.39 — emergency=huge가 이상치 방호.
- 프로브 도구: mock_manager 시나리오 모드 값 출력 (이 라운드에서 추가) + `/tmp/qcf_probe_*.json`
  (repo 외 — 본 문서 §측정 절차가 재현 정본).
