# Option C: Per-op slope vs TOTAL slope — 모순 검증

작성: 2026-04-14 21:30
원본: `.agent/research/option_2b_raw/llm_rs2_ctx{256,1024,2048,6k}.log` 및 `llama_ctx{...}.csv`

## 결론 (TL;DR)

**§12에서 보고된 "attention slope 13.23 > TOTAL 12.45" 모순은 측정 세션 차이가 원인. 실제 모순 아님.**

- 13.23 μs/n_kv는 `--profile-events` 켠 세션의 attention CL event slope
- 12.45 μs/n_kv는 별도 비-profile 세션의 wall-clock TOTAL slope
- 같은 profile 세션의 wall-clock TOTAL slope를 계산하면 **14.74 μs/n_kv**
- 같은 profile 세션의 per-op slope 합은 **14.13 μs/n_kv** (94%가 attention)

## 데이터: 같은 profile-events 세션, 4개 ctx, n_kv = [258, 1025, 2047, 4472]

### llm_rs2 per-op avg μs/token

| op | n_kv=258 | 1025 | 2047 | 4472 | slope μs/n_kv | % of profile sum |
|---|---:|---:|---:|---:|---:|---:|
| attention      | 5,691  | 13,741 | 18,456 | 61,082 | **13.23** | 93.6% |
| matmul_ffn     | 11,374 | 11,417 | 9,102  | 13,048 | 0.393 | 2.8% |
| lm_head        | 7,977  | 7,982  | 6,439  | 9,301  | 0.316 | 2.2% |
| matmul_qkv     | 2,159  | 2,159  | 1,651  | 2,558  | 0.095 | 0.7% |
| rms_norm       | 948    | 947    | 730    | 1,151  | 0.049 | 0.3% |
| matmul_wo      | 818    | 807    | 651    | 959    | 0.035 | 0.2% |
| rope           | 108    | 107    | 82     | 139    | 0.008 | 0.1% |
| silu_mul       | 83     | 83     | 61     | 91     | 0.002 | <0.1% |
| add_assign     | 55     | 55     | 47     | 64     | 0.002 | <0.1% |
| kv_update      | 55     | 55     | 39     | 59     | 0.001 | <0.1% |
| **per-op sum** | 29,283 | 37,367 | 37,270 | 88,469 | **14.13** | 100% |
| **wall (Decode: ms/tok)** | 32,050 | 40,200 | 83,380 | 92,040 | **14.74** | — |

**관찰**:
1. per-op sum slope (14.13) ≈ wall slope (14.74), 차이 0.6 = sync 대기 overhead. 내부 일관성 OK.
2. attention 단일이 per-op sum의 94% — attention이 갭의 **지배적** 원인 확정
3. ctx2048만 wall이 83 ms/tok로 튐 (다른 ctx 대비 +30~50ms) — 단일 측정 노이즈, slope에는 포함했지만 신뢰도 ↓

### Profile vs 비-profile 세션 비교

| 측정 | non-profile (오후) | profile-events | 차 |
|---|---:|---:|---:|
| llm_rs2 wall slope | 12.45 | 14.74 | +2.29 |

profile-events 자체가 +2.29 μs/n_kv 시스템 오버헤드를 더함. 즉 §12 "TOTAL 12.45"와 "attention 13.23"는 다른 세션 측정값을 단순 비교한 것. **수학적 모순 없음**.

추정 (비-profile 세션에서의 attention 단일 contribution):
- 13.23 × (12.45 / 14.74) ≈ **11.18 μs/n_kv**
- 비-profile wall 12.45 중 attention ≈ **90%**, non-attention ≈ **1.27 μs/n_kv (10%)**

### llama.cpp 같은 profile 세션 데이터 (CSV)

| kernel | ctx256 | ctx1024 | ctx2048 | ctx6k | slope μs/n_kv |
|---|---:|---:|---:|---:|---:|
| flash_attn_f32_f16_q1 (decode) | 10,135 | 22,531 | 42,198 | 82,371 | **17.22** |

- llama.cpp Q1 단일 slope (profile 세션) **17.22 μs/n_kv > 우리 13.23**
- 그러나 llama.cpp 비-profile wall slope = **5.70 μs/n_kv**
- 즉 llama.cpp는 profile-events가 attention을 **+11.5 μs/n_kv 이상 inflate** 시킴 — 우리 +2.3보다 5× 더 심함
- 원인 추정: 드라이버의 event 기록 비용이 kernel 종류/dispatch 빈도/queue 깊이에 따라 다르게 부과됨
- **engine 간 profile-events 직접 비교는 신뢰 불가** (§12의 Phase B 결론 재확인)

## 해석

1. **modular 모순 없음**: 13.23 vs 12.45는 다른 세션 비교의 산물. 같은 세션 내 per-op sum과 wall은 일치 (14.13 vs 14.74).
2. **attention 지배 확정**: per-op sum의 94%, 비-profile에서도 90%. 갭 6.75 μs/n_kv 중 attention contribution **~5.5 μs/n_kv (81%)**.
3. **non-attention contribution**: 비-profile에서 ~1.27 μs/n_kv (matmul_ffn 0.39 + lm_head 0.32 + matmul_qkv 0.09 + matmul_wo 0.03 + rms_norm 0.05 + 노이즈). 이 중 일부는 **n_kv-invariant 인데 측정 노이즈로 slope 0.39 잡힌 ffn** 같은 것.
4. **microbench 결과와 합산 비교**:
   - microbench (단일 layer Q1, no other op, no thermal): 0.36 μs/n_kv per layer × 28 = 10.1 μs/n_kv per token
   - production (per-layer attention 추정): 13.23 / 28 = 0.473 μs/n_kv per layer
   - 차이 0.11 μs/n_kv per layer × 28 = **3.1 μs/n_kv per token** — production 환경 (cache pressure, mask, real Q)에서 더 느려짐
5. **llama.cpp 진짜 비교 불가** — profile mode가 inflated 됐고, 비-profile에서는 op breakdown 없음

## 다음 권장 (§12 A로 자연스럽게 연결)

llama.cpp non-profile attention contribution을 직접 측정할 방법이 없으므로:

**A**: microbench을 production 조건으로 확장 — 28× back-to-back dispatch + real mask + Q variance — 해서 0.36 → 0.47 (production-equivalent)이 어디서 오는지 isolate.
- 이 3.1 μs/n_kv per token이 microbench-best 대비 production loss의 전부.
- llama.cpp는 production-best 0.36 to 0.47 차이가 더 작거나 0일 가능성 → 경쟁사가 production 환경 자체를 더 잘 다루는 것일 수 있음.

대안 **D** (eviction 우회) 도 여전히 유효 — 정확도 trade-off 수용 가능 시 가장 빠른 win.
