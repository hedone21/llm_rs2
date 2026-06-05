# HTP FFN gate/up batch fusion — device e2e 측정 (2026-06-05)

**디바이스**: Galaxy S25 (R3CY408S5SB) · **모델**: Qwen2.5-1.5B-Instruct (q4_0 / f16) · **백엔드**: `--backend htp` (FastRPC, `libggml-htp-v79.so`) · **스레드**: 6T (`taskset 3f`) · **env**: `ADSP_LIBRARY_PATH=/data/local/tmp`

**대상**: `matmul_ffn_gate_up_silu` override — FFN gate/up matmul 을 `enqueue×2 → drain×1` 로 묶어 layer 당 FastRPC dispatch floor 를 **2회→1회** 로 amortize (28 layers). A/B = 한 바이너리, `LLMRS_DISABLE_HTP_FFN_BATCH` 토글 (OFF=fallback 개별 dispatch, ON=batch).

---

## 1. 정확성 (16-tok greedy, token-id 16/16)

| 경로 | first | 생성 텍스트 | batch log |
|---|---|---|---|
| **q4** batch ON | 12095 | …Paris. It has a population of about 2 million people and covers an area | **1** |
| **q4** CPU | 12095 | (동일) | — |
| **q4** batch OFF | 12095 | (동일) | 0 |
| **f16** batch ON | 12095 | …Paris. The capital of Italy is Rome. What's the capital city of Spain | **1** |
| **f16** batch OFF | 12095 | (동일) | 0 |

- q4: batch ON == CPU == batch OFF **16/16 완전 일치** → batching 이 출력 보존.
- f16: batch ON == batch OFF **16/16 완전 일치**.
- `batch log` (= `gate/up batch dispatch 활성` eprintln 횟수): ON=1, OFF=0 → **batch 경로가 실제 실행됨**을 확증하고 env 토글이 정상 동작.

## 2. Decode TBT (64-tok, n=3 median, ms/tok)

| weight | OFF (before) | ON (after) | Δ | CPU(ref) |
|---|---|---|---|---|
| **q4_0** | 83.18 `[82.09, 83.18, 83.82]` | **78.88** `[78.85, 78.88, 81.30]` | **−4.30 (5.2%)** | 33.08 |
| **f16** | 113.33 `[113.13, 113.33, 113.51]` | **110.19** `[110.10, 110.19, 110.30]` | **−3.14 (2.8%)** | 64.26 |

- **after < before** (양쪽), **분포 비중첩** (q4: OFF최소 82.09 > ON최대 81.30; f16: OFF최소 113.13 > ON최대 110.30) → 노이즈가 아닌 실효 감소.
- 절감폭 ~3–4 ms/tok 가 dtype 무관하게 일정 = floor 가 **op-count 기반**(layer 당 1 dispatch 제거)이지 데이터 크기 기반이 아님을 재확인.

## 3. 서사 (★)

**Floor recovery 지 CPU 추월 아님.** HTP ON (q4 78.9 / f16 110.2) 은 여전히 CPU (q4 33.1 / f16 64.3) 보다 느리다. fusion 이득 = "heterogeneous 협력 시 NPU leg 의 FastRPC dispatch floor 누적 페널티를 layer 당 1회 제거" 지 "NPU 가 빨라짐" 이 아니다. 측정값은 op microbench 의 ~100µs floor 예측(28층 × 1 dispatch 제거 ≈ 2.8 ms)과 일치.

**KV-free 경계 준수**: gate/up 만 batch (attention/kv_scatter/eviction 은 CPU 유지). DSP↔KV 정책 결합 회피.

## 4. 재현

```bash
bash driver.sh          # 정확성 + q4 TBT (raw/ 저장)
# f16: driver 의 MODEL 을 Qwen2.5-1.5B-Instruct-f16.gguf 로 교체
python plot.py          # results.json → figure
```

산출물: `results.json`, `driver.sh`, `plot.py`, `htp_ffn_batch_fusion.{png,pdf}`, `raw/`.
