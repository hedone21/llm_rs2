# LISWAP-5 v1 (phase-aware async swap) — Production 측정 결과

**Date**: 2026-05-10
**Device**: Galaxy S25 (Adreno 830, 6T)
**Model**: Qwen2.5-1.5B (primary F16, secondary Q4_0)
**Backend**: opencl
**Commit**: `ab1f767` (B-2.5 wire-up 완료 후)

---

## 0. Verdict — **YELLOW (v1 design 불충분)**

Phase R 1.04× of max GREEN microbench가 production decode에 직접 옮기지지 않았다. Wall-clock은 single-shot보다 느리고 correctness가 mid-decode swap 때문에 분기. v1 (per-tensor chunk + forward-thread dispatch)은 보완 필요.

---

## 1. 3-way 비교 (32 token decode, "The capital of France is")

| 모드 | TTFT (ms) | Decode TBT (ms/tok) | End-to-end (ms) | Output 첫 줄 |
|---|---:|---:|---:|---|
| [A] No swap (F16 stays) | 371 | 53.29 | **2023** | "Paris. The French are known for their love of food..." |
| [B] Single-shot Q4_0 swap | 493 | 25.79 | **1293** | "Paris. The country has a population of about 67M..." |
| [C] LISWAP-5 phase-aware | 343 | 48.42 | **1844** | "Paris. The country has a population of 67M, the average..." |

### 분석
- **End-to-end**: B(1293) < C(1844) < A(2023). LISWAP-5는 single-shot보다 +551 ms (43%) 느림.
- **TTFT**: C(343) < A(371) < B(493). LISWAP-5가 swap stall을 prefill에서 제거 — TTFT만 보면 single-shot보다 -150 ms 빠름.
- **Decode TBT**: B(25.79) << C(48.42) ≈ A(53.29). LISWAP-5의 decode가 **거의 F16 forward 속도** — chunk dispatch 오버헤드 + mid-decode mixed F16/Q4_0 forward 때문.
- **Output divergence**: A vs B/C 모두 다름 (다른 weight). B vs C는 "67 million" 까지 같지만 그 후 분기 — swap이 점진적으로 진행되며 forward가 mixed weights로 돌기 때문.

---

## 2. 왜 v1이 실패했나

### 2.1 Per-tensor chunking은 cache-fit window보다 크다
| Tensor | Q4_0 size | mmap-to-host @ 3 GB/s | cache-fit window 980us 대비 |
|---|---:|---:|---:|
| wq, wo | ~9 MB | **3 ms** | **3.0×** overrun |
| wk, wv | ~1.5 MB | 0.5 ms | 0.5× (fits) |
| w_gate, w_up, w_down | ~26 MB | **8.7 ms** | **8.9×** overrun |
| norms | < 100 KB | < 50 us | fits |

ffn weight 3개는 단일 cache-fit window에 절대 안 맞음. 각 chunk dispatch가 forward thread를 ~3-9 ms 막음.

### 2.2 Forward-thread dispatch overhead
`try_dispatch_chunk` = `materialise_cpu_tensor(secondary mmap → cpu Tensor)` + `enqueue_write_async`. materialise는 mmap에서 host memcpy → 해당 tensor size만큼 CPU 시간. 본 v1은 이를 forward thread (op_trace::on_op_end 내부)에서 동기 호출.

→ Cache-fit window가 chunk materialise 시간보다 짧으면, dispatch 자체가 다음 op 시작을 막는다.

### 2.3 Mixed F16/Q4_0 forward
LISWAP-5는 swap이 decode 중 점진적으로 진행. 25 layers swap이 30 token에 걸쳐 분산되면, 첫 token은 F16, 마지막 token은 Q4_0, 중간은 mixed. **이것이 Decode TBT가 F16(53ms)에 가까운 48ms인 이유**.

→ Swap이 완료된 후의 forward만 측정하면 Q4_0(26ms) 근처일 것. 본 5초 decode 중에는 swap이 안 끝남.

### 2.4 Correctness divergence
B vs C 다른 출력 — swap 진행 시점에 따라 forward가 다른 weight로 돌기 때문. 이건 algorithmic divergence가 아니라 byte-equal 비교 자체가 부적절한 상황.

---

## 3. v2 design 옵션

### Option α — Sub-tensor chunking (Phase R 그대로)
- ffn weight를 4 MB 단위 sub-chunk으로 분할
- `enqueue_write_buffer(offset, ...)` 사용 (OpenCL spec 보장)
- 각 sub-chunk = 570 us @ 7 GB/s → cache-fit window 안에 fit
- 단점: chunk granularity 7-9× 증가 (총 ~225 chunk → ~1500 chunk), per-chunk overhead 누적

### Option β — Background thread + chunk pre-staging
- worker thread가 mmap → CPU tensor materialise를 미리 준비
- forward thread의 on_op_end는 enqueue_write_async만 (host overhead < 100 us)
- staging buffer 큐에 N chunks pre-staged
- 단점: extra worker thread, staging memory 4-8 MB

### Option γ — Single-shot + delay measurement (가장 빠른 검증)
- swap을 prefill 직후 (decode 0번째 token 직전) single-shot으로
- 가장 안정적, hide 안 함
- 본 단계에서 single-shot이 결국 wall-clock 최적임을 인정 → LISWAP-5 abandon

### Option δ — Hybrid: pre-stage + chunk
- prefill 동안 worker thread가 모든 tensor materialise 완료 (decode 시작 전)
- decode 동안엔 forward thread가 enqueue_write_buffer만 (chunk-level)
- materialise overhead가 prefill과 overlap → forward에 영향 없음
- 단점: prefill 길이 ≥ materialise 시간 (~700 ms for 25 layers × 100 MB) 필요. prefill이 더 짧으면 decode 일부 영향.

---

## 4. 권장

**Option β (background thread pre-staging) 우선 시도** + sub-tensor chunking을 ffn weights에만 한정 적용.

이유:
- Option γ (LISWAP-5 abandon)는 너무 일찍 포기. Phase R GREEN evidence는 pre-stage가 충분히 hide 가능함을 시사.
- Option α 만으론 forward-thread overhead 문제 그대로.
- Option δ는 prefill ≥ materialise 가정 필요해 generalize 안 됨.
- Option β + α 부분 적용 = 두 문제 모두 해결.

추정: 추가 2-3 dev-day. 신규 worker thread + staging queue + ffn sub-chunking 구현 + 재측정.

또는: **단순화로 LISWAP-1 (incremental per-tick) 와 비교 측정**. LISWAP-1도 frame budget hide 목적이지만 wall-clock +33%. LISWAP-5 v1은 +43%이므로 약간 나쁨. 어느 쪽도 single-shot 대비 wall-clock 손해가 있음.

---

## 5. v1 측정 데이터 raw

```
Single-shot (B):
  weight_swap: force ratio=0.90, swapped 25/28 layers in 279.3ms
  stages: prefault=17.5ms mmap_permute=261.2ms ...
  TTFT: 493.54 ms
  Decode: 25.79 ms/tok (38.8 tok/s)

LISWAP-5 (C):
  weight_swap: phase-aware mode — ratio=0.90, 25 target layers, chunk_size=4 MB
  TTFT: 343.93 ms
  Decode: 48.42 ms/tok (20.7 tok/s)
  (PhaseAwareSwap finalize log 미캡쳐 — disp.is_complete() polling이 decode end 전 trigger 안 했을 가능성)
```

`disp.is_complete()` polling 결과가 finalize log에 안 잡힘 — 이는 decode 32 token 안에 swap이 안 끝났을 가능성. 즉 chunk dispatch가 cache-fit window 단위로 throttle되어 더 긴 decode 필요. Long decode (n=200+) 측정 필요.

---

**End of v1 measurement report**

다음: v2 (β + α partial) 구현 또는 LISWAP-5 abandon 결정 — 사용자 confirm 필요.
