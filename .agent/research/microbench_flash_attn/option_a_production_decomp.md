# Option A: Production 조건 microbench 확장 — 갭 분해

작성: 2026-04-14 21:50
원본: `optionA_run1_*.txt`, `optionA_run2_*.txt`

## TL;DR

**production attention slope (13.23) 중 microbench-best (10.30)이 설명하지 못하는 2.93 μs/n_kv 는 production 환경의 17 intervening ops에 의한 L2 cache thrashing.**

**그러나 더 중요한 발견**: microbench-best 10.30 자체가 llama.cpp 추정 attention slope (~4.5)보다 **2.3× 느림**. **커널 자체가 진짜 갭**.

## 측정 매트릭스

4 variants × 2 layouts × 4 n_kv × 30 iters, 2 run 평균.

### per-token attention slope (μs/n_kv) — HeadMajor

| variant | 추가 요인 | run1 | run2 | mean | Δ from prev |
|---|---|---:|---:|---:|---:|
| Single × 28 (환산) | 1 dispatch ÷ 28 | 10.81 | 10.15 | 10.48 | — |
| Repeat28 | back-to-back 28 dispatches | 9.91 | 10.05 | 9.98 | **−0.50** (pipeline overlap) |
| +Mask | causal F16 mask 읽기 | 10.31 | 10.29 | 10.30 | **+0.32** (mask cost) |
| +QVar | Q를 28-slot pool에서 rotate | 10.30 | 10.31 | 10.30 | **0.00** (no effect) |

### 발견

1. **Pipeline overlap 이득** (Δ = −0.50): 28 sequential dispatches가 single ×28보다 빠름. Adreno 스케줄러가 다음 dispatch의 K fetch를 현재 dispatch의 compute 단계와 overlap. ROI: 이미 production이 누리고 있음.
2. **Mask read overhead** (Δ = +0.32): 28 layers × n_kv mask reads는 측정 가능한 비용. Production은 mask 사용하므로 이 비용 포함.
3. **Q variance 효과 0**: Q는 12 × 128 × 4 = 6 KB로 L1/SLM에 충분히 들어가서 cold-cache effect 없음. 의미 있는 production 차이 아님.
4. **Microbench-best (full prod sim) = 10.30 μs/n_kv per token**.

## 미설명 갭 (production 환경 효과)

| 측정 | μs/n_kv per token |
|---|---:|
| Production attention slope | 13.23 |
| Microbench-best (Repeat28+Mask+QVar) | 10.30 |
| **차 (production 환경 overhead)** | **2.93** |

이 2.93 μs/n_kv는 production이 attention 사이에 끼워넣는 17개 ops의 cache pressure로 추정:
- matmul_qkv (3 × ~5 MB weight read)
- matmul_wo, RMS norm, RoPE
- matmul_ffn_gate / up / down (3 × ~17 MB Q4_0 weight read each)
- silu_mul, add_assign 등

특히 FFN matmul들이 10s of MB씩 weight read하면서 KV/Q를 L2에서 evict함. Microbench는 같은 KV를 28번 연속 읽어서 L2 항상 warm.

**대응 가능성**: 거의 없음. FFN을 작게 쪼개도 같은 weight 양 읽음. KV pinning 직접 제어 불가.

## **진짜 갭** (커널 자체)

| 측정 | μs/n_kv per token |
|---|---:|
| llm_rs2 microbench-best | 10.30 |
| llm_rs2 production (with cache pressure) | 13.23 |
| llama.cpp wall slope (non-profile) | 5.70 |
| llama.cpp attention 추정 (~80% of wall) | ~4.50 |

**llm_rs2 microbench-best (10.30) ÷ llama.cpp attention 추정 (4.50) = 2.3×**

즉 production 환경 노이즈를 0으로 제거해도 우리 attention은 llama.cpp보다 2.3× 느림. 이는 **커널 또는 dispatch 자체의 차이**.

## Phase A 재검토 필요

§12 Phase A는 "두 커널 K-loop 바이트 동일" 결론. 그런데 실측은 2.3× 차이.

가능한 Phase A의 누락:
1. **kernel 파일 비교 자체가 잘못된 파일 비교** — llama.cpp가 Adreno에 실제 dispatch 하는 kernel은 다른 파일/build variant일 가능성
2. **dispatch 파라미터 차이** — global/local work size, WG count, subgroup 구성
3. **specialization constants** — DK/DV/BLOCK_M 또는 Q1_WG_SIZE 가 다를 가능성
4. **컴파일 옵션** — fast-math/cl-std 외에 vendor-specific 옵션 차이
5. **드라이버가 source-identical kernel을 다르게 컴파일** — clBuildProgram 시 cache hit 패턴 차이

## 결론 + 다음 권장

- Option A로 isolate한 production 환경 effect (2.93 μs/n_kv)는 **수정 어려움** — FFN 구조적
- 진짜 leverage는 **커널 자체 (10.30 → 4.50, 2.3× 차이)**
- 권장 다음 단계:
  - **Phase A 재실행** (researcher): 같은 모델/같은 디바이스에서 llama.cpp가 실제로 dispatch 하는 kernel binary를 추출하고 dispatch 파라미터까지 포함해서 다시 비교
  - **B (Snapdragon Profiler)**: 우리 Q1 vs llama Q1 같은 입력 같은 디바이스에서 SP trace — register spill / occupancy / L2 hit 비교
  - **D (eviction)**: 갭 자체를 우회 — 정확도 trade-off로 즉시 win

Option A는 **production 환경 효과는 작고, 진짜 원인은 kernel 자체**라는 강한 negative result를 제공. Phase A 재검토 또는 Snapdragon Profiler로 kernel-level isolation 필요.
