# cuda_embedded vs llama.cpp Decode 성능 격차 원인 분석

**환경**: Jetson AGX Xavier (sm_72, CUDA 11.8, UMA, JetPack R35.5)
**모델**: Llama 3.2 1B F16 (safetensors → llm.rs, gguf → llama.cpp)
**벤치**: 30 토큰 decode, 프롬프트 `"The capital of France is"`, 3회 반복, 각 run 간 30초 idle

## Executive Summary

- **격차**: llm.rs cuda-embedded 26.3 tok/s vs llama.cpp 35.2 tok/s → **-25.2%**
- **1차 병목**: `matmul_ffn` (54.9%) + `lm_head` (16.0%) = decode GPU time의 **70.9%**
- **2차 병목**: CPU/sync overhead 18% (GPU 33.5 ms/tok vs wall-clock 40.9 ms/tok at profile ON, 38.4 ms/tok at profile OFF)
- **구조적 원인**: sm_72는 tensor core 없음 → F16 GEMV가 memory-bound이고 현재 llm.rs는 **61% 대역폭 활용** (113 GB/s 가능한 Xavier에서 83.8 GB/s 달성)

## 재현성 (3회 평균)

| 엔진 | Prefill tok/s | Decode tok/s | Decode ms/tok | 표준편차 |
|---|---:|---:|---:|---:|
| llm.rs cuda-embedded (profile OFF) | 143.2 | **26.3** | 38.05 | ±0.22 |
| llm.rs cuda-embedded (profile ON) | 137.6 | 24.5 | 40.86 | ±0.11 |
| llama.cpp (tg30, -ngl 99) | — | **35.19** | 28.42 | ±0.20 |

- Profiler overhead: **6.8%** decode, 3.9% prefill — Phase C 실험 해석 시 baseline은 profile OFF 사용
- Llama.cpp prefill (pp) 측정은 `-p 0`으로 누락 — 이전 데이터 156 tok/s 참조

## Per-op breakdown (llm.rs run1, 29 decode tokens, profile ON)

| label | calls | total ms | mean ms | pct | ms/token |
|---|---:|---:|---:|---:|---:|
| matmul_ffn | 2784 | 532.7 | 0.191 | **54.9%** | 18.37 |
| lm_head | 62 | 154.8 | 2.497 | **16.0%** | 5.34 |
| matmul_qkv | 2784 | 115.4 | 0.041 | 11.9% | 3.98 |
| matmul_wo | 928 | 60.6 | 0.065 | 6.3% | 2.09 |
| matmul (unlabeled) | 224 | 48.3 | 0.216 | 5.0% | 1.66 |
| rms_norm | 1023 | 12.3 | 0.012 | 1.3% | 0.42 |
| rope | 992 | 10.9 | 0.011 | 1.1% | 0.38 |
| add_assign | 992 | 10.4 | 0.010 | 1.1% | 0.36 |
| attention (decode) | 464 | 9.9 | 0.021 | 1.0% | 0.34 |
| silu_mul | 496 | 5.3 | 0.011 | 0.6% | 0.18 |
| kv_update | 496 | 5.1 | 0.010 | 0.5% | 0.18 |
| cast | 224 | 2.8 | 0.012 | 0.3% | 0.10 |
| flash_attn_prefill | 32 | 0.7 | 0.020 | 0.1% | 0.02 |
| gather | 31 | 0.3 | 0.009 | 0.03% | 0.01 |
| **TOTAL GPU** | **11532** | **969.5** | — | 100% | **33.43** |

## 이론 한계 대비 실효 활용도

Llama 3.2 1B 디코드 1 토큰당 FFN work:
- `gate + up + down`: 2 × (2048 × 8192) + (8192 × 2048) = 50.3 MFLOPs × 16 layers = 805 MFLOPs
- 29 토큰: 23.3 GFLOPs → `matmul_ffn` 532 ms = **43.9 GFLOPS 지속**
- Xavier peak F16 (CUDA 코어): 512 GFLOPS → **실효 활용도 8.6%**

메모리 대역폭 관점:
- FFN per token per layer weight read: 2×(2048×8192×2B) + (8192×2048×2B) = 96 MB
- 16 layers = 1.54 GB/token × 29 = 44.7 GB / 532 ms = **84.0 GB/s** (matmul_ffn만)
- Xavier peak BW: 137 GB/s → **61.3% 활용**
- llama.cpp는 35.19 tok/s 기준 동일 weight로 ~111 GB/s 달성 추정 (**81%**)

**결론**: matmul_ffn은 compute-bound이 아닌 **memory-bound GEMV**. 개선 여지는 전용 F16 GEMV 커널로 bandwidth 활용 20%+ 향상.

## CPU/sync overhead 분석

- GPU kernel time total: 33.43 ms/tok (측정됨)
- Wall-clock decode (profile ON): 40.86 ms/tok → overhead **7.4 ms/tok (18%)**
- Wall-clock decode (profile OFF): 38.05 ms/tok → overhead **4.6 ms/tok (12%)**
  - (profiler 자체 오버헤드 6.8% = 약 2.8 ms/tok, 이 차이가 profile ON/OFF 간극과 일치)

예상 원인:
- `cuda_embedded/mod.rs`의 30+ `synchronize()` 호출로 인한 CPU-GPU stall
- cuGraph 미사용 → 토큰당 160+ kernel launch × ~10µs latency ≈ **1.6 ms/tok** 이론 하한
- 나머지 ~3 ms는 cuBLAS argument binding / PTX dispatch 등 추정

## 가설별 기여도 추정

격차 8.9 tok/s (40.86 → 28.42 ms/tok = -12.4 ms/tok gap) 분해:

| 가설 | 추정 개선 | 근거 |
|---|---:|---|
| **H1 sync 배치화** | -3~4 ms/tok (+2.5 tok/s) | CPU overhead 7.4ms → 3ms 수준 |
| **H2 F16 GEMV (FFN + lm_head)** | -6~7 ms/tok (+4.5 tok/s) | BW 61% → 85% 가정 시 matmul_ffn 18.4→13.3, lm_head 5.4→3.9 |
| **H3 cuGraph** | -1~2 ms/tok (+0.5 tok/s) | H1과 일부 중복, dispatch latency 제거 |
| **잔여** | -1~2 ms/tok | 측정 noise, kernel fusion, 기타 |
| **합** | ~ +9 tok/s | 목표 35 tok/s 도달 가능 |

H2가 단일 항목 최대 기여. H1은 구현 비용 낮음. H3는 구조 복잡.

## 흥미로운 발견

1. **`lm_head` 단일 호출 2.5 ms** — vocab 128K × 2048 F16 = 500MB weight read. per-token BW: 500MB / 2.5ms = 200 GB/s 요구(!) 하지만 measurement 나옴 → 실제 llama-3.2의 vocab은 lm_head에서 tied embedding(`embed_tokens` F16) 사용하므로 이론적으로 같은 weight. Xavier BW 137GB/s 한계에 걸려 있음. 실측 200GB/s은 이상함 → 아마 lm_head에서 top-K 연산(topk sampling) 포함. 추가 조사 필요.
2. **`matmul` 라벨 미부여 224 calls (0.22 ms/call)** — set_op_label 경로 누락된 matmul (probably KV scatter prep or prefill path). 계측 완성도 점검.
3. **`flash_attn_prefill` 0.1% 영향** — prefill이 6 토큰으로 짧아 attention 지연은 decode 병목 아님.
4. **`attention` (decode) 1.0%** — SLM softmax + naive reduction에도 불구하고 절대치 작음 (KV seq_len 최대 35). 긴 컨텍스트(2K+)에서는 중요해질 가능성.

## Phase C 실험 우선순위

1. **C1 (H1 검증, 구현 1-2h)**: `synchronize()` 호출을 decode hot path에서 임시 제거 (layer 끝/토큰 끝만 유지) → decode tok/s 측정
2. **C2 (H2 검증, 구현 1d)**: `mul_mv_f16_f32` (llama.cpp CUDA) 참조해서 간이 F16 GEMV 커널 작성 → matmul_ffn + lm_head 경로만 치환
3. **C3 (H3 검증, 구현 2d)**: 첫 토큰 graph capture → 이후 `cuGraphLaunch`. decode loop만 대상.

단, C1+C2만으로도 +7 tok/s 전후 예상이므로 C3는 선택적.

## 파일

- Raw logs: `raw/llmrs_{profile,baseline}_run{1,2,3}.log`, `raw/llamacpp_run{1,2,3}.log`
- Per-op JSON (Jetson 측): `results/profile/cuda_embedded_decode_*.json` (호스트에 회수 아직 X — 필요 시 scp)
- thermal 체크: `raw/tegrastats_before.txt`

## 관련 커밋

- `42134a3` refactor(cuda_embedded): resync baseline with cuda_pc
- `8705ac3` feat(cuda_embedded): add CUDA event-based per-op profiler
