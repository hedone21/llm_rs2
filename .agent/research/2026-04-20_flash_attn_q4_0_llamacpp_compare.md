# flash attention Q4_0 Prefill 경로 — llm.rs vs llama.cpp

**작성**: 2026-04-20
**환경**: Qwen2.5-1.5B Q4_0, Galaxy S25 Adreno 830, llama.cpp 983df14 + Adreno kernels

## 0. 핵심 결론

본 실측 케이스(Qwen2.5-1.5B Q4_0 + KV=F16 + Adreno 830)에서 **flash attention 커널은 llm.rs가 구조적으로 불리하지 않다**. 오히려 A-3/B-1 subgroup split 덕분에 per-thread register footprint가 절반(~34 float4)이다. **병목은 Q4_0 matmul prefill 경로**:

- llama.cpp는 `GGML_OPENCL_USE_ADRENO_KERNELS` + `GGML_OPENCL_SOA_Q` 경로에서 shape-specific pre-compiled **5개의 gemv/mul_mat 프로그램** 사용 (LINE_STRIDE_A/BLOCK_STRIDE_A가 compile-time 상수).
- llm.rs는 generic `mul_mm_q4_0_f32_l4_lm` 하나만 사용.

**Flash attn 구조 차이는 prefill 시간의 0.1~0.5%에 불과**하며 5.3× 격차를 설명할 수 없음 (FFN matmul FLOPs 303 GFLOP vs flash attn 1.47 GFLOP, 비율 206×).

## 1. llm.rs flash attn prefill 지도 (확인됨)

| 항목 | 위치 | 값 |
|------|------|-----|
| dispatch entry | `engine/src/backend/opencl/mod.rs:2163` | `flash_attention_prefill_gpu(...)` |
| DK=128 커널 | `engine/src/backend/opencl/mod.rs:2197` | `kernel_flash_attn_f32_f16_dk128` |
| compile defines | `engine/src/backend/opencl/mod.rs:701` | `-DDK=128 -DDV=128 -DBLOCK_M=32 -DBLOCK_N=32` |
| WG | `engine/src/backend/opencl/mod.rs:2316-2326` | LWS=[64,1,1], `block_m=32, lanes_per_wg=64` |
| 커널 본문 | `engine/kernels/flash_attn_f32_f16.cl:52-275` | **A-3 B-1 subgroup split** (`FA_SUBGROUP_SPLIT=1`) |
| layer 호출 | `engine/src/layers/transformer_layer/forward.rs:85-115` | in-order queue, sync 없음 (profile 모드만) |

Qwen2.5-1.5B prefill: batch=1, n_q=131, n_heads_q=12, n_heads_kv=2, DK=DV=128
- GWS = [⌈131/32⌉·64, 12, 1] = [320, 12, 1], LWS=[64,1,1] → **60 WG, WG당 64 thread** (3,840 thread)
- per-thread register: `q_priv[16] + o_acc[16] + dot_acc0/1(2) ≈ 34 float4`
- SLM/WG: `l_k 8KB + l_v 8KB + l_dot 1KB ≈ 17 KB`
- barrier/tile: K/V load 1 + inner-loop 16 pair × 2 = **~33/tile**

## 2. llama.cpp flash attn prefill 지도 (확인됨)

| 항목 | 위치 | 값 |
|------|------|-----|
| dispatch entry | `ggml-opencl.cpp:8566` | `ggml_cl_flash_attn(...)` |
| 커널 선택 | `ggml-opencl.cpp:8605-8613` | Q=F32 & K=F16 → `kernels_flash_attn_f32_f16[{128,128}]` |
| compile defines | `ggml-opencl.cpp:1805-1820` | `fa_dims[{128,128,32,32}]` |
| WG | `ggml-opencl.cpp:8693-8698` | LWS=[32,1], `wg_size=block_m=32` |
| 커널 본문 | `kernels/flash_attn_f32_f16.cl:29+` | **subgroup split 없음**, thread 1개당 Q-row 1개 |
| Adreno 분기 | `ggml-opencl.cpp:2410` | flash attn에는 Adreno 전용 분기 **없음** (matmul에만 있음) |

동일 조건:
- GWS = [5·32, 12, 1] = [160, 12, 1], LWS=[32,1,1] → **60 WG, WG당 32 thread** (1,920 thread)
- per-thread register: `q_priv[32] + o_acc[32] = 64 float4` (**Adreno "32 float4 한도" 초과하지만 실측상 문제없음** — 드라이버 allocator의 관대함 시사)
- SLM/WG: `l_k 8KB + l_v 8KB = 16 KB` (l_dot 없음)
- barrier/tile: K/V load 1 = **~1/tile**

## 3. Side-by-side (확인됨)

| 측면 | llm.rs | llama.cpp |
|------|--------|-----------|
| 커널 파일 | `flash_attn_f32_f16.cl` + A-3 B-1 분기 | `flash_attn_f32_f16.cl` single-lane |
| Q4_0 KV dequant | **없음** (KV=F16) | 없음 (동일) |
| LWS | 64 | 32 |
| per-thread | ~34 float4 | ~64 float4 |
| SLM | 17 KB | 16 KB |
| tile당 barrier | ~33 | ~1 |
| Q tile 전체(n_kv=131) | **~165** | **~5** |
| REQD_SUBGROUP_SIZE | `qcom_reqd_sub_group_size("half")` (64 lane) | 없음 |
| softmax/causal/GQA | FA-2 online, in-kernel mask, group ratio div | 동일 |

`flash_attn_f16.cl`, `flash_attn_f32.cl`은 **바이트 단위 동일**. 차이는 오직 `flash_attn_f32_f16.cl` DK=128 분기 + dispatch WG size.

## 4. 격차 추정

- Prefill FLOPs 비율: flash attn 1.47 GFLOP vs FFN matmul 303 GFLOP (28 layer, 131 tok) → **206:1**
- Flash attn 구조 차이가 기여하는 prefill 시간: barrier overhead ~9,600회 (Adreno ~100ns/barrier 추정) = **~1~5 ms** (전체 1135ms 중 **0.1~0.5%**)
- **→ 5.3× 격차는 flash attn으로 설명 불가. Q4_0 matmul 경로가 주범.**

llama.cpp Q4_0 matmul 최적화 (확인됨, `ggml-opencl.cpp:2434-2600`):
- `gemv_noshuffle` + `gemv_noshuffle_general_q4_0_f32` + `mul_mat_Ab_Bi_8x4` + transpose 콤보
- **Shape별 pre-compiled program 5개** (LINE_STRIDE_A / BLOCK_STRIDE_A가 런타임 arg 아닌 compile-time 상수)
- llm.rs는 generic `mul_mm_q4_0_f32_l4_lm` 하나만 사용

실측 `Q4_0 noshuffle SOA prepared: 196 weight tensors` 로그가 SOA layout은 준비되지만, **실제 dispatch되는 matmul 커널이 shape-agnostic**이라는 점이 핵심.

## 5. 최우선 이식 후보

### 후보 1 — Q4_0 matmul prefill을 shape-specific Adreno fast path로 대체 [flash attn 범위 밖, 핵심]
- 출처: `ggml-opencl.cpp:2434-2600`, `kernels/mul_mat_Ab_Bi_8x4.cl`, `kernels/gemv_noshuffle_general_q4_0_f32.cl` (및 q8_0 버전)
- 이식: `engine/src/backend/opencl/mod.rs` Q4_0 dispatcher + shape-specific kernel variants
- 난이도: **높음** (전용 커널 + transpose preprocess + shape program cache)
- 예상 이득: **prefill 3~4×** (격차의 주범 해소)
- Adreno 교훈 준수: llama.cpp 원본 기반이므로 sub_group_reduce 회피, register 사용 제한 기 확인

### 후보 2 — flash_attn_f32_f16 DK=128을 single-lane(llama.cpp 스타일)로 복구 (A/B)
- 이식: `FA_SUBGROUP_SPLIT` 분기 toggle (env/feature gate)
- 난이도: **낮음** (single-lane 코드 이미 파일에 존재)
- 예상 이득: 불확실 (barrier overhead 제거 vs register pressure 증가). prefill 최대 0.5 ms 감소 (~0.5%)
- **이식 가치는 검증용** — Adreno register allocator 관대함 여부 확인

### 후보 3 — subgroup split 유지하며 barrier 횟수 절감
- 현재 inner-loop dot-exchange: publish + consume = 32 barrier/tile
- 개선: `l_dot[BLOCK_M][BLOCK_N/2][2]`로 확장해 tile 전체 publish 후 단일 barrier (SLM 1KB → 4KB)
- 난이도: **중간**
- 예상 이득: barrier 33→3/tile, prefill 0.3~0.5 ms 감소

## 6. 프리필 long-seq 이슈 (확인됨)

- `forward_gen` 경로는 **decode 전용**. 로그의 `[part-dbg] forward_gen entered`는 decode 첫 토큰.
- Prefill은 `forward()` → `flash_attention_prefill_gpu` (LWS=64).
- llama.cpp는 graph-based, `ggml_cl_flash_attn`에서 `n_q==1 ? _q1 : regular`로 분기.

## 7. 후속 제안

- **Phase B (핵심)**: Q4_0 matmul shape-specific Adreno path 이식 (후보 1). Senior Implementer.
- **Phase A-2 (소규모)**: FA subgroup split A/B (후보 2). Senior Implementer, 30분.
- **Phase A-3 (선택)**: FA barrier fold (후보 3). 후속 microbench.

## 참고 파일

- `engine/kernels/flash_attn_f32_f16.cl`, `flash_attn_f16.cl`, `flash_attn_f32.cl`
- `engine/src/backend/opencl/mod.rs:2151-2350, 656-760`
- `engine/src/layers/transformer_layer/forward.rs:85-115`
- `/home/go/Workspace/llama.cpp/ggml/src/ggml-opencl/kernels/flash_attn_f32_f16.cl`
- `/home/go/Workspace/llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp:1786-1851, 2410-2700, 8566-8700`
- `.agent/research/2026-04-20_qwen15b_host_vs_device_bench/llmrs_gpu_q4_0_run1.log`
- `.agent/research/2026-04-20_qwen15b_host_vs_device_bench/llamacpp_gpu_q4_0_run1.log`
