# Decode Matmul — llama.cpp Adreno 대비 llm_rs2 미반영 기법 조사

**작성일**: 2026-04-14
**계기**: c446ca8 Decode 갭 22 ms/tok. attention 43.8% (~36 ms/tok) 이후 2순위 후보인 matmul_ffn 18.8% + matmul_wo 8.2% 점검.
**제약**: Adreno 830 per-thread state 32 float4 상한 (feedback_adreno_gpu_kernel_state_limit.md).

## 핵심 결론

**matmul 영역에는 llama.cpp 대비 미반영 기법이 사실상 없다.** 22 ms/tok 갭의 주 원인은 matmul이 아니다.

근거:
1. Q4_0 GEMV (matmul_ffn, matmul_qkv, lm_head) — llama.cpp의 모든 Adreno 특화 기법(SOA, 2D transpose, image1d R32UI, sub_group_broadcast, vector broadcast, 4-wave K-split, half-wave subgroup, shape-specialized program)을 이미 구현. 완전 동등.
2. F16 GEMV (matmul_wo) — llama.cpp는 오히려 더 **단순한** `1row` 커널 사용 (WG=64, single subgroup). 우리는 WG=256 + 4-wave K-split로 더 공격적.
3. Prefill용 `mul_mat_Ab_Bi_8x4.cl` (image1d 8x4 tile) 양쪽 동일.

## 대조 테이블

### matmul_ffn (Q4_0 GEMV, M=1)

| 기법 | llama.cpp | llm_rs2 | Gap |
|------|-----------|---------|-----|
| SOA Q4_0 (scales/quants 분리) | ON | ON (`transformer.rs:371-`, `mod.rs:1480-`) | 없음 |
| 2D transpose | ON | ON | 없음 |
| image1d_buffer_t weight (R32UI) | ON (`ggml-opencl.cpp:9476-9491`) | ON (`mod.rs:1808-1836`) | 없음 |
| image1d_buffer_t activation | ON (R32UI @N=1) | ON (RGBA32F) | 거의 없음 |
| sub_group_broadcast activation 공유 | ON (`gemv_noshuffle.cl:14-63`) | ON (`gemv_noshuffle_q4_0.cl:56-156`) | 없음 |
| vector sub_group_broadcast | ON | ON (auto-fallback `mod.rs:1767-1786`) | 없음 |
| REQD_SUBGROUP_SIZE_64 | ON | ON | 없음 |
| Shape-specialized program | ON (5 variants 하드코드) | ON (lazy per-ne01, `mod.rs:1757-1796`) | 없음 |
| Local WG [64,4,1] + 4-wave reduction | ON | ON | 없음 |

### matmul_wo (F16 GEMV, [2048, 2048], M=1)

| 기법 | llama.cpp | llm_rs2 | Gap |
|------|-----------|---------|-----|
| Decode 전용 분기 | `1row` (ne11*ne12<4) | 단일 `mul_mv_f16_f32` (N_DST=2, 4-wave) | 구조적 차이 |
| WG size | 64 (single subgroup) | 256 (4 subgroups) | 우리 더 큼 |
| WG count (N=2048) | 2048 | 1024 | 우리 절반 |
| K-split 병렬 | 64-way | 256-way | 우리 더 공격적 |
| sub_group_reduce_add + qcom half-wave | ON | ON | 없음 |
| half4 + float4 vload | ON | ON | 없음 |

## 2차 튜닝 후보 (약한 권고)

### 후보 1 (저리스크 A/B): F16 1row 커널 matmul_wo 분기 연결
- **근거**: llama.cpp는 Adreno F16 decode에 의도적으로 단순 1row 사용. 우리의 공격적 K-split이 작은 K(2048)에서 reduction 오버헤드를 상쇄 못할 가능성.
- **소스**: llama.cpp `kernels/mul_mv_f16_f32_1row.cl:22-94`, dispatch `ggml-opencl.cpp:10192`.
- **우리 상태**: `engine/kernels/mul_mv_f16_f32_1row.cl` **파일 이미 존재**, dispatch 미연결.
- **예상 ROI**: ±3 ms/tok (불확실)
- **난이도**: 낮음 (shape 분기 추가만)
- **리스크**: state 불변 방향, 회귀 가능성 낮음

### 후보 2 (탈락): F16 weight image1d_buffer_t 확장
- llama.cpp도 안 함 → Adreno 구조적 한계 의심 규칙 적용. 탈락.

### 후보 3 (매우 낮음): Q4_0 GEMV WG size Qwen shape 튜닝
- llama.cpp의 하드코드(`ggml-opencl.cpp:9692-9704`)는 Llama shape 전용. Qwen 2048/11008에는 미스매치. sweep 필요하지만 ROI 매우 낮음.

## 진짜 갭이 있을 가능성이 큰 영역 (다음 조사 대상)

1. **Kernel fusion**: llama.cpp `rope_norm`, `norm_mul_add` 등. 우리는 RMSNorm만 fused. **rms_norm 7.6% + rope 4.2% = 11.8%** — matmul_wo(8.2%)보다 큰 영역.
2. **Launch overhead / implicit sync**: Event chain, clEnqueueNDRangeKernel 오버헤드.
3. **kv_update 3.3%**: llama.cpp `set_rows` vs 우리 `kv_scatter_f32_to_f16`.
4. **Layer-level micro-benchmark**: 실제 matmul_wo 단일 실행 시간을 직접 측정하면 matmul 갭 실재 여부 확정 가능.

## 구현 우선순위 권고

**matmul 자체 구현 권고 없음.** 

다음 플랜 제안:
- **Plan A** (실측 최우선): llama.cpp + llm_rs2 layer-level micro-bench으로 matmul_wo 실시간 직접 비교. < 2 ms 차이면 matmul 투자 종결.
- **Plan B** (싼 A/B): F16 1row 커널 matmul_wo에만 분기 연결 실측.
- **Plan C** (조사): Kernel fusion 조사 — llama.cpp의 `rope_norm`/`norm_mul_add`/유사 fused kernel 존재 여부 + 적용 가능성.

## 파일 경로

llama.cpp (/home/go/Workspace/llama.cpp/):
- `ggml/src/ggml-opencl/ggml-opencl.cpp:4178-4188` — use_adreno_kernels threshold
- `ggml/src/ggml-opencl/ggml-opencl.cpp:4211-4295` — SOA Q4_0 host 변환
- `ggml/src/ggml-opencl/ggml-opencl.cpp:9399-9739` — Adreno mul_mat dispatch
- `ggml/src/ggml-opencl/ggml-opencl.cpp:10179-10204` — F16 dispatch (decode 1row 분기)
- `ggml/src/ggml-opencl/ggml-opencl.cpp:2380-2488` — GEMV shape 변형 컴파일
- `ggml/src/ggml-opencl/kernels/gemv_noshuffle.cl:14-268` — Adreno Q4_0 GEMV
- `ggml/src/ggml-opencl/kernels/mul_mv_f16_f32_1row.cl:22-94` — F16 decode 1row
- `ggml/src/ggml-opencl/kernels/mul_mv_f16_f32_l4.cl:25-84` — F16 prefill l4

llm_rs2:
- `engine/src/bin/generate.rs:654-667` — noshuffle auto prepare
- `engine/src/models/transformer.rs:371-` — prepare_noshuffle_buffers
- `engine/src/backend/opencl/mod.rs:1364-1458` — matmul_q4_0 dispatch
- `engine/src/backend/opencl/mod.rs:1730-1872` — matmul_q4_0_noshuffle
- `engine/src/backend/opencl/mod.rs:1757-1796` — gemv_noshuffle_cache
- `engine/kernels/gemv_noshuffle_q4_0.cl:56-324` — Adreno Q4_0 GEMV
- `engine/kernels/mul_mv_f16_f32.cl:42-` — 현재 matmul_wo 커널
- `engine/kernels/mul_mv_f16_f32_1row.cl` — **파일 존재, dispatch 미연결**
