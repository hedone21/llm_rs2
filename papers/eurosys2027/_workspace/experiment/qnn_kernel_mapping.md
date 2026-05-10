# QNN-GPU Kernel Mapping (Phase R Wave 1, R-A1)

**작성일**: 2026-05-09
**SDK**: QNN 2.33.0.250327 (`third_party/qnn_sdk_2.33/include/QNN/QnnOpDef.h`, 329개 prebuilt op)
**Production OpenCL backend**: `engine/src/backend/opencl/`

---

## 0. 결론

**GREEN**: 우리가 사용하는 op 중 **핵심 op는 모두 prebuilt 매칭 또는 graph composition으로 표현 가능**.
부재 prebuilt 2개 (RoPE, attention)는 sin/cos + multiply, matmul + softmax + matmul로 composition 가능.
KIVI quant attention은 우리 프로토타입이므로 마이그레이션 first-stage scope 밖.

---

## 1. Production Kernel Inventory

### 1.1 별도 .cl 파일 (21개 — `engine/src/backend/opencl/` `include_bytes!` 그렙 결과)

| 파일 | 역할 | QNN 매핑 |
|------|------|---------|
| attention_scores.cl | attention score (eviction용) | composition |
| cvt.cl | tensor 변환 | QNN_OP_CAST |
| cvt_q4_0_noshuffle_fused.cl | Q4_0 dequant fused | QNN_OP_DEQUANTIZE (per-block 검증 필요) |
| flash_attn_f32.cl, flash_attn_f32_f16.cl | flash attention | composition (matmul + masked softmax + matmul) |
| gemv_noshuffle_q4_0.cl | Q4_0 GEMV | Dequantize + MatMul composition |
| get_rows.cl | embedding lookup | QNN_OP_GATHER |
| kivi_attn.cl, kivi_q2.cl | KIVI quant attention | **out of scope** (KIVI 프로토타입) |
| mul_mm_*_l4_lm.cl (3개: f16/f32/q4_0) | matmul (prefill) | QNN_OP_MAT_MUL or FullyConnected |
| mul_mv_*.cl (5개: f16/f32/q4_0 변형) | matmul-vector (decode) | QNN_OP_MAT_MUL |
| quantize_q4_0.cl | Q4_0 quantize (KV cache) | composition or UDO |
| score_reduce.cl | score reduction (eviction) | host-side (graph 외부) |
| simple_ops.cl | 29개 fused/simple op (1.2 참조) | (1.2) |
| transpose.cl | transpose | QNN_OP_TRANSPOSE |

### 1.2 simple_ops.cl 안의 29개 kernel

| 우리 kernel | 우리 op | QNN 매핑 |
|------------|---------|---------|
| kernel_rms_norm_opt / _opt_f4 / _simple / _oop / _oop_f4 | RmsNorm | ✓ QNN_OP_RMS_NORM |
| kernel_add_rms_norm_oop / _oop_f4 / _oop_f4_sigflag | residual + RmsNorm fused | composition (Add + RmsNorm) |
| kernel_fused_norm_merge / _f4 | norm + merge fused (tensor partition) | composition (Add + RmsNorm) |
| kernel_softmax_opt / _simple | Softmax | ✓ QNN_OP_SOFTMAX (or MaskedSoftmax) |
| kernel_rope_opt / _simple | RoPE | ✗ **prebuilt 부재** → composition (sin/cos table + Multiply) 또는 UDO |
| kernel_scale_opt / _simple | scalar multiply | ✓ QNN_OP_ELEMENT_WISE_MULTIPLY |
| kernel_add_assign_opt / _simple | in-place add | ✓ QNN_OP_ELEMENT_WISE_ADD |
| kernel_silu_mul_opt / _simple | SiLU * gate (SwiGLU 핵심) | composition (Sigmoid + Multiply + Multiply) |
| kernel_partition_fused_merge_residual_f4 | partition merge + residual | composition |
| kernel_copy_slice_simple | tensor slice | QNN_OP_STRIDED_SLICE |
| kernel_add_row_bias | bias add | ✓ QNN_OP_ELEMENT_WISE_ADD |
| kernel_gelu_tanh_mul | GELU fused | ✓ QNN_OP_GELU + Multiply |
| kernel_attn_gen / _half | attention (decode) | composition (matmul + softmax + matmul) |
| kernel_cast_f32_to_f16 | dtype cast | ✓ QNN_OP_CAST |
| kernel_kv_scatter_f32_to_f16 / _batch | KV update (cast + scatter) | composition (Cast + Concat or ScatterND) |

---

## 2. QNN prebuilt op 가용성

### 2.1 사용 가능한 prebuilt (직접 매칭)
- **MatMul** (`QNN_OP_MAT_MUL`) ✓
- **FullyConnected** (`QNN_OP_FULLY_CONNECTED`) ✓ (matmul + bias 통합 옵션)
- **RmsNorm** (`QNN_OP_RMS_NORM`) ✓ (epsilon, axes 파라미터 지원)
- **Softmax** (`QNN_OP_SOFTMAX`) + **MaskedSoftmax** (`QNN_OP_MASKED_SOFTMAX`) ✓
- **LayerNorm** (`QNN_OP_LAYER_NORM`) ✓
- **Sigmoid** (`QNN_OP_SIGMOID`) ✓
- **Gelu** (`QNN_OP_GELU`) ✓
- **HardSwish** (`QNN_OP_HARD_SWISH`) ✓
- **ElementWise{Add, Multiply, Subtract, Divide, ...}** ✓
- **Cast** (`QNN_OP_CAST`) ✓
- **Concat** (`QNN_OP_CONCAT`, axis 지정) ✓
- **Reshape** / **Transpose** ✓
- **Gather** ✓
- **Dequantize** (`QNN_OP_DEQUANTIZE`) ✓ (per-block scheme 호환성은 R-A4)

### 2.2 prebuilt 부재 → graph composition으로 표현 가능
- **RoPE**: sin/cos lookup table (constant tensor) + `QNN_OP_ELEMENT_WISE_MULTIPLY` + `QNN_OP_ELEMENT_WISE_ADD`로 composition. ORT QNN EP에서 동일 패턴 사용 (PR #23136 reference).
- **Attention (flash_attn 포함)**: `QNN_OP_MAT_MUL` (Q×K^T) + `QNN_OP_ELEMENT_WISE_MULTIPLY` (scale) + `QNN_OP_MASKED_SOFTMAX` + `QNN_OP_MAT_MUL` (×V). flash 패턴은 graph fusion이 아니라 별도 customizing 필요할 수 있음 (R-B2에서 성능 측정).
- **SiLU**: `QNN_OP_SIGMOID` + `QNN_OP_ELEMENT_WISE_MULTIPLY` (x * sigmoid(x)).
- **Add+RmsNorm fused / fused_norm_merge**: 분리 표현 가능. graph optimizer가 fuse 여부는 QNN compiler 책임.

### 2.3 graph 외부 처리 (host-side)
- **score_reduce**: eviction 정책의 일부. KV cache 관리 layer에서 처리 (production에서도 별도 호출).
- **quantize_q4_0** (KV cache용): KV cache eviction 정책 사용 시에만. Phase R first-stage scope에서는 F16/F32 KV cache 우선.

### 2.4 마이그레이션 first-stage scope 밖
- **KIVI** (kivi_attn.cl, kivi_q2.cl): 프로토타입. 마이그레이션 후 별도 검증.
- **MoE** (mul_mv_id_*, gemm_moe_*): Qwen2.5-1.5b는 dense 모델, 본 마이그레이션 scope 외.
- **Conv2D / ssm_conv / Argsort / Pad / Upscale 등**: LLM forward 외 일반 op, 사용 안 함.

---

## 3. 매칭 통계

| 카테고리 | 개수 | 비율 |
|---------|------|------|
| Prebuilt 직접 매칭 | 14 / 17 핵심 op | 82% |
| Composition 가능 (RoPE, attention, SiLU 등) | 3 / 17 | 18% |
| **표현 가능 합계** | **17 / 17** | **100%** |
| 진정 부재 (UDO 또는 host-side 필수) | 0 (first-stage scope 기준) | 0% |

**Pass 기준** (plan §3 R-A1): ≥ 9개 prebuilt 매칭 → **GREEN 충족** (14개).

---

## 4. 리스크 잔존

### 4.1 R-A4로 이관: Q4_0 호환성
QNN_OP_DEQUANTIZE는 per-tensor 또는 per-axis 양자화 scheme. 우리 GGUF Q4_0은 **32 element block마다 FP16 scale**. QNN Tensor의 `Qnn_Quantization_t`가 block-wise scheme 지원하는지 R-A4에서 검증.

### 4.2 R-B2로 이관: flash_attn 성능
prebuilt가 없는 상태에서 graph composition이 우리 flash_attn_v75 (DK=128, online softmax, register-tile)와 비교해서 성능 무손실인지 R-B2에서 측정.

### 4.3 R-A3로 이관: KV cache dynamic shape
Concat을 매 step 새 K/V (1 token)에 대해 호출하는 방식 또는 ScatterND가 production에서 dynamic shape 지원하는지 R-A3에서 검증.

---

## 5. 결정

**R-A1 = GREEN** — Wave 2 진입 가능.

후속 검증 (Wave 2/3로 이관):
- R-A2: RoPE / SiLU composition vs UDO 비용 비교 (composition으로 충분할 가능성 높음)
- R-A3: KV cache dynamic shape (Concat 또는 ScatterND)
- R-A4: Q4_0 block-wise quantization scheme 호환
- R-B2: attention composition vs production flash_attn 성능
