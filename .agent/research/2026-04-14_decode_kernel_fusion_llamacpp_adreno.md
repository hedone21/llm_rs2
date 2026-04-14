# Decode Kernel Fusion — llama.cpp Adreno 대비 llm_rs2 기법 조사

**작성일**: 2026-04-14
**계기**: attention과 matmul 모두 갭 원인 아님 확정 후, rms_norm/rope/silu_mul/add_assign/kv_update 합계 22.2% (~18 ms/tok) 영역의 fusion 가능성 확인.

## 핵심 결론

**kernel fusion은 22 ms/tok 갭의 주 원인이 아니다.** Fusion으로 회수 가능한 최대치는 Top 3 모두 구현 시 ~6 ms/tok (갭의 27%).

**우리가 이미 llama.cpp와 동등 또는 앞서 있다:**
- RMS+MUL fusion: 동등 (`kernel_rms_norm_opt`, `rms_norm_oop`)
- SwiGLU fusion: 동등 (`kernel_silu_mul_simple/opt`)
- **ADD+RMS+MUL 3-op fusion: 우리가 앞섬** (`kernel_add_rms_norm_oop`) — llama.cpp 없음
- **KV scatter F32→F16: 우리가 앞섬** (`kernel_kv_scatter_f32_to_f16`) — llama.cpp `set_rows.cl`은 단순 F32→F32 copy만

## Qwen2 Decode 그래프의 fusion 적용 지점

| # | op | llama.cpp kernel | fusion? |
|---|---|---|---|
| 1 | attn_norm (RMS+mul) | `rms_norm_mul` | ✓ |
| 2-4 | Q/K/V matmul + bias add | 별도 | ✗ |
| 5-6 | RoPE Q/K | `rope_norm_f32` (단독) | ✗ |
| 7 | KV cache write | `cpy_f32_f16` | ✗ |
| 8 | flash attention | `flash_attn` | 내부 흡수 |
| 9 | wo matmul | — | ✗ |
| 10 | attn residual add | `add` | ✗ |
| 11 | ffn_norm (RMS+mul) | `rms_norm_mul` | ✓ |
| 12 | gate/up matmul | — | ✗ |
| 13 | silu*gate | `swiglu` | ✓ |
| 14 | down matmul | — | ✗ |
| 15 | ffn residual add | `add` | ✗ |

llama.cpp가 Qwen decode에서 fuse하는 건 **RMS+MUL 2회 + SwiGLU 1회**만.

## Top 3 Fusion 후보 (우선순위 순)

### F1. FFN residual + next-layer pre-attn RMSNorm (최우선 권장)
- **상태**: 이미 attn residual → ffn_norm에는 `add_rms_norm_oop` 적용 중 (`forward_gen.rs:926,941`). **FFN 끝에만 미적용** (`forward_gen.rs:1076` `add_assign` + 다음 layer `rms_norm_oop`).
- **변경**: 각 layer의 `forward_gen` 진입부에서 `add_rms_norm_oop(x, ws.down, attn_norm_w)`로 합치고, 이전 layer 말미 `add_assign` 제거.
- **ROI**: ~3 ms/tok
- **난이도**: 낮음 (커널 이미 존재, dispatch 리팩토링만)
- **리스크**: 낮음 (이미 attn쪽에서 동일 fusion 검증됨)

### F2. QKV bias fusion into matmul epilogue (Qwen 전용)
- **상태**: llama.cpp도 안 함. Qwen2 Q/K/V projection에 bias 있음 → `add_row_bias` 3회 별도 dispatch (`forward_gen.rs:158~160`).
- **ROI**: ~1~2 ms/tok
- **난이도**: 중간 (`mul_mv_q4_0_f32_*` 계열 shape variants에 옵션 bias 추가)
- **리스크**: Llama 3.2(bias 없음) 분기 필요, 유지보수 비용

### F3. RoPE + K scatter fusion
- **상태**: 미반영 — `rope_inplace` 후 별도 `kv_scatter`.
- **ROI**: ~1.5 ms/tok
- **난이도**: 중간
- **리스크**: K 전용 fused 커널 신규. Register pressure 증가 (sin/cos 상수 + K 값), 32 float4 상한 근처 — **실측 필수**. `feedback_adreno_gpu_kernel_state_limit.md` 경고 적용.

### Top 3 합계 절감 기대
~6 ms/tok (82.6 → 76.6 ms/tok, 13.1 tok/s). 갭 22 ms/tok 중 **27%** 메움.

## 음의 발견 (llama.cpp도 안 하는 fusion)

1. **RoPE + QKV matmul fusion**: 양쪽 없음. RoPE는 head_dim 축 재구성 필요 → matmul tiling과 호환 어려움.
2. **Attention output + wo + residual fusion**: 없음.
3. **GEMV epilogue activation/bias**: llama.cpp `mul_mv_q4_0_*` variants 모두 미지원.
4. **GROUP_NORM fusion** (`group_norm_mul_add`): Qwen/Llama는 GroupNorm 미사용.

## 결론 및 다음 방향

Fusion 영역 투자는 **F1만 채택 권장** (3 ms/tok, 저위험). F2/F3는 갭이 여전히 15 ms 이상일 때 재검토.

**나머지 갭 ~19 ms/tok은 fusion 밖에서 찾아야** 한다:
- **Kernel launch overhead / command queue flush 패턴** (우선 조사 권장)
- Attention decode shape-specific 튜닝 (이미 B-4 수준이지만 Qwen GQA ratio 특화)
- KV cache 접근 레이아웃 비교 (HeadMajor vs llama.cpp contiguous)

## 참고 파일 경로

### llama.cpp (/home/go/Workspace/llama.cpp/)
- `ggml/src/ggml-opencl/ggml-opencl.cpp:3474~3586` — fusion dispatch 로직
- `ggml/src/ggml-opencl/ggml-opencl.cpp:7302~7412` — `ggml_opencl_op_rms_norm_fused`
- `ggml/src/ggml-opencl/kernels/rms_norm.cl:106~190` — `kernel_rms_norm_mul`
- `ggml/src/ggml-opencl/kernels/glu.cl:141~171` — `kernel_swiglu`
- `ggml/src/ggml-opencl/kernels/rope.cl:45~122` — 단독 RoPE (norm_f32는 "normal" RoPE 의미, norm fusion 아님)
- `ggml/src/ggml-opencl/kernels/cpy.cl:96~` — KV F32→F16 copy
- `src/models/qwen2.cpp:21~105` — Qwen2 decode 그래프

### llm_rs2
- `engine/kernels/simple_ops.cl:32~143` — `rms_norm_opt`, `add_rms_norm_oop` (우리 앞섬)
- `engine/kernels/simple_ops.cl:355~510` — `silu_mul_*`, `add_assign_*`
- `engine/kernels/simple_ops.cl:797~818` — `kv_scatter_f32_to_f16` (우리 앞섬)
- `engine/src/layers/transformer_layer/forward_gen.rs:58, 158~160, 199~200, 926, 941, 1061, 1076` — F1 fusion 개입 지점 = 1076 + 다음 iter 58
- `engine/src/backend/opencl/mod.rs:2646~2669, 3042~3052` — add_rms_norm_oop 구현
