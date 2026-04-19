# cuda_embedded `maybe_sync` 호출 지점 분류

작업 파일: `engine/src/backend/cuda_embedded/mod.rs`

원본 HEAD (`4070ddc` + `unused_mut` fix) 기준 33개 호출 지점을 10개 카테고리로 분류.
각 라인은 해당 kernel/cuBLAS 호출 **직후** 혹은 CPU fallback **직전**에 위치.

## 카테고리별 launch sites

| 카테고리 (`SyncCat`) | Mnemonic | decode 경로에서 per-token 발동 횟수 (Llama 3.2 1B, 16 layers) | 해당 launch |
|--|--|--|--|
| `ElemAdd` | add | 32회 (layer당 2회 × 16) | `add_assign` |
| `ElemAct` | act | 16회 (FFN) | `silu_mul`, `gelu_tanh_mul` |
| `ElemMisc` | misc | ~0회 (decode에서는 대부분 미실행) | `scale`, `softmax`, `cast_f16_f32`, `add_row_bias` |
| `RmsNorm` | rms | 33회 (attn-pre + ffn-pre + final, 2x/layer + 1) | `rms_norm`, `rms_norm_oop` |
| `Rope` | rope | 32회 (Q/K 각각 × 16) | `rope_inplace` |
| `Matmul` | mm | ~81회 (QKV 3 + O 1 + FFN gate/up/down 3 = 7/layer, × 16 + lm_head, plus pre-cuBLAS guard당 1) | `cublas sgemm`/`gemm_ex`/`gemv_f16_*`, pre-cuBLAS guard |
| `KvScatter` | kv | 16회 (layer당 1) | `kv_scatter`, `kv_scatter_batch` |
| `Attention` | attn | 16회 (flash_attn_gen layer당 1) | `flash_attn_f32`, `flash_attn_f16kv`, `flash_prefill_*` |
| `Gather` | gather | 1회 (token 1개 embedding) | `gather_f16` |
| `FallbackPre` | fallback | 0회 (F16 weight / F16 KV에서는 fallback path 안 탐) | CPU fallback 진입 직전 (attention_gen scores_out, Q4_0 KV, dtype 미매칭 등) |

**하드코딩 sync (카테고리 외)**: `cast` (F32→F16) 직후 `self.synchronize()` — CPU가 즉시 dst 읽는 경로. 초기화/load 타임만 exercise 되며 decode 경로에는 없음.

## 전체 호출 지점 (라인 번호 + 의미)

`self.maybe_sync_cat(SyncCat::X)` 형태로 전환된 곳. 번호는 HEAD 기준 기존 `maybe_sync` 호출이 있던 라인을 리매핑한 cat 전환 후 상태. 동일 함수에 여러 지점이 있으면 (A, B) 형태로 표기.

| 함수 | 라인(대략) | 의미 | 카테고리 |
|--|--|--|--|
| `flash_attention_prefill` | head_dim unsupported early-return | fallback guard | FallbackPre |
| `flash_attention_prefill` | kernel match unsupported | fallback guard | FallbackPre |
| `flash_attention_prefill` | after kernel launch | attn after | Attention |
| `flash_attention_prefill` | no device ptr | fallback guard | FallbackPre |
| `matmul_transposed` | top of function (pre-cuBLAS) | stream ordering guard | Matmul |
| `matmul_transposed` | gemv_f16_f16_f32 after | GEMV after | Matmul |
| `matmul_transposed` | gemv_f16_f32_f32 after | GEMV after | Matmul |
| `matmul_transposed` | cublas sgemm after | cuBLAS after | Matmul |
| `matmul_transposed` | F32→F16 cast + gemm_ex after | cuBLAS after | Matmul |
| `matmul_transposed` | F16×F16 gemm_ex after | cuBLAS after | Matmul |
| `add_assign` | after kernel | residual | ElemAdd |
| `scale` | after kernel | misc | ElemMisc |
| `silu_mul` | after kernel | FFN activation | ElemAct |
| `gelu_tanh_mul` | after kernel | FFN activation | ElemAct |
| `rms_norm` | after kernel | rmsnorm | RmsNorm |
| `rms_norm_oop` | after kernel | rmsnorm | RmsNorm |
| `softmax` | after kernel | misc (decode 미사용) | ElemMisc |
| `rope_inplace` | after kernel | rope | Rope |
| `cast` F32→F16 | hardcoded `synchronize()` (CPU read after) | **load-time only** | (hardcoded) |
| `cast` F16→F32 | after kernel | misc | ElemMisc |
| `cast` unsupported | CPU fallback guard | fallback | FallbackPre |
| `add_row_bias` | after kernel | misc (Llama bias 없음 → 0회) | ElemMisc |
| `kv_scatter_f32_to_f16` | after kernel | kv-write | KvScatter |
| `kv_scatter_f32_to_f16` | no device ptr fallback | fallback | FallbackPre |
| `kv_scatter_f32_to_f16_batch` | after kernel | kv-write | KvScatter |
| `kv_scatter_f32_to_f16_batch` | no device ptr fallback | fallback | FallbackPre |
| `gather` | non-F16 CPU fallback | fallback | FallbackPre |
| `gather` | after kernel | embedding | Gather |
| `gather` | no device ptr fallback | fallback | FallbackPre |
| `attention_gen` | scores_out CPU fallback | fallback | FallbackPre |
| `attention_gen` | F32 flash_attn after | attn | Attention |
| `attention_gen` | F16 flash_attn after | attn | Attention |
| `attention_gen` | Q4_0/unsupported fallback | fallback | FallbackPre |
| `attention_gen` | no device ptr fallback | fallback | FallbackPre |

## 설계 노트

- `SyncPolicy`는 `AtomicU32` 비트마스크. `SyncCat::X as u32` 비트가 set 되어 있으면 해당 launch site에서 `synchronize()` 실제 호출.
- `defer_sync=true`는 mask와 관계없이 모든 sync 억제 (legacy `--cuda-defer-sync` 호환).
- `legacy Elementwise` 이름은 `elem_add + elem_act + elem_misc` 3-way로 분해됨. Legacy CLI 문자열 `elementwise`는 그대로 3개 bit를 한 번에 세팅하도록 `parse()`에서 처리.
