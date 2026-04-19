# cuda_pc synchronize() → SyncCat 분류표

HEAD `e5154b0` 기준 `engine/src/backend/cuda_pc/mod.rs` 내부의 `self.synchronize()?` 호출을 cuda_embedded의 10-way `SyncCat` enum으로 분류한다.

self_test() 내 3건 (lines 218, 250, 299)은 init 1회만 실행 — 정책 검증 범위 밖이라 그대로 둔다. `read_buffer()` / `copy_from()` 내 sync (lines 1238, 1254)는 API 계약(호출 직후 CPU 읽기 허용) 보장용이라 **정책 대상 아님** — 그대로 둔다. 아래 표는 inference dispatch 경로의 `maybe_sync_cat` 치환 대상만 포함한다.

| # | Line | 위치 / 선행 op | SyncCat | 근거 |
|---|------|---------------|---------|------|
| 1 | 376 | `flash_attention_prefill` head_dim/dtype 미지원 가드 → CPU fallback 직전 | `FallbackPre` | 선행 GPU 쓰기가 CPU 경로로 드랍되기 전 완료 보장 |
| 2 | 412 | `flash_attention_prefill` dtype×head_dim 매칭 실패 → CPU fallback 직전 | `FallbackPre` | 동일 |
| 3 | 438 | `flash_attention_prefill` 커널 launch 직후 성공 path | `Attention` | cuda_embedded line 1789 `flash_attention_prefill` 동일 분류 |
| 4 | 441 | `flash_attention_prefill` 디바이스 포인터 없음 → CPU fallback 직전 | `FallbackPre` | 동일 |
| 5 | 455 | `matmul_transposed` 진입 직후 pre-launch ordering guard | `Matmul` | cuda_embedded line 926 "pre-launch input-ordering guard" 동일 |
| 6 | 505 | `matmul_transposed` F32xF32 sgemm 직후 | `Matmul` | cuda_embedded line 984 cuBLAS GEMM 동일 |
| 7 | 567 | `matmul_transposed` F32xF16 (cast+gemm_ex) 직후 | `Matmul` | cuda_embedded line 1003 동일. cast F32→F16은 같은 스트림이라 Matmul에 fold |
| 8 | 599 | `matmul_transposed` F16xF16 gemm_ex 직후 | `Matmul` | cuda_embedded line 1038 동일 |
| 9 | 638 | `add_assign` kernel 직후 | `ElemAdd` | cuda_embedded line 1184 동일. residual accumulate, 레이어당 2회 |
| 10 | 657 | `scale` kernel 직후 | `ElemMisc` | cuda_embedded line 1206 동일 |
| 11 | 677 | `silu_mul` kernel 직후 | `ElemAct` | cuda_embedded line 1229 동일 |
| 12 | 697 | `gelu_tanh_mul` kernel 직후 | `ElemAct` | cuda_embedded line 1252 동일 |
| 13 | 730 | `rms_norm` kernel 직후 | `RmsNorm` | cuda_embedded line 1288 동일 |
| 14 | 770 | `rms_norm_oop` kernel 직후 | `RmsNorm` | cuda_embedded line 1331 동일 |
| 15 | 810 | `softmax` kernel 직후 | `ElemMisc` | cuda_embedded line 1374 동일. Llama decode에서 거의 안 탐 |
| 16 | 855 | `rope_inplace` kernel 직후 | `Rope` | cuda_embedded line 1422 동일 |
| 17 | 879 | `cast` F32→F16 kernel 직후 (CPU가 dst를 즉시 읽는 경로) | `ElemMisc` | cuda_embedded line 1467 동일 |
| 18 | 894 | `cast` F16→F32 kernel 직후 (CPU가 dst를 즉시 읽는 경로) | `ElemMisc` | cuda_embedded line 1476 참조. 보수적 유지 |
| 19 | 899 | `cast` 미지원 dtype → CPU fallback 직전 | `FallbackPre` | CPU fallback 패턴 |
| 20 | 925 | `add_row_bias` kernel 직후 | `ElemMisc` | cuda_embedded line 1505 동일 |
| 21 | 976 | `kv_scatter_f32_to_f16` kernel 직후 | `KvScatter` | cuda_embedded line 1559 동일 |
| 22 | 980 | `kv_scatter_f32_to_f16` 디바이스 포인터 없음 → CPU fallback 직전 | `FallbackPre` | 동일 |
| 23 | 1034 | `kv_scatter_f32_to_f16_batch` kernel 직후 | `KvScatter` | cuda_embedded line 1620 동일 |
| 24 | 1037 | `kv_scatter_f32_to_f16_batch` 디바이스 포인터 없음 → CPU fallback 직전 | `FallbackPre` | 동일 |
| 25 | 1056 | `gather` F16 아닌 dtype → CPU fallback 직전 | `FallbackPre` | cuda_embedded line 1642 동일 |
| 26 | 1088 | `gather` F16 kernel 직후 | `Gather` | cuda_embedded line 1677 동일 |
| 27 | 1091 | `gather` 디바이스 포인터 없음 → CPU fallback 직전 | `FallbackPre` | cuda_embedded line 1680 동일 |
| 28 | 1116 | `attention_gen` scores_out 요구됨 → CPU fallback 직전 | `FallbackPre` | cuda_embedded line 1705 동일 |
| 29 | 1170 | `attention_gen` F32 KV kernel 직후 | `Attention` | cuda_embedded line 1764 동일 |
| 30 | 1192 | `attention_gen` F16 KV kernel 직후 | `Attention` | cuda_embedded line 1789 동일 |
| 31 | 1197 | `attention_gen` Q4_0/unsupported → CPU fallback 직전 | `FallbackPre` | cuda_embedded line 1794 동일 |
| 32 | 1213 | `attention_gen` 디바이스 포인터 없음 → CPU fallback 직전 | `FallbackPre` | cuda_embedded line 1810 동일 |

총 32건이 `maybe_sync_cat`으로 치환 대상. self_test() 3건(218/250/299) + read_buffer/copy_from 2건(1238/1254)은 정책 독립 → 그대로 유지.

## 카테고리별 hit 빈도 (Llama 3.2 1B decode 1 token 기준 예상)

- `ElemAdd`: 2× (residual 2회 per layer) × 16 layers = 32
- `RmsNorm`: 2× per layer × 16 = 32
- `Rope`: 1× per layer × 16 = 16
- `Matmul`: QKV+Wo+FFN_gate+FFN_up+FFN_down = 5× per layer × 16 + lm_head(1) = 81 (+pre-guard 81)
- `KvScatter`: 1× per layer × 16 = 16
- `Attention`: 1× per layer × 16 = 16
- `Gather`: 1× (embedding lookup, 시작)
- `ElemAct`: 1× silu_mul per layer × 16 = 16
- `ElemMisc`: scale/softmax/add_row_bias/cast — decode 경로에서 거의 0
- `FallbackPre`: 정상 경로에서는 0 (fallback 발동 시에만)

`Matmul` 카테고리가 압도적으로 많음 (~160 syncs/token) — 여기가 핵심 최적화 지점.

## Baseline (pre-port)

- HEAD `e5154b0` 기준 `self.synchronize()?` 호출 총 35건 (self_test 3 + read_buffer 1 + copy_from 1 + dispatch 경로 30 + flash_attention_prefill 2중복).
- 실제 위 표에 나열된 32건이 dispatch 경로 syncs (task 문서의 "40회"는 대략적인 수치).

## 포팅 전략

1. cuda_pc mod.rs 최상단에 cuda_embedded와 동일한 `SyncCat`/`SyncPolicy` + parser 추가.
2. `CudaBackend` 구조체에 `defer_sync: Arc<AtomicBool>` (기존에 없음) + `sync_policy: Arc<AtomicU32>` 필드 추가.
3. `set_defer_sync`/`set_sync_policy`/`maybe_sync_cat` 메서드 추가.
4. 위 32개 호출을 `maybe_sync_cat(cat)` 로 치환. self_test / read_buffer / copy_from 의 `self.synchronize()?`는 그대로 둔다.
5. generate.rs: 기존 `cfg(feature = "cuda-embedded")` 블록 4개 (profiler, defer_sync, sync_policy, weights_device) 중 **sync_policy와 defer_sync** 둘만 `cfg(any(feature = "cuda", feature = "cuda-embedded"))` 로 확장. profiler/weights_device는 cuda_pc 미구현이므로 cuda-embedded 전용 유지.

주의: cuda_pc에는 `profiler`, `weights_device`, `op_label_hint` 기능이 없으므로 **이번 포팅은 sync_policy와 defer_sync만** 도입한다.
