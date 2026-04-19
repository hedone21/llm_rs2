# P3: cuda_embedded 디바이스 전용 weight 할당 (`--cuda-weights-device`)

작성일: 2026-04-19
대상 HEAD: P3 구현 브랜치
근거: `.agent/research/2026-04-19_llamacpp_vs_cuda_embedded_diff.md` [P3] (llama.cpp `ggml-cuda.cu:241`의 `integrated=false` 우회)

## 작업 목적

1. Jetson UMA(pinned host) 캐시 coherency 취약성을 피해 weight를 device-only VRAM/carveout으로 이동.
2. 단독 성능 효과는 작음(예상 +1~1.5 tok/s), 주 가치는 후속 P1(cuGraph)의 안전 전제.
3. activation/KV/workspace는 host-pinned 유지 — 기존 zero-copy 이점 보존.

## 현재 weight 로딩 경로 (조사 결과)

- `load_model()` (loader/mod.rs) → `TensorSource::load_tensor(id, is_weight=true, backend, ...)` → `SafetensorsSource::load_raw`가 safetensors view를 dtype-specific CPU 버퍼로 (de)quantize → `finalize_tensor`에서 `backend.copy_from(&cpu_tensor)`로 GPU 업로드.
- 이후 `generate.rs::main`에서 GPU 경로에 한해 `TransformerModel::migrate_weights_to_cuda()`를 호출해 모든 weight를 다시 `backend.copy_from`으로 복사 (historical 경로 — 원래 CPU 텐서가 남아 있던 시절의 로직). 이 재복사가 loader의 할당을 덮어써, weight 저장 위치의 **최종** 결정자는 `migrate_weights_to_cuda`임.
- 따라서 P3는 (a) loader와 (b) `migrate_weights_to_cuda` 두 지점에서 weight 분기를 걸어야 함.

## 구현 개요

1. **`Backend` trait에 `copy_weight_from()` 추가** (기본 구현은 `copy_from`으로 위임). weight 업로드 전용 경로. OpenCL/CPU는 override 없음 — 기존 동작 유지.
2. **`CudaBackend`**:
   - `weights_device: Arc<AtomicBool>` 플래그 + `set_weights_device()` / `weights_device_enabled()` API.
   - `copy_weight_from()` override: 플래그 on + UMA일 때 `CudaDeviceBuffer::new(size, dtype)` + `copy_from_host(src_ptr, size)` (cuMemAlloc + cuMemcpyHtoD). discrete GPU는 무시 (관리 메모리가 이미 migrate).
   - **입력이 이미 `CudaDeviceBuffer`면 재업로드 대신 `Arc` 재활용** — `migrate_weights_to_cuda`가 loader가 이미 올린 device tensor를 다시 들어오므로 no-op path 필수. 이 처리 전 최초 테스트에서 weight 전체가 zero-fill되어 출력이 EOS만 나오는 회귀 발견 및 수정.
3. **loader**:
   - `safetensors.rs::load_raw`, `gguf.rs::load_raw`, `gguf.rs::load_with_dequant`: `is_weight=true`이면 `copy_weight_from`, 아니면 `copy_from`.
   - `loader/mod.rs`: tied lm_head 복제 + GPU-side `embed_tokens` 업로드도 `copy_weight_from`.
4. **`TransformerModel::migrate_weights_to_cuda`**: wq/wk/wv/wo/w_gate/w_up/w_down/lm_head/embed_tokens는 `copy_weight_from`, norms/biases는 `copy_from`. 매크로를 `migrate_w!`와 `migrate_n!`로 분리.
5. **CLI**: `--cuda-weights-device` 플래그 추가, `CudaBackend` 생성 직후 `set_weights_device(true)` (model load 전).
6. **discrete GPU 방어**: flag on + `is_discrete_gpu()`이면 warning 후 무시.

수정 파일:
- `engine/src/core/backend.rs` — `copy_weight_from` default method
- `engine/src/backend/cuda_embedded/mod.rs` — `weights_device` 필드/API/override
- `engine/src/models/loader/safetensors.rs`, `gguf.rs`, `mod.rs` — weight 경로 분기
- `engine/src/models/transformer.rs::migrate_weights_to_cuda` — weight vs norm 분기
- `engine/src/bin/generate.rs` — CLI 플래그 배선

## 벤치 결과 (Jetson AGX Xavier, Llama 3.2 1B F16, prompt "The capital of France is", -n 30, T=0)

| 조건 | Decode ms/tok | Decode tok/s | Avg TBT tok/s | 출력 |
|---|---|---|---|---|
| baseline (flag off) run 1 | 35.29 | 28.3 | 27.9 | "Paris. It's the most visited city..." |
| baseline run 2 | 35.24 | 28.4 | 28.0 | (동) |
| baseline run 3 | 35.28 | 28.3 | 28.0 | (동) |
| **baseline avg** | **35.27** | **28.33** | **27.97** | OK |
| `--cuda-weights-device` run 1 | 35.62 | 28.1 | 27.7 | "Paris. It's the most visited city..." |
| `--cuda-weights-device` run 2 | 35.18 | 28.4 | 28.1 | (동) |
| `--cuda-weights-device` run 3 | 35.65 | 28.0 | 27.7 | (동) |
| **`--cuda-weights-device` avg** | **35.48** | **28.17** | **27.83** | OK |
| `--cuda-weights-device --cuda-defer-sync` run 1 | 26.98 | 37.1 | 36.4 | garbage |
| run 2 | 27.05 | 37.0 | 36.3 | garbage |
| run 3 | 27.02 | 37.0 | 36.4 | garbage |
| **combined avg** | **27.02** | **37.03** | **36.37** | FAIL |

### 해석

- **Correctness**: flag 유/무 모두 동일한 정확한 출력 확인.
- **성능 regression 없음**: flag off 28.33 tok/s는 기존 28.2 tok/s 기준선과 일치.
- **단독 효과**: 28.17 vs 28.33 — 약 -0.5% (측정 노이즈 범위). 예상 +1~1.5 tok/s 미달.
  - 원인 가설: Xavier UMA에서 cuMemAlloc carveout과 cuMemHostAlloc이 물리적으로 같은 LPDDR4 DRAM을 공유하며, 실측상 L2/cache path 차이가 decode-shape matmul(BW bound)에서는 관측되지 않음. llama.cpp 대비 gap의 주 원인은 GEMV 커널 구조(P0)와 per-op sync(P1/P2)임이 재확인됨.
- **C1 재시도 (`--cuda-weights-device --cuda-defer-sync`)**: 속도는 37.0 tok/s로 Phase C1 이론 상한(35.85)과 동등 이상 달성했으나 **출력 garbage** — UMA coherency 문제가 weight가 아니라 **activation/KV**에도 존재함이 드러남. 즉 P3는 C1의 correctness를 단독으로 해결하지 못하며, P1 cuGraph 또는 activation까지 device로 옮기는 후속 작업이 필요. P3 자체는 "weight 경로의 coherency를 제거"했다는 의미에서 후속 작업의 필요 조건만 충족.

## 결론

- P3 목표 (flag 미지정 시 regression 없음 + weight device-only 경로 제공) 달성.
- C1 correctness 문제의 일부 원인은 제거됐으나 전체 해결에는 activation/KV 처리가 추가로 필요.
- **P3 커밋 진행**. C1 완전 해결은 P1(cuGraph) 작업에서 재검토.

## 로그

- `baseline_run{1,2,3}.log` — flag off
- `weights_device_run{1,2,3}.log` — `--cuda-weights-device`
- `weights_device_with_defer_sync_run{1,2,3}.log` — combined (garbage output 원인 추적용)
