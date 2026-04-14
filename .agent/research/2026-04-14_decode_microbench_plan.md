# Decode Per-Op Micro-Benchmark 설치 계획

**작성일**: 2026-04-14
**계기**: attention/matmul/fusion 3연속 네거티브 → 갭 22 ms/tok의 실제 위치를 절대값 비교로 확정 필요.

## 핵심 발견 (실제 확인 완료)

### llama.cpp
- **`GGML_OPENCL_PROFILING` CMake option 존재** (`ggml/CMakeLists.txt:252`, default OFF)
- 활성화 시 `CL_QUEUE_PROFILING_ENABLE` 자동 (`ggml-opencl.cpp:3015-3019`)
- 모든 커널 dispatch event capture (`ggml-opencl.cpp:674-685`)
- backend `free()` 시 자동 덤프: `cl_profiling.csv` + `cl_trace.json` (`ggml-opencl.cpp:585-662`)
- 측정값 = 순수 GPU `cmd_end - cmd_start` ns → csv에 ms로 기록
- **런타임 환경변수 없음 — 재컴파일 필수**

### llm_rs2 현황
- 현재 `--profile`: op마다 `synchronize()` 2회 (clFinish) → 54ms/token 오버헤드 (`forward_gen.rs:27-51`)
- Queue 생성은 profiling OFF: `mod.rs:372` `Queue::new(..., None)` → 3번째 인자가 properties
- 33지점의 `ocl::core::enqueue_kernel` 마지막 인자가 `None::<&mut Event>` — **시그니처에 이미 event out 자리 있음**
- ocl crate에 `ProfilingInfo::{Start, End}` 제공 (`ocl-core/src/types/enums.rs:1849-1886`)

## 4-Phase 실행 계획

### Phase 1 — llm_rs2 event-based profiling (senior-implementer)

**목표**: `--profile-events` flag 신규. OpenCL event 기반 per-op μs 측정 (synchronize 없이).

**수정 파일 (6개)**:
1. `engine/src/bin/generate.rs` — `--profile-events` CLI flag (~10 LOC)
2. `engine/src/backend/opencl/mod.rs`
   - `:195` 구조체에 `profile_events_enabled`, `profile_events: RefCell<Vec<(String, Event)>>`, `profile_accum: RefCell<HashMap<String, u64>>` (~20 LOC)
   - `:372` Queue::new에 `flags::QUEUE_PROFILING_ENABLE` 분기 (~5 LOC)
   - 신규 `enqueue_kernel_labeled(&self, kernel, label, ...)` 래퍼 (~40 LOC)
   - 33지점 `enqueue_kernel(...)` → `enqueue_kernel_labeled(...)` 치환 (~80 LOC 기계적)
   - `flush_and_aggregate_profile()`: event에서 `End - Start` 읽어 label별 accum (~40 LOC)
3. `engine/src/profile/ops.rs` — `merge_from_events(&HashMap)` (~30 LOC)
4. `engine/src/layers/transformer_layer/forward_gen.rs` — `--profile-events` 모드에서 매크로 비활성화 (~20 LOC)
5. `engine/src/bin/generate.rs` decode loop — 스텝 말미 flush (~10 LOC)
6. `engine/src/backend/opencl/mod.rs` — import (~1 LOC)

**합계**: ~250–350 LOC, 0.5–1일

**리스크**: Event Drop 자동 release (ocl crate 관리), in-order queue에서 profiling info는 event COMPLETE 후 읽기 → 기존 `synchronize()` 이후 flush 호출하면 안전.

**폴백 (B-4)**: `prof_start!`에서 synchronize만 제거 (op당 2→1회). 10 LOC, 1시간. 절대값은 여전히 부정확, 오버헤드 54→27ms.

### Phase 2 — llama.cpp per-op 측정 (implementer 수준, 독립 실행 가능)

**재컴파일 명령**:
```bash
cd /home/go/Workspace/llama.cpp
cmake -B build-android-profile \
  -DCMAKE_TOOLCHAIN_FILE=$ANDROID_NDK/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DGGML_OPENCL=ON -DGGML_OPENCL_EMBED_KERNELS=ON \
  -DGGML_OPENCL_USE_ADRENO_KERNELS=ON \
  -DGGML_OPENCL_PROFILING=ON \
  -DCMAKE_BUILD_TYPE=Release
cmake --build build-android-profile -j --target llama-cli
```

**실행**:
```bash
adb push build-android-profile/bin/llama-cli /data/local/tmp/llama-cli-prof
adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
  ./llama-cli-prof -m Qwen2.5-1.5B-Instruct-q4_0.gguf \
    -f prompt_6k.txt -n 128 -c 6144 -ngl 99 --temp 0 --no-cnv 2>&1 | tee llama_cpp_profile.log"
adb pull /data/local/tmp/cl_profiling.csv .
```

**주의**: profile 빌드 자체의 decode ms/tok은 production보다 느림. 순수 GPU 커널 μs만 유효.

**Prefill/Decode 분리**: CSV에 dispatch 순서대로 기록. 꼬리쪽 `28 layer × ~13 kernel × decode_steps` event만 추출 (prefill = 초반 multi-token chunk × 28 layer).

### Phase 3 — 비교 스크립트 (implementer)

경로: `experiments/benchmarks/compare_layer_timing.py` (~120 LOC)

**입력**:
- `--llama-csv cl_profiling.csv`
- `--llmrs-json results/profile_events/op_profile.json`
- `--layers 28 --decode-steps 120`

**처리**:
1. llama.cpp CSV 파싱: `op_name = "Qcur-<il>"` → 논리 범주 매핑
2. 범주별 per-token μs = 합계 / (layers × decode_steps)
3. llm_rs2 json 읽어 동일 범주로 매핑
4. pandas DataFrame → markdown table

### Phase 4 — 검증 + 결론

**검증**:
- llm_rs2 profile 합 = 실측 decode ms/tok ± 5%
- llama.cpp profile 합 = profile 빌드 decode ms/tok ± 10%

**결론**: 표에서 `Δ μs` 가장 큰 3개 op = 다음 최적화 타겟.

## Op 이름 대조표 (Qwen2 decode per-layer)

| 논리 단계 | llama.cpp tensor name | llm_rs2 OpProfiler | dispatch 수 |
|---|---|---|---|
| 입력 RMSNorm | `attn_norm-N` | `rms_norm` (1st) | 1 |
| Q/K/V matmul | `Qcur-N`, `Kcur-N`, `Vcur-N` | `matmul_qkv` | 3 |
| RoPE (Q, K) | 2nd pass of `Qcur-N`, `Kcur-N` | `rope` | 2 |
| KV write | cpy to KV cache (GGML_OP_CPY) | `kv_update` | 2 |
| Flash attention | FLASH_ATTN_EXT 1회 | `attention` | 1–4 |
| WO projection | `attn_out-N` (matmul) | `matmul_wo` | 1 |
| Attn residual | `ffn_inp-N` (add) | `add_assign` (1st) | 1 |
| FFN RMSNorm | `ffn_norm-N` | `rms_norm` (2nd) | 1 |
| FFN gate | `ffn_gate-N` | `matmul_ffn` (1/3) | 1 |
| FFN up | `ffn_up-N` | `matmul_ffn` (2/3) | 1 |
| SwiGLU | `ffn_gate_par-N` | `silu_mul` | 1 |
| FFN down | `ffn_down-N` | `matmul_ffn` (3/3) | 1 |
| FFN residual | `l_out-N` (add) | `add_assign` (2nd) | 1 |

## 비교표 템플릿

| Op 범주 | llama.cpp μs/tok | llm_rs2 μs/tok | Δ μs | Δ % | 비고 |
|---|---|---|---|---|---|
| RMSNorm (×2) | | | | | attn+ffn 합계 |
| QKV matmul (Q4_0 ×3) | | | | | |
| RoPE (Q+K) | | | | | |
| KV write (K+V) | | | | | |
| Flash attention | | | | | Adreno 특화 유무 확인 |
| WO projection | | | | | |
| Attention residual | | | | | |
| FFN (gate+up+down) | | | | | |
| SwiGLU | | | | | |
| FFN residual | | | | | |
| **per-layer 합** | | | | | |
| **×28 layers** | | | | | |
| **실측 decode ms/tok** | 60.2 | 82.6 | 22.4 | | 자기 일관성 확인 |

## 실행 순서

Phase 2와 Phase 1은 독립 → 병렬 가능. Phase 3/4는 Phase 1·2 완료 후.

## 참고 경로 (실제 확인 완료)

### llama.cpp
- event capture: `ggml/src/ggml-opencl/ggml-opencl.cpp:294-356, 583-662, 674-685, 3015-3019`
- CMake option: `ggml/CMakeLists.txt:252`, `ggml/src/ggml-opencl/CMakeLists.txt:12-15`
- tensor naming: `src/models/qwen2.cpp:28-102`, `src/llama-graph.cpp:995, 1028-1144, 1849`

### llm_rs2
- Queue: `engine/src/backend/opencl/mod.rs:195, 367-372, 2383-2386`
- enqueue_kernel 33지점 예: `mod.rs:1089-1098`
- profile 매크로: `engine/src/layers/transformer_layer/forward_gen.rs:24-51`
- OpProfiler: `engine/src/profile/ops.rs:4-43`
- CLI: `engine/src/bin/generate.rs:119-135, 2210-2220`

### ocl crate
- enqueue_kernel: `ocl-core-0.11.5/src/functions.rs:4022-4031`
- ProfilingInfo: `ocl-core-0.11.5/src/types/enums.rs:1849-1886`
