# Rust Developer TODO

> **역할**: 백엔드 구현, 모델 추론 로직, 성능 최적화, 버그 수정
> **소유 영역**: `engine/src/`, `manager/`, `shared/`

---

## [P0] .cargo/config.toml 타겟 트리플 수정
- **Status**: DONE
- **Sprint**: current
- **Notes**: 커밋 7ada247. `x86_64-linux` → `x86_64-unknown-linux-gnu`. AVX2/FMA 컴파일러 플래그 호스트 빌드에 적용됨.

## [P0] Rayon 스레드 수 자동 감지
- **Status**: DONE
- **Sprint**: current
- **Notes**: 커밋 7ada247. `--threads` CLI 옵션 추가. 기본값: `available_parallelism()` 자동 감지. 8→20 스레드로 +24.5% (37.5→46.7 tok/s).

## [P0] x86 attention AVX2 SIMD 구현
- **Status**: DONE
- **Sprint**: current
- **Notes**: 커밋 7ada247. `dot_f32_avx2()` + `weighted_accum_f32_avx2()` 4x 언롤 FMA. parallel/serial 양쪽 경로 적용. 365 tests passed.

## [P0] 기준선 측정 및 수정 전후 비교
- **Status**: DONE
- **Sprint**: current
- **Notes**: 측정 완료. F32 KV: 51.5 tok/s, Q4 KV: 51.2 tok/s (20 threads). 긴 프롬프트(141 tokens) TTFT 3.4s, decode 49.3 tok/s. 이론 피크 ~120 tok/s 대비 43% 달성. 스레드 스케일링: 4t=20.9, 8t=37.5, 12t=42.4, 16t=45.6, 20t=46.7 tok/s (diminishing returns → memory bandwidth bound).

## [P1] forward_gen 연산별 내부 계측
- **Status**: DONE
- **Sprint**: current
- **Notes**: 커밋 9fb86b6. `--profile` CLI 플래그 추가. OpProfiler로 11개 연산 구간별 Instant 타이밍. 결과: matmul_ffn(67%), matmul_qkv(17%), matmul_wo(8%), attention(5.6%), silu_mul(1.4%), rope(0.5%), rms_norm(0.3%), kv_update(0.4%). matmul 합계 91.7% → memory bandwidth bound 확인.

## [P1] 외부 프로파일링 (flamegraph + perf stat)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: P1 내부 계측 완료 ✅
- **Description**: `cargo flamegraph`로 CPU 시간 분포 시각화, `perf stat`으로 IPC/캐시 미스율 측정. prefill-only vs decode-only 분리 프로파일링. Rayon 오버헤드 확인.
- **Acceptance Criteria**: Flamegraph SVG, perf stat 카운터 테이블, 병목 분류(compute/memory/overhead)

## [P1] micro_bench 확장
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: P1 내부 계측 완료 ✅
- **Description**: 현재 `quantize_row_q8_0`, `vec_dot_q4_0_q8_0`만 벤치마크. 추가 대상: matmul 차원별(N=2048/8192/128256), attention seq_len별(64~2048), rms_norm/rope/silu_mul 개별, Rayon serial vs parallel 비교.
- **Acceptance Criteria**: 연산별 throughput 테이블, Rayon 오버헤드 정량화
- **Notes**: `engine/src/bin/micro_bench.rs`

## [P1] 이론적 피크 대비 달성률 분석
- **Status**: DONE
- **Sprint**: current
- **Notes**: Llama 3.2 1B Q4_0 디코드 기준 이론 피크 ~120 tok/s (DDR5 80GB/s). 실측 51.5 tok/s = 43% 달성. 스레드 12 이상에서 수확체감 → memory bandwidth bound 확인.

## [P2] 소규모 연산 Rayon 오버헤드 제거
- **Status**: DONE
- **Sprint**: current
- **Notes**: 커밋 9fb86b6. PARALLEL_THRESHOLD=16384 (64KB ≈ L1 cache size). add_assign, scale, silu_mul에서 threshold 미만 serial 실행. 프로파일 결과 add_assign 0.0%, silu_mul 1.4% (전체의 미미한 비중이므로 실측 성능 차이 미미).

## [P2] RoPE 주파수 테이블 사전 계산
- **Status**: DONE
- **Sprint**: current
- **Notes**: 커밋 9fb86b6. `powf()` 호출을 inner loop 밖으로 이동하여 freqs Vec 사전 계산. 프로파일 결과 rope 0.5% (전체의 미미한 비중).

## [P2] Q4_0 matmul M≥4 AVX2 구현 (프리필 최적화)
- **Status**: DONE
- **Sprint**: current
- **Notes**: 커밋 9fb86b6. M≥4에서 CpuBackendCommon 스칼라 폴백 → AVX2 quantize_row_q8_0 + vec_dot_q4_0_q8_0 사용. 병렬 A행 양자화 + 행별 병렬 dot product. 프리필 경로 AVX2 커버리지 100%.

---

# 성능 개선 Sprint: llama.cpp 동급 달성

> **목표**: llama.cpp 24.4 tok/s 동급 이상 달성 (현재 CPU 17.5, OpenCL Q4 30.1)
> **방법론**: 분석 → 구현 → 측정 루프 (점진적 개선)
> **llama.cpp 소스**: `/home/go/Workspace/llama.cpp/` (shallow clone)
> **기준 벤치마크**: `docs/31_perf_comparison_llama_cpp.md`
> **측정 조건**: Galaxy S24, "tell me short story", decode 128, 시작 온도 ~40°C

---

## Iteration 1: CPU Q4 Decode (17.5 → 24+ tok/s)

### [P0] 1-1. llama.cpp NEON Q4 dot product 분석 리포트
- **Status**: TODO
- **Sprint**: current
- **Description**: llama.cpp의 `vec_dot_q4_0_q8_0` NEON 구현을 정밀 분석하고 우리 구현과 비교 리포트 작성.
- **분석 대상**:
  - `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` — ARM NEON Q4×Q8 dot product
  - `llama.cpp/ggml/src/ggml-cpu/quants.c` — 공통 양자화 루틴
  - 우리 코드: `engine/src/backend/cpu/neon.rs:625-775` — vec_dot_q4_0_q8_0
- **리포트 내용**: 명령어 수준 비교, 블록 배칭 전략, 수평 합산 방식, 누적기 패턴 차이
- **산출물**: `docs/32_llama_cpp_neon_q4_analysis.md`
- **Acceptance Criteria**: 두 구현의 명령어 수/블록, ILP 패턴, 예상 cycle count 비교 테이블

### [P0] 1-2. llama.cpp 양자화 루틴 분석
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 1-1과 병행 가능
- **Description**: llama.cpp의 `quantize_row_q8_0` NEON 구현과 matmul 내 양자화 타이밍 분석.
- **분석 대상**:
  - `llama.cpp/ggml/src/ggml-cpu/arch/arm/quants.c` — quantize_row_q8_0
  - 우리 코드: `engine/src/backend/cpu/neon.rs:534-621`
- **핵심 질문**: llama.cpp는 fused quantize-dot을 하는가? 양자화 비용이 전체에서 몇 %인가?
- **산출물**: 1-1 리포트에 포함

### [P0] 1-3. NEON dot product 개선 구현 (Iter 1)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 1-1 완료 후
- **Description**: 분석 결과를 기반으로 `vec_dot_q4_0_q8_0` 첫 번째 개선.
- **예상 작업**: 수평 합산 방식 변경, 블록 배칭 증가, 누적기 패턴 적용
- **Acceptance Criteria**: micro_bench에서 기존 대비 개선 확인

### [P0] 1-4. 벤치마크 & 측정 (Iter 1)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 1-3 완료 후
- **Description**: 디바이스에서 decode 128/256 측정, llama.cpp 대비 갭 확인.
- **산출물**: `docs/31_perf_comparison_llama_cpp.md` 업데이트
- **다음 단계**: 갭이 남아있으면 1-3으로 돌아가 추가 최적화

### [P1] 1-5. llama.cpp matmul 병렬화 분석
- **Status**: TODO
- **Sprint**: current
- **Description**: llama.cpp의 matmul 스레딩 전략 분석 (ggml의 task decomposition).
- **분석 대상**:
  - `llama.cpp/ggml/src/ggml-cpu/ggml-cpu.c` or `ops.cpp` — compute_forward_mul_mat
  - 스레드 분할 방식 (N-parallel? M-parallel?)
  - 우리 Rayon 병렬화와 비교
- **핵심 질문**: decode M=1에서 llama.cpp는 어떻게 병렬화하는가?
- **산출물**: 1-1 리포트에 섹션 추가

### [P1] 1-6. 병렬화 개선 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 1-5 완료 후
- **Description**: 분석 결과 기반 Rayon 오버헤드 감소 또는 직접 스레딩.

---

## Iteration 2: OpenCL F16 Matmul (1.7 → 25+ tok/s)

### [P0] 2-1. llama.cpp OpenCL 커널 분석 리포트
- **Status**: TODO
- **Sprint**: next
- **Description**: llama.cpp의 OpenCL matmul 커널(특히 Adreno 최적화)을 분석.
- **분석 대상**:
  - `llama.cpp/ggml/src/ggml-opencl/kernels/mul_mat_f16_f32.cl`
  - `llama.cpp/ggml/src/ggml-opencl/kernels/gemv_noshuffle.cl` — Adreno 특화
  - `llama.cpp/ggml/src/ggml-opencl/ggml-opencl.cpp` — dispatch 로직
  - 우리 코드: `engine/kernels/mul_mat_f16_f32.cl` (범용, 1.7 tok/s)
  - 우리 코드: `engine/kernels/mul_mv_q4_0_f32.cl` (Adreno 특화, 30.1 tok/s)
- **리포트 내용**: 서브그룹 활용, 메모리 접근 패턴, 타일링 전략, dispatch 비교
- **산출물**: `docs/33_llama_cpp_opencl_kernel_analysis.md`

### [P0] 2-2. Adreno F16 GEMV 커널 작성 (Iter 1)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 2-1 완료 후
- **Description**: 기존 Q4 커널(`mul_mv_q4_0_f32.cl`) + llama.cpp 분석 기반으로 Adreno 특화 F16 GEMV 커널 작성.
- **핵심**: 서브그룹 dispatch, `read_imageh()`, 배리어 제거
- **산출물**: `engine/kernels/mul_mv_f16_f32.cl` (신규)
- **Acceptance Criteria**: 디바이스에서 10+ tok/s 달성

### [P0] 2-3. 벤치마크 & 측정 (Iter 1)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 2-2 완료 후
- **Description**: OpenCL F16 decode 128/256 측정, Q4 대비 갭 확인.
- **다음 단계**: 25+ tok/s 미달 시 2-2로 돌아가 커널 튜닝

### [P1] 2-4. OpenCL F16 GEMM 커널 (Prefill Batch)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 2-2 완료 후
- **Description**: M>1 (prefill) 전용 F16 GEMM 커널. GEMV와 별도 dispatch.

---

## Iteration 3: Prefill 최적화 (36 → 10 ms/tok)

### [P1] 3-1. llama.cpp prefill 경로 분석
- **Status**: TODO
- **Sprint**: backlog
- **Description**: llama.cpp가 prefill에서 6.6 ms/tok 달성하는 방법 분석.
- **분석 대상**:
  - batch matmul (M>1) 타일링 전략
  - KV cache 업데이트 방식
  - 프리필과 디코드의 커널 분리 여부
- **산출물**: 리포트 (docs/)

### [P1] 3-2. CPU batch matmul 타일링 구현
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 3-1, Iter 1 완료 후
- **Description**: L1/L2 캐시에 맞는 타일 블로킹 적용.

### [P1] 3-3. OpenCL prefill matmul 최적화
- **Status**: TODO
- **Sprint**: backlog
- **Dependencies**: 3-1, Iter 2 완료 후
- **Description**: M>1 전용 GEMM dispatch 경로.

---

## 측정 기록

| 날짜 | Iter | 변경 | CPU Q4 tok/s | OCL Q4 tok/s | OCL F16 tok/s | 비고 |
|------|------|------|-------------|-------------|--------------|------|
| 2026-03-13 | baseline | - | 17.5 | 30.1 | 1.7 | llama.cpp CPU=24.4, GPU=22.7 |
