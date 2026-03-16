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

# 성능 개선 Sprint: F16 llama.cpp 동급 달성

> **목표**: llama.cpp F16 24.4 tok/s 동급 이상 달성
> **방법론**: 분석 → 구현 → 측정 루프 (점진적 개선)
> **llama.cpp 소스**: `/home/go/Workspace/llama.cpp/` (shallow clone)
> **기준 벤치마크**: `docs/31_perf_comparison_llama_cpp.md`
> **측정 조건**: Galaxy S24, decode 128, 시작 온도 ~40°C
> **이론 피크 (F16)**: ~27.5 tok/s (LPDDR5X 51 GB/s, weight 1.878 GB/token)

---

## [DONE] Iteration 0: CPU Q4 Decode — 목표 달성

- **결과**: 30.5 tok/s (llama.cpp CPU 24.4 대비 **+25%**)
- **커밋**: 86d4885, c889b58 (NEON/AVX2 SIMD)
- **비고**: Q4는 F16보다 3.2x 적은 bandwidth 사용 → 높은 tok/s 달성

---

## Iteration 1: CPU F16 Decode (15.1 → 24+ tok/s) ← CURRENT

### [P0] 1-1. NEON F16 multi-row GEMV 구현
- **Status**: IN_PROGRESS
- **Sprint**: current
- **Description**: llama.cpp tinyBLAS 분석 기반으로 multi-row GEMV 구현.
  현재 단일 행 `vec_dot_f16_f32` (4 accumulators) → 4-row 동시 처리로 전환.
- **분석 결과**:
  - llama.cpp: tinyBLAS RM=4×RN=6 블록 타일 + work-stealing threadpool
  - 우리: 단일 행 GEMV + Rayon flat 병렬화
  - 핵심 차이: 다중 행 처리로 ILP 향상 + 스레드 스케줄링 오버헤드 감소
- **구현 내용**:
  - `vec_dot_f16_f32_4rows()`: 4 weight rows × 8-element inner loop
  - 8 accumulators (4 rows × 2 each), activation 1회 로드 후 4 rows에 재사용
  - Rayon 스레드당 연속 output row 범위 할당 (chunk dispatch 대신)
- **Acceptance Criteria**: 디바이스 decode 128에서 20+ tok/s

### [P0] 1-2. 벤치마크 & 측정
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 1-1 완료 후
- **Description**: 디바이스에서 CPU F16 decode 128 측정, llama.cpp 대비 갭 확인.
- **다음 단계**: 24+ 미달 시 추가 최적화 (software prefetch, NR=8 등)

### [P1] 1-3. 추가 최적화 (필요 시)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 1-2 결과에 따라
- **Description**: software prefetching (`prfm pldl1strm`), NR=8 multi-row, 32-element stride
- **Target**: 이론 피크의 85%+ (23.4+ tok/s)

---

## Iteration 2: OpenCL F16 Matmul (1.9 → 25+ tok/s)

### [P0] 2-1. Adreno F16 GEMV 커널 작성
- **Status**: TODO
- **Sprint**: next
- **Description**: 기존 Q4 커널(`mul_mv_q4_0_f32.cl`) 패턴 기반 Adreno 특화 F16 GEMV 커널.
  현재 `mul_mat_f16_f32.cl`은 범용 tiled GEMM (1.9 tok/s, 사실상 미작동).
- **핵심**: 서브그룹 dispatch, `read_imageh()`, 배리어 제거, 1D workgroup
- **참고**: llama.cpp `gemv_noshuffle.cl` (Adreno 특화)
- **Acceptance Criteria**: 디바이스에서 10+ tok/s 달성

### [P0] 2-2. 벤치마크 & 커널 튜닝
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 2-1 완료 후
- **다음 단계**: 25+ tok/s 미달 시 커널 파라미터 튜닝

---

## Iteration 3: Prefill 최적화 (TTFT 942ms → 100ms)

### [P1] 3-1. CPU F16 batch matmul 타일링
- **Status**: TODO
- **Sprint**: backlog
- **Description**: L1/L2 캐시에 맞는 타일 블로킹 적용.

### [P1] 3-2. OpenCL prefill matmul 최적화
- **Status**: TODO
- **Sprint**: backlog
- **Description**: M>1 전용 GEMM dispatch 경로.

---

## 측정 기록

| 날짜 | Iter | 변경 | CPU F16 tok/s | CPU Q4 tok/s | OCL Q4 tok/s | OCL F16 tok/s | 비고 |
|------|------|------|-------------|-------------|-------------|--------------|------|
| 2026-03-13 | baseline | - | 15.2 | 17.5 | 30.1 | 1.7 | llama.cpp CPU F16=24.4 |
| 2026-03-16 | re-measure | NEON/AVX2 SIMD 적용 후 | 15.1 | 30.5 | 21.8 | 1.9 | Q4 목표 달성, F16 최적화 필요 |
