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
- **Notes**: 커밋 7ada247. `--threads` CLI 옵션 추가. 기본값: `available_parallelism()` 자동 감지.

## [P0] x86 attention AVX2 SIMD 구현
- **Status**: DONE
- **Sprint**: current
- **Notes**: 커밋 7ada247. `dot_f32_avx2()` + `weighted_accum_f32_avx2()`.

## [P0] NEON F16 multi-row GEMV
- **Status**: DONE
- **Sprint**: current
- **Notes**: 커밋 b25bc19. `vec_dot_f16_f32_4rows()` 16-element stride + prefetch. 단일 스레드 성능은 llama.cpp와 동일 (6.2 vs 6.27 tok/s). 멀티스레드 스케일링이 병목.

---

# F16 성능 개선 Sprint: llama.cpp 동급 달성

> **목표**: llama.cpp F16 24.9 tok/s 동급 달성 (현재 15.5 tok/s)
> **핵심 병목**: 멀티스레드 스케일링 (1.76x@4T vs llama.cpp 4.06x@4T)
> **근본 원인**: Rayon per-matmul fork-join 오버헤드 (~300µs × 112 calls = 33ms/token)
> **측정 디바이스**: Galaxy S24 (Snapdragon 8 Gen 3), Llama 3.2 1B F16
> **llama.cpp 소스**: `/home/go/Workspace/llama.cpp/`

---

## 근거 데이터 (2026-03-16 측정)

### 스레드 스케일링 비교

| Threads | llama.cpp tok/s | llm.rs2 tok/s | llama.cpp 스케일링 | llm.rs2 스케일링 |
|---------|-----------------|---------------|-------------------|-----------------|
| 1 | 6.27 | 6.2 | 1.00x | 1.00x |
| 4 | **25.45** | 10.9 | **4.06x** | 1.76x |
| 7 | 24.41 | 15.9 | 3.89x | 2.57x |
| 8 | 24.58 | 12.6 | 3.92x | 2.03x |

**핵심**: 단일 스레드 동일 → NEON inner loop 최적화 불필요. 문제는 100% threading.

### 대역폭 분석

- 단일 코어 대역폭: ~11.5 GB/s (동일)
- llama.cpp 4T: 25.45 tok/s = 46.3 GB/s aggregate (4코어로 대역폭 포화)
- llm.rs2 7T: 15.9 tok/s = 29.5 GB/s aggregate (7코어인데도 대역폭 미포화)
- 갭 원인: 코어 추가가 대역폭 증가로 이어지지 않음 → 스레드 오버헤드

### 프로파일 분석

Per-token 시간 분해 (F16, 7T):
- 프로파일 추적됨: 43.6 ms (matmul 92%, attention 3%, kv_update 4%)
- 미추적 시간: ~15 ms (lm_head matmul + Rayon overhead)
- 112 matmul calls × ~300µs Rayon overhead = ~33 ms 추정

---

## 가설 및 검증 계획

### [H1] Rayon per-matmul fork-join 오버헤드 ← 최우선 가설
- **확신도**: 95%
- **근거**: 1T 동일, 4T에서 2.3x 격차. llama.cpp는 persistent spin-wait threadpool.
- **예상 영향**: 30-40 ms/token 절약 → 15.5 → 24+ tok/s
- **검증**: H1-1 ~ H1-4 참조

### [H2] lm_head matmul 미프로파일링
- **확신도**: 99%
- **근거**: 128256×2048 F16 = 501 MB, 미추적 15ms와 일치
- **영향**: 성능 자체 영향 없음 (lm_head은 어쨌든 실행됨). 프로파일 정확도 개선.
- **검증**: H2-1 참조

### [H3] 이종 코어(big.LITTLE) 로드 밸런싱
- **확신도**: 70%
- **근거**: 8T(12.6) < 7T(15.9). little core가 static partitioning에서 병목.
- **영향**: llama.cpp도 8T에서 7T와 동일 (24.4 vs 24.6) → big.LITTLE 자체는 공통 한계
- **검증**: H1 해결로 함께 해결 가능 (work-stealing → little core에 적게 할당)

### [H4] OpenCL F16 커널 미최적화
- **확신도**: 99%
- **근거**: 1.9 tok/s. 범용 GEMM 템플릿, Adreno 서브그룹 미사용.
- **영향**: OpenCL F16에만 해당 (CPU F16과 독립)
- **검증**: H4-1 ~ H4-2 참조

---

## Iteration 1: Rayon → Spin-wait Threadpool [H1 검증]

### [P0] H1-1. Rayon 오버헤드 정량 측정
- **Status**: TODO
- **Sprint**: current
- **Description**: matmul_transposed 함수 진입/퇴장 시간을 Instant::now()로 측정.
  matmul 내부 순수 연산 시간과 함수 전체 시간의 차이 = Rayon 오버헤드.
- **측정 방법**:
  1. matmul_transposed_f16 시작에 `t_start = Instant::now()`
  2. par_chunks_mut 직전에 `t_par_start`
  3. par_chunks_mut 직후에 `t_par_end`
  4. 함수 끝에 `t_end`
  5. `t_par_end - t_par_start` = Rayon 실행 시간
  6. `(t_end - t_start) - (t_par_end - t_par_start)` = setup 오버헤드
  7. 집계: 총 호출 수, 평균/최대 Rayon 시간, 평균/최대 setup 시간
- **Acceptance Criteria**: per-call 오버헤드 숫자 (예: 300µs), 총 오버헤드 (예: 33ms/token)
- **비고**: 이 측정 자체가 오버헤드를 추가하므로 delta 비교로 평가

### [P0] H1-2. Spin-wait ThreadPool 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: H1-1으로 오버헤드 확인 후
- **Description**: llama.cpp 방식의 persistent spin-wait threadpool 구현.
  Rayon 대체가 아닌 F16 matmul 전용 (기존 Q4/다른 연산은 Rayon 유지).
- **설계**:
  ```
  SpinPool {
    workers: Vec<JoinHandle>,
    shared: Arc<SharedState>,
  }
  SharedState {
    generation: AtomicU64,     // 작업 세대 (workers가 spin-wait)
    work_fn: AtomicPtr<()>,     // 작업 함수 포인터
    work_data: AtomicPtr<()>,   // 작업 데이터 포인터
    next_chunk: AtomicUsize,    // work-stealing용 원자 카운터
    total_chunks: AtomicUsize,  // 총 청크 수
    done_count: AtomicUsize,    // 완료 카운터
  }
  ```
  Worker 루프:
  ```
  loop {
    spin_wait(generation 변경)
    loop {
      chunk_id = next_chunk.fetch_add(1)
      if chunk_id >= total_chunks: break
      execute work_fn(work_data, chunk_id)
    }
    done_count.fetch_add(1)
  }
  ```
  Main thread:
  ```
  fn dispatch(work_fn, data, n_chunks):
    set work_fn, work_data, total_chunks
    next_chunk = 0, done_count = 0
    generation += 1  // workers 깨우기
    spin_wait(done_count == n_workers)
  ```
- **구현 위치**: `engine/src/core/thread_pool.rs` (신규)
- **Acceptance Criteria**: per-matmul 오버헤드 < 10µs, 112 calls < 1.2ms/token
- **핵심**: work-stealing (fetch_add) → big.LITTLE 자연 밸런싱

### [P0] H1-3. F16 matmul을 SpinPool로 전환
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: H1-2 완료 후
- **Description**: `matmul_transposed_f16`에서 `par_chunks_mut` → `spin_pool.dispatch()` 교체.
  SpinPool은 CpuBackendNeon에 필드로 저장 (lazy init).
- **Acceptance Criteria**: CPU F16 tok/s > 20 (4T 기준 비교)
- **Notes**: Q4 matmul은 일단 Rayon 유지 (이미 30.9 tok/s)

### [P0] H1-4. 디바이스 벤치마크 & 스레드 스케일링 재측정
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: H1-3 완료 후
- **Description**: 1T/4T/7T/8T 스케일링 재측정. llama.cpp 대비 갭 확인.
- **목표**: 4T에서 24+ tok/s (llama.cpp 25.45 수준)
- **다음 단계**: 갭 잔존 시 H1-5 프로파일 분석

### [P1] H1-5. Q4 matmul도 SpinPool로 전환
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: H1-4에서 SpinPool 효과 확인 후
- **Description**: Q4 matmul도 SpinPool 적용하여 추가 개선.
- **예상 효과**: Q4 30.9 → 35+ tok/s (Rayon 오버헤드 제거)

---

## Iteration 2: 프로파일 개선 [H2 검증]

### [P1] H2-1. lm_head matmul 프로파일 추가
- **Status**: TODO
- **Sprint**: current
- **Description**: `forward_into()`의 lm_head matmul에 프로파일 계측 추가.
  matmul_lm_head 항목으로 별도 표시.
- **위치**: `engine/src/models/llama/llama_model.rs:532`
- **Acceptance Criteria**: 프로파일에서 lm_head 시간 표시, 미추적 시간 < 5%

---

## Iteration 3: OpenCL F16 커널 [H4 검증]

### [P0] H4-1. Adreno F16 GEMV 커널 작성
- **Status**: TODO
- **Sprint**: next
- **Description**: `mul_mv_f16_f32.cl` 신규 작성. 기존 `mul_mv_q4_0_f32.cl` 패턴 기반.
- **현재**: `mul_mat_f16_f32.cl` (범용 tiled GEMM, 1.9 tok/s)
- **목표**: Adreno 830 특화 GEMV
- **핵심 최적화**:
  - 1D workgroup + 서브그룹 dispatch (64-thread wavefront)
  - `read_imageh()` 또는 `vload_half` 활용 (Adreno 텍스처 캐시)
  - 배리어 0개, 인라인 F16→F32 변환
  - `half4`/`half8` 벡터 타입 활용
  - N_DST=4 rows/subgroup (Q4 커널과 동일 패턴)
- **참고**:
  - llama.cpp `gemv_noshuffle.cl` (Adreno 특화)
  - 우리 `mul_mv_q4_0_f32.cl` (Adreno 특화, 21.8 tok/s 달성)
- **산출물**: `engine/kernels/mul_mv_f16_f32.cl`
- **Acceptance Criteria**: 디바이스에서 15+ tok/s

### [P0] H4-2. OpenCL F16 dispatch 경로 연결
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: H4-1 완료 후
- **Description**: `OpenCLBackend::matmul_transposed()`에서 F16 weight 감지 시
  `mul_mv_f16_f32.cl` 커널 dispatch.
- **위치**: `engine/src/backend/opencl/mod.rs`
- **Acceptance Criteria**: `--backend opencl --weight-dtype f16`에서 새 커널 사용

### [P1] H4-3. 커널 파라미터 튜닝
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: H4-2 완료 후
- **Description**: workgroup size, N_DST, 벡터 너비, 캐시 사용량 튜닝.
- **목표**: 25+ tok/s (llama.cpp GPU 22.7 수준 이상)

---

## Iteration 4: Prefill 최적화

### [P1] 4-1. Prefill TTFT 개선 (942ms → 150ms)
- **Status**: TODO
- **Sprint**: backlog
- **Description**: batch matmul (M>1) 타일링 + SpinPool 적용
- **현재**: TTFT 942ms (F16), llama.cpp 115ms
- **Notes**: Iter 1,2 완료 후 진행

---

## 측정 기록

| 날짜 | 변경 | CPU F16 tok/s | CPU Q4 tok/s | OCL F16 tok/s | 비고 |
|------|------|-------------|-------------|--------------|------|
| 03-13 | baseline | 15.2 | 17.5 | 1.7 | llama.cpp CPU=24.4 |
| 03-16 | NEON SIMD | 15.1 | 30.5 | 1.9 | Q4 목표 달성 |
| 03-16 | multi-row GEMV | 15.5 | 30.9 | - | inner loop 최적화 한계 확인 |
| 03-16 | thread scaling | 15.9(7T) | 30.9 | - | **1T 동일, 멀티스레드 병목 확인** |
