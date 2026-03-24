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

---

# generate.rs eval 루프 리팩토링 (StepHook 추상화)

> **목표**: generate.rs의 eval 루프 중복 코드를 StepHook 추상화로 제거하고, eval/kivi/ppl 3개 모드를 단일 generic 루프로 통합
> **설계 문서**: `docs/38_eval_refactoring.md`
> **구현 순서**: 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 (순차 진행 권장)

---

## [P1] EVAL-1. eval 모듈 골격 생성
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: `engine/src/eval/` 디렉토리 신규 생성. 다음 파일 및 타입 정의.
  - `mod.rs`: 모듈 선언, pub re-export
  - `hook.rs`: `StepHook` trait, `CacheSnapshot` trait, `PostStepResult` 타입
  - `output.rs`: `MetricsSummary`, `EvalOutput`, `EvalConfig` 타입
  - `engine/src/lib.rs`에 `pub mod eval` 추가
- **Acceptance Criteria**:
  - `cargo check` 통과
  - 각 trait/타입에 doc comment 작성
  - `StepHook` trait: `pre_step()`, `post_decode_step()`, `summarize()` 메서드 포함

## [P1] EVAL-2. qcf_helpers.rs 추출
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: EVAL-1 완료 후
- **Description**: `generate.rs`에 산재한 QCF 집계 중복 코드를 `engine/src/eval/qcf_helpers.rs`로 이동.
  이동 대상 함수: `aggregate_metrics`, `build_opr_fields`, `create_qcf_metric_json`
- **Acceptance Criteria**:
  - `generate.rs`에서 해당 함수 직접 정의 제거
  - `qcf_helpers` 모듈에서 import하여 사용
  - 기존 동작 변경 없음 (출력 동일)
  - `cargo test` 통과

## [P1] EVAL-3. EvictionHook 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: EVAL-2 완료 후
- **Description**: `engine/src/eval/eviction_hook.rs` 신규 작성.
  - `CacheManager` + `score_accumulator` 소유
  - `post_decode_step()`에서: budget 검사 → eviction 수행 → attn/caote QCF 수집
  - `KVCacheSnapshot` 구현 (byte copy 방식)
  - `StepHook` trait 구현
- **Acceptance Criteria**:
  - 기존 run_eval_ll 경로와 동일한 eviction 타이밍 및 QCF 값 출력
  - unit test: eviction 발생 여부, QCF 집계 값 검증

## [P1] EVAL-4. KiviHook 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: EVAL-2 완료 후
- **Description**: `engine/src/eval/kivi_hook.rs` 신규 작성.
  - `take_flush_proxies()`로 NMSE + OPR 수집
  - `KiviCacheSnapshot` 구현 (Vec clone 방식)
  - `StepHook` trait 구현
- **Acceptance Criteria**:
  - 기존 run_kivi_eval_ll 경로와 동일한 NMSE/OPR 값 출력
  - unit test: flush 시점, metric 수집 검증

## [P1] EVAL-5. eval_loop.rs 작성
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: EVAL-3, EVAL-4 완료 후
- **Description**: `engine/src/eval/eval_loop.rs` 신규 작성.
  `run_eval_ll_generic<C: KVCacheOps>` 함수 구현.
  - question-choice 루프 통합
  - importance 2-pass 지원
  - chunked prefill 지원
  - 훅 주입(dependency injection)으로 eviction/kivi 분기
- **Acceptance Criteria**:
  - EvictionHook, KiviHook 모두 generic 루프에서 동작 확인
  - 기존 출력과 동일한 JSON 생성

## [P1] EVAL-6. generate.rs 전환
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: EVAL-5 완료 후
- **Description**: `generate.rs`의 `run_eval_ll`, `run_kivi_eval_ll` 함수를 `eval_loop` 호출로 교체.
  교체 완료 후 기존 함수 본체 삭제 (래퍼만 유지하거나 완전 제거).
- **Acceptance Criteria**:
  - generate.rs LOC 순감소
  - eval-ll, kivi 모드 CLI 인터페이스 변경 없음
  - `cargo test` 통과

## [P1] EVAL-7. run_ppl qcf_helpers 전환
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: EVAL-2 완료 후 (EVAL-6과 병렬 가능)
- **Description**: `run_ppl` 함수 내 QCF 집계/OPR 코드를 `qcf_helpers` 모듈 호출로 교체.
  중복 인라인 집계 로직 제거.
- **Acceptance Criteria**:
  - PPL 모드 출력 변경 없음
  - `cargo test` 통과

## [P1] EVAL-8. JSON regression test
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: EVAL-6, EVAL-7 완료 후
- **Description**: 리팩토링 전후 출력 동일성 검증을 위한 golden test 작성.
  대상 모드 4가지: eval-ll, kivi, skip-layers, ppl.
  기존 출력 JSON을 golden file로 저장하고 테스트에서 비교.
- **Acceptance Criteria**:
  - 4개 모드 golden test 모두 통과
  - `cargo test` 통과
  - 테스트 위치: `engine/tests/eval_regression.rs` 또는 각 모듈 내 `#[cfg(test)]`

## [P1] EVAL-9. sanity-check 전체 통과
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: EVAL-8 완료 후
- **Description**: 리팩토링 완료 후 전체 품질 게이트 통과 확인.
  `.agent/skills/developing/scripts/sanity_check.sh` 실행.
  - `cargo fmt` — 포맷 경고 없음
  - `cargo clippy` — 경고 없음
  - `cargo test` — 전체 통과
- **Acceptance Criteria**:
  - sanity_check.sh 스크립트 exit 0
  - 새로 추가된 코드에 clippy allow 어트리뷰트 없음

## [P1] EVAL-10. ARCHITECTURE.md 업데이트
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: EVAL-6 완료 후
- **Description**: ARCHITECTURE.md에 `eval/` 모듈 구조 반영.
  - 디렉토리 트리에 `engine/src/eval/` 추가
  - StepHook 기반 eval 루프 구조 설명 추가
  - generate.rs 설명을 monolithic → hook dispatch 구조로 업데이트
  - 설계 상세는 `docs/38_eval_refactoring.md` 참조로 연결
- **Acceptance Criteria**:
  - ARCHITECTURE.md의 디렉토리 트리가 실제 코드와 일치
  - eval 모듈의 역할과 흐름이 설명됨

---

# Gemma 3 1B 지원

> **목표**: Gemma 3 1B (google/gemma-3-1b-pt) 아키텍처 지원 추가
> **설계 문서**: `docs/40_gemma3_support.md`
> **모델 경로**: `models/gemma3-1b/` (호스트), `/data/local/tmp/models/gemma3-1b` (디바이스)
> **구현 순서**: Phase 1 (1.1→1.6→1.2~1.5→1.7~1.10→1.11→1.12) → Phase 2 (2.1~2.4) → Phase 3

---

## Phase 1: 최소 동작 (CPU, full-context attention)

### [P1] GEMMA-1.1. ModelArch::Gemma3 및 ModelConfig 확장
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: `engine/src/models/config.rs` 수정.
  - `ModelArch` enum에 `Gemma3` 추가
  - `RawHfConfig`에 `rope_local_base_freq`, `sliding_window`, `sliding_window_pattern`, `query_pre_attn_scalar`, `hidden_activation` 필드 추가
  - `ModelConfig`에 `rope_local_theta`, `sliding_window`, `sliding_window_pattern`, `query_pre_attn_scalar`, `embed_scale` 필드 추가 (모두 Option)
  - `detect_arch()`에 `"Gemma3ForCausalLM"`, `"gemma3_text"` 케이스 추가
  - `from_json()`에서 Gemma3 분기: `embed_scale = Some(sqrt(hidden_size))`
- **Acceptance Criteria**:
  - `cargo check` 통과
  - `test_parse_gemma3_config` 단위 테스트 통과
  - Llama/Qwen2 config 파싱에 영향 없음

### [P1] GEMMA-1.2. Backend rms_norm 시그니처 변경 (add_unit: bool)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (1.1과 병렬 가능)
- **Description**: `Backend` trait의 `rms_norm()`, `rms_norm_oop()`, `add_rms_norm_oop()` 시그니처에 `add_unit: bool` 파라미터 추가.
  - trait 정의 변경 (`core/backend.rs`)
  - CPU 구현체 변경 (`backend/cpu/common.rs`, `neon.rs`, `x86.rs`)
  - OpenCL 구현체 변경 (`backend/opencl/mod.rs`)
  - 기존 호출부 ~20개에 `false` 추가 (forward.rs, forward_gen.rs, transformer.rs, test_backend.rs, 테스트)
  - `add_unit == true` 구현: weight 적용 시 `(1.0 + wi)` 사용
- **Acceptance Criteria**:
  - `cargo test` 전체 통과 (기존 경로 회귀 없음)
  - `test_rms_norm_add_unit` 단위 테스트 통과 (add_unit=true 수치 검증)

### [P1] GEMMA-1.3. Backend gelu_tanh_mul() 추가
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (1.2와 병렬 가능)
- **Description**: `Backend` trait에 `gelu_tanh_mul()` default 구현 추가.
  - 공식: `0.5 * x * (1 + tanh(sqrt(2/π) * (x + 0.044715 * x³))) * up`
  - default 구현은 scalar 루프 (Phase 1)
- **Acceptance Criteria**:
  - `test_gelu_tanh_known_values` 단위 테스트 통과 (gelu(0)=0, gelu(1)≈0.8413)

### [P1] GEMMA-1.4. Gemma3Mapper 및 LayerWeightNames 확장
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: GEMMA-1.1 완료 후
- **Description**:
  - `models/mappers/mod.rs`: `LayerWeightNames`에 `pre_ffn_norm`, `post_ffn_norm`, `q_norm`, `k_norm` 4개 `Option<String>` 필드 추가. Llama/Qwen2 mapper에서 `None` 반환.
  - `models/mappers/gemma3.rs` 신규: `Gemma3Mapper` 구현. 13개 텐서 이름 패턴.
    - `ffn_norm` → `post_attention_layernorm` (Gemma3에서 post-attn norm 역할)
    - `pre_ffn_norm` → `pre_feedforward_layernorm`
    - `post_ffn_norm` → `post_feedforward_layernorm`
    - `q_norm` → `self_attn.q_norm.weight`
    - `k_norm` → `self_attn.k_norm.weight`
  - `create_mapper()`에 `ModelArch::Gemma3` 분기 추가
- **Acceptance Criteria**:
  - `test_gemma3_mapper_names` 단위 테스트 통과
  - Llama/Qwen2 mapper 테스트 변경 없음

### [P1] GEMMA-1.5. TransformerLayer 구조체 확장
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: GEMMA-1.4 완료 후
- **Description**: `layers/transformer_layer/mod.rs` 수정.
  - `TransformerLayer`에 `q_norm`, `k_norm`, `pre_ffn_norm`, `post_ffn_norm` (모두 `Option<Tensor>`) 추가
  - `LayerForwardArgs`에 `rms_norm_add_unit: bool`, `use_gelu_tanh: bool`, `is_local_attn: Option<bool>`, `local_attn_window: Option<usize>` 추가
  - `ForwardGenArgs`에도 동일 필드 추가
  - 기존 호출부에서 새 필드에 기본값 전달 (false/None)
- **Acceptance Criteria**:
  - `cargo check` 통과
  - 기존 Llama/Qwen2 forward 경로 변경 없음

### [P1] GEMMA-1.6. TransformerModel 로딩 확장
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: GEMMA-1.4, GEMMA-1.5 완료 후
- **Description**: `models/transformer.rs` 의 `load_with_dtype()` 수정.
  - `mapper.weight_names(i)`의 Optional 필드가 Some이면 해당 텐서 로딩 (q_norm, k_norm, pre_ffn_norm, post_ffn_norm)
  - `TransformerLayer` 생성 시 Optional 필드 채움
  - norm 텐서는 `is_weight: false` (F32 변환)
- **Acceptance Criteria**:
  - Gemma 3 1B safetensors 로드 시 26개 레이어 × 4 추가 텐서 정상 로딩
  - Llama/Qwen2 로딩 경로 영향 없음

### [P1] GEMMA-1.7. forward_prefill() Gemma3 분기
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: GEMMA-1.2, GEMMA-1.3, GEMMA-1.5 완료 후
- **Description**: `layers/transformer_layer/forward.rs` 수정.
  - `rms_norm()` 호출 시 `args.rms_norm_add_unit` 전달
  - QKV projection 후 `if let Some(ref qn) = self.q_norm` → head 단위 rms_norm(add_unit=true) 적용 (RoPE 전)
  - post-attn norm: `if args.rms_norm_add_unit` → `ffn_norm`을 post-attn norm으로 사용, residual add 전에 적용
  - pre-ffn norm: `if let Some(ref pfn) = self.pre_ffn_norm` → pre-FFN norm 적용
  - FFN activation: `if args.use_gelu_tanh` → `gelu_tanh_mul()` 사용, else → `silu_mul()`
  - post-ffn norm: `if let Some(ref pfn) = self.post_ffn_norm` → post-FFN norm 적용
- **Acceptance Criteria**:
  - `cargo check` 통과
  - Llama/Qwen2 forward_prefill 경로 영향 없음 (새 필드가 false/None)

### [P1] GEMMA-1.8. forward_gen() Gemma3 분기
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: GEMMA-1.7과 동일
- **Description**: `layers/transformer_layer/forward_gen.rs` 수정.
  - GEMMA-1.7과 동일한 분기를 decode (seq_len==1) 경로에 적용
  - `add_rms_norm_oop()` 호출 시 `add_unit` 전달
- **Acceptance Criteria**:
  - `cargo check` 통과
  - Llama/Qwen2 forward_gen 경로 영향 없음

### [P1] GEMMA-1.9. forward_into() Gemma3 분기
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: GEMMA-1.6, GEMMA-1.7, GEMMA-1.8 완료 후
- **Description**: `models/transformer.rs`의 `forward_into()` 수정.
  - embed gather 후 `if let Some(scale) = self.config.embed_scale` → `backend.scale(&mut x, scale)`
  - 레이어 루프에서: `is_local_layer(i, config.sliding_window_pattern)` 판별
  - `rope_theta`: 로컬 → `config.rope_local_theta`, 글로벌 → `config.rope_theta`
  - `LayerForwardArgs`에 `rms_norm_add_unit`, `use_gelu_tanh`, `is_local_attn`, `local_attn_window` 전달
  - final norm: `rms_norm(&mut x, &self.norm, eps, is_gemma3)` (add_unit)
- **Acceptance Criteria**:
  - `cargo check` 통과
  - Llama/Qwen2 forward_into 경로 영향 없음

### [P1] GEMMA-1.10. 모델 다운로드 및 호스트 동작 확인
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: GEMMA-1.9 완료 후
- **Description**:
  - `huggingface-cli download google/gemma-3-1b-pt --local-dir models/gemma3-1b`
  - `cargo run --release --bin generate -- --model-path models/gemma3-1b --prompt "Hello" -n 64`
  - 출력이 coherent한 영어 텍스트인지 확인
  - `cargo test` 전체 통과 확인
- **Acceptance Criteria**:
  - 모델 로딩 성공, 텍스트 생성 가능
  - 출력 품질이 명백히 nonsense가 아닌 것을 확인

### [P1] GEMMA-1.11. sanity-check 통과
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: GEMMA-1.10 완료 후
- **Description**: `.agent/skills/developing/scripts/sanity_check.sh` 실행.
  - `cargo fmt` 경고 없음
  - `cargo clippy` 경고 없음
  - `cargo test` 전체 통과
- **Acceptance Criteria**: sanity_check.sh exit 0

### [P1] GEMMA-1.12. 온디바이스 동작 확인
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: GEMMA-1.11 완료 후
- **Description**:
  - Android 크로스 빌드: `source android.source && cargo build --target aarch64-linux-android --release`
  - 모델/바이너리 디바이스 전송
  - CPU 추론: `./generate --model-path models/gemma3-1b --prompt "Hello" -n 64 -b cpu`
  - OpenCL 추론: `./generate --model-path models/gemma3-1b --prompt "Hello" -n 64 -b opencl`
- **Acceptance Criteria**:
  - CPU 및 OpenCL 백엔드에서 텍스트 생성 성공
  - 성능 수치 기록 (tok/s)

---

## Phase 2: Sliding Window Attention

### [P2] GEMMA-2.1. flash_attention에 window_size 파라미터 추가
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Phase 1 완료 후
- **Description**: `layers/attention.rs`의 `flash_attention_forward_strided()`에 `window_size: Option<usize>` 파라미터 추가.
  - causal mask 생성 시 window 바깥 토큰을 `-inf`로 설정
  - `None`이면 기존 full-context 동작
- **Acceptance Criteria**:
  - 기존 Llama/Qwen2 flash attention 경로 변경 없음 (window_size=None)
  - window_size 적용 시 지정 범위만 attend

### [P2] GEMMA-2.2. prefill 로컬 레이어 window_size 전달
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: GEMMA-2.1 완료 후
- **Description**: `forward.rs`에서 로컬 레이어일 때 `flash_attention_forward_strided()`에 `window_size = args.local_attn_window` 전달.
- **Acceptance Criteria**: 로컬 레이어 prefill에서 window mask 적용 확인

### [P2] GEMMA-2.3. decode 로컬 레이어 effective_cache_len 계산
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: GEMMA-2.1 완료 후
- **Description**: `forward_gen.rs`에서 로컬 레이어일 때 `effective_cache_len = min(cache_seq_len, window_size)`.
  - KV cache offset 계산: `kv_start_pos = cache_seq_len - effective_cache_len`
- **Acceptance Criteria**: 로컬 레이어 decode에서 최근 window_size 토큰에만 attend

### [P2] GEMMA-2.4. SW attention 정확도 검증
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: GEMMA-2.2, GEMMA-2.3 완료 후
- **Description**: Phase 1 (full context) vs Phase 2 (SW) 출력 비교.
  - 짧은 프롬프트 (< 512 tok): 동일 출력 기대 (window 내)
  - 긴 프롬프트 (> 512 tok): 차이 발생 확인
- **Acceptance Criteria**: 짧은 프롬프트에서 동일 출력, 긴 프롬프트에서 품질 유지

---

## Phase 3: 최적화 (선택적)

### [P3] GEMMA-3.1. rms_norm(add_unit=true) NEON 가속
- **Status**: TODO
- **Sprint**: backlog
- **Description**: NEON SIMD 경로에서 add_unit=true 분기 최적화

### [P3] GEMMA-3.2. gelu_tanh_mul() NEON 가속
- **Status**: TODO
- **Sprint**: backlog
- **Description**: GELU_tanh의 tanh 근사 NEON 벡터화

### [P3] GEMMA-3.3. OpenCL RMSNorm add_unit 커널
- **Status**: TODO
- **Sprint**: backlog
- **Description**: 별도 `.cl` 파일로 add_unit RMSNorm 커널 추가
