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

## [P1] ZramStore Blosc 필터 실험: bytedelta + 압축 코덱 비교
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (기존 ZramStore + preprocess.rs 위에 구현)
- **Description**: |
  현재 ZramStore(byte-shuffle + LZ4)가 실제 F16 KV 캐시에서 압축률 1.0x.
  Blosc 핵심 아이디어(bytedelta, trunc_prec)를 순수 Rust로 구현하고
  실제 모델 데이터에서 압축률과 해제 속도를 실측한다.

  ### Step 1: bytedelta 구현 (`preprocess.rs`)
  - `bytedelta_encode(data, stream_len, n_streams)` — 바이트 스트림 내 delta 인코딩
  - `bytedelta_decode(data, stream_len, n_streams)` — prefix-sum 복원
  - 유닛 테스트: roundtrip bit-exact 검증

  ### Step 2: trunc_prec 구현 (`preprocess.rs`)
  - `trunc_prec_f16(data, zero_bits)` — F16 mantissa 하위 N비트 마스킹 (lossy)
  - 유닛 테스트: 마스킹 정확성 검증

  ### Step 3: ZramStore 압축 파이프라인 확장 (`zram_store.rs`)
  - 기존: shuffle → LZ4
  - 확장: shuffle → bytedelta → LZ4 (또는 Zstd)
  - `ZramConfig` 구조체로 파이프라인 옵션 제어:
    - `use_bytedelta: bool`
    - `trunc_bits: u32` (0 = 무손실)
    - `codec: Codec` (LZ4 / Zstd)
  - Zstd 사용 시 `zstd` 크레이트 추가 (Cargo.toml)

  ### Step 4: 압축률 벤치마크 테스트
  - 실제 모델(Llama 3.2 1B) KV 캐시 데이터로 다음 조합 실측:
    ```
    ① shuffle + LZ4              (기준선, 현재)
    ② shuffle + bytedelta + LZ4
    ③ shuffle + bytedelta + Zstd(1)
    ④ trunc(3) + shuffle + bytedelta + LZ4
    ⑤ trunc(5) + shuffle + bytedelta + Zstd(1)
    ```
  - 각 조합에서 측정:
    - 압축률 (compressed / original)
    - 레이어별 압축률 분포 (16 layers)
    - 압축 시간 (ms) / 해제 시간 (ms)
    - 4MB 블록 기준 해제 지연 (3ms 예산 대비)

  ### Step 5: E2E 추론 벤치마크
  - `--kv-offload zram` + 각 파이프라인 설정으로 실행
  - 측정: tok/s, RAM 사용량, 생성 텍스트 품질 비교 (trunc_prec 설정별)

- **Acceptance Criteria**:
  1. bytedelta roundtrip 테스트 통과
  2. trunc_prec roundtrip 테스트 통과 (마스킹 후 복원 시 마스킹된 값 동일)
  3. 5개 조합의 압축률/속도 벤치마크 테이블 작성
  4. 레이어별 압축률 분포 확인
  5. 최적 조합 선정 및 근거 문서화
- **Notes**: |
  - Zstd 크레이트: `zstd = "0.13"` (C 바인딩, Android 크로스 컴파일 확인 필요)
  - 순수 Rust LZ4 대안: `lz4_flex` (크로스 컴파일 더 쉬움)
  - bytedelta NEON SIMD 최적화는 후속 작업으로 분리
  - 무손실 상한 기대치: 1.3~1.8x, trunc(5) 시 2.0~3.5x
