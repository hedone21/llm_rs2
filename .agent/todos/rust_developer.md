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
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: P0 수정 완료 ✅
- **Description**: `--profile` CLI 플래그 추가. forward_gen 내부 15개 연산 구간별 `Instant` 타이밍: matmul×7, attention_gen, rms_norm×2, rope×2, silu_mul, add_assign×2, copy_from×2, cast×2. 100토큰 이상 누적 후 평균/비율 출력. attention 스케일링 분석(pos 10/100/500/1000/2048).
- **Acceptance Criteria**: 연산별 평균 시간(us) + 비율(%) 테이블, attention 스케일링 곡선
- **Notes**: `engine/src/layers/llama_layer.rs`, `engine/src/models/llama/llama_model.rs`

## [P1] 외부 프로파일링 (flamegraph + perf stat)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: P0 수정 완료 ✅
- **Description**: `cargo flamegraph`로 CPU 시간 분포 시각화, `perf stat`으로 IPC/캐시 미스율 측정. prefill-only vs decode-only 분리 프로파일링. Rayon 오버헤드 확인.
- **Acceptance Criteria**: Flamegraph SVG, perf stat 카운터 테이블, 병목 분류(compute/memory/overhead)

## [P1] micro_bench 확장
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: P0 수정 완료 ✅
- **Description**: 현재 `quantize_row_q8_0`, `vec_dot_q4_0_q8_0`만 벤치마크. 추가 대상: matmul 차원별(N=2048/8192/128256), attention seq_len별(64~2048), rms_norm/rope/silu_mul 개별, Rayon serial vs parallel 비교.
- **Acceptance Criteria**: 연산별 throughput 테이블, Rayon 오버헤드 정량화
- **Notes**: `engine/src/bin/micro_bench.rs`

## [P1] 이론적 피크 대비 달성률 분석
- **Status**: DONE
- **Sprint**: current
- **Notes**: Llama 3.2 1B Q4_0 디코드 기준 이론 피크 ~120 tok/s (DDR5 80GB/s). 실측 51.5 tok/s = 43% 달성. 스레드 12 이상에서 수확체감 → memory bandwidth bound 확인.

## [P2] 소규모 연산 Rayon 오버헤드 제거
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: P1 계측 결과로 영향 확인 후
- **Description**: rms_norm(2048), add_assign(2048) 등 L1 캐시 크기 이하 연산에서 Rayon dispatch 오버헤드가 연산 비용 초과 가능. 크기 threshold 기반 serial/parallel 분기.
- **Acceptance Criteria**: threshold 이하 연산 serial 실행, 벤치마크로 개선 확인
- **Notes**: `engine/src/backend/cpu/common.rs`

## [P2] RoPE 주파수 테이블 사전 계산
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: P1 계측 결과로 영향 확인 후
- **Description**: 현재 `theta.powf()` + `sin_cos()` 매 토큰 호출 (32Q+8KV heads × 32 pairs = 1280 sin/cos). 모델 초기화 시 주파수 테이블 사전 계산하여 디코드 핫루프에서 lookup만 수행.
- **Acceptance Criteria**: RoPE 시간 감소 확인, 기존 테스트 통과
- **Notes**: `engine/src/backend/cpu/common.rs`

## [P2] Q4_0 matmul M≥4 AVX2 구현 (프리필 최적화)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: P1 계측 결과로 프리필 병목 확인 후
- **Description**: 현재 M≥4 Q4_0 matmul이 스칼라 Common으로 폴백 (x86.rs에서 M<4만 AVX2). M<4 AVX2 커널을 확장하여 블로킹 처리.
- **Acceptance Criteria**: 프리필 속도 개선, 기존 테스트 통과
- **Notes**: `engine/src/backend/cpu/x86.rs`
