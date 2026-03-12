# Rust Developer TODO

> **역할**: 백엔드 구현, 모델 추론 로직, 성능 최적화, 버그 수정
> **소유 영역**: `engine/src/`, `manager/`, `shared/`

---

## [P0] .cargo/config.toml 타겟 트리플 수정
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: `[target.x86_64-linux]` → `[target.x86_64-unknown-linux-gnu]`. 현재 AVX2/FMA 컴파일러 플래그가 호스트 빌드에 전혀 적용되지 않음. 명시적 `#[target_feature]` 함수(matmul)는 동작하지만, 스칼라 폴백 경로(RoPE, softmax, attention dot product, rms_norm 등)의 자동 벡터화 기회 상실.
- **Acceptance Criteria**: `rustc -vV`의 host 트리플과 config.toml 일치, 릴리스 빌드에서 AVX2/FMA 플래그 적용 확인
- **Notes**: 파일 `.cargo/config.toml:7`

## [P0] Rayon 스레드 수 자동 감지
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: `generate.rs:240`에서 `num_threads(8)` 하드코딩. 호스트 CPU 코어 수에 관계없이 8스레드만 사용 (예: 20코어 CPU → 40% 활용). CLI 옵션 `--threads`로 제어하거나 자동 감지로 변경.
- **Acceptance Criteria**: 호스트 CPU 코어 수에 맞게 스레드 풀 생성, CLI 옵션 제공
- **Notes**: 파일 `engine/src/bin/generate.rs:240`

## [P0] x86 attention AVX2 SIMD 구현
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음
- **Description**: `forward_gen` 내부 attention의 Q*K^T와 S*V 연산이 x86에서 순수 스칼라 루프. aarch64에는 NEON 4x 언롤 SIMD 구현이 있으나 x86은 `for i in 0..head_dim { score += q*k }`. AVX2 256-bit FMA로 8 float 동시 처리하면 2-4x 가속 기대. parallel/serial 양쪽 경로 모두 적용 필요.
- **Acceptance Criteria**: x86 attention dot product에 AVX2 구현, `cargo test -p llm_rs2` 통과
- **Notes**: 파일 `engine/src/layers/llama_layer.rs:557-712`. NEON 구현 참고하여 동일 구조로 AVX2 포팅.

## [P0] 기준선 측정 및 수정 전후 비교
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 위 P0 3개 수정 완료
- **Description**: P0 수정 전후 tok/sec, TTFT, TBT 측정. 프롬프트 길이별(short/long), KV 타입별(f32/q4) 비교. 이론적 피크(Llama 3.2 1B Q4_0: ~667MB/token, DDR5 80GB/s → ~120 tok/sec)와 달성률 계산.
- **Acceptance Criteria**: 수정 전후 비교 테이블, 개선 폭 수치화

## [P1] forward_gen 연산별 내부 계측
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: P0 수정 완료
- **Description**: `--profile` CLI 플래그 추가. forward_gen 내부 15개 연산 구간별 `Instant` 타이밍: matmul×7, attention_gen, rms_norm×2, rope×2, silu_mul, add_assign×2, copy_from×2, cast×2. 100토큰 이상 누적 후 평균/비율 출력. attention 스케일링 분석(pos 10/100/500/1000/2048).
- **Acceptance Criteria**: 연산별 평균 시간(us) + 비율(%) 테이블, attention 스케일링 곡선
- **Notes**: `engine/src/layers/llama_layer.rs`, `engine/src/models/llama/llama_model.rs`

## [P1] 외부 프로파일링 (flamegraph + perf stat)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: P0 수정 완료
- **Description**: `cargo flamegraph`로 CPU 시간 분포 시각화, `perf stat`으로 IPC/캐시 미스율 측정. prefill-only vs decode-only 분리 프로파일링. Rayon 오버헤드 확인.
- **Acceptance Criteria**: Flamegraph SVG, perf stat 카운터 테이블, 병목 분류(compute/memory/overhead)

## [P1] micro_bench 확장
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: P0 수정 완료
- **Description**: 현재 `quantize_row_q8_0`, `vec_dot_q4_0_q8_0`만 벤치마크. 추가 대상: matmul 차원별(N=2048/8192/128256), attention seq_len별(64~2048), rms_norm/rope/silu_mul 개별, Rayon serial vs parallel 비교.
- **Acceptance Criteria**: 연산별 throughput 테이블, Rayon 오버헤드 정량화
- **Notes**: `engine/src/bin/micro_bench.rs`

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
