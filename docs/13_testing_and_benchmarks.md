# Chapter 13: 테스트 및 벤치마크 (Testing & Benchmarks)

**이전**: [12. 하이브리드 추론](12_hybrid_inference.md) | **다음**: 없음

---

## 13.1 3-Tier 테스트 전략

Antigravity는 3단계 테스트 전략을 채택한다. 각 단계는 검증 범위와 실행 환경이 다르다.

| Tier | 실행 환경 | 명령어 | 검증 대상 |
|------|-----------|--------|-----------|
| 1. Host Unit Tests | 개발 호스트 | `cargo test` | Tokenizer, shape inference, KVCache 연산, eviction 정책 |
| 2. Backend Verification | Android 디바이스 | `test_backend` | CPU vs OpenCL kernel 정확성 비교 |
| 3. E2E Inference | Android 디바이스 | `generate` | 전체 모델 추론 (토큰 생성 품질, 성능) |

Tier 1은 플랫폼 독립적인 로직을 호스트에서 빠르게 검증하고, Tier 2는 백엔드 간 수치 정확성을 보장하며, Tier 3은 실제 디바이스에서의 end-to-end 동작을 확인한다.

## 13.2 단위 테스트 위치

단위 테스트는 각 소스 파일 내부의 `#[cfg(test)] mod tests` 블록에 작성한다. 별도의 test 디렉토리가 아닌 구현 파일과 같은 위치에 둔다.

주요 테스트 파일 예시:

- **`src/core/kv_cache.rs`**: `prune_prefix` 동작 검증, `memory_usage_bytes` 계산 정확성
- **`src/core/eviction/sliding_window.rs`**: Sliding window eviction 동작, protected prefix 보존 확인
- **`src/core/cache_manager.rs`**: Mock `SystemMonitor`를 사용한 eviction 트리거 조건 검증

테스트 작성 규칙:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prune_prefix_preserves_recent_tokens() {
        // 테스트 구현
    }
}
```

모든 새로운 기능 및 버그 수정에는 반드시 테스트를 포함해야 한다.

## 13.3 test_backend 상세

`src/bin/test_backend.rs`는 백엔드 간 정확성을 비교하는 oracle 테스트 바이너리이다.

### 목적

서로 다른 Backend 구현(NEON, AVX2, OpenCL)이 동일한 입력에 대해 동일한 출력을 생성하는지 검증한다.

### Reference Backend

`CpuBackendCommon` (Scalar 구현)을 ground truth로 사용한다. 모든 SIMD 최적화와 GPU 구현은 이 scalar 결과와 비교된다.

### 테스트 대상 백엔드

| 백엔드 | 설명 |
|--------|------|
| `auto` | 플랫폼 기본 `CpuBackend` (aarch64: NEON, x86_64: AVX2) |
| `scalar` | `CpuBackendCommon` (최적화 없음, reference) |
| `avx2` | `CpuBackendAVX2` (x86_64 전용) |
| `opencl` | `OpenCLBackend` (GPU) |

### 테스트 연산

- `MatMulTransposed` — F32, Q4_0 dtype 지원 (Q4_1은 코드에 있으나 비활성화)
- `MatMulSlice` — F32
- `RMSNorm`
- `Softmax`
- `RoPE`

### 테스트 Shape

Small부터 Large까지 다양한 크기를 순차적으로 테스트한다:

```
(M, K, N):
  (1, 64, 32), (4, 128, 64),
  (1, 256, 128), (4, 256, 128), (8, 256, 128),
  (32, 256, 128), (64, 256, 128),
  (1, 4096, 4096)
```

### 검증 기준

- 검증 위치: 출력 텐서의 `[M/2, N/2]` 위치의 값을 비교
- 오차 임계값: `|reference - actual| > 1e-2` → **FAIL**

### 출력 형식

동적 멀티 백엔드 비교 테이블로 출력된다. `--backends` 인자에 따라 컬럼이 변한다. 각 백엔드는 Duration과 Error 두 컬럼을 가지며, Scalar 대비 speedup이 Duration에 `(x{:.2})` 형식으로 인라인 표시된다.

```
Operation     | Shape            | DType | CPU (Scalar)      |         | CPU (NEON)        |
              |                  |       | Duration    Error  |         | Duration    Error  |
MatMulTrans   | (1, 4096, 4096)  | F32   | 45.20ms     -      |         | 12.10ms(x3.74) 0.0001 |
```

> 참고: "Reference" 열은 수학적으로 계산된 dot product 값(`A[M/2]·B[N/2]`)이며, `CpuBackendCommon`의 출력이 아닌 **호스트에서 별도 계산된 값**이다. Scalar 백엔드는 timing baseline으로만 사용된다.

## 13.4 test_backend 검증 방법

### MatMulTransposed 검증

1. 입력 행렬을 결정적 패턴으로 생성한다:
   ```rust
   A[i] = (i % 100) as f32 * 0.01 - 0.5   // 범위: [-0.5, 0.49]
   B[i] = (i % 123) as f32 * 0.01 - 0.5   // 다른 modulus로 A와 구분
   ```
2. (선택적) `B`를 Q4_0로 quantize한다.
3. 테스트 백엔드에서 연산을 10회 반복 실행한다 (총 시간 측정).
4. Reference 값을 호스트에서 수학적으로 계산한다: `A[M/2]` 행과 `B[N/2]` 행의 dot product.
5. 비교: `|ref - actual| < 1e-2`이면 PASS.

### RMSNorm / Softmax 검증

1. 출력 버퍼 `C`를 초기값으로 채운다.
2. 여러 iteration 실행한다.
3. 출력의 중심 원소(center element)를 reference와 비교한다.

## 13.5 micro_bench 상세

`src/bin/micro_bench.rs`는 저수준 SIMD 연산의 성능을 scalar 구현과 비교하는 마이크로 벤치마크 바이너리이다.

### 벤치마크 항목

| 벤치마크 | 연산 | K | Iterations |
|----------|------|---|------------|
| `quantize_row_q8_0` | F32 → Q8_0 양자화 | 4096 | 10,000 |
| `vec_dot_q4_0_q8_0` | Q4_0 × Q8_0 dot product | 4096 | 10,000 |

### 실행 방법

1. **Warmup**: 전체 iteration의 1/10을 warmup으로 실행한다 (iter/10회).
2. **Timed run**: 전체 iteration을 다시 실행하며 총 시간을 측정한다 (warmup과 별도 루프, iter회).
3. **결과 출력**: 평균 ms, scalar 대비 speedup ratio를 출력한다.

### 출력 형식

```
quantize_row_q8_0 (K=4096, iters=10000):
  scalar:  2.15 ms/iter
  simd:    0.38 ms/iter
  speedup: 5.66x

vec_dot_q4_0_q8_0 (K=4096, iters=10000):
  scalar:  1.87 ms/iter
  simd:    0.29 ms/iter
  speedup: 6.45x
```

### 아키텍처별 백엔드

| 아키텍처 | SIMD Backend |
|----------|-------------|
| aarch64 | `CpuBackendNeon` (ARM NEON + dotprod) |
| x86_64 | `CpuBackendAVX2` (runtime detection) |

x86_64에서 AVX2가 감지되지 않으면 scalar로 fallback한다.

## 13.6 프로파일링 워크플로우

프로파일링은 Android 디바이스에서의 실제 성능을 측정하고 시각화하는 파이프라인이다.

### 도구 체인

| 도구 | 역할 |
|------|------|
| `scripts/android_profile.py` | Android 디바이스에서 프로파일링 실행, JSON 결과 수집 |
| `scripts/visualize_profile.py` | 프로파일 데이터로 성능 그래프 생성 |
| `web_dashboard/` | Flask 기반 대시보드 (localhost:5000), 브라우저에서 결과 탐색 |

### 데이터 경로

| 경로 | 내용 | Git 상태 |
|------|------|----------|
| `results/data/` | JSON 프로파일 결과 | 커밋됨 (test data) |
| `results/plots/` | 시각화 결과 이미지 | gitignored |

`results/data/`의 JSON 파일은 테스트 데이터로서 repo에 커밋된다. 시각화 결과는 로컬에서만 생성한다.

## 13.7 테스트 실행 가이드

### Host 단위 테스트

```bash
# 전체 단위 테스트 실행
cargo test

# 특정 모듈만 테스트
cargo test kv_cache
cargo test sliding_window
```

### Backend 정확성 검증 (디바이스)

```bash
# 기본 백엔드 비교
./.agent/skills/testing/scripts/run_android.sh test_backend --backends auto,scalar

# OpenCL 포함 전체 비교
./.agent/skills/testing/scripts/run_android.sh test_backend --backends auto,scalar,opencl
```

### 마이크로 벤치마크 (디바이스)

```bash
./.agent/skills/testing/scripts/run_android.sh micro_bench
```

### E2E 추론 (디바이스)

```bash
./.agent/skills/testing/scripts/run_android.sh generate --prompt "Hello" -n 128
```

### 프로파일링

```bash
# 디바이스에서 프로파일링 실행
python scripts/android_profile.py --output-name test_run

# 결과 시각화
python scripts/visualize_profile.py results/data/test_run.json

# 대시보드 실행
cd web_dashboard && python app.py
```

### 코드 품질 검사

```bash
# fmt + clippy 실행
./.agent/skills/developing/scripts/sanity_check.sh
```

---

**이전**: [12. 하이브리드 추론](12_hybrid_inference.md) | **다음**: 없음
