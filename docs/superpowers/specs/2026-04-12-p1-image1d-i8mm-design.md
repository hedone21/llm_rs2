# P1 성능 최적화: GPU image1d_buffer_t + CPU i8mm

> **목적**: Q4_0 decode 성능을 llama.cpp 수준에 근접시킨다.
> GPU는 텍스처 캐시 활용, CPU는 i8mm 2-row 연산으로 throughput 향상.
>
> **배경**: 이전 세션에서 noshuffle MVP (+2%) 및 sdot 2-block unrolling (+84%) 완료.
> Global buffer 읽기 → image1d_buffer_t 전환이 GPU 성능 격차(74%)의 핵심.
>
> **참조**: llama.cpp `gemv_noshuffle.cl`, `ggml-opencl.cpp:4710-4750`, `arch/arm/quants.c:261-331`

---

## 1. 결정 사항 요약

| 항목 | 결정 |
|------|------|
| 두 작업 관계 | 독립. 설계 통합, 구현은 GPU → CPU 순서 |
| GPU 커널 전략 | 기존 `.cl` 파일을 image 버전으로 교체 (global buffer MVP 제거) |
| GPU fallback | image1d_buffer_t 미지원 시 기존 matmul_q4_0 (non-noshuffle) fallback |
| CPU i8mm 전략 | matmul 루프를 2-row 단위로 변경 (vec_dot 레벨 아닌 matmul 레벨) |
| CPU i8mm 감지 | `is_aarch64_feature_detected!("i8mm")` 런타임 감지 |
| CPU 홀수 나머지 | sdot fallback |
| 인라인 asm | `smmla` — 현재 `sdot`과 동일한 패턴 |
| scales 접근 | global half2* 유지 (image 아님, llama.cpp 동일) |

---

## 2. 현재 성능 기준선 (Llama 3.2 1B, Adreno 830)

| 경로 | llm.rs | llama.cpp | 달성률 |
|------|--------|-----------|--------|
| GPU Q4 decode | 33.3 tok/s | 45.2 tok/s | 74% |
| CPU Q4 decode | 13.6 tok/s | 36.0 tok/s | 38% |

---

## 3. GPU P1: image1d_buffer_t 전환

### 변경 개요

현재 MVP는 llama.cpp의 `read_imageui()` / `read_imagef()`를 `global uint*` / `global float4*`로 대체한 상태. 이를 원본과 동일한 `image1d_buffer_t` 접근으로 전환하여 Adreno TP(Texture Processor) 캐시를 활용한다.

### 커널 변경

- `src0_q`: `global uint*` → `__read_only image1d_buffer_t` (R32UI)
- `src1`: `global float4*` → `__read_only image1d_buffer_t` (RGBA32F)
- `src0_d`: 변경 없음 (`global half2*`)
- 매크로, reduction, dispatch 크기, 컴파일타임 상수 모두 동일

### Rust 변경

- `NoshuffleSoaEntry`에 `q_img: Mem` 추가 (weight image, 로드 시 1회 생성)
- Activation image는 매 dispatch 시 생성 (lightweight wrapper, 메모리 복사 없음)
- `ocl-core::create_image` + `ImageDescriptor` + `MemObjectType::Image1dBuffer` 사용
- 커널 arg에 image Mem 전달

### Image 포맷

| 대상 | channel order | channel type | width |
|------|--------------|-------------|-------|
| weight q | `CL_R` | `CL_UNSIGNED_INT32` | 총 uint 개수 |
| activation | `CL_RGBA` | `CL_FLOAT` | `ne00 / 4` |

### 리스크

- `CL_DEVICE_IMAGE_MAX_BUFFER_SIZE` 초과 시 noshuffle 비활성화 (기존 fallback)
- Activation image 매 호출 생성 오버헤드 — 프로파일 후 캐시 검토

---

## 4. CPU P1: i8mm 2-row dot product

### 변경 개요

`smmla` (FEAT_I8MM) 명령어로 2 weight row × 1 activation row를 동시 연산. matmul 루프를 2-row 단위로 변경하여 i8mm의 2x8x2 행렬곱 이점을 완전 활용.

### 컴포넌트

- **`vec_dot_q4_0_q8_0_i8mm`**: 2-row dot product 함수. llama.cpp `__ARM_FEATURE_MATMUL_INT8` 경로 기반. `vzip1/2_s64`로 weight 2행 interleave, activation 복제, `smmla` 4회 체이닝.
- **matmul 루프 변경**: `is_aarch64_feature_detected!("i8mm")` 시 j를 2씩 증가, 홀수 나머지는 sdot fallback. 기존 `par_chunks_mut` 병렬화 유지.

### 감지 우선순위

```
i8mm → sdot (dotprod) → 기본 NEON
```

### 리스크

- signed/unsigned 시맨틱: Q4_0 nibble → `-8` 오프셋 후 signed. `smmla`는 signed 연산. 단위 테스트 검증.
- macOS Apple Silicon은 i8mm 미지원 → 디바이스에서만 실행 검증.

---

## 5. 영향 범위

| 파일 | 변경 |
|------|------|
| `engine/kernels/gemv_noshuffle_q4_0.cl` | global → image1d_buffer_t |
| `engine/src/backend/opencl/mod.rs` | NoshuffleSoaEntry 확장, image 생성, dispatch 변경 |
| `engine/src/backend/cpu/neon.rs` | i8mm 2-row dot 함수 추가, matmul 루프 2-row 분기 |

---

## 6. 테스트

### GPU P1

- image1d_buffer_t 생성 성공 + 포맷 검증 (호스트)
- noshuffle image 경로 수치 정확성 vs CPU reference (디바이스)
- image width 상한 초과 시 graceful fallback

### CPU P1

- 2-row dot product vs f64 reference (다양한 차원)
- matmul 짝수/홀수 n, m=1/m>1 정확성
- 런타임 감지 분기 동작

---

## 7. 성공 기준

| 항목 | GPU P1 | CPU P1 |
|------|--------|--------|
| 정확성 | 기존 noshuffle 테스트 전수 통과 | reference 대비 오차 < 1e-4 |
| 성능 목표 | GPU Q4 decode >= 40 tok/s | CPU Q4 decode >= 18 tok/s |
| llama.cpp 달성률 | >= 88% (현재 74%) | >= 50% (현재 38%) |
| 회귀 없음 | F16/Q8_0 경로 변화 없음 | dotprod 경로 변화 없음 |
