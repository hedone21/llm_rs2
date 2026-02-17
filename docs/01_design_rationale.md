# Chapter 1: 설계 결정 및 근거 (Design Rationale)

**이전**: [00. 구현 순서 가이드](00_build_guide.md) | **다음**: [02. Core 추상화](02_core_abstractions.md)

---

이 문서는 Antigravity 프레임워크의 주요 설계 결정과 그 근거를 설명합니다. 각 결정은 on-device LLM inference라는 목표에 최적화되어 있으며, 대안과의 비교를 통해 선택의 이유를 명확히 합니다.

## 1.1 왜 Rust인가

### Memory Safety without GC

LLM inference는 수 GB의 가중치를 메모리에 올리고, 수백만 번의 버퍼 접근을 수행합니다. C/C++에서는 use-after-free, double-free, buffer overflow가 성능 최적화 과정에서 빈번하게 발생합니다. Rust의 ownership 시스템은 이러한 버그를 컴파일 타임에 제거합니다.

GC(Garbage Collection)가 있는 언어는 추론 중 예측 불가능한 pause를 발생시킵니다. 실시간 토큰 생성에서 GC pause는 체감 latency를 크게 높입니다.

### Zero-Cost Abstractions

`Backend` trait, `Buffer` trait 등의 추상화가 런타임 오버헤드 없이 동작합니다. Monomorphization을 통해 trait method 호출이 컴파일 타임에 정적 dispatch로 변환됩니다.

```rust
// trait을 사용해도 vtable 오버헤드 없음 (제네릭 사용 시)
fn forward<B: Backend>(backend: &B, input: &Tensor) -> Tensor {
    backend.rms_norm(input, &weight, eps)  // 정적 dispatch
}
```

### Cross-Compilation

Android NDK toolchain과의 통합이 `cargo`와 `.cargo/config.toml` 설정만으로 완료됩니다. `aarch64-linux-android` target 추가 후 `cargo build --target`으로 바로 크로스 컴파일됩니다.

### FFI for OpenCL

`opencl3` crate를 통해 OpenCL C API를 안전하게 호출합니다. `unsafe` 블록이 FFI 경계에 한정되므로, 커널 실행 로직 자체는 safe Rust로 작성됩니다.

### Fearless Concurrency

Rayon을 활용한 data parallelism이 data race 걱정 없이 가능합니다. matmul의 행 단위 병렬화, 배치 처리 등에서 활용됩니다.

---

## 1.2 OpenCL vs Vulkan

### 비교

| 기준 | OpenCL | Vulkan Compute |
|------|--------|---------------|
| 추상화 수준 | 높음 (커널 중심) | 낮음 (파이프라인/디스크립터 관리 필요) |
| 모바일 GPU 지원 | Adreno, Mali 모두 성숙 | 지원되나 compute shader 생태계 미성숙 |
| 커널 개발 생산성 | `.cl` 파일 직접 작성, 빠른 이터레이션 | SPIR-V 컴파일 필요, 보일러플레이트 과다 |
| 메모리 관리 | `clCreateBuffer` + flags | VkBuffer + VkDeviceMemory + binding |
| matmul 성능 차이 | 충분 | 유의미한 우위 없음 |

### 결정

**OpenCL을 선택**합니다. 이유:

1. **Adreno GPU 타겟팅**: Qualcomm SoC(Snapdragon)의 Adreno GPU는 OpenCL 지원이 가장 안정적입니다. Qualcomm 자체가 OpenCL SDK를 제공하고 최적화 가이드를 배포합니다.

2. **개발 생산성**: `.cl` 커널 파일을 직접 수정하고 런타임에 JIT 컴파일하므로, 앱을 다시 빌드하지 않고도 커널을 튜닝할 수 있습니다.

3. **matmul 워크로드 특성**: LLM inference의 핵심은 대규모 matmul입니다. 이 워크로드에서 Vulkan compute가 OpenCL 대비 유의미한 성능 우위를 보이지 않습니다. Vulkan의 장점(세밀한 동기화, 멀티 큐)은 그래픽스 파이프라인에서 빛나지, compute-only 워크로드에서는 복잡성만 증가시킵니다.

---

## 1.3 matmul_transposed가 기본인 이유

### 배경

일반적인 matmul은 `C = A × B`이며 B의 열(column)에 순차 접근합니다. 그러나 row-major 메모리 레이아웃에서 열 접근은 cache miss를 유발합니다.

### 가중치 저장 형태

HuggingFace 모델의 가중치는 `[out_features, in_features]` shape으로 저장됩니다. 이는 이미 전치된 형태입니다.

```
Weight shape: [out_features, in_features]
             = B^T (B transposed)

matmul_transposed(A, B_T) = A × B_T^T = A × B
```

### Cache 친화적 접근 패턴

`matmul_transposed(A, B^T)`에서:

```
A[i, :] — i번째 행 → 연속 메모리 (순차 접근)
B^T[j, :] — j번째 행 → 연속 메모리 (순차 접근)

dot(A[i,:], B^T[j,:]) = C[i,j]
```

두 피연산자 모두 행(row) 단위로 접근하므로, cache line을 최대한 활용합니다. 일반 matmul에서 B의 열 접근 시 발생하는 stride 점프가 없습니다.

### Q4_0과의 정합성

Q4_0 양자화는 32개 원소를 하나의 block으로 묶습니다. 가중치가 `[out_features, in_features]`로 저장되면, 각 출력 뉴런의 가중치가 연속된 Q4_0 block 시퀀스를 형성합니다.

```
out_feature[0]: [block_0][block_1]...[block_N]  ← 연속 메모리
out_feature[1]: [block_0][block_1]...[block_N]  ← 연속 메모리
```

`matmul_transposed`는 이 레이아웃을 그대로 활용하여, block 단위 순차 접근으로 dequantize + dot product를 수행합니다.

---

## 1.4 Q4_0 양자화 선택

### GGML 호환 포맷

Q4_0은 llama.cpp(GGML)에서 정의한 양자화 포맷으로, 광범위한 검증과 커뮤니티 지원을 받고 있습니다. 자체 포맷을 설계하는 대신 검증된 포맷을 채택합니다.

### 구조와 원리

```
BlockQ4_0 (18 bytes / 32 elements):
┌─────────────────┬──────────────────────────────┐
│  scale (f16)     │  quants ([u8; 16])            │
│  2 bytes         │  16 bytes (32 x 4-bit)        │
└─────────────────┴──────────────────────────────┘

dequantize: value = (nibble - 8) * scale
```

- 4-bit 값에서 8을 빼는 방식 (offset bias)으로 zero-point calibration이 불필요합니다.
- 32개 원소당 18 bytes = 약 4.5 bits/value.

### 압축 효율

| 포맷 | bytes/element | 대비 F32 |
|------|---------------|---------|
| F32  | 4.0           | 1.0x    |
| F16  | 2.0           | 2.0x    |
| Q8_0 | ~1.06         | ~3.8x   |
| Q4_0 | ~0.56         | ~7.1x   |

3B 모델 기준으로 F32 약 12GB → Q4_0 약 1.7GB. 모바일 디바이스의 RAM 제약 (4-8GB) 내에서 동작 가능해집니다.

### 정확도 Trade-off

Q4_0은 F16 대비 약 5%의 perplexity 증가를 보입니다. On-device inference에서 이 정도의 품질 저하는 메모리 절감 대비 충분히 수용 가능합니다.

### Q8_0의 역할

Q8_0은 최종 저장 포맷이 아닌 **중간(intermediate) 포맷**입니다:

```
Activation (F32)
  → quantize_row_q8_0 → Q8_0 (임시)
  → vec_dot_q4_0_q8_0(weight_q4_0, activation_q8_0)
  → 결과 (F32)
```

F32 activation을 Q8_0으로 양자화한 뒤 Q4_0 가중치와 정수 dotprod를 수행하면, F32 x F32 dotprod 대비 **SIMD 활용도**가 크게 향상됩니다. ARM NEON의 `vdotq_s32`는 4개의 i8 x i8 곱을 한 사이클에 누적합니다.

---

## 1.5 start_pos vs current_pos 분리

### 문제 상황

KV cache eviction(예: sliding window)이 발생하면 물리적 cache 슬롯이 감소합니다. 그런데 RoPE(Rotary Position Embedding)는 토큰의 **절대 위치**를 인코딩합니다.

### 두 변수의 역할

```
start_pos:   RoPE에 전달되는 위치. 단조 증가(monotonically increasing).
             → KVCache의 필드가 아닌, 추론 루프(generate.rs 등)에서 관리하는 로컬 변수.
current_pos: KV cache의 물리적 슬롯 위치. eviction 후 감소 가능.
             → KVCache 구조체의 필드.
```

### 왜 분리가 필요한가

토큰 A, B, C, D가 순서대로 생성되었고, sliding window 크기가 3이라 A가 eviction되었다고 합시다:

```
eviction 전:
  KV cache: [A(pos=0), B(pos=1), C(pos=2), D(pos=3)]
  current_pos = 4, start_pos = 4

eviction 후 (A 제거):
  KV cache: [B(pos=1), C(pos=2), D(pos=3)]
  current_pos = 3  ← 물리적 슬롯 하나 줄어듦
  start_pos = 4    ← 변하지 않음

새 토큰 E 추가:
  RoPE position = start_pos = 4  ← B=1, C=2, D=3, E=4 (올바른 상대 거리)
  KV cache slot = current_pos = 3
```

만약 `start_pos`도 `current_pos`를 따라 3으로 줄었다면, E의 RoPE position이 3이 됩니다. 그러면 D(pos=3)와 E(pos=3)의 상대 거리가 0이 되어, attention이 두 토큰을 같은 위치로 인식하게 됩니다. 이는 생성 품질을 심각하게 훼손합니다.

### 핵심 원칙

> RoPE는 **기록 시점의 절대 위치**를 인코딩한다. KV cache의 물리적 레이아웃이 변해도 이미 인코딩된 위치 정보는 유효하다. 따라서 새 토큰의 RoPE position은 물리적 슬롯과 무관하게 단조 증가해야 한다.

---

## 1.6 LayerWorkspace 도입 이유

### 문제

Decode 단계에서는 매 step마다 단일 토큰(seq_len=1)을 처리합니다. 각 layer는 다음 버퍼를 필요로 합니다:

```
q, k, v:          QKV projection 결과
gate, up, down:   FFN intermediate
residual:         residual connection
attn_out:         attention 출력
scores:           Vec<f32> (CPU attention 시)
```

Layer당 **10개의 텐서 할당**(q, k, v, out_attn, gate, up, down, residual, attn_out + scores)이 발생합니다. 16-layer 모델에서 100 토큰을 생성하면 `16 × 100 × 10 = 16,000`회의 할당/해제가 발생합니다.

### 해결

`LayerWorkspace`를 도입하여 한 번 할당 후 모든 generation step에서 재사용합니다:

```rust
struct LayerWorkspace {
    q: Tensor,          // [batch, 1, q_dim]   (q_dim = num_heads * head_dim, flatten)
    k: Tensor,          // [batch, 1, k_dim]   (k_dim = kv_heads * head_dim)
    v: Tensor,          // [batch, 1, v_dim]
    out_attn: Tensor,   // [batch, 1, q_dim]   (attention 출력, projection 전)
    gate: Tensor,       // [batch, 1, ffn_hidden]
    up: Tensor,         // [batch, 1, ffn_hidden]
    down: Tensor,       // [batch, 1, dim]
    residual: Tensor,   // [batch, 1, dim]
    attn_out: Tensor,   // [batch, 1, dim]     (output projection 후)
    scores: Vec<f32>,   // [num_heads * max_seq_len]
}
```

> **주의**: `out_attn`과 `attn_out`은 별개 필드입니다. `out_attn`은 attention 연산의 직접 출력(q_dim 크기), `attn_out`은 o_proj를 거친 후의 결과(dim 크기)입니다.

Decode 시 `seq_len`이 항상 1이므로 버퍼 크기가 고정됩니다. 이 특성 덕분에 사전 할당이 가능합니다.

### 효과

- 할당자 압력(allocator pressure) 제거
- Cache locality 향상 (동일 메모리 반복 접근)
- GPU backend에서 `clCreateBuffer` 호출 최소화 (GPU 버퍼 할당은 특히 비용이 높음)

---

## 1.7 Backend trait default 구현

### 설계 원칙

`Backend` trait의 15+ 연산 중 일부는 성능이 덜 중요하거나 구현이 단순합니다. 이들에 대해 default 구현을 제공합니다:

```rust
trait Backend {
    // 반드시 구현해야 하는 핵심 연산
    fn matmul(&self, ...) -> Tensor;
    fn matmul_transposed(&self, ...) -> Tensor;
    fn matmul_slice(&self, ...) -> Tensor;
    fn rms_norm(&self, ...) -> Tensor;
    fn softmax(&self, ...) -> Tensor;
    fn rope_inplace(&self, ...);
    fn silu_mul(&self, ...) -> Tensor;
    fn add_assign(&self, ...);
    fn scale(&self, ...);
    fn copy_from(&self, ...);
    fn cast(&self, ...) -> Tensor;

    // default 구현 제공 (CPU fallback)
    fn attention_gen(&self, ...) -> Tensor { /* CPU 구현 */ }
    fn gather(&self, ...) -> Tensor { /* CPU 구현 */ }
    fn copy_slice(&self, ...) { /* CPU 구현 */ }
    fn read_buffer(&self, ...) -> Vec<u8> { /* CPU 구현 */ }
    fn synchronize(&self) -> Result<()> { Ok(()) }  // no-op
}
```

### 근거

1. **진입 장벽 최소화**: 새 backend(예: NPU, WebGPU)를 추가할 때 10개 핵심 연산만 구현하면 동작합니다.
2. **점진적 최적화**: 처음에는 default 구현으로 동작을 검증하고, 이후 성능이 중요한 연산부터 네이티브 구현으로 교체합니다.
3. **정확성 보장**: default 구현이 CPU reference 역할을 하므로, 새 구현의 결과를 바로 비교할 수 있습니다.

---

## 1.8 Feature-gating OpenCL

### 문제

OpenCL은 GPU 드라이버가 있는 환경에서만 동작합니다. 개발 머신(Linux/macOS)에서는 GPU가 없거나 OpenCL SDK가 설치되지 않은 경우가 흔합니다. 그러나 코드베이스를 CPU-only와 GPU 버전으로 분리하면 유지보수 비용이 급격히 증가합니다.

### 해결

`Cargo.toml`에서 `opencl` feature를 default로 활성화하되, 조건부 컴파일로 GPU 전용 코드를 격리합니다:

```toml
[features]
default = ["opencl"]
opencl = ["dep:opencl3"]
```

```rust
// Buffer trait에서 조건부 컴파일
trait Buffer {
    fn as_ptr(&self) -> *const u8;
    fn len(&self) -> usize;

    #[cfg(feature = "opencl")]
    fn cl_mem(&self) -> Option<&cl_mem>;
}
```

### 동작 방식

| 환경 | feature | 빌드 | 실행 |
|------|---------|------|------|
| 개발 PC (GPU 없음) | `opencl` (default) | 컴파일 성공 | GPU 연산 호출 시 런타임 에러 반환 |
| 개발 PC (GPU 없음) | `--no-default-features` | 컴파일 성공 | CPU-only 동작 |
| Android (Adreno GPU) | `opencl` (default) | 컴파일 성공 | GPU 가속 동작 |

### 장점

- **단일 코드베이스**: CPU-only 개발과 GPU 가속 배포가 같은 코드에서 이루어집니다.
- **CI 친화적**: GPU가 없는 CI 환경에서도 `cargo check`, `cargo test`가 통과합니다.
- **점진적 전환**: CPU-only로 로직을 검증한 뒤, 동일 코드에 GPU backend를 연결하면 됩니다.

---

**이전**: [00. 구현 순서 가이드](00_build_guide.md) | **다음**: [02. Core 추상화](02_core_abstractions.md)
