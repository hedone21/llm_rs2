# Chapter 3: CPU 백엔드 (CPU Backend)

**이전**: [02. Core 추상화](02_core_abstractions.md) | **다음**: [04. 모델 로딩](04_model_loading.md)

---

이 장에서는 CPU 백엔드의 구현을 상세히 다룹니다. 스칼라 연산부터 ARM NEON, x86 AVX2 SIMD 최적화까지 — 각 연산의 알고리즘과 병렬화 전략을 설명합니다.

---

## 3.1 아키텍처 개요

**파일**: `src/backend/cpu/mod.rs`

CPU 백엔드는 조건부 컴파일을 통해 플랫폼별 최적화 구현을 선택합니다.

```rust
#[cfg(target_arch = "x86_64")]
pub type CpuBackend = CpuBackendAVX2;

#[cfg(target_arch = "aarch64")]
pub type CpuBackend = CpuBackendNeon;

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
pub type CpuBackend = CpuBackendCommon;
```

`CpuBackend`는 type alias로, 컴파일 대상 아키텍처에 따라 자동으로 최적 구현체가 선택됩니다.

| 구현체 | 파일 | 대상 | 설명 |
|--------|------|------|------|
| `CpuBackendCommon` | `common.rs` (~725줄) | 모든 플랫폼 | 이식 가능한 스칼라 구현 |
| `CpuBackendNeon` | `neon.rs` (~548줄) | AArch64 | ARM NEON SIMD 최적화 |
| `CpuBackendAVX2` | `x86.rs` (~598줄) | x86_64 | AVX2 + FMA 최적화 |

### 위임 패턴

NEON과 AVX2 구현체는 SIMD 최적화가 의미 있는 연산(주로 `matmul_transposed`)만 직접 구현하고, 나머지 연산은 `CpuBackendCommon`에 위임합니다.

```rust
// CpuBackendNeon의 예시
impl Backend for CpuBackendNeon {
    fn matmul_transposed(&self, a, b, out) -> Result<()> {
        match b.dtype() {
            DType::F32  => self.matmul_transposed_f32(a, b, out),  // NEON 최적화
            DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out), // NEON 최적화
            _           => CpuBackendCommon::new().matmul_transposed(a, b, out), // 위임
        }
    }

    fn add_assign(&self, a, b) -> Result<()> {
        CpuBackendCommon::new().add_assign(a, b)  // 항상 위임
    }
    // ... 나머지 연산도 CpuBackendCommon에 위임
}
```

---

## 3.2 CpuBackendCommon 스칼라 구현

**파일**: `src/backend/cpu/common.rs`

모든 Backend trait 연산의 기본(baseline) 구현입니다. SIMD 없이도 동작하며, rayon을 활용한 멀티스레드 병렬화가 적용되어 있습니다.

### matmul(a, b, out)

표준 행렬 곱셈 [M,K] x [K,N] → [M,N]. B가 row-major 형태입니다.

```rust
fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
    // rayon par_chunks_mut: M개 행을 병렬 처리
    out_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..k {
                sum += a_data[i * k + l] * b_data[l * n + j];
            }
            row[j] = sum;
        }
    });
}
```

행(row) 단위로 rayon `par_chunks_mut`을 사용하여 M 차원을 병렬화합니다.

### matmul_transposed(a, b, out)

B가 전치된 형태의 행렬 곱셈. LLM 추론에서 가장 빈번하게 호출되는 연산입니다. B의 dtype에 따라 세 가지 경로로 분기합니다.

```rust
fn matmul_transposed(&self, a, b, out) -> Result<()> {
    match b.dtype() {
        DType::F32  => self.matmul_transposed_f32(a, b, out),
        DType::Q4_0 => self.matmul_transposed_q4_0(a, b, out),
        DType::Q4_1 => self.matmul_transposed_q4_1(a, b, out),
        _           => Err(anyhow!("Unsupported dtype")),
    }
}
```

#### matmul_transposed_f32

B가 [N,K] 형태로 저장되어 있으므로, A의 i번째 행과 B의 j번째 행의 내적(dot product)으로 계산합니다.

```rust
// B[j] row와 A[i] row의 dot product
out_data.par_chunks_mut(n).enumerate().for_each(|(i, row)| {
    for j in 0..n {
        let mut sum = 0.0;
        for l in 0..k {
            sum += a_data[i * k + l] * b_data[j * k + l]; // B는 전치 형태
        }
        row[j] = sum;
    }
});
```

#### matmul_transposed_q4_0

B가 Q4_0 양자화된 형태. M의 크기에 따라 두 가지 전략을 사용합니다.

**작은 M (M < 4)**: A를 Q8_0로 양자화한 뒤 정수 내적(`vec_dot_q4_0_q8_0`)을 수행합니다.

```
A (F32) → quantize_row_q8_0 → A' (Q8_0)
A' (Q8_0) x B (Q4_0) → vec_dot_q4_0_q8_0 → out (F32)
```

**큰 M (M >= 4)**: 블록별로 Q4_0 니블을 인라인 역양자화하면서 누적합니다.

```rust
for (bi, block) in b_row_blocks.iter().enumerate() {
    let d = block.d.to_f32();
    let a_slice = &a_row[bi * QK4_0..(bi + 1) * QK4_0];
    let mut isum: f32 = 0.0;
    for z in 0..(QK4_0 / 2) {
        let b = block.qs[z];
        let v0 = (b & 0x0F) as i8 - 8;
        let v1 = (b >> 4) as i8 - 8;
        isum += v0 as f32 * a_slice[z];
        isum += v1 as f32 * a_slice[z + QK4_0 / 2];
    }
    sum += d * isum;
}
```

#### matmul_transposed_q4_1

Q4_1 양자화는 scale(`d`)과 minimum(`m`) 두 파라미터를 사용합니다. 두 개의 누적합 `s0`, `s1`을 유지합니다.

```rust
for (bi, block) in b_row_blocks.iter().enumerate() {
    let d = block.d.to_f32();
    let m = block.m.to_f32();
    let mut s0 = 0.0;  // d 계수에 곱해질 합
    let mut s1 = 0.0;  // m 계수에 곱해질 합 (A 원소의 합)

    for z in 0..(QK4_1 / 2) {
        let b = block.qs[z];
        let v0 = (b & 0x0F) as f32;
        let v1 = (b >> 4) as f32;
        s0 += v0 * a_slice[z] + v1 * a_slice[z + QK4_1 / 2];
        s1 += a_slice[z] + a_slice[z + QK4_1 / 2];
    }
    sum += d * s0 + m * s1;  // value = nibble * d + m → dot = d*s0 + m*s1
}
```

### add_assign(a, b)

요소별 덧셈. `a += b`.

```rust
a_data.par_iter_mut().zip(b_data.par_iter()).for_each(|(x, y)| {
    *x += y;
});
```

### scale(x, v)

스칼라 곱셈. `x *= v`.

```rust
x_data.par_iter_mut().for_each(|val| *val *= v);
```

### silu_mul(a, b)

SiLU (Swish) 활성화 함수 적용 후 요소별 곱셈. FFN 게이트 연산에 사용됩니다.

```rust
// SiLU(x) = x / (1 + exp(-x))
// a = SiLU(a) * b
a_data.par_iter_mut().zip(b_data.par_iter()).for_each(|(x, y)| {
    let silu_x = *x / (1.0 + (-*x).exp());
    *x = silu_x * y;
});
```

### rms_norm(x, w, eps)

RMS (Root Mean Square) 정규화. Transformer의 각 서브레이어 앞에서 호출됩니다.

```rust
// 행 단위 처리
x_data.par_chunks_mut(dim).for_each(|row| {
    let sum_sq: f32 = row.iter().map(|&v| v * v).sum();
    let rms = (sum_sq / dim as f32 + eps).sqrt();
    let scale = 1.0 / rms;
    for (val, weight) in row.iter_mut().zip(w_data.iter()) {
        *val = (*val * scale) * weight;
    }
});
```

수식: `output[i] = (x[i] / sqrt(mean(x^2) + eps)) * weight[i]`

### softmax(x)

행 단위 softmax. 수치 안정성을 위해 최댓값을 먼저 빼줍니다.

```rust
x_data.par_chunks_mut(dim).for_each(|row| {
    let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut sum_exp = 0.0;
    for val in row.iter_mut() {
        *val = (*val - max_val).exp();  // 오버플로 방지
        sum_exp += *val;
    }
    for val in row.iter_mut() {
        *val /= sum_exp;               // 정규화
    }
});
```

### rope_inplace(x, start_pos, theta)

RoPE (Rotary Position Embedding)를 in-place로 적용합니다. 각 head의 차원 쌍 `(x[i], x[i + d/2])`에 위치 기반 회전을 적용합니다.

```rust
for i in 0..head_dim / 2 {
    let freq = theta.powf(-2.0 * (i as f32) / (head_dim as f32));
    let val = pos as f32 * freq;
    let (sin, cos) = val.sin_cos();

    let v0 = head_slice[i];
    let v1 = head_slice[i + head_dim / 2];

    head_slice[i]               = v0 * cos - v1 * sin;
    head_slice[i + head_dim / 2] = v0 * sin + v1 * cos;
}
```

주요 특징:
- 주파수: `freq = theta^(-2i/d)` (차원이 높을수록 낮은 주파수)
- 위치 `pos`는 eviction 후에도 단조 증가 (monotonically increasing)
- Batch 차원에 대해 `par_chunks_mut`으로 병렬화

### cast(src, dst)

타입 변환. 지원하는 변환:

| 소스 | 대상 | 방법 |
|------|------|------|
| F32 | F16 | `f16::from_f32()` |
| F16 | F32 | `f16::to_f32()` |
| F32 | Q4_0 | 블록 양자화: `max_abs / 7.0`으로 scale 계산 |

F32 → Q4_0 변환:

```rust
let max_val = src_block.iter().map(|v| v.abs()).fold(0.0f32, |x, y| x.max(y));
let scale = max_val / 7.0;
let inv_scale = if scale == 0.0 { 0.0 } else { 1.0 / scale };
d[bi].d = f16::from_f32(scale);
for z in 0..16 {
    let v0 = (src_block[z] * inv_scale).round().clamp(-8.0, 7.0) as i8;
    let v1 = (src_block[z + 16] * inv_scale).round().clamp(-8.0, 7.0) as i8;
    d[bi].qs[z] = (v0 + 8) as u8 | (((v1 + 8) as u8) << 4);
}
```

### copy_from(t) -> Tensor

텐서의 전체 복사본을 생성합니다. 새 `SharedBuffer`를 할당하고 `memcpy`로 데이터를 복사합니다.

```rust
fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
    let new_buf = SharedBuffer::new(t.size(), t.dtype());
    unsafe {
        std::ptr::copy_nonoverlapping(t.as_ptr(), new_buf.as_mut_ptr(), t.size());
    }
    Ok(Tensor::new(t.shape().clone(), Arc::new(new_buf), Arc::new(CpuBackendCommon)))
}
```

### attention_gen

단일 쿼리 어텐션 (디코드 단계). GQA (Grouped Query Attention)를 지원하며, KV 캐시의 dtype에 따라 세 가지 경로를 처리합니다.

| KV dtype | 처리 방식 |
|----------|-----------|
| F32 | 직접 내적 |
| F16 | F16→F32 변환 후 내적 (aarch64에서 NEON FMA 사용) |
| Q4_0 | 블록 역양자화 후 내적 (aarch64에서 NEON FMA 사용) |

```rust
// 헤드 단위 rayon 병렬화
out_data.par_chunks_mut(head_dim).enumerate().for_each(|(h, out_h)| {
    let kv_h = h / gqa_ratio;  // GQA: 여러 Q head가 같은 KV head를 공유
    // 1. Q * K^T → scores (내적)
    // 2. softmax(scores)
    // 3. scores * V → output (가중 합)
});
```

특이사항: `common.rs`에서도 `#[cfg(target_arch = "aarch64")]`로 조건부 NEON 코드가 포함되어 있습니다. F16/Q4_0 경로에서 K/V를 F32로 변환한 뒤 NEON `vfmaq_f32`를 사용하여 내적과 가중 합을 가속합니다. 이는 16요소 단위 언롤링으로 파이프라인 효율을 높입니다.

---

## 3.3 quantize_row_q8_0

**파일**: `src/backend/cpu/common.rs` (스칼라), `neon.rs` (NEON), `x86.rs` (AVX2)

F32 입력을 Q8_0 블록 배열로 양자화합니다.

### 스칼라 구현

```rust
pub fn quantize_row_q8_0(&self, x: &[f32], y: &mut [BlockQ8_0], k: usize) {
    let nb = k / QK8_0;  // 블록 수

    for i in 0..nb {
        let src = &x[i * QK8_0..(i + 1) * QK8_0];

        // 1. 최대 절댓값 계산
        let mut amax = 0.0f32;
        for &v in src {
            amax = amax.max(v.abs());
        }

        // 2. scale 계산
        let d = amax / 127.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        y[i].d = f16::from_f32(d);

        // 3. 라운딩하여 i8로 변환
        for j in 0..QK8_0 {
            y[i].qs[j] = (src[j] * id).round() as i8;
        }
    }
}
```

알고리즘:
1. 32개 요소에서 최대 절댓값(`amax`) 탐색
2. `scale = amax / 127.0` 계산 (i8 범위에 맞춤)
3. 각 요소를 `round(x / scale)`로 양자화하여 i8에 저장

---

## 3.4 vec_dot_q4_0_q8_0

**파일**: `src/backend/cpu/common.rs` (스칼라)

Q4_0와 Q8_0 블록 간의 내적(dot product)을 계산합니다. 정수 연산으로 수행하여 부동소수점 대비 높은 효율을 달성합니다.

```rust
pub unsafe fn vec_dot_q4_0_q8_0(&self, n: usize, s: &mut f32,
                                 vx: *const BlockQ4_0, vy: *const BlockQ8_0) {
    let nb = n / QK8_0;
    let mut sumf = 0.0;

    for i in 0..nb {
        let x = &*vx.add(i);  // Q4_0 블록
        let y = &*vy.add(i);  // Q8_0 블록

        let d = x.d.to_f32() * y.d.to_f32();  // scale 곱
        let mut isum = 0i32;

        for j in 0..(QK4_0 / 2) {
            // Q4_0 니블 언패킹
            let v0 = (x.qs[j] & 0x0F) as i8 - 8;  // 하위 4비트
            let v1 = (x.qs[j] >> 4) as i8 - 8;     // 상위 4비트

            // Q8_0 값과 정수 곱셈 누적
            isum += (v0 as i32) * (y.qs[j] as i32);
            isum += (v1 as i32) * (y.qs[j + QK4_0 / 2] as i32);
        }

        sumf += d * isum as f32;  // scale 적용
    }
    *s = sumf;
}
```

핵심: 블록별로 정수 내적(`isum`)을 먼저 계산하고, 마지막에 scale 곱(`d`)을 적용합니다. 이렇게 하면 32번의 정수 곱셈-덧셈 + 1번의 부동소수점 곱셈으로 처리할 수 있습니다.

---

## 3.5 CpuBackendNeon SIMD 최적화

**파일**: `src/backend/cpu/neon.rs`

ARM AArch64 NEON 명령어를 활용한 최적화 구현입니다. `matmul_transposed`의 F32와 Q4_0 경로를 최적화합니다.

### matmul_transposed_f32

#### 직렬/병렬 휴리스틱

```rust
// 작은 행렬: 직렬 처리 (Rayon 오버헤드 > 연산 비용)
if (m * n * k) < 100_000 {
    return self.matmul_transposed_f32_serial(a, b, out);
}
```

`M*N*K < 100,000` 이면 직렬 처리를 수행합니다. 벤치마크에서 [1, 128, 256] 같은 작은 행렬에 대해 **3.6배** 빠른 결과가 확인되었습니다.

#### NEON 벡터화 내적

```rust
// 16요소 단위 언롤링
unsafe {
    let mut sum_v = vdupq_n_f32(0.0);

    while k_idx + 16 <= k {
        sum_v = vfmaq_f32(sum_v, vld1q_f32(a_ptr.add(k_idx)),    vld1q_f32(b_ptr.add(k_idx)));
        sum_v = vfmaq_f32(sum_v, vld1q_f32(a_ptr.add(k_idx+4)),  vld1q_f32(b_ptr.add(k_idx+4)));
        sum_v = vfmaq_f32(sum_v, vld1q_f32(a_ptr.add(k_idx+8)),  vld1q_f32(b_ptr.add(k_idx+8)));
        sum_v = vfmaq_f32(sum_v, vld1q_f32(a_ptr.add(k_idx+12)), vld1q_f32(b_ptr.add(k_idx+12)));
        k_idx += 16;
    }

    // 4요소 단위 나머지 처리
    while k_idx + 4 <= k {
        sum_v = vfmaq_f32(sum_v, vld1q_f32(a_ptr.add(k_idx)), vld1q_f32(b_ptr.add(k_idx)));
        k_idx += 4;
    }

    let sum_s = vaddvq_f32(sum_v);  // 수평 합산
    // ... 스칼라 tail 처리
}
```

사용 NEON intrinsic:
- `vld1q_f32`: 4개 f32 로드 (128비트)
- `vfmaq_f32`: Fused Multiply-Add (`acc += a * b`)
- `vaddvq_f32`: 수평 합산 (4개 레인 → 스칼라)

#### 청크 크기 전략

```rust
let num_threads = rayon::current_num_threads();
let chunk_size = (n + num_threads - 1) / num_threads;
let chunk_size = chunk_size.max(256);  // 최소 256 요소/태스크
```

N 차원을 `num_threads`개로 분할하되, 태스크당 최소 256 요소를 보장합니다. 이는 Rayon 스케줄링 오버헤드와 캐시 효율의 균형을 맞춥니다.

### matmul_transposed_q4_0

#### 전체 흐름

1. A를 Q8_0로 양자화 (NEON 가속)
2. `vec_dot_q4_0_q8_0` 또는 `vec_dot_q4_0_q8_0_sdot`로 내적 계산

```rust
// 1. A → Q8_0 양자화
for i in 0..m {
    unsafe { self.quantize_row_q8_0(a_row, q8_row, k); }
}

// 2. 내적 계산 (dotprod 지원 여부에 따라 분기)
unsafe {
    if std::arch::is_aarch64_feature_detected!("dotprod") {
        self.vec_dot_q4_0_q8_0_sdot(k, &mut sum, b_row_ptr, a_row_ptr);
    } else {
        self.vec_dot_q4_0_q8_0(k, &mut sum, b_row_ptr, a_row_ptr);
    }
}
```

#### vec_dot_q4_0_q8_0 (NEON)

`vmull_s8`(와이드닝 곱셈)과 `vaddlvq_s16`(수평 합산)을 사용합니다. 2블록 단위 언롤링으로 파이프라인 활용도를 높입니다.

```rust
// Q4_0 언패킹: 니블 → i8
let v0_0 = vld1q_u8(x.qs.as_ptr());
let x0_l = vsubq_s8(vreinterpretq_s8_u8(vandq_u8(v0_0, m4b)), s8b);  // 하위 니블 - 8
let x0_h = vsubq_s8(vreinterpretq_s8_u8(vshrq_n_u8(v0_0, 4)), s8b);  // 상위 니블 - 8

// Q8_0 로드
let y0_l = vld1q_s8(y.qs.as_ptr());
let y0_h = vld1q_s8(y.qs.as_ptr().add(16));

// 와이드닝 곱셈 + 수평 합산
let mul_l0 = vmull_s8(vget_low_s8(x0_l), vget_low_s8(y0_l));   // i8 x i8 → i16
let mul_l1 = vmull_high_s8(x0_l, y0_l);
let mul_h0 = vmull_s8(vget_low_s8(x0_h), vget_low_s8(y0_h));
let mul_h1 = vmull_high_s8(x0_h, y0_h);
let acc = vaddlvq_s16(mul_l0) + vaddlvq_s16(mul_l1)             // i16 → i32 합산
        + vaddlvq_s16(mul_h0) + vaddlvq_s16(mul_h1);

sumf += d * acc as f32;
```

#### vec_dot_q4_0_q8_0_sdot (ARMv8.2+)

ARMv8.2의 `sdot` 명령어를 인라인 어셈블리로 직접 호출합니다. `sdot`은 4개의 i8 곱셈-누적을 단일 사이클에 수행합니다.

```rust
#[target_feature(enable = "neon,dotprod")]
pub unsafe fn vec_dot_q4_0_q8_0_sdot(&self, ...) {
    let mut acc = vdupq_n_s32(0);

    // sdot: acc[lane] += x[4*lane+0]*y[4*lane+0] + ... + x[4*lane+3]*y[4*lane+3]
    std::arch::asm!(
        "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
        acc = inout(vreg) acc,
        x = in(vreg) x_l,
        y = in(vreg) y_l,
    );
    std::arch::asm!(
        "sdot {acc:v}.4s, {x:v}.16b, {y:v}.16b",
        acc = inout(vreg) acc,
        x = in(vreg) x_h,
        y = in(vreg) y_h,
    );

    let sum_i = vaddvq_s32(acc);
    sumf += d * sum_i as f32;
}
```

`sdot`을 사용하면 `vmull` + `vaddlvq` 조합 대비 명령어 수가 크게 줄어듭니다.

### quantize_row_q8_0 (NEON)

32개 f32를 NEON으로 병렬 처리하여 Q8_0로 양자화합니다.

```rust
#[target_feature(enable = "neon")]
pub unsafe fn quantize_row_q8_0(&self, x: &[f32], y: &mut [BlockQ8_0], k: usize) {
    // 1. 8개 float32x4 벡터 로드 (32개 f32)
    let v0 = vld1q_f32(src_ptr);
    // ... v1 ~ v7

    // 2. 최대 절댓값 계산 (NEON)
    max_abs = vmaxq_f32(max_abs, vabsq_f32(v0));
    // ... v1 ~ v7
    let amax = vmaxvq_f32(max_abs);  // 수평 최댓값

    // 3. scale 및 역수 계산
    let d = amax / 127.0;
    let id_v = vdupq_n_f32(id);

    // 4. f32 → i32 → i16 → i8 변환 (NEON 내로잉 연산)
    let i32_0 = vcvtnq_s32_f32(vmulq_f32(v0, id_v));  // 반올림 변환
    let i16_0 = vqmovn_s32(i32_0);                      // i32 → i16 (포화)
    // ... 결합 후
    let i8_res = vqmovn_s16(i16_combined);               // i16 → i8 (포화)

    vst1q_s8(y[i].qs.as_mut_ptr(), i8_res);
}
```

사용 NEON intrinsic:
- `vmaxvq_f32`: 4레인 수평 최댓값
- `vcvtnq_s32_f32`: f32 → i32 반올림 변환
- `vqmovn_s32`: i32 → i16 포화 내로잉
- `vqmovn_s16`: i16 → i8 포화 내로잉

---

## 3.6 CpuBackendAVX2 최적화

**파일**: `src/backend/cpu/x86.rs`

x86_64 AVX2 + FMA 명령어를 활용한 최적화입니다.

### matmul_transposed_f32

M의 크기에 따라 세 가지 전략으로 분기합니다.

#### M < 8: N 병렬화

```rust
// 32요소 단위 AVX2 FMA 언롤링
let mut sum_v = _mm256_setzero_ps();

while k_idx + 32 <= k {
    sum_v = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(k_idx)),
                            _mm256_loadu_ps(b_ptr.add(k_idx)), sum_v);
    // ... +8, +16, +24
    k_idx += 32;
}

// 8요소 단위 나머지
while k_idx + 8 <= k {
    sum_v = _mm256_fmadd_ps(_mm256_loadu_ps(a_ptr.add(k_idx)),
                            _mm256_loadu_ps(b_ptr.add(k_idx)), sum_v);
    k_idx += 8;
}

// 수평 합산 (store + scalar sum)
let mut temp = [0.0f32; 8];
_mm256_storeu_ps(temp.as_mut_ptr(), sum_v);
let mut sum = temp.iter().sum::<f32>();
```

`par_iter_mut`으로 출력 요소 단위 병렬화합니다. 디코드 단계(M=1)에서 모든 코어를 활용할 수 있습니다.

#### M in [8, 64): 스칼라 Fallback

```rust
if m < 64 {
    return CpuBackendCommon::new().matmul_transposed_f32(a, b, out);
}
```

이 구간에서는 블록 처리(BLOCK_M=8)의 병렬 태스크 수가 너무 적고(M=8이면 1개 태스크), 요소별 병렬화에서는 M이 충분히 커서 행 단위 병렬화로도 괜찮습니다. Common의 `par_chunks_mut(n)` 행 병렬화를 사용합니다.

#### M >= 64: 블록 병렬화 (BLOCK_M=8)

```rust
const BLOCK_M: usize = 8;

out_data.par_chunks_mut(n * BLOCK_M).enumerate().for_each(|(chunk_idx, out_chunk)| {
    // 8개 A 행에 대한 포인터 준비
    let mut a_ptrs = [std::ptr::null::<f32>(); BLOCK_M];

    for j in 0..n {
        let b_ptr = b_data.as_ptr().add(j * k);
        let mut sums = [_mm256_setzero_ps(); BLOCK_M];

        while k_idx + 8 <= k {
            let vb = _mm256_loadu_ps(b_ptr.add(k_idx));
            for r in 0..rows_in_chunk {
                let va = _mm256_loadu_ps(a_ptrs[r].add(k_idx));
                sums[r] = _mm256_fmadd_ps(va, vb, sums[r]);
            }
            k_idx += 8;
        }
        // ... 수평 합산 및 저장
    }
});
```

8행의 A를 동시에 처리하면서 B의 같은 행을 재사용합니다. 이렇게 하면 B 행렬의 메모리 대역폭을 **8배** 절약합니다.

### Q4_0 헬퍼 함수

#### sum_i16_pairs_float

i16 쌍을 합산하여 f32로 변환합니다.

```rust
#[target_feature(enable = "avx2")]
unsafe fn sum_i16_pairs_float(&self, x: __m256i) -> __m256 {
    let ones = _mm256_set1_epi16(1);
    let summed_pairs = _mm256_madd_epi16(ones, x);  // i16 인접 쌍 합산 → i32
    _mm256_cvtepi32_ps(summed_pairs)                 // i32 → f32
}
```

#### mul_sum_i8_pairs_float

부호 있는 i8 곱셈-누적을 수행합니다. `_mm256_maddubs_epi16`은 unsigned x signed 전용이므로, 부호 처리를 위한 트릭을 사용합니다.

```rust
#[target_feature(enable = "avx2")]
unsafe fn mul_sum_i8_pairs_float(&self, x: __m256i, y: __m256i) -> __m256 {
    let ax = _mm256_sign_epi8(x, x);   // |x| (절댓값)
    let sy = _mm256_sign_epi8(y, x);   // x의 부호를 y에 적용
    self.mul_sum_us8_pairs_float(ax, sy) // unsigned(|x|) * signed(sy)
}
```

#### vec_dot_q4_0_q8_0 (AVX2)

```rust
#[target_feature(enable = "avx2", enable = "fma")]
pub unsafe fn vec_dot_q4_0_q8_0(&self, n: usize, s: &mut f32, ...) {
    let mut acc = _mm256_setzero_ps();
    let off = _mm256_set1_epi8(8);

    for i in 0..nb {
        let d = _mm256_set1_ps(x.d.to_f32() * y.d.to_f32());

        // Q4_0 언패킹: bytes_from_nibbles_32 → sub(8)
        let mut qx = self.bytes_from_nibbles_32(x.qs.as_ptr());
        qx = _mm256_sub_epi8(qx, off);

        // Q8_0 로드
        let qy = _mm256_loadu_si256(y.qs.as_ptr() as *const __m256i);

        // i8 곱셈-누적 → f32
        let q = self.mul_sum_i8_pairs_float(qx, qy);

        // FMA 누적
        acc = _mm256_fmadd_ps(d, q, acc);
    }

    *s = self.hsum_float_8(acc);
}
```

`bytes_from_nibbles_32`는 16바이트의 Q4_0 데이터를 32바이트로 확장합니다. 하위 128비트에 하위 니블, 상위 128비트에 상위 니블을 배치하여 `_mm256_set_m128i`로 결합합니다.

### quantize_row_q8_0 (AVX2)

```rust
#[target_feature(enable = "avx2")]
pub unsafe fn quantize_row_q8_0(&self, x: &[f32], y: &mut [BlockQ8_0], k: usize) {
    let perm_idx = _mm256_setr_epi32(0, 4, 1, 5, 2, 6, 3, 7);

    for i in 0..nb {
        // 1. 4개 AVX 레지스터로 32개 f32 로드
        let v0 = _mm256_loadu_ps(src); // ... v1, v2, v3

        // 2. 최대 절댓값 (andnot으로 부호 비트 제거)
        let max_abs = _mm256_andnot_ps(sign_bit, v0);
        // ... reduce → max_scalar

        // 3. 스케일링 및 반올림
        v0 = _mm256_round_ps(_mm256_mul_ps(v0, mul), _MM_ROUND_NEAREST);

        // 4. 패킹: i32 → i16 → i8
        i0 = _mm256_packs_epi32(i0, i1);  // 8xi32 → 16xi16
        i0 = _mm256_packs_epi16(i0, i2);  // 16xi16 → 32xi8

        // 5. 바이트 순서 보정
        i0 = _mm256_permutevar8x32_epi32(i0, perm_idx);

        _mm256_storeu_si256(y[i].qs.as_mut_ptr() as *mut __m256i, i0);
    }
}
```

AVX2의 `_mm256_packs_epi32`는 두 128비트 레인 내에서 독립적으로 패킹하므로, 최종 결과의 바이트 순서가 뒤섞입니다. `_mm256_permutevar8x32_epi32`로 올바른 순서로 재배열합니다.

---

## 3.7 직렬/병렬 휴리스틱 요약

각 연산의 직렬/병렬 분기 기준을 정리합니다.

| 연산 | 기준 | 직렬 동작 | 병렬 동작 |
|------|------|-----------|-----------|
| **matmul_transposed_f32** (NEON) | `M*N*K < 100,000` | 순수 직렬 루프 | N 차원 청크 병렬화 (min 256/chunk) |
| **matmul_transposed_q4_0** (NEON) | `M*N*K < 100,000` | 직렬 Q4_0 역양자화 루프 | Q8_0 양자화 후 청크 병렬 내적 |
| **matmul_transposed_f32** (AVX2) | `M*N*K < 100,000` | 직렬 AVX2 루프 | M<8: 요소별 병렬, M<64: 행 병렬, M>=64: BLOCK_M=8 블록 병렬 |
| **matmul_transposed_q4_0** (AVX2) | `M*N*K < 100,000` | Common fallback (스칼라) | M<4: Q8_0+요소별 병렬, M>=4: Common fallback |
| **matmul_transposed_q4_0** (Common) | `M < 4` | Q8_0 양자화 + 요소별 병렬 | 행 병렬 인라인 역양자화 |
| **matmul** (Common) | 항상 병렬 | - | 행(M) 단위 `par_chunks_mut` |
| **add_assign** | 항상 병렬 | - | 요소별 `par_iter_mut` |
| **scale** | 항상 병렬 | - | 요소별 `par_iter_mut` |
| **silu_mul** | 항상 병렬 | - | 요소별 `par_iter_mut` |
| **rms_norm** | 항상 병렬 | - | 행(dim) 단위 `par_chunks_mut` |
| **softmax** | 항상 병렬 | - | 행(dim) 단위 `par_chunks_mut` |
| **rope_inplace** | 항상 병렬 | - | 배치 단위 `par_chunks_mut` |
| **attention_gen** | 항상 병렬 | - | 헤드(head) 단위 `par_chunks_mut` |
| **cast** (F32↔F16) | 항상 병렬 | - | 요소별 `par_iter_mut` |

일반적으로 `100,000` 임곗값은 Rayon 스레드 풀 스케줄링 오버헤드(수 마이크로초)와 실제 연산 시간의 균형점입니다. 이 값 이하에서는 직렬 처리가 더 빠릅니다.

---

**이전**: [02. Core 추상화](02_core_abstractions.md) | **다음**: [04. 모델 로딩](04_model_loading.md)
