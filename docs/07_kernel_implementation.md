# Chapter 7: Kernel Implementation

> **이전**: [06. OpenCL 백엔드](06_opencl_backend.md) | **다음**: [08. 메모리 관리](08_memory_management.md)

## 7.1 Overview

이 문서는 `kernels/simple_ops.cl` 파일에 구현된 핵심 OpenCL 커널들의 알고리즘과 최적화 기법을 설명합니다.

### 커널 목록
| Kernel | Purpose | Optimization |
|--------|---------|--------------|
| `kernel_rms_norm_opt` | RMS Normalization | Subgroup + Local memory reduction |
| `kernel_softmax_opt` | Softmax | Subgroup + Local memory reduction |
| `kernel_rope_simple` | Rotary Position Embedding | Per-element parallelism |
| `kernel_silu_mul_simple` | SiLU(x) * y | Per-element parallelism |
| `kernel_add_assign_simple` | x += y | Per-element parallelism |
| `kernel_scale_simple` | x *= scalar | Per-element parallelism |
| `kernel_attn_gen` | Single-query Attention | Workgroup reduction |

---

## 7.2 RMS Normalization Kernel

### 알고리즘
```
RMSNorm(x) = x / sqrt(mean(x²) + eps) * weight
```

### 커널 시그니처
```c
kernel void kernel_rms_norm_opt(
    global float * x,        // [rows, dim] - inplace
    global float * weight,   // [dim]
    int dim,
    float eps,
    local float * scratch    // [local_size]
)
```

### Step-by-Step 구현
```c
// 1. 행(row) 할당
int row = get_group_id(0);
int lid = get_local_id(0);
global float * row_ptr = x + row * dim;

// 2. 제곱합 계산 (각 스레드가 dim/local_size 개씩)
float sum_sq = 0.0f;
for (int i = lid; i < dim; i += local_size) {
    float val = row_ptr[i];
    sum_sq += val * val;
}

// 3. Subgroup 리덕션
sum_sq = sub_group_reduce_add(sum_sq);

// 4. Local memory에 저장 (subgroup당 1개)
if (get_sub_group_local_id() == 0) {
    scratch[get_sub_group_id()] = sum_sq;
}
barrier(CLK_LOCAL_MEM_FENCE);

// 5. 첫 번째 subgroup에서 최종 리덕션
if (get_sub_group_id() == 0) {
    float val = (get_sub_group_local_id() < num_subgroups) 
              ? scratch[get_sub_group_local_id()] : 0.0f;
    sum_sq = sub_group_reduce_add(val);
}

// 6. 결과를 모든 스레드에 브로드캐스트
if (lid == 0) scratch[0] = sum_sq;
barrier(CLK_LOCAL_MEM_FENCE);
sum_sq = scratch[0];

// 7. 정규화 적용
float rms = sqrt(sum_sq / (float)dim + eps);
float scale = 1.0f / rms;

for (int i = lid; i < dim; i += local_size) {
    row_ptr[i] = row_ptr[i] * scale * weight[i];
}
```

### 워크그룹 설정
```
Global: [rows * 64, 1, 1]
Local:  [64, 1, 1]          # Adreno subgroup size
```

---

## 7.3 Softmax Kernel

### 알고리즘
```
softmax(x) = exp(x - max(x)) / sum(exp(x - max(x)))
```

### 커널 시그니처
```c
kernel void kernel_softmax_opt(
    global float * x,        // [rows, dim] - inplace
    int dim,
    local float * scratch
)
```

### 3-Phase 구현

#### Phase 1: Max 찾기
```c
float my_max = -INFINITY;
for (int i = lid; i < dim; i += local_size) {
    my_max = fmax(my_max, x[row * dim + i]);
}
// Subgroup + Local reduction으로 전체 max 계산
```

#### Phase 2: Exp 합계 계산
```c
float my_sum = 0.0f;
for (int i = lid; i < dim; i += local_size) {
    my_sum += exp(x[row * dim + i] - max_val);
}
// Reduction으로 전체 합계
```

#### Phase 3: 정규화
```c
for (int i = lid; i < dim; i += local_size) {
    x[row * dim + i] = exp(x[row * dim + i] - max_val) / sum_val;
}
```

---

## 7.4 RoPE Kernel (Rotary Position Embedding)

### 알고리즘
```
x[2i]   = x[2i]   * cos(θ) - x[2i+1] * sin(θ)
x[2i+1] = x[2i]   * sin(θ) + x[2i+1] * cos(θ)

θ = pos * (theta ^ (-2i/dim))
```

### 커널 시그니처
```c
kernel void kernel_rope_simple(
    global float * x,       // [..., seq_len, num_heads, head_dim]
    int head_dim,
    int num_heads,
    int seq_len,
    int start_pos,
    float theta
)
```

### 구현
```c
int global_id = get_global_id(0);
int pair_idx = global_id % (head_dim / 2);  // 0..head_dim/2
int head = (global_id / (head_dim / 2)) % num_heads;
int seq = global_id / (head_dim / 2 * num_heads);

int pos = start_pos + seq;
float freq = pow(theta, -(2.0f * pair_idx) / (float)head_dim);
float angle = pos * freq;
float cos_val = cos(angle);
float sin_val = sin(angle);

// 원본 값 읽기
int idx = seq * num_heads * head_dim + head * head_dim;
float x0 = x[idx + pair_idx * 2];
float x1 = x[idx + pair_idx * 2 + 1];

// 회전 적용
x[idx + pair_idx * 2]     = x0 * cos_val - x1 * sin_val;
x[idx + pair_idx * 2 + 1] = x0 * sin_val + x1 * cos_val;
```

### 워크그룹 설정
```
Global: [seq_len * num_heads * (head_dim / 2), 1, 1]
Local:  None (per-element parallelism)
```

---

## 7.5 Simple Element-wise Kernels

### SiLU Multiplication
```c
kernel void kernel_silu_mul_simple(
    global float * x,  // inplace
    global float * y,
    int size
) {
    int i = get_global_id(0);
    if (i < size) {
        float val = x[i];
        float sigmoid = 1.0f / (1.0f + exp(-val));
        x[i] = val * sigmoid * y[i];  // SiLU(x) * y
    }
}
```

### Add Assign
```c
kernel void kernel_add_assign_simple(
    global float * x,
    global float * y,
    int size
) {
    int i = get_global_id(0);
    if (i < size) {
        x[i] += y[i];
    }
}
```

### Scale
```c
kernel void kernel_scale_simple(
    global float * x,
    float scale,
    int size
) {
    int i = get_global_id(0);
    if (i < size) {
        x[i] *= scale;
    }
}
```

---

## 7.6 Matrix-Vector Multiplication (Q4_0)

### 파일 위치
`kernels/mul_mv_q4_0_f32.cl`

### 핵심 개념
- **B 행렬**: Q4_0 양자화 (18 bytes per 32 elements)
- **A 행렬**: F32 활성화
- **Output**: F32

### Q4_0 블록 구조
```c
// 메모리 레이아웃 (18 bytes total)
half d;           // 2 bytes: scale
uchar qs[16];     // 16 bytes: 32 x 4-bit values (packed)
```

### Dequantization
```c
// qs[j]에서 2개의 4-bit 값 추출
uchar packed = qs[j];
int val0 = (packed & 0x0F) - 8;  // 하위 4비트
int val1 = (packed >> 4) - 8;     // 상위 4비트

float dequant0 = (float)val0 * d;
float dequant1 = (float)val1 * d;
```

### 워크그룹 설정
```
Global: [((N+3)/4) * 64, M, 1]
Local:  [64, 1, 1]
```

---

## 7.7 Adreno-Specific Optimizations

### Subgroup Extensions
```c
#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif
```

### 커널 어노테이션
```c
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_xxx(...) {
    // Subgroup size = 64 보장
}
```

### 이점
- `sub_group_reduce_add()`: 64개 값을 하드웨어에서 1사이클에 합산
- Local memory barrier 최소화
- 레지스터 활용 극대화
