# Chapter 9: GPU Attention Mechanism

> **이전**: [08. 메모리 관리](08_memory_management.md) | **다음**: [10. 모델 추론](10_model_inference.md)

## 9.1 Overview

이 문서는 GPU에서 실행되는 단일 쿼리 Attention 커널의 구현을 설명합니다.

### 핵심 성과
- **CPU 폴백 제거**: GPU→CPU→GPU 데이터 전송 완전 제거
- **성능 향상**: 36% TBT 개선 (26.16ms → 19.19ms)

---

## 9.2 Attention 알고리즘

### 수식
```
Attention(Q, K, V) = softmax(Q * K^T / sqrt(d_k)) * V
```

### 단일 쿼리 생성 (seq_len = 1)
```
Q: [num_heads_q, head_dim]
K: [cache_seq_len, num_heads_kv, head_dim]  
V: [cache_seq_len, num_heads_kv, head_dim]
Output: [num_heads_q, head_dim]
```

### GQA (Grouped Query Attention)
```
num_heads_q = 32  (query heads)
num_heads_kv = 8   (key/value heads)
gqa_ratio = 32/8 = 4

// Q head 0-3 → KV head 0
// Q head 4-7 → KV head 1
// ...
kv_head = q_head / gqa_ratio
```

---

## 9.3 커널 시그니처

### 파일 위치
`kernels/simple_ops.cl` (kernel_attn_gen)

### 시그니처
```c
kernel void kernel_attn_gen(
    global float * Q,            // [num_heads_q, head_dim]
    global float * K,            // [cache_seq_len, num_heads_kv, head_dim]
    global float * V,            // [cache_seq_len, num_heads_kv, head_dim]
    global float * O,            // [num_heads_q, head_dim]
    int head_dim,
    int num_heads_q,
    int num_heads_kv,
    int cache_seq_len,
    float scale,
    local float * scratch        // [local_size]
)
```

---

## 9.4 커널 구현 상세

### 워크그룹 할당
```
Global: [num_heads_q * 64, 1, 1]
Local:  [64, 1, 1]

// 각 workgroup = 1개의 Q head 처리
int head_idx = get_group_id(0);
int lid = get_local_id(0);
```

### Phase 1: Score 계산 및 Max 찾기
```c
// GQA 매핑
int gqa_ratio = num_heads_q / num_heads_kv;
int kv_head = head_idx / gqa_ratio;

global float * q_ptr = Q + head_idx * head_dim;

// 각 스레드가 cache_seq_len / local_size 개의 토큰 처리
float my_max = -INFINITY;
for (int t = lid; t < cache_seq_len; t += local_size) {
    global float * k_ptr = K + (t * num_heads_kv + kv_head) * head_dim;
    
    // Dot product: Q · K
    float score = 0.0f;
    for (int d = 0; d < head_dim; d++) {
        score += q_ptr[d] * k_ptr[d];
    }
    score *= scale;  // 1/sqrt(head_dim)
    
    my_max = fmax(my_max, score);
}
```

### Phase 1b: Max Reduction
```c
// Local memory reduction
scratch[lid] = my_max;
barrier(CLK_LOCAL_MEM_FENCE);

for (int s = local_size / 2; s > 0; s >>= 1) {
    if (lid < s) {
        scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}

float max_score = scratch[0];
barrier(CLK_LOCAL_MEM_FENCE);
```

### Phase 2: Exp Sum 계산
```c
float my_sum = 0.0f;
for (int t = lid; t < cache_seq_len; t += local_size) {
    // 다시 score 계산 (레지스터 부족으로 저장 못함)
    float score = dot_product(Q, K[t]) * scale;
    my_sum += exp(score - max_score);
}

// Reduction
scratch[lid] = my_sum;
barrier(CLK_LOCAL_MEM_FENCE);
// ... reduction code ...
float total_sum = scratch[0];
```

### Phase 3: Weighted V Sum
```c
// 각 스레드가 head_dim / local_size 개의 출력 원소 담당
for (int d = lid; d < head_dim; d += local_size) {
    float out_val = 0.0f;
    
    for (int t = 0; t < cache_seq_len; t++) {
        // Score 재계산
        float score = dot_product(Q, K[t]) * scale;
        float weight = exp(score - max_score) / total_sum;
        
        global float * v_ptr = V + (t * num_heads_kv + kv_head) * head_dim;
        out_val += weight * v_ptr[d];
    }
    
    O[head_idx * head_dim + d] = out_val;
}
```

---

## 9.5 Host 측 디스패치

### 파일 위치
`src/backend/opencl/mod.rs`

### 구현
```rust
fn attention_gen(
    &self, 
    q: &Tensor, 
    k_cache: &Tensor, 
    v_cache: &Tensor, 
    out: &mut Tensor,
    num_heads_q: usize, 
    num_heads_kv: usize, 
    head_dim: usize, 
    cache_seq_len: usize
) -> Result<()> {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let local_size = 64usize;
    let local_mem_size = local_size * std::mem::size_of::<f32>();
    
    let kernels = self.kernels.lock()?;
    let kernel = &kernels.kernel_attn_gen;
    
    unsafe {
        // 버퍼 인자
        set_kernel_arg(kernel, 0, ArgVal::mem(q_buf))?;
        set_kernel_arg(kernel, 1, ArgVal::mem(k_buf))?;
        set_kernel_arg(kernel, 2, ArgVal::mem(v_buf))?;
        set_kernel_arg(kernel, 3, ArgVal::mem(o_buf))?;
        
        // 정수 인자
        set_kernel_arg(kernel, 4, ArgVal::scalar(&(head_dim as i32)))?;
        set_kernel_arg(kernel, 5, ArgVal::scalar(&(num_heads_q as i32)))?;
        set_kernel_arg(kernel, 6, ArgVal::scalar(&(num_heads_kv as i32)))?;
        set_kernel_arg(kernel, 7, ArgVal::scalar(&(cache_seq_len as i32)))?;
        set_kernel_arg(kernel, 8, ArgVal::scalar(&scale))?;
        
        // Local memory
        set_kernel_arg(kernel, 9, ArgVal::local::<f32>(&local_mem_size))?;
        
        // 디스패치
        let global = [num_heads_q * local_size, 1, 1];
        let local = [local_size, 1, 1];
        enqueue_kernel(&queue, kernel, 1, None, &global, Some(local), ...)?;
    }
    Ok(())
}
```

---

## 9.6 Layer 통합

### 파일 위치
`src/layers/llama_layer.rs`

### 이전 구현 (CPU 폴백)
```rust
// ❌ 성능 병목
if is_opencl {
    backend.read_buffer(&q_rope, &mut q_vec)?;  // GPU→CPU
    backend.read_buffer(&k_cache, &mut k_vec)?; // GPU→CPU (큼!)
    backend.read_buffer(&v_cache, &mut v_vec)?; // GPU→CPU (큼!)
    
    // CPU에서 attention 계산
    for h in 0..n_heads_q { ... }
    
    // CPU→GPU
    ws.out_attn = backend.copy_from(&cpu_out)?;
}
```

### 현재 구현 (GPU 직접)
```rust
// ✅ 최적화됨
if backend.name() == "OpenCL" {
    backend.attention_gen(
        &q_rope, &k_cache, &v_cache, &mut ws.out_attn,
        n_heads_q, n_heads_kv, head_dim, cache_seq_len
    )?;
} else {
    // CPU 백엔드용 폴백
    // ... CPU attention code ...
}
```

---

## 9.7 성능 분석

### 데이터 전송 제거 효과
```
이전:
  GPU→CPU: Q (32KB) + K_cache (1MB+) + V_cache (1MB+)
  CPU 계산: O(heads * cache_len * dim²)
  CPU→GPU: Output (32KB)
  
  데이터 전송: ~2MB+
  시간: 10-20ms

현재:
  GPU 커널: O(heads * cache_len * dim²)
  데이터 전송: 0
  시간: 2-3ms
```

### 벤치마크 결과
| Metric | CPU Fallback | GPU Attention | 개선 |
|--------|--------------|---------------|------|
| Avg TBT | 26.16 ms | 19.19 ms | 36% ↓ |
| Attention Time | ~18 ms | ~2 ms | 89% ↓ |

---

## 9.8 알려진 제한사항

### 현재 구현의 한계
1. **Score 재계산**: 메모리 부족으로 Phase 2-3에서 score를 다시 계산
   - 개선 방안: Local memory에 score 캐싱 (local_size > cache_seq_len 필요)

2. **연속 메모리 요구**: K/V cache가 연속 메모리여야 함
   - 현재 구현으로 충족됨

3. **Causal Mask 미구현**: 현재 generation 전용 (prefill에서는 사용 안 함)

### 향후 최적화 가능
- Flash Attention 알고리즘 적용
- Tiling을 통한 메모리 효율 개선
- Half precision (FP16) 지원
