// llm.rs CUDA custom kernels -- compiled at runtime via NVRTC.
// All kernels use extern "C" for name-mangling-free symbol lookup.
// Targeting SM >= 7.2 (Jetson Xavier+).

#include <cuda_fp16.h>
#include <math_constants.h>

#ifndef INFINITY
#define INFINITY CUDART_INF_F
#endif

#define WARP_SIZE 32

// =================================================================
// Warp/Block reduction infrastructure
// =================================================================

__device__ __forceinline__ float warp_reduce_sum(float x) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        x += __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE);
    return x;
}

__device__ __forceinline__ float warp_reduce_max(float x) {
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, WARP_SIZE));
    return x;
}

// Block-wide reduction using shared memory for inter-warp communication.
// Returns the reduced value to all threads (broadcast from thread 0).
__device__ __forceinline__ float block_reduce_sum(float val) {
    __shared__ float brs_warps[32]; // max 32 warps per block (1024 threads)
    __shared__ float brs_result;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_sum(val);
    if (lane == 0) brs_warps[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < (unsigned)num_warps) ? brs_warps[threadIdx.x] : 0.0f;
    if (warp_id == 0) val = warp_reduce_sum(val);

    if (threadIdx.x == 0) brs_result = val;
    __syncthreads();
    return brs_result;
}

__device__ __forceinline__ float block_reduce_max(float val) {
    __shared__ float brm_warps[32];
    __shared__ float brm_result;
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    int num_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    val = warp_reduce_max(val);
    if (lane == 0) brm_warps[warp_id] = val;
    __syncthreads();

    val = (threadIdx.x < (unsigned)num_warps) ? brm_warps[threadIdx.x] : -INFINITY;
    if (warp_id == 0) val = warp_reduce_max(val);

    if (threadIdx.x == 0) brm_result = val;
    __syncthreads();
    return brm_result;
}

// =================================================================
// 1. RMS Norm (1 block = 1 row)
// =================================================================
// grid=(nrows,1,1), block=(min(ncols,1024),1,1)
// weight multiply included. add_unit support for Gemma3.
extern "C" __global__ void rms_norm_f32(
    float * dst, const float * x, const float * weight,
    int ncols, float eps, int add_unit)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    const float * row_x = x + (long long)row * ncols;
    float * row_dst = dst + (long long)row * ncols;

    // Thread 0 computes the full RMS (simple, no reduction needed for correctness)
    __shared__ float rms_inv_shared;
    if (tid == 0) {
        float sum_sq = 0.0f;
        for (int i = 0; i < ncols; i++) {
            float v = row_x[i];
            sum_sq += v * v;
        }
        rms_inv_shared = rsqrtf(sum_sq / (float)ncols + eps);
    }
    __syncthreads();

    float rms_inv = rms_inv_shared;

    // Apply: dst = x * rms_inv * weight
    for (int i = tid; i < ncols; i += blockDim.x) {
        float w = add_unit ? (1.0f + weight[i]) : weight[i];
        row_dst[i] = row_x[i] * rms_inv * w;
    }
}

// =================================================================
// 2. RoPE inplace (neox style, Llama/Qwen)
// =================================================================
// Supports both decode (seq_len=1) and prefill (seq_len>1).
// grid=(seq_len * n_heads, 1, 1), block=(head_dim/2, 1, 1)
// x layout: [seq_len, n_heads, head_dim] (flattened)
// Each position t gets rotation angle: (start_pos + t) * freq_i
extern "C" __global__ void rope_inplace_f32(
    float * x, int head_dim, int n_heads, int seq_len,
    int start_pos, float theta_base)
{
    int block_id = blockIdx.x;          // 0 .. seq_len*n_heads - 1
    int t = block_id / n_heads;         // sequence position within batch
    int h = block_id % n_heads;         // head index
    int i = threadIdx.x;                // dim pair index: 0..head_dim/2
    if (i >= head_dim / 2) return;

    // freq = theta_base^(-2i/head_dim) = exp(-2i/head_dim * log(theta_base))
    // Using exp+log matches CPU's floating point behavior better than powf.
    float freq = expf(-2.0f * (float)i / (float)head_dim * logf(theta_base));
    float theta = (float)(start_pos + t) * freq;
    float cos_t = cosf(theta);
    float sin_t = sinf(theta);

    int base = (t * n_heads + h) * head_dim;
    float x0 = x[base + 2 * i];
    float x1 = x[base + 2 * i + 1];
    x[base + 2 * i]     = x0 * cos_t - x1 * sin_t;
    x[base + 2 * i + 1] = x0 * sin_t + x1 * cos_t;
}

// =================================================================
// 3. Softmax (1 block = 1 row)
// =================================================================
// grid=(nrows,1,1), block=(min(ncols,1024),1,1)
extern "C" __global__ void softmax_f32(float * x, int ncols)
{
    int row = blockIdx.x;
    int tid = threadIdx.x;
    float * row_x = x + row * ncols;

    // Pass 1: find max
    float max_val = -INFINITY;
    for (int i = tid; i < ncols; i += blockDim.x)
        max_val = fmaxf(max_val, row_x[i]);
    max_val = block_reduce_max(max_val);

    // Pass 2: exp(x - max) + sum
    float sum_exp = 0.0f;
    for (int i = tid; i < ncols; i += blockDim.x) {
        float e = expf(row_x[i] - max_val);
        row_x[i] = e;
        sum_exp += e;
    }
    sum_exp = block_reduce_sum(sum_exp);

    // Pass 3: normalize
    float inv_sum = 1.0f / sum_exp;
    for (int i = tid; i < ncols; i += blockDim.x)
        row_x[i] *= inv_sum;
}

// =================================================================
// 4. SiLU-Mul fused (SwiGLU: silu(gate) * up)
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void silu_mul_f32(
    float * gate, const float * up, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    float g = gate[i];
    gate[i] = (g / (1.0f + expf(-g))) * up[i];
}

// =================================================================
// 5. GELU tanh mul (Gemma3 FFN)
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void gelu_tanh_mul_f32(
    float * gate, const float * up, int k)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    float x = gate[i];
    // gelu_tanh(x) = 0.5*x*(1+tanh(sqrt(2/pi)*(x+0.044715*x^3)))
    float t = tanhf(0.7978845608f * (x + 0.044715f * x * x * x));
    gate[i] = 0.5f * x * (1.0f + t) * up[i];
}

// =================================================================
// 6. Add assign (element-wise)
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void add_assign_f32(float * a, const float * b, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    a[i] += b[i];
}

// =================================================================
// 7. Scale (element-wise)
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void scale_f32(float * x, float v, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    x[i] *= v;
}

// =================================================================
// 8. Cast F32 -> F16
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void cast_f32_to_f16(const float * src, half * dst, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    dst[i] = __float2half(src[i]);
}

// =================================================================
// 9. Cast F16 -> F32
// =================================================================
// grid=(ceil(k/256),1,1), block=(256,1,1)
extern "C" __global__ void cast_f16_to_f32(const half * src, float * dst, int k) {
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= k) return;
    dst[i] = __half2float(src[i]);
}

// =================================================================
// 10. Add row bias
// =================================================================
// grid=(ceil(nrows*ncols/256),1,1), block=(256,1,1)
extern "C" __global__ void add_row_bias_f32(
    float * x, const float * bias, int ncols, int total)
{
    int i = blockDim.x * blockIdx.x + threadIdx.x;
    if (i >= total) return;
    x[i] += bias[i % ncols];
}

// =================================================================
// 11. KV scatter F32->F16 (HeadMajor layout)
// =================================================================
// grid=(kv_heads, 1, 1), block=(head_dim, 1, 1)
extern "C" __global__ void kv_scatter_f32_to_f16(
    const float * k_src, const float * v_src,
    half * k_dst, half * v_dst,
    int head_dim, int capacity, int write_pos)
{
    int h = blockIdx.x;
    int d = threadIdx.x;
    if (d >= head_dim) return;
    int src_off = h * head_dim + d;
    int dst_off = h * capacity * head_dim + write_pos * head_dim + d;
    k_dst[dst_off] = __float2half(k_src[src_off]);
    v_dst[dst_off] = __float2half(v_src[src_off]);
}

// =================================================================
// 12. Gather F16 -> F32 (embedding lookup)
// =================================================================
// grid=(n_tokens, ceil(dim/256), 1), block=(256, 1, 1)
extern "C" __global__ void gather_f16(
    const half * embed, const int * indices, float * dst, int dim)
{
    int token = blockIdx.x;
    int col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= dim) return;
    int row = indices[token];
    dst[token * dim + col] = __half2float(embed[row * dim + col]);
}

// =================================================================
// 13. Attention gen NAIVE (single query decode, F32 KV cache)
// =================================================================
// Retained as fallback. Prefer flash_attn_decode_f32kv.
// grid=(n_heads_q, 1, 1), block=(BLOCK_SIZE, 1, 1)
// HeadMajor layout: k_cache[kv_h, pos, d] = k_cache[kv_h * capacity * head_dim + pos * head_dim + d]
// shared memory: cache_seq_len floats for scores + 33 floats for reduction scratch
extern "C" __global__ void attention_gen_f32_naive(
    const float * q,
    const float * k_cache,
    const float * v_cache,
    float * out,
    int n_heads_q, int kv_heads, int head_dim,
    int capacity, int cache_seq_len)
{
    int h = blockIdx.x;
    int kv_h = h / (n_heads_q / kv_heads);
    int tid = threadIdx.x;
    float scale = rsqrtf((float)head_dim);

    const float * q_vec = q + h * head_dim;
    const float * k_base = k_cache + kv_h * capacity * head_dim;
    const float * v_base = v_cache + kv_h * capacity * head_dim;

    // Dynamic shared memory: scores[cache_seq_len] + scratch[33]
    extern __shared__ float shmem[];
    float * scores = shmem;

    // Phase 1: QK^T scores (each thread handles multiple positions via stride)
    for (int t = tid; t < cache_seq_len; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_vec[d] * k_base[t * head_dim + d];
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: Softmax over scores
    // 2a: find max
    float max_val = -INFINITY;
    for (int t = tid; t < cache_seq_len; t += blockDim.x)
        max_val = fmaxf(max_val, scores[t]);
    max_val = block_reduce_max(max_val);

    // 2b: exp + sum
    float sum_exp = 0.0f;
    for (int t = tid; t < cache_seq_len; t += blockDim.x) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        sum_exp += e;
    }
    sum_exp = block_reduce_sum(sum_exp);

    // 2c: normalize
    float inv_sum = 1.0f / sum_exp;
    for (int t = tid; t < cache_seq_len; t += blockDim.x)
        scores[t] *= inv_sum;
    __syncthreads();

    // Phase 3: Weighted sum of V -- each thread handles a subset of head_dim
    float * out_vec = out + h * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < cache_seq_len; t++) {
            acc += scores[t] * v_base[t * head_dim + d];
        }
        out_vec[d] = acc;
    }
}

// =================================================================
// 14. Attention gen NAIVE (single query decode, F16 KV cache)
// =================================================================
// Retained as fallback. Prefer flash_attn_decode_f16kv.
extern "C" __global__ void attention_gen_f16kv_naive(
    const float * q,
    const half * k_cache,
    const half * v_cache,
    float * out,
    int n_heads_q, int kv_heads, int head_dim,
    int capacity, int cache_seq_len)
{
    int h = blockIdx.x;
    int kv_h = h / (n_heads_q / kv_heads);
    int tid = threadIdx.x;
    float scale = rsqrtf((float)head_dim);

    const float * q_vec = q + h * head_dim;
    const half * k_base = k_cache + kv_h * capacity * head_dim;
    const half * v_base = v_cache + kv_h * capacity * head_dim;

    extern __shared__ float shmem[];
    float * scores = shmem;

    // Phase 1: QK^T (F16 K dequantized on-the-fly)
    for (int t = tid; t < cache_seq_len; t += blockDim.x) {
        float dot = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            dot += q_vec[d] * __half2float(k_base[t * head_dim + d]);
        }
        scores[t] = dot * scale;
    }
    __syncthreads();

    // Phase 2: Softmax
    float max_val = -INFINITY;
    for (int t = tid; t < cache_seq_len; t += blockDim.x)
        max_val = fmaxf(max_val, scores[t]);
    max_val = block_reduce_max(max_val);

    float sum_exp = 0.0f;
    for (int t = tid; t < cache_seq_len; t += blockDim.x) {
        float e = expf(scores[t] - max_val);
        scores[t] = e;
        sum_exp += e;
    }
    sum_exp = block_reduce_sum(sum_exp);

    float inv_sum = 1.0f / sum_exp;
    for (int t = tid; t < cache_seq_len; t += blockDim.x)
        scores[t] *= inv_sum;
    __syncthreads();

    // Phase 3: V weighted sum (F16 V dequantized on-the-fly)
    float * out_vec = out + h * head_dim;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < cache_seq_len; t++) {
            acc += scores[t] * __half2float(v_base[t * head_dim + d]);
        }
        out_vec[d] = acc;
    }
}
