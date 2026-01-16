// Optimized OpenCL kernels for llm_rs2 using local memory reductions
// Based on llama.cpp kernel patterns

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#ifdef cl_intel_subgroups
#pragma OPENCL EXTENSION cl_intel_subgroups : enable
#else
#pragma OPENCL EXTENSION cl_khr_subgroups : enable
#endif

#ifdef cl_intel_required_subgroup_size
#pragma OPENCL EXTENSION cl_intel_required_subgroup_size : enable
#define INTEL_GPU 1
#define REQD_SUBGROUP_SIZE_16 __attribute__((intel_reqd_sub_group_size(16)))
#define REQD_SUBGROUP_SIZE_32 __attribute__((intel_reqd_sub_group_size(32)))
#elif defined(cl_qcom_reqd_sub_group_size)
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#define REQD_SUBGROUP_SIZE_128 __attribute__((qcom_reqd_sub_group_size("full")))
#endif

//------------------------------------------------------------------------------
// Optimized RMS Norm with local memory reduction (following llama.cpp pattern)
// Input x: [rows, dim], weight: [dim], output: x (inplace)
// Each workgroup handles one row, uses local memory for reduction
//------------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_rms_norm_opt(
    global float * x,
    global float * weight,
    int dim,
    float eps,
    local float * scratch  // local memory for reduction
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    global float * row_ptr = x + row * dim;
    
    // Phase 1: Parallel sum of squares
    float sum_sq = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float val = row_ptr[i];
        sum_sq += val * val;
    }
    
    // Subgroup reduction first
    sum_sq = sub_group_reduce_add(sum_sq);
    
    // Store subgroup results to local memory
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Final reduction in first subgroup
    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        sum_sq = sub_group_reduce_add(val);
    }
    
    // Broadcast final result
    if (lid == 0) {
        scratch[0] = sum_sq;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    sum_sq = scratch[0];
    
    // Phase 2: Apply normalization
    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;
    
    for (int i = lid; i < dim; i += local_size) {
        row_ptr[i] = row_ptr[i] * scale * weight[i];
    }
}

//------------------------------------------------------------------------------
// Optimized Softmax with local memory reduction
// Input x: [rows, dim], output: x (inplace)
// Each workgroup handles one row
//------------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_softmax_opt(
    global float * x,
    int dim,
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    global float * row_ptr = x + row * dim;
    
    // Phase 1: Find max
    float lmax = -INFINITY;
    for (int i = lid; i < dim; i += local_size) {
        lmax = fmax(lmax, row_ptr[i]);
    }
    
    // Subgroup reduction for max
    lmax = sub_group_reduce_max(lmax);
    
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = lmax;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Final max reduction
    int num_subgroups = local_size / get_max_sub_group_size();
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : -INFINITY;
        lmax = sub_group_reduce_max(val);
    }
    
    if (lid == 0) {
        scratch[0] = lmax;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float max_val = scratch[0];
    
    // Phase 2: Compute exp and sum
    float lsum = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float exp_val = exp(row_ptr[i] - max_val);
        row_ptr[i] = exp_val;
        lsum += exp_val;
    }
    
    // Subgroup reduction for sum
    lsum = sub_group_reduce_add(lsum);
    
    if (get_sub_group_local_id() == 0) {
        scratch[get_sub_group_id()] = lsum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    
    if (get_sub_group_id() == 0) {
        float val = (get_sub_group_local_id() < num_subgroups) ? scratch[get_sub_group_local_id()] : 0.0f;
        lsum = sub_group_reduce_add(val);
    }
    
    if (lid == 0) {
        scratch[0] = lsum;
    }
    barrier(CLK_LOCAL_MEM_FENCE);
    float sum_exp = scratch[0];
    
    // Phase 3: Normalize
    for (int i = lid; i < dim; i += local_size) {
        row_ptr[i] /= sum_exp;
    }
}

//------------------------------------------------------------------------------
// Optimized RoPE inplace using neox-style layout
// x: [batch * seq * num_heads * head_dim]
// This kernel processes pairs (x[i], x[i + head_dim/2])
//------------------------------------------------------------------------------
kernel void kernel_rope_opt(
    global float * x,
    int head_dim,
    int num_heads,
    int seq_len,
    int start_pos,
    float theta_base
) {
    int gid = get_global_id(0);
    int half_dim = head_dim / 2;
    
    // Calculate position within the tensor
    int pair_idx = gid % half_dim;            // which pair within head_dim
    int head_idx = (gid / half_dim) % num_heads;
    int seq_idx = (gid / (half_dim * num_heads)) % seq_len;
    int batch_idx = gid / (half_dim * num_heads * seq_len);
    
    int pos = start_pos + seq_idx;
    
    // Calculate frequency for this pair
    float freq = 1.0f / pow(theta_base, (float)(pair_idx * 2) / (float)head_dim);
    float angle = (float)pos * freq;
    
    float cos_val = cos(angle);
    float sin_val = sin(angle);
    
    // Calculate indices into the tensor (neox layout)
    int base_offset = batch_idx * (seq_len * num_heads * head_dim) +
                      seq_idx * (num_heads * head_dim) +
                      head_idx * head_dim;
    
    int i0 = base_offset + pair_idx;
    int i1 = base_offset + pair_idx + half_dim;
    
    float x0 = x[i0];
    float x1 = x[i1];
    
    x[i0] = x0 * cos_val - x1 * sin_val;
    x[i1] = x0 * sin_val + x1 * cos_val;
}

//------------------------------------------------------------------------------
// Scale: x[i] *= scale (vectorized float4)
//------------------------------------------------------------------------------
kernel void kernel_scale_opt(
    global float4 * x,
    float scale,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        x[i] *= scale;
    }
}

//------------------------------------------------------------------------------
// Add assign: x[i] += y[i] (vectorized float4)
//------------------------------------------------------------------------------
kernel void kernel_add_assign_opt(
    global float4 * x,
    global float4 * y,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        x[i] += y[i];
    }
}

//------------------------------------------------------------------------------
// SiLU * mul: x[i] = silu(x[i]) * y[i] (vectorized float4)
//------------------------------------------------------------------------------
kernel void kernel_silu_mul_opt(
    global float4 * x,
    global float4 * y,
    int size4
) {
    int i = get_global_id(0);
    if (i < size4) {
        float4 val = x[i];
        float4 sigmoid = 1.0f / (1.0f + exp(-val));
        x[i] = val * sigmoid * y[i];
    }
}

//------------------------------------------------------------------------------
// Simple fallback versions (no subgroup operations)
//------------------------------------------------------------------------------

// Simple RMS Norm - one thread per row (for small dims or fallback)
kernel void kernel_rms_norm_simple(
    global float * x,
    global float * weight,
    global float * output,
    int dim,
    float eps
) {
    int row = get_global_id(0);
    
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = x[row * dim + i];
        sum_sq += val * val;
    }
    
    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;
    
    for (int i = 0; i < dim; i++) {
        output[row * dim + i] = x[row * dim + i] * scale * weight[i];
    }
}

kernel void kernel_softmax_simple(
    global float * x,
    global float * output,
    int dim
) {
    int row = get_global_id(0);
    
    float max_val = -INFINITY;
    for (int i = 0; i < dim; i++) {
        max_val = fmax(max_val, x[row * dim + i]);
    }
    
    float sum_exp = 0.0f;
    for (int i = 0; i < dim; i++) {
        float exp_val = exp(x[row * dim + i] - max_val);
        output[row * dim + i] = exp_val;
        sum_exp += exp_val;
    }
    
    for (int i = 0; i < dim; i++) {
        output[row * dim + i] /= sum_exp;
    }
}

kernel void kernel_rope_simple(
    global float * x,
    int head_dim,
    int num_heads,
    int seq_len,
    int start_pos,
    float theta
) {
    // x is [batch, seq, num_heads, head_dim]
    // Each work item processes one (i, i+head_dim/2) pair
    int gid = get_global_id(0);
    
    int half_dim = head_dim / 2;
    int elements_per_seq = num_heads * head_dim;
    int pairs_per_seq = num_heads * half_dim;
    
    // Decode gid into (seq_idx, head_idx, pair_idx)
    int seq_idx = gid / pairs_per_seq;
    int in_seq = gid % pairs_per_seq;
    int head_idx = in_seq / half_dim;
    int pair_idx = in_seq % half_dim;
    
    int pos = start_pos + seq_idx;
    
    // Calculate frequency for this pair
    float freq = 1.0f / pow(theta, (float)(pair_idx * 2) / (float)head_dim);
    float angle = (float)pos * freq;
    
    float cos_val = cos(angle);
    float sin_val = sin(angle);
    
    // Calculate indices into the tensor (neox layout: first half and second half of head)
    int base_offset = seq_idx * elements_per_seq + head_idx * head_dim;
    int i0 = base_offset + pair_idx;
    int i1 = base_offset + pair_idx + half_dim;
    
    float x0 = x[i0];
    float x1 = x[i1];
    
    x[i0] = x0 * cos_val - x1 * sin_val;
    x[i1] = x0 * sin_val + x1 * cos_val;
}

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

kernel void kernel_silu_mul_simple(
    global float * x,
    global float * y,
    int size
) {
    int i = get_global_id(0);
    if (i < size) {
        float val = x[i];
        float sigmoid = 1.0f / (1.0f + exp(-val));
        x[i] = val * sigmoid * y[i];
    }
}

//------------------------------------------------------------------------------
// Optimized Single-Query Attention Kernel for Generation (GQA-aware)
// Q: [num_heads_q, head_dim] - single query token
// K: [cache_seq_len, num_heads_kv, head_dim] - key cache
// V: [cache_seq_len, num_heads_kv, head_dim] - value cache  
// O: [num_heads_q, head_dim] - output
// Each workgroup processes one query head
// 
// OPTIMIZATION: 2-pass approach
// Pass 1: Compute scores, find max, compute softmax sum (fused)
// Pass 2: Compute weighted V using stored weights
//------------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
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
    local float * scratch        // size = local_size
) {
    int head_idx = get_group_id(0);    // which Q head
    int lid = get_local_id(0);
    int local_size = get_local_size(0);
    
    // GQA: map Q head to KV head
    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;
    
    // Pointers
    global float * q_ptr = Q + head_idx * head_dim;
    
    // === PASS 1: Compute max score using online approach ===
    float my_max = -INFINITY;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global float * k_ptr = K + (t * num_heads_kv + kv_head) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;
        my_max = fmax(my_max, score);
    }
    
    // Reduce max across workgroup
    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // Compute exp sum with known max
    float my_sum = 0.0f;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global float * k_ptr = K + (t * num_heads_kv + kv_head) * head_dim;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        my_sum += exp(score * scale - max_score);
    }
    
    // Reduce sum
    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    
    // === PASS 2: Compute weighted V sum ===
    // Each thread accumulates output for ALL head_dim elements for its subset of tokens
    // Then we reduce across threads
    
    // Initialize output accumulator per thread
    float out_local[64]; // Assuming head_dim <= 64
    for (int d = 0; d < head_dim; d++) {
        out_local[d] = 0.0f;
    }
    
    // Each thread processes its subset of tokens
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global float * k_ptr = K + (t * num_heads_kv + kv_head) * head_dim;
        
        // Compute weight for this token
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        float weight = exp(score * scale - max_score) / total_sum;
        
        // Accumulate V contribution
        global float * v_ptr = V + (t * num_heads_kv + kv_head) * head_dim;
        for (int d = 0; d < head_dim; d++) {
            out_local[d] += weight * v_ptr[d];
        }
    }
    
    // Reduce output across threads for each dimension
    // Use scratch for reduction (one dimension at a time)
    for (int d = 0; d < head_dim; d++) {
        scratch[lid] = out_local[d];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0) {
            O[head_idx * head_dim + d] = scratch[0];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
