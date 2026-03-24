// Fallback OpenCL kernels for platforms without cl_khr_subgroups (e.g., pocl)
// All kernel names and signatures match simple_ops.cl exactly so that
// the Rust dispatch code requires zero changes.
// Subgroup reductions are replaced with local-memory tree reductions.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

//------------------------------------------------------------------------------
// RMS Norm — local memory tree reduction (replaces subgroup version)
//------------------------------------------------------------------------------
kernel void kernel_rms_norm_opt(
    global float * x,
    global float * weight,
    int dim,
    float eps,
    int add_unit,           // Gemma3 style: weight = 1 + weight when nonzero
    local float * scratch
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

    // Tree reduction in local memory
    scratch[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum_sq = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Apply normalization
    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;

    for (int i = lid; i < dim; i += local_size) {
        float w = add_unit ? (1.0f + weight[i]) : weight[i];
        row_ptr[i] = row_ptr[i] * scale * w;
    }
}

//------------------------------------------------------------------------------
// Softmax — local memory tree reduction (replaces subgroup version)
//------------------------------------------------------------------------------
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

    scratch[lid] = lmax;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 2: Compute exp and sum
    float lsum = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float exp_val = exp(row_ptr[i] - max_val);
        row_ptr[i] = exp_val;
        lsum += exp_val;
    }

    scratch[lid] = lsum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float sum_exp = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Phase 3: Normalize
    for (int i = lid; i < dim; i += local_size) {
        row_ptr[i] /= sum_exp;
    }
}

//------------------------------------------------------------------------------
// Below: non-subgroup kernels copied verbatim from simple_ops.cl
//------------------------------------------------------------------------------

kernel void kernel_rope_simple(
    global float * x,
    int head_dim,
    int num_heads,
    int seq_len,
    int start_pos,
    float theta
) {
    int gid = get_global_id(0);

    int half_dim = head_dim / 2;
    int elements_per_seq = num_heads * head_dim;
    int pairs_per_seq = num_heads * half_dim;

    int seq_idx = gid / pairs_per_seq;
    int in_seq = gid % pairs_per_seq;
    int head_idx = in_seq / half_dim;
    int pair_idx = in_seq % half_dim;

    int pos = start_pos + seq_idx;

    float freq = 1.0f / pow(theta, (float)(pair_idx * 2) / (float)head_dim);
    float angle = (float)pos * freq;

    float cos_val = cos(angle);
    float sin_val = sin(angle);

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

kernel void kernel_gelu_tanh_mul(
    global float * x,
    global float * y,
    int size
) {
    int i = get_global_id(0);
    if (i < size) {
        float val = x[i];
        float inner = 0.7978845608f * (val + 0.044715f * val * val * val);
        float gelu = 0.5f * val * (1.0f + tanh(inner));
        x[i] = gelu * y[i];
    }
}

//------------------------------------------------------------------------------
// Simple fallback versions (kept for compatibility)
//------------------------------------------------------------------------------

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

//------------------------------------------------------------------------------
// Attention generation kernel (already uses local memory, no subgroups)
//------------------------------------------------------------------------------
kernel void kernel_attn_gen(
    global float * Q,
    global float * K,
    global float * V,
    global float * O,
    int head_dim,
    int num_heads_q,
    int num_heads_kv,
    int cache_seq_len,
    float scale,
    int kv_pos_stride,
    int kv_head_stride,
    local float * scratch
) {
    int head_idx = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;
    int kv_base = kv_head * kv_head_stride;

    global float * q_ptr = Q + head_idx * head_dim;

    // PASS 1: Compute max score
    float my_max = -INFINITY;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global float * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        score *= scale;
        my_max = fmax(my_max, score);
    }

    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute exp sum
    float my_sum = 0.0f;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global float * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        my_sum += exp(score * scale - max_score);
    }

    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    // PASS 2: Weighted V sum
    float out_local[256]; // Max head_dim across supported models (Gemma 3: 256, Qwen2: 128, Llama: 64)
    for (int d = 0; d < head_dim; d++) {
        out_local[d] = 0.0f;
    }

    for (int t = lid; t < cache_seq_len; t += local_size) {
        global float * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * k_ptr[d];
        }
        float weight = exp(score * scale - max_score) / total_sum;

        global float * v_ptr = V + kv_base + t * kv_pos_stride;
        for (int d = 0; d < head_dim; d++) {
            out_local[d] += weight * v_ptr[d];
        }
    }

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

//------------------------------------------------------------------------------
// Out-of-place RMSNorm: reads x, writes normalized result to out.
//------------------------------------------------------------------------------
kernel void kernel_rms_norm_oop(
    global float * x,
    global float * out,
    global float * weight,
    int dim,
    float eps,
    int add_unit,           // Gemma3 style: weight = 1 + weight when nonzero
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float * x_row = x + row * dim;
    global float * out_row = out + row * dim;

    float sum_sq = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float val = x_row[i];
        sum_sq += val * val;
    }

    scratch[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum_sq = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;

    for (int i = lid; i < dim; i += local_size) {
        float w = add_unit ? (1.0f + weight[i]) : weight[i];
        out_row[i] = x_row[i] * scale * w;
    }
}

//------------------------------------------------------------------------------
// Fused add + out-of-place RMSNorm: x += residual; out = norm(x) * weight
//------------------------------------------------------------------------------
kernel void kernel_add_rms_norm_oop(
    global float * x,
    global float * residual,
    global float * out,
    global float * weight,
    int dim,
    float eps,
    int add_unit,           // Gemma3 style: weight = 1 + weight when nonzero
    local float * scratch
) {
    int row = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    global float * x_row = x + row * dim;
    global float * res_row = residual + row * dim;
    global float * out_row = out + row * dim;

    float sum_sq = 0.0f;
    for (int i = lid; i < dim; i += local_size) {
        float val = x_row[i] + res_row[i];
        x_row[i] = val;
        sum_sq += val * val;
    }

    scratch[lid] = sum_sq;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    sum_sq = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;

    for (int i = lid; i < dim; i += local_size) {
        float w = add_unit ? (1.0f + weight[i]) : weight[i];
        out_row[i] = x_row[i] * scale * w;
    }
}

//------------------------------------------------------------------------------
// F16 KV Cache Support Kernels
//------------------------------------------------------------------------------

kernel void kernel_cast_f32_to_f16(
    global float * src,
    global half * dst,
    int count
) {
    int gid = get_global_id(0);
    if (gid < count) {
        dst[gid] = (half)src[gid];
    }
}

kernel void kernel_attn_gen_half(
    global float * Q,
    global half * K,
    global half * V,
    global float * O,
    int head_dim,
    int num_heads_q,
    int num_heads_kv,
    int cache_seq_len,
    float scale,
    int kv_pos_stride,
    int kv_head_stride,
    local float * scratch
) {
    int head_idx = get_group_id(0);
    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int gqa_ratio = num_heads_q / num_heads_kv;
    int kv_head = head_idx / gqa_ratio;
    int kv_base = kv_head * kv_head_stride;

    global float * q_ptr = Q + head_idx * head_dim;

    float my_max = -INFINITY;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global half * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * vload_half(d, k_ptr);
        }
        score *= scale;
        my_max = fmax(my_max, score);
    }

    scratch[lid] = my_max;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float max_score = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float my_sum = 0.0f;
    for (int t = lid; t < cache_seq_len; t += local_size) {
        global half * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * vload_half(d, k_ptr);
        }
        my_sum += exp(score * scale - max_score);
    }

    scratch[lid] = my_sum;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = local_size / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    float total_sum = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);

    float out_local[256]; // Max head_dim across supported models (Gemma 3: 256, Qwen2: 128, Llama: 64)
    for (int d = 0; d < head_dim; d++) {
        out_local[d] = 0.0f;
    }

    for (int t = lid; t < cache_seq_len; t += local_size) {
        global half * k_ptr = K + kv_base + t * kv_pos_stride;
        float score = 0.0f;
        for (int d = 0; d < head_dim; d++) {
            score += q_ptr[d] * vload_half(d, k_ptr);
        }
        float weight = exp(score * scale - max_score) / total_sum;

        global half * v_ptr = V + kv_base + t * kv_pos_stride;
        for (int d = 0; d < head_dim; d++) {
            out_local[d] += weight * vload_half(d, v_ptr);
        }
    }

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

//------------------------------------------------------------------------------
// Fused F32->F16 Cast + HeadMajor Scatter for KV Cache Update
//------------------------------------------------------------------------------
kernel void kernel_kv_scatter_f32_to_f16(
    global const float * k_src,
    global const float * v_src,
    global half * k_dst,
    global half * v_dst,
    int head_dim,
    int capacity,
    int write_pos
) {
    int gid = get_global_id(0);
    int h = gid / head_dim;
    int d = gid % head_dim;

    int src_idx = h * head_dim + d;
    int dst_idx = h * capacity * head_dim + write_pos * head_dim + d;

    k_dst[dst_idx] = (half)k_src[src_idx];
    v_dst[dst_idx] = (half)v_src[src_idx];
}
