// Simple OpenCL kernels for basic ops - custom implementation for llm_rs2

// RMS Norm: output[i] = x[i] * rsqrt(mean(x^2) + eps) * weight[i % weight_size]
// Input x: [N, D], weight: [D], output: [N, D]
kernel void kernel_rms_norm_simple(
    global float * x,
    global float * weight,
    global float * output,
    int dim,
    float eps
) {
    int row = get_global_id(0);  // which row
    
    // Compute sum of squares for this row
    float sum_sq = 0.0f;
    for (int i = 0; i < dim; i++) {
        float val = x[row * dim + i];
        sum_sq += val * val;
    }
    
    // RMS normalization
    float rms = sqrt(sum_sq / (float)dim + eps);
    float scale = 1.0f / rms;
    
    // Apply normalization and weight
    for (int i = 0; i < dim; i++) {
        output[row * dim + i] = x[row * dim + i] * scale * weight[i];
    }
}

// Softmax: output[i] = exp(x[i] - max) / sum(exp(x - max))
// Input x: [N, D], output: [N, D]  - apply softmax along last dim
kernel void kernel_softmax_simple(
    global float * x,
    global float * output,
    int dim
) {
    int row = get_global_id(0);  // which row
    
    // Find max for numerical stability
    float max_val = -INFINITY;
    for (int i = 0; i < dim; i++) {
        max_val = fmax(max_val, x[row * dim + i]);
    }
    
    // Compute exp and sum
    float sum_exp = 0.0f;
    for (int i = 0; i < dim; i++) {
        float exp_val = exp(x[row * dim + i] - max_val);
        output[row * dim + i] = exp_val;  // Store temporarily
        sum_exp += exp_val;
    }
    
    // Normalize
    for (int i = 0; i < dim; i++) {
        output[row * dim + i] /= sum_exp;
    }
}

// RoPE inplace: applies rotary position embeddings
// Input x: [batch, seq, heads, head_dim], applies rotation on head_dim pairs
kernel void kernel_rope_simple(
    global float * x,
    int head_dim,
    int start_pos,
    float theta
) {
    int idx = get_global_id(0);  // linear index into x
    int pair_idx = idx % (head_dim / 2);  // which pair within head_dim
    int pos = start_pos + (idx / (head_dim * 1));  // approximate position
    
    // Calculate the rotation angle
    float freq = 1.0f / pow(theta, (float)(pair_idx * 2) / (float)head_dim);
    float angle = (float)pos * freq;
    
    float cos_val = cos(angle);
    float sin_val = sin(angle);
    
    // Get the pair indices
    int i0 = (idx / (head_dim / 2)) * head_dim + pair_idx;
    int i1 = i0 + head_dim / 2;
    
    float x0 = x[i0];
    float x1 = x[i1];
    
    x[i0] = x0 * cos_val - x1 * sin_val;
    x[i1] = x0 * sin_val + x1 * cos_val;
}

// Scale: x[i] *= scale
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

// Add assign: x[i] += y[i]
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

// SiLU * mul: x[i] = silu(x[i]) * y[i]  where silu(x) = x * sigmoid(x)
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
