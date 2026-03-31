// GPU-side attention score reduction kernels.
// Accumulates post-softmax attention scores entirely on GPU,
// eliminating per-token GPU->CPU blocking readback (~129ms/token).
//
// Flow per decode token:
//   1. kernel_attn_gen writes scores to persistent GPU buffer
//   2. kernel_score_reduce: per-layer reduction (MAX across layers within step)
//   3. kernel_score_end_step: flush step scores into cumulative importance (with decay)
//   4. kernel_score_clear: zero step buffers for next token
//
// Readback happens only at eviction time (1 blocking read per eviction event).

// Reduce attention scores from one layer into step-local buffers.
// Called once per layer (16x per token for Llama 3.2 1B).
//
// scores: [n_heads_q * score_stride] - post-softmax attention weights from kernel_attn_gen
// step_flat: [max_seq_len] - step-local flat importance (MAX across layers)
// step_head: [n_kv_heads * max_seq_len] - step-local per-KV-head importance (MAX across layers)
//
// Each work-item processes one token position t.
// Flat: sum all Q-head scores for token t, then MAX with step_flat[t].
// Per-head: average Q-heads within each GQA group, then MAX with step_head[kv*max_seq+t].
__kernel void kernel_score_reduce(
    __global const float * scores,
    __global float * step_flat,
    __global float * step_head,
    int n_heads_q,
    int n_kv_heads,
    int cache_seq_len,
    int score_stride,
    int max_seq_len
) {
    int t = get_global_id(0);
    if (t >= cache_seq_len) return;

    int n_rep = n_heads_q / n_kv_heads;

    // Sum all Q-head scores for flat importance
    float total = 0.0f;
    for (int h = 0; h < n_heads_q; h++) {
        total += scores[h * score_stride + t];
    }

    // Accumulate flat (MAX across layers within step)
    step_flat[t] = max(step_flat[t], total);

    // GQA per-KV-head average
    float inv_rep = 1.0f / (float)n_rep;
    for (int kv = 0; kv < n_kv_heads; kv++) {
        float group_sum = 0.0f;
        for (int r = 0; r < n_rep; r++) {
            group_sum += scores[(kv * n_rep + r) * score_stride + t];
        }
        float avg = group_sum * inv_rep;
        int idx = kv * max_seq_len + t;
        step_head[idx] = max(step_head[idx], avg);
    }
}

// Flush step-local importance into cumulative importance with exponential decay.
// Called once per token after all layers are processed.
//
// importance: [max_seq_len] - cumulative flat importance
// step_flat: [max_seq_len] - step-local flat importance (consumed then cleared)
// head_importance: [n_kv_heads * max_seq_len] - cumulative per-head importance
// step_head: [n_kv_heads * max_seq_len] - step-local per-head (consumed then cleared)
// decay_factor: (1.0 - decay), applied to cumulative before adding step scores
//
// Each work-item processes one token position t.
__kernel void kernel_score_end_step(
    __global float * importance,
    __global const float * step_flat,
    __global float * head_importance,
    __global const float * step_head,
    float decay_factor,
    int n_kv_heads,
    int cache_seq_len,
    int max_seq_len
) {
    int t = get_global_id(0);
    if (t >= cache_seq_len) return;

    importance[t] = importance[t] * decay_factor + step_flat[t];

    for (int kv = 0; kv < n_kv_heads; kv++) {
        int idx = kv * max_seq_len + t;
        head_importance[idx] = head_importance[idx] * decay_factor + step_head[idx];
    }
}

// Clear a float buffer (set to zero).
// Used to zero step_flat and step_head after end_step.
__kernel void kernel_score_clear(
    __global float * buf,
    int n
) {
    int i = get_global_id(0);
    if (i < n) buf[i] = 0.0f;
}
