// Flash Decoding reducer (P0-5).
//
// Log-sum-exp merge over per-split partial (m_s, l_s, o_s) tuples produced
// by `flash_attn_*_q1_split`. Partials are UNNORMALIZED (NOT yet divided
// by l_s); the reducer performs the final division after combining all
// splits and folding in optional attention sinks.
//
// Algorithm (see research doc §A.2):
//   m_star = max_s(m_s)
//   alpha_s = exp(m_s - m_star)
//   l_star = sum_s(alpha_s * l_s)
//   o_star[d] = sum_s(alpha_s * o_s[d])
//   if sinks: l_star += exp(sinks[head] - m_star)
//   dst[head, d] = o_star[d] / l_star   (0 if l_star == 0)
//
// Launch geometry:
//   global = [DV, n_head, 1],  local = [DV, 1, 1]   (one WG per head)
//
// Compile-time constants:
//   -DDV=<value>           head_dim (must match the split kernel)
//   -DMAX_KV_SPLITS=<val>  upper bound on kv_splits supported (shared-mem cache)
//
// Types: partials/meta/sinks/dst are all F32 (see research doc §A.3).

#ifndef MAX_KV_SPLITS
#define MAX_KV_SPLITS 32
#endif

__kernel void flash_attn_q1_reduce(
    __global const float* partials,   // [n_head, kv_splits, DV]
    __global const float* meta,       // [n_head, kv_splits, 2] — (m, l) pairs
    __global const void*  sinks_void, // optional sinks buffer (F32, per head)
    const ulong           sinks_offset,
    __global       float* dst,        // [n_head, DV]
    const ulong           dst_offset, // byte offset into dst
    const int             kv_splits,
    const int             n_head,
    const ulong           dst_head_stride // bytes between head rows in dst
) {
    const int d      = get_local_id(0);   // 0..DV-1
    const int head   = get_group_id(1);   // 0..n_head-1

    if (head >= n_head) return;

    // Cache meta in local memory (kv_splits <= MAX_KV_SPLITS). Each thread
    // loads a slice. Bounds check because DV may exceed kv_splits.
    __local float local_m[MAX_KV_SPLITS];
    __local float local_l[MAX_KV_SPLITS];

    const int meta_base = head * kv_splits * 2;
    if (d < kv_splits) {
        local_m[d] = meta[meta_base + d * 2 + 0];
        local_l[d] = meta[meta_base + d * 2 + 1];
    }
    // Pad unused slots so max-reduce ignores them (defensive: main path never
    // reads beyond kv_splits).
    if (d >= kv_splits && d < MAX_KV_SPLITS) {
        local_m[d] = -INFINITY;
        local_l[d] = 0.0f;
    }
    barrier(CLK_LOCAL_MEM_FENCE);

    // Compute global max m_star across splits. Small kv_splits (<=32) →
    // serial scan is cheaper than a reduction tree.
    float m_star = -INFINITY;
    for (int s = 0; s < kv_splits; ++s) {
        const float m_s = local_m[s];
        m_star = max(m_star, m_s);
    }

    // Optional sinks — folded into l_star only; m_star already accounts for
    // per-split maxima and sinks cannot exceed the per-token maxima in a
    // well-trained model, but for numerical safety re-max here too.
    float sink_val = -INFINITY;
    if (sinks_void != NULL) {
        const __global float* sinks = (const __global float*)((const __global char*)sinks_void + sinks_offset);
        sink_val = sinks[head];
        m_star = max(m_star, sink_val);
    }

    // Accumulate l_star and the d-th component of o_star.
    float l_star = 0.0f;
    float o_d = 0.0f;
    const int partial_base = head * kv_splits * DV + d;
    for (int s = 0; s < kv_splits; ++s) {
        const float m_s = local_m[s];
        if (m_s == -INFINITY) continue;  // empty split, contributes nothing
        const float alpha = exp(m_s - m_star);
        l_star += alpha * local_l[s];
        o_d    += alpha * partials[partial_base + s * DV];
    }

    if (sinks_void != NULL && sink_val > -INFINITY) {
        l_star += exp(sink_val - m_star);
    }

    __global float* dst_row = (__global float*)((__global char*)dst + dst_offset + (ulong)head * dst_head_stride);
    dst_row[d] = (l_star > 0.0f) ? (o_d / l_star) : 0.0f;
}
