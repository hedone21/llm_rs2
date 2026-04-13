#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define ACC_TYPE float
#define ACC_TYPE4 float4
#define Q_DATA_TYPE4 float4
#define KV_DATA_TYPE4 half4
#define O_DATA_TYPE4 float4
#define MASK_DATA_TYPE half
#define CONVERT_Q_ACC4(x) (x)
#define CONVERT_KV_ACC4(x) convert_float4(x)
#define CONVERT_O_DATA4(x) (x)

#define DK_VEC (DK/4)
#define DV_VEC (DV/4)
#define WG_SIZE (BLOCK_M)
#define Q1_WG_SIZE 64

inline float get_alibi_slope(
    const float max_bias, const uint h, const uint n_head_log2, const float m0, const float m1
) {
    if (max_bias <= 0.0f) {
        return 1.0f;
    }
    const float base = h < n_head_log2 ? m0 : m1;
    const int   exph = h < n_head_log2 ? h + 1 : 2*(h - n_head_log2) + 1;

    return pow(base, exph);
}
__kernel void flash_attn_f32_f16(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int block_q_idx = get_group_id(0);
    const int head_batch_idx = get_global_id(1);

    const int my_query_row = block_q_idx * BLOCK_M + tid;

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    if (my_query_row < n_q) {
        const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2 + my_query_row * q_nb1;
        const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
        #pragma unroll
        for (int i = 0; i < DK_VEC; ++i) {
            q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
        }
    }

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) {
        o_acc[i] = (ACC_TYPE4)(0.0f);
    }
    ACC_TYPE m_i = -INFINITY;
    ACC_TYPE l_i = 0.0f;

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    __local KV_DATA_TYPE4 l_k[BLOCK_N][DK_VEC];
    __local KV_DATA_TYPE4 l_v[BLOCK_N][DV_VEC];

    for (int k_start = 0; k_start < n_kv; k_start += BLOCK_N) {
        for (int i = tid; i < BLOCK_N * DK_VEC; i += WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (k_row_idx < n_kv) {
                const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global KV_DATA_TYPE4*)(k_base + k_row_offset))[col];
            } else {
                l_k[row][col] = (KV_DATA_TYPE4)(0);
            }
        }
        for (int i = tid; i < BLOCK_N * DV_VEC; i += WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (v_row_idx < n_kv) {
                const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global KV_DATA_TYPE4*)(v_base + v_row_offset))[col];
            } else {
                l_v[row][col] = (KV_DATA_TYPE4)(0);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        if (my_query_row >= n_q) {
            continue;
        }

        for (int j = 0; j < BLOCK_N; j += 2) {
            const int k_row0 = k_start + j;
            const int k_row1 = k_start + j + 1;

            ACC_TYPE4 dot_acc0 = (ACC_TYPE4)(0.0f);
            ACC_TYPE4 dot_acc1 = (ACC_TYPE4)(0.0f);
            #pragma unroll
            for (int k = 0; k < DK_VEC; k++) {
                dot_acc0 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j][k]), dot_acc0);
                dot_acc1 = mad(q_priv[k], CONVERT_KV_ACC4(l_k[j+1][k]), dot_acc1);
            }
            ACC_TYPE score0 = (dot_acc0.s0 + dot_acc0.s1 + dot_acc0.s2 + dot_acc0.s3) * scale;
            ACC_TYPE score1 = (dot_acc1.s0 + dot_acc1.s1 + dot_acc1.s2 + dot_acc1.s3) * scale;

            if (is_causal) {
                if (k_row0 > (n_kv - n_q + my_query_row)) score0 = -INFINITY;
                if (k_row1 > (n_kv - n_q + my_query_row)) score1 = -INFINITY;
            }

            if (k_row0 >= n_kv) score0 = -INFINITY;
            if (k_row1 >= n_kv) score1 = -INFINITY;

            if (mask_base != NULL) {
                const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base + my_query_row * mask_nb1);
                if (k_row0 < n_kv) score0 += slope * (ACC_TYPE)mask_ptr[k_row0];
                if (k_row1 < n_kv) score1 += slope * (ACC_TYPE)mask_ptr[k_row1];
            }

            if (logit_softcap > 0.0f) {
                score0 = logit_softcap * tanh(score0 / logit_softcap);
                score1 = logit_softcap * tanh(score1 / logit_softcap);
            }

            const ACC_TYPE m_new = max(m_i, max(score0, score1));
            const ACC_TYPE p0 = exp(score0 - m_new);
            const ACC_TYPE p1 = exp(score1 - m_new);
            const ACC_TYPE scale_prev = exp(m_i - m_new);

            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] = o_acc[i] * scale_prev + p0 * CONVERT_KV_ACC4(l_v[j][i]) + p1 * CONVERT_KV_ACC4(l_v[j+1][i]);
            }
            l_i = l_i * scale_prev + p0 + p1;
            m_i = m_new;
        }
    }

    if (my_query_row < n_q) {
        if (sinks_void != NULL) {
            const global ACC_TYPE* sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
            const ACC_TYPE m_sink = sinks_ptr[head_idx];
            const ACC_TYPE m_final = max(m_i, m_sink);

            const ACC_TYPE scale_o = exp(m_i - m_final);
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_acc[i] *= scale_o;
            }

            l_i = l_i * exp(m_i - m_final) + exp(m_sink - m_final);
        }

        const ulong o_row_offset = batch_idx * o_nb3 + my_query_row * o_nb2 + head_idx * o_nb1;
        global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
        if (l_i > 0.0f) {
            const ACC_TYPE l_inv = 1.0f / l_i;
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = CONVERT_O_DATA4(o_acc[i] * l_inv);
            }
        } else {
            #pragma unroll
            for (int i = 0; i < DV_VEC; ++i) {
                o_row[i] = (O_DATA_TYPE4)(0.0f);
            }
        }
    }
}

__kernel void flash_attn_f32_f16_q1(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset
) {
    const int tid = get_local_id(0);
    const int head_batch_idx = get_global_id(1);

    const int batch_idx = head_batch_idx / n_head;
    const int head_idx = head_batch_idx % n_head;

    const int gqa_ratio = n_head / n_head_kv;
    const int head_kv_idx = head_idx / gqa_ratio;

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    const global char* mask_base = NULL;
    if (mask_void != NULL) {
        const int mask_head_idx = head_idx % mask_ne2;
        const int mask_batch_idx = batch_idx % mask_ne3;
        mask_base = (const global char*)mask_void + mask_offset + mask_batch_idx * mask_nb3 + mask_head_idx * mask_nb2;
    }

    ACC_TYPE4 q_priv[DK_VEC];
    const ulong q_row_offset = batch_idx * q_nb3 + head_idx * q_nb2;
    const global Q_DATA_TYPE4* q_ptr = (const global Q_DATA_TYPE4*)(q_base + q_row_offset);
    #pragma unroll
    for (int i = 0; i < DK_VEC; ++i) {
        q_priv[i] = CONVERT_Q_ACC4(q_ptr[i]);
    }

    float slope = get_alibi_slope(max_bias, head_idx, n_head_log2, m0, m1);

    const global ACC_TYPE* sinks_ptr = NULL;
    if (sinks_void != NULL) {
        sinks_ptr = (const global ACC_TYPE*)((const global char*)sinks_void + sinks_offset);
    }

    ACC_TYPE m_i = (sinks_ptr != NULL) ? sinks_ptr[head_idx] : -INFINITY;
    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        m_i = max(m_i, score);
    }

    __local ACC_TYPE local_m[Q1_WG_SIZE];
    local_m[tid] = m_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    const ACC_TYPE m_final = local_m[0];

    ACC_TYPE4 o_acc[DV_VEC];
    #pragma unroll
    for (int i = 0; i < DV_VEC; ++i) o_acc[i] = (ACC_TYPE4)(0.0f);
    ACC_TYPE l_i = 0.0f;

    for (int k_idx = tid; k_idx < n_kv; k_idx += Q1_WG_SIZE) {
        const ulong k_row_offset = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_idx * k_nb1;
        const ulong v_row_offset = batch_idx * v_nb3 + head_kv_idx * v_nb2 + k_idx * v_nb1;
        const global KV_DATA_TYPE4* k_ptr = (const global KV_DATA_TYPE4*)(k_base + k_row_offset);
        const global KV_DATA_TYPE4* v_ptr = (const global KV_DATA_TYPE4*)(v_base + v_row_offset);
        ACC_TYPE4 dot_acc = (ACC_TYPE4)(0.0f);
        #pragma unroll
        for (int k = 0; k < DK_VEC; k++) {
            dot_acc = mad(q_priv[k], CONVERT_KV_ACC4(k_ptr[k]), dot_acc);
        }
        ACC_TYPE score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
        if (mask_base != NULL) {
            const global MASK_DATA_TYPE* mask_ptr = (const global MASK_DATA_TYPE*)(mask_base);
            score += slope * (ACC_TYPE)mask_ptr[k_idx];
        }
        if (logit_softcap > 0.0f) {
            score = logit_softcap * tanh(score / logit_softcap);
        }
        const ACC_TYPE p = exp(score - m_final);
        l_i += p;
        #pragma unroll
        for (int i = 0; i < DV_VEC; i++) {
            o_acc[i] = mad(p, CONVERT_KV_ACC4(v_ptr[i]), o_acc[i]);
        }
    }

    __local ACC_TYPE local_l[Q1_WG_SIZE];
    __local ACC_TYPE4 local_o_comp[Q1_WG_SIZE];
    local_l[tid] = l_i;
    barrier(CLK_LOCAL_MEM_FENCE);
    #pragma unroll
    for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
        if (tid < s) local_l[tid] += local_l[tid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    const ulong o_row_offset = batch_idx * o_nb3 + head_idx * o_nb1;
    global O_DATA_TYPE4 *o_row = (global O_DATA_TYPE4 *)(o_base + o_row_offset);
    ACC_TYPE l_final = local_l[0];

    if (sinks_ptr != NULL) {
        l_final += exp(sinks_ptr[head_idx] - m_final);
    }

    if (l_final > 0.0f) {
        const ACC_TYPE l_inv = 1.0f / l_final;
        for (int i = 0; i < DV_VEC; i++) {
            local_o_comp[tid] = o_acc[i];
            barrier(CLK_LOCAL_MEM_FENCE);
            #pragma unroll
            for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                barrier(CLK_LOCAL_MEM_FENCE);
            }
            if (tid == 0) {
                o_row[i] = CONVERT_O_DATA4(local_o_comp[0] * l_inv);
            }
        }
    } else if (tid == 0) {
        #pragma unroll
        for (int i = 0; i < DV_VEC; ++i) o_row[i] = (O_DATA_TYPE4)(0.0f);
    }
}

// =============================================================================
// GQA-aware decode variant: single WG processes `gqa_ratio` Q-heads that share
// one KV-head. KV tile is loaded once into local memory and reused across the
// GQA group → cuts VRAM KV reads by `gqa_ratio` (Llama 3.2 4x, Qwen 2.5 6x).
//
// Dispatch: global = [Q1_WG_SIZE, n_heads_kv, batch], local = [Q1_WG_SIZE, 1, 1].
// Preconditions:
//   - n_q == 1 (single-query decode)
//   - F16 HeadMajor KV, F32 Q/O
//   - head_dim ∈ {DK, DV} matching the compile-time program variant
//   - gqa_ratio ≤ GQA_RATIO_MAX (runtime arg, compile-time bound)
//   - no mask, no ALiBi, no sinks, no softcap (matches q1 decode usage)
//
// P0-5c: see .agent/research/2026-04-13_gqa_kv_sharing_kernel.md §F.
// =============================================================================
#ifndef GQA_RATIO_MAX
#define GQA_RATIO_MAX 8
#endif

#ifndef GQA_KV_TILE
// KV tile depth. Chosen per head_dim so local K+V footprint stays ≤ ~24 KB
// (Adreno 830 reports 32 KB but leaves headroom for barriers/register spill).
//   DK=64:  K+V = 2 * 64 * 2 bytes * tile = 256 * tile  → tile=64 → 16 KB
//   DK=128: K+V = 2 * 128 * 2 bytes * tile = 512 * tile → tile=32 → 16 KB
#if DK == 128
#define GQA_KV_TILE 32
#else
#define GQA_KV_TILE 64
#endif
#endif

__kernel void flash_attn_f32_f16_q1_gqa(
    const global void * q_void, ulong q_offset,
    const global void * k_void, ulong k_offset,
    const global void * v_void, ulong v_offset,
    global void * o_void, ulong o_offset,
    const float scale,
    const int n_q,
    const int n_kv,
    const int is_causal,
    const int n_head,
    const ulong q_nb1, const ulong q_nb2, const ulong q_nb3,
    const ulong k_nb1, const ulong k_nb2, const ulong k_nb3,
    const ulong v_nb1, const ulong v_nb2, const ulong v_nb3,
    const ulong o_nb1, const ulong o_nb2, const ulong o_nb3,
    const float max_bias,
    const float m0,
    const float m1,
    const int n_head_log2,
    const float logit_softcap,
    const int n_head_kv,
    const global void* mask_void,
    const ulong mask_offset,
    const ulong mask_nb1,
    const ulong mask_nb2,
    const ulong mask_nb3,
    const int mask_ne2,
    const int mask_ne3,
    const global void* sinks_void,
    const ulong sinks_offset,
    const int gqa_ratio
) {
    const int tid = get_local_id(0);
    const int head_kv_idx = get_global_id(1);
    const int batch_idx = get_global_id(2);

    const global char* q_base = (const global char*)q_void + q_offset;
    const global char* k_base = (const global char*)k_void + k_offset;
    const global char* v_base = (const global char*)v_void + v_offset;
    global char* o_base = (global char*)o_void + o_offset;

    // -------------------------------------------------------------------------
    // Load Q for all `gqa_ratio` Q-heads into local memory (cooperative).
    // Shape: [GQA_RATIO_MAX][DK_VEC] half4 — row = q-in-group, col = head_dim/4.
    // Only the first `gqa_ratio` rows are populated.
    // -------------------------------------------------------------------------
    __local half4 l_q[GQA_RATIO_MAX][DK_VEC];

    const int q_items = gqa_ratio * DK_VEC;
    for (int i = tid; i < q_items; i += Q1_WG_SIZE) {
        const int gq = i / DK_VEC;
        const int d  = i % DK_VEC;
        const int head_idx = head_kv_idx * gqa_ratio + gq;
        const ulong q_row_off = batch_idx * q_nb3 + head_idx * q_nb2;
        const global float4* q_ptr = (const global float4*)(q_base + q_row_off);
        // F32 -> F16 conversion for compact local storage; accumulation stays F32.
        float4 qf = q_ptr[d];
        l_q[gq][d] = convert_half4(qf);
    }

    // -------------------------------------------------------------------------
    // Per-query streaming state — private arrays sized by compile-time upper
    // bound. Real iteration only touches the first `gqa_ratio` slots.
    // -------------------------------------------------------------------------
    float m_i[GQA_RATIO_MAX];
    float l_i[GQA_RATIO_MAX];
    float4 o_acc[GQA_RATIO_MAX][DV_VEC];
    #pragma unroll
    for (int gq = 0; gq < GQA_RATIO_MAX; ++gq) {
        m_i[gq] = -INFINITY;
        l_i[gq] = 0.0f;
        #pragma unroll
        for (int d = 0; d < DV_VEC; ++d) {
            o_acc[gq][d] = (float4)(0.0f);
        }
    }

    // -------------------------------------------------------------------------
    // KV tile staging — loaded once per iter, reused across all queries.
    // -------------------------------------------------------------------------
    __local half4 l_k[GQA_KV_TILE][DK_VEC];
    __local half4 l_v[GQA_KV_TILE][DV_VEC];

    barrier(CLK_LOCAL_MEM_FENCE);

    for (int k_start = 0; k_start < n_kv; k_start += GQA_KV_TILE) {
        const int tile_n = min(GQA_KV_TILE, n_kv - k_start);

        // Cooperative KV load — K and V share one barrier at end.
        // Out-of-range rows stay zero-initialized on first use or carry over
        // stale data; we handle boundary via score = -INFINITY below.
        for (int i = tid; i < GQA_KV_TILE * DK_VEC; i += Q1_WG_SIZE) {
            const int row = i / DK_VEC;
            const int col = i % DK_VEC;
            const int k_row_idx = k_start + row;
            if (row < tile_n && k_row_idx < n_kv) {
                const ulong k_row_off = batch_idx * k_nb3 + head_kv_idx * k_nb2 + k_row_idx * k_nb1;
                l_k[row][col] = ((__global half4*)(k_base + k_row_off))[col];
            } else {
                l_k[row][col] = (half4)(0);
            }
        }
        for (int i = tid; i < GQA_KV_TILE * DV_VEC; i += Q1_WG_SIZE) {
            const int row = i / DV_VEC;
            const int col = i % DV_VEC;
            const int v_row_idx = k_start + row;
            if (row < tile_n && v_row_idx < n_kv) {
                const ulong v_row_off = batch_idx * v_nb3 + head_kv_idx * v_nb2 + v_row_idx * v_nb1;
                l_v[row][col] = ((__global half4*)(v_base + v_row_off))[col];
            } else {
                l_v[row][col] = (half4)(0);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        // ---------------------------------------------------------------------
        // Score and update per query. Thread partitioning: each thread takes
        // a stride over tile rows, accumulating into its own private state.
        // WG-final reduction happens below.
        // ---------------------------------------------------------------------
        for (int j = tid; j < GQA_KV_TILE; j += Q1_WG_SIZE) {
            const int k_row = k_start + j;
            const bool valid = (j < tile_n) && (k_row < n_kv);

            // Load K[j] into registers once — reused across all gqa queries.
            float4 kj[DK_VEC];
            #pragma unroll
            for (int d = 0; d < DK_VEC; ++d) {
                kj[d] = convert_float4(l_k[j][d]);
            }
            float4 vj[DV_VEC];
            #pragma unroll
            for (int d = 0; d < DV_VEC; ++d) {
                vj[d] = convert_float4(l_v[j][d]);
            }

            for (int gq = 0; gq < gqa_ratio; ++gq) {
                float4 dot_acc = (float4)(0.0f);
                #pragma unroll
                for (int d = 0; d < DK_VEC; ++d) {
                    float4 qd = convert_float4(l_q[gq][d]);
                    dot_acc = mad(qd, kj[d], dot_acc);
                }
                float score = (dot_acc.s0 + dot_acc.s1 + dot_acc.s2 + dot_acc.s3) * scale;
                if (!valid) score = -INFINITY;

                const float m_new = max(m_i[gq], score);
                const float p = exp(score - m_new);
                const float scale_prev = exp(m_i[gq] - m_new);

                #pragma unroll
                for (int d = 0; d < DV_VEC; ++d) {
                    o_acc[gq][d] = o_acc[gq][d] * scale_prev + p * vj[d];
                }
                l_i[gq] = l_i[gq] * scale_prev + p;
                m_i[gq] = m_new;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
    }

    // -------------------------------------------------------------------------
    // WG-wide reduction of per-thread (m, l, o) into a single output per query.
    // Use the same pattern as flash_attn_f32_f16_q1 but loop over `gqa_ratio`.
    // -------------------------------------------------------------------------
    __local float local_m[Q1_WG_SIZE];
    __local float local_l[Q1_WG_SIZE];
    __local float4 local_o_comp[Q1_WG_SIZE];

    for (int gq = 0; gq < gqa_ratio; ++gq) {
        // Stage 1: reduce m.
        local_m[tid] = m_i[gq];
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) local_m[tid] = max(local_m[tid], local_m[tid + s]);
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float m_final = local_m[0];

        // Rescale each thread's (l, o) to the group-wide m_final.
        const float rescale = exp(m_i[gq] - m_final);
        float l_local = l_i[gq] * rescale;
        float4 o_local[DV_VEC];
        #pragma unroll
        for (int d = 0; d < DV_VEC; ++d) {
            o_local[d] = o_acc[gq][d] * rescale;
        }

        // Stage 2: reduce l.
        local_l[tid] = l_local;
        barrier(CLK_LOCAL_MEM_FENCE);
        #pragma unroll
        for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
            if (tid < s) local_l[tid] += local_l[tid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        const float l_final = local_l[0];

        // Stage 3: reduce o per DV dimension and write to output.
        const int head_idx = head_kv_idx * gqa_ratio + gq;
        const ulong o_row_off = batch_idx * o_nb3 + head_idx * o_nb1;
        global float4* o_row = (global float4*)(o_base + o_row_off);

        if (l_final > 0.0f) {
            const float l_inv = 1.0f / l_final;
            for (int d = 0; d < DV_VEC; ++d) {
                local_o_comp[tid] = o_local[d];
                barrier(CLK_LOCAL_MEM_FENCE);
                #pragma unroll
                for (int s = Q1_WG_SIZE / 2; s > 0; s >>= 1) {
                    if (tid < s) local_o_comp[tid] += local_o_comp[tid + s];
                    barrier(CLK_LOCAL_MEM_FENCE);
                }
                if (tid == 0) {
                    o_row[d] = local_o_comp[0] * l_inv;
                }
                barrier(CLK_LOCAL_MEM_FENCE);
            }
        } else if (tid == 0) {
            #pragma unroll
            for (int d = 0; d < DV_VEC; ++d) o_row[d] = (float4)(0.0f);
        }
    }
}
