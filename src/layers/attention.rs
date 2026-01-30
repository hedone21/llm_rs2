use rayon::prelude::*;

/// Naive Standard Attention implementation for verification purposes.
/// 
/// Computes: Softmax(Q * K^T / sqrt(d)) * V
/// 
/// Arguments:
/// - `q`, `k`, `v`: [seq_len, head_dim] flat buffers (for a single head)
/// - `out`: [seq_len, head_dim] output buffer
/// - `seq_len`: Sequence length (N)
/// - `head_dim`: Head dimension (D) (e.g. 64 or 128)
pub fn naive_attention_head(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    seq_len: usize,
    head_dim: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    
    // Q * K^T -> Scores [N, N]
    let mut scores = vec![0.0; seq_len * seq_len];
    
    for i in 0..seq_len {
        for j in 0..seq_len {
            let mut dot = 0.0;
            for d in 0..head_dim {
                dot += q[i * head_dim + d] * k[j * head_dim + d];
            }
            scores[i * seq_len + j] = dot * scale;
        }
    }
    
    // Softmax per row
    for i in 0..seq_len {
        let row_start = i * seq_len;
        let row_end = row_start + seq_len;
        let row = &mut scores[row_start..row_end];
        
        let max_val = row.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum = 0.0;
        for val in row.iter_mut() {
            *val = (*val - max_val).exp();
            sum += *val;
        }
        for val in row.iter_mut() {
            *val /= sum;
        }
    }
    
    // Scores * V -> Out [N, D]
    for i in 0..seq_len {
        for d in 0..head_dim {
            let mut sum = 0.0;
            for j in 0..seq_len {
                let s = scores[i * seq_len + j];
                let v_val = v[j * head_dim + d];
                sum += s * v_val;
            }
            out[i * head_dim + d] = sum;
        }
    }
}

/// Flash Attention implementation (Forward pass)
/// 
/// Uses tiling and online softmax to avoid O(N^2) memory usage.
/// Supports causal masking and arbitrary strides.
/// 
/// Arguments:
/// - `q_stride`, `k_stride`, `v_stride`, `out_stride`: Stride (in elements) between consecutive rows (tokens).
/// - `q_start_pos`: The global position of the first query token (for causal masking check `c <= r + q_start_pos`).
pub fn flash_attention_head(
    q: &[f32], q_stride: usize,
    k: &[f32], k_stride: usize,
    v: &[f32], v_stride: usize,
    out: &mut [f32], out_stride: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    q_start_pos: usize,
    br: usize,
    bc: usize,
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let tr = (q_len + br - 1) / br;
    let tc = (kv_len + bc - 1) / bc;

    for i in 0..tr {
        let r_start = i * br;
        let r_end = std::cmp::min(r_start + br, q_len);
        let cur_br = r_end - r_start;

        let mut oi = vec![0.0; cur_br * head_dim];
        let mut li = vec![0.0; cur_br];
        let mut mi = vec![f32::NEG_INFINITY; cur_br];

        for j in 0..tc {
            let c_start = j * bc;
            let c_end = std::cmp::min(c_start + bc, kv_len);
            let cur_bc = c_end - c_start;

            // Optimization: Causal Masking Check
            // c_start is the first key index in this block.
            // r_end - 1 is the last query LOCAL index in this block.
            // Global Q index: (r_end - 1) + q_start_pos.
            // We can skip block if c_start > (r_end - 1) + q_start_pos.
            if c_start > (r_end - 1) + q_start_pos {
                continue; 
            }

            for r in 0..cur_br {
                let global_r = r_start + r;
                let mut row_max = f32::NEG_INFINITY;
                let mut sij = vec![f32::NEG_INFINITY; cur_bc];
                let mut any_valid = false;

                for c in 0..cur_bc {
                    let global_c = c_start + c;
                    
                    if global_c <= global_r + q_start_pos {
                        let mut dot = 0.0;
                        let q_ptr = global_r * q_stride;
                        let k_ptr = global_c * k_stride;
                        
                        for d in 0..head_dim {
                            dot += q[q_ptr + d] * k[k_ptr + d];
                        }
                        let val = dot * scale;
                        sij[c] = val;
                        if val > row_max {
                            row_max = val;
                        }
                        any_valid = true;
                    }
                }
                
                if !any_valid { continue; }

                let m_prev = mi[r];
                let m_new = m_prev.max(row_max);
                
                let mut p_row = vec![0.0; cur_bc];
                let mut l_new_part = 0.0;
                
                for c in 0..cur_bc {
                    if sij[c] != f32::NEG_INFINITY {
                        p_row[c] = (sij[c] - m_new).exp();
                        l_new_part += p_row[c];
                    }
                }

                let alpha = if m_prev == f32::NEG_INFINITY { 0.0 } else { (m_prev - m_new).exp() };
                
                li[r] = li[r] * alpha + l_new_part;
                mi[r] = m_new;

                for d in 0..head_dim {
                    let mut pv_sum = 0.0;
                    for c in 0..cur_bc {
                        let global_c = c_start + c;
                        if global_c <= global_r + q_start_pos {
                            // v[global_c, d] -> v[global_c * v_stride + d]
                             pv_sum += p_row[c] * v[global_c * v_stride + d];
                        }
                    }
                    oi[r * head_dim + d] = oi[r * head_dim + d] * alpha + pv_sum;
                }
            }
        }

        for r in 0..cur_br {
            let global_r = r_start + r;
            let l_val = li[r];
            let inv_l = if l_val == 0.0 { 0.0 } else { 1.0 / l_val };

            let out_ptr = global_r * out_stride;
            for d in 0..head_dim {
                out[out_ptr + d] = oi[r * head_dim + d] * inv_l;
            }
        }
    }
}

use std::sync::atomic::{AtomicPtr, Ordering};

/// Computes multi-head attention using Flash Attention strategy in parallel.
/// 
/// Arguments:
/// - `q`, `k`, `v`: Flat buffers containing all heads. 
/// - `stride_q`, `stride_k`, `stride_v`, `stride_out`: Stride in elements between consecutive sequence tokens.
///   For [Seq, Heads, Dim] layout, stride is (Heads * Dim).
///   For [Heads, Seq, Dim] layout, stride is (Dim).
pub fn flash_attention_forward(
    q: &[f32], 
    k: &[f32],
    v: &[f32],
    out: &mut [f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    q_len: usize,
    kv_len: usize,
    head_dim: usize,
    q_stride: usize,
    k_stride: usize,
    v_stride: usize,
    out_stride: usize,
    q_start_pos: usize,
    br: usize,
    bc: usize,
) {
    let n_rep = n_heads_q / n_heads_kv;

    // We need to extract pointers BEFORE the closure to avoid capturing `&mut [f32]` which cannot be shared.
    // Use AtomicPtr which is Send + Sync.
    let out_ptr_raw = AtomicPtr::new(out.as_mut_ptr());
    let q_ptr_raw = AtomicPtr::new(q.as_ptr() as *mut f32);
    let k_ptr_raw = AtomicPtr::new(k.as_ptr() as *mut f32);
    let v_ptr_raw = AtomicPtr::new(v.as_ptr() as *mut f32);

    // Re-do with capturing lengths
    let q_total_len = q.len();
    let k_total_len = k.len(); // v len same
    let out_total_len = out.len();
    
    (0..n_heads_q).into_par_iter().for_each(move |h| {
        let kv_h = h / n_rep;
        let q_head_offset = h * head_dim; 
        let k_head_offset = kv_h * head_dim;
        let out_head_offset = h * head_dim;
        
        let q_ptr = q_ptr_raw.load(Ordering::Relaxed);
        let k_ptr = k_ptr_raw.load(Ordering::Relaxed);
        let v_ptr = v_ptr_raw.load(Ordering::Relaxed);
        let out_ptr = out_ptr_raw.load(Ordering::Relaxed);

        unsafe {
             let q_slice = std::slice::from_raw_parts(q_ptr.add(q_head_offset), q_total_len - q_head_offset);
             let k_slice = std::slice::from_raw_parts(k_ptr.add(k_head_offset), k_total_len - k_head_offset);
             let v_slice = std::slice::from_raw_parts(v_ptr.add(k_head_offset), k_total_len - k_head_offset);
             let out_slice = std::slice::from_raw_parts_mut(out_ptr.add(out_head_offset), out_total_len - out_head_offset);
            
             flash_attention_head(
                q_slice, q_stride,
                k_slice, k_stride,
                v_slice, v_stride,
                out_slice, out_stride,
                q_len, kv_len, head_dim, q_start_pos, br, bc
             );
        }
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::Rng;

    fn approx_eq(a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len());
        for i in 0..a.len() {
            if (a[i] - b[i]).abs() > tol {
                panic!("Mismatch at index {}: {} vs {} (diff {})", i, a[i], b[i], (a[i] - b[i]).abs());
            }
        }
    }

    #[test]
    fn test_naive_attention_sanity() {
        let q = vec![1.0];
        let k = vec![1.0];
        let v = vec![2.0];
        let mut out = vec![0.0];
        
        naive_attention_head(&q, &k, &v, &mut out, 1, 1);
        
        assert!((out[0] - 2.0).abs() < 1e-5);
    }

    #[test]
    fn test_flash_attention_vs_naive() {
        let mut rng = rand::rng();
        let seq_len = 128; // Small enough for test
        let head_dim = 32;
        let br = 32;
        let bc = 32;

        let q: Vec<f32> = (0..seq_len * head_dim).map(|_| rng.random::<f32>()).collect();
        let k: Vec<f32> = (0..seq_len * head_dim).map(|_| rng.random::<f32>()).collect();
        let v: Vec<f32> = (0..seq_len * head_dim).map(|_| rng.random::<f32>()).collect();
        
        let mut out_naive = vec![0.0; seq_len * head_dim];
        let mut out_flash = vec![0.0; seq_len * head_dim];
        
        // Simple case: Strid = head_dim (Contiguous row major)
        let stride = head_dim;

        fn causal_naive(q: &[f32], k: &[f32], v: &[f32], out: &mut [f32], N: usize, D: usize) {
             let scale = 1.0 / (D as f32).sqrt();
             for i in 0..N {
                 let mut row_scores = vec![f32::NEG_INFINITY; N];
                 for j in 0..N {
                     if j <= i { // Causal Mask
                         let mut dot = 0.0;
                         for d in 0..D {
                             dot += q[i*D+d] * k[j*D+d];
                         }
                         row_scores[j] = dot * scale;
                     }
                 }
                 
                 let max_val = row_scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
                 let mut sum_exp = 0.0;
                 for s in row_scores.iter_mut() {
                     if *s != f32::NEG_INFINITY {
                        *s = (*s - max_val).exp();
                        sum_exp += *s;
                     } else {
                        *s = 0.0;
                     }
                 }
                 for s in row_scores.iter_mut() { *s /= sum_exp; }
                 
                 for d in 0..D {
                     let mut s_val = 0.0;
                     for j in 0..N {
                         s_val += row_scores[j] * v[j*D+d];
                     }
                     out[i*D+d] = s_val;
                 }
             }
        }
        
        causal_naive(&q, &k, &v, &mut out_naive, seq_len, head_dim);
        
        flash_attention_head(&q, stride, 
                             &k, stride,
                             &v, stride,
                             &mut out_flash, stride,
                             seq_len, seq_len, head_dim, 0, br, bc);
        
        // Floating point error accumulates slightly differently due to online softmax vs standard softmax
        // 1e-4 should be safe.
        approx_eq(&out_naive, &out_flash, 1e-4);
    }
}
