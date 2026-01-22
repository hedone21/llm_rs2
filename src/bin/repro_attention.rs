
use std::time:: {Instant, Duration};
use rand::Rng;

// Simulating the logic in LlamaLayer::forward_gen CPU path
fn attention_cpu_scalar(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize, 
    scores_buffer: &mut [f32], // Shared buffer [seq_len]
    out: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_heads_q / n_heads_kv;

    for h in 0..n_heads_q {
        let kv_h = h / n_rep;
        let q_off = h * head_dim;
        let q_vec = &q[q_off..q_off + head_dim];
        
        // scores for this head
        let scores = &mut scores_buffer[..seq_len];

        // 1. Score Q * K^T
        for ct in 0..seq_len {
            let k_off = (ct * n_heads_kv + kv_h) * head_dim;
            let k_vec = &k[k_off..k_off + head_dim];
            
            let score: f32 = q_vec.iter().zip(k_vec.iter()).map(|(a, b)| a * b).sum();
            scores[ct] = score * scale;
        }

        // 2. Softmax
        let max_val = scores.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
        let mut sum_exp = 0.0;
        for s in scores.iter_mut() {
            *s = (*s - max_val).exp();
            sum_exp += *s;
        }
        for s in scores.iter_mut() { *s /= sum_exp; }

        // 3. Weight Sum
        let out_off = h * head_dim;
        // Zero out
        for d in 0..head_dim {
            out[out_off + d] = 0.0;
        }

        for ct in 0..seq_len {
            let weight = scores[ct];
            let v_off = (ct * n_heads_kv + kv_h) * head_dim;
            let v_vec = &v[v_off..v_off + head_dim];
            
            for d in 0..head_dim {
                out[out_off + d] += weight * v_vec[d];
            }
        }
    }
}

fn attention_cpu_optimized(
    q: &[f32],
    k: &[f32],
    v: &[f32],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    seq_len: usize, 
    all_scores: &mut [f32], // [n_heads_q * seq_len]
    out: &mut [f32],
) {
    let scale = 1.0 / (head_dim as f32).sqrt();
    let n_rep = n_heads_q / n_heads_kv;

    // Zero out output first
    for x in out.iter_mut() { *x = 0.0; }
    
    // 1. Q * K^T 
    // Loop interchange: Outer loop over Time (t), Inner loop over Heads (h)
    for t in 0..seq_len {
        for h in 0..n_heads_q {
            let kv_h = h / n_rep;
            let q_off = h * head_dim;
            let q_vec = &q[q_off..q_off + head_dim];
            
            let k_off = (t * n_heads_kv + kv_h) * head_dim;
            let k_vec = &k[k_off..k_off + head_dim];
            
            let mut score = 0.0;
            for i in 0..head_dim {
                score += q_vec[i] * k_vec[i];
            }
            // Stride is seq_len for simplicity in this bench
            all_scores[h * seq_len + t] = score * scale;
        }
    }

    // 2. Softmax (Per head)
    for h in 0..n_heads_q {
        let scores_h = &mut all_scores[h * seq_len .. h * seq_len + seq_len];
        let mut max_val = f32::NEG_INFINITY;
        for &s in scores_h.iter() { if s > max_val { max_val = s; } }
        let mut sum_exp = 0.0;
        for s in scores_h.iter_mut() {
            *s = (*s - max_val).exp();
            sum_exp += *s;
        }
        let inv_sum = 1.0 / sum_exp;
        for s in scores_h.iter_mut() { *s *= inv_sum; }
    }

    // 3. Score * V
    // Loop interchange
    for t in 0..seq_len {
        for h in 0..n_heads_q {
            let kv_h = h / n_rep;
            let weight = all_scores[h * seq_len + t];
            
            let v_off = (t * n_heads_kv + kv_h) * head_dim;
            let v_vec = &v[v_off..v_off + head_dim];
            
            let out_off = h * head_dim;
            for i in 0..head_dim {
                out[out_off + i] += weight * v_vec[i];
            }
        }
    }
}

fn benchmark(seq_len: usize) -> (f64, f64) {
    let n_heads_q = 32;
    let n_heads_kv = 32; // Llama 2 7B typically has equal heads, or grouped. Let's assume 32.
    let head_dim = 128;
    
    let mut rng = rand::rng();
    
    let q: Vec<f32> = (0..n_heads_q * head_dim).map(|_| rng.random::<f32>()).collect();
    let k: Vec<f32> = (0..seq_len * n_heads_kv * head_dim).map(|_| rng.random::<f32>()).collect();
    let v: Vec<f32> = (0..seq_len * n_heads_kv * head_dim).map(|_| rng.random::<f32>()).collect();
    let mut out = vec![0.0; n_heads_q * head_dim];

    let iter = 10;
    
    let mut scores_scalar = vec![0.0; seq_len];
    let mut scores_opt = vec![0.0; n_heads_q * seq_len];

    // Warmup
    attention_cpu_scalar(&q, &k, &v, n_heads_q, n_heads_kv, head_dim, seq_len, &mut scores_scalar, &mut out);
    attention_cpu_optimized(&q, &k, &v, n_heads_q, n_heads_kv, head_dim, seq_len, &mut scores_opt, &mut out);

    let start = Instant::now();
    for _ in 0..iter {
        attention_cpu_scalar(&q, &k, &v, n_heads_q, n_heads_kv, head_dim, seq_len, &mut scores_scalar, &mut out);
    }
    let duration = start.elapsed();
    let ms_scalar = (duration.as_secs_f64() * 1000.0) / iter as f64;
    
    let start = Instant::now();
    for _ in 0..iter {
        attention_cpu_optimized(&q, &k, &v, n_heads_q, n_heads_kv, head_dim, seq_len, &mut scores_opt, &mut out);
    }
    let duration = start.elapsed();
    let ms_opt = (duration.as_secs_f64() * 1000.0) / iter as f64;
    
    (ms_scalar, ms_opt)
}

fn main() {
    println!("Benchmarking Attention...");
    println!("{:<10} | {:<15} | {:<15} | {:<10}", "Seq Len", "Scalar (ms)", "Opt (ms)", "Speedup");
    println!("{:-<10}-+-{:-<15}-+-{:-<15}-+-{:-<10}", "", "", "", "");
    
    for seq_len in [128, 512, 1024, 2048, 4096].iter() {
        let (ms_s, ms_o) = benchmark(*seq_len);
        println!("{:<10} | {:<15.4} | {:<15.4} | {:<10.2}x", seq_len, ms_s, ms_o, ms_s / ms_o);
    }
}
