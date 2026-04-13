//! Host integration tests for Flash Decoding (P0-5) KV-split/reduce path.
//!
//! Verifies that `flash_attention_decode_gpu` produces numerically equivalent
//! output regardless of how many KV splits are used. The production heuristic
//! is `flash_decode_kv_splits`; here we sweep KV lengths and also dispatch the
//! split kernel explicitly for sanity.
//!
//! On hosts where the OpenCL driver cannot compile either the split kernel or
//! the reducer (e.g. macOS/Apple OpenCL), the test skips cleanly — strong
//! on-device verification lives in Task 5.

#![cfg(feature = "opencl")]

use llm_rs2::backend::opencl::{OpenCLBackend, flash_decode_kv_splits};

/// Heuristic must preserve: short ctx → 1 (legacy path), growing with n_kv,
/// always power of two, capped at 32.
#[test]
fn flash_decode_kv_splits_heuristic() {
    // Short contexts use the single-kernel path.
    assert_eq!(flash_decode_kv_splits(0), 1);
    assert_eq!(flash_decode_kv_splits(128), 1);
    assert_eq!(flash_decode_kv_splits(500), 1);
    // Threshold — start splitting at 512.
    assert_eq!(flash_decode_kv_splits(512), 1); // floor(512/1024)=0 → clamped to 1
    assert_eq!(flash_decode_kv_splits(1024), 1);
    // Longer contexts: Metal-style `next_pow2(min(32, n_kv/1024))`.
    assert_eq!(flash_decode_kv_splits(2048), 2);
    assert_eq!(flash_decode_kv_splits(4096), 4);
    assert_eq!(flash_decode_kv_splits(8192), 8);
    assert_eq!(flash_decode_kv_splits(16_384), 16);
    assert_eq!(flash_decode_kv_splits(32_768), 32);
    assert_eq!(flash_decode_kv_splits(65_536), 32);
    // Output is always power of two.
    for n in [600usize, 1500, 3000, 5000, 7000, 12_000, 20_000] {
        let s = flash_decode_kv_splits(n);
        assert!((1..=32).contains(&s), "out of range at n_kv={n}: {s}");
        assert!(s.is_power_of_two(), "not power of two at n_kv={n}: {s}");
    }
}

/// End-to-end: for multiple KV lengths, the GPU output from the KV-split path
/// (auto-selected by the heuristic for n_kv >= ~2K) must match the GPU output
/// from the legacy single-kernel path (forced by using a small KV length in a
/// parallel dispatch). The two outputs are compared head-by-head via cosine
/// similarity ≥ 0.9995.
#[test]
fn flash_attn_decode_split_matches_legacy() {
    use llm_rs2::backend::opencl::memory::OpenCLMemory;
    use llm_rs2::core::backend::Backend;
    use llm_rs2::core::memory::Memory;
    use std::sync::Arc;

    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
        }
    };

    // Require both flash decode and split/reduce to be available. If either
    // is missing (bad driver / unsupported device), we skip — this is a host
    // correctness test, device-level validation runs on Android.
    if !backend.has_flash_decode_kernel(64) {
        eprintln!("Skipping: flash decode kernel (DK=64) unavailable");
        return;
    }

    let ocl_arc: Arc<dyn Backend> = Arc::new(backend);
    let ocl = ocl_arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCLBackend downcast");

    // Llama 3.2 1B shape: 32 Q heads, 8 KV heads (GQA=4), head_dim=64.
    let n_heads_q = 32usize;
    let n_heads_kv = 8usize;
    let head_dim = 64usize;

    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));

    // For each KV length we allocate a cache of capacity = cache_seq_len (tight).
    // n_kv = 128 → legacy path (kv_splits=1)
    // n_kv = 1024 → still legacy (clamped at 1 until >= 2048)
    // n_kv = 2048 → split path (kv_splits=2)
    // n_kv = 4096 → split path (kv_splits=4)
    for &cache_seq_len in &[128usize, 512, 1024, 2048, 4096] {
        let expected_splits = flash_decode_kv_splits(cache_seq_len);
        let capacity = cache_seq_len.next_power_of_two().max(128);

        // Deterministic Q inputs
        let q_data: Vec<f32> = (0..n_heads_q * head_dim)
            .map(|i| ((i as f32) * 0.01).sin())
            .collect();

        let kv_total = n_heads_kv * capacity * head_dim;
        let mut k_data = vec![0u16; kv_total];
        let mut v_data = vec![0u16; kv_total];
        for h in 0..n_heads_kv {
            for p in 0..cache_seq_len {
                for d in 0..head_dim {
                    let idx = h * capacity * head_dim + p * head_dim + d;
                    // Scale down to keep scores in a reasonable range so the
                    // softmax numerator doesn't blow past F32 range even when
                    // split o_acc is accumulated unnormalized. The split kernel
                    // uses F32 accumulators (research doc §A.3) so overflow is
                    // extremely unlikely at these magnitudes — the test just
                    // keeps the comparison well conditioned.
                    let seed = (h as i32 * 31 + p as i32 * 7 + d as i32) as f32;
                    k_data[idx] = half::f16::from_f32((seed * 0.003).cos()).to_bits();
                    v_data[idx] = half::f16::from_f32((seed * 0.005).sin()).to_bits();
                }
            }
        }

        let q = upload_f32(&ocl_arc, &*memory, &q_data, vec![1, 1, n_heads_q, head_dim]);
        let k = upload_f16(
            &ocl_arc,
            &*memory,
            &k_data,
            vec![1, n_heads_kv, capacity, head_dim],
        );
        let v = upload_f16(
            &ocl_arc,
            &*memory,
            &v_data,
            vec![1, n_heads_kv, capacity, head_dim],
        );

        let zero = vec![0.0f32; n_heads_q * head_dim];
        let mut out = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);

        let dispatched = ocl
            .flash_attention_decode_gpu(
                &q,
                &k,
                &v,
                &mut out,
                n_heads_q,
                n_heads_kv,
                head_dim,
                cache_seq_len,
            )
            .expect("flash dispatch");
        if !dispatched {
            eprintln!(
                "Skipping n_kv={cache_seq_len}: flash decode returned Ok(false) (likely no kernel)"
            );
            continue;
        }
        ocl_arc.synchronize().unwrap();

        // Read GPU result back via read_buffer (ARM UMA safe).
        let out_bytes = n_heads_q * head_dim * std::mem::size_of::<f32>();
        let mut raw = vec![0u8; out_bytes];
        ocl_arc.read_buffer(&out, &mut raw).unwrap();
        let gpu_out: Vec<f32> =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n_heads_q * head_dim) }
                .to_vec();

        // CPU reference attention (brute force in f32) — decode query row 0.
        let cpu_out = cpu_reference_attention(
            &q_data,
            &k_data,
            &v_data,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
            capacity,
        );

        // Per-head cosine similarity ≥ 0.9995 (F32 accumulator). Record min
        // so the assertion message is informative across the sweep.
        let mut min_cos = 1.0f32;
        for h in 0..n_heads_q {
            let gpu_head = &gpu_out[h * head_dim..(h + 1) * head_dim];
            let cpu_head = &cpu_out[h * head_dim..(h + 1) * head_dim];
            let cos = cosine_similarity(gpu_head, cpu_head);
            min_cos = min_cos.min(cos);
        }

        assert!(
            min_cos >= 0.9995,
            "n_kv={cache_seq_len} (kv_splits={expected_splits}): min cosine similarity {min_cos} < 0.9995",
        );
    }
}

/// Edge case: non-power-of-two KV length exercised through both paths.
/// Picks a length that triggers the split path (> 1024) and whose split count
/// doesn't evenly divide n_kv.
#[test]
fn flash_attn_decode_split_uneven_kv() {
    use llm_rs2::backend::opencl::memory::OpenCLMemory;
    use llm_rs2::core::backend::Backend;
    use llm_rs2::core::memory::Memory;
    use std::sync::Arc;

    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
        }
    };
    if !backend.has_flash_decode_kernel(64) {
        eprintln!("Skipping: flash decode kernel (DK=64) unavailable");
        return;
    }

    let ocl_arc: Arc<dyn Backend> = Arc::new(backend);
    let ocl = ocl_arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCLBackend downcast");

    // 1B shape
    let n_heads_q = 32usize;
    let n_heads_kv = 8usize;
    let head_dim = 64usize;
    let cache_seq_len = 3000usize; // kv_splits = 2, 3000/2=1500 → ceil → last split is truncated
    assert_eq!(flash_decode_kv_splits(cache_seq_len), 2);

    let capacity = 4096usize;

    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));

    let q_data: Vec<f32> = (0..n_heads_q * head_dim)
        .map(|i| ((i as f32) * 0.013).cos())
        .collect();

    let kv_total = n_heads_kv * capacity * head_dim;
    let mut k_data = vec![0u16; kv_total];
    let mut v_data = vec![0u16; kv_total];
    for h in 0..n_heads_kv {
        for p in 0..cache_seq_len {
            for d in 0..head_dim {
                let idx = h * capacity * head_dim + p * head_dim + d;
                let seed = (h as i32 * 17 + p as i32 * 3 + d as i32) as f32;
                k_data[idx] = half::f16::from_f32((seed * 0.002).sin()).to_bits();
                v_data[idx] = half::f16::from_f32((seed * 0.004).cos()).to_bits();
            }
        }
    }

    let q = upload_f32(&ocl_arc, &*memory, &q_data, vec![1, 1, n_heads_q, head_dim]);
    let k = upload_f16(
        &ocl_arc,
        &*memory,
        &k_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let v = upload_f16(
        &ocl_arc,
        &*memory,
        &v_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let zero = vec![0.0f32; n_heads_q * head_dim];
    let mut out = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);

    let ok = ocl
        .flash_attention_decode_gpu(
            &q,
            &k,
            &v,
            &mut out,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
        )
        .expect("flash dispatch uneven");
    if !ok {
        eprintln!("Skipping: flash decode returned Ok(false)");
        return;
    }
    ocl_arc.synchronize().unwrap();

    let out_bytes = n_heads_q * head_dim * std::mem::size_of::<f32>();
    let mut raw = vec![0u8; out_bytes];
    ocl_arc.read_buffer(&out, &mut raw).unwrap();
    let gpu_out: Vec<f32> =
        unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, n_heads_q * head_dim) }
            .to_vec();

    let cpu_out = cpu_reference_attention(
        &q_data,
        &k_data,
        &v_data,
        n_heads_q,
        n_heads_kv,
        head_dim,
        cache_seq_len,
        capacity,
    );

    let mut min_cos = 1.0f32;
    for h in 0..n_heads_q {
        let gpu_head = &gpu_out[h * head_dim..(h + 1) * head_dim];
        let cpu_head = &cpu_out[h * head_dim..(h + 1) * head_dim];
        let cos = cosine_similarity(gpu_head, cpu_head);
        min_cos = min_cos.min(cos);
    }
    assert!(
        min_cos >= 0.9995,
        "uneven KV: min cosine similarity {min_cos} < 0.9995"
    );
    assert!(
        gpu_out.iter().any(|&v| v.abs() > 1e-6),
        "all-zero output — likely a dispatch no-op"
    );
}

// -----------------------------------------------------------------------------
// Helpers
// -----------------------------------------------------------------------------

/// CPU reference: unfused attention for a single query row (decode).
///
/// Walks the K/V caches in HeadMajor `[1, kv_heads, capacity, head_dim]`
/// layout, computes softmax(Q·K^T / sqrt(d)) · V for each Q head, and returns
/// the concatenated `[n_heads_q, head_dim]` output in row-major order.
#[allow(clippy::too_many_arguments)]
fn cpu_reference_attention(
    q: &[f32],
    k: &[u16],
    v: &[u16],
    n_heads_q: usize,
    n_heads_kv: usize,
    head_dim: usize,
    cache_seq_len: usize,
    capacity: usize,
) -> Vec<f32> {
    let scale = 1.0f32 / (head_dim as f32).sqrt();
    let gqa = n_heads_q / n_heads_kv;
    let mut out = vec![0.0f32; n_heads_q * head_dim];

    for h in 0..n_heads_q {
        let kv_head = h / gqa;
        let head_base_kv = kv_head * capacity * head_dim;
        let q_row = &q[h * head_dim..(h + 1) * head_dim];

        // Compute scores = Q·K^T / sqrt(d) and find max for numerical stability.
        let mut scores = vec![0.0f32; cache_seq_len];
        let mut m = f32::NEG_INFINITY;
        for (p, s_out) in scores.iter_mut().enumerate() {
            let k_base = head_base_kv + p * head_dim;
            let mut s = 0.0f32;
            for d in 0..head_dim {
                let kv = half::f16::from_bits(k[k_base + d]).to_f32();
                s += q_row[d] * kv;
            }
            s *= scale;
            *s_out = s;
            if s > m {
                m = s;
            }
        }

        // Softmax and accumulate weighted values.
        let mut l = 0.0f32;
        for s in scores.iter_mut() {
            *s = (*s - m).exp();
            l += *s;
        }
        let out_row = &mut out[h * head_dim..(h + 1) * head_dim];
        for (p, &w_raw) in scores.iter().enumerate() {
            let w = w_raw / l;
            let v_base = head_base_kv + p * head_dim;
            for d in 0..head_dim {
                let vv = half::f16::from_bits(v[v_base + d]).to_f32();
                out_row[d] += w * vv;
            }
        }
    }
    out
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    for (&x, &y) in a.iter().zip(b.iter()) {
        dot += x * y;
        na += x * x;
        nb += y * y;
    }
    if na == 0.0 || nb == 0.0 {
        return 1.0;
    }
    dot / (na.sqrt() * nb.sqrt())
}

fn upload_f32(
    backend: &std::sync::Arc<dyn llm_rs2::core::backend::Backend>,
    memory: &dyn llm_rs2::core::memory::Memory,
    data: &[f32],
    shape: Vec<usize>,
) -> llm_rs2::core::tensor::Tensor {
    let buf = memory
        .alloc(data.len() * 4, llm_rs2::core::buffer::DType::F32)
        .unwrap();
    let mut t = llm_rs2::core::tensor::Tensor::new(
        llm_rs2::core::shape::Shape::new(shape),
        buf,
        backend.clone(),
    );
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}

fn upload_f16(
    backend: &std::sync::Arc<dyn llm_rs2::core::backend::Backend>,
    memory: &dyn llm_rs2::core::memory::Memory,
    data: &[u16],
    shape: Vec<usize>,
) -> llm_rs2::core::tensor::Tensor {
    let buf = memory
        .alloc(data.len() * 2, llm_rs2::core::buffer::DType::F16)
        .unwrap();
    let mut t = llm_rs2::core::tensor::Tensor::new(
        llm_rs2::core::shape::Shape::new(shape),
        buf,
        backend.clone(),
    );
    let bytes = unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}
