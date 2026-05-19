//! Phase 4-4: standard generate fallback의 WARMUP block 추출.
//!
//! `bin/generate.rs::main()` L1822~1904를 외과적으로 이동.
//! DVFS ramp-up 유도 (forward 1회 + 50ms CPU spin) 후 KV cache 위치 리셋.
//!
//! Env overrides:
//!   - `LLMRS_SKIP_WARMUP=1`     : warmup 완전 skip
//!   - `LLMRS_WARMUP_TOKENS=N`   : warmup 토큰 수 (default 1)

use std::sync::Arc;

use rayon::prelude::*;

use crate::backend::cpu::CpuBackend;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::kv_cache::KVCache;
use crate::core::memory::Memory;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};

/// Run DVFS warmup before timed prefill.
///
/// Caller is responsible for guarding with `args.eval_ll == false` etc.
/// The function silently no-ops when `LLMRS_SKIP_WARMUP` is set.
pub fn run_warmup(
    model: &mut TransformerModel,
    backend: &Arc<dyn Backend>,
    memory: &Arc<dyn Memory>,
    kv_caches: &mut [KVCache],
    tokens: &[u32],
    vocab_size: usize,
) -> anyhow::Result<()> {
    if std::env::var("LLMRS_SKIP_WARMUP").is_ok() {
        eprintln!("[WARMUP] skipped (LLMRS_SKIP_WARMUP)");
        return Ok(());
    }

    let warmup_tokens: usize = std::env::var("LLMRS_WARMUP_TOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1)
        .max(1)
        .min(tokens.len());

    let warmup_start = std::time::Instant::now();
    let warmup_buf = Galloc::new().alloc(warmup_tokens * 4, DType::U8)?;
    unsafe {
        let ptr = warmup_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(tokens.as_ptr(), ptr, warmup_tokens);
    }
    let warmup_input = Tensor::new(
        Shape::new(vec![1, warmup_tokens]),
        warmup_buf,
        Arc::new(CpuBackend::new()),
    );
    let warmup_input = backend.copy_from(&warmup_input)?;

    let warmup_logits_shape = if warmup_tokens == 1 {
        Shape::new(vec![1, 1, vocab_size])
    } else {
        Shape::new(vec![1, warmup_tokens, vocab_size])
    };
    let warmup_logits_buf = memory.alloc(warmup_tokens * vocab_size * 4, DType::F32)?;
    let mut warmup_logits = Tensor::new(warmup_logits_shape, warmup_logits_buf, backend.clone());

    model.forward_into(TransformerModelForwardArgs {
        input_tokens: &warmup_input,
        start_pos: 0,
        kv_caches,
        backend,
        memory: memory.as_ref(),
        logits_out: &mut warmup_logits,
        x_gen: None,
        workspace: None,
        score_accumulator: None,
        profiler: None,
        skip_config: None,
        importance_collector: None,
        logits_last_only: false,
        variance_collector: None,
        prefill_workspace: None,
        layer_boundary_hook: None,
    })?;
    backend.synchronize()?;
    let warmup_ms = warmup_start.elapsed().as_secs_f64() * 1000.0;
    eprintln!("[WARMUP] tokens={} ms={:.2}", warmup_tokens, warmup_ms);

    // Brief all-core spin to push DVFS governor to max frequency.
    // 50ms is enough for walt governor to ramp up.
    let spin_until = std::time::Instant::now() + std::time::Duration::from_millis(50);
    (0..rayon::current_num_threads())
        .into_par_iter()
        .for_each(|_| {
            while std::time::Instant::now() < spin_until {
                std::hint::spin_loop();
            }
        });

    // Reset KV caches
    for cache in kv_caches.iter_mut() {
        cache.current_pos = 0;
        cache.high_water_pos = 0;
    }

    Ok(())
}
