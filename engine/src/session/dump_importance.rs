//! Phase 4-4: `--dump-importance` 모드 추출.
//!
//! `bin/generate.rs::main()` L1576~1652 분기를 외과적으로 이동.
//! `ImportanceCollector`를 prefill에 부착하여 layer별 importance를 JSON으로
//! stdout에 출력하고 종료한다.

use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::backend::cpu::CpuBackend;
use crate::buffer::DType;
use crate::kv::kv_cache::KVCache;
use crate::memory::Memory;
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::qcf::ImportanceCollector;
use crate::session::eval::EvalCacheKind;
use crate::shape::Shape;
use crate::tensor::Tensor;

pub struct DumpImportanceCtx {
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub model: TransformerModel,
    pub tokenizer: Tokenizer,
    pub kv_caches: Vec<KVCache>,
    pub prompt: String,
    pub vocab_size: usize,
    pub model_path: String,
}

pub fn run_dump_importance(mut ctx: DumpImportanceCtx) -> anyhow::Result<()> {
    let mut collector = ImportanceCollector::new();

    let prompt_enc = ctx
        .tokenizer
        .encode(ctx.prompt.as_str(), true)
        .map_err(|e| anyhow::anyhow!(e))?;
    let prompt_ids: Vec<u32> = prompt_enc.get_ids().to_vec();
    let prompt_len = prompt_ids.len();
    eprintln!("[Importance] Prefill {} tokens...", prompt_len);

    let cpu_buf = Galloc::new().alloc(prompt_len * 4, DType::U8)?;
    unsafe {
        let ptr = cpu_buf.as_mut_ptr() as *mut u32;
        std::ptr::copy_nonoverlapping(prompt_ids.as_ptr(), ptr, prompt_len);
    }
    let cpu_input = Tensor::new(
        Shape::new(vec![1, prompt_len]),
        cpu_buf,
        Arc::new(CpuBackend::new()),
    );
    let input_tensor = ctx.backend.copy_from(&cpu_input)?;

    let logits_buf = ctx
        .memory
        .alloc(prompt_len * ctx.vocab_size * 4, DType::F32)?;
    let mut logits = Tensor::new(
        Shape::new(vec![1, prompt_len, ctx.vocab_size]),
        logits_buf,
        ctx.backend.clone(),
    );

    // Phase α-K ①-d: forward_into → fmt round-trip (prefill, qcf:212 구조적 쌍둥이).
    // ctx.kv_caches 는 owned Vec → 시그니처 변경 불요. disjoint field borrow(model/backend/memory
    // vs kv_caches)로 closure 와 &mut 공존.
    KVCache::forward_fmt_roundtrip(&mut ctx.kv_caches, |fmts| {
        ctx.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &input_tensor,
            start_pos: 0,
            fmts,
            backend: &ctx.backend,
            memory: &*ctx.memory,
            logits_out: &mut logits,
            x_gen: None,
            workspace: None,
            logits_last_only: false,
            score_accumulator: None,
            skip_config: None,
            importance_collector: Some(&mut collector),
            cache_self_need_scores: false,
            layer_boundary_hook: None,
        })
    })?;

    let table = collector.build();

    let importance_entries: Vec<serde_json::Value> = table
        .entries()
        .iter()
        .map(|e| {
            serde_json::json!({
                "layer": e.layer_id,
                "sublayer": format!("{:?}", e.sublayer),
                "importance": e.importance,
                "opr": e.opr,
            })
        })
        .collect();

    let output = serde_json::json!({
        "model": ctx.model_path,
        "num_layers": ctx.model.config.num_hidden_layers,
        "prompt_tokens": prompt_len,
        "importance": importance_entries,
    });
    println!("{}", serde_json::to_string_pretty(&output)?);
    Ok(())
}
