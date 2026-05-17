//! [`ModelForward`] — first concrete [`Forward`] implementation (Phase 4-3).
//!
//! Wraps [`TransformerModel::forward_into`] for the standard `KVCache` path.
//! Owns the backend handle, model `Arc`, KV caches, decode workspace, lazy
//! prefill workspace, and two reusable logits tensors.
//!
//! Out of scope for 4-3 (all kept as `None` in the forward args):
//! `score_accumulator`, `skip_config`, `profiler`, `importance_collector`,
//! `variance_collector`, `layer_boundary_hook`. These are absorbed by
//! `EvictionStage` / `SwapStage` / `DecodeObserver` in Phase 4-4+.

use std::sync::Arc;

use anyhow::Result;

use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::kv_cache::{KVCache, KVLayout};
use crate::core::memory::Memory;
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use crate::layers::workspace::{LayerWorkspace, PrefillWorkspace, WorkspaceConfig};
use crate::memory::galloc::Galloc;
use crate::models::transformer::{TransformerModel, TransformerModelForwardArgs};
use crate::session::traits::{Forward, StepCtx};

/// Standard `Forward` implementation backed by [`TransformerModel::forward_into`]
/// and a `Vec<KVCache>`.
///
/// Workspace policy (Phase 4-3 §P4 "Hybrid"):
/// - `decode_workspace` is allocated eagerly in [`Self::new`] (small,
///   `[1, 1, *]`-shaped).
/// - `prefill_workspace` is allocated lazily on the first `prefill()` call
///   (large, `[1, seq_len, *]`-shaped). Reallocated if a longer prompt
///   arrives.
pub struct ModelForward {
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    cpu_backend: Arc<dyn Backend>,
    model: Arc<TransformerModel>,
    kv_caches: Vec<KVCache>,

    decode_workspace: LayerWorkspace,
    prefill_workspace: Option<PrefillWorkspace>,
    max_seq_len: usize,

    // Owned single-token decode input + per-token x_gen scratch + logits.
    // Allocated once to keep the vtable microbench signal clean (no per-step
    // GPU buffer creation).
    decode_input: Tensor,        // [1, 1] U8 (u32 token id)
    decode_x_gen: Tensor,        // [1, 1, hidden]
    logits_decode: Tensor,       // [1, 1, vocab]
    logits_prefill_last: Tensor, // [1, 1, vocab] (logits_last_only=true)

    vocab_size: usize,
}

impl ModelForward {
    /// Build a `ModelForward` ready to be passed to
    /// [`crate::session::DecodeLoopBuilder::with_forward`].
    ///
    /// `max_seq_len` caps the lazy `PrefillWorkspace` allocation. KV caches
    /// must already be sized for the same context window.
    pub fn new(
        backend: Arc<dyn Backend>,
        memory: Arc<dyn Memory>,
        cpu_backend: Arc<dyn Backend>,
        model: Arc<TransformerModel>,
        kv_caches: Vec<KVCache>,
        max_seq_len: usize,
    ) -> Result<Self> {
        let hidden_size = model.config.hidden_size;
        let vocab_size = model.config.vocab_size;

        let decode_workspace = LayerWorkspace::new(
            workspace_config_for(&model, max_seq_len),
            memory.as_ref(),
            backend.clone(),
        )?;

        let decode_input_buf = memory.alloc(4, DType::U8)?;
        let decode_input = Tensor::new(Shape::new(vec![1, 1]), decode_input_buf, backend.clone());

        let x_gen_buf = memory.alloc(hidden_size * 4, DType::F32)?;
        let decode_x_gen = Tensor::new(
            Shape::new(vec![1, 1, hidden_size]),
            x_gen_buf,
            backend.clone(),
        );

        let logits_decode = alloc_logits(memory.as_ref(), backend.clone(), vocab_size)?;
        let logits_prefill_last = alloc_logits(memory.as_ref(), backend.clone(), vocab_size)?;

        Ok(Self {
            backend,
            memory,
            cpu_backend,
            model,
            kv_caches,
            decode_workspace,
            prefill_workspace: None,
            max_seq_len,
            decode_input,
            decode_x_gen,
            logits_decode,
            logits_prefill_last,
            vocab_size,
        })
    }

    /// Borrow the underlying KV caches (Phase 4-4 `EvictionStage` will reach
    /// in via `&mut self` accessors once those traits are wired).
    pub fn kv_caches(&self) -> &[KVCache] {
        &self.kv_caches
    }

    pub fn kv_caches_mut(&mut self) -> &mut [KVCache] {
        &mut self.kv_caches
    }

    pub fn model(&self) -> &Arc<TransformerModel> {
        &self.model
    }

    /// Construct the input `[1, seq_len]` U32 tensor on the active backend.
    /// CPU-side buffer is built via `Galloc` and uploaded with
    /// `backend.copy_from`, matching the existing prefill path in
    /// `generate.rs`.
    fn build_input_tensor(&self, tokens: &[u32]) -> Result<Tensor> {
        let seq_len = tokens.len();
        let cpu_buf = Galloc::new().alloc(seq_len * 4, DType::U8)?;
        // SAFETY: cpu_buf is a freshly allocated [u8] of size seq_len*4 with
        // alignment from Galloc which satisfies u32 alignment (Galloc returns
        // 64B-aligned blocks). We immediately initialise it.
        unsafe {
            let dst = cpu_buf.as_mut_ptr() as *mut u32;
            std::ptr::copy_nonoverlapping(tokens.as_ptr(), dst, seq_len);
        }
        let cpu_tensor = Tensor::new(
            Shape::new(vec![1, seq_len]),
            cpu_buf,
            self.cpu_backend.clone(),
        );
        self.backend.copy_from(&cpu_tensor)
    }

    /// Lazy allocator for `prefill_workspace` with a seq_len realloc guard
    /// (Phase 4-3 §R4). Reuses the existing workspace when its capacity is
    /// already ≥ `seq_len`; otherwise drops and re-allocates.
    fn ensure_prefill_workspace(&mut self, seq_len: usize) -> Result<()> {
        let needs_alloc = match self.prefill_workspace.as_ref() {
            None => true,
            Some(ws) => ws.seq_len() < seq_len,
        };
        if needs_alloc {
            self.prefill_workspace = None; // drop old GPU buffers first
            let config = workspace_config_for(&self.model, self.max_seq_len);
            let ws = PrefillWorkspace::new(
                &config,
                seq_len.min(self.max_seq_len),
                self.memory.as_ref(),
                self.backend.clone(),
            )?;
            self.prefill_workspace = Some(ws);
        }
        Ok(())
    }

    /// Derive a safe `chunk_size` for prefill. CPU (max_single_alloc=usize::MAX)
    /// returns `seq_len` (no chunking needed). GPU mirrors the heuristic in
    /// `generate.rs::auto_gpu_chunk` — `min(budget/(vocab*4), max_alloc/(hidden*4), 512)`
    /// so neither the logits buffer nor activation buffers exceed device limits.
    fn derive_chunk_size(&self, seq_len: usize) -> usize {
        if !self.backend.is_gpu() {
            return seq_len;
        }
        let max_alloc = self.backend.max_single_alloc();
        if max_alloc == 0 || max_alloc == usize::MAX {
            return seq_len;
        }
        let hidden = self.model.config.hidden_size;
        let budget = max_alloc / 2;
        let by_vocab = (budget / (self.vocab_size * 4)).max(1);
        let by_hidden = (max_alloc / (hidden * 4)).max(1);
        by_vocab.min(by_hidden).min(512).min(seq_len)
    }

    /// Read a `[1, 1, vocab]` logits tensor off the backend into a `Vec<f32>`.
    /// Forces a backend sync first so async backends (CUDA/OpenCL) produce a
    /// stable snapshot.
    fn read_logits(&self, logits: &Tensor) -> Result<Vec<f32>> {
        self.backend.synchronize()?;
        let mut out = vec![0.0f32; self.vocab_size];
        // SAFETY: `out` is a freshly initialised f32 slice of length vocab_size;
        // reinterpreting as [u8; vocab_size*4] is sound for read_buffer (which
        // writes f32 bytes from the GPU buffer back into host memory). The
        // backend implementation does not retain the pointer past the call.
        unsafe {
            let bytes =
                std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, self.vocab_size * 4);
            self.backend.read_buffer(logits, bytes)?;
        }
        Ok(out)
    }
}

impl Forward for ModelForward {
    fn prefill(&mut self, tokens: &[u32]) -> Result<Vec<f32>> {
        if tokens.is_empty() {
            anyhow::bail!("ModelForward::prefill received zero tokens");
        }
        let seq_len = tokens.len();
        let chunk_size = self.derive_chunk_size(seq_len);
        // PrefillWorkspace sized for the largest chunk — single allocation
        // reused across all chunks, avoiding the seq_len-scale memory spike
        // that previously gated the happy path at 256 tokens.
        self.ensure_prefill_workspace(chunk_size)?;

        let mut chunk_start = 0;
        while chunk_start < seq_len {
            let chunk_end = (chunk_start + chunk_size).min(seq_len);
            let chunk = &tokens[chunk_start..chunk_end];
            let input_tensor = self.build_input_tensor(chunk)?;

            // Split mutable handles to avoid double-borrowing `self` inside
            // the FnArgs literal.
            let backend = self.backend.clone();
            let memory_ref: *const dyn Memory = self.memory.as_ref();
            // SAFETY: `self.memory` is owned by `self` and lives across this
            // forward_into call; the raw pointer is dereferenced only on the
            // current stack frame.
            let memory: &dyn Memory = unsafe { &*memory_ref };
            let prefill_ws = self
                .prefill_workspace
                .as_mut()
                .expect("ensure_prefill_workspace populates prefill_workspace");

            self.model.forward_into(TransformerModelForwardArgs {
                input_tokens: &input_tensor,
                start_pos: chunk_start,
                kv_caches: &mut self.kv_caches,
                backend: &backend,
                memory,
                logits_out: &mut self.logits_prefill_last,
                x_gen: None,
                workspace: None,
                prefill_workspace: Some(prefill_ws),
                score_accumulator: None,
                profiler: None,
                skip_config: None,
                importance_collector: None,
                logits_last_only: true,
                variance_collector: None,
                layer_boundary_hook: None,
            })?;

            chunk_start = chunk_end;
        }

        // Only the last chunk's last-token logits are kept; intermediate
        // chunks reused the same `logits_prefill_last` buffer in-place.
        self.read_logits(&self.logits_prefill_last)
    }

    fn step(&mut self, ctx: &StepCtx, token: u32) -> Result<Vec<f32>> {
        // Write the single token into the persistent decode_input buffer.
        // `write_buffer` is the same upload path used by the existing decode
        // loop in `generate.rs:2836`.
        let bytes = token.to_ne_bytes();
        self.backend.write_buffer(&mut self.decode_input, &bytes)?;

        // Same trick as prefill: split &mut borrows so we do not hold &self
        // and &mut self.kv_caches simultaneously inside the args literal.
        let backend = self.backend.clone();
        let memory_ref: *const dyn Memory = self.memory.as_ref();
        let memory: &dyn Memory = unsafe { &*memory_ref };

        self.model.forward_into(TransformerModelForwardArgs {
            input_tokens: &self.decode_input,
            start_pos: ctx.pos,
            kv_caches: &mut self.kv_caches,
            backend: &backend,
            memory,
            logits_out: &mut self.logits_decode,
            x_gen: Some(&mut self.decode_x_gen),
            workspace: Some(&mut self.decode_workspace),
            prefill_workspace: None,
            score_accumulator: None,
            profiler: None,
            skip_config: None,
            importance_collector: None,
            logits_last_only: false,
            variance_collector: None,
            layer_boundary_hook: None,
        })?;

        self.read_logits(&self.logits_decode)
    }

    fn finalize(&mut self) -> Result<()> {
        Ok(())
    }

    fn on_kv_prune(&mut self, _new_pos: usize) {
        // Phase 4-3 wires `NoOpEvictionStage`, so this hook never fires.
        // When `EvictionStage` learns to reach into `ModelForward::kv_caches_mut`
        // in Phase 4-4, this default no-op is overridden to keep the KV cache
        // `current_pos` in sync with the loop counter.
    }
}

fn workspace_config_for(model: &TransformerModel, max_seq_len: usize) -> WorkspaceConfig {
    let head_dim = model.config.head_dim;
    let kv_dim = model.config.num_key_value_heads * head_dim;
    WorkspaceConfig {
        batch_size: 1,
        dim: model.config.hidden_size,
        q_dim: model.config.num_attention_heads * head_dim,
        k_dim: kv_dim,
        v_dim: kv_dim,
        ffn_hidden: model.config.intermediate_size,
        n_heads: model.config.num_attention_heads,
        max_seq_len,
    }
}

fn alloc_logits(
    memory: &dyn Memory,
    backend: Arc<dyn Backend>,
    vocab_size: usize,
) -> Result<Tensor> {
    let buf = memory.alloc(vocab_size * 4, DType::F32)?;
    Ok(Tensor::new(
        Shape::new(vec![1, 1, vocab_size]),
        buf,
        backend,
    ))
}

/// Allocate a standard `KVCache` per layer using the same recipe as
/// `generate.rs:406` — `HeadMajor` layout, dynamic grow, `kv_buf_size`
/// derived from `dtype`. Exposed for `bin/probe_inference_loop.rs` so the
/// microbench does not need to copy this block.
pub fn alloc_standard_kv_caches(
    model: &TransformerModel,
    backend: Arc<dyn Backend>,
    memory: Arc<dyn Memory>,
    initial_capacity: usize,
    max_seq_len: usize,
    dtype: DType,
) -> Result<Vec<KVCache>> {
    let num_layers = model.config.num_hidden_layers;
    let kv_heads = model.config.num_key_value_heads;
    let head_dim = model.config.head_dim;

    let n_values = initial_capacity * kv_heads * head_dim;
    let kv_buf_size = match dtype {
        DType::Q4_0 => {
            use crate::core::quant::{BlockQ4_0, QK4_0};
            (n_values / QK4_0) * std::mem::size_of::<BlockQ4_0>()
        }
        _ => n_values * dtype.size(),
    };

    let mut caches = Vec::with_capacity(num_layers);
    for _ in 0..num_layers {
        let k_buf = memory.alloc_kv(kv_buf_size, dtype)?;
        let v_buf = memory.alloc_kv(kv_buf_size, dtype)?;
        let shape = Shape::new(vec![1, kv_heads, initial_capacity, head_dim]);
        let k = Tensor::new(shape.clone(), k_buf, backend.clone());
        let v = Tensor::new(shape, v_buf, backend.clone());
        caches.push(
            KVCache::new_dynamic(
                k,
                v,
                initial_capacity,
                max_seq_len,
                kv_heads,
                head_dim,
                memory.clone(),
            )
            .with_layout(KVLayout::HeadMajor),
        );
    }
    Ok(caches)
}
