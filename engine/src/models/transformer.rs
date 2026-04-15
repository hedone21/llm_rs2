use anyhow::{Result, anyhow};
use std::sync::Arc;

use crate::core::attention_scores::AttentionScoreAccumulator;
use crate::core::backend::Backend;
use crate::core::buffer::{Buffer, DType};
use crate::core::kv_cache::{KVCache, KVCacheOps};
use crate::core::memory::Memory;
use crate::core::offload::preload_pool::{self, PreloadPool};
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use crate::layers::transformer_layer::{LayerForwardArgs, TransformerLayer};
use crate::layers::workspace::LayerWorkspace;
use crate::models::config::{ModelArch, ModelConfig};

#[cfg(feature = "opencl")]
use crate::backend::opencl::plan::FullKernelPlan;

/// Returns true if this layer uses local (sliding window) attention.
/// Gemma3 pattern: every `pattern`-th layer (1-indexed) uses global attention;
/// all other layers use local attention.
/// Example: pattern=6 → layers 5,11,17,... (0-indexed) are global.
fn is_local_layer(layer_idx: usize, pattern: Option<usize>) -> bool {
    match pattern {
        Some(p) if p > 0 => !(layer_idx + 1).is_multiple_of(p),
        _ => false,
    }
}

/// Run `f` with the OpenCL `op_label_hint` temporarily set to `label`, if the
/// backend is an event-profiling OpenCL backend. No-op on CPU or when
/// event profiling is off. Used by the non-plan path to tag lm_head.
#[inline]
fn with_op_label<F, T>(backend: &Arc<dyn Backend>, label: &'static str, f: F) -> Result<T>
where
    F: FnOnce() -> Result<T>,
{
    #[cfg(feature = "opencl")]
    if let Some(ocl_be) = backend
        .as_any()
        .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
        && ocl_be.profile_events_enabled
    {
        ocl_be.set_op_label(label);
        let r = f();
        ocl_be.clear_op_label();
        return r;
    }
    #[cfg(not(feature = "opencl"))]
    let _ = (backend, label);
    f()
}

pub struct TransformerModel {
    pub config: ModelConfig,
    pub layers: Vec<TransformerLayer>,
    /// embed_tokens is always kept on CPU for model loading.
    /// When the main backend is GPU, `gpu_embed_tokens` holds a device-side copy
    /// so that `gather_embed` can run entirely on the GPU without CPU round-trips.
    pub embed_tokens: Tensor,
    pub norm: Tensor,
    /// lm_head weight tensor.
    /// When `lm_head_on_cpu` is true, this lives on the CPU backend even when the
    /// main backend is GPU (avoids `CL_INVALID_BUFFER_SIZE` for large vocabs).
    pub lm_head: Tensor,
    /// When true, `lm_head` is on CPU and the final matmul must be done via CPU
    /// fallback: read x from GPU → CPU matmul → write logits back.
    /// This happens for models with large tied embeddings (e.g., gemma3-1b:
    /// 262144 × 1152 × 2 = ~604 MB exceeds typical mobile GPU alloc limits).
    pub lm_head_on_cpu: bool,
    /// GPU-side copy of embed_tokens for zero-sync gather on GPU backends.
    /// - Tied weights: shares the same GPU buffer as `lm_head` (zero extra memory).
    /// - Untied weights: a separate GPU copy uploaded once at load time.
    /// - CPU backend: None (unnecessary, gather runs on CPU directly).
    pub(crate) gpu_embed_tokens: Option<Tensor>,
    /// CPU backend used for embed_tokens gather when the main backend is GPU.
    /// None when the main backend is already CPU.
    pub(crate) cpu_backend: Option<Arc<dyn Backend>>,
    /// Persistent thread pool for offload preload operations.
    /// Lazily initialized on first `forward_into_offload` call.
    /// Uses `Mutex` for interior mutability (`forward_into_offload` takes `&self`).
    pub(crate) preload_pool: std::sync::Mutex<Option<PreloadPool>>,
}

pub struct TransformerModelForwardArgs<'a, C: KVCacheOps = KVCache> {
    pub input_tokens: &'a Tensor,
    pub start_pos: usize,
    pub kv_caches: &'a mut [C],
    pub backend: &'a Arc<dyn Backend>,
    pub memory: &'a dyn Memory,
    pub logits_out: &'a mut Tensor,
    pub x_gen: Option<&'a mut Tensor>,
    pub workspace: Option<&'a mut LayerWorkspace>,
    /// Pre-allocated prefill workspace (caller-owned, max_seq_len-sized).
    /// When `Some`, `forward_into` reuses it instead of allocating a fresh one
    /// for each prefill call. Essential on NVIDIA OpenCL where repeated
    /// alloc/free cycles accumulate deferred-release pressure (→ CL_OUT_OF_RESOURCES).
    pub prefill_workspace: Option<&'a mut crate::layers::workspace::PrefillWorkspace>,
    /// Optional attention score accumulator for H2O-style eviction.
    /// When active, post-softmax scores are captured from tracked layers.
    pub score_accumulator: Option<&'a mut AttentionScoreAccumulator>,
    /// Optional per-op profiler.
    pub profiler: Option<&'a mut crate::profile::ops::OpProfiler>,
    /// Optional SWIFT skip configuration for layer skipping.
    pub skip_config: Option<&'a crate::core::skip_config::SkipConfig>,
    /// Optional importance collector for Layer Skip QCF.
    /// When provided during prefill, captures per-layer cosine similarity.
    pub importance_collector: Option<&'a mut crate::core::qcf::ImportanceCollector>,
    /// When true, only compute logits for the last sequence position.
    /// Saves ~3GB GPU memory for long-context prefill (e.g., eval-ll with 5K+ tokens).
    /// logits_out shape should be [1, 1, vocab_size] instead of [1, seq_len, vocab_size].
    pub logits_last_only: bool,
    /// Optional D2O variance collector for layer-level allocation.
    /// When provided during prefill, captures per-layer attention column-sums.
    pub variance_collector:
        Option<&'a mut crate::core::pressure::d2o_layer_alloc::D2OVarianceCollector>,
}

impl TransformerModel {
    pub fn load(model_path: &str, backend: Arc<dyn Backend>, _memory: &dyn Memory) -> Result<Self> {
        Self::load_with_dtype(model_path, backend, _memory, DType::F16)
    }

    pub fn load_with_dtype(
        model_path: &str,
        backend: Arc<dyn Backend>,
        _memory: &dyn Memory,
        weight_dtype: DType,
    ) -> Result<Self> {
        use crate::models::loader::safetensors::SafetensorsSource;
        let source = SafetensorsSource::open(model_path, weight_dtype)?;
        crate::models::loader::load_model(&source, backend, _memory)
    }

    /// Load a model from a GGUF file.
    ///
    /// The weight dtype is determined from the GGUF file's tensor types.
    /// No conversion is performed -- tensors are loaded in their native format.
    pub fn load_gguf(
        model_path: &str,
        backend: Arc<dyn Backend>,
        memory: &dyn Memory,
    ) -> Result<Self> {
        use crate::models::loader::gguf::GgufSource;
        let source = GgufSource::open(std::path::Path::new(model_path))?;
        crate::models::loader::load_model(&source, backend, memory)
    }

    /// Migrate all model weight tensors from CPU to GPU zero-copy memory.
    ///
    /// Creates `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR) for each weight tensor,
    /// maps it for CPU access, copies the data from the original CPU buffer,
    /// and keeps the mapping alive. This gives single-VMA dual access:
    /// - `as_ptr()` → mapped host pointer (valid for CPU backend)
    /// - `cl_mem()` → OpenCL handle (valid for GPU kernels)
    ///
    /// Returns the number of tensors migrated.
    #[cfg(feature = "opencl")]
    pub fn migrate_weights_to_gpu(
        &mut self,
        _gpu_mem: &dyn Memory,
        gpu_backend: &Arc<dyn Backend>,
    ) -> Result<usize> {
        let mut count = 0;
        #[cfg(feature = "opencl")]
        let ocl_queue = gpu_backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            .map(|be| be.queue.clone());
        #[cfg(not(feature = "opencl"))]
        let ocl_queue: Option<()> = None;
        let queue = ocl_queue.ok_or_else(|| anyhow!("GPU backend is not OpenCL"))?;

        let migrate_one = |t: &Tensor| -> Result<Tensor> {
            let size = t.size();
            let src_ptr = t.as_ptr();
            if src_ptr.is_null() {
                return Err(anyhow!("Cannot migrate null-pointer buffer"));
            }
            // Create ALLOC_HOST_PTR buffer, map, copy data from CPU buffer.
            let ub =
                crate::buffer::unified_buffer::UnifiedBuffer::new(queue.clone(), size, t.dtype())?;
            ub.map()?; // clEnqueueMapBuffer → host pointer
            // SAFETY: src_ptr is non-null (checked above), ub.as_mut_ptr() is valid after map().
            unsafe {
                std::ptr::copy_nonoverlapping(src_ptr, ub.as_mut_ptr(), size);
            }
            // Keep mapped: as_ptr()=valid, cl_mem()=valid → dual access.
            let buf: Arc<dyn Buffer> = Arc::new(ub);
            Ok(Tensor::new(t.shape().clone(), buf, gpu_backend.clone()))
        };
        // Layer weights (always small — Q4 ~6MB, F16 ~16MB max per tensor)
        macro_rules! migrate {
            ($t:expr) => {
                $t = migrate_one(&$t)?;
                count += 1;
            };
        }
        for layer in &mut self.layers {
            migrate!(layer.wq);
            migrate!(layer.wk);
            migrate!(layer.wv);
            migrate!(layer.wo);
            migrate!(layer.w_gate);
            migrate!(layer.w_up);
            migrate!(layer.w_down);
            migrate!(layer.attention_norm);
            migrate!(layer.ffn_norm);
            if let Some(ref mut bias) = layer.qkv_bias {
                migrate!(bias.bq);
                migrate!(bias.bk);
                migrate!(bias.bv);
            }
            if let Some(ref t) = layer.q_norm {
                layer.q_norm = Some(migrate_one(t)?);
                count += 1;
            }
            if let Some(ref t) = layer.k_norm {
                layer.k_norm = Some(migrate_one(t)?);
                count += 1;
            }
            if let Some(ref t) = layer.pre_ffn_norm {
                layer.pre_ffn_norm = Some(migrate_one(t)?);
                count += 1;
            }
            if let Some(ref t) = layer.post_ffn_norm {
                layer.post_ffn_norm = Some(migrate_one(t)?);
                count += 1;
            }
        }
        migrate!(self.norm);
        // lm_head + embed_tokens: may be large (>512MB for big vocab).
        // Migrate if possible, otherwise keep on CPU with fallback paths.
        let max_alloc = gpu_backend.max_single_alloc();
        if !self.lm_head_on_cpu {
            if self.lm_head.size() <= max_alloc {
                migrate!(self.lm_head);
            } else {
                self.lm_head_on_cpu = true;
            }
        }
        if self.embed_tokens.size() <= max_alloc {
            self.gpu_embed_tokens = Some(migrate_one(&self.embed_tokens)?);
            count += 1;
        }
        // Set cpu_backend for gather_embed fallback path
        if self.cpu_backend.is_none() {
            self.cpu_backend =
                Some(Arc::new(crate::backend::cpu::CpuBackend::new()) as Arc<dyn Backend>);
        }
        Ok(count)
    }

    /// Make GPU-primary weight buffers CPU-accessible by mapping UnifiedBuffers.
    ///
    /// When `--backend opencl` with `use_zero_copy=true`, `copy_from()` creates
    /// `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR) for each weight. These start
    /// unmapped (as_ptr()=null). This method maps each buffer so the CPU backend
    /// can read weights via `as_ptr()` for GPU→CPU SwitchHw or tensor partition.
    ///
    /// When `use_zero_copy=false`, weights are in `OpenCLBuffer` (device-only).
    /// This method reads each GPU-only weight into a new `UnifiedBuffer`
    /// (ALLOC_HOST_PTR + mapped), replacing the tensor. Single VMA, no
    /// PSS double-counting.
    ///
    /// Call at startup when resilience/tensor-partition is enabled and backend is GPU.
    #[cfg(feature = "opencl")]
    pub fn map_weights_for_cpu(&mut self, gpu_backend: &Arc<dyn Backend>) -> Result<usize> {
        let ocl_be = gpu_backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            .ok_or_else(|| anyhow!("GPU backend is not OpenCL"))?;
        let queue = ocl_be.queue.clone();

        let mut count = 0;

        let map_one = |t: &Tensor, be: &Arc<dyn Backend>| -> Result<(Tensor, bool)> {
            // Already CPU-accessible (mapped UnifiedBuffer, etc.) → just map if needed.
            if !t.buffer().as_ptr().is_null() {
                t.buffer().map_for_cpu()?; // no-op if already mapped
                return Ok((t.clone(), false));
            }
            // Device-only buffer (OpenCLBuffer): read to new UnifiedBuffer.
            let size = t.size();
            let ub =
                crate::buffer::unified_buffer::UnifiedBuffer::new(queue.clone(), size, t.dtype())?;
            ub.map()?;
            // SAFETY: ub.as_mut_ptr() is valid after map(). read_buffer copies from GPU.
            let dst = unsafe { std::slice::from_raw_parts_mut(ub.as_mut_ptr(), size) };
            be.read_buffer(t, dst)?;
            let buf: Arc<dyn Buffer> = Arc::new(ub);
            Ok((Tensor::new(t.shape().clone(), buf, be.clone()), true))
        };

        macro_rules! map_weight {
            ($t:expr) => {
                let (new, changed) = map_one(&$t, gpu_backend)?;
                if changed {
                    $t = new;
                    count += 1;
                }
            };
        }
        for layer in &mut self.layers {
            map_weight!(layer.wq);
            map_weight!(layer.wk);
            map_weight!(layer.wv);
            map_weight!(layer.wo);
            map_weight!(layer.w_gate);
            map_weight!(layer.w_up);
            map_weight!(layer.w_down);
            map_weight!(layer.attention_norm);
            map_weight!(layer.ffn_norm);
            if let Some(ref mut bias) = layer.qkv_bias {
                map_weight!(bias.bq);
                map_weight!(bias.bk);
                map_weight!(bias.bv);
            }
            if let Some(ref t) = layer.q_norm {
                let (new, _) = map_one(t, gpu_backend)?;
                layer.q_norm = Some(new);
                count += 1;
            }
            if let Some(ref t) = layer.k_norm {
                let (new, _) = map_one(t, gpu_backend)?;
                layer.k_norm = Some(new);
                count += 1;
            }
            if let Some(ref t) = layer.pre_ffn_norm {
                let (new, _) = map_one(t, gpu_backend)?;
                layer.pre_ffn_norm = Some(new);
                count += 1;
            }
            if let Some(ref t) = layer.post_ffn_norm {
                let (new, _) = map_one(t, gpu_backend)?;
                layer.post_ffn_norm = Some(new);
                count += 1;
            }
        }
        map_weight!(self.norm);
        if !self.lm_head_on_cpu {
            map_weight!(self.lm_head);
        }
        // embed_tokens: check gpu_embed_tokens first
        if let Some(ref t) = self.gpu_embed_tokens {
            let (new, _) = map_one(t, gpu_backend)?;
            self.gpu_embed_tokens = Some(new);
            count += 1;
        }
        Ok(count)
    }

    /// Prepare tensor partitioning for CPU-GPU cooperative inference.
    ///
    /// Splits FFN gate/up projections row-wise into GPU and CPU partitions:
    ///   - `[0, split_row)` stays on the current (GPU) backend
    ///   - `[split_row, out_dim)` is tagged with `cpu_backend`
    ///
    /// Attention (wq/wk/wv/wo) and FFN down are not partitioned — their
    /// merge overhead exceeds the compute benefit, especially during prefill.
    ///
    /// Each slice is pre-copied into an independent buffer via `Backend::copy_from()`,
    /// ensuring GPU slices have a valid `cl_mem` handle for OpenCL kernel dispatch.
    /// Call after `map_weights_for_cpu()` so weights are CPU-accessible.
    ///
    /// Returns the number of weights actually partitioned.
    pub fn prepare_tensor_partition(
        &mut self,
        gpu_ratio: f32,
        cpu_backend: &Arc<dyn Backend>,
    ) -> Result<usize> {
        use crate::layers::tensor_partition::{PartitionContext, split_weight};

        let mut count = 0;
        for layer in &mut self.layers {
            // Only partition FFN gate/up (large enough, no merge overhead concern)
            let gate = split_weight(&layer.w_gate, gpu_ratio, cpu_backend)?;
            let up = split_weight(&layer.w_up, gpu_ratio, cpu_backend)?;
            count += 2;

            layer.partition_ctx = Some(PartitionContext {
                gpu_ratio,
                cpu_backend: cpu_backend.clone(),
                gate,
                up,
            });
        }
        Ok(count)
    }

    /// Convert all Q4_0 weight tensors to noshuffle SOA layout for Adreno-optimized GEMV.
    ///
    /// For each Q4_0 weight tensor on GPU, runs the SOA conversion pipeline
    /// (GPU nibble rearrange + CPU transpose) and registers the result in the
    /// backend's noshuffle SOA registry. After this, `matmul_q4_0()` will
    /// automatically dispatch to the noshuffle GEMV kernel for decode (m==1).
    ///
    /// Returns the number of weight tensors converted.
    #[cfg(feature = "opencl")]
    pub fn prepare_noshuffle_buffers(&self, backend: &Arc<dyn Backend>) -> Result<usize> {
        use crate::backend::opencl::{NoshuffleSoaEntry, OpenCLBackend, get_cl_mem};

        let ocl_be = backend
            .as_any()
            .downcast_ref::<OpenCLBackend>()
            .ok_or_else(|| anyhow!("Not OpenCL backend"))?;

        let mut count = 0;

        // Helper closure: convert one weight tensor if it is Q4_0 on GPU.
        let process_weight = |tensor: &Tensor| -> Result<bool> {
            if tensor.dtype() != DType::Q4_0 {
                return Ok(false);
            }
            let cl_mem = get_cl_mem(tensor.buffer().as_ref())
                .map_err(|_| anyhow!("Weight buffer has no cl_mem"))?;
            let dims = tensor.shape().dims();
            // Weight shape: [rows, cols] = [ne01, ne00]
            let (ne01, ne00) = (dims[0], dims[1]);
            let num_blocks = ne01 * ne00 / 32; // QK4_0 = 32

            let (q_buf, d_buf, q_img) =
                ocl_be.convert_q4_0_to_noshuffle(cl_mem, num_blocks, ne00, ne01)?;

            let key = cl_mem.as_ptr() as usize;
            ocl_be.register_noshuffle_soa(
                key,
                NoshuffleSoaEntry {
                    q_buf,
                    d_buf,
                    q_img,
                    ne00,
                    ne01,
                },
            );
            Ok(true)
        };

        for layer in &self.layers {
            for weight in [
                &layer.wq,
                &layer.wk,
                &layer.wv,
                &layer.wo,
                &layer.w_gate,
                &layer.w_up,
                &layer.w_down,
            ] {
                if process_weight(weight)? {
                    count += 1;
                }
            }
        }

        // lm_head (only if on GPU)
        if !self.lm_head_on_cpu && process_weight(&self.lm_head)? {
            count += 1;
        }

        eprintln!(
            "[Backend] Q4_0 noshuffle SOA prepared: {} weight tensors",
            count
        );
        Ok(count)
    }

    /// Migrate all model weight tensors to CUDA pinned host memory (CudaHostBuffer).
    ///
    /// Uses `Backend::copy_from()` to allocate pinned memory and memcpy weight data.
    /// On Jetson (UMA), pinned memory is zero-copy accessible from both CPU and GPU,
    /// enabling cuBLAS to access the weight data via device pointers.
    ///
    /// Returns the number of tensors migrated.
    #[cfg(feature = "cuda")]
    pub fn migrate_weights_to_cuda(&mut self, gpu_backend: &Arc<dyn Backend>) -> Result<usize> {
        let mut count = 0;

        let migrate_one = |t: &Tensor| -> Result<Tensor> { gpu_backend.copy_from(t) };

        macro_rules! migrate {
            ($t:expr) => {
                $t = migrate_one(&$t)?;
                count += 1;
            };
        }

        for layer in &mut self.layers {
            migrate!(layer.wq);
            migrate!(layer.wk);
            migrate!(layer.wv);
            migrate!(layer.wo);
            migrate!(layer.w_gate);
            migrate!(layer.w_up);
            migrate!(layer.w_down);
            migrate!(layer.attention_norm);
            migrate!(layer.ffn_norm);
            if let Some(ref mut bias) = layer.qkv_bias {
                migrate!(bias.bq);
                migrate!(bias.bk);
                migrate!(bias.bv);
            }
            if let Some(ref t) = layer.q_norm {
                layer.q_norm = Some(migrate_one(t)?);
                count += 1;
            }
            if let Some(ref t) = layer.k_norm {
                layer.k_norm = Some(migrate_one(t)?);
                count += 1;
            }
            if let Some(ref t) = layer.pre_ffn_norm {
                layer.pre_ffn_norm = Some(migrate_one(t)?);
                count += 1;
            }
            if let Some(ref t) = layer.post_ffn_norm {
                layer.post_ffn_norm = Some(migrate_one(t)?);
                count += 1;
            }
        }

        migrate!(self.norm);

        // lm_head + embed_tokens: may be large, but Jetson has plenty of unified memory.
        // Still apply the same size guard for safety.
        let max_alloc = gpu_backend.max_single_alloc();
        if !self.lm_head_on_cpu {
            if self.lm_head.size() <= max_alloc {
                migrate!(self.lm_head);
            } else {
                self.lm_head_on_cpu = true;
            }
        }
        if self.embed_tokens.size() <= max_alloc {
            self.gpu_embed_tokens = Some(migrate_one(&self.embed_tokens)?);
            count += 1;
        }

        // Set cpu_backend for gather_embed fallback path
        if self.cpu_backend.is_none() {
            self.cpu_backend =
                Some(Arc::new(crate::backend::cpu::CpuBackend::new()) as Arc<dyn Backend>);
        }

        Ok(count)
    }

    pub fn forward(
        &self,
        input_tokens: &Tensor,
        start_pos: usize,
        kv_caches: &mut [KVCache],
        backend: &Arc<dyn Backend>,
        memory: &dyn Memory,
    ) -> Result<Tensor> {
        // input_tokens: [Batch, Seq] (indices) - but wait, Embedding lookup is usually done first.
        // If input_tokens is indices, we need `embedding_lookup`.
        // If input_tokens is already embeddings, we proceed.

        // Let's implement embedding lookup on the fly here or assume input is indices?
        // Usually `forward` takes indices.
        // Embedding layer:
        // x = embed_tokens[input_tokens]

        // This requires gather/embedding support in backend or manual copy.
        // Let's do manual copy for now.

        let batch_size = input_tokens.shape().dims()[0];
        let seq_len = input_tokens.shape().dims()[1];
        let hidden_size = self.config.hidden_size;

        let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
        let mut x = Tensor::new(
            Shape::new(vec![batch_size, seq_len, hidden_size]),
            x_buf,
            backend.clone(),
        );

        // Embedding lookup: CPU gather + upload to backend buffer
        self.gather_embed(input_tokens, &mut x, backend)?;

        // Gemma3: scale embeddings by sqrt(hidden_size)
        if let Some(scale) = self.config.embed_scale {
            backend.scale(&mut x, scale)?;
        }

        let is_gemma3 = self.config.arch == ModelArch::Gemma3;

        // Iterate layers
        for (i, layer) in self.layers.iter().enumerate() {
            let rope_theta = if is_gemma3 && is_local_layer(i, self.config.sliding_window_pattern) {
                self.config
                    .rope_local_theta
                    .unwrap_or(self.config.rope_theta) as f32
            } else {
                self.config.rope_theta as f32
            };
            let is_local = if is_gemma3 {
                Some(is_local_layer(i, self.config.sliding_window_pattern))
            } else {
                None
            };

            layer.forward(LayerForwardArgs {
                x: &mut x,
                kv_cache: &mut kv_caches[i],
                start_pos,
                backend,
                memory,
                rms_norm_eps: self.config.rms_norm_eps as f32,
                rope_theta,
                workspace: None,
                rms_norm_add_unit: is_gemma3,
                use_gelu_tanh: is_gemma3,
                is_local_attn: is_local,
                local_attn_window: self.config.sliding_window,
                need_scores: false,
                head_dim: self.config.head_dim,
                profiler: None,
                layer_id: i,
                skip_attn: false,
                skip_mlp: false,
            })?;
        }

        // Final Norm
        backend.rms_norm(
            &mut x,
            &self.norm,
            self.config.rms_norm_eps as f32,
            is_gemma3,
        )?;

        // Head
        let vocab_size = self.config.vocab_size;
        let logits_buf = memory.alloc(batch_size * seq_len * vocab_size * 4, DType::F32)?;
        let mut logits = Tensor::new(
            Shape::new(vec![batch_size, seq_len, vocab_size]),
            logits_buf,
            backend.clone(),
        );

        if self.lm_head_on_cpu {
            self.lm_head_matmul_cpu(&x, &mut logits, backend)?;
        } else {
            backend.matmul_transposed(&x, &self.lm_head, &mut logits)?;
        }

        Ok(logits)
    }

    /// Comprehensive forward pass that writes logits into a pre-allocated buffer.
    /// Optionally accepts x_gen and workspace for memory optimization during generation.
    ///
    /// Eviction is the caller's responsibility (via `CacheManager`).
    /// Score accumulation is handled internally since it requires per-layer iteration.
    pub fn forward_into<C: KVCacheOps>(&self, args: TransformerModelForwardArgs<C>) -> Result<()> {
        let input_tokens = args.input_tokens;
        let start_pos = args.start_pos;
        let kv_caches = args.kv_caches;
        let backend = args.backend;
        let memory = args.memory;
        let logits_out = args.logits_out;
        let x_gen = args.x_gen;
        let mut workspace = args.workspace;
        let mut profiler = args.profiler;

        let mut score_accumulator = args.score_accumulator;
        let skip_config = args.skip_config;
        let mut importance_collector = args.importance_collector;
        let mut variance_collector = args.variance_collector;

        let batch_size = input_tokens.shape().dims()[0];
        let seq_len = input_tokens.shape().dims()[1];
        let hidden_size = self.config.hidden_size;

        // 1. Embedding lookup
        // Use provided x_gen buffer if available and seq_len == 1
        let mut x = if seq_len == 1 {
            if let Some(xb) = x_gen {
                (*xb).clone()
            } else {
                let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
                Tensor::new(
                    Shape::new(vec![batch_size, seq_len, hidden_size]),
                    x_buf,
                    backend.clone(),
                )
            }
        } else {
            let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
            Tensor::new(
                Shape::new(vec![batch_size, seq_len, hidden_size]),
                x_buf,
                backend.clone(),
            )
        };

        // Embedding lookup: CPU gather + upload to backend buffer
        self.gather_embed(input_tokens, &mut x, backend)?;

        // Gemma3: scale embeddings by sqrt(hidden_size)
        if let Some(scale) = self.config.embed_scale {
            backend.scale(&mut x, scale)?;
        }

        let is_gemma3 = self.config.arch == ModelArch::Gemma3;

        // Prefill workspace: prefer caller-provided (pre-allocated at max_seq_len,
        // reused across calls), fall back to allocating one for this call.
        //
        // Caller reuse eliminates ~10 GPU buffer alloc/release cycles per prefill,
        // which is critical on NVIDIA OpenCL where deferred release pressure
        // accumulates and eventually triggers CL_OUT_OF_RESOURCES.
        let caller_prefill_ws = args.prefill_workspace;
        let mut owned_prefill_ws: Option<crate::layers::workspace::PrefillWorkspace> = None;
        let no_prefill_ws = std::env::var("LLM_NO_PREFILL_WS").is_ok();
        let mut needs_ws_sync = false; // synchronize before owned_prefill_ws drop
        if caller_prefill_ws.is_none() && seq_len > 1 && backend.is_gpu() && !no_prefill_ws {
            use crate::layers::workspace::{PrefillWorkspace, WorkspaceConfig as WsCfg};
            let ws_cfg = WsCfg {
                batch_size,
                dim: hidden_size,
                q_dim: self.config.num_attention_heads * self.config.head_dim,
                k_dim: self.config.num_key_value_heads * self.config.head_dim,
                v_dim: self.config.num_key_value_heads * self.config.head_dim,
                ffn_hidden: self.config.intermediate_size,
                n_heads: self.config.num_attention_heads,
                max_seq_len: 0,
            };
            owned_prefill_ws =
                PrefillWorkspace::new(&ws_cfg, seq_len, memory, backend.clone()).ok();
            needs_ws_sync = owned_prefill_ws.is_some();
        }
        // Unified handle: caller-provided wins, otherwise owned (if created).
        let mut prefill_ws: Option<&mut crate::layers::workspace::PrefillWorkspace> =
            caller_prefill_ws.or(owned_prefill_ws.as_mut());

        // Check if GPU-side score accumulator is active.
        // When active, attention_gen writes scores to a persistent GPU buffer and
        // reduce_layer runs on-device — no CPU readback needed per layer.
        // This means we set need_scores=false for the layer forward path.
        #[cfg(feature = "opencl")]
        let gpu_score_active = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            .and_then(|ocl_be| ocl_be.gpu_score_acc())
            .is_some_and(|acc| acc.is_active());
        #[cfg(not(feature = "opencl"))]
        let gpu_score_active = false;

        // 2. Iterate layers
        for (i, layer) in self.layers.iter().enumerate() {
            // GPU acc active -> GPU handles score collection internally in attention_gen.
            // CPU accumulator still needs need_scores=false to avoid redundant CPU path.
            let need_scores = if gpu_score_active {
                false
            } else {
                score_accumulator
                    .as_ref()
                    .is_some_and(|acc| acc.should_track_layer(i))
            };

            let (s_attn, s_mlp) =
                skip_config.map_or((false, false), |sc| (sc.skip_attn(i), sc.skip_mlp(i)));

            let rope_theta = if is_gemma3 && is_local_layer(i, self.config.sliding_window_pattern) {
                self.config
                    .rope_local_theta
                    .unwrap_or(self.config.rope_theta) as f32
            } else {
                self.config.rope_theta as f32
            };
            let is_local = if is_gemma3 {
                Some(is_local_layer(i, self.config.sliding_window_pattern))
            } else {
                None
            };

            // Snapshot hidden state before layer for importance collection
            if let Some(ref mut coll) = importance_collector {
                let x_data = x.as_slice::<f32>();
                coll.snapshot_before(x_data, seq_len, hidden_size);
            }

            // Prefill with workspace: call forward_prefill directly to avoid
            // carrying prefill_ws through LayerForwardArgs (which changes the
            // struct layout and triggers ARM64 codegen regression — see ae62391).
            if seq_len > 1 {
                if let Some(pws) = prefill_ws.as_deref_mut() {
                    let dim = hidden_size;
                    layer.forward_prefill(
                        &mut x,
                        &mut kv_caches[i],
                        start_pos,
                        backend,
                        memory,
                        self.config.rms_norm_eps as f32,
                        rope_theta,
                        need_scores,
                        self.config.head_dim,
                        batch_size,
                        seq_len,
                        dim,
                        s_attn,
                        s_mlp,
                        is_gemma3,
                        is_gemma3,
                        is_local,
                        self.config.sliding_window,
                        Some(pws),
                        i,
                        variance_collector.as_deref_mut(),
                        profiler.as_deref_mut().map(|p| &mut p.prefill),
                    )?;
                } else {
                    layer.forward(LayerForwardArgs {
                        x: &mut x,
                        kv_cache: &mut kv_caches[i],
                        start_pos,
                        backend,
                        memory,
                        rms_norm_eps: self.config.rms_norm_eps as f32,
                        rope_theta,
                        workspace: None,
                        need_scores,
                        head_dim: self.config.head_dim,
                        profiler: profiler.as_deref_mut(),
                        layer_id: i,
                        skip_attn: s_attn,
                        skip_mlp: s_mlp,
                        rms_norm_add_unit: is_gemma3,
                        use_gelu_tanh: is_gemma3,
                        is_local_attn: is_local,
                        local_attn_window: self.config.sliding_window,
                    })?;
                }
            } else {
                layer.forward(LayerForwardArgs {
                    x: &mut x,
                    kv_cache: &mut kv_caches[i],
                    start_pos,
                    backend,
                    memory,
                    rms_norm_eps: self.config.rms_norm_eps as f32,
                    rope_theta,
                    workspace: workspace.as_deref_mut(),
                    need_scores,
                    head_dim: self.config.head_dim,
                    profiler: profiler.as_deref_mut(),
                    layer_id: i,
                    skip_attn: s_attn,
                    skip_mlp: s_mlp,
                    rms_norm_add_unit: is_gemma3,
                    use_gelu_tanh: is_gemma3,
                    is_local_attn: is_local,
                    local_attn_window: self.config.sliding_window,
                })?;
            }

            // Record importance after layer forward
            if let Some(ref mut coll) = importance_collector {
                let x_data = x.as_slice::<f32>();
                coll.record_after(
                    x_data,
                    seq_len,
                    hidden_size,
                    i,
                    crate::core::qcf::SubLayer::Full,
                );
            }

            // Capture attention scores for H2O/H2O+ accumulator
            if let (Some(acc), Some(ws)) = (&mut score_accumulator, &workspace)
                && acc.should_track_layer(i)
            {
                let cache_seq_len = kv_caches[i].current_pos();
                let n_heads_q = self.config.num_attention_heads;
                let stride = ws.scores.len() / n_heads_q;

                if acc.n_kv_heads() > 0 {
                    let n_kv_heads = self.config.num_key_value_heads;
                    acc.accumulate_layer_gqa(
                        &ws.scores,
                        stride,
                        cache_seq_len,
                        n_heads_q,
                        n_kv_heads,
                    );
                } else {
                    acc.accumulate_layer(&ws.scores, stride, cache_seq_len, n_heads_q);
                }
            }
        }

        // Flush step-local importance (per-layer MAX) into cumulative importance
        if let Some(ref mut acc) = score_accumulator {
            acc.end_step();
        }

        // GPU score accumulator: flush step scores into cumulative importance.
        // This runs kernel_score_end_step + kernel_score_clear on the device.
        #[cfg(feature = "opencl")]
        if gpu_score_active
            && let Some(ocl_be) = backend
                .as_any()
                .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            && let Some(gpu_acc) = ocl_be.gpu_score_acc_mut()
        {
            let cache_seq_len = kv_caches[0].current_pos();
            gpu_acc.end_step(ocl_be.queue.as_core(), cache_seq_len)?;
        }

        // 3. Final Norm
        backend.rms_norm(
            &mut x,
            &self.norm,
            self.config.rms_norm_eps as f32,
            is_gemma3,
        )?;

        // 4. Head — optionally only compute last position's logits to save VRAM
        if args.logits_last_only && seq_len > 1 {
            // Extract last hidden state: x[..., seq_len-1, :] → [1, 1, hidden_size]
            let hidden_size = self.config.hidden_size;
            let last_offset = (seq_len - 1) * hidden_size;
            let last_buf = memory.alloc(hidden_size * 4, DType::F32)?;
            let mut x_last = Tensor::new(
                Shape::new(vec![1, 1, hidden_size]),
                last_buf,
                backend.clone(),
            );
            backend.copy_slice(&x, &mut x_last, last_offset, 0, hidden_size)?;
            if self.lm_head_on_cpu {
                self.lm_head_matmul_cpu(&x_last, logits_out, backend)?;
            } else {
                with_op_label(backend, "lm_head", || {
                    backend.matmul_transposed(&x_last, &self.lm_head, logits_out)
                })?;
            }
        } else if self.lm_head_on_cpu {
            self.lm_head_matmul_cpu(&x, logits_out, backend)?;
        } else {
            with_op_label(backend, "lm_head", || {
                backend.matmul_transposed(&x, &self.lm_head, logits_out)
            })?;
        }

        // Synchronize before owned PrefillWorkspace drop — ensures all GPU kernels
        // referencing workspace buffers have completed before clReleaseMemObject.
        // Without this, Adreno drivers may reclaim mapped memory prematurely.
        if needs_ws_sync {
            backend.synchronize()?;
        }

        Ok(())
    }

    /// Build a pre-bound GPU kernel execution plan for decode (seq_len=1).
    /// Returns None if the backend is not OpenCL or if plan construction fails.
    #[cfg(feature = "opencl")]
    pub fn build_plan(
        &self,
        x: &Tensor,
        logits: &Tensor,
        ws: &LayerWorkspace,
        kv_caches: &mut [KVCache],
        backend: &Arc<dyn Backend>,
    ) -> Option<FullKernelPlan> {
        use crate::backend::opencl::get_cl_mem;
        use crate::backend::opencl::plan::*;
        use crate::core::kv_cache::KVLayout;

        let weight_dtype = self.layers[0].wq.dtype();
        let is_f16 = weight_dtype == crate::core::buffer::DType::F16;
        let is_q4_0 = weight_dtype == crate::core::buffer::DType::Q4_0;
        // GPU plan supports F16 weights or Q4_0 with noshuffle SOA conversion
        if !is_f16 && !is_q4_0 {
            return None;
        }

        // kernel_add_row_bias expects F32 bias buffers. If any layer has
        // a QKV bias that isn't F32, fall back to the legacy path.
        if self.config.has_qkv_bias {
            for layer in &self.layers {
                if let Some(ref bias) = layer.qkv_bias {
                    if bias.bq.dtype() != crate::core::buffer::DType::F32
                        || bias.bk.dtype() != crate::core::buffer::DType::F32
                        || bias.bv.dtype() != crate::core::buffer::DType::F32
                    {
                        return None;
                    }
                }
            }
        }

        if backend.name() != "OpenCL" || kv_caches.is_empty() {
            return None;
        }

        // Helper macro to extract cl_mem from tensor (avoids closure lifetime issues)
        macro_rules! cl {
            ($t:expr) => {
                match get_cl_mem($t.buffer().as_ref()) {
                    Ok(m) => m,
                    Err(_) => return None,
                }
            };
        }

        let dim = self.config.hidden_size;
        let head_dim = self.config.head_dim;
        let n_kv_heads = self.config.num_key_value_heads;
        let capacity = kv_caches[0].capacity();

        let (kv_pos_stride, kv_head_stride) = if kv_caches[0].layout() == KVLayout::HeadMajor {
            (head_dim as i32, (capacity * head_dim) as i32)
        } else {
            ((n_kv_heads * head_dim) as i32, head_dim as i32)
        };

        let ocl_backend = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()?;

        // For Q4_0 weights, verify noshuffle SOA entries are available.
        // If any weight lacks a q_img (noshuffle image), fall back to legacy path.
        if is_q4_0 {
            let first_wq_key = cl!(self.layers[0].wq).as_ptr() as usize;
            match ocl_backend.lookup_noshuffle_soa(first_wq_key) {
                Some(e) if e.q_img.is_some() => {}
                _ => return None,
            }
        }

        // Helper: lookup noshuffle SOA entry and return NoshufflePlanEntry if available.
        let ns_entry = |tensor: &Tensor| -> Option<NoshufflePlanEntry<'_>> {
            if !is_q4_0 {
                return None;
            }
            let key = cl!(tensor).as_ptr() as usize;
            let entry = ocl_backend.lookup_noshuffle_soa(key)?;
            let q_img = entry.q_img.as_ref()?;
            Some(NoshufflePlanEntry {
                q_img,
                d_buf: &entry.d_buf,
                ne00: entry.ne00,
                ne01: entry.ne01,
            })
        };

        // Collect per-layer buffer handles
        let mut layer_bufs = Vec::new();
        let mut kv_bufs_vec = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            // Extract optional QKV bias cl_mem handles. Non-bias models
            // short-circuit this block with (None, None, None). For bias
            // models (e.g. Qwen2), the `cl!` macro returns None from the
            // outer function if any buffer lookup fails.
            let (bq, bk, bv) = match layer.qkv_bias.as_ref() {
                Some(bias) => (Some(cl!(bias.bq)), Some(cl!(bias.bk)), Some(cl!(bias.bv))),
                None => (None, None, None),
            };
            layer_bufs.push(LayerBufs {
                wq: cl!(layer.wq),
                wk: cl!(layer.wk),
                wv: cl!(layer.wv),
                wo: cl!(layer.wo),
                w_gate: cl!(layer.w_gate),
                w_up: cl!(layer.w_up),
                w_down: cl!(layer.w_down),
                attn_norm: cl!(layer.attention_norm),
                ffn_norm: cl!(layer.ffn_norm),
                bq,
                bk,
                bv,
                wq_noshuffle: ns_entry(&layer.wq),
                wk_noshuffle: ns_entry(&layer.wk),
                wv_noshuffle: ns_entry(&layer.wv),
                wo_noshuffle: ns_entry(&layer.wo),
                w_gate_noshuffle: ns_entry(&layer.w_gate),
                w_up_noshuffle: ns_entry(&layer.w_up),
                w_down_noshuffle: ns_entry(&layer.w_down),
            });
            kv_bufs_vec.push(KvBufs {
                k_cache: cl!(kv_caches[i].k_buffer),
                v_cache: cl!(kv_caches[i].v_buffer),
            });
        }

        // Build noshuffle GEMV programs if Q4_0 (compile per unique ne01 dimension).
        let noshuffle_programs = if is_q4_0 {
            // Collect unique ne01 values from all noshuffle entries
            let mut ne01_set: Vec<usize> = layer_bufs
                .iter()
                .flat_map(|lb| {
                    [
                        lb.wq_noshuffle.as_ref(),
                        lb.wk_noshuffle.as_ref(),
                        lb.wv_noshuffle.as_ref(),
                        lb.wo_noshuffle.as_ref(),
                        lb.w_gate_noshuffle.as_ref(),
                        lb.w_up_noshuffle.as_ref(),
                        lb.w_down_noshuffle.as_ref(),
                    ]
                    .into_iter()
                    .flatten()
                    .map(|e| e.ne01)
                })
                .collect();
            // Also include lm_head ne01 if applicable
            if !self.lm_head_on_cpu
                && let Some(ref e) = ns_entry(&self.lm_head)
            {
                ne01_set.push(e.ne01);
            }
            ne01_set.sort_unstable();
            ne01_set.dedup();
            match build_noshuffle_programs(
                &ocl_backend.device,
                &ocl_backend.context,
                &ocl_backend.cl_opts,
                &ne01_set,
            ) {
                Ok(progs) => Some(progs),
                Err(e) => {
                    log::warn!("Failed to build noshuffle programs for plan: {}", e);
                    return None;
                }
            }
        } else {
            None
        };

        // lm_head noshuffle entry (Q4_0 on GPU only)
        let lm_head_noshuffle = if !self.lm_head_on_cpu {
            ns_entry(&self.lm_head)
        } else {
            None
        };

        // Mirror the runtime gate in `attention_gen` — flash attention has no
        // score output, so an active GPU score accumulator forces the legacy
        // attention path. The CPU-side `score_accumulator` is already checked
        // by callers of `build_plan` (see `generate.rs`).
        let plan_needs_scores = ocl_backend
            .gpu_score_acc()
            .is_some_and(|acc| acc.is_active());

        let full_config = FullPlanConfig {
            context: &ocl_backend.context,
            f16_program: &ocl_backend.f16_program,
            f16_l4_program: ocl_backend.f16_l4_program.as_ref(),
            simple_ops_program: &ocl_backend.simple_ops_program,
            q4_0_program: &ocl_backend.q4_0_program,
            flash_attn_f32_f16_program_dk64: ocl_backend.flash_attn_f32_f16_program_dk64.as_ref(),
            flash_attn_f32_f16_program_dk128: ocl_backend.flash_attn_f32_f16_program_dk128.as_ref(),
            needs_attention_scores: plan_needs_scores,
            layer_bufs,
            x_buf: cl!(x),
            q_buf: cl!(ws.q),
            k_buf: cl!(ws.k),
            v_buf: cl!(ws.v),
            out_attn_buf: cl!(ws.out_attn),
            attn_out_buf: cl!(ws.attn_out),
            gate_buf: cl!(ws.gate),
            up_buf: cl!(ws.up),
            down_buf: cl!(ws.down),
            residual_buf: cl!(ws.residual),
            kv_bufs: kv_bufs_vec,
            final_norm_buf: cl!(self.norm),
            lm_head_buf: if self.lm_head_on_cpu {
                None
            } else {
                Some(cl!(self.lm_head))
            },
            logits_buf: cl!(logits),
            dim,
            n_heads_q: self.config.num_attention_heads,
            n_kv_heads,
            head_dim,
            ffn_hidden: self.config.intermediate_size,
            vocab_size: self.config.vocab_size,
            rms_norm_eps: self.config.rms_norm_eps as f32,
            rope_theta: self.config.rope_theta as f32,
            kv_capacity: capacity,
            kv_pos_stride,
            kv_head_stride,
            is_nosub: ocl_backend.is_nosub(),
            noshuffle_programs,
            lm_head_noshuffle,
        };

        match build_full_plan(&full_config) {
            Ok(plan) => {
                log::info!(
                    "GPU kernel plan built ({} layers, capacity={}, q4_noshuffle={})",
                    self.layers.len(),
                    capacity,
                    is_q4_0,
                );
                Some(plan)
            }
            Err(e) => {
                log::warn!("Failed to build GPU kernel plan: {}", e);
                None
            }
        }
    }

    /// Execute a pre-built GPU kernel plan for a single decode token.
    /// Falls back to forward_into() on plan invalidation.
    #[cfg(feature = "opencl")]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_plan(
        &self,
        plan: &FullKernelPlan,
        input_tokens: &Tensor,
        start_pos: usize,
        x_gen: &mut Tensor,
        kv_caches: &mut [KVCache],
        logits_out: &mut Tensor,
        backend: &Arc<dyn Backend>,
    ) -> Result<bool> {
        // 1. Embedding lookup: CPU gather + upload to GPU x-buffer
        self.gather_embed(input_tokens, x_gen, backend)?;

        // 2. Execute plan (all layers + final norm + optionally lm_head)
        let ocl_backend = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            .ok_or_else(|| anyhow!("Backend is not OpenCL"))?;

        match plan.execute(ocl_backend, start_pos, kv_caches) {
            Ok(()) => {
                // If lm_head was skipped in the plan (on CPU), run CPU fallback.
                if plan.lm_head.is_none() {
                    self.lm_head_matmul_cpu(x_gen, logits_out, backend)?;
                }
                Ok(true)
            }
            Err(_) => Ok(false), // plan invalidated, caller should rebuild
        }
    }

    /// Build a pre-bound GPU kernel execution plan for KIVI decode (seq_len=1).
    /// Returns None if the backend is not OpenCL, plan construction fails,
    /// or KIVI GPU buffers are not available.
    #[cfg(feature = "opencl")]
    pub fn build_plan_for_kivi(
        &self,
        x: &Tensor,
        logits: &Tensor,
        ws: &LayerWorkspace,
        kv_caches: &[crate::core::kivi_cache::KiviCache],
        backend: &Arc<dyn Backend>,
    ) -> Option<FullKernelPlan> {
        use crate::backend::opencl::get_cl_mem;
        use crate::backend::opencl::plan::*;

        if self.config.has_qkv_bias {
            return None;
        }
        if self.layers[0].wq.dtype() != crate::core::buffer::DType::F16 {
            return None;
        }
        if backend.name() != "OpenCL" || kv_caches.is_empty() {
            return None;
        }
        if !kv_caches[0].is_gpu() {
            return None;
        }

        macro_rules! cl {
            ($t:expr) => {
                match get_cl_mem($t.buffer().as_ref()) {
                    Ok(m) => m,
                    Err(_) => return None,
                }
            };
        }

        let ocl_backend = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()?;

        let kivi_q2_program = ocl_backend.kivi_q2_program.as_ref()?;

        let bits = kv_caches[0].bits();
        let use_native = ocl_backend.is_nosub() && ocl_backend.has_kivi_attn_kernel(bits);

        let dim = self.config.hidden_size;
        let head_dim = self.config.head_dim;
        let n_kv_heads = self.config.num_key_value_heads;
        let max_seq_len = kv_caches[0].capacity();

        let mut layer_bufs = Vec::new();
        let mut kivi_kv_bufs_vec = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            layer_bufs.push(LayerBufs {
                wq: cl!(layer.wq),
                wk: cl!(layer.wk),
                wv: cl!(layer.wv),
                wo: cl!(layer.wo),
                w_gate: cl!(layer.w_gate),
                w_up: cl!(layer.w_up),
                w_down: cl!(layer.w_down),
                attn_norm: cl!(layer.attention_norm),
                ffn_norm: cl!(layer.ffn_norm),
                // KIVI path shares the same QKV bias handling as the
                // standard plan; Task 3 will populate from layer.qkv_bias.
                bq: None,
                bk: None,
                bv: None,
                // KIVI currently only supports F16 weights — no noshuffle.
                wq_noshuffle: None,
                wk_noshuffle: None,
                wv_noshuffle: None,
                wo_noshuffle: None,
                w_gate_noshuffle: None,
                w_up_noshuffle: None,
                w_down_noshuffle: None,
            });

            let plan_bufs = kv_caches[i].get_plan_gpu_buffers()?;
            kivi_kv_bufs_vec.push(KiviKvBufs {
                res_k: cl!(plan_bufs.res_k),
                res_v: cl!(plan_bufs.res_v),
                q2k: cl!(plan_bufs.q2k),
                q2v: cl!(plan_bufs.q2v),
                attn_k: cl!(plan_bufs.attn_k),
                attn_v: cl!(plan_bufs.attn_v),
                res_cap: plan_bufs.res_cap,
            });
        }

        let kivi_config = KiviFullPlanConfig {
            context: &ocl_backend.context,
            f16_program: &ocl_backend.f16_program,
            f16_l4_program: ocl_backend.f16_l4_program.as_ref(),
            simple_ops_program: &ocl_backend.simple_ops_program,
            q4_0_program: &ocl_backend.q4_0_program,
            kivi_q2_program,
            kivi_attn_program: ocl_backend.kivi_attn_program.as_ref(),
            layer_bufs,
            x_buf: cl!(x),
            q_buf: cl!(ws.q),
            k_buf: cl!(ws.k),
            v_buf: cl!(ws.v),
            out_attn_buf: cl!(ws.out_attn),
            attn_out_buf: cl!(ws.attn_out),
            gate_buf: cl!(ws.gate),
            up_buf: cl!(ws.up),
            down_buf: cl!(ws.down),
            residual_buf: cl!(ws.residual),
            kivi_kv_bufs: kivi_kv_bufs_vec,
            final_norm_buf: cl!(self.norm),
            lm_head_buf: if self.lm_head_on_cpu {
                None
            } else {
                Some(cl!(self.lm_head))
            },
            logits_buf: cl!(logits),
            dim,
            n_heads_q: self.config.num_attention_heads,
            n_kv_heads,
            head_dim,
            ffn_hidden: self.config.intermediate_size,
            vocab_size: self.config.vocab_size,
            rms_norm_eps: self.config.rms_norm_eps as f32,
            rope_theta: self.config.rope_theta as f32,
            max_seq_len,
            bits,
            use_native_attn: use_native,
            is_nosub: ocl_backend.is_nosub(),
        };

        match build_kivi_full_plan(&kivi_config) {
            Ok(plan) => {
                log::info!(
                    "KIVI GPU kernel plan built ({} layers, bits={}, native_attn={})",
                    self.layers.len(),
                    bits,
                    use_native
                );
                Some(plan)
            }
            Err(e) => {
                log::warn!("Failed to build KIVI GPU kernel plan: {}", e);
                None
            }
        }
    }

    /// Execute a pre-built KIVI GPU kernel plan for a single decode token.
    /// Falls back if plan is invalidated.
    #[cfg(feature = "opencl")]
    #[allow(clippy::too_many_arguments)]
    pub fn execute_plan_for_kivi(
        &self,
        plan: &FullKernelPlan,
        input_tokens: &Tensor,
        start_pos: usize,
        x_gen: &mut Tensor,
        kv_caches: &mut [crate::core::kivi_cache::KiviCache],
        logits_out: &mut Tensor,
        backend: &Arc<dyn Backend>,
    ) -> Result<bool> {
        self.gather_embed(input_tokens, x_gen, backend)?;

        let ocl_backend = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            .ok_or_else(|| anyhow!("Backend is not OpenCL"))?;

        match plan.execute(ocl_backend, start_pos, kv_caches) {
            Ok(()) => {
                if plan.lm_head.is_none() {
                    self.lm_head_matmul_cpu(x_gen, logits_out, backend)?;
                }
                Ok(true)
            }
            Err(_) => Ok(false),
        }
    }

    /// Offload-optimized forward pass with adaptive multi-layer prefetch pipeline.
    ///
    /// Uses `PrefetchController` to dynamically adjust how far ahead layers are
    /// preloaded. When preload is slower than forward (pipeline stall), depth increases.
    /// When there is slack, depth decreases to save memory.
    ///
    /// **Fire-and-forget preloads**: A preload task for layer `i+depth` is submitted
    /// to a persistent thread pool during layer `i`'s forward pass, then collected
    /// when layer `i+depth` is needed. This avoids per-token thread spawn/join overhead.
    ///
    /// Uses raw pointers to safely manage concurrent `&mut` access to disjoint cache
    /// elements. SAFETY: layer indices are guaranteed non-overlapping — at most one
    /// thread accesses any given `kv_caches[j]` at a time.
    ///
    /// Score accumulator is forced to None (offload mode doesn't support eviction).
    pub fn forward_into_offload<C: crate::core::kv_cache::PrefetchableCache>(
        &self,
        args: TransformerModelForwardArgs<'_, C>,
        prefetch: &mut crate::core::offload::prefetch::PrefetchController,
    ) -> Result<()> {
        let input_tokens = args.input_tokens;
        let start_pos = args.start_pos;
        let kv_caches = args.kv_caches;
        let backend = args.backend;
        let memory = args.memory;
        let logits_out = args.logits_out;
        let x_gen = args.x_gen;
        let mut workspace = args.workspace;
        let _importance_collector = args.importance_collector; // unused in offload path

        let batch_size = input_tokens.shape().dims()[0];
        let seq_len = input_tokens.shape().dims()[1];
        let hidden_size = self.config.hidden_size;

        // 1. Embedding lookup (identical to forward_into)
        let mut x = if seq_len == 1 {
            if let Some(xb) = x_gen {
                (*xb).clone()
            } else {
                let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
                Tensor::new(
                    Shape::new(vec![batch_size, seq_len, hidden_size]),
                    x_buf,
                    backend.clone(),
                )
            }
        } else {
            let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
            Tensor::new(
                Shape::new(vec![batch_size, seq_len, hidden_size]),
                x_buf,
                backend.clone(),
            )
        };
        // Embedding lookup: CPU gather + upload to backend buffer
        self.gather_embed(input_tokens, &mut x, backend)?;

        // Gemma3: scale embeddings by sqrt(hidden_size)
        if let Some(scale) = self.config.embed_scale {
            backend.scale(&mut x, scale)?;
        }

        let is_gemma3 = self.config.arch == ModelArch::Gemma3;
        let num_layers = self.layers.len();
        let depth = prefetch.depth();
        let caches_ptr = kv_caches.as_mut_ptr();

        // Lazy-init persistent thread pool (sized to max_depth for full concurrency).
        // Mutex is locked once per token — zero contention.
        let mut pool_guard = self.preload_pool.lock().unwrap();
        let pool = pool_guard.get_or_insert_with(|| PreloadPool::new(prefetch.max_depth()));

        // 2. Synchronous initial preload: layers [0..depth)
        // Retained layers (preloaded=true) skip via early-return.
        for j in 0..depth.min(num_layers) {
            // SAFETY: j < num_layers, no concurrent access during sync phase
            unsafe { (*caches_ptr.add(j)).preload() }?;
        }

        // Track pending results: pending[j] = receiver for kv_caches[j]'s preload
        // SAFETY: raw pointer access to kv_caches elements. We guarantee that:
        // 1. Each pool task accesses exactly one element kv_caches[far_idx]
        // 2. The main thread accesses kv_caches[i] for forward
        // 3. far_idx != i (far_idx = i + depth, depth >= 1)
        // 4. Each element is accessed by at most one thread at a time:
        //    - A preload task for element j completes before j is used for forward
        //    - release_buffers(j) happens after forward(j) completes
        let mut pending: Vec<Option<std::sync::mpsc::Receiver<preload_pool::PreloadResult>>> =
            (0..num_layers).map(|_| None).collect();

        // Fire initial background preloads for layers [depth..2*depth)
        #[allow(clippy::needless_range_loop)]
        for j in depth..(2 * depth).min(num_layers) {
            pending[j] = Some(unsafe {
                pool.submit(
                    caches_ptr.add(j) as *mut (),
                    preload_pool::preload_erased::<C>,
                )
            });
        }

        // 3. Layer loop
        for i in 0..num_layers {
            // Collect preload result for layer i (if any).
            if let Some(rx) = pending[i].take() {
                match rx.recv() {
                    Ok(preload_pool::PreloadResult {
                        result: Ok(()),
                        duration,
                    }) => {
                        prefetch.record_preload(duration);
                    }
                    Ok(preload_pool::PreloadResult { result: Err(e), .. }) => {
                        log::warn!("L{i} preload failed: {e}, falling back to sync");
                    }
                    Err(_) => {
                        log::error!("L{i} preload worker dropped result channel");
                    }
                }
            }

            // Fire preload for layer i + depth (if within bounds)
            let far_idx = i + depth;
            if far_idx < num_layers && pending[far_idx].is_none() {
                pending[far_idx] = Some(unsafe {
                    pool.submit(
                        caches_ptr.add(far_idx) as *mut (),
                        preload_pool::preload_erased::<C>,
                    )
                });
            }

            // Forward current layer
            let current = unsafe { &mut *caches_ptr.add(i) };
            let fwd_t0 = std::time::Instant::now();
            let rope_theta_i = if is_gemma3 && is_local_layer(i, self.config.sliding_window_pattern)
            {
                self.config
                    .rope_local_theta
                    .unwrap_or(self.config.rope_theta) as f32
            } else {
                self.config.rope_theta as f32
            };
            let is_local_i = if is_gemma3 {
                Some(is_local_layer(i, self.config.sliding_window_pattern))
            } else {
                None
            };
            self.layers[i].forward(LayerForwardArgs {
                x: &mut x,
                kv_cache: current,
                start_pos,
                backend,
                memory,
                rms_norm_eps: self.config.rms_norm_eps as f32,
                rope_theta: rope_theta_i,
                workspace: workspace.as_deref_mut(),
                need_scores: false,
                head_dim: self.config.head_dim,
                profiler: None,
                layer_id: i,
                skip_attn: false,
                skip_mlp: false,
                rms_norm_add_unit: is_gemma3,
                use_gelu_tanh: is_gemma3,
                is_local_attn: is_local_i,
                local_attn_window: self.config.sliding_window,
            })?;
            let fwd_dur = fwd_t0.elapsed();
            prefetch.record_forward(fwd_dur);

            // Cross-token retention: keep first `depth` layers' buffers alive
            // so next token's preload() early-returns (preloaded=true).
            if i < depth {
                current.retain_preload();
            }

            // Release consumed layer's buffers (skip retained layers)
            if i > 0 && (i - 1) >= depth {
                // SAFETY: layer i-1 forward is complete, preload task (if any) collected
                unsafe { (*caches_ptr.add(i - 1)).release_buffers() };
            }
        }

        // Collect any remaining pending preloads
        for rx in pending.into_iter().flatten() {
            let _ = rx.recv();
        }

        // Release last layer's buffers (unless retained)
        if num_layers >= 1 && (num_layers - 1) >= depth {
            kv_caches[num_layers - 1].release_buffers();
        }

        // Adjust depth for the next token
        prefetch.adjust();

        // 4. Final Norm + Head (identical to forward_into)
        backend.rms_norm(
            &mut x,
            &self.norm,
            self.config.rms_norm_eps as f32,
            is_gemma3,
        )?;
        if self.lm_head_on_cpu {
            self.lm_head_matmul_cpu(&x, logits_out, backend)?;
        } else {
            backend.matmul_transposed(&x, &self.lm_head, logits_out)?;
        }

        Ok(())
    }

    /// Run lm_head matmul on CPU when `self.lm_head_on_cpu` is true.
    ///
    /// 1. Synchronize GPU (ensure x is ready).
    /// 2. Read the normalized hidden state `x` from GPU to a CPU buffer.
    /// 3. Perform `matmul_transposed(x_cpu, lm_head_cpu, logits_cpu)` on CPU.
    /// 4. Write the F32 logits back to the GPU `logits_out` buffer.
    ///
    /// This is only called for models with huge tied embeddings (e.g., gemma3-1b's
    /// 604 MB embed_tokens) that exceed the GPU single-buffer allocation limit.
    fn lm_head_matmul_cpu(
        &self,
        x: &Tensor,
        logits_out: &mut Tensor,
        backend: &Arc<dyn Backend>,
    ) -> Result<()> {
        let cpu_be = self
            .cpu_backend
            .as_ref()
            .ok_or_else(|| anyhow!("lm_head_on_cpu requires cpu_backend"))?;

        // 1. Synchronize GPU to ensure x is fully computed.
        backend.synchronize()?;

        // 2. Read x from GPU into a CPU-backed tensor.
        let x_size = x.size(); // F32 bytes
        let x_cpu_buf = Arc::new(crate::buffer::shared_buffer::SharedBuffer::new(
            x_size,
            DType::F32,
        ));
        let x_cpu = Tensor::new(
            x.shape().clone(),
            x_cpu_buf as Arc<dyn Buffer>,
            cpu_be.clone(),
        );
        // SAFETY: SharedBuffer guarantees a valid writable pointer of `x_size` bytes.
        let x_dst = unsafe { std::slice::from_raw_parts_mut(x_cpu.as_mut_ptr(), x_size) };
        backend.read_buffer(x, x_dst)?;

        // 3. Allocate CPU logits buffer and run matmul.
        let logits_size = logits_out.size();
        let logits_cpu_buf = Arc::new(crate::buffer::shared_buffer::SharedBuffer::new(
            logits_size,
            DType::F32,
        ));
        let mut logits_cpu = Tensor::new(
            logits_out.shape().clone(),
            logits_cpu_buf as Arc<dyn Buffer>,
            cpu_be.clone(),
        );

        cpu_be.matmul_transposed(&x_cpu, &self.lm_head, &mut logits_cpu)?;

        // 4. Write CPU logits to GPU buffer.
        let logits_ptr = logits_cpu.as_ptr();
        // SAFETY: logits_cpu is a valid SharedBuffer with logits_size bytes.
        let logits_bytes = unsafe { std::slice::from_raw_parts(logits_ptr, logits_size) };
        backend.write_buffer(logits_out, logits_bytes)?;

        Ok(())
    }

    /// Perform embedding lookup using CPU gather, then upload the result to the
    /// target backend buffer `dst`.
    ///
    /// When `self.cpu_backend` is Some (i.e. main backend is GPU), embed_tokens
    /// lives on CPU, so we:
    ///   1. Copy input token indices to a CPU buffer (if they are on GPU).
    ///   2. Gather into a temporary CPU F32 buffer.
    ///   3. Write the bytes to the pre-allocated GPU `dst` tensor via `write_buffer`.
    ///
    /// When main backend is CPU, `cpu_backend` is None and we call `gather` directly.
    fn gather_embed(
        &self,
        input_tokens: &Tensor,
        dst: &mut Tensor,
        backend: &Arc<dyn Backend>,
    ) -> Result<()> {
        if let Some(ref gpu_embed) = self.gpu_embed_tokens {
            // GPU-direct path: both embed table and indices are on GPU.
            // Single kernel launch, no CPU round-trip or blocking sync.
            backend.gather(gpu_embed, input_tokens, dst)
        } else if let Some(ref cpu_be) = self.cpu_backend {
            // Fallback GPU path (gpu_embed_tokens not available):
            // indices may be on GPU — read them to a CPU buffer first.
            let num_tokens = input_tokens.size() / 4; // u32 elements
            let hidden_size = self.config.hidden_size;
            let galloc = crate::memory::galloc::Galloc::new();

            // Read indices to CPU (even if already CPU this is a cheap memcpy)
            let idx_size = num_tokens * 4;
            let idx_buf = galloc
                .alloc(idx_size, DType::F32) // DType doesn't matter for raw bytes here
                .map_err(|e| anyhow!("gather_embed: idx alloc failed: {e}"))?;
            let cpu_indices = Tensor::new(input_tokens.shape().clone(), idx_buf, cpu_be.clone());
            let idx_bytes_mut =
                unsafe { std::slice::from_raw_parts_mut(cpu_indices.as_mut_ptr(), idx_size) };
            backend.read_buffer(input_tokens, idx_bytes_mut)?;

            // Gather on CPU
            let tmp_size = num_tokens * hidden_size * 4; // F32 bytes
            let tmp_buf = galloc
                .alloc(tmp_size, DType::F32)
                .map_err(|e| anyhow!("gather_embed: tmp alloc failed: {e}"))?;
            let mut tmp = Tensor::new(
                Shape::new(vec![1, num_tokens, hidden_size]),
                tmp_buf,
                cpu_be.clone(),
            );
            cpu_be.gather(&self.embed_tokens, &cpu_indices, &mut tmp)?;

            // Upload F32 result to GPU dst buffer
            let bytes =
                unsafe { std::slice::from_raw_parts(tmp.as_ptr(), num_tokens * hidden_size * 4) };
            backend.write_buffer(dst, bytes)
        } else {
            // CPU path: direct gather (embed_tokens already on CPU backend)
            backend.gather(&self.embed_tokens, input_tokens, dst)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::core::buffer::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use crate::memory::galloc::Galloc;
    use std::sync::Arc;

    /// Build a minimal TransformerModel-like object with CPU backend and
    /// a small embed_tokens table for gather_embed testing.
    fn make_cpu_model_with_embed(vocab: usize, dim: usize) -> (TransformerModel, Arc<dyn Backend>) {
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();

        // Build embed table: row i = [i as f32; dim]
        let buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for r in 0..vocab {
                for c in 0..dim {
                    slice[r * dim + c] = r as f32;
                }
            }
        }
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), buf, cpu_be.clone());

        // Build a trivial norm weight (all 1s)
        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        {
            let slice =
                unsafe { std::slice::from_raw_parts_mut(norm_buf.as_mut_ptr() as *mut f32, dim) };
            for v in slice.iter_mut() {
                *v = 1.0;
            }
        }
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, cpu_be.clone());
        let lm_head = norm.clone(); // dummy

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 2,
            arch: crate::models::config::ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        };

        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: None, // CPU-only model
            cpu_backend: None,
            preload_pool: std::sync::Mutex::new(None),
        };
        (model, cpu_be)
    }

    #[test]
    fn test_gather_embed_cpu_path() {
        // CPU model: gather_embed should produce correct rows from embed table
        let vocab = 8usize;
        let dim = 4usize;
        let (model, cpu_be) = make_cpu_model_with_embed(vocab, dim);

        let mem = Galloc::new();
        let idx_buf = mem.alloc(2 * 4, DType::F32).unwrap();
        // indices = [3, 5]
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 2) };
        idx_slice[0] = 3;
        idx_slice[1] = 5;
        let indices = Tensor::new(Shape::new(vec![1, 2]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(2 * dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 2, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, 2 * dim) };
        // Row 3 → all 3.0, row 5 → all 5.0
        for v in &result[..dim] {
            assert_eq!(*v, 3.0, "row 3 mismatch");
        }
        for v in &result[dim..] {
            assert_eq!(*v, 5.0, "row 5 mismatch");
        }
    }

    #[test]
    fn test_gather_embed_cpu_backend_none_means_direct() {
        // When cpu_backend is None, gather_embed routes through backend.gather directly
        let vocab = 4usize;
        let dim = 2usize;
        let (model, cpu_be) = make_cpu_model_with_embed(vocab, dim);
        assert!(model.cpu_backend.is_none());

        let mem = Galloc::new();
        let idx_buf = mem.alloc(4, DType::F32).unwrap();
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 1) };
        idx_slice[0] = 2;
        let indices = Tensor::new(Shape::new(vec![1, 1]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 1, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, dim) };
        for v in result {
            assert_eq!(*v, 2.0, "row 2 mismatch");
        }
    }

    #[test]
    fn test_is_local_layer() {
        // pattern=None → all global
        assert!(!is_local_layer(0, None));
        assert!(!is_local_layer(5, None));

        // pattern=0 → all global (degenerate)
        assert!(!is_local_layer(0, Some(0)));
        assert!(!is_local_layer(3, Some(0)));

        // Gemma3 1B: pattern=6 → layer indices 5,11,17,23 are global (1-indexed: 6,12,18,24)
        // All other layers are local.
        assert!(is_local_layer(0, Some(6))); // layer 1 → local
        assert!(is_local_layer(1, Some(6))); // layer 2 → local
        assert!(!is_local_layer(5, Some(6))); // layer 6 → global
        assert!(is_local_layer(6, Some(6))); // layer 7 → local
        assert!(!is_local_layer(11, Some(6))); // layer 12 → global
        assert!(is_local_layer(10, Some(6))); // layer 11 → local
    }

    #[test]
    fn test_embed_scale_applied() {
        // Verify that embed_scale is applied after gather_embed.
        // Build a Gemma3-like model config with embed_scale and run gather_embed + scale.
        use crate::backend::cpu::CpuBackend;
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();

        let vocab = 4usize;
        let dim = 2usize;
        let scale = 2.0f32;

        // embed table: row i = [1.0; dim]
        let buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for v in slice.iter_mut() {
                *v = 1.0;
            }
        }
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), buf, cpu_be.clone());

        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        {
            let s =
                unsafe { std::slice::from_raw_parts_mut(norm_buf.as_mut_ptr() as *mut f32, dim) };
            s.iter_mut().for_each(|v| *v = 1.0);
        }
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, cpu_be.clone());

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 10000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: false,
            eos_token_id: 1,
            arch: ModelArch::Gemma3,
            rope_local_theta: Some(10000.0),
            sliding_window: Some(512),
            sliding_window_pattern: Some(6),
            query_pre_attn_scalar: None,
            embed_scale: Some(scale),
            weight_prefix: String::new(),
        };

        let lm_head = norm.clone();
        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: None,
            cpu_backend: None,
            preload_pool: std::sync::Mutex::new(None),
        };

        // Gather token 0 → should be [1.0, 1.0], then scale → [2.0, 2.0]
        let idx_buf = mem.alloc(4, DType::F32).unwrap();
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 1) };
        idx_slice[0] = 0;
        let indices = Tensor::new(Shape::new(vec![1, 1]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 1, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();
        cpu_be.scale(&mut dst, scale).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, dim) };
        for &v in result {
            assert!(
                (v - scale).abs() < 1e-5,
                "Expected {scale}, got {v} after embed_scale"
            );
        }
    }

    #[test]
    fn test_write_buffer_default_impl() {
        // Verify the default write_buffer impl copies bytes correctly (CPU backend)
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();
        let buf = mem.alloc(8, DType::F32).unwrap();
        let mut t = Tensor::new(Shape::new(vec![2]), buf, cpu_be.clone());

        let src: Vec<u8> = (0u8..8).collect();
        cpu_be.write_buffer(&mut t, &src).unwrap();

        let got = unsafe { std::slice::from_raw_parts(t.as_ptr(), 8) };
        assert_eq!(got, src.as_slice());
    }

    #[test]
    fn test_gather_embed_gpu_direct_path() {
        // When gpu_embed_tokens is Some, gather_embed should use it directly
        // (simulated with CPU backend acting as the "GPU" backend).
        let vocab = 8usize;
        let dim = 4usize;
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();

        // Build embed table on "CPU" side
        let buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for r in 0..vocab {
                for c in 0..dim {
                    slice[r * dim + c] = r as f32;
                }
            }
        }
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), buf, cpu_be.clone());

        // Build "GPU" embed table (same data, simulates GPU copy)
        let gpu_buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let src = unsafe {
                std::slice::from_raw_parts(embed_tokens.as_ptr() as *const u8, vocab * dim * 4)
            };
            let dst =
                unsafe { std::slice::from_raw_parts_mut(gpu_buf.as_mut_ptr(), vocab * dim * 4) };
            dst.copy_from_slice(src);
        }
        let gpu_embed = Tensor::new(Shape::new(vec![vocab, dim]), gpu_buf, cpu_be.clone());

        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, cpu_be.clone());
        let lm_head = norm.clone();

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: true,
            eos_token_id: 2,
            arch: crate::models::config::ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        };

        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: Some(gpu_embed),
            cpu_backend: None,
            preload_pool: std::sync::Mutex::new(None),
        };

        // Gather tokens [1, 6]
        let idx_buf = mem.alloc(2 * 4, DType::F32).unwrap();
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 2) };
        idx_slice[0] = 1;
        idx_slice[1] = 6;
        let indices = Tensor::new(Shape::new(vec![1, 2]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(2 * dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 2, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, 2 * dim) };
        for v in &result[..dim] {
            assert_eq!(*v, 1.0, "row 1 mismatch");
        }
        for v in &result[dim..] {
            assert_eq!(*v, 6.0, "row 6 mismatch");
        }
    }

    #[test]
    fn test_gather_embed_gpu_path_takes_priority_over_cpu_fallback() {
        // When both gpu_embed_tokens and cpu_backend are set,
        // the GPU-direct path should take priority.
        let vocab = 4usize;
        let dim = 2usize;
        let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
        let mem = Galloc::new();

        // CPU embed table: row i = [i as f32; dim]
        let buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for r in 0..vocab {
                for c in 0..dim {
                    slice[r * dim + c] = r as f32;
                }
            }
        }
        let embed_tokens = Tensor::new(Shape::new(vec![vocab, dim]), buf, cpu_be.clone());

        // "GPU" embed table: row i = [(i + 100) as f32; dim]
        // Different values to prove the GPU path is used, not the CPU fallback.
        let gpu_buf = mem.alloc(vocab * dim * 4, DType::F32).unwrap();
        {
            let slice = unsafe {
                std::slice::from_raw_parts_mut(gpu_buf.as_mut_ptr() as *mut f32, vocab * dim)
            };
            for r in 0..vocab {
                for c in 0..dim {
                    slice[r * dim + c] = (r + 100) as f32;
                }
            }
        }
        let gpu_embed = Tensor::new(Shape::new(vec![vocab, dim]), gpu_buf, cpu_be.clone());

        let norm_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let norm = Tensor::new(Shape::new(vec![dim]), norm_buf, cpu_be.clone());
        let lm_head = norm.clone();

        let config = ModelConfig {
            vocab_size: vocab,
            hidden_size: dim,
            num_hidden_layers: 0,
            num_attention_heads: 1,
            num_key_value_heads: 1,
            intermediate_size: dim,
            rms_norm_eps: 1e-5,
            rope_theta: 500_000.0,
            head_dim: dim,
            has_qkv_bias: false,
            tie_word_embeddings: true,
            eos_token_id: 2,
            arch: crate::models::config::ModelArch::Llama,
            rope_local_theta: None,
            sliding_window: None,
            sliding_window_pattern: None,
            query_pre_attn_scalar: None,
            embed_scale: None,
            weight_prefix: String::new(),
        };

        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens,
            norm,
            lm_head,
            lm_head_on_cpu: false,
            gpu_embed_tokens: Some(gpu_embed),
            cpu_backend: Some(cpu_be.clone()), // both set
            preload_pool: std::sync::Mutex::new(None),
        };

        let idx_buf = mem.alloc(4, DType::F32).unwrap();
        let idx_slice =
            unsafe { std::slice::from_raw_parts_mut(idx_buf.as_mut_ptr() as *mut u32, 1) };
        idx_slice[0] = 2;
        let indices = Tensor::new(Shape::new(vec![1, 1]), idx_buf, cpu_be.clone());

        let dst_buf = mem.alloc(dim * 4, DType::F32).unwrap();
        let mut dst = Tensor::new(Shape::new(vec![1, 1, dim]), dst_buf, cpu_be.clone());

        model.gather_embed(&indices, &mut dst, &cpu_be).unwrap();

        let result = unsafe { std::slice::from_raw_parts(dst.as_ptr() as *const f32, dim) };
        // Should get GPU values (102.0), not CPU values (2.0)
        for v in result {
            assert_eq!(*v, 102.0, "GPU path should take priority, expected 102.0");
        }
    }
}
