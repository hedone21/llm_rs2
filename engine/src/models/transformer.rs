use anyhow::{Result, anyhow};
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use serde::Deserialize;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

use crate::core::attention_scores::AttentionScoreAccumulator;
use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::kv_cache::{KVCache, KVCacheOps};
use crate::core::memory::Memory;
use crate::core::offload::preload_pool::{self, PreloadPool};
use crate::core::shape::Shape;
use crate::core::tensor::Tensor;
use crate::layers::transformer_layer::{LayerForwardArgs, TransformerLayer};
use crate::layers::workspace::LayerWorkspace;
use crate::memory::galloc::Galloc;
use crate::models::config::ModelConfig;
use crate::models::mappers::create_mapper;

#[cfg(feature = "opencl")]
use crate::backend::opencl::plan::FullKernelPlan;

/// Release mmap pages after tensor data has been converted.
/// Calls MADV_DONTNEED on the page-aligned region so the kernel can reclaim
/// the source (e.g. BF16) pages that are no longer needed.
#[cfg(unix)]
fn release_source_pages(data: &[u8]) {
    const PAGE_SIZE: usize = 4096;
    let start = data.as_ptr() as usize;
    let end = start + data.len();
    let aligned_start = (start + PAGE_SIZE - 1) & !(PAGE_SIZE - 1);
    let aligned_end = end & !(PAGE_SIZE - 1);
    if aligned_end > aligned_start {
        unsafe {
            libc::madvise(
                aligned_start as *mut libc::c_void,
                aligned_end - aligned_start,
                libc::MADV_DONTNEED,
            );
        }
    }
}

pub struct TransformerModel {
    pub config: ModelConfig,
    pub layers: Vec<TransformerLayer>,
    /// embed_tokens is always kept on CPU to save GPU memory (llama.cpp strategy).
    /// gather is performed on CPU, then the result is uploaded to the GPU x-buffer.
    pub embed_tokens: Tensor,
    pub norm: Tensor,
    /// lm_head is on the device backend (GPU or CPU).
    pub lm_head: Tensor,
    /// CPU backend used for embed_tokens gather when the main backend is GPU.
    /// None when the main backend is already CPU.
    cpu_backend: Option<Arc<dyn Backend>>,
    /// Persistent thread pool for offload preload operations.
    /// Lazily initialized on first `forward_into_offload` call.
    /// Uses `Mutex` for interior mutability (`forward_into_offload` takes `&self`).
    preload_pool: std::sync::Mutex<Option<PreloadPool>>,
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
    pub use_gpu_attn: bool,
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
        let path = Path::new(model_path);

        // 1. Load Config
        let config = ModelConfig::from_json(path)?;
        let mapper = create_mapper(config.arch);

        // 2. Open Safetensors (single file or multi-shard)
        let single_path = path.join("model.safetensors");
        let index_path = path.join("model.safetensors.index.json");

        let (shard_files, weight_map): (Vec<String>, HashMap<String, usize>) =
            if single_path.exists() {
                (vec!["model.safetensors".to_string()], HashMap::new())
            } else if index_path.exists() {
                Self::parse_shard_index(&index_path)?
            } else {
                return Err(anyhow!(
                    "No model.safetensors or model.safetensors.index.json found in {}",
                    model_path
                ));
            };

        if shard_files.len() > 1 {
            eprintln!("Loading {} safetensors shards...", shard_files.len());
        }

        // Arc-wrapped mmaps for zero-copy MmapBuffer weight sharing
        let shard_mmap_arcs: Vec<Arc<memmap2::Mmap>> = shard_files
            .iter()
            .map(|f| {
                let file = File::open(path.join(f))?;
                let mmap = unsafe { MmapOptions::new().map(&file)? };
                // madvise hints for weight mmap:
                // - HUGEPAGE: request 2MB THP (reduces TLB misses ~500x)
                // - SEQUENTIAL: hint sequential readahead (matches GEMV access)
                // NOTE: MADV_WILLNEED removed — it prefaults the entire file into RSS
                // (~2.4 GB for 1B model), inflating peak memory during loading.
                // Pages are faulted on-demand during tensor conversion instead.
                #[cfg(target_os = "linux")]
                {
                    let ptr = mmap.as_ptr() as *mut libc::c_void;
                    let len = mmap.len();
                    unsafe {
                        libc::madvise(ptr, len, libc::MADV_HUGEPAGE);
                        libc::madvise(ptr, len, libc::MADV_SEQUENTIAL);
                    }
                }
                Ok(Arc::new(mmap))
            })
            .collect::<Result<_>>()?;

        let shard_tensors: Vec<SafeTensors> = shard_mmap_arcs
            .iter()
            .map(|m| SafeTensors::deserialize(m.as_ref()).map_err(|e| anyhow!("{}", e)))
            .collect::<Result<_>>()?;

        // Use CPU memory for loading
        let cpu_memory = Galloc::new();
        let cpu_backend = Arc::new(crate::backend::cpu::CpuBackend::new());

        // Helper to load a specific tensor by name (supports multi-shard lookup)
        // is_weight: true for weight matrices (use weight_dtype), false for norms/embeddings (always F32)
        let is_cpu = backend.name().contains("CPU");

        // Helper: return tensor for the right backend (skip copy_from on CPU)
        let finalize_tensor =
            |shape: Shape, buffer: Arc<dyn crate::core::buffer::Buffer>| -> Result<Tensor> {
                if is_cpu {
                    Ok(Tensor::new(shape, buffer, backend.clone()))
                } else {
                    let cpu_tensor = Tensor::new(shape, buffer, cpu_backend.clone());
                    backend.copy_from(&cpu_tensor)
                }
            };

        let load_tensor = |name: &str, is_weight: bool| -> Result<Tensor> {
            let shard_idx = weight_map.get(name).copied().unwrap_or(0);
            let tensor_view = match shard_tensors[shard_idx].tensor(name) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error finding tensor '{}' in shard {}", name, shard_idx);
                    return Err(anyhow!("{}", e));
                }
            };
            let shape = Shape::new(tensor_view.shape().to_vec());
            let num_elements: usize = tensor_view.shape().iter().product();

            let target_dtype = if is_weight { weight_dtype } else { DType::F32 };

            // F16 weight path
            if target_dtype == DType::F16 {
                // Fast path: safetensors stores F16 → reference mmap directly (0 copies)
                if tensor_view.dtype() == safetensors::Dtype::F16 && is_cpu {
                    use crate::buffer::mmap_buffer::MmapBuffer;
                    let data = tensor_view.data();
                    let data_offset =
                        data.as_ptr() as usize - shard_mmap_arcs[shard_idx].as_ptr() as usize;
                    let mmap_arc = shard_mmap_arcs[shard_idx].clone();
                    let buffer: Arc<dyn crate::core::buffer::Buffer> = Arc::new(unsafe {
                        MmapBuffer::new(mmap_arc, data_offset, data.len(), DType::F16)
                    });
                    return Ok(Tensor::new(shape, buffer, backend.clone()));
                }

                // Conversion path: convert directly into Galloc buffer (no intermediate Vec)
                use half::f16;
                let size_bytes = num_elements * 2;
                let buffer = cpu_memory.alloc(size_bytes, DType::F16)?;
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut f16, num_elements)
                };

                match tensor_view.dtype() {
                    safetensors::Dtype::F16 => {
                        let data = tensor_view.data();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                buffer.as_mut_ptr(),
                                data.len(),
                            );
                        }
                    }
                    safetensors::Dtype::BF16 => {
                        let data = tensor_view.data();
                        let src = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const u16, num_elements)
                        };
                        for (i, &b) in src.iter().enumerate() {
                            dst[i] = f16::from_f32(half::bf16::from_bits(b).to_f32());
                        }
                    }
                    safetensors::Dtype::F32 => {
                        let data = tensor_view.data();
                        let src = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, num_elements)
                        };
                        for (i, &v) in src.iter().enumerate() {
                            dst[i] = f16::from_f32(v);
                        }
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported dtype in safetensors: {:?}",
                            tensor_view.dtype()
                        ));
                    }
                }

                #[cfg(unix)]
                release_source_pages(tensor_view.data());

                return finalize_tensor(shape, buffer);
            }

            // Q4_0 path: needs intermediate f32_data for quantization
            if target_dtype == DType::Q4_0 {
                let mut f32_data = vec![0.0f32; num_elements];
                match tensor_view.dtype() {
                    safetensors::Dtype::F32 => unsafe {
                        std::ptr::copy_nonoverlapping(
                            tensor_view.data().as_ptr(),
                            f32_data.as_mut_ptr() as *mut u8,
                            tensor_view.data().len(),
                        );
                    },
                    safetensors::Dtype::BF16 => {
                        let src = unsafe {
                            std::slice::from_raw_parts(
                                tensor_view.data().as_ptr() as *const u16,
                                num_elements,
                            )
                        };
                        for (i, &b) in src.iter().enumerate() {
                            f32_data[i] = half::bf16::from_bits(b).to_f32();
                        }
                    }
                    safetensors::Dtype::F16 => {
                        let src = unsafe {
                            std::slice::from_raw_parts(
                                tensor_view.data().as_ptr() as *const u16,
                                num_elements,
                            )
                        };
                        for (i, &b) in src.iter().enumerate() {
                            f32_data[i] = half::f16::from_bits(b).to_f32();
                        }
                    }
                    _ => {
                        return Err(anyhow!("Unsupported dtype: {:?}", tensor_view.dtype()));
                    }
                }

                #[cfg(unix)]
                release_source_pages(tensor_view.data());

                let rows = shape.dims()[0];
                let cols = shape.dims()[1];
                let nb_k = cols / crate::core::quant::QK4_0;
                if !cols.is_multiple_of(crate::core::quant::QK4_0) {
                    let buffer = cpu_memory.alloc(num_elements * 4, DType::F32)?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(
                            f32_data.as_ptr(),
                            buffer.as_mut_ptr() as *mut f32,
                            num_elements,
                        );
                    }
                    return finalize_tensor(shape, buffer);
                }

                use crate::core::quant::{BlockQ4_0, QK4_0};
                use half::f16;

                let mut blocks = Vec::with_capacity(rows * nb_k);
                for j in 0..rows {
                    for bi in 0..nb_k {
                        let offset = j * cols + bi * QK4_0;
                        let src = &f32_data[offset..offset + QK4_0];
                        let mut block = BlockQ4_0 {
                            d: f16::from_f32(0.0),
                            qs: [0; 16],
                        };
                        let max_val = src.iter().map(|v| v.abs()).fold(0.0f32, |x, y| x.max(y));
                        let d = max_val / 7.0;
                        let id = if d == 0.0 { 0.0 } else { 1.0 / d };
                        block.d = f16::from_f32(d);
                        for z in 0..16 {
                            let v0 = (src[z] * id).round().clamp(-8.0, 7.0) as i8;
                            let v1 = (src[z + 16] * id).round().clamp(-8.0, 7.0) as i8;
                            block.qs[z] = (v0 + 8) as u8 | (((v1 + 8) as u8) << 4);
                        }
                        blocks.push(block);
                    }
                }

                let size_bytes = blocks.len() * std::mem::size_of::<BlockQ4_0>();
                let buffer = cpu_memory.alloc(size_bytes, DType::Q4_0)?;
                unsafe {
                    std::ptr::copy_nonoverlapping(
                        blocks.as_ptr(),
                        buffer.as_mut_ptr() as *mut BlockQ4_0,
                        blocks.len(),
                    );
                }
                return finalize_tensor(shape, buffer);
            }

            // F32 path (norms, embeddings): convert directly into Galloc buffer (no intermediate Vec)
            let buffer = cpu_memory.alloc(num_elements * 4, DType::F32)?;
            let dst = unsafe {
                std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut f32, num_elements)
            };

            match tensor_view.dtype() {
                safetensors::Dtype::F32 => unsafe {
                    std::ptr::copy_nonoverlapping(
                        tensor_view.data().as_ptr(),
                        buffer.as_mut_ptr(),
                        tensor_view.data().len(),
                    );
                },
                safetensors::Dtype::BF16 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            tensor_view.data().as_ptr() as *const u16,
                            num_elements,
                        )
                    };
                    for (i, &b) in src.iter().enumerate() {
                        dst[i] = half::bf16::from_bits(b).to_f32();
                    }
                }
                safetensors::Dtype::F16 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            tensor_view.data().as_ptr() as *const u16,
                            num_elements,
                        )
                    };
                    for (i, &b) in src.iter().enumerate() {
                        dst[i] = half::f16::from_bits(b).to_f32();
                    }
                }
                _ => {
                    return Err(anyhow!("Unsupported dtype: {:?}", tensor_view.dtype()));
                }
            }

            #[cfg(unix)]
            release_source_pages(tensor_view.data());

            finalize_tensor(shape, buffer)
        };

        // 3. Load Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let names = mapper.weight_names(i);
            let qkv_bias = if let Some(bias_names) = mapper.bias_names(i) {
                Some(crate::layers::transformer_layer::QkvBias {
                    bq: load_tensor(&bias_names.bq, false)?,
                    bk: load_tensor(&bias_names.bk, false)?,
                    bv: load_tensor(&bias_names.bv, false)?,
                })
            } else {
                None
            };
            let layer = TransformerLayer {
                wq: load_tensor(&names.wq, true)?,
                wk: load_tensor(&names.wk, true)?,
                wv: load_tensor(&names.wv, true)?,
                wo: load_tensor(&names.wo, true)?,
                w_gate: load_tensor(&names.w_gate, true)?,
                w_up: load_tensor(&names.w_up, true)?,
                w_down: load_tensor(&names.w_down, true)?,
                attention_norm: load_tensor(&names.attention_norm, false)?,
                ffn_norm: load_tensor(&names.ffn_norm, false)?,
                qkv_bias,
            };
            layers.push(layer);
        }

        // 4. Load Other Components
        //
        // embed_tokens is always kept on CPU (same strategy as llama.cpp) to save
        // GPU memory (~300MB for Llama 3.2 1B F16).  Gather is performed on CPU
        // then the resulting F32 embedding is written to the GPU x-buffer.
        //
        // Helper that loads into a CPU tensor regardless of the main backend.
        let load_tensor_cpu = |name: &str, is_weight: bool| -> Result<Tensor> {
            let shard_idx = weight_map.get(name).copied().unwrap_or(0);
            let tensor_view = match shard_tensors[shard_idx].tensor(name) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("Error finding tensor '{}' in shard {}", name, shard_idx);
                    return Err(anyhow!("{}", e));
                }
            };
            let shape = Shape::new(tensor_view.shape().to_vec());
            let num_elements: usize = tensor_view.shape().iter().product();
            let target_dtype = if is_weight { weight_dtype } else { DType::F32 };

            // F16 weight path: prefer MmapBuffer for zero-copy on CPU
            if target_dtype == DType::F16 {
                if tensor_view.dtype() == safetensors::Dtype::F16 {
                    use crate::buffer::mmap_buffer::MmapBuffer;
                    let data = tensor_view.data();
                    let data_offset =
                        data.as_ptr() as usize - shard_mmap_arcs[shard_idx].as_ptr() as usize;
                    let mmap_arc = shard_mmap_arcs[shard_idx].clone();
                    let buffer: Arc<dyn crate::core::buffer::Buffer> = Arc::new(unsafe {
                        MmapBuffer::new(mmap_arc, data_offset, data.len(), DType::F16)
                    });
                    return Ok(Tensor::new(shape, buffer, cpu_backend.clone()));
                }

                // Conversion into Galloc buffer
                use half::f16;
                let size_bytes = num_elements * 2;
                let buffer = cpu_memory.alloc(size_bytes, DType::F16)?;
                let dst = unsafe {
                    std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut f16, num_elements)
                };
                match tensor_view.dtype() {
                    safetensors::Dtype::F16 => {
                        let data = tensor_view.data();
                        unsafe {
                            std::ptr::copy_nonoverlapping(
                                data.as_ptr(),
                                buffer.as_mut_ptr(),
                                data.len(),
                            );
                        }
                    }
                    safetensors::Dtype::BF16 => {
                        let data = tensor_view.data();
                        let src = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const u16, num_elements)
                        };
                        for (i, &b) in src.iter().enumerate() {
                            dst[i] = f16::from_f32(half::bf16::from_bits(b).to_f32());
                        }
                    }
                    safetensors::Dtype::F32 => {
                        let data = tensor_view.data();
                        let src = unsafe {
                            std::slice::from_raw_parts(data.as_ptr() as *const f32, num_elements)
                        };
                        for (i, &v) in src.iter().enumerate() {
                            dst[i] = f16::from_f32(v);
                        }
                    }
                    _ => {
                        return Err(anyhow!(
                            "Unsupported dtype in safetensors: {:?}",
                            tensor_view.dtype()
                        ));
                    }
                }
                #[cfg(unix)]
                release_source_pages(tensor_view.data());
                return Ok(Tensor::new(shape, buffer, cpu_backend.clone()));
            }

            // F32 path
            let buffer = cpu_memory.alloc(num_elements * 4, DType::F32)?;
            let dst = unsafe {
                std::slice::from_raw_parts_mut(buffer.as_mut_ptr() as *mut f32, num_elements)
            };
            match tensor_view.dtype() {
                safetensors::Dtype::F32 => unsafe {
                    std::ptr::copy_nonoverlapping(
                        tensor_view.data().as_ptr(),
                        buffer.as_mut_ptr(),
                        tensor_view.data().len(),
                    );
                },
                safetensors::Dtype::BF16 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            tensor_view.data().as_ptr() as *const u16,
                            num_elements,
                        )
                    };
                    for (i, &b) in src.iter().enumerate() {
                        dst[i] = half::bf16::from_bits(b).to_f32();
                    }
                }
                safetensors::Dtype::F16 => {
                    let src = unsafe {
                        std::slice::from_raw_parts(
                            tensor_view.data().as_ptr() as *const u16,
                            num_elements,
                        )
                    };
                    for (i, &b) in src.iter().enumerate() {
                        dst[i] = half::f16::from_bits(b).to_f32();
                    }
                }
                _ => {
                    return Err(anyhow!("Unsupported dtype: {:?}", tensor_view.dtype()));
                }
            }
            #[cfg(unix)]
            release_source_pages(tensor_view.data());
            Ok(Tensor::new(shape, buffer, cpu_backend.clone()))
        };

        // embed_tokens: always CPU (saves GPU memory; gather is CPU-side)
        let embed_tokens = load_tensor_cpu(mapper.embed_name(), weight_dtype == DType::F16)?;

        let norm = load_tensor(mapper.norm_name(), false)?;

        // lm_head: loaded onto the main backend (GPU if applicable).
        // Tied weights path also goes to main backend so matmul_transposed runs on GPU.
        let lm_head_name = mapper.lm_head_name();
        let has_lm_head = if weight_map.is_empty() {
            shard_tensors[0].names().contains(&lm_head_name)
        } else {
            weight_map.contains_key(lm_head_name)
        };
        let lm_head = if has_lm_head {
            load_tensor(lm_head_name, true)?
        } else {
            // Tied weights: build lm_head from the CPU embed_tokens and upload to device.
            eprintln!(
                "lm_head not found, deriving from embed_tokens ({:?}) for lm_head...",
                weight_dtype
            );
            if is_cpu {
                // CPU: reuse embed_tokens buffer directly (zero-copy, same backend)
                embed_tokens.clone()
            } else {
                // GPU: upload CPU embed_tokens to device.
                // If weight_dtype != embed_tokens.dtype() we need a cast first.
                if embed_tokens.dtype() == weight_dtype || weight_dtype == DType::F32 {
                    // Same dtype: upload as-is
                    backend.copy_from(&embed_tokens)?
                } else {
                    // Different dtype (e.g. embed is F16, weight_dtype is Q4_0): not supported
                    // for tied weights, fall back to F16 upload
                    backend.copy_from(&embed_tokens)?
                }
            }
        };

        // Keep a cpu_backend reference only when the main backend is GPU
        let stored_cpu_backend = if is_cpu {
            None
        } else {
            Some(cpu_backend as Arc<dyn Backend>)
        };

        Ok(Self {
            config,
            layers,
            embed_tokens,
            norm,
            lm_head,
            cpu_backend: stored_cpu_backend,
            preload_pool: std::sync::Mutex::new(None),
        })
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

        // Iterate layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(LayerForwardArgs {
                x: &mut x,
                kv_cache: &mut kv_caches[i],
                start_pos,
                backend,
                memory,
                rms_norm_eps: self.config.rms_norm_eps as f32,
                rope_theta: self.config.rope_theta as f32,
                workspace: None,
                use_gpu_attn: true,
                need_scores: false,
                head_dim: self.config.head_dim,
                profiler: None,
                layer_id: i,
                skip_attn: false,
                skip_mlp: false,
            })?;
        }

        // Final Norm
        backend.rms_norm(&mut x, &self.norm, self.config.rms_norm_eps as f32)?;

        // Head
        let vocab_size = self.config.vocab_size;
        let logits_buf = memory.alloc(batch_size * seq_len * vocab_size * 4, DType::F32)?;
        let mut logits = Tensor::new(
            Shape::new(vec![batch_size, seq_len, vocab_size]),
            logits_buf,
            backend.clone(),
        );

        backend.matmul_transposed(&x, &self.lm_head, &mut logits)?;

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
        let use_gpu_attn = args.use_gpu_attn;
        let mut profiler = args.profiler;

        let mut score_accumulator = args.score_accumulator;
        let skip_config = args.skip_config;
        let mut importance_collector = args.importance_collector;

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

        // 2. Iterate layers
        for (i, layer) in self.layers.iter().enumerate() {
            let need_scores = score_accumulator
                .as_ref()
                .is_some_and(|acc| acc.should_track_layer(i));

            let (s_attn, s_mlp) =
                skip_config.map_or((false, false), |sc| (sc.skip_attn(i), sc.skip_mlp(i)));

            // Snapshot hidden state before layer for importance collection
            if let Some(ref mut coll) = importance_collector {
                let x_data = x.as_slice::<f32>();
                coll.snapshot_before(x_data, seq_len, hidden_size);
            }

            layer.forward(LayerForwardArgs {
                x: &mut x,
                kv_cache: &mut kv_caches[i],
                start_pos,
                backend,
                memory,
                rms_norm_eps: self.config.rms_norm_eps as f32,
                rope_theta: self.config.rope_theta as f32,
                workspace: workspace.as_deref_mut(),
                use_gpu_attn,
                need_scores,
                head_dim: self.config.head_dim,
                profiler: profiler.as_deref_mut(),
                layer_id: i,
                skip_attn: s_attn,
                skip_mlp: s_mlp,
            })?;

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

        // 3. Final Norm
        backend.rms_norm(&mut x, &self.norm, self.config.rms_norm_eps as f32)?;

        // 4. Head
        backend.matmul_transposed(&x, &self.lm_head, logits_out)?;

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

        if self.config.has_qkv_bias {
            return None; // Bias not yet supported in GPU plan
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

        // Collect per-layer buffer handles
        let mut layer_bufs = Vec::new();
        let mut kv_bufs_vec = Vec::new();
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
            });
            kv_bufs_vec.push(KvBufs {
                k_cache: cl!(kv_caches[i].k_buffer),
                v_cache: cl!(kv_caches[i].v_buffer),
            });
        }

        let ocl_backend = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()?;

        let full_config = FullPlanConfig {
            f16_program: &ocl_backend.f16_program,
            simple_ops_program: &ocl_backend.simple_ops_program,
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
            lm_head_buf: cl!(self.lm_head),
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
        };

        match build_full_plan(&full_config) {
            Ok(plan) => {
                log::info!(
                    "GPU kernel plan built ({} layers, capacity={})",
                    self.layers.len(),
                    capacity
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
        _logits_out: &mut Tensor,
        backend: &Arc<dyn Backend>,
    ) -> Result<bool> {
        // 1. Embedding lookup: CPU gather + upload to GPU x-buffer
        self.gather_embed(input_tokens, x_gen, backend)?;

        // 2. Execute plan (all layers + final norm + lm_head)
        let ocl_backend = backend
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
            .ok_or_else(|| anyhow!("Backend is not OpenCL"))?;

        match plan.execute(ocl_backend.queue.as_core(), start_pos, kv_caches) {
            Ok(()) => Ok(true),
            Err(_) => Ok(false), // plan invalidated, caller should rebuild
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
        let use_gpu_attn = args.use_gpu_attn;
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
            self.layers[i].forward(LayerForwardArgs {
                x: &mut x,
                kv_cache: current,
                start_pos,
                backend,
                memory,
                rms_norm_eps: self.config.rms_norm_eps as f32,
                rope_theta: self.config.rope_theta as f32,
                workspace: workspace.as_deref_mut(),
                use_gpu_attn,
                need_scores: false,
                head_dim: self.config.head_dim,
                profiler: None,
                layer_id: i,
                skip_attn: false,
                skip_mlp: false,
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
        backend.rms_norm(&mut x, &self.norm, self.config.rms_norm_eps as f32)?;
        backend.matmul_transposed(&x, &self.lm_head, logits_out)?;

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
        if let Some(ref cpu_be) = self.cpu_backend {
            // GPU path: indices may be on GPU — read them to a CPU buffer first
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

    /// Parse `model.safetensors.index.json` and return (shard_filenames, tensor→shard_idx map).
    fn parse_shard_index(index_path: &Path) -> Result<(Vec<String>, HashMap<String, usize>)> {
        #[derive(Deserialize)]
        struct ShardIndex {
            weight_map: HashMap<String, String>,
        }
        let index: ShardIndex = serde_json::from_reader(File::open(index_path)?)?;

        let mut shard_files: Vec<String> = Vec::new();
        let mut file_to_idx: HashMap<String, usize> = HashMap::new();
        let mut weight_map: HashMap<String, usize> = HashMap::new();

        for (tensor_name, shard_file) in &index.weight_map {
            let idx = if let Some(&i) = file_to_idx.get(shard_file) {
                i
            } else {
                let i = shard_files.len();
                shard_files.push(shard_file.clone());
                file_to_idx.insert(shard_file.clone(), i);
                i
            };
            weight_map.insert(tensor_name.clone(), idx);
        }

        eprintln!(
            "Shard index: {} tensors across {} shards",
            weight_map.len(),
            shard_files.len()
        );

        Ok((shard_files, weight_map))
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
        };

        let model = TransformerModel {
            config,
            layers: vec![],
            embed_tokens,
            norm,
            lm_head,
            cpu_backend: None, // CPU-only model
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
}
