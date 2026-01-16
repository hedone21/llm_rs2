use anyhow::{Result, anyhow};
use serde::Deserialize;
use std::fs::File;
use std::path::Path;
use memmap2::MmapOptions;
use safetensors::SafeTensors;
use std::sync::Arc;

use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use crate::core::shape::Shape;
use crate::core::buffer::DType;
use crate::core::memory::Memory;
use crate::core::kv_cache::KVCache;
use crate::layers::llama_layer::LlamaLayer;
use crate::layers::workspace::LayerWorkspace;
use crate::memory::galloc::Galloc;

#[derive(Debug, Deserialize)]
pub struct LlamaConfig {
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub intermediate_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f64,
    pub rope_theta: f64,
}

pub struct LlamaModel {
    pub config: LlamaConfig,
    pub layers: Vec<LlamaLayer>,
    pub embed_tokens: Tensor,
    pub norm: Tensor,
    pub lm_head: Tensor,
}

impl LlamaModel {
     pub fn load(
        model_path: &str, 
        backend: Arc<dyn Backend>, 
        memory: &dyn Memory
    ) -> Result<Self> {
        let path = Path::new(model_path);
        
        // 1. Load Config
        let config_file = File::open(path.join("config.json"))?;
        let config: LlamaConfig = serde_json::from_reader(config_file)?;

        // 2. Open Safetensors
        let st_file = File::open(path.join("model.safetensors"))?;
        let mmap = unsafe { MmapOptions::new().map(&st_file)? };
        let tensors = SafeTensors::deserialize(&mmap)?;

        // Use CPU memory for loading
        let cpu_memory = Galloc::new();
        let cpu_backend = Arc::new(crate::backend::cpu::CpuBackend::new()); 


        // Helper to load a specific tensor by name
        let load_tensor = |name: &str, quantize: bool| -> Result<Tensor> {
           let tensor_view = match tensors.tensor(name) {
                Ok(v) => v,
                Err(e) => {
                    println!("Error finding tensor: {}", name);
                    return Err(e.into());
                }
           };
           let shape = Shape::new(tensor_view.shape().to_vec());
           let num_elements: usize = tensor_view.shape().iter().product();
           
           // Using captured cpu_memory and cpu_backend
           
           // Read F32 data first
           let mut f32_data = vec![0.0f32; num_elements];
           
           match tensor_view.dtype() {
               safetensors::Dtype::F32 => {
                   unsafe {
                       std::ptr::copy_nonoverlapping(
                           tensor_view.data().as_ptr(),
                           f32_data.as_mut_ptr() as *mut u8,
                           tensor_view.data().len()
                       );
                   }
               },
               safetensors::Dtype::BF16 => {
                   let data = tensor_view.data();
                   let u16_data = unsafe {
                       std::slice::from_raw_parts(data.as_ptr() as *const u16, num_elements)
                   };
                   for (i, &b) in u16_data.iter().enumerate() {
                       f32_data[i] = half::bf16::from_bits(b).to_f32();
                   }
               },
               safetensors::Dtype::F16 => {
                   let data = tensor_view.data();
                   let u16_data = unsafe {
                       std::slice::from_raw_parts(data.as_ptr() as *const u16, num_elements)
                   };
                   for (i, &b) in u16_data.iter().enumerate() {
                       f32_data[i] = half::f16::from_bits(b).to_f32();
                   }
               },
               _ => return Err(anyhow!("Unsupported dtype in safetensors: {:?}", tensor_view.dtype())),
           }

           if quantize {
               // Quantize to Q4_0
               let rows = shape.dims()[0];
               let cols = shape.dims()[1];
               
               let nb_k = cols / crate::core::quant::QK4_0;
               if cols % crate::core::quant::QK4_0 != 0 {
                    // Fallback to F32
                    let buffer = cpu_memory.alloc(num_elements * 4, DType::F32)?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(f32_data.as_ptr(), buffer.as_mut_ptr() as *mut f32, num_elements);
                    }
                    let cpu_tensor = Tensor::new(shape, buffer, cpu_backend.clone());
                    return backend.copy_from(&cpu_tensor);
               }

               use crate::core::quant::{BlockQ4_0, QK4_0};
               use half::f16;
               
               let mut blocks = Vec::with_capacity(rows * nb_k);
               for j in 0..rows {
                   for bi in 0..nb_k {
                       let offset = j * cols + bi * QK4_0;
                       let src = &f32_data[offset..offset+QK4_0];
                       let mut block = BlockQ4_0 { d: f16::from_f32(0.0), qs: [0; 16] };
                       
                       let max_val = src.iter().map(|v| v.abs()).fold(0.0f32, |x, y| x.max(y));
                       let d = max_val / 7.0;
                       let id = if d == 0.0 { 0.0 } else { 1.0 / d };
                       
                       block.d = f16::from_f32(d);
                       for z in 0..16 {
                           let v0 = (src[z] * id).round().clamp(-8.0, 7.0) as i8;
                           let v1 = (src[z + 16] * id).round().clamp(-8.0, 7.0) as i8;
                           let b0 = (v0 + 8) as u8;
                           let b1 = (v1 + 8) as u8;
                           block.qs[z] = b0 | (b1 << 4);
                       }
                       blocks.push(block);
                   }
               }
               
               let size_bytes = blocks.len() * std::mem::size_of::<BlockQ4_0>();
               let buffer = cpu_memory.alloc(size_bytes, DType::Q4_0)?;
               unsafe {
                    std::ptr::copy_nonoverlapping(blocks.as_ptr(), buffer.as_mut_ptr() as *mut BlockQ4_0, blocks.len());
               }
               let cpu_tensor = Tensor::new(shape, buffer, cpu_backend.clone());
               backend.copy_from(&cpu_tensor)

           } else {
               let buffer = cpu_memory.alloc(num_elements * 4, DType::F32)?;
               unsafe {
                   std::ptr::copy_nonoverlapping(f32_data.as_ptr(), buffer.as_mut_ptr() as *mut f32, num_elements);
               }
               let cpu_tensor = Tensor::new(shape, buffer, cpu_backend.clone());
               backend.copy_from(&cpu_tensor)
           }
        };

        // 3. Load Layers
        let mut layers = Vec::with_capacity(config.num_hidden_layers);
        for i in 0..config.num_hidden_layers {
            let layer = LlamaLayer {
                wq: load_tensor(&format!("model.layers.{}.self_attn.q_proj.weight", i), true)?,
                wk: load_tensor(&format!("model.layers.{}.self_attn.k_proj.weight", i), true)?,
                wv: load_tensor(&format!("model.layers.{}.self_attn.v_proj.weight", i), true)?,
                wo: load_tensor(&format!("model.layers.{}.self_attn.o_proj.weight", i), true)?,
                w_gate: load_tensor(&format!("model.layers.{}.mlp.gate_proj.weight", i), true)?,
                w_up: load_tensor(&format!("model.layers.{}.mlp.up_proj.weight", i), true)?,
                w_down: load_tensor(&format!("model.layers.{}.mlp.down_proj.weight", i), true)?,
                attention_norm: load_tensor(&format!("model.layers.{}.input_layernorm.weight", i), false)?,
                ffn_norm: load_tensor(&format!("model.layers.{}.post_attention_layernorm.weight", i), false)?,
            };
            layers.push(layer);
        }

        // 4. Load Other Components
        let embed_tokens = load_tensor("model.embed_tokens.weight", false)?;
        let norm = load_tensor("model.norm.weight", false)?;
        
        let lm_head_name = "lm_head.weight";
        let lm_head = if tensors.names().contains(&lm_head_name) {
             load_tensor(lm_head_name, true)? // Quantize head? Usually yes for FFN/Head
        } else {
             // Tied weights: embed_tokens is F32, quantize to Q4_0 for lm_head
             println!("lm_head not found, quantizing embed_tokens for lm_head...");
             load_tensor("model.embed_tokens.weight", true)?
        };

        Ok(Self {
            config,
            layers,
            embed_tokens,
            norm,
            lm_head,
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
            backend.clone()
        );
        
        // Use gather for embedding lookup
        // input_tokens should be on the same backend as embed_tokens
        backend.gather(&self.embed_tokens, input_tokens, &mut x)?;

        // Iterate layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(
                &mut x, 
                &mut kv_caches[i], 
                start_pos, 
                backend, 
                memory, 
                self.config.rms_norm_eps as f32, 
                self.config.rope_theta as f32,
                None, // No workspace for standard forward
                true  // Always use GPU attention in standard forward
            )?;
        }
        
        // Final Norm
        backend.rms_norm(&mut x, &self.norm, self.config.rms_norm_eps as f32)?;
        
        // Head
        let vocab_size = self.config.vocab_size;
        let logits_buf = memory.alloc(batch_size * seq_len * vocab_size * 4, DType::F32)?;
        let mut logits = Tensor::new(
            Shape::new(vec![batch_size, seq_len, vocab_size]),
            logits_buf,
            backend.clone()
        );
        
        backend.matmul_transposed(&x, &self.lm_head, &mut logits)?;
        
        Ok(logits)
    }
    
    /// Comprehensive forward pass that writes logits into a pre-allocated buffer.
    /// Optionally accepts x_gen and workspace for memory optimization during generation.
    pub fn forward_into(
        &self,
        input_tokens: &Tensor,
        start_pos: usize,
        kv_caches: &mut [KVCache],
        backend: &Arc<dyn Backend>,
        memory: &dyn Memory,
        logits_out: &mut Tensor,
        x_gen: Option<&mut Tensor>,
        mut workspace: Option<&mut LayerWorkspace>,
        use_gpu_attn: bool,
    ) -> Result<()> {

        let batch_size = input_tokens.shape().dims()[0];
        let seq_len = input_tokens.shape().dims()[1];
        let hidden_size = self.config.hidden_size;

        // 1. Embedding lookup
        // Use provided x_gen buffer if available and seq_len == 1
        let mut x = if seq_len == 1 {
            if let Some(xb) = x_gen {
                xb.clone()
            } else {
                let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
                Tensor::new(Shape::new(vec![batch_size, seq_len, hidden_size]), x_buf, backend.clone())
            }
        } else {
            let x_buf = memory.alloc(batch_size * seq_len * hidden_size * 4, DType::F32)?;
            Tensor::new(Shape::new(vec![batch_size, seq_len, hidden_size]), x_buf, backend.clone())
        };

        // Use gather
        backend.gather(&self.embed_tokens, input_tokens, &mut x)?;


        // 2. Iterate layers
        for (i, layer) in self.layers.iter().enumerate() {
            layer.forward(
                &mut x, 
                &mut kv_caches[i], 
                start_pos, 
                backend, 
                memory, 
                self.config.rms_norm_eps as f32, 
                self.config.rope_theta as f32,
                workspace.as_deref_mut(), // Pass workspace if provided (only for seq_len=1)
                use_gpu_attn
            )?;
        }
        
        // 3. Final Norm
        backend.rms_norm(&mut x, &self.norm, self.config.rms_norm_eps as f32)?;
        
        // 4. Head
        backend.matmul_transposed(&x, &self.lm_head, logits_out)?;

        
        Ok(())
    }
}
