use anyhow::Result;
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
               // Assumption: shape is [Out, In] or [In, Out]?
               // Llama linear weights are [Out, In].
               // Row-wise quantization usually.
               // K = In.
               // We need N (rows) and K (cols).
               // Shape is [N, K].
               let rows = shape.dims()[0];
               let cols = shape.dims()[1];
               
               // Reuse the helper logic:
               let nb_k = cols / crate::core::quant::QK4_0;
               // Ensure divisibility?
               if cols % crate::core::quant::QK4_0 != 0 {
                    // Fallback to F32 if not compatible
                    let buffer = memory.alloc(num_elements * 4, DType::F32)?;
                    unsafe {
                        std::ptr::copy_nonoverlapping(f32_data.as_ptr(), buffer.as_mut_ptr() as *mut f32, num_elements);
                    }
                    return Ok(Tensor::new(shape, buffer, backend.clone()));
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
               let buffer = memory.alloc(size_bytes, DType::Q4_0)?;
               unsafe {
                    std::ptr::copy_nonoverlapping(blocks.as_ptr(), buffer.as_mut_ptr() as *mut BlockQ4_0, blocks.len());
               }
               Ok(Tensor::new(shape, buffer, backend.clone()))

           } else {
               let buffer = memory.alloc(num_elements * 4, DType::F32)?;
               unsafe {
                   std::ptr::copy_nonoverlapping(f32_data.as_ptr(), buffer.as_mut_ptr() as *mut f32, num_elements);
               }
               Ok(Tensor::new(shape, buffer, backend.clone()))
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
             let f32_data = embed_tokens.as_slice::<f32>();
             let shape = embed_tokens.shape().clone();
             let rows = shape.dims()[0]; // vocab_size
             let cols = shape.dims()[1]; // hidden_size
             
             use crate::core::quant::{BlockQ4_0, QK4_0};
             use half::f16;
             
             let nb_k = cols / QK4_0;
             if cols % QK4_0 != 0 {
                 // Fallback to F32 if not compatible
                 embed_tokens.clone()
             } else {
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
                 let buffer = memory.alloc(size_bytes, DType::Q4_0)?;
                 unsafe {
                     std::ptr::copy_nonoverlapping(blocks.as_ptr(), buffer.as_mut_ptr() as *mut BlockQ4_0, blocks.len());
                 }
                 Tensor::new(shape, buffer, backend.clone())
             }
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
        
        let tokens = input_tokens.as_slice::<u32>(); // Assuming U32/I32 indices
        let embed_data = self.embed_tokens.as_slice::<f32>();
        let x_data = x.as_mut_slice::<f32>();
        
        // Parallel copy?
        // For each token, copy hidden_size vector.
        for (i, &token_id) in tokens.iter().enumerate() {
            let offset = token_id as usize * hidden_size;
            let target_offset = i * hidden_size;
            x_data[target_offset..target_offset+hidden_size].copy_from_slice(&embed_data[offset..offset+hidden_size]);
        }

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
                None // No workspace for standard forward
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

        let tokens = input_tokens.as_slice::<u32>();
        let embed_data = self.embed_tokens.as_slice::<f32>();
        let x_data = x.as_mut_slice::<f32>();
        
        for (i, &token_id) in tokens.iter().enumerate() {
            let offset = token_id as usize * hidden_size;
            let target_offset = i * hidden_size;
            x_data[target_offset..target_offset+hidden_size].copy_from_slice(&embed_data[offset..offset+hidden_size]);
        }

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
                workspace.as_deref_mut() // Pass workspace if provided (only for seq_len=1)
            )?;
        }
        
        // 3. Final Norm
        backend.rms_norm(&mut x, &self.norm, self.config.rms_norm_eps as f32)?;
        
        // 4. Head
        backend.matmul_transposed(&x, &self.lm_head, logits_out)?;
        
        Ok(())
    }
}
