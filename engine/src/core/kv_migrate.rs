//! KV cache migration between backends (CPU↔GPU).

use std::sync::Arc;

use anyhow::Result;

use crate::core::backend::Backend;
use crate::core::buffer::DType;
use crate::core::kv_cache::KVCache;
use crate::core::memory::Memory;
use crate::core::tensor::Tensor;

/// Migrate KV caches from one backend to another.
///
/// Reads KV data from `src_backend`, creates intermediate CPU tensors via
/// `cpu_backend`/`cpu_memory`, then copies to `dst_backend` if `copy_to_dst`
/// is true. When migrating GPU→CPU, `copy_to_dst=false` keeps data on CPU.
///
/// # Arguments
/// * `kv_caches` - Mutable KV caches to migrate in-place
/// * `src_backend` - Backend to read current KV data from
/// * `dst_backend` - Backend to write migrated KV data to
/// * `cpu_backend` - CPU backend for intermediate tensors
/// * `cpu_memory` - CPU memory allocator for intermediate buffers
/// * `dst_memory` - Memory allocator for the destination backend
/// * `kv_heads` - Number of KV attention heads
/// * `head_dim` - Dimension per attention head
/// * `max_seq_len` - Maximum sequence length (for new KVCache allocation)
/// * `copy_to_dst` - If true, copy CPU tensors to dst_backend; if false, keep on CPU
#[allow(clippy::too_many_arguments)]
pub fn migrate_kv_caches(
    kv_caches: &mut [KVCache],
    src_backend: &Arc<dyn Backend>,
    dst_backend: &Arc<dyn Backend>,
    cpu_backend: &Arc<dyn Backend>,
    cpu_memory: &Arc<dyn Memory>,
    dst_memory: &Arc<dyn Memory>,
    kv_heads: usize,
    head_dim: usize,
    max_seq_len: usize,
    copy_to_dst: bool,
) -> Result<()> {
    for kv in kv_caches.iter_mut() {
        let current_capacity = kv.capacity();
        let saved_pos = kv.current_pos;
        let k_size = current_capacity * kv_heads * head_dim * 4;

        let mut k_data = vec![0u8; k_size];
        let mut v_data = vec![0u8; k_size];
        src_backend.read_buffer(&kv.k_buffer, &mut k_data)?;
        src_backend.read_buffer(&kv.v_buffer, &mut v_data)?;

        let k_cpu_buf = cpu_memory.alloc(k_size, DType::F32)?;
        unsafe {
            std::ptr::copy_nonoverlapping(k_data.as_ptr(), k_cpu_buf.as_mut_ptr(), k_size);
        }
        let k_cpu_tensor = Tensor::new(kv.k_buffer.shape().clone(), k_cpu_buf, cpu_backend.clone());

        let v_cpu_buf = cpu_memory.alloc(k_size, DType::F32)?;
        unsafe {
            std::ptr::copy_nonoverlapping(v_data.as_ptr(), v_cpu_buf.as_mut_ptr(), k_size);
        }
        let v_cpu_tensor = Tensor::new(kv.v_buffer.shape().clone(), v_cpu_buf, cpu_backend.clone());

        let (k_final, v_final) = if copy_to_dst {
            (
                dst_backend.copy_from(&k_cpu_tensor)?,
                dst_backend.copy_from(&v_cpu_tensor)?,
            )
        } else {
            (k_cpu_tensor, v_cpu_tensor)
        };

        let saved_layout = kv.layout();
        let mut new_kv = KVCache::new_dynamic(
            k_final,
            v_final,
            current_capacity,
            max_seq_len,
            kv_heads,
            head_dim,
            dst_memory.clone(),
        )
        .with_layout(saved_layout);
        new_kv.current_pos = saved_pos;
        *kv = new_kv;
    }
    Ok(())
}
