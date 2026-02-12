use super::EvictionPolicy;
use crate::core::kv_cache::KVCache;
use anyhow::Result;

/// Default policy that never evicts. Maintains backward compatibility
/// with existing behavior — cache simply fills up and overflows produce an error.
pub struct NoEvictionPolicy;

impl NoEvictionPolicy {
    pub fn new() -> Self {
        Self
    }
}

impl EvictionPolicy for NoEvictionPolicy {
    fn should_evict(&self, _cache: &KVCache, _mem_available: usize) -> bool {
        false
    }

    fn evict(&self, _cache: &mut KVCache, _target_len: usize) -> Result<()> {
        Ok(())
    }

    fn name(&self) -> &str {
        "none"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::backend::cpu::CpuBackend;
    use crate::buffer::shared_buffer::SharedBuffer;
    use crate::core::buffer::DType;
    use crate::core::shape::Shape;
    use crate::core::tensor::Tensor;
    use std::sync::Arc;

    fn make_cache(pos: usize) -> KVCache {
        let backend = Arc::new(CpuBackend::new());
        let buf_size = 100 * 4 * 4;
        let k = Tensor::new(
            Shape::new(vec![1, 100, 1, 4]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let v = Tensor::new(
            Shape::new(vec![1, 100, 1, 4]),
            Arc::new(SharedBuffer::new(buf_size, DType::F32)),
            backend.clone(),
        );
        let mut cache = KVCache::new(k, v, 100);
        cache.current_pos = pos;
        cache
    }

    #[test]
    fn test_no_eviction_never_evicts() {
        let policy = NoEvictionPolicy::new();
        let cache = make_cache(99);
        assert!(!policy.should_evict(&cache, 0)); // Even with 0 memory available
        assert!(!policy.should_evict(&cache, usize::MAX));
    }

    #[test]
    fn test_no_eviction_evict_is_noop() {
        let policy = NoEvictionPolicy::new();
        let mut cache = make_cache(50);
        policy.evict(&mut cache, 30).unwrap();
        assert_eq!(cache.current_pos, 50); // Unchanged
    }

    #[test]
    fn test_no_eviction_name() {
        assert_eq!(NoEvictionPolicy::new().name(), "none");
    }
}
