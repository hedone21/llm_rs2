//! Phase 4-A: prompt-batch 모드에서 사용하는 helper fn local copy.
//!
//! `bin/generate.rs`의 동명 fn들과 동일 본문. bin 쪽은 다른 분기(standard
//! generate / ppl / eval-ll)에서도 사용 중이라 그대로 둔다. 본 sprint는
//! batch 분기만 격리하며, DRY 통합은 후속 sprint (`session/standard/` 추출
//! 시점)에 backlog로 진행한다.

use std::sync::Arc;
use std::time::{SystemTime, UNIX_EPOCH};

use crate::backend::Backend;
use crate::buffer::DType;
use crate::memory::Memory;

/// JSONL `--prompt-batch` 파일의 단일 entry.
#[derive(serde::Deserialize)]
pub struct PromptBatchEntry {
    pub id: String,
    pub prompt: Option<String>,
    pub prompt_file: Option<String>,
}

/// `--prompt-batch <path>`로 지정된 JSONL 파일을 line 단위로 파싱.
pub fn load_prompt_batch(path: &str) -> anyhow::Result<Vec<PromptBatchEntry>> {
    use std::io::BufRead;
    let file = std::fs::File::open(path)
        .map_err(|e| anyhow::anyhow!("Failed to open prompt batch {}: {}", path, e))?;
    let reader = std::io::BufReader::new(file);
    let mut entries = Vec::new();
    for (i, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let entry: PromptBatchEntry =
            serde_json::from_str(trimmed).map_err(|e| anyhow::anyhow!("Line {}: {}", i + 1, e))?;
        entries.push(entry);
    }
    Ok(entries)
}

/// Entry의 `prompt` 또는 `prompt_file`을 읽어 단일 String으로 반환.
pub fn resolve_prompt(entry: &PromptBatchEntry) -> anyhow::Result<String> {
    if let Some(ref text) = entry.prompt {
        Ok(text.clone())
    } else if let Some(ref path) = entry.prompt_file {
        std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("Failed to read prompt_file {}: {}", path, e))
    } else {
        anyhow::bail!("Entry '{}': needs 'prompt' or 'prompt_file'", entry.id)
    }
}

/// Unix epoch 기준 floating-point seconds (resilience 로깅용).
pub fn unix_ts() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_secs_f64()
}

/// Tensor partition workspace 용 GPU buffer allocator.
///
/// OpenCL: `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR) + permanent map. dual
/// CPU/GPU access. Adreno PSS double-count 회피.
/// 그 외 backend: `memory.alloc()` fallback (CPU / CUDA pinned).
pub fn make_partition_gpu_alloc<'a>(
    backend: &'a dyn Backend,
    memory: &'a dyn Memory,
) -> impl Fn(usize, DType) -> anyhow::Result<Arc<dyn crate::buffer::Buffer>> + 'a {
    #[cfg(feature = "opencl")]
    let ocl_queue: Option<ocl::Queue> = backend
        .as_any()
        .downcast_ref::<crate::backend::opencl::OpenCLBackend>()
        .map(|b| b.queue.clone());

    #[cfg(not(feature = "opencl"))]
    let _ = backend;

    move |size: usize, dtype: DType| -> anyhow::Result<Arc<dyn crate::buffer::Buffer>> {
        #[cfg(feature = "opencl")]
        if let Some(ref q) = ocl_queue {
            let buf = crate::memory::opencl::unified::UnifiedBuffer::new(q.clone(), size, dtype)?;
            buf.map()?;
            return Ok(Arc::new(buf));
        }
        memory.alloc(size, dtype)
    }
}
