//! `BatchRunCtx` — `bin/generate.rs::main()`에서 batch 모드 진입 시점에
//! 살아있는 모든 outer-scope state를 packaging하는 struct.
//!
//! 외과적 lift-and-shift라 구조화·추상화는 최소화한다. CLI args 일부와
//! 객체 ownership을 통째로 받아 `run_prompt_batch`가 종료될 때 process도
//! 종료된다 (caller가 `return Ok(())` 패턴이라 reuse 없음).

use std::sync::Arc;

use tokenizers::Tokenizer;

use crate::backend::Backend;
use crate::inference::attention_scores::AttentionScoreAccumulator;
use crate::inference::skip_config::SkipConfig;
use crate::memory::Memory;
use crate::models::transformer::TransformerModel;
use crate::pressure::cache_manager::CacheManager;
use crate::pressure::kv_cache::KVCache;
use crate::resilience::CommandExecutor;
use crate::session::cli::Args;

pub struct BatchRunCtx {
    // ─── CLI args 전체 (편의상 통째로 보유) ────────────────────────────────
    pub args: Args,

    // ─── 백엔드 / 메모리 / 모델 ──────────────────────────────────────────
    pub backend: Arc<dyn Backend>,
    pub memory: Arc<dyn Memory>,
    pub cpu_backend_arc: Arc<dyn Backend>,
    pub cpu_memory_arc: Arc<dyn Memory>,
    pub gpu_backend_arc: Option<Arc<dyn Backend>>,
    pub gpu_memory_arc: Option<Arc<dyn Memory>>,
    pub model: TransformerModel,
    pub tokenizer: Tokenizer,

    // ─── KV / 상태 객체 (owned) ───────────────────────────────────────────
    pub kv_caches: Vec<KVCache>,
    pub cache_manager: CacheManager,
    pub score_accumulator: Option<AttentionScoreAccumulator>,
    pub command_executor: Option<CommandExecutor>,
    pub skip_config: Option<SkipConfig>,

    // ─── 파생 상태 (main()에서 계산되어 batch 진입 시점에 살아있음) ───────
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub max_seq_len: usize,
    pub is_gpu: bool,
    pub weights_on_gpu: bool,
    pub kv_heads: usize,
    pub head_dim: usize,
    pub kv_type: crate::buffer::DType,
    pub actual_protected_prefix: usize,
    pub score_based_eviction: bool,
    pub throttle_delay_ms: u64,
    pub last_skip_ratio: Option<f32>,
    pub sampling_config: crate::inference::sampling::SamplingConfig,
}
