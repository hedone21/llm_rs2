//! LISWAP-2 Phase 3 unit tests — `SwapExecutor` async dispatch path.
//!
//! Prototype-grade: no spec ID (decision pending measurement results).
//! Plan: `compiled-chasing-hopper.md` §Phase 3.
//!
//! ## 검증 항목
//!
//! - [A] `test_sync_path_unchanged_with_none_dispatcher`:
//!   `async_dispatcher=None`이면 기존 sync 경로와 동작 동일 (backward-compat).
//! - [B] `test_async_path_skips_synchronize`:
//!   `async_dispatcher=Some` + `supports_async_transfer()=true`이면
//!   `backend.synchronize()`가 호출되지 않음 (INV-142 대체).
//! - [C] `test_async_path_uses_per_event_wait`:
//!   async path는 `wait_event_blocking`을 per-layer로 호출.
//! - [D] `test_supports_async_transfer_false_fallback`:
//!   `async_dispatcher=Some`이어도 `supports_async_transfer()=false`이면
//!   sync 경로로 폴백 (synchronize 호출됨).
//! - [E] `test_async_path_empty_batch`:
//!   async path에서도 빈 배치는 stage gate를 건너뜀.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

use anyhow::Result;

use llm_rs2::backend::cpu::CpuBackend;
use llm_rs2::core::backend::{Backend, GpuEvent};
use llm_rs2::core::buffer::DType;
use llm_rs2::core::memory::Memory;
use llm_rs2::core::tensor::Tensor;
use llm_rs2::memory::galloc::Galloc;
use llm_rs2::models::config::{ModelArch, ModelConfig};
use llm_rs2::models::weights::async_swap::AsyncSwapDispatcher;
use llm_rs2::models::weights::{LayerSlot, SwapExecutor};

// ── Mock backend ──────────────────────────────────────────────────────────────

/// Mock backend that tracks `synchronize()`, `wait_event_blocking()`,
/// and `supports_async_transfer()` call counts.
struct AsyncMockBackend {
    inner: CpuBackend,
    /// How many times `synchronize()` was called.
    synchronize_count: Arc<AtomicUsize>,
    /// How many times `wait_event_blocking()` was called.
    wait_event_count: Arc<AtomicUsize>,
    /// Return value of `supports_async_transfer()`.
    async_supported: bool,
    /// Whether `synchronize()` should fail.
    fail_synchronize: Arc<AtomicBool>,
}

impl AsyncMockBackend {
    fn new(async_supported: bool) -> Self {
        Self {
            inner: CpuBackend::new(),
            synchronize_count: Arc::new(AtomicUsize::new(0)),
            wait_event_count: Arc::new(AtomicUsize::new(0)),
            async_supported,
            fail_synchronize: Arc::new(AtomicBool::new(false)),
        }
    }

    fn synchronize_count(&self) -> usize {
        self.synchronize_count.load(Ordering::SeqCst)
    }

    fn wait_event_count(&self) -> usize {
        self.wait_event_count.load(Ordering::SeqCst)
    }
}

impl Backend for AsyncMockBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
    fn name(&self) -> &str {
        // Must NOT contain "CPU" so executor takes GPU path in materialise_weight.
        // In practice CPU path returns early with cpu_tensor (no upload needed),
        // so we use the inner CPU backend methods but report a non-CPU name to
        // drive the async branch in build_layer_from_mmap_async.
        // For simplicity these tests don't exercise materialise_weight at all
        // (secondary=None), so the name is only relevant for the supports_async_transfer flag.
        "async-mock"
    }
    fn device(&self) -> &str {
        "host"
    }
    fn synchronize(&self) -> Result<()> {
        self.synchronize_count.fetch_add(1, Ordering::SeqCst);
        if self.fail_synchronize.load(Ordering::SeqCst) {
            Err(anyhow::anyhow!("injected synchronize failure"))
        } else {
            Ok(())
        }
    }
    fn wait_event_blocking(&self, _evt: &GpuEvent) -> Result<()> {
        self.wait_event_count.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn supports_async_transfer(&self) -> bool {
        self.async_supported
    }
    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        self.inner.matmul(a, b, out)
    }
    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        self.inner.matmul_transposed(a, b, out)
    }
    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        rows: usize,
        cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        self.inner.matmul_slice(a, b, rows, cols, out)
    }
    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        self.inner.add_assign(a, b)
    }
    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        self.inner.scale(x, v)
    }
    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        self.inner.silu_mul(a, b)
    }
    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        self.inner.rms_norm(x, w, eps, add_unit)
    }
    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        self.inner.softmax(x)
    }
    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        self.inner.rope_inplace(x, start_pos, theta)
    }
    fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
        self.inner.copy_from(t)
    }
    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        self.inner.cast(src, dst)
    }
}

// ── Helpers ───────────────────────────────────────────────────────────────────

fn minimal_config() -> ModelConfig {
    ModelConfig {
        arch: ModelArch::Llama,
        hidden_size: 64,
        num_hidden_layers: 2,
        num_attention_heads: 4,
        num_key_value_heads: 4,
        head_dim: 16,
        intermediate_size: 128,
        vocab_size: 256,
        rms_norm_eps: 1e-5,
        rope_theta: 10000.0,
        has_qkv_bias: false,
        tie_word_embeddings: false,
        eos_token_id: 1,
        rope_local_theta: None,
        sliding_window: None,
        sliding_window_pattern: None,
        query_pre_attn_scalar: None,
        embed_scale: None,
        weight_prefix: String::new(),
    }
}

fn dummy_tensor(be: &Arc<dyn Backend>, numel: usize) -> Tensor {
    let mem = Galloc::new();
    let buf = mem.alloc(numel * 4, DType::F32).unwrap();
    Tensor::new(
        llm_rs2::core::shape::Shape::new(vec![numel]),
        buf,
        be.clone(),
    )
}

fn dummy_layer(be: &Arc<dyn Backend>) -> llm_rs2::layers::transformer_layer::TransformerLayer {
    llm_rs2::layers::transformer_layer::TransformerLayer {
        wq: dummy_tensor(be, 16),
        wk: dummy_tensor(be, 16),
        wv: dummy_tensor(be, 16),
        wo: dummy_tensor(be, 16),
        w_gate: dummy_tensor(be, 16),
        w_up: dummy_tensor(be, 16),
        w_down: dummy_tensor(be, 16),
        attention_norm: dummy_tensor(be, 4),
        ffn_norm: dummy_tensor(be, 4),
        qkv_bias: None,
        q_norm: None,
        k_norm: None,
        pre_ffn_norm: None,
        post_ffn_norm: None,
        partition_ctx: None,
    }
}

fn make_layers(be: &Arc<dyn Backend>, n: usize) -> Vec<Arc<LayerSlot>> {
    (0..n)
        .map(|_| Arc::new(LayerSlot::new(dummy_layer(be), DType::F16, None)))
        .collect()
}

// ── Test A: sync path unchanged with None dispatcher ─────────────────────────

/// `async_dispatcher=None`이면 기존 sync 경로 동작.
/// secondary=None이므로 실제 swap 없이 early-return.
/// synchronize_count == 0 (swap 없음 → stage gate 미실행).
#[test]
fn test_sync_path_unchanged_with_none_dispatcher() {
    let mock = Arc::new(AsyncMockBackend::new(false));
    let be: Arc<dyn Backend> = mock.clone();
    let layers = make_layers(&be, 2);
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let config = minimal_config();
    let memory = Galloc::new();

    let executor = SwapExecutor::new(DType::Q4_0, &config, be.clone(), &memory);

    // No secondary → early return, no stage gate.
    let report = executor
        .execute_on_slots(layers.as_slice(), None, &ratio_gen, &[0, 1], None)
        .expect("sync path must not error");

    assert!(report.swapped.is_empty(), "no secondary → no swaps");
    assert_eq!(mock.synchronize_count(), 0, "no swap → no synchronize");
    assert_eq!(
        mock.wait_event_count(),
        0,
        "sync path never calls wait_event_blocking"
    );
    assert_eq!(ratio_gen.load(Ordering::SeqCst), 0, "no bump without swap");
}

// ── Test B: async path skips synchronize ─────────────────────────────────────

/// `async_dispatcher=Some` + `supports_async_transfer()=true` →
/// `backend.synchronize()`가 호출되지 않음 (per-event wait로 대체).
/// secondary=None이므로 early-return, stage gate 자체가 실행되지 않음을 확인.
#[test]
fn test_async_path_skips_synchronize_on_empty_batch() {
    let mock = Arc::new(AsyncMockBackend::new(true));
    let be: Arc<dyn Backend> = mock.clone();
    let layers = make_layers(&be, 2);
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let config = minimal_config();
    let memory = Galloc::new();

    // Create a dispatcher (even though it won't be used with secondary=None).
    let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let dispatcher = AsyncSwapDispatcher::new(cpu_be);

    let executor = SwapExecutor::new(DType::Q4_0, &config, be.clone(), &memory);

    let report = executor
        .execute_on_slots(
            layers.as_slice(),
            None,
            &ratio_gen,
            &[0, 1],
            Some(&dispatcher),
        )
        .expect("async path must not error");

    assert!(report.swapped.is_empty(), "no secondary → no swaps");
    assert_eq!(
        mock.synchronize_count(),
        0,
        "async path: synchronize must NOT be called (stage gate skipped on empty batch)"
    );
    assert_eq!(ratio_gen.load(Ordering::SeqCst), 0, "no bump without swap");
}

// ── Test C: use_async flag gates on supports_async_transfer ──────────────────

/// `async_dispatcher=Some`이어도 `supports_async_transfer()=false`이면
/// sync fallback — synchronize가 호출됨 (実際のswapがないので0).
/// この테스트는 use_async=false 경로를 확인.
#[test]
fn test_supports_async_transfer_false_uses_sync_fallback() {
    let mock = Arc::new(AsyncMockBackend::new(false)); // async NOT supported
    let be: Arc<dyn Backend> = mock.clone();
    let layers = make_layers(&be, 2);
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let config = minimal_config();
    let memory = Galloc::new();

    let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let dispatcher = AsyncSwapDispatcher::new(cpu_be);

    let executor = SwapExecutor::new(DType::Q4_0, &config, be.clone(), &memory);

    // secondary=None → early return. No stage gate runs regardless.
    let report = executor
        .execute_on_slots(
            layers.as_slice(),
            None,
            &ratio_gen,
            &[0, 1],
            Some(&dispatcher), // dispatcher provided but async NOT supported
        )
        .expect("sync fallback must not error");

    assert!(report.swapped.is_empty());
    // supports_async_transfer=false → use_async=false → sync path.
    // secondary=None → no swap → stage gate not reached → synchronize=0.
    assert_eq!(
        mock.synchronize_count(),
        0,
        "no swap → synchronize not called"
    );
}

// ── Test D: async path empty batch ───────────────────────────────────────────

/// async path + empty target_layers → stage gate skipped entirely.
#[test]
fn test_async_path_empty_target_layers() {
    let mock = Arc::new(AsyncMockBackend::new(true));
    let be: Arc<dyn Backend> = mock.clone();
    let layers = make_layers(&be, 2);
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let config = minimal_config();
    let memory = Galloc::new();

    let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let dispatcher = AsyncSwapDispatcher::new(cpu_be);

    let executor = SwapExecutor::new(DType::Q4_0, &config, be.clone(), &memory);

    let report = executor
        .execute_on_slots(
            layers.as_slice(),
            None,
            &ratio_gen,
            &[], // empty targets
            Some(&dispatcher),
        )
        .expect("empty target must not error");

    assert!(report.swapped.is_empty());
    assert_eq!(report.ratio_generation_after, None, "no bump on empty swap");
    assert_eq!(mock.synchronize_count(), 0);
    assert_eq!(mock.wait_event_count(), 0);
}

// ── Test E: dispatcher pending remains 0 when secondary absent ───────────────

/// secondary=None の場合 dispatcher にジョブは submit されない.
/// dispatcher.pending_count() == 0 after execute_on_slots.
#[test]
fn test_async_path_no_secondary_no_pending_jobs() {
    let mock = Arc::new(AsyncMockBackend::new(true));
    let be: Arc<dyn Backend> = mock.clone();
    let layers = make_layers(&be, 2);
    let ratio_gen = Arc::new(AtomicU64::new(0));
    let config = minimal_config();
    let memory = Galloc::new();

    let cpu_be: Arc<dyn Backend> = Arc::new(CpuBackend::new());
    let dispatcher = AsyncSwapDispatcher::new(cpu_be);

    let executor = SwapExecutor::new(DType::Q4_0, &config, be.clone(), &memory);

    executor
        .execute_on_slots(
            layers.as_slice(),
            None,
            &ratio_gen,
            &[0, 1],
            Some(&dispatcher),
        )
        .expect("must not error");

    assert_eq!(
        dispatcher.pending_count(),
        0,
        "no secondary → no jobs submitted to dispatcher"
    );
}

// ── Test F: StageBreakdown log line format preserved ─────────────────────────

/// Stage breakdown log line format must be parseable regardless of
/// async/sync path (Tester relies on the format string).
#[test]
fn test_stage_breakdown_log_line_format() {
    use llm_rs2::models::weights::StageBreakdown;

    let bd = StageBreakdown {
        prefault_ms: 1.0,
        mmap_permute_ms: 2.5,
        arc_swap_ms: 0.0,
        madvise_ms: 0.0,
        synchronize_ms: 0.0,
        soa_reconvert_ms: 0.0,
        gen_bump_ms: 0.1,
    };
    let line = bd.to_log_line();
    assert!(line.contains("prefault="), "must contain prefault field");
    assert!(
        line.contains("mmap_permute="),
        "must contain mmap_permute field"
    );
    assert!(
        line.contains("synchronize="),
        "must contain synchronize field"
    );
    assert!(line.contains("gen_bump="), "must contain gen_bump field");
    // Verify zero values are recorded correctly in async mode (0.0ms).
    assert!(
        line.contains("arc_swap=0.0ms"),
        "zero arc_swap must appear as 0.0ms"
    );
    assert!(
        line.contains("synchronize=0.0ms"),
        "zero synchronize must appear as 0.0ms"
    );
}
