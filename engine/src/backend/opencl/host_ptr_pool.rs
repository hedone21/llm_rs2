//! `HostPtrPool` — Direction A `CL_MEM_ALLOC_HOST_PTR` slot pool for swap.
//!
//! LISWAP-3 prototype (plan: `compiled-chasing-hopper`, Direction A track,
//! Stage 3). Pre-allocates `n_slots` zero-copy `cl_mem` buffers up front so
//! that the per-layer swap path can reuse them across batches without paying
//! `clCreateBuffer` cost on the hot path. Each slot is `max_tensor_size`
//! bytes so any per-layer Q4_0 weight tensor fits.
//!
//! # Stage history
//!
//! - **Stage 1b** (`bin/stage1_host_ptr_microbench`): Path 4 single-tensor
//!   benchmark on Galaxy S25 — Adreno A830 reported -42.4% wall-clock vs
//!   the staging baseline + byte-equal PASS 5/5.
//! - **Stage 2** (`bin/stage2_pool_stability`): multi-tensor / multi-layer
//!   reuse stress test. 14-slot pool (~2 layers worth) became the sweet
//!   spot; >27 slots regressed by 6-9 ms/round on driver bookkeeping.
//! - **Stage 3** (this module): production swap integration behind
//!   `LLMRS_OPENCL_HOST_PTR_POOL` env-gate (default OFF) +
//!   `--swap-zero-copy` CLI flag.
//!
//! # Lifecycle
//!
//! 1. `HostPtrPool::new(backend, config)` allocates `n_slots` buffers.
//! 2. Caller acquires a slot via `acquire(size)` — returns `None` when all
//!    slots are in use OR when the requested `size > max_tensor_size`.
//!    Caller is expected to fall back to the staging path in either case.
//! 3. `HostPtrPoolGuard` holds the slot until dropped. The guard exposes
//!    `cl_mem()` for handing to the backend's fill helper, plus a `slot_idx`
//!    for diagnostics.
//! 4. Slot is returned to the pool via `Drop`.
//!
//! # Concurrency
//!
//! `HostPtrPool` is `Send + Sync`. Slots use `AtomicBool` for in-use
//! tracking so a `&self` acquire is lock-free. The pool itself is intended
//! to be shared via `Arc<HostPtrPool>` across the swap dispatcher / executor.
//!
//! # Safety
//!
//! `cl_mem` handles are context-scoped, not queue-scoped, so a buffer
//! allocated under the backend's main queue stays valid across queue/context
//! reads as long as the `Mem` is alive. The pool keeps one strong reference
//! per slot for the lifetime of the pool.

use anyhow::{Result, anyhow};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};

use crate::backend::opencl::OpenCLBackend;

/// Configuration for the `HostPtrPool`.
#[derive(Debug, Clone, Copy)]
pub struct HostPtrPoolConfig {
    /// Number of pre-allocated slots. Stage 2 measurement on Galaxy S25
    /// (Qwen2.5-1.5B, 28 layers, 7 Q4_0 tensors per layer) reported a
    /// sweet spot around 14 slots (= 2 layers worth of in-flight work).
    pub n_slots: usize,
    /// Maximum bytes per slot. Each slot is allocated up-front at this
    /// size; `acquire(size)` returns `None` when `size > max_tensor_size`.
    /// Default 11 MiB covers Qwen2.5-1.5B's 7.7 MiB FFN tensors with
    /// headroom.
    pub max_tensor_size: usize,
}

impl Default for HostPtrPoolConfig {
    fn default() -> Self {
        Self {
            n_slots: 14,
            max_tensor_size: 11 * 1024 * 1024,
        }
    }
}

/// One pre-allocated `CL_MEM_ALLOC_HOST_PTR` slot in the pool.
struct HostPtrPoolEntry {
    /// The pre-allocated `cl_mem`. Lifetime spans the pool itself.
    mem: ocl::core::Mem,
    /// Allocated capacity in bytes (matches `HostPtrPoolConfig::max_tensor_size`).
    capacity: usize,
    /// Atomically `true` while the slot is held by an outstanding guard.
    in_use: AtomicBool,
}

// SAFETY: `ocl::core::Mem` is `Send + Sync` (context-scoped refcounted handle).
unsafe impl Send for HostPtrPoolEntry {}
unsafe impl Sync for HostPtrPoolEntry {}

/// Pre-allocated pool of `CL_MEM_ALLOC_HOST_PTR | CL_MEM_READ_ONLY` cl_mem
/// slots for runtime weight swap.
///
/// See module docs for stage history and lifecycle.
pub struct HostPtrPool {
    slots: Vec<HostPtrPoolEntry>,
    config: HostPtrPoolConfig,
}

impl HostPtrPool {
    /// Build a fresh pool with `config.n_slots` empty `CL_MEM_ALLOC_HOST_PTR`
    /// buffers of `config.max_tensor_size` bytes each.
    ///
    /// On Adreno UMA the driver returns each buffer as host-pinned + GPU
    /// visible (single VMA), so total RSS overhead is roughly
    /// `n_slots * max_tensor_size`. With the default config (14 × 11 MiB =
    /// 154 MiB) this is the up-front cost paid for skipping the staging copy
    /// on every subsequent swap.
    pub fn new(backend: &OpenCLBackend, config: HostPtrPoolConfig) -> Result<Self> {
        if config.n_slots == 0 {
            return Err(anyhow!("HostPtrPool: n_slots must be > 0"));
        }
        if config.max_tensor_size == 0 {
            return Err(anyhow!("HostPtrPool: max_tensor_size must be > 0"));
        }
        let mut slots: Vec<HostPtrPoolEntry> = Vec::with_capacity(config.n_slots);
        for slot_idx in 0..config.n_slots {
            let mem = backend
                .alloc_host_ptr_buffer_empty(config.max_tensor_size)
                .map_err(|e| {
                    anyhow!(
                        "HostPtrPool: slot {slot_idx} alloc failed (size={}): {e}",
                        config.max_tensor_size
                    )
                })?;
            slots.push(HostPtrPoolEntry {
                mem,
                capacity: config.max_tensor_size,
                in_use: AtomicBool::new(false),
            });
        }
        Ok(Self { slots, config })
    }

    /// Configured number of slots.
    #[inline]
    pub fn n_slots(&self) -> usize {
        self.config.n_slots
    }

    /// Configured max tensor size (= per-slot capacity).
    #[inline]
    pub fn max_tensor_size(&self) -> usize {
        self.config.max_tensor_size
    }

    /// Number of slots currently held by outstanding guards.
    pub fn in_use_count(&self) -> usize {
        self.slots
            .iter()
            .filter(|s| s.in_use.load(Ordering::Acquire))
            .count()
    }

    /// Try to acquire a free slot of at least `size` bytes. Returns `None`
    /// when:
    /// - all slots are currently in use (caller falls back to staging), or
    /// - `size > max_tensor_size` (the slot capacity is fixed).
    ///
    /// The returned guard holds the slot until dropped.
    pub fn acquire(self: &Arc<Self>, size: usize) -> Option<HostPtrPoolGuard> {
        if size > self.config.max_tensor_size {
            return None;
        }
        for (idx, slot) in self.slots.iter().enumerate() {
            // Compare-exchange to atomically claim the slot.
            if slot
                .in_use
                .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
                .is_ok()
            {
                return Some(HostPtrPoolGuard {
                    pool: Arc::clone(self),
                    slot_idx: idx,
                    requested_size: size,
                });
            }
        }
        None
    }
}

/// RAII guard for a `HostPtrPool` slot. Releases the slot on `Drop`.
///
/// Holds an `Arc<HostPtrPool>` so the pool itself can outlive the original
/// caller — useful when the `Tensor` wrapping a pool slot is moved across
/// threads / handed to the async swap dispatcher worker.
pub struct HostPtrPoolGuard {
    pool: Arc<HostPtrPool>,
    slot_idx: usize,
    requested_size: usize,
}

impl HostPtrPoolGuard {
    /// `cl_mem` handle for this slot. Stable across the guard's lifetime.
    #[inline]
    pub fn mem(&self) -> &ocl::core::Mem {
        // SAFETY: slot index is bounded by pool construction; in_use is true
        // while this guard is live so no concurrent access to the same slot.
        &self.pool.slots[self.slot_idx].mem
    }

    /// Bytes the caller asked for at acquire time. Always `<=` slot capacity.
    #[inline]
    pub fn requested_size(&self) -> usize {
        self.requested_size
    }

    /// Slot index inside the pool — for diagnostics only.
    #[inline]
    pub fn slot_idx(&self) -> usize {
        self.slot_idx
    }

    /// Slot capacity (= `max_tensor_size` from pool config).
    #[inline]
    pub fn capacity(&self) -> usize {
        self.pool.slots[self.slot_idx].capacity
    }
}

impl Drop for HostPtrPoolGuard {
    fn drop(&mut self) {
        // Release the slot. Use Release so any writes to mem are visible to
        // the next acquirer.
        self.pool.slots[self.slot_idx]
            .in_use
            .store(false, Ordering::Release);
    }
}

/// Read the `LLMRS_OPENCL_HOST_PTR_POOL` env-gate. Cached after first call.
///
/// Default OFF — Stage 4 measurement-driven decision pending. `=1` /
/// `=true` / `=on` (case-insensitive) enables the pool path; any other
/// value (including unset) keeps it disabled.
pub fn host_ptr_pool_env_enabled() -> bool {
    use std::sync::OnceLock;
    static ENABLED: OnceLock<bool> = OnceLock::new();
    *ENABLED.get_or_init(|| match std::env::var("LLMRS_OPENCL_HOST_PTR_POOL") {
        Ok(v) => {
            let v = v.trim().to_ascii_lowercase();
            v == "1" || v == "true" || v == "on"
        }
        Err(_) => false,
    })
}
