//! Spin-wait thread pool for low-latency matmul dispatch.
//!
//! Unlike Rayon's fork-join model which creates tasks per parallel call,
//! this pool keeps worker threads spinning between dispatches, achieving
//! near-zero per-call overhead (~100ns vs Rayon's ~300µs).
//!
//! Workers use atomic fetch_add for work-stealing, which naturally
//! balances load across heterogeneous cores (big.LITTLE).

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU64, AtomicUsize, Ordering};

/// Function type for work items: (context_pointer, chunk_id)
pub type WorkFn = unsafe fn(*const u8, usize);

struct SharedState {
    /// Incremented each dispatch — workers spin on this.
    generation: AtomicU64,
    /// Workers fetch_add to grab the next chunk (work-stealing).
    next_chunk: AtomicUsize,
    /// Total chunks for current dispatch.
    total_chunks: AtomicUsize,
    /// Workers increment when done with current dispatch.
    done_count: AtomicUsize,
    /// Shutdown signal.
    shutdown: AtomicBool,
    /// Number of worker threads (excludes main thread).
    n_workers: usize,
    /// Work function pointer (stored as usize for atomicity).
    work_fn: AtomicUsize,
    /// Work context pointer (stored as usize for atomicity).
    work_ctx: AtomicUsize,
}

pub struct SpinPool {
    shared: Arc<SharedState>,
    _handles: Vec<std::thread::JoinHandle<()>>,
}

impl SpinPool {
    /// Create a pool with `n_workers` background threads.
    /// The main thread also participates during dispatch (total = n_workers + 1).
    pub fn new(n_workers: usize) -> Self {
        let shared = Arc::new(SharedState {
            generation: AtomicU64::new(0),
            next_chunk: AtomicUsize::new(0),
            total_chunks: AtomicUsize::new(0),
            done_count: AtomicUsize::new(0),
            shutdown: AtomicBool::new(false),
            n_workers,
            work_fn: AtomicUsize::new(0),
            work_ctx: AtomicUsize::new(0),
        });

        let mut handles = Vec::with_capacity(n_workers);
        for _ in 0..n_workers {
            let s = shared.clone();
            handles.push(std::thread::spawn(move || {
                Self::worker_loop(&s);
            }));
        }

        SpinPool {
            shared,
            _handles: handles,
        }
    }

    fn worker_loop(shared: &SharedState) {
        let mut last_gen = 0u64;

        loop {
            // Phase 1: spin-wait for new work (fast path)
            let mut spins = 0u32;
            let cur_gen = loop {
                if shared.shutdown.load(Ordering::Relaxed) {
                    return;
                }
                let g = shared.generation.load(Ordering::Acquire);
                if g != last_gen {
                    break g;
                }
                spins += 1;
                if spins < 4000 {
                    std::hint::spin_loop();
                } else {
                    // Yield to OS after ~1µs of spinning to reduce power
                    std::thread::yield_now();
                    spins = 2000; // don't reset to 0, keep yielding
                }
            };
            last_gen = cur_gen;

            // Work-stealing: grab chunks via atomic counter
            Self::steal_work(shared);

            shared.done_count.fetch_add(1, Ordering::Release);
        }
    }

    #[inline]
    fn steal_work(shared: &SharedState) {
        let work_fn: WorkFn =
            unsafe { std::mem::transmute(shared.work_fn.load(Ordering::Relaxed)) };
        let work_ctx = shared.work_ctx.load(Ordering::Relaxed) as *const u8;
        let total = shared.total_chunks.load(Ordering::Relaxed);

        loop {
            let chunk = shared.next_chunk.fetch_add(1, Ordering::Relaxed);
            if chunk >= total {
                break;
            }
            unsafe {
                work_fn(work_ctx, chunk);
            }
        }
    }

    /// Dispatch `n_chunks` work items. Blocks until all are processed.
    ///
    /// Main thread participates in work-stealing alongside workers.
    ///
    /// # Safety
    /// `work_fn(ctx, chunk_id)` must be safe to call for all chunk_id in 0..n_chunks.
    /// `ctx` must remain valid until dispatch returns.
    pub unsafe fn dispatch(&self, n_chunks: usize, work_fn: WorkFn, ctx: *const u8) {
        if n_chunks == 0 {
            return;
        }

        // Set up work (Relaxed: visible via generation Release below)
        self.shared
            .work_fn
            .store(work_fn as usize, Ordering::Relaxed);
        self.shared.work_ctx.store(ctx as usize, Ordering::Relaxed);
        self.shared.total_chunks.store(n_chunks, Ordering::Relaxed);
        self.shared.next_chunk.store(0, Ordering::Relaxed);
        self.shared.done_count.store(0, Ordering::Relaxed);

        // Wake workers (Release ensures work setup is visible)
        self.shared.generation.fetch_add(1, Ordering::Release);

        // Main thread also steals work
        Self::steal_work(&self.shared);

        // Wait for all workers to finish
        while self.shared.done_count.load(Ordering::Acquire) < self.shared.n_workers {
            std::hint::spin_loop();
        }
    }
}

impl Drop for SpinPool {
    fn drop(&mut self) {
        self.shared.shutdown.store(true, Ordering::Release);
        // Bump generation to wake any sleeping workers
        self.shared.generation.fetch_add(1, Ordering::Release);
        for h in self._handles.drain(..) {
            let _ = h.join();
        }
    }
}

// Global singleton — initialized lazily on first matmul call.
static POOL: std::sync::OnceLock<SpinPool> = std::sync::OnceLock::new();

/// Get the global SpinPool.
/// Uses 3 workers + main thread = 4 total (targeting big cores on ARM SoCs).
/// More workers cause cache/bus contention during non-matmul operations.
pub fn get_pool() -> &'static SpinPool {
    POOL.get_or_init(|| {
        // 3 workers + main = 4 threads (prime + 3 big cores on Snapdragon 8 Gen 3)
        // Optimal: 3 workers + main = 4 threads.
        // On Snapdragon 8 Gen 3: saturates DRAM bandwidth with 4 big cores.
        // More threads cause cache/bus contention from spin-wait overhead.
        SpinPool::new(3)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::AtomicU64;

    #[test]
    fn test_spin_pool_basic() {
        let pool = SpinPool::new(3);
        let counter = AtomicU64::new(0);

        unsafe fn add_work(ctx: *const u8, _chunk_id: usize) {
            let c = &*(ctx as *const AtomicU64);
            c.fetch_add(1, Ordering::Relaxed);
        }

        unsafe {
            pool.dispatch(100, add_work, &counter as *const _ as *const u8);
        }
        assert_eq!(counter.load(Ordering::Relaxed), 100);
    }

    #[test]
    fn test_spin_pool_work_stealing() {
        let pool = SpinPool::new(4);
        let results = vec![AtomicU64::new(0); 1000];

        unsafe fn mark_work(ctx: *const u8, chunk_id: usize) {
            let results = &*(ctx as *const Vec<AtomicU64>);
            results[chunk_id].fetch_add(1, Ordering::Relaxed);
        }

        unsafe {
            pool.dispatch(1000, mark_work, &results as *const _ as *const u8);
        }

        // Every chunk should be processed exactly once
        for (i, r) in results.iter().enumerate() {
            assert_eq!(
                r.load(Ordering::Relaxed),
                1,
                "chunk {} processed {} times",
                i,
                r.load(Ordering::Relaxed)
            );
        }
    }

    #[test]
    fn test_spin_pool_multiple_dispatches() {
        let pool = SpinPool::new(2);
        let counter = AtomicU64::new(0);

        unsafe fn add_work(ctx: *const u8, _chunk_id: usize) {
            let c = &*(ctx as *const AtomicU64);
            c.fetch_add(1, Ordering::Relaxed);
        }

        for _ in 0..50 {
            unsafe {
                pool.dispatch(20, add_work, &counter as *const _ as *const u8);
            }
        }
        assert_eq!(counter.load(Ordering::Relaxed), 1000);
    }
}
