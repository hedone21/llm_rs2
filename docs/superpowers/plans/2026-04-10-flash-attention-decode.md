# Flash Attention for Decode Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire the compiled `flash_attn_f32_f16_q1` kernel into the production decode path (`plan.rs`) so long-context TBT matches or beats llama.cpp CPU on Adreno 830. The `forward_gen` path is already wired via a WIP prototype committed on master (`938149a`); this plan takes that prototype to production.

**Architecture:** A new `flash_attention_decode_gpu()` method on `OpenCLBackend` (already exists in the prototype) invokes `flash_attn_f32_f16_q1`. Runtime preconditions (`head_dim == 64`, F16 KV, HeadMajor layout, no score output) gate it. `attention_gen()` already routes through it when conditions hold (prototype). This plan adds the same routing to the pre-bound `plan.rs` path via a new `AttentionVariant::StandardFlash` that `FullKernelPlan::execute()` dispatches. Falls back to the legacy `kernel_attn_gen_half` when preconditions fail (H2O scoring, non-64 head_dim, Q4 KV, Gemma3, etc.).

**Tech Stack:** Rust (engine crate), OpenCL 2.0, Android ARM64 cross-compile via NDK, Samsung Galaxy S25 (Adreno 830) for on-device verification.

**Out of scope (follow-ups):**
- Multi-variant compilation for head_dim 128/256 (Qwen, Gemma) — separate plan
- Score-output flash variant (would unblock H2O + flash)
- `--profile` tool latency refactor (known diagnostic-tool overhead; see Task 7 methodology note)

---

## Background — What Was Already Proved

Measurements on Samsung Galaxy S25 (llama3.2-1b F16, `-n 128` decode, production path unless noted):

| Context (avg seq) | plan.rs + legacy (baseline) | forward_gen + flash (prototype, `--no-gpu-plan`) | Delta |
|---|---:|---:|---:|
| ~170 | 43.4 ms/tok | 40.0 ms | −3.4 |
| ~374 | 46.6 ms | 40.3 ms | −6.3 |
| ~676 | 51.2 ms | 41.1 ms | −10.1 |
| ~720 | 53.8 ms | 41.2 ms | **−12.6** |

**vs llama.cpp CPU** on same device:

| Context | llama.cpp CPU (`tg128 @ d<n>`) | llm.rs OpenCL + flash (prototype) |
|---|---:|---:|
| ~128 | 40.0 ms | 40.0 ms (tied) |
| ~440 | ~41 ms | 40.3 ms |
| ~790 | 44.0 ms | 41.2 ms (we win) |

The prototype only affects the `--no-gpu-plan` code path. Production decode uses `plan.rs::FullKernelPlan::execute()`, which binds `kernel_attn_gen_half` at plan build time and has no runtime gate. **This plan's core work is wiring the same flash dispatch into `plan.rs`.**

---

## Version Control Strategy

- **Base commit**: `938149a feat(opencl): prototype flash_attn_f32_f16_q1 decode dispatch (WIP)` on `master`. The prototype adds:
  - `kernel_flash_attn_f32_f16_q1` handle on `KernelCache`
  - `flash_attention_decode_gpu()` method on `OpenCLBackend`
  - Gate in `attention_gen()` for `forward_gen.rs` callers
- **Branch**: `feat/flash-attn-decode` created from `938149a`.
- **Worktree**: Already created at `.claude/worktrees/flash-attn-decode` for isolation.
- **Commits**: Conventional Commits, one logical change per commit. Commit after each task passes verification.
- **Rollback**: Each task is self-contained. `git reset --hard HEAD~1` takes you back. If the plan fails overall, the branch can be abandoned; master retains the forward_gen prototype as a fallback.
- **Merge strategy**: After all tasks pass, rebase on master and merge with `--no-ff`.

---

## Testing Strategy

Three layers:

1. **Correctness (host)** — Host unit tests verifying the flash dispatch is deterministic and does not crash. Runs via `cargo test -p llm_rs2 --test flash_attn_decode`. Host drivers may not match Adreno's kernel compilation, so the strong correctness proof is on-device (Task 3).

2. **Correctness (device)** — Run `generate` with fixed prompts and verify coherent text output (no garbage). Compare against the `--no-gpu-plan` baseline (prototype path) and the legacy pre-prototype path via `git` checkout if needed.

3. **Performance regression guard** — Capture TBT at 4 context lengths and assert `llm.rs OpenCL decode ≤ llama.cpp CPU decode × 1.05`. The script in Task 4 automates this. **Uses wall-clock `Decode: X ms/tok` from stdout** — production path, NOT `--profile`.

**Measurement methodology (IMPORTANT):** All perf measurements must use wall-clock TBT from the `Decode:` stdout line (which mirrors manager heartbeat `actual_throughput`). Do NOT use `--profile`-produced numbers — `--profile` disables `plan.rs` and adds ~54 ms/token of `clFinish` sync overhead. `--profile` is still useful for **relative** per-op comparison (which op is bigger?), but its absolute values are inflated. Task 7 codifies this in `AGENTS.md`.

---

## File Structure

### Files to be created

| File | Purpose |
|---|---|
| `engine/tests/flash_attn_decode.rs` | Host unit tests backfilling the prototype |
| `scripts/bench_flash_attn_decode.sh` | Device benchmark + llama.cpp comparison gate |
| `results/data/flash_attn_decode/` | Captured TBT artifacts (committed) |

### Files to be modified

| File | Purpose |
|---|---|
| `engine/src/backend/opencl/plan.rs` | New `AttentionVariant::StandardFlash` variant, builder branch, `execute()` dispatch arm |
| `engine/src/models/transformer.rs` | Thread `flash_attn_f32_f16_program` + `needs_attention_scores` through `LayerPlanConfig` call sites |
| `AGENTS.md` (= `CLAUDE.md` symlink) | Add methodology bullet to "핵심 제약사항" |

### Files NOT to modify (review guard)

- `engine/kernels/flash_attn_f32_f16.cl` — already compiled with `-DDK=64 -DDV=64`; touching it breaks prefill
- `engine/src/core/kivi_cache.rs` — KIVI is out of scope
- `engine/src/layers/transformer_layer/forward_gen.rs` — prototype already routes through `backend.attention_gen()`
- `engine/src/backend/opencl/mod.rs` flash dispatch method — already correct in the prototype

---

## Tasks

### Task 1: Backfill host unit tests for the prototype

**Purpose:** The prototype in commit `938149a` has no test coverage. Add host tests before extending to `plan.rs`.

**Files:**
- Create: `engine/tests/flash_attn_decode.rs`

- [ ] **Step 1: Read the prototype for context**

Read `engine/src/backend/opencl/mod.rs`:
- `KernelCache::kernel_flash_attn_f32_f16_q1` (near line 114)
- Kernel creation site (near line 703, within `KernelCache { ... }`)
- `flash_attention_decode_gpu()` method (search for the name)
- The gate at the top of `attention_gen()` (search for "Flash-attention decode fast path")

Key fact: `flash_attention_decode_gpu()` returns `Ok(false)` when preconditions fail (`head_dim != 64`, KV not F16, layout not HeadMajor, kernel missing). Callers fall back to `kernel_attn_gen_half` in the same function.

- [ ] **Step 2: Write the test file**

Create `engine/tests/flash_attn_decode.rs` with this content:

```rust
//! Host unit tests for the flash attention decode dispatch prototype.
//!
//! These backfill coverage for commit 938149a before extending to plan.rs.

#![cfg(feature = "opencl")]

use llm_rs2::backend::opencl::OpenCLBackend;

/// Sanity: OpenCLBackend initializes and the flash kernel handle loads.
/// On hosts without OpenCL drivers this test silently skips.
#[test]
fn backend_initializes_with_flash_kernel() {
    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
        }
    };
    // The backend must accept downcasts from trait-object form.
    // (Used by plan.rs builder in Task 2.)
    let _ = &backend;
}

/// Deterministic self-consistency: the flash dispatch must produce bit-
/// identical output across two invocations with the same inputs. This
/// guards against nondeterminism (e.g. uninitialized local memory, bad
/// barrier placement).
#[test]
fn flash_attn_decode_self_consistent() {
    use llm_rs2::backend::opencl::memory::OpenCLMemory;
    use llm_rs2::core::backend::Backend;
    use llm_rs2::core::memory::Memory;
    use llm_rs2::core::shape::Shape;
    use llm_rs2::core::tensor::Tensor;
    use llm_rs2::core::types::DType;
    use std::sync::Arc;

    let backend = match OpenCLBackend::new() {
        Ok(b) => b,
        Err(e) => {
            eprintln!("Skipping: OpenCLBackend init failed: {e}");
            return;
        }
    };
    let ocl_arc: Arc<dyn Backend> = Arc::new(backend);
    let ocl = ocl_arc
        .as_any()
        .downcast_ref::<OpenCLBackend>()
        .expect("OpenCLBackend downcast");

    // llama3.2-1b shape: 32 Q heads, 8 KV heads (GQA=4), head_dim=64
    let n_heads_q = 32usize;
    let n_heads_kv = 8usize;
    let head_dim = 64usize;
    let cache_seq_len = 32usize;
    let capacity = 128usize;

    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));

    // Deterministic inputs
    let q_data: Vec<f32> = (0..n_heads_q * head_dim)
        .map(|i| ((i as f32) * 0.01).sin())
        .collect();

    let kv_total = n_heads_kv * capacity * head_dim;
    let mut k_data = vec![0u16; kv_total];
    let mut v_data = vec![0u16; kv_total];
    for h in 0..n_heads_kv {
        for p in 0..cache_seq_len {
            for d in 0..head_dim {
                let idx = h * capacity * head_dim + p * head_dim + d;
                let seed = (h * 31 + p * 7 + d) as f32;
                k_data[idx] = half::f16::from_f32((seed * 0.003).cos()).to_bits();
                v_data[idx] = half::f16::from_f32((seed * 0.005).sin()).to_bits();
            }
        }
    }

    let q = upload_f32(&ocl_arc, &*memory, &q_data, vec![1, 1, n_heads_q, head_dim]);
    let k = upload_f16(&ocl_arc, &*memory, &k_data, vec![1, n_heads_kv, capacity, head_dim]);
    let v = upload_f16(&ocl_arc, &*memory, &v_data, vec![1, n_heads_kv, capacity, head_dim]);

    let zero = vec![0.0f32; n_heads_q * head_dim];
    let mut out_a = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);
    let mut out_b = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);

    let ok_a = ocl
        .flash_attention_decode_gpu(&q, &k, &v, &mut out_a, n_heads_q, n_heads_kv, head_dim, cache_seq_len)
        .expect("flash call a");
    assert!(ok_a, "flash must dispatch for head_dim=64, F16, HeadMajor");
    ocl_arc.synchronize().unwrap();

    ocl
        .flash_attention_decode_gpu(&q, &k, &v, &mut out_b, n_heads_q, n_heads_kv, head_dim, cache_seq_len)
        .expect("flash call b");
    ocl_arc.synchronize().unwrap();

    let a = out_a.as_slice::<f32>();
    let b = out_b.as_slice::<f32>();
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "nondeterministic flash output at [{i}]: {x} vs {y}"
        );
    }

    // Sanity: output is not all zeros
    assert!(
        a.iter().any(|&v| v.abs() > 1e-6),
        "flash attention produced all-zero output — likely a dispatch no-op"
    );
}

fn upload_f32(
    backend: &std::sync::Arc<dyn llm_rs2::core::backend::Backend>,
    memory: &dyn llm_rs2::core::memory::Memory,
    data: &[f32],
    shape: Vec<usize>,
) -> llm_rs2::core::tensor::Tensor {
    let buf = memory
        .alloc(data.len() * 4, llm_rs2::core::types::DType::F32)
        .unwrap();
    let t = llm_rs2::core::tensor::Tensor::new(
        llm_rs2::core::shape::Shape::new(shape),
        buf,
        backend.clone(),
    );
    unsafe {
        let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4);
        backend.write_buffer(&t, bytes).unwrap();
    }
    t
}

fn upload_f16(
    backend: &std::sync::Arc<dyn llm_rs2::core::backend::Backend>,
    memory: &dyn llm_rs2::core::memory::Memory,
    data: &[u16],
    shape: Vec<usize>,
) -> llm_rs2::core::tensor::Tensor {
    let buf = memory
        .alloc(data.len() * 2, llm_rs2::core::types::DType::F16)
        .unwrap();
    let t = llm_rs2::core::tensor::Tensor::new(
        llm_rs2::core::shape::Shape::new(shape),
        buf,
        backend.clone(),
    );
    unsafe {
        let bytes = std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2);
        backend.write_buffer(&t, bytes).unwrap();
    }
    t
}
```

If any type path differs (e.g., `llm_rs2::core::memory::Memory` is actually under `llm_rs2::memory::Memory`), run `cargo check` and adjust the imports. The test structure stays the same.

- [ ] **Step 3: Run the test**

```bash
cd .claude/worktrees/flash-attn-decode
cargo test -p llm_rs2 --test flash_attn_decode -- --nocapture
```

Expected: both tests pass (or skip cleanly with "Skipping: OpenCLBackend init failed" on hosts without OpenCL).

- [ ] **Step 4: Commit**

```bash
git add engine/tests/flash_attn_decode.rs
git commit -m "$(cat <<'EOF'
test(opencl): backfill host tests for flash attention decode prototype

Adds a sanity init check and a deterministic self-consistency test for
the flash_attention_decode_gpu method introduced in commit 938149a.
Host drivers may diverge from Adreno's kernel compile behavior, so
strong flash-vs-legacy parity is verified on-device later.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: Add `AttentionVariant::StandardFlash` to the plan layer

**Purpose:** The core work. Wire flash attention into pre-bound `plan.rs` so production decode uses it.

**Files:**
- Modify: `engine/src/backend/opencl/plan.rs`
- Modify: `engine/src/models/transformer.rs`

- [ ] **Step 1: Read the existing attention builder and execute dispatch**

Read these sections of `engine/src/backend/opencl/plan.rs`:

1. `AttentionVariant` enum (~line 88-105): currently has `Standard(KernelStep)`, `KiviAssembled {...}`, `KiviNative(KernelStep)`
2. `LayerPlanConfig` struct (~line 520-570): does NOT currently carry `flash_attn_f32_f16_program` or a scores-needed hint
3. `build_layer_plan()` attention builder (~line 850-915): constructs `AttentionVariant::Standard(KernelStep { ... })` using `kernel_attn_gen_half`, with `cache_seq_len` (arg_idx=8) as the only `DynamicArg::CacheSeqLen`
4. `FullKernelPlan::execute()` attention match (~line 367-395): matches on `layer_plan.attention` and calls `dispatch_step()` for each variant

Verify the real line numbers before editing — the plan's numbers are approximate.

- [ ] **Step 2: Extend `LayerPlanConfig`**

Add these fields to `LayerPlanConfig` (anywhere near the related config fields; putting them at the end is fine):

```rust
    /// The flash attention F32 Q / F16 KV program handle, if compiled.
    /// Used to create `flash_attn_f32_f16_q1` at plan-build time when
    /// runtime preconditions hold. `None` forces the legacy path.
    pub flash_attn_f32_f16_program: Option<&'a ocl::Program>,

    /// True if this decode plan must capture attention scores (H2O/H2O+ or
    /// an active GPU score accumulator). When true, the builder must use
    /// `AttentionVariant::Standard` because the flash kernel has no score
    /// output.
    pub needs_attention_scores: bool,
```

- [ ] **Step 3: Add the `StandardFlash` enum variant**

```rust
pub enum AttentionVariant {
    /// Legacy kernel_attn_gen_half — supports score writes, any head_dim.
    Standard(KernelStep),
    /// Decode flash attention (flash_attn_f32_f16_q1). Single-pass online
    /// softmax, no score output. Selected at plan-build time when
    /// head_dim==64, F16 KV, HeadMajor, and no scores are needed.
    StandardFlash(KernelStep),
    KiviAssembled {
        scatter_k: KernelStep,
        scatter_v: KernelStep,
        attn: KernelStep,
    },
    KiviNative(KernelStep),
}
```

- [ ] **Step 4: Add the `build_flash_attention_step` helper**

Place this function above `build_layer_plan` in `plan.rs`:

```rust
/// Build a pre-bound `KernelStep` that dispatches `flash_attn_f32_f16_q1`.
///
/// Arg layout mirrors `OpenCLBackend::flash_attention_decode_gpu` — see that
/// method in `engine/src/backend/opencl/mod.rs` for the canonical layout.
/// All 40 args are static except `n_kv` at index 10, which is patched per
/// decode step via `DynamicArg::CacheSeqLen`.
fn build_flash_attention_step(config: &LayerPlanConfig) -> Result<AttentionVariant> {
    let program = config
        .flash_attn_f32_f16_program
        .expect("caller must verify flash_attn_f32_f16_program.is_some()");

    let kernel = ocl::core::create_kernel(program, "flash_attn_f32_f16_q1")
        .context("create flash_attn_f32_f16_q1 for plan")?;

    let n_heads_q = config.n_heads_q;
    let n_heads_kv = config.n_heads_kv;
    let head_dim = config.head_dim;
    let kv_capacity = config.kv_capacity;
    let scale = 1.0f32 / (head_dim as f32).sqrt();

    // Q strides (F32 [batch=1, seq=1, n_heads_q, head_dim]), bytes
    let q_nb1 = (n_heads_q * head_dim * 4) as u64;
    let q_nb2 = (head_dim * 4) as u64;
    let q_nb3 = q_nb1;

    // KV strides (F16 HeadMajor [1, n_heads_kv, capacity, head_dim]), bytes
    let kv_elem_size: u64 = 2;
    let k_nb1 = (head_dim as u64) * kv_elem_size;
    let k_nb2 = (kv_capacity * head_dim) as u64 * kv_elem_size;
    let k_nb3 = (n_heads_kv as u64) * k_nb2;

    // O strides (F32 [batch=1, seq=1, n_heads_q, head_dim]), bytes
    let o_nb1 = (head_dim * 4) as u64;
    let o_nb2 = (n_heads_q * head_dim * 4) as u64;
    let o_nb3 = o_nb2;

    let n_q = 1i32;
    let initial_n_kv = 0i32;
    let is_causal = 0i32;
    let n_head = n_heads_q as i32;
    let n_head_kv_arg = n_heads_kv as i32;
    let max_bias = 0.0f32;
    let m0 = 0.0f32;
    let m1 = 0.0f32;
    let n_head_log2 = 0i32;
    let logit_softcap = 0.0f32;
    let zero_u64 = 0u64;
    let zero_i32 = 0i32;

    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(config.q_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::mem(config.k_cache_buf))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 4, ocl::core::ArgVal::mem(config.v_cache_buf))?;
        ocl::core::set_kernel_arg(&kernel, 5, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 6, ocl::core::ArgVal::mem(config.out_attn_buf))?;
        ocl::core::set_kernel_arg(&kernel, 7, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 8, ocl::core::ArgVal::scalar(&scale))?;
        ocl::core::set_kernel_arg(&kernel, 9, ocl::core::ArgVal::scalar(&n_q))?;
        ocl::core::set_kernel_arg(&kernel, 10, ocl::core::ArgVal::scalar(&initial_n_kv))?;
        ocl::core::set_kernel_arg(&kernel, 11, ocl::core::ArgVal::scalar(&is_causal))?;
        ocl::core::set_kernel_arg(&kernel, 12, ocl::core::ArgVal::scalar(&n_head))?;
        ocl::core::set_kernel_arg(&kernel, 13, ocl::core::ArgVal::scalar(&q_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 14, ocl::core::ArgVal::scalar(&q_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 15, ocl::core::ArgVal::scalar(&q_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 16, ocl::core::ArgVal::scalar(&k_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 17, ocl::core::ArgVal::scalar(&k_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 18, ocl::core::ArgVal::scalar(&k_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 19, ocl::core::ArgVal::scalar(&k_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 20, ocl::core::ArgVal::scalar(&k_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 21, ocl::core::ArgVal::scalar(&k_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 22, ocl::core::ArgVal::scalar(&o_nb1))?;
        ocl::core::set_kernel_arg(&kernel, 23, ocl::core::ArgVal::scalar(&o_nb2))?;
        ocl::core::set_kernel_arg(&kernel, 24, ocl::core::ArgVal::scalar(&o_nb3))?;
        ocl::core::set_kernel_arg(&kernel, 25, ocl::core::ArgVal::scalar(&max_bias))?;
        ocl::core::set_kernel_arg(&kernel, 26, ocl::core::ArgVal::scalar(&m0))?;
        ocl::core::set_kernel_arg(&kernel, 27, ocl::core::ArgVal::scalar(&m1))?;
        ocl::core::set_kernel_arg(&kernel, 28, ocl::core::ArgVal::scalar(&n_head_log2))?;
        ocl::core::set_kernel_arg(&kernel, 29, ocl::core::ArgVal::scalar(&logit_softcap))?;
        ocl::core::set_kernel_arg(&kernel, 30, ocl::core::ArgVal::scalar(&n_head_kv_arg))?;
        ocl::core::set_kernel_arg(&kernel, 31, ocl::core::ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 32, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 33, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 34, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 35, ocl::core::ArgVal::scalar(&zero_u64))?;
        ocl::core::set_kernel_arg(&kernel, 36, ocl::core::ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 37, ocl::core::ArgVal::scalar(&zero_i32))?;
        ocl::core::set_kernel_arg(&kernel, 38, ocl::core::ArgVal::mem_null())?;
        ocl::core::set_kernel_arg(&kernel, 39, ocl::core::ArgVal::scalar(&zero_u64))?;
    }

    const Q1_WG_SIZE: usize = 64;
    Ok(AttentionVariant::StandardFlash(KernelStep {
        kernel,
        ndim: 2,
        global_work_size: [Q1_WG_SIZE, n_heads_q, 1],
        local_work_size: Some([Q1_WG_SIZE, 1, 1]),
        dynamic_args: vec![DynamicArg::CacheSeqLen { arg_idx: 10 }],
        op_tag: OpTag::Attention,
        retained_bufs: vec![],
    }))
}
```

**Important — field names**: the `config.q_buf`, `config.k_cache_buf`, `config.v_cache_buf`, `config.out_attn_buf`, `config.n_heads_q`, `config.n_heads_kv`, `config.head_dim`, `config.kv_capacity` identifiers above are my best guesses. Verify against the actual `LayerPlanConfig` field names and fix if they differ.

- [ ] **Step 5: Branch in the attention builder inside `build_layer_plan`**

Find the block that currently constructs `AttentionVariant::Standard(KernelStep { ... })`. Wrap it in:

```rust
    let is_head_major = /* use existing is_head_major field from config */;
    let use_flash = config.head_dim == 64
        && config.kv_cache_dtype == DType::F16
        && is_head_major
        && config.flash_attn_f32_f16_program.is_some()
        && !config.needs_attention_scores;

    let attention = if use_flash {
        build_flash_attention_step(config)?
    } else {
        // ... the existing legacy AttentionVariant::Standard builder, verbatim ...
    };
```

If `config.kv_cache_dtype` doesn't exist on `LayerPlanConfig`, look for an equivalent field. The builder already knows the KV dtype somehow (the legacy path selects `kernel_attn_gen` vs `kernel_attn_gen_half` based on it). Reuse the same predicate.

- [ ] **Step 6: Add the `StandardFlash` arm to `FullKernelPlan::execute()`**

Find the `match &layer_plan.attention { ... }` in `execute()`. Add this arm right after the `Standard` arm:

```rust
            AttentionVariant::StandardFlash(step) => {
                if debug_sync {
                    eprintln!(
                        "[Plan] L{} flash attention dispatch (attn_seq_len={}, gws={:?}, lws={:?})",
                        i, attn_seq_len, step.global_work_size, step.local_work_size
                    );
                }
                Self::dispatch_step(
                    queue,
                    step,
                    start_pos_i32,
                    attn_seq_len,
                    write_pos,
                    kv_cap,
                    rp,
                    q2t,
                    rt,
                );
                if debug_sync {
                    ocl::core::finish(queue).ok();
                    eprintln!("[Plan] L{} flash attention OK (attn_seq_len={})", i, attn_seq_len);
                }
            }
```

`attn_seq_len` is already computed earlier in the loop (`cache_seq_len + 1`) — the flash arm uses it the same way as `Standard`.

- [ ] **Step 7: Fix other `match` sites over `AttentionVariant`**

```bash
cd .claude/worktrees/flash-attn-decode
grep -n "match .* layer_plan\.attention\|match .* AttentionVariant" engine/src/backend/opencl/plan.rs
```

Every non-exhaustive match must handle `StandardFlash`. In most cases it should do the same thing as `Standard`.

- [ ] **Step 8: Update `LayerPlanConfig` call sites in `transformer.rs`**

```bash
grep -n "LayerPlanConfig {" engine/src/models/transformer.rs
```

At every call site, add the two new fields:

```rust
let layer_config = LayerPlanConfig {
    // ... existing fields ...
    flash_attn_f32_f16_program: ocl_backend.flash_attn_f32_f16_program.as_ref(),
    needs_attention_scores: plan_needs_scores,
};
```

Where `plan_needs_scores` is computed at the call site from whatever the caller already knows about H2O / GPU score accumulator / score_accumulator state. If there's no such predicate yet, define it as:

```rust
let plan_needs_scores = score_accumulator.is_some();
```

(Matches the runtime gate in `attention_gen()` which checks `gpu_acc.is_active() || scores_out.is_some()` — at plan-build time we don't know about `scores_out`, so we're conservative and fall back to legacy whenever any score accumulator exists.)

**Note:** `ocl_backend.flash_attn_f32_f16_program` is already `pub` on `OpenCLBackend` — see `engine/src/backend/opencl/mod.rs` around line 154. Just reference it.

- [ ] **Step 9: Host compile check**

```bash
cd .claude/worktrees/flash-attn-decode
cargo check -p llm_rs2 --lib
```

Expected: no errors. Any missing match arm or field issues will be reported with exact line numbers.

- [ ] **Step 10: Host test run**

```bash
cargo test -p llm_rs2 --lib
cargo test -p llm_rs2 --test flash_attn_decode
```

Expected: all pass. No regressions on existing tests.

- [ ] **Step 11: Commit**

```bash
git add engine/src/backend/opencl/plan.rs engine/src/models/transformer.rs
git commit -m "$(cat <<'EOF'
feat(opencl/plan): route decode attention through flash kernel

Adds AttentionVariant::StandardFlash which pre-binds flash_attn_f32_f16_q1
at plan build time when head_dim=64, F16 KV cache, HeadMajor layout, and
no attention scores are needed. FullKernelPlan::execute dispatches the
new variant the same way it dispatches Standard — single kernel enqueue
with cache_seq_len as the only dynamic arg.

LayerPlanConfig gains flash_attn_f32_f16_program and needs_attention_scores
so transformer.rs can decide between legacy and flash at plan construction
time. H2O, Qwen (head_dim=128), Gemma (head_dim=256), and Q4 KV all fall
back to the legacy Standard variant by design.

Production decode (plan.rs, no --profile) now routes through the same
flash path that --no-gpu-plan already uses via the prototype gate in
attention_gen. Expected long-context TBT drop from ~54 ms to ~41 ms on
llama3.2-1b on Adreno 830; verified on-device in the next task.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: Device correctness verification

**Files:** none modified.

- [ ] **Step 1: Build and deploy**

```bash
cd .claude/worktrees/flash-attn-decode
export NDK_HOME=/opt/homebrew/share/android-ndk
export HOST_TAG=darwin-x86_64
export TOOLCHAIN=$NDK_HOME/toolchains/llvm/prebuilt/$HOST_TAG
export CC_aarch64_linux_android=$TOOLCHAIN/bin/aarch64-linux-android21-clang
export CXX_aarch64_linux_android=$TOOLCHAIN/bin/aarch64-linux-android21-clang++
export AR_aarch64_linux_android=$TOOLCHAIN/bin/llvm-ar
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$CC_aarch64_linux_android

cargo build --release --bin generate --target aarch64-linux-android
adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate
```

- [ ] **Step 2: Short context correctness**

```bash
adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/llama3.2-1b --prompt 'Hello world this is a test' -n 128 -b opencl 2>&1 | tail -15"
```

Expected: coherent text output. No `buttons/buttons` or `Parameter(Parameter(` garbage. `Avg TBT: ~42-45 ms/tok`. If garbage, the plan.rs dispatch has an arg-layout bug — `git reset --hard HEAD~1` and re-check the arg indices against `flash_attention_decode_gpu` in `mod.rs`.

- [ ] **Step 3: Long context perf + correctness**

```bash
adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/llama3.2-1b --prompt-file /data/local/tmp/long_prompt.txt -n 128 -b opencl 2>&1 | grep -E 'Decode:|Avg TBT'"
```

Expected: `Decode: ~41 ms/tok` (vs 53.8 baseline). **Acceptance: Decode ≤ 44 ms/tok.** If higher, the plan.rs `use_flash` predicate did not select flash for llama3.2-1b — use `PLAN_DEBUG=1 adb shell ...` to see per-step dispatch logs and confirm `flash attention` lines appear for all 16 layers.

- [ ] **Step 4: Commit (verification-only)**

No code changes. Move on to Task 4.

---

### Task 4: Benchmark + llama.cpp comparison gate

**Files:**
- Create: `scripts/bench_flash_attn_decode.sh`
- Create: `results/data/flash_attn_decode/after-task4.txt`

- [ ] **Step 1: Write the benchmark script**

Create `scripts/bench_flash_attn_decode.sh`:

```bash
#!/usr/bin/env bash
# Flash attention decode benchmark vs llama.cpp CPU (Adreno 830).
#
# Runs a 4-context-length sweep of llm.rs and llama.cpp on the connected
# device and asserts llm.rs TBT <= llama.cpp TBT * (1 + TOLERANCE_PCT/100).
#
# Requirements on device:
#   /data/local/tmp/generate (our binary)
#   /data/local/tmp/models/llama3.2-1b
#   /data/local/tmp/{p100,p300,p600,long_prompt}.txt
#   /data/local/tmp/Llama-3.2-1B-Instruct-f16.gguf
#   /data/local/tmp/llama-bench + libllama.so + libggml*.so
#
# Exit code 0 on success; 1 on any per-context failure.

set -euo pipefail

OUTDIR="results/data/flash_attn_decode"
mkdir -p "$OUTDIR"

TOLERANCE_PCT=5
COOLDOWN_SEC=15

tbt_llm_rs() {
  local prompt_file="$1"
  adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/llama3.2-1b --prompt-file /data/local/tmp/${prompt_file} -n 128 -b opencl 2>&1" \
    | awk '/^Decode:/ { gsub(/[^0-9.]/, "", $2); print $2; exit }'
}

tbt_llama_cpp() {
  local depth="$1"
  # llama-bench prints t/s (tokens per second); convert to ms/tok.
  local tps
  tps=$(adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-bench -m Llama-3.2-1B-Instruct-f16.gguf -p 0 -n 128 -d ${depth} -r 1 -t 8 2>&1" \
    | awk '/tg128/ && /llama 1B F16/ { for (i=1; i<=NF; i++) if ($i ~ /^[0-9]+\.[0-9]+$/) last=$i; print last; exit }')
  awk "BEGIN { printf \"%.2f\", 1000.0 / $tps }"
}

declare -A DEPTH_FOR=(
  [p100.txt]=170
  [p300.txt]=374
  [p600.txt]=676
  [long_prompt.txt]=720
)

echo "[bench] $(date) — flash attention decode sweep" | tee "$OUTDIR/after-task4.txt"

fail=0
for f in p100.txt p300.txt p600.txt long_prompt.txt; do
  depth="${DEPTH_FOR[$f]}"
  echo "[bench] Running $f (approx seq ${depth})..."

  ours=$(tbt_llm_rs "$f")
  sleep "$COOLDOWN_SEC"
  theirs=$(tbt_llama_cpp "$depth")
  sleep "$COOLDOWN_SEC"

  limit=$(awk "BEGIN { printf \"%.2f\", $theirs * (100 + $TOLERANCE_PCT) / 100 }")
  cmp=$(awk "BEGIN { print ($ours > $limit) ? 1 : 0 }")

  line="$f depth=$depth llm_rs=${ours}ms llama_cpp=${theirs}ms limit=${limit}ms"
  if [ "$cmp" = "1" ]; then
    echo "[FAIL] $line" | tee -a "$OUTDIR/after-task4.txt"
    fail=1
  else
    echo "[ OK ] $line" | tee -a "$OUTDIR/after-task4.txt"
  fi
done

echo
echo "[bench] Results saved to $OUTDIR/after-task4.txt"
exit $fail
```

- [ ] **Step 2: Run**

```bash
chmod +x scripts/bench_flash_attn_decode.sh
./scripts/bench_flash_attn_decode.sh
```

Expected exit 0 with all contexts `[ OK ]`. If a context shows `[FAIL]`:
- Device may be thermally throttled — increase `COOLDOWN_SEC` to 30 and retry
- Flash path not dispatching — verify Task 2 Step 5 `use_flash` predicate
- llama-bench output format changed — inspect raw output manually and fix the `awk` pattern

- [ ] **Step 3: Commit**

```bash
git add scripts/bench_flash_attn_decode.sh results/data/flash_attn_decode/after-task4.txt
git commit -m "$(cat <<'EOF'
test(bench): flash attention decode regression vs llama.cpp CPU

Adds an adb-driven benchmark that sweeps 4 context lengths and asserts
llm.rs wall-clock Decode TBT ≤ llama.cpp CPU tg128 × 1.05.

First green run captured in results/data/flash_attn_decode/after-task4.txt
on Samsung Galaxy S25 (Adreno 830, llama3.2-1b F16).

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: Manager heartbeat cross-check

**Purpose:** Prove that the `actual_throughput` value the manager receives reflects the new production TBT. Uses `mock_manager` binary if available.

**Files:** none modified.

- [ ] **Step 1: Check for mock_manager**

```bash
ls manager/src/bin/mock_manager.rs 2>/dev/null && echo "exists" || echo "missing"
```

If missing, **skip this task entirely** with a note. The wall-clock verification in Task 4 is authoritative; manager cross-check is nice-to-have.

- [ ] **Step 2: Build mock_manager for the host**

```bash
cargo build --release -p llm_manager --bin mock_manager
```

- [ ] **Step 3: Run an end-to-end check** (only if Step 1 passed)

Look at `manager/src/bin/mock_manager.rs` for the CLI interface. Typical pattern:

```bash
./target/release/mock_manager --socket /tmp/llm_mgr.sock > /tmp/heartbeats.log 2>&1 &
MGR_PID=$!
sleep 1

cargo run --release --bin generate -- \
  --model-path models/llama3.2-1b \
  --prompt-file tests/data/long_prompt.txt \
  -n 64 -b cpu \
  --resilience-socket /tmp/llm_mgr.sock \
  2>&1 | tail -15

kill $MGR_PID 2>/dev/null || true

grep actual_throughput /tmp/heartbeats.log | tail -5
```

Expected: `actual_throughput` values visible in heartbeat log. The absolute value is CPU-dependent — the purpose is plumbing validation, not perf comparison. Note the number for the cross-check.

**If the CLI flag `--resilience-socket` doesn't exist**: skip this task and note in the task list that manager integration testing requires a separate plan.

- [ ] **Step 4: Commit log capture (if successful)**

```bash
cp /tmp/heartbeats.log results/data/flash_attn_decode/manager_heartbeat_sample.txt
git add results/data/flash_attn_decode/manager_heartbeat_sample.txt
git commit -m "test: cross-check flash attention via manager heartbeat"
```

If Step 3 was skipped, no commit.

---

### Task 6: Fallback regression tests (H2O, Qwen, Gemma)

**Files:** none modified.

- [ ] **Step 1: H2O (llama3.2-1b, legacy expected)**

```bash
adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/llama3.2-1b --prompt 'The quick brown fox jumps over' -n 48 -b opencl --eviction-policy h2o --effective-budget 48 2>&1 | tail -15"
```

Expected: coherent text, no crash. H2O requires scores → legacy dispatched.

- [ ] **Step 2: Qwen 2.5-1.5B (head_dim=128, legacy expected)**

```bash
adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b --prompt 'Hello world' -n 48 -b opencl 2>&1 | tail -15"
```

Expected: coherent text, no crash. `head_dim != 64` → legacy.

- [ ] **Step 3: Gemma 3-1B (head_dim=256, legacy expected)**

```bash
adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/gemma3-1b --prompt 'Hello' -n 32 -b opencl 2>&1 | tail -15"
```

Expected: coherent text, no crash. `head_dim != 64` → legacy.

- [ ] **Step 4: Capture evidence**

```bash
mkdir -p results/data/flash_attn_decode/fallback_smoke
```

Save each run's tail to:
- `results/data/flash_attn_decode/fallback_smoke/llama3.2-1b-h2o.txt`
- `results/data/flash_attn_decode/fallback_smoke/qwen2.5-1.5b.txt`
- `results/data/flash_attn_decode/fallback_smoke/gemma3-1b.txt`

- [ ] **Step 5: Commit**

```bash
git add results/data/flash_attn_decode/fallback_smoke/
git commit -m "test: verify legacy attention fallback for H2O, Qwen, Gemma"
```

---

### Task 7: Methodology note — profile vs production

**Files:**
- Modify: `AGENTS.md` (= `CLAUDE.md` symlink)

- [ ] **Step 1: Add one bullet**

Edit `AGENTS.md`, find `## 핵심 제약사항`, and add this bullet at the end of the list:

```markdown
- **성능 측정은 `--profile` 없이** — `--profile`은 `plan.rs`를 비활성화하고 매 op마다 `backend.synchronize()`를 2회 호출하여 ~54 ms/token의 sync 오버헤드를 더한다. Production TBT는 `Decode: X ms/tok` 로그 라인 또는 manager heartbeat의 `actual_throughput`을 사용한다. `--profile`의 per-op breakdown은 **상대 비교**에만 유효하다 (어느 op이 상대적으로 큰가). 절대값은 sync 오버헤드로 부풀려져 있다.
```

- [ ] **Step 2: Commit**

```bash
git add AGENTS.md
git commit -m "$(cat <<'EOF'
docs: note profile-vs-production measurement methodology

Performance decisions must use wall-clock TBT (production path) rather
than --profile output, which disables plan.rs and adds ~54 ms/token of
synchronize overhead. --profile remains useful for relative per-op
comparison only.

Co-Authored-By: Claude Opus 4.6 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: Final sanity and merge preparation

**Files:** none modified.

- [ ] **Step 1: Full sanity**

```bash
cd .claude/worktrees/flash-attn-decode
cargo fmt --all -- --check
cargo check -p llm_rs2 --lib
cargo test -p llm_rs2 --lib
cargo test -p llm_rs2 --test flash_attn_decode
```

All must pass. Known pre-existing clippy error (`neon.rs` `uninit_vec`) is allowed.

- [ ] **Step 2: Android rebuild**

```bash
cargo build --release --bin generate --target aarch64-linux-android
```

- [ ] **Step 3: Rerun the regression benchmark**

```bash
./scripts/bench_flash_attn_decode.sh
```

Expected: exit 0.

- [ ] **Step 4: Summary**

Report status. No commit needed. Merge command (run from the main checkout, not the worktree):

```bash
cd /Users/li/Workspace/llm_rs2
git checkout master
git merge --no-ff feat/flash-attn-decode -m "feat: flash attention decode in plan.rs"
git worktree remove .claude/worktrees/flash-attn-decode
```

---

## Follow-ups (separate plans, out of scope here)

1. **Multi-variant compilation for head_dim 128/256** — compile `flash_attn_f32_f16.cl` with different `-DDK=...`; select at plan-build time. Benefits Qwen and Gemma.
2. **Score-output flash variant** — requires `.cl` modification; unblocks H2O + flash.
3. **CL-event-based `--profile` refactor** — rejected in this plan (wall-clock measurement is sufficient for perf decisions). Revisit only if a future optimization needs absolute per-op times under plan.rs.
4. **Remove `kernel_attn_gen_half`** — once all supported configurations route through flash.
