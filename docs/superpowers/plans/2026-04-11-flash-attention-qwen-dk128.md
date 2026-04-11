# Flash Attention Decode for Qwen (head_dim=128) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Extend the flash attention decode path (currently head_dim=64 only, llama3.2-1b) to support head_dim=128 so Qwen 2.5-1.5B decode on Adreno 830 matches or beats llama.cpp CPU tg128.

**Architecture:** `flash_attn_f32_f16.cl` hardcodes DK/DV as compile-time macros because the kernel declares `float4 q_priv[DK_VEC]` / `float4 o_acc[DV_VEC]` as private arrays. Runtime DK is impossible. The fix is to compile the same source file a second time with `-DDK=128 -DDV=128`, producing a second `ocl::Program` and a second cached kernel handle. The dispatcher (`flash_attention_decode_gpu`) selects between them by `head_dim`. `plan.rs` gates flash selection on head_dim matching an available program. Everything else — strides, GQA math, dispatch shape — is unchanged because the kernel itself already handles GQA (`head_kv_idx = head_idx / gqa_ratio`).

**Tech Stack:** Rust (Cargo workspace: `llm_rs2`), OpenCL 2.0 on Adreno 830 (Snapdragon 8 Elite), Samsung Galaxy S25 (ARM64 Android). NDK r26 toolchain for cross-compile. Test infrastructure: `cargo test -p llm_rs2`, `adb shell` for device runs, `scripts/bench_flash_attn_decode*.sh` for regression gates.

---

## Scope Note

This plan covers **only head_dim=128** (Qwen 2.5-1.5B). head_dim=256 (Gemma3) is out of scope and tracked as a separate follow-up because Gemma3 requires non-trivial `plan.rs` integration work (post-FFN norm, GELU_tanh mul, QK-norm) that is independent from this change.

---

## Prerequisites (Device Setup)

**Before Task 5:**
- `adb shell ls /data/local/tmp/models/qwen2.5-1.5b` must succeed (safetensors model directory).
- A Qwen-compatible GGUF for llama-bench: `/data/local/tmp/Qwen2.5-1.5B-Instruct-f16.gguf` (f16 variant preferred to match llm.rs F16 KV).
- Existing prompts reused: `/data/local/tmp/{p100,p300,p600,long_prompt}.txt`.

If any of these are missing, Task 5 / Task 6 will report the gap and you must push the missing asset via `adb push` before continuing.

---

## File Structure

**Files to modify:**
- `engine/src/backend/opencl/mod.rs` — Add second flash program compilation, second cached kernel, kernel selector in `flash_attention_decode_gpu`, parameterized `has_flash_decode_kernel`.
- `engine/src/backend/opencl/plan.rs` — Add `flash_attn_f32_f16_program_dk128` to `LayerPlanConfig` / `FullPlanConfig`, update `build_flash_attention_step` to pick program by head_dim, update `use_flash` gate.
- `engine/src/models/transformer.rs` — Thread new `flash_attn_f32_f16_program_dk128` from backend into `FullPlanConfig`.

**Files to create:**
- `engine/tests/flash_attn_decode_dk128.rs` — Host integration test for Qwen-shape dispatch + self-consistency.
- `scripts/bench_flash_attn_decode_qwen.sh` — Qwen regression gate vs llama.cpp CPU.
- `results/data/flash_attn_decode/task5_qwen_device.txt` — Committed device verification log.
- `results/data/flash_attn_decode/qwen_after_c1.txt` — Committed benchmark results.

**Files NOT to modify:**
- `engine/kernels/flash_attn_f32_f16.cl` — The kernel source is already parameterized by DK/DV macros. No `.cl` edits needed.

---

## Naming Decisions (Lock These In)

Current code has a single pair:
- `flash_attn_f32_f16_program: Option<Program>` (implicitly DK=64)
- `kernel_flash_attn_f32_f16_q1: Option<CoreKernel>` (implicitly DK=64)

After this change, **rename to explicit variants**:
- `flash_attn_f32_f16_program_dk64: Option<Program>` (was `flash_attn_f32_f16_program`)
- `flash_attn_f32_f16_program_dk128: Option<Program>` (new)
- `kernel_flash_attn_f32_f16_q1_dk64: Option<CoreKernel>` (was `kernel_flash_attn_f32_f16_q1`)
- `kernel_flash_attn_f32_f16_q1_dk128: Option<CoreKernel>` (new)

The **prefill** kernels (`kernel_flash_attn_f32`, `kernel_flash_attn_f32_f16`) remain unchanged — they're not decode-critical and only head_dim=64 prefill is used today. Follow-up plans may parameterize them if needed.

`has_flash_decode_kernel()` gains a `head_dim: usize` parameter. Call sites update accordingly.

---

## Task 1: Host test for head_dim=128 self-consistency (TDD Red)

**Files:**
- Create: `engine/tests/flash_attn_decode_dk128.rs`

**Why TDD:** The test must fail first (backend has no dk=128 program yet), then Task 2 adds the backend plumbing that makes it pass.

- [ ] **Step 1: Create the failing test file**

```rust
//! Host integration test for flash attention decode with head_dim=128 (Qwen shape).
//!
//! This test verifies that `flash_attention_decode_gpu` dispatches successfully
//! for head_dim=128 and produces deterministic, non-zero output. Numerical
//! correctness against a CPU reference is covered by on-device Qwen inference
//! sanity in Task 5.

#![cfg(feature = "opencl")]

use llm_rs2::backend::opencl::OpenCLBackend;

/// Qwen 2.5-1.5B shape: 12 Q heads, 2 KV heads (GQA ratio = 6), head_dim=128.
/// Runs flash_attention_decode_gpu twice with identical inputs and asserts
/// bit-identical output. On hosts where the DK=128 program fails to compile
/// (e.g. macOS Apple OpenCL), the test skips cleanly — strong verification
/// lives on device in Task 5.
#[test]
fn flash_attn_decode_dk128_self_consistent() {
    use llm_rs2::backend::opencl::memory::OpenCLMemory;
    use llm_rs2::core::backend::Backend;
    use llm_rs2::core::memory::Memory;
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

    // Qwen 2.5-1.5B shape
    let n_heads_q = 12usize;
    let n_heads_kv = 2usize;
    let head_dim = 128usize;
    let cache_seq_len = 48usize;
    let capacity = 128usize;

    let memory: Arc<dyn Memory> = Arc::new(OpenCLMemory::new(
        ocl.context.clone(),
        ocl.queue.clone(),
        true,
    ));

    // Deterministic inputs
    let q_data: Vec<f32> = (0..n_heads_q * head_dim)
        .map(|i| ((i as f32) * 0.005).sin())
        .collect();

    let kv_total = n_heads_kv * capacity * head_dim;
    let mut k_data = vec![0u16; kv_total];
    let mut v_data = vec![0u16; kv_total];
    for h in 0..n_heads_kv {
        for p in 0..cache_seq_len {
            for d in 0..head_dim {
                let idx = h * capacity * head_dim + p * head_dim + d;
                let seed = (h * 17 + p * 3 + d) as f32;
                k_data[idx] = half::f16::from_f32((seed * 0.002).cos()).to_bits();
                v_data[idx] = half::f16::from_f32((seed * 0.004).sin()).to_bits();
            }
        }
    }

    let q = upload_f32(&ocl_arc, &*memory, &q_data, vec![1, 1, n_heads_q, head_dim]);
    let k = upload_f16(
        &ocl_arc,
        &*memory,
        &k_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );
    let v = upload_f16(
        &ocl_arc,
        &*memory,
        &v_data,
        vec![1, n_heads_kv, capacity, head_dim],
    );

    let zero = vec![0.0f32; n_heads_q * head_dim];
    let mut out_a = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);
    let mut out_b = upload_f32(&ocl_arc, &*memory, &zero, vec![1, 1, n_heads_q, head_dim]);

    let ok_a = ocl
        .flash_attention_decode_gpu(
            &q,
            &k,
            &v,
            &mut out_a,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
        )
        .expect("flash call a");
    if !ok_a {
        eprintln!(
            "Skipping: flash DK=128 unavailable on this host \
             (kernel compile likely failed or backend returned Ok(false))"
        );
        return;
    }
    ocl_arc.synchronize().unwrap();

    let ok_b = ocl
        .flash_attention_decode_gpu(
            &q,
            &k,
            &v,
            &mut out_b,
            n_heads_q,
            n_heads_kv,
            head_dim,
            cache_seq_len,
        )
        .expect("flash call b");
    assert!(
        ok_b,
        "flash must dispatch for second call if first call dispatched"
    );
    ocl_arc.synchronize().unwrap();

    // Readback via Backend::read_buffer to avoid ARM UMA stale-cache pitfalls.
    let out_len_bytes = n_heads_q * head_dim * std::mem::size_of::<f32>();
    let mut raw_a = vec![0u8; out_len_bytes];
    let mut raw_b = vec![0u8; out_len_bytes];
    ocl_arc.read_buffer(&out_a, &mut raw_a).unwrap();
    ocl_arc.read_buffer(&out_b, &mut raw_b).unwrap();

    let a: &[f32] = unsafe {
        std::slice::from_raw_parts(raw_a.as_ptr() as *const f32, n_heads_q * head_dim)
    };
    let b: &[f32] = unsafe {
        std::slice::from_raw_parts(raw_b.as_ptr() as *const f32, n_heads_q * head_dim)
    };
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!(
            (x - y).abs() < 1e-6,
            "nondeterministic flash DK=128 output at [{i}]: {x} vs {y}"
        );
    }
    assert!(
        a.iter().any(|&v| v.abs() > 1e-6),
        "flash DK=128 produced all-zero output — likely a dispatch no-op"
    );
}

fn upload_f32(
    backend: &std::sync::Arc<dyn llm_rs2::core::backend::Backend>,
    memory: &dyn llm_rs2::core::memory::Memory,
    data: &[f32],
    shape: Vec<usize>,
) -> llm_rs2::core::tensor::Tensor {
    let buf = memory
        .alloc(data.len() * 4, llm_rs2::core::buffer::DType::F32)
        .unwrap();
    let mut t = llm_rs2::core::tensor::Tensor::new(
        llm_rs2::core::shape::Shape::new(shape),
        buf,
        backend.clone(),
    );
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 4) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}

fn upload_f16(
    backend: &std::sync::Arc<dyn llm_rs2::core::backend::Backend>,
    memory: &dyn llm_rs2::core::memory::Memory,
    data: &[u16],
    shape: Vec<usize>,
) -> llm_rs2::core::tensor::Tensor {
    let buf = memory
        .alloc(data.len() * 2, llm_rs2::core::buffer::DType::F16)
        .unwrap();
    let mut t = llm_rs2::core::tensor::Tensor::new(
        llm_rs2::core::shape::Shape::new(shape),
        buf,
        backend.clone(),
    );
    let bytes =
        unsafe { std::slice::from_raw_parts(data.as_ptr() as *const u8, data.len() * 2) };
    backend.write_buffer(&mut t, bytes).unwrap();
    t
}
```

- [ ] **Step 2: Run the test and verify it fails (or skips) correctly**

Run: `cargo test -p llm_rs2 --test flash_attn_decode_dk128 flash_attn_decode_dk128_self_consistent`

Expected on a host WITH OpenCL:
- Either skips ("Skipping: flash DK=128 unavailable on this host...") because the current backend's `flash_attention_decode_gpu` returns `Ok(false)` for `head_dim != 64`
- OR: the compile step fails cleanly and the test skips

Expected on a host WITHOUT OpenCL: skips at `OpenCLBackend::new()`.

Either skip path counts as "RED" for TDD purposes — the test does NOT assert deterministic output because no real dispatch happens yet. Task 2 will turn this from skip → pass.

- [ ] **Step 3: Commit**

```bash
git add engine/tests/flash_attn_decode_dk128.rs
git commit -m "test(flash-attn): add failing dk=128 self-consistency test for Qwen shape"
```

---

## Task 2: Backend — Compile DK=128 variant + selector

**Files:**
- Modify: `engine/src/backend/opencl/mod.rs` (kernel cache fields, program compile, dispatcher, accessor)

**Context for the implementer:**
- Current `flash_attn_defines` is `"-DDK=64 -DDV=64 -DBLOCK_M=64 -DBLOCK_N=32"` at `mod.rs:508`.
- Current single program field: `flash_attn_f32_f16_program: Option<Program>` at `mod.rs:157`.
- Current single kernel field: `kernel_flash_attn_f32_f16_q1: Option<CoreKernel>` at `mod.rs:116`.
- `flash_attention_decode_gpu` (starts `mod.rs:1358`) currently returns `Ok(false)` when `head_dim != 64` at `mod.rs:1371-1373`.
- `has_flash_decode_kernel()` at `mod.rs:3182` is a parameterless accessor used by plan.rs gating and host tests.
- The prefill kernel `kernel_flash_attn_f32_f16` and the F32-Q/F32-KV prefill kernel `kernel_flash_attn_f32` remain DK=64 — this plan does not touch prefill.

- [ ] **Step 1: Rename existing fields to `_dk64` variants**

In `mod.rs`, update the `KernelCache` struct (around line 112-116):

Replace:
```rust
    kernel_flash_attn_f32: Option<CoreKernel>,
    kernel_flash_attn_f32_f16: Option<CoreKernel>,
    /// Decode-specialized flash attention (single-query, online softmax).
    /// Same program as kernel_flash_attn_f32_f16 (Q=F32, KV=F16, DK=DV=64).
    kernel_flash_attn_f32_f16_q1: Option<CoreKernel>,
```

With:
```rust
    kernel_flash_attn_f32: Option<CoreKernel>,
    kernel_flash_attn_f32_f16: Option<CoreKernel>,
    /// Decode-specialized flash attention, head_dim=64 variant
    /// (Q=F32, KV=F16, compiled with -DDK=64 -DDV=64).
    kernel_flash_attn_f32_f16_q1_dk64: Option<CoreKernel>,
    /// Decode-specialized flash attention, head_dim=128 variant
    /// (Q=F32, KV=F16, compiled with -DDK=128 -DDV=128).
    kernel_flash_attn_f32_f16_q1_dk128: Option<CoreKernel>,
```

In the `OpenCLBackend` struct (around line 156-157), replace:
```rust
    pub flash_attn_f32_program: Option<Program>,
    pub flash_attn_f32_f16_program: Option<Program>,
```

With:
```rust
    pub flash_attn_f32_program: Option<Program>,
    /// Flash attention F32-Q / F16-KV program, head_dim=64 variant.
    /// Used by prefill (`flash_attn_f32_f16` kernel) and decode
    /// (`flash_attn_f32_f16_q1` kernel) at head_dim=64.
    pub flash_attn_f32_f16_program_dk64: Option<Program>,
    /// Flash attention F32-Q / F16-KV program, head_dim=128 variant.
    /// Decode-only (no prefill dispatcher for this DK). Dispatched
    /// by `flash_attention_decode_gpu` for models with head_dim=128
    /// (e.g. Qwen 2.5-1.5B).
    pub flash_attn_f32_f16_program_dk128: Option<Program>,
```

- [ ] **Step 2: Compile the DK=128 program alongside DK=64**

In `mod.rs`, replace the current DK=64 compile block (lines 506-546, the block starting with `// Flash attention kernels — compiled with head_dim-specific defines` and ending with the match on `flash_attn_f32_f16_src` build result) with:

```rust
        // Flash attention kernels — compiled with head_dim-specific defines.
        // Each DK variant is a separate program because DK is a compile-time
        // constant (q_priv[DK_VEC] / o_acc[DV_VEC] are private arrays).
        let flash_attn_src = include_str!("../../../kernels/flash_attn_f32_f16.cl");
        let flash_attn_f32_src = include_str!("../../../kernels/flash_attn_f32.cl");

        // DK=64 (Llama 3.2 1B, other head_dim=64 models)
        let flash_attn_dk64_defines = "-DDK=64 -DDV=64 -DBLOCK_M=64 -DBLOCK_N=32";
        let flash_attn_f32_opts_dk64 = format!("{} {}", cl_opts, flash_attn_dk64_defines);

        let flash_attn_f32_program = match Program::builder()
            .devices(device)
            .src(flash_attn_f32_src)
            .cmplr_opt(&flash_attn_f32_opts_dk64)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32.cl compiled (BLOCK_M=64, BLOCK_N=32, DK=64)");
                Some(p)
            }
            Err(e) => {
                log::warn!("flash_attn_f32.cl failed: {}. GPU prefill F32 disabled.", e);
                None
            }
        };

        let flash_attn_f32_f16_program_dk64 = match Program::builder()
            .devices(device)
            .src(flash_attn_src)
            .cmplr_opt(&flash_attn_f32_opts_dk64)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=64)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "flash_attn_f32_f16.cl DK=64 failed: {}. GPU flash F16 KV disabled for DK=64.",
                    e
                );
                None
            }
        };

        // DK=128 (Qwen 2.5-1.5B). Same source file, different macros.
        // Decode-only; prefill at DK=128 is not routed through a kernel yet.
        let flash_attn_dk128_defines = "-DDK=128 -DDV=128 -DBLOCK_M=64 -DBLOCK_N=32";
        let flash_attn_f32_opts_dk128 = format!("{} {}", cl_opts, flash_attn_dk128_defines);

        let flash_attn_f32_f16_program_dk128 = match Program::builder()
            .devices(device)
            .src(flash_attn_src)
            .cmplr_opt(&flash_attn_f32_opts_dk128)
            .build(&context)
        {
            Ok(p) => {
                log::info!("flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=128)");
                Some(p)
            }
            Err(e) => {
                log::warn!(
                    "flash_attn_f32_f16.cl DK=128 failed: {}. GPU flash F16 KV disabled for DK=128 (Qwen fallback).",
                    e
                );
                None
            }
        };
```

- [ ] **Step 3: Update the kernel cache initializer**

In `mod.rs`, find the cache init block around lines 700-705:

```rust
            kernel_flash_attn_f32_f16: flash_attn_f32_f16_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16").ok()),
            kernel_flash_attn_f32_f16_q1: flash_attn_f32_f16_program
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16_q1").ok()),
```

Replace with:
```rust
            kernel_flash_attn_f32_f16: flash_attn_f32_f16_program_dk64
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16").ok()),
            kernel_flash_attn_f32_f16_q1_dk64: flash_attn_f32_f16_program_dk64
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16_q1").ok()),
            kernel_flash_attn_f32_f16_q1_dk128: flash_attn_f32_f16_program_dk128
                .as_ref()
                .and_then(|p| ocl::core::create_kernel(p, "flash_attn_f32_f16_q1").ok()),
```

- [ ] **Step 4: Update `Ok(Self { ... })` backend init**

In `mod.rs`, find the struct init block around lines 790-791:

```rust
            flash_attn_f32_program,
            flash_attn_f32_f16_program,
```

Replace with:
```rust
            flash_attn_f32_program,
            flash_attn_f32_f16_program_dk64,
            flash_attn_f32_f16_program_dk128,
```

- [ ] **Step 5: Update the decode dispatcher to select by head_dim**

In `mod.rs`, find `flash_attention_decode_gpu` (around line 1359). Replace the early-return head_dim gate and kernel lookup (lines 1370-1390):

```rust
        // Head dim must match the DK/DV compile-time constant in flash_attn_f32_f16.cl.
        if head_dim != 64 {
            return Ok(false);
        }
        // Only F16 KV on HeadMajor GPU buffer is supported in this prototype.
        if k_cache.dtype() != DType::F16 {
            return Ok(false);
        }
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == n_heads_kv && k_shape[1] != k_shape[2];
        if !is_head_major {
            return Ok(false);
        }
        let kv_capacity = k_shape[2];

        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match &kernels.kernel_flash_attn_f32_f16_q1 {
            Some(k) => k,
            None => return Ok(false),
        };
```

With:
```rust
        // Only F16 KV on HeadMajor GPU buffer is supported.
        if k_cache.dtype() != DType::F16 {
            return Ok(false);
        }
        let k_shape = k_cache.shape().dims();
        let is_head_major =
            k_shape.len() >= 3 && k_shape[1] == n_heads_kv && k_shape[1] != k_shape[2];
        if !is_head_major {
            return Ok(false);
        }
        let kv_capacity = k_shape[2];

        // Head dim must match the DK/DV compile-time constant of one of the
        // compiled flash attention programs. Add new variants here when
        // adding a new head_dim (remember to compile the program in
        // `OpenCLBackend::new` and cache the kernel in `KernelCache`).
        let kernels = unsafe { &*self.kernels.get() };
        let kernel = match head_dim {
            64 => match &kernels.kernel_flash_attn_f32_f16_q1_dk64 {
                Some(k) => k,
                None => return Ok(false),
            },
            128 => match &kernels.kernel_flash_attn_f32_f16_q1_dk128 {
                Some(k) => k,
                None => return Ok(false),
            },
            _ => return Ok(false),
        };
```

- [ ] **Step 6: Parameterize `has_flash_decode_kernel`**

In `mod.rs`, find `has_flash_decode_kernel` (around line 3182):

```rust
    /// Returns true if the flash attention decode kernel (flash_attn_f32_f16_q1)
    /// was successfully created at backend init time. Used by plan.rs to gate
    /// the StandardFlash attention variant, and by host tests to decide whether
    /// to exercise the flash path.
    pub fn has_flash_decode_kernel(&self) -> bool {
        let kernels = unsafe { &*self.kernels.get() };
        kernels.kernel_flash_attn_f32_f16_q1.is_some()
    }
```

Replace with:
```rust
    /// Returns true if the flash attention decode kernel is available for the
    /// given head_dim. Used by plan.rs to gate the `StandardFlash` attention
    /// variant, and by host tests to decide whether to exercise the flash path.
    /// Supported head_dim values match the DK variants compiled in `new()`.
    pub fn has_flash_decode_kernel(&self, head_dim: usize) -> bool {
        let kernels = unsafe { &*self.kernels.get() };
        match head_dim {
            64 => kernels.kernel_flash_attn_f32_f16_q1_dk64.is_some(),
            128 => kernels.kernel_flash_attn_f32_f16_q1_dk128.is_some(),
            _ => false,
        }
    }
```

- [ ] **Step 7: Fix any remaining references to the old field names**

Run: `cargo check -p llm_rs2`

Expected: zero errors. If there are errors, they will be references to `flash_attn_f32_f16_program` (no suffix) or `kernel_flash_attn_f32_f16_q1` (no suffix), or `has_flash_decode_kernel()` (no argument). Fix each call site:
- In `transformer.rs`, any `has_flash_decode_kernel()` call becomes `has_flash_decode_kernel(head_dim)` where `head_dim` is already in scope.
- Any lookup of `flash_attn_f32_f16_program` without suffix becomes `flash_attn_f32_f16_program_dk64` for the existing decode path (Task 3 will add the dk128 wiring in plan.rs; for this task, just make the codebase compile).

Use grep to find references:
```bash
cargo check -p llm_rs2 2>&1 | head -60
```

Resolve each reported undefined field or missing arg.

- [ ] **Step 8: Run the Task 1 test — expect PASS (or skip cleanly) on dev host**

Run: `cargo test -p llm_rs2 --test flash_attn_decode_dk128 flash_attn_decode_dk128_self_consistent -- --nocapture`

Expected on macOS dev host: skip with message like "Skipping: flash DK=128 unavailable on this host (kernel compile likely failed or backend returned Ok(false))" because Apple OpenCL often fails to compile the flash kernel. Non-blocking — Task 5 exercises it on Adreno.

Expected on Linux host with Intel/AMD GPU: passes if DK=128 compiles; otherwise skips.

- [ ] **Step 9: Run existing flash dk=64 test — must still pass**

Run: `cargo test -p llm_rs2 --test flash_attn_decode flash_attn_decode_self_consistent -- --nocapture`

Expected: pass or same skip behavior as before. **This test is the regression gate for the rename** — if it breaks, the rename broke the dk64 path.

- [ ] **Step 10: Run the full sanity check**

Run: `./.agent/skills/developing/scripts/sanity_check.sh`

Expected: fmt ok, clippy ok, all existing tests pass.

- [ ] **Step 11: Commit**

```bash
git add engine/src/backend/opencl/mod.rs
git commit -m "feat(opencl): compile flash_attn_f32_f16.cl with DK=128 for Qwen decode

Adds a second program compilation of the flash attention F32-Q / F16-KV
kernel with -DDK=128 -DDV=128 alongside the existing DK=64 variant.
The decode dispatcher flash_attention_decode_gpu selects between them
by head_dim. Enables Qwen 2.5-1.5B (head_dim=128) to use flash decode.

Renames flash_attn_f32_f16_program → *_dk64 and introduces *_dk128.
Likewise for kernel_flash_attn_f32_f16_q1. has_flash_decode_kernel
gains a head_dim parameter. Prefill kernels unchanged."
```

---

## Task 3: Wire dk128 program into plan.rs

**Files:**
- Modify: `engine/src/backend/opencl/plan.rs` (`LayerPlanConfig`, `FullPlanConfig`, `build_flash_attention_step`, `use_flash` gate, config propagation in `build_full_plan`)

**Context for the implementer:**
- `LayerPlanConfig` currently has `flash_attn_f32_f16_program: Option<&'a ocl::Program>` at `plan.rs:601`.
- `build_flash_attention_step` at `plan.rs:712` `.expect()`s that field.
- `use_flash` gate at `plan.rs:1029-1034` requires head_dim=64.
- `FullPlanConfig` has the parallel field at `plan.rs:1258`.
- `build_full_plan` propagates from FullPlanConfig to LayerPlanConfig at `plan.rs:1371`.
- There's also a test helper that sets `flash_attn_f32_f16_program: None` at `plan.rs:1898`.

- [ ] **Step 1: Update `LayerPlanConfig` with both dk64 and dk128 program fields**

In `plan.rs`, find `LayerPlanConfig` (around line 598-601):

```rust
    /// The flash attention F32 Q / F16 KV program handle, if compiled.
    /// Used to create `flash_attn_f32_f16_q1` at plan-build time when
    /// runtime preconditions hold. `None` forces the legacy path.
    pub flash_attn_f32_f16_program: Option<&'a ocl::Program>,
```

Replace with:
```rust
    /// Flash attention F32-Q / F16-KV program handle for head_dim=64.
    /// Used to create `flash_attn_f32_f16_q1` at plan-build time when
    /// runtime preconditions hold and the layer's head_dim is 64.
    /// `None` forces the legacy path for head_dim=64 models.
    pub flash_attn_f32_f16_program_dk64: Option<&'a ocl::Program>,
    /// Flash attention F32-Q / F16-KV program handle for head_dim=128.
    /// Used for Qwen 2.5-1.5B and other head_dim=128 models.
    /// `None` forces the legacy path for head_dim=128 models.
    pub flash_attn_f32_f16_program_dk128: Option<&'a ocl::Program>,
```

- [ ] **Step 2: Update `build_flash_attention_step` to select by head_dim**

In `plan.rs`, find `build_flash_attention_step` (around line 712-718):

```rust
fn build_flash_attention_step(config: &LayerPlanConfig) -> Result<AttentionVariant> {
    let program = config
        .flash_attn_f32_f16_program
        .expect("caller must verify flash_attn_f32_f16_program.is_some()");

    let kernel = ocl::core::create_kernel(program, "flash_attn_f32_f16_q1")
        .context("create flash_attn_f32_f16_q1 for plan")?;
```

Replace with:
```rust
fn build_flash_attention_step(config: &LayerPlanConfig) -> Result<AttentionVariant> {
    // Pick the program matching this layer's head_dim. The caller
    // (`use_flash` gate) guarantees the matching program is `Some`.
    let program = match config.head_dim {
        64 => config.flash_attn_f32_f16_program_dk64,
        128 => config.flash_attn_f32_f16_program_dk128,
        _ => None,
    }
    .expect("caller must verify flash program is Some for this head_dim");

    let kernel = ocl::core::create_kernel(program, "flash_attn_f32_f16_q1")
        .context("create flash_attn_f32_f16_q1 for plan")?;
```

- [ ] **Step 3: Update the `use_flash` gate to accept both dk64 and dk128**

In `plan.rs`, find the gate (around lines 1029-1034):

```rust
    let is_head_major = config.kv_pos_stride == config.head_dim as i32
        && config.kv_head_stride == (config.kv_capacity * config.head_dim) as i32;
    let use_flash = config.head_dim == 64
        && is_head_major
        && config.flash_attn_f32_f16_program.is_some()
        && !config.needs_attention_scores;
```

Replace with:
```rust
    let is_head_major = config.kv_pos_stride == config.head_dim as i32
        && config.kv_head_stride == (config.kv_capacity * config.head_dim) as i32;
    // Flash attention is gated per head_dim because each DK variant is a
    // separate compiled program. Add a new arm here when adding DK=256
    // (Gemma3) etc. `needs_attention_scores` forces legacy because no
    // flash variant emits scores (see C2 follow-up plan).
    let flash_program_available = match config.head_dim {
        64 => config.flash_attn_f32_f16_program_dk64.is_some(),
        128 => config.flash_attn_f32_f16_program_dk128.is_some(),
        _ => false,
    };
    let use_flash =
        is_head_major && flash_program_available && !config.needs_attention_scores;
```

- [ ] **Step 4: Update `FullPlanConfig` with the same fields**

In `plan.rs`, find `FullPlanConfig` (around lines 1255-1258):

```rust
    /// Optional flash attention program (`flash_attn_f32_f16`). When `Some`,
    /// the layer builder may select `AttentionVariant::StandardFlash` if all
    /// other preconditions hold.
    pub flash_attn_f32_f16_program: Option<&'a ocl::Program>,
```

Replace with:
```rust
    /// Flash attention program for head_dim=64. When `Some`, the layer
    /// builder may select `AttentionVariant::StandardFlash` for layers
    /// with head_dim=64.
    pub flash_attn_f32_f16_program_dk64: Option<&'a ocl::Program>,
    /// Flash attention program for head_dim=128. When `Some`, the layer
    /// builder may select `AttentionVariant::StandardFlash` for layers
    /// with head_dim=128 (e.g. Qwen 2.5-1.5B).
    pub flash_attn_f32_f16_program_dk128: Option<&'a ocl::Program>,
```

- [ ] **Step 5: Update `build_full_plan` to propagate both fields**

In `plan.rs`, find the `LayerPlanConfig { ... }` construction inside `build_full_plan` (around line 1371):

```rust
            flash_attn_f32_f16_program: config.flash_attn_f32_f16_program,
            needs_attention_scores: config.needs_attention_scores,
```

Replace with:
```rust
            flash_attn_f32_f16_program_dk64: config.flash_attn_f32_f16_program_dk64,
            flash_attn_f32_f16_program_dk128: config.flash_attn_f32_f16_program_dk128,
            needs_attention_scores: config.needs_attention_scores,
```

- [ ] **Step 6: Update the test helper that sets the old field to `None`**

In `plan.rs`, find the test helper/fixture (around line 1898). It should look like:

```rust
            flash_attn_f32_f16_program: None,
```

Replace with:
```rust
            flash_attn_f32_f16_program_dk64: None,
            flash_attn_f32_f16_program_dk128: None,
```

- [ ] **Step 7: Verify compilation**

Run: `cargo check -p llm_rs2`

Expected: zero errors. If an error mentions `flash_attn_f32_f16_program` without suffix in `transformer.rs`, leave it — Task 4 will fix it.

If there's a hard error in `plan.rs` test code, fix it in-place by propagating `None` to both fields.

- [ ] **Step 8: Run plan.rs unit tests**

Run: `cargo test -p llm_rs2 --lib backend::opencl::plan`

Expected: all plan.rs unit tests pass. They should exercise the non-flash path because test helpers set both program fields to `None`.

- [ ] **Step 9: Commit**

```bash
git add engine/src/backend/opencl/plan.rs
git commit -m "feat(opencl): plan.rs accepts flash_attn dk64 and dk128 programs

Adds flash_attn_f32_f16_program_dk128 alongside the existing dk64 field
in LayerPlanConfig and FullPlanConfig. build_flash_attention_step
selects the matching program by head_dim. use_flash gate accepts
head_dim ∈ {64, 128} with respective program availability."
```

---

## Task 4: Transformer.rs — Thread dk128 program through FullPlanConfig

**Files:**
- Modify: `engine/src/models/transformer.rs` (`build_plan` or equivalent, lines around 1504-1542)

**Context for the implementer:**
- `transformer.rs:1509` currently has: `flash_attn_f32_f16_program: ocl_backend.flash_attn_f32_f16_program.as_ref(),`
- After Task 2, the backend field is renamed `flash_attn_f32_f16_program_dk64`, so this line won't compile. Also need to add the dk128 line.

- [ ] **Step 1: Update the FullPlanConfig construction**

In `transformer.rs`, find the line:

```rust
            flash_attn_f32_f16_program: ocl_backend.flash_attn_f32_f16_program.as_ref(),
```

Replace with:
```rust
            flash_attn_f32_f16_program_dk64: ocl_backend.flash_attn_f32_f16_program_dk64.as_ref(),
            flash_attn_f32_f16_program_dk128: ocl_backend.flash_attn_f32_f16_program_dk128.as_ref(),
```

- [ ] **Step 2: Verify compilation**

Run: `cargo check -p llm_rs2`

Expected: zero errors.

- [ ] **Step 3: Run all existing integration tests**

Run: `cargo test -p llm_rs2`

Expected: all pre-existing tests pass. The dk128 host test still skips cleanly on dev host (Task 5 verifies it on device). The existing dk64 test must still pass / skip as before.

- [ ] **Step 4: Run the full sanity check**

Run: `./.agent/skills/developing/scripts/sanity_check.sh`

Expected: fmt ok, clippy ok, all tests pass.

- [ ] **Step 5: Commit**

```bash
git add engine/src/models/transformer.rs
git commit -m "feat(transformer): pass flash_attn dk64 and dk128 programs to plan config

Wires the new dk128 flash attention program from OpenCLBackend through
to FullPlanConfig so Qwen (head_dim=128) can take the flash decode path."
```

---

## Task 5: On-device Qwen verification

**Files:**
- Create: `results/data/flash_attn_decode/task5_qwen_device.txt` (committed verification log)

**Context for the implementer:**
- This task is **device-only**; you need a connected Samsung Galaxy S25 via adb.
- Qwen safetensors must already be at `/data/local/tmp/models/qwen2.5-1.5b`. Check with `adb shell ls /data/local/tmp/models/qwen2.5-1.5b`.
- Requires an Android cross-build: `source android.source && cargo build --target aarch64-linux-android --release -p llm_rs2 --bin generate`.
- Push binary with `adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate`.
- Use `prompt_file=/data/local/tmp/p300.txt` (374 tokens depth) to exercise the flash path with a realistic context.

- [ ] **Step 1: Preflight — verify device assets**

Run:
```bash
adb shell 'ls /data/local/tmp/models/qwen2.5-1.5b/config.json'
adb shell 'ls /data/local/tmp/p300.txt'
```

Expected both: file path echoed back. If either is missing, push them first:
```bash
adb push models/qwen2.5-1.5b /data/local/tmp/models/qwen2.5-1.5b
# p300.txt should already exist from the llama3.2-1b work
```

- [ ] **Step 2: Cross-compile and deploy the updated binary**

Run:
```bash
source android.source
cargo build --target aarch64-linux-android --release -p llm_rs2 --bin generate
adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate
```

Expected: build succeeds, push succeeds (`~20 MB pushed`).

- [ ] **Step 3: Run Qwen decode and capture log**

Run:
```bash
mkdir -p results/data/flash_attn_decode
adb shell '
  cd /data/local/tmp &&
  RUST_LOG=info ./generate \
    --model-path /data/local/tmp/models/qwen2.5-1.5b \
    --prompt-file /data/local/tmp/p300.txt \
    -n 128 -b opencl 2>&1
' | tee results/data/flash_attn_decode/task5_qwen_device.txt
```

Expected log lines to appear:
- `OpenCL Device: Adreno (TM) 830` (or similar)
- `flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=64)`
- `flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=128)` ← **CRITICAL: confirms DK=128 compiled**
- `GPU kernel plan built (28 layers, capacity=...)` ← **CRITICAL: confirms plan.rs path active for Qwen**
- `Decode: <NN>.<NN> ms/tok` (single line summary)

- [ ] **Step 4: Grep the log for all critical assertions**

Run:
```bash
grep -E "DK=128|DK=64|GPU kernel plan built|Decode:" results/data/flash_attn_decode/task5_qwen_device.txt
```

Expected: all four lines present. If `DK=128` log line is missing, the kernel failed to compile on Adreno (register spill / compile error). Read the warning line `flash_attn_f32_f16.cl DK=128 failed: ...` and capture the full error — this is a blocker that needs investigation before proceeding.

If `GPU kernel plan built` appears but decode still looks slow, confirm the flash path is actually taken by adding a temporary `log::debug!("flash variant selected for layer {i}")` inside `build_flash_attention_step` (not required for the test to pass, only for diagnosis).

- [ ] **Step 5: Output sanity — not all whitespace or gibberish**

Grep the generated tokens (printed by the binary after prompt processing):

```bash
grep -A 30 "Generating" results/data/flash_attn_decode/task5_qwen_device.txt | head -40
```

Expected: human-readable English continuation of the p300 prompt. Definitely not all spaces, not all the same token, not an all-zero decoding. If garbled, the flash kernel is producing wrong numerics for DK=128 — this is a blocker and you must investigate (likely a kernel param binding bug or a register corruption issue for DK=128 on Adreno).

- [ ] **Step 6: Commit the verification log**

```bash
git add results/data/flash_attn_decode/task5_qwen_device.txt
git commit -m "test(device): verify Qwen flash DK=128 path on Adreno 830

Captures the first on-device Qwen run with the DK=128 flash program
enabled. Confirms:
- flash_attn_f32_f16.cl compiles at DK=128
- GPU kernel plan built for all 28 Qwen layers
- Generated tokens are sane English continuation of p300 prompt"
```

---

## Task 6: Benchmark — Qwen flash decode vs llama.cpp CPU

**Files:**
- Create: `scripts/bench_flash_attn_decode_qwen.sh` (executable)
- Create: `results/data/flash_attn_decode/qwen_after_c1.txt` (committed benchmark log)

**Context for the implementer:**
- Mirrors `scripts/bench_flash_attn_decode.sh` pattern but targets Qwen. Keep the two scripts in sync.
- The llama.cpp Qwen GGUF must be at `/data/local/tmp/Qwen2.5-1.5B-Instruct-f16.gguf`. Check with `adb shell ls /data/local/tmp/Qwen2.5-1.5B-Instruct-f16.gguf`. If missing, skip the llama-bench compare and just record llm.rs numbers (the script should still succeed-exit but log a warning).
- Tolerance: llm.rs TBT ≤ llama.cpp TBT × 1.05 (5%, same as llama3.2-1b gate).
- Legacy Qwen baseline was ~62 ms/tok. Expected post-flash: 55-60 ms/tok.

- [ ] **Step 1: Create the benchmark script**

Create `scripts/bench_flash_attn_decode_qwen.sh`:

```bash
#!/usr/bin/env bash
# Flash attention decode benchmark for Qwen 2.5-1.5B (head_dim=128) on
# Adreno 830, vs llama.cpp CPU tg128 for the same GGUF.
#
# Keep in sync with scripts/bench_flash_attn_decode.sh. Only the model
# path / GGUF file / expected baseline differ.
#
# Requirements on device:
#   /data/local/tmp/generate (our binary)
#   /data/local/tmp/models/qwen2.5-1.5b
#   /data/local/tmp/{p100,p300,p600,long_prompt}.txt
#   /data/local/tmp/Qwen2.5-1.5B-Instruct-f16.gguf
#   /data/local/tmp/llama-bench + libllama.so + libggml*.so
#
# Exit code 0 on success; 1 on any per-context failure.
# If the Qwen GGUF is missing, the script records llm.rs TBT only and
# exits 0 with a warning (no regression gate in that mode).

set -euo pipefail

# Force byte-level locale for multibyte awk matches (llama-bench's ±).
export LC_ALL=C

# Preflight: require adb and a connected device.
command -v adb >/dev/null 2>&1 || {
  echo "[bench-qwen] adb not on PATH" >&2
  exit 2
}
adb get-state 1>/dev/null 2>&1 || {
  echo "[bench-qwen] no device connected (adb get-state failed)" >&2
  exit 2
}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
OUTDIR="$REPO_ROOT/results/data/flash_attn_decode"
mkdir -p "$OUTDIR"

TOLERANCE_PCT="${TOLERANCE_PCT:-5}"
COOLDOWN_SEC="${COOLDOWN_SEC:-30}"

# Detect whether the Qwen GGUF is present for llama-bench comparison.
HAS_LLAMA_BENCH=1
if ! adb shell 'ls /data/local/tmp/Qwen2.5-1.5B-Instruct-f16.gguf' >/dev/null 2>&1; then
  HAS_LLAMA_BENCH=0
  echo "[bench-qwen] WARNING: Qwen GGUF not on device; recording llm.rs numbers only"
fi

tbt_llm_rs() {
  local prompt_file="$1"
  adb shell "cd /data/local/tmp && ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b --prompt-file /data/local/tmp/${prompt_file} -n 128 -b opencl 2>&1" \
    | awk '/^Decode:/ { gsub(/[^0-9.]/, "", $2); print $2; exit }'
}

tbt_llama_cpp() {
  local depth="$1"
  local tps
  tps=$(adb shell "cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp ./llama-bench -m Qwen2.5-1.5B-Instruct-f16.gguf -p 0 -n 128 -d ${depth} -r 1 -t 8 2>&1" \
    | awk '
        /tg128/ {
          if (match($0, /\|[[:space:]]*[0-9]+\.[0-9]+[[:space:]]*\xc2\xb1/)) {
            s = substr($0, RSTART+1, RLENGTH-3);
            gsub(/[[:space:]]/, "", s);
            print s;
            exit;
          }
        }')
  if [ -z "$tps" ]; then
    echo "ERROR" >&2
    return 1
  fi
  awk "BEGIN { printf \"%.2f\", 1000.0 / $tps }"
}

PROMPTS=(p100.txt p300.txt p600.txt long_prompt.txt)
DEPTHS=(170 374 676 720)

{
  echo "# $(date) — Qwen flash attention decode benchmark vs llama.cpp CPU"
  echo "# Device: Samsung Galaxy S25 (Adreno 830)"
  echo "# Model: Qwen 2.5-1.5B (head_dim=128)"
  echo "# Tolerance: llm.rs TBT <= llama.cpp TBT * 1.0${TOLERANCE_PCT}"
  if [ "$HAS_LLAMA_BENCH" -eq 0 ]; then
    echo "# NOTE: Qwen GGUF missing — recording llm.rs only"
  fi
  echo
} > "$OUTDIR/qwen_after_c1.txt"

echo "[bench-qwen] Starting 4-context sweep..."

fail=0
for i in 0 1 2 3; do
  f="${PROMPTS[$i]}"
  depth="${DEPTHS[$i]}"
  echo "[bench-qwen] $f (approx seq ${depth})..."

  ours=$(tbt_llm_rs "$f") || ours=""
  if [ -z "$ours" ]; then
    echo "[FAIL] $f: failed to parse llm.rs TBT" | tee -a "$OUTDIR/qwen_after_c1.txt"
    fail=1
    continue
  fi
  sleep "$COOLDOWN_SEC"

  if [ "$HAS_LLAMA_BENCH" -eq 0 ]; then
    echo "[ INFO ] $f depth=$depth llm_rs=${ours}ms (no llama.cpp baseline)" | tee -a "$OUTDIR/qwen_after_c1.txt"
    continue
  fi

  theirs=$(tbt_llama_cpp "$depth") || theirs=""
  if [ -z "$theirs" ]; then
    echo "[FAIL] $f: failed to parse llama.cpp TBT" | tee -a "$OUTDIR/qwen_after_c1.txt"
    fail=1
    sleep "$COOLDOWN_SEC"
    continue
  fi
  sleep "$COOLDOWN_SEC"

  limit=$(awk "BEGIN { printf \"%.2f\", $theirs * (100 + $TOLERANCE_PCT) / 100 }")
  cmp=$(awk "BEGIN { print ($ours > $limit) ? 1 : 0 }")

  line="$f depth=$depth llm_rs=${ours}ms llama_cpp=${theirs}ms limit=${limit}ms"
  if [ "$cmp" = "1" ]; then
    echo "[FAIL] $line" | tee -a "$OUTDIR/qwen_after_c1.txt"
    fail=1
  else
    echo "[ OK ] $line" | tee -a "$OUTDIR/qwen_after_c1.txt"
  fi
done

echo
echo "[bench-qwen] Results saved to $OUTDIR/qwen_after_c1.txt"
exit $fail
```

- [ ] **Step 2: Make the script executable**

Run: `chmod +x scripts/bench_flash_attn_decode_qwen.sh`

- [ ] **Step 3: Run the benchmark (requires device)**

Run: `./scripts/bench_flash_attn_decode_qwen.sh`

Expected outcomes:
- **Happy path** (Qwen GGUF present, all contexts pass gate): 4 `[ OK ]` lines, exit 0.
- **No GGUF**: 4 `[ INFO ]` lines recording llm.rs TBT only, exit 0.
- **Regression**: 1+ `[FAIL]` lines, exit 1. Read the numbers — if llm.rs is slightly over the 5% gate but still reasonable, this is not a kernel bug and may just need more cooldown or a context-specific investigation.

If llm.rs is within 5% of llama.cpp across all 4 contexts, C1 goal is achieved.

- [ ] **Step 4: Inspect the results file**

Run: `cat results/data/flash_attn_decode/qwen_after_c1.txt`

Expected: 4 lines with timing data. The llm.rs numbers should be lower than the ~62 ms/tok pre-flash baseline captured in `results/data/flash_attn_decode/fallback_smoke/qwen2.5-1.5b.txt` (for the same long-context case). Expected improvement: 2-5 ms/tok.

If llm.rs numbers are **identical** to the legacy baseline, the flash path is NOT being taken. Re-check Task 5 logs for "GPU kernel plan built" and `DK=128 compiled` — flash won't activate if either is missing.

- [ ] **Step 5: Commit the script and results**

```bash
git add scripts/bench_flash_attn_decode_qwen.sh results/data/flash_attn_decode/qwen_after_c1.txt
git commit -m "test(bench): add Qwen flash decode regression gate vs llama.cpp CPU

Mirrors bench_flash_attn_decode.sh for the Qwen 2.5-1.5B head_dim=128
configuration. Asserts llm.rs TBT <= llama.cpp TBT * 1.05 across the
4-context sweep. Gracefully degrades to 'llm.rs only' mode when the
Qwen GGUF is not present on device.

Initial results committed for reference."
```

---

## Self-Review

**Spec coverage:**
- [x] Compile `flash_attn_f32_f16.cl` with `-DDK=128 -DDV=128` → Task 2
- [x] Route head_dim=128 through flash path in dispatcher → Task 2
- [x] Route head_dim=128 through plan.rs for decode → Task 3
- [x] Wire from backend to plan config → Task 4
- [x] Host test for dk=128 dispatch → Task 1
- [x] On-device verification → Task 5
- [x] Regression gate vs llama.cpp → Task 6
- [x] Qwen-specific — no Gemma3 work → confirmed out of scope

**Naming consistency:**
- `flash_attn_f32_f16_program_dk64` / `flash_attn_f32_f16_program_dk128` (fields)
- `kernel_flash_attn_f32_f16_q1_dk64` / `kernel_flash_attn_f32_f16_q1_dk128` (kernel cache)
- `has_flash_decode_kernel(head_dim: usize)` (accessor)
- All tasks reference the exact same names.

**TDD ordering:**
- Task 1 (red test) → Task 2 (backend makes it green) → Tasks 3-4 (wire through) → Tasks 5-6 (device verify + bench).
- Intermediate sanity: Task 2 step 9 runs existing dk=64 test to catch rename breakage.

**No placeholders:** Every code step shows the exact replacement block. Every command is runnable. Every test has full source.

**Risk mitigation:**
- **Register pressure for DK=128**: if `flash_attn_f32_f16.cl DK=128 failed:` appears in log, graceful degradation (program is `None`, flash path skipped, legacy used). Task 5 Step 4 catches this immediately.
- **macOS dev host compile fail**: Task 1 test skips cleanly, unblocking CI. Task 5 runs on actual device.
- **Existing dk=64 regression**: Task 2 step 9 runs the existing dk=64 test explicitly.
- **Qwen GGUF missing on device**: Task 6 script detects and degrades to `[ INFO ]` mode, not a hard fail.
