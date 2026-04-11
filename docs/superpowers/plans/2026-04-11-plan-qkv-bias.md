# Plan.rs QKV Bias Support (C1.5-a) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Lift the `has_qkv_bias` early-return in `transformer.rs:build_plan()` so Qwen 2.5-1.5B (which has QKV biases) can take the pre-bound GPU kernel plan path, eliminating per-decode-step kernel arg rebinding overhead (~8-14 ms/token).

**Architecture:** `kernel_add_row_bias` already exists in `simple_ops.cl` and is used by `forward_gen.rs` for the non-plan path. We extend `LayerPlanConfig`/`FullPlanConfig`/`LayerBufs` with optional `bq_buf`/`bk_buf`/`bv_buf` fields, then in `build_layer_plan` conditionally append a `kernel_add_row_bias` `KernelStep` after each QKV matmul step when the corresponding bias buffer is `Some`. `transformer.rs` extracts bias `cl_mem` handles from `layer.qkv_bias: Option<QkvBias>` and passes them through.

**Tech Stack:** Rust (Cargo workspace `llm_rs2`), OpenCL 2.0 on Adreno 830 (Snapdragon 8 Elite), Android ARM64 cross-compile via NDK.

---

## Pre-fix baseline (V9 strict thermal isolation)

From `results/data/flash_attn_decode/thermal/qwen_strict_isolation.txt`:
- **llm.rs Qwen long** (pf=720, dc=128): **69.66 ms/tok** (±0.3%)
- **llama.cpp CPU long**: 57.37 ms/tok
- **Current gap**: +21.4%
- **Per-token overhead attributable to plan.rs bypass**: estimated 8-14 ms/token from kernel arg rebinding across 28 layers

**Success criterion**: V9 re-bench shows llm.rs long ≤ 62 ms/tok (better than +8% over llama.cpp). Stretch: ≤ 57.37 ms/tok (llama.cpp parity or better).

---

## Key code references

- `engine/src/backend/opencl/plan.rs:~825` — `build_layer_plan` function, QKV matmul steps at lines ~868, 883, 898
- `engine/src/backend/opencl/plan.rs:~540` — `LayerBufs` struct (layer weight handles)
- `engine/src/backend/opencl/plan.rs:~570` — `LayerPlanConfig` struct
- `engine/src/backend/opencl/plan.rs:~1250` — `FullPlanConfig` struct
- `engine/src/backend/opencl/plan.rs:~1320` — `build_full_plan` propagation
- `engine/src/backend/opencl/plan.rs:~1895` — KIVI test helper using `LayerPlanConfig` (must also be updated)
- `engine/src/models/transformer.rs:1437` — the `has_qkv_bias` early-return to remove
- `engine/src/models/transformer.rs:1471-1490` — per-layer `LayerBufs` construction loop
- `engine/src/backend/opencl/mod.rs:2461-2492` — runtime `add_row_bias` dispatch reference (for arg layout)
- `engine/kernels/simple_ops.cl:487-497` — `kernel_add_row_bias` source
- `engine/src/layers/transformer_layer/mod.rs:181-206` — `QkvBias` struct and its use in `TransformerLayer`

## Kernel arg contract (from mod.rs:2461)

```rust
// kernel_add_row_bias(x: global float*, bias: global float*, dim: int, total: int)
// Dispatch: 1D, global = total.div_ceil(64) * 64, no local size
```

For QKV decode (seq_len=1, batch=1):
- Q: `total = n_heads_q * head_dim`, `dim = n_heads_q * head_dim`
- K: `total = n_kv_heads * head_dim`, `dim = n_kv_heads * head_dim`
- V: same as K

Both `total` and `dim` are equal because decode has exactly 1 row. The `x[gid] += bias[gid % dim]` reduces to `x[i] += bias[i]` for this single row.

---

## Task 1: Plan bias field scaffolding (no behavior change yet)

**Files:**
- Modify: `engine/src/backend/opencl/plan.rs`

**Goal:** Thread optional bias buffer handles through all plan config structs. Do NOT yet use them in `build_layer_plan`. This is a pure refactor step.

- [ ] **Step 1: Add bias fields to `LayerBufs`**

Find `LayerBufs` struct (around plan.rs:1300-1310):
```rust
pub struct LayerBufs<'a> {
    pub wq: &'a Mem,
    pub wk: &'a Mem,
    pub wv: &'a Mem,
    pub wo: &'a Mem,
    pub w_gate: &'a Mem,
    pub w_up: &'a Mem,
    pub w_down: &'a Mem,
    pub attn_norm: &'a Mem,
    pub ffn_norm: &'a Mem,
}
```

Replace with:
```rust
pub struct LayerBufs<'a> {
    pub wq: &'a Mem,
    pub wk: &'a Mem,
    pub wv: &'a Mem,
    pub wo: &'a Mem,
    pub w_gate: &'a Mem,
    pub w_up: &'a Mem,
    pub w_down: &'a Mem,
    pub attn_norm: &'a Mem,
    pub ffn_norm: &'a Mem,
    /// Optional QKV bias buffers (F32). Present for models with
    /// `has_qkv_bias=true` (e.g. Qwen2). When `None`, the plan builder
    /// skips the `kernel_add_row_bias` step after each QKV matmul.
    pub bq: Option<&'a Mem>,
    pub bk: Option<&'a Mem>,
    pub bv: Option<&'a Mem>,
}
```

- [ ] **Step 2: Add bias fields to `LayerPlanConfig`**

Find `LayerPlanConfig` (around plan.rs:565-610). Near the weight buffer fields, add:

```rust
    // QKV bias buffers (optional — only for models with has_qkv_bias=true)
    pub bq_buf: Option<&'a Mem>,
    pub bk_buf: Option<&'a Mem>,
    pub bv_buf: Option<&'a Mem>,
```

Add these right after `pub wv_buf: &'a Mem,` (before `wo_buf` or wherever the weight buffers end).

- [ ] **Step 3: Add bias fields to `FullPlanConfig`**

Find `FullPlanConfig` (around plan.rs:1250-1297). This is the outer config — bias buffers live in `layer_bufs: Vec<LayerBufs<'a>>` already after Step 1, so FullPlanConfig itself doesn't need new scalar fields. The propagation happens via `LayerBufs.bq/bk/bv` → `LayerPlanConfig.bq_buf/bk_buf/bv_buf` inside `build_full_plan`.

- [ ] **Step 4: Propagate bias through `build_full_plan`**

Find the `LayerPlanConfig { ... }` literal inside `build_full_plan` (around plan.rs:1331-1373). Add:

```rust
            bq_buf: lb.bq,
            bk_buf: lb.bk,
            bv_buf: lb.bv,
```

right after the existing `wv_buf: lb.wv,` line (matching the field order in LayerPlanConfig).

- [ ] **Step 5: Update KIVI test helper and any other LayerPlanConfig callers**

Search for existing `LayerPlanConfig { ... }` construction sites:

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
grep -n "LayerPlanConfig {" engine/src/backend/opencl/plan.rs
```

Expected sites:
- `build_full_plan` — already handled in Step 4
- `build_kivi_full_plan` (around plan.rs:1895) — needs `bq_buf: None, bk_buf: None, bv_buf: None`

For each site that doesn't have an associated `LayerBufs` (e.g. internal constructors), add:
```rust
            bq_buf: None,
            bk_buf: None,
            bv_buf: None,
```

Also search for `LayerBufs {` construction sites:

```bash
grep -n "LayerBufs {" engine/src/backend/opencl/plan.rs engine/src/models/transformer.rs
```

For any site NOT in transformer.rs's build_plan (Task 3 will handle that one), add:
```rust
                bq: None,
                bk: None,
                bv: None,
```

- [ ] **Step 6: Verify compile + existing tests**

Run:
```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
cargo check -p llm_rs2
cargo check -p llm_rs2 --tests
cargo test -p llm_rs2 --test flash_attn_decode -- --nocapture
cargo test -p llm_rs2 --test flash_attn_decode_dk128 -- --nocapture
```

Expected: all pass (or skip on macOS for device tests). No behavior change because `build_layer_plan` doesn't yet read the new fields.

- [ ] **Step 7: Commit**

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
git add engine/src/backend/opencl/plan.rs
git commit -m "refactor(plan): add optional QKV bias fields to LayerBufs and LayerPlanConfig

Pure scaffolding — the new bq/bk/bv fields are plumbed through
LayerBufs, LayerPlanConfig, and build_full_plan propagation but
build_layer_plan does not yet consume them. Non-bias models set
all three to None and see no behavior change. Existing KIVI test
helper (build_kivi_full_plan) sets them to None."
```

---

## Task 2: `add_row_bias` step in `build_layer_plan`

**Files:**
- Modify: `engine/src/backend/opencl/plan.rs` — `build_layer_plan` function

**Goal:** After each QKV matmul step, conditionally append a `kernel_add_row_bias` `KernelStep` when the corresponding bias buffer is `Some`. Non-bias models remain byte-identical.

- [ ] **Step 1: Write a helper for the bias step**

Add a new helper function right before `build_layer_plan` (around plan.rs:816):

```rust
/// Build a pre-bound `kernel_add_row_bias` step that adds the given bias
/// buffer to the given `x` buffer in-place. Used after QKV matmul steps
/// for models with `has_qkv_bias=true` (Qwen2 etc.).
///
/// Kernel signature (from simple_ops.cl:487):
///   kernel_add_row_bias(float* x, const float* bias, int dim, int total)
///
/// Dispatch: 1D, global = total.div_ceil(64) * 64, no local size.
/// For decode seq_len=1 batch=1, `total == dim == n_heads * head_dim`.
fn build_add_row_bias_step(
    simple_ops_program: &ocl::Program,
    x_buf: &Mem,
    bias_buf: &Mem,
    dim: usize,
    op_tag: OpTag,
) -> Result<KernelStep> {
    let kernel = ocl::core::create_kernel(simple_ops_program, "kernel_add_row_bias")
        .context("create kernel_add_row_bias for plan")?;
    let dim_i32 = dim as i32;
    let total_i32 = dim as i32; // decode: 1 row × dim elements
    unsafe {
        ocl::core::set_kernel_arg(&kernel, 0, ocl::core::ArgVal::mem(x_buf))?;
        ocl::core::set_kernel_arg(&kernel, 1, ocl::core::ArgVal::mem(bias_buf))?;
        ocl::core::set_kernel_arg(&kernel, 2, ocl::core::ArgVal::scalar(&dim_i32))?;
        ocl::core::set_kernel_arg(&kernel, 3, ocl::core::ArgVal::scalar(&total_i32))?;
    }
    let gws = dim.div_ceil(64) * 64;
    Ok(KernelStep {
        kernel,
        ndim: 1,
        global_work_size: [gws, 1, 1],
        local_work_size: None,
        dynamic_args: vec![],
        op_tag,
        retained_bufs: vec![],
    })
}
```

- [ ] **Step 2: Insert bias steps after each QKV matmul in `build_layer_plan`**

Find the three QKV matmul `steps_pre_kv.push(make_f16_matmul_step(...))` calls (around plan.rs:868-908). After each one, add a conditional bias step:

Replace the block:
```rust
    // -----------------------------------------------------------------------
    // 2. matmul Q (residual -> q)
    // -----------------------------------------------------------------------
    steps_pre_kv.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wq_buf,
        config.q_buf,
        config.n_q,
        k,
        OpTag::MatmulQKV,
        None,
        config.is_nosub,
    )?);

    // -----------------------------------------------------------------------
    // 3. matmul K (residual -> k)
    // -----------------------------------------------------------------------
    steps_pre_kv.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wk_buf,
        config.k_buf,
        config.n_k,
        k,
        OpTag::MatmulQKV,
        None,
        config.is_nosub,
    )?);

    // -----------------------------------------------------------------------
    // 4. matmul V (residual -> v)
    // -----------------------------------------------------------------------
    steps_pre_kv.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wv_buf,
        config.v_buf,
        config.n_v,
        k,
        OpTag::MatmulQKV,
        None,
        config.is_nosub,
    )?);
```

With:
```rust
    // -----------------------------------------------------------------------
    // 2. matmul Q (residual -> q) [+ optional bq add]
    // -----------------------------------------------------------------------
    steps_pre_kv.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wq_buf,
        config.q_buf,
        config.n_q,
        k,
        OpTag::MatmulQKV,
        None,
        config.is_nosub,
    )?);
    if let Some(bq) = config.bq_buf {
        steps_pre_kv.push(build_add_row_bias_step(
            config.simple_ops_program,
            config.q_buf,
            bq,
            config.n_q,
            OpTag::MatmulQKV,
        )?);
    }

    // -----------------------------------------------------------------------
    // 3. matmul K (residual -> k) [+ optional bk add]
    // -----------------------------------------------------------------------
    steps_pre_kv.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wk_buf,
        config.k_buf,
        config.n_k,
        k,
        OpTag::MatmulQKV,
        None,
        config.is_nosub,
    )?);
    if let Some(bk) = config.bk_buf {
        steps_pre_kv.push(build_add_row_bias_step(
            config.simple_ops_program,
            config.k_buf,
            bk,
            config.n_k,
            OpTag::MatmulQKV,
        )?);
    }

    // -----------------------------------------------------------------------
    // 4. matmul V (residual -> v) [+ optional bv add]
    // -----------------------------------------------------------------------
    steps_pre_kv.push(make_f16_matmul_step(
        config.f16_program,
        config.residual_buf,
        config.wv_buf,
        config.v_buf,
        config.n_v,
        k,
        OpTag::MatmulQKV,
        None,
        config.is_nosub,
    )?);
    if let Some(bv) = config.bv_buf {
        steps_pre_kv.push(build_add_row_bias_step(
            config.simple_ops_program,
            config.v_buf,
            bv,
            config.n_v,
            OpTag::MatmulQKV,
        )?);
    }
```

Also update the `steps_pre_kv` capacity estimate from `Vec::with_capacity(6)` to `Vec::with_capacity(9)` to accommodate up to 3 additional bias steps.

- [ ] **Step 3: Verify compile + Llama3.2-1b regression**

Run:
```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
cargo check -p llm_rs2
cargo check -p llm_rs2 --tests
cargo test -p llm_rs2 --test flash_attn_decode -- --nocapture
cargo test -p llm_rs2 --test flash_attn_decode_dk128 -- --nocapture
```

Expected: all pass. Non-bias models (Llama3.2-1b) don't populate `bq_buf/bk_buf/bv_buf`, so the `if let Some(...)` guards skip the bias steps → zero behavior change.

- [ ] **Step 4: Commit**

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
git add engine/src/backend/opencl/plan.rs
git commit -m "feat(plan): add kernel_add_row_bias step after QKV matmul when bias is Some

When LayerPlanConfig carries Some(bq/bk/bv), append a pre-bound
kernel_add_row_bias KernelStep after each respective QKV matmul.
Non-bias models pass None for all three and see zero new steps
or behavior change.

This is the second step of lifting the has_qkv_bias early-return
in transformer.rs::build_plan. Task 3 wires the bias buffers from
layer.qkv_bias into the new fields."
```

---

## Task 3: Lift `has_qkv_bias` gate in `transformer.rs::build_plan`

**Files:**
- Modify: `engine/src/models/transformer.rs`

**Goal:** Remove the `if self.config.has_qkv_bias { return None; }` early-return. For each layer, extract `cl_mem` handles from `layer.qkv_bias` (if present) and pass them through `LayerBufs.bq/bk/bv`.

- [ ] **Step 1: Remove the early return**

Find at `transformer.rs:1437`:

```rust
        if self.config.has_qkv_bias {
            return None; // Bias not yet supported in GPU plan
        }
```

Delete this entire 3-line block.

- [ ] **Step 2: Thread bias buffers through the per-layer LayerBufs construction**

Find the per-layer loop around `transformer.rs:1472-1490`:

```rust
        // Collect per-layer buffer handles
        let mut layer_bufs = Vec::new();
        let mut kv_bufs_vec = Vec::new();
        for (i, layer) in self.layers.iter().enumerate() {
            layer_bufs.push(LayerBufs {
                wq: cl!(layer.wq),
                wk: cl!(layer.wk),
                wv: cl!(layer.wv),
                wo: cl!(layer.wo),
                w_gate: cl!(layer.w_gate),
                w_up: cl!(layer.w_up),
                w_down: cl!(layer.w_down),
                attn_norm: cl!(layer.attention_norm),
                ffn_norm: cl!(layer.ffn_norm),
            });
            kv_bufs_vec.push(KvBufs {
                k_cache: cl!(kv_caches[i].k_buffer),
                v_cache: cl!(kv_caches[i].v_buffer),
            });
        }
```

Replace the `LayerBufs { ... }` push with:

```rust
            // Extract optional QKV bias handles. If any bias cl_mem lookup
            // fails (should be rare — same backend, same allocator), fall
            // back to the legacy non-plan path by returning None.
            let (bq, bk, bv) = if let Some(ref bias) = layer.qkv_bias {
                let bq_mem = get_cl_mem(bias.bq.buffer().as_ref()).ok()?;
                let bk_mem = get_cl_mem(bias.bk.buffer().as_ref()).ok()?;
                let bv_mem = get_cl_mem(bias.bv.buffer().as_ref()).ok()?;
                (Some(bq_mem), Some(bk_mem), Some(bv_mem))
            } else {
                (None, None, None)
            };
            layer_bufs.push(LayerBufs {
                wq: cl!(layer.wq),
                wk: cl!(layer.wk),
                wv: cl!(layer.wv),
                wo: cl!(layer.wo),
                w_gate: cl!(layer.w_gate),
                w_up: cl!(layer.w_up),
                w_down: cl!(layer.w_down),
                attn_norm: cl!(layer.attention_norm),
                ffn_norm: cl!(layer.ffn_norm),
                bq,
                bk,
                bv,
            });
```

**Important**: the `get_cl_mem(bias.bq.buffer().as_ref())` call returns `Result<&Mem, _>`, and we need `Option<&'a Mem>`. The `.ok()?` idiom returns from the outer `build_plan` function with `None` on error, which matches the existing behavior for weight buffer lookup failures (see the `cl!` macro).

- [ ] **Step 3: Verify Qwen weight dtype precondition**

The current gate at `transformer.rs:1442` checks:
```rust
        // GPU plan only supports F16 weights (kernel_mul_mat_f16_f32)
        if self.layers[0].wq.dtype() != crate::core::buffer::DType::F16 {
            return None;
        }
```

Qwen2.5-1.5B uses F16 weights (verified in the Task 5 device log from C1-Qwen which showed `KV cache type: F16`). So this check should already pass for Qwen.

**However**, the bias tensor dtype is also important. `kernel_add_row_bias` expects F32 bias. Verify bias dtype matches:

Add a bias dtype precondition check right after the F16 weight check, before the per-layer loop:

```rust
        // kernel_add_row_bias expects F32 bias buffers. If any layer has
        // a QKV bias that isn't F32, fall back to the legacy path.
        if self.config.has_qkv_bias {
            for layer in &self.layers {
                if let Some(ref bias) = layer.qkv_bias {
                    if bias.bq.dtype() != crate::core::buffer::DType::F32
                        || bias.bk.dtype() != crate::core::buffer::DType::F32
                        || bias.bv.dtype() != crate::core::buffer::DType::F32
                    {
                        return None;
                    }
                }
            }
        }
```

Insert this block right after the `wq.dtype() != F16` check (before line 1446 `if backend.name() != "OpenCL"`).

- [ ] **Step 4: Verify compile**

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
cargo check -p llm_rs2
cargo check -p llm_rs2 --tests
```

Expected: zero errors.

- [ ] **Step 5: Run existing host tests (must all still pass)**

```bash
cargo test -p llm_rs2 --test flash_attn_decode -- --nocapture
cargo test -p llm_rs2 --test flash_attn_decode_dk128 -- --nocapture
```

Expected: pass or skip cleanly on macOS.

- [ ] **Step 6: Commit**

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
git add engine/src/models/transformer.rs
git commit -m "feat(plan): lift has_qkv_bias gate — Qwen can now use pre-bound kernel path

Removes the early-return in build_plan() that forced Qwen (and any
model with has_qkv_bias=true) to fall back to the per-token
forward_into path. Extracts QKV bias cl_mem handles from
layer.qkv_bias and threads them through LayerBufs.bq/bk/bv.

Adds a F32 dtype precondition for bias buffers (matching
kernel_add_row_bias's kernel signature). Falls back to the
legacy path if any bias is not F32.

Non-bias models are unaffected because layer.qkv_bias is None
and bq/bk/bv stay None through the whole pipeline."
```

---

## Task 4: Qwen on-device verification (plan active)

**Files:**
- Create: `results/data/flash_attn_decode/regression/c15a_qwen_device.txt` (committed verification log)

**Context**: This task exercises the Qwen binary on Adreno 830 and confirms that:
1. `GPU kernel plan built (28 layers, ...)` appears in the log (proving the `has_qkv_bias` gate was successfully lifted).
2. Greedy output is bit-identical to master (`eb689d1`) — prove no numerical regression from the bias addition step.
3. Decode TBT is captured for comparison with V9 baseline.

- [ ] **Step 1: Cross-compile for Android**

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
export NDK_HOME=/opt/homebrew/share/android-ndk HOST_TAG=darwin-x86_64
export TOOLCHAIN=$NDK_HOME/toolchains/llvm/prebuilt/$HOST_TAG
export CC_aarch64_linux_android=$TOOLCHAIN/bin/aarch64-linux-android21-clang
export CXX_aarch64_linux_android=$TOOLCHAIN/bin/aarch64-linux-android21-clang++
export AR_aarch64_linux_android=$TOOLCHAIN/bin/llvm-ar
export CARGO_TARGET_AARCH64_LINUX_ANDROID_LINKER=$TOOLCHAIN/bin/aarch64-linux-android21-clang

# libOpenCL.so stub must exist for linker
if [ ! -f libs/aarch64/libOpenCL.so ]; then
  cp /Users/li/Workspace/llm_rs2/libs/aarch64/libOpenCL.so libs/aarch64/
fi

cargo build --target aarch64-linux-android --release -p llm_rs2 --bin generate
```

- [ ] **Step 2: Deploy**

```bash
adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate
adb shell 'chmod +x /data/local/tmp/generate'
```

- [ ] **Step 3: Zombie check + thermal baseline**

```bash
adb shell 'ps -A 2>&1 | grep -iE "(llama-cli|generate_master)" | grep -v grep || echo no-zombies'
adb shell 'for z in 1 28; do cat /sys/class/thermal/thermal_zone$z/temp; done'
```

Expected: no zombies, thermal ≤ 50°C. If thermal is higher, wait 2-3 min.

- [ ] **Step 4: Run Qwen and capture full log**

```bash
mkdir -p results/data/flash_attn_decode/regression
adb shell '
  cd /data/local/tmp &&
  RUST_LOG=info ./generate \
    --model-path /data/local/tmp/models/qwen2.5-1.5b \
    --prompt-file /data/local/tmp/p300.txt \
    -n 128 -b opencl 2>&1
' | tee results/data/flash_attn_decode/regression/c15a_qwen_device.txt
```

- [ ] **Step 5: Verify "GPU kernel plan built (28 layers, ...)" appears**

```bash
grep -E "GPU kernel plan built|flash_attn_f32_f16.*DK=128|Decode:" results/data/flash_attn_decode/regression/c15a_qwen_device.txt
```

Expected output includes:
- `flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=128)` — flash kernel ready
- `GPU kernel plan built (28 layers, capacity=...)` — **this is the new line confirming the plan is now active for Qwen**
- `Decode: NN.NN ms/tok (... tok/s) [... tokens, forward only]` — performance

If `GPU kernel plan built` is missing, the bias-related code path failed silently (likely `get_cl_mem` on a bias buffer) and the function returned None. Investigation needed before proceeding.

- [ ] **Step 6: Greedy output parity vs master**

Compare greedy output with the C1-Qwen merge baseline (master `eb689d1`). Run both with identical args:

```bash
# Feature branch
adb shell '
  cd /data/local/tmp &&
  ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b \
    --prompt "The capital of France is" -n 30 -b opencl --greedy 2>&1
' > /tmp/c15a_feat_greedy.txt

# Master binary (already on device as generate_master if still present; if not, rebuild)
adb shell 'ls /data/local/tmp/generate_master' 2>&1
# If missing, you need to rebuild master's binary:
#   (cd /Users/li/Workspace/llm_rs2 && export ... && cargo build --target aarch64-linux-android --release -p llm_rs2 --bin generate)
#   adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate_master
adb shell '
  cd /data/local/tmp &&
  ./generate_master --model-path /data/local/tmp/models/qwen2.5-1.5b \
    --prompt "The capital of France is" -n 30 -b opencl --greedy 2>&1
' > /tmp/c15a_master_greedy.txt

# Extract the continuation text from both and compare
grep -A 1 "The capital of France is" /tmp/c15a_feat_greedy.txt | tail -1 > /tmp/feat_text.txt
grep -A 1 "The capital of France is" /tmp/c15a_master_greedy.txt | tail -1 > /tmp/master_text.txt
diff /tmp/feat_text.txt /tmp/master_text.txt && echo "PARITY OK" || echo "MISMATCH"
```

Expected: `PARITY OK` — bit-identical greedy continuation. If there's a mismatch, the bias add step is introducing numerical drift (likely a wrong arg type or dispatch shape).

- [ ] **Step 7: Commit the verification log**

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
git add results/data/flash_attn_decode/regression/c15a_qwen_device.txt
git commit -m "test(device): verify Qwen plan.rs path with QKV bias on Adreno 830

Captures the first on-device Qwen run with has_qkv_bias gate lifted.
Confirms:
- GPU kernel plan built (28 layers) line appears — plan path is active
- flash_attn_f32_f16.cl DK=128 compiled successfully
- Greedy output bit-identical to master (eb689d1)
- Decode TBT captured for V9 comparison in Task 5"
```

---

## Task 5: V9 strict isolation re-bench for Qwen

**Files:**
- Create: `results/data/flash_attn_decode/thermal/c15a_qwen_strict_isolation.txt`

**Context**: Re-run `scripts/bench_strict_thermal_isolation.sh` with the new binary. Compare vs V9 baseline (llm.rs long 69.66 ms, llama.cpp long 57.37 ms).

**Takes ~75 minutes.** User's constraint: 5-minute rest between every run, CPU/GPU phases fully separated. Do not shorten.

- [ ] **Step 1: Verify binary is fresh on device**

```bash
adb shell 'md5sum /data/local/tmp/generate'
md5sum /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias/target/aarch64-linux-android/release/generate
```

The two MD5s must match. If not, re-push:
```bash
adb push target/aarch64-linux-android/release/generate /data/local/tmp/generate
adb shell 'chmod +x /data/local/tmp/generate'
```

- [ ] **Step 2: Run the strict isolation bench**

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
./scripts/bench_strict_thermal_isolation.sh 2>&1 | tee /tmp/c15a_strict.log
```

The script writes results to `results/data/flash_attn_decode/thermal/qwen_strict_isolation.txt`. **This file will be overwritten** — before running, rename the existing V9 baseline:

```bash
mv results/data/flash_attn_decode/thermal/qwen_strict_isolation.txt \
   results/data/flash_attn_decode/thermal/qwen_strict_isolation_v9_baseline.txt 2>/dev/null || true
```

After the bench, rename the new result to the expected filename:
```bash
mv results/data/flash_attn_decode/thermal/qwen_strict_isolation.txt \
   results/data/flash_attn_decode/thermal/c15a_qwen_strict_isolation.txt
```

- [ ] **Step 3: Compute deltas vs V9 baseline**

V9 baseline (from `results/data/flash_attn_decode/thermal/qwen_strict_isolation.txt` on master):
- Short (pf=7, dc=64): llm.rs=58.59, llama.cpp=50.71, delta=+15.5%
- Long (pf=720, dc=128): llm.rs=69.66, llama.cpp=57.37, delta=+21.4%

Expected C1.5-a improvement (from briefing):
- Short: -3 to -5 ms/tok (less benefit because fewer matmul calls in 64 tokens)
- Long: -8 to -14 ms/tok (larger benefit over 128 tokens and deeper KV)

Target:
- Short llm.rs ≤ 56 ms/tok (still ~10% over llama.cpp)
- **Long llm.rs ≤ 62 ms/tok** (closing to +8% gap)
- **Stretch: Long llm.rs ≤ 57.37 ms/tok** (llama.cpp parity)

- [ ] **Step 4: Write a comparison note**

Create `results/data/flash_attn_decode/thermal/C15A_COMPARISON.md` with a table:

```markdown
# C1.5-a (Plan QKV Bias) vs V9 Baseline

| Combo | Metric | V9 baseline | C1.5-a | Delta |
|---|---|---|---|---|
| Short (pf=7, dc=64) | llm.rs GPU | 58.59 ms | ??? | ??? |
| Short | llama.cpp CPU | 50.71 ms | ??? | ??? |
| Short | GPU vs CPU | +15.5% | ??? | ??? |
| Long (pf=720, dc=128) | llm.rs GPU | 69.66 ms | ??? | ??? |
| Long | llama.cpp CPU | 57.37 ms | ??? | ??? |
| Long | GPU vs CPU | +21.4% | ??? | ??? |

Raw per-run values in qwen_strict_isolation_v9_baseline.txt and
c15a_qwen_strict_isolation.txt.
```

Fill in the `???` values after the re-bench completes.

- [ ] **Step 5: Commit**

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
git add results/data/flash_attn_decode/thermal/c15a_qwen_strict_isolation.txt \
        results/data/flash_attn_decode/thermal/qwen_strict_isolation_v9_baseline.txt \
        results/data/flash_attn_decode/thermal/C15A_COMPARISON.md
git commit -m "test(bench): C1.5-a V9 strict isolation re-bench for Qwen

Re-runs bench_strict_thermal_isolation.sh with the plan-qkv-bias
binary. Compares llm.rs GPU and llama.cpp CPU medians under
identical thermal parity conditions (5-min rest, CPU/GPU batched).

Also preserves the master V9 baseline as
qwen_strict_isolation_v9_baseline.txt for reference.

See C15A_COMPARISON.md for the delta table."
```

---

## Task 6: CPU + GPU regression suite

**Files:**
- Create: `results/data/flash_attn_decode/regression/c15a_regression_summary.md`

**Goal**: Comprehensive regression check across 4 model × backend combinations + existing feature regressions. User explicitly requested "cpu / gpu regression 모두 검증" — do both CPU and GPU paths for Llama and Qwen.

**Do NOT shorten this task.** Each test must produce verifiable output and be logged.

### Test matrix

| # | Model | Backend | Expected |
|---|---|---|---|
| a | llama3.2-1b | cpu | Sane English output, Decode TBT logged |
| b | llama3.2-1b | opencl | Sane English output, plan path active, DK=64 flash active |
| c | qwen2.5-1.5b | cpu | Sane English output, Decode TBT logged |
| d | qwen2.5-1.5b | opencl | Sane English output, plan path NOW ACTIVE (new behavior), DK=128 flash active |
| e | llama3.2-1b H2O | opencl | H2O eviction + sane output (V4 regression) |
| f | llama3.2-1b --profile | opencl | Per-op breakdown + JSON export (V6 regression) |

- [ ] **Step 1: Zombie check + cooldown**

```bash
adb shell 'ps -A 2>&1 | grep -iE "(llama-cli|generate)" | grep -v grep || echo no-zombies'
# Wait until thermal <= 50°C:
for i in 1 28; do adb shell "cat /sys/class/thermal/thermal_zone$i/temp"; done
```

- [ ] **Step 2: Test (a) — Llama3.2-1b CPU**

```bash
adb shell '
  cd /data/local/tmp &&
  ./generate --model-path /data/local/tmp/models/llama3.2-1b \
    --prompt "The quick brown fox jumps over" -n 32 -b cpu 2>&1
' > results/data/flash_attn_decode/regression/c15a_a_llama_cpu.txt

tail -5 results/data/flash_attn_decode/regression/c15a_a_llama_cpu.txt
```

Expected: "the lazy dog" continuation and `Decode: NN.NN ms/tok` line. No crash.

- [ ] **Step 3: Test (b) — Llama3.2-1b OpenCL GPU**

Wait 2 minutes for thermal cooldown:
```bash
sleep 120
```

```bash
adb shell '
  cd /data/local/tmp &&
  RUST_LOG=info ./generate --model-path /data/local/tmp/models/llama3.2-1b \
    --prompt "The quick brown fox jumps over" -n 32 -b opencl 2>&1
' > results/data/flash_attn_decode/regression/c15a_b_llama_gpu.txt

grep -E "GPU kernel plan built|DK=64|Decode:" results/data/flash_attn_decode/regression/c15a_b_llama_gpu.txt
```

Expected:
- `flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=64)`
- `GPU kernel plan built (16 layers, ...)` — still active for Llama
- `Decode: NN.NN ms/tok`
- Sane English output

- [ ] **Step 4: Test (c) — Qwen CPU**

```bash
sleep 120
adb shell '
  cd /data/local/tmp &&
  ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b \
    --prompt "The capital of France is" -n 32 -b cpu 2>&1
' > results/data/flash_attn_decode/regression/c15a_c_qwen_cpu.txt

tail -5 results/data/flash_attn_decode/regression/c15a_c_qwen_cpu.txt
```

Expected: "Paris" continuation and `Decode: NN.NN ms/tok`. No crash. CPU backend does not touch plan.rs, so this is a pure smoke test that our refactor didn't break anything.

- [ ] **Step 5: Test (d) — Qwen OpenCL GPU (the key new behavior)**

```bash
sleep 120
adb shell '
  cd /data/local/tmp &&
  RUST_LOG=info ./generate --model-path /data/local/tmp/models/qwen2.5-1.5b \
    --prompt "The capital of France is" -n 32 -b opencl 2>&1
' > results/data/flash_attn_decode/regression/c15a_d_qwen_gpu.txt

grep -E "GPU kernel plan built|DK=128|Decode:" results/data/flash_attn_decode/regression/c15a_d_qwen_gpu.txt
```

Expected:
- `flash_attn_f32_f16.cl compiled (Q=F32, KV=F16, DK=128)`
- **`GPU kernel plan built (28 layers, ...)`** — this is the critical new assertion
- `Decode: NN.NN ms/tok`
- "Paris" continuation

- [ ] **Step 6: Test (e) — H2O eviction regression**

```bash
sleep 120
adb shell '
  cd /data/local/tmp &&
  ./generate --model-path /data/local/tmp/models/llama3.2-1b \
    --prompt "The quick brown fox jumps over" \
    -n 128 -b opencl \
    --eviction-policy h2o --eviction-window 1024 --h2o-keep-ratio 0.5 2>&1
' > results/data/flash_attn_decode/regression/c15a_e_h2o.txt

grep -E "GPU Score|Eviction|CacheEvent|Decode:" results/data/flash_attn_decode/regression/c15a_e_h2o.txt
```

Expected (matching V4 from master):
- `[GPU Score] Accumulator initialized`
- `Eviction: policy=h2o, ...`
- `[CacheEvent] Eviction completed: ...`
- `Decode: NN.NN ms/tok`

- [ ] **Step 7: Test (f) — Profile mode regression**

```bash
sleep 120
adb shell '
  cd /data/local/tmp &&
  ./generate --model-path /data/local/tmp/models/llama3.2-1b \
    --prompt "The quick brown fox jumps over" \
    -n 32 -b opencl --profile --profile-dir /data/local/tmp/profile_out 2>&1
' > results/data/flash_attn_decode/regression/c15a_f_profile.txt

grep -E "Profile.*breakdown|Exported|Decode:" results/data/flash_attn_decode/regression/c15a_f_profile.txt
```

Expected (matching V6 from master):
- `[Profile] Per-op breakdown (accumulated over NNN layer-calls)`
- `[Profile] Exported to /data/local/tmp/profile_out/...json`
- Per-op table with matmul_ffn, matmul_qkv, attention, rms_norm, etc.
- `Decode: NN.NN ms/tok`

- [ ] **Step 8: Write summary**

Create `results/data/flash_attn_decode/regression/c15a_regression_summary.md`:

```markdown
# C1.5-a Regression Suite

| # | Test | Result | TBT (ms/tok) | Notes |
|---|---|---|---|---|
| a | llama3.2-1b CPU | ??? | ??? | ??? |
| b | llama3.2-1b OpenCL | ??? | ??? | GPU kernel plan built: ??? |
| c | qwen2.5-1.5b CPU | ??? | ??? | ??? |
| d | qwen2.5-1.5b OpenCL | ??? | ??? | **GPU kernel plan built: ??? (new behavior)** |
| e | llama3.2-1b H2O | ??? | ??? | Eviction: ??? |
| f | llama3.2-1b --profile | ??? | ??? | Per-op breakdown: ??? |

All 6 tests must show sane output (continuation text, non-garbled, no all-zero)
and a valid Decode TBT line. For OpenCL tests, the GPU kernel plan build message
must appear (layers=16 for Llama, layers=28 for Qwen).
```

Fill in the `???` values from the captured logs.

- [ ] **Step 9: Commit regression suite**

```bash
cd /Users/li/Workspace/llm_rs2/.claude/worktrees/plan-qkv-bias
git add results/data/flash_attn_decode/regression/c15a_*.txt \
        results/data/flash_attn_decode/regression/c15a_regression_summary.md
git commit -m "test(device): C1.5-a CPU+GPU regression suite

Six-test matrix on Adreno 830 / Snapdragon 8 Elite CPU:
- (a) llama3.2-1b CPU: sane English + Decode TBT logged
- (b) llama3.2-1b OpenCL: GPU plan active (16 layers), DK=64 flash
- (c) qwen2.5-1.5b CPU: sane English + Decode TBT logged
- (d) qwen2.5-1.5b OpenCL: GPU plan NOW active (28 layers, new!),
      DK=128 flash, first Qwen run on the pre-bound kernel path
- (e) H2O eviction: eviction events fire, sane output (V4 regression)
- (f) --profile: per-op breakdown + JSON export (V6 regression)

Each test was run with 2-minute cooldown between runs to avoid
thermal interference. Zombie preflight check passed."
```

---

## Self-Review

**Spec coverage:**
- [x] Plan.rs bias field scaffolding → Task 1
- [x] kernel_add_row_bias step in build_layer_plan → Task 2
- [x] transformer.rs gate lift + bias threading → Task 3
- [x] Qwen on-device verification + parity → Task 4
- [x] V9 strict isolation re-bench → Task 5
- [x] CPU + GPU regression suite → Task 6

**Naming consistency:**
- `bq_buf` / `bk_buf` / `bv_buf` in LayerPlanConfig
- `bq` / `bk` / `bv` in LayerBufs (matching the wq/wk/wv pattern)
- All tasks reference the exact same names

**TDD ordering:**
- Task 1 is pure scaffolding (no behavior change)
- Task 2 adds the step (non-bias models unaffected because `if let Some` guards)
- Task 3 lifts the gate (Qwen now enters the plan path)
- Task 4-6 verify behavior on device

**No placeholders:** Every step shows concrete code / commands with expected output.

**Risk mitigation:**
- **Bias dtype mismatch**: explicit F32 precondition in Task 3 Step 3
- **cl_mem extraction failure**: `.ok()?` returns None, falling back to legacy path
- **Llama regression**: Tasks 1-2 are no-op for non-bias models; Task 6 tests (a)(b)(e)(f) all use Llama
- **Thermal contamination**: Task 5 uses existing strict-isolation bench; Task 6 adds 2-min cooldowns
- **Zombie processes**: Tasks 4 and 6 explicitly check for zombies first
