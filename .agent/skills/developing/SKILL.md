---
name: developing
description: The definitive guide for building, testing, and deploying the llm_rs2 project.
---

# Developing Skill

Use this skill when modifying code, adding features, or verifying changes. It covers the entire lifecycle from compilation to on-device testing.

## 1. Environment Setup

### Prerequisites
- **Rust Toolchain**: Stable (latest)
- **Target**: `aarch64-linux-android` (for Android)
- **Android NDK**: Required for cross-compilation.
- **Environment Variables**: You **MUST** source `android.source` before running any Android build commands.

```bash
# Always run this first in a new shell session
source android.source
```

## 2. Build Workflows

### Host Build (Linux)
Use for quick syntax checking and logic verification that doesn't depend on Android-specific hardware.

```bash
# Check syntax
cargo check

# Run generic tests
cargo test
```

### Android Build (Target)
The primary build artifact is the `generate` binary or `test_backend`.

```bash
# Build main inference binary (Release recommended for performance)
cargo build --target aarch64-linux-android --release --bin generate

# Build backend verification tool
cargo build --target aarch64-linux-android --release --bin test_backend
```

## 3. Testing & Verification

We employ a 3-tier testing strategy.

### Tier 1: Local Unit Tests (Host)
Run standard Rust tests. This catches logic errors in platform-agnostic code (tokenizers, shape inference, etc.).

```bash
cargo test
```

### Tier 2: Backend Verification (On-Device)
Verifies that the OpenCL/CPU kernels produce mathematically correct results on the actual hardware. **Crucial for kernel development.**

**Use Testing Skill:**
```bash
./.agent/skills/testing/scripts/run_android.sh test_backend
```

### Tier 3: Integration/E2E Test (On-Device)
Runs the full `generate` pipeline.

**Use Testing Skill:**
```bash
./.agent/skills/testing/scripts/run_android.sh generate --prompt "Hello" -n 128
```

## 4. Code Quality (Automation)

**Sanity Check Script**:
Run this script before verification to ensure code style and linting are correct.

```bash
./.agent/skills/developing/scripts/sanity_check.sh
```
*   Runs `cargo fmt` (auto-fixes if needed).
*   Runs `cargo clippy` to catch common mistakes.

## 5. Development Tips

### Kernel Development (OpenCL)
- `.cl` files are in `kernels/`.
- Embedded into the binary at build time.
- **Workflow**: Modify `.cl` -> Build `test_backend` -> Push -> Run on device.
- **Debugging**: Use `printf` inside OpenCL kernels (requires flushing stdout) or check `test_backend` output for mismatches.

## 6. Cheat Sheet

| Action | Command |
| :--- | :--- |
| **Setup Env** | `source android.source` |
| **Sanity Check** | `./.agent/skills/developing/scripts/sanity_check.sh` |
| **Build (Android)** | `cargo build --target aarch64-linux-android --release` |
| **Verify Backend** | `./.agent/skills/testing/scripts/run_android.sh test_backend` |
