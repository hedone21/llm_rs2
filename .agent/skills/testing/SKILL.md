---
name: testing
description: Run tests and benchmarks for the llm_rs2 project, specifically focusing on Android targets.
---

# Testing Skill

Use this skill when the user wants to run tests, benchmarks, or verify functionality on Android devices or locally.

## Capabilities

### 1. Android Run (Automated)
Use the helper script to Build -> Push -> Run in one go.

**Command:**
```bash
./.agent/skills/testing/scripts/run_android.sh <binary_name> [args...]
```

**Examples:**

*   **Run Inference**:
    ```bash
    ./.agent/skills/testing/scripts/run_android.sh generate \
        --model-path /data/local/tmp/llm_rs2/models/llama3.2-1b \
        --prompt "Hello" \
        -n 128
    ```

*   **Verify Backend**:
    ```bash
    ./.agent/skills/testing/scripts/run_android.sh test_backend
    ```

### 2. Local Unit Tests
Run standard Rust unit tests on the host machine.
- Command: `cargo test`
- For specific backend: `cargo test --bin test_backend`

## Common Issues
- **Device Not Found**: Check `adb devices`.
- **Permission Denied**: The script automatically runs `chmod +x`, but ensure the device is accessible.
