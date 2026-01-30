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

### 3. Stress & Stability Testing
Simulate real-world background usage (e.g., user switching apps while model loads) to ensure stability.

**Use `stress_test_adb.py`:**
This script runs your command on-device while simultaneously switching foreground apps (Settings <-> Home) via ADB.

```bash
python3 ./.agent/skills/testing/scripts/stress_test_adb.py \
    --cmd "/data/local/tmp/generate --model-path ... -n 256" \
    --duration 60 \
    --switch-interval 5
```

**Example: YouTube Background Play:**
Runs the inference while playing a YouTube video (simulates foreground multimedia usage).

```bash
python3 ./.agent/skills/testing/scripts/stress_test_adb.py \
    --cmd "/data/local/tmp/generate --model-path ... -n 128" \
    --background-apps "http://youtube.com/watch?v=dQw4w9WgXcQ" \
    --switch-interval 20
```



### 2. Local Unit Tests
Run standard Rust unit tests on the host machine.
- Command: `cargo test`
- For specific backend: `cargo test --bin test_backend`

## Common Issues
- **Device Not Found**: Check `adb devices`.
- **Permission Denied**: The script automatically runs `chmod +x`, but ensure the device is accessible.
