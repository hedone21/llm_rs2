---
name: profiling
description: Analyze performance bottlenecks using profiling tools and scripts, and maintain the benchmark record.
---

# Profiling Skill

Use this skill when the user needs to investigate performance issues, generate flamegraphs, or analyze kernel execution times.
**IMPORTANT**: All profiling runs that produce useful data MUST be recorded in the project's benchmark log.

## Tools
- `.agent/skills/profiling/scripts/auto_profile.sh`: **Primary Tool**. Automates the entire data collection and reporting pipeline.

## Workflows

### 1. Automated Profiling (Recommended)
Use the `auto_profile.sh` script to Run -> Visualize -> Update Log in one step.

**Command:**
```bash
./.agent/skills/profiling/scripts/auto_profile.sh \
    --cmd "<full_android_command>" \
    --output-name "<identifier>"
```

**Example:**
```bash
./.agent/skills/profiling/scripts/auto_profile.sh \
    --cmd "/data/local/tmp/generate --model-path /data/local/tmp/llm_rs2/models/llama3.2-1b --prompt-file /data/local/tmp/llm_rs2/eval/short_len.txt --num-tokens 128 -b cpu" \
    --output-name "cpu_short_run"
```

### 2. Manual Profiling (Legacy)
If you need granular control, refer to `results/GUIDE.md` for individual script usage (`android_profile.py`, `visualize_profile.py`, etc.).

## Best Practices
- **Build First**: Ensure the binary is built and pushed (use `testing` skill for this).
- **Clean State**: Ensure no other heavy tasks are running on the device.
