# Gemini CLI Skill

The `gemini-cli` skill is an integration module that bridges the context and instructions from previous Claude Code sessions into the Gemini CLI environment.

## Purpose

The user previously used Claude for development on this repository, which generated artifacts such as `CLAUDE.md` and settings in `.claude/`. To preserve these outputs while transitioning to or co-developing with Gemini CLI, this skill guides Gemini to correctly interpret and utilize those historical artifacts.

## Functionality

- **Context Integration**: Enforces Gemini to read `CLAUDE.md` to understand existing build processes, tier-based testing strategies, and architectural rules.
- **Artifact Preservation**: Instructs the agent not to overwrite or modify the Claude-specific configuration files.
- **Workflow Continuity**: Aligns Gemini's actions with the established patterns (e.g., `source android.source` requirement, OpenCL constraints) initially set up with Claude.

## Usage

Agents using this workspace should automatically load this skill and abide by its principles. Developers can run the test script included to ensure that the skill configuration meets standard structures.

## Example

To run the defined tests confirming the skill structure is valid and properly recognized by the system conventions:
```bash
chmod +x .agent/skills/gemini-cli/tests/test_skill_load.sh
./.agent/skills/gemini-cli/tests/test_skill_load.sh
```
