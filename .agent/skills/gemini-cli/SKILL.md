---
name: gemini-cli
description: Integrates previous Claude artifacts and development context to enhance Gemini CLI workflows.
---

# `gemini-cli` Skill

This skill is designed to allow the `gemini` cli agent to seamlessly integrate with and leverage the existing configuration and development context left by previous Claude Code (`claude.ai/code`) sessions without modifying them. 

## Objectives

1. Leverage the context defined in `CLAUDE.md`, which contains comprehensive instructions on project architecture, commands, and constraints.
2. Ensure that `.claude/` files and settings are preserved in their original form.
3. Enhance the `gemini` agent's understanding of the `llm_rs2` repository by building upon rather than replacing established project documentation.

## How to use this skill

When working on tasks related to the architecture, build system, or testing strategy of this repository:

1. **Read `CLAUDE.md`**: Treat `CLAUDE.md` as primary context alongside `PROJECT_CONTEXT.md` and `ARCHITECTURE.md`. It holds valuable cheat codes for Android cross-compilation, OpenCL backend specifics, and zero-copy semantics.
2. **Follow constraints**: Adhere strictly to the constraints outlined in `CLAUDE.md` (e.g., Do NOT modify `.cl` kernel files unless requested, use standard testing procedures).
3. **Respect Claude's domain**: Do not modify files inside the `.claude/` directory unless specifically instructed to migrate or delete them.
4. **Communication Guidelines**: 
   - **Language Preference**: Please provide explanations, summaries, and walkthroughs (e.g. `walkthrough.md`) in **Korean**, translating technical details appropriately.
   - **Proof of Work**: When explaining a fix or optimization, you MUST include comparative test results (e.g., Before vs. After logs or benchmark numbers) to clearly demonstrate that the objective was successfully met.

## Usage Example

When building the application for Android:
Do not invent build commands. Refer to the **Build Commands** section of `CLAUDE.md` or use the `developing` skill. 

To verify the integration, run the provided test case script:
```bash
./.agent/skills/gemini-cli/tests/test_skill_load.sh
```
