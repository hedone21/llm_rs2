---
description: Run sanity checks (formatting, linting, testing) before pushing code.
---

// turbo-all

1. Check Formatting
   cargo fmt -- --check

2. Run Linter (Clippy)
   cargo clippy -- -D warnings

3. Run Unit Tests (Host)
   cargo test
