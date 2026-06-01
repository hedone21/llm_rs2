---
name: grill-me
description: Interview the user relentlessly about a plan or design until reaching shared understanding, resolving each branch of the decision tree. Use when user wants to stress-test a plan, get grilled on their design, or mentions "grill me".
---

Interview me relentlessly about every aspect of this plan until we reach a shared understanding. Walk down each branch of the design tree, resolving dependencies between decisions one-by-one. For each question, provide your recommended answer.

Ask the questions one at a time.

If a question can be answered by exploring the codebase, explore the codebase instead.

When a question involves changes to code, show the concrete artifact alongside your recommended answer — not just prose:

- **Code-level change** (new or changed trait / struct / enum / signature): show the actual Rust, with real field names and signatures. Mark each piece — ✅ existing or already-agreed, 🔵 proposed this turn (under review), 🟡 deferred / grows later, ⏸️ out of scope. Ground names in the real codebase (cite `file:line` for existing types so the user can verify).
- **Structural change** (module / trait relationships, dispatch flow, state machine): show ASCII UML — trait+impl tree, dispatch sequence with `──▶`, or before/after diagrams. (Mermaid is for `docs/`; inline grill answers use ASCII.)

Tie the artifact to the recommendation so the user sees exactly what they would be agreeing to before they answer.
