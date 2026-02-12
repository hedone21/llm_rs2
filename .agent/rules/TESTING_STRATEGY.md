# Testing Strategy

We follow a strict **Test-First** philosophy to ensure reliability and maintainability.

## 1. Design for Testability
Before writing implementation code, design your components to be testable.
-   **Dependency Injection**: Avoid hardcoded dependencies. Pass them in (e.g., via traits or structs).
-   **Pure Functions**: Prefer functions that compute outputs solely from inputs without side effects.
-   **Modularity**: Keep functions and structs small and focused.

## 2. Mandatory Unit Tests
**Every new feature or bug fix must include a unit test.**
-   If you fix a bug, write a test that reproduces the bug (fail) and then passes with the fix.
-   If you add a feature, write tests that verify its behavior under normal and edge cases.
-   **No Exemptions**: "It works on my machine" is not a valid verification.

## 3. Verification via Tests
-   Use unit tests as the primary method of verification.
-   Avoid relying solely on manual ad-hoc testing (like running `main` and printing output).
-   If a behavior is complex, break it down into smaller, testable units.

## 4. Test Structure
-   Place unit tests in a `tests` module within the same file (standard Rust pattern).
-   Use descriptive test names (e.g., `test_matmul_dimensions_mismatch` instead of `test_fail`).
