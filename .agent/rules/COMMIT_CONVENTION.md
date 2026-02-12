# Commit Convention

This project follows the **Conventional Commits** specification.
All commit messages must be formatted as follows:

```
<type>(<scope>): <subject>

<body>

<footer>
```

## Allowed Types

| Type | Description |
| :--- | :--- |
| `feat` | A new feature |
| `fix` | A bug fix |
| `docs` | Documentation only changes |
| `style` | Changes that do not affect the meaning of the code (white-space, formatting, etc) |
| `refactor` | A code change that neither fixes a bug nor adds a feature |
| `perf` | A code change that improves performance |
| `test` | Adding missing tests or correcting existing tests |
| `build` | Changes that affect the build system or external dependencies |
| `ci` | Changes to our CI configuration files and scripts |
| `chore` | Other changes that don't modify src or test files |
| `revert` | Reverts a previous commit |

## Examples

```bash
feat(attention): add flash attention implementation
fix(opencl): resolve memory leak in matmul kernel
docs(readme): update build instructions for Android
perf(cpu): optimize dot product with AVX2 intrinsics
```

## Rules
1.  The **subject** line should be imperative, present tense ("change" not "changed" nor "changes").
2.  The **body** should explain *what* and *why* vs. *how*.
