# Accuracy Benchmark: Base vs Sliding vs H2O

Generated: 2026-03-10T19:46:03+09:00

| Model | Prompt | Policy | EMR | FDT | ROUGE-L | BLEU-4 | Top-K Overlap | Evictions |
|-------|--------|--------|-----|-----|---------|--------|---------------|-----------|
| 1B | PPL01 | Base | 1.000 | N/A | 1.000 | 1.000 | 1.000 | 0 |
| 1B | PPL01 | SLIDE | 0.514 | 129 | 0.602 | 0.544 | 0.555 | 1 |
| 1B | PPL01 | H2O | 0.529 | 134 | 0.653 | 0.607 | 0.573 | 1 |
| 1B | PPL03 | Base | 1.000 | N/A | 1.000 | 1.000 | 1.000 | 0 |
| 1B | PPL03 | SLIDE | 0.510 | 129 | 0.550 | 0.466 | 0.525 | 1 |
| 1B | PPL03 | H2O | 0.510 | 129 | 0.551 | 0.499 | 0.524 | 1 |
| 3B | PPL01 | Base | 1.000 | N/A | 1.000 | 1.000 | 1.000 | 0 |
| 3B | PPL01 | SLIDE | 0.510 | 129 | 0.628 | 0.561 | 0.621 | 1 |
| 3B | PPL01 | H2O | 0.510 | 129 | 0.638 | 0.569 | 0.610 | 1 |
| 3B | PPL03 | Base | 1.000 | N/A | 1.000 | 1.000 | 1.000 | 0 |
| 3B | PPL03 | SLIDE | 0.514 | 129 | 0.620 | 0.496 | 0.614 | 1 |
| 3B | PPL03 | H2O | 0.522 | 129 | 0.585 | 0.508 | 0.604 | 1 |

## Parameters
- Decode tokens: 256
- Max seq len: 512
- Eviction trigger: decode token 128 (memory_critical)
- Eviction ratio: 0.50 (keep 50%)
- H2O: keep_ratio=0.5, decay=0.0, time-normalized (default)
- Sampling: greedy (temperature=0)
- Backend: CPU (host)
