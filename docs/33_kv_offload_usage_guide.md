# KV Cache Offload Usage Guide

## Overview

KV cache offload moves per-layer KV data from in-memory tensors to an external store (disk files or LZ4-compressed memory). This reduces peak memory usage at the cost of I/O during attention computation.

A **per-layer prefetch pipeline** (LMCache-inspired) overlaps I/O with compute: while layer N runs its forward pass, layer N+1's KV data is loaded in a background thread.

## CLI Options

```bash
cargo run --release --bin generate -- \
  --model-path models/llama3.2-1b \
  --kv-offload zram \          # "disk" or "zram" (default: "none")
  --kv-type f16 \              # must be f16 or f32 (Q4_0 not supported)
  --kv-layout seq \            # must be seq (SeqMajor layout)
  --offload-dir /tmp/offload \ # disk store directory (disk mode only)
  --prompt "Hello" -n 128
```

| Flag | Values | Default | Description |
|------|--------|---------|-------------|
| `--kv-offload` | `none`, `disk`, `zram` | `none` | Offload mode |
| `--offload-dir` | path | `/tmp/llm_rs2_offload` | Directory for disk store temp files |
| `--kv-type` | `f16`, `f32` | — | KV data type (Q4_0 not supported) |
| `--kv-layout` | `seq` | — | Must be SeqMajor |

## Constraints (Phase 3)

- **SeqMajor layout only** (`--kv-layout seq` required)
- **F16 or F32 only** (Q4_0 KV not supported)
- **No eviction support** — cache grows until `max_seq_len`
- **No score accumulation** — H2O/D2O eviction policies incompatible

## Store Backends

### ZramStore (recommended)

In-memory LZ4 compression with byte-shuffle preprocessing.

- **Compression ratio**: ~1.5–2.3x (F16), ~1.3–2.0x (F32)
- **Latency**: ~1.2ms load per layer (Llama 3.2 1B, F16, 2048 tokens)
- **Memory**: compressed data + 2 layer attn buffers (lazy allocated)

### DiskStore

Raw KV data written to temporary files.

- **Compression**: none (1:1)
- **Latency**: ~2.7ms (F16) to ~5.3ms (F32) per layer load
- **I/O pattern**: sequential read/write with batched fsync (every 64 tokens)
- **Warning**: F32 + DiskStore is not recommended — I/O exceeds compute time, pipeline stalls

## Per-Layer Prefetch Pipeline

During decode, the pipeline overlaps I/O with compute:

```
[Load L0] → [Compute L0 | Load L1] → [Compute L1 | Load L2] → ... → [Compute L15]
             ↑ overlap starts         ↑ L0 buffers released
```

Key design decisions:

1. **Lazy attn buffers**: Only 2 layers hold buffers simultaneously (current + next), saving ~75% vs pre-allocating all 16 layers
2. **SharedBuffer reuse**: Output tensors reuse pre-allocated buffers across `get_view()` calls
3. **`std::thread::scope`**: Safe scoped threads — no `unsafe`, automatic join
4. **Sync fallback**: If preload fails, `get_view()` transparently falls back to synchronous load

## Memory Budget (Llama 3.2 1B, F16, seq=2048)

| Component | Baseline KVCache | Offload (ZramStore) |
|-----------|-----------------|---------------------|
| KV data | 64 MB (16 layers × 4 MB) | ~28 MB (compressed) |
| Attn buffers | — | 8 MB (2 layers × 4 MB) |
| Output buffers | — | ~4 MB (reused) |
| **Total** | **64 MB** | **~40 MB** |
| **Savings** | — | **~37%** |

## Prefill vs Decode

- **Prefill**: Uses standard `forward_into()` — batch token processing, no prefetch needed
- **Decode**: Uses `forward_into_offload()` — per-layer prefetch pipeline active

The transition is automatic: prefill stores all tokens, then decode switches to the pipelined path.

## Performance Notes

- ZramStore F16 is the sweet spot: good compression + I/O < compute → pipeline effective
- DiskStore F32 will show warnings if I/O consistently exceeds compute time
- First decode token has ~2.7ms extra latency (layer 0 sync preload)
- Thread spawn overhead: ~0.05ms × 16 layers = ~0.8ms on Linux x86, higher on Android ARM64

## Architecture

```
OffloadKVCache
├── store: Box<dyn OffloadStore>    # DiskStore or ZramStore
├── attn_k_buf: Option<Vec<u8>>     # Lazy-allocated, released after use
├── attn_v_buf: Option<Vec<u8>>     # Lazy-allocated, released after use
├── preloaded: bool                  # Set by preload(), cleared by get_view()
├── out_k_buf: Option<SharedBuffer> # Reusable output buffer
└── out_v_buf: Option<SharedBuffer> # Reusable output buffer

LlamaModel::forward_into_offload()
├── Embedding lookup (same as forward_into)
├── Preload layer 0 (sync)
├── Layer loop:
│   ├── split_at_mut(i+1) → current, next
│   ├── thread::scope:
│   │   ├── spawn: next.preload()
│   │   └── foreground: layer[i].forward()
│   └── release_buffers(i-1)
└── Final norm + head (same as forward_into)
```
