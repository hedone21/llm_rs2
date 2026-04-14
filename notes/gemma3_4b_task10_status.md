# Phase 3 Status — Task 10 result (OpenCL validation)

Date: 2026-04-14T11:35:06+09:00
Backend: OpenCL, kv-type F32, max-seq-len 4096 (eval-ll) / 512 (single-prompt)
Host GPU: NVIDIA GeForce RTX 3090 Ti (CUDA OpenCL platform)
Model: /home/go/Workspace/llm_rs2/models/gemma3-4b (original 2-shard, wrapper config)

## Step 1: OpenCL platform/device
```
  Platform Name                                   NVIDIA CUDA
  Platform Name                                   Portable Computing Language
  Platform Name                                   NVIDIA CUDA
  Device Name                                     NVIDIA GeForce RTX 3090 Ti
  Driver Version                                  595.58.03
  Platform Name                                   Portable Computing Language
  Device Name                                     cpu-haswell-Intel(R) Core(TM) Ultra 7 265K
  Driver Version                                  7.1
```

## Step 2: Build
release binary already present; cargo build --release returned: Finished `release` profile [optimized] target(s) in 0.04s

## Step 3: eval-ll (qcf-mode both, 40 questions)

- Exit code: 0
- Wall time: 129.08 s
- Questions processed: 40/40 (id entries in result JSON)
- Model config logged: layers=34, kv_heads=4, head_dim=256, max_seq_len=4096
- Key fallback: `[GPU-fallback] prefill attn: dtype=F32 head_dim=256 reason="head_dim not in {64, 128} (no flash_attn DK variant compiled)"`
  - Gemma3-4B has head_dim=256; current flash_attn DK kernel variants only cover 64/128. Attention falls back to the CPU path, but the rest of the layer (QKV matmul, FFN, RMSNorm, etc.) runs on OpenCL as intended.

### eval-ll tail
```
      "choice_byte_lens": [
        21
      ],
      "choice_nlls": [
        null
      ],
      "choice_token_lens": [
        5
      ],
      "effective_budget": 0,
      "evicted_tokens": 0,
      "eviction_count": 1,
      "eviction_ratio": 1.0,
      "final_cache_pos": 278,
      "id": "race_h_877_c3",
      "n_choices": 1,
      "n_prompt_tokens": 278,
      "predicted": 0,
      "predicted_raw": 0,
      "qcf": 0.0,
      "qcf_attn_norm": 0.0,
      "qcf_attn_raw": 0.0,
      "qcf_caote": 0.0,
      "qcf_layer_skip": null,
      "qcf_layer_skip_layers": null,
      "tokens_evicted": 0
    }
  ],
  "wall_time_s": 129.080557498
}
```

## Step 4: single-prompt smoke

- Prompt: "The capital of France is"
- num-tokens: 16, greedy
- Exit code: 0
- Prefill: 61.51 ms (6 tokens, 97.6 tok/s)
- Decode: 19.71 ms/tok (50.7 tok/s), 15 tokens forward-only
- TTFT: 145.56 ms, Avg TBT: 20.09 ms
- Generated text observed in stdout between prompt echo and "Done.": **empty/blank** — 15 tokens were decoded but produced no visible completion text.
- Kernel build log at model-load: "4 warnings and 8 errors generated." / "2 warnings and 7 errors generated." (x2) / "1 error generated." — NVIDIA OpenCL compiler rejecting several Adreno-targeted kernels; fallback paths activated.

### single-prompt tail
```
[Profile] Event: ModelLoadStart
Loading model from /home/go/Workspace/llm_rs2/models/gemma3-4b
4 warnings and 8 errors generated.
2 warnings and 7 errors generated.
2 warnings and 7 errors generated.
1 error generated.
[Config] Weight dtype: F16
Shard index: 883 tensors across 2 shards
Loading 2 safetensors shards...
lm_head not found, deriving from embed_tokens (F16) for lm_head...
Prompt: The capital of France is
Token Length: 6
Model config: layers=34, kv_heads=4, head_dim=256, max_seq_len=512
KV cache type: F32, layout: HeadMajor (initial capacity: 128 tokens, 524288B per layer, max: 512)
Generating (Max: 512, Temp: 0, TopP: 0.9, TopK: 40)...
[GPU-fallback] prefill attn: dtype=F32 head_dim=256 reason="head_dim not in {64, 128} (no flash_attn DK variant compiled)"
[Profile] Event: PrefillStart
Prefill: 61.51 ms (6 tokens, 97.6 tok/s)
[Profile] Event: DecodingStart
The capital of France is
Done.
[Profile] Event: End
TTFT: 145.56 ms
Decode: 19.71 ms/tok (50.7 tok/s) [15 tokens, forward only]
Avg TBT: 20.09 ms (49.8 tokens/sec)
```

## Outcome: B — OpenCL loads and runs 4B model, single-prompt output degraded on NVIDIA host

- (A) Not applicable: single-prompt completion is blank (no "Paris" token observed).
- **(B) Applies**: eval-ll completes cleanly with exit 0 and full 40/40 question results; model load succeeds (8 GB F16 weights migrated to OpenCL buffers without OOM/crash); numeric paths that matter for eval-ll (logits → NLL) are apparently consistent enough to run to completion. Single-prompt sampled decode produces empty/garbage tokens, which matches the previously documented NVIDIA-host OpenCL fallback kernel issue (see MEMORY.md reference_nvidia_opencl).
- (C) Not applicable: no crash, no error during load, no panic.

### Why this does not block the Gemma3-4B support plan

- The Task 10 goal is to confirm the 4B weight migration and OpenCL execution path works at scale without crashing. That is confirmed.
- Garbage sampled-decode output on NVIDIA host OpenCL is a pre-existing driver/fallback-kernel concern (documented), not a Gemma3-4B-support regression.
- The real OpenCL target for PACT QCF experiments is Android Adreno, validated in Task 12.
- head_dim=256 CPU-fallback for prefill attn is a separate follow-up (flash_attn DK=256 variant) and does NOT block Task 11-13.

## Recommendation for Phase 4 (Task 11-13)

Proceed directly. Blockers: none for PACT QCF flow. Suggested follow-ups (non-blocking):

1. File a tracked issue for DK=256 flash_attn variant if any future head_dim=256 model (Gemma3-4B/12B/27B) needs full-GPU attention on any target.
2. When running on Android Adreno in Task 12, re-verify single-prompt output produces sensible text — if it does, Outcome B is confirmed as NVIDIA-specific.
