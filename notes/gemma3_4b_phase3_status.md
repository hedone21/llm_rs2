# Phase 3 Status — Task 8 result

Date: 2026-04-14
Model dir state: original wrapper config + 2-shard safetensors
Backend: CPU, kv-type F32, max-seq-len 4096

## Outcome
**No crash.** Eval-ll completed 40/40 questions cleanly. Exit code 0.
Wall time: 447.9 s (~7.5 min).

## Model dir state verified
- config.json = Gemma3ForConditionalGeneration wrapper (original)
- model-00001-of-00002.safetensors + model-00002-of-00002.safetensors = original 2 shards
- model.safetensors.index.json = original 2-shard index
- Workaround single-file + stripped index preserved as .workaround.bak

## Key loading log lines
```
Loading model from /home/go/Workspace/llm_rs2/models/gemma3-4b
[Backend] CPU primary, GPU secondary available (SwitchHw ready)
[Config] Weight dtype: F16
Shard index: 883 tensors across 2 shards
Loading 2 safetensors shards...
lm_head not found, deriving from embed_tokens (F16) for lm_head...
[Backend] Migrated 445 weight tensors to GPU zero-copy (ALLOC_HOST_PTR)
Model config: layers=34, kv_heads=4, head_dim=256, max_seq_len=4096
KV cache type: F32, layout: HeadMajor (initial capacity: 4096 tokens, 16777216B per layer)
[Eval-LL] 40 questions, policy=none, kv_budget=0, kv_budget_ratio=0, mode=full-prefill
```

Per-question progression was uniform (see /tmp/gemma3_4b_after_phase2.log):
- 1/40 .. 40/40, latency 6.8–17.1 s per question, no regressions, no anomalies.

## Log tail (last 30 lines)
```
      "choice_byte_lens": [
        21
      ],
      "choice_nlls": [
        14.99360743371091
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
  "wall_time_s": 447.913628722
}
```

## Comparison to Task 1 baseline
- Task 1 (workaround single-file): 40/40 PASS, no crash.
- Task 8 (original 2-shard + wrapper config, post-Phase 2): 40/40 PASS, no crash.
- Loading path now routes through the normal `from_json` + `SafetensorsSource::open` pipeline with the prefix-aware mapper and multimodal-config flatten added in Phase 2.

## Conclusion
Phase 2 (prefix-aware mapper + config flatten + Gemma3 text_config defaults) fully
resolves the Gemma 3 4B loading path. The previously reported `free(): invalid size`
crash was an artifact of the earlier workaround assets (manually stripped prefix +
regenerated single safetensors) or of the pre-Phase 2 loader state; it does not
reproduce on the current code against the original HF distribution layout.

Phase 3 Task 9 (ASan debug) is therefore unnecessary. Proceed to Task 10 (OpenCL eval).
