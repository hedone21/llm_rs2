# Phase 4-4.7 S25 OpenCL 디바이스 게이트 측정

**일시**: 2026-05-18
**HEAD pre**: `f558c4f2` (Phase 4-4.6 sampler 가드)
**HEAD post**: `ca4408f6` (Phase 4-4.7 C3 plan-aware ModelForward)
**디바이스**: Galaxy S25 R3CY408S5SB (Adreno 830 OpenCL)
**모델**: qwen2.5-1.5b-q4_0.gguf
**프롬프트**: `"The capital of France is"`
**Tokens**: 32

## 게이트

| Gate | 기준 | 결과 |
|---|---|---|
| G6' bit-identical 32 tok (`--greedy --repetition-penalty 1.1`) | path_pre == path_post | TBD |
| G7' avg_tbt Δ ≤ 5% (n=5, default args + `--greedy`) | Δ ≤ 5% vs Phase 4-4.6 baseline | TBD |

## 측정 명령어

```bash
# pre/post 모두 동일 args
adb -s R3CY408S5SB shell 'cd /data/local/tmp && LD_LIBRARY_PATH=/data/local/tmp \
    ./generate --model-path /data/local/tmp/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path /data/local/tmp/qwen-tokenizer.json \
    --backend opencl --prompt "The capital of France is" \
    --num-tokens 32 --max-seq-len 512 \
    --greedy --repetition-penalty 1.1'
```

## G6' tokens_pre vs tokens_post

TBD (측정 후 채움)

## G7' avg_tbt n=5

TBD (측정 후 채움)

## 결론

TBD
