# 5-F frozen baseline (legacy 삭제 전 동결, S25)

**캡처**: 2026-06-05, HEAD=`8e7ffc67`(F0 완료 직후, F1 삭제 직전)
**장비**: Galaxy S25 (R3CY408S5SB), `opencl --opencl-rpcmem`, 6T
**모델**: `qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf` (★F16 weight = build_plan SUCCESS = 비-vacuous plan)
**명령**: `<bin> -b opencl --opencl-rpcmem -m <f16.gguf> --tokenizer-path <tok> --greedy -n 32 --threads 6 --max-seq-len 512 --kv-type {f16,f32,q4} --prompt 'The history of computing began with'` (argus 는 `--no-resilience` 추가)
**sig 정의**: 생성텍스트(TTFT 직전 라인) + `generated=N (first=X + run=Y) ... final_pos=Z` summary 의 md5.

## bit-identical 기준 (삭제 후 argus 가 재현해야 함)

| KV dtype | BASELINE_SIG (md5) | legacy(OLD) vs argus(fmt) | non-vacuous |
|---|---|---|---|
| f16 | `304f4ada4d902789768e3fda3728272e` | MATCH | build_plan=1, wrap=1 |
| f32 | `684d01d98dcb7ed3cec66db39f926920` | MATCH | build_plan=1, wrap=1 |
| q4  | `1cfba27397867242cdb54f09f22b693a` | MATCH | build_plan=1, wrap=1 |

- 전 dtype legacy≡argus = F0(fmt production 기본) device 검증 PASS. F16 weight 라 plan 경로(execute_plan_fmt ≡ execute<C>, Step 3) → W-2 CPU carve-out 미적용.
- device 백업: `/data/local/tmp/blA_{legacy,argus}_{f16,f32,q4}.out` (legacy 는 곧 삭제되나 .out 은 보존).

## avg_tbt baseline (Decode ms/tok, argus fmt-default, n=5)

| KV dtype | 측정값 | median |
|---|---|---|
| f16 | 53.98 53.91 54.73 54.22 54.35 | **54.22** |
| f32 | 54.01 54.20 54.63 53.90 54.04 | **54.04** |
| q4  | 53.44 53.79 53.39 54.46 55.24 | **53.79** |

post-deletion gate: argus median Decode ms/tok 이 위 대비 Δ≤+3% (f16≤55.8 / f32≤55.7 / q4≤55.4).

## 사용처
F1~F3 (OLD chain + trait + legacy 삭제) 후 argus_cli(fmt-only) 재측정:
- sig md5 가 위 BASELINE_SIG 와 **동일**해야 (동일 fmt 경로, parallel path 제거뿐).
- avg_tbt Δ≤+3%.
- 불일치 시 cutover commit `git revert`.
