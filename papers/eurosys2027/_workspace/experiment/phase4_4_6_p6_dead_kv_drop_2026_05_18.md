# Phase 4-4.6 P6 — Dead production-fallback KV drop, G6 검증

날짜: 2026-05-18
디바이스: Galaxy S25 (R3CY408S5SB), Adreno OpenCL
모델: qwen2.5-1.5b-q4_0.gguf, F16 KV, HeadMajor, max_seq=512
Prompt: `The capital of France is` (5 tokens), budget=32
Deterministic 옵션: `--temperature 0 --top-k 1 --no-gpu-plan` (baseline/post-fix 동일)

## 변경

1. **P1 (유지)**: `engine/src/session/forward/model_forward.rs::alloc_standard_kv_caches`
   K/V buffer zero-fill (write_buffer with `vec![0u8; kv_buf_size]`).
2. **P6 (신규)**: `engine/src/bin/generate.rs:3046` happy path 진입 직후
   `drop(kv_caches);` — main() 초입 line 406에서 alloc된
   production-fallback용 KV cache를 build_standard_loop 호출 전에 회수.

빌드: HEAD `1da21e67` + 2 파일 변경, opencl+vulkan+qnn features, release PASS.

## Happy path 진입 확인

post-fix stderr:
```
[Phase4-4.5] standard happy path → DecodeLoop+ModelForward (tokens=5, budget=32)
```

## 생성 결과 (32 토큰, deterministic)

**Baseline** (`/data/local/tmp/generate_baseline`):
```
The capital of France is Paris. It has a population of about 2 million people
and covers an area of 104 square kilometers (km2). The city is divided into
```

**Post-fix** (P1+P6):
```
The capital of France is Paris. Paris is a very big city. It has about
2.2 million people. It is also a very old city. It was built in the
```

첫 8 생성 토큰:
- Baseline: ` Paris`, `.`, ` It`, ` has`, ` a`, ` population`, ` of`, ` about`
- Post-fix: ` Paris`, `.`, ` Paris`, ` is`, ` a`, ` very`, ` big`, ` city`

## G6 결과: **FAIL**

공통 prefix: 2 토큰 (` Paris`, `.`). step 1 (생성 3번째 토큰)부터 분기:
baseline ` It` vs post-fix ` Paris`. P1 단독(반증) → P6 추가도 동일 위치 분기.
→ **P6 가설 반증**. KV pool offset 차이는 numeric divergence의 원인이 아님.

## G7 (참고용, FAIL이지만 기록)

n=1 측정:
- Baseline TTFT 217.92 ms, Decode 27.84 ms/tok, Avg TBT 29.47 ms
- Post-fix TTFT 91.20 ms, Decode 30.00 ms/tok, Avg TBT 31.91 ms
- Δ Avg TBT: +8.3% (post-fix 느림). G6 FAIL이라 5-run 미수행.

post-fix TTFT 91 ms vs baseline 218 ms는 prefill chunked path 차이로 추정 (별도 이슈).

## 부수 발견

deterministic 옵션 전달 시 baseline은 `Temp=0, TopP=0.9, TopK=1` 표시,
post-fix도 동일 표시. 그러나 옵션 미전달 시 baseline은 Temp=0 기본,
post-fix는 Temp=0.8/TopK=40 기본 — happy path sampling config default가
production fallback과 다르게 묶여 있을 가능성. 본 실험과는 별개.

## 결론 / 다음

P1 + P6 둘 다 반증. step 1 분기는 KV 초기 상태나 dead pool offset이
원인이 아님. 가설 후보:
- P7: prefill_workspace=None 분기에서 logits scatter 경로 차이 (Phase 4-4.5 commit `7f7c6856`에서 prefill_workspace 제거됨)
- P8: chunked prefill path가 마지막 chunk logits를 production과 다른 layout으로 returnSampling
- P9: ModelForward decode_workspace eager alloc과 production lazy alloc의
  GPU residual buffer offset 차이

다음 세션은 P7 우선 — prefill 마지막 token의 logits emit 코드를
production fallback과 byte-level 비교 권장.
