# Phase 4-4.6 P2 — logits[:10] dump (S25 Adreno OpenCL)

## Setup
- Device: Galaxy S25 (R3CY408S5SB), backend=opencl, Qwen2.5-1.5b Q4_0
- Prompt: `"The capital of France is"` (5 tokens prefill, num_tokens=6 budget → 5 decode steps)
- temperature=0 top-k=1 (deterministic)
- Same binary; gate via `LLMRS_DISABLE_HAPPY_PATH`. `LLMRS_DEBUG_LOGITS=1` 양쪽.

## Raw logits[:10]

### Fallback path (`LLMRS_DISABLE_HAPPY_PATH=1`, decoded "Paris. It has a")
```
step=0 pos=5 token_in=12095 logits=[16.959558, 12.825894, 8.900485, 9.357085, 4.908535, 10.455515, 15.26332, 16.525312, 11.999002, 11.31981]
step=1 pos=6 token_in=13    logits=[0.8238802, -9.327144, -0.44022107, -5.307108, -4.6007667, -0.5881728, -8.250183, -0.50563204, -4.744521, -2.588394]
step=2 pos=7 token_in=1084  logits=[9.584013, 13.7106495, 10.713775, 9.728693, 10.305268, 12.986595, 17.34536, 16.11277, 9.483118, 13.304924]
step=3 pos=8 token_in=702   logits=[8.29175, 11.421771, 6.9095116, 10.040481, 7.621001, 9.09417, 12.264516, 15.375124, 9.609966, 10.61619]
step=4 pos=9 token_in=264   logits=[6.9787188, 8.279046, 1.9933202, 6.7901564, 4.3026814, 6.6487527, 9.617596, 10.891675, 10.360263, 9.088124]
```

### Happy path (no gate, decoded "Paris" 또는 다른 시퀀스, step별 token_in 분기)
```
pos=5 token_in=12095 logits=[16.959558, 12.825894, 8.900485, 9.357085, 4.908535, 10.455515, 15.26332, 16.525312, 11.999002, 11.31981]
pos=6 token_in=13    logits=[0.8238802, -9.327144, -0.44022107, -5.307108, -4.6007667, -0.5881728, -8.250183, -0.50563204, -4.744521, -2.588394]
pos=7 token_in=12095 logits=[11.503639, 11.427719, 7.781163, 7.3476596, 6.33926, 10.51921, 18.14788, 15.281556, 10.778319, 11.771475]
pos=8 token_in=374   logits=[9.701899, 11.106151, 6.902771, 7.97177, 7.736678, 8.072691, 10.972839, 14.679221, 8.129877, 11.6079855]
pos=9 token_in=264   logits=[5.755911, 9.870023, 3.7019193, 7.6646776, 4.87836, 8.970051, 10.339092, 11.50298, 10.495646, 11.064325]
```

## Step별 매칭

| step | pos | token_in (FB / HP) | max abs Δ (logits[:10]) | 매칭 |
|------|-----|--------------------|--------------------------|------|
| 0    | 5   | 12095 / 12095      | 0.0                      | **bit-identical** |
| 1    | 6   | 13 / 13            | 0.0                      | **bit-identical** |
| 2    | 7   | 1084 / **12095**   | (입력 다름, 비교 무의미) | DIVERGE @ token_in |
| 3    | 8   | 702 / 374          | (입력 다름)              | DIVERGE |
| 4    | 9   | 264 / 264          | ~2.3 (e.g. logits[5] 6.65 vs 8.97) | 입력 같지만 KV 상태 다름 |

## 핵심 발견

1. **Step 0/1 logits bit-identical** — prefill + 첫 decode까지 양쪽 path가 **완전히 동일한 수치**를 낸다. `forward_into` contract 자체에 numeric divergence가 없다.
2. **분기는 step 2 진입 input에서 발생.** fallback은 `token_in=1084` (` It`), happy는 `token_in=12095` (`Paris`).
   - step 1 logits가 양쪽 bit-identical인데, sampling 결과가 다른 token으로 흘러 들어간다는 것은 **sampling 또는 token feedback loop에서 발생하는 차이**임을 의미한다 (logits 계산 자체는 정상).
3. step 1 token=`13` (`.`)에서 argmax는 양쪽 모두 같은 logits → 같은 다음 token이어야 하지만, happy path는 step 2 입력으로 `12095`(Paris)를 다시 넣고 있다. 즉, **happy path가 step 2의 input feeding 시 직전 sample 결과가 아닌 다른 토큰(예: 가장 처음 sampled token, 또는 인덱싱 오류로 다른 위치 token)을 사용**하고 있는 것으로 보인다.
4. step 4에서 양쪽 모두 token_in=264 (` a`)로 우연히 다시 같아졌지만 그 시점 KV 상태가 이미 분기되어 logits는 차이 (max Δ ~ 2.3).

## 결론

- `forward_into` 출력 자체는 양쪽 path에서 **완전히 동일**.
- 차이는 **decode loop의 next-token feeding 로직**에 있음. 구체적으로 happy path(DecodeLoop+ModelForward)의 step 2 진입 시 input token이 직전 step 1의 sample 결과(13 → argmax) 대신 prompt 마지막 token(12095=Paris)을 재투입.
- P2 가설(forward contract 위반)은 **반증**. 진짜 버그는 **DecodeLoop의 token state 관리** (직전 sample을 다음 step input으로 올바르게 전달하는 경로).

## 다음 조치 권장

- `engine/src/session/forward/decode_loop.rs`에서 prev_token / next_input 갱신 로직 점검
- 특히 step 0 (prefill 직후 첫 decode)와 step 1 사이 transition에서 sample → next input copy가 누락/잘못된 인덱스를 참조하는지 확인
- token_in 자체도 dump하면 `decode_loop`가 어디서 잘못된 token을 forward에 주는지 즉시 식별 가능
