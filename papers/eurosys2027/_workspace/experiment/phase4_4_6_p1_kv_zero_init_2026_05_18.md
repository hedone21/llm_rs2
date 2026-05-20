# Phase 4-4.6 P1 — KV cache buffer zero-init (REFUTED)

- 일자: 2026-05-18
- 디바이스: Galaxy S25 (R3CY408S5SB), backend=opencl
- 모델: qwen2.5-1.5b-q4_0.gguf, KV F16 HeadMajor (cap=128, max_seq=512)
- 변경: `engine/src/session/forward/model_forward.rs::alloc_standard_kv_caches`에 `backend.write_buffer`로 K/V 양쪽 zero-fill 추가 (post-fix `--features opencl,vulkan,qnn`, release)
- 호스트 게이트: PASS (`cargo build --release` 50.4s, 21 session unit test, layer_lint baseline diff 0)
- 측정 조건: `--temperature 0 --top-k 1`, prompt "The capital of France is", num-tokens=32, baseline은 `--no-gpu-plan`
- 결정성 확인: baseline 3/3 동일, post-fix 3/3 동일

## G6 결과 — FAIL (변화 없음)

| Path | 토큰 시퀀스 (32 토큰) |
|---|---|
| baseline | `Paris. It has a population of about 2 million people and covers an area of 104 square kilometers (km2). The city is divided into` |
| post-fix (KV zero-init) | `Paris. Paris is a very big city. It has about 2.2 million people. It is also a very old city. It was built in the` |

첫 8 토큰 비교 (prompt 이후):
- baseline: ` Paris` `.` ` It` ` has` ` a` ` population` ` of` ` about`
- post-fix: ` Paris` `.` ` Paris` ` is` ` a` ` very` ` big` ` city`

분기 지점: **step 1 (input=`.`)** — Phase 4-4.5 진단과 동일. step 0 출력 `.`까지는 일치하나 step 1부터 다른 토큰(` It` vs ` Paris`). KV zero-init 적용 이전과 step별 분기 지점이 바뀌지 않음.

## G7 — n=1 참고치만 (G6 FAIL이므로 비공식)

| Path | TTFT | Decode ms/tok | Avg TBT |
|---|---|---|---|
| baseline `--no-gpu-plan` | 215.59 ms | 27.89 | 29.60 |
| post-fix (happy path) | 92.16 ms | 29.96 | 31.90 |

post-fix가 TTFT는 절반(plan 활성), Avg TBT +7.8%. **수치 일치 미달성 상태에서의 TBT 비교는 의미 없음** — 비공식 참고치.

## happy path 진입 증거

```
[Phase4-4.5] standard happy path → DecodeLoop+ModelForward (tokens=5, budget=32)
[Phase4-4.5] generated=32 (first=12095 + run=31) stopped_by=BudgetExhausted final_pos=36
```

`first=12095` = ` Paris` token. prefill 출력은 양 path 동일 (Phase 4-4.5와 일치).

## 가설 반증 결론

**KV cache GPU buffer의 초기 garbage 값은 step 1 분기의 원인이 아니다.** 양쪽 path에서 step 0 출력이 일치하면서 step 1에서만 갈라진다는 사실은, KV 영역에 기록되는 데이터(step 0 K/V projection 결과)는 동일하나 그 *주변* (혹은 다른 alloc 위치의) GPU memory가 step 1 attention 경로에 영향을 주고 있음을 의미한다. KV 영역 자체는 step 0에서 모두 덮어 써지므로 zero-init이 의미를 가지려면 `pos > 0`인 영역의 garbage가 attention에 새는 경우뿐인데, flash-attn은 `pos` mask로 제한되어 그럴 가능성이 낮음을 본 실험이 정량적으로 확인했다.

## 다음 단계 후보 (P2 이후)

1. **workspace tensor zero-init** — `prefill_workspace=None`은 이미 반증되었으나 `decode_workspace` eager alloc 시 다른 위치를 점유할 가능성.
2. **alloc 순서 trace** — happy path vs fallback에서 `memory.alloc*` 호출 순서를 stderr dump로 비교 (cl_mem id/size sequence).
3. **step 1 직전 KV cache 내용 dump** — F16 K[layer=0, head=0, pos=0..6, :8] 양 path 비교. 일치한다면 attention/RoPE 외부 요인, 불일치면 KV write_back 경로 차이.
4. **`plan` 비활성화로 post-fix 측정** — happy path를 `--no-gpu-plan`과 동일 조건으로 격리 (이미 시도되었는지 확인 필요).

본 P1은 **REFUTED**. G6 비트 일치 회복 실패. Phase 4-4.5에서 진단된 step 1 분기 지점이 KV 초기 값과 무관함이 정량적으로 확정됨.
