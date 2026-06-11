# AB-6 frozen weight-swap baseline (command-driven SwapWeights, S25)

**캡처**: 2026-06-11, HEAD=`055dfef2`(AB-6 host 완료 직후, device 게이트 run)
**장비**: Galaxy S25 (R3CY408S5SB), `opencl --opencl-rpcmem`, 6T
**모델**: primary `qwen2.5-1.5b-f16.gguf` + secondary `qwen2.5-1.5b-q4_0.gguf` (`--secondary-gguf`), KV f16
**용도**: AB-6 WeightSwapStage(OneShot·WeightMutate·D5 multi-tick) directive 경로의 회귀 anchor.
AB-2 및 후속 파이프라인 변경 후 동일 시나리오 재실행 시 sig·marker·tick 시퀀스가 본 문서와 MATCH 해야 한다.

## 시나리오 (command-driven swap — manager directive → WeightSwapStage)

mock_manager 가 TCP 로 `SwapWeights{ratio:0.5, target_dtype:Q4_0}` directive 를 송신, dispatcher
`submit_swap`(transient — 게이트 없음, §5.6.4) → WeightSwapStage OneShot 이 decode step 0 의
`WeightMutate` phase 에서 commit(§1~7: validate→in-flight→decider→QCF→Incremental plan 설치) 후
step 0~6 에서 7 tick multi-tick drain(per_tick=2, LISWAP-6) → `Consumed`. greedy 출력이 run 간
byte-identical (3/3 실측 — tick 내 동기 swap 이라 결정적).

```bash
# device /data/local/tmp 에서:
(./mock_manager --tcp 127.0.0.1:9899 --command SwapWeights --ratio 0.5 --target-dtype q4_0 --wait-secs 2 &)
sleep 1
RUST_LOG=info ./argus_bench -b opencl --opencl-rpcmem \
    -m models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --secondary-gguf models/qwen2.5-1.5b/qwen2.5-1.5b-q4_0.gguf \
    --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
    --greedy -n 128 --threads 6 --max-seq-len 1536 --kv-type f16 \
    --prompt-file medium_qa.txt --protected-prefix 4 \
    --resilience-transport tcp:127.0.0.1:9899
# medium_qa.txt = verify/fixtures/prompts/medium_qa.txt (918 tokens)
```

## 동결 기준 (n=3, 전 run 일치)

sig recipe: `{ grep -B2 '^TTFT' out | head -2; grep -o 'generated=.*' out; } | md5sum` (AB-4 동형 —
md5 절대값과 함께 **원본 라인 대조가 정본**).

| 항목 | 동결 값 |
|---|---|
| sig (md5) | `c5e4eb0247d49a095b0dc14c5a90093f` (3/3 IDENTICAL) |
| 필수 marker | `[Resilience] Directive seq=1: SwapWeights { ratio: 0.5, target_dtype: Q4_0 }` + `[Decider] allow_boundary_layers=false (ratio=0.5000, mode=Incremental)` + `[WeightSwap] manager path (Incremental): ratio=0.50, 14 layers, per_tick=2 (7 ticks), qcf=0.5000` |
| drain tick 시퀀스 (7 tick, 3/3 동일) | `chunk=[1, 2]`→`[4, 6]`→`[8, 10]`→`[12, 14]`→`[15, 17]`→`[19, 21]`→`[23, 25]`, 각 `swapped=2`, remaining 12→0 (boundary layer 0·26·27 회피 = `allow_boundary_layers=false` 산출) |
| generated summary | `generated=128 (first=16 + run=127) stopped_by=BudgetExhausted final_pos=1045` |
| Decode ms/tok (median) | **50.39** (49.89/50.39/52.23) |

- 회계 불변식: `final_pos = 917(prompt) + 128 = 1045` — swap 은 KV/pos 불변 (§5.6.1 pos-환류 없음).
- **방향성 참고**: Decode 50.39 < α-K F16 happy-path 54.31 — 14 layers 가 Q4_0 로 바뀌어 weight
  대역폭 감소(예상 방향). swap 시나리오 tbt 는 happy-path frozen 게이트와 별도 (참고 추세만).
- TTFT 참고치: ~5.0s (directive 는 prefill 중 도착, drain 은 decode step 0 부터 — TTFT 무영향).
- raw 백업: device `/data/local/tmp/ab6_swap_f16_dir_{1..3}.out` — 본 문서가 정본.

## 게이트 사용법 (AB-2/후속)

1. 동일 명령 재실행 (n≥1) → sig + marker 3종 + tick 시퀀스 + generated summary **전부 MATCH**.
2. 같은 run 의 α-K happy-path 재검증(§frozen_baseline_alpha_k_5f)이 선행 — swap 미발화 경로 무회귀 확인.
3. **범위**: Incremental mode 한정. IntraForward/LayerImmediate(forward slot greenfield 미배선,
   eprintln 경고)·PhaseAware 의 device 검증은 hook 실배선 시점에 별도 (§5.6.3, ADR-0006 Deferred).
   swap 역전(RestoreDefaults) 비대응 = ADR-0006 §6 Deferred 그대로.
