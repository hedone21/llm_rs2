# AB-2 frozen KvQuantDynamic baseline (command-driven KIVI bit transition, S25)

**캡처**: 2026-06-11, HEAD=`36ebc769`(AB-2 host 완료 + mock kv_dtype 가시화 직후, device 게이트 run)
**장비**: Galaxy S25 (R3CY408S5SB), `opencl --opencl-rpcmem`, 6T
**모델**: `qwen2.5-1.5b-f16.gguf` (KIVI 경로 — KV는 KiviCache, `--kv-type` 무관)
**용도**: AB-2 KiviQuantStage(OneShot·KvMutate·sticky last-applied) directive 경로의 회귀 anchor.
후속 파이프라인 변경 후 동일 시나리오 재실행 시 sig·marker·heartbeat가 본 문서와 MATCH 해야 한다.

## 시나리오 (command-driven quant — manager directive → KiviQuantStage)

mock_manager가 TCP로 `KvQuantDynamic{target_bits:4}` directive를 송신, dispatcher
`submit_kv_quant`(sticky last-applied 게이트, §5.7.3) → KiviQuantStage OneShot이 decode step 0의
`KvMutate` phase에서 28-layer 전 cache `transition_bits(4)` (16→4, dequant→requant) 후 `Consumed`.
greedy 출력이 run 간 byte-identical (3/3 실측 — 동기 전환이라 결정적).

```bash
# device /data/local/tmp 에서:
(./mock_manager --tcp 127.0.0.1:9899 --command KvQuantDynamic --target-bits 4 --wait-secs 2 &)
sleep 1
RUST_LOG=info ./argus_bench -b opencl --opencl-rpcmem \
    -m models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --tokenizer-path models/qwen2.5-1.5b/tokenizer.json \
    --greedy -n 128 --threads 6 --max-seq-len 1536 --kv-dynamic-quant \
    --prompt-file medium_qa.txt \
    --resilience-transport tcp:127.0.0.1:9899
# --kv-dynamic-quant = KIVI 런타임 진입 + initial bits=16(residual-only, (1536/32)*32=1536 cap)
# medium_qa.txt = verify/fixtures/prompts/medium_qa.txt (918 tokens)
```

## 동결 기준 (n=3, 전 run 일치)

sig recipe: `{ grep -B2 '^TTFT' out | head -2; grep -o 'generated=.*' out; } | md5sum` (AB-4/6 동형 —
md5 절대값과 함께 **원본 라인 대조가 정본**).

| 항목 | 동결 값 |
|---|---|
| sig (md5) | `728ab92e414fcdc7f7c886a8f617f78d` (3/3 IDENTICAL) |
| 필수 marker | `[Resilience] Directive seq=1: KvQuantDynamic { target_bits: 4 }` + `[KIVI-Resilience] Transitioned KV cache to 4bit` (verify YAML kvquant_to_q4 regex 글자단위, transition error 라인 0) |
| heartbeat kv_dtype | mock post-directive Heartbeat에 `kv_dtype=q4` (3/3 — §5.7.6 wire 검증, mock 출력 가시화 `36ebc769`) |
| generated summary | `generated=128 (first=16 + run=127) stopped_by=BudgetExhausted final_pos=1045` |
| Decode ms/tok (median) | **104.75** (104.59/104.75/123.14 — run 2 thermal 분산 관찰) |

- 회계 불변식: `final_pos = 917(prompt) + 128 = 1045` — quant 전환은 KV 토큰 수/pos 불변 (§5.7.2 pos-환류 없음).
- **비-vacuous 대조 (happy KIVI)**: 동일 명령 `--no-resilience`(directive 없음) run은 transition marker 0 +
  출력 상이 — Q4 전환의 정밀도 영향 실재. happy(16bit 잔류) Decode 125.90 ms/tok.
- **방향성 참고**: directive run(전환 후 Q4) 104.75 < happy(16bit) 125.90 — Q4화로 attention 대역폭
  감소(예상 방향). KIVI 경로 자체가 Standard(54.3)보다 느린 것은 KIVI attention 구현 특성으로
  **AB-2 무관**(KIVI tbt frozen 게이트는 본 문서가 최초·유일).
- **품질 관찰**: 전환 후 decode 후반 텍스트 품질 저하(verify YAML이 "controlled precision loss,
  divergence by design"으로 명시 — functional_only 게이트). happy 16bit는 decode 후반까지 정합 텍스트.
- TTFT 참고치: 5.4~6.9s (directive는 prefill 중 도착, 전환은 decode step 0 — TTFT 무영향).
- raw 백업: device `/data/local/tmp/ab2_kvq_{dir_{1..3},mock_{1..3},happy,staticq4}.out` +
  호스트 `/tmp/ab2_dev/` — 본 문서가 정본.

## 동반 게이트 (같은 run 에서 GREEN 확인, 2026-06-11)

1. **α-K happy-path 재검증**: sig 15/15 MATCH + tbt median f16 54.29(Δ+0.13%)/f32 54.33(+0.54%)/
   q4 53.43(−0.67%) 전부 Δ≤+3% + trace non-vacuous(build_plan SUCCESS·wrapped 28 KVCache 전 dtype).
2. **KIVI 오라클**: `test_backend --backends opencl` Q2/Q4/Q8 attention 3/3 PASS **L2=0.000000**
   (커널 비트정확 유지, `1f93d6b1` 게이트 동일). FAIL 24종(MatMulTransposed 16+MatMulSlice 8)은
   pre-β 하네스 이슈 사전존재 — AB 회귀로 오판 금지.
3. **static KIVI Q4 (C12 정적 경로)**: `--kv-mode kivi --kv-kivi-bits 4` 정상 완주
   `generated=128 final_pos=1045`, Decode 203.88 ms/tok (참고치).
4. **C12 production e2e 흡수**: directive/happy/static 3종 run 전부 OpenCL KIVI decode —
   `kivi_format::attention_native`/`kivi_cache::update_gpu` production 소비자 경로 최초 e2e 실행
   (backend_capability_plugin_design.md C12 잔여 해소).

## 게이트 사용법 (후속)

1. 동일 명령 재실행 (n≥1) → sig + marker 2종 + heartbeat kv_dtype + generated summary **전부 MATCH**.
2. 같은 run의 α-K happy-path 재검증(§frozen_baseline_alpha_k_5f)이 선행 — KIVI 미진입 경로 무회귀 확인.
3. **범위**: 16→4 전환 한정 동결. 2/8bit 전환·역전환(→16)·RestoreDefaults(guard clear만, 전환 없음
   — §5.7.3 partition과 비대칭)·multi-directive 시퀀스는 미동결 (필요 시 본 틀 복제).
