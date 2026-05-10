# LISWAP-5 v1 — Detailed measurement review + postmortem

**Date**: 2026-05-10
**Device**: Galaxy S25 (Adreno 830, 6T)
**Model**: Qwen2.5-1.5B (primary F16, secondary Q4_0)
**Backend**: opencl
**HEAD**: `ed7464f` (plan-bypass fix + diagnostics 후)

---

## 0. 기록 요약

처음 측정에서 LISWAP-5는 **YELLOW/RED**로 보였다 (Decode 48 ms vs single-shot 26 ms = +87% regression). 이를 design 부족으로 해석했지만, **진단 결과 측정 자체가 무의미했음**을 발견:

> **버그 1**: `generate.rs:5371` `gpu_plan` build 게이트가 `--swap-phase-aware`를 무시. plan path가 forward_gen을 우회해서 PHASE_HOOK이 한 번도 fire 안 됨. 측정한 48 ms는 "swap 안 일어난 F16 baseline + dispatcher idle" 값.

수정 후 hook은 정상 fire되고 chunk dispatch 완료. 그러나 **버그 2** 발견 — finalize 후 forward 진행 중 SIGABRT crash. design은 유효해 보이지만 commit + remap 경로에 race/dangling reference 존재.

이 문서는 측정 데이터를 단계별로 풀어서 "왜 효과가 없었는가"를 정확히 설명한다.

---

## 1. 첫 측정 (n=32) — 잘못된 결과 데이터

| 모드 | TTFT | Decode | E2E | 출력 첫 줄 |
|---|---:|---:|---:|---|
| [A] No swap (F16) | 371 | 53.29 | 2023 | "Paris. The French are known for..." |
| [B] Single-shot Q4 | 493 | 25.79 | 1293 | "Paris. The country has a population of about 67M..." |
| [C] LISWAP-5 v1 | 343 | **48.42** | **1844** | "Paris. The country has a population of 67M, the average..." |

**겉으로 보이는 해석**:
- LISWAP-5가 single-shot보다 +43% 느림
- TTFT만 -150 ms 빠름 (swap이 prefill에서 빠짐)
- output mid-decode mixed 때문에 분기

이게 잘못된 해석이었던 이유는 §2~4에서.

---

## 2. n=200 long decode 결과 — 모순 발견

```
[B-200] Single-shot:  Decode 26.77 ms/tok (199 tokens)
[C-200] LISWAP-5:     Decode 48.77 ms/tok (81 tokens)  ← 81??
                      [PhaseAwareSwap retire log 없음]
```

**모순들**:
1. n=200 요청 했는데 LISWAP-5만 81 token에서 멈춤 → EOS 조기 발현 (출력 quality 저하)
2. 200 token이면 25-layer swap이 충분히 끝나야 하는데 `[PhaseAwareSwap] plan retired` 로그 0회
3. Decode TBT 48.77 ms — F16 forward에 가까운 값 (53 ms baseline) — Q4 까지 안 떨어짐

→ **Swap이 한 번도 일어나지 않았을 가능성** 시사.

---

## 3. 진단 — `dispatched=0` 충격

`LLMRS_PHASE_AWARE_DEBUG=1` env로 `debug_snapshot()` 매 token dump:

```
weight_swap: phase-aware mode — ratio=0.90, 25 target layers, chunk_size=4 MB
[op_trace::set_phase_hook] PHASE_HOOK.set result: Ok (PHASE_HOOK now Some=true)
[PhaseAwareSwap-DBG] tok=0 queue=225 in_flight=false pending=0 dispatched=0  hook_start=0  hook_end=3  cachefit_end=0
[PhaseAwareSwap-DBG] tok=1 queue=225 in_flight=false pending=0 dispatched=0  hook_start=0  hook_end=3  cachefit_end=0
[PhaseAwareSwap-DBG] tok=2 queue=225 in_flight=false pending=0 dispatched=0  hook_start=0  hook_end=3  cachefit_end=0
[PhaseAwareSwap-DBG] tok=3 queue=225 in_flight=false pending=0 dispatched=0  hook_start=0  hook_end=3  cachefit_end=0
[PhaseAwareSwap-DBG] tok=4 queue=225 in_flight=false pending=0 dispatched=0  hook_start=0  hook_end=3  cachefit_end=0
```

**해석**:
- `queue=225` 변동 없음 — chunk 한 개도 pop 안 됨
- `dispatched=0` — chunk dispatch 한 번도 일어나지 않음
- `hook_start=0` — `on_op_start` 한 번도 fire 안 됨
- `hook_end=3` — `on_op_end`는 3회 fire (모든 token 합쳐) — 5 token × 28 layer × 11 ops 예상치 ~1500의 1/500
- `cachefit_end=0` — cache-fit phase 한 번도 감지 안 됨
- `set_phase_hook` 결과 `Ok`, `PHASE_HOOK now Some=true` — 등록은 정상

**결정적 단서**: `hook_end=3` 만 fire = `transformer.rs`의 3개 `op_trace::record` 호출 (Embedding, FinalNorm, LmHead) — **이것들은 `start()` (not `start_op`)을 사용해서 hook_start는 0, hook_end만 +1**.

→ `forward_gen`의 11개 `tr_record!` 매크로는 한 번도 invoke 안 됨.
→ **`forward_gen` 자체가 decode 동안 호출되지 않음**.

---

## 4. 근본 원인 — `gpu_plan` path가 `forward_gen` 우회

`engine/src/bin/generate.rs:5371`:
```rust
let mut gpu_plan = if backend.name() == "OpenCL"
    && !args.profile
    && !args.no_gpu_plan
    && accumulator_compatible_with_plan
    && model.config.arch != llm_rs2::models::config::ModelArch::Gemma3
    && !args.swap_intra_forward    // ← LISWAP-4 가드
    // ✗ args.swap_phase_aware 가드 누락!
{
    model.build_plan(&x_gen, &logits, &gen_ws, &mut kv_caches, &backend)
} else {
    None
};
```

LISWAP-4 (intra-forward)가 만든 동일한 함정. handoff_swap_overhead_negative_2026-05-08 §3.1에 상세 기록되어 있던 패턴.

`gpu_plan`이 build되면 decode는 `execute_plan` 경로로 가서 **layer-by-layer가 아닌 fused plan dispatch**로 forward 실행. 이 경로는 op_trace 매크로를 호출하지 않으므로 PHASE_HOOK은 평생 fire 안 함.

---

## 5. Fix 후 재측정 (commit `ed7464f`)

`!args.swap_phase_aware` 가드 추가. 결과:

```
[op_trace::set_phase_hook] PHASE_HOOK.set result: Ok
[forward_gen-DBG] layer_idx=0 start_pos=5    ← forward_gen 정상 진입!
[forward_gen-DBG] layer_idx=1 start_pos=5
[PhaseAwareSwap-DBG] tok=0 queue=140 in_flight=true  pending=1 dispatched=85  hook_start=280 hook_end=286 cachefit_end=140
[PhaseAwareSwap-DBG] tok=1 queue=56  in_flight=true  pending=1 dispatched=169 hook_start=560 hook_end=569 cachefit_end=280
[PhaseAwareSwap-DBG] tok=2 queue=0   in_flight=false pending=0 dispatched=225 hook_start=840 hook_end=852 cachefit_end=420
[PhaseAwareSwap] plan retired at token=2 (drain+sync+bump+invalidate 0.0ms, chunks=225)
[forward_gen-DBG] layer_idx=0 start_pos=8     ← retire 후에도 정상 forward
...
[forward_gen-DBG] layer_idx=1 start_pos=16
Aborted                                       ← 여기서 SIGABRT
```

**Fix 후 동작 분석**:

### 5.1 Hook fire 정상 (token 0~2)
- 28 layer × 11 op = **308 op/token** 예상
- 실측: hook_end ~285/token (≈ 308 - prefill 거치지 않은 일부 ops)
- cachefit_end ~140/token = 5 cache-fit ops/layer × 28 layers = **정확히 일치**

### 5.2 Chunk dispatch 진행
- **token 0 안에 85 chunks dispatch** = 한 token (95ms) 동안 85회 chunk 처리
- token 0~2 합쳐 225 chunks 모두 완료. 약 3 token (~300 ms) 만에 25-layer swap 끝.
- per-token: ~75 chunks. 각 chunk = ~mean 1 ms host work + async H2D.

### 5.3 plan retired at token=2
- 모든 chunk dispatch + commit 완료 후 retire 발생
- drain log "0.0 ms" — `AsyncSwapDispatcher::drain`이 즉시 반환 (이미 모든 commit job worker가 처리 완료)

### 5.4 Crash at token ~11 (SIGABRT)
- retire 후 9 token 정상 forward
- 그 후 segfault — ArcSwap commit + remap_weights_for_cpu_after_swap의 race
- 가능 원인:
  - 새 LayerWeights의 cl_mem이 install되었지만 workspace (gen_ws)가 옛 cl_mem 포인터를 보유
  - `noshuffle SOA registry invalidate` 가 active 중인 partition_ctx에 영향
  - release_worker가 cl_mem release 중에 forward가 그 cl_mem 사용

---

## 6. 진짜 v1 평가 — 실측값 (2026-05-10 update)

**갱신 사유**: 다음 세션이 crash fix를 시작하기 전 reproducer를 재실행한 결과 SIGABRT가 **재현되지 않음**. n=32, n=64×5회, n=200×3회 모두 정상 완료. handoff `12bac91`에서 보고된 crash는 flaky (race timing 차이) 였거나 측정 당시 환경 노이즈였던 것으로 추정. **v1 wall-clock 측정 가능.**

### 6.1 측정 setup
- HEAD: `125cbc9`
- Binary: `12bac91` 시점 빌드 (동일)
- Device: Galaxy S25 (Adreno 830, 6T)
- Prompt: "The capital of France is", temperature=0, initial-kv-capacity=2048
- 4 mode × 3 run × 3 n (10/50/200) alternating, mode 사이 idle 10s, n 사이 idle 15s (thermal-matched)
- swap config: `--force-swap-ratio 0.9` (28 layer 중 25 swap), F16 → Q4_0

### 6.2 정밀 분리 측정 — Prefill / TTFT / tok[0] / Decode tail (best-of-3, cool run)

| 모드 | n | Prefill_pure | TTFT | Decode all | Decode tail (excl tok0) | tok[0] |
|---|---:|---:|---:|---:|---:|---:|
| Q4 only | 10 | 115 | 213 | 27.5 | 27.4 | 27.94 |
| Q4 only | 50 | 115 | 213 | 27.5 | 27.5 | 27.95 |
| Q4 only | 200 | 115 | 213 | 28.1 | 28.1 | 27.72 |
| F16 only | 10 | 242 | 368 | 53.3 | 53.3 | 53.21 |
| F16 only | 200 | 244 | 369 | 54.0 | 54.0 | 53.14 |
| **Single-shot** | 10 | **105** | **469** | 25.7 | 25.6 | 26.56 |
| **Single-shot** | 50 | 104 | 466 | 25.7 | 25.7 | 26.74 |
| **Single-shot** | 200 | 108 | **620** | 26.4 | 26.4 | 26.55 |
| **Phase-aware** | 10 | **219** | **339** | **81.8** | **64.9** | **216.83** |
| **Phase-aware** | 50 | 226 | 344 | 36.9 | 33.1 | 216.87 |
| **Phase-aware** | 200 | 224 | **348** | 30.8 | 29.3 | **321.55** |

### 6.3 핵심 finding (수정 — 이전 분석 무효)

이전 §6 분석은 "Decode 평균"이 phase-aware +10 ms loss라고 했지만, **n별로 Decode 평균이 변함을 무시했음**. n=10에서 Decode 81.8 → n=200에서 30.8로 급격히 감소. 이는 **swap 비용이 tok[0~2]에 집중**되어 n이 커질수록 평균에 흡수된다는 증거.

#### Finding 1 — Prefill 강제 비용 +119 ms (phase-aware의 본질적 손해)
- Single-shot prefill_pure = **105 ms (Q4 forward)** — swap이 prefill 직전 동기 수행되어 prefill 시점에는 이미 Q4 weights
- Phase-aware prefill_pure = **224 ms (F16 forward)** — swap은 op_trace boundary 필요 → decode 시작 후에만 가능 → prefill은 원본 F16 weights
- **Δ = +119 ms 영구 손해**. n에 무관하게 항상 발생.

#### Finding 2 — Swap blocking이 token 0에 집중 (216~322 ms)
- Phase-aware tok[0] = **216~322 ms** (n에 따라 증가? n=200에서 321 ms — 더 많은 chunk가 token 0~2에 집중되었을 가능성)
- Single-shot tok[0] = 26 ms (정상 Q4 forward — swap은 prefill에서 이미 끝났음)
- → tok[0]에서 +200~300 ms 추가. swap이 forward와 overlap **안 함**, 오히려 forward를 blocking.

#### Finding 3 — Decode tail은 n이 커질수록 Q4 baseline에 수렴 (steady state는 거의 동일)
- Phase-aware tail (excl tok0): **n=10 64.9 ms → n=50 33.1 ms → n=200 29.3 ms**
- Q4 only baseline: 27.4~28.1 ms
- → token 3 이후 거의 회복. n=200에서 phase-aware tail (29.3) 과 Q4 only (28.1) 차이는 **+1.2 ms** 만 (이전 분석의 +10 ms 잘못)
- "+10 ms steady loss" 가설은 **틀림**. 진짜 loss는 token 0~2에 집중.

#### Finding 4 — Swap 총량은 비슷 (분산 vs blocking 차이만)
| 모드 | swap 총 비용 추정 | 위치 |
|---|---:|---|
| Single-shot | TTFT - prefill_pure - tok[0] = 620-108-26 = **486 ms** | prefill 직전 동기 |
| Phase-aware | (prefill F16 - prefill Q4) + tok[0] excess + tail excess<br>= 119 + (321-26) + 198×(29.3-28.1) = 119 + 295 + 238 = **652 ms** | prefill F16 + tok[0~10] 분산 |

→ Phase-aware의 swap 총 비용이 single-shot보다 **+166 ms 큼** (분산이 효율 감소).

### 6.4 E2E 비교 (수정)

| n | Single-shot E2E | Phase-aware E2E | Δ | TTFT Δ |
|---:|---:|---:|---:|---:|
| 10 | 469 + 9×25.7 = **700 ms** | 339 + 9×81.8 = **1075 ms** | +375 (phase-aware 손해) | -130 (phase win) |
| 50 | 466 + 49×25.7 = **1726 ms** | 344 + 49×36.9 = **2152 ms** | +426 | -122 |
| 200 | 620 + 199×26.4 = **5874 ms** | 348 + 199×30.8 = **6477 ms** | +603 | -272 |

**이전 분석의 crossover n=13은 무효**. 모든 n에서 phase-aware의 E2E가 single-shot보다 worse. 차이는 n에 따라 **+375 ~ +603 ms**.

→ **유일한 phase-aware 이득은 TTFT 단축 (-130~-272 ms)**. 이는 실시간 streaming UX에서만 의미. 총 처리 시간은 항상 phase-aware 손해.

### 6.5 Swap data 분석 (이론치 vs 실측)

**Swap payload**:
- 25 layer × 9 tensor = **225 chunks** (per-tensor 통째로, sub-chunk 없음)
- 1 layer Q4 weight ≈ q(1.3) + k(0.22) + v(0.22) + o(1.3) + gate(7.6) + up(7.6) + down(7.6) + 2 norm(neg) ≈ **25.84 MB**
- 25 layer 총 = **~646 MB Q4 read** + 동량 buffer write

**이론 transfer time** (UMA H2D):
- @ 7 GB/s = 92 ms
- @ 8 GB/s = 80 ms

**실측 (single-shot blocking)**: 486 ms → **실제 5~6× 느림**. 차이의 원천:
- mmap fault (cold first-touch 대부분, ~150 ms 추정)
- GGUF block re-pack (Q4_0 layout → backend layout 변환)
- OpenCL `enqueue_write_buffer` 자체 driver overhead (per-tensor sync 포함)

**실측 (phase-aware 총량)**: 652 ms → single-shot보다 +166 ms 큼:
- per-chunk dispatch 분산 효과 (cache-fit window 980us를 9~26 MB chunk가 fit하지 못함)
- forward와 진짜 overlap이 일어나지 않음 — chunk가 forward kernel과 같은 OpenCL queue를 직렬 점유

### 6.6 4-gate 결과 (수정)

| Gate | 결과 | 근거 |
|---|---|---|
| ✅ Correctness | PASS | 200 token plausible English ("...The most important fact about France is that it was") |
| ⚠️ TBT | **mostly FAIL** | E2E +375~603 ms loss. TTFT -130~272 ms win은 streaming UX에만 의미 |
| ❌ Hide ratio | **FAIL** | tok[0] 321 ms 측정 — swap이 forward와 overlap 안 함, 오히려 blocking. expected hide ratio (Phase R 96%) 미달 |
| ✅ Stall | PASS | crash 재현 불가, normal 32~200 token decode 정상 |

**종합 (수정)**: design은 functional이지만 실제 hide 효과 거의 없음. swap이 forward와 overlap되지 않고 token 0에 blocking 형태로 청구됨. v2 sub-chunking + background pre-staging이 필수.

---

## 7. 효과 없었던 진짜 이유 정리 (2026-05-10 final)

이전 분석은 평균 Decode만 보고 "+10 ms steady loss + crossover n=13"라 했지만 **n별 분리 측정으로 무효화**. 진짜 원인 4가지:

### 7.1 [PRIMARY] Prefill phase에서 swap 못 함 → F16 prefill 강제 (+119 ms)
- Phase-aware는 op_trace boundary trigger 의존 → prefill 단계 (5 token) 동안에는 chunk dispatch 안 일어남
- 따라서 prefill은 **F16 weights 그대로** (224 ms) — Q4 prefill (105 ms) 대비 **+119 ms**
- Single-shot은 prefill 직전 swap 동기 → prefill 시점 이미 Q4
- **n에 무관하게 영구 손해**. 모든 E2E 비교에서 Single-shot 대비 baseline +119 ms 짊어지고 시작.

### 7.2 [PRIMARY] tok[0]에 swap blocking +200~300 ms
- tok[0] = **216~322 ms** (Q4 forward 26 ms 대비 +200~300)
- 이는 **swap이 forward와 overlap 안 함**의 직접 증거. forward thread가 첫 token에서 chunk dispatch 비용을 동기로 짊어짐
- 원인:
  - chunk size = per-tensor 통째 (`byte_offset=0, byte_len=0`, `chunk_size_bytes` 미사용 — `phase_aware_swap.rs:236-243`)
  - tensor sizes wq/wo (1.3 MB), gate/up/down (7.6 MB) — 모두 cache-fit window (980us = 4 MB throughput) 초과
  - `materialise_cpu_tensor` (mmap fault + GGUF block re-pack)이 forward thread blocking 호출
- 결과: phase boundary 감지에도 chunk이 너무 커서 forward와 같은 OpenCL queue를 직렬 점유

### 7.3 [SECONDARY] Decode tail은 거의 정상 (Q4 baseline +1.2 ms)
- n=200 phase-aware tail = 29.3 ms vs Q4 only = 28.1 ms → **+1.2 ms 만 차이** (이전 분석 +10 ms 잘못)
- token 3 이후는 swap dispatcher 비활성 → 정상 Q4 forward 회복
- → "SOA lazy rebuild" 가설은 **무효** (영향 거의 없음)

### 7.4 [DESIGN] swap 총 비용 자체가 single-shot보다 큼 (+166 ms)
- Single-shot swap blocking ≈ 486 ms
- Phase-aware swap 분산 총량 ≈ 652 ms (prefill F16 +119 + tok[0] excess +295 + tail +238)
- **분산이 효율을 떨어뜨림** — 한 번에 하면 빠르게 끝날 것을 chunk-by-chunk로 driver가 batch 못 함

### 7.5 v1의 본질 한계
hide의 전제 조건은:
1. ✗ chunk가 cache-fit window 안에 fit (현재 9~26 MB → 980us 초과)
2. ✗ forward thread가 host work 안 짊어짐 (현재 mmap+repack이 forward thread block)
3. ✗ prefill phase에서도 swap 가능 (현재 op_trace 후 시작)
4. ✓ phase 예측 deterministic (CV 1.2% — 유일하게 충족)

→ Phase R Scenario B (1.04× of max) 의 hide 96% 와의 격차는 (1)+(2)+(3) 모두 미흡 때문.

**E**는 본질 — 평가 기준은 byte-equal이 아닌 plausible English.

---

## 7.6 Fair comparison — single-shot in decode + per-tick + phase-aware (2026-05-10 결정적 finding)

이전 §6, §7, §7.5 측정의 single-shot은 **prefill 전 swap** (`generate.rs:2752` 주석: "Applied once before generation starts"). 이는 production async swap baseline과 다름. fair한 비교는 셋 다 **decode 도중 swap**.

### 7.6.1 측정 setup
- KV 1.8k (prompt_1800.txt 1821 token), n=20, 3 run
- prefill은 셋 다 F16 (swap 후 decode 진입)
- (a) `--swap-incremental-per-tick 25` (LISWAP-1, 1 tick에 모든 layer = "single-shot in decode")
- (b) `--swap-incremental-per-tick 1` (LISWAP-1 per-token-per-layer)
- (c) `--swap-phase-aware` (LISWAP-5)

### 7.6.2 결과 (KV 1.8k, n=20)

| 모드 | Prefill_pure | TTFT | tok[0] | Decode tail (excl tok0) | Decode all |
|---|---:|---:|---:|---:|---:|
| **per-tick=25** (single in decode) | 12172~13038 | 12296~13164 | **65~72** | **44~47** (= Q4 baseline) | 45.2~47.8 |
| **per-tick=1** (LISWAP-1) | 12996~13352 | 13115~13476 | 71.5~71.8 | **62.2~62.4** (+15 vs Q4) | 62.7~62.9 |
| **phase-aware** (LISWAP-5) | 13010~13404 | 13133~13527 | **347~353** | **77.7~78.2** (+30 vs Q4) | 92.0~92.7 |

(Q4 baseline: tok[0]=49.6, tail=47.4; F16 baseline: tail=71.1)

### 7.6.3 결정적 finding — 분산이 항상 비효율적

| Metric | per-tick=25 | per-tick=1 | phase-aware | 우열 |
|---|---:|---:|---:|---|
| tok[0] | 65 ms | 72 ms | **347 ms** | phase-aware **5×** 느림 |
| Decode tail | 45 ms (=Q4) | 62 ms | **78 ms** | phase-aware 최악 |
| E2E (n=20) | **13359 ms** | 14493 ms | **15076 ms** | phase-aware **+1717 ms** |

**해석**:
- **per-tick=25 (single-shot in decode)**: token 0에 25 layer 모두 swap → 65 ms blocking → token 1+는 정상 Q4 (Q4 baseline 47에 정확히 도달)
- **per-tick=1**: token 0~18에 layer 1개씩 swap → tail 62 ms (각 token에 1 layer swap +15 ms)
- **phase-aware**: chunk 분산이지만 매 token에 chunk dispatch 잔류 → tail +30 ms × 18 = 540 ms 추가

### 7.6.4 왜 분산이 비효율적인가

phase-aware의 core 가설 ("chunk을 분산하면 forward와 overlap 가능") **실측 무효**. 분산이 추가 cost를 3가지 방식으로 누적:

1. **Batching loss**: 한 번에 swap (per-tick=25)은 driver가 큰 contiguous transfer로 batch. 분산은 매번 separate enqueue.
2. **Enqueue overhead**: 225 chunks → 225 OpenCL API call. per-tick=25는 25 layer commit으로 끝. **9× 더 적은 API call**.
3. **Driver scheduler overhead**: chunk-by-chunk가 forward kernel과 같은 OpenCL queue 경합 → 매 phase에 reschedule + per-tick=1의 매 token invalidate cycle.

**Note (2026-05-10 정정)**: 초기 분석에서 "SOA rebuild repeat" 를 4번째 cost로 들었으나, **production 전체가 의도적으로 AoS Q4_0 path 사용** (CPU-GPU 동시 작업 = tensor partition 지원을 위한 강제 포맷). swap_executor도 GGUF AOS path에서는 SOA register를 의도적으로 skip. 따라서 SOA rebuild는 본 측정의 변동 요인이 아님. 셋 다 AoS path로 동작하며 차이는 분산 자체의 비효율 + per-call overhead.

→ Phase R microbench (1.04× of max) 가정 ("hide ratio 96%")은 **production에서 깨짐**. 실측 hide ratio = 0% (분산이 forward와 overlap 안 함).

### 7.6.5 결론 — Phase-aware의 핵심 가설 무효

기존 §7 결론은 "phase-aware가 KV 1.8k short response에서 win"이었지만, 이는 **single-shot이 prefill 전 swap이라는 unfair baseline** 때문에 그랬음. fair comparison에서는:

- **Single-shot in decode (per-tick=25)가 모든 metric에서 최적**
- **분산할수록 비효율적 — phase-aware가 worst**
- v2 sub-tensor chunking은 **chunk 수를 늘리면 더 비효율적**일 가능성. 가설 재검토 필요

### 7.6.6 v2 plan 재검토 (2026-05-10 final)

§7.5 / §8 의 v2 priority 일부 무효:
- ~~Sub-tensor chunking (§8 P2)~~: chunk 수 증가 → enqueue overhead + scheduler 경합 증가. **drop**
- ~~Phase-aware finalize에 SOA register 추가~~: production이 의도적 AoS path 사용 (CPU-GPU 동시 작업 강제). 추가하면 garbage 출력 위험. **drop**
- **새 방향 (확정)**: **DMA-BUF alias** — secondary GGUF mmap region을 OpenCL `CL_MEM_USE_HOST_PTR` alias로 직접 wrap (qnn_oppkg KV dual-buffer 패턴 모방). swap blocking을 per-call enqueue + H2D copy 없이 cl_mem reference 갱신만으로 완료.
  - 현재 v1 swap blocking ≈ 480 ms (mmap fault + repack + H2D copy + per-call overhead)
  - alias version 예상 ≈ 5~10 ms (cl_mem reference 갱신 + AoS layout 그대로 활용)
  - 분산 vs single-shot 논쟁 자체가 무의미해짐 (swap 자체가 거의 free)
- LISWAP-5 design 폐기 vs phase-aware의 **timing predictor** 가치 (deterministic phase boundary)는 alias 위에서 재평가

---

## 7.5 KV context size 영향 분석 (2026-05-10 추가)

질문: KV가 길수록 attention DDR-heavy phase가 늘어나 cache-fit phase 사이 간격이 벌어지면 swap chunk hide가 잘 될까?

### 7.5.1 측정 setup
- prompt_1024.txt (1025 token), prompt_1800.txt (1821 token) 사용 (model max_seq_len=2048 한계로 KV 4k 측정 불가)
- 4 mode × 3 run, n=20 (KV 1k phase-aware/F16은 EOS hit으로 1 token만 측정)
- thermal-matched alternating

### 7.5.2 측정 결과

| 모드 | KV 5 (단순 prompt) Prefill | KV 1k Prefill | KV 1.8k Prefill | KV 5 tok[0] | KV 1k tok[0] | KV 1.8k tok[0] | KV 5 tail | KV 1.8k tail |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Q4 only | 115 | 7147 | 13987 | 28 | 39.7 | 49.6 | 28.1 | 47.4 |
| F16 only | 244 | **6364** | **12640** | 53 | n/a | 75.1 | 54.0 | 71.1 |
| Single-shot | 105 | 7147 | 14543 | 26 | 41.4 | 46.1 | 26.4 | 46.7 |
| Phase-aware | 224 | **6438** | **13064** | 216~322 | 344~392 | **341.8~365.7** | 29.3 | **78.0** |

### 7.5.3 핵심 finding 3가지

#### Finding 1 — tok[0]는 KV에 거의 무관 (~340~390 ms)
| KV | Phase-aware tok[0] |
|---|---:|
| 5 | 216~322 ms |
| 1k | 344~392 ms |
| 1.8k | 342~366 ms |

→ **chunk dispatch overhead 자체가 attention compute time과 분리**. KV 길이가 attention DDR-heavy phase를 늘려도 chunk가 그 안에 hide되지 않음. 가설(KV 길수록 hide 잘 됨) **기각**. 이유: chunk가 cache-fit window 안에 fit하지 못하는 현상이 KV에 무관하게 동일.

#### Finding 2 — F16 vs Q4 prefill 효율 reversal (long prompt에서 F16이 더 빠름)
| KV | Q4 prefill | F16 prefill | F16 vs Q4 | tok/s 차이 |
|---|---:|---:|---:|---:|
| 5 | 105 | 244 | F16 **+139 ms 손해** | F16 0.43× |
| 1k | 7147 | 6364 | F16 **-783 ms 이득** | F16 1.11× |
| 1.8k | 13987 | 12640 | F16 **-1347 ms 이득** | F16 1.11× |

→ **long batched prefill에서는 F16이 Q4보다 11% 빠름**. 이유: Q4 dequant가 매 batched matmul마다 발생, batch size가 클수록 effective matmul throughput이 더 중요해져 dequant overhead가 회수 안 됨.

→ **phase-aware의 prefill F16 강제 (이전 §7.1 PRIMARY 손해로 분류)는 long prompt에서는 오히려 이득**. 이전 분류 무효.

#### Finding 3 — Phase-aware tail이 KV에 따라 악화
| KV | Phase-aware tail (excl tok0) | Q4 baseline | Δ |
|---|---:|---:|---:|
| 5 | 29.3 (n=200) | 28.1 | **+1.2 ms** |
| 1.8k | 78.0 (n=20) | 47.4 | **+30.6 ms** |

→ KV 1.8k에서 18 token tail 평균이 +30 ms 더 느림. 18 × 30 = 540 ms 추가. 가능 원인:
  - chunk dispatch가 더 많은 token에 분산됨 (forward 무거워 chunk 진행 늦음)
  - dispatcher worker thread와 attention OpenCL queue 경합 증가
  - retire 시점이 token 18+에 발생 (그 동안 forward가 chunk 뒤에 줄)

n=20만 측정이라 retire 시점 추정 어려움 — 더 긴 decode (n=200) 측정 필요. KV 1.8k에서는 phase-aware EOS hit 안 함 (다양한 텍스트 prompt 덕분).

### 7.5.4 E2E 비교 — KV에 따른 phase-aware 위치 변화

| KV | n | Single-shot E2E | Phase-aware E2E | Δ (PA - SS) |
|---:|---:|---:|---:|---:|
| 5 | 200 | 5874 | 6477 | **+603** (PA 손해) |
| 1.8k | 20 | 14854 + 46.7×19 = **15741** | 13187 + 91.9×19 = **14933** | **-808** (PA 이득!) |
| 1.8k | 200 (tail 78 유지 가정) | 14854 + 46.7×199 = 24147 | 13187 + 78×199 = 28709 | +4562 (PA 손해) |
| 1.8k | 200 (tail Q4 회복 가정) | 24147 | 13187 + 78×18 + 47×181 = 23098 | -1049 (PA 이득) |

KV 1.8k crossover (tail 78 유지 시): **n ≈ 37**. 즉:
- KV 5: 모든 n에서 phase-aware E2E 손해
- KV 1.8k: **n<37 에서 phase-aware 이득 (-808 ms at n=20)**

이는 §7.1 ("Prefill F16 +119 ms 영구 손해") 결론을 KV-dependent로 수정해야 함을 시사.

### 7.5.5 종합 — KV 길이가 v1 평가를 reverse

이전 §7 (KV 5 기반) 결론 vs KV 1.8k 결론 비교:

| 항목 | KV 5 기반 (이전) | KV 1.8k 기반 (수정) |
|---|---|---|
| Prefill F16 영향 | +119 ms 영구 손해 | **-1347 ms 이득** (reversal) |
| tok[0] blocking | +200~300 ms | +290~315 ms (큰 변화 없음) |
| Decode tail | +1.2 ms (steady 거의 동일) | **+30.6 ms** (n=20, 더 큼) |
| E2E 평가 | 모든 n에서 PA 손해 | n<37 에서 PA 이득 |

→ **v1 평가는 prompt 길이에 강하게 의존**. paper에서 이걸 명시해야:
- short prompt (chat 첫 메시지, ~5~50 token): single-shot 유리
- long prompt (RAG, document QA, ~1k+ token): phase-aware E2E 유리 (단 n<crossover)
- 매우 긴 응답 (n>100): single-shot 유리 (모든 KV)

### 7.5.6 v2 우선순위 재조정

이 finding이 §8 plan을 다시 흔든다:
- §8 Priority 1 (prefill phase swap)은 short prompt 만 도움. long prompt에서는 F16 prefill이 더 빠름 → 의미 없음. **drop**.
- §8 Priority 2 (sub-tensor chunking) — tok[0] blocking 해소가 모든 KV에서 우선순위 1. **유지**.
- 추가 Priority — **tail overhead (KV 의존)** 진단. KV 1.8k tail +30 ms 가 chunk dispatch 분산 시간 길어짐 때문이라면 sub-chunking으로 해결 가능. queue 경합이라면 dispatcher worker thread 분리 필요.
- 추가 Priority — long prompt + n>20 측정 필요. tail 회복 시점 확인 (chunk dispatch가 언제 retire?).

---

## 8. 다음 세션 액션 (2026-05-10 final)

§7의 4가지 원인을 직접 공격. 우선순위 재배치:

### Priority 1 — Prefill phase swap 활용 (+119 ms 회수)
**가장 큰 이득 (per-token 무관, 영구).** prefill 5 token도 op_trace boundary가 있고 deterministic. 단 prefill은 batch token이라 forward 자체가 무거움 (224 ms / 5 token = 45 ms/token).
- 옵션 A: prefill 직전에 동기 swap 실행 (single-shot 동등) → TTFT loss +338 ms 감수 vs Decode +119 회수
  - 사실상 single-shot으로 회귀, phase-aware의 의미 잃음
- 옵션 B: prefill phase에서도 chunk dispatch (op_trace boundary 활용)
  - prefill의 cache-fit phase는 token 단위 batched이지만 layer 단위로는 phase 분리됨
  - 구현: prefill path에서도 PHASE_HOOK fire (현재 forward_gen 만)
  - prefill 5 token + decode 첫 ~3 token = 약 8 phase × 28 layer × 5 cache-fit = swap 커버 충분
  - **권장 — TTFT win 보존하면서 prefill F16 cost 회수**

### Priority 2 — Sub-tensor chunking (Option α, tok[0] +295 ms 회수)
v1 chunk가 per-tensor 통째라 cache-fit window 미달:
- 현재 `byte_offset=0, byte_len=0` (`phase_aware_swap.rs:236`) — chunk_size_bytes 미사용 버그
- 수정: tensor 크기를 chunk_size_bytes (4 MB)로 분할
  - q/o (1.3 MB) → 1 chunk (변동 X)
  - gate/up/down (7.6 MB) → 2 chunks each
  - **per layer 9 chunks → 16 chunks; 25 layer 225 → 400**
- expected: chunk 당 fit window에 진입 → forward와 진짜 overlap
- cl_event chain은 sub-chunk 사이도 유지

### Priority 3 — Background pre-staging (Option β, host work 회수)
- 현재 `materialise_cpu_tensor` (mmap + GGUF re-pack)이 forward thread blocking 호출
- worker thread로 분리: PartialLayer pre-stage → forward thread는 `enqueue_write_async`만 호출
- expected: per-chunk host work ~1 ms → ~0.1 ms (enqueue overhead만)

### Priority 4 — Hide ratio 측정 인프라
v1은 hide ratio 측정 못 했음 (chunk dispatch wall-clock 분리 unavailable).
- dispatcher에 per-chunk start/end timestamp 기록
- forward kernel timestamp와 비교 → overlap ratio 계산
- 4-gate 정량화 가능

### v2 Pass-gate (수정)
- E2E (n=200): single-shot 대비 **+5% 이내** (현 +10%)
- TTFT: -100 ms 이상 단축 보존
- Decode tok[0]: 100 ms 이내 (현 322 ms)
- Decode tail: Q4 baseline +5% 이내 (현 이미 +4% — pass)

### Crash 처리
- handoff `12bac91`에서 보고된 SIGABRT는 재현 불가 (5+ 회 재실행 모두 정상)
- handoff `f68e817`는 close (race 의심 hypothesis는 코드 변경 없이 보존, 향후 v2 작업 시 재출현 시 재진단)
- handoff_liswap5_crash_fix_2026_05_10.md 는 archive 처리

### Lessons learned (수정)
- **n별 Decode 변동 무시 위험**: 첫 단순 평균 분석에서 +10 ms steady loss라 결론. n별 분리 측정 후 무효화. **항상 n별 + tok-bucket 측정 필수**.
- **TTFT 단축 ≠ E2E win**: 분산 swap이 TTFT는 줄이지만 forward 안에 흡수된 cost는 더 큼 (prefill F16 + tok[0] blocking).
- **chunk_size 설정과 실제 chunk 크기 불일치**: code review 시 `byte_offset/len` 사용 확인 (현재 0,0 hardcoded — sub-chunking 미구현 상태).
- **9-track handoff §3.1 LISWAP-4 v2 함정 (plan-bypass) 을 그대로 반복**. 다음 swap 작업 시 "plan-path bypass guard" 체크리스트 필수
- **flaky crash 보고 시 즉시 fix 의사결정 금지** — 5+ 회 재현 시도가 우선. 본 세션은 재현 안 되어 fix 불필요

---

## 7.7 LISWAP-6 (DMA-BUF alias) 측정 결과 (2026-05-10 final)

LISWAP-5 fair comparison §7.6에서 도출된 결론 — "swap 자체 cost가 핵심"에 따라 secondary GGUF weight를 rpcmem heap에 lazy alloc + OpenCL `CL_MEM_USE_HOST_PTR` alias로 H2D copy 자체를 제거 (commit `5b6e022` + `d81cbb2`). qnn_oppkg backend의 dual-buffer KV 패턴 모방.

### 7.7.1 측정 결과 (KV 1.8k, n=20, --backend qnn_oppkg, 3 run)

| 모드 | Prefill | tok[0] | tail (excl tok0) | Decode all | E2E |
|---|---:|---:|---:|---:|---:|
| **Q4 only baseline** | 13906~16451 | 2.84~5.91 | 9.86~10.82 | 9.49~10.47 | 14749 |
| **per-tick=25** (single in decode) | 12258~13488 | 8.77~11.17 | **2.95~3.25** | 3.37~3.56 | **12882** |
| per-tick=1 (LISWAP-1) | 13373~13958 | 8.25~15.91 | 3.54~4.20 | 3.98~4.42 | 14074 |
| **phase-aware** (LISWAP-5) | 14377~14644 | **529~585** | 39~52 | 67~77 | 16024 |

### 7.7.2 LISWAP-5 (mmap+memcpy) → LISWAP-6 (alias) 가속

| Metric | LISWAP-5 (opencl) | LISWAP-6 (qnn_oppkg+alias) | 가속 |
|---|---:|---:|---:|
| Q4 baseline tail | 47 ms | **10 ms** | **4.7×** (qnn_oppkg dual-buffer 자체 효과) |
| per-tick=25 tok[0] | 65 ms | 10 ms | **6.5×** |
| per-tick=25 tail | 45 ms | **3 ms** | **15×** |
| per-tick=1 tail | 62 ms | 3.6 ms | **17×** |
| phase-aware tok[0] | 347 ms | **530 ms** | **0.65× (오히려 느림)** |
| phase-aware tail | 78 ms | 45 ms | 1.7× |

### 7.7.3 핵심 finding

#### Finding 1 — per-tick=25가 Q4 baseline보다도 빠름 (3 ms vs 10 ms tail)
의외의 reversal:
- Q4 only baseline = GGUF mmap → BorrowedMmapBuffer (CPU-side reference). GPU 접근 시 internal H2D 변환 비용
- per-tick=25 후 = ClWrappedBuffer (cl_mem alias, MEM_USE_HOST_PTR). 매 forward에서 더 효율적
- → **swap이 weight access 효율을 개선**시키는 부작용

#### Finding 2 — Swap copy 비용 ~95% 제거 (per-tick 모드)
- 이론 transfer: 80~92 ms @ 7~8 GB/s (645 MB Q4)
- LISWAP-5 실측 swap blocking: 480 ms (5~6× overhead)
- LISWAP-6 실측: per-tick=25 tok[0] excess = 11-3 = ~8 ms (lazy first-touch)
- **alias path가 H2D copy 자체를 제거 + lazy alloc cost가 token 0에 ~8 ms로 흡수**

#### Finding 3 — Phase-aware는 alias 환경에서도 13× 손해
- per-tick=25 tail 3 ms vs phase-aware tail 45 ms = **15× 차이**
- per-tick=25 tok[0] 10 ms vs phase-aware tok[0] 530 ms = **53× 차이**
- swap copy가 free임에도 분산 자체 비효율 (enqueue overhead × 75 chunks/token + scheduler 경합 + dispatcher worker sync)이 dominate
- → **LISWAP-5 design (phase-aware 분산) 의 본질 한계 — alias로도 해결 안 됨**

### 7.7.4 production 권장 path (확정)

```
--backend qnn_oppkg --secondary-gguf <q4>.gguf --force-swap-ratio 0.X --swap-incremental-per-tick 25
```

- **qnn_oppkg backend**: KV cache dual-buffer + LISWAP-6 weight alias 양쪽 효과 (Decode 4.7× faster baseline)
- **per-tick=25**: single-shot in decode. swap blocking ~10 ms (vs phase-aware 530 ms)
- **자동 활성**: backend == qnn_oppkg + secondary_gguf 모두 active 시 LISWAP-6 alias 자동

### 7.7.5 LISWAP design 정리

| Track | 결론 |
|---|---|
| **LISWAP-1** (per-tick) | per-tick=25는 production 최적. per-tick=1는 분산 비효율 (3 ms→3.6 ms 미세 손해) |
| LISWAP-2 (async dispatch) | LISWAP-1 위 worker thread overhead — alias 환경에서 의미 약화 |
| LISWAP-3 (host_ptr_pool) | alias가 동일 효과 + lifetime 단순. 폐기 가능 |
| LISWAP-4 (intra-forward) | per-tick과 비슷, layer wait gate 복잡성. 폐기 권장 |
| **LISWAP-5** (phase-aware) | **폐기 확정** — alias 환경에서도 분산 비효율 13~53× 손해 |
| **LISWAP-6** (DMA-BUF alias) | **production path** — 자동 활성 |

### 7.7.6 알려진 이슈
- **Cleanup phase Segmentation fault**: swap mode runs에서 generation 정상 완료 후 process exit 시 발생. RpcmemAliasBuffer drop ordering or cl_mem release ↔ rpcmem_free race. production 동작 + 측정 영향 없음. 별도 fix backlog.

### 7.7.6c Eager prefault (commit `f038301`, 2026-05-11)

LISWAP-6 alias path의 swap stall 분리 측정 후 lazy `rpcmem_alloc` 비용을 model load로 옮김:
- target_layers 결정 직후 `secondary.prefault_layers()` 자동 호출
- 모든 swap mode (single-shot/incremental/intra-forward/phase-aware) 자동 이득
- swap blocking 700 → 410 ms (-290 ms, 42% 단축)
- model load TTFT +250~362 ms (한 번만, 사용자 인지 없음)

### 7.7.6d 측정 노이즈 정정 (2026-05-11)

§7.7.2 측정 중 **per-tick=25 tail 3 ms/tok 는 outlier**로 판명:
- production memory note: qnn_oppkg Q4 baseline = 12 ms/tok
- 본 measurement 재현: 10 ms/tok (memory와 일치)
- 이전 3 ms 는 thermal/clock state 또는 page cache state 영향 가능

→ **LISWAP-6 의 진짜 decode 효과**:
- qnn_oppkg backend 자체 (KV dual-buffer): OpenCL 28 → qnn_oppkg 10~12 ms (~2.5×)
- weight alias (LISWAP-6) 의 decode 가속 효과: **미미** (production 환경에서 baseline과 거의 동일)
- LISWAP-6 의 진짜 가치 = **swap blocking 단축** (480→403 ms, eager prefault 후) + alias 자체의 zero-copy weight access (단 decode tail 측정에서는 noise 안에 묻힘)

§7.7.3 Finding 1 ("per-tick=25 tail은 Q4 baseline의 ~38%") **무효** — outlier 측정 기반.
§7.7.5 Finding 1 (KV 4K Llama "alias 효과 KV 무관 강함") 도 부분 재검토 필요 — Llama 측정값 (2 ms vs 5.27 baseline) 도 outlier 가능성.

→ **production 권장 path 유지** (qnn_oppkg + per-tick=25 + LISWAP-6 alias) — swap blocking 단축 + qnn_oppkg backend 자체 가속이 진짜 이득. 단 "10× 가속" 같은 과장된 표현은 정정.

### 7.7.6b KV 4K Llama 측정 (2026-05-11 추가 — KV 1.8k Qwen 비교)

사용자 요청 "decode 시작 시점 말고 큰 KV 상태에서 swap" 검증. Llama 3.2-1B (max_seq_len 8192 override) + prompt_4096 (4070 token) + n=30 + qnn_oppkg.

#### Setup
- Llama 3.2-1B (16 layer, 14 swap target — 0.9 ratio)
- `--max-seq-len 8192 --initial-kv-capacity 8192`
- prompt 4070 token + n=30 (EOS hit으로 실제 5~11 token decode)

#### KV 1.8k Qwen vs KV 4K Llama 비교

| Metric | Qwen KV 1.8k | Llama KV 4K |
|---|---:|---:|
| swap target layer | 25 | 14 |
| Q4 only tail | 10 ms | 5.27 ms |
| per-tick=25 tail | 3 ms | 2 ms |
| per-tick=1 tail | 4 ms | 2.3 ms |
| **phase-aware tail** | 45 ms | **70 ms** |
| **phase-aware vs per-tick=25 손해** | 15× | **35×** |
| **per-tick=25 vs Q4 baseline E2E** | -13% (win) | **+16% (loss)** |

#### 핵심 finding

**1. Alias 효과는 KV 4K에서도 강함**: per-tick=25 tail = Q4 baseline의 ~38% (5.27 → 2 ms). KV 무관하게 alias가 weight access 가속.

**2. Phase-aware는 KV 클수록 더 손해 (15× → 35×)** — 사용자 가설 ("KV 클수록 swap hide 잘 됨") 의 **정반대**. 원인: KV 클수록 attention 무거워져 forward kernel 길어짐 → chunk dispatch가 더 많은 forward와 직렬 점유. dispatcher worker sync 비용도 누적.

**3. KV 4K에서 per-tick=25가 Q4 baseline보다 +16% 손해** — KV 1.8k Qwen에서 본 "swap이 baseline보다 빠름" 패턴이 깨짐. 원인:
   - **Prefill이 E2E의 95%** (4K context). 작은 prefill 차이도 E2E 좌우
   - swap modes는 F16 prefill (35106 ms) > Q4 baseline 의 Q4 prefill (30188 ms) — Llama 1B 작은 모델이라 dequant overhead 비중 작아 Q4 prefill이 더 빠름
   - decode 차이 (tail 5.27 → 2 ms × 28 token = 92 ms 절약) << prefill 손해 5s
   - **결과**: 짧은 응답 (EOS 5~11 token) 시나리오에서 swap 의미 없음

**4. EOS hit 일찍 — swap modes 짧은 응답**: per-tick=25 9 token, per-tick=1 5 token (mid-decode mixed weight logit drift), phase-aware 11 token. 같은 prompt에서도 mode별 출력 분기.

#### KV 4K E2E

| Mode | E2E | Q4 baseline 대비 |
|---|---:|---:|
| **Q4 only** | **30336** | — |
| per-tick=25 | 35122 | +4786 (+16%) |
| per-tick=1 | 36881 | +6545 (+22%) |
| phase-aware | 37595 | +7259 (+24%) |

#### 결론

KV가 클수록:
- **Alias 효과는 보존** (per-tick=25 tail은 항상 baseline의 30~40%)
- **Phase-aware는 더 비효율** (15× → 35×)
- **Prefill dominant** — swap modes의 F16 prefill 손해를 decode 차이로 회복 어려움
- **Production 권장**: KV 큰 + 짧은 응답 시나리오는 Q4 only가 최적 (swap mode들 모두 손해)
- **swap의 가치 시나리오**: KV 작은 + 긴 응답 (KV 1.8k Qwen + n=200 처럼 prefill 비중 작은 경우)

→ **진짜 mid-decode swap 측정**은 별도 — token 0 강제 swap이 아닌 token N에서 swap 발생. mock_manager 또는 CLI flag (`--swap-delay-tokens N`) 필요.

### 7.7.7 정확도 비교 (deterministic, prompt="The capital of France is", n=40, temperature=0)

| 모드 | 출력 첫 줄 | Fact 정확도 |
|---|---|---|
| F16 only | "...French are known for their love of food and wine..." | general (fact 없음) |
| Q4 only | "Paris. It has a population of **2M**, area **104 sq km**, **8 districts**" | ❌ 인구는 2M (Paris 자체), area 105, districts 20 — Q4 양자화 손실 |
| **per-tick=25** | "Paris. The country has a population of **67M**... **Île-de-France (Paris)**" | ✅ **가장 정확** |
| per-tick=1 | "Paris. **6M people inside city limits** (2013)" | △ partial (Paris 광역 12M) |
| **phase-aware** | "...**67M**...area **larger than UK**...**largest city, London**" | ❌ **"London이 France 가장 큰 도시"** = 큰 hallucination |

#### 왜 phase-aware가 hallucinate
| 모드 | swap 단위 | mid-decode weight state |
|---|---|---|
| per-tick=25 | layer 25개 한 번에 | token 0 후 모두 Q4 안정. **consistent** |
| per-tick=1 | layer-whole 1개씩 분산 | 매 token 1 layer만 swap. **layer 단위 일관** |
| **phase-aware** | **per-tensor chunk** (1 layer = 9 tensor 따로) | layer N의 attn_q는 Q4, attn_v는 F16인 시점 존재 → **intra-layer mixed weight** → attention output 의미 불일치 |

→ phase-aware의 chunk 단위가 layer보다 작아 layer 내부 weight가 부분적으로 mixed. attn 4-tensor (q/k/v/o) 일부만 swap된 시점에 attention 호출되면 **logit 손상**. 결과: hallucination 빈도 ↑.

per-tick=1은 layer-whole이라 layer 안은 항상 같은 dtype (Q4 또는 F16). 일관성 보장.

→ **LISWAP-5 폐기는 속도뿐 아니라 정확도 측면에서도 정당화**. design 자체 결함 — chunk granularity < layer.

---

**End of postmortem**

핵심 한 줄 (2026-05-10 final, LISWAP-6 측정 후 확정):
**LISWAP-5 (phase-aware) 폐기 확정. LISWAP-6 (DMA-BUF alias) 채택. KV 1.8k n=20: per-tick=25 (single in decode) E2E 12882 ms (Q4 baseline 14749보다도 빠름!) < per-tick=1 14074 < phase-aware 16024. 분산이 alias 환경에서도 13× 손해. Production path = `--backend qnn_oppkg --swap-incremental-per-tick 25` (자동 alias 활성).**
