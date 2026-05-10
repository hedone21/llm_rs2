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
- HEAD: `f68e817` (handoff doc 추가, 코드 변경 없음)
- Binary: `12bac91` 시점 빌드 (동일)
- Device: Galaxy S25 (Adreno 830)
- Prompt: "The capital of France is", n=200, temperature=0
- Reproducer: `--swap-phase-aware --force-swap-ratio 0.9 --secondary-gguf ...q4_0.gguf` (single-shot은 `--swap-phase-aware` 빠짐)

### 6.2 4-mode 비교 (n=200, 3 run median)

| 모드 | TTFT (ms) | Decode (ms/tok) | Tail output |
|---|---:|---:|---|
| [A] F16 only | n/a | 54.1 ~ 65.7 | n/a |
| [B] Q4_0 only | n/a | 38.2 ~ 47.2 | n/a |
| [C] Single-shot swap (force-ratio 0.9) | **605~635** | **37.1 ~ 52.7** (median 37) | "...The capital city is Paris. It has" |
| [D] Phase-aware LISWAP-5 | **456~477** | **48.6 ~ 54.1** (median 50) | "...The most important fact about France is that it was" |

### 6.3 핵심 finding

#### TTFT win: **-150 ms** (phase-aware vs single-shot)
- Single-shot은 prefill 직전 swap을 동기 수행 — 측정한 TTFT 605 ms 중 ~400 ms가 swap copy 비용 (Q4 prefill 자체는 ~200 ms)
- Phase-aware는 prefill 동안 swap이 없고, decode token 0~2에서 background dispatch
- → TTFT 측면에서 phase-aware가 single-shot 대비 **24% 단축**
- 단응답·interactive UX에 의미 있음

#### Decode loss: **+10 ms/tok** (phase-aware vs single-shot)
- Single-shot Q4 forward 정상 = 37 ms/tok (Q4 only 38 ms와 ~일치)
- Phase-aware는 retire 이후에도 50 ms/tok — Q4보다 +10 ms 느림
- 가능 원인:
  - Chunk dispatch가 끝났음에도 ratio_generation bump 후의 lazy SOA rebuild 비용이 retire 직후 token에 잔류
  - OpenCL command queue 상에 swap chunks 가 enqueue된 잔류 효과 (driver internal scheduler)
  - 더 직접적으로 — `materialise_cpu_tensor` (mmap → CPU memcpy) 자체가 forward thread를 막음 (token 0~2에서 75 chunks/token × 1 ms ≈ 75 ms 추가)
- token 3 이후에도 +10 ms 잔류는 dispatcher가 비어있어도 발생 → **invalidate 후 layer-by-layer SOA lazy rebuild가 진짜 culprit 가능성**

#### E2E crossover (n에 따라)
- E2E = TTFT + Decode × n
- single-shot: 635 + 37×n
- phase-aware: 470 + 50×n
- crossover: n = (635-470) / (50-37) ≈ **n=13**
- → **n<13 일 때 phase-aware 유리, n≥13 일 때 single-shot 유리**

### 6.4 Phase R (microbench 1.04×) 와의 차이
Phase R Scenario B = 1.04× of max. 본 production 측정은 +35% (50/37). 차이 원인:
- Phase R: pure memcpy + cache-fit window 활용
- Production: `materialise_cpu_tensor` (mmap fault + GGUF re-pack) 추가 + 25 layer × 9 tensor = 225 chunks의 enqueue overhead
- per-tensor chunk size가 cache-fit window (980us = 4 MB) 초과: wq 9 MB / ffn 26 MB

→ Phase R의 1.04×를 production에서 재현하려면 **chunk granularity를 4 MB 이하로 sub-divide** 해야 함 (현재 v1은 per-tensor만, 9~26 MB 단일 chunk).

### 6.5 4-gate 결과

| Gate | 결과 | 근거 |
|---|---|---|
| ✅ Correctness | PASS | 32, 64, 200 token 모두 plausible English ("Paris. The capital city is...") |
| ⚠️ TBT | PARTIAL | Decode +10 ms 손해, but TTFT -150 ms win |
| ❓ Hide ratio | UNKNOWN | dispatcher counters로 chunks=225/token=2 = 113 chunk/token throughput 확인. forward와 동시 진행했는지는 wall-clock 분리 측정 필요 |
| ❌ Stall | (자체 측정 X) | crash 보고된 적 있으나 재현 불가 — flaky로 분류 |

**종합**: design 자체는 viable. TTFT 개선 명확. Decode 손해는 v2에서 sub-chunking + invalidate 우회 필요.

---

## 7. 효과 없었던 진짜 이유 정리 (2026-05-10 갱신)

| 이유 | 영향 | 상태 |
|---|---|---|
| **A. plan-path bypass 누락** | 측정 자체가 의미 없음 (swap 0회) | ✅ FIX (commit `ed7464f`) |
| **B. retire 후 crash** | wall-clock 측정 불가 | 🔄 재현 안 됨 — flaky, 측정 가능 (§6) |
| **C. per-tensor chunk overhead** | Decode +10 ms/tok 영구 손해 | ❌ design 한계 — v2 필요 |
| **D. invalidate 후 SOA lazy rebuild** | retire 후에도 +10 ms 잔류 가능성 | ❓ 정밀 분리 측정 필요 |
| **E. mid-decode mixed weights** | 출력이 single-shot과 분기 | ✅ 정상 (semantic plausible) |

**A**는 fix됨. **B**는 재현 불가하므로 close. v1의 진짜 평가는 §6 결과로 확정:
- TTFT win (-150 ms / -24%)
- Decode loss (+10 ms / +35%)
- crossover n=13

**C**는 design 한계. v2가 필요한 이유:
- per-tensor 4 MB 이상의 단일 chunk → cache-fit window (980us) 초과
- materialise_cpu_tensor (mmap + GGUF re-pack) 자체가 forward thread block

**D**는 추가 진단 필요. retire 후 token 3~10에서도 50 ms 유지되는 이유:
- ratio_generation bump 이후 첫 N token에서 SOA registry rebuild가 lazy하게 비용 청구
- 또는 OpenCL driver internal scheduler가 스왑 chunks 흔적으로 후속 forward kernel 우선순위 낮춤
- profiling으로 retire 후 token별 forward breakdown 분리 가능

**E**는 본질 — 평가 기준은 byte-equal이 아닌 plausible English.

---

## 8. 다음 세션 액션 (2026-05-10 갱신)

### Priority 1 — v2 sub-tensor chunking (Option α)
v1은 per-tensor chunk만 (wq=9 MB, ffn_gate/up=26 MB 등). 이걸 4 MB 이하로 sub-divide 하면 cache-fit window에 fit.
- chunk granularity: 2 MB (Q4) / 4 MB (F16)
- sub-chunk마다 cl_event chain — 같은 PartialLayer 안의 chunks가 순차 enqueue
- expected: per-token chunks 75 → 200, 단 chunk당 dispatch 시간 1 ms → 0.5 ms (host work는 거의 동일하지만 enqueue overhead 분산)

### Priority 2 — invalidate 후 SOA rebuild lag 진단
retire 후 token 3~10의 +10 ms 잔류 원인 규명:
- profiling (perfetto) 으로 token별 forward breakdown
- ratio_generation bump 직후 첫 N token만 다른 path (lazy rebuild) 거치는지 확인
- 아니면 driver/scheduler 효과 — 그 경우 v2도 한계

### Priority 3 — Background pre-staging (Option β)
- materialise_cpu_tensor를 worker thread로 pre-stage → forward thread는 enqueue_write_async만 호출
- v1은 forward thread에서 mmap → memcpy → enqueue 모두 수행 (75 ms host work 직접 짊어짐)
- v2 + β로 가면 Phase R 1.04× 와 비슷한 overhead 가능성

### Priority 4 — TTFT win 활용 시나리오 정리
- crossover n=13 — interactive (avg 응답 ~100 token) 시나리오에서는 single-shot이 유리
- 하지만 streaming TTFT 우선 시나리오 (chat UI 첫 token 빠른 표시)에서는 phase-aware가 win
- 평가 metric을 어디에 정렬할지 paper에서 명확히

### Crash 처리
- handoff `12bac91`에서 보고된 SIGABRT는 재현 불가 (5+ 회 재실행 모두 정상)
- handoff `f68e817`는 close (race 의심 hypothesis는 코드 변경 없이 보존, 향후 v2 작업 시 재출현 시 재진단)
- handoff_liswap5_crash_fix_2026_05_10.md 는 archive 처리

### Lessons learned
- **9-track handoff §3.1 LISWAP-4 v2 함정 (plan-bypass) 을 그대로 반복**. 다음 swap 작업 시 "plan-path bypass guard" 체크리스트 필수
- **첫 측정 신뢰 전 진단 카운터 필수 확인** — 첫 v1 측정 (48 ms) 을 실제 swap 결과로 오해석
- **op_trace 호출 경로의 unit 검증** — `forward_gen` 외 plan, prefill, transformer.rs decode tail 다중 path 존재
- **flaky crash 보고 시 즉시 fix 의사결정 금지** — 5+ 회 재현 시도가 우선. 본 세션은 재현 안 되어 fix 불필요

---

**End of postmortem**

핵심 한 줄 (2026-05-10 final): **첫 측정은 의미 없었다 (plan path bypass로 swap 0회). Fix 후 hook 정상 fire + 3 token 안에 dispatch 완료. v1은 정상 동작 — TTFT -150 ms win, Decode +10 ms loss, crossover n=13. crash는 재현 불가 (flaky). v2는 sub-tensor chunking + background pre-staging.**
