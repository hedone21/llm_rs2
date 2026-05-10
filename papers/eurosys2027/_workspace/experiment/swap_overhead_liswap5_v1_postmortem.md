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

## 6. 진짜 v1 평가 (가능한 만큼)

위 fix로 hook이 정상 fire하지만 retire 후 crash로 wall-clock 측정 불가. 다만 token 0~10 동안의 데이터로 일부 추정 가능:

### 6.1 Chunk dispatch overhead 추정
- 95 ms/token forward (F16) 동안 ~75 chunks dispatch
- 각 chunk = `materialise_cpu_tensor` (mmap → cpu copy) + `enqueue_write_async`
- 75 chunks × ~1 ms host work = **~75 ms host overhead per token** (forward thread 막음)
- → 95 ms baseline + 75 ms overhead = **170 ms/token 예상** (가정)

### 6.2 Phase R 비교
Phase R Scenario B는 1.04× of max = 4% overhead. 본 production은:
- materialise 자체가 호스트 mmap memcpy → 단순 std::memcpy 보다 무거움 (page fault, GGUF format 변환)
- per-tensor chunk이 cache-fit window (980us) 보다 큰 경우 대부분 — wq 9 MB = 3 ms, ffn 26 MB = 8.7 ms

→ **Phase R의 1.04×는 production에 그대로 안 옮겨짐**.

### 6.3 추정 wall-clock (crash가 없다고 가정)
- baseline F16 forward: 53 ms
- + chunk dispatch overhead: ~75 ms (token 0~2까지만, 그 후 0)
- swap 완료 후 forward = Q4 forward 26 ms
- Total decode for 32 tokens 가정:
  - token 0~2: 3 × (53 + 75) = 384 ms
  - token 3~31: 29 × 26 = 754 ms
  - = 1138 ms decode
- + TTFT 343 ms = **1481 ms E2E** (single-shot 1293 보다 +14% 느림)

→ 만약 v1이 정상 작동했다면 single-shot보다 **약간** 느리지만 LISWAP-1 (+33%) 보다 나았을 가능성. 단 이 추정은 crash 없이 retire 후 정상 forward 가정.

---

## 7. 효과 없었던 진짜 이유 정리

| 이유 | 영향 | 책임 |
|---|---|---|
| **A. plan-path bypass 누락** | 측정 자체가 의미 없음 (swap 0회) | generate.rs:5371 가드 (이번 fix됨) |
| **B. retire 후 crash** | 정확한 v1 wall-clock 측정 불가 | commit + remap 경로 race (다음 세션 fix 필요) |
| **C. per-tensor chunk overhead** | 추정상 token당 +75 ms host work | dispatcher v1 design 한계 |
| **D. mid-decode mixed weights** | 출력이 single-shot과 분기 (b1 첫 측정 [C]) | swap 자체의 본질 — 평가 기준 재정의 필요 |

**A는 단순 버그**, B-D는 design 차원.

A를 fix한 지금 (commit `ed7464f`), B를 해결하지 않으면 v1의 진짜 성능을 볼 수 없다. B는 ArcSwap commit + workspace 동기화 + release worker race 의 어딘가 — 자세히 봐야 한다.

C는 design 차원 — sub-tensor chunking + worker thread pre-staging이 필요할 수 있다. 그런데 v1을 fully measurement하기 전엔 C가 진짜 문제인지 알 수 없다.

D는 기능적 본질 — swap이 decode 중간에 일어나면 출력이 달라지는 건 정상.

---

## 8. 다음 세션 액션

### Priority 1 — Crash fix (B)
- IntraForwardSwapHook::finalize 패턴과 비교 (intra_forward_swap.rs:547+)
- AsyncSwapDispatcher::drain 후 추가 synchronize 필요한지 확인
- ratio_generation bump 시점 vs forward thread workspace cl_mem 사용 시점 race 분석
- partition_ctx 무효화 race 가능성

### Priority 2 — Wall-clock 재측정 (B fix 후)
- n=32, n=200 양쪽
- 4-gate (correctness, TBT, hide ratio, stall) 재실행
- v1이 진짜 fail인지 (Phase R 안 옮겨짐) vs marginal pass인지 판정

### Priority 3 — v2 검토 (B 후에 데이터 보고)
- Option α (sub-tensor chunking) 또는 β (background pre-staging) 우선순위 데이터 기반 결정

### Lessons learned
- **9-track handoff §3.1 LISWAP-4 v2 함정을 그대로 반복함**. 다음에 swap 쪽 작업할 때 "plan-path bypass guard" 체크리스트 필수.
- **measurement 신뢰 전 항상 진단 카운터 확인** — 첫 v1 측정 (48 ms) 을 실제 swap 결과로 오해석.
- **op_trace 호출 경로의 unit 검증 필요** — `forward_gen` 외 다른 path 존재 (plan, prefill, transformer.rs decode tail).

---

**End of postmortem**

핵심 한 줄: **첫 측정은 의미 없었다 (plan path bypass로 swap 0회). Fix 후 hook 정상 fire하고 3 token 안에 dispatch 완료하지만, retire 후 SIGABRT로 진짜 wall-clock 측정은 다음 세션 디버그 후 가능.**
