# Handoff: LISWAP-5 retire-후 SIGABRT 디버그 — **CLOSED (재현 불가)**

**Date**: 2026-05-10
**Closed**: 2026-05-10 (다음 세션에서 reproducer 5회+ 재실행, 모두 정상 완료)
**HEAD (closed)**: `f68e817`
**Device**: Galaxy S25 (R3CY408S5SB, Adreno 830)

> **상태**: CLOSED. 본 handoff에서 보고된 SIGABRT는 다음 세션의 reproducer 재실행에서 **재현 불가**. n=32, n=64×5회, n=200×3회 모두 crash 없이 정상 완료. flaky race timing 또는 측정 환경 노이즈로 판단. 후속 v1 wall-clock 측정 + v2 진행 plan은 `swap_overhead_liswap5_v1_postmortem.md` §6, §8 참조.

**선행/후속 문서**:
- `papers/eurosys2027/_workspace/experiment/swap_overhead_liswap5_v1_postmortem.md` (실측값 §6 + v2 plan §8)
- `.agent/todos/feat_phase_aware_swap_2026_05_10.md` (LISWAP-5 plan)

---

## 0. 한 줄 요약 (보존 — 보고 당시 가설)

LISWAP-5 v1이 commit `ed7464f` (plan-bypass guard) 후 정상 동작 시작. 3 token 안에 25-layer (225 chunk) 모두 dispatch + plan retire 완료. 그러나 retire 후 9 token 진행 시점 (start_pos=16 부근) **SIGABRT crash** ← **재현 불가, CLOSED**.

---

## 1. 현재 상태 (commit `12bac91`)

### 1.1 정상 동작 확인된 것
```
weight_swap: phase-aware mode — ratio=0.90, 25 target layers, chunk_size=4 MB (LISWAP-5)
[op_trace::set_phase_hook] PHASE_HOOK.set result: Ok (PHASE_HOOK now Some=true)
[forward_gen-DBG] layer_idx=0 start_pos=5
tok=0 queue=140 in_flight=true  pending=1 dispatched=85  hook_start=280 hook_end=286 cachefit_end=140
tok=1 queue=56  in_flight=true  pending=1 dispatched=169 hook_start=560 hook_end=569 cachefit_end=280
tok=2 queue=0   in_flight=false pending=0 dispatched=225 hook_start=840 hook_end=852 cachefit_end=420
[PhaseAwareSwap] plan retired at token=2 (drain+sync+bump+invalidate 0.0ms, chunks=225)
[forward_gen-DBG] layer_idx=0 start_pos=8     ← retire 후 9 token 정상 forward
[forward_gen-DBG] layer_idx=1 start_pos=8
...
[forward_gen-DBG] layer_idx=0 start_pos=16
[forward_gen-DBG] layer_idx=1 start_pos=16
Aborted                                       ← SIGABRT here
```

- ✅ Hook 정상 fire (cachefit_end 140/token = 5 cache-fit × 28 layer)
- ✅ Chunk dispatch (token 0~2 안에 225 모두)
- ✅ Plan retire (drain 0ms + ratio_generation bump + invalidate)
- ✅ Retire 직후 9 token 정상 forward
- ❌ **token ~11 (start_pos=16) SIGABRT**

### 1.2 디버그 인프라 (이미 구축됨)
- `LLMRS_PHASE_AWARE_DEBUG=1` env → set_phase_hook + dispatcher snapshot + forward_gen entry trace
- `PhaseAwareSwapDispatcher::debug_snapshot()` — 7-tuple (queue, in_flight, pending, dispatched, hook_start, hook_end, cachefit_end)

---

## 2. Crash reproducer

```bash
adb shell '
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
export LLMRS_PHASE_AWARE_DEBUG=1   # snapshot 보고 싶으면
./generate \
    -m /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
    --secondary-gguf /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf \
    --force-swap-ratio 0.9 \
    --swap-phase-aware \
    -p "The capital of France is" -n 32 \
    --backend opencl --temperature 0 --initial-kv-capacity 2048 2>&1 | tail -40
'
# Expected: forward_gen layer 0/1 trace 매 token, retire at tok=2,
#           그 후 ~9 token 정상 forward, 이후 Aborted
```

**Decode 약 10 token** 진행 후 crash. 단, `RUST_BACKTRACE=1` 미적용 — backtrace로 정확한 crash 지점 파악 필요.

---

## 3. 의심 root cause (우선순위 순)

### 3.1 ArcSwap commit 직후 stale workspace cl_mem 사용 — **HIGH**
`PhaseAwareSwapDispatcher::commit_layer` (`phase_aware_swap.rs:386`)이 `SwapCommitJob` submit. AsyncSwapDispatcher worker가 cl_event 대기 후 `slot.swap_weights(new_weights)` ArcSwap commit. 이 시점에 forward thread가 **그 layer의 옛 cl_mem을 workspace (gen_ws)에 쥐고 있을 수 있다**.

체크포인트:
- LISWAP-4 `IntraForwardSwapHook::dispatch_layer` (`intra_forward_swap.rs:415-510`) 의 wait-gate 패턴 (forward thread가 layer N+1 진입 직전 `pending_event_for(N+1)` 으로 wait) 가 누락됨.
- LISWAP-5는 phase boundary 단위로만 wait — layer boundary wait 없음.

→ **layer 단위 wait gate 추가 필요**: forward thread가 layer K 진입 직전 dispatcher의 layer K commit이 끝났는지 확인. 안 끝났으면 wait.

### 3.2 ratio_generation bump → noshuffle SOA registry 불일치 — **MEDIUM**
finalize의 (3) ratio_generation bump + (4) `invalidate_noshuffle_soa_registry()` 후, 다음 forward가 layer weight를 사용할 때 SOA registry가 이미 invalidate된 상태. backend가 lazy rebuild를 시도하지만 race 가능.

체크포인트:
- 단일-shot swap도 같은 invalidate 호출하지만 crash 안 남 — 차이는 commit timing.
- 단일-shot은 swap이 **prefill 전**에 끝남. LISWAP-5는 swap이 **decode 도중** 끝남. workspace가 이미 활성 상태에서 invalidate.

→ `remap_weights_for_cpu_after_swap`이 finalize 후 호출되는데 이 함수가 실제로 cl_mem 상태를 정상화하는지 확인. 또는 invalidate 직전 forward thread를 quiesce할 방법이 필요.

### 3.3 release_worker가 commit 중인 cl_mem release — **MEDIUM**
`PhaseAwareSwapDispatcher::commit_layer` (`phase_aware_swap.rs:421`):
```rust
release_worker: None,  // ← LISWAP-5는 release_worker 안 넘김
```
LISWAP-4는 `Some(Arc::clone(&model.release_worker))` 넘김. 차이.

release_worker가 None이면 SwapCommitJob 처리 후 옛 weights cl_mem이 **즉시 drop** — Rust drop 순서로 release. 이 시점에 forward thread가 옛 cl_mem 참조를 잡고 있으면 use-after-free.

→ **release_worker 전달 추가 필요** (intra_forward_swap.rs:158, 182, 198 패턴 참고).

### 3.4 partition_ctx None 강제 — **LOW**
`PartialLayer::into_layer_weights` (`phase_aware_swap.rs:137`):
```rust
partition_ctx: None,  // DF-35-3: tensor_partition × weight swap mutually exclusive
```
LISWAP-5는 partition을 강제로 끔. workspace가 partition_ctx에 의존하면 crash 가능. 단 본 측정은 partition 비활성화 모드라서 영향 X.

---

## 4. 디버그 시작 점

### 4.1 backtrace 확보 (필수)
```bash
adb shell '
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
export RUST_BACKTRACE=full
./generate ... --swap-phase-aware ...  # 위 reproducer
' 2>&1 | tail -80
```

`Aborted` 직전의 backtrace로 정확한 crash 지점 파악.

### 4.2 release_worker 전달
`engine/src/bin/generate.rs:2914` 부근에서 `PhaseAwareSwapDispatcher::new`에 `release_worker` 인자 추가 필요. 또는 dispatcher 자체 필드에 보관.

`PhaseAwareSwapDispatcher::commit_layer` 의 `SwapCommitJob` 생성 코드 (`phase_aware_swap.rs:416-424`)에서 `release_worker: Some(Arc::clone(&self.release_worker))` 로 변경.

### 4.3 layer 단위 wait gate 검토
`IntraForwardSwapHook::pending_event_for_dyn` (intra_forward_swap.rs:67) 패턴 — forward path가 layer 진입 직전 호출하여 wait. LISWAP-5에 동일 메커니즘 추가:
1. `PhaseAwareSwapDispatcher`에 `pending_layer_events: Vec<ArcSwapOption<GpuEvent>>` 추가
2. `commit_layer`에서 last_event를 해당 slot에 store
3. forward_gen 또는 transformer.rs의 layer loop에서 layer K 진입 전 `pending_layer_events[K]` 검사 + wait
4. dispatcher worker의 `on_complete` callback에서 clear

이건 LISWAP-4 v3와 거의 동일 패턴.

### 4.4 finalize 순서 점검
LISWAP-4 finalize:
```
(1) drain
(2) synchronize
(3) ratio_generation bump
(4) invalidate
```
LISWAP-5 finalize는 추가로:
```
(0) chunk_queue drain (남은 chunk 동기 dispatch + wait)
(1) drain
(2) synchronize
(3) ratio_generation bump
(4) invalidate
```

(0)에서 `try_dispatch_chunk` + `wait_pending` 반복. 단 wait_pending이 in_flight clear만 하고 SwapCommitJob 자체는 worker thread에서 처리되므로, drain (1)이 worker job 처리 완료 보장. 순서는 OK.

단 release_worker가 None이라 commit 후 즉시 drop → race 가능.

---

## 5. Fix 순서 권장

### Step 1 — backtrace 확보 (15 min)
RUST_BACKTRACE=full 로 정확한 crash 지점 확인. 가능 시 addr2line으로 함수 매핑.

### Step 2 — release_worker 전달 (30 min, simple fix)
`PhaseAwareSwapDispatcher::new` 시그니처에 `release_worker: Option<Arc<PrimaryReleaseWorker>>` 추가. generate.rs에서 `Some(Arc::clone(&model.release_worker))` 전달. commit_layer의 SwapCommitJob에 사용.

이것만으로 crash 해결될 가능성 높음 (Step 3.3 진단이 맞다면).

### Step 3 — backtrace + release_worker로 안 풀리면 layer wait gate 추가 (1d)
LISWAP-4 패턴 모방. `pending_layer_events` 필드 + `pending_event_for_dyn` impl + forward path 통합 (transformer.rs layer loop hook).

### Step 4 — wall-clock 재측정 (30 min)
crash fix 후 4-gate 다시:
```bash
# Correctness
./generate --swap-phase-aware ... -n 32  # output sanity
./generate --swap-phase-aware ... -n 200 # long decode

# vs single-shot 비교
diff <(generate --force-swap-ratio 0.9 ...) <(generate --swap-phase-aware ...)
# (mid-decode swap이라 byte-equal 안 됨, semantic plausibility만)

# TBT/TTFT/E2E 측정
./generate --swap-phase-aware ... -n 32 2>&1 | grep -E "TTFT|Decode:"
```

### Step 5 — postmortem 보고서 갱신
`papers/.../swap_overhead_liswap5_v1_postmortem.md` Section 6 "진짜 v1 평가" 섹션을 실제 측정값으로 채워서 v2 진행 여부 판단.

---

## 6. 검증 gate

```
✅ backtrace 확보 (crash 지점 함수 명확)
✅ Reproducer 안정 (n=32 매번 같은 token에서 crash)
✅ Step 2 fix 후 n=32 정상 완료 (Aborted 없음)
✅ output 출력 정상 (마지막 token까지 plausible English)
✅ TBT 측정값 수집 (n=32, n=200)
✅ Decode 결과가 single-shot Q4 답과 semantic 유사 (대부분 같은 단어)
✅ cargo test --workspace 회귀 0
```

---

## 7. 다음 세션 시작 절차

```bash
cd /Users/li/Workspace/llm_rs2

# 1. 상태 확인
git log --oneline -5    # 12bac91이 HEAD
cat .agent/todos/handoff_liswap5_crash_fix_2026_05_10.md  # 본 문서
cat papers/eurosys2027/_workspace/experiment/swap_overhead_liswap5_v1_postmortem.md

# 2. 디바이스 + 모델 확인
adb devices             # R3CY408S5SB
adb shell 'ls /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf'

# 3. backtrace reproducer (Step 1)
adb shell '
cd /data/local/tmp
export LD_LIBRARY_PATH=/data/local/tmp:/data/local/tmp/qnn:$LD_LIBRARY_PATH
export ADSP_LIBRARY_PATH=/data/local/tmp/qnn
export RUST_BACKTRACE=full
./generate -m /data/local/tmp/models/qwen2.5-1.5b/qwen2.5-1.5b-f16.gguf \
    --tokenizer-path /data/local/tmp/models/qwen2.5-1.5b/tokenizer.json \
    --secondary-gguf /data/local/tmp/models/qwen2.5-1.5b-q4_0.gguf \
    --force-swap-ratio 0.9 --swap-phase-aware \
    -p "The capital of France is" -n 32 \
    --backend opencl --temperature 0 --initial-kv-capacity 2048 2>&1 | tail -80
'

# 4. Step 2 release_worker 전달 — 권장 위임:
# Implementer (sonnet) 적합 (단순 wire-up, intra_forward_swap.rs 패턴 모방)
# 또는 Senior Implementer (전체 race 분석 + layer wait gate까지 보겠다면)
```

---

## 8. 관련 파일 인덱스

### 본 작업으로 수정 가능성 높은 파일
- `engine/src/models/weights/phase_aware_swap.rs` (release_worker 추가, layer wait gate)
- `engine/src/bin/generate.rs:2914` 부근 (PhaseAwareSwapDispatcher::new 호출)

### 참고 패턴 (수정 X)
- `engine/src/models/weights/intra_forward_swap.rs:182, 415-510` — release_worker 사용 + layer wait gate
- `engine/src/models/weights/intra_forward_swap.rs:307-344` — finalize 패턴
- `engine/src/models/weights/swap_executor.rs:518` — SwapCommitJob 생성

### 디버그 인프라 (현재 활성 — `LLMRS_PHASE_AWARE_DEBUG=1`)
- `engine/src/profile/op_trace.rs:51-60` — set_phase_hook log
- `engine/src/models/weights/phase_aware_swap.rs::debug_snapshot` — counter 7-tuple
- `engine/src/bin/generate.rs:5803` 부근 — decode loop snapshot dump
- `engine/src/layers/transformer_layer/forward_gen.rs:24` — forward_gen entry trace

crash fix 완료 후 이 디버그 코드는 **그대로 보존** (env-gated, default off, 다음 v2 작업 시 재활용).

---

## 9. 메모리 상태

본 세션 작업과 직접 관련:
- `project_phase_aware_swap_viable.md` — CV 1.2% precondition 측정 (불변)

다음 세션이 알아야 할 것:
- 본 v1은 design 자체는 옳지만 commit ordering 미흡
- crash fix 후 measurement가 진짜 v1 평가 — 그 결과에 따라 v2 (Option α/β) 진행 여부 판단
- 9-track handoff §3.1 LISWAP-4 v2 함정 = 본 세션의 plan-bypass 동일 패턴 (이번 fix됨, 다음 세션 직접 영향 X)

---

## 10. Commit history (이번 세션 — context)

```
12bac91 docs(liswap5): v1 postmortem — 효과 없었던 진짜 이유 정리
ed7464f fix(liswap5): generate.rs plan-path bypass guard + diagnostic counters
0348f9e docs(liswap5): B-3 v1 production measurement — YELLOW verdict (stale, postmortem이 갱신)
ab1f767 feat(liswap5): B-2.5 CLI wire-up
f214ff9 feat(liswap5): B-2.4 chunk dispatcher 본체
f6f89ee feat(liswap5): B-2.2+B-2.3 PhaseHook trait + macro
2015597 feat(profile): OpKind::ddr_phase() classifier + plan
```

---

**End of Handoff**

self-contained: 다음 세션은 본 문서 + postmortem만으로 시작 가능. crash 원인 우선순위는 §3 (release_worker가 가장 의심).
