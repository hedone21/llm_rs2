# Phase-aware win 시도 — 4-Phase 실험 보고서 (2026-05-11)

**목표**: phase-aware swap이 single-shot (per-tick=25) wall time을 이길 수 있는지 검증.

**Metric**: A (wall time) 우선. KV 1.8k Qwen2.5-1.5B, qnn_oppkg + LISWAP-6 alias, n=30/60.

---

## 결과 요약 (wall ms, median 3 run)

| Phase | 변경 | per-tick=25 | per-tick=1 | phase-aware | phase-aware vs single-shot |
|---|---|---:|---:|---:|---|
| **0** | LISWAP-6 alias only | 735 | 1525 | 1411 | +92% (loss) |
| **1** | + cached cl_mem alias | **405** ★ | 1437 | 1029 | +154% (worse) |
| **2** | + throttle (K=4) | 405 | 1437 | **1388** (K=4) | 무효 (wall ↑) |
| **3** | + mid-decode (delay=30) | 402 | 1793 | 1229 | +205% (loss in production) |
| **4** | + async worker thread | TBD | TBD | TBD | TBD |

(swap-to-decode-end wall, n=30 except Phase 3/4 = n=60)

---

## Phase 별 상세 결과

### Phase 1 — Cached cl_mem alias (가장 큰 single 효과)

**가설**: alias dispatch 4.5 ms × 252 chunks = 1146 ms inflation 의 root cause = `clCreateBuffer(USE_HOST_PTR)` per chunk.

**구현**: `RpcmemSecondaryStore::ensure_layer_loaded()` 안에서 모든 (layer, tensor) cl_mem alias를 eager alloc + cache (336 entries).
- Cycle 차단: `RpcmemAliasBuffer::secondary_arc` 를 `Weak<SecondaryMmap>` 로 demote
- Lifetime: cache가 strong Arc 보유, model drop 시 cascade

**결과**:
- per-tick=25: 735 → **405 ms (-45%)** ★
- per-tick=1: 1525 → 1437 (-6%)
- phase-aware: 1411 → 1029 (-27%)
- swap blocking (per-tick=25): 499 → 92 ms (5.4×)

**검증**: dispatch 4.5 → ~0.5 ms 단축 확정. cl_mem 생성이 bottleneck 이었음.

**Verdict**: ✅ **A win 확보**, single-shot 405 ms 가 새 baseline.

---

### Phase 2 — Throttle (chunks-per-token 제한)

**가설**: phase-aware의 chunk dispatch 누적이 forward inflation 원인. token당 K chunks로 제한하면 분산 효과로 max stall 단축.

**구현**: `phase_aware_swap.rs::on_op_end` 에 max_chunks_per_token guard. CLI `--swap-phase-aware-max-chunks-per-token N`.

**결과** (K sweep, phase-aware mode):

| K | wall (ms) | max_stall (ms) | swap 완료? |
|---|---:|---:|---|
| 1 | 558 | 66 | ❌ swap incomplete |
| 2 | 858 | 70 | ❌ |
| 4 | 1388 | 98 | ❌ |
| 8 | 2715 | 127 | ❌ |
| 16 | 2118 | 153 | ⚠ 부분 (2/3 run) |
| **84 (default)** | **1029** | 410 | ✅ |

**해석**: K↓ 시 swap이 n=30 안에 완료되지 않음 (252 chunks 소진 불가). 완료 보장하는 K=16/84 중 K=84 가 wall 최소.

**Verdict**: ❌ **Throttle 무효** (metric A 기준). K=84 = no-throttle = 최선이지만 single-shot 못 이김.

---

### Phase 3 — Mid-decode swap (production scenario)

**가설**: 현재 측정은 swap이 token 0 강제 → TTFT inflation. 실제 production 은 mid-decode signal → 다른 trade-off 가능.

**구현**: `--swap-delay-tokens N` CLI flag. dispatch_force_swap! macro로 trigger 코드 통합. delay > 0 이면 prefill 후 deferred, decode loop 의 token N 시점 trigger.

**결과** (delay=30, n=60):

| Mode | swap-to-end wall | max_stall (post-trigger) | vs Phase 1 (delay=0) |
|---|---:|---:|---:|
| **per-tick=25** | **402 ms** | **12.7 ms** ★ | ~동일 (405) |
| per-tick=1 | 1793 ms | 12.5 ms | +25% (1437) ⬆ worse |
| phase-aware | 1229 ms | 342 ms | +19% (1029) ⬆ worse |

**해석**:
- Single-shot은 mid-decode에서도 동일한 wall — Phase 1 cached alias가 swap blocking 자체를 단축했기 때문 (12 ms max stall, user-imperceptible).
- per-tick=1은 mid-decode에서 +25% 악화: decoding 끝나가는데 swap dispatch 28 token 동안 누적.
- phase-aware도 +19% 악화: chunk dispatch가 forward에 분산되는 효과는 그대로.

**Verdict**: ❌ **모든 시나리오에서 single-shot 압승** 확인. Phase 1 cached alias 가 single-shot의 user-stall도 ~12 ms로 단축해 UX 메트릭에서도 win.

---

### Phase 4 — Async worker thread (Phase-aware 마지막 희망)

**가설**: phase-aware의 forward inflation 원인은 `try_dispatch_chunk`가 forward thread에서 sync 실행. worker thread로 옮기면 forward는 channel push (~10 us)만 → cache-fit window 점유 안 함.

**구현**:
- `AsyncSwapDispatcher::SwapJob::DispatchChunk(ChunkDispatchJob)` 추가 (closure-based, dependency cycle 회피)
- `PhaseAwareSwapDispatcher::on_op_end` → `submit_dispatch_chunk(weak_self.clone())` (forward 즉시 return)
- Worker thread가 `try_dispatch_chunk_worker` 호출
- `OnceLock<Weak<Self>>` + `install_self_weak()` post-construction 패턴
- `is_complete()` race guard, `finalize()` pre-drain

**잔여 이슈**: worker cadence (1 chunk in-flight 직렬)가 cache-fit hook fire (~140/token) 못 따라감 → 252 chunks 처리에 ~50 token 필요.

**결과** (n=60):

| Mode | wall | swap retired | 비고 |
|---|---:|:---:|---|
| per-tick=25 | 795 ms | 2 tick | n=30 normalize 397 — Phase 1과 동일 (worker thread는 single-shot 무관) |
| per-tick=1 | 2220 ms | 28 tick | swap blocking 변함 없음 (LISWAP-1 path) |
| **phase-aware** | **3807 ms** | **❌ NOT retired** | 252 chunks가 60 token 안 끝나지 못함, forward inflation 100 ms |

**해석**:
1. Worker thread cadence (1 chunk in-flight 직렬) 가 cache-fit hook fire (~140/token) 못 따라감 → swap 50+ token 필요
2. Worker GPU op (`enqueue_write_async`) 가 forward GPU op과 같은 OpenCL queue 사용 → driver FIFO 직렬화 → forward thread `wait_pending` 더 길어짐
3. **결국 forward inflation 더 커짐** (Phase 1 phase-aware 1029 → Phase 4 ~1900 ms, +85%)

**Verdict**: ❌ **Phase-aware win 시도 완전 실패**. Architectural ceiling 확인.

---

## 종합 결론

### 4-Phase 누적 결과 (KV 1.8k Qwen, qnn_oppkg + LISWAP-6 alias, wall median)

| Phase | per-tick=25 (single-shot) | phase-aware | 차이 |
|---|---:|---:|---:|
| 0 — alias only | 735 | 1411 | +92% |
| 1 — + cached cl_mem | **405** ★ | 1029 | +154% |
| 2 — + throttle | 405 | 1029~1388 | 무효 |
| 3 — + mid-decode (delay=30) | 402 | 1229 | +205% |
| 4 — + worker thread | 405 | **1903** | +369% (worse) |

### Phase별 winner

- **Phase 1 cached cl_mem alias**: ★ **유일하게 의미있는 win** — single-shot wall 735 → 405 ms (-45%). phase-aware 도 같이 1411 → 1029 (-27%) 개선되지만 single-shot 못 이김.
- **Phase 2/3/4**: phase-aware win 시도, 모두 실패.

### Phase-aware win 불가능의 본질적 이유

**Adreno OpenCL alias 환경의 architectural ceiling**:

1. **alias = transfer 0**: phase-aware의 "transfer hide" 가설 자체가 무효 (hide할 게 없음)
2. **Single OpenCL queue → driver FIFO 직렬화**: worker thread로 옮겨도 chunk dispatch가 forward GPU kernel과 같은 queue에서 reorder 불가
3. **Cache-fit window (984us) < dispatch overhead (0.5 ms)** even after Phase 1 cached alias: 1 chunk dispatch조차 cache-fit window 점유
4. **252 chunks 처리는 inherently single-shot의 multiple → 분산이 항상 손해**

### 진짜 win 한 단일 조치 = Phase 1 (cached cl_mem alias)

이건 phase-aware 와 무관하게 **모든 swap mode 가속**:
- single-shot per-tick=25: 735 → 405 ms
- per-tick=1: 1525 → 1437 ms (작은 효과)
- phase-aware: 1411 → 1029 ms

→ **Phase 1 만 production 적용**. Phase 2/3/4 는 negative result로 paper 에 기록.

### 후속 조사 후보 (이번 작업 outside scope)

- **Multi-queue (가능성 5)**: Adreno OpenCL의 sub-device / multiple cl_command_queue 지원 여부 — 진짜 GPU-level parallel 가능 시 phase-aware 재평가
- **Hide ratio 측정 인프라 (가능성 12)**: per-chunk timestamp + forward kernel timestamp diff 으로 진짜 overlap 여부 정량화. 현재는 inferred only
- **CPU-side staging buffer**: alias가 아닌 CPU memcpy 환경에서 phase-aware (LISWAP-5 v2 sub-tensor + background pre-staging) 가 microbench Scenario B를 production에서 검증할 마지막 시도 (postmortem §8 P2+P3, 미측정)

### 권장 paper claim

> "DMA-BUF alias swap (LISWAP-6) reduces per-chunk dispatch from 4.5 ms to 0.5 ms when cl_mem aliases are eagerly cached at model load (Phase 1 +45% wall reduction). Phase-aware async swap, despite its theoretical 1.04× of max overlap potential (Phase R Scenario B microbench), does not yield wall improvement on Adreno OpenCL because (1) the alias path eliminates the transfer cost that phase-aware was designed to hide, and (2) chunk dispatch via a single OpenCL queue is serialized by driver FIFO, blocking the same forward kernels it should overlap with. Single-shot in-decode swap remains the production winner across all measurement scenarios (decode-start, mid-decode, n=30/60)."

---

## 변경 코드 (commit 대상)

- `engine/src/buffer/rpcmem_alias_buffer.rs` — Weak<SecondaryMmap> 변환
- `engine/src/models/weights/rpcmem_secondary.rs` — alias_cache + backend Arc + install_self_arc
- `engine/src/models/weights/secondary_mmap.rs` — backend 전달
- `engine/src/models/transformer.rs` — install_self_arc 호출
- `engine/src/models/weights/swap_executor.rs` — cached_alias try-first
- `engine/src/models/weights/phase_aware_swap.rs` — throttle (Phase 2) + worker submit (Phase 4) + install_self_weak
- `engine/src/models/weights/async_swap.rs` — SwapJob::DispatchChunk variant
- `engine/src/bin/generate.rs` — 3 CLI flags + dispatch_force_swap! macro + decode loop hooks + mid-decode trigger
