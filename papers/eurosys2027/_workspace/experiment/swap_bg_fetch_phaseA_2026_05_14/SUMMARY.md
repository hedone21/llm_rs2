# LISWAP-8 Phase A: Background Fetch (2026-05-14)

## 목적
mmap_permute (1577 ms @ K=32) + cuMemAlloc + cuMemcpyAsync를 main thread에서 dispatcher worker thread로 완전히 옮겨 forward와 hide.

## 구현 요지
- 신규 env gate `LLMRS_SWAP_BG_FETCH=1` (`swap_executor.rs:557`)
- 신규 dispatch path (`swap_executor.rs:684`) — `use_async && bg_fetch && !par_build`
- Standalone fn 3종 (`swap_executor.rs:2212+`): SwapExecutor `&'a` borrow 우회
- Worker closure: build → wait_event → arc_swap → release-chain (전부 worker thread `llmrs-async-swap`)

## 측정 결과 (Jetson Llama 3.1 8B F16+Q4_0, K=32 async, 50 tokens)

| run | TTFT | forward (excl tok[0]) | Avg TBT | mmap_permute (**main**) | dispatcher_pending | release_pending |
|---|---|---|---|---|---|---|
| k32_baseline | 1724.59 | 66.99 ms | 110.40 ms | 1801.7 ms | 1 | 0 |
| k32_par_seq | 1239.16 | 68.22 ms | 130.09 ms | 1570.7 ms | 31 | 0 |
| **k32_bg_fetch** | 1605.85 | **67.82 ms** ✓ | **112.03 ms** | **0.0 ms** ✓ | 31 | 0 |

## 합격 기준 검증

| 기준 | 목표 | 측정값 | 결과 |
|---|---|---|---|
| forward ≤ 70 ms | par_seq 68 ms baseline 유지 | 67.82 ms | ✅ |
| mmap_permute main = 0 | 모든 비용 worker로 이동 | 0.0 ms | ✅ |
| Avg TBT ≤ par_seq | mmap_permute hide | 112 ms (par_seq 130 vs baseline 110) | ✅ |

## 핵심 발견

1. **mmap_permute 완전 hide 성공** — main thread stage timing 0 ms. 1577 ms 전체가 dispatcher worker 안에서 forward와 overlap.
2. **dispatcher_pending=31에서도 forward 67.82 ms 유지** — par_seq 데이터 (forward 68 ms @ pending=31) 그대로 재현. Single-thread background work는 forward에 영향 없음 확정.
3. **release_pending=0 유지** — 메모리 spike 없음. Worker 안에서 release_worker chain까지 동기적으로 처리.
4. **baseline(110) vs bg_fetch(112) 거의 동등** — sub-batch wait 활성 baseline은 release wall이 자연 hide되는 형태로 mmap_permute가 chunks로 forward 사이 분산. bg_fetch는 명시적 worker dispatch로 같은 효과를 더 deterministic하게.

## baseline vs bg_fetch 메커니즘 차이

- **baseline** (LLMRS_SWAP_BG_FETCH 없음): main에서 build → submit_commit. mmap_permute 1801 ms가 main thread에 chunks로 분산 (49 tokens × 36.8 ms/token forward 사이에 끼임). dispatcher_pending=1 (sub-batch wait가 release queue depth=1 강제).
- **bg_fetch**: main thread는 submit_dispatch_chunk만 (μs). mmap_permute 1577 ms가 worker thread 안에서 forward와 명시적 overlap. dispatcher_pending=31에서 폭주 시작.

paper 관점: bg_fetch는 main thread "기여 0%"를 deterministic하게 보장. K가 작아질 때 (K=4/K=8) 또는 secondary가 크면 (Llama 70B) 차이가 벌어질 가능성.

## 후속 측정 후보

1. K=4 / K=8 / K=16 sweep on bg_fetch (작은 K에서 hide 비율 변화)
2. ×3 반복 (noise 평가)
3. Llama 8B Q4→F16 (역방향)
4. dynamic-K + bg_fetch 결합

## 다음 트랙 후보

- **Phase B (Layer Object Pool)**: cuMemAlloc/cuMemFree 호출 제거 → driver lock 경쟁 0. Pool=2 rotating buffer. ~300 LOC + 신규 `Backend::enqueue_write_into_async`. Phase A 결과로 안전성 검증됐으니 진입 결정 사용자 권한.

## 데이터 파일
- `k32_baseline.{stdout,stderr,tbt.jsonl}` — 기본 path
- `k32_par_seq.{stdout,stderr,tbt.jsonl}` — single-thread prebuild
- `k32_bg_fetch.{stdout,stderr,tbt.jsonl}` — Phase A 신규
