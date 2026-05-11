# Swap mode 채택 기준 — 메모리 spike 회피 (Hard Constraint, 2026-05-11)

## Ground Rule (사용자 명시 요구)

**메모리 spike 발생 = production 채택 불가**, wall latency 와 무관.

- swap mode / KV cache management / weight loading 등 **메모리 사용 패턴 결정 시 spike 회피가 latency 보다 우선**
- bulk burst (N layer 동시 enqueue / alloc / commit, release_worker queue backlog 등) = spike → 채택 불가
- steady-state **1 layer extra (~114 MB on Qwen2.5-1.5B)** 까지만 허용
- spike 발생 mode 의 측정값은 paper baseline / 비교용으로만, production 권장 절대 금지

## 측정 환경

- KV 1.8k, Qwen2.5-1.5B-F16 primary + Q4_0 secondary
- Galaxy S25, qnn_oppkg backend, LISWAP-6 alias
- n=30 decode, 3 run median

## Layer 크기 reference (Qwen2.5-1.5B)

| | F16 | Q4_0 |
|---|---:|---:|
| 1 layer (q/k/v/o + gate/up/down + 2 norm) | **~89 MB** | **~25 MB** |
| 25 layer (전체 swap target) | ~2.2 GB | ~625 MB |

`extra peak (1 layer in-flight)` = old F16 (release queue) + new Q4 (committed) = **~114 MB**

## 채택 결정 (2026-05-11 backpressure 제거 + queue depth 직접 측정 후 정정)

### 잘못된 진단 정정

이전 진단 (작업 B 재적용 시 sync25 winner) 은 **두 가지 오류** 있었음:
1. **backpressure 가 wall cost 만들어내는 메커니즘**: `release_worker.drain` 호출 자체가 polling/IPC overhead ~50 ms/layer 를 만들어냄. release worker 의 cl_mem drop 자체는 sub-ms (alias 환경). senior-implementer 의 "Adreno GPU flush 30~50ms" 가설은 잘못된 인과관계.
2. **Correctness 보고 오류**: prompt continuation prefill 출력 부분만 보고 "8 mode 모두 동일" 결론. 실제 generated tokens 는 mode 마다 다름 (mixed weight progression 분기).

### ✅ Production 채택 (1-layer peak 자연 보장, 사용자 정책: async 기본)

| 순위 | Mode | F+S (ms) | rel_max | disp_max | 1-peak? | decode |
|---|---|---:|---:|---:|:---:|---|
| 🥇 | **async2 (--swap-incremental-per-tick 2 --swap-async-dispatch)** ★ | **316.7** | **0** | **1** | ✓ | 정상 |
| 🥈 | sync1 (명시적 요청 시에만) | 320.1 | **0** | **0** | ✓ | 정상 |
| 🥉 | async1 | 322.6 | **0** | **0** | ✓ | 정상 |

**정책**: weight swap 은 async path 가 default (`--swap-async-dispatch`). sync 는 명시적 요청 시에만 사용 (`feedback_swap_async_default.md`).

**Safe K 상한 = 2** (Adreno Qwen2.5-1.5B + LISWAP-6 alias 실측). K=3 부터 rel_max=1 + decode garbage (mixed-weight race) 발생.

### ❌ Production 채택 불가 (release queue backlog spike 또는 decode 깨짐)

| Mode | F+S (ms) | rel_max | host RAM extra | 폐기 사유 |
|---|---:|---:|---:|---|
| **async3** | 308.2 | **1** | ~90 MB | **decode garbage** (`.jpg` etc., mixed-weight race) + 1-peak violation |
| sync25 | 313.2 | **23** | ~2 GB | 사용자 우려 spike 실측 확정 |
| async25 | 320.6 | 22 | ~2 GB | 동일 |
| sync8 | 312.0 | 7 | ~630 MB | batch 안 sub-ms 간격으로 release worker 못 따라잡음 |
| async8 | 327.4 | 6 | ~540 MB | 동일 |
| sync4 | 316.0 | 3 | ~270 MB | 동일 |
| async4 | 317.1 | 2 | ~180 MB | 동일 |

### 핵심 통찰

- **pertick=1 만 자연 1-layer peak 보장** — token 간격 (~10ms forward) > release worker drop 시간 (sub-ms) → 다음 layer enqueue 시점 queue empty
- **pertick≥2 batch loop 은 sub-ms 간격으로 N layer enqueue** → release worker 가 못 따라잡음 → queue 가 N-1 까지 backlog
- **drain backpressure 추가는 잘못된 fix** — wall cost 만 만들고 1-peak 보장은 자연 흡수에 맡기는 게 정답
- 같은 pertick 의 sync/async 는 byte-equal token 시퀀스 → race 없음 입증

## 구현 상태

### 적용된 fix (swap_executor.rs, 최종)

1. ~~**작업 A**~~ — async path drain backpressure: **제거** (잘못된 fix, wall cost 만 만들어냄)
2. ~~**작업 B**~~ — sync path release backpressure: **제거** (동일)
3. **작업 C** (line 700-712) — sync path: `SecondaryMmap::Rpcmem` match 시 batch-end `backend.synchronize()` skip
   - alias 환경에서 H2D drain 대상 없어 무용
   - sync1: 1714 → 320 ms (-81%) 효과 (작업 C 단독으로 충분)
4. **작업 2 (모니터링)** (line 453-484, 829-839) — env-gated `LLMRS_SWAP_DRAIN_DIAG=1` 시 release_worker pending + dispatcher pending max 측정 → batch 끝 `[SwapPeak]` 출력
   - production runs 영향 없음 (default off)

## 회귀 가드

- correctness: 4 mode 모두 baseline 동일 출력 (일관된 영문)
- weights 모듈 unit test 64/64 PASS
- cargo build --release --target aarch64-linux-android PASS
- mid-decode swap (delay > 0) 미측정 — 후속 검증 필요

## 후속 작업 (Priority)

| P | 작업 | 설명 |
|---|---|---|
| P1 | **`--swap-incremental-per-tick 2 --swap-async-dispatch` 를 CLI default 로** | 진짜 production winner (async 기본 + safe K 상한) |
| P2 | **`per_tick > 2` warning 또는 hard cap** | K=3+ 부터 spike + decode garbage 확정. CLI 검증에서 K>2 면 warning, K_max=2 강제 옵션 |
| P2.5 | adaptive K (probe-based) | 검토 완료, 권장 안함 — non-deterministic output + 이득 미미 (~1-2ms). 정적 K=2 가 더 단순 |
| P3 | non-alias 환경 측정 | host OpenCL backend / 다른 디바이스에서 spike 회피 검증 (token 간격 vs release time 비율 다를 수 있음) |
| P4 | 다른 모델 (Qwen 7B, Gemma 2B) 검증 | layer 크기 다른 환경에서 token 간격 흡수 가능한지 |
| P5 | mid-decode swap (--swap-delay-tokens N) 재측정 | production scenario 검증 |
| P6 | LISWAP-6 cleanup segfault 수정 (task #33) | Done. 후 segfault, 별도 디버깅 |

## 변경 파일 (커밋 후보)

- `engine/src/models/weights/swap_executor.rs` — 작업 A + C
- (기존 미커밋: Phase 5b 의 `build_tensor_from_mmap_async_for_hook` alias skip)

## 참고

- 진단 보고서: 본 세션 transcript (Adreno release_worker GPU flush 발견)
- 이전 작업: `handoff_phase_aware_win_attempt_followup_2026_05_11.md` (Phase 5b)
- 측정 raw: `/tmp/swap_bench_out/`, `/tmp/swap_pertick_sweep_20260511_104011/`
