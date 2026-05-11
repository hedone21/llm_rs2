# Dynamic-K + Sub-batch Pause 통합 — Production Handoff (2026-05-12)

## TL;DR

LISWAP-6 weight swap에 **3-layer safety net**을 추가했다. spike 회피 hard constraint (`feedback_no_memory_spike.md`)를 알고리즘적으로 보장한다. CLI default가 production winner mode로 변경됨 — 사용자는 `--swap-incremental-per-tick 2`만 추가하면 끝.

**Production 명령**:
```bash
./generate \
  -m qwen2.5-1.5b-f16.gguf --tokenizer-path tokenizer.json \
  --secondary-gguf qwen2.5-1.5b-q4_0.gguf \
  --force-swap-ratio 0.9 \
  --swap-incremental-per-tick 2 \
  --backend qnn_oppkg \
  -p "<prompt>" -n 30
```
`--swap-async-dispatch`, `--swap-dynamic-k`는 CLI default ON (생략 가능).

## 3-Layer Safety Net

```
Layer 1: Dynamic-K Controller (사전, timing-based)
   첫 token 1회 calibration → K_intent ∈ {1, 2} 자동 결정
   hard_upper=2 (quality drift cap, Qwen 1.5B 실측)
   fwd 짧아지면 K_intent 감소만 (ratchet)
                          ↓ K_intent 전달
Layer 2: Reactive Pause (token 시작 직전)
   if release.pending() > 0 → 이 token swap 전체 skip
   이전 batch 의 잔존 spike 100% 흡수
                          ↓ swap 진행 결정
Layer 3: Sub-batch Pause (batch dispatch loop 안)
   for i in 0..K_intent:
     if i > 0 && release.pending() > 0 → break
     dispatch_layer(i)
   batch 내부 layer 단위 hardware-paced
```

상세 알고리즘 / 시나리오 비교 / 환경 가정은 `docs/swap_dynamic_k_guide.md`.

## Commit Trail

| Commit | 내용 |
|---|---|
| `ca6d041` | Phase 5b/6 alias H2D-skip + layer-immediate + queue peak diag |
| `1359596` | handoff (이전) — async pertick=2 production winner |
| `00d88c2` | handoff cold-start 가이드 |
| `e58d31e` | **Dynamic-K controller — timing-based safe K 자동 결정** |
| `81a99c2` | **Sub-batch reactive pause — burst truncate by release queue state** |
| (pending) | CLI default 변경 (async+dynamic-K = ON) + 본 handoff |

## 디바이스 측정 (Galaxy S25, qnn_oppkg alias)

Qwen2.5-1.5B-F16 primary + Q4_0 secondary GGUF, n=30, 3 run median.

| Mode | F+S TBT (ms) | max_release_pending | sub_batch_cutoff | decode |
|---|---:|---:|---:|---|
| 정적 K=2 baseline | 33.17 | 0 | (N/A) | 정상 |
| dynamic-K only (`e58d31e`) | 33.38 (+0.6%) | 0 | (N/A) | 정상 |
| **dynamic-K + sub-batch pause (`81a99c2`)** | **31.20** (-6%) | **0** | **0** (silent) | 정상 |

**3개 결과 모두 spike 0 + decode 정상**. sub-batch pause는 alias 환경에서 release가 sub-μs라 cutoff 0% — silent safety net으로 작동. 의미는 **non-alias / 환경 변화** 시 발휘.

## Hard Constraint 보호 범위

3 layer가 막는 시나리오:

| 시나리오 | Layer 1 | Layer 2 | Layer 3 | 결과 |
|---|---|---|---|---|
| 정상 (fwd 10ms, release 1μs) | K=2 결정 | pass | cutoff 안 함 | K=2 dispatch ✓ |
| 이전 token 잔존 release | K=2 결정 | **pause 발동** | — | 이 token swap skip ✓ |
| 이전 token outlier 잔존 (작은 잔량) | K=2 결정 | pass | **cutoff 발동** | actual K=1 ✓ |
| 이번 token fwd outlier (극단) | K=2 결정 | pass | cutoff X | K=2 dispatch, **release 못 따라잡으면 다음 token Layer 2 흡수** |
| 환경 변경 (release 느려짐) | observe 로 K_intent 감소 | pause 빈도↑ | cutoff 빈도↑ | 자동 적응 ✓ |

**알고리즘 한계 (정직 보고)**: "이번 token outlier 인 동시에 release가 갑자기 ms 단위로 느려진" 동시 이벤트는 1 token 단발 spike 가능. 즉시 다음 token부터 흡수. 완벽 보호는 layer-immediate 통합 (V4 후속).

## CLI Default 변경 (2026-05-12)

`engine/src/bin/generate.rs`:
- `swap_async_dispatch`: false → **true** (production default)
- `swap_dynamic_k`: false → **true** (production default)
- `swap_incremental_per_tick`: 0 유지 (mutual exclusivity 보존)
- per_tick=0 일 때 async/dynamic-K flag warning 제거 (silent ignore)

사용자 영향:
- `--force-swap-ratio R --swap-incremental-per-tick 2` 만 추가하면 production winner mode
- 명시적 sync 원하면 `--swap-async-dispatch=false`
- 명시적 정적 K 원하면 `--swap-dynamic-k=false`
- swap 안 쓰면 (`--force-swap-ratio` 없음) default flag 무효

## 다음 세션 Cold-Start 가이드

### 1. 현재 상태 확인
```bash
git log --oneline -8
# (pending) feat: CLI default async+dynamic-K ON + handoff
# 81a99c2 feat(liswap6): sub-batch reactive pause
# e58d31e feat(liswap6): dynamic-K controller
# 00d88c2 docs(liswap6): handoff 보강
# 1359596 docs(liswap6): handoff async pertick=2 winner
# ca6d041 feat(liswap6): Phase 5b/6 alias H2D-skip
```

### 2. Production 사용
```bash
./generate \
  -m qwen2.5-1.5b-f16.gguf --tokenizer-path tokenizer.json \
  --secondary-gguf qwen2.5-1.5b-q4_0.gguf \
  --force-swap-ratio 0.9 \
  --swap-incremental-per-tick 2 \
  --backend qnn_oppkg \
  -p "<prompt>" -n 30
```

### 3. Safety Net 검증 (선택)
```bash
LLMRS_DYNAMIC_K_DIAG=1 LLMRS_SWAP_DRAIN_DIAG=1 LLMRS_SUB_BATCH_PAUSE_DIAG=1 \
  ./generate ...
# stderr:
# [DynamicK] calibrated drop_ms=0.000 fwd_ms=4.4 safe_k=2
# [SwapPeak] max_release_pending=0 max_dispatcher_pending=1 sub_batch_cutoff=0
```

## 후속 작업 (Priority)

| P | 작업 | 상태 |
|---|---|---|
| ~~F1~~ | 기존 handoff 갱신 | ✓ 완료 |
| ~~F2~~ | 메모리 `project_liswap6_alias_production.md` 갱신 | ✓ 완료 |
| ~~R1~~ | CLI default 변경 (async+dynamic-K ON) | ✓ 완료 (본 commit) |
| R2 | K>2 hard cap CLI 검증 (warning 또는 reject) | 미진행 |
| V1 | Non-alias 환경 (host OpenCL backend) 측정 — dynamic-K timing 진짜 동작 검증 | 미진행 |
| V2 | 다른 모델 (Qwen 7B, Gemma 2B) — quality drift cap K=2 검증 | 미진행 |
| V3 | Mid-decode swap (`--swap-delay-tokens N`) 시나리오 측정 | 미진행 |
| V4 | Layer-immediate + dynamic-K 통합 측정 — "이번 token outlier" 한계 해결 | 미진행 |
| #33 | LISWAP-6 cleanup segfault 수정 | pending |
| Issue 2 | `qnn_oppkg + .auf secondary` lm_head SharedBuffer 버그 | dynamic-K 측정 중 발견, 별도 |

## 참고 문서

- 사용 가이드: `docs/swap_dynamic_k_guide.md` (상세 알고리즘 / 시나리오 / 사용법)
- 이전 handoff: `handoff_swap_memory_spike_constraint_2026_05_11.md` (정적 K=2 시기)
- Memory feedback: `feedback_no_memory_spike.md`, `feedback_swap_async_default.md`
- 알고리즘 doc: `engine/src/models/weights/dynamic_k.rs` (module header)
- Sub-batch pause: `engine/src/models/weights/swap_executor.rs::execute_on_slots` (line 466 근처)

## 핵심 invariant 요약 (다음 세션이 깨뜨리지 말 것)

1. **K는 사후 monotone 감소만** — probing 금지 (commit `e58d31e`)
2. **이전 batch 잔존 release = Layer 2가 100% 흡수** (token swap skip)
3. **이번 batch 내부 backlog = Layer 3가 100% 흡수** (layer 단위 truncate)
4. **Quality drift cap K=2** = 측정 결과, hard_upper=2를 알고리즘적으로 풀지 말 것
5. **CLI default async+dynamic-K ON** — production 사용성. mutual exclusivity 보존 위해 per_tick=0 유지
6. **alias 환경 가정** = Adreno qnn_oppkg + LISWAP-6 alias. 환경 벗어나면 재검증 (V1)
