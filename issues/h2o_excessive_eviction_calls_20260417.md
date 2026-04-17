# H2O eviction: 비정상적 호출 빈도로 compaction 오버헤드가 attention 절감 효과를 상쇄

**Date filed**: 2026-04-17
**Filed by**: PACT 2026 action profile v2 분석 (papers repo, action_profile_v2 batch)
**Scope**: `engine/src/core/eviction/h2o.rs` + `evict_with_scores` 호출 측 (resilience executor 또는 generate loop)
**Severity**: high — H2O 액션의 paper-level 가치 (KV 메모리 절감으로 latency/thermal 완화)가 측정에서 보이지 않음. Fig 2 (cross-domain effects) 결과의 정확성에 영향.

## Summary

S25 / Qwen 2.5 1.5B / decode 5000 tokens (target TBT 150 ms) 환경에서 H2O eviction이 600 s 측정 중 92~112회 트리거된다. Eviction 자체는 코드 (`compact_keep_positions`)대로 정확히 동작하지만 호출 빈도가 너무 높아 매 호출의 K/V 텐서 compaction (memcpy) 비용이 attention 비용 절감을 상쇄한다. 결과적으로 **3가지 keep-ratio 어디서도 baseline-corrected fwd 시간이 줄지 않고**, 오히려 keep-75 % (가장 가벼운 정책)가 keep-25 %보다 더 느리다.

## Reproduction

```bash
# papers repo의 action_profile_v2 batch (이미 실행됨)
bash pact2026/experiments/scripts/action_profile_v2_run.sh evict_h2o_25 1
bash pact2026/experiments/scripts/action_profile_v2_run.sh evict_h2o_50 1
bash pact2026/experiments/scripts/action_profile_v2_run.sh evict_h2o_75 1

# 결과 위치
pact2026/experiments/results/action_profile_v2/evict_h2o_{25,50,75}_tbt150_run{1,2,3}_*
```

## Evidence

### (1) Eviction이 매우 자주 트리거됨 (run1, 600 s 측정)

| 변형 | eviction 호출 수 | 첫 eviction (removed → new_pos) |
|---|---|---|
| evict_h2o_25 | **112회** | 1398 → 466 |
| evict_h2o_50 | **101회** | 937 → 937 |
| evict_h2o_75 | **92회** | 468 → 1401 |

각 호출은 `cache.compact_keep_positions(...)` 를 트리거하므로 [H2 (#L132–L173)](engine/src/core/eviction/h2o.rs#L132)에서 정의된 sort + memcpy 비용을 매번 지불한다. decode 토큰이 ~2000개라고 가정하면 평균 18~22 토큰마다 1회 — 한 번의 compaction이 수 ms 단위로 잡혀도 누적이 무시할 수 없다.

### (2) keep-ratio가 작아질수록 (KV가 더 많이 잘려야 함) 효과가 안 나옴

baseline-corrected `forward_ms` 평균 (action 전 평균 → 종료 직전 1000-token 정상상태):

| 액션 | pre fwd | post fwd | Δfwd | baseline-corrected (vs baseline Δ+49) |
|---|---|---|---|---|
| Baseline (GPU)        | 121 ms | 170 ms | +49 ms | — |
| H2O keep-25 %         | 135 ms | 174 ms | +39 ms | **−10 ms** |
| H2O keep-50 %         | 139 ms | 171 ms | +32 ms | **−17 ms** |
| H2O keep-75 %         | 142 ms | 193 ms | +51 ms | **+2 ms** |

KV의 75 %를 버리는 keep-25 %가 25 %만 버리는 keep-75 %보다 단 7 ms 더 빠르다. attention 비용은 sequence length에 quadratic이므로 keep-25 %는 압도적으로 빨라야 정상이다 → eviction의 compaction 오버헤드가 attention 절감을 거의 모두 잡아먹고 있다는 신호.

### (3) RSS는 keep-ratio와 무관하게 동일

| 액션 | RSS pre→action+3s |
|---|---|
| H2O keep-25 % | 3927 → 3927 MB |
| H2O keep-50 % | 3927 → 3927 MB |
| H2O keep-75 % | 3927 → 3927 MB |

(이 부분은 KV buffer가 max-seq-len으로 pre-allocated되는 설계상 정상일 수 있음 — ★ 확인 필요. cache buffer를 truncate하지 않는다면 메모리 절감을 paper에서 주장할 수 없으므로 design 결정이라면 명시 필요.)

### (4) Wall-clock TBT 영향

| 액션 | post-action TBT mean | TBT p99 |
|---|---|---|
| Baseline | 187 ms | 233 ms |
| H2O keep-25 | 187 ms | 264 ms |
| H2O keep-50 | 182 ms | 264 ms |
| H2O keep-75 | 203 ms | 304 ms |

모든 H2O 변형이 baseline 대비 p99가 +30~70 ms — eviction 호출 시점의 stall이 tail에 집중됨.

## Hypothesis (root cause 후보)

1. **Eviction trigger를 매 N 토큰마다 (N≈18) 또는 매 cache 임계치 초과마다 계속 호출** — 현재 H2O는 `should_evict() → false` (signal-driven)이므로 트리거는 외부에서 옴. resilience executor 또는 generate loop가 eviction directive를 너무 자주 재발행하는 듯.
   - 확인: `Eviction completed` 로그의 timestamp 간격을 보면 토큰 단위 주기인지 시간 단위 주기인지 알 수 있음.
2. **target_len이 작게 들어와서 매번 큰 폭의 compaction이 발생** — 첫 호출은 1398 토큰 evict (정상). 두 번째 호출부터 33 토큰만 evict인데 매 호출마다 compact_keep_positions 전체를 수행 → 작은 evict size에 비해 sort + memcpy의 amortized 비용이 큼. 동적 batching이나 lazy compaction이 필요할 수 있음.
3. **compact_keep_positions가 실제로 `keep_all` 길이만큼만 memcpy하는지** — 만약 max-seq-len 전체 슬롯을 매번 다루는 구현이라면 evict size와 무관하게 비용이 일정해 호출 빈도가 그대로 비용에 반영됨.

## Suggested fix

- **단기**: H2O eviction directive 발행 빈도를 throttle (예: 100 토큰당 최대 1회). 또는 `evict_with_scores` 내부에서 "마지막 eviction 이후 추가된 토큰이 임계치 미만이면 noop" 가드 추가.
- **중기**: lazy compaction — `cache.current_pos`만 줄이고 실제 memcpy는 다음 prefill / 임계치 도달 시 한꺼번에. 단 attention 커널이 logical position을 따라가야 함.
- **확인 필요**: `compact_keep_positions` 실제 memcpy 범위 (전체 vs `keep_all.len()`만). `cache.k_buffer.shape()`가 max-seq라면 compact는 어떻게 in-place로 동작하는지.
- **Paper-side**: eviction 호출 빈도를 줄였을 때의 fwd 시간 재측정이 필요. 현 결과로는 H2O를 "효과 없음"으로 잘못 결론낼 위험.

## Related

- `pact2026/experiments/results/action_profile_v2/HANDOFF.md` — 측정 protocol 및 데이터 layout
- 로그 샘플: `evict_h2o_25_tbt150_run1_gen.log` ("Eviction completed: policy='h2o', removed=33, new_pos=466" 패턴 다수)

## Resolution (2026-04-17)

**Fix**: `EvictionHandler::handle()`과 `CacheManager::run_policy_eviction()`에 `MIN_EVICT_TOKENS = 64` 가드 추가 (`engine/src/core/pressure/eviction_handler.rs`, `engine/src/core/cache_manager.rs`).

- 제거할 토큰 수 (`current_pos - target_len`)가 64 미만이면 eviction을 skip하고 `ActionResult::NoOp` (혹은 `EvictionResult { evicted: false }`) 반환.
- Sticky eviction 경로(`adjusted_ratio = target_pos / current_pos`)에서 발생하던 "33 토큰씩 반복 compaction" 패턴 차단.
- Lazy compaction은 중기 과제로 별도 이슈화 예정.
