# Phase 4-4.8 S25 OpenCL plan-path 진단 sprint 측정

**일시**: 2026-05-18
**HEAD pre**: `badf4b44` (Phase 4-4.7 plan-aware ModelForward, sticky lock-out)
**HEAD post**: (Phase 4-4.8 publish fix, 본 commit)
**디바이스**: Galaxy S25 R3CY408S5SB (Adreno 830 OpenCL)
**모델**: qwen2.5-1.5b-q4_0.gguf

## Root cause 식별 (C2)

Phase 4-4.7 G7' FAIL +7.73%는 `ModelForward.sticky_disabled` 트리거에서 비롯됐다.
`LLMRS_BUILD_PLAN_TRACE=1`로 line-by-line None-return 경로를 추적한 결과:

```
[build_plan-trace] layer0 wq key=0xb40000779f1fa5d0 is_NoshuffleWeightBuffer=false
[build_plan-trace] None at Q4_0 SOA entry missing layer=0 (transformer.rs:2090)
[fwd-trace] build_plan returned None → sticky lock
```

`is_NoshuffleWeightBuffer=false` = `prepare_noshuffle_buffers`가 weight를
NoshuffleWeightBuffer로 swap하지 못한 상태로 끝났음을 의미.

`LLMRS_NOSHUFFLE_SOA_TRACE=1` 보조 trace:
```
[noshuffle-soa-trace] MISS key=0xb400007b09f67c80 registry_size=196
   sample_keys=[b400007b09f91b20, ...]
```

196개 entry는 정상 등록되었으나 lookup key (AOS cl_mem 주소)가 registry key
(d_buf 주소)와 disjoint. SOA 등록은 됐는데 layer tensor는 원래 AOS 그대로.

### 진짜 원인

`transformer.rs::prepare_noshuffle_buffers` (l.1101~1146)이 **RCU pattern의
publish step을 누락**했다. 구체적으로:

```rust
let snapshot = slot.load_weights();
let mut layer = (*snapshot).clone();         // local clone
for weight in [...] {
    process_weight(weight, true)?;            // layer.wq 등 buffer를 swap
}
// ❌ slot.store_weights_same_dtype(Arc::new(layer)) 호출 누락
//   → 변경된 layer는 scope exit 시 drop, slot은 snapshot 유지
```

결과: `slot.load_weights().wq.buffer()`는 영원히 원래 AOS cl_mem 반환.
`build_plan`은 `cl!(layer.wq).as_ptr()`로 lookup → MISS → `None` 반환 →
ModelForward sticky lock-out. **plan path는 happy/production 양쪽 모두 미사용
상태**가 됐고, decode는 standard Q4_0 GEMV fallback으로 동작.

## Fix (C3)

```rust
let mut layer_mutated = false;
for weight in [...] {
    if process_weight(weight, true)? {
        layer_mutated = true;
    }
}
// partition_ctx swap도 동일하게 mutation flag set
if layer_mutated {
    slot.store_weights_same_dtype(Arc::new(layer));   // ✅ RCU publish
}
```

## 게이트 결과

| Gate | 기준 | 결과 |
|---|---|---|
| G1~G4 host | build/fmt/clippy/test | PASS (cli.rs pre-existing clippy 제외) |
| G6' bit-identical 32 tok (`--greedy --repetition-penalty 1.1`) | 출력 string 동일 | **PASS** — "...104 square kilometers (km2). The city is divided into" |
| G7' avg_tbt n=5 Δ ≤ 5% vs Phase 4-4.7 post (32.06 ms) | Δ ≤ 5% | **FAIL +13.7%** (median 36.52 ms, mean 36.51) |

### G7' n=5 세부

| run | avg_tbt (ms) |
|-----|-------------|
| 1 | 36.46 |
| 2 | 36.52 |
| 3 | 36.51 |
| 4 | 36.52 |
| 5 | 36.54 |
| **median** | **36.52** |
| stdev | 0.03 |

### plan ON vs --no-gpu-plan 비교 (publish fix 동일 binary)

| 조건 | median avg_tbt | 메모 |
|---|---|---|
| plan ON (default) | 36.48 ms | build_plan SUCCESS, NoshuffleWeightBuffer 활성 |
| `--no-gpu-plan` | 36.53 ms | plan-bypass, matmul_q4_0의 m==1 noshuffle dispatch 자동 |

차이 거의 없음 (Δ ≤ 0.2%) → plan path 자체는 정상 진입했으나 Adreno에서
noshuffle GEMV가 standard Q4_0 GEMV 대비 speedup 미미.

## 결론

- **C2/C3 목표 달성**: Phase 4-4.7 sticky lock-out root cause 식별 + 정확성 fix
  (`prepare_noshuffle_buffers` RCU publish 누락 해소).
- **G6' PASS**: bit-identical 출력 보존.
- **G7' FAIL**: publish fix가 NoshuffleWeightBuffer를 활성화하면서 매 token의
  matmul_q4_0이 noshuffle GEMV path로 dispatch. Phase 4-4.6 baseline 측정
  (29.76 ms) 대비 +22.7%, Phase 4-4.7 post (32.06 ms) 대비 +13.7%.
- **격리 측정 (plan ON vs --no-gpu-plan)**으로 회귀의 origin이 plan path가
  아니라 **NoshuffleWeightBuffer + matmul_q4_0 noshuffle dispatch**임을 확인.
  Adreno 830에서 noshuffle Q4_0 GEMV가 standard AOS Q4_0 GEMV 대비 측정상
  느림 — feedback `feedback_adreno_subgroup_reduce.md`의 "이론에 끌리지 말고
  cross-run 실측 후 결정" 원칙에 정확히 부합하는 사례.

## 다음 sprint 후보

1. **Phase 4-4.9 noshuffle Adreno tuning**: matmul_q4_0의 m==1 noshuffle
   dispatch를 ENV-gated로 비활성화 (`LLMRS_DISABLE_NOSHUFFLE_DECODE=1`) 후
   measurement. 또는 noshuffle GEMV kernel 자체 cross-run profiling으로
   Adreno-friendly variant 도출.
2. **Phase 4-4.10 keep_for_cpu publish path**: `keep_for_cpu=true` 경로에서는
   buffer swap 없이 register만. 그 경우 publish 없이도 AOS 주소 lookup
   match → plan path 사용 가능 + matmul_q4_0의 noshuffle dispatch는 여전히
   AOS path 유지.
3. **Phase 4-5 chat 진입**: Phase 4-4.9가 별도 트랙으로 가는 동안 chat REPL
   재작성 진행.
