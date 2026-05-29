# Score Formula Consolidation — 설계 노트

> **상태**: **구조 확정 / 구현 대기** (2026-05-29 `/improve-codebase-architecture` grill 후보 ① 산출물). interface-preserving 내부 deepening 2건.
> **출처**: improve-codebase-architecture 후보 ① (deep-module 렌즈). 후보 리포트 맥락은 `.agent/todos/handoff_kv_weight_grill_2026_05_28.md` R0′.
> **의존**: GPU conformance 검증은 후보 ② harness(`arch/backend_conformance_harness.md`)의 CPU-reference-as-oracle 위에 얹힌다.

## 0. 한 줄 요약

리포트는 ①을 "ScoreCollector capability + GpuScoreAccess 흡수(inversion)" 라는 큰 추상화로 그렸으나, **실측 호출 분포 + deletion test 로 깎아낸 결과 정직한 deepening 은 두 개의 작고 내부적인, interface-preserving 변경**이다. 소비자가 보는 interface(`importance_scores() -> &[f32]`, `evict_with_scores(&[f32])`)는 **무변경**.

## 1. 문제

score formula(`(per-layer, per-head attention 가중치) → per-token importance`; layer 가로질러 MAX, head SUM, step 가로질러 exponential decay)가 **3곳에 중복**:
- CPU flat: `accumulate_layer()` (`engine/src/inference/attention_scores.rs:167`)
- CPU GQA: `accumulate_layer_gqa()` (같은 파일:198) — Q-head → KV-group 평균
- GPU: `score_reduce.cl` `kernel_score_fused_reduce` (`end_step` 의 fused reduce)

그리고 `GpuScoreAccess` (backend.rs:184) 9-accessor trait 이 내부 GPU 버퍼 layout 을 노출(겉보기 shallow).

## 2. ①-a — `GpuScoreAccess` accessor 정리 (9 → 2, inversion 없음)

**실측 호출 분포가 결정적** — 대부분 이미 dead 거나 내부 전용:
- `n_heads_q` / `n_layers` / `steps_accumulated`: **외부 호출 0** → trait 에서 삭제.
- `current_layer_idx` / `layer_offset_elems`: **mod.rs 내부 2곳뿐** (`6028`, `6111`), 외부 0 → trait 이 아닌 **inherent 로 강등**.
- `set_current_layer_idx`: **forward_gen 1곳** (`forward_gen.rs:508`) → 한 줄 유지, 또는 plan 경로처럼 offset 을 `attention_gen` 인자로 직접 전달해 mutable 상태 제거.
- eviction 소비자(`eviction_hook.rs:321`, `eviction_trigger.rs:77`)는 `is_active()` + `sync_to_cpu()` 만 사용 — layout accessor 0.

→ trait 9 → 2 (`is_active` / `set_active`). **opaque sink / inversion 안 함** (§5 기각 이유).

## 3. ①-b — score formula 단일화 (pure fn + conformance)

CPU-Rust ≠ GPU-`.cl` 라 formula 코드 물리 공유는 불가. 현실적 "single home" = **권위 정의 1개 + conformance-gated replica**:

```
score_formula.rs (신규)  ← 권위 정의(spec) = #2 oracle
  fold_layer(step_acc: &mut [f32], layer_scores: &[f32], grouping: HeadGrouping)
        // flat / GQA 를 grouping 인자로 물리 통합 (CPU 3 → 2)
  apply_decay(cumulative: &mut [f32], step_acc: &[f32], decay: f32, step: usize)
        // 상태는 인자로 — 입력에 대해 결정적(pure)

AttentionScoreAccumulator  ← pure fn 의 thin stateful caller.
                              소비자 read surface 그대로 (importance_scores() -> &[f32], attention_scores.rs:278)
GpuScoreAccumulator + score_reduce.cl  ← 커널 그대로. #2 harness(device-side)로 pure fn 대비 일치 검증.
                                          import_gpu_scores (attention_scores.rs:300) → CPU accumulator 흐름 그대로
EvictionPolicy::evict_with_scores(&[f32])  ← interface 무변경 (eviction/mod.rs:26)
```

**왜 신규 trait 이 없는가**: 소비자는 importance 를 CPU accumulator **단일 surface** 에서만 읽고, GPU 는 거기로 `import_gpu_scores` 한다. GPU impl 은 소비자에게 직접 소비되지 않음 → 공유 trait 은 유효 consumer-facing impl 1개 → deletion test 실패 → shallow. formula 단일화는 pure fn + conformance 로 달성되지, trait 으로 달성되는 게 아니다.

부수 이득: 작은 pure fn 은 GPU `.cl` 과 **코드 리뷰에서 눈으로 대조 가능** → 테스트(확률적)뿐 아니라 review 로도 divergence 포착.

## 4. tested-duplication 의 정직한 한계

GPU `.cl` ↔ CPU pure fn 의 중복은 substrate 때문에 **환원 불가능**. conformance(#2)는 "테스트된 입력에서의 divergence" 만 잡음 — 물리 공유와 동등하지 않다(확률적). "한 쪽 고치고 다른 쪽 깜빡" 같은 흔한 divergence 는 확실히 잡고, 미묘한 입력 의존 divergence 는 커버리지 비례. 그래서 ①-b 의 주장은 "중복 제거" 가 아니라 **"환원 불가능한 1쌍 중복을 oracle 기반 conformance 로 관리 가능한 risk 로 낮춤"**.

## 5. 기각된 대안 (미래 review 가 재제안하지 않도록)

- **opaque-sink inversion** (`GpuScoreAccess` → "accumulator 가 자기 write 타깃을 커널에 바인딩"): **기각**. (a) layout leak 을 *남의 커널 ABI(flash attn arg 40~43)* 결합으로 맞바꿈 — 결합 품질 악화, (b) plan 경로가 offset 을 build 시점 pre-bake(plan.rs:4078) 하므로 런타임 바인딩 모델과 충돌 → `layer_offset_elems` 살아남음, (c) 외부 layout leak 실수요가 `set_current_layer_idx` 1곳뿐이라 ROI ~0.
- **`ScoreAccumulator` / `ScoreCollector` capability trait** (CPU+GPU 공유): **기각**. deletion test 실패(소비자는 CPU 단일 surface 에서 `&[f32]` 읽음, GPU 는 import-in → 유효 impl 1개). CPU(Vec fold) / GPU(buffer + fused reduce + import) write 메커니즘이 근본적으로 달라 공유 behavioral trait 이 한쪽에 억지.

→ ①의 정직한 형태는 pure fn 추출 + accessor 청소 + GPU conformance. 새 trait·inversion 없음.

## 6. 효과 / 위치

- deletion test ✓ — formula 가 흩어진 3곳 → CPU pure fn 1곳(권위) + GPU replica(tested). accessor 9 → 2.
- interface 무변경(`importance_scores`/`evict_with_scores`) — 내부만 응축 = 좋은 deep-module 변화.
- `①-b` 의 "CPU reference = oracle" 은 #2 의 oracle 결정(CPU backend = oracle)을 score 도메인에서 구체화.
- 구현 위치: cleanup 트랙 (Phase α-K 인접). GPU conformance 는 #2 harness device-side 에 추가.
