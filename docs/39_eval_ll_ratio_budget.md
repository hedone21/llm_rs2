# 39. eval-ll 모드 비율 기반 KV 버짓 지원

## 배경

PACT 2026 논문에서 KV cache eviction의 품질 영향을 벤치마크별로 공정하게 비교해야 한다.
현재 실험에서 고정 토큰 수 버짓(예: `--kv-budget 250`)을 사용하고 있는데,
벤치마크마다 prompt 길이가 다르기 때문에 동일한 절대값이 전혀 다른 압축률을 의미한다.

| 벤치마크 | prompt 길이 | budget=250 유지율 |
|----------|-----------|------------------|
| DREAM | ~300 tok | ~83% |
| RACE-H | ~500 tok | ~50% |
| NIAH 8블록 | ~800 tok | ~31% |
| NIAH 16블록 | ~1,600 tok | ~16% |

같은 "budget=250"이 DREAM에서는 거의 무손실이고 NIAH 16블록에서는 극한 압축이 된다.
Cross-benchmark, cross-model 비교를 위해 **"KV cache의 X%를 유지"** 라는 비율 기반 버짓이 필요하다.

## 구현 상태

**구현 완료.** `--kv-budget-ratio`가 eval-ll 모드에서 per-question으로 동작한다.

### 데이터 흐름

```
generate.rs (eval-ll 진입부)
  ├─ ratio_mode = kv_budget_ratio > 0.0
  ├─ effective_budget = 0 (ratio_mode일 때, per-question에서 결정)
  ├─ EvalConfig { kv_budget_ratio, effective_budget: 0, ... }
  ├─ EvictionHook::new(..., hook_budget=0, ...)
  └─ run_eval_ll_generic()
      └─ eval_loop.rs (per-question 루프)
          └─ for question in questions {
              ├─ hook.reset_caches()
              ├─ prompt 토큰화 → prompt_len
              ├─ if kv_budget_ratio > 0.0:
              │   ├─ budget = (prompt_len × ratio).max(1)
              │   ├─ hook.set_effective_budget(budget)
              │   └─ effective_eval_config.effective_budget = budget
              ├─ run_prefill(effective_eval_config)
              │   └─ budget_mode = effective_budget > 0 → true
              │   └─ run_chunked_prefill() (first_chunk = budget)
              └─ choice 평가 (hook.post_decode_step에서 eviction 체크)
          }
```

### 수정된 파일

| 파일 | 변경 내용 |
|------|----------|
| `eval/output.rs` | `EvalConfig`에 `kv_budget_ratio: f32` 필드 추가 |
| `eval/hook.rs` | `StepHook` trait에 `set_effective_budget()` 메서드 추가 (default no-op) |
| `eval/eviction_hook.rs` | `set_effective_budget()` 구현 (self.effective_budget 갱신) |
| `eval/eval_loop.rs` | per-question 루프에서 ratio 기반 budget 계산 + hook/config 갱신 |
| `generate.rs` | eval-ll 진입부에서 ratio를 EvalConfig로 전달, EvictionHook에 hook_budget=0 |

### chunked prefill 연쇄 효과

`run_chunked_prefill`에서 `effective_budget`은 두 가지 역할을 겸한다:

1. **first_chunk_len**: 첫 번째 배치 prefill의 토큰 수 (= budget)
2. **eviction trigger**: `kv_caches[0].current_pos() > effective_budget` 체크

ratio mode에서 per-question budget이 바뀌면 chunk size도 함께 바뀐다.
prompt 800 토큰에 ratio=0.4이면 first_chunk=320, 나머지 480 토큰은 1-by-1 decode+eviction.

## 기대 동작

```
# 모든 question에서 prompt의 40%만 KV cache에 유지
generate --eval-ll --eval-batch batch.json \
  --eviction-policy h2o --kv-budget-ratio 0.4

# question A (prompt 800 tok) → effective_budget = 320
# question B (prompt 1600 tok) → effective_budget = 640
```

- `--kv-budget-ratio`와 `--kv-budget`은 상호배타 (기존과 동일)
- ratio가 설정되면 question마다 `effective_budget`이 재계산됨
- per-question JSON 출력에 실제 적용된 `effective_budget` 값이 포함됨
- PPL 모드의 기존 ratio 동작은 변경 없음

## 수용 기준

- [x] `--kv-budget-ratio 0.4 --eviction-policy h2o`로 eval-ll 실행 시, prompt 길이에 비례한 eviction 발생
- [x] 동일 ratio에서 prompt 길이가 다른 question들의 유지율이 일정 (±1 token 허용)
- [x] per-question JSON에 `effective_budget` 필드가 실제 적용값을 반영
- [x] `--kv-budget` (절대값) 모드의 기존 동작에 영향 없음
- [x] PPL 모드의 기존 ratio 동작에 영향 없음
- [x] KiviHook에는 해당 없음 (KIVI는 residual-size 기반, 별도 체계) — `set_effective_budget()` default no-op

## 참고 자료

- 현재 PPL ratio 구현: `generate.rs:3088-3115`
- eval 루프 리팩토링 문서: `docs/38_eval_refactoring.md`
- StepHook 트레이트: `eval/hook.rs`
- EvictionHook: `eval/eviction_hook.rs` (`.effective_budget` pub 필드)
- 실험 스크립트: `experiments/benchmarks/run_eval.py` (이미 `--kv-budget-ratio` 옵션 지원)
