# 카테고리별 Sync Bisection 결과

측정 환경:
- Jetson AGX Xavier (sm_72, UMA=true)
- Llama 3.2 1B, F16 weight, F16 KV
- Prompt `"The capital of France is"`, `-n 30`, `--temperature 0.0`
- `--cuda-weights-device` 항상 활성화 (P3)
- Baseline 정상 출력: `"Paris. It's the most visited city in Europe and one of the world's most popular tourist destinations."`

## 1단계: 단일 카테고리 활성 (나머지 defer)

`--cuda-sync-policy custom:<cat>,fallback` 형태. `fallback`은 safety 목적으로 항상 포함 (CPU fallback path에서 stale GPU write 방지).

| 활성 카테고리 | 정확성 | tok/s | 비고 |
|--|--|--|--|
| `elementwise,fallback` | CORRECT | 33.6 / 34.1 / 33.9 → **avg 33.87** | 유일한 correct-preserving 단독 카테고리 |
| `rmsnorm,fallback` | garbage ("Cell:W:K:IC…") | 35.0 | |
| `rope,fallback` | garbage ("our, you will have…") | 35.2 | |
| `matmul,fallback` | garbage ("a dea") | 29.4 | matmul sync 자체는 비쌈 |
| `kv_scatter,fallback` | garbage | 35.8 | |
| `attention,fallback` | garbage | 35.7 | |
| `gather,fallback` | garbage ("isisinininsin…") | 36.7 | |

**발견**: `Elementwise` 카테고리만 단독으로 correctness 보존. 다른 카테고리는 단독으로는 부족.

## 2단계: 1개 카테고리 제거 (나머지 on)

`all - <drop>` 형태로 bisection.

| 제거한 카테고리 | 정확성 | tok/s | 비고 |
|--|--|--|--|
| (no drop = all) | CORRECT | **28.17 avg** (28.5/27.9/28.1) | baseline |
| `rmsnorm` | CORRECT | 28.4 | sync 없어도 무방 |
| `rope` | CORRECT | 28.5 | sync 없어도 무방 |
| `matmul` | CORRECT | **31.0** | matmul sync 제거 효과 큼 |
| `kv_scatter` | CORRECT | 28.5 | 무방 |
| `attention` | CORRECT | 28.6 | 무방 |
| `gather` | CORRECT | 28.4 | 무방 |
| `elementwise` | **garbage** ("dcultlldcleblrender…") | 28.9 | **Elementwise만 필수** |

**확정**: Elementwise 이외 모든 카테고리는 correctness에 불필요 → 제거 가능.

## 3단계: Elementwise 내부 세분화

Elementwise를 3개 서브카테고리로 분해:
- `ElemAdd` = `add_assign` (residual, 32회/token)
- `ElemAct` = `silu_mul`, `gelu_tanh_mul` (FFN activation, 16회/token)
- `ElemMisc` = `scale`, `softmax`, `cast_f16_f32`, `add_row_bias` (decode에서 대부분 0회)

| 활성 서브카테고리 | 정확성 | tok/s |
|--|--|--|
| `elem_add,fallback` | CORRECT | 34.8 / 35.0 / 34.5 → **avg 34.77** |
| `elem_act,fallback` | garbage ("Dealer'ssandersinand…") | 35.7 |
| `elem_misc,fallback` | garbage ("isisinin…") | 37.0 |
| `elem_add` (fallback 없이) | CORRECT | 34.8 / 34.7 / 34.7 → avg 34.73 |

**핵심 발견**: `add_assign` 1개 launch의 sync만 유지하면 correctness 완벽 보존.
- Residual `add_assign` (layer당 2회, 16 layer = **32회/token**) 이 유일한 load-bearing sync.
- 이는 Jetson UMA 특성상 `add_assign` 직후 `rms_norm_oop`의 동일 버퍼 read가 일어날 때, 이전 layer의 GPU write와 CPU-visible 캐시 간 coherency가 필요하기 때문으로 추정.

## 최종 `minimal` preset 검증 (재빌드 포함 3회)

`--cuda-sync-policy minimal` = `ElemAdd + FallbackPre`

| Run | 정확성 | tok/s |
|--|--|--|
| 1 | CORRECT | 34.6 |
| 2 | CORRECT | 34.9 |
| 3 | CORRECT | 34.9 |
| **avg** | | **34.80** |

## 성능 요약

| 정책 | tok/s (avg) | correctness | sync 횟수/token (이론) | 비고 |
|--|--|--|--|--|
| `all` | **28.17** | CORRECT | ~278 | 현재 production |
| `minimal` | **34.80** | CORRECT | **~32** (ElemAdd 32 + FallbackPre 0) | 새로 도입 |
| `llamacpp` | 36.3 | garbage | ~0 (FallbackPre만) | 참고용 |
| `none` | 36.9 | garbage | 0 | legacy `--cuda-defer-sync` |

- `all` → `minimal` 개선: **+6.63 tok/s (+23.5%)**.
- llama.cpp 35.19 tok/s 대비 격차: **-1.1%** (이전 -19.9%에서 94.5% 해소).
- sync 횟수: 278 → 32 (~88% 감축). llama.cpp의 14 대비 여전히 ~2.3배지만, add_assign 32회는 residual 설계상 제거 불가.

## Raw 로그

`.agent/research/2026-04-19_sync_bisect/raw/`:
- `policy_all_run{1,2,3}.log`
- `policy_minimal_run{1,2,3}.log`
- `policy_none_run1.log`, `policy_llamacpp_run1.log`
- `only_<cat>_run1.log` (7개 단독 카테고리 실험)
- `drop_<cat>_run1.log` (7개 1-drop 실험)
- `only_elem_add_run{1,2,3}.log`, `only_elem_add_nofb_run{1,2,3}.log`
- `only_elem_act_fallback_run1.log`, `only_elem_misc_fallback_run1.log`
- `combo_*_run1.log`
