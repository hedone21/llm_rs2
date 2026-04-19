# cuda_embedded Sync Bisection (2026-04-19)

## 목표

`cuda_embedded` 백엔드의 `maybe_sync()` 호출(토큰당 ~278회)을 카테고리별로 bisect 하여
llama.cpp 수준(토큰당 ~14회)에 근접하면서도 correctness를 보존하는 최소 sync 세트를 발견한다.

## 배경

- HEAD `4070ddc`: baseline 28.4 tok/s, llama.cpp 35.19 tok/s, **gap -19.9%**.
- Phase C1 전면 defer (`--cuda-defer-sync`): 37.0 tok/s이지만 **garbage 출력**.
- 기존 bisection 시도 2회 (write_buffer / gather hard sync) 실패 — 모두 특정 sync만 복구해도 여전히 garbage.
- llama.cpp가 14/token으로도 작동하는 이유 파악 필요 → 카테고리 bisect.

## 결론

**`add_assign` (residual accumulate) 의 per-op sync만 유지**하면 correctness 완벽 보존 + 성능 대폭 개선.

| 지표 | Before (`all`) | After (`minimal`) | Δ |
|--|--|--|--|
| tok/s (3회 평균) | 28.17 | **34.80** | **+6.63 (+23.5%)** |
| 출력 정확성 | CORRECT | CORRECT | — |
| per-token sync 횟수 | ~278 | ~32 | -246 (-88.5%) |
| llama.cpp gap | -19.9% | **-1.1%** | 94.5% 해소 |

## 근본 원인

Jetson UMA의 pinned host-mapped 메모리에서 GPU write → 다음 kernel read 사이의 캐시 일관성 보장에
**residual `add_assign` 직후 동일 버퍼를 다시 읽는 `rms_norm_oop`가 coherency barrier를 요구**한다.
이는 `add_assign`이 다음 layer의 attention 입력을 누적하는 구조적 특성 때문으로, 다른 kernel들은
in-place 연산이 아니거나 CUDA stream 내부 ordering으로 충분.

단일 카테고리 실험에서 `Elementwise`만 단독으로 correctness를 유지했고, 세분 결과 그 안에서도 실제
필수는 `add_assign` (ElemAdd) 뿐. `silu_mul`/`gelu_tanh_mul` (ElemAct), `softmax`/`scale`/`cast`/`add_row_bias`
(ElemMisc)은 sync 불필요.

## 구현

- `engine/src/backend/cuda_embedded/mod.rs`:
  - 기존 `fn maybe_sync()` 제거 (legacy 전면 on/off 기능은 `defer_sync` 플래그가 담당).
  - `SyncCat` enum (10-way): ElemAdd/ElemAct/ElemMisc/RmsNorm/Rope/Matmul/KvScatter/Attention/Gather/FallbackPre.
  - `SyncPolicy(AtomicU32)` 비트마스크. 프리셋: `ALL`, `EMPTY`, `LLAMACPP`, `MINIMAL`.
  - `maybe_sync_cat(cat)` 헬퍼가 `defer_sync` + policy bit를 검사하여 실제 `synchronize()` 호출.
  - 기존 33개 `self.maybe_sync()` 호출 지점을 전부 카테고리-tagged 호출로 변환.
- `engine/src/bin/generate.rs`:
  - `--cuda-sync-policy <spec>` 추가. 값: `all`(default) / `none` / `llamacpp` / `minimal` / `custom:A,B,...`.
  - 기존 `--cuda-defer-sync`는 그대로 유지 (shorthand, policy 위에서 override).

## 사용법

```bash
# 권장 (correctness + perf)
generate -b cuda --cuda-weights-device --cuda-sync-policy minimal ...

# 실험 (카테고리 조합)
generate -b cuda --cuda-weights-device --cuda-sync-policy custom:elem_add,fallback ...

# Baseline (제거 전 거동)
generate -b cuda --cuda-weights-device --cuda-sync-policy all ...
```

## 산출물

- `sync_categories.md`: 33개 호출 지점 × 10개 카테고리 분류표.
- `bisection_matrix.md`: 3단계 실험 결과 (단독/1-drop/서브카테고리).
- `raw/`: 실행 로그.

## 남은 질문

- llama.cpp는 14/token으로도 정상. `add_assign` 32회와의 격차 18회는 layer-level fusion 또는 stream pipelining 이점일 것.
  추후 `add_assign + rms_norm_oop` fused kernel (이미 `add_rms_norm_oop`가 trait에는 있으나 `add_assign` 후 sync
  → `rms_norm_oop` 순서로 호출하도록 구현되어 있어 sync는 그대로 발생) 혹은 CUDA graph API로 더 줄일 수 있을지 조사.
- `--cuda-profile` 이벤트로 실측 sync 횟수를 카운트하는 기능 추가 여부는 이번 작업 범위 밖.
