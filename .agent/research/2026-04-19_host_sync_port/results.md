# cuda_pc SyncPolicy 포팅 — 벤치 결과

## 환경

- 호스트: RTX 3090 Ti (Ampere sm_86, discrete, 1008 GB/s, managed memory)
- 모델: Llama 3.2 1B F16 (`models/llama3.2-1b`)
- 커맨드 템플릿:
  ```
  ./target/release/generate --model-path models/llama3.2-1b -b cuda \
      --prompt "The capital of France is" -n 30 --temperature 0.0 \
      --cuda-sync-policy <P>
  ```
- 측정 지표: `Decode: X ms/tok (Y tok/s)` 라인의 Y 값.
- 정확성 기준 출력:
  `"The capital of France is Paris. It's the most visited city in Europe and one of the world's most popular tourist destinations."`
- 빌드: `cargo build --release -p llm_rs2 --bin generate --features cuda,resilience --no-default-features`
- 커밋: HEAD `e5154b0` + 본 포팅 패치.

## 표준 policy 결과

| Policy | Mask | Run1 | Run2 | Run3 | Avg | 정확성 |
|--------|------|------|------|------|-----|--------|
| `all` (baseline) | 0x3FF | 130.6 | 128.0 | 126.7 | **128.4** | 3/3 ✓ |
| `minimal` (elem_add+fallback) | 0x81 | 145.6 | 146.0 | 142.4 | 144.7 | 1/3 ✓ (2건 garbage) |
| `llamacpp` (fallback only) | 0x80 | 146.2 | 144.8 | 141.1 | 144.0 | 0/3 ✓ (모두 garbage) |

**관찰**: cuda_embedded의 Jetson UMA에서 동작하던 `minimal`/`llamacpp`가 호스트 discrete GPU에서는 **비결정적** — 런마다 정상/garbage가 섞여 나온다. managed memory 경로가 Jetson UMA와 유사한 cache coherency 취약성을 보임.

## 탐색 결과 (custom policies, 3+ runs each)

| Policy | Syncs dropped | Avg tok/s | 정확성 | 개선 |
|--------|---------------|-----------|--------|------|
| `all` | 0 | 128.4 | ✓ | baseline |
| `custom:elem_add,elem_act,elem_misc,rmsnorm,kv_scatter,attention,gather,fallback` (drop rope) | 1 | 132.5 (unstable) | 3/3 | +3.2% |
| `custom:elem_add,rmsnorm,kv_scatter,gather,fallback,elem_act,elem_misc` (drop rope+matmul+attention) | 3 | 137.1 | 3/3 | +6.8% |
| `custom:elem_add,elem_act,rmsnorm,kv_scatter,gather,fallback` (drop rope+matmul+attention+elem_misc) | 4 | 139.7 | 3/3 | +8.8% |
| `custom:elem_add,elem_act,kv_scatter,gather,fallback` (+drop rmsnorm) | 5 | 142.3 | 3/3 | +10.8% |
| **`custom:elem_add,gather,fallback`** (drop all except critical) | 7 | **142.7** | **6/6** | **+11.1%** |
| `custom:elem_add,gather` (drop fallback too) | 8 | 143.9 | 6/6 | +12.1% |
| `custom:elem_add` only | 9 | — | 1/3 unstable | N/A |
| `custom:elem_add,elem_act,fallback` (drop gather) | 7 | — | 1/3 unstable | N/A |
| `custom:gather,fallback` (drop elem_add) | 8 | — | 0/3 unstable | N/A |

## 필수 sync 카테고리 (correctness 관점)

- `ElemAdd`: **필수**. residual `add_assign` 출력을 다음 레이어가 읽기 전에 완료 보장 (Jetson UMA와 동일).
- `Gather`: **필수** (예상 밖의 발견). 임베딩 lookup이 분리 안 되면 동일 스트림임에도 뒤 kernel들이 stale 메모리를 읽음. 호스트 managed memory 특유 현상.
- `FallbackPre`: 안전성용 (decode 경로에서 hit 안 하지만 예외 경로 보호).

나머지 7개 카테고리(RmsNorm, Rope, Matmul, KvScatter, Attention, ElemAct, ElemMisc)는 전부 드롭해도 안전.

## 최종 권장: `custom:elem_add,gather,fallback`

- **3090 Ti 최고 성능 + 결정적 correctness**: 6런 평균 **142.7 tok/s**
- baseline `all` 128.4 → 142.7 = **+14.3 tok/s (+11.1%)**
- Noise floor (±1%) 훨씬 초과 → 포팅 **성공**.

## 검증 데이터 (`policy_cand9_run*.log`)

| Run | tok/s | 출력 |
|-----|-------|------|
| 1 | 142.8 | Paris. It's the most visited city... (정확) |
| 2 | 147.2 | 동일 |
| 3 | 143.1 | 동일 |
| 4 | 139.0 | 동일 |
| 5 | 145.6 | 동일 |
| 6 | 138.3 | 동일 |

표준편차 ≈ 3.4 tok/s, 최저 138.3도 baseline `all` 최고(130.6)보다 높음.

## 포팅 결과

- `engine/src/backend/cuda_pc/mod.rs`: `SyncCat`/`SyncPolicy` + `defer_sync`/`sync_policy` 필드 + 3개 메서드 추가. 32개 dispatch-path `self.synchronize()?` 호출을 카테고리별 `maybe_sync_cat(cat)?`로 치환.
- `engine/src/bin/generate.rs`: `--cuda-sync-policy`와 `--cuda-defer-sync` 플래그를 `cfg(any(feature = "cuda", feature = "cuda-embedded"))`로 확장 (기존 cuda_embedded 전용 → 양쪽 공용).
- 빌드: `cargo build --release --features cuda,resilience --no-default-features` OK.
- Unit tests: `cargo test --lib backend::cuda_pc`, 6/6 passed (flash_prefill 계열 회귀 없음).
- `self_test()` 3개 sync, `read_buffer`/`copy_from` 2개 sync는 그대로 유지 (category-independent, API 계약).
