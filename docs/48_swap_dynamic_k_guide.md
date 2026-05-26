# Weight Swap Dynamic-K + Sub-batch Pause 사용 가이드

LISWAP-6 weight swap 의 production 사용법. spike 회피 hard constraint 를 알고리즘적으로 보장하는 3-layer safety net 설명.

## TL;DR

```bash
./generate \
  -m qwen2.5-1.5b-f16.gguf --tokenizer-path tokenizer.json \
  --secondary-gguf qwen2.5-1.5b-q4_0.gguf \
  --force-swap-ratio 0.9 \
  --swap-incremental-per-tick 2 \
  --backend opencl --opencl-rpcmem \
  -p "<prompt>" -n 30
```

> `--secondary-gguf`는 W-AUF-1(Sprint 1, 2026-05-19) 도입 후 **deprecated alias**로 stderr 경고 1회 후 그대로 동작. 향후 AUF single-file (`--model-path foo.auf`)로 통일 예정.

`--swap-async-dispatch`, `--swap-dynamic-k` 는 **CLI default ON** (2026-05-12). 사용자는 `--swap-incremental-per-tick 2` 만 추가하면 production winner mode.

## 배경 — Weight Swap 의 메모리 spike 위험

LISWAP-6 alias 모드에서 weight 를 한꺼번에 K layer 교체하면:
- 새 weight K 개 (committed, ArcSwap pointer 갱신됨)
- 옛 weight K 개 (release_worker queue 에서 drop 대기)

→ 동시 in-flight = 2K layer. K=25 (전체 batch) 시 ~2.85 GB transient peak.

사용자 정책 (`feedback_no_memory_spike.md`): **steady-state 1 layer extra 만 허용**. K 가 클수록 burst 가 정책 위반.

## 3-Layer Safety Net

세 단계가 협력하여 spike 발생을 알고리즘적으로 방지:

```
┌─────────────────────────────────────────────────────────────┐
│ Layer 1: Dynamic-K Controller (사전, timing-based)          │
│   첫 token 1회 calibration → K_intent ∈ {1, 2} 자동 결정    │
│   hard_upper=2 (quality drift cap, Qwen 1.5B 실측)          │
│   fwd 짧아지면 K_intent 감소만 (ratchet)                    │
└─────────────────────────────────────────────────────────────┘
                          ↓ K_intent 전달
┌─────────────────────────────────────────────────────────────┐
│ Layer 2: Reactive Pause (token 시작 직전)                   │
│   if release.pending() > 0 → 이 token swap 전체 skip       │
│   이전 batch 의 잔존 spike 100% 흡수                        │
└─────────────────────────────────────────────────────────────┘
                          ↓ swap 진행 결정
┌─────────────────────────────────────────────────────────────┐
│ Layer 3: Sub-batch Pause (batch dispatch loop 안)           │
│   for i in 0..K_intent:                                     │
│     if i > 0 && release.pending() > 0 → break              │
│     dispatch_layer(i)                                       │
│   batch 자체를 layer 단위로 hardware-paced                  │
└─────────────────────────────────────────────────────────────┘
```

### Layer 1: Dynamic-K Controller (`dynamic_k.rs`)

**알고리즘** (option C, timing-based, NO probing):

```
state:
  k = 1
  drop_ms_per_layer: f32 (calibration 후)
  fwd_min_ms: f32 (forward time 의 최소값)
  hard_upper = 2  # quality drift cap (K=3+ garbage)
  margin = 0.5
  calibrated = false

Phase 0 (첫 swap batch, K=1 sync calibration):
  dispatch 1 layer
  spin-wait until release.pending == 0
  drop_ms = elapsed
  fwd_min_ms = forward_time
  safe_k = floor(fwd_min_ms * margin / drop_ms).clamp(1, hard_upper)
  k = safe_k

Phase 1+ (이후 매 token, async):
  if fwd_ms < fwd_min_ms:
    fwd_min_ms = fwd_ms
    new_safe = floor(fwd_min_ms * margin / drop_ms).clamp(1, hard_upper)
    k = k.min(new_safe)  # 감소만, 증가 금지
```

**핵심 invariant**:
- K 는 calibration 후 **monotone non-increasing**. probing 금지 (= spike 위험).
- `hard_upper=2` 는 Qwen 1.5B F16↔Q4 의 mid-forward commit quality drift 임계. K=3+ 부터 decode garbage (`.jpg` 등).
- alias 환경에서는 drop_ms ≈ 0 (sub-μs) → defensive clamp (1e-3) 으로 hard_upper 도달. 즉 alias = K=2 안정 수렴.

### Layer 2: Reactive Pause

token 시작 직전 매번 체크:

```rust
if release_worker.pending_count() > 0 {
    skip swap this token;  // K 자체는 안 깎음 — transient noise 대응
}
```

`release_worker.pending` = 옛 weight 메모리가 아직 살아있는 layer 수. = transient memory peak 의 직접 측정.

### Layer 3: Sub-batch Pause (`swap_executor.rs::execute_on_slots`)

batch 안 layer-by-layer hardware-paced:

```rust
for (i, layer) in target_layers.iter().enumerate() {
    if i > 0 && release.pending_count() > 0 {
        break;  // K=2 의도였어도 actual K=1 truncate
    }
    dispatch_layer(layer);
}
```

**첫 layer (i=0) 는 무조건 dispatch** — swap 진행 보장. **두 번째 이상은 release 가 따라잡았을 때만**.

## 보호 시나리오 매트릭스

| 시나리오 | Layer 1 | Layer 2 | Layer 3 | 결과 |
|---|---|---|---|---|
| 정상 (fwd 10ms, release 1μs) | K=2 결정 | pass | cutoff X | K=2 dispatch ✓ |
| 이전 token 잔존 release | K=2 결정 | **pause 발동** | — | 이 token swap skip ✓ |
| 이전 token outlier 잔존 (작은 잔량) | K=2 결정 | pass | **cutoff 발동** | actual K=1 ✓ |
| 이번 token fwd outlier (극단) | K=2 결정 | pass | cutoff X | K=2 dispatch, **다음 token Layer 2 가 흡수** |
| 환경 변경 (release 느려짐) | observe 로 K_intent 감소 | pause 빈도↑ | cutoff 빈도↑ | 자동 적응 ✓ |

**알고리즘 한계**: "이번 token outlier 인 동시에 release 가 ms 단위로 느려진" 동시 이벤트 = 1 token 단발 spike 가능. 즉시 다음 token 부터 흡수. 완벽 보호는 layer-immediate 통합 (후속 V4).

## 디바이스 측정 (Galaxy S25, opencl --opencl-rpcmem alias)

Qwen2.5-1.5B-F16 primary + Q4_0 secondary GGUF, n=30 decode, 3 run median.

| Mode | F+S TBT (ms) | max_release_pending | sub_batch_cutoff | decode |
|---|---:|---:|---:|---|
| 정적 K=2 (baseline) | 33.17 | 0 | (N/A) | 정상 |
| dynamic-K only | 33.38 (+0.6%) | 0 | (N/A) | 정상 |
| **dynamic-K + sub-batch pause** | **31.20** (-6%) | **0** | **0** (silent) | 정상 |

alias 환경에서 sub-batch pause cutoff 0% — silent safety net. 비-alias / 환경 변화 시 효과 발휘.

## CLI 사용법

### Production (권장)

```bash
./generate \
  -m <primary_f16>.gguf --tokenizer-path tokenizer.json \
  --secondary-gguf <secondary_q4>.gguf \
  --force-swap-ratio <0.0~1.0> \
  --swap-incremental-per-tick 2 \
  --backend opencl --opencl-rpcmem \
  -p "<prompt>" -n <decode_tokens>
```

`--swap-async-dispatch`, `--swap-dynamic-k` 는 default ON (2026-05-12).

### 명시적 정적 K (production winner 와 동등 안전, 동적 적응 없음)

```bash
./generate ... \
  --swap-incremental-per-tick 2 \
  --swap-dynamic-k=false
```

### 명시적 sync path (디버깅 / 비교용)

```bash
./generate ... \
  --swap-incremental-per-tick 2 \
  --swap-async-dispatch=false
```

### 검증 (env-gated diagnostic)

```bash
LLMRS_DYNAMIC_K_DIAG=1 \
LLMRS_SWAP_DRAIN_DIAG=1 \
LLMRS_SUB_BATCH_PAUSE_DIAG=1 \
  ./generate ...
```

stderr 로그 예:
```
[DynamicK] calibrated t=0 drop_ms=0.000 fwd_ms=4.4 safe_k=2
[SwapPeak] mode=async target_layers=25 max_release_pending=0 max_dispatcher_pending=1 sub_batch_cutoff=0
```

검증 기준:
- `max_release_pending = 0` → spike 없음 ✓
- `safe_k <= 2` → quality drift cap 준수 ✓
- decode 출력 plausible language → quality 회귀 없음 ✓

## 안전 영역 vs 위험 영역

### ✓ 검증된 안전 영역

- **Galaxy S25 + opencl --opencl-rpcmem + Qwen 1.5B + LISWAP-6 alias**: 실측 spike 0, garbage 0, F+S 31ms (이전 qnn_oppkg 측정 = Sprint 2b 통합 전 표현)

### △ 미검증 (환경 가정 다름)

| 환경 | 위험 요소 | 권장 |
|---|---|---|
| Qwen 7B 등 큰 모델 | quality drift cap K=2 가 모델 의존 | 측정 후 사용 |
| Host OpenCL backend | release time 이 ms 단위 (alias 와 다름) | dynamic-K calibration 의미 발휘. 측정 후 사용 |
| 다른 quant 페어 (F16↔Q8 등) | quality drift 임계 다름 | 측정 후 hard_upper 재조정 |
| mid-decode swap (`--swap-delay-tokens > 0`) | 미측정 | 실험적 사용 |

### ✗ 위험 (사용 금지)

- `--swap-incremental-per-tick 3` 이상 강제 — decode garbage 확정 (Qwen 1.5B). hard cap CLI 검증은 후속 작업 (P2).
- `--force-swap-ratio` 없이 swap flag 만 — silent no-op

## Trouble-shooting

### `[SwapPeak] max_release_pending > 0`

알고리즘적으로 발생하지 않아야 함. 발생 시:
1. `--swap-incremental-per-tick N` 의 N 확인 — 3 이상이면 hard cap 위반
2. mid-decode swap 사용 여부 확인
3. issue report — non-alias 환경 또는 thermal throttle 가능성

### decode 출력 garbage

1. `--swap-incremental-per-tick` 가 3 이상인지 확인 → 2 로 감소
2. secondary GGUF 파일 무결성 검증
3. `--temperature 0` 로 deterministic 비교 후 baseline (`--force-swap-ratio 0`) 과 출력 차이 확인

### F+S TBT 가 기존 baseline 보다 큼

1. `LLMRS_SUB_BATCH_PAUSE_DIAG=1` 켜고 cutoff 비율 확인. 0% 가 정상. cutoff 발생 시 release worker 가 느린 환경 → dynamic-K 가 K=1 자동 선택 중일 가능성
2. dynamic-K 비활성화 (`--swap-dynamic-k=false`) 로 정적 K=2 강제 후 비교

## 코드 위치 (다음 세션 참조)

- Algorithm doc: `engine/src/models/weights/dynamic_k.rs` (module header)
- Controller: `engine/src/models/weights/dynamic_k.rs::DynamicKController`
- Sub-batch pause: `engine/src/models/weights/swap_executor.rs::execute_on_slots` (line 466 근처)
- CLI integration: `engine/src/bin/generate.rs` `swap_*` flag 정의 + decode loop

## 후속 작업 (Backlog)

| P | 작업 | 의미 |
|---|---|---|
| R2 | `--swap-incremental-per-tick > 2` warning 또는 hard reject | hard cap CLI 검증 |
| V1 | Non-alias 환경 측정 (host OpenCL backend) | dynamic-K timing 진짜 동작 검증 |
| V2 | 다른 모델 (Qwen 7B, Gemma 2B) quality drift cap 측정 | hard_upper 모델별 재결정 |
| V3 | Mid-decode swap (`--swap-delay-tokens N`) 시나리오 측정 | production scenario 확장 |
| V4 | Layer-immediate + dynamic-K 통합 측정 | "이번 token outlier" 한계 해결 |

## 참고

- Handoff: `.agent/todos/handoff_dynamic_k_2026_05_12.md`
- 이전 handoff: `.agent/todos/handoff_swap_memory_spike_constraint_2026_05_11.md`
- Memory feedback: `feedback_no_memory_spike.md`, `feedback_swap_async_default.md`
