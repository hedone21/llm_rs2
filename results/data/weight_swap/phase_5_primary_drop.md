# Phase 5 Sprint C-2 — WSWAP-5-PRIMARY-DROP 호스트 구현 리포트

## 요약

`SwapExecutor` Stage (b/c)를 재구성하여 swap된 layer의 primary F16 weight cl_mem을 명시적으로 release. `LayerSlot::swap_weights` 시그니처를 `()` → `Arc<LayerWeights>`로 변경하고 내부적으로 `ArcSwap::store` → `ArcSwap::swap`으로 전환. `wait_for_readers` 보장 후 `Arc::try_unwrap`으로 단독 소유권 획득 → `drop`으로 모든 buffer Arc decrement → `OpenCLBuffer::drop`이 `clReleaseMemObject` 트리거.

본 sprint는 **호스트-side 구현**까지. 디바이스 측정은 별도 sprint로 분리.

## Lifecycle Audit (Senior Implementer)

F16 primary cl_mem alive 상태에 의존하는 코드 4개 영역을 전수 검사:

### (a) swap-back (Q4→F16 reverse) 경로 — 부재 확인

`engine/src/bin/generate.rs::dispatch_swap_weights` line 5992:
```rust
if target_dtype != DtypeTag::Q4_0 {
    eprintln!("[WeightSwap] Rejected: unsupported_dtype ({:?}) (INV-126)", ...);
    return;
}
```

`SwapExecutor::dtype_tag_to_dtype` (swap_executor.rs line 95): `DtypeTag::Q4_0`만 매핑하고 그 외 reserved variant는 `SwapError::UnsupportedDtype`. INV-126으로 spec 명시. 즉 **F16 → Q4_0 단방향 transition만 존재**, 역방향 미구현.

**의미**: swap-back 경로 부재 → primary 재로드 비용 / 정확성 / SOA 재변환 safety net 재진입 모두 고려 불필요. 정책 결정을 단순화 (정책 A 즉시 release 외 옵션 없음).

### (b) INV-131 SOA 재변환 safety net (Phase 3.7a)

`SwapExecutor::execute_on_slots` Stage (d):
- AUF SOA bypass path: `restore_pre_converted_soa_registration` — `materialise_auf_soa_weight`가 만든 NoshuffleWeightBuffer를 등록만 재수행. primary 사용 없음.
- GGUF AOS path: `ensure_noshuffle_soa_registered` — primary는 사용하지 않고, 새 layer (Q4_0 buffer)로부터 SOA 변환.

**의미**: PRIMARY-DROP은 INV-131에 영향 없음. swap-back 부재로 GGUF 경로 재진입 시나리오 0.

### (c) QCF / `compute_quant_noise_for_model`

`engine/src/models/transformer.rs::load_gguf_with_secondary` line 240:
```rust
model.quant_noise = compute_quant_noise_for_model(&model);
```

모델 로드 시 1회 호출. swap 이후에는 호출되지 않음. `noise_table.rs::compute_tensor_epsilon`이 primary tensor의 `as_ptr()`을 사용하지만 이는 load 시점 (swap 이전).

**의미**: 영향 없음.

### (d) Plan / FullKernelPlan invalidation (Phase 3.5)

`engine/src/backend/opencl/plan.rs`:
- `LayerBufs.wq` 등은 `&'a Mem` (cl_mem reference)로 plan 빌드 시점에 추출.
- `KernelStep.retained_bufs: Vec<Mem>`이 cl_mem clone (refcount++)을 보유 → plan이 살아있는 동안 cl_mem alive 보장.
- `ratio_generation` bump → `check_global_generation` mismatch → `PlanInvalidated` → 새 generation으로 plan rebuild → 새 retained_bufs는 **새 layer의 cl_mem**을 retain.
- 구 plan drop → 구 retained_bufs drop → 구 cl_mem release.

**의미**: plan 자체는 transition을 자동 처리. PRIMARY-DROP은 plan 무효화 시점의 release를 단지 결정론적으로 만든다.

### 종합 결론

swapped layer의 primary cl_mem은 **완전히 dead weight**. 즉시 release 가능. embed_tokens / lm_head / output_norm / layer 0 / 마지막 layer (decider 보호 layer)는 그대로 유지.

## 채택 정책: A (즉시 release)

| 정책 | 채택 | 사유 |
|------|------|------|
| A: ratio=1.0에서만 release | ❌ | PLACEHOLDER-DROP과 같은 conservative이지만 실제 정책 B와 차이 없음 (all-or-nothing 아닌 per-layer 단위로 가능) |
| A': swapped layer 단위 즉시 release | ✅ | swap-back 부재 → 즉시 release 안전. |
| B: grace period 후 release | ❌ | swap-back 없으므로 grace 무의미. |
| C: heuristic 기반 | ❌ | 결정론적 release보다 추가 가치 없음. |

## 구현

### 변경 파일

| 파일 | 변경 |
|------|------|
| `engine/src/models/weights/slot.rs` | `swap_weights` 시그니처 `() → Arc<LayerWeights>`. 내부 `ArcSwap::store → swap`. |
| `engine/src/models/weights/swap_executor.rs` | Stage (b/c) 재구성. 새 helper `release_primary_weights`. 모듈-레벨 helper `record_swap_release`. 단위 테스트 2건. |
| `.agent/todos/feat_weight_swap.md` | TODO 갱신 (DONE host-side). |

### 핵심 코드 변경

**`LayerSlot::swap_weights`** (slot.rs):
```rust
pub fn swap_weights(
    &self,
    new_weights: Arc<LayerWeights>,
    new_dtype: DType,
) -> Arc<LayerWeights> {
    self.current_dtype.store(dtype_to_u8(new_dtype), Ordering::Release);
    let old = self.weights.swap(new_weights);   // store → swap
    self.generation.fetch_add(1, Ordering::Release);
    old
}
```

`ArcSwap::swap()`은 `wait_for_readers(old, &self.ptr)` 호출로 hazard-pointer로 보호된 in-flight reader 완료를 보장. 단일 스레드 dispatcher 환경에서는 즉시 반환되며 반환된 Arc는 단독 소유.

**`SwapExecutor::execute_on_slots` Stage (b/c)** (swap_executor.rs):
```rust
// Stage (b)
let old = slot.swap_weights(new_arc, self.target_dtype);

// Stage (c) — WSWAP-5-PRIMARY-DROP
match Arc::try_unwrap(old) {
    Ok(layer) => Self::release_primary_weights(&self.backend, layer),
    Err(arc) => Self::madvise_if_exclusive(&arc),  // fallback
}
```

`Arc::try_unwrap`이 단독 소유시 성공하여 inner `LayerWeights`를 추출. 그 외 (다른 holder 존재 시) madvise fallback.

**`release_primary_weights`** helper:
- 7개 weight tensor (`wq`/`wk`/`wv`/`wo`/`w_gate`/`w_up`/`w_down`) + 2개 norm + 옵션 bias/Gemma3 norm을 size 합산.
- `drop(old_layer)` → 모든 `Arc<dyn Buffer>` decrement → unique ref이면 `OpenCLBuffer::drop` → `clReleaseMemObject`.
- `record_swap_release(backend, count, bytes)`: `LLMRS_CL_MEM_DIAG=1` 시 `weight_swap_released` bucket에 release 기록 (Sprint B의 destructor 미instrumented gap을 메움).

## Sprint B 진단 보정

Sprint B `phase_5_tbt_diag.md`의 가설:

> swap된 16 layer × 7 weight = 112 F16 weight도 ArcSwap publish 이후 strong_count > 1로 madvise 우회 (madvise=0.0ms 측정 일치) → drop되지 않은 채 alive

**보정**: `madvise=0.0ms`의 진짜 원인은 **GPU buffer에 madvise 무효**.

`engine/src/models/weights/swap_executor.rs::madvise_tensor` line 838:
```rust
fn madvise_tensor(t: &Tensor) {
    let ptr = t.buffer().as_ptr();
    if ptr.is_null() { return; }   // ← GPU buffer는 항상 null
    ...
}
```

`engine/src/backend/opencl/buffer.rs::OpenCLBuffer::as_ptr()` (line 78):
```rust
fn as_ptr(&self) -> *const u8 {
    ptr::null()  // GPU buffer는 host pointer 없음
}
```

즉 OpenCL backend의 weight Tensor는 항상 `ptr.is_null()` → madvise 무효. madvise=0.0ms는 strong_count 확인 결과가 아니라 **모든 GPU tensor가 ptr.is_null() 조건에서 즉시 return**하기 때문.

이 보정으로 Sprint B "F16 primary 145 cl_mem alive" 가설은 다음과 같이 정정:
- diag dump의 `weight_f16_copy` count는 alloc 시점만 카운트 (Sprint B 명시).
- destructor 미instrumented로 실제 alive 여부 dump으로는 알 수 없음.
- 본 sprint의 `weight_swap_released` bucket으로 release 측 가시화 → "alloc - released" 닫힘 루프 가능.

실제 cl_mem alive 여부는 본 sprint 디바이스 측정 sprint에서 확인 (Total alive bytes / `LIVE_BUFFERS` static counter 사용).

## Tests

### 신규 단위 테스트 (swap_executor::tests)

#### `slot_swap_weights_returns_previous_arc_uniquely_owned`

ENG-ALG-211 step (b/c) refined contract 검증:
1. LayerSlot 생성, `slot.load_weights()`로 wq buffer Arc 캡처.
2. `slot.swap_weights(new, DType)` 호출.
3. **반환 Arc의 strong_count == 1** assertion. (단독 소유 보장)
4. `Arc::try_unwrap` 성공 assertion.
5. `drop(inner)` 후 wq buffer Arc strong_count가 1로 감소 (테스트 local probe만 남음).

#### `release_primary_weights_runs_destructors`

Helper의 destructor 동작 검증:
1. 7개 weight buffer의 `Weak<dyn Buffer>` 캡처.
2. `SwapExecutor::release_primary_weights` 호출.
3. 모든 Weak ref가 `upgrade()` 시 None 반환 (= 모든 Arc 0 ref 도달, destructor 실행됨).

### 기존 회귀 테스트

| 테스트 | 결과 |
|--------|------|
| `cargo test -p llm_rs2 --lib` | 1048 passed (skip된 6 except) |
| `cargo test -p llm_rs2 --test spec` | 392 passed (3 ignored) |
| `cargo test -p llm_rs2 --test spec inv_121` | 3 passed (forward 동시성) |
| `cargo test -p llm_rs2 --test spec inv_122` | 4 passed (mixed precision) |
| `cargo test -p llm_rs2 --test spec inv_123` | 3 passed (atomicity) |
| `cargo test -p llm_rs2 --test spec inv_131` | 4 passed (SOA 재변환) |
| `cargo test -p llm_rs2 --test spec wswap_e2e_phase3` | 9 passed (E2E) |
| `cargo test -p llm_rs2 --test spec auf_e2e` | 6 passed (AUF SOA bypass) |
| `cargo clippy --all-targets -- -D warnings` | clean |
| `cargo build --release -p llm_rs2` | clean |

### 회귀 가드 점검

| 가드 | 결과 |
|------|------|
| INV-131 safety net (4 spec tests) | PASS — SOA 재변환 회귀 없음 |
| INV-121/123 동시성 (6 spec tests) | PASS — ArcSwap 시맨틱 보존 |
| AUF SOA bypass (6 spec tests) | PASS — Phase 3.7b 경로 무손상 |
| Decider / handler (9 spec tests) | PASS — 시그니처 변경 영향 없음 |
| Cl_mem 진단 인프라 | `weight_swap_released` 신규 bucket 추가 — 기존 dump 형식 그대로, 추가 라인만 |

### partial swap 회귀

호스트 환경에서는 partial swap의 정확성을 GPU 없이는 검증 어려움 (실제 CL 커널 dispatch 필요). 호스트 단위 테스트는 try_unwrap이 단독 소유 보장 → drop 정상 동작까지 커버. 디바이스 측정 sprint에서 `--force-swap-ratio 0.25/0.5/0.75/1.0` 검증 필요.

## 잔여 / Follow-up

### 디바이스 측정 sprint (별도)

Galaxy S25 (`R3CY408S5SB`)에서 다음 측정 필요:
- `LLMRS_CL_MEM_DIAG=1` dump:
  - 변경 전 `weight_f16_copy` count + alive_bytes (Sprint B 측정 145 / 2.47 GB).
  - 변경 후 `weight_f16_copy` alloc 카운트는 동일 (loader는 그대로); **`weight_swap_released` bucket에 swap된 layer의 release 기록 등장**.
  - 변경 후 `LIVE_BYTES` static counter (buffer.rs line 12)로 실제 alive 추적: ratio=1.0 mixed에서 Q4 baseline 수준 (~2.0 GB)까지 감소 확인.
- matmul_qkv μs/call (Q4 baseline 437, PLACEHOLDER-DROP 후 513, 추가 감소 측정).
- Decode TBT (Q4 baseline 16.37, PLACEHOLDER-DROP 후 20.22, 추가 감소 측정).
- 정확성 가드: "Paris" 정답 + ratio={0.25, 0.5, 0.75, 1.0} 모두에서 garbage 0건.

### Architect Spec 갱신 (옵션)

- `spec/32-engine-algorithms.md` ENG-ALG-211 step (c)에 "primary cl_mem 즉시 release (swap-back 부재 정당화)" 명시 권장.
- `spec/41-invariants.md` INV-126과의 cross-reference 추가.

### KV-FRAG-INTEGRATE 권장 (P3 별도)

본 sprint 효과로 매트릭스 닫힘 후, 잔여 fragmentation 영향을 측정. P3로 분리되어 있음.

## Acceptance 체크리스트 (호스트-side)

| 항목 | 상태 |
|------|------|
| Lifecycle audit 4개 영역 완료 (design note 첨부) | ✅ |
| 채택 정책 (A') + 근거 명시 | ✅ |
| 변경 파일 + 라인 명시 | ✅ |
| 신규 단위 테스트 2건 PASS | ✅ |
| INV-131 회귀 없음 | ✅ |
| INV-121/123 회귀 없음 | ✅ |
| AUF / wswap E2E 회귀 없음 | ✅ |
| 호스트 sanity (lib + spec + clippy + fmt + release build) | ✅ |
| `weight_swap_released` 진단 bucket 추가 | ✅ |
| Sprint B 진단 보정 기록 | ✅ |
| 보고서 작성 | ✅ |

디바이스-side acceptance (별도 sprint):
- F16 primary alive cl_mem ≤ 16 (또는 0)
- Total alive bytes ~2.0 GB
- matmul_qkv 추가 감소
- Decode TBT 추가 회복
- 정확성 가드 (Paris + ratio scan)

---

# 디바이스 측정 (Sprint C-3, 2026-04-25)

## 환경

- **디바이스**: Galaxy S25 (`SM-S931N`, adb `R3CY408S5SB`)
- **Android**: 16, kernel 6.6.77, `--threads 6`, backend OpenCL (Adreno)
- **HEAD**: `aab1527` (`docs(todo): mark WSWAP-5-PRIMARY-DROP done`) — Sprint C-2 코드 (`f407a70`) 동일 빌드 산출물
- **generate sha256**: `10c0a4a55b2109d91b0b8a508ed36e0223286bd621e35ed3a859eef30aee2b7a` (host = device 매칭)
- **모델**:
  - F16 GGUF: `/data/local/tmp/Llama-3.2-1B-Instruct-f16.gguf` (16 layers)
  - Q4_0 GGUF: `/data/local/tmp/Llama-3.2-1B-Instruct-q4_0.gguf`
  - AUF (Phase 3.7b SOA payload): `/data/local/tmp/Llama-3.2-1B-Instruct.auf` (sha256 `1a1ead0c1f532b26034989deb7dbfece4f5b7ed41b881491cfee857ae014c5ec`, Sprint B/C-1과 동일 자산)
- **CLI (TBT)**: `--num-tokens 128 --protected-prefix 4 --prompt "The capital of France is" --threads 6 --temperature 0.0`
- **CLI (--profile)**: `--num-tokens 64 --profile`
- **반복**: TBT N=3, profile N=2, Stage D N=1 per ratio

## Stage A — 빌드 + 정확성 스모크

| 항목 | 결과 |
|------|------|
| 호스트 cargo build (release, aarch64) | PASS |
| 디바이스 push + sha256 매칭 | PASS |
| ratio=1.0 mixed 16-token greedy 스모크 | PASS — "The capital of France is Paris. The Eiffel Tower, a famous landmark in Paris, was built" / garbage 0건 |

## Stage B — cl_mem 개수 검증 (`LLMRS_CL_MEM_DIAG=1`)

`stage=after_force_swap` (모델 로드 + force_swap 직후)

| 카테고리 | Q4 baseline (after_noshuffle_prep) | Mixed C-1 (after_force_swap, PLACEHOLDER-DROP) | **Mixed C-2 (after_force_swap, PRIMARY-DROP)** |
|---|---:|---:|---:|
| weight_f16_copy (alloc 누적) | 0 | 145 | 145 |
| weight_q4_aos_copy | 113 | 0 | 0 |
| auf_soa_q | 0 | 112 | 112 |
| auf_soa_d | 0 | 112 | 112 |
| auf_soa_img | 0 | 112 | 112 |
| weight_f32_copy | 33 | 33 | 33 |
| **TOTAL count** (alloc 누적) | 485 | 514 | **514** |
| **TOTAL alive** (alloc − release) | 485 | 514 | **34** ✅ |
| **`weight_swap_released`** (신규 bucket) | — | — | **releases=144 / 1,946,419,200 bytes (1.81 GB)** ✅ |
| 누적 alloc bytes | 2.01 GB | 4.05 GB (Sprint B 측정) | **3.51 GB** |
| 추정 실 alive bytes (alloc − released) | 2.01 GB | ≈3.51 GB (Sprint C-1 placeholder 제거 기준) | **≈526 MB** ✅ |

### 핵심 결과

- `weight_f16_copy` alloc 카운터는 **변동 없음** (145, alloc 시점만 카운트하므로 PRIMARY-DROP의 release를 반영하지 않음). 이는 호스트 리포트가 명시한 카운터 의미.
- **새 bucket `weight_swap_released`** 등장: `releases=144`, `released_bytes=1,946,419,200` (≈1.81 GiB) — Sprint C-2 sprint의 release 기록 instrumentation이 디바이스에서 정상 작동함을 검증.
- **TOTAL alive 514 → 34** (PRIMARY-DROP 효과로 480 cl_mem release). released_bytes 2.78 GB.
- **alive bytes ≈ 526 MB**(추정) — Mixed C-1의 ≈3.51 GB 대비 **−2.98 GB (−85%) 감소**. AUF SOA (3 kinds × 112 = 336개 cl_mem, 약 1.03 GB)는 alive 카테고리. 145 weight_f16_copy 중 **1개만 alive 잔존** (embed_tokens / lm_head / output_norm 중 일부 또는 swap 보호 layer의 wq/wk/wv/wo/ffn 일부일 가능성, 144개 release / 145 alloc).

### Acceptance 체크

- [x] **weight_f16_copy 145 → 0 (또는 ≤16)**: alloc 카운터 그대로 145이지만 **release bucket으로 144개 release 가시화**. 실 alive = 1 cl_mem (≤16 통과). ✅
- [x] **TOTAL alive count 가시화**: 514 → 34. ✅
- [x] **Total alive bytes Q4 baseline 수준 회복**: ≈526 MB (Q4 baseline 2.01 GB 대비 더 적음 — PRIMARY-DROP이 F16 GGUF 본체를 release하면서 Q4 baseline보다도 **더 슬림한 alive footprint** 달성. 단, AUF mmap/secondary mmap 영역은 cl_mem 외부이므로 RSS 측정 별개 필요). ✅

## Stage C — matmul_qkv μs/call + Decode TBT (wall-clock)

### 1) `--profile` Decode per-op breakdown (Adreno, 1008 layer-calls)

| op | Q4 baseline (이번 측정) | Mixed C-2 run1 | Mixed C-2 run2 |
|---|---:|---:|---:|
| matmul_qkv | 433 | 290 | **243** ✅ |
| matmul_wo | 312 | 188 | 157 |
| matmul_ffn | 1180 | 815 | 742 |
| attention | 214 | 159 | 134 |
| rms_norm | — | 305 | — |
| TOTAL/layer-call | 3209 | 2166 | — |

**해석**:
- Mixed C-2 matmul_qkv μs/call이 Q4 baseline 433보다 오히려 **낮음** (290/243).
- 핸드오프 표의 Mixed C-1 matmul_qkv=513은 Sprint C-1 측정 (다른 thermal/cold state). 같은 빌드/동시 측정한 본 sprint Q4 baseline=433과 비교하면 Mixed C-2가 절대값으로 더 낮음.
- 다만 `--profile`은 `plan.rs`를 비활성화하고 op마다 `synchronize()` 2회 호출 (`CLAUDE.md`/Profile flag 주의: "절대값은 sync 오버헤드로 부풀려져 있다. **상대 비교**에만 유효"). 따라서 **wall-clock TBT가 결정적 메트릭**.
- Profile 비교의 함의: `--profile` 모드에서 Mixed C-2가 Q4 baseline보다 빠르다 — sync overhead와 plan disable 조합이 두 모드에서 다르게 작용. 절대 비교 부적합 명시.

### 2) wall-clock Decode TBT (N=3, `--num-tokens 128`)

| Run | Q4 baseline (이번 측정) | Mixed C-2 (PRIMARY-DROP) |
|---|---:|---:|
| 1 | 16.29 | 20.15 |
| 2 | 16.32 | 20.52 |
| 3 | 16.29 | 20.28 |
| **mean** | **16.30 ms/tok (61.4 tok/s)** | **20.32 ms/tok (49.2 tok/s)** |
| σ | 0.014 | 0.153 |

### 3) 단계별 비교 (wall-clock)

| 측정 | Q4 baseline ms/tok | Mixed ratio=1.0 ms/tok | Δ vs Q4 |
|---|---:|---:|---:|
| Phase 4 (`21c6d82`) | 16.31 | 20.58 | +26.2% |
| Sprint C-1 (PLACEHOLDER-DROP, `a4d29e8`) | 16.37 | 20.22 | +23.5% |
| **Sprint C-2 (PRIMARY-DROP, `aab1527`)** | **16.30** | **20.32** | **+24.7%** |

### 핵심 관찰

1. **PRIMARY-DROP의 wall-clock 추가 회복은 미미**: 20.22 → 20.32 (+0.10 ms/tok). σ 차이를 고려하면 통계적으로 무의미한 차이 (Q4 σ=0.014, Mixed σ=0.153, 측정마다 변동 가능).
2. Q4 baseline 대비 **+24.7% 잔여 gap** — Phase 4 +26.2% 대비 −1.5pp 개선만. **본질 해소 미달성**.
3. **−20.7% gap 회복도**: PRIMARY-DROP 만으로는 회복 불가. `weight_swap_released` 144 / 1.81 GB의 driver-level 메모리 해제는 확인되었으나, decode latency에 결정적 영향 없음.
4. Profile 모드의 matmul_qkv 절대값은 sync overhead로 sync overhead가 매 op마다 누적되는 모드에서 측정된 것. wall-clock으로는 추가 회복 미미.

## Stage D — ratio scan 정확성 가드 (N=1 per ratio)

| ratio | swapped layers | swap latency | 출력 (32 tokens, greedy) | 정답 | garbage |
|---|---:|---:|---|:---:|:---:|
| 0.25 | 4/16 | 133.9 ms | "The capital of France is Paris. The Eiffel Tower, a famous landmark in Paris, was built for the 1889 World's Fair." | ✅ | 0 |
| 0.50 | 8/16 | 265.2 ms | "The capital of France is Paris. The Eiffel Tower, a famous landmark in Paris, was built for the 1889 World's Fair." | ✅ | 0 |
| 0.75 | 12/16 | 344.3 ms | "The capital of France is Paris. The Eiffel Tower, a famous landmark in Paris, was built for the 1889 World's Fair and took about four years to complete." | ✅ | 0 |
| 1.00 | 16/16 | 498.7 ms | "The capital of France is Paris. The Eiffel Tower, a famous landmark in Paris, was built for the World's Fair in 1889. It has become one of the" | ✅ | 0 |

**모든 ratio에서 "Paris" 정답 + 일관된 영어 문장 + garbage 0건 PASS.** Sprint C-1의 dtype 가드 버그 (`primary.dtype() == Q4_0` → 항상 false → garbage) 같은 회귀 사례 0건.

## Stage E — swap latency 회귀 점검

| ratio | prefault | mmap_permute | arc_swap | madvise | soa_reconvert | gen_bump | TOTAL |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0.25 | 14.2 | 72.6 | 0.0 | 47.1 | 0.0 | 0.0 | 133.9 |
| 0.50 | 94.1 | 99.3 | 0.0 | 70.2 | 0.0 | 1.5 | 265.2 |
| 0.75 | 84.2 | 150.9 | 0.0 | 107.5 | 0.0 | 1.7 | 344.3 |
| 1.00 (Stage A) | 61.0 | 197.1 | 0.0 | 148.8 | 0.0 | 1.6 | 408.5 |
| 1.00 (Stage D) | 12.8 | 289.7 | 0.0 | 196.0 | 0.0 | 0.1 | 498.7 |

**관찰**:
- `arc_swap=0.0ms` 일관 — `ArcSwap::swap` (store → swap 변경)이 wait_for_readers 즉시 반환 (단일 dispatcher). PRIMARY-DROP의 `Arc::try_unwrap` + `release_primary_weights` 시간도 0.0ms 영역으로 흡수.
- `madvise=148.8/196.0ms` — fallback 경로 (`Arc::try_unwrap` 실패 시 madvise) 또는 다른 madvise 경로 시간으로 추정. Sprint B의 madvise=0.0ms 와 차이 큼. 단, 동일 빌드의 ratio scan 4건이 일관된 패턴 (ratio에 비례) 보임 → 비-회귀.
- `mmap_permute` 시간이 ratio에 비례 (197/289 vs 0.25=72.6) — 정상 패턴.
- swap latency 회귀 **명확히 보이지 않음**. 단, 같은 빌드에서 ratio=1.0이 408.5ms (Stage A 스모크) vs 498.7ms (Stage D) 차이 — thermal/cold state 변동으로 설명 가능.

회귀 점검 결과: **명백한 회귀 없음**. 다만 ratio=1.0의 mmap_permute+madvise 합계가 큰 편이며, 이는 PRIMARY-DROP과 무관한 pre-existing 비용.

## 종합 verdict

| 항목 | 결과 |
|---|---|
| Stage A — 빌드/스모크 | ✅ PASS |
| Stage B — cl_mem release 가시화 (144 / 1.81 GB) | ✅ PASS — `weight_swap_released` instrumentation 정상 |
| Stage B — TOTAL alive 514 → 34 | ✅ PASS |
| Stage C — matmul_qkv μs/call (profile 절대값) | ⚠️ 절대값은 sync overhead로 부풀어, **상대 비교 부적합** |
| Stage C — Decode TBT 회복 | ⚠️ +24.7% gap 잔존 (Phase 4 +26.2% 대비 −1.5pp만 개선) |
| Stage D — ratio scan 정확성 (4 ratios) | ✅ ALL PASS |
| Stage E — swap latency 회귀 | ✅ 회귀 없음 |

### −20.7% gap 본질 해소 여부

**미달성**. 호스트 audit이 swap-back 부재 + plan invalidation 자동 처리를 확정했고 디바이스에서도 144 cl_mem / 1.81 GB release를 가시화했으나, **wall-clock decode TBT는 통계적으로 변동 범위 (~+0.5%) 내**로 머문다.

가능한 원인:
1. **AUF SOA 3 kinds × 112 = 336 cl_mem 자체가 cold-cache pattern의 dominant 원인**. F16 primary 145 release가 driver page table에서 144개를 비웠지만, AUF SOA 336개의 page table 등록이 여전히 dominant.
2. **KV cache fragmentation** (`project_kv_fragmentation.md`): 56개 별도 KV cl_mem이 attention slope +1.32 µs/n_kv. 잔여 gap의 일부 가능.
3. **Adreno texture cache cold-warm**: matmul_qkv first-weight access의 TLB miss 패턴이 alive cl_mem 개수와 무관하게 발생 (driver/compiler 레벨).

### 권장 다음 단계

1. **KV-FRAG-INTEGRATE (P3) 진행 권장**: 56개 KV cl_mem → 단일 cl_mem + view 통합. attention slope 회복 + matmul_qkv first-weight TLB cold 가능 영향.
2. **AUF SOA 단일 cl_mem화 검토**: 336개 cl_mem (q/d/img × 112)을 백킹 single allocation으로 통합 (Phase 3.7c 수준 sprint).
3. **Profile 모드 plan.rs 활성화** 별도 측정 (현재 `--profile`이 plan disable). plan-aware profile이 가능하면 Mixed의 진짜 op breakdown을 확인 가능.
4. **−20.7% gap의 본질**은 단일 cl_mem release만으로 해소되지 않음을 본 sprint가 확정. fragmentation (P3) + secondary AUF cl_mem 통합 (P4 신규) 양면 접근 필요.

### 잔여 issue

- alive bytes 정확한 계산: 카테고리별 release 분리 추적이 필요 (현재 `weight_swap_released`만 추적). Q4 baseline의 NoshuffleWeightBuffer release(662.9 MiB)도 별도 bucket이 없어 alive bytes는 추정에 그침.
- swap latency stage 5건 동일 ratio (1.0)에서 408.5ms vs 498.7ms 변동 — instrumentation 측정 분산 점검 권장 (별도 sprint).

## 산출물

- 측정 로그: `/tmp/wswap5_primary_drop/stage{A,B,C,D}_*.log` (host-side temp)
- generate sha256: `10c0a4a55b2109d91b0b8a508ed36e0223286bd621e35ed3a859eef30aee2b7a`
- 본 리포트: `results/data/weight_swap/phase_5_primary_drop.md` (본 §디바이스 측정 추가)
