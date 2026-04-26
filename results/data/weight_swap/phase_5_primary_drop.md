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
