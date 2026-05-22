# Handoff: B-5b Phase 0/1 종결 → Phase 2/3 다음 세션

**작성**: 2026-05-22
**HEAD**: `7be03c5b build(layer_lint): B-5b-1c — DATA_CONSUMER_PATTERNS allowlist + baseline regen`
**브랜치/Worktree**: `worktree-b5_trait_extension` (Phase 4 master FF 예정)
**다음 세션 진입 문장**: `"B-5b Phase 2 진행 (Backend trait default impl 4 method + SecondaryStore trait)"`
**선행 진입점**: [[handoff_b5a_partition_option_delta_2026_05_22]]

---

## TL;DR

INV-LAYER-001 잔여 27건 해소 sprint B-5b의 Phase 0 (architect 결정 라운드) + Phase 1 (R8 위치 정정 3건 + (b') DATA_CONSUMER_PATTERNS allowlist) 종결. 결정 핵심: §13.8-K 신설 회피 + spec 본문 무수정 + DATA_CONSUMER_PATTERNS 자동 분류로 weight consumer dataflow를 baseline에서 제외. baseline **216 → 206 (-10)**. **멈춘 이유: 사용자 결정 — Phase 2(R1 Backend trait default impl 4 method + R2 SecondaryStore trait + gpu_yield 흡수)는 S25 microbench 게이트 필요한 큰 작업이라 다음 세션으로**.

---

## 진행 상태

### 본 sprint commits (4건)

| HEAD | scope | 변경 |
|---|---|---|
| `1cdba86b` | docs(arch) | Phase 0 architect 결정 — R1=(a)/R2=(b)/R4=(b')/R8=3건. 산출물 3건: `arch/sprint_b5b_phase0_decision.md` 신규, `ARCHITECTURE.md` §13.5 V-?? plan 행 7건, `spec/41-invariants.md` INV-LAYER-001 비고 data consumer 카테고리 1단락 |
| `0091aeed` | refactor(layer) | B-5b-1a — `OpenClEventGpuMeter` struct만 `backend/opencl/gpu_self_meter.rs`로 분리. `GpuSelfMeter` trait + `NoOpGpuMeter`는 resilience 유지. 신규 1건 false positive (layer_lint 분류 한계) |
| `45bfd16f` | refactor(layer) | B-5b-1b — `KVCacheOps` trait L2 격상 (`engine/src/kv_cache_ops.rs` 신규). `KVLayout` enum + `KiviRawBuffers<'a>` struct도 method signature 의존이라 함께 격상. `pressure/kv_cache.rs`에 BC re-export |
| `7be03c5b` | build(layer_lint) | B-5b-1c — `scripts/layer_lint.py`에 `DATA_CONSUMER_PATTERNS` 정규식 allowlist + baseline regen. 9건 자동 제외 (qnn_oppkg/opencl의 LayerSlot/TransformerLayer/SecondaryMmap/RpcmemLayerRegion/KVCache struct) |

### 게이트 결과

| 게이트 | 결과 |
|---|---|
| `cargo build -p llm_rs2` | PASS |
| `cargo clippy --lib --no-deps -- -D warnings` | clean 0 |
| `cargo test --test spec inv_layer` | **8 PASS / 0 FAIL** |
| `python3 scripts/test_layer_lint.py` | **9 PASS / 0 FAIL** |
| `cargo test --lib -p llm_rs2` | 회귀 0 (B-5a 1207 PASS 기준, GPU device-required 테스트 제외 1200+ PASS) |
| **baseline JSON** | 216 → **206 (-10)** |

### baseline 변동 누적 (B-1~B-5b Phase 1)

| 단계 | 변동 | baseline |
|---|---|---|
| 진입 (Step 5b 종결 후) | — | 296 |
| B-1 EvictMethod | -9 | 287 |
| B-2 profile inversion | -33 | 254 |
| B-4 eval L4 격상 | -34 | 220 |
| B-5a tensor_partition | -4 | 216 |
| **B-5b Phase 0/1** | **-10** | **206** |
| 누적 | **-90 (-30.4%)** | |

### Phase 0 결정 (재인용)

| R | 안 | 핵심 |
|---|---|---|
| R1 | (a) | Backend trait + default impl 4 method (cpu_companion / cpu_kernels / as_opencl_secondary / yield_after_layer). ISP 누적은 backlog |
| R2 | (b) | SecondaryStore trait만 추출 (SecondaryMmap + RpcmemLayerRegion). TransformerLayer/LayerSlot은 R4로 grandfather |
| R4 | **(b')** | INV-LAYER-001 본문 무수정 + 비고에 data consumer 카테고리 + layer_lint DATA_CONSUMER_PATTERNS allowlist. **§13.8-K 신설 회피** (정책 6개 유지) |
| R8 | 3건 위치 정정 | gpu_self_meter struct (✅ 완료) / KVCacheOps trait L2 (✅ 완료) / gpu_yield (Phase 2로 미룸 — transformer.rs:1841 호출 흡수 시점에 함께 처리) |

상세: `arch/sprint_b5b_phase0_decision.md` (commit `1cdba86b`)

### Phase 1 실측 차이

Phase 0 plan(-13) vs 실제(-10) 차이 -3:
- DATA_CONSUMER_PATTERNS 9건 (예상 11에서 -2건) — `rpcmem_secondary::RpcmemLayerRegion`이 일부 위치에서 다른 규칙으로 분류
- **신규 위반 +1**: `backend/opencl/gpu_self_meter.rs:10 → crate::resilience::gpu_self_meter::GpuSelfMeter` (cross-cutting *trait* import — INV-LAYER-001 본문상 허용 패턴이지만 layer_lint가 자동 분류로 위반 잡음. false positive)

---

## 다음 작업 (Phase 2 → Phase 3 → Phase 4)

### Phase 2 — R1 Backend trait default impl 4 method + R2 SecondaryStore trait (1.5~2일)

**Backend trait 확장** (`engine/src/backend.rs`):
```rust
pub trait Backend: Send + Sync {
    // ... 기존 57 method ...

    /// CPU companion. CpuBackend = self, GPU = injected.
    fn cpu_companion(&self) -> &dyn Backend { self }

    /// CPU kernel function pointer set. CpuBackend variants만 Some.
    fn cpu_kernels(&self) -> Option<&'static CpuKernelSet> { None }

    /// LISWAP-6 OpenCL secondary capability. OpenCLBackend만 Some.
    #[cfg(feature = "opencl")]
    fn as_opencl_secondary(&self) -> Option<&dyn crate::secondary::OpenClSecondary> { None }

    /// Intra-token GPU yield hook. Default no-op.
    fn yield_after_layer(&self, _layer: usize, _is_decode: bool) {}
}
```

**신규 L2 자산**:
- `engine/src/cpu_kernels.rs` — `pub struct CpuKernelSet { fn pointers... }`
- `engine/src/secondary.rs` — `pub trait OpenClSecondary` + `pub trait SecondaryStore`

**GPU backend init 갱신**:
- `cuda_embedded/mod.rs`, `cuda_pc/mod.rs`, `opencl/mod.rs` — struct field `cpu_companion: Arc<dyn Backend>` 추가, init 시 `Arc::new(CpuBackend::new())` 주입
- `OpenCLBackend::yield_after_layer` override (resilience::gpu_yield 로직 흡수)
- `OpenCLBackend::as_opencl_secondary` override (LISWAP-6 rpcmem alias)

**호출지 치환**:
- C5 cpu_fallback 6건 (cuda_embedded 3 + cuda_pc 2 + opencl 1) → `backend.cpu_companion().matmul(...)`
- C7 NEON 4건 (opencl/plan.rs:696/717/1638/1705) → `backend.cpu_kernels().expect("...").fused_matmul_f16(...)`
- C6 with_opencl 2건 (qnn_oppkg/mod.rs:134,140) → `backend.as_opencl_secondary()` 경유
- C8b gpu_yield (plan.rs:1834 + models/transformer.rs:1841) → `backend.yield_after_layer(i, true)`. `resilience/gpu_yield.rs` 모듈은 deprecated/제거

**SecondaryStore trait 추출**:
- `engine/src/secondary.rs::SecondaryStore` — 공통 method `as_bytes() -> &[u8]` + `len() -> usize`
- `SecondaryMmap`/`RpcmemLayerRegion` impl
- 사용처 (R4 (b')로 이미 data consumer 자동 분류된 위치 외) 호출 패턴 전수조사 필요

**검증 게이트**:
- clippy 0 + spec test 8 PASS + cargo test --lib 회귀 0
- **S25 microbench 필수** (Qwen 2.5 1.5B Q4_0 TBT ±3% — 함수 포인터 호출 inline 불가로 -1~3% 회귀 가능). 회귀 시 fallback 결정
- baseline 206 → 192 (-14 예상)

**위임 후보**: senior-implementer (NEON hot path 영향 + GPU backend struct 변경 + microbench)

### Phase 3 — hybrid_attention 2건 §J 확장 + 잔여 cleanup (0.5~1일)

- `backend/opencl/plan.rs:1546,1552` (`compute_kv_split`, `current`) → §13.8-J zone marker
- ARCHITECTURE.md §13.8-J 본문에 "L1 build-time 정책 query 한정, runtime hot path 호출 금지" 보강 1줄
- baseline 192 → 189 (-3)

### Phase 4 — handoff + 종결 commits + master FF (30분)

- handoff doc 작성
- master FF + push
- notify-send

### 예상 최종

baseline **206 → 189 (-17 추가, 본 sprint 총 -27)**. 누적 B sprint -107 (-36.1%).

---

## Landmines / 미해결

### 본 sprint 중 발견된 문제

1. **layer_lint 자동 분류의 trait vs struct 구분 한계** — `backend/opencl/gpu_self_meter.rs:10`의 `use crate::resilience::gpu_self_meter::GpuSelfMeter;`는 cross-cutting *trait* import로 INV-LAYER-001 본문상 *허용* 패턴이지만 layer_lint가 위반으로 잡음. false positive 1건이 baseline에 등재되어 잔존. 향후 layer_lint 개선 또는 RESILIENCE_TRAIT_ALLOW allowlist 도입 검토. 본 sprint 와중에 처리하지 않음.

2. **KVLayout/KiviRawBuffers 우발적 L2 격상** — Phase 1 작업 B에서 `KVCacheOps` trait의 method signature가 `KVLayout` enum + `KiviRawBuffers<'a>` struct를 노출하여 함께 L2로 격상됨. 이 두 type이 본질적으로 L2 자산인지 (도메인 어휘) vs L3 자산인지 (cache 내부 구현)는 향후 architect 라운드 후 재배치 가능성. 현재는 BC re-export로 호출지 영향 0.

3. **DATA_CONSUMER_PATTERNS 9건 vs 예상 11건** — `rpcmem_secondary::RpcmemLayerRegion` 일부 위치가 패턴에 매칭되지 않음. Phase 2 진입 전 패턴 보강 검토 가능 (정규식 `crate::models::weights::[a-z_]+::[A-Z]` 추가 등).

### 후속 cleanup backlog

1. **BC re-export 제거** — `engine/src/pressure/kv_cache.rs`의 `pub use crate::kv_cache_ops::{KVCacheOps, KVLayout, KiviRawBuffers};` (Phase 1-1b 도입). Phase 4 master FF 후 별도 sprint에서 제거 검토.
2. **`instrument.rs` TOP_LEVEL_L2 누락** — B-2 잔재, B-5a/B-5b 모두 미해소.
3. **Backend trait fat interface (57 → 61 method)** — R1=(a) 결정 시점에 ISP 누적 인정. 별도 sprint "Backend trait segregation"으로 분리.

### Phase 2 진입 전 확인 필수

1. **`SecondaryStore` trait method set 확정** — `grep -rn "SecondaryMmap\|RpcmemLayerRegion" engine/src/backend/`로 호출 패턴 전수조사. method set이 부족하면 downcast 회귀.
2. **S25 디바이스 준비** — Galaxy S25 (Adreno OpenCL backend), Qwen 2.5 1.5B Q4_0 GGUF. `/profile` 또는 `python scripts/run_device.py -d s25 generate`로 TBT 측정.
3. **gpu_yield 처리 통합** — `resilience/gpu_yield.rs` 모듈 자체는 backend/opencl로 이동하지 말고 *deprecated 후 제거*. trait method 흡수가 본질이라 mv는 불요.

### "이 길은 가지 마라"

- **Backend trait 4 method를 hot path inline 호출로 전환하지 말 것** — vtable indirection이 default impl 패턴의 비용 — 대신 cpu_kernels/cpu_companion은 cold path 한정으로 사용. C7 NEON은 hot path지만 함수 포인터라 indirection은 동일.
- **`yield_after_layer`을 hot path에서 매 op마다 호출하지 말 것** — 매 layer 끝 1회만. 더 fine-grained 호출은 plan-level batching 깨짐.
- **gpu_self_meter false positive를 무시하고 신규 false positive 누적 허용하지 말 것** — layer_lint 분류 한계는 향후 개선 대상이지 *허용 패턴*이 아님. Phase 2/3 작업 중 새로운 false positive 발견 시 보고 + 즉시 처리.

### microbench 게이트 — Phase 2 합격 기준

- S25 (Adreno OpenCL) Qwen 2.5 1.5B Q4_0 6T baseline vs 변경 후 TBT 차이 |Δ| ≤ 3%
- 측정 방법: `python scripts/run_device.py -d s25 generate --model-path ... --prompt "..." --max-new-tokens 64` × 5 runs avg
- Δ > 3% 회귀 시: C7 CpuKernelSet 함수 포인터 패턴 후퇴 + `crate::cpu_kernels::*` L2 격상으로 inline 가능한 형태로 재구성

---

## 참고 자료

- 메모리: [[layered-architecture-decision]] — 본 sprint의 원안
- 직전 handoff: `.agent/todos/handoff_b5a_partition_option_delta_2026_05_22.md`
- 정책 문서: `arch/sprint_b5b_phase0_decision.md` (Phase 0 결정 4건 상세)
- ARCHITECTURE.md §13.5: V-01/V-03/V-04/V-05/gpu_yield/KVCacheOps/CpuKernelSet/hybrid_attention plan 행 7건 (commit `1cdba86b`)
- 변경된 spec: `spec/41-invariants.md` INV-LAYER-001 비고 data consumer 카테고리
- 신규 L2: `engine/src/kv_cache_ops.rs` (Phase 1-1b 도입)
- 신규 backend type: `engine/src/backend/opencl/gpu_self_meter.rs` (Phase 1-1a 도입)
- DATA_CONSUMER_PATTERNS: `scripts/layer_lint.py:?` (Phase 1-1c 도입)

## 자기점검 결과

- [x] 진입 문장이 한 줄로 명확? `"B-5b Phase 2 진행 (Backend trait default impl 4 method + SecondaryStore trait)"`
- [x] "왜 멈췄는가"? 사용자 결정 — Phase 2는 S25 microbench 게이트 필요한 큰 작업, 다음 세션으로
- [x] 가장 큰 landmine 표면화? layer_lint trait vs struct 분류 한계 + KVLayout/KiviRawBuffers 우발 격상 + S25 microbench 회귀 시 fallback 결정 필요
- [x] 검증 게이트 수치/명령? clippy clean 0, spec test 8 PASS, baseline 216→206, S25 ±3%
- [x] 길이 적정? 본문 ~1100 토큰, 4 commits + Phase 분해표 + 4 landmine + microbench 게이트 명시
