# Handoff: INV-LAYER A + B sprint (S-1 → S-2 → S-3) 종결

**작성**: 2026-05-24
**HEAD**: `aec296dd refactor(layer): S-3 — Backend::alloc_alias_weight_buffer trait object signature`
**브랜치**: `worktree-b5_trait_extension` (master FF 대기)
**다음 세션 진입 문장**: "INV-LAYER 다음 sub-sprint 후보 리뷰" 또는 "INV-LAYER-003 122건 카테고리 정리 진입"

---

## TL;DR

INV-LAYER A (INV-LAYER-001) + B (INV-LAYER-002) 동시 정리 sprint 3단계 종결. S-1 (A4 chain 예외 + A3 NEON L2 격상 + B4 rollback) → S-2 (cooperative yield trait default body + cpu_singleton DI) → S-3 (alloc_alias_weight_buffer trait object signature). **baseline 176 → 163 (−13)**. 사용자 결정 옵션 B (큰 작업) 선택 → 사전 dry-run 게이트 통과 → trait 무 추가로 깔끔 해소. 멈춘 이유: sprint scope 완료, 다음 후보 선택 대기.

---

## 진행 상태

| Sprint | Status | Commit | Baseline Δ | 핵심 변경 |
|---|---|---|---|---|
| S-1.1 A4 cross-backend chain 예외 | ✅ DONE | `4c315417` | −2 | ARCHITECTURE.md §13.8-K 신설. layer_lint ALLOWED_BACKEND_CHAINS = {(qnn_oppkg, opencl)}. |
| S-1.2 A3 NEON L2 격상 (cascade 5 함수) | ✅ DONE | `4c315417` | −3 | `engine/src/quant/flash_neon.rs` 신규. opencl/plan + forward_gen import path 갱신. INV-001 2건 + INV-003 1건 동시 해소. |
| S-1.3 B4 host_ptr_pool_buffer 이동 | ❌ ROLLBACK | — | 0 | 파일 이동이 swap_executor INV-003 신규 1건 동반 → net 0 → 사용자 결정 rollback. |
| S-2.1+.2+.3 A1 cooperative yield trait method | ✅ DONE | `9b8bcddd` | −4 | `engine/src/yield_policy.rs` (L2) 신규. Backend::yield_after_layer default body가 logic 보유. 4 GPU backend override 제거. |
| S-2.5 A2 cpu_singleton 격상 | ✅ DONE | `9b8bcddd` | 0 | 3 GPU backend 의 `Arc::new(CpuBackend::new())` → `cpu_singleton()`. allocator overhead 제거. |
| S-3 B1 SecondaryStorage trait object signature | ✅ DONE | `aec296dd` | −4 | `Arc<SecondaryMmap>` → `Arc<dyn MmapKeepAlive>` + `Arc<RpcmemLayerRegion>` → `Arc<dyn RpcmemRegionGuard>`. 기존 trait 재사용 (신규 생성 0). |

**baseline 추이**: 176 → 174 (S-1.1 A4) → 171 (S-1.2 A3) → 167 (S-2 A1) → 163 (S-3 B1). 총 **−13**.

**INV-LAYER 카테고리별 (HEAD aec296dd)**:
- INV-LAYER-001: 6 (← 14)
- INV-LAYER-002: 4 (← 8)
- INV-LAYER-003: 122 (← 123, A3 forward_gen 1건 부수 효과)
- INV-LAYER-004: 4 (변화 없음)
- INV-LAYER-005: 27 (변화 없음, generate.rs L5)

**검증 (호스트)**:
- `cargo build -p llm_rs2 --lib` PASS.
- `cargo test --test spec inv_layer` 8 PASS.
- `cargo fmt --all` + `cargo clippy -p llm_rs2 --lib -- -D warnings` clean.
- `layer_lint --baseline` new_violations = 0.

**디바이스 게이트**: **미수행** — host gate로 충분 판단. trait method 변경 (cooperative_yield, alloc_alias_weight_buffer)이 hot path에 영향 가능하나 변경 본질은:
- yield_after_layer: 호출지/parameter 동일, 본문 logic만 trait default body로 이동 (compiler가 inline 가능).
- alloc_alias_weight_buffer: parameter type만 trait object로 — vtable 부분은 caller가 trait object cast 시점에 1회. signature 호출 부담 없음.

추가 디바이스 회귀 우려 시 S25 OpenCL Qwen2.5-1.5b Q4_0 32 토큰 bit-identical 또는 Δ≤2.5% 검증 가능 (별도 trigger).

---

## 다음 작업

### 1순위: INV-LAYER-003 122건 카테고리 정리

baseline 163 중 **75% (122건)**가 INV-LAYER-003 (L3 → L1 backend impl 직접 의존 / L3 cross-domain concrete import / L3 → cross-cutting concrete). 가장 큰 카테고리.

- 사전 분석: `python3 scripts/layer_lint.py --json | jq '.violations[] | select(.rule=="INV-LAYER-003")' | head -30`
- 그룹 분해: (a) L3-inference → L1 (downcast 패턴), (b) L3 → cross-cutting concrete (events/profile), (c) L3-pressure ↔ L3-inference, (d) L3 → resilience.
- 게이트: baseline -10 이상.
- 예상 작업량: 1~2일.

### 2순위: INV-LAYER-001 잔여 6건 + INV-LAYER-002 잔여 4건

- INV-LAYER-001 6건 (A1 5/14 → 6, A2 3, A3 0, A4 0, A5 1, gpu_self_meter 1). 작은 규모, 외과적.
- INV-LAYER-002 4건. 본 sprint 잔여 (auf/dtype_convert 1건 + B2 test block 2건 + …).
- 양쪽 종합 -5~-10 가능.

### 3순위: INV-LAYER-005 27건 — backlog 결정 선행

- generate.rs L5 → 상위 import 27건. legacy 보존 + "generate 바이너리 분할" sprint 결정 후 진입 가능.

---

## Landmines / 미해결 / 안 가본 길

1. **B4 (host_ptr_pool_buffer 이동) net 0 효과** — 파일을 memory/opencl → backend/opencl로 이동하면 INV-LAYER-002 1건 해소되나 swap_executor (L3) → 새 위치 (L1)가 INV-LAYER-003 신규 1건 만들어 net 0. 사용자가 rollback 결정. 향후 swap_executor 의 backend impl 의존 자체를 trait method (factory) 패턴으로 해소하는 작업이 같이 진행되어야 baseline 효과 발생.

2. **A2 cpu_singleton 격상은 baseline 효과 0 의도** — `Arc::new(CpuBackend::new())` → `cpu_singleton()` 치환은 cross-backend import 패턴 자체는 동일 (kind = L1↔L1 cross-backend import). baseline에서 import column만 `CpuBackend::new` → `cpu_singleton`로 갱신. 사용자 결정으로 진행 (allocator overhead 제거 가치). 향후 같은 패턴 결정 시 baseline 효과 0임을 다시 확인할 것.

3. **GpuSelfMeter 1건 (opencl/gpu_self_meter.rs:10) skip** — `crate::resilience::gpu_self_meter::GpuSelfMeter` struct 보유 패턴이라 trait extraction 비용 큼. 별 sprint 보류. baseline 1건 영구 잔존 가능성.

4. **`Arc::clone(&arc)` 형태는 trait object coerce 안 됨** — turbofish 없으면 type-infer가 concrete 결정 → unsized coercion 실패. `let dyn_arc: Arc<dyn ...> = arc.clone();` (method call form) 사용해야 let-binding 시점에 coerce 적용. S-3 4 caller 모두 이 패턴으로 fix.

5. **`alloc_alias_weight_buffer`의 dyn coerce caller 위치** — 4 caller (rpcmem_secondary.rs, swap_executor.rs ×2, qnn_oppkg/mod.rs internal forward) 모두 같은 패턴. 향후 같은 trait method에 새 caller 추가 시 동일 변환 필요.

6. **gpu_yield_impl 자체 제거됨** — `engine/src/resilience/gpu_yield.rs`는 thin re-export shim으로 축소. 외부 코드가 `gpu_yield_impl`을 직접 import하면 break. 향후 sprint에서 같은 위치를 손댈 때 grep 강제.

---

## 진입 게이트 (다음 세션)

- 1순위 INV-LAYER-003 진입 시:
  - 사전 측정: `python3 scripts/layer_lint.py --json | jq '.violations[] | select(.rule=="INV-LAYER-003")' | head -50`
  - 122건의 (source, target, kind) 분포 분석 → 그룹화 (data consumer / observability concrete / cross-domain / resilience).
  - 그룹별 작업 패턴 결정.
- 2순위 INV-LAYER-001+002 진입 시: review 스킬로 Plan 사전 리뷰.

## 참고 파일

- `engine/src/quant/flash_neon.rs` (S-1.2 신규, 5 함수 격상)
- `engine/src/yield_policy.rs` (S-2 신규, L2 env var cache)
- `engine/src/backend.rs:1195-` (`yield_after_layer` default body), `:903-936` (`alloc_alias_weight_buffer` 2 variants signature)
- `engine/src/backend/cpu/mod.rs` (`cpu_singleton()` helper)
- `engine/src/resilience/gpu_yield.rs` (thin re-export shim — 외부 호환만 유지)
- `engine/src/backend/opencl/mod.rs:5300-` (alloc_alias override), `:1540` (cpu_singleton)
- `engine/src/backend/qnn_oppkg/mod.rs:613-` (alloc_alias override)
- `engine/src/backend/cuda_embedded/mod.rs:530`, `engine/src/backend/cuda_pc/mod.rs:328` (cpu_singleton)
- `engine/src/models/weights/rpcmem_secondary.rs:413-` (caller coerce)
- `engine/src/models/weights/swap_executor.rs:1710-`, `:2869-` (caller coerce)
- `ARCHITECTURE.md` §13.8-K (chain 예외 정책)
- `scripts/layer_lint.py:266+` (`ALLOWED_BACKEND_CHAINS`)
- `engine/tests/spec/inv_layer_baseline.json` (163 entries)
