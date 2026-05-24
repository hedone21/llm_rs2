# Handoff: S-D1+S-D2 sprint (INV-LAYER-001/002 7건 해소) → 다음 sub-sprint

**작성**: 2026-05-24
**HEAD**: `f75b6eab refactor(layer): S-D1+S-D2 — INV-LAYER-001/002 7건 해소 (baseline 35→28)`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "§13.8-L hot path sub-trait 격상 시작" 또는 "§13.8-O WeightSwapDispatch trait 격상" 또는 "tensor_partition L2 격상"

---

## TL;DR

S-D1+S-D2 단일 commit으로 INV-LAYER-001/002 7건 해소. baseline 35 → 28 (−7). S-D1 (INV-LAYER-002, 2건)은 진짜 구조 정리: `f16_to_f32`/`quantize_q4_0` L2 격상 + `host_ptr_pool_buffer.rs` 파일 이동. S-D2 (INV-LAYER-001, 5건)는 marker 처리: §13.8-P 신설 (`cross_backend_bootstrap`, 4 marker) + §N 확장 (L1→cross-cutting trait, 1 marker). 잔여 28건은 INV-LAYER-005 27 (generate.rs 분할 결정 선행) + INV-LAYER-001 1 (V-02 tensor_partition L2 격상 별 sprint 대기).

---

## 진행 상태

### sub-sprint 결과

| sub-sprint | 대상 | 처리 방식 | baseline Δ |
|---|---|---|---|
| S-D1.1 | INV-LAYER-002 `auf/dtype_convert.rs:22` | `f16_to_f32`/`quantize_q4_0` 를 `quant/convert.rs` (L2) 격상, `models/loader/convert.rs` re-export, auf import path 갱신 | −1 |
| S-D1.2 | INV-LAYER-002 `memory/opencl/host_ptr_pool_buffer.rs:26` | 파일 이동 `memory/opencl/` → `backend/opencl/`, 3 caller import 갱신 | −1 |
| S-D2-§P | INV-LAYER-001 `cpu_singleton` 3건 + `CpuBackend::new` placeholder 1건 | §13.8-P 신설 + 4 marker (`// LAYER-EXEMPT: cross_backend_bootstrap`) | −4 |
| S-D2-§N확장 | INV-LAYER-001 `gpu_self_meter.rs:10` GpuSelfMeter | §N 적용 범위에 L1→cross-cutting trait 추가 + 1 marker (`cross_cutting_trait_usage`) | −1 |
| **누적** | | | **35 → 28 (−7)** |

### 검증

- `cargo test -p llm_rs2 --test spec inv_layer`: **8/8 PASS**
- `cargo test -p llm_rs2 --lib`: 1204 PASS / 24 fail (OpenCL host test = backlog [P3] device-required 환경 이슈, 본 sprint 회귀 아님)
- `cargo fmt --all && cargo clippy -p llm_rs2 --lib -- -D warnings`: clean
- `python3 scripts/layer_lint.py --baseline engine/tests/spec/inv_layer_baseline.json`: **new_violations = 0**
- 호스트 빌드만 검증 — 디바이스 게이트 미수행 (파일 이동 + re-export + marker만 변경, hot path 비영향)

### 새 / 갱신 정책 (ARCHITECTURE.md §13.8)

| § | marker | 적용 |
|---|---|---|
| §P (신설) | `cross_backend_bootstrap` | L1↔L1 cpu_companion field init + placeholder Tensor backend Arc (4건) |
| §N (확장) | `cross_cutting_trait_usage` | L1 → cross-cutting trait impl-only import 케이스 추가 (V-01 GpuSelfMeter 1건) |

### baseline 현황 (28건)

```
INV-LAYER-005: 27 (engine/src/bin/generate.rs — "generate 분할" 결정 선행)
INV-LAYER-001:  1 (V-02 backend/opencl/plan.rs:16 tensor_partition — L2 격상 별 sprint)
INV-LAYER-002:  0
INV-LAYER-003:  0
INV-LAYER-004:  0
```

---

## 다음 작업 (3 갈래)

### A. tensor_partition L2 격상 (S-D3, 잔여 INV-LAYER-001 1건 직진)

| 단계 | 대상 | 예상 |
|---|---|---|
| 검토 | `engine/src/layers/tensor_partition.rs` (1575 LOC) — 3 caller (transformer.rs, transformer_layer/mod.rs, plan.rs). PartitionContext / PartitionPath / partition_plan_* helpers 분리 가능성 | 30분 |
| 옵션 1 | tensor_partition module 전체를 top-level (L2 `engine/src/tensor_partition.rs`)로 이동 — PartitionContext/PartitionPath가 L3 dep 없는 pure config면 가능 | 1~2h |
| 옵션 2 | partition_plan_* helper만 추출하여 `engine/src/partition_config.rs` (L2)로 격상, PartitionContext는 layers/에 유지. plan.rs는 config helper만 import | 1h |

→ baseline 28 → 27. 디바이스 게이트는 partition path 영향 없으면 호스트 빌드 충분.

### B. §13.8-L hot path sub-trait 격상 (backlog [P2])

- **상세**: §L L-marker 75건 중 hot path 14건 (forward.rs/forward_gen.rs/attention.rs).
- **trait 설계**: `OpenCLContext` / `CudaContext` / `QnnOppkgContext` sub-trait + Plan executor 추상화.
- **비용**: 3~5일. S25 Adreno OpenCL + Jetson CUDA + S25 qnn_oppkg 3종 bit-identical 디바이스 게이트 필수.

### C. §13.8-O cross-L3 vocabulary trait inversion (backlog [P2])

- **3 트랙 분할**:
  1. WeightSwapDispatch trait (3건)
  2. PrefetchAccess + PreloadPool L2 격상 (3건)
  3. KvCacheView trait (3건)
- **비용**: 1~3일.

---

## Landmines / 미해결

### 1. §13.8 정책 비대화 — register 표 신설 검토 필요
- §13.8-L (L-marker 75건), §13.8-O (9건), §13.8-N (7건 = 6 + S-D2 1건), §13.8-P (4건) 모두 5건 초과.
- ARCHITECTURE.md §13.4에 cross-marker register 표 신설 검토 후보. 본 sprint에서도 미수행.

### 2. `_find_exempt_zone_ranges` fall-through은 일반 `//` 주석을 code line으로 간주
- marker 추가 후 부연 일반 주석을 그 다음 라인에 두면 zone 인식이 부연 주석 라인에서 끝나서 실제 code 라인을 cover 안 함.
- S-D2 진행 중 V-01 GpuSelfMeter / V-?? CpuBackend placeholder 2건이 이 패턴으로 잡혔다가 marker를 code 라인 바로 위 1줄로 축소하여 해결.
- 향후 marker 박을 때 부연 설명은 marker 위로, marker 자체는 code 라인 직전 단일 줄 형태로 유지.

### 3. §N L1 → cross-cutting trait 확장은 적용 범위 모호성 위험
- §N original 결정문: "L3 ↔ cross-cutting". S-D2 확장으로 L1→cross-cutting까지 허용.
- 위험: 향후 L4/L5 → cross-cutting trait import 케이스도 같은 marker로 처리하려는 유혹.
- 가드: §N marker는 *trait impl-only import* 한정 (S-D2 추가 문구). concrete struct/function import 금지 운용 정책 유지.

### 4. V-02 tensor_partition baseline 유지 — 별 sprint 가치
- plan.rs:16의 `crate::layers::tensor_partition::{PartitionContext, PartitionPath, partition_plan_debug_enabled, partition_plan_enabled, partition_trace_enabled, record_partition_timing}` import는 본질적으로 L2 격상 필요.
- 본 sprint에서 marker 처리도 가능했으나 (`partition_config_access` 같은 신규 정책 신설) 정책 비대화 회피로 baseline 유지 선택.
- 후속: tensor_partition module 자체를 top-level L2 또는 partition_config helper만 추출.

### 5. OpenCL host test 24 fail은 기존 [P3] backlog 환경 이슈
- 본 sprint 회귀 아님. spec test 8/8 + lib 1204 PASS / 24 fail (OpenCL device-required) 패턴은 이전 sprint와 동일.
- 디바이스 게이트는 별 sprint (§L hot path) 진입 시점부터 강제.

### 6. 가지 않은 길
- INV-LAYER-005 generate.rs 27건 (baseline 의 96%) — "generate 바이너리 분할" 결정 선행 미수행.
- §13.4 register 표 신설 (마커 운영 가시화)
- §N 확장 외 §P 와의 의미 명확화 (현재는 dst layer 분리로 충분)

---

## 진입 명령 (다음 세션)

```
"tensor_partition L2 격상"               # S-D3 (baseline 28 → 27)
"§13.8-L hot path sub-trait 격상 시작"   # backlog [P2] B 트랙
"§13.8-O WeightSwapDispatch trait 격상"  # backlog [P2] C 트랙 1
"generate 분할 설계 라운드"              # INV-LAYER-005 27건 해소 전제
```
