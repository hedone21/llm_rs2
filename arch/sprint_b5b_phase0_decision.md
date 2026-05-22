# Sprint B-5b Phase 0 — Backend Trait Extension + Data Consumer Category 결정

**작성**: 2026-05-22
**Sprint**: B-5b (옵션 2 + 옵션 3 통합)
**Phase**: 0 (구현 시작 전 architect 결정 라운드)
**다음 진입**: Phase 1 — R8 위치 정정 3건 + (b') DATA_CONSUMER_PATTERNS allowlist

---

## 본 라운드 목적

INV-LAYER-001 잔여 27건 (baseline 216 중) 해소를 위한 구현 시작 전 결정 라운드. 사전 분석에서 §13.8-J zone marker 단순 부착으로는 정책 무력화 risk가 큰 것이 드러나, 4 결정 항목(R1/R2/R4/R8)을 사전 확정 후 Phase 1~3 mechanical 진행.

## 결정 항목별 단일안

### R1. Backend trait segregation = (a) 현재 trait + default impl

**선택**: Backend trait에 4 method를 default impl로 추가. ISP 위반 누적은 별도 backlog로 분리.

추가 method:
- `cpu_companion(&self) -> &dyn Backend` — CpuBackend = self, GPU = injected CpuBackend
- `cpu_kernels(&self) -> Option<&CpuKernelSet>` — CpuBackend만 Some, GPU는 None (default)
- `as_opencl_secondary(&self) -> Option<&dyn OpenClSecondary>` — OpenCLBackend만 Some
- `yield_after_layer(&self, layer, is_decode)` — default no-op, OpenCLBackend override

**근거**: 후보 (b) `GpuBackend: Backend` sub-trait 분리는 `Arc<dyn Backend>` 사용처 291건을 모두 식별·타입 변경해야 하므로 sprint scope 초과. (c) `BackendExt` blanket impl + downcast는 capability query 패턴이 모호하고 trait 추출과 등가. default impl 패턴은 작업량 최소 + 호출지 영향 0 + impl 확장 자유.

**ISP 누적 우려**: Backend trait 57 → 61 method. 본질 정리는 별도 backlog "Backend trait segregation"으로 분리. 본 sprint scope 외.

**적용 method 위치**:
- `engine/src/backend.rs`에 trait method 4개 추가
- 신규 L2 trait: `engine/src/secondary.rs::OpenClSecondary` (R2와 결합), `engine/src/cpu_kernels.rs::CpuKernelSet` (struct of fn pointers)

### R2. opaque handle trait = (b) 부분 추출 (SecondaryStore만)

**선택**:
- `SecondaryStore` trait 신규 (engine/src/secondary.rs, L2) — `SecondaryMmap` + `RpcmemLayerRegion` 공통 API
- 공통 method 후보: `as_bytes() -> &[u8]`, `len() -> usize` (호출지 전수조사 후 확정)
- TransformerLayer / LayerSlot은 R4 (b') 카테고리로 grandfather (trait 추출 안 함)

**근거**: TransformerLayer는 10+ pub field (wq/wk/wv/wo/w_gate/w_up/w_down/attention_norm/ffn_norm/qkv_bias)로, trait 추상화 시 14+ method trait → 거짓 wrapper. 본질이 weight consumer dataflow임을 R4에서 인정하여 trait 추출 대신 data consumer 카테고리로 처리.

LayerSlot도 ArcSwap<LayerWeights> + secondary_mmap_handle을 들고 있는 swap-aware wrapper이므로 trait으로 표현 시 swap-time 동작이 leak. 마찬가지 grandfather.

SecondaryMmap / RpcmemLayerRegion 두 struct만 공통 API가 byte storage 단순 abstraction이라 trait 추출 ROI 있음.

**Risk**: `SecondaryStore` method set이 사용처 호출 패턴을 cover 못 하면 downcast 회귀. Phase 2 진입 전 호출지 전수조사 의무 (`grep -rn "SecondaryMmap\|RpcmemLayerRegion" engine/src/backend/`).

### R4. INV-LAYER-001 완화 = (b') kind 세분화 + DATA_CONSUMER_PATTERNS allowlist

**선택**:
- `spec/41-invariants.md` INV-LAYER-001 **본문 무수정**
- INV-LAYER-001 비고에 "data consumer 카테고리" 1단락 추가
- `scripts/layer_lint.py`에 `DATA_CONSUMER_PATTERNS` 정규식 allowlist 신설
- import path가 weight struct/enum 이름이면 data consumer로 자동 분류, baseline에서 제외

**DATA_CONSUMER_PATTERNS allowlist 초안**:
```python
DATA_CONSUMER_PATTERNS = [
    r"crate::models::weights::[A-Z]",              # struct/enum (UpperCamelCase 식별자)
    r"crate::layers::transformer_layer::TransformerLayer",
    r"crate::pressure::kv_cache::KVCache$",        # struct only, KVCacheOps trait 제외
]
```

**분류 결과 (예상)**:
- `LayerSlot` (3건): graph_cache.rs:17 + mod.rs:34 + 기타 → **data consumer**
- `SecondaryMmap` (3건): mod.rs:617 + mod.rs:5295 + 기타 → **data consumer**
- `RpcmemLayerRegion` (2건): mod.rs:618 + mod.rs:5296 → **data consumer**
- `TransformerLayer` (2건): layer_graph.rs:41,167 → **data consumer**
- `KVCache` struct (1건): mod.rs:7181 → **data consumer**
- `KVCacheOps` trait (1건): plan.rs:1250 → **control (Phase 1에서 L2 격상)**

→ 자동 분류 11건, baseline 216 → 205 (-11)

**근거**:
- 후보 (a) 정의 유지 → 거짓 wrapper 강제 + sprint 작업량 폭증
- 후보 (b) spec 본문 -001a/-001b 분할 → spec ID semantics 변경 + dataflow vs control 자동 detection 어려움
- 후보 (c) §13.8-K 신설 → 정책 인플레이션 (6 → 7), §F~J 누적과 일관성 risk

(b')는 spec 본문 무수정 + 정책 수 유지 + import path pattern으로 자동 detection 가능 + B-2/B-5a allowlist 정책과 동일 운용 원리 (5건 누적 시 명시화).

**Risk**:
- DATA_CONSUMER_PATTERNS 누락 시 false negative. Phase 1 진입 시 27건 manual review + spec test로 게이트
- 새 backend가 weight struct 우회 형태로 잘못된 import 시 자동 허용 가능. spec 비고에 "data consumer = weight struct/enum import만, function/trait method 호출은 본 카테고리 밖" 명시 필수

### R8. 위치 정정 추가 발굴 = 3건

**선택**:

| # | 대상 | 현재 위치 | 정정 위치 | 근거 |
|---|---|---|---|---|
| 1 | `OpenClEventGpuMeter` (struct + field type) | `resilience/gpu_self_meter.rs` | `backend/opencl/gpu_self_meter.rs` | OpenCL 전용 type. cl_event 추상화에 종속 |
| 2 | `maybe_yield_after_layer` (free fn) | `resilience/gpu_yield.rs` | `backend/opencl/gpu_yield.rs` + Backend trait `yield_after_layer` default | GPU command queue 조작이 본질. resilience 일반 인터페이스가 아님 |
| 3 | `KVCacheOps` (trait) | `pressure/kv_cache.rs` (struct + trait 합) | `engine/src/kv_cache_ops.rs` (L2, trait 부분만) | cache ops 추상화는 도메인 어휘 자산 (§G shared identifier promotion 패턴). KVCache struct는 R4 (b') data consumer로 처리 |

**추가 sweep 결과**: 위 3건 외 본 sprint에서 추가 위치 정정 후보 0. 나머지 위반은 trait extension (Phase 2) 또는 §J zone marker (Phase 3) 영역. C5 cpu_fallback, C6 with_opencl, C7 cpu::neon은 모두 cross-backend hot path/init-time이라 위치 정정 불가.

## 통합 Phase 분해 (216 → 186)

| Phase | 작업 | 게이트 | baseline 변동 |
|---|---|---|---|
| 0 | 본 결정 라운드 (산출물 3 commit) | architect 결정 4건 단일안 | 216 |
| 1 | R8 위치 정정 3건 + (b') layer_lint allowlist | clippy 0 + spec test 8 PASS + baseline regen | 216→201 (-15: 위치 정정 -4 + DATA_CONSUMER_PATTERNS -11) |
| 2 | R1 Backend trait default 4 method + R2 SecondaryStore trait | clippy 0 + spec test PASS + S25 Qwen 2.5 1.5B Q4_0 TBT ±3% | 201→189 (-12) |
| 3 | hybrid_attention 2건 §J 확장 + 잔여 cleanup | clippy 0 + spec test 8 PASS + ARCHITECTURE.md §J 본문 1줄 보강 | 189→186 (-3) |
| 4 | handoff + master FF + notify-send | 6~8 commits master FF | — |

**예상 최종**: 216 → 186 (-30, **13.9% 추가 감소**, INV-LAYER-001 27건 중 24건 + R8 3건 = -30)

**누적 B sprint**: 296 → 186 (-110, -37.2%)

## §13.8 정책 영향

- §13.8 정책 수 6 유지 (§F~§J, §K 신설 없음)
- (b') DATA_CONSUMER_PATTERNS는 spec 비고 + lint allowlist로 처리
- §J는 Phase 3 hybrid_attention 적용 시 본문에 "L1 build-time 정책 query에 한정. runtime hot path 호출 금지" 한 줄 보강 필요

## 산출물 (본 commit, 3건)

1. **본 문서** (`arch/sprint_b5b_phase0_decision.md`) — 결정 4건 + Phase 분해 + 정책 영향
2. **`ARCHITECTURE.md` §13.5** — V-?? Resolution Log에 plan 행 (V-01 gpu_self_meter / V-03 weight consumer / V-04 with_opencl / V-05 cpu_fallback / 신규 V-?? gpu_yield / V-?? KVCacheOps)
3. **`spec/41-invariants.md` INV-LAYER-001 비고** — data consumer 카테고리 1단락 추가

## 자기점검

- [x] 결정 4건 모두 단일안 확정 (a/b/c 양다리 없음)
- [x] 각 결정의 risk 명시
- [x] Phase 분해 게이트가 수치 또는 명령으로 표현
- [x] 정책 인플레이션 회피 (§13.8 6개 유지)
- [x] (b') DATA_CONSUMER_PATTERNS 초안 제공 (Phase 1 implementer가 27건 manual review로 확정)
- [x] R1 default impl 4 method가 ISP 누적 risk를 별도 backlog로 분리
- [x] hot path 영향 (R1 cpu_kernels 함수 포인터 호출)이 Phase 2 S25 microbench로 게이트
