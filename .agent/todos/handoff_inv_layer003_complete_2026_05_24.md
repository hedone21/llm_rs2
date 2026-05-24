# Handoff: D-1 sprint (INV-LAYER-003 cross-domain 종결) → 다음 sub-sprint

**작성**: 2026-05-24
**HEAD**: `140640db refactor(layer): S-C3 — D2OVarianceCollector trait inversion + §13.8-O cross-L3 marker`
**브랜치**: `worktree-b5_trait_extension`
**다음 세션 진입 문장**: "INV-LAYER 잔여 사이클 진행" 또는 "§L hot path sub-trait 격상 시작"

---

## TL;DR

D-1 sprint (S-C1 + S-C2b + S-C5 + S-C4 + S-C3) 5개 commit으로 INV-LAYER-003 cross-domain 101건 → 0건 완전 해소. baseline 141 → 35 (-106). 정책 신설 3개 (§13.8-L/N/O) + §13.8-E 알고리즘 fix. 진짜 trait inversion (D2OVarianceCollector → VarianceObserver) 1건 + marker 우회 ~94건. 본질 trait inversion(L hot path 14건 + O 9건)은 backlog 등재. 잔여 35건은 INV-LAYER-005 (generate.rs 분할 결정 선행) + INV-LAYER-001 6 + INV-LAYER-002 2.

---

## 진행 상태

### sub-sprint commit chain

| sub-sprint | commit | baseline Δ | 핵심 |
|---|---|---|---|
| S-C1 | `c82727b0` | 141 → 83 (−58) | §13.8-L 정책 신설: EXT-anchored auto + `backend_concrete_downcast` marker. 38 fn/use marker 박음 |
| S-C2b | `d9710935` | 83 → 56 (−27) | §13.8-E `_find_test_block_ranges` 알고리즘 fix (entered_block flag). is_test_block 자동 INV-LAYER-001/002/003 제외 |
| S-C5 | `fbe61164` | 56 → 52 (−4) | qcf 잔재 1건 marker + §13.8-L 확장 (cross-L3 default init) |
| S-C4 | `65e4540c` | 52 → 46 (−6) | §13.8-N 정책 신설: `cross_cutting_trait_usage` marker. L3 ↔ cross-cutting trait/enum 6건 |
| S-C3 | `140640db` | 46 → 35 (−11) | D2OVarianceCollector → VarianceObserver trait 격상 + §13.8-O 신설: `cross_l3_vocabulary` 9 marker |
| **누적** | | **141 → 35 (−106)** | INV-LAYER-003 = 0, INV-LAYER-004 = 0 |

### 검증

- `cargo test -p llm_rs2 --test spec inv_layer`: 8/8 PASS (S-C1~S-C3 각 단계)
- `cargo test -p llm_rs2 --lib`: 1209 PASS / 19 failed (OpenCL host test = device-required, backlog [P3] 알려진 환경 이슈, 본 sprint 회귀 아님)
- `cargo fmt --all && cargo clippy -p llm_rs2 --lib`: clean
- `python3 scripts/layer_lint.py --baseline ... --json | jq '.new_violations | length'`: 0 (모든 sprint에서)
- 호스트 빌드만 검증 — 디바이스 게이트 미수행 (marker는 zero-cost, hot path forward 변경 없음)

### 새 정책 (ARCHITECTURE.md §13.8)

| § | marker | 적용 |
|---|---|---|
| §L | `backend_concrete_downcast` | L3→L1 backend impl downcast + cross-L3 default init |
| §N | `cross_cutting_trait_usage` | L3 ↔ cross-cutting trait/enum (V-10/V-12 등 §F 패턴) |
| §O | `cross_l3_vocabulary` | L3 ↔ L3 cross-domain (type alias default, public API surface) |
| §E 갱신 | (자동) | `#[cfg(test)] #[allow(...)]` 다중 attribute 패턴 인식 |

### baseline 현황 (35건)

```
INV-LAYER-005: 27 (engine/src/bin/generate.rs — "generate 분할" 결정 선행)
INV-LAYER-001: 6  (L4→L1 backend trait method 격상 — 이전 S-2 패턴)
INV-LAYER-002: 2  (top-level L2 위치 — 이전 S-1 패턴)
INV-LAYER-003: 0
INV-LAYER-004: 0
```

---

## 다음 작업 (3 갈래)

### A. 잔여 INV-LAYER 사이클 (baseline 35 → 0 직진)

| sub-sprint | 대상 | 예상 |
|---|---|---|
| S-D1 | INV-LAYER-002 2건 (`backend.rs` 2 + `auf/dtype_convert.rs` 1 + `memory/opencl/host_ptr_pool_buffer.rs` 1) — top-level L2 위치 정리 (이전 S-1 B4 패턴 재사용) | 30분 |
| S-D2 | INV-LAYER-001 6건 (opencl/mod.rs 2 + gpu_self_meter 1 + plan 1 + cuda_embedded/pc 각 1) — gpu_yield 같은 trait method 격상 (이전 S-2 패턴 재사용) | 1시간 |
| S-D3 | INV-LAYER-005 27건 (generate.rs) — "generate 바이너리 분할" 설계 라운드 선행. backlog [P2] `generate 바이너리 분할 + Manager 통합` | 별도 plan 필요 |

검증 게이트: 호스트 빌드 + spec test PASS + new_violations=0.

### B. §13.8-L hot path sub-trait 격상 (backlog [P2])

- **상세**: `[P2] §13.8-L hot path sub-trait 격상` backlog 항목 신설 (이번 sprint).
- **대상**: §L L-marker 75건 중 hot path 14건 (forward.rs/forward_gen.rs/attention.rs).
- **trait 설계**: `OpenCLContext` / `CudaContext` / `QnnOppkgContext` sub-trait + Plan executor 추상화 (가장 큰 산).
- **비용**: 3~5일. S25 Adreno OpenCL + Jetson CUDA + S25 qnn_oppkg 3종 bit-identical 디바이스 게이트 필수.
- **연관**: 기존 `[P2] KiviCache hot path downcast resolve` 와 통합.

### C. §13.8-O cross-L3 vocabulary trait inversion (backlog [P2])

- **3 트랙 분할**:
  1. **WeightSwapDispatch trait** (3건): `pressure/weight_swap_handler` → `models/weights/swap_handler` 이동 + trait. ActionResult enum 의존 잔존.
  2. **PrefetchAccess + PreloadPool L2 격상** (3건): forward_into_offload 분리 결정 선행.
  3. **KvCacheView trait** (3건): KVCacheOps trait의 default 명시 강제. 외부 API ripple 큼.
- **비용**: 1~3일 (3 트랙 sub-sprint 분할 가능). 디바이스 게이트는 ModelForward path 한정.

---

## Landmines / 미해결

### 1. `_find_test_block_ranges` 알고리즘 fix는 layer_lint 전반에 영향
- S-C2b에서 `entered_block` flag 추가로 false positive 차단. 이전에 인식 안 되던 test block이 새로 인식됨 → INV-LAYER-001/002/003 baseline에서 자동 제외되어 갑작스러운 baseline 변동 가능. 측정값 비교 시 sprint 경계 주의.

### 2. `layer_lint.py`의 zone marker 인식은 `entered_block` 의존
- §L (S-C1) + §J (기존)의 fn form zone도 entered_block flag 적용됨. 단일 줄 fn(`fn name() -> &str { "x" }`)도 brace_depth 양수가 발생하므로 정상 인식. 단 marker 위치가 attribute 뒤에 박혀야 함 (extract_imports의 attribute skip 로직 의존).

### 3. §13.8-O marker는 본질 trait inversion 회피
- D-1 sprint의 시간 단축을 위한 cost-effective 결정. backlog [P2] §O 항목으로 본질 격상 후보 9건 등재. 향후 trait inversion 진행 시 register에서 항목 제거 + marker 삭제.
- ActionResult enum(`pressure::ActionResult`)이 weight_swap orchestrator의 §F enum-as-data identifier — WeightSwapHandler를 models로 이동 시 잔존 위반 1건. §F 패턴으로 자동 제외 가능.

### 4. INV-LAYER-005 generate.rs 27건은 backlog [P2] generate 분할 결정 선행
- "generate 바이너리 분할 + Manager 통합" backlog 항목. cli/chat/experiment/resilience 잠정 분할. 본 sprint scope 밖.

### 5. 마커 운영 정책 — 5건 이상 누적 시 register 표 신설
- §13.8-L (L-marker 75건), §13.8-O (9건), §13.8-N (6건) 모두 5건 초과 — ARCHITECTURE.md §13.4에 register 표 신설 검토 후보.

### 6. 가지 않은 길
- KVCache L2 격상 (pressure-specific eviction method 분리 필요, ripple 큼)
- Plan executor 추상화 (§L hot path sub-trait 격상의 가장 큰 산)
- INV-LAYER-005 generate.rs 분할 — 별도 설계 라운드 필요

---

## 진입 명령 (다음 세션)

```
"§13.8-L hot path sub-trait 격상 시작"   # backlog [P2] B 트랙
"§13.8-O WeightSwapDispatch trait 격상"  # backlog [P2] C 트랙 1
"INV-LAYER 잔여 8건 정리"                # S-D1 + S-D2 (35건 → 27건)
"generate 분할 설계 라운드"              # INV-LAYER-005 27건 해소 전제
```
