# Handoff: B-5a Option δ (PartitionWs L2 격상 + §13.8-J zone) 종결 → 다음 sub-sprint

**작성**: 2026-05-22
**HEAD**: `5797267b docs(arch): B-5a-4 — V-02 RESOLVED partial SHA 채움`
**브랜치/Worktree**: `worktree-b5a_partition_option_delta`
**다음 세션 진입 문장**: `"B-5 잔여 카테고리 결정 (C2/C3+4/C5+6+7+8)"`
**선행 진입점**: [[handoff_b4_eval_l4_promotion_2026_05_22]]

---

## TL;DR

INV-LAYER-001 baseline 31건 중 C1(tensor_partition 7건)을 Option δ로 부분 해소. **§13.8-J dispatch orchestrator zone 정책 신설** + layer_lint zone marker parser 구현 + PartitionWs* L2 격상(§13.8-G 재적용) + PartitionPolicySnapshot 도입. baseline **220 → 216 (-4)**. 잔존 C1 1건(`plan.rs:17` multi-item use)은 PartitionContext가 함수 시그니처라 use 선언 제거 불가 — 후속 sprint. **멈춘 이유: B-5a 종결 + 다음 sub-sprint(B-5b~d 또는 C1 잔여 1건) 결정 대기**.

---

## 진행 상태

### 본 sprint commits (5건)

| HEAD | scope | 변경 |
|---|---|---|
| `83768154` | docs(arch) | B-5a-0 §13.8-J 정책 신설 (8 sub-section) + V-02 마킹 + spec INV-LAYER-001/003 비고 |
| `430d479c` | build(layer_lint) | B-5a-0b zone marker parser (+116 LOC) + 9 unit test |
| `232d45ec` | refactor(layer) | B-5a-1 PartitionWsCell/PartitionWorkspace L2 격상 (engine/src/partition_workspace.rs 신규 167 LOC) + 4 caller 시그니처 갱신 |
| `98293b25` | refactor(layer) | B-5a-2/3 PartitionPolicySnapshot field 추가 + build-time capture + §13.8-J zone marker (2건) + runtime self.policy 참조 |
| `5797267b` | docs(arch) | B-5a-4 V-02 RESOLVED partial SHA 채움 |

### 게이트 결과

| 게이트 | 결과 |
|---|---|
| `cargo build -p llm_rs2` | PASS |
| `cargo build --release` | PASS (64초) |
| `cargo fmt --all --check` | clean 0 |
| `cargo clippy --lib --no-deps -- -D warnings` | clean 0 |
| `cargo test --lib -p llm_rs2` | **1207 PASS** / 19 FAIL (B-4 종결 시 1205/21 대비 회귀 0) |
| `cargo test --test spec inv_layer` | **8 PASS / 0 FAIL** |
| `python3 scripts/test_layer_lint.py` | **9 PASS / 0 FAIL** (zone parser 단위 테스트) |
| **baseline JSON** | 220 → **216** (-4) |

### baseline 변동 누적 (B-1~B-5a)

| 단계 | 변동 | baseline |
|---|---|---|
| 진입 (Step 5b 종결 후) | — | 296 |
| B-1 EvictMethod | -9 | 287 |
| B-2 profile inversion | -33 | 254 |
| B-4 eval L4 격상 | -34 | 220 |
| **B-5a tensor_partition** | **-4** | **216** |
| 누적 | **-80 (-27.0%)** | |

### C1 카테고리 해소 매트릭스

| 위반 | 해법 | 상태 |
|---|---|---|
| `plan.rs:16 PartitionContext` (multi-use, trailing `::`) | use 선언 split + 함수 시그니처 분해 필요 | 잔존 |
| `plan.rs:20 PartitionWsCell` | L2 격상 (`crate::partition_workspace`) | ✅ 해소 |
| `plan.rs:427 PartitionWorkspace` | L2 격상 | ✅ 해소 |
| `plan.rs:515 partition_poll_flag_enabled()` (runtime) | `self.policy.poll_flag` (snapshot 참조) | ✅ 해소 |
| `plan.rs:3972 partition_poll_flag_enabled()` (build-time) | §13.8-J zone marker (`build_full_plan`) | ✅ 해소 |
| `build_partitioned_layer_plan` 안의 `partition_plan_enabled()` 등 | §13.8-J zone marker | ✅ 해소 |

총 7건 중 **6건 해소, 1건 잔존**.

### ARCHITECTURE.md 정책 신규 — §13.8-J

L1 backend 모듈의 *명시 zone marker* (`// LAYER-EXEMPT: dispatch_orchestrator`)가 표시된 함수/블록 내에서 L3 정책 query 함수 호출(env flag/feature flag readback, read-only build-time decision)을 INV-LAYER-001 baseline에서 제외. zone 밖에서는 위반 그대로. §G(식별자)/§H(매크로)/§I(모듈)/§J(zone) 4-way 직교성.

판별 4 기준:
1. marker 위치 (함수 시그니처 위 또는 블록 시작 다음)
2. zone 범위 (함수 본문 전체 또는 `// LAYER-EXEMPT-END`까지)
3. 호출 형태 제약 (정책 query 함수만, struct/trait method 금지)
4. 부수효과 금지 (read-only)

### Pattern별 해법

| Pattern | 해법 | 적용 위치 |
|---|---|---|
| **A. struct identifier 격상** | §13.8-G + TOP_LEVEL_L2 set 등록 | `engine/src/partition_workspace.rs` |
| **B. runtime env flag → snapshot** | build-time capture in `PartitionStep::new` wiring | `PartitionPolicySnapshot { poll_flag }` |
| **C. build-time env flag readback** | §13.8-J zone marker | `build_full_plan`, `build_partitioned_layer_plan` |
| **D. 함수 인자가 L3 type** | raw primitive 분해 | `PartitionWorkspace::new(gate_split_row, up_split_row, ...)` |

---

## 다음 작업

B-5 후속 후보 4건. 권장 순서: **C1 잔여 1건 → C5+C6+C7+C8 → C3+C4 → C2**.

### 옵션 1. C1 잔여 1건 해소 (0.5~1일)

- `plan.rs:17` use 선언의 `PartitionContext` 참조 제거
- 방법: `build_partitioned_layer_plan(partition_ctx: &PartitionContext, ...)` 시그니처를 raw primitive로 분해 (`PartitionWorkspace::new` 패턴 답습)
- 영향: caller 1~2건 + build 함수 본문 안 `partition_ctx.gate.split_row` 등 접근 모두 raw value로 치환
- 예상 baseline 감소: -1 (216 → 215)
- 우선도: 낮음 (B-5b/c가 더 큰 효과)

### 옵션 2. C5+C6+C7+C8 cross-backend + cpu::neon + resilience (1~1.5일)

- 14건 (C5 cpu_fallback 5 + C6 with_opencl 2 + C7 cpu::neon 4 + C8 resilience 2)
- 대부분 §13.8-J zone marker 또는 allowlist로 해소 가능
- 예상 baseline 감소: ~14 (216 → ~202)
- 위임: implementer (mechanical)

### 옵션 3. C3+C4 opaque handle trait (2~3일)

- 10건 (C3 TransformerLayer/LayerSlot 5 + C4 SecondaryMmap/RpcmemLayerRegion/KVCache 5)
- `BackendLayerHandle` trait + `SecondaryStore` trait 설계 필요
- 위임: architect (trait 설계) → senior-implementer (qnn_oppkg 적용)
- 예상 baseline 감소: ~10 (216 → ~206)

### 옵션 4. C2 hybrid_attention (0.5~1일)

- 2건만 (compute_kv_split + current scope)
- 위치 결정 (L2 격상 / L3 trait inversion / §13.8-J zone) 사전 라운드 필요
- 예상 baseline 감소: -2

---

## Landmines / 미해결

### 본 sprint 중 발견된 문제

1. **architect의 위치 권장 오류** — 초기 권장 위치 `inference/partition_workspace.rs`는 L3 도메인이라 INV-LAYER-001 해소 안 됨. **진짜 L2 격상 = `engine/src/` top-level + `TOP_LEVEL_L2` set 등록 + `classify_import` 매핑**. main session에서 직접 수정 (implementer 재호출 안 함).
2. **multi-item use 선언의 한계** — layer_lint이 multi-item `use crate::xxx::{A, B, C};`를 single entry로 record. 그 중 단 하나만 L3이라도 위반 1건 기록. 해소 = 모든 항목을 use에서 제거 (함수 시그니처/본문 의존 모두 해소). 부분 제거 불가.
3. **architect agent의 master worktree 직접 land 재발 안 함** — B-5a-0에서 worktree에 정상 land. patch 추출 fallback 발동 안 됨.

### 후속 cleanup backlog

1. **BC re-export 제거** — `engine/src/layers/workspace.rs:11`의 `#[deprecated] pub use crate::partition_workspace::*` (B-5a-1에서 1 sprint 한정 도입). 다음 sprint 진입 시 제거 검토.
2. **`instrument.rs` TOP_LEVEL_L2 누락** — B-2에서 `engine/src/instrument.rs`가 신규 L2였지만 layer_lint TOP_LEVEL_L2 set에 미등록. 현재 `classify_module('instrument') → unknown`. 본 sprint와 무관한 backlog.
3. **잔존 partition 위반 1건** — `plan.rs:17` use 선언. 후속 sprint 명시.

### "이 길은 가지 마라"

- **`inference/` 산하에 L2 자산 두지 말 것** — L3 도메인이라 backend에서 보면 똑같이 위반. L2 격상 = `engine/src/<name>.rs` top-level + `TOP_LEVEL_L2` set 등록 필수.
- **§13.8-J marker를 무차별 부착하지 말 것** — runtime hot path에 marker 부착하면 의도 왜곡. zone marker는 build-time 함수/블록 한정 (4 판별 기준 엄수).
- **multi-item use 선언을 split하지 말 것** — `use crate::xxx::{A, B};`를 두 줄로 split해도 layer_lint은 각각 entry로 잡아 같은 위반 수. 의미 없음.

### microbench skip 근거

- Option δ는 본질적으로 *struct 위치 이동 + build-time snapshot capture*. runtime hot path 코드 변경 0.
- PartitionStep::execute의 `partition_poll_flag_enabled()` → `self.policy.poll_flag`는 *struct field read*로 동등 비용 (오히려 더 빠를 수도).
- host release build PASS + INV-LAYER spec test PASS로 충분. device 측정 ROI 0.

---

## 참고 자료

- 메모리: [[layered-architecture-decision]] — sprint B 원안
- 직전 handoff: `.agent/todos/handoff_b4_eval_l4_promotion_2026_05_22.md`
- B-1/B-2 handoff: `.agent/todos/handoff_b1_evict_method_relocation_2026_05_22.md` / `handoff_b2_profile_inversion_2026_05_22.md`
- 정책 문서: `ARCHITECTURE.md` §13.8-J (신규), §13.5 V-02 RESOLVED partial 마킹
- 변경된 spec: `spec/41-invariants.md` INV-LAYER-001 §13.8-J 예외 + INV-LAYER-003 §13.8-J 향후 확장 여지
- 신규 L2: `engine/src/partition_workspace.rs`
- zone parser: `scripts/layer_lint.py` (+116 LOC) + `scripts/test_layer_lint.py` (신규 293 LOC, 9 PASS)
- 외부 contributor 진입 (§13.6): "build-time L3 정책 query 필요" → `// LAYER-EXEMPT: dispatch_orchestrator` marker (§13.8-J 참조)

## 자기점검 결과

- [x] 진입 문장이 한 줄로 명확? `"B-5 잔여 카테고리 결정 (C2/C3+4/C5+6+7+8)"`
- [x] "왜 멈췄는가"? B-5a C1 종결 (-4) + 잔여 카테고리 결정 대기
- [x] 가장 큰 landmine 표면화? architect 위치 권장 오류 (inference/ → top-level 정정) + multi-item use 선언 한계
- [x] 검증 게이트 수치/명령? clippy clean 0, test_inv_layer 8 PASS, baseline 220→216, test_layer_lint 9 PASS
- [x] 길이 적정? 본문 ~800 토큰, 5 commits + 패턴 매트릭스 + 4 후속 후보 + 5 landmine
