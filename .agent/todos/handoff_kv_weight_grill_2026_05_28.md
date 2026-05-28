# Handoff — KvBundle/WeightBundle Grill 종결 → Phase α-W 진입 (본 sub-grill 2026-05-28~29 누적)

> **일자**: 2026-05-28 (본 grill) + 2026-05-28~29 (본 sub-grill 누적)
> **작성자**: Architect (orchestrator 보고용)
> **진입 문장 (현재)**: **"본 sub-grill 잔여 미해결 sub-trait detail finalize — Q-#1-3 K/V raw read 노출 여부 부터"**
> **이전 진입 문장 (본 sub-grill 후 supersede)**: ~~"Pipeline stage Phase α-W 진입 — Weight + PipelineStage 인프라 + Bundle 폐기"~~
> **선행 문서**:
> - `arch/pipeline_stage_design.md` (본 sprint 단일 진실원본, 2026-05-27 23 라운드 grill + **2026-05-28 본 grill 12 결정 + 후속 2 결정 + 2026-05-28~29 본 sub-grill 4 결정 + 갈래 B 메타 결정** 반영. **§0 Executive Overview 진입점**.)
> - `docs/adr/0001-kv-dispatch-paradigm.md` (KV dispatch Generic → Trait object 정식 결정)
> - `spec/41-invariants.md` §3.28 (INV-DECODE-STAGE-001/004~007 + INV-KVCACHELAYER-* + INV-STAGE-LAYER-HANDLE; INV-DECODE-STAGE-002/003 폐기; INV-LAYER-006 / INV-STAGE-LAYER-HANDLE / INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC 본문 갱신 — 본 sub-grill 결정 반영)
> **이전 handoff**: `.agent/todos/handoff_pipeline_stage_design_2026_05_27.md` — 본 grill 결정으로 supersede (Phase α 분리 등). 이전 handoff 의 일부 결정사항은 본 grill 에서 변경됨, 변경 매트릭스는 R6 참조.

---

## R0. 본 sub-grill (2026-05-28~29) 후 신규 진입점

본 sub-grill 종결 후 다음 세션 진입 추천 순서:

### 신규 진입 추천 순서

1. **Q-#1-3 — K/V raw read 노출 여부** (다음 sub-grill 진입점, R5 #1 Q24-1 detail)
   - D2O cosine similarity 가 K read 필요 — KVCacheView 에 raw K/V slice 노출 요구.
   - 갈래 (a) noexpose + paradigm-specific helper / (b) raw K/V slice 노출 (paradigm-agnostic 가정 깨짐) / (c) Tier 3 capability trait (`Arc<dyn DenseKVRead>`)
2. **Q-#1-4 — capacity 중복** (Q-#1-3 와 같은 sub-grill round)
3. **Q-#1-5 — mutation 누설** (Q-#1-3/4 와 같은 sub-grill round)
4. **#2 — WeightLayerView** (Phase α-W 진입 전)
5. **#3 — SecondaryStore** (Phase α-W 진입과 같이 진행)
6. **#4 — SparsePattern** (별 sprint)
7. **#6 — system/ 명명** (Phase α-W stages/ 디렉토리 신설 commit 전)
8. **#11 — Layer impl backend ref 보유 패턴** (Phase α-W 진입 전 필수)
9. **#12 — KVCacheLayer / WeightLayer impl 시그니처 detail finalize** (Phase α-W 진입 전 필수)

본 sub-grill 결정 4 건 + 갈래 B 메타 결정 요약은 R1 참조. 미해결 sub-grill 매트릭스는 R5 + arch §13.6 참조.

### 본 sub-grill (2026-05-28~29) 누적 결정 요약

- **결정 #15**: 3-tier Stage 패턴 (Tier 1 Primitive-only / Tier 2 Paradigm-specific concrete Arc / Tier 3 Cross-paradigm capability trait Stage 측 정의). 부산물: `StorageSpec` 폐기, `apply_storage(spec)` method 폐기, `KVCacheLayer::as_any()` 부재.
- **결정 #16**: Stage cardinality 자유 (1/N/0 layer).
- **결정 #17**: Score 도메인 별 sprint (pragmatic deferral + asymmetry intentional). EvictionHook 1:1 wrap. R5 #13 신규 등록.
- **결정 #18**: `KVCacheView::dtype()` 폐기. Backend / Layer 내부 보관.
- **메타 결정 — 갈래 B (Boundary 명시)**: PipelineStage 적용 범위 = KV/Weight state mutation 한정. Cross-cutting 은 자기 패턴 인정. asymmetry intentional.

### llm.npu / mllm-NPU (ASPLOS 2025) 정합도 평가

본 grill 구조 vs llm.npu 3-level reconstruction:
- **Prompt level chunking** → ✓ GREEN (`PrefillChunkingStage` Tier 1 신설 가능, forward-compat)
- **Tensor level outlier** → ✓ GREEN (Tensor Partition 패턴 + `LayerDispatch::Partition` 정합)
- **Block level scheduling** → △ YELLOW (Layer impl backend ref 보유 = OK, 단 async/out-of-order PipelineStage sync 가정 충돌 → `AsyncBlockScheduler` 별 추상화 필요)

본 sprint scope 외, 본 grill 구조가 **막지 않음 (forward-compat)** 단 흡수도 안 함. Phase NPU-1~5 가설 plan 으로 점진 도입 가능. 상세는 arch §13.6.5.

### Architect 위임 누적 변경 매트릭스 (본 sub-grill commit 후 상태)

| 파일 | 변경 내역 |
|---|---|
| `arch/pipeline_stage_design.md` | §0 Executive Overview 신설 + §3.5/3.6/5.2/5.3/5.4/13.1 갱신 (3-tier 패턴 + cardinality + StorageSpec 폐기 + dtype 폐기) + §13.6 본 sub-grill 매트릭스 신설 (결정 4 + 갈래 B + 미해결 9건 + 발견 모순 10건 + llm.npu 부록) + §16 누적 결정 14 → 18 + §16.3 변경 추적성 4 row 추가 |
| `spec/41-invariants.md` | INV-LAYER-006 본문 갱신 (PipelineDispatcher 위치 L4 finalize) + INV-STAGE-LAYER-HANDLE 본문 갱신 (3-tier + cardinality 자유) + INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC 본문 보강 (Tier 1 한정 + dtype 노출 X) + §3.28 변경 요약에 본 sub-grill 5 결정 추가 |
| `arch/README.md` | pipeline_stage_design.md 행 갱신 + 본 sub-grill (2026-05-28~29) callout 절 추가 (3차 review 표시: §0 Overview + 4 결정 + 갈래 B + llm.npu 정합도) |
| `.agent/todos/handoff_kv_weight_grill_2026_05_28.md` | R0 신규 진입점 + R1 결정 14 → 18 + R5 #5 StorageSpec 해소 + #10 위치 결정 (L4 finalize) + #13/#14 신규 등록 + R6 본 sub-grill 변경 내역 명시 + 자기점검 row 추가 |

신규 file 작성 없음 (handoff 통합). 다음 세션 진입은 본 파일 R0 + arch §0 Executive Overview 동시 진입.

### 자기점검 (handoff-doc 스킬, 본 sub-grill 진입 검증)

| 점검 항목 | 확인 |
|----------|------|
| 진입 문장 신규 | "본 sub-grill 잔여 미해결 sub-trait detail finalize — Q-#1-3 K/V raw read 노출 여부 부터" |
| §0 Executive Overview 진입점 | arch/pipeline_stage_design.md §0 standalone 으로 본 sprint 전체 그림 파악 가능 |
| 누적 결정 명시 | 본 grill 14 (12 + 후속 2) + 본 sub-grill 4 = 18 결정 + 갈래 B 메타 결정 |
| 미해결 sub-grill 매트릭스 | R5 (Q-#1-3~#12, 9건) + arch §13.6.3 (동일 표) |
| 발견 모순 매트릭스 | arch §13.6.4 (10건) → 갈래 B 채택 근거 |
| llm.npu 정합도 평가 | arch §13.6.5 부록 (Phase NPU-1~5 가설 plan, 본 sprint scope 외 forward-compat) |
| Architect 위임 누적 변경 매트릭스 | R0 (4 파일 변경 내역) |
| 신규 sub-grill 등록 (Score 도메인) | R5 #13 + #14 (Multiple EvictionStage) |

---

## R1. What was decided (결정사항 요약 — 본 grill 14 결정 + 본 sub-grill 4 결정 = 누적 18)

### 본 grill + 본 sub-grill 누적 결정 18 건 (본문 12 + post-grill 후속 2 + 본 sub-grill 4)

| # | 결정 | 의미 | 출처 |
|---|---|---|---|
| 1 | **(β) sync model** | Buffer-level lazy + access-mode-aware. R1/R2/R3 자동. 새 sync 인프라 0건. INV-KVBUNDLE-SYNC 폐기. | 본 grill |
| 2 | **(d-1) primitive only — storage-format-agnostic** | KVCacheLayer method 가 token/layer/range granularity 만 알고 storage paradigm 모름. KVCacheLayer impl 이 mechanism 캡슐화. | 본 grill |
| 3 | **Q7 (A) 유지** (AttentionKernels sub-trait 분리 (A') 보류) | 본 프로젝트 KiviAttentionBackend / GpuScoreAccess 패턴 + frequency × cost 정당화 부족. | 본 grill |
| 4 | **Q8 Mixed storage 허용** | Layer 별 다른 storage paradigm OK (layer 0 = KIVI Q4, layer 1 = TurboQuant, ...). | 본 grill |
| 5 | **Q9 호환성 차단 인프라 X** | 만나면 panic. | 본 grill |
| 6 | **KvBundle / WeightBundle trait 폐기** | Stage register 시점 layer handle 보관 → ctx layer field 자체 폐기. | 본 grill |
| 7 | **StageContext 5 field → 3 field** | `kv` / `weights` 폐기. `step` / `backend_ext` / `profiler` 유지. | 본 grill |
| 8 | **(γ) Layer handle 전환** | KVCacheLayer / WeightLayer trait + interior mutability. Stage 가 `Arc<dyn ...>` 보관. LayerSlot::rcu_weights 자연 확장. | 본 grill |
| 9 | **KV dispatch 갈래 2 Trait object 전면 채택** | kv_cache_ops.rs:53 정책 정면 반전. **ADR-0001 작성 필수**. 갈래 1/3/4 REJECTED. | 본 grill |
| 10 | **Weight dispatch: LayerSlot + WeightLayer thin wrap** | Forward path 무변경. ~5 file 추가. KV 와 비대칭. | 본 grill |
| 11 | **Sprint 분리: Phase α-W → ADR-0001 → Phase α-K** | Risk 분산 + escape hatch. | 본 grill |
| 12 | **LayerDispatch enum: Fixed 3 variant (Full / Skip / Partition)** | enum + match exhaustive 가독성 우월. | 본 grill |
| **13** | **PipelineDispatcher trait 유지** (post-grill 후속) | 단일 impl 우려 반박 — deletion test (INV-LAYER-006 위반) + mock 패턴 정합 (mock_engine/mock_manager 정착) + vtable cost noise 이하. 위치 (L2 vs L4) 만 별 sub-grill (R5 #10). | 본 grill 후속 |
| **14** | **BackendExtensions trait 폐기** (post-grill 후속) | §13.8-L Backend trait capability provider 패턴 중복 추상화 + `as_opencl_secondary()` 명명 leak. (γ) 정신 일관 — Layer impl 이 backend ref 보유 + capability 내부 호출. **StageContext 3 → 2 field** (`step` / `profiler` 만). Sub-grill 2건 분리 (R5 #11/#12). | 본 grill 후속 |
| **15** | **3-tier Stage 패턴 (Tier 1 Primitive-only / Tier 2 Paradigm-specific / Tier 3 Cross-paradigm)** | Tier 2 concrete `Arc<ConcreteLayer>` register 시점 compile-time type 강제, downcast 0. Tier 3 capability trait Stage 측 정의 (base trait 변경 0, OCP 보존). **부산물**: `StorageSpec` / `WeightStorageSpec` trait 폐기 (Q24-5 자연 해소), `apply_storage(spec)` method 폐기, `KVCacheLayer::as_any()` 부재. | **본 sub-grill (2026-05-28~29)** |
| **16** | **Stage cardinality 자유 (1/N/0 layer)** | 임의 1-layer 가정 폐기. 1 layer (signal-driven paradigm-specific) / N layer (cross-layer policy) / 0 layer (backend-only). 본 grill 결정 #6 자연 해석 명문화. | **본 sub-grill (2026-05-28~29)** |
| **17** | **Score 도메인 별 sprint — pragmatic deferral + asymmetry intentional** | EvictionHook → EvictionStage 1:1 wrap. score_accumulator field 보존 (concrete type). Score 도메인 hot path (collection backend inline) / cold path (aggregation+read PipelineStage) 의 자연 책임 분리 — asymmetry intentional. 별 sprint 추천 갈래 4/7/2, Pre-rejected 1/3/5/8. R5 #13 신규 등록. | **본 sub-grill (2026-05-28~29)** |
| **18** | **`KVCacheView::dtype()` 폐기** | dtype 사용처 5건 모두 backend / layer 내부 — 외부 노출 0 필요. Mixed paradigm 의미 모호. INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC 정신 정합. Backend / Layer 내부 보관. Profiler / Manager IPC 필요 시 별 capability trait (`LayerProfileable`) 또는 backend trait 경로. | **본 sub-grill (2026-05-28~29)** |

### 메타 결정 — 갈래 B (Boundary 명시, 본 sub-grill 2026-05-28~29)

본 sub-grill 의 발견 모순 10건 (arch §13.6.4) 의 본질 = **"single pattern fits all" 가정의 한계**. PipelineStage 가 모든 cross-cutting 책임을 흡수하려 하지 않는다. 채택:
- **PipelineStage 적용 범위 = KV/Weight state mutation 도메인 한정**
- **Cross-cutting (score collection / dispatch / cross-paradigm policy / Backend capability) = 자기 패턴 인정**
- **발견된 asymmetry 는 도메인 본질의 정직한 표현 — intentional**

### Spec INV 변경 (본 grill 결정 → 카탈로그 반영)

| 변화 | INV | 사유 |
|---|---|---|
| **폐기** | INV-DECODE-STAGE-002 (KVBUNDLE-CONSISTENCY) | KvBundle trait 자체 폐기로 자연 해소. layer-wide vs per-layer 구분 사라짐. |
| **폐기** | INV-DECODE-STAGE-003 (KVBUNDLE-SYNC) | (β) sync 모델 채택. R1/R2/R3 자동 처리 흡수. |
| **신규** | INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC | KVCacheLayer mutation method 는 storage-format-agnostic. Stage 가 dtype/codebook/rotation 모름. |
| **신규** | INV-KVCACHELAYER-PAIRED-KERNEL | KVCacheLayer impl 과 paired backend attention kernel 매핑 의무. |
| **신규** | INV-STAGE-LAYER-HANDLE | PipelineStage 가 layer handle 을 register 시점 보관. ctx 에 kv/weights field 두지 않음. |
| **수정** | INV-DECODE-STAGE-001 (KV-PHASE) | mutation method 표현이 ctx.kv → Stage 보유 KVCacheLayer handle 로 변경. 정신 유지. |
| **수정** | INV-DECODE-STAGE-006 (CTX-AUTHORITY) | 5 field → 3 field (본 grill) → **2 field** (`step` / `profiler`, 본 grill 후속 결정 14). `backend_ext` 폐기 — BackendExtensions trait 자체 폐기. 권한 강제 영역 = profiler 1 field 한정. |

### Sprint 분리 (본 grill 결정 #11)

| Phase | 기간 | 게이트 |
|---|---|---|
| Pre-α-1 | 2~3일 | design round PASS (sub-trait detail finalize) |
| **Phase α-W** | **2~3주** | (a) S25 bit-identical, (b) avg_tbt Δ ≤ +3%, (c) Weight swap 정확성 회귀 0건, (d) 신규 INV PASS |
| **ADR-0001** | **2-3일** | Architect review + kv_cache_ops.rs:53 주석 갱신 (구현은 Phase α-K) |
| **Phase α-K** | **4~6주** | (a) S25 bit-identical (5 KV paradigm), (b) avg_tbt Δ ≤ +3%, (c) Mixed storage test PASS, (d) R-G1~G5 GREEN |
| Phase β | 3~4주 | DecodeLoop 재작성 + arch/inference_pipeline.md v2 |
| Phase γ | 3~4주 | legacy generate.rs 잔여 마이그레이션 + PACT2026 PoC |

**총 12~19주** (이전 grill 8~13주 → 본 grill +4-6주, KV refactor risk 분리).

---

## R2. Why this matters (배경 + 의미)

### 본 grill 이 풀어낸 것

1. **이전 grill (2026-05-27) 의 KvBundle 8 method / WeightBundle 10 method 의 god abstraction 문제**:
   - KvBundle trait 가 ctx field 로 모든 layer state 노출 → PipelineRegistry 정신 (stage 객체 안 캡슐화) 직접 위반
   - god ctx 5 field 인정 + code review 책임 (M1만) 보강이 정합하지 않음 (kv/weights 권한 경계 위반 risk 가 ctx 안에서 잠재)
2. **본 프로젝트 LayerSlot/RCU 패턴 미활용**: 이전 grill 은 KvBundle/WeightBundle 을 신규 trait 으로 정의했으나, 본 프로젝트는 이미 `LayerSlot::rcu_weights` (slot.rs:158) 패턴으로 layer-self mutation 정착. (γ) interior mutability 가 자연 확장.
3. **KV/Weight 도메인 비대칭 미인식**: 이전 grill 은 두 도메인을 동일 패턴 (Bundle trait) 으로 묶었으나, 실제는 Weight 가 RCU 정착 (~5 file 추가), KV 가 Generic monomorphization 강결합 (~20 file refactor). Sprint 분리가 risk 분산에 필수.
4. **kv_cache_ops.rs:53 의 명시 정책**: 이전 grill 은 KV dispatch paradigm 미명시 (Generic 가정). 본 grill 에서 정책 반전을 정식 ADR 로 격상 — 미래 explorer 가 정당화 근거를 찾을 수 있음.

### 본 grill 의 인정 비용

- **R-G1 vtable cost (RPN 144)**: Layer-step KV write 매 `Arc<dyn>::call` ~1-3 ns. Phase α-K PoC 게이트 (Δ ≤ +3%) 통과 필수.
- **R-G4 refactor scope (RPN 168)**: CacheManager / EvictionPolicy / D2OHandler / CachePressurePipeline ~10 component 마이그레이션.
- **R-G5 bit-identical (RPN 168)**: compact(keep, merges) atomic semantic 으로 인한 D2O/KIVI/SnapKV 출력 정확성 회귀 risk.
- **총 작업 기간 +4-6주** (8~13주 → 12~19주).

---

## R3. Next actions (다음 행동)

### Phase α-W (Weight + PipelineStage 인프라 + Bundle 폐기, 2-3주)

**범위**:
1. **L2 trait 정의**: `engine/src/pipeline_stage.rs` (PipelineStage + LifecyclePhase + StageContext 3 field + StageOutcome + StageLifecycle) + `engine/src/pipeline_dispatcher.rs` + `engine/src/profiler.rs` + `engine/src/backend_extensions.rs`
2. **Weight 도메인 trait**: `engine/src/weight_layer.rs` (WeightLayer + LayerDispatch enum + SwapMetrics + WeightLayerView sub-trait) + LayerSlot thin wrap impl
3. **L4 PipelineRegistry**: `engine/src/session/pipeline_registry.rs` + dispatcher impl
4. **Stage impl (Weight + 일부)**:
   - EvictionStage (KV — Phase α-K 까지 KVCacheOps generic 유지로 placeholder)
   - SwapDispatchStage (Weight, WeightLayer 사용)
   - OneShotEvictStage / OneShotSwapStage / OneShotPartitionStage / OneShotLayerSkipStage / OneShotSwitchDeviceStage (5종)
5. **Entry point 마이그레이션**: argus_cli 또는 작은 entry point 1개
6. **PoC test 4건** (`arch/pipeline_stage_design.md` §11):
   - PipelineStage v0 + EvictionStage + SwapDispatchStage 정상 동작
   - S25 Qwen2.5-1.5B Q4_0 32 token bit-identical
   - S25 avg_tbt Δ ≤ +3%
   - Weight swap 정확성 회귀 0건

**제약 (본 grill 결정 적용)**:
- KvBundle / WeightBundle **신규 신설 금지** (이전 grill 결정 reverted)
- ctx 에 kv / weights field **추가 금지**
- Stage 가 layer handle (`Arc<dyn KVCacheLayer>` placeholder OK, 본격은 Phase α-K) 보관 패턴 사용

### ADR-0001 작성 (Phase α-W 종료 후, 2-3일)

**작업**:
1. `docs/adr/0001-kv-dispatch-paradigm.md` 완성 (현재 초안 완료, Phase α-W 결과 인용)
2. `engine/src/kv_cache_ops.rs:53` 주석에 ADR-0001 참조 추가 plan (구현은 Phase α-K)
3. Architect design review
4. Status: Accepted 확정

### Phase α-K (KV Generic → Trait object 전환, 4-6주)

**범위**:
1. **L2 KVCacheLayer trait 정의**: `engine/src/kv_layer.rs` (KVCacheLayer 5 method + KVCacheView sub-trait + StorageSpec trait)
2. **KVCacheOps 15 method → KVCacheLayer 5 method 매핑 표 작성** (R-G2 mitigation, Phase α-K 진입 전 필수)
3. **CacheManager / EvictionPolicy / D2OHandler / CachePressurePipeline 마이그레이션 plan** (R-G4 mitigation, Phase α-K 진입 전 필수)
4. **KVCacheLayer impl**: StandardLayer (KVCache wrap) + KIVILayer (KiviCache wrap) + paired backend attention kernel 매핑 검증
5. **Forward path generic → trait object 전환** (~20 file)
6. **Mixed storage test**: layer 0 Q4 + layer 1 F16 시나리오 PASS

**Phase α-K 종료 게이트** (ADR-0001 Validation gate, §6):
- S25 Qwen2.5-1.5B Q4_0 32 token bit-identical (모든 KV paradigm: Sliding / H2O / D2O / KIVI / SnapKV)
- Decode TBT Δ ≤ ±3%
- Mixed storage gate PASS
- R-G1~G5 모두 GREEN

---

## R4. Risk + escape hatch (리스크 + 후퇴 시점)

### RPN 매트릭스 (`arch/pipeline_stage_design.md` §9 참조)

| ID | 리스크 | RPN | 완화 |
|----|------|-----|------|
| R-K1 | mid-forward KV mutation → corruption | **270** | INV-DECODE-STAGE-001 spec + trait 주석 |
| R-V3-1 | 12~19주 sunk cost | 224 | Phase α-W (low risk) → α-K (high risk) 분리 + escape hatch |
| **R-G4** | **CacheManager 등 ~10 component KVCacheLayer 분해** | **168** | Phase α-K 진입 전 마이그레이션 plan 작성 |
| **R-G5** | **5 mutation primitive 통합으로 bit-identical 회귀** | **168** | Phase α-K 종료 게이트 (bit-identical 5 paradigm) |
| R-K2 | KIVI prefill mid-layer | 160 | OneShotKvQuantStage PreForward phase 제한 |
| **R-G1** | **KV dispatch hot path vtable cost** | **144** | Phase α-K PoC TBT Δ ≤ +3% 게이트 |
| R-V3-4 | inference_pipeline.md 전면 재작성 | 126 | v1 git history 보존 + 보존/철회 매트릭스 |
| R-V3-2 | code review 운영 부담 (3 field) | 112 (축소) | INV-DECODE-STAGE-006 + PR checklist |
| **R-G2** | KVCacheOps 15 method → KVCacheLayer 5 매핑 시 semantic loss | 105 | Phase α-K 진입 전 매핑 표 작성 |
| R-V3-3 | WeightLayer 추상화 폭 미확정 | 84 (축소) | Phase α-W 종료 spec test |
| **R-G3** | ADR 미작성 시 미래 회귀 시도 | 60 | ADR-0001 작성 (Status: Accepted) + kv_cache_ops.rs:53 주석 갱신 |

### Escape hatch (단계별 후퇴 시점)

| 시점 | 조건 | 후퇴 |
|------|------|------|
| Pre-α-1 종료 | sub-trait detail design 합의 실패 | v2 (b₁) 7-trait 후퇴 |
| Phase α-W 종료 | PoC 4 test 중 1+ fail | v2 (b₁) 7-trait — Weight 인프라 (LayerSlot/RCU) 는 본 grill 후퇴 후에도 그대로 유지 (sunk 0) |
| **ADR-0001 review** | **Architect / user reject** | **갈래 1 (Generic 유지 + Mixed storage 포기) 후퇴, Phase α-K 종료** |
| Phase α-K 진입 전 | KVCacheOps 매핑 표 / CacheManager 마이그레이션 plan 미완성 | Phase α-K 진입 지연 + design round 추가 |
| Phase α-K 종료 | bit-identical fail (any KV paradigm) | ADR-0001 revoke + 갈래 1 후퇴 |
| Phase α-K 종료 | Δ > +3% AND vtable root cause 확인 | ADR-0001 revoke + 갈래 1 후퇴 |
| Phase β 진입 후 | — | 후퇴 권장 안 함. 게이트 실패 시 stage impl 재설계로 해소 |

---

## R5. Open questions (미해결 결정점)

본 grill 에서 풀지 않았으나 Phase α-W / Phase α-K 진입 전 해결해야 한다.

### Phase α-W 진입 전

1. **sub-trait detail finalize** (`arch/pipeline_stage_design.md` §13.1 / §13.6):
   - **Q24-1**: `KVCacheView` (KVCacheLayer::view 반환 type) — 본 sub-grill 결정 #18 (dtype 폐기) 반영. Q-#1-3 (K/V raw read 노출) / Q-#1-4 (capacity 중복) / Q-#1-5 (mutation 누설) 미해결.
   - **Q24-2**: `WeightLayerView` (WeightLayer::view 반환 type) — Llama / Qwen / Mistral 흡수. dtype() 부재 정합 (결정 #18).
   - **Q24-3**: `SecondaryStore` (backlog [P2] `arch/weights_pressure_split.md §7.5` 확정, Phase α-W 와 같이 진행)
   - **Q24-4**: `SparsePattern` (stub or 별 sprint)
   - **~~Q24-5 (해소 본 sub-grill 2026-05-28~29)~~**: ~~`StorageSpec` + `WeightStorageSpec`~~ — **자연 폐기** (본 sub-grill 결정 #15 — 3-tier Stage 패턴 채택 → Tier 2 paradigm-specific Stage 가 concrete struct method 직접 호출로 흡수, spec object 패턴 불필요).

### Phase α-K 진입 전 (R-G2 + R-G4 mitigation)

2. **KVCacheOps 15 method → KVCacheLayer 5 method 매핑 표 작성**: `engine/src/kv_cache_ops.rs:55` 의 모든 method 가 (a) KVCacheLayer mutation primitive, (b) KVCacheView read-only method, (c) 폐기 (mechanism 안으로 흡수) 중 어디로 가는지 전수.
3. **CacheManager / EvictionPolicy / D2OHandler / CachePressurePipeline 마이그레이션 plan**: 각 component 가 KVCacheLayer trait object 모델에서 어떻게 분해 / 흡수 / 보존되는지 design round.

### Phase β 진입 전

4. **`KvOffloadStage` Persistent vs OneShot 선택** — **권장**: OneShot 우선, Persistent 는 Phase γ 이후 backlog.
5. **`OneShotQcfReportStage` phase 매핑** — Manager `RequestQcf` 명령. `PreEviction` 또는 `DecodeEnd` 후보.

### Phase β scope

6. **arch/inference_pipeline.md v2 재작성 범위** — Phase β scope, 별 sprint.

### post-grill review 2026-05-28 신규 (다이어그램 정정 + stages 위치 이동 후속)

본 grill 종결 후 사용자 추가 review 결과 식별된 후속 결정점. 모두 별 sub-grill round 로 분리.

7. **`BackendExtensions` trait 재설계 sub-grill** — 우선순위: **Phase α-W 종료 후, Phase α-K 진입 전** (ADR-0001 timing 과 같이):
   - 문제: `BackendExtensions::as_opencl_secondary()` 메소드가 trait 에 backend variant 이름을 박음 → **leaky abstraction (§13.8-O cross-L3 vocabulary trait inversion 위반 신호)**
   - 식별 경위: §2 다이어그램 정정 과정에서 이전 `bext --> be` (backend impl 을 의존하는 것처럼) 화살표 방향 오류가 본질 leaky abstraction 의 외화임을 확인
   - 갈래 (sub-grill 에서 결정):
     - (A) capability lookup pattern (`extensions::<dyn OpenClSecondary>()` generic dispatch)
     - (B) service locator (`lookup<T: Any>()` 런타임 type lookup)
     - (C) 현재 유지 + OCP 비용 정당화
   - 본 grill 에서 시그니처 재설계 X — Phase α-W 본 작업 (Weight infra) 와 직교, α-K 진입 전 sub-grill 로 분리

8. **`engine/src/stages/mod.rs` 신규 stage 추가 가이드 doc 작성** (Phase α-W architect detail):
   - 외부 기여자 진입점 — PipelineStage trait 학습 → layer handle 보관 패턴 → submit() 호출 시점
   - INV-DECODE-STAGE-001 (KV-PHASE) / INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC / INV-STAGE-LAYER-HANDLE 체크리스트
   - 어느 sub-directory (kv/ vs weight/ vs system/) 에 넣을지 분류 규약 (§5.4 표)
   - 작성 시점: Phase α-W stages/ 디렉토리 신설 commit 과 같은 commit

9. **`system/` 모듈 명명 검토** (Phase α-W architect detail):
   - 후보: `system/` vs `misc/` vs `dispatch/`
   - 결정 기준: backend / DecodeLoop 협업의 본질 표현 + 외부 기여자 직관성
   - 결정 시점: Phase α-W stages/ 디렉토리 신설 commit 전

### post-grill review 2026-05-28 후속 (결정 13/14 본 grill 누적, Architect 위임 2 차)

본 grill 후속 결정 13 (PipelineDispatcher trait 유지) + 결정 14 (BackendExtensions trait 폐기) 누적 결과 추가 sub-grill 3 건 등록. arch §13.5 참조.

10. ~~**`PipelineDispatcher` trait 위치 재검토** (별 sub-grill, Phase α-W detail)~~ — **결정 본 sub-grill 2026-05-28~29: L4 `engine/src/session/` finalize**:
    - 본 grill 후속 결정 13 에서 trait **유지** 확정 (deletion test PASS + INV-LAYER-006 강제 + mock 패턴 정합).
    - 본 sub-grill (2026-05-28~29) 에서 위치 **L4 `engine/src/session/`** finalize 확정. 근거:
      - §6.1 sequence 점검 — 모든 `dispatch()` 호출지가 L4 `DecodeLoop::run()` 또는 L4 `Forward::step()` 내부 (`PreLayer` / `PostLayer` phase) 안. L3 inference code 호출 없음 → L4 정합.
      - PipelineRegistry impl 위치 (L4 `session/pipeline_registry.rs`) 와 trait 위치 같은 sub-directory 로 통일 → discoverability + cohesion.
      - INV-LAYER-006 본문 갱신 (spec/41-invariants.md) — PipelineDispatcher trait 위치 = L4 명시.
    - 갈래 (rejected):
      - ~~(A) L2 유지~~ — abstraction primitive 정신 보존 명목이나 실제 사용 위치 L4 한정으로 정당화 약함
      - **(B) L4 `session/` 이동** — **CHOSEN**, 실제 사용 위치 기반 정합 + impl 과 같은 sub-directory

11. **Backend trait `as_opencl_secondary()` 명명 정합 sub-grill** (별 sub-grill, Phase α-W 종료 후 ~ Phase α-K 진입 전, ADR-0001 timing 권장):
    - 본 grill 후속 결정 14 에서 `BackendExtensions` trait 자체 폐기 확정. 진행 중 sprint 충돌:
      - `engine/src/backend.rs:1232-1255` Backend trait Stage 2 sprint 가 `as_opencl_secondary()` method default impl 추가 중 → 본 결정의 명명 정합 충돌 (variant 이름 박힘).
    - 갈래 (sub-grill 에서 결정):
      - (A) `secondary_store_handle()` 명명 정합 — backend variant 이름 (`opencl`) 제거
      - (B) cold-path `get_extension(EXT_SECONDARY_STORE)` 격하 — 확장 API 패턴
      - (C) sub-trait 분리 (`SecondaryStoreProvider`) — Backend trait 결합도 분산
    - §13.8-O cross-L3 vocabulary trait inversion 정신 정합 필수.
    - 우선순위 근거: Phase α-W (Weight infra) 와 직교, ADR-0001 timing 과 같이 sub-grill round 진행.

12. **`KVCacheLayer` / `WeightLayer` impl 의 backend ref 보유 패턴** (별 sub-grill, Phase α-W 진입 전 필수):
    - 본 grill 후속 결정 14 의 직접 결과 — Stage 가 backend capability 모름 → Layer impl 이 backend ref 보유 + capability 내부 호출.
    - 후보:
      - (a) Full `Arc<dyn Backend>` 보유 — layer impl 단순, vtable 1 lookup, 단 layer 가 Backend trait 전체 의식
      - (b) Capability sub-trait 별 보유 (예: `Arc<dyn KiviAttentionBackend>` + `Arc<dyn GpuScoreAccess>` 등 ISP-split) — ISP 강화, 단 layer impl 복잡 + sub-trait 정의 비용
    - 결정 방법: layer impl 별 capability 의존도 표 작성 후 결정. 예시:
      - `KIVILayer` = `KiviAttentionBackend` + `GpuScoreAccess`
      - `StandardLayer` = `GpuScoreAccess` 만
      - `OffloadLayer` = `SecondaryStore`
    - 결정 시점: Phase α-W KVCacheLayer / WeightLayer impl 시그니처 detail finalize 와 같은 sub-grill round.

### 본 sub-grill (2026-05-28~29) 신규 등록 sub-grill 2건

13. **Score domain refactor** (별 sprint, 본 sub-grill 결정 #17 직접 결과):
    - **Prerequisite**: #11 (Layer impl backend ref 보유 패턴 — backend capability 의존도 표) 선결.
    - Score 도메인 의 본질:
      - **Hot path** (collection 매 layer = backend inline) — F32 KV: backend kernel inline accumulation. Q4/F16 KV: separate compute_attention_scores pass.
      - **Cold path** (aggregation + read = PipelineStage on_phase) — eviction 시 score buffer read + sort + decide_keep.
      - **이 asymmetry 가 어색이 아니라 도메인 본질의 정직한 표현** — 통일 패턴 강제 안 함 (갈래 B Boundary 명시).
    - **추천 갈래** (sub-grill 에서 결정):
      - **갈래 4**: Generic capability lookup (Stage 가 `Arc<dyn ScoreReadable>` 보유, score buffer read 만 추상화)
      - **갈래 7**: Nested PipelineStage cost 명시 (score read 를 PostEviction phase 내부 sub-stage 로 분리)
      - **갈래 2**: Monomorphic input fallback (F32 vs Q4 별 concrete struct, dispatch 시점 분기)
    - **Pre-rejected** (자명히 부적합):
      - 갈래 1 (trait method overload) — semantic 모호
      - 갈래 3 (Decorator) — score buffer 가 owner-aware 가 아님
      - 갈래 5 (Event sourcing enum) — hot path frequency × cost 폭증
      - 갈래 8 (Closure) — RAII 패턴 위반
    - **Asymmetry note**: hot path 와 cold path 의 책임 분리는 통일 패턴 강제 시 god abstraction 재발생. 별 sprint 에서 결정.
    - 결정 시점: Phase α-K 종료 후 또는 별 sprint round.

14. **Multiple EvictionStage 시나리오** (별 sprint, #13 와 묶음):
    - 본 sub-grill 결정 #17 의 직접 결과 — score domain refactor 시 multiple EvictionStage (예: layer 0~5 sliding + layer 6~15 H2O 등) 시나리오 finalize.
    - 시나리오:
      - (a) Per-layer-group EvictionStage — `Vec<EvictionStage>` (각 stage 가 cardinality K layer 보유)
      - (b) Single EvictionStage + policy table — `EvictionStage { policy_per_layer: Vec<Box<dyn EvictionPolicy>> }`
      - (c) PipelineRegistry 가 layer group 별 dispatcher 분리
    - cardinality 자유 (결정 #16) 기반에서 자연 표현 가능 — 단 score_accumulator ownership 패턴 finalize 필요 (#13 결과 의존).
    - 결정 시점: Phase α-K 종료 후 또는 #13 와 같은 sub-grill round.

### 본 grill 에서 해결된 open questions (이전 grill 미해결 → 본 grill 결정)

- ~~`score_accumulator` ownership~~ — 본 grill #8 (Stage register 시점 layer handle 보관) 후: `KVCacheLayer::view()::score_handle()` 으로 layer 내부 흡수 → 외부 ownership 패턴 불필요. **본 sub-grill 2026-05-28~29 결정 #17 추가**: EvictionHook → EvictionStage 1:1 wrap + `score_accumulator: Option<AttentionScoreAccumulator>` concrete type 보존 (pragmatic deferral). 별 sprint refactor (#13).

---

## R6. References (참조)

### 본 grill 산출물

- `arch/pipeline_stage_design.md` — 메인 진실원본 (16 절, 본 grill 14 결정 반영 (본문 12 + post-grill 후속 2), Q1~Q35 매트릭스). **post-grill review 2026-05-28** 1차: §2 다이어그램 화살표 정정 + concrete stages 위치 L4 session/ → L3 cross-cutting `engine/src/stages/{kv,weight,system}/` 이동 + §5.4 sub-structure 신설 + §13.4 후속 결정점 4건. **post-grill review 2026-05-28** 2차: §3.3 ctx 3 → 2 field (backend_ext 폐기) + §3.4 PipelineDispatcher trait 유지 명문화 (§3.4.1 추가) + §3.7 제목 변경 ("Profiler (L2)") + §3.7.1 BackendExtensions trait 폐기 sub-section + §5.2 EvictionStage 예시 backend_ext 사용 금지 + KIVI 예시 교정 + §6 DecodeLoop backend_ext field 삭제 + `pipeline: Arc<dyn PipelineDispatcher>` 변경 + §2 Mermaid 다이어그램 `bext` 노드 + 관련 화살표 삭제 + §10 Phase α-W 게이트 갱신 + §13.5 후속 결정점 3건 (R5 #10/#11/#12) + §16 누적 결정 12 → 14 건 + §16.3 변경 추적성 2 row 추가. **본 sub-grill 2026-05-28~29 (3차, 본 위임)**: **§0 Executive Overview 신설** (외부 explorer 진입점, standalone 으로 본 sprint 전체 그림 파악 가능) + §3.5 KVCacheLayer 시그니처 갱신 (apply_storage / as_any / dtype 폐기, 5 → 3 mutation primitive) + §3.6 WeightLayer 동일 갱신 (apply_storage 폐기, 4 → 3 method) + §5.2 Stage 패턴 3-tier 분리 (Tier 1/2/3 예시) + §5.3 cardinality + tier 컬럼 추가 + §5.4 3-tier 패턴 가이드 + §13.1 Q24-5 (StorageSpec) 해소 + §13.6 본 sub-grill 매트릭스 신설 (결정 4 + 갈래 B + 미해결 9건 + 발견 모순 10건 + llm.npu / mllm-NPU ASPLOS 2025 정합도 부록) + §16 누적 결정 14 → 18 + §16.3 변경 추적성 4 row 추가.
- `docs/adr/0001-kv-dispatch-paradigm.md` — KV dispatch Generic → Trait object 정식 결정 (Status: Accepted).
- `docs/adr/README.md` — ADR 디렉토리 신설.
- `spec/41-invariants.md` §3.28 — INV 표 갱신 (INV-DECODE-STAGE-002/003 폐기 + 신규 3건 추가). **post-grill review 2026-05-28** 1차: INV-STAGE-MODULE-LOCATION 후보 등록 — 즉시 추가 X, Phase α-W 진입 commit 에서 추가 (R5 #4 + arch §13.4). **post-grill review 2026-05-28** 2차 (본 위임): INV-DECODE-STAGE-006 (CTX-AUTHORITY) 본문 갱신 — 3 field → 2 field (backend_ext 폐기) + §3.28 변경 요약에 본 grill 후속 결정 14 1줄 추가.
- `arch/README.md` — cross-reference 갱신.
- `arch/inference_pipeline.md` v1 — deprecation notice 정련 (v3 본 grill 결정 인용).

### 이전 handoff supersede

- `.agent/todos/handoff_pipeline_stage_design_2026_05_27.md` — 이전 grill (2026-05-27) Phase α 진입 handoff. **본 grill 2026-05-28 의 결정 12 건으로 supersede**:
  - 이전 단일 Phase α (2-3주) → 본 grill Phase α-W (2-3주) + ADR-0001 + Phase α-K (4-6주)
  - 이전 KvBundle 8 method / WeightBundle 10 method 시그니처 → 본 grill 폐기 + KVCacheLayer (5 method) / WeightLayer (4 method)
  - 이전 god ctx 5 field 인정 → 본 grill 3 field 축소
  - 이전 Pre-α-2 PoC scope → 본 grill Phase α-W PoC scope (Weight swap 정확성 추가)
  - 이전 Q24 sub-trait 4종 (KVCacheView / LayerView / SecondaryStore / SparsePattern) → 본 grill Q24 5종 (Q24-5 StorageSpec 추가, LayerView → WeightLayerView 재명명)

### 본 grill 선행 문서

- `arch/inference_pipeline.md` v1 — Phase 4-2/4-3/4-4-2.3 7-trait. Phase β 에서 v2 재작성.
- `ARCHITECTURE.md` §13.8 — INV-LAYER 시리즈 + §13.8-O cross-L3 vocabulary trait inversion.
- `arch/weights_pressure_split.md §7.5` — `SecondaryStore` trait inversion (Phase α-W 본격 작업).

### 본 grill 핵심 코드 참조

- `engine/src/kv_cache_ops.rs:53` — Generic monomorphization 명시 정책 (본 grill 에서 ADR-0001 로 반전).
- `engine/src/kv_cache_ops.rs:55` — KVCacheOps trait 정의 (15 method, Phase α-K 매핑 대상).
- `engine/src/models/weights/slot.rs:158` — `LayerSlot::rcu_weights` 패턴 (본 grill (γ) interior mutability 자연 확장 근거).

### 관련 memory 항목

- `project_layered_architecture_decision.md` — L1~L5 layer + cross-cutting 합의.
- `project_eurosys2027_post_paper.md` — paper deadline 만료, 리팩토링이 default 메인 트랙.
- `feedback_arch_component_centric.md` — arch/ 문서는 컴포넌트 중심.
- `feedback_mermaid_diagrams.md` — Mermaid 사용 규칙.
- `feedback_spec_tests_required.md` — spec ID 관련 작업 시 `tests/spec/` 테스트 필수.
- `feedback_tbt_metric_tok0_inclusive.md` — TBT 측정은 tok0 inclusive.

### 관련 commit

- Phase 4-2 `584496b7` — 7-trait + Builder + defaults DONE (v1).
- Phase 4-3 `c63190d1` — ModelForward + probe microbench DONE.
- Phase 4-4-2.3 a/c/b `9313670b` + `bcb221e2` + `02cb7106` — decode_fallback 추출 DONE.
- Pipeline stage design v3 (2026-05-27) `6f07af8d` — pipeline_stage_design.md 초안 commit.

---

## 자기점검 (handoff-doc 스킬)

| 점검 항목 | 확인 |
|----------|------|
| 진입 문장 명시 | "Pipeline stage Phase α-W 진입 — Weight + PipelineStage 인프라 + Bundle 폐기" |
| 선행 문서 link | `arch/pipeline_stage_design.md`, `docs/adr/0001-kv-dispatch-paradigm.md`, `spec/41-invariants.md` §3.28 |
| 이전 handoff supersede 명시 | R6 (`handoff_pipeline_stage_design_2026_05_27.md` 변경 매트릭스 5건 매핑) |
| 미해결 결정점 명시 | R5 (Phase α-W 진입 전 Q24-1~5 / Phase α-K 진입 전 매핑 표 + 마이그레이션 plan / Phase β scope) |
| 다음 행동 검증 가능 | Phase α-W PoC 4 test PASS / ADR-0001 review PASS / Phase α-K 종료 게이트 (bit-identical + Δ ≤ +3% + Mixed storage) |
| Escape hatch 명시 | R4 (단계별 후퇴 7 시점) |
| 리스크 RPN 명시 | R4 (RPN ≥ 100 항목 11건 매트릭스, R-G1~G5 본 grill 신규) |
| 본 grill 핵심 결정 12 건 명시 | R1 (전체 표) |
| Spec INV 변경 명시 | R1 (폐기 2건 + 신규 3건 + 수정 2건 매트릭스) |
| Sprint 분리 명시 | R1 (Phase α-W 2-3주 → ADR-0001 → Phase α-K 4-6주, 총 12-19주) |
| post-grill review 2026-05-28 추가 (1차) | R5 #7~#9 (BackendExtensions 재설계 sub-grill + stages/mod.rs 가이드 doc + system/ 명명 검토) + R6 (arch/pipeline_stage_design.md §2 다이어그램 정정 + §5.4 sub-structure + §13.4 결정점 4건) |
| post-grill review 2026-05-28 추가 (2차, 본 위임) | R1 결정 13/14 추가 (PipelineDispatcher trait 유지 + BackendExtensions trait 폐기) + R5 #10~#12 (PipelineDispatcher 위치 + Backend trait 명명 정합 + Layer impl backend ref 보유 패턴) + R6 본 위임 arch/spec/handoff 변경 내역 명시 + ctx 3 → 2 field 본문 갱신 |
| **본 sub-grill 2026-05-28~29 추가 (3차, 본 위임)** | R1 결정 15~18 추가 (3-tier Stage 패턴 + cardinality 자유 + Score 도메인 별 sprint + KVCacheView::dtype() 폐기) + 메타 결정 갈래 B (Boundary 명시) + R5 #5 StorageSpec 해소 + #10 PipelineDispatcher 위치 결정 (L4 finalize) + #13 Score domain refactor 신규 + #14 Multiple EvictionStage 신규 + R6 본 위임 arch/spec/README/handoff 변경 내역 명시 + §0 Executive Overview 신설 + llm.npu / mllm-NPU 정합도 평가 |
