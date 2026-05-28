# Handoff — KvBundle/WeightBundle Grill 종결 → Phase α-W 진입

> **일자**: 2026-05-28
> **작성자**: Architect (orchestrator 보고용)
> **진입 문장**: **"Pipeline stage Phase α-W 진입 — Weight + PipelineStage 인프라 + Bundle 폐기"**
> **선행 문서**:
> - `arch/pipeline_stage_design.md` (본 sprint 단일 진실원본, 2026-05-27 23 라운드 grill + **2026-05-28 본 grill 12 결정** 반영)
> - `docs/adr/0001-kv-dispatch-paradigm.md` (KV dispatch Generic → Trait object 정식 결정)
> - `spec/41-invariants.md` §3.28 (INV-DECODE-STAGE-001/004~007 + INV-KVCACHELAYER-* + INV-STAGE-LAYER-HANDLE; INV-DECODE-STAGE-002/003 폐기)
> **이전 handoff**: `.agent/todos/handoff_pipeline_stage_design_2026_05_27.md` — 본 grill 결정으로 supersede (Phase α 분리 등). 이전 handoff 의 일부 결정사항은 본 grill 에서 변경됨, 변경 매트릭스는 R6 참조.

---

## R1. What was decided (결정사항 요약 — 본 grill 12 결정)

### 본 grill 누적 결정 12 건

| # | 결정 | 의미 |
|---|---|---|
| 1 | **(β) sync model** | Buffer-level lazy + access-mode-aware. R1/R2/R3 자동. 새 sync 인프라 0건. INV-KVBUNDLE-SYNC 폐기. |
| 2 | **(d-1) primitive only — storage-format-agnostic** | KVCacheLayer method 가 token/layer/range granularity 만 알고 storage paradigm 모름. KVCacheLayer impl 이 mechanism 캡슐화. |
| 3 | **Q7 (A) 유지** (AttentionKernels sub-trait 분리 (A') 보류) | 본 프로젝트 KiviAttentionBackend / GpuScoreAccess 패턴 + frequency × cost 정당화 부족. |
| 4 | **Q8 Mixed storage 허용** | Layer 별 다른 storage paradigm OK (layer 0 = KIVI Q4, layer 1 = TurboQuant, ...). |
| 5 | **Q9 호환성 차단 인프라 X** | 만나면 panic. |
| 6 | **KvBundle / WeightBundle trait 폐기** | Stage register 시점 layer handle 보관 → ctx layer field 자체 폐기. |
| 7 | **StageContext 5 field → 3 field** | `kv` / `weights` 폐기. `step` / `backend_ext` / `profiler` 유지. |
| 8 | **(γ) Layer handle 전환** | KVCacheLayer / WeightLayer trait + interior mutability. Stage 가 `Arc<dyn ...>` 보관. LayerSlot::rcu_weights 자연 확장. |
| 9 | **KV dispatch 갈래 2 Trait object 전면 채택** | kv_cache_ops.rs:53 정책 정면 반전. **ADR-0001 작성 필수**. 갈래 1/3/4 REJECTED. |
| 10 | **Weight dispatch: LayerSlot + WeightLayer thin wrap** | Forward path 무변경. ~5 file 추가. KV 와 비대칭. |
| 11 | **Sprint 분리: Phase α-W → ADR-0001 → Phase α-K** | Risk 분산 + escape hatch. |
| 12 | **LayerDispatch enum: Fixed 3 variant (Full / Skip / Partition)** | enum + match exhaustive 가독성 우월. |

### Spec INV 변경 (본 grill 결정 → 카탈로그 반영)

| 변화 | INV | 사유 |
|---|---|---|
| **폐기** | INV-DECODE-STAGE-002 (KVBUNDLE-CONSISTENCY) | KvBundle trait 자체 폐기로 자연 해소. layer-wide vs per-layer 구분 사라짐. |
| **폐기** | INV-DECODE-STAGE-003 (KVBUNDLE-SYNC) | (β) sync 모델 채택. R1/R2/R3 자동 처리 흡수. |
| **신규** | INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC | KVCacheLayer mutation method 는 storage-format-agnostic. Stage 가 dtype/codebook/rotation 모름. |
| **신규** | INV-KVCACHELAYER-PAIRED-KERNEL | KVCacheLayer impl 과 paired backend attention kernel 매핑 의무. |
| **신규** | INV-STAGE-LAYER-HANDLE | PipelineStage 가 layer handle 을 register 시점 보관. ctx 에 kv/weights field 두지 않음. |
| **수정** | INV-DECODE-STAGE-001 (KV-PHASE) | mutation method 표현이 ctx.kv → Stage 보유 KVCacheLayer handle 로 변경. 정신 유지. |
| **수정** | INV-DECODE-STAGE-006 (CTX-AUTHORITY) | 5 field → 3 field. kv/weights 책임 경계 위반 risk 자체 사라짐. |

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

1. **sub-trait detail finalize** (`arch/pipeline_stage_design.md` §13.1):
   - **Q24-1**: `KVCacheView` (KVCacheLayer::view 반환 type)
   - **Q24-2**: `WeightLayerView` (WeightLayer::view 반환 type) — Llama / Qwen / Mistral 흡수
   - **Q24-3**: `SecondaryStore` (backlog [P2] `arch/weights_pressure_split.md §7.5` 확정, Phase α-W 와 같이 진행)
   - **Q24-4**: `SparsePattern` (stub or 별 sprint)
   - **Q24-5 (신규)**: `StorageSpec` + `WeightStorageSpec` (apply_storage 흡수 spec object 시그니처)

### Phase α-K 진입 전 (R-G2 + R-G4 mitigation)

2. **KVCacheOps 15 method → KVCacheLayer 5 method 매핑 표 작성**: `engine/src/kv_cache_ops.rs:55` 의 모든 method 가 (a) KVCacheLayer mutation primitive, (b) KVCacheView read-only method, (c) 폐기 (mechanism 안으로 흡수) 중 어디로 가는지 전수.
3. **CacheManager / EvictionPolicy / D2OHandler / CachePressurePipeline 마이그레이션 plan**: 각 component 가 KVCacheLayer trait object 모델에서 어떻게 분해 / 흡수 / 보존되는지 design round.

### Phase β 진입 전

4. **`KvOffloadStage` Persistent vs OneShot 선택** — **권장**: OneShot 우선, Persistent 는 Phase γ 이후 backlog.
5. **`OneShotQcfReportStage` phase 매핑** — Manager `RequestQcf` 명령. `PreEviction` 또는 `DecodeEnd` 후보.

### Phase β scope

6. **arch/inference_pipeline.md v2 재작성 범위** — Phase β scope, 별 sprint.

### 본 grill 에서 해결된 open questions (이전 grill 미해결 → 본 grill 결정)

- ~~`score_accumulator` ownership~~ — 본 grill #8 (Stage register 시점 layer handle 보관) 후: `KVCacheLayer::view()::score_handle()` 으로 layer 내부 흡수 → 외부 ownership 패턴 불필요.

---

## R6. References (참조)

### 본 grill 산출물

- `arch/pipeline_stage_design.md` — 메인 진실원본 (16 절, 본 grill 12 결정 반영, Q1~Q35 매트릭스).
- `docs/adr/0001-kv-dispatch-paradigm.md` — KV dispatch Generic → Trait object 정식 결정 (Status: Accepted).
- `docs/adr/README.md` — ADR 디렉토리 신설.
- `spec/41-invariants.md` §3.28 — INV 표 갱신 (INV-DECODE-STAGE-002/003 폐기 + 신규 3건 추가).
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
