# Handoff — DecodeLoop v3 (PipelineStage Hook Pattern) Phase α 진입

> **일자**: 2026-05-27
> **작성자**: Architect (orchestrator 보고용)
> **진입 문장**: "Pipeline stage Phase α 진입 — Pre-α-1 design round (Q24 sub-trait 4종) 시작"
> **선행 문서**: `arch/pipeline_stage_design.md` (본 sprint 단일 진실원본, 23 라운드 grill 결정 반영)
> **선행 spec**: `spec/41-invariants.md` §3.28 (INV-DECODE-STAGE-001~007)
>
> ---
>
> ## **⚠ 본 handoff 는 2026-05-28 KvBundle/WeightBundle grill 결정으로 SUPERSEDED**
>
> **후속 handoff**: `.agent/todos/handoff_kv_weight_grill_2026_05_28.md` — 본 grill 12 결정 반영 (KvBundle/WeightBundle trait 폐기 + KVCacheLayer/WeightLayer trait + ctx 5→3 field + KV dispatch Generic → Trait object + Sprint 분리 Phase α-W → ADR-0001 → Phase α-K).
>
> **본 handoff 의 다음 결정이 supersede 됨**:
> | 본 handoff (2026-05-27) | 2026-05-28 본 grill 변경 |
> |---|---|
> | 단일 Phase α (WeightBundle prerequisite, 2-3주) | **Phase α-W (Weight + PipelineStage 인프라, 2-3주) → ADR-0001 → Phase α-K (KV refactor 4-6주)** |
> | KvBundle 8 method 시그니처 + WeightBundle 10 method 시그니처 | **KvBundle / WeightBundle trait 폐기 → KVCacheLayer (5 method) / WeightLayer (4 method) + SwapMetrics 별 trait** |
> | god ctx 5 field 인정 (kv / weights 포함) | **3 field 축소 (step / backend_ext / profiler)** — kv / weights 는 Stage 가 register 시점 layer handle 보관 |
> | Pre-α-2 PoC scope (KIVI per-layer mix 포함) | **Phase α-W PoC scope (Weight swap 정확성 회귀 0건 포함)** — KIVI 시나리오는 Phase α-K 로 이동 |
> | Q24 sub-trait 4종 (KVCacheView / LayerView / SecondaryStore / SparsePattern) | **Q24 5종 (Q24-5 StorageSpec 추가, LayerView → WeightLayerView 재명명)** |
> | 총 작업 기간 8~13주 | **총 12~19주** (KV refactor risk 분리로 +4-6주) |
>
> **본 handoff 의 보존된 결정** (2026-05-28 후에도 유효):
> - 단일 PipelineStage trait + LifecyclePhase enum (P3 21 variant + P4 feature-gated) + PipelineRegistry 패턴
> - W-1 Forward / TokenSampler 별 trait 유지
> - Manager IPC 위치 = DecodeLoop owned (stage 외부)
> - error handling = panic on Err
> - registry = `Arc<PipelineRegistry>` + `Mutex<Vec<Arc<dyn PipelineStage>>>` interior mutability
> - OneShot lifecycle = `StageLifecycle::OneShot` + `StageOutcome::Consumed` + dispatcher 자동 GC
>
> **INV 변경**:
> - **폐기**: INV-DECODE-STAGE-002 (KVBUNDLE-CONSISTENCY) — KvBundle trait 폐기로 자연 해소
> - **폐기**: INV-DECODE-STAGE-003 (KVBUNDLE-SYNC) — (β) sync 모델 자동 처리
> - **신규**: INV-KVCACHELAYER-PRIMITIVE-AGNOSTIC / INV-KVCACHELAYER-PAIRED-KERNEL / INV-STAGE-LAYER-HANDLE
> - **수정**: INV-DECODE-STAGE-001 (ctx.kv → Stage 보유 layer handle) / INV-DECODE-STAGE-006 (5 field → 3 field)
>
> 후속 handoff 의 R6 (References) 가 본 handoff 와의 supersede 매트릭스 5건을 상세 기술. **다음 세션 진입 문장은 후속 handoff 의 진입 문장 "Pipeline stage Phase α-W 진입" 을 사용**.

---

## R1. What was decided (결정사항 요약)

### 핵심 결정
- DecodeLoop v3 = **단일 `PipelineStage` trait + `LifecyclePhase` enum (P3 21 variant + P4 feature-gated) + entry point별 `PipelineRegistry`** 패턴.
- v2 7-trait (`Forward` / `EvictionStage` / `SwapStage` / `CommandSource` / `TokenSampler` / `DecodeObserver` / `ResilienceStage` 후보) 중 `Forward` + `TokenSampler` 만 보존 (W-1), 나머지 5 trait 폐기.
- god ctx 인정 (`StageContext` 5 field: `step` / `kv` / `weights` / `backend_ext` / `profiler`) + 권한 강제는 code review (M1만).
- Manager IPC 위치 = DecodeLoop owned (stage 외부) — `CommandExecutor::poll()` + heartbeat outbound 모두 DecodeLoop 본체.
- error handling = **panic on Err** (partial commit 없음).
- registry = `Arc<PipelineRegistry>` + `Mutex<Vec<Arc<dyn PipelineStage>>>` interior mutability (M-γ).
- OneShot lifecycle = `StageLifecycle::OneShot` + `StageOutcome::Consumed` + dispatcher 자동 GC.

### 신규 INV 7건 (INV-DECODE-STAGE-001~007)
| ID | 한줄 요약 |
|----|---------|
| INV-DECODE-STAGE-001 | KV mutation 허용 phase 제약 (PreLayer/PostLayer/Fine(*) 금지) |
| INV-DECODE-STAGE-002 | KvBundle method scope 명시 (layer-wide vs per-layer) |
| INV-DECODE-STAGE-003 | KvBundle mutation method 호출 후 GPU sync 보장 |
| INV-DECODE-STAGE-004 | StageOutcome 3 variant 처리 (Continue/Stop/Consumed) |
| INV-DECODE-STAGE-005 | Stage 등록 순서 = dispatch 순서 (caller 책임) |
| INV-DECODE-STAGE-006 | god ctx 권한 강제 X — code review 책임 |
| INV-DECODE-STAGE-007 | OneShot lifecycle + dispatcher 자동 GC |

### 작업 분해 (Phase α/β/γ)
| Phase | 기간 | 게이트 |
|---|---|---|
| Pre-α-1 | 2~3일 | design round PASS (Q24 sub-trait 4종 finalize) |
| Pre-α-2 | 1주 | PoC 4 test PASS |
| Phase α | 2~3주 | WeightBundle prerequisite + spec test |
| Phase β | 3~4주 | DecodeLoop 재작성 + S25 bit-identical + TBT Δ ≤ +3% + INV 7건 PASS |
| Phase γ | 3~4주 | legacy generate.rs 잔여 마이그레이션 + PACT2026 PoC |

**총 8~13주**.

---

## R2. Why this matters (배경 + 의미)

### v2 7-trait 의 잔여 문제
1. **R-2** — `EvictionStage` 시그니처 부족 → pressure handler 다양화 (D2O / KIVI / SnapKV / KvOffload / Sparse) 흡수 위한 책임 재분배 깊이 미확정.
2. **R-5** — `ResilienceStage` 신설 결정점 미해결 (Manager IPC ResilienceAction 흡수처).
3. **God Ctx** — `decode_fallback/{eviction_trigger,swap_dispatch}.rs` 시그니처 12/21 필드 — 추출했지만 책임 분리 미달.
4. **IPC 격차** — `arch/inference_pipeline.md:177` 명시: `cmd_source.poll()` 결과 accept-and-drop, ExecutionPlan 적용 0 LOC.
5. **신규 비용** — speculative decoding / per-op measurement / KIVI per-layer / SnapKV / Sparse 등 추가마다 새 trait 또는 기존 trait의 신규 method, 양쪽 모두 OCP 위반.

### v3 hook 패턴이 해결하는 것
- **SRP/OCP 양립** — `PipelineStage::on_phase` 1 method, 차이는 phase enum + stage 구현체로 분기.
- **신규 책임 추가 비용** = 새 stage struct + 새 entry point에서 `submit()` 1줄.
- **Manager IPC 명확화** — DecodeLoop 본체 책임으로 격리, OneShot stage 변환 후 `registry.submit()`.

### 인정 비용
- god ctx (5 field) 명시 수용 + 권한 강제는 code review (M1만) — compile-time 강제 시 sub-trait 보유 비용이 과대.
- PR checklist 의무화로 보강 (INV-DECODE-STAGE-006).

---

## R3. Next actions (다음 행동)

### Pre-α-1 (별 design round) — Q24 sub-trait 4종 finalize

다음 4 sub-trait 시그니처를 별 grill 라운드에서 풀어야 한다. 본 grill 에서는 풀지 않았다.

| sub-trait | 위치 | 작업 |
|-----------|------|------|
| `KVCacheView` | L2 | `engine/src/kv_cache_ops.rs` (B-5b-1b `45bfd16f`) 현 method set 확인 + 추가 method 필요성 + `KvBundle::layer_view` 반환 type 확정 |
| `LayerView` | L2 | Llama / Qwen / Mistral 흡수 가능한 시그니처 검토. Generic `weight_tensor(name)` vs specific method 9개 |
| `SecondaryStore` | L2 | backlog [P2] `arch/weights_pressure_split.md §7.5` 확정. mmap-based vs file-based vs streaming. AUF integration. **Phase α 본격 작업과 같이 진행** |
| `SparsePattern` | L2 (stub or 별 sprint) | 본 sprint scope 결정 |

**기타 미해결 결정점**:
1. `KvOffloadStage` 정책 — Persistent (pressure 자동) vs OneShot (Manager 명령). **권장**: OneShot 우선, Persistent는 후속 backlog.
2. `score_accumulator` ownership — `Arc<Mutex<>>` 외부 owned + `EvictionPolicyStage` + `OneShotQcfReportStage` 양쪽 inject 패턴 finalize.
3. `OneShotQcfReportStage` phase — Manager `RequestQcf` 명령이 어느 phase 에서 처리? (PreCommandPoll 폐기됐으니 `PreEviction` 등으로 매핑 필요)

### Pre-α-2 (PoC) — 4 test 게이트

```
1. PipelineStage v0 + EvictionPolicyStage 1개 — 정상 동작 확인
2. S25 Qwen2.5-1.5B Q4_0 32토큰 bit-identical (baseline vs PoC)
3. S25 microbench — stage N=7~8 × phase M=21 (P3) × 32토큰 TBT
   임계: avg_tbt Δ ≤ +3% (INV-147 noise 기준 — tok0 inclusive)
4. KIVI per-layer mix (5 layer만 Q4 dynamic) — TBT regression < 5%
```

**측정 방법**:
- (2): `python scripts/run_device.py -d s25 generate --backend opencl --opencl-rpcmem --model-path qwen2.5-1.5b-q40.auf --prompt "..." --max-tokens 32 --seed 42`. baseline (HEAD `master`) vs PoC branch token id sequence 비교.
- (3): tok0 inclusive avg_tbt (`feedback_tbt_metric_tok0_inclusive.md`). `n ≥ 5, median`.
- (4): `OneShotKvQuantStage` × 5 layer submit. PreForward phase 처리.

---

## R4. Risk + escape hatch (리스크 + 후퇴 시점)

### RPN 매트릭스 (요약)
| ID | 리스크 | RPN |
|----|------|-----|
| R-K1 | mid-forward KV mutation → corruption | **270** |
| R-V3-1 | 8~13주 sunk cost | 224 |
| R-V3-2 | god ctx code review 운영 부담 | 168 |
| R-K2 | KIVI prefill mid-layer | 160 |
| R-V3-3 | WeightBundle 추상화 폭 미확정 | 140 |
| R-V3-4 | inference_pipeline.md 전면 재작성 | 126 |
| R-K4 | GPU sync | 120 |
| R-V3-5 | hot path vtable N × M dispatch | 108 |
| R-K3 | layer-wide vs per-layer 일관성 | 105 |
| R-K5 | Recall position drift | 72 |

상세 + 완화책은 `arch/pipeline_stage_design.md` §9 참조.

### Escape hatch (v2 (b₁) 7-trait 후퇴 시점)
| 시점 | 조건 | 후퇴 |
|------|------|------|
| Pre-α-1 종료 | design round 합의 실패 | v2 (b₁) 7-trait |
| Pre-α-2 PoC 종료 | 4 test 중 1+ fail | v2 (b₁) 7-trait — WeightBundle 인프라는 (b₁)에서도 활용 (sunk 0) |
| Phase α 종료 | WeightBundle 추상화 폭 미해결 | v2 (b₁) + WeightBundle 별 sprint 자산 |
| Phase β 진입 후 | — | 후퇴 권장 안 함. 게이트 실패 시 stage impl 재설계로 해소 |

---

## R5. Open questions (미해결 결정점)

본 grill 에서 풀지 않았으나 Pre-α-1 또는 Phase α 진입 전 해결해야 한다.

1. **Q24-1 ~ Q24-4** (sub-trait 시그니처 4종) — Pre-α-1 design round.
2. **`KvOffloadStage` Persistent vs OneShot 선택** — Phase β 진입 전.
3. **`score_accumulator` ownership 패턴** — Phase β 진입 전.
4. **`OneShotQcfReportStage` phase 매핑** — Phase β 진입 전.
5. **arch/inference_pipeline.md v2 재작성 범위** — Phase β scope. 본 sprint 외 별 sprint.

---

## R6. References (참조)

### 본 sprint 산출물
- `arch/pipeline_stage_design.md` — 메인 진실원본 (15 절 + 결정 매트릭스 Q1~Q23).
- `spec/41-invariants.md` §3.28 (INV-DECODE-STAGE-001~007).

### 선행 문서
- `arch/inference_pipeline.md` v1 — Phase 4-2/4-3/4-4-2.3 7-trait 설계. Phase β 시점에 v2로 재작성.
- `ARCHITECTURE.md` §13.8 — INV-LAYER 시리즈 + §13.8-O cross-L3 vocabulary trait inversion.
- `arch/weights_pressure_split.md §7.5` — `SecondaryStore` trait inversion (Q24-3, Phase α 본격 작업).

### 관련 memory 항목
- `project_layered_architecture_decision.md` — L1~L5 layer + cross-cutting 합의.
- `project_eurosys2027_post_paper.md` — paper deadline 만료, 리팩토링이 default 메인 트랙.
- `feedback_arch_component_centric.md` — arch/ 문서는 컴포넌트 중심.
- `feedback_mermaid_diagrams.md` — Mermaid 사용 규칙.
- `feedback_spec_tests_required.md` — spec ID 관련 작업 시 `tests/spec/` 테스트 필수.
- `feedback_tbt_metric_tok0_inclusive.md` — TBT 측정은 tok0 inclusive.

### 관련 commit
- Phase 4-2 `584496b7` — 7-trait + Builder + defaults DONE.
- Phase 4-3 `c63190d1` — ModelForward + probe microbench DONE.
- Phase 4-4-2.3 a/c/b `9313670b` + `bcb221e2` + `02cb7106` — decode_fallback 추출 DONE.

---

## 자기점검 (handoff-doc 스킬)

| 점검 항목 | 확인 |
|----------|------|
| 진입 문장 명시 | "Pipeline stage Phase α 진입 — Pre-α-1 design round (Q24 sub-trait 4종) 시작" |
| 선행 문서 link | `arch/pipeline_stage_design.md`, `spec/41-invariants.md` §3.28 |
| 미해결 결정점 명시 | R5 (Q24-1~4, KvOffloadStage 정책, score_accumulator ownership, OneShotQcfReportStage phase, arch v2 재작성 범위) |
| 다음 행동 검증 가능 | Pre-α-1 design round → design-review PASS / Pre-α-2 PoC → 4 test PASS |
| Escape hatch 명시 | R4 (v2 7-trait 후퇴 시점 4 단계) |
| 리스크 RPN 명시 | R4 (RPN ≥ 100 항목 10건 매트릭스) |

