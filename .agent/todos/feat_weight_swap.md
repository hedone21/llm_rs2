# Weight Swap Feature — Runtime Dynamic Layer Weight Swap (Manager-Driven)

> **목표**: 메모리 극한 Android 환경에서 manager signal 기반으로 layer 단위 weight를 **런타임 동적** 교체. F16 → Q4_0 단방향 swap. 평시 제로 오버헤드.
> **스코프 확정 (2026-04-24)**: 정적 프로파일 기반(기존 Phase A) **전면 폐기**. 동적 swap 단독 노선.
> **신호 모델**: `ResilienceAction::SwapWeights { ratio: f32 }` — 전체 layer 대비 Q4_0 전환 비율.
> **QCF 측정**: Prefill 종료 직후 1회. Decode 중 신호 수신 시 다음 prefill까지 대기 (fallback은 Architect 결정).
> **타겟 디바이스**: Galaxy S25 (Snapdragon 8 Elite, 12GB RAM).
> **타겟 모델**: Llama 3.2 1B (우선).
> **작성일**: 2026-04-24 (초안) / 2026-04-24 (동적 전환 재작성).
> **연관 브랜치**: `feat/weight`.
>
> **이전 초안(정적 프로파일 기반) 폐기 사유**: 사용자 의도는 manager 신호에 의한 런타임 동적 swap. TOML 프로파일/`quantize_profile` 바이너리/`--layer-dtype-profile` 플래그는 요구사항과 불일치. 관련 작업 모두 삭제하고 동적 노선 기준으로 재작성.

---

## 재설계 전제 (변경 불가)

1. **동적 전용**: manager pressure signal 기반 런타임 swap만. 정적 프로파일 없음.
2. **단방향**: F16 → Q4_0 만. 역방향 복귀 경로는 이번 스코프 외.
3. **신호 payload**: `ResilienceAction::SwapWeights { ratio: f32 }` (0.0~1.0). 개별 layer idx 아님.
4. **QCF 측정**: prefill 종료 직후 1회. QCF-PPL 상관은 기존 KV eviction/압축 정책에서 증명됨 → PPL 별도 측정 불필요.
5. **파일 구성**: F16 GGUF + Q4_0 GGUF 두 벌 디스크 상주 (동일 shape/metadata). 두 파일 모두 시작 시 mmap handle 확보, 평시 F16만 forward 접근.
6. **swap 단위**: layer 전체. tensor 단위/sublayer 단위 불허.
7. **평시 오버헤드 제로**: QCF collector는 on-demand 활성화 플래그 모드. swap 없으면 계측 경로 비활성.

---

## Phase 구조 요약

| Phase | 내용 | 예상 작업 수 | 예상 기간 | 상태 | 우선순위 |
|-------|------|-----|----------|------|----------|
| **1 — 인프라** | `LayerSlot` 구조, secondary 파일 mmap handle 보관, 기존 단일 dtype 초기 로드 경로 보존 (회귀 방지) | 5 | 5–7일 | DONE (2026-04-24, `07b8fa3`) | P1 |
| **2 — Swap 실행** | `WeightSwapHandler` + `SwapExecutor` + `ratio_generation` 재진입 차단. 수동 CLI/테스트 트리거 | 5 | 6–8일 | DONE (2026-04-24, `07b8fa3`) | P1 |
| **3 — Manager 연동** | `EngineCommand::SwapWeights{ratio, DtypeTag}` direct dispatch + `WeightSwapDecider` (importance×ε bottom-k) + `QuantNoiseTable` eager ε + on-demand `ImportanceCollector` 주입 + `QcfEstimate.layer_swap` 확장 + LuaPolicy | 7 | 7–9일 | DONE (2026-04-24, `07b8fa3`) | P2 |
| **3.5 — plan invalidation 배선** | `_entry_ratio_generation` → `FullKernelPlan` plan rebuild 통합. PlanInvalidated lazy fallback. tensor_partition × swap 상호 배타 | 4 | 3–4일 | DONE (2026-04-25, `54017ed`) | P1 |
| **3.6 — Noshuffle SOA registry coherence** | SwapExecutor가 OpenCLBackend `noshuffle_soa_registry` invalidate 추가. Q4_0 swap 시 stale cl_mem key 제거 | 1 | 1–2일 | DONE (2026-04-25, `54017ed`) | P1 |
| **3.7a — SOA 재변환 safety net** | swap 직후 `convert_aos_to_soa()` + registry 등록. ENG-ALG-222, INV-131. Phase 3.6 후 발견된 "Paris 정답 + 후속 garbage" 해결. | 3 | 2~3일 | **CURRENT** | P1 |
| **3.7b — AUF v0.1 self-contained format** | ARGUS_W magic + 256B header + section table. META/TOKENIZER/TENSOR_INDEX/WEIGHTS_* sections. auf-tool CLI. ENG-DAT-096, ENG-ALG-223, INV-132~134. | 5 | 4~6일 | **CURRENT** | P1 |
| **3.7c — UMA zero-copy** | AUF mmap → `CL_MEM_USE_HOST_PTR`. swap latency < 5ms/layer 목표. | 1 | 2~3일 | DEFERRED (Phase 5 가능) | P3 |
| **4 — 실측/튜닝** | Galaxy S25에서 PSS / swap latency / TBT 영향 실측, INV-122 임계값 조정 | 4 | 4–6일 | DONE (2026-04-26, `2900f5f`, border-line) | P2 |
| **5 — Phase 4 후속 sprint (Sprint A 종결, B 진행)** | Sprint A: COLD-UNIFORM(부분 충족 종결) + INV122-SWEEP(spec 위임) — 2026-04-26 완료. Sprint B: TBT-DIAG(진행중) + INV122-SPEC(진행중). C: KIVI fallback(대기) | 5 | 8–12일 | **CURRENT** | P1~P2 |

**진행 순서 강제**: Phase 1 → 2 → 3 → 3.5 → 3.6 → **3.7a → 3.7b** → 4.
- Phase 3.7a와 3.7b는 별도 PR로 분리 가능하지만, 3.7a는 Phase 4 진입 필수 머지 (INV-131 + Q4_0 swap 정확성).
- 3.7b는 Phase 4와 병행 가능 — 3.7a로 정확성을 확보하고, 3.7b는 추후 latency 최적화 PR.
- 3.7c는 3.7b + Phase 4 실측 후 latency 결과에 따라 결정.

---

## 선행 조사 (사전 확인, 재조사 불필요)

- `engine/src/models/loader/gguf.rs:342-347` — tensor별 zero-copy mmap slice 경로. secondary 파일도 동일 방식으로 mmap 준비.
- `engine/src/core/qcf/layer_importance.rs` — `ImportanceCollector`/`ImportanceTable` 존재. on-demand 활성화 플래그 모드는 신규 추가.
- `engine/src/core/pressure/swap_handler.rs` — 스텁. `WeightSwapHandler`로 확장 (KV swap과 완전 독립).
- `engine/src/core/pressure/mod.rs` — `CachePressureHandler` trait, `ActionResult` enum 확장 포인트.
- `engine/src/models/transformer.rs:378` — `prepare_tensor_partition()`가 `ratio_generation` counter로 plan invalidate 선례. 동일 패턴 재사용.
- `engine/src/resilience/strategy/memory.rs` — `MemoryStrategy::Critical`과 `ResilienceAction` 매핑 확장 포인트.
- `shared/` — `SystemSignal`/`ResilienceAction` serde, manager-engine 프로토콜.
- **KV swap과 weight swap은 독립 handler로 분리** (MEMORY.md 기록 참조).

**이전 Architect 작업에서 유지/재활용**:
- `spec/41-invariants.md` §3.13 `INV-122` — mixed precision 정확성 불변식은 **그대로 유효**.
- `spec/32-engine-algorithms.md` §3.12 `ENG-ALG-210` — 정적 전제 반영 부분 재작성 필요.
- `spec/33-engine-data.md` §3.17 `ENG-DAT-090` — 유지. `ENG-DAT-091` (LayerDtypeProfile TOML) — **폐기**.
- `arch/weight_swap.md` — Phase A 정적 부분 삭제, Phase B 골격을 중심으로 전면 재구성.

---

# Phase 1 — 인프라 (LayerSlot + Secondary mmap) — DONE (2026-04-24, `07b8fa3`)

> **목표**: `TransformerWeights`를 `LayerSlot` 기반으로 리팩토링하되 **단일 F16 경로가 기존과 동일**하게 동작. Secondary Q4_0 GGUF의 mmap handle만 보관 (lazy, forward 미사용).
> **예상 기간**: 5–7일 (5개 작업)
> **기대 효과**: swap의 토대. 단독으로는 PSS 감소 없음. 이 단계에서 성능 회귀가 없음을 확인한 후 Phase 2 진입.
> **완료 요약**: spec 340 pass, clippy --all-targets clean. LayerSlot/SecondaryMmap/TransformerWeights 도입 완료. 자세한 내역은 MEMORY: `project_weight_swap_phase3_handoff.md` 참조.

## [P0] WSWAP-1-SPEC. Architect: Phase 1 Spec/Arch 재작성
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: 없음
- **담당 권장**: Architect
- **Description**:
  - `arch/weight_swap.md` 전면 재작성 (Phase 1–4 구조, Mermaid 다이어그램, component-centric 서술)
  - `spec/33-engine-data.md` — `ENG-DAT-091` (LayerDtypeProfile TOML) 폐기, 대신 `LayerSlot` 데이터 구조 spec 신설
  - `spec/32-engine-algorithms.md` — `ENG-ALG-210` 재작성 (정적→동적 전제 반영)
  - `spec/41-invariants.md` — 신규 INV: (1) swap 실행 중 forward 재진입 금지 (INV-121 강화 또는 신규), (2) `LayerSlot::current_dtype` 일관성, (3) secondary mmap 생존 불변식
  - `INV-122` 유지 (mixed precision 정확성): NMSE ≤ 0.01, top-5 overlap ≥ 0.9, top-1 match ≥ 0.95 (실측 후 조정 여지)
  - Prefill-tail 측정 "마지막 N 토큰" 정책 결정 (N 값, decode 중 신호 수신 시 fallback policy)
- **Acceptance Criteria**:
  - `arch/weight_swap.md` 재작성 완료 (Mermaid 포함, feedback_mermaid_diagrams.md / feedback_arch_component_centric.md 준수)
  - `ENG-DAT-091` 폐기 표기, 신규 `LayerSlot` spec ID 할당
  - 신규 INV ID 3개 할당 + 조건 명시
  - fallback policy 결정 (예: "다음 prefill 대기, 단 K 토큰 초과 시 강제 measure 건너뛰고 uniform ratio 적용")
  - `tests/spec/` 테스트 요구사항 초안 작성 (feedback_spec_tests_required.md 준수)
- **Notes**: spec-manage 스킬 사용. 이 작업 완료 전까지 Phase 1 구현 작업 착수 금지.

## [P0] WSWAP-1-SLOT. `LayerSlot` 구조 도입 + `TransformerWeights` 리팩토링
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-1-SPEC
- **담당 권장**: Senior Implementer (Arc swap / 메모리 semantics)
- **Description**:
  - `LayerSlot { current_dtype, weights: LayerWeights, secondary_mmap_handle: Option<...> }` 구조 정의
  - `TransformerWeights`의 layer 벡터 → `Vec<Arc<LayerSlot>>` (또는 `ArcSwap<LayerSlot>` 검토)
  - forward 경로에서 layer 진입 시 `Arc` clone으로 snapshot 획득 → 교체 중 stale 참조 안전성 확보
  - `Mutex` 사용은 **최후 수단**. lock-free snapshot-read 경로 우선 (R-new-1 회귀 방지)
  - 초기화 시 `current_dtype = F16`, `secondary_mmap_handle = None` (Phase 2에서 채움)
- **Acceptance Criteria**:
  - 기존 F16 단일 경로 forward가 동일 bit 결과 (greedy 텍스트 diff 0)
  - 모든 기존 유닛/디바이스 테스트 통과 (`cargo test --workspace`)
  - decode tok/s 회귀 ≤ 1% (Galaxy S25 6T, 5회 평균)
  - 호스트 클리피 클린 (`cargo clippy --workspace -- -D warnings`)
- **Notes**: **가장 큰 리스크 작업 (R-new-1)**. 착수 전 Architect와 설계 리뷰 필수. Arc 간접 참조 오버헤드 벤치 우선 확인. 이 작업이 회귀를 남기면 Phase 2/3 전체가 의미 없음.

## [P1] WSWAP-1-MMAP. Secondary GGUF mmap handle 보관 인프라
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-1-SLOT
- **담당 권장**: Implementer
- **Description**:
  - `GgufLoader`에 `open_secondary(path)` API 추가 — Q4_0 파일을 mmap만 수행, tensor slice 추출은 lazy
  - tensor name → byte range 메타데이터는 즉시 파싱 (swap 시점 재파싱 비용 회피)
  - 각 `LayerSlot`에 secondary handle 주입 (layer idx → tensor range 맵)
  - primary(F16) vs secondary(Q4_0)의 shape/metadata 일치 검증 (swap 전에 반드시 검증)
  - mmap 소유권: `TransformerWeights`가 생존 기간 내내 유지 (lifetime 보장)
- **Acceptance Criteria**:
  - F16 단독 모델 로드 시 secondary handle 미생성 (평시 오버헤드 제로)
  - F16+Q4_0 두 파일 동시 open 시 가상 메모리 usage 표기 (실제 RSS는 증가 최소, lazy fault)
  - shape mismatch 시 에러 반환 (unit test)
  - mmap 실패 / 파일 부재 시 fallback = F16 단독 경로 (degrade gracefully)
- **Notes**: Android UFS 4.0에서 mmap은 page fault 시점까지 RSS 영향 없음. 평시 오버헤드 제로 원칙 준수.

## [P1] WSWAP-1-CLI. Secondary 파일 경로 CLI 최소화
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-1-MMAP
- **담당 권장**: Implementer
- **Description**:
  - `generate` 바이너리: `--model-path-secondary <q4_0.gguf>` 플래그 추가 (optional)
  - 미지정 시 secondary 미개방, swap 기능 비활성 (signal 수신해도 no-op 또는 error log)
  - 지정 시 WSWAP-1-MMAP 경로로 secondary handle 준비
  - **폐기**: `--layer-dtype-profile` 플래그 (기존 정적 경로). Architect 판단에 따라 코드 잔해 제거 여부 결정
- **Acceptance Criteria**:
  - CLI help에 신규 플래그 등장 (설명은 "Runtime weight swap용 secondary GGUF 경로")
  - 플래그 없는 기존 호출 100% 동일 동작
  - 플래그 지정 + shape mismatch 시 clear error
- **Notes**: `generate.rs:779` 확장자 분기 로직과 충돌 없도록. 기존 `--layer-dtype-profile` / `quantize_profile` 바이너리 삭제는 WSWAP-1-CLEANUP에서 처리.

## [P1] WSWAP-1-CLEANUP. 폐기 코드 제거 (정적 프로파일 경로)
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-1-SPEC, WSWAP-1-CLI
- **담당 권장**: Implementer
- **Description**:
  - `quantize_profile` 바이너리 삭제 (`engine/src/bin/quantize_profile.rs` 존재 시)
  - `LoadConfig::per_layer_dtype` / `LayerDtypeProfile` TOML 구조 제거
  - `--layer-dtype-profile` CLI 플래그 제거 (문서화된 사용처 없음 확인 후)
  - 관련 테스트 / docs 링크 정리
- **Acceptance Criteria**:
  - 삭제 커밋에 삭제 사유 명시 (의도 변경: 정적 → 동적)
  - 삭제 후 `cargo build --workspace` / `cargo test --workspace` 통과
  - docs / README 참조 업데이트 또는 확인
- **Notes**: 아직 master merge 전이라 안전. 만약 master에 일부 merge된 상태면 revert 전략을 Architect와 조율.

---

# Phase 2 — Swap 실행 (Handler + Executor + 재진입 차단) — DONE (2026-04-24, `07b8fa3`)

> **목표**: `WeightSwapHandler` + `SwapExecutor` 구현. 수동 CLI/테스트 트리거로 F16 → Q4_0 layer swap이 bit-정확(target dtype 기준) / 안전하게 동작. Manager 연동 없이 내부 API만.
> **예상 기간**: 6–8일 (5개 작업)
> **기대 효과**: swap 경로 검증. 이 단계에서 정확성/동시성 안전성 확보 후 Phase 3에서 신호 연동.
> **완료 요약**: SwapExecutor (a~e 단계) + ratio_generation 재진입 차단 + WeightSwapHandler + `--force-swap-ratio` CLI + LoadConfig 시그니처 통일 모두 완료.

## [P1] WSWAP-2-EXEC. `SwapExecutor` 구현 (Arc swap + madvise)
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-1-SLOT, WSWAP-1-MMAP
- **담당 권장**: Senior Implementer (메모리 semantics, Arc swap, madvise FFI)
- **Description**:
  - `SwapExecutor::swap_layer(idx) -> ActionResult::Swapped`
  - 단계: (a) secondary mmap 내 Q4_0 tensor slice 추출 → (b) Q/K permutation 적용 → (c) `LayerWeights::Q4_0` 구성 → (d) 새 `LayerSlot` 생성 (`current_dtype = Q4_0`) → (e) `Arc` 교체 (atomic) → (f) 구 F16 페이지에 `madvise(MADV_DONTNEED)` 호출
  - madvise는 Linux/Android 분기. 호스트(비 Android) 테스트는 no-op.
  - 복귀 경로(Q4_0→F16) **구현하지 않음**. Q4_0 상태의 layer는 해당 generation 내에서 고정.
- **Acceptance Criteria**:
  - swap 후 forward가 Q4_0 weight로 정확하게 동작 (Q4_0 단독 경로 bit-identity 검증)
  - `ActionResult::Swapped { layer_idx, from_dtype, to_dtype, latency_ms }` 이벤트 발행
  - `CacheEvent::WeightSwapped` 로그 발행
  - madvise 실패 시 warn log + 계속 진행 (swap 자체는 성공)
- **Notes**: R-new-3. Android 커널 버전별 `MADV_DONTNEED` 동작 상이. `MADV_PAGEOUT` (Android Q+) 병행 검토. PSS 실측은 Phase 4에서.

## [P1] WSWAP-2-REENTRY. `ratio_generation` 재진입 차단
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-2-EXEC
- **담당 권장**: Senior Implementer (동시성)
- **Description**:
  - `transformer.rs:378`의 `ratio_generation` counter 패턴 재사용
  - swap 실행 전 `generation.fetch_add(1)` → forward가 해당 layer 시작 시 generation snapshot 획득
  - swap 중 forward 진입 시: (옵션 A) 대기, (옵션 B) 이전 snapshot 유지하여 완주 후 다음 token부터 새 dtype — Architect가 spec에서 결정
  - Phase 1에서 도입한 Arc snapshot과 결합: forward 진입 시 Arc clone → 중간에 교체되어도 해당 token은 기존 Arc로 완주
  - 동시 swap 요청 직렬화 (`Mutex<SwapExecutor>` 또는 single-threaded executor thread)
- **Acceptance Criteria**:
  - 동시성 테스트: swap + forward 1000회 반복 → panic / deadlock / stale dtype 없음
  - Spec INV (WSWAP-1-SPEC에서 신설한 재진입 금지 불변식) 준수 검증 테스트 (`tests/spec/`)
  - miri 통과 (선택, 가능한 범위)
- **Notes**: R-new-1 회귀 완화. Arc clone 경로가 forward hot loop에 있으므로 반드시 tok/s 측정.

## [P1] WSWAP-2-HANDLER. `WeightSwapHandler` + Pressure Pipeline 연결
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-2-EXEC, WSWAP-2-REENTRY
- **담당 권장**: Implementer
- **Description**:
  - `engine/src/core/pressure/swap_handler.rs` 스텁을 `WeightSwapHandler`로 확장
  - `CachePressureHandler` trait 구현
  - 입력: `HandlerContext { ratio: f32, layer_importance: ImportanceTable, ... }` → 가장 중요도 낮은 layer 집합 선택 (count = ratio × total_layers)
  - 각 선택된 layer에 대해 `SwapExecutor::swap_layer()` 호출
  - 결과: `ActionResult::Swapped { swapped_layers: Vec<usize>, total_latency_ms }`
  - 이미 Q4_0인 layer는 skip (idempotent). 복귀 없음
  - KV swap과 독립된 handler로 등록
- **Acceptance Criteria**:
  - `CachePressurePipeline`에 handler 등록 시 dispatch 정상
  - unit test: mock `SwapExecutor` + ratio=0.25 → 4개 layer 선택 확인 (16-layer 모델)
  - 이미 swap된 layer 재요청 시 no-op (idempotent 테스트)
  - `ActionResult::Swapped`에 per-layer latency 기록
- **Notes**: MEMORY.md "Weight swap은 KV swap과 독립" 원칙. 개별 handler로 분리하되 Pipeline 내 순서는 Architect spec에 명시.

## [P1] WSWAP-2-CLI-TRIGGER. 수동 swap 트리거 CLI (디버그 전용) + LoadConfig 시그니처 전환
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-2-HANDLER
- **담당 권장**: Implementer
- **Description**:
  - **(A) `--force-swap-ratio` 디버그 플래그 추가**
    - `generate` 바이너리에 `--force-swap-ratio <0.0-1.0>` 디버그 플래그 추가 (optional, hidden)
    - prefill 완료 후 즉시 (decode 시작 전) handler를 수동 호출하여 swap 발동
    - layer importance는 임시로 uniform 또는 layer idx 기반 (정식 measurement은 Phase 3)
    - 정식 배포 빌드에서는 `#[cfg(debug_assertions)]` 또는 feature flag로 보호
  - **(B) Loader 시그니처 `LoadConfig` 일괄 전환 (ENG-DAT-090 마감)**
    - 현재 `load_gguf_with_secondary` 등 loader 엔트리는 `primary_path`, `default_dtype`, `Option<secondary_path>` 를 **낱개 파라미터**로 받는다. (Phase 1 shim)
    - 이 커밋에서 공개 엔트리를 `pub fn load_model(config: LoadConfig) -> Result<TransformerModel, LoadError>` 단일 함수로 전환한다.
    - CLI 파싱 → `LoadConfig` 구성 → `load_model` 호출의 단일 경로만 남긴다. 낱개 파라미터를 받는 기존 API는 제거.
    - `generate.rs` 호출부 갱신, 내부 gguf.rs 헬퍼는 `LoadConfig`를 받거나 그대로 유지하되 공개 경계에서 통일.
    - `--force-swap-ratio` 추가와 함께 묶는 이유: CLI 파라미터가 `LoadConfig` 사용 흐름과 동시에 등장하므로 한 번에 옮기는 편이 변경 표면적 최소.
  - **(C) (선택) `TransformerWeights` 죽은 선언 / `SecondaryMmap::cross_layer_offsets` 정리**
    - 본 커밋 또는 별도 cleanup 커밋에서 처리 가능. spec 반영 이미 완료(ENG-DAT-093/094).
    - `engine/src/models/weights/transformer_weights.rs` 파일 및 `mod.rs`의 pub re-export 삭제.
    - `SecondaryMmap`의 `cross_layer_offsets` 필드 및 populate 코드 삭제.
- **Acceptance Criteria**:
  - `generate --model-path <f16> --model-path-secondary <q4_0> --force-swap-ratio 0.5` 실행 성공
  - swap 후 decode 진행 → 텍스트 생성 완료 (coherence 검증)
  - 플래그 없는 호출에 영향 없음
  - `load_model(LoadConfig)` 단일 엔트리로 호출부 통일, 낱개 파라미터 버전 제거
  - `cargo build --workspace` + `cargo clippy --workspace -- -D warnings` + `cargo test --workspace` 통과
- **Notes**: Phase 3에서 manager 연동 완료 후 이 플래그는 유지해도 되고 제거해도 됨 (테스트 유용성). LoadConfig 전환은 **이 커밋 이전에는 시도하지 말 것** (Phase 1 범위 초과).

## [P1] WSWAP-2-TEST. Swap 정확성 테스트 스위트
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-2-HANDLER, WSWAP-2-CLI-TRIGGER
- **담당 권장**: Tester (설계) + Implementer (구현)
- **Description**:
  - 호스트 테스트: mock GGUF 두 벌 (작은 모델, 예: 2-layer 64-dim) → 각 layer swap → forward 결과가 Q4_0 단독 경로와 bit-identity
  - 동시성 테스트: 다중 thread forward + swap 반복 → panic/deadlock 없음
  - Shape mismatch 테스트: 잘못된 secondary 파일 → 명확한 에러
  - INV-122 검증 (NMSE, top-5 overlap) 회귀 테스트 프레임
  - `tests/spec/` 디렉토리 배치 (feedback_spec_tests_required.md)
- **Acceptance Criteria**:
  - 최소 10개 테스트 케이스 (bit-identity, concurrency, shape mismatch, idempotency, partial swap, full swap 등)
  - `cargo test --workspace` 통과
  - INV-122 임계값 하드코딩 대신 spec의 값 import (tunable)
- **Notes**: Galaxy S25 실측은 Phase 4. 이 단계는 호스트 회귀 안전망 중심.

---

# Phase 3 — Manager 연동 (Direct dispatch + On-demand QCF + Decider) — DONE (2026-04-24, `07b8fa3`)

> **목표**: `EngineCommand::SwapWeights { ratio, target_dtype }`가 실제 manager → engine IPC 경로를 타고 `WeightSwapDecider` + `SwapExecutor`를 발동. RequestQcf → 다음 prefill에서 `ImportanceCollector` 주입 → `QcfEstimate.layer_swap` 응답으로 manager 결정 피드백. ε(quantization noise)는 engine init에서 eager 계산.
> **예상 기간**: 7–9일 (7개 작업)
> **기대 효과**: 신호 기반 end-to-end 동작. Phase 4 실측의 전제.
> **근거 스펙**: ENG-ALG-214-ROUTE, ENG-ALG-215~218, ENG-DAT-095, INV-126~128, MSG-042/082/088/089.
> **라우팅 결정 (ENG-ALG-214-ROUTE)**: Pipeline 비경유. `generate.rs`의 command dispatch에서 직접 수신 → Decider → Executor.
> **완료 요약**: 7개 task 모두 완료. spec 340 pass, clippy --all-targets clean.

## [P2] WSWAP-3-CMD. `EngineCommand::SwapWeights` + `DtypeTag` shared 프로토콜
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: Phase 2 완료
- **담당 권장**: Implementer
- **Description**:
  - **shared/src/lib.rs**:
    - `EngineCommand::SwapWeights { ratio: f32, target_dtype: DtypeTag }` variant 추가 (MSG-042)
    - `DtypeTag` enum 신규 (`Q4_0` / `F16` / `F32` / `Q8_0`, snake_case 직렬화) (MSG-082)
    - `EngineMessage::WeightSwapReport(WeightSwapReport)` variant 추가 (MSG-089)
    - `WeightSwapReport { layers_swapped, freed_bytes, latency_ms, qcf_swap_actual }` struct (MSG-089)
    - `LayerSwapEntry { layer_idx, from_dtype, to_dtype }` struct
    - `QcfEstimate`에 `layer_swap: Option<LayerSwapEstimate>` 필드 추가 (MSG-088)
    - `LayerSwapEstimate { per_layer_importance, per_layer_noise, qcf_swap_at_ratio }` struct
  - 모든 필드에 `#[serde(default, skip_serializing_if = "Option::is_none")]` 원칙 적용 (INV-028)
  - verify 하네스 (`verify/scenarios/`) 기존 시나리오 호환성 확인
- **Acceptance Criteria**:
  - shared 크레이트 serde round-trip test (`shared/tests/spec/test_msg_042_swap_weights_cmd.rs`, `test_msg_082_dtype_tag.rs`, `test_msg_088_qcf_estimate_layer_swap.rs`, `test_msg_089_weight_swap_report.rs`)
  - 구 Manager(`layer_swap` 필드 미인지)가 신규 Engine payload 역직렬화 성공 (INV-028)
  - mock_manager/mock_engine이 신규 variant 수신 시 최소 no-op
- **Notes**: MSG-080/081 ID는 사용하지 않음 (초안 단계 vacant, 실제 ID는 MSG-042/082/088/089).

## [P2] WSWAP-3-NOISE. `QuantNoiseTable` + ε eager 계산 (ENG-ALG-216)
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-1-MMAP (Phase 1 완료, secondary mmap)
- **담당 권장**: Senior Implementer (dequantize + Frobenius 경로, CPU SIMD 활용 가능)
- **Description**:
  - `engine/src/models/weights/quant_noise.rs` 신규 파일
  - `QuantNoiseTable { per_layer: Vec<f32>, computed_at_init: bool }` 구조체
  - `QuantNoiseTable::new_from_frobenius(primary, secondary) -> Self` — loader에서 engine init 시 호출
    - 각 decoder layer의 Q/K/V/O/gate/up/down tensor별 dequantize(secondary) → Frobenius 상대 오차² 합산
    - 개별 layer 실패 시 `ε_i = NaN`; 전체 실패 시 `uniform_ones`
  - `QuantNoiseTable::uniform_ones(num_layers)` fallback
  - `epsilon(layer_idx) -> Option<f32>` (NaN은 None)
  - `TransformerModel` 또는 engine service state에 보관 (arch §5.1)
  - Progress log는 optional — 2s+ 예상 시 stderr에 layer 단위 출력
- **Acceptance Criteria**:
  - unit test (`engine/tests/spec/test_eng_dat_095_quant_noise_table.rs`): ones fallback / NaN layer / 정상 Frobenius 계산
  - unit test (`test_eng_alg_216_quant_noise_calc.rs`): 알려진 Q4_0 예시에 대해 ε 값 손검증
  - engine 기동 latency 추가 측정 (Llama 1B 호스트: <1s 기대)
- **Notes**: dequantize는 `engine/src/backend/cpu_neon.rs` 기존 Q4_0 경로 재사용. Init 1회 비용이므로 SIMD 최적화는 2차 고려. INV-127 근거.

## [P2] WSWAP-3-DECIDER. `WeightSwapDecider` 구현 (ENG-ALG-215)
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-3-NOISE, WSWAP-2-HANDLER
- **담당 권장**: Implementer
- **Description**:
  - `engine/src/models/weights/decider.rs` 신규 파일 (Stage 2 `weight_swap_handler.rs`와 분리)
  - `WeightSwapDecider::decide(ratio_max, num_layers, importance, noise, already_swapped, protected) -> Vec<usize>`
  - 알고리즘 (ENG-ALG-215):
    1. Protected set = {0, num_layers-1} ∪ caller-supplied protected
    2. `target_count = floor(ratio_max × num_layers)`
    3. `needed = target_count - already_swapped.len()` (saturating)
    4. Candidates = non-protected ∧ non-swapped ∧ `epsilon(i).is_some()` (INV-127)
    5. Importance 있으면: key = `importance × ε`, ascending sort, tie → idx asc
    6. Importance 없으면: uniform index fallback (재현성, 랜덤 금지)
    7. truncate to needed
  - 관련 함수 `compute_qcf_swap(swap_set, noise, importance) -> f32` 구현 (ENG-ALG-217)
    - 파일 위치: `engine/src/core/qcf/quant_qcf.rs` (신규) 또는 `ImportanceTable` impl block
    - 분자/분모 모두 `importance × ε`로 정규화 → `[0, 1]` 보장
- **Acceptance Criteria**:
  - unit test (`test_eng_alg_215_weight_swap_decider.rs`): 
    - Llama 16-layer (protect 0 & 15) + ratio=0.5 → 8 layer 선택, 하위 8개 key
    - 보호 layer 제외 확인
    - NaN ε layer 제외 확인 (INV-127)
    - tie-breaking idx asc 확인
    - Uniform fallback (importance None) 확인
    - already_swapped 반영 (ratio 상한 의미)
  - unit test (`test_eng_alg_217_qcf_swap_formula.rs`):
    - 빈 set → 0.0, 전체 set → 1.0
    - 단조성 (S1 ⊆ S2 ⇒ QCF(S1) ≤ QCF(S2))
    - ε = 상수 1일 때 uniform importance-only 경로와 동치
- **Notes**: ENG-ALG-213은 이 Decider의 uniform fallback 경로로 흡수됨.

## [P2] WSWAP-3-DISPATCH. `generate.rs` command dispatch (ENG-ALG-214-ROUTE + INV-126)
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-3-CMD, WSWAP-3-DECIDER
- **담당 권장**: Implementer
- **Description**:
  - `generate.rs`의 EngineCommand dispatch match에 `SwapWeights` arm 추가
  - Dispatch 순서 (arch §3.2):
    1. secondary_mmap None ⇒ `Rejected { reason: "NoSecondary" }`
    2. ratio 범위 밖 (≤0 또는 >1) ⇒ `Rejected { reason: "InvalidRatio" }`
    3. target_dtype ≠ Q4_0 ⇒ `Rejected { reason: "UnsupportedDtype" }` (INV-126)
    4. Decider 호출 → layer_set
    5. layer_set 비었으면 `Ok` + 빈 `WeightSwapReport`
    6. `SwapExecutor::execute_swap(layer_set, Q4_0)` → `WeightSwapReport` 구성
    7. `CommandResult::Ok` 즉시 반환
    8. 이후 `EngineMessage::WeightSwapReport` 별도 송출
  - `DtypeTag::Q4_0` → `DType::Q4_0` 변환 helper
  - Stage 2 `weight_swap_handler.rs`를 내부 orchestrator로 유지 (Pipeline 등록은 제거, CachePressureHandler impl도 제거)
  - `ResilienceAction::SwapWeights` (engine 내부) fallback 경로도 같은 dispatch 함수로 귀결되도록 공통 helper 추출
- **Acceptance Criteria**:
  - unit test (`engine/tests/spec/test_eng_alg_214_route_dispatch.rs`):
    - Rejected 4종(NoSecondary, InvalidRatio, UnsupportedDtype, 각 정상 + 비정상 케이스)
    - 정상 Q4_0 swap이 Ok + WeightSwapReport 2-message 순서로 송출됨
    - 이미 전체 swap 완료된 상태에서 ratio=0.5 재수신 ⇒ Ok + 빈 report
  - unit test (`test_inv_126_swap_dtype_reject.rs`): F16/F32/Q8_0 payload 각각 Rejected
  - Pipeline에 WeightSwapHandler 등록되지 않음 확인
- **Notes**: engine-internal `ResilienceAction::SwapWeights`와 shared `EngineCommand::SwapWeights`는 서로 다른 타입. `MemoryStrategy`에서 내리는 fallback action도 이 dispatch helper로 귀결되어야 함.

## [P2] WSWAP-3-QCF. `RequestQcf` → On-demand `ImportanceCollector` 주입 (ENG-ALG-218)
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-3-NOISE, WSWAP-3-DECIDER
- **담당 권장**: Implementer (QCF 경로) + Senior Implementer (hot path 영향 검토)
- **Description**:
  - `ImportanceCollector`에 `active: AtomicBool` 플래그 추가 (기존 `engine/src/core/qcf/layer_importance.rs`)
  - Engine service state에 `collector_flag: enum { Idle, Armed, Collecting, Finalizing }` 추가
  - `EngineCommand::RequestQcf` dispatch 시:
    - `collector_flag = Armed` (다음 prefill에 주입 예약)
    - `CommandResponse::Ok` 즉시 응답 (기존 동일)
  - Prefill 경로:
    - Prefill 시작 시 `collector_flag == Armed` ⇒ `active = true, state = Collecting, tail_target_token = last`
    - Layer loop 내부: `active`가 true일 때만 snapshot_before / record_after
    - Prefill 종료 시 `Collecting` ⇒ finalize → `ImportanceTable`을 engine service state에 보관
  - `QcfEstimate` 응답 빌드:
    - 기존 KV action estimates (ENG-ALG-050) + `layer_swap: Some(LayerSwapEstimate)` 
    - `qcf_swap_at_ratio`에 0.25, 0.5, 1.0 기본 계산
  - `EngineMessage::QcfEstimate` 송출 → `collector_flag = Idle`
  - K=512 fallback: decode 중 pending + K 초과 ⇒ `layer_swap: None`으로 송출 (차선 방식, 측정 실패 통지)
- **Acceptance Criteria**:
  - unit test (`test_eng_alg_218_importance_on_demand.rs`):
    - active=false 평시: forward 경로에 측정 코드 미실행 (micro-bench ≤ 0.5% 회귀)
    - RequestQcf → next prefill → QcfEstimate 송출 시퀀스
    - K=512 초과 시 layer_swap None 포함한 QcfEstimate 송출
  - unit test (`test_inv_128_qcf_collector_leak.rs`):
    - Armed 상태로 prefill 진행 후 QcfEstimate 1회 송출 + Idle 복귀 확인
    - 누수 탐지: 두 번째 prefill에서 active=false 유지
- **Notes**: R-new-2 완화. prefill 경로 hot path에 분기 추가 → cold branch(early return) 유지.

## [P2] WSWAP-3-LUA. Manager LuaPolicy 정책 확장
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-3-CMD, WSWAP-3-DISPATCH
- **담당 권장**: Implementer (Lua 바인딩)
- **Description**:
  - `manager/src/.../policy/`의 Lua 바인딩 확장:
    - `ctx.qcf.layer_swap` (from `QcfEstimate.layer_swap`) — per_layer_importance/noise, qcf_swap_at_ratio
    - `ctx.engine.last_swap` (from `WeightSwapReport`) — layers_swapped, freed_bytes, latency_ms, qcf_swap_actual
  - `policy_default.lua`에 swap 결정 예시 추가 (arch §3.4 참조):
    - Critical + qcf_at_half < 0.3 → swap_weights ratio=0.5 q4_0
    - Emergency → swap_weights ratio=1.0 q4_0
  - 기존 `MemoryStrategy` engine-internal fallback 매핑도 함께 문서화
  - Shared `EngineCommand::SwapWeights` 발행 API helper
- **Acceptance Criteria**:
  - Lua 바인딩 unit test (ctx 구조 노출 확인)
  - policy_default.lua e2e test: Critical signal → SwapWeights command 발행
  - verify 하네스 시나리오 추가 (`verify/scenarios/weight_swap_critical.yaml` 등)
- **Notes**: Manager DPP/LinUCB와의 상호작용은 향후 작업 (Phase 5). 현재는 단순 기반 결정만.

## [P2] WSWAP-3-TEST. Phase 3 spec 테스트 + E2E 통합
- **Status**: DONE
- **Sprint**: done
- **Dependencies**: WSWAP-3-DISPATCH, WSWAP-3-QCF, WSWAP-3-LUA
- **담당 권장**: Tester (설계) + Implementer (구현)
- **Description**:
  - Spec 테스트 (`tests/spec/`):
    - `test_inv_126_swap_dtype_reject.rs` — reserved dtype 처리
    - `test_inv_127_noise_nan_exclusion.rs` — NaN layer 제외 (decider 직접 호출)
    - `test_inv_128_qcf_collector_leak.rs` — collector 누수 금지
    - `test_eng_alg_214_route_dispatch.rs` — dispatch 결정 매트릭스
    - `test_eng_alg_215_weight_swap_decider.rs` — decider 알고리즘
    - `test_eng_alg_216_quant_noise_calc.rs` — ε 계산
    - `test_eng_alg_217_qcf_swap_formula.rs` — QCF_swap 수식
    - `test_eng_alg_218_importance_on_demand.rs` — on-demand 주입
    - `test_eng_dat_095_quant_noise_table.rs` — QuantNoiseTable 구조
    - `shared/tests/spec/test_msg_042/082/088/089.rs` — 프로토콜
  - E2E (verify 하네스):
    - Scenario A: 평시 (신호 없음) → zero overhead 검증
    - Scenario B: RequestQcf → prefill → QcfEstimate layer_swap 수신 → SwapWeights 발행 → WeightSwapReport 수신
    - Scenario C: decode 중 RequestQcf + K=512 초과 → layer_swap None
  - ImportanceCollector armed 누수 fuzz (1000회 RequestQcf/prefill 반복, QcfEstimate 1:1 확인)
- **Acceptance Criteria**:
  - 모든 spec 테스트 통과
  - 3개 E2E 시나리오 모두 통과 (호스트 CI)
  - fuzz 테스트 누수 0건
- **Notes**: feedback_spec_tests_required.md 준수 (inline `#[cfg(test)]` 불충분, `tests/spec/` 배치 필수).

---

# Phase 3.5 — `_entry_ratio_generation` Plan Invalidation 배선 (current sprint)

> **독립 PR 단위로 분리**. tensor_partition plan rebuild 통합 배선.
> **전제**: Phase 1~3 완료(`07b8fa3`).
> **머지 조건**: Phase 4 디바이스 실측 진입 전 필수.
>
> **결정사항 (사용자 확정 — 모두 Architect 권장안 채택, 2026-04-25)**
> - **DF-35-1 (plan 비교 지점)**: `FullKernelPlan::execute()` 진입부 1회. Acquire load 1회 비교 후 본 경로 진행.
> - **DF-35-2 (재빌드 주기)**: lazy. mismatch 시 `PlanInvalidated` 반환 → caller가 `forward_gen` fallback 후 다음 호출에 plan rebuild.
> - **DF-35-3 (tensor_partition × swap)**: 상호 배타. swap 실행 시 `partition_ctx = None` 강제. 둘 다 plan을 무효화하므로 동시 활성은 정의되지 않은 동작.
> - **DF-35-4 (선행 범위)**: Phase 3.5는 plan 무효화만. Noshuffle SOA registry invalidation은 별도 **WSWAP-3.6**으로 분리.

## [P1] WSWAP-3.5-SPEC. Architect: spec 문서 갱신 (ENG-ALG-219/220 + INV-129)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: 없음 (선행 task)
- **담당 권장**: Architect
- **예상 기간**: 0.5일
- **Description**:
  - `spec/32-engine-algorithms.md`에 두 개 알고리즘 신설:
    - `ENG-ALG-219` — Plan Generation Capture & Compare (build 시 `ratio_generation_at_build` 캡처, execute 진입 시 atomic Acquire load 1회 비교, mismatch → `PlanInvalidated` 반환)
    - `ENG-ALG-220` — Plan Invalidation Lazy Rebuild Fallback (caller가 `PlanInvalidated` 수신 시 `forward_gen` 경로로 fallback 후 다음 forward 호출에서 plan을 lazy rebuild)
  - `spec/41-invariants.md`에 `INV-129` 신설 — Plan ↔ Global Generation Coherence (plan 캡처 generation과 모델 현재 generation이 다르면 plan 사용 금지)
  - `INV-120` (기존 partition generation 일관성)과의 관계 명시 — INV-120은 partition 경로 한정, INV-129는 global path 전반 (둘 다 trigger source일 수 있음)
  - spec 문서의 알고리즘/불변식 카운트 갱신 (`spec/COVERAGE.md` 동기화)
- **Acceptance Criteria**:
  - ENG-ALG-219, ENG-ALG-220, INV-129 ID 할당 + 본문 작성
  - INV-120 vs INV-129 비교/우선순위 표 작성
  - `spec/COVERAGE.md`의 알고리즘/불변식 매핑 갱신
  - DF-35-3 결정 (tensor_partition × swap 상호 배타)을 INV-129 또는 ENG-ALG-220 본문에 명시
  - 스펙 테스트 요구사항 초안 (feedback_spec_tests_required.md 준수)
- **Notes**: spec-manage 스킬 사용. 이 작업 완료 전까지 IMPL 착수 금지. WSWAP-3.5-ARCH와 일부 병행 가능하지만 ENG-ALG ID 발급은 이 작업이 먼저.

## [P1] WSWAP-3.5-ARCH. Architect: arch 문서 v5 §2.2.1 + cross-ref 추가
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-3.5-SPEC
- **담당 권장**: Architect
- **예상 기간**: 0.5일
- **Description**:
  - `arch/weight_swap.md` v4 → v5
    - 새 §2.2.1 "plan 경로 소비 규약" 서브섹션 추가
    - 내용: `FullKernelPlan` 빌드 시 `ratio_generation_at_build` 캡처, execute 진입 시 단 1회 Acquire load, mismatch → caller `forward_gen` lazy fallback (DF-35-1, DF-35-2)
    - DF-35-3 상호 배타 결정을 §3 dispatch flow에 반영 (swap 실행 시 partition_ctx 강제 None)
  - `arch/plan_partition_integration.md` 부록 A.11 갱신
    - ENG-ALG-219 cross-ref 추가
    - INV-120(partition generation) ↔ INV-129(global generation) 비교표 (스코프, trigger source, 우선순위, 위반 시 동작)
  - feedback_arch_component_centric.md 준수 (component-centric 서술)
  - feedback_mermaid_diagrams.md 준수 (필요 시 Mermaid 시퀀스 다이어그램)
- **Acceptance Criteria**:
  - `arch/weight_swap.md` 제목 줄에 v5 표기 + changelog 항목 추가
  - §2.2.1 신설 + DF-35 결정 4건 모두 인용
  - `arch/plan_partition_integration.md` A.11에 INV-120 vs INV-129 표 작성
  - cross-ref ID는 spec과 정확히 일치
- **Notes**: spec-manage 스킬 동기화. arch와 spec ID/조건이 완전히 일치해야 함.

## [P1] WSWAP-3.5-TEST. Implementer: spec 테스트 신규 작성
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-3.5-SPEC, WSWAP-3.5-ARCH
- **담당 권장**: Implementer
- **예상 기간**: 1일
- **Description**:
  - `engine/tests/spec/test_eng_alg_219_plan_invalidation.rs` 신규
  - 테스트 케이스:
    1. **Basic stale detection**: mock으로 `ratio_generation.fetch_add(1)` 후 `plan.execute()` → `PlanInvalidated` 반환 확인
    2. **Same generation pass**: ratio_generation 변동 없을 때 plan.execute() 정상 진행
    3. **INV-120/INV-129 양방향 trigger interleaving**: tensor_partition rebuild trigger와 weight swap trigger가 교차 발생 시 둘 다 invalidation 감지
    4. **INV-129 stale detection 단위 테스트**: plan 캡처 generation < 현재 generation → invalidation
    5. **Lazy rebuild path**: PlanInvalidated 수신 후 caller가 forward_gen으로 fallback, 다음 호출에 새 plan 빌드되는 통합 시나리오 (mock backend)
  - WSWAP-3.6 (SOA registry)과 분리하여 plan invalidation 단독 회귀 안전망 확보
- **Acceptance Criteria**:
  - 5개 케이스 모두 통과
  - `cargo test --workspace` clean
  - `cargo clippy --workspace -- -D warnings` clean
  - feedback_spec_tests_required.md 준수: `tests/spec/` 배치, inline `#[cfg(test)]` 불충분
  - 각 테스트 함수 docstring에 spec ID 명시
- **Notes**: IMPL 작업과 병행 가능 — 테스트 먼저 빨갛게 작성한 뒤 IMPL이 초록으로 만드는 것을 권장(TDD 친화).

## [P1] WSWAP-3.5-IMPL. Implementer: plan 캡처 + execute 비교 + dispatch 배선
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-3.5-SPEC, WSWAP-3.5-ARCH
- **담당 권장**: Implementer
- **예상 기간**: 1~2일
- **Description**:
  - **(A) FullKernelPlan 필드 추가** — `engine/src/backend/opencl/plan.rs::FullKernelPlan`
    - `ratio_generation_at_build: u64`
    - `ratio_generation_counter: Arc<AtomicU64>`
    - `execute()` 진입부 1회 `counter.load(Ordering::Acquire)` → `ratio_generation_at_build`와 비교
    - mismatch 시 `Err(PlanError::PlanInvalidated)` 반환 (또는 `Result<_, PlanInvalidated>` 형태)
  - **(B) plan build 시 캡처** — `build_full_plan` 호출부에서 `model.ratio_generation.clone()` 캡처
  - **(C) forward_into 배선** — `engine/src/models/transformer.rs::forward_into`
    - 기존 `_entry_ratio_generation` 인자의 underscore 제거 → 정상 사용
    - plan 경로로 `ratio_generation_counter` Arc 전달
  - **(D) dispatch lazy fallback** — `engine/src/bin/generate.rs`
    - `PlanInvalidated` 수신 시 forward_gen fallback (기존 tensor_partition stale 감지 패턴 재활용)
    - 다음 호출에 plan rebuild trigger
  - **(E) DF-35-3 상호 배타** — `SwapExecutor::execute_swap` 또는 `dispatch_swap_weights`
    - swap 실행 시 `partition_ctx = None` 강제 (구조적으로 둘 다 plan invalidate, 동시 활성 정의되지 않음)
    - clear에 대한 이벤트/로그 발행
  - 기존 tensor_partition rebuild 경로 (INV-120) 회귀 없음 보장
- **Acceptance Criteria**:
  - WSWAP-3.5-TEST의 5개 케이스 통과
  - 기존 spec 340 + 신규 케이스 합쳐 전체 pass
  - `cargo build --workspace --release` clean
  - `cargo fmt --all` + `cargo clippy --workspace --all-targets -- -D warnings` clean
  - 호스트 generate 회귀 없음 (단순 prompt → 정상 토큰 생성)
  - swap → 다음 forward에서 정확히 1회 PlanInvalidated → forward_gen fallback → 그 다음 forward에서 plan rebuild 발생 (event 또는 log로 검증)
  - `--tensor-partition <r>` + swap 동시 시도 시 swap 직후 partition_ctx None 확인 (디버그 로그 또는 단위 테스트)
- **Notes**: 가장 큰 위험 — plan.rs 시그니처 변경으로 호출부 다수 영향. plan 시그니처는 architect와 사전 합의된 형태 유지. forward 핫패스에 atomic load 1회 추가는 negligible (Acquire load는 ARM relaxed 기준 단일 dmb ld). 완료 후 자동 커밋 + `notify-send`.

---

# Phase 3.6 — Noshuffle SOA Registry Coherence (Phase 3.5와 독립)

> **목표**: Q4_0 weight swap 시 OpenCLBackend의 `noshuffle_soa_registry` 무효화. 디바이스 한정 silent correctness bug 차단.
> **전제**: Phase 3 완료. Phase 3.5와 병행 가능.
> **머지 조건**: Phase 4 디바이스 실측 진입 전 필수.
>
> **배경**:
> - OpenCLBackend의 `noshuffle_soa_registry`는 HashMap (key = `cl_mem` 포인터 주소).
> - Q4_0 weight swap은 layer의 weight 버퍼를 새 `cl_mem`으로 교체하므로 기존 key가 stale이 됨.
> - 현재 `SwapExecutor`는 이 registry를 invalidate하지 않음 → 다음 forward에서 stale entry 조회 시 silent garbage.
> - 호스트 NVIDIA 백엔드에서는 fallback 경로가 다르므로 발현 안 됨. **디바이스(Adreno) 한정** correctness 이슈.

## [P1] WSWAP-3.6-SOA. SwapExecutor가 OpenCL Noshuffle SOA registry invalidate
- **Status**: DONE (2026-04-25, `54017ed` — Phase 3.6 완료. S25 실측 "Paris" 첫 토큰만 정답, 후속 garbage 발견 → Phase 3.7로 이행)
- **Sprint**: done
- **Dependencies**: Phase 3 완료 (Phase 3.5와 독립적으로 진행 가능, Phase 4 진입 전 머지 필수)
- **담당 권장**: Senior Implementer (`backend/opencl/` + cl_mem 수명관리 영역)
- **예상 기간**: 1~2일
- **Description**:
  - **(A) Spec 신설** — `spec/41-invariants.md`에 `INV-130` 추가
    - "Noshuffle SOA Registry Coherence": weight swap 후 OpenCLBackend의 noshuffle SOA registry는 stale `cl_mem` 키를 보유하지 않아야 한다. 모든 key는 현재 layer가 보유한 활성 buffer를 가리킨다.
  - **(B) Backend API 노출** — `engine/src/backend/opencl/mod.rs`
    - `clear_noshuffle_soa_registry()` 또는 `invalidate_noshuffle_soa_layer(layer_idx)` API 추가
    - 후자가 우선(범위 좁음). 단 현재 구조상 layer→cl_mem 역인덱스가 없으면 전체 clear가 단순/안전
  - **(C) SwapExecutor 호출 배선** — `engine/src/models/weights/swap_executor.rs` (또는 동등 경로)
    - step e (Arc swap) 직전 또는 직후에 backend.clear_noshuffle_soa_registry() 호출
    - 호출 시점은 forward 진입 가능 윈도우 직전이어야 함 (Architect와 사전 합의 권장)
  - **(D) 자연 재등록 검증**
    - 다음 forward → plan rebuild → noshuffle SOA 등록 경로 재진입 → 새 cl_mem 주소로 자연 재등록되는 흐름 확인
  - **(E) 테스트** — `engine/tests/spec/test_inv_130_noshuffle_soa_coherence.rs` 신규
    - mock OpenCLBackend로 swap 전후 registry 상태 검증
    - 디바이스 실측은 Phase 4 (`WSWAP-4-LATENCY`/`WSWAP-4-PSS`와 통합 시나리오)
- **Acceptance Criteria**:
  - INV-130 spec 신설 + arch 문서(`arch/weight_swap.md`)에 cross-ref
  - 호스트 단위 테스트 통과 (registry clear, swap 후 stale key 없음)
  - 호스트 generate 회귀 없음 (CPU 백엔드 / OpenCL fallback 백엔드 모두)
  - clippy clean, fmt clean
  - SwapExecutor 호출 순서가 docstring에 명시 (Arc swap 시점과의 ordering)
- **Notes**:
  - **임팩트 우선순위**: 디바이스 silent correctness bug → Phase 4 실측 결과 신뢰성에 직결. 따라서 P1.
  - Phase 3.5와 독립이지만 둘 다 Phase 4 진입 전 필수 머지. 진행 우선순위: 3.5와 병렬 처리 권장.
  - Architect 사전 검토 권장 항목: registry per-layer invalidation API vs 전체 clear 트레이드오프.
  - 완료 후 자동 커밋 + `notify-send`.

---

# Phase 3.7 — SOA 재변환 + AUF v0.1 self-contained format (current sprint)

> **목표**: Phase 3.6 디바이스 실측에서 발견된 "swap 후 첫 토큰만 정답, 후속 garbage" 문제 해결.
> - **3.7a (런타임 safety net)**: Q4_0 swap 시 `convert_aos_to_soa()` 명시 호출 + registry 등록.
> - **3.7b (AUF 포맷)**: GGUF에서 빌드 시점에 모든 backend variant payload를 사전 변환하여 single self-contained `.auf` 파일에 보관.
> - **3.7c (선택, Phase 5로 미룰 수 있음)**: UMA zero-copy `CL_MEM_USE_HOST_PTR` (Adreno 한정).
>
> **전제**: Phase 1~3.6 완료 (`54017ed`). S25 실측 "Paris" 정답 확인 후 후속 garbage. Spec 348 pass.
>
> **머지 조건**: 3.7a 단독으로도 Phase 4 디바이스 실측 진입 가능. 3.7b는 후속 머지로 진행 가능.
>
> **사용자 확정 결정 (2026-04-25)**:
> - DF-37-1: AUF 운영 모드 = Mode B (self-contained). GGUF 없이 동작.
> - DF-37-2: B-2 + Selective Strip — 빌드 시 모든 variants 포함, 배포 시 strip.
> - DF-37-3: 3-tier 버저닝 = semver + capability flags + section table.
> - DF-37-4: Magic = `"ARGUS_W\0"` (8B). format_major=0 기간은 실험적, v0.1로 시작.
> - DF-37-5: Header 256B 고정. device_tag는 헤더가 아니라 `WEIGHTS_*` section tag로 표현.
> - DF-37-6: Section payload alignment = 64KB (Linux/Android THP 친화). Architect 권고 채택.
> - DF-37-7: source_hash 알고리즘 = hybrid (size + mtime + head/tail 8MB sha256). Architect 권고 채택.
> - DF-37-8: auf-tool 위치 = `engine/src/bin/auf_tool.rs` (workspace 단순). Architect 권고 채택.
> - DF-37-9: TOKENIZER 직렬화 = GGUF tokens 그대로 보존 + 경량 이진 헤더. Architect 권고 채택.
> - DF-37-10: 자동 strip / 자동 cache 정책 = 도입 안 함. v0.2 이후 사용자 피드백 후 재결정.
> - DF-37-11: Repacker (auf-tool repack) = Phase 3.7 범위 외. Phase 5로 미룸. v0.1 auf-tool은 build/info/strip/verify만.

## Phase 3.7a — Adreno SOA 재변환 safety net

### [P1] WSWAP-3.7A-SPEC. Architect: spec 문서 갱신 (ENG-ALG-222, INV-131)
- **Status**: DONE (2026-04-25, 본 작업)
- **Sprint**: current
- **Dependencies**: Phase 3.6 완료
- **담당 권장**: Architect
- **예상 기간**: 0.5일
- **Description**:
  - `spec/32-engine-algorithms.md` §3.12.16 — ENG-ALG-222 신설 (Adreno SOA 재변환). AUF cache hit / convert_aos_to_soa fallback 분기.
  - `spec/41-invariants.md` §3.16 — INV-131 신설 (swap 후 첫 GPU matmul 직전 SOA registry 등록 보장).
  - `spec/COVERAGE.md` 갱신.
  - `arch/weight_swap.md` 후속 갱신은 implementation 단계에서 Architect가 §2.2.4 추가 (현재 본 task에서는 spec/COVERAGE만).
- **Acceptance Criteria**:
  - ENG-ALG-222, INV-131 ID 할당 + 본문 작성. **DONE**.
  - INV-130(stale 제거)와 INV-131(신규 등록)의 dual 관계 명시. **DONE**.
- **Notes**: 본 architect task로 완료.

### [P1] WSWAP-3.7A-IMPL. Implementer: SwapExecutor SOA 재변환 호출 배선
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-3.7A-SPEC
- **담당 권장**: Senior Implementer (`backend/opencl/` + Q4_0 layout 영역)
- **예상 기간**: 1~2일
- **Description**:
  - `engine/src/models/weights/swap_executor.rs` (또는 동등 경로): step (e) `clear_noshuffle_soa_registry()` 직후, 각 swap layer의 Q4_0 weight tensor에 대해:
    - (1) AUF cache (Phase 3.7b 통합 시) lookup → hit이면 SOA descriptor 등록.
    - (2) miss이면 `OpenCLBackend::convert_aos_to_soa(tensor)` 호출 → 새 cl_mem들로 SOA payload 업로드 → `register_noshuffle_soa(addr, descriptor)`.
    - 호스트/CPU/CUDA 백엔드에서는 NoOp.
  - 호출 시점: `ratio_generation` bump 직전 (다음 forward 진입 전 등록 완료 보장).
  - tensor 종류: `attn_q`, `attn_k`, `attn_v`, `attn_o`, `ffn_gate`, `ffn_up`, `ffn_down` (norm은 Q4_0 아니므로 제외).
- **Acceptance Criteria**:
  - 호스트 generate 회귀 없음.
  - 디바이스(S25) "Paris" + 후속 토큰 정상 — INV-131 충족.
  - clippy clean, fmt clean.
  - INV-122 임계 (logit NMSE ≤ 0.01, top-5 ≥ 0.9, top-1 ≥ 0.95) S25에서 통과.
- **Notes**:
  - 이 작업 단독으로도 Phase 4 진입 가능.
  - 변환 비용은 매 swap마다 발생 (~50ms/layer 예상). AUF 도입(3.7b) 시 0으로 감소.

### [P1] WSWAP-3.7A-TEST. Implementer: spec 테스트 신규 작성
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-3.7A-SPEC
- **담당 권장**: Implementer
- **예상 기간**: 0.5일
- **Description**:
  - `engine/tests/spec/test_inv_131_soa_reconversion.rs` 신규.
  - 테스트 케이스:
    1. **Mock Adreno SOA registry**: swap 전 layer X의 cl_mem이 등록되어 있고, swap 후 새 cl_mem이 등록되어 있는지 확인.
    2. **Host backend NoOp**: registry 자체가 비어 있을 때 swap 시 panic 없음.
    3. **Cache miss → convert_aos_to_soa fallback**: AUF cache 부재 시 fallback 호출 발생 검증 (mock).
- **Acceptance Criteria**:
  - 3개 케이스 통과.
  - 디바이스 실측은 Phase 4 통합 시나리오에 포함 (manual).
- **Notes**: feedback_spec_tests_required.md 준수.

## Phase 3.7b — AUF v0.1 self-contained format

### [P1] WSWAP-3.7B-SPEC. Architect: AUF v0.1 spec 신설
- **Status**: DONE (2026-04-25, 본 작업)
- **Sprint**: current
- **Dependencies**: 없음
- **담당 권장**: Architect
- **예상 기간**: 1일
- **Description**:
  - `spec/33-engine-data.md` §3.22 — ENG-DAT-096 신설 (AUF v0.1 binary format).
  - `spec/32-engine-algorithms.md` §3.12.17 — ENG-ALG-223 신설 (reader/writer/stripper 알고리즘).
  - `spec/41-invariants.md` §3.16 — INV-132 ~ INV-134 신설.
  - `arch/auf_format.md` 신규 (component-centric).
  - `docs/auf_format_changelog.md` 신규.
  - `docs/auf_tool_guide.md` 신규.
- **Acceptance Criteria**: 모두 **DONE**.
- **Notes**: 본 architect task로 완료.

### [P1] WSWAP-3.7B-AUF-CRATE. Implementer: AUF reader/writer 모듈 신설
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-3.7B-SPEC
- **담당 권장**: Implementer
- **예상 기간**: 2~3일
- **Description**:
  - 모듈 위치 권고: `engine/src/auf/` (mod.rs + header.rs + section.rs + reader.rs + writer.rs + stripper.rs + variant_*.rs).
  - 의존: `memmap2`, `sha2`, `serde_json` (META JSON), 기존 GGUF parser.
  - `AufHeader::parse / serialize`, `SectionTable::parse / serialize / find / validate`.
  - `auf_read(path, backend_tag) -> Result<AufView, AufError>` — reader.
  - `auf_build(opts) -> Result<()>` — writer.
  - `auf_strip(in_path, opts) -> Result<()>` — stripper.
  - Variant 변환 함수: `variant_adreno_soa.rs`, `variant_cuda_aos.rs`, `variant_cpu_aos.rs`. 각각 GGUF layer tensor → variant payload byte vector.
  - `AufError` enum: `Truncated`, `NotAuf`, `FormatTooNew`, `UnknownCapability`, `SectionOverlap`, `SectionOutOfBounds`, `RequiredMissing`, `WeightsMissing`, `IoError`, `JsonError` 등.
  - **panic 금지** (ENG-ALG-C10): 모든 무결성 위반은 `Result::Err`.
  - **atomic write** (ENG-ALG-C11): writer/stripper는 tempfile + rename.
- **Acceptance Criteria**:
  - 모듈 컴파일 + 단위 테스트 통과.
  - 모든 INV-132/133/134 케이스에 대해 reader가 panic 없이 Err 반환.
  - clippy clean, fmt clean.
- **Notes**:
  - Adreno SOA 변환은 기존 `OpenCLBackend::convert_aos_to_soa`의 정적 변환 로직을 재사용. 단, AUF writer는 GPU 미사용 (호스트에서 byte-level 변환).
  - CUDA AOS 변환은 단순 padding (Q/K permute는 GGUF parser에 이미 적용된 경우 추가 변환 불요).
  - 모듈 boundary: AUF는 GGUF에 의존하나 OpenCL backend에 의존하지 않는다 (AUF writer는 호스트 byte 변환만).

### [P1] WSWAP-3.7B-CLI. Implementer: auf-tool CLI binary 신설
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-3.7B-AUF-CRATE
- **담당 권장**: Implementer
- **예상 기간**: 1~2일
- **Description**:
  - `engine/src/bin/auf_tool.rs` 신규.
  - 서브커맨드: `build`, `info`, `strip`, `verify`. (`repack`은 미구현, "Phase 5 reserved" 메시지로 stub)
  - 인자 파싱: 기존 generate.rs와 동일한 stdlib `std::env::args` 또는 `clap` (workspace 의존성 확인 후 결정).
  - `build`: `--input`, `--output`, `--variants`, `[--created-by]`.
  - `info`: positional path. stdout에 헤더 + section 목록 출력.
  - `strip`: `--keep`, `[--no-backup]`, positional path.
  - `verify`: `[--source]`, positional path.
  - `--variants all`은 자동으로 `WEIGHTS_ADRENO_SOA,WEIGHTS_CUDA_AOS,WEIGHTS_CPU_AOS` 확장.
  - 에러 메시지는 사용자 친화 (stderr + exit code != 0).
- **Acceptance Criteria**:
  - `auf-tool build` → `auf-tool info` → `auf-tool strip` → `auf-tool info` 라이프사이클이 호스트에서 정상 동작.
  - 잘못된 입력에 대해 명확한 에러 메시지 + non-zero exit code.
  - `auf-tool verify` 가 INV-132/133/134 모든 위반 케이스를 검출.
  - clippy clean, fmt clean.
- **Notes**:
  - cargo workspace에서 `cargo build -p llm_rs2 --bin auf_tool`로 빌드 가능.
  - Android 디바이스용 빌드 검증은 선택 사항 (workstation tool이 주 용도).

### [P1] WSWAP-3.7B-ENGINE. Implementer: Engine `--secondary-source` AUF 지원
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-3.7B-AUF-CRATE
- **담당 권장**: Implementer
- **예상 기간**: 1일
- **Description**:
  - `engine/src/bin/generate.rs`: `--secondary-source <path>` 처리에서 확장자 분기:
    - `.gguf` → 기존 GGUF loader.
    - `.auf` → `auf_read(path, backend_tag)` 호출. backend_tag은 현재 backend로부터 도출.
    - 그 외 → 에러.
  - `engine/src/models/loader/mod.rs` 또는 `loader/auf.rs` 신규: AUF view → `SecondaryMmap` 어댑터. `LayerSlot::secondary_mmap_handle`로 연결.
  - `SwapExecutor`가 AUF backed `SecondaryMmap`에서 layer payload를 읽을 때, payload는 이미 backend variant 형식이므로 추가 변환 불필요 — 직접 `Tensor::from_buffer`로 wrap.
  - WSWAP-3.7A의 convert_aos_to_soa fallback 경로는 **AUF 미사용** 시에만 활성. AUF cache hit이면 등록만.
- **Acceptance Criteria**:
  - 호스트 CPU + AUF (CPU AOS) 시나리오: GGUF primary + AUF secondary로 swap 동작.
  - 디바이스(S25) Adreno + AUF (ADRENO_SOA) 시나리오: swap 후 INV-122 통과 + 변환 비용 0 검증 (latency 측정).
  - Self-contained 검증: AUF만으로 모델 메타데이터/tokenizer 로딩 확인 (Mode B).
  - clippy clean, fmt clean.
- **Notes**:
  - Phase 3.7b 핵심 통합 작업. Mode B 자립성의 검증.
  - 향후 primary GGUF도 AUF로 대체 가능하지만 v0.1 범위는 secondary only.

### [P1] WSWAP-3.7B-TEST. Implementer: spec 테스트 신규 작성
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-3.7B-AUF-CRATE
- **담당 권장**: Implementer
- **예상 기간**: 1일
- **Description**:
  - `engine/tests/spec/test_eng_dat_096_auf_format.rs` 신규: header/section table round-trip serde.
  - `engine/tests/spec/test_eng_alg_223_auf_io.rs` 신규: build/read/strip 라이프사이클.
  - `engine/tests/spec/test_inv_132_auf_reader_reject.rs` 신규: magic/format/capability mismatch reject.
  - `engine/tests/spec/test_inv_133_auf_required_sections.rs` 신규: META/TOKENIZER/TENSOR_INDEX/WEIGHTS_* 누락 reject.
  - `engine/tests/spec/test_inv_134_auf_section_integrity.rs` 신규: overlap/out-of-bounds/duplicate-tag reject.
  - 테스트 fixture: 작은 mock GGUF (또는 byte buffer) → AUF build → reader 검증.
- **Acceptance Criteria**:
  - 5개 테스트 파일, 각 최소 3 케이스. 모두 통과.
  - feedback_spec_tests_required.md 준수: `tests/spec/` 배치, inline `#[cfg(test)]` 불충분.
  - 각 테스트 함수 docstring에 spec ID 명시.
- **Notes**: TDD 친화 — IMPL 진입 전 또는 병행하여 작성.

## Phase 3.7c (선택, Phase 5로 미룸 가능)

### [P3] WSWAP-3.7C-UMA. UMA zero-copy `CL_MEM_USE_HOST_PTR` (Adreno 한정)
- **Status**: DEFERRED
- **Sprint**: backlog
- **Dependencies**: WSWAP-3.7B-ENGINE
- **담당 권장**: Senior Implementer
- **예상 기간**: 2~3일 (실측 + 튜닝 포함)
- **Description**:
  - AUF mmap된 `WEIGHTS_ADRENO_SOA` payload를 그대로 `CL_MEM_USE_HOST_PTR`로 wrap하여 GPU에 전달.
  - swap latency 추가 감소 + DRAM 사용량 감소.
  - Adreno 드라이버의 페이지 핀 동작 검증 필요 (MEMORY.md `feedback_adreno_gpu_kernel_state_limit.md` 참조).
- **Acceptance Criteria**:
  - swap latency < 5ms/layer (Adreno UMA, 변환 비용 0 + 페이지 핀 0 가정).
  - INV-122 통과 유지.
- **Notes**:
  - Phase 3.7c는 사용자 결정 — Phase 5로 미루는 것이 기본. AUF 도입(3.7a/3.7b) 후 latency가 50ms/layer 이하라면 미룰 수 있음.

---

# Phase 4 — 실측 / 튜닝 (Galaxy S25)

> **목표**: Galaxy S25에서 PSS 감소량 / swap latency / TBT 영향 / INV-122 임계값을 실측하고 spec 값 조정.
> **예상 기간**: 4–6일 (4개 작업)
> **기대 효과**: production 적용 가능성 판단 + spec 값 확정.
> **블로커**: Phase 3.5(plan invalidation) + Phase 3.6(SOA registry) + **Phase 3.7a(SOA 재변환 IMPL) 최소 머지 후 진입**. Phase 3.7b(AUF) 머지는 권장이지만 필수는 아님 — 3.7a 단독으로도 INV-122 통과 가능.
>
> **종결 (2026-04-26, HEAD `2900f5f`)**: SOA bypass 본격 구현 + 5차 디바이스 측정으로 per-layer p50 −74.4% (206→52.8 ms), soa_reconvert −90.1% (172→17 ms) 달성. 단 SLA <50 ms는 p50 5.7% 초과 border-line으로 사용자 수용. 잔여 4개 항목(cold-path 균일화 / INV-122 full sweep / TBT gap 진단 / KIVI mixed fallback)은 **Phase 5 별도 sprint**로 분리. 측정 자료: `results/data/weight_swap/phase_4_*.md`. 핸드오프: `project_weight_swap_phase4_handoff.md`.

## [P2] WSWAP-4-LATENCY. Swap latency 실측 (Galaxy S25)
- **Status**: DONE (2026-04-26, `2900f5f`, border-line)
- **Sprint**: done
- **Dependencies**: Phase 3.5 (WSWAP-3.5-IMPL) + Phase 3.6 (WSWAP-3.6-SOA) 머지 완료
- **담당 권장**: Tester
- **Description**:
  - `SwapExecutor` 내부 타임스탬프: (a) secondary slice 추출, (b) Q/K permutation, (c) LayerWeights 구성, (d) Arc swap, (e) madvise
  - Galaxy S25 UFS 4.0 환경에서 ratio={0.25, 0.5, 1.0}별 측정 (V10 thermal isolation)
  - p50/p99/max 기록
  - 타겟: 평균 <50 ms/layer, p99 <100 ms/layer
- **Acceptance Criteria**:
  - `results/data/weight_swap/phase_4_latency.md` 리포트
  - per-단계 breakdown 히스토그램
  - SLA 초과 시 원인 분석 (Q/K permutation vs IO vs madvise)
- **Notes**: R4 (Q/K permutation 비용) 여전히 유효. 측정 결과에 따라 "사전 permuted 파일 저장" 옵션 재검토.

## [P2] WSWAP-4-PSS. PSS 감소량 실측
- **Status**: DONE (2026-04-26, `2900f5f`)
- **Sprint**: done
- **Dependencies**: WSWAP-4-LATENCY
- **담당 권장**: Tester
- **Description**:
  - 시나리오: pre-swap PSS → 신호 주입 (ratio={0.25, 0.5, 1.0}) → post-swap PSS
  - 측정 도구: `dumpsys meminfo`, `/proc/[pid]/smaps_rollup`
  - 타겟: ratio=1.0에서 PSS 감소 > 150 MB (Llama 3.2 1B, F16 baseline)
  - madvise 실제 페이지 회수 확인 (비교: madvise 호출 전/후)
- **Acceptance Criteria**:
  - PSS 감소 표 (ratio × 시간 경과)
  - madvise 효과 정량화 (호출 전/후 delta)
  - R-new-3 검증 완료
- **Notes**: Android 커널 버전 기록 필수. OnePlus/Pixel 디바이스에서도 sanity check (optional).

## [P2] WSWAP-4-TBT. TBT / 처리량 영향 측정
- **Status**: DONE (2026-04-26, `2900f5f`, gap 잔존)
- **Sprint**: done
- **Dependencies**: WSWAP-4-LATENCY
- **담당 권장**: Tester
- **Description**:
  - 측정 (6T, `--profile` 미사용):
    - (a) F16 단독 baseline: decode tok/s
    - (b) Q4_0 단독 baseline: decode tok/s
    - (c) Arc snapshot 경로만 (swap 전): decode tok/s — R-new-1 검증
    - (d) swap 후 mixed (ratio={0.25, 0.5, 1.0}): decode tok/s
  - Prefill-tail QCF 측정 활성화 시 prefill latency 영향
  - 타겟: (c) 회귀 ≤ 1%, (d) ratio=0.5 시 tok/s 열화 ≤ 10% (vs F16), ratio=1.0 시 Q4_0 baseline 수준
- **Acceptance Criteria**:
  - tok/s / TBT 표 (ratio × configuration)
  - R-new-1 (Arc 간접 참조 회귀) 정량 판정
  - 리포트: `results/data/weight_swap/phase_4_throughput.md`
- **Notes**: `--profile` 금지. `Decode: X ms/tok` 로그 라인 사용. V10 thermal isolation 준수.

## [P2] WSWAP-4-INV122. INV-122 임계값 실측 기반 조정
- **Status**: PARTIAL (2026-04-26, greedy sanity 3건만; full sweep은 WSWAP-5-INV122-SWEEP로 이관)
- **Sprint**: done
- **Dependencies**: WSWAP-4-TBT
- **담당 권장**: Tester (측정) + Architect (spec 업데이트)
- **Description**:
  - 다양한 prompt set (최소 100개 prompt)에서 F16 vs mixed forward 결과 비교
  - 측정: NMSE (logits), top-5 overlap, top-1 match, ROUGE-L (생성 텍스트)
  - ratio=0.25, 0.5, 0.75, 1.0 각각
  - 권장 임계값 (NMSE ≤ 0.01, top-5 ≥ 0.9, top-1 ≥ 0.95) 검증 및 조정
- **Acceptance Criteria**:
  - `spec/41-invariants.md` INV-122 임계값 업데이트 (실측 근거 포함)
  - `tests/spec/` 회귀 테스트 임계값 동기화
  - 리포트: `results/data/weight_swap/phase_4_accuracy.md`
- **Notes**: QCF-PPL 상관은 기존에 증명됨 (KV eviction 정책) → PPL 별도 측정 불필요. QCF + forward 정확성만 측정.

---

# Phase 5 — Phase 4 후속 4개 sprint (2026-04-26 신설)

> **목표**: Phase 4 종결 시점에 잔존한 4개 항목을 별도 sprint로 분리하여 처리. 각 항목은 독립 머지 가능하나 일부는 측정 자산을 공유.
> **블로커**: Phase 4 종결 (`2900f5f`) — 해소됨.
> **공통 자산**: 디바이스 신규 AUF (`Llama-3.2-1B-Instruct.auf`, sha256 `1a1ead0c...`, mtime 2026-04-26). 5-stage instrumentation (`engine/src/models/weights/swap_executor.rs`). 4차/5차 측정 결과 표 (`results/data/weight_swap/phase_4_*.md`).
> **공통 측정 환경**: Galaxy S25 6T, V10 thermal isolation, `--profile` 미사용, wall-clock 기준.

## [P1] WSWAP-5-COLD-UNIFORM. cold-path 균일화 (mmap_permute 양봉 53/36 ms 해소)

- **Status**: DONE (2026-04-26, `ca7aaeb`, **부분 충족 / 사용자 종결**)
- **결과 요약**: per-layer p50 52.8→51.6 ms (3.2% 초과, 점진 개선). 변동 폭 ±25%→±13.6% (절반 감소). 적용 C(madvise+warmup), 1966 tests PASS. 본질 미충족(SLA <50ms / ±10% 둘 다 미달) 사유로 사용자 종결 결정 — 후속(C2 mlock / C3 async / C4 posix_fadvise)은 ROI 낮아 미진행.
- **Sprint**: closed
- **Dependencies**: 없음 (Phase 4 자산 활용)
- **담당 권장**: Senior Implementer (`secondary_mmap.rs` + AUF mmap demand-paging 영역)
- **추정 작업량**: M (3~5일)
- **Description**:
  - **목적/배경**: 5차 측정 per-layer 양봉 분포 (3 runs cold ~53 ms / 2 runs warm ~36 ms). SLA <50 ms p50 5.7% 초과의 직접 원인. AUF mmap demand-paging이 cold path에서 발생하는 것으로 추정 (`mmap_permute` stage가 5-stage instrumentation에서 가장 큰 변동 보임).
  - **scope (in)**:
    - 후보 A: `madvise(MADV_WILLNEED)` prefault — AUF mmap 직후 호출
    - 후보 B: 명시적 warmup pass — swap 전 더미 read로 페이지 캐시 hit
    - 후보 C: A+B 결합 + per-layer 단계 측정 재실행 비교
    - 5-stage instrumentation 재활용하여 mmap_permute stage delta 정량화
  - **scope (out)**:
    - UMA zero-copy (Phase 3.7c 별도 trail)
    - AUF 포맷 변경
    - cl_mem 통합 작업 (WSWAP-5-TBT-DIAG 영역)
- **Acceptance Criteria**:
  - per-layer p50 < 50 ms (SLA 충족) **또는** cold/warm 변동 폭 < ±10%
  - 5차 측정 대비 mmap_permute stage p50 단조 감소 (정량 표 갱신)
  - INV-122 회귀 없음 (greedy sanity 통과)
  - clippy clean, fmt clean
  - 호스트 generate 회귀 없음
  - 리포트: `results/data/weight_swap/phase_5_cold_path.md`
- **Notes**:
  - 참고 파일: `engine/src/models/weights/swap_executor.rs` (5-stage instrumentation), `engine/src/models/weights/secondary_mmap.rs`
  - 자동 커밋 + `notify-send`.
  - madvise는 Android 커널 버전별 동작 상이 가능 (R-new-3 참고). MADV_PAGEOUT/MADV_COLD와 혼동 주의.

## [P1] WSWAP-5-INV122-SWEEP. INV-122 full 100-prompt sweep

- **Status**: DONE (2026-04-26, `6e1979d`, **측정 성공 / 임계값 FAIL → spec 조정**)
- **결과 요약**: 500 runs (100 prompt × 5 ratio), wall-clock 69분, 정확성 회귀 0건. ratio=1.0에서 NMSE p50=0.00964 PASS / p95=0.03744 FAIL, Δtop-1=76.56pp FAIL. **본질 진단**: 32-token greedy decode 누적 drift로 단일-token 기준 임계값과 미스매치 (ratio=0.25에서도 Δtop-1=44.85pp). 사용자 결정 — 옵션 A(단일-token NMSE 재정의) 채택, Architect에게 spec 위임 (WSWAP-5-INV122-SPEC).
- **Sprint**: closed
- **Dependencies**: 없음 (Phase 4 자산 + 신규 AUF 활용)
- **담당 권장**: Tester (측정) + Architect (spec 임계값 조정 시)
- **추정 작업량**: S (1~2일, 측정 시간 제외)
- **Description**:
  - **목적/배경**: 5차 측정에서 time-budget으로 greedy sanity 3건만 진행. INV-122 v2 정식 acceptance 미충족.
  - **scope (in)**:
    - 100-prompt sweep으로 ratio=1.0 mixed AUF의 통계적 정확성 검증
    - 측정: NMSE (logits), top-5 overlap, top-1 match, ROUGE-L (생성 텍스트)
    - ratio={0.25, 0.5, 0.75, 1.0} 각각
    - 4차/5차 결과 합본 표 작성 (`results/data/weight_swap/phase_4_accuracy.md`에 5th sweep 추가, 또는 `phase_5_accuracy.md` 신설)
    - INV-122 v2 임계값(NMSE ≤ 0.01, Δtop-1 ≤ 1pp) 충족 여부 판정
  - **scope (out)**:
    - PPL 별도 측정 (QCF-PPL 상관 기존 증명)
    - Qwen 등 타 모델 (Llama 3.2 1B 단일 타겟)
- **Acceptance Criteria**:
  - 100 prompt × 4 ratio 측정 완료
  - NMSE ≤ 0.01 + Δtop-1 ≤ 1pp 통계적 검증 (ratio=1.0)
  - 임계값 미달 시 spec/41-invariants.md INV-122 임계값 조정 PR 분리 (Architect 협업)
  - 리포트: `results/data/weight_swap/phase_5_accuracy_sweep.md` 또는 `phase_4_accuracy.md` 5th sweep 섹션
- **Notes**:
  - 자산: 디바이스 신규 AUF (`Llama-3.2-1B-Instruct.auf`, sha256 `1a1ead0c...`, mtime 2026-04-26), `experiments/prompts/`
  - V10 thermal isolation 준수
  - 100-prompt 측정은 디바이스 wall-clock 시간이 길 수 있음 — 필요 시 백그라운드 배치

## [P1] WSWAP-5-INV122-SPEC. INV-122 v2.1 spec 재정의 (단일-token NMSE)

- **Status**: DONE (2026-04-26, `709bb51`)
- **결과 요약**: spec/32 §3.12.6 재구성(§3.12.6.1~11), spec/41 INV-122 카탈로그 v2.1 라벨, arch/weight_swap v7→v8 §5.1 확장 + §5.1.0/§5.1.8 신설(decoupling principle). 임계값(NMSE ≤ 0.01, Δtop-1 ≤ 1pp) 변동 없이 측정 단위만 단일-token 고정. 32-token decode/ROUGE-L/top-5는 보조 sanity FYI. Phase 4 자료(mean=0.0062) 재사용 PASS 명시. cross-ref 정합 검증 완료.
- **Sprint**: closed
- **후속 위임**: Implementer — `engine/tests/spec/test_inv_122_mixed_precision.rs` docstring v2.1 명시 (코드 변경 없으면 docstring만)
- **Dependencies**: WSWAP-5-INV122-SWEEP (DONE) — sweep 결과로 임계값-측정 미스매치 확정
- **담당**: Architect (spec/arch 문서만, 코드 수정 없음)
- **추정 작업량**: S (1일)
- **Description**:
  - Sprint A INV-122 sweep 결과 32-token decode 누적 drift로 단일-token 기준 임계값(NMSE ≤ 0.01, Δtop-1 ≤ 1pp)과 미스매치 발견
  - 사용자 결정 — 옵션 A: 단일 next-token NMSE로 재정의 (학술 표준, Phase 4 mean=0.0062 자료 재사용)
  - **scope**:
    - spec/41-invariants.md INV-122 v2.1 갱신 (단일-token + 측정 protocol)
    - spec/32-engine-algorithms.md §3.12.6 본문 갱신 (책임 분리: 단일-token vs decode window)
    - arch/weight_swap.md §5.1 측정 방법론 보완 (decoupling principle)
    - changelog 추가
    - cross-reference 정합 검증 (`grep -r INV-122 spec/ arch/`)
- **Acceptance Criteria**:
  - spec INV-122 v2.1 단일-token 재정의 완료
  - Phase 4 자료(mean=0.0062, +0.33pp)가 신 spec 기준 PASS임 명시
  - cross-ref 정합
  - 자동 커밋
  - spec 테스트 코드 docstring 갱신은 후속 (Implementer 별도 위임 시 표시)
- **Notes**:
  - INV-131/132/133/134 (AUF 관련) 영향 없음
  - 보조 metric (ROUGE-L, top-5 overlap) 임계값 게이트 아닌 회귀 보조용으로 명시

## [P2] WSWAP-5-TBT-DIAG. ratio=1.0 mixed −20.7% TBT gap 진단 (cl_mem fragmentation 가설)

- **Status**: DONE (2026-04-26, Senior Implementer Sprint B 완료. 가설 (c) 복합 원인으로 판정. 후속 sprint 후보 3개 design note 작성. `results/data/weight_swap/phase_5_tbt_diag.md`)
- **Sprint**: current
- **Dependencies**: 없음 (Phase 4 측정 자산 활용)
- **담당 권장**: Senior Implementer (OpenCL backend / cl_mem 통계 영역) + Tester (per-layer profiling)
- **추정 작업량**: M (3~5일, 진단까지)
- **Description**:
  - **목적/배경**: ratio=1.0 mixed Decode TBT가 Q4 baseline 대비 −20.7% (느림). 가설: AUF SOA bypass에서도 16 layers × N tensors 별도 cl_mem 패턴이 attention slope를 증가시킬 가능성.
  - **참고 메모**: `project_kv_fragmentation.md` — KV cache cl_mem fragmentation은 56개 별도 buffer가 HeadMajor attention slope +1.32 μs/n_kv 증가 (llama.cpp는 단일 cl_mem). 본 weight swap에서는 16 layers × N tensors 별도 cl_mem.
  - **scope (in)**:
    - AUF SOA path에서 cl_mem 개수 측정 (count + total bytes)
    - steady-state attention timing per-layer profiling (TBT 분해)
    - llama.cpp 단일 cl_mem 패턴과 비교 (reference)
    - 가설 검증 결과 도출:
      - (a) cl_mem fragmentation이 −20.7% gap의 주 원인 → 해소 sprint scope 정의 (후속)
      - (b) 다른 원인 (예: register spill, scheduler overhead) → 원인 문서화
  - **scope (out)**:
    - 해소 구현 자체는 후속 sprint (이 sprint는 진단까지)
    - cl_mem 통합 리팩토링 (별도 P1 sprint로 승급 시점에 결정)
- **Acceptance Criteria**:
  - cl_mem 개수 + 분포 표 (per-layer breakdown)
  - per-layer attention timing 분해 (4차 baseline vs ratio=1.0 mixed)
  - 가설 판정: fragmentation 주 원인 / 부분 원인 / 무관 중 하나로 결론
  - 후속 sprint scope 1-2 페이지 design note (해소 방안 outline + 추정 effort)
  - 리포트: `results/data/weight_swap/phase_5_tbt_diag.md`
- **Notes**:
  - 4차 측정 baseline은 `phase_4_*.md` 재활용
  - llama.cpp cl_mem 패턴 참고: `reference_llama_cpp_source.md`
  - `--profile` 금지. wall-clock 기준 (`feedback_opencl_profile_events_cross_engine.md` 준수)
  - 진단 결과에 따라 후속 sprint는 P1로 승급 가능 (해소 implementation)

## [P1] WSWAP-5-AUF-PLACEHOLDER-DROP. AUF SOA bypass placeholder cl_mem 112개 제거

- **Status**: IN_PROGRESS (2026-04-26, Sprint B 후속, Senior Implementer 의뢰)
- **Sprint**: current
- **Dependencies**: WSWAP-5-TBT-DIAG (DONE) — 진단 결과에 근거
- **담당 권장**: Senior Implementer (`swap_executor.rs` materialise + LayerWeights 분기)
- **추정 작업량**: S (1~2일)
- **Description**:
  - **목적/배경**: TBT-DIAG에서 ratio=1.0 mixed의 −20.7% TBT gap 본질이 alive cl_mem 257개 (F16 primary 145 + AUF placeholder 112)임을 확정. matmul_qkv +22% (TLB cold). 본 작업은 placeholder 112개 제거.
  - **현재 코드** (`swap_executor.rs:664-723` `materialise_auf_soa_weight`):
    - AUF SOA bytes를 `copy_weight_from()`으로 GPU에 올림 → cl_mem 1개 신규 할당
    - forward path는 이 cl_mem을 절대 안 읽음 (SOA registry의 q_buf/d_buf 사용)
    - 단순히 LayerWeights.wq 같은 Tensor 슬롯 채우기용 placeholder
  - **scope (in)**:
    - LayerWeights에서 AUF SOA bypass 모드일 때 placeholder Tensor 생성 우회
    - 옵션 후보:
      - 1) `LayerWeights.w*: Option<Tensor>` (None이면 SOA registry key로 lookup)
      - 2) lightweight stub Tensor (cl_mem 없이 metadata만)
      - 3) Registry key 직접 보관 (`Either<Tensor, SoaRegistryKey>`)
    - 권장: 옵션 2 (구조 변경 최소, forward path 변경 적음)
    - INV-131 safety net 영향 점검 — placeholder가 fallback 시 필요한지 확인
    - 진단 instrumentation (`LLMRS_CL_MEM_DIAG=1`) 활용하여 0개 검증
  - **scope (out)**:
    - F16 primary cl_mem 145개 제거 (PRIMARY-DROP 별도 sprint)
    - AUF 포맷 변경
    - SOA registry 자체 변경
- **Acceptance Criteria**:
  - AUF SOA bypass 모드(ratio=1.0 mixed)에서 placeholder cl_mem **0개** (`LLMRS_CL_MEM_DIAG=1` dump 검증)
  - matmul_qkv μs/call 감소 (Q4 baseline 437 μs/call에 근접 목표, 부분 회복도 PASS)
  - 정확성 회귀 없음 ("Paris" 가드, garbage 출력 없음)
  - 호스트 sanity PASS (cargo test + clippy + fmt)
  - INV-131 safety net 회귀 없음 (해당 spec test 통과)
  - 디바이스 N≥3 측정으로 TBT 변화 정량
  - 리포트: `results/data/weight_swap/phase_5_placeholder_drop.md` (또는 phase_5_tbt_diag.md에 §추가)
- **Notes**:
  - 효과 추정: matmul_qkv +22% 해소 시 −20.7% gap의 ~53% 회복
  - Phase 4 핸드오프에서 "placeholder cl_mem 디버깅 함정" 으로 미리 표기됨 — 본 작업으로 해소
  - 자동 커밋 + `notify-send`

## [P2] WSWAP-5-PRIMARY-DROP. F16 primary cl_mem 145개 lifecycle audit + release

- **Status**: TODO (PLACEHOLDER-DROP 결과 보고 후 진행 결정)
- **Sprint**: next
- **Dependencies**: WSWAP-5-AUF-PLACEHOLDER-DROP — 결과 측정으로 추가 필요성 판단
- **담당 권장**: Senior Implementer (lifecycle audit) + Architect (swap-back 정책 결정)
- **추정 작업량**: M (3~5일)
- **Description**:
  - **목적/배경**: ratio=1.0 mixed에서 F16 primary cl_mem 145개가 alive 상태 점유. 이미 사용 안 됨. 명시적 release 미구현.
  - **scope (in)**:
    - Lifecycle audit: F16 primary cl_mem alive에 의존하는 코드 전체 검사
      - swap-back (Q4→F16 reverse) 경로
      - INV-131 SOA 재변환 safety net (Phase 3.7a)
      - QCF importance collection
    - swap 완료 시점에 fully-swapped layer의 primary cl_mem 명시적 release
    - swap-back 시 primary 재로드 비용 측정
    - 정책 결정: "ratio=1.0에서만 release vs fully-swapped layer 단위로 release"
  - **scope (out)**:
    - mmap GGUF 자체 release (이전 `acf01af`/`9795ab7`에서 일부 처리됨)
    - PLACEHOLDER-DROP과 동시 진행 금지 (회귀 원인 분리 위해)
- **Acceptance Criteria**:
  - F16 primary cl_mem ratio=1.0에서 ≤ 16개 (또는 0개, lifecycle 결정에 따라)
  - swap-back 정확성 회귀 없음
  - INV-131 safety net 회귀 없음
  - Total alive bytes Q4 baseline (~2.0 GB) 수준 회복
  - 추가 TBT 개선폭 측정
  - 리포트: `results/data/weight_swap/phase_5_primary_drop.md`
- **Notes**:
  - 사용자 결정 사항: PLACEHOLDER-DROP만으로 −20.7% gap 충분 해소되면 본 작업 보류 가능
  - swap-back 비용 vs alive cl_mem 점유 trade-off
  - 회귀 risk 중 — lifecycle 변경은 광범위

## [P3] WSWAP-5-KIVI-FALLBACK. KIVI plan mixed state legacy fallback 진단

- **Status**: TODO
- **Sprint**: next
- **Dependencies**: 없음 (Phase 3.7a 발견 사항)
- **담당 권장**: Implementer (plan path 영역) + Architect (legacy fallback 수용 결정 시)
- **추정 작업량**: S (1~2일, 진단 + 결정)
- **Description**:
  - **목적/배경**: Phase 3.7a 부수 발견. mixed F16+Q4_0 state에서 KIVI plan path가 legacy fallback로 떨어짐. 정확성은 OK이나 성능 저하.
  - **scope (in)**:
    - 원인 진단: 어느 조건에서 plan path가 legacy fallback로 떨어지는지 (mixed state 감지 로직 / plan invalidation 트리거 / kernel selection)
    - 영향 측정: legacy fallback의 TBT 영향 (현재 기본 ratio={0.25, 0.5, 1.0}에서)
    - 결정:
      - (a) plan path 수정 (mixed state 지원 추가)
      - (b) legacy fallback 수용 + spec 명시
    - 결정에 따라 후속 작업 정의 (수정 시 별도 sprint, 수용 시 본 sprint로 spec 갱신 마무리)
  - **scope (out)**:
    - plan path 대규모 리팩토링 (수정 결정 시 별도 sprint)
    - KIVI 알고리즘 자체 변경
- **Acceptance Criteria**:
  - 원인 진단 1 페이지 design note (분기 조건 + 호출 stack)
  - legacy fallback TBT 영향 정량 표
  - 결정 (수정 / 수용) + 근거 기록
  - 수용 결정 시: spec 갱신 PR (Architect 협업)
  - 리포트: `results/data/weight_swap/phase_5_kivi_fallback.md` (또는 인라인 design note)
- **Notes**:
  - 참고: Phase 3.7 핸드오프 메모 (`project_weight_swap_phase37_handoff.md`)에 해당 기록 확인 필요
  - 사용자 가치 낮음 (정확성 OK, 성능만 저하) → 우선순위 P3
  - 4개 sprint 중 가장 후순위. 다른 3개 머지 후 진행 가능.

---

# Phase 5 우선순위 추천 (PM, 2026-04-26)

> 4개 항목의 우선순위 ranking과 병행 가능 여부.

## Ranking

1. **WSWAP-5-COLD-UNIFORM (P1, 1순위)** — cold-path 균일화
   - **사용자 가치**: 매우 높음. SLA <50 ms p50 5.7% 초과의 직접 해소. Phase 4 border-line acceptance를 정상 acceptance로 격상.
   - **구현 risk**: 중. madvise/warmup은 표준 패턴이나 Android 커널 버전별 차이 존재 (R-new-3).
   - **기존 측정 자료 활용도**: 매우 높음. 5-stage instrumentation + 5차 측정 표 그대로 재사용.
   - **추정 effort**: M (3~5일).

2. **WSWAP-5-INV122-SWEEP (P1, 2순위)** — INV-122 full sweep
   - **사용자 가치**: 높음. Phase 4 정식 acceptance의 마지막 미충족 게이트. 통계적 정확성 보장.
   - **구현 risk**: 낮음. 측정 작업이며 코드 변경 거의 없음.
   - **기존 측정 자료 활용도**: 높음. 신규 AUF 디바이스 자산 + experiments/prompts 그대로 활용.
   - **추정 effort**: S (1~2일, 디바이스 측정 시간 제외).

3. **WSWAP-5-TBT-DIAG (P2, 3순위)** — TBT gap 진단
   - **사용자 가치**: 중. ratio=1.0 mixed의 Decode 처리량 회복은 production 가치 있으나, ratio=1.0은 critical scenario 한정 → 우선순위 차순.
   - **구현 risk**: 낮음 (진단까지). 해소 구현은 후속 sprint로 분리되어 risk 격리.
   - **기존 측정 자료 활용도**: 높음. 4차 baseline + llama.cpp 비교 자료 그대로.
   - **추정 effort**: M (3~5일, 진단까지).

4. **WSWAP-5-KIVI-FALLBACK (P3, 4순위)** — KIVI mixed fallback
   - **사용자 가치**: 낮음. 정확성 OK, 성능만 저하. 사용자 영향 제한적.
   - **구현 risk**: 낮음 (진단까지).
   - **기존 측정 자료 활용도**: 중. 별도 측정 필요.
   - **추정 effort**: S (1~2일, 진단 + 결정).

## 병행 가능 여부

| 조합 | 병행 가능 | 근거 |
|------|----------|------|
| 1 + 2 | ✅ 권장 | 코드 영역 무관 (cold-path는 swap_executor/secondary_mmap, sweep은 측정 작업). 디바이스 측정 시간 슬롯이 겹칠 수 있으나 prompt set만 분리하면 동시 진행 가능. |
| 1 + 3 | ✅ 가능 | cold-path는 mmap demand-paging, TBT-DIAG는 cl_mem fragmentation. 코드 영역 다름. 단 둘 다 디바이스 측정 자원 사용 → 시간 분할 필요. |
| 2 + 3 | ✅ 가능 | 둘 다 측정 중심. 디바이스 시간 슬롯 분배만 필요. |
| 1 + 4 | ✅ 가능 | 코드 영역 무관. KIVI는 plan path, cold-path는 mmap. |
| 모든 항목 동시 | ⚠️ 비권장 | 디바이스 측정 자원 경합 + 결과 해석 시 변수 분리 어려움. 1+2 우선 진행 후 3+4 후속 권장. |

## 권장 진행 순서

```
Sprint A (1주):  WSWAP-5-COLD-UNIFORM (Senior Impl)
              + WSWAP-5-INV122-SWEEP (Tester)         ← 병행
                ↓ 머지 후
Sprint B (1주):  WSWAP-5-TBT-DIAG (Senior Impl + Tester)
                ↓ 머지 후
Sprint C (3일):  WSWAP-5-KIVI-FALLBACK (Implementer)
```

- Sprint A 완료 시점에 Phase 4 정식 acceptance 충족 가능 (SLA 50 ms 달성 + INV-122 v2 sweep 통과).
- Sprint B 결과에 따라 cl_mem 통합 sprint를 P1로 승급 가능 (사용자 결정 사항).
- Sprint C는 사용자 가치가 낮아 다른 feature 작업과 병행하거나 backlog로 미룰 수 있음.

---

# 이번 스프린트 최우선 5개 작업 (PM 추천, 2026-04-25 Phase 3.7 갱신)

> Phase 1~3.6는 `54017ed` 시점에 완료. Phase 3.7-SPEC 산출물(spec/arch/docs 6개)은 Architect 본 작업으로 완료. 이번 스프린트는 Phase 3.7a (런타임 safety net) + Phase 3.7b (AUF 도입) IMPL 단계 진행.

1. **WSWAP-3.7A-IMPL** — Senior Implementer: SOA 재변환 호출 배선 (Senior Implementer, 1~2일) [최우선]
   - 이유: Phase 3.6 후 발견된 디바이스 정확성 버그(Paris 정답 + 후속 garbage)의 즉각적 해결. AUF 의존 없이 단독 머지 가능. Phase 4 진입 전 필수.
   - 산출물: `engine/src/models/weights/swap_executor.rs`에 convert_aos_to_soa 호출 + register_noshuffle_soa 등록. INV-122 디바이스 검증.
2. **WSWAP-3.7A-TEST** — Implementer: SOA 재변환 spec 테스트 (Implementer, 0.5일)
   - 이유: TDD 친화. IMPL과 병행 가능. mock OpenCLBackend로 cl_mem 등록 검증.
3. **WSWAP-3.7B-AUF-CRATE** — Implementer: AUF reader/writer 모듈 신설 (Implementer, 2~3일)
   - 이유: AUF v0.1 핵심 인프라. Header/SectionTable parse/serialize, AufReader/Writer/Stripper. CLI/Engine 연동의 토대.
   - 의존: WSWAP-3.7B-SPEC (완료). 3.7A와 독립 진행 가능.
4. **WSWAP-3.7B-CLI** — Implementer: auf-tool binary (Implementer, 1~2일)
   - 이유: build/info/strip/verify 4개 서브커맨드. 워크스테이션 빌드 도구. Phase 3.7b의 사용자 인터페이스.
   - 의존: WSWAP-3.7B-AUF-CRATE.
5. **WSWAP-3.7B-ENGINE** — Implementer: Engine `--secondary-source` AUF 분기 (Implementer, 1일)
   - 이유: Phase 3.7b 통합의 마무리. AUF cache hit 경로로 SOA 재변환 비용 0 검증.
   - 의존: WSWAP-3.7B-AUF-CRATE + WSWAP-3.7A-IMPL (둘 다 완료된 후가 안전 — convert_aos_to_soa fallback 경로 검증을 위해).

**스프린트 목표**:
- 단기 (1주): Phase 3.7a 머지 → Phase 4 디바이스 실측 진입 가능.
- 중기 (2~3주): Phase 3.7b 머지 → swap latency 추가 감소 + self-contained 자산 인프라.

**작업 순서 / 의존**:
- 직렬: WSWAP-3.7A-IMPL (단독 가능) → Phase 4 진입 가능.
- 직렬: WSWAP-3.7B-AUF-CRATE → (WSWAP-3.7B-CLI, WSWAP-3.7B-ENGINE) → WSWAP-3.7B-TEST.
- 병렬: 3.7A 묶음과 3.7B 묶음은 코드 영역이 다르므로 독립 진행 권장 (Senior + Implementer 병행).

**TEST 작업** (WSWAP-3.7A-TEST, WSWAP-3.7B-TEST)은 각각의 IMPL과 같은 sprint에서 병행 작성 (TDD 친화).

---

# 리스크 및 가정

## 신규 리스크 (동적 노선)

| ID | 리스크 | 영향 | 완화 방안 |
|----|--------|------|----------|
| R-new-1 | `TransformerWeights` → `Arc<LayerSlot>` 리팩토링이 forward hot-path 회귀 유발 (Arc 간접 참조 오버헤드) | 고 | WSWAP-1-SLOT Acceptance에서 tok/s 회귀 ≤ 1% 강제. `ArcSwap`/lock-free snapshot 우선, `Mutex` 금지. Phase 4 WSWAP-4-TBT에서 최종 검증. |
| R-new-2 | Prefill-tail 측정이 첫 prefill에만 발동되면 반복 쿼리 시 측정 기회 제한 | 중 | Architect WSWAP-1-SPEC에서 fallback policy 확정. 기본안: "매 prefill마다 stale check 후 재측정 가능, decode 중 신호는 다음 prefill까지 대기하되 K token 초과 시 uniform fallback". |
| R-new-3 | `madvise(MADV_DONTNEED)` 동작이 Android 커널 버전별 상이 (즉시 회수 안 될 수 있음) | 중 | WSWAP-4-PSS에서 실측 필수. `MADV_PAGEOUT` (Android Q+) 병행 검토. 필요 시 munmap/remap fallback. |
| R-new-4 | Manager-Engine 프로토콜 확장(신규 variant)이 기존 verify 하네스와 호환 깨짐 | 저 | WSWAP-3-SIGNAL에서 기존 시나리오 회귀 테스트. `#[serde(deny_unknown_fields)]` 여부 사전 확인. |

## 유효 유지 (기존 리스크 중 동적 노선에서도 적용)

| ID | 리스크 | 영향 | 완화 방안 |
|----|--------|------|----------|
| R4 (유지) | Swap latency가 50 ms/layer SLA 초과. Q/K permutation은 **매 swap마다 재발생** | 중 | WSWAP-4-LATENCY에서 per-단계 breakdown. SLA 초과 시 "사전 permuted Q4_0 파일" 옵션 재검토 (offline 전처리로 복구 가능). |
| R6 (유지) | Mixed precision forward divergence | 중 | INV-122 유지. WSWAP-2-TEST 호스트 회귀 + WSWAP-4-INV122 실측 임계값 조정. |

## 가정

- **가정 1**: F16 + Q4_0 두 GGUF는 동일 tensor shape, 동일 metadata, 동일 layer 구조. (llama-quantize 표준 출력 가정)
- **가정 2**: Llama 3.2 1B의 16 layer 모두 독립적으로 교체 가능 (cross-layer weight tying 없음).
- **가정 3**: Android UFS 4.0 순차 read > 3 GB/s → 150 MB layer가 50 ms 내 로드 가능 (이론값, 실측은 WSWAP-4).
- **가정 4**: `ImportanceTable`이 layer importance 기반 선택에 충분한 signal 제공. per-head 세분화는 이번 스코프 외.
- **가정 5**: KV swap(`SwapHandler` 스텁)과 weight swap은 독립 handler. 동시 활성화 시 서로 간섭 없음 (KV swap 미구현 상태이므로 이번 스코프 내 안전).
- **가정 6**: swap은 단방향 (F16→Q4_0). Q4_0→F16 복귀는 이번 스코프 외 — 필요 시 engine 재시작으로 리셋.

---

# Architect에게 의뢰할 Spec 재작성 항목

Phase 1 진입 직전 Architect 작업 (WSWAP-1-SPEC 단일 작업으로 묶임):

1. **`arch/weight_swap.md` 전면 재작성**
   - 기존 Phase A (정적) 서술 삭제
   - Phase 1–4 구조 + Mermaid 컴포넌트 다이어그램 (feedback_mermaid_diagrams.md 준수)
   - Component-centric 서술 (feedback_arch_component_centric.md 준수)
   - 흐름: Manager signal → Resilience → WeightSwapHandler → (측정/결정/실행) → LayerSlot swap
   - `ratio_generation` 재진입 차단 흐름 명시
2. **`spec/33-engine-data.md` 업데이트**
   - `ENG-DAT-091` (LayerDtypeProfile TOML) **폐기 표기**
   - 신규 `LayerSlot` 데이터 구조 spec ID 할당
   - Secondary mmap handle 메타데이터 구조
3. **`spec/32-engine-algorithms.md` 업데이트**
   - `ENG-ALG-210` 재작성 (동적 전제)
   - 신규 `WeightSwapHandler` 알고리즘 spec ID
   - 신규 on-demand `ImportanceCollector` 활성화 정책 spec ID
   - 신규 `SwapDecider` (ratio → layer set) 알고리즘 spec ID
4. **`spec/41-invariants.md` 업데이트**
   - `INV-122` (mixed precision 정확성) 유지. 실측 전 권장값: NMSE ≤ 0.01, top-5 overlap ≥ 0.9, top-1 match ≥ 0.95
   - 신규 INV: swap 실행 중 forward 재진입 금지 (`INV-121` 강화 또는 신규)
   - 신규 INV: `LayerSlot::current_dtype` ↔ 로드된 실제 weight dtype 일치
   - 신규 INV: secondary mmap handle 생존 보장 (`TransformerWeights` lifetime 동안 유지)
5. **Shared 프로토콜 spec (Manager ↔ Engine)**
   - `MSG-NNN` 신규 ID 할당: `ResilienceAction::SwapWeights { ratio: f32 }`
   - ratio 범위 (0.0~1.0), clamp 규칙
   - MemoryStrategy 매핑 (Warning/Critical/Emergency → ratio 값)
6. **Fallback Policy 결정 (Architect 판단)**
   - Prefill-tail 측정 "마지막 N 토큰" N 값 (권장: N=1, 근거 spec에 기록)
   - Decode 중 신호 수신 시 fallback: "다음 prefill 대기" 기본. 단 K 토큰(제안: K=512) 초과 시 uniform ratio 적용
   - 측정 없는 상태에서 신호 수신 시 `SwapDecider`의 uniform fallback 경로
7. **`tests/spec/` 테스트 요구사항 초안**
   - 각 신규 INV에 대응하는 테스트 파일 경로/이름 제안 (feedback_spec_tests_required.md 준수, inline `#[cfg(test)]`만으로는 불충분)

---

# 사용자 확인 필요한 미결 사항

> Phase 1~3.6 완료 시점에 1~2, 6번 항목은 구현 선택지로 해소. 아래는 Phase 3.7 진입 이후 잔존 미결 항목.

1. ~~**Arc snapshot 구체 구현 선택**~~ — `Arc<LayerSlot>` 채택 완료 (`07b8fa3`).
2. ~~**Fallback policy의 K 값**~~ — K=512 채택 완료.
3. **`--force-swap-ratio` CLI 플래그 유지 범위**: 디버그 전용이지만 feat/weight merge 후에도 유지할지, Phase 4 완료 시 제거할지. 기본 제안: feature flag로 보호하여 유지 (현장 디버깅 용도). **결정 보류**.
4. **KV swap handler와의 상호작용**: 현재 KV `SwapHandler`는 스텁 상태. Weight swap 우선으로 진행하되, 향후 KV swap 구현 시 handler 실행 순서 (weight 먼저 vs KV 먼저) 결정 필요. Architect가 spec에 기록. **결정 보류**.
5. **Qwen 2.5 1.5B 포함 여부**: 기존 초안에서는 검증 대상이었으나 재설계 전제에는 Llama 3.2 1B만 명시. Phase 4 실측에 Qwen 포함 여부 확인 필요. **결정 보류**.
6. ~~**Phase 3.6 SOA invalidation 범위**~~ — 전체 `clear_noshuffle_soa_registry()` 채택 완료 (`54017ed`).
7. **Phase 3.7c UMA zero-copy 진입 시점**: AUF (3.7b) 머지 후 swap latency 측정 결과 50ms/layer 이하면 미루고, 초과하면 Phase 4 직후 우선 처리. **Phase 4 실측 결과 기반 결정**.
8. **AUF Repacker 도입 시점**: Phase 5에서 도입 권장이나, 사용자 운영 중 strip 후 variant 추가 요구가 빈번하면 v0.2에서 우선 추가. **사용자 피드백 후 결정**.
9. **AUF 자동 strip / 자동 cache 정책**: v0.1은 수동 strip만. 향후 사용자 피드백 후 결정 (DF-37-10).
10. **AUF auf-tool 위치 (engine/src/bin vs 별도 crate)**: Architect 권고 = workspace 단순성 위해 `engine/src/bin/auf_tool.rs`. 단, v1.0 stable 후 binary 격리 필요성 재평가. **현재 결정 = engine 내부 bin**.

---

# 변경 이력

- 2026-04-24 (초안): Phase A/B/C 정적 설계 기반 작성.
- 2026-04-24 (재작성): 사용자 의도 재확인 결과 **정적 프로파일 노선 전면 폐기**. 동적 swap 노선으로 Phase 1–4 구조 재설계. 폐기: `quantize_profile` 바이너리, `LayerDtypeProfile` TOML, `--layer-dtype-profile` CLI. 신규: `LayerSlot` + `SwapExecutor` + `WeightSwapHandler` + `SwapDecider` + `ResilienceAction::SwapWeights { ratio }`. QCF 측정은 prefill-tail 1회 + on-demand 플래그 모드. INV-122 유지, `ENG-DAT-091` 폐기.
- 2026-04-24 (Phase 1~3 완료): 커밋 `07b8fa3`. spec 340 pass, clippy --all-targets clean. Phase 1~3 모든 task DONE 처리. handoff 메모: `project_weight_swap_phase3_handoff.md`.
- 2026-04-25 (Phase 3.5 진입): 사용자 결정 DF-35-1~4 확정. 기존 Phase 3.5 placeholder(INVESTIGATE/DESIGN/IMPL) 폐기 후 4-task 구체화 — SPEC(ENG-ALG-219/220 + INV-129) → ARCH(weight_swap.md v5 §2.2.1 + plan_partition_integration.md A.11) → TEST(test_eng_alg_219_plan_invalidation.rs) → IMPL(FullKernelPlan 캡처/execute 비교, forward_into underscore 제거, lazy rebuild, DF-35-3 상호 배타). 신규 Phase 3.6 추가 — SOA(OpenCLBackend noshuffle_soa_registry invalidation, INV-130 신설). Phase 4는 3.5+3.6 머지 후 진입으로 BLOCKED 처리.
- 2026-04-25 (Phase 3.7 진입): 커밋 `54017ed`로 Phase 3.5/3.6 완료. S25 실측 swap 후 첫 토큰 "Paris" 정답 확인했으나 후속 토큰 garbage 발견 (Adreno noshuffle SOA 누락 부분 변환). Phase 3.7로 이행. 사용자 결정 DF-37-1~11 확정 — AUF (Argus Unified Format) v0.1 신설, Mode B (self-contained) 단일, B-2 multi-variant + selective strip, 3-tier 버저닝(semver + capability + section table), 256B 헤더, section table 48B/entry, hybrid source_hash, 64KB align, JSON-in-binary META, BPE tokenizer 보존. Phase 3.7a (런타임 SOA 재변환 safety net, ENG-ALG-222 + INV-131) + Phase 3.7b (AUF v0.1 + auf-tool CLI, ENG-DAT-096 + ENG-ALG-223 + INV-132~134) + Phase 3.7c (선택, UMA zero-copy, DEFERRED). Repacker는 Phase 5로 미룸. Architect 산출물 6개 + TODO 1개 작성 완료.
- 2026-04-26 (Phase 4 종결 + Phase 5 신설): 커밋 `2900f5f`로 Phase 4 종결. SOA bypass 본격 구현 + 5차 디바이스 측정으로 per-layer p50 −74.4% (206→52.8 ms), soa_reconvert −90.1% (172→17 ms) 달성. 단 SLA <50 ms는 p50 5.7% 초과 border-line으로 사용자 수용. Phase 4 작업 4개 상태 갱신 — WSWAP-4-LATENCY/PSS/TBT DONE, WSWAP-4-INV122 PARTIAL (full sweep은 WSWAP-5-INV122-SWEEP로 이관). Phase 5 신설하여 잔여 4개 항목을 별도 sprint로 분리: (1) WSWAP-5-COLD-UNIFORM (cold-path 균일화, P1, M), (2) WSWAP-5-INV122-SWEEP (100-prompt sweep, P1, S), (3) WSWAP-5-TBT-DIAG (cl_mem fragmentation 가설 진단, P2, M), (4) WSWAP-5-KIVI-FALLBACK (KIVI mixed legacy fallback, P3, S). 권장 진행 순서: Sprint A(1+2 병행) → Sprint B(3) → Sprint C(4). 핸드오프: `project_weight_swap_phase4_handoff.md`. 측정 자료: `results/data/weight_swap/phase_4_*.md`.
