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
| **3.5 — plan invalidation 배선** | `_entry_ratio_generation` → `FullKernelPlan` plan rebuild 통합. PlanInvalidated lazy fallback. tensor_partition × swap 상호 배타 | 4 | 3–4일 | **CURRENT** | P1 |
| **3.6 — Noshuffle SOA registry coherence** | SwapExecutor가 OpenCLBackend `noshuffle_soa_registry` invalidate 추가. Q4_0 swap 시 stale cl_mem key 제거 | 1 | 1–2일 | NEW | P1 |
| **4 — 실측/튜닝** | Galaxy S25에서 PSS / swap latency / TBT 영향 실측, INV-122 임계값 조정 | 4 | 4–6일 | BLOCKED (3.5 + 3.6 머지 후) | P2 |

**진행 순서 강제**: Phase 1 → 2 → 3 → **3.5 → 3.6** → 4.
- Phase 3.5와 3.6은 독립이지만 둘 다 Phase 4 디바이스 실측 진입 전 머지 필수.
- 3.5는 plan rebuild 배선만 다룸. 3.6은 OpenCL backend의 SOA registry invalidation을 별도로 처리 (Architect 결정 DF-35-4).

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
- **Status**: TODO
- **Sprint**: current
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

# Phase 4 — 실측 / 튜닝 (Galaxy S25)

> **목표**: Galaxy S25에서 PSS 감소량 / swap latency / TBT 영향 / INV-122 임계값을 실측하고 spec 값 조정.
> **예상 기간**: 4–6일 (4개 작업)
> **기대 효과**: production 적용 가능성 판단 + spec 값 확정.
> **블로커**: Phase 3.5(plan invalidation) + Phase 3.6(SOA registry) 머지 후 진입.

## [P2] WSWAP-4-LATENCY. Swap latency 실측 (Galaxy S25)
- **Status**: BLOCKED
- **Sprint**: next
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
- **Status**: TODO
- **Sprint**: next
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
- **Status**: TODO
- **Sprint**: next
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
- **Status**: TODO
- **Sprint**: next
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

# 이번 스프린트 최우선 5개 작업 (PM 추천, 2026-04-25 갱신)

> Phase 1~3는 `07b8fa3` 시점에 완료. 이번 스프린트는 Phase 3.5(plan invalidation 배선) + Phase 3.6(SOA registry coherence) 마무리에 집중.

1. **WSWAP-3.5-SPEC** — Architect: ENG-ALG-219/220 + INV-129 신설 (Architect, 0.5일) [선행]
   - 이유: ID 발급 + 본문이 있어야 ARCH/TEST/IMPL 모두 cross-ref 가능. 가장 먼저 처리.
2. **WSWAP-3.5-ARCH** — Architect: arch v5 §2.2.1 + INV-120/INV-129 비교표 (Architect, 0.5일)
   - 이유: SPEC과 동시 또는 직후. IMPL에 진입하기 전 component-centric 서술과 DF-35 결정 4건 모두 문서에 반영.
3. **WSWAP-3.5-TEST** — Implementer: spec 테스트 5케이스 신규 (Implementer, 1일)
   - 이유: SPEC/ARCH 완료 후 IMPL과 병행 가능. TDD 친화적으로 IMPL이 초록을 만드는 구조.
4. **WSWAP-3.5-IMPL** — Implementer: plan 캡처 + execute 비교 + dispatch 배선 (Implementer, 1~2일)
   - 이유: Phase 3.5의 핵심 구현. plan.rs 시그니처 변경 + forward_into 인자 underscore 제거 + lazy fallback + DF-35-3 상호 배타 강제.
5. **WSWAP-3.6-SOA** — Senior Implementer: OpenCL Noshuffle SOA registry invalidation (Senior Implementer, 1~2일)
   - 이유: 디바이스 한정 silent correctness bug. Phase 4 실측 신뢰성의 전제. Phase 3.5와 독립이라 병렬 처리 가능. INV-130 신설 + SwapExecutor 호출 배선 + 단위 테스트.

**스프린트 목표**: Phase 3.5 + Phase 3.6 머지 → Phase 4 디바이스 실측 진입 가능 상태.

**작업 순서 / 의존**:
- 직렬: WSWAP-3.5-SPEC → WSWAP-3.5-ARCH → (WSWAP-3.5-TEST, WSWAP-3.5-IMPL 병렬) → 머지
- 병렬: WSWAP-3.6-SOA는 Phase 3.5와 독립 진행 (Senior Implementer 별도 자원)
- WSWAP-3.5-TEST와 WSWAP-3.5-IMPL은 같은 작업 묶음(IMPL이 TEST를 통과시킴) — TDD 방식 권장하나 강제는 아님

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

> Phase 1~3 완료 시점에 1~2번 항목은 구현 선택지로 해소(`Arc<LayerSlot>` 채택, fallback K=512). 아래는 Phase 3.5 진입 이후 잔존 미결 항목.

1. ~~**Arc snapshot 구체 구현 선택**~~ — `Arc<LayerSlot>` 채택 완료 (`07b8fa3`).
2. ~~**Fallback policy의 K 값**~~ — K=512 채택 완료.
3. **`--force-swap-ratio` CLI 플래그 유지 범위**: 디버그 전용이지만 feat/weight merge 후에도 유지할지, Phase 4 완료 시 제거할지. 기본 제안: feature flag로 보호하여 유지 (현장 디버깅 용도). **결정 보류**.
4. **KV swap handler와의 상호작용**: 현재 KV `SwapHandler`는 스텁 상태. Weight swap 우선으로 진행하되, 향후 KV swap 구현 시 handler 실행 순서 (weight 먼저 vs KV 먼저) 결정 필요. Architect가 spec에 기록. **결정 보류**.
5. **Qwen 2.5 1.5B 포함 여부**: 기존 초안에서는 검증 대상이었으나 재설계 전제에는 Llama 3.2 1B만 명시. Phase 4 실측에 Qwen 포함 여부 확인 필요. **결정 보류**.
6. **Phase 3.6 SOA invalidation 범위**: per-layer entry 제거 vs 전체 clear 트레이드오프. Senior Implementer가 WSWAP-3.6-SOA 착수 시 backend 내부 데이터 구조 보고 결정 — registry 크기와 swap 빈도 비교해서 전체 clear가 단순/안전하면 후자 채택 권장. Architect 사전 검토 권장.

---

# 변경 이력

- 2026-04-24 (초안): Phase A/B/C 정적 설계 기반 작성.
- 2026-04-24 (재작성): 사용자 의도 재확인 결과 **정적 프로파일 노선 전면 폐기**. 동적 swap 노선으로 Phase 1–4 구조 재설계. 폐기: `quantize_profile` 바이너리, `LayerDtypeProfile` TOML, `--layer-dtype-profile` CLI. 신규: `LayerSlot` + `SwapExecutor` + `WeightSwapHandler` + `SwapDecider` + `ResilienceAction::SwapWeights { ratio }`. QCF 측정은 prefill-tail 1회 + on-demand 플래그 모드. INV-122 유지, `ENG-DAT-091` 폐기.
- 2026-04-24 (Phase 1~3 완료): 커밋 `07b8fa3`. spec 340 pass, clippy --all-targets clean. Phase 1~3 모든 task DONE 처리. handoff 메모: `project_weight_swap_phase3_handoff.md`.
- 2026-04-25 (Phase 3.5 진입): 사용자 결정 DF-35-1~4 확정. 기존 Phase 3.5 placeholder(INVESTIGATE/DESIGN/IMPL) 폐기 후 4-task 구체화 — SPEC(ENG-ALG-219/220 + INV-129) → ARCH(weight_swap.md v5 §2.2.1 + plan_partition_integration.md A.11) → TEST(test_eng_alg_219_plan_invalidation.rs) → IMPL(FullKernelPlan 캡처/execute 비교, forward_into underscore 제거, lazy rebuild, DF-35-3 상호 배타). 신규 Phase 3.6 추가 — SOA(OpenCLBackend noshuffle_soa_registry invalidation, INV-130 신설). Phase 4는 3.5+3.6 머지 후 진입으로 BLOCKED 처리.
