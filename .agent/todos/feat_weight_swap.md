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

| Phase | 내용 | 예상 작업 수 | 예상 기간 | 우선순위 |
|-------|------|-----|----------|----------|
| **1 — 인프라** | `LayerSlot` 구조, secondary 파일 mmap handle 보관, 기존 단일 dtype 초기 로드 경로 보존 (회귀 방지) | 5 | 5–7일 | P1 |
| **2 — Swap 실행** | `WeightSwapHandler` + `SwapExecutor` + `ratio_generation` 재진입 차단. 수동 CLI/테스트 트리거 | 5 | 6–8일 | P1 |
| **3 — Manager 연동** | `ResilienceAction::SwapWeights { ratio }` + on-demand `ImportanceCollector` + `SwapDecider` + MemoryStrategy | 5 | 5–7일 | P2 |
| **4 — 실측/튜닝** | Galaxy S25에서 PSS / swap latency / TBT 영향 실측, INV-122 임계값 조정 | 4 | 4–6일 | P2 |

**진행 순서 강제**: Phase 1 → 2 → 3 → 4. Phase 1의 `LayerSlot` 도입이 forward 경로 전체에 영향을 주므로 가장 먼저, 그리고 성능 회귀 없음을 확인한 후 Phase 2 진입.

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

# Phase 1 — 인프라 (LayerSlot + Secondary mmap)

> **목표**: `TransformerWeights`를 `LayerSlot` 기반으로 리팩토링하되 **단일 F16 경로가 기존과 동일**하게 동작. Secondary Q4_0 GGUF의 mmap handle만 보관 (lazy, forward 미사용).
> **예상 기간**: 5–7일 (5개 작업)
> **기대 효과**: swap의 토대. 단독으로는 PSS 감소 없음. 이 단계에서 성능 회귀가 없음을 확인한 후 Phase 2 진입.

## [P0] WSWAP-1-SPEC. Architect: Phase 1 Spec/Arch 재작성
- **Status**: TODO
- **Sprint**: current
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
- **Status**: TODO
- **Sprint**: current
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
- **Status**: TODO
- **Sprint**: current
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
- **Status**: TODO
- **Sprint**: current
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
- **Status**: TODO
- **Sprint**: current
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

# Phase 2 — Swap 실행 (Handler + Executor + 재진입 차단)

> **목표**: `WeightSwapHandler` + `SwapExecutor` 구현. 수동 CLI/테스트 트리거로 F16 → Q4_0 layer swap이 bit-정확(target dtype 기준) / 안전하게 동작. Manager 연동 없이 내부 API만.
> **예상 기간**: 6–8일 (5개 작업)
> **기대 효과**: swap 경로 검증. 이 단계에서 정확성/동시성 안전성 확보 후 Phase 3에서 신호 연동.

## [P1] WSWAP-2-EXEC. `SwapExecutor` 구현 (Arc swap + madvise)
- **Status**: TODO
- **Sprint**: current
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
- **Status**: TODO
- **Sprint**: current
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
- **Status**: TODO
- **Sprint**: current
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

## [P1] WSWAP-2-CLI-TRIGGER. 수동 swap 트리거 CLI (디버그 전용)
- **Status**: TODO
- **Sprint**: current
- **Dependencies**: WSWAP-2-HANDLER
- **담당 권장**: Implementer
- **Description**:
  - `generate` 바이너리에 `--force-swap-ratio <0.0-1.0>` 디버그 플래그 추가 (optional, hidden)
  - prefill 완료 후 즉시 (decode 시작 전) handler를 수동 호출하여 swap 발동
  - layer importance는 임시로 uniform 또는 layer idx 기반 (정식 measurement은 Phase 3)
  - 정식 배포 빌드에서는 `#[cfg(debug_assertions)]` 또는 feature flag로 보호
- **Acceptance Criteria**:
  - `generate --model-path <f16> --model-path-secondary <q4_0> --force-swap-ratio 0.5` 실행 성공
  - swap 후 decode 진행 → 텍스트 생성 완료 (coherence 검증)
  - 플래그 없는 호출에 영향 없음
- **Notes**: Phase 3에서 manager 연동 완료 후 이 플래그는 유지해도 되고 제거해도 됨 (테스트 유용성).

## [P1] WSWAP-2-TEST. Swap 정확성 테스트 스위트
- **Status**: TODO
- **Sprint**: current
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

# Phase 3 — Manager 연동 (Signal + On-demand QCF + Decider)

> **목표**: `ResilienceAction::SwapWeights { ratio }` signal이 실제 manager → engine 경로를 타고 `WeightSwapHandler`를 발동. On-demand `ImportanceCollector`로 prefill tail에서 1회 QCF 측정 → `SwapDecider`가 ratio를 layer 집합으로 변환.
> **예상 기간**: 5–7일 (5개 작업)
> **기대 효과**: 신호 기반 end-to-end 동작. Phase 4 실측의 전제.

## [P2] WSWAP-3-SIGNAL. `ResilienceAction::SwapWeights { ratio }` + Shared 프로토콜
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Phase 2 완료
- **담당 권장**: Implementer
- **Description**:
  - `shared/` 크레이트에 `ResilienceAction::SwapWeights { ratio: f32 }` variant 추가 (serde)
  - `SystemSignal` → `ResilienceAction` 매핑 로직에 신규 variant 반영
  - manager → engine 프로토콜 문서 업데이트
  - verify 하네스 (`verify/scenarios/`) 호환성 확인: 기존 시나리오가 신규 variant로 인해 깨지지 않는지 점검
  - Spec: `MSG-NNN` 신규 ID 할당 (Architect WSWAP-1-SPEC에서 예약)
- **Acceptance Criteria**:
  - shared 크레이트 unit test (serialize/deserialize roundtrip)
  - verify 하네스 기본 시나리오 pass
  - `mock_manager` / `mock_engine`이 신규 variant를 받아 no-op (handler 미설치 시) 또는 정상 dispatch
- **Notes**: R-new-4 완화. verify 스키마 변경 최소화.

## [P2] WSWAP-3-QCF. On-demand `ImportanceCollector` 활성화 플래그 모드
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: WSWAP-1-SPEC (Architect가 측정 정책 결정)
- **담당 권장**: Implementer (QCF 확장) + Senior Implementer (hot path 영향 검토)
- **Description**:
  - `ImportanceCollector`에 `active: AtomicBool` 플래그 추가
  - 평시 `active = false` → forward 경로에서 hot-path 분기 early-return (zero overhead 원칙)
  - Prefill 종료 직후 조건: (a) secondary handle 존재, (b) swap 신호 pending, (c) 이번 prefill이 첫 측정 또는 stale → `active = true` set
  - 마지막 N 토큰 (spec 결정값)에서 layer importance 수집
  - 수집 완료 후 `active = false` → `ImportanceTable` freeze 후 `SwapDecider`에 전달
  - Decode 중 신호 수신 시: 다음 prefill까지 대기. fallback policy는 Architect spec 결정 (예: K 토큰 초과 시 uniform distribution)
- **Acceptance Criteria**:
  - active=false일 때 forward 측정 오버헤드 ≤ 0.5% (microbench, 6T)
  - active=true일 때 prefill tail N 토큰에서 per-layer score 수집 확인
  - 호스트 unit test: 모의 prefill → score 수집 → freeze → retrieve
  - fallback policy 테스트 (spec 결정값 기준)
- **Notes**: R-new-2. "첫 prefill에만 측정"의 한계 — fallback 없으면 반복 쿼리 시 측정 기회 제한. Architect가 spec에 명시한 정책을 정확히 반영.

## [P2] WSWAP-3-DECIDER. `SwapDecider` 구현 (ratio → layer set)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: WSWAP-3-QCF, WSWAP-2-HANDLER
- **담당 권장**: Implementer
- **Description**:
  - `SwapDecider::select(ratio: f32, table: &ImportanceTable) -> Vec<usize>`
  - 알고리즘: layer importance 오름차순 정렬 → 하위 `floor(ratio × n_layers)` 선택
  - 이미 Q4_0인 layer 제외 (누적 swap 대응)
  - tie-breaking 규칙 (idx 오름차순) 명시
  - importance table 부재 시 fallback: uniform (layer idx 순) 또는 spec 결정
- **Acceptance Criteria**:
  - unit test: 16-layer + ratio=0.25 → 4개 layer 선택, 하위 4개 importance 일치
  - edge case: ratio=0.0 (빈 vec), ratio=1.0 (전체), ratio=0.75 + 이미 일부 swap됨
  - fallback 경로 테스트
- **Notes**: 간단하지만 swap 정책의 핵심. 향후 per-head / per-tensor 세분화 여지는 별도 feature.

## [P2] WSWAP-3-STRATEGY. `MemoryStrategy` 매핑 확장
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: WSWAP-3-SIGNAL, WSWAP-3-DECIDER
- **담당 권장**: Implementer
- **Description**:
  - `engine/src/resilience/strategy/memory.rs` — `MemoryStrategy::react()`에 `SwapWeights { ratio }` 반환 경로 추가
  - 매핑 (Architect spec 참조, 예시 값):
    - Warning → ratio=0.0 (no swap) 또는 ratio=0.25
    - Critical → ratio=0.5
    - Emergency → ratio=1.0
  - 기존 Evict와 중첩 여부: spec에서 결정 (Architect)
  - `policy_default.lua` 연동은 후행 작업 (수동 주입 우선)
- **Acceptance Criteria**:
  - MemoryLevel → ratio 매핑 단위 테스트
  - 기존 Evict 경로 회귀 없음
  - verify 하네스 시나리오 추가 (MemoryCritical → swap 발동)
- **Notes**: 기존 `project_dpp_policy.md` 참조. Lua policy 통합은 Phase 4 이후.

## [P2] WSWAP-3-E2E. End-to-End 통합 테스트 (mock manager → engine)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: WSWAP-3-SIGNAL, WSWAP-3-QCF, WSWAP-3-DECIDER, WSWAP-3-STRATEGY
- **담당 권장**: Tester
- **Description**:
  - `mock_manager` → `mock_engine` 경유로 `SwapWeights { ratio }` 주입
  - 시나리오: (1) prefill → importance 측정 → 신호 수신 → swap 발동 → decode 진행
  - 시나리오: (2) 평시 → 신호 없음 → 측정/swap 미발동 (zero overhead 확인)
  - 시나리오: (3) decode 중 신호 → 다음 prefill까지 대기 → 측정 → swap
  - verify 하네스에 신규 시나리오 YAML 추가
- **Acceptance Criteria**:
  - 3개 시나리오 모두 pass
  - 로그에서 각 단계 이벤트 확인 (`CacheEvent::WeightSwapped`, `ImportanceCollector activated`)
  - 호스트 CI 경로에서 통합 테스트 실행 가능 (Android 디바이스 없이)
- **Notes**: verify 하네스 확장. 기존 시나리오 호환성 유지 필수.

---

# Phase 4 — 실측 / 튜닝 (Galaxy S25)

> **목표**: Galaxy S25에서 PSS 감소량 / swap latency / TBT 영향 / INV-122 임계값을 실측하고 spec 값 조정.
> **예상 기간**: 4–6일 (4개 작업)
> **기대 효과**: production 적용 가능성 판단 + spec 값 확정.

## [P2] WSWAP-4-LATENCY. Swap latency 실측 (Galaxy S25)
- **Status**: TODO
- **Sprint**: next
- **Dependencies**: Phase 3 완료
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

# 이번 스프린트 최우선 5개 작업 (PM 추천)

1. **WSWAP-1-SPEC** — Architect Spec/Arch 재작성 (Architect, 2–3일)
   - 이유: 폐기된 정적 전제가 spec 여러 곳에 섞여 있음. 구현 착수 전 반드시 동적 노선 기준으로 재정리. `LayerSlot` 데이터 구조, `SwapWeights { ratio }` 메시지, on-demand `ImportanceCollector` 활성화 정책, fallback policy, INV 3종 신설 필요.
2. **WSWAP-1-SLOT** — `LayerSlot` + `TransformerWeights` 리팩토링 (Senior Implementer, 3–4일)
   - 이유: forward 경로 전체에 영향. R-new-1 (Arc 간접 참조 회귀) 완화 여부가 Phase 2/3 전체 의미를 결정. 이 작업이 회귀 없이 완료되어야 후속 진행 가치 있음.
3. **WSWAP-1-MMAP** — Secondary GGUF mmap handle 인프라 (Implementer, 1–2일)
   - 이유: SLOT 작업과 병행 가능. secondary 파일 open/shape 검증/lazy slice 메타데이터 준비.
4. **WSWAP-1-CLEANUP** — 폐기 코드 제거 (Implementer, 0.5–1일)
   - 이유: 정적 프로파일 경로(`quantize_profile`, `LayerDtypeProfile`, `--layer-dtype-profile`)를 명시적으로 삭제하여 잔여 혼선 방지. feat/weight 브랜치 정리 차원.
5. **WSWAP-2-EXEC** — `SwapExecutor` 구현 (Senior Implementer, 3–4일)
   - 이유: Phase 2의 핵심 단위. Arc swap + madvise + Q/K permutation을 단일 원자 연산으로 묶는 실행기. Phase 1 완료 직후 착수.

**스프린트 목표**: Phase 1 완료 + Phase 2의 `SwapExecutor` 착수. 즉 "LayerSlot 구조가 forward 회귀 없이 도입되고, Secondary mmap이 lazy로 붙으며, 수동 trigger로 swap이 동작"하는 상태까지.

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

1. **Arc snapshot 구체 구현 선택**: `Arc<LayerSlot>` + `Mutex` vs `arc_swap::ArcSwap` vs custom generation-counter 방식. Senior Implementer가 WSWAP-1-SLOT 착수 전 PoC로 비교 측정 권장. 사용자 선호가 있으면 알려주세요.
2. **Fallback policy의 K 값 (decode 중 신호 수신 시 강제 uniform 트리거 기준)**: 제안 K=512. 실제 사용 시나리오(장문 생성 vs 단문 응답)에 따라 조정 필요. Architect가 spec에서 결정하되 사용자 검토 요청.
3. **`--force-swap-ratio` CLI 플래그 유지 범위**: 디버그 전용이지만 feat/weight merge 후에도 유지할지, Phase 4 완료 시 제거할지. 기본 제안: feature flag로 보호하여 유지 (현장 디버깅 용도).
4. **KV swap handler와의 상호작용**: 현재 KV `SwapHandler`는 스텁 상태. Weight swap 우선으로 진행하되, 향후 KV swap 구현 시 handler 실행 순서 (weight 먼저 vs KV 먼저) 결정 필요. Architect가 spec에 기록.
5. **Qwen 2.5 1.5B 포함 여부**: 기존 초안에서는 검증 대상이었으나 재설계 전제에는 Llama 3.2 1B만 명시. Phase 4 실측에 Qwen 포함 여부 확인 필요.

---

# 변경 이력

- 2026-04-24 (초안): Phase A/B/C 정적 설계 기반 작성.
- 2026-04-24 (재작성): 사용자 의도 재확인 결과 **정적 프로파일 노선 전면 폐기**. 동적 swap 노선으로 Phase 1–4 구조 재설계. 폐기: `quantize_profile` 바이너리, `LayerDtypeProfile` TOML, `--layer-dtype-profile` CLI. 신규: `LayerSlot` + `SwapExecutor` + `WeightSwapHandler` + `SwapDecider` + `ResilienceAction::SwapWeights { ratio }`. QCF 측정은 prefill-tail 1회 + on-demand 플래그 모드. INV-122 유지, `ENG-DAT-091` 폐기.
