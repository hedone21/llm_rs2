# INV Coverage Tracker

> 전체: 70개 | ✅ 48 | ⬜ 4 | 🔶 17 (INV-130 추가, 2026-04-25 Weight Swap Phase 3.6 Noshuffle SOA coherence)

## 범례

- ✅ 테스트 구현 완료
- ⬜ 테스트 미구현
- 🔶 제약사항 (static 검증 전용, 자동 테스트 제외)

---

## System/Component (INV-001 ~ INV-018)

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-001 | 시스템 = 2 독립 프로세스. Engine-Manager 직접 코드 의존 금지. | Safety | 🔶 | (static: Cargo workspace 의존 구조) |
| INV-002 | NEON SIMD는 ARM64에서만 활성화. | Safety | 🔶 | (static: `#[cfg(target_arch)]`) |
| INV-003 | `config.json`의 `architectures`가 지원 목록에 없으면 로딩 거부. | Correctness | ✅ | `engine/tests/spec/test_inv_003.rs` |
| INV-004 | QCF 수집 활성 상태에서 lossy action 실행 시 QcfMetric 생성 필수. | Correctness | ✅ | `engine/tests/spec/test_inv_004_017.rs` |
| INV-005 | Manager 장애가 Engine 추론 루프를 중단시키지 않음. | Safety | ✅ | `engine/tests/spec/test_inv_005_006.rs` |
| INV-006 | Engine 장애가 Manager 모니터링 루프를 중단시키지 않음. | Safety | ✅ | `engine/tests/spec/test_inv_005_006.rs` |
| INV-010 | Engine-Manager 직접 코드 의존 금지. Shared가 유일한 공유 의존성. | Safety | 🔶 | (static: Cargo.toml) |
| INV-011 | Shared는 Engine/Manager 내부 구현에 의존 금지. | Safety | 🔶 | (static: Cargo.toml) |
| INV-012 | Backend trait이 유일한 하드웨어 추상화점. Backend 우회 직접 호출 금지. | Correctness | 🔶 | (static: 코드 리뷰) |
| INV-013 | Monitor 스레드 장애가 다른 Monitor에 전파 금지. | Safety | 🔶 | (static, test: 아키텍처) |
| INV-014 | EngineDirective.seq_id는 세션 내 단조 증가. | Correctness | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| INV-015 | Capability는 세션당 정확히 1회 전송. | Correctness | ✅ | `engine/tests/spec/test_inv_015.rs` |
| INV-016 | 동일 배타 그룹 액션 동시 활성화 금지. | Correctness | ✅ | `manager/tests/spec/test_inv_016.rs` |
| INV-017 | QCF 수집 활성 + lossy action 실행 시 QcfMetric 생성 필수. (=> INV-004) | Correctness | ✅ | `engine/tests/spec/test_inv_004_017.rs` |
| INV-018 | 추론 루프(Prefill/Decode)는 단일 스레드. | Safety | 🔶 | (static: 아키텍처) |

## Protocol (INV-020 ~ INV-028)

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-020 | seq_id 단조 증가: `seq_id(N+1) > seq_id(N)`. | Correctness | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| INV-021 | 동일 seq_id 재사용 금지. | Correctness | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| INV-022 | 모든 Directive는 정확히 1개 Response를 유발. | Correctness | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| INV-023 | `CommandResponse.seq_id == EngineDirective.seq_id`. | Correctness | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| INV-024 | `len(CommandResponse.results) == len(EngineDirective.commands)`. | Correctness | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| INV-025 | `len(CommandResponse.results) == len(EngineDirective.commands)`. (=> INV-024) | Correctness | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| INV-026 | Engine은 수신한 seq_id에 대해서만 Response 전송. | Correctness | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| INV-027 | Shared serde 어노테이션 변경 = 프로토콜 버전 변경. | Compatibility | 🔶 | (static: 코드 리뷰) |
| INV-028 | 새 필드 추가 시 `#[serde(default)]` 필수. 하위 호환 유지. | Compatibility | 🔶 | (static: 코드 리뷰) |

## Manager Algorithm (INV-030 ~ INV-051)

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-030 | `can_act = false`일 때 integral 미변경. | Correctness | ✅ | `manager/tests/spec/test_inv_030_031.rs` |
| INV-031 | `integral in [0, integral_clamp]` 항상 유지. | Correctness | ✅ | `manager/tests/spec/test_inv_030_031.rs` |
| INV-032 | 에스컬레이션은 즉시. Normal에서 Critical 직행 가능. | Correctness | ✅ | `manager/tests/spec/test_inv_032_033.rs` |
| INV-033 | 디에스컬레이션은 반드시 1단계씩. | Correctness | ✅ | `manager/tests/spec/test_inv_032_033.rs` |
| INV-034 | `warning_release < warning_threshold`. | Correctness | ✅ | `manager/tests/spec/test_inv_034_036.rs` |
| INV-035 | `critical_release < critical_threshold`. | Correctness | ✅ | `manager/tests/spec/test_inv_034_036.rs` |
| INV-036 | `warning_threshold < critical_threshold`. | Correctness | ✅ | `manager/tests/spec/test_inv_034_036.rs` |
| INV-037 | Warning 모드에서 Lossy 액션 선택 금지. | Correctness | ✅ | `manager/tests/spec/test_inv_037_038.rs` |
| INV-038 | 이미 활성 중인 액션은 재선택 금지. | Correctness | ✅ | `manager/tests/spec/test_inv_037_038.rs` |
| INV-039 | Lossless 액션의 cost = 항상 0. | Correctness | ✅ | `manager/tests/spec/test_inv_039_040.rs` |
| INV-040 | QCF 값 없는 Lossy 액션 = INFINITY cost. | Correctness | ✅ | `manager/tests/spec/test_inv_039_040.rs` |
| INV-041 | 동일 배타 그룹 액션은 하나의 조합에 동시 미포함. (=> INV-016) | Correctness | ✅ | `manager/tests/spec/test_inv_041_042.rs` |
| INV-042 | 조합의 총 latency 악화 > latency_budget이면 배제. | Performance | ✅ | `manager/tests/spec/test_inv_041_042.rs` |
| INV-043 | 완전 해소 가능 조합 > best-effort 조합 (항상 우선). | Correctness | ✅ | `manager/tests/spec/test_inv_043_044.rs` |
| INV-044 | parametrize 출력 value는 [range.min, range.max] 범위 내. | Correctness | ✅ | `manager/tests/spec/test_inv_043_044.rs` |
| INV-045 | primary_domain 매핑: SwitchHw/Throttle/LayerSkip -> Compute. | Correctness | 🔶 | (static: 코드) |
| INV-046 | RLS gain vector k = f(P, phi). lambda는 망각 인수. | Correctness | ✅ | `manager/tests/spec/test_inv_046_049.rs` |
| INV-047 | bias는 W 갱신 후 잔여 오차에 EMA(lr=0.1) 적용. | Correctness | ✅ | `manager/tests/spec/test_inv_046_049.rs` |
| INV-048 | P matrix: D x D 대칭 양정치. 초기값 100 * I. | Correctness | ✅ | `manager/tests/spec/test_inv_046_049.rs` |
| INV-049 | `lambda in (0, 1]`. lambda=1.0이면 forgetting 없음. | Correctness | ✅ | `manager/tests/spec/test_inv_046_049.rs` |
| INV-050 | 관찰 relief의 latency 차원 = 항상 0.0. | Correctness | ✅ | `manager/tests/spec/test_inv_050.rs` |
| INV-051 | 동시 적용 시 전체 relief가 각 액션에 귀속 (개별 분리 불가). | Correctness | 🔶 | (static: 설계 한계) |

## Engine Architecture (INV-060 ~ INV-065)

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-060 | `CommandExecutor.poll()`은 토큰당 최대 1회 호출. | Performance | 🔶 | (static: 코드 구조) |
| INV-061 | ExecutionPlan: 생성 즉시 소비, 1회성. | Safety | 🔶 | (static: 코드 구조) |
| INV-062 | Suspend 포함 ExecutionPlan: evict/switch_device/prepare_device = None. | Safety | ✅ | `engine/tests/spec/test_inv_062_064.rs` |
| INV-063 | MessageLoop 스레드는 Transport의 유일한 소유자. | Safety | 🔶 | (static: ownership) |
| INV-064 | heartbeat_interval 내 최소 1회 Heartbeat 전송. | Correctness | ✅ | `engine/tests/spec/test_inv_062_064.rs` |
| INV-065 | Backend trait 구현체는 `Send + Sync`. | Safety | 🔶 | (static: trait bound) |

## Engine State Machine (INV-070 ~ INV-076)

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-070 | `OperatingMode.from_levels()` = 순수 함수. 이전 상태 미의존. | Correctness | ✅ | `engine/tests/spec/test_fsm_operating_mode.rs` |
| INV-071 | EngineState 전이는 CommandExecutor 내부에서만. | Correctness | ✅ | `engine/tests/spec/test_inv_072_076.rs` |
| INV-072 | `resolve_conflicts()`: Suspend 존재 시 반환 = `[Suspend]`. | Safety | ✅ | `engine/tests/spec/test_inv_072_076.rs` |
| INV-073 | `resolve_conflicts()`: RestoreDefaults는 다른 제약 없을 때만. | Correctness | ✅ | `engine/tests/spec/test_inv_072_076.rs` |
| INV-074 | `plan.suspended == true`이면 evict/switch_device/prepare_device = None. | Safety | ✅ | `engine/tests/spec/test_inv_072_076.rs` |
| INV-075 | Resume: compute/memory_level을 Normal로, throttle_delay_ms를 0으로. | Correctness | ✅ | `engine/tests/spec/test_inv_072_076.rs` |
| INV-076 | RestoreDefaults: active_actions 비움, throttle 0, levels Normal. | Correctness | ✅ | `engine/tests/spec/test_inv_072_076.rs` |

## Cross-cutting (INV-080 ~ INV-085)

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-080 | async 런타임 사용 금지. std::thread + mpsc만 허용. | Safety | 🔶 | (static: Cargo.toml, 코드 리뷰) |
| INV-081 | IPC 직렬화는 JSON (serde_json) 전용. | Compatibility | ✅ | `engine/tests/spec/test_inv_081_082.rs` |
| INV-082 | 1:1 단일 클라이언트 연결. 다중 Engine 동시 연결 금지. | Safety | ✅ | `engine/tests/spec/test_inv_081_082.rs` |
| INV-083 | PI Controller output은 [0, 1] 범위 내. | Correctness | ✅ | `manager/tests/spec/test_inv_083_085.rs` |
| INV-084 | ActionSelector = stateless. ReliefEstimator.predict = 읽기 전용. | Correctness | 🔶 | (static: 코드 구조) |
| INV-085 | Normal 모드에서 액션 미발행. | Correctness | ✅ | `manager/tests/spec/test_inv_083_085.rs` |

## Engine Self-Utilization (INV-091 ~ INV-092, 2026-04)

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-091 | `self_cpu_pct`, `self_gpu_pct` ∈ [0.0, 1.0] (Engine 측 clamp). | Correctness | 🆕 미구현 | `engine/tests/spec/test_msg_060_self_util.rs`, `manager/tests/spec/test_mgr_dat_075_076_engine_util.rs` |
| INV-092 | 측정 실패 시 self_cpu_pct/self_gpu_pct = 0.0 fallback, Heartbeat 송출 차단 금지. | Correctness | 🆕 미구현 | `engine/tests/spec/test_msg_060_self_util.rs` |

## Plan × Tensor Partition (INV-120, 2026-04)

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-120 | PartitionStep::run 진입 시 ratio_generation mismatch 시 PlanInvalidated 반환. caller는 재빌드 또는 forward_gen fallback. | Safety/Correctness | 🆕 미구현 | `tests/spec/inv_120_plan_partition_stale.rs` |

## Dynamic Weight Swap (INV-121 ~ INV-128, 2026-04-24)

> 이전 Phase A 정적 노선(TOML `LayerDtypeProfile` + `quantize_profile`)은 2026-04-24에 폐기되었다. ENG-DAT-091 ID는 재사용 금지.

> **Phase 3 Manager 통합 (2026-04-24 추가)**: INV-126~128, ENG-ALG-214-ROUTE, ENG-ALG-215~218, ENG-DAT-095.

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-121 | Forward 재진입 금지. 토큰 진입 시 per-layer Arc snapshot 1회 획득 + 토큰 내내 재사용. mid-token swap은 다음 토큰부터 관측. stale/half-swapped 관찰 0건. | Correctness | 🆕 미구현 | `engine/tests/spec/test_inv_121_swap_reentrancy.rs` |
| INV-122 | Dynamic swap 후 forward: logit NMSE ≤ 0.01, top-5 overlap ≥ 0.9, top-1 match ≥ 0.95 (primary baseline 대비). **layer 간 dtype 혼합은 정상 상태이며 위반 아님.** Llama/Qwen 양쪽. | Correctness | 🆕 미구현 | `engine/tests/spec/test_inv_122_mixed_precision.rs` |
| INV-123 | Swap 단위 = `LayerSlot.weights.store()` 1회 (단일 원자 단계). 토큰 경계 밖 swap은 다음 토큰부터 관측. Partial state 외부 노출 금지. | Safety/Correctness | 🆕 미구현 | `engine/tests/spec/test_inv_123_swap_atomicity.rs` |
| INV-124 | LayerSlot::current_dtype == weights snapshot의 실제 tensor dtype (swap 전후 항상 일치). | Correctness | 🆕 미구현 | `engine/tests/spec/test_inv_124_slot_dtype_consistency.rs` |
| INV-125 | TransformerModel.secondary_mmap(구 TransformerWeights::secondary_mmap)이 Some인 동안 Arc<SecondaryMmap>은 drop되지 않는다. flat 배치 기준. | Safety | 🆕 미구현 | `engine/tests/spec/test_inv_125_secondary_mmap_lifetime.rs` |
| INV-126 | `EngineCommand::SwapWeights.target_dtype`에서 `DtypeTag::Q4_0` 이외 variant는 panic 없이 Rejected 응답. reserved variant payload 호환성 보장. | Safety/Correctness | 🆕 미구현 | `shared/tests/spec/test_inv_126_swap_dtype_reject.rs` |
| INV-127 | `QuantNoiseTable::epsilon(i).is_none()`(NaN 저장) layer는 `WeightSwapDecider`에서 제외. | Correctness | 🆕 미구현 | `engine/tests/spec/test_inv_127_noise_nan_exclusion.rs` |
| INV-128 | `ImportanceCollector`가 Armed/Collecting 상태로 prefill 완료 시 반드시 `QcfEstimate` 송출 + Idle 복귀. 누수 금지. | Correctness | 🆕 미구현 | `engine/tests/spec/test_inv_128_qcf_collector_leak.rs` |

## Weight Swap × Plan Invalidation (INV-129, 2026-04-25 Phase 3.5)

> Phase 3.5에서 도입. `FullKernelPlan` 진입 1회 atomic load 비교 + INV-120(per-partition)과 OR 결합. tensor_partition × weight swap은 상호 배타(DF-35-3).

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-129 | `FullKernelPlan::execute()` 진입 시 `plan.ratio_generation_at_build` ↔ `model.ratio_generation` Acquire 비교, mismatch 시 `PlanInvalidated`. INV-120과 OR 결합. weight swap/partition re-prep 모두 trigger. | Safety/Correctness | 🆕 미구현 | `engine/tests/spec/test_eng_alg_219_plan_invalidation.rs` |

## Weight Swap × Noshuffle SOA Coherence (INV-130, 2026-04-25 Phase 3.6)

> Phase 3.6에서 도입. SwapExecutor batch 종료 시 `OpenCLBackend::noshuffle_soa_registry` invalidate. **디바이스(Adreno 830) 한정 발현** — 호스트는 SOA registry가 비어 있어 관측 불가, 수동 디바이스 검증 필수.

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-130 | Q4_0 weight swap으로 cl_mem이 교체되면 `OpenCLBackend::noshuffle_soa_registry`의 stale entry는 swap 직후 invalidate되어야 한다 (전체 clear 또는 per-layer 제거). 디바이스 한정 silent correctness bug. | Correctness | 🆕 미구현 | `engine/tests/spec/test_inv_130_noshuffle_soa_coherence.rs` + 디바이스 동작 확인 (manual) |

---

# Part II — 행위 명세 (PREFIX-NNN) 추적

> 추적 대상: ~62개 | ✅ 49 | ⬜ 1

## 선별 기준

| 분류 | 설명 | 예시 |
|------|------|------|
| (A) Pseudocode | PRE/POST가 있는 함수/알고리즘 | eviction, PI controller |
| (B) Formula | 수학 공식, 계산식 | EnergyConstraint 수식, 타이밍 관계 |
| (C) Transition Table | 상태 전이 완전 열거 | OperatingMode FSM, ConnectionState |
| (D) Field Spec | 필드명, 타입, 범위가 구체적인 데이터 구조 | Frame 구조, Config 기본값 |
| (E) Sequence | 단계별 시퀀스 정의 | Handshake, Steady-State |

## Protocol

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| PROTO-010 | (D) | Frame 구조 (4-byte BE length prefix) | ✅ | `engine/tests/spec/test_proto_010_062.rs` |
| PROTO-012 | (D) | MAX_PAYLOAD 64KB 가드 | ✅ | `engine/tests/spec/test_proto_010_062.rs` |
| PROTO-042 | (C) | Connection 3-state FSM (Listening/Connected/Disconnected) | ✅ | `engine/tests/spec/test_proto_042_073.rs` |
| PROTO-073 | (A) | try_recv 드레인 (while let Ok 배치 처리) | ✅ | `engine/tests/spec/test_proto_042_073.rs` |
| PROTO-074 | (A) | seq_id 단조 증가 생성 | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| PROTO-075 | (D) | Directive-Response 1:1 대응 | ✅ | `engine/tests/spec/test_inv_020_026.rs` |

## Message (Shared)

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| MSG-010 | (D) | ManagerMessage serde round-trip | ✅ | `shared/tests/spec/test_msg_010_100.rs` |
| MSG-011 | (D) | EngineMessage 4종 serde | ✅ | `shared/tests/spec/test_msg_010_100.rs` |
| MSG-020 | (D) | EngineDirective serde | ✅ | `shared/tests/spec/test_msg_010_100.rs` |
| MSG-030 | (D) | EngineCommand 13종 serde | ✅ | `shared/tests/spec/test_msg_010_100.rs` |
| MSG-034b | (D) | KvMergeD2o serde round-trip | ✅ | `shared/tests/spec/test_msg_010_100.rs` |
| MSG-035 | (D) | KvStreaming serde round-trip | ✅ | `shared/tests/spec/test_msg_010_100.rs` |
| MSG-060 | (D) | EngineStatus 18필드 serde + self_cpu_pct/self_gpu_pct 하위호환 (default=0.0) | 🆕 | `engine/tests/spec/test_msg_060_self_util.rs` |
| MSG-067 | (A) | self_cpu_pct 계산식(/proc/self/stat + CLK_TCK + num_cpus) + clamp | 🆕 | `engine/tests/spec/test_msg_060_self_util.rs` |
| MSG-068 | (D) | self_gpu_pct Phase 1 placeholder (항상 0.0) | 🆕 | `engine/tests/spec/test_msg_060_self_util.rs` |
| MSG-069 | (D) | ctx.engine.cpu_pct/gpu_pct LuaPolicy 노출 계약 | 🆕 | `manager/tests/spec/test_mgr_dat_075_076_engine_util.rs` |
| MSG-042 | (D) | `EngineCommand::SwapWeights { ratio, target_dtype: DtypeTag }` serde + dispatch contract (ENG-ALG-214-ROUTE) | 🆕 | `shared/tests/spec/test_msg_042_swap_weights_cmd.rs` |
| MSG-080 | (D) | [DEPRECATED 재정의됨, 2026-04-24 Phase 3]: Phase 2 초안의 `ResilienceAction::SwapWeights` serde 항목. shared crate에서는 `EngineCommand::SwapWeights`(MSG-042)로 흡수되고, engine 내부 `ResilienceAction::SwapWeights`는 Rust-only 타입으로 테스트 대상 아님. MSG-080 ID는 새 용도 할당 없음. | — | — |
| MSG-081 | (D) | [DEPRECATED, 2026-04-24 Phase 3]: 구 `CommandResponse::WeightSwapped` variant 제안. Phase 3에서는 별도 `WeightSwapReport` EngineMessage로 대체(MSG-089). MSG-081 ID 폐기. | — | — |
| MSG-082 | (D) | `DtypeTag` enum (Q4_0 유효, F16/F32/Q8_0 reserved) serde round-trip | 🆕 | `shared/tests/spec/test_msg_082_dtype_tag.rs` |
| MSG-088 | (D) | QcfEstimate `layer_swap` 필드 확장 (LayerSwapEstimate: per_layer_importance + per_layer_noise + qcf_swap_at_ratio). `#[serde(default)]` 전방 호환. | 🆕 | `shared/tests/spec/test_msg_088_qcf_estimate_layer_swap.rs` |
| MSG-089 | (D) | `EngineMessage::WeightSwapReport` + `LayerSwapEntry` serde + 순서 보증 (CommandResponse → WeightSwapReport) | 🆕 | `shared/tests/spec/test_msg_089_weight_swap_report.rs` |

## Sequence

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| SEQ-020 | (E) | Handshake 시퀀스 | ✅ | `engine/tests/spec/test_seq_020_035.rs` |
| SEQ-030 | (E) | Steady-State 루프 | ✅ | `engine/tests/spec/test_seq_020_035.rs` |
| SEQ-040 | (E) | Pressure Escalation 시퀀스 | ✅ | `engine/tests/spec/test_seq_040_064.rs` |
| SEQ-095 | (E) | RequestQcf 시퀀스 | ✅ | `engine/tests/spec/test_seq_095_098.rs` |
| SEQ-096 | (E) | QcfEstimate 응답 | ✅ | `engine/tests/spec/test_seq_095_098.rs` |

## Manager Algorithm

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| MGR-ALG-010 | (B) | PI Controller 비례+적분 계산 | ✅ | `manager/tests/spec/test_mgr_alg_010_014.rs` |
| MGR-ALG-011 | (A) | Gain Scheduling (구간별 Kp) | ✅ | `manager/tests/spec/test_mgr_alg_010_014.rs` |
| MGR-ALG-012 | (A) | Anti-Windup (can_act=false 시 적분 동결) | ✅ | `manager/tests/spec/test_mgr_alg_010_014.rs` |
| MGR-ALG-013 | (D) | PI 인스턴스 파라미터 (Kp/Ki/setpoint) | ✅ | `manager/tests/spec/test_mgr_alg_010_014.rs` |
| MGR-ALG-013a | (A) | Memory 임계값 직접 매핑 (Descending ThresholdEvaluator) | ✅ | `manager/tests/spec/test_mgr_alg_013a_016.rs` |
| MGR-ALG-014 | (A) | Measurement Normalization (CPU/온도/메모리 → [0,1]) | ✅ | `manager/tests/spec/test_mgr_alg_013a_016.rs` |
| MGR-ALG-015 | (B) | EnergyConstraint → compute 보조 압력 수식 | ✅ | `manager/tests/spec/test_mgr_alg_013a_016.rs` |
| MGR-ALG-016 | (A) | Elapsed dt 계산 (첫 호출=기본값, 후속=실측) | ✅ | `manager/tests/spec/test_mgr_alg_013a_016.rs` |

## Manager State

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| MGR-050 | (C) | OperatingMode FSM (Normal/Warning/Critical) | ✅ | `manager/tests/spec/test_mgr_050_054.rs` |
| MGR-055 | (C) | OperatingMode 하강 hold_time | ✅ | `manager/tests/spec/test_mgr_050_054.rs` |
| MGR-060 | (C) | ConnectionState FSM (Listening/Connected/Disconnected) | ✅ | `manager/tests/spec/test_mgr_060_061.rs` |
| MGR-061 | (C) | ConnectionState 재연결 (Disconnected→Connected) | ✅ | `manager/tests/spec/test_mgr_060_061.rs` |
| MGR-067 | (C) | ThresholdEvaluator Ascending 에스컬레이션 | ✅ | `manager/tests/spec/test_mgr_067_072.rs` |
| MGR-072 | (C) | ThresholdEvaluator Descending 에스컬레이션 | ✅ | `manager/tests/spec/test_mgr_067_072.rs` |

## Manager Data

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| MGR-DAT-020 | (D) | Config 최상위 구조 | ✅ | `manager/tests/spec/test_mgr_dat_020_056.rs` |
| MGR-DAT-021 | (D) | PolicyConfig 기본값 | ✅ | `manager/tests/spec/test_mgr_dat_020_056.rs` |
| MGR-DAT-022 | (D) | MemoryMonitorConfig 기본값 | ✅ | `manager/tests/spec/test_mgr_dat_022_024.rs` |
| MGR-DAT-023 | (D) | ThermalMonitorConfig 기본값 | ✅ | `manager/tests/spec/test_mgr_dat_022_024.rs` |
| MGR-DAT-024 | (D) | ComputeMonitorConfig 기본값 | ✅ | `manager/tests/spec/test_mgr_dat_022_024.rs` |
| MGR-DAT-075 | (D) | EngineStatus.self_cpu_pct 의미/범위/측정/실패 fallback | 🆕 | `manager/tests/spec/test_mgr_dat_075_076_engine_util.rs` |
| MGR-DAT-076 | (D) | EngineStatus.self_gpu_pct Phase 1 placeholder (ctx.engine.gpu_pct 노출, 값=0.0) | 🆕 | `manager/tests/spec/test_mgr_dat_075_076_engine_util.rs` |

## Engine State

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| ENG-ST-011 | (A) | OperatingMode worst-wins 결정 | ✅ | `engine/tests/spec/test_fsm_operating_mode.rs` |
| ENG-ST-013 | (C) | OperatingMode 전이 테이블 | ✅ | `engine/tests/spec/test_fsm_operating_mode.rs` |
| ENG-ST-020 | (C) | EngineState 전이 (Idle→Running→Suspended) | ✅ | `engine/tests/spec/test_eng_st_010_035.rs` |
| ENG-ST-021 | (C) | EngineState 전이 (Resume→Running) | ✅ | `engine/tests/spec/test_eng_st_010_035.rs` |
| ENG-ST-031 | (D) | active_actions 추적 | ✅ | `engine/tests/spec/test_eng_st_010_035.rs` |
| ENG-ST-032 | (D) | available_actions 동적 계산 | ✅ | `engine/tests/spec/test_eng_st_032.rs` |
| ENG-ST-033 | (C) | Command 13종 처리 결과 | ✅ | `engine/tests/spec/test_eng_st_010_035.rs` |

## Engine Algorithm

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| ENG-ALG-010 | (A) | H2O Eviction 알고리즘 | ✅ | `engine/tests/spec/test_eng_alg_010_012.rs` |
| ENG-ALG-011 | (A) | Sliding Window Eviction | ✅ | `engine/tests/spec/test_eng_alg_010_012.rs` |
| ENG-ALG-012 | (A) | D2O Compensation | ✅ | `engine/tests/spec/test_eng_alg_010_012.rs` |
| ENG-ALG-020 | (A) | KIVI 양자화 | ✅ | `engine/tests/spec/test_eng_alg_020_022.rs` |
| ENG-ALG-051 | (B) | Unified QCF (attention output perturbation 통합 메트릭) | ⬜ | 미구현 |
| ENG-ALG-200 | (A) | GPU Plan × Tensor Partition 협업 (PartitionStep, FfnVariant::Partitioned) | 🆕 미구현 | `tests/spec/eng_alg_200_plan_partition.rs` |
| ENG-ALG-210 | (A) | 초기 uniform 로딩 + secondary mmap handle 예약 | 🆕 미구현 | `engine/tests/spec/test_eng_alg_210_initial_load.rs` |
| ENG-ALG-211 | (A) | SwapExecutor: per-layer ArcSwap + permutation + madvise + **batch 완료 후 ratio_generation 정확히 1회 bump**. | 🆕 미구현 | `engine/tests/spec/test_eng_alg_211_swap_executor.rs` |
| ENG-ALG-212 | (A) | On-demand ImportanceCollector 활성화 (prefill-tail + K=512 fallback) | 🆕 미구현 | `engine/tests/spec/test_eng_alg_212_importance_activation.rs` |
| ENG-ALG-213 | (A) | SwapDecider 로직 (ratio × layer 수, uniform fallback) | 🆕 미구현 | `engine/tests/spec/test_eng_alg_213_swap_decider.rs` |
| ENG-ALG-214 | (A) | WeightSwapHandler (CachePressureHandler 구현) + per-token forward snapshot 규약 (214-SNAP) | 🆕 미구현 | `engine/tests/spec/test_eng_alg_214_weight_swap_handler.rs` |
| ENG-ALG-214-ROUTE | (A) | EngineCommand::SwapWeights → generate.rs 직접 dispatch (Pipeline 비경유) + Rejected 조건 | 🆕 미구현 | `engine/tests/spec/test_eng_alg_214_route_dispatch.rs` |
| ENG-ALG-215 | (A) | WeightSwapDecider: importance × ε ascending bottom-k + 보호 layer (0, last) + uniform fallback | 🆕 미구현 | `engine/tests/spec/test_eng_alg_215_weight_swap_decider.rs` |
| ENG-ALG-216 | (A) | ε 계산: per-tensor Frobenius 상대 오차² 합산 → per-layer ε_i. Engine init eager. 실패 fallback 규약 | 🆕 미구현 | `engine/tests/spec/test_eng_alg_216_quant_noise_calc.rs` |
| ENG-ALG-217 | (B) | QCF_swap 공식 = Σ_{i∈S} importance × ε / Σ_all importance × ε. `[0, 1]` 범위. 단조성. | 🆕 미구현 | `engine/tests/spec/test_eng_alg_217_qcf_swap_formula.rs` |
| ENG-ALG-218 | (E) | On-demand ImportanceCollector: RequestQcf → next prefill arm → finalize → QcfEstimate + K=512 fallback | 🆕 미구현 | `engine/tests/spec/test_eng_alg_218_importance_on_demand.rs` |
| ENG-ALG-219 | (A) | `FullKernelPlan::execute()` 진입 1회 atomic load 비교 (`plan.ratio_generation_at_build` ↔ `model.ratio_generation`). mismatch 시 `PlanInvalidated`. INV-120과 OR 결합. tensor_partition × weight swap 상호 배타(DF-35-3). | 🆕 미구현 | `engine/tests/spec/test_eng_alg_219_plan_invalidation.rs` |
| ENG-ALG-220 | (A) | `forward_into` per-token entry에서 `entry_ratio_generation = model.ratio_generation.load(Acquire)` capture → plan에 전달 → 동일 토큰 내 재사용. mid-token swap은 다음 토큰부터 관측. INV-121 per-token snapshot과 동일 시점. | 🆕 미구현 | `engine/tests/spec/test_eng_alg_219_plan_invalidation.rs` |
| ENG-ALG-221 | (A) | SwapExecutor batch 종료 직후 `OpenCLBackend::noshuffle_soa_registry` invalidate (전체 clear 또는 per-layer 제거). 다음 forward의 plan rebuild 경로에서 새 cl_mem 주소로 자연 재등록. **디바이스(Adreno 830) 한정 발현**, 호스트는 NoOp. ENG-ALG-211 step (e)와 동일 단계. | 🆕 미구현 | `engine/tests/spec/test_inv_130_noshuffle_soa_coherence.rs` + 디바이스 manual 검증 |

## Engine Data

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| ENG-DAT-012 | (D) | KVCache 구현 | ✅ | `engine/tests/spec/test_eng_dat_012_031.rs` |
| ENG-DAT-020 | (D) | Buffer trait | ✅ | `engine/tests/spec/test_eng_dat_012_031.rs` |
| ENG-DAT-C05 | (D) | KvStreaming protocol path (EvictPlan 생성, active/available_actions) | ✅ | `engine/tests/spec/test_eng_dat_c05_streaming.rs` |
| ENG-DAT-C06 | (D) | KvMergeD2o protocol path (EvictPlan 생성, Pipeline dispatch) | ✅ | `engine/tests/spec/test_eng_dat_c06_d2o.rs` |
| ENG-DAT-090 | (D) | LoadConfig primary/secondary_source 필드 (동적 swap 예약) | 🆕 미구현 | `engine/tests/spec/test_eng_dat_090_load_config.rs` |
| ENG-DAT-091 | (D) | **[DEPRECATED 2026-04-24]** LayerDtypeProfile TOML schema. ID 재사용 금지. | — | — |
| ENG-DAT-092 | (D) | LayerSlot 구조 (current_dtype, weights Arc, secondary_mmap_handle, generation) | 🆕 미구현 | `engine/tests/spec/test_eng_dat_092_layer_slot.rs` |
| ENG-DAT-093 | (D) | TransformerModel flat 배치: layers(Vec<LayerSlot>) + secondary_mmap + ratio_generation + 기존 embedding/final_norm/lm_head. 별도 wrapper struct 없음. | 🆕 미구현 | `engine/tests/spec/test_eng_dat_093_transformer_weights.rs` |
| ENG-DAT-094 | (D) | SecondaryMmap (mmap + decoder layer tensor index only). cross_layer_offsets 필드 없음. | 🆕 미구현 | `engine/tests/spec/test_eng_dat_094_secondary_mmap.rs` |
| ENG-DAT-095 | (D) | QuantNoiseTable (per-layer ε, eager 계산, NaN 결측 표기, uniform_ones fallback) | 🆕 미구현 | `engine/tests/spec/test_eng_dat_095_quant_noise_table.rs` |

## Cross-cutting

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| CROSS-060 | (D) | 타이밍 상수 정의 (heartbeat_interval, MAX_PAYLOAD_SIZE) | ✅ | `engine/tests/spec/test_cross_060_061.rs` |
| CROSS-061 | (B) | 타이밍 관계 수식 (heartbeat > recv_timeout) | ✅ | `engine/tests/spec/test_cross_060_061.rs` |

## Test Tools

| PREFIX-NNN | 분류 | 설명 | 상태 | 테스트 위치 |
|------------|------|------|------|-----------|
| TOOL-010 | (E) | mock_engine: Capability 전송 (SEQ-022) | ✅ | `manager/src/bin/mock_engine.rs` (inline) |
| TOOL-013 | (E) | mock_engine: Heartbeat 주기 전송 | ✅ | `manager/src/bin/mock_engine.rs` (inline) |
| TOOL-014 | (D) | mock_engine: EngineStatus 필드 정확성 | 🔶 | active_actions/available_actions 미반영 |
| TOOL-016 | (A) | mock_engine: 13종 command 처리 | ✅ | `manager/src/bin/mock_engine.rs` (inline) |
| TOOL-017 | (E) | mock_engine: INV-022 (1 Directive = 1 Response) | ✅ | `manager/src/bin/mock_engine.rs` (inline) |
| TOOL-018 | (E) | mock_engine: INV-023/024 (seq_id/results 일치) | ✅ | `manager/src/bin/mock_engine.rs` (inline) |
| TOOL-019 | (E) | mock_engine: QcfEstimate 전송 (SEQ-096) | ⬜ | 미구현 |
| TOOL-030 | (E) | mock_manager: Unix 소켓 서버 | ⬜ | 미구현 |
| TOOL-035 | (E) | mock_manager: Directive 전송 | ⬜ | 미구현 |
| TOOL-036 | (A) | mock_manager: seq_id 단조 증가 (INV-020/021) | ⬜ | 미구현 |
| TOOL-038 | (E) | mock_manager: QcfEstimate 수신 | ⬜ | 미구현 |
| TOOL-048 | (B) | mock_manager: 프로토콜 불변식 검증 출력 | ⬜ | 미구현 |
| TOOL-050 | (E) | 상호운용: mock_engine ↔ mock_manager E2E | ⬜ | 미구현 |
