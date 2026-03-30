# INV Coverage Tracker

> 전체: 65개 | ✅ 29 | ⬜ 19 | 🔶 17

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
| INV-003 | `config.json`의 `architectures`가 지원 목록에 없으면 로딩 거부. | Correctness | ⬜ | |
| INV-004 | QCF 수집 활성 상태에서 lossy action 실행 시 QcfMetric 생성 필수. | Correctness | ⬜ | |
| INV-005 | Manager 장애가 Engine 추론 루프를 중단시키지 않음. | Safety | ⬜ | |
| INV-006 | Engine 장애가 Manager 모니터링 루프를 중단시키지 않음. | Safety | ⬜ | |
| INV-010 | Engine-Manager 직접 코드 의존 금지. Shared가 유일한 공유 의존성. | Safety | 🔶 | (static: Cargo.toml) |
| INV-011 | Shared는 Engine/Manager 내부 구현에 의존 금지. | Safety | 🔶 | (static: Cargo.toml) |
| INV-012 | Backend trait이 유일한 하드웨어 추상화점. Backend 우회 직접 호출 금지. | Correctness | 🔶 | (static: 코드 리뷰) |
| INV-013 | Monitor 스레드 장애가 다른 Monitor에 전파 금지. | Safety | 🔶 | (static, test: 아키텍처) |
| INV-014 | EngineDirective.seq_id는 세션 내 단조 증가. | Correctness | ✅ | `engine/tests/spec/test_inv_020_026.rs` |
| INV-015 | Capability는 세션당 정확히 1회 전송. | Correctness | ✅ | `engine/tests/spec/test_inv_015.rs` |
| INV-016 | 동일 배타 그룹 액션 동시 활성화 금지. | Correctness | ✅ | `manager/tests/spec/test_inv_016.rs` |
| INV-017 | QCF 수집 활성 + lossy action 실행 시 QcfMetric 생성 필수. (=> INV-004) | Correctness | ⬜ | |
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
| INV-037 | Warning 모드에서 Lossy 액션 선택 금지. | Correctness | ⬜ | |
| INV-038 | 이미 활성 중인 액션은 재선택 금지. | Correctness | ⬜ | |
| INV-039 | Lossless 액션의 cost = 항상 0. | Correctness | ✅ | `manager/tests/spec/test_inv_039_040.rs` |
| INV-040 | QCF 값 없는 Lossy 액션 = INFINITY cost. | Correctness | ✅ | `manager/tests/spec/test_inv_039_040.rs` |
| INV-041 | 동일 배타 그룹 액션은 하나의 조합에 동시 미포함. (=> INV-016) | Correctness | ⬜ | |
| INV-042 | 조합의 총 latency 악화 > latency_budget이면 배제. | Performance | ⬜ | |
| INV-043 | 완전 해소 가능 조합 > best-effort 조합 (항상 우선). | Correctness | ✅ | `manager/tests/spec/test_inv_043_044.rs` |
| INV-044 | parametrize 출력 value는 [range.min, range.max] 범위 내. | Correctness | ✅ | `manager/tests/spec/test_inv_043_044.rs` |
| INV-045 | primary_domain 매핑: SwitchHw/Throttle/LayerSkip -> Compute. | Correctness | 🔶 | (static: 코드) |
| INV-046 | RLS gain vector k = f(P, phi). lambda는 망각 인수. | Correctness | ⬜ | |
| INV-047 | bias는 W 갱신 후 잔여 오차에 EMA(lr=0.1) 적용. | Correctness | ⬜ | |
| INV-048 | P matrix: D x D 대칭 양정치. 초기값 100 * I. | Correctness | ⬜ | |
| INV-049 | `lambda in (0, 1]`. lambda=1.0이면 forgetting 없음. | Correctness | ⬜ | |
| INV-050 | 관찰 relief의 latency 차원 = 항상 0.0. | Correctness | ✅ | `manager/tests/spec/test_inv_050.rs` |
| INV-051 | 동시 적용 시 전체 relief가 각 액션에 귀속 (개별 분리 불가). | Correctness | 🔶 | (static: 설계 한계) |

## Engine Architecture (INV-060 ~ INV-065)

| INV | 설명 | 카테고리 | 상태 | 테스트 위치 |
|-----|------|---------|------|-----------|
| INV-060 | `CommandExecutor.poll()`은 토큰당 최대 1회 호출. | Performance | 🔶 | (static: 코드 구조) |
| INV-061 | ExecutionPlan: 생성 즉시 소비, 1회성. | Safety | 🔶 | (static: 코드 구조) |
| INV-062 | Suspend 포함 ExecutionPlan: evict/switch_device/prepare_device = None. | Safety | ⬜ | |
| INV-063 | MessageLoop 스레드는 Transport의 유일한 소유자. | Safety | 🔶 | (static: ownership) |
| INV-064 | heartbeat_interval 내 최소 1회 Heartbeat 전송. | Correctness | ⬜ | |
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
| INV-081 | IPC 직렬화는 JSON (serde_json) 전용. | Compatibility | ⬜ | |
| INV-082 | 1:1 단일 클라이언트 연결. 다중 Engine 동시 연결 금지. | Safety | ⬜ | |
| INV-083 | PI Controller output은 [0, 1] 범위 내. | Correctness | ⬜ | |
| INV-084 | ActionSelector = stateless. ReliefEstimator.predict = 읽기 전용. | Correctness | 🔶 | (static: 코드 구조) |
| INV-085 | Normal 모드에서 액션 미발행. | Correctness | ⬜ | |
