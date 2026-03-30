# Invariants Catalog -- Architecture

> spec/41-invariants.md의 **컴포넌트별 구현 매핑**. 65개 INV를 구현 컴포넌트 기준으로 그룹핑.

---

## Cargo Workspace (INV-001, INV-010, INV-011, INV-080, INV-081)

의존 구조와 런타임 선택으로 보장하는 아키텍처 불변식.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-001 | `Cargo.toml` (workspace) | Engine/Manager Cargo.toml에 상대방 크레이트 의존 없음. Shared만 공유 의존 | static |
| INV-010 | `engine/Cargo.toml`, `manager/Cargo.toml` | 크레이트 의존 그래프에서 Engine↔Manager 경로 없음 | static |
| INV-011 | `shared/Cargo.toml` | Shared 외부 의존: serde, serde_json만. Engine/Manager 내부 타입 미참조 | static |
| INV-080 | `Cargo.toml` (전체) | async 런타임(tokio 등) 의존 없음. `std::thread` + `mpsc` 전용 | static |
| INV-081 | `shared/Cargo.toml` | IPC 직렬화 의존: serde, serde_json만. 바이너리 포맷 없음 | static |

---

## shared/src/lib.rs (INV-027, INV-028)

프로토콜 호환성을 코드 리뷰로 보장하는 불변식.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-027 | `shared/src/lib.rs` | serde 어노테이션 변경 = 프로토콜 버전 변경 -- 코드 리뷰로 보장 | static |
| INV-028 | `shared/src/lib.rs` | 새 필드에 `#[serde(default)]` 적용 (예: `EngineStatus`의 5개 필드) -- 코드 리뷰 | static |

---

## Backend trait (INV-002, INV-012, INV-065)

하드웨어 추상화와 플랫폼 격리 불변식.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-002 | `engine/src/backend/cpu/neon.rs` | `#[cfg(target_arch = "aarch64")]` 게이트. x86에서 NEON 코드 컴파일 제외 | static |
| INV-012 | `engine/src/core/backend.rs` | `pub trait Backend: Send + Sync` -- 유일한 hw 추상화점. 모든 수치 연산 디스패치 | static |
| INV-065 | `engine/src/core/backend.rs` | Backend trait 바운드: `Send + Sync` | static |

---

## Model Loading (INV-003)

모델 로딩 시 아키텍처 검증.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-003 | `engine/src/models/config.rs` | `architectures` 필드 매칭 실패 시 `bail!("Unsupported model architecture")` | runtime |

---

## QCF (INV-004, INV-017)

lossy action 시 품질 비용 수집 보장.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-004 | `engine/src/core/qcf/` | `eviction_qcf.rs`, `quant_qcf.rs` -- lossy action 실행 시 `QcfMetric` 생성 → `qcf_sink.push()` | test |
| INV-017 | `engine/src/core/qcf/` | INV-004 재확인 | test |

---

## generate.rs 추론 루프 (INV-018, INV-060, INV-061)

추론 루프의 단일 스레드 및 ExecutionPlan 소비 패턴.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-018 | `engine/src/bin/generate.rs` | 단일 스레드 추론 루프 (main thread에서 forward + decode). 동시 접근 없음 | static |
| INV-060 | `engine/src/bin/generate.rs` | decode loop에서 `executor.poll(&kv_snap)` 호출이 loop iteration 당 1회 | static |
| INV-061 | `engine/src/bin/generate.rs` | `let plan = executor.poll(...)` -- 즉시 소비, 변수 재할당으로 폐기. 1회성 | static |

---

## CommandExecutor / poll() (INV-005, INV-015, INV-022~026, INV-062, INV-064, INV-071, INV-074~076)

Engine의 명령 처리 중심 컴포넌트.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-005 | `engine/src/resilience/executor.rs` | Resilience 실패 시 추론 루프 지속. `command_executor = None`이면 checkpoint skip | test |
| INV-015 | `engine/src/resilience/executor.rs` | Capability 세션당 1회 전송 -- 연결 초기화 시에만 | runtime, test |
| INV-022 | `engine/src/resilience/executor.rs` | `poll()` -- 각 Directive에 대해 정확히 1개 CommandResponse 전송 | runtime, test |
| INV-023 | `engine/src/resilience/executor.rs` | `CommandResponse.seq_id = directive.seq_id` 직접 대입 | runtime |
| INV-024 | `engine/src/resilience/executor.rs` | `results.len() == commands.len()` -- commands iterate 시 1:1 결과 생성 | runtime |
| INV-025 | `engine/src/resilience/executor.rs` | INV-024 재확인 | runtime |
| INV-026 | `engine/src/resilience/executor.rs` | 수신 seq_id에 대해서만 Response 생성 -- `poll()` 내부 처리 | runtime |
| INV-062 | `engine/src/resilience/executor.rs` | `poll()` step 5 -- `plan.suspended == true` → `evict/switch_device/prepare_device = None` | runtime |
| INV-064 | `engine/src/resilience/executor.rs` | `poll()` -- `elapsed >= heartbeat_interval` → `send_heartbeat()` | runtime |
| INV-071 | `engine/src/resilience/executor.rs` | `apply_command()` 내부에서만 EngineState 전이. private 필드로 캡슐화 | static |
| INV-074 | `engine/src/resilience/executor.rs` | `poll()` step 5 -- `plan.suspended == true` → evict/switch/prepare = None (INV-062 재확인) | runtime |
| INV-075 | `engine/src/resilience/executor.rs` | `apply_command(Resume)` -- `compute_level=Normal, memory_level=Normal, throttle_delay_ms=0` | runtime |
| INV-076 | `engine/src/resilience/executor.rs` | `apply_command(RestoreDefaults)` -- `active_actions.clear(), throttle=0, compute/memory=Normal` | runtime |

---

## Transport / MessageLoop (INV-063)

Transport 소유권과 스레드 안전.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-063 | `engine/src/resilience/transport.rs` | `MessageLoop::spawn()` -- Transport를 `move` 클로저로 이동, 단일 소유자 | static |

---

## OperatingMode / resolve_conflicts() (INV-070, INV-072, INV-073)

Engine 상태 머신과 충돌 해소.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-070 | `engine/src/resilience/state.rs` | `OperatingMode::from_levels()` -- 순수 함수, 입력 4종 Level만 참조, `&self` 없음 | static |
| INV-072 | `engine/src/resilience/strategy/mod.rs` | `resolve_conflicts()` -- Suspend 존재 시 `return vec![Suspend]` | runtime, test |
| INV-073 | `engine/src/resilience/strategy/mod.rs` | `resolve_conflicts()` -- RestoreDefaults는 다른 제약 없을 때만 반환 | runtime, test |

---

## Monitor 시스템 (INV-006, INV-013)

Manager 모니터링의 장애 격리.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-006 | `manager/src/main.rs`, `manager/src/emitter/` | Engine 미연결 시 `emit()` skip, Ok 반환. Monitor 루프 계속 | test |
| INV-013 | `manager/src/main.rs` | 각 Monitor는 독립 `std::thread::spawn`, mpsc 채널로만 통신. 공유 상태 없음 | static, test |

---

## PiController (INV-030, INV-031, INV-083)

PI 제어기의 수치 안전.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-030 | `manager/src/pi_controller.rs` | `update()` -- `if self.can_act { integral += ... }`, false 시 skip | runtime, test |
| INV-031 | `manager/src/pi_controller.rs` | `update()` -- `(self.integral + error * dt).clamp(0.0, self.integral_clamp)` | runtime |
| INV-083 | `manager/src/pi_controller.rs` | `update()` -- `(kp * error + ki * integral).clamp(0.0, 1.0)`, PI output [0,1] | runtime |

---

## HierarchicalPolicy / Pipeline (INV-014, INV-020, INV-021, INV-051, INV-085)

정책 파이프라인의 시퀀스 ID와 액션 발행 규칙.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-014 | `manager/src/pipeline.rs` | `static SEQ_COUNTER: AtomicU64::new(1)`, `next_seq_id()` → `fetch_add(1, Relaxed)` | runtime |
| INV-020 | `manager/src/pipeline.rs` | `SEQ_COUNTER` 단조 증가 (INV-014와 동일 메커니즘) | runtime |
| INV-021 | `manager/src/pipeline.rs` | AtomicU64 단조 증가 -- 재사용 수학적 불가 | runtime |
| INV-051 | `manager/src/pipeline.rs` | ObservationContext -- `for action in ctx.applied_actions` → 동일 relief를 각 action에 귀속 | runtime |
| INV-085 | `manager/src/pipeline.rs` | `process_signal()` -- `OperatingMode::Normal` → `needs_action=false`, 액션 미발행 | runtime, test |

---

## SupervisoryLayer (INV-032~036)

운영 모드 전이 규칙과 임계값 부등식.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-032 | `manager/src/supervisory.rs` | `next_mode()` -- `peak >= critical_threshold` → Critical 직행. Normal→Critical 가능 | test |
| INV-033 | `manager/src/supervisory.rs` | `next_mode()` -- Critical→Warning (1단계), Warning→Normal (1단계). hold_time 경과 필수 | test |
| INV-034 | `manager/src/config.rs` | `SupervisoryConfig::default()` -- `warning_release(0.25) < warning_threshold(0.4)` | runtime |
| INV-035 | `manager/src/config.rs` | `SupervisoryConfig::default()` -- `critical_release(0.50) < critical_threshold(0.7)` | runtime |
| INV-036 | `manager/src/config.rs` | `SupervisoryConfig::default()` -- `warning_threshold(0.4) < critical_threshold(0.7)` | runtime |

---

## ActionSelector (INV-016, INV-037~044, INV-084)

액션 조합 탐색의 정확성 보장. Stateless unit struct.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-016 | `manager/src/selector.rs` | `has_exclusion_conflict()` -- O(N^2) 쌍별 검사, 동일 그룹 시 skip | runtime, test |
| INV-037 | `manager/src/selector.rs` | `filter_candidates()` -- `mode==Warning && kind==Lossy` → continue | runtime, test |
| INV-038 | `manager/src/selector.rs` | `filter_candidates()` -- `active_actions.contains(&meta.id)` → skip | runtime |
| INV-039 | `manager/src/selector.rs` | `compute_cost()` -- `kind == Lossless` → `0.0` | runtime |
| INV-040 | `manager/src/selector.rs` | `compute_cost()` -- QCF 미제공 Lossy → `f32::INFINITY` | runtime |
| INV-041 | `manager/src/selector.rs` | `has_exclusion_conflict()` → skip (INV-016 재확인) | runtime, test |
| INV-042 | `manager/src/selector.rs` | `find_optimal()` -- `total_relief.latency < -latency_budget` → 조합 skip | runtime |
| INV-043 | `manager/src/selector.rs` | `find_optimal()` -- `best_mask` (완전 해소) 존재 시 `best_effort_mask`보다 우선 | runtime |
| INV-044 | `manager/src/selector.rs` | `parametrize()` -- `value.clamp(range.min, range.max)` | runtime |
| INV-084 | `manager/src/selector.rs` | `ActionSelector` = unit struct (상태 없음). `select()` = 연관 함수 | static |

---

## ActionId / types.rs (INV-045, INV-050)

데이터 타입의 정적 매핑과 연산 규칙.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-045 | `manager/src/types.rs` | `ActionId::primary_domain()` -- SwitchHw/Throttle/LayerSkip→Compute, 나머지→Memory | static |
| INV-050 | `manager/src/types.rs` | `impl Sub for PressureVector` -- `ReliefVector { latency: 0.0 }` | runtime |

---

## OnlineLinearEstimator (INV-046~049)

RLS 기반 Relief Estimator의 수학적 불변식.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-046 | `manager/src/relief/linear.rs` | `LinearModel::observe()` -- RLS gain vector `k = P * phi / denom` | test |
| INV-047 | `manager/src/relief/linear.rs` | `LinearModel::observe()` -- bias EMA `(1-lr)*b + lr*residual`, lr=0.1 | test |
| INV-048 | `manager/src/relief/linear.rs` | `LinearModel::new()` -- P = 100.0 * I (DxD 단위 행렬 * 100) | runtime |
| INV-049 | `manager/src/config.rs` | `ReliefModelConfig::default()` -- `forgetting_factor = 0.995`, (0, 1] 범위 | runtime |

---

## Manager Channel (INV-082)

IPC 연결 모델.

| INV | 구현 컴포넌트 | 구현 함수/메서드 | 검증 방법 |
|-----|-------------|----------------|----------|
| INV-082 | `manager/src/channel/unix_socket.rs`, `manager/src/channel/tcp.rs` | 단일 `accept()` → 1:1 클라이언트 연결. 다중 Engine 미지원 | runtime |

---

## 카테고리별 통계

| 카테고리 | INV 목록 | 개수 |
|---------|---------|------|
| **Safety** (16) | 001, 002, 005, 006, 010, 011, 013, 018, 061, 062, 063, 065, 072, 074, 080, 082 | 16 |
| **Correctness** (44) | 003, 004, 012, 014~017, 020~026, 028, 030~051, 070, 071, 073, 075, 076, 083~085 | 44 |
| **Performance** (2) | 042, 060 | 2 |
| **Compatibility** (3) | 027, 028, 081 | 3 |

## 검증 방법별 참조

### static 검증 (20개 주 검증)

컴파일 타임 또는 코드 구조로 보장:

| INV 그룹 | 검증 메커니즘 |
|----------|-------------|
| 001, 010, 011 | Cargo.toml 의존 구조 |
| 002 | `#[cfg(target_arch)]` feature gate |
| 012, 065 | trait 바운드 (`Send + Sync`) |
| 013 | `std::thread::spawn` + mpsc 아키텍처 |
| 018, 060, 061 | 코드 구조 (단일 스레드, loop 패턴) |
| 027, 028, 081 | serde 어노테이션 -- 코드 리뷰 |
| 045, 084 | 함수/struct 구조 (stateless, match) |
| 063 | Rust ownership (`move` 클로저) |
| 070, 071 | 함수 시그니처, private 필드 캡슐화 |
| 080 | Cargo.toml 의존 부재 |

### runtime 검증 (32개 주 검증)

실행 중 clamp, 조건 검사, AtomicU64 등으로 보장:

| INV 그룹 | 주요 메커니즘 |
|----------|-------------|
| 003 | `bail!` on unsupported architecture |
| 014, 020, 021 | `AtomicU64::fetch_add` |
| 022~026 | Directive-Response 1:1 매핑 로직 |
| 030~031, 083 | `.clamp()` |
| 034~036 | Default 값의 부등식 관계 |
| 037~044 | ActionSelector 필터/비용/조합 조건 |
| 048~051 | 초기화 값, 연산 결과 |
| 062, 064, 074~076 | `poll()` 내부 조건 분기 |
| 082 | 단일 accept 모델 |
| 085 | `needs_action = false` 조건 |

### test 검증 (12개 주 검증)

단위/통합/장애 주입 테스트:

| INV 그룹 | 테스트 위치 |
|----------|-----------|
| 004, 017 | `engine/src/core/qcf/` 모듈 테스트 |
| 005, 006 | 장애 주입 테스트 (resilience) |
| 032, 033 | `manager/src/supervisory.rs` `#[cfg(test)]` |
| 046, 047 | `manager/src/relief/linear.rs` `#[cfg(test)]` |
| 072, 073 | `engine/src/resilience/strategy/mod.rs` `#[cfg(test)]` |

---

## 재확인(Restatement) 관계

| INV | 재확인 대상 | 비고 |
|-----|-----------|------|
| INV-017 | INV-004 | QCF 수집 재확인 |
| INV-025 | INV-024 | `results.len() == commands.len()` 재확인 |
| INV-041 | INV-016 | 배타 그룹 검사 재확인 |
| INV-074 | INV-062 | Suspend 시 evict/switch/prepare = None 재확인 |