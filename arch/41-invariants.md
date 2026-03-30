# Invariants Catalog -- Architecture

> spec/41-invariants.md의 구현 매핑. 65개 INV의 코드 구현 위치.

## INV 구현 매핑

### 3.1 System/Component Invariants [INV-001 ~ INV-018]

| INV | 원본 파일 | 코드 위치 | 구현 방법 | 검증 방법 |
|-----|----------|----------|----------|----------|
| INV-001 | 00-overview | `Cargo.toml` (workspace) | Engine/Manager Cargo.toml에 상대방 크레이트 의존 없음. Shared만 공유 의존 | static |
| INV-002 | 00-overview | `engine/src/backend/cpu/neon.rs` | `#[cfg(target_arch = "aarch64")]` 게이트. x86에서 NEON 코드 컴파일 제외 | static |
| INV-003 | 00-overview | `engine/src/models/config.rs:125-147` | `architectures` 필드 매칭 실패 시 `bail!("Unsupported model architecture")` | runtime |
| INV-004 | 00-overview | `engine/src/core/qcf/` (eviction_qcf.rs, quant_qcf.rs) | lossy action 실행 시 `QcfMetric` 생성 → `qcf_sink.push()` | test |
| INV-005 | 00-overview | `engine/src/resilience/executor.rs` | Resilience 실패 시 추론 루프 지속. `command_executor = None`이면 checkpoint skip | test |
| INV-006 | 00-overview | `manager/src/main.rs`, `manager/src/emitter/` | Engine 미연결 시 `emit()` skip, Ok 반환. Monitor 루프 계속 | test |
| INV-010 | 01-architecture | `engine/Cargo.toml`, `manager/Cargo.toml` | 크레이트 의존 그래프에서 Engine↔Manager 경로 없음 | static |
| INV-011 | 01-architecture | `shared/Cargo.toml`, `shared/src/lib.rs` | Shared 의존: serde, serde_json만. Engine/Manager 내부 타입 미참조 | static |
| INV-012 | 01-architecture | `engine/src/core/backend.rs:5` | `pub trait Backend: Send + Sync` — 유일한 hw 추상화점. 모든 수치 연산 디스패치 | static |
| INV-013 | 01-architecture | `manager/src/main.rs` | 각 Monitor는 독립 `std::thread::spawn`, mpsc 채널로만 통신. 공유 상태 없음 | static, test |
| INV-014 | 01-architecture | `manager/src/pipeline.rs:65` | `static SEQ_COUNTER: AtomicU64::new(1)`, `fetch_add(1, Relaxed)` | runtime |
| INV-015 | 01-architecture | `engine/src/resilience/executor.rs` | Capability 세션당 1회 전송 — 연결 초기화 시에만 | runtime, test |
| INV-016 | 01-architecture | `manager/src/selector.rs` | `has_exclusion_conflict()` — O(N^2) 쌍별 검사, 동일 그룹 시 skip | runtime, test |
| INV-017 | 01-architecture | `engine/src/core/qcf/` | INV-004의 재확인. lossy action 시 QcfMetric 생성 | test |
| INV-018 | 01-architecture | `engine/src/bin/generate.rs` | 단일 스레드 추론 루프 (main thread에서 forward + decode) | static |

### 3.2 Protocol Invariants [INV-020 ~ INV-028]

| INV | 원본 파일 | 코드 위치 | 구현 방법 | 검증 방법 |
|-----|----------|----------|----------|----------|
| INV-020 | 10-protocol | `manager/src/pipeline.rs:65` | `SEQ_COUNTER: AtomicU64::new(1)` + `fetch_add(1, Relaxed)` | runtime |
| INV-021 | 10-protocol | `manager/src/pipeline.rs:65` | AtomicU64 단조 증가 — 재사용 수학적 불가 | runtime |
| INV-022 | 10-protocol | `engine/src/resilience/executor.rs` `poll()` | 각 Directive에 대해 정확히 1개 CommandResponse 전송 | runtime, test |
| INV-023 | 10-protocol | `engine/src/resilience/executor.rs` | `CommandResponse.seq_id = directive.seq_id` 직접 대입 | runtime |
| INV-024 | 10-protocol | `engine/src/resilience/executor.rs` | `results.len() == commands.len()` — commands iterate 시 1:1 결과 생성 | runtime |
| INV-025 | 11-protocol-messages | `engine/src/resilience/executor.rs` | INV-024 재확인. `results.len() == commands.len()` | runtime |
| INV-026 | 11-protocol-messages | `engine/src/resilience/executor.rs` | 수신 seq_id에 대해서만 Response 생성 — poll() 내부 처리 | runtime |
| INV-027 | 11-protocol-messages | `shared/src/lib.rs` | serde 어노테이션 변경 = 프로토콜 버전 변경 — 코드 리뷰로 보장 | static |
| INV-028 | 11-protocol-messages | `shared/src/lib.rs` | 새 필드에 `#[serde(default)]` 적용 — 코드 리뷰로 보장 | static |

### 3.3 Manager Algorithm Invariants [INV-030 ~ INV-051]

| INV | 원본 파일 | 코드 위치 | 구현 방법 | 검증 방법 |
|-----|----------|----------|----------|----------|
| INV-030 | 22-manager-algorithms | `manager/src/pi_controller.rs` | `if self.can_act { integral += ... }` — false 시 skip | runtime, test |
| INV-031 | 22-manager-algorithms | `manager/src/pi_controller.rs:66` | `integral.clamp(0.0, self.integral_clamp)` | runtime |
| INV-032 | 22-manager-algorithms | `manager/src/supervisory.rs` | `peak >= critical_threshold` → Critical 직행. Normal→Critical 가능 | test |
| INV-033 | 22-manager-algorithms | `manager/src/supervisory.rs` | `try_de_escalate()` — current_mode에서 1단계만 하강 | test |
| INV-034 | 22-manager-algorithms | `manager/src/config.rs:234-244` | Default 값으로 보증: `warning_release(0.25) < warning_threshold(0.4)` | runtime |
| INV-035 | 22-manager-algorithms | `manager/src/config.rs:234-244` | Default 값으로 보증: `critical_release(0.50) < critical_threshold(0.7)` | runtime |
| INV-036 | 22-manager-algorithms | `manager/src/config.rs:234-244` | Default 값으로 보증: `warning_threshold(0.4) < critical_threshold(0.7)` | runtime |
| INV-037 | 22-manager-algorithms | `manager/src/selector.rs` | `filter_candidates()` — `mode==Warning` 시 Lossy continue | runtime, test |
| INV-038 | 22-manager-algorithms | `manager/src/selector.rs` | `filter_candidates()` — active_actions 체크, 이미 활성 시 skip | runtime |
| INV-039 | 22-manager-algorithms | `manager/src/selector.rs` | `compute_cost()` — Lossless → `0.0` | runtime |
| INV-040 | 22-manager-algorithms | `manager/src/selector.rs` | `compute_cost()` — QCF 없으면 `f32::INFINITY` | runtime |
| INV-041 | 22-manager-algorithms | `manager/src/selector.rs` | `has_exclusion_conflict()` → skip. INV-016 재확인 | runtime, test |
| INV-042 | 22-manager-algorithms | `manager/src/selector.rs` | `total_relief.latency < -latency_budget` → 조합 skip | runtime |
| INV-043 | 22-manager-algorithms | `manager/src/selector.rs` | `best_mask` (완전 해소) 존재 시 `best_effort_mask`보다 우선 | runtime |
| INV-044 | 22-manager-algorithms | `manager/src/selector.rs` | `value.clamp(range.min, range.max)` — parametrize 출력 범위 제한 | runtime |
| INV-045 | 22-manager-algorithms | `manager/src/types.rs:47-56` | `ActionId::primary_domain()` — SwitchHw/Throttle/LayerSkip→Compute, 나머지→Memory | static |
| INV-046 | 22-manager-algorithms | `manager/src/relief/linear.rs` | RLS gain vector `k = P * phi / denom` | test |
| INV-047 | 22-manager-algorithms | `manager/src/relief/linear.rs` | bias EMA `(1-lr)*b + lr*residual`, lr=0.1 — W 갱신 후 잔여 오차 | test |
| INV-048 | 22-manager-algorithms | `manager/src/relief/linear.rs` | `new_model()` 내부: P = 100 * I (DxD 단위 행렬) | runtime |
| INV-049 | 22-manager-algorithms | `manager/src/config.rs:275` | `forgetting_factor = 0.995` — `(0, 1]` 범위 | runtime |
| INV-050 | 22-manager-algorithms | `manager/src/types.rs:103-113` | `PressureVector::sub()` → `ReliefVector { latency: 0.0 }` | runtime |
| INV-051 | 22-manager-algorithms | `manager/src/pipeline.rs:242-245` | `for action in ctx.applied_actions` — 동일 relief를 각 action에 귀속 | runtime |

### 3.4 Engine Architecture Invariants [INV-060 ~ INV-065]

| INV | 원본 파일 | 코드 위치 | 구현 방법 | 검증 방법 |
|-----|----------|----------|----------|----------|
| INV-060 | 30-engine | `engine/src/bin/generate.rs` decode loop | `executor.poll(&kv_snap)` 호출이 토큰당 1회 — loop iteration 당 1회 | static |
| INV-061 | 30-engine | `engine/src/bin/generate.rs` decode loop | `let plan = executor.poll(...)` — 즉시 소비, 변수 재할당으로 폐기 | static |
| INV-062 | 30-engine | `engine/src/resilience/executor.rs` `poll()` step 5 | `plan.suspended == true` → `evict/switch_device/prepare_device = None` | runtime |
| INV-063 | 30-engine | `engine/src/resilience/transport.rs` | `MessageLoop::spawn()` — Transport를 `move` 클로저로 이동, 단일 소유자 | static |
| INV-064 | 30-engine | `engine/src/resilience/executor.rs` `poll()` | `elapsed >= heartbeat_interval` → `send_heartbeat()` | runtime |
| INV-065 | 30-engine | `engine/src/core/backend.rs:5` | Backend trait 바운드: `Send + Sync` | static |

### 3.5 Engine State Machine Invariants [INV-070 ~ INV-076]

| INV | 원본 파일 | 코드 위치 | 구현 방법 | 검증 방법 |
|-----|----------|----------|----------|----------|
| INV-070 | 31-engine-state | `engine/src/resilience/state.rs` | `from_levels()` — 순수 함수, 입력 4종 Level만 참조, `&self` 없음 | static |
| INV-071 | 31-engine-state | `engine/src/resilience/executor.rs` | `apply_command()` 내부에서만 EngineState 전이. private 필드로 캡슐화 | static |
| INV-072 | 31-engine-state | `engine/src/resilience/strategy/mod.rs:52` | `resolve_conflicts()`: Suspend 존재 시 `return vec![Suspend]` | runtime, test |
| INV-073 | 31-engine-state | `engine/src/resilience/strategy/mod.rs:52` | `resolve_conflicts()`: RestoreDefaults는 다른 제약 없을 때만 반환 | runtime, test |
| INV-074 | 31-engine-state | `engine/src/resilience/executor.rs` `poll()` step 5 | `plan.suspended == true` → evict/switch/prepare = None. INV-062 재확인 | runtime |
| INV-075 | 31-engine-state | `engine/src/resilience/executor.rs` `apply_command(Resume)` | `compute_level=Normal, memory_level=Normal, throttle_delay_ms=0` | runtime |
| INV-076 | 31-engine-state | `engine/src/resilience/executor.rs` `apply_command(RestoreDefaults)` | `active_actions.clear(), throttle=0, compute/memory=Normal` | runtime |

### 3.6 Cross-cutting Invariants [INV-080 ~ INV-085]

| INV | 원본 파일 | 코드 위치 | 구현 방법 | 검증 방법 |
|-----|----------|----------|----------|----------|
| INV-080 | 40-cross-cutting | `Cargo.toml` (workspace 전체) | async 런타임(tokio 등) 의존 없음. `std::thread` + `mpsc` 전용 | static |
| INV-081 | 40-cross-cutting | `shared/Cargo.toml` | IPC 직렬화 의존: serde, serde_json만. 바이너리 포맷 없음 | static |
| INV-082 | 40-cross-cutting | `manager/src/channel/unix_socket.rs`, `manager/src/channel/tcp.rs` | 단일 `accept()` → 1:1 클라이언트 연결. 다중 Engine 미지원 | runtime |
| INV-083 | 40-cross-cutting | `manager/src/pi_controller.rs:70` | `(kp * error + ki * integral).clamp(0.0, 1.0)` — PI output [0,1] | runtime |
| INV-084 | 40-cross-cutting | `manager/src/selector.rs:14` | `ActionSelector` = unit struct (상태 없음, stateless). `predict()` = `&self` 읽기 전용 | static |
| INV-085 | 40-cross-cutting | `manager/src/pipeline.rs:312-313` | `OperatingMode::Normal => false` — Normal 모드에서 `needs_action=false`, 액션 미발행 | runtime, test |

## 카테고리별 코드 위치 분포

| 카테고리 | 주 코드 위치 | INV 수 |
|---------|------------|--------|
| Safety (16) | `Cargo.toml`, `engine/src/core/backend.rs`, `engine/src/resilience/`, `manager/src/main.rs`, `manager/src/channel/` | 16 |
| Correctness (44) | `manager/src/pi_controller.rs`, `manager/src/selector.rs`, `manager/src/supervisory.rs`, `engine/src/resilience/executor.rs`, `engine/src/resilience/state.rs` | 44 |
| Performance (2) | `engine/src/bin/generate.rs` (INV-060), `manager/src/selector.rs` (INV-042) | 2 |
| Compatibility (3) | `shared/src/lib.rs`, `shared/Cargo.toml` | 3 |

## 검증 방법별 참조

### static 검증 (20개 주 검증)

컴파일 타임 또는 코드 구조로 보장되는 INV:

| INV 그룹 | 검증 방법 |
|----------|----------|
| INV-001, INV-010, INV-011 | Cargo.toml 의존 구조 |
| INV-002 | `#[cfg(target_arch)]` feature gate |
| INV-012, INV-065 | trait 바운드 (`Send + Sync`) |
| INV-013 | `std::thread::spawn` + mpsc 아키텍처 |
| INV-018, INV-060, INV-061 | 코드 구조 (단일 스레드, loop 패턴) |
| INV-027, INV-028, INV-081 | serde 어노테이션 — 코드 리뷰 |
| INV-045, INV-084 | 함수/struct 구조 (stateless, match) |
| INV-063 | Rust ownership (`move` 클로저) |
| INV-070, INV-071 | 함수 시그니처, private 필드 캡슐화 |
| INV-080 | Cargo.toml 의존 부재 |

### runtime 검증 (32개 주 검증)

assert, clamp, 조건 검사, AtomicU64 등으로 실행 중 보장:

| INV 그룹 | 주요 메커니즘 |
|----------|-------------|
| INV-003 | bail! on unsupported architecture |
| INV-014, INV-020, INV-021 | `AtomicU64::fetch_add` |
| INV-022~026 | Directive-Response 1:1 매핑 로직 |
| INV-030~031, INV-083 | `f64::clamp` / `f32::clamp` |
| INV-034~036 | Default 값의 부등식 관계 |
| INV-037~044 | ActionSelector 필터/비용/조합 조건 |
| INV-048~051 | 초기화 값, 연산 결과 |
| INV-062, INV-064, INV-074~076 | `poll()` 내부 조건 분기 |
| INV-082 | 단일 accept 모델 |
| INV-085 | `needs_action = false` 조건 |

### test 검증 (12개 주 검증)

단위/통합/장애 주입 테스트로 검증:

| INV 그룹 | 테스트 위치 |
|----------|-----------|
| INV-004, INV-017 | `engine/src/core/qcf/` 모듈 테스트 |
| INV-005, INV-006 | 장애 주입 테스트 (resilience) |
| INV-032, INV-033 | `manager/src/supervisory.rs` #[cfg(test)] |
| INV-046, INV-047 | `manager/src/relief/linear.rs` #[cfg(test)] |
| INV-072, INV-073 | `engine/src/resilience/strategy/mod.rs` #[cfg(test)] |