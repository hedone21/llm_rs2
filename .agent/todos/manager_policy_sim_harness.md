# Manager Policy/Supervisory Host Simulation Harness

**시작**: 2026-04-14
**플랜**: `/home/go/.claude/plans/iterative-squishing-map.md` (승인됨)
**목적**: LuaPolicy + EwmaReliefTable을 host에서 반응형 시뮬레이션으로 검증. 실기 배포 의존도 제거.

## 스코프

- **MVP (v1)**: A(Throughput 세분화), B(Thermal 경합), C(Action composition: 분리규칙 + 상호작용항), D(Observation delay 정확성), E(Memory bandwidth 독립 추적)
- **v1.1**: F(관측 이원성), G(Warm-up), H(KV dtype 전환), M(디버깅 출력)
- **v1 추가**: J(Noise 주입, 기본 off / seed 고정 opt-in)
- **후속**: I(Failure injection), K(실기 calibration 바이너리), L(성능 최적화)

## Phase 진행 상황

### Phase 1 — 스켈레톤 및 YAML 로더 🔄
- [ ] `Cargo.toml` dev-dependencies 추가: `serde_yaml`, `validator`, `evalexpr`, `rand_chacha`
- [ ] `manager/tests/common/sim/mod.rs` 모듈 골격
- [ ] `config.rs`: 전체 YAML struct + `Bytes`/`Megabytes` newtype + deny_unknown_fields + deep merge (extends) + validator 통과
- [ ] `expr.rs`: evalexpr 래퍼 (로드 시 AST 컴파일, 런타임 HashMapContext 바인딩)
- [ ] `manager/tests/fixtures/sim/baseline.yaml` 초안
- [ ] 단위 테스트: `tests/sim/test_config.rs` (파싱/상속/검증/타입 변환)

### Phase 2 — 물리 엔진 ⏳
- [ ] `state.rs`: PhysicalState, EngineStateModel 분리
- [ ] `physics.rs`: 1st-order lag 적분, DVFS 피드백, thermal coupling, bandwidth 계산
- [ ] `compose.rs`: dimension-wise 결합 + interaction_term
- [ ] `derived` expression 평가 훅 (phase_throughput, throttle_factor, skip_boost 내장 함수)
- [ ] 단위 테스트: Evict kv bytes 50%, Throttle+Partition throughput 곱 + interaction, DVFS thermal→freq

### Phase 3 — 관측 투영 / 시계 ⏳
- [ ] `clock.rs`: VirtualClock + binary heap 이벤트 큐
- [ ] `signal.rs`: derive_signals (4가지 SystemSignal variants, 소스별 polling), derive_heartbeat
- [ ] `noise.rs`: 옵션 Gaussian (ChaCha8, seed_key별 스트림)
- [ ] 단위 테스트: 10초 시나리오 → expected signal sequence, noise 재현성

### Phase 4 — 하니스 ⏳
- [ ] `harness.rs`: Simulator (Box<dyn PolicyStrategy> 주입), tick(dt), run_for/run_until
- [ ] `trajectory.rs`: CSV/JSON dump, assertion 헬퍼
- [ ] 통합 테스트: LuaPolicy + baseline.yaml memory 램프 시나리오

### Phase 5 — 시나리오 YAML / spec 테스트 마이그레이션 ✅ (2026-04-14, 144755d)
- [x] `fixtures/sim/scenarios/` 3종 (memory_pressure_steady, thermal_ramp_with_decode, partition_contention)
- [x] `fixtures/sim/lua/` 4종 fixture Lua 스크립트
- [x] `insta` crate dev-dependency 추가 (v1, yaml+json)
- [x] `PolicyStrategy::relief_snapshot()` default method + LuaPolicy 구현
- [x] `TrajectorySummary` / `PhysicalStateSummary` 타입 추가
- [x] `test_scenarios.rs` 4개 시나리오 + insta 스냅샷 (.snap 4개 포함)
- [x] 98 tests pass (94 기존 + 4 신규)
- [x] `test_mgr_alg_080_083_ewma_relief.rs` — 24개 todo!() → 24/24 pass (2026-04-14, 0656040)
  - EwmaReliefTable/ReliefEntry pub 노출 + ManualClock 주입으로 MGR-ALG-080~083, INV-086~090, MGR-DAT-070~074 검증
  - spec 테스트: 8→32 pass, sim 테스트: 104 pass 비회귀
- [ ] 기존 실기 의존 테스트 마이그레이션 (Phase 6 범위로 이동)

#### 마이그레이션 후보 (향후 작업)
- `test_mgr_alg_080_083_ewma_relief.rs` — 완료 (2026-04-14)
- `test_inv_*.rs` 중 signal→directive 경로 검증 항목: 시나리오 YAML + MockPolicy로 대체 가능
  - `test_inv_002_memory_warning_evict.rs` — memory_pressure_steady 시나리오로 커버
  - `test_inv_004_thermal_alert_throttle.rs` — thermal_ramp 시나리오로 커버
- `test_mgr_dat_075_076_engine_util.rs` — 유지 (단순 Lua ctx 통과 검증, 시뮬레이터 불필요)

### Phase 6 — 디바이스 preset 및 문서 ⏳
- [ ] `s25_galaxy.yaml` (baseline extends, manual estimates)
- [ ] `docs/` 사용 가이드

## 재사용 대상

- `manager/src/pipeline.rs:20-44` — PolicyStrategy trait
- `manager/src/lua_policy.rs:206-290` — EwmaReliefTable, Pressure6D
- `manager/src/monitor/{memory,compute,thermal}.rs` — evaluate() 공식 (pub(crate) 확장 필요 시 최소 수정)
- `shared/src/lib.rs` — EngineStatus/Message/Directive/SystemSignal
- `manager/tests/spec/helpers.rs` — 헬퍼 패턴 참고
- `engine/src/resilience/executor.rs:209-227` — throughput EMA (ALPHA=0.1)

## 품질 게이트

- `cargo test -p llm_manager --test sim` (신규 테스트) 전부 통과
- `cargo test -p llm_manager` (기존 spec) 비회귀
- `cargo fmt --all --check` + `cargo clippy --all-targets` clean
- Noise seed 재현성 (byte-for-byte 동일)
- 60s 시나리오 × 10 반복 CI 10s 이내

## 주의

- Calibration 도구(K)는 별도 후속. 당분간 YAML 상수는 보수적 추정치 + TODO 주석
- validator는 deep merge 이후에만 실행
- evalexpr 오타는 로드 시점에 전수 검증 (dry-run 바인딩)
- 시뮬레이터는 1B 단일 모델 가정으로 시작
