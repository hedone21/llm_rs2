# Policy Simulator 사용 가이드

> **대상**: llm.rs 기여자 및 policy 개발자  
> **관련 인프라**: Phase 1~5 + Clock PR 1~3 + TASK-C 완료 기준  
> **관련 Spec ID**: MGR-ALG-080~083, MGR-DAT-070~074, INV-086~090, MSG-060

---

## 1. 개요 — 왜 시뮬레이터가 필요한가

LuaPolicy와 EwmaReliefTable은 Manager 서비스의 핵심 적응 로직이다. 그러나 이 로직을 검증하려면 기존 방식으로는:

- Android 빌드 및 디바이스 배포 필요
- Llama 3.2 1B 모델 로딩 (~수초)
- 추론 실행 중 SystemSignal이 비결정적으로 주입됨
- 회귀 원인 규명이 어렵고 CI 반복 속도가 느림

**Policy Simulator**는 이 의존성을 제거한다. Engine 상태 모델(물리 레이어) + 반응 물리 모델을 host에서 돌려 닫힌 루프 시뮬레이션을 제공한다. 핵심 기능:

- `VirtualClock` 기반 결정론적 시간 제어 (seed 지정 시 byte-level 재현)
- `LuaPolicy`에 `VirtualClockHandle` 자동 주입 → 3초 관측 지연도 가상 시간 기준으로 정확히 반영
- EwmaReliefTable 학습이 실제로 수렴하는지 host에서 검증 가능
- insta 스냅샷(골든 테스트) 패턴으로 회귀 방지

---

## 2. 빠른 시작 — 최소 예제

```rust
use std::{path::PathBuf, time::Duration};
use crate::common::sim::{config::load_scenario, harness::Simulator};

#[test]
fn quick_start_example() {
    let fixtures = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/sim");

    // 1. 시나리오 로드 (YAML + extends 상속)
    let cfg = load_scenario(fixtures.join("scenarios/memory_pressure_steady.yaml"))
        .expect("시나리오 로드 실패");

    // 2. Simulator 생성 (LuaPolicy + VirtualClockHandle 자동 배선)
    #[cfg(feature = "lua")]
    let mut sim = Simulator::with_lua_policy(
        cfg,
        fixtures.join("lua/memory_evict_graduated.lua"),
        llm_manager::config::AdaptationConfig::default(),
    ).expect("Simulator 생성 실패");

    // 3. 30초 시뮬 실행
    sim.run_for(Duration::from_secs(30)).expect("실행 실패");

    // 4. 결과 확인
    assert!(sim.trajectory().heartbeat_count() >= 1);
}
```

MockPolicy를 사용하는 경우 (`lua` feature 없는 CI 환경 등):

```rust
use crate::common::sim::{mock_policy::MockPolicy, harness::Simulator};

let mut sim = Simulator::new(cfg, Box::new(MockPolicy::new()));
sim.run_for(Duration::from_secs(5)).expect("실행 실패");
```

---

## 3. 시나리오 YAML 구조

시나리오 파일은 `manager/tests/fixtures/sim/` 아래에 위치한다.

### 3.1 섹션 개요

| 섹션 | 역할 |
|------|------|
| `extends` | 부모 YAML 파일 경로 (선택). deep merge 후 검증 |
| `initial_state` | 시뮬 시작 시 물리 상태 초기값 |
| `actions` | EngineCommand variant별 물리 효과 명세 |
| `composition` | 다중 action 동시 적용 시 차원별 결합 연산자 |
| `interactions` | 특정 action 조합의 추가 상호작용 항 |
| `passive_dynamics` | 열역학·KV 캐시·BW 자율 진화 규칙 |
| `dvfs` | 열→주파수 피드백 (CPU/GPU 각각) |
| `derived` | expression 기반 파생 변수 (throughput_tps, tbt_ms 등) |
| `external_injections` | 시나리오 스크립트에서 외부 부하 주입 (시간 윈도우) |
| `observation` | heartbeat / signal 폴링 주기 + noise 설정 |
| `rng_seed` | 결정론 제어 (null이면 noise 비활성화) |

### 3.2 `extends` 상속 메커니즘

```yaml
extends: baseline.yaml
```

- 부모 파일을 먼저 파싱 후 `serde_yaml::Value` 레벨에서 deep merge
- 미지정 필드는 부모에서 상속, 명시된 필드만 override
- `deny_unknown_fields`는 merge **완료 후** 최종 결과에 적용
- validator(`#[validate]`)도 merge 이후에만 실행됨

체인 상속은 지원하지 않는다 (A→B→C). 단계별 override는 preset + 시나리오 2단계까지 권장.

### 3.3 `initial_state` 주요 필드

| 필드 | 타입 | 단위 | 의미 |
|------|------|------|------|
| `kv_cache_bytes` | Bytes | bytes | KV 캐시 현재 점유 (`"512 MiB"` 또는 정수) |
| `kv_cache_capacity_bytes` | Bytes | bytes | KV 캐시 최대 할당 가능 크기 |
| `kv_cache_tokens` | u32 | tokens | 현재 캐시된 토큰 수 |
| `kv_cache_token_capacity` | u32 | tokens | 최대 토큰 용량 |
| `kv_dtype` | String | — | `"f16"` / `"q8"` / `"q4"` |
| `device_memory_total_mb` | u32 | MB | 디바이스 RAM 총량 |
| `device_memory_used_mb` | u32 | MB | OS + 앱 포함 총 메모리 사용량 |
| `memory_bw_utilization_pct` | f64 | % | LPDDR 대역폭 사용률 |
| `engine_cpu_pct` | f64 | % | 엔진 CPU 사용률 |
| `external_cpu_pct` | f64 | % | 다른 앱 CPU 사용률 |
| `cpu_freq_mhz` | u32 | MHz | 현재 CPU 주파수 |
| `cpu_max_freq_mhz` | u32 | MHz | CPU 최대 주파수 |
| `cpu_min_freq_mhz` | u32 | MHz | CPU 최소 주파수 |
| `engine_gpu_pct` | f64 | % | 엔진 GPU 사용률 |
| `external_gpu_pct` | f64 | % | 다른 앱 GPU 사용률 |
| `gpu_freq_mhz` | u32 | MHz | 현재 GPU 주파수 |
| `gpu_max_freq_mhz` | u32 | MHz | GPU 최대 주파수 |
| `gpu_min_freq_mhz` | u32 | MHz | GPU 최소 주파수 |
| `thermal_c` | f64 | °C | 집계 온도 (`derived`에서 재계산) |
| `cpu_cluster_thermal_c` | f64 | °C | CPU 클러스터 온도 |
| `gpu_cluster_thermal_c` | f64 | °C | GPU 클러스터 온도 |
| `throttle_threshold_c` | f64 | °C | throttling 진입 온도 |
| `phase` | String | — | `"idle"` / `"prefill"` / `"decode"` |
| `base_tps_decode_gpu` | f64 | tok/s | GPU 디코드 기준 처리량 |
| `base_tps_decode_cpu` | f64 | tok/s | CPU 디코드 기준 처리량 |
| `base_tps_decode_partition` | f64 | tok/s | partition 최적 ratio 기준 처리량 |
| `base_tps_prefill_gpu` | f64 | tok/s | GPU prefill 기준 처리량 |
| `active_device` | String | — | `"opencl"` / `"cpu"` |
| `active_actions` | Vec\<String\> | — | 활성 action 이름 목록 |
| `partition_ratio` | f64 | 0.0~1.0 | GPU/CPU 분할 비율 |
| `throttle_delay_ms` | f64 | ms | 현재 throttle 지연 |
| `tbt_target_ms` | f64 | ms | 목표 time-between-tokens |

---

## 4. Expression 문법

시나리오 YAML의 `when`, `factor`, `expr` 등에 사용하는 표현식 언어.

### 4.1 기본 규칙

- **평가기**: `evalexpr` crate (산술·논리·문자열 조건 지원)
- **변수 참조**: 베어 식별자 (중괄호 없음) — `target_ratio`, `thermal_c`
- **YAML 내 문자열**: 작은따옴표 YAML 스트링 안에 표현식을 그대로 작성
- **문자열 리터럴**: YAML 큰따옴표 + evalexpr 큰따옴표 중첩

```yaml
# 올바른 예
when: "target_ratio > 0 && target_ratio < 1"
factor: "max(0.3, 1.0 - delay_ms / 500.0)"

# 문자열 값 설정
active_device:
  op: set
  value: '"cpu"'   # YAML 작은따옴표 안에 evalexpr 큰따옴표
```

- **로드 시 AST 컴파일**: `evalexpr::build_operator_tree()`로 즉시 컴파일. 오타/미정의 함수는 YAML 파싱 시 에러
- **dry-run 바인딩**: 로드 직후 더미 바인딩으로 expression 전수 검증 → tick 중 panic 방지

### 4.2 내장 함수 6개

| 함수 | 시그니처 | 의미 |
|------|---------|------|
| `phase_throughput` | `(phase, device, ratio, cpu_freq_r, gpu_freq_r, bw_util, base_gpu, base_cpu, base_part, base_prefill) → f64` | 현재 phase/backend/partition 조합의 처리량 계산 |
| `throttle_factor` | `(delay_ms) → f64` | throttle 지연에 따른 처리량 배율 (`1/(1+delay_ms/10)`) |
| `skip_boost` | `(skip_ratio) → f64` | layer skip 비율에 따른 처리량 증가 (`1 + ratio*0.8`) |
| `base_by_phase` | `(phase) → f64` | idle→0, prefill→70, decode→10 (BW 기준 부하) |
| `dtype_from_bits` | `(bits) → String` | bits=16→`"f16"`, bits=8→`"q8"`, bits=4→`"q4"` |
| `merge_overhead` | `(ratio) → f64` | partition ratio에 따른 merge 오버헤드 (0~0.15) |

`derived.throughput_tps` 예시:

```yaml
derived:
  throughput_tps:
    expr: >-
      phase_throughput(phase, active_device, partition_ratio,
                       cpu_freq_mhz / cpu_max_freq_mhz,
                       gpu_freq_mhz / gpu_max_freq_mhz,
                       memory_bw_utilization_pct,
                       base_tps_decode_gpu, base_tps_decode_cpu,
                       base_tps_decode_partition, base_tps_prefill_gpu) *
      throttle_factor(throttle_delay_ms) *
      skip_boost(skip_ratio)
    tau_s: 0.0
```

### 4.3 `when` 조건식

action의 `when` 조건이 false이면 해당 tick에서 effect를 적용하지 않는다:

```yaml
actions:
  SwitchHw:
    when: "device == \"cpu\""   # device 인자가 "cpu"일 때만 적용
    active_device:
      op: set
      value: '"cpu"'
```

---

## 5. Action Composition

### 5.1 차원별 결합 연산자

여러 action이 동시에 활성화될 때 동일 차원(state variable)에 효과가 중복되면 `composition`의 연산자로 결합한다:

```yaml
composition:
  default: multiply        # 기본: scale 효과 곱셈
  per_dimension:
    throughput_tps: multiply
    engine_cpu_pct: max    # 경쟁 action 중 가장 높은 % 사용
    device_memory_used_mb: add    # 메모리 절감 누적
    thermal_c: add_delta   # delta를 합산
```

지원 연산자: `multiply`, `add`, `add_delta`, `max`, `min`

### 5.2 Interaction 항

특정 action 조합이 동시 활성화될 때 추가 효과를 적용한다:

```yaml
interactions:
  - actions: [Throttle, SetPartitionRatio]
    # Throttle + Partition 조합 시 merge 오버헤드가 throughput을 15% 추가 감소
    throughput_tps:
      op: scale
      factor: 0.85
```

### 5.3 적용 순서

매 tick 다음 순서로 상태 갱신:

```
1. DVFS 피드백 (thermal → freq)
2. passive_dynamics (열역학, KV 성장, BW)
3. external_injections (시간 윈도우 활성 구간)
4. active_actions effect 적용 (1st-order lag)
5. derived expression 재계산 (throughput_tps, tbt_ms 등)
```

---

## 6. Lua 스크립트 작성

### 6.1 기본 형태

```lua
function decide(ctx)
  local cmds = {}

  -- ctx.coef.pressure.memory: 0.0~1.0 정규화 메모리 압박
  local mem_pct = ctx.coef and ctx.coef.pressure and ctx.coef.pressure.memory or 0.0

  if mem_pct >= 0.7 then
    table.insert(cmds, { type = "kv_evict_sliding", keep_ratio = 0.5 })
  elseif mem_pct >= 0.4 then
    table.insert(cmds, { type = "kv_evict_sliding", keep_ratio = 0.8 })
  end

  return cmds
end
```

### 6.2 ctx 구조

`ctx` 테이블은 `LuaPolicy.build_ctx()`가 구성한다. 주요 필드:

```lua
ctx.signal = {
  memory  = { level = "Warning", available = ..., total = ... },
  compute = { level = "Normal",  cpu_pct = ..., gpu_pct = ... },
  thermal = { level = "Normal",  temperature_mc = ... },
  energy  = { level = "Normal",  power_budget_mw = ... },
}
ctx.coef = {
  pressure = { memory = 0.0~1.0, compute = 0.0~1.0 },
  trigger  = { mem_low = bool, compute_high = bool },
}
ctx.engine = {
  active_device  = "opencl" | "cpu",
  active_actions = { ... },
  cpu_pct        = ...,  -- 최근 heartbeat 기준
  gpu_pct        = ...,
}
```

`level` 문자열: `"Normal"`, `"Warning"`, `"Critical"`, `"Emergency"`

### 6.3 반환 커맨드 형식

반환 테이블의 `type` 필드가 `shared/src/lib.rs`의 `EngineCommand` variant에 매핑된다:

| Lua type 문자열 | EngineCommand variant | 필수 인자 |
|----------------|----------------------|-----------|
| `"kv_evict_sliding"` | `KvEvictSliding` | `keep_ratio: f64` |
| `"kv_evict_h2o"` | `KvEvictH2o` | `keep_ratio: f64` |
| `"kv_merge_d2o"` | `KvMergeD2o` | `keep_ratio: f64` |
| `"throttle"` | `Throttle` | `delay_ms: u64` |
| `"switch_hw"` | `SwitchHw` | `device: String` |
| `"set_partition_ratio"` | `SetPartitionRatio` | `ratio: f64` |
| `"layer_skip"` | `LayerSkip` | `ratio: f64` |
| `"restore_defaults"` | `RestoreDefaults` | — |

### 6.4 재사용 Lua fixture

`manager/tests/fixtures/sim/lua/` 에 5개 스크립트가 준비되어 있다:

| 파일 | 동작 |
|------|------|
| `memory_evict_graduated.lua` | pressure.memory ≥0.7→evict0.5, ≥0.4→evict0.8 |
| `always_evict_sliding.lua` | 모든 signal에 대해 keep_ratio=0.8 evict |
| `thermal_switch_backend.lua` | ThermalAlert Warning→CPU 전환 |
| `partition_adaptive.lua` | ComputeGuidance Warning→partition ratio 조정 |
| `memory_and_thermal_combined.lua` | memory + thermal 두 신호 동시 처리 |

---

## 7. Simulator 사용 패턴

### 7.1 LuaPolicy (권장)

```rust
use crate::common::sim::{config::load_scenario, harness::Simulator};
use llm_manager::config::AdaptationConfig;

let cfg = load_scenario("tests/fixtures/sim/scenarios/memory_pressure_steady.yaml")?;

// VirtualClockHandle이 자동으로 LuaPolicy에 주입된다.
let mut sim = Simulator::with_lua_policy(
    cfg,
    "tests/fixtures/sim/lua/memory_evict_graduated.lua",
    AdaptationConfig::default(),
)?;

sim.run_for(Duration::from_secs(30))?;
```

`with_lua_policy`는 내부적으로:
1. `Arc<Mutex<VirtualClock>>` 생성
2. `VirtualClockHandle::new(Arc::clone(&clock))` 어댑터 생성
3. `LuaPolicy::new(script, cfg, Arc::new(handle))` 주입
4. 동일 `Arc`를 `Simulator`가 보유 → `advance()` 호출 시 LuaPolicy에도 반영

### 7.2 MockPolicy (단위 테스트)

```rust
use crate::common::sim::mock_policy::MockPolicy;
use llm_shared::{EngineCommand, EngineDirective, Level, SystemSignal};

let mut mock = MockPolicy::new();
mock.directive_on_signal = Some(Box::new(|sig| {
    if let SystemSignal::MemoryPressure { level, .. } = sig {
        if *level >= Level::Warning {
            return Some(EngineDirective {
                seq_id: 1,
                commands: vec![EngineCommand::KvEvictSliding { keep_ratio: 0.8 }],
            });
        }
    }
    None
}));

let mut sim = Simulator::new(cfg, Box::new(mock));
sim.run_for(Duration::from_secs(5))?;
```

### 7.3 run_for vs run_until

```rust
// 고정 시간 실행
sim.run_for(Duration::from_secs(30))?;

// predicate 충족 시까지 (최대 max)
sim.run_until(
    |s| s.trajectory().directive_count() >= 3,
    Duration::from_secs(60),
)?;
```

`run_until`의 HORIZON은 60초로 고정 (`max_duration > 60s`이면 60s 범위까지만 이벤트 프리로드). 60초 초과 시나리오는 현재 미지원 (Known Limitations §13 참조).

### 7.4 tick 크기 변경

기본 tick은 50ms. 정밀도가 필요한 테스트에서 변경 가능:

```rust
let mut sim = Simulator::new(cfg, Box::new(mock))
    .with_tick_dt(Duration::from_millis(10));
```

---

## 8. Trajectory + Assertion

### 8.1 TrajectorySummary

```rust
use crate::common::sim::trajectory::TrajectorySummary;

let summary = TrajectorySummary::from_trajectory(
    sim.trajectory(),
    cfg.initial_state.cpu_max_freq_mhz as f64,
    cfg.initial_state.gpu_max_freq_mhz as f64,
);

println!("heartbeats: {}", summary.heartbeat_count);
println!("directives: {}", summary.directive_count);
println!("first directive at: {:?}", summary.first_directive_at_s);
println!("final throughput: {:?}", summary.state_final.map(|s| s.throughput_tps));
```

`TrajectorySummary` 필드:

| 필드 | 타입 | 의미 |
|------|------|------|
| `duration_s` | f64 | 총 시뮬 시간 (초) |
| `heartbeat_count` | usize | heartbeat 총 횟수 |
| `signal_count_by_kind` | BTreeMap\<String, usize\> | 종류별 signal 횟수 |
| `directive_count` | usize | 총 directive 횟수 |
| `directive_kinds` | BTreeMap\<String, usize\> | command 종류별 횟수 |
| `first_directive_at_s` | Option\<f64\> | 첫 directive 발생 시각 (초) |
| `state_final` | Option\<PhysicalStateSummary\> | 마지막 상태 요약 |

### 8.2 Trajectory 조회 메서드

```rust
let traj = sim.trajectory();

traj.heartbeat_count()                    // heartbeat 총 개수
traj.signal_count()                       // signal 총 개수
traj.signal_count_by_kind("memory_pressure")  // 종류별
traj.directive_count()                    // directive 총 개수
traj.first_directive_at_or_after(5.0)    // t≥5s 첫 directive 시각
traj.observation_due_count_for("kv_evict_sliding")  // ObservationDue 개수
traj.state_at(|s| s.throughput_tps, 15.0)  // t=15s 근방 처리량
traj.assert_contains_directive_kind("KvEvict")  // Ok / Err

traj.dump_json(path)?;      // JSON 덤프
traj.dump_csv_states(path)? // StateSnapshot만 CSV
```

### 8.3 relief_snapshot

`LuaPolicy`에만 유효. EwmaReliefTable의 현재 상태를 반환한다:

```rust
let relief = sim.policy.relief_snapshot().unwrap_or_default();
// HashMap<String, [f32; 6]>
// 키: action 이름 (예: "kv_evict_sliding")
// 값: [memory, compute, thermal, energy, gpu_pct, cpu_pct] relief 값
```

30초 이상 시뮬 + VirtualClockHandle 주입 시 relief가 비어있지 않아야 한다 (3초 관측 지연 충족).

---

## 9. insta 스냅샷 (Golden Testing)

### 9.1 최초 스냅샷 생성

```bash
# 최초 실행: 스냅샷이 없으면 INSTA_UPDATE 필요
INSTA_UPDATE=always cargo test -p llm_manager --test sim

# 또는 reject/accept 인터랙티브
cargo test -p llm_manager --test sim
cargo insta review
```

### 9.2 스냅샷 사용 패턴

Float은 반올림된 `TrajectorySummary`를 insta 대상으로 사용:

```rust
insta::with_settings!({ sort_maps => true, snapshot_suffix => "" }, {
    insta::assert_yaml_snapshot!("memory_pressure_summary", summary);
});
```

스냅샷 파일 위치: `manager/tests/sim/snapshots/*.snap`

### 9.3 정당한 변경 시

물리 모델, YAML 파라미터, 내장 함수 수정 등 legitimate 변경 시:

1. `INSTA_UPDATE=always cargo test -p llm_manager --test sim` 실행
2. `git diff manager/tests/sim/snapshots/` 로 diff 리뷰
3. 변경 의도와 일치하면 `git add` 후 커밋

---

## 10. 디바이스 Preset

### 10.1 제공 preset

| 파일 | 디바이스 | 상태 |
|------|---------|------|
| `baseline.yaml` | 일반 8 GB 디바이스 (추정치) | 기본 preset, 최상위 parent |
| `s25_galaxy.yaml` | Galaxy S25 (Snapdragon 8 Elite / Adreno 750) | 추정치, calibration 필요 |

### 10.2 s25_galaxy.yaml 주요 override

`baseline.yaml`에서 다음 값만 override된다:

- `device_memory_total_mb: 12288` (12 GB)
- `device_memory_used_mb: 8192` (일반 사용 중 추정)
- `gpu_max_freq_mhz: 1100`, `cpu_max_freq_mhz: 4200`
- `throttle_threshold_c: 82.0` (S25 실측 근사)
- `base_tps_decode_gpu: 18.5`, `base_tps_decode_cpu: 5.2`
- `dvfs.cpu.k_thermal: 0.12`, `dvfs.gpu.k_thermal: 0.15`
- `thermal_coupling.cpu_to_gpu: 0.18`, `gpu_to_cpu: 0.15`

`TODO(K):` 주석이 붙은 14개 필드는 실측 기반 refit이 필요하다 (calibration 도구 K 미완).

### 10.3 새 디바이스 preset 추가

1. `manager/tests/fixtures/sim/<device_name>.yaml` 생성
2. `extends: baseline.yaml` 선언 후 override 최소화
3. calibration이 필요한 필드에 `# TODO(K):` 주석 추가
4. smoke 테스트 추가 (`test_scenarios.rs` 패턴 참고)

```yaml
extends: baseline.yaml

initial_state:
  device_memory_total_mb: 16384  # Pixel 9 Pro XL
  gpu_max_freq_mhz: 1100
  # TODO(K): Tensor G4 주파수 실측 후 교체
  gpu_freq_mhz: 800
```

---

## 11. Noise (결정론 vs 확률적)

### 11.1 기본: 결정론

```yaml
rng_seed: ~   # null — noise 비활성화
```

`rng_seed: null`이면 `observation.*.noise.sigma`가 있어도 **모두 무시**된다. 스냅샷 테스트에 적합.

### 11.2 opt-in 확률적 모드

```yaml
rng_seed: 42  # 임의의 u64

observation:
  heartbeat:
    noise:
      throughput_tps: { sigma: 0.3, seed_key: "hb.tps" }
  signals:
    memory:
      noise:
        available_bytes: { sigma_mb: 8.0, seed_key: "sig.mem" }
```

- `seed_key`별로 독립 `ChaCha8Rng` 스트림 생성
- 동일 `rng_seed` → byte-level 동일 trajectory (재현성 보장)

```rust
// 재현성 검증 예시
let run1 = run_with_seed(42);
let run2 = run_with_seed(42);
assert_eq!(run1, run2, "동일 seed → 동일 trajectory");
```

---

## 12. Clock Abstraction (테스트 작성자용)

### 12.1 타입 계층

```
Clock (trait, manager/src/clock.rs)
  ├── SystemClock        — 실 wall-clock (프로덕션)
  └── VirtualClockHandle — VirtualClock 어댑터 (테스트)
         └── VirtualClock — Duration 누적 + BinaryHeap 이벤트 큐
```

자세한 설계는 `arch/clock_abstraction.md` 참조.

### 12.2 Simulator에서의 주입

`Simulator::with_lua_policy`가 자동으로 처리:

```
Arc<Mutex<VirtualClock>>
  ├─ Simulator가 보유 → tick마다 advance() 호출
  └─ VirtualClockHandle(Arc::clone) → LuaPolicy에 주입
```

`LuaPolicy`의 observation_delay(3초)는 `VirtualClockHandle::now()`를 통해 가상 시간 기준으로 계산된다. 따라서 실제 wall-clock이 100ms만 걸려도 가상 시간상 3초가 지나면 relief 학습이 정상 작동한다.

### 12.3 프로덕션 경로

`LuaPolicy::with_system_clock`은 `SystemClock`을 주입한다. 테스트에서 직접 wall-clock 의존 LuaPolicy를 생성하는 경우:

```rust
let lua_policy = LuaPolicy::with_system_clock(
    script_path,
    AdaptationConfig::default(),
)?;
```

이 경우 `relief_snapshot()`은 wall-clock 3초 지연을 기다려야 하므로 단기 시뮬에서는 비어있을 수 있다. 항상 `Simulator::with_lua_policy`를 사용할 것을 권장한다.

---

## 13. Known Limitations / TODO

| 이슈 | 현황 | 우선순위 제안 |
|------|------|-------------|
| `HierarchicalPolicy` Clock 주입 미완 | PR 4 계획 | 중 (HierarchicalPolicy 사용 테스트 필요 시) |
| FPS helper 잔여 `Instant` 참조 | PR 5 계획 | 낮 (테스트 영향 없음) |
| `derived` 토폴로지 정렬 미구현 | insertion order 의존 | 낮 (현재 순서 변경 없음) |
| Calibration 자동화 (K) 미구현 | 수동 추정치 사용 중 | 높 (s25_galaxy.yaml 14개 TODO 필드) |
| 60초 초과 시나리오 lazy re-inject | HORIZON=60s 고정 | 중 (장기 thermal 시나리오 필요 시) |
| `Simulator::with_lua_policy` signature에 `AdaptationConfig` 요구 | — | `AdaptationConfig::default()` 사용 가능 |

---

## 14. 관련 파일 / 참고

### 소스 파일

| 경로 | 역할 |
|------|------|
| `manager/tests/common/sim/mod.rs` | 모듈 re-export |
| `manager/tests/common/sim/config.rs` | YAML 스키마 (serde + validator) |
| `manager/tests/common/sim/expr.rs` | evalexpr 래퍼, 내장 함수 6개 |
| `manager/tests/common/sim/state.rs` | PhysicalState, EngineStateModel |
| `manager/tests/common/sim/physics.rs` | 1st-order lag, DVFS, thermal coupling |
| `manager/tests/common/sim/compose.rs` | Action composition + interaction |
| `manager/tests/common/sim/signal.rs` | PhysicalState → SystemSignal 투영 |
| `manager/tests/common/sim/clock.rs` | VirtualClock + BinaryHeap 이벤트 큐 |
| `manager/tests/common/sim/clock_adapter.rs` | VirtualClockHandle (Clock trait 어댑터) |
| `manager/tests/common/sim/noise.rs` | ChaCha8Rng 기반 노이즈 |
| `manager/tests/common/sim/harness.rs` | Simulator 메인 구조 |
| `manager/tests/common/sim/trajectory.rs` | Trajectory, TrajectorySummary |
| `manager/tests/common/sim/mock_policy.rs` | MockPolicy (테스트 전용) |

### Fixture 파일

| 경로 | 설명 |
|------|------|
| `manager/tests/fixtures/sim/baseline.yaml` | 기본 preset |
| `manager/tests/fixtures/sim/s25_galaxy.yaml` | Galaxy S25 preset |
| `manager/tests/fixtures/sim/scenarios/*.yaml` | 3종 시나리오 |
| `manager/tests/fixtures/sim/lua/*.lua` | 5종 Lua 스크립트 |
| `manager/tests/sim/snapshots/*.snap` | insta 스냅샷 |

### 예시 테스트 파일

| 경로 | 내용 |
|------|------|
| `manager/tests/sim/test_harness.rs` | Simulator 기본 사용 (12개 테스트) |
| `manager/tests/sim/test_scenarios.rs` | 시나리오 기반 + insta 스냅샷 |
| `manager/tests/spec/test_mgr_alg_080_083_ewma_relief.rs` | EwmaReliefTable 직접 + LuaPolicy 블랙박스 |
| `manager/tests/sim/test_config.rs` | YAML 파싱/상속/검증 |
| `manager/tests/sim/test_physics.rs` | 물리 엔진 단위 테스트 |

### 설계 문서

- `arch/clock_abstraction.md` — Clock trait / VirtualClock / 주입 패턴 설계 결정
- `spec/` — MGR-ALG-080~083, MGR-DAT-070~074, INV-086~090, MSG-060
