# LuaPolicy Dynamic Coefficient Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** LuaPolicy가 EWMA로 학습되는 동적 계수(relief, pressure, trigger)를 `ctx.coef`로 받아 정책을 결정하도록 하고, HierarchicalPolicy를 deprecated 처리한다.

**Architecture:** LuaPolicy 내부에 SignalState(6D pressure), TriggerEngine(3개 trigger + hysteresis), EwmaReliefTable(per-action EWMA 학습)을 추가. SystemSignal을 캐시하여 pressure/trigger를 계산하고 `build_ctx()`에서 Lua에 노출. HierarchicalPolicy와 의존 컴포넌트는 `hierarchical` feature flag 뒤로 이동.

**Tech Stack:** Rust, mlua (Lua 54), serde/serde_json (영속화), toml (config)

**Spec:** `docs/superpowers/specs/2026-04-10-lua-policy-dynamic-coef-design.md`

---

### Task 1: Feature flag 추가 및 HierarchicalPolicy deprecated 처리

**Files:**
- Modify: `manager/Cargo.toml`
- Modify: `manager/src/lib.rs`
- Modify: `manager/src/main.rs`

- [ ] **Step 1: `Cargo.toml`에 `hierarchical` feature 추가**

```toml
[features]
dbus = ["zbus"]
lua = ["mlua"]
hierarchical = []
default = ["dbus", "lua"]
```

`hierarchical`은 default에 포함하지 않는다. `lua`를 default에 추가한다 (LuaPolicy가 기본 정책이므로).

- [ ] **Step 2: `lib.rs`에 conditional compilation 적용**

```rust
#[cfg(feature = "hierarchical")]
pub mod action_registry;
pub mod channel;
pub mod config;
pub mod emitter;
#[cfg(feature = "hierarchical")]
pub mod evaluator;
#[cfg(feature = "lua")]
pub mod lua_policy;
pub mod monitor;
#[cfg(feature = "hierarchical")]
pub mod pi_controller;
pub mod pipeline;
#[cfg(feature = "hierarchical")]
pub mod relief;
#[cfg(feature = "hierarchical")]
pub mod selector;
#[cfg(feature = "hierarchical")]
pub mod supervisory;
pub mod types;
```

- [ ] **Step 3: `pipeline.rs`에서 HierarchicalPolicy를 conditional로 변경**

`pipeline.rs`의 `PolicyStrategy` trait 정의와 `next_seq_id()` 함수는 유지. `HierarchicalPolicy` struct/impl 전체와 그에 필요한 `use` 문에 `#[cfg(feature = "hierarchical")]` 적용.

파일 상단의 use 문 중 hierarchical 전용:
```rust
#[cfg(feature = "hierarchical")]
use crate::action_registry::ActionRegistry;
#[cfg(feature = "hierarchical")]
use crate::evaluator::ThresholdEvaluator;
// ... 기타 hierarchical 전용 use
```

`PolicyStrategy` trait (L34-58)과 `next_seq_id()` 함수는 그대로 유지.
`HierarchicalPolicy` struct (L100-130)부터 파일 끝까지 (impl + tests 포함) 전체를 `#[cfg(feature = "hierarchical")]`로 감싼다.

- [ ] **Step 4: `types.rs`에서 HierarchicalPolicy 전용 타입을 conditional로 변경**

`OperatingMode` enum은 유지 (LuaPolicy의 `mode()` 반환값으로 사용).

나머지에 `#[cfg(feature = "hierarchical")]` 적용:
- `ActionId` enum (L7)
- `ActionKind` enum (L69)
- `Domain` enum (L76)
- `PressureVector` struct (L92)
- `ReliefVector` struct (L125)
- `FeatureVector` struct (L163)
- `feature` mod (L176)
- `ActionMeta` struct (L194)
- `ParamRange` struct (L204)
- `ActionParams` struct (L212)
- `ActionCommand` struct (L218)
- `Operation` enum (L224)

- [ ] **Step 5: `config.rs`에서 HierarchicalPolicy 전용 config를 conditional로 변경**

`Config`, `ManagerConfig`, 각 MonitorConfig는 유지.

`#[cfg(feature = "hierarchical")]` 적용:
- `PolicyConfig` struct (L176)
- `PiControllerConfig` struct (L188)
- `SupervisoryConfig` struct (L226)
- `SelectorConfig` struct (L249)
- `ReliefModelConfig` struct (L266)
- `ActionConfig` struct (L285)

- [ ] **Step 6: `main.rs`에서 `create_policy()` 수정**

```rust
/// `--policy-script`가 지정되면 LuaPolicy를, 아니면 에러를 반환한다.
/// HierarchicalPolicy는 --features hierarchical로 빌드 시에만 사용 가능.
fn create_policy(args: &Args, config: &Config) -> anyhow::Result<Box<dyn PolicyStrategy>> {
    if let Some(ref script_path) = args.policy_script {
        return create_lua_policy(script_path);
    }

    #[cfg(feature = "hierarchical")]
    {
        let policy_cfg = load_policy_config(args, config);
        let mut p = HierarchicalPolicy::new(&policy_cfg);
        let storage_dir = if policy_cfg.relief_model.storage_dir.starts_with('~') {
            let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
            policy_cfg.relief_model.storage_dir.replacen('~', &home, 1)
        } else {
            policy_cfg.relief_model.storage_dir.clone()
        };
        p.set_relief_model_path(&storage_dir);
        log::info!("HierarchicalPolicy initialized");
        return Ok(Box::new(p));
    }

    #[cfg(not(feature = "hierarchical"))]
    {
        anyhow::bail!(
            "--policy-script is required. HierarchicalPolicy is deprecated \
             (enable with: cargo build --features hierarchical)"
        )
    }
}
```

`use llm_manager::pipeline::HierarchicalPolicy`에도 `#[cfg(feature = "hierarchical")]` 적용.

- [ ] **Step 7: 빌드 확인**

```bash
# default features (dbus + lua, hierarchical 없음)
cargo build -p llm_manager
# hierarchical feature 포함
cargo build -p llm_manager --features hierarchical
```

둘 다 컴파일 성공해야 한다. 경고가 나올 수 있지만 에러는 없어야 함.

- [ ] **Step 8: 기존 테스트 확인**

```bash
# hierarchical 포함 시 기존 테스트 통과
cargo test -p llm_manager --features hierarchical
# default (hierarchical 없음) 시 테스트 통과 (hierarchical 테스트는 스킵)
cargo test -p llm_manager
```

- [ ] **Step 9: 커밋**

```bash
git add manager/
git commit -m "refactor(manager): deprecate HierarchicalPolicy behind feature flag

Move PI Controller, SupervisoryLayer, ActionSelector, ReliefEstimator,
and related types behind #[cfg(feature = \"hierarchical\")].
LuaPolicy becomes the default and only policy without the flag."
```

---

### Task 2: AdaptationConfig 추가

**Files:**
- Modify: `manager/src/config.rs`

- [ ] **Step 1: `AdaptationConfig` 구조체 추가**

`config.rs` 끝에 추가 (기존 config 타입 아래):

```rust
/// Configuration for LuaPolicy online adaptation (trigger, EWMA, relief defaults).
#[derive(Debug, Clone, Deserialize)]
pub struct AdaptationConfig {
    /// EWMA smoothing factor (default: 0.875 = 7/8, Jacobson TCP RTT).
    #[serde(default = "default_ewma_alpha")]
    pub ewma_alpha: f32,

    /// Path to save/load the learned relief table (empty = disabled).
    #[serde(default)]
    pub relief_table_path: String,

    /// Safe temperature baseline for thermal normalization (Celsius).
    #[serde(default = "default_temp_safe")]
    pub temp_safe_c: f32,

    /// Critical temperature ceiling for thermal normalization (Celsius).
    #[serde(default = "default_temp_critical")]
    pub temp_critical_c: f32,

    /// Trigger thresholds.
    #[serde(default)]
    pub trigger: TriggerConfig,

    /// Per-action default relief values [gpu, cpu, memory, thermal, latency, main_app_qos].
    #[serde(default)]
    pub default_relief: std::collections::HashMap<String, Vec<f32>>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TriggerConfig {
    #[serde(default = "default_tbt_enter")]
    pub tbt_enter: f64,
    #[serde(default = "default_tbt_exit")]
    pub tbt_exit: f64,
    #[serde(default = "default_tbt_warmup")]
    pub tbt_warmup_tokens: u32,
    #[serde(default = "default_mem_enter")]
    pub mem_enter: f64,
    #[serde(default = "default_mem_exit")]
    pub mem_exit: f64,
    #[serde(default = "default_temp_enter")]
    pub temp_enter: f64,
    #[serde(default = "default_temp_exit")]
    pub temp_exit: f64,
}

impl Default for AdaptationConfig {
    fn default() -> Self {
        Self {
            ewma_alpha: 0.875,
            relief_table_path: String::new(),
            temp_safe_c: 35.0,
            temp_critical_c: 50.0,
            trigger: TriggerConfig::default(),
            default_relief: std::collections::HashMap::new(),
        }
    }
}

impl Default for TriggerConfig {
    fn default() -> Self {
        Self {
            tbt_enter: 0.30,
            tbt_exit: 0.10,
            tbt_warmup_tokens: 20,
            mem_enter: 0.80,
            mem_exit: 0.60,
            temp_enter: 0.70,
            temp_exit: 0.50,
        }
    }
}

fn default_ewma_alpha() -> f32 { 0.875 }
fn default_temp_safe() -> f32 { 35.0 }
fn default_temp_critical() -> f32 { 50.0 }
fn default_tbt_enter() -> f64 { 0.30 }
fn default_tbt_exit() -> f64 { 0.10 }
fn default_tbt_warmup() -> u32 { 20 }
fn default_mem_enter() -> f64 { 0.80 }
fn default_mem_exit() -> f64 { 0.60 }
fn default_temp_enter() -> f64 { 0.70 }
fn default_temp_exit() -> f64 { 0.50 }
```

- [ ] **Step 2: `Config` struct에 `adaptation` 필드 추가**

```rust
pub struct Config {
    // ... 기존 필드 ...

    /// Online adaptation settings for LuaPolicy.
    #[serde(default)]
    pub adaptation: AdaptationConfig,
}
```

- [ ] **Step 3: 빌드 확인**

```bash
cargo build -p llm_manager
```

- [ ] **Step 4: 커밋**

```bash
git add manager/src/config.rs
git commit -m "feat(manager): add AdaptationConfig for LuaPolicy dynamic coefficients

Includes trigger thresholds (TBT/mem/temp enter/exit), EWMA alpha,
thermal normalization constants, relief table path, and per-action
default relief values."
```

---

### Task 3: EwmaReliefTable 구현 + 테스트

**Files:**
- Modify: `manager/src/lua_policy.rs` (모듈 내부에 struct 추가)
- Test: `manager/src/lua_policy.rs` (인라인 `#[cfg(test)]`)

- [ ] **Step 1: EwmaReliefTable 테스트 작성**

`lua_policy.rs` 끝에 테스트 모듈 추가:

```rust
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ewma_first_observation_replaces_default() {
        let mut table = EwmaReliefTable::new(0.875, std::collections::HashMap::new());
        let observed = [0.6, -0.2, 0.0, 0.4, -0.1, 0.0];
        table.observe("switch_hw", &observed);

        let predicted = table.predict("switch_hw");
        assert_eq!(predicted, observed);
        assert_eq!(table.observation_count("switch_hw"), 1);
    }

    #[test]
    fn ewma_converges_toward_observed() {
        let mut table = EwmaReliefTable::new(0.875, std::collections::HashMap::new());
        // 첫 관측: 직접 대입
        table.observe("throttle", &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);

        // 반복 관측: 0.2로 수렴해야 함
        for _ in 0..50 {
            table.observe("throttle", &[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }

        let predicted = table.predict("throttle");
        assert!((predicted[0] - 0.2).abs() < 0.01, "gpu relief should converge to 0.2, got {}", predicted[0]);
    }

    #[test]
    fn ewma_unknown_action_returns_default_from_config() {
        let mut defaults = std::collections::HashMap::new();
        defaults.insert("switch_hw".to_string(), vec![0.5, -0.3, 0.0, 0.3, -0.1, 0.0]);

        let table = EwmaReliefTable::new(0.875, defaults);
        let predicted = table.predict("switch_hw");
        assert_eq!(predicted, [0.5, -0.3, 0.0, 0.3, -0.1, 0.0]);
    }

    #[test]
    fn ewma_unknown_action_no_config_returns_zeros() {
        let table = EwmaReliefTable::new(0.875, std::collections::HashMap::new());
        let predicted = table.predict("nonexistent");
        assert_eq!(predicted, [0.0; 6]);
    }

    #[test]
    fn ewma_save_load_roundtrip() {
        let mut table = EwmaReliefTable::new(0.875, std::collections::HashMap::new());
        table.observe("switch_hw", &[0.5, -0.3, 0.0, 0.3, -0.1, 0.0]);
        table.observe("throttle", &[0.0, 0.3, 0.0, 0.2, -0.2, 0.0]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("relief.json");

        table.save(&path).unwrap();
        let loaded = EwmaReliefTable::load(&path, 0.875, std::collections::HashMap::new()).unwrap();

        assert_eq!(loaded.predict("switch_hw"), table.predict("switch_hw"));
        assert_eq!(loaded.predict("throttle"), table.predict("throttle"));
        assert_eq!(loaded.observation_count("switch_hw"), 1);
    }

    #[test]
    fn ewma_negative_relief_handled() {
        let mut table = EwmaReliefTable::new(0.875, std::collections::HashMap::new());
        table.observe("switch_hw", &[0.5, -0.3, 0.0, 0.3, -0.1, 0.0]);

        let predicted = table.predict("switch_hw");
        assert!(predicted[1] < 0.0, "cpu relief should be negative for switch_hw");
        assert!(predicted[4] < 0.0, "latency relief should be negative for switch_hw");
    }
}
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cargo test -p llm_manager --features lua ewma -- --nocapture
```

Expected: FAIL — `EwmaReliefTable` 미정의.

- [ ] **Step 3: EwmaReliefTable 구현**

`lua_policy.rs` 내부, `LuaPolicy` struct 위쪽에 추가:

```rust
use serde::{Serialize, Deserialize};
use std::collections::HashMap;
use std::path::Path;

const RELIEF_DIMS: usize = 6; // gpu, cpu, memory, thermal, latency, main_app_qos

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ReliefEntry {
    relief: [f32; RELIEF_DIMS],
    observation_count: u32,
}

#[derive(Debug)]
struct EwmaReliefTable {
    entries: HashMap<String, ReliefEntry>,
    alpha: f32,
    defaults: HashMap<String, Vec<f32>>,
}

impl EwmaReliefTable {
    fn new(alpha: f32, defaults: HashMap<String, Vec<f32>>) -> Self {
        Self {
            entries: HashMap::new(),
            alpha,
            defaults,
        }
    }

    fn predict(&self, action: &str) -> [f32; RELIEF_DIMS] {
        if let Some(entry) = self.entries.get(action) {
            return entry.relief;
        }
        if let Some(default) = self.defaults.get(action) {
            let mut relief = [0.0f32; RELIEF_DIMS];
            for (i, v) in default.iter().enumerate().take(RELIEF_DIMS) {
                relief[i] = *v;
            }
            return relief;
        }
        [0.0; RELIEF_DIMS]
    }

    fn observe(&mut self, action: &str, observed: &[f32; RELIEF_DIMS]) {
        let entry = self.entries.entry(action.to_string()).or_insert_with(|| ReliefEntry {
            relief: [0.0; RELIEF_DIMS],
            observation_count: 0,
        });

        if entry.observation_count == 0 {
            entry.relief = *observed;
        } else {
            let a = self.alpha;
            for i in 0..RELIEF_DIMS {
                entry.relief[i] = a * entry.relief[i] + (1.0 - a) * observed[i];
            }
        }
        entry.observation_count += 1;
    }

    fn observation_count(&self, action: &str) -> u32 {
        self.entries.get(action).map_or(0, |e| e.observation_count)
    }

    fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.entries)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        std::fs::write(path, json)
    }

    fn load(
        path: &Path,
        alpha: f32,
        defaults: HashMap<String, Vec<f32>>,
    ) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let entries: HashMap<String, ReliefEntry> = serde_json::from_str(&json)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::Other, e))?;
        Ok(Self { entries, alpha, defaults })
    }
}
```

- [ ] **Step 4: 테스트 실행 — 통과 확인**

```bash
cargo test -p llm_manager --features lua ewma -- --nocapture
```

Expected: 6 tests PASS.

- [ ] **Step 5: 커밋**

```bash
git add manager/src/lua_policy.rs
git commit -m "feat(manager): add EwmaReliefTable with per-action 6D EWMA learning

Jacobson TCP RTT style: first observation replaces default, subsequent
observations use EWMA (alpha=0.875). Supports JSON save/load and
per-action config defaults."
```

---

### Task 4: SignalState + TriggerEngine 구현 + 테스트

**Files:**
- Modify: `manager/src/lua_policy.rs`

- [ ] **Step 1: 테스트 작성**

기존 tests 모듈에 추가:

```rust
    #[test]
    fn signal_state_pressure_from_compute() {
        let mut state = SignalState::default();
        state.update_compute(45.2, 82.1);
        let p = state.pressure(None);
        assert!((p.gpu - 0.821).abs() < 0.001);
        assert!((p.cpu - 0.452).abs() < 0.001);
    }

    #[test]
    fn signal_state_pressure_from_memory() {
        let mut state = SignalState::default();
        state.update_memory(2_000_000, 8_000_000);
        let p = state.pressure(None);
        assert!((p.memory - 0.75).abs() < 0.001); // 1 - 2M/8M = 0.75
    }

    #[test]
    fn signal_state_pressure_thermal_normalized() {
        let mut state = SignalState::default();
        // 42.5°C, safe=35, critical=50 → (42.5-35)/(50-35) = 0.5
        state.update_thermal(42500, false); // millidegrees
        let p = state.pressure_with_thermal(35.0, 50.0, None);
        assert!((p.thermal - 0.5).abs() < 0.01);
    }

    #[test]
    fn signal_state_pressure_thermal_clamped() {
        let mut state = SignalState::default();
        state.update_thermal(55000, true); // above critical
        let p = state.pressure_with_thermal(35.0, 50.0, None);
        assert!((p.thermal - 1.0).abs() < 0.001); // clamped to 1.0
    }

    #[test]
    fn trigger_tbt_warmup_then_degrade() {
        let config = TriggerConfig {
            tbt_enter: 0.30, tbt_exit: 0.10, tbt_warmup_tokens: 5,
            mem_enter: 0.80, mem_exit: 0.60, temp_enter: 0.70, temp_exit: 0.50,
        };
        let mut engine = TriggerEngine::new(config);

        // Warmup: 5 tokens at 10 tok/s → baseline TBT ~100ms
        for _ in 0..5 {
            engine.update_tbt_from_throughput(10.0);
        }
        assert!(!engine.state().tbt_degraded);

        // Degrade: throughput drops to 5 tok/s → TBT 200ms → 100% increase > 30%
        for _ in 0..10 {
            engine.update_tbt_from_throughput(5.0);
        }
        assert!(engine.state().tbt_degraded);
    }

    #[test]
    fn trigger_tbt_zero_throughput_skipped() {
        let config = TriggerConfig::default();
        let mut engine = TriggerEngine::new(config);
        engine.update_tbt_from_throughput(0.0); // idle/prefill
        assert!(!engine.state().tbt_degraded);
    }

    #[test]
    fn trigger_hysteresis_mem() {
        let config = TriggerConfig {
            tbt_enter: 0.30, tbt_exit: 0.10, tbt_warmup_tokens: 20,
            mem_enter: 0.80, mem_exit: 0.60, temp_enter: 0.70, temp_exit: 0.50,
        };
        let mut engine = TriggerEngine::new(config);

        // Enter: pressure 0.85 > 0.80
        engine.update_mem(0.85);
        assert!(engine.state().mem_low);

        // Still active at 0.65 (between exit=0.60 and enter=0.80)
        engine.update_mem(0.65);
        assert!(engine.state().mem_low);

        // Exit: pressure 0.55 < 0.60
        engine.update_mem(0.55);
        assert!(!engine.state().mem_low);
    }

    #[test]
    fn trigger_hysteresis_temp() {
        let config = TriggerConfig {
            tbt_enter: 0.30, tbt_exit: 0.10, tbt_warmup_tokens: 20,
            mem_enter: 0.80, mem_exit: 0.60, temp_enter: 0.70, temp_exit: 0.50,
        };
        let mut engine = TriggerEngine::new(config);

        engine.update_temp(0.75); // > 0.70 enter
        assert!(engine.state().temp_high);

        engine.update_temp(0.55); // between exit=0.50 and enter=0.70
        assert!(engine.state().temp_high);

        engine.update_temp(0.45); // < 0.50 exit
        assert!(!engine.state().temp_high);
    }
```

- [ ] **Step 2: 테스트 실행 — 실패 확인**

```bash
cargo test -p llm_manager --features lua signal_state -- --nocapture
cargo test -p llm_manager --features lua trigger -- --nocapture
```

Expected: FAIL — `SignalState`, `TriggerEngine` 미정의.

- [ ] **Step 3: Pressure6D, SignalState 구현**

`lua_policy.rs` 내부, `EwmaReliefTable` 위쪽에 추가:

```rust
#[derive(Debug, Clone, Default)]
struct Pressure6D {
    gpu: f32,
    cpu: f32,
    memory: f32,
    thermal: f32,
    latency: f32,
    main_app: f32,
}

#[derive(Debug, Default)]
struct SignalState {
    cpu_pct: f64,
    gpu_pct: f64,
    mem_available: u64,
    mem_total: u64,
    temp_mc: i32,
    throttling: bool,
}

impl SignalState {
    fn update_compute(&mut self, cpu_pct: f64, gpu_pct: f64) {
        self.cpu_pct = cpu_pct;
        self.gpu_pct = gpu_pct;
    }

    fn update_memory(&mut self, available: u64, total: u64) {
        self.mem_available = available;
        self.mem_total = total;
    }

    fn update_thermal(&mut self, temp_mc: i32, throttling: bool) {
        self.temp_mc = temp_mc;
        self.throttling = throttling;
    }

    fn pressure(&self, latency_ratio: Option<f64>) -> Pressure6D {
        self.pressure_with_thermal(35.0, 50.0, latency_ratio)
    }

    fn pressure_with_thermal(
        &self,
        temp_safe_c: f32,
        temp_critical_c: f32,
        latency_ratio: Option<f64>,
    ) -> Pressure6D {
        let mem_pressure = if self.mem_total > 0 {
            1.0 - (self.mem_available as f32 / self.mem_total as f32)
        } else {
            0.0
        };

        let temp_c = self.temp_mc as f32 / 1000.0;
        let temp_range = temp_critical_c - temp_safe_c;
        let thermal = if temp_range > 0.0 {
            ((temp_c - temp_safe_c) / temp_range).clamp(0.0, 1.0)
        } else {
            0.0
        };

        Pressure6D {
            gpu: (self.gpu_pct as f32 / 100.0).clamp(0.0, 1.0),
            cpu: (self.cpu_pct as f32 / 100.0).clamp(0.0, 1.0),
            memory: mem_pressure.clamp(0.0, 1.0),
            thermal,
            latency: latency_ratio.unwrap_or(0.0) as f32,
            main_app: 0.0,
        }
    }
}
```

- [ ] **Step 4: TriggerEngine 구현**

```rust
use crate::config::TriggerConfig;

#[derive(Debug, Clone, Default)]
struct TriggerState {
    tbt_degraded: bool,
    mem_low: bool,
    temp_high: bool,
}

#[derive(Debug)]
struct TbtTracker {
    ewma: f64,
    baseline: Option<f64>,
    warmup_count: u32,
    warmup_target: u32,
}

impl TbtTracker {
    fn new(warmup_target: u32) -> Self {
        Self {
            ewma: 0.0,
            baseline: None,
            warmup_count: 0,
            warmup_target,
        }
    }

    /// Update with a new TBT value (ms). Returns None during warmup.
    fn observe(&mut self, tbt_ms: f64) {
        if self.warmup_count == 0 {
            self.ewma = tbt_ms;
        } else {
            self.ewma = 0.875 * self.ewma + 0.125 * tbt_ms;
        }
        self.warmup_count += 1;

        if self.baseline.is_none() && self.warmup_count >= self.warmup_target {
            self.baseline = Some(self.ewma);
        }
    }

    fn degradation_ratio(&self) -> Option<f64> {
        self.baseline.map(|b| {
            if b > 0.0 { (self.ewma - b) / b } else { 0.0 }
        })
    }
}

#[derive(Debug)]
struct TriggerEngine {
    config: TriggerConfig,
    tbt: TbtTracker,
    trigger: TriggerState,
}

impl TriggerEngine {
    fn new(config: TriggerConfig) -> Self {
        Self {
            tbt: TbtTracker::new(config.tbt_warmup_tokens),
            config,
            trigger: TriggerState::default(),
        }
    }

    fn update_tbt_from_throughput(&mut self, throughput: f32) {
        if throughput <= 0.0 {
            return; // idle/prefill — skip
        }
        let tbt_ms = 1000.0 / throughput as f64;
        self.tbt.observe(tbt_ms);

        if let Some(ratio) = self.tbt.degradation_ratio() {
            if self.trigger.tbt_degraded {
                if ratio < self.config.tbt_exit {
                    self.trigger.tbt_degraded = false;
                }
            } else if ratio > self.config.tbt_enter {
                self.trigger.tbt_degraded = true;
            }
        }
    }

    fn update_mem(&mut self, pressure: f64) {
        if self.trigger.mem_low {
            if pressure < self.config.mem_exit {
                self.trigger.mem_low = false;
            }
        } else if pressure > self.config.mem_enter {
            self.trigger.mem_low = true;
        }
    }

    fn update_temp(&mut self, normalized: f64) {
        if self.trigger.temp_high {
            if normalized < self.config.temp_exit {
                self.trigger.temp_high = false;
            }
        } else if normalized > self.config.temp_enter {
            self.trigger.temp_high = true;
        }
    }

    fn state(&self) -> &TriggerState {
        &self.trigger
    }

    fn tbt_degradation_ratio(&self) -> Option<f64> {
        self.tbt.degradation_ratio()
    }
}
```

- [ ] **Step 5: 테스트 실행 — 통과 확인**

```bash
cargo test -p llm_manager --features lua signal_state -- --nocapture
cargo test -p llm_manager --features lua trigger -- --nocapture
```

Expected: 8 tests PASS.

- [ ] **Step 6: 커밋**

```bash
git add manager/src/lua_policy.rs
git commit -m "feat(manager): add SignalState (6D pressure) and TriggerEngine (3 triggers + hysteresis)

SignalState caches SystemSignal data and computes 6D pressure vector.
TriggerEngine tracks TBT baseline via EWMA (1000/throughput), memory
pressure, and thermal normalized values with enter/exit hysteresis."
```

---

### Task 5: LuaPolicy 통합 — process_signal, build_ctx, observation

**Files:**
- Modify: `manager/src/lua_policy.rs`

- [ ] **Step 1: LuaPolicy 구조체 확장**

기존 `LuaPolicy` 구조체를 수정:

```rust
pub struct LuaPolicy {
    lua: Lua,
    engine_state: Option<EngineStatus>,
    // ── 신규 필드 ──
    signal_state: SignalState,
    trigger_engine: TriggerEngine,
    relief_table: EwmaReliefTable,
    observation: Option<ObservationContext>,
    adaptation_config: AdaptationConfig,
}

struct ObservationContext {
    action: String,
    before: Pressure6D,
    timestamp: Instant,
}

const OBSERVATION_DELAY_SECS: f64 = 3.0;
```

- [ ] **Step 2: `LuaPolicy::new()` 수정**

config 경로를 추가 인자로 받거나, `AdaptationConfig`를 직접 받는다. relief table을 load 시도:

```rust
impl LuaPolicy {
    pub fn new(script_path: &str, config: AdaptationConfig) -> anyhow::Result<Self> {
        // ... 기존 Lua VM 초기화 코드 유지 ...

        let relief_table = if !config.relief_table_path.is_empty() {
            let path = std::path::Path::new(&config.relief_table_path);
            EwmaReliefTable::load(path, config.ewma_alpha, config.default_relief.clone())
                .unwrap_or_else(|_| {
                    log::info!("No existing relief table at {}, starting fresh", config.relief_table_path);
                    EwmaReliefTable::new(config.ewma_alpha, config.default_relief.clone())
                })
        } else {
            EwmaReliefTable::new(config.ewma_alpha, config.default_relief.clone())
        };

        let trigger_engine = TriggerEngine::new(config.trigger.clone());

        Ok(Self {
            lua,
            engine_state: None,
            signal_state: SignalState::default(),
            trigger_engine,
            relief_table,
            observation: None,
            adaptation_config: config,
        })
    }
}
```

- [ ] **Step 3: `process_signal()` 수정 — SystemSignal 활용**

```rust
impl PolicyStrategy for LuaPolicy {
    fn process_signal(&mut self, signal: &SystemSignal) -> Option<EngineDirective> {
        // 1. SignalState 업데이트
        match signal {
            SystemSignal::MemoryPressure { available_bytes, total_bytes, .. } => {
                self.signal_state.update_memory(*available_bytes, *total_bytes);
                let pressure = if *total_bytes > 0 {
                    1.0 - (*available_bytes as f64 / *total_bytes as f64)
                } else {
                    0.0
                };
                self.trigger_engine.update_mem(pressure);
            }
            SystemSignal::ComputeGuidance { cpu_usage_pct, gpu_usage_pct, .. } => {
                self.signal_state.update_compute(*cpu_usage_pct, *gpu_usage_pct);
            }
            SystemSignal::ThermalAlert { temperature_mc, throttling_active, .. } => {
                self.signal_state.update_thermal(*temperature_mc, *throttling_active);
                let temp_c = *temperature_mc as f64 / 1000.0;
                let safe = self.adaptation_config.temp_safe_c as f64;
                let critical = self.adaptation_config.temp_critical_c as f64;
                let range = critical - safe;
                let normalized = if range > 0.0 {
                    ((temp_c - safe) / range).clamp(0.0, 1.0)
                } else {
                    0.0
                };
                self.trigger_engine.update_temp(normalized);
            }
            SystemSignal::EnergyConstraint { .. } => {
                // Energy는 trigger에 포함되지 않음 (초기 구현)
            }
        }

        // 2. TBT 업데이트 (throughput에서 역산)
        if let Some(ref status) = self.engine_state {
            if status.phase == "decode" {
                self.trigger_engine.update_tbt_from_throughput(status.actual_throughput);
            }
        }

        // 3. Observation 체크 (이전 액션의 relief 측정)
        self.check_observation();

        // 4. Lua decide(ctx) 호출
        let commands = self.call_decide();
        if commands.is_empty() {
            None
        } else {
            // Observation 시작 (단일 액션만)
            if commands.len() == 1 {
                let action_name = engine_command_to_action_name(&commands[0]);
                let pressure = self.signal_state.pressure_with_thermal(
                    self.adaptation_config.temp_safe_c,
                    self.adaptation_config.temp_critical_c,
                    self.trigger_engine.tbt_degradation_ratio(),
                );
                self.observation = Some(ObservationContext {
                    action: action_name,
                    before: pressure,
                    timestamp: Instant::now(),
                });
            } else {
                self.observation = None; // 복수 액션: observation 건너뜀
            }

            Some(EngineDirective {
                seq_id: next_seq_id(),
                commands,
            })
        }
    }

    fn save_model(&self) {
        if !self.adaptation_config.relief_table_path.is_empty() {
            let path = std::path::Path::new(&self.adaptation_config.relief_table_path);
            if let Err(e) = self.relief_table.save(path) {
                log::error!("Failed to save relief table: {}", e);
            } else {
                log::info!("Relief table saved to {}", self.adaptation_config.relief_table_path);
            }
        }
    }

    // update_engine_state, mode: 기존 유지
}
```

- [ ] **Step 4: `check_observation()` 헬퍼 구현**

```rust
impl LuaPolicy {
    fn check_observation(&mut self) {
        let obs = match self.observation.take() {
            Some(obs) => obs,
            None => return,
        };

        if obs.timestamp.elapsed().as_secs_f64() < OBSERVATION_DELAY_SECS {
            // 아직 settling time 미경과 → 다시 넣기
            self.observation = Some(obs);
            return;
        }

        // After 스냅샷
        let after = self.signal_state.pressure_with_thermal(
            self.adaptation_config.temp_safe_c,
            self.adaptation_config.temp_critical_c,
            self.trigger_engine.tbt_degradation_ratio(),
        );

        // Observed relief = before - after (양수 = 압력 감소 = 좋음)
        let observed = [
            obs.before.gpu - after.gpu,
            obs.before.cpu - after.cpu,
            obs.before.memory - after.memory,
            obs.before.thermal - after.thermal,
            obs.before.latency - after.latency,
            after.main_app - obs.before.main_app, // QoS는 증가가 좋음
        ];

        log::info!(
            "Relief observation: action={}, observed=[{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
            obs.action, observed[0], observed[1], observed[2], observed[3], observed[4], observed[5]
        );

        self.relief_table.observe(&obs.action, &observed);
    }
}
```

- [ ] **Step 5: `engine_command_to_action_name()` 헬퍼**

```rust
fn engine_command_to_action_name(cmd: &EngineCommand) -> String {
    match cmd {
        EngineCommand::Throttle { .. } => "throttle",
        EngineCommand::SetTargetTbt { .. } => "set_target_tbt",
        EngineCommand::LayerSkip { .. } => "layer_skip",
        EngineCommand::KvEvictH2o { .. } => "kv_evict_h2o",
        EngineCommand::KvEvictSliding { .. } => "kv_evict_sliding",
        EngineCommand::KvStreaming { .. } => "kv_streaming",
        EngineCommand::KvMergeD2o { .. } => "kv_merge_d2o",
        EngineCommand::KvQuantDynamic { .. } => "kv_quant_dynamic",
        EngineCommand::SwitchHw { .. } => "switch_hw",
        EngineCommand::RestoreDefaults => "restore_defaults",
        EngineCommand::Suspend => "suspend",
        EngineCommand::Resume => "resume",
        EngineCommand::SetPartitionRatio { .. } => "set_partition_ratio",
        EngineCommand::SetPrefillPolicy { .. } => "set_prefill_policy",
        EngineCommand::RequestQcf => "request_qcf",
    }.to_string()
}
```

- [ ] **Step 6: `build_ctx()` 확장 — `ctx.signal` + `ctx.coef` 추가**

기존 `build_ctx()` 끝에 추가:

```rust
    fn build_ctx(&self) -> LuaResult<Table> {
        let lua = &self.lua;
        let ctx = lua.create_table()?;

        // ctx.engine — 기존 코드 유지
        // ctx.active — 기존 코드 유지

        // ctx.signal (신규)
        let signal_tbl = lua.create_table()?;
        {
            let mem = lua.create_table()?;
            mem.set("available", self.signal_state.mem_available)?;
            mem.set("total", self.signal_state.mem_total)?;
            signal_tbl.set("memory", mem)?;

            let compute = lua.create_table()?;
            compute.set("cpu_pct", self.signal_state.cpu_pct)?;
            compute.set("gpu_pct", self.signal_state.gpu_pct)?;
            signal_tbl.set("compute", compute)?;

            let thermal = lua.create_table()?;
            thermal.set("temp_c", self.signal_state.temp_mc as f64 / 1000.0)?;
            thermal.set("throttling", self.signal_state.throttling)?;
            signal_tbl.set("thermal", thermal)?;
        }
        ctx.set("signal", signal_tbl)?;

        // ctx.coef (신규)
        let coef = lua.create_table()?;
        {
            // coef.pressure
            let pressure = self.signal_state.pressure_with_thermal(
                self.adaptation_config.temp_safe_c,
                self.adaptation_config.temp_critical_c,
                self.trigger_engine.tbt_degradation_ratio(),
            );
            let p_tbl = lua.create_table()?;
            p_tbl.set("gpu", pressure.gpu)?;
            p_tbl.set("cpu", pressure.cpu)?;
            p_tbl.set("memory", pressure.memory)?;
            p_tbl.set("thermal", pressure.thermal)?;
            p_tbl.set("latency", pressure.latency)?;
            p_tbl.set("main_app", pressure.main_app)?;
            coef.set("pressure", p_tbl)?;

            // coef.trigger
            let trigger = self.trigger_engine.state();
            let t_tbl = lua.create_table()?;
            t_tbl.set("tbt_degraded", trigger.tbt_degraded)?;
            t_tbl.set("mem_low", trigger.mem_low)?;
            t_tbl.set("temp_high", trigger.temp_high)?;
            coef.set("trigger", t_tbl)?;

            // coef.relief
            let r_tbl = lua.create_table()?;
            let action_names = [
                "switch_hw", "throttle", "set_target_tbt", "layer_skip",
                "kv_evict_h2o", "kv_evict_sliding", "kv_streaming",
                "kv_merge_d2o", "kv_quant_dynamic", "set_partition_ratio",
            ];
            for name in &action_names {
                let relief = self.relief_table.predict(name);
                let entry = lua.create_table()?;
                entry.set("gpu", relief[0])?;
                entry.set("cpu", relief[1])?;
                entry.set("mem", relief[2])?;
                entry.set("therm", relief[3])?;
                entry.set("lat", relief[4])?;
                entry.set("qos", relief[5])?;
                r_tbl.set(*name, entry)?;
            }
            coef.set("relief", r_tbl)?;
        }
        ctx.set("coef", coef)?;

        Ok(ctx)
    }
```

- [ ] **Step 7: `main.rs`에서 `create_lua_policy()` 수정 — config 전달**

```rust
#[cfg(feature = "lua")]
fn create_lua_policy(
    script_path: &std::path::Path,
    config: &Config,
) -> anyhow::Result<Box<dyn PolicyStrategy>> {
    let path_str = script_path
        .to_str()
        .ok_or_else(|| anyhow::anyhow!("Invalid UTF-8 in policy script path"))?;
    let policy = llm_manager::lua_policy::LuaPolicy::new(path_str, config.adaptation.clone())?;
    log::info!("LuaPolicy initialized from {}", path_str);
    Ok(Box::new(policy))
}
```

`create_policy()` 내 `create_lua_policy` 호출도 `config` 인자 추가:
```rust
return create_lua_policy(script_path, config);
```

- [ ] **Step 8: 빌드 확인**

```bash
cargo build -p llm_manager --features lua
```

- [ ] **Step 9: 커밋**

```bash
git add manager/
git commit -m "feat(manager): integrate SignalState/TriggerEngine/EwmaReliefTable into LuaPolicy

process_signal() now caches SystemSignal data, computes 6D pressure,
evaluates triggers with hysteresis, and exposes everything via
ctx.signal and ctx.coef to Lua decide(). Observation measures
before/after pressure delta and updates EWMA relief table."
```

---

### Task 6: Lua 예시 스크립트 업데이트

**Files:**
- Modify: `manager/scripts/policy_example.lua`

- [ ] **Step 1: 새 스크립트 작성**

```lua
-- policy_example.lua
-- Dynamic coefficient policy for llm_manager.
--
-- Uses ctx.coef (pressure, trigger, relief) provided by Rust.
-- Relief values are learned via EWMA at runtime.
--
-- Usage:
--   llm_manager --policy-script manager/scripts/policy_example.lua \
--               --transport unix:/tmp/llm.sock

function decide(ctx)
    local c = ctx.coef
    local t = c.trigger

    -- No contention evidence → no intervention
    if not t.tbt_degraded and not t.mem_low and not t.temp_high then
        -- Release active actions if pressure is low
        if #ctx.active > 0 then
            local p = c.pressure
            if p.gpu < 0.3 and p.memory < 0.3 and p.thermal < 0.3 then
                return {{type = "restore_defaults"}}
            end
        end
        return {}
    end

    -- Find the domain with highest pressure
    local p = c.pressure
    local domains = {gpu = p.gpu, cpu = p.cpu, memory = p.memory, thermal = p.thermal}
    local max_domain, max_val = nil, 0
    for k, v in pairs(domains) do
        if v > max_val then
            max_domain = k
            max_val = v
        end
    end

    if max_domain == nil then
        return {}
    end

    -- Map relief table keys to domain keys
    local domain_key = max_domain
    if domain_key == "memory" then domain_key = "mem" end
    if domain_key == "thermal" then domain_key = "therm" end

    -- Select the action with highest relief for the bottleneck domain,
    -- while respecting latency budget (lat >= -0.15)
    local best_action = nil
    local best_relief = -999

    for action, r in pairs(c.relief) do
        local relief_val = r[domain_key] or 0
        if relief_val > best_relief and (r.lat or 0) >= -0.15 then
            best_action = action
            best_relief = relief_val
        end
    end

    if best_action and best_relief > 0 then
        -- Build action with sensible defaults
        local cmd = {type = best_action}
        if best_action == "kv_evict_h2o" or best_action == "kv_evict_sliding"
           or best_action == "kv_merge_d2o" then
            cmd.keep_ratio = 0.5
        elseif best_action == "throttle" then
            cmd.delay_ms = 50
        elseif best_action == "set_target_tbt" then
            cmd.target_ms = 150
        elseif best_action == "layer_skip" then
            cmd.skip_ratio = 0.25
        elseif best_action == "switch_hw" then
            cmd.device = "cpu"
        elseif best_action == "kv_quant_dynamic" then
            cmd.target_bits = 4
        elseif best_action == "set_partition_ratio" then
            cmd.ratio = 0.5
        end
        return {cmd}
    end

    return {}
end
```

- [ ] **Step 2: 커밋**

```bash
git add manager/scripts/policy_example.lua
git commit -m "feat(manager): update policy_example.lua to use ctx.coef dynamic coefficients

Replaces hardcoded thresholds with trigger-based contention detection
and relief-guided action selection from EWMA-learned coefficients."
```

---

### Task 7: Config 파일 업데이트 + 통합 테스트

**Files:**
- Modify: `manager/policy_config.toml` (또는 해당 config 파일)
- Modify: `manager/src/lua_policy.rs` (통합 테스트 추가)

- [ ] **Step 1: policy_config.toml에 adaptation 섹션 추가**

파일이 없으면 `manager/policy_config.toml`에 생성:

```toml
[adaptation]
ewma_alpha = 0.875
relief_table_path = ""
temp_safe_c = 35.0
temp_critical_c = 50.0

[adaptation.trigger]
tbt_enter = 0.30
tbt_exit = 0.10
tbt_warmup_tokens = 20
mem_enter = 0.80
mem_exit = 0.60
temp_enter = 0.70
temp_exit = 0.50

[adaptation.default_relief]
switch_hw = [0.5, -0.3, 0.0, 0.3, -0.1, 0.0]
kv_evict_h2o = [0.1, 0.0, 0.4, 0.1, 0.0, 0.0]
kv_evict_sliding = [0.1, 0.0, 0.3, 0.1, 0.0, 0.0]
throttle = [0.0, 0.3, 0.0, 0.2, -0.2, 0.0]
set_target_tbt = [0.0, 0.2, 0.0, 0.1, -0.1, 0.0]
layer_skip = [0.2, 0.1, 0.0, 0.1, -0.1, 0.0]
kv_quant_dynamic = [0.0, 0.0, 0.5, 0.0, 0.0, 0.0]
kv_merge_d2o = [0.1, 0.0, 0.3, 0.1, 0.0, 0.0]
set_partition_ratio = [0.3, -0.2, 0.0, 0.1, 0.0, 0.0]
```

- [ ] **Step 2: 통합 테스트 — LuaPolicy end-to-end**

`lua_policy.rs`의 tests 모듈에 추가:

```rust
    #[test]
    fn lua_policy_e2e_signal_to_ctx() {
        // 최소 Lua 스크립트: ctx.coef를 검증하고 반환
        let script = r#"
            function decide(ctx)
                -- coef가 존재하는지 검증
                assert(ctx.coef, "ctx.coef missing")
                assert(ctx.coef.pressure, "ctx.coef.pressure missing")
                assert(ctx.coef.trigger, "ctx.coef.trigger missing")
                assert(ctx.coef.relief, "ctx.coef.relief missing")

                -- pressure 값 검증
                assert(ctx.coef.pressure.gpu >= 0)
                assert(ctx.coef.pressure.memory >= 0)

                -- trigger 값 검증
                assert(type(ctx.coef.trigger.tbt_degraded) == "boolean")
                assert(type(ctx.coef.trigger.mem_low) == "boolean")

                -- relief 테이블 검증
                assert(ctx.coef.relief.switch_hw, "switch_hw relief missing")
                assert(ctx.coef.relief.switch_hw.gpu, "switch_hw.gpu missing")

                return {}
            end
        "#;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("test_policy.lua");
        std::fs::write(&script_path, script).unwrap();

        let config = AdaptationConfig {
            default_relief: {
                let mut m = HashMap::new();
                m.insert("switch_hw".to_string(), vec![0.5, -0.3, 0.0, 0.3, -0.1, 0.0]);
                m
            },
            ..AdaptationConfig::default()
        };

        let mut policy = LuaPolicy::new(script_path.to_str().unwrap(), config).unwrap();

        // Feed a SystemSignal
        let signal = SystemSignal::ComputeGuidance {
            level: llm_shared::Level::Warning,
            recommended_backend: llm_shared::RecommendedBackend::Cpu,
            reason: llm_shared::ComputeReason::CpuExhausted,
            cpu_usage_pct: 45.0,
            gpu_usage_pct: 82.0,
        };

        let result = policy.process_signal(&signal);
        // Script returns {}, so no directive
        assert!(result.is_none());
    }

    #[test]
    fn lua_policy_e2e_trigger_fires() {
        let script = r#"
            function decide(ctx)
                if ctx.coef.trigger.mem_low then
                    return {{type = "kv_evict_h2o", keep_ratio = 0.5}}
                end
                return {}
            end
        "#;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("test_trigger.lua");
        std::fs::write(&script_path, script).unwrap();

        let config = AdaptationConfig::default();
        let mut policy = LuaPolicy::new(script_path.to_str().unwrap(), config).unwrap();

        // Memory pressure > 80% → trigger fires
        let signal = SystemSignal::MemoryPressure {
            level: llm_shared::Level::Warning,
            available_bytes: 1_000_000,
            total_bytes: 8_000_000,
            reclaim_target_bytes: 0,
        };

        let result = policy.process_signal(&signal);
        assert!(result.is_some());
        let directive = result.unwrap();
        assert_eq!(directive.commands.len(), 1);
        assert!(matches!(directive.commands[0], EngineCommand::KvEvictH2o { keep_ratio } if (keep_ratio - 0.5).abs() < f32::EPSILON));
    }
```

- [ ] **Step 3: 테스트 실행**

```bash
cargo test -p llm_manager --features lua lua_policy -- --nocapture
```

Expected: 모든 테스트 PASS.

- [ ] **Step 4: feature flag 빌드 테스트**

```bash
# default (lua 포함, hierarchical 없음)
cargo build -p llm_manager
cargo test -p llm_manager

# hierarchical 포함
cargo build -p llm_manager --features hierarchical
cargo test -p llm_manager --features hierarchical
```

- [ ] **Step 5: 커밋**

```bash
git add manager/
git commit -m "feat(manager): add adaptation config, integration tests, policy_config.toml

End-to-end test confirms SystemSignal → ctx.coef pipeline works.
Trigger test confirms mem_low fires and Lua returns KvEvictH2o."
```

---

### Task 8: fmt + clippy + 최종 검증

**Files:** 전체 manager 크레이트

- [ ] **Step 1: cargo fmt**

```bash
cargo fmt -p llm_manager
```

- [ ] **Step 2: cargo clippy**

```bash
cargo clippy -p llm_manager --features lua -- -D warnings
cargo clippy -p llm_manager --features lua,hierarchical -- -D warnings
```

경고가 있으면 수정.

- [ ] **Step 3: 전체 테스트**

```bash
cargo test -p llm_manager --features lua
cargo test -p llm_manager --features lua,hierarchical
```

- [ ] **Step 4: 커밋 (수정이 있으면)**

```bash
git add manager/
git commit -m "style(manager): fmt + clippy fixes"
```
