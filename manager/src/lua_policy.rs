//! Lua scripting integration for policy decision.
//!
//! `LuaPolicy` implements `PolicyStrategy` by delegating the `decide(ctx)` call
//! to a user-supplied Lua script.  The Lua VM is created once at startup and
//! kept alive for the entire session, allowing scripts to maintain state across
//! invocations (e.g. EMA accumulators).
//!
//! # Sandbox
//!
//! Only `table`, `string`, `math` standard libraries are loaded.  I/O is
//! restricted to the `sys.*` helpers registered by Rust (read-only sysfs/procfs
//! access).  Memory is capped at 4 MB.

use llm_shared::{EngineCommand, EngineDirective, EngineMessage, EngineStatus, SystemSignal};
use mlua::{Lua, Result as LuaResult, StdLib, Table, Value};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::clock::{Clock, LogicalInstant, SystemClock};
use crate::config::{AdaptationConfig, TriggerConfig};
use crate::pipeline::{PolicyStrategy, next_seq_id};
use crate::types::OperatingMode;

/// 6D relief 벡터 차원 수 (gpu, cpu, memory, thermal, latency, main_app_qos).
#[doc(hidden)]
pub const RELIEF_DIMS: usize = 6;

#[derive(Debug, Clone, Default)]
struct Pressure6D {
    gpu: f32,
    cpu: f32,
    memory: f32,
    thermal: f32,
    #[allow(dead_code)]
    latency: f32,
    #[allow(dead_code)]
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
        self.baseline
            .map(|b| if b > 0.0 { (self.ewma - b) / b } else { 0.0 })
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
            return;
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

    #[allow(dead_code)]
    fn tbt_degradation_ratio(&self) -> Option<f64> {
        self.tbt.degradation_ratio()
    }
}

/// 단일 액션의 학습된 EWMA relief 벡터 + 관측 횟수.
///
/// JSON으로 직렬화되어 `relief_table_path`에 저장된다 (MGR-DAT-071).
/// `relief[i]` 부호 규약 (MGR-DAT-073, INV-089):
///   - dims 0~4: before – after (양수 = 압박 감소)
///   - dim 5 (main_app_qos): after – before (양수 = QoS 향상)
#[doc(hidden)]
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReliefEntry {
    pub relief: [f32; RELIEF_DIMS],
    pub observation_count: u32,
}

/// EWMA-기반 relief 학습 테이블 (MGR-ALG-080 ~ MGR-ALG-083).
///
/// Integration test에서 직접 생성·조작하기 위해 공개된다.
/// Production 코드는 `LuaPolicy` 내부에서만 사용한다.
#[doc(hidden)]
pub struct EwmaReliefTable {
    pub entries: HashMap<String, ReliefEntry>,
    pub alpha: f32,
    pub defaults: HashMap<String, Vec<f32>>,
}

impl EwmaReliefTable {
    pub fn new(alpha: f32, defaults: HashMap<String, Vec<f32>>) -> Self {
        Self {
            entries: HashMap::new(),
            alpha,
            defaults,
        }
    }

    pub fn predict(&self, action: &str) -> [f32; RELIEF_DIMS] {
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

    pub fn observe(&mut self, action: &str, observed: &[f32; RELIEF_DIMS]) {
        let entry = self
            .entries
            .entry(action.to_string())
            .or_insert_with(|| ReliefEntry {
                relief: [0.0; RELIEF_DIMS],
                observation_count: 0,
            });

        if entry.observation_count == 0 {
            entry.relief = *observed;
        } else {
            let a = self.alpha;
            for (i, &obs_val) in observed.iter().enumerate() {
                entry.relief[i] = a * entry.relief[i] + (1.0 - a) * obs_val;
            }
        }
        entry.observation_count += 1;
    }

    pub fn observation_count(&self, action: &str) -> u32 {
        self.entries.get(action).map_or(0, |e| e.observation_count)
    }

    /// 현재 테이블 상태를 스냅샷으로 반환한다 (테스트/시뮬레이터 관측용).
    pub fn snapshot(&self) -> HashMap<String, [f32; RELIEF_DIMS]> {
        self.entries
            .iter()
            .map(|(k, v)| (k.clone(), v.relief))
            .collect()
    }

    pub fn save(&self, path: &Path) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(&self.entries).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    pub fn load(
        path: &Path,
        alpha: f32,
        defaults: HashMap<String, Vec<f32>>,
    ) -> std::io::Result<Self> {
        let json = std::fs::read_to_string(path)?;
        let entries: HashMap<String, ReliefEntry> =
            serde_json::from_str(&json).map_err(std::io::Error::other)?;
        Ok(Self {
            entries,
            alpha,
            defaults,
        })
    }
}

pub const OBSERVATION_DELAY_SECS: f64 = 3.0;

/// 동시에 대기 가능한 in-flight observation 최대 개수.
///
/// 정책이 3s 관측 지연을 채우기 전에 새 directive를 방출하면 이전 observation이
/// 소실되던 single-slot 구조의 한계를 해소하기 위해 FIFO 큐로 전환했다
/// (2026-04-15). 기본 용량 32는 ~10 Hz directive rate × 3 s = 30 동시 in-flight를
/// 수용하도록 설정한다. 용량을 넘으면 가장 오래된 observation을 드롭하고
/// `observation_overrun_count`를 증가시킨다.
#[doc(hidden)]
pub const MAX_PENDING_OBSERVATIONS: usize = 32;

/// Relief 테이블 업데이트 이벤트 (관측성 훅).
///
/// 시뮬레이터가 [`LuaPolicy::drain_relief_updates`]로 드레인해 trajectory에
/// 기록한다. production에서는 로그용으로만 소비된다.
#[doc(hidden)]
#[derive(Debug, Clone, serde::Serialize)]
pub struct ReliefUpdateEvent {
    /// observe() 대상 action 이름.
    pub action: String,
    /// observe() 호출 전 relief 벡터.
    pub before: [f32; RELIEF_DIMS],
    /// observe() 호출 후 relief 벡터 (EWMA 적용 결과).
    pub after: [f32; RELIEF_DIMS],
    /// 이번에 관측된 델타 (before_pressure - after_pressure).
    pub observed: [f32; RELIEF_DIMS],
    /// 업데이트 후 total observation count.
    pub observation_count: u32,
    /// ObservationContext가 기록된 이후 경과 시간 (초).
    pub age_s: f64,
}

struct ObservationContext {
    action: String,
    before: Pressure6D,
    timestamp: LogicalInstant,
}

/// Lua-based policy strategy.
///
/// Wraps an `mlua::Lua` VM with a loaded `decide(ctx)` function.
/// Engine heartbeat state is cached and forwarded as `ctx.engine`.
pub struct LuaPolicy {
    lua: Lua,
    /// Latest engine heartbeat (None until first heartbeat received).
    engine_state: Option<EngineStatus>,
    // ── 신규 필드 ──
    signal_state: SignalState,
    trigger_engine: TriggerEngine,
    relief_table: EwmaReliefTable,
    observations: VecDeque<ObservationContext>,
    adaptation_config: AdaptationConfig,
    clock: Arc<dyn Clock>,
    /// 최근 발생한 relief 업데이트 이벤트 (시뮬레이터가 drain한다).
    pending_relief_updates: Vec<ReliefUpdateEvent>,
    /// `MAX_PENDING_OBSERVATIONS` 용량 초과로 드롭된 observation 수.
    /// 큐 용량이 충분하면 0이어야 한다 — 값이 계속 증가하면 directive 방출률이
    /// 용량을 초과했다는 뜻이므로 `MAX_PENDING_OBSERVATIONS` 조정을 검토.
    observation_overrun_count: u64,
}

impl std::fmt::Debug for LuaPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LuaPolicy")
            .field("engine_state", &self.engine_state.is_some())
            .field("signal_state", &self.signal_state)
            .field("trigger_engine", &self.trigger_engine)
            .finish()
    }
}

impl LuaPolicy {
    /// Create a new LuaPolicy by loading and evaluating `script_path`.
    ///
    /// The script must define a global `decide(ctx)` function.
    /// `sys.*` helper functions are registered before the script is evaluated.
    ///
    /// `clock`은 관측 지연 계산에 사용된다. Production은 `SystemClock`, 테스트/시뮬레이터는
    /// `VirtualClockHandle`을 주입한다.
    pub fn new(
        script_path: &str,
        config: AdaptationConfig,
        clock: Arc<dyn Clock>,
    ) -> anyhow::Result<Self> {
        // Sandbox: only table + string + math (no io, os, debug, etc.)
        // Safety: we intentionally restrict stdlib to TABLE | STRING | MATH.
        // unsafe_new_with is required because mlua considers any stdlib subset
        // potentially unsafe (e.g. missing debug library).
        let lua = unsafe {
            Lua::unsafe_new_with(
                StdLib::TABLE | StdLib::STRING | StdLib::MATH,
                mlua::LuaOptions::default(),
            )
        };

        // Memory limit: 4 MB
        let _ = lua.set_memory_limit(4 * 1024 * 1024);

        // Register sys.* helpers
        register_sys_helpers(&lua)
            .map_err(|e| anyhow::anyhow!("Failed to register sys helpers: {}", e))?;

        // Load and execute the user script (defines `decide` globally)
        let script = std::fs::read_to_string(script_path)
            .map_err(|e| anyhow::anyhow!("Failed to read Lua script {}: {}", script_path, e))?;
        lua.load(&script)
            .set_name(script_path)
            .exec()
            .map_err(|e| anyhow::anyhow!("Failed to evaluate Lua script {}: {}", script_path, e))?;

        // Verify `decide` function exists
        let globals = lua.globals();
        let decide: Value = globals
            .get("decide")
            .map_err(|e| anyhow::anyhow!("Failed to get 'decide' global: {}", e))?;
        if !decide.is_function() {
            anyhow::bail!(
                "Lua script {} must define a global `decide(ctx)` function",
                script_path
            );
        }

        log::info!("LuaPolicy loaded from {}", script_path);

        let relief_table = if !config.relief_table_path.is_empty() {
            let path = std::path::Path::new(&config.relief_table_path);
            EwmaReliefTable::load(path, config.ewma_alpha, config.default_relief.clone())
                .unwrap_or_else(|_| {
                    log::info!(
                        "No existing relief table at {}, starting fresh",
                        config.relief_table_path
                    );
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
            observations: VecDeque::new(),
            adaptation_config: config,
            clock,
            pending_relief_updates: Vec::new(),
            observation_overrun_count: 0,
        })
    }

    /// Production 기본 생성자 — SystemClock을 자동 주입한다.
    pub fn with_system_clock(script_path: &str, config: AdaptationConfig) -> anyhow::Result<Self> {
        Self::new(script_path, config, Arc::new(SystemClock::new()))
    }

    /// Build the `ctx` Lua table from current engine state.
    fn build_ctx(&self) -> LuaResult<Table> {
        let lua = &self.lua;
        let ctx = lua.create_table()?;

        // ctx.engine
        let engine_tbl = lua.create_table()?;
        if let Some(ref status) = self.engine_state {
            engine_tbl.set("device", status.active_device.as_str())?;
            engine_tbl.set("throughput", status.actual_throughput)?;
            engine_tbl.set("kv_util", status.kv_cache_utilization)?;
            engine_tbl.set("cache_tokens", status.kv_cache_tokens)?;
            engine_tbl.set("cache_bytes", status.kv_cache_bytes)?;
            engine_tbl.set("tokens_generated", status.tokens_generated)?;
            let state_str = match status.state {
                llm_shared::EngineState::Idle => "idle",
                llm_shared::EngineState::Running => "running",
                llm_shared::EngineState::Suspended => "suspended",
            };
            engine_tbl.set("state", state_str)?;
            engine_tbl.set("kv_dtype", status.kv_dtype.as_str())?;
            engine_tbl.set("skip_ratio", status.skip_ratio)?;
            engine_tbl.set("phase", status.phase.as_str())?;
            engine_tbl.set("prefill_pos", status.prefill_pos)?;
            engine_tbl.set("prefill_total", status.prefill_total)?;
            engine_tbl.set("partition_ratio", status.partition_ratio)?;
            // MGR-DAT-075/076, MSG-069: Engine process 자가 사용률 (Phase 1: CPU만 실측, GPU는 0.0 placeholder).
            engine_tbl.set("cpu_pct", status.self_cpu_pct)?;
            engine_tbl.set("gpu_pct", status.self_gpu_pct)?;
        } else {
            // No heartbeat yet -- provide defaults
            engine_tbl.set("device", "unknown")?;
            engine_tbl.set("throughput", 0.0)?;
            engine_tbl.set("kv_util", 0.0)?;
            engine_tbl.set("cache_tokens", 0)?;
            engine_tbl.set("cache_bytes", 0)?;
            engine_tbl.set("tokens_generated", 0)?;
            engine_tbl.set("state", "idle")?;
            engine_tbl.set("kv_dtype", "")?;
            engine_tbl.set("skip_ratio", 0.0)?;
            engine_tbl.set("phase", "")?;
            engine_tbl.set("prefill_pos", 0)?;
            engine_tbl.set("prefill_total", 0)?;
            engine_tbl.set("partition_ratio", 0.0)?;
            // MGR-DAT-075/076, MSG-069: heartbeat 없을 때 0.0 default (INV-092).
            engine_tbl.set("cpu_pct", 0.0)?;
            engine_tbl.set("gpu_pct", 0.0)?;
        }
        ctx.set("engine", engine_tbl)?;

        // ctx.active -- list of currently active action names
        let active_tbl = lua.create_table()?;
        if let Some(ref status) = self.engine_state {
            for (i, action) in status.active_actions.iter().enumerate() {
                active_tbl.set(i + 1, action.as_str())?;
            }
        }
        ctx.set("active", active_tbl)?;

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
                "switch_hw",
                "throttle",
                "set_target_tbt",
                "layer_skip",
                "kv_evict_h2o",
                "kv_evict_sliding",
                "kv_streaming",
                "kv_merge_d2o",
                "kv_quant_dynamic",
                "set_partition_ratio",
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

    /// Call `decide(ctx)` and parse the returned table into `Vec<EngineCommand>`.
    fn call_decide(&self) -> Vec<EngineCommand> {
        let result: LuaResult<Vec<EngineCommand>> = (|| {
            let globals = self.lua.globals();
            let decide: mlua::Function = globals.get("decide")?;
            let ctx = self.build_ctx()?;
            let result: Value = decide.call(ctx)?;
            parse_actions(result)
        })();

        match result {
            Ok(commands) => commands,
            Err(e) => {
                log::error!("Lua decide() error: {}", e);
                Vec::new()
            }
        }
    }
}

impl LuaPolicy {
    fn check_observation(&mut self) {
        // FIFO 큐: 오래된 observation부터 순서대로 확인한다. 큐가 삽입 시각 기준
        // 단조 증가이므로 front가 아직 성숙하지 않았다면 뒤쪽도 성숙하지 않았다.
        let now = self.clock.now();
        while let Some(front) = self.observations.front() {
            let elapsed = now.saturating_duration_since(front.timestamp);
            if elapsed.as_secs_f64() < OBSERVATION_DELAY_SECS {
                break;
            }
            // 성숙한 observation은 pop해서 처리.
            let obs = self.observations.pop_front().expect("front checked above");
            self.mature_observation(obs, elapsed.as_secs_f64());
        }
    }

    /// 성숙한 observation 하나를 처리하고 relief 테이블을 업데이트한다.
    fn mature_observation(&mut self, obs: ObservationContext, age_s: f64) {
        let after = self.signal_state.pressure_with_thermal(
            self.adaptation_config.temp_safe_c,
            self.adaptation_config.temp_critical_c,
            self.trigger_engine.tbt_degradation_ratio(),
        );

        let observed = [
            obs.before.gpu - after.gpu,
            obs.before.cpu - after.cpu,
            obs.before.memory - after.memory,
            obs.before.thermal - after.thermal,
            obs.before.latency - after.latency,
            after.main_app - obs.before.main_app,
        ];

        let before_relief = self.relief_table.predict(&obs.action);

        log::info!(
            "Relief observation: action={}, observed=[{:.3}, {:.3}, {:.3}, {:.3}, {:.3}, {:.3}]",
            obs.action,
            observed[0],
            observed[1],
            observed[2],
            observed[3],
            observed[4],
            observed[5]
        );

        self.relief_table.observe(&obs.action, &observed);

        // 업데이트 후 스냅샷 기록 (관측성 훅)
        let after_entry = self.relief_table.entries.get(&obs.action);
        let after_relief = after_entry.map(|e| e.relief).unwrap_or(before_relief);
        let count = after_entry.map(|e| e.observation_count).unwrap_or(0);
        self.pending_relief_updates.push(ReliefUpdateEvent {
            action: obs.action,
            before: before_relief,
            after: after_relief,
            observed,
            observation_count: count,
            age_s,
        });
    }

    /// 관측성 훅: 지난 drain 이후 축적된 relief 업데이트 이벤트를 반환하고 버퍼를 비운다.
    ///
    /// 시뮬레이터가 매 tick 호출해 Trajectory에 기록한다.
    #[doc(hidden)]
    pub fn drain_relief_updates(&mut self) -> Vec<ReliefUpdateEvent> {
        std::mem::take(&mut self.pending_relief_updates)
    }

    /// 관측성 훅: OBSERVATION_DELAY_SECS를 채우기 전에 덮어써진 observation 수.
    ///
    /// Single-slot observation 구조로 인한 학습 누락 감지용. 이 값이 증가하는
    /// 만큼 정책이 빠르게 직전 action 결과를 측정하지 못했다는 뜻.
    #[doc(hidden)]
    pub fn observation_overrun_count(&self) -> u64 {
        self.observation_overrun_count
    }
}

impl PolicyStrategy for LuaPolicy {
    fn process_signal(&mut self, signal: &SystemSignal) -> Option<EngineDirective> {
        // 1. SignalState 업데이트
        match signal {
            SystemSignal::MemoryPressure {
                available_bytes,
                total_bytes,
                ..
            } => {
                self.signal_state
                    .update_memory(*available_bytes, *total_bytes);
                let pressure = if *total_bytes > 0 {
                    1.0 - (*available_bytes as f64 / *total_bytes as f64)
                } else {
                    0.0
                };
                self.trigger_engine.update_mem(pressure);
            }
            SystemSignal::ComputeGuidance {
                cpu_usage_pct,
                gpu_usage_pct,
                ..
            } => {
                self.signal_state
                    .update_compute(*cpu_usage_pct, *gpu_usage_pct);
            }
            SystemSignal::ThermalAlert {
                temperature_mc,
                throttling_active,
                ..
            } => {
                self.signal_state
                    .update_thermal(*temperature_mc, *throttling_active);
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
                // Energy는 trigger에 포함되지 않음
            }
        }

        // 2. TBT 업데이트
        if let Some(ref status) = self.engine_state
            && status.phase == "decode"
        {
            self.trigger_engine
                .update_tbt_from_throughput(status.actual_throughput);
        }

        // 3. Observation 체크
        self.check_observation();

        // 4. Lua decide(ctx) 호출
        let commands = self.call_decide();
        if commands.is_empty() {
            None
        } else {
            // Observation 큐잉 (단일 액션만). Multi-command directive는 어느
            // command가 이후 pressure 변화를 일으켰는지 귀속 불가하므로 큐를
            // 비운다. 큐 용량 초과 시 가장 오래된 observation을 드롭하고
            // overrun을 기록한다.
            if commands.len() == 1 {
                let action_name = engine_command_to_action_name(&commands[0]);
                let pressure = self.signal_state.pressure_with_thermal(
                    self.adaptation_config.temp_safe_c,
                    self.adaptation_config.temp_critical_c,
                    self.trigger_engine.tbt_degradation_ratio(),
                );
                if self.observations.len() >= MAX_PENDING_OBSERVATIONS {
                    let _ = self.observations.pop_front();
                    self.observation_overrun_count += 1;
                }
                self.observations.push_back(ObservationContext {
                    action: action_name,
                    before: pressure,
                    timestamp: self.clock.now(),
                });
            } else {
                // Multi-command: 귀속 불가. 쌓여있던 관측들을 overrun으로 처리.
                if !self.observations.is_empty() {
                    self.observation_overrun_count += self.observations.len() as u64;
                    self.observations.clear();
                }
            }

            Some(EngineDirective {
                seq_id: next_seq_id(),
                commands,
            })
        }
    }

    fn cancel_last_observation(&mut self) {
        self.observations.pop_back();
    }

    fn update_engine_state(&mut self, msg: &EngineMessage) {
        if let EngineMessage::Heartbeat(status) = msg {
            self.engine_state = Some(status.clone());
        }
    }

    fn mode(&self) -> OperatingMode {
        // Lua scripts manage their own mode concept; we report Normal by default.
        OperatingMode::Normal
    }

    fn save_model(&self) {
        if !self.adaptation_config.relief_table_path.is_empty() {
            let path = std::path::Path::new(&self.adaptation_config.relief_table_path);
            if let Err(e) = self.relief_table.save(path) {
                log::error!("Failed to save relief table: {}", e);
            } else {
                log::info!(
                    "Relief table saved to {}",
                    self.adaptation_config.relief_table_path
                );
            }
        }
    }

    fn relief_snapshot(&self) -> Option<std::collections::HashMap<String, [f32; 6]>> {
        Some(self.relief_table.snapshot())
    }

    fn drain_relief_updates(&mut self) -> Vec<ReliefUpdateEvent> {
        LuaPolicy::drain_relief_updates(self)
    }

    fn observation_overrun_count(&self) -> u64 {
        LuaPolicy::observation_overrun_count(self)
    }
}

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
        EngineCommand::PrepareComputeUnit { .. } => "prepare_compute_unit",
    }
    .to_string()
}

/// Parse the Lua return value from `decide()` into a list of `EngineCommand`.
///
/// Expected format: a Lua table used as an array, where each element is a table
/// with a `type` field and optional action-specific fields.
fn parse_actions(value: Value) -> LuaResult<Vec<EngineCommand>> {
    let table = match value {
        Value::Table(t) => t,
        Value::Nil => return Ok(Vec::new()),
        other => {
            log::error!(
                "decide() must return a table or nil, got: {}",
                other.type_name()
            );
            return Ok(Vec::new());
        }
    };

    let len = table.len()?;
    if len == 0 {
        return Ok(Vec::new());
    }

    let mut commands = Vec::with_capacity(len as usize);

    for i in 1..=len {
        let entry: Table = match table.get(i) {
            Ok(t) => t,
            Err(e) => {
                log::error!("decide() return[{}]: expected table, error: {}", i, e);
                continue;
            }
        };

        let action_type: String = match entry.get("type") {
            Ok(t) => t,
            Err(e) => {
                log::error!("decide() return[{}]: missing 'type' field: {}", i, e);
                continue;
            }
        };

        match parse_single_action(&action_type, &entry) {
            Ok(cmd) => commands.push(cmd),
            Err(e) => {
                log::error!(
                    "decide() return[{}]: failed to parse action '{}': {}",
                    i,
                    action_type,
                    e
                );
            }
        }
    }

    Ok(commands)
}

/// Parse a single action table into an `EngineCommand`.
fn parse_single_action(action_type: &str, entry: &Table) -> LuaResult<EngineCommand> {
    let cmd = match action_type {
        "throttle" => {
            let delay_ms: u64 = entry.get("delay_ms")?;
            EngineCommand::Throttle { delay_ms }
        }
        "set_target_tbt" => {
            let target_ms: u64 = entry.get("target_ms")?;
            EngineCommand::SetTargetTbt { target_ms }
        }
        "layer_skip" => {
            let skip_ratio: f32 = entry.get("skip_ratio")?;
            EngineCommand::LayerSkip { skip_ratio }
        }
        "kv_evict_h2o" => {
            let keep_ratio: f32 = entry.get("keep_ratio")?;
            EngineCommand::KvEvictH2o { keep_ratio }
        }
        "kv_evict_sliding" => {
            let keep_ratio: f32 = entry.get("keep_ratio")?;
            EngineCommand::KvEvictSliding { keep_ratio }
        }
        "kv_streaming" => {
            let sink_size: usize = entry.get("sink_size")?;
            let window_size: usize = entry.get("window_size")?;
            EngineCommand::KvStreaming {
                sink_size,
                window_size,
            }
        }
        "kv_merge_d2o" => {
            let keep_ratio: f32 = entry.get("keep_ratio")?;
            EngineCommand::KvMergeD2o { keep_ratio }
        }
        "kv_quant_dynamic" => {
            let target_bits: u8 = entry.get("target_bits")?;
            EngineCommand::KvQuantDynamic { target_bits }
        }
        "switch_hw" => {
            let device: String = entry.get("device")?;
            EngineCommand::SwitchHw { device }
        }
        "restore_defaults" => EngineCommand::RestoreDefaults,
        "suspend" => EngineCommand::Suspend,
        "resume" => EngineCommand::Resume,
        "set_partition_ratio" => {
            let ratio: f32 = entry.get("ratio")?;
            EngineCommand::SetPartitionRatio { ratio }
        }
        "set_prefill_policy" => {
            let chunk_size: Option<usize> = entry.get("chunk_size").ok();
            let yield_ms: Option<u32> = entry.get("yield_ms").ok();
            let cpu_chunk_size: Option<usize> = entry.get("cpu_chunk_size").ok();
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            }
        }
        unknown => {
            return Err(mlua::Error::external(format!(
                "unknown action type: '{}'",
                unknown
            )));
        }
    };
    Ok(cmd)
}

// ---- sys.* helper registration ----------------------------------------------

/// Register `sys.*` helper functions in the Lua global scope.
fn register_sys_helpers(lua: &Lua) -> LuaResult<()> {
    let sys = lua.create_table()?;

    // sys.read(path) -> string
    sys.set(
        "read",
        lua.create_function(|_, path: String| -> LuaResult<String> {
            Ok(std::fs::read_to_string(&path)
                .map(|s| s.trim().to_string())
                .unwrap_or_default())
        })?,
    )?;

    // sys.meminfo() -> {total, available, free} (KB)
    sys.set(
        "meminfo",
        lua.create_function(|lua_inner, ()| -> LuaResult<Table> {
            let tbl = lua_inner.create_table()?;
            let (total, available, free) = read_meminfo();
            tbl.set("total", total)?;
            tbl.set("available", available)?;
            tbl.set("free", free)?;
            Ok(tbl)
        })?,
    )?;

    // sys.thermal(zone) -> float (degrees C)
    sys.set(
        "thermal",
        lua.create_function(|_, zone: u32| -> LuaResult<f64> {
            let path = format!("/sys/class/thermal/thermal_zone{}/temp", zone);
            let temp = std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| s.trim().parse::<f64>().ok())
                .map(|mc| mc / 1000.0)
                .unwrap_or(-1.0);
            Ok(temp)
        })?,
    )?;

    // sys.gpu_busy() -> int (0-100)
    sys.set(
        "gpu_busy",
        lua.create_function(|_, ()| -> LuaResult<i64> {
            let path = "/sys/class/kgsl/kgsl-3d0/gpu_busy_percentage";
            let pct = std::fs::read_to_string(path)
                .ok()
                .and_then(|s| {
                    // Format may be "N %" or just "N"
                    s.split_whitespace()
                        .next()
                        .and_then(|v| v.parse::<i64>().ok())
                })
                .unwrap_or(-1);
            Ok(pct)
        })?,
    )?;

    // sys.gpu_freq() -> int (Hz)
    sys.set(
        "gpu_freq",
        lua.create_function(|_, ()| -> LuaResult<i64> {
            let path = "/sys/class/kgsl/kgsl-3d0/gpuclk";
            let freq = std::fs::read_to_string(path)
                .ok()
                .and_then(|s| s.trim().parse::<i64>().ok())
                .unwrap_or(-1);
            Ok(freq)
        })?,
    )?;

    // sys.cpu_freq(cpu_index) -> int (KHz)
    sys.set(
        "cpu_freq",
        lua.create_function(|_, cpu: u32| -> LuaResult<i64> {
            let path = format!(
                "/sys/devices/system/cpu/cpu{}/cpufreq/scaling_cur_freq",
                cpu
            );
            let freq = std::fs::read_to_string(&path)
                .ok()
                .and_then(|s| s.trim().parse::<i64>().ok())
                .unwrap_or(-1);
            Ok(freq)
        })?,
    )?;

    // sys.foreground_fps(pkg) -> float|nil
    //
    // Measures the foreground frame rate of `pkg` by reading SurfaceFlinger
    // frame counters via `dumpsys SurfaceFlinger`. First call returns nil
    // (only stores baseline). Subsequent calls return FPS since last call.
    // Returns nil if the package surface is not found, or if called too soon
    // (< 100 ms since last call).
    //
    // NOTE: `dumpsys` is an Android command and is not available on the host.
    // The function will return nil gracefully when the command is unavailable.
    {
        struct FpsState {
            prev_frame: u64,
            prev_time: Instant,
            initialized: bool,
        }

        let fps_state = Arc::new(Mutex::new(FpsState {
            prev_frame: 0,
            prev_time: Instant::now(),
            initialized: false,
        }));

        sys.set(
            "foreground_fps",
            lua.create_function(move |_, pkg: String| -> LuaResult<Option<f32>> {
                let output = match std::process::Command::new("dumpsys")
                    .args(["SurfaceFlinger"])
                    .output()
                {
                    Ok(o) => o,
                    Err(_) => return Ok(None), // dumpsys not available (host dev)
                };

                let text = String::from_utf8_lossy(&output.stdout);

                // Find "SurfaceView[{pkg}" marker that has a "frame=" in its
                // next few lines.  SurfaceFlinger may list multiple layers for the
                // same package (e.g. Background layer without frame counter and
                // BLAST layer with frame counter).  We iterate all occurrences.
                let marker = format!("SurfaceView[{}", pkg);
                let mut frame_count: Option<u64> = None;
                let mut search_from = 0usize;
                while let Some(rel) = text[search_from..].find(&marker) {
                    let pos = search_from + rel;
                    let after = &text[pos..];
                    frame_count = after.lines().take(5).find_map(|line| {
                        let idx = line.find("frame=")?;
                        let rest = &line[idx + 6..];
                        let num_str: String =
                            rest.chars().take_while(|c| c.is_ascii_digit()).collect();
                        num_str.parse().ok()
                    });
                    if frame_count.is_some() {
                        break; // Found a layer with frame counter
                    }
                    search_from = pos + marker.len();
                }

                let cur_frame = match frame_count {
                    Some(f) => f,
                    None => return Ok(None),
                };

                let mut state = fps_state.lock().unwrap();
                if !state.initialized {
                    state.prev_frame = cur_frame;
                    state.prev_time = Instant::now();
                    state.initialized = true;
                    return Ok(None); // First call: no delta yet
                }

                let now = Instant::now();
                let dt = now.duration_since(state.prev_time).as_secs_f32();
                if dt < 0.1 {
                    return Ok(None); // Too soon, skip
                }

                let delta_frames = cur_frame.saturating_sub(state.prev_frame);
                let fps = delta_frames as f32 / dt;

                state.prev_frame = cur_frame;
                state.prev_time = now;

                Ok(Some(fps))
            })?,
        )?;
    }

    lua.globals().set("sys", sys)?;
    Ok(())
}

/// Parse `/proc/meminfo` and return (total_kb, available_kb, free_kb).
fn read_meminfo() -> (u64, u64, u64) {
    let content = match std::fs::read_to_string("/proc/meminfo") {
        Ok(c) => c,
        Err(_) => return (0, 0, 0),
    };

    let mut total: u64 = 0;
    let mut available: u64 = 0;
    let mut free: u64 = 0;

    for line in content.lines() {
        if let Some(rest) = line.strip_prefix("MemTotal:") {
            total = parse_meminfo_value(rest);
        } else if let Some(rest) = line.strip_prefix("MemAvailable:") {
            available = parse_meminfo_value(rest);
        } else if let Some(rest) = line.strip_prefix("MemFree:") {
            free = parse_meminfo_value(rest);
        }
    }

    (total, available, free)
}

/// Parse a meminfo line value like "  12345 kB" into u64.
fn parse_meminfo_value(s: &str) -> u64 {
    s.split_whitespace()
        .next()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Write;

    fn create_temp_script(content: &str) -> tempfile::NamedTempFile {
        let mut f = tempfile::NamedTempFile::new().unwrap();
        f.write_all(content.as_bytes()).unwrap();
        f.flush().unwrap();
        f
    }

    #[test]
    fn test_empty_decide() {
        let script = create_temp_script("function decide(ctx) return {} end");
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert!(cmds.is_empty());
    }

    #[test]
    fn test_nil_decide() {
        let script = create_temp_script("function decide(ctx) return nil end");
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert!(cmds.is_empty());
    }

    #[test]
    fn test_throttle_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "throttle", delay_ms = 50}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::Throttle { delay_ms } => assert_eq!(*delay_ms, 50),
            other => panic!("expected Throttle, got {:?}", other),
        }
    }

    #[test]
    fn test_set_target_tbt_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "set_target_tbt", target_ms = 150}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::SetTargetTbt { target_ms } => assert_eq!(*target_ms, 150),
            other => panic!("expected SetTargetTbt, got {:?}", other),
        }
    }

    #[test]
    fn test_layer_skip_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "layer_skip", skip_ratio = 0.25}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::LayerSkip { skip_ratio } => {
                assert!((skip_ratio - 0.25).abs() < f32::EPSILON);
            }
            other => panic!("expected LayerSkip, got {:?}", other),
        }
    }

    #[test]
    fn test_kv_evict_h2o_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "kv_evict_h2o", keep_ratio = 0.5}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::KvEvictH2o { keep_ratio } => {
                assert!((keep_ratio - 0.5).abs() < f32::EPSILON);
            }
            other => panic!("expected KvEvictH2o, got {:?}", other),
        }
    }

    #[test]
    fn test_kv_evict_sliding_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "kv_evict_sliding", keep_ratio = 0.6}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::KvEvictSliding { keep_ratio } => {
                assert!((keep_ratio - 0.6).abs() < f32::EPSILON);
            }
            other => panic!("expected KvEvictSliding, got {:?}", other),
        }
    }

    #[test]
    fn test_kv_streaming_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "kv_streaming", sink_size = 4, window_size = 512}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::KvStreaming {
                sink_size,
                window_size,
            } => {
                assert_eq!(*sink_size, 4);
                assert_eq!(*window_size, 512);
            }
            other => panic!("expected KvStreaming, got {:?}", other),
        }
    }

    #[test]
    fn test_kv_merge_d2o_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "kv_merge_d2o", keep_ratio = 0.75}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::KvMergeD2o { keep_ratio } => {
                assert!((keep_ratio - 0.75).abs() < f32::EPSILON);
            }
            other => panic!("expected KvMergeD2o, got {:?}", other),
        }
    }

    #[test]
    fn test_kv_quant_dynamic_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "kv_quant_dynamic", target_bits = 4}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::KvQuantDynamic { target_bits } => assert_eq!(*target_bits, 4),
            other => panic!("expected KvQuantDynamic, got {:?}", other),
        }
    }

    #[test]
    fn test_switch_hw_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "switch_hw", device = "cpu"}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::SwitchHw { device } => assert_eq!(device, "cpu"),
            other => panic!("expected SwitchHw, got {:?}", other),
        }
    }

    #[test]
    fn test_restore_defaults_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "restore_defaults"}}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::RestoreDefaults));
    }

    #[test]
    fn test_suspend_resume_actions() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {
                    {type = "suspend"},
                    {type = "resume"},
                }
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 2);
        assert!(matches!(cmds[0], EngineCommand::Suspend));
        assert!(matches!(cmds[1], EngineCommand::Resume));
    }

    #[test]
    fn test_multiple_actions() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {
                    {type = "kv_evict_h2o", keep_ratio = 0.5},
                    {type = "set_target_tbt", target_ms = 200},
                }
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 2);
        assert!(matches!(cmds[0], EngineCommand::KvEvictH2o { .. }));
        assert!(matches!(cmds[1], EngineCommand::SetTargetTbt { .. }));
    }

    #[test]
    fn test_ctx_engine_state() {
        // Verify that ctx.engine fields are accessible from Lua
        let script = create_temp_script(
            r#"function decide(ctx)
                if ctx.engine.kv_util > 0.7 then
                    return {{type = "kv_evict_sliding", keep_ratio = 0.5}}
                end
                return {}
            end"#,
        );
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();

        // No heartbeat yet -- kv_util defaults to 0.0
        let cmds = policy.call_decide();
        assert!(cmds.is_empty());

        // Send heartbeat with high kv_util
        let status = EngineStatus {
            active_device: "opencl".to_string(),
            compute_level: llm_shared::ResourceLevel::Normal,
            actual_throughput: 12.5,
            memory_level: llm_shared::ResourceLevel::Normal,
            kv_cache_bytes: 84_000_000,
            kv_cache_tokens: 1200,
            kv_cache_utilization: 0.78,
            memory_lossless_min: 0.5,
            memory_lossy_min: 0.25,
            state: llm_shared::EngineState::Running,
            tokens_generated: 450,
            available_actions: vec![],
            active_actions: vec!["throttle".to_string()],
            eviction_policy: "sliding".to_string(),
            kv_dtype: "f16".to_string(),
            skip_ratio: 0.0,
            phase: String::new(),
            prefill_pos: 0,
            prefill_total: 0,
            partition_ratio: 0.0,
            self_cpu_pct: 0.0,
            self_gpu_pct: 0.0,
        };
        policy.update_engine_state(&EngineMessage::Heartbeat(status));

        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::KvEvictSliding { .. }));
    }

    #[test]
    fn test_ctx_active_actions() {
        let script = create_temp_script(
            r#"function decide(ctx)
                if #ctx.active > 0 then
                    return {{type = "restore_defaults"}}
                end
                return {}
            end"#,
        );
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();

        // No heartbeat -> empty active
        let cmds = policy.call_decide();
        assert!(cmds.is_empty());

        // Send heartbeat with active actions
        let status = EngineStatus {
            active_device: "cpu".to_string(),
            compute_level: llm_shared::ResourceLevel::Normal,
            actual_throughput: 5.0,
            memory_level: llm_shared::ResourceLevel::Normal,
            kv_cache_bytes: 0,
            kv_cache_tokens: 0,
            kv_cache_utilization: 0.0,
            memory_lossless_min: 0.5,
            memory_lossy_min: 0.25,
            state: llm_shared::EngineState::Running,
            tokens_generated: 100,
            available_actions: vec![],
            active_actions: vec!["throttle".to_string()],
            eviction_policy: "none".to_string(),
            kv_dtype: "f16".to_string(),
            skip_ratio: 0.0,
            phase: String::new(),
            prefill_pos: 0,
            prefill_total: 0,
            partition_ratio: 0.0,
            self_cpu_pct: 0.0,
            self_gpu_pct: 0.0,
        };
        policy.update_engine_state(&EngineMessage::Heartbeat(status));

        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::RestoreDefaults));
    }

    #[test]
    fn test_lua_error_returns_empty() {
        let script = create_temp_script(
            r#"function decide(ctx)
                error("intentional error")
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert!(cmds.is_empty());
    }

    #[test]
    fn test_unknown_action_skipped() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {
                    {type = "nonexistent_action"},
                    {type = "throttle", delay_ms = 10},
                }
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        // Unknown action is skipped, throttle is kept
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::Throttle { delay_ms: 10 }));
    }

    #[test]
    fn test_missing_decide_function() {
        let script = create_temp_script("-- no decide function");
        let result = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        );
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("decide"));
    }

    #[test]
    fn test_process_signal_returns_directive() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "throttle", delay_ms = 100}}
            end"#,
        );
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let signal = SystemSignal::MemoryPressure {
            level: llm_shared::Level::Warning,
            available_bytes: 1_000_000,
            total_bytes: 8_000_000,
            reclaim_target_bytes: 500_000,
        };
        let directive = policy.process_signal(&signal);
        assert!(directive.is_some());
        let d = directive.unwrap();
        assert_eq!(d.commands.len(), 1);
        assert!(d.seq_id > 0);
    }

    #[test]
    fn test_process_signal_returns_none_for_empty() {
        let script = create_temp_script("function decide(ctx) return {} end");
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let signal = SystemSignal::MemoryPressure {
            level: llm_shared::Level::Normal,
            available_bytes: 7_000_000,
            total_bytes: 8_000_000,
            reclaim_target_bytes: 0,
        };
        let directive = policy.process_signal(&signal);
        assert!(directive.is_none());
    }

    #[test]
    fn test_lua_state_preserved_across_calls() {
        let script = create_temp_script(
            r#"
            call_count = 0
            function decide(ctx)
                call_count = call_count + 1
                if call_count >= 3 then
                    return {{type = "throttle", delay_ms = call_count}}
                end
                return {}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();

        // First two calls return empty
        assert!(policy.call_decide().is_empty());
        assert!(policy.call_decide().is_empty());

        // Third call triggers throttle
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::Throttle { delay_ms } => assert_eq!(*delay_ms, 3),
            other => panic!("expected Throttle, got {:?}", other),
        }
    }

    #[test]
    fn test_mode_returns_normal() {
        let script = create_temp_script("function decide(ctx) return {} end");
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        assert_eq!(policy.mode(), OperatingMode::Normal);
    }

    #[test]
    fn test_sys_meminfo_accessible() {
        // sys.meminfo() is registered and callable (values depend on host)
        let script = create_temp_script(
            r#"function decide(ctx)
                local mem = sys.meminfo()
                if mem.total > 0 then
                    return {{type = "throttle", delay_ms = 1}}
                end
                return {}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        // On any Linux host, total should be > 0
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn test_sys_read_nonexistent_returns_empty() {
        let script = create_temp_script(
            r#"function decide(ctx)
                local val = sys.read("/nonexistent/path/12345")
                if val == "" then
                    return {{type = "throttle", delay_ms = 1}}
                end
                return {}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn test_sys_thermal_nonexistent_zone_returns_negative() {
        let script = create_temp_script(
            r#"function decide(ctx)
                local temp = sys.thermal(999)
                if temp < 0 then
                    return {{type = "throttle", delay_ms = 1}}
                end
                return {}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn test_set_partition_ratio_action_parsing() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {
                    { type = "set_partition_ratio", ratio = 0.65 },
                }
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::SetPartitionRatio { ratio } => {
                assert!((*ratio - 0.65).abs() < f32::EPSILON);
            }
            _ => panic!("Expected SetPartitionRatio"),
        }
    }

    #[test]
    fn test_set_prefill_policy_action_parsing() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {
                    { type = "set_prefill_policy", chunk_size = 48, yield_ms = 10, cpu_chunk_size = 16 },
                }
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                assert_eq!(*chunk_size, Some(48));
                assert_eq!(*yield_ms, Some(10));
                assert_eq!(*cpu_chunk_size, Some(16));
            }
            other => panic!("expected SetPrefillPolicy, got {:?}", other),
        }
    }

    #[test]
    fn test_set_prefill_policy_partial() {
        // Only chunk_size provided → yield_ms and cpu_chunk_size should be None
        let script = create_temp_script(
            r#"function decide(ctx)
                return {
                    { type = "set_prefill_policy", chunk_size = 64 },
                }
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                assert_eq!(*chunk_size, Some(64));
                assert_eq!(*yield_ms, None);
                assert_eq!(*cpu_chunk_size, None);
            }
            other => panic!("expected SetPrefillPolicy, got {:?}", other),
        }
    }

    #[test]
    fn test_set_prefill_policy_no_fields() {
        // No fields at all — all None (valid: "update nothing, keep current")
        let script = create_temp_script(
            r#"function decide(ctx)
                return {
                    { type = "set_prefill_policy" },
                }
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::SetPrefillPolicy {
                chunk_size,
                yield_ms,
                cpu_chunk_size,
            } => {
                assert_eq!(*chunk_size, None);
                assert_eq!(*yield_ms, None);
                assert_eq!(*cpu_chunk_size, None);
            }
            other => panic!("expected SetPrefillPolicy, got {:?}", other),
        }
    }

    #[test]
    fn test_prefill_phase_in_ctx() {
        // Verify ctx.engine.phase/prefill_pos/prefill_total are accessible from Lua
        let script = create_temp_script(
            r#"function decide(ctx)
                if ctx.engine.phase == "prefill" and ctx.engine.prefill_total > 0 then
                    local progress = ctx.engine.prefill_pos / ctx.engine.prefill_total
                    if progress < 0.5 then
                        return {{ type = "set_prefill_policy", chunk_size = 32 }}
                    end
                end
                return {}
            end"#,
        );
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();

        // No heartbeat yet — phase defaults to ""
        let cmds = policy.call_decide();
        assert!(cmds.is_empty());

        // Send heartbeat in prefill phase at 20%
        let status = EngineStatus {
            active_device: "opencl".to_string(),
            compute_level: llm_shared::ResourceLevel::Normal,
            actual_throughput: 0.0,
            memory_level: llm_shared::ResourceLevel::Normal,
            kv_cache_bytes: 0,
            kv_cache_tokens: 0,
            kv_cache_utilization: 0.0,
            memory_lossless_min: 0.5,
            memory_lossy_min: 0.25,
            state: llm_shared::EngineState::Running,
            tokens_generated: 0,
            available_actions: vec![],
            active_actions: vec![],
            eviction_policy: "none".to_string(),
            kv_dtype: "f16".to_string(),
            skip_ratio: 0.0,
            phase: "prefill".to_string(),
            prefill_pos: 200,
            prefill_total: 1000,
            partition_ratio: 0.0,
            self_cpu_pct: 0.0,
            self_gpu_pct: 0.0,
        };
        policy.update_engine_state(&EngineMessage::Heartbeat(status));

        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::SetPrefillPolicy { chunk_size, .. } => {
                assert_eq!(*chunk_size, Some(32));
            }
            other => panic!("expected SetPrefillPolicy, got {:?}", other),
        }
    }

    #[test]
    fn test_prefill_phase_defaults_when_no_heartbeat() {
        // Verify default values are exposed before any heartbeat arrives
        let script = create_temp_script(
            r#"function decide(ctx)
                -- Verify all three fields exist and have correct defaults
                if ctx.engine.phase == "" and
                   ctx.engine.prefill_pos == 0 and
                   ctx.engine.prefill_total == 0 then
                    return {{ type = "throttle", delay_ms = 1 }}
                end
                return {}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::Throttle { delay_ms: 1 }));
    }

    #[test]
    fn test_sys_foreground_fps_registered() {
        // Verify sys.foreground_fps is callable (returns nil on host since dumpsys is unavailable)
        let script = create_temp_script(
            r#"function decide(ctx)
                local fps = sys.foreground_fps("com.example.game")
                -- On host, fps is nil (dumpsys not available)
                -- Either nil or a number is acceptable
                if fps == nil or type(fps) == "number" then
                    return {{ type = "throttle", delay_ms = 1 }}
                end
                return {}
            end"#,
        );
        let policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn ewma_first_observation_replaces_default() {
        let mut table = EwmaReliefTable::new(0.875, HashMap::new());
        let observed = [0.6, -0.2, 0.0, 0.4, -0.1, 0.0];
        table.observe("switch_hw", &observed);
        let predicted = table.predict("switch_hw");
        assert_eq!(predicted, observed);
        assert_eq!(table.observation_count("switch_hw"), 1);
    }

    #[test]
    fn ewma_converges_toward_observed() {
        let mut table = EwmaReliefTable::new(0.875, HashMap::new());
        table.observe("throttle", &[0.5, 0.0, 0.0, 0.0, 0.0, 0.0]);
        for _ in 0..50 {
            table.observe("throttle", &[0.2, 0.0, 0.0, 0.0, 0.0, 0.0]);
        }
        let predicted = table.predict("throttle");
        assert!(
            (predicted[0] - 0.2).abs() < 0.01,
            "gpu should converge to 0.2, got {}",
            predicted[0]
        );
    }

    #[test]
    fn ewma_unknown_action_returns_default_from_config() {
        let mut defaults = HashMap::new();
        defaults.insert(
            "switch_hw".to_string(),
            vec![0.5, -0.3, 0.0, 0.3, -0.1, 0.0],
        );
        let table = EwmaReliefTable::new(0.875, defaults);
        let predicted = table.predict("switch_hw");
        assert_eq!(predicted, [0.5, -0.3, 0.0, 0.3, -0.1, 0.0]);
    }

    #[test]
    fn ewma_unknown_action_no_config_returns_zeros() {
        let table = EwmaReliefTable::new(0.875, HashMap::new());
        let predicted = table.predict("nonexistent");
        assert_eq!(predicted, [0.0; 6]);
    }

    #[test]
    fn ewma_save_load_roundtrip() {
        let mut table = EwmaReliefTable::new(0.875, HashMap::new());
        table.observe("switch_hw", &[0.5, -0.3, 0.0, 0.3, -0.1, 0.0]);
        table.observe("throttle", &[0.0, 0.3, 0.0, 0.2, -0.2, 0.0]);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("relief.json");

        table.save(&path).unwrap();
        let loaded = EwmaReliefTable::load(&path, 0.875, HashMap::new()).unwrap();

        assert_eq!(loaded.predict("switch_hw"), table.predict("switch_hw"));
        assert_eq!(loaded.predict("throttle"), table.predict("throttle"));
        assert_eq!(loaded.observation_count("switch_hw"), 1);
    }

    #[test]
    fn ewma_negative_relief_handled() {
        let mut table = EwmaReliefTable::new(0.875, HashMap::new());
        table.observe("switch_hw", &[0.5, -0.3, 0.0, 0.3, -0.1, 0.0]);
        let predicted = table.predict("switch_hw");
        assert!(predicted[1] < 0.0, "cpu should be negative");
        assert!(predicted[4] < 0.0, "latency should be negative");
    }

    #[test]
    fn signal_state_pressure_from_compute() {
        let mut state = SignalState::default();
        state.update_compute(45.2, 82.1);
        let p = state.pressure_with_thermal(35.0, 50.0, None);
        assert!((p.gpu - 0.821).abs() < 0.001);
        assert!((p.cpu - 0.452).abs() < 0.001);
    }

    #[test]
    fn signal_state_pressure_from_memory() {
        let mut state = SignalState::default();
        state.update_memory(2_000_000, 8_000_000);
        let p = state.pressure_with_thermal(35.0, 50.0, None);
        assert!((p.memory - 0.75).abs() < 0.001);
    }

    #[test]
    fn signal_state_pressure_thermal_normalized() {
        let mut state = SignalState::default();
        state.update_thermal(42500, false);
        let p = state.pressure_with_thermal(35.0, 50.0, None);
        assert!((p.thermal - 0.5).abs() < 0.01);
    }

    #[test]
    fn signal_state_pressure_thermal_clamped() {
        let mut state = SignalState::default();
        state.update_thermal(55000, true);
        let p = state.pressure_with_thermal(35.0, 50.0, None);
        assert!((p.thermal - 1.0).abs() < 0.001);
    }

    #[test]
    fn trigger_tbt_warmup_then_degrade() {
        let config = TriggerConfig {
            tbt_enter: 0.30,
            tbt_exit: 0.10,
            tbt_warmup_tokens: 5,
            mem_enter: 0.80,
            mem_exit: 0.60,
            temp_enter: 0.70,
            temp_exit: 0.50,
        };
        let mut engine = TriggerEngine::new(config);

        for _ in 0..5 {
            engine.update_tbt_from_throughput(10.0);
        }
        assert!(!engine.state().tbt_degraded);

        for _ in 0..10 {
            engine.update_tbt_from_throughput(5.0);
        }
        assert!(engine.state().tbt_degraded);
    }

    #[test]
    fn trigger_tbt_zero_throughput_skipped() {
        let config = TriggerConfig::default();
        let mut engine = TriggerEngine::new(config);
        engine.update_tbt_from_throughput(0.0);
        assert!(!engine.state().tbt_degraded);
    }

    #[test]
    fn trigger_hysteresis_mem() {
        let config = TriggerConfig {
            tbt_enter: 0.30,
            tbt_exit: 0.10,
            tbt_warmup_tokens: 20,
            mem_enter: 0.80,
            mem_exit: 0.60,
            temp_enter: 0.70,
            temp_exit: 0.50,
        };
        let mut engine = TriggerEngine::new(config);

        engine.update_mem(0.85);
        assert!(engine.state().mem_low);

        engine.update_mem(0.65);
        assert!(engine.state().mem_low); // still active (between exit and enter)

        engine.update_mem(0.55);
        assert!(!engine.state().mem_low); // below exit threshold
    }

    #[test]
    fn trigger_hysteresis_temp() {
        let config = TriggerConfig {
            tbt_enter: 0.30,
            tbt_exit: 0.10,
            tbt_warmup_tokens: 20,
            mem_enter: 0.80,
            mem_exit: 0.60,
            temp_enter: 0.70,
            temp_exit: 0.50,
        };
        let mut engine = TriggerEngine::new(config);

        engine.update_temp(0.75);
        assert!(engine.state().temp_high);

        engine.update_temp(0.55);
        assert!(engine.state().temp_high);

        engine.update_temp(0.45);
        assert!(!engine.state().temp_high);
    }

    #[test]
    fn lua_policy_e2e_signal_to_ctx() {
        use crate::config::AdaptationConfig;
        use crate::pipeline::PolicyStrategy;
        use llm_shared::SystemSignal;

        let script = r#"
            function decide(ctx)
                assert(ctx.coef, "ctx.coef missing")
                assert(ctx.coef.pressure, "ctx.coef.pressure missing")
                assert(ctx.coef.trigger, "ctx.coef.trigger missing")
                assert(ctx.coef.relief, "ctx.coef.relief missing")
                assert(ctx.coef.pressure.gpu >= 0)
                assert(ctx.coef.pressure.memory >= 0)
                assert(type(ctx.coef.trigger.tbt_degraded) == "boolean")
                assert(type(ctx.coef.trigger.mem_low) == "boolean")
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
                m.insert(
                    "switch_hw".to_string(),
                    vec![0.5, -0.3, 0.0, 0.3, -0.1, 0.0],
                );
                m
            },
            ..AdaptationConfig::default()
        };

        let mut policy =
            LuaPolicy::with_system_clock(script_path.to_str().unwrap(), config).unwrap();

        let signal = SystemSignal::ComputeGuidance {
            level: llm_shared::Level::Warning,
            recommended_backend: llm_shared::RecommendedBackend::Cpu,
            reason: llm_shared::ComputeReason::CpuBottleneck,
            cpu_usage_pct: 45.0,
            gpu_usage_pct: 82.0,
        };

        let result = policy.process_signal(&signal);
        assert!(result.is_none());
    }

    #[test]
    fn lua_policy_e2e_trigger_fires() {
        use crate::config::AdaptationConfig;
        use crate::pipeline::PolicyStrategy;
        use llm_shared::{EngineCommand, SystemSignal};

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
        let mut policy =
            LuaPolicy::with_system_clock(script_path.to_str().unwrap(), config).unwrap();

        // Memory pressure: available=1MB / total=8MB = 87.5% used > 80% enter
        let signal = SystemSignal::MemoryPressure {
            level: llm_shared::Level::Warning,
            available_bytes: 1_000_000,
            total_bytes: 8_000_000,
            reclaim_target_bytes: 0,
        };

        let result = policy.process_signal(&signal);
        assert!(result.is_some(), "Expected directive when mem_low triggers");
        let directive = result.unwrap();
        assert_eq!(directive.commands.len(), 1);
        assert!(
            matches!(directive.commands[0], EngineCommand::KvEvictH2o { keep_ratio } if (keep_ratio - 0.5).abs() < f32::EPSILON),
            "Expected KvEvictH2o with keep_ratio=0.5"
        );
    }
}
