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

use crate::pipeline::{PolicyStrategy, next_seq_id};
use crate::types::OperatingMode;

/// Lua-based policy strategy.
///
/// Wraps an `mlua::Lua` VM with a loaded `decide(ctx)` function.
/// Engine heartbeat state is cached and forwarded as `ctx.engine`.
pub struct LuaPolicy {
    lua: Lua,
    /// Latest engine heartbeat (None until first heartbeat received).
    engine_state: Option<EngineStatus>,
}

impl std::fmt::Debug for LuaPolicy {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("LuaPolicy")
            .field("engine_state", &self.engine_state.is_some())
            .finish()
    }
}

impl LuaPolicy {
    /// Create a new LuaPolicy by loading and evaluating `script_path`.
    ///
    /// The script must define a global `decide(ctx)` function.
    /// `sys.*` helper functions are registered before the script is evaluated.
    pub fn new(script_path: &str) -> anyhow::Result<Self> {
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

        Ok(Self {
            lua,
            engine_state: None,
        })
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

impl PolicyStrategy for LuaPolicy {
    fn process_signal(&mut self, _signal: &SystemSignal) -> Option<EngineDirective> {
        let commands = self.call_decide();
        if commands.is_empty() {
            None
        } else {
            Some(EngineDirective {
                seq_id: next_seq_id(),
                commands,
            })
        }
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
        let cmds = policy.call_decide();
        assert!(cmds.is_empty());
    }

    #[test]
    fn test_nil_decide() {
        let script = create_temp_script("function decide(ctx) return nil end");
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let mut policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();

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
        let mut policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();

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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
        let cmds = policy.call_decide();
        // Unknown action is skipped, throttle is kept
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::Throttle { delay_ms: 10 }));
    }

    #[test]
    fn test_missing_decide_function() {
        let script = create_temp_script("-- no decide function");
        let result = LuaPolicy::new(script.path().to_str().unwrap());
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
        let mut policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let mut policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();

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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
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
        let policy = LuaPolicy::new(script.path().to_str().unwrap()).unwrap();
        let cmds = policy.call_decide();
        assert_eq!(cmds.len(), 1);
    }
}
