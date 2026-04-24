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

use llm_shared::{
    EngineCommand, EngineDirective, EngineMessage, EngineStatus, QcfEstimate, SystemSignal,
};
use mlua::{Lua, Result as LuaResult, StdLib, Table, Value};
use serde::{Deserialize, Serialize};
use std::collections::{HashMap, VecDeque};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::Instant;

use crate::clock::{Clock, LogicalInstant, SystemClock};
use crate::config::{AdaptationConfig, TriggerConfig};
use crate::monitor::compute::SharedGpuProvider;
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
        // 첫 관측도 EWMA로 처리한다. entry 초기값은 default_relief prior로 설정하여
        // outlier 단일 관측이 prior를 완전히 소실시키는 문제를 방지한다.
        let default = self
            .defaults
            .get(action)
            .map(|v| {
                let mut r = [0.0f32; RELIEF_DIMS];
                for (i, &val) in v.iter().enumerate().take(RELIEF_DIMS) {
                    r[i] = val;
                }
                r
            })
            .unwrap_or([0.0f32; RELIEF_DIMS]);

        let entry = self
            .entries
            .entry(action.to_string())
            .or_insert_with(|| ReliefEntry {
                relief: default,
                observation_count: 0,
            });

        // 항상 EWMA 적용 (첫 관측도 동일하게)
        let a = self.alpha;
        for (i, &obs_val) in observed.iter().enumerate() {
            entry.relief[i] = a * entry.relief[i] + (1.0 - a) * obs_val;
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

    /// defaults에 설정된 초기값 스냅샷을 반환한다 (sim_run 진단용).
    pub fn initial_snapshot(&self) -> HashMap<String, [f32; RELIEF_DIMS]> {
        self.defaults
            .iter()
            .map(|(k, v)| {
                let mut arr = [0.0f32; RELIEF_DIMS];
                for (i, &val) in v.iter().enumerate().take(RELIEF_DIMS) {
                    arr[i] = val;
                }
                (k.clone(), arr)
            })
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

/// LinUCB feature vector 차원 (13).
///
/// indices:
///   0 = KV_OCCUPANCY (kv_cache_utilization)
///   1 = IS_GPU (active_device contains "opencl")
///   2 = TOKEN_PROGRESS (tokens_generated / 2048, clamped to 1.0)
///   3 = IS_PREFILL (phase == "prefill")
///   4 = KV_DTYPE_NORM (f32=0.0, f16=0.5, q4=1.0)
///   5 = TBT_RATIO (tbt_degradation_ratio)
///   6 = TOKENS_GEN_NORM (same as TOKEN_PROGRESS – kept for symmetry)
///   7 = ACTIVE_SWITCH_HW
///   8 = ACTIVE_THROTTLE
///   9 = ACTIVE_KV_OFFLOAD
///  10 = ACTIVE_EVICTION (kv_evict_*)
///  11 = ACTIVE_LAYER_SKIP
///  12 = ACTIVE_KV_QUANT
pub const LINUCB_FEATURE_DIM: usize = 13;

/// LinUCB exploration bonus 계산기.
///
/// 각 액션마다 D×D P matrix (= A_a^{-1})를 관리한다.
/// UCB bonus = sqrt(phi^T · P_a · phi).
///
/// 평균 relief는 EwmaReliefTable이 담당하고,
/// 이 구조체는 exploration bonus 전용이다.
struct LinUcbTable {
    /// action_name → D×D P matrix (flat row-major, length = feature_dim²)
    matrices: HashMap<String, Vec<f32>>,
    feature_dim: usize,
}

impl LinUcbTable {
    fn new() -> Self {
        Self {
            matrices: HashMap::new(),
            feature_dim: LINUCB_FEATURE_DIM,
        }
    }

    /// P matrix가 없으면 identity로 초기화 (최대 불확실성 = 최대 탐색).
    fn ensure_matrix(&mut self, action: &str) {
        if self.matrices.contains_key(action) {
            return;
        }
        let d = self.feature_dim;
        let mut p = vec![0.0f32; d * d];
        for i in 0..d {
            p[i * d + i] = 1.0;
        }
        self.matrices.insert(action.to_string(), p);
    }

    /// UCB bonus = sqrt(max(0, phi^T · P · phi)).
    /// P matrix가 없으면 1.0 반환 (identity 기대값 ≈ ||phi||, cold-start 탐색 최대).
    fn ucb_bonus(&self, action: &str, phi: &[f32]) -> f32 {
        let p = match self.matrices.get(action) {
            Some(m) => m,
            None => return 1.0,
        };
        let d = self.feature_dim;
        // v = P · phi  (D벡터)
        let mut v = vec![0.0f32; d];
        for i in 0..d {
            v[i] = phi.iter().enumerate().map(|(j, &x)| p[i * d + j] * x).sum();
        }
        // phi^T · v (스칼라)
        let val: f32 = phi.iter().zip(v.iter()).map(|(&x, &vi)| x * vi).sum();
        val.max(0.0).sqrt()
    }

    /// Sherman-Morrison P matrix 업데이트:
    /// P ← P − (P·φ·φᵀ·P) / (1 + φᵀ·P·φ)
    ///
    /// λ=1.0 (망각 없음). 탐색 목적이므로 P는 단조 감소가 맞다.
    fn update(&mut self, action: &str, phi: &[f32]) {
        self.ensure_matrix(action);
        let p = self.matrices.get_mut(action).unwrap();
        let d = self.feature_dim;

        // p_phi = P · phi
        let mut p_phi = vec![0.0f32; d];
        for i in 0..d {
            p_phi[i] = phi.iter().enumerate().map(|(j, &x)| p[i * d + j] * x).sum();
        }

        // denom = 1 + phi^T · p_phi
        let denom: f32 = 1.0
            + phi
                .iter()
                .zip(p_phi.iter())
                .map(|(&x, &v)| x * v)
                .sum::<f32>();

        // P ← P − (p_phi · p_phi^T) / denom
        for i in 0..d {
            for j in 0..d {
                p[i * d + j] -= p_phi[i] * p_phi[j] / denom;
            }
        }
    }
}

pub const OBSERVATION_DELAY_SECS: f64 = 3.0;

/// Lua 스크립트의 POLICY_META 테이블에서 읽어온 이름과 버전.
#[derive(Debug, Clone, Default)]
#[allow(dead_code)]
struct PolicyMeta {
    name: String,
    version: String,
}

/// 히스토리 링 버퍼에 저장되는 단일 tick 상태 스냅샷.
#[derive(Debug, Clone)]
struct HistoryEntry {
    /// 기록 시각 (단조 시계 기준 초).
    at_s: f64,
    /// 6D 압박 벡터 [gpu, cpu, memory, thermal, latency, main_app].
    pressure: [f32; 6],
    /// 해당 tick에 엔진에 활성화된 action 이름 목록.
    active_actions: Vec<String>,
}

/// Lua 오류가 이 횟수 이상 연속 발생하면 영구 fallback으로 전환한다.
const FALLBACK_ERROR_THRESHOLD: u32 = 3;

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
    feature_vec: [f32; LINUCB_FEATURE_DIM], // 큐잉 시점의 phi 스냅샷
}

/// Lua-based policy strategy.
///
/// Wraps an `mlua::Lua` VM with a loaded `decide(ctx)` function.
/// Engine heartbeat state is cached and forwarded as `ctx.engine`.
pub struct LuaPolicy {
    lua: Lua,
    /// Latest engine heartbeat (None until first heartbeat received).
    engine_state: Option<EngineStatus>,
    /// 엔진이 Capability 로 보고한 `available_actions` 목록.
    /// Heartbeat 보다 먼저 도착하므로 (그리고 이후 heartbeat 에는 포함되지 않음)
    /// 별도 필드로 보관하여 decide() 의 available 필터에 사용한다.
    engine_available_actions: Vec<String>,
    // ── 신규 필드 ──
    signal_state: SignalState,
    trigger_engine: TriggerEngine,
    relief_table: EwmaReliefTable,
    linucb: LinUcbTable,
    /// 현재 tick의 feature vector. 관측 큐잉 시 스냅샷.
    feature_state: [f32; LINUCB_FEATURE_DIM],
    /// LinUCB UCB 탐색 가중치 (config.linucb_alpha).
    linucb_alpha: f32,
    /// QCF cost cache: action_name → (cost, observed_at)
    qcf_cache: HashMap<String, (f32, LogicalInstant)>,
    /// RequestQcf 발행 시각. None이면 pending 없음.
    qcf_pending_at: Option<LogicalInstant>,
    /// RequestQcf 를 유발한 signal. `complete_qcf_selection` 에서 재사용하여
    /// 2-step handshake (QCF 응답 수신 직후 decide() 실행) 를 성사시킨다.
    /// 이 필드가 없으면 QcfEstimate 수신 후 cache만 갱신되고 후속 directive 가
    /// 발행되지 않아 signal 경로가 seq=1 에서 멈춘다.
    qcf_pending_signal: Option<SystemSignal>,
    /// QCF quality penalty weight (V_Q). config에서 주입.
    qcf_penalty_weight: f32,
    /// QCF cache TTL (초). 초과 시 stale로 판단해 재요청.
    qcf_stale_secs: f64,
    observations: VecDeque<ObservationContext>,
    adaptation_config: AdaptationConfig,
    clock: Arc<dyn Clock>,
    /// 최근 발생한 relief 업데이트 이벤트 (시뮬레이터가 drain한다).
    pending_relief_updates: Vec<ReliefUpdateEvent>,
    /// `MAX_PENDING_OBSERVATIONS` 용량 초과로 드롭된 observation 수.
    /// 큐 용량이 충분하면 0이어야 한다 — 값이 계속 증가하면 directive 방출률이
    /// 용량을 초과했다는 뜻이므로 `MAX_PENDING_OBSERVATIONS` 조정을 검토.
    observation_overrun_count: u64,
    /// 연속 Lua 오류 횟수. FALLBACK_ERROR_THRESHOLD 도달 시 permanent_fallback으로 전환.
    consecutive_errors: u32,
    /// true이면 Lua VM을 호출하지 않고 내장 fallback 로직을 사용한다.
    permanent_fallback: bool,
    /// 로드된 Lua 스크립트 경로 (hot-reload에 사용).
    script_path: std::path::PathBuf,
    /// 로드된 Lua 스크립트의 POLICY_META (없으면 None).
    policy_meta: Option<PolicyMeta>,
    /// 최근 10 tick의 상태 스냅샷 (오래된 것이 front).
    history: VecDeque<HistoryEntry>,
    /// relief table 마지막 자동 저장 시각.
    last_persist_at: Option<std::time::Instant>,
    /// 배타 그룹 맵 — ctx.is_joint_valid()에서 참조한다.
    exclusion_groups: HashMap<String, Vec<String>>,
    /// GPU telemetry provider — `sys.gpu_freq()`, `sys.gpu_busy()` Lua 헬퍼에서 참조한다.
    /// `ComputeMonitor`와 동일 인스턴스를 공유하여 tegrastats child 중복 spawn을 방지한다.
    gpu_provider: SharedGpuProvider,
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

/// Lua globals에서 POLICY_META 테이블을 읽어 로그에 기록하고 반환한다.
///
/// 스크립트가 POLICY_META를 정의하지 않으면 None을 반환한다.
fn log_policy_meta(lua: &Lua) -> Option<PolicyMeta> {
    let globals = lua.globals();
    let meta_table: mlua::Result<mlua::Table> = globals.get("POLICY_META");
    match meta_table {
        Ok(t) => {
            let name: String = t.get("name").unwrap_or_default();
            let version: String = t.get("version").unwrap_or_default();
            log::info!("Policy loaded: name={:?} version={:?}", name, version);
            Some(PolicyMeta { name, version })
        }
        Err(_) => {
            log::debug!("POLICY_META not defined in Lua script");
            None
        }
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
        Self::new_with_exclusions(script_path, config, clock, HashMap::new())
    }

    /// 배타 그룹 맵을 주입하여 LuaPolicy를 생성한다.
    ///
    /// `exclusion_groups`는 PolicyConfig에서 파싱한 문자열 맵이며,
    /// `ctx.is_joint_valid(actions)` Lua 함수에서 참조된다.
    pub fn new_with_exclusions(
        script_path: &str,
        config: AdaptationConfig,
        clock: Arc<dyn Clock>,
        exclusion_groups: HashMap<String, Vec<String>>,
    ) -> anyhow::Result<Self> {
        Self::new_with_gpu(
            script_path,
            config,
            clock,
            exclusion_groups,
            crate::monitor::gpu_provider::shared_null(),
        )
    }

    /// Production 경로 — 공유 GPU provider를 주입받는다.
    /// `ComputeMonitor`와 동일 인스턴스를 공유하면 tegrastats child가 중복 spawn되지 않는다.
    pub fn new_with_gpu(
        script_path: &str,
        config: AdaptationConfig,
        clock: Arc<dyn Clock>,
        exclusion_groups: HashMap<String, Vec<String>>,
        gpu_provider: SharedGpuProvider,
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
        register_sys_helpers(&lua, Arc::clone(&gpu_provider))
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
        let policy_meta = log_policy_meta(&lua);

        Ok(Self {
            lua,
            engine_state: None,
            engine_available_actions: Vec::new(),
            signal_state: SignalState::default(),
            trigger_engine,
            relief_table,
            linucb: LinUcbTable::new(),
            feature_state: [0.0f32; LINUCB_FEATURE_DIM],
            linucb_alpha: config.linucb_alpha,
            qcf_cache: HashMap::new(),
            qcf_pending_at: None,
            qcf_pending_signal: None,
            qcf_penalty_weight: config.qcf_penalty_weight,
            qcf_stale_secs: config.qcf_stale_secs,
            observations: VecDeque::new(),
            adaptation_config: config,
            clock,
            pending_relief_updates: Vec::new(),
            observation_overrun_count: 0,
            consecutive_errors: 0,
            permanent_fallback: false,
            script_path: script_path.into(),
            policy_meta,
            history: VecDeque::with_capacity(10),
            // 시작 시각을 기준점으로 초기화하여 첫 auto-persist는 인터벌 경과 후에 발생한다.
            // None이면 첫 process_signal 호출 즉시 저장되므로 MGR-093 위반.
            last_persist_at: Some(std::time::Instant::now()),
            exclusion_groups,
            gpu_provider,
        })
    }

    /// 편의 생성자 — SystemClock + null GPU provider.
    /// 테스트/시뮬레이터 또는 GPU telemetry가 필요 없는 환경에서 사용.
    pub fn with_system_clock(script_path: &str, config: AdaptationConfig) -> anyhow::Result<Self> {
        Self::new(script_path, config, Arc::new(SystemClock::new()))
    }

    /// Production 경로 — SystemClock + 공유 GPU provider.
    pub fn with_system_clock_and_gpu(
        script_path: &str,
        config: AdaptationConfig,
        gpu_provider: SharedGpuProvider,
    ) -> anyhow::Result<Self> {
        Self::new_with_gpu(
            script_path,
            config,
            Arc::new(SystemClock::new()),
            HashMap::new(),
            gpu_provider,
        )
    }

    /// QCF cache가 stale한지 확인 (비어있거나 TTL 초과 시 true).
    fn qcf_cache_is_stale(&self) -> bool {
        if self.qcf_cache.is_empty() {
            return true;
        }
        self.qcf_cache.values().any(|(_, observed_at)| {
            self.clock.elapsed_since(*observed_at).as_secs_f64() > self.qcf_stale_secs
        })
    }

    /// 현재 신호 상태에서 QCF 요청이 필요한지 판단.
    fn should_request_qcf(&self) -> bool {
        // V_Q = 0이면 QCF penalty를 사용하지 않으므로 요청 불필요
        if self.qcf_penalty_weight == 0.0 {
            return false;
        }
        if self.qcf_pending_at.is_some() {
            return false;
        }
        let mem_pressure = if self.signal_state.mem_total > 0 {
            1.0 - (self.signal_state.mem_available as f64 / self.signal_state.mem_total as f64)
        } else {
            0.0
        };
        let mem_high = mem_pressure >= 0.6;
        let cpu_high = self.signal_state.cpu_pct >= 70.0;
        let thermal_high = {
            let temp_c = self.signal_state.temp_mc as f32 / 1000.0;
            let safe = self.adaptation_config.temp_safe_c;
            let critical = self.adaptation_config.temp_critical_c;
            let range = critical - safe;
            if range > 0.0 {
                ((temp_c - safe) / range) >= 0.7
            } else {
                false
            }
        };
        if !mem_high && !cpu_high && !thermal_high {
            return false;
        }
        self.qcf_cache_is_stale()
    }

    /// 현재 엔진 상태 + signal_state에서 13차원 feature vector를 빌드.
    fn build_feature_vec(&self) -> [f32; LINUCB_FEATURE_DIM] {
        let mut phi = [0.0f32; LINUCB_FEATURE_DIM];

        if let Some(ref status) = self.engine_state {
            // 0: KV_OCCUPANCY
            phi[0] = status.kv_cache_utilization.clamp(0.0, 1.0);
            // 1: IS_GPU
            phi[1] = if status.active_device.to_lowercase().contains("opencl") {
                1.0
            } else {
                0.0
            };
            // 2 & 6: TOKEN_PROGRESS / TOKENS_GEN_NORM (같은 값)
            let tok_norm = (status.tokens_generated as f32 / 2048.0).min(1.0);
            phi[2] = tok_norm;
            phi[6] = tok_norm;
            // 3: IS_PREFILL
            phi[3] = if status.phase == "prefill" { 1.0 } else { 0.0 };
            // 4: KV_DTYPE_NORM
            phi[4] = match status.kv_dtype.as_str() {
                "f32" => 0.0,
                "f16" => 0.5,
                _ => 1.0, // q4_0 등 양자화 포맷
            };
            // 7-12: ACTIVE_* flags
            for action in &status.active_actions {
                match action.as_str() {
                    "switch_hw" => phi[7] = 1.0,
                    "throttle" | "set_target_tbt" => phi[8] = 1.0,
                    "kv_offload_disk" => phi[9] = 1.0,
                    "kv_evict_h2o" | "kv_evict_sliding" | "kv_merge_d2o" => phi[10] = 1.0,
                    "layer_skip" => phi[11] = 1.0,
                    "kv_quant_dynamic" => phi[12] = 1.0,
                    _ => {}
                }
            }
        }

        // 5: TBT_RATIO
        phi[5] = self.trigger_engine.tbt_degradation_ratio().unwrap_or(0.0) as f32;

        phi
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

        // ctx.available -- 엔진이 실행 가능하다고 보고한 action 이름 목록.
        // 비어있으면 "엔진이 아직 capability 를 보고하지 않음" 으로 해석 —
        // 이 경우 Lua 측은 필터링하지 않는다 (backward compat).
        // Capability 메시지가 heartbeat 보다 먼저 도착하므로 engine_state 가 None
        // 이어도 engine_available_actions 에서 가져온다.
        // 엔진이 F16 KV 를 사용하면 kv_quant_dynamic 이 제외되는 등, cache dtype /
        // eviction_policy 에 따라 동적으로 결정된다
        // (engine/src/resilience/executor.rs `compute_available_actions`).
        let avail_tbl = lua.create_table()?;
        let avail_source: &[String] = if let Some(ref status) = self.engine_state
            && !status.available_actions.is_empty()
        {
            &status.available_actions
        } else {
            &self.engine_available_actions
        };
        for (i, action) in avail_source.iter().enumerate() {
            avail_tbl.set(i + 1, action.as_str())?;
        }
        ctx.set("available", avail_tbl)?;

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
                let ucb = self.linucb.ucb_bonus(name, &self.feature_state) * self.linucb_alpha;
                let entry = lua.create_table()?;
                entry.set("gpu", relief[0])?;
                entry.set("cpu", relief[1])?;
                entry.set("memory", relief[2])?;
                entry.set("thermal", relief[3])?;
                entry.set("lat", relief[4])?;
                entry.set("qos", relief[5])?;
                entry.set("ucb_bonus", ucb)?;
                // QCF cost: cache에서 조회. 미수신 시 0 (penalty 없음), age=MAX
                let (qcf_cost, qcf_age_s) = match self.qcf_cache.get(*name) {
                    Some(&(cost, observed_at)) => {
                        let age = self.clock.elapsed_since(observed_at).as_secs_f64() as f32;
                        (cost, age)
                    }
                    None => (0.0_f32, f32::MAX),
                };
                entry.set("qcf_cost", qcf_cost)?;
                entry.set("qcf_age_s", qcf_age_s)?;
                r_tbl.set(*name, entry)?;
            }
            coef.set("relief", r_tbl)?;

            // coef.qcf_penalty_weight — Lua script가 V_Q를 config 값으로 override할 때 사용
            coef.set("qcf_penalty_weight", self.qcf_penalty_weight)?;
        }
        ctx.set("coef", coef)?;

        // ctx.history — 최근 N tick의 상태 스냅샷 (오래된 것이 index 1)
        let history_table = self.lua.create_table()?;
        for (i, entry) in self.history.iter().enumerate() {
            let h = self.lua.create_table()?;
            h.set("at_s", entry.at_s)?;

            let p = self.lua.create_table()?;
            p.set("gpu", entry.pressure[0])?;
            p.set("cpu", entry.pressure[1])?;
            p.set("memory", entry.pressure[2])?;
            p.set("thermal", entry.pressure[3])?;
            p.set("latency", entry.pressure[4])?;
            p.set("main_app", entry.pressure[5])?;
            h.set("pressure", p)?;

            let acts = self.lua.create_table()?;
            for (j, a) in entry.active_actions.iter().enumerate() {
                acts.set(j + 1, a.as_str())?;
            }
            h.set("active", acts)?;

            history_table.set(i + 1, h)?;
        }
        ctx.set("history", history_table)?;

        // ctx.is_joint_valid(action_list) — Lua에서 joint action 검증에 사용
        // 인자: string 배열 (Lua table). 반환: boolean.
        // 배타 그룹에 속한 두 액션이 동시에 포함되면 false를 반환한다.
        {
            let groups = self.exclusion_groups.clone();
            let is_joint_valid = self.lua.create_function(move |_, tbl: Table| {
                let mut names: Vec<String> = Vec::new();
                for pair in tbl.pairs::<mlua::Value, String>() {
                    match pair {
                        Ok((_, v)) => names.push(v),
                        Err(_) => continue,
                    }
                }
                for members in groups.values() {
                    let mut count = 0usize;
                    for name in &names {
                        if members.contains(name) {
                            count += 1;
                        }
                        if count >= 2 {
                            return Ok(false);
                        }
                    }
                }
                Ok(true)
            })?;
            ctx.set("is_joint_valid", is_joint_valid)?;
        }

        Ok(ctx)
    }

    /// 내장 fallback 정책 — Lua VM 없이 신호 레벨만으로 커맨드를 생성한다.
    ///
    /// `permanent_fallback`이 true일 때 또는 Lua 오류 발생 시 대체 경로로 사용된다.
    fn fallback_decide(&self, signal: &SystemSignal) -> Vec<EngineCommand> {
        use llm_shared::Level;
        match signal {
            SystemSignal::MemoryPressure { level, .. } => match level {
                Level::Normal => vec![EngineCommand::RestoreDefaults],
                Level::Warning => vec![EngineCommand::KvEvictSliding { keep_ratio: 0.85 }],
                Level::Critical => vec![EngineCommand::KvEvictSliding { keep_ratio: 0.50 }],
                Level::Emergency => vec![
                    EngineCommand::KvEvictSliding { keep_ratio: 0.25 },
                    EngineCommand::SetTargetTbt { target_ms: 500 },
                ],
            },
            SystemSignal::ThermalAlert { level, .. } => match level {
                Level::Normal | Level::Warning => vec![],
                Level::Critical => vec![EngineCommand::Throttle { delay_ms: 100 }],
                Level::Emergency => vec![EngineCommand::Throttle { delay_ms: 200 }],
            },
            SystemSignal::ComputeGuidance { level, .. } => match level {
                Level::Normal | Level::Warning => vec![],
                Level::Critical => vec![EngineCommand::Throttle { delay_ms: 50 }],
                Level::Emergency => vec![EngineCommand::Throttle { delay_ms: 150 }],
            },
            SystemSignal::EnergyConstraint { .. } => vec![],
        }
    }

    /// Call `decide(ctx)` and parse the returned table into `Vec<EngineCommand>`.
    ///
    /// 반환값: `(commands, was_fallback)`.
    /// Lua 오류가 `FALLBACK_ERROR_THRESHOLD`회 연속 발생하면 `permanent_fallback = true`로
    /// 전환하고 이후 호출에서는 Lua VM을 완전히 우회한다.
    fn call_decide(&mut self, signal: &SystemSignal) -> (Vec<EngineCommand>, bool) {
        // 영구 fallback 모드이면 즉시 반환
        if self.permanent_fallback {
            return (self.fallback_decide(signal), true);
        }

        let result: LuaResult<Vec<EngineCommand>> = (|| {
            let globals = self.lua.globals();
            let decide: mlua::Function = globals.get("decide")?;
            let ctx = self.build_ctx()?;
            let result: Value = decide.call(ctx)?;
            parse_actions(result)
        })();

        match result {
            Ok(commands) => {
                self.consecutive_errors = 0;
                (commands, false)
            }
            Err(e) => {
                self.consecutive_errors += 1;
                log::error!(
                    "Lua decide() error (consecutive={}): {}",
                    self.consecutive_errors,
                    e
                );
                if self.consecutive_errors >= FALLBACK_ERROR_THRESHOLD {
                    self.permanent_fallback = true;
                    log::error!(
                        "LuaPolicy: {} consecutive errors — switching to permanent fallback \
                         (script: {})",
                        self.consecutive_errors,
                        self.script_path.display()
                    );
                }
                (self.fallback_decide(signal), true)
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

        // LinUCB P matrix 업데이트 — 해당 방향 탐색 완료 처리
        self.linucb.update(&obs.action, &obs.feature_vec);

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

    /// Relief table을 인터벌 조건 충족 시 자동 저장한다 (INV-103: 실패 시 로그만).
    fn maybe_persist(&mut self) {
        let interval = self.adaptation_config.persist_interval_secs;
        if interval <= 0.0 {
            return;
        }
        let path = &self.adaptation_config.relief_table_path;
        if path.is_empty() {
            return;
        }

        let should_persist = match &self.last_persist_at {
            None => true,
            Some(t) => t.elapsed().as_secs_f64() >= interval,
        };
        if !should_persist {
            return;
        }

        let path_buf = self.adaptation_config.relief_table_path.clone();
        let path = std::path::Path::new(&path_buf);
        match self.relief_table.save(path) {
            Ok(()) => {
                log::debug!("Relief table auto-persisted to {}", path.display());
                self.last_persist_at = Some(std::time::Instant::now());
            }
            Err(e) => {
                log::warn!("Relief table auto-persist failed: {}", e);
                // INV-103: 저장 실패는 로그만, 동작 중단하지 않음
            }
        }
    }

    /// `signal` 에 대해 Lua `decide(ctx)` 를 실행하고, 반환된 커맨드로
    /// `EngineDirective` 를 조립한다. Observation 큐잉과 fallback 처리도 동일하게
    /// 수행한다. `process_signal` 본문과 `complete_qcf_selection` 이 공유한다.
    fn decide_and_build_directive(&mut self, signal: &SystemSignal) -> Option<EngineDirective> {
        // feature_state를 현재 시점으로 갱신 (LinUCB용)
        self.feature_state = self.build_feature_vec();
        let (commands, was_fallback) = self.call_decide(signal);
        if commands.is_empty() {
            return None;
        }
        // fallback 경로의 커맨드는 relief 학습 대상이 아니므로 observation 큐잉 생략.
        if !was_fallback {
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
                    feature_vec: self.feature_state,
                });
            } else if !self.observations.is_empty() {
                // Multi-command directive: 어느 command가 relief를 유발했는지
                // 귀속 불가 → 이전 observation 을 모두 overrun 으로 비운다.
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

        // 3. History 갱신 (check_observation 전에 현재 상태를 스냅샷)
        {
            let at_s = self.clock.now().as_duration_since_start().as_secs_f64();
            let pressure = self.signal_state.pressure_with_thermal(
                self.adaptation_config.temp_safe_c,
                self.adaptation_config.temp_critical_c,
                self.trigger_engine.tbt_degradation_ratio(),
            );
            let active_actions = self
                .engine_state
                .as_ref()
                .map(|s| s.active_actions.clone())
                .unwrap_or_default();
            let entry = HistoryEntry {
                at_s,
                pressure: [
                    pressure.gpu,
                    pressure.cpu,
                    pressure.memory,
                    pressure.thermal,
                    pressure.latency,
                    pressure.main_app,
                ],
                active_actions,
            };
            if self.history.len() >= 10 {
                self.history.pop_front();
            }
            self.history.push_back(entry);
        }

        // 4. Observation 체크
        self.check_observation();

        // 5. QCF 요청: 압박 있고 cache stale이면 RequestQcf 선발행
        if self.should_request_qcf() {
            self.qcf_pending_at = Some(self.clock.now());
            // 현재 signal 을 저장해 두어야 QcfEstimate 도착 시점에
            // decide() 를 동일한 signal 로 재실행할 수 있다 (2-step handshake).
            self.qcf_pending_signal = Some(signal.clone());
            return Some(EngineDirective {
                seq_id: next_seq_id(),
                commands: vec![EngineCommand::RequestQcf],
            });
        }

        // 6. Lua decide(ctx) 호출 및 directive 조립
        let directive = self.decide_and_build_directive(signal);

        // 7. Relief table 자동 저장 (인터벌 초과 시)
        self.maybe_persist();

        directive
    }

    fn cancel_last_observation(&mut self) {
        self.observations.pop_back();
    }

    fn update_engine_state(&mut self, msg: &EngineMessage) {
        match msg {
            EngineMessage::Heartbeat(status) => {
                // 엔진 heartbeat 의 available_actions 는 빈 리스트 — Capability 로 보고된
                // 값이 `engine_available_actions` 에 이미 저장되어 있으므로 그걸 보존한다.
                let mut merged = status.clone();
                if merged.available_actions.is_empty() && !self.engine_available_actions.is_empty()
                {
                    merged.available_actions = self.engine_available_actions.clone();
                }
                self.engine_state = Some(merged);
            }
            EngineMessage::Capability(cap) => {
                // 엔진이 capability 보고 — 이후 모든 decide() 에서 available 필터로 사용.
                // Capability 는 heartbeat 이전에 한 번 오므로, engine_state 가 None 인
                // 시점에도 ctx.available 을 채우기 위해 별도 필드에 저장한다.
                self.engine_available_actions = cap.available_actions.clone();
                if let Some(ref mut status) = self.engine_state {
                    status.available_actions = cap.available_actions.clone();
                }
            }
            _ => {}
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

    fn initial_relief_snapshot(&self) -> Option<std::collections::HashMap<String, [f32; 6]>> {
        Some(self.relief_table.initial_snapshot())
    }

    fn drain_relief_updates(&mut self) -> Vec<ReliefUpdateEvent> {
        LuaPolicy::drain_relief_updates(self)
    }

    fn observation_overrun_count(&self) -> u64 {
        LuaPolicy::observation_overrun_count(self)
    }

    fn complete_qcf_selection(&mut self, qcf: &QcfEstimate) -> Option<EngineDirective> {
        // pending이 없으면 무시 (중복 응답 방어)
        self.qcf_pending_at.take()?;
        // RequestQcf 를 유발한 signal 도 함께 해제한다. 다음 signal 이 오면
        // 새로 세팅되므로 중복 사용 위험은 없다.
        let pending_signal = self.qcf_pending_signal.take();
        // cache 갱신
        let now = self.clock.now();
        for (name, &cost) in &qcf.estimates {
            self.qcf_cache.insert(name.clone(), (cost, now));
        }
        log::debug!(
            "LuaPolicy QCF cache updated: {} actions",
            self.qcf_cache.len()
        );
        // 2-step handshake 완료: 갱신된 QCF 값을 반영해 즉시 decide() 를 실행하여
        // seq=2 directive 를 발행한다. signal 이 기록되지 않은 경우(예: 외부에서
        // QcfEstimate 가 선행 도착) 는 None 반환 — 다음 process_signal() 이 스코어링.
        let signal = pending_signal?;
        self.decide_and_build_directive(&signal)
    }

    fn check_qcf_timeout(&mut self) -> Option<EngineDirective> {
        const QCF_TIMEOUT_SECS: f64 = 1.0;
        let pending_at = self.qcf_pending_at?;
        let elapsed = self.clock.elapsed_since(pending_at).as_secs_f64();
        if elapsed >= QCF_TIMEOUT_SECS {
            log::warn!(
                "LuaPolicy QCF estimate timeout ({:.1}s) — clearing pending",
                elapsed
            );
            self.qcf_pending_at = None;
            // cache는 비우지 않음: 이전 값 유지 (stale이라도 없는 것보다 낫다)
        }
        None
    }

    /// Lua 스크립트 핫-리로드.
    ///
    /// 새 VM에서 스크립트 로드 + `decide` 함수 존재 검증에 성공한 경우에만
    /// `self.lua`와 `self.script_path`를 교체한다. 실패 시 `self` 변경 없이 Err.
    fn reload_script(&mut self, path: &std::path::Path) -> anyhow::Result<()> {
        // 1. 새 Lua VM 생성 (new()의 초기화 로직 복제)
        // Safety: TABLE | STRING | MATH만 로드, I/O 없음.
        let new_lua = unsafe {
            Lua::unsafe_new_with(
                StdLib::TABLE | StdLib::STRING | StdLib::MATH,
                mlua::LuaOptions::default(),
            )
        };
        let _ = new_lua.set_memory_limit(4 * 1024 * 1024);

        // 2. sys.* 헬퍼 등록
        register_sys_helpers(&new_lua, Arc::clone(&self.gpu_provider))
            .map_err(|e| anyhow::anyhow!("reload: register_sys_helpers failed: {}", e))?;

        // 3. 스크립트 파일 읽기 + 실행
        let script = std::fs::read_to_string(path)
            .map_err(|e| anyhow::anyhow!("reload: read '{}' failed: {}", path.display(), e))?;
        new_lua
            .load(&script)
            .set_name(path.to_string_lossy().as_ref())
            .exec()
            .map_err(|e| anyhow::anyhow!("reload: exec '{}' failed: {}", path.display(), e))?;

        // 4. decide 함수 존재 확인
        let globals = new_lua.globals();
        let decide: Value = globals
            .get("decide")
            .map_err(|e| anyhow::anyhow!("reload: get 'decide' failed: {}", e))?;
        if !decide.is_function() {
            anyhow::bail!(
                "reload: '{}' must define a global `decide(ctx)` function",
                path.display()
            );
        }

        // 5. 모두 성공한 경우에만 self 교체
        log::info!("LuaPolicy reloaded: path={}", path.display());
        self.lua = new_lua;
        self.script_path = path.to_path_buf();
        self.consecutive_errors = 0;
        self.permanent_fallback = false;
        // POLICY_META 로깅 + 필드 갱신 (log_policy_meta가 name/version을 함께 출력)
        self.policy_meta = log_policy_meta(&self.lua);

        Ok(())
    }

    fn script_path(&self) -> Option<&std::path::Path> {
        Some(&self.script_path)
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
        EngineCommand::KvOffload { .. } => "kv_offload",
        EngineCommand::SwitchHw { .. } => "switch_hw",
        EngineCommand::RestoreDefaults => "restore_defaults",
        EngineCommand::Suspend => "suspend",
        EngineCommand::Resume => "resume",
        EngineCommand::SetPartitionRatio { .. } => "set_partition_ratio",
        EngineCommand::SetPrefillPolicy { .. } => "set_prefill_policy",
        EngineCommand::RequestQcf => "request_qcf",
        EngineCommand::PrepareComputeUnit { .. } => "prepare_compute_unit",
        EngineCommand::SwapWeights { .. } => "swap_weights",
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
        "swap_weights" => {
            let ratio: f32 = entry.get("ratio")?;
            // ratio must be in [0.0, 0.9]; clamp and warn on over-limit (ENG-ALG-214-ROUTE)
            let ratio = if ratio > 0.9 {
                log::warn!(
                    "[LuaPolicy] swap_weights ratio {:.4} exceeds 0.9 limit — clamped to 0.9",
                    ratio
                );
                0.9f32
            } else {
                ratio
            };
            let dtype_str: String = entry.get("dtype")?;
            let target_dtype = match dtype_str.as_str() {
                "q4_0" => llm_shared::DtypeTag::Q4_0,
                other => {
                    return Err(mlua::Error::external(format!(
                        "swap_weights: unsupported dtype '{}', only 'q4_0' is valid",
                        other
                    )));
                }
            };
            EngineCommand::SwapWeights {
                ratio,
                target_dtype,
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
///
/// `gpu_provider`는 `sys.gpu_busy()` / `sys.gpu_freq()`의 백엔드로 사용된다.
/// Tegra/Adreno/Mali 플랫폼 차이는 provider 내부에서 흡수된다.
fn register_sys_helpers(lua: &Lua, gpu_provider: SharedGpuProvider) -> LuaResult<()> {
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

    // sys.gpu_busy() -> int (0-100, -1 if unavailable)
    let provider_busy = Arc::clone(&gpu_provider);
    sys.set(
        "gpu_busy",
        lua.create_function(move |_, ()| -> LuaResult<i64> {
            let pct = provider_busy
                .lock()
                .ok()
                .and_then(|mut g| g.util_pct())
                .map(|v| v.round() as i64)
                .unwrap_or(-1);
            Ok(pct)
        })?,
    )?;

    // sys.gpu_freq() -> int (Hz, -1 if unavailable)
    let provider_freq = Arc::clone(&gpu_provider);
    sys.set(
        "gpu_freq",
        lua.create_function(move |_, ()| -> LuaResult<i64> {
            let freq = provider_freq
                .lock()
                .ok()
                .and_then(|mut g| g.freq_hz())
                .map(|v| v as i64)
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

    /// 테스트용 더미 SystemSignal (Lua parse 테스트에서 signal 내용은 무관).
    fn dummy_signal() -> SystemSignal {
        SystemSignal::MemoryPressure {
            level: llm_shared::Level::Normal,
            available_bytes: 4 * 1024 * 1024 * 1024,
            total_bytes: 8 * 1024 * 1024 * 1024,
            reclaim_target_bytes: 0,
        }
    }

    #[test]
    fn test_empty_decide() {
        let script = create_temp_script("function decide(ctx) return {} end");
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
        assert!(cmds.is_empty());
    }

    #[test]
    fn test_nil_decide() {
        let script = create_temp_script("function decide(ctx) return nil end");
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
        assert!(cmds.is_empty());
    }

    #[test]
    fn test_throttle_action() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{type = "throttle", delay_ms = 50}}
            end"#,
        );
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let cmds = policy.call_decide(&dummy_signal()).0;
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

        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let cmds = policy.call_decide(&dummy_signal()).0;
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

        let cmds = policy.call_decide(&dummy_signal()).0;
        assert_eq!(cmds.len(), 1);
        assert!(matches!(cmds[0], EngineCommand::RestoreDefaults));
    }

    #[test]
    fn test_lua_error_returns_fallback() {
        // Lua 오류 시 fallback_decide()가 호출되어 was_fallback=true + non-empty commands 반환.
        let script = create_temp_script(
            r#"function decide(ctx)
                error("intentional error")
            end"#,
        );
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let (cmds, was_fallback) = policy.call_decide(&dummy_signal());
        // dummy_signal()은 Level::Normal MemoryPressure → RestoreDefaults
        assert!(was_fallback, "should be fallback on Lua error");
        assert_eq!(cmds.len(), 1);
        assert!(
            matches!(cmds[0], EngineCommand::RestoreDefaults),
            "Level::Normal fallback should yield RestoreDefaults"
        );
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();

        // First two calls return empty
        assert!(policy.call_decide(&dummy_signal()).0.is_empty());
        assert!(policy.call_decide(&dummy_signal()).0.is_empty());

        // Third call triggers throttle
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let cmds = policy.call_decide(&dummy_signal()).0;
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

        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
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
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
        assert_eq!(cmds.len(), 1);
    }

    #[test]
    fn ewma_first_observation_replaces_default() {
        // defaults가 없는 경우: entry 초기값 = [0.0; 6].
        // 첫 관측 후 값 = alpha * 0.0 + (1-alpha) * observed = 0.125 * observed
        let mut table = EwmaReliefTable::new(0.875, HashMap::new());
        let observed = [0.6, -0.2, 0.0, 0.4, -0.1, 0.0];
        table.observe("switch_hw", &observed);
        let predicted = table.predict("switch_hw");
        let expected: [f32; 6] = observed.map(|v| 0.125 * v);
        for (i, (&p, &e)) in predicted.iter().zip(expected.iter()).enumerate() {
            assert!((p - e).abs() < 1e-6, "dim {}: expected {}, got {}", i, e, p);
        }
        assert_eq!(table.observation_count("switch_hw"), 1);
    }

    #[test]
    fn ewma_first_observation_blends_with_prior() {
        // defaults가 있는 경우: entry 초기값 = default.
        // 첫 관측 후 값 = alpha * default + (1-alpha) * observed
        let mut defaults = HashMap::new();
        defaults.insert(
            "switch_hw".to_string(),
            vec![0.5f32, -0.3, 0.0, 0.3, -0.1, 0.0],
        );
        let mut table = EwmaReliefTable::new(0.875, defaults);
        let observed = [0.6f32, -0.2, 0.0, 0.4, -0.1, 0.0];
        let default = [0.5f32, -0.3, 0.0, 0.3, -0.1, 0.0];
        table.observe("switch_hw", &observed);
        let predicted = table.predict("switch_hw");
        for i in 0..6 {
            let expected = 0.875 * default[i] + 0.125 * observed[i];
            assert!(
                (predicted[i] - expected).abs() < 1e-5,
                "dim {}: expected {}, got {}",
                i,
                expected,
                predicted[i]
            );
        }
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

        // qcf_penalty_weight=0.0 으로 QCF 요청 로직 비활성화 (이 테스트는 Lua 결정 경로만 검증)
        let config = AdaptationConfig {
            qcf_penalty_weight: 0.0,
            ..AdaptationConfig::default()
        };
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

    #[test]
    fn ctx_is_joint_valid_returns_true_for_non_excluded_pair() {
        let script = r#"
            function decide(ctx)
                -- throttle + layer_skip는 배타 그룹에 없으므로 valid
                assert(ctx.is_joint_valid({"throttle", "layer_skip"}) == true,
                    "throttle+layer_skip should be valid")
                return {}
            end
        "#;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("test.lua");
        std::fs::write(&script_path, script).unwrap();

        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        groups.insert(
            "kv_quality".to_string(),
            vec![
                "kv_evict_sliding".to_string(),
                "kv_evict_h2o".to_string(),
                "kv_merge_d2o".to_string(),
                "kv_quant_dynamic".to_string(),
            ],
        );

        let policy = LuaPolicy::new_with_exclusions(
            script_path.to_str().unwrap(),
            AdaptationConfig::default(),
            Arc::new(SystemClock::new()),
            groups,
        );
        let mut policy = policy.unwrap();
        let result = policy.process_signal(&dummy_signal());
        // decide()에서 assert 통과 후 {} 반환 → None
        assert!(result.is_none());
    }

    #[test]
    fn ctx_is_joint_valid_returns_false_for_excluded_pair() {
        let script = r#"
            function decide(ctx)
                -- kv_evict_sliding + kv_quant_dynamic은 kv_quality 배타 그룹
                assert(ctx.is_joint_valid({"kv_evict_sliding", "kv_quant_dynamic"}) == false,
                    "kv_evict_sliding+kv_quant_dynamic should be invalid")
                return {}
            end
        "#;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("test2.lua");
        std::fs::write(&script_path, script).unwrap();

        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        groups.insert(
            "kv_quality".to_string(),
            vec![
                "kv_evict_sliding".to_string(),
                "kv_evict_h2o".to_string(),
                "kv_merge_d2o".to_string(),
                "kv_quant_dynamic".to_string(),
            ],
        );

        let policy = LuaPolicy::new_with_exclusions(
            script_path.to_str().unwrap(),
            AdaptationConfig::default(),
            Arc::new(SystemClock::new()),
            groups,
        );
        let mut policy = policy.unwrap();
        let result = policy.process_signal(&dummy_signal());
        assert!(result.is_none());
    }

    #[test]
    fn ctx_is_joint_valid_single_action_always_valid() {
        let script = r#"
            function decide(ctx)
                assert(ctx.is_joint_valid({"kv_evict_sliding"}) == true,
                    "single action should always be valid")
                assert(ctx.is_joint_valid({}) == true,
                    "empty list should be valid")
                return {}
            end
        "#;

        let dir = tempfile::tempdir().unwrap();
        let script_path = dir.path().join("test3.lua");
        std::fs::write(&script_path, script).unwrap();

        let mut groups: HashMap<String, Vec<String>> = HashMap::new();
        groups.insert(
            "kv_quality".to_string(),
            vec![
                "kv_evict_sliding".to_string(),
                "kv_quant_dynamic".to_string(),
            ],
        );

        let mut policy = LuaPolicy::new_with_exclusions(
            script_path.to_str().unwrap(),
            AdaptationConfig::default(),
            Arc::new(SystemClock::new()),
            groups,
        )
        .unwrap();
        let result = policy.process_signal(&dummy_signal());
        assert!(result.is_none());
    }

    #[test]
    fn linucb_cold_start_returns_positive_bonus() {
        let table = LinUcbTable::new();
        let phi = [1.0f32; LINUCB_FEATURE_DIM];
        // P matrix 없으면 1.0 반환
        assert_eq!(table.ucb_bonus("kv_evict_sliding", &phi), 1.0);
    }

    #[test]
    fn linucb_bonus_decreases_after_update() {
        let mut table = LinUcbTable::new();
        let phi = [0.5f32; LINUCB_FEATURE_DIM];
        // 첫 번째: P 없으므로 1.0
        assert_eq!(table.ucb_bonus("throttle", &phi), 1.0);
        // update 후: P가 초기화되고 해당 방향 불확실성 감소
        table.update("throttle", &phi);
        let bonus_after = table.ucb_bonus("throttle", &phi);
        // identity P의 초기 bonus < 1.0이 되어야 함
        assert!(
            bonus_after < 1.0,
            "bonus after update should decrease: {}",
            bonus_after
        );
        // 동일한 phi로 반복 업데이트하면 계속 감소
        for _ in 0..5 {
            table.update("throttle", &phi);
        }
        let bonus_later = table.ucb_bonus("throttle", &phi);
        assert!(
            bonus_later <= bonus_after,
            "bonus should keep decreasing: {} vs {}",
            bonus_later,
            bonus_after
        );
    }

    #[test]
    fn linucb_actions_are_independent() {
        let mut table = LinUcbTable::new();
        let phi = [0.3f32; LINUCB_FEATURE_DIM];
        // action A 10회 업데이트
        for _ in 0..10 {
            table.update("kv_evict_sliding", &phi);
        }
        // action B는 cold-start → 1.0
        assert_eq!(table.ucb_bonus("throttle", &phi), 1.0);
    }

    // ── QCF cache 관련 테스트 ───────────────────────────────────────────

    fn make_policy_with_system_clock() -> LuaPolicy {
        let script = create_temp_script("function decide(ctx) return {} end");
        LuaPolicy::with_system_clock(script.path().to_str().unwrap(), AdaptationConfig::default())
            .unwrap()
    }

    #[test]
    fn qcf_cache_is_stale_when_empty() {
        let policy = make_policy_with_system_clock();
        // 빈 cache는 stale
        assert!(policy.qcf_cache_is_stale());
    }

    #[test]
    fn complete_qcf_selection_populates_cache() {
        let mut policy = make_policy_with_system_clock();
        // pending 없으면 cache 갱신도 없이 None 즉시 반환
        let qcf = QcfEstimate {
            estimates: [("kv_evict_sliding".to_string(), 0.3f32)]
                .into_iter()
                .collect(),
            layer_swap: None,
        };
        let result = policy.complete_qcf_selection(&qcf);
        assert!(result.is_none(), "pending 없으면 None을 반환해야 한다");
        assert!(
            policy.qcf_cache.is_empty(),
            "pending 없으면 cache 갱신하지 않아야 한다"
        );

        // pending_at 만 설정하고 pending_signal 없는 legacy 경로 — cache 는 갱신되지만
        // decide() 는 실행되지 않아 None 반환.
        policy.qcf_pending_at = Some(policy.clock.now());
        let result2 = policy.complete_qcf_selection(&qcf);
        assert!(
            result2.is_none(),
            "pending_signal 없으면 cache만 갱신되고 None을 반환해야 한다"
        );
        assert!(
            policy.qcf_pending_at.is_none(),
            "pending_at이 소비되어야 한다"
        );
        assert_eq!(policy.qcf_cache.len(), 1);
        let (cost, _) = policy.qcf_cache["kv_evict_sliding"];
        assert!((cost - 0.3).abs() < f32::EPSILON);
    }

    /// 2-step handshake 회귀 테스트: `process_signal` 이 Critical 에서 RequestQcf 를
    /// 내고 `pending_signal` 을 기록한다. 이어서 `complete_qcf_selection` 이 도착하면
    /// 저장된 signal 로 decide() 를 재실행하여 seq=2 EngineDirective (비어있지 않은
    /// commands) 를 반환해야 한다. 이 경로가 깨져 있으면 signal 경로가 seq=1 에서 멈춘다.
    #[test]
    fn complete_qcf_selection_emits_directive_after_2step_handshake() {
        use crate::pipeline::PolicyStrategy;

        let script = r#"
            POLICY_META = { name = "test_2step", version = "1.0" }
            function decide(ctx)
                if ctx.coef.trigger.mem_low then
                    return {{ type = "kv_evict_sliding", keep_ratio = 0.5 }}
                end
                return {}
            end
        "#;
        let script_file = create_temp_script(script);
        let mut policy = LuaPolicy::new(
            script_file.path().to_str().unwrap(),
            AdaptationConfig::default(),
            Arc::new(crate::clock::SystemClock::new()),
        )
        .expect("policy load");

        // (1) Critical MemoryPressure 주입 → RequestQcf directive 발행 + pending_signal 기록
        let signal = SystemSignal::MemoryPressure {
            level: llm_shared::Level::Critical,
            available_bytes: 30_000_000,
            total_bytes: 8_000_000_000,
            reclaim_target_bytes: 100_000_000,
        };
        let d1 = policy.process_signal(&signal).expect("seq=1 directive");
        assert_eq!(
            d1.commands.len(),
            1,
            "첫 directive 는 RequestQcf 1건이어야 한다"
        );
        assert!(
            matches!(d1.commands[0], EngineCommand::RequestQcf),
            "첫 command 는 RequestQcf 여야 한다, got {:?}",
            d1.commands[0]
        );
        assert!(
            policy.qcf_pending_signal.is_some(),
            "pending_signal 이 기록되어야 한다"
        );

        // (2) 엔진이 QcfEstimate 를 반환 → complete_qcf_selection 이 seq=2 directive 발행
        let qcf = QcfEstimate {
            estimates: [("kv_evict_sliding".to_string(), 0.25f32)]
                .into_iter()
                .collect(),
            layer_swap: None,
        };
        let d2 = policy
            .complete_qcf_selection(&qcf)
            .expect("2-step handshake 에서 seq=2 directive 발행 필요");
        assert!(
            !d2.commands.is_empty(),
            "decide() 가 비지 않은 commands 를 반환해야 한다"
        );
        assert!(
            matches!(d2.commands[0], EngineCommand::KvEvictSliding { .. }),
            "Lua policy 가 kv_evict_sliding 을 선택해야 한다, got {:?}",
            d2.commands[0]
        );
        // pending 상태가 완전 해제되어야 한다.
        assert!(policy.qcf_pending_at.is_none());
        assert!(policy.qcf_pending_signal.is_none());
    }

    /// 결정론적 시간 제어를 위한 테스트 전용 Clock.
    struct FixedClock {
        current: std::sync::atomic::AtomicU64,
    }

    impl FixedClock {
        fn new(millis: u64) -> Arc<Self> {
            Arc::new(Self {
                current: std::sync::atomic::AtomicU64::new(millis),
            })
        }

        fn advance(&self, millis: u64) {
            self.current
                .fetch_add(millis, std::sync::atomic::Ordering::Relaxed);
        }
    }

    impl Clock for FixedClock {
        fn now(&self) -> LogicalInstant {
            let ms = self.current.load(std::sync::atomic::Ordering::Relaxed);
            LogicalInstant::from_duration_since_start(std::time::Duration::from_millis(ms))
        }
    }

    #[test]
    fn check_qcf_timeout_clears_pending_after_1s() {
        let clock = FixedClock::new(0);
        let script = create_temp_script("function decide(ctx) return {} end");
        let mut policy = LuaPolicy::new(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
            clock.clone() as Arc<dyn Clock>,
        )
        .unwrap();

        // pending 없으면 no-op
        assert!(policy.check_qcf_timeout().is_none());

        // t=0에서 pending 설정
        policy.qcf_pending_at = Some(policy.clock.now());

        // t=500ms: 아직 timeout 아님
        clock.advance(500);
        let result = policy.check_qcf_timeout();
        assert!(result.is_none());
        assert!(
            policy.qcf_pending_at.is_some(),
            "500ms 경과: 아직 pending 유지되어야 한다"
        );

        // t=1100ms: 1초 초과 → pending 소거
        clock.advance(600);
        let result2 = policy.check_qcf_timeout();
        assert!(result2.is_none());
        assert!(
            policy.qcf_pending_at.is_none(),
            "1초 경과 후 pending_at이 소거되어야 한다"
        );
    }

    // ── SwapWeights Lua binding tests (WSWAP-3-LUA) ──────────────────────────

    /// Lua policy that emits swap_weights on Emergency memory pressure.
    #[test]
    fn test_lua_policy_emits_swap_weights_on_emergency() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{
                    type = "swap_weights",
                    ratio = 0.50,
                    dtype = "q4_0",
                }}
            end"#,
        );
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::SwapWeights {
                ratio,
                target_dtype,
            } => {
                assert!(
                    (*ratio - 0.50).abs() < 1e-6,
                    "ratio should be 0.50, got {ratio}"
                );
                assert_eq!(*target_dtype, llm_shared::DtypeTag::Q4_0);
            }
            other => panic!("Expected SwapWeights, got {other:?}"),
        }
    }

    /// ratio > 0.9 is clamped to 0.9 (ENG-ALG-214-ROUTE upper limit).
    #[test]
    fn test_lua_policy_clamps_ratio_above_limit() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{
                    type = "swap_weights",
                    ratio = 0.95,
                    dtype = "q4_0",
                }}
            end"#,
        );
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        let cmds = policy.call_decide(&dummy_signal()).0;
        assert_eq!(cmds.len(), 1);
        match &cmds[0] {
            EngineCommand::SwapWeights { ratio, .. } => {
                assert!(
                    (*ratio - 0.9).abs() < 1e-6,
                    "ratio should be clamped to 0.9, got {ratio}"
                );
            }
            other => panic!("Expected SwapWeights, got {other:?}"),
        }
    }

    /// Invalid dtype string must cause the action to be skipped (error, not panic).
    #[test]
    fn test_lua_policy_rejects_invalid_dtype_string() {
        let script = create_temp_script(
            r#"function decide(ctx)
                return {{
                    type = "swap_weights",
                    ratio = 0.50,
                    dtype = "f32",
                }}
            end"#,
        );
        let mut policy = LuaPolicy::with_system_clock(
            script.path().to_str().unwrap(),
            AdaptationConfig::default(),
        )
        .unwrap();
        // parse_single_action returns Err for unsupported dtype → action skipped, no panic
        let cmds = policy.call_decide(&dummy_signal()).0;
        assert!(
            cmds.is_empty(),
            "invalid dtype must produce no commands (got {cmds:?})"
        );
    }

    #[test]
    fn should_request_qcf_false_when_no_pressure() {
        let policy = make_policy_with_system_clock();
        // signal_state 기본값: mem_total=0, cpu_pct=0, temp_mc=0 → 압박 없음
        assert!(!policy.should_request_qcf());
    }

    #[test]
    fn should_request_qcf_true_when_mem_high_and_cache_empty() {
        let mut policy = make_policy_with_system_clock();
        // 메모리 압박 >= 0.6 설정: total=100, available=30 → used=0.7
        policy.signal_state.update_memory(30, 100);
        assert!(policy.should_request_qcf());
    }

    #[test]
    fn should_request_qcf_false_when_pending_exists() {
        let mut policy = make_policy_with_system_clock();
        policy.signal_state.update_memory(30, 100);
        // pending 설정 → false
        policy.qcf_pending_at = Some(policy.clock.now());
        assert!(!policy.should_request_qcf());
    }
}
