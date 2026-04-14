//! Trajectory: 시뮬레이션 실행 중 발생한 이벤트 + 상태 스냅샷 기록.
//!
//! - `record_*` 메서드로 각 이벤트를 추가한다.
//! - `dump_json`, `dump_csv_states`로 파일에 저장할 수 있다.
//! - assertion 헬퍼로 통합 테스트에서 기대 동작을 검증한다.

#![allow(dead_code)]

use std::collections::BTreeMap;
use std::path::Path;
use std::time::Duration;

use serde::Serialize;

use super::state::{EngineStateModel, PhysicalState};

// ─────────────────────────────────────────────────────────
// Dump 타입 (serde-직렬화 가능한 미러 타입)
// ─────────────────────────────────────────────────────────

/// PhysicalState의 직렬화 가능 스냅샷.
#[derive(Debug, Clone, Serialize)]
pub struct PhysicalStateSnapshot {
    pub kv_cache_bytes: f64,
    pub kv_cache_tokens: f64,
    pub kv_dtype: String,
    pub device_memory_used_mb: f64,
    pub device_memory_total_mb: f64,
    pub memory_bw_utilization_pct: f64,
    pub engine_cpu_pct: f64,
    pub engine_gpu_pct: f64,
    pub cpu_freq_mhz: f64,
    pub gpu_freq_mhz: f64,
    pub thermal_c: f64,
    pub cpu_cluster_thermal_c: f64,
    pub gpu_cluster_thermal_c: f64,
    pub throughput_tps: f64,
    pub tbt_ms: f64,
}

impl PhysicalStateSnapshot {
    pub fn from_state(s: &PhysicalState) -> Self {
        Self {
            kv_cache_bytes: s.kv_cache_bytes,
            kv_cache_tokens: s.kv_cache_tokens,
            kv_dtype: s.kv_dtype.clone(),
            device_memory_used_mb: s.device_memory_used_mb,
            device_memory_total_mb: s.device_memory_total_mb,
            memory_bw_utilization_pct: s.memory_bw_utilization_pct,
            engine_cpu_pct: s.engine_cpu_pct,
            engine_gpu_pct: s.engine_gpu_pct,
            cpu_freq_mhz: s.cpu_freq_mhz,
            gpu_freq_mhz: s.gpu_freq_mhz,
            thermal_c: s.thermal_c,
            cpu_cluster_thermal_c: s.cpu_cluster_thermal_c,
            gpu_cluster_thermal_c: s.gpu_cluster_thermal_c,
            throughput_tps: s.throughput_tps,
            tbt_ms: s.tbt_ms,
        }
    }
}

/// EngineStateModel의 직렬화 가능 스냅샷.
#[derive(Debug, Clone, Serialize)]
pub struct EngineStateSnapshot {
    pub active_device: String,
    pub active_actions: Vec<String>,
    pub phase: String,
    pub partition_ratio: f64,
    pub throttle_delay_ms: f64,
    pub skip_ratio: f64,
}

impl EngineStateSnapshot {
    pub fn from_engine(e: &EngineStateModel) -> Self {
        Self {
            active_device: e.active_device.clone(),
            active_actions: e.active_actions.clone(),
            phase: e.phase.clone(),
            partition_ratio: e.partition_ratio,
            throttle_delay_ms: e.throttle_delay_ms,
            skip_ratio: e.skip_ratio,
        }
    }
}

/// EngineMessage 요약 덤프.
#[derive(Debug, Clone, Serialize)]
pub struct EngineMessageDump {
    pub kind: String,
    pub actual_throughput: Option<f32>,
    pub memory_level: Option<String>,
    pub compute_level: Option<String>,
    pub kv_cache_tokens: Option<usize>,
}

impl EngineMessageDump {
    pub fn from_msg(msg: &llm_shared::EngineMessage) -> Self {
        match msg {
            llm_shared::EngineMessage::Heartbeat(status) => Self {
                kind: "heartbeat".to_string(),
                actual_throughput: Some(status.actual_throughput),
                memory_level: Some(format!("{:?}", status.memory_level)),
                compute_level: Some(format!("{:?}", status.compute_level)),
                kv_cache_tokens: Some(status.kv_cache_tokens),
            },
            _ => Self {
                kind: "other".to_string(),
                actual_throughput: None,
                memory_level: None,
                compute_level: None,
                kv_cache_tokens: None,
            },
        }
    }
}

/// SystemSignal 요약 덤프.
#[derive(Debug, Clone, Serialize)]
pub struct SignalDump {
    pub kind: String,
    pub level: String,
    pub detail: serde_json::Value,
}

impl SignalDump {
    pub fn from_signal(sig: &llm_shared::SystemSignal) -> Self {
        match sig {
            llm_shared::SystemSignal::MemoryPressure {
                level,
                available_bytes,
                total_bytes,
                reclaim_target_bytes,
            } => Self {
                kind: "memory_pressure".to_string(),
                level: format!("{:?}", level),
                detail: serde_json::json!({
                    "available_bytes": available_bytes,
                    "total_bytes": total_bytes,
                    "reclaim_target_bytes": reclaim_target_bytes,
                }),
            },
            llm_shared::SystemSignal::ComputeGuidance {
                level,
                cpu_usage_pct,
                gpu_usage_pct,
                ..
            } => Self {
                kind: "compute_guidance".to_string(),
                level: format!("{:?}", level),
                detail: serde_json::json!({
                    "cpu_usage_pct": cpu_usage_pct,
                    "gpu_usage_pct": gpu_usage_pct,
                }),
            },
            llm_shared::SystemSignal::ThermalAlert {
                level,
                temperature_mc,
                throttling_active,
                ..
            } => Self {
                kind: "thermal_alert".to_string(),
                level: format!("{:?}", level),
                detail: serde_json::json!({
                    "temperature_mc": temperature_mc,
                    "throttling_active": throttling_active,
                }),
            },
            llm_shared::SystemSignal::EnergyConstraint {
                level,
                power_budget_mw,
                ..
            } => Self {
                kind: "energy_constraint".to_string(),
                level: format!("{:?}", level),
                detail: serde_json::json!({
                    "power_budget_mw": power_budget_mw,
                }),
            },
        }
    }
}

/// EngineDirective 요약 덤프.
#[derive(Debug, Clone, Serialize)]
pub struct DirectiveDump {
    pub seq_id: u64,
    pub commands: Vec<String>,
}

impl DirectiveDump {
    pub fn from_directive(dir: &llm_shared::EngineDirective) -> Self {
        Self {
            seq_id: dir.seq_id,
            commands: dir.commands.iter().map(|c| format!("{:?}", c)).collect(),
        }
    }
}

// ─────────────────────────────────────────────────────────
// TrajectoryEntry
// ─────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum TrajectoryEntry {
    StateSnapshot {
        at_s: f64,
        state: PhysicalStateSnapshot,
        engine: EngineStateSnapshot,
    },
    Heartbeat {
        at_s: f64,
        message: EngineMessageDump,
    },
    Signal {
        at_s: f64,
        signal: SignalDump,
    },
    Directive {
        at_s: f64,
        trigger: SignalDump,
        directive: DirectiveDump,
    },
    ObservationDue {
        at_s: f64,
        action: String,
        recorded_at_s: f64,
    },
    InjectionEvent {
        at_s: f64,
        idx: usize,
        started: bool,
    },
    Custom {
        at_s: f64,
        name: String,
    },
}

// ─────────────────────────────────────────────────────────
// Trajectory
// ─────────────────────────────────────────────────────────

/// 시뮬레이션 전체 이력(trajectory)을 기록한다.
#[derive(Debug, Default, Serialize)]
pub struct Trajectory {
    pub entries: Vec<TrajectoryEntry>,
}

impl Trajectory {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record_state_snapshot(&mut self, at: Duration, s: &PhysicalState, e: &EngineStateModel) {
        self.entries.push(TrajectoryEntry::StateSnapshot {
            at_s: at.as_secs_f64(),
            state: PhysicalStateSnapshot::from_state(s),
            engine: EngineStateSnapshot::from_engine(e),
        });
    }

    pub fn record_heartbeat(&mut self, at: Duration, msg: &llm_shared::EngineMessage) {
        self.entries.push(TrajectoryEntry::Heartbeat {
            at_s: at.as_secs_f64(),
            message: EngineMessageDump::from_msg(msg),
        });
    }

    pub fn record_signal(&mut self, at: Duration, sig: &llm_shared::SystemSignal) {
        self.entries.push(TrajectoryEntry::Signal {
            at_s: at.as_secs_f64(),
            signal: SignalDump::from_signal(sig),
        });
    }

    pub fn record_directive(
        &mut self,
        at: Duration,
        trigger: &llm_shared::SystemSignal,
        dir: &llm_shared::EngineDirective,
    ) {
        self.entries.push(TrajectoryEntry::Directive {
            at_s: at.as_secs_f64(),
            trigger: SignalDump::from_signal(trigger),
            directive: DirectiveDump::from_directive(dir),
        });
    }

    pub fn record_observation_due(&mut self, at: Duration, action: &str, recorded_at: Duration) {
        self.entries.push(TrajectoryEntry::ObservationDue {
            at_s: at.as_secs_f64(),
            action: action.to_string(),
            recorded_at_s: recorded_at.as_secs_f64(),
        });
    }

    pub fn record_injection_event(&mut self, at: Duration, idx: usize, started: bool) {
        self.entries.push(TrajectoryEntry::InjectionEvent {
            at_s: at.as_secs_f64(),
            idx,
            started,
        });
    }

    pub fn record_custom(&mut self, at: Duration, name: &str) {
        self.entries.push(TrajectoryEntry::Custom {
            at_s: at.as_secs_f64(),
            name: name.to_string(),
        });
    }

    // ─────────────────────────────────────────────────────
    // 파일 덤프
    // ─────────────────────────────────────────────────────

    /// JSON 형식으로 파일에 저장한다.
    pub fn dump_json<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let json = serde_json::to_string_pretty(self).map_err(std::io::Error::other)?;
        std::fs::write(path, json)
    }

    /// StateSnapshot 항목만 CSV로 저장한다.
    pub fn dump_csv_states<P: AsRef<Path>>(&self, path: P) -> std::io::Result<()> {
        let mut lines = vec![
            "at_s,kv_cache_bytes,kv_cache_tokens,device_memory_used_mb,\
             engine_cpu_pct,engine_gpu_pct,cpu_freq_mhz,gpu_freq_mhz,\
             thermal_c,throughput_tps,tbt_ms"
                .to_string(),
        ];
        for entry in &self.entries {
            if let TrajectoryEntry::StateSnapshot { at_s, state, .. } = entry {
                lines.push(format!(
                    "{},{},{},{},{},{},{},{},{},{},{}",
                    at_s,
                    state.kv_cache_bytes,
                    state.kv_cache_tokens,
                    state.device_memory_used_mb,
                    state.engine_cpu_pct,
                    state.engine_gpu_pct,
                    state.cpu_freq_mhz,
                    state.gpu_freq_mhz,
                    state.thermal_c,
                    state.throughput_tps,
                    state.tbt_ms,
                ));
            }
        }
        std::fs::write(path, lines.join("\n"))
    }

    // ─────────────────────────────────────────────────────
    // Assertion 헬퍼
    // ─────────────────────────────────────────────────────

    /// Directive 총 개수.
    pub fn directive_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, TrajectoryEntry::Directive { .. }))
            .count()
    }

    /// Signal matcher에 부합하는 Directive 목록 반환.
    pub fn directives_for_signal<F>(&self, matcher: F) -> Vec<&DirectiveDump>
    where
        F: Fn(&SignalDump) -> bool,
    {
        self.entries
            .iter()
            .filter_map(|e| {
                if let TrajectoryEntry::Directive {
                    trigger, directive, ..
                } = e
                {
                    matcher(trigger).then_some(directive)
                } else {
                    None
                }
            })
            .collect()
    }

    /// t_s 이후 첫 번째 Directive의 시각을 반환한다.
    pub fn first_directive_at_or_after(&self, t_s: f64) -> Option<f64> {
        self.entries
            .iter()
            .filter_map(|e| {
                if let TrajectoryEntry::Directive { at_s, .. } = e {
                    (*at_s >= t_s).then_some(*at_s)
                } else {
                    None
                }
            })
            .next()
    }

    /// 지정 시각에 가장 가까운 StateSnapshot에서 proj를 적용한 값을 반환한다.
    pub fn state_at<F: Fn(&PhysicalStateSnapshot) -> f64>(&self, proj: F, t_s: f64) -> Option<f64> {
        let mut best: Option<(f64, f64)> = None; // (dist, value)
        for entry in &self.entries {
            if let TrajectoryEntry::StateSnapshot { at_s, state, .. } = entry {
                let dist = (at_s - t_s).abs();
                if best.is_none() || dist < best.unwrap().0 {
                    best = Some((dist, proj(state)));
                }
            }
        }
        best.map(|(_, v)| v)
    }

    /// Heartbeat 총 개수.
    pub fn heartbeat_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, TrajectoryEntry::Heartbeat { .. }))
            .count()
    }

    /// Signal 총 개수 (종류 무관).
    pub fn signal_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, TrajectoryEntry::Signal { .. }))
            .count()
    }

    /// 지정 kind 문자열을 포함하는 Signal 개수.
    pub fn signal_count_by_kind(&self, kind: &str) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, TrajectoryEntry::Signal { signal, .. } if signal.kind == kind))
            .count()
    }

    /// ObservationDue 총 개수.
    pub fn observation_due_count(&self) -> usize {
        self.entries
            .iter()
            .filter(|e| matches!(e, TrajectoryEntry::ObservationDue { .. }))
            .count()
    }

    /// 지정 action 이름에 해당하는 ObservationDue 개수.
    pub fn observation_due_count_for(&self, action: &str) -> usize {
        self.entries
            .iter()
            .filter(
                |e| matches!(e, TrajectoryEntry::ObservationDue { action: a, .. } if a == action),
            )
            .count()
    }

    /// 특정 command 키워드가 포함된 Directive가 있으면 Ok, 없으면 Err.
    pub fn assert_contains_directive_kind(&self, kind: &str) -> Result<(), String> {
        for entry in &self.entries {
            if let TrajectoryEntry::Directive { directive, .. } = entry
                && directive.commands.iter().any(|c| c.contains(kind))
            {
                return Ok(());
            }
        }
        Err(format!(
            "no directive containing {:?} found in trajectory",
            kind
        ))
    }

    /// trajectory의 마지막 시각을 반환한다.
    pub fn last_at_s(&self) -> Option<f64> {
        let entry_at_s = |e: &TrajectoryEntry| -> f64 {
            match e {
                TrajectoryEntry::StateSnapshot { at_s, .. } => *at_s,
                TrajectoryEntry::Heartbeat { at_s, .. } => *at_s,
                TrajectoryEntry::Signal { at_s, .. } => *at_s,
                TrajectoryEntry::Directive { at_s, .. } => *at_s,
                TrajectoryEntry::ObservationDue { at_s, .. } => *at_s,
                TrajectoryEntry::InjectionEvent { at_s, .. } => *at_s,
                TrajectoryEntry::Custom { at_s, .. } => *at_s,
            }
        };
        self.entries.iter().rev().map(entry_at_s).next()
    }

    // ─────────────────────────────────────────────────────
    // ASCII 타임라인 포매터
    // ─────────────────────────────────────────────────────

    /// 모든 이벤트를 포함한 ASCII 타임라인을 문자열로 렌더.
    ///
    /// 실패한 테스트의 원인 추적이나 `SIM_TIMELINE=1 cargo test -- --nocapture`에서
    /// 시뮬레이터 동작을 한눈에 보기 위한 용도. State snapshot까지 전부 포함하므로
    /// 30s 시나리오(tick 50ms)에서 ~600줄이 나온다. 짧게 보려면
    /// [`format_timeline_compact`](Self::format_timeline_compact)를 사용한다.
    pub fn format_timeline(&self) -> String {
        self.format_timeline_inner(TimelineMode::Full)
    }

    /// State snapshot을 1초 간격으로 샘플링하고 heartbeat를 생략한 간결 타임라인.
    ///
    /// 시그널/지시/주입/관측 같은 구조적 이벤트는 모두 유지. 상태 변화의 대략적인
    /// 흐름만 확인하면 충분할 때 사용.
    pub fn format_timeline_compact(&self) -> String {
        self.format_timeline_inner(TimelineMode::Compact)
    }

    /// `SIM_TIMELINE` 환경변수가 설정된 경우 타임라인을 stderr로 출력.
    ///
    /// - `SIM_TIMELINE=1` 또는 `full`: [`format_timeline`](Self::format_timeline)
    /// - `SIM_TIMELINE=compact`: [`format_timeline_compact`](Self::format_timeline_compact)
    /// - 미설정: no-op
    ///
    /// 테스트 본문 끝에서 호출하면 `cargo test -- --nocapture`로 필요할 때만
    /// 타임라인을 확인할 수 있다.
    pub fn print_timeline_if_enabled(&self) {
        match std::env::var("SIM_TIMELINE").ok().as_deref() {
            Some("1") | Some("full") => eprintln!("{}", self.format_timeline()),
            Some("compact") => eprintln!("{}", self.format_timeline_compact()),
            _ => {}
        }
    }

    fn format_timeline_inner(&self, mode: TimelineMode) -> String {
        use std::fmt::Write;
        let mut out = String::new();

        let duration = self.last_at_s().unwrap_or(0.0);
        writeln!(out, "━━━ Simulation Timeline ━━━").ok();
        writeln!(
            out,
            "duration={:.2}s  heartbeats={}  signals={}  directives={}  obs={}",
            duration,
            self.heartbeat_count(),
            self.signal_count(),
            self.directive_count(),
            self.observation_due_count(),
        )
        .ok();
        writeln!(out).ok();

        let mut last_state_sec: Option<i64> = None;
        for entry in &self.entries {
            match entry {
                TrajectoryEntry::StateSnapshot {
                    at_s,
                    state,
                    engine,
                } => {
                    if matches!(mode, TimelineMode::Compact) {
                        let sec = at_s.floor() as i64;
                        if last_state_sec == Some(sec) {
                            continue;
                        }
                        last_state_sec = Some(sec);
                    }
                    writeln!(
                        out,
                        "t={:7.2}s  [STATE]  kv={:6.2}MiB  mem={:6.1}MB  cpu={:5.1}%  gpu={:5.1}%  therm={:5.1}°C  tps={:6.2}  dev={}  phase={}  actions={:?}",
                        at_s,
                        state.kv_cache_bytes / (1024.0 * 1024.0),
                        state.device_memory_used_mb,
                        state.engine_cpu_pct,
                        state.engine_gpu_pct,
                        state.thermal_c,
                        state.throughput_tps,
                        engine.active_device,
                        engine.phase,
                        engine.active_actions,
                    )
                    .ok();
                }
                TrajectoryEntry::Heartbeat { at_s, message } => {
                    if matches!(mode, TimelineMode::Compact) {
                        continue;
                    }
                    writeln!(
                        out,
                        "t={:7.2}s  [HB]     tps={:6.2}  mem={}  compute={}  kv_tok={}",
                        at_s,
                        message.actual_throughput.unwrap_or(0.0),
                        message.memory_level.as_deref().unwrap_or("?"),
                        message.compute_level.as_deref().unwrap_or("?"),
                        message.kv_cache_tokens.unwrap_or(0),
                    )
                    .ok();
                }
                TrajectoryEntry::Signal { at_s, signal } => {
                    writeln!(
                        out,
                        "t={:7.2}s  [SIG]    {}@{}  {}",
                        at_s, signal.kind, signal.level, signal.detail,
                    )
                    .ok();
                }
                TrajectoryEntry::Directive {
                    at_s,
                    trigger,
                    directive,
                } => {
                    writeln!(
                        out,
                        "t={:7.2}s  [DIR] ← {}@{}  seq={}  cmds={:?}",
                        at_s, trigger.kind, trigger.level, directive.seq_id, directive.commands,
                    )
                    .ok();
                }
                TrajectoryEntry::ObservationDue {
                    at_s,
                    action,
                    recorded_at_s,
                } => {
                    writeln!(
                        out,
                        "t={:7.2}s  [OBS]    {} (recorded t={:.2}s)",
                        at_s, action, recorded_at_s,
                    )
                    .ok();
                }
                TrajectoryEntry::InjectionEvent { at_s, idx, started } => {
                    writeln!(
                        out,
                        "t={:7.2}s  [INJ]    idx={} {}",
                        at_s,
                        idx,
                        if *started { "started" } else { "ended" },
                    )
                    .ok();
                }
                TrajectoryEntry::Custom { at_s, name } => {
                    writeln!(out, "t={:7.2}s  [CUSTOM] {}", at_s, name).ok();
                }
            }
        }

        out
    }
}

#[derive(Debug, Clone, Copy)]
enum TimelineMode {
    Full,
    Compact,
}

// ─────────────────────────────────────────────────────────
// TrajectorySummary — 스냅샷용 요약 타입
// ─────────────────────────────────────────────────────────

/// 마지막 PhysicalState의 주요 숫자 (반올림 정제).
#[derive(Debug, Clone, Serialize)]
pub struct PhysicalStateSummary {
    /// KV 캐시 크기 (MiB, 소수 2자리).
    pub kv_cache_bytes_mib: f64,
    /// 디바이스 메모리 사용량 (MB, 소수 1자리).
    pub device_memory_used_mb: f64,
    /// 엔진 CPU 사용률 (%, 소수 1자리).
    pub engine_cpu_pct: f64,
    /// 엔진 GPU 사용률 (%, 소수 1자리).
    pub engine_gpu_pct: f64,
    /// CPU 주파수 비율 freq/max (소수 3자리).
    pub cpu_freq_ratio: f64,
    /// GPU 주파수 비율 freq/max (소수 3자리).
    pub gpu_freq_ratio: f64,
    /// 집계 온도 °C (소수 1자리).
    pub thermal_c: f64,
    /// 처리량 tokens/s (소수 2자리).
    pub throughput_tps: f64,
}

impl PhysicalStateSummary {
    fn from_snapshot(s: &PhysicalStateSnapshot, cfg_cpu_max: f64, cfg_gpu_max: f64) -> Self {
        fn r1(v: f64) -> f64 {
            (v * 10.0).round() / 10.0
        }
        fn r2(v: f64) -> f64 {
            (v * 100.0).round() / 100.0
        }
        fn r3(v: f64) -> f64 {
            (v * 1000.0).round() / 1000.0
        }
        let cpu_max = if cfg_cpu_max > 0.0 {
            cfg_cpu_max
        } else {
            s.cpu_freq_mhz.max(1.0)
        };
        let gpu_max = if cfg_gpu_max > 0.0 {
            cfg_gpu_max
        } else {
            s.gpu_freq_mhz.max(1.0)
        };
        Self {
            kv_cache_bytes_mib: r2(s.kv_cache_bytes / (1024.0 * 1024.0)),
            device_memory_used_mb: r1(s.device_memory_used_mb),
            engine_cpu_pct: r1(s.engine_cpu_pct),
            engine_gpu_pct: r1(s.engine_gpu_pct),
            cpu_freq_ratio: r3(s.cpu_freq_mhz / cpu_max),
            gpu_freq_ratio: r3(s.gpu_freq_mhz / gpu_max),
            thermal_c: r1(s.thermal_c),
            throughput_tps: r2(s.throughput_tps),
        }
    }
}

/// 시뮬레이션 실행 결과 요약 (insta 스냅샷 대상).
#[derive(Debug, Clone, Serialize)]
pub struct TrajectorySummary {
    /// 시뮬레이션 총 시간 (초, 소수 2자리).
    pub duration_s: f64,
    /// 기록된 heartbeat 수.
    pub heartbeat_count: usize,
    /// 신호 종류별 개수.
    pub signal_count_by_kind: BTreeMap<String, usize>,
    /// directive 총 개수.
    pub directive_count: usize,
    /// directive command 종류별 개수.
    pub directive_kinds: BTreeMap<String, usize>,
    /// 첫 directive 발생 시각 (초, 소수 2자리).
    pub first_directive_at_s: Option<f64>,
    /// 마지막 물리 상태 요약.
    pub state_final: Option<PhysicalStateSummary>,
}

impl TrajectorySummary {
    /// Trajectory에서 요약을 생성한다.
    ///
    /// `cpu_max_mhz` / `gpu_max_mhz`: DVFS 비율 계산용 (baseline cfg에서 전달).
    pub fn from_trajectory(traj: &Trajectory, cpu_max_mhz: f64, gpu_max_mhz: f64) -> Self {
        // 신호 종류별 집계
        let mut signal_count_by_kind: BTreeMap<String, usize> = BTreeMap::new();
        for entry in &traj.entries {
            if let TrajectoryEntry::Signal { signal, .. } = entry {
                *signal_count_by_kind.entry(signal.kind.clone()).or_insert(0) += 1;
            }
        }

        // directive 집계
        let mut directive_kinds: BTreeMap<String, usize> = BTreeMap::new();
        let mut first_directive_at_s: Option<f64> = None;
        let mut directive_count = 0usize;
        for entry in &traj.entries {
            if let TrajectoryEntry::Directive {
                at_s, directive, ..
            } = entry
            {
                directive_count += 1;
                if first_directive_at_s.is_none() {
                    first_directive_at_s = Some((at_s * 100.0).round() / 100.0);
                }
                for cmd in &directive.commands {
                    // "KvEvictSliding { keep_ratio: 0.8 }" → "KvEvictSliding"
                    let kind = cmd
                        .split_once('{')
                        .map(|(k, _)| k.trim())
                        .unwrap_or(cmd.as_str());
                    *directive_kinds.entry(kind.to_string()).or_insert(0) += 1;
                }
            }
        }

        // 마지막 StateSnapshot
        let state_final = traj.entries.iter().rev().find_map(|e| {
            if let TrajectoryEntry::StateSnapshot { state, .. } = e {
                Some(PhysicalStateSummary::from_snapshot(
                    state,
                    cpu_max_mhz,
                    gpu_max_mhz,
                ))
            } else {
                None
            }
        });

        let duration_s = traj
            .last_at_s()
            .map(|v| (v * 100.0).round() / 100.0)
            .unwrap_or(0.0);

        Self {
            duration_s,
            heartbeat_count: traj.heartbeat_count(),
            signal_count_by_kind,
            directive_count,
            directive_kinds,
            first_directive_at_s,
            state_final,
        }
    }
}
