use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// 적응형 액션 식별자
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionId {
    SwitchHw,
    Throttle,
    KvOffloadDisk,
    KvEvictSliding,
    KvEvictH2o,
    KvQuantDynamic,
    LayerSkip,
}

impl ActionId {
    /// 문자열 식별자로부터 ActionId 변환
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<ActionId> {
        match s {
            "switch_hw" => Some(ActionId::SwitchHw),
            "throttle" => Some(ActionId::Throttle),
            "kv_offload_disk" => Some(ActionId::KvOffloadDisk),
            "kv_evict_sliding" => Some(ActionId::KvEvictSliding),
            "kv_evict_h2o" => Some(ActionId::KvEvictH2o),
            "kv_quant_dynamic" => Some(ActionId::KvQuantDynamic),
            "layer_skip" => Some(ActionId::LayerSkip),
            _ => None,
        }
    }

    /// 모든 ActionId 값을 반환
    pub fn all() -> &'static [ActionId] {
        &[
            ActionId::SwitchHw,
            ActionId::Throttle,
            ActionId::KvOffloadDisk,
            ActionId::KvEvictSliding,
            ActionId::KvEvictH2o,
            ActionId::KvQuantDynamic,
            ActionId::LayerSkip,
        ]
    }

    /// 이 액션의 주 도메인 (파라미터 결정 시 사용)
    pub fn primary_domain(&self) -> Domain {
        match self {
            ActionId::SwitchHw | ActionId::Throttle | ActionId::LayerSkip => Domain::Compute,
            ActionId::KvOffloadDisk
            | ActionId::KvEvictSliding
            | ActionId::KvEvictH2o
            | ActionId::KvQuantDynamic => Domain::Memory,
        }
    }
}

/// 액션이 lossless인지 lossy인지
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ActionKind {
    Lossless,
    Lossy,
}

/// Pressure 도메인
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Domain {
    Compute,
    Memory,
    Thermal,
}

/// 시스템 운영 모드
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum OperatingMode {
    Normal,
    Warning,
    Critical,
}

/// 3차원 pressure vector (PI Controller 출력)
#[derive(Debug, Clone, Copy, Default)]
pub struct PressureVector {
    pub compute: f32,
    pub memory: f32,
    pub thermal: f32,
}

impl PressureVector {
    pub fn max(&self) -> f32 {
        self.compute.max(self.memory).max(self.thermal)
    }
}

impl std::ops::Sub for PressureVector {
    type Output = ReliefVector;
    fn sub(self, rhs: PressureVector) -> ReliefVector {
        ReliefVector {
            compute: self.compute - rhs.compute,
            memory: self.memory - rhs.memory,
            thermal: self.thermal - rhs.thermal,
            latency: 0.0,
        }
    }
}

/// 4차원 relief vector (액션의 도메인별 완화 효과)
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize)]
pub struct ReliefVector {
    pub compute: f32,
    pub memory: f32,
    pub thermal: f32,
    pub latency: f32, // 음수 = 악화
}

impl ReliefVector {
    pub fn zero() -> Self {
        Self::default()
    }
}

impl std::ops::Add for ReliefVector {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self {
            compute: self.compute + rhs.compute,
            memory: self.memory + rhs.memory,
            thermal: self.thermal + rhs.thermal,
            latency: self.latency + rhs.latency,
        }
    }
}

impl std::ops::AddAssign for ReliefVector {
    fn add_assign(&mut self, rhs: Self) {
        self.compute += rhs.compute;
        self.memory += rhs.memory;
        self.thermal += rhs.thermal;
        self.latency += rhs.latency;
    }
}

/// Feature vector for Relief Estimator (13 features)
pub const FEATURE_DIM: usize = 13;

#[derive(Debug, Clone)]
pub struct FeatureVector {
    pub values: [f32; FEATURE_DIM],
}

impl FeatureVector {
    pub fn zeros() -> Self {
        Self {
            values: [0.0; FEATURE_DIM],
        }
    }
}

/// Feature indices
pub mod feature {
    pub const KV_OCCUPANCY: usize = 0;
    pub const IS_GPU: usize = 1;
    pub const TOKEN_PROGRESS: usize = 2;
    pub const IS_PREFILL: usize = 3;
    pub const KV_DTYPE_NORM: usize = 4;
    pub const TBT_RATIO: usize = 5;
    pub const TOKENS_GENERATED_NORM: usize = 6;
    pub const ACTIVE_SWITCH_HW: usize = 7;
    pub const ACTIVE_THROTTLE: usize = 8;
    pub const ACTIVE_KV_OFFLOAD: usize = 9;
    pub const ACTIVE_EVICTION: usize = 10;
    pub const ACTIVE_LAYER_SKIP: usize = 11;
    pub const ACTIVE_KV_QUANT: usize = 12;
}

/// 액션 메타데이터 (Registry에서 관리)
#[derive(Debug, Clone)]
pub struct ActionMeta {
    pub id: ActionId,
    pub kind: ActionKind,
    pub reversible: bool,
    pub param_range: Option<ParamRange>,
    pub exclusion_group: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ParamRange {
    pub param_name: String,
    pub min: f32,
    pub max: f32,
}

/// 액션 파라미터 (Directive에 포함)
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct ActionParams {
    pub values: HashMap<String, f32>,
}

/// 액션 명령 (Selector 출력)
#[derive(Debug, Clone)]
pub struct ActionCommand {
    pub action: ActionId,
    pub operation: Operation,
}

#[derive(Debug, Clone)]
pub enum Operation {
    Apply(ActionParams),
    Release,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pressure_vector_max() {
        let p = PressureVector {
            compute: 0.3,
            memory: 0.8,
            thermal: 0.5,
        };
        assert!((p.max() - 0.8).abs() < f32::EPSILON);
    }

    #[test]
    fn relief_vector_add() {
        let a = ReliefVector {
            compute: 0.3,
            memory: 0.0,
            thermal: 0.2,
            latency: -0.1,
        };
        let b = ReliefVector {
            compute: 0.0,
            memory: 0.5,
            thermal: 0.0,
            latency: -0.3,
        };
        let sum = a + b;
        assert!((sum.compute - 0.3).abs() < f32::EPSILON);
        assert!((sum.memory - 0.5).abs() < f32::EPSILON);
        assert!((sum.latency - (-0.4)).abs() < f32::EPSILON);
    }

    #[test]
    fn pressure_sub_gives_relief() {
        let before = PressureVector {
            compute: 0.8,
            memory: 0.6,
            thermal: 0.5,
        };
        let after = PressureVector {
            compute: 0.3,
            memory: 0.4,
            thermal: 0.5,
        };
        let relief = before - after;
        assert!((relief.compute - 0.5).abs() < f32::EPSILON);
        assert!((relief.memory - 0.2).abs() < f32::EPSILON);
        assert!((relief.thermal).abs() < f32::EPSILON);
    }

    #[test]
    fn action_id_primary_domain() {
        assert_eq!(ActionId::SwitchHw.primary_domain(), Domain::Compute);
        assert_eq!(ActionId::KvEvictSliding.primary_domain(), Domain::Memory);
    }

    #[test]
    fn action_id_serialization() {
        let id = ActionId::KvEvictSliding;
        let json = serde_json::to_string(&id).unwrap();
        assert_eq!(json, r#""kv_evict_sliding""#);
        let back: ActionId = serde_json::from_str(&json).unwrap();
        assert_eq!(back, id);
    }

    #[test]
    fn feature_vector_zeros() {
        let fv = FeatureVector::zeros();
        assert_eq!(fv.values.len(), FEATURE_DIM);
        assert!(fv.values.iter().all(|&v| v == 0.0));
    }
}
