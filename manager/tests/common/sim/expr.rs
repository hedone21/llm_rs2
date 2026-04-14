//! evalexpr 래퍼: 로드 시 AST 컴파일, 런타임 HashMapContext 바인딩.
//!
//! Phase 1에서는 ExprContext와 CompiledExpr을 제공한다.
//! Phase 2에서 register_builtins()에 실제 내장 함수를 채운다.

// Phase 2+에서 사용될 API stub이 포함되므로 dead_code 경고를 억제한다.
#![allow(dead_code)]

use evalexpr::{ContextWithMutableVariables, EvalexprError, HashMapContext, Node, Value};
use serde::{Deserialize, Deserializer, de};

/// 컴파일된 expression 노드 + 원본 소스 문자열.
/// `serde::Deserialize` 시 즉시 AST를 컴파일한다.
#[derive(Debug, Clone)]
pub struct CompiledExpr {
    pub source: String,
    pub tree: Node,
}

impl<'de> Deserialize<'de> for CompiledExpr {
    fn deserialize<D: Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
        let s = String::deserialize(d)?;
        let tree = evalexpr::build_operator_tree(&s).map_err(|e| {
            de::Error::custom(format!("expression compile error in {:?}: {}", s, e))
        })?;
        Ok(CompiledExpr { source: s, tree })
    }
}

// ─────────────────────────────────────────────────────────
// ExprContext
// ─────────────────────────────────────────────────────────

/// evalexpr HashMapContext 래퍼. 변수 바인딩 및 평가 API를 제공한다.
pub struct ExprContext {
    pub ctx: HashMapContext,
}

impl ExprContext {
    pub fn new() -> Self {
        let mut ctx = HashMapContext::new();
        register_builtins(&mut ctx);
        ExprContext { ctx }
    }

    pub fn set_f64(&mut self, key: &str, v: f64) {
        self.ctx
            .set_value(key.to_string(), Value::Float(v))
            .expect("set_f64 should not fail");
    }

    pub fn set_str(&mut self, key: &str, v: &str) {
        self.ctx
            .set_value(key.to_string(), Value::String(v.to_string()))
            .expect("set_str should not fail");
    }

    pub fn eval_f64(&self, expr: &Node) -> Result<f64, EvalexprError> {
        expr.eval_float_with_context(&self.ctx)
    }

    pub fn eval_bool(&self, expr: &Node) -> Result<bool, EvalexprError> {
        expr.eval_boolean_with_context(&self.ctx)
    }
}

impl Default for ExprContext {
    fn default() -> Self {
        Self::new()
    }
}

// ─────────────────────────────────────────────────────────
// Builtin stubs (Phase 2에서 실제 구현)
// ─────────────────────────────────────────────────────────

/// Phase 2+에서 내장 함수로 주입할 helper 등록용.
/// Phase 1에서는 각 stub이 0.0을 반환한다.
pub fn register_builtins(_ctx: &mut HashMapContext) {
    // Phase 2에서 채울 함수들:
    //   phase_throughput(phase, device, partition_ratio, cpu_freq_ratio, gpu_freq_ratio, bw_util)
    //   throttle_factor(throttle_delay_ms)
    //   skip_boost(skip_ratio)
    //   base_by_phase(phase)
    //
    // evalexpr는 사용자 정의 함수를 HashMapContext에 Function variant로 등록한다.
    // 예시:
    // use evalexpr::{Function, Value};
    // ctx.set_function("phase_throughput".to_string(), Function::new(|_args| Ok(Value::Float(0.0))))
    //    .expect("register builtin");
}

// ─────────────────────────────────────────────────────────
// Dry-run 바인딩용 더미 context
// ─────────────────────────────────────────────────────────

/// config 로드 후 expression 전수 검증에 사용하는 더미 context.
/// initial_state의 모든 필드 + 알려진 action parameter를 0으로 바인딩한다.
pub fn make_dry_run_context() -> ExprContext {
    let mut ctx = ExprContext::new();

    // initial_state fields (숫자형)
    let numeric_fields = [
        "kv_cache_bytes",
        "kv_cache_capacity_bytes",
        "kv_cache_tokens",
        "kv_cache_token_capacity",
        "device_memory_total_mb",
        "device_memory_used_mb",
        "memory_bw_utilization_pct",
        "engine_cpu_pct",
        "external_cpu_pct",
        "cpu_freq_mhz",
        "cpu_max_freq_mhz",
        "cpu_min_freq_mhz",
        "engine_gpu_pct",
        "external_gpu_pct",
        "gpu_freq_mhz",
        "gpu_max_freq_mhz",
        "gpu_min_freq_mhz",
        "thermal_c",
        "cpu_cluster_thermal_c",
        "gpu_cluster_thermal_c",
        "throttle_threshold_c",
        "base_tps_decode_gpu",
        "base_tps_decode_cpu",
        "base_tps_decode_partition",
        "base_tps_prefill_gpu",
        "partition_ratio",
        "throttle_delay_ms",
        "tbt_target_ms",
        "skip_ratio",
        // action parameters
        "target_ratio",
        "delay_ms",
        "ratio",
        "bits",
    ];
    for f in &numeric_fields {
        ctx.set_f64(f, 0.0);
    }

    // derived 변수 (expression 참조용)
    let derived_fields = [
        "throughput_tps",
        "tbt_ms",
        "latency_degradation",
        "skip_ratio",
    ];
    for f in &derived_fields {
        ctx.set_f64(f, 0.0);
    }

    // 문자열 필드
    ctx.set_str("phase", "idle");
    ctx.set_str("active_device", "opencl");
    ctx.set_str("kv_dtype", "f16");
    ctx.set_str("device", "opencl");

    ctx
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compiled_expr_basic() {
        let src = "1.0 + 2.0";
        let tree = evalexpr::build_operator_tree(src).unwrap();
        let expr = CompiledExpr {
            source: src.to_string(),
            tree,
        };
        let ctx = ExprContext::new();
        let v = ctx.eval_f64(&expr.tree).unwrap();
        assert!((v - 3.0).abs() < 1e-9);
    }

    #[test]
    fn test_dry_run_context_has_target_ratio() {
        let ctx = make_dry_run_context();
        // evalexpr v11: 중괄호 없이 식별자 참조
        let src = "target_ratio * 2.0";
        let tree = evalexpr::build_operator_tree(src).unwrap();
        let v = ctx.eval_f64(&tree).unwrap();
        assert!((v - 0.0).abs() < 1e-9);
    }
}
