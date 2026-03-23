//! Eval output types: unified result structures for all eval-ll modes.

use super::hook::MetricsSummary;

/// Configuration for the generic eval loop.
#[derive(Debug, Clone)]
pub struct EvalConfig {
    pub max_seq_len: usize,
    pub effective_budget: usize,
    pub greedy: bool,
    pub kv_type: String,
    pub use_gpu_attn: bool,
    pub qcf_mode: String,
    /// Model vocabulary size (used for logits buffer allocation).
    pub vocab_size: usize,
    /// Model hidden dimension (used for x_gen buffer allocation).
    pub hidden_size: usize,
}

/// A single evaluation question (grouped format).
#[derive(Debug, Clone)]
pub struct EvalQuestion {
    pub id: String,
    pub prompt: String,
    pub choices: Vec<String>,
}

/// Unified output from the generic eval loop.
#[derive(Debug)]
pub struct EvalOutput {
    /// Per-question results.
    pub results: Vec<serde_json::Value>,
    /// Run configuration.
    pub config: serde_json::Value,
    /// Wall-clock time in seconds.
    pub wall_time_s: f64,
    /// Aggregated QCF/OPR metrics.
    pub metrics_summary: MetricsSummary,
    /// Layer importance table (if skip_config active).
    pub layer_importance: Option<serde_json::Value>,
    /// Layer skip QCF (cos_sim based).
    pub layer_skip_qcf: Option<f32>,
    /// Layer skip QCF normalized.
    pub layer_skip_qcf_normalized: Option<f32>,
    /// Layer skip OPR (residual norm ratio).
    pub opr_layer_skip: Option<f64>,
    /// Number of skipped layers.
    pub opr_layer_skip_layers: Option<usize>,
}

impl EvalOutput {
    /// Serialize to JSON matching the existing output format.
    pub fn to_json(&self) -> anyhow::Result<String> {
        let mut output = serde_json::json!({
            "results": self.results,
            "config": self.config,
            "wall_time_s": self.wall_time_s,
        });

        if let Some(ref li) = self.layer_importance {
            output["layer_importance"] = li.clone();
        }
        if let Some(qcf) = self.layer_skip_qcf {
            output["layer_skip_qcf"] = serde_json::json!(qcf);
        }
        if let Some(n) = self.layer_skip_qcf_normalized {
            output["layer_skip_qcf_normalized"] = serde_json::json!(n);
        }
        if let Some(opr) = self.opr_layer_skip {
            output["opr_layer_skip"] = serde_json::json!(opr);
        }
        if let Some(n) = self.opr_layer_skip_layers {
            output["opr_layer_skip_layers"] = serde_json::json!(n);
        }

        serde_json::to_string_pretty(&output).map_err(Into::into)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::eval::qcf_helpers::build_opr_fields;

    // ── helpers ──────────────────────────────────────────────────────────────

    fn minimal_output() -> EvalOutput {
        EvalOutput {
            results: vec![],
            config: serde_json::json!({}),
            wall_time_s: 0.0,
            metrics_summary: MetricsSummary::default(),
            layer_importance: None,
            layer_skip_qcf: None,
            layer_skip_qcf_normalized: None,
            opr_layer_skip: None,
            opr_layer_skip_layers: None,
        }
    }

    fn parse(output: &EvalOutput) -> serde_json::Value {
        let s = output.to_json().expect("to_json should not fail");
        serde_json::from_str(&s).expect("should be valid JSON")
    }

    // ── 1. EvalOutput.to_json() 필수 키 구조 검증 ────────────────────────────

    /// 항상 존재해야 하는 최상위 키(results, config, wall_time_s)가 올바른 타입으로
    /// 포함되어 있는지 확인한다.
    #[test]
    fn test_eval_output_json_required_keys() {
        let output = EvalOutput {
            results: vec![serde_json::json!({"id": "q1", "predicted": 0})],
            config: serde_json::json!({"model": "test"}),
            wall_time_s: 1.23,
            ..minimal_output()
        };
        let v = parse(&output);

        assert!(v["results"].is_array(), "results must be an array");
        assert!(v["config"].is_object(), "config must be an object");
        assert!(v["wall_time_s"].is_number(), "wall_time_s must be a number");
        assert!((v["wall_time_s"].as_f64().unwrap() - 1.23).abs() < 1e-9);
    }

    /// results 배열이 정확한 원소 수를 유지하는지 확인한다.
    #[test]
    fn test_eval_output_results_length() {
        let output = EvalOutput {
            results: vec![
                serde_json::json!({"id": "q1"}),
                serde_json::json!({"id": "q2"}),
                serde_json::json!({"id": "q3"}),
            ],
            ..minimal_output()
        };
        let v = parse(&output);
        assert_eq!(v["results"].as_array().unwrap().len(), 3);
    }

    // ── 2. layer_skip 필드 조건부 포함 검증 ──────────────────────────────────

    /// layer_skip 관련 필드가 모두 Some일 때 JSON에 포함되는지 확인한다.
    #[test]
    fn test_eval_output_with_layer_skip_fields_present() {
        let output = EvalOutput {
            layer_skip_qcf: Some(0.15),
            layer_skip_qcf_normalized: Some(0.176),
            opr_layer_skip: Some(0.384),
            opr_layer_skip_layers: Some(3),
            ..minimal_output()
        };
        let v = parse(&output);

        // 값 존재 확인
        assert!(
            !v["layer_skip_qcf"].is_null(),
            "layer_skip_qcf should be present"
        );
        assert!(
            !v["layer_skip_qcf_normalized"].is_null(),
            "layer_skip_qcf_normalized should be present"
        );
        assert!(
            !v["opr_layer_skip"].is_null(),
            "opr_layer_skip should be present"
        );
        assert!(
            !v["opr_layer_skip_layers"].is_null(),
            "opr_layer_skip_layers should be present"
        );

        // 값 정확도 확인 (f32 → JSON → f64 변환 허용 오차)
        let qcf = v["layer_skip_qcf"].as_f64().unwrap();
        assert!(
            (qcf - 0.15).abs() < 1e-5,
            "layer_skip_qcf value mismatch: {}",
            qcf
        );
        let norm = v["layer_skip_qcf_normalized"].as_f64().unwrap();
        assert!(
            (norm - 0.176).abs() < 1e-5,
            "layer_skip_qcf_normalized value mismatch: {}",
            norm
        );
        assert!((v["opr_layer_skip"].as_f64().unwrap() - 0.384).abs() < 1e-9);
        assert_eq!(v["opr_layer_skip_layers"].as_u64().unwrap(), 3);
    }

    // ── 3. layer_skip 없을 때 필드 부재 확인 ─────────────────────────────────

    /// layer_skip 관련 필드가 모두 None일 때 JSON 키 자체가 없어야 한다.
    #[test]
    fn test_eval_output_without_layer_skip_fields_absent() {
        let output = minimal_output(); // 모든 layer_skip 필드가 None
        let v = parse(&output);

        // serde_json::Value::get() 은 키가 없으면 None을 반환한다.
        assert!(
            v.get("layer_skip_qcf").is_none(),
            "layer_skip_qcf should be absent when None"
        );
        assert!(
            v.get("layer_skip_qcf_normalized").is_none(),
            "layer_skip_qcf_normalized should be absent when None"
        );
        assert!(
            v.get("opr_layer_skip").is_none(),
            "opr_layer_skip should be absent when None"
        );
        assert!(
            v.get("opr_layer_skip_layers").is_none(),
            "opr_layer_skip_layers should be absent when None"
        );
    }

    /// layer_importance도 None일 때 키가 없어야 한다.
    #[test]
    fn test_eval_output_layer_importance_absent_when_none() {
        let output = minimal_output();
        let v = parse(&output);
        assert!(v.get("layer_importance").is_none());
    }

    /// layer_importance가 Some일 때 JSON 배열로 포함되어야 한다.
    #[test]
    fn test_eval_output_layer_importance_present_when_some() {
        let output = EvalOutput {
            layer_importance: Some(serde_json::json!([
                {"layer": 0, "sublayer": "Attn", "importance": 0.9, "opr": 0.1}
            ])),
            ..minimal_output()
        };
        let v = parse(&output);
        let li = &v["layer_importance"];
        assert!(li.is_array(), "layer_importance should be an array");
        assert_eq!(li.as_array().unwrap().len(), 1);
    }

    // ── 4. build_opr_fields — eviction 경로 검증 ─────────────────────────────

    /// opr_eviction이 Some일 때 opr_eviction과 opr_eviction_events가 숫자로 포함된다.
    #[test]
    fn test_build_opr_fields_eviction_present() {
        let ms = MetricsSummary {
            opr_eviction: Some(0.232),
            opr_eviction_events: 100,
            opr_quantization: None,
            opr_quantization_events: 0,
            ..Default::default()
        };
        let fields = build_opr_fields(&ms);

        assert!((fields["opr_eviction"].as_f64().unwrap() - 0.232).abs() < 1e-9);
        assert_eq!(fields["opr_eviction_events"].as_u64().unwrap(), 100);
        // quantization 필드는 null이어야 한다
        assert!(fields["opr_quantization"].is_null());
        assert!(fields["opr_quantization_events"].is_null());
    }

    /// opr_eviction이 None일 때 opr_eviction_events도 null이어야 한다.
    #[test]
    fn test_build_opr_fields_eviction_absent() {
        let ms = MetricsSummary {
            opr_eviction: None,
            opr_eviction_events: 5, // None이면 events도 null로 억제됨
            ..Default::default()
        };
        let fields = build_opr_fields(&ms);

        assert!(fields["opr_eviction"].is_null());
        // opr_eviction이 None → opr_eviction_events도 null (map 패턴)
        assert!(fields["opr_eviction_events"].is_null());
    }

    // ── 5. build_opr_fields — KIVI 경로 검증 ─────────────────────────────────

    /// opr_quantization이 Some일 때 opr_quantization과 opr_quantization_events가
    /// 숫자로 포함되고, eviction 필드는 null이어야 한다.
    #[test]
    fn test_build_opr_fields_kivi_present() {
        let ms = MetricsSummary {
            opr_eviction: None,
            opr_eviction_events: 0,
            opr_quantization: Some(0.089),
            opr_quantization_events: 12,
            ..Default::default()
        };
        let fields = build_opr_fields(&ms);

        assert!(fields["opr_eviction"].is_null());
        assert!(fields["opr_eviction_events"].is_null());
        assert!((fields["opr_quantization"].as_f64().unwrap() - 0.089).abs() < 1e-9);
        assert_eq!(fields["opr_quantization_events"].as_u64().unwrap(), 12);
    }

    /// opr_quantization이 None일 때 opr_quantization_events도 null이어야 한다.
    #[test]
    fn test_build_opr_fields_kivi_absent() {
        let ms = MetricsSummary {
            opr_quantization: None,
            opr_quantization_events: 7,
            ..Default::default()
        };
        let fields = build_opr_fields(&ms);

        assert!(fields["opr_quantization"].is_null());
        assert!(fields["opr_quantization_events"].is_null());
    }

    // ── 6. MetricsSummary Default 검증 ────────────────────────────────────────

    /// Default MetricsSummary는 모든 optional 필드가 None이어야 한다.
    #[test]
    fn test_metrics_summary_default_none_fields() {
        let ms = MetricsSummary::default();
        assert!(ms.opr_eviction.is_none());
        assert!(ms.opr_quantization.is_none());
        assert_eq!(ms.opr_eviction_events, 0);
        assert_eq!(ms.opr_quantization_events, 0);
        assert_eq!(ms.qcf_attn_total, 0.0);
        assert_eq!(ms.qcf_caote_total, 0.0);
        assert_eq!(ms.qcf_normalized_total, 0.0);
    }

    // ── 7. to_json() 결정론적 직렬화 검증 ────────────────────────────────────

    /// 동일한 EvalOutput을 두 번 직렬화하면 동일한 문자열이 나와야 한다.
    #[test]
    fn test_eval_output_to_json_deterministic() {
        let output = EvalOutput {
            results: vec![serde_json::json!({"id": "q1", "predicted": 0})],
            config: serde_json::json!({"kv_type": "f32"}),
            wall_time_s: 3.14,
            layer_skip_qcf: Some(0.25),
            layer_skip_qcf_normalized: Some(0.333),
            opr_layer_skip: Some(0.5),
            opr_layer_skip_layers: Some(2),
            ..minimal_output()
        };
        let json1 = output.to_json().unwrap();
        let json2 = output.to_json().unwrap();
        assert_eq!(json1, json2);
    }

    /// to_json() 결과가 유효한 pretty-printed JSON인지 확인한다.
    #[test]
    fn test_eval_output_to_json_is_valid_json() {
        let output = minimal_output();
        let json_str = output.to_json().unwrap();
        // serde_json::from_str이 성공하면 유효한 JSON이다.
        let result: Result<serde_json::Value, _> = serde_json::from_str(&json_str);
        assert!(result.is_ok(), "to_json() must produce valid JSON");
    }
}
