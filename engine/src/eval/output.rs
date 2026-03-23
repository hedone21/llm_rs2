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
