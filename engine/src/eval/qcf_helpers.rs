//! Layer-swap QCF dump infrastructure for the `--qcf-dump` workflow.
//!
//! Collected by `generate.rs` and serialized to a JSON file consumed by
//! the external harness (`pact2026/experiments/scripts/`) to correlate
//! `qcf_swap_predicted` with actual NLL/quality metrics.

use super::output::EvalOutput;
use crate::core::qcf::layer_importance::ImportanceTable;
use crate::models::weights::QuantNoiseTable;

/// Context for one layer-swap QCF measurement run.
pub struct QcfSwapDumpContext<'a> {
    pub model_arch: &'a str,
    pub model_path: &'a str,
    pub secondary_path: Option<&'a str>,
    pub primary_dtype: &'a str,
    pub secondary_dtype: &'a str,
    pub num_layers: usize,
    pub force_swap_ratio: Option<f32>,
    /// Decoder layer indices that were swapped (empty if ratio=0 or no secondary).
    pub swap_set: &'a [usize],
    /// QCF_swap predicted value from `WeightSwapDecider` (ENG-ALG-217).
    pub qcf_swap_predicted: f32,
    /// `true` when uniform fallback was used (importance/noise absent).
    pub fallback_used: bool,
    /// Full importance table built from warmup prefill (optional).
    pub importance_table: Option<&'a ImportanceTable>,
    /// Quantization noise table built from secondary mmap (optional).
    pub noise_table: Option<&'a QuantNoiseTable>,
    /// Perplexity result from `run_ppl()` (None in generation mode).
    pub ppl: Option<f64>,
    /// Average negative log-likelihood (None in generation mode).
    pub avg_nll: Option<f64>,
    /// Number of tokens evaluated in the main measurement.
    pub n_eval_tokens: usize,
    /// Total wall-clock time in seconds (warmup + swap + measurement).
    pub wall_time_s: f64,
    /// Number of warmup prefill tokens used to build the importance table.
    pub warmup_tokens: usize,
    /// Backend string (e.g. "cpu", "opencl", "cuda").
    pub backend: &'a str,
    /// KV cache dtype string (e.g. "f16", "f32", "q4_0").
    pub kv_type: &'a str,
    /// Path to the PPL reference corpus file (None in generation mode).
    pub ppl_corpus: Option<&'a str>,
    /// EvalOutput from `--eval-ll` mode (per-question NLL summary).
    ///
    /// Set when `--eval-ll` and `--qcf-dump` are both active. None in PPL/generation mode.
    /// The full `EvalOutput` is serialized as `eval_ll_output` in the JSON dump so the
    /// external harness can compute `qcf_swap_predicted ↔ ΔNLL` Spearman ρ directly.
    pub eval_ll_output: Option<&'a EvalOutput>,
}

/// Serialize a `QcfSwapDumpContext` to a JSON file (schema_version 1).
///
/// The JSON schema matches the external harness expectation exactly:
/// - All fields are always present (`null` when absent, NOT omitted).
/// - `importance_table` and `noise_table` include all collected entries.
/// - `noise_table` excludes NaN/non-finite ε entries.
/// - `swap_count` is derived as `swap_set.len()`.
pub fn dump_qcf_swap_json(
    path: &std::path::Path,
    ctx: &QcfSwapDumpContext<'_>,
) -> anyhow::Result<()> {
    use serde_json::{Value, json};

    // importance_table entries
    let importance_arr: Value = match ctx.importance_table {
        Some(table) => {
            let entries: Vec<Value> = table
                .entries()
                .iter()
                .map(|e| {
                    json!({
                        "layer": e.layer_id,
                        "sublayer": format!("{:?}", e.sublayer),
                        "importance": e.importance,
                        "opr": e.opr,
                    })
                })
                .collect();
            Value::Array(entries)
        }
        None => Value::Null,
    };

    // noise_table entries: only finite ε values
    let noise_arr: Value = match ctx.noise_table {
        Some(table) => {
            let entries: Vec<Value> = (0..ctx.num_layers)
                .filter_map(|i| {
                    table.epsilon(i).map(|eps| {
                        json!({
                            "layer": i,
                            "epsilon": eps,
                        })
                    })
                })
                .collect();
            Value::Array(entries)
        }
        None => Value::Null,
    };

    // eval_ll_output: serialize as JSON object when present, null otherwise.
    let eval_ll_output_val: Value = match ctx.eval_ll_output {
        Some(output) => {
            let json_str = output.to_json().unwrap_or_else(|_| "null".to_string());
            serde_json::from_str(&json_str).unwrap_or(Value::Null)
        }
        None => Value::Null,
    };

    let doc = json!({
        "schema_version": 1,
        "model_arch": ctx.model_arch,
        "model_path": ctx.model_path,
        "secondary_path": ctx.secondary_path,
        "primary_dtype": ctx.primary_dtype,
        "secondary_dtype": ctx.secondary_dtype,
        "num_layers": ctx.num_layers,
        "force_swap_ratio": ctx.force_swap_ratio,
        "swap_set": ctx.swap_set,
        "swap_count": ctx.swap_set.len(),
        "qcf_swap_predicted": ctx.qcf_swap_predicted,
        "fallback_used": ctx.fallback_used,
        "importance_table": importance_arr,
        "noise_table": noise_arr,
        "ppl": ctx.ppl,
        "avg_nll": ctx.avg_nll,
        "n_eval_tokens": ctx.n_eval_tokens,
        "wall_time_s": ctx.wall_time_s,
        "warmup_tokens": ctx.warmup_tokens,
        "backend": ctx.backend,
        "kv_type": ctx.kv_type,
        "ppl_corpus": ctx.ppl_corpus,
        "eval_ll_output": eval_ll_output_val,
    });

    let json_str = serde_json::to_string_pretty(&doc)?;
    std::fs::write(path, json_str)
        .map_err(|e| anyhow::anyhow!("Failed to write QCF dump to {}: {}", path.display(), e))?;

    Ok(())
}
