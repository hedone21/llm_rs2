/// Per-operation profiler for forward_gen timing breakdown.
/// Accumulates microseconds per operation across layers and tokens.
#[derive(Default)]
pub struct OpProfiler {
    pub rms_norm: u64,
    pub matmul_qkv: u64,
    pub rope: u64,
    pub kv_update: u64,
    pub attention: u64,
    pub matmul_wo: u64,
    pub matmul_ffn: u64,
    pub silu_mul: u64,
    pub add_assign: u64,
    pub copy_residual: u64,
    pub cast: u64,
    pub count: u64,
}

impl OpProfiler {
    pub fn new() -> Self {
        Self::default()
    }

    /// Returns all operations as (name, value) pairs sorted by typical impact.
    fn ops(&self) -> [(&'static str, u64); 11] {
        [
            ("matmul_qkv", self.matmul_qkv),
            ("matmul_wo", self.matmul_wo),
            ("matmul_ffn", self.matmul_ffn),
            ("attention", self.attention),
            ("rms_norm", self.rms_norm),
            ("rope", self.rope),
            ("silu_mul", self.silu_mul),
            ("add_assign", self.add_assign),
            ("copy_residual", self.copy_residual),
            ("kv_update", self.kv_update),
            ("cast", self.cast),
        ]
    }

    fn total(&self) -> u64 {
        self.ops().iter().map(|(_, v)| v).sum()
    }

    pub fn print_report(&self) {
        let total = self.total();
        let n = if self.count > 0 { self.count } else { 1 };
        let pct = |v: u64| -> f64 {
            if total > 0 {
                v as f64 / total as f64 * 100.0
            } else {
                0.0
            }
        };
        eprintln!(
            "\n[Profile] Per-op breakdown (accumulated over {} layer-calls):",
            n
        );
        eprintln!(
            "  {:<20} {:>10} {:>10} {:>8}",
            "Operation", "Total(us)", "Avg(us)", "%"
        );
        eprintln!("  {:-<20} {:-<10} {:-<10} {:-<8}", "", "", "", "");
        for (name, val) in &self.ops() {
            eprintln!(
                "  {:<20} {:>10} {:>10} {:>7.1}%",
                name,
                val,
                val / n,
                pct(*val)
            );
        }
        eprintln!("  {:<20} {:>10} {:>10}", "TOTAL", total, total / n,);
    }

    /// Serialize to JSON for profile export.
    pub fn to_json(&self) -> serde_json::Value {
        let total = self.total();
        let n = self.count.max(1);
        let pct = |v: u64| -> f64 {
            if total > 0 {
                v as f64 / total as f64 * 100.0
            } else {
                0.0
            }
        };

        let mut breakdown = serde_json::Map::new();
        for (name, val) in &self.ops() {
            breakdown.insert(
                name.to_string(),
                serde_json::json!({
                    "total_us": val,
                    "avg_us": val / n,
                    "pct": (pct(*val) * 10.0).round() / 10.0,
                }),
            );
        }

        serde_json::json!({
            "count": self.count,
            "total_us": total,
            "breakdown": breakdown,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_op_profiler_default_is_zero() {
        let p = OpProfiler::new();
        assert_eq!(p.total(), 0);
        assert_eq!(p.count, 0);
    }

    #[test]
    fn test_op_profiler_total() {
        let mut p = OpProfiler::new();
        p.matmul_qkv = 100;
        p.attention = 200;
        p.matmul_ffn = 300;
        assert_eq!(p.total(), 600);
    }

    #[test]
    fn test_op_profiler_to_json_structure() {
        let mut p = OpProfiler::new();
        p.matmul_qkv = 1000;
        p.attention = 2000;
        p.matmul_ffn = 3000;
        p.rms_norm = 500;
        p.count = 10;

        let json = p.to_json();

        assert_eq!(json["count"], 10);
        assert_eq!(json["total_us"], 6500);

        let breakdown = json["breakdown"].as_object().unwrap();
        assert_eq!(breakdown.len(), 11); // all 11 ops present

        // Verify specific op values
        assert_eq!(breakdown["matmul_qkv"]["total_us"], 1000);
        assert_eq!(breakdown["matmul_qkv"]["avg_us"], 100); // 1000 / 10
        assert_eq!(breakdown["attention"]["total_us"], 2000);
    }

    #[test]
    fn test_op_profiler_to_json_percentages() {
        let mut p = OpProfiler::new();
        p.matmul_qkv = 500;
        p.matmul_ffn = 500;
        p.count = 1;

        let json = p.to_json();
        let breakdown = json["breakdown"].as_object().unwrap();

        // Each should be 50%
        assert_eq!(breakdown["matmul_qkv"]["pct"], 50.0);
        assert_eq!(breakdown["matmul_ffn"]["pct"], 50.0);
        // Others should be 0%
        assert_eq!(breakdown["attention"]["pct"], 0.0);
    }

    #[test]
    fn test_op_profiler_to_json_zero_count() {
        let mut p = OpProfiler::new();
        p.matmul_qkv = 100;
        // count = 0 → should use 1 to avoid division by zero
        let json = p.to_json();
        assert_eq!(json["breakdown"]["matmul_qkv"]["avg_us"], 100);
    }

    #[test]
    fn test_op_profiler_to_json_zero_total() {
        let p = OpProfiler::new();
        let json = p.to_json();
        assert_eq!(json["total_us"], 0);
        // All percentages should be 0 when total is 0
        let breakdown = json["breakdown"].as_object().unwrap();
        for (_name, val) in breakdown {
            assert_eq!(val["pct"], 0.0);
        }
    }
}
