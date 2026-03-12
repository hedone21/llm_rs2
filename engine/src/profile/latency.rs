/// Per-token decode latency tracker.
#[derive(Default)]
pub struct LatencyTracker {
    records: Vec<LatencyRecord>,
}

pub struct LatencyRecord {
    pub step: usize,
    pub forward_us: u64,
    pub sample_us: u64,
    pub total_us: u64,
    pub cache_len: usize,
}

impl LatencyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn record(
        &mut self,
        step: usize,
        forward_us: u64,
        sample_us: u64,
        total_us: u64,
        cache_len: usize,
    ) {
        self.records.push(LatencyRecord {
            step,
            forward_us,
            sample_us,
            total_us,
            cache_len,
        });
    }

    pub fn records(&self) -> &[LatencyRecord] {
        &self.records
    }

    pub fn to_json(&self) -> serde_json::Value {
        let records: Vec<serde_json::Value> = self
            .records
            .iter()
            .map(|r| {
                serde_json::json!({
                    "step": r.step,
                    "forward_us": r.forward_us,
                    "sample_us": r.sample_us,
                    "total_us": r.total_us,
                    "cache_len": r.cache_len,
                })
            })
            .collect();
        serde_json::json!({ "records": records })
    }
}
