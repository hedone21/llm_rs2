/// Attention entropy tracker (optional probe).
///
/// Computes Shannon entropy of post-softmax attention distributions
/// to measure attention focus vs. diffuseness per layer.
#[derive(Default)]
pub struct EntropyTracker {
    records: Vec<EntropyRecord>,
}

pub struct EntropyRecord {
    pub step: usize,
    pub per_layer: Vec<f32>,
}

impl EntropyTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn records(&self) -> &[EntropyRecord] {
        &self.records
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "records": [] })
    }
}
