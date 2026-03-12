/// KV cache utilization tracker (optional probe).
#[derive(Default)]
pub struct CacheTracker {
    records: Vec<CacheRecord>,
}

pub struct CacheRecord {
    pub step: usize,
    pub capacity: usize,
    pub current_pos: usize,
    pub utilization: f32,
    pub memory_bytes: usize,
}

impl CacheTracker {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn records(&self) -> &[CacheRecord] {
        &self.records
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({ "records": [] })
    }
}
