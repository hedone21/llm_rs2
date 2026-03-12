/// H2O attention score snapshot tracker.
///
/// Records per-step importance score snapshots and eviction events
/// for post-hoc visualization (heatmaps, partition evolution, etc.).
#[derive(Default)]
pub struct ScoreTracker {
    snapshots: Vec<ScoreSnapshot>,
    evictions: Vec<EvictionEvent>,
    snapshot_interval: usize,
    track_per_head: bool,
}

/// A snapshot of attention importance scores at a single decode step.
pub struct ScoreSnapshot {
    pub step: usize,
    pub token_id: u32,
    pub cache_len: usize,
    /// Per-position importance scores `[cache_len]`.
    pub importance: Vec<f32>,
    /// Optional per-KV-head importance `[n_kv_heads * cache_len]` (row-major).
    pub head_importance: Option<Vec<f32>>,
}

/// An eviction event captured by the profiler.
pub struct EvictionEvent {
    pub step: usize,
    pub policy: String,
    pub before_len: usize,
    pub after_len: usize,
    pub evicted_count: usize,
    pub partition: PartitionInfo,
}

/// Partition boundaries at the time of eviction.
pub struct PartitionInfo {
    /// Protected prefix boundary: `[0..prefix_end)`.
    pub prefix_end: usize,
    /// Number of heavy hitter tokens selected.
    pub hh_count: usize,
    /// Start of the recent window: `[recent_start..)`.
    pub recent_start: usize,
}

impl ScoreTracker {
    pub fn new(snapshot_interval: usize, track_per_head: bool) -> Self {
        Self {
            snapshots: Vec::new(),
            evictions: Vec::new(),
            snapshot_interval: snapshot_interval.max(1),
            track_per_head,
        }
    }

    pub fn snapshot_interval(&self) -> usize {
        self.snapshot_interval
    }

    pub fn track_per_head(&self) -> bool {
        self.track_per_head
    }

    pub fn snapshots(&self) -> &[ScoreSnapshot] {
        &self.snapshots
    }

    pub fn evictions(&self) -> &[EvictionEvent] {
        &self.evictions
    }

    pub fn record_eviction(&mut self, event: EvictionEvent) {
        self.evictions.push(event);
    }

    pub fn to_json(&self) -> serde_json::Value {
        serde_json::json!({
            "snapshot_interval": self.snapshot_interval,
            "snapshots": [],
            "evictions": [],
        })
    }
}
