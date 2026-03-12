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
    /// Optional per-KV-head importance `[n_kv_heads][cache_len]` (row-major).
    pub head_importance: Option<Vec<f32>>,
    /// Number of KV heads (for interpreting head_importance layout).
    pub n_kv_heads: usize,
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

    /// Take a snapshot of current importance scores.
    ///
    /// Only records if `step % snapshot_interval == 0`.
    /// Reads from `importance[..cache_len]` and optionally `head_importance`.
    pub fn take_snapshot(
        &mut self,
        step: usize,
        token_id: u32,
        cache_len: usize,
        importance: &[f32],
        head_importance: Option<&[f32]>,
        n_kv_heads: usize,
    ) {
        if !step.is_multiple_of(self.snapshot_interval) {
            return;
        }

        let len = cache_len.min(importance.len());
        let imp = importance[..len].to_vec();

        let head_imp = if self.track_per_head {
            head_importance.map(|hi| {
                let head_len = n_kv_heads * cache_len;
                hi[..head_len.min(hi.len())].to_vec()
            })
        } else {
            None
        };

        self.snapshots.push(ScoreSnapshot {
            step,
            token_id,
            cache_len,
            importance: imp,
            head_importance: head_imp,
            n_kv_heads,
        });
    }

    pub fn record_eviction(&mut self, event: EvictionEvent) {
        self.evictions.push(event);
    }

    pub fn to_json(&self) -> serde_json::Value {
        let snapshots: Vec<serde_json::Value> = self
            .snapshots
            .iter()
            .map(|s| {
                let mut obj = serde_json::json!({
                    "step": s.step,
                    "token_id": s.token_id,
                    "cache_len": s.cache_len,
                    "importance": s.importance,
                });
                if let Some(ref hi) = s.head_importance {
                    obj["head_importance"] = serde_json::json!(hi);
                    obj["n_kv_heads"] = serde_json::json!(s.n_kv_heads);
                }
                obj
            })
            .collect();

        let evictions: Vec<serde_json::Value> = self
            .evictions
            .iter()
            .map(|e| {
                serde_json::json!({
                    "step": e.step,
                    "policy": e.policy,
                    "before_len": e.before_len,
                    "after_len": e.after_len,
                    "evicted_count": e.evicted_count,
                    "partition": {
                        "prefix_end": e.partition.prefix_end,
                        "hh_count": e.partition.hh_count,
                        "recent_start": e.partition.recent_start,
                    },
                })
            })
            .collect();

        serde_json::json!({
            "snapshot_interval": self.snapshot_interval,
            "snapshot_count": self.snapshots.len(),
            "eviction_count": self.evictions.len(),
            "snapshots": snapshots,
            "evictions": evictions,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_score_tracker_default() {
        let t = ScoreTracker::default();
        assert_eq!(t.snapshot_interval(), 0);
        assert!(!t.track_per_head());
        assert!(t.snapshots().is_empty());
        assert!(t.evictions().is_empty());
    }

    #[test]
    fn test_score_tracker_new() {
        let t = ScoreTracker::new(5, true);
        assert_eq!(t.snapshot_interval(), 5);
        assert!(t.track_per_head());
    }

    #[test]
    fn test_snapshot_interval_min_one() {
        let t = ScoreTracker::new(0, false);
        assert_eq!(t.snapshot_interval(), 1);
    }

    #[test]
    fn test_take_snapshot_basic() {
        let mut t = ScoreTracker::new(1, false);
        let scores = vec![0.5, 0.3, 0.1, 0.05, 0.05];
        t.take_snapshot(0, 42, 5, &scores, None, 0);

        assert_eq!(t.snapshots().len(), 1);
        let s = &t.snapshots()[0];
        assert_eq!(s.step, 0);
        assert_eq!(s.token_id, 42);
        assert_eq!(s.cache_len, 5);
        assert_eq!(s.importance.len(), 5);
        assert!((s.importance[0] - 0.5).abs() < 1e-6);
        assert!(s.head_importance.is_none());
    }

    #[test]
    fn test_take_snapshot_interval_skip() {
        let mut t = ScoreTracker::new(3, false);
        let scores = vec![0.1; 10];

        for step in 0..10 {
            t.take_snapshot(step, 0, 10, &scores, None, 0);
        }

        // Only steps 0, 3, 6, 9 should be captured
        assert_eq!(t.snapshots().len(), 4);
        assert_eq!(t.snapshots()[0].step, 0);
        assert_eq!(t.snapshots()[1].step, 3);
        assert_eq!(t.snapshots()[2].step, 6);
        assert_eq!(t.snapshots()[3].step, 9);
    }

    #[test]
    fn test_take_snapshot_clones_data() {
        let mut t = ScoreTracker::new(1, false);
        let mut scores = vec![1.0, 2.0, 3.0];
        t.take_snapshot(0, 0, 3, &scores, None, 0);

        // Modify original — snapshot should be unchanged
        scores[0] = 999.0;
        assert!((t.snapshots()[0].importance[0] - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_take_snapshot_cache_len_bounds() {
        let mut t = ScoreTracker::new(1, false);
        // cache_len larger than scores buffer — should clamp
        let scores = vec![0.1, 0.2, 0.3];
        t.take_snapshot(0, 0, 10, &scores, None, 0);

        assert_eq!(t.snapshots()[0].importance.len(), 3);
    }

    #[test]
    fn test_take_snapshot_with_head_importance() {
        let mut t = ScoreTracker::new(1, true);
        let scores = vec![0.5, 0.3, 0.2];
        // 2 KV heads * 3 cache_len = 6 values
        let head_scores = vec![0.6, 0.2, 0.2, 0.4, 0.4, 0.2];
        t.take_snapshot(0, 42, 3, &scores, Some(&head_scores), 2);

        assert_eq!(t.snapshots().len(), 1);
        let hi = t.snapshots()[0].head_importance.as_ref().unwrap();
        assert_eq!(hi.len(), 6);
        assert!((hi[0] - 0.6).abs() < 1e-6);
        assert_eq!(t.snapshots()[0].n_kv_heads, 2);
    }

    #[test]
    fn test_take_snapshot_head_importance_not_tracked() {
        let mut t = ScoreTracker::new(1, false); // track_per_head = false
        let scores = vec![0.5, 0.3];
        let head_scores = vec![0.6, 0.2, 0.4, 0.4];
        t.take_snapshot(0, 0, 2, &scores, Some(&head_scores), 2);

        // head_importance should be None because track_per_head is false
        assert!(t.snapshots()[0].head_importance.is_none());
    }

    #[test]
    fn test_record_eviction() {
        let mut t = ScoreTracker::new(1, false);
        t.record_eviction(EvictionEvent {
            step: 50,
            policy: "h2o".to_string(),
            before_len: 256,
            after_len: 200,
            evicted_count: 56,
            partition: PartitionInfo {
                prefix_end: 4,
                hh_count: 98,
                recent_start: 102,
            },
        });

        assert_eq!(t.evictions().len(), 1);
        let e = &t.evictions()[0];
        assert_eq!(e.step, 50);
        assert_eq!(e.policy, "h2o");
        assert_eq!(e.before_len, 256);
        assert_eq!(e.after_len, 200);
        assert_eq!(e.evicted_count, 56);
        assert_eq!(e.partition.prefix_end, 4);
        assert_eq!(e.partition.hh_count, 98);
        assert_eq!(e.partition.recent_start, 102);
    }

    #[test]
    fn test_record_multiple_evictions() {
        let mut t = ScoreTracker::new(1, false);
        for i in 0..3 {
            t.record_eviction(EvictionEvent {
                step: i * 100,
                policy: "h2o".to_string(),
                before_len: 256 - i * 20,
                after_len: 200 - i * 20,
                evicted_count: 56,
                partition: PartitionInfo {
                    prefix_end: 4,
                    hh_count: 98 - i * 10,
                    recent_start: 102 - i * 10,
                },
            });
        }
        assert_eq!(t.evictions().len(), 3);
        assert_eq!(t.evictions()[2].step, 200);
    }

    #[test]
    fn test_to_json_empty() {
        let t = ScoreTracker::new(1, false);
        let json = t.to_json();

        assert_eq!(json["snapshot_interval"], 1);
        assert_eq!(json["snapshot_count"], 0);
        assert_eq!(json["eviction_count"], 0);
        assert!(json["snapshots"].as_array().unwrap().is_empty());
        assert!(json["evictions"].as_array().unwrap().is_empty());
    }

    #[test]
    fn test_to_json_with_snapshots() {
        let mut t = ScoreTracker::new(1, false);
        let scores = vec![0.8, 0.15, 0.05];
        t.take_snapshot(0, 100, 3, &scores, None, 0);

        let json = t.to_json();
        assert_eq!(json["snapshot_count"], 1);

        let snap = &json["snapshots"][0];
        assert_eq!(snap["step"], 0);
        assert_eq!(snap["token_id"], 100);
        assert_eq!(snap["cache_len"], 3);

        let imp = snap["importance"].as_array().unwrap();
        assert_eq!(imp.len(), 3);
        assert!((imp[0].as_f64().unwrap() - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_to_json_with_head_importance() {
        let mut t = ScoreTracker::new(1, true);
        let scores = vec![0.5, 0.5];
        let head_scores = vec![0.6, 0.4, 0.3, 0.7];
        t.take_snapshot(0, 10, 2, &scores, Some(&head_scores), 2);

        let json = t.to_json();
        let snap = &json["snapshots"][0];
        assert_eq!(snap["n_kv_heads"], 2);

        let hi = snap["head_importance"].as_array().unwrap();
        assert_eq!(hi.len(), 4);
    }

    #[test]
    fn test_to_json_with_evictions() {
        let mut t = ScoreTracker::new(1, false);
        t.record_eviction(EvictionEvent {
            step: 42,
            policy: "d2o".to_string(),
            before_len: 512,
            after_len: 384,
            evicted_count: 128,
            partition: PartitionInfo {
                prefix_end: 8,
                hh_count: 188,
                recent_start: 196,
            },
        });

        let json = t.to_json();
        assert_eq!(json["eviction_count"], 1);

        let ev = &json["evictions"][0];
        assert_eq!(ev["step"], 42);
        assert_eq!(ev["policy"], "d2o");
        assert_eq!(ev["before_len"], 512);
        assert_eq!(ev["after_len"], 384);
        assert_eq!(ev["evicted_count"], 128);
        assert_eq!(ev["partition"]["prefix_end"], 8);
        assert_eq!(ev["partition"]["hh_count"], 188);
        assert_eq!(ev["partition"]["recent_start"], 196);
    }

    #[test]
    fn test_to_json_combined() {
        let mut t = ScoreTracker::new(2, true);
        let scores = vec![0.5, 0.3, 0.2];
        let head_scores = vec![0.6, 0.2, 0.2, 0.4, 0.4, 0.2];

        // Steps 0,1,2,3,4 — interval 2 captures 0, 2, 4
        for step in 0..5 {
            t.take_snapshot(step, step as u32, 3, &scores, Some(&head_scores), 2);
        }
        t.record_eviction(EvictionEvent {
            step: 3,
            policy: "h2o".to_string(),
            before_len: 3,
            after_len: 2,
            evicted_count: 1,
            partition: PartitionInfo {
                prefix_end: 1,
                hh_count: 1,
                recent_start: 2,
            },
        });

        let json = t.to_json();
        assert_eq!(json["snapshot_count"], 3); // steps 0, 2, 4
        assert_eq!(json["eviction_count"], 1);
        assert_eq!(json["snapshot_interval"], 2);
    }
}
