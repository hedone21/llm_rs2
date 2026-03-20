//! Math utility functions for KV cache compression algorithms.

/// In-place 1D average pooling with same-size output.
///
/// `kernel_size` must be odd. Padding is `kernel_size / 2` (symmetric).
/// Uses a temporary buffer to avoid in-place aliasing issues.
pub fn avg_pool_1d(data: &mut [f32], kernel_size: usize) {
    if kernel_size <= 1 || data.is_empty() {
        return;
    }
    let pad = kernel_size / 2;
    let len = data.len();
    let mut buf = vec![0.0f32; len];

    for (i, out) in buf.iter_mut().enumerate().take(len) {
        let start = i.saturating_sub(pad);
        let end = (i + pad + 1).min(len);
        let sum: f32 = data[start..end].iter().sum();
        *out = sum / (end - start) as f32;
    }

    data.copy_from_slice(&buf);
}

/// Per-head top-k index selection from a flat score array.
///
/// `scores` layout: `[n_heads * prefix_len]` (head-major flat).
/// Returns `Vec<Vec<usize>>` where each inner vec is sorted ascending indices.
///
/// Uses `select_nth_unstable_by` for O(n) average partial sort.
pub fn topk_indices_per_head(
    scores: &[f32],
    n_heads: usize,
    prefix_len: usize,
    keep_count: usize,
) -> Vec<Vec<usize>> {
    assert_eq!(
        scores.len(),
        n_heads * prefix_len,
        "scores.len()={} != n_heads({}) * prefix_len({})",
        scores.len(),
        n_heads,
        prefix_len
    );

    let keep = keep_count.min(prefix_len);

    (0..n_heads)
        .map(|h| {
            let head_scores = &scores[h * prefix_len..(h + 1) * prefix_len];
            let mut indexed: Vec<(usize, f32)> = head_scores
                .iter()
                .enumerate()
                .map(|(i, &s)| (i, s))
                .collect();

            if keep >= prefix_len {
                // Keep all
                return (0..prefix_len).collect();
            }

            // Partial sort: put top-k elements in first k positions
            indexed.select_nth_unstable_by(keep, |a, b| {
                b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal)
            });

            let mut top: Vec<usize> = indexed[..keep].iter().map(|(i, _)| *i).collect();
            top.sort_unstable(); // Preserve positional order
            top
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_avg_pool_1d_identity() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        avg_pool_1d(&mut data, 1);
        assert_eq!(data, vec![1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn test_avg_pool_1d_kernel3() {
        let mut data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        avg_pool_1d(&mut data, 3);
        // i=0: avg(1,2) = 1.5
        // i=1: avg(1,2,3) = 2.0
        // i=2: avg(2,3,4) = 3.0
        // i=3: avg(3,4,5) = 4.0
        // i=4: avg(4,5) = 4.5
        assert!((data[0] - 1.5).abs() < 1e-5);
        assert!((data[1] - 2.0).abs() < 1e-5);
        assert!((data[2] - 3.0).abs() < 1e-5);
        assert!((data[3] - 4.0).abs() < 1e-5);
        assert!((data[4] - 4.5).abs() < 1e-5);
    }

    #[test]
    fn test_avg_pool_1d_kernel5() {
        let mut data = vec![0.0, 0.0, 10.0, 0.0, 0.0];
        avg_pool_1d(&mut data, 5);
        // Smoothing a spike — all should converge toward 2.0
        assert!((data[2] - 10.0 / 5.0).abs() < 1e-5); // center: avg(0,0,10,0,0)=2.0
    }

    #[test]
    fn test_avg_pool_1d_empty() {
        let mut data: Vec<f32> = vec![];
        avg_pool_1d(&mut data, 3);
        assert!(data.is_empty());
    }

    #[test]
    fn test_avg_pool_1d_length_preserved() {
        let mut data = vec![1.0; 100];
        avg_pool_1d(&mut data, 5);
        assert_eq!(data.len(), 100);
    }

    #[test]
    fn test_topk_indices_basic() {
        // 2 heads, 5 prefix tokens, keep 3
        let scores = vec![
            // head 0: positions with scores [1, 5, 2, 4, 3]
            1.0, 5.0, 2.0, 4.0, 3.0, // head 1: positions with scores [5, 1, 4, 2, 3]
            5.0, 1.0, 4.0, 2.0, 3.0,
        ];
        let result = topk_indices_per_head(&scores, 2, 5, 3);
        assert_eq!(result.len(), 2);
        // head 0: top-3 by score → indices 1(5.0), 3(4.0), 4(3.0) → sorted: [1,3,4]
        assert_eq!(result[0], vec![1, 3, 4]);
        // head 1: top-3 → indices 0(5.0), 2(4.0), 4(3.0) → sorted: [0,2,4]
        assert_eq!(result[1], vec![0, 2, 4]);
    }

    #[test]
    fn test_topk_indices_keep_all() {
        let scores = vec![1.0, 2.0, 3.0];
        let result = topk_indices_per_head(&scores, 1, 3, 5);
        assert_eq!(result[0], vec![0, 1, 2]);
    }

    #[test]
    fn test_topk_indices_keep_one() {
        let scores = vec![1.0, 5.0, 3.0, 2.0];
        let result = topk_indices_per_head(&scores, 1, 4, 1);
        assert_eq!(result[0], vec![1]); // index 1 has highest score
    }

    #[test]
    fn test_topk_indices_sorted_ascending() {
        let scores = vec![0.1, 0.9, 0.5, 0.8, 0.3, 0.7, 0.2, 0.6];
        let result = topk_indices_per_head(&scores, 1, 8, 4);
        // Top-4: idx 1(0.9), 3(0.8), 5(0.7), 7(0.6) → sorted: [1,3,5,7]
        assert_eq!(result[0], vec![1, 3, 5, 7]);
    }

    #[test]
    fn test_topk_per_head_independence() {
        // Each head selects different tokens
        let scores = vec![
            10.0, 1.0, 1.0, 1.0, // head 0: prefers idx 0
            1.0, 10.0, 1.0, 1.0, // head 1: prefers idx 1
        ];
        let result = topk_indices_per_head(&scores, 2, 4, 1);
        assert_eq!(result[0], vec![0]);
        assert_eq!(result[1], vec![1]);
    }
}
