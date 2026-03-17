//! Token sampling strategies for autoregressive generation.
//!
//! Extracted from `generate.rs` to enable unit testing and reuse.

/// Configuration for token sampling.
pub struct SamplingConfig {
    pub temperature: f32,
    pub top_p: f32,
    pub top_k: usize,
    pub repetition_penalty: f32,
    pub repetition_window: usize,
}

/// Sample a next token from logits using repetition penalty, temperature,
/// top-k, and top-p filtering.
///
/// Modifies `logits` in-place (applies penalty, temperature scaling, softmax).
/// `indices_buf` is a reusable scratch buffer (len >= vocab_size) to avoid
/// per-token heap allocation. Pass `None` to allocate internally.
pub fn sample(
    logits: &mut [f32],
    recent_tokens: &[u32],
    vocab_size: usize,
    config: &SamplingConfig,
    indices_buf: Option<&mut Vec<usize>>,
) -> u32 {
    // 1. Repetition Penalty
    let start_idx = recent_tokens.len().saturating_sub(config.repetition_window);
    for &token_id in &recent_tokens[start_idx..] {
        let token_id = token_id as usize;
        if token_id < vocab_size {
            let logit = &mut logits[token_id];
            if *logit < 0.0 {
                *logit *= config.repetition_penalty;
            } else {
                *logit /= config.repetition_penalty;
            }
        }
    }

    // 2. Temperature
    if config.temperature == 0.0 {
        // Greedy
        return logits
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(idx, _)| idx as u32)
            .unwrap();
    }

    for l in logits.iter_mut() {
        *l /= config.temperature;
    }

    // Softmax
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let mut exp_sum = 0.0;
    for l in logits.iter_mut() {
        *l = (*l - max_logit).exp();
        exp_sum += *l;
    }
    for l in logits.iter_mut() {
        *l /= exp_sum;
    }

    // 3. Top-K — partial sort O(n) instead of full sort O(n log n)
    let top_k = config.top_k.min(vocab_size).max(1);
    let n = logits.len();

    // Reuse pre-allocated indices buffer or allocate fresh
    let mut owned_buf;
    let indices = match indices_buf {
        Some(buf) => {
            buf.clear();
            buf.extend(0..n);
            buf
        }
        None => {
            owned_buf = (0..n).collect::<Vec<usize>>();
            &mut owned_buf
        }
    };

    if top_k < n {
        indices.select_nth_unstable_by(top_k, |&a, &b| logits[b].total_cmp(&logits[a]));
        indices.truncate(top_k);
    }
    // Sort only the top-k (40 elements) for top-p cumulative sum
    indices.sort_unstable_by(|&a, &b| logits[b].total_cmp(&logits[a]));

    // 4. Top-P
    let mut cumulative_prob = 0.0;
    let mut cutoff_index = indices.len();

    for (i, &idx) in indices.iter().enumerate() {
        cumulative_prob += logits[idx];
        if cumulative_prob > config.top_p {
            cutoff_index = i + 1;
            break;
        }
    }
    let valid = &indices[..cutoff_index];

    // 5. Sample
    let mut rng = rand::rng();
    let r: f32 = rand::Rng::random(&mut rng); // [0, 1)

    // Normalize probabilities of valid indices
    let mut prob_sum = 0.0;
    for &idx in valid {
        prob_sum += logits[idx];
    }

    let mut thread_r = r * prob_sum;
    for &idx in valid {
        thread_r -= logits[idx];
        if thread_r <= 0.0 {
            return idx as u32;
        }
    }

    valid.first().copied().unwrap_or(0) as u32
}

/// Compute log P(token_id) from raw logits using numerically stable log-softmax.
///
/// Uses f64 arithmetic to avoid precision loss when accumulating over many tokens.
/// Formula: log_softmax(x)_i = (x_i - max) - log(sum(exp(x_j - max)))
pub fn compute_log_prob(logits: &[f32], token_id: u32, vocab_size: usize) -> f64 {
    let logits = &logits[..vocab_size];
    let max_logit = logits.iter().fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let log_sum_exp: f64 = logits
        .iter()
        .map(|&x| ((x - max_logit) as f64).exp())
        .sum::<f64>()
        .ln()
        + max_logit as f64;
    logits[token_id as usize] as f64 - log_sum_exp
}

#[cfg(test)]
mod tests {
    use super::*;

    fn default_config() -> SamplingConfig {
        SamplingConfig {
            temperature: 0.0,
            top_p: 0.9,
            top_k: 40,
            repetition_penalty: 1.1,
            repetition_window: 64,
        }
    }

    #[test]
    fn test_greedy_sampling() {
        let mut logits = vec![0.1, 0.5, 0.3, 0.9, 0.2];
        let config = default_config(); // temp=0.0 → greedy
        let token = sample(&mut logits, &[], 5, &config, None);
        assert_eq!(token, 3); // highest logit at index 3
    }

    #[test]
    fn test_repetition_penalty_reduces_repeated() {
        let mut logits = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let config = SamplingConfig {
            repetition_penalty: 2.0,
            ..default_config()
        };
        // Token 4 (logit=5.0, positive) should be divided by 2.0 → 2.5
        // Token 3 (logit=4.0) stays → should win greedy
        let token = sample(&mut logits, &[4], 5, &config, None);
        assert_eq!(token, 3);
    }

    #[test]
    fn test_greedy_with_negative_logits() {
        let mut logits = vec![-5.0, -1.0, -3.0];
        let config = default_config();
        let token = sample(&mut logits, &[], 3, &config, None);
        assert_eq!(token, 1); // -1.0 is the highest
    }

    #[test]
    fn test_compute_log_prob_uniform() {
        // Uniform logits → each token has prob 1/4 → log(1/4) ≈ -1.386
        let logits = vec![1.0, 1.0, 1.0, 1.0];
        let lp = compute_log_prob(&logits, 0, 4);
        assert!((lp - (-4.0f64.ln())).abs() < 1e-10);
    }

    #[test]
    fn test_compute_log_prob_peaked() {
        // One very high logit should get log_prob close to 0
        let logits = vec![100.0, 0.0, 0.0, 0.0];
        let lp = compute_log_prob(&logits, 0, 4);
        assert!(lp > -1e-10); // close to 0
        // The other tokens should have very negative log_prob
        let lp_other = compute_log_prob(&logits, 1, 4);
        assert!(lp_other < -90.0);
    }

    #[test]
    fn test_compute_log_prob_sums_to_one() {
        let logits = vec![2.0, 1.0, 0.5, 3.0, -1.0];
        let total: f64 = (0..5)
            .map(|i| compute_log_prob(&logits, i as u32, 5).exp())
            .sum();
        assert!((total - 1.0).abs() < 1e-10);
    }
}
