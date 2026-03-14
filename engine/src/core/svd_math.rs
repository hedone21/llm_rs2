//! SVD linear algebra for KV cache compression.
//!
//! Ported from compress_lab's f64 implementation to f32 for inference pipeline use.
//! Computes rank-k approximation via power iteration on A^T A (Gram matrix).

/// Compute top-k eigenvalues/eigenvectors of a symmetric matrix via power iteration + deflation.
///
/// Input: `ata` is a flattened `n × n` symmetric matrix (row-major).
/// Returns `(eigenvalues, eigenvectors)` sorted by eigenvalue descending.
/// Each eigenvector is stored as a contiguous `[n]` slice within the returned Vec.
#[allow(clippy::needless_range_loop)]
pub fn svd_eigen_f32(ata: &[f32], n: usize, max_k: usize) -> (Vec<f32>, Vec<f32>) {
    debug_assert_eq!(ata.len(), n * n);
    let mut eigenvalues = Vec::with_capacity(max_k);
    // eigenvectors: flattened [k × n]
    let mut eigenvectors = Vec::with_capacity(max_k * n);
    let mut mat = ata.to_vec();

    for _ in 0..max_k {
        // Power iteration for largest eigenvalue
        let inv_sqrt_n = 1.0 / (n as f32).sqrt();
        let mut v = vec![inv_sqrt_n; n];
        for _ in 0..100 {
            let mut new_v = vec![0.0f32; n];
            for i in 0..n {
                let row = &mat[i * n..(i + 1) * n];
                let mut sum = 0.0f32;
                for j in 0..n {
                    sum += row[j] * v[j];
                }
                new_v[i] = sum;
            }
            let norm: f32 = new_v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm < 1e-10 {
                return (eigenvalues, eigenvectors);
            }
            let inv_norm = 1.0 / norm;
            for x in &mut new_v {
                *x *= inv_norm;
            }
            v = new_v;
        }

        // Eigenvalue = v^T A v
        let mut av = vec![0.0f32; n];
        for i in 0..n {
            let row = &mat[i * n..(i + 1) * n];
            let mut sum = 0.0f32;
            for j in 0..n {
                sum += row[j] * v[j];
            }
            av[i] = sum;
        }
        let eigenval: f32 = v.iter().zip(av.iter()).map(|(a, b)| a * b).sum();
        if eigenval < 1e-6 {
            break;
        }
        eigenvalues.push(eigenval);
        eigenvectors.extend_from_slice(&v);

        // Deflate: A = A - λ v v^T
        for i in 0..n {
            for j in 0..n {
                mat[i * n + j] -= eigenval * v[i] * v[j];
            }
        }
    }

    (eigenvalues, eigenvectors)
}

/// Compute A^T A (Gram matrix) from a set of row vectors.
///
/// `data`: flattened `[num_rows × dim]` row-major.
/// Returns flattened `[dim × dim]` symmetric matrix.
#[allow(clippy::needless_range_loop)]
pub fn compute_gram_matrix(data: &[f32], num_rows: usize, dim: usize) -> Vec<f32> {
    debug_assert_eq!(data.len(), num_rows * dim);
    let mut ata = vec![0.0f32; dim * dim];
    for t in 0..num_rows {
        let row = &data[t * dim..(t + 1) * dim];
        for i in 0..dim {
            let vi = row[i];
            // Only compute upper triangle, then symmetrize
            for j in i..dim {
                ata[i * dim + j] += vi * row[j];
            }
        }
    }
    // Symmetrize
    for i in 0..dim {
        for j in 0..i {
            ata[i * dim + j] = ata[j * dim + i];
        }
    }
    ata
}

/// Project a single token onto the basis: coeffs[j] = dot(token, basis[j]).
///
/// `token`: `[dim]` values.
/// `basis`: flattened `[k × dim]`.
/// Returns `[k]` coefficients.
pub fn project_token(token: &[f32], basis: &[f32], k: usize, dim: usize) -> Vec<f32> {
    debug_assert_eq!(token.len(), dim);
    debug_assert_eq!(basis.len(), k * dim);
    let mut coeffs = Vec::with_capacity(k);
    for j in 0..k {
        let basis_row = &basis[j * dim..(j + 1) * dim];
        let dot: f32 = token.iter().zip(basis_row).map(|(a, b)| a * b).sum();
        coeffs.push(dot);
    }
    coeffs
}

/// Reconstruct tokens from coefficients × basis into an output buffer.
///
/// Reconstructs `token_count` tokens starting from `token_start` for a single head.
/// Output layout: SeqMajor `[max_seq × kv_heads × head_dim]`.
///
/// `basis`: flattened `[k × head_dim]` for this head.
/// `coeffs`: flattened `[compressed_tokens × k]` for this head.
#[allow(clippy::needless_range_loop, clippy::too_many_arguments)]
pub fn reconstruct_into(
    basis: &[f32],
    coeffs: &[f32],
    token_start: usize,
    token_count: usize,
    k: usize,
    head_dim: usize,
    kv_heads: usize,
    head_idx: usize,
    out: &mut [f32],
) {
    debug_assert_eq!(basis.len(), k * head_dim);
    for t in 0..token_count {
        let coeff_row = &coeffs[(token_start + t) * k..(token_start + t + 1) * k];
        let pos = (token_start + t) * kv_heads * head_dim + head_idx * head_dim;
        // Zero first (since we accumulate)
        for d in 0..head_dim {
            out[pos + d] = 0.0;
        }
        for j in 0..k {
            let c = coeff_row[j];
            let basis_row = &basis[j * head_dim..(j + 1) * head_dim];
            for d in 0..head_dim {
                out[pos + d] += c * basis_row[d];
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_diagonal_eigendecomposition() {
        // diag(5, 3, 1): distinct eigenvalues → robust power iteration
        #[rustfmt::skip]
        let ata = vec![
            5.0, 0.0, 0.0,
            0.0, 3.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let (vals, vecs) = svd_eigen_f32(&ata, 3, 3);
        assert_eq!(vals.len(), 3, "expected 3 eigenvalues");
        assert!((vals[0] - 5.0).abs() < 0.01, "λ0 = {}, expected 5", vals[0]);
        assert!((vals[1] - 3.0).abs() < 0.01, "λ1 = {}, expected 3", vals[1]);
        assert!((vals[2] - 1.0).abs() < 0.01, "λ2 = {}, expected 1", vals[2]);
        // Each eigenvector should be unit length
        for i in 0..3 {
            let ev = &vecs[i * 3..(i + 1) * 3];
            let norm: f32 = ev.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 0.01, "eigenvector {i} norm = {norm}");
        }
    }

    #[test]
    fn test_known_matrix_eigenvalues() {
        // [[4, 2], [2, 1]] has eigenvalues 5 and 0
        #[rustfmt::skip]
        let ata = vec![4.0, 2.0, 2.0, 1.0];
        let (vals, vecs) = svd_eigen_f32(&ata, 2, 2);
        assert_eq!(vals.len(), 1); // second eigenvalue < threshold
        assert!(
            (vals[0] - 5.0).abs() < 0.01,
            "first eigenvalue should be 5, got {}",
            vals[0]
        );
        // Eigenvector for λ=5: proportional to [2, 1]
        let ev = &vecs[0..2];
        let ratio = ev[0] / ev[1];
        assert!(
            (ratio - 2.0).abs() < 0.1,
            "eigenvector ratio should be ~2, got {ratio}"
        );
    }

    #[test]
    fn test_project_reconstruct_roundtrip() {
        let dim = 8;
        let k = 3;
        let kv_heads = 1;
        let num_tokens = 5;

        // Create synthetic data: tokens that lie in a low-rank subspace
        let mut data = vec![0.0f32; num_tokens * dim];
        for t in 0..num_tokens {
            for d in 0..dim {
                // Use first 3 dimensions as the "real" data
                data[t * dim + d] = if d < k {
                    (t * dim + d) as f32 * 0.1
                } else {
                    0.0
                };
            }
        }

        // Compute basis via Gram matrix + eigendecomposition
        let ata = compute_gram_matrix(&data, num_tokens, dim);
        let (_, eigvecs) = svd_eigen_f32(&ata, dim, k);
        let actual_k = eigvecs.len() / dim;
        assert!(actual_k > 0);

        // Project all tokens
        let mut all_coeffs = Vec::new();
        for t in 0..num_tokens {
            let token = &data[t * dim..(t + 1) * dim];
            let c = project_token(token, &eigvecs, actual_k, dim);
            all_coeffs.extend_from_slice(&c);
        }

        // Reconstruct
        let mut out = vec![0.0f32; num_tokens * kv_heads * dim];
        reconstruct_into(
            &eigvecs[..actual_k * dim],
            &all_coeffs,
            0,
            num_tokens,
            actual_k,
            dim,
            kv_heads,
            0,
            &mut out,
        );

        // Check cosine similarity
        let mut dot = 0.0f32;
        let mut norm_a = 0.0f32;
        let mut norm_b = 0.0f32;
        for i in 0..data.len() {
            dot += data[i] * out[i];
            norm_a += data[i] * data[i];
            norm_b += out[i] * out[i];
        }
        let cos_sim = dot / (norm_a.sqrt() * norm_b.sqrt() + 1e-10);
        assert!(
            cos_sim > 0.999,
            "roundtrip cosine similarity = {cos_sim}, expected > 0.999"
        );
    }

    #[test]
    fn test_gram_matrix_symmetry() {
        let dim = 4;
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let ata = compute_gram_matrix(&data, 2, dim);
        for i in 0..dim {
            for j in 0..dim {
                assert!(
                    (ata[i * dim + j] - ata[j * dim + i]).abs() < 1e-6,
                    "Gram matrix not symmetric at ({i},{j})"
                );
            }
        }
    }

    #[test]
    fn test_project_token_basic() {
        // Basis = identity-like (first 2 of 4 dims)
        let dim = 4;
        let k = 2;
        #[rustfmt::skip]
        let basis = vec![
            1.0, 0.0, 0.0, 0.0,
            0.0, 1.0, 0.0, 0.0,
        ];
        let token = vec![3.0, 7.0, 1.0, 2.0];
        let coeffs = project_token(&token, &basis, k, dim);
        assert!((coeffs[0] - 3.0).abs() < 1e-6);
        assert!((coeffs[1] - 7.0).abs() < 1e-6);
    }
}
