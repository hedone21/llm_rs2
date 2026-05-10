//! Qwen 2.5-1.5B 1-layer DAG metadata for QNN OpPackage (M2.H).
//!
//! The full graph build (`graphCreate` / `graphAddNode` / `graphFinalize`) is
//! performed on the device side by
//! `engine/src/bin/microbench_qnn_qwen_layer.rs` — keeping it out of the
//! `qnn_oppkg` crate preserves the (INV-151) "no extra crate dependencies"
//! invariant. This module exposes the dimensions and the declared node /
//! intermediate counts so host unit tests can verify the topology assumption
//! without linking against `Qnn_*` bindings.
//!
//! ## Qwen 2.5-1.5B layer DAG (14 nodes)
//!
//! ```text
//! x_in (F32, [1, 1, dim])
//!    │
//!    └─→ RmsNorm(pre)                           → y1
//!         ├─→ MatMulQ40F32(Q proj)              → q
//!         ├─→ MatMulQ40F32(K proj)              → k
//!         └─→ MatMulQ40F32(V proj)              → v
//!
//! q ─→ RoPE(Q)                                  → q_rot
//! k ─→ RoPE(K)                                  → k_rot
//!
//! (k_rot, v) ─→ KvScatter(write_pos = pos)      → k_cache, v_cache (alias)
//!
//! (q_rot, k_cache, v_cache) ─→ FlashAttn        → attn_out
//!
//! attn_out ─→ MatMulQ40F32(O proj)              → o
//!
//! (o, x_in)  ─→ Add (residual #1)               → x_attn
//!
//! x_attn ─→ RmsNorm(post)                       → y2
//!         ├─→ MatMulQ40F32(gate)                → gate
//!         └─→ MatMulQ40F32(up)                  → up
//!
//! (gate, up) ─→ SiluMul (in-place on gate)      → silu_out
//!
//! silu_out ─→ MatMulQ40F32(down)                → down
//!
//! (down, x_attn) ─→ Add (residual #2)           → x_out
//! ```

/// Total number of `graphAddNode` calls expected to assemble one Qwen layer.
pub const LAYER_NODE_COUNT: usize = 14;

/// Number of intermediate (`QNN_TENSOR_TYPE_NATIVE`) tensors carrying outputs
/// between the 14 nodes.
///
/// Excludes graph endpoints (`x_in` APP_WRITE, `x_out` APP_READ) and weights /
/// KV cache (APP_WRITE memhandles registered against external rpcmem buffers).
///
/// Intermediate breakdown:
///   y1, q, k, v, q_rot, k_rot, attn_out, o, x_attn, y2, gate, up, silu_out
/// = 13 NATIVE tensors. (`silu_out` aliases `gate`'s buffer via M2.G
/// `OutputTensorAliased` but is still a distinct `Qnn_Tensor_t` in the graph
/// representation.)
pub const LAYER_INTERMEDIATE_COUNT: usize = 13;

/// Per-layer dimensions. Defaults match Qwen 2.5-1.5B (16 layers, used here
/// for the 1-layer microbench).
#[derive(Clone, Copy, Debug)]
pub struct LayerConfig {
    /// Hidden dim. Qwen 2.5-1.5B: 1536.
    pub dim: u32,
    /// Query heads. Qwen 2.5-1.5B: 12.
    pub n_head: u32,
    /// KV heads (GQA). Qwen 2.5-1.5B: 2.
    pub n_kv_heads: u32,
    /// Per-head dim. Qwen 2.5-1.5B: 128.
    pub head_dim: u32,
    /// FFN inner dim. Qwen 2.5-1.5B: 8960.
    pub ffn_dim: u32,
    /// Max context (`max_position_embeddings`). Qwen 2.5-1.5B: 2048.
    pub kv_capacity: u32,
}

impl LayerConfig {
    /// Qwen 2.5-1.5B reference dimensions.
    pub const fn qwen2p5_1p5b() -> Self {
        Self {
            dim: 1536,
            n_head: 12,
            n_kv_heads: 2,
            head_dim: 128,
            ffn_dim: 8960,
            kv_capacity: 2048,
        }
    }

    /// `n_head * head_dim` — the QKV projection out_dim for Q.
    pub const fn q_proj_out(&self) -> u32 {
        self.n_head * self.head_dim
    }

    /// `n_kv_heads * head_dim` — the KV projection out_dim.
    pub const fn kv_proj_out(&self) -> u32 {
        self.n_kv_heads * self.head_dim
    }

    /// Bytes for one `Q4_0` weight matrix `[N, K]` packed in SOA layout —
    /// `q_bytes` (4-bit packed quants) and `d_bytes` (FP16 per-block scales).
    /// Returns `(q_bytes, d_bytes)`. `K` must be a multiple of 32.
    pub const fn q40_bytes(n: u32, k: u32) -> (u64, u64) {
        const QK4_0: u32 = 32;
        const QS_PER_BLOCK: u32 = 16;
        let num_blocks = (n as u64) * (k as u64) / (QK4_0 as u64);
        let q_bytes = num_blocks * (QS_PER_BLOCK as u64);
        let d_bytes = num_blocks * 2;
        (q_bytes, d_bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn layer_config_qwen2p5_defaults() {
        let cfg = LayerConfig::qwen2p5_1p5b();
        assert_eq!(cfg.dim, 1536);
        assert_eq!(cfg.n_head, 12);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.ffn_dim, 8960);
        assert_eq!(cfg.kv_capacity, 2048);
    }

    #[test]
    fn q_proj_out_matches_n_head_times_head_dim() {
        let cfg = LayerConfig::qwen2p5_1p5b();
        // Qwen2.5-1.5B: 12 * 128 = 1536 (== dim, isotropic).
        assert_eq!(cfg.q_proj_out(), 1536);
    }

    #[test]
    fn kv_proj_out_matches_n_kv_heads_times_head_dim() {
        let cfg = LayerConfig::qwen2p5_1p5b();
        // Qwen2.5-1.5B GQA 6:1 → 2 * 128 = 256.
        assert_eq!(cfg.kv_proj_out(), 256);
    }

    #[test]
    fn layer_dag_node_count_is_14() {
        // 1 norm + 3 proj + 2 rope + 1 scatter + 1 fa + 1 oproj + 1 add
        // + 1 norm + 2 ffn_proj + 1 silu + 1 down + 1 add = 14
        assert_eq!(LAYER_NODE_COUNT, 14);
    }

    #[test]
    fn layer_intermediate_count_is_13() {
        // y1, q, k, v, q_rot, k_rot, attn_out, o, x_attn, y2, gate, up, silu_out
        assert_eq!(LAYER_INTERMEDIATE_COUNT, 13);
    }

    #[test]
    fn q40_bytes_matches_known_qwen_dims() {
        // O proj: N=1536, K=1536  → num_blocks = 1536*1536/32 = 73728
        //   q = 73728 * 16 = 1_179_648 bytes
        //   d = 73728 * 2  =   147_456 bytes
        let (q, d) = LayerConfig::q40_bytes(1536, 1536);
        assert_eq!(q, 1_179_648);
        assert_eq!(d, 147_456);

        // gate / up: N=8960, K=1536
        let (q, d) = LayerConfig::q40_bytes(8960, 1536);
        assert_eq!(q, 8960 * 1536 / 32 * 16);
        assert_eq!(d, 8960 * 1536 / 32 * 2);
    }
}
