//! Single-layer 14-node QNN graph wrapper (M3.2).
//!
//! Spec: `spec/30-engine.md` 부록 C.4 (ENG-QNN-221~230, INV-176~177),
//! `arch/30-engine.md` §18.5~§18.6.
//!
//! ## DAG topology (M2.H 검증, `crates/qnn_oppkg::graph::layer` re-export)
//!
//! 14-node Qwen2.5-1.5B 1-layer graph:
//! `RmsNorm(pre) → Q/K/V matmul (3) → RoPE(Q,K) (2) → KvScatter →
//! FlashAttn → O proj → Add (residual #1) → RmsNorm(post) → gate/up matmul (2)
//! → SiluMul → down proj → Add (residual #2)` = 14 nodes.
//!
//! ## 본 단계 (M3.2) 적용 범위
//! - struct + `LayerGraph::build` / `LayerGraph::execute` signature
//! - `LayerConfig` re-export (`crates/qnn_oppkg::graph::layer::LayerConfig`)
//! - `LAYER_NODE_COUNT == 14` build-time const sanity
//! - host 빌드: build/execute 모두 명확한 Err (디바이스 빌드에서만 진행)
//! - device 빌드: M2.H의 graph build 본문은 M3.3에서 본격 이식 — 본 단계는
//!   signature + path 검증만
//!
//! M2 핵심 결정 (M3.3 본격 이식 시 보존 필수):
//! - FlashAttn sinks `flat_dims: vec![n_head]` (M2.H lines ~2087-2168)
//! - RoPE OOP (`kernel_rope_simple_oop`, simple_ops.cl 추가됨)
//! - SiluMul `OutputTensorAliased` (M2 INV-164)
//! - KvScatter multi-output (k_rot → KV_K, v → KV_V)
//! - Q4_0 weight `RawBytes(18, N/32)` (M2 INV-165)

use crate::backend::qnn_oppkg::runtime::QnnOppkgRuntime;
use crate::layers::transformer_layer::TransformerLayer;
use anyhow::{Result, anyhow};

// host metadata only — qnn_oppkg crate에서 LayerConfig + LAYER_NODE_COUNT를
// 직접 사용. 본 type은 ffi 의존성 없이 host build에서도 컴파일된다.
pub use qnn_oppkg::graph::layer::{LAYER_INTERMEDIATE_COUNT, LAYER_NODE_COUNT, LayerConfig};

/// `graphFinalize` 1회 호출 budget (ms). ENG-QNN-209/INV-167.
pub const FINALIZE_BUDGET_MS: u32 = 200;

/// 14-node single-layer QNN graph (M2.H 이식).
///
/// 본 struct는 model load 시점에 `build()` 1회로 채워지고 process lifetime
/// 동안 재사용된다 (INV-167). M3.3 dispatch path는 `execute()`로 1 layer를
/// 1회 graphExecute에 dispatch한다.
///
/// 본 단계 (M3.2)는 struct + signature + cache lifecycle 골격만 마련. graph
/// handle / weight handle / KV/IO handle 본격 채움은 M3.3에서 device 측 본문을
/// 이식할 때 수행된다.
pub struct LayerGraph {
    /// Layer index (0..n_layers).
    pub layer_idx: usize,
    /// `graphFinalize` 측정값 (ms). M3.2 build path가 기록한다 (host 빌드는 0).
    pub finalize_ms: u32,
    /// 본 graph가 보유한 weight handle 개수 (Q4_0 7개 + norm 2개 = 9개 baseline).
    /// M3.3 본문 진입 시 `Vec<u64>`로 교체 (qnn_mem_handle 목록).
    pub weight_handle_count: usize,
}

impl LayerGraph {
    /// 1-layer graph build — model load 시점 1회 호출.
    ///
    /// - device path (Android): `crates/qnn_oppkg::graph::layer::build_layer_graph`
    ///   호출 후 `graphFinalize` 측정 (M3.3에서 본격).
    /// - host path: build 자체가 의미 없으므로 명확한 Err.
    ///
    /// `runtime`은 `QnnOppkgBackend::new()`에서 생성된 dlopen handle.
    /// `weights`는 `LayerSlot::load_weights()` snapshot. M3.3에서 weights의
    /// raw bytes를 rpcmem-backed buffer로 복사 (또는 mmap이 이미 rpcmem이면
    /// 직접 share)하여 graph weight handle로 baked.
    /// `cfg`는 layer dimension metadata (ENG-QNN-225 KV layout).
    pub fn build(
        runtime: &QnnOppkgRuntime,
        layer_idx: usize,
        weights: &TransformerLayer,
        cfg: &LayerConfig,
    ) -> Result<Self> {
        // unused에서 빠져나가도록 use only — host 빌드에서는 본문에 진입하지 않음.
        let _ = (runtime, weights, cfg);

        #[cfg(target_os = "android")]
        {
            // M3.3 device 본문: M2.H `microbench_qnn_qwen_layer.rs` lines
            // 1629-2184의 graph build 시퀀스를 production helper로 이식.
            //
            // 핵심 호출 시퀀스:
            //   1. graphCreate(ctx, name, configs) → graph_handle
            //   2. weight tensors RawBytes(18, N/32) 등록 — Q/K/V/O/gate/up/down 7개
            //      (INV-165 Q4_0 layout 보존)
            //   3. RmsNorm pre/post weight 2개 등록
            //   4. KV cache K/V 2개 등록 (rpcmem-backed, INV-171)
            //   5. x_in / x_out F32 [1, 1, dim] 등록 (APP_WRITE / APP_READ)
            //   6. mask buffer 등록 (ENG-QNN-228)
            //   7. 14× graphAddNode (FlashAttn sinks `flat_dims: vec![n_head]` 필수)
            //   8. graphFinalize (timed) — INV-167 ≤ 200ms
            //
            // 본 단계는 컴파일 게이트만이므로 즉시 Err — M3.3 진입 시 본문 이식.
            return Err(anyhow!(
                "LayerGraph::build (android, layer={layer_idx}) — M3.3에서 M2.H graph builder 이식 후 본격 동작"
            ));
        }
        #[cfg(not(target_os = "android"))]
        {
            // host 빌드: graph build는 의미 없음. caller가 호출하지 않도록
            // `prebuild_graph_cache`가 host_init Err에서 차단되지만, signature
            // 게이트로 추가 방어.
            Err(anyhow!(
                "LayerGraph::build (host, layer={layer_idx}) — host build에서는 graph 빌드 불가. 디바이스 빌드 + Android runtime에서 M3.3 진입 후 동작."
            ))
        }
    }

    /// 1-layer graph dispatch — token decode 시점 layer당 1회 호출 (M3.3).
    ///
    /// - x_in_bytes: F32 `[1, 1, dim]` host pointer (rpcmem-backed buffer 권장,
    ///   현재 stub은 `&[u8]` 이지만 M3.3에서는 backend가 `Tensor` → buffer host_ptr
    ///   접근으로 대체).
    /// - kv_caches: (K, V) 두 buffer의 host pointer + qnn_mem_handle.
    /// - pos: 매 forward call마다 graph의 scalar arg로 갱신 (ENG-QNN-227).
    /// - mask_bytes: `[2048] f32` 또는 `[2048] f16` mask buffer (ENG-QNN-228).
    /// - x_out_bytes: F32 `[1, 1, dim]` host pointer (APP_READ).
    ///
    /// 본 method는 M3.2 단계에서는 stub — Err 반환. M3.3에서 M2.H graphExecute
    /// 호출 본체를 이식한다.
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        runtime: &QnnOppkgRuntime,
        x_in_bytes: &[u8],
        kv_k_bytes: &mut [u8],
        kv_v_bytes: &mut [u8],
        pos: usize,
        mask_bytes: &[u8],
        x_out_bytes: &mut [u8],
    ) -> Result<()> {
        let _ = (
            runtime,
            x_in_bytes,
            kv_k_bytes,
            kv_v_bytes,
            pos,
            mask_bytes,
            x_out_bytes,
        );
        Err(anyhow!(
            "LayerGraph::execute (layer={}, pos={pos}) — M3.3에서 graphExecute 본문 이식 후 동작",
            self.layer_idx
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// INV-176: build-time const synchronization.
    /// `crates/qnn_oppkg::graph::layer::LAYER_NODE_COUNT`가 14인지를 본 path
    /// 에서도 직접 검증. 추후 phase analyzer (M4.0) const table과 동기화 강제.
    #[test]
    fn layer_node_count_is_14() {
        assert_eq!(LAYER_NODE_COUNT, 14);
        assert_eq!(LAYER_INTERMEDIATE_COUNT, 13);
    }

    /// INV-176: layer config Qwen2.5-1.5B 기본값 정합. dim=1536, n_head=12,
    /// n_kv_heads=2, head_dim=128, ffn_dim=8960, kv_capacity=2048.
    #[test]
    fn layer_config_qwen2p5_1p5b_defaults() {
        let cfg = LayerConfig::qwen2p5_1p5b();
        assert_eq!(cfg.dim, 1536);
        assert_eq!(cfg.n_head, 12);
        assert_eq!(cfg.n_kv_heads, 2);
        assert_eq!(cfg.head_dim, 128);
        assert_eq!(cfg.ffn_dim, 8960);
        assert_eq!(cfg.kv_capacity, 2048);
    }

    /// FINALIZE_BUDGET_MS는 INV-167과 동기화 (200 ms/layer).
    #[test]
    fn finalize_budget_matches_inv167() {
        assert_eq!(FINALIZE_BUDGET_MS, 200);
    }
}
