//! KIVI fused attention capability (§3.3).
//!
//! Phase α-W-4 에서 `backend.rs` 의 `KiviAttentionBackend` trait 정의를 이리로 이동했다.
//! `backend.rs` 는 본 정의를 re-export 하므로 기존 `crate::backend::KiviAttentionBackend`
//! import path 가 그대로 동작한다(byte-identical).

use crate::tensor::Tensor;
use anyhow::Result;

/// §13.8-L S-L-3 — KIVI native attention dispatch 추상화.
///
/// KIVI Q2/Q4/Q8 quantized KV cache 의 fused attention + residual update
/// 커널은 현재 OpenCL backend 만 보유합니다. forward_gen 및 KiviCache
/// 가 OpenCL backend 의 inherent 메서드 4종을 호출하기 위해 사용하던
/// downcast 패턴을 본 trait method 호출로 통합합니다.
///
/// `is_nosub_device` 는 KIVI-specific 한 분기에서 sub-group reduce 비
/// 가용 device (예: Adreno) 를 식별. OpenCL 외 backend 는 `false` (기본
/// 무관) 또는 trait 미구현 (`Backend::as_kivi_attention` 이 `None`).
pub trait KiviAttentionBackend: Send + Sync {
    /// KIVI fused attention 커널이 해당 bit-width 로 컴파일되어 있는지.
    fn has_kivi_attn_kernel(&self, bits: u8) -> bool;

    /// Sub-group reduce 가 비활성인 device 인지 (KIVI native attention 분기).
    fn is_nosub_device(&self) -> bool;

    /// KIVI fused attention dispatch. residual K/V + quantized K/V 를
    /// 한 커널 안에서 dequantize + attention.
    #[allow(clippy::too_many_arguments)]
    fn attention_gen_kivi(
        &self,
        q: &Tensor,
        qk_buf: &Tensor,
        qv_buf: &Tensor,
        res_k: &Tensor,
        res_v: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        q_tokens: usize,
        res_tokens: usize,
        res_cap: usize,
        scale: f32,
        scores_out: Option<&mut [f32]>,
        bits: u8,
    ) -> Result<()>;

    /// KV residual gather + scatter update. 다음 token 의 K/V 를 residual
    /// circular buffer 에 누적.
    #[allow(clippy::too_many_arguments)]
    fn kivi_gather_update(
        &self,
        input: &Tensor,
        residual: &mut Tensor,
        kv_heads: usize,
        res_cap: usize,
        head_dim: usize,
        seq_len: usize,
        res_pos: usize,
    ) -> Result<()>;
}
