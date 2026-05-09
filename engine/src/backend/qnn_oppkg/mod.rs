//! QNN OpPackage backend (M3.1 skeleton).
//!
//! Spec: `spec/30-engine.md` 부록 C (ENG-QNN-201~240),
//! `spec/41-invariants.md` §3.24 (INV-166~180),
//! `arch/30-engine.md` §18.
//!
//! M3.1 산출물 — Backend trait 모든 필수 method를 stub으로 구현하여
//! `cargo build --features qnn,opencl`이 PASS하도록 한다. 실제 forward는
//! 모두 `unimplemented!("M3.3에서 구현")`으로 marker.
//!
//! 책임 분리:
//! - `runtime` — `libQnnGpu.so` + `libqnn_oppkg.so` dlopen + V2.0 fn-ptr
//! - `memory`  — rpcmem(DMA-BUF heap) allocator (M3.2 본격 도입)
//! - `buffer`  — host_ptr + qnn_mem_handle pair (M3.2 본격 도입)
//! - `graph_cache` — `Vec<Arc<LayerGraph>>` 28 slot (M3.2 prebuild 진입)
//! - `layer_graph` — M2.H 14-node graph wrapper (M3.2 이식 대상)
//!
//! # 호스트(non-Android) 빌드
//! `cargo build --features qnn,opencl`은 PASS이지만 `QnnOppkgBackend::new()`는
//! 명확한 `Err`로 fail한다 (libQnnGpu.so 부재). 디바이스 빌드 + Android
//! runtime에서만 model load까지 진행 가능하며, forward는 M3.3 진입 후 정상
//! 동작한다.

pub mod buffer;
pub mod graph_cache;
pub mod layer_graph;
pub mod memory;
pub mod runtime;

use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use anyhow::{Result, anyhow};
use std::sync::{Arc, Mutex};

use self::graph_cache::GraphCache;
use self::runtime::QnnOppkgRuntime;

/// QNN OpPackage backend — `Backend` trait 구현 진입점.
///
/// ENG-QNN-201/INV-166: 모든 필수 trait method를 OpenCL과 동일 시그니처로
/// 구현한다. 본 단계 (M3.1)에서는 trivial method만 정상 구현하고 forward는
/// 모두 `unimplemented!`. M3.3에서 graph fast path (`execute_layer_graph`)가
/// 본격 구현되면 trait fallback method (matmul 등)는 호출되지 않아야 한다
/// (INV-175).
///
/// thread-safety: runtime + graph cache는 `Arc`로 공유하며, M3.2에서 graph
/// cache build/invalidate 시점에만 `Mutex`가 필요할 가능성이 있다.
pub struct QnnOppkgBackend {
    /// `libQnnGpu.so` + OpPackage runtime 초기화 결과 (M3.1 stub Err).
    #[allow(dead_code)]
    runtime: Arc<QnnOppkgRuntime>,
    /// 28 layer graph cache (M3.2 prebuild 진입).
    #[allow(dead_code)]
    graph_cache: Mutex<GraphCache>,
}

impl QnnOppkgBackend {
    /// Backend 초기화 — `libQnnGpu.so` dlopen + `libqnn_oppkg.so`
    /// register + V2.0 fn-pointer 캐싱. 호스트(non-Android)에서는 명확한
    /// `Err`로 실패한다 (디바이스 빌드에서만 진행 가능).
    ///
    /// `_args`는 generate.rs Args struct를 향후 받기 위한 placeholder. M3.1
    /// 단계에서는 사용하지 않는다 (`--qnn-graph-cache-prebuild`는 M3.2 진입
    /// 시점에 의미가 생긴다).
    pub fn new() -> Result<Self> {
        let runtime = QnnOppkgRuntime::init()?;
        Ok(Self {
            runtime,
            graph_cache: Mutex::new(GraphCache::new()),
        })
    }
}

/// Forward 미구현 marker — M3.3 진입 시 graph fast path로 모두 대체된다.
fn forward_unimplemented(method: &'static str) -> ! {
    unimplemented!(
        "QnnOppkgBackend::{} — M3.3에서 graph fast path 도입 시 fallback 경로로만 호출",
        method
    )
}

impl Backend for QnnOppkgBackend {
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn name(&self) -> &str {
        "QNN OpPackage (Adreno GPU)"
    }

    fn device(&self) -> &str {
        "QNN-GPU"
    }

    fn is_gpu(&self) -> bool {
        true
    }

    fn is_discrete_gpu(&self) -> bool {
        false
    }

    fn synchronize(&self) -> Result<()> {
        // M3.3에서 graph executor 완료 대기 도입. 본 단계는 noop.
        Ok(())
    }

    fn flush(&self) -> Result<()> {
        Ok(())
    }

    /// ENG-QNN-211 / INV-174 — backend는 layer graph fast path를 지원한다고
    /// 선언한다. transformer.rs forward 진입 시 본 method가 1회 호출되어
    /// fast path 분기를 결정한다. idempotent (M3.1 stub 단계는 항상 true).
    fn supports_layer_graph(&self) -> bool {
        true
    }

    /// ENG-QNN-211/213/214 — 14-node single-layer graph dispatch.
    /// M3.3 진입 시 본격 구현.
    #[allow(unused_variables)]
    fn execute_layer_graph(
        &self,
        layer_idx: usize,
        x: &Tensor,
        kv_cache_k: &mut Tensor,
        kv_cache_v: &mut Tensor,
        pos: usize,
        x_out: &mut Tensor,
    ) -> Result<()> {
        // M3.3에서 self.graph_cache.lock()....get(layer_idx).execute(...) 본격 구현.
        Err(anyhow!(
            "execute_layer_graph(layer={layer_idx}, pos={pos}) — M3.3에서 구현"
        ))
    }

    // ── Below: trait method stubs. 모두 M3.3 fast path 도입 후 호출되지 않아야 한다 (INV-175).
    //    M3.1 단계는 호스트 빌드/링크만 게이트이므로 unimplemented! 마커로 충분.

    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> {
        forward_unimplemented("matmul")
    }

    fn matmul_transposed(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> {
        forward_unimplemented("matmul_transposed")
    }

    fn matmul_slice(
        &self,
        _a: &Tensor,
        _b: &Tensor,
        _rows: usize,
        _cols: usize,
        _out: &mut Tensor,
    ) -> Result<()> {
        forward_unimplemented("matmul_slice")
    }

    fn add_assign(&self, _a: &mut Tensor, _b: &Tensor) -> Result<()> {
        forward_unimplemented("add_assign")
    }

    fn scale(&self, _x: &mut Tensor, _v: f32) -> Result<()> {
        forward_unimplemented("scale")
    }

    fn silu_mul(&self, _a: &mut Tensor, _b: &Tensor) -> Result<()> {
        forward_unimplemented("silu_mul")
    }

    fn rms_norm(&self, _x: &mut Tensor, _w: &Tensor, _eps: f32, _add_unit: bool) -> Result<()> {
        forward_unimplemented("rms_norm")
    }

    fn rms_norm_oop(
        &self,
        _x: &Tensor,
        _out: &mut Tensor,
        _w: &Tensor,
        _eps: f32,
        _add_unit: bool,
    ) -> Result<()> {
        forward_unimplemented("rms_norm_oop")
    }

    fn softmax(&self, _x: &mut Tensor) -> Result<()> {
        forward_unimplemented("softmax")
    }

    fn rope_inplace(&self, _x: &mut Tensor, _start_pos: usize, _theta: f32) -> Result<()> {
        forward_unimplemented("rope_inplace")
    }

    fn copy_from(&self, _t: &Tensor) -> Result<Tensor> {
        forward_unimplemented("copy_from")
    }

    fn cast(&self, _src: &Tensor, _dst: &mut Tensor) -> Result<()> {
        forward_unimplemented("cast")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 호스트(non-Android, libQnnGpu 부재)에서는 `new()`가 명확한 Err로 실패해야
    /// 한다 (INV-180 sanity). 디바이스 빌드/Android runtime에서만 model load
    /// 까지 진행 가능.
    #[test]
    fn host_new_returns_err_without_device() {
        let r = QnnOppkgBackend::new();
        assert!(
            r.is_err(),
            "host (non-Android) build에서는 libQnnGpu.so 부재로 init이 실패해야 한다"
        );
    }
}
