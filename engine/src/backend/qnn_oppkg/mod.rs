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
use crate::models::weights::LayerSlot;
use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

use self::graph_cache::GraphCache;
use self::layer_graph::LayerConfig;
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
    /// `libQnnGpu.so` + OpPackage runtime 초기화 결과.
    runtime: Arc<QnnOppkgRuntime>,
    /// 28 layer graph cache.
    graph_cache: Mutex<GraphCache>,
    /// `--qnn-graph-cache-prebuild` flag (D1 결정, default true). false면
    /// `prebuild_graph_cache`가 noop이 되고 forward는 lazy build를 시도한다
    /// (M3.2 미지원 — fallback path).
    args_prebuild: bool,
    /// ENG-QNN-216 / INV-175 — fallback method 호출 카운터. fast path가 정상
    /// 동작하면 0이어야 한다. test에서 검증.
    fallback_call_count: AtomicU64,
}

impl QnnOppkgBackend {
    /// Backend 초기화 — `libQnnGpu.so` dlopen + `libqnn_oppkg.so`
    /// register + V2.0 fn-pointer 캐싱. 호스트(non-Android)에서는 명확한
    /// `Err`로 실패한다 (디바이스 빌드에서만 진행 가능).
    ///
    /// 호출자는 이후 `prebuild_graph_cache()`로 N×graphFinalize를 수행해야
    /// `execute_layer_graph` fast path가 동작한다 (D1 결정).
    pub fn new() -> Result<Self> {
        Self::with_prebuild(true)
    }

    /// Backend 초기화 — `--qnn-graph-cache-prebuild` flag와 함께. CLI에서
    /// 직접 호출 (default `true`).
    pub fn with_prebuild(args_prebuild: bool) -> Result<Self> {
        let runtime = QnnOppkgRuntime::init()?;
        Ok(Self {
            runtime,
            graph_cache: Mutex::new(GraphCache::new()),
            args_prebuild,
            fallback_call_count: AtomicU64::new(0),
        })
    }

    /// Runtime 핸들 공유 — `QnnOppkgMemory` (ENG-QNN-204) 와 layer_graph
    /// build/execute에서 V2.25 vtable + context handle 접근을 위해 사용.
    /// host 빌드에서는 backend init이 Err로 fail하므로 본 method는 호출되지
    /// 않지만 (Memory::alloc이 host fallback path로 빠짐), API 안정성 게이트로
    /// 노출. dead_code 허용 — Android만 활용.
    #[cfg_attr(not(target_os = "android"), allow(dead_code))]
    pub(crate) fn runtime_arc(&self) -> Arc<QnnOppkgRuntime> {
        self.runtime.clone()
    }

    /// ENG-QNN-216 / INV-175 — fallback 호출 카운터 (test 게이트).
    /// fast path가 정상 발동하면 본 카운터는 0을 유지해야 한다.
    pub fn fallback_call_count(&self) -> u64 {
        self.fallback_call_count.load(Ordering::Relaxed)
    }

    /// fallback 호출 카운터 reset (test helper).
    pub fn reset_fallback_call_count(&self) {
        self.fallback_call_count.store(0, Ordering::Relaxed);
    }

    /// Eager prebuild — model load 완료 시점에 1회 호출 (`--qnn-graph-cache-prebuild=true` 시).
    ///
    /// `slots`는 `TransformerWeights::layers` 순서. `cfg`는 layer dimension
    /// 메타데이터 (Qwen2.5-1.5B = `LayerConfig::qwen2p5_1p5b()`).
    ///
    /// `args_prebuild=false` 면 noop으로 빠지고 graph cache는 비어있는 상태로
    /// 유지된다 (M3.2에서 lazy build path 미지원 — caller가 forward 시점에
    /// Err를 받게 됨).
    ///
    /// host 빌드: `LayerGraph::build`가 즉시 Err로 fail하여 본 method가 Err
    /// 전파 — caller가 명확하게 catch + bail.
    pub fn prebuild_graph_cache(&self, slots: &[Arc<LayerSlot>], cfg: &LayerConfig) -> Result<()> {
        if !self.args_prebuild {
            // `--qnn-graph-cache-prebuild=false` 명시 — lazy build path 진입.
            // M3.2 단계에서 lazy build는 미지원이지만, caller (generate.rs)가
            // 명시적으로 끈 상태이므로 noop이 정상.
            return Ok(());
        }
        let mut cache = self.graph_cache.lock().expect("graph_cache mutex poisoned");
        cache.prebuild(&self.runtime, slots, cfg)?;
        eprintln!(
            "[qnn_oppkg] eager prebuild: {} layers, total finalize {} ms",
            cache.len(),
            cache.finalize_total_ms()
        );
        Ok(())
    }

    /// Cache lookup — M3.3 forward fast path에서 사용. M3.2 단계는 노출만.
    #[allow(dead_code)]
    pub(crate) fn graph_for_layer(
        &self,
        layer_idx: usize,
    ) -> Option<Arc<self::layer_graph::LayerGraph>> {
        self.graph_cache.lock().ok().and_then(|c| c.get(layer_idx))
    }
}

/// Forward 미구현 marker — fast path가 정상 동작하면 호출되지 않아야 한다
/// (INV-175). 만약 호출되면 production은 panic으로 명확히 fail (qnn_oppkg
/// backend는 trait 단위 fine-grained ops를 지원하지 않음).
///
/// `--qnn-allow-fallback`이 도입되면 (M3.4+) panic 대신 OpenCL secondary
/// 위임으로 분기. 본 단계는 fast path 단일 path만 지원.
fn forward_unimplemented(this: &QnnOppkgBackend, method: &'static str) -> ! {
    this.fallback_call_count.fetch_add(1, Ordering::Relaxed);
    unimplemented!(
        "QnnOppkgBackend::{} — fast path 미발동 (INV-175 위반). \
         transformer.rs forward가 supports_layer_graph()를 검사하지 않았거나 \
         graph cache prebuild가 누락된 상태. M3.4 fallback hook (--qnn-allow-fallback) 진입 시 OpenCL secondary 위임.",
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
    /// fast path 분기를 결정한다. idempotent (동일 인스턴스에 대해 항상 동일 값).
    ///
    /// 본 method는 graph cache가 prebuild 완료된 경우에만 true를 반환한다.
    /// host 빌드는 backend init이 Err로 fail하여 본 method가 호출되지 않는다.
    /// Android 빌드에서 cache가 비어있거나 부분 채워진 상태는 false를 반환하여
    /// transformer.rs가 (현재는 미지원이므로) bail하도록 한다.
    fn supports_layer_graph(&self) -> bool {
        // INV-174 idempotent: cache 길이는 prebuild 후 변하지 않으므로 항상 동일.
        let cache_ok = self
            .graph_cache
            .lock()
            .map(|c| !c.is_empty())
            .unwrap_or(false);
        cache_ok && self.runtime.is_initialized()
    }

    /// ENG-QNN-211/213/214 — 14-node single-layer graph dispatch.
    ///
    /// transformer.rs `forward_into` decode loop가 본 method를 layer당 1회
    /// 호출한다. host 빌드는 backend init Err로 본 path 진입 불가. Android
    /// 빌드에서 14-node graph 본문이 부재하면 LayerGraph::execute가 Err.
    fn execute_layer_graph(
        &self,
        layer_idx: usize,
        x: &Tensor,
        kv_cache_k: &mut Tensor,
        kv_cache_v: &mut Tensor,
        pos: usize,
        x_out: &mut Tensor,
    ) -> Result<()> {
        let lg = self.graph_for_layer(layer_idx).ok_or_else(|| {
            anyhow!("execute_layer_graph(layer={layer_idx}): graph cache miss — prebuild 누락")
        })?;

        // ENG-QNN-213/214 pre-conditions.
        // x: F32 [1, 1, dim] or [1, dim] depending on rank. KV cache: F16 buffer.
        let x_bytes = unsafe { std::slice::from_raw_parts(x.as_ptr(), x.size()) };
        let kv_k_bytes =
            unsafe { std::slice::from_raw_parts_mut(kv_cache_k.as_mut_ptr(), kv_cache_k.size()) };
        let kv_v_bytes =
            unsafe { std::slice::from_raw_parts_mut(kv_cache_v.as_mut_ptr(), kv_cache_v.size()) };
        let x_out_bytes =
            unsafe { std::slice::from_raw_parts_mut(x_out.as_mut_ptr(), x_out.size()) };

        // mask: ENG-QNN-228 — model load 시 1회 alloc (M3.4에서 backend 내부
        // scratch buffer로 도입). 본 단계는 빈 slice (LayerGraph::execute가
        // 내부 rpcmem buffer 사용).
        let empty_mask: &[u8] = &[];

        lg.execute(
            self.runtime.as_ref(),
            x_bytes,
            kv_k_bytes,
            kv_v_bytes,
            pos,
            empty_mask,
            x_out_bytes,
        )
    }

    // ── Below: trait method stubs. 모두 fast path 도입 후 호출되지 않아야 한다 (INV-175).
    //    호출되면 fallback_call_count 증가 + panic — fast path 분기 누락 명확히 검출.

    fn matmul(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> {
        forward_unimplemented(self, "matmul")
    }

    fn matmul_transposed(&self, _a: &Tensor, _b: &Tensor, _out: &mut Tensor) -> Result<()> {
        forward_unimplemented(self, "matmul_transposed")
    }

    fn matmul_slice(
        &self,
        _a: &Tensor,
        _b: &Tensor,
        _rows: usize,
        _cols: usize,
        _out: &mut Tensor,
    ) -> Result<()> {
        forward_unimplemented(self, "matmul_slice")
    }

    fn add_assign(&self, _a: &mut Tensor, _b: &Tensor) -> Result<()> {
        forward_unimplemented(self, "add_assign")
    }

    fn scale(&self, _x: &mut Tensor, _v: f32) -> Result<()> {
        forward_unimplemented(self, "scale")
    }

    fn silu_mul(&self, _a: &mut Tensor, _b: &Tensor) -> Result<()> {
        forward_unimplemented(self, "silu_mul")
    }

    fn rms_norm(&self, _x: &mut Tensor, _w: &Tensor, _eps: f32, _add_unit: bool) -> Result<()> {
        forward_unimplemented(self, "rms_norm")
    }

    fn rms_norm_oop(
        &self,
        _x: &Tensor,
        _out: &mut Tensor,
        _w: &Tensor,
        _eps: f32,
        _add_unit: bool,
    ) -> Result<()> {
        forward_unimplemented(self, "rms_norm_oop")
    }

    fn softmax(&self, _x: &mut Tensor) -> Result<()> {
        forward_unimplemented(self, "softmax")
    }

    fn rope_inplace(&self, _x: &mut Tensor, _start_pos: usize, _theta: f32) -> Result<()> {
        forward_unimplemented(self, "rope_inplace")
    }

    fn copy_from(&self, _t: &Tensor) -> Result<Tensor> {
        forward_unimplemented(self, "copy_from")
    }

    fn cast(&self, _src: &Tensor, _dst: &mut Tensor) -> Result<()> {
        forward_unimplemented(self, "cast")
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
