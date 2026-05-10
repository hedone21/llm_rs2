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
pub mod weight_pack;

use crate::core::backend::Backend;
use crate::core::tensor::Tensor;
use crate::models::weights::LayerSlot;
use anyhow::{Result, anyhow};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Mutex};

/// 본 backend가 trait fallback을 호출하지 않고 host buffer 재사용 path를
/// 우선시한다는 의미. weight tensor는 model load 시점에 host buffer (GGUF mmap)을
/// 그대로 보존하고, graph build 시점에 SOA 변환 + rpcmem alloc 1회만 수행한다.
const _DOC_HOST_BUFFER_PASSTHROUGH: () = ();

use self::graph_cache::GraphCache;
use self::layer_graph::LayerConfig;
use self::runtime::QnnOppkgRuntime;

/// QNN OpPackage backend — `Backend` trait 구현 진입점.
///
/// ENG-QNN-201/INV-166: 모든 필수 trait method를 OpenCL과 동일 시그니처로
/// 구현한다. M3.4 단계: prefill 및 model load 단계에서 trait fallback method
/// (matmul 등) 호출은 OpenCL secondary backend로 위임된다 (`--qnn-allow-fallback`
/// 활성). decode (seq_len=1) fast path만 graph 통해 직접 처리.
///
/// thread-safety: runtime + graph cache는 `Arc`로 공유하며, graph cache는 build
/// 시점에만 `Mutex`로 보호.
pub struct QnnOppkgBackend {
    /// `libQnnGpu.so` + OpPackage runtime 초기화 결과.
    runtime: Arc<QnnOppkgRuntime>,
    /// 28 layer graph cache.
    graph_cache: Mutex<GraphCache>,
    /// `--qnn-graph-cache-prebuild` flag (D1 결정, default true). false면
    /// `prebuild_graph_cache`가 noop이 되고 forward는 lazy build를 시도한다
    /// (M3.2 미지원 — fallback path).
    args_prebuild: bool,
    /// ENG-QNN-216 / INV-175 — fallback method 호출 카운터. decode fast path가
    /// 정상 동작하면 0이어야 한다 (prefill/model load는 fallback 정상 사용).
    fallback_call_count: AtomicU64,
    /// OpenCL secondary backend (fallback 위임 대상). 없으면 fallback method
    /// 호출 시 명확한 panic.
    fallback_backend: Mutex<Option<Arc<dyn Backend>>>,
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
            fallback_backend: Mutex::new(None),
        })
    }

    /// OpenCL secondary backend 설정 (post-init). prefill 및 model load 시점에
    /// trait fallback method 호출을 본 secondary로 위임한다. dispatcher
    /// (generate.rs)가 backend init 직후 호출.
    pub fn set_fallback_backend(&self, fallback: Arc<dyn Backend>) {
        if let Ok(mut slot) = self.fallback_backend.lock() {
            *slot = Some(fallback);
        }
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

/// Forward fallback helper — prefill 및 model load 단계에서 호출되며 OpenCL
/// secondary backend로 위임한다. fallback이 미설정이면 panic으로 명확히 fail.
///
/// decode (seq_len=1) fast path는 transformer.rs에서 `execute_layer_graph`를
/// 직접 호출하므로 본 path는 거치지 않는다. fallback_call_count 증가는 fallback
/// 호출 횟수 추적용 (decode 동안 0 유지가 INV-175 게이트, prefill은 무관).
fn fallback_or_panic<F, R>(this: &QnnOppkgBackend, method: &'static str, f: F) -> R
where
    F: FnOnce(&Arc<dyn Backend>) -> R,
{
    this.fallback_call_count.fetch_add(1, Ordering::Relaxed);
    let slot = this
        .fallback_backend
        .lock()
        .expect("fallback_backend mutex poisoned");
    if let Some(be) = slot.as_ref() {
        f(be)
    } else {
        panic!(
            "QnnOppkgBackend::{} — fast path 미발동 + fallback backend 미설정. \
             generate.rs dispatcher가 set_fallback_backend()를 호출하지 않은 상태.",
            method
        )
    }
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

    // ── Below: trait method fallback wrappers — prefill 및 model load에서
    //    OpenCL secondary backend로 위임. decode fast path는 execute_layer_graph
    //    direct call이므로 본 path 비통과 (INV-175 게이트는 decode 한정).

    fn matmul(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        fallback_or_panic(self, "matmul", |be| be.matmul(a, b, out))
    }

    fn matmul_transposed(&self, a: &Tensor, b: &Tensor, out: &mut Tensor) -> Result<()> {
        fallback_or_panic(self, "matmul_transposed", |be| {
            be.matmul_transposed(a, b, out)
        })
    }

    fn matmul_slice(
        &self,
        a: &Tensor,
        b: &Tensor,
        rows: usize,
        cols: usize,
        out: &mut Tensor,
    ) -> Result<()> {
        fallback_or_panic(self, "matmul_slice", |be| {
            be.matmul_slice(a, b, rows, cols, out)
        })
    }

    fn add_assign(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        fallback_or_panic(self, "add_assign", |be| be.add_assign(a, b))
    }

    fn scale(&self, x: &mut Tensor, v: f32) -> Result<()> {
        fallback_or_panic(self, "scale", |be| be.scale(x, v))
    }

    fn silu_mul(&self, a: &mut Tensor, b: &Tensor) -> Result<()> {
        fallback_or_panic(self, "silu_mul", |be| be.silu_mul(a, b))
    }

    fn rms_norm(&self, x: &mut Tensor, w: &Tensor, eps: f32, add_unit: bool) -> Result<()> {
        fallback_or_panic(self, "rms_norm", |be| be.rms_norm(x, w, eps, add_unit))
    }

    fn rms_norm_oop(
        &self,
        x: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        fallback_or_panic(self, "rms_norm_oop", |be| {
            be.rms_norm_oop(x, out, w, eps, add_unit)
        })
    }

    fn softmax(&self, x: &mut Tensor) -> Result<()> {
        fallback_or_panic(self, "softmax", |be| be.softmax(x))
    }

    fn rope_inplace(&self, x: &mut Tensor, start_pos: usize, theta: f32) -> Result<()> {
        fallback_or_panic(self, "rope_inplace", |be| {
            be.rope_inplace(x, start_pos, theta)
        })
    }

    /// Model load passthrough — qnn_oppkg backend는 weight tensor를 host buffer
    /// (GGUF mmap) 형태로 보존하고, graph build 시점에 SOA 변환 + rpcmem alloc
    /// 1회 수행한다. 따라서 `copy_from`은 source의 buffer를 그대로 clone하여
    /// backend reference만 본 backend로 변경한다.
    ///
    /// 이는 trait fallback이 아니라 정상 path — weight은 forward 동안 본 backend가
    /// 직접 접근하지 않으며, `execute_layer_graph`는 LayerSlot에서 별도 추출.
    /// `fallback_call_count`는 증가시키지 않는다 (INV-175).
    fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
        Ok(Tensor::new(
            t.shape().clone(),
            t.buffer().clone(),
            // backend 자체는 trait object Arc<dyn Backend>가 필요하지만 qnn_oppkg
            // 외부에서 backend ref를 새로 만들 수 없으므로 source tensor의
            // backend를 그대로 사용. weight은 forward 시 본 backend가 직접
            // 접근하지 않으므로 backend 일관성은 무관 (graph build에서만 host
            // pointer 사용).
            t.backend().clone(),
        ))
    }

    fn cast(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        fallback_or_panic(self, "cast", |be| be.cast(src, dst))
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
