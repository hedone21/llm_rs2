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
pub mod hybrid_memory;
pub mod kv_buffer;
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
    /// Phase C perf 측정 — execute_layer_graph 호출 수 (28-layer 모두 fast path
    /// 활성 시 = 28 × decode_steps).
    exec_calls: AtomicU64,
    /// Phase C — fast path bridge read 누적 (3 read_buffer per call: x + kv_k + kv_v).
    bridge_read_ns: AtomicU64,
    /// Phase C — bridge_read 분해: x (small ~6KB), kv_k (large ~1MB), kv_v (large ~1MB).
    bridge_read_x_ns: AtomicU64,
    bridge_read_kv_k_ns: AtomicU64,
    bridge_read_kv_v_ns: AtomicU64,
    /// Phase C — fast path graph execute 누적 (lg.execute, QNN GPU 실행).
    graph_exec_ns: AtomicU64,
    /// Phase C — fast path bridge write 누적 (3 write_buffer per call).
    bridge_write_ns: AtomicU64,
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
            exec_calls: AtomicU64::new(0),
            bridge_read_ns: AtomicU64::new(0),
            bridge_read_x_ns: AtomicU64::new(0),
            bridge_read_kv_k_ns: AtomicU64::new(0),
            bridge_read_kv_v_ns: AtomicU64::new(0),
            graph_exec_ns: AtomicU64::new(0),
            bridge_write_ns: AtomicU64::new(0),
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

    /// OpenCL secondary backend을 closure로 노출. forward path가 OpenCL queue를
    /// 직접 필요로 할 때 사용 (e.g., flash-attn CPU fallback이 device-only buffer로
    /// 결과 write back). secondary가 OpenCL backend가 아니거나 미설정이면 None.
    #[cfg(feature = "opencl")]
    pub fn with_opencl_secondary<R>(
        &self,
        f: impl FnOnce(&crate::backend::opencl::OpenCLBackend) -> R,
    ) -> Option<R> {
        let slot = self.fallback_backend.lock().ok()?;
        let be = slot.as_ref()?;
        let ocl = be
            .as_any()
            .downcast_ref::<crate::backend::opencl::OpenCLBackend>()?;
        Some(f(ocl))
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

impl Drop for QnnOppkgBackend {
    fn drop(&mut self) {
        let calls = self.exec_calls.load(Ordering::Relaxed);
        if calls == 0 {
            return;
        }
        let r = self.bridge_read_ns.load(Ordering::Relaxed) as f64;
        let e = self.graph_exec_ns.load(Ordering::Relaxed) as f64;
        let w = self.bridge_write_ns.load(Ordering::Relaxed) as f64;
        let total = r + e + w;
        let calls_f = calls as f64;
        eprintln!(
            "[qnn_oppkg-timing] fast_path_calls={} fallback_calls={} | total={:.1}ms | \
             bridge_read={:.1}ms ({:.0}%, {:.0}μs/call) graph_exec={:.1}ms ({:.0}%, {:.0}μs/call) \
             bridge_write={:.1}ms ({:.0}%, {:.0}μs/call)",
            calls,
            self.fallback_call_count.load(Ordering::Relaxed),
            total / 1e6,
            r / 1e6,
            100.0 * r / total,
            r / calls_f / 1e3,
            e / 1e6,
            100.0 * e / total,
            e / calls_f / 1e3,
            w / 1e6,
            100.0 * w / total,
            w / calls_f / 1e3,
        );
        // bridge_read 분해: x (small) vs kv_k (large) vs kv_v (large).
        let rx = self.bridge_read_x_ns.load(Ordering::Relaxed) as f64;
        let rkk = self.bridge_read_kv_k_ns.load(Ordering::Relaxed) as f64;
        let rkv = self.bridge_read_kv_v_ns.load(Ordering::Relaxed) as f64;
        eprintln!(
            "[qnn_oppkg-timing-rd] bridge_read_x={:.1}ms ({:.0}μs/call) \
             bridge_read_kv_k={:.1}ms ({:.0}μs/call) bridge_read_kv_v={:.1}ms ({:.0}μs/call)",
            rx / 1e6,
            rx / calls_f / 1e3,
            rkk / 1e6,
            rkk / calls_f / 1e3,
            rkv / 1e6,
            rkv / calls_f / 1e3,
        );
        // lg.execute 내부 breakdown — graph_exec 1561μs/call이 어디서 오는지.
        let ci = self::layer_graph::LG_COPY_IN_NS.load(Ordering::Relaxed) as f64;
        let ep = self::layer_graph::LG_EXEC_PURE_NS.load(Ordering::Relaxed) as f64;
        let co = self::layer_graph::LG_COPY_OUT_NS.load(Ordering::Relaxed) as f64;
        let lg_total = ci + ep + co;
        if lg_total > 0.0 {
            eprintln!(
                "[qnn_oppkg-timing-lg] copy_in={:.1}ms ({:.0}%, {:.0}μs/call) \
                 graphExecute_pure={:.1}ms ({:.0}%, {:.0}μs/call) \
                 copy_out={:.1}ms ({:.0}%, {:.0}μs/call)",
                ci / 1e6,
                100.0 * ci / lg_total,
                ci / calls_f / 1e3,
                ep / 1e6,
                100.0 * ep / lg_total,
                ep / calls_f / 1e3,
                co / 1e6,
                100.0 * co / lg_total,
                co / calls_f / 1e3,
            );
        }
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
        // D-D.6 후속 디버깅 단계 — fast path infra (cl_mem↔host bridge + n_kv
        // input tensor + n_kv_buf binding)은 빌드/실행 PASS이지만 multi-token
        // decode token sequence가 OpenCL primary와 byte-equal하지 않음. 잔존
        // 가설:
        //   - graph 14-node pipeline 자체가 OpenCL forward path와 다른 결과
        //   - mask buffer layout (D-D.6에서 kv_capacity 크기로 확장)이 kernel
        //     index 가정과 mismatch
        //   - KV stride가 graph vs OpenCL에서 다른 가정
        //   - weight binding (28 layer cache hit) 검증 필요
        //
        // 디버깅 동안 fast path 비활성 (옵션 1 default — fallback OpenCL이 모든
        // forward 처리하여 token sequence는 OpenCL primary와 byte-equal). 활성화
        // 시 `LLMRS_QNN_OPPKG_FAST_PATH=1` env로 개별 enable 가능 (별도 phase).
        let force_fast = std::env::var("LLMRS_QNN_OPPKG_FAST_PATH").as_deref() == Ok("1");
        if !force_fast {
            return false;
        }
        // D-D.6 디버깅: fresh_build mode면 prebuild cache가 비어도 fast path 활성.
        // execute_layer_graph 내부에서 fresh_build helper로 매번 build.
        let fresh = std::env::var("LLMRS_QNN_OPPKG_FAST_PATH_FRESH_BUILD").as_deref() == Ok("1");
        if fresh {
            return self.runtime.is_initialized();
        }
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
    ///
    /// D-F 옵션 2 — cl_mem ↔ host bridge:
    /// `x`, `kv_cache_k/v`, `x_out`는 OpenCL secondary가 owning하는 cl_mem
    /// (UnifiedBuffer) tensor이고 host_ptr이 unmapped 상태일 수 있다. graph는
    /// rpcmem buffer만 알기 때문에 fallback backend의 read_buffer/write_buffer로
    /// 양방향 sync한다. KV cache는 매 layer 매 step read+write (graph 자체
    /// rpcmem KV slot에 누적되지만 caller도 OpenCL cl_mem에 동일 데이터를
    /// 보유하도록 mirror) — 옵션 3 (KV rpcmem alloc)에서 zero-copy 가능.
    fn execute_layer_graph(
        &self,
        layer_idx: usize,
        x: &Tensor,
        kv_cache_k: &mut Tensor,
        kv_cache_v: &mut Tensor,
        pos: usize,
        n_kv: usize,
        x_out: &mut Tensor,
    ) -> Result<()> {
        // D-D.6 디버깅: fresh build mode — `LLMRS_QNN_OPPKG_FAST_PATH_FRESH_BUILD=1`
        // 시 cache hit 무시하고 매 execute마다 build_layer_graph 호출. microbench와
        // 동일한 lifecycle (build → execute → drop)을 production에 inject.
        let lg = if std::env::var("LLMRS_QNN_OPPKG_FAST_PATH_FRESH_BUILD").as_deref() == Ok("1") {
            let cache = self.graph_cache.lock().expect("graph_cache mutex poisoned");
            cache.fresh_build(self.runtime.as_ref(), layer_idx)?
        } else {
            self.graph_for_layer(layer_idx).ok_or_else(|| {
                anyhow!("execute_layer_graph(layer={layer_idx}): graph cache miss — prebuild 누락")
            })?
        };

        // fallback backend (OpenCL secondary)를 통해 cl_mem buffer를 host bytes로
        // 가져온다. 이는 INV-175 (decode fast path 동안 fallback_call_count 0)
        // 게이트와 무관 — bridge용 read/write_buffer는 본 method 자체에서 직접
        // 호출하며 fallback 카운터를 증가시키지 않는다.
        let slot = self
            .fallback_backend
            .lock()
            .expect("fallback_backend mutex poisoned");
        let fb = slot.as_ref().ok_or_else(|| {
            anyhow!("execute_layer_graph: fallback backend (OpenCL secondary) 미설정")
        })?;

        let t_read = std::time::Instant::now();
        let t_x = std::time::Instant::now();
        let mut x_bytes = vec![0u8; x.size()];
        fb.read_buffer(x, &mut x_bytes)?;
        let x_ns = t_x.elapsed().as_nanos() as u64;
        let t_kk = std::time::Instant::now();
        let mut kv_k_bytes = vec![0u8; kv_cache_k.size()];
        fb.read_buffer(kv_cache_k, &mut kv_k_bytes)?;
        let kk_ns = t_kk.elapsed().as_nanos() as u64;
        let t_kv = std::time::Instant::now();
        let mut kv_v_bytes = vec![0u8; kv_cache_v.size()];
        fb.read_buffer(kv_cache_v, &mut kv_v_bytes)?;
        let kv_ns = t_kv.elapsed().as_nanos() as u64;
        let mut x_out_bytes = vec![0u8; x_out.size()];
        let read_ns = t_read.elapsed().as_nanos() as u64;
        self.bridge_read_x_ns.fetch_add(x_ns, Ordering::Relaxed);
        self.bridge_read_kv_k_ns.fetch_add(kk_ns, Ordering::Relaxed);
        self.bridge_read_kv_v_ns.fetch_add(kv_ns, Ordering::Relaxed);

        // D-D.6 디버깅: OpenCL secondary와 QNN GPU 사이 GPU state 간섭 가능성
        // 검증 — read_buffer (blocking) 후에도 OpenCL queue가 GPU에서 fully
        // idle이 아닐 수 있음. 명시적 synchronize() 추가.
        if std::env::var("LLMRS_QNN_OPPKG_FAST_PATH_FORCE_SYNC").as_deref() == Ok("1") {
            fb.synchronize()?;
        }

        // mask: graph 내부 rpcmem buffer 사용 (M3.4 시점은 빈 slice).
        let empty_mask: &[u8] = &[];

        let t_exec = std::time::Instant::now();
        lg.execute(
            self.runtime.as_ref(),
            &x_bytes,
            &mut kv_k_bytes,
            &mut kv_v_bytes,
            pos,
            n_kv,
            empty_mask,
            &mut x_out_bytes,
        )?;
        let exec_ns = t_exec.elapsed().as_nanos() as u64;

        // graph rpcmem → OpenCL cl_mem write back.
        let t_write = std::time::Instant::now();
        fb.write_buffer(x_out, &x_out_bytes)?;
        fb.write_buffer(kv_cache_k, &kv_k_bytes)?;
        fb.write_buffer(kv_cache_v, &kv_v_bytes)?;
        let write_ns = t_write.elapsed().as_nanos() as u64;

        self.exec_calls.fetch_add(1, Ordering::Relaxed);
        self.bridge_read_ns.fetch_add(read_ns, Ordering::Relaxed);
        self.graph_exec_ns.fetch_add(exec_ns, Ordering::Relaxed);
        self.bridge_write_ns.fetch_add(write_ns, Ordering::Relaxed);
        Ok(())
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

    fn gather(&self, src: &Tensor, indices: &Tensor, dst: &mut Tensor) -> Result<()> {
        fallback_or_panic(self, "gather", |be| be.gather(src, indices, dst))
    }

    #[allow(clippy::too_many_arguments)]
    fn attention_gen(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        num_heads_q: usize,
        num_heads_kv: usize,
        head_dim: usize,
        cache_seq_len: usize,
        scores_out: Option<&mut [f32]>,
    ) -> Result<()> {
        fallback_or_panic(self, "attention_gen", move |be| {
            be.attention_gen(
                q,
                k_cache,
                v_cache,
                out,
                num_heads_q,
                num_heads_kv,
                head_dim,
                cache_seq_len,
                scores_out,
            )
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn flash_attention_prefill(
        &self,
        q: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        out: &mut Tensor,
        n_heads_q: usize,
        n_heads_kv: usize,
        seq_len: usize,
        cache_seq_len: usize,
        head_dim: usize,
        kv_capacity: usize,
        batch_size: usize,
        is_head_major: bool,
    ) -> Result<bool> {
        fallback_or_panic(self, "flash_attention_prefill", |be| {
            be.flash_attention_prefill(
                q,
                k_cache,
                v_cache,
                out,
                n_heads_q,
                n_heads_kv,
                seq_len,
                cache_seq_len,
                head_dim,
                kv_capacity,
                batch_size,
                is_head_major,
            )
        })
    }

    /// LISWAP-6 — delegate alias buffer creation to the OpenCL secondary
    /// backend. `qnn_oppkg` shares the OpenCL context with its secondary
    /// (`with_opencl_secondary`), so the alias `cl_mem` lives in the same
    /// context the production swap path uses.
    #[cfg(feature = "opencl")]
    fn alloc_alias_weight_buffer(
        &self,
        host_ptr: *mut u8,
        offset: usize,
        size: usize,
        dtype: crate::core::buffer::DType,
        secondary_arc: std::sync::Arc<crate::models::weights::SecondaryMmap>,
        layer_region: std::sync::Arc<crate::models::weights::rpcmem_secondary::RpcmemLayerRegion>,
    ) -> Result<Option<std::sync::Arc<dyn crate::core::buffer::Buffer>>> {
        let res = self.with_opencl_secondary(|ocl| {
            ocl.alloc_alias_weight_buffer(
                host_ptr,
                offset,
                size,
                dtype,
                secondary_arc.clone(),
                layer_region.clone(),
            )
        });
        match res {
            Some(r) => r,
            None => Ok(None), // OpenCL secondary unset → caller falls back
        }
    }

    fn kv_scatter_f32_to_f16(
        &self,
        k_src: &Tensor,
        v_src: &Tensor,
        k_dst: &mut Tensor,
        v_dst: &mut Tensor,
        head_dim: usize,
        capacity: usize,
        write_pos: usize,
    ) -> Result<()> {
        fallback_or_panic(self, "kv_scatter_f32_to_f16", |be| {
            be.kv_scatter_f32_to_f16(k_src, v_src, k_dst, v_dst, head_dim, capacity, write_pos)
        })
    }

    fn buffer_shift(
        &self,
        tensor: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        fallback_or_panic(self, "buffer_shift", |be| {
            be.buffer_shift(tensor, src_offset, dst_offset, count)
        })
    }

    fn copy_slice(
        &self,
        src: &Tensor,
        dst: &mut Tensor,
        src_offset: usize,
        dst_offset: usize,
        count: usize,
    ) -> Result<()> {
        fallback_or_panic(self, "copy_slice", |be| {
            be.copy_slice(src, dst, src_offset, dst_offset, count)
        })
    }

    fn copy_into(&self, src: &Tensor, dst: &mut Tensor) -> Result<()> {
        fallback_or_panic(self, "copy_into", |be| be.copy_into(src, dst))
    }

    fn read_buffer(&self, t: &Tensor, dst: &mut [u8]) -> Result<()> {
        fallback_or_panic(self, "read_buffer", |be| be.read_buffer(t, dst))
    }

    fn write_buffer(&self, t: &mut Tensor, src: &[u8]) -> Result<()> {
        fallback_or_panic(self, "write_buffer", |be| be.write_buffer(t, src))
    }

    fn write_buffer_range(&self, t: &mut Tensor, src: &[u8], dst_offset: usize) -> Result<()> {
        fallback_or_panic(self, "write_buffer_range", |be| {
            be.write_buffer_range(t, src, dst_offset)
        })
    }

    fn add_row_bias(&self, x: &mut Tensor, bias: &Tensor) -> Result<()> {
        fallback_or_panic(self, "add_row_bias", |be| be.add_row_bias(x, bias))
    }

    fn gelu_tanh_mul(&self, gate: &mut Tensor, up: &Tensor) -> Result<()> {
        fallback_or_panic(self, "gelu_tanh_mul", |be| be.gelu_tanh_mul(gate, up))
    }

    fn add_rms_norm_oop(
        &self,
        x: &mut Tensor,
        residual: &Tensor,
        out: &mut Tensor,
        w: &Tensor,
        eps: f32,
        add_unit: bool,
    ) -> Result<()> {
        fallback_or_panic(self, "add_rms_norm_oop", |be| {
            be.add_rms_norm_oop(x, residual, out, w, eps, add_unit)
        })
    }

    fn matmul_ffn_gate_up_silu(
        &self,
        x: &Tensor,
        w_gate: &Tensor,
        w_up: &Tensor,
        out: &mut Tensor,
        up_scratch: &mut Tensor,
    ) -> Result<()> {
        fallback_or_panic(self, "matmul_ffn_gate_up_silu", |be| {
            be.matmul_ffn_gate_up_silu(x, w_gate, w_up, out, up_scratch)
        })
    }

    fn ensure_noshuffle_soa_registered(&self, tensor: &Tensor) -> Result<()> {
        fallback_or_panic(self, "ensure_noshuffle_soa_registered", |be| {
            be.ensure_noshuffle_soa_registered(tensor)
        })
    }

    fn invalidate_noshuffle_soa_registry(&self) {
        fallback_or_panic(self, "invalidate_noshuffle_soa_registry", |be| {
            be.invalidate_noshuffle_soa_registry()
        })
    }

    fn max_single_alloc(&self) -> usize {
        let slot = self
            .fallback_backend
            .lock()
            .expect("fallback_backend mutex poisoned");
        if let Some(be) = slot.as_ref() {
            be.max_single_alloc()
        } else {
            // host test path: use trait default.
            usize::MAX
        }
    }

    /// Model load weight upload — D-F: fallback (OpenCL secondary)에 위임하여
    /// source buffer를 OpenCL cl_mem으로 promote한다. fallback이 없으면 기존
    /// passthrough (host test / non-Android build path).
    ///
    /// 배경 (M3.4 RED root cause):
    /// `gguf::load_raw`가 mmap CPU tensor를 만들고 `backend.copy_weight_from`을
    /// 호출한다. 이전 구현은 source buffer를 그대로 passthrough → weight
    /// tensor가 MmapBuffer로 남아 noshuffle prep에서 `get_cl_mem` fail
    /// (`Weight buffer has no cl_mem`) → prefill matmul fallback의 cl_mem
    /// dereference에서 segv. fallback에 위임하여 OpenCL이 cl_mem으로
    /// 직접 alloc하면 noshuffle SOA prep + prefill 모두 OpenCL primary처럼
    /// 동작한다.
    ///
    /// `fallback_call_count`는 증가시키지 않는다 — model load는 INV-175
    /// (decode fast path) 게이트와 무관.
    fn copy_from(&self, t: &Tensor) -> Result<Tensor> {
        let slot = self
            .fallback_backend
            .lock()
            .expect("fallback_backend mutex poisoned");
        if let Some(be) = slot.as_ref() {
            be.copy_from(t)
        } else {
            Ok(Tensor::new(
                t.shape().clone(),
                t.buffer().clone(),
                t.backend().clone(),
            ))
        }
    }

    /// Weight upload — `copy_from`과 동일하게 fallback에 위임한다. Backend
    /// trait의 default impl은 `self.copy_from`을 호출하지만, 명시 override로
    /// fallback의 `copy_weight_from` (weight-specific allocation policy)을
    /// 직접 사용한다. OpenCL secondary는 `copy_weight_from`이 default impl
    /// (= `copy_from`)이므로 동작은 동일하나, CUDA backend가 secondary가
    /// 되는 경우엔 weight policy가 적용된다 (현 시점 미사용 path).
    fn copy_weight_from(&self, t: &Tensor) -> Result<Tensor> {
        let slot = self
            .fallback_backend
            .lock()
            .expect("fallback_backend mutex poisoned");
        if let Some(be) = slot.as_ref() {
            be.copy_weight_from(t)
        } else {
            Ok(Tensor::new(
                t.shape().clone(),
                t.buffer().clone(),
                t.backend().clone(),
            ))
        }
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
