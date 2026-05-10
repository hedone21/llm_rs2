//! Single-layer 14-node QNN graph wrapper (M3.4).
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
//! ## 본 단계 (M3.4) 적용 범위
//!
//! Android target: 14-node body 본격 이식 (microbench `microbench_qnn_qwen_layer.rs`
//! lines 1680-2174 시퀀스). graphCreate + tensor 등록 + 14×graphAddNode +
//! graphFinalize.
//!
//! M2 핵심 결정 (4개) 보존:
//! - FlashAttn sinks `flat_dims: vec![n_head]` (line ~1417)
//! - RoPE OOP (`kernel_rope_simple_oop`, simple_ops.cl)
//! - SiluMul `OutputTensorAliased` (M2 INV-164)
//! - KvScatter multi-output (k_rot → KV_K, v → KV_V)
//! - Q4_0 weight `RawBytes(18, N/32)` (M2 INV-165, AOS layout from production GGUF
//!   변환 후 SOA — `weight_pack::aos_to_soa_q4_0` 사용)
//!
//! ## execute path (M3.4 D-D.3, 2026-05-10)
//!
//! `pos`는 RoPE `start_pos` (Q/K 2 nodes) + KvScatter `write_pos` (1 node)에
//! 모두 영향. M3.4 RED 시점 root cause는 SCALAR op param이 graphFinalize에
//! 시점에 baked → multi-token decode 불가. D-D.1+D-D.2에서 ops descriptor +
//! `.cl` kernel을 input tensor 방식으로 갱신: pos를 `INT_32 [1]` rank-1
//! tensor로 graph에 추가 + 매 graphExecute 직전에 host pointer로 값 write.
//!
//! 본 D-D.3에서는 layer당 1개 pos_buf rpcmem slot을 alloc하고, RoPE Q/K +
//! KvScatter 3 nodes의 input tensor 배열에 동일 pos_buf을 share한다 (graph
//! 내부에서 read-only, write race 없음). `execute()`는 graphExecute 직전에
//! `*(pos_host_ptr as *mut i32) = pos as i32`를 한 줄로 수행한다.

use crate::backend::qnn_oppkg::runtime::QnnOppkgRuntime;
use crate::layers::transformer_layer::TransformerLayer;
use anyhow::{Result, anyhow};

// host metadata only — qnn_oppkg crate에서 LayerConfig + LAYER_NODE_COUNT를
// 직접 사용. 본 type은 ffi 의존성 없이 host build에서도 컴파일된다.
pub use qnn_oppkg::graph::layer::{LAYER_INTERMEDIATE_COUNT, LAYER_NODE_COUNT, LayerConfig};

/// `graphFinalize` 1회 호출 budget (ms). ENG-QNN-209/INV-167.
///
/// M3.4 디바이스 측정 (S25 Adreno 830, Qwen2.5-1.5B Q4_0): layer 0 = 1181 ms.
/// D-D.6 Phase B (BiasAdd 3개 추가, 17 nodes): layer 0 = 1554 ms.
/// 17 nodes로 확장되며 finalize 비용도 비례 증가 (~25%). budget 2000 ms로 갱신.
pub const FINALIZE_BUDGET_MS: u32 = 2000;

/// 14-node single-layer QNN graph (M2.H 이식).
///
/// 본 struct는 model load 시점에 `build()` 1회로 채워지고 process lifetime
/// 동안 재사용된다 (INV-167). dispatch path는 `execute()`로 1 layer를 1회
/// graphExecute에 dispatch한다.
pub struct LayerGraph {
    /// Layer index (0..n_layers).
    pub layer_idx: usize,
    /// `graphFinalize` 측정값 (ms). build path가 기록한다 (host 빌드는 0).
    pub finalize_ms: u32,
    /// 본 graph가 보유한 weight handle 개수. M2.H = 7 Q4_0 (× 2 buffers SOA) + 2 norm = 16.
    pub weight_handle_count: usize,
    /// Android-only graph handle storage. host build에서는 본 필드가 부재.
    #[cfg(target_os = "android")]
    inner: android::AndroidLayerGraph,
}

impl LayerGraph {
    /// 1-layer graph build — model load 시점 1회 호출.
    ///
    /// - Android: graphCreate + rpcmem alloc + memRegister + 14×graphAddNode + graphFinalize.
    /// - host: build 자체가 의미 없으므로 명확한 Err.
    pub fn build(
        runtime: &QnnOppkgRuntime,
        layer_idx: usize,
        weights: &TransformerLayer,
        cfg: &LayerConfig,
    ) -> Result<Self> {
        #[cfg(target_os = "android")]
        {
            let (inner, finalize_ms, weight_handle_count) =
                android::build_layer_graph(runtime, layer_idx, weights, cfg)?;
            Ok(Self {
                layer_idx,
                finalize_ms,
                weight_handle_count,
                inner,
            })
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = (runtime, weights, cfg);
            Err(anyhow!(
                "LayerGraph::build (host, layer={layer_idx}) — host build에서는 graph 빌드 불가. 디바이스 빌드 + Android runtime에서 동작."
            ))
        }
    }

    /// 1-layer graph dispatch — token decode 시점 layer당 1회 호출.
    ///
    /// `mask_bytes`가 비어있으면 backend 내부 scratch buffer를 사용한다 (M3.4
    /// default — caller from `mod.rs::execute_layer_graph`).
    #[allow(clippy::too_many_arguments)]
    pub fn execute(
        &self,
        runtime: &QnnOppkgRuntime,
        x_in_bytes: &[u8],
        kv_k_bytes: &mut [u8],
        kv_v_bytes: &mut [u8],
        pos: usize,
        n_kv: usize,
        mask_bytes: &[u8],
        x_out_bytes: &mut [u8],
    ) -> Result<()> {
        #[cfg(target_os = "android")]
        {
            self.inner.execute(
                runtime,
                x_in_bytes,
                kv_k_bytes,
                kv_v_bytes,
                pos,
                n_kv,
                mask_bytes,
                x_out_bytes,
            )
        }
        #[cfg(not(target_os = "android"))]
        {
            let _ = (
                runtime,
                x_in_bytes,
                kv_k_bytes,
                kv_v_bytes,
                pos,
                n_kv,
                mask_bytes,
                x_out_bytes,
            );
            Err(anyhow!(
                "LayerGraph::execute (host, layer={}, pos={pos}, n_kv={n_kv}) — host build에서는 graphExecute 불가.",
                self.layer_idx
            ))
        }
    }
}

/// Phase C — lg.execute 내부 breakdown counters (process-global).
/// graphExecute 자체 vs internal memcpy (rpcmem ↔ host bytes) 분리.
pub(crate) static LG_COPY_IN_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LG_EXEC_PURE_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);
pub(crate) static LG_COPY_OUT_NS: std::sync::atomic::AtomicU64 =
    std::sync::atomic::AtomicU64::new(0);

#[cfg(target_os = "android")]
mod android {
    //! Android-only graph build/execute body (M3.4 — 14-node body 이식).

    use crate::backend::qnn_oppkg::runtime::{QnnOppkgRuntime, ffi};
    use crate::backend::qnn_oppkg::weight_pack::{Q4_0_BLOCK_BYTES, aos_to_soa_q4_0};
    use crate::core::tensor::Tensor;
    use crate::layers::transformer_layer::TransformerLayer;
    use anyhow::{Result, anyhow, ensure};
    use std::ffi::CString;
    use std::os::raw::c_char;
    use std::ptr;
    use std::time::Instant;

    use super::LayerConfig;

    /// rpcmem 할당 1개 슬롯 — host_ptr + fd + size + memHandle.
    pub(super) struct RpcmemSlot {
        pub host_ptr: *mut u8,
        pub fd: i32,
        pub size: usize,
        pub mem_handle: ffi::Qnn_MemHandle_t,
    }

    /// 14-node graph + 모든 baked tensor + memHandle 묶음.
    pub(super) struct AndroidLayerGraph {
        graph: ffi::Qnn_GraphHandle_t,
        /// 17 weight slots: x(in) + rms_pre + rms_post + 7×(qq,qd) Q4_0 SOA = 17.
        /// (KV K/V/mask는 별도 slot)
        slots: Vec<RpcmemSlot>,
        /// graphExecute 시 input tensors (Qnn_Tensor_t with memHandle baked).
        exec_inputs: Vec<ffi::Qnn_Tensor_t>,
        /// graphExecute 시 output tensors (KV K, KV V, x_out).
        exec_outputs: Vec<ffi::Qnn_Tensor_t>,
        /// Slot index 매핑 (host pointer copy 시 사용).
        idx_x_in: usize,
        idx_x_out: usize,
        /// D-F 옵션 2: graph rpcmem KV slot index. execute 진입 시 외부 KV
        /// cl_mem (OpenCL secondary가 owning) → graph rpcmem으로 sync,
        /// execute 종료 시 graph rpcmem → 외부 KV cl_mem으로 write back.
        /// `bytes_kv`는 K/V 동일 크기.
        idx_kv_k: usize,
        idx_kv_v: usize,
        /// M3.4 D-D.3: pos_buf rpcmem slot의 host pointer. graphExecute 직전
        /// `*(pos_host_ptr as *mut i32) = pos as i32`로 갱신. RoPE Q/K +
        /// KvScatter 3 nodes가 동일 pos_buf input tensor를 share.
        pos_host_ptr: *mut i32,
        /// M3.4 D-D.6: n_kv_buf rpcmem slot의 host pointer. graphExecute 직전
        /// `*(n_kv_host_ptr) = n_kv as i32`로 갱신. FlashAttn node가 input
        /// tensor로 read하여 multi-token decode에서 누적 KV attention 가능.
        n_kv_host_ptr: *mut i32,
        /// LayerConfig snapshot.
        #[allow(dead_code)]
        cfg: LayerConfig,
    }

    // SAFETY: graph handle은 build 후 immutable. graphExecute는 caller (GraphCache
    // Mutex)에서 직렬화.
    unsafe impl Send for AndroidLayerGraph {}
    unsafe impl Sync for AndroidLayerGraph {}

    impl AndroidLayerGraph {
        pub(super) fn execute(
            &self,
            runtime: &QnnOppkgRuntime,
            x_in_bytes: &[u8],
            kv_k_bytes: &mut [u8],
            kv_v_bytes: &mut [u8],
            pos: usize,
            n_kv: usize,
            _mask_bytes: &[u8],
            x_out_bytes: &mut [u8],
        ) -> Result<()> {
            let t_copy_in = std::time::Instant::now();
            // Copy input x into rpcmem-backed input slot.
            let in_slot = &self.slots[self.idx_x_in];
            ensure!(
                x_in_bytes.len() <= in_slot.size,
                "x_in_bytes={} > slot.size={}",
                x_in_bytes.len(),
                in_slot.size
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    x_in_bytes.as_ptr(),
                    in_slot.host_ptr,
                    x_in_bytes.len(),
                );
            }

            // D-F 옵션 2: KV cache bridge — caller (execute_layer_graph)는
            // OpenCL cl_mem KV를 host bytes로 read해서 넘긴다. graph는
            // rpcmem KV slot을 사용하므로 외부 KV bytes를 graph rpcmem에
            // sync한 뒤 execute. KvScatter가 새 token을 graph rpcmem에 write,
            // 종료 시점에 graph rpcmem → caller bytes로 write back하면 caller가
            // OpenCL cl_mem에 propagate한다.
            let kv_k_slot = &self.slots[self.idx_kv_k];
            let kv_v_slot = &self.slots[self.idx_kv_v];
            ensure!(
                kv_k_bytes.len() <= kv_k_slot.size,
                "kv_k_bytes={} > slot.size={}",
                kv_k_bytes.len(),
                kv_k_slot.size
            );
            ensure!(
                kv_v_bytes.len() <= kv_v_slot.size,
                "kv_v_bytes={} > slot.size={}",
                kv_v_bytes.len(),
                kv_v_slot.size
            );
            unsafe {
                std::ptr::copy_nonoverlapping(
                    kv_k_bytes.as_ptr(),
                    kv_k_slot.host_ptr,
                    kv_k_bytes.len(),
                );
                std::ptr::copy_nonoverlapping(
                    kv_v_bytes.as_ptr(),
                    kv_v_slot.host_ptr,
                    kv_v_bytes.len(),
                );
            }

            // M3.4 D-D.3: pos_buf write — RoPE Q/K + KvScatter 3 nodes가
            // 동일 pos_buf을 input tensor로 read. SAFETY: pos_host_ptr는
            // build 시점에 alloc된 rpcmem slot의 4-byte aligned host pointer.
            // 동일 layer graph는 GraphCache Mutex로 직렬화되어 race 없음.
            unsafe {
                std::ptr::write(self.pos_host_ptr, pos as i32);
            }
            // M3.4 D-D.6: n_kv_buf write — FlashAttn node가 input tensor로
            // read하여 attention loop를 [0..n_kv)로 동적 결정.
            unsafe {
                std::ptr::write(self.n_kv_host_ptr, n_kv as i32);
            }

            let copy_in_ns = t_copy_in.elapsed().as_nanos() as u64;
            super::LG_COPY_IN_NS.fetch_add(copy_in_ns, std::sync::atomic::Ordering::Relaxed);

            // graphExecute.
            let v = runtime.v();
            let graph_execute = v
                .graphExecute
                .ok_or_else(|| anyhow!("graphExecute fn-pointer is NULL"))?;
            // SAFETY: exec_inputs/outputs는 build 시점에 valid memHandle로 baked.
            // graph handle도 valid.
            let t_exec_pure = std::time::Instant::now();
            let err = unsafe {
                graph_execute(
                    self.graph,
                    self.exec_inputs.as_ptr(),
                    self.exec_inputs.len() as u32,
                    self.exec_outputs.as_ptr() as *mut _,
                    self.exec_outputs.len() as u32,
                    ptr::null_mut(),
                    ptr::null_mut(),
                )
            };
            let exec_pure_ns = t_exec_pure.elapsed().as_nanos() as u64;
            super::LG_EXEC_PURE_NS.fetch_add(exec_pure_ns, std::sync::atomic::Ordering::Relaxed);
            if err != 0 {
                return Err(anyhow!("graphExecute err=0x{:x}", err));
            }
            let t_copy_out = std::time::Instant::now();

            // Copy x_out from rpcmem-backed output slot.
            let out_slot = &self.slots[self.idx_x_out];
            ensure!(
                x_out_bytes.len() <= out_slot.size,
                "x_out_bytes={} > slot.size={}",
                x_out_bytes.len(),
                out_slot.size
            );
            // D-D.6 디버깅: pos=0에서 in/out rpcmem byte를 dump해서 microbench
            // (직접 graphExecute, baked rpcmem)와 비교 — `lg.execute` wrap 자체가
            // 결과를 변형하는지 확인.
            if pos == 0
                && std::env::var("LLMRS_QNN_OPPKG_FAST_PATH_DUMP").as_deref() == Ok("1")
            {
                let in_f32 = unsafe {
                    std::slice::from_raw_parts(in_slot.host_ptr as *const f32, 8)
                };
                let out_f32 = unsafe {
                    std::slice::from_raw_parts(out_slot.host_ptr as *const f32, 8)
                };
                eprintln!(
                    "[lg.execute pos=0 n_kv={n_kv}] in_slot[0..8]={:?} out_slot[0..8]={:?}",
                    in_f32, out_f32
                );
            }
            unsafe {
                std::ptr::copy_nonoverlapping(
                    out_slot.host_ptr,
                    x_out_bytes.as_mut_ptr(),
                    x_out_bytes.len(),
                );
            }

            // D-F 옵션 2: KV write back — KvScatter가 graph rpcmem에 새 token을
            // append했으니 caller bytes에도 mirror해서 caller가 OpenCL cl_mem에
            // 반영하게 한다.
            unsafe {
                std::ptr::copy_nonoverlapping(
                    kv_k_slot.host_ptr,
                    kv_k_bytes.as_mut_ptr(),
                    kv_k_bytes.len(),
                );
                std::ptr::copy_nonoverlapping(
                    kv_v_slot.host_ptr,
                    kv_v_bytes.as_mut_ptr(),
                    kv_v_bytes.len(),
                );
            }
            let copy_out_ns = t_copy_out.elapsed().as_nanos() as u64;
            super::LG_COPY_OUT_NS.fetch_add(copy_out_ns, std::sync::atomic::Ordering::Relaxed);

            Ok(())
        }
    }

    /// Helper — Tensor에서 owned host bytes 추출.
    ///
    /// D-F path: weight tensor가 OpenCL secondary로 promote된 경우 buffer는
    /// `UnifiedBuffer` (CL_MEM_ALLOC_HOST_PTR, unmapped 기본 상태). map → copy →
    /// unmap으로 host bytes를 안전하게 가져온다. unmap 후 GPU access 정상 복귀
    /// (production OpenCL forward path에서 cl_mem 사용).
    ///
    /// MmapBuffer / SharedBuffer 등 host-resident buffer는 `as_ptr` 직접 read.
    /// 호출은 graph build 시 1회/weight (28 layer × 7 weight)이라 owned Vec
    /// 복사 비용 무관.
    fn tensor_bytes_owned(t: &Tensor) -> Result<Vec<u8>> {
        let size = t.size();
        let host_ptr = t.as_ptr();
        if !host_ptr.is_null() {
            return Ok(unsafe { std::slice::from_raw_parts(host_ptr, size).to_vec() });
        }
        // UnifiedBuffer (Adreno UMA OpenCL alloc): map → copy → unmap.
        if let Some(ub) = t
            .buffer()
            .as_any()
            .downcast_ref::<crate::buffer::unified_buffer::UnifiedBuffer>()
        {
            let mapped = ub.map()?;
            ensure!(
                !mapped.is_null(),
                "UnifiedBuffer.map() returned null (size={size})"
            );
            let bytes = unsafe { std::slice::from_raw_parts(mapped, size).to_vec() };
            ub.unmap()?;
            return Ok(bytes);
        }
        Err(anyhow!(
            "tensor_bytes_owned: host pointer null and buffer is not UnifiedBuffer (size={size})"
        ))
    }

    /// rpcmem alloc + memRegister 1회. byte_size 단위.
    fn alloc_rpcmem_slot(runtime: &QnnOppkgRuntime, byte_size: usize) -> Result<RpcmemSlot> {
        const RPCMEM_HEAP_ID_SYSTEM: i32 = 25;
        const RPCMEM_DEFAULT_FLAGS: u32 = 1;

        let (rpcmem_alloc, rpcmem_free, rpcmem_to_fd) = runtime.rpcmem_fns();
        let host_ptr = unsafe {
            rpcmem_alloc(
                RPCMEM_HEAP_ID_SYSTEM,
                RPCMEM_DEFAULT_FLAGS,
                byte_size as i32,
            )
        };
        if host_ptr.is_null() {
            return Err(anyhow!(
                "rpcmem_alloc(size={byte_size}) returned NULL (heap exhaustion?)"
            ));
        }
        let fd = unsafe { rpcmem_to_fd(host_ptr) };
        if fd < 0 {
            unsafe { rpcmem_free(host_ptr) };
            return Err(anyhow!("rpcmem_to_fd returned {fd}"));
        }

        // memRegister with byte-typed shape (M2 microbench pattern: per-tensor
        // dim/dtype is overridden at tensor binding time).
        let mut dims: [u32; 1] = [byte_size as u32];
        let desc = ffi::Qnn_MemDescriptor_t {
            memShape: ffi::Qnn_MemShape_t {
                numDim: 1,
                dimSize: dims.as_mut_ptr(),
                shapeConfig: ptr::null(),
            },
            dataType: ffi::Qnn_DataType_t_QNN_DATATYPE_UINT_8,
            memType: ffi::Qnn_MemType_t_QNN_MEM_TYPE_DMA_BUF,
            __bindgen_anon_1: ffi::Qnn_MemDescriptor_t__bindgen_ty_1 {
                dmaBufInfo: ffi::Qnn_MemDmaBufInfo_t { fd, data: host_ptr },
            },
        };
        let mut mh: ffi::Qnn_MemHandle_t = ptr::null_mut();
        let v = runtime.v();
        let mem_register = v
            .memRegister
            .ok_or_else(|| anyhow!("memRegister fn-pointer is NULL"))?;
        let err = unsafe { mem_register(runtime.context(), &desc, 1, &mut mh) };
        if err != 0 {
            unsafe { rpcmem_free(host_ptr) };
            return Err(anyhow!("QnnMem_register(size={byte_size}) err=0x{:x}", err));
        }

        Ok(RpcmemSlot {
            host_ptr: host_ptr as *mut u8,
            fd,
            size: byte_size,
            mem_handle: mh,
        })
    }

    /// rpcmem slot에 host buffer 복사.
    fn copy_into_slot(slot: &RpcmemSlot, src: &[u8]) -> Result<()> {
        ensure!(
            src.len() <= slot.size,
            "src.len()={} > slot.size={}",
            src.len(),
            slot.size
        );
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr(), slot.host_ptr, src.len());
        }
        Ok(())
    }

    /// rpcmem slot에 u16 buffer 복사 (Q4_0 d_halves).
    fn copy_u16_into_slot(slot: &RpcmemSlot, src: &[u16]) -> Result<()> {
        let nbytes = src.len() * 2;
        ensure!(
            nbytes <= slot.size,
            "nbytes={} > slot.size={}",
            nbytes,
            slot.size
        );
        unsafe {
            std::ptr::copy_nonoverlapping(src.as_ptr() as *const u8, slot.host_ptr, nbytes);
        }
        Ok(())
    }

    /// Production GGUF Q4_0 weight (AOS) bytes 추출 + SOA 변환.
    /// `n` = output rows, `k` = input cols.
    fn pack_weight_q4_0(t: &Tensor, n: usize, k: usize) -> Result<(Vec<u8>, Vec<u16>)> {
        let aos = tensor_bytes_owned(t)?;
        let num_blocks = n * k / 32;
        let expected = num_blocks * Q4_0_BLOCK_BYTES;
        ensure!(
            aos.len() >= expected,
            "weight tensor bytes={} < expected {} (n={n}, k={k})",
            aos.len(),
            expected
        );
        Ok(aos_to_soa_q4_0(&aos[..expected], n, k))
    }

    pub(super) fn build_layer_graph(
        runtime: &QnnOppkgRuntime,
        layer_idx: usize,
        weights: &TransformerLayer,
        cfg: &LayerConfig,
    ) -> Result<(AndroidLayerGraph, u32, usize)> {
        if !runtime.is_initialized() {
            return Err(anyhow!(
                "build_layer_graph(layer={layer_idx}) — runtime not initialized"
            ));
        }
        let v = runtime.v();
        let ctx = runtime.context();

        let dim = cfg.dim as usize;
        let n_head = cfg.n_head as usize;
        let n_kv_heads = cfg.n_kv_heads as usize;
        let head_dim = cfg.head_dim as usize;
        let ffn_dim = cfg.ffn_dim as usize;
        let kv_capacity = cfg.kv_capacity as usize;
        let q_proj_out = n_head * head_dim;
        let kv_proj_out = n_kv_heads * head_dim;
        // M3.4 D-D.3: pos는 input tensor (pos_buf)로 graphExecute 직전에
        // host write. graph build 시점은 placeholder 0으로 alloc만.
        // n_kv는 attention 시점 KV cache의 valid token 수. 1로 고정 (single
        // token decode + 매 layer single-token write_pos는 pos_buf로 동적).
        let n_kv: usize = 1;
        // RoPE theta — production은 model.config.rope_theta를 사용하지만 본
        // 단계는 Qwen2.5 기본값.
        let theta: f32 = 1_000_000.0;

        // ── 1. graphCreate (precision=USER_PROVIDED) ───────────────────────
        // D-D.6 디버깅: process-unique counter로 graph name 충돌 방지
        // (fresh_build mode가 같은 layer를 반복 build).
        static GRAPH_NAME_COUNTER: std::sync::atomic::AtomicU64 =
            std::sync::atomic::AtomicU64::new(0);
        let counter = GRAPH_NAME_COUNTER.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        let graph_name = CString::new(format!("qnn_layer_{layer_idx}_{counter}")).unwrap();
        #[repr(C)]
        struct QnnGpuGraphCustomConfig {
            precision: u32,
            disable_memory_optimizations: u8,
            disable_node_optimizations: u8,
            disable_queue_recording: u8,
            _pad: u8,
        }
        const QNN_GPU_PRECISION_USER_PROVIDED: u32 = 3;
        let mut gpu_custom = QnnGpuGraphCustomConfig {
            precision: QNN_GPU_PRECISION_USER_PROVIDED,
            disable_memory_optimizations: 0,
            disable_node_optimizations: 0,
            disable_queue_recording: 0,
            _pad: 0,
        };
        let graph_cfg = ffi::QnnGraph_Config_t {
            option: ffi::QnnGraph_ConfigOption_t_QNN_GRAPH_CONFIG_OPTION_CUSTOM,
            __bindgen_anon_1: ffi::QnnGraph_Config_t__bindgen_ty_1 {
                customConfig: &mut gpu_custom as *mut _ as *mut _,
            },
        };
        let mut configs: [*const ffi::QnnGraph_Config_t; 2] = [&graph_cfg as *const _, ptr::null()];

        let mut graph: ffi::Qnn_GraphHandle_t = ptr::null_mut();
        let graph_create = v
            .graphCreate
            .ok_or_else(|| anyhow!("graphCreate fn-pointer is NULL"))?;
        let err =
            unsafe { graph_create(ctx, graph_name.as_ptr(), configs.as_mut_ptr(), &mut graph) };
        if err != 0 {
            return Err(anyhow!("graphCreate(layer={layer_idx}) err=0x{:x}", err));
        }

        // ── 2. SOA 변환 — 7 Q4_0 weights (production AOS → SOA) ──────────
        let (qq_q, qq_d) = pack_weight_q4_0(&weights.wq, q_proj_out, dim)?;
        let (qk_q, qk_d) = pack_weight_q4_0(&weights.wk, kv_proj_out, dim)?;
        let (qv_q, qv_d) = pack_weight_q4_0(&weights.wv, kv_proj_out, dim)?;
        let (qo_q, qo_d) = pack_weight_q4_0(&weights.wo, dim, q_proj_out)?;
        let (qg_q, qg_d) = pack_weight_q4_0(&weights.w_gate, ffn_dim, dim)?;
        let (qu_q, qu_d) = pack_weight_q4_0(&weights.w_up, ffn_dim, dim)?;
        let (qd_q, qd_d) = pack_weight_q4_0(&weights.w_down, dim, ffn_dim)?;
        // D-D.6 디버깅: layer 0 weight bytes hash dump (FAST_PATH_DUMP 시).
        if layer_idx == 0
            && std::env::var("LLMRS_QNN_OPPKG_FAST_PATH_DUMP").as_deref() == Ok("1")
        {
            let aos = tensor_bytes_owned(&weights.wq)?;
            eprintln!(
                "[graph-build layer=0 wq] aos.len={}, aos[0..16]={:02x?} qq_q.len={} qq_q[0..16]={:02x?} qq_d[0..4]={:?}",
                aos.len(),
                &aos[..16.min(aos.len())],
                qq_q.len(),
                &qq_q[..16.min(qq_q.len())],
                &qq_d[..4.min(qq_d.len())]
            );
            let rms_pre = tensor_bytes_owned(&weights.attention_norm)?;
            let rms_pre_f32 = unsafe {
                std::slice::from_raw_parts(rms_pre.as_ptr() as *const f32, 8.min(rms_pre.len() / 4))
            };
            eprintln!(
                "[graph-build layer=0 attention_norm] f32[0..8]={:?}",
                rms_pre_f32
            );
        }

        // RMS norm weights — production은 F32 (1D, dim).
        let rms_pre_bytes = tensor_bytes_owned(&weights.attention_norm)?;
        let rms_post_bytes = tensor_bytes_owned(&weights.ffn_norm)?;

        // D-D.6 Phase B: Qwen2.5 attention QKV bias — F32 (1D, q_proj_out / kv_proj_out).
        // bias가 없는 모델 (Llama/Gemma)은 zero bytes (BiasAdd effectively no-op).
        let q_bias_bytes = if let Some(ref bias) = weights.qkv_bias {
            tensor_bytes_owned(&bias.bq)?
        } else {
            vec![0u8; q_proj_out * 4]
        };
        let k_bias_bytes = if let Some(ref bias) = weights.qkv_bias {
            tensor_bytes_owned(&bias.bk)?
        } else {
            vec![0u8; kv_proj_out * 4]
        };
        let v_bias_bytes = if let Some(ref bias) = weights.qkv_bias {
            tensor_bytes_owned(&bias.bv)?
        } else {
            vec![0u8; kv_proj_out * 4]
        };

        // ── 3. rpcmem allocations (per-tensor) ─────────────────────────────
        let bytes_x = dim * 4;
        let bytes_rms = dim * 4;
        let bytes_kv = n_kv_heads * kv_capacity * head_dim * 2; // F16
        let bytes_x_out = dim * 4;
        // D-D.6: mask sized to max kv_capacity F16 (n_kv now dynamic).
        // dims_mask와 size 일치 필수 — 이전 `n_kv * 2 = 2 bytes`는 graph가
        // 4096 bytes mask를 read할 때 OOB read로 garbage attention 발생.
        let _ = n_kv; // n_kv는 build 시점 placeholder 1, mask size에 미사용.
        let bytes_mask = kv_capacity * 2; // F16, [kv_capacity]
        let bytes_sinks = n_head * 4; // F32
        let bytes_dummy = 4_usize;

        let mut slots: Vec<RpcmemSlot> = Vec::with_capacity(23);

        let slot_x = alloc_rpcmem_slot(runtime, bytes_x)?;
        slots.push(slot_x);
        let idx_x_in = slots.len() - 1;

        let slot_rms_pre = alloc_rpcmem_slot(runtime, bytes_rms)?;
        copy_into_slot(&slot_rms_pre, &rms_pre_bytes)?;
        slots.push(slot_rms_pre);

        let slot_rms_post = alloc_rpcmem_slot(runtime, bytes_rms)?;
        copy_into_slot(&slot_rms_post, &rms_post_bytes)?;
        slots.push(slot_rms_post);

        // 7× (q_bytes, d_halves) — Q4_0 SOA pairs.
        macro_rules! push_q40_pair {
            ($q:expr, $d:expr) => {{
                let s_q = alloc_rpcmem_slot(runtime, $q.len())?;
                copy_into_slot(&s_q, &$q)?;
                slots.push(s_q);
                let s_d = alloc_rpcmem_slot(runtime, $d.len() * 2)?;
                copy_u16_into_slot(&s_d, &$d)?;
                slots.push(s_d);
            }};
        }
        push_q40_pair!(qq_q, qq_d);
        push_q40_pair!(qk_q, qk_d);
        push_q40_pair!(qv_q, qv_d);
        push_q40_pair!(qo_q, qo_d);
        push_q40_pair!(qg_q, qg_d);
        push_q40_pair!(qu_q, qu_d);
        push_q40_pair!(qd_q, qd_d);

        // KV cache K/V — APP_READ outputs (multi-output KvScatter).
        let slot_kc = alloc_rpcmem_slot(runtime, bytes_kv)?;
        unsafe { std::ptr::write_bytes(slot_kc.host_ptr, 0, bytes_kv) };
        slots.push(slot_kc);
        let idx_kv_k = slots.len() - 1;
        let slot_vc = alloc_rpcmem_slot(runtime, bytes_kv)?;
        unsafe { std::ptr::write_bytes(slot_vc.host_ptr, 0, bytes_kv) };
        slots.push(slot_vc);
        let idx_kv_v = slots.len() - 1;

        // mask / sinks / score / x_out.
        let slot_mask = alloc_rpcmem_slot(runtime, bytes_mask)?;
        unsafe { std::ptr::write_bytes(slot_mask.host_ptr, 0, bytes_mask) };
        slots.push(slot_mask);

        let slot_sinks = alloc_rpcmem_slot(runtime, bytes_sinks)?;
        // sinks = -1e30 per head (M2 INV pattern: neutralise sink-attention).
        let neg_huge: f32 = -1.0e30;
        unsafe {
            for i in 0..n_head {
                std::ptr::write_unaligned(
                    (slot_sinks.host_ptr as *mut u8).add(i * 4) as *mut f32,
                    neg_huge,
                );
            }
        }
        slots.push(slot_sinks);

        let slot_score = alloc_rpcmem_slot(runtime, bytes_dummy)?;
        unsafe { std::ptr::write_bytes(slot_score.host_ptr, 0, bytes_dummy) };
        slots.push(slot_score);

        let slot_xout = alloc_rpcmem_slot(runtime, bytes_x_out)?;
        unsafe { std::ptr::write_bytes(slot_xout.host_ptr, 0, bytes_x_out) };
        slots.push(slot_xout);
        let idx_x_out = slots.len() - 1;

        // M3.4 D-D.3: pos_buf — INT_32 [1] (4 bytes). RoPE Q/K + KvScatter
        // 3 nodes가 input tensor로 share. graph build 시점은 placeholder 0,
        // execute 시점에 host pointer로 갱신.
        let bytes_pos = 4_usize;
        let slot_pos = alloc_rpcmem_slot(runtime, bytes_pos)?;
        unsafe {
            std::ptr::write(slot_pos.host_ptr as *mut i32, 0i32);
        }
        let pos_host_ptr = slot_pos.host_ptr as *mut i32;
        slots.push(slot_pos);
        let idx_pos = slots.len() - 1;

        // M3.4 D-D.6: n_kv_buf — INT_32 [1]. FlashAttn node가 input tensor로
        // 사용하여 매 graphExecute 직전 host write로 동적 n_kv 갱신.
        // build 시점은 placeholder 1.
        let bytes_n_kv = 4_usize;
        let slot_n_kv = alloc_rpcmem_slot(runtime, bytes_n_kv)?;
        unsafe {
            std::ptr::write(slot_n_kv.host_ptr as *mut i32, 1i32);
        }
        let n_kv_host_ptr = slot_n_kv.host_ptr as *mut i32;
        slots.push(slot_n_kv);
        let idx_n_kv = slots.len() - 1;

        // D-D.6 Phase B: QKV bias 3개 — 끝에 push (기존 slot indexing 보존).
        let slot_q_bias = alloc_rpcmem_slot(runtime, q_bias_bytes.len())?;
        copy_into_slot(&slot_q_bias, &q_bias_bytes)?;
        slots.push(slot_q_bias);
        let idx_q_bias = slots.len() - 1;
        let slot_k_bias = alloc_rpcmem_slot(runtime, k_bias_bytes.len())?;
        copy_into_slot(&slot_k_bias, &k_bias_bytes)?;
        slots.push(slot_k_bias);
        let idx_k_bias = slots.len() - 1;
        let slot_v_bias = alloc_rpcmem_slot(runtime, v_bias_bytes.len())?;
        copy_into_slot(&slot_v_bias, &v_bias_bytes)?;
        slots.push(slot_v_bias);
        let idx_v_bias = slots.len() - 1;

        // ── 4. Build tensors + register — 36 tensors total ─────────────────
        // Dimensions storage. Each Vec must outlive graph build (pointer stored).
        let mut dims_x: Vec<u32> = vec![1, dim as u32];
        let mut dims_rms_pre: Vec<u32> = vec![dim as u32];
        let mut dims_rms_post: Vec<u32> = vec![dim as u32];
        let mut dims_y1: Vec<u32> = vec![1, dim as u32];
        let mut dims_qq: Vec<u32> = vec![qq_q.len() as u32];
        let mut dims_qd: Vec<u32> = vec![qq_d.len() as u32];
        let mut dims_kq: Vec<u32> = vec![qk_q.len() as u32];
        let mut dims_kd: Vec<u32> = vec![qk_d.len() as u32];
        let mut dims_vq: Vec<u32> = vec![qv_q.len() as u32];
        let mut dims_vd: Vec<u32> = vec![qv_d.len() as u32];
        let mut dims_oq: Vec<u32> = vec![qo_q.len() as u32];
        let mut dims_od: Vec<u32> = vec![qo_d.len() as u32];
        let mut dims_gq: Vec<u32> = vec![qg_q.len() as u32];
        let mut dims_gd: Vec<u32> = vec![qg_d.len() as u32];
        let mut dims_uq: Vec<u32> = vec![qu_q.len() as u32];
        let mut dims_ud: Vec<u32> = vec![qu_d.len() as u32];
        let mut dims_dq: Vec<u32> = vec![qd_q.len() as u32];
        let mut dims_dd: Vec<u32> = vec![qd_d.len() as u32];
        let mut dims_q: Vec<u32> = vec![1, n_head as u32, head_dim as u32];
        let mut dims_kvec: Vec<u32> = vec![1, n_kv_heads as u32, head_dim as u32];
        let mut dims_vvec: Vec<u32> = vec![1, n_kv_heads as u32, head_dim as u32];
        // D-D.6 Phase B: bias 1D + biased intermediate (post-bias matmul).
        let mut dims_q_bias: Vec<u32> = vec![q_proj_out as u32];
        let mut dims_k_bias: Vec<u32> = vec![kv_proj_out as u32];
        let mut dims_v_bias: Vec<u32> = vec![kv_proj_out as u32];
        let mut dims_q_biased: Vec<u32> = vec![1, n_head as u32, head_dim as u32];
        let mut dims_k_biased: Vec<u32> = vec![1, n_kv_heads as u32, head_dim as u32];
        let mut dims_v_biased: Vec<u32> = vec![1, n_kv_heads as u32, head_dim as u32];
        let mut dims_q_rope: Vec<u32> = vec![1, n_head as u32, head_dim as u32];
        let mut dims_k_rope: Vec<u32> = vec![1, n_kv_heads as u32, head_dim as u32];
        let mut dims_kcache: Vec<u32> =
            vec![1, n_kv_heads as u32, kv_capacity as u32, head_dim as u32];
        let mut dims_vcache: Vec<u32> = dims_kcache.clone();
        let mut dims_attn_o: Vec<u32> = vec![1, n_head as u32, head_dim as u32];
        let mut dims_o: Vec<u32> = vec![1, dim as u32];
        let mut dims_x_attn: Vec<u32> = vec![1, dim as u32];
        let mut dims_y2: Vec<u32> = vec![1, dim as u32];
        let mut dims_gate: Vec<u32> = vec![1, ffn_dim as u32];
        let mut dims_up: Vec<u32> = vec![1, ffn_dim as u32];
        let mut dims_silu_out: Vec<u32> = vec![1, ffn_dim as u32];
        let mut dims_down: Vec<u32> = vec![1, dim as u32];
        let mut dims_x_out: Vec<u32> = vec![1, dim as u32];
        // D-D.6: mask sized to max kv_capacity (n_kv now dynamic).
        let mut dims_mask: Vec<u32> = vec![kv_capacity as u32];
        let mut dims_sinks: Vec<u32> = vec![n_head as u32];
        let mut dims_score: Vec<u32> = vec![1];
        // M3.4 D-D.3: pos_buf — INT_32 rank-1 [1].
        let mut dims_pos: Vec<u32> = vec![1];
        // D-D.6: n_kv_buf — INT_32 rank-1 [1]. FlashAttn input.
        let mut dims_n_kv: Vec<u32> = vec![1];

        // CString tensor names — must outlive graph build.
        let nm_x = CString::new("x").unwrap();
        let nm_rms_pre = CString::new("rms_pre").unwrap();
        let nm_rms_post = CString::new("rms_post").unwrap();
        let nm_y1 = CString::new("y1").unwrap();
        let nm_qq = CString::new("qq").unwrap();
        let nm_qd = CString::new("qd").unwrap();
        let nm_kq = CString::new("kq").unwrap();
        let nm_kd = CString::new("kd").unwrap();
        let nm_vq = CString::new("vq").unwrap();
        let nm_vd = CString::new("vd").unwrap();
        let nm_oq = CString::new("oq").unwrap();
        let nm_od = CString::new("od").unwrap();
        let nm_gq = CString::new("gq").unwrap();
        let nm_gd = CString::new("gd").unwrap();
        let nm_uq = CString::new("uq").unwrap();
        let nm_ud = CString::new("ud").unwrap();
        let nm_dq = CString::new("dq").unwrap();
        let nm_dd = CString::new("dd").unwrap();
        let nm_q = CString::new("q").unwrap();
        let nm_k = CString::new("k").unwrap();
        let nm_v = CString::new("v").unwrap();
        let nm_q_rope = CString::new("q_rope").unwrap();
        let nm_k_rope = CString::new("k_rope").unwrap();
        let nm_kcache = CString::new("kcache").unwrap();
        let nm_vcache = CString::new("vcache").unwrap();
        let nm_attn_o = CString::new("attn_o").unwrap();
        let nm_o = CString::new("o").unwrap();
        let nm_x_attn = CString::new("x_attn").unwrap();
        let nm_y2 = CString::new("y2").unwrap();
        let nm_gate = CString::new("gate").unwrap();
        let nm_up = CString::new("up").unwrap();
        let nm_silu_out = CString::new("silu_out").unwrap();
        let nm_down = CString::new("down").unwrap();
        let nm_x_out = CString::new("x_out").unwrap();
        let nm_mask = CString::new("mask").unwrap();
        let nm_sinks = CString::new("sinks").unwrap();
        let nm_score = CString::new("score").unwrap();
        let nm_pos = CString::new("pos").unwrap();
        let nm_n_kv = CString::new("n_kv_buf").unwrap();
        // D-D.6 Phase B: bias tensor names.
        let nm_q_bias = CString::new("q_bias").unwrap();
        let nm_k_bias = CString::new("k_bias").unwrap();
        let nm_v_bias = CString::new("v_bias").unwrap();
        let nm_q_biased = CString::new("q_biased").unwrap();
        let nm_k_biased = CString::new("k_biased").unwrap();
        let nm_v_biased = CString::new("v_biased").unwrap();

        let qp = ffi::Qnn_QuantizeParams_t {
            encodingDefinition: ffi::Qnn_Definition_t_QNN_DEFINITION_UNDEFINED,
            quantizationEncoding:
                ffi::Qnn_QuantizationEncoding_t_QNN_QUANTIZATION_ENCODING_UNDEFINED,
            __bindgen_anon_1: ffi::Qnn_QuantizeParams_t__bindgen_ty_1 {
                scaleOffsetEncoding: ffi::Qnn_ScaleOffset_t {
                    scale: 0.0,
                    offset: 0,
                },
            },
        };
        let mk_tv1 = |ttype, dtype, rank: u32, dims_ptr: *mut u32| ffi::Qnn_TensorV1_t {
            id: 0,
            name: ptr::null(),
            type_: ttype,
            dataFormat: ffi::QNN_TENSOR_DATA_FORMAT_FLAT_BUFFER,
            dataType: dtype,
            quantizeParams: qp,
            rank,
            dimensions: dims_ptr,
            memType: ffi::Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_RAW,
            __bindgen_anon_1: ffi::Qnn_TensorV1_t__bindgen_ty_1 {
                clientBuf: ffi::Qnn_ClientBuffer_t {
                    data: ptr::null_mut(),
                    dataSize: 0,
                },
            },
        };
        let build_tensor = |ttype, dtype, rank: u32, dims_ptr: *mut u32, name: *const c_char| {
            let mut t = ffi::Qnn_Tensor_t {
                version: ffi::Qnn_TensorVersion_t_QNN_TENSOR_VERSION_1,
                __bindgen_anon_1: ffi::Qnn_Tensor_t__bindgen_ty_1 {
                    v1: mk_tv1(ttype, dtype, rank, dims_ptr),
                },
            };
            t.__bindgen_anon_1.v1.name = name;
            t
        };

        let app_w = ffi::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_WRITE;
        let app_r = ffi::Qnn_TensorType_t_QNN_TENSOR_TYPE_APP_READ;
        let native = ffi::Qnn_TensorType_t_QNN_TENSOR_TYPE_NATIVE;
        let f32_t = ffi::Qnn_DataType_t_QNN_DATATYPE_FLOAT_32;
        let f16_t = ffi::Qnn_DataType_t_QNN_DATATYPE_FLOAT_16;
        let u8_t = ffi::Qnn_DataType_t_QNN_DATATYPE_UINT_8;
        let i32_t = ffi::Qnn_DataType_t_QNN_DATATYPE_INT_32;

        // Endpoints (APP_WRITE / APP_READ).
        let mut t_x = build_tensor(app_w, f32_t, 2, dims_x.as_mut_ptr(), nm_x.as_ptr());
        let mut t_rms_pre = build_tensor(
            app_w,
            f32_t,
            1,
            dims_rms_pre.as_mut_ptr(),
            nm_rms_pre.as_ptr(),
        );
        let mut t_rms_post = build_tensor(
            app_w,
            f32_t,
            1,
            dims_rms_post.as_mut_ptr(),
            nm_rms_post.as_ptr(),
        );
        let mut t_qq = build_tensor(app_w, u8_t, 1, dims_qq.as_mut_ptr(), nm_qq.as_ptr());
        let mut t_qd = build_tensor(app_w, f16_t, 1, dims_qd.as_mut_ptr(), nm_qd.as_ptr());
        let mut t_kq = build_tensor(app_w, u8_t, 1, dims_kq.as_mut_ptr(), nm_kq.as_ptr());
        let mut t_kd = build_tensor(app_w, f16_t, 1, dims_kd.as_mut_ptr(), nm_kd.as_ptr());
        let mut t_vq = build_tensor(app_w, u8_t, 1, dims_vq.as_mut_ptr(), nm_vq.as_ptr());
        let mut t_vd = build_tensor(app_w, f16_t, 1, dims_vd.as_mut_ptr(), nm_vd.as_ptr());
        let mut t_oq = build_tensor(app_w, u8_t, 1, dims_oq.as_mut_ptr(), nm_oq.as_ptr());
        let mut t_od = build_tensor(app_w, f16_t, 1, dims_od.as_mut_ptr(), nm_od.as_ptr());
        let mut t_gq = build_tensor(app_w, u8_t, 1, dims_gq.as_mut_ptr(), nm_gq.as_ptr());
        let mut t_gd = build_tensor(app_w, f16_t, 1, dims_gd.as_mut_ptr(), nm_gd.as_ptr());
        let mut t_uq = build_tensor(app_w, u8_t, 1, dims_uq.as_mut_ptr(), nm_uq.as_ptr());
        let mut t_ud = build_tensor(app_w, f16_t, 1, dims_ud.as_mut_ptr(), nm_ud.as_ptr());
        let mut t_dq = build_tensor(app_w, u8_t, 1, dims_dq.as_mut_ptr(), nm_dq.as_ptr());
        let mut t_dd = build_tensor(app_w, f16_t, 1, dims_dd.as_mut_ptr(), nm_dd.as_ptr());
        // KV cache: APP_READ (multi-output KvScatter — M2.H 결정).
        let mut t_kcache = build_tensor(
            app_r,
            f16_t,
            4,
            dims_kcache.as_mut_ptr(),
            nm_kcache.as_ptr(),
        );
        let mut t_vcache = build_tensor(
            app_r,
            f16_t,
            4,
            dims_vcache.as_mut_ptr(),
            nm_vcache.as_ptr(),
        );
        let mut t_mask = build_tensor(app_w, f16_t, 1, dims_mask.as_mut_ptr(), nm_mask.as_ptr());
        let mut t_sinks = build_tensor(app_w, f32_t, 1, dims_sinks.as_mut_ptr(), nm_sinks.as_ptr());
        let mut t_score = build_tensor(app_w, f32_t, 1, dims_score.as_mut_ptr(), nm_score.as_ptr());
        let mut t_x_out = build_tensor(app_r, f32_t, 2, dims_x_out.as_mut_ptr(), nm_x_out.as_ptr());
        // M3.4 D-D.3: pos input tensor — APP_WRITE INT_32 rank-1 [1].
        let mut t_pos = build_tensor(app_w, i32_t, 1, dims_pos.as_mut_ptr(), nm_pos.as_ptr());
        // M3.4 D-D.6: n_kv_buf input tensor — APP_WRITE INT_32 rank-1 [1].
        let mut t_n_kv = build_tensor(app_w, i32_t, 1, dims_n_kv.as_mut_ptr(), nm_n_kv.as_ptr());
        // D-D.6 Phase B: bias tensors — APP_WRITE F32 rank-1.
        let mut t_q_bias = build_tensor(
            app_w,
            f32_t,
            1,
            dims_q_bias.as_mut_ptr(),
            nm_q_bias.as_ptr(),
        );
        let mut t_k_bias = build_tensor(
            app_w,
            f32_t,
            1,
            dims_k_bias.as_mut_ptr(),
            nm_k_bias.as_ptr(),
        );
        let mut t_v_bias = build_tensor(
            app_w,
            f32_t,
            1,
            dims_v_bias.as_mut_ptr(),
            nm_v_bias.as_ptr(),
        );

        // Intermediates (NATIVE).
        let mut t_y1 = build_tensor(native, f32_t, 2, dims_y1.as_mut_ptr(), nm_y1.as_ptr());
        let mut t_q = build_tensor(native, f32_t, 3, dims_q.as_mut_ptr(), nm_q.as_ptr());
        let mut t_k = build_tensor(native, f32_t, 3, dims_kvec.as_mut_ptr(), nm_k.as_ptr());
        let mut t_v = build_tensor(native, f32_t, 3, dims_vvec.as_mut_ptr(), nm_v.as_ptr());
        // D-D.6 Phase B: post-bias intermediates (BiasAdd output).
        let mut t_q_biased = build_tensor(
            native,
            f32_t,
            3,
            dims_q_biased.as_mut_ptr(),
            nm_q_biased.as_ptr(),
        );
        let mut t_k_biased = build_tensor(
            native,
            f32_t,
            3,
            dims_k_biased.as_mut_ptr(),
            nm_k_biased.as_ptr(),
        );
        let mut t_v_biased = build_tensor(
            native,
            f32_t,
            3,
            dims_v_biased.as_mut_ptr(),
            nm_v_biased.as_ptr(),
        );
        let mut t_q_rope = build_tensor(
            native,
            f32_t,
            3,
            dims_q_rope.as_mut_ptr(),
            nm_q_rope.as_ptr(),
        );
        let mut t_k_rope = build_tensor(
            native,
            f32_t,
            3,
            dims_k_rope.as_mut_ptr(),
            nm_k_rope.as_ptr(),
        );
        let mut t_attn_o = build_tensor(
            native,
            f32_t,
            3,
            dims_attn_o.as_mut_ptr(),
            nm_attn_o.as_ptr(),
        );
        let mut t_o = build_tensor(native, f32_t, 2, dims_o.as_mut_ptr(), nm_o.as_ptr());
        let mut t_x_attn = build_tensor(
            native,
            f32_t,
            2,
            dims_x_attn.as_mut_ptr(),
            nm_x_attn.as_ptr(),
        );
        let mut t_y2 = build_tensor(native, f32_t, 2, dims_y2.as_mut_ptr(), nm_y2.as_ptr());
        let mut t_gate = build_tensor(native, f32_t, 2, dims_gate.as_mut_ptr(), nm_gate.as_ptr());
        let mut t_up = build_tensor(native, f32_t, 2, dims_up.as_mut_ptr(), nm_up.as_ptr());
        let mut t_silu_out = build_tensor(
            native,
            f32_t,
            2,
            dims_silu_out.as_mut_ptr(),
            nm_silu_out.as_ptr(),
        );
        let mut t_down = build_tensor(native, f32_t, 2, dims_down.as_mut_ptr(), nm_down.as_ptr());

        // Tensor 등록.
        let registrations: &mut [(&str, &mut ffi::Qnn_Tensor_t)] = &mut [
            ("x", &mut t_x),
            ("rms_pre", &mut t_rms_pre),
            ("rms_post", &mut t_rms_post),
            ("qq", &mut t_qq),
            ("qd", &mut t_qd),
            ("kq", &mut t_kq),
            ("kd", &mut t_kd),
            ("vq", &mut t_vq),
            ("vd", &mut t_vd),
            ("oq", &mut t_oq),
            ("od", &mut t_od),
            ("gq", &mut t_gq),
            ("gd", &mut t_gd),
            ("uq", &mut t_uq),
            ("ud", &mut t_ud),
            ("dq", &mut t_dq),
            ("dd", &mut t_dd),
            ("kcache", &mut t_kcache),
            ("vcache", &mut t_vcache),
            ("mask", &mut t_mask),
            ("sinks", &mut t_sinks),
            ("score", &mut t_score),
            ("x_out", &mut t_x_out),
            ("y1", &mut t_y1),
            ("q", &mut t_q),
            ("k", &mut t_k),
            ("v", &mut t_v),
            ("q_rope", &mut t_q_rope),
            ("k_rope", &mut t_k_rope),
            ("attn_o", &mut t_attn_o),
            ("o", &mut t_o),
            ("x_attn", &mut t_x_attn),
            ("y2", &mut t_y2),
            ("gate", &mut t_gate),
            ("up", &mut t_up),
            ("silu_out", &mut t_silu_out),
            ("down", &mut t_down),
            ("pos", &mut t_pos),
            ("n_kv_buf", &mut t_n_kv),
            // D-D.6 Phase B: bias inputs + post-bias intermediates.
            ("q_bias", &mut t_q_bias),
            ("k_bias", &mut t_k_bias),
            ("v_bias", &mut t_v_bias),
            ("q_biased", &mut t_q_biased),
            ("k_biased", &mut t_k_biased),
            ("v_biased", &mut t_v_biased),
        ];
        let tcg = v
            .tensorCreateGraphTensor
            .ok_or_else(|| anyhow!("tensorCreateGraphTensor NULL"))?;
        for (label, t) in registrations.iter_mut() {
            let err = unsafe { tcg(graph, *t) };
            if err != 0 {
                return Err(anyhow!(
                    "tensorCreate(layer={layer_idx}, {label}) err=0x{:x}",
                    err
                ));
            }
        }

        // ── 5. 14 nodes ────────────────────────────────────────────────────
        let pkg = CString::new("qnn_oppkg").unwrap();
        let make_op = |name: *const c_char,
                       typ: *const c_char,
                       ins: *mut ffi::Qnn_Tensor_t,
                       n_in: u32,
                       outs: *mut ffi::Qnn_Tensor_t,
                       n_out: u32,
                       params: *mut ffi::Qnn_Param_t,
                       n_params: u32|
         -> ffi::Qnn_OpConfig_t {
            ffi::Qnn_OpConfig_t {
                version: ffi::Qnn_OpConfigVersion_t_QNN_OPCONFIG_VERSION_1,
                __bindgen_anon_1: ffi::Qnn_OpConfig_t__bindgen_ty_1 {
                    v1: ffi::Qnn_OpConfigV1_t {
                        name,
                        packageName: pkg.as_ptr(),
                        typeName: typ,
                        numOfParams: n_params,
                        params,
                        numOfInputs: n_in,
                        inputTensors: ins,
                        numOfOutputs: n_out,
                        outputTensors: outs,
                    },
                },
            }
        };

        // Param names.
        // M3.4 D-D.3: `start_pos` / `write_pos` SCALAR params 제거 — 둘 다
        // pos_buf input tensor로 동적 처리.
        let pn_theta = CString::new("theta").unwrap();
        let pn_head_dim = CString::new("head_dim").unwrap();
        let pn_capacity = CString::new("capacity").unwrap();
        // D-D.6: pn_n_kv SCALAR 제거 — n_kv는 input tensor (n_kv_buf).
        let pn_n_head = CString::new("n_head").unwrap();
        let pn_n_head_kv = CString::new("n_head_kv").unwrap();
        let pn_kv_capacity = CString::new("kv_capacity").unwrap();
        let pn_head_dim_fa = CString::new("head_dim").unwrap();

        // M3.4 D-D.3: RoPE param은 theta SCALAR 1개만. start_pos는 pos_buf
        // (input tensor)로 매 graphExecute 직전 host write.
        let mk_rope_params = |th: f32| {
            [ffi::Qnn_Param_t {
                paramType: ffi::Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
                name: pn_theta.as_ptr(),
                __bindgen_anon_1: ffi::Qnn_Param_t__bindgen_ty_1 {
                    scalarParam: ffi::Qnn_Scalar_t {
                        dataType: ffi::Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
                        __bindgen_anon_1: ffi::Qnn_Scalar_t__bindgen_ty_1 { floatValue: th },
                    },
                },
            }]
        };
        let mut rope_q_params = mk_rope_params(theta);
        let mut rope_k_params = mk_rope_params(theta);

        let scalar_i32 = |name: *const c_char, val: i32| ffi::Qnn_Param_t {
            paramType: ffi::Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
            name,
            __bindgen_anon_1: ffi::Qnn_Param_t__bindgen_ty_1 {
                scalarParam: ffi::Qnn_Scalar_t {
                    dataType: ffi::Qnn_DataType_t_QNN_DATATYPE_INT_32,
                    __bindgen_anon_1: ffi::Qnn_Scalar_t__bindgen_ty_1 { int32Value: val },
                },
            },
        };

        // M3.4 D-D.3: write_pos SCALAR 제거. head_dim + capacity 2개만 SCALAR.
        let mut kvs_params = [
            scalar_i32(pn_head_dim.as_ptr(), head_dim as i32),
            scalar_i32(pn_capacity.as_ptr(), kv_capacity as i32),
        ];

        // D-D.6: n_kv SCALAR 제거 — n_kv_buf input tensor로 동적 처리.
        let mut fa_params = [
            scalar_i32(pn_n_head.as_ptr(), n_head as i32),
            scalar_i32(pn_n_head_kv.as_ptr(), n_kv_heads as i32),
            scalar_i32(pn_kv_capacity.as_ptr(), kv_capacity as i32),
            scalar_i32(pn_head_dim_fa.as_ptr(), head_dim as i32),
        ];

        // Op type names.
        let ot_rms = CString::new("CustomRmsNorm").unwrap();
        let ot_q40 = CString::new("CustomMatMulQ40F32").unwrap();
        let ot_rope = CString::new("CustomRope").unwrap();
        let ot_kvs = CString::new("CustomKvScatter").unwrap();
        let ot_fa = CString::new("CustomFlashAttn").unwrap();
        let ot_silu = CString::new("CustomSiluMul").unwrap();
        let ot_add = CString::new("CustomAdd").unwrap();
        // D-D.6 Phase B: bias add op type.
        let ot_bias = CString::new("CustomBiasAdd").unwrap();

        // Op names.
        let on_rms_pre = CString::new("rms_pre_op").unwrap();
        let on_q_proj = CString::new("q_proj").unwrap();
        let on_k_proj = CString::new("k_proj").unwrap();
        let on_v_proj = CString::new("v_proj").unwrap();
        let on_rope_q = CString::new("rope_q").unwrap();
        let on_rope_k = CString::new("rope_k").unwrap();
        let on_kvs = CString::new("kvs").unwrap();
        let on_fa = CString::new("fa").unwrap();
        let on_o_proj = CString::new("o_proj").unwrap();
        let on_add1 = CString::new("add1").unwrap();
        let on_rms_post = CString::new("rms_post_op").unwrap();
        let on_gate_proj = CString::new("gate_proj").unwrap();
        let on_up_proj = CString::new("up_proj").unwrap();
        let on_silu = CString::new("silu").unwrap();
        let on_down_proj = CString::new("down_proj").unwrap();
        let on_add2 = CString::new("add2").unwrap();
        // D-D.6 Phase B: bias op names.
        let on_q_bias = CString::new("q_bias_add").unwrap();
        let on_k_bias = CString::new("k_bias_add").unwrap();
        let on_v_bias = CString::new("v_bias_add").unwrap();

        let mut in_rms_pre = [t_x, t_rms_pre];
        let mut out_rms_pre = [t_y1];
        let mut in_q_proj = [t_qq, t_qd, t_y1];
        let mut out_q_proj = [t_q];
        let mut in_k_proj = [t_kq, t_kd, t_y1];
        let mut out_k_proj = [t_k];
        let mut in_v_proj = [t_vq, t_vd, t_y1];
        let mut out_v_proj = [t_v];
        // D-D.6 Phase B: BiasAdd nodes (Q/K/V matmul 직후 → RoPE/KvScatter 입력).
        let mut in_q_bias_node = [t_q, t_q_bias];
        let mut out_q_bias_node = [t_q_biased];
        let mut in_k_bias_node = [t_k, t_k_bias];
        let mut out_k_bias_node = [t_k_biased];
        let mut in_v_bias_node = [t_v, t_v_bias];
        let mut out_v_bias_node = [t_v_biased];
        // M3.4 D-D.3: RoPE Q/K input은 (x_in, pos_buf) 2개. KvScatter input은
        // (k_src, v_src, pos_buf) 3개. 동일 `t_pos`를 share — read-only.
        // D-D.6 Phase B: x_in을 post-bias intermediate로 변경 (t_q→t_q_biased 등).
        let mut in_rope_q = [t_q_biased, t_pos];
        let mut out_rope_q = [t_q_rope];
        let mut in_rope_k = [t_k_biased, t_pos];
        let mut out_rope_k = [t_k_rope];
        let mut in_kvs = [t_k_rope, t_v_biased, t_pos];
        let mut out_kvs = [t_kcache, t_vcache];
        // D-D.6: FlashAttn input은 (q, k, v, mask, sinks, score, n_kv_buf) 7개.
        let mut in_fa = [t_q_rope, t_kcache, t_vcache, t_mask, t_sinks, t_score, t_n_kv];
        let mut out_fa = [t_attn_o];
        let mut in_o_proj = [t_oq, t_od, t_attn_o];
        let mut out_o_proj = [t_o];
        let mut in_add1 = [t_o, t_x];
        let mut out_add1 = [t_x_attn];
        let mut in_rms_post = [t_x_attn, t_rms_post];
        let mut out_rms_post = [t_y2];
        let mut in_gate = [t_gq, t_gd, t_y2];
        let mut out_gate = [t_gate];
        let mut in_up = [t_uq, t_ud, t_y2];
        let mut out_up = [t_up];
        let mut in_silu = [t_gate, t_up];
        let mut out_silu = [t_silu_out];
        let mut in_down = [t_dq, t_dd, t_silu_out];
        let mut out_down = [t_down];
        let mut in_add2 = [t_down, t_x_attn];
        let mut out_add2 = [t_x_out];

        type NodeSpec<'a> = (
            *const c_char,
            *const c_char,
            *mut ffi::Qnn_Tensor_t,
            u32,
            *mut ffi::Qnn_Tensor_t,
            u32,
            *mut ffi::Qnn_Param_t,
            u32,
            &'a str,
        );
        let nodes: [NodeSpec<'_>; 17] = [
            (
                on_rms_pre.as_ptr(),
                ot_rms.as_ptr(),
                in_rms_pre.as_mut_ptr(),
                2,
                out_rms_pre.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "RmsNorm(pre)",
            ),
            (
                on_q_proj.as_ptr(),
                ot_q40.as_ptr(),
                in_q_proj.as_mut_ptr(),
                3,
                out_q_proj.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "Q proj",
            ),
            (
                on_k_proj.as_ptr(),
                ot_q40.as_ptr(),
                in_k_proj.as_mut_ptr(),
                3,
                out_k_proj.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "K proj",
            ),
            (
                on_v_proj.as_ptr(),
                ot_q40.as_ptr(),
                in_v_proj.as_mut_ptr(),
                3,
                out_v_proj.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "V proj",
            ),
            // D-D.6 Phase B: 3 BiasAdd ops (Q/K/V matmul 직후 → RoPE/KvScatter 입력).
            (
                on_q_bias.as_ptr(),
                ot_bias.as_ptr(),
                in_q_bias_node.as_mut_ptr(),
                2,
                out_q_bias_node.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "Q BiasAdd",
            ),
            (
                on_k_bias.as_ptr(),
                ot_bias.as_ptr(),
                in_k_bias_node.as_mut_ptr(),
                2,
                out_k_bias_node.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "K BiasAdd",
            ),
            (
                on_v_bias.as_ptr(),
                ot_bias.as_ptr(),
                in_v_bias_node.as_mut_ptr(),
                2,
                out_v_bias_node.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "V BiasAdd",
            ),
            (
                on_rope_q.as_ptr(),
                ot_rope.as_ptr(),
                in_rope_q.as_mut_ptr(),
                2, // M3.4 D-D.3: x_in + pos_buf
                out_rope_q.as_mut_ptr(),
                1,
                rope_q_params.as_mut_ptr(),
                1, // M3.4 D-D.3: theta only (start_pos 제거)
                "RoPE Q",
            ),
            (
                on_rope_k.as_ptr(),
                ot_rope.as_ptr(),
                in_rope_k.as_mut_ptr(),
                2, // M3.4 D-D.3: x_in + pos_buf
                out_rope_k.as_mut_ptr(),
                1,
                rope_k_params.as_mut_ptr(),
                1, // M3.4 D-D.3: theta only
                "RoPE K",
            ),
            (
                on_kvs.as_ptr(),
                ot_kvs.as_ptr(),
                in_kvs.as_mut_ptr(),
                3, // M3.4 D-D.3: k_src + v_src + pos_buf
                out_kvs.as_mut_ptr(),
                2,
                kvs_params.as_mut_ptr(),
                2, // M3.4 D-D.3: head_dim + capacity (write_pos 제거)
                "KvScatter",
            ),
            (
                on_fa.as_ptr(),
                ot_fa.as_ptr(),
                in_fa.as_mut_ptr(),
                7, // D-D.6: q + k + v + mask + sinks + score + n_kv_buf
                out_fa.as_mut_ptr(),
                1,
                fa_params.as_mut_ptr(),
                4, // D-D.6: n_head + n_head_kv + kv_capacity + head_dim (n_kv 제거)
                "FlashAttn",
            ),
            (
                on_o_proj.as_ptr(),
                ot_q40.as_ptr(),
                in_o_proj.as_mut_ptr(),
                3,
                out_o_proj.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "O proj",
            ),
            (
                on_add1.as_ptr(),
                ot_add.as_ptr(),
                in_add1.as_mut_ptr(),
                2,
                out_add1.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "Add(residual1)",
            ),
            (
                on_rms_post.as_ptr(),
                ot_rms.as_ptr(),
                in_rms_post.as_mut_ptr(),
                2,
                out_rms_post.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "RmsNorm(post)",
            ),
            (
                on_gate_proj.as_ptr(),
                ot_q40.as_ptr(),
                in_gate.as_mut_ptr(),
                3,
                out_gate.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "gate proj",
            ),
            (
                on_up_proj.as_ptr(),
                ot_q40.as_ptr(),
                in_up.as_mut_ptr(),
                3,
                out_up.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "up proj",
            ),
            (
                on_silu.as_ptr(),
                ot_silu.as_ptr(),
                in_silu.as_mut_ptr(),
                2,
                out_silu.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "SiluMul",
            ),
        ];
        let trailing: [NodeSpec<'_>; 2] = [
            (
                on_down_proj.as_ptr(),
                ot_q40.as_ptr(),
                in_down.as_mut_ptr(),
                3,
                out_down.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "down proj",
            ),
            (
                on_add2.as_ptr(),
                ot_add.as_ptr(),
                in_add2.as_mut_ptr(),
                2,
                out_add2.as_mut_ptr(),
                1,
                ptr::null_mut(),
                0,
                "Add(residual2)",
            ),
        ];
        let graph_add_node = v.graphAddNode.ok_or_else(|| anyhow!("graphAddNode NULL"))?;
        for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in nodes {
            let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
            let err = unsafe { graph_add_node(graph, op) };
            if err != 0 {
                return Err(anyhow!(
                    "graphAddNode(layer={layer_idx}, {label}) err=0x{:x}",
                    err
                ));
            }
        }
        for (nm, ty, ins, n_in, outs, n_out, params, n_params, label) in trailing {
            let op = make_op(nm, ty, ins, n_in, outs, n_out, params, n_params);
            let err = unsafe { graph_add_node(graph, op) };
            if err != 0 {
                return Err(anyhow!(
                    "graphAddNode(layer={layer_idx}, {label}) err=0x{:x}",
                    err
                ));
            }
        }

        // ── 6. graphFinalize (timed) ───────────────────────────────────────
        let t_fin = Instant::now();
        let graph_finalize = v
            .graphFinalize
            .ok_or_else(|| anyhow!("graphFinalize NULL"))?;
        let err = unsafe { graph_finalize(graph, ptr::null_mut(), ptr::null_mut()) };
        let finalize_ms_f64 = t_fin.elapsed().as_secs_f64() * 1000.0;
        let finalize_ms = finalize_ms_f64.min(u32::MAX as f64) as u32;
        eprintln!(
            "[qnn_oppkg] layer={layer_idx} graphFinalize err=0x{:x} elapsed={:.2} ms",
            err, finalize_ms_f64
        );
        if err != 0 {
            return Err(anyhow!("graphFinalize(layer={layer_idx}) err=0x{:x}", err));
        }

        // ── 7. Build exec_inputs / exec_outputs with memHandle baked ───────
        // slots order (M3.4 D-D.3):
        //   0: x  1: rms_pre  2: rms_post
        //   3-4: qq/qd  5-6: kq/kd  7-8: vq/vd  9-10: oq/od
        //   11-12: gq/gd  13-14: uq/ud  15-16: dq/dd
        //   17: kcache  18: vcache
        //   19: mask  20: sinks  21: score
        //   22: x_out
        //   23: pos_buf (M3.4 D-D.3 신규)
        let set_mh = |t: &mut ffi::Qnn_Tensor_t, h: ffi::Qnn_MemHandle_t| {
            t.__bindgen_anon_1.v1.memType = ffi::Qnn_TensorMemType_t_QNN_TENSORMEMTYPE_MEMHANDLE;
            t.__bindgen_anon_1.v1.__bindgen_anon_1.memHandle = h;
        };
        let mut t_x_mh = t_x;
        set_mh(&mut t_x_mh, slots[0].mem_handle);
        let mut t_rms_pre_mh = t_rms_pre;
        set_mh(&mut t_rms_pre_mh, slots[1].mem_handle);
        let mut t_rms_post_mh = t_rms_post;
        set_mh(&mut t_rms_post_mh, slots[2].mem_handle);
        let mut t_qq_mh = t_qq;
        set_mh(&mut t_qq_mh, slots[3].mem_handle);
        let mut t_qd_mh = t_qd;
        set_mh(&mut t_qd_mh, slots[4].mem_handle);
        let mut t_kq_mh = t_kq;
        set_mh(&mut t_kq_mh, slots[5].mem_handle);
        let mut t_kd_mh = t_kd;
        set_mh(&mut t_kd_mh, slots[6].mem_handle);
        let mut t_vq_mh = t_vq;
        set_mh(&mut t_vq_mh, slots[7].mem_handle);
        let mut t_vd_mh = t_vd;
        set_mh(&mut t_vd_mh, slots[8].mem_handle);
        let mut t_oq_mh = t_oq;
        set_mh(&mut t_oq_mh, slots[9].mem_handle);
        let mut t_od_mh = t_od;
        set_mh(&mut t_od_mh, slots[10].mem_handle);
        let mut t_gq_mh = t_gq;
        set_mh(&mut t_gq_mh, slots[11].mem_handle);
        let mut t_gd_mh = t_gd;
        set_mh(&mut t_gd_mh, slots[12].mem_handle);
        let mut t_uq_mh = t_uq;
        set_mh(&mut t_uq_mh, slots[13].mem_handle);
        let mut t_ud_mh = t_ud;
        set_mh(&mut t_ud_mh, slots[14].mem_handle);
        let mut t_dq_mh = t_dq;
        set_mh(&mut t_dq_mh, slots[15].mem_handle);
        let mut t_dd_mh = t_dd;
        set_mh(&mut t_dd_mh, slots[16].mem_handle);
        let mut t_kcache_mh = t_kcache;
        set_mh(&mut t_kcache_mh, slots[17].mem_handle);
        let mut t_vcache_mh = t_vcache;
        set_mh(&mut t_vcache_mh, slots[18].mem_handle);
        let mut t_mask_mh = t_mask;
        set_mh(&mut t_mask_mh, slots[19].mem_handle);
        let mut t_sinks_mh = t_sinks;
        set_mh(&mut t_sinks_mh, slots[20].mem_handle);
        let mut t_score_mh = t_score;
        set_mh(&mut t_score_mh, slots[21].mem_handle);
        let mut t_x_out_mh = t_x_out;
        set_mh(&mut t_x_out_mh, slots[22].mem_handle);
        let mut t_pos_mh = t_pos;
        set_mh(&mut t_pos_mh, slots[idx_pos].mem_handle);
        // D-D.6: n_kv_buf input tensor binding.
        let mut t_n_kv_mh = t_n_kv;
        set_mh(&mut t_n_kv_mh, slots[idx_n_kv].mem_handle);
        // D-D.6 Phase B: QKV bias input tensor binding.
        let mut t_q_bias_mh = t_q_bias;
        set_mh(&mut t_q_bias_mh, slots[idx_q_bias].mem_handle);
        let mut t_k_bias_mh = t_k_bias;
        set_mh(&mut t_k_bias_mh, slots[idx_k_bias].mem_handle);
        let mut t_v_bias_mh = t_v_bias;
        set_mh(&mut t_v_bias_mh, slots[idx_v_bias].mem_handle);

        let exec_inputs = vec![
            t_x_mh,
            t_rms_pre_mh,
            t_rms_post_mh,
            t_qq_mh,
            t_qd_mh,
            t_kq_mh,
            t_kd_mh,
            t_vq_mh,
            t_vd_mh,
            t_oq_mh,
            t_od_mh,
            t_gq_mh,
            t_gd_mh,
            t_uq_mh,
            t_ud_mh,
            t_dq_mh,
            t_dd_mh,
            t_mask_mh,
            t_sinks_mh,
            t_score_mh,
            t_pos_mh,    // M3.4 D-D.3
            t_n_kv_mh,   // M3.4 D-D.6
            t_q_bias_mh, // D-D.6 Phase B
            t_k_bias_mh,
            t_v_bias_mh,
        ];
        let exec_outputs = vec![t_kcache_mh, t_vcache_mh, t_x_out_mh];

        let weight_handle_count = 16; // 7 Q4_0 × 2 + 2 norm = 16
        Ok((
            AndroidLayerGraph {
                graph,
                slots,
                exec_inputs,
                exec_outputs,
                idx_x_in,
                idx_x_out,
                idx_kv_k,
                idx_kv_v,
                pos_host_ptr,
                n_kv_host_ptr,
                cfg: *cfg,
            },
            finalize_ms,
            weight_handle_count,
        ))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// INV-176: build-time const synchronization.
    #[test]
    fn layer_node_count_is_14() {
        assert_eq!(LAYER_NODE_COUNT, 14);
        assert_eq!(LAYER_INTERMEDIATE_COUNT, 13);
    }

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

    #[test]
    fn finalize_budget_matches_inv167() {
        // M3.4 디바이스 측정 후 1500 ms로 조정 (S25 Adreno 830 ~1.2 s/layer 실측).
        // INV-167은 process lifetime 동안 cache invalidation이 swap path에서만
        // 발동하는 것이 게이트 — finalize wall-clock은 D1 "eager prebuild ~33s"
        // 결정으로 흡수.
        assert_eq!(FINALIZE_BUDGET_MS, 1500);
    }
}
