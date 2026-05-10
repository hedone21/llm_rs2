//! `CustomFlashAttn` op descriptor — decode flash attention (n_q=1) with online softmax.
//!
//! Maps to `flash_attn_f32_f16_q1` in `engine/kernels/flash_attn_f32_f16.cl`.
//! Single Q token (decode), F32 Q + F16 K/V + F32 O. Production hot path —
//! 44 kernel args (0..43).
//!
//! ## Scope (M2.F)
//!
//! Limited to the **decode + Qwen2.5-1.5B base case**:
//!   - n_q = 1 (single query token)
//!   - F32 Q, F16 K/V (HeadMajor `[1, n_head_kv, capacity, head_dim]`), F32 O
//!   - is_causal = 0 (production uses 0; the kernel handles masking via
//!     `mask_void`, which is null for decode without ALiBi)
//!   - max_bias = 0.0 (ALiBi disabled)
//!   - logit_softcap = 0.0
//!   - mask_void = null (dummy 1-element FLOAT_16 buffer; kernel checks
//!     `mask_void != NULL` before use)
//!   - sinks_void = null (dummy 1-element FLOAT_32 buffer)
//!   - write_scores = 0 (dummy 1-element FLOAT_32 score buffer; kernel
//!     guards writes behind `if (write_scores)`)
//!
//! All inactive paths (mask, sinks, score-write) are bound to dummy 1-element
//! buffers. The kernel's null-checks short-circuit before any actual access.
//!
//! ## Tensor convention (graph view)
//!
//!   inputs[0]  = Q     F32 [1, n_head, head_dim]
//!   inputs[1]  = K     F16 [1, n_head_kv, capacity, head_dim] (HeadMajor)
//!   inputs[2]  = V     F16 [1, n_head_kv, capacity, head_dim] (HeadMajor)
//!   inputs[3]  = mask  F16 [1] (dummy)
//!   inputs[4]  = sinks F32 [1] (dummy)
//!   inputs[5]  = S     F32 [1] (dummy score buffer)
//!   outputs[0] = O     F32 [1, n_head, head_dim]
//!
//! ## Op params
//!
//! All required (no defaults):
//!   params[i].name = "n_kv"          INT_32 — current KV cache occupancy
//!   params[i].name = "n_head"        INT_32 — Q heads (e.g. 32 for Qwen2.5-1.5B)
//!   params[i].name = "n_head_kv"     INT_32 — KV heads (e.g. 2 for Qwen2.5-1.5B)
//!   params[i].name = "head_dim"      INT_32 — derived from Q rank-3 dim if absent
//!   params[i].name = "kv_capacity"   INT_32 — KV cache capacity (stride basis)
//!   params[i].name = "scale"         FLOAT_32 — defaults to 1/sqrt(head_dim)
//!
//! ## Args (44 total, kernel declaration order — see kernel signature in
//! `engine/kernels/flash_attn_f32_f16.cl` line 470)
//!
//!   0:  InputTensor(0)   q_void
//!   1:  ULong(0)         q_offset
//!   2:  InputTensor(1)   k_void
//!   3:  ULong(0)         k_offset
//!   4:  InputTensor(2)   v_void
//!   5:  ULong(0)         v_offset
//!   6:  OutputTensor(0)  o_void
//!   7:  ULong(0)         o_offset
//!   8:  Float(scale)
//!   9:  Int(n_q=1)
//!   10: Int(n_kv)
//!   11: Int(is_causal=0)
//!   12: Int(n_head)
//!   13: ULong(q_nb1)     — n_head * head_dim * 4 bytes
//!   14: ULong(q_nb2)     — head_dim * 4 bytes
//!   15: ULong(q_nb3)     — q_nb1 (single batch)
//!   16: ULong(k_nb1)     — head_dim * 2 bytes
//!   17: ULong(k_nb2)     — capacity * head_dim * 2 bytes
//!   18: ULong(k_nb3)     — n_head_kv * k_nb2
//!   19: ULong(v_nb1)     — same as k_nb1 (V mirrors K layout)
//!   20: ULong(v_nb2)     — same as k_nb2
//!   21: ULong(v_nb3)     — same as k_nb3
//!   22: ULong(o_nb1)     — head_dim * 4 bytes
//!   23: ULong(o_nb2)     — n_head * head_dim * 4 bytes
//!   24: ULong(o_nb3)     — same as o_nb2
//!   25: Float(0.0)       — max_bias (ALiBi off)
//!   26: Float(0.0)       — m0 (unused when max_bias=0)
//!   27: Float(0.0)       — m1
//!   28: Int(0)           — n_head_log2
//!   29: Float(0.0)       — logit_softcap
//!   30: Int(n_head_kv)
//!   31: InputTensor(3)   mask_void  (dummy)
//!   32: ULong(0)         mask_offset
//!   33: ULong(0)         mask_nb1
//!   34: ULong(0)         mask_nb2
//!   35: ULong(0)         mask_nb3
//!   36: Int(0)           mask_ne2
//!   37: Int(0)           mask_ne3
//!   38: InputTensor(4)   sinks_void (dummy)
//!   39: ULong(0)         sinks_offset
//!   40: InputTensor(5)   S          (dummy score buffer)
//!   41: Int(0)           score_layer_offset
//!   42: Int(0)           score_stride
//!   43: Int(0)           write_scores
//!
//! ## Workgroup
//!
//!   global_work = [Q1_WG_SIZE, n_head, 1] = [64, n_head, 1]
//!   local_work  = [64, 1, 1]
//!
//! Mirrors `build_flash_attention_step` in `engine/src/backend/opencl/plan.rs`.
//!
//! ## Build options
//!
//! head_dim=128 → `-DDK=128 -DDV=128 -DBLOCK_M=32 -DBLOCK_N=32`.
//! head_dim=64  → `-DDK=64 -DDV=64 -DBLOCK_M=64 -DBLOCK_N=32`.
//!
//! Build options are kernel-source-level macros. They cannot be parametrised
//! per-OpConfig at this layer (the descriptor's `build_options` is `&'static
//! str`). M2.F targets `head_dim=128` (Qwen2.5-1.5B). Other head_dims would
//! need a separate descriptor variant.

use crate::args::{ArgSpec, MemoryObjectSpec, OpDescriptor, OpError, OpImplLayout};
use crate::qnn::{
    Qnn_DataType_t_QNN_DATATYPE_FLOAT_16, Qnn_DataType_t_QNN_DATATYPE_FLOAT_32, Qnn_OpConfigV1_t,
    Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR,
};

pub static DESCRIPTOR: OpDescriptor = OpDescriptor {
    op_type: "CustomFlashAttn",
    kernel_name: "flash_attn_f32_f16_q1",
    kernel_source: include_str!("../../../../engine/kernels/flash_attn_f32_f16.cl"),
    // M2.F targets head_dim=128 (Qwen2.5-1.5B). DK/DV/BLOCK_M/BLOCK_N are
    // compile-time macros baked into the kernel program.
    build_options: "-cl-std=CL2.0 -cl-mad-enable -cl-fast-relaxed-math -DDK=128 -DDV=128 -DBLOCK_M=32 -DBLOCK_N=32",
    build_layout,
};

/// Read a scalar `INT_32` op param by name.
fn read_param_int(v1: &Qnn_OpConfigV1_t, name: &str) -> Option<i32> {
    if v1.params.is_null() || v1.numOfParams == 0 {
        return None;
    }
    for i in 0..v1.numOfParams {
        let p = unsafe { *v1.params.add(i as usize) };
        if p.paramType != Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR || p.name.is_null() {
            continue;
        }
        let pname = match unsafe { std::ffi::CStr::from_ptr(p.name).to_str() } {
            Ok(s) => s,
            Err(_) => continue,
        };
        if pname == name {
            return Some(unsafe { p.__bindgen_anon_1.scalarParam.__bindgen_anon_1.int32Value });
        }
    }
    None
}

/// Read a scalar `FLOAT_32` op param by name.
fn read_param_float(v1: &Qnn_OpConfigV1_t, name: &str) -> Option<f32> {
    if v1.params.is_null() || v1.numOfParams == 0 {
        return None;
    }
    for i in 0..v1.numOfParams {
        let p = unsafe { *v1.params.add(i as usize) };
        if p.paramType != Qnn_ParamType_t_QNN_PARAMTYPE_SCALAR || p.name.is_null() {
            continue;
        }
        let pname = match unsafe { std::ffi::CStr::from_ptr(p.name).to_str() } {
            Ok(s) => s,
            Err(_) => continue,
        };
        if pname == name {
            return Some(unsafe { p.__bindgen_anon_1.scalarParam.__bindgen_anon_1.floatValue });
        }
    }
    None
}

pub(crate) fn build_layout(v1: &Qnn_OpConfigV1_t) -> Result<OpImplLayout, OpError> {
    // Require: 6 inputs (Q, K, V, mask, sinks, S), 1 output (O).
    if v1.numOfInputs < 6 || v1.numOfOutputs < 1 {
        return Err(OpError::ValidationFailure);
    }

    // Required scalars from op params.
    let n_kv = read_param_int(v1, "n_kv").ok_or(OpError::ValidationFailure)?;
    let n_head = read_param_int(v1, "n_head").ok_or(OpError::ValidationFailure)?;
    let n_head_kv = read_param_int(v1, "n_head_kv").ok_or(OpError::ValidationFailure)?;
    let kv_capacity = read_param_int(v1, "kv_capacity").ok_or(OpError::ValidationFailure)?;
    let head_dim = read_param_int(v1, "head_dim").ok_or(OpError::ValidationFailure)?;

    // M2.H 5th attempt: param + tensor-dim diagnostic dump (gated on
    // QNN_OPPKG_DEBUG=1 to avoid noise in production). Verifies what the SDK
    // actually delivers vs what the graph code passed in.
    if std::env::var("QNN_OPPKG_DEBUG").as_deref() == Ok("1") {
        let scale_seen = read_param_float(v1, "scale").unwrap_or(f32::NAN);
        let dim_dump = |idx: u32| -> String {
            let t = unsafe { (*v1.inputTensors.add(idx as usize)).__bindgen_anon_1.v1 };
            if t.dimensions.is_null() || t.rank == 0 {
                return "<null>".to_string();
            }
            let mut s = String::from("[");
            for i in 0..t.rank {
                if i > 0 {
                    s.push(',');
                }
                let d = unsafe { *t.dimensions.add(i as usize) };
                s.push_str(&d.to_string());
            }
            s.push(']');
            s
        };
        let out_dump = |idx: u32| -> String {
            let t = unsafe { (*v1.outputTensors.add(idx as usize)).__bindgen_anon_1.v1 };
            if t.dimensions.is_null() || t.rank == 0 {
                return "<null>".to_string();
            }
            let mut s = String::from("[");
            for i in 0..t.rank {
                if i > 0 {
                    s.push(',');
                }
                let d = unsafe { *t.dimensions.add(i as usize) };
                s.push_str(&d.to_string());
            }
            s.push(']');
            s
        };
        eprintln!(
            "[oppkg-fa] params: n_kv={} n_head={} n_head_kv={} kv_capacity={} head_dim={} scale={}",
            n_kv, n_head, n_head_kv, kv_capacity, head_dim, scale_seen
        );
        eprintln!(
            "[oppkg-fa] in dims: q={} k={} v={} mask={} sinks={} S={} | out o={}",
            dim_dump(0),
            dim_dump(1),
            dim_dump(2),
            dim_dump(3),
            dim_dump(4),
            dim_dump(5),
            out_dump(0),
        );
    }

    if n_kv <= 0 || n_head <= 0 || n_head_kv <= 0 || kv_capacity <= 0 || head_dim <= 0 {
        return Err(OpError::InvalidArgument);
    }
    if n_head % n_head_kv != 0 {
        return Err(OpError::InvalidArgument);
    }
    if !(head_dim as u32).is_multiple_of(4) {
        return Err(OpError::InvalidArgument);
    }

    let n_head_u = n_head as u32;
    let n_head_kv_u = n_head_kv as u32;
    let head_dim_u = head_dim as u32;
    let capacity_u = kv_capacity as u32;

    // scale defaults to 1/sqrt(head_dim) when absent.
    let scale = read_param_float(v1, "scale").unwrap_or_else(|| 1.0f32 / (head_dim as f32).sqrt());

    // Q/O strides (F32 [1, 1, n_head, head_dim])
    let q_nb1 = (n_head_u * head_dim_u * 4) as u64;
    let q_nb2 = (head_dim_u * 4) as u64;
    let q_nb3 = q_nb1;
    let o_nb1 = (head_dim_u * 4) as u64;
    let o_nb2 = (n_head_u * head_dim_u * 4) as u64;
    let o_nb3 = o_nb2;

    // KV strides (F16 HeadMajor [1, n_head_kv, capacity, head_dim])
    let kv_elem_size: u64 = 2;
    let k_nb1 = (head_dim_u as u64) * kv_elem_size;
    let k_nb2 = (capacity_u * head_dim_u) as u64 * kv_elem_size;
    let k_nb3 = (n_head_kv_u as u64) * k_nb2;

    // Tensor sizes
    let q_total = n_head_u
        .checked_mul(head_dim_u)
        .ok_or(OpError::InvalidArgument)?;
    let kv_total = n_head_kv_u
        .checked_mul(capacity_u)
        .and_then(|x| x.checked_mul(head_dim_u))
        .ok_or(OpError::InvalidArgument)?;
    let o_total = q_total;

    // Mem objects: Q, K, V, mask (F16 [n_kv], zero-init), sinks (F32 [1]),
    // S (F32 [1]), O.
    //
    // Mask sizing rationale: the kernel branches on `if (mask_void != NULL)`.
    // OpPackage `InputTensor(3)` always binds a non-null mem object, so the
    // branch is taken. The kernel reads `mask_ptr[k_idx]` for k_idx ∈ [0, n_kv).
    // We size the mask buffer at `n_kv` halves to keep these reads in-bounds.
    // With `mask_nb*` strides all bound to 0 (see args), `mask_base = mask_void`
    // — so `mask_ptr[k_idx] = mask_void[k_idx]`. Zero-init mask values give
    // `score += slope * 0.0`, a no-op.
    //
    // `mask_ne2` and `mask_ne3` are bound to 1 (not 0) to avoid `head_idx %
    // mask_ne2` modulo-by-zero — index always evaluates to 0, so even with
    // strides=0 the address stays at `mask_void[k_idx]`.
    let mask_dim = n_kv as u32;
    let mem_objects = vec![
        // Q (F32)
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![q_total],
            flat_offsets: vec![0u32],
        },
        // K (F16)
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            flat_dims: vec![kv_total],
            flat_offsets: vec![0u32],
        },
        // V (F16)
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            flat_dims: vec![kv_total],
            flat_offsets: vec![0u32],
        },
        // mask (F16 [n_kv], zero-init by host) — see rationale above
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_16,
            flat_dims: vec![mask_dim],
            flat_offsets: vec![0u32],
        },
        // sinks (F32 [n_head]) — kernel indexes `sinks_ptr[head_idx]` for
        // every head when `sinks_void != NULL`. The OpPackage path always
        // binds a non-null mem object (no `mem_null()` equivalent), so the
        // buffer must be sized to n_head halves. Host must populate it with
        // a large negative value (e.g. -1e30) to neutralise the sink path:
        //   m_i      = sinks_ptr[head_idx]   = -1e30 ≈ -INFINITY
        //   l_final += exp(sinks - m_final)  = exp(-1e30) = 0
        // matching the raw-OpenCL `mem_null()` baseline.
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![n_head_u],
            flat_offsets: vec![0u32],
        },
        // S dummy (F32 [1])
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![1],
            flat_offsets: vec![0u32],
        },
        // O (F32) — claimed output
        MemoryObjectSpec {
            data_type: Qnn_DataType_t_QNN_DATATYPE_FLOAT_32,
            flat_dims: vec![o_total],
            flat_offsets: vec![0u32],
        },
    ];

    // 44 args, kernel declaration order. Mirrors `build_flash_attention_step`
    // in plan.rs (decode path: is_causal=0, mask null path, write_scores=0).
    let args = vec![
        // 0-1: Q
        ArgSpec::InputTensor(0),
        ArgSpec::ULong(0),
        // 2-3: K
        ArgSpec::InputTensor(1),
        ArgSpec::ULong(0),
        // 4-5: V
        ArgSpec::InputTensor(2),
        ArgSpec::ULong(0),
        // 6-7: O (output)
        ArgSpec::OutputTensor(0),
        ArgSpec::ULong(0),
        // 8: scale
        ArgSpec::Float(scale),
        // 9: n_q
        ArgSpec::Int(1),
        // 10: n_kv
        ArgSpec::Int(n_kv),
        // 11: is_causal (decode = 0; mask path inactive when mask_void=NULL)
        ArgSpec::Int(0),
        // 12: n_head
        ArgSpec::Int(n_head),
        // 13-15: q strides
        ArgSpec::ULong(q_nb1),
        ArgSpec::ULong(q_nb2),
        ArgSpec::ULong(q_nb3),
        // 16-18: k strides
        ArgSpec::ULong(k_nb1),
        ArgSpec::ULong(k_nb2),
        ArgSpec::ULong(k_nb3),
        // 19-21: v strides (same layout)
        ArgSpec::ULong(k_nb1),
        ArgSpec::ULong(k_nb2),
        ArgSpec::ULong(k_nb3),
        // 22-24: o strides
        ArgSpec::ULong(o_nb1),
        ArgSpec::ULong(o_nb2),
        ArgSpec::ULong(o_nb3),
        // 25-29: ALiBi/softcap (all disabled)
        ArgSpec::Float(0.0),
        ArgSpec::Float(0.0),
        ArgSpec::Float(0.0),
        ArgSpec::Int(0),
        ArgSpec::Float(0.0),
        // 30: n_head_kv
        ArgSpec::Int(n_head_kv),
        // 31-37: mask (real buffer of n_kv halves, zero-init).
        //
        // OpPackage cannot bind a null tensor for `mask_void`; every
        // `InputTensor(N)` resolves to a registered mem object. The kernel's
        // `if (mask_void != NULL)` therefore evaluates to true and the
        // mask-add branch executes. We make this branch a no-op:
        //
        //   - `mask_offset = 0` and all `mask_nb* = 0` ⇒ `mask_base =
        //     mask_void` for every (head, batch).
        //   - `mask_ne2 = mask_ne3 = 1` avoid the `head_idx % mask_ne2`
        //     modulo-by-zero UB. The result is `mask_head_idx = 0` which is
        //     fine since the strides above already collapse the address.
        //   - The mask buffer is zero-initialised by the host (size = n_kv
        //     halves), so `mask_ptr[k_idx]` is always 0.0f → `score += slope *
        //     0.0` is a no-op.
        ArgSpec::InputTensor(3),
        ArgSpec::ULong(0),
        ArgSpec::ULong(0),
        ArgSpec::ULong(0),
        ArgSpec::ULong(0),
        ArgSpec::Int(1),
        ArgSpec::Int(1),
        // 38-39: sinks
        ArgSpec::InputTensor(4),
        ArgSpec::ULong(0),
        // 40-43: scores
        ArgSpec::InputTensor(5),
        ArgSpec::Int(0),
        ArgSpec::Int(0),
        ArgSpec::Int(0),
    ];

    // global = [Q1_WG_SIZE, n_head, 1], local = [Q1_WG_SIZE, 1, 1]
    const Q1_WG_SIZE: usize = 64;

    // M2.H 5th: dump every ArgSpec to confirm what kernel sees (gated).
    if std::env::var("QNN_OPPKG_DEBUG").as_deref() == Ok("1") {
        eprintln!("[oppkg-fa] args ({}):", args.len());
        for (i, a) in args.iter().enumerate() {
            let s = match a {
                ArgSpec::InputTensor(t) => format!("InputTensor({})", t),
                ArgSpec::OutputTensor(t) => format!("OutputTensor({})", t),
                ArgSpec::InOutTensor(t) => format!("InOutTensor({})", t),
                ArgSpec::OutputTensorAliased(t) => format!("OutputTensorAliased({})", t),
                ArgSpec::LocalMem(b) => format!("LocalMem({})", b),
                ArgSpec::Int(v) => format!("Int({})", v),
                ArgSpec::UInt(v) => format!("UInt({})", v),
                ArgSpec::ULong(v) => format!("ULong({})", v),
                ArgSpec::Float(v) => format!("Float({})", v),
            };
            eprintln!("  [{:>2}] {}", i, s);
        }
        eprintln!(
            "[oppkg-fa] global={:?} local={:?}",
            [Q1_WG_SIZE, n_head as usize, 1],
            [Q1_WG_SIZE, 1usize, 1]
        );
    }

    Ok(OpImplLayout {
        mem_objects,
        args,
        output_claims: None,
        global_work_dim: 2,
        local_work_dim: 3,
        global_work: [Q1_WG_SIZE, n_head as usize, 1],
        local_work: [Q1_WG_SIZE, 1, 1],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_layout(
        n_head: i32,
        n_head_kv: i32,
        head_dim: i32,
        kv_capacity: i32,
        n_kv: i32,
    ) -> Result<OpImplLayout, OpError> {
        crate::__test_support::build_flash_attn_layout_for(
            n_head,
            n_head_kv,
            head_dim,
            kv_capacity,
            n_kv,
        )
    }

    #[test]
    fn flash_attn_descriptor_op_type() {
        assert_eq!(DESCRIPTOR.op_type, "CustomFlashAttn");
    }

    #[test]
    fn flash_attn_kernel_source_contains_target() {
        assert!(
            DESCRIPTOR.kernel_source.contains("flash_attn_f32_f16_q1"),
            "kernel source must contain 'flash_attn_f32_f16_q1'"
        );
    }

    #[test]
    fn flash_attn_build_layout_for_qwen() {
        // Qwen2.5-1.5B decode: n_head=32 (q), n_head_kv=2 (kv), head_dim=128,
        // capacity=2048, n_kv=1024.
        let layout = make_layout(32, 2, 128, 2048, 1024).expect("build_layout must succeed");
        assert_eq!(layout.args.len(), 44, "must have 44 kernel args");
        assert_eq!(layout.mem_objects.len(), 7, "must have 7 mem_objects");
        // global_work = [64, 32, 1], local = [64, 1, 1]
        assert_eq!(layout.global_work, [64, 32, 1]);
        assert_eq!(layout.local_work, [64, 1, 1]);
    }
}
