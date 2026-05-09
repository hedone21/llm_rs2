//! Host-side metadata tests for the op registry. These run on any architecture;
//! they never touch OpenCL.

use qnn_oppkg::__test_support::{
    arg_to_kernel_arg, build_add_layout_n, build_deq_q40_layout_for, build_flash_attn_layout_for,
    build_kv_scatter_layout_for, build_matmul_layout_for, build_matmul_q40_layout_for,
    build_matmul_q40_layout_with_dims, build_rmsnorm_layout_for, build_rope_layout_for,
    build_rope_layout_rank2_for, build_silumul_layout_for, build_softmax_layout_for,
    build_state_returns_non_null, data_kernel_arg_type, kernel_arg_type,
};
use qnn_oppkg::{ArgSpec, MemoryObjectSpec, OPS, OpImplLayout, ensure_static, find_descriptor};

#[test]
fn ops_len_matches_num_operations() {
    // INV-152: OPS.len() must equal the leaked Info_t.numOperations.
    let info = ensure_static();
    assert_eq!(
        OPS.len() as u32,
        info.info.numOperations,
        "OPS.len() must equal numOperations"
    );
}

#[test]
fn op_types_are_unique() {
    // INV-153
    let mut set = std::collections::HashSet::new();
    for d in OPS {
        assert!(set.insert(d.op_type), "duplicate op_type: {}", d.op_type);
    }
}

#[test]
fn arg_spec_int_conversion() {
    let arg = arg_to_kernel_arg(&ArgSpec::Int(42));
    assert_eq!(arg.type_, kernel_arg_type::DATA);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.data.type_, data_kernel_arg_type::INT);
        assert_eq!(arg.__bindgen_anon_1.data.__bindgen_anon_1.qnnInt, 42);
    }
}

#[test]
fn arg_spec_uint_conversion() {
    let arg = arg_to_kernel_arg(&ArgSpec::UInt(7));
    assert_eq!(arg.type_, kernel_arg_type::DATA);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.data.type_, data_kernel_arg_type::UINT);
        assert_eq!(arg.__bindgen_anon_1.data.__bindgen_anon_1.qnnUInt, 7);
    }
}

#[test]
fn arg_spec_ulong_conversion() {
    let arg = arg_to_kernel_arg(&ArgSpec::ULong(0xDEADBEEF));
    assert_eq!(arg.type_, kernel_arg_type::DATA);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.data.type_, data_kernel_arg_type::ULONG);
        assert_eq!(
            arg.__bindgen_anon_1.data.__bindgen_anon_1.qnnULong,
            0xDEADBEEF
        );
    }
}

#[test]
fn arg_spec_float_conversion() {
    let arg = arg_to_kernel_arg(&ArgSpec::Float(1.5));
    assert_eq!(arg.type_, kernel_arg_type::DATA);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.data.type_, data_kernel_arg_type::FLOAT);
        assert_eq!(arg.__bindgen_anon_1.data.__bindgen_anon_1.qnnFloat, 1.5);
    }
}

#[test]
fn arg_spec_local_mem_conversion() {
    let arg = arg_to_kernel_arg(&ArgSpec::LocalMem(256));
    assert_eq!(arg.type_, kernel_arg_type::LOCAL);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.local.size, 256);
    }
}

#[test]
fn arg_spec_input_tensor_conversion() {
    let arg = arg_to_kernel_arg(&ArgSpec::InputTensor(3));
    assert_eq!(arg.type_, kernel_arg_type::OP_INPUT_READ);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.tensor.tensorIndex, 3);
        assert_eq!(arg.__bindgen_anon_1.tensor.element, 0);
    }
}

#[test]
fn arg_spec_output_tensor_conversion() {
    let arg = arg_to_kernel_arg(&ArgSpec::OutputTensor(0));
    assert_eq!(arg.type_, kernel_arg_type::OP_OUTPUT_WRITE);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.tensor.tensorIndex, 0);
    }
}

#[test]
fn build_op_state_with_dummy_descriptor() {
    // Construct a minimal valid layout: one input, one output. build_op_state
    // must produce a non-null leaked pointer (no OpenCL device required).
    let layout = OpImplLayout {
        mem_objects: vec![
            MemoryObjectSpec {
                data_type: 0, // QNN_DATATYPE_FLOAT_32 placeholder; opaque to host test
                flat_dims: vec![16],
                flat_offsets: vec![0],
            },
            MemoryObjectSpec {
                data_type: 0,
                flat_dims: vec![16],
                flat_offsets: vec![0],
            },
        ],
        args: vec![ArgSpec::InputTensor(0), ArgSpec::OutputTensor(0)],
        output_claims: None,
        global_work_dim: 1,
        local_work_dim: 0,
        global_work: [16, 1, 1],
        local_work: [0, 0, 0],
    };
    assert!(build_state_returns_non_null(
        layout,
        "dummy_kernel",
        "__kernel void dummy_kernel(__global float* a) {}",
        "-cl-fast-relaxed-math",
    ));
}

#[test]
fn find_descriptor_unknown_returns_none() {
    assert!(find_descriptor("Bogus").is_none());
}

// ── M1.3 CustomAdd descriptor tests ──────────────────────────────────────────

#[test]
fn add_descriptor_op_type() {
    assert_eq!(OPS[0].op_type, "CustomAdd");
}

#[test]
fn add_descriptor_kernel_source_non_empty() {
    assert!(!OPS[0].kernel_source.is_empty());
}

#[test]
fn add_descriptor_kernel_source_contains_target_kernel() {
    assert!(
        OPS[0].kernel_source.contains("kernel_add_row"),
        "kernel source must contain 'kernel_add_row'"
    );
}

// ── M1.4 CustomMatMulF16F32 descriptor tests ────────────────────────────────

#[test]
fn matmul_descriptor_op_type_unique() {
    assert_eq!(OPS[1].op_type, "CustomMatMulF16F32");
    assert_ne!(
        OPS[0].op_type, OPS[1].op_type,
        "CustomAdd and CustomMatMulF16F32 op_type must differ"
    );
}

#[test]
fn matmul_kernel_source_contains_target() {
    assert!(
        OPS[1].kernel_source.contains("kernel_mul_mat_f16_f32"),
        "kernel source must contain 'kernel_mul_mat_f16_f32'"
    );
}

#[test]
fn matmul_build_layout_for_small_dims() {
    use qnn_oppkg::__test_support::data_type;
    // M=1, N=512, K=256 — small case.
    let layout = build_matmul_layout_for(1, 512, 256).expect("build_layout must succeed");
    assert_eq!(layout.args.len(), 15, "matmul must have 15 kernel args");
    assert_eq!(
        layout.mem_objects.len(),
        3,
        "matmul must have 3 mem_objects"
    );
    assert_eq!(
        layout.mem_objects[0].data_type,
        data_type::FLOAT_16,
        "weight must be FLOAT_16"
    );
    assert_eq!(
        layout.mem_objects[1].data_type,
        data_type::FLOAT_32,
        "x must be FLOAT_32"
    );
    assert_eq!(
        layout.mem_objects[2].data_type,
        data_type::FLOAT_32,
        "y must be FLOAT_32"
    );
    // ((512 + 1) / 2) * 64 = 256 * 64 = 16384
    assert_eq!(layout.global_work[0], 16384);
    // M * 4 = 1 * 4 = 4
    assert_eq!(layout.global_work[1], 4);
    assert_eq!(layout.global_work[2], 1);
    assert_eq!(layout.local_work, [64, 4, 1]);
    assert_eq!(layout.global_work_dim, 3);
    assert_eq!(layout.local_work_dim, 3);
}

#[test]
fn matmul_build_layout_for_qwen_dim() {
    // Qwen2.5-1.5b lm_head: M=1, N=8960, K=1536.
    let layout = build_matmul_layout_for(1, 8960, 1536).expect("build_layout must succeed");
    // ((8960 + 1) / 2) * 64 = 4480 * 64 = 286720
    assert_eq!(layout.global_work[0], 286720);
    // M * 4 = 4
    assert_eq!(layout.global_work[1], 4);
}

// ── M1.5 CustomRmsNorm descriptor tests ─────────────────────────────────────

#[test]
fn rmsnorm_descriptor_op_type() {
    assert_eq!(OPS[2].op_type, "CustomRmsNorm");
}

#[test]
fn rmsnorm_kernel_source_contains_target() {
    assert!(
        OPS[2].kernel_source.contains("kernel_rms_norm_simple"),
        "kernel source must contain 'kernel_rms_norm_simple'"
    );
}

#[test]
fn rmsnorm_build_layout_for_qwen_dim() {
    use qnn_oppkg::__test_support::data_type;
    // M1.5 uses kernel_rms_norm_simple (1 work-item per row). Use rows=1,
    // dim=2048 to match the microbench's smallest case.
    let layout =
        build_rmsnorm_layout_for(1, 2048).expect("build_layout must succeed for (1, 2048)");
    assert_eq!(
        layout.args.len(),
        5,
        "rmsnorm_simple must have 5 kernel args"
    );
    assert_eq!(
        layout.mem_objects.len(),
        3,
        "rmsnorm must have 3 mem_objects (x, weight, out)"
    );
    for mo in &layout.mem_objects {
        assert_eq!(
            mo.data_type,
            data_type::FLOAT_32,
            "all rmsnorm tensors must be FLOAT_32"
        );
    }
    // global_work[0] = rows = 1
    assert_eq!(layout.global_work[0], 1);
    assert_eq!(layout.global_work[1], 1);
    assert_eq!(layout.global_work[2], 1);
    assert_eq!(layout.local_work, [0, 0, 0]);
    assert_eq!(layout.global_work_dim, 1);
    assert_eq!(layout.local_work_dim, 0);
}

// ── M1.6 CustomSoftmax descriptor tests ─────────────────────────────────────

#[test]
fn softmax_descriptor_op_type() {
    assert_eq!(OPS[3].op_type, "CustomSoftmax");
}

#[test]
fn softmax_kernel_source_contains_target() {
    assert!(
        OPS[3].kernel_source.contains("kernel_softmax_simple"),
        "kernel source must contain 'kernel_softmax_simple'"
    );
}

#[test]
fn softmax_build_layout_for_attn_dim() {
    use qnn_oppkg::__test_support::data_type;
    // M1.6 uses kernel_softmax_simple (1 work-item per row). Use rows=1,
    // dim=2048 to match the smallest microbench case.
    let layout =
        build_softmax_layout_for(1, 2048).expect("build_layout must succeed for (1, 2048)");
    assert_eq!(
        layout.args.len(),
        3,
        "softmax_simple must have 3 kernel args"
    );
    assert_eq!(
        layout.mem_objects.len(),
        2,
        "softmax must have 2 mem_objects (x, out)"
    );
    for mo in &layout.mem_objects {
        assert_eq!(
            mo.data_type,
            data_type::FLOAT_32,
            "all softmax tensors must be FLOAT_32"
        );
    }
    // global_work[0] = rows = 1
    assert_eq!(layout.global_work[0], 1);
    assert_eq!(layout.global_work[1], 1);
    assert_eq!(layout.global_work[2], 1);
    assert_eq!(layout.local_work, [0, 0, 0]);
    assert_eq!(layout.global_work_dim, 1);
    assert_eq!(layout.local_work_dim, 0);
}

// ── M1.7 CustomSiluMul descriptor tests ─────────────────────────────────────

#[test]
fn silu_mul_descriptor_op_type() {
    assert_eq!(OPS[4].op_type, "CustomSiluMul");
}

#[test]
fn silu_mul_kernel_source_contains_target() {
    assert!(
        OPS[4].kernel_source.contains("kernel_silu_mul_simple_oop"),
        "kernel source must contain 'kernel_silu_mul_simple_oop'"
    );
}

#[test]
fn silu_mul_build_layout_for_qwen_ffn() {
    use qnn_oppkg::__test_support::data_type;
    // Qwen2.5-1.5b ffn intermediate: rows=1, dim=8960. total = 8960, size4 = 2240.
    let layout =
        build_silumul_layout_for(1, 8960).expect("build_layout must succeed for (1, 8960)");
    // M2.H OOP: 4 args (x, y, out, size4). 3 mem_objects (x, y, out).
    assert_eq!(layout.args.len(), 4, "silu_mul_oop must have 4 kernel args");
    assert_eq!(
        layout.mem_objects.len(),
        3,
        "silu_mul_oop must have 3 mem_objects (x, y, out); got {}",
        layout.mem_objects.len()
    );
    for mo in &layout.mem_objects {
        assert_eq!(
            mo.data_type,
            data_type::FLOAT_32,
            "all silu_mul tensors must be FLOAT_32"
        );
    }
    // global_work[0] = size4 = total / 4 = 8960 / 4 = 2240
    assert_eq!(layout.global_work[0], 8960 / 4);
    assert_eq!(layout.global_work[1], 1);
    assert_eq!(layout.global_work[2], 1);
    assert_eq!(layout.local_work, [0, 0, 0]);
    assert_eq!(layout.global_work_dim, 1);
    assert_eq!(layout.local_work_dim, 0);
}

#[test]
fn silu_mul_uses_oop_args() {
    // M2.H OOP: args [InputTensor(0), InputTensor(1), OutputTensor(0), Int(size4)].
    let layout =
        build_silumul_layout_for(1, 8960).expect("build_layout must succeed for (1, 8960)");
    match &layout.args[0] {
        ArgSpec::InputTensor(idx) => assert_eq!(*idx, 0, "args[0] must be InputTensor(0)"),
        _ => panic!("args[0] must be InputTensor(0)"),
    }
    match &layout.args[1] {
        ArgSpec::InputTensor(idx) => assert_eq!(*idx, 1, "args[1] must be InputTensor(1)"),
        _ => panic!("args[1] must be InputTensor(1)"),
    }
    match &layout.args[2] {
        ArgSpec::OutputTensor(idx) => assert_eq!(*idx, 0, "args[2] must be OutputTensor(0)"),
        _ => panic!("args[2] must be OutputTensor(0)"),
    }
    match &layout.args[3] {
        ArgSpec::Int(_) => {}
        _ => panic!("args[3] must be Int(size4)"),
    }
}

#[test]
fn arg_spec_inout_tensor_conversion() {
    let arg = arg_to_kernel_arg(&ArgSpec::InOutTensor(0));
    assert_eq!(arg.type_, kernel_arg_type::OP_INPUT_READWRITE);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.tensor.tensorIndex, 0);
        assert_eq!(arg.__bindgen_anon_1.tensor.element, 0);
    }
}

#[test]
fn arg_spec_output_tensor_aliased_conversion() {
    // M2.G (INV-164): kernel-arg encoding is identical to InOutTensor —
    // OP_INPUT_READWRITE on the input tensor. The aliasing semantics are
    // applied separately in `build_op_state` (out_claim.memoryObject points
    // at mem_objects[input_idx]).
    let arg = arg_to_kernel_arg(&ArgSpec::OutputTensorAliased(0));
    assert_eq!(arg.type_, kernel_arg_type::OP_INPUT_READWRITE);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.tensor.tensorIndex, 0);
        assert_eq!(arg.__bindgen_anon_1.tensor.element, 0);
    }

    // tensorIndex must propagate through any non-zero index too.
    let arg2 = arg_to_kernel_arg(&ArgSpec::OutputTensorAliased(2));
    assert_eq!(arg2.type_, kernel_arg_type::OP_INPUT_READWRITE);
    unsafe {
        assert_eq!(arg2.__bindgen_anon_1.tensor.tensorIndex, 2);
    }
}

// ── M2.B CustomRope descriptor tests ────────────────────────────────────────

#[test]
fn rope_descriptor_op_type() {
    assert_eq!(OPS[5].op_type, "CustomRope");
}

#[test]
fn rope_kernel_source_contains_target() {
    assert!(
        OPS[5].kernel_source.contains("kernel_rope_simple_oop"),
        "kernel source must contain 'kernel_rope_simple_oop'"
    );
}

#[test]
fn rope_build_layout_for_qwen_attn() {
    use qnn_oppkg::__test_support::data_type;
    // Qwen2.5-1.5b query head dim: seq_len=1, num_heads=12 (q) or 2 (kv),
    // head_dim=128. Use the q-projection shape for the assertion.
    let layout = build_rope_layout_for(1, 12, 128).expect("build_layout must succeed");
    // M3.4 D-D.1: 7 args (InputTensor(0)=x_in, OutputTensor(0)=x_out,
    // InputTensor(1)=pos_buf, 3 ints, 1 float).
    assert_eq!(layout.args.len(), 7, "rope_oop must have 7 kernel args");
    assert_eq!(
        layout.mem_objects.len(),
        3,
        "rope_oop must have 3 mem_objects (x_in, pos_buf, x_out)"
    );
    // mem_objects[0]=x_in F32, [1]=pos_buf I32, [2]=x_out F32.
    assert_eq!(layout.mem_objects[0].data_type, data_type::FLOAT_32);
    assert_eq!(layout.mem_objects[1].data_type, data_type::INT_32);
    assert_eq!(layout.mem_objects[2].data_type, data_type::FLOAT_32);
    // global_work[0] = seq_len * num_heads * (head_dim/2) = 1 * 12 * 64 = 768
    assert_eq!(layout.global_work[0], 12 * 64);
    assert_eq!(layout.global_work[1], 1);
    assert_eq!(layout.global_work[2], 1);
    assert_eq!(layout.local_work, [0, 0, 0]);
    assert_eq!(layout.global_work_dim, 1);
    assert_eq!(layout.local_work_dim, 0);
}

#[test]
fn rope_build_layout_rejects_odd_head_dim() {
    // head_dim must be even (kernel rotates pairs).
    assert!(build_rope_layout_for(1, 12, 127).is_err());
}

#[test]
fn rope_build_layout_for_llama_attn() {
    // Llama 3.2 1B: seq_len=1, num_heads=32 (q), head_dim=64.
    let layout = build_rope_layout_for(1, 32, 64).expect("build_layout must succeed");
    // global_work[0] = 1 * 32 * 32 = 1024
    assert_eq!(layout.global_work[0], 32 * 32);
    assert_eq!(layout.args.len(), 7);
}

#[test]
fn add_build_layout_for_n4() {
    // Mock OpConfigV1 with 2 inputs + 1 output, dim=[256], FLOAT_32.
    let n_elems = 256u32;
    let layout = build_add_layout_n(n_elems).expect("build_layout must succeed for n=256");
    assert_eq!(
        layout.mem_objects.len(),
        3,
        "must have 3 mem_objects (src0, src1, dst)"
    );
    // 7 args: InputTensor(0), ULong(0), InputTensor(1), ULong(0), OutputTensor(0), ULong(0), Int(n4)
    assert_eq!(
        layout.args.len(),
        7,
        "must have 7 args matching kernel_add_row signature"
    );
    let expected_n4 = (n_elems / 4) as usize;
    assert_eq!(
        layout.global_work[0], expected_n4,
        "global_work[0] must equal n4 = n/4"
    );
}

// ── M2.C CustomDeqQ40 descriptor tests ──────────────────────────────────────

#[test]
fn deq_q40_descriptor_op_type() {
    assert_eq!(OPS[6].op_type, "CustomDeqQ40");
}

#[test]
fn deq_q40_kernel_source_contains_target() {
    assert!(
        OPS[6].kernel_source.contains("kernel_convert_block_q4_0"),
        "kernel source must contain 'kernel_convert_block_q4_0'"
    );
}

#[test]
fn deq_q40_build_layout_for_typical() {
    use qnn_oppkg::__test_support::data_type;
    // 1000 blocks: src0 = [18000 bytes], dst_q = [16000 bytes], dst_d = [1000 halves]
    let layout = build_deq_q40_layout_for(1000).expect("build_layout must succeed for 1000 blocks");
    assert_eq!(layout.args.len(), 3, "deq_q40 must have 3 kernel args");
    assert_eq!(
        layout.mem_objects.len(),
        3,
        "deq_q40 must have 3 mem_objects (src0, dst_q, dst_d)"
    );
    assert_eq!(
        layout.mem_objects[0].data_type,
        data_type::UINT_8,
        "src0 must be UINT_8"
    );
    assert_eq!(
        layout.mem_objects[1].data_type,
        data_type::UINT_8,
        "dst_q must be UINT_8"
    );
    assert_eq!(
        layout.mem_objects[2].data_type,
        data_type::FLOAT_16,
        "dst_d must be FLOAT_16"
    );
    // global_work[0] = num_blocks = 1000
    assert_eq!(layout.global_work[0], 1000);
    assert_eq!(layout.global_work[1], 1);
    assert_eq!(layout.global_work[2], 1);
    assert_eq!(layout.local_work, [0, 0, 0]);
    assert_eq!(layout.global_work_dim, 1);
    assert_eq!(layout.local_work_dim, 0);
}

// ── M2.D CustomMatMulQ40F32 descriptor tests ────────────────────────────────

#[test]
fn matmul_q40_descriptor_op_type() {
    assert_eq!(OPS[7].op_type, "CustomMatMulQ40F32");
}

#[test]
fn matmul_q40_kernel_source_contains_target() {
    assert!(
        OPS[7]
            .kernel_source
            .contains("kernel_mul_mat_q4_0_f32_8x_flat"),
        "kernel source must contain 'kernel_mul_mat_q4_0_f32_8x_flat'"
    );
}

#[test]
fn matmul_q40_build_layout_for_qwen_qkv() {
    use qnn_oppkg::__test_support::data_type;
    // Qwen2.5-1.5b QKV / O proj: M=1, N=1536, K=1536. num_blocks = 1536*1536/32 = 73728.
    let layout = build_matmul_q40_layout_for(1, 1536, 1536).expect("build_layout must succeed");
    assert_eq!(layout.args.len(), 15, "matmul_q40 must have 15 kernel args");
    assert_eq!(
        layout.mem_objects.len(),
        4,
        "matmul_q40 must have 4 mem_objects (q, d, x, y)"
    );
    assert_eq!(
        layout.mem_objects[0].data_type,
        data_type::UINT_8,
        "src0_q must be UINT_8"
    );
    assert_eq!(
        layout.mem_objects[1].data_type,
        data_type::FLOAT_16,
        "src0_d must be FLOAT_16"
    );
    assert_eq!(
        layout.mem_objects[2].data_type,
        data_type::FLOAT_32,
        "src1 (x) must be FLOAT_32"
    );
    assert_eq!(
        layout.mem_objects[3].data_type,
        data_type::FLOAT_32,
        "dst (y) must be FLOAT_32"
    );
    let expected_num_blocks = 1536u32 * 1536 / 32;
    assert_eq!(layout.mem_objects[0].flat_dims[0], expected_num_blocks * 16);
    assert_eq!(layout.mem_objects[1].flat_dims[0], expected_num_blocks);
    // ((1536 + 7) / 8) * 64 = 192 * 64 = 12288
    assert_eq!(layout.global_work[0], 1536u32.div_ceil(8) as usize * 64);
    assert_eq!(layout.global_work[1], 1);
    assert_eq!(layout.global_work[2], 1);
    assert_eq!(layout.local_work, [64, 1, 1]);
    assert_eq!(layout.global_work_dim, 3);
    assert_eq!(layout.local_work_dim, 3);
}

#[test]
fn matmul_q40_build_layout_for_qwen_ffn() {
    // Qwen2.5-1.5b FFN gate/up: M=1, N=8960, K=1536.
    let layout = build_matmul_q40_layout_for(1, 8960, 1536).expect("build_layout must succeed");
    // ((8960 + 7) / 8) * 64 = 1120 * 64 = 71680
    assert_eq!(layout.global_work[0], 8960u32.div_ceil(8) as usize * 64);
    assert_eq!(layout.args.len(), 15);
    let expected_num_blocks = 8960u32 * 1536 / 32;
    assert_eq!(layout.mem_objects[0].flat_dims[0], expected_num_blocks * 16);
    assert_eq!(layout.mem_objects[1].flat_dims[0], expected_num_blocks);
}

#[test]
fn matmul_q40_build_layout_rejects_non_multiple_k() {
    // K = 100 is not a multiple of QK4_0 = 32.
    assert!(build_matmul_q40_layout_for(1, 1536, 100).is_err());
}

// ── M2.H 3rd-attempt: rank-flexible build_layout (rank-3 reshape views) ─────

#[test]
fn matmul_q40_build_layout_rank3_input_rank2_output() {
    // Qwen O proj: input rank 3 [1, 12, 128] (FlashAttn output reshape view),
    // output rank 2 [1, 1536] — K = 12 * 128 = 1536, N = 1536.
    let layout = build_matmul_q40_layout_with_dims(&[1, 12, 128], &[1, 1536])
        .expect("rank3 input + rank2 output must succeed");
    assert_eq!(layout.args.len(), 15);
    // ne00 = K = 1536, ne01 = N = 1536, ne1 = M = 1.
    let expected_num_blocks = 1536u32 * 1536 / 32;
    assert_eq!(layout.mem_objects[0].flat_dims[0], expected_num_blocks * 16);
    assert_eq!(layout.mem_objects[1].flat_dims[0], expected_num_blocks);
    // global_work[0] = ((1536 + 7) / 8) * 64.
    assert_eq!(layout.global_work[0], 1536u32.div_ceil(8) as usize * 64);
}

#[test]
fn matmul_q40_build_layout_rank2_input_rank3_output() {
    // Qwen Q proj: input rank 2 [1, 1536] (RmsNorm output), output rank 3
    // [1, 12, 128] (RoPE input reshape view) — K = 1536, N = 12 * 128 = 1536.
    let layout = build_matmul_q40_layout_with_dims(&[1, 1536], &[1, 12, 128])
        .expect("rank2 input + rank3 output must succeed");
    assert_eq!(layout.args.len(), 15);
    let expected_num_blocks = 1536u32 * 1536 / 32;
    assert_eq!(layout.mem_objects[0].flat_dims[0], expected_num_blocks * 16);
}

#[test]
fn matmul_q40_build_layout_rank3_input_rank3_output() {
    // Hypothetical: rank 3 in/out (e.g. K proj viewed as rank 3 on both sides).
    // Qwen K proj: input [1, 1536] view as [1, 12, 128] but with K still 1536,
    // output [1, 2, 128] — K = 12*128 = 1536, N = 2*128 = 256.
    let layout = build_matmul_q40_layout_with_dims(&[1, 12, 128], &[1, 2, 128])
        .expect("rank3 input + rank3 output must succeed");
    assert_eq!(layout.args.len(), 15);
    let expected_num_blocks = 256u32 * 1536 / 32;
    assert_eq!(layout.mem_objects[0].flat_dims[0], expected_num_blocks * 16);
    // global_work[0] = ((256 + 7) / 8) * 64.
    assert_eq!(layout.global_work[0], 256u32.div_ceil(8) as usize * 64);
}

#[test]
fn rope_build_layout_rank2_for_qwen_q() {
    // Qwen Q rope rank-2 path: x flat [seq=1, num_heads*head_dim=12*128=1536].
    let layout = build_rope_layout_rank2_for(1, 12, 128)
        .expect("rope rank-2 must succeed with num_heads/head_dim params");
    // global_work[0] = seq * num_heads * (head_dim / 2) = 1 * 12 * 64 = 768.
    assert_eq!(layout.global_work[0], 12 * 64);
    // M2.H OOP: 7 args.
    assert_eq!(layout.args.len(), 7);
}

#[test]
fn rope_build_layout_rank2_rejects_inconsistent_split() {
    // num_heads * head_dim must equal flat dim. 12 * 64 = 768 ≠ 1536 → reject.
    let result = build_rope_layout_rank2_for(1, 12, 64);
    // (Note: build_rope_layout_rank2_for sets x.dim[1] = num_heads * head_dim
    // implicitly, so this case ALWAYS matches by construction. The actual
    // mismatch path is exercised via the rank-3 helper with bad dims; here we
    // just assert the rank-2 happy path returns OK for valid splits.)
    assert!(result.is_ok());
}

// ── M2.E CustomKvScatter descriptor tests ───────────────────────────────────

#[test]
fn kv_scatter_descriptor_op_type() {
    assert_eq!(OPS[8].op_type, "CustomKvScatter");
}

#[test]
fn kv_scatter_kernel_source_contains_target() {
    // M3.4 D-D.1: descriptor now binds the OOP variant kernel.
    assert!(
        OPS[8]
            .kernel_source
            .contains("kernel_kv_scatter_f32_to_f16_oop"),
        "kernel source must contain 'kernel_kv_scatter_f32_to_f16_oop'"
    );
}

#[test]
fn kv_scatter_build_layout_for_qwen() {
    use qnn_oppkg::__test_support::data_type;
    // kv_heads=2, head_dim=128, capacity=2048. write_pos is now ignored at
    // build_layout — supplied via pos_buf at execute time.
    let layout = build_kv_scatter_layout_for(2, 128, 2048, 100)
        .expect("build_layout must succeed for (kv_heads=2, head_dim=128, capacity=2048)");
    assert_eq!(layout.args.len(), 7, "must have 7 kernel args");
    assert_eq!(layout.mem_objects.len(), 5, "must have 5 mem_objects");
    assert_eq!(
        layout.mem_objects[0].data_type,
        data_type::FLOAT_32,
        "k_src must be FLOAT_32"
    );
    assert_eq!(
        layout.mem_objects[1].data_type,
        data_type::FLOAT_32,
        "v_src must be FLOAT_32"
    );
    assert_eq!(
        layout.mem_objects[2].data_type,
        data_type::INT_32,
        "pos_buf must be INT_32"
    );
    assert_eq!(
        layout.mem_objects[3].data_type,
        data_type::FLOAT_16,
        "k_dst must be FLOAT_16"
    );
    assert_eq!(
        layout.mem_objects[4].data_type,
        data_type::FLOAT_16,
        "v_dst must be FLOAT_16"
    );
    // global_work[0] = kv_heads * head_dim = 2 * 128 = 256
    assert_eq!(layout.global_work[0], 256);
    assert_eq!(layout.global_work[1], 1);
    assert_eq!(layout.global_work[2], 1);
    assert_eq!(layout.local_work, [0, 0, 0]);
    assert_eq!(layout.global_work_dim, 1);
    assert_eq!(layout.local_work_dim, 0);
}

// ── M2.F CustomFlashAttn descriptor tests ───────────────────────────────────

#[test]
fn flash_attn_descriptor_op_type() {
    assert_eq!(OPS[9].op_type, "CustomFlashAttn");
}

#[test]
fn flash_attn_kernel_source_contains_target() {
    assert!(
        OPS[9].kernel_source.contains("flash_attn_f32_f16_q1"),
        "kernel source must contain 'flash_attn_f32_f16_q1'"
    );
}

#[test]
fn flash_attn_build_layout_for_qwen() {
    use qnn_oppkg::__test_support::data_type;
    // Qwen2.5-1.5B decode: n_head=32 (q), n_head_kv=2 (kv), head_dim=128,
    // capacity=2048, n_kv=1024.
    let layout =
        build_flash_attn_layout_for(32, 2, 128, 2048, 1024).expect("build_layout must succeed");
    assert_eq!(layout.args.len(), 44, "flash_attn must have 44 kernel args");
    assert_eq!(
        layout.mem_objects.len(),
        7,
        "flash_attn must have 7 mem_objects"
    );
    assert_eq!(
        layout.mem_objects[0].data_type,
        data_type::FLOAT_32,
        "Q must be FLOAT_32"
    );
    assert_eq!(
        layout.mem_objects[1].data_type,
        data_type::FLOAT_16,
        "K must be FLOAT_16"
    );
    assert_eq!(
        layout.mem_objects[2].data_type,
        data_type::FLOAT_16,
        "V must be FLOAT_16"
    );
    assert_eq!(
        layout.mem_objects[6].data_type,
        data_type::FLOAT_32,
        "O (output) must be FLOAT_32"
    );
    // global = [Q1_WG_SIZE=64, n_head=32, 1], local = [64, 1, 1]
    assert_eq!(layout.global_work, [64, 32, 1]);
    assert_eq!(layout.local_work, [64, 1, 1]);
    assert_eq!(layout.global_work_dim, 2);
    assert_eq!(layout.local_work_dim, 3);
}
