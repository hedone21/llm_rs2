//! Host-side metadata tests for the op registry. These run on any architecture;
//! they never touch OpenCL.

use qnn_oppkg::__test_support::{
    arg_to_kernel_arg, build_add_layout_n, build_matmul_layout_for, build_rmsnorm_layout_for,
    build_silumul_layout_for, build_softmax_layout_for, build_state_returns_non_null,
    data_kernel_arg_type, kernel_arg_type,
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
        OPS[4].kernel_source.contains("kernel_silu_mul_simple"),
        "kernel source must contain 'kernel_silu_mul_simple'"
    );
}

#[test]
fn silu_mul_build_layout_for_qwen_ffn() {
    use qnn_oppkg::__test_support::data_type;
    // Qwen2.5-1.5b ffn intermediate: rows=1, dim=8960. total = 8960, size4 = 2240.
    let layout =
        build_silumul_layout_for(1, 8960).expect("build_layout must succeed for (1, 8960)");
    assert_eq!(
        layout.args.len(),
        3,
        "silu_mul_simple must have 3 kernel args"
    );
    assert!(
        layout.mem_objects.len() >= 2,
        "silu_mul must have at least 2 mem_objects (x, y); got {}",
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
fn arg_spec_inout_tensor_conversion() {
    let arg = arg_to_kernel_arg(&ArgSpec::InOutTensor(0));
    assert_eq!(arg.type_, kernel_arg_type::OP_INPUT_READWRITE);
    unsafe {
        assert_eq!(arg.__bindgen_anon_1.tensor.tensorIndex, 0);
        assert_eq!(arg.__bindgen_anon_1.tensor.element, 0);
    }
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
