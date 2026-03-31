#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// Adreno-optimized F16 GEMV for large-N matmuls (lm_head, FFN)
//
// N_DST=4 variant: each fiber handles 4 output rows, reusing activation across them.
// WG covers 64 * 4 = 256 output rows (vs 128 for N_DST=2).
// Halves the number of workgroups for large vocab (e.g. 262144 -> 1024 WGs).
//
// Layout: local_size = [64, 4, 1]  (4 subgroups x 64 lanes)
//   - 4 waves split K dimension: each wave handles K/4
//   - Each fiber handles 4 output rows (N_DST=4)
//   - WG covers 64 * 4 = 256 output rows
//   - Wave-level partial sums reduced via local memory
//
// Dispatch: global = [ceil(n/256)*64, 4, m], local = [64, 4, 1]
#define N_DST 4
#define N_SIMDGROUP 4
#define N_SIMDWIDTH 64

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32_l4(
        global void * src0,
        ulong offset0,
        global float * src1,
        ulong offset1,
        global float * dst,
        ulong offsetd,
        int ne00,
        int ne01,
        int ne02,
        int ne10,
        int ne12,
        int ne0,
        int ne1,
        int r2,
        int r3
) {
    global half * x = (global half *)((global char*)src0 + offset0);
    global float * y = (global float *)((global char*)src1 + offset1);
    dst = (global float*)((global char*)dst + offsetd);

    int r0 = get_group_id(0);   // WG index along output rows
    int r1 = get_group_id(1);   // batch m
    int im = get_group_id(2);   // multi-head batch

    int tid = get_sub_group_local_id();   // lane 0..63
    int wave_id = get_local_id(1);        // wave 0..3

    // 256 rows per WG, 4 rows per fiber
    int first_row = r0 * (N_SIMDWIDTH * N_DST) + tid * N_DST;

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong x_offset = (ulong)first_row * ne00
                    + (ulong)(i12/r2) * ((ulong)ne00 * ne01)
                    + (ulong)(i13/r3) * ((ulong)ne00 * ne01 * ne02);

    global half * xr0 = x + x_offset;
    global half * xr1 = xr0 + ne00;
    global half * xr2 = xr1 + ne00;
    global half * xr3 = xr2 + ne00;

    global float * yr = y + r1 * ne10 + im * ne00 * ne1;

    // K-split: each wave handles a contiguous chunk of K
    int k_per_wave = (ne00 + N_SIMDGROUP - 1) / N_SIMDGROUP;
    int k_start = wave_id * k_per_wave;
    int k_end = k_start + k_per_wave;
    if (k_end > ne00) k_end = ne00;

    float sumf0 = 0.0f;
    float sumf1 = 0.0f;
    float sumf2 = 0.0f;
    float sumf3 = 0.0f;

    bool row0_valid = (first_row     < ne01);
    bool row1_valid = (first_row + 1 < ne01);
    bool row2_valid = (first_row + 2 < ne01);
    bool row3_valid = (first_row + 3 < ne01);

    // Vectorized main loop: float4 dot products
    int vec4_end = k_start + ((k_end - k_start) / (N_SIMDWIDTH * 4)) * (N_SIMDWIDTH * 4);

    for (int i = k_start + tid * 4; i < vec4_end; i += N_SIMDWIDTH * 4) {
        float4 act = vload4(0, yr + i);
        if (row0_valid) sumf0 += dot(convert_float4(vload4(0, xr0 + i)), act);
        if (row1_valid) sumf1 += dot(convert_float4(vload4(0, xr1 + i)), act);
        if (row2_valid) sumf2 += dot(convert_float4(vload4(0, xr2 + i)), act);
        if (row3_valid) sumf3 += dot(convert_float4(vload4(0, xr3 + i)), act);
    }

    // Scalar tail: remaining elements in this wave's chunk
    for (int i = vec4_end + tid; i < k_end; i += N_SIMDWIDTH) {
        float yval = yr[i];
        if (row0_valid) sumf0 += convert_float(xr0[i]) * yval;
        if (row1_valid) sumf1 += convert_float(xr1[i]) * yval;
        if (row2_valid) sumf2 += convert_float(xr2[i]) * yval;
        if (row3_valid) sumf3 += convert_float(xr3[i]) * yval;
    }

    // -- Inter-wave reduction via local memory --
    // Waves 1-3 write partial sums, wave 0 accumulates all.
    local float4 reduce_lm[N_SIMDWIDTH * 3];  // 64 * 3 float4 = 3072 bytes

    float4 partial = (float4)(sumf0, sumf1, sumf2, sumf3);

    if (wave_id == 1) reduce_lm[tid]                    = partial;
    if (wave_id == 2) reduce_lm[tid + N_SIMDWIDTH]      = partial;
    if (wave_id == 3) reduce_lm[tid + N_SIMDWIDTH * 2]  = partial;
    barrier(CLK_LOCAL_MEM_FENCE);

    if (wave_id == 0) {
        partial += reduce_lm[tid];
        partial += reduce_lm[tid + N_SIMDWIDTH];
        partial += reduce_lm[tid + N_SIMDWIDTH * 2];

        int out_base = r1 * ne0 + im * ne0 * ne1;
        if (row0_valid) dst[out_base + first_row]     = partial.s0;
        if (row1_valid) dst[out_base + first_row + 1] = partial.s1;
        if (row2_valid) dst[out_base + first_row + 2] = partial.s2;
        if (row3_valid) dst[out_base + first_row + 3] = partial.s3;
    }
}
