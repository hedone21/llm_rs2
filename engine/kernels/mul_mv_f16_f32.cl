#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

// Adreno-optimized F16 GEMV kernel.
// N_DST=4 output rows per wave, vectorized half4/float4 loads,
// pre-computed row pointers, float4 accumulator for better ILP.
#define N_DST 4
#define N_SIMDGROUP 1
#define N_SIMDWIDTH 64

#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
kernel void kernel_mul_mat_f16_f32(
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

    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = (r0 * N_SIMDGROUP + get_sub_group_id()) * N_DST;

    int i12 = im % ne12;
    int i13 = im / ne12;

    // Weight base pointer for first_row
    ulong x_base = (ulong)first_row * ne00
                  + (ulong)(i12/r2) * ((ulong)ne00 * ne01)
                  + (ulong)(i13/r3) * ((ulong)ne00 * ne01 * ne02);

    // Pre-compute row pointers (avoid repeated multiply in inner loop)
    global half * xr0 = x + x_base;
    global half * xr1 = xr0 + ne00;
    global half * xr2 = xr1 + ne00;
    global half * xr3 = xr2 + ne00;

    // Activation pointer
    global float * yr = y + r1 * ne10 + im * ne00 * ne1;

    int tid = get_sub_group_local_id();

    // float4 accumulators: exploit SIMD lanes for horizontal sum later
    float4 acc0 = (float4)(0.f);
    float4 acc1 = (float4)(0.f);
    float4 acc2 = (float4)(0.f);
    float4 acc3 = (float4)(0.f);

    // Vectorized main loop: 4 F16/F32 elements per thread per iteration
    int ne00_vec = (ne00 / 4) * 4;
    for (int i = tid * 4; i < ne00_vec; i += N_SIMDWIDTH * 4) {
        float4 act = vload4(0, yr + i);

        if (first_row     < ne01) acc0 += convert_float4(vload4(0, xr0 + i)) * act;
        if (first_row + 1 < ne01) acc1 += convert_float4(vload4(0, xr1 + i)) * act;
        if (first_row + 2 < ne01) acc2 += convert_float4(vload4(0, xr2 + i)) * act;
        if (first_row + 3 < ne01) acc3 += convert_float4(vload4(0, xr3 + i)) * act;
    }

    // Reduce float4 accumulators to scalar
    float sumf0 = acc0.s0 + acc0.s1 + acc0.s2 + acc0.s3;
    float sumf1 = acc1.s0 + acc1.s1 + acc1.s2 + acc1.s3;
    float sumf2 = acc2.s0 + acc2.s1 + acc2.s2 + acc2.s3;
    float sumf3 = acc3.s0 + acc3.s1 + acc3.s2 + acc3.s3;

    // Scalar tail
    for (int i = ne00_vec + tid; i < ne00; i += N_SIMDWIDTH) {
        float yval = yr[i];
        if (first_row     < ne01) sumf0 += convert_float(xr0[i]) * yval;
        if (first_row + 1 < ne01) sumf1 += convert_float(xr1[i]) * yval;
        if (first_row + 2 < ne01) sumf2 += convert_float(xr2[i]) * yval;
        if (first_row + 3 < ne01) sumf3 += convert_float(xr3[i]) * yval;
    }

    // 4 parallel subgroup reductions
    float tot0 = sub_group_reduce_add(sumf0);
    float tot1 = sub_group_reduce_add(sumf1);
    float tot2 = sub_group_reduce_add(sumf2);
    float tot3 = sub_group_reduce_add(sumf3);

    int out_base = r1*ne0 + im*ne0*ne1;
    if (tid == 0) {
        if (first_row     < ne01) dst[out_base + first_row    ] = tot0;
        if (first_row + 1 < ne01) dst[out_base + first_row + 1] = tot1;
        if (first_row + 2 < ne01) dst[out_base + first_row + 2] = tot2;
        if (first_row + 3 < ne01) dst[out_base + first_row + 3] = tot3;
    }
}
