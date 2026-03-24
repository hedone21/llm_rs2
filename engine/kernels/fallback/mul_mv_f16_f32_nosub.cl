// Fallback F16 GEMV kernel without subgroup operations (for NVIDIA, pocl, etc.)
// Same function name and signature as mul_mv_f16_f32.cl
// Uses local memory tree reduction instead of sub_group_reduce_add().
// Reads F16 via vload_half() which is core OpenCL (no cl_khr_fp16 needed).

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define N_DST 4

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

    // Each workgroup handles N_DST=4 rows
    int r0 = get_group_id(0);
    int r1 = get_group_id(1);
    int im = get_group_id(2);

    int first_row = r0 * N_DST;

    int lid = get_local_id(0);
    int local_size = get_local_size(0);

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong x_base = (ulong)first_row * ne00
                  + (ulong)(i12/r2) * ((ulong)ne00 * ne01)
                  + (ulong)(i13/r3) * ((ulong)ne00 * ne01 * ne02);

    global half * xrow = x + x_base;
    global float * yr = y + r1 * ne10 + im * ne00 * ne1;

    // Partial dot products for 4 rows
    float sumf[N_DST];
    for (int row = 0; row < N_DST; row++) sumf[row] = 0.0f;

    for (int i = lid; i < ne00; i += local_size) {
        float yval = yr[i];
        for (int row = 0; row < N_DST; row++) {
            if (first_row + row < ne01)
                sumf[row] += vload_half(i, xrow + (ulong)row * ne00) * yval;
        }
    }

    // Tree reduction in local memory, one row at a time
    local float scratch[256];
    int out_base = r1 * ne0 + im * ne0 * ne1;

    for (int row = 0; row < N_DST; row++) {
        scratch[lid] = sumf[row];
        barrier(CLK_LOCAL_MEM_FENCE);
        for (int s = local_size / 2; s > 0; s >>= 1) {
            if (lid < s) scratch[lid] += scratch[lid + s];
            barrier(CLK_LOCAL_MEM_FENCE);
        }
        if (lid == 0 && first_row + row < ne01)
            dst[out_base + first_row + row] = scratch[0];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
}
