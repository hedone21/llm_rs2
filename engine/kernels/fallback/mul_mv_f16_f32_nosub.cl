// Optimized F16 GEMV — single-wave, vectorized, no subgroup ops.
// Adreno 830 compatible. Correct on all GPUs.
//
// Key optimizations vs original nosub:
//   1. Manual 4-row unroll with separate registers
//   2. float4 vectorized weight reads via convert_float4(vload4)
//   3. float4 tree reduction: all 4 rows in 1 barrier chain (7 steps)
//
// Dispatch: global = [ceil(n/4)*64, m, 1], local = [64, 1, 1]

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define N_DST 4
#define WG_SIZE 64

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

    int first_row = r0 * N_DST;
    int lid = get_local_id(0);

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong x_base = (ulong)first_row * ne00
                  + (ulong)(i12/r2) * ((ulong)ne00 * ne01)
                  + (ulong)(i13/r3) * ((ulong)ne00 * ne01 * ne02);

    global half * xr0 = x + x_base;
    global half * xr1 = xr0 + ne00;
    global half * xr2 = xr1 + ne00;
    global half * xr3 = xr2 + ne00;
    global float * yr = y + r1 * ne10 + im * ne00 * ne1;

    bool v0 = (first_row     < ne01);
    bool v1 = (first_row + 1 < ne01);
    bool v2 = (first_row + 2 < ne01);
    bool v3 = (first_row + 3 < ne01);

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

    // Vectorized main loop: 4 elements per iteration per thread
    int vec4_end = (ne00 / (WG_SIZE * 4)) * (WG_SIZE * 4);

    for (int i = lid * 4; i < vec4_end; i += WG_SIZE * 4) {
        float4 act = vload4(0, yr + i);
        if (v0) { float4 w = convert_float4(vload4(0, xr0 + i)); s0 += dot(w, act); }
        if (v1) { float4 w = convert_float4(vload4(0, xr1 + i)); s1 += dot(w, act); }
        if (v2) { float4 w = convert_float4(vload4(0, xr2 + i)); s2 += dot(w, act); }
        if (v3) { float4 w = convert_float4(vload4(0, xr3 + i)); s3 += dot(w, act); }
    }

    // Scalar tail
    for (int i = vec4_end + lid; i < ne00; i += WG_SIZE) {
        float yval = yr[i];
        if (v0) s0 += convert_float(xr0[i]) * yval;
        if (v1) s1 += convert_float(xr1[i]) * yval;
        if (v2) s2 += convert_float(xr2[i]) * yval;
        if (v3) s3 += convert_float(xr3[i]) * yval;
    }

    // float4 tree reduction: all 4 rows in single barrier chain
    local float4 scratch[WG_SIZE];
    scratch[lid] = (float4)(s0, s1, s2, s3);
    barrier(CLK_LOCAL_MEM_FENCE);

    for (int s = WG_SIZE / 2; s > 0; s >>= 1) {
        if (lid < s) scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    if (lid == 0) {
        int out_base = r1 * ne0 + im * ne0 * ne1;
        float4 result = scratch[0];
        if (v0) dst[out_base + first_row]     = result.s0;
        if (v1) dst[out_base + first_row + 1] = result.s1;
        if (v2) dst[out_base + first_row + 2] = result.s2;
        if (v3) dst[out_base + first_row + 3] = result.s3;
    }
}
