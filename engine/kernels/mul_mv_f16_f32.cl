#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64  __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define N_DST 4

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

    int first_row = r0 * N_DST;

    int i12 = im % ne12;
    int i13 = im / ne12;

    ulong x_offset = (ulong)first_row * ne00
                    + (ulong)(i12/r2) * ((ulong)ne00 * ne01)
                    + (ulong)(i13/r3) * ((ulong)ne00 * ne01 * ne02);
    global half * xr = x + x_offset;
    global float * yr = y + r1 * ne10 + im * ne00 * ne1;

    int tid = get_sub_group_local_id();
    float sumf[N_DST] = {0.f, 0.f, 0.f, 0.f};

    // Simple scalar loop — no vectorization, just correctness
    for (int i = tid; i < ne00; i += get_sub_group_size()) {
        float yval = yr[i];
        for (int row = 0; row < N_DST; row++) {
            if (first_row + row < ne01) {
                sumf[row] += (float)(xr[(ulong)row * ne00 + i]) * yval;
            }
        }
    }

    float tot[N_DST] = {
        sub_group_reduce_add(sumf[0]),
        sub_group_reduce_add(sumf[1]),
        sub_group_reduce_add(sumf[2]),
        sub_group_reduce_add(sumf[3])
    };

    for (int row = 0; row < N_DST; row++) {
        if (tid == 0 && first_row + row < ne01) {
            dst[r1*ne0 + im*ne0*ne1 + first_row + row] = tot[row];
        }
    }
}
