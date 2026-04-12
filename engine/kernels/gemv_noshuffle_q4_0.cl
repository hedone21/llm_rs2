// gemv_noshuffle_q4_0.cl — Global buffer MVP of llama.cpp's gemv_noshuffle.
//
// Ported from llama.cpp (image1d_buffer_t -> global buffer reads).
// Keeps the core optimizations: SOA layout, 4-wave K-split, sub_group_broadcast,
// local memory reduction.
//
// Weight layout (SOA, produced by kernel_convert_block_q4_0_noshuffle in cvt.cl):
//   src0_q: rearranged nibbles, 16 bytes per block (QK4_0/2)
//   src0_d: half scale per block
//
// Each work-item computes 2 output rows (float2 totalSum).
//   global_work_size = [ne01/2, N_SIMDGROUP, 1]  (ne01 = number of rows)
//   local_work_size  = [SIMDGROUP_WIDTH, N_SIMDGROUP, 1]
//
// Compile-time defines required:
//   -DLINE_STRIDE_A=<ne01/2>  (row-pair count = half the total rows)
//   -DBLOCK_STRIDE_A=<ne01>   (total rows, for stepping through SOA nibble words)
//   -DSIMDGROUP_WIDTH=<64>    (subgroup width on Adreno with half-wave)
//
// Understanding the SOA nibble layout:
//
// After kernel_convert_block_q4_0_noshuffle, for a matrix with `num_rows` rows
// and `K` columns (K/QK4_0 blocks per row), the SOA nibble buffer has:
//   - Total blocks = num_rows * (K / QK4_0)
//   - Each block = 16 bytes = 8 ushort = 4 uint
//
// The image1d_buffer_t in llama.cpp wraps this as R32UI format (1 uint per texel).
// read_imageui(src0_q, texel_idx).x returns one uint (4 bytes = 4 nibbles = 2 ushort).
//
// For the global buffer version, we use uint* indexing with the same texel offsets.
// The layout is: all blocks are stored contiguously, row after row.
//   texel_idx = row * (K/QK4_0) * 4 + block * 4 + word_in_block
// But the kernel uses the pre-computed strides LINE_STRIDE_A and BLOCK_STRIDE_A:
//   texel_idx = gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * word_idx
//
// where gid = get_global_id(0) = row_pair_index.
//
// In the llama.cpp image kernel:
//   LINE_STRIDE_A  = ne00/QK4_0/2  (= blocks_per_row / 2, since 2 rows per pair,
//                                    but actually = ne01/2 for row interleaving)
//   BLOCK_STRIDE_A = ne00/QK4_0    (blocks_per_row)
//
// Wait — re-reading the llama.cpp dispatch code more carefully:
//   LINE_STRIDE_A = ne00 / 32 / 2 = blocks_per_row / 2
//   BLOCK_STRIDE_A = ne00 / 32 = blocks_per_row
// These operate in "texel" (uint) units within the image1d_buffer_t, where
// each texel = 1 uint = 4 bytes of the SOA nibble buffer.
//
// Since each block = 16 bytes = 4 uint texels, and there are blocks_per_row blocks
// per row, each row occupies blocks_per_row * 4 texels. Two consecutive rows
// occupy blocks_per_row * 8 texels.
//
// The indexing gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * i:
//   - gid selects which row-pair (0 .. ne01/2-1)
//   - k selects which K-block (0 .. blocks_per_row-1)
//   - LINE_STRIDE_A * i steps through the 8 uint words of a block-pair
//
// This means the data is stored interleaved: for each "column" (uint position
// within a block), all row-pairs are stored together. This is a transposed
// layout compared to naive row-major. That's what "noshuffle" refers to —
// the SOA conversion already arranges data for coalesced access.
//
// For our global uint* buffer, we use identical indexing.

#pragma OPENCL EXTENSION cl_khr_fp16 : enable
#pragma OPENCL EXTENSION cl_khr_subgroups : enable

#ifdef cl_qcom_reqd_sub_group_size
#pragma OPENCL EXTENSION cl_qcom_reqd_sub_group_size : enable
#define ADRENO_GPU 1
#define REQD_SUBGROUP_SIZE_64 __attribute__((qcom_reqd_sub_group_size("half")))
#endif

#define QK4_0 32
#define N_SIMDGROUP 4

// ---------------------------------------------------------------------------
// Dequantize-and-accumulate macros (scalar sub_group_broadcast, 1 element at a time).
// Identical logic to llama.cpp gemv_noshuffle.cl.
// ---------------------------------------------------------------------------

#define dequantizeBlockAccum_ns_sgbroadcast_1_hi(total_sums, bits4, scale, y) \
    float shared_y; \
    shared_y = sub_group_broadcast(y.s0, 0); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 0); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 0); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 0); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 0); \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 0); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 0); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 0); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 1); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 1); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 1); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 1); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 1); \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 1); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 1); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 1); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \


#define dequantizeBlockAccum_ns_sgbroadcast_1_lo(total_sums, bits4, scale, y) \
    shared_y = sub_group_broadcast(y.s0, 2); \
    total_sums.s0 += ((bits4.s0 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s1 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 2); \
    total_sums.s0 += (((bits4.s0 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 2); \
    total_sums.s0 += (((bits4.s0 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 2); \
    total_sums.s0 += (((bits4.s0 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s1 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 2); \
    total_sums.s0 += ((bits4.s2 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s3 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 2); \
    total_sums.s0 += (((bits4.s2 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 2); \
    total_sums.s0 += (((bits4.s2 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 2); \
    total_sums.s0 += (((bits4.s2 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s3 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s0, 3); \
    total_sums.s0 += ((bits4.s4 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s5 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s1, 3); \
    total_sums.s0 += (((bits4.s4 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s2, 3); \
    total_sums.s0 += (((bits4.s4 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s3, 3); \
    total_sums.s0 += (((bits4.s4 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s5 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s4, 3); \
    total_sums.s0 += ((bits4.s6 & 0x000F) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += ((bits4.s7 & 0x000F) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s5, 3); \
    total_sums.s0 += (((bits4.s6 & 0x00F0) >> 4) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x00F0) >> 4) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s6, 3); \
    total_sums.s0 += (((bits4.s6 & 0x0F00) >> 8) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0x0F00) >> 8) - 8) * scale.s1 * shared_y; \
    shared_y = sub_group_broadcast(y.s7, 3); \
    total_sums.s0 += (((bits4.s6 & 0xF000) >> 12) - 8) * scale.s0 * shared_y; \
    total_sums.s1 += (((bits4.s7 & 0xF000) >> 12) - 8) * scale.s1 * shared_y; \

// ---------------------------------------------------------------------------
// kernel_gemv_noshuffle_q4_0
//
// Global-buffer MVP: reads weight nibbles and scales from global memory
// instead of image1d_buffer_t. Activation (src1) is also a plain global
// float buffer.
//
// The data layout is identical to the image1d_buffer_t version. The image
// format was R32UI — each texel is one uint (4 bytes). We read from
// global uint* with the same linear offsets.
//
// For activation, the image format was RGBA32F — each texel is float4.
// We read from global float4*.
//
// Dispatch:
//   global = [ne01/2, N_SIMDGROUP]
//   local  = [SIMDGROUP_WIDTH, N_SIMDGROUP]
// ---------------------------------------------------------------------------
#ifdef ADRENO_GPU
REQD_SUBGROUP_SIZE_64
#endif
__kernel void kernel_gemv_noshuffle_q4_0(
        global uint   * src0_q,   // SOA nibbles (uint for R32UI texel compat)
        global half2  * src0_d,   // SOA scales (half2 = scale for 2 rows)
        global float4 * src1,     // activation vector (float4 for RGBA32F texel compat)
        global float  * dst,      // output vector
        int ne00,                 // K dimension (number of elements per row)
        int ne01                  // M dimension (number of output rows)
)
{
    uint groupId = get_local_id(1);    // wave index (0..3)
    uint gid     = get_global_id(0);   // row-pair index
    ushort slid  = get_sub_group_local_id();

    __private uint4     regA;
    __private half2     regS;
    __private float8    regB;

    __private float2 totalSum = (float2)(0.0f);

    // loop along K in block granularity, skip N_SIMDGROUP blocks every iter
    for (uint k = groupId; k < ((uint)ne00 / QK4_0); k += N_SIMDGROUP) {

        // Load scale for this row-pair's block
        regS = src0_d[gid + k * LINE_STRIDE_A];

        // First 4 fibers in each wave load 8 activation values (= 1 block of QK4_0=32)
        // llama.cpp: regB.s0123 = read_imagef(src1, (slid * 2 + k * 8))
        //            regB.s4567 = read_imagef(src1, (1 + slid * 2 + k * 8))
        // read_imagef returns float4 from RGBA32F texels.
        // src1 is [K/4] float4 texels. k*8 = k * (QK4_0/4) since QK4_0=32 → 8 texels/block.
        if (slid < 4) {
            regB.s0123 = src1[slid * 2 + k * 8];
            regB.s4567 = src1[1 + slid * 2 + k * 8];
        }

        // Load weight nibbles. read_imageui(src0_q, offset).x → src0_q[offset]
        regA.s0 = src0_q[gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 0];
        regA.s1 = src0_q[gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 1];
        regA.s2 = src0_q[gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 2];
        regA.s3 = src0_q[gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 3];

        dequantizeBlockAccum_ns_sgbroadcast_1_hi(totalSum, as_ushort8(regA), regS, regB);

        regA.s0 = src0_q[gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 4];
        regA.s1 = src0_q[gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 5];
        regA.s2 = src0_q[gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 6];
        regA.s3 = src0_q[gid + k * BLOCK_STRIDE_A + LINE_STRIDE_A * 7];

        dequantizeBlockAccum_ns_sgbroadcast_1_lo(totalSum, as_ushort8(regA), regS, regB);
    }

    // Reduction in local memory, assumes N_SIMDGROUP=4
    __local float2 reduceLM[SIMDGROUP_WIDTH * 3];
    if (groupId == 1) reduceLM[SIMDGROUP_WIDTH * 0 + slid] = totalSum;
    if (groupId == 2) reduceLM[SIMDGROUP_WIDTH * 1 + slid] = totalSum;
    if (groupId == 3) reduceLM[SIMDGROUP_WIDTH * 2 + slid] = totalSum;
    barrier(CLK_LOCAL_MEM_FENCE);
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 0 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 1 + slid];
    if (groupId == 0) totalSum += reduceLM[SIMDGROUP_WIDTH * 2 + slid];

    // 2 outputs per fiber in wave 0
    if (groupId == 0) {
        vstore2(totalSum, 0, &(dst[gid * 2]));
    }
}
