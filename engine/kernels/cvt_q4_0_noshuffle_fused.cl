//------------------------------------------------------------------------------
// cvt_q4_0_noshuffle_fused.cl  —  ENG-ALG-226 / INV-140
//
// Fused single-dispatch kernel that combines:
//   1) AOS block_q4_0 → noshuffle SOA (nibble rearrange, equivalent to
//      `kernel_convert_block_q4_0_noshuffle` in cvt.cl).
//   2) Row-major → column-major 2D transpose for both dst_q (ushort) and
//      dst_d (half), so the GEMV kernel can issue coalesced reads.
//
// This kernel replaces the legacy 4-step path:
//   GPU convert → host read → CPU transpose → host write × 2
// for Phase 6.5 weight-swap overhead reduction (Galaxy S25 measurement:
// soa_reconvert stage 758 ms / 25 layers / 7 tensors → ~100–150 ms).
//
// Layout contract (matches `OpenCLBackend::convert_q4_0_to_noshuffle` 4-step
// fallback after CPU transpose):
//
//   Source AOS Q4_0:
//     Each block of QK4_0=32 elements is `half d` + `uchar qs[16]`
//     (18 bytes total). num_blocks = ne01 * (ne00 / 32) blocks total.
//     Block layout in memory is row-major:
//       block(row, k) at index `row * blocks_per_row + k`,
//       blocks_per_row = ne00 / 32.
//
//   Destination dst_q (column-major ushort, total ne01 * cols_ushort ushorts):
//     cols_ushort = ne00 / 4   (each block = 16 bytes = 8 ushort).
//     For block (row, k), output position of the i-th ushort is:
//       col          = k * 8 + i,           i ∈ [0, 8)
//       q_t[col * ne01 + row]               (col-major)
//
//   Destination dst_d (column-major half, total ne01 * blocks_per_row halfs):
//     For block (row, k):
//       d_t[k * ne01 + row]
//
// Nibble rearrangement (same as cvt.cl noshuffle):
//   For i ∈ [0, 8):
//     uchar x0 = qs[2*i + 0];
//     uchar x1 = qs[2*i + 1];
//     out_q_lo = (x0 & 0x0F) | ((x1 & 0x0F) << 4);   → out byte i + 0
//     out_q_hi = ((x0 & 0xF0) >> 4) | (x1 & 0xF0);   → out byte i + 8
//   The 16 output bytes are then re-grouped as 8 little-endian ushorts:
//     ushort idx u (u ∈ [0,8)) = (out_byte[2u + 1] << 8) | out_byte[2u + 0]
//
// Dispatch:
//   global_size = num_blocks (= ne01 * blocks_per_row), 1-D.
//   Each work-item handles exactly one source block. Per-thread state:
//     - 16 uchar source nibbles (loaded as uint4 worth of bytes)
//     - 16 uchar rearranged bytes
//     - 8 ushort output values
//   ≈ 12 scalar registers ≈ 3 float4 equivalents → far below the per-thread
//   32 float4 register-spill ceiling on Adreno (CLAUDE.md feedback).
//
//   No subgroup reductions, no SLM. Pure scalar per-thread fan-out write.
//   Adreno OpenCL compiler should keep this in register-only state.
//------------------------------------------------------------------------------

#pragma OPENCL EXTENSION cl_khr_fp16 : enable

#define QK4_0 32

typedef ushort uint16_t;
typedef uchar  uint8_t;

struct block_q4_0
{
    half  d;
    uchar qs[QK4_0 / 2];   // 16 bytes
};

//------------------------------------------------------------------------------
// kernel_convert_block_q4_0_noshuffle_fused
//
// Args:
//   src        — global block_q4_0* (AOS, num_blocks blocks)
//   dst_q      — global ushort* (col-major, ne01 * cols_ushort ushorts)
//   dst_d      — global half*   (col-major, ne01 * blocks_per_row halfs)
//   ne01       — M (number of rows)
//   blocks_per_row — ne00 / 32 (Q4_0 blocks per row)
//
// Note: cols_ushort = blocks_per_row * 8.  We do not pass cols_ushort
// separately — it is derived implicitly from the (k, i) iteration.
//------------------------------------------------------------------------------
__kernel void kernel_convert_block_q4_0_noshuffle_fused(
    __global const struct block_q4_0 * src,
    __global ushort * dst_q,
    __global half   * dst_d,
    const uint ne01,
    const uint blocks_per_row)
{
    const uint gid = get_global_id(0);

    // Decompose linear block index into (row, k).
    //   gid = row * blocks_per_row + k
    const uint row = gid / blocks_per_row;
    const uint k   = gid - row * blocks_per_row;

    // Bounds check (in case global_size is rounded up by the host launcher).
    if (row >= ne01) {
        return;
    }

    __global const struct block_q4_0 * b = src + gid;

    // ── d: write `b->d` to col-major position d_t[k * ne01 + row] ────────────
    dst_d[k * ne01 + row] = b->d;

    // ── q: nibble rearrange + col-major scatter ──────────────────────────────
    //
    // Read the 16 source bytes. Use scalar loads — uchar16 vload would also
    // work but Adreno often emits the same code path, and scalar keeps the
    // output pattern unambiguous for the compiler's DCE.
    uchar out[16];

    #pragma unroll
    for (int i = 0; i < 8; ++i) {
        uchar x0 = b->qs[2*i + 0];
        uchar x1 = b->qs[2*i + 1];

        // convert_uchar() preserves Adreno's pattern from cvt.cl. The original
        // kernel relies on these explicit conversions to coax the compiler
        // into producing correct code on QC's OpenCL stack.
        out[i + 0] = convert_uchar( (x0 & 0x0F) | ((x1 & 0x0F) << 4) );
        out[i + 8] = convert_uchar( ((x0 & 0xF0) >> 4) |  (x1 & 0xF0)  );
    }

    // Repack 16 bytes as 8 little-endian ushorts and scatter to col-major
    // positions q_t[col * ne01 + row], col = k*8 + u, u ∈ [0, 8).
    const uint base_col = k * 8u;

    #pragma unroll
    for (int u = 0; u < 8; ++u) {
        ushort v = (ushort)out[2*u + 0] | ((ushort)out[2*u + 1] << 8);
        dst_q[(base_col + (uint)u) * ne01 + row] = v;
    }
}
