// Standalone experiment only. Logical K16 scale groups, physical K32 with zeros.
// No A4 quantization, integer accumulation across scales, or production dispatch.
extern "C" __global__ void ts_bench_gemv_nvfp4_int8mma_8(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int outDim, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xd, float alpha, float* __restrict__ y)
{
    // Intentionally no maxrregcount or launch_bounds register pressure override.
    // The offline gate rejects >64 allocated registers or ANY stack/spill.
    __shared__ float partial[32][129];
    const int oldLane = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int g = lane >> 2, t = lane & 3, tile = blockIdx.x * 16;
    if (tile >= outDim) return; // uniform across the entire CTA
    const long long r0 = min(tile + g, outDim - 1), r1 = min(tile + g + 8, outDim - 1);
    const int nblocks = k / 64, scalesPerRow = k / 32;
    float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
    for (int b = oldLane; b < nblocks; b += 32)
    {
        const uint32_t dw0 = *(const uint32_t*)(scales + r0 * (k / 16) + (long long)b * 4);
        const uint32_t dw1 = *(const uint32_t*)(scales + r1 * (k / 16) + (long long)b * 4);
#pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            const int kk = b * 64 + s * 16, sb = kk >> 5;
            // Low nibbles are logical0..7; high nibbles are logical8..15.
            const int word = 2 * s + (t & 1);
            const int2 w0 = ts_q35_e2m1x8(*(const int*)(codes + r0 * (k / 2) + (long long)b * 32 + word * 4));
            const int2 w1 = ts_q35_e2m1x8(*(const int*)(codes + r1 * (k / 2) + (long long)b * 32 + word * 4));
            const int a0 = t < 2 ? w0.x : w0.y, a1 = t < 2 ? w1.x : w1.y;
            const int xb = *(const int*)(xq + (long long)g * k + kk + 4 * t);
            int d0 = 0, d1 = 0, d2 = 0, d3 = 0; // reset for EACH scale16 group
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
            asm volatile(
                "mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 "
                "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                : "+r"(d0), "+r"(d1), "+r"(d2), "+r"(d3)
                : "r"(a0), "r"(a1), "r"(0), "r"(0), "r"(xb), "r"(0));
#else
            __trap();
#endif
            const float ds0 = ts_q35_ue4m3(dw0 >> (8 * s)), ds1 = ts_q35_ue4m3(dw1 >> (8 * s));
            // D columns are2*t and2*t+1, NOT the B-loader's column g.
            const float xs0 = xd[(long long)(2 * t) * scalesPerRow + sb];
            const float xs1 = xd[(long long)(2 * t + 1) * scalesPerRow + sb];
            // Keep the production expression and b/s order. Compiled equivalence
            // is established only by the actual complete-output bit gate.
            acc[0] += ds0 * xs0 * (float)d0;
            acc[1] += ds0 * xs1 * (float)d1;
            acc[2] += ds1 * xs0 * (float)d2;
            acc[3] += ds1 * xs1 * (float)d3;
        }
    }
    // Empty b-warps still write +0 and join the barrier. Scalar stores are
    // deliberate: stride129 makes odd oldLane bases misaligned for float2.
    partial[oldLane][g * 8 + 2 * t] = acc[0];
    partial[oldLane][g * 8 + 2 * t + 1] = acc[1];
    partial[oldLane][(g + 8) * 8 + 2 * t] = acc[2];
    partial[oldLane][(g + 8) * 8 + 2 * t + 1] = acc[3];
    __syncthreads();
    for (int output = oldLane; output < 128; output += 32)
    {
        // Physical lane l reloads the OLD lane l: same xor16,8,4,2,1 tree.
        const float v = ts_q35_warp_sum(partial[lane][output]);
        const int row = tile + output / 8, n = output & 7;
        if (lane == 0 && row < outDim)
            y[(long long)n * outDim + row] = alpha * v;
    }
}
