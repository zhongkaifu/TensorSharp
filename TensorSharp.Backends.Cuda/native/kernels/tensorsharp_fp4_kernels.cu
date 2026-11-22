// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// NVFP4 products for the direct-CUDA backend's quantized matmul, whatever model runs on it, on GPUs whose image
// carries the block-scaled FP4 tensor cores (sm_120a / sm_121a). A resident NVFP4 matrix (ggml block_nvfp4: 64 values
// in 36 bytes - four UE4M3 scales, one per 16 values, then 32 bytes of E2M1 codes; within 16 values the low nibbles
// are values 0-7, the high ones 8-15) is kept as two planes instead of 36-byte blocks: codes [rows][k/2] (ggml's
// nibble order kept) then UE4M3 scales [rows][k/16]. Same bytes, but every 64-value code run is 16-byte aligned, which
// the cp.async, TMA and vector loads need.
//
//   * ts_fp4_repack_nvfp4: ggml block_nvfp4 rows -> the planes (once, as the weight becomes resident).
//   * ts_fp4_quantize_q81_f32 + ts_fp4_gemv_nvfp4_<n>: up to 16 rows (decode, verify windows) against q8 activations, every
//     row's sums a one-row call's.
//   * ts_fp4_quantize_nvfp4_f32: F32 rows -> NVFP4 activation planes and row scales (ggml-cuda's quantize_mmq_nvfp4);
//     ts_fp4_quantize_swiglu_nvfp4_f32 the planes of silu(gate) * up of fused gate|up rows, ts_fp4_add_rmsnorm_nvfp4_f32
//     those of a residual add's RMSNorm.
//   * ts_fp4_gemm_w4a4_<bn>: 17+ rows on the FP4 tensor cores through a cp.async ring, split over K for few rows.
//   * ts_fp4_gemm_w4a4_tma<bm>: prompt rows, TMA-fed and warp-specialized (the same products in the same order);
//     ts_fp4_gemm_w4a4_tma<bm>_swiglu a fused gate|up projection's rows with silu(gate) * up in the epilogue.
//   * ts_fp4_get_rows_f32: rows of a planar matrix as F32 (an NVFP4 table's lookup).
//   * ts_fp4_quantize_q81_f16 + ts_fp4_gemm_w4a8_tma: rows of q8_1 activations on the F16 tensor cores, TMA-fed, for a
//     model that keeps 8-bit activations on its prompt rows (--nvfp4-prompt-activations q8): the matvec's products at
//     any number of rows; ts_fp4_gemm_w4a8 the same products on the int8 tensor cores where TMA cannot take the matrix.
//
// The kernels came from the removed direct-CUDA Qwen3.5 engine (bit-identical arithmetic). The tensor-core bodies exist
// only in an sm_120a / sm_121a image (CommittedBaseline false: no compute_75 baseline); ts_fp4_mma_caps reports whether
// the loaded image has them, and the backend keeps NVFP4 weights as ggml blocks on its other kernels where it does not.
//
// Bump TS_FP4_VERSION, and CudaFp4Kernels.CurrentVersion with it, whenever a kernel is added or changes signature.
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>

#define TS_FP4_NVFP4_BLOCK 64
#define TS_FP4_NVFP4_BLOCK_BYTES 36

#define TS_FP4_VERSION 9

__device__ __forceinline__ float ts_fp4_warp_sum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
    return v;
}

__device__ __forceinline__ float ts_fp4_warp_max(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off, 32));
    return v;
}
__device__ __forceinline__ float ts_fp4_silu(float x) { return x / (1.0f + expf(-x)); }

__device__ __forceinline__ int ts_fp4_dp4a(int a, int b, int c)
{
    return __dp4a(a, b, c);
}

// UE4M3 block scale as ggml-cuda decodes it: the NaN codes read as zero, and the value is
// halved because the E2M1 table below is doubled to stay integral.
__device__ __forceinline__ float ts_fp4_ue4m3(uint32_t x)
{
    x &= 0xFF;
    if (x == 0x7F || x == 0xFF)
        return 0.0f;
    const uint32_t e = (x >> 3) & 0xF, m = x & 7;
    // Subnormal: m * 2^-9; normal: (1 + m/8) * 2^(e-7). Both exact in F32.
    const float v = e == 0 ? (float)m * (1.0f / 512.0f) : __uint_as_float(((e + 120) << 23) | (m << 20));
    return 0.5f * v;
}

// Eight 4-bit E2M1 codes (one int, low nibble first) -> two ints of int8 values (2x the E2M1
// value): .x holds the low nibbles of the four bytes, .y the high nibbles. The table
// {0,1,2,3,4,6,8,12, 0,-1,-2,-3,-4,-6,-8,-12} lives in four registers; byte_perm picks the
// positive or negative half per code by its sign bit.
__device__ __forceinline__ int2 ts_fp4_e2m1x8(int q)
{
    const uint32_t t0 = 0x03020100u, t1 = 0x0C080604u, t2 = 0xFDFEFF00u, t3 = 0xF4F8FAFCu;
    const uint32_t q4 = (uint32_t)q;
    const uint32_t select = 0x32103210u | ((q4 & 0x88888888u) >> 1);
    uint32_t tmp[2];
#pragma unroll
    for (int i = 0; i < 2; ++i)
    {
        const uint32_t shift = 16 * i;
        const uint32_t lo = __byte_perm(t0, t1, q4 >> shift);
        const uint32_t hi = __byte_perm(t2, t3, q4 >> shift);
        tmp[i] = __byte_perm(lo, hi, select >> shift);
    }
    // tmp holds the values in nibble order (lo0, hi0, lo1, hi1, ...); split them.
    return make_int2((int)__byte_perm(tmp[0], tmp[1], 0x6420), (int)__byte_perm(tmp[0], tmp[1], 0x7531));
}

// ---------------------------------------------------------------------------
// Activation quantization: one warp per 32-value block of a row. scale = max|x| / 127,
// q = round(x / scale) - ggml's quantize_q8_1 (the block sum it also stores is unused by
// NVFP4's dot). xq: [rows, k] int8, xd: [rows, k / 32] F32.
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_fp4_quantize_q81_f32(
    const float* __restrict__ x, int8_t* __restrict__ xq, float* __restrict__ xd, int k, int rows)
{
    const int blocks = k >> 5;
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= (long long)blocks * rows)
        return;
    const long long row = warp / blocks;
    const int b = (int)(warp % blocks);
    const float v = x[row * k + b * 32 + lane];
    const float amax = ts_fp4_warp_max(fabsf(v));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : __float2int_rn(v / d);
    xq[row * k + b * 32 + lane] = (int8_t)q;
    if (lane == 0)
        xd[row * blocks + b] = d;
}

// The same quantization with the integers stored as F16 (exact), for the W4A8 GEMM on the F16 tensor cores
// (ts_fp4_gemm_w4a8_tma). xh: [rows, k] F16, xd: [rows, k / 32] F32.
extern "C" __global__ void ts_fp4_quantize_q81_f16(
    const float* __restrict__ x, __half* __restrict__ xh, float* __restrict__ xd, int k, int rows)
{
    const int blocks = k >> 5;
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= (long long)blocks * rows)
        return;
    const long long row = warp / blocks;
    const int b = (int)(warp % blocks);
    const float v = x[row * k + b * 32 + lane];
    const float amax = ts_fp4_warp_max(fabsf(v));
    const float d = amax / 127.0f;
    const int q = amax == 0.0f ? 0 : __float2int_rn(v / d);
    xh[row * k + b * 32 + lane] = __int2half_rn(q);
    if (lane == 0)
        xd[row * blocks + b] = d;
}

// ---------------------------------------------------------------------------
// ggml block_nvfp4 rows -> code plane [rows][k/2] and scale plane [rows][k/16]. One thread
// per 64-value block; the 36-byte blocks are 4-byte aligned. parts > 1: source row rowBase + r of
// part `part` lands at plane row (row / 8) * 8 * parts + part * 8 + row % 8 (groups of 8 rows of
// each part interleaved, as the fused gate|up projection is stored); otherwise at row r.
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_fp4_repack_nvfp4(
    const uint8_t* __restrict__ src, long long srcRowBytes, uint8_t* __restrict__ codes,
    uint8_t* __restrict__ scales, int rows, int k, int rowBase, int parts, int part)
{
    const int perRow = k / TS_FP4_NVFP4_BLOCK;
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * perRow)
        return;
    const long long rs = i / perRow;
    const int b = (int)(i % perRow);
    const long long row = rowBase + rs;
    const long long r = parts > 1 ? row / 8 * 8 * parts + part * 8 + row % 8 : rs;
    const uint32_t* w = (const uint32_t*)(src + rs * srcRowBytes + (long long)b * TS_FP4_NVFP4_BLOCK_BYTES);
    *(uint32_t*)(scales + r * (k / 16) + (long long)b * 4) = w[0];
    uint4* dst = (uint4*)(codes + r * (k / 2) + (long long)b * 32);
    dst[0] = make_uint4(w[1], w[2], w[3], w[4]);
    dst[1] = make_uint4(w[5], w[6], w[7], w[8]);
}

// ---------------------------------------------------------------------------
// NVFP4 GEMV for N <= 16 activation rows (a decode step's, a verify window's: each row's sums the same whatever N). A warp owns R consecutive weight rows; lane l reads
// blocks l, l + 32, ... of each (36 contiguous bytes, so the warp's loads cover contiguous
// spans) and every activation slice it loads serves all R rows: past a few activation rows the
// kernel is bound by those loads, not by the weight stream.
// y[n, r] = alpha * sum_k W[r, k] x[n, k].
// ---------------------------------------------------------------------------
template <int N, int R>
__device__ __forceinline__ void ts_fp4_gemv_nvfp4(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int outDim, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xd, float alpha, float* __restrict__ y)
{
    const int warpsPerBlock = blockDim.x >> 5;
    const int r0 = (blockIdx.x * warpsPerBlock + (threadIdx.x >> 5)) * R;
    const int lane = threadIdx.x & 31;
    if (r0 >= outDim)
        return;
    const int nblocks = k / TS_FP4_NVFP4_BLOCK;
    const int scalesPerRow = k >> 5;

    float acc[N][R];
#pragma unroll
    for (int n = 0; n < N; ++n)
#pragma unroll
        for (int j = 0; j < R; ++j)
            acc[n][j] = 0.0f;

    for (int b = lane; b < nblocks; b += 32)
    {
        uint32_t dw[R], qs[R][8];
#pragma unroll
        for (int j = 0; j < R; ++j)
        {
            // Rows past the end repeat the last row; their results are dropped.
            const long long r = min(r0 + j, outDim - 1);
            const uint4* c = (const uint4*)(codes + r * (k / 2) + (long long)b * 32);
            const uint4 c0 = c[0], c1 = c[1];
            qs[j][0] = c0.x; qs[j][1] = c0.y; qs[j][2] = c0.z; qs[j][3] = c0.w;
            qs[j][4] = c1.x; qs[j][5] = c1.y; qs[j][6] = c1.z; qs[j][7] = c1.w;
            dw[j] = *(const uint32_t*)(scales + r * (k / 16) + (long long)b * 4);
        }
#pragma unroll
        for (int s = 0; s < 4; ++s)
        {
            const int kk = b * TS_FP4_NVFP4_BLOCK + s * 16;
            const int sb = kk >> 5;
            int2 v0[R], v1[R];
            float ds[R];
#pragma unroll
            for (int j = 0; j < R; ++j)
            {
                ds[j] = ts_fp4_ue4m3(dw[j] >> (8 * s));
                v0[j] = ts_fp4_e2m1x8((int)qs[j][2 * s]);       // values 0-3 | 8-11
                v1[j] = ts_fp4_e2m1x8((int)qs[j][2 * s + 1]);   // values 4-7 | 12-15
            }
#pragma unroll
            for (int n = 0; n < N; ++n)
            {
                const int4 a = *(const int4*)(xq + (long long)n * k + kk);
                const float xs = xd[(long long)n * scalesPerRow + sb];
#pragma unroll
                for (int j = 0; j < R; ++j)
                {
                    int sumi = ts_fp4_dp4a(v0[j].x, a.x, 0);
                    sumi = ts_fp4_dp4a(v1[j].x, a.y, sumi);
                    sumi = ts_fp4_dp4a(v0[j].y, a.z, sumi);
                    sumi = ts_fp4_dp4a(v1[j].y, a.w, sumi);
                    acc[n][j] += ds[j] * xs * (float)sumi;
                }
            }
        }
    }
#pragma unroll
    for (int n = 0; n < N; ++n)
#pragma unroll
        for (int j = 0; j < R; ++j)
        {
            const float v = ts_fp4_warp_sum(acc[n][j]);
            if (lane == 0 && r0 + j < outDim)
                y[(long long)n * outDim + r0 + j] = alpha * v;
        }
}

// Rows per warp by activation rows: one row keeps the most warps in flight while the weight
// stream is the only cost; more activation rows amortize their loads over more weight rows.
#define TS_FP4_GEMV(N, R) \
    extern "C" __global__ void ts_fp4_gemv_nvfp4_##N( \
        const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int outDim, int k, \
        const int8_t* __restrict__ xq, const float* __restrict__ xd, float alpha, float* __restrict__ y) \
    { ts_fp4_gemv_nvfp4<N, R>(codes, scales, outDim, k, xq, xd, alpha, y); }

#ifndef TS_FP4_GEMV_ROWS
#define TS_FP4_GEMV_ROWS(N) ((N) <= 1 ? 1 : (N) <= 3 ? 2 : 4)
#endif
TS_FP4_GEMV(1, TS_FP4_GEMV_ROWS(1))
TS_FP4_GEMV(2, TS_FP4_GEMV_ROWS(2))
TS_FP4_GEMV(3, TS_FP4_GEMV_ROWS(3))
TS_FP4_GEMV(4, TS_FP4_GEMV_ROWS(4))
TS_FP4_GEMV(5, TS_FP4_GEMV_ROWS(5))
TS_FP4_GEMV(6, TS_FP4_GEMV_ROWS(6))
TS_FP4_GEMV(7, TS_FP4_GEMV_ROWS(7))
TS_FP4_GEMV(8, TS_FP4_GEMV_ROWS(8))
TS_FP4_GEMV(9, TS_FP4_GEMV_ROWS(9))
TS_FP4_GEMV(10, TS_FP4_GEMV_ROWS(10))
TS_FP4_GEMV(11, TS_FP4_GEMV_ROWS(11))
TS_FP4_GEMV(12, TS_FP4_GEMV_ROWS(12))
TS_FP4_GEMV(13, TS_FP4_GEMV_ROWS(13))
TS_FP4_GEMV(14, TS_FP4_GEMV_ROWS(14))
TS_FP4_GEMV(15, TS_FP4_GEMV_ROWS(15))
TS_FP4_GEMV(16, TS_FP4_GEMV_ROWS(16))

/// Weight rows one warp of ts_fp4_gemv_nvfp4_<n> owns (the launcher sizes its grid by it).
extern "C" __global__ void ts_fp4_gemv_rows_per_warp(int* out)
{
#pragma unroll
    for (int n = 1; n <= 16; ++n)
        out[n] = TS_FP4_GEMV_ROWS(n);
}


// ---------------------------------------------------------------------------
// Activation rows -> NVFP4 planes + per-row scale, ggml-cuda's quantize_mmq_nvfp4 numerics.
// Codes [rows][k/2], scales [rows][k/16], rowScale [rows].
// The packed nibble order is the weights': within 16 values, byte j holds value j (low) and
// value j + 8 (high), so the tensor-core product pairs every activation with its weight.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t ts_fp4_fp32_to_e4m3(float x)
{
    if (!(x > 0.0f))
        return 0;
    return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float ts_fp4_nvfp4_error(const float* v, float invRow, float invScale, float scale)
{
    // Squared error of the 16 values coded at this scale (dequantized at 2 * scale, the
    // E2M1 value times the UE4M3 value).
    float err = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; i += 4)
    {
        const __nv_fp4x4_e2m1 q(make_float4(v[i] * invRow * invScale, v[i + 1] * invRow * invScale,
                                            v[i + 2] * invRow * invScale, v[i + 3] * invRow * invScale));
        const float4 d = static_cast<float4>(q);
        const float e0 = fabsf(v[i] * invRow) - fabsf(d.x) * 2.0f * scale;
        const float e1 = fabsf(v[i + 1] * invRow) - fabsf(d.y) * 2.0f * scale;
        const float e2 = fabsf(v[i + 2] * invRow) - fabsf(d.z) * 2.0f * scale;
        const float e3 = fabsf(v[i + 3] * invRow) - fabsf(d.w) * 2.0f * scale;
        err = fmaf(e0, e0, err);
        err = fmaf(e1, e1, err);
        err = fmaf(e2, e2, err);
        err = fmaf(e3, e3, err);
    }
    return err;
}

__device__ __forceinline__ uint32_t ts_fp4_fp4x4(float a, float b, float c, float d)
{
    const __nv_fp4x4_e2m1 q(make_float4(a, b, c, d));
    return (uint32_t)(*reinterpret_cast<const uint16_t*>(&q));
}

// The 16 values of sub-block sb, coded at the scale the five-code search finds best.
__device__ __forceinline__ void ts_fp4_quantize_sub16(const float* v, float invRow, uint8_t* codes, uint8_t* scales, long long row,
    int k, int sb)
{
    float amaxSub = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        amaxSub = fmaxf(amaxSub, fabsf(v[i] * invRow));
    const int first = ts_fp4_fp32_to_e4m3(amaxSub / 6.0f);
    int code = first;
    float scale = ts_fp4_ue4m3(code);
    float best = ts_fp4_nvfp4_error(v, invRow, scale > 0.0f ? 0.5f / scale : 0.0f, scale);
    const int offsets[4] = { -1, 1, -2, 2 };
#pragma unroll
    for (int t = 0; t < 4; ++t)
    {
        const int c = first + offsets[t];
        if (c < 0 || c > 0x7e)
            continue;
        const float sc = ts_fp4_ue4m3(c);
        const float err = ts_fp4_nvfp4_error(v, invRow, sc > 0.0f ? 0.5f / sc : 0.0f, sc);
        if (err < best)
        {
            best = err;
            code = c;
            scale = sc;
        }
    }
    const float s = invRow * (scale > 0.0f ? 0.5f / scale : 0.0f);
    const uint32_t q0 = ts_fp4_fp4x4(v[0] * s, v[8] * s, v[1] * s, v[9] * s)
        | (ts_fp4_fp4x4(v[2] * s, v[10] * s, v[3] * s, v[11] * s) << 16);
    const uint32_t q1 = ts_fp4_fp4x4(v[4] * s, v[12] * s, v[5] * s, v[13] * s)
        | (ts_fp4_fp4x4(v[6] * s, v[14] * s, v[7] * s, v[15] * s) << 16);
    *(uint2*)(codes + row * (k / 2) + sb * 8) = make_uint2(q0, q1);
    scales[row * (k / 16) + sb] = (uint8_t)code;
}

__device__ __forceinline__ float ts_fp4_block_reduce(float v, float* partial, bool isMax)
{
    v = isMax ? ts_fp4_warp_max(v) : ts_fp4_warp_sum(v);
    if ((threadIdx.x & 31) == 0)
        partial[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float t = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.0f;
        t = isMax ? ts_fp4_warp_max(t) : ts_fp4_warp_sum(t);
        if (threadIdx.x == 0)
            partial[0] = t;
    }
    __syncthreads();
    const float r = partial[0];
    __syncthreads();
    return r;
}

// The planes and row scale of a row already in shared memory (act, k floats) whose largest magnitude is amax (this
// thread's part of it).
__device__ __forceinline__ void ts_fp4_quantize_staged(const float* act, float amax, float* partial, uint8_t* __restrict__ codes,
    uint8_t* __restrict__ scales, float* __restrict__ rowScale, long long row, int k)
{
    amax = ts_fp4_block_reduce(amax, partial, true);
    const float rs = amax / (6.0f * 448.0f);
    const float invRow = rs > 0.0f ? 1.0f / rs : 0.0f;
    if (threadIdx.x == 0)
        rowScale[row] = rs;
    for (int sb = threadIdx.x; sb < k / 16; sb += blockDim.x)
    {
        float v[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4)
        {
            const float4 t = *(const float4*)(act + sb * 16 + i);
            v[i] = t.x; v[i + 1] = t.y; v[i + 2] = t.z; v[i + 3] = t.w;
        }
        ts_fp4_quantize_sub16(v, invRow, codes, scales, row, k, sb);
    }
}

// The row reaches shared memory once, through one of two prologues, and is quantized from there:
//   MODE 0: x itself [rows][k];
//   MODE 1: silu(gate) * up of a fused gate|up row [rows][2k] (the down projection's input).
// One block per row, k floats of dynamic shared memory.
template <int MODE>
__device__ __forceinline__ void ts_fp4_quantize_row(
    const float* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales, float* __restrict__ rowScale, int k)
{
    extern __shared__ float act[];
    __shared__ float partial[32];
    const long long row = blockIdx.x;
    float amax = 0.0f;
    // Four float4 groups per thread in flight before any is used (one dependent round trip
    // per group otherwise); the maximum is exact in any order.
    const float* xr = x + row * (MODE == 1 ? 2LL * k : k);
    const int step = blockDim.x * 4;
    for (int base = threadIdx.x * 4; base < k; base += 4 * step)
    {
        float4 v[4], u[4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int i = base + j * step;
            if (i < k)
            {
                v[j] = *(const float4*)(xr + i);
                if (MODE == 1)
                    u[j] = *(const float4*)(xr + k + i);
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int i = base + j * step;
            if (i >= k)
                break;
            float4 t = v[j];
            if (MODE == 1)
                t = make_float4(ts_fp4_silu(t.x) * u[j].x, ts_fp4_silu(t.y) * u[j].y, ts_fp4_silu(t.z) * u[j].z,
                                ts_fp4_silu(t.w) * u[j].w);
            *(float4*)(act + i) = t;
            amax = fmaxf(fmaxf(amax, fabsf(t.x)), fmaxf(fmaxf(fabsf(t.y), fabsf(t.z)), fabsf(t.w)));
        }
    }
    ts_fp4_quantize_staged(act, amax, partial, codes, scales, rowScale, row, k);
}

extern "C" __global__ void __launch_bounds__(512) ts_fp4_quantize_nvfp4_f32(
    const float* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
    float* __restrict__ rowScale, int k)
{
    ts_fp4_quantize_row<0>(x, codes, scales, rowScale, k);
}

// The NVFP4 planes of silu(gate) * up for fused gate|up rows [rows][2k] (gate first): a SwiGLU feed-forward's down
// projection input, never stored as F32.
extern "C" __global__ void __launch_bounds__(512) ts_fp4_quantize_swiglu_nvfp4_f32(
    const float* __restrict__ gu, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
    float* __restrict__ rowScale, int k)
{
    ts_fp4_quantize_row<1>(gu, codes, scales, rowScale, k);
}

// ---------------------------------------------------------------------------
// A residual add, the RMSNorm after it and the next product's NVFP4 quantization of the normed row in one pass (the
// direct-CUDA Qwen3.5 engine's add_rmsnorm_nvfp4): v = a + b (b null: v = a), n = v * rsqrt(mean(v^2) + eps) * w; the sum
// v, the normed row n and n's planes and row scale are each written where their pointer is not null. The per-thread
// sums, reductions and expressions of ts_graph_add_rms_norm_f32 at 512 threads, so v and n are that kernel's to the bit
// (n again when a is its sum and b null), and the planes are ts_fp4_quantize_nvfp4_f32's of n. PER > 0 (k = 512 PER)
// keeps a thread's values in registers with every load issued up front; 0 walks any k. One block of 512 threads per
// row, k floats of dynamic shared memory.
// ---------------------------------------------------------------------------
template <int PER>
__device__ __forceinline__ void ts_fp4_add_rmsnorm_nvfp4(const float* __restrict__ a, const float* __restrict__ b,
    const float* __restrict__ w, float eps, float* __restrict__ sum, float* __restrict__ normed, uint8_t* __restrict__ codes,
    uint8_t* __restrict__ scales, float* __restrict__ rowScale, int k)
{
    extern __shared__ float act[];
    __shared__ float partial[32];
    constexpr int T = 512;
    const long long row = blockIdx.x;
    const int tid = threadIdx.x;
    const float* ar = a + row * k;
    float ss = 0.0f, amax = 0.0f;
    if constexpr (PER > 0)
    {
        float v[PER], wv[PER];
#pragma unroll
        for (int p = 0; p < PER; ++p)
            v[p] = b != nullptr ? ar[tid + p * T] + b[row * k + tid + p * T] : ar[tid + p * T];
#pragma unroll
        for (int p = 0; p < PER; ++p)
            wv[p] = w[tid + p * T];
#pragma unroll
        for (int p = 0; p < PER; ++p)
        {
            if (sum != nullptr)
                sum[row * k + tid + p * T] = v[p];
            ss += v[p] * v[p];
        }
        ss = ts_fp4_block_reduce(ss, partial, false);
        const float scale = rsqrtf(ss / (float)k + eps);
#pragma unroll
        for (int p = 0; p < PER; ++p)
        {
            const float n = v[p] * scale * wv[p];
            if (normed != nullptr)
                normed[row * k + tid + p * T] = n;
            act[tid + p * T] = n;
            amax = fmaxf(amax, fabsf(n));
        }
    }
    else
    {
        for (int i = tid; i < k; i += T)
        {
            const float v = b != nullptr ? ar[i] + b[row * k + i] : ar[i];
            if (sum != nullptr)
                sum[row * k + i] = v;
            act[i] = v;
            ss += v * v;
        }
        ss = ts_fp4_block_reduce(ss, partial, false);
        const float scale = rsqrtf(ss / (float)k + eps);
        for (int i = tid; i < k; i += T)
        {
            const float n = act[i] * scale * w[i];
            if (normed != nullptr)
                normed[row * k + i] = n;
            act[i] = n;
            amax = fmaxf(amax, fabsf(n));
        }
    }
    if (codes != nullptr)
        ts_fp4_quantize_staged(act, amax, partial, codes, scales, rowScale, row, k);
}

extern "C" __global__ void __launch_bounds__(512) ts_fp4_add_rmsnorm_nvfp4_f32(const float* __restrict__ a,
    const float* __restrict__ b, const float* __restrict__ w, float eps, float* __restrict__ sum, float* __restrict__ normed,
    uint8_t* __restrict__ codes, uint8_t* __restrict__ scales, float* __restrict__ rowScale, int k)
{
    switch (k % 512 == 0 ? k / 512 : 0)
    {
        case 4: ts_fp4_add_rmsnorm_nvfp4<4>(a, b, w, eps, sum, normed, codes, scales, rowScale, k); break;
        case 5: ts_fp4_add_rmsnorm_nvfp4<5>(a, b, w, eps, sum, normed, codes, scales, rowScale, k); break;
        case 6: ts_fp4_add_rmsnorm_nvfp4<6>(a, b, w, eps, sum, normed, codes, scales, rowScale, k); break;
        case 8: ts_fp4_add_rmsnorm_nvfp4<8>(a, b, w, eps, sum, normed, codes, scales, rowScale, k); break;
        case 10: ts_fp4_add_rmsnorm_nvfp4<10>(a, b, w, eps, sum, normed, codes, scales, rowScale, k); break;
        case 12: ts_fp4_add_rmsnorm_nvfp4<12>(a, b, w, eps, sum, normed, codes, scales, rowScale, k); break;
        case 16: ts_fp4_add_rmsnorm_nvfp4<16>(a, b, w, eps, sum, normed, codes, scales, rowScale, k); break;
        default: ts_fp4_add_rmsnorm_nvfp4<0>(a, b, w, eps, sum, normed, codes, scales, rowScale, k); break;
    }
}

#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define TS_FP4_MMA 1
#endif

/// out[0] = TS_FP4_VERSION, out[1] = the __CUDA_ARCH__ the image was compiled for (CudaPrefillModule's probe). One thread.
extern "C" __global__ void ts_fp4_caps(int* out)
{
    out[0] = TS_FP4_VERSION;
#ifdef __CUDA_ARCH__
    out[1] = __CUDA_ARCH__;
#else
    out[1] = 0;
#endif
}

/// out[0] = 1 when the image the driver loaded carries the FP4 tensor-core bodies (an sm_120a / sm_121a image; the
/// plain compute_12x PTX, or other-architecture code JIT-compiled for a CC 12 device, traps in every GEMM). A launch
/// rather than a symbol lookup, since a fatbin holds both kinds of image. One thread.
extern "C" __global__ void ts_fp4_mma_caps(int* out)
{
#ifdef TS_FP4_MMA
    out[0] = 1;
#else
    out[0] = 0;
#endif
}

#define TS_FP4_BM 128
#define TS_FP4_BK 256
#define TS_FP4_CROW (TS_FP4_BK / 2 + 16)
#define TS_FP4_SROW (TS_FP4_BK / 16)

__device__ __forceinline__ uint32_t ts_fp4_smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// cp.async and ldmatrix need sm_80. Below it (the compute_75 fallback, or sm_75 SASS in a
// release fatbin) these helpers trap like the MMA ones, so the module still builds; ts_fp4_caps
// reports 0 there and CudaFp4Kernels never launches a GEMM.
__device__ __forceinline__ void ts_fp4_cp16(void* dst, const void* src, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(ts_fp4_smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp4_cp_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#else
    __trap();
#endif
}

template <int N>
__device__ __forceinline__ void ts_fp4_cp_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp4_ldsm_x4(uint32_t (&r)[4], const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ts_fp4_smem_u32(p)));
#else
    r[0] = r[1] = r[2] = r[3] = 0;
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp4_mma_fp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                               uint32_t sa, uint32_t sb)
{
#ifdef TS_FP4_MMA
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#else
    __trap();
#endif
}

template <int BN, int WM, int WN, int STAGES, int BK = TS_FP4_BK>
struct TsFp4GemmSmem
{
    static constexpr int CROW = BK / 2 + 16, SROW = BK / 16;
    uint8_t ac[STAGES][TS_FP4_BM * CROW];
    uint8_t as[STAGES][TS_FP4_BM * SROW];
    uint8_t bc[STAGES][BN * CROW];
    uint8_t bs[STAGES][BN * SROW];
};

__device__ __forceinline__ void ts_fp4_cp8(void* dst, const void* src, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :: "r"(ts_fp4_smem_u32(dst)), "l"(src), "r"(valid ? 8 : 0));
#else
    __trap();
#endif
}

template <int BN, int WM, int WN, int STAGES, int BK = TS_FP4_BK, bool COLS_FAST = false>
__device__ __forceinline__ void ts_fp4_gemm_w4a4(
    const uint8_t* __restrict__ wCodes, const uint8_t* __restrict__ wScales, int outDim, int k,
    const uint8_t* __restrict__ xCodes, const uint8_t* __restrict__ xScales, const float* __restrict__ rowScale,
    int m, float alpha, float* __restrict__ y, float* __restrict__ partial, int* __restrict__ counters)
{
    static_assert(WM * WN == 8, "eight warps");
    constexpr int WROWS = TS_FP4_BM / WM, WCOLS = BN / WN;
    constexpr int MT = WROWS / 16, NT = WCOLS / 8;
    extern __shared__ __align__(16) uint8_t smemRaw[];
    using Smem = TsFp4GemmSmem<BN, WM, WN, STAGES, BK>;
    constexpr int CROW = Smem::CROW, SROW = Smem::SROW, CHUNKS = BK / 32;
    auto& sm = *reinterpret_cast<Smem*>(smemRaw);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WM, wn = warp / WM;
    // COLS_FAST: activation tiles vary fastest across the grid, so the CTAs in flight share
    // weight tiles in L2 (otherwise a weight matrix larger than L2 streams from memory once per
    // activation tile): +8% at 2048 rows, -2% at 128 or fewer.
    const int row0 = (COLS_FAST ? blockIdx.y : blockIdx.x) * TS_FP4_BM, col0 = (COLS_FAST ? blockIdx.x : blockIdx.y) * BN;
    const int splits = gridDim.z;
    const int stagesTotal = k / BK;
    const int perSplit = (stagesTotal + splits - 1) / splits;
    const int sBegin = blockIdx.z * perSplit;
    const int sEnd = min(stagesTotal, sBegin + perSplit);
    const int nStages = max(0, sEnd - sBegin);
    const long long wRowBytes = k / 2, wScaleBytes = k / 16;

    auto load = [&](int slot, int stage) {
        const long long kc = (long long)stage * (BK / 2), ks = (long long)stage * SROW;
        for (int i = tid; i < TS_FP4_BM * CHUNKS; i += 256)
        {
            const int r = i / CHUNKS, c = i % CHUNKS;
            const int gr = row0 + r;
            const bool ok = gr < outDim;
            ts_fp4_cp16(sm.ac[slot] + r * CROW + c * 16, wCodes + (ok ? gr : 0) * wRowBytes + kc + c * 16, ok);
        }
        for (int r = tid; r < TS_FP4_BM; r += 256)
        {
            const int gr = row0 + r;
            const bool ok = gr < outDim;
            if (SROW == 16)
                ts_fp4_cp16(sm.as[slot] + r * SROW, wScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
            else
                ts_fp4_cp8(sm.as[slot] + r * SROW, wScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
        }
        for (int i = tid; i < BN * CHUNKS; i += 256)
        {
            const int r = i / CHUNKS, c = i % CHUNKS;
            const int gr = col0 + r;
            const bool ok = gr < m;
            ts_fp4_cp16(sm.bc[slot] + r * CROW + c * 16, xCodes + (ok ? gr : 0) * wRowBytes + kc + c * 16, ok);
        }
        for (int r = tid; r < BN; r += 256)
        {
            const int gr = col0 + r;
            const bool ok = gr < m;
            if (SROW == 16)
                ts_fp4_cp16(sm.bs[slot] + r * SROW, xScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
            else
                ts_fp4_cp8(sm.bs[slot] + r * SROW, xScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < nStages)
            load(s, sBegin + s);
        ts_fp4_cp_commit();
    }
    const int g = lane >> 2, q = lane & 3;
    for (int it = 0; it < nStages; ++it)
    {
        ts_fp4_cp_wait<STAGES - 2>();
        __syncthreads();
        // The slot this fills was consumed last iteration, which every warp has finished.
        const int next = it + STAGES - 1;
        if (next < nStages)
            load(next % STAGES, sBegin + next);
        ts_fp4_cp_commit();
        const int slot = it % STAGES;
        const uint8_t* ac = sm.ac[slot];
        const uint8_t* as = sm.as[slot];
        const uint8_t* bc = sm.bc[slot];
        const uint8_t* bs = sm.bs[slot];
#pragma unroll
        for (int kstep = 0; kstep < BK / 64; ++kstep)
        {
            uint32_t a[MT][4], sa[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i)
            {
                const int r = wm * WROWS + i * 16;
                ts_fp4_ldsm_x4(a[i], ac + (r + (lane & 15)) * CROW + kstep * 32 + (lane >> 4) * 16);
                sa[i] = *(const uint32_t*)(as + (r + g + (lane & 1) * 8) * SROW + kstep * 4);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                const int c = wn * WCOLS + j * 8 + g;
                const uint8_t* bp = bc + c * CROW + kstep * 32 + q * 4;
                const uint32_t b0 = *(const uint32_t*)bp, b1 = *(const uint32_t*)(bp + 16);
                const uint32_t sb = *(const uint32_t*)(bs + c * SROW + kstep * 4);
#pragma unroll
                for (int i = 0; i < MT; ++i)
                    ts_fp4_mma_fp4(acc[i][j], a[i], b0, b1, sa[i], sb);
            }
        }
    }
    ts_fp4_cp_wait<0>();

    // Epilogue: D rows are weight rows (g, g + 8), columns activation rows (2q, 2q + 1).
    auto element = [&](int i, int j, int e, int& r, int& c) {
        r = row0 + wm * WROWS + i * 16 + g + (e >= 2 ? 8 : 0);
        c = col0 + wn * WCOLS + j * 8 + q * 2 + (e & 1);
    };
    if (splits == 1)
    {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    int r, c;
                    element(i, j, e, r, c);
                    if (r < outDim && c < m)
                        y[(long long)c * outDim + r] = acc[i][j][e] * rowScale[c] * alpha;
                }
        return;
    }
    float* mine = partial + (long long)blockIdx.z * m * outDim;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                int r, c;
                element(i, j, e, r, c);
                if (r < outDim && c < m)
                    mine[(long long)c * outDim + r] = acc[i][j][e];
            }
    __threadfence();
    __syncthreads();
    __shared__ int last;
    const int tile = blockIdx.y * gridDim.x + blockIdx.x;
    if (tid == 0)
        last = atomicAdd(counters + tile, 1) == splits - 1;
    __syncthreads();
    if (!last)
        return;
    __threadfence();
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                int r, c;
                element(i, j, e, r, c);
                if (r >= outDim || c >= m)
                    continue;
                float sum = 0.0f;
                for (int z = 0; z < splits; ++z)
                    sum += __ldcg(partial + ((long long)z * m + c) * outDim + r);
                y[(long long)c * outDim + r] = sum * rowScale[c] * alpha;
            }
    if (tid == 0)
        counters[tile] = 0;
}
#define TS_FP4_GEMM(NAME, BN, WM, WN, STAGES, COLS_FAST) \
    extern "C" __global__ __launch_bounds__(256) void NAME( \
        const uint8_t* __restrict__ wCodes, const uint8_t* __restrict__ wScales, int outDim, int k, \
        const uint8_t* __restrict__ xCodes, const uint8_t* __restrict__ xScales, const float* __restrict__ rowScale, \
        int m, float alpha, float* __restrict__ y, float* __restrict__ partial, int* __restrict__ counters) \
    { ts_fp4_gemm_w4a4<BN, WM, WN, STAGES, TS_FP4_BK, COLS_FAST>(wCodes, wScales, outDim, k, xCodes, xScales, rowScale, m, \
        alpha, y, partial, counters); }

// grid (weight tiles, activation tiles, splits); the 128-row kernel takes (activation tiles, weight tiles, splits).
TS_FP4_GEMM(ts_fp4_gemm_w4a4_32, 32, 8, 1, 3, false)
TS_FP4_GEMM(ts_fp4_gemm_w4a4_64, 64, 4, 2, 3, false)
TS_FP4_GEMM(ts_fp4_gemm_w4a4_128, 128, 4, 2, 2, true)

/// Dynamic shared memory of ts_fp4_gemm_w4a4_<bn>, by bn (32, 64, 128).
extern "C" __global__ void ts_fp4_gemm_smem_bytes(int* out)
{
    out[0] = (int)sizeof(TsFp4GemmSmem<32, 8, 1, 3>);
    out[1] = (int)sizeof(TsFp4GemmSmem<64, 4, 2, 3>);
    out[2] = (int)sizeof(TsFp4GemmSmem<128, 4, 2, 2>);
}

// ---------------------------------------------------------------------------
// Prompt rows on the FP4 tensor cores, TMA-fed and warp-specialized (sm_120a), the structure of the nvfp4_w4a4_tma
// kernel Ignis vendors from NInfer (Apache-2.0), on the planes: activation and weight codes [rows][k/2] and UE4M3
// scales [rows][k/16], read as they are.
//   * One producer thread issues every copy: per 128-value K tile, the BM x 64-byte activation codes and 128 x 64-byte
//     weight codes (64-byte swizzle) and, on even tiles, one 16-byte scale box per row for both operands (TMA's
//     narrowest box; it covers two K tiles).
//   * Eight consumer warps (4 over tokens x 2 over weight rows) run the block-scaled FP4 MMA with activations as the A
//     operand.
// Accumulation is the K64 MMA sequence of ts_fp4_gemm_w4a4 in the same order (bit-identical accumulators):
// y = acc * rowScale * alpha. The activation tile runs fastest across the grid: the CTAs in flight share a weight tile.
// Partial weight tiles (outDim not a multiple of 128) are zero-filled by TMA and not stored. grid (activation tiles,
// 128-row weight tiles), block 256 consumers + 32 (BM 128) or 128 (BM 256) producers; outDim % 16 == 0, k % 128 == 0.
// ---------------------------------------------------------------------------
struct alignas(64) TsFp4TmaMap
{
    unsigned long long words[16];
};

struct alignas(64) TsFp4TmaMaps
{
    TsFp4TmaMap xCodes, xScales, wCodes, wScales;
};

template <int BM, int STAGES>
struct TsFp4TmaSmem
{
    alignas(1024) uint8_t xc[STAGES][BM * 64];
    alignas(1024) uint8_t wc[STAGES][128 * 64];
    alignas(128) uint8_t xs[2][BM * 16];
    alignas(128) uint8_t ws[2][128 * 16];
    alignas(8) unsigned long long full[STAGES];
    alignas(8) unsigned long long empty[STAGES];
};

#if defined(TS_FP4_MMA)
__device__ __forceinline__ void ts_fp4_mbar_init(unsigned long long* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(ts_fp4_smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void ts_fp4_mbar_wait(unsigned long long* bar, uint32_t parity)
{
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "ts_fp4_wait:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@!done bra ts_fp4_wait;\n"
                 "}\n"
                 :: "r"(ts_fp4_smem_u32(bar)), "r"(parity), "r"(0x989680) : "memory");
}

__device__ __forceinline__ void ts_fp4_mbar_arrive(unsigned long long* bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(ts_fp4_smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void ts_fp4_mbar_expect(unsigned long long* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(ts_fp4_smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void ts_fp4_tma_2d(void* dst, const TsFp4TmaMap* map, int c0, int c1, unsigned long long* bar)
{
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(ts_fp4_smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(ts_fp4_smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void ts_fp4_ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void ts_fp4_ldsm_x4a(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// 64-byte TMA swizzle: address bits [4:5] ^= bits [7:8] (CUTLASS Swizzle<2,4,3>).
__device__ __forceinline__ uint32_t ts_fp4_sw64(uint32_t addr) { return addr ^ (((addr >> 7) & 3u) << 4); }

#endif

// SWIGLU: the fused gate|up projection of a SwiGLU feed-forward (outDim = 2 ff weight rows, the ff gate rows first) with
// the activation in the epilogue: y [m][ff] = silu(gate) * up, gate and up each the value the plain kernel stores, so the
// down projection's quantizer reads the activation it would otherwise compute from the [m][2 ff] product (half the
// bytes written, half read). A CTA takes 64 gate rows and the same columns' 64 up rows (two 64-row TMA boxes per
// operand, smem rows 0-63 and 64-127); a consumer warp's n8 tiles 0-3 are gate rows wn * 32 + 8j, tiles 4-7 the same
// columns' up rows, so a column's gate and up meet in one thread. grid (activation tiles, ff / 64); ff % 64 == 0.
template <int BM, int STAGES, bool SWIGLU = false>
__device__ __forceinline__ void ts_fp4_gemm_w4a4_tma(const TsFp4TmaMaps& maps, int outDim, int k,
    const float* __restrict__ rowScale, int m, float alpha, float* __restrict__ y)
{
    static_assert(BM == 128 || BM == 256, "token tile");
    static_assert(STAGES == 2 || STAGES == 3, "the two scale slots are rewritten at tile 2p + 4, after tile 2p + 1 is consumed");
#if defined(TS_FP4_MMA)
    constexpr int PRODUCERS = BM == 256 ? 128 : 32, WARP_M = BM / 4, MT = WARP_M / 16, NT = 8;
    extern __shared__ __align__(1024) uint8_t tmaSmem[];
    auto& sm = *reinterpret_cast<TsFp4TmaSmem<BM, STAGES>*>(tmaSmem);
    const int token0 = blockIdx.x * BM, row0 = blockIdx.y * (SWIGLU ? 64 : 128), kTiles = k / 128;

    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            ts_fp4_mbar_init(&sm.full[s], 1);
            ts_fp4_mbar_init(&sm.empty[s], 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (threadIdx.x < PRODUCERS)
    {
        if constexpr (PRODUCERS == 128)
            asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (threadIdx.x == 0)
        {
            constexpr uint32_t CODES = BM * 64 + 128 * 64, SCALES = BM * 16 + 128 * 16;
#pragma unroll 1
            for (int kt = 0; kt < kTiles; ++kt)
            {
                const int s = kt % STAGES;
                ts_fp4_mbar_wait(&sm.empty[s], 1u ^ ((kt / STAGES) & 1u));
                const bool scales = (kt & 1) == 0;
                ts_fp4_mbar_expect(&sm.full[s], scales ? CODES + SCALES : CODES);
                ts_fp4_tma_2d(sm.xc[s], &maps.xCodes, kt * 64, token0, &sm.full[s]);
                ts_fp4_tma_2d(sm.wc[s], &maps.wCodes, kt * 64, row0, &sm.full[s]);
                if constexpr (SWIGLU)
                    ts_fp4_tma_2d(sm.wc[s] + 64 * 64, &maps.wCodes, kt * 64, outDim / 2 + row0, &sm.full[s]);
                if (scales)
                {
                    ts_fp4_tma_2d(sm.xs[(kt >> 1) & 1], &maps.xScales, (kt >> 1) * 16, token0, &sm.full[s]);
                    ts_fp4_tma_2d(sm.ws[(kt >> 1) & 1], &maps.wScales, (kt >> 1) * 16, row0, &sm.full[s]);
                    if constexpr (SWIGLU)
                        ts_fp4_tma_2d(sm.ws[(kt >> 1) & 1] + 64 * 16, &maps.wScales, (kt >> 1) * 16, outDim / 2 + row0, &sm.full[s]);
                }
            }
        }
        return;
    }
    if constexpr (PRODUCERS == 128)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");

    const int ct = threadIdx.x - PRODUCERS, lane = ct & 31, warp = ct >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int g = lane >> 2, q = lane & 3;
    // The smem weight row of n8 tile j's first row, past the warp's own (wn * 64, or wn * 32 for the gate|up split).
    auto tileRow = [](int j) { return SWIGLU ? (j >> 2) * 64 + (j & 3) * 8 : j * 8; };
    const int warpRow = SWIGLU ? wn * 32 : wn * 64;
    // ldmatrix row addresses: A x4 = rows 0-15 at byte 0 then 16; B x2 = rows 0-7 at bytes 0, 16.
    const int aRow = wm * WARP_M + (lane & 15), aByte = (lane >> 4) * 16;
    const int bRow = warpRow + (lane & 7), bByte = ((lane >> 3) & 1) * 16;
    const int saRow = wm * WARP_M + g + (lane & 1) * 8, sbRow = warpRow + g;

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

#pragma unroll 1
    for (int kt = 0; kt < kTiles; ++kt)
    {
        const int s = kt % STAGES;
        ts_fp4_mbar_wait(&sm.full[s], (kt / STAGES) & 1u);
        const uint32_t xc = ts_fp4_smem_u32(sm.xc[s]), wc = ts_fp4_smem_u32(sm.wc[s]);
        const uint8_t* xs = sm.xs[(kt >> 1) & 1] + (kt & 1) * 8;
        const uint8_t* ws = sm.ws[(kt >> 1) & 1] + (kt & 1) * 8;
#pragma unroll
        for (int k64 = 0; k64 < 2; ++k64)
        {
            uint32_t a[MT][4], sa[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i)
            {
                ts_fp4_ldsm_x4a(a[i], ts_fp4_sw64(xc + (aRow + i * 16) * 64 + k64 * 32 + aByte));
                sa[i] = *(const uint32_t*)(xs + (saRow + i * 16) * 16 + k64 * 4);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                uint32_t b0, b1;
                ts_fp4_ldsm_x2(b0, b1, ts_fp4_sw64(wc + (bRow + tileRow(j)) * 64 + k64 * 32 + bByte));
                const uint32_t sb = *(const uint32_t*)(ws + (sbRow + tileRow(j)) * 16 + k64 * 4);
#pragma unroll
                for (int i = 0; i < MT; ++i)
                    ts_fp4_mma_fp4(acc[i][j], a[i], b0, b1, sa[i], sb);
            }
        }
        if (lane == 0)
            ts_fp4_mbar_arrive(&sm.empty[s]);
    }

    // D rows are tokens (g, g + 8), columns weight rows (2q, 2q + 1).
    const int tokBase = token0 + wm * WARP_M + g, rowBase = row0 + warpRow + 2 * q;
    if constexpr (SWIGLU)
    {
        const int ff = outDim / 2;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int c = tokBase + i * 16 + h * 8;
                if (c >= m)
                    continue;
                const float rs = rowScale[c];
#pragma unroll
                for (int j = 0; j < NT / 2; ++j)
                {
                    float out[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        const float gate = acc[i][j][h * 2 + e] * rs * alpha, up = acc[i][j + NT / 2][h * 2 + e] * rs * alpha;
                        out[e] = ts_fp4_silu(gate) * up;
                    }
                    *(float2*)(y + (long long)c * ff + rowBase + j * 8) = make_float2(out[0], out[1]);
                }
            }
        return;
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const int c = tokBase + i * 16 + h * 8;
            if (c >= m)
                continue;
            const float rs = rowScale[c];
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                const int r = rowBase + j * 8;
                if (r >= outDim)
                    continue;
                const float v0 = acc[i][j][h * 2] * rs * alpha, v1 = acc[i][j][h * 2 + 1] * rs * alpha;
                *(float2*)(y + (long long)c * outDim + r) = make_float2(v0, v1);
            }
        }
#else
    __trap();
#endif
}

// grid (activation tiles, 128-row weight tiles), block 256 consumers + 32 (BM 128) or 128
// (BM 256) producers; outDim % 16 == 0 and k % 128 == 0 (CudaFp4Kernels.TakesTma checks).
// _swiglu: the gate|up projection with the SwiGLU epilogue (weight maps in 64-row boxes).
#define TS_FP4_GEMM_TMA(NAME, BM, STAGES, SWIGLU) \
    extern "C" __global__ __launch_bounds__(256 + (BM == 256 ? 128 : 32), 1) void NAME( \
        const __grid_constant__ TsFp4TmaMaps maps, int outDim, int k, const float* __restrict__ rowScale, int m, \
        float alpha, float* __restrict__ y) \
    { ts_fp4_gemm_w4a4_tma<BM, STAGES, SWIGLU>(maps, outDim, k, rowScale, m, alpha, y); }
TS_FP4_GEMM_TMA(ts_fp4_gemm_w4a4_tma256, 256, 3, false)
TS_FP4_GEMM_TMA(ts_fp4_gemm_w4a4_tma128, 128, 3, false)
TS_FP4_GEMM_TMA(ts_fp4_gemm_w4a4_tma256_swiglu, 256, 3, true)
TS_FP4_GEMM_TMA(ts_fp4_gemm_w4a4_tma128_swiglu, 128, 3, true)

/// Dynamic shared memory of the TMA GEMMs: [BM 256, BM 128] (three stages each).
extern "C" __global__ void ts_fp4_gemm_tma_smem_bytes(int* out)
{
    out[0] = (int)sizeof(TsFp4TmaSmem<256, 3>);
    out[1] = (int)sizeof(TsFp4TmaSmem<128, 3>);
}

// Rows `idx` of a planar NVFP4 matrix [rows][k] as F32: codes at `planes`, scales after rows * k / 2 code bytes.
// One block per gathered row; indices are int32 (or float when idxF32).
extern "C" __global__ void ts_fp4_get_rows_f32(const uint8_t* __restrict__ planes, int rows, int k, const void* __restrict__ idx,
    int idxF32, float* __restrict__ out)
{
    const long long r = idxF32 ? (long long)((const float*)idx)[blockIdx.x] : (long long)((const int*)idx)[blockIdx.x];
    const uint8_t* codes = planes + r * (k / 2);
    const uint8_t* scales = planes + (long long)rows * (k / 2) + r * (k / 16);
    float* o = out + (long long)blockIdx.x * k;
    for (int c = threadIdx.x; c < k; c += blockDim.x)
    {
        const uint8_t b = codes[c / 16 * 8 + c % 8];
        const int nibble = c % 16 < 8 ? b & 15 : b >> 4;
        const float mag = (nibble & 7) < 4 ? 0.5f * (nibble & 7) : (nibble & 7) == 4 ? 2.0f : (nibble & 7) == 5 ? 3.0f
            : (nibble & 7) == 6 ? 4.0f : 6.0f;
        o[c] = (nibble & 8 ? -mag : mag) * 2.0f * ts_fp4_ue4m3(scales[c / 16]);
    }
}

// ---------------------------------------------------------------------------
// ts_fp4_gemm_w4a4 for a fused gate|up matrix [2 ff][k] (the ff gate rows first) with the SwiGLU in its epilogue: a CTA's 128
// weight rows are 64 gate rows and their 64 up rows (rows g0 .. g0 + 63 and ff + g0 ..), so it holds both halves of its
// outputs and writes y[m][ff] = silu(gate) * up, where the plain product wrote [m][2 ff] for a SwiGLU pass to read back.
// Every gate and up value is the plain product's for its row (the same tiles of K, the same MMAs, the same split-K partials
// summed in split order and scaled as it scales them) and silu is ts_fp4_silu, so y is what ts_fp4_quantize_swiglu_nvfp4_f32
// stages from the plain product's rows bit for bit. The main loop is ts_fp4_gemm_w4a4's (copied), only the weight rows each
// tile row reads differ. ff % 64 == 0. Grid (ff / 64, activation tiles, splits): the plain product's tile count, so its
// split-K counters serve.
// ---------------------------------------------------------------------------
template <int BN, int WM, int WN, int STAGES, int BK = TS_FP4_BK>
__device__ __forceinline__ void ts_fp4_gemm_w4a4_swiglu(
    const uint8_t* __restrict__ wCodes, const uint8_t* __restrict__ wScales, int outDim, int k,
    const uint8_t* __restrict__ xCodes, const uint8_t* __restrict__ xScales, const float* __restrict__ rowScale,
    int m, float alpha, float* __restrict__ y, float* __restrict__ partial, int* __restrict__ counters)
{
    static_assert(WM * WN == 8, "eight warps");
    constexpr int WROWS = TS_FP4_BM / WM, WCOLS = BN / WN;
    constexpr int MT = WROWS / 16, NT = WCOLS / 8, HALF = TS_FP4_BM / 2;
    extern __shared__ __align__(16) uint8_t smemRaw[];
    using Smem = TsFp4GemmSmem<BN, WM, WN, STAGES, BK>;
    constexpr int CROW = Smem::CROW, SROW = Smem::SROW, CHUNKS = BK / 32;
    auto& sm = *reinterpret_cast<Smem*>(smemRaw);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WM, wn = warp / WM;
    const int ff = outDim / 2, g0 = blockIdx.x * HALF, col0 = blockIdx.y * BN;
    const int splits = gridDim.z;
    const int stagesTotal = k / BK;
    const int perSplit = (stagesTotal + splits - 1) / splits;
    const int sBegin = blockIdx.z * perSplit;
    const int sEnd = min(stagesTotal, sBegin + perSplit);
    const int nStages = max(0, sEnd - sBegin);
    const long long wRowBytes = k / 2, wScaleBytes = k / 16;
    // Tile row r: gate row g0 + r, or up row ff + g0 + r - 64; valid while its gate row is.
    auto weightRow = [&](int r) { return r < HALF ? g0 + r : ff + g0 + (r - HALF); };
    auto validRow = [&](int r) { return (r < HALF ? g0 + r : g0 + (r - HALF)) < ff; };

    auto load = [&](int slot, int stage) {
        const long long kc = (long long)stage * (BK / 2), ks = (long long)stage * SROW;
        for (int i = tid; i < TS_FP4_BM * CHUNKS; i += 256)
        {
            const int r = i / CHUNKS, c = i % CHUNKS;
            const bool ok = validRow(r);
            const int gr = weightRow(r);
            ts_fp4_cp16(sm.ac[slot] + r * CROW + c * 16, wCodes + (ok ? gr : 0) * wRowBytes + kc + c * 16, ok);
        }
        for (int r = tid; r < TS_FP4_BM; r += 256)
        {
            const bool ok = validRow(r);
            const int gr = weightRow(r);
            if (SROW == 16)
                ts_fp4_cp16(sm.as[slot] + r * SROW, wScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
            else
                ts_fp4_cp8(sm.as[slot] + r * SROW, wScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
        }
        for (int i = tid; i < BN * CHUNKS; i += 256)
        {
            const int r = i / CHUNKS, c = i % CHUNKS;
            const int gr = col0 + r;
            const bool ok = gr < m;
            ts_fp4_cp16(sm.bc[slot] + r * CROW + c * 16, xCodes + (ok ? gr : 0) * wRowBytes + kc + c * 16, ok);
        }
        for (int r = tid; r < BN; r += 256)
        {
            const int gr = col0 + r;
            const bool ok = gr < m;
            if (SROW == 16)
                ts_fp4_cp16(sm.bs[slot] + r * SROW, xScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
            else
                ts_fp4_cp8(sm.bs[slot] + r * SROW, xScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
        }
    };

    float acc[MT][NT][4];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < nStages)
            load(s, sBegin + s);
        ts_fp4_cp_commit();
    }
    const int g = lane >> 2, q = lane & 3;
    for (int it = 0; it < nStages; ++it)
    {
        ts_fp4_cp_wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < nStages)
            load(next % STAGES, sBegin + next);
        ts_fp4_cp_commit();
        const int slot = it % STAGES;
        const uint8_t* ac = sm.ac[slot];
        const uint8_t* as = sm.as[slot];
        const uint8_t* bc = sm.bc[slot];
        const uint8_t* bs = sm.bs[slot];
#pragma unroll
        for (int kstep = 0; kstep < BK / 64; ++kstep)
        {
            uint32_t a[MT][4], sa[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i)
            {
                const int r = wm * WROWS + i * 16;
                ts_fp4_ldsm_x4(a[i], ac + (r + (lane & 15)) * CROW + kstep * 32 + (lane >> 4) * 16);
                sa[i] = *(const uint32_t*)(as + (r + g + (lane & 1) * 8) * SROW + kstep * 4);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                const int c = wn * WCOLS + j * 8 + g;
                const uint8_t* bp = bc + c * CROW + kstep * 32 + q * 4;
                const uint32_t b0 = *(const uint32_t*)bp, b1 = *(const uint32_t*)(bp + 16);
                const uint32_t sb = *(const uint32_t*)(bs + c * SROW + kstep * 4);
#pragma unroll
                for (int i = 0; i < MT; ++i)
                    ts_fp4_mma_fp4(acc[i][j], a[i], b0, b1, sa[i], sb);
            }
        }
    }
    ts_fp4_cp_wait<0>();
    __syncthreads();

    // Tile element (i, j, e): tile row lr (weight row weightRow(lr)), tile column lc (activation row col0 + lc).
    auto element = [&](int i, int j, int e, int& lr, int& lc) {
        lr = wm * WROWS + i * 16 + g + (e >= 2 ? 8 : 0);
        lc = wn * WCOLS + j * 8 + q * 2 + (e & 1);
    };
    float* ys = reinterpret_cast<float*>(smemRaw);   // [128 tile rows][BN]: the products' values, the pipeline done with
    if (splits == 1)
    {
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    int lr, lc;
                    element(i, j, e, lr, lc);
                    const int c = col0 + lc;
                    if (validRow(lr) && c < m)
                        ys[lr * BN + lc] = acc[i][j][e] * rowScale[c] * alpha;
                }
    }
    else
    {
        float* mine = partial + (long long)blockIdx.z * m * outDim;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    int lr, lc;
                    element(i, j, e, lr, lc);
                    const int c = col0 + lc;
                    if (validRow(lr) && c < m)
                        mine[(long long)c * outDim + weightRow(lr)] = acc[i][j][e];
                }
        __threadfence();
        __syncthreads();
        __shared__ int last;
        const int tile = blockIdx.y * gridDim.x + blockIdx.x;
        if (tid == 0)
            last = atomicAdd(counters + tile, 1) == splits - 1;
        __syncthreads();
        if (!last)
            return;
        __threadfence();
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    int lr, lc;
                    element(i, j, e, lr, lc);
                    const int c = col0 + lc;
                    if (!validRow(lr) || c >= m)
                        continue;
                    const int r = weightRow(lr);
                    float sum = 0.0f;
                    for (int z = 0; z < splits; ++z)
                        sum += __ldcg(partial + ((long long)z * m + c) * outDim + r);
                    ys[lr * BN + lc] = sum * rowScale[c] * alpha;
                }
        if (tid == 0)
            counters[tile] = 0;
    }
    __syncthreads();
    for (int idx = tid; idx < HALF * BN; idx += 256)
    {
        const int lc = idx / HALF, lr = idx % HALF, c = col0 + lc;
        if (g0 + lr < ff && c < m)
            y[(long long)c * ff + g0 + lr] = ts_fp4_silu(ys[lr * BN + lc]) * ys[(lr + HALF) * BN + lc];
    }
}

#define TS_FP4_GEMM_SWIGLU(NAME, BN, WM, WN, STAGES) \
    extern "C" __global__ __launch_bounds__(256) void NAME( \
        const uint8_t* __restrict__ wCodes, const uint8_t* __restrict__ wScales, int outDim, int k, \
        const uint8_t* __restrict__ xCodes, const uint8_t* __restrict__ xScales, const float* __restrict__ rowScale, \
        int m, float alpha, float* __restrict__ y, float* __restrict__ partial, int* __restrict__ counters) \
    { ts_fp4_gemm_w4a4_swiglu<BN, WM, WN, STAGES, TS_FP4_BK>(wCodes, wScales, outDim, k, xCodes, xScales, rowScale, m, \
        alpha, y, partial, counters); }

// grid (ff / 64, activation tiles, splits).
TS_FP4_GEMM_SWIGLU(ts_fp4_gemm_w4a4_32_swiglu, 32, 8, 1, 3)
TS_FP4_GEMM_SWIGLU(ts_fp4_gemm_w4a4_64_swiglu, 64, 4, 2, 3)

// ---------------------------------------------------------------------------
// W4A8: prompt rows of q8_1 activations (ts_fp4_quantize_q81_f32: int8 [rows][k], one F32 scale per 32 values) times a
// planar NVFP4 matrix on the int8 tensor cores, for a model that keeps 8-bit activations on its prompt rows
// (--nvfp4-prompt-activations q8). The quantities are the q8 matvec's: E2M1 codes as int8 at twice their value
// (ts_fp4_e2m1x8), each 16-value block's int32 sum (one mma.m16n8k16, a block's K) scaled by its halved UE4M3 weight scale
// and its 32-value activation scale into F32. Only the order of the F32 sums differs from the matvec's.
// CTA: 64 activation rows x 64 weight rows, 4 warps of 32 x 32, K in steps of 64 (four weight blocks, two activation
// blocks); a register-staged double buffer keeps the next step's loads in flight over the current step's products.
// k a multiple of 64 (CudaFp4Kernels.TakesWidth); rows and outDim any.
// ---------------------------------------------------------------------------
#define TS_W4A8_BM 64
#define TS_W4A8_BN 64
#define TS_W4A8_BK 64
// Shared rows padded by 16 bytes: the eight rows of a fragment load then fall in distinct banks.
#define TS_W4A8_LD (TS_W4A8_BK + 16)

__device__ __forceinline__ void ts_w4a8_mma(int (&d)[4], uint32_t a0, uint32_t a1, uint32_t b0)
{
    d[0] = d[1] = d[2] = d[3] = 0;
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%0,%1,%2,%3};\n"
        : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
        : "r"(a0), "r"(a1), "r"(b0));
#else
    (void) a0; (void) a1; (void) b0;
    __trap();
#endif
}

extern "C" __global__ void __launch_bounds__(128) ts_fp4_gemm_w4a8(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int outDim, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xd, int rows, float alpha, float* __restrict__ y)
{
    __shared__ __align__(16) int8_t As[2][TS_W4A8_BM][TS_W4A8_LD];
    __shared__ __align__(16) int8_t Bs[2][TS_W4A8_BN][TS_W4A8_LD];
    __shared__ float Asc[2][TS_W4A8_BM][2];   // activation scales: two 32-value blocks a step
    __shared__ float Bsc[2][TS_W4A8_BN][4];   // halved UE4M3 weight scales: four 16-value blocks a step

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int g = lane >> 2, tg = lane & 3;
    const int wm = (warp >> 1) * 32, wn = (warp & 1) * 32;
    const long long row0 = (long long)blockIdx.y * TS_W4A8_BM, col0 = (long long)blockIdx.x * TS_W4A8_BN;
    const int steps = k / TS_W4A8_BK;
    const int scalesPerRow = k >> 5;

    // Each thread stages: two 16-byte runs of activations (row tid / 2, bytes (tid & 1) * 32 ..), one 16-byte run of
    // codes (weight row tid / 2, the step's 32 code bytes' half tid & 1), one activation scale and one weight-scale word.
    const int ar = tid >> 1, ah = tid & 1;
    const bool aIn = row0 + ar < rows, bIn = col0 + ar < outDim;
    const int8_t* aSrc = xq + (aIn ? (row0 + ar) : 0) * (long long)k + ah * 32;
    const uint8_t* bSrc = codes + (bIn ? (col0 + ar) : 0) * (long long)(k / 2) + ah * 16;
    const uint8_t* sSrc = scales + (bIn ? (col0 + ar) : 0) * (long long)(k / 16);
    const float* dSrc = xd + (aIn ? (row0 + ar) : 0) * (long long)scalesPerRow + ah;

    int4 ra0, ra1, rb;
    float rd;
    uint32_t rs;
    auto fetch = [&](int s)
    {
        if (aIn)
        {
            const int4* p = (const int4*)(aSrc + (long long)s * TS_W4A8_BK);
            ra0 = p[0]; ra1 = p[1];
            rd = dSrc[s * 2];
        }
        else
        {
            ra0 = ra1 = make_int4(0, 0, 0, 0);
            rd = 0.0f;
        }
        if (bIn)
        {
            rb = *(const int4*)(bSrc + (long long)s * (TS_W4A8_BK / 2));
            rs = *(const uint32_t*)(sSrc + (long long)s * 4);
        }
        else
        {
            rb = make_int4(0, 0, 0, 0);
            rs = 0;
        }
    };
    auto stage = [&](int buf)
    {
        *(int4*)&As[buf][ar][ah * 32] = ra0;
        *(int4*)&As[buf][ar][ah * 32 + 16] = ra1;
        Asc[buf][ar][ah] = rd;
        // 32 codes: two 16-value blocks of this half of the step, each to int8 in value order. Within a block byte j
        // holds value j (low nibble) and value j + 8 (high): ts_fp4_e2m1x8 of bytes 0-3 gives values 0-3 | 8-11, of
        // bytes 4-7 values 4-7 | 12-15.
        const uint32_t w[4] = { (uint32_t)rb.x, (uint32_t)rb.y, (uint32_t)rb.z, (uint32_t)rb.w };
#pragma unroll
        for (int blk = 0; blk < 2; ++blk)
        {
            const int2 v0 = ts_fp4_e2m1x8((int)w[2 * blk]), v1 = ts_fp4_e2m1x8((int)w[2 * blk + 1]);
            *(int4*)&Bs[buf][ar][(ah * 2 + blk) * 16] = make_int4(v0.x, v1.x, v0.y, v1.y);
        }
        if (ah == 0)
        {
#pragma unroll
            for (int q = 0; q < 4; ++q)
                Bsc[buf][ar][q] = ts_fp4_ue4m3(rs >> (8 * q));
        }
    };

    float acc[2][4][4];
#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                acc[mi][ni][e] = 0.0f;

    if (steps > 0)
    {
        fetch(0);
        stage(0);
    }
    __syncthreads();
    for (int s = 0; s < steps; ++s)
    {
        const int buf = s & 1;
        if (s + 1 < steps)
            fetch(s + 1);
#pragma unroll
        for (int kb = 0; kb < 4; ++kb)
        {
            const int kk = kb * 16;
            uint32_t a[2][2], b[4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
            {
                a[mi][0] = *(const uint32_t*)&As[buf][wm + mi * 16 + g][kk + tg * 4];
                a[mi][1] = *(const uint32_t*)&As[buf][wm + mi * 16 + g + 8][kk + tg * 4];
            }
#pragma unroll
            for (int ni = 0; ni < 4; ++ni)
                b[ni] = *(const uint32_t*)&Bs[buf][wn + ni * 8 + g][kk + tg * 4];
#pragma unroll
            for (int mi = 0; mi < 2; ++mi)
            {
                const float d0 = Asc[buf][wm + mi * 16 + g][kb >> 1], d1 = Asc[buf][wm + mi * 16 + g + 8][kb >> 1];
#pragma unroll
                for (int ni = 0; ni < 4; ++ni)
                {
                    int c[4];
                    ts_w4a8_mma(c, a[mi][0], a[mi][1], b[ni]);
                    const float w0 = Bsc[buf][wn + ni * 8 + tg * 2][kb], w1 = Bsc[buf][wn + ni * 8 + tg * 2 + 1][kb];
                    acc[mi][ni][0] += (float)c[0] * (w0 * d0);
                    acc[mi][ni][1] += (float)c[1] * (w1 * d0);
                    acc[mi][ni][2] += (float)c[2] * (w0 * d1);
                    acc[mi][ni][3] += (float)c[3] * (w1 * d1);
                }
            }
        }
        if (s + 1 < steps)
            stage(buf ^ 1);
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < 2; ++mi)
#pragma unroll
        for (int ni = 0; ni < 4; ++ni)
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const long long r = row0 + wm + mi * 16 + g + (e >> 1) * 8;
                const long long c = col0 + wn + ni * 8 + tg * 2 + (e & 1);
                if (r < rows && c < outDim)
                    y[r * outDim + c] = alpha * acc[mi][ni][e];
            }
}


// ---------------------------------------------------------------------------
// W4A8 on the F16 tensor cores, TMA-fed and warp-specialized like ts_fp4_gemm_w4a4_tma (sm_120a). The q8_1 activations
// arrive as F16 integers (ts_fp4_quantize_q81_f16) with their F32 scales; each weight becomes q * UE4M3 in F16, exact
// (at most 6 significant bits, within F16's range), so one m16n8k16 per 16-value block gives the block's scaled sum
// s * sum(a q) exactly in F32, the two blocks of a 32-value activation scale accumulate in the MMA, and one FFMA applies
// that scale: the q8 matvec's quantities, with only the order of the F32 sums its own.
//   * A producer warpgroup (one thread issues) copies per 128-value K tile: the activations in two 64-value boxes of
//     BM x 128 bytes (128-byte swizzle), their scales (16 bytes a row), the weight codes (128 x 64 bytes, 64-byte
//     swizzle) and, on even tiles, the weight scales (16 bytes a row: two tiles).
//   * Eight consumer warps (2 over tokens x 4 over weight rows, 64 x 32 each) build each B fragment pair from one 16-bit
//     read of two code bytes (values 2tg, 2tg+1 and 2tg+8, 2tg+9) and the block's scale.
// grid (128-token tiles, 128-row weight tiles), block 128 producers + 256 consumers; k % 256 == 0, outDim even.
// ---------------------------------------------------------------------------
template <int STAGES>
struct TsW4A8Smem
{
    alignas(1024) uint8_t xh[STAGES][2][128 * 128];
    alignas(1024) uint8_t wc[STAGES][128 * 64];
    alignas(128) uint8_t xs[STAGES][128 * 16];
    alignas(128) uint8_t ws[2][128 * 16];
    alignas(8) unsigned long long full[STAGES];
    alignas(8) unsigned long long empty[STAGES];
};

#if defined(TS_FP4_MMA)
__device__ __forceinline__ void ts_w4a8_mma_f16(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
    asm("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t ts_w4a8_hmul2(uint32_t a, uint32_t s)
{
    __half2 h = __hmul2(*reinterpret_cast<__half2*>(&a), *reinterpret_cast<__half2*>(&s));
    return *reinterpret_cast<uint32_t*>(&h);
}

// Two equal UE4M3 bytes (NaN codes already zero) -> F16x2.
__device__ __forceinline__ uint32_t ts_w4a8_scale2(uint32_t two)
{
    uint32_t r;
    asm("{ .reg .b16 t; cvt.u16.u32 t, %1; cvt.rn.f16x2.e4m3x2 %0, t; }\n" : "=r"(r) : "r"(two));
    return r;
}

// B fragment pair of a 16-value block from its code bytes 2tg, 2tg+1 (x: values 2tg, 2tg+8, 2tg+1, 2tg+9 as nibbles):
// b0 = (2tg, 2tg+1), b1 = (2tg+8, 2tg+9), each q * scale in F16 (the E2M1 table as F16 high bytes, low bytes zero).
__device__ __forceinline__ void ts_w4a8_bfrag(uint32_t x, uint32_t sc2, uint32_t& b0, uint32_t& b1)
{
    const uint32_t t0 = 0x3E3C3800u, t1 = 0x46444240u, t2 = 0xBEBCB880u, t3 = 0xC6C4C2C0u;
    const uint32_t mag = __byte_perm(t0, t1, x), neg = __byte_perm(t2, t3, x);
    const uint32_t r = __byte_perm(mag, neg, 0x3210u | ((x & 0x8888u) >> 1));
    b0 = ts_w4a8_hmul2(__byte_perm(r, 0u, 0x2404), sc2);
    b1 = ts_w4a8_hmul2(__byte_perm(r, 0u, 0x3414), sc2);
}

// 128-byte TMA swizzle: address bits [4:6] ^= bits [7:9] (CUTLASS Swizzle<3,4,3>).
__device__ __forceinline__ uint32_t ts_fp4_sw128(uint32_t addr) { return addr ^ (((addr >> 7) & 7u) << 4); }
#endif

template <int STAGES>
__device__ __forceinline__ void ts_fp4_gemm_w4a8_tma_body(const TsFp4TmaMaps& maps, int outDim, int k, int m, float alpha,
    float* __restrict__ y)
{
    static_assert(STAGES == 2 || STAGES == 3, "the two weight-scale slots are rewritten at tile 2p + 4, after tile 2p + 1 is consumed");
#if defined(TS_FP4_MMA)
    extern __shared__ __align__(1024) uint8_t w4a8Smem[];
    auto& sm = *reinterpret_cast<TsW4A8Smem<STAGES>*>(w4a8Smem);
    const int token0 = blockIdx.x * 128, row0 = blockIdx.y * 128, kTiles = k / 128;
    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            ts_fp4_mbar_init(&sm.full[s], 1);
            ts_fp4_mbar_init(&sm.empty[s], 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
    }
    __syncthreads();

    if (threadIdx.x < 128)
    {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;" ::: "memory");
        if (threadIdx.x == 0)
        {
            constexpr uint32_t TILE = 2 * 128 * 128 + 128 * 16 + 128 * 64, WS = 128 * 16;
#pragma unroll 1
            for (int kt = 0; kt < kTiles; ++kt)
            {
                const int s = kt % STAGES;
                ts_fp4_mbar_wait(&sm.empty[s], 1u ^ ((kt / STAGES) & 1u));
                const bool scales = (kt & 1) == 0;
                ts_fp4_mbar_expect(&sm.full[s], scales ? TILE + WS : TILE);
                ts_fp4_tma_2d(sm.xh[s][0], &maps.xCodes, kt * 256, token0, &sm.full[s]);
                ts_fp4_tma_2d(sm.xh[s][1], &maps.xCodes, kt * 256 + 128, token0, &sm.full[s]);
                ts_fp4_tma_2d(sm.xs[s], &maps.xScales, kt * 16, token0, &sm.full[s]);
                ts_fp4_tma_2d(sm.wc[s], &maps.wCodes, kt * 64, row0, &sm.full[s]);
                if (scales)
                    ts_fp4_tma_2d(sm.ws[(kt >> 1) & 1], &maps.wScales, (kt >> 1) * 16, row0, &sm.full[s]);
            }
        }
        return;
    }
    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");

    const int ct = threadIdx.x - 128, lane = ct & 31, warp = ct >> 5;
    const int g = lane >> 2, tg = lane & 3;
    const int wm = (warp >> 2) * 64, wn = (warp & 3) * 32;
    constexpr int MI = 4, NI = 4;
    float acc[MI][NI][4];
#pragma unroll
    for (int i = 0; i < MI; ++i)
#pragma unroll
        for (int j = 0; j < NI; ++j)
            acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.0f;
    // ldmatrix x4 of a 16 x 16 F16 tile: rows 0-15 at the block's first 8 values, then its last 8.
    const int aRow = lane & 15, aChunk = lane >> 4;

#pragma unroll 1
    for (int kt = 0; kt < kTiles; ++kt)
    {
        const int s = kt % STAGES;
        ts_fp4_mbar_wait(&sm.full[s], (kt / STAGES) & 1u);
        const uint32_t wc = ts_fp4_smem_u32(sm.wc[s]);
        const float* xs = (const float*)sm.xs[s];
        const uint8_t* ws = sm.ws[(kt >> 1) & 1] + (kt & 1) * 8;
#pragma unroll 1
        for (int q = 0; q < 2; ++q)   // the tile's two 64-value halves (one activation box each)
        {
            const uint32_t xh = ts_fp4_smem_u32(sm.xh[s][q]);
            uint32_t sc[NI][4];
#pragma unroll
            for (int ni = 0; ni < NI; ++ni)
            {
                // The half's four block scales of this thread's weight row; NaN codes (0x7F, 0xFF) read as zero.
                uint32_t w = *(const uint32_t*)(ws + (wn + ni * 8 + g) * 16 + q * 4) & 0x7F7F7F7Fu;
                w &= ~__vcmpeq4(w, 0x7F7F7F7Fu);
#pragma unroll
                for (int kb = 0; kb < 4; ++kb)
                    sc[ni][kb] = ts_w4a8_scale2(__byte_perm(w, 0u, 0x4400 | (kb * 0x11)));
            }
#pragma unroll
            for (int pp = 0; pp < 2; ++pp)   // the half's two activation blocks
            {
                const int p = q * 2 + pp;
                uint32_t b[NI][2][2];
#pragma unroll
                for (int ni = 0; ni < NI; ++ni)
#pragma unroll
                    for (int kk = 0; kk < 2; ++kk)
                    {
                        uint32_t x;
                        asm volatile("ld.shared.u16 %0, [%1];"
                            : "=r"(x) : "r"(ts_fp4_sw64(wc + (wn + ni * 8 + g) * 64 + (2 * p + kk) * 8 + tg * 2)));
                        ts_w4a8_bfrag(x, sc[ni][2 * pp + kk], b[ni][kk][0], b[ni][kk][1]);
                    }
#pragma unroll
                for (int mi = 0; mi < MI; ++mi)
                {
                    uint32_t a0[4], a1[4];
                    ts_fp4_ldsm_x4a(a0, ts_fp4_sw128(xh + (wm + mi * 16 + aRow) * 128 + (pp * 4 + aChunk) * 16));
                    ts_fp4_ldsm_x4a(a1, ts_fp4_sw128(xh + (wm + mi * 16 + aRow) * 128 + (pp * 4 + 2 + aChunk) * 16));
                    const float d0 = xs[(wm + mi * 16 + g) * 4 + p], d1 = xs[(wm + mi * 16 + g + 8) * 4 + p];
#pragma unroll
                    for (int ni = 0; ni < NI; ++ni)
                    {
                        float t[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
                        ts_w4a8_mma_f16(t, a0, b[ni][0][0], b[ni][0][1]);
                        ts_w4a8_mma_f16(t, a1, b[ni][1][0], b[ni][1][1]);
                        acc[mi][ni][0] = fmaf(t[0], d0, acc[mi][ni][0]);
                        acc[mi][ni][1] = fmaf(t[1], d0, acc[mi][ni][1]);
                        acc[mi][ni][2] = fmaf(t[2], d1, acc[mi][ni][2]);
                        acc[mi][ni][3] = fmaf(t[3], d1, acc[mi][ni][3]);
                    }
                }
            }
        }
        __syncwarp();
        if (lane == 0)
            ts_fp4_mbar_arrive(&sm.empty[s]);
    }
    // D rows are tokens (g, g + 8), columns weight rows (2tg, 2tg + 1).
#pragma unroll
    for (int mi = 0; mi < MI; ++mi)
#pragma unroll
        for (int h = 0; h < 2; ++h)
        {
            const int c = token0 + wm + mi * 16 + g + h * 8;
            if (c >= m)
                continue;
#pragma unroll
            for (int ni = 0; ni < NI; ++ni)
            {
                const int r = row0 + wn + ni * 8 + tg * 2;
                if (r >= outDim)
                    continue;
                *(float2*)(y + (long long)c * outDim + r) = make_float2(alpha * acc[mi][ni][h * 2], alpha * acc[mi][ni][h * 2 + 1]);
            }
        }
#else
    __trap();
#endif
}

extern "C" __global__ __launch_bounds__(384, 1) void ts_fp4_gemm_w4a8_tma(const __grid_constant__ TsFp4TmaMaps maps, int outDim,
    int k, int m, float alpha, float* __restrict__ y)
{
    ts_fp4_gemm_w4a8_tma_body<2>(maps, outDim, k, m, alpha, y);
}

/// Dynamic shared memory of ts_fp4_gemm_w4a8_tma. One thread.
extern "C" __global__ void ts_fp4_gemm_w4a8_tma_smem_bytes(int* out)
{
    out[0] = (int)sizeof(TsW4A8Smem<2>);
}
