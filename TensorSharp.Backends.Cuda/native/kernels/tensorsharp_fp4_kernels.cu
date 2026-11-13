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
//   * ts_fp4_quantize_q81_f32 + ts_fp4_gemv_nvfp4_<n>: up to 8 rows (decode, verify windows) against q8 activations.
//   * ts_fp4_quantize_nvfp4_f32: F32 rows -> NVFP4 activation planes and row scales (ggml-cuda's quantize_mmq_nvfp4).
//   * ts_fp4_gemm_w4a4_<bn>: 9+ rows on the FP4 tensor cores through a cp.async ring, split over K for few rows.
//   * ts_fp4_gemm_w4a4_tma<bm>: prompt rows, TMA-fed and warp-specialized (the same products in the same order).
//   * ts_fp4_get_rows_f32: rows of a planar matrix as F32 (an NVFP4 table's lookup).
//
// The kernels are the direct-CUDA Qwen3.5 engine's (bit-identical arithmetic). The tensor-core bodies exist only in an
// sm_120a / sm_121a image (CommittedBaseline false: no compute_75 baseline); ts_fp4_mma_caps reports whether the
// loaded image has them, and the backend keeps NVFP4 weights as ggml blocks on its other kernels where it does not.
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

#define TS_FP4_VERSION 2

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
// NVFP4 GEMV for N <= 8 activation rows. A warp owns R consecutive weight rows; lane l reads
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

/// Weight rows one warp of ts_fp4_gemv_nvfp4_<n> owns (the launcher sizes its grid by it).
extern "C" __global__ void ts_fp4_gemv_rows_per_warp(int* out)
{
#pragma unroll
    for (int n = 1; n <= 8; ++n)
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

// The row reaches shared memory once, through one of three prologues, and is quantized from there:
//   MODE 0: x itself [rows][k];
//   MODE 1: silu(gate) * up of a fused gate|up row [rows][2k] (the down projection's input);
//   MODE 2: x += delta (written back; delta may be null), then rmsnorm(x) * w with eps - the
//           pre-norm and the next projection's quantization in one pass, the normed row never
//           stored (the same per-thread sums and reductions as ts_fp4_add_rmsnorm_f32 at 512
//           threads, so the values match it exactly).
// One block per row, k floats of dynamic shared memory.
template <int MODE>
__device__ __forceinline__ void ts_fp4_quantize_row(
    float* __restrict__ x, const float* __restrict__ aux, const float* __restrict__ w, float eps,
    uint8_t* __restrict__ codes, uint8_t* __restrict__ scales, float* __restrict__ rowScale, int k)
{
    extern __shared__ float act[];
    __shared__ float partial[32];
    const long long row = blockIdx.x;
    float amax = 0.0f;
    if (MODE == 2)
    {
        float* xr = x + row * k;
        float ss = 0.0f;
        for (int i = threadIdx.x; i < k; i += blockDim.x)
        {
            float v = xr[i];
            if (aux != nullptr)
            {
                v += aux[row * k + i];
                xr[i] = v;
            }
            act[i] = v;
            ss += v * v;
        }
        ss = ts_fp4_block_reduce(ss, partial, false);
        const float scale = rsqrtf(ss / (float)k + eps);
        for (int i = threadIdx.x; i < k; i += blockDim.x)
        {
            const float v = act[i] * scale * w[i];
            act[i] = v;
            amax = fmaxf(amax, fabsf(v));
        }
    }
    else
    {
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
    }
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

extern "C" __global__ void __launch_bounds__(512) ts_fp4_quantize_nvfp4_f32(
    const float* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
    float* __restrict__ rowScale, int k)
{
    ts_fp4_quantize_row<0>(const_cast<float*>(x), nullptr, nullptr, 0.0f, codes, scales, rowScale, k);
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
// reports 0 there and Q35Kernels never launches a GEMM.
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

template <int BM, int STAGES>
__device__ __forceinline__ void ts_fp4_gemm_w4a4_tma(const TsFp4TmaMaps& maps, int outDim, int k,
    const float* __restrict__ rowScale, int m, float alpha, float* __restrict__ y)
{
    static_assert(BM == 128 || BM == 256, "token tile");
    static_assert(STAGES == 2 || STAGES == 3, "the two scale slots are rewritten at tile 2p + 4, after tile 2p + 1 is consumed");
#if defined(TS_FP4_MMA)
    constexpr int PRODUCERS = BM == 256 ? 128 : 32, WARP_M = BM / 4, MT = WARP_M / 16, NT = 8;
    extern __shared__ __align__(1024) uint8_t tmaSmem[];
    auto& sm = *reinterpret_cast<TsFp4TmaSmem<BM, STAGES>*>(tmaSmem);
    const int token0 = blockIdx.x * BM, row0 = blockIdx.y * 128, kTiles = k / 128;

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
                if (scales)
                {
                    ts_fp4_tma_2d(sm.xs[(kt >> 1) & 1], &maps.xScales, (kt >> 1) * 16, token0, &sm.full[s]);
                    ts_fp4_tma_2d(sm.ws[(kt >> 1) & 1], &maps.wScales, (kt >> 1) * 16, row0, &sm.full[s]);
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
    // ldmatrix row addresses: A x4 = rows 0-15 at byte 0 then 16; B x2 = rows 0-7 at bytes 0, 16.
    const int aRow = wm * WARP_M + (lane & 15), aByte = (lane >> 4) * 16;
    const int bRow = wn * 64 + (lane & 7), bByte = ((lane >> 3) & 1) * 16;
    const int saRow = wm * WARP_M + g + (lane & 1) * 8, sbRow = wn * 64 + g;

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
                ts_fp4_ldsm_x2(b0, b1, ts_fp4_sw64(wc + (bRow + j * 8) * 64 + k64 * 32 + bByte));
                const uint32_t sb = *(const uint32_t*)(ws + (sbRow + j * 8) * 16 + k64 * 4);
#pragma unroll
                for (int i = 0; i < MT; ++i)
                    ts_fp4_mma_fp4(acc[i][j], a[i], b0, b1, sa[i], sb);
            }
        }
        if (lane == 0)
            ts_fp4_mbar_arrive(&sm.empty[s]);
    }

    // D rows are tokens (g, g + 8), columns weight rows (2q, 2q + 1).
    const int tokBase = token0 + wm * WARP_M + g, rowBase = row0 + wn * 64 + 2 * q;
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
#define TS_FP4_GEMM_TMA(NAME, BM, STAGES) \
    extern "C" __global__ __launch_bounds__(256 + (BM == 256 ? 128 : 32), 1) void NAME( \
        const __grid_constant__ TsFp4TmaMaps maps, int outDim, int k, const float* __restrict__ rowScale, int m, \
        float alpha, float* __restrict__ y) \
    { ts_fp4_gemm_w4a4_tma<BM, STAGES>(maps, outDim, k, rowScale, m, alpha, y); }
TS_FP4_GEMM_TMA(ts_fp4_gemm_w4a4_tma256, 256, 3)
TS_FP4_GEMM_TMA(ts_fp4_gemm_w4a4_tma128, 128, 3)

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
