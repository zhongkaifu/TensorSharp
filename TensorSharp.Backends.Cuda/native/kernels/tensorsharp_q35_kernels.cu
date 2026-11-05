// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Qwen3.5 / Qwen3.8 dense (qwen35) kernels for the direct-CUDA backend.
//
// The checkpoints this engine serves store every projection as NVFP4 (ggml block_nvfp4: 64
// values in 36 bytes - four UE4M3 scales, one per 16 values, then 32 bytes of E2M1 codes;
// within a 16-value sub-block the low nibbles are values 0-7 and the high nibbles 8-15) with
// a per-tensor F32 scale applied to the product.
//
// The engine keeps each matrix as two planes instead of 36-byte blocks: codes [rows][k/2]
// (ggml's nibble order within every 16 values kept) and UE4M3 scales [rows][k/16]. Same bytes,
// but every 64-value code run is 16-byte aligned, which the cp.async and vector loads need.
//
//   * ts_q35_repack_nvfp4: ggml block_nvfp4 rows -> the two planes (once, at load).
//   * ts_q35_quantize_q81_f32: F32 activation rows -> int8 rows plus one F32 scale per 32
//     values (q8_1 without the interleaving, so a 16-value slice is one aligned 16-byte load).
//   * ts_q35_gemv_nvfp4_<n>: y[n rows] = alpha * x[n rows] W^T for n <= 8 activation rows, dp4a
//     against the quantized activations. This is the integer path ggml-cuda's mmvq takes for
//     NVFP4 (same E2M1 table, same UE4M3 decode, same q8_1 rounding).
//   * ts_q35_quantize_nvfp4_f32: F32 activation rows -> NVFP4 planes plus a per-row F32 scale,
//     ggml-cuda's quantize_mmq_nvfp4 numerics (row scale amax / (6 * 448), the sub-block scale
//     searched over five UE4M3 codes for the least squared error).
//   * ts_q35_gemm_w4a4_<bn>: y = alpha * rowScale * (x W^T) for more activation rows, on the
//     block-scaled FP4 tensor cores (mma kind::mxf4nvf4, sm_120a): weight codes go from shared
//     memory to the MMA by ldmatrix with no per-value decode. ggml-cuda's mmq runs the same
//     product; this one has no stream-k fixup pass (split-K merges in the tile's last CTA, in
//     a fixed order).
// ---------------------------------------------------------------------------

#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_fp4.h>
#include <stdint.h>

#define TS_Q35_NVFP4_BLOCK 64
#define TS_Q35_NVFP4_BLOCK_BYTES 36

__device__ __forceinline__ float ts_q35_warp_sum(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xffffffff, v, off, 32);
    return v;
}

__device__ __forceinline__ float ts_q35_warp_max(float v)
{
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v = fmaxf(v, __shfl_xor_sync(0xffffffff, v, off, 32));
    return v;
}

__device__ __forceinline__ float ts_q35_sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }
__device__ __forceinline__ float ts_q35_silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float ts_q35_softplus(float x) { return x > 20.0f ? x : log1pf(expf(x)); }

// dp4a is native on every target CUDA 13 compiles (compute_75 and newer).
__device__ __forceinline__ int ts_q35_dp4a(int a, int b, int c)
{
    return __dp4a(a, b, c);
}

// UE4M3 block scale as ggml-cuda decodes it: the NaN codes read as zero, and the value is
// halved because the E2M1 table below is doubled to stay integral.
__device__ __forceinline__ float ts_q35_ue4m3(uint32_t x)
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
__device__ __forceinline__ int2 ts_q35_e2m1x8(int q)
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
extern "C" __global__ void ts_q35_quantize_q81_f32(
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
    const float amax = ts_q35_warp_max(fabsf(v));
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
extern "C" __global__ void ts_q35_repack_nvfp4(
    const uint8_t* __restrict__ src, long long srcRowBytes, uint8_t* __restrict__ codes,
    uint8_t* __restrict__ scales, int rows, int k, int rowBase, int parts, int part)
{
    const int perRow = k / TS_Q35_NVFP4_BLOCK;
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * perRow)
        return;
    const long long rs = i / perRow;
    const int b = (int)(i % perRow);
    const long long row = rowBase + rs;
    const long long r = parts > 1 ? row / 8 * 8 * parts + part * 8 + row % 8 : rs;
    const uint32_t* w = (const uint32_t*)(src + rs * srcRowBytes + (long long)b * TS_Q35_NVFP4_BLOCK_BYTES);
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
__device__ __forceinline__ void ts_q35_gemv_nvfp4(
    const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int outDim, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xd, float alpha, float* __restrict__ y)
{
    const int warpsPerBlock = blockDim.x >> 5;
    const int r0 = (blockIdx.x * warpsPerBlock + (threadIdx.x >> 5)) * R;
    const int lane = threadIdx.x & 31;
    if (r0 >= outDim)
        return;
    const int nblocks = k / TS_Q35_NVFP4_BLOCK;
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
            const int kk = b * TS_Q35_NVFP4_BLOCK + s * 16;
            const int sb = kk >> 5;
            int2 v0[R], v1[R];
            float ds[R];
#pragma unroll
            for (int j = 0; j < R; ++j)
            {
                ds[j] = ts_q35_ue4m3(dw[j] >> (8 * s));
                v0[j] = ts_q35_e2m1x8((int)qs[j][2 * s]);       // values 0-3 | 8-11
                v1[j] = ts_q35_e2m1x8((int)qs[j][2 * s + 1]);   // values 4-7 | 12-15
            }
#pragma unroll
            for (int n = 0; n < N; ++n)
            {
                const int4 a = *(const int4*)(xq + (long long)n * k + kk);
                const float xs = xd[(long long)n * scalesPerRow + sb];
#pragma unroll
                for (int j = 0; j < R; ++j)
                {
                    int sumi = ts_q35_dp4a(v0[j].x, a.x, 0);
                    sumi = ts_q35_dp4a(v1[j].x, a.y, sumi);
                    sumi = ts_q35_dp4a(v0[j].y, a.z, sumi);
                    sumi = ts_q35_dp4a(v1[j].y, a.w, sumi);
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
            const float v = ts_q35_warp_sum(acc[n][j]);
            if (lane == 0 && r0 + j < outDim)
                y[(long long)n * outDim + r0 + j] = alpha * v;
        }
}

// Rows per warp by activation rows: one row keeps the most warps in flight while the weight
// stream is the only cost; more activation rows amortize their loads over more weight rows.
#define TS_Q35_GEMV(N, R) \
    extern "C" __global__ void ts_q35_gemv_nvfp4_##N( \
        const uint8_t* __restrict__ codes, const uint8_t* __restrict__ scales, int outDim, int k, \
        const int8_t* __restrict__ xq, const float* __restrict__ xd, float alpha, float* __restrict__ y) \
    { ts_q35_gemv_nvfp4<N, R>(codes, scales, outDim, k, xq, xd, alpha, y); }

#ifndef TS_Q35_GEMV_ROWS
#define TS_Q35_GEMV_ROWS(N) ((N) <= 1 ? 1 : (N) <= 3 ? 2 : 4)
#endif
TS_Q35_GEMV(1, TS_Q35_GEMV_ROWS(1))
TS_Q35_GEMV(2, TS_Q35_GEMV_ROWS(2))
TS_Q35_GEMV(3, TS_Q35_GEMV_ROWS(3))
TS_Q35_GEMV(4, TS_Q35_GEMV_ROWS(4))
TS_Q35_GEMV(5, TS_Q35_GEMV_ROWS(5))
TS_Q35_GEMV(6, TS_Q35_GEMV_ROWS(6))
TS_Q35_GEMV(7, TS_Q35_GEMV_ROWS(7))
TS_Q35_GEMV(8, TS_Q35_GEMV_ROWS(8))

/// Weight rows one warp of ts_q35_gemv_nvfp4_<n> owns (the launcher sizes its grid by it).
extern "C" __global__ void ts_q35_gemv_rows_per_warp(int* out)
{
#pragma unroll
    for (int n = 1; n <= 8; ++n)
        out[n] = TS_Q35_GEMV_ROWS(n);
}


// ---------------------------------------------------------------------------
// Activation rows -> NVFP4 planes + per-row scale, ggml-cuda's quantize_mmq_nvfp4 numerics.
// Codes [rows][k/2], scales [rows][k/16], rowScale [rows].
// The packed nibble order is the weights': within 16 values, byte j holds value j (low) and
// value j + 8 (high), so the tensor-core product pairs every activation with its weight.
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint8_t ts_q35_fp32_to_e4m3(float x)
{
    if (!(x > 0.0f))
        return 0;
    return (uint8_t)__nv_cvt_float_to_fp8(x, __NV_SATFINITE, __NV_E4M3);
}

__device__ __forceinline__ float ts_q35_nvfp4_error(const float* v, float invRow, float invScale, float scale)
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

__device__ __forceinline__ uint32_t ts_q35_fp4x4(float a, float b, float c, float d)
{
    const __nv_fp4x4_e2m1 q(make_float4(a, b, c, d));
    return (uint32_t)(*reinterpret_cast<const uint16_t*>(&q));
}

// The 16 values of sub-block sb, coded at the scale the five-code search finds best.
__device__ __forceinline__ void ts_q35_quantize_sub16(const float* v, float invRow, uint8_t* codes, uint8_t* scales, long long row,
    int k, int sb)
{
    float amaxSub = 0.0f;
#pragma unroll
    for (int i = 0; i < 16; ++i)
        amaxSub = fmaxf(amaxSub, fabsf(v[i] * invRow));
    const int first = ts_q35_fp32_to_e4m3(amaxSub / 6.0f);
    int code = first;
    float scale = ts_q35_ue4m3(code);
    float best = ts_q35_nvfp4_error(v, invRow, scale > 0.0f ? 0.5f / scale : 0.0f, scale);
    const int offsets[4] = { -1, 1, -2, 2 };
#pragma unroll
    for (int t = 0; t < 4; ++t)
    {
        const int c = first + offsets[t];
        if (c < 0 || c > 0x7e)
            continue;
        const float sc = ts_q35_ue4m3(c);
        const float err = ts_q35_nvfp4_error(v, invRow, sc > 0.0f ? 0.5f / sc : 0.0f, sc);
        if (err < best)
        {
            best = err;
            code = c;
            scale = sc;
        }
    }
    const float s = invRow * (scale > 0.0f ? 0.5f / scale : 0.0f);
    const uint32_t q0 = ts_q35_fp4x4(v[0] * s, v[8] * s, v[1] * s, v[9] * s)
        | (ts_q35_fp4x4(v[2] * s, v[10] * s, v[3] * s, v[11] * s) << 16);
    const uint32_t q1 = ts_q35_fp4x4(v[4] * s, v[12] * s, v[5] * s, v[13] * s)
        | (ts_q35_fp4x4(v[6] * s, v[14] * s, v[7] * s, v[15] * s) << 16);
    *(uint2*)(codes + row * (k / 2) + sb * 8) = make_uint2(q0, q1);
    scales[row * (k / 16) + sb] = (uint8_t)code;
}

__device__ __forceinline__ float ts_q35_block_reduce(float v, float* partial, bool isMax)
{
    v = isMax ? ts_q35_warp_max(v) : ts_q35_warp_sum(v);
    if ((threadIdx.x & 31) == 0)
        partial[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float t = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.0f;
        t = isMax ? ts_q35_warp_max(t) : ts_q35_warp_sum(t);
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
//           stored (the same per-thread sums and reductions as ts_q35_add_rmsnorm_f32 at 512
//           threads, so the values match it exactly).
// One block per row, k floats of dynamic shared memory.
template <int MODE>
__device__ __forceinline__ void ts_q35_quantize_row(
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
        ss = ts_q35_block_reduce(ss, partial, false);
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
                    t = make_float4(ts_q35_silu(t.x) * u[j].x, ts_q35_silu(t.y) * u[j].y, ts_q35_silu(t.z) * u[j].z,
                                    ts_q35_silu(t.w) * u[j].w);
                *(float4*)(act + i) = t;
                amax = fmaxf(fmaxf(amax, fabsf(t.x)), fmaxf(fmaxf(fabsf(t.y), fabsf(t.z)), fabsf(t.w)));
            }
        }
    }
    amax = ts_q35_block_reduce(amax, partial, true);
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
        ts_q35_quantize_sub16(v, invRow, codes, scales, row, k, sb);
    }
}

extern "C" __global__ void __launch_bounds__(512) ts_q35_quantize_nvfp4_f32(
    const float* __restrict__ x, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
    float* __restrict__ rowScale, int k)
{
    ts_q35_quantize_row<0>(const_cast<float*>(x), nullptr, nullptr, 0.0f, codes, scales, rowScale, k);
}

extern "C" __global__ void __launch_bounds__(512) ts_q35_quantize_swiglu_nvfp4_f32(
    const float* __restrict__ gu, uint8_t* __restrict__ codes, uint8_t* __restrict__ scales,
    float* __restrict__ rowScale, int k)
{
    ts_q35_quantize_row<1>(const_cast<float*>(gu), nullptr, nullptr, 0.0f, codes, scales, rowScale, k);
}

extern "C" __global__ void __launch_bounds__(512) ts_q35_add_rmsnorm_nvfp4_f32(
    float* __restrict__ x, const float* __restrict__ delta, const float* __restrict__ w, float eps,
    uint8_t* __restrict__ codes, uint8_t* __restrict__ scales, float* __restrict__ rowScale, int k)
{
    ts_q35_quantize_row<2>(x, delta, w, eps, codes, scales, rowScale, k);
}

// ts_q35_add_rmsnorm_nvfp4_f32 for k = 5120 at 512 threads: the row, its delta and the norm weights
// in registers with every global load issued up front, where the generic loop waits one dependent
// round trip per element (a verify of 9+ rows runs two per layer). The same per-thread sums,
// reductions, expressions and quantizer: bit-identical planes, row scales and residual.
// RTX 5090, 8-32 rows: 5.5 -> 3.5 us.
extern "C" __global__ void __launch_bounds__(512) ts_q35_add_rmsnorm_nvfp4_5120_f32(
    float* __restrict__ x, const float* __restrict__ delta, const float* __restrict__ w, float eps,
    uint8_t* __restrict__ codes, uint8_t* __restrict__ scales, float* __restrict__ rowScale, int k)
{
    constexpr int K = 5120, T = 512, PER = K / T;
    __shared__ __align__(16) float act[K];
    __shared__ float partial[32];
    const long long row = blockIdx.x;
    const int tid = threadIdx.x;
    float* xr = x + row * K;
    float v[PER], wv[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p)
    {
        v[p] = xr[tid + p * T];
        if (delta != nullptr)
            v[p] += delta[row * K + tid + p * T];
    }
#pragma unroll
    for (int p = 0; p < PER; ++p)
        wv[p] = w[tid + p * T];
    float ss = 0.0f;
#pragma unroll
    for (int p = 0; p < PER; ++p)
    {
        if (delta != nullptr)
            xr[tid + p * T] = v[p];
        ss += v[p] * v[p];
    }
    ss = ts_q35_block_reduce(ss, partial, false);
    const float scale = rsqrtf(ss / (float)K + eps);
    float amax = 0.0f;
#pragma unroll
    for (int p = 0; p < PER; ++p)
    {
        const float n = v[p] * scale * wv[p];
        act[tid + p * T] = n;
        amax = fmaxf(amax, fabsf(n));
    }
    amax = ts_q35_block_reduce(amax, partial, true);
    const float rs = amax / (6.0f * 448.0f);
    const float invRow = rs > 0.0f ? 1.0f / rs : 0.0f;
    if (tid == 0)
        rowScale[row] = rs;
    for (int sb = tid; sb < K / 16; sb += T)
    {
        float q[16];
#pragma unroll
        for (int i = 0; i < 16; i += 4)
        {
            const float4 t = *(const float4*)(act + sb * 16 + i);
            q[i] = t.x; q[i + 1] = t.y; q[i + 2] = t.z; q[i + 3] = t.w;
        }
        ts_q35_quantize_sub16(q, invRow, codes, scales, row, K, sb);
    }
}

// ---------------------------------------------------------------------------
// W4A4 GEMM on the block-scaled FP4 tensor cores.
//
// A CTA owns 128 weight rows x BN activation rows and streams K through a cp.async ring of
// 256-value stages: per stage 128 x 128 code bytes and 128 x 16 scale bytes of weights, BN x
// 144 bytes of activations. Eight warps split the tile WM x WN; each MMA step is m16n8k64
// (16 weight rows x 8 activation rows x 64 K values) with one UE4M3 scale per 16 K values on
// both sides. Code rows are padded to 144 bytes so ldmatrix and the B fragment loads hit 32
// distinct banks.
//
// Split-K (gridDim.z > 1) writes F32 partials and the tile's last CTA to finish sums them in
// split order, then resets its counter for the next launch: deterministic, and no separate
// fixup kernel.
// ---------------------------------------------------------------------------
#if defined(__CUDA_ARCH__) && (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL))
#define TS_Q35_FP4_MMA 1
#endif

/// 1 when the image the driver loaded carries the tensor-core bodies: the FP4 MMA above and the
/// mma / ldmatrix paths guarded by __CUDA_ARCH__ >= 800 (implied). Any other target - the plain
/// compute_12x PTX, or sm_8x / sm_9x code JIT-compiled for a CC 12 device - writes 0, and there
/// every GEMM would __trap(). A symbol lookup cannot tell: a fatbin holds both images.
extern "C" __global__ void ts_q35_caps(int* out)
{
#ifdef TS_Q35_FP4_MMA
    out[0] = 1;
#else
    out[0] = 0;
#endif
}

#define TS_Q35_BM 128
#define TS_Q35_BK 256
#define TS_Q35_CROW (TS_Q35_BK / 2 + 16)
#define TS_Q35_SROW (TS_Q35_BK / 16)

__device__ __forceinline__ uint32_t ts_q35_smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

// cp.async and ldmatrix need sm_80. Below it (the compute_75 fallback, or sm_75 SASS in a
// release fatbin) these helpers trap like the MMA ones, so the module still builds; ts_q35_caps
// reports 0 there and Q35Kernels never launches a GEMM.
__device__ __forceinline__ void ts_q35_cp16(void* dst, const void* src, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(ts_q35_smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_q35_cp_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#else
    __trap();
#endif
}

template <int N>
__device__ __forceinline__ void ts_q35_cp_wait()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group %0;\n" :: "n"(N));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_q35_ldsm_x4(uint32_t (&r)[4], const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ts_q35_smem_u32(p)));
#else
    r[0] = r[1] = r[2] = r[3] = 0;
    __trap();
#endif
}

__device__ __forceinline__ void ts_q35_mma_fp4(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1,
                                               uint32_t sa, uint32_t sb)
{
#ifdef TS_Q35_FP4_MMA
    asm volatile(
        "mma.sync.aligned.kind::mxf4nvf4.block_scale.scale_vec::4X.m16n8k64.row.col.f32.e2m1.e2m1.f32.ue4m3 "
        "{%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, %10, {0, 0}, %11, {0, 0};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "r"(sb));
#else
    __trap();
#endif
}

template <int BN, int WM, int WN, int STAGES, int BK = TS_Q35_BK>
struct TsQ35GemmSmem
{
    static constexpr int CROW = BK / 2 + 16, SROW = BK / 16;
    uint8_t ac[STAGES][TS_Q35_BM * CROW];
    uint8_t as[STAGES][TS_Q35_BM * SROW];
    uint8_t bc[STAGES][BN * CROW];
    uint8_t bs[STAGES][BN * SROW];
};

__device__ __forceinline__ void ts_q35_cp8(void* dst, const void* src, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.ca.shared.global [%0], [%1], 8, %2;\n"
                 :: "r"(ts_q35_smem_u32(dst)), "l"(src), "r"(valid ? 8 : 0));
#else
    __trap();
#endif
}

template <int BN, int WM, int WN, int STAGES, int BK = TS_Q35_BK, bool COLS_FAST = false, bool SWIGLU = false,
          bool ADD_RESIDUAL = false, bool SWIGLU_BF16 = false>
__device__ __forceinline__ void ts_q35_gemm_w4a4(
    const uint8_t* __restrict__ wCodes, const uint8_t* __restrict__ wScales, int outDim, int k,
    const uint8_t* __restrict__ xCodes, const uint8_t* __restrict__ xScales, const float* __restrict__ rowScale,
    int m, float alpha, float* __restrict__ y, float* __restrict__ partial, int* __restrict__ counters,
    float* __restrict__ residualX = nullptr)
{
    static_assert(WM * WN == 8, "eight warps");
    static_assert(!ADD_RESIDUAL || (BN == 8 && WN == 1 && !SWIGLU), "residual epilogue is plain BN8 only");
    static_assert(!SWIGLU_BF16 || (SWIGLU && BN == 8 && WM == 8 && WN == 1 && STAGES == 2 && !ADD_RESIDUAL),
                  "BF16 SwiGLU epilogue is the single-split BN8 NInfer component only");
    constexpr int WROWS = TS_Q35_BM / WM, WCOLS = BN / WN;
    constexpr int MT = WROWS / 16, NT = WCOLS / 8;
    extern __shared__ __align__(16) uint8_t smemRaw[];
    using Smem = TsQ35GemmSmem<BN, WM, WN, STAGES, BK>;
    constexpr int CROW = Smem::CROW, SROW = Smem::SROW, CHUNKS = BK / 32;
    auto& sm = *reinterpret_cast<Smem*>(smemRaw);

    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int wm = warp % WM, wn = warp / WM;
    // COLS_FAST: activation tiles vary fastest across the grid, so the CTAs in flight share
    // weight tiles in L2 (otherwise a weight matrix larger than L2 streams from memory once per
    // activation tile): +8% at 2048 rows, -2% at 128 or fewer.
    const int row0 = (COLS_FAST ? blockIdx.y : blockIdx.x) * TS_Q35_BM, col0 = (COLS_FAST ? blockIdx.x : blockIdx.y) * BN;
    const int splits = gridDim.z;
    const int stagesTotal = k / BK;
    const int perSplit = (stagesTotal + splits - 1) / splits;
    const int sBegin = blockIdx.z * perSplit;
    const int sEnd = min(stagesTotal, sBegin + perSplit);
    const int nStages = max(0, sEnd - sBegin);
    const long long wRowBytes = k / 2, wScaleBytes = k / 16;

    auto load = [&](int slot, int stage) {
        const long long kc = (long long)stage * (BK / 2), ks = (long long)stage * SROW;
        for (int i = tid; i < TS_Q35_BM * CHUNKS; i += 256)
        {
            const int r = i / CHUNKS, c = i % CHUNKS;
            const int gr = row0 + r;
            const bool ok = gr < outDim;
            ts_q35_cp16(sm.ac[slot] + r * CROW + c * 16, wCodes + (ok ? gr : 0) * wRowBytes + kc + c * 16, ok);
        }
        for (int r = tid; r < TS_Q35_BM; r += 256)
        {
            const int gr = row0 + r;
            const bool ok = gr < outDim;
            if (SROW == 16)
                ts_q35_cp16(sm.as[slot] + r * SROW, wScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
            else
                ts_q35_cp8(sm.as[slot] + r * SROW, wScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
        }
        for (int i = tid; i < BN * CHUNKS; i += 256)
        {
            const int r = i / CHUNKS, c = i % CHUNKS;
            const int gr = col0 + r;
            const bool ok = gr < m;
            ts_q35_cp16(sm.bc[slot] + r * CROW + c * 16, xCodes + (ok ? gr : 0) * wRowBytes + kc + c * 16, ok);
        }
        for (int r = tid; r < BN; r += 256)
        {
            const int gr = col0 + r;
            const bool ok = gr < m;
            if (SROW == 16)
                ts_q35_cp16(sm.bs[slot] + r * SROW, xScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
            else
                ts_q35_cp8(sm.bs[slot] + r * SROW, xScales + (ok ? gr : 0) * wScaleBytes + ks, ok);
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
        ts_q35_cp_commit();
    }
    const int g = lane >> 2, q = lane & 3;
    for (int it = 0; it < nStages; ++it)
    {
        ts_q35_cp_wait<STAGES - 2>();
        __syncthreads();
        // The slot this fills was consumed last iteration, which every warp has finished.
        const int next = it + STAGES - 1;
        if (next < nStages)
            load(next % STAGES, sBegin + next);
        ts_q35_cp_commit();
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
                ts_q35_ldsm_x4(a[i], ac + (r + (lane & 15)) * CROW + kstep * 32 + (lane >> 4) * 16);
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
                    ts_q35_mma_fp4(acc[i][j], a[i], b0, b1, sa[i], sb);
            }
        }
    }
    ts_q35_cp_wait<0>();

    // Epilogue: D rows are weight rows (g, g + 8), columns activation rows (2q, 2q + 1).
    auto element = [&](int i, int j, int e, int& r, int& c) {
        r = row0 + wm * WROWS + i * 16 + g + (e >= 2 ? 8 : 0);
        c = col0 + wn * WCOLS + j * 8 + q * 2 + (e & 1);
    };
    if (SWIGLU)
    {
        // Interleaved gate|up rows: an m16 tile's rows g (gate) and g + 8 (up) are the same FFN
        // column, so each thread holds both halves of its outputs: y [m][outDim / 2] =
        // silu(gate) * up, each half rounded as the plain epilogue stores it (no split-K).
        const int ff = outDim / 2;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
                for (int e = 0; e < 2; ++e)
                {
                    int r, c;
                    element(i, j, e, r, c);
                    if (r >= outDim || c >= m)
                        continue;
                    if constexpr (SWIGLU_BF16)
                    {
                        // Preserve the old F32 projection stores' two RN multiplies
                        // before the exact ts_ni_swiglu expf/divide/product sequence.
                        // Only the final activation is BF16-rounded and widened to F32.
                        const float gate = __fmul_rn(__fmul_rn(acc[i][j][e], rowScale[c]), alpha);
                        const float up = __fmul_rn(__fmul_rn(acc[i][j][e + 2], rowScale[c]), alpha);
                        const float activation = __fmul_rn(ts_q35_silu(gate), up);
                        y[(long long)c * ff + r / 16 * 8 + r % 16] =
                            __bfloat162float(__float2bfloat16_rn(activation));
                    }
                    else
                    {
                        const float gate = acc[i][j][e] * rowScale[c] * alpha, up = acc[i][j][e + 2] * rowScale[c] * alpha;
                        y[(long long)c * ff + r / 16 * 8 + r % 16] = ts_q35_silu(gate) * up;
                    }
                }
        return;
    }
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
                    {
                        if constexpr (ADD_RESIDUAL)
                        {
                            const long long index = (long long)c * outDim + r;
                            // Preserve both old projection FMUL boundaries and the
                            // following add_norm's F32 residual addition before BF16.
                            const float value = __fmul_rn(__fmul_rn(acc[i][j][e], rowScale[c]), alpha);
                            y[index] = value;
                            residualX[index] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(residualX[index], value)));
                        }
                        else
                            y[(long long)c * outDim + r] = acc[i][j][e] * rowScale[c] * alpha;
                    }
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
                if constexpr (ADD_RESIDUAL)
                {
                    // Only the existing last-split winner reaches this store.
                    const long long index = (long long)c * outDim + r;
                    const float value = __fmul_rn(__fmul_rn(sum, rowScale[c]), alpha);
                    y[index] = value;
                    residualX[index] = __bfloat162float(__float2bfloat16_rn(__fadd_rn(residualX[index], value)));
                }
                else
                    y[(long long)c * outDim + r] = sum * rowScale[c] * alpha;
            }
    if (tid == 0)
        counters[tile] = 0;
}

#define TS_Q35_GEMM(NAME, BN, WM, WN, STAGES, COLS_FAST, SWIGLU) \
    extern "C" __global__ __launch_bounds__(256) void NAME( \
        const uint8_t* __restrict__ wCodes, const uint8_t* __restrict__ wScales, int outDim, int k, \
        const uint8_t* __restrict__ xCodes, const uint8_t* __restrict__ xScales, const float* __restrict__ rowScale, \
        int m, float alpha, float* __restrict__ y, float* __restrict__ partial, int* __restrict__ counters) \
    { ts_q35_gemm_w4a4<BN, WM, WN, STAGES, TS_Q35_BK, COLS_FAST, SWIGLU>(wCodes, wScales, outDim, k, xCodes, xScales, rowScale, m, \
        alpha, y, partial, counters); }

// grid (weight tiles, activation tiles, splits); the 128-row kernels take (activation tiles, weight tiles, splits).
// _swiglu: the fused gate|up projection with SwiGLU in the epilogue (y is the FFN activation).
TS_Q35_GEMM(ts_q35_gemm_w4a4_32, 32, 8, 1, 3, false, false)
// Exact T8 specialization: same live-token MMA/split-K arithmetic as _32.
// Separate entry retains the existing kernels and their shared-memory requirements.
TS_Q35_GEMM(ts_q35_gemm_w4a4_8, 8, 8, 1, 2, false, false)
#define TS_Q35_HAS_GEMM_W4A4_8 1
// Isolated NInfer component: final activation, not a raw projection buffer.
// Existing entries and the ordinary SwiGLU epilogue retain their original behavior.
extern "C" __global__ __launch_bounds__(256) void ts_q35_gemm_w4a4_8_swiglu_bf16(
    const uint8_t* __restrict__ wCodes, const uint8_t* __restrict__ wScales, int outDim, int k,
    const uint8_t* __restrict__ xCodes, const uint8_t* __restrict__ xScales, const float* __restrict__ rowScale,
    int m, float alpha, float* __restrict__ y, float* __restrict__ partial, int* __restrict__ counters)
{
    if (m != 8 || outDim != 34816 || k != 5120 || gridDim.x != 272 || gridDim.y != 1 || gridDim.z != 1
        || blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1)
    {
        asm("trap;");
        return;
    }
    ts_q35_gemm_w4a4<8, 8, 1, 2, TS_Q35_BK, false, true, false, true>(
        wCodes, wScales, outDim, k, xCodes, xScales, rowScale, m, alpha, y, partial, counters);
}
TS_Q35_GEMM(ts_q35_gemm_w4a4_64, 64, 4, 2, 3, false, false)
TS_Q35_GEMM(ts_q35_gemm_w4a4_128, 128, 4, 2, 2, true, false)
TS_Q35_GEMM(ts_q35_gemm_w4a4_32_swiglu, 32, 8, 1, 3, false, true)
TS_Q35_GEMM(ts_q35_gemm_w4a4_64_swiglu, 64, 4, 2, 3, false, true)
TS_Q35_GEMM(ts_q35_gemm_w4a4_128_swiglu, 128, 4, 2, 2, true, true)

// ---------------------------------------------------------------------------
// NInfer prefill W4A4 GEMM, TMA-fed and warp-specialized (sm_120a), the structure of the
// nvfp4_w4a4_tma kernel Ignis vendors from NInfer (Apache-2.0), on TensorSharp's planes:
// activation and weight codes [rows][k/2] and UE4M3 scales [rows][k/16], read as they are.
//   * One producer thread issues every copy: per 128-value K tile, the BM x 64-byte activation
//     codes and 128 x 64-byte weight codes (64-byte swizzle) and, on even tiles, one 16-byte
//     scale box per row for both operands (TMA's narrowest box; it covers two K tiles).
//   * Eight consumer warps (4 over tokens x 2 over weight rows) run the block-scaled FP4 MMA
//     with activations as the A operand, so a warp's 64 weight rows are eight n8 tiles and
//     the gate|up interleave (8 gate rows, then their 8 up rows) puts a column's gate and up
//     in the same thread.
// Accumulation is the K64 MMA sequence of ts_q35_gemm_w4a4 in the same order (bit-identical
// accumulators), and NI_EPI applies the BF16 boundaries NInfer otherwise runs as separate
// passes over F32 buffers (ts_ni_round, ts_ni_swiglu, ts_ni_quant_nvfp4), same arithmetic:
//   0: y = acc * rowScale * alpha                         (plain projection)
//   1: y = BF16(acc * rowScale * alpha)                   (projection, ts_ni_round)
//   2: y = BF16(silu(BF16(gate)) * BF16(up))              (projection, ts_ni_round, ts_ni_swiglu)
//   3: y = BF16(silu(gate) * up), gate/up unrounded F32   (projection, ts_ni_swiglu)
//   4/5: 2/3, then the down projection's NVFP4 quantization of that activation (codes, E4M3
//        group scales, row scale 1) instead of storing it (ts_ni_quant_nvfp4).
//   6: y = silu(gate) * up in F32                          (the GGUF route's gate|up: the
//      ts_q35_gemm_w4a4_<bn>_swiglu epilogue; with 0 the GGUF prefill projections run here too)
// (A residual-add epilogue, y = BF16(y + projection), was measured 2.4x slower per GEMM: the
// read of y after the MMA loop stalls the consumer warps; ts_ni_add_norm_quant adds it instead.)
// The activation tile runs fastest across the grid: the CTAs in flight share a weight tile.
// Partial weight tiles (outDim not a multiple of 128) are zero-filled by TMA and not stored.
// ---------------------------------------------------------------------------
struct alignas(64) TsQ35TmaMap
{
    unsigned long long words[16];
};

struct alignas(64) TsQ35TmaMaps
{
    TsQ35TmaMap xCodes, xScales, wCodes, wScales;
};

template <int BM, int STAGES>
struct TsQ35TmaSmem
{
    alignas(1024) uint8_t xc[STAGES][BM * 64];
    alignas(1024) uint8_t wc[STAGES][128 * 64];
    alignas(128) uint8_t xs[2][BM * 16];
    alignas(128) uint8_t ws[2][128 * 16];
    alignas(8) unsigned long long full[STAGES];
    alignas(8) unsigned long long empty[STAGES];
};

#if defined(TS_Q35_FP4_MMA)
__device__ __forceinline__ void ts_q35_mbar_init(unsigned long long* bar, uint32_t count)
{
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;" :: "r"(ts_q35_smem_u32(bar)), "r"(count) : "memory");
}

__device__ __forceinline__ void ts_q35_mbar_wait(unsigned long long* bar, uint32_t parity)
{
    asm volatile("{\n"
                 ".reg .pred done;\n"
                 "ts_q35_wait:\n"
                 "mbarrier.try_wait.parity.shared::cta.b64 done, [%0], %1, %2;\n"
                 "@!done bra ts_q35_wait;\n"
                 "}\n"
                 :: "r"(ts_q35_smem_u32(bar)), "r"(parity), "r"(0x989680) : "memory");
}

__device__ __forceinline__ void ts_q35_mbar_arrive(unsigned long long* bar)
{
    asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];" :: "r"(ts_q35_smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void ts_q35_mbar_expect(unsigned long long* bar, uint32_t bytes)
{
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" :: "r"(ts_q35_smem_u32(bar)), "r"(bytes) : "memory");
}

__device__ __forceinline__ void ts_q35_tma_2d(void* dst, const TsQ35TmaMap* map, int c0, int c1, unsigned long long* bar)
{
    asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0], [%1, {%2, %3}], [%4];"
                 :: "r"(ts_q35_smem_u32(dst)), "l"(map), "r"(c0), "r"(c1), "r"(ts_q35_smem_u32(bar)) : "memory");
}

__device__ __forceinline__ void ts_q35_ldsm_x2(uint32_t& r0, uint32_t& r1, uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n" : "=r"(r0), "=r"(r1) : "r"(addr));
}

__device__ __forceinline__ void ts_q35_ldsm_x4a(uint32_t (&r)[4], uint32_t addr)
{
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(addr));
}

// 64-byte TMA swizzle: address bits [4:5] ^= bits [7:8] (CUTLASS Swizzle<2,4,3>).
__device__ __forceinline__ uint32_t ts_q35_sw64(uint32_t addr) { return addr ^ (((addr >> 7) & 3u) << 4); }

__device__ __forceinline__ unsigned ts_q35_e2m1_code(float qv)
{
    const float av = fabsf(qv);
    return (av <= 0.25f ? 0u : av < 0.75f ? 1u : av <= 1.25f ? 2u : av < 1.75f ? 3u
          : av <= 2.5f ? 4u : av < 3.5f ? 5u : av <= 5.0f ? 6u : 7u) | ((__float_as_uint(qv) >> 28) & 8u);
}
#endif

template <int BM, int STAGES, int NI_EPI>
__device__ __forceinline__ void ts_q35_gemm_w4a4_tma(const TsQ35TmaMaps& maps, int outDim, int k,
    const float* __restrict__ rowScale, int m, float alpha, float* __restrict__ y,
    uint8_t* __restrict__ qCodes, uint8_t* __restrict__ qScales, float* __restrict__ qRowScale, float qInputD)
{
    static_assert(BM == 128 || BM == 256, "token tile");
    static_assert(NI_EPI >= 0 && NI_EPI <= 6, "epilogues 0-6");
    static_assert(STAGES == 2 || STAGES == 3, "the two scale slots are rewritten at tile 2p + 4, after tile 2p + 1 is consumed");
#if defined(TS_Q35_FP4_MMA)
    constexpr int PRODUCERS = BM == 256 ? 128 : 32, WARP_M = BM / 4, MT = WARP_M / 16, NT = 8;
    extern __shared__ __align__(1024) uint8_t tmaSmem[];
    auto& sm = *reinterpret_cast<TsQ35TmaSmem<BM, STAGES>*>(tmaSmem);
    const int token0 = blockIdx.x * BM, row0 = blockIdx.y * 128, kTiles = k / 128;

    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            ts_q35_mbar_init(&sm.full[s], 1);
            ts_q35_mbar_init(&sm.empty[s], 8);
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
                ts_q35_mbar_wait(&sm.empty[s], 1u ^ ((kt / STAGES) & 1u));
                const bool scales = (kt & 1) == 0;
                ts_q35_mbar_expect(&sm.full[s], scales ? CODES + SCALES : CODES);
                ts_q35_tma_2d(sm.xc[s], &maps.xCodes, kt * 64, token0, &sm.full[s]);
                ts_q35_tma_2d(sm.wc[s], &maps.wCodes, kt * 64, row0, &sm.full[s]);
                if (scales)
                {
                    ts_q35_tma_2d(sm.xs[(kt >> 1) & 1], &maps.xScales, (kt >> 1) * 16, token0, &sm.full[s]);
                    ts_q35_tma_2d(sm.ws[(kt >> 1) & 1], &maps.wScales, (kt >> 1) * 16, row0, &sm.full[s]);
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
        ts_q35_mbar_wait(&sm.full[s], (kt / STAGES) & 1u);
        const uint32_t xc = ts_q35_smem_u32(sm.xc[s]), wc = ts_q35_smem_u32(sm.wc[s]);
        const uint8_t* xs = sm.xs[(kt >> 1) & 1] + (kt & 1) * 8;
        const uint8_t* ws = sm.ws[(kt >> 1) & 1] + (kt & 1) * 8;
#pragma unroll
        for (int k64 = 0; k64 < 2; ++k64)
        {
            uint32_t a[MT][4], sa[MT];
#pragma unroll
            for (int i = 0; i < MT; ++i)
            {
                ts_q35_ldsm_x4a(a[i], ts_q35_sw64(xc + (aRow + i * 16) * 64 + k64 * 32 + aByte));
                sa[i] = *(const uint32_t*)(xs + (saRow + i * 16) * 16 + k64 * 4);
            }
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                uint32_t b0, b1;
                ts_q35_ldsm_x2(b0, b1, ts_q35_sw64(wc + (bRow + j * 8) * 64 + k64 * 32 + bByte));
                const uint32_t sb = *(const uint32_t*)(ws + (sbRow + j * 8) * 16 + k64 * 4);
#pragma unroll
                for (int i = 0; i < MT; ++i)
                    ts_q35_mma_fp4(acc[i][j], a[i], b0, b1, sa[i], sb);
            }
        }
        if (lane == 0)
            ts_q35_mbar_arrive(&sm.empty[s]);
    }

    // D rows are tokens (g, g + 8), columns weight rows (2q, 2q + 1).
    const int tokBase = token0 + wm * WARP_M + g, rowBase = row0 + wn * 64 + 2 * q;
    if constexpr (NI_EPI == 4 || NI_EPI == 5)
    {
        // A 16-value NVFP4 group of FFN columns is two gate|up tile pairs; this thread holds its
        // values 2q, 2q + 1 (pair 0) and 8 + 2q, 9 + 2q (pair 1), the group's other values in
        // the three lanes sharing g. Byte b of a group packs values b (low) and b + 8 (high).
        const int ff = outDim / 2, groups = ff / 16;
#pragma unroll
        for (int i = 0; i < MT; ++i)
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int c = tokBase + i * 16 + h * 8;
                const bool live = c < m;
                const float rs = live ? rowScale[c] : 0.0f;
#pragma unroll
                for (int gp = 0; gp < 2; ++gp)
                {
                    float act[2][2];
#pragma unroll
                    for (int p = 0; p < 2; ++p)
#pragma unroll
                        for (int e = 0; e < 2; ++e)
                        {
                            const int jt = gp * 4 + p * 2;
                            float gate = __fmul_rn(__fmul_rn(acc[i][jt][h * 2 + e], rs), alpha);
                            float up = __fmul_rn(__fmul_rn(acc[i][jt + 1][h * 2 + e], rs), alpha);
                            if constexpr (NI_EPI == 4)
                            {
                                gate = __bfloat162float(__float2bfloat16_rn(gate));
                                up = __bfloat162float(__float2bfloat16_rn(up));
                            }
                            act[p][e] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(ts_q35_silu(gate), up)));
                        }
                    float a = fmaxf(fmaxf(fabsf(act[0][0]), fabsf(act[0][1])), fmaxf(fabsf(act[1][0]), fabsf(act[1][1])));
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 1));
                    a = fmaxf(a, __shfl_xor_sync(0xffffffffu, a, 2));
                    const uint8_t sc = __nv_cvt_float_to_fp8(__fdiv_rn(qInputD * a, 6.0f), __NV_SATFINITE, __NV_E4M3);
                    __nv_fp8_e4m3 dec;
                    dec.__x = sc;
                    const float sv = (float)dec;
                    unsigned code[2][2];
#pragma unroll
                    for (int p = 0; p < 2; ++p)
#pragma unroll
                        for (int e = 0; e < 2; ++e)
                            code[p][e] = ts_q35_e2m1_code(sc ? __fdiv_rn(act[p][e] * qInputD, sv) : 0.0f);
                    if (live)
                    {
                        const int grp = (row0 + wn * 64) / 32 + gp;
                        const uint16_t packed = (uint16_t)((code[0][0] | (code[1][0] << 4)) | ((code[0][1] | (code[1][1] << 4)) << 8));
                        *(uint16_t*)(qCodes + (long long)c * (ff / 2) + grp * 8 + 2 * q) = packed;
                        if (q == 0)
                        {
                            qScales[(long long)c * groups + grp] = sc;
                            if (grp == 0)
                                qRowScale[c] = 1.0f;
                        }
                    }
                }
            }
        return;
    }
    if constexpr (NI_EPI == 2 || NI_EPI == 3 || NI_EPI == 6)
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
                for (int jp = 0; jp < NT / 2; ++jp)
                {
                    const int r = rowBase + jp * 16;
                    if (r >= outDim)
                        continue;
                    float out[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        float gate = __fmul_rn(__fmul_rn(acc[i][2 * jp][h * 2 + e], rs), alpha);
                        float up = __fmul_rn(__fmul_rn(acc[i][2 * jp + 1][h * 2 + e], rs), alpha);
                        if constexpr (NI_EPI == 2)
                        {
                            gate = __bfloat162float(__float2bfloat16_rn(gate));
                            up = __bfloat162float(__float2bfloat16_rn(up));
                        }
                        if constexpr (NI_EPI == 6)
                            out[e] = ts_q35_silu(gate) * up;
                        else
                            out[e] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(ts_q35_silu(gate), up)));
                    }
                    *(float2*)(y + (long long)c * ff + r / 16 * 8 + r % 16) = make_float2(out[0], out[1]);
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
                float v0 = acc[i][j][h * 2] * rs * alpha, v1 = acc[i][j][h * 2 + 1] * rs * alpha;
                if constexpr (NI_EPI == 1)
                {
                    v0 = __bfloat162float(__float2bfloat16_rn(v0));
                    v1 = __bfloat162float(__float2bfloat16_rn(v1));
                }
                *(float2*)(y + (long long)c * outDim + r) = make_float2(v0, v1);
            }
        }
#else
    __trap();
#endif
}

// grid (activation tiles, 128-row weight tiles), block 256 consumers + 32 (BM 128) or 128
// (BM 256) producers; outDim and k multiples of 128 (the managed router checks).
#define TS_Q35_GEMM_TMA(NAME, BM, STAGES, NI) \
    extern "C" __global__ __launch_bounds__(256 + (BM == 256 ? 128 : 32), 1) void NAME( \
        const __grid_constant__ TsQ35TmaMaps maps, int outDim, int k, const float* __restrict__ rowScale, int m, \
        float alpha, float* __restrict__ y, uint8_t* __restrict__ qCodes, uint8_t* __restrict__ qScales, \
        float* __restrict__ qRowScale, float qInputD) \
    { ts_q35_gemm_w4a4_tma<BM, STAGES, NI>(maps, outDim, k, rowScale, m, alpha, y, qCodes, qScales, qRowScale, qInputD); }
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma256_ni_plain, 256, 3, 0)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma256_ni_round, 256, 3, 1)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma256_ni_swiglu_rin, 256, 2, 2)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma256_ni_swiglu, 256, 2, 3)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma256_ni_swiglu_rin_q, 256, 2, 4)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma256_ni_swiglu_q, 256, 2, 5)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma128_ni_plain, 128, 3, 0)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma128_ni_round, 128, 3, 1)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma128_ni_swiglu_rin, 128, 3, 2)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma128_ni_swiglu, 128, 3, 3)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma128_ni_swiglu_rin_q, 128, 3, 4)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma128_ni_swiglu_q, 128, 3, 5)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma256_gguf_swiglu, 256, 2, 6)
TS_Q35_GEMM_TMA(ts_q35_gemm_w4a4_tma128_gguf_swiglu, 128, 3, 6)

/// Dynamic shared memory of the TMA GEMMs: [BM256 3-stage, BM256 2-stage, BM128 3-stage].
extern "C" __global__ void ts_q35_gemm_tma_smem_bytes(int* out)
{
    out[0] = (int)sizeof(TsQ35TmaSmem<256, 3>);
    out[1] = (int)sizeof(TsQ35TmaSmem<256, 2>);
    out[2] = (int)sizeof(TsQ35TmaSmem<128, 3>);
}

// Qualified residual entry; caller supplies disjoint residual/projection buffers,
// grid(40,1,8), block256 and 43520 dynamic shared bytes. Managed dispatch admits
// full-valid T8/B1 verification only. The original twelve-argument entries remain unchanged.
extern "C" __global__ __launch_bounds__(256) void ts_q35_gemm_w4a4_8_residual_bf16(
    const uint8_t* __restrict__ wCodes, const uint8_t* __restrict__ wScales, int outDim, int k,
    const uint8_t* __restrict__ xCodes, const uint8_t* __restrict__ xScales, const float* __restrict__ rowScale,
    int m, float alpha, float* __restrict__ y, float* __restrict__ partial, int* __restrict__ counters,
    float* __restrict__ residualX)
{
    if (m != 8 || outDim != 5120 || (k != 6144 && k != 17408) || residualX == nullptr)
    {
        asm("trap;");
        return;
    }
    ts_q35_gemm_w4a4<8, 8, 1, 2, TS_Q35_BK, false, false, true>(wCodes, wScales, outDim, k,
        xCodes, xScales, rowScale, m, alpha, y, partial, counters, residualX);
}

/// Dynamic shared memory of ts_q35_gemm_w4a4_<bn>, by bn (32, 64, 128).
extern "C" __global__ void ts_q35_gemm_smem_bytes(int* out)
{
    out[0] = (int)sizeof(TsQ35GemmSmem<32, 8, 1, 3>);
    out[1] = (int)sizeof(TsQ35GemmSmem<64, 4, 2, 3>);
    out[2] = (int)sizeof(TsQ35GemmSmem<128, 4, 2, 2>);
}

// Readonly norm over already BF16-rounded F32 residuals. Its
// reduction is the original NInfer add_norm's 256-thread order, without adding
// +0, rounding or storing x. In particular, input negative zero stays negative.
// The qualified caller uses width5120 and grid8; the bounded width stays a
// runtime argument so the mean retains the original runtime-width division.
// Keep the initial embedding norm and every unfused residual on the old helper.
extern "C" __global__ __launch_bounds__(256) void ts_q35_ninfer_plain_norm_f32(
    const float* __restrict__ x, const float* __restrict__ weight, float* __restrict__ y, int width, float eps)
{
    if (blockDim.x != 256 || blockDim.y != 1 || blockDim.z != 1 || gridDim.x != 8 || gridDim.y != 1
        || gridDim.z != 1 || width < 1 || width > 5120)
    {
        asm("trap;");
        return;
    }
    const int row = blockIdx.x, lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    float values[20], weights[20], ss = 0;
#pragma unroll
    for (int j = 0; j < 20; ++j)
    {
        const int i = threadIdx.x + j * blockDim.x;
        if (i < width)
        {
            const float v = x[(long long)row * width + i];
            values[j] = v;
            // The original F32 effective weight (including its exact unit offset)
            // is independent of the reduction and can be fetched in the first pass.
            weights[j] = weight[i];
            ss += v * v;
        }
    }
    ss = ts_q35_warp_sum(ss);
    __shared__ float partial[32], inv;
    if (!lane) partial[warp] = ss;
    __syncthreads();
    if (!warp)
    {
        ss = lane < (blockDim.x >> 5) ? partial[lane] : 0;
        ss = ts_q35_warp_sum(ss);
        if (!lane) inv = rsqrtf(ss / (float)width + eps);
    }
    __syncthreads();
#pragma unroll
    for (int j = 0; j < 20; ++j)
    {
        const int i = threadIdx.x + j * blockDim.x;
        if (i < width)
            y[(long long)row * width + i] = __bfloat162float(__float2bfloat16_rn(values[j] * inv * weights[j]));
    }
}

// ===========================================================================
// Layer kernels. Activations are F32 rows [tokens][width]. A launch covers nSeq sequences of
// rowsPerSeq consecutive rows each (one sequence's prompt chunk: nSeq = 1; a batched decode or
// verify step: one window per sequence), each sequence bringing its own caches and states
// through device arrays of pointers.
// ===========================================================================

#define TS_Q35_HD 256          // attention head width
#define TS_Q35_GDN_HD 128      // Gated DeltaNet head width (keys and values)
// Attention caches live in blocks of this many tokens, shared between slots (a checkpoint and the
// sequence it came from hold the same blocks): per attention layer, K then V, each
// [nKv][TS_Q35_KV_BLOCK][272 B] (a head's keys contiguous). A sequence's blocks are found through
// its row of the block table.
#define TS_Q35_KV_BLOCK 256
#define TS_Q35_ROPE_DIMS 64    // rotated dims per attention head (NEOX pairs (i, i + 32))
// One 256-value q8_0 row: the 256 int8 codes, then the 8 F16 block scales (block j = values
// 32j..32j+31, value = d_j * code). ggml's q8_0 numerics in a layout whose codes are 16-byte
// aligned, so they feed cp.async, ldmatrix and int8 mma directly.
#define TS_Q35_Q8_ROW 272
#define TS_Q35_Q8_SCALES 256   // byte offset of the scales within a row

// ---------------------------------------------------------------------------
// Token embedding from a Q4_0 table: one block per token, rows [tokens][width].
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_q35_embed_q4_0(
    const uint8_t* __restrict__ table, long long rowBytes, const int* __restrict__ tokens,
    float* __restrict__ out, int width)
{
    const int t = blockIdx.x;
    const uint8_t* row = table + (long long)tokens[t] * rowBytes;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
    {
        const uint8_t* blk = row + (i >> 5) * 18;
        const float d = __half2float(*(const half*)blk);
        const int j = i & 31;
        const int q = j < 16 ? (blk[2 + j] & 0xF) : (blk[2 + j - 16] >> 4);
        out[(long long)t * width + i] = (float)(q - 8) * d;
    }
}

// ---------------------------------------------------------------------------
// x += delta (when delta is not null), then y = rmsnorm(x) * w. One block per row.
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_q35_add_rmsnorm_f32(
    float* __restrict__ x, const float* __restrict__ delta, const float* __restrict__ w,
    float* __restrict__ y, int width, float eps)
{
    __shared__ float partial[32];
    const long long row = blockIdx.x;
    float* xr = x + row * width;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
    {
        float v = xr[i];
        if (delta != nullptr)
        {
            v += delta[row * width + i];
            xr[i] = v;
        }
        ss += v * v;
    }
    ss = ts_q35_warp_sum(ss);
    if ((threadIdx.x & 31) == 0)
        partial[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float v = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.0f;
        v = ts_q35_warp_sum(v);
        if (threadIdx.x == 0)
            partial[0] = v;
    }
    __syncthreads();
    const float scale = rsqrtf(partial[0] / (float)width + eps);
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        y[row * width + i] = xr[i] * scale * w[i];
}

// ts_q35_add_rmsnorm_f32 for width 5120 at 512 threads with the row in registers: the same values
// per thread, sum order, reductions and expressions (bit-identical output), but every load issued
// up front rather than one dependent round trip per iteration (a verify step runs ~140 of these).
extern "C" __global__ void __launch_bounds__(512) ts_q35_add_rmsnorm_5120_f32(
    float* __restrict__ x, const float* __restrict__ delta, const float* __restrict__ w,
    float* __restrict__ y, int width, float eps)
{
    constexpr int PER = 10;
    __shared__ float partial[32];
    const long long row = blockIdx.x;
    float* xr = x + row * width;
    float v[PER];
#pragma unroll
    for (int p = 0; p < PER; ++p)
    {
        v[p] = xr[threadIdx.x + p * 512];
        if (delta != nullptr) v[p] += delta[row * width + threadIdx.x + p * 512];
    }
    float ss = 0.0f;
#pragma unroll
    for (int p = 0; p < PER; ++p)
    {
        if (delta != nullptr) xr[threadIdx.x + p * 512] = v[p];
        ss += v[p] * v[p];
    }
    ss = ts_q35_warp_sum(ss);
    if ((threadIdx.x & 31) == 0)
        partial[threadIdx.x >> 5] = ss;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float t = threadIdx.x < (blockDim.x >> 5) ? partial[threadIdx.x] : 0.0f;
        t = ts_q35_warp_sum(t);
        if (threadIdx.x == 0)
            partial[0] = t;
    }
    __syncthreads();
    const float scale = rsqrtf(partial[0] / (float)width + eps);
#pragma unroll
    for (int p = 0; p < PER; ++p)
        y[row * width + threadIdx.x + p * 512] = v[p] * scale * w[threadIdx.x + p * 512];
}

// ---------------------------------------------------------------------------
// Attention front end. qkv rows are [Q|gate per head (nHead x 2 x 256) | K (nKv x 256) |
// V (nKv x 256)]. One warp per (row, unit): units 0..nHead-1 are query heads, then nKv key heads,
// then nKv value heads. Lane l holds dims l + 32j, so the NEOX pair (i, i + 32) of the rotated
// first 64 dims is lane-local, and q8_0 block j (dims 32j..32j+31) is one value per lane.
// Q (normed, roped) and the raw gate go to q/gate [rows][nHead][256]; K (normed, roped) and V
// are quantized to q8_0 into the sequence's cache blocks at cacheRow (see TS_Q35_KV_BLOCK).
// ropePos holds the four M-RoPE position components [4][ropeStride]; sections are (t, h, w) pair
// counts, interleaved t, h, w, t, h, w, ... over the 32 rotated pairs (ggml IMROPE).
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_q35_attn_prep_f32(
    const float* __restrict__ qkv, int rows, int rowsPerSeq, int nHead, int nKv,
    const float* __restrict__ qNorm, const float* __restrict__ kNorm, float eps,
    const int* __restrict__ ropePos, int ropeStride, int sectT, int sectH, int sectW, float ropeBase,
    const int* __restrict__ cacheRow, uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    float* __restrict__ q, float* __restrict__ gate)
{
    const int units = nHead + 2 * nKv;
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= (long long)rows * units)
        return;
    const int row = (int)(warp / units), unit = (int)(warp % units);
    const int seq = row / rowsPerSeq;
    const long long rowWidth = (long long)nHead * 2 * TS_Q35_HD + 2LL * nKv * TS_Q35_HD;
    const float* src;
    if (unit < nHead)
        src = qkv + row * rowWidth + (long long)unit * 2 * TS_Q35_HD;
    else if (unit < nHead + nKv)
        src = qkv + row * rowWidth + (long long)nHead * 2 * TS_Q35_HD + (long long)(unit - nHead) * TS_Q35_HD;
    else   // V follows the nKv key heads
        src = qkv + row * rowWidth + (long long)nHead * 2 * TS_Q35_HD + (long long)(unit - nHead) * TS_Q35_HD;

    float v[8];
#pragma unroll
    for (int j = 0; j < 8; ++j)
        v[j] = src[lane + 32 * j];

    if (unit < nHead + nKv)
    {
        // Per-head RMS norm, then rotary on dims [0, 64): pair (lane, lane + 32) is v[0], v[1].
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            ss += v[j] * v[j];
        ss = ts_q35_warp_sum(ss);
        const float scale = rsqrtf(ss / (float)TS_Q35_HD + eps);
        const float* nw = unit < nHead ? qNorm : kNorm;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = v[j] * scale * nw[lane + 32 * j];
        const int pair = lane;   // 0..31
        int comp;
        if (pair % 3 == 1 && pair < 3 * sectH) comp = 1;
        else if (pair % 3 == 2 && pair < 3 * sectW) comp = 2;
        else if (pair % 3 == 0 && pair < 3 * sectT) comp = 0;
        else comp = 3;
        const float p = (float)ropePos[(long long)comp * ropeStride + row];
        const float theta = p * powf(ropeBase, -2.0f * (float)pair / (float)TS_Q35_ROPE_DIMS);
        float sn, cs;
        sincosf(theta, &sn, &cs);
        const float x0 = v[0], x1 = v[1];
        v[0] = x0 * cs - x1 * sn;
        v[1] = x0 * sn + x1 * cs;
    }
    if (unit < nHead)
    {
        float* qd = q + ((long long)row * nHead + unit) * TS_Q35_HD;
        float* gd = gate + ((long long)row * nHead + unit) * TS_Q35_HD;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            qd[lane + 32 * j] = v[j];
            gd[lane + 32 * j] = src[TS_Q35_HD + lane + 32 * j];
        }
        return;
    }
    const bool isK = unit < nHead + nKv;
    const int head = isK ? unit - nHead : unit - nHead - nKv;
    const int cr = cacheRow[row];
    uint8_t* cache = blockTable[(long long)seq * maxBlocks + cr / TS_Q35_KV_BLOCK] + (isK ? kOff : vOff)
                   + ((long long)head * TS_Q35_KV_BLOCK + cr % TS_Q35_KV_BLOCK) * TS_Q35_Q8_ROW;
#pragma unroll
    for (int j = 0; j < 8; ++j)
    {
        // ggml quantize_row_q8_0: d = amax / 127, q = round(x / d).
        const float amax = ts_q35_warp_max(fabsf(v[j]));
        const float d = amax / 127.0f;
        const float id = d != 0.0f ? 1.0f / d : 0.0f;
        cache[32 * j + lane] = (uint8_t)(int8_t)(int)roundf(v[j] * id);
        if (lane == 0)
            *(half*)(cache + TS_Q35_Q8_SCALES + 2 * j) = __float2half(d);
    }
}

// ---------------------------------------------------------------------------
// out = attn * sigmoid(gate), elementwise over [rows][nHead * 256].
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_q35_gate_mul_f32(float* __restrict__ attn, const float* __restrict__ gate, long long n)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        attn[i] *= ts_q35_sigmoid(gate[i]);
}

// ---------------------------------------------------------------------------
// SwiGLU over the fused gate|up projection: out[r][i] = silu(gu[r][i]) * gu[r][ff + i].
// ---------------------------------------------------------------------------
// gu rows [gate | up] (interleaved = 0), or groups of 8 gate then 8 up values (interleaved = 1,
// the target's gate|up planes, see ts_q35_repack_nvfp4).
extern "C" __global__ void ts_q35_swiglu_f32(const float* __restrict__ gu, float* __restrict__ out, int ff, int interleaved)
{
    // grid (column blocks, rows)
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= ff)
        return;
    const long long r = blockIdx.y;
    const float* row = gu + r * 2 * ff;
    const int gi = interleaved ? c / 8 * 16 + c % 8 : c, ui = interleaved ? gi + 8 : ff + c;
    out[r * ff + c] = ts_q35_silu(row[gi]) * row[ui];
}

// ---------------------------------------------------------------------------
// Gated DeltaNet front end over the packed in-projection rows [qkv (convDim) | z (vDim) |
// beta (hv) | alpha (hv)]. ts_q35_gdn_conv_f32: one thread per (row, channel) runs the causal
// convolution (kernel dConv; the state [dConv - 1][convDim] holds the inputs before this window,
// oldest first) with SiLU; rawOut, when set, keeps the window's unconvolved inputs [rows][convDim]
// for a later state commit. ts_q35_gdn_conv_state_f32 then moves the state past the window.
// ts_q35_gdn_prep_f32: one warp per (row, head) L2-normalizes q and k in place (ggml:
// rms_norm(x, eps / n) / sqrt(n)) and computes beta = sigmoid(b) and g = softplus(alpha +
// dtBias) * ssmA (a log decay). The convolved q | k | v land in qkvOut [rows][convDim].
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_q35_gdn_conv_f32(
    const float* __restrict__ packed, int packedWidth, int rowsPerSeq, int convDim, int dConv,
    const float* __restrict__ convW, float* const* __restrict__ convStates, float* __restrict__ qkvOut,
    float* __restrict__ rawOut)
{
    // One thread per (row, channel); grid (channel blocks, rows, sequences).
    const int seq = blockIdx.z, t = blockIdx.y;
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    if (ch >= convDim)
        return;
    const float* state = convStates[seq];   // [dConv - 1][convDim], oldest first
    const long long base = (long long)seq * rowsPerSeq;
    const float* wk = convW + (long long)ch * dConv;
    float acc = 0.0f;
    for (int j = 0; j < dConv; ++j)
    {
        const int src = t - (dConv - 1) + j;   // row of the input tap j reads
        const float x = src >= 0 ? packed[(base + src) * packedWidth + ch] : state[(long long)(dConv - 1 + src) * convDim + ch];
        acc += x * wk[j];
    }
    qkvOut[(base + t) * convDim + ch] = ts_q35_silu(acc);
    if (rawOut != nullptr)
        rawOut[(base + t) * convDim + ch] = packed[(base + t) * packedWidth + ch];
}

// The convolution state after the first count[seq] (null: all rowsPerSeq) rows of a window:
// the last dConv - 1 inputs, from those rows and, when there are fewer, the old state (read
// ahead of every write: entry i takes old entry i + rows, so ascending i never reads what it
// already replaced). The rows sit at src [seq * rowsPerSeq + t][srcWidth]; grid (channel
// blocks, sequences, layers), layer z reading src + z * layerStride and the state
// convStates[z * ptrStride + seq] (null: no recurrent layer there).
extern "C" __global__ void ts_q35_gdn_conv_state_f32(
    const float* __restrict__ src, int srcWidth, long long layerStride, int rowsPerSeq, const int* __restrict__ count,
    int convDim, int dConv, float* const* __restrict__ convStates, int ptrStride)
{
    const int seq = blockIdx.y, layer = blockIdx.z;
    const int ch = blockIdx.x * blockDim.x + threadIdx.x;
    float* state = convStates[(long long)layer * ptrStride + seq];
    if (ch >= convDim || state == nullptr)
        return;
    const int rows = count != nullptr ? count[seq] : rowsPerSeq;
    const float* in = src + layer * layerStride + (long long)seq * rowsPerSeq * srcWidth;
    for (int i = 0; i < dConv - 1; ++i)
    {
        const int r = rows - (dConv - 1) + i;
        state[(long long)i * convDim + ch] = r >= 0
            ? in[(long long)r * srcWidth + ch]
            : state[(long long)(dConv - 1 + r) * convDim + ch];
    }
}

// count (null: every row real): sequence s's rows past count[s] are padding (rowsPerSeq rows per
// sequence) and get beta = 0 and g = 0, which leaves the delta rule's state exactly as it was.
extern "C" __global__ void ts_q35_gdn_prep_f32(
    float* __restrict__ qkv, const float* __restrict__ packed, int packedWidth, int rows,
    int hk, int hv, int convDim, int vDim, const float* __restrict__ dtBias, const float* __restrict__ ssmA,
    float eps, float* __restrict__ g, float* __restrict__ beta, const int* __restrict__ count, int rowsPerSeq)
{
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int units = 2 * hk + hv;
    if (warp >= (long long)rows * units)
        return;
    const long long row = warp / units;
    const int unit = (int)(warp % units);
    if (unit < 2 * hk)
    {
        float* p = qkv + row * convDim + (long long)unit * TS_Q35_GDN_HD;
        float v[4], ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            v[j] = p[lane + 32 * j];
            ss += v[j] * v[j];
        }
        ss = ts_q35_warp_sum(ss);
        const float n = (float)TS_Q35_GDN_HD;
        const float scale = rsqrtf(ss / n + eps / n) / sqrtf(n);
#pragma unroll
        for (int j = 0; j < 4; ++j)
            p[lane + 32 * j] = v[j] * scale;
        return;
    }
    const int h = unit - 2 * hk;
    if (lane == 0)
    {
        const float* pr = packed + row * packedWidth + convDim + vDim;
        const bool pad = count != nullptr && row % rowsPerSeq >= count[row / rowsPerSeq];
        beta[row * hv + h] = pad ? 0.0f : ts_q35_sigmoid(pr[h]);
        g[row * hv + h] = pad ? 0.0f : ts_q35_softplus(pr[hv + h] + dtBias[h]) * ssmA[h];
    }
}

// ---------------------------------------------------------------------------
// The gated delta rule, token by token (ggml-cuda's gated_delta_net_cuda arithmetic). State per
// (sequence, v-head) is [128 cols][128 rows] with M[col][i] = S[i][col]; value head h reads key
// head h % hk. Block: 4 warps x 4 columns per warp (8 lanes per column, 16 rows per lane), grid
// (hv, nSeq, 128 / 16). out [rows][hv][128] scaled by 1/sqrt(128).
//   ts_q35_gdn_scan_f32:        rowsPerSeq rows per sequence, outputs and the advanced state.
//   ts_q35_gdn_scan_verify_f32: outputs only; the state stays as it was (a speculative window).
//   ts_q35_gdn_replay_f32:      the state only, advanced over the first kept[seq] rows of a
//                               verified window - every recurrent layer in one launch: grid z is
//                               layer * 8 + column block, the layer's inputs at
//                               layer * qkvStride / gStride and its state states[layer * ptrStride + seq].
// ---------------------------------------------------------------------------
template <bool OUT, bool STATE>
__device__ __forceinline__ void ts_q35_gdn_scan(
    const float* __restrict__ qkv, int convDim, int hk, int hv, int rowsPerSeq, int rows,
    const float* __restrict__ g, const float* __restrict__ beta, float* __restrict__ states, float* __restrict__ out, int colBlock)
{
    constexpr int S = TS_Q35_GDN_HD, COLS = 4, LANES = 32 / COLS, PER = S / LANES;
    const int h = blockIdx.x, seq = blockIdx.y;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    const int colInWarp = lane / LANES, laneInCol = lane % LANES;
    const int col = (colBlock * (blockDim.x >> 5) + warp) * COLS + colInWarp;
    const int kh = h % hk;
    float* M = states + ((long long)h * S + col) * S;
    float s[PER];
#pragma unroll
    for (int r = 0; r < PER; ++r)
        s[r] = M[r * LANES + laneInCol];
    const float scale = rsqrtf((float)S);
    for (int t = 0; t < rows; ++t)
    {
        const long long row = (long long)seq * rowsPerSeq + t;
        const float* qt = qkv + row * convDim + (long long)kh * S;
        const float* kt = qkv + row * convDim + (long long)(hk + kh) * S;
        const float vt = qkv[row * convDim + 2LL * hk * S + (long long)h * S + col];
        const float gv = expf(g[row * hv + h]);
        const float bv = beta[row * hv + h];
        float kr[PER], kv = 0.0f;
#pragma unroll
        for (int r = 0; r < PER; ++r)
        {
            kr[r] = kt[r * LANES + laneInCol];
            kv += s[r] * kr[r];
        }
#pragma unroll
        for (int off = LANES / 2; off > 0; off >>= 1)
            kv += __shfl_xor_sync(0xffffffff, kv, off, 32);
        const float delta = (vt - gv * kv) * bv;
#pragma unroll
        for (int r = 0; r < PER; ++r)
            s[r] = gv * s[r] + kr[r] * delta;
        if (OUT)
        {
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < PER; ++r)
                o += s[r] * qt[r * LANES + laneInCol];
#pragma unroll
            for (int off = LANES / 2; off > 0; off >>= 1)
                o += __shfl_xor_sync(0xffffffff, o, off, 32);
            if (laneInCol == 0)
                out[(row * hv + h) * S + col] = o * scale;
        }
    }
    if (STATE)
    {
#pragma unroll
        for (int r = 0; r < PER; ++r)
            M[r * LANES + laneInCol] = s[r];
    }
}

extern "C" __global__ void __launch_bounds__(128) ts_q35_gdn_scan_f32(
    const float* __restrict__ qkv, int convDim, int hk, int hv, int rowsPerSeq,
    const float* __restrict__ g, const float* __restrict__ beta,
    float* const* __restrict__ states, float* __restrict__ out)
{
    ts_q35_gdn_scan<true, true>(qkv, convDim, hk, hv, rowsPerSeq, rowsPerSeq, g, beta, states[blockIdx.y], out, blockIdx.z);
}

extern "C" __global__ void __launch_bounds__(128) ts_q35_gdn_scan_verify_f32(
    const float* __restrict__ qkv, int convDim, int hk, int hv, int rowsPerSeq,
    const float* __restrict__ g, const float* __restrict__ beta,
    float* const* __restrict__ states, float* __restrict__ out)
{
    ts_q35_gdn_scan<true, false>(qkv, convDim, hk, hv, rowsPerSeq, rowsPerSeq, g, beta, states[blockIdx.y], out, blockIdx.z);
}

extern "C" __global__ void __launch_bounds__(128) ts_q35_gdn_replay_f32(
    const float* __restrict__ qkv, long long qkvStride, int convDim, int hk, int hv, int rowsPerSeq, const int* __restrict__ kept,
    const float* __restrict__ g, const float* __restrict__ beta, long long gStride,
    float* const* __restrict__ states, int ptrStride)
{
    constexpr int BLOCKS = TS_Q35_GDN_HD / 16;
    const int layer = blockIdx.z / BLOCKS;
    float* state = states[(long long)layer * ptrStride + blockIdx.y];
    if (state == nullptr)
        return;
    ts_q35_gdn_scan<false, true>(qkv + layer * qkvStride, convDim, hk, hv, rowsPerSeq, kept[blockIdx.y],
        g + layer * gStride, beta + layer * gStride, state, nullptr, blockIdx.z % BLOCKS);
}

// ---------------------------------------------------------------------------
// The same rule over chunks of C = 16 tokens (prefill). With G_t the chunk's cumulative log
// decay through token t and S0 the incoming state [128 k][V], the token loop unrolls to
//   U = (I + A)^-1 D,   A[t][s] = beta_t e^(G_t - G_s) k_t.k_s (s < t),
//                       D[t]    = beta_t (v_t - e^G_t S0^T k_t)
//   o_t = e^G_t S0^T q_t + sum_{s <= t} e^(G_t - G_s) (q_t.k_s) U[s]
//   S   = e^G_last S0 + K^T (U * e^(G_last - G_s))
// (the WY form ggml-cuda's gated-delta-net-mma.cu evaluates). Every product runs on BF16 tensor
// cores with each operand split into a BF16 value and its BF16 residual (lo*hi + hi*lo + hi*hi,
// about 16 mantissa bits); the state stays F32 in registers across chunks. One block of 8 warps
// owns one (sequence, v-head, 32-column value slice): grid (hv * 4, nSeq). Warp w keeps state
// rows 16w..16w+15. Operands sit in shared memory with the contraction index innermost, rows
// padded to 68 or 12 words so the fragment loads are conflict-free.
// ---------------------------------------------------------------------------
#define TS_Q35_GDN_C 16                     // tokens per chunk
#define TS_Q35_GDN_V 32                     // value columns per block
#define TS_Q35_GDN_KS (TS_Q35_GDN_HD + 8)   // BF16 row stride of the 128-wide operands
#define TS_Q35_GDN_CS (TS_Q35_GDN_C + 8)    // BF16 row stride of the 16-wide operands

struct TsQ35GdnChunk
{
    // [hi | lo][rows][stride]
    __nv_bfloat16 q[2][TS_Q35_GDN_C][TS_Q35_GDN_KS];   // queries, scaled by 1/sqrt(128)
    __nv_bfloat16 k[2][TS_Q35_GDN_C][TS_Q35_GDN_KS];
    union
    {
        __nv_bfloat16 s[2][TS_Q35_GDN_V][TS_Q35_GDN_KS];   // S0 as [v][k]
        struct
        {
            __nv_bfloat16 u[2][TS_Q35_GDN_V][TS_Q35_GDN_CS];   // U as [v][t]
            __nv_bfloat16 w[2][TS_Q35_GDN_V][TS_Q35_GDN_CS];   // U * e^(G_last - G_t)
        } uw;
    };
    __nv_bfloat16 p[2][TS_Q35_GDN_C][TS_Q35_GDN_CS];     // e^(G_t - G_s) q_t.k_s, s <= t
    __nv_bfloat16 inv[2][TS_Q35_GDN_C][TS_Q35_GDN_CS];   // (I + A)^-1
    __nv_bfloat16 d[2][TS_Q35_GDN_V][TS_Q35_GDN_CS];     // D as [v][t]
    float lower[TS_Q35_GDN_C][TS_Q35_GDN_C];              // A
    float prefix[TS_Q35_GDN_C], beta[TS_Q35_GDN_C];
};

// A BF16 value and its residual, written to plane 0 and plane `PLANE` elements later.
template <int PLANE>
__device__ __forceinline__ void ts_q35_put_split(__nv_bfloat16* m, int i, float x)
{
    const __nv_bfloat16 h = __float2bfloat16(x);
    m[i] = h;
    m[PLANE + i] = __float2bfloat16(x - __bfloat162float(h));
}

__device__ __forceinline__ uint32_t ts_q35_split_bf16x2(float x0, float x1, uint32_t& lo)
{
    const __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
    const float2 r = __bfloat1622float2(h);
    const __nv_bfloat162 l = __floats2bfloat162_rn(x0 - r.x, x1 - r.y);
    lo = *reinterpret_cast<const uint32_t*>(&l);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t ts_q35_ld32(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ void ts_q35_mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// c (16 x 8) += A B^T over K: A [16][K] and B [8][K] row-major BF16 (strides sa, sb) whose
// residual planes sit aPlane / bPlane elements later.
template <int K>
__device__ __forceinline__ void ts_q35_mma3(float (&c)[4], const __nv_bfloat16* a, int sa, int aPlane,
                                            const __nv_bfloat16* b, int sb, int bPlane)
{
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int k = 0; k < K; k += 16)
    {
        const __nv_bfloat16* pa = a + g * sa + k + 2 * t;
        const __nv_bfloat16* pb = b + g * sb + k + 2 * t;
        const uint32_t ah[4] = { ts_q35_ld32(pa), ts_q35_ld32(pa + 8 * sa), ts_q35_ld32(pa + 8), ts_q35_ld32(pa + 8 * sa + 8) };
        const uint32_t al[4] = { ts_q35_ld32(pa + aPlane), ts_q35_ld32(pa + aPlane + 8 * sa), ts_q35_ld32(pa + aPlane + 8),
                                 ts_q35_ld32(pa + aPlane + 8 * sa + 8) };
        const uint32_t bh0 = ts_q35_ld32(pb), bh1 = ts_q35_ld32(pb + 8);
        const uint32_t bl0 = ts_q35_ld32(pb + bPlane), bl1 = ts_q35_ld32(pb + bPlane + 8);
        ts_q35_mma_bf16(c, al, bh0, bh1);
        ts_q35_mma_bf16(c, ah, bl0, bl1);
        ts_q35_mma_bf16(c, ah, bh0, bh1);
    }
}

extern "C" __global__ void __launch_bounds__(256) ts_q35_gdn_chunk_f32(
    const float* __restrict__ qkv, int convDim, int hk, int hv, int rowsPerSeq,
    const float* __restrict__ g, const float* __restrict__ beta,
    float* const* __restrict__ states, float* __restrict__ out)
{
    constexpr int S = TS_Q35_GDN_HD, C = TS_Q35_GDN_C, V = TS_Q35_GDN_V, KS = TS_Q35_GDN_KS, CS = TS_Q35_GDN_CS;
    constexpr int QP = C * KS, SP = V * KS, CP = C * CS, VP = V * CS;   // residual plane offsets
    __shared__ __align__(16) TsQ35GdnChunk sm;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
    const int h = blockIdx.x / (S / V), v0 = (blockIdx.x % (S / V)) * V, seq = blockIdx.y, kh = h % hk;
    const long long rowBase = (long long)seq * rowsPerSeq;
    float* M = states[seq] + (long long)h * S * S;   // M[col][i] = S[i][col]

    // st[j]: this warp's 16 state rows x value columns 8j..8j+7 (mma accumulator layout:
    // element e is row gq + 8 (e >> 1), column 2 tq + (e & 1)).
    const int d0 = warp * 16;
    float st[V / 8][4];
#pragma unroll
    for (int j = 0; j < V / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            st[j][e] = M[(long long)(v0 + 8 * j + 2 * tq + (e & 1)) * S + d0 + gq + 8 * (e >> 1)];

    // Thread tid stages 8 dims of one token's q and k; the next chunk's are loaded while this one runs.
    const int ldT = tid >> 4, ldD = (tid & 15) * 8;
    float4 qn[2], kn[2];
    auto loadQk = [&](int t0)
    {
        qn[0] = qn[1] = kn[0] = kn[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
        if (t0 + ldT < rowsPerSeq)
        {
            const float* row = qkv + (rowBase + t0 + ldT) * convDim;
            qn[0] = *(const float4*)(row + kh * S + ldD);
            qn[1] = *(const float4*)(row + kh * S + ldD + 4);
            kn[0] = *(const float4*)(row + (hk + kh) * S + ldD);
            kn[1] = *(const float4*)(row + (hk + kh) * S + ldD + 4);
        }
    };
    loadQk(0);
    const float qScale = rsqrtf((float)S);
    for (int t0 = 0; t0 < rowsPerSeq; t0 += C)
    {
        const int valid = min(C, rowsPerSeq - t0);
        {
            const float qv[8] = { qn[0].x, qn[0].y, qn[0].z, qn[0].w, qn[1].x, qn[1].y, qn[1].z, qn[1].w };
            const float kv[8] = { kn[0].x, kn[0].y, kn[0].z, kn[0].w, kn[1].x, kn[1].y, kn[1].z, kn[1].w };
            uint32_t qh[4], ql[4], kh2[4], kl[4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
            {
                qh[i] = ts_q35_split_bf16x2(qv[2 * i] * qScale, qv[2 * i + 1] * qScale, ql[i]);
                kh2[i] = ts_q35_split_bf16x2(kv[2 * i], kv[2 * i + 1], kl[i]);
            }
            *(uint4*)&sm.q[0][ldT][ldD] = make_uint4(qh[0], qh[1], qh[2], qh[3]);
            *(uint4*)&sm.q[1][ldT][ldD] = make_uint4(ql[0], ql[1], ql[2], ql[3]);
            *(uint4*)&sm.k[0][ldT][ldD] = make_uint4(kh2[0], kh2[1], kh2[2], kh2[3]);
            *(uint4*)&sm.k[1][ldT][ldD] = make_uint4(kl[0], kl[1], kl[2], kl[3]);
        }
        if (warp == 0)
        {
            // Inclusive prefix of the log decays; tokens past the window decay by e^0.
            float gv = lane < valid ? g[(rowBase + t0 + lane) * hv + h] : 0.0f;
#pragma unroll
            for (int off = 1; off < C; off <<= 1)
            {
                const float x = __shfl_up_sync(0xffffffff, gv, off, 32);
                if (lane >= off)
                    gv += x;
            }
            if (lane < C)
            {
                sm.prefix[lane] = gv;
                sm.beta[lane] = lane < valid ? beta[(rowBase + t0 + lane) * hv + h] : 0.0f;
            }
        }
#pragma unroll
        for (int j = 0; j < V / 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                ts_q35_put_split<SP>(&sm.s[0][0][0], (8 * j + 2 * tq + (e & 1)) * KS + d0 + gq + 8 * (e >> 1), st[j][e]);
        __syncthreads();
        if (t0 + C < rowsPerSeq)
            loadQk(t0 + C);

        // Warps 0-3: e^G_t (Q S0) for value columns 8w..8w+7, kept for the output.
        // Warps 4-7: D for columns 8(w-4).., then one tile of A (w = 4, 5) or of P (w = 6, 7).
        float base[4];
        if (warp < 4)
        {
            float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ts_q35_mma3<S>(c, &sm.q[0][0][0], KS, QP, &sm.s[0][8 * warp][0], KS, SP);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                base[e] = expf(sm.prefix[gq + 8 * (e >> 1)]) * c[e];
        }
        else
        {
            const int j = warp - 4;
            float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ts_q35_mma3<S>(c, &sm.k[0][0][0], KS, QP, &sm.s[0][8 * j][0], KS, SP);
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int t = gq + 8 * (e >> 1), v = 8 * j + 2 * tq + (e & 1);
                const float vt = t < valid ? qkv[(rowBase + t0 + t) * convDim + 2LL * hk * S + (long long)h * S + v0 + v] : 0.0f;
                ts_q35_put_split<VP>(&sm.d[0][0][0], v * CS + t, sm.beta[t] * (vt - expf(sm.prefix[t]) * c[e]));
            }
            const bool gram = j < 2;
            const int s0 = (j & 1) * 8;
            float x[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ts_q35_mma3<S>(x, gram ? &sm.k[0][0][0] : &sm.q[0][0][0], KS, QP, &sm.k[0][s0][0], KS, QP);
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int t = gq + 8 * (e >> 1), s = s0 + 2 * tq + (e & 1);
                const float decay = s <= t ? expf(sm.prefix[t] - sm.prefix[s]) : 0.0f;
                if (gram)
                    sm.lower[t][s] = s < t ? sm.beta[t] * decay * x[e] : 0.0f;
                else
                    ts_q35_put_split<CP>(&sm.p[0][0][0], t * CS + s, decay * x[e]);
            }
        }
        __syncthreads();

        // (I + A)^-1 is unit lower-triangular: warp w forward-substitutes rows w and w + 8, lane = column.
        for (int row = warp; row < C; row += 8)
        {
            float x = lane == row ? 1.0f : lane < row ? -sm.lower[row][lane] : 0.0f;
            for (int pivot = row - 1; pivot > 0; --pivot)
            {
                const float xp = __shfl_sync(0xffffffff, x, pivot, 32);
                if (lane < pivot)
                    x -= xp * sm.lower[pivot][lane];
            }
            if (lane < C)
                ts_q35_put_split<CP>(&sm.inv[0][0][0], row * CS + lane, x);
        }
        __syncthreads();

        // U = (I + A)^-1 D, and U decayed to the chunk's end (S0 is consumed: U and W reuse it).
        const float gLast = sm.prefix[C - 1];
        if (warp >= 4)
        {
            const int j = warp - 4;
            float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ts_q35_mma3<C>(c, &sm.inv[0][0][0], CS, CP, &sm.d[0][8 * j][0], CS, VP);
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int t = gq + 8 * (e >> 1), v = 8 * j + 2 * tq + (e & 1);
                ts_q35_put_split<VP>(&sm.uw.u[0][0][0], v * CS + t, c[e]);
                ts_q35_put_split<VP>(&sm.uw.w[0][0][0], v * CS + t, c[e] * expf(gLast - sm.prefix[t]));
            }
        }
        __syncthreads();

        if (warp < 4)
        {
            ts_q35_mma3<C>(base, &sm.p[0][0][0], CS, CP, &sm.uw.u[0][8 * warp][0], CS, VP);
#pragma unroll
            for (int half = 0; half < 2; ++half)
            {
                const int t = gq + 8 * half;
                if (t < valid)
                    *(float2*)&out[((rowBase + t0 + t) * hv + h) * S + v0 + 8 * warp + 2 * tq] =
                        make_float2(base[2 * half], base[2 * half + 1]);
            }
        }
        // S = e^G_last S0 + K^T W: the A operand is K transposed, gathered from sm.k pairwise.
        {
            const unsigned short* kr = reinterpret_cast<const unsigned short*>(&sm.k[0][0][0]);
            auto pair = [&](int plane, int t, int d)
            {
                const unsigned short* p = kr + plane * QP + t * KS + d;
                return (uint32_t)p[0] | ((uint32_t)p[KS] << 16);
            };
            uint32_t ah[4], al[4];
#pragma unroll
            for (int r = 0; r < 4; ++r)
            {
                const int t = 2 * tq + 8 * (r >> 1), d = d0 + gq + 8 * (r & 1);
                ah[r] = pair(0, t, d);
                al[r] = pair(1, t, d);
            }
            const float decay = expf(gLast);
#pragma unroll
            for (int j = 0; j < V / 8; ++j)
            {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    st[j][e] *= decay;
                const __nv_bfloat16* pb = &sm.uw.w[0][8 * j + gq][2 * tq];
                const uint32_t bh0 = ts_q35_ld32(pb), bh1 = ts_q35_ld32(pb + 8);
                const uint32_t bl0 = ts_q35_ld32(pb + VP), bl1 = ts_q35_ld32(pb + VP + 8);
                ts_q35_mma_bf16(st[j], al, bh0, bh1);
                ts_q35_mma_bf16(st[j], ah, bl0, bl1);
                ts_q35_mma_bf16(st[j], ah, bh0, bh1);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < V / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            M[(long long)(v0 + 8 * j + 2 * tq + (e & 1)) * S + d0 + gq + 8 * (e >> 1)] = st[j][e];
}

#include "tensorsharp_q35_gdn_prepared.cuh"

// ---------------------------------------------------------------------------
// Gated RMS norm of the delta-rule output: y = rmsnorm(o_head) * ssmNorm * silu(z), one warp per
// (row, v-head); z is the packed projection's z slice.
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_q35_gdn_out_f32(
    const float* __restrict__ o, const float* __restrict__ packed, int packedWidth, int convDim, int rows, int hv,
    const float* __restrict__ ssmNorm, float eps, float* __restrict__ y)
{
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= (long long)rows * hv)
        return;
    const long long row = warp / hv;
    const int h = (int)(warp % hv);
    const float* src = o + (row * hv + h) * TS_Q35_GDN_HD;
    float v[4], ss = 0.0f;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        v[j] = src[lane + 32 * j];
        ss += v[j] * v[j];
    }
    ss = ts_q35_warp_sum(ss);
    const float scale = rsqrtf(ss / (float)TS_Q35_GDN_HD + eps);
    const float* z = packed + row * packedWidth + convDim + (long long)h * TS_Q35_GDN_HD;
    float* dst = y + (row * hv + h) * TS_Q35_GDN_HD;
#pragma unroll
    for (int j = 0; j < 4; ++j)
    {
        const int i = lane + 32 * j;
        dst[i] = v[j] * scale * ssmNorm[i] * ts_q35_silu(z[i]);
    }
}

// ---------------------------------------------------------------------------
// Causal attention over the q8_0 caches, GQA. One block of 4 warps owns one KV head and up to
// 64 query rows of a sequence, row r being (token r / G, query head kvHead * G + r % G); row
// r attends keys [0, keyEnd[token]). Key tiles of 32 are dequantized once into F16 shared
// memory (d * q, rounded to F16 as ggml's q8_0 -> F16 conversion does) and served to every
// warp; Q is rounded to F16, scores and the online softmax stay F32, P is rounded to F16 for
// the value product, accumulation F32 (mma.m16n8k16).
//   SPLIT = false: blockIdx.x is the 64-row tile; the block sweeps every key its rows see and
//   writes out = softmax(QK^T * scale) V * sigmoid(gate).
//   SPLIT = true (decode / verify, at most 64 rows per sequence and KV head): blockIdx.x is a
//   key range of `chunk` keys; the block writes unnormalized partials (acc[256], ln-domain max,
//   sum) that ts_q35_attn_combine_f32 merges in split order.
// ---------------------------------------------------------------------------
#define TS_Q35_ATTN_KT 32
#define TS_Q35_ATTN_ROW (TS_Q35_HD + 8)   // F16 tile row stride: 528 bytes, bank-spread

__device__ __forceinline__ void ts_q35_mma_f16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ __forceinline__ uint32_t ts_q35_pack_h2(float lo, float hi)
{
    const half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void ts_q35_ldsm_x2_t(uint32_t& b0, uint32_t& b1, const half* row)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1) : "r"(ts_q35_smem_u32(row)));
#else
    b0 = b1 = 0;
    __trap();
#endif
}

// 32 keys x 256 values of one head from a q8_0 cache into F16 rows; keys past `limit` are zero
// (finite, so their masked probabilities times V stay zero).
__device__ __forceinline__ void ts_q35_dequant_tile(const uint8_t* __restrict__ head, long long keyStride, int firstKey, int limit, half* tile)
{
    for (int i = threadIdx.x; i < TS_Q35_ATTN_KT * 8; i += blockDim.x)
    {
        const int key = i >> 3, b = i & 7;
        half* dst = tile + key * TS_Q35_ATTN_ROW + b * 32;
        if (firstKey + key >= limit)
        {
#pragma unroll
            for (int j = 0; j < 4; ++j)
                *(uint4*)(dst + 8 * j) = make_uint4(0, 0, 0, 0);
            continue;
        }
        const uint8_t* row = head + (long long)(firstKey + key) * keyStride;
        const float d = __half2float(*(const half*)(row + TS_Q35_Q8_SCALES + 2 * b));
        const uint4* codes = (const uint4*)(row + b * 32);
#pragma unroll
        for (int c = 0; c < 2; ++c)
        {
            const uint4 w = codes[c];
            const uint32_t words[4] = { w.x, w.y, w.z, w.w };
#pragma unroll
            for (int p = 0; p < 2; ++p)
            {
                uint32_t h[4];
#pragma unroll
                for (int e = 0; e < 4; ++e)
                {
                    const uint32_t x = words[p * 2 + e / 2] >> (16 * (e & 1));
                    h[e] = ts_q35_pack_h2((float)(int8_t)(x & 0xFF) * d, (float)(int8_t)((x >> 8) & 0xFF) * d);
                }
                *(uint4*)(dst + 16 * c + 8 * p) = make_uint4(h[0], h[1], h[2], h[3]);
            }
        }
    }
}

template <bool SPLIT>
__device__ __forceinline__ void ts_q35_attention(
    const float* __restrict__ q, const float* __restrict__ gate, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, int chunk, int splits,
    float* __restrict__ partial, float* __restrict__ out)
{
    __shared__ __align__(16) half ks[TS_Q35_ATTN_KT * TS_Q35_ATTN_ROW];
    __shared__ __align__(16) half vs[TS_Q35_ATTN_KT * TS_Q35_ATTN_ROW];
    const int kvh = blockIdx.y, seq = blockIdx.z;
    const int G = nHead / nKv;
    const int R = rowsPerSeq * G;
    // 16 query rows per warp: a block takes blockDim / 2 rows (prefill: 8 warps, so every key
    // tile it dequantizes serves 128 rows).
    const int blockRows = blockDim.x >> 1;
    const int rowBase = SPLIT ? 0 : blockIdx.x * blockRows;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, qd = lane & 3;
    const int* ends = keyEnd + (long long)seq * rowsPerSeq;

    // Keys the block's rows see: up to the last token's end.
    const int lastRow = min(rowBase + blockRows, R) - 1;
    int kHi = ends[lastRow / G];
    int kLo = 0;
    if (SPLIT)
    {
        kLo = blockIdx.x * chunk;
        kHi = min(kHi, kLo + chunk);
    }
    // Cache blocks hold this head's TS_Q35_KV_BLOCK keys contiguously, and a 32-key tile never
    // leaves its block.
    const long long keyStride = TS_Q35_Q8_ROW;
    uint8_t* const* blocks = blockTable + (long long)seq * maxBlocks;

    // This warp's rows: r0 = rowBase + warp * 16 + g, r1 = r0 + 8.
    const int r0 = rowBase + warp * 16 + g, r1 = r0 + 8;
    const bool live0 = r0 < R, live1 = r1 < R;
    const int end0 = live0 ? ends[r0 / G] : 0, end1 = live1 ? ends[r1 / G] : 0;
    auto qptr = [&](int r) {
        const int t = r / G, h = kvh * G + r % G;
        return q + (((long long)seq * rowsPerSeq + t) * nHead + h) * TS_Q35_HD;
    };
    const bool warpLive = rowBase + warp * 16 < R;

    uint32_t qa[16][4];
    {
        const float* q0 = live0 ? qptr(r0) : nullptr;
        const float* q1 = live1 ? qptr(r1) : nullptr;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk)
        {
            const int c = kk * 16 + qd * 2;
            qa[kk][0] = q0 ? ts_q35_pack_h2(q0[c], q0[c + 1]) : 0u;
            qa[kk][1] = q1 ? ts_q35_pack_h2(q1[c], q1[c + 1]) : 0u;
            qa[kk][2] = q0 ? ts_q35_pack_h2(q0[c + 8], q0[c + 9]) : 0u;
            qa[kk][3] = q1 ? ts_q35_pack_h2(q1[c + 8], q1[c + 9]) : 0u;
        }
    }
    float acc[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float log2Scale = scale * 1.4426950408889634f;

    for (int first = kLo; first < kHi; first += TS_Q35_ATTN_KT)
    {
        __syncthreads();
        const uint8_t* tileBase = blocks[first / TS_Q35_KV_BLOCK]
                                + ((long long)kvh * TS_Q35_KV_BLOCK + first % TS_Q35_KV_BLOCK) * TS_Q35_Q8_ROW;
        ts_q35_dequant_tile(tileBase + kOff, keyStride, 0, kHi - first, ks);
        ts_q35_dequant_tile(tileBase + vOff, keyStride, 0, kHi - first, vs);
        __syncthreads();
        if (!warpLive)
            continue;
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
            const half* kb = ks + (j * 8 + g) * TS_Q35_ATTN_ROW + qd * 2;
#pragma unroll
            for (int kk = 0; kk < 16; ++kk)
                ts_q35_mma_f16(s[j], qa[kk], *(const uint32_t*)(kb + kk * 16), *(const uint32_t*)(kb + kk * 16 + 8));
        }
        float t0 = m0, t1 = m1;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int key = first + j * 8 + qd * 2;
            s[j][0] = key < end0 ? s[j][0] * log2Scale : -INFINITY;
            s[j][1] = key + 1 < end0 ? s[j][1] * log2Scale : -INFINITY;
            s[j][2] = key < end1 ? s[j][2] * log2Scale : -INFINITY;
            s[j][3] = key + 1 < end1 ? s[j][3] * log2Scale : -INFINITY;
            t0 = fmaxf(t0, fmaxf(s[j][0], s[j][1]));
            t1 = fmaxf(t1, fmaxf(s[j][2], s[j][3]));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1)
        {
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, off));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffff, t1, off));
        }
        // A row may see no key of this tile (causal tail, split range): keep its state.
        const float c0 = t0 == -INFINITY ? 1.0f : exp2f(m0 - t0);
        const float c1 = t1 == -INFINITY ? 1.0f : exp2f(m1 - t1);
        m0 = t0;
        m1 = t1;
        l0 *= c0;
        l1 *= c1;
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
            acc[n][0] *= c0; acc[n][1] *= c0;
            acc[n][2] *= c1; acc[n][3] *= c1;
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            s[j][0] = t0 == -INFINITY ? 0.0f : exp2f(s[j][0] - t0);
            s[j][1] = t0 == -INFINITY ? 0.0f : exp2f(s[j][1] - t0);
            s[j][2] = t1 == -INFINITY ? 0.0f : exp2f(s[j][2] - t1);
            s[j][3] = t1 == -INFINITY ? 0.0f : exp2f(s[j][3] - t1);
            l0 += s[j][0] + s[j][1];
            l1 += s[j][2] + s[j][3];
        }
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
        {
            const uint32_t pa[4] = {
                ts_q35_pack_h2(s[2 * kk][0], s[2 * kk][1]), ts_q35_pack_h2(s[2 * kk][2], s[2 * kk][3]),
                ts_q35_pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]), ts_q35_pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]) };
            const half* vrow = vs + (kk * 16 + (lane & 15)) * TS_Q35_ATTN_ROW;
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                uint32_t b0, b1;
                ts_q35_ldsm_x2_t(b0, b1, vrow + n * 8);
                ts_q35_mma_f16(acc[n], pa, b0, b1);
            }
        }
    }
    if (!warpLive)
        return;
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
    {
        l0 += __shfl_xor_sync(0xffffffff, l0, off);
        l1 += __shfl_xor_sync(0xffffffff, l1, off);
    }
#pragma unroll
    for (int half_ = 0; half_ < 2; ++half_)
    {
        const int r = half_ ? r1 : r0;
        if (r >= R)
            continue;
        const int t = r / G, h = kvh * G + r % G;
        const long long base = ((long long)seq * rowsPerSeq + t) * nHead + h;
        if (SPLIT)
        {
            float* p = partial + (base * splits + blockIdx.x) * (TS_Q35_HD + 2);
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                const int x = n * 8 + qd * 2;
                p[x] = acc[n][half_ * 2];
                p[x + 1] = acc[n][half_ * 2 + 1];
            }
            if (qd == 0)
            {
                const float m = half_ ? m1 : m0;
                p[TS_Q35_HD] = m == -INFINITY ? -INFINITY : m * 0.6931471805599453f;   // log2 -> ln domain
                p[TS_Q35_HD + 1] = half_ ? l1 : l0;
            }
        }
        else
        {
            const float l = half_ ? l1 : l0;
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            float* o = out + base * TS_Q35_HD;
            const float* gt = gate + base * TS_Q35_HD;
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                const int x = n * 8 + qd * 2;
                o[x] = acc[n][half_ * 2] * inv * ts_q35_sigmoid(gt[x]);
                o[x + 1] = acc[n][half_ * 2 + 1] * inv * ts_q35_sigmoid(gt[x + 1]);
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) ts_q35_attn_q8_f32(
    const float* __restrict__ q, const float* __restrict__ gate, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, float* __restrict__ out)
{
    ts_q35_attention<false>(q, gate, keyEnd, blockTable, maxBlocks, kOff, vOff, rowsPerSeq, nHead, nKv, scale, 0, 1, nullptr, out);
}

extern "C" __global__ void __launch_bounds__(128) ts_q35_attn_q8_split_f32(
    const float* __restrict__ q, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, int chunk, float* __restrict__ partial)
{
    ts_q35_attention<true>(q, nullptr, keyEnd, blockTable, maxBlocks, kOff, vOff, rowsPerSeq, nHead, nKv, scale, chunk,
                           gridDim.x, partial, nullptr);
}

// Merge the split partials of every (row, head) in split order and apply the gate:
// out = (sum_s acc_s e^(m_s - M)) / (sum_s l_s e^(m_s - M)) * sigmoid(gate). One block per
// (row, head), 256 threads (one per value).
extern "C" __global__ void ts_q35_attn_combine_f32(
    const float* __restrict__ partial, const float* __restrict__ gate, int splits, float* __restrict__ out)
{
    const long long base = blockIdx.x;
    const float* p = partial + base * splits * (TS_Q35_HD + 2);
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s)
        mx = fmaxf(mx, p[s * (TS_Q35_HD + 2) + TS_Q35_HD]);
    float num = 0.0f, den = 0.0f;
    const int x = threadIdx.x;
    for (int s = 0; s < splits; ++s)
    {
        const float* ps = p + s * (TS_Q35_HD + 2);
        const float w = mx == -INFINITY ? 0.0f : expf(ps[TS_Q35_HD] - mx);
        num = fmaf(ps[x], w, num);
        den = fmaf(ps[TS_Q35_HD + 1], w, den);
    }
    out[base * TS_Q35_HD + x] = (den > 0.0f ? num / den : 0.0f) * ts_q35_sigmoid(gate[base * TS_Q35_HD + x]);
}

// ---------------------------------------------------------------------------
// Prefill attention: the rows, grid and output of ts_q35_attn_q8_f32, about 1.7 times as fast. K
// and V tiles stream through a two-stage cp.async pipeline of raw cache rows and are dequantized
// to F16 in shared memory (the values ts_q35_dequant_tile gives). The mma accumulate in F16, which
// runs at twice the rate of F32 accumulation on sm_120, over short runs only: each 16-value step
// of QK^T (Q pre-scaled to log2 units) is rounded once and summed in F32, and PV over one 32-key
// tile (probabilities at most 1) is rounded twice before it joins the F32 accumulators. Against
// exact q8_0 attention on Qwen3.8's shape (eng/validation/tsg-q35-attn-bench.cu) its error is
// that of ts_q35_attn_q8_f32.
// ---------------------------------------------------------------------------
#define TS_Q35_AI_KT 32
#define TS_Q35_AI_TILE (TS_Q35_AI_KT * TS_Q35_Q8_ROW)   // bytes of one head's 32 cache rows
#define TS_Q35_AI_ROWS 128                              // query rows per block (8 warps)
#define TS_Q35_AH_CHAIN 1                               // QK k-steps per F16 accumulation run

// m16n8k16 with F16 accumulation (twice the rate of F32 accumulation on sm_120).
__device__ __forceinline__ void ts_q35_mma_f16h(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_q35_ldsm_x4_t(uint32_t (&r)[4], const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ts_q35_smem_u32(p)));
#else
    r[0] = r[1] = r[2] = r[3] = 0;
    __trap();
#endif
}

// Four int8 codes times an F16 scale as two half2, each value the F16 rounding of code * d (as
// ts_q35_dequant_tile computes it): 1024 + (code + 128) is exact in F16, so subtracting 1152
// restores the code exactly before the one rounding multiply.
__device__ __forceinline__ void ts_q35_i8x4_to_h2(uint32_t codes, half2 d, half2& lo, half2& hi)
{
    const uint32_t u = codes ^ 0x80808080u;
    const uint32_t a = __byte_perm(u, 0x64646464u, 0x7170), b = __byte_perm(u, 0x64646464u, 0x7372);
    const half2 bias = __float2half2_rn(1152.0f);
    lo = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&a), bias), d);
    hi = __hmul2(__hsub2(*reinterpret_cast<const half2*>(&b), bias), d);
}

struct TsQ35AttnPrefill
{
    uint8_t raw[2][2][TS_Q35_AI_TILE];                // [stage][K, V] cache rows as stored
    half k[TS_Q35_AI_KT * TS_Q35_ATTN_ROW];           // this tile's K in F16
    half v[TS_Q35_AI_KT * TS_Q35_ATTN_ROW];           // this tile's V in F16
};

__device__ __forceinline__ void ts_q35_tile_to_h(const uint8_t* raw, half* dst, int tid)
{
    const int key = tid & 31, b = tid >> 5;
    const uint8_t* row = raw + key * TS_Q35_Q8_ROW;
    const half2 d = __half2half2(*(const half*)(row + TS_Q35_Q8_SCALES + 2 * b));
    const uint4* codes = (const uint4*)(row + b * 32);
    uint4* out = (uint4*)(dst + key * TS_Q35_ATTN_ROW + b * 32);
#pragma unroll
    for (int c = 0; c < 2; ++c)
    {
        const uint4 w = codes[c];
        half2 h[8];
        ts_q35_i8x4_to_h2(w.x, d, h[0], h[1]);
        ts_q35_i8x4_to_h2(w.y, d, h[2], h[3]);
        ts_q35_i8x4_to_h2(w.z, d, h[4], h[5]);
        ts_q35_i8x4_to_h2(w.w, d, h[6], h[7]);
        out[2 * c] = *reinterpret_cast<const uint4*>(&h[0]);
        out[2 * c + 1] = *reinterpret_cast<const uint4*>(&h[4]);
    }
}

// splits > 1 (a grid too small to fill the device): block x = (row tile, key split); each split
// takes a 32-key-aligned share of its rows' keys (chunk 0), or keys [split * chunk, (split + 1) *
// chunk) (chunk > 0: the fixed boundaries of decode and verify steps, see
// ts_q35_attn_q8_split_pipe_f32), and writes unnormalized partials (acc, ln-domain max, sum) that
// ts_q35_attn_combine_f32 merges.
__device__ __forceinline__ void ts_q35_attn_pipe(
    const float* __restrict__ q, const float* __restrict__ gate, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, int splits, int chunk, int splitBase, int splitCount,
    float* __restrict__ partial, float* __restrict__ out)
{
    extern __shared__ __align__(16) uint8_t ts_q35_attn_smem[];
    TsQ35AttnPrefill& sm = *reinterpret_cast<TsQ35AttnPrefill*>(ts_q35_attn_smem);
    const int kvh = blockIdx.y, seq = blockIdx.z;
    const int G = nHead / nKv;
    const int R = rowsPerSeq * G;
    // Block x = (row tile, split splitBase + x % splitCount): a launch can take a subset of the splits
    // (the key range several sequences share, or each sequence's own); partials use all splits' stride.
    const int split = splitBase + (int)(blockIdx.x % splitCount);
    const int rowBase = (blockIdx.x / splitCount) * TS_Q35_AI_ROWS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
    const int* ends = keyEnd + (long long)seq * rowsPerSeq;
    const int kAll = ends[(min(rowBase + TS_Q35_AI_ROWS, R) - 1) / G];
    const int share = chunk > 0 ? chunk
                    : ((kAll + splits - 1) / splits + TS_Q35_AI_KT - 1) / TS_Q35_AI_KT * TS_Q35_AI_KT;
    const int kLo = min(kAll, split * share), kHi = min(kAll, kLo + share);
    uint8_t* const* blocks = blockTable + (long long)seq * maxBlocks;

    const int r0 = rowBase + warp * 16 + g, r1 = r0 + 8;
    const bool live0 = r0 < R, live1 = r1 < R;
    const int end0 = live0 ? ends[r0 / G] : 0, end1 = live1 ? ends[r1 / G] : 0;
    const bool warpLive = rowBase + warp * 16 < R;

    auto load = [&](int st, int first) {
        const long long at = ((long long)kvh * TS_Q35_KV_BLOCK + first % TS_Q35_KV_BLOCK) * TS_Q35_Q8_ROW;
        const uint8_t* base = blocks[first / TS_Q35_KV_BLOCK] + at;
        for (int i = tid; i < 2 * TS_Q35_AI_TILE / 16; i += blockDim.x)
        {
            const int isV = i >= TS_Q35_AI_TILE / 16, c = isV ? i - TS_Q35_AI_TILE / 16 : i;
            const bool ok = first + c / (TS_Q35_Q8_ROW / 16) < kHi;
            ts_q35_cp16(sm.raw[st][isV] + c * 16, base + (isV ? vOff : kOff) + (ok ? c * 16 : 0), ok);
        }
    };
    if (kHi > kLo)
        load(0, kLo);
    ts_q35_cp_commit();

    const float log2Scale = scale * 1.4426950408889634f;
    uint32_t qa[16][4];
    {
        const int t0 = r0 / G, t1 = r1 / G;
        const float* q0 = live0 ? q + (((long long)seq * rowsPerSeq + t0) * nHead + kvh * G + r0 % G) * TS_Q35_HD : nullptr;
        const float* q1 = live1 ? q + (((long long)seq * rowsPerSeq + t1) * nHead + kvh * G + r1 % G) * TS_Q35_HD : nullptr;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk)
        {
            const int c = kk * 16 + qd * 2;
            qa[kk][0] = q0 ? ts_q35_pack_h2(q0[c] * log2Scale, q0[c + 1] * log2Scale) : 0u;
            qa[kk][1] = q1 ? ts_q35_pack_h2(q1[c] * log2Scale, q1[c + 1] * log2Scale) : 0u;
            qa[kk][2] = q0 ? ts_q35_pack_h2(q0[c + 8] * log2Scale, q0[c + 9] * log2Scale) : 0u;
            qa[kk][3] = q1 ? ts_q35_pack_h2(q1[c + 8] * log2Scale, q1[c + 9] * log2Scale) : 0u;
        }
    }

    float acc[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int first = kLo, t = 0; first < kHi; first += TS_Q35_AI_KT, ++t)
    {
        const int st = t & 1;
        ts_q35_cp_wait<0>();
        __syncthreads();   // this tile landed; every warp is done with the previous one
        if (first + TS_Q35_AI_KT < kHi)
            load(st ^ 1, first + TS_Q35_AI_KT);
        ts_q35_cp_commit();
        ts_q35_tile_to_h(sm.raw[st][0], sm.k, tid);
        ts_q35_tile_to_h(sm.raw[st][1], sm.v, tid);
        __syncthreads();   // this tile's K and V are in F16
        if (!warpLive)
            continue;

        float s[4][4];
        {
            // B fragments of 2 n-tiles x 1 k-step per ldmatrix.x4 (rows = keys).
            const half* kb = sm.k + ((lane & 7) + ((lane >> 4) << 3)) * TS_Q35_ATTN_ROW + ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int np = 0; np < 2; ++np)
            {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    s[2 * np][e] = s[2 * np + 1][e] = 0.0f;
#pragma unroll
                for (int ch = 0; ch < 16 / TS_Q35_AH_CHAIN; ++ch)
                {
                    uint32_t h0[2] = { 0u, 0u }, h1[2] = { 0u, 0u };
#pragma unroll
                    for (int i = 0; i < TS_Q35_AH_CHAIN; ++i)
                    {
                        const int kk = ch * TS_Q35_AH_CHAIN + i;
                        uint32_t b[4];
                        ts_q35_ldsm_x4(b, kb + np * 16 * TS_Q35_ATTN_ROW + kk * 16);
                        ts_q35_mma_f16h(h0, qa[kk], b[0], b[1]);
                        ts_q35_mma_f16h(h1, qa[kk], b[2], b[3]);
                    }
                    const float2 a0 = __half22float2(*reinterpret_cast<const half2*>(&h0[0]));
                    const float2 a1 = __half22float2(*reinterpret_cast<const half2*>(&h0[1]));
                    const float2 c0 = __half22float2(*reinterpret_cast<const half2*>(&h1[0]));
                    const float2 c1 = __half22float2(*reinterpret_cast<const half2*>(&h1[1]));
                    s[2 * np][0] += a0.x; s[2 * np][1] += a0.y; s[2 * np][2] += a1.x; s[2 * np][3] += a1.y;
                    s[2 * np + 1][0] += c0.x; s[2 * np + 1][1] += c0.y; s[2 * np + 1][2] += c1.x; s[2 * np + 1][3] += c1.y;
                }
            }
        }

        float t0 = m0, t1 = m1;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int key = first + j * 8 + qd * 2;
            s[j][0] = key < end0 ? s[j][0] : -INFINITY;
            s[j][1] = key + 1 < end0 ? s[j][1] : -INFINITY;
            s[j][2] = key < end1 ? s[j][2] : -INFINITY;
            s[j][3] = key + 1 < end1 ? s[j][3] : -INFINITY;
            t0 = fmaxf(t0, fmaxf(s[j][0], s[j][1]));
            t1 = fmaxf(t1, fmaxf(s[j][2], s[j][3]));
        }
#pragma unroll
        for (int off = 1; off < 4; off <<= 1)
        {
            t0 = fmaxf(t0, __shfl_xor_sync(0xffffffff, t0, off));
            t1 = fmaxf(t1, __shfl_xor_sync(0xffffffff, t1, off));
        }
        const float c0 = t0 == -INFINITY ? 1.0f : exp2f(m0 - t0);
        const float c1 = t1 == -INFINITY ? 1.0f : exp2f(m1 - t1);
        m0 = t0;
        m1 = t1;
        l0 *= c0;
        l1 *= c1;
        if (!__all_sync(0xffffffff, c0 == 1.0f && c1 == 1.0f))
        {
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                acc[n][0] *= c0; acc[n][1] *= c0;
                acc[n][2] *= c1; acc[n][3] *= c1;
            }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            s[j][0] = t0 == -INFINITY ? 0.0f : exp2f(s[j][0] - t0);
            s[j][1] = t0 == -INFINITY ? 0.0f : exp2f(s[j][1] - t0);
            s[j][2] = t1 == -INFINITY ? 0.0f : exp2f(s[j][2] - t1);
            s[j][3] = t1 == -INFINITY ? 0.0f : exp2f(s[j][3] - t1);
            l0 += s[j][0] + s[j][1];
            l1 += s[j][2] + s[j][3];
        }
        uint32_t pa[2][4];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
        {
            pa[kk][0] = ts_q35_pack_h2(s[2 * kk][0], s[2 * kk][1]);
            pa[kk][1] = ts_q35_pack_h2(s[2 * kk][2], s[2 * kk][3]);
            pa[kk][2] = ts_q35_pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[kk][3] = ts_q35_pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
        }
        const half* vrow = sm.v + ((lane & 7) + (((lane >> 3) & 1) << 3)) * TS_Q35_ATTN_ROW + ((lane >> 4) << 3);
#pragma unroll
        for (int dn = 0; dn < 16; ++dn)
        {
            uint32_t h0[2] = { 0u, 0u }, h1[2] = { 0u, 0u };
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
            {
                uint32_t b[4];
                ts_q35_ldsm_x4_t(b, vrow + kk * 16 * TS_Q35_ATTN_ROW + dn * 16);
                ts_q35_mma_f16h(h0, pa[kk], b[0], b[1]);
                ts_q35_mma_f16h(h1, pa[kk], b[2], b[3]);
            }
            const float2 a0 = __half22float2(*reinterpret_cast<const half2*>(&h0[0]));
            const float2 a1 = __half22float2(*reinterpret_cast<const half2*>(&h0[1]));
            const float2 b0 = __half22float2(*reinterpret_cast<const half2*>(&h1[0]));
            const float2 b1 = __half22float2(*reinterpret_cast<const half2*>(&h1[1]));
            acc[2 * dn][0] += a0.x; acc[2 * dn][1] += a0.y; acc[2 * dn][2] += a1.x; acc[2 * dn][3] += a1.y;
            acc[2 * dn + 1][0] += b0.x; acc[2 * dn + 1][1] += b0.y; acc[2 * dn + 1][2] += b1.x; acc[2 * dn + 1][3] += b1.y;
        }
    }
    ts_q35_cp_wait<0>();
    if (!warpLive)
        return;
#pragma unroll
    for (int off = 1; off < 4; off <<= 1)
    {
        l0 += __shfl_xor_sync(0xffffffff, l0, off);
        l1 += __shfl_xor_sync(0xffffffff, l1, off);
    }
#pragma unroll
    for (int half_ = 0; half_ < 2; ++half_)
    {
        const int r = half_ ? r1 : r0;
        if (r >= R)
            continue;
        const int t = r / G, h = kvh * G + r % G;
        const long long base = ((long long)seq * rowsPerSeq + t) * nHead + h;
        const float l = half_ ? l1 : l0;
        if (splits > 1)
        {
            float* p = partial + (base * splits + split) * (TS_Q35_HD + 2);
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                const int x = n * 8 + qd * 2;
                p[x] = acc[n][half_ * 2];
                p[x + 1] = acc[n][half_ * 2 + 1];
            }
            if (qd == 0)
            {
                const float m = half_ ? m1 : m0;
                p[TS_Q35_HD] = m == -INFINITY ? -INFINITY : m * 0.6931471805599453f;   // log2 -> ln domain
                p[TS_Q35_HD + 1] = l;
            }
            continue;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* o = out + base * TS_Q35_HD;
        const float* gt = gate + base * TS_Q35_HD;
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
            const int x = n * 8 + qd * 2;
            o[x] = acc[n][half_ * 2] * inv * ts_q35_sigmoid(gt[x]);
            o[x + 1] = acc[n][half_ * 2 + 1] * inv * ts_q35_sigmoid(gt[x + 1]);
        }
    }
}


extern "C" __global__ void __launch_bounds__(256) ts_q35_attn_prefill_f32(
    const float* __restrict__ q, const float* __restrict__ gate, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out)
{
    ts_q35_attn_pipe(q, gate, keyEnd, blockTable, maxBlocks, kOff, vOff, rowsPerSeq, nHead, nKv, scale, splits, 0,
                     0, splits, partial, out);
}

// Decode and verify attention (at most 128 query rows per KV head): ts_q35_attn_q8_split_f32's key
// ranges (fixed boundaries every chunk keys, so a row's sums never depend on the other rows of the
// step) on the prefill kernel's two-stage cp.async pipeline. The synchronous split kernel reads a
// long q8_0 history at about half the device bandwidth (one 32-key tile in flight per 4-warp block).
// splits == 1 writes the gated output; otherwise partials for ts_q35_attn_combine_f32. Splits
// [splitBase, splitBase + splitCount) only: sequences that share their first keys (clones of one cached
// prefix share its cache blocks) run those splits once with every sequence's rows as one sequence
// (rows are [sequence][token] in every buffer), then their own splits separately.
extern "C" __global__ void __launch_bounds__(256) ts_q35_attn_q8_split_pipe_f32(
    const float* __restrict__ q, const float* __restrict__ gate, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, int splits, int chunk, int splitBase, int splitCount,
    float* __restrict__ partial, float* __restrict__ out)
{
    ts_q35_attn_pipe(q, gate, keyEnd, blockTable, maxBlocks, kOff, vOff, rowsPerSeq, nHead, nKv, scale, splits, chunk,
                     splitBase, splitCount, partial, out);
}

// Dynamic shared memory of ts_q35_attn_prefill_f32, for the host.
extern "C" __global__ void ts_q35_attn_smem_bytes(int* out)
{
    out[0] = (int)sizeof(TsQ35AttnPrefill);
}

#include "tensorsharp_q35_attn8.cuh"

// ---------------------------------------------------------------------------
// Q6_K weights x q8_1 activations on the int8 tensor cores (the drafter's Q6_K matrices; the LM
// head runs the tiled kernels below). Q6_K superblock (256 values, 210 bytes): ql[128] | qh[64] |
// scales int8[16] | half d; value 128h + 32m + l (h < 2, m < 4, l < 32) is (ql nibble | 2 qh bits)
// - 32 with scale index 8h + 2m + l / 16. One mma.m16n8k16.s8 per 16-value Q6_K sub-block, its exact integer sum scaled by the
// sub-block's weight scale (d * sc) and the activation's 32-value scale and accumulated in F32 -
// the GEMV's integer products, summed in a different order. Block: 4 warps x 16 weight rows
// against NT * 8 activation rows; per 256-value superblock the activations and each warp's
// decoded weight rows (int8, value - 32) are staged in shared memory, rows padded to 68 words so
// the fragment loads are conflict-free. Grid (row tiles, K splits): with more than one split
// every block adds its share atomically into a zeroed y.
// ---------------------------------------------------------------------------
#define TS_Q35_Q6K_ROW (256 + 16)

__device__ __forceinline__ void ts_q35_mma_s8(int (&c)[4], int a0, int a1, int b0)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5}, {%6}, {%7,%8,%9,%10};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a0), "r"(a1), "r"(b0), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    __trap();
#endif
}

template <int NT>
__device__ __forceinline__ void ts_q35_gemm_q6k(
    const uint8_t* __restrict__ w, long long rowBytes, int outDim, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xd, int n, float* __restrict__ y)
{
    constexpr int NA = NT * 8;
    __shared__ __align__(16) int8_t xs[NA][TS_Q35_Q6K_ROW];
    __shared__ float xds[NA][8];
    __shared__ __align__(16) int8_t ws[4][16][TS_Q35_Q6K_ROW];
    __shared__ float wsc[4][16][16];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int row0 = blockIdx.x * 64 + warp * 16;
    const int nsb = k / 256, per = (nsb + gridDim.y - 1) / gridDim.y;
    const int sb0 = blockIdx.y * per, sb1 = min(nsb, sb0 + per);
    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            acc[j][e] = 0.0f;
    for (int sb = sb0; sb < sb1; ++sb)
    {
        __syncthreads();
        // Activations: NA rows x 256 int8 (16 bytes per thread-step) and their 8 scales.
        for (int i = tid; i < NA * 16; i += blockDim.x)
        {
            const int r = i >> 4, c = (i & 15) * 16;
            *(int4*)&xs[r][c] = r < n ? *(const int4*)(xq + (long long)r * k + sb * 256 + c) : make_int4(0, 0, 0, 0);
        }
        for (int i = tid; i < NA * 8; i += blockDim.x)
        {
            const int r = i >> 3;
            xds[r][i & 7] = r < n ? xd[(long long)r * (k >> 5) + sb * 8 + (i & 7)] : 0.0f;
        }
        // This warp's 16 weight rows: lane takes (row, half, 4-value group) items, 16 values each.
        for (int item = lane; item < 16 * 2 * 8; item += 32)
        {
            const int r = item >> 4, h = (item >> 3) & 1, l0 = (item & 7) * 4;
            const int gr = row0 + r;
            uint32_t q[4] = { 0, 0, 0, 0 };
            if (gr < outDim)
            {
                const uint8_t* blk = w + (long long)gr * rowBytes + (long long)sb * 210;
                const uint8_t* ql = blk + 64 * h;
                const uint8_t* qh = blk + 128 + 32 * h;
                const uint32_t a = *(const uint16_t*)(ql + l0) | ((uint32_t)*(const uint16_t*)(ql + l0 + 2) << 16);
                const uint32_t b = *(const uint16_t*)(ql + 32 + l0) | ((uint32_t)*(const uint16_t*)(ql + 32 + l0 + 2) << 16);
                const uint32_t c = *(const uint16_t*)(qh + l0) | ((uint32_t)*(const uint16_t*)(qh + l0 + 2) << 16);
                q[0] = __vsubss4((a & 0x0F0F0F0Fu) | ((c & 0x03030303u) << 4), 0x20202020);
                q[1] = __vsubss4((b & 0x0F0F0F0Fu) | (((c >> 2) & 0x03030303u) << 4), 0x20202020);
                q[2] = __vsubss4(((a >> 4) & 0x0F0F0F0Fu) | (((c >> 4) & 0x03030303u) << 4), 0x20202020);
                q[3] = __vsubss4(((b >> 4) & 0x0F0F0F0Fu) | (((c >> 6) & 0x03030303u) << 4), 0x20202020);
                if (l0 == 0)
                {
                    const float d = __half2float(*(const half*)(blk + 208));
                    const int8_t* sc = (const int8_t*)(blk + 192) + 8 * h;
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                        wsc[warp][r][8 * h + j] = d * (float)sc[j];
                }
            }
            else if (l0 == 0)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    wsc[warp][r][8 * h + j] = 0.0f;
            }
#pragma unroll
            for (int m = 0; m < 4; ++m)
                *(uint32_t*)&ws[warp][r][128 * h + 32 * m + l0] = q[m];
        }
        __syncthreads();
#pragma unroll 4
        for (int j = 0; j < 16; ++j)   // 16-value sub-blocks
        {
            const int a0 = *(const int*)&ws[warp][g][16 * j + 4 * t];
            const int a1 = *(const int*)&ws[warp][g + 8][16 * j + 4 * t];
            const float s0 = wsc[warp][g][j], s1 = wsc[warp][g + 8][j];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                int c[4];
                ts_q35_mma_s8(c, a0, a1, *(const int*)&xs[8 * nt + g][16 * j + 4 * t]);
                const float x0 = xds[8 * nt + 2 * t][j >> 1], x1 = xds[8 * nt + 2 * t + 1][j >> 1];
                acc[nt][0] += (float)c[0] * s0 * x0;
                acc[nt][1] += (float)c[1] * s0 * x1;
                acc[nt][2] += (float)c[2] * s1 * x0;
                acc[nt][3] += (float)c[3] * s1 * x1;
            }
        }
    }
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const int r = row0 + g + 8 * (e >> 1), col = 8 * nt + 2 * t + (e & 1);
            if (r < outDim && col < n)
            {
                if (gridDim.y > 1)
                    atomicAdd(y + (long long)col * outDim + r, acc[nt][e]);
                else
                    y[(long long)col * outDim + r] = acc[nt][e];
            }
        }
}

#define TS_Q35_GEMM_Q6K(NT) \
    extern "C" __global__ void __launch_bounds__(128) ts_q35_gemm_q6k_##NT( \
        const uint8_t* __restrict__ w, long long rowBytes, int outDim, int k, \
        const int8_t* __restrict__ xq, const float* __restrict__ xd, int n, float* __restrict__ y) \
    { ts_q35_gemm_q6k<NT>(w, rowBytes, outDim, k, xq, xd, n, y); }
TS_Q35_GEMM_Q6K(1)
TS_Q35_GEMM_Q6K(2)
TS_Q35_GEMM_Q6K(4)
TS_Q35_GEMM_Q6K(8)

// ---------------------------------------------------------------------------
// LM head on a tiled Q6_K layout. A 210-byte superblock gives a row's pieces no 16-byte
// boundary: on the stored rows the head loaded 2 bytes at a time, and ts_q35_gemm_q6k stages one
// superblock per __syncthreads with nothing in flight: 830 GB/s on the RTX 5090, 2.5 ms of a
// 4-sequence verify step for the target and draft heads. ts_q35_repack_q6k_tiled stores the head once, at
// load, in tiles of 64 rows: for each tile and superblock one 13,440-byte chunk (the 64
// superblocks' own bytes) laid out as
//   half 0: ql [64][64] | qh [64][32] | sc [64][8] | d [64] (half)    (6,784 bytes)
//   half 1: ql [64][64] | qh [64][32] | sc [64][8]                    (6,656 bytes)
// so a GEMM block streams each half superblock of its 64 rows as one contiguous, 16-byte-aligned
// run. The kernels compute the same integer products, the same scale products and the same
// per-output accumulation order as the stored-row kernels did (the one-row dp4a GEMV, ts_q35_gemm_q6k
// above it): bit-identical logits. RTX 5090, 248,320 x 5,120 head: 1 row 1,058 -> 869 us, 8 rows
// 888 -> 693 us, 32 rows 1,266 -> 739 us (1.41 TB/s), 64 rows 1,953 -> 1,073 us.
// ---------------------------------------------------------------------------
#define TS_Q35_Q6KT_ROWS 64
#define TS_Q35_Q6KT_CHUNK 13440
#define TS_Q35_Q6KT_H1 6784
#define TS_Q35_Q6KT_QH 4096
#define TS_Q35_Q6KT_SC 6144
#define TS_Q35_Q6KT_D 6656

/// Bytes of the tiled head: whole 64-row tiles (rows past outDim are zero).
__host__ __device__ __forceinline__ long long ts_q35_q6kt_bytes(int outDim, int k)
{
    return ((long long)outDim + TS_Q35_Q6KT_ROWS - 1) / TS_Q35_Q6KT_ROWS * (k / 256) * TS_Q35_Q6KT_CHUNK;
}

// Byte offsets, within a tile's chunk, of row rr's pieces of half h.
__device__ __forceinline__ int ts_q35_q6kt_ql(int h, int rr) { return h * TS_Q35_Q6KT_H1 + rr * 64; }
__device__ __forceinline__ int ts_q35_q6kt_qh(int h, int rr) { return h * TS_Q35_Q6KT_H1 + TS_Q35_Q6KT_QH + rr * 32; }
__device__ __forceinline__ int ts_q35_q6kt_sc(int h, int rr) { return h * TS_Q35_Q6KT_H1 + TS_Q35_Q6KT_SC + rr * 8; }

// One thread per (row, superblock) of `rows` source rows [rows][srcRowBytes] landing at rowBase.
extern "C" __global__ void ts_q35_repack_q6k_tiled(
    const uint8_t* __restrict__ src, long long srcRowBytes, int rows, int k, int rowBase, uint8_t* __restrict__ dst)
{
    const int nsb = k / 256;
    const long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (i >= (long long)rows * nsb)
        return;
    const long long rs = i / nsb;
    const int b = (int)(i % nsb);
    const long long r = rowBase + rs;
    const int rr = (int)(r % TS_Q35_Q6KT_ROWS);
    const uint8_t* blk = src + rs * srcRowBytes + (long long)b * 210;
    uint8_t* chunk = dst + (r / TS_Q35_Q6KT_ROWS * nsb + b) * TS_Q35_Q6KT_CHUNK;
#pragma unroll
    for (int h = 0; h < 2; ++h)
    {
        for (int j = 0; j < 64; ++j)
            chunk[ts_q35_q6kt_ql(h, rr) + j] = blk[64 * h + j];
        for (int j = 0; j < 32; ++j)
            chunk[ts_q35_q6kt_qh(h, rr) + j] = blk[128 + 32 * h + j];
        for (int j = 0; j < 8; ++j)
            chunk[ts_q35_q6kt_sc(h, rr) + j] = blk[192 + 8 * h + j];
    }
    chunk[TS_Q35_Q6KT_D + 2 * rr] = blk[208];
    chunk[TS_Q35_Q6KT_D + 2 * rr + 1] = blk[209];
}

// (float)c for |c| < 2^22 without the quarter-rate I2F: the same value, exactly.
__device__ __forceinline__ float ts_q35_i2f_small(int c)
{
    return __int_as_float(c + 0x4B400000) - 12582912.0f;
}

// One row of activations, one warp per weight row: lane j takes 128-value halves j, j + 32, ...
// (in that order), four consecutive values per dp4a, then the warp sum. 16-byte loads; each lane
// issues both of its halves' loads before their arithmetic.
extern "C" __global__ void ts_q35_gemv_q6kt_1(
    const uint8_t* __restrict__ w, int outDim, int k, const int8_t* __restrict__ xq, const float* __restrict__ xd,
    float* __restrict__ y)
{
    const int r = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (r >= outDim)
        return;
    const int nsb = k / 256, halves = k / 128, rr = r % TS_Q35_Q6KT_ROWS;
    const uint8_t* tile = w + (long long)(r / TS_Q35_Q6KT_ROWS) * nsb * TS_Q35_Q6KT_CHUNK;
    float acc = 0.0f;
    // Halves lane, lane + 32, ... in that order; each pair's loads are issued before its arithmetic.
    for (int u1 = lane; u1 < halves; u1 += 64)
    {
        uint32_t qlw[2][16], qhw[2][8];
        uint2 scw[2];
        float dd[2];
#pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            const int u = u1 + 32 * p;
            if (u >= halves)
                continue;
            const int h = u & 1;
            const uint8_t* chunk = tile + (long long)(u >> 1) * TS_Q35_Q6KT_CHUNK;
#pragma unroll
            for (int i = 0; i < 4; ++i)
                *(uint4*)&qlw[p][4 * i] = __ldg((const uint4*)(chunk + ts_q35_q6kt_ql(h, rr)) + i);
#pragma unroll
            for (int i = 0; i < 2; ++i)
                *(uint4*)&qhw[p][4 * i] = __ldg((const uint4*)(chunk + ts_q35_q6kt_qh(h, rr)) + i);
            scw[p] = __ldg((const uint2*)(chunk + ts_q35_q6kt_sc(h, rr)));
            dd[p] = __half2float(__ldg((const half*)(chunk + TS_Q35_Q6KT_D) + rr));
        }
#pragma unroll
        for (int p = 0; p < 2; ++p)
        {
            const int u = u1 + 32 * p;
            if (u >= halves)
                continue;
            const int8_t* scb = (const int8_t*)&scw[p];
            const float d = dd[p];
            const int v0 = (u >> 1) * 256 + 128 * (u & 1);
#pragma unroll
            for (int l0 = 0; l0 < 32; l0 += 4)
            {
                const uint32_t a = qlw[p][l0 / 4], b = qlw[p][8 + l0 / 4], c = qhw[p][l0 / 4];
                int wq[4];
                wq[0] = (int)__vsubss4((a & 0x0F0F0F0Fu) | ((c & 0x03030303u) << 4), 0x20202020);
                wq[1] = (int)__vsubss4((b & 0x0F0F0F0Fu) | (((c >> 2) & 0x03030303u) << 4), 0x20202020);
                wq[2] = (int)__vsubss4(((a >> 4) & 0x0F0F0F0Fu) | (((c >> 4) & 0x03030303u) << 4), 0x20202020);
                wq[3] = (int)__vsubss4(((b >> 4) & 0x0F0F0F0Fu) | (((c >> 6) & 0x03030303u) << 4), 0x20202020);
#pragma unroll
                for (int m = 0; m < 4; ++m)
                {
                    const int v = v0 + 32 * m + l0;
                    const float ws = d * (float)scb[2 * m + l0 / 16];
                    const int xi = *(const int*)(xq + v);
                    acc += ws * xd[v >> 5] * ts_q35_i2f_small(ts_q35_dp4a(wq[m], xi, 0));
                }
            }
        }
    }
    const float v = ts_q35_warp_sum(acc);
    if (lane == 0)
        y[r] = v;
}

// Many rows (a verify window, the draft head): TILES x 64 weight rows (four warps of 16 each) against
// NT * 8 activation rows per block, one mma.m16n8k16.s8 per 16-value sub-block as ts_q35_gemm_q6k.
// A stage is half a superblock of the block's tiles and of the activations, STAGES deep; shared
// rows are padded (ql 80, qh 48, activations 144 bytes) so the fragment loads hit 32 banks.
template <int NT, int TILES>
struct TsQ35Q6ktStage
{
    static constexpr int BR = TILES * TS_Q35_Q6KT_ROWS, NA = NT * 8, QL = 80, QH = 48, SC = 8, XA = 144;
    uint8_t ql[BR * QL];
    uint8_t qh[BR * QH];
    int8_t sc[BR * SC];
    half d[BR];
    int8_t x[NA * XA];
    float xd[NA * 4];
};

template <int NT, int TILES, int STAGES>
__device__ __forceinline__ void ts_q35_gemm_q6kt(
    const uint8_t* __restrict__ w, int outDim, int k, const int8_t* __restrict__ xq, const float* __restrict__ xd, int n,
    float* __restrict__ y)
{
    using Stage = TsQ35Q6ktStage<NT, TILES>;
    constexpr int BR = Stage::BR, NA = Stage::NA, QL = Stage::QL, QH = Stage::QH, SC = Stage::SC, XA = Stage::XA;
    constexpr int THREADS = TILES * 128;
    static_assert(sizeof(Stage) % 16 == 0, "16-byte stage slots");
    extern __shared__ __align__(16) uint8_t q6ktSmem[];
    Stage* st = reinterpret_cast<Stage*>(q6ktSmem);
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, t = lane & 3;
    const int tile0 = blockIdx.x * TILES, row0 = tile0 * TS_Q35_Q6KT_ROWS;
    const int nsb = k / 256, per = (nsb + gridDim.y - 1) / gridDim.y;
    const int sb0 = blockIdx.y * per, sb1 = min(nsb, sb0 + per);
    const int u0 = 2 * sb0, nH = max(0, 2 * (sb1 - sb0));
    const int tiles = (outDim + TS_Q35_Q6KT_ROWS - 1) / TS_Q35_Q6KT_ROWS;

    auto load = [&](int slot, int u) {
        Stage& s = st[slot];
        const int sbIdx = u >> 1, h = u & 1;
        for (int i = tid; i < BR * 4; i += THREADS)          // ql: 4 x 16 bytes per row
        {
            const int r = i >> 2, c = i & 3, tl = tile0 + r / TS_Q35_Q6KT_ROWS;
            const bool ok = tl < tiles;
            const uint8_t* chunk = w + ((long long)(ok ? tl : 0) * nsb + sbIdx) * TS_Q35_Q6KT_CHUNK;
            ts_q35_cp16(s.ql + r * QL + 16 * c, chunk + ts_q35_q6kt_ql(h, r % TS_Q35_Q6KT_ROWS) + 16 * c, ok);
        }
        for (int i = tid; i < BR * 2; i += THREADS)          // qh: 2 x 16 bytes per row
        {
            const int r = i >> 1, c = i & 1, tl = tile0 + r / TS_Q35_Q6KT_ROWS;
            const bool ok = tl < tiles;
            const uint8_t* chunk = w + ((long long)(ok ? tl : 0) * nsb + sbIdx) * TS_Q35_Q6KT_CHUNK;
            ts_q35_cp16(s.qh + r * QH + 16 * c, chunk + ts_q35_q6kt_qh(h, r % TS_Q35_Q6KT_ROWS) + 16 * c, ok);
        }
        for (int i = tid; i < BR / 2; i += THREADS)          // sc: 8 bytes per row, 16 per pair
        {
            const int r = 2 * i, tl = tile0 + r / TS_Q35_Q6KT_ROWS;
            const bool ok = tl < tiles;
            const uint8_t* chunk = w + ((long long)(ok ? tl : 0) * nsb + sbIdx) * TS_Q35_Q6KT_CHUNK;
            ts_q35_cp16(s.sc + r * SC, chunk + ts_q35_q6kt_sc(h, r % TS_Q35_Q6KT_ROWS), ok);
        }
        if (h == 0)
            for (int i = tid; i < BR / 8; i += THREADS)      // d: 2 bytes per row, half 0 only
            {
                const int r = 8 * i, tl = tile0 + r / TS_Q35_Q6KT_ROWS;
                const bool ok = tl < tiles;
                const uint8_t* chunk = w + ((long long)(ok ? tl : 0) * nsb + sbIdx) * TS_Q35_Q6KT_CHUNK;
                ts_q35_cp16(s.d + r, chunk + TS_Q35_Q6KT_D + 2 * (r % TS_Q35_Q6KT_ROWS), ok);
            }
        for (int i = tid; i < NA * 8; i += THREADS)
        {
            const int r = i >> 3, c = i & 7;
            const bool ok = r < n;
            ts_q35_cp16(s.x + r * XA + 16 * c, xq + (long long)(ok ? r : 0) * k + 128LL * u + 16 * c, ok);
        }
        for (int r = tid; r < NA; r += THREADS)
        {
            const bool ok = r < n;
            ts_q35_cp16(s.xd + 4 * r, xd + (long long)(ok ? r : 0) * (k >> 5) + 4LL * u, ok);
        }
    };

    float acc[NT][4];
#pragma unroll
    for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            acc[j][e] = 0.0f;

#pragma unroll
    for (int s = 0; s < STAGES - 1; ++s)
    {
        if (s < nH)
            load(s, u0 + s);
        ts_q35_cp_commit();
    }
    const int wr = warp * 16 + g;   // this thread's first weight row in the block (the second is wr + 8)
    float d0 = 0.0f, d1 = 0.0f;
    for (int it = 0; it < nH; ++it)
    {
        ts_q35_cp_wait<STAGES - 2>();
        __syncthreads();
        const int next = it + STAGES - 1;
        if (next < nH)
            load(next % STAGES, u0 + next);
        ts_q35_cp_commit();
        const Stage& s = st[it % STAGES];
        if ((it & 1) == 0)   // u0 is even: half 0 of a superblock carries its d
        {
            d0 = __half2float(s.d[wr]);
            d1 = __half2float(s.d[wr + 8]);
        }
        // Rows wr and wr + 8, values l = 16 lh + 4t .. + 3 of this half: ql bytes l (m even, low
        // nibbles for m < 2) and 32 + l (m odd), qh byte l (bits 2m, 2m + 1).
        uint32_t qlw[2][2][2], qhw[2][2];
#pragma unroll
        for (int rr = 0; rr < 2; ++rr)
#pragma unroll
            for (int lh = 0; lh < 2; ++lh)
            {
                const int row = wr + 8 * rr;
                qlw[rr][lh][0] = *(const uint32_t*)(s.ql + row * QL + 16 * lh + 4 * t);
                qlw[rr][lh][1] = *(const uint32_t*)(s.ql + row * QL + 32 + 16 * lh + 4 * t);
                qhw[rr][lh] = *(const uint32_t*)(s.qh + row * QH + 16 * lh + 4 * t);
            }
#pragma unroll
        for (int m = 0; m < 4; ++m)
        {
            float x0[NT], x1[NT];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                x0[nt] = s.xd[(8 * nt + 2 * t) * 4 + m];
                x1[nt] = s.xd[(8 * nt + 2 * t + 1) * 4 + m];
            }
#pragma unroll
            for (int lh = 0; lh < 2; ++lh)
            {
                int a[2];
#pragma unroll
                for (int rr = 0; rr < 2; ++rr)
                    a[rr] = (int)__vsubss4(((qlw[rr][lh][m & 1] >> (4 * (m >> 1))) & 0x0F0F0F0Fu)
                                               | (((qhw[rr][lh] >> (2 * m)) & 0x03030303u) << 4), 0x20202020);
                const int jj = 2 * m + lh;
                const float s0 = d0 * (float)s.sc[wr * SC + jj], s1 = d1 * (float)s.sc[(wr + 8) * SC + jj];
#pragma unroll
                for (int nt = 0; nt < NT; ++nt)
                {
                    int c[4];
                    ts_q35_mma_s8(c, a[0], a[1], *(const int*)(s.x + (8 * nt + g) * XA + 32 * m + 16 * lh + 4 * t));
                    acc[nt][0] += ts_q35_i2f_small(c[0]) * s0 * x0[nt];
                    acc[nt][1] += ts_q35_i2f_small(c[1]) * s0 * x1[nt];
                    acc[nt][2] += ts_q35_i2f_small(c[2]) * s1 * x0[nt];
                    acc[nt][3] += ts_q35_i2f_small(c[3]) * s1 * x1[nt];
                }
            }
        }
    }
    ts_q35_cp_wait<0>();
#pragma unroll
    for (int nt = 0; nt < NT; ++nt)
#pragma unroll
        for (int e = 0; e < 4; ++e)
        {
            const int r = row0 + wr + 8 * (e >> 1), col = 8 * nt + 2 * t + (e & 1);
            if (r < outDim && col < n)
            {
                if (gridDim.y > 1)
                    atomicAdd(y + (long long)col * outDim + r, acc[nt][e]);
                else
                    y[(long long)col * outDim + r] = acc[nt][e];
            }
        }
}

// grid (ceil(outDim / 64) tiles, K splits), block 128, dynamic shared memory
// ts_q35_gemm_q6kt_smem_bytes reports (3 stages of TsQ35Q6ktStage<NT, 1>).
#define TS_Q35_GEMM_Q6KT(NT) \
    extern "C" __global__ void __launch_bounds__(128) ts_q35_gemm_q6kt_##NT( \
        const uint8_t* __restrict__ w, int outDim, int k, const int8_t* __restrict__ xq, const float* __restrict__ xd, int n, \
        float* __restrict__ y) \
    { ts_q35_gemm_q6kt<NT, 1, 3>(w, outDim, k, xq, xd, n, y); }
TS_Q35_GEMM_Q6KT(1)
TS_Q35_GEMM_Q6KT(2)
TS_Q35_GEMM_Q6KT(4)
TS_Q35_GEMM_Q6KT(8)

extern "C" __global__ void ts_q35_gemm_q6kt_smem_bytes(int* out)
{
    out[0] = 3 * (int)sizeof(TsQ35Q6ktStage<1, 1>);
    out[1] = 3 * (int)sizeof(TsQ35Q6ktStage<2, 1>);
    out[2] = 3 * (int)sizeof(TsQ35Q6ktStage<4, 1>);
    out[3] = 3 * (int)sizeof(TsQ35Q6ktStage<8, 1>);
}

// dst[r] = src[idx[r]] for rows of `width` floats; one block per row.
extern "C" __global__ void ts_q35_rows_by_index_f32(
    const float* __restrict__ src, int width, const int* __restrict__ idx, float* __restrict__ dst)
{
    const float* in = src + (long long)idx[blockIdx.x] * width;
    float* o = dst + (long long)blockIdx.x * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        o[i] = in[i];
}

// dst[row][0, width) = src[row][0, width), rows of their own strides (floats); one block per row.
extern "C" __global__ void ts_q35_copy_rows_f32(
    const float* __restrict__ src, long long srcStride, float* __restrict__ dst, long long dstStride, int width)
{
    const float* in = src + blockIdx.x * srcStride;
    float* o = dst + blockIdx.x * dstStride;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        o[i] = in[i];
}

// ---------------------------------------------------------------------------
// Row argmax (first index of the maximum, as ggml's argmax): one 1024-thread block per row.
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_q35_argmax_f32(const float* __restrict__ x, int n, int* __restrict__ out)
{
    __shared__ float bv[32];
    __shared__ int bi[32];
    const float* row = x + (long long)blockIdx.x * n;
    // TokenSampler.Argmax seeds its running maximum with row[0]. When that
    // value is NaN every later comparison is false, so CPU sampling returns 0.
    // This branch is uniform across the CTA and precedes every synchronization.
    if (isnan(row[0]))
    {
        if (threadIdx.x == 0)
            out[blockIdx.x] = 0;
        return;
    }
    float best = -INFINITY;
    int idx = 0x7fffffff;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float v = row[i];
        if (v > best || (v == best && i < idx))
        {
            best = v;
            idx = i;
        }
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
    {
        const float ov = __shfl_xor_sync(0xffffffff, best, off);
        const int oi = __shfl_xor_sync(0xffffffff, idx, off);
        if (ov > best || (ov == best && oi < idx))
        {
            best = ov;
            idx = oi;
        }
    }
    if ((threadIdx.x & 31) == 0)
    {
        bv[threadIdx.x >> 5] = best;
        bi[threadIdx.x >> 5] = idx;
    }
    __syncthreads();
    if (threadIdx.x < 32)
    {
        const int warps = blockDim.x >> 5;
        best = threadIdx.x < warps ? bv[threadIdx.x] : -INFINITY;
        idx = threadIdx.x < warps ? bi[threadIdx.x] : 0x7fffffff;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            const float ov = __shfl_xor_sync(0xffffffff, best, off);
            const int oi = __shfl_xor_sync(0xffffffff, idx, off);
            if (ov > best || (ov == best && oi < idx))
            {
                best = ov;
                idx = oi;
            }
        }
        if (threadIdx.x == 0)
            out[blockIdx.x] = idx;
    }
}

#include "tensorsharp_q35_dflash.cuh"

// ---------------------------------------------------------------------------
// FP8 E4M3 row-scaled weights (NInfer v3 fp8_e4m3fn_row_bf16: codes [rows][k], one BF16 scale per
// row), the representation of the NInfer Qwen3.8-27B artifact's embedding, head, attention/GDN
// projections and last MLP layers.
//   * ts_q35_fp8_<x>_t<TT>_<epilogue>: small-T (decode / verify / drafter head) BF16 tensor-core
//     projection, see tensorsharp_small_t_mma.cuh. A16: activations are never quantized.
//   * ts_q35_fp8_quant_act: per-row E4M3 activation codes with an F32 scale amax / 448 (NInfer
//     fp8_a8.cu fp8_a8_quantize_kernel numerics), for prefill's A8 projection.
//   * ts_q35_gemm_w8a8_tma<bm>_<epilogue>: prefill W8A8 GEMM, the warp-specialized TMA structure of
//     ts_q35_gemm_w4a4_tma with the m16n8k32 E4M3 MMA (64 K values per 64-byte tile) and the
//     scales applied once at the store: y = acc * xScale[token] * wScale[row].
//   * ts_q35_embed_fp8: token rows, BF16(code * row scale) as F32.
// ---------------------------------------------------------------------------
#include "tensorsharp_small_t_mma.cuh"

// T <= 8 reads the F32 activations directly; wider tiles take BF16 activations (cast once per
// projection) and share each staged activation slab across RT 16-row weight tiles.
#define TS_Q35_FP8_SMALL(NAME, XBF, YBF, OUT, TT, KW, ST, RT) \
    extern "C" __global__ __launch_bounds__(KW * 32) void NAME(const uint8_t* __restrict__ w, const uint16_t* __restrict__ s, \
        const void* __restrict__ x, void* __restrict__ y, int n, int k, int m, int ldy) \
    { ts_smt::project<ts_smt::kFp8Row, XBF, YBF, OUT, TT, KW, ST, RT>(w, s, x, y, n, k, m, ldy); }
TS_Q35_FP8_SMALL(ts_q35_fp8_f32_t8_plain, false, false, 0, 8, 4, 3, 1)
TS_Q35_FP8_SMALL(ts_q35_fp8_f32_t8_round, false, false, 1, 8, 4, 3, 1)
TS_Q35_FP8_SMALL(ts_q35_fp8_f32_t8_swiglu, false, false, 2, 8, 4, 3, 1)
TS_Q35_FP8_SMALL(ts_q35_fp8_bx_t16_plain, true, false, 0, 16, 4, 2, 2)
TS_Q35_FP8_SMALL(ts_q35_fp8_bx_t16_round, true, false, 1, 16, 4, 2, 2)
TS_Q35_FP8_SMALL(ts_q35_fp8_bx_t16_swiglu, true, false, 2, 16, 4, 2, 2)
TS_Q35_FP8_SMALL(ts_q35_fp8_bx_t32_plain, true, false, 0, 32, 2, 2, 4)
TS_Q35_FP8_SMALL(ts_q35_fp8_bx_t32_round, true, false, 1, 32, 2, 2, 4)
TS_Q35_FP8_SMALL(ts_q35_fp8_bx_t32_swiglu, true, false, 2, 32, 2, 2, 4)
TS_Q35_FP8_SMALL(ts_q35_fp8_bf16_t8_plain, true, true, 0, 8, 4, 3, 1)
TS_Q35_FP8_SMALL(ts_q35_fp8_bf16_t16_plain, true, true, 0, 16, 4, 2, 2)
TS_Q35_FP8_SMALL(ts_q35_fp8_bf16_t32_plain, true, true, 0, 32, 2, 2, 4)

// T = 1 (non-speculative decode): CUDA-core GEMV, NInfer's RTX 5090 shape choices: b = 4 warps x
// 4 rows x 16 codes (the 16384-row GDN input projection), c = 8 warps x 2 rows x 16 codes.
#define TS_Q35_FP8_GEMV(NAME, WARPS, RPW, VPL, OUT, MINB) \
    extern "C" __global__ __launch_bounds__(WARPS * 32, MINB) void NAME(const uint8_t* __restrict__ w, \
        const uint16_t* __restrict__ s, const float* __restrict__ x, float* __restrict__ y, int k) \
    { ts_smt::gemv<WARPS, RPW, VPL, OUT>(w, s, x, y, k); }
TS_Q35_FP8_GEMV(ts_q35_fp8_gemv_b_plain, 4, 4, 16, 0, 1)
TS_Q35_FP8_GEMV(ts_q35_fp8_gemv_b_round, 4, 4, 16, 1, 1)
TS_Q35_FP8_GEMV(ts_q35_fp8_gemv_b_swiglu, 4, 4, 16, 2, 1)
TS_Q35_FP8_GEMV(ts_q35_fp8_gemv_c_plain, 8, 2, 16, 0, 2)
TS_Q35_FP8_GEMV(ts_q35_fp8_gemv_c_round, 8, 2, 16, 1, 2)
TS_Q35_FP8_GEMV(ts_q35_fp8_gemv_c_swiglu, 8, 2, 16, 2, 2)

/// One CTA (256 threads) per row: scale = amax / 448, codes = E4M3(x / scale) (satfinite).
extern "C" __global__ __launch_bounds__(256) void ts_q35_fp8_quant_act(const float* __restrict__ x,
    uint8_t* __restrict__ codes, float* __restrict__ scales, int k)
{
    __shared__ float warpMax[8];
    __shared__ float rowScale;
    const long long base = (long long)blockIdx.x * k;
    const float4* x4 = reinterpret_cast<const float4*>(x + base);
    float amax = 0.0f;
    for (int i = threadIdx.x; i < k / 4; i += 256)
    {
        const float4 v = x4[i];
        amax = fmaxf(amax, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w))));
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, off));
    if ((threadIdx.x & 31) == 0) warpMax[threadIdx.x >> 5] = amax;
    __syncthreads();
    if (threadIdx.x < 32)
    {
        float v = threadIdx.x < 8 ? warpMax[threadIdx.x] : 0.0f;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, off));
        if (threadIdx.x == 0)
        {
            rowScale = v > 0.0f ? v / 448.0f : 0.0f;
            scales[blockIdx.x] = rowScale;
        }
    }
    __syncthreads();
    const float inverse = rowScale > 0.0f ? 1.0f / rowScale : 0.0f;
    uint32_t* out = reinterpret_cast<uint32_t*>(codes + base);
    for (int i = threadIdx.x; i < k / 4; i += 256)
    {
        const float4 v = x4[i];
        const uint32_t lo = __nv_cvt_float2_to_fp8x2(make_float2(v.x * inverse, v.y * inverse), __NV_SATFINITE, __NV_E4M3);
        const uint32_t hi = __nv_cvt_float2_to_fp8x2(make_float2(v.z * inverse, v.w * inverse), __NV_SATFINITE, __NV_E4M3);
        out[i] = lo | (hi << 16);
    }
}

/// out[row][c] = BF16(E4M3(codes[id][c]) * scale[id]) as F32; one CTA per token row.
extern "C" __global__ void ts_q35_embed_fp8(const uint8_t* __restrict__ codes, const uint16_t* __restrict__ scales,
    const int* __restrict__ ids, float* __restrict__ out, int width, int vocab)
{
    const int row = blockIdx.x, id = ids[row];
    if (id < 0 || id >= vocab) { asm("trap;"); return; }
    const float s = ts_smt::bf16_bits(scales[id]);
    const uint8_t* src = codes + (long long)id * width;
    for (int i = threadIdx.x * 2; i < width; i += blockDim.x * 2)
    {
        const uint32_t bf = ts_smt::e4m3x2_bf16x2(*reinterpret_cast<const uint16_t*>(src + i));
        out[(long long)row * width + i] = ts_smt::round_bf16(__uint_as_float(bf << 16) * s);
        out[(long long)row * width + i + 1] = ts_smt::round_bf16(__uint_as_float(bf & 0xffff0000u) * s);
    }
}

template <int BM, int STAGES>
struct TsQ35Fp8TmaSmem
{
    alignas(1024) uint8_t xc[STAGES][BM * 64];
    alignas(1024) uint8_t wc[STAGES][128 * 64];
    alignas(8) unsigned long long full[STAGES];
    alignas(8) unsigned long long empty[STAGES];
};

// E4M3 x E4M3 with F32 accumulation. On RTX 5090 (sm_120a) the plain m16n8k32 form accumulates in
// F32 at half rate; the block-scaled kind::mxf8f6f4 form with unit UE8M0 scales (0x7f = 2^0) runs at
// full rate and computes the same raw dot product (NInfer 1882c6ad src/ops/common/cuda/mma.cuh).
__device__ __forceinline__ void ts_q35_mma_fp8(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(TS_Q35_FP4_MMA)
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%10}, {%11,%12};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x7fu), "h"((unsigned short)0), "h"((unsigned short)0));
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// EPI: 0 y = v, 1 y = BF16(v), 2 y = BF16(silu(BF16(gate)) * BF16(up)), 3 y = BF16(silu(gate) * up),
// v = acc * xScale[token] * wScale[row]; SwiGLU weights are gate|up interleaved by 8 rows.
template <int BM, int STAGES, int EPI>
__device__ __forceinline__ void ts_q35_gemm_w8a8_tma(const TsQ35TmaMaps& maps, int outDim, int k,
    const float* __restrict__ xScale, const uint16_t* __restrict__ wScale, int m, float* __restrict__ y)
{
    static_assert(BM == 128 || BM == 256, "token tile");
    static_assert(EPI >= 0 && EPI <= 3, "epilogues 0-3");
#if defined(TS_Q35_FP4_MMA)
    constexpr int PRODUCERS = BM == 256 ? 128 : 32, WARP_M = BM / 4, MT = WARP_M / 16, NT = 8;
    extern __shared__ __align__(1024) uint8_t tmaSmem[];
    auto& sm = *reinterpret_cast<TsQ35Fp8TmaSmem<BM, STAGES>*>(tmaSmem);
    const int token0 = blockIdx.x * BM, row0 = blockIdx.y * 128, kTiles = k / 64;

    if (threadIdx.x == 0)
    {
#pragma unroll
        for (int s = 0; s < STAGES; ++s)
        {
            ts_q35_mbar_init(&sm.full[s], 1);
            ts_q35_mbar_init(&sm.empty[s], 8);
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
            constexpr uint32_t CODES = BM * 64 + 128 * 64;
#pragma unroll 1
            for (int kt = 0; kt < kTiles; ++kt)
            {
                const int s = kt % STAGES;
                ts_q35_mbar_wait(&sm.empty[s], 1u ^ ((kt / STAGES) & 1u));
                ts_q35_mbar_expect(&sm.full[s], CODES);
                ts_q35_tma_2d(sm.xc[s], &maps.xCodes, kt * 64, token0, &sm.full[s]);
                ts_q35_tma_2d(sm.wc[s], &maps.wCodes, kt * 64, row0, &sm.full[s]);
            }
        }
        return;
    }
    if constexpr (PRODUCERS == 128)
        asm volatile("setmaxnreg.inc.sync.aligned.u32 232;" ::: "memory");

    const int ct = threadIdx.x - PRODUCERS, lane = ct & 31, warp = ct >> 5;
    const int wm = warp >> 1, wn = warp & 1;
    const int g = lane >> 2, q = lane & 3;
    const int aRow = wm * WARP_M + (lane & 15), aByte = (lane >> 4) * 16;
    const int bRow = wn * 64 + (lane & 7), bByte = ((lane >> 3) & 1) * 16;

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
        ts_q35_mbar_wait(&sm.full[s], (kt / STAGES) & 1u);
        const uint32_t xc = ts_q35_smem_u32(sm.xc[s]), wc = ts_q35_smem_u32(sm.wc[s]);
#pragma unroll
        for (int k32 = 0; k32 < 2; ++k32)
        {
            uint32_t a[MT][4];
#pragma unroll
            for (int i = 0; i < MT; ++i)
                ts_q35_ldsm_x4a(a[i], ts_q35_sw64(xc + (aRow + i * 16) * 64 + k32 * 32 + aByte));
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                uint32_t b0, b1;
                ts_q35_ldsm_x2(b0, b1, ts_q35_sw64(wc + (bRow + j * 8) * 64 + k32 * 32 + bByte));
#pragma unroll
                for (int i = 0; i < MT; ++i)
                    ts_q35_mma_fp8(acc[i][j], a[i], b0, b1);
            }
        }
        if (lane == 0)
            ts_q35_mbar_arrive(&sm.empty[s]);
    }

    // D rows are tokens (g, g + 8), columns weight rows (2q, 2q + 1).
    const int tokBase = token0 + wm * WARP_M + g, rowBase = row0 + wn * 64 + 2 * q;
    float ws[NT][2];
#pragma unroll
    for (int j = 0; j < NT; ++j)
    {
        const int r = rowBase + j * 8;
        ws[j][0] = r < outDim ? ts_smt::bf16_bits(wScale[r]) : 0.0f;
        ws[j][1] = r + 1 < outDim ? ts_smt::bf16_bits(wScale[r + 1]) : 0.0f;
    }
    if constexpr (EPI == 2 || EPI == 3)
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
                const float xs = xScale[c];
#pragma unroll
                for (int jp = 0; jp < NT / 2; ++jp)
                {
                    const int r = rowBase + jp * 16;
                    if (r >= outDim)
                        continue;
                    float out[2];
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        float gate = acc[i][2 * jp][h * 2 + e] * xs * ws[2 * jp][e];
                        float up = acc[i][2 * jp + 1][h * 2 + e] * xs * ws[2 * jp + 1][e];
                        if constexpr (EPI == 2)
                        {
                            gate = __bfloat162float(__float2bfloat16_rn(gate));
                            up = __bfloat162float(__float2bfloat16_rn(up));
                        }
                        out[e] = __bfloat162float(__float2bfloat16_rn(__fmul_rn(ts_q35_silu(gate), up)));
                    }
                    *(float2*)(y + (long long)c * ff + r / 16 * 8 + r % 16) = make_float2(out[0], out[1]);
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
            const float xs = xScale[c];
#pragma unroll
            for (int j = 0; j < NT; ++j)
            {
                const int r = rowBase + j * 8;
                if (r >= outDim)
                    continue;
                float v0 = acc[i][j][h * 2] * xs * ws[j][0], v1 = acc[i][j][h * 2 + 1] * xs * ws[j][1];
                if constexpr (EPI == 1)
                {
                    v0 = __bfloat162float(__float2bfloat16_rn(v0));
                    v1 = __bfloat162float(__float2bfloat16_rn(v1));
                }
                *(float2*)(y + (long long)c * outDim + r) = make_float2(v0, v1);
            }
        }
#else
    __trap();
#endif
}

#define TS_Q35_GEMM_W8A8_TMA(NAME, BM, STAGES, EPI) \
    extern "C" __global__ __launch_bounds__(256 + (BM == 256 ? 128 : 32), 1) void NAME( \
        const __grid_constant__ TsQ35TmaMaps maps, int outDim, int k, const float* __restrict__ xScale, \
        const uint16_t* __restrict__ wScale, int m, float* __restrict__ y) \
    { ts_q35_gemm_w8a8_tma<BM, STAGES, EPI>(maps, outDim, k, xScale, wScale, m, y); }
TS_Q35_GEMM_W8A8_TMA(ts_q35_gemm_w8a8_tma256_plain, 256, 4, 0)
TS_Q35_GEMM_W8A8_TMA(ts_q35_gemm_w8a8_tma256_round, 256, 4, 1)
TS_Q35_GEMM_W8A8_TMA(ts_q35_gemm_w8a8_tma256_swiglu_rin, 256, 4, 2)
TS_Q35_GEMM_W8A8_TMA(ts_q35_gemm_w8a8_tma256_swiglu, 256, 4, 3)
TS_Q35_GEMM_W8A8_TMA(ts_q35_gemm_w8a8_tma128_plain, 128, 4, 0)
TS_Q35_GEMM_W8A8_TMA(ts_q35_gemm_w8a8_tma128_round, 128, 4, 1)
TS_Q35_GEMM_W8A8_TMA(ts_q35_gemm_w8a8_tma128_swiglu_rin, 128, 4, 2)
TS_Q35_GEMM_W8A8_TMA(ts_q35_gemm_w8a8_tma128_swiglu, 128, 4, 3)

/// Dynamic shared memory of the W8A8 TMA GEMMs: [BM256 4-stage, BM128 4-stage].
extern "C" __global__ void ts_q35_gemm_w8a8_tma_smem_bytes(int* out)
{
    out[0] = (int)sizeof(TsQ35Fp8TmaSmem<256, 4>);
    out[1] = (int)sizeof(TsQ35Fp8TmaSmem<128, 4>);
}
