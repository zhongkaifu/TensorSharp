// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// Small-T projections on the BF16 tensor cores, for decode, speculative verification and the
// drafter: y[t][r] = rowScale(r) * sum_k W[r][k] * x[t][k] for up to TT activation rows per CTA,
// every weight byte read once per CTA. The structure follows NInfer's sliced-K A16 kernels
// (Neroued/ninfer 1882c6ad, src/ops/linear/kernels/fp8/fp8_a16_sliced_k_mma.cuh, Apache-2.0);
// this is an independent TensorSharp implementation:
//   * A CTA owns 16 weight rows. Its KW warps split every KW*64-value K slab, 64 values each,
//     and each warp streams its own slice (weights and activations) through an ST-deep cp.async
//     ring, so warps never wait on each other until the final cross-warp sum.
//   * Weights widen exactly to BF16 MMA operands (every E4M3 and INT8 code is a BF16 value).
//     A W8G32 group scale is applied in F32 per 32 K values; an FP8 row scale once at the end.
//     Activations are BF16 values (F32 inputs are BF16-rounded by their producers).
//   * The MMA K order is permuted: lane quad q holds K values 4q..4q+3 of each 16 for both A
//     and B, so every operand fragment is a single aligned shared-memory load. A dot product is
//     invariant under a common permutation of K.
//   * SwiGLU output: the 16 rows of a CTA are 8 gate rows then their 8 up rows (the gate|up
//     weights are interleaved by 8 rows at load), so a thread holds a gate and its up value.
#pragma once
#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <stdint.h>

namespace ts_smt {

enum : int { kFp8Row = 0, kW8G32 = 1, kBf16 = 2 };
enum : int { kOutPlain = 0, kOutRound = 1, kOutSwiGlu = 2 };

__device__ __forceinline__ uint32_t smem_u32(const void* p) { return (uint32_t)__cvta_generic_to_shared(p); }

__device__ __forceinline__ void cp16(void* dst, const void* src, bool valid)
{
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" :: "r"(smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
}
__device__ __forceinline__ void commit() { asm volatile("cp.async.commit_group;\n" ::); }
template <int N> __device__ __forceinline__ void wait() { asm volatile("cp.async.wait_group %0;\n" :: "n"(N)); }

__device__ __forceinline__ void mma(float (&c)[4], uint32_t a0, uint32_t a1, uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1)
{
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__device__ __forceinline__ uint32_t pack_bf16(float lo, float hi)
{
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<uint32_t*>(&v);
}

// Two E4M3FN codes (low byte = lower K) -> BF16x2, exact. sm_120a with CUDA 13.2+ widens the pair
// in one instruction (PTX 9.2 cvt.rn.bf16x2.e4m3x2, as NInfer's fp8_e4m3x2_to_bf16x2_bits);
// otherwise E4M3 -> F16 -> F32 -> BF16.
__device__ __forceinline__ uint32_t e4m3x2_bf16x2(uint32_t two)
{
#if (defined(__CUDA_ARCH_FEAT_SM120_ALL) || defined(__CUDA_ARCH_FEAT_SM121_ALL)) && \
    (__CUDACC_VER_MAJOR__ > 13 || (__CUDACC_VER_MAJOR__ == 13 && __CUDACC_VER_MINOR__ >= 2))
    uint32_t r;
    asm("cvt.rn.bf16x2.e4m3x2 %0, %1;" : "=r"(r) : "h"((unsigned short)(two & 0xffffu)));
    return r;
#else
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    const float2 f = __half22float2(*reinterpret_cast<const __half2*>(&h));
    return pack_bf16(f.x, f.y);
#endif
}

// Two INT8 codes -> BF16x2, exact.
__device__ __forceinline__ uint32_t s8x2_bf16x2(uint32_t two)
{
    return pack_bf16((float)(int8_t)(two & 0xffu), (float)(int8_t)((two >> 8) & 0xffu));
}

__device__ __forceinline__ float bf16_bits(uint16_t bits) { return __uint_as_float((uint32_t)bits << 16); }
__device__ __forceinline__ float silu(float x) { return x / (1.0f + expf(-x)); }
__device__ __forceinline__ float round_bf16(float x) { return __bfloat162float(__float2bfloat16_rn(x)); }

template <int FMT, bool XBF, int TT, int KW, int ST, int RT = 1>
struct Geometry
{
    static constexpr int WB = FMT == kBf16 ? 2 : 1;           // weight bytes per value
    static constexpr int XB = XBF ? 2 : 4;                     // activation bytes per value
    static constexpr int WROW = 64 * WB + (FMT == kBf16 ? 32 : 16);
    static constexpr int XROW = 64 * XB + (XBF ? 32 : 64);
    static constexpr int WSTAGE = 16 * RT * WROW, XSTAGE = TT * XROW;
    static constexpr int STAGE = WSTAGE + XSTAGE;
    static constexpr int SHARED = KW * ST * STAGE;
    static constexpr int REDUCE = KW * RT * TT * 64;          // KW x RT x (TT/8) x 32 lanes x 4 floats
    static_assert(SHARED >= REDUCE, "the reduction reuses the ring");
    static_assert(SHARED <= 48 * 1024, "static shared memory");
    static_assert(TT % 8 == 0 && TT <= 32 && ST >= 2 && KW >= 1 && RT >= 1 && RT <= 4, "geometry");
};

/// grid.x = output-row tiles (16 * RT weight rows, or 8 * RT SwiGLU outputs), grid.y = TT-row
/// activation tiles; block = KW warps. n = weight rows (multiple of 16 * RT), k = K (multiple of
/// KW*64), m = activation rows, ldy = output row stride in elements. sc: FP8 BF16 row scales [n];
/// W8G32 F16 scales [n][k/32]. RT row tiles share every staged activation slab.
template <int FMT, bool XBF, bool YBF, int OUT, int TT, int KW, int ST, int RT = 1>
__device__ __forceinline__ void project(const uint8_t* __restrict__ w, const void* __restrict__ sc,
    const void* __restrict__ xv, void* __restrict__ yv, int n, int k, int m, int ldy)
{
    using G = Geometry<FMT, XBF, TT, KW, ST, RT>;
    __shared__ __align__(16) uint8_t smem[G::SHARED];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, gid = lane >> 2, lid = lane & 3;
    const int row0 = blockIdx.x * 16 * RT, tok0 = blockIdx.y * TT;
    const int slabs = k / (KW * 64);
    uint8_t* ring = smem + warp * ST * G::STAGE;
    const uint8_t* xb = reinterpret_cast<const uint8_t*>(xv);

    auto load = [&](int stage, int slab) {
        uint8_t* ws = ring + stage * G::STAGE;
        uint8_t* xs = ws + G::WSTAGE;
        const int k0 = slab * KW * 64 + warp * 64;
        constexpr int WCH = 4 * G::WB;   // 16-byte chunks per weight row slice
#pragma unroll
        for (int c = lane; c < 16 * RT * WCH; c += 32)
        {
            const int r = c / WCH, part = c % WCH;
            cp16(ws + r * G::WROW + part * 16, w + ((long long)(row0 + r) * k + k0) * G::WB + part * 16, true);
        }
        constexpr int XCH = 4 * G::XB;
#pragma unroll
        for (int c = lane; c < TT * XCH; c += 32)
        {
            const int t = c / XCH, part = c % XCH, token = tok0 + t;
            const bool live = token < m;
            cp16(xs + t * G::XROW + part * 16, xb + ((long long)(live ? token : 0) * k + k0) * G::XB + part * 16, live);
        }
    };

    float acc[RT][TT / 8][4];
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < TT / 8; ++j) acc[r][j][0] = acc[r][j][1] = acc[r][j][2] = acc[r][j][3] = 0.0f;

#pragma unroll
    for (int s = 0; s < ST - 1; ++s)
    {
        if (s < slabs) load(s, s);
        commit();
    }
#pragma unroll 1
    for (int i = 0; i < slabs; ++i)
    {
        wait<ST - 2>();
        __syncwarp();
        const int nx = i + ST - 1;
        if (nx < slabs) load(nx % ST, nx);
        commit();
        const uint8_t* ws = ring + (i % ST) * G::STAGE;
        const uint8_t* xs = ws + G::WSTAGE;
        float gs[2][RT][2];   // W8G32: [group][row tile][row gid, gid+8]
        if constexpr (FMT == kW8G32)
        {
            const __half* s16 = reinterpret_cast<const __half*>(sc);
            const int groups = k / 32, g0 = (i * KW * 64 + warp * 64) / 32;
#pragma unroll
            for (int g = 0; g < 2; ++g)
#pragma unroll
                for (int r = 0; r < RT; ++r)
                {
                    gs[g][r][0] = __half2float(s16[(long long)(row0 + r * 16 + gid) * groups + g0 + g]);
                    gs[g][r][1] = __half2float(s16[(long long)(row0 + r * 16 + gid + 8) * groups + g0 + g]);
                }
        }
#pragma unroll
        for (int g = 0; g < 2; ++g)
        {
            float part[RT][TT / 8][4];
            if constexpr (FMT == kW8G32)
            {
#pragma unroll
                for (int r = 0; r < RT; ++r)
#pragma unroll
                    for (int j = 0; j < TT / 8; ++j) part[r][j][0] = part[r][j][1] = part[r][j][2] = part[r][j][3] = 0.0f;
            }
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int s16 = g * 2 + h;
                uint32_t b0[TT / 8], b1[TT / 8];
#pragma unroll
                for (int j = 0; j < TT / 8; ++j)
                {
                    if constexpr (XBF)
                    {
                        const uint2 v = *reinterpret_cast<const uint2*>(xs + (j * 8 + gid) * G::XROW + s16 * 32 + 8 * lid);
                        b0[j] = v.x; b1[j] = v.y;
                    }
                    else
                    {
                        const float4 v = *reinterpret_cast<const float4*>(xs + (j * 8 + gid) * G::XROW + s16 * 64 + 16 * lid);
                        b0[j] = pack_bf16(v.x, v.y); b1[j] = pack_bf16(v.z, v.w);
                    }
                }
#pragma unroll
                for (int r = 0; r < RT; ++r)
                {
                    const uint8_t* wr = ws + r * 16 * G::WROW;
                    uint32_t a0, a1, a2, a3;
                    if constexpr (FMT == kBf16)
                    {
                        const uint2 lo = *reinterpret_cast<const uint2*>(wr + gid * G::WROW + s16 * 32 + 8 * lid);
                        const uint2 hi = *reinterpret_cast<const uint2*>(wr + (gid + 8) * G::WROW + s16 * 32 + 8 * lid);
                        a0 = lo.x; a2 = lo.y; a1 = hi.x; a3 = hi.y;
                    }
                    else
                    {
                        const uint32_t lo = *reinterpret_cast<const uint32_t*>(wr + gid * G::WROW + s16 * 16 + 4 * lid);
                        const uint32_t hi = *reinterpret_cast<const uint32_t*>(wr + (gid + 8) * G::WROW + s16 * 16 + 4 * lid);
                        if constexpr (FMT == kFp8Row)
                        {
                            a0 = e4m3x2_bf16x2(lo); a2 = e4m3x2_bf16x2(lo >> 16);
                            a1 = e4m3x2_bf16x2(hi); a3 = e4m3x2_bf16x2(hi >> 16);
                        }
                        else
                        {
                            a0 = s8x2_bf16x2(lo); a2 = s8x2_bf16x2(lo >> 16);
                            a1 = s8x2_bf16x2(hi); a3 = s8x2_bf16x2(hi >> 16);
                        }
                    }
#pragma unroll
                    for (int j = 0; j < TT / 8; ++j)
                    {
                        if constexpr (FMT == kW8G32) mma(part[r][j], a0, a1, a2, a3, b0[j], b1[j]);
                        else mma(acc[r][j], a0, a1, a2, a3, b0[j], b1[j]);
                    }
                }
            }
            if constexpr (FMT == kW8G32)
            {
#pragma unroll
                for (int r = 0; r < RT; ++r)
#pragma unroll
                    for (int j = 0; j < TT / 8; ++j)
                    {
                        acc[r][j][0] = fmaf(part[r][j][0], gs[g][r][0], acc[r][j][0]);
                        acc[r][j][1] = fmaf(part[r][j][1], gs[g][r][0], acc[r][j][1]);
                        acc[r][j][2] = fmaf(part[r][j][2], gs[g][r][1], acc[r][j][2]);
                        acc[r][j][3] = fmaf(part[r][j][3], gs[g][r][1], acc[r][j][3]);
                    }
            }
        }
    }
    wait<0>();
    __syncthreads();

    // Cross-warp sum in a fixed order: warps 1..KW-1 publish, warp 0 adds and stores.
    float* red = reinterpret_cast<float*>(smem);
    if (warp > 0)
#pragma unroll
        for (int r = 0; r < RT; ++r)
#pragma unroll
            for (int j = 0; j < TT / 8; ++j)
                *reinterpret_cast<float4*>(red + (((warp * RT + r) * (TT / 8) + j) * 32 + lane) * 4) = make_float4(acc[r][j][0], acc[r][j][1], acc[r][j][2], acc[r][j][3]);
    __syncthreads();
    if (warp != 0) return;
#pragma unroll
    for (int r = 0; r < RT; ++r)
#pragma unroll
        for (int j = 0; j < TT / 8; ++j)
            for (int v = 1; v < KW; ++v)
            {
                const float4 p = *reinterpret_cast<const float4*>(red + (((v * RT + r) * (TT / 8) + j) * 32 + lane) * 4);
                acc[r][j][0] += p.x; acc[r][j][1] += p.y; acc[r][j][2] += p.z; acc[r][j][3] += p.w;
            }

    auto store = [&](int token, int row, float value) {
        if (token >= m) return;
        if constexpr (YBF)
            reinterpret_cast<__nv_bfloat16*>(yv)[(long long)token * ldy + row] = __float2bfloat16_rn(value);
        else
            reinterpret_cast<float*>(yv)[(long long)token * ldy + row] = OUT == kOutPlain ? value : round_bf16(value);
    };
#pragma unroll
    for (int r = 0; r < RT; ++r)
    {
        const int rb = row0 + r * 16;
        float s0 = 1.0f, s1 = 1.0f;
        if constexpr (FMT == kFp8Row)
        {
            const uint16_t* rs = reinterpret_cast<const uint16_t*>(sc);
            s0 = bf16_bits(rs[rb + gid]);
            s1 = bf16_bits(rs[rb + gid + 8]);
        }
#pragma unroll
        for (int j = 0; j < TT / 8; ++j)
        {
            const int t = tok0 + j * 8 + 2 * lid;
            if constexpr (OUT == kOutSwiGlu)
            {
                const int out = (blockIdx.x * RT + r) * 8 + gid;
                store(t, out, silu(acc[r][j][0] * s0) * (acc[r][j][2] * s1));
                store(t + 1, out, silu(acc[r][j][1] * s0) * (acc[r][j][3] * s1));
            }
            else
            {
                store(t, rb + gid, acc[r][j][0] * s0);
                store(t + 1, rb + gid, acc[r][j][1] * s0);
                store(t, rb + gid + 8, acc[r][j][2] * s1);
                store(t + 1, rb + gid + 8, acc[r][j][3] * s1);
            }
        }
    }
}

// Two E4M3FN codes (low byte = lower K) -> F32 pair, exact.
__device__ __forceinline__ float2 e4m3x2_f32x2(uint32_t two)
{
    uint32_t h;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(h) : "h"((unsigned short)(two & 0xffffu)));
    return __half22float2(*reinterpret_cast<const __half2*>(&h));
}

template <int VPL>
__device__ __forceinline__ void load_codes(uint32_t (&dst)[VPL / 4], const uint8_t* p)
{
    if constexpr (VPL == 16)
        asm volatile("ld.global.nc.L1::no_allocate.v4.u32 {%0,%1,%2,%3}, [%4];\n"
                     : "=r"(dst[0]), "=r"(dst[1]), "=r"(dst[2]), "=r"(dst[3]) : "l"(p));
    else
        asm volatile("ld.global.nc.L1::no_allocate.v2.u32 {%0,%1}, [%2];\n" : "=r"(dst[0]), "=r"(dst[1]) : "l"(p));
}

/// T = 1 FP8 row-scaled projection on the CUDA cores (NInfer's fp8_a16_gemv structure,
/// src/ops/linear/kernels/fp8/fp8_a16_gemv.cuh at 1882c6ad; independent implementation). A warp
/// owns RPW weight rows and its lanes stream VPL consecutive codes of each per phase (32 * VPL
/// values a phase) with 16- or 8-byte loads that bypass L1, so the activation vector stays
/// L1-resident. Activations are BF16-rounded F32 (the small-T MMA operand values); products
/// accumulate in F32 in four chains, then a fixed-order warp reduction and one row scale.
/// SwiGLU: the warp's rows are RPW / 2 gate rows and their up rows (gate|up interleaved by 8
/// rows), y = BF16(silu(gate) * up) per output. grid.x = n / (WARPS * RPW), block = WARPS warps.
template <int WARPS, int RPW, int VPL, int OUT>
__device__ __forceinline__ void gemv(const uint8_t* __restrict__ w, const uint16_t* __restrict__ sc,
    const float* __restrict__ x, float* __restrict__ y, int k)
{
    static_assert(VPL == 8 || VPL == 16, "codes per lane");
    static_assert(OUT != kOutSwiGlu || RPW % 2 == 0, "SwiGLU rows come in gate/up pairs");
    constexpr int CH = 4;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    int rows[RPW];
    if constexpr (OUT == kOutSwiGlu)
    {
#pragma unroll
        for (int p = 0; p < RPW / 2; ++p)
        {
            const int o = (blockIdx.x * WARPS + warp) * (RPW / 2) + p;
            rows[p] = (o >> 3) * 16 + (o & 7);
            rows[p + RPW / 2] = rows[p] + 8;
        }
    }
    else
    {
#pragma unroll
        for (int r = 0; r < RPW; ++r) rows[r] = (blockIdx.x * WARPS + warp) * RPW + r;
    }
    float acc[RPW][CH];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
#pragma unroll
        for (int c = 0; c < CH; ++c) acc[r][c] = 0.0f;

    const int phases = k / (32 * VPL);
#pragma unroll 2
    for (int ph = 0; ph < phases; ++ph)
    {
        const int v0 = ph * 32 * VPL + lane * VPL;
        uint32_t codes[RPW][VPL / 4];
#pragma unroll
        for (int r = 0; r < RPW; ++r) load_codes<VPL>(codes[r], w + (long long)rows[r] * k + v0);
        float xv[VPL];
#pragma unroll
        for (int i = 0; i < VPL / 4; ++i)
        {
            const float4 f = __ldg(reinterpret_cast<const float4*>(x + v0) + i);
            xv[4 * i] = round_bf16(f.x); xv[4 * i + 1] = round_bf16(f.y);
            xv[4 * i + 2] = round_bf16(f.z); xv[4 * i + 3] = round_bf16(f.w);
        }
#pragma unroll
        for (int p = 0; p < VPL / 2; ++p)
#pragma unroll
            for (int r = 0; r < RPW; ++r)
            {
                const float2 v = e4m3x2_f32x2(codes[r][p >> 1] >> ((p & 1) * 16));
                acc[r][(2 * p) & (CH - 1)] = fmaf(v.x, xv[2 * p], acc[r][(2 * p) & (CH - 1)]);
                acc[r][(2 * p + 1) & (CH - 1)] = fmaf(v.y, xv[2 * p + 1], acc[r][(2 * p + 1) & (CH - 1)]);
            }
    }
    float total[RPW];
#pragma unroll
    for (int r = 0; r < RPW; ++r)
    {
        float s = (acc[r][0] + acc[r][1]) + (acc[r][2] + acc[r][3]);
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
        total[r] = s * bf16_bits(sc[rows[r]]);
    }
    if (lane != 0) return;
    if constexpr (OUT == kOutSwiGlu)
    {
#pragma unroll
        for (int p = 0; p < RPW / 2; ++p)
            y[(blockIdx.x * WARPS + warp) * (RPW / 2) + p] = round_bf16(silu(total[p]) * total[p + RPW / 2]);
    }
    else
    {
#pragma unroll
        for (int r = 0; r < RPW; ++r) y[rows[r]] = OUT == kOutPlain ? total[r] : round_bf16(total[r]);
    }
}

} // namespace ts_smt
