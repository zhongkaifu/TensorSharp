// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// 8-bit prompt attention for Qwen3.5/3.8 (24 query heads over 4 KV heads of 256), causal, one
// sequence. The design follows the FP8 path of mkyt/qwen3-27b-5090 (engine/src/attention.cu, MIT)
// as Ignis adapts it (kernel/src/hq_prompt_attention.cu at gpillon/ignis ad81d52e, Apache-2.0):
// keys are converted ONCE per chunk and layer into a "band" instead of per CTA tile, Q K^T runs on
// the INT8 tensor cores (exact int32), P V on the block-scaled E4M3 tensor cores (full rate on
// sm_120, where the plain FP8 form accumulates F32 at half rate). This is an independent
// TensorSharp implementation over TensorSharp's own cache formats:
//
//   band (ts_q35_attn8_band_*), per KV head, for keys [0, keys):
//     K8  [keysCap][256] int8, one F32 scale per key (amax / 127; Ignis uses one per 32-key tile)
//     VT  [tiles][256][32] E4M3, V transposed, the 32 keys of a tile in the P-fragment order below,
//         one UE8M0 power-of-two scale per tile (amax / scale <= 448)
//   attention (ts_q35_attn8_prefill_f32): GQA-packed rows as ts_q35_attn_prefill_f32 (row r =
//     token r / G, head kvHead * G + r % G), 128 rows per CTA (8 warps x 16), 32-key tiles in two
//     cp.async stages. Q is quantized per row in registers (amax / 127). Scores are
//     S_int * qScale * kScale * scale * log2(e); the online softmax runs in base 2 with
//     P = 2^(s - m + 8) in E4M3 (at most 256, so small probabilities stay normal), the row sum l
//     taken over the rounded P the value product actually uses. O / l, then * sigmoid(gate).
//
// Key order in a VT tile: A-fragment column j = 4 * q + i (quad lane q, byte i) of a 32-key
// m16n8k32 step holds key perm(j) = 16 * (j >> 4) + {2q, 2q + 1, 8 + 2q, 9 + 2q}[i], exactly the
// keys a thread's S accumulators of n-tiles (0,1) and (2,3) hold, so P never leaves registers.
#pragma once

#define TS_Q35_A8_KT 32                    // keys per tile
#define TS_Q35_A8_ROWS 128                 // GQA rows per CTA
#define TS_Q35_A8_KSTRIDE (TS_Q35_HD + 16) // smem K row: 272 bytes, conflict-free B fragments
#define TS_Q35_A8_VSTRIDE (TS_Q35_A8_KT + 16) // smem V^T row: 48 bytes

__device__ __forceinline__ int ts_q35_a8_perm(int j)
{
    const int q = (j & 15) >> 2, i = j & 3;
    return (j & 16) + (i < 2 ? 2 * q + i : 8 + 2 * q + i - 2);
}

__device__ __forceinline__ float ts_q35_a8_ex2(float x)
{
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;" : "=f"(y) : "f"(x));
    return y;
}

__device__ __forceinline__ void ts_q35_mma_s8(int (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(d[0]), "+r"(d[1]), "+r"(d[2]), "+r"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

// D += A * (B * 2^(sfb - 127)), E4M3 operands, A unscaled. sfb: the same UE8M0 byte in every lane.
__device__ __forceinline__ void ts_q35_mma_e4m3_sb(float (&d)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1, uint32_t sfb)
{
#if defined(TS_Q35_FP4_MMA)
    asm volatile("mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e4m3.e4m3.f32.ue8m0 "
                 "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3}, {%10}, {%11,%12}, {%13}, {%11,%12};"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0x7f7f7f7fu),
                   "h"((unsigned short)0), "h"((unsigned short)0), "r"(sfb));
#else
    __trap();
#endif
}

// Two F32 -> E4M3 (satfinite), lo in the low byte.
__device__ __forceinline__ uint32_t ts_q35_e4m3x2(float lo, float hi)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    unsigned short r;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;" : "=h"(r) : "f"(hi), "f"(lo));
    return r;
#else
    __trap();
    return 0;
#endif
}

// Sum of the four E4M3 bytes of a word, as F32 (the rounded probabilities the MMA consumes).
__device__ __forceinline__ float ts_q35_e4m3x4_sum(uint32_t w)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 890
    uint32_t lo, hi;
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(lo) : "h"((unsigned short)(w & 0xffffu)));
    asm("cvt.rn.f16x2.e4m3x2 %0, %1;" : "=r"(hi) : "h"((unsigned short)(w >> 16)));
    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&lo));
    const float2 b = __half22float2(*reinterpret_cast<const half2*>(&hi));
    return (a.x + a.y) + (b.x + b.y);
#else
    __trap();
    return 0.0f;
#endif
}

// Writes one 32-key tile of one KV head into the band: K8 rows, their scales, the V^T tile and its
// scale. kv[key][256] are the tile's F32 values (K then V halves of `vals`), staged by the caller.
__device__ __forceinline__ void ts_q35_a8_band_tile(const float (*k)[TS_Q35_HD], const float (*v)[TS_Q35_HD], int valid,
    int8_t* __restrict__ k8, float* __restrict__ ks, uint8_t* __restrict__ vt, uint8_t* __restrict__ vs, uint8_t* vtStage, float* red)
{
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    // K: warp w handles keys w, w + 8, w + 16, w + 24; a lane owns 8 dims.
    for (int key = warp; key < TS_Q35_A8_KT; key += 8)
    {
        float x[8], amax = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            x[e] = key < valid ? k[key][lane * 8 + e] : 0.0f;
            amax = fmaxf(amax, fabsf(x[e]));
        }
#pragma unroll
        for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
        const float s = amax > 0.0f ? amax / 127.0f : 1.0f, inv = 1.0f / s;
        uint32_t w0 = 0, w1 = 0;
#pragma unroll
        for (int e = 0; e < 8; ++e)
        {
            const int c = max(-127, min(127, __float2int_rn(x[e] * inv)));
            if (e < 4) w0 |= (uint32_t)(c & 0xff) << (8 * e); else w1 |= (uint32_t)(c & 0xff) << (8 * (e - 4));
        }
        *reinterpret_cast<uint2*>(k8 + (long long)key * TS_Q35_HD + lane * 8) = make_uint2(w0, w1);
        if (lane == 0) ks[key] = key < valid ? s : 0.0f;
    }
    // V: tile amax -> one power-of-two scale.
    float amax = 0.0f;
    for (int i = tid; i < TS_Q35_A8_KT * TS_Q35_HD; i += blockDim.x)
        if (i / TS_Q35_HD < valid) amax = fmaxf(amax, fabsf(v[i / TS_Q35_HD][i % TS_Q35_HD]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    if (lane == 0) red[warp] = amax;
    __syncthreads();
    if (tid == 0)
    {
        float m = 0.0f;
        for (int w = 0; w < (int)(blockDim.x >> 5); ++w) m = fmaxf(m, red[w]);
        // Smallest e with m / 2^e <= 448 (m == 0: e = 0).
        int e = 0;
        if (m > 0.0f) { frexpf(m / 448.0f, &e); if (ldexpf(1.0f, e - 1) >= m / 448.0f) --e; }
        e = max(-127, min(127, e));
        red[32] = ldexpf(1.0f, -e);
        *vs = (uint8_t)(e + 127);
    }
    __syncthreads();
    const float inv = red[32];
    // vtStage[d][j] = E4M3(V[perm(j)][d] * inv); then one coalesced copy of the 8 KB tile.
    for (int i = tid; i < TS_Q35_HD * TS_Q35_A8_KT / 2; i += blockDim.x)
    {
        const int d = i / (TS_Q35_A8_KT / 2), j = 2 * (i % (TS_Q35_A8_KT / 2));
        const int k0 = ts_q35_a8_perm(j), k1 = ts_q35_a8_perm(j + 1);
        const float a = k0 < valid ? v[k0][d] * inv : 0.0f, b = k1 < valid ? v[k1][d] * inv : 0.0f;
        *reinterpret_cast<unsigned short*>(vtStage + d * TS_Q35_A8_KT + j) = (unsigned short)ts_q35_e4m3x2(a, b);
    }
    __syncthreads();
    for (int i = tid; i < TS_Q35_HD * TS_Q35_A8_KT / 16; i += blockDim.x)
        reinterpret_cast<uint4*>(vt)[i] = reinterpret_cast<const uint4*>(vtStage)[i];
}

struct TsQ35A8Band
{
    float k[TS_Q35_A8_KT][TS_Q35_HD];
    float v[TS_Q35_A8_KT][TS_Q35_HD];
    uint8_t vt[TS_Q35_HD * TS_Q35_A8_KT];
    float red[33];
};

// One cache row's values 32b .. 32b + 31 as F32. FMT 0: TensorSharp q8_0 (272 B: codes, then 8 F16
// block scales; value = code * d, ggml's numerics); 1: BF16 (512 B); 2: NInfer FP8 (272 B: E4M3
// codes, then one F32 scale per 128 values).
template <int FMT>
__device__ __forceinline__ void ts_q35_a8_row32(const uint8_t* __restrict__ row, int b, float* __restrict__ dst)
{
    if constexpr (FMT == 1)
    {
#pragma unroll
        for (int c = 0; c < 4; ++c)
        {
            const uint4 w = *reinterpret_cast<const uint4*>(row + b * 64 + c * 16);
            const uint32_t a[4] = { w.x, w.y, w.z, w.w };
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                dst[c * 8 + 2 * e] = __uint_as_float(a[e] << 16);
                dst[c * 8 + 2 * e + 1] = __uint_as_float(a[e] & 0xffff0000u);
            }
        }
    }
    else
    {
        const float d = FMT == 0 ? __half2float(*reinterpret_cast<const half*>(row + TS_Q35_Q8_SCALES + 2 * b))
                                 : *reinterpret_cast<const float*>(row + 256 + 4 * (b >> 2));
#pragma unroll
        for (int c = 0; c < 2; ++c)
        {
            const uint4 w = *reinterpret_cast<const uint4*>(row + b * 32 + c * 16);
            const uint32_t a[4] = { w.x, w.y, w.z, w.w };
#pragma unroll
            for (int x = 0; x < 4; ++x)
            {
                if constexpr (FMT == 0)
                {
#pragma unroll
                    for (int e = 0; e < 4; ++e)
                        dst[c * 16 + x * 4 + e] = (float)(int8_t)(a[x] >> (8 * e)) * d;
                }
                else
                {
#pragma unroll
                    for (int e = 0; e < 2; ++e)
                    {
                        // cuda_fp8.h: PTX cvt.rn.f16x2.e4m3x2 on sm_89+, exact integer widening below.
                        const float2 f = __half22float2(__half2(__nv_cvt_fp8x2_to_halfraw2(
                            (__nv_fp8x2_storage_t)((a[x] >> (16 * e)) & 0xffffu), __NV_E4M3)));
                        dst[c * 16 + x * 4 + 2 * e] = f.x * d;
                        dst[c * 16 + x * 4 + 2 * e + 1] = f.y * d;
                    }
                }
            }
        }
    }
}

/// Band of one sequence's caches (rows of rowBytes in TS_Q35_KV_BLOCK-token blocks, each block
/// [nKv][TS_Q35_KV_BLOCK] rows at kOff / vOff): grid (tiles of keys [0, keys), nKv), 256 threads,
/// dynamic smem sizeof(TsQ35A8Band).
template <int FMT>
__device__ __forceinline__ void ts_q35_a8_band(uint8_t* const* __restrict__ blocks, long long kOff, long long vOff, int keys, int keysCap,
    int8_t* __restrict__ k8, float* __restrict__ ks, uint8_t* __restrict__ vt, uint8_t* __restrict__ vs)
{
    constexpr int ROW = FMT == 1 ? 512 : 272;
    extern __shared__ __align__(16) uint8_t ts_q35_a8_smem[];
    TsQ35A8Band& sm = *reinterpret_cast<TsQ35A8Band*>(ts_q35_a8_smem);
    const int tile = blockIdx.x, kvh = blockIdx.y, first = tile * TS_Q35_A8_KT, tid = threadIdx.x;
    const int valid = min(TS_Q35_A8_KT, keys - first);
    {
        const int key = tid >> 3, b = tid & 7;   // thread = (key, 32-value block)
        if (key < valid)
        {
            const int t = first + key;
            const uint8_t* base = blocks[t / TS_Q35_KV_BLOCK] + ((long long)kvh * TS_Q35_KV_BLOCK + t % TS_Q35_KV_BLOCK) * ROW;
            ts_q35_a8_row32<FMT>(base + kOff, b, &sm.k[key][b * 32]);
            ts_q35_a8_row32<FMT>(base + vOff, b, &sm.v[key][b * 32]);
        }
    }
    __syncthreads();
    const long long head = (long long)kvh * keysCap;
    ts_q35_a8_band_tile(sm.k, sm.v, valid, k8 + (head + first) * TS_Q35_HD, ks + head + first,
                        vt + (head + first) * TS_Q35_HD, vs + (long long)kvh * (keysCap / TS_Q35_A8_KT) + tile, sm.vt, sm.red);
}

#define TS_Q35_A8_BAND(NAME, FMT) \
    extern "C" __global__ void __launch_bounds__(256) NAME(uint8_t* const* __restrict__ blocks, long long kOff, long long vOff, \
        int keys, int keysCap, int8_t* __restrict__ k8, float* __restrict__ ks, uint8_t* __restrict__ vt, uint8_t* __restrict__ vs) \
    { ts_q35_a8_band<FMT>(blocks, kOff, vOff, keys, keysCap, k8, ks, vt, vs); }
TS_Q35_A8_BAND(ts_q35_attn8_band_q8, 0)
TS_Q35_A8_BAND(ts_q35_attn8_band_bf16, 1)
TS_Q35_A8_BAND(ts_q35_attn8_band_fp8, 2)

struct TsQ35A8Smem
{
    int8_t k[2][TS_Q35_A8_KT * TS_Q35_A8_KSTRIDE];
    uint8_t vt[2][TS_Q35_HD * TS_Q35_A8_VSTRIDE];
    float ks[2][TS_Q35_A8_KT];
};

/// out[token][head][256] = softmax(Q K^T * scale) V * sigmoid(gate) for one sequence of rowsPerSeq
/// tokens over the band of its keys; token t sees keys [0, keyEnd[t]). grid (ceil(rowsPerSeq * G /
/// 128), nKv), 256 threads, dynamic smem sizeof(TsQ35A8Smem).
extern "C" __global__ void __launch_bounds__(256, 1) ts_q35_attn8_prefill_f32(
    const float* __restrict__ q, const float* __restrict__ gate, const int* __restrict__ keyEnd,
    const int8_t* __restrict__ k8, const float* __restrict__ ks, const uint8_t* __restrict__ vt, const uint8_t* __restrict__ vs,
    int keysCap, int rowsPerSeq, int nHead, int nKv, float scale, float* __restrict__ out)
{
    extern __shared__ __align__(16) uint8_t ts_q35_a8_smem[];
    TsQ35A8Smem& sm = *reinterpret_cast<TsQ35A8Smem*>(ts_q35_a8_smem);
    const int kvh = blockIdx.y, G = nHead / nKv, R = rowsPerSeq * G;
    const int rowBase = blockIdx.x * TS_Q35_A8_ROWS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
    const int kAll = keyEnd[(min(rowBase + TS_Q35_A8_ROWS, R) - 1) / G];
    const long long head = (long long)kvh * keysCap;
    const int8_t* kh = k8 + head * TS_Q35_HD;
    const float* ksh = ks + head;
    const uint8_t* vth = vt + head * TS_Q35_HD;
    const uint8_t* vsh = vs + (long long)kvh * (keysCap / TS_Q35_A8_KT);

    const int r0 = rowBase + warp * 16 + g, r1 = r0 + 8;
    const bool live0 = r0 < R, live1 = r1 < R;
    const int end0 = live0 ? keyEnd[r0 / G] : 0, end1 = live1 ? keyEnd[r1 / G] : 0;
    const bool warpLive = rowBase + warp * 16 < R;
    int warpEnd = max(end0, end1);   // MMAs need the whole warp: skip tiles past every row's end
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) warpEnd = max(warpEnd, __shfl_xor_sync(0xffffffffu, warpEnd, o));

    auto load = [&](int st, int first) {
        const int8_t* kt = kh + (long long)first * TS_Q35_HD;
        const uint8_t* vtt = vth + (long long)first * TS_Q35_HD;
        for (int c = tid; c < TS_Q35_A8_KT * TS_Q35_HD / 16; c += blockDim.x)
        {
            const int key = c / (TS_Q35_HD / 16), part = c % (TS_Q35_HD / 16);
            ts_q35_cp16(sm.k[st] + key * TS_Q35_A8_KSTRIDE + part * 16, kt + c * 16, true);
            const int d = c / (TS_Q35_A8_KT / 16), half_ = c % (TS_Q35_A8_KT / 16);
            ts_q35_cp16(sm.vt[st] + d * TS_Q35_A8_VSTRIDE + half_ * 16, vtt + c * 16, true);
        }
        if (tid < TS_Q35_A8_KT / 4)
            ts_q35_cp16(sm.ks[st] + tid * 4, ksh + first + tid * 4, true);
    };
    if (kAll > 0)
        load(0, 0);
    ts_q35_cp_commit();

    // Q rows in registers as INT8 A fragments with one F32 scale per row (amax over the quad).
    uint32_t qa[8][4];
    float qs0 = 0.0f, qs1 = 0.0f;
    {
        const float* q0 = live0 ? q + ((long long)(r0 / G) * nHead + kvh * G + r0 % G) * TS_Q35_HD : nullptr;
        const float* q1 = live1 ? q + ((long long)(r1 / G) * nHead + kvh * G + r1 % G) * TS_Q35_HD : nullptr;
        float a0 = 0.0f, a1 = 0.0f;
#pragma unroll
        for (int k = 0; k < 8; ++k)
#pragma unroll
            for (int h = 0; h < 2; ++h)
            {
                const int c = k * 32 + h * 16 + 4 * qd;
                if (q0) { const float4 v = *reinterpret_cast<const float4*>(q0 + c); a0 = fmaxf(a0, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)))); }
                if (q1) { const float4 v = *reinterpret_cast<const float4*>(q1 + c); a1 = fmaxf(a1, fmaxf(fmaxf(fabsf(v.x), fabsf(v.y)), fmaxf(fabsf(v.z), fabsf(v.w)))); }
            }
        a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 1)); a0 = fmaxf(a0, __shfl_xor_sync(0xffffffffu, a0, 2));
        a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 1)); a1 = fmaxf(a1, __shfl_xor_sync(0xffffffffu, a1, 2));
        qs0 = a0 > 0.0f ? a0 / 127.0f : 1.0f;
        qs1 = a1 > 0.0f ? a1 / 127.0f : 1.0f;
        const float i0 = 1.0f / qs0, i1 = 1.0f / qs1;
        auto pack = [](const float* p, float inv) {
            if (!p) return 0u;
            const float4 v = *reinterpret_cast<const float4*>(p);
            const int c0 = max(-127, min(127, __float2int_rn(v.x * inv))), c1 = max(-127, min(127, __float2int_rn(v.y * inv)));
            const int c2 = max(-127, min(127, __float2int_rn(v.z * inv))), c3 = max(-127, min(127, __float2int_rn(v.w * inv)));
            return (uint32_t)(c0 & 0xff) | (uint32_t)(c1 & 0xff) << 8 | (uint32_t)(c2 & 0xff) << 16 | (uint32_t)(c3 & 0xff) << 24;
        };
#pragma unroll
        for (int k = 0; k < 8; ++k)
        {
            const int c = k * 32 + 4 * qd;
            qa[k][0] = pack(q0 ? q0 + c : nullptr, i0);
            qa[k][1] = pack(q1 ? q1 + c : nullptr, i1);
            qa[k][2] = pack(q0 ? q0 + c + 16 : nullptr, i0);
            qa[k][3] = pack(q1 ? q1 + c + 16 : nullptr, i1);
        }
    }
    const float log2Scale = scale * 1.4426950408889634f;
    const float sc0 = qs0 * log2Scale, sc1 = qs1 * log2Scale;

    float acc[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int first = 0, t = 0; first < kAll; first += TS_Q35_A8_KT, ++t)
    {
        const int st = t & 1;
        ts_q35_cp_wait<0>();
        __syncthreads();   // this tile landed; every warp is done with the previous one
        if (first + TS_Q35_A8_KT < kAll)
            load(st ^ 1, first + TS_Q35_A8_KT);
        ts_q35_cp_commit();
        if (!warpLive || first >= warpEnd)
            continue;

        // S = Q K^T (int32, exact) for 4 n-tiles of 8 keys.
        int si[4][4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) si[nt][0] = si[nt][1] = si[nt][2] = si[nt][3] = 0;
        const int8_t* kb = sm.k[st] + g * TS_Q35_A8_KSTRIDE + 4 * qd;
#pragma unroll
        for (int k = 0; k < 8; ++k)
#pragma unroll
            for (int nt = 0; nt < 4; ++nt)
            {
                const int8_t* p = kb + nt * 8 * TS_Q35_A8_KSTRIDE + k * 32;
                ts_q35_mma_s8(si[nt], qa[k], *reinterpret_cast<const uint32_t*>(p), *reinterpret_cast<const uint32_t*>(p + 16));
            }

        // Base-2 scores, causal mask, tile maxima.
        float s[4][4], tm0 = -INFINITY, tm1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 4; ++nt)
#pragma unroll
            for (int e = 0; e < 2; ++e)
            {
                const int key = first + nt * 8 + 2 * qd + e;
                const float kscale = sm.ks[st][nt * 8 + 2 * qd + e];
                s[nt][e] = key < end0 ? (float)si[nt][e] * sc0 * kscale : -INFINITY;
                s[nt][2 + e] = key < end1 ? (float)si[nt][2 + e] * sc1 * kscale : -INFINITY;
                tm0 = fmaxf(tm0, s[nt][e]);
                tm1 = fmaxf(tm1, s[nt][2 + e]);
            }
        tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 1)); tm0 = fmaxf(tm0, __shfl_xor_sync(0xffffffffu, tm0, 2));
        tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 1)); tm1 = fmaxf(tm1, __shfl_xor_sync(0xffffffffu, tm1, 2));
        const float n0 = fmaxf(m0, tm0), n1 = fmaxf(m1, tm1);
        const float al0 = n0 == -INFINITY ? 1.0f : ts_q35_a8_ex2(m0 - n0), al1 = n1 == -INFINITY ? 1.0f : ts_q35_a8_ex2(m1 - n1);
        m0 = n0; m1 = n1;
        const float b0 = n0 == -INFINITY ? 0.0f : 8.0f - n0, b1 = n1 == -INFINITY ? 0.0f : 8.0f - n1;

        // P = 2^(s - m + 8) in E4M3, straight into the A fragment (see the key order above).
        auto p2 = [](float x, float bias) { return x == -INFINITY ? 0.0f : ts_q35_a8_ex2(x + bias); };
        uint32_t pa[4];
        pa[0] = ts_q35_e4m3x2(p2(s[0][0], b0), p2(s[0][1], b0)) | ts_q35_e4m3x2(p2(s[1][0], b0), p2(s[1][1], b0)) << 16;
        pa[1] = ts_q35_e4m3x2(p2(s[0][2], b1), p2(s[0][3], b1)) | ts_q35_e4m3x2(p2(s[1][2], b1), p2(s[1][3], b1)) << 16;
        pa[2] = ts_q35_e4m3x2(p2(s[2][0], b0), p2(s[2][1], b0)) | ts_q35_e4m3x2(p2(s[3][0], b0), p2(s[3][1], b0)) << 16;
        pa[3] = ts_q35_e4m3x2(p2(s[2][2], b1), p2(s[2][3], b1)) | ts_q35_e4m3x2(p2(s[3][2], b1), p2(s[3][3], b1)) << 16;
        float ps0 = ts_q35_e4m3x4_sum(pa[0]) + ts_q35_e4m3x4_sum(pa[2]);
        float ps1 = ts_q35_e4m3x4_sum(pa[1]) + ts_q35_e4m3x4_sum(pa[3]);
        ps0 += __shfl_xor_sync(0xffffffffu, ps0, 1); ps0 += __shfl_xor_sync(0xffffffffu, ps0, 2);
        ps1 += __shfl_xor_sync(0xffffffffu, ps1, 1); ps1 += __shfl_xor_sync(0xffffffffu, ps1, 2);
        l0 = fmaf(l0, al0, ps0);
        l1 = fmaf(l1, al1, ps1);
        if (__any_sync(0xffffffffu, al0 != 1.0f || al1 != 1.0f))
        {
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                acc[n][0] *= al0; acc[n][1] *= al0;
                acc[n][2] *= al1; acc[n][3] *= al1;
            }
        }

        // O += P V^T: 32 n-tiles of 8 dims, one k32 step (the tile), V's tile scale as the B scale.
        const uint32_t sfb = vsh[first / TS_Q35_A8_KT] * 0x01010101u;
        const uint8_t* vb = sm.vt[st] + g * TS_Q35_A8_VSTRIDE + 4 * qd;
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
            const uint8_t* p = vb + n * 8 * TS_Q35_A8_VSTRIDE;
            ts_q35_mma_e4m3_sb(acc[n], pa, *reinterpret_cast<const uint32_t*>(p), *reinterpret_cast<const uint32_t*>(p + 16), sfb);
        }
    }
    ts_q35_cp_wait<0>();

    // O / l (the 2^8 of P cancels), gated, F32 rows [token][head][256].
    const float inv0 = l0 > 0.0f ? 1.0f / l0 : 0.0f, inv1 = l1 > 0.0f ? 1.0f / l1 : 0.0f;
    float* o0 = live0 ? out + ((long long)(r0 / G) * nHead + kvh * G + r0 % G) * TS_Q35_HD : nullptr;
    float* o1 = live1 ? out + ((long long)(r1 / G) * nHead + kvh * G + r1 % G) * TS_Q35_HD : nullptr;
    const float* g0 = live0 ? gate + ((long long)(r0 / G) * nHead + kvh * G + r0 % G) * TS_Q35_HD : nullptr;
    const float* g1 = live1 ? gate + ((long long)(r1 / G) * nHead + kvh * G + r1 % G) * TS_Q35_HD : nullptr;
#pragma unroll
    for (int n = 0; n < 32; ++n)
    {
        const int d = n * 8 + 2 * qd;
        if (o0)
        {
            o0[d] = acc[n][0] * inv0 * ts_q35_sigmoid(g0[d]);
            o0[d + 1] = acc[n][1] * inv0 * ts_q35_sigmoid(g0[d + 1]);
        }
        if (o1)
        {
            o1[d] = acc[n][2] * inv1 * ts_q35_sigmoid(g1[d]);
            o1[d + 1] = acc[n][3] * inv1 * ts_q35_sigmoid(g1[d + 1]);
        }
    }
}

// Dynamic shared memory of the band and attention kernels, for the host.
extern "C" __global__ void ts_q35_attn8_smem_bytes(int* out)
{
    out[0] = (int)sizeof(TsQ35A8Band);
    out[1] = (int)sizeof(TsQ35A8Smem);
}
