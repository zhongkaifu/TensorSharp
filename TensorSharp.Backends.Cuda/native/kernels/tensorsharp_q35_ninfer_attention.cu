// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// NInfer BF16 KV attention. Shares the Qwen block-table/slot geometry, while
// retaining BF16 Q/K/V/P operands and F32 tensor-core accumulation.
#include <cuda_bf16.h>
#include <stdint.h>
#define TS_NINFER_HD 256
#define TS_NINFER_KV_BLOCK 256
#define TS_NINFER_ROPE_DIMS 64
#define TS_NINFER_Q8_ROW 512 // BF16 cache bytes, not q8 storage
__device__ __forceinline__ float ts_ninfer_bf(float v) { return __bfloat162float(__float2bfloat16_rn(v)); }
__device__ __forceinline__ float ts_ninfer_warp_sum(float v) {
#pragma unroll
 for(int off=16;off;off>>=1) v+=__shfl_xor_sync(0xffffffff,v,off); return v;
}
__device__ __forceinline__ float ts_ninfer_sigmoid(float v) { return 1.0f/(1.0f+expf(-v)); }
__device__ __forceinline__ uint32_t ts_ninfer_smem_u32(const void* v) { return static_cast<uint32_t>(__cvta_generic_to_shared(v)); }
extern "C" __global__ void ts_ninfer_attn_prep_f32(
    const float* __restrict__ qkv, int rows, int rowsPerSeq, int nHead, int nKv,
    const float* __restrict__ qNorm, const float* __restrict__ kNorm, float eps,
    const int* __restrict__ ropePos, int ropeStride, int sectT, int sectH, int sectW, float ropeBase,
    const int* __restrict__ cacheRow, uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    float* __restrict__ q, float* __restrict__ gate, const float* __restrict__ invFrequency)
{
    const int units = nHead + 2 * nKv;
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (warp >= (long long)rows * units)
        return;
    const int row = (int)(warp / units), unit = (int)(warp % units);
    const int seq = row / rowsPerSeq;
    const long long rowWidth = (long long)nHead * 2 * TS_NINFER_HD + 2LL * nKv * TS_NINFER_HD;
    const float* src;
    if (unit < nHead)
        src = qkv + row * rowWidth + (long long)unit * 2 * TS_NINFER_HD;
    else if (unit < nHead + nKv)
        src = qkv + row * rowWidth + (long long)nHead * 2 * TS_NINFER_HD + (long long)(unit - nHead) * TS_NINFER_HD;
    else   // V follows the nKv key heads
        src = qkv + row * rowWidth + (long long)nHead * 2 * TS_NINFER_HD + (long long)(unit - nHead) * TS_NINFER_HD;

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
        ss = ts_ninfer_warp_sum(ss);
        const float scale = rsqrtf(ss / (float)TS_NINFER_HD + eps);
        const float* nw = unit < nHead ? qNorm : kNorm;
#pragma unroll
        for (int j = 0; j < 8; ++j)
            v[j] = ts_ninfer_bf(v[j] * scale * nw[lane + 32 * j]);
        const int pair = lane;   // 0..31
        int comp;
        if (pair % 3 == 1 && pair < 3 * sectH) comp = 1;
        else if (pair % 3 == 2 && pair < 3 * sectW) comp = 2;
        else if (pair % 3 == 0 && pair < 3 * sectT) comp = 0;
        else comp = 3;
        const float p = (float)ropePos[(long long)comp * ropeStride + row];
        const float theta = p * invFrequency[pair];
        float sn, cs;
        sincosf(theta, &sn, &cs);
        const float x0 = v[0], x1 = v[1];
        v[0] = ts_ninfer_bf(x0 * cs - x1 * sn);
        v[1] = ts_ninfer_bf(x0 * sn + x1 * cs);
    }
    if (unit < nHead)
    {
        float* qd = q + ((long long)row * nHead + unit) * TS_NINFER_HD;
        float* gd = gate + ((long long)row * nHead + unit) * TS_NINFER_HD;
#pragma unroll
        for (int j = 0; j < 8; ++j)
        {
            qd[lane + 32 * j] = v[j];
            gd[lane + 32 * j] = src[TS_NINFER_HD + lane + 32 * j];
        }
        return;
    }
    const bool isK = unit < nHead + nKv;
    const int head = isK ? unit - nHead : unit - nHead - nKv;
    const int cr = cacheRow[row];
    uint8_t* cache = blockTable[(long long)seq * maxBlocks + cr / TS_NINFER_KV_BLOCK] + (isK ? kOff : vOff)
                   + ((long long)head * TS_NINFER_KV_BLOCK + cr % TS_NINFER_KV_BLOCK) * TS_NINFER_Q8_ROW;
#pragma unroll
    for (int j = 0; j < 8; ++j)
        ((__nv_bfloat16*)cache)[32*j+lane] = __float2bfloat16_rn(v[j]);
}

#define TS_NINFER_ATTN_KT 32
#define TS_NINFER_ATTN_ROW (TS_NINFER_HD + 8)   // F16 tile row stride: 528 bytes, bank-spread

__device__ __forceinline__ void ts_ninfer_mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ __forceinline__ uint32_t ts_ninfer_pack_b2(float lo, float hi)
{
    const __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

__device__ __forceinline__ void ts_ninfer_ldsm_x2_t(uint32_t& b0, uint32_t& b1, const __nv_bfloat16* row)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(b0), "=r"(b1) : "r"(ts_ninfer_smem_u32(row)));
#else
    b0 = b1 = 0;
    __trap();
#endif
}

// 32 keys x 256 values of one head from a q8_0 cache into F16 rows; keys past `limit` are zero
// (finite, so their masked probabilities times V stay zero).
__device__ __forceinline__ void ts_ninfer_dequant_tile(const uint8_t* head, long long stride, int firstKey, int limit, __nv_bfloat16* tile)
{
 for(int i=threadIdx.x;i<TS_NINFER_ATTN_KT*TS_NINFER_HD;i+=blockDim.x) {
  const int key=i/TS_NINFER_HD,d=i%TS_NINFER_HD;
  tile[key*TS_NINFER_ATTN_ROW+d]=firstKey+key<limit ? ((const __nv_bfloat16*)(head+(long long)(firstKey+key)*stride))[d] : __float2bfloat16_rn(0.0f);
 }
}

template <bool SPLIT>
__device__ __forceinline__ void ts_ninfer_attention(
    const float* __restrict__ q, const float* __restrict__ gate, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, int chunk, int splits,
    float* __restrict__ partial, float* __restrict__ out)
{
    __shared__ __align__(16) __nv_bfloat16 ks[TS_NINFER_ATTN_KT * TS_NINFER_ATTN_ROW];
    __shared__ __align__(16) __nv_bfloat16 vs[TS_NINFER_ATTN_KT * TS_NINFER_ATTN_ROW];
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
    // Cache blocks hold this head's TS_NINFER_KV_BLOCK keys contiguously, and a 32-key tile never
    // leaves its block.
    const long long keyStride = TS_NINFER_Q8_ROW;
    uint8_t* const* blocks = blockTable + (long long)seq * maxBlocks;

    // This warp's rows: r0 = rowBase + warp * 16 + g, r1 = r0 + 8.
    const int r0 = rowBase + warp * 16 + g, r1 = r0 + 8;
    const bool live0 = r0 < R, live1 = r1 < R;
    const int end0 = live0 ? ends[r0 / G] : 0, end1 = live1 ? ends[r1 / G] : 0;
    auto qptr = [&](int r) {
        const int t = r / G, h = kvh * G + r % G;
        return q + (((long long)seq * rowsPerSeq + t) * nHead + h) * TS_NINFER_HD;
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
            qa[kk][0] = q0 ? ts_ninfer_pack_b2(q0[c], q0[c + 1]) : 0u;
            qa[kk][1] = q1 ? ts_ninfer_pack_b2(q1[c], q1[c + 1]) : 0u;
            qa[kk][2] = q0 ? ts_ninfer_pack_b2(q0[c + 8], q0[c + 9]) : 0u;
            qa[kk][3] = q1 ? ts_ninfer_pack_b2(q1[c + 8], q1[c + 9]) : 0u;
        }
    }
    float acc[32][4];
#pragma unroll
    for (int n = 0; n < 32; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float log2Scale = scale * 1.4426950408889634f;

    for (int first = kLo; first < kHi; first += TS_NINFER_ATTN_KT)
    {
        __syncthreads();
        const uint8_t* tileBase = blocks[first / TS_NINFER_KV_BLOCK]
                                + ((long long)kvh * TS_NINFER_KV_BLOCK + first % TS_NINFER_KV_BLOCK) * TS_NINFER_Q8_ROW;
        ts_ninfer_dequant_tile(tileBase + kOff, keyStride, 0, kHi - first, ks);
        ts_ninfer_dequant_tile(tileBase + vOff, keyStride, 0, kHi - first, vs);
        __syncthreads();
        if (!warpLive)
            continue;
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
            const __nv_bfloat16* kb = ks + (j * 8 + g) * TS_NINFER_ATTN_ROW + qd * 2;
#pragma unroll
            for (int kk = 0; kk < 16; ++kk)
                ts_ninfer_mma_bf16(s[j], qa[kk], *(const uint32_t*)(kb + kk * 16), *(const uint32_t*)(kb + kk * 16 + 8));
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
                ts_ninfer_pack_b2(s[2 * kk][0], s[2 * kk][1]), ts_ninfer_pack_b2(s[2 * kk][2], s[2 * kk][3]),
                ts_ninfer_pack_b2(s[2 * kk + 1][0], s[2 * kk + 1][1]), ts_ninfer_pack_b2(s[2 * kk + 1][2], s[2 * kk + 1][3]) };
            const __nv_bfloat16* vrow = vs + (kk * 16 + (lane & 15)) * TS_NINFER_ATTN_ROW;
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                uint32_t b0, b1;
                ts_ninfer_ldsm_x2_t(b0, b1, vrow + n * 8);
                ts_ninfer_mma_bf16(acc[n], pa, b0, b1);
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
            float* p = partial + (base * splits + blockIdx.x) * (TS_NINFER_HD + 2);
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
                p[TS_NINFER_HD] = m == -INFINITY ? -INFINITY : m * 0.6931471805599453f;   // log2 -> ln domain
                p[TS_NINFER_HD + 1] = half_ ? l1 : l0;
            }
        }
        else
        {
            const float l = half_ ? l1 : l0;
            const float inv = l > 0.0f ? 1.0f / l : 0.0f;
            float* o = out + base * TS_NINFER_HD;
            const float* gt = gate + base * TS_NINFER_HD;
#pragma unroll
            for (int n = 0; n < 32; ++n)
            {
                const int x = n * 8 + qd * 2;
                o[x] = ts_ninfer_bf(ts_ninfer_bf(acc[n][half_ * 2] * inv) * ts_ninfer_sigmoid(gt[x]));
                o[x + 1] = ts_ninfer_bf(ts_ninfer_bf(acc[n][half_ * 2 + 1] * inv) * ts_ninfer_sigmoid(gt[x + 1]));
            }
        }
    }
}

extern "C" __global__ void __launch_bounds__(256) ts_ninfer_attn_bf16_f32(
    const float* __restrict__ q, const float* __restrict__ gate, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, float* __restrict__ out)
{
    ts_ninfer_attention<false>(q, gate, keyEnd, blockTable, maxBlocks, kOff, vOff, rowsPerSeq, nHead, nKv, scale, 0, 1, nullptr, out);
}

extern "C" __global__ void __launch_bounds__(128) ts_ninfer_attn_bf16_split_f32(
    const float* __restrict__ q, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, int chunk, float* __restrict__ partial)
{
    ts_ninfer_attention<true>(q, nullptr, keyEnd, blockTable, maxBlocks, kOff, vOff, rowsPerSeq, nHead, nKv, scale, chunk,
                           gridDim.x, partial, nullptr);
}

// Merge the split partials of every (row, head) in split order and apply the gate:
// out = (sum_s acc_s e^(m_s - M)) / (sum_s l_s e^(m_s - M)) * sigmoid(gate). One block per
// (row, head), 256 threads (one per value).
extern "C" __global__ void ts_ninfer_attn_combine_f32(
    const float* __restrict__ partial, const float* __restrict__ gate, int splits, float* __restrict__ out)
{
    const long long base = blockIdx.x;
    const float* p = partial + base * splits * (TS_NINFER_HD + 2);
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s)
        mx = fmaxf(mx, p[s * (TS_NINFER_HD + 2) + TS_NINFER_HD]);
    float num = 0.0f, den = 0.0f;
    const int x = threadIdx.x;
    for (int s = 0; s < splits; ++s)
    {
        const float* ps = p + s * (TS_NINFER_HD + 2);
        const float w = mx == -INFINITY ? 0.0f : expf(ps[TS_NINFER_HD] - mx);
        num = fmaf(ps[x], w, num);
        den = fmaf(ps[TS_NINFER_HD + 1], w, den);
    }
    out[base * TS_NINFER_HD + x] = ts_ninfer_bf(ts_ninfer_bf(den > 0.0f ? num / den : 0.0f) * ts_ninfer_sigmoid(gate[base * TS_NINFER_HD + x]));
}

