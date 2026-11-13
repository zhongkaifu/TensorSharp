// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Tensor-core flash prefill attention: GQA groups folded into the query tile, F16 K/V tiles in
// shared memory, online softmax, causal masking with a floor, split-KV for long contexts
// (ts_flash_prefill_f16_d<hd>, CudaGraphExecutor's attention for prompt rows).
//
// The committed compute_75 PTX in native/ptx is what every build without nvcc loads, so it is a
// tracked artifact that must be refreshed with the source (DEVELOPMENT.md). The probe
// is how the host tells a stale image from a current one: CudaFlashPrefillKernels
// compares TS_FLASH_PREFILL_VERSION, as the loaded image reports it, with the version its
// C# side was written against, and refuses a module that does not match - a
// stale committed PTX then keeps every route on its fallback instead of launching
// kernels that are not there (or that take other arguments).
//
// Bump TS_FLASH_PREFILL_VERSION, and CudaFlashPrefillKernels.CurrentVersion with it, whenever
// a kernel is added or its signature changes.
//
// sm_80-only code (cp.async, m16n8k16, bf16 mma) must sit under
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 and be gated on the host with
// CudaKernels.ArchGuardedKernelsUsable; prefer sm_75-clean code (m16n8k8 + ldmatrix)
// so the committed compute_75 PTX carries working kernels.
// ---------------------------------------------------------------------------

#define TS_FLASH_PREFILL_VERSION 3

/// Writes the module version to out[0] and the architecture this image was compiled for
/// (__CUDA_ARCH__ of the code the driver selected: the SASS of a fatbin, or the PTX target
/// a JIT compiled from) to out[1]. One thread; launch <<<1, 1>>>.
extern "C" __global__ void ts_flash_prefill_caps(int* out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    out[0] = TS_FLASH_PREFILL_VERSION;
#if defined(__CUDA_ARCH__)
    out[1] = __CUDA_ARCH__;
#else
    out[1] = 0;
#endif
}

// ---------------------------------------------------------------------------
// Prompt attention over a head-first F16 cache (CudaGraphExecutor's attention for prompt rows): the direct-CUDA
// Qwen3.5 engine's prefill attention (ts_q35_attn_pipe) on the graph op's operands.
//   q     [heads][nq][HD] F32 (the query heads of a KV head are its G = heads / nKv consecutive heads)
//   k, v  [nKv][cap][HD] F16 (the cache; keys 0..len-1 are the window)
//   out   [nq][heads][HD] F32
// Query t sees keys [floor, len - nq + t] (causal, the rows the last nq of the window). A block of 8 warps takes 128
// rows of one KV head (row r: query t = r / G of head kvh * G + r % G) and streams 32-key K and V tiles through a
// two-stage cp.async pipeline. The mma accumulate in F16 over short runs only: each 16-value step of QK^T (Q
// pre-scaled to log2 units) is rounded once and summed in F32, and PV over one 32-key tile (probabilities at most 1)
// is rounded twice before it joins the F32 accumulators. With dyn set (a recorded decode step) the window's length is
// dyn[0], read on the device. splits > 1 (a grid too small to fill the device: a few
// query rows over a long window): block x = (row tile, key split), each split takes a 32-key-aligned share of its
// rows' keys and writes unnormalized partials (acc, ln-domain max, sum) that ts_flash_prefill_combine_f32 merges.
// sm_80+ (cp.async, ldmatrix, mma.f16): the compute_75 image traps here; the host launches it only where
// CudaPrefillModule.ArchGuardedKernelsUsable holds.
// ---------------------------------------------------------------------------
#include <cuda_fp16.h>
#include <stdint.h>

#define TS_FP_KT 32      // keys per tile
#define TS_FP_ROWS 128   // query rows per block (8 warps x 16)

__device__ __forceinline__ uint32_t ts_fp_smem_u32(const void* p)
{
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void ts_fp_cp16(void* dst, const void* src, bool valid)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n"
                 :: "r"(ts_fp_smem_u32(dst)), "l"(src), "r"(valid ? 16 : 0));
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp_cp_commit()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.commit_group;\n" ::);
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp_cp_wait0()
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("cp.async.wait_group 0;\n" ::);
#else
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp_ldsm_x4(uint32_t (&r)[4], const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ts_fp_smem_u32(p)));
#else
    r[0] = r[1] = r[2] = r[3] = 0;
    __trap();
#endif
}

__device__ __forceinline__ void ts_fp_ldsm_x4_t(uint32_t (&r)[4], const void* p)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(ts_fp_smem_u32(p)));
#else
    r[0] = r[1] = r[2] = r[3] = 0;
    __trap();
#endif
}

// m16n8k16 with F16 accumulation.
__device__ __forceinline__ void ts_fp_mma_f16h(uint32_t (&c)[2], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%3,%4,%5}, {%6,%7}, {%0,%1};\n"
                 : "+r"(c[0]), "+r"(c[1]) : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
#else
    __trap();
#endif
}

__device__ __forceinline__ uint32_t ts_fp_pack_h2(float lo, float hi)
{
    const half2 v = __floats2half2_rn(lo, hi);
    return *reinterpret_cast<const uint32_t*>(&v);
}

// Dynamic shared memory of ts_flash_prefill_f16_d<HD>: 2 stages x (K, V) x TS_FP_KT rows of HD + 8 halves
// (CudaFlashPrefillKernels.SharedBytes computes the same).
template <int HD>
struct TsFpAttnSmem
{
    static constexpr int ROW = HD + 8;   // F16 tile row stride (bank-spread)
    half k[2][TS_FP_KT * ROW];
    half v[2][TS_FP_KT * ROW];
};

template <int HD>
__device__ __forceinline__ void ts_fp_attn(
    const float* __restrict__ q, const half* __restrict__ kc, const half* __restrict__ vc, int cap, int nq, int len,
    int floor, int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out,
    const int* __restrict__ dyn)
{
    // A recorded decode step reads the window's length from the step's parameter block (CudaDecodeDynParams).
    if (dyn != nullptr)
        len = dyn[0];
    constexpr int ROW = TsFpAttnSmem<HD>::ROW, KS = HD / 16, NS = HD / 8;
    extern __shared__ __align__(16) uint8_t ts_fp_attn_smem[];
    TsFpAttnSmem<HD>& sm = *reinterpret_cast<TsFpAttnSmem<HD>*>(ts_fp_attn_smem);
    const int kvh = blockIdx.y;
    const int G = nHead / nKv;
    const int R = nq * G;
    const int split = (int)(blockIdx.x % splits);
    const int rowBase = (blockIdx.x / splits) * TS_FP_ROWS;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
    const int past = len - nq;   // keys before the first query row
    const int kAll = past + (min(rowBase + TS_FP_ROWS, R) - 1) / G + 1;
    const int share = ((kAll + splits - 1) / splits + TS_FP_KT - 1) / TS_FP_KT * TS_FP_KT;
    const int kLo = max(min(kAll, split * share), min(kAll, floor / TS_FP_KT * TS_FP_KT)), kHi = min(kAll, split * share + share);

    const int r0 = rowBase + warp * 16 + g, r1 = r0 + 8;
    const bool live0 = r0 < R, live1 = r1 < R;
    const int end0 = live0 ? past + r0 / G + 1 : 0, end1 = live1 ? past + r1 / G + 1 : 0;
    const bool warpLive = rowBase + warp * 16 < R;
    const half* kHead = kc + (long long)kvh * cap * HD;
    const half* vHead = vc + (long long)kvh * cap * HD;

    auto load = [&](int st, int first) {
        constexpr int CHUNKS = HD / 8;   // 16-byte chunks of a key row
        for (int i = tid; i < 2 * TS_FP_KT * CHUNKS; i += blockDim.x)
        {
            const int isV = i >= TS_FP_KT * CHUNKS, c = isV ? i - TS_FP_KT * CHUNKS : i;
            const int key = c / CHUNKS, part = c % CHUNKS;
            const bool ok = first + key < kHi;
            const half* src = (isV ? vHead : kHead) + (long long)(ok ? first + key : 0) * HD + part * 8;
            ts_fp_cp16((isV ? sm.v[st] : sm.k[st]) + key * ROW + part * 8, src, ok);
        }
    };
    if (kHi > kLo)
        load(0, kLo);
    ts_fp_cp_commit();

    const float log2Scale = scale * 1.4426950408889634f;
    uint32_t qa[KS][4];
    {
        const float* q0 = live0 ? q + ((long long)(kvh * G + r0 % G) * nq + r0 / G) * HD : nullptr;
        const float* q1 = live1 ? q + ((long long)(kvh * G + r1 % G) * nq + r1 / G) * HD : nullptr;
#pragma unroll
        for (int kk = 0; kk < KS; ++kk)
        {
            const int c = kk * 16 + qd * 2;
            qa[kk][0] = q0 ? ts_fp_pack_h2(q0[c] * log2Scale, q0[c + 1] * log2Scale) : 0u;
            qa[kk][1] = q1 ? ts_fp_pack_h2(q1[c] * log2Scale, q1[c + 1] * log2Scale) : 0u;
            qa[kk][2] = q0 ? ts_fp_pack_h2(q0[c + 8] * log2Scale, q0[c + 9] * log2Scale) : 0u;
            qa[kk][3] = q1 ? ts_fp_pack_h2(q1[c + 8] * log2Scale, q1[c + 9] * log2Scale) : 0u;
        }
    }

    float acc[NS][4];
#pragma unroll
    for (int n = 0; n < NS; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;

    for (int first = kLo, t = 0; first < kHi; first += TS_FP_KT, ++t)
    {
        const int st = t & 1;
        ts_fp_cp_wait0();
        __syncthreads();   // this tile landed; every warp is done with the previous one
        if (first + TS_FP_KT < kHi)
            load(st ^ 1, first + TS_FP_KT);
        ts_fp_cp_commit();
        if (!warpLive)
            continue;

        float s[4][4];
        {
            // B fragments of 2 n-tiles x 1 k-step per ldmatrix.x4 (rows = keys).
            const half* kb = sm.k[st] + ((lane & 7) + ((lane >> 4) << 3)) * ROW + ((lane >> 3) & 1) * 8;
#pragma unroll
            for (int np = 0; np < 2; ++np)
            {
#pragma unroll
                for (int e = 0; e < 4; ++e)
                    s[2 * np][e] = s[2 * np + 1][e] = 0.0f;
#pragma unroll
                for (int kk = 0; kk < KS; ++kk)
                {
                    uint32_t h0[2] = { 0u, 0u }, h1[2] = { 0u, 0u };
                    uint32_t b[4];
                    ts_fp_ldsm_x4(b, kb + np * 16 * ROW + kk * 16);
                    ts_fp_mma_f16h(h0, qa[kk], b[0], b[1]);
                    ts_fp_mma_f16h(h1, qa[kk], b[2], b[3]);
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
            s[j][0] = key < end0 && key >= floor ? s[j][0] : -INFINITY;
            s[j][1] = key + 1 < end0 && key + 1 >= floor ? s[j][1] : -INFINITY;
            s[j][2] = key < end1 && key >= floor ? s[j][2] : -INFINITY;
            s[j][3] = key + 1 < end1 && key + 1 >= floor ? s[j][3] : -INFINITY;
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
            for (int n = 0; n < NS; ++n)
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
            pa[kk][0] = ts_fp_pack_h2(s[2 * kk][0], s[2 * kk][1]);
            pa[kk][1] = ts_fp_pack_h2(s[2 * kk][2], s[2 * kk][3]);
            pa[kk][2] = ts_fp_pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
            pa[kk][3] = ts_fp_pack_h2(s[2 * kk + 1][2], s[2 * kk + 1][3]);
        }
        const half* vrow = sm.v[st] + ((lane & 7) + (((lane >> 3) & 1) << 3)) * ROW + ((lane >> 4) << 3);
#pragma unroll
        for (int dn = 0; dn < HD / 16; ++dn)
        {
            uint32_t h0[2] = { 0u, 0u }, h1[2] = { 0u, 0u };
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
            {
                uint32_t b[4];
                ts_fp_ldsm_x4_t(b, vrow + kk * 16 * ROW + dn * 16);
                ts_fp_mma_f16h(h0, pa[kk], b[0], b[1]);
                ts_fp_mma_f16h(h1, pa[kk], b[2], b[3]);
            }
            const float2 a0 = __half22float2(*reinterpret_cast<const half2*>(&h0[0]));
            const float2 a1 = __half22float2(*reinterpret_cast<const half2*>(&h0[1]));
            const float2 b0 = __half22float2(*reinterpret_cast<const half2*>(&h1[0]));
            const float2 b1 = __half22float2(*reinterpret_cast<const half2*>(&h1[1]));
            acc[2 * dn][0] += a0.x; acc[2 * dn][1] += a0.y; acc[2 * dn][2] += a1.x; acc[2 * dn][3] += a1.y;
            acc[2 * dn + 1][0] += b0.x; acc[2 * dn + 1][1] += b0.y; acc[2 * dn + 1][2] += b1.x; acc[2 * dn + 1][3] += b1.y;
        }
    }
    ts_fp_cp_wait0();
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
        const long long base = (long long)t * nHead + h;
        const float l = half_ ? l1 : l0;
        if (splits > 1)
        {
            float* p = partial + (base * splits + split) * (HD + 2);
#pragma unroll
            for (int n = 0; n < NS; ++n)
            {
                const int x = n * 8 + qd * 2;
                p[x] = acc[n][half_ * 2];
                p[x + 1] = acc[n][half_ * 2 + 1];
            }
            if (qd == 0)
            {
                const float m = half_ ? m1 : m0;
                p[HD] = m == -INFINITY ? -INFINITY : m * 0.6931471805599453f;   // log2 -> ln domain
                p[HD + 1] = l;
            }
            continue;
        }
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* o = out + base * HD;
#pragma unroll
        for (int n = 0; n < NS; ++n)
        {
            const int x = n * 8 + qd * 2;
            o[x] = acc[n][half_ * 2] * inv;
            o[x + 1] = acc[n][half_ * 2 + 1] * inv;
        }
    }
}

#define TS_FP_ATTN(HD) \
    extern "C" __global__ void __launch_bounds__(256) ts_flash_prefill_f16_d##HD( \
        const float* __restrict__ q, const half* __restrict__ kc, const half* __restrict__ vc, int cap, int nq, int len, \
        int floor, int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out, \
        const int* __restrict__ dyn) \
    { ts_fp_attn<HD>(q, kc, vc, cap, nq, len, floor, nHead, nKv, scale, splits, partial, out, dyn); }

TS_FP_ATTN(128)
TS_FP_ATTN(256)

// The split partials of one (query, head): out = sum_s e^(m_s - m) acc_s / sum_s e^(m_s - m) l_s.
// Launch: grid nq * heads, block HD.
extern "C" __global__ void ts_flash_prefill_combine_f32(const float* __restrict__ partial, int splits, int hd, float* __restrict__ out)
{
    const long long base = blockIdx.x;
    const float* p = partial + base * splits * (hd + 2);
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s)
        mx = fmaxf(mx, p[s * (hd + 2) + hd]);
    float num = 0.0f, den = 0.0f;
    const int x = threadIdx.x;
    for (int s = 0; s < splits; ++s)
    {
        const float* ps = p + s * (hd + 2);
        const float w = mx == -INFINITY ? 0.0f : expf(ps[hd] - mx);
        num = fmaf(ps[x], w, num);
        den = fmaf(ps[hd + 1], w, den);
    }
    out[base * hd + x] = den > 0.0f ? num / den : 0.0f;
}
