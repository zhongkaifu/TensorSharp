// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// Recurrent-layer prefill on the device: the chunked Gated DeltaNet of a prompt chunk
// (ts_ssm_gdn_chunk_f32, CudaGraphExecutor's ggml_gated_delta_net for prompt rows).
//
// The committed compute_75 PTX in native/ptx is what every build without nvcc loads, so it is a
// tracked artifact that must be refreshed with the source (DEVELOPMENT.md). The probe
// is how the host tells a stale image from a current one: CudaSsmPrefillKernels
// compares TS_SSM_PREFILL_VERSION, as the loaded image reports it, with the version its
// C# side was written against, and refuses a module that does not match - a
// stale committed PTX then keeps every route on its fallback instead of launching
// kernels that are not there (or that take other arguments).
//
// Bump TS_SSM_PREFILL_VERSION, and CudaSsmPrefillKernels.CurrentVersion with it, whenever
// a kernel is added or its signature changes.
//
// sm_80-only code (cp.async, m16n8k16, bf16 mma) must sit under
// #if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800 and be gated on the host with
// CudaKernels.ArchGuardedKernelsUsable; prefer sm_75-clean code (m16n8k8 + ldmatrix)
// so the committed compute_75 PTX carries working kernels.
// ---------------------------------------------------------------------------

#define TS_SSM_PREFILL_VERSION 2

/// Writes the module version to out[0] and the architecture this image was compiled for
/// (__CUDA_ARCH__ of the code the driver selected: the SASS of a fatbin, or the PTX target
/// a JIT compiled from) to out[1]. One thread; launch <<<1, 1>>>.
extern "C" __global__ void ts_ssm_prefill_caps(int* out)
{
    if (threadIdx.x != 0 || blockIdx.x != 0)
        return;
    out[0] = TS_SSM_PREFILL_VERSION;
#if defined(__CUDA_ARCH__)
    out[1] = __CUDA_ARCH__;
#else
    out[1] = 0;
#endif
}

// ---------------------------------------------------------------------------
// ggml_gated_delta_net over a prompt chunk, in chunks of 16 tokens (the WY form of ggml-cuda's
// gated-delta-net-mma.cu, the direct-CUDA Qwen3.5 engine's ts_q35_gdn_chunk_f32 on the graph op's operands):
//   q, k  [n_s, n_t, hk, 128]  (L2-normed; value head h reads key head h % hk)
//   v     [n_s, n_t, hv, 128]
//   g, beta [n_s, n_t, hv]     (log decay per head; beta)
//   stateIn / stateOut [n_s, hv, 128, 128]: row j of a head's state holds column j of S
//   out   [n_s, n_t, hv, 128]  y_t = S_t^T q_t / sqrt(128), S_t after token t
// Per token: S *= exp(g); delta = (v - S^T k) * beta; S += k delta^T. Within a chunk:
//   o_t = e^G_t S0^T q_t + sum_{s <= t} e^(G_t - G_s) (q_t.k_s) U[s]
//   S   = e^G_last S0 + K^T (U * e^(G_last - G_s))
// Every product runs on BF16 tensor cores with each operand split into a BF16 value and its BF16 residual
// (lo*hi + hi*lo + hi*hi, about 16 mantissa bits); the state stays F32 in registers across chunks. One block of
// 8 warps owns one (sequence, v-head, 32-column value slice): grid (hv * 4, n_s). Warp w keeps state rows
// 16w..16w+15. sm_80+ (mma.m16n8k16.bf16): the compute_75 image traps here, so the host launches it only where
// CudaPrefillModule.ArchGuardedKernelsUsable holds.
// ---------------------------------------------------------------------------
#include <cuda_bf16.h>
#include <stdint.h>

#define TS_SSM_GDN_HD 128      // Gated DeltaNet head width (keys and values)

#define TS_SSM_GDN_C 16                     // tokens per chunk
#define TS_SSM_GDN_V 32                     // value columns per block
#define TS_SSM_GDN_KS (TS_SSM_GDN_HD + 8)   // BF16 row stride of the 128-wide operands
#define TS_SSM_GDN_CS (TS_SSM_GDN_C + 8)    // BF16 row stride of the 16-wide operands

struct TsSsmGdnChunk
{
    // [hi | lo][rows][stride]
    __nv_bfloat16 q[2][TS_SSM_GDN_C][TS_SSM_GDN_KS];   // queries, scaled by 1/sqrt(128)
    __nv_bfloat16 k[2][TS_SSM_GDN_C][TS_SSM_GDN_KS];
    union
    {
        __nv_bfloat16 s[2][TS_SSM_GDN_V][TS_SSM_GDN_KS];   // S0 as [v][k]
        struct
        {
            __nv_bfloat16 u[2][TS_SSM_GDN_V][TS_SSM_GDN_CS];   // U as [v][t]
            __nv_bfloat16 w[2][TS_SSM_GDN_V][TS_SSM_GDN_CS];   // U * e^(G_last - G_t)
        } uw;
    };
    __nv_bfloat16 p[2][TS_SSM_GDN_C][TS_SSM_GDN_CS];     // e^(G_t - G_s) q_t.k_s, s <= t
    __nv_bfloat16 inv[2][TS_SSM_GDN_C][TS_SSM_GDN_CS];   // (I + A)^-1
    __nv_bfloat16 d[2][TS_SSM_GDN_V][TS_SSM_GDN_CS];     // D as [v][t]
    float lower[TS_SSM_GDN_C][TS_SSM_GDN_C];              // A
    float prefix[TS_SSM_GDN_C], beta[TS_SSM_GDN_C];
};

// A BF16 value and its residual, written to plane 0 and plane `PLANE` elements later.
template <int PLANE>
__device__ __forceinline__ void ts_ssm_put_split(__nv_bfloat16* m, int i, float x)
{
    const __nv_bfloat16 h = __float2bfloat16(x);
    m[i] = h;
    m[PLANE + i] = __float2bfloat16(x - __bfloat162float(h));
}

__device__ __forceinline__ uint32_t ts_ssm_split_bf16x2(float x0, float x1, uint32_t& lo)
{
    const __nv_bfloat162 h = __floats2bfloat162_rn(x0, x1);
    const float2 r = __bfloat1622float2(h);
    const __nv_bfloat162 l = __floats2bfloat162_rn(x0 - r.x, x1 - r.y);
    lo = *reinterpret_cast<const uint32_t*>(&l);
    return *reinterpret_cast<const uint32_t*>(&h);
}

__device__ __forceinline__ uint32_t ts_ssm_ld32(const __nv_bfloat16* p) { return *reinterpret_cast<const uint32_t*>(p); }

__device__ __forceinline__ void ts_ssm_mma_bf16(float (&c)[4], const uint32_t (&a)[4], uint32_t b0, uint32_t b1)
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
__device__ __forceinline__ void ts_ssm_mma3(float (&c)[4], const __nv_bfloat16* a, int sa, int aPlane,
                                            const __nv_bfloat16* b, int sb, int bPlane)
{
    const int lane = threadIdx.x & 31, g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int k = 0; k < K; k += 16)
    {
        const __nv_bfloat16* pa = a + g * sa + k + 2 * t;
        const __nv_bfloat16* pb = b + g * sb + k + 2 * t;
        const uint32_t ah[4] = { ts_ssm_ld32(pa), ts_ssm_ld32(pa + 8 * sa), ts_ssm_ld32(pa + 8), ts_ssm_ld32(pa + 8 * sa + 8) };
        const uint32_t al[4] = { ts_ssm_ld32(pa + aPlane), ts_ssm_ld32(pa + aPlane + 8 * sa), ts_ssm_ld32(pa + aPlane + 8),
                                 ts_ssm_ld32(pa + aPlane + 8 * sa + 8) };
        const uint32_t bh0 = ts_ssm_ld32(pb), bh1 = ts_ssm_ld32(pb + 8);
        const uint32_t bl0 = ts_ssm_ld32(pb + bPlane), bl1 = ts_ssm_ld32(pb + bPlane + 8);
        ts_ssm_mma_bf16(c, al, bh0, bh1);
        ts_ssm_mma_bf16(c, ah, bl0, bl1);
        ts_ssm_mma_bf16(c, ah, bh0, bh1);
    }
}

extern "C" __global__ void __launch_bounds__(256) ts_ssm_gdn_chunk_f32(
    const float* __restrict__ qIn, const float* __restrict__ kIn, const float* __restrict__ vIn, int hk, int hv, int rowsPerSeq,
    const float* __restrict__ g, const float* __restrict__ beta,
    const float* stateIn, float* stateOut, float* __restrict__ out)
{
    constexpr int S = TS_SSM_GDN_HD, C = TS_SSM_GDN_C, V = TS_SSM_GDN_V, KS = TS_SSM_GDN_KS, CS = TS_SSM_GDN_CS;
    constexpr int QP = C * KS, SP = V * KS, CP = C * CS, VP = V * CS;   // residual plane offsets
    __shared__ __align__(16) TsSsmGdnChunk sm;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
    const int h = blockIdx.x / (S / V), v0 = (blockIdx.x % (S / V)) * V, seq = blockIdx.y, kh = h % hk;
    const long long rowBase = (long long)seq * rowsPerSeq;
    // M[col][i] = S[i][col]: the state before the chunk, and where the state after it goes (the same memory, or not).
    const float* M = stateIn + ((long long)seq * hv + h) * S * S;
    float* Mout = stateOut + ((long long)seq * hv + h) * S * S;

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
            const long long at = ((rowBase + t0 + ldT) * hk + kh) * S + ldD;
            qn[0] = *(const float4*)(qIn + at);
            qn[1] = *(const float4*)(qIn + at + 4);
            kn[0] = *(const float4*)(kIn + at);
            kn[1] = *(const float4*)(kIn + at + 4);
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
                qh[i] = ts_ssm_split_bf16x2(qv[2 * i] * qScale, qv[2 * i + 1] * qScale, ql[i]);
                kh2[i] = ts_ssm_split_bf16x2(kv[2 * i], kv[2 * i + 1], kl[i]);
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
                ts_ssm_put_split<SP>(&sm.s[0][0][0], (8 * j + 2 * tq + (e & 1)) * KS + d0 + gq + 8 * (e >> 1), st[j][e]);
        __syncthreads();
        if (t0 + C < rowsPerSeq)
            loadQk(t0 + C);

        // Warps 0-3: e^G_t (Q S0) for value columns 8w..8w+7, kept for the output.
        // Warps 4-7: D for columns 8(w-4).., then one tile of A (w = 4, 5) or of P (w = 6, 7).
        float base[4];
        if (warp < 4)
        {
            float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ts_ssm_mma3<S>(c, &sm.q[0][0][0], KS, QP, &sm.s[0][8 * warp][0], KS, SP);
#pragma unroll
            for (int e = 0; e < 4; ++e)
                base[e] = expf(sm.prefix[gq + 8 * (e >> 1)]) * c[e];
        }
        else
        {
            const int j = warp - 4;
            float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ts_ssm_mma3<S>(c, &sm.k[0][0][0], KS, QP, &sm.s[0][8 * j][0], KS, SP);
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int t = gq + 8 * (e >> 1), v = 8 * j + 2 * tq + (e & 1);
                const float vt = t < valid ? vIn[((rowBase + t0 + t) * hv + h) * S + v0 + v] : 0.0f;
                ts_ssm_put_split<VP>(&sm.d[0][0][0], v * CS + t, sm.beta[t] * (vt - expf(sm.prefix[t]) * c[e]));
            }
            const bool gram = j < 2;
            const int s0 = (j & 1) * 8;
            float x[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ts_ssm_mma3<S>(x, gram ? &sm.k[0][0][0] : &sm.q[0][0][0], KS, QP, &sm.k[0][s0][0], KS, QP);
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int t = gq + 8 * (e >> 1), s = s0 + 2 * tq + (e & 1);
                const float decay = s <= t ? expf(sm.prefix[t] - sm.prefix[s]) : 0.0f;
                if (gram)
                    sm.lower[t][s] = s < t ? sm.beta[t] * decay * x[e] : 0.0f;
                else
                    ts_ssm_put_split<CP>(&sm.p[0][0][0], t * CS + s, decay * x[e]);
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
                ts_ssm_put_split<CP>(&sm.inv[0][0][0], row * CS + lane, x);
        }
        __syncthreads();

        // U = (I + A)^-1 D, and U decayed to the chunk's end (S0 is consumed: U and W reuse it).
        const float gLast = sm.prefix[C - 1];
        if (warp >= 4)
        {
            const int j = warp - 4;
            float c[4] = { 0.0f, 0.0f, 0.0f, 0.0f };
            ts_ssm_mma3<C>(c, &sm.inv[0][0][0], CS, CP, &sm.d[0][8 * j][0], CS, VP);
#pragma unroll
            for (int e = 0; e < 4; ++e)
            {
                const int t = gq + 8 * (e >> 1), v = 8 * j + 2 * tq + (e & 1);
                ts_ssm_put_split<VP>(&sm.uw.u[0][0][0], v * CS + t, c[e]);
                ts_ssm_put_split<VP>(&sm.uw.w[0][0][0], v * CS + t, c[e] * expf(gLast - sm.prefix[t]));
            }
        }
        __syncthreads();

        if (warp < 4)
        {
            ts_ssm_mma3<C>(base, &sm.p[0][0][0], CS, CP, &sm.uw.u[0][8 * warp][0], CS, VP);
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
                const uint32_t bh0 = ts_ssm_ld32(pb), bh1 = ts_ssm_ld32(pb + 8);
                const uint32_t bl0 = ts_ssm_ld32(pb + VP), bl1 = ts_ssm_ld32(pb + VP + 8);
                ts_ssm_mma_bf16(st[j], al, bh0, bh1);
                ts_ssm_mma_bf16(st[j], ah, bl0, bl1);
                ts_ssm_mma_bf16(st[j], ah, bh0, bh1);
            }
        }
        __syncthreads();
    }
#pragma unroll
    for (int j = 0; j < V / 8; ++j)
#pragma unroll
        for (int e = 0; e < 4; ++e)
            Mout[(long long)(v0 + 8 * j + 2 * tq + (e & 1)) * S + d0 + gq + 8 * (e >> 1)] = st[j][e];
}
