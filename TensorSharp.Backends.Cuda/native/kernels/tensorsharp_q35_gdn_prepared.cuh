// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Input-only preparation for the existing GDN chunk rule. Every BF16 high/residual
// conversion, MMA term, prefix addition, expf and inverse operation is unchanged.
// The legacy ts_q35_gdn_chunk_f32 remains the exact-comparison oracle.
#pragma once

struct __align__(16) TsQ35GdnQkPacket
{
    __nv_bfloat16 q[2][TS_Q35_GDN_C][TS_Q35_GDN_HD];
    __nv_bfloat16 k[2][TS_Q35_GDN_C][TS_Q35_GDN_HD];
};
struct __align__(16) TsQ35GdnControlPacket
{
    __nv_bfloat16 p[2][TS_Q35_GDN_C][TS_Q35_GDN_C];
    __nv_bfloat16 inv[2][TS_Q35_GDN_C][TS_Q35_GDN_C];
    float prefix[TS_Q35_GDN_C], beta[TS_Q35_GDN_C];
};
static_assert(sizeof(TsQ35GdnQkPacket) == 16384, "managed Q/K packet size");
static_assert(sizeof(TsQ35GdnControlPacket) == 2176, "managed control packet size");

__device__ __forceinline__ void ts_q35_gdn_load_packet(const TsQ35GdnQkPacket& packet, __nv_bfloat16* q, __nv_bfloat16* k)
{
    constexpr int C = TS_Q35_GDN_C, KS = TS_Q35_GDN_KS;
    const int token = threadIdx.x >> 4, dim = (threadIdx.x & 15) * 8;
#pragma unroll
    for (int plane = 0; plane < 2; ++plane)
    {
        *(uint4*)(q + plane * C * KS + token * KS + dim) = *(const uint4*)&packet.q[plane][token][dim];
        *(uint4*)(k + plane * C * KS + token * KS + dim) = *(const uint4*)&packet.k[plane][token][dim];
    }
}

extern "C" __global__ void __launch_bounds__(256) ts_q35_gdn_prepare_qk_f32(
    const float* __restrict__ qkv, int convDim, int hk, int rowsPerSeq,
    TsQ35GdnQkPacket* __restrict__ packets)
{
    constexpr int S = TS_Q35_GDN_HD, C = TS_Q35_GDN_C;
    const int chunk = blockIdx.x, kh = blockIdx.y, seq = blockIdx.z;
    const int ldT = threadIdx.x >> 4, ldD = (threadIdx.x & 15) * 8, t0 = chunk * C;
    const int chunks = (rowsPerSeq - 1) / C + 1;
    auto& packet = packets[((long long)seq * hk + kh) * chunks + chunk];
    float4 qn[2], kn[2];
    qn[0] = qn[1] = kn[0] = kn[1] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    if (t0 + ldT < rowsPerSeq)
    {
        const float* row = qkv + ((long long)seq * rowsPerSeq + t0 + ldT) * convDim;
        qn[0] = *(const float4*)(row + kh * S + ldD);
        qn[1] = *(const float4*)(row + kh * S + ldD + 4);
        kn[0] = *(const float4*)(row + (hk + kh) * S + ldD);
        kn[1] = *(const float4*)(row + (hk + kh) * S + ldD + 4);
    }
    const float qScale = rsqrtf((float)S);
    const float qv[8] = { qn[0].x, qn[0].y, qn[0].z, qn[0].w, qn[1].x, qn[1].y, qn[1].z, qn[1].w };
    const float kv[8] = { kn[0].x, kn[0].y, kn[0].z, kn[0].w, kn[1].x, kn[1].y, kn[1].z, kn[1].w };
    uint32_t qh[4], ql[4], kh2[4], kl[4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
    {
        qh[i] = ts_q35_split_bf16x2(qv[2 * i] * qScale, qv[2 * i + 1] * qScale, ql[i]);
        kh2[i] = ts_q35_split_bf16x2(kv[2 * i], kv[2 * i + 1], kl[i]);
    }
    *(uint4*)&packet.q[0][ldT][ldD] = make_uint4(qh[0], qh[1], qh[2], qh[3]);
    *(uint4*)&packet.q[1][ldT][ldD] = make_uint4(ql[0], ql[1], ql[2], ql[3]);
    *(uint4*)&packet.k[0][ldT][ldD] = make_uint4(kh2[0], kh2[1], kh2[2], kh2[3]);
    *(uint4*)&packet.k[1][ldT][ldD] = make_uint4(kl[0], kl[1], kl[2], kl[3]);
}

struct TsQ35GdnControlShared
{
    __nv_bfloat16 q[2][TS_Q35_GDN_C][TS_Q35_GDN_KS];
    __nv_bfloat16 k[2][TS_Q35_GDN_C][TS_Q35_GDN_KS];
    __nv_bfloat16 p[2][TS_Q35_GDN_C][TS_Q35_GDN_CS];
    __nv_bfloat16 inv[2][TS_Q35_GDN_C][TS_Q35_GDN_CS];
    float lower[TS_Q35_GDN_C][TS_Q35_GDN_C];
    float prefix[TS_Q35_GDN_C], beta[TS_Q35_GDN_C];
};
extern "C" __global__ void __launch_bounds__(256) ts_q35_gdn_prepare_control_f32(
    const TsQ35GdnQkPacket* __restrict__ qkPackets, const float* __restrict__ g, const float* __restrict__ beta,
    int hk, int hv, int rowsPerSeq, TsQ35GdnControlPacket* __restrict__ packets)
{
    constexpr int S = TS_Q35_GDN_HD, C = TS_Q35_GDN_C, KS = TS_Q35_GDN_KS, CS = TS_Q35_GDN_CS;
    constexpr int QP = C * KS, CP = C * CS;
    __shared__ __align__(16) TsQ35GdnControlShared sm;
    const int chunk = blockIdx.x, h = blockIdx.y, seq = blockIdx.z, kh = h % hk;
    const int chunks = (rowsPerSeq - 1) / C + 1, t0 = chunk * C, valid = min(C, rowsPerSeq - t0);
    const long long rowBase = (long long)seq * rowsPerSeq;
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, gq = lane >> 2, tq = lane & 3;
    ts_q35_gdn_load_packet(qkPackets[((long long)seq * hk + kh) * chunks + chunk], &sm.q[0][0][0], &sm.k[0][0][0]);
    if (warp == 0)
    {
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
    __syncthreads();
    if (warp >= 4)
    {
        const int j = warp - 4;
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
    auto& packet = packets[((long long)seq * hv + h) * chunks + chunk];
    const int row = tid / C, col = tid % C;
#pragma unroll
    for (int plane = 0; plane < 2; ++plane)
    {
        packet.p[plane][row][col] = sm.p[plane][row][col];
        packet.inv[plane][row][col] = sm.inv[plane][row][col];
    }
    if (tid < C)
    {
        packet.prefix[tid] = sm.prefix[tid];
        packet.beta[tid] = sm.beta[tid];
    }
}

// The state/V-dependent remainder of the legacy chunk kernel. Arithmetic and
// st accumulator order are unchanged; only input-only control data are loaded.
extern "C" __global__ void __launch_bounds__(256) ts_q35_gdn_chunk_prepared_f32(
    const float* __restrict__ qkv, int convDim, int hk, int hv, int rowsPerSeq,
    const float* __restrict__ g, const float* __restrict__ beta,
    float* const* __restrict__ states, float* __restrict__ out,
    const TsQ35GdnQkPacket* __restrict__ qkPackets, const TsQ35GdnControlPacket* __restrict__ controlPackets)
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

    const int chunks = (rowsPerSeq - 1) / C + 1;
    for (int t0 = 0; t0 < rowsPerSeq; t0 += C)
    {
        const int valid = min(C, rowsPerSeq - t0);
        const int chunk = t0 / C;
        ts_q35_gdn_load_packet(qkPackets[((long long)seq * hk + kh) * chunks + chunk], &sm.q[0][0][0], &sm.k[0][0][0]);
        const auto& control = controlPackets[((long long)seq * hv + h) * chunks + chunk];
        const int controlRow = tid / C, controlCol = tid % C;
#pragma unroll
        for (int plane = 0; plane < 2; ++plane)
        {
            sm.p[plane][controlRow][controlCol] = control.p[plane][controlRow][controlCol];
            sm.inv[plane][controlRow][controlCol] = control.inv[plane][controlRow][controlCol];
        }
        if (tid < C)
        {
            sm.prefix[tid] = control.prefix[tid];
            sm.beta[tid] = control.beta[tid];
        }
#pragma unroll
        for (int j = 0; j < V / 8; ++j)
#pragma unroll
            for (int e = 0; e < 4; ++e)
                ts_q35_put_split<SP>(&sm.s[0][0][0], (8 * j + 2 * tq + (e & 1)) * KS + d0 + gq + 8 * (e >> 1), st[j][e]);
        __syncthreads();

        // Warps 0-3: e^G_t (Q S0) for value columns 8w..8w+7, kept for the output.
        // Warps 4-7: D for columns 8(w-4)..; A/P and the inverse were prepared once.
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
