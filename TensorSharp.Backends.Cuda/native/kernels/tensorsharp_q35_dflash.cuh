// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ---------------------------------------------------------------------------
// DFlash2 block drafter kernels for the Qwen3.5 / Qwen3.8 direct-CUDA engine (included by
// tensorsharp_q35_kernels.cu). The drafter's math is the graph ggml_ops_dflash.cpp builds:
//   encode:  g = rmsnorm(fc @ [target residuals entering the tapped layers]) * enc_norm;
//   inject:  per draft layer k = rope(rmsnorm_head(attn_k g) * k_norm), v = attn_v g, written to the
//            sequence's KV ring at the row's position;
//   draft:   [anchor, MASK x (b - 1)] through the draft layers (grouped dynamic convolution around
//            each attention and FFN sublayer, attention over the live ring rows and the block's own
//            keys, non-causal within the block), the target's LM head over the proposal rows, a
//            top-k per row and the DFlash2 candidate selector's greedy walk.
// The drafter's weights are Q4_K (a few Q6_K); its matmuls run on the int8 tensor cores against
// q8_1 activations, as ggml-cuda's mmq does.
// ---------------------------------------------------------------------------

#define TS_Q35_DF_HD 128                     // draft head width
#define TS_Q35_DF_ROW (TS_Q35_DF_HD + 8)     // F16 tile row stride (68 words)

// ---------------------------------------------------------------------------
// Q4_K x q8_1 on the int8 tensor cores. Q4_K superblock (256 values, 144 bytes): half d, half
// dmin, scales[12] (eight 6-bit scale / min pairs), qs[128]; value 64c + l (c < 4, l < 64) is
// nibble (l >= 32) of qs[32c + l % 32], in sub-block 2c + (l >= 32), and decodes to
// d * sc * q - dmin * m. One mma.m16n8k32.s8 per 32-value sub-block (exactly one q8_1 block):
//   sum_i (d sc q_i - dmin m) xd xq_i = d sc xd (sum q xq) - dmin m xd (sum xq).
// Block: 4 warps x 16 weight rows against NT * 8 activation rows, as ts_q35_gemm_q6k; grid
// (row tiles, K splits). accumulate: y += instead of y =; with more than one split every block
// adds its share atomically (the caller zeroes y first unless it accumulates).
// ---------------------------------------------------------------------------
__device__ __forceinline__ void ts_q35_q4k_scale_min(int j, const uint8_t* s, int& sc, int& m)
{
    if (j < 4)
    {
        sc = s[j] & 63;
        m = s[j + 4] & 63;
    }
    else
    {
        sc = (s[j + 4] & 0xF) | ((s[j - 4] >> 6) << 4);
        m = (s[j + 4] >> 4) | ((s[j] >> 6) << 4);
    }
}

__device__ __forceinline__ void ts_q35_mma_s8_k32(int (&c)[4], const int (&a)[4], int b0, int b1)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%10,%11,%12,%13};\n"
                 : "=r"(c[0]), "=r"(c[1]), "=r"(c[2]), "=r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(0), "r"(0), "r"(0), "r"(0));
#else
    __trap();
#endif
}

template <int NT>
__device__ __forceinline__ void ts_q35_gemm_q4k(
    const uint8_t* __restrict__ w, long long rowBytes, int outDim, int k,
    const int8_t* __restrict__ xq, const float* __restrict__ xd, int n, float* __restrict__ y, int accumulate)
{
    constexpr int NA = NT * 8;
    __shared__ __align__(16) int8_t xs[NA][TS_Q35_Q6K_ROW];
    __shared__ float xds[NA][8], xms[NA][8];
    __shared__ __align__(16) int8_t ws[4][16][TS_Q35_Q6K_ROW];
    __shared__ float wsd[4][16][8], wsm[4][16][8];
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
        // This warp's 16 weight rows: lane takes (row, chunk, half) items of 16 code bytes.
        for (int item = lane; item < 16 * 4 * 2; item += 32)
        {
            const int r = item >> 3, c = (item >> 1) & 3, hl = item & 1;
            const int gr = row0 + r;
            uint4 v = make_uint4(0, 0, 0, 0);
            if (gr < outDim)
            {
                const uint8_t* blk = w + (long long)gr * rowBytes + (long long)sb * 144;
                v = *(const uint4*)(blk + 16 + 32 * c + 16 * hl);
                if (c == 0 && hl == 0)
                {
                    const float d = __half2float(*(const half*)blk), dmin = __half2float(*(const half*)(blk + 2));
#pragma unroll
                    for (int j = 0; j < 8; ++j)
                    {
                        int sc, m;
                        ts_q35_q4k_scale_min(j, blk + 4, sc, m);
                        wsd[warp][r][j] = d * (float)sc;
                        wsm[warp][r][j] = dmin * (float)m;
                    }
                }
            }
            else if (c == 0 && hl == 0)
            {
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    wsd[warp][r][j] = wsm[warp][r][j] = 0.0f;
            }
            int8_t* lo = &ws[warp][r][64 * c + 16 * hl];
            *(uint4*)lo = make_uint4(v.x & 0x0F0F0F0Fu, v.y & 0x0F0F0F0Fu, v.z & 0x0F0F0F0Fu, v.w & 0x0F0F0F0Fu);
            *(uint4*)(lo + 32) = make_uint4((v.x >> 4) & 0x0F0F0F0Fu, (v.y >> 4) & 0x0F0F0F0Fu,
                                            (v.z >> 4) & 0x0F0F0F0Fu, (v.w >> 4) & 0x0F0F0F0Fu);
        }
        __syncthreads();
        // Activation block sums, scaled: xd * sum(xq) per 32-value block.
        for (int i = tid; i < NA * 8; i += blockDim.x)
        {
            const int r = i >> 3, b = i & 7;
            const int* p = (const int*)&xs[r][32 * b];
            int sum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j)
                sum = ts_q35_dp4a(p[j], 0x01010101, sum);
            xms[r][b] = xds[r][b] * (float)sum;
        }
        __syncthreads();
#pragma unroll 2
        for (int j = 0; j < 8; ++j)   // 32-value sub-blocks
        {
            const int a[4] = { *(const int*)&ws[warp][g][32 * j + 4 * t], *(const int*)&ws[warp][g + 8][32 * j + 4 * t],
                               *(const int*)&ws[warp][g][32 * j + 16 + 4 * t], *(const int*)&ws[warp][g + 8][32 * j + 16 + 4 * t] };
            const float d0 = wsd[warp][g][j], d1 = wsd[warp][g + 8][j];
            const float m0 = wsm[warp][g][j], m1 = wsm[warp][g + 8][j];
#pragma unroll
            for (int nt = 0; nt < NT; ++nt)
            {
                int c[4];
                const int* xr = (const int*)&xs[8 * nt + g][32 * j];
                ts_q35_mma_s8_k32(c, a, xr[t], xr[4 + t]);
                const int c0 = 8 * nt + 2 * t;
                const float x0 = xds[c0][j], x1 = xds[c0 + 1][j], s0 = xms[c0][j], s1 = xms[c0 + 1][j];
                acc[nt][0] += (float)c[0] * d0 * x0 - m0 * s0;
                acc[nt][1] += (float)c[1] * d0 * x1 - m0 * s1;
                acc[nt][2] += (float)c[2] * d1 * x0 - m1 * s0;
                acc[nt][3] += (float)c[3] * d1 * x1 - m1 * s1;
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
                float* dst = y + (long long)col * outDim + r;
                if (gridDim.y > 1)
                    atomicAdd(dst, acc[nt][e]);
                else
                    *dst = accumulate ? *dst + acc[nt][e] : acc[nt][e];
            }
        }
}

#define TS_Q35_GEMM_Q4K(NT) \
    extern "C" __global__ void __launch_bounds__(128) ts_q35_gemm_q4k_##NT( \
        const uint8_t* __restrict__ w, long long rowBytes, int outDim, int k, \
        const int8_t* __restrict__ xq, const float* __restrict__ xd, int n, float* __restrict__ y, int accumulate) \
    { ts_q35_gemm_q4k<NT>(w, rowBytes, outDim, k, xq, xd, n, y, accumulate); }
TS_Q35_GEMM_Q4K(1)
TS_Q35_GEMM_Q4K(2)
TS_Q35_GEMM_Q4K(4)
TS_Q35_GEMM_Q4K(8)

// ---------------------------------------------------------------------------
// Per-head RMS norm and NeoX RoPE (pairs (i, i + 64), theta_i = pos * base^(-i / 64)) of the
// draft projections, one warp per (row, head): q heads (when qOut is set) land in qOut
// [rows][nHead][128] F32; K and V heads, as F16, at dst[lane][kvh * headStride + slotRow * 128]
// where lane = row / rowsPerLane and slotRow = ringRows ? pos % ringRows : row % rowsPerLane.
// With ringRows set (an inject into the KV rings) the row's ring slot also records its position.
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_q35_df_prep_f32(
    const float* __restrict__ q, const float* __restrict__ k, const float* __restrict__ v, int rows, int rowsPerLane,
    int nHead, int nKv, const float* __restrict__ qNorm, const float* __restrict__ kNorm, float eps,
    const int* __restrict__ pos, float ropeBase, float* __restrict__ qOut,
    half* const* __restrict__ kDst, half* const* __restrict__ vDst, long long headStride, int ringRows,
    int* const* __restrict__ slotPos, const int* __restrict__ laneOf)
{
    const long long warp = ((long long)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int qUnits = qOut != nullptr ? nHead : 0;
    const int units = qUnits + 2 * nKv;
    if (warp >= (long long)rows * units)
        return;
    const int row = (int)(warp / units), u = (int)(warp % units);
    const int p = pos[row];
    const int laneIdx = laneOf != nullptr ? laneOf[row] : row / rowsPerLane;
    const int slotRow = ringRows > 0 ? p % ringRows : row % rowsPerLane;
    const bool isV = u >= qUnits + nKv;
    const float* src = u < qUnits ? q + ((long long)row * nHead + u) * TS_Q35_DF_HD
                     : !isV ? k + ((long long)row * nKv + (u - qUnits)) * TS_Q35_DF_HD
                     : v + ((long long)row * nKv + (u - qUnits - nKv)) * TS_Q35_DF_HD;
    float x[4];
#pragma unroll
    for (int j = 0; j < 4; ++j)
        x[j] = src[lane + 32 * j];
    if (!isV)
    {
        float ss = 0.0f;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            ss += x[j] * x[j];
        ss = ts_q35_warp_sum(ss);
        const float inv = rsqrtf(ss / (float)TS_Q35_DF_HD + eps);
        const float* wn = u < qUnits ? qNorm : kNorm;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            x[j] *= inv * wn[lane + 32 * j];
        // Pairs (lane, lane + 64) and (lane + 32, lane + 96): dims j and j + 2 of this lane.
#pragma unroll
        for (int j = 0; j < 2; ++j)
        {
            const int i = lane + 32 * j;
            const float theta = (float)p * powf(ropeBase, -(float)i / (TS_Q35_DF_HD / 2));
            float sn, cs;
            sincosf(theta, &sn, &cs);
            const float a = x[j], b = x[j + 2];
            x[j] = a * cs - b * sn;
            x[j + 2] = a * sn + b * cs;
        }
    }
    if (u < qUnits)
    {
        float* o = qOut + ((long long)row * nHead + u) * TS_Q35_DF_HD;
#pragma unroll
        for (int j = 0; j < 4; ++j)
            o[lane + 32 * j] = x[j];
        return;
    }
    const int kvh = isV ? u - qUnits - nKv : u - qUnits;
    half* o = (isV ? vDst : kDst)[laneIdx] + kvh * headStride + (long long)slotRow * TS_Q35_DF_HD;
#pragma unroll
    for (int j = 0; j < 4; ++j)
        o[lane + 32 * j] = __float2half(x[j]);
    if (ringRows > 0 && u == qUnits && lane == 0)
        slotPos[laneIdx][slotRow] = p;
}

// ---------------------------------------------------------------------------
// Draft attention, one block per (key range, KV head, lane): the lane's b * G query rows (row r:
// block position r / G, head kvh * G + r % G) against keys [0, ringRows) from the lane's ring and
// [ringRows, ringRows + b) from the block itself. A ring key holding position p is visible to the
// query at position qp when p >= 0, p < anchor and qp - p < swa; block keys always are (DFlash
// drafts non-causally). F16 operands, F32 scores and online softmax (mma.m16n8k16); every block
// writes unnormalized partials (acc[128], ln-domain max, sum) that ts_q35_df_combine_f32 merges.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(128) ts_q35_df_attn_f16(
    const float* __restrict__ q, half* const* __restrict__ ringK, half* const* __restrict__ ringV,
    int* const* __restrict__ slotPos, int ringRows, const half* __restrict__ blockK, const half* __restrict__ blockV,
    int b, int nHead, int nKv, const int* __restrict__ anchor, int swa, float scale, int chunk, int splits,
    float* __restrict__ partial)
{
    __shared__ __align__(16) half ks[32 * TS_Q35_DF_ROW];
    __shared__ __align__(16) half vs[32 * TS_Q35_DF_ROW];
    __shared__ int kpos[32];
    const int split = blockIdx.x, kvh = blockIdx.y, ln = blockIdx.z;
    const int G = nHead / nKv, R = b * G;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, g = lane >> 2, qd = lane & 3;
    const int total = ringRows + b;
    const int kLo = split * chunk, kHi = min(total, kLo + chunk);
    const half* rk = ringK[ln] + (long long)kvh * ringRows * TS_Q35_DF_HD;
    const half* rv = ringV[ln] + (long long)kvh * ringRows * TS_Q35_DF_HD;
    const half* bk = blockK + ((long long)ln * nKv + kvh) * b * TS_Q35_DF_HD;
    const half* bv = blockV + ((long long)ln * nKv + kvh) * b * TS_Q35_DF_HD;
    const int* sp = slotPos[ln];
    const int anc = anchor[ln];

    const int r0 = warp * 16 + g, r1 = r0 + 8;
    const bool live0 = r0 < R, live1 = r1 < R;
    const int qp0 = anc + (live0 ? r0 / G : 0), qp1 = anc + (live1 ? r1 / G : 0);
    auto qptr = [&](int r) { return q + (((long long)ln * b + r / G) * nHead + kvh * G + r % G) * TS_Q35_DF_HD; };
    const bool warpLive = warp * 16 < R;

    uint32_t qa[8][4];
    {
        const float* q0 = live0 ? qptr(r0) : nullptr;
        const float* q1 = live1 ? qptr(r1) : nullptr;
#pragma unroll
        for (int kk = 0; kk < 8; ++kk)
        {
            const int c = kk * 16 + qd * 2;
            qa[kk][0] = q0 ? ts_q35_pack_h2(q0[c], q0[c + 1]) : 0u;
            qa[kk][1] = q1 ? ts_q35_pack_h2(q1[c], q1[c + 1]) : 0u;
            qa[kk][2] = q0 ? ts_q35_pack_h2(q0[c + 8], q0[c + 9]) : 0u;
            qa[kk][3] = q1 ? ts_q35_pack_h2(q1[c + 8], q1[c + 9]) : 0u;
        }
    }
    float acc[16][4];
#pragma unroll
    for (int n = 0; n < 16; ++n)
        acc[n][0] = acc[n][1] = acc[n][2] = acc[n][3] = 0.0f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.0f, l1 = 0.0f;
    const float log2Scale = scale * 1.4426950408889634f;

    for (int first = kLo; first < kHi; first += 32)
    {
        __syncthreads();
        for (int i = threadIdx.x; i < 32 * 16; i += blockDim.x)
        {
            const int key = i >> 4, c = (i & 15) * 8, idx = first + key;
            uint4 kv = make_uint4(0, 0, 0, 0), vv = make_uint4(0, 0, 0, 0);
            if (idx < kHi)
            {
                const half* ksrc = idx < ringRows ? rk + (long long)idx * TS_Q35_DF_HD : bk + (long long)(idx - ringRows) * TS_Q35_DF_HD;
                const half* vsrc = idx < ringRows ? rv + (long long)idx * TS_Q35_DF_HD : bv + (long long)(idx - ringRows) * TS_Q35_DF_HD;
                kv = *(const uint4*)(ksrc + c);
                vv = *(const uint4*)(vsrc + c);
            }
            *(uint4*)(ks + key * TS_Q35_DF_ROW + c) = kv;
            *(uint4*)(vs + key * TS_Q35_DF_ROW + c) = vv;
        }
        if (threadIdx.x < 32)
        {
            const int idx = first + threadIdx.x;
            // -2: a block key (always visible); -1: dead or past the range.
            kpos[threadIdx.x] = idx >= kHi ? -1 : idx >= ringRows ? -2 : sp[idx];
        }
        __syncthreads();
        if (!warpLive)
            continue;
        float s[4][4];
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            s[j][0] = s[j][1] = s[j][2] = s[j][3] = 0.0f;
            const half* kb = ks + (j * 8 + g) * TS_Q35_DF_ROW + qd * 2;
#pragma unroll
            for (int kk = 0; kk < 8; ++kk)
                ts_q35_mma_f16(s[j], qa[kk], *(const uint32_t*)(kb + kk * 16), *(const uint32_t*)(kb + kk * 16 + 8));
        }
        auto visible = [&](int key, int qp) {
            const int p = kpos[key];
            return p == -2 || (p >= 0 && p < anc && qp - p < swa);
        };
        float t0 = m0, t1 = m1;
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            const int key = j * 8 + qd * 2;
            s[j][0] = visible(key, qp0) ? s[j][0] * log2Scale : -INFINITY;
            s[j][1] = visible(key + 1, qp0) ? s[j][1] * log2Scale : -INFINITY;
            s[j][2] = visible(key, qp1) ? s[j][2] * log2Scale : -INFINITY;
            s[j][3] = visible(key + 1, qp1) ? s[j][3] * log2Scale : -INFINITY;
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
#pragma unroll
        for (int n = 0; n < 16; ++n)
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
            const half* vrow = vs + (kk * 16 + (lane & 15)) * TS_Q35_DF_ROW;
#pragma unroll
            for (int n = 0; n < 16; ++n)
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
        const long long base = ((long long)ln * b + r / G) * nHead + kvh * G + r % G;
        float* p = partial + (base * splits + split) * (TS_Q35_DF_HD + 2);
#pragma unroll
        for (int n = 0; n < 16; ++n)
        {
            const int x = n * 8 + qd * 2;
            p[x] = acc[n][half_ * 2];
            p[x + 1] = acc[n][half_ * 2 + 1];
        }
        if (qd == 0)
        {
            const float m = half_ ? m1 : m0;
            p[TS_Q35_DF_HD] = m == -INFINITY ? -INFINITY : m * 0.6931471805599453f;
            p[TS_Q35_DF_HD + 1] = half_ ? l1 : l0;
        }
    }
}

// One block of 128 threads per (row, head): out = sum_s w_s acc_s / sum_s w_s l_s.
extern "C" __global__ void ts_q35_df_combine_f32(const float* __restrict__ partial, int splits, float* __restrict__ out)
{
    const long long base = blockIdx.x;
    const float* p = partial + base * splits * (TS_Q35_DF_HD + 2);
    float mx = -INFINITY;
    for (int s = 0; s < splits; ++s)
        mx = fmaxf(mx, p[s * (TS_Q35_DF_HD + 2) + TS_Q35_DF_HD]);
    float num = 0.0f, den = 0.0f;
    const int x = threadIdx.x;
    for (int s = 0; s < splits; ++s)
    {
        const float* ps = p + s * (TS_Q35_DF_HD + 2);
        const float w = mx == -INFINITY ? 0.0f : expf(ps[TS_Q35_DF_HD] - mx);
        num = fmaf(ps[x], w, num);
        den = fmaf(ps[TS_Q35_DF_HD + 1], w, den);
    }
    out[base * TS_Q35_DF_HD + x] = den > 0.0f ? num / den : 0.0f;
}

// ---------------------------------------------------------------------------
// DFlash2 grouped dynamic depthwise convolution over each lane's block of b rows:
//   out[r][c] = sum_t (base[side][t][c] + coef[r][(side * taps + t) * groups + c / groupSize]) * x[r - t][c]
// with tap t dropped for the first t rows of the block. base is [2][taps][hidden]; grid
// (hidden blocks, rows).
// ---------------------------------------------------------------------------
extern "C" __global__ void ts_q35_df_conv_f32(
    const float* __restrict__ x, const float* __restrict__ coef, int coefWidth, const float* __restrict__ base,
    int side, int taps, int groupSize, int hidden, int b, float* __restrict__ out)
{
    const int c = blockIdx.x * blockDim.x + threadIdx.x, r = blockIdx.y;
    if (c >= hidden)
        return;
    const int groups = hidden / groupSize, local = r % b;
    float acc = 0.0f;
    for (int t = 0; t < taps && t <= local; ++t)
    {
        const float kern = base[((long long)side * taps + t) * hidden + c] + coef[(long long)r * coefWidth + (side * taps + t) * groups + c / groupSize];
        acc += kern * x[(long long)(r - t) * hidden + c];
    }
    out[(long long)r * hidden + c] = acc;
}

// dst[r] = src[(r / count) * perLane + first + r % count]: one lane's proposal rows of a block.
extern "C" __global__ void ts_q35_df_gather_f32(
    const float* __restrict__ src, int width, int perLane, int first, int count, float* __restrict__ dst)
{
    const int r = blockIdx.x;
    const float* in = src + ((long long)(r / count) * perLane + first + r % count) * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x)
        dst[(long long)r * width + i] = in[i];
}

#include "tensorsharp_q35_topk.cuh"

// ---------------------------------------------------------------------------
// Top k (k <= 16) of each row, largest first (ties: lower index first): one 256-thread block per
// row. Every thread keeps a sorted top 16 of its stride in registers (compile-time
// compare-and-swap, no local memory), then k block-wide extractions of the list heads merge them.
// ---------------------------------------------------------------------------
extern "C" __global__ void __launch_bounds__(256) ts_q35_topk_f32(
    const float* __restrict__ x, int n, int k, int* __restrict__ idxOut, float* __restrict__ valOut)
{
    __shared__ float sv[16][257];
    __shared__ int si[16][257];
    __shared__ float bv[8];
    __shared__ int bi[8], bslot[8];
    __shared__ int head[256];
    const float* row = x + (long long)blockIdx.x * n;
    float v[16];
    int ix[16];
#pragma unroll
    for (int j = 0; j < 16; ++j)
    {
        v[j] = -INFINITY;
        ix[j] = 0x7fffffff;
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x)
    {
        const float val = row[i];
        if (val > v[15])   // ascending i: an equal value never displaces an earlier index
        {
            v[15] = val;
            ix[15] = i;
#pragma unroll
            for (int j = 15; j > 0; --j)
                if (v[j] > v[j - 1])
                {
                    const float tv = v[j]; v[j] = v[j - 1]; v[j - 1] = tv;
                    const int ti = ix[j]; ix[j] = ix[j - 1]; ix[j - 1] = ti;
                }
        }
    }
#pragma unroll
    for (int j = 0; j < 16; ++j)
    {
        sv[j][threadIdx.x] = v[j];
        si[j][threadIdx.x] = ix[j];
    }
    head[threadIdx.x] = 0;
    __syncthreads();
    for (int out = 0; out < k; ++out)
    {
        const int h = head[threadIdx.x];
        float best = h < 16 ? sv[h][threadIdx.x] : -INFINITY;
        int bidx = h < 16 ? si[h][threadIdx.x] : 0x7fffffff, slot = threadIdx.x;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
        {
            const float ov = __shfl_xor_sync(0xffffffff, best, off);
            const int oi = __shfl_xor_sync(0xffffffff, bidx, off), os = __shfl_xor_sync(0xffffffff, slot, off);
            if (ov > best || (ov == best && oi < bidx))
            {
                best = ov;
                bidx = oi;
                slot = os;
            }
        }
        if ((threadIdx.x & 31) == 0)
        {
            bv[threadIdx.x >> 5] = best;
            bi[threadIdx.x >> 5] = bidx;
            bslot[threadIdx.x >> 5] = slot;
        }
        __syncthreads();
        if (threadIdx.x == 0)
        {
            int w = 0;
            for (int i = 1; i < 8; ++i)
                if (bv[i] > bv[w] || (bv[i] == bv[w] && bi[i] < bi[w]))
                    w = i;
            idxOut[(long long)blockIdx.x * k + out] = bi[w];
            valOut[(long long)blockIdx.x * k + out] = bv[w];
            head[bslot[w]]++;
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------
// DFlash2 candidate selector and its greedy walk, one 256-thread block per lane. With ph[e] =
// selector_hidden(proposal row e), keys = selector_successor rows and preds = selector_predecessor
// rows (Q4_K, 256 wide):
//   score[0][c]    = <succ[cand[0][c]], pred[anchor] * ph[0]> + unary[0][c]
//   score[e][p][c] = <succ[cand[e][c]], pred[cand[e-1][p]] * ph[e]> + unary[e][c]
// The walk takes the argmax of row 0, then of row (e, previous choice) - so only the chosen
// predecessor's row is ever scored. conf[e] is the softmax of that row at the choice.
// ---------------------------------------------------------------------------
__device__ __forceinline__ float ts_q35_q4k_value(const uint8_t* blk, int i)
{
    const int c = i >> 6, l = i & 63, j = 2 * c + (l >= 32);
    const int q = (blk[16 + 32 * c + (l & 31)] >> (l >= 32 ? 4 : 0)) & 15;
    int sc, m;
    ts_q35_q4k_scale_min(j, blk + 4, sc, m);
    return __half2float(*(const half*)blk) * (float)sc * (float)q - __half2float(*(const half*)(blk + 2)) * (float)m;
}

extern "C" __global__ void __launch_bounds__(256) ts_q35_df_select(
    const int* __restrict__ cand, const float* __restrict__ unary, const float* __restrict__ ph,
    const uint8_t* __restrict__ pred, const uint8_t* __restrict__ succ, long long rowBytes,
    const int* __restrict__ anchors, int gamma, int k, int rank, float logitScale,
    int* __restrict__ draftsOut, float* __restrict__ confOut)
{
    __shared__ float m[256];
    __shared__ float score[16];
    __shared__ int chosen;
    const int ln = blockIdx.x, tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    if (tid == 0)
        chosen = -1;
    __syncthreads();
    for (int e = 0; e < gamma; ++e)
    {
        const int* ce = cand + ((long long)ln * gamma + e) * k;
        // m = pred[predecessor] * ph[e]: the anchor for position 0, else the previous choice.
        const int predTok = e == 0 ? anchors[ln] : cand[((long long)ln * gamma + e - 1) * k + chosen];
        for (int i = tid; i < rank; i += blockDim.x)
            m[i] = ts_q35_q4k_value(pred + (long long)predTok * rowBytes + (i >> 8) * 144, i & 255)
                 * ph[((long long)ln * gamma + e) * rank + i];
        __syncthreads();
        for (int c = warp; c < k; c += blockDim.x >> 5)
        {
            const uint8_t* row = succ + (long long)ce[c] * rowBytes;
            float s = 0.0f;
            for (int i = lane; i < rank; i += 32)
                s += ts_q35_q4k_value(row + (i >> 8) * 144, i & 255) * m[i];
            s = ts_q35_warp_sum(s);
            if (lane == 0)
                score[c] = s + unary[((long long)ln * gamma + e) * k + c] * logitScale;
        }
        __syncthreads();
        if (tid == 0)
        {
            int best = 0;
            for (int c = 1; c < k; ++c)
                if (score[c] > score[best])
                    best = c;
            float den = 0.0f;
            for (int c = 0; c < k; ++c)
                den += expf(score[c] - score[best]);
            chosen = best;
            draftsOut[(long long)ln * gamma + e] = ce[best];
            confOut[(long long)ln * gamma + e] = 1.0f / den;
        }
        __syncthreads();
    }
}
