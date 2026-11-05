// Standalone experiment only. Production kernels and exported ABI are untouched.
// Each new warp owns one half of an ORIGINAL16-query warp. Keep that half in
// the same MMA m-row positions; zero the other half without carrying dummy C.
__device__ __forceinline__ void ts_bench_mma_kept(
    uint32_t& kept, uint32_t a0, uint32_t a1, uint32_t b0, uint32_t b1, bool upper)
{
    uint32_t discarded;
    // PTX matrix operands are registers. One constant-zero register is reused;
    // no permanent four-register A fragment or dummy C accumulator is retained.
    if (upper)
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%1,%0}, {%6,%2,%6,%3}, {%4,%5}, {%6,%0};\n"
            : "+r"(kept), "=r"(discarded) : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(0u));
    else
        asm volatile("mma.sync.aligned.m16n8k16.row.col.f16.f16.f16.f16 {%0,%1}, {%2,%6,%3,%6}, {%4,%5}, {%0,%6};\n"
            : "+r"(kept), "=r"(discarded) : "r"(a0), "r"(a1), "r"(b0), "r"(b1), "r"(0u));
}

// Four int32 votes fit each K-row's eight unused half-padding values. The two
// rows used below are never read by ldmatrix, and dequant writes only0..255.
__device__ __forceinline__ volatile int* ts_bench_vote(TsQ35AttnPrefill& sm, int warp)
{
#if TS_BENCH_COMPACT_SWIZZLE
    // The compact variant has no padding. A separate all-CTA barrier after QK
    // proves these first32K bytes dead until next iteration's dequantization.
    return reinterpret_cast<volatile int*>(sm.k) + warp;
#else
    return reinterpret_cast<volatile int*>(sm.k + (warp >> 2) * TS_Q35_ATTN_ROW + TS_Q35_HD) + (warp & 3);
#endif
}

#if TS_BENCH_COMPACT_SWIZZLE
__device__ __forceinline__ half* ts_bench_swizzled_at(half* base, int key, int col)
{
    return base + key * TS_Q35_HD + (col ^ ((key & 7) * 8));
}

// The original conversion is unchanged. Only the four8-half uint4 stores move:
// XOR touches col bits3..5, preserving each16-byte vector's order/alignment.
__device__ __forceinline__ void ts_bench_tile_to_h_swizzled(const uint8_t* raw, half* dst, int tid)
{
    const int key = tid & 31, b = tid >> 5;
    const uint8_t* row = raw + key * TS_Q35_Q8_ROW;
    const half2 d = __half2half2(*(const half*)(row + TS_Q35_Q8_SCALES + 2 * b));
    const uint4* codes = (const uint4*)(row + b * 32);
#pragma unroll
    for (int c = 0; c < 2; ++c)
    {
        const uint4 w = codes[c];
        half2 h[8];
        ts_q35_i8x4_to_h2(w.x, d, h[0], h[1]);
        ts_q35_i8x4_to_h2(w.y, d, h[2], h[3]);
        ts_q35_i8x4_to_h2(w.z, d, h[4], h[5]);
        ts_q35_i8x4_to_h2(w.w, d, h[6], h[7]);
        *reinterpret_cast<uint4*>(ts_bench_swizzled_at(dst, key, b * 32 + c * 16)) = *reinterpret_cast<const uint4*>(&h[0]);
        *reinterpret_cast<uint4*>(ts_bench_swizzled_at(dst, key, b * 32 + c * 16 + 8)) = *reinterpret_cast<const uint4*>(&h[4]);
    }
}
#endif

extern "C" __global__ void __launch_bounds__(256) ts_bench_attn_prefill(
    const float* __restrict__ q, const float* __restrict__ gate, const int* __restrict__ keyEnd,
    uint8_t* const* __restrict__ blockTable, int maxBlocks, long long kOff, long long vOff,
    int rowsPerSeq, int nHead, int nKv, float scale, int splits, float* __restrict__ partial, float* __restrict__ out)
{
    extern __shared__ __align__(16) uint8_t ts_q35_attn_smem[];
    TsQ35AttnPrefill& sm = *reinterpret_cast<TsQ35AttnPrefill*>(ts_q35_attn_smem);
    const int kvh = blockIdx.y, seq = blockIdx.z, G = nHead / nKv, R = rowsPerSeq * G;
    const int split = blockIdx.x % splits, sub = (blockIdx.x / splits) & 1;
    const int logicalBase = (blockIdx.x / (2 * splits)) * TS_Q35_AI_ROWS;
    const int rowBase = logicalBase + sub * 64;
    if (rowBase >= R) return; // Entire CTA has no output rows; no buddy can be outside this CTA.
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5, g = lane >> 2, qd = lane & 3;
    const bool upper = (warp & 1) != 0;
    const int* ends = keyEnd + (long long)seq * rowsPerSeq;
    // DO NOT derive kAll from the new64-row subtile. Every original query must
    // retain its old tile's exact32-key range and split-combine ordering.
    const int kAll = ends[(min(logicalBase + TS_Q35_AI_ROWS, R) - 1) / G];
    const int share = ((kAll + splits - 1) / splits + TS_Q35_AI_KT - 1) / TS_Q35_AI_KT * TS_Q35_AI_KT;
    const int kLo = min(kAll, split * share), kHi = min(kAll, kLo + share);
    uint8_t* const* blocks = blockTable + (long long)seq * maxBlocks;
    const int r = rowBase + warp * 8 + g;
    const bool live = r < R, warpLive = rowBase + warp * 8 < R;
    const int end = live ? ends[r / G] : 0;
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
    if (kHi > kLo) load(0, kLo);
    ts_q35_cp_commit();
    const float log2Scale = scale * 1.4426950408889634f;
    uint32_t qa[16][2];
    {
        const float* qr = live ? q + (((long long)seq * rowsPerSeq + r / G) * nHead + kvh * G + r % G) * TS_Q35_HD : nullptr;
#pragma unroll
        for (int kk = 0; kk < 16; ++kk)
        {
            const int c = kk * 16 + qd * 2;
            qa[kk][0] = qr ? ts_q35_pack_h2(qr[c] * log2Scale, qr[c + 1] * log2Scale) : 0u;
            qa[kk][1] = qr ? ts_q35_pack_h2(qr[c + 8] * log2Scale, qr[c + 9] * log2Scale) : 0u;
        }
    }
    float acc[32][2];
#pragma unroll
    for (int n = 0; n < 32; ++n) acc[n][0] = acc[n][1] = 0.0f;
    float m = -INFINITY, l = 0.0f;
    for (int first = kLo, t = 0; first < kHi; first += TS_Q35_AI_KT, ++t)
    {
        const int st = TS_BENCH_RAW_STAGES == 1 ? 0 : t & 1;
        ts_q35_cp_wait<0>();
        __syncthreads(); // A: old K/V consumers done, current raw ready.
#if TS_BENCH_RAW_STAGES == 2
        if (first + TS_Q35_AI_KT < kHi) load(st ^ 1, first + TS_Q35_AI_KT);
        ts_q35_cp_commit();
#endif
#if TS_BENCH_COMPACT_SWIZZLE
        ts_bench_tile_to_h_swizzled(sm.raw[st][0], sm.k, tid);
        ts_bench_tile_to_h_swizzled(sm.raw[st][1], sm.v, tid);
#else
        ts_q35_tile_to_h(sm.raw[st][0], sm.k, tid);
        ts_q35_tile_to_h(sm.raw[st][1], sm.v, tid);
#endif
        __syncthreads(); // B: all raw readers finished, K/V ready.
#if TS_BENCH_RAW_STAGES == 1
        if (first + TS_Q35_AI_KT < kHi) load(0, first + TS_Q35_AI_KT);
        ts_q35_cp_commit();
#endif
        float s[4][2], nextM = m, c = 1.0f;
        if (warpLive)
        {
#if !TS_BENCH_COMPACT_SWIZZLE
            const half* kb = sm.k + ((lane & 7) + ((lane >> 4) << 3)) * TS_Q35_ATTN_ROW + ((lane >> 3) & 1) * 8;
#endif
#pragma unroll
            for (int np = 0; np < 2; ++np)
            {
#pragma unroll
                for (int e = 0; e < 2; ++e) s[2 * np][e] = s[2 * np + 1][e] = 0.0f;
#pragma unroll
                for (int kk = 0; kk < 16; ++kk) // Original AH_CHAIN=1; preserve every half rounding.
                {
                    uint32_t h0 = 0u, h1 = 0u, b[4];
#if TS_BENCH_COMPACT_SWIZZLE
                    ts_q35_ldsm_x4(b, ts_bench_swizzled_at(sm.k, (lane & 7) + ((lane >> 4) << 3) + np * 16, ((lane >> 3) & 1) * 8 + kk * 16));
#else
                    ts_q35_ldsm_x4(b, kb + np * 16 * TS_Q35_ATTN_ROW + kk * 16);
#endif
                    ts_bench_mma_kept(h0, qa[kk][0], qa[kk][1], b[0], b[1], upper);
                    ts_bench_mma_kept(h1, qa[kk][0], qa[kk][1], b[2], b[3], upper);
                    const float2 a = __half22float2(*reinterpret_cast<const half2*>(&h0));
                    const float2 b_ = __half22float2(*reinterpret_cast<const half2*>(&h1));
                    s[2 * np][0] += a.x; s[2 * np][1] += a.y;
                    s[2 * np + 1][0] += b_.x; s[2 * np + 1][1] += b_.y;
                }
            }
#pragma unroll
            for (int j = 0; j < 4; ++j)
            {
                const int key = first + j * 8 + qd * 2;
                s[j][0] = key < end ? s[j][0] : -INFINITY;
                s[j][1] = key + 1 < end ? s[j][1] : -INFINITY;
                nextM = fmaxf(nextM, fmaxf(s[j][0], s[j][1]));
            }
#pragma unroll
            for (int off = 1; off < 4; off <<= 1) nextM = fmaxf(nextM, __shfl_xor_sync(0xffffffff, nextM, off));
            c = nextM == -INFINITY ? 1.0f : exp2f(m - nextM);
            m = nextM;
            l *= c;
        }
#if TS_BENCH_COMPACT_SWIZZLE
        // ALL warps must finish their last K ldmatrix before its live data is
        // borrowed for votes. Non-live buddies participate here as well.
        __syncthreads();
#endif
        const int ownVote = __all_sync(0xffffffff, c == 1.0f);
        if (lane == 0) *ts_bench_vote(sm, warp) = ownVote;
        // Reproduce ORIGINAL16-query warp voting exactly. Inactive buddy warps
        // also writetrue and join this barrier; ragged tails must not deadlock.
        __syncthreads();
        const bool originalSkip = *ts_bench_vote(sm, warp) && *ts_bench_vote(sm, warp ^ 1);
        if (!warpLive) continue;
        if (!originalSkip)
        {
#pragma unroll
            for (int n = 0; n < 32; ++n) { acc[n][0] *= c; acc[n][1] *= c; }
        }
#pragma unroll
        for (int j = 0; j < 4; ++j)
        {
            s[j][0] = nextM == -INFINITY ? 0.0f : exp2f(s[j][0] - nextM);
            s[j][1] = nextM == -INFINITY ? 0.0f : exp2f(s[j][1] - nextM);
            l += s[j][0] + s[j][1];
        }
        uint32_t pa[2][2];
#pragma unroll
        for (int kk = 0; kk < 2; ++kk)
        {
            pa[kk][0] = ts_q35_pack_h2(s[2 * kk][0], s[2 * kk][1]);
            pa[kk][1] = ts_q35_pack_h2(s[2 * kk + 1][0], s[2 * kk + 1][1]);
        }
#if !TS_BENCH_COMPACT_SWIZZLE
        const half* vrow = sm.v + ((lane & 7) + (((lane >> 3) & 1) << 3)) * TS_Q35_ATTN_ROW + ((lane >> 4) << 3);
#endif
#pragma unroll
        for (int dn = 0; dn < 16; ++dn)
        {
            uint32_t h0 = 0u, h1 = 0u;
#pragma unroll
            for (int kk = 0; kk < 2; ++kk)
            {
                uint32_t b[4];
#if TS_BENCH_COMPACT_SWIZZLE
                ts_q35_ldsm_x4_t(b, ts_bench_swizzled_at(sm.v, (lane & 7) + (((lane >> 3) & 1) << 3) + kk * 16, ((lane >> 4) << 3) + dn * 16));
#else
                ts_q35_ldsm_x4_t(b, vrow + kk * 16 * TS_Q35_ATTN_ROW + dn * 16);
#endif
                ts_bench_mma_kept(h0, pa[kk][0], pa[kk][1], b[0], b[1], upper);
                ts_bench_mma_kept(h1, pa[kk][0], pa[kk][1], b[2], b[3], upper);
            }
            const float2 a = __half22float2(*reinterpret_cast<const half2*>(&h0));
            const float2 b = __half22float2(*reinterpret_cast<const half2*>(&h1));
            acc[2 * dn][0] += a.x; acc[2 * dn][1] += a.y;
            acc[2 * dn + 1][0] += b.x; acc[2 * dn + 1][1] += b.y;
        }
    }
    ts_q35_cp_wait<0>();
    if (!warpLive) return;
#pragma unroll
    for (int off = 1; off < 4; off <<= 1) l += __shfl_xor_sync(0xffffffff, l, off);
    if (!live) return;
    const long long base = ((long long)seq * rowsPerSeq + r / G) * nHead + kvh * G + r % G;
    if (splits > 1)
    {
        float* p = partial + (base * splits + split) * (TS_Q35_HD + 2);
#pragma unroll
        for (int n = 0; n < 32; ++n) { const int x = n * 8 + qd * 2; p[x] = acc[n][0]; p[x + 1] = acc[n][1]; }
        if (qd == 0) { p[TS_Q35_HD] = m == -INFINITY ? -INFINITY : m * 0.6931471805599453f; p[TS_Q35_HD + 1] = l; }
    }
    else
    {
        const float inv = l > 0.0f ? 1.0f / l : 0.0f;
        float* o = out + base * TS_Q35_HD;
        const float* gt = gate + base * TS_Q35_HD;
#pragma unroll
        for (int n = 0; n < 32; ++n)
        {
            const int x = n * 8 + qd * 2;
            o[x] = acc[n][0] * inv * ts_q35_sigmoid(gt[x]);
            o[x + 1] = acc[n][1] * inv * ts_q35_sigmoid(gt[x + 1]);
        }
    }
}
