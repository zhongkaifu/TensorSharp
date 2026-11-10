// tsm_df_tail.metal - the end of a DFlash2 draft on the GPU (TensorSharp's tsm library): the target's output head over the
// block's proposal rows, the top-k of each row, and the candidate selector's transition lattice.
//
// ggml runs this as a matrix-vector product per proposal row (ggml-metal's mul_mv_ext on the q5_K head: ~177 GB/s at
// 7 rows, against ~237 GB/s at one), a top-k over the 248K-wide vocabulary, and ~15 small selector ops. Here the head
// runs on the matrix units over all proposal rows at once (8 activation columns), reading ggml's own q5_K blocks; the
// top-k is two passes; the lattice is one kernel.
//
// The top-k passes and the selector's walk follow Strata (https://github.com/Niko1221/Strata,
// metal/src/engine/msl/dflash2.metal df2_topk_part / df2_topk_merge / df2_select, and df2_mma8 for the fragment
// mapping; commit 146df3c4, MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors; tsm/LICENSE-STRATA).
// Changes: the head reads ggml's q5_K blocks in place (Strata repacks Q8_0 rows of a draft vocabulary), the number of
// top-k blocks is a run-time argument, and the selector kernel writes the whole lattice (the k anchor scores, then a
// [k(predecessor)][k(candidate)] matrix per later position) for the host's greedy walk instead of walking it.

// ---- shared: the K-quant scale and min of sub-block j (ggml's get_scale_min_k4)
static inline void tsm_df_scale_min(uint j, device const uchar* q, thread uint& sc, thread uint& m) {
    if (j < 4) {
        sc = q[j] & 63u;
        m = q[j + 4] & 63u;
    } else {
        sc = (q[j + 4] & 0xFu) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
    }
}

struct TsmDfHeadArgs {
    uint K;          // the hidden size (a multiple of 256)
    uint rows;       // output rows (the vocabulary)
    uint row_bytes;  // bytes of one weight row
    uint ncol;       // activation columns (proposal rows), 1..8
    uint x_row0;     // the first proposal row of x
    float scale;     // the head's quantization scale (sidecar), 1 without one
    uint pad0, pad1;
};

// The activation operand: xh[c][kb * 32 + perm(l)] = half(x[x_row0 + c][kb * 32 + l]) for c < ncol, zero for the other
// columns of 8, with the 32 values of each sub-block permuted so that a lane of the head's fragments reads 8
// contiguous code bytes (see tsm_df_head_q5k); xs[c][kb] = the sum of the sub-block's (rounded) values, the term the
// sub-block minimum multiplies. One thread per (sub-block, column): grid (K / 32, 8).
kernel void tsm_df_xprep(device const float* x [[buffer(0)]], device half* xh [[buffer(1)]], device float* xs [[buffer(2)]],
                         constant TsmDfHeadArgs& a [[buffer(3)]], uint2 gid [[thread_position_in_grid]]) {
    const uint kb = gid.x, c = gid.y, nkb = a.K / 32;
    if (kb >= nkb || c >= 8) return;
    device const float* src = x + ulong(a.x_row0 + c) * a.K + kb * 32;
    device half* dst = xh + ulong(c) * a.K + kb * 32;
    float s = 0.0f;
    for (uint l = 0; l < 32; ++l) {
        const half v = c < a.ncol ? half(src[l]) : half(0.0f);
        // element l of the sub-block sits at k step (l % 8) / 2, fragment column 2 * (l / 8) + l % 2
        dst[8 * ((l & 7) >> 1) + 2 * (l >> 3) + (l & 1)] = v;
        s += float(v);
    }
    xs[ulong(c) * nkb + kb] = s;
}

// y[c][row] = scale * sum_k W[row][k] x[c][k] for the ncol columns: q5_K rows (176-byte blocks of 256: half d, half dmin,
// 12 bytes of 6-bit scales and mins, 32 bytes of high bits, 128 bytes of low nibbles) on simdgroup_matrix 8x8 tiles, RT
// tiles of 8 rows per simdgroup, SG simdgroups per threadgroup. Per sub-block of 32: C = codes (0..31, exact in half) .
// x (half), then acc += d * sc * C - dmin * m * xs. The fragment of lane (row fm, columns fn, fn + 1) at k step s holds
// the codes of sub-block positions 8 * (fn / 2) + 2 s + e (the permutation tsm_df_xprep applied to x), so a lane's 8
// codes of a sub-block are 8 contiguous bytes: one uint2 of low nibbles per pair of sub-blocks and one uint2 of high
// bits per block.
template <int RT, int SG>
kernel void tsm_df_head_q5k(device const uchar* W [[buffer(0)]], device const half* xh [[buffer(1)]],
                            device const float* xs [[buffer(2)]], device float* y [[buffer(3)]],
                            constant TsmDfHeadArgs& a [[buffer(4)]], uint3 tg [[threadgroup_position_in_grid]],
                            ushort sgi [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    const uint nsb = a.K / 256, nkb = a.K / 32;
    const uint nrt = (a.rows + 7) / 8;
    const uint rt0 = (tg.x * SG + sgi) * RT;
    if (rt0 >= nrt) return;
    const uint qid = uint(lane) >> 2;
    const uint fm = (qid & 4) + ((uint(lane) >> 1) & 3);
    const uint fn = (qid & 2) * 2 + (uint(lane) & 1) * 2;
    const uint byte0 = 8 * (fn >> 1);
    float acc0[RT], acc1[RT];
    device const uchar* wr[RT];
    for (int r = 0; r < RT; ++r) {
        acc0[r] = 0.0f; acc1[r] = 0.0f;
        const uint row = min((rt0 + uint(r)) * 8 + fm, a.rows - 1);
        wr[r] = W + ulong(row) * a.row_bytes;
    }
    device const float* xs0 = xs + ulong(fn) * nkb;
    device const float* xs1 = xs + ulong(fn + 1) * nkb;
    for (uint sb = 0; sb < nsb; ++sb) {
        float dsc[RT][8], dmn[RT][8];
        uint2 qh[RT];
        for (int r = 0; r < RT; ++r) {
            device const uchar* blk = wr[r] + sb * 176;
            const float d = float(*(device const half*)blk), dmin = float(*(device const half*)(blk + 2));
            for (uint j = 0; j < 8; ++j) {
                uint sc, m;
                tsm_df_scale_min(j, blk + 4, sc, m);
                dsc[r][j] = d * float(sc);
                dmn[r][j] = dmin * float(m);
            }
            qh[r] = *(device const uint2*)(blk + 16 + byte0);
        }
        for (uint jp = 0; jp < 4; ++jp) {
            uint2 ql[RT];
            for (int r = 0; r < RT; ++r) ql[r] = *(device const uint2*)(wr[r] + sb * 176 + 48 + jp * 32 + byte0);
            for (uint h = 0; h < 2; ++h) {
                const uint j = jp * 2 + h, kb = sb * 8 + j;
                device const half* xb = xh + kb * 32;
                simdgroup_half8x8 B0, B1, B2, B3;
                simdgroup_load(B0, xb + 0, a.K, ulong2(0, 0), true);
                simdgroup_load(B1, xb + 8, a.K, ulong2(0, 0), true);
                simdgroup_load(B2, xb + 16, a.K, ulong2(0, 0), true);
                simdgroup_load(B3, xb + 24, a.K, ulong2(0, 0), true);
                const float x0 = xs0[kb], x1 = xs1[kb];
                for (int r = 0; r < RT; ++r) {
                    // byte i (= 2 s + e) of the lane's run: the low nibble (even sub-block) or high nibble (odd), plus
                    // the high bit j of the same position
                    const uint lo = ql[r].x, hi = ql[r].y, b0 = qh[r].x, b1 = qh[r].y;
                    const uint sh = 4 * h;
                    simdgroup_half8x8 A0, A1, A2, A3;
                    #define TSM_DF_Q5(word, hb, byte) half(float((((word) >> (8 * (byte) + sh)) & 15u) | ((((hb) >> (8 * (byte) + j)) & 1u) << 4)))
                    A0.thread_elements()[0] = TSM_DF_Q5(lo, b0, 0); A0.thread_elements()[1] = TSM_DF_Q5(lo, b0, 1);
                    A1.thread_elements()[0] = TSM_DF_Q5(lo, b0, 2); A1.thread_elements()[1] = TSM_DF_Q5(lo, b0, 3);
                    A2.thread_elements()[0] = TSM_DF_Q5(hi, b1, 0); A2.thread_elements()[1] = TSM_DF_Q5(hi, b1, 1);
                    A3.thread_elements()[0] = TSM_DF_Q5(hi, b1, 2); A3.thread_elements()[1] = TSM_DF_Q5(hi, b1, 3);
                    #undef TSM_DF_Q5
                    simdgroup_float8x8 C = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
                    simdgroup_multiply_accumulate(C, A0, B0, C);
                    simdgroup_multiply_accumulate(C, A1, B1, C);
                    simdgroup_multiply_accumulate(C, A2, B2, C);
                    simdgroup_multiply_accumulate(C, A3, B3, C);
                    acc0[r] = fma(dsc[r][j], C.thread_elements()[0], fma(-dmn[r][j], x0, acc0[r]));
                    acc1[r] = fma(dsc[r][j], C.thread_elements()[1], fma(-dmn[r][j], x1, acc1[r]));
                }
            }
        }
    }
    for (int r = 0; r < RT; ++r) {
        const uint row = (rt0 + uint(r)) * 8 + fm;
        if (rt0 + uint(r) >= nrt || row >= a.rows) continue;
        if (fn < a.ncol) y[ulong(fn) * a.rows + row] = a.scale * acc0[r];
        if (fn + 1 < a.ncol) y[ulong(fn + 1) * a.rows + row] = a.scale * acc1[r];
    }
}

// One row tile per simdgroup: on an M5 Pro (Flash Next's 2560 x 248320 head, 7 rows) 1.95 ms, ~224 GB/s; two or four
// tiles per simdgroup ran out of registers (3.2 ms), staging the activations in threadgroup memory did not pay (2.0 ms),
// and unpacking four codes per integer operation was slower (2.1 ms).
template [[host_name("tsm_df_head_q5k")]] kernel decltype(tsm_df_head_q5k<1, 4>) tsm_df_head_q5k<1, 4>;

// ---- top-k of each row of logits [rows][n] (Strata's df2_topk_part / df2_topk_merge): TSM_DF_TOPK best of each of
// `blocks` column blocks, then the TSM_DF_TOPK best of those. Ties go to the lower index.
#define TSM_DF_TOPK 16
#define TSM_DF_TOPB_MAX 128
struct TsmDfTopkArgs { int n; int blocks; };

static inline bool tsm_df_better(float va, int ia, float vb, int ib) { return va > vb || (va == vb && ia < ib); }

// grid (rows, blocks), 256 threads; a block holds at most 4096 columns
kernel void tsm_df_topk_part(device const float* logits [[buffer(0)]], device float* cval [[buffer(1)]], device int* cidx [[buffer(2)]],
                             constant TsmDfTopkArgs& a [[buffer(3)]], uint3 tg [[threadgroup_position_in_grid]],
                             ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                             ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float vals[4096];
    threadgroup float wv[8];
    threadgroup int wi[8];
    const int row = int(tg.x), blk = int(tg.y), n = a.n;
    const int cs = (n + a.blocks - 1) / a.blocks;
    const int c0 = blk * cs, c1 = min(c0 + cs, n);
    for (int i = tid; i < cs; i += 256) vals[i] = (c0 + i < c1) ? logits[long(row) * n + c0 + i] : -INFINITY;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < TSM_DF_TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
        for (int i = tid; i < cs; i += 256)
            if (tsm_df_better(vals[i], i, bv, bi)) { bv = vals[i]; bi = i; }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const int oi = simd_shuffle_xor(bi, o);
            if (tsm_df_better(ov, oi, bv, bi)) { bv = ov; bi = oi; }
        }
        if (lane == 0) { wv[sg] = bv; wi[sg] = bi; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0];
            for (int w = 1; w < 8; ++w)
                if (tsm_df_better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; }
            const int o = (row * a.blocks + blk) * TSM_DF_TOPK + k;
            cval[o] = v;
            cidx[o] = v == -INFINITY ? -1 : c0 + ix;
            if (v != -INFINITY) vals[ix] = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// grid (rows), 256 threads: tok[row][k] = the column of the row's k-th best logit
kernel void tsm_df_topk_merge(device const float* cval [[buffer(0)]], device const int* cidx [[buffer(1)]],
                              device int* tok [[buffer(2)]], constant TsmDfTopkArgs& a [[buffer(3)]],
                              uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                              ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float sv[TSM_DF_TOPB_MAX * TSM_DF_TOPK];
    threadgroup int si[TSM_DF_TOPB_MAX * TSM_DF_TOPK];
    threadgroup float wv[8];
    threadgroup int wi[8], wp[8];
    const int row = int(tg.x), nc = a.blocks * TSM_DF_TOPK;
    for (int i = tid; i < nc; i += 256) { sv[i] = cval[row * nc + i]; si[i] = cidx[row * nc + i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = 0; k < TSM_DF_TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff, bp = -1;
        for (int i = tid; i < nc; i += 256)
            if (si[i] >= 0 && tsm_df_better(sv[i], si[i], bv, bi)) { bv = sv[i]; bi = si[i]; bp = i; }
        for (ushort o = 16; o > 0; o >>= 1) {
            const float ov = simd_shuffle_xor(bv, o);
            const int oi = simd_shuffle_xor(bi, o);
            const int op = simd_shuffle_xor(bp, o);
            if (tsm_df_better(ov, oi, bv, bi)) { bv = ov; bi = oi; bp = op; }
        }
        if (lane == 0) { wv[sg] = bv; wi[sg] = bi; wp[sg] = bp; }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            float v = wv[0];
            int ix = wi[0], p = wp[0];
            for (int w = 1; w < 8; ++w)
                if (tsm_df_better(wv[w], wi[w], v, ix)) { v = wv[w]; ix = wi[w]; p = wp[w]; }
            if (p >= 0) si[p] = -1;
            tok[row * TSM_DF_TOPK + k] = p >= 0 ? ix : 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// ---- the selector's transition lattice over q4_K codebooks of rank 256 (one 144-byte block a row: half d, half dmin,
// 12 bytes of scales and mins, 128 bytes of nibbles). Threadgroup t = 0 scores position 0's candidates against the
// anchor; t = 1 + (e - 1) * k + p scores position e's candidates against position e - 1's candidate p:
// score[c] = unary(e, c) + succ_scale * dot(succ[tok[e][c]], pred_scale * pred[prev] * ph[e]), with unary the head logit
// of the candidate through the target's logit scale and tanh softcap. 256 threads (one per rank element), 8
// simdgroups of two candidates each.
struct TsmDfSelArgs {
    uint rank, k, gamma, vocab;
    int anchor;
    float pred_scale, succ_scale, logit_scale, softcap;
    uint pad0, pad1, pad2;
};

static inline float tsm_df_q4k(device const uchar* row, uint i) {
    const uint j = i >> 5, l = i & 31;
    uint sc, m;
    tsm_df_scale_min(j, row + 4, sc, m);
    const float d = float(*(device const half*)row), dmin = float(*(device const half*)(row + 2));
    const uint q = (row[16 + 32 * (j >> 1) + l] >> (4 * (j & 1))) & 15u;
    return d * float(sc) * float(q) - dmin * float(m);
}

kernel void tsm_df_select_q4k(device const float* ph [[buffer(0)]], device const float* logits [[buffer(1)]],
                              device const int* tok [[buffer(2)]], device const uchar* pred [[buffer(3)]],
                              device const uchar* succ [[buffer(4)]], device float* out [[buffer(5)]],
                              constant TsmDfSelArgs& a [[buffer(6)]], uint3 tg [[threadgroup_position_in_grid]],
                              ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                              ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup float m[256];
    const uint t = tg.x, k = a.k;
    const uint e = t == 0 ? 0 : 1 + (t - 1) / k, p = t == 0 ? 0 : (t - 1) % k;
    const int prev = t == 0 ? a.anchor : tok[(e - 1) * k + p];
    const ulong rb = (a.rank / 256) * 144;
    m[tid] = a.pred_scale * tsm_df_q4k(pred + ulong(prev) * rb, tid) * ph[ulong(e) * a.rank + tid];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint c = uint(sg) * 2; c < uint(sg) * 2 + 2; ++c) {
        const int cand = tok[e * k + c];
        device const uchar* row = succ + ulong(cand) * rb;
        float acc = 0.0f;
        for (uint i = uint(lane) * 8; i < uint(lane) * 8 + 8; ++i) acc = fma(tsm_df_q4k(row, i), m[i], acc);
        acc = simd_sum(acc);
        if (lane == 0) {
            float u = logits[ulong(e) * a.vocab + uint(cand)] * a.logit_scale;
            if (a.softcap > 0.0f) u = a.softcap * precise::tanh(u / a.softcap);
            const float v = u + a.succ_scale * acc;
            if (t == 0) out[c] = v;
            else out[k + (e - 1) * k * k + p * k + c] = v;
        }
    }
}
