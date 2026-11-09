// A QSA layer's cell selection for a prompt chunk (TensorSharp's): the cells each token attends, as the attention mask,
// in two passes instead of ggml's chain over [cells, T] (head sum, bias add, a transpose, a gather of every cell's block
// score, a transpose back, the mask's cast and add, a top-k sort of every row, the mask fill). The contract is
// eng/tests/qwen4exp-qsa.py's: the `width` best cells by (block score + bias) + mask among the cells the mask allows,
// every strictly better cell taken, exact cardinality; cells tied at the cutoff are taken lowest index first.
//
//   tsm_qsa_block_scores  S[t][b] = (sum over heads, in order, of relu(scores[t][h][b])) + bias[t][b]
//   tsm_qsa_select        per token: the cutoff by a 4-pass radix select over the cells' order-preserving keys, then
//                         mask_out = mask_in at the chosen cells, -inf elsewhere
//
// Layouts (ggml's tensors): scores [T][H][B], bias and S [T][B], cell_blocks [cells], masks [T][cells] (F16).

struct TsmQsaArgs {
    uint n_tok;
    uint n_cells;
    uint n_blocks;
    uint heads;
    uint width;
    uint write_mask;   // mask_out = the selected mask
    uint write_idx;    // idx [T][width] = the selected cells in ascending order, cnt [T] = their number
    uint pad;
};

// grid (ceil(B / 256), T), 256 threads
kernel void tsm_qsa_block_scores(device const float* scores [[buffer(0)]], device const float* bias [[buffer(1)]],
                                 device float* S [[buffer(2)]], constant TsmQsaArgs& a [[buffer(3)]],
                                 uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const uint b = tg.x * 256 + tid, t = tg.y, B = a.n_blocks;
    if (b >= B) return;
    device const float* row = scores + ulong(t) * a.heads * B + b;
    float s = max(row[0], 0.0f);
    for (uint h = 1; h < a.heads; ++h) s = s + max(row[h * B], 0.0f);
    S[ulong(t) * B + b] = s + bias[ulong(t) * B + b];
}

// The order-preserving key of a finite float (larger float -> larger key).
static inline uint tsm_qsa_key(float e) {
    const uint u = as_type<uint>(e);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

constant constexpr uint kQsaThreads = 256;

// The cell's value as ggml computes it: (S + bias) gathered by block, plus the mask (F16 -> F32).
static inline float tsm_qsa_value(device const float* srow, device const int* cell_blocks, device const half* mrow,
                                  uint c) {
    return srow[cell_blocks[c]] + float(mrow[c]);
}

// grid (T), 256 threads
kernel void tsm_qsa_select(device const float* S [[buffer(0)]], device const int* cell_blocks [[buffer(1)]],
                           device const half* mask_in [[buffer(2)]], device half* mask_out [[buffer(3)]],
                           constant TsmQsaArgs& a [[buffer(4)]], device int* idx [[buffer(5)]], device int* cnt [[buffer(6)]],
                           uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint shared_u[4];       // 0 eligible, 1 prefix, 2 remaining
    threadgroup uint sg_ties[kQsaThreads / 32];
    threadgroup uint sg_sel[kQsaThreads / 32];
    const uint t = tg.x, n = a.n_cells;
    device const float* srow = S + ulong(t) * a.n_blocks;
    device const half* min_row = mask_in + ulong(t) * n;
    device half* mout_row = mask_out + ulong(t) * n;

    // Eligible (finite) cells.
    if (tid == 0) atomic_store_explicit(&hist[0], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint mine = 0;
    for (uint c = tid; c < n; c += kQsaThreads) mine += isfinite(tsm_qsa_value(srow, cell_blocks, min_row, c)) ? 1u : 0u;
    mine = simd_sum(mine);
    if (lane == 0) atomic_fetch_add_explicit(&hist[0], mine, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint eligible = atomic_load_explicit(&hist[0], memory_order_relaxed);
    const uint take = min(a.width, eligible);
    // Every allowed cell is taken (a window no wider than `width`): no select.
    const bool all = take == eligible;

    // The take-th largest key, 8 bits a pass from the top.
    if (tid == 0) { shared_u[1] = 0u; shared_u[2] = take; }
    for (uint p = 0; p < 4 && !all; ++p) {
        const uint shift = 24u - 8u * p;
        const uint hi_mask = p == 0 ? 0u : (0xFFFFFFFFu << (32u - 8u * p));
        threadgroup_barrier(mem_flags::mem_threadgroup);
        atomic_store_explicit(&hist[tid], 0u, memory_order_relaxed);   // kQsaThreads == 256 bins
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const uint prefix = shared_u[1];
        for (uint c = tid; c < n; c += kQsaThreads) {
            const float e = tsm_qsa_value(srow, cell_blocks, min_row, c);
            if (!isfinite(e)) continue;
            const uint k = tsm_qsa_key(e);
            if ((k & hi_mask) == prefix) atomic_fetch_add_explicit(&hist[(k >> shift) & 255u], 1u, memory_order_relaxed);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid == 0) {
            uint remaining = shared_u[2], acc = 0;
            for (int d = 255; d >= 0; --d) {
                const uint h = atomic_load_explicit(&hist[d], memory_order_relaxed);
                if (acc + h >= remaining) {
                    shared_u[1] = prefix | (uint(d) << shift);
                    shared_u[2] = remaining - acc;
                    break;
                }
                acc += h;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint cutoff = shared_u[1], ties_taken = shared_u[2];

    // Every key above the cutoff, and the first ties_taken cells at it in cell order; the chosen cells' positions in
    // the ascending list by a second prefix sum.
    uint base = 0, sel_base = 0;
    device int* irow = idx + ulong(t) * a.width;
    for (uint c0 = 0; c0 < n; c0 += kQsaThreads) {
        const uint c = c0 + tid;
        uint k = 0;
        bool fin = false;
        if (c < n) {
            const float e = tsm_qsa_value(srow, cell_blocks, min_row, c);
            fin = isfinite(e);
            if (fin) k = tsm_qsa_key(e);
        }
        const uint tie = (!all && fin && k == cutoff) ? 1u : 0u;
        const uint before_in_sg = simd_prefix_exclusive_sum(tie);
        const uint sg_total = simd_sum(tie);
        if (lane == 0) sg_ties[sg] = sg_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before = base + before_in_sg, tile_total = 0;
        for (uint s = 0; s < kQsaThreads / 32; ++s) {
            if (s < sg) before += sg_ties[s];
            tile_total += sg_ties[s];
        }
        const bool sel = c < n && fin && (all || k > cutoff || (tie != 0u && before < ties_taken));
        if (a.write_mask && c < n) mout_row[c] = sel ? min_row[c] : half(-INFINITY);
        if (a.write_idx) {
            const uint s1 = sel ? 1u : 0u;
            const uint sel_before_in_sg = simd_prefix_exclusive_sum(s1);
            const uint sel_sg_total = simd_sum(s1);
            if (lane == 0) sg_sel[sg] = sel_sg_total;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            uint pos = sel_base + sel_before_in_sg, sel_tile = 0;
            for (uint s = 0; s < kQsaThreads / 32; ++s) {
                if (s < sg) pos += sg_sel[s];
                sel_tile += sg_sel[s];
            }
            if (sel && pos < a.width) irow[pos] = int(c);
            sel_base += sel_tile;
        }
        base += tile_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (a.write_idx && tid == 0) cnt[t] = int(min(sel_base, a.width));
}

// ---- tsm_qsa_attn: a QSA layer's attention for a prompt chunk over each token's selected cells only (the list
// tsm_qsa_select writes), instead of every cell of the window. One threadgroup per (token, kv head), 4 simdgroups: the
// kv head's query heads are the rows of the products (up to 16; Flash Next has 12), the cells come 32 at a time,
// gathered from the cache into threadgroup memory:
//   S[16 x 32] = Q . K^T (half operands, float accumulation), s = S * scale + mask, online softmax in float,
//   O[16 x 256] = diag(alpha) . O + P . V (P half), O in each simdgroup's registers (64 columns each)
// q [D][T][H] (ggml's [D, T, H], F32), the cache [D, cells, kv heads] (F16, strides in elements), idx [T][W] and cnt [T],
// the mask [T][cells] (F16), out [T][H][D] (ggml's flash-attention layout [D, H, T]).
struct TsmQsaAttnArgs {
    uint n_tok;
    uint n_head;
    uint n_head_kv;
    uint n_cells;        // the mask's row length
    uint width;          // idx's row length
    uint k_row, k_head;  // K's strides in elements (cell, kv head)
    uint v_row, v_head;
    float scale;
    uint pad0, pad1;
};

constant constexpr short kAttnD = 256, kAttnC = 32, kAttnR = 16;

// The queries as the attention kernels read them: half, per (token, kv head) 16 rows (its query heads, then zeros),
// qh[t][g][r][d]. grid (16, T, kv heads), 256 threads (D = 256).
kernel void tsm_qsa_q_half(device const float* q [[buffer(0)]], device half* qh [[buffer(1)]],
                           constant TsmQsaAttnArgs& a [[buffer(2)]], uint3 tg [[threadgroup_position_in_grid]],
                           ushort tid [[thread_index_in_threadgroup]]) {
    const uint r = tg.x, t = tg.y, g = tg.z, T = a.n_tok, hpg = a.n_head / a.n_head_kv;
    qh[((ulong(t) * a.n_head_kv + g) * 16 + r) * 256 + tid] =
        r < hpg ? half(q[(ulong(g * hpg + r) * T + t) * 256 + tid]) : half(0.0f);
}

kernel void tsm_qsa_attn(device const half* qd [[buffer(0)]], device const half* kc [[buffer(1)]],
                         device const half* vc [[buffer(2)]], device const int* idx [[buffer(3)]],
                         device const int* cnt [[buffer(4)]], device const half* mask [[buffer(5)]],
                         device float* out [[buffer(6)]], constant TsmQsaAttnArgs& a [[buffer(7)]],
                         uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                         ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup half kv[kAttnC * kAttnD];          // 16 KB: K, then V, then O's rows at the end (as float)
    threadgroup float st[kAttnR * kAttnC];         //  2 KB: S
    threadgroup half pt[kAttnR * kAttnC];          //  1 KB: P
    threadgroup float dg[2 * 64];                  //  diag(alpha) for the two row tiles
    threadgroup int cells[kAttnC];
    threadgroup float lsum[kAttnR];
    threadgroup int rescale;

    const uint t = tg.x, g = tg.y, H = a.n_head, hpg = H / a.n_head_kv, T = a.n_tok;
    const int n_sel = cnt[t];

    // Q tiles straight from the half copy (L1-resident across the chunks): the (token, kv head)'s 16 rows.
    device const half* qrow = qd + (ulong(t) * a.n_head_kv + g) * kAttnR * kAttnD;
    const ulong qstride = kAttnD;
    for (uint i = tid; i < 2 * 64; i += 128) dg[i] = 0.0f;

    // Each simdgroup owns O columns [64 sg, 64 sg + 64): 2 row tiles x 8 column tiles.
    simdgroup_float8x8 o[2][8];
    for (short r = 0; r < 2; ++r)
        for (short j = 0; j < 8; ++j) o[r][j] = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);

    // Softmax lanes: row = tid / 8 (this simdgroup's rows 4 sg .. 4 sg + 3), columns (tid % 8) * 4 .. + 3.
    const ushort srow = tid >> 3, scol = (tid & 7) * 4;
    const bool row_valid = srow < hpg;
    float m_run = -INFINITY, l_run = 0.0f;

    device const int* row_idx = idx + ulong(t) * a.width;
    device const half* mrow = mask + ulong(t) * a.n_cells;
    device const half* kbase = kc + ulong(g) * a.k_head;
    device const half* vbase = vc + ulong(g) * a.v_head;

    for (int c0 = 0; c0 < n_sel; c0 += kAttnC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < kAttnC) cells[tid] = c0 + int(tid) < n_sel ? row_idx[c0 + tid] : -1;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // K rows: 4 threads a row, 64 halves each (cells past the list are zero).
        {
            const ushort r = tid >> 2, part = (tid & 3) * 64;
            const int cell = cells[r];
            threadgroup half4* dst = (threadgroup half4*)(kv + r * kAttnD + part);
            if (cell >= 0) {
                device const half4* src = (device const half4*)(kbase + ulong(cell) * a.k_row + part);
                for (short i = 0; i < 16; ++i) dst[i] = src[i];
            } else {
                for (short i = 0; i < 16; ++i) dst[i] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // S: simdgroup sg computes cell tile sg for both row tiles.
        {
            simdgroup_float8x8 s0 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            simdgroup_float8x8 s1 = make_filled_simdgroup_matrix<float, 8, 8>(0.0f);
            for (short d0 = 0; d0 < kAttnD; d0 += 8) {
                simdgroup_half8x8 kt, q0, q1;
                simdgroup_load(kt, kv + (sg * 8) * kAttnD + d0, kAttnD, ulong2(0, 0), true);
                simdgroup_load(q0, qrow + d0, qstride);
                simdgroup_load(q1, qrow + 8 * qstride + d0, qstride);
                simdgroup_multiply_accumulate(s0, q0, kt, s0);
                simdgroup_multiply_accumulate(s1, q1, kt, s1);
            }
            simdgroup_store(s0, st + 0 * kAttnC + sg * 8, kAttnC);
            simdgroup_store(s1, st + 8 * kAttnC + sg * 8, kAttnC);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // Online softmax over this chunk's 32 cells. The reference max moves only when a row's max passes it by more
        // than 8 (P = exp(s - m) <= e^8 stays in half's range): most chunks then leave O as it is (no rescale).
        if (tid == 0) rescale = 0;
        {
            float x[4];
            float mx = -INFINITY;
            for (short j = 0; j < 4; ++j) {
                const int cell = cells[scol + j];
                float v = -INFINITY;
                if (row_valid && cell >= 0) v = st[srow * kAttnC + scol + j] * a.scale + float(mrow[cell]);
                x[j] = v;
                mx = max(mx, v);
            }
            mx = max(mx, simd_shuffle_xor(mx, 4));
            mx = max(mx, simd_shuffle_xor(mx, 2));
            mx = max(mx, simd_shuffle_xor(mx, 1));
            const bool move = mx > m_run + 8.0f;
            const float m_new = move ? mx : m_run;
            const float alpha = move ? exp(m_run - m_new) : 1.0f;   // m_run = -inf: 0 (O is still zero)
            float ps = 0.0f;
            for (short j = 0; j < 4; ++j) {
                const float p = m_new == -INFINITY ? 0.0f : exp(x[j] - m_new);
                pt[srow * kAttnC + scol + j] = half(p);
                ps += p;
            }
            ps += simd_shuffle_xor(ps, 4);
            ps += simd_shuffle_xor(ps, 2);
            ps += simd_shuffle_xor(ps, 1);
            l_run = l_run * alpha + ps;
            m_run = m_new;
            if ((tid & 7) == 0) {
                dg[(srow / 8) * 64 + (srow % 8) * 9] = alpha;   // diagonal entry (srow % 8, srow % 8)
                if (move) rescale = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // V rows into the same buffer.
        {
            const ushort r = tid >> 2, part = (tid & 3) * 64;
            const int cell = cells[r];
            threadgroup half4* dst = (threadgroup half4*)(kv + r * kAttnD + part);
            if (cell >= 0) {
                device const half4* src = (device const half4*)(vbase + ulong(cell) * a.v_row + part);
                for (short i = 0; i < 16; ++i) dst[i] = src[i];
            } else {
                for (short i = 0; i < 16; ++i) dst[i] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // O = diag(alpha) . O + P . V over this simdgroup's 64 columns.
        {
            simdgroup_float8x8 d0m, d1m;
            simdgroup_load(d0m, dg + 0, 8);
            simdgroup_load(d1m, dg + 64, 8);
            simdgroup_half8x8 p0[4], p1[4];
            for (short k = 0; k < 4; ++k) {
                simdgroup_load(p0[k], pt + 0 * kAttnC + k * 8, kAttnC);
                simdgroup_load(p1[k], pt + 8 * kAttnC + k * 8, kAttnC);
            }
            const bool scale_o = rescale != 0;
            for (short j = 0; j < 8; ++j) {
                const short col = sg * 64 + j * 8;
                simdgroup_float8x8 a0 = o[0][j], a1 = o[1][j];
                if (scale_o) {
                    simdgroup_multiply(a0, d0m, o[0][j]);
                    simdgroup_multiply(a1, d1m, o[1][j]);
                }
                for (short k = 0; k < 4; ++k) {
                    simdgroup_half8x8 vt;
                    simdgroup_load(vt, kv + (k * 8) * kAttnD + col, kAttnD);
                    simdgroup_multiply_accumulate(a0, p0[k], vt, a0);
                    simdgroup_multiply_accumulate(a1, p1[k], vt, a1);
                }
                o[0][j] = a0;
                o[1][j] = a1;
            }
        }
    }
    // O / l into out[t][head][d].
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 7) == 0) lsum[srow] = l_run;
    threadgroup float* of = (threadgroup float*)kv;   // 16 x 256 floats = 16 KB
    for (short j = 0; j < 8; ++j) {
        simdgroup_store(o[0][j], of + 0 * kAttnD + sg * 64 + j * 8, kAttnD);
        simdgroup_store(o[1][j], of + 8 * kAttnD + sg * 64 + j * 8, kAttnD);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = tid; i < hpg * kAttnD; i += 128) {
        const uint r = i / kAttnD, d = i % kAttnD;
        const float l = lsum[r];
        out[(ulong(t) * H + g * hpg + r) * kAttnD + d] = l > 0.0f ? of[r * kAttnD + d] / l : 0.0f;
    }
}
