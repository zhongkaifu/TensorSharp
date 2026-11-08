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
    uint pad0, pad1, pad2;
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
                           constant TsmQsaArgs& a [[buffer(4)]],
                           uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]],
                           ushort sg [[simdgroup_index_in_threadgroup]], ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup atomic_uint hist[256];
    threadgroup uint shared_u[4];       // 0 eligible, 1 prefix, 2 remaining
    threadgroup uint sg_ties[kQsaThreads / 32];
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
    if (take == eligible) {
        // Every allowed cell is taken: the mask is unchanged.
        for (uint c = tid; c < n; c += kQsaThreads) mout_row[c] = min_row[c];
        return;
    }

    // The take-th largest key, 8 bits a pass from the top.
    if (tid == 0) { shared_u[1] = 0u; shared_u[2] = take; }
    for (uint p = 0; p < 4; ++p) {
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

    // Every key above the cutoff, and the first ties_taken cells at it in cell order.
    uint base = 0;
    for (uint c0 = 0; c0 < n; c0 += kQsaThreads) {
        const uint c = c0 + tid;
        uint k = 0;
        bool fin = false;
        if (c < n) {
            const float e = tsm_qsa_value(srow, cell_blocks, min_row, c);
            fin = isfinite(e);
            if (fin) k = tsm_qsa_key(e);
        }
        const uint tie = (fin && k == cutoff) ? 1u : 0u;
        const uint before_in_sg = simd_prefix_exclusive_sum(tie);
        const uint sg_total = simd_sum(tie);
        if (lane == 0) sg_ties[sg] = sg_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint before = base + before_in_sg, tile_total = 0;
        for (uint s = 0; s < kQsaThreads / 32; ++s) {
            if (s < sg) before += sg_ties[s];
            tile_total += sg_ties[s];
        }
        if (c < n) {
            const bool sel = fin && (k > cutoff || (tie != 0u && before < ties_taken));
            mout_row[c] = sel ? min_row[c] : half(-INFINITY);
        }
        base += tile_total;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
