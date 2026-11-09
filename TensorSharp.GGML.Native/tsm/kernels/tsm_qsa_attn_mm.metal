// A QSA layer's attention for a prompt chunk over each token's chosen cells only, on the Metal 4 tensor API
// (MetalPerformancePrimitives matmul2d: the M5's matrix units), compiled as its own library (Metal language 4.0) so a
// device or SDK without it keeps every other tsm kernel. The structure follows ggml's tensor flash attention
// (ggml-metal kernels/fa_tensor.metal: Q.K^T and P.V as matmul2d over simdgroups, the online softmax with lazy
// rescaling, O in a cooperative tensor scaled per row) with the keys and values of the chosen cells gathered 32 at a
// time into threadgroup memory - a QSA row's cells are not contiguous. The kv head's query heads are the rows (Strata's
// qsa_attn_mm arrangement, metal/kernels/qsa_prompt.metal, MIT).
//
// qd: the queries in half, [t][kv head][16][D] (tsm_qsa_q_half: the kv head's query heads, then zero rows); the caches
// F16 [D, cells, kv heads] (strides in elements); idx [t][width] and cnt [t] (tsm_qsa_select); the mask [t][cells];
// out [t][H][D] (ggml's flash-attention layout [D, H, T]).

struct TsmQsaAttnArgs {
    uint n_tok;
    uint n_head;
    uint n_head_kv;
    uint n_cells;
    uint width;
    uint k_row, k_head;
    uint v_row, v_head;
    float scale;
    uint pad0, pad1;
};

// grid (T, kv heads), 128 threads (4 simdgroups)
kernel void tsm_qsa_attn_mm(device const half* qd [[buffer(0)]], device const half* kc [[buffer(1)]],
                            device const half* vc [[buffer(2)]], device const int* idx [[buffer(3)]],
                            device const int* cnt [[buffer(4)]], device const half* mask [[buffer(5)]],
                            device float* out [[buffer(6)]], constant TsmQsaAttnArgs& a [[buffer(7)]],
                            uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    constexpr int D = 256, R = 16, C = 32, NSG = 4, NT = 32 * NSG;
    threadgroup half skv[C * D];     // 16 KB: the chunk's keys, then its values
    threadgroup float ss[R * C];     //  2 KB: S
    threadgroup half sp[R * C];      //  1 KB: P
    threadgroup float sr[R];         //  per row: the rescale, then 1 / l
    threadgroup int cells[C];
    threadgroup int rescale;

    const uint t = tg.x, g = tg.y, H = a.n_head, hpg = H / a.n_head_kv;
    const int n_sel = cnt[t];
    device const int* row_idx = idx + ulong(t) * a.width;
    device const half* mrow = mask + ulong(t) * a.n_cells;
    device const half* kbase = kc + ulong(g) * a.k_head;
    device const half* vbase = vc + ulong(g) * a.v_head;

    auto tq = tensor((device half*)(qd + (ulong(t) * a.n_head_kv + g) * R * D), dextents<int32_t, 2>(D, R));
    auto tk = tensor(skv, dextents<int32_t, 2>(D, C));
    auto tv = tensor(skv, dextents<int32_t, 2>(D, C));
    auto ts = tensor(ss, dextents<int32_t, 2>(C, R));
    auto tp = tensor(sp, dextents<int32_t, 2>(C, R));

    matmul2d<matmul2d_descriptor(R, C, D, false, true, false, matmul2d_descriptor::mode::multiply),
             execution_simdgroups<NSG>> mm_qk;
    matmul2d<matmul2d_descriptor(R, D, C, false, false, false, matmul2d_descriptor::mode::multiply_accumulate),
             execution_simdgroups<NSG>> mm_pv;
    auto co = mm_pv.template get_destination_cooperative_tensor<decltype(tp), decltype(tv), float>();
    for (short i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) co[i] = 0.0f;

    // Softmax lanes: row = tid / 8, columns (tid % 8) * 4 .. + 3 (8 lanes a row, inside one simdgroup).
    const ushort srow = tid >> 3, scol = (tid & 7) * 4;
    const bool row_valid = srow < hpg;
    float m_run = -INFINITY, l_run = 0.0f;

    for (int c0 = 0; c0 < n_sel; c0 += C) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tid < C) cells[tid] = c0 + int(tid) < n_sel ? row_idx[c0 + tid] : -1;
        if (tid == 0) rescale = 0;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // K rows: 4 threads a row, 64 halves each (cells past the list are zero)
        {
            const ushort r = tid >> 2, part = (tid & 3) * 64;
            const int cell = cells[r];
            threadgroup half4* dst = (threadgroup half4*)(skv + r * D + part);
            if (cell >= 0) {
                device const half4* src = (device const half4*)(kbase + ulong(cell) * a.k_row + part);
                for (short i = 0; i < 16; ++i) dst[i] = src[i];
            } else {
                for (short i = 0; i < 16; ++i) dst[i] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        mm_qk.run(tq, tk, ts);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // online softmax, the running max moved only when it grows by more than 8 (e^8 fits in half)
        {
            float x[4];
            float mx = -INFINITY;
            for (short j = 0; j < 4; ++j) {
                const int cell = cells[scol + j];
                float v = -INFINITY;
                if (row_valid && cell >= 0) v = ss[srow * C + scol + j] * a.scale + float(mrow[cell]);
                x[j] = v;
                mx = max(mx, v);
            }
            mx = max(mx, simd_shuffle_xor(mx, 4));
            mx = max(mx, simd_shuffle_xor(mx, 2));
            mx = max(mx, simd_shuffle_xor(mx, 1));
            const bool move = mx > m_run + 8.0f;
            const float m_new = move ? mx : m_run;
            const float alpha = move ? exp(m_run - m_new) : 1.0f;
            float ps = 0.0f;
            for (short j = 0; j < 4; ++j) {
                const half p = half(m_new == -INFINITY ? 0.0f : exp(x[j] - m_new));
                sp[srow * C + scol + j] = p;
                ps += float(p);   // the sum of the rounded values P.V uses
            }
            ps += simd_shuffle_xor(ps, 4);
            ps += simd_shuffle_xor(ps, 2);
            ps += simd_shuffle_xor(ps, 1);
            l_run = l_run * alpha + ps;
            m_run = m_new;
            if ((tid & 7) == 0) {
                sr[srow] = alpha;
                if (move) rescale = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        // V rows into the same buffer (the keys are consumed)
        {
            const ushort r = tid >> 2, part = (tid & 3) * 64;
            const int cell = cells[r];
            threadgroup half4* dst = (threadgroup half4*)(skv + r * D + part);
            if (cell >= 0) {
                device const half4* src = (device const half4*)(vbase + ulong(cell) * a.v_row + part);
                for (short i = 0; i < 16; ++i) dst[i] = src[i];
            } else {
                for (short i = 0; i < 16; ++i) dst[i] = half4(0.0h);
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (rescale != 0)
            for (short i = 0; i < co.get_capacity(); ++i)
                if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
        mm_pv.run(tp, tv, co);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if ((tid & 7) == 0) sr[srow] = l_run > 0.0f ? 1.0f / l_run : 0.0f;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (short i = 0; i < co.get_capacity(); ++i)
        if (co.is_valid_element(i)) co[i] *= sr[co.get_multidimensional_index(i)[1]];
    // rows past the kv head's query heads are clipped by the extents
    device float* pdst = out + (ulong(t) * H + g * hpg) * D;
    auto td = tensor(pdst, dextents<int32_t, 2>(D, int(hpg)), array<int, 2>({1, D}));
    co.store(td);
}
