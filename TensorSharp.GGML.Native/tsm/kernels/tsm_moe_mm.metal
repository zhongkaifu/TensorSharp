// Ported from Strata (https://github.com/Niko1221/Strata) metal/kernels/moe_mm.metal at 146df3c4.
// MIT License, Copyright (c) 2026 Niko1221 and the Strata contributors; see ../LICENSE-STRATA.
//
// Routed experts for a PROMPT chunk as a grouped GEMM on the matrix units (Metal 4's MetalPerformancePrimitives
// matmul2d; on the M5 its neural accelerators). The chunk's T x topk (token, expert) entries are sorted by expert;
// each expert's run is padded to TM rows; a threadgroup takes one tile (TM rows of one expert) and TN output rows.
//   mm_hist / mm_scan / mm_place  the plan (counts per block of 1024 entries, padded offsets and the tile list, rows)
//   mm_gather                     XS[row] = half(x[token of row]), 0 on padding rows
//   mm_gu3                        HS[row][f] = silu(x . gate_f) * (x . up_f), fp32 accumulation, stored as half
//   mm_down3                      YE[row][d] = h . down_d, fp32
//   mm_combine                    y[t][d] = sum_k w(t, k) * YE[pos(t, k)][d], k in order
// Changes from upstream (TensorSharp): only these kernels are kept (the plan, gather, the v3 GEMMs and combine; not
// mm_gu/mm_down/mm_gu2/mm_down2/mm_gu4..6/mm_scan6), only the Q2_0 decode (W4<T_Q2_0> and the generic W8) and only
// the Q2_0 instantiations at upstream's chosen tiles (gate/up 16x32x32 on 4 simdgroups, down 16x64x32 on 2); the
// kernel bodies are unchanged. Compiled after quant.metal (fmt, load_tab) in TensorSharp's tensor-API library.

struct MmPlanArgs {
    uint n_ent;      // T * topk
    uint topk;
    uint n_expert;
    uint tm;         // the tile height (rows of one expert)
    uint n_blocks;   // ceil(n_ent / 1024)
};

// counts per block: bcnt[block * n_expert + e]
kernel void mm_hist(device const int* ids [[buffer(0)]], device uint* bcnt [[buffer(1)]],
                    constant MmPlanArgs& a [[buffer(2)]],
                    uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup atomic_uint h[1024];
    for (uint e = tid; e < a.n_expert; e += 1024) atomic_store_explicit(&h[e], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint i = tg.x * 1024 + tid;
    if (i < a.n_ent) atomic_fetch_add_explicit(&h[ids[i]], 1u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint e = tid; e < a.n_expert; e += 1024)
        bcnt[tg.x * a.n_expert + e] = atomic_load_explicit(&h[e], memory_order_relaxed);
}

// one threadgroup of n_expert (<= 1024) threads.  boff[block][e] = expert e's rows before this block (within e);
// eoff[e] = expert e's first padded row; tiles[i] = (expert, row0); meta[0] = n_tiles, meta[1] = padded rows;
// indirect grids: meta[2..4] = (gx_gu, n_tiles, 1) for the gate/up GEMM, meta[5..7] = (gx_down, n_tiles, 1) for the
// down GEMM (gx_gu in meta[2], gx_down in meta[5] written by the host once), meta[8..10] = (rows, 1, 1) for mm_gather.
kernel void mm_scan(device const uint* bcnt [[buffer(0)]], device uint* boff [[buffer(1)]],
                    device uint* eoff [[buffer(2)]], device int2* tiles [[buffer(3)]], device uint* meta [[buffer(4)]],
                    constant MmPlanArgs& a [[buffer(5)]],
                    ushort tid [[thread_index_in_threadgroup]], ushort sg [[simdgroup_index_in_threadgroup]],
                    ushort lane [[thread_index_in_simdgroup]]) {
    threadgroup uint s_rows[32], s_tiles[32];
    // simdgroups past the launched ones must count 0 (512 threads fill only 16 of the 32 slots)
    if (tid < 32) { s_rows[tid] = 0; s_tiles[tid] = 0; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint e = tid;
    uint tot = 0;
    if (e < a.n_expert)
        for (uint b = 0; b < a.n_blocks; ++b) {
            boff[b * a.n_expert + e] = tot;
            tot += bcnt[b * a.n_expert + e];
        }
    const uint nt = (tot + a.tm - 1) / a.tm, rows = nt * a.tm;
    const uint pr = simd_prefix_exclusive_sum(rows), pt = simd_prefix_exclusive_sum(nt);
    if (lane == 31) { s_rows[sg] = pr + rows; s_tiles[sg] = pt + nt; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint orow = 0, otile = 0, all_rows = 0, all_tiles = 0;
    for (int i = 0; i < 32; ++i) {
        if (i < int(sg)) { orow += s_rows[i]; otile += s_tiles[i]; }
        all_rows += s_rows[i];
        all_tiles += s_tiles[i];
    }
    if (e < a.n_expert) {
        const uint r0 = orow + pr, t0 = otile + pt;
        eoff[e] = r0;
        for (uint k = 0; k < nt; ++k) tiles[t0 + k] = int2(int(e), int(r0 + k * a.tm));
    }
    if (tid == 0) {
        meta[0] = all_tiles;
        meta[1] = all_rows;
        meta[3] = all_tiles;   // indirect grid of the gate/up GEMM: (meta[2] given, tiles, 1)
        meta[4] = 1;
        meta[6] = all_tiles;   // of the down GEMM: (meta[5] given, tiles, 1)
        meta[7] = 1;
        meta[8] = all_rows;    // of the gather: (rows, 1, 1)
        meta[9] = 1;
        meta[10] = 1;
    }
}

// grid (n_blocks), 1024 threads.  srt_* must be cleared to the padding value (-1 / 0) beforehand by mm_clear.
kernel void mm_place(device const int* ids [[buffer(0)]], device const float* w [[buffer(1)]],
                     device const uint* boff [[buffer(2)]], device const uint* eoff [[buffer(3)]],
                     device int* srt_tok [[buffer(4)]], device float* srt_w [[buffer(5)]], device int* pos [[buffer(6)]],
                     constant MmPlanArgs& a [[buffer(7)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup int s_id[1024];
    const uint i = tg.x * 1024 + tid;
    const int e = i < a.n_ent ? ids[i] : -1;
    s_id[tid] = e;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (i >= a.n_ent) return;
    uint rank = 0;
    for (uint j = 0; j < tid; ++j) rank += s_id[j] == e;
    const uint row = eoff[e] + boff[tg.x * a.n_expert + e] + rank;
    srt_tok[row] = int(i / a.topk);
    srt_w[row] = w[i];
    pos[i] = int(row);
}

kernel void mm_clear(device int* srt_tok [[buffer(0)]], constant uint& n [[buffer(1)]], uint gid [[thread_position_in_grid]]) {
    if (gid < n) srt_tok[gid] = -1;
}

// weights of one expert: gate at W + gate_off + e * gu_stride (+ up_off for up), down at W + down_off + e * d_stride
struct MmArgs {
    ulong gate_off;
    ulong up_off;
    ulong down_off;
    ulong gu_stride;
    ulong d_stride;
    uint gu_row;     // bytes per gate/up row (n_embd values)
    uint d_row;      // bytes per down row (n_ff values)
    uint n_embd;
    uint n_ff;
};

// grid (T * n_embd / 256), 256 threads
kernel void mm_combine(device const float* YE [[buffer(0)]], device const int* pos [[buffer(1)]],
                       device const float* srt_w [[buffer(2)]], device float* y [[buffer(3)]],
                       constant uint4& a [[buffer(4)]],   // (T, n_embd, topk, -)
                       uint gid [[thread_position_in_grid]]) {
    const uint t = gid / a.y, d = gid % a.y;
    if (t >= a.x) return;
    float s = 0.0f;
    for (uint k = 0; k < a.z; ++k) {
        const int r = pos[t * a.z + k];
        s = fma(srt_w[r], YE[ulong(r) * a.y + d], s);
    }
    y[gid] = s;
}

// ---- v2: activations gathered once per layer as fp16 rows, read by matmul2d from device memory ---------------------
// mm_gather: XS[row][k] = half(x[srt_tok[row]][k]) (0 on padding rows).  grid (rows), 256 threads.
kernel void mm_gather(device const float* x [[buffer(0)]], device const int* srt_tok [[buffer(1)]],
                      device half* XS [[buffer(2)]], constant uint& n [[buffer(3)]],
                      uint3 tg [[threadgroup_position_in_grid]], ushort tid [[thread_index_in_threadgroup]]) {
    const int t = srt_tok[tg.x];
    device half* o = XS + ulong(tg.x) * n;
    for (uint k = tid; k < n; k += 256) o[k] = t >= 0 ? half(x[ulong(t) * n + k]) : half(0.0f);
}

// The SSD-streaming arena (as in moe.metal): with function constant 20, expert e's slot is smap[e] in the chunks W,
// W1..W3 (buffers 5..8 of mm_gu2 / mm_down2 / mm_gu3 / mm_down3); gate_off / up_off / down_off are offsets in the slot.
#ifndef STRATA_ARENA_METAL
#define STRATA_ARENA_METAL
constant bool kArenaFc [[function_constant(20)]];
constant bool kArena = is_function_constant_defined(kArenaFc) && kArenaFc;
static inline device const uchar* arena_slot(uint v, device const uchar* W, device const uchar* W1, device const uchar* W2,
                                             device const uchar* W3) {
    const uint c = v >> 28;
    device const uchar* b = c == 0 ? W : (c == 1 ? W1 : (c == 2 ? W2 : W3));
    return b + (ulong)(v & 0x0FFFFFFFu) * 16384ul;
}
#endif

// a probe for the bench (function constant 5): 1 = the weights not decoded (a cheap fill), 2 = no matrix multiply
constant int MM_PROBE [[function_constant(5)]];
constant int mm_probe = is_function_constant_defined(MM_PROBE) ? MM_PROBE : 0;

#ifndef STRATA_COOP_K4
#define STRATA_COOP_K4
// function constant 7: the right-input cooperative tensor holds runs of 4 consecutive k at one n per lane (the M5's
// layout; tests/coop_layout).  false (the M4: runs of 2 n at one k): every element decoded on its own.
constant bool COOP_K4 [[function_constant(7)]];
constant bool coop_k4 = is_function_constant_defined(COOP_K4) ? COOP_K4 : true;
#endif
template <int F> struct W4;
// the four weights k .. k + 3 (k % 4 == 0) of a row, in quant.metal's deq arithmetic (code * scale in fp32), as fp16
// the threadgroup forms of quant.metal's field readers (blocks staged in threadgroup memory)
static inline uint rd32(threadgroup const uchar* p) {
    threadgroup const ushort* s = (threadgroup const ushort*)p;
    return uint(s[0]) | (uint(s[1]) << 16);
}
static inline float rdh(threadgroup const uchar* p) { return float(*(threadgroup const half*)p); }

static inline half4 w4_signed(uint g, uint s, float sc) {
    const uchar4 v = as_type<uchar4>(g);
    const float4 q = float4((s & 1) ? -float(v[0]) : float(v[0]), (s & 2) ? -float(v[1]) : float(v[1]),
                            (s & 4) ? -float(v[2]) : float(v[2]), (s & 8) ? -float(v[3]) : float(v[3]));
    return half4(q * sc);
}
template <> struct W4<T_Q2_0> {   // 64 values a block: fp16 d, 16 bytes of 2-bit codes (value v at byte v / 4)
    // fp32 math then fp16 (an fp16-math decode was measured slower: 4.2 vs 3.5 ms a layer)
    static inline half4 run(device const uchar* row, int k, threadgroup const uint*) {
        device const uchar* b = row + (k >> 6) * 18;
        const float d = float(*(device const half*)b);
        const uint q = b[2 + ((k & 63) >> 2)];
        return half4(float4(float(int(q & 3) - 1), float(int((q >> 2) & 3) - 1), float(int((q >> 4) & 3) - 1),
                            float(int(q >> 6) - 1)) * d);
    }
};

template <int F> struct W8 {
    static inline void run(device const uchar* row, int k, threadgroup const uint* tab, thread half4& a, thread half4& b) {
        a = W4<F>::run(row, k, tab);
        b = W4<F>::run(row, k + 16, tab);
    }
};

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_gu3(device const uchar* W [[buffer(0)]], device const half* XS [[buffer(1)]],
                   device half* HS [[buffer(2)]], device const int2* tiles [[buffer(3)]], constant MmArgs& a [[buffer(4)]], device const uint* smap [[buffer(5)]],
                     device const uchar* W1 [[buffer(6)]], device const uchar* W2 [[buffer(7)]], device const uchar* W3 [[buffer(8)]],
                   uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                   ushort lane [[thread_index_in_simdgroup]], ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup float ot[SG][TM * TN];
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* E = kArena ? arena_slot(smap[e], W, W1, W2, W3) : W + ulong(e) * a.gu_stride;
    device const uchar* g0 = E + a.gate_off + ulong(n0) * a.gu_row;
    device const uchar* u0 = E + a.up_off + ulong(n0) * a.gu_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tX((device half*)XS, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    auto rg = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto ru = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tX.template slice<TK, TM>(0, row0);
    auto accg = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    auto accu = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rg)>, float>();
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) { accg[i] = 0.0f; accu[i] = 0.0f; }
    for (int k0 = 0; k0 < int(a.n_embd); k0 += TK) {
        const uint16_t cap = rg.get_capacity(), half_cap = cap / 2;
        if (!coop_k4) {
            for (uint16_t i = 0; i < cap; ++i) {
                const auto ix = rg.get_multidimensional_index(i);
                const int k = k0 + ix[0], n = ix[1];
                rg[i] = W4<F>::run(g0 + ulong(n) * a.gu_row, k & ~3, tab)[k & 3];
                ru[i] = W4<F>::run(u0 + ulong(n) * a.gu_row, k & ~3, tab)[k & 3];
            }
        } else
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < half_cap; i += 4) {
            const auto ix = rg.get_multidimensional_index(i);
            const auto iy = rg.get_multidimensional_index(i + half_cap);
            const int k = k0 + ix[0], n = ix[1];
            device const uchar* rgw = g0 + ulong(n) * a.gu_row;
            device const uchar* ruw = u0 + ulong(n) * a.gu_row;
            half4 wg, wu, wg2, wu2;
            if (mm_probe == 1) {   // the bench probe: no decode (constant weights)
                wg = half4(half(n & 7)); wu = wg; wg2 = half4(half(k & 7)); wu2 = wg2;
            } else if (TK == 32) {
                // the lane's two runs of this row sit in one 32-group: element i + cap / 2 is (k + 16, n) - part of
                // the COOP_K4 layout (coop_probe checks it); a runtime check of it here cost 30%
                W8<F>::run(rgw, k, tab, wg, wg2);
                W8<F>::run(ruw, k, tab, wu, wu2);
            } else {
                const int k2 = k0 + iy[0];
                device const uchar* rgw2 = g0 + ulong(iy[1]) * a.gu_row;
                device const uchar* ruw2 = u0 + ulong(iy[1]) * a.gu_row;
                wg = W4<F>::run(rgw, k, tab); wu = W4<F>::run(ruw, k, tab);
                wg2 = W4<F>::run(rgw2, k2, tab); wu2 = W4<F>::run(ruw2, k2, tab);
            }
            rg[i] = wg.x; rg[i + 1] = wg.y; rg[i + 2] = wg.z; rg[i + 3] = wg.w;
            ru[i] = wu.x; ru[i + 1] = wu.y; ru[i + 2] = wu.z; ru[i + 3] = wu.w;
            const uint16_t j = i + half_cap;
            rg[j] = wg2.x; rg[j + 1] = wg2.y; rg[j + 2] = wg2.z; rg[j + 3] = wg2.w;
            ru[j] = wu2.x; ru[j + 1] = wu2.y; ru[j + 2] = wu2.z; ru[j + 3] = wu2.w;
        }
        tA = tX.template slice<TK, TM>(k0, row0);
        op.run(tA, rg, accg);
        op.run(tA, ru, accu);
    }
    for (uint16_t i = 0; i < accg.get_capacity(); ++i)
        if (accg.is_valid_element(i)) {
            const float g = accg[i];
            accg[i] = g / (1.0f + exp(-g)) * accu[i];
        }
    tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline> tO(ot[sg], dextents<int32_t, 2>(TN, TM));
    accg.store(tO);
    simdgroup_barrier(mem_flags::mem_threadgroup);
    for (int it = lane; it < TM * TN; it += 32) {
        const int m = it / TN, f = it - m * TN;
        HS[ulong(row0 + m) * a.n_ff + n0 + f] = half(ot[sg][it]);
    }
}

template <int F, int TM, int TN, int TK, int SG>
kernel void mm_down3(device const uchar* W [[buffer(0)]], device const half* HS [[buffer(1)]],
                     device float* YE [[buffer(2)]], device const int2* tiles [[buffer(3)]], constant MmArgs& a [[buffer(4)]], device const uint* smap [[buffer(5)]],
                     device const uchar* W1 [[buffer(6)]], device const uchar* W2 [[buffer(7)]], device const uchar* W3 [[buffer(8)]],
                     uint3 tg [[threadgroup_position_in_grid]], ushort sg [[simdgroup_index_in_threadgroup]],
                     ushort tid [[thread_index_in_threadgroup]]) {
    threadgroup uint tab[fmt<F>::TAB];
    load_tab<F>(tab, tid, SG * 32);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const int2 tl = tiles[tg.y];
    const int e = tl.x, row0 = tl.y, n0 = (int(tg.x) * SG + sg) * TN;
    device const uchar* w0 = (kArena ? arena_slot(smap[e], W, W1, W2, W3) : W + ulong(e) * a.d_stride) + a.down_off + ulong(n0) * a.d_row;
    constexpr auto desc = matmul2d_descriptor(TM, TN, TK, false, true, false, matmul2d_descriptor::mode::multiply_accumulate);
    matmul2d<desc, execution_simdgroup> op;
    tensor<device half, dextents<int32_t, 2>, tensor_inline> tH((device half*)HS, dextents<int32_t, 2>(a.n_ff, 1 << 30));
    auto rw = op.template get_right_input_cooperative_tensor<half, half, float>();
    auto tA = tH.template slice<TK, TM>(0, row0);
    auto acc = op.template get_destination_cooperative_tensor<__tensor_ops_detail::__remove_addrspace_t<decltype(tA)>, __tensor_ops_detail::__remove_addrspace_t<decltype(rw)>, float>();
    for (uint16_t i = 0; i < acc.get_capacity(); ++i)
        if (acc.is_valid_element(i)) acc[i] = 0.0f;
    for (int k0 = 0; k0 < int(a.n_ff); k0 += TK) {
        if (!coop_k4) {
            for (uint16_t i = 0; i < rw.get_capacity(); ++i) {
                const auto ix = rw.get_multidimensional_index(i);
                const int k = k0 + ix[0];
                rw[i] = W4<F>::run(w0 + ulong(ix[1]) * a.d_row, k & ~3, tab)[k & 3];
            }
        } else if (TK == 32) {
            // the paired runs (element i + cap / 2 is (k + 16, n), the COOP_K4 layout): one W8 a pair
            const uint16_t half_cap = rw.get_capacity() / 2;
#pragma clang loop unroll(full)
            for (uint16_t i = 0; i < half_cap; i += 4) {
                const auto ix = rw.get_multidimensional_index(i);
                half4 w, w2;
                W8<F>::run(w0 + ulong(ix[1]) * a.d_row, k0 + ix[0], tab, w, w2);
                rw[i] = w.x; rw[i + 1] = w.y; rw[i + 2] = w.z; rw[i + 3] = w.w;
                const uint16_t j = i + half_cap;
                rw[j] = w2.x; rw[j + 1] = w2.y; rw[j + 2] = w2.z; rw[j + 3] = w2.w;
            }
        } else
#pragma clang loop unroll(full)
        for (uint16_t i = 0; i < rw.get_capacity(); i += 4) {
            const auto ix = rw.get_multidimensional_index(i);
            const half4 w = W4<F>::run(w0 + ulong(ix[1]) * a.d_row, k0 + ix[0], tab);
            rw[i] = w.x; rw[i + 1] = w.y; rw[i + 2] = w.z; rw[i + 3] = w.w;
        }
        tA = tH.template slice<TK, TM>(k0, row0);
        op.run(tA, rw, acc);
    }
    tensor<device float, dextents<int32_t, 2>, tensor_inline> tY(YE, dextents<int32_t, 2>(a.n_embd, 1 << 30));
    acc.store(tY.slice(n0, row0));
}

#define TSM_MMGU3(F, M, N, K, S) template [[host_name("mm_gu3_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_gu3<F, M, N, K, S>) mm_gu3<F, M, N, K, S>;
#define TSM_MMDN3(F, M, N, K, S) template [[host_name("mm_down3_" #F "_" #M "_" #N "_" #K "_" #S)]] kernel decltype(mm_down3<F, M, N, K, S>) mm_down3<F, M, N, K, S>;
TSM_MMGU3(42, 16, 32, 32, 4)
TSM_MMDN3(42, 16, 64, 32, 2)
