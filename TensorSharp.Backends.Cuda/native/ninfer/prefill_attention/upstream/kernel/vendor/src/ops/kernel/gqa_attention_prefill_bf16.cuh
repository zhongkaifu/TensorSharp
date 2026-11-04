#pragma once

// BF16-plane GQA prompt kernel. INT8 has an independent kernel body and resource
// policy in gqa_attention_prefill_i8.cuh.
//
//   * Br = 64 query rows and Bc = 64 key columns per CTA tile.
//   * 4 warps / 128 threads; each warp owns 16 query rows of the tile.
//   * Q, K, V staged in 96 KiB of dynamic shared memory (single-buffered), with
//     the cp.async of the next K/V tile overlapped against the current
//     QK / PV tensor-core work (exactly FA's single-buffer overlap pattern).
//   * m16n8k16 bf16 MMA for both S = Q Kᵀ and O += P V, online softmax in exp2.
//
// The op first writes the new chunk K/V into absolute positions in the paged cache,
// then computes causal GQA attention for
// every chunk token over all cached history using bottom-right causal alignment
// (query row i attends to keys [0, base_pos + i]).
//
// Rotated frame instantiation (hq-e8-2b): the K/V planes passed by the launcher
// are one-shot decoded bf16 scratch in the codec's rotated frame (linear
// [kv_head][position] rows of `scratch_span` per head, see
// gqa_attention_prefill_hq_scratch_kernel). Staged query rows are rotated into
// that frame after landing (bf16 -> FWHT -> bf16, one rounding, same as the
// fill-side rotation), and each output row is un-rotated once in FP32 in the
// epilogue. Rotation is orthogonal, so scores and the online-softmax path are
// frame-independent and stay on the shared code above.

#include <math_constants.h>

#include "ops/kernel/gqa_attention_prefill_common.cuh"
#include "ops/kernel/hq_codec.cuh"

namespace ninfer::ops {

inline constexpr std::size_t kGqaPrefillRotatedSmemBytes =
    kGqaPrefillSmemBytes + kHqHeadDim;

template <typename Geometry, typename Metadata>
__global__ void gqa_attention_prefill_fill_bf16_kernel(
    const __nv_bfloat16* __restrict__ k, const __nv_bfloat16* __restrict__ v,
    const std::int32_t* __restrict__ positions, Metadata metadata,
    __nv_bfloat16* __restrict__ cache_k, __nv_bfloat16* __restrict__ cache_v, std::int32_t width) {
    constexpr int VecElems = 8; // 8 bf16 == 16 B, matching the cache row alignment.
    const int tokens       = metadata.valid_tokens(width);
    const std::int64_t idx = static_cast<std::int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    const std::int64_t n =
        static_cast<std::int64_t>(tokens) * Geometry::KVHeads * (kGqaPrefillHeadDim / VecElems);
    if (idx >= n) { return; }

    const int vec                   = static_cast<int>(idx % (kGqaPrefillHeadDim / VecElems));
    const int tmp                   = static_cast<int>(idx / (kGqaPrefillHeadDim / VecElems));
    const int kv_head               = tmp % Geometry::KVHeads;
    const int token                 = tmp / Geometry::KVHeads;
    const int d                     = vec * VecElems;
    const int position              = positions[0] + token;
    const int lane                  = static_cast<int>(threadIdx.x) & 31;
    const std::int32_t* block_table = metadata.block_table();
    int physical_page               = lane == 0 ? paged_kv_physical_page(block_table, position) : 0;
    const std::int64_t src_off =
        static_cast<std::int64_t>(d) +
        static_cast<std::int64_t>(kGqaPrefillHeadDim) * (kv_head + Geometry::KVHeads * token);
    const int4 k_value = load_vec<int4>(&k[src_off]);
    const int4 v_value = load_vec<int4>(&v[src_off]);

    physical_page = __shfl_sync(0xffffffffu, physical_page, 0);

    const std::int64_t cache_off = paged_kv_element_offset<kGqaPrefillHeadDim, Geometry::KVHeads>(
        physical_page, kv_head, position & kPagedKVPageMask, d);
    store_vec(&cache_k[cache_off], k_value);
    store_vec(&cache_v[cache_off], v_value);
}

// Stage one [Bc, D] K or V tile from the per-kv-head contiguous cache into the
// swizzled smem buffer. Keys beyond max_query_abs (which the causal mask always
// drops) are zeroed so the padded/uninitialized cache tail never feeds NaNs into
// the tensor cores. Mirrors FA's predicated K/V cp.async + Clear_OOB path.
// Rotated: `cache` is decoded linear scratch ([kv_head][position] rows of
// scratch_span per head) and physical_page is ignored.
template <typename Geometry, bool Rotated = false>
__device__ __forceinline__ void gqa_prefill_stage_kv(__nv_bfloat16* dst, const __nv_bfloat16* cache,
                                                     int kv_head, int k0, int max_query_abs,
                                                     int physical_page, int tid,
                                                     std::int32_t scratch_span, int key_base = 0,
                                                     int key_limit = 0x7fffffff) {
    constexpr int D         = kGqaPrefillHeadDim;
    constexpr int Bc        = kGqaPrefillBc;
    constexpr int Threads   = kGqaPrefillThreads;
    constexpr int VecPerRow = D / 8; // 8 bf16 per 16B cp.async
    // Banded (Rotated scratch) staging is band-local: absolute key k0 addresses scratch row
    // k0 - key_base, and rows past key_limit are beyond the decoded band - zero-fill them.
    const int bound         = min(max_query_abs, key_limit);
    const bool full_tile    = (k0 + Bc - 1) <= bound;
    // Block base pointer computed once (int64); per-element offsets stay 32-bit.
    const __nv_bfloat16* cache_block =
        Rotated ? cache + (static_cast<std::int64_t>(kv_head) * scratch_span + k0 - key_base) * D
                : cache + paged_kv_element_offset<kGqaPrefillHeadDim, Geometry::KVHeads>(
                              physical_page, kv_head, k0 & kPagedKVPageMask, 0);
    if (full_tile) {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;        // / VecPerRow (32)
            const int d      = (chunk & 31) << 3; // (chunk % 32) * 8
            __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
            cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
        }
    } else {
#pragma unroll
        for (int chunk = tid; chunk < Bc * VecPerRow; chunk += Threads) {
            const int key_l  = chunk >> 5;        // / VecPerRow (32)
            const int d      = (chunk & 31) << 3; // (chunk % 32) * 8
            __nv_bfloat16* p = &dst[key_l * D + gqa_prefill_swz(key_l, d)];
            if ((k0 + key_l) <= bound) {
                cp_async<16, Cache::cg>(p, &cache_block[key_l * D + d]);
            } else {
                store_vec(p, make_int4(0, 0, 0, 0));
            }
        }
    }
}

// FlashAttention-2 forward, one CTA per (query 64-row block, query head). Grid is
// (ceil(tokens/64), q_heads). seqlen_q = tokens, seqlen_k = base_pos + tokens, with
// bottom-right causal alignment (query row i sees keys [0, base_pos + i]).
// Carry (banded hq prompt route): the kernel covers only keys [key_begin, key_end) of the
// scratch, starts its online-softmax state from the carried (m, l, acc) buffers when key_begin
// > 0, and writes the state back unnormalized instead of normalizing into `out` when
// store_carry is set. The buffers use `out`'s [head_dim, q_heads, width] layout for acc and
// [q_heads, width] for m/l; acc crosses band boundaries as bf16, matching the decode split
// partial contract. Single-band launches pass the defaults and stay on the original path.
template <typename Geometry, typename Metadata, bool Rotated = false, bool Carry = false>
__launch_bounds__(kGqaPrefillThreads, 1) __global__
    void gqa_attention_prefill_bf16_kernel(const __nv_bfloat16* __restrict__ q,
                                           const __nv_bfloat16* __restrict__ cache_k,
                                           const __nv_bfloat16* __restrict__ cache_v,
                                           Metadata metadata,
                                           const std::int32_t* __restrict__ positions, float scale,
                                           __nv_bfloat16* __restrict__ out, std::int32_t width,
                                           std::int32_t scratch_span,
                                           std::int32_t key_begin = 0,
                                           std::int32_t key_end   = 0x7fffffff,
                                           __nv_bfloat16* __restrict__ carry_acc   = nullptr,
                                           float* __restrict__ carry_m             = nullptr,
                                           float* __restrict__ carry_l             = nullptr,
                                           std::int32_t store_carry                = 0,
                                           float* __restrict__ partial_acc         = nullptr,
                                           float* __restrict__ partial_m           = nullptr,
                                           float* __restrict__ partial_l           = nullptr,
                                           std::int32_t split_count                = 1) {
    // Key-split partials (ROADMAP WI-K1a) share the small-T layout contract with the INT8
    // kernel; split_count > 1 is only legal on non-Carry launches (banded carry state is
    // sequential across bands, enforced by the launcher).
    static_assert(!Carry || Rotated, "band carry exists only on the rotated scratch route");
    constexpr int D             = kGqaPrefillHeadDim; // 256
    constexpr int Br            = kGqaPrefillBr;      // 64 query rows
    constexpr int Bc            = kGqaPrefillBc;      // 64 key cols
    constexpr int Threads       = kGqaPrefillThreads; // 128
    constexpr int QKNt          = Bc / 8;             // 8  QK score n-tiles
    constexpr int QKKs          = D / 16;             // 16 QK contraction steps over head_dim
    constexpr int PVNt          = D / 8;              // 32 PV output n-tiles
    constexpr int PVKs          = Bc / 16;            // 4  PV contraction steps over keys
    constexpr float Log2E       = 1.4426950408889634074f;
    constexpr unsigned FullMask = 0xffffffffu;

    static_assert(Threads == 128);
    if constexpr (!Rotated) { (void)scratch_span; }
    if constexpr (!Carry) {
        (void)key_begin;
        (void)key_end;
        (void)carry_acc;
        (void)carry_m;
        (void)carry_l;
        (void)store_carry;
    }

    extern __shared__ __align__(16) __nv_bfloat16 gqa_smem[];
    __nv_bfloat16* q_s = gqa_smem;     // [Br, D] swizzled
    __nv_bfloat16* k_s = q_s + Br * D; // [Bc, D] swizzled
    __nv_bfloat16* v_s = k_s + Bc * D; // [Bc, D] swizzled
    // RHT diagonal for the Rotated instantiation (launcher passes the enlarged
    // dynamic-smem size); untouched padding otherwise.
    std::int8_t* signs = reinterpret_cast<std::int8_t*>(v_s + Bc * D);

    const int q_block = static_cast<int>(blockIdx.x);
    const int q_head  = static_cast<int>(blockIdx.y);
    const int tid     = static_cast<int>(threadIdx.x);
    const int warp    = tid >> 5;
    const int lane    = tid & 31;
    const int q0      = q_block * Br;
    const int kv_head = q_head / Geometry::GroupSize;
    const int tokens  = metadata.valid_tokens(width);

    if (q_head >= Geometry::QHeads || q0 >= width) { return; }
    if (q0 >= tokens) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, q0, min(q0 + Br, width), tid, Threads);
        return;
    }
    const int base_pos              = positions[0];
    const std::int32_t* block_table = metadata.block_table();

    const int gid = lane >> 2;
    const int lid = lane & 3;

    const int a_mat     = lane >> 3;
    const int a_rin     = lane & 7;
    const int a_rowoff  = a_rin + ((a_mat & 1) << 3);
    const int b_rin     = lane & 7;
    const int b_koff    = ((lane >> 3) & 1) << 3;
    const int warp_row0 = warp * 16; // this warp owns rows [warp_row0, warp_row0+16)

    // Per-lane precomputed swizzled ldmatrix base addresses (see gqa_prefill_swz_addr).
    const unsigned q_sbase = smem_addr(q_s);
    const unsigned k_sbase = smem_addr(k_s);
    const unsigned v_sbase = smem_addr(v_s);
    // Q A-fragment: row = warp_row0 + a_rowoff, col = k*16 + a_coloff.
    const unsigned q_lane_base = q_sbase + static_cast<unsigned>((warp_row0 + a_rowoff) * 512);
    const unsigned q_as        = static_cast<unsigned>((a_mat >> 1) << 4);
    const unsigned q_r         = static_cast<unsigned>(a_rin << 4);
    // K B-fragment via ldmatrix.x4 (2 n-tiles/instr): lanes 16-31 fetch the +8-key
    // half (extra 4096 bytes), lanes with bit3 set fetch the +8 d-contract half.
    const unsigned k_lane_base =
        k_sbase + static_cast<unsigned>(b_rin * 512) + (static_cast<unsigned>(lane >> 4) << 12);
    const unsigned k_as = static_cast<unsigned>((b_koff >> 3) << 4);
    const unsigned k_r  = static_cast<unsigned>(b_rin << 4);
    // V B-fragment via ldmatrix.x4.trans (2 n-tiles/instr): row = k*16 + (bit3)*8 + b_rin,
    // col = n*8 + (lane>>4)*8.
    const unsigned v_lane_base = v_sbase + static_cast<unsigned>(((lane >> 3) & 1) * 4096) +
                                 static_cast<unsigned>(b_rin * 512);
    const unsigned v_as = static_cast<unsigned>((lane >> 4) << 4);
    const unsigned v_r  = static_cast<unsigned>(b_rin << 4);

    // Stage Q into smem once via cp.async (overlaps with the K(0) prologue load
    // below); it stays resident for the whole key loop. Global Q rows are 256 bf16
    // contiguous, with a token stride of 256*QHeads.
    {
        constexpr int VecPerRow      = D / 8;
        constexpr int QRowStride     = D * Geometry::QHeads; // global stride between tokens
        const __nv_bfloat16* q_block = q + gqa_prefill_q_index<Geometry>(q_head, 0, q0);
        if (q0 + Br <= tokens) {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
            }
        } else {
#pragma unroll
            for (int chunk = tid; chunk < Br * VecPerRow; chunk += Threads) {
                const int row    = chunk >> 5;
                const int d      = (chunk & 31) << 3;
                __nv_bfloat16* p = &q_s[row * D + gqa_prefill_swz(row, d)];
                if (q0 + row < tokens) {
                    cp_async<16, Cache::cg>(p, &q_block[row * QRowStride + d]);
                } else {
                    store_vec(p, make_int4(0, 0, 0, 0));
                }
            }
        }
    }

    float acc[PVNt][4];
#pragma unroll
    for (int n = 0; n < PVNt; ++n) {
#pragma unroll
        for (int i = 0; i < 4; ++i) { acc[n][i] = 0.0f; }
    }
    float m0 = -CUDART_INF_F, m1 = -CUDART_INF_F, l0 = 0.0f, l1 = 0.0f;

    const int tile_rows     = min(Br, tokens - q0);
    const int max_query_abs = base_pos + q0 + tile_rows - 1;
    const int key_limit     = Carry ? min(key_end, max_query_abs + 1) : (max_query_abs + 1);
    const int kb_begin      = Carry ? (key_begin / Bc) : 0;
    const int n_block_max   = (key_limit + Bc - 1) / Bc; // tiles [kb_begin, n_block_max)

    // Key-split segment of the tile range (ROADMAP WI-K1a): identical merge contract to the
    // INT8 kernel — each split stores pre-normalized FP32 partials for the shared reducer.
    const int split           = static_cast<int>(blockIdx.z);
    const int tiles_per_split = (n_block_max - kb_begin + split_count - 1) / split_count;
    const int kb_split_begin  = kb_begin + split * tiles_per_split;
    const int kb_split_end    = min(kb_split_begin + tiles_per_split, n_block_max);

    if constexpr (Carry) {
        if (key_begin > 0) {
            // Resume the online softmax from the previous band's state. The carried acc is
            // stored un-rotated (out frame): load each row, FWHT-rotate into the codec frame,
            // stage through the same 8-row smem window the epilogue uses, and rebuild the C
            // fragments. Runs before Q/K(0) staging, so k_s is dead storage here.
            hq_engine_signs_fill(signs);
            __syncthreads();
            float* epi = reinterpret_cast<float*>(k_s) + warp * 8 * D;
            for (int half = 0; half < 2; ++half) {
                for (int i = 0; i < 8; ++i) {
                    const int qrow = q0 + warp_row0 + half * 8 + i;
                    float reg[8];
                    if (qrow < tokens) {
#pragma unroll
                        for (int si = 0; si < 8; ++si) {
                            reg[si] = __bfloat162float(
                                carry_acc[gqa_prefill_q_index<Geometry>(q_head, si * 32 + lane,
                                                                        qrow)]);
                        }
                        hq_fwht256_sign(reg, signs, 0, lane);
                    } else {
#pragma unroll
                        for (int si = 0; si < 8; ++si) { reg[si] = 0.0f; }
                    }
#pragma unroll
                    for (int si = 0; si < 8; ++si) { epi[i * D + si * 32 + lane] = reg[si]; }
                }
                __syncwarp(FullMask);
#pragma unroll
                for (int n = 0; n < PVNt; ++n) {
                    const int d0 = n * 8 + 2 * lid;
                    if (half == 0) {
                        acc[n][0] = epi[gid * D + d0 + 0];
                        acc[n][1] = epi[gid * D + d0 + 1];
                    } else {
                        acc[n][2] = epi[gid * D + d0 + 0];
                        acc[n][3] = epi[gid * D + d0 + 1];
                    }
                }
                __syncwarp(FullMask);
            }
            // l runs as per-lane partials inside the key loop (the epilogue's warp_sum<4>
            // completes the row), so the full carried l splits evenly across the row's four
            // lanes; m is row-uniform after each tile's warp_max.
            const int row0_c = q0 + warp_row0 + gid;
            const int row1_c = row0_c + 8;
            if (row0_c < tokens) {
                const std::int64_t stat = static_cast<std::int64_t>(q_head) * width + row0_c;
                m0 = carry_m[stat];
                l0 = 0.25f * carry_l[stat];
            }
            if (row1_c < tokens) {
                const std::int64_t stat = static_cast<std::int64_t>(q_head) * width + row1_c;
                m1 = carry_m[stat];
                l1 = 0.25f * carry_l[stat];
            }
            __syncthreads();
        }
    }

    // Fold softmax_scale into the exp2 (FA-style): scores stay raw, so the
    // per-element "* scale" multiply drops out of the QK epilogue entirely.
    const float scale_l2 = scale * Log2E;
    int physical_page    = block_table[kb_begin];

    // Prologue: commit Q, then kick off K(0). The loop's wait<0> below drains both. An empty
    // split segment skips the K(0) stage entirely — the block table past this CTA's key range
    // holds no valid page, and the staged rows would never be consumed anyway.
    ninfer::ops::cp_commit();
    if (kb_split_begin < kb_split_end) {
        physical_page = block_table[kb_split_begin];
        gqa_prefill_stage_kv<Geometry, Rotated>(k_s, cache_k, kv_head, kb_split_begin * Bc,
                                                max_query_abs, physical_page, tid, scratch_span,
                                                Carry ? key_begin : 0,
                                                Carry ? key_end : 0x7fffffff);
        ninfer::ops::cp_commit();
    }

    if constexpr (Rotated) {
        // Rotate the staged q rows into the codec frame before any QK MMA reads
        // them: bf16 -> FWHT(+signs) -> bf16, the same single rounding the fill
        // side applies to K/V. Zero-padded tail rows stay exactly zero.
        ninfer::ops::cp_wait<0>(); // Q landed (drains K(0) early; harmless)
        __syncthreads();
        hq_engine_signs_fill(signs);
        __syncthreads();
        for (int row = warp; row < Br; row += Threads / 32) {
            float reg[8];
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                reg[s] = __bfloat162float(q_s[row * D + gqa_prefill_swz(row, s * 32 + lane)]);
            }
            hq_fwht256_sign(reg, signs, 0, lane);
#pragma unroll
            for (int s = 0; s < 8; ++s) {
                q_s[row * D + gqa_prefill_swz(row, s * 32 + lane)] = __float2bfloat16(reg[s]);
            }
        }
        __syncthreads();
    }

    for (int kb = kb_split_begin; kb < kb_split_end; ++kb) {
        const int k0                 = kb * Bc;
        const int next_physical_page =
            (kb + 1 < kb_split_end) ? block_table[kb + 1] : physical_page;

        ninfer::ops::cp_wait<0>(); // K(kb) landed (also publishes q_s / prev PV done)
        __syncthreads();

        // Overlap V(kb) load against the QK MMA below.
        gqa_prefill_stage_kv<Geometry, Rotated>(v_s, cache_v, kv_head, k0, max_query_abs,
                                                physical_page, tid, scratch_span,
                                                Carry ? key_begin : 0,
                                                Carry ? key_end : 0x7fffffff);
        ninfer::ops::cp_commit();

        // S = Q Kᵀ for this warp's 16 rows over all Bc keys, in registers.
        // Software-pipelined like cute's gemm: issue the ldmatrix for contraction
        // step k+1 while the m16n8k16 MMAs for step k run, so the LSU (ldmatrix)
        // and tensor pipes overlap instead of stalling on each other.
        float score[QKNt][4];
#pragma unroll
        for (int nt = 0; nt < QKNt; ++nt) {
            score[nt][0] = score[nt][1] = score[nt][2] = score[nt][3] = 0.0f;
        }
        // Swizzled ldmatrix addresses via precomputed per-lane bases + immediates.
        unsigned af[2][4];
        unsigned bf[2][QKNt][2];
        {
            ldmatrix_x4(af[0][0], af[0][1], af[0][2], af[0][3],
                        gqa_prefill_swz_addr(q_lane_base, 0u, q_as, q_r));
#pragma unroll
            for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                ldmatrix_x4(bf[0][nt2][0], bf[0][nt2][1], bf[0][nt2 + 1][0], bf[0][nt2 + 1][1],
                            gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096),
                                                 0u, k_as, k_r));
            }
        }
#pragma unroll
        for (int k = 0; k < QKKs; ++k) {
            const int cur = k & 1;
            const int nxt = cur ^ 1;
            if (k + 1 < QKKs) {
                const unsigned ck = static_cast<unsigned>((k + 1) << 5);
                ldmatrix_x4(af[nxt][0], af[nxt][1], af[nxt][2], af[nxt][3],
                            gqa_prefill_swz_addr(q_lane_base, ck, q_as, q_r));
#pragma unroll
                for (int nt2 = 0; nt2 < QKNt; nt2 += 2) {
                    ldmatrix_x4(
                        bf[nxt][nt2][0], bf[nxt][nt2][1], bf[nxt][nt2 + 1][0], bf[nxt][nt2 + 1][1],
                        gqa_prefill_swz_addr(k_lane_base + static_cast<unsigned>(nt2 * 4096), ck,
                                             k_as, k_r));
                }
            }
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                mma_bf16(score[nt][0], score[nt][1], score[nt][2], score[nt][3], af[cur][0],
                         af[cur][1], af[cur][2], af[cur][3], bf[cur][nt][0], bf[cur][nt][1]);
            }
        }

        const int row0             = warp_row0 + gid;
        const int row1             = warp_row0 + gid + 8;
        const int qrow0            = q0 + row0;
        const int qrow1            = q0 + row1;
        const int qabs0            = (qrow0 < tokens) ? base_pos + qrow0 : -1;
        const int qabs1            = (qrow1 < tokens) ? base_pos + qrow1 : -1;
        const bool full_score_tile =
            (q0 + Br <= tokens) && ((k0 + Bc - 1) <= (base_pos + q0)) &&
            (!Carry || (k0 + Bc) <= key_end);

        // block row-max on raw (unscaled) scores; scale is folded into exp2 below
        float bm0 = -CUDART_INF_F, bm1 = -CUDART_INF_F;
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                bm0 = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1 = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const int key0 = k0 + nt * 8 + 2 * lid;
                const int key1 = key0 + 1;
                const bool key0_ok =
                    key0 <= qabs0 && (!Carry || key0 < key_end);
                const bool key1_ok =
                    key1 <= qabs0 && (!Carry || key1 < key_end);
                score[nt][0]   = (qrow0 < tokens && key0_ok) ? score[nt][0] : -CUDART_INF_F;
                score[nt][1]   = (qrow0 < tokens && key1_ok) ? score[nt][1] : -CUDART_INF_F;
                score[nt][2]   = (qrow1 < tokens && key0 <= qabs1 && (!Carry || key0 < key_end))
                                     ? score[nt][2]
                                     : -CUDART_INF_F;
                score[nt][3]   = (qrow1 < tokens && key1 <= qabs1 && (!Carry || key1 < key_end))
                                     ? score[nt][3]
                                     : -CUDART_INF_F;
                bm0            = fmaxf(bm0, fmaxf(score[nt][0], score[nt][1]));
                bm1            = fmaxf(bm1, fmaxf(score[nt][2], score[nt][3]));
            }
        }
        bm0 = warp_max<4>(bm0, FullMask);
        bm1 = warp_max<4>(bm1, FullMask);

        const float nm0        = fmaxf(m0, bm0);
        const float nm1        = fmaxf(m1, bm1);
        const float nm0_scaled = nm0 * scale_l2;
        const float nm1_scaled = nm1 * scale_l2;
        // A split segment can lie entirely past a row's causal limit, so both m and the
        // block max stay -inf here: the naive exp2 would compute -inf - (-inf) = NaN. The
        // INT8 kernel guards the same case; the single-pass bf16 launch never met it.
        const float alpha0 = (m0 == -CUDART_INF_F) ? 0.0f
                                               : exp2_approx(__fmaf_rn(m0, scale_l2, -nm0_scaled));
        const float alpha1 = (m1 == -CUDART_INF_F) ? 0.0f
                                               : exp2_approx(__fmaf_rn(m1, scale_l2, -nm1_scaled));

        // P = exp2(S - m), repacked into the PV A-fragment layout, plus local block row-sum.
        // The row-sum allreduce is deferred to the epilogue; only row max must be reduced per tile.
        float bl0 = 0.0f, bl1 = 0.0f;
        unsigned p_frag[PVKs][4];
        if (full_score_tile) {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled));
                const float p01 = exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled));
                const float p10 = exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled));
                const float p11 = exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled));
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        } else {
#pragma unroll
            for (int nt = 0; nt < QKNt; ++nt) {
                const float p00 = (score[nt][0] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][0], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p01 = (score[nt][1] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][1], scale_l2, -nm0_scaled))
                                      : 0.0f;
                const float p10 = (score[nt][2] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][2], scale_l2, -nm1_scaled))
                                      : 0.0f;
                const float p11 = (score[nt][3] > -CUDART_INF_F)
                                      ? exp2_approx(__fmaf_rn(score[nt][3], scale_l2, -nm1_scaled))
                                      : 0.0f;
                bl0 += p00 + p01;
                bl1 += p10 + p11;
                const int pk = nt >> 1;
                if ((nt & 1) == 0) {
                    p_frag[pk][0] = pack_bf16x2(p00, p01);
                    p_frag[pk][1] = pack_bf16x2(p10, p11);
                } else {
                    p_frag[pk][2] = pack_bf16x2(p00, p01);
                    p_frag[pk][3] = pack_bf16x2(p10, p11);
                }
            }
        }

        l0 = __fmaf_rn(l0, alpha0, bl0);
        l1 = __fmaf_rn(l1, alpha1, bl1);
        m0 = nm0;
        m1 = nm1;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            acc[n][0] *= alpha0;
            acc[n][1] *= alpha0;
            acc[n][2] *= alpha1;
            acc[n][3] *= alpha1;
        }

        ninfer::ops::cp_wait<0>(); // V(kb) landed; QK done reading k_s
        __syncthreads();

        // Prefetch K(kb+1) into the (now-free) K buffer, overlapping the PV MMA.
        if (kb + 1 < kb_split_end) {
            physical_page = next_physical_page;
            gqa_prefill_stage_kv<Geometry, Rotated>(k_s, cache_k, kv_head, (kb + 1) * Bc,
                                                    max_query_abs, physical_page, tid,
                                                    scratch_span, Carry ? key_begin : 0,
                                                    Carry ? key_end : 0x7fffffff);
            ninfer::ops::cp_commit();
        }

        // O += P V, contracting over the Bc keys. The (k, n) iteration space is
        // flattened and software-pipelined: the transposed ldmatrix for the next
        // V fragment is issued while the current MMA runs.
        // Each x4.trans load covers 2 output n-tiles (16 dims); pipeline the next
        // load against the current pair of MMAs.
        constexpr int PVHalf  = PVNt / 2;      // 16 n-tile pairs
        constexpr int PVLoads = PVKs * PVHalf; // 64 x4.trans loads
        // Swizzled V x4.trans addresses via precomputed per-lane base + immediates.
        unsigned vf[2][4];
        {
            ldmatrix_x4_t(vf[0][0], vf[0][1], vf[0][2], vf[0][3],
                          gqa_prefill_swz_addr(v_lane_base, 0u, v_as, v_r));
        }
#pragma unroll
        for (int li = 0; li < PVLoads; ++li) {
            const int k   = li / PVHalf;
            const int n2  = (li % PVHalf) * 2;
            const int cur = li & 1;
            const int nxt = cur ^ 1;
            if (li + 1 < PVLoads) {
                const int k2       = (li + 1) / PVHalf;
                const int n2b      = ((li + 1) % PVHalf) * 2;
                const unsigned ckv = static_cast<unsigned>(n2b << 4);
                ldmatrix_x4_t(vf[nxt][0], vf[nxt][1], vf[nxt][2], vf[nxt][3],
                              gqa_prefill_swz_addr(v_lane_base + static_cast<unsigned>(k2 * 8192),
                                                   ckv, v_as, v_r));
            }
            mma_bf16(acc[n2][0], acc[n2][1], acc[n2][2], acc[n2][3], p_frag[k][0], p_frag[k][1],
                     p_frag[k][2], p_frag[k][3], vf[cur][0], vf[cur][1]);
            mma_bf16(acc[n2 + 1][0], acc[n2 + 1][1], acc[n2 + 1][2], acc[n2 + 1][3], p_frag[k][0],
                     p_frag[k][1], p_frag[k][2], p_frag[k][3], vf[cur][2], vf[cur][3]);
        }
    }

    l0 = warp_sum<4>(l0, FullMask);
    l1 = warp_sum<4>(l1, FullMask);

    if (split_count > 1) {
        // Split mode: drain any pending Q staging (empty segments never entered the loop),
        // publish per-row (m, l) from the warp's registers, and neutralize dead rows that
        // still own merge slots. Rows past width own no slot and are never written; acc
        // partials are stored pre-normalized inside the epilogues below.
        ninfer::ops::cp_wait<0>();
        if (lid == 0) {
            const int r0 = q0 + warp_row0 + gid;
            const int r1 = r0 + 8;
            if (r0 < width) {
                partial_m[gqa_prefill_partial_stat_index<Geometry>(q_head, r0, split, width)] =
                    r0 < tokens ? m0 : -CUDART_INF_F;
                partial_l[gqa_prefill_partial_stat_index<Geometry>(q_head, r0, split, width)] =
                    r0 < tokens ? l0 : 0.0f;
            }
            if (r1 < width) {
                partial_m[gqa_prefill_partial_stat_index<Geometry>(q_head, r1, split, width)] =
                    r1 < tokens ? m1 : -CUDART_INF_F;
                partial_l[gqa_prefill_partial_stat_index<Geometry>(q_head, r1, split, width)] =
                    r1 < tokens ? l1 : 0.0f;
            }
        }
    }

    if constexpr (Rotated) {
        // Un-rotate each output row once in FP32, then normalize and store bf16.
        // Each warp stages its 16 rows through a private 8-row FP32 window in
        // the (now dead) K staging buffer — half the rows per pass — so the
        // whole row is visible to the warp's shuffle-based inverse FWHT. Rows
        // past `tokens` rotate with garbage accumulators but only the final
        // store is predicated, keeping every shuffle converged.
        float* epi = reinterpret_cast<float*>(k_s) + warp * 8 * D;
        for (int half = 0; half < 2; ++half) {
#pragma unroll
            for (int n = 0; n < PVNt; ++n) {
                const int d0 = n * 8 + 2 * lid;
                if (half == 0) {
                    epi[gid * D + d0 + 0] = acc[n][0];
                    epi[gid * D + d0 + 1] = acc[n][1];
                } else {
                    epi[gid * D + d0 + 0] = acc[n][2];
                    epi[gid * D + d0 + 1] = acc[n][3];
                }
            }
            __syncwarp(FullMask);
            for (int i = 0; i < 8; ++i) {
                // Row l lives on lanes [4i, 4i+4) of the C fragment; after the
                // 4-lane butterfly reduction any of them holds the full sum.
                const float lrow = __shfl_sync(FullMask, half == 0 ? l0 : l1, i << 2);
                const float mrow = __shfl_sync(FullMask, half == 0 ? m0 : m1, i << 2);
                const float inv_l = (lrow > 0.0f) ? __frcp_rn(lrow) : 0.0f;
                float reg[8];
#pragma unroll
                for (int s = 0; s < 8; ++s) { reg[s] = epi[i * D + s * 32 + lane]; }
                hq_ifwht256_sign(reg, signs, 0, lane);
                const int qrow = q0 + warp_row0 + half * 8 + i;
                if (qrow < tokens) {
                    if (split_count > 1) {
                        // Split partial in the un-rotated output frame, pre-normalized (FP32)
                        // for the shared prefill reducer.
#pragma unroll
                        for (int s = 0; s < 8; ++s) {
                            partial_acc[gqa_prefill_partial_acc_index<Geometry>(
                                q_head, s * 32 + lane, qrow, split, width)] = reg[s] * inv_l;
                        }
                    } else if constexpr (Carry) {
                        if (store_carry != 0) {
                            // Hand the unnormalized un-rotated accumulator and the running
                            // softmax state to the next band; the row's lanes [4i,4i+4) agree
                            // on m/l, so one of them writes.
#pragma unroll
                            for (int s = 0; s < 8; ++s) {
                                carry_acc[gqa_prefill_q_index<Geometry>(q_head, s * 32 + lane,
                                                                        qrow)] =
                                    __float2bfloat16(reg[s]);
                            }
                            if (lane == (i << 2)) {
                                const std::int64_t stat =
                                    static_cast<std::int64_t>(q_head) * width + qrow;
                                carry_m[stat] = mrow;
                                carry_l[stat] = lrow;
                            }
                        } else {
#pragma unroll
                            for (int s = 0; s < 8; ++s) {
                                out[gqa_prefill_q_index<Geometry>(q_head, s * 32 + lane, qrow)] =
                                    __float2bfloat16(reg[s] * inv_l);
                            }
                        }
                    } else {
#pragma unroll
                        for (int s = 0; s < 8; ++s) {
                            out[gqa_prefill_q_index<Geometry>(q_head, s * 32 + lane, qrow)] =
                                __float2bfloat16(reg[s] * inv_l);
                        }
                    }
                }
            }
            __syncwarp(FullMask);
        }
    } else {
        // Normalize once per row via reciprocal-multiply instead of 128 IEEE divides.
        const float inv_l0 = (l0 > 0.0f) ? __frcp_rn(l0) : 0.0f;
        const float inv_l1 = (l1 > 0.0f) ? __frcp_rn(l1) : 0.0f;
#pragma unroll
        for (int n = 0; n < PVNt; ++n) {
            const int d0    = n * 8 + 2 * lid;
            const int qrow0 = q0 + warp_row0 + gid;
            const int qrow1 = q0 + warp_row0 + gid + 8;
            if (split_count > 1) {
                if (qrow0 < tokens) {
                    float2* slot = reinterpret_cast<float2*>(
                        &partial_acc[gqa_prefill_partial_acc_index<Geometry>(q_head, d0, qrow0,
                                                                             split, width)]);
                    *slot        = make_float2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
                }
                if (qrow1 < tokens) {
                    float2* slot = reinterpret_cast<float2*>(
                        &partial_acc[gqa_prefill_partial_acc_index<Geometry>(q_head, d0, qrow1,
                                                                             split, width)]);
                    *slot        = make_float2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
                }
            } else {
                if (qrow0 < tokens) {
                    *reinterpret_cast<unsigned*>(
                        &out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow0)]) =
                        pack_bf16x2(acc[n][0] * inv_l0, acc[n][1] * inv_l0);
                }
                if (qrow1 < tokens) {
                    *reinterpret_cast<unsigned*>(
                        &out[gqa_prefill_q_index<Geometry>(q_head, d0, qrow1)]) =
                        pack_bf16x2(acc[n][2] * inv_l1, acc[n][3] * inv_l1);
                }
            }
        }
    }
    if (split_count == 1) {
        gqa_prefill_zero_output_rows<Geometry>(out, q_head, tokens, min(q0 + Br, width), tid,
                                               Threads);
    }
}

} // namespace ninfer::ops
