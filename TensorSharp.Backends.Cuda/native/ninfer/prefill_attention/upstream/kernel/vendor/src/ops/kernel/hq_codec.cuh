#pragma once

// ninfer::ops - HyperQuant KV cache codec (shared device helpers): per-row
// randomized Hadamard rotation, E8 lattice true-nearest quantization,
// structural strip, and Rice entropy coding packed into a FIXED per-row byte
// budget. The format keeps the paged-KV contract's fixed-bytes-per-token
// property: every (token, kv_head) row occupies exactly kHqRowBudgetBytes of
// code plane and kHqMetaBytes of metadata plane, so capacity math, page
// addressing, and CUDA Graph address stability are unchanged from the
// fixed-width formats.
//
// Pipeline per row x (dim 256), all in the rotated frame:
//   1. norm = ||x||;  u = FWHT256(x . signs) * sqrt(256) / norm
//      (per-scalar variance ~1; signs are the fixed engine-global RHT
//      diagonal, one vector of 256 signed bytes shared by both roles)
//   2. y = nearest point of 2*E8 for alpha * u  (both cosets tried, mod-4
//      sum parity fixed by the least-cost single +/-2 flip; this is the
//      exact Conway-Sloane decoder - the half-integer coset is NOT collapsed)
//   3. strip coset + parity redundancy -> 8 zigzagged symbols per 8-D word
//   4. Rice-pack all 32 words' symbols (256 symbols) into the row budget,
//      one Rice parameter per row chosen exactly from 8 accumulators
//
// Budget guarantee: a row whose symbols do not fit is re-encoded at
// alpha/2 (then alpha/4), which strictly shrinks the symbols; a row of
// all-zero lattice codes always fits (256 one-bit codes = 32 bytes <=
// budget), so encoding terminates in bounded deterministic time with no
// host involvement (graph-safe). Escalation is recorded in the metadata
// flags; at the shipped alpha a small fraction of Gaussian rows (measured
// ~0.4% on the unit-variance corpus, more on heavy-tailed rows) needs the
// first halving.
//
// Storage planes per pool (one page id shared across all planes):
//   codes: U8 [kHqCodePlaneExtent=64, 64, Hkv, Nphysical]  (page-major)
//   meta : U8 [kHqMetaPlaneExtent=8, 64, Hkv, Nphysical]
// Meta row layout (8 bytes):
//   [0..1] FP16 bits of the row L2 norm (multiplier, like INT8-G64 scales)
//   [2]    Rice k (low 4 bits) | escalation count (bits 4..5) | reserved
//   [3..4] used bits: meta[3] plus bits 0..1 of meta[4] (<= 512, 10 bits)
//   [4]    bits 2..4: the 9th bit of each segment offset; bits 5..7 reserved
//   [5..7] segment offsets, low 8 bits: the BIT where symbol 64/128/192
//          starts (<= 512). Consumers decode one row with four threads, each
//          starting its serial Rice scan at these bit offsets; the bitstream
//          itself is identical to the sequential format.
//
// Dequantized row value (rotated frame): y[i] * norm / (alpha * sqrt(256)).
// Consumers rotate queries with the same signs+FWHT and un-rotate attention
// outputs once per output row (the rotation is orthogonal). Plane pointers
// are computed by the attention/fill kernels via paged_kv_element_offset
// with the extents below; this header stays pointer-free so it builds and
// qualifies standalone.
//
// Stored-stream invariant (load-bearing for the group decoder): every code
// byte at or past ceil(used/8) is ZERO (the packer zero-initializes the
// staging row and the terminal fallback writes 64 zero-tail bytes), so decoders
// may read all 16 words of a row and treat bits past `used` as unary-guard
// padding without consulting `used` for word bounds.

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <cstdint>

#include "ninfer/ops/gqa_attention.h"

namespace ninfer::ops {

inline constexpr int kHqHeadDim        = 256;
inline constexpr int kHqLatticeDim     = 8;
inline constexpr int kHqWordsPerRow    = kHqHeadDim / kHqLatticeDim; // 32
inline constexpr int kHqRowBudgetBytes = 64;  // 512-bit code budget = 2 bits/dim
inline constexpr int kHqMetaBytes      = 8;
inline constexpr int kHqCodePlaneExtent = kHqRowBudgetBytes;
inline constexpr int kHqMetaPlaneExtent = kHqMetaBytes;
inline constexpr int kHqMaxRiceK       = 7;
// A valid row fits 512 bits, so no legitimate unary quotient can exceed
// the budget; runs past that are zero/garbage padding and bail fast.
inline constexpr std::uint32_t kHqUnaryGuard = kHqRowBudgetBytes * 8;

// Calibrated for ~1.9 payload bits/scalar on unit-variance post-RHT rows
// (Philox Gaussian corpus, dim 256, measured on RTX 5090); kept slightly
// under the rate-matched value to leave saturation headroom.
inline constexpr float kHqAlpha = 1.45f;

// ---- zig-zag + E8int (2*E8) exact nearest point ---------------------------

// ---- subtractive dither -----------------------------------------------------
//
// Long-window bias remedy (HyperQuant Table 6 class): the fixed-rate E8 row's
// quantization error is data-DEPENDENT, so per-vector bias compounds over
// hundreds of thousands of distractor rows. Subtracting a pseudo-random dither
// d in [-0.5, 0.5)^8 per (row, word) before the lattice nearest-point and
// adding it back at decode makes the reconstruction error zero-mean and
// independent of the row. d is derived from a counter hash of
// (kv_head, position, role, word) — regenerated identically at every encode
// and decode site, never stored. The dither is ABSOLUTE (not rescaled by the
// escalation halvings): encode quantizes u*2^-e - d, decode reconstructs
// (y + d)*2^e, so every attempt level agrees. Never-written rows (zeroed
// metadata) still decode to exact zeros; the terminal fallback row decodes to
// bounded dither noise instead of exact zero (post-N2 censuses hit it zero
// times on real corpora).

__device__ __forceinline__ std::uint64_t hq_dither_row_seed(int kv_head, std::int64_t position,
                                                            bool role_v) {
    std::uint64_t x = 0x5DEECE66Dull ^
                      (static_cast<std::uint64_t>(kv_head) * 0x2545F4914F6CDD1Dull) ^
                      (static_cast<std::uint64_t>(position) * 0x9E3779B97F4A7C15ull) ^
                      (role_v ? 0xA5A5A5A5A5A5A5A5ull : 0x1B873593B5244C61ull);
    x ^= x >> 33;
    x *= 0xFF51AFD7ED558CCDull;
    x ^= x >> 33;
    return x;
}

__device__ __forceinline__ std::uint64_t hq_dither_word_seed(std::uint64_t row_seed, int word) {
    std::uint64_t x =
        row_seed ^ (0x9E3779B97F4A7C15ull * (static_cast<std::uint64_t>(word) + 1u));
    x ^= x >> 30;
    x *= 0xBF58476D1CE4E5B9ull;
    x ^= x >> 27;
    x *= 0x94D049BB133111EBull;
    x ^= x >> 31;
    return x;
}

// Coordinate dither from byte j of the word seed: 8 bits of entropy per
// coordinate. Scale calibration: the classical full-Voronoi-cell dither
// (spacing 2 -> [-1, 1)) explodes the fixed-budget escalation rate (17% of
// corpus rows at alpha/2, net cosine 0.910); the half-cell dither below
// removes most of the data-dependent bias while keeping the rescue rate at
// ~1.5% and the pooled cosine at 0.936.
__device__ __forceinline__ float hq_dither(std::uint64_t word_seed, int j) {
    const std::uint32_t bits =
        static_cast<std::uint32_t>((word_seed >> (8 * j)) & 0xFFull);
    return static_cast<float>(bits) * (1.0f / 255.0f) - 0.5f;
}

__device__ __forceinline__ std::int32_t hq_zigzag(std::int32_t v) {
    return v >= 0 ? (static_cast<std::uint32_t>(v) << 1)
                  : ((~static_cast<std::uint32_t>(v)) << 1) + 1u;
}

__device__ __forceinline__ std::int32_t hq_unzigzag(std::uint32_t z) {
    return (z & 1u) ? -static_cast<std::int32_t>((z + 1u) >> 1)
                    : static_cast<std::int32_t>(z >> 1);
}

__device__ __forceinline__ int hq_rint(float x) {
    return static_cast<int>(nearbyintf(x));
}

// Force sum(u) == 0 (mod 4) for a same-parity vector by one +/-2 flip at the
// least-confident coordinate. Exact for both E8int cosets.
template <int N>
__device__ __forceinline__ void hq_fix_parity_mod4(int (&u)[N], const float (&x)[N]) {
    int sum = 0;
#pragma unroll
    for (int i = 0; i < N; ++i) sum += u[i];
    if ((sum & 3) == 0) { return; }
    int best_j = 0;
    float best_abs = -1.0f;
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const float delta = fabsf(static_cast<float>(u[i]) - x[i]);
        if (delta > best_abs) {
            best_abs = delta;
            best_j   = i;
        }
    }
    u[best_j] += (static_cast<float>(u[best_j]) >= x[best_j]) ? -2 : 2;
}

// Exact nearest point of 2*E8 = { y in Z^8 : shared parity, sum == 0 mod 4 }:
// nearest even-coset and nearest odd-coset candidates, each parity-fixed,
// closer one wins. The odd coset is the doubled half-integer coset of E8.
__device__ __forceinline__ void hq_quantize_e8int(const float (&x)[8], int (&y)[8]) {
    int u0[8], u1[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        u0[i] = hq_rint(x[i] * 0.5f) * 2;
        u1[i] = hq_rint((x[i] - 1.0f) * 0.5f) * 2 + 1;
    }
    hq_fix_parity_mod4(u0, x);
    hq_fix_parity_mod4(u1, x);
    float d0 = 0.0f, d1 = 0.0f;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const float e0 = static_cast<float>(u0[i]) - x[i];
        const float e1 = static_cast<float>(u1[i]) - x[i];
        d0 += e0 * e0;
        d1 += e1 * e1;
    }
    const int* pick = (d0 <= d1) ? u0 : u1;
#pragma unroll
    for (int i = 0; i < 8; ++i) { y[i] = pick[i]; }
}

// Structural strip (E8): 8 symbols from a valid lattice word.
__device__ __forceinline__ void hq_strip_word(const int (&y)[8], std::uint32_t (&z)[8]) {
    const std::uint32_t c = static_cast<std::uint32_t>(y[0] & 1);
    int s[8];
    int p = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) { s[i] = (y[i] - static_cast<int>(c)) >> 1; }
#pragma unroll
    for (int i = 0; i < 7; ++i) { p += s[i]; }
    const int t = (s[7] - (p & 1)) >> 1;
#pragma unroll
    for (int i = 0; i < 7; ++i) { z[i] = hq_zigzag(s[i]); }
    z[7] = 2u * hq_zigzag(t) + c;
}

__device__ __forceinline__ void hq_unstrip_word(const std::uint32_t (&z)[8], int (&y)[8]) {
    int s[8];
    int p = 0;
#pragma unroll
    for (int i = 0; i < 7; ++i) { s[i] = hq_unzigzag(z[i]); }
#pragma unroll
    for (int i = 0; i < 7; ++i) { p += s[i]; }
    const int t   = hq_unzigzag(z[7] >> 1);
    const int c   = static_cast<int>(z[7] & 1u);
    s[7]          = (t << 1) + (p & 1);
#pragma unroll
    for (int i = 0; i < 8; ++i) { y[i] = (s[i] << 1) + c; }
}

// ---- Rice bit primitives (row-local, word-aligned) -------------------------

struct HqBitReader {
    const std::uint32_t* words;
    int n_words;
    int word_idx;
    std::uint32_t c1, c2;
    int c1_bits, c2_bits;
    std::uint64_t reg;
    int bits_in_reg;

    __device__ HqBitReader(const std::uint32_t* w, int n)
        : words(w), n_words(n), c1(0u), c2(0u), c1_bits(0), c2_bits(0), reg(0ull),
          bits_in_reg(0) {
        word_idx = 2;
        c1       = (0 < n) ? w[0] : 0u;
        c2       = (1 < n) ? w[1] : 0u;
        c1_bits  = (0 < n) ? 32 : 0;
        c2_bits  = (1 < n) ? 32 : 0;
    }

    __device__ __forceinline__ void promote() {
        c1       = c2;
        c1_bits  = c2_bits;
        c2       = (word_idx < n_words) ? words[word_idx] : 0u;
        c2_bits  = 32;
        ++word_idx;
    }

    __device__ __forceinline__ void refill(int needed) {
        while (bits_in_reg < needed) {
            if (c1_bits == 0) { promote(); }
            const int room = 64 - bits_in_reg;
            const int take = (c1_bits < room) ? c1_bits : room;
            const std::uint32_t piece = c1 >> (32 - take);
            reg |= static_cast<std::uint64_t>(piece) << (64 - bits_in_reg - take);
            c1       = (take >= 32) ? 0u : (c1 << take);
            c1_bits -= take;
            bits_in_reg += take;
        }
    }

    __device__ __forceinline__ std::uint32_t read_bits(int n) {
        if (n <= 0) { return 0u; }
        refill(n);
        const std::uint32_t v = static_cast<std::uint32_t>(reg >> (64 - n));
        reg <<= n;
        bits_in_reg -= n;
        return v;
    }

    // Unary quotient via CLZ over the 64-bit register; corrupt all-zero runs
    // return the guard instead of walking.
    __device__ __forceinline__ std::uint32_t read_unary() {
        std::uint32_t q = 0u;
        while (true) {
            refill(1);
            std::uint64_t probe = reg;
            if (bits_in_reg < 64) {
                probe |= (1ull << (64 - bits_in_reg)) - 1ull;
            }
            const std::uint32_t lz =
                (probe == 0ull) ? 64u : static_cast<std::uint32_t>(__clzll(probe));
            q += lz;
            if (lz < static_cast<std::uint32_t>(bits_in_reg)) {
                const int consumed = static_cast<int>(lz) + 1;
                reg       = (consumed >= 64) ? 0ull : (reg << consumed);
                bits_in_reg -= consumed;
                return q;
            }
            reg         = 0ull;
            bits_in_reg = 0;
            if (q > kHqUnaryGuard) { return kHqUnaryGuard; }
        }
    }
};

// ---- warp-cooperative row codec -------------------------------------------
//
// One warp encodes or decodes one row. The encode path stages the rotated,
// alpha-scaled coordinates in shared memory (`u_scaled`, 256 floats) because
// the FWHT lives in per-lane registers while the E8 words need 8 consecutive
// coordinates in one thread.

constexpr int kHqSmemFloatsPerRow = kHqHeadDim;         // 256
constexpr int kHqSmemSymbolsPerRow = kHqHeadDim;        // 256 uint32

// In-register FWHT256 with sign pre-multiply (forward) for one row held as
// 8 register slots per lane (element e = slot*32 + lane). All shuffles run
// on the converged full warp.
__device__ __forceinline__ void hq_fwht256_sign(float (&reg)[8], const std::int8_t* signs,
                                                int sign_base, int lane) {
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        reg[s] *= static_cast<float>(signs[sign_base + s * 32 + lane]);
    }
#pragma unroll
    for (int len = 1; len < 32; len <<= 1) {
        const bool low = (lane & len) == 0;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const float partner = __shfl_xor_sync(0xFFFFFFFFu, reg[s], len);
            reg[s] = low ? (reg[s] + partner) : (partner - reg[s]);
        }
    }
#pragma unroll
    for (int ls = 1; ls < 8; ls <<= 1) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if ((s & ls) == 0) {
                const float a = reg[s];
                const float b = reg[s + ls];
                reg[s]   = a + b;
                reg[s + ls] = a - b;
            }
        }
    }
    const float inv = rsqrtf(static_cast<float>(kHqHeadDim));
#pragma unroll
    for (int s = 0; s < 8; ++s) { reg[s] *= inv; }
}

// Inverse rotation: butterfly first, then signs and 1/sqrt(256).
__device__ __forceinline__ void hq_ifwht256_sign(float (&reg)[8], const std::int8_t* signs,
                                                 int sign_base, int lane) {
#pragma unroll
    for (int len = 1; len < 32; len <<= 1) {
        const bool low = (lane & len) == 0;
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            const float partner = __shfl_xor_sync(0xFFFFFFFFu, reg[s], len);
            reg[s] = low ? (reg[s] + partner) : (partner - reg[s]);
        }
    }
#pragma unroll
    for (int ls = 1; ls < 8; ls <<= 1) {
#pragma unroll
        for (int s = 0; s < 8; ++s) {
            if ((s & ls) == 0) {
                const float a = reg[s];
                const float b = reg[s + ls];
                reg[s]   = a + b;
                reg[s + ls] = a - b;
            }
        }
    }
    const float inv = rsqrtf(static_cast<float>(kHqHeadDim));
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        reg[s] *= inv * static_cast<float>(signs[sign_base + s * 32 + lane]);
    }
}

// Encode one bf16 row (256 dims, contiguous) into codes[64] + meta[8].
// Scratch: u_scaled (256 floats) and syms (256 uint32) in shared memory,
// provided by the caller. One warp per row; lane 0 commits the outputs.
// dither_seed (from hq_dither_row_seed) subtracts the per-word dither before
// the lattice nearest-point; the decoders add it back.
__device__ __forceinline__ void hq_encode_row_warp(const __nv_bfloat16* row,
                                                   const std::int8_t* signs, int sign_base,
                                                   float* u_scaled, std::uint32_t* syms,
                                                   std::uint8_t* codes_out,
                                                   std::uint8_t* meta_out,
                                                   std::uint64_t dither_seed) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    float reg[8];
    float sumsq = 0.0f;
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        const int e = s * 32 + lane;
        reg[s] = __bfloat162float(row[e]);
        sumsq += reg[s] * reg[s];
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        sumsq += __shfl_down_sync(0xFFFFFFFFu, sumsq, off);
    }
    // __shfl_down leaves the full sum only on lane 0; broadcast it so every
    // lane scales with the SAME row norm.
    sumsq = __shfl_sync(0xFFFFFFFFu, sumsq, 0);
    const float norm = sqrtf(fmaxf(sumsq, 1e-30f));
    const float scale = kHqAlpha * sqrtf(static_cast<float>(kHqHeadDim)) / norm;
    hq_fwht256_sign(reg, signs, sign_base, lane);
#pragma unroll
    for (int s = 0; s < 8; ++s) {
        u_scaled[s * 32 + lane] = reg[s] * scale;
    }
    __syncwarp();

    int escalation = 0;
    int used_bits  = -1;
    int seg_off[3] = {0, 0, 0};
    const std::uint32_t best_k = 0;  // k=0 invariant; see the packer below
#pragma unroll 1
    for (int attempt = 0; attempt < 3; ++attempt) {
        if (attempt > 0) {
            // Each retry HALVES the effective alpha (cumulative 1/2x, 1/4x):
            // shrinking the staged coordinates shrinks the lattice symbols, so
            // a retry strictly reduces the bit count and an overflowing row is
            // rescued. (Doubling would make things strictly worse — a row fits
            // iff sum(z) <= 256, and doubling gives sum ~2x.) The decode side
            // multiplies by 1<<escalation to match.
#pragma unroll
            for (int i = lane * 8; i < lane * 8 + 8; ++i) { u_scaled[i] *= 0.5f; }
            __syncwarp();
        }
        // lane w owns lattice word w: quantize + strip into registers.
        std::uint32_t z8[8];
        {
            float x8[8];
            int y8[8];
            const std::uint64_t wseed = hq_dither_word_seed(dither_seed, lane);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                x8[j] = u_scaled[lane * 8 + j] - hq_dither(wseed, j);
            }
            hq_quantize_e8int(x8, y8);
            hq_strip_word(y8, z8);
        }
        // Parallel pack. Rice k = 0 is invariant for every storable row (a
        // row that fits at all fits at k=0, and this encoder always selects
        // the minimum-bit k), so each symbol is z unary zeros plus one
        // terminator 1: the whole row is zero words with 256 set bits, and
        // packing reduces to a per-lane length prefix-sum plus one atomicOr
        // per symbol into a zeroed staging row. The row total is known
        // BEFORE any write, so budget overflow needs no rollback. This
        // replaces the former lane-0 bit-serial writer, whose dynamically
        // indexed word buffer lived in local memory (47k LDL/STL per
        // instantiation in SASS).
        int len8 = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) { len8 += static_cast<int>(z8[j]) + 1; }
        int incl = len8;
#pragma unroll
        for (int d = 1; d < 32; d <<= 1) {
            const int up = __shfl_up_sync(0xFFFFFFFFu, incl, d);
            if (lane >= d) { incl += up; }
        }
        const int total = __shfl_sync(0xFFFFFFFFu, incl, 31);
        const int base  = incl - len8;
        if (total <= kHqRowBudgetBytes * 8) {
            escalation = attempt;
            used_bits  = total;
            seg_off[0] = __shfl_sync(0xFFFFFFFFu, base, 8);
            seg_off[1] = __shfl_sync(0xFFFFFFFFu, base, 16);
            seg_off[2] = __shfl_sync(0xFFFFFFFFu, base, 24);
            // Stage the bit row in the first 16 words of the symbol scratch.
            if (lane < kHqRowBudgetBytes / 4) { syms[lane] = 0u; }
            __syncwarp();
            int pos = base;
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                pos += static_cast<int>(z8[j]);
                atomicOr(&syms[pos >> 5], 1u << (31 - (pos & 31)));
                ++pos;
            }
            __syncwarp();
            break;
        }
    }
    __syncwarp();

    if (lane == 0) {
        __half h = __float2half_rn(norm);
        std::uint16_t norm_bits = *reinterpret_cast<std::uint16_t*>(&h);
        meta_out[0] = static_cast<std::uint8_t>(norm_bits & 0xFFu);
        meta_out[1] = static_cast<std::uint8_t>(norm_bits >> 8);
        std::uint32_t bits;
        if (used_bits >= 0) {
            bits = static_cast<std::uint32_t>(used_bits);
            meta_out[2] = static_cast<std::uint8_t>(best_k |
                                                    (static_cast<std::uint32_t>(escalation) << 4));
#pragma unroll
            for (int b = 0; b < kHqRowBudgetBytes; ++b) {
                codes_out[b] = reinterpret_cast<const std::uint8_t*>(syms)[b];
            }
            meta_out[5] = static_cast<std::uint8_t>(seg_off[0]);
            meta_out[6] = static_cast<std::uint8_t>(seg_off[1]);
            meta_out[7] = static_cast<std::uint8_t>(seg_off[2]);
            meta_out[4] = static_cast<std::uint8_t>(
                ((used_bits >> 8) & 3u) | (((seg_off[0] >> 8) & 1u) << 2) |
                (((seg_off[1] >> 8) & 1u) << 3) | (((seg_off[2] >> 8) & 1u) << 4));
        } else {
            // Terminal fallback: all-zero lattice codes. 256 zero symbols at
            // k=0 are 256 one-bit codes = 32 bytes of 0xFF; every decoded
            // value is exactly zero regardless of norm or escalation.
            bits = 256u;
            meta_out[2] = 0u;
            meta_out[4] = 1u; // used = 256; offsets 64/128/192 fit 8 bits
#pragma unroll
            for (int b = 0; b < kHqRowBudgetBytes; ++b) {
                codes_out[b] = (b < 32) ? 0xFFu : 0x00u;
            }
            meta_out[5] = 64u;
            meta_out[6] = 128u;
            meta_out[7] = 192u;
        }
        meta_out[3] = static_cast<std::uint8_t>(bits & 0xFFu);
    }
}

// ---- engine-global RHT diagonal --------------------------------------------
//
// Deterministic engine-wide sign vector shared by the K and V roles (a single
// shared diagonal provides the isotropy the quantizer needs). Every frame
// boundary — fill encode, scratch decode, and the rotated-frame prompt
// attention kernel — must use this same diagonal.

__device__ __forceinline__ float hq_engine_sign(std::int32_t d) {
    std::uint32_t x = 0x5EED01u ^ (static_cast<std::uint32_t>(d) * 0x9E3779B9u);
    x ^= x >> 16;
    x *= 0x85EBCA6Bu;
    x ^= x >> 13;
    return (x & 1u) ? 1.0f : -1.0f;
}

// Fill signs[0..kHqHeadDim) from any block width (strided over blockDim.x).
__device__ __forceinline__ void hq_engine_signs_fill(std::int8_t* signs) {
    for (int i = static_cast<int>(threadIdx.x); i < kHqHeadDim;
         i += static_cast<int>(blockDim.x)) {
        signs[i] = static_cast<std::int8_t>(hq_engine_sign(i));
    }
}

// ---- residual side-plane rows -----------------------------------------------
//
// The residual window keeps exact bf16 K/V rows for the sink and recent
// positions in the codec's ROTATED frame (both roles: every consumer runs QK
// and PV over rotated rows and un-rotates only the output), laid out per
// (slot row) as [kGqaHqSinkKeys + kGqaHqRecentKeys][KVHeads][256]. One row is
// `key`'s side row: sink rows hold absolute key < kGqaHqSinkKeys directly;
// every other key lives in the recent ring slot key & (kGqaHqRecentKeys - 1).

template <typename Geometry>
__device__ __forceinline__ std::int64_t hq_residual_slot_stride() {
    return static_cast<std::int64_t>(kGqaHqSinkKeys + kGqaHqRecentKeys) * Geometry::KVHeads *
         kHqHeadDim;
}

// Side-plane row pointer for one (slot, kv_head, key). `slot` is the block-table
// row index; layer views arrive pre-sliced to their slot and pass 0. The pointer
// type is deduced (const for fetches, mutable for the dual-write appends).
template <typename Geometry, typename Bf16>
__device__ __forceinline__ Bf16* hq_residual_row(Bf16* plane, std::int32_t slot,
                                                 std::int32_t kv_head, std::int32_t key) {
    const std::int32_t row =
        key < static_cast<std::int32_t>(kGqaHqSinkKeys)
            ? key
            : static_cast<std::int32_t>(kGqaHqSinkKeys) +
                  (key & (static_cast<std::int32_t>(kGqaHqRecentKeys) - 1));
    return plane + static_cast<std::int64_t>(slot) * hq_residual_slot_stride<Geometry>() +
         (static_cast<std::int64_t>(row) * Geometry::KVHeads + kv_head) * kHqHeadDim;
}

// Recent-ring validity bit test: `ring_valid` holds four u32 words per slot row
// (bit r = ring slot r holds the row the next fetch will name). A null bitmap
// means every ring slot is valid (pre-filled planes; the standalone gates).
__device__ __forceinline__ bool hq_ring_slot_valid(const std::uint32_t* ring_valid,
                                                   std::int32_t key) {
    if (ring_valid == nullptr) { return true; }
    const int r = key & (static_cast<int>(kGqaHqRecentKeys) - 1);
    return (ring_valid[r >> 5] >> (r & 31)) & 1u;
}

// Store one exact bf16 row into a side plane in the rotated frame: bf16 ->
// FP32 FWHT(+signs) -> bf16, the same single rounding the encode staging
// applies before quantization. One full warp per row; 8 scattered 2-byte
// stores per lane (append paths only — not on the decode fetch path).
__device__ __forceinline__ void hq_store_rotated_row_warp(const __nv_bfloat16* src,
                                                          const std::int8_t* signs,
                                                          __nv_bfloat16* dst) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    float reg[8];
#pragma unroll
    for (int s = 0; s < 8; ++s) { reg[s] = __bfloat162float(src[s * 32 + lane]); }
    hq_fwht256_sign(reg, signs, 0, lane);
#pragma unroll
    for (int s = 0; s < 8; ++s) { dst[s * 32 + lane] = __float2bfloat16(reg[s]); }
}

// Mark one appended key's ring slot valid (idempotent; one atomic per row).
// Sink rows need no bit — they are immutable once written and always precede
// any fetch that names them.
__device__ __forceinline__ void hq_ring_mark_valid(std::uint32_t* ring_valid,
                                                   std::int32_t key) {
    if (ring_valid == nullptr || key < static_cast<std::int32_t>(kGqaHqSinkKeys)) { return; }
    const int r = key & (static_cast<int>(kGqaHqRecentKeys) - 1);
    atomicOr(&ring_valid[r >> 5], 1u << (r & 31));
}


// Decode one row's codes[64] + meta[8] into 256 bf16 values (rotated frame).
// ONE THREAD per row: the Rice scan is strictly serial within the row, so
// the engine stages tiles by giving each thread one row (the measured
// one-thread-per-row shape from the standalone codec benchmarks). dither_seed
// must equal the encode-side seed for the same (kv_head, position, role).
__device__ __forceinline__ void hq_decode_row_thread(const std::uint8_t* codes,
                                                     const std::uint8_t* meta,
                                                     __nv_bfloat16* out,
                                                     std::uint64_t dither_seed) {
    const std::uint16_t norm_bits = static_cast<std::uint16_t>(meta[0]) |
                                    (static_cast<std::uint16_t>(meta[1]) << 8);
    __half h;
    *reinterpret_cast<std::uint16_t*>(&h) = norm_bits;
    const float norm = __half2float(h);
    const std::uint32_t k = meta[2] & 0x0Fu;
    const std::uint32_t escalation = (meta[2] >> 4) & 0x3u;
    const float inv_scale =
        static_cast<float>(1u << escalation) /
        (kHqAlpha * sqrtf(static_cast<float>(kHqHeadDim)));

    const unsigned used_bits =
        static_cast<unsigned>(meta[3]) | (static_cast<unsigned>(meta[4] & 0x3) << 8);
    if (used_bits == 0 || k > kHqMaxRiceK) {
        // Never-written row (zeroed metadata): decodes to exact zeros.
#pragma unroll 1
        for (int i = 0; i < kHqHeadDim; ++i) { out[i] = __float2bfloat16(0.0f); }
        return;
    }
    HqBitReader br(reinterpret_cast<const std::uint32_t*>(codes),
                   static_cast<int>((used_bits + 31) >> 5));
    std::uint32_t z[8];
#pragma unroll 1
    for (int w = 0; w < kHqWordsPerRow; ++w) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const std::uint32_t q = br.read_unary();
            const std::uint32_t r = br.read_bits(static_cast<int>(k));
            z[j] = (q >= kHqUnaryGuard) ? 0u : ((q << k) | r);
        }
        int y[8];
        hq_unstrip_word(z, y);
        const std::uint64_t wseed = hq_dither_word_seed(dither_seed, w);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            out[w * 8 + j] = __float2bfloat16(
                (static_cast<float>(y[j]) + hq_dither(wseed, j)) * norm * inv_scale);
        }
    }
}

// ---- 8-lane cooperative row decode -----------------------------------------
//
// Eight lanes decode one row. Each lane owns one 64-bit window of the Rice
// stream (two u32 loads composed MSB-first), and the group resolves symbol
// boundaries
// with a speculative scan plus a sequential fixup chain: window j's parse
// depends only on the entry state handed over by window j-1, so the critical
// path is eight ~26-symbol window scans and their shuffles instead of one
// 256-symbol serial walk (the 4-way segment decoder's chain is 64 symbols per
// thread with per-word global promotes on the chain). Zigzag symbols are
// staged as u16 codes into the output row itself; after a group barrier each
// lane unstrips four complete E8 words reassembled from the staged row, so
// lattice words split across lane boundaries need no exchange. Produces
// bit-identical values to hq_decode_row_thread.
//
// Contract: the eight lanes [gbase, gbase+8) of the warp are converged and
// active when called; `lane` is the group-local lane index (0..7).

// k bits at window positions [p, p+k), p <= 64; bits past the window come
// from w_next (zeros beyond the row budget).
__device__ __forceinline__ std::uint32_t hq_group_read(unsigned long long w,
                                                       unsigned long long w_next, int p,
                                                       std::uint32_t k) {
    if (k == 0) { return 0u; }
    if (p + static_cast<int>(k) <= 64) {
        return static_cast<std::uint32_t>((w << p) >> (64 - k));
    }
    const int here = 64 - p;
    const std::uint32_t lo =
        here > 0 ? static_cast<std::uint32_t>((w << p) >> (64 - here)) : 0u;
    const std::uint32_t hi = static_cast<std::uint32_t>(w_next >> (64 - (k - here)));
    return (lo << (k - here)) | hi;
}

// Element index remap for decoding a row directly into an XOR-swizzled tile
// (gqa_small_t_tc_swz): the swizzle moves whole 8-element chunks (chunk ^
// xor_chunk) and keeps each chunk's 8 elements contiguous, so both the u16
// symbol staging and the final bf16 values land at ldmatrix-visible positions
// through the same map. xor_chunk = 0 is the linear layout.
__device__ __forceinline__ int hq_swz_element(int e, int xor_chunk) {
    return e ^ (xor_chunk << 3);  // flips only bits 3..5: the 8-element chunk index
}

// Walk one 64-bit window from an entry state: entry >= 0 with boundary=true
// means a symbol starts at that bit; boundary=false means an unary run from a
// previous window is pending (find its terminator first, own nothing there).
// At most `cap` symbols are taken (the row decode stops at 256 symbols, so
// trailing padding never spawns phantom symbols). Reports the state handed to
// the next window and how many symbols START in this window (a symbol whose
// terminator lies beyond the window still counts here; its owner decodes it
// by chasing later windows).
__device__ __forceinline__ void hq_group_scan_window(unsigned long long w, std::uint32_t k,
                                                     int entry, bool boundary, int cap,
                                                     int& exit_pos, int& exit_boundary,
                                                     int& count) {
    int cursor = boundary ? entry : -1;
    count = 0;
    while (cursor < 64 && count < cap) {
        const int from = cursor >= 0 ? cursor : 0;
        const unsigned long long tail = w << from;
        if (tail == 0ull) {
            if (cursor >= 0) { ++count; }
            exit_pos      = 0;
            exit_boundary = 0;
            return;
        }
        const int t = from + __clzll(tail);
        if (cursor >= 0) { ++count; }
        cursor = t + 1 + static_cast<int>(k);
    }
    if (cursor < 64) {
        // Cap reached mid-window: later windows own nothing (their cap is 0),
        // so the hand-off state is never consumed.
        exit_pos      = cursor;
        exit_boundary = 1;
    } else {
        exit_pos      = cursor - 64;
        exit_boundary = 1;
    }
}

//
// `symbols` (fork 2026-09-10, cometkim feat/1m-context): where the u16 symbol
// codes are staged. By default they are staged into the output row itself; a
// caller whose output is device memory passes an on-chip buffer of kHqHeadDim
// u16 per group instead, so only the reconstructed BF16 row reaches memory.
// `Packed` moves each lattice word -- one aligned 16-byte vector in both the
// staged codes and the output -- as one vector instead of eight scalars; it
// needs 16-byte aligned `out` and `symbols`. Neither changes a value.
template <bool Packed = false>
__device__ __forceinline__ void hq_decode_row_group(const std::uint8_t* codes,
                                                    const std::uint8_t* meta,
                                                    __nv_bfloat16* out, int lane,
                                                    int xor_chunk, std::uint64_t dither_seed,
                                                    std::uint16_t* symbols = nullptr) {
    if (symbols == nullptr) { symbols = reinterpret_cast<std::uint16_t*>(out); }
    const std::uint32_t mw0 = *reinterpret_cast<const std::uint32_t*>(meta);
    const std::uint32_t mw1 = *reinterpret_cast<const std::uint32_t*>(meta + 4);
    const unsigned used_bits = ((mw0 >> 24) & 0xFFu) | ((mw1 & 0x3u) << 8);
    const std::uint32_t k = (mw0 >> 16) & 0xFu;
    if (used_bits == 0 || k > kHqMaxRiceK) {
        // Never-written row (zeroed metadata): decodes to exact zeros.
#pragma unroll 1
        for (int i = lane; i < kHqHeadDim; i += 8) {
            out[hq_swz_element(i, xor_chunk)] = __float2bfloat16(0.0f);
        }
        return;
    }

    const int warp_lane = static_cast<int>(threadIdx.x & 31u);
    const int gbase     = warp_lane & ~7;
    const unsigned gmask = 0xFFu << gbase;

    // Stream layout: 16 u32 words; each word's VALUE carries its 32 stream
    // bits MSB-first (the encoder's staging row holds u32 values stored
    // little-endian). Lane L's 64-bit window = stream bits [64L, 64L+64) =
    // words {2L, 2L+1} concatenated MSB-first.
    const std::uint32_t* w32 = reinterpret_cast<const std::uint32_t*>(codes);
    const unsigned long long w =
        (static_cast<unsigned long long>(w32[2 * lane]) << 32) | w32[2 * lane + 1];

    int idx_total;
    if (k == 0) {
        // Unary fast path. Every stored row is k = 0 (a row that fits the
        // 512-bit budget at all fits at k = 0, and the encoder selects the
        // minimum-bit k), so every 1-bit terminates exactly one symbol and
        // symbol boundaries are prefix-sum computable — no serial parse.
        const int c = static_cast<int>(__popcll(w));
        // Exclusive prefix sum of per-window symbol counts -> this window's
        // first symbol index.
        int inc = c;
#pragma unroll
        for (int delta = 1; delta < 8; delta <<= 1) {
            const int left = __shfl_up_sync(gmask, inc, delta, 8);
            if (lane >= delta) { inc += left; }
        }
        const int base = inc - c;
        idx_total      = __shfl_sync(gmask, inc, gbase + 7);
        // Segmented scan of (all-zero, trailing zeros): carry-in zeros since
        // the last 1-bit of the previous windows (an open unary run).
        int run = (w == 0ull) ? 64 : (static_cast<int>(__ffsll(w)) - 1);
        int az  = (w == 0ull) ? 1 : 0;
#pragma unroll
        for (int delta = 1; delta < 8; delta <<= 1) {
            const int lrun = __shfl_up_sync(gmask, run, delta, 8);
            const int laz  = __shfl_up_sync(gmask, az, delta, 8);
            if (lane >= delta && az != 0) {
                run += lrun;
                az &= laz;
            }
        }
        int carry = __shfl_up_sync(gmask, run, 1, 8);
        if (lane == 0) { carry = 0; }
        // Decode this window's symbols: z = zeros before each 1-bit.
        std::uint16_t* out16 = symbols;
        unsigned long long v = w;
        int n                = 0;
        while (v != 0ull && base + n < kHqHeadDim) {
            const int p = __clzll(v);
            int z       = carry + p;
            if (z >= static_cast<int>(kHqUnaryGuard)) { z = 0; }
            out16[hq_swz_element(base + n, xor_chunk)] = static_cast<std::uint16_t>(z);
            ++n;
            v = (v << p) << 1;  // two shifts: p can be 63
            carry = 0;
        }
    } else {
        // General-k fallback (defensive; the in-tree encoder never stores
        // k > 0). Speculative scan + sequential fixup over the eight windows.
        unsigned long long remote[8];
#pragma unroll
        for (int m = 1; m <= 8; ++m) {
            const unsigned long long v = __shfl_sync(gmask, w, gbase + ((lane + m) & 7));
            remote[m - 1] = (lane + m <= 7) ? v : 0ull;
        }
        int exit_pos = 0, exit_cnt = 0, exit_bnd = 1;
        hq_group_scan_window(w, k, 0, true, kHqHeadDim, exit_pos, exit_bnd, exit_cnt);
        int u = 0, idx = 0, bnd = 1;
        int my_e = 0, my_i = 0, my_b = 1;
        for (int j = 0; j < 8; ++j) {
            int eu = 0, ec = 0, eb = 1;
            if (lane == j) {
                my_e = u;
                my_i = idx;
                my_b = bnd;
                const int cap = kHqHeadDim - idx;
                if (cap <= 0) {
                    eu = 0;
                    eb = 1;
                    ec = 0;
                } else if (u != 0 || bnd == 0 || exit_cnt > cap) {
                    hq_group_scan_window(w, k, u, bnd != 0, cap, eu, eb, ec);
                } else {
                    eu = exit_pos;
                    eb = exit_bnd;
                    ec = exit_cnt;
                }
            }
            u   = __shfl_sync(gmask, eu, gbase + j);
            bnd = __shfl_sync(gmask, eb, gbase + j);
            idx += __shfl_sync(gmask, ec, gbase + j);
        }
        idx_total = idx;

        std::uint16_t* out16 = symbols;
        int pending = -1;
        int s       = my_i;
        {
            const int cap = kHqHeadDim - my_i;
            int cursor    = my_b != 0 ? my_e : -1;
            int n         = 0;
            while (cursor < 64 && n < cap) {
                const int from = cursor >= 0 ? cursor : 0;
                const unsigned long long tail = w << from;
                if (tail == 0ull) {
                    if (cursor >= 0) { pending = cursor; }
                    break;
                }
                const int t = from + __clzll(tail);
                if (cursor >= 0) {
                    const std::uint32_t q = static_cast<std::uint32_t>(t - cursor);
                    const std::uint32_t r = hq_group_read(w, remote[0], t + 1, k);
                    out16[hq_swz_element(s, xor_chunk)] = static_cast<std::uint16_t>((q << k) | r);
                    ++s;
                    ++n;
                }
                cursor = t + 1 + static_cast<int>(k);
            }
        }
        if (pending >= 0) {
            std::uint32_t z = 0;
            int d           = 64 - pending;
#pragma unroll
            for (int m = 1; m <= 8; ++m) {
                const unsigned long long pw = remote[m - 1];
                if (pw == 0ull) {
                    if (m == 8) { break; }  // past the row budget: guarded symbol is 0
                    d += 64;
                    continue;
                }
                const int t2 = __clzll(pw);
                d += t2;
                if (d < static_cast<int>(kHqUnaryGuard)) {
                    const std::uint32_t r =
                        hq_group_read(pw, m < 8 ? remote[m] : 0ull, t2 + 1, k);
                    z = (static_cast<std::uint32_t>(d) << k) | r;
                }
                break;
            }
            out16[hq_swz_element(s, xor_chunk)] = static_cast<std::uint16_t>(z);
        }
    }

    // Defensive tail: symbols beyond the last parseable boundary decode to
    // zero, exactly like the sequential reader's unary guard. Well-formed
    // rows always parse 256 symbols, so this never runs in the engine.
    if (idx_total < kHqHeadDim && lane == 0) {
        std::uint16_t* out16 = symbols;
#pragma unroll 1
        for (int t = idx_total; t < kHqHeadDim; ++t) { out16[hq_swz_element(t, xor_chunk)] = 0; }
    }
    __syncwarp(gmask);

    // Unstrip/scale four complete E8 words per lane, reassembled from the
    // staged row (word ownership is static, so lane-boundary words need no
    // cross-lane exchange). The staged u16 codes and the final bf16 values
    // share the same (possibly swizzled) element slots.
    const std::uint32_t escalation = (mw0 >> 20) & 0x3u;
    __half h;
    *reinterpret_cast<std::uint16_t*>(&h) = static_cast<std::uint16_t>(mw0 & 0xFFFFu);
    const float norm = __half2float(h);
    const float inv_scale =
        static_cast<float>(1u << escalation) /
        (kHqAlpha * sqrtf(static_cast<float>(kHqHeadDim)));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int word = lane + 8 * i;
        const int base = hq_swz_element(word * 8, xor_chunk);
        std::uint32_t z[8];
        if constexpr (Packed) {
            const uint4 packed = *reinterpret_cast<const uint4*>(symbols + base);
            const auto* codes16 = reinterpret_cast<const std::uint16_t*>(&packed);
#pragma unroll
            for (int j = 0; j < 8; ++j) { z[j] = codes16[j]; }
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) { z[j] = symbols[base + j]; }
        }
        int y[8];
        hq_unstrip_word(z, y);
        const std::uint64_t wseed = hq_dither_word_seed(dither_seed, word);
        if constexpr (Packed) {
            uint4 packed;
            auto* values = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                values[j] = __float2bfloat16(
                    (static_cast<float>(y[j]) + hq_dither(wseed, j)) * norm * inv_scale);
            }
            *reinterpret_cast<uint4*>(out + base) = packed;
        } else {
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                out[base + j] = __float2bfloat16(
                    (static_cast<float>(y[j]) + hq_dither(wseed, j)) * norm * inv_scale);
            }
        }
    }
}

// ---- 8-lane cooperative row decode, throughput form -------------------------
//
// ignis, ADR 0031 (GitHub #268): hq_decode_row_group<true>'s values from fewer
// instructions, for the hq verify attention kernel's throughput route, whose
// round cost is this decode
// (docs/findings/2026-09-24-hq-verify-decode-cost.md). The arguments, the
// output placement (`out` 16-byte aligned, the row written through
// hq_swz_element) and the symbols staged in `out` itself are
// hq_decode_row_group<true>'s; only a k = 0 row takes a new path, every other
// row (never written, k > kHqMaxRiceK, general k) is hq_decode_row_group<true>.
//   - Warp-converged: all 32 lanes of the warp make the call together, each
//     8-lane group for its own row or `active = false` for none (an inactive
//     group reads and writes nothing), so the group exchanges run on the full
//     warp without the divergence checks an 8-lane mask costs.
//   - A lane walks its window as its two 32-bit stream words, one terminator
//     of each per step: 32-bit clz and a clamped funnel shift in place of
//     64-bit shifts, both chains stepped without branches (a spent word only
//     shifts zeros; only the symbol store is conditional).
//   - Unstrip by bit identities instead of a zigzag branch and a running sum:
//     y = 2*s + c = z ^ -(z & 1) ^ c (2*s is even) for two symbols per 32-bit
//     half pair, and the parity of the sum of s is bit 1 of the xor of the
//     first seven y.
//   - The dither and scale are the reference's own expression per element;
//     the dither byte enters as the float 2^23 + b folded into the reference's
//     single rounding of b/255 - 0.5, so every value takes the same roundings.

// Leading zeros of a word known to be non-zero, in one instruction (clz's
// own lowering adds a subtract). On zero it returns -1, which shifts by 0.
__device__ __forceinline__ int hq_clz_nonzero(std::uint32_t v) {
    std::uint32_t r;
    asm("bfind.shiftamt.u32 %0, %1;" : "=r"(r) : "r"(v));
    return static_cast<int>(r);
}

__device__ __forceinline__ void hq_decode_row_group_fast(const std::uint8_t* codes,
                                                         const std::uint8_t* meta,
                                                         __nv_bfloat16* out, int lane,
                                                         int xor_chunk,
                                                         std::uint64_t dither_seed,
                                                         bool active) {
    const int warp_lane = static_cast<int>(threadIdx.x & 31u);
    const int gbase     = warp_lane & ~7;
    constexpr unsigned kWarp = 0xFFFFFFFFu;

    std::uint32_t mw0 = 0u, mw1 = 0u;
    if (active) {
        mw0 = *reinterpret_cast<const std::uint32_t*>(meta);
        mw1 = *reinterpret_cast<const std::uint32_t*>(meta + 4);
    }
    const unsigned used_bits = ((mw0 >> 24) & 0xFFu) | ((mw1 & 0x3u) << 8);
    const std::uint32_t k    = (mw0 >> 16) & 0xFu;
    const bool fast          = active && used_bits != 0 && k == 0;
    std::uint16_t* symbols   = reinterpret_cast<std::uint16_t*>(out);
    if (active && (used_bits == 0 || k > kHqMaxRiceK)) {
        // Never-written row (zeroed metadata): decodes to exact zeros.
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            *reinterpret_cast<uint4*>(out + hq_swz_element((lane + 8 * i) * 8, xor_chunk)) =
                make_uint4(0u, 0u, 0u, 0u);
        }
    } else if (active && k != 0) {
        hq_decode_row_group<true>(codes, meta, out, lane, xor_chunk, dither_seed);
    }
    // Lane L's window is stream words 2L (x0, the earlier 32 bits) and 2L+1 (x1);
    // a row off this path has no terminators here.
    uint2 xw = make_uint2(0u, 0u);
    if (fast) { xw = *reinterpret_cast<const uint2*>(codes + 8 * lane); }
    const std::uint32_t x0 = xw.x;
    const std::uint32_t x1 = xw.y;

    const int c0 = __popc(x0);
    const int c  = c0 + __popc(x1);
    int inc      = c;
    // Zeros since the last terminator before this window: the reference's
    // segmented scan of (trailing zeros, all-zero) over the windows.
    int run = x1 != 0u ? __ffs(x1) - 1 : (x0 != 0u ? 31 + __ffs(x0) : 64);
    int az  = (x0 | x1) == 0u ? 1 : 0;
#pragma unroll
    for (int delta = 1; delta < 8; delta <<= 1) {
        const int left = __shfl_up_sync(kWarp, inc, delta, 8);
        const int lrun = __shfl_up_sync(kWarp, run, delta, 8);
        const int laz  = __shfl_up_sync(kWarp, az, delta, 8);
        if (lane >= delta) { inc += left; }
        if (lane >= delta && az != 0) {
            run += lrun;
            az &= laz;
        }
    }

    // Symbol i of the row sits at element slot i ^ (xor_chunk << 3), as in the
    // reference. Chain j walks word j; its first symbol also counts the zeros
    // before the word (x1's: x0's trailing zeros, or the carry plus all of x0),
    // under the unary guard. Symbols from index kHqHeadDim on are not stored.
    const int idx_total = __shfl_sync(kWarp, inc, gbase + 7);
    int carry           = __shfl_up_sync(kWarp, run, 1, 8);
    if (lane == 0) { carry = 0; }
    const int base  = inc - c;
    const int swz   = xor_chunk << 3;
    int i0          = base;
    int i1          = base + c0;
    const int e0    = min(base + c0, kHqHeadDim);
    const int e1    = min(base + c, kHqHeadDim);
    std::uint32_t v0 = x0;
    std::uint32_t v1 = x1;
    const int lead1  = x0 != 0u ? __ffs(x0) - 1 : carry + 32;
    if (i0 < e0) {
        const int p = hq_clz_nonzero(v0);
        int z       = carry + p;
        if (z >= static_cast<int>(kHqUnaryGuard)) { z = 0; }
        symbols[i0 ^ swz] = static_cast<std::uint16_t>(z);
        v0 = __funnelshift_lc(0u, v0, p + 1);
        ++i0;
    }
    if (i1 < e1) {
        const int p = hq_clz_nonzero(v1);
        int z       = lead1 + p;
        if (z >= static_cast<int>(kHqUnaryGuard)) { z = 0; }
        symbols[i1 ^ swz] = static_cast<std::uint16_t>(z);
        v1 = __funnelshift_lc(0u, v1, p + 1);
        ++i1;
    }
    // Steady state: both chains step each iteration and only the store is
    // conditional, so the two chains issue interleaved instead of as branches.
    const int steps = max(e0 - i0, e1 - i1);
#pragma unroll 1
    for (int step = 0; step < steps; ++step) {
        const int p0 = hq_clz_nonzero(v0);
        const int p1 = hq_clz_nonzero(v1);
        if (i0 < e0) { symbols[i0 ^ swz] = static_cast<std::uint16_t>(p0); }
        if (i1 < e1) { symbols[i1 ^ swz] = static_cast<std::uint16_t>(p1); }
        v0 = __funnelshift_lc(0u, v0, p0 + 1);
        v1 = __funnelshift_lc(0u, v1, p1 + 1);
        ++i0;
        ++i1;
    }

    // Defensive tail, as in the reference: symbols past the last terminator are zero.
    if (fast && idx_total < kHqHeadDim && lane == 0) {
#pragma unroll 1
        for (int t = idx_total; t < kHqHeadDim; ++t) { symbols[hq_swz_element(t, xor_chunk)] = 0; }
    }
    __syncwarp(kWarp);
    if (!fast) { return; }

    const std::uint32_t escalation = (mw0 >> 20) & 0x3u;
    __half h;
    *reinterpret_cast<std::uint16_t*>(&h) = static_cast<std::uint16_t>(mw0 & 0xFFFFu);
    const float norm = __half2float(h);
    const float inv_scale =
        static_cast<float>(1u << escalation) /
        (kHqAlpha * sqrtf(static_cast<float>(kHqHeadDim)));
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int word     = lane + 8 * i;
        const int wb       = hq_swz_element(word * 8, xor_chunk);
        const uint4 staged = *reinterpret_cast<const uint4*>(symbols + wb);
        const std::uint32_t pairs[4] = {staged.x, staged.y, staged.z, staged.w};
        // Two coordinates per staged word, one per 16-bit half: a k = 0
        // symbol is below kHqUnaryGuard, so every y fits a signed half.
        const std::uint32_t coset  = (pairs[3] >> 16) & 1u;  // z7 & 1
        const std::uint32_t coset2 = coset * 0x00010001u;
        std::uint32_t y2[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y2[j] = pairs[j] ^ ((pairs[j] & 0x00010001u) * 0xFFFFu) ^ coset2;
        }
        const std::uint32_t x      = y2[0] ^ y2[1] ^ y2[2] ^ (y2[3] & 0xFFFFu);  // y0..y6
        const std::uint32_t parity = ((x >> 1) ^ (x >> 17)) & 1u;
        const std::uint32_t u      = pairs[3] >> 17;          // z7 >> 1
        const std::uint32_t t2     = u ^ (0u - (u & 1u));      // 2*unzigzag(z7 >> 1)
        int y[8];
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            y[2 * j]     = static_cast<std::int16_t>(y2[j] & 0xFFFFu);
            y[2 * j + 1] = static_cast<std::int16_t>(y2[j] >> 16);
        }
        y[7] = static_cast<int>(((t2 ^ parity) << 1) ^ coset);
        // (2^23 + b) * c - (2^23 * c + 0.5) is b * c - 0.5 exactly, and
        // 2^23 * c + 0.5 is itself exact.
        const std::uint64_t wseed = hq_dither_word_seed(dither_seed, word);
        const std::uint32_t wlo   = static_cast<std::uint32_t>(wseed);
        const std::uint32_t whi   = static_cast<std::uint32_t>(wseed >> 32);
        constexpr float kInv255   = 1.0f / 255.0f;
        constexpr float kFold     = 8388608.0f * kInv255 + 0.5f;
        uint4 packed;
        auto* values = reinterpret_cast<__nv_bfloat16*>(&packed);
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float magic = __uint_as_float(
                __byte_perm(j < 4 ? wlo : whi, 0x4B000000u, 0x7540u | static_cast<unsigned>(j & 3)));
            const float dither = __fmaf_rn(magic, kInv255, -kFold);
            values[j] = __float2bfloat16((static_cast<float>(y[j]) + dither) * norm * inv_scale);
        }
        *reinterpret_cast<uint4*>(out + wb) = packed;
    }
}

} // namespace ninfer::ops
