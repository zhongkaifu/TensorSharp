// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Weight products ggml-cpu has no fast kernel for on ARM64 (see ggml_ops_cpu_graph.h, cpu_weight_product), computed by
// TensorSharp as GGML_OP_CUSTOM nodes over ggml's own thread pool:
//   - BF16 below prompt width: ggml-cpu's BF16 dot has no NEON body (vec.cpp's ARM build runs the scalar loop, ~12 tok/s
//     decode on a 1.7B model). Here the weight rows are widened to F32 (exact) and multiplied with the F32 activations as
//     they are, every thread on its own rows.
//   - BF16 products of prompt width (from kSgemmRowsBf16 rows, Apple): the weight is widened to F32 by every thread (one
//     graph tensor, row tiles past kTileBytes), then multiplied by one Accelerate sgemm call (Apple's matrix units), as
//     llama.cpp's BLAS backend does for the products it takes.
// Both read F32 activations unrounded (ggml rounds them to BF16).
//   - NVFP4 products of kNvfp4GemmRows rows and more (ARM64 with the dot-product instructions): ggml-cpu's own NVFP4
//     product, a row at a time (nrows = 1: every weight block is decoded again for every activation row), as a GEMM.
//     Four weight rows are decoded once into int8 vectors laid out for SDOT by lane, and tiles of four activation rows run
//     against them with the columns in vector lanes. The activations are ggml's own Q8_0 rows
//     (ggml-cpu's from_float), and every output is ggml_vec_dot_nvfp4_q8_0's NEON arithmetic evaluated lane-wise (exact
//     integer sums per 16-value sub-block, one fused multiply-add per sub-block into its own accumulator, the four
//     accumulators summed as vaddvq_f32 sums them): the same bits as ggml's product, faster.

#include "ggml_ops_cpu_graph.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

// Every ARM64 build: clang and GCC define __aarch64__, MSVC only _M_ARM64 (the guard ggml_ops_cpu_graph.cpp's repack
// default uses too). With the two guards apart, an MSVC ARM64 build repacked the quantized weights but left BF16 weights
// on ggml-cpu's scalar dot.
#if (defined(__aarch64__) && defined(__ARM_NEON)) || defined(_M_ARM64)
#include <arm_neon.h>
#define TSG_CPU_BF16_NEON 1
#endif

#if defined(TSG_GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

// The NVFP4 GEMM needs SDOT by lane (ARMv8.2 dot product; every Apple silicon core has it).
#if defined(__aarch64__) && defined(__ARM_NEON) && defined(__ARM_FEATURE_DOTPROD) && defined(__ARM_FEATURE_FMA)
#define TSG_CPU_NVFP4_SDOT 1
#endif

namespace tsg
{
    namespace
    {
        // Activation rows from which a BF16 product's weight goes through sgemm. The conversion to F32 is paid per product
        // whatever its width, so sgemm pays off from the width where its speed covers it (M5 Pro, LFM3-1.7B at 18 threads,
        // interleaved): the widened weight loses to the NEON rows at 32 (121 vs 185 tok/s) and leads from 64 (210 vs 199;
        // 241 vs 203 at 96, 315 vs 283 at 128). (NVFP4 went this way from 256 rows until its GEMM below outran it at
        // every width: LFM3-1.7B NVFP4 pp512 683 -> 1038 tok/s.)
        constexpr std::int64_t kSgemmRowsBf16 = 64;
        // The F32 copy of a prompt product's weight is at most this big (larger weights go in row tiles).
        constexpr std::int64_t kTileBytes = std::int64_t(128) << 20;

        inline float bf16_to_f32(std::uint16_t h)
        {
            const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
            float f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }

#if defined(TSG_CPU_BF16_NEON)
        // The low / high four BF16 lanes of v as F32, exactly: each 16-bit pattern moved to the top of its 32-bit lane.
        // MSVC gets the ARMv7-era widen-then-shift pair instead of the shift-by-lane-width SHLL forms (the same bits;
        // no MSVC ARM64 toolchain has compiled either here).
        inline float32x4_t bf16_low_f32(uint16x8_t v)
        {
#if defined(_MSC_VER) && !defined(__clang__)
            return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_low_u16(v)), 16));
#else
            return vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16));
#endif
        }

        inline float32x4_t bf16_high_f32(uint16x8_t v)
        {
#if defined(_MSC_VER) && !defined(__clang__)
            return vreinterpretq_f32_u32(vshlq_n_u32(vmovl_u16(vget_high_u16(v)), 16));
#else
            return vreinterpretq_f32_u32(vshll_high_n_u16(v, 16));
#endif
        }
#endif

#if defined(TSG_CPU_NVFP4_SDOT)
        // NVFP4's UE4M3 sub-block scales as floats, read from ggml's own decoding (a block whose first code is 1).
        const float* nvfp4_scales()
        {
            static const std::vector<float> table = [] {
                std::vector<float> t(256);
                const ggml_to_float_t to_float = ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float;
                std::uint8_t block[36];
                float y[64];
                for (int b = 0; b < 256; b++)
                {
                    std::memset(block, 0, sizeof(block));
                    block[0] = static_cast<std::uint8_t>(b);
                    block[4] = 1;   // value 0: code 1 (kvalues_fp4[1] = 1) times the scale
                    to_float(block, y, 64);
                    t[static_cast<std::size_t>(b)] = y[0];
                }
                return t;
            }();
            return table.data();
        }
#endif

#if defined(TSG_GGML_USE_ACCELERATE)
        // `rows` rows of w from `row0`, as F32 (BF16 widened exactly; other types through ggml's to_float).
        void rows_to_f32(const ggml_tensor* w, std::int64_t row0, std::int64_t rows, float* dst)
        {
            const std::int64_t K = w->ne[0];
            const char* base = static_cast<const char*>(w->data) + row0 * w->nb[1];
            if (w->type == GGML_TYPE_BF16)
            {
                for (std::int64_t r = 0; r < rows; r++)
                {
                    const auto* s = reinterpret_cast<const std::uint16_t*>(base + r * w->nb[1]);
                    float* d = dst + r * K;
                    std::int64_t k = 0;
#if defined(TSG_CPU_BF16_NEON)
                    for (; k + 8 <= K; k += 8)
                    {
                        const uint16x8_t v = vld1q_u16(s + k);
                        vst1q_f32(d + k, bf16_low_f32(v));
                        vst1q_f32(d + k + 4, bf16_high_f32(v));
                    }
#endif
                    for (; k < K; k++) d[k] = bf16_to_f32(s[k]);
                }
                return;
            }
            const ggml_to_float_t to_float = ggml_get_type_traits(w->type)->to_float;
            for (std::int64_t r = 0; r < rows; r++)
                to_float(base + r * w->nb[1], dst + r * K, K);
        }
#endif

#if defined(TSG_CPU_BF16_NEON)
        // out[r] = w · x_r for R activation rows (x_r = x + r * ldx): the BF16 row widened exactly, F32 multiply-adds.
        template <int R>
        inline void bf16_dot_rows(const std::uint16_t* w, const float* x, std::int64_t ldx, std::int64_t K, float* out, std::int64_t ldo)
        {
            float32x4_t a[R][4];
            for (int r = 0; r < R; r++)
                for (int j = 0; j < 4; j++) a[r][j] = vdupq_n_f32(0.0f);
            std::int64_t k = 0;
            for (; k + 16 <= K; k += 16)
            {
                const uint16x8_t v0 = vld1q_u16(w + k), v1 = vld1q_u16(w + k + 8);
                const float32x4_t w0 = bf16_low_f32(v0), w1 = bf16_high_f32(v0);
                const float32x4_t w2 = bf16_low_f32(v1), w3 = bf16_high_f32(v1);
                for (int r = 0; r < R; r++)
                {
                    const float* xr = x + r * ldx + k;
                    a[r][0] = vfmaq_f32(a[r][0], w0, vld1q_f32(xr));
                    a[r][1] = vfmaq_f32(a[r][1], w1, vld1q_f32(xr + 4));
                    a[r][2] = vfmaq_f32(a[r][2], w2, vld1q_f32(xr + 8));
                    a[r][3] = vfmaq_f32(a[r][3], w3, vld1q_f32(xr + 12));
                }
            }
            for (int r = 0; r < R; r++)
            {
                float s = vaddvq_f32(vaddq_f32(vaddq_f32(a[r][0], a[r][1]), vaddq_f32(a[r][2], a[r][3])));
                for (std::int64_t kk = k; kk < K; kk++) s += bf16_to_f32(w[kk]) * x[r * ldx + kk];
                out[r * ldo] = s;
            }
        }
#endif

        // dst [M, N] = w [K, M] · x [K, N] below prompt width (BF16): this thread's rows of the result (16-row blocks),
        // each weight row widened once for every activation row.
        void bf16_rows_product(ggml_tensor* dst, int ith, int nth, void*)
        {
#if defined(TSG_CPU_BF16_NEON)
            const ggml_tensor* w = dst->src[0];
            const ggml_tensor* x = dst->src[1];
            const std::int64_t K = w->ne[0], M = w->ne[1], N = x->ne[1] * x->ne[2];
            const std::int64_t ldx = x->nb[1] / static_cast<std::int64_t>(sizeof(float));
            const std::int64_t ldy = dst->nb[1] / static_cast<std::int64_t>(sizeof(float));
            const auto* X = static_cast<const float*>(x->data);
            auto* Y = static_cast<float*>(dst->data);
            const std::int64_t blocks = (M + 15) / 16;
            const std::int64_t m0 = std::min(M, blocks * ith / nth * 16), m1 = std::min(M, blocks * (ith + 1) / nth * 16);
            for (std::int64_t m = m0; m < m1; m++)
            {
                const auto* wr = reinterpret_cast<const std::uint16_t*>(static_cast<const char*>(w->data) + m * w->nb[1]);
                std::int64_t n = 0;
                for (; n + 4 <= N; n += 4) bf16_dot_rows<4>(wr, X + n * ldx, ldx, K, Y + n * ldy + m, ldy);
                switch (N - n)
                {
                case 3: bf16_dot_rows<3>(wr, X + n * ldx, ldx, K, Y + n * ldy + m, ldy); break;
                case 2: bf16_dot_rows<2>(wr, X + n * ldx, ldx, K, Y + n * ldy + m, ldy); break;
                case 1: bf16_dot_rows<1>(wr, X + n * ldx, ldx, K, Y + n * ldy + m, ldy); break;
                default: break;
                }
            }
#else
            (void) dst; (void) ith; (void) nth;
#endif
        }

#if defined(TSG_GGML_USE_ACCELERATE)
        // dst [K, rows] F32 = the weight rows of src[0] (a 2-D weight or row block), widened / dequantized by every
        // thread (16-row blocks).
        void rows_to_f32_op(ggml_tensor* dst, int ith, int nth, void*)
        {
            const ggml_tensor* w = dst->src[0];
            const std::int64_t K = w->ne[0], M = w->ne[1], blocks = (M + 15) / 16;
            const std::int64_t m0 = std::min(M, blocks * ith / nth * 16), m1 = std::min(M, blocks * (ith + 1) / nth * 16);
            if (m0 < m1) rows_to_f32(w, m0, m1 - m0, static_cast<float*>(dst->data) + m0 * K);
        }

        // dst [M, N] = src[0] [K, M] (F32, contiguous) · src[1] [K, N]: one Accelerate call (it spreads the product over
        // Apple's matrix units itself: 2.5 TFLOP/s for 15360 x 2048 x 512 on an M5 Pro, where every ggml thread calling
        // sgemm on its own slice reached 1.8).
        void sgemm_op(ggml_tensor* dst, int ith, int, void*)
        {
            if (ith != 0) return;
            const ggml_tensor* w = dst->src[0];
            const ggml_tensor* x = dst->src[1];
            const int K = static_cast<int>(w->ne[0]), M = static_cast<int>(w->ne[1]), N = static_cast<int>(x->ne[1] * x->ne[2]);
            cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasTrans, N, M, K, 1.0f, static_cast<const float*>(x->data),
                static_cast<int>(x->nb[1] / sizeof(float)), static_cast<const float*>(w->data), K, 0.0f,
                static_cast<float*>(dst->data), static_cast<int>(dst->nb[1] / sizeof(float)));
        }
#endif

#if defined(TSG_CPU_NVFP4_SDOT)
        // Activation rows from which an NVFP4 product runs as the GEMM below. One row stays on ggml's own dot (the same
        // bits; a decode step reads every weight once either way).
        constexpr std::int64_t kNvfp4GemmRows = 2;
        constexpr std::int64_t kNvfp4Block = 64;          // values a block (QK_NVFP4): four 16-value sub-blocks
        constexpr std::size_t kNvfp4BlockBytes = 36;      // 4 UE4M3 scales, 32 bytes of 4-bit codes
        constexpr std::size_t kQ8BlockBytes = 34;         // block_q8_0: an fp16 scale, 32 int8 values

        // dst (Q8_0 [K, N]) = the activation rows x [K, N] as ggml-cpu's MUL_MAT converts them for an NVFP4 weight (its
        // vec_dot_type, Q8_0, through the same from_float).
        void q8_0_rows_op(ggml_tensor* dst, int ith, int nth, void*)
        {
            const ggml_tensor* x = dst->src[0];
            const std::int64_t K = x->ne[0], N = x->ne[1] * x->ne[2];
            const ggml_from_float_t from_float = ggml_get_type_traits_cpu(GGML_TYPE_Q8_0)->from_float;
            for (std::int64_t r = N * ith / nth; r < N * (ith + 1) / nth; r++)
                from_float(reinterpret_cast<const float*>(static_cast<const char*>(x->data) + r * x->nb[1]),
                           static_cast<char*>(dst->data) + r * dst->nb[1], K);
        }

        // Four NVFP4 weight rows (w + c * row_bytes) decoded for SDOT by lane: per block and sub-block, four int8x16 vectors
        // (one a group of four values: lanes 4c..4c+3 hold row c's values) and the sub-block's four UE4M3 scales as a float
        // vector (lane c: row c).
        void nvfp4_decode4(const std::uint8_t* w, std::size_t row_bytes, std::int64_t blocks, int8x16_t* q, float32x4_t* sc)
        {
            static const std::int8_t kvalues[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };   // kvalues_mxfp4
            const int8x16_t lut = vld1q_s8(kvalues);
            const uint8x16_t m4 = vdupq_n_u8(0x0F);
            const float* scales = nvfp4_scales();
            for (std::int64_t b = 0; b < blocks; b++)
            {
                int8x16_t v[4][4];   // [row][sub-block]: the sub-block's 16 values in order
                float s[4][4];       // [sub-block][row]
                for (int c = 0; c < 4; c++)
                {
                    const std::uint8_t* blk = w + c * row_bytes + b * kNvfp4BlockBytes;
                    for (int si = 0; si < 4; si++) s[si][c] = scales[blk[si]];
                    for (int h = 0; h < 2; h++)
                    {
                        // 16 code bytes: sub-blocks 2h and 2h+1, eight bytes each (value j in byte j's low nibble, value
                        // j + 8 in its high nibble).
                        const uint8x16_t codes = vld1q_u8(blk + 4 + 16 * h);
                        const uint8x16_t lo = vandq_u8(codes, m4), hi = vshrq_n_u8(codes, 4);
                        const uint8x16_t first = vreinterpretq_u8_u64(vzip1q_u64(vreinterpretq_u64_u8(lo), vreinterpretq_u64_u8(hi)));
                        const uint8x16_t second = vreinterpretq_u8_u64(vzip2q_u64(vreinterpretq_u64_u8(lo), vreinterpretq_u64_u8(hi)));
                        v[c][2 * h] = vqtbl1q_s8(lut, first);
                        v[c][2 * h + 1] = vqtbl1q_s8(lut, second);
                    }
                }
                for (int si = 0; si < 4; si++)
                {
                    // 4x4 transpose of 32-bit groups: vector g holds group g of every row.
                    const uint32x4_t a0 = vreinterpretq_u32_s8(v[0][si]), a1 = vreinterpretq_u32_s8(v[1][si]);
                    const uint32x4_t a2 = vreinterpretq_u32_s8(v[2][si]), a3 = vreinterpretq_u32_s8(v[3][si]);
                    const uint32x4_t t0 = vtrn1q_u32(a0, a1), t1 = vtrn2q_u32(a0, a1);
                    const uint32x4_t t2 = vtrn1q_u32(a2, a3), t3 = vtrn2q_u32(a2, a3);
                    int8x16_t* out = q + (b * 4 + si) * 4;
                    out[0] = vreinterpretq_s8_u64(vtrn1q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
                    out[1] = vreinterpretq_s8_u64(vtrn1q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
                    out[2] = vreinterpretq_s8_u64(vtrn2q_u64(vreinterpretq_u64_u32(t0), vreinterpretq_u64_u32(t2)));
                    out[3] = vreinterpretq_s8_u64(vtrn2q_u64(vreinterpretq_u64_u32(t1), vreinterpretq_u64_u32(t3)));
                    sc[b * 4 + si] = vld1q_f32(s[si]);
                }
            }
        }

        // R activation rows (Q8_0, x + r * ldx bytes) against four decoded weight rows: y[r * ldy + c] for c < 4. Per row and
        // weight row this is ggml_vec_dot_nvfp4_q8_0's NEON loop: sub-block si's exact integer sum, times (its UE4M3 scale
        // times the activation block's scale), fused into accumulator si; the four accumulators summed pairwise.
        template <int R>
        inline void nvfp4_tile(const int8x16_t* q, const float32x4_t* sc, std::int64_t blocks, const std::uint8_t* x, std::size_t ldx,
                               float* y, std::int64_t ldy)
        {
            float32x4_t acc[R][4];
            for (int r = 0; r < R; r++)
                for (int si = 0; si < 4; si++) acc[r][si] = vdupq_n_f32(0.0f);
            const int32x4_t zero = vdupq_n_s32(0);
            for (std::int64_t b = 0; b < blocks; b++)
            {
                for (int si = 0; si < 4; si++)
                {
                    const int8x16_t* w = q + (b * 4 + si) * 4;
                    const int8x16_t w0 = w[0], w1 = w[1], w2 = w[2], w3 = w[3];
                    const float32x4_t s = sc[b * 4 + si];
                    for (int r = 0; r < R; r++)
                    {
                        // The sub-block's 16 activations: Q8_0 block 2b + si/2, from value (si % 2) * 16.
                        const std::uint8_t* xb = x + r * ldx + (2 * b + si / 2) * kQ8BlockBytes;
                        const int8x16_t xv = vld1q_s8(reinterpret_cast<const std::int8_t*>(xb + 2 + (si % 2) * 16));
                        __fp16 dh;
                        std::memcpy(&dh, xb, sizeof(dh));
                        int32x4_t sum = vdotq_laneq_s32(zero, w0, xv, 0);
                        sum = vdotq_laneq_s32(sum, w1, xv, 1);
                        sum = vdotq_laneq_s32(sum, w2, xv, 2);
                        sum = vdotq_laneq_s32(sum, w3, xv, 3);
                        acc[r][si] = vfmaq_f32(acc[r][si], vcvtq_f32_s32(sum), vmulq_n_f32(s, static_cast<float>(dh)));
                    }
                }
            }
            for (int r = 0; r < R; r++)
                vst1q_f32(y + r * ldy, vaddq_f32(vaddq_f32(acc[r][0], acc[r][1]), vaddq_f32(acc[r][2], acc[r][3])));
        }

        // dst [M, N] = w (NVFP4 [K, M], a weight or a row block of one) · the Q8_0 activation rows src[1] [K, N]: this thread's
        // groups of four weight rows, each decoded once and run against every activation row.
        void nvfp4_gemm_op(ggml_tensor* dst, int ith, int nth, void*)
        {
            const ggml_tensor* w = dst->src[0];
            const ggml_tensor* xq = dst->src[1];
            const std::int64_t K = w->ne[0], M = w->ne[1], N = xq->ne[1], blocks = K / kNvfp4Block;
            const std::int64_t groups = M / 4, g0 = groups * ith / nth, g1 = groups * (ith + 1) / nth;
            if (g0 >= g1) return;
            thread_local std::vector<int8x16_t> q;
            thread_local std::vector<float32x4_t> sc;
            if (q.size() < static_cast<std::size_t>(blocks * 16)) q.resize(static_cast<std::size_t>(blocks * 16));
            if (sc.size() < static_cast<std::size_t>(blocks * 4)) sc.resize(static_cast<std::size_t>(blocks * 4));
            const auto* X = static_cast<const std::uint8_t*>(xq->data);
            const std::size_t ldx = xq->nb[1];
            const std::int64_t ldy = dst->nb[1] / static_cast<std::int64_t>(sizeof(float));
            auto* Y = static_cast<float*>(dst->data);
            for (std::int64_t g = g0; g < g1; g++)
            {
                nvfp4_decode4(static_cast<const std::uint8_t*>(w->data) + 4 * g * w->nb[1], w->nb[1], blocks, q.data(), sc.data());
                std::int64_t r = 0;
                for (; r + 4 <= N; r += 4) nvfp4_tile<4>(q.data(), sc.data(), blocks, X + r * ldx, ldx, Y + r * ldy + 4 * g, ldy);
                switch (N - r)
                {
                case 3: nvfp4_tile<3>(q.data(), sc.data(), blocks, X + r * ldx, ldx, Y + r * ldy + 4 * g, ldy); break;
                case 2: nvfp4_tile<2>(q.data(), sc.data(), blocks, X + r * ldx, ldx, Y + r * ldy + 4 * g, ldy); break;
                case 1: nvfp4_tile<1>(q.data(), sc.data(), blocks, X + r * ldx, ldx, Y + r * ldy + 4 * g, ldy); break;
                default: break;
                }
            }
        }
#endif
    }

    ggml_tensor* cpu_weight_product(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x)
    {
        if (w->ne[2] != 1 || w->ne[3] != 1 || w->nb[0] != ggml_type_size(w->type) || x->type != GGML_TYPE_F32
            || x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 || x->ne[3] != 1 || x->ne[0] != w->ne[0]
            || (x->ne[2] > 1 && x->nb[2] != x->nb[1] * x->ne[1]))
            return nullptr;
#if defined(TSG_CPU_BF16_NEON)
        [[maybe_unused]] const std::int64_t rows = x->ne[1] * x->ne[2];
#if defined(TSG_CPU_NVFP4_SDOT)
        if (w->type == GGML_TYPE_NVFP4 && rows >= kNvfp4GemmRows && w->ne[0] % kNvfp4Block == 0 && w->ne[1] % 4 == 0)
        {
            // ggml's Q8_0 activation rows (a tensor of the graph), then the GEMM over them.
            ggml_tensor* quant_args[1] = { x };
            ggml_tensor* xq = ggml_custom_4d(ctx, GGML_TYPE_Q8_0, w->ne[0], rows, 1, 1, quant_args, 1, q8_0_rows_op, GGML_N_TASKS_MAX, nullptr);
            ggml_tensor* gemm_args[2] = { w, xq };
            return ggml_custom_4d(ctx, GGML_TYPE_F32, w->ne[1], x->ne[1], x->ne[2], 1, gemm_args, 2, nvfp4_gemm_op, GGML_N_TASKS_MAX, nullptr);
        }
#endif
#if defined(TSG_GGML_USE_ACCELERATE)
        if (w->type == GGML_TYPE_BF16 && rows >= kSgemmRowsBf16)
        {
            // The weight as F32 (one graph tensor, at most kTileBytes: a wider weight - a vocabulary head read for every
            // row - goes in row tiles whose products are concatenated), then one sgemm per tile.
            const std::int64_t K = w->ne[0], M = w->ne[1];
            const std::int64_t tile = std::max<std::int64_t>(16, kTileBytes / (K * static_cast<std::int64_t>(sizeof(float))) / 16 * 16);
            ggml_tensor* result = nullptr;
            for (std::int64_t m0 = 0; m0 < M; m0 += tile)
            {
                const std::int64_t mt = std::min(tile, M - m0);
                ggml_tensor* wt = mt == M ? w : ggml_view_2d(ctx, w, K, mt, w->nb[1], static_cast<std::size_t>(m0) * w->nb[1]);
                ggml_tensor* conv_args[1] = { wt };
                ggml_tensor* wf = ggml_custom_4d(ctx, GGML_TYPE_F32, K, mt, 1, 1, conv_args, 1, rows_to_f32_op, GGML_N_TASKS_MAX, nullptr);
                ggml_tensor* gemm_args[2] = { wf, x };
                ggml_tensor* y = ggml_custom_4d(ctx, GGML_TYPE_F32, mt, x->ne[1], x->ne[2], 1, gemm_args, 2, sgemm_op, 1, nullptr);
                result = result == nullptr ? y : ggml_concat(ctx, result, y, 0);
            }
            return result;
        }
#endif
        if (w->type == GGML_TYPE_BF16)
        {
            ggml_tensor* args[2] = { w, x };
            return ggml_custom_4d(ctx, GGML_TYPE_F32, w->ne[1], x->ne[1], x->ne[2], 1, args, 2, bf16_rows_product, GGML_N_TASKS_MAX, nullptr);
        }
#endif
        return nullptr;
    }
}
