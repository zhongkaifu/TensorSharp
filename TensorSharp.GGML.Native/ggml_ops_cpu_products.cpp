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
//   - BF16 and NVFP4 products of prompt width (from kSgemmRowsBf16 / kSgemmRowsNvfp4 rows, Apple): the weight is widened / dequantized to F32
//     by every thread (one graph tensor, row tiles past kTileBytes), then multiplied by one Accelerate sgemm call (Apple's
//     matrix units), as llama.cpp's BLAS backend does for the products it takes.
// Both read F32 activations unrounded (ggml rounds them to the weight's dot type: BF16, or Q8_0 for NVFP4).

#include "ggml_ops_cpu_graph.h"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <vector>

#if defined(__aarch64__) && defined(__ARM_NEON)
#include <arm_neon.h>
#define TSG_CPU_BF16_NEON 1
#endif

#if defined(TSG_GGML_USE_ACCELERATE)
#include <Accelerate/Accelerate.h>
#endif

namespace tsg
{
    namespace
    {
        // Activation rows from which a product's weight goes through sgemm, per weight type. The conversion to F32 is
        // paid per product whatever its width, so sgemm pays off from the width where its speed covers it (M5 Pro,
        // LFM3-1.7B at 18 threads, interleaved): BF16 (widened) loses to the NEON rows at 32 (121 vs 185 tok/s) and
        // leads from 64 (210 vs 199; 241 vs 203 at 96, 315 vs 283 at 128); NVFP4 (dequantized) loses to ggml-cpu's own
        // NVFP4 product at 128 (261 vs 374) and leads from 256 (468 vs 353; 638 vs 348 at 512).
        constexpr std::int64_t kSgemmRowsBf16 = 64;
        constexpr std::int64_t kSgemmRowsNvfp4 = 256;
        // The F32 copy of a prompt product's weight is at most this big (larger weights go in row tiles).
        constexpr std::int64_t kTileBytes = std::int64_t(128) << 20;

        inline float bf16_to_f32(std::uint16_t h)
        {
            const std::uint32_t u = static_cast<std::uint32_t>(h) << 16;
            float f;
            std::memcpy(&f, &u, sizeof(f));
            return f;
        }

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
                        vst1q_f32(d + k, vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v), 16)));
                        vst1q_f32(d + k + 4, vreinterpretq_f32_u32(vshll_high_n_u16(v, 16)));
                    }
#endif
                    for (; k < K; k++) d[k] = bf16_to_f32(s[k]);
                }
                return;
            }
#if defined(TSG_CPU_BF16_NEON)
            if (w->type == GGML_TYPE_NVFP4 && K % 64 == 0)
            {
                // ggml's dequantize_row_nvfp4 (value = kvalues_fp4[code] * ue4m3(scale)) with a table lookup per 16 codes:
                // the same float products, bit for bit.
                static const std::int8_t kvalues[16] = { 0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12 };
                const int8x16_t lut = vld1q_s8(kvalues);
                const float* scales = nvfp4_scales();
                const uint8x8_t low = vdup_n_u8(0x0F);
                for (std::int64_t r = 0; r < rows; r++)
                {
                    const auto* blk = reinterpret_cast<const std::uint8_t*>(base + r * w->nb[1]);
                    float* d = dst + r * K;
                    for (std::int64_t b = 0; b < K / 64; b++, blk += 36)
                        for (int sb = 0; sb < 4; sb++, d += 16)
                        {
                            const float32x4_t scale = vdupq_n_f32(scales[blk[sb]]);
                            const uint8x8_t q = vld1_u8(blk + 4 + sb * 8);
                            const int8x16_t v = vqtbl1q_s8(lut, vcombine_u8(vand_u8(q, low), vshr_n_u8(q, 4)));
                            const int16x8_t v0 = vmovl_s8(vget_low_s8(v)), v1 = vmovl_high_s8(v);
                            vst1q_f32(d + 0, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v0))), scale));
                            vst1q_f32(d + 4, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(v0)), scale));
                            vst1q_f32(d + 8, vmulq_f32(vcvtq_f32_s32(vmovl_s16(vget_low_s16(v1))), scale));
                            vst1q_f32(d + 12, vmulq_f32(vcvtq_f32_s32(vmovl_high_s16(v1)), scale));
                        }
                }
                return;
            }
#endif
            const ggml_to_float_t to_float = ggml_get_type_traits(w->type)->to_float;
            for (std::int64_t r = 0; r < rows; r++)
                to_float(base + r * w->nb[1], dst + r * K, K);
        }

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
                const float32x4_t w0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v0), 16));
                const float32x4_t w1 = vreinterpretq_f32_u32(vshll_high_n_u16(v0, 16));
                const float32x4_t w2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(v1), 16));
                const float32x4_t w3 = vreinterpretq_f32_u32(vshll_high_n_u16(v1, 16));
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
    }

    ggml_tensor* cpu_weight_product(ggml_context* ctx, ggml_tensor* w, ggml_tensor* x)
    {
        if (w->ne[2] != 1 || w->ne[3] != 1 || w->nb[0] != ggml_type_size(w->type) || x->type != GGML_TYPE_F32
            || x->nb[0] != sizeof(float) || x->nb[1] % sizeof(float) != 0 || x->ne[3] != 1 || x->ne[0] != w->ne[0]
            || (x->ne[2] > 1 && x->nb[2] != x->nb[1] * x->ne[1]))
            return nullptr;
#if defined(TSG_CPU_BF16_NEON)
        [[maybe_unused]] const std::int64_t rows = x->ne[1] * x->ne[2];
#if defined(TSG_GGML_USE_ACCELERATE)
        if ((w->type == GGML_TYPE_BF16 && rows >= kSgemmRowsBf16) || (w->type == GGML_TYPE_NVFP4 && rows >= kSgemmRowsNvfp4))
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
