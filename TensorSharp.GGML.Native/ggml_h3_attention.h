// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// ============================================================================
// MiniMax-H3's graph-building pieces of attention: the signed sine table and the
// rotate-half RoPE over a table (h3_vit_rope), and the bidirectional attention
// over a packed sequence (h3_attend) with its FP16 V pre-scale and, on ggml-metal
// with the tensor API, the keys padded to 64 and the queries chunked
// (ggml_ops_metal_flash_pad.h).
//
// MiniMax-H3's native graphs (ggml_ops_minimax_h3.cpp) build them, and the generic
// graph builder (ggml_ops_graph_builder.cpp) lowers GraphFusion.DitAttention with
// h3_attend, so both build the same nodes. Moved verbatim from
// ggml_ops_minimax_h3.cpp. TensorSharp-owned; ggml is not changed.
// ============================================================================
#pragma once

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml_ops_metal_flash_pad.h"

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <vector>

namespace tsg {

// The sine table rotate-half RoPE multiplies the half-swapped head by:
// [-sin(first half), sin(second half)]. sin: [rot, seq]. Built once per graph.
inline ggml_tensor* h3_rope_signed_sin(ggml_context* ctx, ggml_tensor* sinf)
{
    const std::int64_t rot = sinf->ne[0], half = rot / 2;
    ggml_tensor* first = ggml_view_2d(ctx, sinf, half, sinf->ne[1], sinf->nb[1], 0);
    ggml_tensor* second = ggml_view_2d(ctx, sinf, half, sinf->ne[1], sinf->nb[1],
                                       static_cast<std::size_t>(half) * sizeof(float));
    return ggml_concat(ctx, ggml_neg(ctx, ggml_cont(ctx, first)), second, 0);
}

// Rotate-half RoPE over the leading `rot` dims of each head.
// x: [head_dim, heads, seq], contiguous or a row-contiguous view;
// cos and signedSin (h3_rope_signed_sin): [rot, seq].
//
// x*cos + rotate_half(x)*sin with rotate_half(x) = [-x2, x1]. The half-swapped
// head [x2, x1] is read straight out of x, and the sign moves onto the sine:
// (-a)*b and a*(-b) are both exactly -(a*b), so every element is the same
// product, sum and rounding as the explicit rotate_half -- five passes where
// slicing, negating and re-concatenating the halves took ten.
inline ggml_tensor* h3_vit_rope(ggml_context* ctx, ggml_tensor* x,
                         ggml_tensor* cosf, ggml_tensor* signedSin,
                         int hd, int heads, int seq, int rot)
{
    if (rot <= 0) return x;
    const std::size_t f = sizeof(float);
    const int half = rot / 2;
    ggml_tensor* xr = ggml_view_3d(ctx, x, rot, heads, seq, x->nb[1], x->nb[2], 0);
    ggml_tensor* swapped = ggml_concat(ctx,
        ggml_view_3d(ctx, x, half, heads, seq, x->nb[1], x->nb[2], static_cast<std::size_t>(half) * f),
        ggml_view_3d(ctx, x, half, heads, seq, x->nb[1], x->nb[2], 0), 0);
    ggml_tensor* c3 = ggml_reshape_3d(ctx, cosf, rot, 1, seq);
    ggml_tensor* s3 = ggml_reshape_3d(ctx, signedSin, rot, 1, seq);
    ggml_tensor* rotated = ggml_add(ctx, ggml_mul(ctx, xr, c3), ggml_mul(ctx, swapped, s3));
    if (rot == hd) return rotated;
    return ggml_concat(ctx, rotated, ggml_view_3d(ctx, x, hd - rot, heads, seq,
                                                  x->nb[1], x->nb[2],
                                                  static_cast<std::size_t>(rot) * f), 0);
}

// Effective keys one FP16 softmax numerator is allowed to sum over before V is
// pre-scaled. See h3_flash_v_scale for the arithmetic; 256 leaves the
// accumulator 8 * 65504 / 256 = 2047 of room on |V| itself, an order of
// magnitude above the activations h3_mm below documents as "reaching the
// thousands".
constexpr int kH3FlashKeyBudget = 256;

// Smallest power of two that brings `keys` down to the budget, and exactly 1.0
// for any sequence already short enough -- so short sequences (every oracle
// fixture in the test suite among them) stay bit-identical.
inline float h3_flash_v_scale(int keys)
{
    float s = 1.0f;
    while (keys > kH3FlashKeyBudget * s) s *= 2.0f;
    return s;
}

// The key-padding masks of one graph (ggml_ops_metal_flash_pad.h), one per
// sequence length, so the 50 DiT blocks share one rather than building 50.
struct H3FlashMasks
{
    struct Entry { int seq, keys, rows; ggml_tensor* mask; };
    std::vector<Entry> entries;

    ggml_tensor* get(ggml_context* ctx, int seq, int keys, int rows)
    {
        for (const Entry& e : entries)
            if (e.seq == seq && e.keys == keys && e.rows == rows) return e.mask;
        ggml_tensor* mask = tsg_metal_flash_kv_pad_mask(ctx, seq, keys, rows);
        entries.push_back({ seq, keys, rows, mask });
        return mask;
    }
};

// h3_attend's flash call with the keys padded to `keys` (a multiple of 64) and
// the queries in chunks sharing one small mask (ggml_ops_metal_flash_pad.h).
// q/k/v as h3_attend takes them; returns [inner, seq] still divided by vScale,
// or null when the backend cannot run the padded call.
inline ggml_tensor* h3_attend_padded(ggml_context* ctx, ggml_backend_t backend,
                              ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                              int heads, int seq, int keys, float scale, float vScale,
                              H3FlashMasks& masks)
{
    const int chunk = tsg_metal_flash_query_chunk(seq, heads);
    // ggml_pad zero-fills the K/V rows past seq, and its output stands in for the
    // ggml_cont the permute needs. V may be a strided view, so it is pre-scaled
    // after the pad: the pad copies values and its zeros stay zero.
    ggml_tensor* vp = ggml_pad(ctx, v, 0, 0, keys - seq, 0);
    ggml_tensor* vs = vScale == 1.0f ? vp : ggml_scale(ctx, vp, 1.0f / vScale);
    ggml_tensor* kf = ggml_cast(ctx,
        ggml_permute(ctx, ggml_pad(ctx, k, 0, 0, keys - seq, 0), 0, 2, 1, 3), GGML_TYPE_F16);
    ggml_tensor* vf = ggml_cast(ctx, ggml_permute(ctx, vs, 0, 2, 1, 3), GGML_TYPE_F16);
    ggml_tensor* merged = tsg_metal_flash_attn_chunked(ctx, backend, ggml_permute(ctx, q, 0, 2, 1, 3),
                                                       kf, vf, masks.get(ctx, seq, keys, chunk),
                                                       chunk, scale);
    static bool reported = false;
    if (merged != nullptr && !reported)
    {
        reported = true;
        std::fprintf(stderr, "  [h3] flash attention: %d keys padded to %d for ggml-metal's "
                     "tensor-API kernel (TS_METAL_FLASH_KV_PAD=0 disables)\n", seq, keys);
    }
    return merged;
}

// Full bidirectional attention over a packed sequence.
//
// q/k/v arrive as [head_dim, heads, seq]. The explicit path materializes a
// [seq, seq, heads] score matrix, which at 640x384 would be ~14 GB for the DiT and
// ~6 GB per VAE layer -- the reference sidesteps that by tiling the VAE. Flash
// attention never materializes it, so one call covers the whole frame with no tile
// seams. K/V are cast to F16 because that is what the kernel takes.
//
// THE ACCUMULATE IS NOT RELIABLY F32, whatever ggml_prec_set_acc below looks
// like it asks for. ggml-cuda's tensor-core kernel keeps the softmax NUMERATOR
// -- sum_j exp(s_j - max) * V_j -- in FP16 registers (T_C_VKQ = tile<16, 8,
// half2>, ggml-cuda/fattn-mma-f16.cuh), and neither ggml-cuda nor ggml-metal
// reads the precision slot of GGML_OP_FLASH_ATTN_EXT (op_params[3]; they read
// only scale, max_bias, logit_softcap and n_kv_max), so the request cannot move
// them. ggml-metal's F16 kernels accumulate in float anyway (FA_TYPES,
// ggml-metal/kernels/fa.metal); ggml-vulkan is the one backend that honours the
// request and switches to an F32 accumulator. (Checked against the vendored ggml
// at 353b63b.) What the CUDA kernel does give its accumulator is three bits of
// headroom: FATTN_KQ_MAX_OFFSET (ggml-cuda/fattn-common.cuh) inflates the
// running maximum by log(8), capping every softmax weight at 1/8. A row of N keys
// therefore reaches N/8 * |V| and overflows to Inf once N * |V| > 8 * 65504.
//
// H3 walks straight into that ceiling and no other model here does. Its attention
// is bidirectional over ONE packed sequence with no mask, so N is the whole clip
// rather than a sliding window (2364 tokens for 22 frames, 8646 for 107), and V is
// the one stream carrying no norm: q_norm and k_norm exist in the checkpoint,
// v_norm does not, which is the same unbounded magnitude that makes h3_mm's guard
// necessary one op later. Measured at 640x384: 73 frames (N=6134, tolerates
// |V| <= 85) renders correctly and 107 frames (N=8646, tolerates only 61) came
// back pure black, because one Inf here poisons the shared trunk and takes the
// video AND the audio stream down with it.
//
// Attention is LINEAR in V, so scaling V down before the kernel and scaling the
// result back up afterwards is exact -- and by a power of two it is exact through
// the FP16 cast as well, being nothing but an exponent shift.
//
// On ggml-metal with the Metal 4 tensor API the keys are padded to a multiple of
// 64 and the pads masked, which is what puts a long sequence on the tensor-API
// kernel: 2.49x per call at 1280x768x107 (ggml_ops_metal_flash_pad.h). `masks`
// shares that mask between the graph's attentions; null leaves the keys unpadded.
inline ggml_tensor* h3_attend(ggml_context* ctx, ggml_backend_t backend,
                       ggml_tensor* q, ggml_tensor* k, ggml_tensor* v,
                       int hd, int heads, int seq, float scale, bool allowFlash,
                       H3FlashMasks* masks = nullptr)
{
    ggml_tensor* qp = ggml_cont(ctx, ggml_permute(ctx, q, 0, 2, 1, 3));   // [hd, seq, heads]
    ggml_tensor* kp = ggml_cont(ctx, ggml_permute(ctx, k, 0, 2, 1, 3));

    // TS_H3_NO_FLASH=1 forces the explicit softmax path, for isolating a
    // flash-attention kernel problem from a modelling one.
    if (const char* off = std::getenv("TS_H3_NO_FLASH"))
        if (off[0] == '1') allowFlash = false;

    if (allowFlash)
    {
        // Keep the kernel's FP16 numerator in range no matter how long the clip
        // is: the accumulator now sums over at most kH3FlashKeyBudget effective
        // keys' worth of V, and the output is scaled back on the way out.
        const float vScale = h3_flash_v_scale(seq);
        const int keys = masks != nullptr ? tsg_metal_flash_kv_length(backend, seq, seq) : seq;
        if (keys != seq)
        {
            ggml_tensor* merged = h3_attend_padded(ctx, backend, q, k, v, heads, seq, keys,
                                                   scale, vScale, *masks);
            if (merged != nullptr)
                return vScale == 1.0f ? merged : ggml_scale(ctx, merged, vScale);
        }
        ggml_tensor* vp = ggml_cont(ctx, ggml_permute(ctx, v, 0, 2, 1, 3));
        ggml_tensor* kf = ggml_cast(ctx, kp, GGML_TYPE_F16);
        ggml_tensor* vf = ggml_cast(ctx,
            vScale == 1.0f ? vp : ggml_scale(ctx, vp, 1.0f / vScale), GGML_TYPE_F16);
        ggml_tensor* out = ggml_flash_attn_ext(ctx, qp, kf, vf, nullptr, scale, 0.0f, 0.0f);
        if (out != nullptr && ggml_backend_supports_op(backend, out))
        {
            // Inert on ggml-cuda and ggml-metal, honoured only by ggml-vulkan (see
            // the note above the function); kept because the request is the right
            // one and costs nothing, but the V pre-scale is what actually keeps the
            // FP16 accumulator finite.
            ggml_prec_set_acc(out, GGML_PREC_F32);
            // [hd, heads, seq] contiguous -> [inner, seq].
            ggml_tensor* merged = ggml_reshape_2d(ctx, ggml_cont(ctx, out), hd * heads, seq);
            return vScale == 1.0f ? merged : ggml_scale(ctx, merged, vScale);
        }
    }

    ggml_tensor* kq = ggml_mul_mat(ctx, kp, qp);
    ggml_tensor* probs = ggml_soft_max_ext(ctx, kq, nullptr, scale, 0.0f);
    ggml_tensor* vt = ggml_cont(ctx, ggml_permute(ctx, v, 1, 2, 0, 3));
    ggml_tensor* kqv = ggml_mul_mat(ctx, vt, probs);
    return ggml_reshape_2d(ctx,
        ggml_cont(ctx, ggml_permute(ctx, kqv, 0, 2, 1, 3)), hd * heads, seq);
}

} // namespace tsg
