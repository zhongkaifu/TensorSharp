// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"

#include <cmath>
#include <cstdint>

// Per-head norms for graphs that carry many tokens (prefill, verify).
//
// The *_heads helpers take a contiguous `x` that holds n_cols columns of n_heads
// rows of head_dim values (any 2-D or 3-D shape with that memory layout, e.g.
// [head_dim * n_heads, n_cols]) and normalise every head_dim row. The result is
// [head_dim, n_heads, n_cols].
//
// They norm the 3-D view, not a flattened [head_dim, n_heads * n_cols] one. Both
// are the same memory and normalise the same rows, but ggml-vulkan dispatches
// RMS_NORM as one workgroup per ne01 row on the X axis and clamps (and loops over)
// only Y and Z. Flattened, the X axis is n_heads * n_cols, past the 65536
// maxComputeWorkGroupCount[0] Intel GPUs report as soon as a prefill step is long
// enough: on an Intel Arc 140V, Qwen 3.5's gated-delta-net output norm (32 value
// heads) aborted ggml_vk_dispatch_pipeline at a 2049-token step, and its 16-head
// attention q norm and gated-delta-net q/k L2 norms at 4097. In 3-D, X is n_heads
// and the column count rides on Y. (ggml's own L2_NORM and NORM need no such care:
// ggml-vulkan already spreads their rows over 512 x 512 x Z workgroups. The
// gated-delta-net L2 norm below is an RMS_NORM, so it does.)
namespace tsg
{
    // RMSNorm over head_dim, then scaled by w [head_dim] unless `w` is null.
    inline ggml_tensor* rms_norm_heads(ggml_context* ctx, ggml_tensor* x,
                                       std::int64_t head_dim, std::int64_t n_heads, std::int64_t n_cols,
                                       ggml_tensor* w, float eps)
    {
        ggml_tensor* y = ggml_rms_norm(ctx, ggml_reshape_3d(ctx, x, head_dim, n_heads, n_cols), eps);
        return w != nullptr ? ggml_mul(ctx, y, w) : y;
    }

    // Qwen3.5 GatedDeltaNet normalizes each Q/K head with
    //
    //     x * rsqrt(sum(x^2) + eps)
    //
    // ggml_l2_norm uses a different epsilon convention. Express the exact
    // operation through RMSNorm instead: RMSNorm(x, eps / n) / sqrt(n), where
    // n is the head dimension (ne[0]). This mirrors llama.cpp's GDN helper.
    // It norms `x` in the shape given; multi-token callers use gdn_l2_norm_heads.
    inline ggml_tensor* build_gdn_l2_norm(ggml_context* ctx, ggml_tensor* x, float eps)
    {
        const float n = static_cast<float>(x->ne[0]);
        return ggml_scale(ctx, ggml_rms_norm(ctx, x, eps / n), 1.0f / std::sqrt(n));
    }

    // build_gdn_l2_norm on the 3-D [head_dim, n_heads, n_cols] view.
    inline ggml_tensor* gdn_l2_norm_heads(ggml_context* ctx, ggml_tensor* x,
                                          std::int64_t head_dim, std::int64_t n_heads, std::int64_t n_cols,
                                          float eps)
    {
        return build_gdn_l2_norm(ctx, ggml_reshape_3d(ctx, x, head_dim, n_heads, n_cols), eps);
    }
}
