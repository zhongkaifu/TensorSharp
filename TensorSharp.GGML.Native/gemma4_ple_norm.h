// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml.h"

#include <cstdint>

// Gemma 4 per-layer-embedding (PLE) hidden-projection norm, shared by the
// multi-column graphs (TSGgml_Gemma4ModelVerify, TSGgml_Gemma4ModelDecodeBatched).
//
// `proj` is per_layer_model_proj's output, [num_layers * ple_dim, n_cols]. Every
// (layer, column) slice of ple_dim values is RMS-normalised and scaled by
// `norm_w` [ple_dim]; the result has proj's shape again.
//
// The norm runs on a 3-D view [ple_dim, num_layers, n_cols], as llama.cpp's
// gemma3n graph does (reshape_3d(n_embd_altup, n_layer, n_tokens) before
// build_norm). It is the same memory as the flattened [ple_dim, num_layers *
// n_cols] view this replaces, so every row normalises exactly as before, but the
// dispatch is different: ggml-vulkan runs RMS_NORM as one workgroup per ne01 row
// on the X axis and clamps (and loops over) only Y and Z. Flattened, a 42-layer
// model asked for 42 * n_cols X workgroups, past the 65536
// maxComputeWorkGroupCount[0] Intel GPUs report from n_cols = 1561 on - which the
// Vulkan verify graph, padding N to a multiple of 64, reached at a 1537-token
// prefill step (1600 columns; GGML_ASSERT in ggml_vk_dispatch_pipeline). In 3-D,
// X is num_layers and the column count rides on Y. (The per-op TSGgml_NormF32 that
// the managed ComputePLE calls already splits such norms into row chunks, see
// ggml_ops_norm_attn.cpp; inside a graph the 3-D view needs no extra nodes.)
namespace tsg_gemma4_ple
{
    inline ggml_tensor* proj_norm(ggml_context* ctx, ggml_tensor* proj,
                                  int ple_dim, int num_layers,
                                  ggml_tensor* norm_w, float eps)
    {
        const std::int64_t n_cols = proj->ne[1];
        ggml_tensor* rows = ggml_reshape_3d(ctx, ggml_cont(ctx, proj), ple_dim, num_layers, n_cols);
        rows = ggml_mul(ctx, ggml_rms_norm(ctx, rows, eps), norm_w);
        return ggml_reshape_2d(ctx, rows, static_cast<std::int64_t>(ple_dim) * num_layers, n_cols);
    }
}
