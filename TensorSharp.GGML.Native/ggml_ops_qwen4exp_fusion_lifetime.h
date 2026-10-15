// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml.h"

namespace tsg {
// A per-slot position VIEW is a graph node, unlike solo's leaf position input.
// Visit it before the K branch so DFS cannot insert it between RMS/MUL and ROPE,
// changing the backend's adjacent-node fusion (and ultimately F16 KV rounding).
// This adds no tensor, allocation, output flag or arithmetic operation.
inline void qwen4exp_expand_rotary_position_input(ggml_cgraph* graph, ggml_tensor* positions) {
    if (positions != nullptr && positions->op != GGML_OP_NONE)
        ggml_build_forward_expand(graph, positions);
}

// Arena and solo decode must expose the same fusion inputs to the allocator.
// Views do not own allocations: protect the ultimate producer instead.
inline void qwen4exp_keep_fusion_input(ggml_tensor* tensor, bool enabled) noexcept {
    if (!enabled || tensor == nullptr) return;
    while (tensor->view_src != nullptr) tensor = tensor->view_src;
    if (tensor->op != GGML_OP_NONE) ggml_set_output(tensor);
}
}
