// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstddef>
#include "ggml-backend.h"
#include "ggml-alloc.h"

namespace tsg {
    // Only these owned allocation/free pairs are tracked. Returned buffers retain
    // their ORIGINAL buft, context AND iface: CUDA/Metal use free_buffer identity
    // for device-copy dispatch. No backend interface is patched.
    ggml_backend_buffer_t graph_budget_alloc_buffer(ggml_backend_buffer_type_t type, std::size_t bytes, int rank);
    ggml_backend_buffer_t graph_budget_alloc_ctx_tensors(ggml_context* context, ggml_backend_t backend, int rank);
    void graph_budget_free_buffer(ggml_backend_buffer_t buffer);

    // Single-buffer-type gallocr only. Use the matching allocate/free entry points;
    // calling upstream reserve/reserve_size/free directly bypasses ownership.
    // The owner serializes allocator mutation, not concurrent graph execution.
    ggml_gallocr_t graph_budget_gallocr_new(ggml_backend_buffer_type_t type, int rank);
    bool graph_budget_gallocr_alloc_graph(ggml_gallocr_t allocator, ggml_cgraph* graph);
    void graph_budget_gallocr_free(ggml_gallocr_t allocator);
}
