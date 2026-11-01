// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include "ggml-alloc.h"

namespace tsg
{
    // Rebuild the lifetime/alias plan for this graph while retaining backing
    // buffers that already fit. A prior larger graph's plan can fit every
    // tensor yet change aliasing and backend fusion for the current graph.
    inline bool alloc_graph_with_current_plan(ggml_gallocr_t allocator, ggml_cgraph* graph)
    {
        return allocator != nullptr && graph != nullptr &&
            ggml_gallocr_reserve(allocator, graph) &&
            ggml_gallocr_alloc_graph(allocator, graph);
    }
}
