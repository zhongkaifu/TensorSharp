// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <unordered_map>
#include <vector>

// The allocation dependencies a backend's graph optimizer asks for: each node mapped to the tensors that must stay
// allocated until it has been computed.
using tsg_alloc_dependencies = std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>>;

// Runs the backend's graph optimizer (reorders and fusions) on graph and collects the allocation dependencies it asks
// for into dependencies, filling ggml's optimize parameters by field (their layout differs between ggml revisions:
// upstream added add_scratch after add_alloc_dep). A backend asking for scheduler scratch is refused. No-op without an
// optimizer.
void tsg_collect_graph_optimize_dependencies(ggml_backend_t backend, ggml_cgraph * graph, tsg_alloc_dependencies & dependencies);

// Direct-compute equivalent of the scheduler's allocation-dependency sink.
// Call before allocation. Dependency views belong to ctx and remain valid for
// exactly the same lifetime as the graph. A tight metadata budget falls back
// to retaining the affected storage through the end of the graph.
void tsg_optimize_graph_with_alloc_dependencies(ggml_backend_t backend, ggml_context * ctx, ggml_cgraph * graph);

// The allocation dependencies of ggml-metal's fusions alone, without the backend's reorder: ggml-metal fuses a
// graph's chains when it encodes them whatever order they are in, and a fused kernel reads inputs the allocator
// would otherwise hand to a node that runs concurrently with it. No-op off Metal builds.
void tsg_add_metal_fusion_alloc_dependencies(ggml_context * ctx, ggml_cgraph * graph);
