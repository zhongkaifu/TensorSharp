// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_graph_optimize.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#if defined(TSG_GGML_USE_METAL)
#include "ggml-metal/ggml-metal-fusion.h"
#endif

namespace {
    using Dependencies = std::unordered_map<ggml_tensor *, std::vector<ggml_tensor *>>;

    void add_dependency(void * data, ggml_tensor * tensor, ggml_tensor * until) {
        if (!tensor || !until) throw std::runtime_error("Backend optimizer requested a null allocation dependency");
        auto & keep = (*static_cast<Dependencies *>(data))[until];
        if (std::find(keep.begin(), keep.end(), tensor) == keep.end()) keep.push_back(tensor);
    }

    void insert_dependencies(ggml_context * ctx, ggml_cgraph * graph, const Dependencies & dependencies);
}

void tsg_optimize_graph_with_alloc_dependencies(ggml_backend_t backend, ggml_context * ctx, ggml_cgraph * graph) {
    if (!backend || !graph || !backend->iface.graph_optimize) return;
    Dependencies dependencies;
    ggml_backend_graph_optimize_params params = { add_dependency, &dependencies };
    backend->iface.graph_optimize(backend, graph, &params);
    insert_dependencies(ctx, graph, dependencies);
}

void tsg_add_metal_fusion_alloc_dependencies(ggml_context * ctx, ggml_cgraph * graph) {
#if defined(TSG_GGML_USE_METAL)
    if (!graph) return;
    Dependencies dependencies;
    ggml_metal_fusion_add_alloc_deps(&dependencies, add_dependency, graph);
    insert_dependencies(ctx, graph, dependencies);
#else
    (void)ctx; (void)graph;
#endif
}

namespace {
void insert_dependencies(ggml_context * ctx, ggml_cgraph * graph, const Dependencies & dependencies) {
    if (dependencies.empty()) return;

    std::unordered_set<ggml_tensor *> nodes(graph->nodes, graph->nodes + graph->n_nodes);
    size_t count = 0;
    for (const auto & entry : dependencies) {
        if (!nodes.count(entry.first))
            throw std::runtime_error("Backend optimizer requested a dependency on a node outside the graph");
        count += (entry.second.size() + GGML_MAX_SRC - 1) / GGML_MAX_SRC;
    }
    const size_t tensor_overhead = ggml_tensor_overhead();
    const bool graph_room = count <= size_t(graph->size - graph->n_nodes);
    const bool context_room = ctx && ggml_used_mem(ctx) <= ggml_get_mem_size(ctx) &&
        count <= (ggml_get_mem_size(ctx) - ggml_used_mem(ctx)) / tensor_overhead;
    if (!graph_room || !context_room) {
        // Gallocr never reuses an OUTPUT tensor's storage. Retain the base as
        // well as its view: pinning just a view does not necessarily protect
        // the underlying allocation from reuse by another alias.
        for (const auto & entry : dependencies)
            for (auto * tensor : entry.second)
                for (auto * storage = tensor; storage; storage = storage->view_src)
                    ggml_set_output(storage);
        return;
    }

    std::vector<ggml_tensor *> original(graph->nodes, graph->nodes + graph->n_nodes);
    graph->n_nodes = 0;
    for (auto * node : original) {
        graph->nodes[graph->n_nodes++] = node;
        const auto found = dependencies.find(node);
        if (found == dependencies.end()) continue;
        const auto & keep = found->second;
        for (size_t k = 0; k < keep.size(); k += GGML_MAX_SRC) {
            // ggml_view_tensor is an OP_NONE metadata view. Its source edges
            // extend allocator lifetimes without adding computation, just as
            // in upstream ggml_backend_sched's allocation graph.
            auto * dep = ggml_view_tensor(ctx, keep[k]);
            for (size_t s = 0; s < GGML_MAX_SRC && k + s < keep.size(); ++s)
                dep->src[s] = keep[k + s];
            ggml_set_name(dep, "tsg.optimizer_allocation_dependency");
            graph->nodes[graph->n_nodes++] = dep;
        }
    }
}
}
