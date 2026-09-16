// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"
#include "ggml-backend.h"
#include <array>
#include <new>
#include <utility>
#include <vector>

// Allocate a newly built, unexecuted graph. The caller must discard its
// scheduler and graph on failure; neither may be reused after a failed reserve.
// Upstream's implicit allocation path can retry after a failed reserve without
// checking its result. An explicit checked reserve keeps that failure outside
// execution. Reserve resets scheduler placement, so restore every explicit
// assignment before allocating the graph against the successfully sized arena.
inline bool tsg_scheduler_alloc_graph(ggml_backend_sched_t scheduler, ggml_cgraph * graph) {
    struct original_sources {
        struct entry {
            ggml_tensor * node;
            std::array<ggml_tensor *, GGML_MAX_SRC> sources;
        };
        std::vector<entry> nodes;
        bool restored = false;
        void restore() {
            if (restored) return;
            for (const auto & item : nodes)
                for (size_t j = 0; j < item.sources.size(); ++j) item.node->src[j] = item.sources[j];
            restored = true;
        }
        ~original_sources() { restore(); }
    } originals;
    try {
        originals.nodes.reserve(size_t(ggml_graph_n_nodes(graph)));
        std::vector<std::pair<ggml_tensor *, ggml_backend_t>> placements;
        placements.reserve(size_t(ggml_graph_n_nodes(graph)) * 2);
        auto remember = [&](ggml_tensor * tensor) {
            if (auto backend = ggml_backend_sched_get_tensor_backend(scheduler, tensor))
                placements.emplace_back(tensor, backend);
        };
        for (int i = 0; i < ggml_graph_n_nodes(graph); ++i) {
            auto * node = ggml_graph_node(graph, i);
            original_sources::entry entry{node, {}};
            for (size_t j = 0; j < entry.sources.size(); ++j) entry.sources[j] = node->src[j];
            originals.nodes.push_back(entry);
            remember(node);
            // Every participating leaf is a source of a graph node. Repeated
            // assignments are harmless and avoid dependence on ggml's private
            // graph/allocator structures.
            for (auto * source : node->src) if (source) remember(source);
        }
        // A throwable C++ dispatch boundary keeps MSVC /EHsc from assuming these
        // C-linkage entry points cannot propagate a metadata allocation exception.
        // Volatile prevents devirtualizing this boundary back to a direct C call.
        using scheduler_call = bool (*)(ggml_backend_sched_t, ggml_cgraph *) noexcept(false);
        scheduler_call volatile reserve = &ggml_backend_sched_reserve;
        const bool reserved = reserve(scheduler, graph);
        // Splitting for reserve rewrites cross-device sources to copies owned by
        // the scheduler's temporary context. The next split replaces that context.
        // Restore original edges before it can invalidate those copies, including
        // on failure when the caller will discard this scheduler.
        originals.restore();
        if (!reserved) return false;
        for (const auto & [tensor, backend] : placements)
            ggml_backend_sched_set_tensor_backend(scheduler, tensor, backend);
        scheduler_call volatile allocate = &ggml_backend_sched_alloc_graph;
        return allocate(scheduler, graph);
    } catch (const std::bad_alloc &) {
        // Metadata allocation can also fail under host memory pressure. Return
        // through the caller's normal discard path so a partially built batched
        // or drafter graph cannot remain published in the graph cache.
        // Keep the backup alive outside this try. With MSVC /EHsc, an
        // extern-C call may not register ordinary unwind cleanup even when a
        // C++ allocation inside the callback throws. Restore explicitly here.
        originals.restore();
        return false;
    }
}
