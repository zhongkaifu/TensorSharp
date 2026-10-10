// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "dsv4_host_expert_read_graph.h"
#include "ggml-backend-impl.h"
#include "ggml-impl.h"
#include "ggml-cpu.h"
#include <unordered_map>

#if defined(__linux__)
#include <sys/resource.h>
namespace tsg_dsv4 {
struct host_expert_backend_state {
    ggml_backend_t cpu;
    std::unordered_map<const void *, host_expert_read_context *> weights;
    std::vector<std::pair<host_expert_read_context *, const ggml_tensor *>> seen;
    expert_read_feedback feedback;
    host_expert_reader * reader = nullptr;
    bool adaptive = true;
    uint64_t bypassed_graphs = 0;
    long observed_faults = -1;
    int observe_faults() {
        struct rusage usage{};
        if (getrusage(RUSAGE_SELF, &usage) != 0) {
            feedback.reset(); observed_faults = -1; return -1;
        }
        const bool changed = observed_faults != usage.ru_majflt;
        if (changed) {
            feedback.reset();
            // An OS eviction also invalidates the cheaper per-expert hints.
            // Repeating this for the three projection entries is harmless.
            for (const auto & entry : weights) entry.second->hints.invalidate();
        }
        observed_faults = usage.ru_majflt;
        return changed ? 1 : 0;
    }
};
// Borrow the model's CPU backend and its device/buffer types. Prepare registered
// expert weights on the submitting thread before their ordinary matmuls.
// Every node (including custom callbacks) remains an upstream CPU operation.
// This avoids holding all compute workers at a barrier during blocking I/O.
inline ggml_backend_t host_expert_backend(ggml_backend_t cpu, bool adaptive = true) {
    if (!ggml_backend_is_cpu(cpu)) throw std::runtime_error("Demand reader requires a CPU backend");
    static ggml_guid guid = {0xad,0x3e,0x67,0x21,0x9c,0x42,0x4a,0x7f,0x81,0x65,0x13,0x8b,0xc2,0x70,0x6d,0x14};
    ggml_backend_i iface{};
    iface.get_name = [](ggml_backend_t) { return "TensorSharp_CPU_IO"; };
    iface.free = [](ggml_backend_t backend) {
        delete static_cast<host_expert_backend_state *>(backend->context); delete backend;
    };
    iface.graph_compute = [](ggml_backend_t backend, ggml_cgraph * graph) -> ggml_status {
        auto & state = *static_cast<host_expert_backend_state *>(backend->context);
        if (state.reader && state.reader->error()[0]) return GGML_STATUS_FAILED;
        if (state.adaptive) state.observe_faults();
        state.seen.clear();
        // Inspect shape only, never route values before their producer runs.
        bool decode = true;
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * node = graph->nodes[i];
            if (node->op == GGML_OP_MUL_MAT_ID && node->src[2]->ne[1] != 1) decode = false;
        }
        if (!decode) state.feedback.reset();
        if (state.adaptive && decode && state.feedback.hot()) {
            ++state.bypassed_graphs;
            const auto result = ggml_backend_graph_compute(state.cpu, graph);
            state.observe_faults();
            return result;
        }
        const auto read_before = state.reader ? state.reader->read : 0;
        int start = 0;
        const auto flush = [&](int end) {
            if (start == end) return GGML_STATUS_SUCCESS;
            bool work = false;
            for (int i = start; i < end; ++i)
                work |= !ggml_op_is_empty(graph->nodes[i]->op) && (graph->nodes[i]->flags & GGML_TENSOR_FLAG_COMPUTE);
            if (!work) return GGML_STATUS_SUCCESS;
            auto view = ggml_graph_view(graph, start, end);
            return ggml_backend_graph_compute(state.cpu, &view);
        };
        for (int i = 0; i < graph->n_nodes; ++i) {
            auto * node = graph->nodes[i];
            if (node->op == GGML_OP_MUL_MAT_ID && (node->flags & GGML_TENSOR_FLAG_COMPUTE)) {
                const auto found = state.weights.find(node->src[0]->data);
                const auto key = std::make_pair(found != state.weights.end() ? found->second : nullptr,
                    static_cast<const ggml_tensor *>(node->src[2]));
                if (found != state.weights.end() && std::find(state.seen.begin(), state.seen.end(), key) == state.seen.end()) {
                    const auto status = flush(i);
                    if (status != GGML_STATUS_SUCCESS) return status;
                    auto & context = *found->second;
                    host_expert_read_selected(context, node->src[2]);
                    if (context.reader->error()[0]) {
                        std::fprintf(stderr, "[dsv4] %s\n", context.reader->error());
                        return GGML_STATUS_FAILED;
                    }
                    state.seen.push_back(key);
                    start = i; // keep this and later matmuls in the original graph
                }
            }

        }
        const auto result = flush(graph->n_nodes);
        if (state.adaptive && decode && state.reader && !state.seen.empty() && result == GGML_STATUS_SUCCESS) {
            if (state.observe_faults() == 0) state.feedback.prepared(state.reader->read - read_before);
            if (state.feedback.ready()) state.feedback.arm(state.observed_faults);
        }
        return result;
    };
    auto state = std::make_unique<host_expert_backend_state>();
    state->cpu = cpu;
    state->adaptive = adaptive;
    auto * result = new ggml_backend{&guid, iface, ggml_backend_get_device(cpu), state.get()};
    state.release();
    return result;
}

inline void host_expert_backend_register(ggml_backend_t backend, host_expert_read_context * context) {
    auto & state = *static_cast<host_expert_backend_state *>(backend->context);
    if (!context || !context->reader || (state.reader && state.reader != context->reader))
        throw std::runtime_error("CPU expert backend requires one model reader");
    state.reader = context->reader;
    for (const auto & projection : context->projections)
        if (!state.weights.emplace(projection.mapped, context).second)
            throw std::runtime_error("Duplicate CPU expert mapping");
}

inline uint64_t host_expert_backend_bypassed(ggml_backend_t backend) {
    return backend ? static_cast<host_expert_backend_state *>(backend->context)->bypassed_graphs : 0;
}
} // namespace tsg_dsv4
#endif
