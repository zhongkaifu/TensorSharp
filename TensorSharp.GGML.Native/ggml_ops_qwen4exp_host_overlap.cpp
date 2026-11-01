// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_qwen4exp_host_overlap.h"
#include "ggml-impl.h"
#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace tsg::q4e_host_overlap {
namespace {
bool pending_view(const ggml_tensor* tensor, const ggml_tensor* pending) {
    for (int depth = 0; tensor; ++depth, tensor = tensor->view_src) {
        if (tensor == pending || depth == 64) return true;
    }
    return false;
}
bool metadata(const ggml_tensor* tensor) {
    return tensor->op == GGML_OP_VIEW || tensor->op == GGML_OP_RESHAPE;
}
bool allowed(const ggml_tensor* node) {
    switch (node->op) {
        case GGML_OP_MUL_MAT: case GGML_OP_MUL: case GGML_OP_VIEW:
        case GGML_OP_RESHAPE: case GGML_OP_CONT: return true;
        case GGML_OP_UNARY: return ggml_get_unary_op(node) == GGML_UNARY_OP_SILU ||
            ggml_get_unary_op(node) == GGML_UNARY_OP_SIGMOID;
        default: return false; // In particular CPY/SET/ACC and custom ops.
    }
}
bool range(const ggml_tensor* tensor, std::uintptr_t& begin, std::uintptr_t& end) {
    if (!tensor || !tensor->data) return false;
    begin = reinterpret_cast<std::uintptr_t>(tensor->data);
    const std::size_t bytes = ggml_nbytes(tensor);
    if (!bytes || bytes > std::numeric_limits<std::uintptr_t>::max() - begin) return false;
    end = begin + bytes;
    return true;
}
bool aliases(const ggml_tensor* a, const ggml_tensor* b) {
    std::uintptr_t ab, ae, bb, be;
    return range(a, ab, ae) && range(b, bb, be) && ab < be && bb < ae;
}
bool pinned(const ggml_tensor* tensor) {
    for (int depth = 0; tensor; ++depth, tensor = tensor->view_src)
        if (depth == 64 || !(tensor->flags & GGML_TENSOR_FLAG_OUTPUT)) return false;
    return true;
}
struct Stats {
    std::array<std::atomic<unsigned long long>, static_cast<unsigned>(Count::count)> counts{};
    std::array<std::atomic<unsigned long long>, static_cast<unsigned>(Reject::count)> rejects{};
    std::atomic<unsigned long long> submit_us{0}, drain_us{0}, serial_us{0};
    std::atomic<bool> reported{false};
    Stats() { for (auto& value : counts) value.store(0); for (auto& value : rejects) value.store(0); }
    ~Stats() { report(); }
};
Stats& stats() { static Stats value; return value; }
} // namespace

int prefix_end(ggml_cgraph* graph, int begin, int limit, const ggml_tensor* pending, Reject& why) {
    why = Reject::geometry;
    if (!graph || !pending || begin < 0 || limit <= begin || limit > ggml_graph_n_nodes(graph)) return -1;
    // Qwen's independent shared branch begins with its complete gate matmul,
    // never with an activation whose producer could be fused before the cut.
    if (ggml_graph_node(graph, begin)->op != GGML_OP_MUL_MAT) { why = Reject::fusion_boundary; return -1; }
    for (int i = begin; i < limit && i - begin <= 64; ++i) {
        const ggml_tensor* node = ggml_graph_node(graph, i);
        bool dependent = pending_view(node, pending);
        for (const auto* source : node->src) dependent |= source && pending_view(source, pending);
        if (dependent) {
            if (i == begin) { why = Reject::no_prefix; return -1; }
            // facdc44a's Metal fusion table has no generic MUL+ADD fusion.
            // This exact rejoin keeps the matmul/activation fusions inside the
            // prefix and never cuts a MUL_MAT+ADD or ADD-chain fusion.
            const ggml_tensor* last = ggml_graph_node(graph, i - 1);
            if (node->op != GGML_OP_ADD || node->src[0] != pending || node->src[1] != last || last->op != GGML_OP_MUL) {
                why = Reject::fusion_boundary; return -1;
            }
            why = Reject::none;
            return i;
        }
        if (!allowed(node)) { why = Reject::operation; return -1; }
        // Full-graph order is a contract, including inputs produced before the
        // original seam. Never pull a future producer into this prefix.
        for (const auto* source : node->src) {
            if (!source || source->op == GGML_OP_NONE) continue;
            bool produced = false;
            for (int p = 0; p < i; ++p) if (ggml_graph_node(graph, p) == source) { produced = true; break; }
            if (!produced) { why = Reject::dependency; return -1; }
        }
    }
    why = Reject::no_prefix;
    return -1;
}

bool safe_allocations(ggml_cgraph* graph, int begin, int end, const ggml_tensor* pending,
    const ggml_tensor* const* inputs, int input_count, Reject& why) {
    why = Reject::geometry;
    if (!graph || !pending || !inputs || input_count <= 0 || begin < 0 || end <= begin || end > ggml_graph_n_nodes(graph)) return false;
    why = Reject::unpinned;
    if ((pending->flags & (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) != (GGML_TENSOR_FLAG_INPUT | GGML_TENSOR_FLAG_OUTPUT)) return false;
    for (int j = 0; j < input_count; ++j) if (!inputs[j] || !pinned(inputs[j])) return false;
    std::uintptr_t lo, hi;
    why = Reject::allocation;
    if (!range(pending, lo, hi)) return false;
    for (int j = 0; j < input_count; ++j) if (!range(inputs[j], lo, hi)) return false;
    for (int i = begin; i < end; ++i) {
        const ggml_tensor* node = ggml_graph_node(graph, i);
        why = Reject::allocation;
        if (!range(node, lo, hi)) return false;
        why = Reject::alias;
        if (pending_view(node, pending) || aliases(node, pending)) return false;
        for (const auto* source : node->src) {
            if (!source) continue;
            why = Reject::allocation;
            if (!range(source, lo, hi)) return false;
            why = Reject::alias;
            if (pending_view(source, pending) || aliases(source, pending)) return false;
        }
        if (!metadata(node))
            for (int j = 0; j < input_count; ++j) if (aliases(node, inputs[j])) return false;
    }
    why = Reject::none;
    return true;
}

int mode() {
    static const int selected = [] {
        const char* value = std::getenv("TS_Q4E_METAL_HOST_OVERLAP");
        return value && std::strcmp(value, "1") == 0 ? 1 : value && std::strcmp(value, "2") == 0 ? 2 : 0;
    }();
    return selected;
}
void count(Count value, unsigned long long amount) { stats().counts[static_cast<unsigned>(value)].fetch_add(amount, std::memory_order_relaxed); }
void reject(Reject value) { stats().rejects[static_cast<unsigned>(value)].fetch_add(1, std::memory_order_relaxed); }
const char* reject_name(Reject value) {
    static const char* names[] = { "none", "geometry", "no_prefix", "dependency", "operation", "fusion_boundary", "unpinned", "allocation", "alias" };
    return names[static_cast<unsigned>(value)];
}
void time(double submit_ms, double drain_ms, double serial_ms) {
    stats().submit_us.fetch_add(static_cast<unsigned long long>(std::max(0.0, submit_ms) * 1000), std::memory_order_relaxed);
    stats().drain_us.fetch_add(static_cast<unsigned long long>(std::max(0.0, drain_ms) * 1000), std::memory_order_relaxed);
    stats().serial_us.fetch_add(static_cast<unsigned long long>(std::max(0.0, serial_ms) * 1000), std::memory_order_relaxed);
}
void report() {
    if (!mode()) return;
    Stats& value = stats();
    std::array<unsigned long long, static_cast<unsigned>(Count::count)> totals{};
    bool nonzero = false;
    for (unsigned i = 0; i < totals.size(); ++i) { totals[i] = value.counts[i].exchange(0); nonzero |= totals[i] != 0; }
    if (!nonzero && value.reported.exchange(true)) return;
    value.reported.store(true);
    std::fprintf(stderr, "[Q4E-HOST-OVERLAP] mode=%d considered=%llu ineligible=%llu rejected=%llu eligible=%llu async=%llu serial=%llu prefix_nodes=%llu "
        "submit_failures=%llu drain_failures=%llu fast_fallbacks=%llu cached_cuts=%llu submit_wall_ms=%.3f drain_wall_ms=%.3f serial_prefix_wall_ms=%.3f "
        "(wall timings include host encoding/waits; not GPU-device duration)\n",
        mode(), totals[0], totals[1], totals[2], totals[3], totals[4], totals[5], totals[6], totals[7], totals[8], totals[9], totals[10],
        value.submit_us.exchange(0) / 1000.0, value.drain_us.exchange(0) / 1000.0, value.serial_us.exchange(0) / 1000.0);
    for (unsigned i = 0; i < static_cast<unsigned>(Reject::count); ++i)
        if (const auto amount = value.rejects[i].exchange(0)) std::fprintf(stderr, "[Q4E-HOST-OVERLAP] rejected_%s=%llu\n", reject_name(static_cast<Reject>(i)), amount);
    std::fflush(stderr);
}
} // namespace tsg::q4e_host_overlap
