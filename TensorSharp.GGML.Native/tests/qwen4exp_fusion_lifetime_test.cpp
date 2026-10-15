// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_qwen4exp_fusion_lifetime.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <cstdint>
#include <cstdio>
#include <memory>
#include <stdexcept>

static int checks = 0;
static void require(bool condition) {
    ++checks;
    if (!condition) throw std::runtime_error("Qwen4Exp fusion lifetime invariant failed");
}
static bool overlaps(const ggml_tensor* a, const ggml_tensor* b) {
    const auto x = reinterpret_cast<std::uintptr_t>(a->data);
    const auto y = reinterpret_cast<std::uintptr_t>(b->data);
    return x < y + ggml_nbytes(b) && y < x + ggml_nbytes(a);
}

static int index_of(ggml_cgraph* graph, ggml_tensor* tensor) {
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
        if (ggml_graph_node(graph, i) == tensor) return i;
    return -1;
}

// The real upstream DFS used to insert the arena's position VIEW between K's
// MUL and ROPE. Q (visited later) did not have the interruption. Exercise the
// same production ordering helper with an explicit unfixed negative control.
static void rotary_position_ordering() {
    for (int slot : {0, 1}) for (bool view : {false, true}) for (bool prepare : {false, true}) {
        ggml_init_params params{ggml_tensor_overhead() * 64 + ggml_graph_overhead_custom(64, false), nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        require(ctx != nullptr);
        auto* graph = ggml_new_graph_custom(ctx.get(), 64, false);
        auto* positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, view ? 2 : 1);
        ggml_set_input(positions);
        auto* pos = view ? ggml_view_1d(ctx.get(), positions, 1, slot * sizeof(int32_t)) : positions;
        if (prepare) tsg::qwen4exp_expand_rotary_position_input(graph, pos);
        auto* gamma = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 256);
        for (int branch = 0; branch < 2; ++branch) {
            auto* input = ggml_new_tensor_3d(ctx.get(), GGML_TYPE_F32, 256, branch == 0 ? 2 : 24, 1);
            auto* norm = ggml_rms_norm(ctx.get(), input, 1e-6f);
            auto* scaled = ggml_mul(ctx.get(), norm, gamma);
            auto* rotated = ggml_rope_ext(ctx.get(), scaled, pos, nullptr, 64, 2,
                0, 1e7f, 1.f, 0.f, 1.f, 0.f, 0.f);
            ggml_build_forward_expand(graph, rotated);
            const int n = index_of(graph, norm);
            require(n >= 0 && index_of(graph, scaled) == n + 1);
            const bool adjacent = index_of(graph, rotated) == n + 2;
            require(adjacent == (!view || prepare || branch == 1));
            if (view && !prepare && branch == 0) require(index_of(graph, pos) == n + 2);
            require((norm->flags & GGML_TENSOR_FLAG_OUTPUT) == 0);
            require((scaled->flags & GGML_TENSOR_FLAG_OUTPUT) == 0);
        }
        require(ggml_graph_n_nodes(graph) == (view ? 7 : 6));
        require((pos->flags & GGML_TENSOR_FLAG_OUTPUT) == 0);
        require(pos->view_src == (view ? positions : nullptr));
    }
}

// This tests the production lifetime helper and a real CPU gallocr, not CUDA
// fusion arithmetic. Vary graph pressure and view nesting; a protected producer
// must remain physically disjoint from normalization/rotary destinations.
int main() {
    rotary_position_ordering();
    for (int rows : {1, 2, 8}) for (bool pressure : {false, true}) {
        ggml_init_params params{ggml_tensor_overhead() * 256 + ggml_graph_overhead_custom(256, false), nullptr, true};
        std::unique_ptr<ggml_context, decltype(&ggml_free)> ctx(ggml_init(params), ggml_free);
        require(ctx != nullptr);
        auto* input = ggml_new_tensor_2d(ctx.get(), GGML_TYPE_F32, 64 * 8, rows);
        ggml_set_input(input);
        auto* producer = ggml_scale(ctx.get(), input, 0.5f);
        auto* query = ggml_reshape_3d(ctx.get(), producer, 64, 8, rows);
        auto* nested = ggml_view_3d(ctx.get(), query, 64, 8, rows, query->nb[1], query->nb[2], 0);
        tsg::qwen4exp_keep_fusion_input(nested, false);
        require((producer->flags & GGML_TENSOR_FLAG_OUTPUT) == 0);
        tsg::qwen4exp_keep_fusion_input(nullptr, true);
        tsg::qwen4exp_keep_fusion_input(input, true);
        require((input->flags & GGML_TENSOR_FLAG_OUTPUT) == 0); // leaf needs no producer pin
        tsg::qwen4exp_keep_fusion_input(nested, true);
        require((producer->flags & GGML_TENSOR_FLAG_OUTPUT) != 0);
        auto* gamma = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_F32, 64);
        ggml_set_input(gamma);
        auto* positions = ggml_new_tensor_1d(ctx.get(), GGML_TYPE_I32, rows);
        ggml_set_input(positions);
        auto* norm = ggml_rms_norm(ctx.get(), query, 1e-6f);
        auto* scaled = ggml_mul(ctx.get(), norm, gamma);
        auto* rotated = ggml_rope_ext(ctx.get(), scaled, positions, nullptr, 32, 2,
            0, 10000.f, 1.f, 0.f, 1.f, 0.f, 0.f);
        auto* graph = ggml_new_graph_custom(ctx.get(), 256, false);
        if (pressure) {
            auto* scratch = ggml_add(ctx.get(), input, ggml_scale(ctx.get(), input, 2.f));
            ggml_build_forward_expand(graph, scratch);
        }
        ggml_set_output(rotated);
        ggml_build_forward_expand(graph, rotated);
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)> allocation(
            ggml_gallocr_new(ggml_backend_cpu_buffer_type()), ggml_gallocr_free);
        require(allocation != nullptr && ggml_gallocr_alloc_graph(allocation.get(), graph));
        require(producer->data != nullptr && rotated->data != nullptr);
        require(!overlaps(producer, norm));
        require(!overlaps(producer, scaled));
        require(!overlaps(producer, rotated));
    }
    std::printf("PASS: %d production fusion-lifetime checks; real CPU gallocr, no CUDA arithmetic claim\n", checks);
}
