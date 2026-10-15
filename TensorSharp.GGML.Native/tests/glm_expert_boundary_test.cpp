// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <stdexcept>

int main() {
    int checks = 0;
    auto require = [&](bool value, const char* message) {
        ++checks; if (!value) throw std::runtime_error(message);
    };
    auto* backend = ggml_backend_cpu_init();
    auto* context = ggml_init({ggml_tensor_overhead() * 128 + ggml_graph_overhead_custom(128, false), nullptr, true});
    require(backend && context, "actual CPU boundary runtime");
    auto* logits = ggml_new_tensor_2d(context, GGML_TYPE_F32, 6, 3);
    ggml_set_input(logits);
    auto* ids = ggml_argsort_top_k(context, logits, 2);
    ggml_set_output(ids);
    require(ids->ne[0] == 2 && ids->nb[1] == 6 * sizeof(int32_t) && !ggml_is_contiguous(ids),
        "top-k view retains full expert row stride");
    auto* packed = ggml_cont(context, ids);
    ggml_set_output(packed);
    auto* gate = ggml_new_tensor_1d(context, GGML_TYPE_F32, 6);
    auto* up = ggml_new_tensor_1d(context, GGML_TYPE_F32, 6);
    ggml_set_input(gate); ggml_set_input(up);
    auto* clipped_gate = ggml_clamp(context, gate, -std::numeric_limits<float>::infinity(), 1.f);
    auto* clipped_up = ggml_clamp(context, up, -1.f, 1.f);
    ggml_set_output(clipped_gate); ggml_set_output(clipped_up);
    auto* activation = ggml_mul(context, ggml_silu(context, clipped_gate), clipped_up);
    ggml_set_output(activation);
    auto* graph = ggml_new_graph_custom(context, 128, false);
    ggml_build_forward_expand(graph, packed); ggml_build_forward_expand(graph, activation);
    auto* allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
    require(allocator && ggml_gallocr_alloc_graph(allocator, graph), "actual boundary graph allocation");
    float routing[18] = {9,8,0,-1,-2,-3, 0,7,8,-1,-2,-3, 0,1,2,7,9,-3};
    float gates[6] = {20,-20,.5f,1.5f,-2,.125f}, ups[6] = {30,-30,4,-.5f,5,.25f};
    ggml_backend_tensor_set(logits, routing, 0, sizeof(routing));
    ggml_backend_tensor_set(gate, gates, 0, sizeof(gates)); ggml_backend_tensor_set(up, ups, 0, sizeof(ups));
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "actual graph packing and clamp execution");
    int32_t raw[6]{}, result[6]{}, expected[6] = {0,1,2,1,4,3};
    ggml_backend_tensor_get(ids, raw, 0, sizeof(raw)); ggml_backend_tensor_get(packed, result, 0, sizeof(result));
    require(!std::equal(raw, raw + 6, expected), "unpacked flat read demonstrates cross-request routing corruption");
    for (int i = 0; i < 6; ++i) require(result[i] == expected[i], "owned packing preserves every independent top-k row");
    float actual_gate[6]{}, actual_up[6]{}, actual_activation[6]{};
    ggml_backend_tensor_get(clipped_gate, actual_gate, 0, sizeof(actual_gate));
    ggml_backend_tensor_get(clipped_up, actual_up, 0, sizeof(actual_up));
    ggml_backend_tensor_get(activation, actual_activation, 0, sizeof(actual_activation));
    for (int i = 0; i < 6; ++i) {
        const float g = std::min(gates[i], 1.f), u = std::clamp(ups[i], -1.f, 1.f);
        require(actual_gate[i] == g && actual_up[i] == u, "GLM clamp keeps negative gate and bounds both up signs");
        require(std::abs(actual_activation[i] - g / (1.f + std::exp(-g)) * u) < 1e-6f,
            "clamp executes before actual SiLU and multiplication");
    }
    ggml_gallocr_free(allocator); ggml_free(context); ggml_backend_free(backend);
    std::printf("PASS %d actual CPU independent routing and saturating pre-SiLU boundary checks\n", checks);
}
