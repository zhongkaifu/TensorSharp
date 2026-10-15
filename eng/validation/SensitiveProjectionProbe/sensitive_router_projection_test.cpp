// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_matmul_precision.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#include "ggml_ops_dsv4_fused.h"
#endif
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <numeric>
#include <vector>
#include "precision_test_utils.h"

// Independent F32 router oracle. The diagnostic outputs deliberately expose
// logits/probabilities/IDs, so this is not an ordinary model fusion/performance
// test. No shared allocation ledger or process physical limit is installed.
static void check_router(ggml_backend_t backend, int tokens, bool near_tie, bool precise, bool owned_cuda)
{
    constexpr int inner = 256, experts = 16, used = 8;
    constexpr double maximum_logit_error = 5e-5, maximum_relative_error = 5e-6;
    constexpr double maximum_probability_error = 2e-7, maximum_weight_error = 1e-6;
    auto* ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot create router oracle context");
    input_tensor w(ctx, GGML_TYPE_F32, {inner, experts, 1, 1}, true);
    input_tensor x(ctx, GGML_TYPE_F32, {inner, tokens, 1, 1}, true);
    input_tensor bias(ctx, GGML_TYPE_F32, {experts, 1, 1, 1}, true);
    for (int row = 0; row < experts; ++row) {
        for (int k = 0; k < inner; ++k)
            w.values[size_t(row) * inner + k] = near_tie
                ? (k == 0 ? float(row) : k == 1 ? -float(row) : 0.0f)
                : float((row * 17 + k * 31) % 127 - 63) / 4096.0f;
        bias.values[row] = near_tie ? std::ldexp(row % 2 ? 1.0f : -1.0f, -23) : float(row) / 8.0f;
    }
    for (int column = 0; column < tokens; ++column)
        for (int k = 0; k < inner; ++k)
            x.values[size_t(column) * inner + k] = near_tie
                ? (k == 0 ? 1.0f + std::ldexp(float(column % 4 + 1), -16) : k == 1 ? 1.0f : 0.0f)
                : std::sin(float(k * 13 + column * 7) * 0.019f) + float(column % 5) / 16384.0f;

    const auto role = precise ? tsg_projection_role::routing_logits : tsg_projection_role::throughput;
    auto* logits = ggml_mul_mat(ctx, tsg_projection_weight(ctx, w.tensor, role), x.tensor);
    if (precise) tsg_require_projection_precision(ctx, logits, role, owned_cuda);
    else ggml_prec_set_acc(logits, GGML_PREC_F32); // Existing GLM router baseline, including its TF32 allowance.
    require(logits->op == (precise && owned_cuda ? GGML_OP_CUSTOM : GGML_OP_MUL_MAT),
        "Router did not use the declared precise/ordinary construction");
    auto* probs = ggml_sigmoid(ctx, logits);
    auto* selection = ggml_add(ctx, probs, bias.tensor);
    auto* ids = ggml_argsort_top_k(ctx, selection, used);
    auto* packed_ids = ggml_cont(ctx, ids); // Top-k view has an experts-wide row stride.
    auto* weights = ggml_get_rows(ctx, ggml_reshape_3d(ctx, probs, 1, experts, tokens), ids);
    weights = ggml_reshape_2d(ctx, weights, used, tokens);
    weights = ggml_div(ctx, weights, ggml_clamp(ctx, ggml_sum_rows(ctx, weights), 6.103515625e-5f, INFINITY));
    for (auto* output : {logits, probs, packed_ids, weights}) ggml_set_output(output);
    auto* graph = ggml_new_graph(ctx);
    for (auto* output : {logits, probs, packed_ids, weights}) ggml_build_forward_expand(graph, output);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "Cannot allocate independent router fixture");
    w.upload(); x.upload(); bias.upload();
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Router fixture graph failed");
    ggml_backend_synchronize(backend);
    std::vector<float> actual_logits(size_t(experts) * tokens), actual_probs(actual_logits.size());
    std::vector<int32_t> actual_ids(size_t(used) * tokens);
    std::vector<float> actual_weights(actual_ids.size());
    ggml_backend_tensor_get(logits, actual_logits.data(), 0, actual_logits.size() * sizeof(float));
    ggml_backend_tensor_get(probs, actual_probs.data(), 0, actual_probs.size() * sizeof(float));
    ggml_backend_tensor_get(packed_ids, actual_ids.data(), 0, actual_ids.size() * sizeof(int32_t));
    ggml_backend_tensor_get(weights, actual_weights.data(), 0, actual_weights.size() * sizeof(float));

    double max_logit = 0, squared = 0, reference_squared = 0, max_probability = 0, max_weight = 0;
    double minimum_cut_gap = INFINITY, maximum_normalization_error = 0;
    size_t differing_ids = 0, differing_exact_cancellations = 0;
    for (int column = 0; column < tokens; ++column) {
        std::array<double, experts> oracle_logits{}, oracle_probs{}, oracle_selection{};
        std::array<int, experts> order{};
        std::iota(order.begin(), order.end(), 0);
        for (int row = 0; row < experts; ++row) {
            for (int k = 0; k < inner; ++k) oracle_logits[row] += double(w.at(k, row)) * x.at(k, column);
            oracle_probs[row] = 1.0 / (1.0 + std::exp(-oracle_logits[row]));
            oracle_selection[row] = oracle_probs[row] + double(bias.at(row, 0));
            const size_t index = size_t(column) * experts + row;
            require(std::isfinite(actual_logits[index]) && std::isfinite(actual_probs[index]), "Router produced nonfinite values");
            const double error = double(actual_logits[index]) - oracle_logits[row];
            max_logit = std::max(max_logit, std::abs(error)); squared += error * error;
            reference_squared += oracle_logits[row] * oracle_logits[row];
            max_probability = std::max(max_probability, std::abs(double(actual_probs[index]) - oracle_probs[row]));
            if (near_tie && actual_logits[index] != float(oracle_logits[row])) ++differing_exact_cancellations;
        }
        std::sort(order.begin(), order.end(), [&](int a, int b) { return oracle_selection[a] > oracle_selection[b]; });
        const double cut_gap = oracle_selection[order[used - 1]] - oracle_selection[order[used]];
        minimum_cut_gap = std::min(minimum_cut_gap, cut_gap);
        require(cut_gap > 8 * maximum_probability_error,
            "Independent router oracle has insufficient separation at its hard top-8 cut");
        double selected_sum = 0, actual_sum = 0;
        for (int slot = 0; slot < used; ++slot) {
            if (near_tie) require(order[slot] == experts - slot - 1, "Near-tie oracle construction changed its top-8 order");
            selected_sum += oracle_probs[order[slot]];
        }
        for (int slot = 0; slot < used; ++slot) {
            const size_t index = size_t(column) * used + slot;
            require(actual_ids[index] >= 0 && actual_ids[index] < experts, "Router selected an invalid expert");
            require(std::isfinite(actual_weights[index]) && actual_weights[index] > 0, "Router produced invalid normalized weights");
            differing_ids += actual_ids[index] != order[slot];
            max_weight = std::max(max_weight, std::abs(double(actual_weights[index]) - oracle_probs[order[slot]] / selected_sum));
            actual_sum += actual_weights[index];
        }
        maximum_normalization_error = std::max(maximum_normalization_error, std::abs(actual_sum - 1.0));
    }
    const double relative = std::sqrt(squared / reference_squared);
    if (precise) {
        require(max_logit <= maximum_logit_error && relative <= maximum_relative_error, "F32 router failed its independent FP64 logit gate");
        require(max_probability <= maximum_probability_error, "F32 router probability error exceeds the declared top-8 margin bound");
        require(differing_ids == 0, "F32 router changed independently specified top-8 IDs or order");
        require(!near_tie || differing_exact_cancellations == 0, "F32 router narrowed an exactly representable activation residual");
        require(max_weight <= maximum_weight_error && maximum_normalization_error <= maximum_weight_error,
            "F32 router did not normalize the selected unbiased probabilities");
    }
    w.check_unchanged(); x.check_unchanged(); bias.check_unchanged();
    std::printf("SENSITIVE_ROUTER backend=%s rows=%d case=%s policy=%s full_logits=%zu top8_ids=%zu differing_ids=%zu exact_cancel_differences=%zu max_abs=%.9g relative_l2=%.9g probability_max_abs=%.9g weight_max_abs=%.9g normalization_max_abs=%.9g minimum_cut_gap=%.9g\n",
        ggml_backend_name(backend), tokens, near_tie ? "near-tie" : "dense", precise ? "owned-role" : "ordinary-observation",
        actual_logits.size(), actual_ids.size(), differing_ids, differing_exact_cancellations, max_logit, relative,
        max_probability, max_weight, maximum_normalization_error, minimum_cut_gap);
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef TSG_GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    require(!tsg_f32_projection_environment_conflicts(std::getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE")),
        "Clear conflicting cuBLAS override before router qualification");
    auto* ordinary = ggml_backend_cuda_init(0);
    require(ordinary != nullptr, "Cannot initialize original CUDA backend");
    auto* precise = tsg_dsv4_fused_backend_init(ordinary);
    constexpr bool owned_cuda = true;
#else
    auto* ordinary = ggml_backend_cpu_init();
    if (ordinary) ggml_backend_cpu_set_n_threads(ordinary, 4);
    auto* precise = ordinary;
    constexpr bool owned_cuda = false;
#endif
    require(precise != nullptr, "Cannot initialize independent router backend");
    for (int tokens : {1, 3, 8, 512, 774, 809})
        for (bool near_tie : {true, false}) {
            check_router(precise, tokens, near_tie, true, owned_cuda);
            check_router(ordinary, tokens, near_tie, false, false);
        }
#ifdef TSG_GGML_USE_CUDA
    ggml_backend_free(precise);
#endif
    ggml_backend_free(ordinary);
    std::printf("PASS: independent F32 router FP64 oracle, hard top-8 near-tie/dense cases, complete logits/probabilities/IDs/weights, preserved inputs/guards and explicit ggml lifecycle release; no shared-ledger, physical-hard-cap, whole-model quality or performance claim\n");
    return 0;
}
