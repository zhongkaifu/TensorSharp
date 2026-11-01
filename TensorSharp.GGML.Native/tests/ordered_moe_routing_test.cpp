// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Exercise the production ordered selector with unchanged ggml CPU/CUDA backends.
// --legacy-top-k substitutes the former selector while retaining every assertion;
// it is a manual reproduction mode, not an expected-failure CTest or a passing skip.
#include "ggml_ops_transformer_common.h"
#include "ggml-cpu.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
constexpr int experts = 256, used = 8, channels = 4;
constexpr int rebuilds = 3, repeats = 8;
constexpr std::array<float, used> distinct_scores{129.f, 64.f, 32.f, 16.f, 8.f, 4.f, 2.f, 1.f};

void require(bool condition, const std::string& message) {
    if (!condition) throw std::runtime_error(message);
}

template<class T> bool same_bits(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}

struct Snapshot {
    std::vector<std::int32_t> ids;
    std::vector<float> weights, output;
};

struct Context {
    ggml_context* value = ggml_init({1u << 20, nullptr, true});
    Context() { require(value != nullptr, "context allocation failed"); }
    ~Context() { ggml_free(value); }
};
struct Buffer {
    ggml_backend_buffer_t value = nullptr;
    ~Buffer() { if (value) ggml_backend_buffer_free(value); }
};

enum class WeightFormula { ExactNormalized, NonPowerNormalized, SelectedLogitSoftmax };

struct Fixture {
    Context context;
    Buffer buffer;
    ggml_cgraph* graph = nullptr;
    ggml_tensor *scores = nullptr, *values = nullptr, *ids = nullptr, *weights = nullptr, *output = nullptr;
    int rows;
    int expert_count, used_count;
    bool ties;
    WeightFormula formula;
    std::string label;
    std::vector<float> score_data, value_data;
    std::vector<std::int32_t> canonical;

    Fixture(ggml_backend_t backend, int count, bool four_dimensional, bool tied, bool legacy,
            int n_experts = experts, int n_used = used, WeightFormula weighting = WeightFormula::ExactNormalized)
        : rows(count), expert_count(n_experts), used_count(n_used), ties(tied), formula(weighting),
          label(std::string(tied ? "ties/" : "distinct/") +
            (four_dimensional ? "4x2x4" : "T" + std::to_string(count))),
          score_data(expert_count * rows), value_data(channels * expert_count * rows), canonical(used_count * rows) {
        if (formula != WeightFormula::ExactNormalized)
            label = std::string(formula == WeightFormula::SelectedLogitSoftmax ? "selected-softmax/" : "non-power-normalized/") +
                "E" + std::to_string(expert_count) + "/K" + std::to_string(used_count) + "/T" + std::to_string(rows);
        auto* ctx = context.value;
        scores = four_dimensional
            ? ggml_new_tensor_4d(ctx, GGML_TYPE_F32, expert_count, 4, 2, 4)
            : ggml_new_tensor_2d(ctx, GGML_TYPE_F32, expert_count, rows);
        values = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, channels, expert_count, rows);
        ggml_set_name(scores, "routing_scores"); ggml_set_input(scores);
        ggml_set_name(values, "expert_values"); ggml_set_input(values);
        ids = legacy ? ggml_top_k(ctx, scores, used_count) : tsg::ordered_moe_top_k(ctx, scores, used_count);
        ggml_set_name(ids, "ordered_expert_ids"); ggml_set_output(ids);
        require(ids->type == GGML_TYPE_I32 && ids->ne[0] == used_count, label + ": wrong ID type/width");
        std::size_t stride = sizeof(std::int32_t);
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            require(ids->ne[d] == (d == 0 ? used_count : scores->ne[d]), label + ": row dimensions changed");
            require(ids->nb[d] == stride, label + ": selected IDs are not compact");
            stride *= static_cast<std::size_t>(ids->ne[d]);
        }
        require(ggml_is_contiguous(ids) && ggml_nbytes(ids) == used_count * rows * sizeof(std::int32_t),
            label + ": selected IDs retain full-argsort row gaps");

        auto* ids_2d = ggml_reshape_2d(ctx, ids, used_count, rows);
        auto* gathered_scores = ggml_get_rows(ctx, ggml_reshape_3d(ctx, scores, 1, expert_count, rows), ids_2d);
        auto* selected_scores = ggml_reshape_2d(ctx, gathered_scores, used_count, rows);
        weights = formula == WeightFormula::SelectedLogitSoftmax
            ? ggml_soft_max(ctx, selected_scores)
            : ggml_div(ctx, selected_scores, ggml_sum_rows(ctx, selected_scores));
        ggml_set_name(weights, "normalized_routing_weights"); ggml_set_output(weights);
        auto* selected_values = ggml_get_rows(ctx, values, ids_2d);
        auto* weighted = ggml_mul(ctx, selected_values, ggml_reshape_3d(ctx, weights, 1, used_count, rows));
        // The expert slot order is also the accumulation order used by routed FFNs.
        // Keep the explicit add chain, including a cancellation-sensitive channel.
        for (int slot = 0; slot < used_count; ++slot) {
            auto* term = ggml_view_2d(ctx, weighted, channels, rows, weighted->nb[2], slot * weighted->nb[1]);
            output = output ? ggml_add(ctx, output, term) : term;
        }
        ggml_set_name(output, "weighted_expert_sum"); ggml_set_output(output);
        graph = ggml_new_graph_custom(ctx, 128, false);
        ggml_build_forward_expand(graph, ids);
        ggml_build_forward_expand(graph, weights);
        ggml_build_forward_expand(graph, output);
        buffer.value = ggml_backend_alloc_ctx_tensors(ctx, backend);
        require(buffer.value != nullptr, label + ": backend allocation failed");

        for (int row = 0; row < rows; ++row) {
            for (int rank = 0; rank < expert_count; ++rank) {
                const int id = (rank * 73 + row * 19 + 11) % expert_count;
                float score;
                if (formula == WeightFormula::SelectedLogitSoftmax)
                    // Negative raw logits distinguish softmax(selected logits)
                    // from probability normalization; each row also has an offset.
                    score = (rank < used_count ? -1.75f - rank * .37f : -8.f - rank * .025f) + row * .013f;
                else if (formula == WeightFormula::NonPowerNormalized)
                    score = rank < used_count ? 1.11f - rank * .07f + row * .003f : .3f / (rank + 1);
                else
                    score = ties && rank < 16 ? 2.f :
                        (!ties && rank < used_count ? distinct_scores[rank] : (expert_count - rank) / 2048.f);
                score_data[row * expert_count + id] = score;
                if (rank < used_count) canonical[row * used_count + rank] = id;
                float* v = &value_data[(row * expert_count + id) * channels];
                // Distinct selected scores sum to 256: slots1/2/3 contribute
                // exactly +2^26, -2^26, +1. Permuting the last two loses the1.
                v[0] = ties ? static_cast<float>((id % 17) - 8) :
                    (rank == 1 ? 268435456.f : rank == 2 ? -536870912.f : rank == 3 ? 16.f : 0.f);
                v[1] = static_cast<float>(id + 1);
                v[2] = static_cast<float>((id % 13) - 6);
                v[3] = static_cast<float>(row + 1);
                if (formula != WeightFormula::ExactNormalized) {
                    // Bounded signed values keep the scalar formula oracle
                    // sensitive to weight/ID mismatch without amplifying its
                    // permitted exp/div rounding into an ill-conditioned sum.
                    v[0] = static_cast<float>((id % 17) - 8) + rank * .03125f;
                    v[1] = id / 32.f + .125f;
                }
            }
        }
    }

    Snapshot compute(ggml_backend_t backend) {
        // Repeat uploads as inference does; inputs and their metadata stay fixed.
        ggml_backend_tensor_set(scores, score_data.data(), 0, score_data.size() * sizeof(float));
        ggml_backend_tensor_set(values, value_data.data(), 0, value_data.size() * sizeof(float));
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, label + ": graph compute failed");
        Snapshot result;
        result.ids.resize(used_count * rows); result.weights.resize(used_count * rows); result.output.resize(channels * rows);
        ggml_backend_tensor_get(ids, result.ids.data(), 0, result.ids.size() * sizeof(std::int32_t));
        ggml_backend_tensor_get(weights, result.weights.data(), 0, result.weights.size() * sizeof(float));
        ggml_backend_tensor_get(output, result.output.data(), 0, result.output.size() * sizeof(float));
        return result;
    }

    void check(const Snapshot& result) const {
        for (int row = 0; row < rows; ++row) {
            std::vector<bool> seen(expert_count, false);
            float previous = INFINITY;
            double total = 0;
            std::array<float, channels> expected{};
            std::array<double, channels> formula_output{}, formula_scale{};
            std::vector<double> reference_weights(used_count);
            if (formula != WeightFormula::ExactNormalized) {
                const double maximum = score_data[row * expert_count + canonical[row * used_count]];
                double denominator = 0;
                for (int slot = 0; slot < used_count; ++slot) {
                    const double score = score_data[row * expert_count + canonical[row * used_count + slot]];
                    reference_weights[slot] = formula == WeightFormula::SelectedLogitSoftmax
                        ? std::exp(score - maximum) : score;
                    denominator += reference_weights[slot];
                }
                if (formula == WeightFormula::NonPowerNormalized) {
                    int exponent;
                    require(std::frexp(denominator, &exponent) != .5,
                        label + ": normalization denominator accidentally became a power of two");
                }
                for (double& weight : reference_weights) weight /= denominator;
            }
            for (int slot = 0; slot < used_count; ++slot) {
                const int offset = row * used_count + slot;
                const int id = result.ids[offset];
                require(id >= 0 && id < expert_count && !seen[id], label + ": invalid or duplicated expert ID");
                seen[id] = true;
                const float score = score_data[row * expert_count + id];
                require(score <= previous, label + ": IDs are not in descending score order at row " +
                    std::to_string(row) + ", slot " + std::to_string(slot));
                previous = score;
                if (!ties) require(id == canonical[offset], label + ": wrong selected expert");
                else require(score == 2.f, label + ": tied selection included a lower-scoring expert");
                const double expected_weight = formula == WeightFormula::ExactNormalized
                    ? (ties ? .125f : distinct_scores[slot] / 256.f) : reference_weights[slot];
                const double weight_error = std::abs(static_cast<double>(result.weights[offset]) - expected_weight);
                // The formula oracle uses double exp and accumulation. This
                // tolerance permits F32 CPU/CUDA exp/div rounding, not a wrong
                // selected ID or a permuted probability. Repeat checks stay bit-exact.
                require(std::isfinite(result.weights[offset]) && (formula == WeightFormula::ExactNormalized
                    ? weight_error == 0 : weight_error <= 2e-7 + 2e-6 * std::abs(expected_weight)),
                    label + ": routing weight detached from selected ID or weighted incorrectly");
                total += result.weights[offset];
                for (int channel = 0; channel < channels; ++channel) {
                    // Force separate F32 multiply/add, matching the two graph ops.
                    const float value = value_data[(row * expert_count + id) * channels + channel];
                    volatile float product = value * static_cast<float>(expected_weight);
                    volatile float sum = expected[channel] + product;
                    expected[channel] = sum;
                    formula_output[channel] += static_cast<double>(value) * expected_weight;
                    formula_scale[channel] += std::abs(static_cast<double>(value) * expected_weight);
                }
            }
            require(formula == WeightFormula::ExactNormalized ? total == 1.0 : std::abs(total - 1.0) <= 2e-6,
                label + ": normalized weights do not sum to one");
            if (!ties && formula == WeightFormula::ExactNormalized) require(result.output[row * channels] == 1.f,
                label + ": cancellation-sensitive ordered expert sum changed");
            for (int channel = 0; channel < channels; ++channel)
                require(std::isfinite(result.output[row * channels + channel]) &&
                    (formula == WeightFormula::ExactNormalized
                        ? result.output[row * channels + channel] == expected[channel]
                        : std::abs(result.output[row * channels + channel] - formula_output[channel]) <=
                            3e-6 + 3e-6 * formula_scale[channel]),
                    label + ": weighted expert sum differs from independent scalar reference");
        }
    }
};

void run(ggml_backend_t backend, bool legacy) {
    volatile float large = 67108864.f, one = 1.f;
    volatile float lose_one = (large + one) - large;
    volatile float keep_one = (large - large) + one;
    require(lose_one != keep_one, "cancellation fixture is not order-sensitive");
    int cases = 0, executions = 0;
    for (bool ties : {false, true}) {
        for (int shape = 0; shape < 3; ++shape) {
            Snapshot first;
            for (int rebuild = 0; rebuild < rebuilds; ++rebuild) {
                Fixture fixture(backend, shape == 0 ? 1 : 32, shape == 2, ties, legacy);
                for (int repeat = 0; repeat < repeats; ++repeat) {
                    Snapshot current = fixture.compute(backend);
                    fixture.check(current);
                    if (rebuild == 0 && repeat == 0) first = current;
                    else {
                        require(same_bits(first.ids, current.ids), fixture.label + ": IDs changed across replay/rebuild");
                        require(same_bits(first.weights, current.weights), fixture.label + ": weights changed across replay/rebuild");
                        require(same_bits(first.output, current.output), fixture.label + ": sum changed across replay/rebuild");
                    }
                    ++executions;
                }
            }
            ++cases;
        }
    }
    // Broader selector shapes and both routing formulas. E512/K10 exercises the
    // helper generically; it does not claim Qwen4exp calls this helper or provide
    // end-to-end model coverage. The existing exact/cancellation cases stay above.
    for (const auto& shape : {std::array<int, 2>{32, 4}, std::array<int, 2>{512, 10}}) {
        for (WeightFormula formula : {WeightFormula::NonPowerNormalized, WeightFormula::SelectedLogitSoftmax}) {
            for (int rows : {1, 32}) {
                Snapshot first;
                for (int rebuild = 0; rebuild < rebuilds; ++rebuild) {
                    Fixture fixture(backend, rows, false, false, legacy, shape[0], shape[1], formula);
                    for (int repeat = 0; repeat < repeats; ++repeat) {
                        Snapshot current = fixture.compute(backend);
                        fixture.check(current);
                        if (rebuild == 0 && repeat == 0) first = current;
                        else {
                            require(same_bits(first.ids, current.ids), fixture.label + ": IDs changed across replay/rebuild");
                            require(same_bits(first.weights, current.weights), fixture.label + ": weights changed across replay/rebuild");
                            require(same_bits(first.output, current.output), fixture.label + ": sum changed across replay/rebuild");
                        }
                        ++executions;
                    }
                }
                ++cases;
            }
        }
    }
    std::printf("PASS: %d routing cases, %d graph builds, %d executions; ordered IDs, compact layout, "
        "paired normalized/selected-softmax weights, non-power sums, cancellation-sensitive sums, "
        "ties, replay/rebuild consistency\n",
        cases, cases * rebuilds, executions);
}
}

int main(int argc, char** argv) {
    bool cuda = false, legacy = false;
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "cpu") == 0) cuda = false;
        else if (std::strcmp(argv[i], "cuda") == 0) cuda = true;
        else if (std::strcmp(argv[i], "--legacy-top-k") == 0) legacy = true;
        else { std::fprintf(stderr, "usage: %s [cpu|cuda] [--legacy-top-k]\n", argv[0]); return 1; }
    }
    ggml_backend_t backend = nullptr;
    if (cuda) {
#ifdef TSG_GGML_USE_CUDA
        if (ggml_backend_cuda_get_device_count() > 0) backend = ggml_backend_cuda_init(0);
#endif
        if (!backend) { std::fprintf(stderr, "SKIP: CUDA backend/device unavailable; no CUDA assertions ran\n"); return 77; }
    } else {
        backend = ggml_backend_cpu_init();
        if (!backend) { std::fprintf(stderr, "FAIL: CPU backend initialization failed\n"); return 1; }
        ggml_backend_cpu_set_n_threads(backend, 4);
    }
    std::printf("backend=%s selector=%s (ties are compared only within this backend)\n",
        cuda ? "cuda" : "cpu", legacy ? "legacy-top-k" : "ordered-moe-top-k");
    try {
        run(backend, legacy);
        ggml_backend_free(backend);
        return 0;
    } catch (const std::exception& ex) {
        std::fprintf(stderr, "FAIL: %s\n", ex.what());
        ggml_backend_free(backend);
        return 1;
    }
}
