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
#include <cstring>
#include <vector>
#include "precision_test_utils.h"

// Construct checkpoint Q8 bytes directly, without a production quantizer or
// its dot implementation. Guard rows check that execution never writes weights.
struct q8_checkpoint {
    ggml_tensor* storage;
    ggml_tensor* tensor;
    int inner, rows;
    size_t row_bytes;
    std::vector<unsigned char> bytes;

    q8_checkpoint(ggml_context* ctx, int k, int m) : inner(k), rows(m), row_bytes(size_t(k / 32) * 34) {
        require(k % 32 == 0 && ggml_type_size(GGML_TYPE_Q8_0) == 34,
            "Q8 fixture requires the declared 32-value/34-byte checkpoint format");
        storage = ggml_new_tensor_2d(ctx, GGML_TYPE_Q8_0, k, m + 2);
        tensor = ggml_view_2d(ctx, storage, k, m, row_bytes, row_bytes);
        bytes.resize(row_bytes * size_t(m + 2), 0x5a);
        for (int row = 0; row < m; ++row) {
            for (int block = 0; block < k / 32; ++block) {
                auto* dst = bytes.data() + row_bytes * size_t(row + 1) + size_t(block) * 34;
                const uint16_t scale = row == 0 ? uint16_t(0x3c00)
                    : uint16_t(0x1800 + (row * 71 + block * 19) % 1024);
                std::memcpy(dst, &scale, 2);
                for (int j = 0; j < 32; ++j) {
                    const int8_t q = row == 0 ? int8_t(block == 0 && j == 0 ? 1 : 0)
                        : int8_t((row * 29 + block * 61 + j * 37) % 255 - 127);
                    std::memcpy(dst + 2 + j, &q, 1);
                }
            }
        }
    }

    void set_first_difference() {
        // First stage: x[0] - x[1]. Second-stage row zero selects this residual.
        const int8_t minus_one = -1;
        std::memcpy(bytes.data() + row_bytes + 3, &minus_one, 1);
    }

    double at(int k, int row) const {
        const auto* block = bytes.data() + row_bytes * size_t(row + 1) + size_t(k / 32) * 34;
        uint16_t scale; int8_t q;
        std::memcpy(&scale, block, 2); std::memcpy(&q, block + 2 + k % 32, 1);
        // All fixture scales are positive normalized FP16. Decode independently
        // of ggml's half conversion and quantization/dot traits.
        require((scale & 0x8000) == 0 && (scale >> 10) > 0 && (scale >> 10) < 31,
            "Unexpected Q8 fixture scale");
        return std::ldexp(1.0 + double(scale & 1023) / 1024.0, int(scale >> 10) - 15) * q;
    }

    void upload() const { ggml_backend_tensor_set(storage, bytes.data(), 0, bytes.size()); }
    void check_unchanged() const {
        std::vector<unsigned char> actual(bytes.size());
        ggml_backend_tensor_get(storage, actual.data(), 0, actual.size());
        require(actual == bytes, "Sensitive Q8 projection modified checkpoint bytes or guard rows");
    }
};

struct chain_result { std::vector<float> first, second; };

static chain_result check_chain(ggml_backend_t backend, int tokens) {
    constexpr int inner = 256, rank = 64, rows = 128;
    auto* ctx = ggml_init({2 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot create Q8 sensitive projection context");
    q8_checkpoint a(ctx, inner, rank), b(ctx, rank, rows);
    a.set_first_difference();
    input_tensor x(ctx, GGML_TYPE_F32, {inner, tokens, 1, 1}, true);
    const auto role = tsg_projection_role::recurrent_decay;
    require(tsg_projection_weight(ctx, a.tensor, role) == a.tensor,
        "Sensitive Q8 projection widened resident checkpoint weights");
    require(tsg_projection_q8_f32_if_required(ctx, a.tensor, x.tensor, tsg_projection_role::throughput) == nullptr,
        "Ordinary throughput projection changed construction");
    auto* first = tsg_projection_q8_f32_if_required(ctx, a.tensor, x.tensor, role);
    require(first && tsg_is_matmul_q8_f32(first), "Sensitive Q8 role bypassed the owned F32-activation operator");
    auto* second = tsg_projection_q8_f32_if_required(ctx, b.tensor, first, tsg_projection_role::residual_mixing);
    require(second && tsg_is_matmul_q8_f32(second), "Chained sensitive Q8 role bypassed its operator");
    require(first->src[0] == a.tensor && first->src[1] == x.tensor && second->src[0] == b.tensor
        && second->src[1] == first && first->type == GGML_TYPE_F32 && second->type == GGML_TYPE_F32
        && first->ne[2] == 1 && first->ne[3] == 1 && second->ne[2] == 1 && second->ne[3] == 1,
        "Sensitive Q8 projection changed source ownership or two-dimensional shape");
    ggml_set_output(first);
    auto* graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, second);
    auto* buffer = ggml_backend_alloc_ctx_tensors(ctx, backend);
    require(buffer != nullptr, "Cannot allocate Q8 sensitive projection fixture");
    for (int column = 0; column < tokens; ++column) {
        for (int k = 0; k < inner; ++k)
            x.values[size_t(column) * inner + k] = std::sin(float(k * 13 + column * 7) * 0.019f) * 0.8f
                + float((k + column) % 17) * 0.000013f;
        x.values[size_t(column) * inner] = 1.0f + std::ldexp(float(column % 4 + 1), -16);
        x.values[size_t(column) * inner + 1] = 1.0f;
    }
    a.upload(); b.upload(); x.upload();
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS,
        "Q8 sensitive projection graph execution failed");
    ggml_backend_synchronize(backend);
    chain_result actual{std::vector<float>(size_t(rank) * tokens), std::vector<float>(size_t(rows) * tokens)};
    ggml_backend_tensor_get(first, actual.first.data(), 0, actual.first.size() * sizeof(float));
    ggml_backend_tensor_get(second, actual.second.data(), 0, actual.second.size() * sizeof(float));
    std::vector<double> oracle_first(actual.first.size()), oracle_second(actual.second.size());
    for (int column = 0; column < tokens; ++column) {
        for (int row = 0; row < rank; ++row)
            for (int k = 0; k < inner; ++k)
                oracle_first[size_t(column) * rank + row] += a.at(k, row) * x.at(k, column);
        for (int row = 0; row < rows; ++row)
            for (int k = 0; k < rank; ++k)
                oracle_second[size_t(column) * rows + row] += b.at(k, row) * oracle_first[size_t(column) * rank + k];
        const float residual = x.at(0, column) - x.at(1, column);
        require(residual > 0 && actual.first[size_t(column) * rank] == residual
            && actual.second[size_t(column) * rows] == residual,
            "Q8 projection quantized away the F32 activation residual");
    }
    const auto measure = [](const std::vector<float>& values, const std::vector<double>& oracle) {
        double maximum = 0, squared = 0, reference_squared = 0;
        for (size_t i = 0; i < values.size(); ++i) {
            require(std::isfinite(values[i]), "Q8 projection emitted a nonfinite value");
            const double delta = double(values[i]) - oracle[i];
            maximum = std::max(maximum, std::abs(delta)); squared += delta * delta;
            reference_squared += oracle[i] * oracle[i];
        }
        const double relative = std::sqrt(squared / reference_squared);
        require(maximum <= 5e-5 && relative <= 5e-6, "Q8 sensitive chain failed its declared FP64 error gates");
        return std::pair<double, double>{maximum, relative};
    };
    const auto first_error = measure(actual.first, oracle_first), second_error = measure(actual.second, oracle_second);
    a.check_unchanged(); b.check_unchanged(); x.check_unchanged();
    std::printf("SENSITIVE_Q8_PROJECTION backend=%s rows=%d full_values=%zu residual_columns=%d first_max_abs=%.9g first_relative_l2=%.9g second_max_abs=%.9g second_relative_l2=%.9g resident_q8=1\n",
        ggml_backend_name(backend), tokens, actual.first.size() + actual.second.size(), tokens,
        first_error.first, first_error.second, second_error.first, second_error.second);
    ggml_backend_buffer_free(buffer); ggml_free(ctx);
    return actual;
}

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    // These explicit experiments change K reduction order for <=8 columns.
    // Reject rather than silently rewrite the environment for identity gates.
    for (const char* name : {"TS_GGML_Q8_PARALLEL_VECTOR", "TS_GGML_Q8_PARALLEL_SMALL_BATCH"}) {
        const char* value = std::getenv(name);
        require(!value || std::strcmp(value, "1") != 0, "Clear experimental Q8 parallel-reduction flags for column-identity qualification");
    }
    const char* tile = std::getenv("TS_GGML_Q8_PREFILL_TILE");
    std::printf("SENSITIVE_Q8_ENV prefill_tile=%s parallel_vector=0 parallel_small_batch=0\n", tile ? tile : "auto");
#ifdef TSG_GGML_USE_CUDA
    if (ggml_backend_cuda_get_device_count() == 0) return 77;
    auto* original = ggml_backend_cuda_init(0);
    require(original != nullptr, "Cannot initialize original CUDA backend");
    auto* backend = tsg_dsv4_fused_backend_init(original);
#else
    auto* backend = ggml_backend_cpu_init();
    if (backend) ggml_backend_cpu_set_n_threads(backend, 4);
#endif
    require(backend != nullptr, "Cannot initialize Q8 sensitive projection backend");
    std::vector<chain_result> results;
    for (int tokens : {1, 3, 8, 512, 809}) results.push_back(check_chain(backend, tokens));
    size_t overlap_values = 0;
    const auto exact_overlap = [&overlap_values](const chain_result& narrow, const chain_result& wider) {
        const auto overlap = [&overlap_values](const std::vector<float>& narrow, const std::vector<float>& wide_values) {
            require(std::memcmp(narrow.data(), wide_values.data(), narrow.size() * sizeof(float)) == 0,
                "Shared Q8 projection columns differ within a qualified width class");
            overlap_values += narrow.size();
        };
        overlap(narrow.first, wider.first); overlap(narrow.second, wider.second);
    };
    // Preserve the existing precision policy: short decode/verify columns are
    // invariant within <=8, and the wide K-ordered implementation is invariant
    // across prefill widths. Do not impose a stronger inter-class bit contract.
    exact_overlap(results[0], results[2]); exact_overlap(results[1], results[2]);
    exact_overlap(results[3], results[4]);
    size_t cross_values = 0, cross_changed_bits = 0;
    double cross_maximum = 0, cross_squared = 0, cross_reference_squared = 0;
    for (size_t index = 0; index < 3; ++index) {
        const auto measure_overlap = [&](const std::vector<float>& short_values, const std::vector<float>& wide_values) {
            for (size_t i = 0; i < short_values.size(); ++i) {
                const double delta = double(short_values[i]) - wide_values[i];
                cross_maximum = std::max(cross_maximum, std::abs(delta));
                cross_squared += delta * delta;
                cross_reference_squared += double(wide_values[i]) * wide_values[i];
                cross_changed_bits += std::memcmp(&short_values[i], &wide_values[i], sizeof(float)) != 0;
                ++cross_values;
            }
        };
        measure_overlap(results[index].first, results[4].first);
        measure_overlap(results[index].second, results[4].second);
    }
    std::printf("SENSITIVE_Q8_CROSS_CLASS observations=%zu changed_bits=%zu max_abs=%.9g relative_l2=%.9g observation_only=1\n",
        cross_values, cross_changed_bits, cross_maximum, std::sqrt(cross_squared / cross_reference_squared));
    ggml_backend_free(backend);
#ifdef TSG_GGML_USE_CUDA
    ggml_backend_free(original);
#endif
    std::printf("PASS: Q8 sensitive two-stage complete outputs at1/3/8/512/809, %zu within-class bit-exact overlap values, FP64 stored-checkpoint oracle, F32 activation residual, unchanged weights/guards and backend release; no cross-class bit identity, full-model quality/performance or hard-cap claim\n", overlap_values);
    return 0;
}
