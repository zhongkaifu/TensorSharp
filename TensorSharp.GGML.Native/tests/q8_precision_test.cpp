// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_q8_precision.h"
#include "ggml-alloc.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml_ops_dsv4_fused.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>
#else
#include "ggml-cpu.h"
#endif
#include "precision_test_utils.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <string>

namespace {
bool parallel_vector = false;
enum class pattern { random, residual, large, subnormal, non_power_scale, decoded_weight_residual, long_cancellation };

double parallel_rounding_bound(int inner, double absolute_products) {
    const double depth = inner / 32 + 5;
    const double nu = depth * 0x1p-24, oracle_nu = inner * 0x1p-53;
    // Relative error follows the longest reduction path. Absolute underflow
    // errors from ALL lane FMAs/additions can accumulate at the root, so they
    // must count K FMAs + 31 tree additions, not just the longest path length.
    return (nu / (1 - nu) + oracle_nu / (1 - oracle_nu)) * absolute_products
        + (inner + 31.0) * std::ldexp(1.0, -150) / (1 - nu);
}

bool random_activations(pattern data) {
    return data == pattern::random || data == pattern::non_power_scale;
}

struct test_case {
    int inner = 96, rows = 65, columns = 5;
    bool padded = false, interleaved = false, transposed = false, pipeline = false;
    pattern data = pattern::random;
    int target_column = -1;
};

// Write the published Q8_0 byte layout directly. The oracle consumes separate
// logical values, never the device bytes/strides or the production callback.
struct quantized_weights {
    ggml_tensor * storage;
    ggml_tensor * tensor;
    size_t offset, stride;
    std::vector<unsigned char> bytes;
    std::vector<double> values;

    quantized_weights(ggml_context * ctx, const test_case & c) {
        require(ggml_type_size(GGML_TYPE_Q8_0) == 34 && ggml_blck_size(GGML_TYPE_Q8_0) == 32,
                "Q8_0 format contract changed");
        stride = size_t(c.inner / 32 + (c.padded ? 3 : 0)) * 34;
        offset = c.padded ? 34 : 0;
        bytes.resize(offset + stride * c.rows + 34, 0xff);
        storage = ggml_new_tensor_1d(ctx, GGML_TYPE_Q8_0, int64_t(bytes.size() / 34) * 32);
        tensor = ggml_view_2d(ctx, storage, c.inner, c.rows, stride, offset);
        values.resize(size_t(c.inner) * c.rows);
    }

    void upload(const test_case & c, int revision) {
        for (int row = 0; row < c.rows; ++row) {
            for (int block = 0; block < c.inner / 32; ++block) {
                float scale = 1.0f;
                if (random_activations(c.data)) {
                    // Every significand is representable in F16, while a
                    // signed-byte product generally needs more F16 mantissa
                    // bits. This detects narrowing decoded Q8 weights.
                    const float significand = c.data == pattern::random ? 1.0f
                        : 1.0f + float(1 + (row * 71 + block * 19 + revision * 13) % 1023) / 1024.0f;
                    scale = std::ldexp(significand, -7 - (row + block) % 4);
                } else if (c.data == pattern::subnormal) {
                    scale = std::ldexp(1.0f, -24);
                } else if (c.data == pattern::decoded_weight_residual) {
                    scale = 1.0f + 0x1p-10f;
                }
                const ggml_fp16_t bits = ggml_fp32_to_fp16(scale);
                require(ggml_fp16_to_fp32(bits) == scale, "Test scale must be exactly representable");
                unsigned char * dst = bytes.data() + offset + size_t(row) * stride + size_t(block) * 34;
                std::memcpy(dst, &bits, sizeof(bits));
                for (int j = 0; j < 32; ++j) {
                    const int k = block * 32 + j;
                    const int q = random_activations(c.data) ? ((k * 37 + row * 19 + revision * 11) % 256) - 128
                        : c.data == pattern::long_cancellation ? 1
                        : c.data == pattern::decoded_weight_residual ? (k == 0 ? 127 : 0)
                        : c.data == pattern::subnormal ? (k == 0 ? 1 : 0)
                        : (k == 0 ? 1 : k == 1 ? -1 : 0);
                    const int8_t encoded = int8_t(q);
                    std::memcpy(dst + 2 + j, &encoded, sizeof(encoded));
                    values[size_t(row) * c.inner + k] = double(scale) * q;
                }
            }
        }
        ggml_backend_tensor_set(storage, bytes.data(), 0, bytes.size());
    }

    void check_unchanged() const {
        std::vector<unsigned char> actual(bytes.size());
        ggml_backend_tensor_get(storage, actual.data(), 0, actual.size());
        require(actual == bytes, "Q8 projection modified weights or guard bytes");
    }
};

void fill_input(input_tensor & x, const test_case & c, int revision) {
    for (int col = 0; col < c.columns; ++col) {
        const int key = col == c.target_column ? 7 : col + 101;
        for (int k = 0; k < c.inner; ++k) {
            float value;
            if (random_activations(c.data)) {
                const uint32_t hash = uint32_t(k + 1) * 2654435761u ^ uint32_t(key + 1) * 2246822519u;
                value = float(int(hash % 65537) - 32768) / 32768.0f;
                // Preserve bits below F16 and TF32 resolution in ordinary data.
                value += float((hash >> 17) % 13) * 0x1p-22f;
            } else if (c.data == pattern::residual) {
                value = k == 0 ? 1.0f + 0x1p-16f : k == 1 ? 1.0f : 0.0f;
            } else if (c.data == pattern::large) {
                value = k == 0 ? 70000.125f : k == 1 ? 70000.0f : 0.0f;
            } else if (c.data == pattern::decoded_weight_residual) {
                value = k == 0 ? 1.0f : 0.0f;
            } else if (c.data == pattern::long_cancellation) {
                value = k % 4 == 0 ? 100000000.0f : k % 4 == 2 ? -100000000.0f : 1.0f;
            } else {
                value = k == 0 ? 0x1p-110f : 0.0f;
            }
            x.values[x.logical(k, col, 0, 0)] = revision == 0 ? value : value * 0.5f;
        }
    }
    x.upload();
}

std::vector<float> check(ggml_backend_t allocator, ggml_backend_t backend, const test_case & c) {
    auto * leaves = ggml_init({4 * 1024 * 1024, nullptr, true});
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(leaves && ctx, "Cannot create Q8 projection contexts");
    quantized_weights weights(leaves, c);
    const std::array<int64_t, 4> input_shape = c.transposed
        ? std::array<int64_t, 4>{c.columns, c.inner, 1, 1} : std::array<int64_t, 4>{c.inner, c.columns, 1, 1};
    input_tensor input(leaves, GGML_TYPE_F32, input_shape, c.padded, c.interleaved);
    if (c.transposed) input.tensor = ggml_transpose(leaves, input.tensor);
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_GGML_TEST_HOOKS)
    auto * baseline = c.columns == 1 ? ggml_new_tensor_1d(leaves, GGML_TYPE_F32, c.rows + 2) : nullptr;
#endif
    auto * leaf_buffer = ggml_backend_alloc_ctx_tensors(leaves, allocator);
    require(leaf_buffer != nullptr, "Cannot allocate Q8 projection inputs");
    auto * activation = c.pipeline ? ggml_scale(ctx, input.tensor, 2.0f) : input.tensor;
    auto * projected = tsg_matmul_q8_f32(ctx, weights.tensor, activation);
    require(projected && projected->op == GGML_OP_CUSTOM, "Q8 F32 projection must use the owned operation");
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_GGML_TEST_HOOKS)
    if (baseline) {
        // The old/new comparison runs after this graph. Do not let a later
        // in-place scale overwrite either the actual input or projected result.
        ggml_set_output(activation);
        ggml_set_output(projected);
    }
#endif
    auto * output = c.pipeline ? ggml_scale(ctx, projected, 0.5f) : projected;
    ggml_set_output(output);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, output);
    auto * graph_allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(allocator));
    require(graph_allocator && ggml_gallocr_alloc_graph(graph_allocator, graph), "Cannot allocate Q8 projection graph");
    std::vector<float> first;
    for (int revision = 0; revision < 2; ++revision) {
        weights.upload(c, revision);
        fill_input(input, c, revision);
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Q8 projection compute failed");
        std::vector<float> result(size_t(c.rows) * c.columns);
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
#if defined(TSG_GGML_USE_CUDA) && defined(TSG_GGML_TEST_HOOKS)
        if (baseline) {
            const float canary = -12345.625f;
            std::vector<float> reference(size_t(c.rows) + 2, canary), actual(size_t(c.rows));
            ggml_backend_tensor_set(baseline, reference.data(), 0, reference.size() * sizeof(float));
            require(tsg_matmul_q8_cuda_launch_reference(weights.tensor->data, activation->data,
                static_cast<float *>(baseline->data) + 1, c.inner, c.rows, 1,
                weights.tensor->nb[1], activation->nb[0], activation->nb[1], nullptr) == 0,
                "Previous Q8 decode launch failed");
            require(cudaDeviceSynchronize() == cudaSuccess, "Previous Q8 decode synchronization failed");
            ggml_backend_tensor_get(baseline, reference.data(), 0, reference.size() * sizeof(float));
            ggml_backend_tensor_get(projected, actual.data(), 0, actual.size() * sizeof(float));
            require(reference.front() == canary && reference.back() == canary, "Q8 reference output canary changed");
            if (!parallel_vector)
                require(std::memcmp(reference.data() + 1, actual.data(), actual.size() * sizeof(float)) == 0,
                    "Single-column Q8 kernel differs bitwise from previous K-ordered kernel");
            // Exercise the new raw launcher with output guards as well: graph
            // buffers alone would not catch a one-row tail overwrite.
            std::fill(reference.begin(), reference.end(), canary);
            ggml_backend_tensor_set(baseline, reference.data(), 0, reference.size() * sizeof(float));
            require(tsg_matmul_q8_cuda_launch(weights.tensor->data, activation->data,
                static_cast<float *>(baseline->data) + 1, c.inner, c.rows, 1,
                weights.tensor->nb[1], activation->nb[0], activation->nb[1], nullptr) == 0,
                "Single-column Q8 guarded launch failed");
            require(cudaDeviceSynchronize() == cudaSuccess, "Single-column Q8 guarded synchronization failed");
            ggml_backend_tensor_get(baseline, reference.data(), 0, reference.size() * sizeof(float));
            require(reference.front() == canary && reference.back() == canary, "Single-column Q8 output canary changed");
            require(std::memcmp(reference.data() + 1, actual.data(), actual.size() * sizeof(float)) == 0,
                "Guarded single-column Q8 launch changed output bytes");
        }
#endif
        double error_squared = 0, reference_squared = 0, maximum_error = 0;
        size_t failures = 0;
        for (int col = 0; col < c.columns; ++col) {
            for (int row = 0; row < c.rows; ++row) {
                double expected = 0, absolute_products = 0;
                for (int k = 0; k < c.inner; ++k) {
                    const double product = weights.values[size_t(row) * c.inner + k] * input.at(k, col);
                    expected += product;
                    absolute_products += std::abs(product);
                }
                if (c.data == pattern::decoded_weight_residual)
                    require(expected == (revision == 0 ? 127.1240234375 : 63.56201171875),
                            "Decoded-weight fixture lost its independently specified exact product");
                const float actual = result[size_t(col) * c.rows + row];
                const double error = std::abs(double(actual) - expected);
                const double tolerance = random_activations(c.data) ? 0.0001 + 0.000006 * std::abs(expected) : 0;
                if (!std::isfinite(actual) || error > tolerance) ++failures;
                if (parallel_vector && c.columns == 1) {
                    // Independent scalar FP64 reference, plus a tighter
                    // shape-derived error bound for the new reduction tree.
                    // Q8 half-scale * signed-byte is exactly representable in
                    // F32. Each lane performs K/32 FMAs, then five additions.
                    const double bound = parallel_rounding_bound(c.inner, absolute_products);
                    require(error <= bound, "Parallel Q8 independent FP64 tree-rounding bound failed");
                }
                maximum_error = std::max(maximum_error, error);
                error_squared += error * error;
                reference_squared += expected * expected;
            }
        }
        const double relative = std::sqrt(error_squared / std::max(reference_squared, std::numeric_limits<double>::min()));
        std::printf("Q8/F32 K=%d M=%d N=%d padded=%d interleaved=%d transposed=%d pipeline=%d pattern=%d revision=%d max=%.9g relL2=%.9g failures=%zu\n",
                    c.inner, c.rows, c.columns, int(c.padded), int(c.interleaved), int(c.transposed), int(c.pipeline), int(c.data), revision,
                    maximum_error, relative, failures);
        require(failures == 0 && relative <= 0.000004, "Q8 F32 independent double oracle mismatch");
        weights.check_unchanged();
        input.check_unchanged();
        if (revision == 0) first = std::move(result);
    }
    ggml_backend_synchronize(backend);
    ggml_gallocr_free(graph_allocator);
    ggml_backend_buffer_free(leaf_buffer);
    ggml_free(ctx);
    ggml_free(leaves);
    return first;
}

void run(ggml_backend_t allocator, ggml_backend_t backend) {
    int cases = 0;
    auto execute = [&](const test_case & c) { ++cases; return check(allocator, backend, c); };
    // Single-column warp boundaries/tails and noncontiguous source strides.
    // Every output is checked against an independent FP64 dot product; CUDA
    // additionally compares every N=1 projection with the previous kernel.
    for (int rows : {1, 15, 16, 17, 31, 32, 33, 64, 127, 129}) {
        test_case c;
        c.inner = 1024; c.rows = rows; c.columns = 1;
        c.padded = c.interleaved = true;
        c.data = pattern::non_power_scale;
        execute(c);
    }
    if (parallel_vector) {
        for (const auto shape : {std::array<int, 2>{32, 3}, {1024, 4}, {3584, 5},
                                 {2048, 1024}, {3584, 1024}, {1024, 16}, {1024, 6144}}) {
            test_case c;
            c.inner = shape[0]; c.rows = shape[1]; c.columns = 1;
            c.padded = c.interleaved = true; c.data = pattern::non_power_scale;
            execute(c);
        }
        test_case cancellation;
        cancellation.inner = 4096; cancellation.rows = 5; cancellation.columns = 1;
        cancellation.data = pattern::long_cancellation;
        cancellation.padded = cancellation.interleaved = true;
        execute(cancellation); // Serial K-order returns 1, independent FP64 is 2048.
    }
    // CTA row, column and activation-stride boundaries, including graph reuse.
    for (int columns : {1, 4, 8, 9, 16, 17, 31, 32, 33, 65}) {
        test_case c;
        c.columns = columns;
        execute(c);
        c.rows = 63;
        c.padded = c.interleaved = true;
        execute(c);
    }
    for (int columns : {1, 9, 33}) {
        test_case c;
        c.columns = columns;
        c.padded = c.interleaved = c.transposed = true;
        execute(c);
    }
    // Actual fused QKV, output, FFN up/down shapes from MiniLM and Snowflake.
    for (const auto shape : {std::array<int, 2>{384, 1152}, {384, 384}, {384, 1536}, {1536, 384},
                             {1024, 3072}, {1024, 1024}, {1024, 4096}, {4096, 1024}}) {
        for (int columns : {1, 17}) {
            test_case c;
            c.inner = shape[0]; c.rows = shape[1]; c.columns = columns;
            c.pipeline = true;
            execute(c);
            c.data = pattern::non_power_scale;
            execute(c);
        }
    }
    // Large packed-token dimensions with bounded oracle cost.
    for (int columns : {512, 8192, 65536}) {
        test_case c;
        c.inner = 32; c.rows = 3; c.columns = columns;
        execute(c);
    }
    // Exact fixtures reject activation quantization/narrowing and CUDA FTZ.
    for (pattern data : {pattern::residual, pattern::large, pattern::subnormal, pattern::decoded_weight_residual}) {
        for (int columns : {1, 9, 33}) {
            test_case c;
            c.inner = 64; c.rows = 65; c.columns = columns;
            c.padded = c.interleaved = true; c.data = data;
            execute(c);
        }
    }
    for (int columns : {1, 17, 33}) {
        test_case c;
        c.columns = columns;
        c.padded = c.interleaved = true;
        c.transposed = columns == 17;
        c.data = pattern::non_power_scale;
        execute(c);
    }
    // The default keeps cross-column bit patterns. The diagnostic parallel-K
    // route intentionally changes N=1 summation: every column must still pass
    // the SAME independent FP64 gate above, and its cross-N deviation must fit
    // the sum of the two pre-existing error bounds. It is never called exact.
    test_case c;
    c.inner = 384; c.rows = 73; c.columns = 1; c.target_column = 0;
    const auto reference = execute(c);
    for (int columns : {2, 8, 9, 16, 17, 33, 65}) {
        if (columns == 2 && !parallel_vector) continue;
        c.columns = columns;
        for (int position : {0, columns - 1}) {
            c.target_column = position;
            c.padded = c.interleaved = true;
            const auto result = execute(c);
            if (!parallel_vector)
                require(std::memcmp(reference.data(), result.data() + size_t(position) * c.rows,
                                    size_t(c.rows) * sizeof(float)) == 0,
                        "Q8 projection changed a column when packed with different companions");
            else {
                double errors = 0, norm = 0, maximum = 0;
                for (int row = 0; row < c.rows; ++row) {
                    const double expected = reference[row], actual = result[size_t(position) * c.rows + row];
                    const double difference = std::abs(expected - actual);
                    require(difference <= 2 * (0.0001 + 0.000006 * std::max(std::abs(expected), std::abs(actual))),
                        "Q8 prefill/decode column difference exceeded independent-oracle bounds");
                    errors += difference * difference; norm += expected * expected;
                    maximum = std::max(maximum, difference);
                }
                const double relative = std::sqrt(errors / std::max(norm, std::numeric_limits<double>::min()));
                require(relative <= 0.000008, "Q8 prefill/decode column relative error exceeded oracle bounds");
                std::printf("Parallel vector vs serial prefill N=%d position=%d max=%.9g relL2=%.9g\n", columns, position, maximum, relative);
            }
        }
    }
    std::printf("Q8 F32 projection: %d cases passed, two executions each.\n", cases);
}

#if defined(TSG_GGML_USE_CUDA)
void check_parallel_underflow(ggml_backend_t allocator) {
    // Every product is exactly halfway between zero and min-subnormal F32.
    // Correct round-to-nearest-even lane FMAs produce zero, so an FP64 relative
    // gate is inappropriate. Check the exact IEEE result AND the global
    // absolute underflow bound independently of the ordinary-data gates.
    auto * context = ggml_init({1024 * 1024, nullptr, true});
    require(context != nullptr, "Cannot allocate underflow fixture metadata");
    auto * weight = ggml_new_tensor_2d(context, GGML_TYPE_Q8_0, 32, 5);
    auto * input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 32);
    auto * output = ggml_new_tensor_1d(context, GGML_TYPE_F32, 7);
    auto * buffer = ggml_backend_alloc_ctx_tensors(context, allocator);
    require(buffer != nullptr, "Cannot allocate underflow fixture payload");
    std::vector<unsigned char> weights(34 * 5, 1);
    for (int row = 0; row < 5; ++row) {
        weights[size_t(row) * 34] = 1; weights[size_t(row) * 34 + 1] = 0; // half 2^-24
    }
    std::vector<float> activations(32, std::ldexp(1.0f, -126));
    std::vector<float> result(7, -12345.625f);
    ggml_backend_tensor_set(weight, weights.data(), 0, weights.size());
    ggml_backend_tensor_set(input, activations.data(), 0, activations.size() * sizeof(float));
    ggml_backend_tensor_set(output, result.data(), 0, result.size() * sizeof(float));
    require(tsg_matmul_q8_cuda_launch(weight->data, input->data, static_cast<float *>(output->data) + 1,
        32, 5, 1, 34, sizeof(float), 32 * sizeof(float), nullptr) == 0, "Underflow fixture launch failed");
    require(cudaDeviceSynchronize() == cudaSuccess, "Underflow fixture synchronization failed");
    ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
    const double expected = 32 * std::ldexp(1.0, -150);
    require(result.front() == -12345.625f && result.back() == -12345.625f, "Underflow fixture output canary changed");
    for (int row = 0; row < 5; ++row) {
        require(result[size_t(row) + 1] == 0.0f, "Half-min-subnormal tie did not round to even zero");
        require(expected <= parallel_rounding_bound(32, expected), "Parallel absolute underflow bound lost lane contributions");
        require(expected > 6 * std::ldexp(1.0, -150), "Underflow fixture no longer rejects a depth-only bound");
    }
    ggml_backend_buffer_free(buffer); ggml_free(context);
    std::puts("Parallel Q8 subnormal tie fixture: 5 rows, exact IEEE rounding, global absolute bound and canaries passed.");
}
#endif
} // namespace

int main(int argc, char ** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    require(argc == 1 || (argc == 2 && std::string(argv[1]) == "--parallel-vector"), "Unexpected Q8 precision arguments");
    parallel_vector = argc == 2;
#ifdef _WIN32
    require(_putenv_s("TS_GGML_Q8_PARALLEL_VECTOR", parallel_vector ? "1" : "0") == 0, "Cannot configure Q8 test arithmetic");
#else
    require(setenv("TS_GGML_Q8_PARALLEL_VECTOR", parallel_vector ? "1" : "0", 1) == 0, "Cannot configure Q8 test arithmetic");
#endif
    std::printf("Q8 test arithmetic: %s\n", parallel_vector ? "experimental parallel-K vector; serial prefill" : "qualified K-ordered");
#ifdef TSG_GGML_USE_CUDA
    const int devices = ggml_backend_cuda_get_device_count();
    if (devices == 0) return 77;
    for (int device = 0; device < devices; ++device) {
        std::printf("Q8 projection CUDA device %d\n", device);
        auto * raw = ggml_backend_cuda_init(device);
        require(raw != nullptr, "Cannot initialize Q8 CUDA backend");
        auto * wrapped = tsg_dsv4_fused_backend_init(raw);
        require(wrapped != nullptr, "Cannot initialize owned Q8 CUDA wrapper");
        run(raw, wrapped);
        if (parallel_vector) check_parallel_underflow(raw);
        ggml_backend_free(wrapped);
        ggml_backend_free(raw);
    }
#else
    auto * backend = ggml_backend_cpu_init();
    require(backend != nullptr, "Cannot initialize Q8 CPU backend");
    ggml_backend_cpu_set_n_threads(backend, 4);
    run(backend, backend);
    ggml_backend_free(backend);
#endif
    return 0;
}
