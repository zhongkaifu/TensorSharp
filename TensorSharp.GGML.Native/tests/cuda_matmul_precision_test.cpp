// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_matmul_precision.h"
#include "ggml-alloc.h"
#include "ggml-backend.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml_ops_dsv4_fused.h"
#include "ggml-cuda.h"
#else
#include "ggml-cpu.h"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <vector>

#include "precision_test_utils.h"

enum class pattern { random, activation_residual, large_activation, weight_residual };

struct test_case
{
    int tokens = 5, inner = 256, rows = 64;
    bool indexed = false, precise = true, padded = false, interleaved = false, convert = false, pipeline = false;
    int experts = 4, used = 4, slots = 1;
    int a2 = 1, a3 = 1, b2 = 1, b3 = 1;
    ggml_type weights = GGML_TYPE_F32;
    pattern data = pattern::random;
};

static void check(ggml_backend_t allocator, ggml_backend_t backend, const test_case & c)
{
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot create precision test context");
    input_tensor a(ctx, c.weights, {c.inner, c.rows, c.indexed ? c.experts : c.a2, c.indexed ? 1 : c.a3},
                   c.padded, c.interleaved);
    input_tensor b(ctx, GGML_TYPE_F32,
                   {c.inner, c.indexed ? c.slots : c.tokens, c.indexed ? c.tokens : c.b2, c.indexed ? 1 : c.b3},
                   c.padded, c.interleaved);
    input_tensor ids(ctx, GGML_TYPE_I32, {c.indexed ? c.used : 1, c.indexed ? c.tokens : 1, 1, 1}, c.padded,
                     c.indexed && c.interleaved);
    auto * activation = c.pipeline ? ggml_scale(ctx, b.tensor, 2.0f) : b.tensor;
    ggml_tensor * output;
    if (!c.precise || c.convert)
    {
        output = c.indexed ? ggml_mul_mat_id(ctx, a.tensor, activation, ids.tensor) : ggml_mul_mat(ctx, a.tensor, activation);
        if (c.precise) tsg_matmul_require_f32(ctx, output);
    }
    else output = c.indexed ? tsg_matmul_id_f32(ctx, a.tensor, activation, ids.tensor) : tsg_matmul_f32(ctx, a.tensor, activation);
    require(output != nullptr, "Cannot construct precision matmul");
    if (c.precise) require(output->op == GGML_OP_CUSTOM, "Explicit F32 must use the TensorSharp implementation");
    auto * graph_output = ggml_scale(ctx, output, 0.5f);
    auto * graph = ggml_new_graph(ctx);
    ggml_build_forward_expand(graph, graph_output);
    auto * buffer = ggml_backend_alloc_ctx_tensors(ctx, allocator);
    require(buffer != nullptr, "Precision test allocation failed");
    std::mt19937 random(41091);
    std::uniform_real_distribution<float> uniform(-1.0f, 1.0f);
    for (size_t i = 0; i < a.values.size(); ++i)
    {
        const int k = i % c.inner, row = (i / c.inner) % c.rows;
        a.values[i] = c.data == pattern::random ? uniform(random)
            : (k % 2 == 0 ? 1.0f : -1.0f) * (1 + row % 4);
        if (c.data == pattern::weight_residual && k % 2 == 0) a.values[i] += 0x1p-16f;
    }
    for (size_t i = 0; i < b.values.size(); ++i)
    {
        const int k = i % c.inner;
        const float residue = (1 + (i / c.inner) % 5) * 0x1p-16f;
        // One large pair isolates F16 overflow/TF32 rounding from the legal
        // F32 error of summing hundreds of large positive partial products.
        b.values[i] = c.data == pattern::random ? uniform(random)
            : c.data == pattern::large_activation ? (k < 2 ? 70000.0f + (k == 0 ? 0.125f : 0.0f) : 0.0f)
            : c.data == pattern::activation_residual ? 1.0f + (k % 2 == 0 ? residue : 0.0f) : 1.0f;
    }
    for (int t = 0; t < ids.tensor->ne[1]; ++t)
        for (int e = 0; e < ids.tensor->ne[0]; ++e)
            // Upstream's sorted expert path requires unique IDs, as top-k
            // produces. The owned-only padded cases also stress duplicate IDs.
            ids.values[ids.logical(e, t, 0, 0)] = (3 * t + (c.padded ? e / 2 : e)) % c.experts;
    a.upload(); b.upload(); ids.upload();

    auto validate = [&]()
    {
        std::vector<float> result(ggml_nelements(graph_output));
        ggml_backend_tensor_get(graph_output, result.data(), 0, result.size() * sizeof(float));
        double squared_error = 0, squared_reference = 0, maximum = 0;
        size_t failures = 0;
        for (int64_t w = 0; w < output->ne[3]; ++w)
        for (int64_t z = 0; z < output->ne[2]; ++z)
        for (int64_t y = 0; y < output->ne[1]; ++y)
        for (int64_t row = 0; row < output->ne[0]; ++row)
        {
            const int64_t az = c.indexed ? static_cast<int64_t>(ids.at(y, z)) : z / (c.b2 / c.a2);
            const int64_t aw = c.indexed ? 0 : w / (c.b3 / c.a3);
            const int64_t by = c.indexed ? y % c.slots : y;
            double reference = 0;
            for (int64_t k = 0; k < c.inner; ++k) reference += double(a.at(k, row, az, aw)) * b.at(k, by, z, w);
            reference *= c.pipeline ? 1.0 : 0.5;
            const size_t index = row + output->ne[0] * (y + output->ne[1] * (z + output->ne[2] * w));
            require(std::isfinite(result[index]) && std::isfinite(reference), "Non-finite precision matmul result");
            const double error = std::abs(result[index] - reference);
            maximum = std::max(maximum, error);
            squared_error += error * error;
            squared_reference += reference * reference;
            const double tolerance = 3e-5 * std::max(1.0, std::sqrt(c.inner / 256.0)) + 3e-6 * std::abs(reference);
            if (c.precise && error > tolerance)
            {
                if (failures++ == 0) std::fprintf(stderr, "Mismatch index=%zu expected=%.12g actual=%.12g error=%.8g tolerance=%.8g\n",
                    index, reference, result[index], error, tolerance);
            }
        }
        const double relative = std::sqrt(squared_error / std::max(1e-30, squared_reference));
        std::printf(" max_abs=%.8g rel_l2=%.8g", maximum, relative);
        require(failures == 0, "Explicit F32 matmul lost precision or used the wrong input");
        if (!c.precise)
            require(relative < (c.weights == GGML_TYPE_BF16 ? 0.02 : 0.005),
                "Default matmul exceeds expected source-rounding error");
    };

    std::printf("%s %s nt=%d weights=%s inner=%d rows=%d slots=%d batches=%d,%d/%d,%d padded=%d stride0=%d pattern=%d convert=%d",
        ggml_backend_name(backend), c.indexed ? "MUL_MAT_ID" : "MUL_MAT", c.tokens, ggml_type_name(c.weights),
        c.inner, c.rows, c.slots, c.a2, c.a3, c.b2, c.b3, c.padded, c.interleaved, int(c.data), c.convert);
    std::printf(" precision=%s pipeline=%d", c.precise ? "F32" : "default", c.pipeline);
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Precision matmul failed");
    validate();
    const auto start = std::chrono::steady_clock::now();
    for (int repeat = 0; repeat < 5; ++repeat)
        require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Repeated precision matmul failed");
    ggml_backend_synchronize(backend);
    const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / 5;
    std::printf(" mean_us=%.1f", us);
    // Reuse the graph with changed activations and expert IDs, as decode does.
    // Exact cancellation fixtures keep exact products; random perturbation here
    // would test reduction-order error instead of lost source precision.
    for (float & value : b.values)
        value = c.data == pattern::large_activation || c.data == pattern::weight_residual
            ? 2.0f * value : value + 0.125f * uniform(random);
    for (float & value : ids.values) value = float((int(value) + 1) % c.experts);
    b.upload(); ids.upload();
    require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "Precision graph reuse failed");
    validate();
    std::printf("\n");
    a.check_unchanged(); b.check_unchanged(); ids.check_unchanged();
    ggml_backend_buffer_free(buffer);
    ggml_free(ctx);
}

static void run(ggml_backend_t allocator, ggml_backend_t backend)
{
    for (bool indexed : {false, true}) for (int tokens : {1, 4, 5, 16, 31})
    for (ggml_type weights : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16})
    for (bool precise : {false, true})
    {
        test_case c;
        c.indexed = indexed; c.tokens = tokens; c.weights = weights; c.precise = precise;
        check(allocator, backend, c);
    }
    for (ggml_type weights : {GGML_TYPE_F32, GGML_TYPE_F16, GGML_TYPE_BF16})
    {
        test_case batch;
        batch.weights = weights; batch.b2 = 3; batch.b3 = 2;
        check(allocator, backend, batch);
        batch.a2 = batch.b2; batch.a3 = batch.b3;
        check(allocator, backend, batch);
        for (bool indexed : {false, true}) for (pattern data : {pattern::activation_residual, pattern::large_activation})
        {
            test_case c;
            c.indexed = indexed; c.tokens = 16; c.weights = weights; c.data = data;
            check(allocator, backend, c);
        }
        for (bool interleaved : {false, true})
        {
            test_case c;
            c.weights = weights; c.padded = true; c.interleaved = interleaved; c.convert = true;
            c.inner = 257; c.rows = 37; c.a2 = 2; c.a3 = 2; c.b2 = 4; c.b3 = 6;
            check(allocator, backend, c);
            c.indexed = true; c.slots = 2;
            check(allocator, backend, c);
            c.slots = c.used;
            check(allocator, backend, c);
        }
    }
    for (int tokens : {16, 31}) for (bool precise : {false, true})
    {
        test_case c;
        c.tokens = tokens; c.inner = 128; c.rows = 1280; c.precise = precise;
        check(allocator, backend, c);
    }
    for (bool indexed : {false, true})
    {
        test_case c;
        c.indexed = indexed; c.inner = 4096; c.rows = 17; c.data = pattern::weight_residual;
        check(allocator, backend, c);
        c.inner = 256; c.rows = 64; c.data = pattern::random; c.pipeline = true;
        check(allocator, backend, c);
    }
}

int main()
{
    std::setvbuf(stdout, nullptr, _IONBF, 0);
#ifdef TSG_GGML_USE_CUDA
    const int devices = ggml_backend_cuda_get_device_count();
    if (devices == 0) return 77;
    for (int device = 0; device < devices; ++device)
    {
        auto * cuda = ggml_backend_cuda_init(device);
        require(cuda != nullptr, "Cannot initialize a visible CUDA device");
        auto * backend = tsg_dsv4_fused_backend_init(cuda);
        require(backend != nullptr, "Cannot initialize TensorSharp CUDA precision backend");
        run(cuda, backend);
        ggml_backend_free(backend);
        ggml_backend_free(cuda);
    }
#else
    auto * backend = ggml_backend_cpu_init();
    require(backend != nullptr, "Cannot initialize CPU precision backend");
    ggml_backend_cpu_set_n_threads(backend, 4);
    run(backend, backend);
    ggml_backend_free(backend);
#endif
    return 0;
}
