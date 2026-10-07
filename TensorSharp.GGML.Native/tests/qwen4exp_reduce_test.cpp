// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Exact prefill-combine parity, input lifetime/stride checks, and an optional
// allocation/timing benchmark. No model or device scenario is silently passed.
#include "ggml_ops_qwen4exp_reduce.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml_ops_precision_policy.h"
#include "ggml-alloc.h"
#include "ggml-cpu.h"
#include "precision_test_utils.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif
#include <algorithm>
#include <chrono>
#include <cmath>
#include <memory>
#include <random>
#include <string>

struct graph_case
{
    ggml_context * ctx;
    input_tensor experts, routes;
    ggml_tensor * output;
    ggml_cgraph * graph;
    ggml_gallocr_t allocator;
    bool owned;

    graph_case(ggml_backend_t backend, int width, int used, int tokens, bool padded,
               bool candidate, int data)
        : ctx(ggml_init({4 * 1024 * 1024, nullptr, true})),
          experts(ctx, GGML_TYPE_F32, {width, used, tokens, 1}, padded),
          routes(ctx, GGML_TYPE_F32, {1, used, tokens, 1}, padded)
    {
        require(ctx != nullptr, "Cannot initialize combine test graph");
        ggml_set_input(experts.storage);
        ggml_set_input(routes.storage);
        // This fixture replays the same uploaded inputs and checks their padding.
        // Gallocr may otherwise legally reuse an input after its last reader.
        ggml_set_output(experts.storage);
        ggml_set_output(routes.storage);
        output = candidate ? tsg_q4e_prefill_expert_reduce(ctx, experts.tensor, routes.tensor) : nullptr;
        owned = output != nullptr;
        require(owned == (candidate && tokens > TSG_PRECISION_DECODE_COLUMNS && tokens <= 65535),
            "Combine supported token range changed");
        if (!output)
        {
            auto * weighted = ggml_cont(ctx, ggml_mul(ctx, experts.tensor, routes.tensor));
            output = ggml_view_2d(ctx, weighted, width, tokens, weighted->nb[2], 0);
            for (int k = 1; k < used; ++k)
                output = ggml_add(ctx, output, ggml_view_2d(ctx, weighted, width, tokens,
                    weighted->nb[2], (size_t)k * weighted->nb[1]));
        }
        // Test an ordinary node following a CUSTOM node without introducing a
        // backend-specific arithmetic boundary (CUDA SCALE also adds +0).
        output = ggml_cont(ctx, output);
        ggml_set_output(output);
        graph = ggml_new_graph(ctx);
        ggml_build_forward_expand(graph, output);
        allocator = ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend));
        require(allocator && ggml_gallocr_alloc_graph(allocator, graph), "Cannot allocate combine graph");
        std::mt19937 random(825 + width + tokens + used);
        std::uniform_real_distribution<float> distribution(-4.0f, 4.0f);
        for (float & value : experts.values) value = distribution(random);
        for (float & value : routes.values) value = distribution(random);
        if (data == 1)
        {
            for (size_t i = 0; i < experts.values.size(); ++i) experts.values[i] = i & 1 ? -0.0f : 0.0f;
            for (size_t i = 0; i < routes.values.size(); ++i) routes.values[i] = i & 1 ? -1.0f : 1.0f;
        }
        if (data == 2)
        {
            // Cancellation, halfway products and differently scaled terms:
            // changing to an FMA or reassociating the expert sum loses bits.
            const float xs[] = {1.0f, 1.0f + 0x1p-23f, -1.0f, 0x1p-23f, 8192.0f + 0x1p-10f,
                -8192.0f, -1.0f - 0x1p-23f, 0x1p-100f};
            const float ws[] = {-1.0f, 1.0f - 0x1p-23f, 1.0f, 0.5f, 1.0f + 0x1p-23f};
            for (int t = 0; t < tokens; ++t)
            for (int k = 0; k < used; ++k)
            {
                routes.values[routes.logical(0, k, t, 0)] = ws[(k + t) % 5];
                for (int x = 0; x < width; ++x)
                    experts.values[experts.logical(x, k, t, 0)] = xs[(k + x + t) % 8];
            }
        }
        experts.upload(); routes.upload();
    }

    ~graph_case() { ggml_gallocr_free(allocator); ggml_free(ctx); }
    size_t bytes() const { return ggml_gallocr_get_buffer_size(allocator, 0); }
    std::vector<float> read() const
    {
        std::vector<float> result(ggml_nelements(output));
        ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
        return result;
    }
};

static void compute(ggml_backend_t backend, graph_case & graph)
{
    require(ggml_backend_graph_compute(backend, graph.graph) == GGML_STATUS_SUCCESS, "Combine compute failed");
}

static void compare(ggml_backend_t baseline_backend, ggml_backend_t candidate_backend,
    graph_case & baseline, graph_case & candidate)
{
    compute(baseline_backend, baseline);
    compute(candidate_backend, candidate);
    ggml_backend_synchronize(baseline_backend);
    ggml_backend_synchronize(candidate_backend);
    const auto expected = baseline.read(), actual = candidate.read();
    require(expected.size() == actual.size(), "Combine result size mismatch");
    if (std::memcmp(expected.data(), actual.data(), actual.size() * sizeof(float)) != 0)
    {
        size_t different = 0;
        for (size_t i = 0; i < actual.size(); ++i)
            different += std::memcmp(&expected[i], &actual[i], sizeof(float)) != 0;
        std::fprintf(stderr, "Combine mismatch width=%lld used=%lld tokens=%lld words=%zu\n",
            (long long)candidate.experts.tensor->ne[0], (long long)candidate.experts.tensor->ne[1],
            (long long)candidate.experts.tensor->ne[2], different);
        std::exit(1);
    }
    // Independent serial oracle additionally fixes the exact operation order.
    const int width = (int)candidate.experts.tensor->ne[0];
    const int used = (int)candidate.experts.tensor->ne[1];
    for (size_t i = 0; i < actual.size(); ++i)
    {
        const int x = (int)(i % width), t = (int)(i / width);
        volatile float sum = candidate.experts.at(x, 0, t) * candidate.routes.at(0, 0, t);
        for (int k = 1; k < used; ++k)
        {
            volatile float product = candidate.experts.at(x, k, t) * candidate.routes.at(0, k, t);
            sum = sum + product;
        }
        const float oracle = sum;
        require(std::memcmp(&oracle, &actual[i], sizeof(float)) == 0, "Combine independent oracle mismatch");
    }
    baseline.experts.check_unchanged(); baseline.routes.check_unchanged();
    candidate.experts.check_unchanged(); candidate.routes.check_unchanged();
}

static double time(ggml_backend_t backend, graph_case & graph)
{
    constexpr int repeats = 10;
    const auto start = std::chrono::steady_clock::now();
    for (int i = 0; i < repeats; ++i) compute(backend, graph);
    ggml_backend_synchronize(backend);
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - start).count() / repeats;
}

static void check_token_launch_limit()
{
    // Metadata only: no backing activation allocation or large kernel launch.
    ggml_context * ctx = ggml_init({1024 * 1024, nullptr, true});
    require(ctx != nullptr, "Cannot initialize combine shape test");
    for (int tokens : {65535, 65536})
    {
        auto * experts = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 32, 2, tokens);
        auto * routes = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, 1, 2, tokens);
        auto * output = tsg_q4e_prefill_expert_reduce(ctx, experts, routes);
        require((output != nullptr) == (tokens <= 65535), "Combine CUDA grid.y limit changed");
    }
    ggml_free(ctx);
    std::puts("PASS combine token launch limit: 65535 accepted, 65536 falls back (metadata only)");
}

int main(int argc, char ** argv)
{
    bool cuda = false, benchmark = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "cuda") == 0) cuda = true;
        else if (std::strcmp(argv[i], "cpu") == 0) cuda = false;
        else if (std::strcmp(argv[i], "--benchmark") == 0) benchmark = true;
        else { std::fprintf(stderr, "Usage: %s [cpu|cuda] [--benchmark]\n", argv[0]); return 2; }
    }
    check_token_launch_limit();
    ggml_backend_t backend = nullptr, wrapper = nullptr;
#ifdef TSG_GGML_USE_CUDA
    if (cuda)
    {
        if (ggml_backend_cuda_get_device_count() == 0) { std::puts("SKIP: CUDA device unavailable"); return 77; }
        backend = ggml_backend_cuda_init(0);
        wrapper = tsg_dsv4_fused_backend_init(backend);
    }
#else
    if (cuda) { std::puts("SKIP: CUDA not compiled"); return 77; }
#endif
    if (!cuda) { backend = ggml_backend_cpu_init(); ggml_backend_cpu_set_n_threads(backend, 4); }
    require(backend && (!cuda || wrapper), "Cannot initialize combine backend");
    ggml_backend_t candidate_backend = wrapper ? wrapper : backend;
    int cases = 0;
    for (int tokens : {1, 8, 9, 32, 128, 512})
    for (int used : {2, 10})
    for (int layout = 0; layout < 3; ++layout)
    for (int data = 0; data < 3; ++data)
    {
        const int width = layout == 0 ? 64 : layout == 1 ? 65 : 32;
        const bool padded = layout == 2;
        graph_case baseline(backend, width, used, tokens, padded, false, data);
        graph_case candidate(backend, width, used, tokens, padded, true, data);
        compare(backend, candidate_backend, baseline, candidate);
        ++cases;
    }
    std::printf("PASS %s exact combine: %d cases (decode/verify fallback, vector/scalar, strides, zero and adversarial)\n",
        cuda ? "CUDA" : "CPU", cases);
    if (benchmark)
    {
        std::puts("width,used,tokens,owned,baseline_bytes,candidate_bytes,baseline_us,candidate_us,speedup");
        for (int width : {2560, 1280})
        for (int tokens : {1, 8, 9, 32, 128, 512})
        {
            graph_case baseline(backend, width, 10, tokens, false, false, 0);
            graph_case candidate(backend, width, 10, tokens, false, true, 0);
            compare(backend, candidate_backend, baseline, candidate);
            for (int i = 0; i < 3; ++i) { compute(backend, baseline); compute(candidate_backend, candidate); }
            ggml_backend_synchronize(candidate_backend);
            std::vector<double> old_times, new_times;
            for (int i = 0; i < 5; ++i)
            {
                if (i % 2) { new_times.push_back(time(candidate_backend, candidate)); old_times.push_back(time(backend, baseline)); }
                else { old_times.push_back(time(backend, baseline)); new_times.push_back(time(candidate_backend, candidate)); }
            }
            std::sort(old_times.begin(), old_times.end()); std::sort(new_times.begin(), new_times.end());
            std::printf("%d,10,%d,%d,%zu,%zu,%.3f,%.3f,%.3f\n", width, tokens, candidate.owned ? 1 : 0,
                baseline.bytes(), candidate.bytes(), old_times[2], new_times[2], old_times[2] / new_times[2]);
        }
    }
    if (wrapper) ggml_backend_free(wrapper);
    ggml_backend_free(backend);
    return 0;
}
