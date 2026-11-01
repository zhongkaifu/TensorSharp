// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Current-graph planning must preserve dependencies after topology/shape changes.
// --legacy-plan exercises the former size-only reuse policy with the same checks.
#include "ggml_ops_graph_alloc.h"
#include "ggml-backend.h"
#include "ggml-cpu.h"
#include "ggml.h"
#ifdef TSG_GGML_USE_CUDA
#include "ggml-cuda.h"
#endif

#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace
{
    void require(bool value, const std::string& message)
    {
        if (!value) throw std::runtime_error(message);
    }

    struct Allocator
    {
        ggml_gallocr_t value;
        explicit Allocator(ggml_backend_t backend)
            : value(ggml_gallocr_new(ggml_backend_get_default_buffer_type(backend)))
        { require(value != nullptr, "allocator initialization failed"); }
        ~Allocator() { ggml_gallocr_free(value); }
    };

    struct Graph
    {
        ggml_context* context = ggml_init({1u << 20, nullptr, true});
        ggml_cgraph* graph = nullptr;
        ggml_tensor *x = nullptr, *c = nullptr, *a = nullptr, *b = nullptr, *output = nullptr;
        std::vector<float> x_data, c_data, expected;
        bool use_a;

        Graph(int count, bool final_uses_a) : x_data(count), c_data(count), expected(count), use_a(final_uses_a)
        {
            require(context != nullptr, "graph context allocation failed");
            x = ggml_new_tensor_1d(context, GGML_TYPE_F32, count);
            c = ggml_new_tensor_1d(context, GGML_TYPE_F32, count);
            ggml_set_input(x); ggml_set_input(c);
            ggml_set_name(x, "x"); ggml_set_name(c, "c");
            a = ggml_add(context, x, c);
            b = ggml_scale(context, a, 3.f);
            output = ggml_add(context, b, final_uses_a ? a : c);
            ggml_set_name(a, "a"); ggml_set_name(b, "b"); ggml_set_name(output, "output");
            ggml_set_output(output);
            graph = ggml_new_graph_custom(context, 32, false);
            ggml_build_forward_expand(graph, output);
            require(ggml_graph_n_nodes(graph) == 3, "fixture must retain exactly add/scale/add");
            for (int i = 0; i < count; ++i)
            {
                // Exact binary fractions keep this an alias/lifetime test rather
                // than a floating-point tolerance test. Element zero is x=c=1.
                x_data[i] = 1.f + static_cast<float>(i % 7) * .25f;
                c_data[i] = 1.f + static_cast<float>(i % 5) * .25f;
                const float av = x_data[i] + c_data[i];
                expected[i] = 3.f * av + (final_uses_a ? av : c_data[i]);
            }
        }
        ~Graph() { ggml_free(context); }

        void allocate(ggml_gallocr_t allocator, bool legacy)
        {
            const bool allocated = legacy ? ggml_gallocr_alloc_graph(allocator, graph)
                : tsg::alloc_graph_with_current_plan(allocator, graph);
            require(allocated, "graph allocation failed");
        }

        std::vector<float> compute(ggml_backend_t backend) const
        {
            ggml_backend_tensor_set(x, x_data.data(), 0, x_data.size() * sizeof(float));
            ggml_backend_tensor_set(c, c_data.data(), 0, c_data.size() * sizeof(float));
            require(ggml_backend_graph_compute(backend, graph) == GGML_STATUS_SUCCESS, "graph compute failed");
            std::vector<float> result(expected.size());
            ggml_backend_tensor_get(output, result.data(), 0, result.size() * sizeof(float));
            for (std::size_t i = 0; i < result.size(); ++i)
                require(result[i] == expected[i], std::string(use_a ? "B" : "A") +
                    " output mismatch at " + std::to_string(i) + ": expected " +
                    std::to_string(expected[i]) + ", actual " + std::to_string(result[i]));
            return result;
        }
    };

    void run(ggml_backend_t backend, bool legacy)
    {
        int graphs = 0, executions = 0;
        {
            Allocator allocator(backend);
            // Same three nodes, two input leaves, shapes and op types. Only the
            // final source changes: A permits a/b aliasing, B still needs a.
            for (bool use_a : {false, true, false, true})
            {
                Graph fixture(32, use_a);
                fixture.allocate(allocator.value, legacy);
                ++graphs;
                for (int repeat = 0; repeat < 3; ++repeat)
                {
                    fixture.compute(backend);
                    ++executions;
                }
            }
        }
        {
            Allocator allocator(backend);
            std::vector<float> first;
            ggml_backend_buffer_t grown_buffer = nullptr;
            void* grown_base = nullptr;
            std::size_t grown_bytes = 0;
            for (int count : {32, 512, 32, 32})
            {
                Graph fixture(count, false);
                fixture.allocate(allocator.value, legacy);
                ++graphs;
                if (count == 512)
                {
                    grown_buffer = fixture.x->buffer;
                    grown_base = ggml_backend_buffer_get_base(grown_buffer);
                    grown_bytes = ggml_gallocr_get_buffer_size(allocator.value, 0);
                }
                else if (grown_buffer != nullptr)
                {
                    // This tiny graph fits one backend buffer. Shrinking its
                    // plan must keep the existing allocation, not free/recreate it.
                    require(fixture.x->buffer == grown_buffer &&
                        ggml_backend_buffer_get_base(fixture.x->buffer) == grown_base &&
                        ggml_gallocr_get_buffer_size(allocator.value, 0) == grown_bytes,
                        "shrinking/repeating the graph replaced its fitting backing buffer");
                }
                for (int repeat = 0; repeat < 3; ++repeat)
                {
                    const std::vector<float> current = fixture.compute(backend);
                    if (count == 32)
                    {
                        if (first.empty()) first = current;
                        else require(std::memcmp(first.data(), current.data(), first.size() * sizeof(float)) == 0,
                            "same graph changed after larger-shape allocation or replay");
                    }
                    ++executions;
                }
            }
        }
        std::printf("PASS: 2 allocation cases, %d fresh graphs, %d executions; current dependencies, "
            "shape history, replay, fitting backing-buffer retention\n", graphs, executions);
    }
}

int main(int argc, char** argv)
{
    bool cuda = false, legacy = false;
    for (int i = 1; i < argc; ++i)
    {
        if (std::strcmp(argv[i], "cpu") == 0) cuda = false;
        else if (std::strcmp(argv[i], "cuda") == 0) cuda = true;
        else if (std::strcmp(argv[i], "--legacy-plan") == 0) legacy = true;
        else { std::fprintf(stderr, "usage: %s [cpu|cuda] [--legacy-plan]\n", argv[0]); return 1; }
    }
    ggml_backend_t backend = nullptr;
    if (cuda)
    {
#ifdef TSG_GGML_USE_CUDA
        if (ggml_backend_cuda_get_device_count() > 0) backend = ggml_backend_cuda_init(0);
#endif
        if (!backend) { std::fprintf(stderr, "SKIP: CUDA backend/device unavailable; no CUDA assertions ran\n"); return 77; }
    }
    else
    {
        backend = ggml_backend_cpu_init();
        if (!backend) { std::fprintf(stderr, "FAIL: CPU backend initialization failed\n"); return 1; }
        ggml_backend_cpu_set_n_threads(backend, 4);
    }
    std::printf("backend=%s allocation=%s\n", cuda ? "cuda" : "cpu", legacy ? "legacy-plan" : "current-plan");
    try
    {
        run(backend, legacy);
        ggml_backend_free(backend);
        return 0;
    }
    catch (const std::exception& ex)
    {
        std::fprintf(stderr, "FAIL: %s\n", ex.what());
        ggml_backend_free(backend);
        return 1;
    }
}
