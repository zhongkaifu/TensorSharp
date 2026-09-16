// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_scheduler_alloc.h"
#include "ggml-cpu.h"
#include "ggml-cuda.h"
#include <cuda_runtime_api.h>

#include <array>
#include <climits>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <stdexcept>

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}

struct allocation_log {
    int cuda_out_of_memory = 0;
    static void write(ggml_log_level, const char * text, void * data) {
        auto & state = *static_cast<allocation_log *>(data);
        std::fputs(text, stderr);
        if (std::strstr(text, "cudaMalloc failed:") && std::strstr(text, "out of memory"))
            ++state.cuda_out_of_memory;
    }
};

struct backend_pair {
    std::array<ggml_backend_t, 2> handles{};
    ~backend_pair() {
        for (auto backend : handles) if (backend) ggml_backend_free(backend);
    }
};

struct graph_owner {
    ggml_context * context = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    ggml_cgraph * graph = nullptr;
    ~graph_owner() {
        if (scheduler) ggml_backend_sched_free(scheduler);
        if (context) ggml_free(context);
    }
    void initialize(backend_pair & backends) {
        // Tensor metadata only: no_alloc=true is essential for the oversized
        // graph. Never allocate a host vector or host tensor of that size.
        context = ggml_init({1024 * 1024, nullptr, true});
        require(context != nullptr, "Could not allocate small tensor metadata context");
        std::array<ggml_backend_buffer_type_t, 2> types{
            ggml_backend_get_default_buffer_type(backends.handles[0]), ggml_backend_cpu_buffer_type()};
        scheduler = ggml_backend_sched_new(backends.handles.data(), types.data(), int(types.size()),
                                           GGML_DEFAULT_GRAPH_SIZE, false, true);
        require(scheduler != nullptr, "Could not create graph scheduler");
        graph = ggml_new_graph(context);
    }
};

void fail_device_allocation(backend_pair & backends, size_t physical_total, allocation_log & log) {
    require(sizeof(size_t) >= 8, "The oversized device-allocation fixture requires a 64-bit process");
    require(physical_total > 0 && physical_total < std::numeric_limits<size_t>::max() / 4,
            "Invalid physical CUDA memory size");
    constexpr int64_t columns = 1 << 20;
    constexpr size_t row_bytes = size_t(columns) * sizeof(float);
    // Greater than twice the entire physical device capacity, independently of
    // current free memory or any ggml virtual-device capacity subdivision.
    const int64_t rows = int64_t((2 * physical_total) / row_bytes) + 1;
    require(rows > 0 && rows <= INT_MAX && columns <= INT_MAX,
            "Oversized tensor dimensions exceed the intended int-safe metadata fixture");
    graph_owner failed;
    failed.initialize(backends);
    auto * scalar = ggml_new_tensor_1d(failed.context, GGML_TYPE_F32, 1);
    ggml_set_input(scalar);
    auto * output = ggml_repeat_4d(failed.context, scalar, columns, rows, 1, 1);
    ggml_set_name(output, "deliberate_device_allocation_failure_do_not_execute");
    ggml_set_output(output);
    require(ggml_nbytes(output) > physical_total * 2, "Oversized output does not exceed physical VRAM");
    require(scalar->data == nullptr && output->data == nullptr, "Metadata fixture allocated host tensor storage");
    require(ggml_backend_supports_op(backends.handles[0], output), "CUDA does not support the oversized fixture operation");
    ggml_build_forward_expand(failed.graph, output);
    ggml_backend_sched_set_tensor_backend(failed.scheduler, scalar, backends.handles[0]);
    ggml_backend_sched_set_tensor_backend(failed.scheduler, output, backends.handles[0]);
    std::printf("Deliberate CUDA allocation: shape=[%lld,%lld], output_bytes=%zu, physical_total_bytes=%zu\n",
                static_cast<long long>(columns), static_cast<long long>(rows), ggml_nbytes(output), physical_total);
    std::fflush(stdout);
    const int previous_errors = log.cuda_out_of_memory;
    const bool allocated = tsg_scheduler_alloc_graph(failed.scheduler, failed.graph);
    require(!allocated, "Oversized CUDA graph unexpectedly allocated; it must never be executed");
    require(log.cuda_out_of_memory == previous_errors + 1,
            "Expected exactly one real cudaMalloc out-of-memory failure, without an implicit retry");
    require(output->data == nullptr, "Failed graph exposed an output allocation");
    require(output->src[0] == scalar, "Failed reserve did not restore the original graph source");
    // Destruction is the supported failure contract. Do not reset/reuse this
    // scheduler, retry this graph, or reset/replace the surviving CUDA backend.
}

void fresh_cross_device_graph(backend_pair & backends) {
    graph_owner recovered;
    recovered.initialize(backends);
    auto * input = ggml_new_tensor_1d(recovered.context, GGML_TYPE_F32, 64);
    auto * bias = ggml_new_tensor_1d(recovered.context, GGML_TYPE_F32, 64);
    ggml_set_input(input);
    ggml_set_input(bias);
    auto * intermediate = ggml_scale(recovered.context, input, 3.0f);
    auto * output = ggml_add(recovered.context, intermediate, bias);
    ggml_set_output(output);
    ggml_build_forward_expand(recovered.graph, output);
    auto * gpu = backends.handles[0];
    auto * cpu = backends.handles[1];
    ggml_backend_sched_set_tensor_backend(recovered.scheduler, input, cpu);
    ggml_backend_sched_set_tensor_backend(recovered.scheduler, bias, cpu);
    ggml_backend_sched_set_tensor_backend(recovered.scheduler, intermediate, gpu);
    ggml_backend_sched_set_tensor_backend(recovered.scheduler, output, cpu);
    require(tsg_scheduler_alloc_graph(recovered.scheduler, recovered.graph),
            "Fresh small cross-device graph failed allocation after CUDA OOM");
    require(ggml_backend_sched_get_tensor_backend(recovered.scheduler, input) == cpu &&
            ggml_backend_sched_get_tensor_backend(recovered.scheduler, bias) == cpu &&
            ggml_backend_sched_get_tensor_backend(recovered.scheduler, intermediate) == gpu &&
            ggml_backend_sched_get_tensor_backend(recovered.scheduler, output) == cpu,
            "Explicit CPU/GPU placement was lost across checked reserve");
    require(intermediate->src[0] != input && output->src[0] != intermediate,
            "Fixture did not exercise both CPU-to-CUDA and CUDA-to-CPU copy sources");
    require(ggml_backend_buffer_is_host(input->buffer) && ggml_backend_buffer_is_host(output->buffer),
            "Recovery input/output storage is not on CPU");
    require(ggml_backend_buffer_get_type(intermediate->buffer) == ggml_backend_get_default_buffer_type(gpu),
            "Recovery scale did not allocate in the CUDA arena");
    require(ggml_backend_buffer_get_type(intermediate->src[0]->buffer) == ggml_backend_get_default_buffer_type(gpu) &&
            ggml_backend_buffer_is_host(output->src[0]->buffer), "Cross-device sources belong to the wrong arenas");

    std::array<float, 64> values{}, biases{}, actual{};
    for (int pass = 0; pass < 2; ++pass) {
        for (size_t i = 0; i < values.size(); ++i) {
            values[i] = float(int(i) - 31) * 0.125f + 2.0f * pass;
            biases[i] = float(int(i % 7) - 3) * 0.25f - 0.5f * pass;
        }
        ggml_backend_tensor_set(input, values.data(), 0, sizeof(values));
        ggml_backend_tensor_set(bias, biases.data(), 0, sizeof(biases));
        require(ggml_backend_sched_graph_compute(recovered.scheduler, recovered.graph) == GGML_STATUS_SUCCESS,
                "Fresh cross-device graph execution failed after CUDA allocation failure");
        ggml_backend_tensor_get(output, actual.data(), 0, sizeof(actual));
        for (size_t i = 0; i < actual.size(); ++i)
            require(actual[i] == 3.0f * values[i] + biases[i],
                    "Recovered CUDA/CPU graph disagrees with the exact independent scalar oracle");
        std::printf("Recovery pass %d: 64 exact outputs, changed CPU inputs, CUDA scale and CPU output passed\n", pass + 1);
    }
}
}

int main() {
    allocation_log log;
    ggml_log_set(allocation_log::write, &log);
    try {
        int count = 0;
        const cudaError_t devices = cudaGetDeviceCount(&count);
        if (devices != cudaSuccess || count == 0) {
            std::printf("SKIP: CUDA unavailable (%s, device_count=%d)\n", cudaGetErrorString(devices), count);
            ggml_log_set(nullptr, nullptr);
            return 77;
        }
        // cudaMallocManaged may oversubscribe VRAM and would invalidate the
        // controlled OOM mechanism. This environment change affects this test
        // process only, before creating any ggml CUDA backend.
#ifdef _WIN32
        require(_putenv_s("GGML_CUDA_ENABLE_UNIFIED_MEMORY", "") == 0, "Cannot disable managed allocation in fixture");
#else
        require(unsetenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") == 0, "Cannot disable managed allocation in fixture");
#endif
        {
            backend_pair backends;
            backends.handles[0] = ggml_backend_cuda_init(0);
            backends.handles[1] = ggml_backend_cpu_init();
            require(backends.handles[0] && backends.handles[1], "Could not initialize CUDA/CPU backends");
            ggml_backend_cpu_set_n_threads(backends.handles[1], 1);
            size_t logical_free = 0, logical_total = 0;
            ggml_backend_cuda_get_device_memory(0, &logical_free, &logical_total);
            // ggml's query selects the physical device behind backend zero.
            // Query that device directly because virtual ggml devices report
            // only a share of physical capacity, which cannot guarantee OOM.
            size_t physical_free = 0, physical_total = 0;
            require(cudaMemGetInfo(&physical_free, &physical_total) == cudaSuccess, "CUDA device memory query failed");
            std::printf("CUDA backend 0: free_bytes=%zu total_bytes=%zu logical_total_bytes=%zu; unified_memory=disabled\n",
                        physical_free, physical_total, logical_total);
            fail_device_allocation(backends, physical_total, log);
            fresh_cross_device_graph(backends);
        }
        ggml_log_set(nullptr, nullptr);
        std::puts("Actual CUDA allocation failure, discard and fresh small-graph recovery passed. This is not full-model recovery evidence.");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        ggml_log_set(nullptr, nullptr);
        return 1;
    }
}
