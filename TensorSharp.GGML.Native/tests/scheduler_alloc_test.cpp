// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#include "ggml_ops_scheduler_alloc.h"
#include "ggml-backend-impl.h"
#include "ggml-cpu.h"
#include <array>
#include <cstdio>
#include <new>
#include <stdexcept>

namespace {
void require(bool condition, const char * message) {
    if (!condition) throw std::runtime_error(message);
}
struct failing_buffer {
    ggml_backend_buffer_type type{};
    ggml_backend_buffer_type_t cpu = ggml_backend_cpu_buffer_type();
    bool fail = false;
    bool throw_on_failure = false;
    int attempts = 0;
    int live_buffers = 0;
    void (*cpu_free_buffer)(ggml_backend_buffer_t) = nullptr;
    failing_buffer() {
        type.context = this;
        type.device = ggml_backend_buft_get_device(cpu);
        type.iface.get_name = [](ggml_backend_buffer_type_t) { return "TensorSharpAllocationFailureFixture"; };
        type.iface.get_alignment = [](ggml_backend_buffer_type_t t) {
            return ggml_backend_buft_get_alignment(static_cast<failing_buffer *>(t->context)->cpu);
        };
        type.iface.is_host = [](ggml_backend_buffer_type_t) { return true; };
        type.iface.alloc_buffer = [](ggml_backend_buffer_type_t t, size_t bytes) -> ggml_backend_buffer_t {
            auto & state = *static_cast<failing_buffer *>(t->context);
            ++state.attempts;
            if (state.fail) {
                if (state.throw_on_failure) throw std::bad_alloc();
                return nullptr;
            }
            auto * buffer = ggml_backend_buft_alloc_buffer(state.cpu, bytes);
            if (!buffer) return nullptr;
            state.cpu_free_buffer = buffer->iface.free_buffer;
            ++state.live_buffers;
            // Keep the real CPU allocation/storage callbacks, but expose this
            // fixture's buffer identity to supports_buft and track destruction.
            buffer->buft = t;
            buffer->iface.free_buffer = [](ggml_backend_buffer_t buffer) {
                auto & owner = *static_cast<failing_buffer *>(buffer->buft->context);
                --owner.live_buffers;
                if (owner.cpu_free_buffer) owner.cpu_free_buffer(buffer);
            };
            return buffer;
        };
    }
};

// CPU storage and execution, but mutually incompatible device buffer types.
// Two ordinary CPU handles both accept host buffers and never enter the
// scheduler's cross-device source-copy rewrite path; this fixture must do so.
struct incompatible_cpu_backend {
    ggml_backend_t cpu = ggml_backend_cpu_init();
    failing_buffer buffer;
    ggml_backend_device device{};
    ggml_backend backend{};
    incompatible_cpu_backend() {
        require(cpu != nullptr, "Isolated CPU backend creation failed");
        ggml_backend_cpu_set_n_threads(cpu, 1);
        device.context = this;
        device.iface.get_name = [](ggml_backend_dev_t) { return "TensorSharpIsolatedCpuDevice"; };
        device.iface.get_description = device.iface.get_name;
        device.iface.get_type = [](ggml_backend_dev_t) { return GGML_BACKEND_DEVICE_TYPE_CPU; };
        device.iface.get_memory = [](ggml_backend_dev_t, size_t * free, size_t * total) { *free = *total = 0; };
        device.iface.get_props = [](ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
            const auto & self = *static_cast<incompatible_cpu_backend *>(dev->context);
            ggml_backend_dev_get_props(ggml_backend_get_device(self.cpu), props);
        };
        device.iface.get_buffer_type = [](ggml_backend_dev_t dev) {
            return &static_cast<incompatible_cpu_backend *>(dev->context)->buffer.type;
        };
        device.iface.supports_op = [](ggml_backend_dev_t dev, const ggml_tensor * op) {
            return ggml_backend_supports_op(static_cast<incompatible_cpu_backend *>(dev->context)->cpu, op);
        };
        device.iface.supports_buft = [](ggml_backend_dev_t dev, ggml_backend_buffer_type_t type) {
            return type == &static_cast<incompatible_cpu_backend *>(dev->context)->buffer.type;
        };
        buffer.type.device = &device;
        // CPU callbacks receive their original context/guid. The wrapper owns
        // neither: only cpu is freed, after every scheduler using it is gone.
        backend = *cpu;
        backend.device = &device;
        backend.iface.free = nullptr;
    }
    ~incompatible_cpu_backend() { ggml_backend_free(cpu); }
};
struct graph_fixture {
    ggml_context * context = nullptr;
    ggml_backend_sched_t scheduler = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_tensor * input = nullptr;
    ggml_tensor * second_input = nullptr;
    ggml_tensor * output = nullptr;
    ggml_tensor * intermediate = nullptr;
    graph_fixture(std::array<ggml_backend_t, 2> & backends, std::array<ggml_backend_buffer_type_t, 2> & types) {
        context = ggml_init({1024 * 1024, nullptr, true});
        require(context != nullptr, "Context allocation failed");
        scheduler = ggml_backend_sched_new(backends.data(), types.data(), 2, GGML_DEFAULT_GRAPH_SIZE, false, true);
        require(scheduler != nullptr, "Scheduler creation failed");
        input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 64);
        ggml_set_input(input);
        second_input = ggml_new_tensor_1d(context, GGML_TYPE_F32, 64);
        ggml_set_input(second_input);
        intermediate = ggml_scale(context, input, 3.0f);
        // Both backend arenas must allocate, and the final output is explicitly
        // placed on the lower-priority CPU handle. Losing that placement across
        // reserve's reset would otherwise silently execute on handle zero.
        output = ggml_add(context, intermediate, second_input);
        ggml_set_output(output);
        graph = ggml_new_graph(context);
        ggml_build_forward_expand(graph, output);
        ggml_backend_sched_set_tensor_backend(scheduler, input, backends[0]);
        ggml_backend_sched_set_tensor_backend(scheduler, second_input, backends[1]);
        ggml_backend_sched_set_tensor_backend(scheduler, intermediate, backends[0]);
        ggml_backend_sched_set_tensor_backend(scheduler, output, backends[1]);
    }
    ~graph_fixture() {
        ggml_backend_sched_free(scheduler);
        ggml_free(context);
    }
};
void numerical_run(graph_fixture & fixture, const std::array<ggml_backend_t, 2> & backends, float offset = 0) {
    require(ggml_backend_sched_get_tensor_backend(fixture.scheduler, fixture.input) == backends[0], "Leaf placement lost");
    require(ggml_backend_sched_get_tensor_backend(fixture.scheduler, fixture.second_input) == backends[1], "Second leaf placement lost");
    require(ggml_backend_sched_get_tensor_backend(fixture.scheduler, fixture.intermediate) == backends[0], "Intermediate placement lost");
    require(ggml_backend_sched_get_tensor_backend(fixture.scheduler, fixture.output) == backends[1], "Output placement lost");
    std::array<float, 64> input{}, output{};
    for (size_t i = 0; i < input.size(); ++i) input[i] = float(int(i) - 31) / 8 + offset;
    ggml_backend_tensor_set(fixture.input, input.data(), 0, sizeof(input));
    ggml_backend_tensor_set(fixture.second_input, input.data(), 0, sizeof(input));
    require(ggml_backend_sched_graph_compute(fixture.scheduler, fixture.graph) == GGML_STATUS_SUCCESS, "Recovered graph computation failed");
    ggml_backend_tensor_get(fixture.output, output.data(), 0, sizeof(output));
    for (size_t i = 0; i < input.size(); ++i) require(output[i] == 4 * input[i], "Recovered graph result differs from scalar oracle");
}

void cross_device_sources() {
    std::array<incompatible_cpu_backend, 2> devices;
    std::array<ggml_backend_t, 2> backends{&devices[0].backend, &devices[1].backend};
    std::array<ggml_backend_buffer_type_t, 2> types{&devices[0].buffer.type, &devices[1].buffer.type};
    require(!ggml_backend_supports_buft(backends[0], types[1]) &&
            !ggml_backend_supports_buft(backends[1], types[0]), "Fixture devices are not incompatible");
    {
        // Establish the mechanism directly: checked reserve itself mutates
        // the caller's source link even though it does not execute the graph.
        graph_fixture probe(backends, types);
        require(ggml_backend_sched_reserve(probe.scheduler, probe.graph), "Cross-device probe reserve failed");
        require(probe.output->src[0] != probe.intermediate,
                "Fixture failed to exercise reserve's cross-device source rewrite");
    }
    require(devices[0].buffer.live_buffers == 0 && devices[1].buffer.live_buffers == 0, "Probe leaked an arena");
    for (bool throw_on_failure : {false, true})
    for (int failing_rank = 0; failing_rank < 2; ++failing_rank) {
        auto & failed_buffer = devices[failing_rank].buffer;
        std::printf("Cross-device failure rank=%d kind=%s\n", failing_rank, throw_on_failure ? "bad_alloc" : "null");
        failed_buffer.fail = true;
        failed_buffer.throw_on_failure = throw_on_failure;
        const int previous_attempts = failed_buffer.attempts;
        {
            graph_fixture failed(backends, types);
            require(!tsg_scheduler_alloc_graph(failed.scheduler, failed.graph), "Cross-device allocation failure was ignored");
            std::printf("Failure helper returned false rank=%d kind=%s\n", failing_rank, throw_on_failure ? "bad_alloc" : "null");
            require(failed_buffer.attempts == previous_attempts + 1, "Cross-device allocation was retried unsafely");
            require(failed.output->src[0] == failed.intermediate && failed.output->src[1] == failed.second_input,
                    "Failed reserve left scheduler-owned source copies in the caller's graph");
            require(failed.output->data == nullptr, "Failed cross-device graph exposed output storage");
        }
        std::printf("Failed graph destroyed rank=%d kind=%s\n", failing_rank, throw_on_failure ? "bad_alloc" : "null");
        require(devices[0].buffer.live_buffers == 0 && devices[1].buffer.live_buffers == 0,
                "Failed cross-device scheduler leaked a partial allocation");
        failed_buffer.fail = false;
        failed_buffer.throw_on_failure = false;
        {
            graph_fixture recovered(backends, types);
            require(tsg_scheduler_alloc_graph(recovered.scheduler, recovered.graph), "Fresh cross-device request did not recover");
            require(recovered.output->src[0] != recovered.intermediate,
                    "Allocated cross-device graph did not retain its live copy source");
            require(ggml_backend_buffer_get_type(recovered.output->src[0]->buffer) == types[1],
                    "Cross-device copy source belongs to the wrong arena");
            numerical_run(recovered, backends);
            numerical_run(recovered, backends, 3.0f);
        }
        require(devices[0].buffer.live_buffers == 0 && devices[1].buffer.live_buffers == 0, "Recovered cross-device request leaked an arena");
    }
    std::puts("Incompatible CPU devices: reserve rewrites sources; null/bad_alloc failures restore links; live cross-copy replay and changed-input recovery passed");
}
}

int main() {
    try {
        std::array<ggml_backend_t, 2> backends{ggml_backend_cpu_init(), ggml_backend_cpu_init()};
        for (auto backend : backends) {
            require(backend != nullptr, "CPU backend creation failed");
            ggml_backend_cpu_set_n_threads(backend, 1);
        }
        for (int failing_rank = 0; failing_rank < 2; ++failing_rank) {
            std::array<failing_buffer, 2> buffers;
            std::array<ggml_backend_buffer_type_t, 2> types{&buffers[0].type, &buffers[1].type};
            buffers[failing_rank].fail = true;
            {
                graph_fixture failed(backends, types);
                require(!tsg_scheduler_alloc_graph(failed.scheduler, failed.graph), "Injected allocation failure was ignored");
                require(buffers[failing_rank].attempts == 1, "Failed allocation was retried unsafely");
                require(failed.output->data == nullptr, "Failed graph exposed allocated output data");
            }
            require(buffers[0].live_buffers == 0 && buffers[1].live_buffers == 0, "Failed scheduler leaked an arena");
            // Free the failed scheduler, then prove that the same backend
            // handles can allocate and execute a new request without poisoning.
            buffers[failing_rank].fail = false;
            {
                graph_fixture recovered(backends, types);
                require(tsg_scheduler_alloc_graph(recovered.scheduler, recovered.graph), "Fresh graph could not recover after allocation failure");
                numerical_run(recovered, backends);
                numerical_run(recovered, backends);
            }
        }
        for (auto backend : backends) ggml_backend_free(backend);
        cross_device_sources();
        std::puts("Allocation failure on first/later backend, no unsafe retry, destruction, explicit placement and fresh-request recovery passed");
        return 0;
    } catch (const std::exception & error) {
        std::fprintf(stderr, "%s\n", error.what());
        return 1;
    }
}
