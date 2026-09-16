// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
// Build the actual owned helper into this executable: a global new override in
// a separate executable cannot intercept allocations inside a linked DLL.
#include "ggml_ops_attention_precision.h"
#include "ggml-cpu.h"
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <new>

#if defined(_MSC_VER)
#define TSG_NOINLINE __declspec(noinline)
#else
#define TSG_NOINLINE __attribute__((noinline))
#endif

static thread_local bool inject_pending_vector = false;
static thread_local int injected = 0, destroyed = 0;

void * operator new(std::size_t size)
{
    // on_backend builds graph metadata through ggml's arena, then constructs
    // vector<ggml_tensor*>{output}. The first C++ allocation is this one pointer.
    if (inject_pending_vector && size == sizeof(ggml_tensor *))
    {
        inject_pending_vector = false;
        ++injected;
        throw std::bad_alloc();
    }
    if (void * memory = std::malloc(size ? size : 1)) return memory;
    throw std::bad_alloc();
}
void operator delete(void * memory) noexcept { std::free(memory); }
void operator delete(void * memory, std::size_t) noexcept { std::free(memory); }

struct unwind_scope
{
    TSG_NOINLINE ~unwind_scope() { ++destroyed; }
};

static TSG_NOINLINE bool catch_helper_failure(ggml_context * ctx, ggml_backend_sched_t scheduler,
    ggml_backend_t cpu, ggml_tensor * q, ggml_tensor * k, ggml_tensor * v)
{
    try
    {
        unwind_scope guard;
        inject_pending_vector = true;
        tsg_attention_f32_on_backend(ctx, scheduler, cpu, q, k, v, nullptr, nullptr, 1.0f);
        inject_pending_vector = false;
    }
    catch (const std::bad_alloc &)
    {
        inject_pending_vector = false;
        return true;
    }
    return false;
}

static bool compute_exact(ggml_backend_sched_t scheduler, ggml_cgraph * graph,
    ggml_tensor * v, ggml_tensor * output, const float (&values)[4])
{
    ggml_backend_tensor_set(v, values, 0, sizeof(values));
    if (ggml_backend_sched_graph_compute(scheduler, graph) != GGML_STATUS_SUCCESS) return false;
    float actual[4] = {};
    ggml_backend_tensor_get(output, actual, 0, sizeof(actual));
    // One key has probability exactly one, independent of the dot product.
    for (int i = 0; i < 4; ++i) if (actual[i] != values[i]) return false;
    return true;
}

int main()
{
    // Report a failing process without opening a Windows crash dialog when the
    // C-linkage declaration regresses to MSVC's implicit no-throw assumption.
    std::set_terminate([] {
        std::fputs("FAIL: allocation exception terminated across owned helper boundary\n", stderr);
        std::fflush(stderr);
        std::_Exit(1);
    });
    auto * cpu = ggml_backend_cpu_init();
    if (!cpu) return 1;
    ggml_backend_cpu_set_n_threads(cpu, 1);
    auto * scheduler = ggml_backend_sched_new(&cpu, nullptr, 1, 256, false, true);
    auto * ctx = ggml_init({4 * 1024 * 1024, nullptr, true});
    if (!scheduler || !ctx) return 1;
    auto * q = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 1, 1, 1);
    auto * k = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 1, 1, 1);
    auto * v = ggml_new_tensor_4d(ctx, GGML_TYPE_F32, 4, 1, 1, 1);
    ggml_set_input(q); ggml_set_input(k); ggml_set_input(v);
    auto * output = tsg_attention_f32_on_backend(ctx, scheduler, cpu, q, k, v, nullptr, nullptr, 1.0f);
    ggml_set_output(output);
    auto * graph = ggml_new_graph_custom(ctx, 256, false);
    ggml_build_forward_expand(graph, output);
    if (!ggml_backend_sched_alloc_graph(scheduler, graph)) return 1;
    const float zeros[4] = {}, first[4] = {1.25f, -2.5f, 3.75f, 0.5f};
    const float changed[4] = {-8.5f, 12.25f, 0.125f, -32.0f};
    ggml_backend_tensor_set(q, zeros, 0, sizeof(zeros));
    ggml_backend_tensor_set(k, zeros, 0, sizeof(zeros));
    bool valid_before = compute_exact(scheduler, graph, v, output, first);
    bool caught = catch_helper_failure(ctx, scheduler, cpu, q, k, v);
    bool valid_after = compute_exact(scheduler, graph, v, output, changed);
    bool passed = valid_before && caught && injected == 1 && destroyed == 1 && valid_after;
    std::printf("attention exception boundary: before=%d caught=%d injected=%d unwound=%d changed_input=%d\n",
        valid_before, caught, injected, destroyed, valid_after);
    ggml_backend_sched_free(scheduler);
    ggml_free(ctx);
    ggml_backend_free(cpu);
    return passed ? 0 : 1;
}
