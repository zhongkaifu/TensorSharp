// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_internal.h"
#include "ggml_ops_q8_weight_policy.h"
#include "ggml_ops_q8_precision.h"
#include "ggml_ops_dsv4_fused.h"
#include "ggml-impl.h"
#include <limits>
#if defined(TSG_GGML_USE_CUDA)
#include "ggml-cuda.h"
#endif

namespace {
std::mutex policy_mutex;
std::unordered_map<const void*, std::int64_t> precise_keys;
std::atomic<std::size_t> precise_key_count{0};
std::mutex backend_mutex;
// Wrappers borrow the underlying CUDA backend. Graphs/plans never retain them;
// each submission resolves its current wrapper so cache teardown can release it.
std::unordered_map<ggml_backend_t, ggml_backend_t> execution_backends;
#if defined(TSG_GGML_TEST_HOOKS)
std::atomic<bool> fail_next_registration{false};
#endif

bool has_precise_key(const void* key)
{
    if (precise_key_count.load(std::memory_order_acquire) == 0) return false;
    std::lock_guard<std::mutex> lock(policy_mutex);
    return precise_keys.find(key) != precise_keys.end();
}

bool contains_q8_f32(ggml_cgraph* graph)
{
    for (int i = 0; i < ggml_graph_n_nodes(graph); ++i)
    {
        if (tsg_is_matmul_q8_f32(ggml_graph_node(graph, i))) return true;
    }
    return false;
}
}

TSG_EXPORT int TSGgml_RegisterQ8F32Weight(void* key)
{
    if (key == nullptr) { tsg::set_last_error("Precise Q8 weight key is null."); return 0; }
#if defined(TSG_GGML_TEST_HOOKS)
    if (fail_next_registration.exchange(false, std::memory_order_acq_rel))
    {
        tsg::set_last_error("Injected precise Q8 registration failure.");
        return 0;
    }
#endif
    try
    {
        std::lock_guard<std::mutex> lock(policy_mutex);
        auto& count = precise_keys[key];
        if (count == std::numeric_limits<std::int64_t>::max())
            throw std::overflow_error("Precise Q8 weight reference count overflow.");
        ++count;
        precise_key_count.store(precise_keys.size(), std::memory_order_release);
        return 1;
    }
    catch (const std::exception& error) { tsg::set_last_error(error.what()); return 0; }
}

TSG_EXPORT void TSGgml_UnregisterQ8F32Weight(void* key)
{
    std::lock_guard<std::mutex> lock(policy_mutex);
    const auto found = precise_keys.find(key);
    if (found == precise_keys.end()) return;
    if (--found->second == 0) precise_keys.erase(found);
    precise_key_count.store(precise_keys.size(), std::memory_order_release);
}

#if defined(TSG_GGML_TEST_HOOKS)
#define TSG_Q8_TEST_EXPORT TSG_EXPORT
TSG_Q8_TEST_EXPORT void TSGgml_TestQ8F32FailNextRegistration()
{
    fail_next_registration.store(true, std::memory_order_release);
}
TSG_Q8_TEST_EXPORT std::int64_t TSGgml_TestQ8F32WeightRegistrationCount(void* key)
{
    std::lock_guard<std::mutex> lock(policy_mutex);
    const auto found = precise_keys.find(key);
    return found == precise_keys.end() ? 0 : found->second;
}
#undef TSG_Q8_TEST_EXPORT
#endif

namespace tsg {
ggml_tensor* weight_mul_mat(ggml_context* context, ggml_tensor* weight,
    ggml_tensor* input, const void* key)
{
    if (g_backend_type == BACKEND_TYPE_CUDA && weight->type == GGML_TYPE_Q8_0 &&
        input->type == GGML_TYPE_F32 && weight->ne[2] == 1 && weight->ne[3] == 1 &&
        input->ne[2] == 1 && input->ne[3] == 1 && has_precise_key(key))
        return tsg_matmul_q8_f32(context, weight, input);
    return ggml_mul_mat(context, weight, input);
}

ggml_backend_t q8_f32_execution_backend(ggml_backend_t backend, ggml_cgraph* graph)
{
#if defined(TSG_GGML_USE_CUDA)
    if (g_backend_type != BACKEND_TYPE_CUDA || !contains_q8_f32(graph)) return backend;
    // Other graph families can supply their own compatible fused-op wrapper.
    // Only raw CUDA backends may be wrapped; those wrappers already handle Q8.
    if (!ggml_backend_is_cuda(backend)) return backend;
    std::lock_guard<std::mutex> lock(backend_mutex);
    for (const auto& item : execution_backends)
        if (item.second == backend) return backend;
    const auto found = execution_backends.find(backend);
    if (found != execution_backends.end()) return found->second;
    ggml_backend_t wrapper = tsg_dsv4_fused_backend_init(backend);
    if (wrapper == nullptr) throw std::runtime_error("Cannot initialize precise Q8 execution backend.");
    try { execution_backends.emplace(backend, wrapper); }
    catch (...) { ggml_backend_free(wrapper); throw; }
    return wrapper;
#else
    return backend;
#endif
}

void clear_q8_f32_backends()
{
    std::lock_guard<std::mutex> lock(backend_mutex);
    for (const auto& item : execution_backends)
    {
        ggml_backend_synchronize(item.second);
        ggml_backend_free(item.second);
    }
    execution_backends.clear();
}
}
