// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_graph_budget.h"
#include "ggml_ops_shared_cache_budget.h"
#include "ggml-backend-impl.h"
#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

namespace tsg {
namespace {
    struct Owner {
        ggml_backend_buffer_type type{};
        ggml_backend_buffer_type_t original;
        int rank;
        std::mutex operation;
        std::vector<std::shared_ptr<SharedCacheCharge>> charges;
        bool gallocr = false;
        bool generation_started = false;

        static Owner& self(ggml_backend_buffer_type_t type) { return *static_cast<Owner*>(type->context); }
        static const char* name(ggml_backend_buffer_type_t type) { return ggml_backend_buft_name(self(type).original); }
        static size_t alignment(ggml_backend_buffer_type_t type) { return ggml_backend_buft_get_alignment(self(type).original); }
        static size_t maximum(ggml_backend_buffer_type_t type) { return ggml_backend_buft_get_max_size(self(type).original); }
        static size_t tensor_size(ggml_backend_buffer_type_t type, const ggml_tensor* tensor)
        { return ggml_backend_buft_get_alloc_size(self(type).original, tensor); }
        static size_t tensors_size(ggml_backend_buffer_type_t type, ggml_tensor** tensors, int count)
        { return ggml_backend_buft_get_alloc_size_n(self(type).original, tensors, count); }
        static bool host(ggml_backend_buffer_type_t type) { return ggml_backend_buft_is_host(self(type).original); }

        template<class Allocate>
        ggml_backend_buffer_t allocate(size_t requested, Allocate allocator) noexcept
        {
            try {
                if (gallocr && !generation_started) {
                    // Pinned ggml-alloc.c's single-buft reserve frees ALL old
                    // vbuffer chunks before the first new allocation callback.
                    // This is a lifecycle boundary, not pointer-address inference.
                    generation_started = true;
                    charges.clear();
                }
                auto charge = requested == 0 ? nullptr : SharedCacheCharge::reserve(rank, 2, requested);
                if (requested != 0 && !charge) return nullptr;
                std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
                    buffer(allocator(), ggml_backend_buffer_free);
                if (!buffer) return nullptr;
                // Budget GGML's declared buffer payload, not hidden driver/page
                // rounding. With a configured graph budget, commit rejects an
                // allocator exceeding its bound; free precedes reservation release.
                // Unconfigured/cache-only scopes preserve the original backend's
                // padding behavior while retaining a live marker for future attach.
                const size_t actual = ggml_backend_buffer_get_size(buffer.get());
                if (!charge && actual != 0) return nullptr;
                if (charge) {
                    charges.push_back(charge); // all fallible bookkeeping precedes publication
                    if (!charge->commit(actual)) { charges.pop_back(); return nullptr; }
                }
                return buffer.release();
            } catch (...) { return nullptr; }
        }
        static ggml_backend_buffer_t alloc(ggml_backend_buffer_type_t type, size_t bytes)
        {
            auto& owner = self(type);
            return owner.allocate(bytes, [&] { return ggml_backend_buft_alloc_buffer(owner.original, bytes); });
        }
        static ggml_backend_buffer_t alloc_n(ggml_backend_buffer_type_t type, ggml_tensor** tensors, int count)
        {
            auto& owner = self(type);
            try {
                const size_t bytes = ggml_backend_buft_get_alloc_size_n(owner.original, tensors, count);
                return owner.allocate(bytes, [&] { return ggml_backend_buft_alloc_buffer_n(owner.original, tensors, count); });
            } catch (...) { return nullptr; }
        }
        Owner(ggml_backend_buffer_type_t source, int device_rank) : original(source), rank(device_rank)
        {
            type.iface = { name, alloc, source->iface.alloc_buffer_n ? alloc_n : nullptr,
                alignment, maximum, tensor_size, source->iface.get_alloc_size_n ? tensors_size : nullptr, host };
            type.device = source->device;
            type.context = this;
        }
    };

    struct Registry {
        std::mutex mutex;
        std::unordered_map<ggml_backend_buffer_t, std::shared_ptr<Owner>> buffers;
        std::unordered_map<ggml_gallocr_t, std::shared_ptr<Owner>> allocators;
    };
    Registry& registry() { static auto* value = new Registry; return *value; }

    template<class Allocate>
    ggml_backend_buffer_t allocate_owned(ggml_backend_buffer_type_t type, int rank, Allocate allocate)
    {
        if (!type || rank < 0) return nullptr;
        try {
            auto owner = std::make_shared<Owner>(type, rank);
            // Declaration order keeps all charges until the physical free,
            // including map-insertion failures and partial context-allocation failure.
            std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)>
                buffer(allocate(owner), ggml_backend_buffer_free);
            if (!buffer) return nullptr;
            auto& r = registry();
            {
                std::lock_guard<std::mutex> lock(r.mutex);
                if (!r.buffers.emplace(buffer.get(), owner).second) return nullptr;
            }
            return buffer.release();
        } catch (...) { return nullptr; }
    }
}

ggml_backend_buffer_t graph_budget_alloc_buffer(ggml_backend_buffer_type_t type, size_t bytes, int rank)
{
    return allocate_owned(type, rank, [&](const std::shared_ptr<Owner>& owner) {
        // Bypass upstream's zero-size wrapper-buft dummy: even an empty buffer
        // must keep its original backend type, never a temporary wrapper address.
        return owner->allocate(bytes, [&] { return ggml_backend_buft_alloc_buffer(type, bytes); });
    });
}

ggml_backend_buffer_t graph_budget_alloc_ctx_tensors(ggml_context* context, ggml_backend_t backend, int rank)
{
    if (!context || !backend) return nullptr;
    auto original = ggml_backend_get_default_buffer_type(backend);
    return allocate_owned(original, rank, [&](const std::shared_ptr<Owner>& owner) {
        return ggml_backend_alloc_ctx_tensors_from_buft(context, &owner->type);
    });
}

void graph_budget_free_buffer(ggml_backend_buffer_t buffer)
{
    if (!buffer) return;
    std::shared_ptr<Owner> owner;
    auto& r = registry();
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        auto found = r.buffers.find(buffer);
        if (found != r.buffers.end()) { owner = std::move(found->second); r.buffers.erase(found); }
    }
    // Never call physical free or managed accounting under the registry mutex.
    ggml_backend_buffer_free(buffer);
    // owner destruction refunds only after the original free has returned.
}

ggml_gallocr_t graph_budget_gallocr_new(ggml_backend_buffer_type_t type, int rank)
{
    if (!type || rank < 0) return nullptr;
    try {
        auto owner = std::make_shared<Owner>(type, rank);
        owner->gallocr = true;
        std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>
            allocator(ggml_gallocr_new(&owner->type), ggml_gallocr_free);
        if (!allocator) return nullptr;
        auto& r = registry();
        {
            std::lock_guard<std::mutex> lock(r.mutex);
            if (!r.allocators.emplace(allocator.get(), owner).second) return nullptr;
        }
        return allocator.release();
    } catch (...) { return nullptr; }
}

bool graph_budget_gallocr_alloc_graph(ggml_gallocr_t allocator, ggml_cgraph* graph)
{
    std::shared_ptr<Owner> owner;
    auto& r = registry();
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        auto found = r.allocators.find(allocator);
        if (found != r.allocators.end()) owner = found->second;
    }
    if (!owner) return ggml_gallocr_alloc_graph(allocator, graph);
    std::lock_guard<std::mutex> lock(owner->operation);
    owner->generation_started = false;
    const bool ok = ggml_gallocr_alloc_graph(allocator, graph);
    // On a failed single-buft vbuffer allocation upstream frees every successful
    // partial chunk before returning. An unchanged old generation is kept.
    if (!ok && owner->generation_started) owner->charges.clear();
    return ok;
}

void graph_budget_gallocr_free(ggml_gallocr_t allocator)
{
    if (!allocator) return;
    std::shared_ptr<Owner> owner;
    auto& r = registry();
    {
        std::lock_guard<std::mutex> lock(r.mutex);
        auto found = r.allocators.find(allocator);
        if (found != r.allocators.end()) { owner = std::move(found->second); r.allocators.erase(found); }
    }
    if (owner) {
        std::lock_guard<std::mutex> lock(owner->operation);
        ggml_gallocr_free(allocator);
        owner->charges.clear();
    } else ggml_gallocr_free(allocator);
}
}
