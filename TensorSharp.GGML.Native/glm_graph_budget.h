// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml-backend-impl.h"
#include "glm_graph_backend.h"
#include "ggml_ops_shared_cache_budget.h"
#include <memory>
#include <vector>

namespace tsg_glm {

// A cached GLM graph owns one scheduler, allocated exactly once. Its original
// backend buffers keep their original type/context/interfaces; only allocation
// requests pass through these temporary buffer-type descriptors. Destroy the
// scheduler before this owner so every physical free precedes its credit refund.
// Unlike a growable shared scheduler, no partial-generation refund is inferred.
class graph_budget {
    struct buffer_type {
        ggml_backend_buffer_type descriptor{};
        ggml_backend_buffer_type_t original;
        int rank, kind;
        std::vector<std::shared_ptr<tsg::SharedCacheCharge>> charges;
        static buffer_type& self(ggml_backend_buffer_type_t t) { return *static_cast<buffer_type*>(t->context); }
        static const char* name(ggml_backend_buffer_type_t t) { return ggml_backend_buft_name(self(t).original); }
        static size_t alignment(ggml_backend_buffer_type_t t) { return ggml_backend_buft_get_alignment(self(t).original); }
        static size_t maximum(ggml_backend_buffer_type_t t) { return ggml_backend_buft_get_max_size(self(t).original); }
        static size_t tensor_size(ggml_backend_buffer_type_t t, const ggml_tensor* v) { return ggml_backend_buft_get_alloc_size(self(t).original, v); }
        static size_t tensors_size(ggml_backend_buffer_type_t t, ggml_tensor** v, int n) { return ggml_backend_buft_get_alloc_size_n(self(t).original, v, n); }
        static bool host(ggml_backend_buffer_type_t t) { return ggml_backend_buft_is_host(self(t).original); }
        template<class Allocate> ggml_backend_buffer_t allocate(size_t bytes, Allocate allocate_original) noexcept {
            try {
                auto charge = bytes ? tsg::SharedCacheCharge::reserve(rank, kind, bytes) : nullptr;
                if (bytes && !charge) return nullptr;
                std::unique_ptr<ggml_backend_buffer, decltype(&ggml_backend_buffer_free)> buffer(allocate_original(), ggml_backend_buffer_free);
                if (!buffer) return nullptr;
                const size_t actual = ggml_backend_buffer_get_size(buffer.get());
                if (!charge && actual) return nullptr;
                if (charge) {
                    charges.push_back(charge);
                    if (!charge->commit(actual)) { charges.pop_back(); return nullptr; }
                }
                return buffer.release();
            } catch (...) { return nullptr; }
        }
        static ggml_backend_buffer_t alloc(ggml_backend_buffer_type_t t, size_t bytes) {
            auto& owner = self(t);
            return owner.allocate(bytes, [&] { return ggml_backend_buft_alloc_buffer(owner.original, bytes); });
        }
        static ggml_backend_buffer_t alloc_n(ggml_backend_buffer_type_t t, ggml_tensor** tensors, int count) {
            auto& owner = self(t);
            try {
                return owner.allocate(ggml_backend_buft_get_alloc_size_n(owner.original, tensors, count),
                    [&] { return ggml_backend_buft_alloc_buffer_n(owner.original, tensors, count); });
            } catch (...) { return nullptr; }
        }
        buffer_type(ggml_backend_buffer_type_t source, int device_rank, int allocation_kind)
            : original(source), rank(device_rank), kind(allocation_kind) {
            descriptor.iface = { name, alloc, source->iface.alloc_buffer_n ? alloc_n : nullptr,
                alignment, maximum, tensor_size, source->iface.get_alloc_size_n ? tensors_size : nullptr, host };
            descriptor.device = source->device;
            descriptor.context = this;
        }
    };
    std::vector<std::unique_ptr<buffer_type>> owners_;
    std::vector<ggml_backend_buffer_type_t> types_;
    std::vector<std::unique_ptr<graph_backend>> backend_owners_;
    std::vector<ggml_backend_t> backends_;
    static ggml_backend_buffer_type_t unwrap(void* context, ggml_backend_buffer_type_t type) {
        const auto& owner = *static_cast<graph_budget*>(context);
        for (const auto& item : owner.owners_) if (&item->descriptor == type) return item->original;
        return type;
    }
public:
    graph_budget() = default;
    graph_budget(const graph_budget&) = delete;
    graph_budget& operator=(const graph_budget&) = delete;
    ggml_backend_buffer_type_t* prepare(ggml_backend_buffer_type_t* types, ggml_backend_t* backends, int count, int n_gpu, int kind) {
        if (!owners_.empty()) return nullptr; // a graph scheduler has a single allocation generation
        for (int i = 0; i < count; ++i) {
            const int rank = ggml_backend_buft_is_host(types[i]) ? n_gpu :
                [&] { for (int d = 0; d < n_gpu; ++d) if (ggml_backend_get_default_buffer_type(backends[d]) == types[i]) return d; return 0; }();
            auto owner = std::unique_ptr<buffer_type>(new buffer_type(types[i], rank, kind));
            types_.push_back(&owner->descriptor);
            owners_.push_back(std::move(owner));
            auto backend = std::unique_ptr<graph_backend>(new graph_backend(backends[i], this, unwrap));
            backends_.push_back(backend->get());
            backend_owners_.push_back(std::move(backend));
        }
        return types_.data();
    }
    ggml_backend_t* scheduler_backends() { return backends_.data(); }
    ggml_backend_t scheduler_backend(ggml_backend_t original) {
        for (const auto& facade : backend_owners_) if (facade->original() == original) return facade->get();
        return nullptr;
    }
};
}
