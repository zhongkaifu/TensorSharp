// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Uses unchanged upstream ggml; this size-only wrapper never allocates payloads.
#pragma once
#include "ggml-alloc.h"
#include "ggml-backend-impl.h"
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>

namespace tsg::native18_draft {

// ggml_gallocr_new_n aliases repeated buft POINTERS. A distinct size-only
// descriptor allows measuring a scratch owner separately from the real weight
// type, preserving the real type's alignment/quantized padding calculations.
class SizeOnlyBufferType final {
    ggml_backend_buffer_type_t real_;
    ggml_backend_buffer_type proxy_{};
    std::size_t allocation_attempts_ = 0;

    static SizeOnlyBufferType& self(ggml_backend_buffer_type_t value) {
        return *static_cast<SizeOnlyBufferType*>(value->context);
    }
    static const char* name(ggml_backend_buffer_type_t) { return "TS-private-size-only"; }
    static ggml_backend_buffer_t deny_allocate(ggml_backend_buffer_type_t value, std::size_t) {
        ++self(value).allocation_attempts_; return nullptr;
    }
    static ggml_backend_buffer_t deny_allocate_n(ggml_backend_buffer_type_t value, ggml_tensor**, int) {
        ++self(value).allocation_attempts_; return nullptr;
    }
    static std::size_t alignment(ggml_backend_buffer_type_t value) {
        return ggml_backend_buft_get_alignment(self(value).real_);
    }
    static std::size_t max_size(ggml_backend_buffer_type_t value) {
        return ggml_backend_buft_get_max_size(self(value).real_);
    }
    static std::size_t alloc_size(ggml_backend_buffer_type_t value, const ggml_tensor* tensor) {
        return ggml_backend_buft_get_alloc_size(self(value).real_, tensor);
    }
    static std::size_t alloc_size_n(ggml_backend_buffer_type_t value, ggml_tensor** tensors, int count) {
        return ggml_backend_buft_get_alloc_size_n(self(value).real_, tensors, count);
    }
    static bool host(ggml_backend_buffer_type_t value) {
        return ggml_backend_buft_is_host(self(value).real_);
    }
public:
    explicit SizeOnlyBufferType(ggml_backend_buffer_type_t real) : real_(real) {
        if (!real) throw std::invalid_argument("missing real buffer type");
        proxy_.iface = {name, deny_allocate, deny_allocate_n, alignment,
            max_size, alloc_size, alloc_size_n, host};
        proxy_.device = real->device;
        proxy_.context = this;
    }
    SizeOnlyBufferType(const SizeOnlyBufferType&) = delete;
    SizeOnlyBufferType& operator=(const SizeOnlyBufferType&) = delete;
    ggml_backend_buffer_type_t get() noexcept { return &proxy_; }
    std::size_t allocation_attempts() const noexcept { return allocation_attempts_; }
};

struct SplitBytes {
    // IDs: narrow 0; persistent weight leaves 1; wide 2.
    std::size_t narrow = 0, weights = 0, wide = 0;
    std::size_t total() const {
        const auto limit = static_cast<std::size_t>(std::numeric_limits<std::int64_t>::max());
        if (weights > limit || narrow > limit - weights || wide > limit - weights - narrow)
            throw std::overflow_error("split native ownership exceeds signed budget range");
        return weights + narrow + wide;
    }
};

// Graph must be newly built, entirely no-alloc, with weight leaves protected as
// outputs and IDs explicitly mapped by caller. No fake data/buffer may be used.
// This helper intentionally does NOT touch a published/live allocator.
inline SplitBytes measure_split_graph(ggml_backend_buffer_type_t real,
    ggml_cgraph* graph, const int* node_ids, const int* leaf_ids) {
    if (!graph || !node_ids || !leaf_ids)
        throw std::invalid_argument("fresh split graph and complete ownership maps required");
    for (int index = 0; index < graph->n_nodes; ++index) {
        const auto* tensor = graph->nodes[index];
        if (!tensor || (node_ids[index] != 0 && node_ids[index] != 2)
            || tensor->data || tensor->buffer)
            throw std::invalid_argument("fresh scratch node has wrong ownership or storage");
    }
    int weight_leaves = 0;
    for (int index = 0; index < graph->n_leafs; ++index) {
        const auto* tensor = graph->leafs[index];
        if (!tensor || leaf_ids[index] < 0 || leaf_ids[index] > 2
            || tensor->data || tensor->buffer)
            throw std::invalid_argument("fresh leaf has wrong ownership or storage");
        if (leaf_ids[index] == 1) {
            if (!ggml_is_quantized(tensor->type) || !(tensor->flags & GGML_TENSOR_FLAG_OUTPUT))
                throw std::invalid_argument("persistent quantized weight leaf is unprotected");
            ++weight_leaves;
        }
    }
    if (weight_leaves != 3)
        throw std::invalid_argument("exact three persistent weight leaves required");
    SizeOnlyBufferType narrow(real), wide(real);
    ggml_backend_buffer_type_t types[3] = {narrow.get(), real, wide.get()};
    if (types[0] == types[1] || types[0] == types[2] || types[1] == types[2])
        throw std::logic_error("split planner buffer types aliased");
    using Planner = std::unique_ptr<ggml_gallocr, decltype(&ggml_gallocr_free)>;
    Planner planner(ggml_gallocr_new_n(types, 3), ggml_gallocr_free);
    if (!planner) throw std::runtime_error("cannot create split graph planner");
    std::array<std::size_t, 3> size{};
    ggml_gallocr_reserve_n_size(planner.get(), graph, node_ids, leaf_ids, size.data());
    if (narrow.allocation_attempts() || wide.allocation_attempts())
        throw std::logic_error("size-only planner attempted physical scratch allocation");
    for (int id = 0; id < 3; ++id)
        if (ggml_gallocr_get_buffer_size(planner.get(), id) != 0)
            throw std::logic_error("fresh size-only planner owns physical storage");
    SplitBytes result{size[0], size[1], size[2]};
    result.total();
    return result;
}

// NOT an authorization to invoke reserve_n_size on a live allocator. Caller
// must also prove actual external weights, same graph/padding/ownership IDs,
// fenced old work, physical base identity, fresh graph UID, and no implicit grow.
inline bool is_single_chunk_candidate(ggml_backend_buffer_type_t real,
    std::size_t new_scratch, std::size_t charged_physical_scratch) {
    return real && ggml_backend_buft_get_max_size(real) == std::numeric_limits<std::size_t>::max()
        && new_scratch < std::numeric_limits<std::size_t>::max() / 2
        && charged_physical_scratch < std::numeric_limits<std::size_t>::max() / 2
        && new_scratch <= charged_physical_scratch;
}

} // namespace tsg::native18_draft
