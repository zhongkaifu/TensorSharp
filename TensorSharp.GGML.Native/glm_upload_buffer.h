// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_shared_cache_budget.h"
#include "glm_upload_plan.h"
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>

namespace tsg_glm {

struct upload_allocator {
    static uint8_t * allocate(size_t bytes) { return new uint8_t[bytes]; }
    static void release(uint8_t * data) noexcept { delete[] data; }
};

// A worker's disposable read/upload scratch. Unlike vector growth, capacity is
// exact and charged BEFORE physical allocation. Contents are overwritten by the
// next file read, so growth can free the old buffer before reserving its
// replacement. The caller must finish the synchronous backend upload first.
// The charge covers payload capacity, not allocator bookkeeping, FILE buffers,
// thread stacks, driver staging or the model's pageable mappings.
template<class Allocator = upload_allocator>
class upload_buffer {
    struct deleter { void operator()(uint8_t * data) const noexcept { Allocator::release(data); } };
    // Destruction order also frees physical storage before returning credit.
    std::shared_ptr<tsg::SharedCacheCharge> charge_;
    std::unique_ptr<uint8_t[], deleter> data_;
    size_t size_ = 0;
public:
    upload_buffer() = default;
    upload_buffer(const upload_buffer &) = delete;
    upload_buffer & operator=(const upload_buffer &) = delete;
    size_t size() const noexcept { return size_; }
    uint8_t * data() noexcept { return data_.get(); }
    void clear() noexcept {
        data_.reset();
        size_ = 0;
        charge_.reset();
    }
    void resize(size_t bytes) {
        if (bytes <= size_) return;
        clear();
        auto charge = tsg::SharedCacheCharge::reserve(0, 3, bytes);
        if (!charge) throw std::runtime_error("GLM upload workspace exceeds shared host budget");
        std::unique_ptr<uint8_t[], deleter> data(Allocator::allocate(bytes));
        if (!data) throw std::bad_alloc();
        if (!charge->commit(bytes)) throw std::runtime_error("GLM upload workspace commit refused");
        charge_ = std::move(charge);
        data_ = std::move(data);
        size_ = bytes;
    }
};

// Non-owning worker slices cannot grow, allocate or create another budget owner.
class upload_buffer_view {
    uint8_t * data_;
    size_t size_;
public:
    upload_buffer_view(uint8_t * data, size_t bytes) : data_(data), size_(bytes) {}
    size_t size() const noexcept { return size_; }
    uint8_t * data() const noexcept { return data_; }
    void resize(size_t bytes) {
        if (bytes > size_) throw std::runtime_error("GLM upload exceeded its preplanned worker slice");
    }
};

template<class Allocator = upload_allocator>
class upload_arena {
    struct deleter { void operator()(uint8_t * data) const noexcept { Allocator::release(data); } };
    // Member destruction (also on a throwing constructor) is physical free,
    // then credit release. A failing commit cannot strand payload or a ticket.
    std::shared_ptr<tsg::SharedCacheCharge> charge_;
    std::unique_ptr<uint8_t[], deleter> data_;
    size_t workers_, staging_, slab_, stride_;
public:
    explicit upload_arena(upload_plan & plan)
        : charge_(std::move(plan.reservation)), workers_(plan.workers), staging_(plan.staging),
          slab_(plan.slab), stride_(plan.worker_bytes) {
        if (!charge_ || !plan.accounted || !plan.aggregate)
            throw std::invalid_argument("GLM upload arena requires an accounted plan");
        data_.reset(Allocator::allocate(plan.aggregate));
        if (!data_) throw std::bad_alloc();
        if (!charge_->commit(plan.aggregate)) throw std::runtime_error("GLM upload workspace commit refused");
    }
    upload_arena(const upload_arena &) = delete;
    upload_arena & operator=(const upload_arena &) = delete;
    upload_buffer_view staging(size_t worker) {
        if (worker >= workers_) throw std::out_of_range("GLM upload worker slice");
        return {data_.get() + worker * stride_, staging_};
    }
    upload_buffer_view slab(size_t worker) {
        if (worker >= workers_) throw std::out_of_range("GLM upload worker slab");
        return {data_.get() + worker * stride_ + staging_, slab_};
    }
};
} // namespace tsg_glm
