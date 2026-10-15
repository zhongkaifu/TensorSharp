// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_shared_cache_budget.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <stdexcept>
#include <vector>

namespace tsg_dsv4 {

inline size_t warm_add(size_t a, size_t b) {
    if (b > (std::numeric_limits<size_t>::max)() - a)
        throw std::overflow_error("file warm workspace sum overflow");
    return a + b;
}
inline size_t warm_multiply(size_t a, size_t b) {
    if (a && b > (std::numeric_limits<size_t>::max)() / a)
        throw std::overflow_error("file warm workspace product overflow");
    return a * b;
}

struct file_warm_budget_plan {
    size_t workers = 0, block = 0, residency = 0, stride = 0, aggregate = 0;
    size_t refusals = 0;
    bool accounted = false, skipped = false;
    // Even an unaccounted ticket prevents attaching a new scope over active
    // legacy workers. A registry snapshot followed by allocation would race.
    std::shared_ptr<tsg::SharedCacheCharge> reservation;
};

// No file opens, page reads, worker creation or payload allocation here. The
// scratch bound covers a possibly unaligned mincore span: ceil(block/page)+1.
inline file_warm_budget_plan plan_file_warm_budget(uint64_t total, size_t threads,
    size_t block, size_t page, bool residency) {
    file_warm_budget_plan plan;
    if (!total) return plan;
    if (!threads || !block || !page) throw std::invalid_argument("invalid file warm workspace geometry");
    plan.block = block;
    plan.workers = (size_t) (std::min)(uint64_t(threads), total / block + (total % block != 0));
    const size_t minimum = (size_t) (std::min)(uint64_t(block), (std::min)(total, uint64_t(4096)));
    for (;;) {
        plan.residency = residency ? warm_add((plan.block - 1) / page, size_t(2)) : 0;
        plan.stride = warm_add(plan.block, plan.residency);
        plan.aggregate = warm_multiply(plan.workers, plan.stride);
        if (uint64_t(plan.aggregate) > uint64_t((std::numeric_limits<int64_t>::max)()))
            throw std::overflow_error("file warm workspace exceeds shared-budget integer range");
        plan.reservation = tsg::SharedCacheCharge::reserve(0, 3, plan.aggregate);
        if (plan.reservation) {
            plan.accounted = plan.reservation->is_accounted();
            return plan;
        }
        ++plan.refusals;
        if (plan.workers > 1) plan.workers = plan.workers / 2 + plan.workers % 2;
        else if (plan.block > minimum)
            plan.block = (std::max)(minimum, (size_t) (std::min)(total, uint64_t(plan.block / 2)));
        else {
            // Optional warming cannot turn a small shared quota into a loading
            // failure or an unaccounted fallback. Demand reads remain unchanged.
            plan.skipped = true;
            return plan;
        }
    }
}

// A legacy lazy vector or a fixed slice in the pre-reserved arena. The fixed
// mode cannot silently grow while mincore is executing on a worker.
class file_warm_scratch {
    std::vector<unsigned char> legacy_;
    unsigned char * fixed_ = nullptr;
    size_t capacity_ = 0;
public:
    file_warm_scratch() = default;
    file_warm_scratch(unsigned char * data, size_t capacity) : fixed_(data), capacity_(capacity) {}
    void resize(size_t bytes) {
        if (fixed_) {
            if (bytes > capacity_) throw std::out_of_range("file warm residency exceeded its reserved slice");
        } else legacy_.resize(bytes);
    }
    unsigned char * data() noexcept { return fixed_ ? fixed_ : legacy_.data(); }
    unsigned char & operator[](size_t index) noexcept { return data()[index]; }
};

struct file_warm_allocator {
    static unsigned char * allocate(size_t bytes) { return new unsigned char[bytes]; }
    static void release(unsigned char * data) noexcept { delete[] data; }
};

template<class Allocator = file_warm_allocator>
class file_warm_arena {
    struct deleter { void operator()(unsigned char * p) const noexcept { Allocator::release(p); } };
    // Reverse destruction order is physical free before credit release, also
    // when allocation or commit throws during construction.
    std::shared_ptr<tsg::SharedCacheCharge> charge_;
    std::unique_ptr<unsigned char[], deleter> data_;
    size_t workers_, block_, residency_, stride_;
public:
    explicit file_warm_arena(file_warm_budget_plan & plan)
        : workers_(plan.workers), block_(plan.block), residency_(plan.residency), stride_(plan.stride) {
        if (!plan.accounted) return; // Preserve the unscoped lazy path.
        charge_ = std::move(plan.reservation);
        if (!charge_ || plan.skipped || !plan.aggregate)
            throw std::invalid_argument("file warm arena requires an admitted plan");
        data_.reset(Allocator::allocate(plan.aggregate));
        if (!data_) throw std::bad_alloc();
        if (!charge_->commit(plan.aggregate)) throw std::runtime_error("file warm workspace commit refused");
    }
    file_warm_arena(const file_warm_arena &) = delete;
    file_warm_arena & operator=(const file_warm_arena &) = delete;
    unsigned char * payload(size_t worker) {
        if (worker >= workers_) throw std::out_of_range("file warm worker slice");
        return data_ ? data_.get() + worker * stride_ : nullptr;
    }
    file_warm_scratch scratch(size_t worker) {
        auto p = payload(worker);
        return p ? file_warm_scratch(p + block_, residency_) : file_warm_scratch();
    }
};
} // namespace tsg_dsv4
