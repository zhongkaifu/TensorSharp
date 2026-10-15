// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_shared_cache_budget.h"
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace tsg_glm {

inline size_t upload_add(size_t a, size_t b) {
    if (b > (std::numeric_limits<size_t>::max)() - a)
        throw std::overflow_error("GLM upload payload sum overflow");
    return a + b;
}
inline size_t upload_multiply(size_t a, size_t b) {
    if (a && b > (std::numeric_limits<size_t>::max)() / a)
        throw std::overflow_error("GLM upload payload product overflow");
    return a * b;
}

inline size_t upload_option(const char * text, size_t fallback, size_t unit, size_t maximum) {
    if (!unit) throw std::invalid_argument("GLM upload option unit must be positive");
    if (!text || !*text) return fallback;
    size_t value = 0;
    for (const char * p = text; *p; ++p) {
        if (*p < '0' || *p > '9') throw std::invalid_argument("GLM upload option must be a nonnegative integer");
        const size_t digit = size_t(*p - '0');
        if (value > maximum / unit / 10 ||
            (value == maximum / unit / 10 && digit > maximum / unit % 10))
            throw std::overflow_error("GLM upload option exceeds supported byte/thread range");
        value = value * 10 + digit;
    }
    return value ? value * unit : fallback;
}

inline void validate_upload_range(size_t offset, size_t bytes, size_t stride, size_t strip) {
    const uint64_t maximum = uint64_t((std::numeric_limits<int64_t>::max)());
    if (!bytes || uint64_t(offset) > maximum) throw std::invalid_argument("GLM upload has invalid file range");
    if (!stride) {
        if (uint64_t(bytes) > maximum - offset) throw std::overflow_error("GLM upload file range overflow");
    } else {
        if (!strip || stride < strip || bytes % strip) throw std::invalid_argument("GLM upload has invalid TP row geometry");
        if (uint64_t(strip) > maximum - offset || uint64_t(bytes / strip - 1) > (maximum - offset - strip) / stride)
            throw std::overflow_error("GLM upload strided file range overflow");
    }
}

struct upload_plan {
    size_t workers = 0, chunk = 0, slab_limit = 0;
    size_t staging = 0, slab = 0, worker_bytes = 0, aggregate = 0;
    size_t attempts = 0, refusals = 0;
    bool accounted = false;
    // Held until all workers have joined, including on the unaccounted path:
    // a host-budget attach cannot adopt an in-progress unbudgeted upload.
    std::shared_ptr<tsg::SharedCacheCharge> reservation;
};

class upload_budget_refused : public std::runtime_error {
public:
    const size_t refusals, minimum_payload;
    upload_budget_refused(size_t count, size_t bytes)
        : std::runtime_error("GLM upload cannot reserve its minimum useful host workspace before upload"),
          refusals(count), minimum_payload(bytes) {}
};

inline size_t upload_segment_bytes(size_t remaining, size_t chunk, size_t stride, size_t strip) {
    if (!remaining || !chunk) throw std::invalid_argument("GLM upload has zero job/chunk bytes");
    if (!stride) return (std::min)(remaining, chunk);
    if (!strip || stride < strip || remaining % strip)
        throw std::invalid_argument("GLM upload has invalid TP row geometry");
    const size_t rows = (std::min)(remaining / strip, (std::max)(size_t(1), chunk / strip));
    return upload_multiply(rows, strip);
}

inline size_t upload_slab_capacity(size_t bytes, size_t stride, size_t strip, size_t limit) {
    if (!stride) return 0;
    if (!bytes || !strip || stride < strip || bytes % strip)
        throw std::invalid_argument("GLM upload has invalid TP slab geometry");
    limit = (std::max)(limit, strip);
    const size_t rows = (std::min)(bytes / strip, size_t(1) + (limit - strip) / stride);
    return upload_add(upload_multiply(rows - 1, stride), strip);
}

// Jobs expose len/src_row/row_len. No file reads, physical payload allocation,
// workers or tensor writes occur here. Refused proposals are negotiation only,
// not retries of a partially uploaded model. The first successful atomic ticket
// protects the entire workspace even if other budget users subsequently race.
template<class Jobs>
upload_plan plan_upload(const Jobs & jobs, size_t requested_workers, size_t requested_chunk) {
    upload_plan plan;
    if (jobs.empty()) return plan;
    if (!requested_workers || !requested_chunk)
        throw std::invalid_argument("GLM upload requires positive workers and chunk");
    plan.workers = (std::min)(requested_workers, jobs.size());
    plan.chunk = requested_chunk;
    size_t max_job = 0, max_strip = 0;
    for (const auto & job : jobs) {
        (void) upload_segment_bytes(job.len, plan.chunk, job.src_row, job.row_len);
        max_job = (std::max)(max_job, job.len);
        if (job.src_row) max_strip = (std::max)(max_strip, job.row_len);
    }
    // Keep a bounded useful I/O unit, rather than admitting an arbitrarily tiny
    // quota by issuing one-byte disk/upload operations. This is a portable I/O
    // policy, not a claim about OS page size or a fixed memory-budget allowance.
    const size_t minimum_chunk = (std::min)(requested_chunk,
        (std::min)(max_job, (std::max)(size_t(4096), max_strip)));
    for (;;) {
        plan.staging = plan.slab = 0;
        plan.slab_limit = (std::min)(size_t(1024 * 1024), plan.chunk);
        for (const auto & job : jobs) {
            const size_t bytes = upload_segment_bytes(job.len, plan.chunk, job.src_row, job.row_len);
            plan.staging = (std::max)(plan.staging, bytes);
            plan.slab = (std::max)(plan.slab,
                upload_slab_capacity(bytes, job.src_row, job.row_len, plan.slab_limit));
        }
        plan.worker_bytes = upload_add(plan.staging, plan.slab);
        plan.aggregate = upload_multiply(plan.workers, plan.worker_bytes);
        if (uint64_t(plan.aggregate) > uint64_t((std::numeric_limits<int64_t>::max)()))
            throw std::overflow_error("GLM upload aggregate exceeds shared-budget integer range");
        ++plan.attempts;
        plan.reservation = tsg::SharedCacheCharge::reserve(0, 3, plan.aggregate);
        if (plan.reservation) {
            plan.accounted = plan.reservation->is_accounted();
            if (!plan.accounted) {
                // The legacy lazy path consumes whole jobs and keeps its 1MiB
                // TP read window. Report its actual capacity upper bound, not
                // the smaller proposal used by the accounted segmentation.
                plan.staging = plan.slab = 0;
                plan.slab_limit = size_t(1024 * 1024);
                for (const auto & job : jobs) {
                    plan.staging = (std::max)(plan.staging, job.len);
                    plan.slab = (std::max)(plan.slab,
                        upload_slab_capacity(job.len, job.src_row, job.row_len, plan.slab_limit));
                }
                plan.worker_bytes = upload_add(plan.staging, plan.slab);
                plan.aggregate = upload_multiply(plan.workers, plan.worker_bytes);
            }
            return plan;
        }
        ++plan.refusals;
        if (plan.workers > 1) {
            plan.workers = plan.workers / 2 + plan.workers % 2;
        } else if (plan.chunk > minimum_chunk) {
            plan.chunk = (std::max)(minimum_chunk, (std::min)(max_job, plan.chunk / 2));
        } else {
            throw upload_budget_refused(plan.refusals, plan.aggregate);
        }
    }
}
} // namespace tsg_glm
