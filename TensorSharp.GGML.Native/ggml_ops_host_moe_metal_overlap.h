// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml.h"

namespace tsg::host_moe_metal_overlap {
enum class Reject { none, geometry, no_prefix, dependency, operation, fusion_boundary, unpinned, allocation, alias, count };
// A pending host result must be a fresh contiguous non-view F32 leaf. Returns
// the first consumer index (exclusive prefix end), or -1. Does not
// reorder nodes, add outputs, change allocation, or expand another graph.
int prefix_end(ggml_cgraph* graph, int begin, int limit, const ggml_tensor* pending, Reject&);
bool safe_allocations(ggml_cgraph* graph, int begin, int end, const ggml_tensor* pending,
    const ggml_tensor* const* inputs, int input_count, Reject&);
int mode(); // startup opt-in: 0 ordinary, 1 async overlap, 2 serial split

// Backend-independent lifetime primitive, exercised with fake drains in tests.
// Arm BEFORE submission: a failed submission can still have queued some work.
class Drain {
public:
    using Callback = bool (*)(void*) noexcept;
    Drain(Callback callback, void* state) : callback_(callback), state_(state) {}
    Drain(const Drain&) = delete;
    Drain& operator=(const Drain&) = delete;
    ~Drain() { finish(); }
    void arm() { pending_ = true; }
    bool finish() noexcept {
        if (pending_) { pending_ = false; failed_ = !callback_(state_) || failed_; }
        return !failed_;
    }
private:
    Callback callback_;
    void* state_;
    bool pending_ = false, failed_ = false;
};

enum class Count { considered, ineligible, rejected, eligible, async_submits, serial_submits,
    prefix_nodes, submit_failures, drain_failures, fast_fallbacks, cuts, count };
void count(Count, unsigned long long amount = 1);
void reject(Reject);
const char* reject_name(Reject);
void time(double submit_ms, double drain_ms, double serial_ms);
void report();
} // namespace tsg::host_moe_metal_overlap
