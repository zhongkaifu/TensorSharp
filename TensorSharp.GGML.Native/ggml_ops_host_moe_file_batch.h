// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_file_source.h"
#include "ggml_ops_shared_cache_budget.h"
#include <array>
#include <cmath>
#include <cstring>
#include <limits>

namespace tsg {
inline std::size_t host_moe_checked_add(std::size_t a, std::size_t b) {
    if (a > (std::numeric_limits<std::size_t>::max)() - b) throw std::overflow_error("Host MoE byte sum overflow");
    return a + b;
}
inline std::size_t host_moe_checked_mul(std::size_t a, std::size_t b) {
    if (b && a > (std::numeric_limits<std::size_t>::max)() / b) throw std::overflow_error("Host MoE byte product overflow");
    return a * b;
}
inline std::size_t host_moe_align(std::size_t bytes) { return host_moe_checked_add(bytes, 63) & ~std::size_t(63); }

// One serial layer's raw quantized payload, not a resident expert cache. Credit
// covers the exact allocation including alignment slack. Reuse never retains
// source pointers or expert identities. Maintenance must quiesce its caller.
class HostMoeCpuArena {
    std::shared_ptr<SharedCacheCharge> charge_;
    std::unique_ptr<std::uint8_t[]> allocation_;
    std::uint8_t* data_ = nullptr;
    std::size_t capacity_ = 0;
public:
    void clear() noexcept { allocation_.reset(); data_ = nullptr; capacity_ = 0; charge_.reset(); }
    ~HostMoeCpuArena() { clear(); }
    void reserve(std::size_t bytes) {
        if (bytes <= capacity_) return;
        clear(); // free before refund; growth does not require both capacities
        const auto actual = host_moe_checked_add(bytes, 63);
        auto charge = SharedCacheCharge::reserve(0, 3, actual);
        if (!charge || !charge->is_accounted()) throw std::runtime_error("Host MoE CPU staging budget refused");
        std::unique_ptr<std::uint8_t[]> allocation(new std::uint8_t[actual]);
        if (!charge->commit(actual)) throw std::runtime_error("Host MoE CPU staging commit refused");
        data_ = reinterpret_cast<std::uint8_t*>((reinterpret_cast<std::uintptr_t>(allocation.get()) + 63) & ~std::uintptr_t(63));
        capacity_ = bytes; charge_ = std::move(charge); allocation_ = std::move(allocation);
    }
    std::uint8_t* data() const noexcept { return data_; }
    std::size_t capacity() const noexcept { return capacity_; }
};

struct HostMoeFileBatch {
    std::array<std::shared_ptr<ExpertFileSource>, 3> sources;
    std::array<std::size_t, 3> stride{}, offset{}, bytes{}, extent{};
    std::vector<std::int32_t> experts, ids;
    std::size_t payload = 0;
    bool duplicate_routes = false;
    void plan(const std::int32_t* original, const float* routes, int rows, int used, int count,
        const std::array<std::size_t, 3>& totals) {
        if (!original || !routes || rows <= 0 || used <= 0 || count <= 0 || used > count)
            throw std::invalid_argument("Invalid host MoE file batch");
        const auto n = host_moe_checked_mul(static_cast<std::size_t>(rows), used);
        std::vector<std::int32_t> remap(count, -1), last_row(count, -1);
        duplicate_routes = false;
        ids.resize(n); experts.clear();
        // Validate every route before I/O, allocation of payload, or output.
        for (std::size_t i = 0; i < n; ++i) {
            const int id = original[i];
            if (id < 0 || id >= count || !std::isfinite(routes[i]))
                throw std::invalid_argument("Invalid host MoE expert ID or nonfinite route");
            const auto row = static_cast<std::int32_t>(i / used);
            if (last_row[id] == row) duplicate_routes = true;
            last_row[id] = row;
            if (remap[id] < 0) { remap[id] = static_cast<int>(experts.size()); experts.push_back(id); }
        }
        // Compact physical expert order lets adjacent experts share one read.
        // Only the weight axis changes: route slots and their reduction order
        // stay exactly as supplied by the caller, including duplicate IDs.
        std::sort(experts.begin(), experts.end());
        for (std::size_t slot = 0; slot < experts.size(); ++slot)
            remap[experts[slot]] = static_cast<std::int32_t>(slot);
        for (std::size_t i = 0; i < n; ++i) ids[i] = remap[original[i]];
        payload = 0;
        for (int part = 0; part < 3; ++part) {
            if (!totals[part] || totals[part] % count) throw std::invalid_argument("Invalid expert stack extent");
            extent[part] = totals[part];
            stride[part] = totals[part] / count;
            bytes[part] = host_moe_checked_mul(stride[part], experts.size());
            offset[part] = payload;
            payload = host_moe_checked_add(payload, host_moe_align(bytes[part]));
        }
    }
    // No gaps, projection boundaries, or alignment padding enter a read. The
    // source and compact destination are both contiguous within each run.
    template<class Visit> void for_each_read_range(Visit&& visit) const {
        for (int part = 0; part < 3; ++part) {
            for (std::size_t first = 0; first < experts.size();) {
                std::size_t end = first + 1;
                while (end < experts.size()
                    && static_cast<std::int64_t>(experts[end]) == static_cast<std::int64_t>(experts[end - 1]) + 1)
                    ++end;
                const auto source = host_moe_checked_mul(static_cast<std::size_t>(experts[first]), stride[part]);
                const auto destination = host_moe_checked_mul(first, stride[part]);
                const auto length = host_moe_checked_mul(end - first, stride[part]);
                if (source > extent[part] || length > extent[part] - source
                    || destination > bytes[part] || length > bytes[part] - destination)
                    throw std::runtime_error("Host MoE merged expert read exceeds its projection");
                const auto target = host_moe_checked_add(offset[part], destination);
                if (target > payload || length > payload - target)
                    throw std::runtime_error("Host MoE merged expert read exceeds its arena");
                visit(part, source, target, length);
                first = end;
            }
        }
    }
    void read(HostMoeCpuArena& arena) const {
        for (int part = 0; part < 3; ++part)
            if (!sources[part] || sources[part]->bytes() != extent[part])
                throw std::runtime_error("Host MoE expert source extent changed");
        arena.reserve(payload);
        std::vector<ExpertReadTask> tasks;
        tasks.reserve(host_moe_checked_mul(experts.size(), 3));
        for_each_read_range([&](int part, std::size_t source, std::size_t destination, std::size_t length) {
            tasks.push_back({sources[part].get(), source, arena.data() + destination, length});
        });
        read_expert_files(tasks, [](std::size_t) {});
    }
};

// 0: ordinary path, 1: all three exact sources, -1: invalid partial registration.
int host_moe_file_batch_sources(const void* const* pointers, const std::int64_t* bytes,
    std::array<std::shared_ptr<ExpertFileSource>, 3>& result);
void host_moe_cpu_staging_release();
} // namespace tsg
