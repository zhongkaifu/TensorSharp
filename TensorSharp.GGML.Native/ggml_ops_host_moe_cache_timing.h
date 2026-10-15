// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <array>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <limits>

namespace tsg {

// Diagnostic host wall intervals only. Different stages may nest/overlap;
// neither enqueue time nor a backend wait is a CUDA kernel duration.
enum class HostMoeCacheStage : std::size_t {
    plan, allocation, wide_replace, file_pipeline, upload_enqueue, upload_wait,
    input_bridge, compute, output_bridge, count
};

struct HostMoeCacheTiming {
    static constexpr auto stages = static_cast<std::size_t>(HostMoeCacheStage::count);
    std::array<std::uint64_t, stages> nanoseconds{}, calls{};
    std::array<unsigned, stages> depth{};
    std::uint64_t file_bytes = 0, file_windows = 0, graph_submissions = 0;

    static bool enabled(const char* value) noexcept {
        return value && std::strcmp(value, "1") == 0;
    }
    static void add(std::uint64_t& target, std::uint64_t amount) noexcept {
        const auto maximum = (std::numeric_limits<std::uint64_t>::max)();
        target = amount > maximum - target ? maximum : target + amount;
    }
    void record(HostMoeCacheStage stage, std::int64_t elapsed_ns) noexcept {
        const auto index = static_cast<std::size_t>(stage);
        add(calls[index], 1);
        if (elapsed_ns > 0) add(nanoseconds[index], static_cast<std::uint64_t>(elapsed_ns));
    }
    double milliseconds(HostMoeCacheStage stage) const noexcept {
        return nanoseconds[static_cast<std::size_t>(stage)] / 1e6;
    }
};

// A null owner performs no clock read. Recursive planning regions of the same
// stage count only their outer interval; different stages stay explicitly
// nested rather than being presented as an additive latency breakdown.
template<class Clock = std::chrono::steady_clock>
class HostMoeCacheStageTimer {
    HostMoeCacheTiming* owner_;
    HostMoeCacheStage stage_;
    bool outer_ = false;
    typename Clock::time_point start_{};
public:
    HostMoeCacheStageTimer(HostMoeCacheTiming* owner, HostMoeCacheStage stage) noexcept
        : owner_(owner), stage_(stage) {
        if (owner_) {
            auto& depth = owner_->depth[static_cast<std::size_t>(stage_)];
            outer_ = depth++ == 0;
            if (outer_) start_ = Clock::now();
        }
    }
    ~HostMoeCacheStageTimer() noexcept {
        if (!owner_) return;
        --owner_->depth[static_cast<std::size_t>(stage_)];
        if (outer_) owner_->record(stage_,
            std::chrono::duration_cast<std::chrono::nanoseconds>(Clock::now() - start_).count());
    }
    HostMoeCacheStageTimer(const HostMoeCacheStageTimer&) = delete;
    HostMoeCacheStageTimer& operator=(const HostMoeCacheStageTimer&) = delete;
};

} // namespace tsg
