// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <chrono>
#include <cstddef>

namespace tsg {
// Per-cache-entry feedback, protected by the expert cache's existing mutex.
// This is a scheduling hint only: every selected byte is still uploaded and no
// additional expert, pinned copy or persistent host buffer is retained.
struct ExpertPrefetchPolicy
{
    using Clock = std::chrono::steady_clock;
    bool read_ahead = true;

    static bool eligible(std::size_t misses, std::size_t bytes)
    {
        return misses >= 2 && bytes >= (std::size_t(4) << 20);
    }

    bool should_prefetch(std::size_t misses, std::size_t bytes) const
    {
        return read_ahead && eligible(misses, bytes);
    }

    void observe_read(std::size_t bytes, Clock::duration elapsed)
    {
        // Once page touches are cheap, avoid the worker dispatch on hot rows.
        // Both a latency floor and a byte-scaled allowance avoid judging a
        // large resident span by the same threshold as a small one.
        const double allowance = 0.0005 + static_cast<double>(bytes) / (16.0 * (1ull << 30));
        read_ahead = std::chrono::duration<double>(elapsed).count() > allowance;
    }

    void observe_upload(std::size_t bytes, Clock::duration elapsed)
    {
        // Pageable uploads return after source staging. A newly expensive
        // source read re-arms the hint (e.g. a different request or RAM pressure).
        // Do not synchronize the GPU just to collect this signal.
        const double allowance = 0.001 + static_cast<double>(bytes) / (4.0 * (1ull << 30));
        if (std::chrono::duration<double>(elapsed).count() > allowance) read_ahead = true;
    }
};
}
