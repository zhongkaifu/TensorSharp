// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <mutex>

namespace tsg
{
    // The owner mutex also protects cache publication/removal and the counters.
    // Reserve before calling the backend, without holding the mutex through the
    // allocation. A pending token must outlive its provisional physical buffer.
    class CacheAllocationReservation
    {
        std::mutex& mutex_;
        std::int64_t& reserved_;
        std::int64_t& committed_;
        const std::int64_t& limit_;
        std::int64_t bytes_ = 0;
    public:
        CacheAllocationReservation(std::mutex& mutex, std::int64_t& reserved,
            std::int64_t& committed, const std::int64_t& limit, std::size_t bytes)
            : mutex_(mutex), reserved_(reserved), committed_(committed), limit_(limit)
        {
            std::lock_guard<std::mutex> lock(mutex_);
            if (bytes == 0 || !fits_locked(bytes)) return;
            bytes_ = static_cast<std::int64_t>(bytes);
            reserved_ += bytes_;
        }
        CacheAllocationReservation(const CacheAllocationReservation&) = delete;
        CacheAllocationReservation& operator=(const CacheAllocationReservation&) = delete;
        ~CacheAllocationReservation()
        {
            if (bytes_ == 0) return;
            std::lock_guard<std::mutex> lock(mutex_);
            reserved_ -= bytes_;
        }
        explicit operator bool() const { return bytes_ != 0; }

        // Caller holds mutex_. Recheck the physical size returned by the backend;
        // reservations by other threads remain protected during this conversion.
        bool fits_locked(std::size_t actual) const
        {
            const auto maximum = std::numeric_limits<std::int64_t>::max();
            if (actual > static_cast<std::size_t>(maximum)) return false;
            const std::int64_t ceiling = limit_ > 0 ? limit_ : maximum;
            const std::int64_t others = reserved_ - bytes_;
            return committed_ <= ceiling && others <= ceiling - committed_
                && static_cast<std::int64_t>(actual) <= ceiling - committed_ - others;
        }
        void publish_locked(std::size_t actual)
        {
            assert(bytes_ != 0 && fits_locked(actual));
            reserved_ -= bytes_;
            committed_ += static_cast<std::int64_t>(actual);
            bytes_ = 0;
        }
    };
}
