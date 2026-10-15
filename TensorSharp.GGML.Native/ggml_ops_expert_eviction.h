// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstddef>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tsg {
// Metadata only: preserve frequent experts without retaining extra weight bytes.
// Counts age at a capacity-dependent epoch, so earlier requests cannot make a
// popular expert immortal. Equal frequencies retain the existing LRU ordering.
class ExpertEvictionPolicy {
    std::vector<std::uint16_t> frequency_;
    unsigned epoch_ = 0, rows_ = 0;
    static unsigned capacity_epoch(int capacity) {
        unsigned epoch = 32;
        while (epoch < 1024 && epoch / 2 < static_cast<unsigned>(capacity)) epoch *= 2;
        return epoch;
    }
public:
    void configure(int experts, int capacity, bool enabled) {
        if (experts <= 0 || capacity <= 0 || capacity > experts)
            throw std::invalid_argument("Invalid expert eviction geometry");
        frequency_.clear();
        epoch_ = rows_ = 0;
        if (!enabled) return;
        frequency_.resize(experts, 0);
        // Power-of-two epoch in [32, 1024], about twice the number of slots.
        // Small caches forget old routing sooner; no model name is involved.
        epoch_ = capacity_epoch(capacity);
    }
    // Expansion keeps both popularity and the current aging position. Only
    // the longer capacity-dependent window changes; it cannot make routing
    // history immortal or reset an old hot expert to zero frequency.
    void grow_capacity(int capacity) {
        if (capacity <= 0 || (epoch_ && static_cast<std::size_t>(capacity) > frequency_.size()))
            throw std::invalid_argument("Invalid expanded eviction capacity");
        if (!epoch_) return;
        const auto next = capacity_epoch(capacity);
        if (next < epoch_) throw std::invalid_argument("Eviction capacity expansion shortened its epoch");
        epoch_ = next;
    }
    unsigned epoch() const { return epoch_; }
    void observe(const std::int32_t* ids, int count) {
        if (!epoch_) return;
        if (++rows_ == epoch_) {
            for (auto& f : frequency_) f >>= 1;
            rows_ = 0;
        }
        for (int i = 0; i < count; ++i) {
            auto& f = frequency_.at(ids[i]);
            if (f != UINT16_MAX) ++f;
        }
    }
    bool prefer(int candidate, std::uint64_t candidate_age,
                int incumbent, std::uint64_t incumbent_age) const {
        if (epoch_) {
            // Failed or never-filled slots must precede valid payloads.
            const int a = candidate < 0 ? -1 : frequency_[candidate];
            const int b = incumbent < 0 ? -1 : frequency_[incumbent];
            if (a != b) return a < b;
        }
        return candidate_age < incumbent_age;
    }
};
}
