// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tsg {
// Exact compact-table identity, excluding immutable source addresses: pooled
// tables deliberately reuse their storage across layers with this geometry.
struct HostMoeGroupKey {
    // gate/up/down types, hidden, FF width, top-k, expert count, pool extent.
    std::array<int, 8> values{};
    std::uint32_t clamp_bits = 0;
    bool operator==(const HostMoeGroupKey& other) const noexcept {
        return values == other.values && clamp_bits == other.clamp_bits;
    }
    bool operator<(const HostMoeGroupKey& other) const noexcept {
        return values != other.values ? values < other.values : clamp_bits < other.clamp_bits;
    }
    bool valid() const noexcept {
        return values[0] >= 0 && values[1] >= 0 && values[2] >= 0
            && values[3] > 0 && values[4] > 0 && values[5] > 0
            && values[5] <= values[6] && values[6] <= values[7]
            && (clamp_bits & 0x7f800000u) != 0x7f800000u;
    }
};
struct HostMoeGroupMinimum { HostMoeGroupKey key; std::size_t bytes = 0; };

// Pure metadata. Each group's measured floor is protected first, then surplus
// bytes are divided evenly in canonical key order. No multiplicative weights,
// floating-point rounding, physical allocation or shared-credit mutation.
class HostMoeGroupQuotaPlan {
    std::vector<HostMoeGroupMinimum> groups_;
    std::size_t minimum_ = 0;
public:
    explicit HostMoeGroupQuotaPlan(std::vector<HostMoeGroupMinimum> groups) : groups_(std::move(groups)) {
        if (groups_.empty()) throw std::invalid_argument("GLM group plan is empty");
        std::sort(groups_.begin(), groups_.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
        const auto limit = (std::min)(static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()),
            static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()));
        for (std::size_t index = 0; index < groups_.size(); ++index) {
            const auto& group = groups_[index];
            if (!group.key.valid() || !group.bytes || (index && groups_[index - 1].key == group.key))
                throw std::invalid_argument("GLM group plan has invalid geometry, a zero floor or duplicate identity");
            if (group.bytes > limit || minimum_ > limit - group.bytes)
                throw std::overflow_error("GLM group plan exceeds signed shared-budget range");
            minimum_ += group.bytes;
        }
    }
    std::size_t minimum() const noexcept { return minimum_; }
    std::size_t size() const noexcept { return groups_.size(); }
    const std::vector<HostMoeGroupMinimum>& groups() const noexcept { return groups_; }
    std::size_t quota(const HostMoeGroupKey& key, std::size_t budget) const noexcept {
        if (budget < minimum_ || budget > static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)())) return 0;
        for (std::size_t index = 0; index < groups_.size(); ++index) if (groups_[index].key == key) {
            const auto surplus = budget - minimum_;
            return groups_[index].bytes + surplus / groups_.size() + (index < surplus % groups_.size() ? 1 : 0);
        }
        return 0; // Unknown groups never inherit another group's unused credit.
    }
};
} // namespace tsg
