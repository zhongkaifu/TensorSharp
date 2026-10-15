// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include "ggml_ops_host_moe_group_quota.h"

namespace tsg {
struct HostMoePhaseGroupMinimum {
    HostMoeGroupKey key;
    std::size_t narrow_bytes = 0, wide_bytes = 0;
};

// Serialized helper phases may reclaim inactive groups. The minimum protects
// one whole wide union plus every other group's narrow decode floor. It is a
// feasibility bound, not a reservation or permission to refund a live owner.
class HostMoePhaseQuotaPlan {
    std::vector<HostMoePhaseGroupMinimum> groups_;
    std::size_t narrow_total_ = 0, minimum_ = 0;
public:
    explicit HostMoePhaseQuotaPlan(std::vector<HostMoePhaseGroupMinimum> groups)
        : groups_(std::move(groups)) {
        if (groups_.empty()) throw std::invalid_argument("GLM phase plan is empty");
        std::sort(groups_.begin(), groups_.end(), [](const auto& a, const auto& b) { return a.key < b.key; });
        const auto limit = (std::min)(static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)()),
            static_cast<std::uint64_t>((std::numeric_limits<std::size_t>::max)()));
        for (std::size_t i = 0; i < groups_.size(); ++i) {
            const auto& group = groups_[i];
            if (!group.key.valid() || !group.narrow_bytes || group.wide_bytes < group.narrow_bytes
                || (i && groups_[i - 1].key == group.key))
                throw std::invalid_argument("GLM phase plan has invalid geometry or floors");
            if (group.wide_bytes > limit || narrow_total_ > limit - group.narrow_bytes)
                throw std::overflow_error("GLM phase floors exceed shared-budget range");
            narrow_total_ += group.narrow_bytes;
        }
        for (const auto& group : groups_) {
            const auto others = narrow_total_ - group.narrow_bytes;
            if (others > limit - group.wide_bytes)
                throw std::overflow_error("GLM phase minimum exceeds shared-budget range");
            minimum_ = (std::max)(minimum_, others + group.wide_bytes);
        }
    }
    const std::vector<HostMoePhaseGroupMinimum>& groups() const noexcept { return groups_; }
    std::size_t size() const noexcept { return groups_.size(); }
    std::size_t minimum() const noexcept { return minimum_; }
    std::size_t narrow_total() const noexcept { return narrow_total_; }
    std::size_t floor(const HostMoeGroupKey& key, bool wide) const noexcept {
        for (const auto& group : groups_) if (group.key == key)
            return wide ? group.wide_bytes : group.narrow_bytes;
        return 0;
    }
    bool admits(std::size_t budget) const noexcept {
        return budget >= minimum_ && budget <= static_cast<std::uint64_t>((std::numeric_limits<std::int64_t>::max)());
    }
    bool all_wide_fit(std::size_t budget, std::size_t& total) const noexcept {
        total = 0;
        if (!admits(budget)) return false;
        for (const auto& group : groups_) {
            // A sum of all-wide floors may exceed signed range even though one
            // wide phase fits. Never overflow or reject the valid phase plan.
            if (group.wide_bytes > budget - total) { total = 0; return false; }
            total += group.wide_bytes;
        }
        return true;
    }
    std::size_t quota(const HostMoeGroupKey& key, std::size_t budget, bool wide) const noexcept {
        if (!admits(budget)) return 0;
        for (std::size_t i = 0; i < groups_.size(); ++i) if (groups_[i].key == key) {
            if (wide) return budget - (narrow_total_ - groups_[i].narrow_bytes);
            const auto surplus = budget - narrow_total_;
            return groups_[i].narrow_bytes + surplus / groups_.size() + (i < surplus % groups_.size() ? 1 : 0);
        }
        return 0;
    }
    std::size_t cold_quota(const HostMoeGroupKey& key, std::size_t budget,
        std::size_t already_owned, bool wide) const noexcept {
        const auto desired = quota(key, budget, wide);
        if (!desired || already_owned > budget) return 0;
        if (wide) {
            std::size_t all_wide = 0;
            if (all_wide_fit(budget, all_wide)) {
                // When all real groups can coexist, protect their measured wide
                // floors before assigning optional cold capacity. The sum of
                // these cold limits is exactly budget, so early groups cannot
                // fill most of the pool and evict otherwise affordable peers.
                for (std::size_t i = 0; i < groups_.size(); ++i) if (groups_[i].key == key) {
                    const auto spare = budget - all_wide;
                    return groups_[i].wide_bytes + spare / groups_.size()
                        + (i < spare % groups_.size() ? 1 : 0);
                }
                return 0;
            }
            // Under real phase pressure, admit the complete wide union, but do
            // not spend every borrowable byte on optional initial hot slots.
            // Existing affordable large tables are retained by entry_quota.
            return (std::max)(floor(key, true), quota(key, budget, false));
        }
        // Prefer an affordable narrow table to evicting somebody else's hot
        // wide table for optional capacity. A genuine floor deficit may still
        // reclaim inactive owners later, after exact metadata planning.
        return (std::max)(floor(key, false), (std::min)(desired, budget - already_owned));
    }
};
} // namespace tsg
