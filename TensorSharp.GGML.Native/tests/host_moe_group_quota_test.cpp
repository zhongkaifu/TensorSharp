// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_group_quota.h"
#include <cstdio>
#include <cstdlib>

int main() {
    int checks = 0;
    const auto require = [&](bool value) {
        ++checks;
        if (!value) { std::fprintf(stderr, "FAIL group quota check %d\n", checks); std::exit(1); }
    };
    const auto key = [](int type) { return tsg::HostMoeGroupKey{{type, type, 18, 4096, 2048, 8, 288, 12960}, 0}; };
    const std::vector<tsg::HostMoeGroupMinimum> groups{{key(16), 300}, {key(19), 200}, {key(23), 400}};
    tsg::HostMoeGroupQuotaPlan plan(groups);
    require(plan.minimum() == 900 && plan.size() == 3);
    for (std::size_t budget = 900; budget < 1001; ++budget) {
        std::size_t sum = 0;
        for (const auto& group : groups) {
            const auto quota = plan.quota(group.key, budget);
            require(quota >= group.bytes); sum += quota;
        }
        require(sum == budget);
    }
    require(plan.quota(key(16), 902) == 301 && plan.quota(key(19), 902) == 201 && plan.quota(key(23), 902) == 400);
    auto reversed = groups; std::reverse(reversed.begin(), reversed.end());
    tsg::HostMoeGroupQuotaPlan reverse_plan(reversed);
    for (const auto& group : groups) require(plan.quota(group.key, 902) == reverse_plan.quota(group.key, 902));
    for (auto budget : {std::size_t(0), std::size_t(899)})
        for (const auto& group : groups) require(plan.quota(group.key, budget) == 0);
    require(plan.quota(key(24), 1000) == 0);
    auto changed = key(16); changed.values[3]++;
    require(plan.quota(changed, 1000) == 0);
    changed = key(16); changed.clamp_bits = 1;
    require(plan.quota(changed, 1000) == 0);
    const auto invalid = [&](std::vector<tsg::HostMoeGroupMinimum> candidate) {
        bool refused = false;
        try { tsg::HostMoeGroupQuotaPlan rejected(std::move(candidate)); }
        catch (const std::exception&) { refused = true; }
        require(refused);
        require(plan.minimum() == 900 && plan.quota(key(16), 902) == 301);
    };
    invalid({});
    invalid({groups[0], groups[0]});
    invalid({{key(16), 0}});
    changed = key(16); changed.values[5] = 289;
    invalid({{changed, 1}});
    changed = key(16); changed.clamp_bits = 0x7fc00000;
    invalid({{changed, 1}});
    invalid({{key(16), static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)())}, {key(19), 1}});
    // Actual historical model group minima: their sum fits 7000 MiB although
    // three copies of the heaviest group do not. This does not replace a fresh
    // allocator plan for the current split-ownership implementation.
    constexpr std::size_t cap = std::size_t(7000) << 20;
    tsg::HostMoeGroupQuotaPlan actual({{key(16), 2186236672ULL}, {key(19), 1884246784ULL}, {key(23), 2544849664ULL}});
    require(actual.minimum() < cap && 3ULL * 2544849664ULL > cap);
    require(actual.quota(key(16), cap) + actual.quota(key(19), cap) + actual.quota(key(23), cap) == cap);
    std::printf("PASS %d pure measured-group quota checks; no backend or allocation qualification\n", checks);
}
