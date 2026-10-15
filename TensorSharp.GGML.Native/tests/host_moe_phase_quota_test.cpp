// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_phase_quota.h"
#include <cstdio>
#include <cstdlib>

int main() {
    int checks = 0;
    const auto require = [&](bool value) {
        if (++checks && !value) { std::fprintf(stderr, "FAIL phase quota check %d\n", checks); std::exit(1); }
    };
    const auto key = [](int type) { return tsg::HostMoeGroupKey{{type, type, 18, 4096, 2048, 8, 288, 12960}, 0}; };
    const std::vector<tsg::HostMoePhaseGroupMinimum> groups{{key(16), 30, 300}, {key(19), 20, 200}, {key(23), 40, 400}};
    tsg::HostMoePhaseQuotaPlan plan(groups);
    require(plan.narrow_total() == 90 && plan.minimum() == 450 && plan.size() == 3);
    for (std::size_t budget = 450; budget < 551; ++budget) {
        std::size_t sum = 0;
        for (const auto& group : groups) {
            const auto narrow = plan.quota(group.key, budget, false), wide = plan.quota(group.key, budget, true);
            require(narrow >= group.narrow_bytes && wide >= group.wide_bytes);
            require(wide + plan.narrow_total() - group.narrow_bytes == budget);
            sum += narrow;
        }
        require(sum == budget);
    }
    auto reversed = groups; std::reverse(reversed.begin(), reversed.end());
    tsg::HostMoePhaseQuotaPlan reverse_plan(reversed);
    for (const auto& group : groups) for (bool wide : {false, true})
        require(plan.quota(group.key, 451, wide) == reverse_plan.quota(group.key, 451, wide));
    for (auto budget : {std::size_t(0), std::size_t(449)}) for (const auto& group : groups)
        require(!plan.admits(budget) && plan.quota(group.key, budget, false) == 0 && plan.quota(group.key, budget, true) == 0);
    require(plan.quota(key(24), 500, true) == 0);
    require(plan.cold_quota(key(19), 450, 400, false) == 50);
    require(plan.cold_quota(key(19), 450, 440, false) == 20);
    require(plan.cold_quota(key(19), 450, 451, false) == 0);
    require(plan.cold_quota(key(19), 450, 400, true) == 200);
    require(plan.cold_quota(key(16), 450, 0, true) == 300);
    for (std::size_t budget = 900; budget < 1001; ++budget) {
        std::size_t total_wide = 0, total_limits = 0;
        require(plan.all_wide_fit(budget, total_wide) && total_wide == 900);
        for (const auto& group : groups) {
            const auto cold = plan.cold_quota(group.key, budget, 0, true);
            require(cold >= group.wide_bytes && cold <= plan.quota(group.key, budget, true));
            total_limits += cold;
        }
        require(total_limits == budget);
    }
    std::size_t no_total = 123;
    require(!plan.all_wide_fit(899, no_total) && no_total == 0);
    // All-wide sum overflows signed size but one phase remains valid.
    const auto half_max = static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)()) / 2;
    tsg::HostMoePhaseQuotaPlan huge({{key(16), 1, half_max + 1}, {key(19), 1, half_max + 1}});
    require(huge.admits(half_max + 2) && !huge.all_wide_fit(half_max + 2, no_total));
    require(huge.cold_quota(key(16), half_max + 2, 0, true) == half_max + 1);
    auto changed = key(16); changed.values[3]++;
    require(plan.quota(changed, 500, false) == 0);
    const auto invalid = [&](std::vector<tsg::HostMoePhaseGroupMinimum> candidate) {
        bool refused = false;
        try { tsg::HostMoePhaseQuotaPlan rejected(std::move(candidate)); }
        catch (const std::exception&) { refused = true; }
        require(refused && plan.minimum() == 450);
    };
    invalid({}); invalid({groups[0], groups[0]});
    invalid({{key(16), 0, 1}}); invalid({{key(16), 2, 1}});
    changed = key(16); changed.values[5] = 289;
    invalid({{changed, 1, 1}});
    const auto maximum = static_cast<std::size_t>((std::numeric_limits<std::int64_t>::max)());
    invalid({{key(16), maximum, maximum}, {key(19), 1, 1}});
    invalid({{key(16), 1, maximum}, {key(19), 1, 1}}); // Phase sum overflow, narrow sum does fit.
    tsg::HostMoePhaseQuotaPlan narrow_only({{key(16), 30, 30}, {key(19), 20, 20}});
    require(narrow_only.minimum() == 50 && narrow_only.quota(key(16), 50, true) == 30);
    tsg::HostMoePhaseQuotaPlan single({{key(16), 30, 300}});
    require(single.minimum() == 300 && single.quota(key(16), 301, false) == 301);
    // CPU ownership state machine: no credit is borrowed until a physical owner
    // has actually been removed. This supplements, not replaces, the CUDA ledger.
    std::array<std::size_t, 3> live{300, 20, 40};
    auto total = [&] { return live[0] + live[1] + live[2]; };
    const auto saved = live;
    require(!plan.admits(449) && live == saved); // Sub-floor resize is nonmutating.
    live[0] = 0; // Synchronous physical retirement before B is admitted.
    require(total() + 200 <= plan.minimum()); live[1] = 200;
    live[2] = 0; live[1] = 0; // Retire B before C's wide allocation.
    require(total() + 400 <= plan.minimum()); live[2] = 400;
    live = {}; require(total() == 0); // Cancellation/trim.
    live = {30, 20, 40}; require(total() == plan.narrow_total()); // N3 resume.
    std::printf("PASS %d pure phase-plan checks; real shared tickets and CUDA remain separate qualification\n", checks);
}
