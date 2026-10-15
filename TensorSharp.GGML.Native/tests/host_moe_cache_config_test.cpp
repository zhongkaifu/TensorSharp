// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_cache_config.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <thread>
#include <vector>

namespace {
int checks = 0;
void require(bool value) { ++checks; if (!value) std::abort(); }
template<class Action> void fails(Action action) {
    bool caught = false;
    try { action(); } catch (const std::runtime_error&) { caught = true; }
    require(caught);
}
}
int main() {
    using namespace tsg;
    HostMoeCacheConfiguration config({17, 48, 8});
    require(config.fast_budget() == 17 && config.settings().layers == 48 && config.settings().max_rows == 8);
    require(config.settings().max_decode_rows == 1);
    require(!config.settings().capacity_growth && config.growth_generation() == 0);
    for (auto value : {HostMoeCacheSettings{100, 0, 64}, {100, 48, 0}, {100, 48, 1025},
            {100, 48, 32, 0}, {100, 48, 32, 9}})
        require(config.begin(value, true) == 0);
    require(config.begin({100, 12, 64}, false) == 0); // A live legacy cache is never adopted.
    const auto token = config.begin({100, 12, 1, 3}, true);
    require(token != 0 && config.owns(token) && config.fast_budget() == 100);
    require(config.settings().layers == 12 && config.settings().max_rows == 1
        && config.settings().max_decode_rows == 3);
    require(config.begin({200, 48, 128}, true) == 0);
    int trims = 0;
    std::size_t simulated_live = 90;
    auto trim = [&](std::size_t capacity, std::size_t layers) {
        ++trims; require(layers == 12);
        simulated_live = (std::min)(simulated_live, capacity);
    };
    require(!config.update(token + 1, 0, trim) && !config.update(token, 101, trim) && trims == 0);
    require(config.update(token, 40, trim) && simulated_live == 40 && config.fast_budget() == 40);
    require(config.settings().max_decode_rows == 3);
    fails([&] { config.update(token, 10, [&](std::size_t, std::size_t) {
        simulated_live = 20; // Partial physical retirement need not be rolled back.
        throw std::runtime_error("injected trim failure");
    }); });
    require(config.fast_budget() == 40 && config.owns(token) && simulated_live == 20);
    require(config.update(token, 0, trim) && config.fast_budget() == 0 && simulated_live == 0);
    require(config.update(token, 100, trim) && config.fast_budget() == 100); // Regrow only within initial ceiling.
    require(config.growth_generation() == 0); // Ordinary scoped callers never opt in.
    fails([&] { config.end(token, [] { throw std::runtime_error("injected release failure"); }); });
    require(config.owns(token) && config.fast_budget() == 100);
    bool released = false;
    require(!config.end(token + 1, [&] { released = true; }) && !released);
    require(config.end(token, [&] { released = true; }) && released);
    require(!config.owns(token) && config.fast_budget() == 17 && config.settings().layers == 48 && config.settings().max_rows == 8);
    require(config.settings().max_decode_rows == 1);
    const auto second = config.begin({0, 1, 1}, true);
    require(second != 0 && second != token && config.fast_budget() == 0);
    require(!config.end(token, [] {}) && config.owns(second));
    require(config.end(second, [] {}) && config.fast_budget() == 17);
    const auto wide_glm = config.begin({100, 12, 1024, 3}, true);
    require(wide_glm != 0 && config.settings().max_rows == 1024 && config.settings().max_decode_rows == 3);
    require(config.end(wide_glm, [] {}) && config.settings().max_rows == 8);

    const auto growth = config.begin({100, 12, 32, 3, true}, true);
    require(growth != 0 && config.settings().capacity_growth && config.growth_generation() == 1);
    require(config.update(growth, 40, trim) && config.growth_generation() == 1);
    fails([&] { config.update(growth, 100, [](std::size_t, std::size_t) {
        throw std::runtime_error("growth trim failure");
    }); });
    require(config.fast_budget() == 40 && config.growth_generation() == 1);
    require(config.update(growth, 100, trim) && config.growth_generation() == 2);
    require(config.update(growth, 100, trim) && config.growth_generation() == 2);
    require(config.update(growth, 0, trim) && config.growth_generation() == 2);
    require(!config.update(growth + 1, 100, trim) && config.growth_generation() == 2);
    require(config.update(growth, 100, trim) && config.growth_generation() == 3);
    fails([&] { config.end(growth, [] { throw std::runtime_error("growth release failure"); }); });
    require(config.owns(growth) && config.settings().capacity_growth && config.growth_generation() == 3);
    require(config.end(growth, [] {}) && !config.settings().capacity_growth && config.growth_generation() == 0);

    // Concurrent clients use the production contract's one enclosing gate.
    // Only one may own settings; a loser cannot overwrite or restore that owner.
    std::mutex gate;
    std::vector<std::thread> clients;
    std::vector<std::uint64_t> tokens(16);
    for (std::size_t i = 0; i < tokens.size(); ++i)
        clients.emplace_back([&, i] { std::lock_guard<std::mutex> lock(gate); tokens[i] = config.begin({64, 4, 128}, true); });
    for (auto& client : clients) client.join();
    std::uint64_t winner = 0; int winners = 0;
    for (auto value : tokens) if (value) { winner = value; ++winners; }
    require(winners == 1 && config.owns(winner));
    require(config.end(winner, [] {}));

    HostMoeCacheConfiguration exhausted({0, 0, 8}, UINT64_MAX);
    const auto last = exhausted.begin({1, 1, 1}, true);
    require(last == UINT64_MAX && exhausted.end(last, [] {}));
    require(exhausted.begin({1, 1, 1}, true) == 0); // Never reuse a stale token after overflow.
    require(exhausted.settings().layers == 0); // Restore even a disabled legacy environment configuration.
    std::printf("host expert cache configuration: %d checks passed (CPU policy/lifecycle only)\n", checks);
}
