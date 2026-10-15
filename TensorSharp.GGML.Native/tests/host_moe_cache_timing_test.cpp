// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_cache_timing.h"
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

struct FakeClock {
    using duration = std::chrono::nanoseconds;
    using time_point = std::chrono::time_point<FakeClock>;
    static inline std::int64_t tick = 0;
    static inline int reads = 0;
    static time_point now() noexcept { ++reads; return time_point(duration(tick)); }
};

int main() {
    int checks = 0;
    const auto require = [&](bool value) {
        ++checks;
        if (!value) { std::fprintf(stderr, "FAIL cache timing check %d\n", checks); std::exit(1); }
    };
    using Stage = tsg::HostMoeCacheStage;
    using Timer = tsg::HostMoeCacheStageTimer<FakeClock>;
    using Timing = tsg::HostMoeCacheTiming;
    require(Timing::enabled("1"));
    const char* disabled_values[]{nullptr, "", "0", "01", "1x", " 1", "1 "};
    for (auto value : disabled_values)
        require(!Timing::enabled(value));
    { Timer disabled(nullptr, Stage::compute); }
    require(FakeClock::reads == 0);
    Timing timing;
    {
        Timer outer(&timing, Stage::plan);
        FakeClock::tick = 20;
        { Timer inner(&timing, Stage::plan); FakeClock::tick = 50; }
        require(FakeClock::reads == 1);
        FakeClock::tick = 100;
    }
    require(FakeClock::reads == 2);
    require(timing.calls[static_cast<std::size_t>(Stage::plan)] == 1);
    require(timing.nanoseconds[static_cast<std::size_t>(Stage::plan)] == 100);
    require(timing.depth[static_cast<std::size_t>(Stage::plan)] == 0);
    try {
        Timer failed(&timing, Stage::compute);
        FakeClock::tick = 150;
        throw std::runtime_error("injected compute failure");
    } catch (const std::runtime_error&) { }
    require(timing.calls[static_cast<std::size_t>(Stage::compute)] == 1);
    require(timing.nanoseconds[static_cast<std::size_t>(Stage::compute)] == 50);
    require(timing.depth[static_cast<std::size_t>(Stage::compute)] == 0);
    timing.record(Stage::output_bridge, -1);
    require(timing.milliseconds(Stage::output_bridge) == 0);
    timing.record(Stage::output_bridge, 1000000);
    require(timing.milliseconds(Stage::output_bridge) == 1);
    require(timing.calls[static_cast<std::size_t>(Stage::output_bridge)] == 2);
    auto maximum = (std::numeric_limits<std::uint64_t>::max)();
    Timing::add(maximum, 1);
    require(maximum == (std::numeric_limits<std::uint64_t>::max)());
    std::printf("PASS %d host-only cache timing checks; disabled path reads no clock\n", checks);
}
