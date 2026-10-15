// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "ggml_ops_qwen4exp_arena_scope.h"
#include <cstdio>
#include <stdexcept>
#include <limits>

int main() {
    int checks = 0, releases = 0;
    auto check = [&](bool ok) { ++checks; if (!ok) throw std::runtime_error("arena scope invariant failed"); };
    // Reject the empty-prefix join as well as no-wrap/rotary violations. A
    // nonempty media prefix may legitimately have rotary position zero.
    check(!tsg::qwen4exp_arena_position_valid(0, 256, 0));
    check(!tsg::qwen4exp_arena_position_valid(-1, 256, 0));
    check(!tsg::qwen4exp_arena_position_valid(256, 256, 256));
    check(!tsg::qwen4exp_arena_position_valid(1, 0, 1));
    check(!tsg::qwen4exp_arena_position_valid(1, 256, -1));
    check(tsg::qwen4exp_arena_position_valid(1, 256, 0));
    check(tsg::qwen4exp_arena_position_valid(255, 256, 255));
    check(tsg::qwen4exp_arena_position_valid((std::numeric_limits<int>::max)() - 1,
        (std::numeric_limits<int>::max)(), (std::numeric_limits<int>::max)()));
    check(!tsg::qwen4exp_arena_position_valid((std::numeric_limits<int>::max)(),
        (std::numeric_limits<int>::max)(), 1));
    tsg::Qwen4ExpArenaScope scope;
    check(scope.entries() == 4 && !scope.active());
    check(!scope.begin(0, true) && !scope.begin(5, true) && !scope.begin(1, false));
    const auto token = scope.begin(1, true);
    check(token && scope.entries() == 1 && scope.owns(token));
    check(!scope.begin(1, true));
    check(!scope.trim(token + 1, [&] { ++releases; }) && releases == 0);
    bool failed = false;
    try { scope.end(token, [&] { ++releases; throw std::runtime_error("injected partial flush"); }); }
    catch (const std::runtime_error&) { failed = true; }
    check(failed && releases == 1 && scope.owns(token) && scope.entries() == 1);
    check(scope.trim(token, [&] { ++releases; }) && scope.owns(token) && scope.entries() == 1);
    check(scope.end(token, [&] { ++releases; }) && releases == 3 && !scope.active() && scope.entries() == 4);
    check(!scope.end(token, [&] { ++releases; }) && releases == 3);
    const auto next = scope.begin(2, true);
    check(next != token && scope.entries() == 2 && !scope.owns(token));
    check(scope.end(next, [] {}));
    tsg::Qwen4ExpArenaScope exhausted((std::numeric_limits<std::uint64_t>::max)());
    const auto last = exhausted.begin(1, true);
    check(last && exhausted.end(last, [] {}));
    check(!exhausted.begin(1, true) && !exhausted.active());
    std::printf("PASS: %d arena scope checks (retention, failure retry, stale token, exhaustion)\n", checks);
}
