// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
// Pure layout oracle: no native backend, model, file I/O, or weight allocation.
#include "../ggml_ops_host_moe_file_batch.h"
#include <cstdio>
#include <stdexcept>

namespace {
int checks = 0;
void require(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
template<class Action> void rejects(Action action) {
    bool caught = false;
    try { action(); } catch (const std::exception&) { caught = true; }
    require(caught, "invalid range accepted");
}
void verify(const std::vector<std::int32_t>& original, int count, int used) {
    using namespace tsg;
    const std::array<std::size_t, 3> strides{3, 5, 7}; // Different projection layouts.
    std::array<std::size_t, 3> totals{};
    for (int p = 0; p < 3; ++p) totals[p] = strides[p] * count;
    std::vector<float> routes(original.size(), 0.25f);
    HostMoeFileBatch plan;
    plan.plan(original.data(), routes.data(), static_cast<int>(original.size()) / used, used, count, totals);
    bool duplicate = false;
    for (std::size_t row = 0; row < original.size(); row += used)
        for (int slot = 0; slot < used; ++slot)
            for (int previous = 0; previous < slot; ++previous)
                duplicate |= original[row + slot] == original[row + previous];
    require(plan.duplicate_routes == duplicate, "per-row duplicate classification changed");
    require(std::is_sorted(plan.experts.begin(), plan.experts.end()), "experts not physically sorted");
    require(std::adjacent_find(plan.experts.begin(), plan.experts.end()) == plan.experts.end(), "duplicate compact expert");
    for (std::size_t i = 0; i < original.size(); ++i)
        require(plan.experts.at(plan.ids.at(i)) == original[i], "route slot changed");
    std::size_t expected_runs = 1;
    for (std::size_t i = 1; i < plan.experts.size(); ++i)
        if (plan.experts[i] != plan.experts[i - 1] + 1) ++expected_runs;
    std::array<std::vector<int>, 3> touched;
    for (int p = 0; p < 3; ++p) touched[p].resize(totals[p]);
    std::vector<int> target(plan.payload, -1);
    std::size_t calls = 0;
    plan.for_each_read_range([&](int part, std::size_t source, std::size_t destination, std::size_t length) {
        ++calls;
        require(source % strides[part] == 0 && length % strides[part] == 0, "unaligned expert range");
        require(source <= totals[part] && length <= totals[part] - source, "read crosses projection");
        require(destination >= plan.offset[part] && destination + length <= plan.offset[part] + plan.bytes[part], "write crosses compact projection");
        for (std::size_t i = 0; i < length; ++i) {
            require(++touched[part][source + i] == 1, "source byte read twice");
            require(target[destination + i] == -1, "overlapping destination");
            target[destination + i] = static_cast<int>(source + i);
        }
    });
    require(calls == expected_runs * 3, "adjacent experts not maximally merged");
    for (int p = 0; p < 3; ++p) {
        for (int expert = 0; expert < count; ++expert) {
            const int expected = std::binary_search(plan.experts.begin(), plan.experts.end(), expert) ? 1 : 0;
            for (std::size_t byte = 0; byte < strides[p]; ++byte)
                require(touched[p][expert * strides[p] + byte] == expected, "unselected gap read or selected byte omitted");
        }
        for (std::size_t slot = 0; slot < plan.experts.size(); ++slot)
            for (std::size_t byte = 0; byte < strides[p]; ++byte)
                require(target[plan.offset[p] + slot * strides[p] + byte] == static_cast<int>(plan.experts[slot] * strides[p] + byte), "merged compact payload differs from per-expert oracle");
        for (std::size_t i = plan.bytes[p]; i < host_moe_align(plan.bytes[p]); ++i)
            require(target[plan.offset[p] + i] == -1, "alignment padding written");
    }
    // Fail closed on malformed extents/arena even before a read is submitted.
    auto invalid = plan; invalid.extent[0] = 0;
    int visited = 0;
    rejects([&] { invalid.for_each_read_range([&](int, std::size_t, std::size_t, std::size_t) { ++visited; }); });
    require(visited == 0, "invalid first projection emitted a read");
    invalid = plan; invalid.payload = 0;
    rejects([&] { invalid.for_each_read_range([](int, std::size_t, std::size_t, std::size_t) {}); });
}
}
int main() {
    try {
        // Every nonempty subset, descending routes and repeated IDs. Includes
        // first/last expert, full runs, alternating gaps and isolated experts.
        for (int mask = 1; mask < (1 << 9); ++mask) {
            std::vector<std::int32_t> ids;
            for (int i = 8; i >= 0; --i) if (mask & (1 << i)) { ids.push_back(i); ids.push_back(i); }
            verify(ids, 9, 1);
        }
        verify({6, 6, 5, 2, 3, 2}, 7, 3);
        verify({6, 5, 4, 6, 4, 5}, 7, 3); // Across-row reuse is not a duplicate route.
        verify({2, 2, 2}, 7, 3); // Legal repeated routes with unique count < top K.
        const std::int32_t id = 0; const float route = 1;
        tsg::HostMoeFileBatch plan;
        const auto limit = (std::numeric_limits<std::size_t>::max)();
        rejects([&] { plan.plan(&id, &route, 1, 1, 1, {limit, 1, 1}); });
        rejects([&] { plan.plan(&id, &route, 1, 1, 1, {limit / 2, limit / 2, 128}); });
        const float nonfinite = std::numeric_limits<float>::infinity();
        rejects([&] { plan.plan(&id, &nonfinite, 1, 1, 1, {1, 1, 1}); });
        std::printf("PASS: %d checks; sorted remap, merged ranges, no gaps, bounded extents (pure CPU layout only)\n", checks);
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1;
    }
}
