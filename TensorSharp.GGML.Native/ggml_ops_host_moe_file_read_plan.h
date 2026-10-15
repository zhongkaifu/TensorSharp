// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace tsg {
struct HostMoeFileReadPiece {
    int slot = 0, part = 0;
    std::size_t expert_offset = 0, staging_offset = 0, bytes = 0;
};
struct HostMoeFileReadRun {
    int part = 0;
    std::size_t source_offset = 0, staging_offset = 0, bytes = 0;
    std::size_t first_piece = 0, piece_count = 0;
};

// Read only the selected misses, in physical source order. Adjacent expert
// extents share one bounded read while uploads scatter into their original
// cache slots. Windows never overlap and reuse the caller's existing arena.
// Cap an individual run separately so several workers can read one window.
template<class Visit>
void host_moe_for_each_file_read_window(const std::vector<std::pair<int, int>>& misses,
    const std::array<std::size_t, 3>& stride, std::size_t window, std::size_t maximum_read,
    Visit&& visit) {
    if (!window || !maximum_read) throw std::invalid_argument("Expert read plan requires positive bounds");
    for (const auto bytes : stride) if (!bytes) throw std::invalid_argument("Expert read plan has an empty projection");
    auto ordered = misses;
    std::sort(ordered.begin(), ordered.end(), [](const auto& a, const auto& b) { return a.second < b.second; });
    std::vector<int> slots; slots.reserve(ordered.size());
    for (std::size_t i = 0; i < ordered.size(); ++i) {
        const auto& miss = ordered[i];
        if (miss.first < 0 || miss.second < 0 || (i && ordered[i - 1].second == miss.second))
            throw std::invalid_argument("Expert read plan has an invalid or repeated miss");
        slots.push_back(miss.first);
        for (const auto bytes : stride)
            if (static_cast<std::size_t>(miss.second) >= (std::numeric_limits<std::size_t>::max)() / bytes)
                throw std::overflow_error("Expert read source extent overflows");
    }
    std::sort(slots.begin(), slots.end());
    if (std::adjacent_find(slots.begin(), slots.end()) != slots.end())
        throw std::invalid_argument("Expert read plan repeats a destination slot");
    std::vector<HostMoeFileReadRun> runs;
    std::vector<HostMoeFileReadPiece> pieces;
    std::size_t used = 0;
    const auto flush = [&] {
        if (runs.empty()) return;
        visit(runs, pieces, used);
        runs.clear(); pieces.clear(); used = 0;
    };
    for (int part = 0; part < 3; ++part) for (const auto& miss : ordered) {
        const auto source = static_cast<std::size_t>(miss.second) * stride[part];
        for (std::size_t offset = 0; offset < stride[part];) {
            if (used == window) flush();
            const bool merge = !runs.empty() && runs.back().part == part
                && runs.back().source_offset + runs.back().bytes == source + offset
                && runs.back().staging_offset + runs.back().bytes == used
                && runs.back().bytes < maximum_read;
            const auto remaining_read = merge ? maximum_read - runs.back().bytes : maximum_read;
            const auto count = (std::min)({stride[part] - offset, window - used, remaining_read});
            if (!merge) runs.push_back({part, source + offset, used, 0, pieces.size(), 0});
            pieces.push_back({miss.first, part, offset, used, count});
            runs.back().bytes += count; ++runs.back().piece_count;
            used += count; offset += count;
        }
    }
    flush();
}
} // namespace tsg
