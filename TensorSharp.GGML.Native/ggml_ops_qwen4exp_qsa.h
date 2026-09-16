// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <algorithm>
#include <array>
#include <cstdint>
#include <limits>
#include <map>
#include <numeric>
#include <stdexcept>
#include <vector>

// A single holder owns a contiguous cache. Unlike a unified llama cache there
// are no foreign sequence cells or holes below live_keys. Coordinates are
// interleaved T,H,W; graph rotary inputs are planar T,H,W,T.
struct Q4eQsaPlan
{
    std::vector<int32_t> cell_blocks, block_cells, block_positions, query_positions;
    std::vector<float> bias;
    bool ranked = false;
};

inline Q4eQsaPlan q4e_qsa_plan(const int32_t* positions, int live_keys,
    int padded_keys, int query_start, int query_count, int ratio)
{
    if (!positions || live_keys <= 0 || padded_keys < live_keys || query_start < 0
        || query_count <= 0 || query_start > live_keys - query_count
        || ratio <= 0 || ratio > 64 || padded_keys > INT32_MAX - ratio)
        throw std::invalid_argument("qwen4exp QSA: invalid position plan");
    const int blocks = (padded_keys + ratio - 1) / ratio;
    if ((int64_t)blocks * ratio > INT32_MAX || (int64_t)blocks * 4 > INT32_MAX
        || (int64_t)query_count * blocks > INT32_MAX)
        throw std::invalid_argument("qwen4exp QSA: position plan overflow");
    auto coord = [&](int c) { return std::array<int32_t, 3>{positions[3LL*c], positions[3LL*c+1], positions[3LL*c+2]}; };
    std::vector<int> order(live_keys), indices(live_keys);
    std::iota(order.begin(), order.end(), 0);
    std::map<int32_t, int> seen;
    Q4eQsaPlan result;
    for (int i = 0; i < live_keys; ++i)
    {
        auto p = coord(i);
        if (p[0] < 0 || p[1] < 0 || p[2] < 0)
            throw std::invalid_argument("qwen4exp QSA: negative coordinate");
        result.ranked |= !seen.emplace(p[0], i).second;
        indices[i] = p[0];
    }
    if (result.ranked)
    {
        // Equal full coordinates keep arrival order deterministically. The
        // causal comparison below uses upper_bound, so all equal coordinates
        // share the same tail decision.
        std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return coord(a) < coord(b); });
        for (int i = 0; i < live_keys; ++i) indices[order[i]] = i;
    }
    std::map<int, std::vector<int>> buckets;
    for (int c = 0; c < live_keys; ++c)
    {
        auto& cells = buckets[indices[c] / ratio];
        if (cells.empty()) cells.assign(ratio, -1);
        cells[indices[c] % ratio] = c;
    }
    result.cell_blocks.assign(padded_keys, -1);
    result.block_cells.assign((size_t)ratio * blocks, 0);
    result.block_positions.assign((size_t)4 * blocks, 0);
    result.query_positions.resize((size_t)4 * query_count);
    result.bias.assign((size_t)blocks * query_count, -std::numeric_limits<float>::infinity());
    std::vector<int> starts;
    for (const auto& pair : buckets)
    {
        if (std::find(pair.second.begin(), pair.second.end(), -1) != pair.second.end()) continue;
        const int b = (int)starts.size();
        if (b >= blocks) throw std::logic_error("qwen4exp QSA: too many complete blocks");
        starts.push_back(pair.first * ratio);
        const auto p = result.ranked ? coord(pair.second[0])
            : std::array<int32_t, 3>{starts.back(), starts.back(), starts.back()};
        for (int s = 0; s < 4; ++s) result.block_positions[(size_t)s * blocks + b] = p[s == 3 ? 0 : s];
        for (int s = 0; s < ratio; ++s)
        {
            const int c = pair.second[s];
            result.block_cells[(size_t)b * ratio + s] = c;
            result.cell_blocks[c] = b;
        }
    }
    const int full = (int)starts.size();
    const int dead = full < blocks ? full : blocks - 1;
    for (int& b : result.cell_blocks) if (b < 0) b = dead;
    for (int i = 0; i < query_count; ++i)
    {
        const auto p = coord(query_start + i);
        for (int s = 0; s < 4; ++s) result.query_positions[(size_t)s * query_count + i] = p[s == 3 ? 0 : s];
        int64_t q = p[0];
        if (result.ranked)
            q = std::upper_bound(order.begin(), order.end(), p,
                [&](const std::array<int32_t, 3>& value, int c) { return value < coord(c); }) - order.begin() - 1;
        const int64_t tail = (q + 1) / ratio * ratio;
        for (int b = 0; b < full; ++b)
            result.bias[(size_t)i * blocks + b] = starts[b] >= tail ? 1e9f : 0.f;
        if (full < blocks) result.bias[(size_t)i * blocks + dead] = 1e9f;
    }
    return result;
}
