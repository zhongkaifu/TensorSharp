// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_host_moe_file_read_plan.h"
#include "../ggml_ops_host_moe_cache_policy.h"
#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <map>

int main() {
    int checks = 0;
    const auto require = [&](bool value) {
        ++checks;
        if (!value) { std::fprintf(stderr, "FAIL expert read plan check %d\n", checks); std::exit(1); }
    };
    const std::array<std::size_t, 3> stride{7, 13, 19};
    // Router order and cache-slot order deliberately differ from file order.
    const std::vector<std::pair<int, int>> misses{{5, 8}, {2, 1}, {9, 0}, {0, 2}, {7, 5}};
    for (const std::size_t window : {1, 8, 17, 31, 128, 1024})
        for (const std::size_t maximum_read : {1, 9, 32, 512}) {
            std::size_t windows = 0, total = 0, merged = 0;
            std::map<std::pair<int, int>, std::vector<std::uint8_t>> output;
            for (const auto& miss : misses) for (int part = 0; part < 3; ++part)
                output[{miss.first, part}].resize(stride[part], 0xff);
            tsg::host_moe_for_each_file_read_window(misses, stride, window, maximum_read,
                [&](const auto& runs, const auto& pieces, std::size_t used) {
                    ++windows; require(used > 0 && used <= window && !runs.empty());
                    std::vector<std::uint8_t> staging(used, 0);
                    std::vector<bool> covered(used, false);
                    for (const auto& run : runs) {
                        require(run.bytes > 0 && run.bytes <= maximum_read && run.part >= 0 && run.part < 3
                            && run.staging_offset <= used && run.bytes <= used - run.staging_offset
                            && run.first_piece < pieces.size() && run.piece_count > 0
                            && run.piece_count <= pieces.size() - run.first_piece);
                        if (run.piece_count > 1) ++merged;
                        std::size_t piece_bytes = 0;
                        for (std::size_t i = 0; i < run.bytes; ++i) {
                            require(!covered[run.staging_offset + i]);
                            covered[run.staging_offset + i] = true;
                            staging[run.staging_offset + i] = static_cast<std::uint8_t>(run.part * 41 + run.source_offset + i);
                        }
                        for (std::size_t p = run.first_piece; p < run.first_piece + run.piece_count; ++p) {
                            const auto& piece = pieces[p];
                            const auto found = std::find_if(misses.begin(), misses.end(), [&](const auto& miss) { return miss.first == piece.slot; });
                            require(found != misses.end() && piece.part == run.part
                                && piece.bytes > 0 && piece.expert_offset < stride[piece.part]
                                && piece.bytes <= stride[piece.part] - piece.expert_offset
                                && piece.staging_offset == run.staging_offset + piece_bytes
                                && static_cast<std::size_t>(found->second) * stride[piece.part] + piece.expert_offset
                                    == run.source_offset + piece_bytes);
                            auto& target = output.at({piece.slot, piece.part});
                            std::copy_n(staging.begin() + piece.staging_offset, piece.bytes, target.begin() + piece.expert_offset);
                            piece_bytes += piece.bytes;
                        }
                        require(piece_bytes == run.bytes); total += run.bytes;
                    }
                    require(std::all_of(covered.begin(), covered.end(), [](bool value) { return value; }));
                });
            require(total == misses.size() * (stride[0] + stride[1] + stride[2]) && windows > 0);
            if (window >= 128 && maximum_read >= 32) require(merged > 0);
            for (const auto& miss : misses) for (int part = 0; part < 3; ++part) {
                const auto& target = output.at({miss.first, part});
                for (std::size_t i = 0; i < stride[part]; ++i)
                    require(target[i] == static_cast<std::uint8_t>(part * 41 + miss.second * stride[part] + i));
            }
        }
    const auto rejects = [&](const auto& selected, const auto& extents, std::size_t window, std::size_t read) {
        int callbacks = 0; bool refused = false;
        try { tsg::host_moe_for_each_file_read_window(selected, extents, window, read,
            [&](const auto&, const auto&, std::size_t) { ++callbacks; }); }
        catch (const std::exception&) { refused = true; }
        require(refused && callbacks == 0);
    };
    rejects(misses, stride, 0, 4); rejects(misses, stride, 4, 0);
    rejects(misses, std::array<std::size_t, 3>{0, 3, 5}, 4, 4);
    rejects(std::vector<std::pair<int, int>>{{0, -1}}, stride, 4, 4);
    rejects(std::vector<std::pair<int, int>>{{-1, 0}}, stride, 4, 4);
    rejects(std::vector<std::pair<int, int>>{{0, 1}, {1, 1}}, stride, 4, 4);
    rejects(std::vector<std::pair<int, int>>{{0, 1}, {0, 2}}, stride, 4, 4);
    rejects(std::vector<std::pair<int, int>>{{0, 2}},
        std::array<std::size_t, 3>{(std::numeric_limits<std::size_t>::max)(), 1, 1}, 4, 4);
    int callbacks = 0;
    tsg::host_moe_for_each_file_read_window({}, stride, 1, 1,
        [&](const auto&, const auto&, std::size_t) { ++callbacks; });
    require(callbacks == 0);
    std::printf("PASS %d pure selected expert read checks; exact source bytes, bounded runs/windows, scattered slots and pre-I/O rejection; no CUDA or model loaded\n", checks);
}
