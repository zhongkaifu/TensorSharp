// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#include "../ggml_ops_file_source.h"
#include <cstdio>
#include <cstring>

namespace {
std::size_t checks = 0, direct_pieces = 0, buffered_pieces = 0;
void require(bool value, const char* message) {
    ++checks; if (!value) throw std::runtime_error(message);
}
std::uint8_t source_byte(int part, std::uint64_t absolute) {
    return static_cast<std::uint8_t>(absolute * 37 + part * 53 + absolute / 251);
}
void exercise(std::size_t window, std::size_t alignment, std::uint64_t base, std::size_t address_shift) {
    const std::array<std::size_t, 3> stride{31, 1025, 8128};
    std::array<tsg::HostMoeDirectFileGeometry, 3> files{};
    for (int part = 0; part < 3; ++part) files[part] = {base + static_cast<unsigned>(part), stride[part] * 9, alignment};
    const std::vector<std::pair<int, int>> misses{{2, 8}, {0, 0}, {3, 5}, {1, 1}};
    // Test-only storage. Production receives its already charged pinned arena.
    std::vector<std::uint8_t> allocation(window + 8192 + 4, 0xa7);
    const auto raw = reinterpret_cast<std::uintptr_t>(allocation.data() + 1);
    const auto arena_address = (raw + 4095) / 4096 * 4096 + address_shift;
    auto* arena = reinterpret_cast<std::uint8_t*>(arena_address);
    std::array<std::vector<std::uint8_t>, 3> output;
    for (int part = 0; part < 3; ++part) output[part].assign(4 * stride[part] + 2, 0xa7);
    std::size_t payload = 0;
    tsg::host_moe_for_each_direct_read_window(misses, stride, files, window, arena_address,
        [&](const std::vector<tsg::HostMoeDirectFilePiece>& pieces, std::size_t used) {
            require(!pieces.empty() && used > 0 && used <= window, "window escaped fixed arena");
            std::fill(arena, arena + window, 0xa7);
            std::size_t raw_end = 0;
            for (const auto& piece : pieces) {
                const auto& file = files[piece.part];
                const auto read_offset = piece.direct_bytes ? piece.direct_offset : piece.source_offset;
                const auto destination = piece.direct_bytes ? piece.direct_staging_offset : piece.staging_offset;
                const auto bytes = piece.direct_bytes ? piece.direct_bytes : piece.bytes;
                require(bytes > 0 && destination >= raw_end && destination <= used && bytes <= used - destination,
                    "raw read spans overlap or escape fixed window");
                raw_end = destination + bytes;
                require(read_offset <= file.bytes && bytes <= file.bytes - read_offset, "read crosses registered extent");
                require(piece.bytes > 0 && piece.expert_offset + piece.bytes <= stride[piece.part], "logical expert span invalid");
                if (piece.direct_bytes) {
                    ++direct_pieces;
                    require(alignment && (file.offset + read_offset) % alignment == 0
                        && (arena_address + destination) % alignment == 0 && bytes % alignment == 0,
                        "direct source/destination/count not aligned");
                    require(piece.source_offset - read_offset == piece.staging_offset - destination
                        && piece.bytes <= bytes - (piece.staging_offset - destination), "logical payload not inside padded read");
                } else {
                    ++buffered_pieces;
                    require(read_offset == piece.source_offset && bytes == piece.bytes, "fallback read expanded logical payload");
                }
                for (std::size_t i = 0; i < bytes; ++i) arena[destination + i] = source_byte(piece.part, file.offset + read_offset + i);
            }
            for (const auto& piece : pieces) {
                auto& target = output[piece.part];
                std::memcpy(target.data() + 1 + piece.slot * stride[piece.part] + piece.expert_offset,
                    arena + piece.staging_offset, piece.bytes);
                payload += piece.bytes;
            }
        });
    require(payload == 4 * (stride[0] + stride[1] + stride[2]), "logical payload count differs");
    for (int part = 0; part < 3; ++part) {
        require(output[part].front() == 0xa7 && output[part].back() == 0xa7, "logical scatter overwrote guard");
        for (const auto& miss : misses) for (std::size_t i = 0; i < stride[part]; ++i)
            require(output[part][1 + miss.first * stride[part] + i]
                == source_byte(part, files[part].offset + miss.second * stride[part] + i), "padded scatter changed requested source bytes");
    }
    require(allocation.front() == 0xa7 && allocation.back() == 0xa7, "fixed arena guard changed");
}
}
int main() {
    try {
        for (const auto window : {std::size_t(1), std::size_t(8), std::size_t(513), std::size_t(8192)})
            for (const auto alignment : {std::size_t(0), std::size_t(4), std::size_t(512), std::size_t(4096)})
                for (const auto base : {std::uint64_t(0), std::uint64_t(1), std::uint64_t(4095)})
                    for (const auto address_shift : {std::size_t(0), std::size_t(3)}) exercise(window, alignment, base, address_shift);
        const std::array<std::size_t, 3> stride{16, 16, 16};
        const std::array<tsg::HostMoeDirectFileGeometry, 3> geometry{{{0, 64, 4096}, {0, 64, 0}, {0, 64, 512}}};
        const auto rejected = [&](auto misses, auto widths, auto files, std::size_t window, std::uintptr_t address) {
            int visits = 0; bool failed = false;
            try { tsg::host_moe_for_each_direct_read_window(misses, widths, files, window, address,
                [&](const auto&, auto) { ++visits; }); } catch (const std::exception&) { failed = true; }
            require(failed && visits == 0, "invalid complete plan performed partial I/O");
        };
        rejected(std::vector<std::pair<int,int>>{{0, 0}, {1, 4}}, stride, geometry, 8, 4096);
        rejected(std::vector<std::pair<int,int>>{{0, 0}, {0, 1}}, stride, geometry, 8, 4096);
        rejected(std::vector<std::pair<int,int>>{{-1, 0}}, stride, geometry, 8, 4096);
        rejected(std::vector<std::pair<int,int>>{{0, -1}}, stride, geometry, 8, 4096);
        rejected(std::vector<std::pair<int,int>>{{0, 0}}, stride, geometry, 0, 4096);
        rejected(std::vector<std::pair<int,int>>{{0, 0}}, stride, geometry, 8, UINTPTR_MAX - 7);
        auto invalid = geometry; invalid[2].alignment = 3;
        rejected(std::vector<std::pair<int,int>>{{0, 0}}, stride, invalid, 8, 4096);
        invalid = geometry; invalid[2].offset = UINT64_MAX - 63;
        rejected(std::vector<std::pair<int,int>>{{0, 0}}, stride, invalid, 8, 4096);
        int empty_visits = 0;
        tsg::host_moe_for_each_direct_read_window({}, stride, geometry, 8, 4096,
            [&](const auto&, auto) { ++empty_visits; });
        require(empty_visits == 0 && direct_pieces > 0 && buffered_pieces > 0, "direct/fallback/empty coverage absent");
        std::printf("PASS: %zu bounded aligned direct-read planner checks; direct=%zu buffered=%zu, exact logical bytes, padding/window/extent/guard/invalid-complete-plan proof\n",
            checks, direct_pieces, buffered_pieces);
        return 0;
    } catch (const std::exception& error) { std::fprintf(stderr, "FAIL: %s\n", error.what()); return 1; }
}
