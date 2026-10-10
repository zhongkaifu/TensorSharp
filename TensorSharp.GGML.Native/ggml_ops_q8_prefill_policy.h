// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <cstdint>
#include <initializer_list>

// Prefer reuse only when padding wastes at most a quarter of a column tile
// and the grid can populate up to two resident blocks on each SM. A large
// column tile otherwise reduces useful parallelism, especially for row strips.
inline int tsg_q8_prefill_auto_columns(int rows, int columns, int multiprocessors,
                                     int active64, int active128) {
    if (rows <= 0 || columns < 64 || multiprocessors <= 0) return 32;
    const int64_t row_tiles = (int64_t(rows) + 63) / 64;
    for (int tile : {128, 64}) {
        const int active = tile == 128 ? active128 : active64;
        if (columns < tile || active <= 0) continue;
        const int64_t column_tiles = (int64_t(columns) + tile - 1) / tile;
        if (int64_t(columns) * 4 < column_tiles * tile * 3) continue;
        const int64_t target = int64_t(multiprocessors) * (active < 2 ? active : 2);
        if (row_tiles * column_tiles >= target) return tile;
    }
    return 32;
}
