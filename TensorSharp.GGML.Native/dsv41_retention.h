// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <cstdint>
#include <limits>

// Conservative per-device admission: all cached graph arenas are charged,
// including graphs also used by active slots. Reserve space for one more full
// slot and an arena as large as the largest observed shape. This is admission,
// not a promise that an arbitrary future graph allocation cannot fail.
inline bool dsv41_retention_fits(uint64_t slot_bytes, uint64_t graph_bytes,
    uint64_t largest_graph, uint64_t retained_count, uint64_t budget,
    bool device_memory_known, uint64_t free_bytes, uint64_t reserve_bytes)
{
    if (!budget || graph_bytes > budget || retained_count == UINT64_MAX) return false;
    if (slot_bytes > (budget - graph_bytes) / (retained_count + 1)) return false;
    if (!device_memory_known) return true; // CPU: explicit byte budget only.
    if (slot_bytes > free_bytes) return false;
    free_bytes -= slot_bytes;
    if (largest_graph > free_bytes) return false;
    return reserve_bytes <= free_bytes - largest_graph;
}
