// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include <cstdint>

// A verify call feeds the anchor plus at most B drafts. A retained accepted
// prefix always includes the anchor, so at most B rows can be abandoned.
// Padding every compressor ring by B preserves the previous incomplete block
// at ANY alignment. Absolute compressed rows beyond the kept head are masked
// and are recomputed before becoming visible on the replacement branch.
inline bool dsv41_dspark_can_rewind(int64_t verify_begin, int64_t verify_end,
    int64_t current, int64_t target, int64_t block_size, int64_t raw_span)
{
    return block_size > 0 && raw_span >= block_size && verify_begin >= 0 &&
        verify_end > verify_begin && verify_end - verify_begin <= block_size + 1 &&
        current <= verify_end && current > verify_begin &&
        target > verify_begin && target <= current && verify_end - target <= block_size;
}
