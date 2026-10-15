// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once
#include "ggml_ops_precision_policy.h"
#include <algorithm>
#include <array>
#include <cstdint>

namespace tsg
{
    constexpr int kHostMoeCacheMaxPrefillRows = 128;
    // GLM's explicit selected-expert adapter plans a serialized whole union.
    // Keep Qwen's environment/caller bound unchanged while permitting larger
    // admitted GLM chunks to amortize the same source-file expert reads.
    constexpr int kHostMoeCacheMaxGlmPrefillRows = 1024;
    constexpr int kHostMoeCacheMaxDecodeRows = 8;
    // Qwen keeps its original 16 KiB routing validation bound. The explicit
    // GLM adapter owns a separate 32 KiB bound for a complete1024/top8 block.
    constexpr int kHostMoeCacheMaxRouteElements = 4096;
    constexpr int kHostMoeCacheMaxGlmRouteElements = 8192;

    // Experimental uncached reads only serve a genuine GLM bulk prefill.
    // Scalar, short prefill/MTP verification and independent fused decode must
    // retain buffered reads: repeatedly bypassing the OS cache regressed decode.
    inline bool host_moe_cache_direct_read_phase(bool glm_bulk_prefill, int rows,
        bool independent_decode_rows) noexcept
    {
        return glm_bulk_prefill && !independent_decode_rows
            && rows > kHostMoeCacheMaxDecodeRows && rows <= kHostMoeCacheMaxGlmPrefillRows;
    }

    inline bool host_moe_cache_routes_fit(int rows, int selected, bool explicit_glm = false)
    {
        return rows > 0 && selected > 0
            && rows <= (explicit_glm ? kHostMoeCacheMaxGlmRouteElements : kHostMoeCacheMaxRouteElements) / selected;
    }

    inline bool host_moe_cache_prefill_fits(int rows, int selected, bool explicit_glm)
    {
        return rows <= (explicit_glm ? kHostMoeCacheMaxGlmPrefillRows : kHostMoeCacheMaxPrefillRows)
            && host_moe_cache_routes_fit(rows, selected, explicit_glm);
    }

    // Expert FFNs are independent for every token row. Select the largest
    // already-planned exact-width microgroup whose complete routed union fits
    // the existing slot table. This never grows a cache or allocates scratch.
    // All IDs must have passed the enclosing full-block validation first.
    inline int host_moe_cache_prefill_group_rows(const std::int32_t* ids, int selected,
        int remaining_rows, int planned_rows, int capacity)
    {
        if (ids == nullptr || selected <= 0 || selected > 64 || remaining_rows <= 0
            || planned_rows <= 0 || planned_rows > kHostMoeCacheMaxDecodeRows || capacity <= 0) return 0;
        std::array<std::int32_t, 64 * kHostMoeCacheMaxDecodeRows> sorted{};
        for (int rows = std::min(remaining_rows, planned_rows); rows > 0; --rows) {
            const int count = rows * selected;
            std::copy(ids, ids + count, sorted.begin());
            std::sort(sorted.begin(), sorted.begin() + count);
            const int distinct = static_cast<int>(std::unique(sorted.begin(), sorted.begin() + count) - sorted.begin());
            if (distinct <= capacity) return rows;
        }
        return 0;
    }

    // This opt-in only widens the compact expert helper, never the attention,
    // router, recurrent-state or whole-model precision policy. Invalid values
    // retain the existing decode-width behavior; they cannot enlarge a bound.
    inline int host_moe_cache_prefill_rows(const char* value)
    {
        const int fallback = static_cast<int>(TSG_PRECISION_DECODE_COLUMNS);
        if (value == nullptr || *value == '\0') return fallback;
        int result = 0;
        for (const char* p = value; *p; ++p)
        {
            if (*p < '0' || *p > '9') return fallback;
            result = result * 10 + (*p - '0');
            if (result > kHostMoeCacheMaxPrefillRows) return fallback;
        }
        return result > fallback ? result : fallback;
    }
}
