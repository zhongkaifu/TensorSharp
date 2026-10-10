// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#if defined(__linux__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace tsg {
// A transient optimization hint, never an ownership or validity check. Sources
// remain owned by the caller, and may be reclaimed after mincore returns. An
// unknown result keeps the ordinary upload path. Inspect pages without reading
// model bytes, allocating a payload buffer, or pinning the mapping.
inline bool mapped_upload_has_nonresident_pages(const void* source, std::size_t bytes)
{
#if defined(__linux__)
    if (!source || !bytes) return false;
    static const long queried_page = sysconf(_SC_PAGESIZE);
    if (queried_page <= 0) return false;
    const auto page = static_cast<std::size_t>(queried_page);
    const auto address = reinterpret_cast<std::uintptr_t>(source);
    if (bytes > std::numeric_limits<std::uintptr_t>::max() - address) return false;
    const auto start = address - address % page;
    const auto end = address + bytes;
    unsigned char resident[4096];
    const auto window = sizeof(resident) * page;
    for (auto current = start; current < end;)
    {
        const auto length = std::min<std::uintptr_t>(window, end - current);
        if (mincore(reinterpret_cast<void*>(current), length, resident) != 0) return false;
        const auto count = length / page + (length % page != 0);
        for (std::size_t i = 0; i < count; ++i)
            if ((resident[i] & 1) == 0) return true;
        current += length;
    }
#else
    (void)source; (void)bytes;
#endif
    return false;
}
}
