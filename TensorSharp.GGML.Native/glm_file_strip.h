// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <limits>
#include <vector>

namespace tsg_glm {

struct seek_file_end
{
    int operator()(FILE * file) const
    {
#if defined(_WIN32)
        return _fseeki64(file, 0, SEEK_END);
#else
        return fseeko(file, 0, SEEK_END);
#endif
    }
};

struct tell_file_position
{
    int64_t operator()(FILE * file) const
    {
#if defined(_WIN32)
        return _ftelli64(file);
#else
        return ftello(file);
#endif
    }
};

// A failed size observation must not authorize loading an unchecked shard.
template<class Seek = seek_file_end, class Tell = tell_file_position>
inline bool read_file_size(FILE * file, size_t & bytes, Seek seek = {}, Tell tell = {})
{
    bytes = 0;
    if (!file || seek(file) != 0) return false;
    const int64_t position = tell(file);
    if (position < 0 || uint64_t(position) > (std::numeric_limits<size_t>::max)()) return false;
    bytes = static_cast<size_t>(position);
    return true;
}

struct seek_file_start
{
    int operator()(FILE * file, int64_t offset) const
    {
#if defined(_WIN32)
        return _fseeki64(file, offset, SEEK_SET);
#else
        return fseeko(file, (off_t) offset, SEEK_SET);
#endif
    }
};

// A failed seek must not turn a subsequent successful read at the old cursor
// into an apparently valid weight upload. A stateless default keeps the native
// seek inline; the callback lets portable tests inject failure without bad I/O.
template<class Seek = seek_file_start>
inline bool read_file_range(FILE * file, uint64_t offset, size_t bytes, void * output, Seek seek = {})
{
    const uint64_t max_offset = uint64_t(std::numeric_limits<int64_t>::max());
    if (!file || !output || bytes == 0 || offset > max_offset || bytes > max_offset - offset)
        return false;
    if (seek(file, (int64_t) offset) != 0) return false;
    return fread(output, 1, bytes, file) == bytes;
}

// Gather a TP rank's strip from each checkpoint row. Reading one strip at a
// time caused millions of small reads on network storage. Bounded source
// slabs amortize the reads; each worker reuses its scratch allocation.
template<class Scratch>
inline bool read_file_strips(FILE * file, uint64_t offset, size_t stride, size_t strip,
                             size_t bytes, void * output, Scratch & scratch,
                             size_t slab_bytes = 1024 * 1024)
{
    if (!file || !output || strip == 0 || stride < strip || bytes == 0 || bytes % strip != 0)
        return false;
    const size_t rows = bytes / strip;
    const uint64_t max_offset = uint64_t(std::numeric_limits<int64_t>::max());
    if (offset > max_offset || strip > max_offset - offset ||
        rows - 1 > (max_offset - offset - strip) / stride)
        return false;
    slab_bytes = std::max(slab_bytes, strip);
    const size_t rows_per_slab = std::min(rows, 1 + (slab_bytes - strip) / stride);
    const size_t capacity = (rows_per_slab - 1) * stride + strip;
    if (scratch.size() < capacity) scratch.resize(capacity);
    auto * dest = static_cast<uint8_t *>(output);
    for (size_t row = 0; row < rows;)
    {
        const size_t count = std::min(rows_per_slab, rows - row);
        const size_t span = (count - 1) * stride + strip;
        const uint64_t start = offset + uint64_t(row) * stride;
#if defined(_WIN32)
        if (_fseeki64(file, (int64_t) start, SEEK_SET) != 0) return false;
#else
        if (fseeko(file, (off_t) start, SEEK_SET) != 0) return false;
#endif
        // The last slab ends at the final strip, not at the end of its source
        // row: a slice of the final tensor can end exactly at checkpoint EOF.
        if (fread(scratch.data(), 1, span, file) != span) return false;
        for (size_t i = 0; i < count; ++i)
            std::memcpy(dest + (row + i) * strip, scratch.data() + i * stride, strip);
        row += count;
    }
    return true;
}

} // namespace tsg_glm
