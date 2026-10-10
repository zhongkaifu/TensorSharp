// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;

namespace TensorSharp.GGML;

public partial class GgmlBasicOps
{
    /// <summary>Read one initialized rank's native cache payload accounting.
    /// Returns false for an invalid rank or an unavailable/older native library.
    /// Graph arenas, KV slots, backend pools, allocator rounding and driver
    /// overhead are not included; compare with device measurements separately.
    /// Explicit preloads retain their own policy outside the lazy-copy quota.</summary>
    public static bool TryGetCacheMemoryUsage(int rank, out GgmlCacheMemoryUsage usage)
    {
        usage = default;
        if (rank < 0) return false;
        try { return GgmlNative.TryGetCacheMemoryUsage(rank, out usage); }
        catch (EntryPointNotFoundException) { return false; }
        catch (DllNotFoundException) { return false; }
    }
}
