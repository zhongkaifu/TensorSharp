// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    [LibraryImport(DllName, EntryPoint = "TSGgml_AttachSharedCacheBudget")]
    internal static partial int AttachSharedCacheBudget(IntPtr context, IntPtr reserve, IntPtr commit, IntPtr release);

    [LibraryImport(DllName, EntryPoint = "TSGgml_AttachSharedCacheBudgetEx")]
    internal static partial int AttachSharedCacheBudgetEx(IntPtr context, IntPtr reserve, IntPtr commit, IntPtr release, int includeGraphBuffers);

    [LibraryImport(DllName, EntryPoint = "TSGgml_DetachSharedCacheBudget")]
    internal static partial int DetachSharedCacheBudget(IntPtr context);

    [LibraryImport(DllName, EntryPoint = "TSGgml_TrimHostMoeExpertCache")]
    internal static partial long TrimHostMoeExpertCache(long targetBytes);
}

public partial class GgmlBasicOps
{
    /// <summary>At a quiescent request boundary, retire least-recently-used expert
    /// graphs until their process-wide accounted bytes are at most targetBytes.
    /// Zero empties this cache. Returns released accounted bytes (including its
    /// workspace allowance), not a driver free-memory measurement. Physical
    /// storage is freed before shared credit; model/KV state and cumulative hit
    /// counters survive. This does not change the configured cache ceiling or
    /// trim other caches. Stop model work on every rank before calling.</summary>
    public static long TrimHostMoeExpertCache(long targetBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetBytes);
        long released = GgmlNative.TrimHostMoeExpertCache(targetBytes);
        if (released < 0)
            throw new InvalidOperationException("GGML expert-cache trim failed: " + GgmlNative.LastNativeError());
        return released;
    }
}
