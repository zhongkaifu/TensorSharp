// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Runtime.InteropServices;

namespace TensorSharp.GGML;

internal static partial class GgmlNative
{
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeExpertCacheBudgetKind")]
    internal static partial int HostMoeExpertCacheBudgetKind();
    [LibraryImport(DllName, EntryPoint = "TSGgml_BeginHostMoeExpertCacheScope")]
    internal static partial int BeginHostMoeExpertCacheScope(long bytes, int maxRows, int layers, out ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_BeginHostMoeExpertCacheScopeWithDecodeRows")]
    internal static partial int BeginHostMoeExpertCacheScopeWithDecodeRows(long bytes, int maxRows, int layers,
        int maxDecodeRows, out ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_BeginHostMoeExpertCacheScopeWithCapacityGrowth")]
    internal static partial int BeginHostMoeExpertCacheScopeWithCapacityGrowth(long bytes, int maxRows, int layers,
        int maxDecodeRows, int enableCapacityGrowth, out ulong token);
    [LibraryImport(DllName, EntryPoint = "TSGgml_UpdateHostMoeExpertCacheScope")]
    internal static partial int UpdateHostMoeExpertCacheScope(ulong token, long bytes, out long released);
    [LibraryImport(DllName, EntryPoint = "TSGgml_TrimHostMoeExpertCacheScope")]
    internal static partial int TrimHostMoeExpertCacheScope(ulong token, long bytes, out long released);
    [LibraryImport(DllName, EntryPoint = "TSGgml_EndHostMoeExpertCacheScope")]
    internal static partial int EndHostMoeExpertCacheScope(ulong token, out long released);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeExpertCacheStats")]
    internal static partial int HostMoeExpertCacheStats(out long accounted, out long budget,
        out long hits, out long misses, out long calls);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeExpertCacheGrowthStats")]
    internal static unsafe partial int HostMoeExpertCacheGrowthStats(long* values, int length);
    [LibraryImport(DllName, EntryPoint = "TSGgml_HostMoeExpertCacheEntryStats")]
    internal static unsafe partial int HostMoeExpertCacheEntryStats(long* values, int rowCapacity, out int count);
}

/// <summary>Process-wide compact expert-cache counters. AccountedBytes includes
/// its graph payload and fixed workspace allowance; it is not physical VRAM.
/// Host file-read staging and vendor/driver pools are excluded. Counters are
/// cumulative until the existing full cache teardown resets them.</summary>
public readonly record struct GgmlHostMoeExpertCacheUsage(long AccountedBytes, long BudgetBytes,
    long Hits, long Misses, long Calls);

/// <summary>One retained cache table. AccountedBytes includes the fixed
/// allowance. ResidentExperts counts valid payload slots, not all allocated
/// capacity; ExpertBytes is the three original quantized projections per slot.</summary>
public readonly record struct GgmlHostMoeExpertCacheEntryUsage(int Layer, int Capacity, int ResidentExperts,
    int PlannedDecodeRows, long AccountedBytes, int LocalExperts, int PooledExpertExtent, long ExpertBytes);

/// <summary>Cumulative growth diagnostics until full cache teardown. Peak bytes
/// count simultaneous old and new payload ownership plus the fixed allowance;
/// vendor/driver storage and whole-board VRAM are not included.</summary>
public readonly record struct GgmlHostMoeExpertCacheGrowthUsage(long Attempts, long Successes, long Refusals,
    long Failures, long UnchangedPlans, long RetainedHotSlots, long CopiedBytes, long PeakCoexistingAccountedBytes);

public partial class GgmlBasicOps
{
    public static void RequireSharedExpertCacheBudgetKind()
    {
        try
        {
            if (GgmlNative.HostMoeExpertCacheBudgetKind() != 4)
                throw new NotSupportedException("Unified concurrent execution requires dedicated selected-expert budget ownership (kind 4).");
        }
        catch (EntryPointNotFoundException error)
        {
            throw new NotSupportedException("Unified concurrent execution requires the matching GgmlOps native library with shared expert-cache ownership.", error);
        }
    }
    public static GgmlHostMoeExpertCacheUsage GetHostMoeExpertCacheUsage()
    {
        if (GgmlNative.HostMoeExpertCacheStats(out long accounted, out long budget,
                out long hits, out long misses, out long calls) != 1)
            throw new InvalidOperationException("GGML expert-cache usage read failed: " + GgmlNative.LastNativeError());
        return new(accounted, budget, hits, misses, calls);
    }

    public static unsafe GgmlHostMoeExpertCacheGrowthUsage GetHostMoeExpertCacheGrowthUsage()
    {
        Span<long> values = stackalloc long[8];
        try
        {
            fixed (long* pointer = values)
                if (GgmlNative.HostMoeExpertCacheGrowthStats(pointer, values.Length) != 1)
                    throw new InvalidOperationException("Expert-cache growth usage read failed: " + GgmlNative.LastNativeError());
        }
        catch (EntryPointNotFoundException error)
        { throw new NotSupportedException("Expert-cache growth diagnostics require the matching native library.", error); }
        return new(values[0], values[1], values[2], values[3], values[4], values[5], values[6], values[7]);
    }

    /// <summary>Diagnostic snapshot, independent of retained weight ownership.
    /// Prefer a quiescent boundary. A concurrent entry-count change retries at
    /// most three times; these reads are not enabled in normal token execution.</summary>
    public static unsafe GgmlHostMoeExpertCacheEntryUsage[] GetHostMoeExpertCacheEntries()
    {
        try
        {
            if (GgmlNative.HostMoeExpertCacheEntryStats(null, 0, out int count) != 1)
                throw new InvalidOperationException("Expert-cache entry count read failed: " + GgmlNative.LastNativeError());
            for (int attempt = 0; attempt < 3; ++attempt)
            {
                if (count == 0) return [];
                if (count is < 0 or > 1048576) throw new InvalidOperationException("Expert-cache entry metadata exceeds its diagnostic bound.");
                int capacity = count;
                long[] values = new long[checked(capacity * 8)];
                int status;
                fixed (long* pointer = values)
                    status = GgmlNative.HostMoeExpertCacheEntryStats(pointer, capacity, out count);
                if (status == 2) continue;
                if (status != 1 || count < 0 || count > capacity)
                    throw new InvalidOperationException("Expert-cache entry read failed: " + GgmlNative.LastNativeError());
                var entries = new GgmlHostMoeExpertCacheEntryUsage[count];
                for (int row = 0; row < entries.Length; ++row)
                {
                    int offset = row * 8;
                    entries[row] = new(checked((int)values[offset]), checked((int)values[offset + 1]),
                        checked((int)values[offset + 2]), checked((int)values[offset + 3]), values[offset + 4],
                        checked((int)values[offset + 5]), checked((int)values[offset + 6]), values[offset + 7]);
                }
                return entries;
            }
            throw new InvalidOperationException("Expert-cache entry count kept changing during the diagnostic snapshot.");
        }
        catch (EntryPointNotFoundException error)
        { throw new NotSupportedException("Expert-cache entry diagnostics require the matching native library.", error); }
    }
}

internal interface IHostMoeExpertCacheScopeNative
{
    ulong Begin(long bytes, int maxRows, int layers, int maxDecodeRows);
    ulong Begin(long bytes, int maxRows, int layers, int maxDecodeRows, bool enableCapacityGrowth)
    {
        if (enableCapacityGrowth) throw new NotSupportedException("The native expert-cache scope does not support capacity growth.");
        return Begin(bytes, maxRows, layers, maxDecodeRows);
    }
    long Update(ulong token, long bytes);
    long Trim(ulong token, long bytes);
    long End(ulong token);
    GgmlHostMoeExpertCacheUsage ReadUsage();
}

/// <summary>Exclusive, process-wide configuration of the optional Qwen4Exp/GLM CUDA
/// compact expert cache. This changes no environment variable or model placement.
/// Install the shared budget first and create this scope before loading the model,
/// with the expert cache empty. Quiesce all model work before changing or releasing
/// this scope. Dispose after model cleanup and before detaching its shared budget.
/// The cache's existing allocation tickets remain active; this is a capacity
/// setting, not a complete VRAM/RAM or driver-memory cap. No finalizer performs GPU
/// work. Failed disposal retains the native token and may be retried.</summary>
public sealed class GgmlHostMoeExpertCacheScope : IDisposable
{
    private readonly object _gate = new();
    private readonly IHostMoeExpertCacheScopeNative _native;
    private readonly ulong _token;
    private long _budgetBytes;
    private bool _disposed;

    public GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount)
        : this(capacityBytes, maxPrefillRows, layerCount, NativeApi.Instance, 1) { }

    /// <summary>Plan exact-width decode variants up to the session's actual
    /// supported concurrency. Variants share the same resident expert weights;
    /// their measured allocation remains inside the cache capacity. GLM's
    /// explicitly installed phase plan permits prefill widths through 1024;
    /// ordinary Qwen compact-prefill policy retains its separate 128-row cap.</summary>
    public GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount, int maxDecodeRows)
        : this(capacityBytes, maxPrefillRows, layerCount, NativeApi.Instance, maxDecodeRows) { }

    /// <summary>Optionally grow retained tables after a larger scoped budget is
    /// admitted. Expansion preserves hot payloads and charges simultaneous old
    /// and new storage; a normal admission refusal keeps the old table. Disabled
    /// by default pending model and performance qualification.</summary>
    public GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount, int maxDecodeRows,
        bool enableCapacityGrowth)
        : this(capacityBytes, maxPrefillRows, layerCount, NativeApi.Instance, maxDecodeRows, enableCapacityGrowth) { }

    internal GgmlHostMoeExpertCacheScope(long capacityBytes, int maxPrefillRows, int layerCount,
        IHostMoeExpertCacheScopeNative native, int maxDecodeRows = 1, bool enableCapacityGrowth = false)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(capacityBytes);
        if (maxPrefillRows is < 1 or > 1024) throw new ArgumentOutOfRangeException(nameof(maxPrefillRows));
        if (maxDecodeRows is < 1 or > 8) throw new ArgumentOutOfRangeException(nameof(maxDecodeRows));
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(layerCount);
        ArgumentNullException.ThrowIfNull(native);
        _native = native;
        MaximumBudgetBytes = capacityBytes;
        MaxPrefillRows = maxPrefillRows;
        MaxDecodeRows = maxDecodeRows;
        LayerCount = layerCount;
        CapacityGrowthEnabled = enableCapacityGrowth;
        // All managed allocations/validation precede the native mutation.
        // There is no managed allocation after a successful Begin.
        _token = native.Begin(capacityBytes, maxPrefillRows, layerCount, maxDecodeRows, enableCapacityGrowth);
        _budgetBytes = capacityBytes;
    }

    /// <summary>The initial ceiling; pressure recovery cannot exceed this value.</summary>
    public long MaximumBudgetBytes { get; }
    public long BudgetBytes { get { lock (_gate) return _budgetBytes; } }
    public int MaxPrefillRows { get; }
    public int MaxDecodeRows { get; }
    public int LayerCount { get; }
    public bool CapacityGrowthEnabled { get; }
    /// <summary>Accounted device-cache bytes, including the fixed allowance;
    /// excludes the separately charged host file-read workspace.</summary>
    public long CachedBytes => ReadUsage().AccountedBytes;
    public GgmlHostMoeExpertCacheUsage ReadUsage()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _native.ReadUsage();
        }
    }

    /// <summary>Set the current ceiling and synchronously evict entries exceeding
    /// the total or per-layer allowance. Zero disables this tier. Later growth is
    /// limited to MaximumBudgetBytes. Returns released accounted device bytes,
    /// not measured driver free memory. On failure the old ceiling remains active,
    /// although successfully retired entries need not be recreated.</summary>
    public long UpdateBudget(long capacityBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(capacityBytes);
        if (capacityBytes > MaximumBudgetBytes) throw new ArgumentOutOfRangeException(nameof(capacityBytes));
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            long released = _native.Update(_token, capacityBytes);
            _budgetBytes = capacityBytes;
            return released;
        }
    }

    /// <summary>Retire LRU entries down to targetBytes without changing the ceiling.
    /// Always releases reusable CPU prefill weight/work staging and its dedicated
    /// CUDA file-stream graph arena at this quiescent
    /// boundary, even when targetBytes leaves every device entry resident.
    /// Zero also releases the host file-read workspace once entries are empty.
    /// Return value counts device cache tickets, excluding released host workspace.</summary>
    public long Trim(long targetBytes)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(targetBytes);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            return _native.Trim(_token, targetBytes);
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            _native.End(_token); // Frees physical storage before restoring old settings.
            _disposed = true; // Do not lose ownership if native cleanup fails.
        }
    }

    private sealed class NativeApi : IHostMoeExpertCacheScopeNative
    {
        public static readonly NativeApi Instance = new();
        private static void Require(int status)
        {
            if (status != 1) throw new InvalidOperationException("GGML expert-cache scope failed: " + GgmlNative.LastNativeError());
        }
        public ulong Begin(long bytes, int maxRows, int layers, int maxDecodeRows)
        {
            try
            {
                ulong token;
                // The scalar API continues to work with older native libraries.
                // A wider plan requires the matching ABI and must never silently
                // return a scope that would rebuild weights on first admission.
                Require(maxDecodeRows == 1
                    ? GgmlNative.BeginHostMoeExpertCacheScope(bytes, maxRows, layers, out token)
                    : GgmlNative.BeginHostMoeExpertCacheScopeWithDecodeRows(bytes, maxRows, layers, maxDecodeRows, out token));
                return token;
            }
            catch (EntryPointNotFoundException error)
            {
                throw new NotSupportedException("Expert-cache configuration scopes require the matching updated GgmlOps native library.", error);
            }
        }
        public ulong Begin(long bytes, int maxRows, int layers, int maxDecodeRows, bool enableCapacityGrowth)
        {
            if (!enableCapacityGrowth) return Begin(bytes, maxRows, layers, maxDecodeRows);
            try
            {
                Require(GgmlNative.BeginHostMoeExpertCacheScopeWithCapacityGrowth(bytes, maxRows, layers,
                    maxDecodeRows, 1, out ulong token));
                return token;
            }
            catch (EntryPointNotFoundException error)
            {
                throw new NotSupportedException("Capacity growth requires the matching updated GgmlOps native library.", error);
            }
        }
        public long Update(ulong token, long bytes)
        { Require(GgmlNative.UpdateHostMoeExpertCacheScope(token, bytes, out long released)); return released; }
        public long Trim(ulong token, long bytes)
        { Require(GgmlNative.TrimHostMoeExpertCacheScope(token, bytes, out long released)); return released; }
        public long End(ulong token)
        { Require(GgmlNative.EndHostMoeExpertCacheScope(token, out long released)); return released; }
        public GgmlHostMoeExpertCacheUsage ReadUsage() => GgmlBasicOps.GetHostMoeExpertCacheUsage();
    }
}
