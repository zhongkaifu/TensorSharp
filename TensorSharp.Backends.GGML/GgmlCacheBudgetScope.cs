// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Memory;

namespace TensorSharp.GGML;

/// <summary>Opt-in, process-wide budget ownership for new GGML lazy device-copy
/// and explicit-preload cache allocations. Install before those allocations exist.
/// Each rank maps to one or more existing budget pools; listing RAM and GPU for
/// UMA applies two constraints to one allocation, not two physical copies.
/// Graph scratch, live KV outside these caches, host-pointer wrappers, backend
/// pools, allocator rounding and driver overhead are not covered. Cache refusal
/// can use unbudgeted per-graph streaming; this is not a whole-model memory cap.
/// Dispose only after model work has stopped and its caches have been cleared.
/// Failed disposal keeps callbacks rooted and accounting active for a retry.</summary>
public sealed class GgmlCacheBudgetScope : IDisposable
{
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate ulong ReserveCallback(IntPtr context, int rank, int kind, long bytes);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate int CommitCallback(IntPtr context, ulong token);
    [UnmanagedFunctionPointer(CallingConvention.Cdecl)]
    private delegate void ReleaseCallback(IntPtr context, ulong token);
    private static readonly ReserveCallback ReserveRoot = ReserveNative;
    private static readonly CommitCallback CommitRoot = CommitNative;
    private static readonly ReleaseCallback ReleaseRoot = ReleaseNative;
    private readonly object _gate = new();
    private readonly object _lifecycleGate = new();
    private readonly MemoryBudget _budget;
    private readonly string[][] _rankPools;
    private readonly Dictionary<ulong, BudgetReservation> _allocations = new();
    private GCHandle _handle;
    private ulong _nextToken;
    private Exception? _callbackError;
    private bool _disposed;

    public GgmlCacheBudgetScope(MemoryBudget budget, IEnumerable<IEnumerable<string>> rankPools)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ArgumentNullException.ThrowIfNull(rankPools);
        _budget = budget;
        _rankPools = rankPools.Select(pools => pools?.ToArray()
            ?? throw new ArgumentException("Every rank needs a pool mapping.", nameof(rankPools))).ToArray();
        if (_rankPools.Length == 0 || _rankPools.Any(p => p.Length == 0 || p.Distinct(StringComparer.Ordinal).Count() != p.Length))
            throw new ArgumentException("Map each rank to one or more distinct budget pools.", nameof(rankPools));
        foreach (var pools in _rankPools)
            budget.CanEverFit(pools.Select(pool => new MemoryCharge(pool, 0))); // Validate names before native registration.
        _handle = GCHandle.Alloc(this);
        try
        {
            if (GgmlNative.AttachSharedCacheBudget(GCHandle.ToIntPtr(_handle),
                Marshal.GetFunctionPointerForDelegate(ReserveRoot), Marshal.GetFunctionPointerForDelegate(CommitRoot),
                Marshal.GetFunctionPointerForDelegate(ReleaseRoot)) != 1)
                throw new InvalidOperationException("Install the GGML cache budget before cache allocations, with no other active cache budget scope.");
        }
        catch
        {
            _handle.Free();
            throw;
        }
    }

    /// <summary>Unexpected callback failures fail admission closed and never cross
    /// the unmanaged boundary. Ordinary quota rejection is not an error.</summary>
    public Exception? CallbackError { get { lock (_gate) return _callbackError; } }
    public int ActiveAllocations { get { lock (_gate) return _allocations.Count; } }

    public void Dispose()
    {
        // Do not hold _gate while calling native: a callback can already own the
        // native registry gate while it waits for this scope's accounting lock.
        lock (_lifecycleGate)
        {
            if (_disposed) return;
            lock (_gate)
                if (_allocations.Count != 0)
                    throw new InvalidOperationException("GGML cache allocations still own budget credit. Stop model work and clear its caches before disposing the budget scope.");
            if (GgmlNative.DetachSharedCacheBudget(GCHandle.ToIntPtr(_handle)) != 1)
                throw new InvalidOperationException("A GGML cache allocation is still in flight. Retry disposal after model work and cache cleanup finish.");
            _handle.Free(); // Native detach proves no callback can still use it.
            _disposed = true;
        }
    }

    private static GgmlCacheBudgetScope Owner(IntPtr context) => (GgmlCacheBudgetScope)GCHandle.FromIntPtr(context).Target!;
    private static ulong ReserveNative(IntPtr context, int rank, int kind, long bytes)
    {
        var owner = Owner(context);
        lock (owner._gate)
        {
            if (owner._callbackError != null) return 0;
            BudgetReservation? reservation = null;
            try
            {
                if ((uint)rank >= (uint)owner._rankPools.Length || bytes <= 0 || (kind != 0 && kind != 1)) return 0;
                reservation = owner._budget.TryReserve(owner._rankPools[rank].Select(pool => new MemoryCharge(pool, bytes)));
                if (reservation == null) return 0;
                ulong token = checked(++owner._nextToken);
                owner._allocations.Add(token, reservation);
                return token;
            }
            catch (Exception ex)
            {
                owner._callbackError ??= ex;
                try { reservation?.Dispose(); } catch (Exception cleanup) { owner._callbackError = cleanup; }
                return 0;
            }
        }
    }
    private static int CommitNative(IntPtr context, ulong token)
    {
        var owner = Owner(context);
        lock (owner._gate)
        {
            try { owner._allocations[token].Commit(); return 1; }
            catch (Exception ex) { owner._callbackError ??= ex; return 0; }
        }
    }
    private static void ReleaseNative(IntPtr context, ulong token)
    {
        var owner = Owner(context);
        lock (owner._gate)
        {
            try
            {
                if (!owner._allocations.TryGetValue(token, out var reservation)) return;
                reservation.Dispose();
                owner._allocations.Remove(token);
            }
            catch (Exception ex) { owner._callbackError ??= ex; }
        }
    }
}
