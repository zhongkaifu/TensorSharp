// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Memory;

namespace TensorSharp.GGML;

/// <summary>Instrumented payload ownership, counted once even with multiple
/// constraint pools. Pending bytes are reservations, not proven allocations.</summary>
public sealed record GgmlAllocationUsage(int Rank, int Kind, int Allocations, long PendingBytes, long CommittedBytes);

/// <summary>A refused allocation is ordinary pressure, separate from a callback
/// error. RemainingEnvelope is null outside a request. Observations are diagnostic,
/// not reusable capacity; other owners can change immediately after the callback.</summary>
public sealed record GgmlAllocationRefusal(int Rank, int Kind, long Bytes,
    IReadOnlyList<MemoryCharge>? RemainingEnvelope, IReadOnlyList<MemoryPoolSnapshot> Budget);

/// <summary>Opt-in, process-wide budget ownership for new GGML lazy device-copy
/// and explicit-preload cache allocations. Install before those allocations exist.
/// Each rank maps to one or more existing budget pools; listing RAM and GPU for
/// UMA applies two constraints to one allocation, not two physical copies.
/// With includeGraphBuffers=true, additionally charge the explicitly routed
/// TensorSharp-owned context buffers and shared reuse graph allocators, including
/// Qwen4Exp graph arenas, recurrent state and device state snapshots, Qwen-Image
/// retained graphs/prefix storage, and MiniMax-H3 execution buffers. Buffers
/// retain their original native backend interfaces. Other allocator paths, live KV
/// outside these buffers, host-pointer wrappers, backend pools, allocator rounding
/// and driver overhead are not covered; this is not a whole-model memory cap.
/// Cache-only refusal can use unbudgeted per-graph streaming. Dispose only after
/// model work has stopped and all covered caches/graphs have been released.
/// Supplying hostPools additionally accounts for the compact expert file-read
/// arena and DeepSeek demand-read staging against the same budget, independently
/// of rank/device pool mappings.
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
    private readonly string[] _hostPools;
    private sealed class Allocation(BudgetReservation reservation, int rank, int kind, long bytes)
    {
        public readonly BudgetReservation Reservation = reservation;
        public readonly int Rank = rank, Kind = kind;
        public readonly long Bytes = bytes;
        public bool Committed;
    }
    private readonly Dictionary<ulong, Allocation> _allocations = new();
    private GCHandle _handle;
    private ulong _nextToken;
    private Exception? _callbackError;
    private bool _disposed;
    private bool _disposing;
    private BudgetReservation? _executionEnvelope;
    private long _allocationRefusalCount;
    private GgmlAllocationRefusal? _firstAllocationRefusal, _lastAllocationRefusal;

    /// <summary>Bounded failure telemetry: no growing event log or successful-allocation sampling.</summary>
    public (long Count, GgmlAllocationRefusal? First, GgmlAllocationRefusal? Last) AllocationRefusals
    { get { lock (_gate) return (_allocationRefusalCount, _firstAllocationRefusal, _lastAllocationRefusal); } }

    public GgmlCacheBudgetScope(MemoryBudget budget, IEnumerable<IEnumerable<string>> rankPools)
        : this(budget, rankPools, includeGraphBuffers: false) { }

    public GgmlCacheBudgetScope(MemoryBudget budget, IEnumerable<IEnumerable<string>> rankPools, bool includeGraphBuffers)
        : this(budget, rankPools, includeGraphBuffers, hostPools: null) { }

    /// <summary>Also charge explicitly routed native host staging to the supplied
    /// host pools. Null excludes these buffers, preserving the legacy contract.
    /// This mapping is process-wide, independent of device-rank pools. It covers
    /// the compact expert file arena and DeepSeek demand-read staging,
    /// not mmap residency or all host allocations.</summary>
    public GgmlCacheBudgetScope(MemoryBudget budget, IEnumerable<IEnumerable<string>> rankPools,
        bool includeGraphBuffers, IEnumerable<string>? hostPools)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ArgumentNullException.ThrowIfNull(rankPools);
        _budget = budget;
        IncludesGraphBuffers = includeGraphBuffers;
        _hostPools = hostPools?.ToArray() ?? [];
        if (hostPools != null && (_hostPools.Length == 0 || _hostPools.Distinct(StringComparer.Ordinal).Count() != _hostPools.Length))
            throw new ArgumentException("Map host staging to one or more distinct pools, or use null to exclude it.", nameof(hostPools));
        if (_hostPools.Length > 0)
            budget.CanEverFit(_hostPools.Select(pool => new MemoryCharge(pool, 0)));
        _rankPools = rankPools.Select(pools => pools?.ToArray()
            ?? throw new ArgumentException("Every rank needs a pool mapping.", nameof(rankPools))).ToArray();
        if (_rankPools.Length == 0 || _rankPools.Any(p => p.Length == 0 || p.Distinct(StringComparer.Ordinal).Count() != p.Length))
            throw new ArgumentException("Map each rank to one or more distinct budget pools.", nameof(rankPools));
        foreach (var pools in _rankPools)
            budget.CanEverFit(pools.Select(pool => new MemoryCharge(pool, 0))); // Validate names before native registration.
        _handle = GCHandle.Alloc(this);
        try
        {
            IntPtr context = GCHandle.ToIntPtr(_handle);
            IntPtr reserve = Marshal.GetFunctionPointerForDelegate(ReserveRoot);
            IntPtr commit = Marshal.GetFunctionPointerForDelegate(CommitRoot);
            IntPtr release = Marshal.GetFunctionPointerForDelegate(ReleaseRoot);
            int attached = IncludesHostBuffers
                ? GgmlNative.AttachSharedCacheBudgetWithHost(context, reserve, commit, release, includeGraphBuffers ? 1 : 0)
                : includeGraphBuffers
                ? GgmlNative.AttachSharedCacheBudgetEx(context, reserve, commit, release, 1)
                : GgmlNative.AttachSharedCacheBudget(context, reserve, commit, release);
            if (attached != 1)
                throw new InvalidOperationException("Install the GGML budget before covered cache/graph allocations, with no other active budget scope.");
        }
        catch (Exception error)
        {
            _handle.Free();
            if (IncludesHostBuffers && error is EntryPointNotFoundException)
                throw new NotSupportedException(
                    "Expert host-buffer budgeting requires a native library with TSGgml_AttachSharedCacheBudgetWithHost. Update the native library together with the managed assemblies.", error);
            throw;
        }
    }

    /// <summary>Unexpected callback failures fail admission closed and never cross
    /// the unmanaged boundary. Ordinary quota rejection is not an error.</summary>
    public Exception? CallbackError { get { lock (_gate) return _callbackError; } }
    public int ActiveAllocations { get { lock (_gate) return _allocations.Count; } }
    /// <summary>Kind 0: lazy device copies; 1: preloaded weights; 2: graph/context
    /// buffers; 3: native host staging. Excludes uninstrumented driver/OS memory.</summary>
    public IReadOnlyList<GgmlAllocationUsage> AllocationUsage
    {
        get
        {
            lock (_gate) return _allocations.Values.GroupBy(a => (a.Rank, a.Kind))
                .Select(g => new GgmlAllocationUsage(g.Key.Rank, g.Key.Kind, g.Count(),
                    g.Where(a => !a.Committed).Sum(a => a.Bytes), g.Where(a => a.Committed).Sum(a => a.Bytes)))
                .OrderBy(a => a.Rank).ThenBy(a => a.Kind).ToArray();
        }
    }
    public bool IncludesGraphBuffers { get; }
    public bool IncludesHostBuffers => _hostPools.Length != 0;

    /// <summary>Route synchronous model execution allocations through an admitted
    /// request, including callbacks on native worker threads. The caller must
    /// hold the model compute lock and may not overlap execution or await.</summary>
    public IDisposable EnterExecution(BudgetReservation envelope)
    {
        ArgumentNullException.ThrowIfNull(envelope);
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_disposing) throw new InvalidOperationException("The GGML budget is being detached.");
            if (!ReferenceEquals(envelope.Budget, _budget)) throw new ArgumentException("Request belongs to a different budget.", nameof(envelope));
            if (_executionEnvelope != null) throw new InvalidOperationException("Concurrent or nested GGML execution envelopes are unsupported.");
            _executionEnvelope = envelope;
            return new Execution(this);
        }
    }

    private sealed class Execution(GgmlCacheBudgetScope owner) : IDisposable
    {
        private bool _disposed;
        public void Dispose() { lock (owner._gate) { if (_disposed) return; owner._executionEnvelope = null; _disposed = true; } }
    }

    public void Dispose()
    {
        // Do not hold _gate while calling native: a callback can already own the
        // native registry gate while it waits for this scope's accounting lock.
        lock (_lifecycleGate)
        {
            if (_disposed) return;
            lock (_gate)
            {
                if (_allocations.Count != 0 || _executionEnvelope != null)
                    throw new InvalidOperationException("GGML allocations still own budget credit. Stop model work and release covered caches/graphs before disposing the budget scope.");
                _disposing = true;
            }
            try
            {
                if (GgmlNative.DetachSharedCacheBudget(GCHandle.ToIntPtr(_handle)) != 1)
                    throw new InvalidOperationException("A covered GGML allocation is still in flight. Retry disposal after model work and cleanup finish.");
                _handle.Free(); // Native detach proves no callback can still use it.
                lock (_gate) _disposed = true;
            }
            finally { lock (_gate) _disposing = false; }
        }
    }

    private static GgmlCacheBudgetScope Owner(IntPtr context) => (GgmlCacheBudgetScope)GCHandle.FromIntPtr(context).Target!;
    private static ulong ReserveNative(IntPtr context, int rank, int kind, long bytes)
    {
        var owner = Owner(context);
        lock (owner._gate)
        {
            if (owner._callbackError != null || owner._disposing || owner._disposed) return 0;
            BudgetReservation? reservation = null;
            try
            {
                if (bytes <= 0) return 0;
                string[] pools;
                if (kind == 3 && owner.IncludesHostBuffers)
                    pools = owner._hostPools;
                else if ((uint)rank < (uint)owner._rankPools.Length
                    && (kind == 0 || kind == 1 || (kind == 2 && owner.IncludesGraphBuffers)))
                    pools = owner._rankPools[rank];
                else return 0;
                var charges = pools.Select(pool => new MemoryCharge(pool, bytes));
                reservation = owner._executionEnvelope == null ? owner._budget.TryReserve(charges)
                    : owner._executionEnvelope.TryTake(charges);
                if (reservation == null)
                {
                    var refusal = new GgmlAllocationRefusal(rank, kind, bytes,
                        owner._executionEnvelope?.Charges, owner._budget.Snapshot());
                    owner._allocationRefusalCount++;
                    owner._firstAllocationRefusal ??= refusal;
                    owner._lastAllocationRefusal = refusal;
                    return 0;
                }
                ulong token = checked(++owner._nextToken);
                owner._allocations.Add(token, new(reservation, rank, kind, bytes));
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
            try
            {
                var allocation = owner._allocations[token];
                allocation.Reservation.Commit();
                allocation.Committed = true;
                return 1;
            }
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
                reservation.Reservation.Dispose();
                owner._allocations.Remove(token);
            }
            catch (Exception ex) { owner._callbackError ??= ex; }
        }
    }
}
