// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Memory;

namespace TensorSharp.GGML;

/// <summary>A bounded CUDA Q8_0 projection workspace, with no weight-key cache or
/// graph capture. The caller owns packed host input [tokens,K], a reusable Q8
/// weight tile [rows,K], and output tile [tokens,rows]. Calls are synchronous.
/// PayloadBytes accounts for the aligned input/weight/output CUDA allocation;
/// CUDA context/stream metadata and caller host storage need separate headroom.
/// Dispose before releasing the corresponding budget. Failed disposal can be
/// retried and retains both the native handle and its reservation.</summary>
public sealed class GgmlQ8StreamingSession : IDisposable
{
    private readonly object _gate = new();
    private BudgetReservation? _reservation;
    private IntPtr _handle;
    private bool _ready;
    public long PayloadBytes { get; }
    public int MaxTileRows { get; }
    public int TokenCount { get; }
    public long InputWidth { get; }

    public static long GetPayloadBytes(long inputWidth, int maxTileRows, int tokenCount)
    {
        if (inputWidth <= 0 || inputWidth > int.MaxValue || inputWidth % 32 != 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth), "Q8_0 input width must be a positive multiple of 32.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTileRows);
        if (tokenCount <= 0 || tokenCount > 65535 * 8) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        long result = GgmlNative.TSGgml_Q8StreamingPayloadBytes(inputWidth, maxTileRows, tokenCount);
        if (result <= 0) throw Failure("size calculation");
        return result;
    }

    public GgmlQ8StreamingSession(MemoryBudget budget, IEnumerable<string> devicePools, int rank,
        long inputWidth, int maxTileRows, int tokenCount, IntPtr hostInput)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ArgumentNullException.ThrowIfNull(devicePools);
        ArgumentOutOfRangeException.ThrowIfNegative(rank);
        if (hostInput == IntPtr.Zero) throw new ArgumentException("A packed F32 input buffer is required.", nameof(hostInput));
        string[] pools = devicePools.ToArray();
        if (pools.Length == 0 || pools.Distinct(StringComparer.Ordinal).Count() != pools.Length)
            throw new ArgumentException("Map the device to at least one distinct budget pool.", nameof(devicePools));
        InputWidth = inputWidth; MaxTileRows = maxTileRows; TokenCount = tokenCount;
        PayloadBytes = GetPayloadBytes(inputWidth, maxTileRows, tokenCount);
        _reservation = budget.Reserve(pools.Select(pool => new MemoryCharge(pool, PayloadBytes)));
        try
        {
            if (GgmlNative.TSGgml_Q8StreamingCreate(rank, inputWidth, maxTileRows, tokenCount,
                hostInput, PayloadBytes, out _handle) != 1)
                throw Failure("creation");
            _reservation.Commit();
            _ready = true;
        }
        catch (Exception creation)
        {
            try { Dispose(); }
            catch (Exception cleanup)
            {
                throw new GgmlQ8StreamingAllocationException(this, new AggregateException(creation, cleanup));
            }
            throw;
        }
    }

    /// <summary>Project one packed Q8_0 output-row tile. Host buffers may be reused
    /// when this method returns. Scatter each token's rows into the full output
    /// on the caller side. Apply weight Scale/bias once after projection.</summary>
    public void Execute(IntPtr tile, int rowCount, IntPtr hostOutput)
    {
        if (tile == IntPtr.Zero) throw new ArgumentException("A packed Q8_0 tile is required.", nameof(tile));
        if (hostOutput == IntPtr.Zero) throw new ArgumentException("An output tile buffer is required.", nameof(hostOutput));
        if (rowCount <= 0 || rowCount > MaxTileRows) throw new ArgumentOutOfRangeException(nameof(rowCount));
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_handle == IntPtr.Zero, this);
            if (!_ready) throw new InvalidOperationException("The streaming session failed; dispose it before retrying with a new session.");
            if (GgmlNative.TSGgml_Q8StreamingExecute(_handle, tile, rowCount, hostOutput) != 1)
            {
                _ready = false;
                throw Failure("projection");
            }
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            _ready = false;
            if (_handle != IntPtr.Zero)
            {
                if (GgmlNative.TSGgml_Q8StreamingDestroy(_handle) != 1) throw Failure("release (ownership retained; retry Dispose)");
                _handle = IntPtr.Zero;
            }
            _reservation?.Dispose();
            _reservation = null;
        }
    }

    private static InvalidOperationException Failure(string operation)
        => new($"GGML Q8 streaming {operation} failed: {GgmlNative.LastNativeError()}");
}

/// <summary>Construction failed and native cleanup could not prove physical
/// release. This exception retains the session and charged reservation; retry
/// UnreleasedSession.Dispose rather than returning that credit to admission.</summary>
public sealed class GgmlQ8StreamingAllocationException : InvalidOperationException
{
    public GgmlQ8StreamingSession UnreleasedSession { get; }
    internal GgmlQ8StreamingAllocationException(GgmlQ8StreamingSession session, Exception inner)
        : base("Q8 streaming construction failed with unreleased native resources. Retry disposal of UnreleasedSession.", inner)
        => UnreleasedSession = session;
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    internal static partial long TSGgml_Q8StreamingPayloadBytes(long inner, int rows, int columns);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_Q8StreamingCreate(int rank, long inner, int rows, int columns,
        IntPtr input, long capacity, out IntPtr handle);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_Q8StreamingExecute(IntPtr handle, IntPtr weights, int rows, IntPtr output);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_Q8StreamingDestroy(IntPtr handle);
}
