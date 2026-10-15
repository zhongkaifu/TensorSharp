// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Memory;

namespace TensorSharp.GGML;

public enum GgmlWeightStreamingArithmetic { FullPrecision = 0, ResidentCuda = 1 }

/// <summary>A bounded CUDA Q8_0 or F16 projection workspace, with no weight-key cache or
/// graph capture. The caller owns packed host input [tokens,K], a reusable
/// weight tile [rows,K], and output tile [tokens,rows]. Calls are synchronous.
/// PayloadBytes accounts for the aligned input/weight/output CUDA allocation;
/// CUDA context/stream metadata and caller host storage need separate headroom.
/// Dispose before releasing the corresponding budget. Failed disposal can be
/// retried and retains both the native handle and its reservation.</summary>
public sealed class GgmlWeightStreamingSession : IDisposable
{
    private readonly object _gate = new();
    private BudgetReservation? _reservation;
    private IntPtr _handle;
    private bool _ready;
    public long PayloadBytes { get; }
    public int MaxTileRows { get; }
    public int MaxTokenCount { get; }
    public int TokenCount { get; private set; }
    public long InputWidth { get; }
    public int WeightType { get; }
    public GgmlWeightStreamingArithmetic Arithmetic { get; }

    public static long GetPayloadBytes(int weightType, long inputWidth, int maxTileRows, int tokenCount,
        GgmlWeightStreamingArithmetic arithmetic = GgmlWeightStreamingArithmetic.FullPrecision,
        int logicalTokenCount = 0, long logicalOutputRows = 0, int rank = 0)
    {
        if (weightType != 1 && weightType != 8) throw new ArgumentOutOfRangeException(nameof(weightType), "Only GGML F16 (1) and Q8_0 (8) are supported.");
        if (inputWidth <= 0 || inputWidth > int.MaxValue || (weightType == 8 && inputWidth % 32 != 0))
            throw new ArgumentOutOfRangeException(nameof(inputWidth), "Input width must be positive; Q8_0 requires a multiple of 32.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(maxTileRows);
        if (tokenCount <= 0 || tokenCount > 65535 * 8) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        if (!Enum.IsDefined(arithmetic)) throw new ArgumentOutOfRangeException(nameof(arithmetic));
        ArgumentOutOfRangeException.ThrowIfNegative(rank);
        logicalTokenCount = logicalTokenCount == 0 ? tokenCount : logicalTokenCount;
        logicalOutputRows = logicalOutputRows == 0 ? maxTileRows : logicalOutputRows;
        if (logicalTokenCount < tokenCount || logicalOutputRows < maxTileRows) throw new ArgumentOutOfRangeException(nameof(logicalTokenCount));
        long result = GgmlNative.TSGgml_WeightStreamingPayloadBytesEx(weightType, inputWidth, maxTileRows, tokenCount,
            rank, (int)arithmetic, logicalTokenCount, logicalOutputRows);
        if (result <= 0) throw Failure("size calculation");
        return result;
    }

    public GgmlWeightStreamingSession(MemoryBudget budget, IEnumerable<string> devicePools, int rank, int weightType,
        long inputWidth, int maxTileRows, int tokenCount, IntPtr hostInput,
        GgmlWeightStreamingArithmetic arithmetic = GgmlWeightStreamingArithmetic.FullPrecision,
        int logicalTokenCount = 0, long logicalOutputRows = 0)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ArgumentNullException.ThrowIfNull(devicePools);
        ArgumentOutOfRangeException.ThrowIfNegative(rank);
        if (hostInput == IntPtr.Zero) throw new ArgumentException("A packed F32 input buffer is required.", nameof(hostInput));
        string[] pools = devicePools.ToArray();
        if (pools.Length == 0 || pools.Distinct(StringComparer.Ordinal).Count() != pools.Length)
            throw new ArgumentException("Map the device to at least one distinct budget pool.", nameof(devicePools));
        WeightType = weightType; InputWidth = inputWidth; MaxTileRows = maxTileRows;
        MaxTokenCount = tokenCount; TokenCount = tokenCount;
        Arithmetic = arithmetic;
        logicalTokenCount = logicalTokenCount == 0 ? tokenCount : logicalTokenCount;
        logicalOutputRows = logicalOutputRows == 0 ? maxTileRows : logicalOutputRows;
        PayloadBytes = GetPayloadBytes(weightType, inputWidth, maxTileRows, tokenCount, arithmetic, logicalTokenCount, logicalOutputRows, rank);
        _reservation = budget.Reserve(pools.Select(pool => new MemoryCharge(pool, PayloadBytes)));
        try
        {
            if (GgmlNative.TSGgml_WeightStreamingCreateEx(rank, weightType, inputWidth, maxTileRows, tokenCount,
                (int)arithmetic, logicalTokenCount, logicalOutputRows,
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
                throw new GgmlWeightStreamingAllocationException(this, new AggregateException(creation, cleanup));
            }
            throw;
        }
    }

    /// <summary>Synchronously replace the packed F32 input without changing the
    /// workspace allocation or its budget reservation. The caller may reuse the
    /// host buffer after return. TokenCount changes only after a successful upload;
    /// subsequent output tiles contain [TokenCount,rows] elements.</summary>
    public void UploadInput(IntPtr hostInput, int tokenCount)
    {
        if (hostInput == IntPtr.Zero) throw new ArgumentException("A packed F32 input buffer is required.", nameof(hostInput));
        if (tokenCount <= 0 || tokenCount > MaxTokenCount) throw new ArgumentOutOfRangeException(nameof(tokenCount));
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_handle == IntPtr.Zero, this);
            if (!_ready) throw new InvalidOperationException("The streaming session failed; dispose it before retrying with a new session.");
            if (GgmlNative.TSGgml_WeightStreamingUploadInput(_handle, hostInput, tokenCount) != 1)
            {
                _ready = false;
                throw Failure("input upload");
            }
            TokenCount = tokenCount;
        }
    }

    /// <summary>Project one packed Q8_0 or F16 output-row tile. Host buffers may be reused
    /// when this method returns. Scatter each token's rows into the full output
    /// on the caller side. Apply weight Scale/bias once after projection.</summary>
    public void Execute(IntPtr tile, int rowCount, IntPtr hostOutput)
    {
        if (tile == IntPtr.Zero) throw new ArgumentException("A packed weight tile is required.", nameof(tile));
        if (hostOutput == IntPtr.Zero) throw new ArgumentException("An output tile buffer is required.", nameof(hostOutput));
        if (rowCount <= 0 || rowCount > MaxTileRows) throw new ArgumentOutOfRangeException(nameof(rowCount));
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_handle == IntPtr.Zero, this);
            if (!_ready) throw new InvalidOperationException("The streaming session failed; dispose it before retrying with a new session.");
            if (GgmlNative.TSGgml_WeightStreamingExecute(_handle, tile, rowCount, hostOutput) != 1)
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
                if (GgmlNative.TSGgml_WeightStreamingDestroy(_handle) != 1) throw Failure("release (ownership retained; retry Dispose)");
                _handle = IntPtr.Zero;
            }
            _reservation?.Dispose();
            _reservation = null;
        }
    }

    private static InvalidOperationException Failure(string operation)
        => new($"GGML weight streaming {operation} failed: {GgmlNative.LastNativeError()}");
}

/// <summary>Construction failed and native cleanup could not prove physical
/// release. This exception retains the session and charged reservation; retry
/// UnreleasedSession.Dispose rather than returning that credit to admission.</summary>
public sealed class GgmlWeightStreamingAllocationException : InvalidOperationException
{
    public GgmlWeightStreamingSession UnreleasedSession { get; }
    internal GgmlWeightStreamingAllocationException(GgmlWeightStreamingSession session, Exception inner)
        : base("Weight streaming construction failed with unreleased native resources. Retry disposal of UnreleasedSession.", inner)
        => UnreleasedSession = session;
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    internal static partial long TSGgml_WeightStreamingPayloadBytesEx(int weightType, long inner, int rows, int columns,
        int rank, int arithmetic, int logicalColumns, long logicalRows);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_WeightStreamingCreateEx(int rank, int weightType, long inner, int rows, int columns,
        int arithmetic, int logicalColumns, long logicalRows, IntPtr input, long capacity, out IntPtr handle);
    [LibraryImport(DllName)]
    internal static partial long TSGgml_WeightStreamingPayloadBytes(int weightType, long inner, int rows, int columns);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_WeightStreamingCreate(int rank, int weightType, long inner, int rows, int columns,
        IntPtr input, long capacity, out IntPtr handle);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_WeightStreamingUploadInput(IntPtr handle, IntPtr input, int columns);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_WeightStreamingExecute(IntPtr handle, IntPtr weights, int rows, IntPtr output);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_WeightStreamingDestroy(IntPtr handle);
}
