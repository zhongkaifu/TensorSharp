// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Collections.Generic;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.Memory;

namespace TensorSharp.GGML;

/// <summary>Temporarily holds one complete logical Q8_0/F16 matrix and its input/output
/// on CUDA so the unchanged resident MMQ/cuBLAS shape and reduction order are preserved.
/// Host transfers can be split into bounded consecutive tiles. The complete CUDA
/// input, weights, output, quantization and fixup payload is reserved before allocation;
/// caller host storage and CUDA context/stream metadata need separate headroom.
/// No host-pointer cache or persistent model-weight allocation is used.
/// Dispose retains ownership and its reservation if native release fails.</summary>
public sealed class GgmlResidentWeightSession : IDisposable
{
    private readonly object _gate = new();
    private BudgetReservation? _reservation;
    private IntPtr _handle;
    private bool _ready, _projected;
    private int _uploadedRows, _uploadedTokens;

    public long PayloadBytes { get; }
    public long InputWidth { get; }
    public int OutputRows { get; }
    public int TokenCount { get; }
    public int WeightType { get; }

    public static long GetPayloadBytes(int rank, int weightType, long inputWidth, int outputRows, int tokenCount)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(rank);
        if (weightType != 8 && weightType != 1) throw new ArgumentOutOfRangeException(nameof(weightType), "Only Q8_0 (8) and F16 (1) are supported.");
        if (inputWidth <= 0 || inputWidth > int.MaxValue || (weightType == 8 && inputWidth % 32 != 0))
            throw new ArgumentOutOfRangeException(nameof(inputWidth), "Input width must be positive; Q8_0 requires a multiple of 32.");
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(outputRows);
        if (tokenCount <= (weightType == 8 ? 8 : 16) || tokenCount > 65535)
            throw new ArgumentOutOfRangeException(nameof(tokenCount), "Complete resident-shape streaming requires Q8_0 N>8 or F16 N>16, at most 65535 tokens.");
        long bytes = GgmlNative.TSGgml_ResidentWeightPayloadBytes(rank, weightType, inputWidth, outputRows, tokenCount);
        if (bytes <= 0) throw Failure("size calculation");
        return bytes;
    }

    public GgmlResidentWeightSession(MemoryBudget budget, IEnumerable<string> devicePools,
        int rank, int weightType, long inputWidth, int outputRows, int tokenCount)
    {
        ArgumentNullException.ThrowIfNull(budget);
        ArgumentNullException.ThrowIfNull(devicePools);
        string[] pools = devicePools.ToArray();
        if (pools.Length == 0 || pools.Distinct(StringComparer.Ordinal).Count() != pools.Length)
            throw new ArgumentException("Map the device to at least one distinct budget pool.", nameof(devicePools));
        WeightType = weightType; InputWidth = inputWidth; OutputRows = outputRows; TokenCount = tokenCount;
        PayloadBytes = GetPayloadBytes(rank, weightType, inputWidth, outputRows, tokenCount);
        _reservation = budget.Reserve(pools.Select(pool => new MemoryCharge(pool, PayloadBytes)));
        try
        {
            if (GgmlNative.TSGgml_ResidentWeightCreate(rank, weightType, inputWidth, outputRows, tokenCount,
                    PayloadBytes, out _handle) != 1)
                throw Failure("creation");
            _reservation.Commit();
            _ready = true;
        }
        catch (Exception creation)
        {
            try { Dispose(); }
            catch (Exception cleanup)
            {
                throw new GgmlResidentWeightAllocationException(this, new AggregateException(creation, cleanup));
            }
            throw;
        }
    }

    /// <summary>Synchronously copy packed Q8_0/F16 rows. Upload the complete logical
    /// matrix once, consecutively from firstRow=0. The host buffer is reusable on return.</summary>
    public void UploadWeightRows(IntPtr hostRows, int firstRow, int rowCount)
    {
        if (hostRows == IntPtr.Zero) throw new ArgumentException("Packed weight rows are required.", nameof(hostRows));
        CheckRange(firstRow, rowCount, OutputRows, nameof(firstRow), nameof(rowCount));
        lock (_gate)
        {
            CheckReady();
            if (firstRow != _uploadedRows) throw new ArgumentException("Weight rows must be uploaded once, consecutively from zero.", nameof(firstRow));
            if (GgmlNative.TSGgml_ResidentWeightUploadRows(_handle, hostRows, firstRow, rowCount) != 1)
            {
                _ready = false;
                throw Failure("weight upload");
            }
            _uploadedRows += rowCount;
        }
    }

    /// <summary>Synchronously copy packed F32 [tokenCount,K] input. Upload all N
    /// tokens consecutively. Starting at firstToken=0 begins a new input and
    /// invalidates any previous output while retaining the loaded weights.</summary>
    public void UploadInputTokens(IntPtr hostInput, int firstToken, int tokenCount)
    {
        if (hostInput == IntPtr.Zero) throw new ArgumentException("Packed F32 input tokens are required.", nameof(hostInput));
        CheckRange(firstToken, tokenCount, TokenCount, nameof(firstToken), nameof(tokenCount));
        lock (_gate)
        {
            CheckReady();
            if (firstToken != 0 && firstToken != _uploadedTokens)
                throw new ArgumentException("Input tokens must be uploaded consecutively from zero.", nameof(firstToken));
            if (GgmlNative.TSGgml_ResidentWeightUploadInput(_handle, hostInput, firstToken, tokenCount) != 1)
            {
                _ready = false;
                throw Failure("input upload");
            }
            _uploadedTokens = firstToken + tokenCount;
            _projected = false;
        }
    }

    /// <summary>Compute the complete original M/N projection once after all rows
    /// and tokens have arrived. Repeating this call reuses the completed output.</summary>
    public void Project()
    {
        lock (_gate)
        {
            CheckReady();
            if (_uploadedRows != OutputRows || _uploadedTokens != TokenCount)
                throw new InvalidOperationException("Projection requires every original weight row and input token.");
            if (_projected) return;
            if (GgmlNative.TSGgml_ResidentWeightProject(_handle) != 1)
            {
                _ready = false;
                throw Failure("projection");
            }
            _projected = true;
        }
    }

    /// <summary>Synchronously read a completed output rectangle as packed
    /// [tokenCount,rowCount] F32. The caller provides that exact buffer capacity.</summary>
    public void Download(IntPtr hostOutput, int firstToken, int tokenCount, int firstRow, int rowCount)
    {
        if (hostOutput == IntPtr.Zero) throw new ArgumentException("A packed F32 output buffer is required.", nameof(hostOutput));
        CheckRange(firstToken, tokenCount, TokenCount, nameof(firstToken), nameof(tokenCount));
        CheckRange(firstRow, rowCount, OutputRows, nameof(firstRow), nameof(rowCount));
        lock (_gate)
        {
            CheckReady();
            if (!_projected) throw new InvalidOperationException("Download requires a completed projection of the current input.");
            if (GgmlNative.TSGgml_ResidentWeightDownload(_handle, hostOutput, firstToken, tokenCount, firstRow, rowCount) != 1)
            {
                _ready = false;
                throw Failure("output download");
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
                if (GgmlNative.TSGgml_WeightStreamingDestroy(_handle) != 1)
                    throw Failure("release (ownership retained; retry Dispose)");
                _handle = IntPtr.Zero;
            }
            _reservation?.Dispose();
            _reservation = null;
        }
    }

    private void CheckReady()
    {
        ObjectDisposedException.ThrowIf(_handle == IntPtr.Zero, this);
        if (!_ready) throw new InvalidOperationException("The resident weight session failed; dispose it before retrying with a new session.");
    }
    private static void CheckRange(int first, int count, int capacity, string firstName, string countName)
    {
        if (first < 0 || first >= capacity) throw new ArgumentOutOfRangeException(firstName);
        if (count <= 0 || count > capacity - first) throw new ArgumentOutOfRangeException(countName);
    }
    private static InvalidOperationException Failure(string operation)
        => new($"GGML resident weight streaming {operation} failed: {GgmlNative.LastNativeError()}");
}

/// <summary>Construction failed and physical release could not be confirmed.
/// The retained session still owns its reservation; retry its Dispose.</summary>
public sealed class GgmlResidentWeightAllocationException : InvalidOperationException
{
    public GgmlResidentWeightSession UnreleasedSession { get; }
    internal GgmlResidentWeightAllocationException(GgmlResidentWeightSession session, Exception inner)
        : base("Resident weight streaming construction failed with unreleased resources. Retry UnreleasedSession.Dispose.", inner)
        => UnreleasedSession = session;
}

internal static partial class GgmlNative
{
    [LibraryImport(DllName)]
    internal static partial long TSGgml_ResidentWeightPayloadBytes(int rank, int weightType, long inner, int rows, int columns);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_ResidentWeightCreate(int rank, int weightType, long inner, int rows, int columns,
        long capacity, out IntPtr handle);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_ResidentWeightUploadRows(IntPtr handle, IntPtr rows, int firstRow, int rowCount);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_ResidentWeightUploadInput(IntPtr handle, IntPtr input, int firstToken, int tokenCount);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_ResidentWeightProject(IntPtr handle);
    [LibraryImport(DllName)]
    internal static partial int TSGgml_ResidentWeightDownload(IntPtr handle, IntPtr output,
        int firstToken, int tokenCount, int firstRow, int rowCount);
}
