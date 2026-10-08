// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Buffers;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Memory;
using TensorSharp.Runtime;

namespace TensorSharp.Models;

/// <summary>One fixed file-read tile; no tensor-sized host copy, mmap, weight
/// cache key or retained graph. Operations are synchronous before buffers recycle.</summary>
internal sealed class WeightStreamingExecutor : IDisposable
{
    private readonly object _gate = new();
    private readonly WeightStreamingOptions _options;
    private readonly GgmlWeightStreamingArithmetic _arithmetic;
    private readonly GgufMemoryCatalog _catalog;
    private readonly StreamingHostBuffer _weights;
    private GgmlWeightStreamingSession _activeSession;
    private GgmlResidentWeightSession _activeResidentSession;
    private bool _disposed;
    private long _weightBytes, _readBytes, _linearTiles, _embeddingRows, _peakHost, _peakDevice;
    private long _sessionCreations, _inputUploads, _matrixProjections;

    internal WeightStreamingExecutor(GgufFile gguf, WeightStreamingOptions options,
        GgmlWeightStreamingArithmetic arithmetic = GgmlWeightStreamingArithmetic.FullPrecision)
    {
        _options = options;
        _arithmetic = arithmetic;
        _catalog = gguf.CreateMemoryCatalog("stream-" + Guid.NewGuid().ToString("N"));
        try
        {
            _weights = new StreamingHostBuffer(options, options.TileBytes);
            _peakHost = _weights.AllocatedBytes;
        }
        catch { _catalog.Dispose(); throw; }
    }

    internal QuantizedWeight CreateWeight(GgufTensorInfo info)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            var (_, source) = _catalog.Get(info.Name);
            if (info.Shape.Length != 2) throw new NotSupportedException("Only two-dimensional streamed weights are supported.");
            var weight = QuantizedWeight.CreateFileBacked(source, (int)info.Type, checked((long)info.Shape[0]), checked((long)info.Shape[1]));
            if (weight.StreamingRowBytes > _options.TileBytes)
                throw new MemoryPressureException($"Weight '{info.Name}' needs at least one complete row ({weight.StreamingRowBytes} bytes); increase TileBytes.");
            _weightBytes = checked(_weightBytes + weight.RawBytes);
            return weight;
        }
    }

    internal WeightStreamingStatistics Statistics
    {
        get { lock (_gate) return new(_weightBytes, _readBytes, _linearTiles, _embeddingRows, _peakHost, _peakDevice,
            _sessionCreations, _inputUploads, _matrixProjections); }
    }

    internal unsafe void Linear(QuantizedWeight weight, IntPtr input, IntPtr output, int tokens, int rank = 0)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_activeSession != null || _activeResidentSession != null)
                throw new InvalidOperationException("A failed CUDA workspace cleanup must be retried before another streamed operation.");
            if (!weight.IsStreamed || input == IntPtr.Zero || output == IntPtr.Zero || tokens <= 0)
                throw new ArgumentException("Streaming linear requires a file-backed weight and valid input/output.");
            if (_arithmetic == GgmlWeightStreamingArithmetic.ResidentCuda
                && ((weight.GgmlType == 8 && tokens > 8) || (weight.GgmlType == 1 && tokens > 16)))
            {
                LinearResidentMatrix(weight, input, output, tokens, rank);
                return;
            }
            long rowBytes = weight.StreamingRowBytes;
            var (tileRows, tokenRows) = SelectTileLayout(_options, weight.Ne0, weight.Ne1, tokens,
                (width, rows, count) => GgmlWeightStreamingSession.GetPayloadBytes(weight.GgmlType, width, rows, count,
                    _arithmetic, tokens, weight.Ne1, rank), weight.GgmlType, _arithmetic);
            // Packed output is copied into its strided destination before the next
            // tile. This buffer is separate from, and charged alongside, file reads.
            using var tileOutput = new StreamingHostBuffer(_options, checked(tileRows * tokenRows * sizeof(float)));
            _peakHost = Math.Max(_peakHost, checked(_weights.AllocatedBytes + tileOutput.AllocatedBytes));
            try
            {
                _activeSession = new GgmlWeightStreamingSession(_options.Budget, _options.DevicePools,
                    rank, weight.GgmlType, weight.Ne0, tileRows, tokenRows, input, _arithmetic, tokens, weight.Ne1);
            }
            catch (GgmlWeightStreamingAllocationException failure)
            {
                _activeSession = failure.UnreleasedSession;
                throw;
            }
            _sessionCreations++;
            _inputUploads++;
            _peakDevice = Math.Max(_peakDevice, _activeSession.PayloadBytes);
            try
            {
                // Retain the same reservation and allocation across token chunks.
                // A short final chunk only changes the valid input/output extent.
                for (int token = 0; token < tokens;)
                {
                    int count = Math.Min(tokenRows, tokens - token);
                    if (token != 0)
                    {
                        _activeSession.UploadInput(input + checked((nint)((long)token * weight.Ne0 * sizeof(float))), count);
                        _inputUploads++;
                    }
                    for (long row = 0; row < weight.Ne1; row += tileRows)
                    {
                        int rows = checked((int)Math.Min(tileRows, weight.Ne1 - row));
                        int bytes = checked((int)(rows * rowBytes));
                        Read(weight, checked(row * rowBytes), bytes);
                        _activeSession.Execute(_weights.Pointer, rows, tileOutput.Pointer);
                        for (int t = 0; t < count; t++)
                        {
                            long dst = checked(((long)(token + t) * weight.Ne1 + row) * sizeof(float));
                            long src = checked((long)t * rows * sizeof(float));
                            long copyBytes = checked((long)rows * sizeof(float));
                            Buffer.MemoryCopy((byte*)tileOutput.Pointer + src, (byte*)output + dst, copyBytes, copyBytes);
                        }
                        _linearTiles++;
                    }
                    token = checked(token + count);
                }
            }
            finally { ReleaseSession(); }
        }
    }

    // CUDA MMQ/cuBLAS reduction partitions depend on the original M and N. Tiny
    // roundoff differences from independent row/token GEMMs can cross later
    // quantization boundaries, so preserve one logical projection here. Only
    // its temporary device matrix is resident; file reads and host downloads
    // stay bounded, and the entire allocation is charged before creation.
    private unsafe void LinearResidentMatrix(QuantizedWeight weight, IntPtr input, IntPtr output, int tokens, int rank)
    {
        int outputRows = checked((int)weight.Ne1);
        long payload = GgmlResidentWeightSession.GetPayloadBytes(rank, weight.GgmlType, weight.Ne0, outputRows, tokens);
        var (tileRows, tokenRows) = SelectTileLayout(_options, weight.Ne0, weight.Ne1, tokens,
            (_, _, _) => payload, weight.GgmlType);
        using var tileOutput = new StreamingHostBuffer(_options, checked(tileRows * tokenRows * sizeof(float)));
        _peakHost = Math.Max(_peakHost, checked(_weights.AllocatedBytes + tileOutput.AllocatedBytes));
        try
        {
            _activeResidentSession = new GgmlResidentWeightSession(_options.Budget, _options.DevicePools,
                rank, weight.GgmlType, weight.Ne0, outputRows, tokens);
        }
        catch (GgmlResidentWeightAllocationException failure)
        {
            _activeResidentSession = failure.UnreleasedSession;
            throw;
        }
        _sessionCreations++;
        _peakDevice = Math.Max(_peakDevice, _activeResidentSession.PayloadBytes);
        try
        {
            for (int row = 0; row < outputRows;)
            {
                int count = Math.Min(tileRows, outputRows - row);
                Read(weight, checked((long)row * weight.StreamingRowBytes), checked((int)(count * weight.StreamingRowBytes)));
                _activeResidentSession.UploadWeightRows(_weights.Pointer, row, count);
                _linearTiles++;
                row = checked(row + count);
            }
            for (int token = 0; token < tokens;)
            {
                int count = Math.Min(tokenRows, tokens - token);
                _activeResidentSession.UploadInputTokens(input + checked((nint)((long)token * weight.Ne0 * sizeof(float))), token, count);
                _inputUploads++;
                token = checked(token + count);
            }
            _activeResidentSession.Project();
            _matrixProjections++;
            for (int token = 0; token < tokens;)
            {
                int count = Math.Min(tokenRows, tokens - token);
                for (int row = 0; row < outputRows;)
                {
                    int rows = Math.Min(tileRows, outputRows - row);
                    _activeResidentSession.Download(tileOutput.Pointer, token, count, row, rows);
                    for (int t = 0; t < count; t++)
                    {
                        long dst = checked(((long)(token + t) * outputRows + row) * sizeof(float));
                        long src = checked((long)t * rows * sizeof(float));
                        long bytes = checked((long)rows * sizeof(float));
                        Buffer.MemoryCopy((byte*)tileOutput.Pointer + src, (byte*)output + dst, bytes, bytes);
                    }
                    row = checked(row + rows);
                }
                token = checked(token + count);
            }
        }
        finally { ReleaseSession(); }
    }

    internal unsafe void Embedding(QuantizedWeight weight, int[] tokens, IntPtr output)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ArgumentNullException.ThrowIfNull(tokens);
            if (!weight.IsStreamed || output == IntPtr.Zero) throw new ArgumentException("Invalid streamed embedding.");
            foreach (int token in tokens)
                if (token < 0 || token >= weight.Ne1) throw new ArgumentOutOfRangeException(nameof(tokens));
            int rowBytes = checked((int)weight.StreamingRowBytes);
            for (int i = 0; i < tokens.Length; i++)
            {
                Read(weight, checked((long)tokens[i] * rowBytes), rowBytes);
                NativeDequant.DequantizeToFloat32Native(weight.GgmlType, _weights.Pointer,
                    output + checked((nint)((long)i * weight.Ne0 * sizeof(float))), weight.Ne0);
                _embeddingRows++;
            }
        }
    }

    private void Read(QuantizedWeight weight, long offset, int bytes)
    {
        weight.FileSource.ReadAsync(offset, _weights.Memory[..bytes]).AsTask().GetAwaiter().GetResult();
        _readBytes = checked(_readBytes + bytes);
    }

    // Use remaining shared capacity, not a private subquota. Reduce output rows
    // first, then token rows; the reduction dimension is never split. Reservation
    // still arbitrates races with another owner after this planning snapshot.
    internal static (int WeightRows, int TokenRows) SelectTileLayout(WeightStreamingOptions options,
        long inputWidth, long outputRows, int tokens, Func<long, int, int, long> payloadBytes, int weightType = 8,
        GgmlWeightStreamingArithmetic arithmetic = GgmlWeightStreamingArithmetic.FullPrecision)
    {
        if (inputWidth <= 0 || inputWidth > int.MaxValue || outputRows <= 0 || tokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth));
        long rowBytes = QuantizedWeight.GetStreamingRowBytes(weightType, inputWidth);
        int maximumRows = checked((int)Math.Min(outputRows, options.TileBytes / rowBytes));
        // Resident Q8 quantization uses grid.y for the active input columns.
        // This cap applies only to the workspace, never the original logical N
        // captured by payloadBytes (which still chooses the resident arithmetic).
        int maximumTokens = arithmetic == GgmlWeightStreamingArithmetic.ResidentCuda && weightType == 8
            ? 65535 : 65535 * 8;
        // Validate irreducible dimensions and hardware before shrinking any
        // candidate. An unsupported device/operation must remain its native
        // error, rather than being mislabeled as temporary budget pressure.
        if (maximumRows > 0 && arithmetic == GgmlWeightStreamingArithmetic.ResidentCuda)
            _ = payloadBytes(inputWidth, 1, 1);
        var available = options.Budget.Snapshot().ToDictionary(x => x.Pool, x => x.Available, StringComparer.Ordinal);
        for (int batch = Math.Min(maximumTokens, Math.Min(tokens, options.TokenTileRows)); batch > 0; batch = batch == 1 ? 0 : Math.Max(1, batch / 2))
        {
            int low = 1, high = maximumRows, best = 0;
            while (low <= high)
            {
                int rows = low + (high - low) / 2;
                long output = checked((long)rows * batch * sizeof(float));
                long host = checked((output + 63) / 64 * 64);
                if (!FitsResidentIndexing(inputWidth, rows, batch, tokens, weightType, arithmetic))
                {
                    high = rows - 1;
                    continue;
                }
                long device = payloadBytes(inputWidth, rows, batch);
                bool fits = output <= int.MaxValue && host <= available[options.HostPool] &&
                    options.DevicePools.All(pool => device <= available[pool] - (pool == options.HostPool ? host : 0));
                if (fits) { best = rows; low = rows + 1; }
                else high = rows - 1;
            }
            if (best > 0) return (best, batch);
        }
        throw new MemoryPressureException("The available shared pools cannot hold one complete weight row, one input token and their output staging.");
    }

    private static bool FitsResidentIndexing(long width, int rows, int tokens, int logicalTokens, int type,
        GgmlWeightStreamingArithmetic arithmetic)
    {
        if (arithmetic != GgmlWeightStreamingArithmetic.ResidentCuda) return true;
        if (type == 8)
        {
            long paddedRows = ((long)rows + 127) / 128 * 128;
            long paddedWidth = (width + 511) / 512 * 512;
            return paddedRows <= int.MaxValue && paddedRows * tokens <= int.MaxValue
                && paddedRows * (width / 32) <= int.MaxValue
                && (long)tokens * (paddedWidth / 32) * 9 <= int.MaxValue;
        }
        if (type == 1 && logicalTokens is > 1 and <= 16)
        {
            long paddedRows = ((long)rows + 31) / 32 * 32;
            return paddedRows <= int.MaxValue && paddedRows * (width / 2) <= int.MaxValue
                && (long)tokens * (width / 2) <= int.MaxValue && paddedRows * tokens <= int.MaxValue;
        }
        return true;
    }

    // A previous CUDA cleanup failure retains the session and its quota.
    // Reset may retry that release, but must not reset model state or reopen
    // forwarding until every outstanding workspace has actually been freed.
    internal void ResetForReplay()
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            ReleaseSession();
        }
    }

    private void ReleaseSession()
    {
        _activeSession?.Dispose();
        _activeSession = null; // Only after native memory was actually freed.
        _activeResidentSession?.Dispose();
        _activeResidentSession = null;
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_disposed) return;
            ReleaseSession();
            ((IDisposable)_weights).Dispose();
            _catalog.Dispose();
            _disposed = true;
        }
    }
}

/// <summary>Exact bounded unmanaged staging, reserved before allocation and
/// refunded only after the allocation is freed. No GC array or allocator pool.</summary>
internal sealed unsafe class StreamingHostBuffer : MemoryManager<byte>
{
    private readonly BudgetReservation _charge;
    private readonly int _length;
    private nint _pointer;
    internal StreamingHostBuffer(WeightStreamingOptions options, int length)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(length);
        _length = length;
        AllocatedBytes = checked(((long)length + 63) / 64 * 64);
        _charge = options.Budget.Reserve(new[] { new MemoryCharge(options.HostPool, AllocatedBytes) });
        try
        {
            _pointer = (nint)NativeMemory.AlignedAlloc(checked((nuint)AllocatedBytes), 64);
            if (_pointer == 0) throw new OutOfMemoryException();
            _charge.Commit();
        }
        catch
        {
            if (_pointer != 0) NativeMemory.AlignedFree((void*)_pointer);
            _pointer = 0;
            _charge.Dispose();
            throw;
        }
    }
    internal long AllocatedBytes { get; }
    internal IntPtr Pointer { get { ObjectDisposedException.ThrowIf(_pointer == 0, this); return _pointer; } }
    public override Span<byte> GetSpan() => new((void*)Pointer, _length);
    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if ((uint)elementIndex > (uint)_length) throw new ArgumentOutOfRangeException(nameof(elementIndex));
        return new MemoryHandle((byte*)Pointer + elementIndex);
    }
    public override void Unpin() { }
    protected override void Dispose(bool disposing)
    {
        nint pointer = System.Threading.Interlocked.Exchange(ref _pointer, 0);
        if (pointer == 0) return;
        NativeMemory.AlignedFree((void*)pointer);
        _charge.Dispose();
    }
}
