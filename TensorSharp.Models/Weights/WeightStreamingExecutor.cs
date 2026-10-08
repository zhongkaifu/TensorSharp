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
    private readonly GgufMemoryCatalog _catalog;
    private readonly StreamingHostBuffer _weights;
    private GgmlQ8StreamingSession _activeSession;
    private bool _disposed;
    private long _weightBytes, _readBytes, _linearTiles, _embeddingRows, _peakHost, _peakDevice;

    internal WeightStreamingExecutor(GgufFile gguf, WeightStreamingOptions options)
    {
        _options = options;
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
            if (weight.Ne0 / 32 * 34 > _options.TileBytes)
                throw new MemoryPressureException($"Weight '{info.Name}' needs at least one complete Q8_0 row ({weight.Ne0 / 32 * 34} bytes); increase TileBytes.");
            _weightBytes = checked(_weightBytes + weight.RawBytes);
            return weight;
        }
    }

    internal WeightStreamingStatistics Statistics
    {
        get { lock (_gate) return new(_weightBytes, _readBytes, _linearTiles, _embeddingRows, _peakHost, _peakDevice); }
    }

    internal unsafe void Linear(QuantizedWeight weight, IntPtr input, IntPtr output, int tokens, int rank = 0)
    {
        lock (_gate)
        {
            ObjectDisposedException.ThrowIf(_disposed, this);
            if (_activeSession != null) throw new InvalidOperationException("A failed CUDA workspace cleanup must be retried before another streamed operation.");
            if (!weight.IsStreamed || input == IntPtr.Zero || output == IntPtr.Zero || tokens <= 0)
                throw new ArgumentException("Streaming linear requires a file-backed weight and valid input/output.");
            long rowBytes = checked(weight.Ne0 / 32 * 34);
            var (tileRows, tokenRows) = SelectTileLayout(_options, weight.Ne0, weight.Ne1, tokens,
                GgmlQ8StreamingSession.GetPayloadBytes);
            // Packed output is copied into its strided destination before the next
            // tile. This buffer is separate from, and charged alongside, file reads.
            using var tileOutput = new StreamingHostBuffer(_options, checked(tileRows * tokenRows * sizeof(float)));
            _peakHost = Math.Max(_peakHost, checked(_weights.AllocatedBytes + tileOutput.AllocatedBytes));
            for (int token = 0; token < tokens;)
            {
                int count = Math.Min(tokenRows, tokens - token);
                try
                {
                    _activeSession = new GgmlQ8StreamingSession(_options.Budget, _options.DevicePools,
                        rank, weight.Ne0, tileRows, count, input + checked((nint)((long)token * weight.Ne0 * sizeof(float))));
                }
                catch (GgmlQ8StreamingAllocationException failure)
                {
                    _activeSession = failure.UnreleasedSession;
                    throw;
                }
                _peakDevice = Math.Max(_peakDevice, _activeSession.PayloadBytes);
                try
                {
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
                }
                finally { ReleaseSession(); }
                token = checked(token + count);
            }
        }
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
            int rowBytes = checked((int)(weight.Ne0 / 32 * 34));
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
        long inputWidth, long outputRows, int tokens, Func<long, int, int, long> payloadBytes)
    {
        if (inputWidth <= 0 || inputWidth > int.MaxValue || inputWidth % 32 != 0 || outputRows <= 0 || tokens <= 0)
            throw new ArgumentOutOfRangeException(nameof(inputWidth));
        long rowBytes = checked(inputWidth / 32 * 34);
        int maximumRows = checked((int)Math.Min(outputRows, options.TileBytes / rowBytes));
        var available = options.Budget.Snapshot().ToDictionary(x => x.Pool, x => x.Available, StringComparer.Ordinal);
        for (int batch = Math.Min(tokens, options.TokenTileRows); batch > 0; batch = batch == 1 ? 0 : Math.Max(1, batch / 2))
        {
            int low = 1, high = maximumRows, best = 0;
            while (low <= high)
            {
                int rows = low + (high - low) / 2;
                long output = checked((long)rows * batch * sizeof(float));
                long host = checked((output + 63) / 64 * 64);
                long device = payloadBytes(inputWidth, rows, batch);
                bool fits = output <= int.MaxValue && host <= available[options.HostPool] &&
                    options.DevicePools.All(pool => device <= available[pool] - (pool == options.HostPool ? host : 0));
                if (fits) { best = rows; low = rows + 1; }
                else high = rows - 1;
            }
            if (best > 0) return (best, batch);
        }
        throw new MemoryPressureException("The available shared pools cannot hold one complete Q8_0 weight row, one input token and their output staging.");
    }

    private void ReleaseSession()
    {
        _activeSession?.Dispose();
        _activeSession = null; // Only after native memory was actually freed.
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
