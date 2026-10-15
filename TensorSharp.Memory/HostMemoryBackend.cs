// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers;
using System.Runtime.InteropServices;

namespace TensorSharp.Memory;

/// <summary>Native host allocations; no hidden GC arrays or TensorSharp-side pool. This is
/// ordinary pageable RAM, not OS page-locked DMA memory.</summary>
public sealed class HostMemoryBackend : IMemoryBackend
{
    private readonly string _pool;
    public HostMemoryBackend(string pool, string node = "local", string numa = "ram")
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(pool);
        _pool = pool;
        Location = new(node, numa, MemoryTier.Host);
    }
    public MemoryLocation Location { get; }
    public IReadOnlyList<MemoryCharge> GetAllocationCharges(long byteLength)
        => new[] { new MemoryCharge(_pool, MemoryRange.Align(byteLength, 64)) };
    public ValueTask<IResourceBuffer> AllocateAsync(long byteLength, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        return ValueTask.FromResult<IResourceBuffer>(new HostBuffer(byteLength));
    }

    private sealed unsafe class HostBuffer : IResourceBuffer
    {
        private nint _pointer;
        public HostBuffer(long byteLength)
        {
            ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
            ByteLength = byteLength;
            var allocated = checked((nuint)MemoryRange.Align(byteLength, 64));
            _pointer = (nint)NativeMemory.AlignedAlloc(allocated, 64);
            if (_pointer == 0) throw new OutOfMemoryException();
            NativeMemory.Clear((void*)_pointer, allocated);
        }
        public long ByteLength { get; }
        public nint Pointer { get { ObjectDisposedException.ThrowIf(_pointer == 0, this); return _pointer; } }
        public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            MemoryRange.Check(ByteLength, offset, destination.Length);
            new ReadOnlySpan<byte>((byte*)Pointer + offset, destination.Length).CopyTo(destination.Span);
            return ValueTask.CompletedTask;
        }
        public ValueTask WriteAsync(long offset, ReadOnlyMemory<byte> source, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            MemoryRange.Check(ByteLength, offset, source.Length);
            source.Span.CopyTo(new Span<byte>((byte*)Pointer + offset, source.Length));
            return ValueTask.CompletedTask;
        }
        public void Dispose()
        {
            var ptr = Interlocked.Exchange(ref _pointer, 0);
            if (ptr != 0) NativeMemory.AlignedFree((void*)ptr);
        }
    }
}

internal sealed unsafe class TransferMemory : MemoryManager<byte>
{
    private nint _pointer;
    private readonly int _length;
    internal TransferMemory(int length)
    {
        _length = length;
        _pointer = (nint)NativeMemory.AlignedAlloc(checked((nuint)MemoryRange.Align(length, 64)), 64);
        if (_pointer == 0) throw new OutOfMemoryException();
    }
    public override Span<byte> GetSpan()
    {
        ObjectDisposedException.ThrowIf(_pointer == 0, this);
        return new Span<byte>((void*)_pointer, _length);
    }
    public override MemoryHandle Pin(int elementIndex = 0)
    {
        if ((uint)elementIndex > (uint)_length) throw new ArgumentOutOfRangeException(nameof(elementIndex));
        ObjectDisposedException.ThrowIf(_pointer == 0, this);
        return new MemoryHandle((byte*)_pointer + elementIndex);
    }
    public override void Unpin() { }
    protected override void Dispose(bool disposing)
    {
        var ptr = Interlocked.Exchange(ref _pointer, 0);
        if (ptr != 0) NativeMemory.AlignedFree((void*)ptr);
    }
}
