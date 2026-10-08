// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Collections.Generic;
using System.Threading;
using System.Threading.Tasks;
using TensorSharp.Cuda.Interop;
using TensorSharp.Memory;

namespace TensorSharp.Cuda;

/// <summary>Explicit CUDA allocations for the unified residency API. Does not use
/// CudaStorage's implicit host mirrors or device pool. Transfers are synchronous in
/// this first adapter; execution must supply a completion fence before releasing a
/// lease. It does not retrofit captured GGML graphs or existing model weight pointers.</summary>
public sealed class CudaResidencyBackend : IMemoryBackend
{
    private readonly CudaContext _context;
    private readonly string _pool;
    private readonly long _allocationGranularity;
    public CudaResidencyBackend(CudaContext context, string pool, string node = "local", long allocationGranularity = 2L << 20)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        ArgumentException.ThrowIfNullOrWhiteSpace(pool);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(allocationGranularity);
        _pool = pool;
        _allocationGranularity = allocationGranularity;
        Location = new(node, $"cuda:{context.DeviceId}", MemoryTier.Accelerator);
    }
    public MemoryLocation Location { get; }
    public IReadOnlyList<MemoryCharge> GetAllocationCharges(long byteLength)
    {
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        return new[] { new MemoryCharge(_pool, checked((byteLength + _allocationGranularity - 1) / _allocationGranularity * _allocationGranularity)) };
    }
    public ValueTask<IResourceBuffer> AllocateAsync(long byteLength, CancellationToken cancellationToken = default)
    {
        cancellationToken.ThrowIfCancellationRequested();
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(byteLength);
        return ValueTask.FromResult<IResourceBuffer>(new Buffer(_context, byteLength));
    }

    private sealed class Buffer : IResourceBuffer
    {
        private readonly CudaContext _context;
        private nint _pointer;
        internal Buffer(CudaContext context, long bytes)
        {
            _context = context;
            ByteLength = bytes;
            context.MakeCurrent();
            CudaDriverApi.cuMemAlloc(out _pointer, checked((nuint)bytes)).ThrowOnError();
            try
            {
                CudaDriverApi.cuMemsetD8(_pointer, 0, checked((nuint)bytes)).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
            }
            catch { Dispose(); throw; }
        }
        public long ByteLength { get; }
        public nint Pointer { get { ObjectDisposedException.ThrowIf(_pointer == 0, this); return _pointer; } }
        private void Check(long offset, int count)
        {
            if (offset < 0 || offset > ByteLength || count > ByteLength - offset) throw new ArgumentOutOfRangeException(nameof(offset));
        }
        public unsafe ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Check(offset, destination.Length);
            _context.MakeCurrent();
            using var pin = destination.Pin();
            CudaDriverApi.cuMemcpyDtoH((nint)pin.Pointer, checked(Pointer + (nint)offset), (nuint)destination.Length).ThrowOnError();
            return ValueTask.CompletedTask;
        }
        public unsafe ValueTask WriteAsync(long offset, ReadOnlyMemory<byte> source, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            Check(offset, source.Length);
            _context.MakeCurrent();
            using var pin = source.Pin();
            CudaDriverApi.cuMemcpyHtoD(checked(Pointer + (nint)offset), (nint)pin.Pointer, (nuint)source.Length).ThrowOnError();
            // Pageable HtoD may return after staging, before device DMA finishes.
            // Publish residency only after the default-stream copy has completed.
            CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
            return ValueTask.CompletedTask;
        }
        public void Dispose()
        {
            if (_pointer == 0) return;
            _context.MakeCurrent();
            CudaDriverApi.cuMemFree(_pointer).ThrowOnError();
            _pointer = 0;
        }
    }
}
