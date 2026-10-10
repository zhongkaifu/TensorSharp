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
    private readonly bool _enablePeerCopies;
    private long _deviceCopies, _peerCopies;
    public long DeviceCopyCount => Interlocked.Read(ref _deviceCopies);
    public long PeerCopyCount => Interlocked.Read(ref _peerCopies);
    public CudaResidencyBackend(CudaContext context, string pool, string node = "local", long allocationGranularity = 2L << 20,
        bool enablePeerCopies = false)
    {
        _context = context ?? throw new ArgumentNullException(nameof(context));
        ArgumentException.ThrowIfNullOrWhiteSpace(pool);
        ArgumentOutOfRangeException.ThrowIfNegativeOrZero(allocationGranularity);
        _pool = pool;
        _allocationGranularity = allocationGranularity;
        // Some PCIe/IOMMU topologies advertise P2P but corrupt transfers. The
        // safe default is bounded host staging; opt in only after the directed
        // pair checks in UnifiedMemory.CudaProbe pass on the actual deployment.
        _enablePeerCopies = enablePeerCopies;
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
        return ValueTask.FromResult<IResourceBuffer>(new Buffer(_context, byteLength, _enablePeerCopies, peer =>
        {
            if (peer) Interlocked.Increment(ref _peerCopies);
            else Interlocked.Increment(ref _deviceCopies);
        }));
    }

    private sealed class Buffer : IResourceBuffer, IDirectCopyTarget
    {
        private readonly CudaContext _context;
        private nint _pointer;
        private readonly bool _enablePeerCopies;
        private readonly Action<bool> _recordCopy;
        internal Buffer(CudaContext context, long bytes, bool enablePeerCopies, Action<bool> recordCopy)
        {
            _context = context;
            _enablePeerCopies = enablePeerCopies;
            _recordCopy = recordCopy;
            ByteLength = bytes;
            context.MakeCurrent();
            int allocationResult = CudaDriverApi.cuMemAlloc(out _pointer, checked((nuint)bytes));
            if (allocationResult == 2 /* CUDA_ERROR_OUT_OF_MEMORY */)
                throw new OutOfMemoryException($"CUDA device {context.DeviceId} refused {bytes} bytes.");
            allocationResult.ThrowOnError();
            try
            {
                CudaDriverApi.cuMemsetD8(_pointer, 0, checked((nuint)bytes)).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
            }
            catch (Exception initializationFailure)
            {
                try { Dispose(); }
                catch (Exception cleanupFailure)
                {
                    // The scheduler must retain both this allocation and its
                    // budget charge until a later cleanup attempt succeeds.
                    throw new ResourceAllocationException(this,
                        new AggregateException(initializationFailure, cleanupFailure));
                }
                throw;
            }
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
        public ValueTask<bool> TryCopyFromAsync(IResourceSource source, CancellationToken cancellationToken = default)
        {
            cancellationToken.ThrowIfCancellationRequested();
            if (source is not Buffer other) return ValueTask.FromResult(false);
            if (other.ByteLength != ByteLength) throw new ArgumentException("Device copy lengths differ.");
            _context.MakeCurrent();
            if (_context.DeviceId == other._context.DeviceId)
            {
                CudaDriverApi.cuMemcpyDtoD(Pointer, other.Pointer, (nuint)ByteLength).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError();
                _recordCopy(false);
                return ValueTask.FromResult(true);
            }
            if (!_enablePeerCopies) return ValueTask.FromResult(false);
            CudaDriverApi.cuDeviceGet(out int destinationDevice, _context.DeviceId).ThrowOnError();
            CudaDriverApi.cuDeviceGet(out int sourceDevice, other._context.DeviceId).ThrowOnError();
            CudaDriverApi.cuDeviceCanAccessPeer(out int accessible, destinationDevice, sourceDevice).ThrowOnError();
            if (accessible == 0) return ValueTask.FromResult(false);
            int enable = CudaDriverApi.cuCtxEnablePeerAccess(other._context.Handle, 0);
            // CUDA_ERROR_PEER_ACCESS_ALREADY_ENABLED. Do not disable shared primary
            // context access when this adapter ends; other TensorSharp paths use it.
            if (enable != 0 && enable != 704) return ValueTask.FromResult(false);
            try
            {
                CudaDriverApi.cuMemcpyPeerAsync(Pointer, _context.Handle, other.Pointer,
                    other._context.Handle, (nuint)ByteLength, IntPtr.Zero).ThrowOnError();
            }
            finally { CudaDriverApi.cuStreamSynchronize(IntPtr.Zero).ThrowOnError(); }
            _recordCopy(true);
            return ValueTask.FromResult(true);
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
