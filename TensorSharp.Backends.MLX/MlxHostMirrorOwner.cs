// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.Runtime.InteropServices;
using TensorSharp.Memory;

namespace TensorSharp.MLX;

/// <summary>Owns the explicit host mirror only, independently of the MLX worker,
/// lazy arrays and Metal allocator cache. Its charged aligned allocation payload
/// is not an RSS measurement or a charge for a second imaginary UMA device copy.
/// Several configured host pools constrain this same allocation; do not sum them.
/// The containing MlxStorage owns finalization; this owner never calls MLX.</summary>
internal sealed class MlxHostMirrorOwner : IDisposable
{
    internal interface IAllocator
    {
        IntPtr Allocate(nuint bytes, nuint alignment);
        void Clear(IntPtr pointer, nuint bytes);
        void Free(IntPtr pointer);
    }

    private sealed unsafe class NativeAllocator : IAllocator
    {
        public IntPtr Allocate(nuint bytes, nuint alignment) => (IntPtr)NativeMemory.AlignedAlloc(bytes, alignment);
        public void Clear(IntPtr pointer, nuint bytes) => NativeMemory.Clear(pointer.ToPointer(), bytes);
        public void Free(IntPtr pointer) => NativeMemory.AlignedFree(pointer.ToPointer());
    }

    private static readonly IAllocator Native = new NativeAllocator();
    private readonly object _gate = new();
    private readonly IAllocator _allocator;
    private IntPtr _pointer;
    private HostAllocationBudgetScope.Allocation _credit;

    private MlxHostMirrorOwner(long allocatedBytes, IAllocator allocator)
    { AllocatedBytes = allocatedBytes; _allocator = allocator; }

    public long AllocatedBytes { get; }
    public IntPtr Pointer
    {
        get
        {
            lock (_gate)
            {
                ObjectDisposedException.ThrowIf(_pointer == IntPtr.Zero, this);
                return _pointer;
            }
        }
    }

    internal static MlxHostMirrorOwner Allocate(long bytes, IAllocator allocator = null)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(bytes);
        long aligned = checked((Math.Max(bytes, 1) + 63) / 64 * 64);
        nuint size = checked((nuint)aligned);
        var owner = new MlxHostMirrorOwner(aligned, allocator ?? Native);
        // Owner metadata exists before reserving; quota denial cannot call malloc.
        owner._credit = HostAllocationBudgetScope.Reserve(aligned);
        try
        {
            owner._pointer = owner._allocator.Allocate(size, 64);
            if (owner._pointer == IntPtr.Zero) throw new OutOfMemoryException("MLX host mirror allocation failed.");
            owner._allocator.Clear(owner._pointer, size);
            owner._credit?.Commit();
            return owner;
        }
        catch (Exception initialization)
        {
            try { owner.Dispose(); }
            catch (Exception cleanup)
            {
                throw new MlxHostMirrorAllocationException(owner, new AggregateException(initialization, cleanup));
            }
            throw;
        }
    }

    public void Dispose()
    {
        lock (_gate)
        {
            if (_pointer != IntPtr.Zero)
            {
                // Keep the pointer and credit if the allocator reports failure.
                // In production AlignedFree returns void; this does not invent a
                // recoverable status for a fatal native allocator error.
                _allocator.Free(_pointer);
                _pointer = IntPtr.Zero;
            }
            _credit?.Dispose();
            _credit = null;
        }
    }
}

internal sealed class MlxHostMirrorAllocationException : InvalidOperationException
{
    internal MlxHostMirrorAllocationException(MlxHostMirrorOwner owner, Exception inner)
        : base("MLX host mirror initialization and cleanup failed; retain the owner and retry disposal.", inner)
    { UnreleasedOwner = owner; }
    public MlxHostMirrorOwner UnreleasedOwner { get; }
}
