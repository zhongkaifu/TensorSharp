using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;

namespace TensorSharp.GGML
{
    /// <summary>
    /// Memory pool for GGML Metal backend. Reuses allocations to reduce allocator overhead.
    /// On macOS, uses page-aligned blocks for Metal shared memory (zero-copy).
    /// Apple Silicon uses 16KB pages; Intel Macs use 4KB. We use 16KB on macOS for optimal
    /// Metal buffer_from_host_ptr compatibility (required for zero-copy).
    /// </summary>
    internal sealed class GgmlMemoryPool
    {
        /// <summary>16KB - Apple Silicon page size; required for Metal newBufferWithBytesNoCopy.</summary>
        private const int MetalPageSize = 16 * 1024;
        private const int BlockSize = 32 * 1024 * 1024; // 32 MB per block
        private const int InitialBlockCount = 4;
        private const int MaxPooledBlocks = 64;

        private readonly object _lock = new object();
        private readonly List<PoolBlock> _available = new List<PoolBlock>();
        private readonly bool _useAlignedAlloc;
        private readonly int _pageSize;

        public GgmlMemoryPool()
        {
            _useAlignedAlloc = RuntimeInformation.IsOSPlatform(OSPlatform.OSX);
            // Apple Silicon uses 16KB pages; use MetalPageSize on macOS for zero-copy compatibility
            _pageSize = _useAlignedAlloc ? MetalPageSize : 4096;
        }

        public IntPtr Allocate(long byteLength)
        {
            nuint size = (nuint)byteLength;
            nuint alignedSize = AlignSize(size);

            lock (_lock)
            {
                for (int i = 0; i < _available.Count; i++)
                {
                    if (_available[i].Size >= alignedSize)
                    {
                        PoolBlock block = _available[i];
                        _available.RemoveAt(i);
                        return block.Ptr;
                    }
                }
            }

            return AllocateNew(alignedSize);
        }

        public void Free(IntPtr ptr, long byteLength)
        {
            if (ptr == IntPtr.Zero) return;

            nuint size = (nuint)byteLength;
            nuint alignedSize = AlignSize(size);

            lock (_lock)
            {
                if (_available.Count < MaxPooledBlocks)
                {
                    _available.Add(new PoolBlock(ptr, alignedSize));
                    return;
                }
            }

            FreeToSystem(ptr, alignedSize);
        }

        private nuint AlignSize(nuint size)
        {
            if (size == 0) return (nuint)_pageSize;
            return ((size + (nuint)(_pageSize - 1)) / (nuint)_pageSize) * (nuint)_pageSize;
        }

        private IntPtr AllocateNew(nuint alignedSize)
        {
            if (_useAlignedAlloc)
            {
                unsafe
                {
                    return (IntPtr)NativeMemory.AlignedAlloc(alignedSize, (nuint)_pageSize);
                }
            }
            return Marshal.AllocHGlobal((nint)alignedSize);
        }

        private void FreeToSystem(IntPtr ptr, nuint size)
        {
            if (_useAlignedAlloc)
            {
                unsafe
                {
                    NativeMemory.AlignedFree((void*)ptr);
                }
            }
            else
            {
                Marshal.FreeHGlobal(ptr);
            }
        }

        internal void EnsureInitialBlocks()
        {
            lock (_lock)
            {
                while (_available.Count < InitialBlockCount)
                {
                    IntPtr ptr = AllocateNew((nuint)BlockSize);
                    _available.Add(new PoolBlock(ptr, (nuint)BlockSize));
                }
            }
        }

        private readonly struct PoolBlock
        {
            public readonly IntPtr Ptr;
            public readonly nuint Size;

            public PoolBlock(IntPtr ptr, nuint size)
            {
                Ptr = ptr;
                Size = size;
            }
        }
    }
}
