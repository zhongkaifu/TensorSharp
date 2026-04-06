// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Threading;

namespace TensorSharp.GGML
{
    /// <summary>
    /// Memory pool for GGML allocations. Reuses allocations to reduce allocator overhead.
    /// GGML host-ptr buffers require aligned addresses, so use aligned allocations on every
    /// platform: 16KB on macOS for Metal shared memory, 32 bytes elsewhere for GGML CPU.
    /// </summary>
    internal sealed class GgmlMemoryPool
    {
        /// <summary>16KB - Apple Silicon page size; required for Metal newBufferWithBytesNoCopy.</summary>
        private const int MetalPageSize = 16 * 1024;
        private const int GgmlHostPtrAlignment = 32;
        private const int BlockSize = 32 * 1024 * 1024; // 32 MB per block
        private const int InitialBlockCount = 4;
        private const int MaxPooledBlocks = 64;

        private readonly object _lock = new object();
        private readonly List<PoolBlock> _available = new List<PoolBlock>();
        private readonly bool _useAlignedAlloc;
        private readonly int _pageSize;

        public GgmlMemoryPool()
        {
            _useAlignedAlloc = true;
            _pageSize = RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? MetalPageSize : GgmlHostPtrAlignment;
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
