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
        private const int DefaultInitialBlockCount = 4;
        private const int DefaultMaxPooledBlocks = 64;
        private const int CudaInitialBlockCount = 0;
        private const int CudaMaxPooledBlocks = 8;
        private const long CudaMaxRetainedBlockSize = 8L * 1024 * 1024;
        // Apple Silicon (Metal, integrated GPU, unified memory) and the GGML CPU backend
        // both benefit from short-lived intermediate buffers (KV cache append, attention
        // scores, FFN gate/up etc.) being recycled. Keeping every block ever freed in the
        // pool, however, lets pure-decoder workloads grow the pooled footprint to several
        // GB on long contexts where prefill allocates per-layer score / KV-staging tensors
        // that decode never reuses. Cap the per-block retention to 64 MB and the total to
        // 32 blocks: that's >>2 GB of fast turnaround buffers but bounds the retained set
        // so the resident memory stays close to the model size + KV cache.
        private const int IntegratedMaxPooledBlocks = 32;
        private const long IntegratedMaxRetainedBlockSize = 64L * 1024 * 1024;

        private readonly object _lock = new object();
        private readonly List<PoolBlock> _available = new List<PoolBlock>();
        private readonly bool _useVirtualAlloc;
        private readonly int _pageSize;
        private readonly int _initialBlockCount;
        private readonly int _maxPooledBlocks;
        private readonly nuint _maxRetainedBlockSize;

        public GgmlMemoryPool(GgmlBackendType backendType)
        {
            int systemPageSize = Environment.SystemPageSize;
            _pageSize = RuntimeInformation.IsOSPlatform(OSPlatform.OSX)
                ? Math.Max(MetalPageSize, systemPageSize)
                : Math.Max(GgmlHostPtrAlignment, systemPageSize);
            _useVirtualAlloc = true;

            if (backendType == GgmlBackendType.Cuda)
            {
                // CUDA weights / KV caches are mirrored on device, so holding onto large
                // freed host buffers just bloats RAM without helping steady-state decode.
                _initialBlockCount = CudaInitialBlockCount;
                _maxPooledBlocks = CudaMaxPooledBlocks;
                _maxRetainedBlockSize = (nuint)CudaMaxRetainedBlockSize;
            }
            else
            {
                _initialBlockCount = DefaultInitialBlockCount;
                _maxPooledBlocks = IntegratedMaxPooledBlocks;
                _maxRetainedBlockSize = (nuint)IntegratedMaxRetainedBlockSize;
            }
        }

        public IntPtr Allocate(long byteLength)
        {
            nuint size = (nuint)byteLength;
            nuint alignedSize = AlignSize(size);

            lock (_lock)
            {
                // Best-fit search: find the smallest pooled block that satisfies the
                // request. Avoids handing out a 64 MB scratch block for a 32 KB
                // intermediate tensor (which would later force a brand-new allocation
                // for the next 32 MB request because the only large block is gone).
                int bestIdx = -1;
                nuint bestSize = nuint.MaxValue;
                for (int i = 0; i < _available.Count; i++)
                {
                    nuint blockSize = _available[i].Size;
                    if (blockSize >= alignedSize && blockSize < bestSize)
                    {
                        bestIdx = i;
                        bestSize = blockSize;
                        if (blockSize == alignedSize)
                            break;
                    }
                }

                if (bestIdx >= 0)
                {
                    PoolBlock block = _available[bestIdx];
                    _available.RemoveAt(bestIdx);
                    return block.Ptr;
                }
            }

            return AllocateNew(alignedSize);
        }

        public void Free(IntPtr ptr, long byteLength)
        {
            if (ptr == IntPtr.Zero) return;

            nuint size = (nuint)byteLength;
            nuint alignedSize = AlignSize(size);

            if (_maxPooledBlocks <= 0 || alignedSize > _maxRetainedBlockSize)
            {
                FreeToSystem(ptr, alignedSize);
                return;
            }

            lock (_lock)
            {
                if (_available.Count < _maxPooledBlocks)
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
            if (_useVirtualAlloc)
            {
                IntPtr ptr = AllocateVirtual(alignedSize);
                if (ptr != IntPtr.Zero)
                    return ptr;
            }

            return Marshal.AllocHGlobal((nint)alignedSize);
        }

        private void FreeToSystem(IntPtr ptr, nuint size)
        {
            if (_useVirtualAlloc && FreeVirtual(ptr, size))
            {
                return;
            }

            Marshal.FreeHGlobal(ptr);
        }

        internal void EnsureInitialBlocks()
        {
            lock (_lock)
            {
                while (_available.Count < _initialBlockCount)
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

        private static IntPtr AllocateVirtual(nuint alignedSize)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return WindowsVirtualAlloc(IntPtr.Zero, alignedSize, WindowsMemCommit | WindowsMemReserve, WindowsPageReadWrite);

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
            {
                int flags = UnixMapPrivate | (RuntimeInformation.IsOSPlatform(OSPlatform.OSX) ? UnixMapAnonMac : UnixMapAnonymous);
                IntPtr ptr = UnixMmap(IntPtr.Zero, alignedSize, UnixProtRead | UnixProtWrite, flags, -1, IntPtr.Zero);
                return ptr == UnixMapFailed ? IntPtr.Zero : ptr;
            }

            return IntPtr.Zero;
        }

        private static bool FreeVirtual(IntPtr ptr, nuint size)
        {
            if (ptr == IntPtr.Zero)
                return true;

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return WindowsVirtualFree(ptr, UIntPtr.Zero, WindowsMemRelease);

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || RuntimeInformation.IsOSPlatform(OSPlatform.OSX))
                return UnixMunmap(ptr, size) == 0;

            return false;
        }

        private static readonly IntPtr UnixMapFailed = new IntPtr(-1);
        private const int UnixProtRead = 0x1;
        private const int UnixProtWrite = 0x2;
        private const int UnixMapPrivate = 0x02;
        private const int UnixMapAnonymous = 0x20;
        private const int UnixMapAnonMac = 0x1000;

        private const uint WindowsMemCommit = 0x1000;
        private const uint WindowsMemReserve = 0x2000;
        private const uint WindowsMemRelease = 0x8000;
        private const uint WindowsPageReadWrite = 0x04;

        [DllImport("kernel32.dll", SetLastError = true)]
        private static extern IntPtr WindowsVirtualAlloc(IntPtr lpAddress, nuint dwSize, uint flAllocationType, uint flProtect);

        [DllImport("kernel32.dll", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static extern bool WindowsVirtualFree(IntPtr lpAddress, UIntPtr dwSize, uint dwFreeType);

        [DllImport("libc", EntryPoint = "mmap", SetLastError = true)]
        private static extern IntPtr UnixMmap(IntPtr addr, nuint length, int prot, int flags, int fd, IntPtr offset);

        [DllImport("libc", EntryPoint = "munmap", SetLastError = true)]
        private static extern int UnixMunmap(IntPtr addr, nuint length);
    }
}
