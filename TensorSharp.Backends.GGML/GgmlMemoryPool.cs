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
using System.IO;
using TensorSharp.Memory;

namespace TensorSharp.GGML
{
    /// <summary>
    /// Memory pool for GGML allocations. Reuses allocations to reduce allocator overhead.
    /// GGML host-ptr buffers require aligned addresses, so use aligned allocations on every
    /// platform: 16KB on macOS for Metal shared memory, 32 bytes elsewhere for GGML CPU.
    /// </summary>
    internal sealed partial class GgmlMemoryPool
    {
        /// <summary>16KB - Apple Silicon page size; required for Metal newBufferWithBytesNoCopy.</summary>
        private const int MetalPageSize = 16 * 1024;
        private const int GgmlHostPtrAlignment = 32;
        private const int BlockSize = 32 * 1024 * 1024; // 32 MB per block
        private const int DefaultInitialBlockCount = 4;
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
        private readonly List<PoolBlock> _pendingRelease = new List<PoolBlock>();
        private readonly Func<IntPtr, nuint, bool> _freeVirtual;
        // True size and allocation provenance of every block currently handed
        // out. The best-fit search below can serve a request from a LARGER
        // pooled block, so the byteLength the caller passes back to Free() is a
        // lower bound, not the block's real size. Freeing (or re-pooling) with
        // the caller's size would munmap only part of the mapping and leak the
        // tail as an unreclaimable VMA — enough churn eventually exhausts
        // vm.max_map_count, munmap starts failing, and the old fallback then
        // handed an mmap pointer to free(), aborting the process with
        // "free(): invalid pointer" / "munmap_chunk(): invalid pointer".
        private readonly Dictionary<IntPtr, PoolBlock> _outstanding = new Dictionary<IntPtr, PoolBlock>();
        private readonly bool _useVirtualAlloc;
        private readonly int _pageSize;
        private readonly int _initialBlockCount;
        private readonly int _maxPooledBlocks;
        private readonly nuint _maxRetainedBlockSize;

        public GgmlMemoryPool(GgmlBackendType backendType, Func<IntPtr, nuint, bool> freeVirtual = null)
        {
            _freeVirtual = freeVirtual ?? FreeVirtual;
            int systemPageSize = Environment.SystemPageSize;
            _pageSize = IsAppleOS()
                ? Math.Max(MetalPageSize, systemPageSize)
                : Math.Max(GgmlHostPtrAlignment, systemPageSize);
            _useVirtualAlloc = true;

            if (backendType == GgmlBackendType.Cuda || backendType == GgmlBackendType.Vulkan)
            {
                // CUDA/Vulkan weights / KV caches are mirrored on device, so holding onto
                // large freed host buffers just bloats RAM without helping steady-state decode.
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
            ArgumentOutOfRangeException.ThrowIfNegative(byteLength);
            nuint size = checked((nuint)byteLength);
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
                    // Pools predating an opt-in scope must acquire credit before
                    // their retained allocation can serve budgeted execution.
                    if (block.Credit == null)
                    {
                        var credit = HostAllocationBudgetScope.Reserve(checked((long)block.Size));
                        credit?.Commit();
                        block = new PoolBlock(block.Ptr, block.Size, block.IsVirtual, credit);
                    }
                    _available.RemoveAt(bestIdx);
                    _outstanding[block.Ptr] = block;
                    return block.Ptr;
                }
            }

            PoolBlock fresh = AllocateNew(alignedSize);
            lock (_lock)
            {
                _outstanding[fresh.Ptr] = fresh;
            }
            return fresh.Ptr;
        }

        public void Free(IntPtr ptr, long byteLength)
        {
            if (ptr == IntPtr.Zero) return;

            PoolBlock block;
            lock (_lock)
            {
                if (!_outstanding.Remove(ptr, out block))
                {
                    // In particular, a second free must not insert the same
                    // address twice and hand it to two live tensors.
                    throw new InvalidOperationException("Pointer is not an outstanding allocation of this GGML pool.");
                }

                if (_maxPooledBlocks > 0 && block.Size <= _maxRetainedBlockSize &&
                    _available.Count < _maxPooledBlocks)
                {
                    _available.Add(block);
                    return;
                }
            }

            if (!FreeToSystem(block))
            {
                // Ownership has returned to the pool. Keep failed frees out of
                // the reusable list; Trim retries them without losing credit.
                lock (_lock) _pendingRelease.Add(block);
            }
        }

        /// <summary>
        /// Free every block the pool is holding for reuse and return the bytes released.
        /// Outstanding blocks are untouched; the pool refills itself on demand, so the
        /// only cost of trimming is that the next few allocations map fresh memory.
        /// </summary>
        public long Trim()
        {
            List<PoolBlock> release;
            lock (_lock)
            {
                if (_available.Count == 0 && _pendingRelease.Count == 0)
                    return 0;
                release = new List<PoolBlock>(_available);
                release.AddRange(_pendingRelease);
                _available.Clear();
                _pendingRelease.Clear();
            }
            long bytes = 0;
            bool failed = false;
            foreach (PoolBlock block in release)
            {
                if (FreeToSystem(block)) bytes = checked(bytes + (long)block.Size);
                else
                {
                    lock (_lock) _pendingRelease.Add(block);
                    failed = true;
                }
            }
            if (failed) throw new IOException("The OS refused to release a GGML host mapping; retry trimming after resolving the failure.");
            return bytes;
        }

        private nuint AlignSize(nuint size)
        {
            if (size == 0) return (nuint)_pageSize;
            return checked(((size + (nuint)(_pageSize - 1)) / (nuint)_pageSize) * (nuint)_pageSize);
        }

        private PoolBlock AllocateNew(nuint alignedSize)
        {
            var credit = HostAllocationBudgetScope.Reserve(checked((long)alignedSize));
            try
            {
                bool isVirtual = _useVirtualAlloc;
                IntPtr ptr = isVirtual ? AllocateVirtual(alignedSize) : IntPtr.Zero;
                if (ptr == IntPtr.Zero)
                {
                    ptr = Marshal.AllocHGlobal(checked((nint)alignedSize));
                    isVirtual = false;
                }
                credit?.Commit();
                return new PoolBlock(ptr, alignedSize, isVirtual, credit);
            }
            catch { credit?.Dispose(); throw; }
        }

        private bool FreeToSystem(in PoolBlock block)
        {
            if (block.IsVirtual)
            {
                // If munmap/VirtualFree fails (e.g. vm.max_map_count exhausted),
                // leaking the mapping is the only safe outcome — this pointer
                // did not come from the C heap, so it must NEVER reach free().
                if (!_freeVirtual(block.Ptr, block.Size)) return false;
            }
            else Marshal.FreeHGlobal(block.Ptr);
            block.Credit?.Dispose();
            return true;
        }

        internal void EnsureInitialBlocks()
        {
            lock (_lock)
            {
                try
                {
                    while (_available.Count < _initialBlockCount)
                        _available.Add(AllocateNew((nuint)BlockSize));
                }
                catch
                {
                    // A rejected constructor has no caller to trim the partially
                    // initialized pool. Roll back its already allocated blocks.
                    Trim();
                    throw;
                }
            }
        }

        private readonly struct PoolBlock
        {
            public readonly IntPtr Ptr;
            public readonly nuint Size;
            public readonly bool IsVirtual;
            public readonly HostAllocationBudgetScope.Allocation Credit;

            public PoolBlock(IntPtr ptr, nuint size, bool isVirtual, HostAllocationBudgetScope.Allocation credit = null)
            {
                Ptr = ptr;
                Size = size;
                IsVirtual = isVirtual;
                Credit = credit;
            }
        }

        /// <summary>
        /// True on every Darwin platform TensorSharp runs on (macOS, iOS/iPadOS,
        /// Mac Catalyst, tvOS). They share the 16 KB Metal page size and the
        /// Darwin-specific MAP_ANON value, and - critically - they all feed
        /// host pointers to ggml-metal's newBufferWithBytesNoCopy, which
        /// requires page-aligned memory. Falling through to the
        /// Marshal.AllocHGlobal path on any of them would hand Metal unaligned
        /// buffers.
        /// </summary>
        private static bool IsAppleOS()
        {
            return OperatingSystem.IsMacOS() || OperatingSystem.IsIOS()
                || OperatingSystem.IsMacCatalyst() || OperatingSystem.IsTvOS();
        }

        private static IntPtr AllocateVirtual(nuint alignedSize)
        {
            if (RuntimeInformation.IsOSPlatform(OSPlatform.Windows))
                return WindowsVirtualAlloc(IntPtr.Zero, alignedSize, WindowsMemCommit | WindowsMemReserve, WindowsPageReadWrite);

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || IsAppleOS())
            {
                int flags = UnixMapPrivate | (IsAppleOS() ? UnixMapAnonMac : UnixMapAnonymous);
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

            if (RuntimeInformation.IsOSPlatform(OSPlatform.Linux) || IsAppleOS())
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

        [LibraryImport("kernel32.dll", EntryPoint = "VirtualAlloc", SetLastError = true)]
        private static partial IntPtr WindowsVirtualAlloc(IntPtr lpAddress, nuint dwSize, uint flAllocationType, uint flProtect);

        [LibraryImport("kernel32.dll", EntryPoint = "VirtualFree", SetLastError = true)]
        [return: MarshalAs(UnmanagedType.Bool)]
        private static partial bool WindowsVirtualFree(IntPtr lpAddress, UIntPtr dwSize, uint dwFreeType);

        [LibraryImport("libc", EntryPoint = "mmap", SetLastError = true)]
        private static partial IntPtr UnixMmap(IntPtr addr, nuint length, int prot, int flags, int fd, IntPtr offset);

        [LibraryImport("libc", EntryPoint = "munmap", SetLastError = true)]
        private static partial int UnixMunmap(IntPtr addr, nuint length);
    }
}
