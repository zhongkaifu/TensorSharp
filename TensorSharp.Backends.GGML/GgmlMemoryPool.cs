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
    internal sealed partial class GgmlMemoryPool
    {
        /// <summary>16KB - Apple Silicon page size; required for Metal newBufferWithBytesNoCopy.</summary>
        private const int MetalPageSize = 16 * 1024;
        private const int GgmlHostPtrAlignment = 32;
        private const int BlockSize = 32 * 1024 * 1024; // 32 MB per block
        private const int DefaultInitialBlockCount = 4;
        private const int CudaInitialBlockCount = 0;
        private const int CudaMaxPooledBlocks = 8;
        private const int CudaTpMaxPooledBlocks = 128;
        private const long CudaMaxRetainedBlockSize = 8L * 1024 * 1024;
        private const long HostHugePageMinimumBytes = 32L * 1024 * 1024;
        // CUDA per-request caches (K/V holders, recurrent state, checkpoints) are
        // allocated and freed per request, and a fresh anonymous mapping costs a
        // page fault plus kernel zeroing per 4 KiB on first touch: ~1-2 GB/s, which
        // made every 150 MB state copy or 700 MB K/V download into a new holder
        // fault-bound. Freed blocks of any size are kept under a byte budget and
        // reused best-fit within a small waste margin. Env TS_GGML_HOST_POOL_MB
        // (0 disables); default min(8 GiB, available RAM / 16).
        private const long CudaRetainBudgetCapBytes = 8L * 1024 * 1024 * 1024;
        private const long RetainBudgetMinimumWasteBytes = 2L * 1024 * 1024;
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
        // Byte-budgeted retention (CUDA): oldest first, evicted from the front.
        private readonly List<PoolBlock> _budgeted = new List<PoolBlock>();
        private readonly long _retainBudgetBytes;
        private long _budgetedBytes;
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
        private readonly GgmlBackendType _backendType;
        private int _maxPooledBlocks;
        private readonly nuint _maxRetainedBlockSize;
        private long _poolHits, _freshAllocations, _systemFrees, _systemFreeFailures;
        private long _retainedBytes, _peakRetainedBytes;
        private readonly bool _hostHugePageAdviceEnabled;
        private long _hugePageAdviceRequests, _hugePageAdviceAccepted, _hugePageAdviceFailures, _hugePageAdvisedBytes;

        public GgmlMemoryPool(GgmlBackendType backendType)
            : this(backendType, Environment.GetEnvironmentVariable("TS_GGML_HOST_HUGEPAGES"))
        {
        }

        internal GgmlMemoryPool(GgmlBackendType backendType, string hugePagesSetting)
            : this(backendType, hugePagesSetting, backendType == GgmlBackendType.Cuda
                ? ResolveRetainBudgetBytes(Environment.GetEnvironmentVariable("TS_GGML_HOST_POOL_MB"),
                    GC.GetGCMemoryInfo().TotalAvailableMemoryBytes)
                : 0)
        {
        }

        internal GgmlMemoryPool(GgmlBackendType backendType, string hugePagesSetting, long retainBudgetBytes)
        {
            _backendType = backendType;
            _retainBudgetBytes = Math.Max(0, retainBudgetBytes);
            _hostHugePageAdviceEnabled = ShouldAdviseHostHugePages(
                OperatingSystem.IsLinux(), backendType, hugePagesSetting, (nuint)HostHugePageMinimumBytes);
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

        internal static long ResolveRetainBudgetBytes(string setting, long availableMemoryBytes)
        {
            if (long.TryParse(setting, out long mb) && mb >= 0)
                return mb * 1024 * 1024;
            return availableMemoryBytes > 0 ? Math.Min(CudaRetainBudgetCapBytes, availableMemoryBytes / 16) : 0;
        }

        internal static bool ShouldAdviseHostHugePages(bool isLinux, GgmlBackendType backendType,
            string setting, nuint mappedBytes)
            => isLinux && backendType == GgmlBackendType.Cuda && setting == "1"
                && mappedBytes >= (nuint)HostHugePageMinimumBytes;

        // Explicit owner opt-in; constructing a multi-device context alone does
        // not change retention for other models or sequential layer placement.
        internal bool ConfigureCudaTensorParallel(int degree)
        {
            if (degree <= 0) throw new ArgumentOutOfRangeException(nameof(degree));
            lock (_lock)
            {
                if (_closed || _backendType != GgmlBackendType.Cuda || degree == 1)
                    return false;
                _maxPooledBlocks = Math.Max(_maxPooledBlocks,
                    CudaMaxPooledBlocks * Math.Min(degree, CudaTpMaxPooledBlocks / CudaMaxPooledBlocks));
                return true;
            }
        }

        internal GgmlMemoryPoolStatistics GetStatistics()
        {
            lock (_lock)
                return new GgmlMemoryPoolStatistics(_poolHits, _freshAllocations, _systemFrees,
                    _systemFreeFailures, _available.Count + _budgeted.Count, _outstanding.Count, _retainedBytes,
                    _peakRetainedBytes, _maxPooledBlocks, (long)_maxRetainedBlockSize, _closed)
                {
                    HugePageAdviceEnabled = _hostHugePageAdviceEnabled,
                    HugePageAdviceRequests = _hugePageAdviceRequests,
                    HugePageAdviceAccepted = _hugePageAdviceAccepted,
                    HugePageAdviceFailures = _hugePageAdviceFailures,
                    HugePageAdvisedBytes = _hugePageAdvisedBytes,
                };
        }

        /// <summary>The host memory this pool holds: every block it handed out and has not been given back, and every freed
        /// block it keeps for reuse. A GGML device copy of any of it is keyed inside these ranges.</summary>
        internal List<(IntPtr Base, long Length)> BlockRanges()
        {
            lock (_lock)
            {
                var ranges = new List<(IntPtr, long)>(_outstanding.Count + _available.Count + _budgeted.Count);
                foreach (PoolBlock b in _outstanding.Values) ranges.Add((b.Ptr, (long)b.Size));
                foreach (PoolBlock b in _available) ranges.Add((b.Ptr, (long)b.Size));
                foreach (PoolBlock b in _budgeted) ranges.Add((b.Ptr, (long)b.Size));
                return ranges;
            }
        }

        public IntPtr Allocate(long byteLength) => AllocateReportingZero(byteLength, out _);

        /// <param name="zeroed">True when the block is a fresh anonymous mapping, which the OS
        /// hands out zero-filled; a reused pooled block and the heap fallback may hold old bytes.</param>
        public IntPtr AllocateReportingZero(long byteLength, out bool zeroed)
        {
            zeroed = false;
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
                    _retainedBytes -= (long)block.Size;
                    _poolHits++;
                    _outstanding[block.Ptr] = block;
                    return block.Ptr;
                }

                // Budgeted blocks only serve requests they fit closely, so one large
                // freed buffer is not tied up by a much smaller allocation.
                nuint waste = (nuint)Math.Max(RetainBudgetMinimumWasteBytes, (long)(alignedSize / 4));
                for (int i = 0; i < _budgeted.Count; i++)
                {
                    nuint blockSize = _budgeted[i].Size;
                    if (blockSize >= alignedSize && blockSize - alignedSize <= waste && blockSize < bestSize)
                    {
                        bestIdx = i;
                        bestSize = blockSize;
                        if (blockSize == alignedSize)
                            break;
                    }
                }
                if (bestIdx >= 0)
                {
                    PoolBlock block = _budgeted[bestIdx];
                    _budgeted.RemoveAt(bestIdx);
                    _budgetedBytes -= (long)block.Size;
                    _retainedBytes -= (long)block.Size;
                    _poolHits++;
                    _outstanding[block.Ptr] = block;
                    return block.Ptr;
                }
            }

            PoolBlock fresh = AllocateNew(alignedSize);
            lock (_lock)
            {
                _freshAllocations++;
                _outstanding[fresh.Ptr] = fresh;
            }
            zeroed = fresh.IsVirtual;   // mmap / VirtualAlloc pages are zero-filled
            return fresh.Ptr;
        }

        public void Free(IntPtr ptr, long byteLength)
        {
            if (ptr == IntPtr.Zero) return;

            PoolBlock block;
            List<PoolBlock> evicted = null;
            lock (_lock)
            {
                bool known = _outstanding.Remove(ptr, out block);
                if (!known)
                {
                    // Unknown pointer: not handed out by this pool. Freeing it
                    // with a guessed size/allocator can only corrupt the heap,
                    // so treat it as a virtual mapping of the caller-claimed
                    // size and let FreeToSystem leak it if munmap declines.
                    block = new PoolBlock(ptr, AlignSize((nuint)byteLength), _useVirtualAlloc);
                }

                if (!_closed && _maxPooledBlocks > 0 && block.Size <= _maxRetainedBlockSize &&
                    _available.Count < _maxPooledBlocks)
                {
                    _available.Add(block);
                    _retainedBytes += (long)block.Size;
                    _peakRetainedBytes = Math.Max(_peakRetainedBytes, _retainedBytes);
                    return;
                }

                if (!_closed && known && block.IsVirtual && (long)block.Size <= _retainBudgetBytes)
                {
                    _budgeted.Add(block);
                    _budgetedBytes += (long)block.Size;
                    _retainedBytes += (long)block.Size;
                    _peakRetainedBytes = Math.Max(_peakRetainedBytes, _retainedBytes);
                    if (_budgetedBytes <= _retainBudgetBytes)
                        return;
                    evicted = new List<PoolBlock>();
                    while (_budgetedBytes > _retainBudgetBytes && _budgeted.Count > 0)
                    {
                        PoolBlock oldest = _budgeted[0];
                        _budgeted.RemoveAt(0);
                        _budgetedBytes -= (long)oldest.Size;
                        _retainedBytes -= (long)oldest.Size;
                        evicted.Add(oldest);
                    }
                }
            }

            if (evicted == null)
            {
                ReleaseToSystem(block);
                return;
            }
            foreach (PoolBlock old in evicted)
                ReleaseToSystem(old);
        }

        private bool _closed;

        /// <summary>
        /// The owner is gone: return every pooled block now, and from here on unmap each
        /// block as it is freed instead of keeping it.
        ///
        /// <para>
        /// A pool belongs to one model's context, and nothing else can reach it. Keeping
        /// freed blocks for a model that has been unloaded is a leak for the life of the
        /// process: up to 32 blocks of up to 64 MiB of written memory per unload, and per
        /// Qwen-Image picture, whose text encoder is a model of its own. Measured in the Mac
        /// app before this existed: every switch between two chat models grew the next
        /// load's footprint by about 220 MB, and image turns by far more. Tensors that are
        /// freed after their model (a holder released late, a finalizer) land here too, so
        /// a closed pool does not keep them either.
        /// </para>
        /// </summary>
        /// <returns>The bytes returned to the system by this call.</returns>
        public long Close()
        {
            lock (_lock)
                _closed = true;
            return Trim();
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
                if (_available.Count == 0 && _budgeted.Count == 0)
                    return 0;
                release = new List<PoolBlock>(_available);
                release.AddRange(_budgeted);
                _available.Clear();
                _budgeted.Clear();
                _budgetedBytes = 0;
                _retainedBytes = 0;
            }
            long bytes = 0;
            foreach (PoolBlock block in release)
            {
                bytes += (long)block.Size;
                ReleaseToSystem(block);
            }
            return bytes;
        }

        private nuint AlignSize(nuint size)
        {
            if (size == 0) return (nuint)_pageSize;
            return ((size + (nuint)(_pageSize - 1)) / (nuint)_pageSize) * (nuint)_pageSize;
        }

        private PoolBlock AllocateNew(nuint alignedSize)
        {
            if (_useVirtualAlloc)
            {
                IntPtr ptr = AllocateVirtual(alignedSize);
                if (ptr != IntPtr.Zero)
                {
                    // Only our own successful Linux anonymous mapping is advised.
                    // Heap fallbacks and borrowed pointers never reach this path.
                    if (_hostHugePageAdviceEnabled && alignedSize >= (nuint)HostHugePageMinimumBytes)
                        AdviseHostHugePages(ptr, alignedSize);
                    return new PoolBlock(ptr, alignedSize, isVirtual: true);
                }
            }

            return new PoolBlock(Marshal.AllocHGlobal((nint)alignedSize), alignedSize, isVirtual: false);
        }

        private void AdviseHostHugePages(IntPtr ptr, nuint size)
        {
            // This is a hint, not MAP_HUGETLB or a forced collapse. A rejected or
            // unavailable hint must leave a usable ordinary mmap with the same
            // ownership/munmap extent. Counters avoid per-allocation log spam.
            bool accepted;
            try { accepted = UnixMadvise(ptr, size, UnixMadvHugePage) == 0; }
            catch (DllNotFoundException) { accepted = false; }
            catch (EntryPointNotFoundException) { accepted = false; }
            lock (_lock)
            {
                _hugePageAdviceRequests++;
                if (accepted)
                {
                    _hugePageAdviceAccepted++;
                    _hugePageAdvisedBytes += (long)size;
                }
                else _hugePageAdviceFailures++;
            }
        }

        private void ReleaseToSystem(in PoolBlock block)
        {
            // The OS operation remains outside the pool lock. Count completed
            // releases separately from failed unmaps; never claim a failed
            // release was reclaimed. Snapshots are exact once workers join.
            bool freed = FreeToSystem(block);
            lock (_lock)
            {
                if (freed) _systemFrees++;
                else _systemFreeFailures++;
            }
        }

        private static bool FreeToSystem(in PoolBlock block)
        {
            if (block.IsVirtual)
            {
                // If munmap/VirtualFree fails (e.g. vm.max_map_count exhausted),
                // leaking the mapping is the only safe outcome — this pointer
                // did not come from the C heap, so it must NEVER reach free().
                return FreeVirtual(block.Ptr, block.Size);
            }

            Marshal.FreeHGlobal(block.Ptr);
            return true;
        }

        internal void EnsureInitialBlocks()
        {
            lock (_lock)
            {
                while (_available.Count < _initialBlockCount)
                {
                    PoolBlock block = AllocateNew((nuint)BlockSize);
                    _available.Add(block);
                    _freshAllocations++;
                    _retainedBytes += (long)block.Size;
                    _peakRetainedBytes = Math.Max(_peakRetainedBytes, _retainedBytes);
                }
            }
        }

        private readonly struct PoolBlock
        {
            public readonly IntPtr Ptr;
            public readonly nuint Size;
            public readonly bool IsVirtual;

            public PoolBlock(IntPtr ptr, nuint size, bool isVirtual)
            {
                Ptr = ptr;
                Size = size;
                IsVirtual = isVirtual;
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
        private const int UnixMadvHugePage = 14; // Linux MADV_HUGEPAGE; never called on Darwin.

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

        [LibraryImport("libc", EntryPoint = "madvise", SetLastError = true)]
        private static partial int UnixMadvise(IntPtr addr, nuint length, int advice);
    }
}
