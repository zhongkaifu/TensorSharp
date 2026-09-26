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
using System.Collections.Concurrent;
using System.Runtime.InteropServices;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Native host buffers for weight copies and repack staging: the fused and
    /// requantized weights a load builds, and the scratch a device upload copies from.
    ///
    /// On macOS and iOS a buffer of <see cref="MappedThreshold"/> or more is mapped
    /// straight from the kernel instead of malloc'd. The system allocator keeps freed
    /// huge blocks mapped and dirty ("Malloc Large (empty)" in vmmap), and
    /// malloc_zone_pressure_relief does not return them, so every buffer a load frees
    /// after its device upload stayed in the process footprint: 7.5 GB after a
    /// Qwen3.8-27B UD-Q4_K_XL MLX load. munmap gives the pages back at once.
    /// Elsewhere (glibc already maps large blocks, Windows) it is the aligned heap.
    ///
    /// Free only with <see cref="Free"/>: a mapped buffer is not a heap pointer.
    /// </summary>
    public static unsafe partial class HostBuffers
    {
        public const long MappedThreshold = 16L << 20;

        private const int PROT_READ = 0x1;
        private const int PROT_WRITE = 0x2;
        private const int MAP_PRIVATE = 0x2;
        private const int MAP_ANON = 0x1000;

        private static readonly ConcurrentDictionary<nint, nuint> Mapped = new();
        private static readonly bool MapLargeBuffers = OperatingSystem.IsMacOS() || OperatingSystem.IsIOS();

        /// <summary>Allocate <paramref name="size"/> bytes aligned to at least
        /// <paramref name="alignment"/>. Mapped buffers are page-aligned and zeroed;
        /// heap buffers are neither zeroed nor more than <paramref name="alignment"/>-aligned.</summary>
        public static IntPtr Allocate(long size, int alignment = 64)
        {
            if (size <= 0)
                throw new ArgumentOutOfRangeException(nameof(size));

            if (MapLargeBuffers && size >= MappedThreshold)
            {
                void* mapped = null;
                try { mapped = mmap(null, (nuint)size, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0); }
                catch (DllNotFoundException) { }
                catch (EntryPointNotFoundException) { }
                if (mapped != null && (nint)mapped != -1)
                {
                    Mapped[(nint)mapped] = (nuint)size;
                    return (IntPtr)mapped;
                }
            }

            void* heap = NativeMemory.AlignedAlloc((nuint)size, (nuint)alignment);
            if (heap == null)
                throw new OutOfMemoryException($"Unable to allocate {size} bytes of host buffer.");
            return (IntPtr)heap;
        }

        public static void Free(IntPtr ptr)
        {
            if (ptr == IntPtr.Zero)
                return;
            if (Mapped.TryRemove(ptr, out nuint length))
            {
                _ = munmap((void*)ptr, length);
                return;
            }
            NativeMemory.AlignedFree(ptr.ToPointer());
        }

        [LibraryImport("libc", EntryPoint = "mmap", SetLastError = true)]
        private static partial void* mmap(void* addr, nuint length, int prot, int flags, int fd, long offset);

        [LibraryImport("libc", EntryPoint = "munmap", SetLastError = true)]
        private static partial int munmap(void* addr, nuint length);
    }
}
