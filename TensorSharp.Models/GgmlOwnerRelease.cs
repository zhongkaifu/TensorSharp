// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models
{
    /// <summary>
    /// An owner of host memory whose GGML device copies (and the backend's other records) are keyed inside it: a model, a
    /// pipeline component. It lists that memory so it can be released on its own (<see cref="GgmlOwnerRelease"/>) while
    /// other owners stay live.
    /// </summary>
    internal interface IGgmlHostMemoryOwner
    {
        /// <summary>Adds the owner's host memory: <paramref name="ranges"/> it owns whole (a GGUF mapping, a memory pool's
        /// blocks, a buffer it allocated), and the <paramref name="keys"/> its weights' device copies are kept under where
        /// those need not lie inside the ranges (an opaque cache key after a preload, a buffer whose length it does not
        /// keep).</summary>
        void CollectHostMemory(List<(IntPtr Base, long Length)> ranges, List<IntPtr> keys);
    }

    /// <summary>
    /// Releases owners' GGML device memory and nothing else: every device copy, zero-copy wrap and backend record keyed in
    /// their memory, and the shared graph caches (persistent decode and draft graphs, attention sessions, vendor
    /// workspaces), which every live model rebuilds on its next call. Another live model keeps its device copies, the
    /// plans bound to them and its device-resident state, which <see cref="GgmlBasicOps.ClearHostBufferCache"/> would free:
    /// that wipe is the last GGML model's (<c>ModelBase.Dispose</c>) and backend shutdown's alone.
    /// </summary>
    internal static class GgmlOwnerRelease
    {
        /// <summary>Releases the device memory of <paramref name="owners"/> (null entries skipped); returns how many device
        /// copies went.</summary>
        public static int Release(params IGgmlHostMemoryOwner[] owners)
        {
            var ranges = new List<(IntPtr Base, long Length)>();
            var keys = new List<IntPtr>();
            foreach (IGgmlHostMemoryOwner owner in owners)
                owner?.CollectHostMemory(ranges, keys);
            keys.RemoveAll(k => k == IntPtr.Zero);
            int dropped = GgmlBasicOps.ReleaseHostRanges(ranges, keys);
            GgmlBasicOps.ResetSharedGraphCaches();
            return dropped;
        }

        /// <summary>The ranges a GGUF file has mapped (also those it has unmapped since), without mapping anything.</summary>
        public static void AddMappings(List<(IntPtr Base, long Length)> ranges, GgufFile file)
        {
            if (file != null) ranges.AddRange(file.MappingsMade());
        }
    }
}
