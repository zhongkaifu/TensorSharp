// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp.GGML;
using TensorSharp.Runtime;

namespace TensorSharp.Models.MiniMaxH3
{
    /// <summary>Tells the native weight uploads which file each of a network's mappings is.
    /// A weight is bound as a pointer into the mapped GGUF, and uploading it out of the
    /// mapping pages the file in through the page cache from inside the copy. Where the
    /// native side can read the file itself (ggml_ops_direct_read.cpp) it only needs to know
    /// which file a pointer belongs to, and the page cache stays out of it.</summary>
    internal static class MiniMaxH3DirectRead
    {
        /// <summary>True when uploads on <paramref name="backend"/> read weight files directly,
        /// so that a read-ahead of the same file would only read it a second time.</summary>
        public static bool Supported(BackendType backend) =>
            backend == BackendType.GgmlCuda && GgmlBasicOps.DirectReadSupported();

        /// <summary>Registers every mapping behind <paramref name="gguf"/> and returns the ones
        /// to hand to <see cref="Unregister"/>. Call it last in a constructor: a registration
        /// that outlives its mapping would describe whatever is mapped there next.</summary>
        public static IntPtr[] Register(GgufFile gguf, BackendType backend)
        {
            if (!Supported(backend)) return Array.Empty<IntPtr>();
            var bases = new List<IntPtr>();
            foreach (var (address, length, path) in gguf.MappedRegions())
                if (GgmlBasicOps.DirectReadRegister(address, length, Path.GetFullPath(path)))
                    bases.Add(address);
            return bases.ToArray();
        }

        public static void Unregister(IntPtr[] bases)
        {
            if (bases == null) return;
            foreach (IntPtr address in bases) GgmlBasicOps.DirectReadUnregister(address);
        }
    }
}
