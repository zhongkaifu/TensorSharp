// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Linq;

using TensorSharp.Runtime;

namespace TensorSharp.Models.Architecture
{
    /// <summary>Backend sets that recur in the families' serving contracts.</summary>
    internal static class ServingBackends
    {
        /// <summary>The four GGML backends.</summary>
        public static readonly BackendType[] Ggml =
            { BackendType.GgmlCpu, BackendType.GgmlMetal, BackendType.GgmlCuda, BackendType.GgmlVulkan };

        /// <summary>Where the per-op (managed) tensor-parallel forwards run: the backends whose
        /// <see cref="TensorSharp.ITensorParallelGroup"/> spans several devices.</summary>
        public static readonly BackendType[] PerOpTensorParallel =
            { BackendType.Cuda, BackendType.GgmlCuda, BackendType.GgmlVulkan };

        /// <summary>The multi-device GGML backends.</summary>
        public static readonly BackendType[] GgmlMultiDevice = { BackendType.GgmlCuda, BackendType.GgmlVulkan };

        public static BackendType[] All => ServingContract.AllBackends.ToArray();
    }
}
