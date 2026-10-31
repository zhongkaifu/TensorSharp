// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the grouped on-device MoE prefill kernels
// (native/kernels/tensorsharp_moe_prefill.cu, compiled to tensorsharp_moe_prefill.ptx):
// WP2b/WP3c of the direct-CUDA prefill plan. Scaffold: the module holds only its capability
// probe, and nothing routes to it yet; the public entry point will be CudaMoePrefillOps.
using System;

namespace TensorSharp.Cuda
{
    internal sealed class CudaMoePrefillKernels : CudaPrefillModule
    {
        public const string PtxFileName = "tensorsharp_moe_prefill.ptx";
        public const string CapsKernelName = "ts_moe_prefill_caps";

        /// <summary>TS_MOE_PREFILL_VERSION in tensorsharp_moe_prefill.cu: bump both together whenever a
        /// kernel is added or changes its signature (CudaPrefillModuleTests pins that they agree).</summary>
        public const int CurrentVersion = 1;

        private CudaMoePrefillKernels(CudaModule module) : base(module, CapsKernelName, CurrentVersion)
        {
        }

        /// <summary>The module this process loads (the build output's PTX, or the fatbin beside it),
        /// whether or not it is current; see <see cref="CudaPrefillModule.IsCurrent"/>. Requires a
        /// current context.</summary>
        public static CudaMoePrefillKernels Create() => FromModule(LoadModule(PtxFileName));

        /// <summary>Kernels over an already loaded module (tests that load a specific image), which
        /// they own. Requires the module's context to be current.</summary>
        internal static CudaMoePrefillKernels FromModule(CudaModule module) => Own(module, m => new CudaMoePrefillKernels(m));

        /// <summary>The current module, or false with the reason it cannot be used.</summary>
        public static bool TryCreate(out CudaMoePrefillKernels kernels, out string reason)
            => TryCreate(PtxFileName, m => new CudaMoePrefillKernels(m), out kernels, out reason);
    }
}
