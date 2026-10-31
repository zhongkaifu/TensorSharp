// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the recurrent-layer prefill kernels (Mamba2, chunked Gated DeltaNet)
// (native/kernels/tensorsharp_ssm_prefill.cu, compiled to tensorsharp_ssm_prefill.ptx):
// WP3a/WP5 of the direct-CUDA prefill plan. Scaffold: the module holds only its capability
// probe, and nothing routes to it yet; the public entry point will be CudaSsmPrefillOps.
using System;

namespace TensorSharp.Cuda
{
    internal sealed class CudaSsmPrefillKernels : CudaPrefillModule
    {
        public const string PtxFileName = "tensorsharp_ssm_prefill.ptx";
        public const string CapsKernelName = "ts_ssm_prefill_caps";

        /// <summary>TS_SSM_PREFILL_VERSION in tensorsharp_ssm_prefill.cu: bump both together whenever a
        /// kernel is added or changes its signature (CudaPrefillModuleTests pins that they agree).</summary>
        public const int CurrentVersion = 1;

        private CudaSsmPrefillKernels(CudaModule module) : base(module, CapsKernelName, CurrentVersion)
        {
        }

        /// <summary>The module this process loads (the build output's PTX, or the fatbin beside it),
        /// whether or not it is current; see <see cref="CudaPrefillModule.IsCurrent"/>. Requires a
        /// current context.</summary>
        public static CudaSsmPrefillKernels Create() => FromModule(LoadModule(PtxFileName));

        /// <summary>Kernels over an already loaded module (tests that load a specific image), which
        /// they own. Requires the module's context to be current.</summary>
        internal static CudaSsmPrefillKernels FromModule(CudaModule module) => Own(module, m => new CudaSsmPrefillKernels(m));

        /// <summary>The current module, or false with the reason it cannot be used.</summary>
        public static bool TryCreate(out CudaSsmPrefillKernels kernels, out string reason)
            => TryCreate(PtxFileName, m => new CudaSsmPrefillKernels(m), out kernels, out reason);
    }
}
