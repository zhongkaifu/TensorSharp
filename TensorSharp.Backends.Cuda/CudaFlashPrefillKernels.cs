// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the tensor-core flash prefill attention kernels
// (native/kernels/tensorsharp_flash_prefill.cu, compiled to tensorsharp_flash_prefill.ptx):
// WP1 of the direct-CUDA prefill plan. Scaffold: the module holds only its capability probe,
// and nothing routes to it yet; the attention entry points are CudaKernelOps.TryGqaPrefillAttention
// and TryGqaPrefillAttentionWithSinks.
using System;

namespace TensorSharp.Cuda
{
    internal sealed class CudaFlashPrefillKernels : CudaPrefillModule
    {
        public const string PtxFileName = "tensorsharp_flash_prefill.ptx";
        public const string CapsKernelName = "ts_flash_prefill_caps";

        /// <summary>TS_FLASH_PREFILL_VERSION in tensorsharp_flash_prefill.cu: bump both together whenever a
        /// kernel is added or changes its signature (CudaPrefillModuleTests pins that they agree).</summary>
        public const int CurrentVersion = 1;

        private CudaFlashPrefillKernels(CudaModule module) : base(module, CapsKernelName, CurrentVersion)
        {
        }

        /// <summary>The module this process loads (the build output's PTX, or the fatbin beside it),
        /// whether or not it is current; see <see cref="CudaPrefillModule.IsCurrent"/>. Requires a
        /// current context.</summary>
        public static CudaFlashPrefillKernels Create() => FromModule(LoadModule(PtxFileName));

        /// <summary>Kernels over an already loaded module (tests that load a specific image), which
        /// they own. Requires the module's context to be current.</summary>
        internal static CudaFlashPrefillKernels FromModule(CudaModule module) => Own(module, m => new CudaFlashPrefillKernels(m));

        /// <summary>The current module, or false with the reason it cannot be used.</summary>
        public static bool TryCreate(out CudaFlashPrefillKernels kernels, out string reason)
            => TryCreate(PtxFileName, m => new CudaFlashPrefillKernels(m), out kernels, out reason);
    }
}
