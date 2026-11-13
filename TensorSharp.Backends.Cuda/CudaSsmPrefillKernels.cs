// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the recurrent-layer prefill kernels (native/kernels/tensorsharp_ssm_prefill.cu,
// compiled to tensorsharp_ssm_prefill.ptx): the chunked Gated DeltaNet CudaGraphExecutor runs a
// prompt chunk's ggml_gated_delta_net on.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed class CudaSsmPrefillKernels : CudaPrefillModule
    {
        public const string PtxFileName = "tensorsharp_ssm_prefill.ptx";
        public const string CapsKernelName = "ts_ssm_prefill_caps";

        /// <summary>TS_SSM_PREFILL_VERSION in tensorsharp_ssm_prefill.cu: bump both together whenever a
        /// kernel is added or changes its signature (CudaPrefillModuleTests pins that they agree).</summary>
        public const int CurrentVersion = 2;

        /// <summary>Key and value head width <see cref="LaunchGatedDeltaNetChunks"/> is written for.</summary>
        public const int GdnHeadDim = 128;

        private readonly IntPtr _gdnChunk;

        private CudaSsmPrefillKernels(CudaModule module) : base(module, CapsKernelName, CurrentVersion)
        {
            if (IsCurrent)
                _gdnChunk = module.GetFunction("ts_ssm_gdn_chunk_f32");
        }

        /// <summary>Whether <see cref="LaunchGatedDeltaNetChunks"/> runs here (its tensor-core body needs sm_80 code).</summary>
        public bool RunsGatedDeltaNetChunks => IsCurrent && ArchGuardedKernelsUsable;

        /// <summary>
        /// ggml_gated_delta_net over <paramref name="tokens"/> rows of <paramref name="sequences"/> sequences in chunks of
        /// 16 (ts_ssm_gdn_chunk_f32): q, k [n_s, n_t, hk, 128], v [n_s, n_t, hv, 128], g and beta [n_s, n_t, hv], the
        /// state [n_s, hv, 128, 128] read from <paramref name="stateIn"/> and the state after the last row written to
        /// <paramref name="stateOut"/> (it may be the same memory), out [n_s, n_t, hv, 128].
        /// </summary>
        public unsafe void LaunchGatedDeltaNetChunks(IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr beta, IntPtr stateIn,
            IntPtr stateOut, IntPtr output, int keyHeads, int valueHeads, int tokens, int sequences, IntPtr stream)
        {
            IntPtr a0 = q, a1 = k, a2 = v, a6 = g, a7 = beta, a8 = stateIn, a9 = stateOut, a10 = output;
            int a3 = keyHeads, a4 = valueHeads, a5 = tokens;
            void** args = stackalloc void*[] { &a0, &a1, &a2, &a3, &a4, &a5, &a6, &a7, &a8, &a9, &a10 };
            CudaDriverApi.cuLaunchKernel(_gdnChunk, (uint)(valueHeads * (GdnHeadDim / 32)), (uint)sequences, 1,
                256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
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
