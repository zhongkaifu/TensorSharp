// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Launchers for the graph executor's kernels (native/kernels/tensorsharp_graph_kernels.cu, compiled to
// tensorsharp_graph_kernels.ptx): the program ops with no generic tensor-op equivalent, with ggml's
// semantics.
using System;
using TensorSharp.Cuda.Interop;

namespace TensorSharp.Cuda
{
    internal sealed unsafe class CudaGraphKernels : CudaPrefillModule
    {
        public const string PtxFileName = "tensorsharp_graph_kernels.ptx";
        public const string CapsKernelName = "ts_graph_caps";

        /// <summary>TS_GRAPH_KERNELS_VERSION in tensorsharp_graph_kernels.cu: bump both together whenever a
        /// kernel is added or changes its signature.</summary>
        public const int CurrentVersion = 4;

        private readonly IntPtr _gatedDeltaNet, _gatedDeltaNetRows4, _softplus, _ssmConv, _deltaNetPrep, _gatedRmsNorm;

        private CudaGraphKernels(CudaModule module) : base(module, CapsKernelName, CurrentVersion)
        {
            if (!IsCurrent) return;
            _gatedDeltaNet = module.GetFunction("ts_graph_gated_delta_net_f32");
            _gatedDeltaNetRows4 = module.GetFunction("ts_graph_gated_delta_net_rows4_f32");
            _softplus = module.GetFunction("ts_graph_softplus_f32");
            _ssmConv = module.GetFunction("ts_graph_ssm_conv_f32");
            _deltaNetPrep = module.GetFunction("ts_graph_deltanet_prep_f32");
            _gatedRmsNorm = module.GetFunction("ts_graph_gated_rms_norm_f32");
        }

        /// <summary>The current module, or false with the reason it cannot be used (requires a current context).</summary>
        public static bool TryCreate(out CudaGraphKernels kernels, out string reason)
            => TryCreate(PtxFileName, m => new CudaGraphKernels(m), out kernels, out reason);

        /// <summary>ggml_gated_delta_net over contiguous F32 device buffers (see the kernel's contract);
        /// <paramref name="s"/> % 32 == 0 and <= 256.</summary>
        public void LaunchGatedDeltaNet(IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr beta, IntPtr state, IntPtr output,
            int s, int heads, int keyHeads, int tokens, int sequences, int slots, bool perKeyDecay, IntPtr stream)
        {
            int kda = perKeyDecay ? 1 : 0;
            void** args = stackalloc void*[] { &q, &k, &v, &g, &beta, &state, &output, &s, &heads, &keyHeads, &tokens, &sequences, &slots, &kda };
            CudaDriverApi.cuLaunchKernel(_gatedDeltaNet, (uint)(sequences * heads), (uint)((s + 3) / 4), 1, 128, 1, 1, 0, stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary><see cref="LaunchGatedDeltaNet"/> with four state rows a warp (a prompt chunk reads each token's q and k
        /// once for four rows); the same results.</summary>
        public void LaunchGatedDeltaNetRows4(IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr beta, IntPtr state, IntPtr output,
            int s, int heads, int keyHeads, int tokens, int sequences, int slots, bool perKeyDecay, IntPtr stream)
        {
            int kda = perKeyDecay ? 1 : 0;
            void** args = stackalloc void*[] { &q, &k, &v, &g, &beta, &state, &output, &s, &heads, &keyHeads, &tokens, &sequences, &slots, &kda };
            CudaDriverApi.cuLaunchKernel(_gatedDeltaNetRows4, (uint)(sequences * heads), (uint)((s + 15) / 16), 1, 128, 1, 1, 0, stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>ggml_softplus over <paramref name="n"/> contiguous floats.</summary>
        public void LaunchSoftplus(IntPtr x, IntPtr y, long n, IntPtr stream)
        {
            void** args = stackalloc void*[] { &x, &y, &n };
            CudaDriverApi.cuLaunchKernel(_softplus, (uint)((n + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>ggml_ssm_conv: x [n_s, channels, length] (time innermost), w [channels, width] (contiguous):
        /// out [n_s, length - width + 1, channels].</summary>
        public void LaunchSsmConv(IntPtr x, IntPtr w, IntPtr output, int channels, int length, int width, int sequences, IntPtr stream)
        {
            int nt = length - width + 1;
            long total = (long)sequences * nt * channels;
            void** args = stackalloc void*[] { &x, &w, &output, &channels, &length, &width, &nt, &sequences };
            CudaDriverApi.cuLaunchKernel(_ssmConv, (uint)((total + 255) / 256), 1, 1, 256, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>GraphFusion.DeltaNetPrep over contiguous F32 buffers (see the kernel's contract).</summary>
        public void LaunchDeltaNetPrep(IntPtr window, IntPtr qkv, IntPtr convW, IntPtr alpha, IntPtr beta, IntPtr dt, IntPtr a,
            IntPtr convIn, IntPtr q, IntPtr k, IntPtr v, IntPtr g, IntPtr betaOut,
            int channels, int convRows, int width, int d, int kHeads, int vHeads, int rows, float eps, IntPtr stream)
        {
            void** args = stackalloc void*[]
            {
                &window, &qkv, &convW, &alpha, &beta, &dt, &a, &convIn, &q, &k, &v, &g, &betaOut,
                &channels, &convRows, &width, &d, &kHeads, &vHeads, &rows, &eps,
            };
            CudaDriverApi.cuLaunchKernel(_deltaNetPrep, (uint)(2 * kHeads + vHeads), (uint)rows, 1, (uint)d, 1, 1, 0, stream,
                (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }

        /// <summary>GraphFusion.GatedRmsNorm over <paramref name="rows"/> contiguous rows of <paramref name="d"/>.</summary>
        public void LaunchGatedRmsNorm(IntPtr y, IntPtr z, IntPtr w, IntPtr output, int rows, int d, float eps, IntPtr stream)
        {
            void** args = stackalloc void*[] { &y, &z, &w, &output, &d, &eps };
            CudaDriverApi.cuLaunchKernel(_gatedRmsNorm, (uint)rows, 1, 1, (uint)d, 1, 1, 0, stream, (IntPtr)args, IntPtr.Zero).ThrowOnError();
        }
    }
}
