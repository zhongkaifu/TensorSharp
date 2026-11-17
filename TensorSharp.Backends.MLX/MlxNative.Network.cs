// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;

namespace TensorSharp.MLX
{
    /// <summary>The MLX ops the graph executor's non-decoder ops run on (MlxGraphExecutor.Network.cs).</summary>
    internal static partial class MlxNative
    {
        /// <summary>erf, elementwise.</summary>
        internal static MlxArray Erf(MlxArray input)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_erf(out result, input, DefaultStream()), "running MLX erf");
                return result;
            });
        }

        /// <summary>mlx_conv2d: input [N, H, W, C_in], weight [C_out, KH, KW, C_in / groups]: [N, OH, OW, C_out]. Stride,
        /// padding and dilation are (H, W) pairs.</summary>
        internal static MlxArray Conv2d(MlxArray input, MlxArray weight, int strideH, int strideW, int padH, int padW,
            int dilationH, int dilationW, int groups)
        {
            return MlxWorker.Shared.Invoke(() =>
            {
                MlxArray result;
                Check(mlx_conv2d(out result, input, weight, strideH, strideW, padH, padW, dilationH, dilationW, groups, DefaultStream()),
                    "running MLX conv2d");
                return result;
            });
        }

        [LibraryImport(LibraryName, EntryPoint = "mlx_erf")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_erf(out MlxArray result, MlxArray input, MlxStream stream);

        [LibraryImport(LibraryName, EntryPoint = "mlx_conv2d")]
        [UnmanagedCallConv(CallConvs = new[] { typeof(CallConvCdecl) })]
        private static partial int mlx_conv2d(out MlxArray result, MlxArray input, MlxArray weight, int stride0, int stride1,
            int padding0, int padding1, int dilation0, int dilation1, int groups, MlxStream stream);
    }
}
