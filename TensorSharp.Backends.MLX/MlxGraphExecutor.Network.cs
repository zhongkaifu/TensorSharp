// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;

namespace TensorSharp.MLX
{
    /// <summary>MLX's kernels for the ops non-decoder networks add (TensorGraphExecutor.Network.cs): MLX's erf, its grouped
    /// conv2d and its fused layer norm, each one lazy MLX op where the reference chains several.</summary>
    public sealed partial class MlxGraphExecutor
    {
        /// <summary>0.5 x (1 + erf(x / sqrt 2)) with MLX's erf.</summary>
        protected override Tensor GeluErf(Tensor x)
        {
            if (!(x.Storage is MlxStorage) || x.ElementType != DType.Float32) return base.GeluErf(x);
            var arrays = new MlxNative.MlxArray[9];
            try
            {
                arrays[0] = View(x);
                arrays[1] = MlxNative.NewScalar(0.70710678118654752f);
                arrays[2] = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, arrays[0], arrays[1]);
                arrays[3] = MlxNative.Erf(arrays[2]);
                arrays[4] = MlxNative.NewScalar(1f);
                arrays[5] = MlxNative.Binary(MlxNative.MlxBinaryOp.Add, arrays[3], arrays[4]);
                arrays[6] = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, arrays[0], arrays[5]);
                arrays[7] = MlxNative.NewScalar(0.5f);
                arrays[8] = MlxNative.Binary(MlxNative.MlxBinaryOp.Mul, arrays[6], arrays[7]);
                return Wrap(arrays[8], x.Sizes);
            }
            finally
            {
                foreach (var a in arrays) MlxNative.FreeArray(a);
            }
        }

        /// <summary>The depthwise conv as MLX's conv2d with one group a channel: x [N, H, W, C] is NHWC as it lies, the kernel
        /// [C, KH, KW] is the [C_out, KH, KW, 1] weight.</summary>
        protected override Tensor DepthwiseConv2d(Tensor x, Tensor kernel, int s0, int s1, int p0, int p1, int d0, int d1)
        {
            if (!(x.Storage is MlxStorage && kernel.Storage is MlxStorage) || x.ElementType != DType.Float32
                || kernel.ElementType != DType.Float32)
                return base.DepthwiseConv2d(x, kernel, s0, s1, p0, p1, d0, d1);
            long b = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2], c = x.Sizes[3], kh = kernel.Sizes[1], kw = kernel.Sizes[2];
            long oh = (h + 2L * p1 - d1 * (kh - 1) - 1) / s1 + 1, ow = (w + 2L * p0 - d0 * (kw - 1) - 1) / s0 + 1;
            MlxNative.MlxArray input = default, weight = default, r = default;
            try
            {
                input = View(x);
                weight = ((MlxStorage)kernel.Storage).CreateArrayView(new[] { (int)c, (int)kh, (int)kw, 1 },
                    new[] { kernel.Strides[0], kernel.Strides[1], kernel.Strides[2], 1L }, kernel.StorageOffset);
                r = MlxNative.Conv2d(input, weight, s1, s0, p1, p0, d1, d0, (int)c);
                return Wrap(r, new[] { b, oh, ow, c });
            }
            finally
            {
                foreach (var a in new[] { input, weight, r }) MlxNative.FreeArray(a);
            }
        }

        /// <summary>ggml_norm as MLX's fused layer norm with no weight and no bias.</summary>
        protected override Tensor NormRows(Tensor x, float eps)
        {
            if (!(x.Storage is MlxStorage) || x.ElementType != DType.Float32) return base.NormRows(x, eps);
            MlxNative.MlxArray input = default, r = default;
            try
            {
                input = View(x);
                r = MlxNative.FastLayerNorm(input, default, default, eps);
                return Wrap(r, x.Sizes);
            }
            finally
            {
                MlxNative.FreeArray(input);
                MlxNative.FreeArray(r);
            }
        }
    }
}
