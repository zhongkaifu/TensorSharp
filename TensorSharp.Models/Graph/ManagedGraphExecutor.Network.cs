// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Threading.Tasks;

using TensorSharp.Cpu;

namespace TensorSharp.Models.Graph
{
    /// <summary>The managed kernels of the ops non-decoder networks add (TensorGraphExecutor.Network.cs): single host passes
    /// where the reference runs a chain of elementwise ops.</summary>
    public sealed partial class ManagedGraphExecutor
    {
        // Abramowitz and Stegun 7.1.26 (ggml-metal's erf_approx): |error of erf| below 1.5e-7.
        private const float ErfP = 0.3275911f, ErfA1 = 0.254829592f, ErfA2 = -0.284496736f, ErfA3 = 1.421413741f,
            ErfA4 = -1.453152027f, ErfA5 = 1.061405429f, InvSqrt2 = 0.70710678118654752f;

        /// <summary>Elements a parallel chunk of an elementwise pass takes.</summary>
        private const long ElementwiseChunk = 1 << 16;

        /// <summary>0.5 x (1 + erf(x / sqrt 2)) as ggml-metal's kernel evaluates it.</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        internal static float GeluErfValue(float x)
        {
            float t = 1f / (1f + ErfP * MathF.Abs(x) * InvSqrt2);
            float poly = ((((ErfA5 * t + ErfA4) * t + ErfA3) * t + ErfA2) * t + ErfA1) * t;
            float erf = 1f - poly * MathF.Exp(-0.5f * x * x);
            return 0.5f * x * (1f + (x < 0f ? -erf : erf));
        }

        protected override unsafe Tensor GeluErf(Tensor x)
        {
            if (x.ElementType != DType.Float32 || !x.IsContiguous()) return base.GeluErf(x);
            long count = x.ElementCount();
            var result = new Tensor(Allocator, DType.Float32, x.Sizes);
            long sa = (long)TensorComputePrimitives.GetFloatPointer(x), da = (long)TensorComputePrimitives.GetFloatPointer(result);
            long chunks = (count + ElementwiseChunk - 1) / ElementwiseChunk;

            void Chunk(long index)
            {
                long first = index * ElementwiseChunk, n = Math.Min(ElementwiseChunk, count - first);
                float* s = (float*)sa + first, d = (float*)da + first;
                for (long i = 0; i < n; i++) d[i] = GeluErfValue(s[i]);
            }

            if (chunks > 1) Parallel.For(0L, chunks, Chunk);
            else if (chunks == 1) Chunk(0);
            return result;
        }

        /// <summary>
        /// <see cref="TensorSharp.Graph.GraphOp.DepthwiseConv2d"/> in one host pass: each output row (n, oy) in parallel, every
        /// output pixel's channels summed over the taps in (ky, kx) order (as ggml-cpu sums them), the channels vectorized
        /// with the taps laid out channels innermost.
        /// </summary>
        protected override unsafe Tensor DepthwiseConv2d(Tensor x, Tensor kernel, int s0, int s1, int p0, int p1, int d0, int d1)
        {
            if (x.ElementType != DType.Float32 || kernel.ElementType != DType.Float32 || !x.IsContiguous() || !kernel.IsContiguous())
                return base.DepthwiseConv2d(x, kernel, s0, s1, p0, p1, d0, d1);
            long b = x.Sizes[0], h = x.Sizes[1], w = x.Sizes[2], c = x.Sizes[3], kh = kernel.Sizes[1], kw = kernel.Sizes[2];
            long oh = (h + 2L * p1 - d1 * (kh - 1) - 1) / s1 + 1, ow = (w + 2L * p0 - d0 * (kw - 1) - 1) / s0 + 1;
            float* kp = TensorComputePrimitives.GetFloatPointer(kernel);
            var taps = new float[kh * kw * c];
            for (long ch = 0; ch < c; ch++)
                for (long t = 0; t < kh * kw; t++)
                    taps[t * c + ch] = kp[ch * kh * kw + t];
            var result = new Tensor(Allocator, DType.Float32, b, oh, ow, c);
            long xa = (long)TensorComputePrimitives.GetFloatPointer(x), ra = (long)TensorComputePrimitives.GetFloatPointer(result);
            int lanes = Vector<float>.Count;
            fixed (float* tp = taps)
            {
                long ta = (long)tp;

                void Row(long row)
                {
                    long n = row / oh, oy = row % oh;
                    float* src = (float*)xa + n * h * w * c;
                    float* dst = (float*)ra + row * ow * c;
                    for (long ox = 0; ox < ow; ox++)
                    {
                        float* o = dst + ox * c;
                        new Span<float>(o, checked((int)c)).Clear();
                        for (long ky = 0; ky < kh; ky++)
                        {
                            long iy = oy * s1 + ky * d1 - p1;
                            if (iy < 0 || iy >= h) continue;
                            for (long kx = 0; kx < kw; kx++)
                            {
                                long ix = ox * s0 + kx * d0 - p0;
                                if (ix < 0 || ix >= w) continue;
                                float* sp = src + (iy * w + ix) * c, k = (float*)ta + (ky * kw + kx) * c;
                                long i = 0;
                                for (; i <= c - lanes; i += lanes)
                                    TensorComputePrimitives.StoreVector(o + i, TensorComputePrimitives.LoadVector(o + i)
                                        + TensorComputePrimitives.LoadVector(k + i) * TensorComputePrimitives.LoadVector(sp + i));
                                for (; i < c; i++) o[i] += k[i] * sp[i];
                            }
                        }
                    }
                }

                long rows = b * oh;
                if (rows > 1 && rows * ow * c * kh * kw >= 1L << 18) Parallel.For(0L, rows, Row);
                else for (long r = 0; r < rows; r++) Row(r);
            }
            return result;
        }
    }
}
