// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.
using System;
using System.Numerics;
using System.Runtime.CompilerServices;
using System.Runtime.Intrinsics;
using System.Runtime.Intrinsics.Arm;
using System.Threading.Tasks;

namespace TensorSharp.Models
{
    // ------------------------------------------------------------------
    // BF16 weights on ARM64.
    //
    // Hosts without the x86 float panel run BF16 products through
    // DequantMatMulColumns: each weight row widened into an F32 scratch, then
    // dotted with the activation rows four at a time (TensorComputePrimitives.Dot4,
    // one accumulator a row) and the leftover rows one at a time (Dot, two
    // accumulators). These kernels compute exactly that - the same widening (a
    // BF16 value is the top half of its F32), the same multiplies and adds in the
    // same order per output, the same Vector.Sum reduction and scalar tail - with
    // the widening done in registers and a register tile of four rows by four
    // columns, so the scratch round trip and the re-read of the weight row per
    // four rows are gone. The bits do not change (ManagedQuantArmTests).
    // ------------------------------------------------------------------
    internal static partial class ManagedQuantizedOps
    {
        /// <summary>DequantMatMulColumns' result for BF16 weights on ARM64 (see above); false elsewhere.</summary>
        internal static unsafe bool TryBf16Arm(
            int ggmlType, byte* weights, long rowBytes, int inDim, int outDim,
            float* input, int inputRowStride, int rowCount, float* output, int outputRowStride, ParallelOptions options)
        {
            if ((GgmlTensorType)ggmlType != GgmlTensorType.BF16 || !AdvSimd.Arm64.IsSupported || Vector<float>.Count != 4
                || rowCount <= 0 || outDim <= 0 || inDim <= 0)
                return false;

            int groups = (outDim + 3) / 4;
            int dop = ResolveDop(options);
            nint w = (nint)weights, x = (nint)input, o = (nint)output;
            int inStride = inputRowStride, outStride = outputRowStride;
            void RunGroups(int g0, int g1)
            {
                for (int g = g0; g < g1; g++)
                {
                    int c0 = g * 4, cols = Math.Min(4, outDim - c0);
                    ushort* w0 = (ushort*)((byte*)w + c0 * rowBytes);
                    ushort* w1 = cols > 1 ? (ushort*)((byte*)w0 + rowBytes) : w0;
                    ushort* w2 = cols > 2 ? (ushort*)((byte*)w0 + 2 * rowBytes) : w0;
                    ushort* w3 = cols > 3 ? (ushort*)((byte*)w0 + 3 * rowBytes) : w0;
                    int r = 0;
                    for (; r + 4 <= rowCount; r += 4)
                        Bf16Tile4x4(w0, w1, w2, w3, (float*)x + (long)r * inStride, inStride, inDim,
                            (float*)o + (long)r * outStride + c0, outStride, cols);
                    for (; r < rowCount; r++)
                        Bf16Row1x4(w0, w1, w2, w3, (float*)x + (long)r * inStride, inDim, (float*)o + (long)r * outStride + c0, cols);
                }
            }

            if (dop <= 1 || (long)rowCount * outDim < 512 || outDim < 128)
            {
                RunGroups(0, groups);
                return true;
            }
            int colBlock = ParallelColumnBlock(outDim, rowBytes, rowCount, dop);
            int groupsPerBlock = Math.Max(1, (colBlock + 3) / 4);
            int blocks = (groups + groupsPerBlock - 1) / groupsPerBlock;
            RunParallelBlocks(blocks, options, b => RunGroups(b * groupsPerBlock, Math.Min(groups, (b + 1) * groupsPerBlock)));
            return true;
        }

        /// <summary>Four BF16 values widened to F32 (each the top half of its F32).</summary>
        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe Vector128<float> LoadBf16x4(ushort* p)
            => AdvSimd.Arm64.ZipLow(Vector128<ushort>.Zero, AdvSimd.LoadVector64(p).ToVector128Unsafe()).AsSingle();

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe float Bf16ToSingle(ushort v) => BitConverter.Int32BitsToSingle(v << 16);

        /// <summary>Dot4's arithmetic for four activation rows against four BF16 weight rows: per (row, column) one
        /// vector accumulator over the 4-value chunks, Vector.Sum, then the scalar tail.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Bf16Tile4x4(ushort* w0, ushort* w1, ushort* w2, ushort* w3, float* x, int xStride, int n,
            float* o, int oStride, int cols)
        {
            float* x0 = x, x1 = x + xStride, x2 = x + 2L * xStride, x3 = x + 3L * xStride;
            Vector128<float> a00 = default, a01 = default, a02 = default, a03 = default;
            Vector128<float> a10 = default, a11 = default, a12 = default, a13 = default;
            Vector128<float> a20 = default, a21 = default, a22 = default, a23 = default;
            Vector128<float> a30 = default, a31 = default, a32 = default, a33 = default;
            int i = 0;
            for (; i <= n - 4; i += 4)
            {
                Vector128<float> v0 = LoadBf16x4(w0 + i), v1 = LoadBf16x4(w1 + i), v2 = LoadBf16x4(w2 + i), v3 = LoadBf16x4(w3 + i);
                Vector128<float> r = AdvSimd.LoadVector128(x0 + i);
                a00 += r * v0; a01 += r * v1; a02 += r * v2; a03 += r * v3;
                r = AdvSimd.LoadVector128(x1 + i);
                a10 += r * v0; a11 += r * v1; a12 += r * v2; a13 += r * v3;
                r = AdvSimd.LoadVector128(x2 + i);
                a20 += r * v0; a21 += r * v1; a22 += r * v2; a23 += r * v3;
                r = AdvSimd.LoadVector128(x3 + i);
                a30 += r * v0; a31 += r * v1; a32 += r * v2; a33 += r * v3;
            }
            Bf16FinishRow(a00, a01, a02, a03, x0, w0, w1, w2, w3, i, n, o, cols);
            Bf16FinishRow(a10, a11, a12, a13, x1, w0, w1, w2, w3, i, n, o + oStride, cols);
            Bf16FinishRow(a20, a21, a22, a23, x2, w0, w1, w2, w3, i, n, o + 2L * oStride, cols);
            Bf16FinishRow(a30, a31, a32, a33, x3, w0, w1, w2, w3, i, n, o + 3L * oStride, cols);
        }

        [MethodImpl(MethodImplOptions.AggressiveInlining)]
        private static unsafe void Bf16FinishRow(Vector128<float> a0, Vector128<float> a1, Vector128<float> a2, Vector128<float> a3,
            float* x, ushort* w0, ushort* w1, ushort* w2, ushort* w3, int i, int n, float* o, int cols)
        {
            float s0 = Vector.Sum(a0.AsVector()), s1 = Vector.Sum(a1.AsVector());
            float s2 = Vector.Sum(a2.AsVector()), s3 = Vector.Sum(a3.AsVector());
            for (; i < n; i++)
            {
                float xi = x[i];
                s0 += xi * Bf16ToSingle(w0[i]);
                s1 += xi * Bf16ToSingle(w1[i]);
                s2 += xi * Bf16ToSingle(w2[i]);
                s3 += xi * Bf16ToSingle(w3[i]);
            }
            o[0] = s0;
            if (cols > 1) o[1] = s1;
            if (cols > 2) o[2] = s2;
            if (cols > 3) o[3] = s3;
        }

        /// <summary>Dot's arithmetic for one activation row against four BF16 weight rows: two accumulators over
        /// alternating 4-value chunks, their sum, a last single chunk, Vector.Sum, then the scalar tail.</summary>
        [MethodImpl(MethodImplOptions.AggressiveOptimization)]
        private static unsafe void Bf16Row1x4(ushort* w0, ushort* w1, ushort* w2, ushort* w3, float* x, int n, float* o, int cols)
        {
            Vector128<float> e0 = default, e1 = default, e2 = default, e3 = default;
            Vector128<float> f0 = default, f1 = default, f2 = default, f3 = default;
            int i = 0;
            for (; i <= n - 8; i += 8)
            {
                Vector128<float> xa = AdvSimd.LoadVector128(x + i), xb = AdvSimd.LoadVector128(x + i + 4);
                e0 += xa * LoadBf16x4(w0 + i); f0 += xb * LoadBf16x4(w0 + i + 4);
                e1 += xa * LoadBf16x4(w1 + i); f1 += xb * LoadBf16x4(w1 + i + 4);
                e2 += xa * LoadBf16x4(w2 + i); f2 += xb * LoadBf16x4(w2 + i + 4);
                e3 += xa * LoadBf16x4(w3 + i); f3 += xb * LoadBf16x4(w3 + i + 4);
            }
            Vector128<float> a0 = e0 + f0, a1 = e1 + f1, a2 = e2 + f2, a3 = e3 + f3;
            for (; i <= n - 4; i += 4)
            {
                Vector128<float> xa = AdvSimd.LoadVector128(x + i);
                a0 += xa * LoadBf16x4(w0 + i);
                a1 += xa * LoadBf16x4(w1 + i);
                a2 += xa * LoadBf16x4(w2 + i);
                a3 += xa * LoadBf16x4(w3 + i);
            }
            Bf16FinishRow(a0, a1, a2, a3, x, w0, w1, w2, w3, i, n, o, cols);
        }
    }
}
