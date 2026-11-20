// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Runtime.InteropServices;

using TensorSharp;
using TensorSharp.MLX;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

/// <summary>
/// MLX products over unquantized F32, F16 and BF16 weights kept as their GGUF bytes (MlxQuantizedOps.TryAddmmFloatWeight),
/// which MlxGraphExecutor used to run as a host CPU GEMM (an F16 tower took 2.9 s where F32 copies took 30 ms): the
/// matrix-vector kernel (decode-sized row counts) and the GEMM path against a double-precision reference over the weight's
/// exact values, with output rows not a multiple of the kernel's eight and an input width the 4-wide kernel does not take.
/// </summary>
public sealed class MlxFloatWeightProductTests
{
    [MlxTheory]
    [InlineData(GgmlTensorType.F32, 256, 13)]
    [InlineData(GgmlTensorType.F16, 256, 13)]
    [InlineData(GgmlTensorType.BF16, 256, 13)]
    [InlineData(GgmlTensorType.F16, 2048, 40)]
    [InlineData(GgmlTensorType.BF16, 2048, 40)]
    [InlineData(GgmlTensorType.F16, 66, 9)]
    [InlineData(GgmlTensorType.BF16, 66, 9)]
    public unsafe void FloatWeightProductsMatchTheReference(GgmlTensorType type, int inDim, int outDim)
    {
        var random = new Random((int)type * 31 + inDim);
        int elementBytes = type == GgmlTensorType.F32 ? 4 : 2;
        var exact = new double[outDim * inDim];
        IntPtr raw = (IntPtr)NativeMemory.AlignedAlloc((nuint)(outDim * inDim * elementBytes), 64);
        try
        {
            for (int i = 0; i < exact.Length; i++)
            {
                float v = (float)(random.NextDouble() * 2 - 1);
                switch (type)
                {
                    case GgmlTensorType.F32:
                        ((float*)raw)[i] = v;
                        exact[i] = v;
                        break;
                    case GgmlTensorType.F16:
                        var h = (System.Half)v;
                        ((System.Half*)raw)[i] = h;
                        exact[i] = (double)(float)h;
                        break;
                    default:
                        ushort bits = (ushort)(BitConverter.SingleToUInt32Bits(v) >> 16);
                        ((ushort*)raw)[i] = bits;
                        exact[i] = BitConverter.UInt32BitsToSingle((uint)bits << 16);
                        break;
                }
            }
            using var allocator = new MlxAllocator();
            try
            {
                foreach (int rows in new[] { 1, 3, 8, 9, 40 })
                {
                    var x = new float[rows, inDim];
                    for (int r = 0; r < rows; r++)
                        for (int c = 0; c < inDim; c++) x[r, c] = (float)(random.NextDouble() * 4 - 2);
                    using var input = Tensor.FromArray(allocator, x);
                    using var result = new Tensor(allocator, DType.Float32, rows, outDim);
                    Assert.True(MlxQuantizedOps.TryAddmmFloatWeight(result, input, raw, raw, (int)type, inDim, outDim,
                        (long)outDim * inDim * elementBytes), $"{type} {rows} rows: declined");
                    float[] actual = result.GetElementsAsFloat(rows * outDim);
                    // The matrix-vector kernel sums in F32 (rows up to 8, a width of whole float4s); MLX's GEMM takes F16 rows
                    // in half and the F32 and BF16 weights in F32 or TF32 (MLX_ENABLE_TF32, the process's choice).
                    bool matvec = rows <= 8 && inDim % 4 == 0;
                    double tolerance = matvec ? 1e-5 : type == GgmlTensorType.F16 ? 3e-3 : 4e-3;
                    double num = 0, den = 0;
                    for (int r = 0; r < rows; r++)
                        for (int o = 0; o < outDim; o++)
                        {
                            double expected = 0;
                            for (int c = 0; c < inDim; c++) expected += exact[o * inDim + c] * x[r, c];
                            double d = actual[r * outDim + o] - expected;
                            num += d * d;
                            den += expected * expected;
                        }
                    double rel = Math.Sqrt(num / Math.Max(den, 1e-30));
                    Assert.True(rel <= tolerance, $"{type} [{outDim} x {inDim}] x {rows} rows ({(matvec ? "matvec" : "gemm")}): relL2 {rel:E2} > {tolerance:E0}");
                }
            }
            finally
            {
                MlxQuantizedOps.ReleaseQuantizedWeight(allocator, raw);
            }
        }
        finally
        {
            NativeMemory.AlignedFree((void*)raw);
        }
    }
}
