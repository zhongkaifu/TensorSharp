// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A Q6_K vocabulary head regrouped in place for the tiled int8 tensor-core product (CudaQuantizedOps.TryTileQ6kHead): its
/// products agree with the stored rows' product (both over q8_1 activations, which round apart), and a row's logits are
/// the same bits alone as in a window of any width, so a verify window's rows are the decode rows'.
/// </summary>
public sealed class CudaTiledQ6kHeadTests
{
    private const int Q6KType = 14, BlockBytes = 210;

    [CudaTheory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(8)]
    [InlineData(13)]
    [InlineData(70)]
    public void TiledHeadMatchesTheStoredRowsAndEachRowAlone(int rows)
    {
        using var allocator = new CudaAllocator();
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            Assert.True(kernels.TakesTiledQ6kHeads, $"this image has no sm_80 bodies (arch {kernels.ImageArch}): the tiled head is not used here");
            const int k = 768, outDim = 64 * 5;
            var rng = new Random(41 + rows);
            byte[] blocks = Q6KBlocks(rng, outDim, k);
            float[] x = Enumerable.Range(0, rows * k).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            // Two copies of the bytes: one stays on its stored rows, the other is tiled.
            GCHandle stored = GCHandle.Alloc((byte[])blocks.Clone(), GCHandleType.Pinned), tiled = GCHandle.Alloc((byte[])blocks.Clone(), GCHandleType.Pinned);
            try
            {
                IntPtr storedKey = stored.AddrOfPinnedObject(), tiledKey = tiled.AddrOfPinnedObject();
                Assert.True(CudaQuantizedOps.TryEnsureResident(allocator, tiledKey, tiledKey, Q6KType, k, outDim, blocks.Length, out _, out _));
                Assert.True(CudaQuantizedOps.TryTileQ6kHead(allocator, tiledKey, kernels), "the head was not tiled");
                Assert.False(CudaQuantizedOps.TryGetResidentDevicePtr(allocator, tiledKey, out _, out _), "a tiled head was handed out as Q6_K blocks");
                float[] reference = Product(allocator, storedKey, x, rows, k, outDim, blocks.Length);
                float[] product = Product(allocator, tiledKey, x, rows, k, outDim, blocks.Length);
                double diff = 0, norm = 0;
                for (int i = 0; i < product.Length; i++)
                {
                    diff += (double)(product[i] - reference[i]) * (product[i] - reference[i]);
                    norm += (double)reference[i] * reference[i];
                }
                Assert.True(norm > 0, "the stored rows' product read back as zeros");
                Assert.True(Math.Sqrt(diff / norm) < 1e-2, $"{rows} rows: relative L2 {Math.Sqrt(diff / norm):E2} against the stored rows' product");
                // Each row alone: the same bits as in the window.
                for (int r = 0; r < rows; r++)
                {
                    float[] alone = Product(allocator, tiledKey, x.AsSpan(r * k, k).ToArray(), 1, k, outDim, blocks.Length);
                    for (int o = 0; o < outDim; o++)
                        Assert.True(BitConverter.SingleToInt32Bits(alone[o]) == BitConverter.SingleToInt32Bits(product[r * outDim + o]),
                            $"row {r} of {rows}, output {o}: {alone[o]} alone against {product[r * outDim + o]} in the window");
                }
            }
            finally
            {
                CudaQuantizedOps.ReleaseQuantizedWeight(allocator, stored.AddrOfPinnedObject());
                CudaQuantizedOps.ReleaseQuantizedWeight(allocator, tiled.AddrOfPinnedObject());
                stored.Free();
                tiled.Free();
            }
        }
    }

    private static float[] Product(CudaAllocator allocator, IntPtr key, float[] x, int rows, int k, int outDim, long bytes)
    {
        using var input = new Tensor(allocator, DType.Float32, rows, k);
        input.SetElementsAsFloat(x);
        using var result = new Tensor(allocator, DType.Float32, rows, outDim);
        Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(result, input, key, key, Q6KType, k, outDim, bytes), "the product did not run");
        allocator.Stream.Synchronize();
        return result.GetElementsAsFloat(rows * outDim);
    }

    /// <summary>Random Q6_K rows: random quants and sub-block scales, a small positive super-block scale.</summary>
    private static byte[] Q6KBlocks(Random rng, int rows, int k)
    {
        int perRow = k / 256;
        var bytes = new byte[(long)rows * perRow * BlockBytes];
        rng.NextBytes(bytes);
        for (int b = 0; b < rows * perRow; b++)
        {
            int at = b * BlockBytes;
            for (int s = 0; s < 16; s++) bytes[at + 192 + s] = (byte)(sbyte)rng.Next(-32, 33);
            ushort d = BitConverter.HalfToUInt16Bits((System.Half)(0.002 + rng.NextDouble() * 0.004));
            bytes[at + 208] = (byte)d;
            bytes[at + 209] = (byte)(d >> 8);
        }
        return bytes;
    }
}
