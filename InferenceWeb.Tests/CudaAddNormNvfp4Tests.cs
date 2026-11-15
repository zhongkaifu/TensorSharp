// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Graph;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A residual add and the RMSNorm after it whose rows an NVFP4 product quantizes. The fused kernel (the add, the norm and
/// the product's NVFP4 quantization in one pass) gives the sum, the normed rows and the planes the graph's add-and-norm
/// kernel and the quantizer give, to the bit. The CUDA graph executor leaves the add and norm of more rows than the matvec
/// takes for the product that reads them: a program's products, sum and normed rows equal the unfused kernels', to the bit.
/// </summary>
public sealed class CudaAddNormNvfp4Tests
{
    private const int Nvfp4 = 40;
    private const float Eps = 1e-6f;
    private readonly ITestOutputHelper _output;

    public CudaAddNormNvfp4Tests(ITestOutputHelper output) => _output = output;

    /// <summary>The fused kernel over register-held rows (5120 = 512 x 10) and the generic walk (2304, 17408).</summary>
    [CudaTheory]
    [InlineData(5120, 70)]
    [InlineData(2304, 3)]
    [InlineData(17408, 9)]
    public void FusedAddNormQuantizeMatchesTheAddNormKernelAndTheQuantizer(int k, int rows)
    {
        using var allocator = new CudaAllocator();
        if (allocator.Fp4Kernels is not { } fp4)
        {
            _output.WriteLine("no FP4 tensor-core products here: the fused quantizer is not used");
            return;
        }
        allocator.Context.MakeCurrent();
        Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
        using (kernels)
        {
            var rng = new Random(k + rows);
            using Tensor a = Rows(allocator, rng, rows, k, 3f), b = Rows(allocator, rng, rows, k, 1f), w = Rows(allocator, rng, 1, k, 1f);
            using Tensor sumRef = Floats(allocator, rows, k), normedRef = Floats(allocator, rows, k);
            using Tensor sum = Floats(allocator, rows, k), normed = Floats(allocator, rows, k);
            var (codeBytes, scaleBytes, _) = CudaFp4Kernels.ActivationBytes(rows, k);
            using Tensor codesRef = Bytes(allocator, codeBytes), scalesRef = Bytes(allocator, scaleBytes), rowScaleRef = Floats(allocator, rows, 1);
            using Tensor codes = Bytes(allocator, codeBytes), scales = Bytes(allocator, scaleBytes), rowScale = Floats(allocator, rows, 1);
            IntPtr stream = allocator.Stream.Handle;
            kernels.LaunchAddRmsNorm(Ptr(a), Ptr(b), Ptr(w), Ptr(sumRef), Ptr(normedRef), rows, k, Eps, stream);
            fp4.Quantize(Ptr(normedRef), Ptr(codesRef), Ptr(scalesRef), Ptr(rowScaleRef), k, rows, stream);
            fp4.AddRmsNormQuantize(Ptr(a), Ptr(b), Ptr(w), Eps, Ptr(sum), IntPtr.Zero, Ptr(codes), Ptr(scales), Ptr(rowScale), k, rows, stream);
            // The normed rows again from the sum alone (a deferred norm whose product already wrote the sum).
            fp4.AddRmsNormQuantize(Ptr(sumRef), IntPtr.Zero, Ptr(w), Eps, IntPtr.Zero, Ptr(normed), IntPtr.Zero, IntPtr.Zero, IntPtr.Zero,
                k, rows, stream);
            foreach (Tensor t in new[] { sumRef, normedRef, sum, normed, codesRef, scalesRef, rowScaleRef, codes, scales, rowScale })
                ((CudaStorage)t.Storage).MarkDeviceModified();
            allocator.Stream.Synchronize();
            AssertSameBits("sum", sumRef, sum);
            AssertSameBits("normed rows from the sum", normedRef, normed);
            AssertSameBits("row scales", rowScaleRef, rowScale);
            AssertSameBytes("codes", codesRef, codes, codeBytes);
            AssertSameBytes("scales", scalesRef, scales, scaleBytes);
        }
    }

    /// <summary>
    /// A program's add, weighted norm, an NVFP4 product of the normed rows and a SwiGLU feed-forward region over them: the
    /// executor's products, sum and normed rows (an output too, so it is finished as it is read) equal the add-and-norm
    /// kernel's rows and the products run on them, for TMA rows (the gate|up epilogue), the cp.async GEMM's rows and
    /// matvec rows (nothing deferred).
    /// </summary>
    [CudaTheory]
    [InlineData(70)]
    [InlineData(20)]
    [InlineData(4)]
    public void ExecutorProductsOfAnAddedNormMatchTheUnfusedKernels(int rows)
    {
        const int k = 512, outDim = 384, ff = 256, hidden = 512;
        using var allocator = new CudaAllocator();
        if (allocator.Fp4Kernels == null)
        {
            _output.WriteLine("no FP4 tensor-core products here: nothing is deferred");
            return;
        }
        var rng = new Random(rows);
        float[] aHost = Values(rng, rows * k, 3f), bHost = Values(rng, rows * k, 1f);
        var weights = new[] { CudaNvfp4MatmulTests.Nvfp4Rows(outDim, k, 21), CudaNvfp4MatmulTests.Nvfp4Rows(2 * ff, k, 22),
            CudaNvfp4MatmulTests.Nvfp4Rows(hidden, ff, 23) };
        var hosts = new IntPtr[weights.Length];
        var handles = new GCHandle[2];
        try
        {
            for (int i = 0; i < weights.Length; i++)
            {
                hosts[i] = Marshal.AllocHGlobal(weights[i].Length);
                Marshal.Copy(weights[i], 0, hosts[i], weights[i].Length);
            }
            var product = new Weight(hosts[0], k, outDim, weights[0].Length);
            var gateUp = new Weight(hosts[1], k, 2 * ff, weights[1].Length);
            var down = new Weight(hosts[2], ff, hidden, weights[2].Length);
            using Tensor norm = Rows(allocator, rng, 1, k, 1f);

            var g = new GraphProgramBuilder(record: true);
            int a = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { k, rows }, IntPtr.Zero, (long)rows * k * 4);
            int b = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { k, rows }, IntPtr.Zero, (long)rows * k * 4);
            using Tensor normVector = norm.View(k);
            int wNorm = g.Leaf(GraphLeafRole.Weight, GraphFormat.F32, new long[] { k }, IntPtr.Zero, k * 4, handle: normVector);
            int wProduct = g.Leaf(GraphLeafRole.Weight, Nvfp4, new long[] { k, outDim }, hosts[0], weights[0].Length, handle: product);
            int wGateUp = g.Leaf(GraphLeafRole.Weight, Nvfp4, new long[] { k, 2 * ff }, hosts[1], weights[1].Length, handle: gateUp);
            int wDown = g.Leaf(GraphLeafRole.Weight, Nvfp4, new long[] { ff, hidden }, hosts[2], weights[2].Length, handle: down);
            int sumNode = g.Op(GraphOp.Add, a, b);
            int normNode = g.Op(GraphOp.RmsNorm, stackalloc[] { sumNode }, default, stackalloc[] { Eps });
            int x = g.Op(GraphOp.Mul, normNode, wNorm);
            int p = g.Op(GraphOp.MulMat, wProduct, x);
            g.BeginFused(GraphFusion.SwigluFfn, stackalloc[] { x, wGateUp, wDown });
            int y = g.Op(GraphOp.MulMat, wDown, g.Op(GraphOp.Swiglu, g.Op(GraphOp.MulMat, wGateUp, x)));
            g.EndFused(stackalloc[] { y });
            g.Output(p, (long)rows * outDim);
            g.Output(y, (long)rows * hidden);
            g.Output(sumNode, (long)rows * k);
            g.Output(x, (long)rows * k);

            float[][] got;
            long normedProducts;
            using (var executor = new CudaGraphExecutor(allocator))
            using (IGraphPlan plan = executor.Compile(g.ToProgram(replayable: true, reorder: false)))
            {
                handles[0] = GCHandle.Alloc(aHost, GCHandleType.Pinned);
                handles[1] = GCHandle.Alloc(bHost, GCHandleType.Pinned);
                long bytes = (long)rows * k * 4;
                Assert.True(plan.Run(new[] { a, b }, new[] { handles[0].AddrOfPinnedObject(), handles[1].AddrOfPinnedObject() },
                    new[] { bytes, bytes }), plan.LastError);
                got = Array.ConvertAll(plan.Outputs, o => (float[])o.Clone());
                normedProducts = executor.NormedProducts;
            }
            // Past the matvec's rows both products quantize the normed rows in the add and norm's pass.
            Assert.Equal(rows > CudaFp4Kernels.GemvMaxRows ? 2 : 0, normedProducts);

            // The unfused kernels: the add and norm, then each product over the normed rows (the feed-forward as its
            // region ran before: the [rows, 2 ff] product and the down projection that quantizes its SwiGLU).
            allocator.Context.MakeCurrent();
            Assert.True(CudaGraphKernels.TryCreate(out CudaGraphKernels kernels, out string reason), reason);
            using (kernels)
            {
                using Tensor xa = Upload(allocator, aHost, rows, k), xb = Upload(allocator, bHost, rows, k);
                using Tensor sum = Floats(allocator, rows, k), normed = Floats(allocator, rows, k);
                kernels.LaunchAddRmsNorm(Ptr(xa), Ptr(xb), Ptr(norm), Ptr(sum), Ptr(normed), rows, k, Eps, allocator.Stream.Handle);
                ((CudaStorage)sum.Storage).MarkDeviceModified();
                ((CudaStorage)normed.Storage).MarkDeviceModified();
                using Tensor pRef = Floats(allocator, rows, outDim), gu = Floats(allocator, rows, 2 * ff), yRef = Floats(allocator, rows, hidden);
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(pRef, normed, hosts[0], hosts[0], Nvfp4, k, outDim, weights[0].Length));
                Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(gu, normed, hosts[1], hosts[1], Nvfp4, k, 2 * ff, weights[1].Length));
                if (!CudaQuantizedOps.TryAddmmSwiGluNvfp4(yRef, gu, hosts[2], hosts[2], Nvfp4, ff, hidden, weights[2].Length))
                {
                    using Tensor act = Floats(allocator, rows, ff);
                    allocator.Kernels.LaunchSiLUMulSplitF32(Ptr(gu), Ptr(act), rows, ff, allocator.Stream.Handle);
                    ((CudaStorage)act.Storage).MarkDeviceModified();
                    Assert.True(CudaQuantizedOps.TryAddmmQuantizedToFloat32(yRef, act, hosts[2], hosts[2], Nvfp4, ff, hidden, weights[2].Length));
                }
                AssertSameBits($"{rows} rows: the product", pRef.GetElementsAsFloat(rows * outDim), got[0]);
                AssertSameBits($"{rows} rows: the feed-forward", yRef.GetElementsAsFloat(rows * hidden), got[1]);
                AssertSameBits($"{rows} rows: the sum", sum.GetElementsAsFloat(rows * k), got[2]);
                AssertSameBits($"{rows} rows: the normed rows", normed.GetElementsAsFloat(rows * k), got[3]);
            }
        }
        finally
        {
            foreach (GCHandle h in handles)
                if (h.IsAllocated) h.Free();
            foreach (IntPtr host in hosts)
            {
                if (host == IntPtr.Zero) continue;
                CudaQuantizedOps.ReleaseQuantizedWeight(allocator, host);
                Marshal.FreeHGlobal(host);
            }
        }
    }

    private sealed record Weight(IntPtr Data, long Ne0, long Ne1, long Bytes) : IGraphWeight
    {
        public int Format => Nvfp4;
        public long Ne2 => 1;
        public IntPtr CacheKey => Data;
    }

    private static float[] Values(Random rng, int count, float scale)
    {
        var v = new float[count];
        for (int i = 0; i < count; i++) v[i] = scale * (float)(rng.NextDouble() * 2 - 1);
        return v;
    }

    private static Tensor Rows(CudaAllocator allocator, Random rng, int rows, int cols, float scale)
        => Upload(allocator, Values(rng, rows * cols, scale), rows, cols);

    private static Tensor Upload(CudaAllocator allocator, float[] values, int rows, int cols)
    {
        var t = new Tensor(allocator, DType.Float32, rows, cols);
        t.SetElementsAsFloat(values);
        ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        return t;
    }

    private static Tensor Floats(CudaAllocator allocator, int rows, int cols)
    {
        var t = new Tensor(allocator, DType.Float32, rows, cols);
        ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        return t;
    }

    private static Tensor Bytes(CudaAllocator allocator, long count)
    {
        var t = new Tensor(allocator, DType.UInt8, count);
        ((CudaStorage)t.Storage).EnsureDeviceCurrent();
        return t;
    }

    private static IntPtr Ptr(Tensor t) => ((CudaStorage)t.Storage).DevicePtrAtElement(t.StorageOffset);

    private static void AssertSameBits(string what, Tensor want, Tensor got)
        => AssertSameBits(what, want.GetElementsAsFloat((int)want.ElementCount()), got.GetElementsAsFloat((int)got.ElementCount()));

    private static void AssertSameBits(string what, float[] want, float[] got)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
                Assert.Fail($"{what}: element {i} is {got[i]:R}, the unfused kernels give {want[i]:R}");
    }

    private static void AssertSameBytes(string what, Tensor want, Tensor got, long count)
    {
        var w = new byte[count];
        var g = new byte[count];
        want.CopyToArray(w);
        got.CopyToArray(g);
        for (long i = 0; i < count; i++)
            if (w[i] != g[i])
                Assert.Fail($"{what}: byte {i} is {g[i]}, the quantizer writes {w[i]}");
    }
}
