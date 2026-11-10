// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.Dsv4ExpertKernelTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The direct engine's routing of Q8_0 decode projections between the pair matvec (CudaKernels.LaunchQuantMatmulQ80Pair)
/// and the decode-once dense kernels (Dsv4Kernels.DenseQ81Rows): a weight that keeps the decode-once kernels
/// (DeviceWeight.DecodeOnce) never takes the pair, at any row count. And, timing only, Flash Next's Q8_0 decode
/// projections (the TS_Q4E_HC_Q8 hyper-connection mixers, the MTP block's dense weights, and the trunk shapes a Q8_0 pack
/// would have) on both kernels at 1-16 rows, graph-captured, with cold weights (enough copies of each matrix to stream
/// past the L2).
/// </summary>
public sealed class Q80PairVsDenseFastBenchTests
{
    private const int Q8_0 = (int)GgmlTensorType.Q8_0;
    private readonly ITestOutputHelper _output;

    public Q80PairVsDenseFastBenchTests(ITestOutputHelper output) => _output = output;

    private static IntPtr Ptr(Tensor t) => Dsv4CudaEngine.Ptr(t);

    [Fact]
    public void ADecodeOnceWeight_NeverTakesThePair()
    {
        var trunk = new DeviceWeight { Type = Q8_0, Ne0 = 2560, Ne1 = 10240, RowBytes = 2560 / 32 * 34 };
        var once = trunk;
        once.DecodeOnce = true;
        bool pairOn = Environment.GetEnvironmentVariable("TS_Q4E_Q80_PAIR") != "0";
        for (int rows = 1; rows <= CudaKernels.Q80PairMaxRows; rows++)
        {
            Assert.False(Q4eCudaEngine.TakesQ80Pair(once, rows, hasPair: true), $"a decode-once weight took the pair at {rows} rows");
            Assert.Equal(pairOn, Q4eCudaEngine.TakesQ80Pair(trunk, rows, hasPair: true));
            Assert.False(Q4eCudaEngine.TakesQ80Pair(trunk, rows, hasPair: false));
        }
        Assert.False(Q4eCudaEngine.TakesQ80Pair(trunk, CudaKernels.Q80PairMaxRows + 1, hasPair: true));
        var odd = trunk;
        odd.Ne0 = 2560 + 16;
        Assert.False(Q4eCudaEngine.TakesQ80Pair(odd, 4, hasPair: true));
        var q6 = trunk;
        q6.Type = (int)GgmlTensorType.Q6_K;
        Assert.False(Q4eCudaEngine.TakesQ80Pair(q6, 4, hasPair: true));
    }

    private static double GraphMicros(IntPtr stream, Action<int> launch, int n, int reps)
    {
        CudaDriverApi.cuEventCreate(out IntPtr start, 0).ThrowOnError();
        CudaDriverApi.cuEventCreate(out IntPtr end, 0).ThrowOnError();
        try
        {
            CudaDriverApi.cuStreamBeginCapture(stream, CudaDriverApi.CU_STREAM_CAPTURE_MODE_RELAXED).ThrowOnError();
            for (int i = 0; i < n; i++)
                launch(i);
            CudaDriverApi.cuStreamEndCapture(stream, out IntPtr graph).ThrowOnError();
            CudaDriverApi.cuGraphInstantiateWithFlags(out IntPtr exec, graph, 0).ThrowOnError();
            try
            {
                for (int r = 0; r < 3; r++)
                    CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                CudaDriverApi.cuStreamSynchronize(stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(start, stream).ThrowOnError();
                for (int r = 0; r < reps; r++)
                    CudaDriverApi.cuGraphLaunch(exec, stream).ThrowOnError();
                CudaDriverApi.cuEventRecord(end, stream).ThrowOnError();
                CudaDriverApi.cuEventSynchronize(end).ThrowOnError();
                CudaDriverApi.cuEventElapsedTime(out float ms, start, end).ThrowOnError();
                return ms * 1000.0 / ((double)n * reps);
            }
            finally
            {
                CudaDriverApi.cuGraphExecDestroy(exec);
                CudaDriverApi.cuGraphDestroy(graph);
            }
        }
        finally
        {
            CudaDriverApi.cuEventDestroy(start);
            CudaDriverApi.cuEventDestroy(end);
        }
    }

    [CudaFact(TensorCoreModule = "tensorsharp_kernels")]
    [Trait("Category", "Bench")]
    public void FlashNextQ8_0Projections_PairAgainstDenseFast()
    {
        var shapes = new (string Name, int In, int Out)[]
        {
            ("hc down", 10240, 320),
            ("hc up", 320, 10240),
            ("mtp eh_proj", 5120, 2560),
            ("q / qkv", 2560, 10240),
            ("gdn gate", 2560, 6144),
            ("attn q", 2560, 12288),
            ("ssm_out / attn o", 6144, 2560),
            ("attn k / v", 2560, 512),
            ("shexp gate / up", 2560, 640),
            ("shexp down", 640, 2560),
        };
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.True(allocator.Kernels.HasQ80Pair, "no pair matvec in the loaded module");
        IntPtr stream = allocator.Stream.Handle;
        var rnd = new Random(5);
        foreach (var (name, inDim, outDim) in shapes)
        {
            Assert.True(dk.DenseFastReads(Q8_0, inDim), $"{name}: the decode-once kernels do not read Q8_0 at width {inDim}");
            long rowBytes = inDim / 32 * 34L, matBytes = rowBytes * outDim;
            int copies = (int)Math.Max(4, Math.Min(64, (768L << 20) / matBytes));
            byte[] w = QuantRows(Q8_0, outDim, rowBytes, seed: inDim + outDim);
            var mats = new List<Tensor>();
            try
            {
                for (int c = 0; c < copies; c++)
                    mats.Add(Up(allocator, w));
                var x = new float[16 * inDim];
                for (int i = 0; i < x.Length; i++)
                    x[i] = (float)(rnd.NextDouble() * 2 - 1);
                using Tensor xT = Up(allocator, x);
                using Tensor xq = new Tensor(allocator, DType.UInt8, 16L * (inDim / 32) * CudaKernels.Q81BlockBytes);
                using Tensor y = new Tensor(allocator, DType.Float32, 16, outDim);
                allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(xT), Ptr(xq), inDim, 16, stream, warpCooperative: true);
                string line = $"{name} {inDim} -> {outDim} ({matBytes / 1048576.0:F2} MiB):";
                for (int rows = 1; rows <= 16; rows++)
                {
                    int n = Math.Max(copies, 64);
                    double pair = GraphMicros(stream, i => allocator.Kernels.LaunchQuantMatmulQ80Pair(Ptr(mats[i % copies]), Ptr(xq),
                        Ptr(y), inDim, outDim, rows, stream), n, 10);
                    double fast = GraphMicros(stream, i => dk.DenseQ81Rows(Ptr(mats[i % copies]), Ptr(xq), Ptr(y), Q8_0, outDim, inDim,
                        rowBytes, rows, stream), n, 10);
                    line += $" {rows}r pair {pair:F2} / fast {fast:F2} us;";
                }
                _output.WriteLine(line);
            }
            finally
            {
                allocator.Synchronize();
                foreach (Tensor t in mats)
                    t.Dispose();
            }
        }
    }

    /// <summary>
    /// Timing only: each of the pair's kernels (the dp4a twin at 1-4 rows, the dp4a lane kernel at 1-8, the tensor
    /// cores at 1-16; all give the same bits) at Flash Next's Q8_0 decode shapes, graph-captured with cold weights,
    /// beside the decode-once dense kernels (other arithmetic) where they read the shape, and the kernel
    /// <see cref="CudaKernels.Q80PairKernelFor"/>'s heuristic names. The ISTA checkpoint's TS_Q4E_HC_Q8 up mixers and
    /// its two Q8_0 shared-expert downs take the pair; its down mixers and the MTP block's weights keep the dense kernels.
    /// </summary>
    [CudaFact(TensorCoreModule = "tensorsharp_kernels")]
    [Trait("Category", "Bench")]
    public void FlashNextQ8_0Shapes_EachPairKernel()
    {
        var shapes = new (string Name, int In, int Out)[]
        {
            ("hc up", 320, 10240),
            ("shexp down", 640, 2560),
            ("hc down", 10240, 320),
            ("q / qkv", 2560, 10240),
            ("gdn gate", 2560, 6144),
            ("attn q", 2560, 12288),
            ("ssm_out / attn o", 6144, 2560),
            ("attn k / v", 2560, 512),
            ("shexp gate / up", 2560, 640),
            ("lm_head", 2560, 248320),
        };
        using var allocator = new CudaAllocator(0);
        allocator.Context.MakeCurrent();
        using var dk = Dsv4Kernels.Create();
        Assert.True(allocator.Kernels.HasQ80Pair, "no pair matvec in the loaded module");
        IntPtr stream = allocator.Stream.Handle;
        var rnd = new Random(7);
        var device = CudaDevice.GetDevice(allocator.DeviceId);
        _output.WriteLine($"{device.Name}, sm_{device.ComputeCapabilityMajor}{device.ComputeCapabilityMinor}");
        foreach (var (name, inDim, outDim) in shapes)
        {
            long rowBytes = inDim / 32 * 34L, matBytes = rowBytes * outDim;
            int copies = (int)Math.Max(2, Math.Min(64, (768L << 20) / matBytes));
            byte[] w = QuantRows(Q8_0, outDim, rowBytes, seed: inDim + outDim);
            var mats = new List<Tensor>();
            try
            {
                for (int c = 0; c < copies; c++)
                    mats.Add(Up(allocator, w));
                var x = new float[16 * inDim];
                for (int i = 0; i < x.Length; i++)
                    x[i] = (float)(rnd.NextDouble() * 2 - 1);
                using Tensor xT = Up(allocator, x);
                using Tensor xq = new Tensor(allocator, DType.UInt8, 16L * (inDim / 32) * CudaKernels.Q81BlockBytes);
                using Tensor y = new Tensor(allocator, DType.Float32, 16, outDim);
                allocator.Kernels.LaunchQuantizeQ81Rows(Ptr(xT), Ptr(xq), inDim, 16, stream, warpCooperative: true);
                _output.WriteLine($"{name} {inDim} -> {outDim} ({matBytes / 1048576.0:F2} MiB, " +
                    $"{CudaKernels.Q80PairSlices(inDim, outDim)} K slices), us per call:");
                for (int rows = 1; rows <= 16; rows++)
                {
                    int n = Math.Max(copies, 64);
                    string line = $"  {rows,2}r";
                    double best = double.MaxValue;
                    CudaKernels.Q80PairKernel bestKernel = CudaKernels.Q80PairKernel.Mma;
                    foreach (var kernel in new[] { CudaKernels.Q80PairKernel.Twin, CudaKernels.Q80PairKernel.Lane, CudaKernels.Q80PairKernel.Mma })
                    {
                        int maxRows = kernel == CudaKernels.Q80PairKernel.Twin ? 4 : kernel == CudaKernels.Q80PairKernel.Lane ? 8 : 16;
                        if (rows > maxRows)
                        {
                            line += $" {kernel.ToString().ToLowerInvariant()}     -  ";
                            continue;
                        }
                        double us = GraphMicros(stream, i => allocator.Kernels.LaunchQuantMatmulQ80PairKernel(kernel,
                            Ptr(mats[i % copies]), Ptr(xq), Ptr(y), inDim, outDim, rows, stream), n, 10);
                        line += $" {kernel.ToString().ToLowerInvariant()} {us,7:F2}";
                        if (us < best)
                        {
                            best = us;
                            bestKernel = kernel;
                        }
                    }
                    var picked = CudaKernels.Q80PairKernelFor(inDim, outDim, rows);
                    line += $" | best {bestKernel.ToString().ToLowerInvariant()}, heuristic {picked.ToString().ToLowerInvariant()}";
                    if (dk.DenseFastReads(Q8_0, inDim))
                    {
                        double dense = GraphMicros(stream, i => dk.DenseQ81Rows(Ptr(mats[i % copies]), Ptr(xq), Ptr(y), Q8_0,
                            outDim, inDim, rowBytes, rows, stream), n, 10);
                        line += $" | dense {dense,7:F2}";
                    }
                    _output.WriteLine(line);
                }
            }
            finally
            {
                allocator.Synchronize();
                foreach (Tensor t in mats)
                    t.Dispose();
            }
        }
    }
}
