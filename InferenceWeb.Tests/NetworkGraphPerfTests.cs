// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Graph;
using TensorSharp.Models;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Speed sanity of non-decoder networks as one graph program (NetworkGraph) at an LFM3 vision tower's size: 14 blocks
/// 1024 wide over 1024 rows (a 32 x 32 patch grid), F16 weights. Reports milliseconds (median of timed runs after warm-up)
/// and checks the results agree; the bounds asserted are loose (a shared, noisy machine), the numbers are the point.
/// </summary>
public sealed class NetworkGraphPerfTests
{
    private const int Width = 1024, Ff = 2560, Layers = 14, Grid = 32, Rows = Grid * Grid, Heads = 16, KvHeads = 8, HeadDim = 64;
    private const float Eps = 1e-5f;
    private readonly ITestOutputHelper _output;

    public NetworkGraphPerfTests(ITestOutputHelper output) => _output = output;

    private void Report(string line)
    {
        _output.WriteLine(line);
        Console.WriteLine(line);
    }

    /// <summary>Pre-norm mixer + SwiGLU FFN blocks: one graph program a call against the same ops dispatched one by one
    /// (each GGML op its own graph run, as the per-op encoders run).</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void MetalBlockStackOneGraphAgainstPerOpDispatch()
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Metal), 0);
        try
        {
            using var weights = new NetworkWeights(allocator, BackendType.GgmlMetal);
            var rng = new Random(1);
            for (int l = 0; l < Layers; l++)
            {
                weights.AddF32($"{l}.ln1", NetworkGraphTests.Rand(rng, Width, 0.5f, 1.5f), Width);
                weights.AddF32($"{l}.ln2", NetworkGraphTests.Rand(rng, Width, 0.5f, 1.5f), Width);
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.mix", GraphFormat.F16, Width, Width, rng);
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.gate", GraphFormat.F16, Width, Ff, rng);
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.up", GraphFormat.F16, Width, Ff, rng);
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.down", GraphFormat.F16, Ff, Width, rng);
            }
            float[] input = NetworkGraphTests.Rand(rng, Width * Rows);
            using var graph = new NetworkGraph(weights, "perf stack");
            float[] Graph() => graph.Run(b =>
            {
                NetValue x = b.Input(input, Width, Rows);
                for (int l = 0; l < Layers; l++)
                {
                    x = b.Add(x, b.Linear(b.RmsNorm(x, $"{l}.ln1", Eps), $"{l}.mix"));
                    x = b.Add(x, b.FeedForward(b.RmsNorm(x, $"{l}.ln2", Eps), $"{l}.gate", $"{l}.up", $"{l}.down"));
                }
                b.Output(x);
            })[0];

            float[] PerOp()
            {
                var made = new List<Tensor>();
                Tensor Keep(Tensor t) { made.Add(t); return t; }
                Tensor Product(Tensor x, string name)
                {
                    QuantizedWeight q = weights.Get(name).Quantized;
                    Tensor y = Keep(new Tensor(allocator, DType.Float32, x.Sizes[0], q.Ne1));
                    GgmlBasicOps.AddmmQuant(y, x, q.Data, q.GgmlType, q.Ne0, q.Ne1, q.RawBytes);
                    return y;
                }
                try
                {
                    Tensor x = Keep(new Tensor(allocator, DType.Float32, Rows, Width));
                    x.SetElementsAsFloat(input);
                    for (int l = 0; l < Layers; l++)
                    {
                        Tensor n1 = Keep(Ops.RMSNorm(null, x, weights.Get($"{l}.ln1").Dense, null, Eps));
                        x = Keep(Ops.Add(null, x, Product(n1, $"{l}.mix")));
                        Tensor n2 = Keep(Ops.RMSNorm(null, x, weights.Get($"{l}.ln2").Dense, null, Eps));
                        Tensor h = Keep(Ops.SiLUMul(null, Product(n2, $"{l}.gate"), Product(n2, $"{l}.up")));
                        x = Keep(Ops.Add(null, x, Product(h, $"{l}.down")));
                    }
                    return x.GetElementsAsFloat(Rows * Width);
                }
                finally
                {
                    foreach (Tensor t in made) t.Dispose();
                }
            }

            float[] one = Graph(), perOp = PerOp();
            NetworkGraphTests.AssertClose(perOp, one, 2e-2, "one graph vs per-op");
            double graphMs = Median(() => Graph()), perOpMs = Median(() => PerOp());
            double gflop = 2.0 * Rows * Layers * ((double)Width * Width + 3.0 * Width * Ff) / 1e9;
            Report($"metal block stack {Layers}x{Width} (ffn {Ff}) over {Rows} rows, F16 weights ({gflop:F0} GFLOP): " +
                $"one graph {graphMs:F1} ms ({gflop / graphMs:F1} TFLOP/s), per-op dispatch {perOpMs:F1} ms; {graph.Builds} build(s), {graph.Replays} replays");
            Assert.True(graphMs < 10_000, $"one graph took {graphMs} ms");
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    /// <summary>The LFM3 vision tower's shape (8 gated depthwise-conv blocks, 6 GQA blocks with 2-D rotary, SwiGLU FFNs) as one
    /// graph: the depthwise conv as GraphOp.DepthwiseConv2d against the same conv composed of nine tap views (multiply-add),
    /// and with ggml-metal's concurrency reorder.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void MetalVisionShapedTower()
    {
        var allocator = new GgmlAllocator(new GgmlContext([0], GgmlBackendType.Metal), 0);
        try
        {
            using var weights = VisionWeights(allocator, BackendType.GgmlMetal);
            float[] input = NetworkGraphTests.Rand(new Random(5), Width * Rows);
            using var graph = new NetworkGraph(weights, "vision-shaped tower");
            float[] Run(bool composed) => graph.Run(b => b.Output(VisionTower(b, input, composed)))[0];
            float[] withOp = Run(false), withViews = Run(true);
            NetworkGraphTests.AssertClose(withOp, withViews, 1e-2, "depthwise op vs composed taps");
            double opMs = Median(() => Run(false)), viewsMs = Median(() => Run(true));
            graph.ClearCache();
            graph.Reorder = true;
            float[] reordered = Run(false);
            NetworkGraphTests.AssertClose(withOp, reordered, 1e-2, "reordered");
            double reorderMs = Median(() => Run(false));
            Report($"metal LFM3-vision-shaped tower {Layers}x{Width} over {Grid}x{Grid}: depthwise op {opMs:F1} ms, composed taps {viewsMs:F1} ms, " +
                $"op + reorder {reorderMs:F1} ms ({graph.LastNodeCount} nodes)");
        }
        finally
        {
            GgmlBasicOps.ReleaseReuseComputeBuffers();
            GgmlBasicOps.ClearHostBufferCache();
        }
    }

    [MlxFact]
    public void MlxVisionShapedTower()
    {
        using var allocator = new TensorSharp.MLX.MlxAllocator();
        using var weights = VisionWeights(allocator, BackendType.Mlx);
        float[] input = NetworkGraphTests.Rand(new Random(5), Width * Rows);
        using var graph = new NetworkGraph(weights, "vision-shaped tower");
        float[] Run() => graph.Run(b => b.Output(VisionTower(b, input, false)))[0];
        Run();
        double ms = Median(() => Run());
        Report($"mlx LFM3-vision-shaped tower {Layers}x{Width} over {Grid}x{Grid}: {ms:F1} ms ({graph.LastNodeCount} nodes, compiled every call)");
    }

    private static NetworkWeights VisionWeights(IAllocator allocator, BackendType backend)
    {
        var weights = new NetworkWeights(allocator, backend);
        var rng = new Random(3);
        for (int l = 0; l < Layers; l++)
        {
            weights.AddF32($"{l}.ln1", NetworkGraphTests.Rand(rng, Width, 0.5f, 1.5f), Width);
            weights.AddF32($"{l}.ln2", NetworkGraphTests.Rand(rng, Width, 0.5f, 1.5f), Width);
            NetworkGraphTests.MakeProductWeight(weights, $"{l}.gate", GraphFormat.F16, Width, Ff, rng);
            NetworkGraphTests.MakeProductWeight(weights, $"{l}.up", GraphFormat.F16, Width, Ff, rng);
            NetworkGraphTests.MakeProductWeight(weights, $"{l}.down", GraphFormat.F16, Ff, Width, rng);
            if (IsConv(l))
            {
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.in_proj", GraphFormat.F16, Width, 3 * Width, rng);
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.out_proj", GraphFormat.F16, Width, Width, rng);
                float[] kernel = NetworkGraphTests.Rand(rng, 9 * Width, -0.3f, 0.3f);
                weights.AddF32($"{l}.dw", kernel, 3, 3, 1, Width);
                var taps = new float[9 * Width];
                for (int c = 0; c < Width; c++) for (int t = 0; t < 9; t++) taps[t * Width + c] = kernel[c * 9 + t];
                weights.AddF32($"{l}.taps", taps, Width, 9);
            }
            else
            {
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.q", GraphFormat.F16, Width, Heads * HeadDim, rng);
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.k", GraphFormat.F16, Width, KvHeads * HeadDim, rng);
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.v", GraphFormat.F16, Width, KvHeads * HeadDim, rng);
                NetworkGraphTests.MakeProductWeight(weights, $"{l}.o", GraphFormat.F16, Heads * HeadDim, Width, rng);
                weights.AddF32($"{l}.qn", NetworkGraphTests.Rand(rng, HeadDim, 0.5f, 1.5f), HeadDim);
                weights.AddF32($"{l}.kn", NetworkGraphTests.Rand(rng, HeadDim, 0.5f, 1.5f), HeadDim);
            }
        }
        return weights;
    }

    private static bool IsConv(int layer) => layer < 2 || layer % 2 == 1;

    private static NetValue VisionTower(NetworkBuilder b, float[] input, bool composedConv)
    {
        NetValue x = b.Input(input, Width, Rows);
        NetValue table = b.RasterPositions(Grid, Grid);
        for (int l = 0; l < Layers; l++)
        {
            NetValue h = b.RmsNorm(x, $"{l}.ln1", Eps);
            NetValue mixed;
            if (IsConv(l))
            {
                NetValue p = b.Linear(h, $"{l}.in_proj");
                NetValue gated = b.Reshape(b.Mul(b.Columns(p, 0, Width), b.Columns(p, 2 * Width, Width)), Width, Grid, Grid);
                NetValue conv = composedConv ? ComposedDepthwise(b, gated, $"{l}.taps") : b.DepthwiseConv2d(gated, b.Weight($"{l}.dw"), 1, 1, 1, 1);
                mixed = b.Linear(b.Mul(b.Columns(p, Width, Width), b.Reshape(conv, Width, Rows)), $"{l}.out_proj");
            }
            else
            {
                mixed = b.SelfAttention(h, Heads, KvHeads, HeadDim, $"{l}.q", $"{l}.k", $"{l}.v", $"{l}.o", $"{l}.qn", $"{l}.kn", Eps,
                    v => b.Rope2D(v, table, 1e6f), 1f / MathF.Sqrt(HeadDim), default, default, default, out _, out _);
            }
            x = b.Add(x, mixed);
            x = b.Add(x, b.FeedForward(b.RmsNorm(x, $"{l}.ln2", Eps), $"{l}.gate", $"{l}.up", $"{l}.down"));
        }
        return x;
    }

    /// <summary>A 3x3, stride 1, pad 1 depthwise conv of [C, W, H] from ops every executor has: the padded map's nine tap
    /// views, each times its channel vector (rows of the [C, 9] taps weight), summed.</summary>
    private static NetValue ComposedDepthwise(NetworkBuilder b, NetValue x, string taps)
    {
        long c = x.Ne0, w = x.Ne1, h = x.Ne2;
        NetValue padded = b.PadMap(x, 1, 1);
        NetValue tapWeights = b.Weight(taps);
        NetValue acc = default;
        for (int ky = 0; ky < 3; ky++)
            for (int kx = 0; kx < 3; kx++)
            {
                NetValue view = b.View(padded, new[] { c, w, h }, new[] { c * 4, (w + 2) * c * 4 }, (kx * c + ky * (w + 2) * c) * 4);
                NetValue term = b.Mul(view, b.Rows(tapWeights, ky * 3 + kx, 1));
                acc = acc.Valid ? b.Add(acc, term) : term;
            }
        return acc;
    }

    private static double Median(Action run, int warmup = 1, int timed = 5)
    {
        for (int i = 0; i < warmup; i++) run();
        var ms = new double[timed];
        for (int i = 0; i < timed; i++)
        {
            var sw = Stopwatch.StartNew();
            run();
            ms[i] = sw.Elapsed.TotalMilliseconds;
        }
        Array.Sort(ms);
        return ms[timed / 2];
    }
}
