// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;
using Xunit;
using static InferenceWeb.Tests.NetworkGraphTests;

namespace InferenceWeb.Tests;

/// <summary>
/// What the LFM3 speech networks added to the network foundation (TensorSharp.Models.Graph), on every executor this machine
/// runs (managed CPU, the tensor executor's reference decompositions = direct CUDA's, GGML cpu and metal, MLX):
/// <list type="bullet">
/// <item><see cref="NetworkBuilder.MulMatId"/> over <see cref="NetworkBuilder.Experts"/>: matrices of a stacked table picked
/// per row by indices that ride the run (one program for every pick);</item>
/// <item><see cref="NetworkBuilder.AttentionByProducts"/> (<see cref="NetworkGraph.AttentionByProducts"/>): attention as
/// F32 products around a masked softmax;</item>
/// <item>a state one program writes and another only reads, replayed (GGML: below the 4 KB device-cache threshold such a
/// state lives in compute memory and is written back after every run).</item>
/// </list>
/// </summary>
public sealed class NetworkGraphSpeechFoundationTests
{
    public static IEnumerable<object[]> Formats => new[] { GraphFormat.F16, GraphFormat.Q8_0 }.Select(f => new object[] { f });

    [Theory]
    [MemberData(nameof(Formats))]
    public void ExpertsManaged(int format) => Experts(Lane.Managed, format);

    [Theory]
    [MemberData(nameof(Formats))]
    public void ExpertsReference(int format) => Experts(Lane.Reference, format);

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(Formats))]
    public void ExpertsGgmlCpu(int format) => Experts(Lane.GgmlCpu, format);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(Formats))]
    public void ExpertsGgmlMetal(int format) => Experts(Lane.GgmlMetal, format);

    [MlxTheory]
    [MemberData(nameof(Formats))]
    public void ExpertsMlx(int format) => Experts(Lane.Mlx, format);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(Formats))]
    public void ExpertsGgmlCuda(int format) => Experts(Lane.GgmlCuda, format);

    [GgmlTheory(BackendType.GgmlVulkan)]
    [MemberData(nameof(Formats))]
    public void ExpertsGgmlVulkan(int format) => Experts(Lane.GgmlVulkan, format);

    [CudaTheory]
    [MemberData(nameof(Formats))]
    public void ExpertsDirectCuda(int format) => Experts(Lane.Cuda, format);

    /// <summary>A [64, 5 x 40] table read as 5 matrices of 40 rows: rows of x each multiplied by the matrix its index picks;
    /// three different picks replay one program. Where the format is held dense (F16 on MLX) the stack is not made and the
    /// caller multiplies every matrix (checked: <see cref="NetworkBuilder.Experts"/> returns no value).</summary>
    private static void Experts(Lane laneKind, int format)
    {
        using var lane = new LaneContext(laneKind);
        const int Width = 64, Rows = 40, Count = 5, N = 4;
        var rng = new Random(31 + format);
        MakeProductWeight(lane.Weights, "table", format, Width, Rows * Count, rng);
        float[] table = lane.Weights.ReadF32("table");
        bool stacked = lane.Weights.Get("table").IsQuantized;
        float[] x = Rand(rng, Width * N);
        foreach (int[] ids in new[] { new[] { 4, 0, 2, 2 }, new[] { 1, 3, 0, 4 }, new[] { 3, 3, 3, 3 } })
        {
            if (!stacked)
            {
                lane.Graph.Run(b =>
                {
                    Assert.False(b.Experts("table", Count).Valid);
                    b.Output(b.Input(x, Width, N));
                });
                return;
            }
            float[] got = lane.Run(b => b.MulMatId(b.Experts("table", Count), b.Input(x, Width, 1, N), b.InputI32(ids, 1, N)));
            var expected = new float[Rows * N];
            for (int t = 0; t < N; t++)
                for (int o = 0; o < Rows; o++)
                {
                    double s = 0;
                    for (int i = 0; i < Width; i++) s += (double)table[((long)ids[t] * Rows + o) * Width + i] * x[t * Width + i];
                    expected[t * Rows + o] = (float)s;
                }
            AssertClose(expected, got, format == GraphFormat.Q8_0 ? 2e-2 : 4e-3, $"{laneKind} mul_mat_id {string.Join(",", ids)}");
        }
        if (lane.Graph.Executor.Capabilities.Replays) Assert.Equal(1, lane.Graph.Builds);
    }

    public static IEnumerable<object[]> AttentionKinds => new[] { "full", "sliding_prefix" }.Select(k => new object[] { k });

    [Theory]
    [MemberData(nameof(AttentionKinds))]
    public void AttentionByProductsManaged(string kind) => AttentionByProducts(Lane.Managed, kind);

    [Theory]
    [MemberData(nameof(AttentionKinds))]
    public void AttentionByProductsReference(string kind) => AttentionByProducts(Lane.Reference, kind);

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(AttentionKinds))]
    public void AttentionByProductsGgmlCpu(string kind) => AttentionByProducts(Lane.GgmlCpu, kind);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(AttentionKinds))]
    public void AttentionByProductsGgmlMetal(string kind) => AttentionByProducts(Lane.GgmlMetal, kind);

    [MlxTheory]
    [MemberData(nameof(AttentionKinds))]
    public void AttentionByProductsMlx(string kind) => AttentionByProducts(Lane.Mlx, kind);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(AttentionKinds))]
    public void AttentionByProductsGgmlCuda(string kind) => AttentionByProducts(Lane.GgmlCuda, kind);

    [GgmlTheory(BackendType.GgmlVulkan)]
    [MemberData(nameof(AttentionKinds))]
    public void AttentionByProductsGgmlVulkan(string kind) => AttentionByProducts(Lane.GgmlVulkan, kind);

    [CudaTheory]
    [MemberData(nameof(AttentionKinds))]
    public void AttentionByProductsDirectCuda(string kind) => AttentionByProducts(Lane.Cuda, kind);

    /// <summary>Grouped-query attention (8 / 2 heads of 32) as products, against a host softmax: bidirectional over 12 rows
    /// (the depthformer's), and a sliding window of 30 over 29 carried keys plus 6 rows (the codec's streaming call).</summary>
    private static void AttentionByProducts(Lane laneKind, string kind)
    {
        using var lane = new LaneContext(laneKind);
        lane.Graph.AttentionByProducts = true;
        lane.Graph.AttentionHalfKv = false;
        const int D = 32, Heads = 8, KvHeads = 2;
        bool sliding = kind == "sliding_prefix";
        int rows = sliding ? 6 : 12, keys = sliding ? 29 + 6 : 12, start = sliding ? 3 : 0;
        var rng = new Random(rows);
        float[] q = Rand(rng, D * Heads * rows), k = Rand(rng, D * KvHeads * keys), v = Rand(rng, D * KvHeads * keys);
        ushort[] bits = sliding ? NetworkMasks.SlidingWindow(rows, keys, 30, start, start - 29) : null;
        float scale = 1f / MathF.Sqrt(D);
        float[] got = lane.Run(b => b.Attention(b.Input(q, D, Heads, rows), b.Input(k, D, KvHeads, keys), b.Input(v, D, KvHeads, keys), scale,
            sliding ? b.Mask(bits, keys, rows) : default));
        var expected = new float[D * Heads * rows];
        for (int t = 0; t < rows; t++)
            for (int h = 0; h < Heads; h++)
            {
                int kh = h / (Heads / KvHeads);
                var s = new double[keys];
                double max = double.NegativeInfinity;
                for (int j = 0; j < keys; j++)
                {
                    if (sliding && bits[t * keys + j] != NetworkMasks.Open) { s[j] = double.NegativeInfinity; continue; }
                    double dot = 0;
                    for (int i = 0; i < D; i++) dot += (double)q[(t * Heads + h) * D + i] * k[(j * KvHeads + kh) * D + i];
                    s[j] = dot * scale;
                    max = Math.Max(max, s[j]);
                }
                double sum = 0;
                for (int j = 0; j < keys; j++) sum += s[j] = Math.Exp(s[j] - max);
                for (int i = 0; i < D; i++)
                {
                    double acc = 0;
                    for (int j = 0; j < keys; j++) acc += s[j] / sum * v[(j * KvHeads + kh) * D + i];
                    expected[(t * Heads + h) * D + i] = (float)acc;
                }
            }
        // ggml-metal's and MLX's F32 products round through F16 operands (llama.cpp's Metal run of these graphs too).
        AssertClose(expected, got, laneKind == Lane.Mlx || HalfOperandLane(laneKind) ? 2e-3 : 1e-5, $"{laneKind} attention by products ({kind})");
    }

    public static IEnumerable<object[]> StateSizes => new[] { 24, 3000 }.Select(n => new object[] { n });

    [Theory]
    [MemberData(nameof(StateSizes))]
    public void ReadOnlyStateReplaysManaged(int n) => ReadOnlyStateReplays(Lane.Managed, n);

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(StateSizes))]
    public void ReadOnlyStateReplaysGgmlCpu(int n) => ReadOnlyStateReplays(Lane.GgmlCpu, n);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(StateSizes))]
    public void ReadOnlyStateReplaysGgmlMetal(int n) => ReadOnlyStateReplays(Lane.GgmlMetal, n);

    [MlxTheory]
    [MemberData(nameof(StateSizes))]
    public void ReadOnlyStateReplaysMlx(int n) => ReadOnlyStateReplays(Lane.Mlx, n);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(StateSizes))]
    public void ReadOnlyStateReplaysGgmlCuda(int n) => ReadOnlyStateReplays(Lane.GgmlCuda, n);

    [GgmlTheory(BackendType.GgmlVulkan)]
    [MemberData(nameof(StateSizes))]
    public void ReadOnlyStateReplaysGgmlVulkan(int n) => ReadOnlyStateReplays(Lane.GgmlVulkan, n);

    [CudaTheory]
    [MemberData(nameof(StateSizes))]
    public void ReadOnlyStateReplaysDirectCuda(int n) => ReadOnlyStateReplays(Lane.Cuda, n);

    /// <summary>
    /// One program writes a state, another only reads it (the depthformer's projection, kept from a frame's first pass for its
    /// next two): replaying the reader must keep reading what the writer left. GGML keeps a small state (below its 4 KB
    /// device-cache threshold) in a program's compute memory, uploaded before and written back after every run: the
    /// write-back must not read memory the allocator reused after the state's last read (it did before the state was made a
    /// graph output: the second replay read garbage).
    /// </summary>
    private static void ReadOnlyStateReplays(Lane laneKind, int n)
    {
        using var lane = new LaneContext(laneKind);
        var rng = new Random(n);
        float[] written = Rand(rng, n), other = Rand(rng, n);
        using var state = new Tensor(lane.Allocator, DType.Float32, n);
        state.SetElementsAsFloat(new float[n]);
        lane.Graph.Run(b =>
        {
            NetValue v = b.Scale(b.Input(written, n), 2f);
            b.Write(v, b.State(state));
            b.Output(v);
        });
        for (int run = 0; run < 3; run++)
        {
            // The state's last read is the first op; the rest of the program can take its memory.
            float[] got = lane.Run(b =>
            {
                NetValue s = b.Add(b.State(state), b.Input(other, n));
                NetValue t = b.Silu(b.Scale(s, 0.5f));
                return b.Sub(b.Mul(t, t), b.Input(other, n));
            });
            var expected = new float[n];
            for (int i = 0; i < n; i++)
            {
                double s = 2 * written[i] + other[i], t = s * 0.5 / (1 + Math.Exp(-s * 0.5));
                expected[i] = (float)(t * t - other[i]);
            }
            AssertClose(expected, got, 1e-5, $"{laneKind} read-only state ({n} values), run {run}");
        }
    }
}
