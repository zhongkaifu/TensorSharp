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
using TensorSharp.Models.Graph;
using TensorSharp.Runtime;
using Xunit;
using static InferenceWeb.Tests.NetworkGraphTests;

namespace InferenceWeb.Tests;

/// <summary>
/// What the LFM3 speech networks need of the network foundation (TensorSharp.Models.Graph), on every executor this machine
/// runs: a state one program writes and another only reads, replayed (GGML: below the 4 KB device-cache threshold such a
/// state lives in compute memory and is written back after every run).
/// </summary>
public sealed class NetworkGraphSpeechFoundationTests
{
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
