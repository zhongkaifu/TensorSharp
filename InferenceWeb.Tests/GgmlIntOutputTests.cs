// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using System.Runtime.InteropServices;
using TensorSharp.GGML;
using TensorSharp.Graph;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// An I32 node that leaves a GGML graph reads back as its values, like every other output (<see cref="GraphProgramBuilder.Output"/>:
/// F32 values): the ids of a top-k that is not a greedy head (StandardDecoderModel's draft chain reads its ten), and of a
/// one-pick top-k over a row wide enough to lower to ggml_argmax (a block drafter's draft). They had come back as their raw
/// bits, which read as denormals: every draft id was token 0, and on ggml_cuda no MTP draft was ever accepted.
/// </summary>
public sealed class GgmlIntOutputTests
{
    [Theory]
    [InlineData(64, 10)]
    [InlineData(151643, 10)]   // Qwen3.8's scored draft vocabulary: ids past 2^16
    [InlineData(151643, 1)]    // one pick over a wide row: ggml_argmax
    public void TopKIdsReadBackAsTheirValues(int vocab, int k)
    {
        GgmlBackendType backend = TestGates.PinnedGgmlBackendType;
        GgmlBasicOps.EnsureBackendAvailable(backend);
        const int rows = 3;
        var rng = new Random(20261124 + vocab + k);
        var logits = new float[(long)vocab * rows];
        for (int i = 0; i < logits.Length; i++) logits[i] = (float)(rng.NextDouble() * 4 - 3);
        // Distinct maxima, the largest late in the row, so the expected ids are unambiguous and large.
        for (int r = 0; r < rows; r++)
            for (int j = 0; j < k; j++)
                logits[(long)r * vocab + vocab - 1 - 3 * j - r] = 10f - j;

        var g = new GraphProgramBuilder(record: true);
        int input = g.Leaf(GraphLeafRole.Input, GraphFormat.F32, new long[] { vocab, rows }, IntPtr.Zero, logits.LongLength * 4);
        // As the draft head emits them (StandardDecoderModel.EmitDraftLayer): the top-k's ids and the values they pick.
        int ids = g.Op(GraphOp.Cont, stackalloc[] { g.Op(GraphOp.TopK, stackalloc[] { input }, stackalloc long[] { k }) });
        int idsOut = g.Output(ids, (long)k * rows);
        int byRow = g.Op(GraphOp.Reshape, stackalloc[] { input }, stackalloc long[] { 1, vocab, rows });
        int valuesOut = g.Output(g.Op(GraphOp.GetRows, stackalloc[] { byRow, ids }), (long)k * rows);
        using IGraphPlan plan = GgmlGraphExecutor.For(backend).Compile(g.ToProgram(replayable: true, reorder: false));
        GCHandle pin = GCHandle.Alloc(logits, GCHandleType.Pinned);
        try
        {
            for (int run = 0; run < 2; run++)
            {
                Assert.True(plan.Run(new[] { input }, new[] { pin.AddrOfPinnedObject() }, new[] { logits.LongLength * 4 }), plan.LastError);
                for (int r = 0; r < rows; r++)
                {
                    float[] got = plan.Outputs[idsOut].AsSpan(r * k, k).ToArray();
                    int[] expected = Enumerable.Range(0, k).Select(j => vocab - 1 - 3 * j - r).ToArray();
                    Assert.True(got.All(v => v == MathF.Floor(v) && v >= 0 && v < vocab),
                        $"{backend} vocab {vocab} k {k} run {run} row {r}: ids [{string.Join(", ", got)}] are not ids");
                    Assert.Equal(expected.OrderBy(x => x), got.Select(v => (int)v).OrderBy(x => x));
                    // The largest first: the draft chain takes ids[0] as its draft.
                    Assert.Equal(expected[0], (int)got[0]);
                    for (int j = 0; j < k; j++)
                        Assert.Equal(logits[(long)r * vocab + (int)got[j]], plan.Outputs[valuesOut][r * k + j]);
                }
            }
        }
        finally
        {
            pin.Free();
        }
    }
}
