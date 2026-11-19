// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// A sampled batched verify draws each row where the model holds it (<see cref="IBatchedSpeculativeTarget"/>'s
/// TryBatchedSpecVerifyLogits with a scratch fallback): the graph model hands back its verify output, which must start
/// with the very rows the copying overload writes, and leaves the caller's scratch alone; a target without such a buffer
/// copies the rows into the scratch, grown to the asked length. On the managed CPU executor.
/// </summary>
public sealed class BatchedVerifyHeldLogitsTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("held-logits").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static int[] Tokens(int n, int salt = 0) => Enumerable.Range(0, n).Select(i => 3 + (i * 37 + i / 7 + salt) % 250).ToArray();

    private static int[] Bits(ReadOnlySpan<float> values)
    {
        var bits = new int[values.Length];
        for (int i = 0; i < values.Length; i++) bits[i] = BitConverter.SingleToInt32Bits(values[i]);
        return bits;
    }

    [Fact]
    public void HeldRowsAreTheRowsTheCopyingVerifyWrites()
    {
        string path = new Qwen35DenseSyntheticModelBuilder { Layers = 4, UntiedOutput = true }.Write(Path.Combine(_dir, "trunk.gguf"));
        using var copying = new Qwen35RuntimeModel(path, BackendType.Cpu);
        using var holding = new Qwen35RuntimeModel(path, BackendType.Cpu);
        int vocab = copying.Config.VocabSize, window = 8;
        int[] prompt = Tokens(40), rows = Tokens(window, 3);
        foreach (var model in new[] { copying, holding })
        {
            var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
            unified.EnsureSequenceSlot("a", 128);
            Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { prompt }, new[] { 0 }, new float[1][]));
        }

        var copied = new float[window * vocab];
        Assert.True(((IBatchedSpeculativeTarget)copying).TryBatchedSpecVerifyLogits(new[] { "a" }, new[] { 40 }, window, rows, copied, null));

        float[]? scratch = null;
        Assert.True(((IBatchedSpeculativeTarget)holding).TryBatchedSpecVerifyLogits(new[] { "a" }, new[] { 40 }, window, rows, null,
            ref scratch, (long)window * vocab, out float[] held));
        Assert.Null(scratch);                                    // the model holds the rows: no scratch made
        Assert.True(held.LongLength >= (long)window * vocab);
        Assert.Contains(copied, x => x != 0);
        Assert.Equal(Bits(copied), Bits(held.AsSpan(0, window * vocab)));

        // The window commits as a copying verify's does: the next step's rows match too.
        foreach (var model in new[] { copying, holding })
            ((IBatchedSpeculativeTarget)model).CommitBatchedSpecVerify(new[] { "a" }, new[] { 40 }, new[] { 3 });
        int[] next = Tokens(window, 9);
        Assert.True(((IBatchedSpeculativeTarget)copying).TryBatchedSpecVerifyLogits(new[] { "a" }, new[] { 43 }, window, next, copied, null));
        Assert.True(((IBatchedSpeculativeTarget)holding).TryBatchedSpecVerifyLogits(new[] { "a" }, new[] { 43 }, window, next, null,
            ref scratch, (long)window * vocab, out held));
        Assert.Null(scratch);
        Assert.Equal(Bits(copied), Bits(held.AsSpan(0, window * vocab)));
    }

    [Fact]
    public void TargetWithoutHeldRowsCopiesThemIntoTheScratch()
    {
        var target = new CopyingTarget();
        float[]? scratch = new float[3];
        Assert.True(((IBatchedSpeculativeTarget)target).TryBatchedSpecVerifyLogits(new[] { "a" }, new[] { 5 }, 2, new[] { 1, 2 }, null,
            ref scratch, 8, out float[] logits));
        Assert.NotNull(scratch);
        Assert.Equal(8, scratch!.Length);                         // grown to the asked length
        Assert.Same(scratch, logits);
        Assert.Equal(new[] { 0f, 1f, 2f, 3f, 4f, 5f, 6f, 7f }, logits);

        float[] kept = scratch;
        Assert.True(((IBatchedSpeculativeTarget)target).TryBatchedSpecVerifyLogits(new[] { "a" }, new[] { 5 }, 2, new[] { 1, 2 }, null,
            ref scratch, 6, out logits));
        Assert.Same(kept, scratch);                               // long enough: reused
    }

    /// <summary>A target with only the copying overload (every target but the graph models).</summary>
    private sealed class CopyingTarget : IBatchedSpeculativeTarget
    {
        public int BatchedSpecWindow(int sequences) => 2;
        public int FusedCacheSeqLen(string requestId) => 0;
        public IDisposable BeginBatchedSpeculation() => new MemoryStream();
        public bool TryBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int window, int[] tokens, int[] drawn,
            float[] features) => false;
        public void CommitBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int[] kept) { }
        public void BindDraftState(string? requestId) { }
        public void ReleaseDraftState(string requestId) { }

        public bool TryBatchedSpecVerifyLogits(IReadOnlyList<string> requestIds, int[] positions, int window, int[] tokens,
            float[] logits, float[] features)
        {
            for (int i = 0; i < logits.Length; i++) logits[i] = i;
            return true;
        }
    }
}
