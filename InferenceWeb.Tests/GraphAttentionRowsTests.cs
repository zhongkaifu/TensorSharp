// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Graph;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The prompt-chunk mark (FlashAttnExt ip[2] = GraphAttentionRows.PromptChunk), which an executor may attend with arithmetic
/// decode rows do not share (the CUDA executor's 8-bit prompt attention): set on a sequence's attention exactly when its rows
/// are a prompt chunk of more than 16 rows reaching the model through a prompt entry point (the unified step, ForwardPrompt,
/// ForwardRefill, the speculative prefill), never on decode rows, short prompt tails, verify windows of any width, a
/// re-forward of generated rows or a plain Forward. Unmarked programs keep their signatures. On the managed CPU executor, whose
/// arithmetic ignores the mark.
/// </summary>
public sealed class GraphAttentionRowsTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("attention-rows").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Trunk() => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "trunk.gguf"));

    private static int[] Tokens(int n, int salt = 0) => Enumerable.Range(0, n).Select(i => 3 + (i * 37 + i / 7 + salt) % 250).ToArray();

    private static TensorGraphExecutor Executor(StandardDecoderModel model)
        => Assert.IsAssignableFrom<TensorGraphExecutor>(model.GraphExecutorForTests);

    /// <summary>The attention ops a step of <paramref name="run"/> marked and left unmarked (of more than one row).</summary>
    private static (long Marked, long Other) Count(StandardDecoderModel model, Action run)
    {
        TensorGraphExecutor executor = Executor(model);
        long marked = executor.PromptChunkAttentions, other = executor.OtherMultiRowAttentions;
        run();
        return (executor.PromptChunkAttentions - marked, executor.OtherMultiRowAttentions - other);
    }

    /// <summary>The attention layers of the trunk: the multi-row ops one unmarked 40-row forward runs.</summary>
    private static long AttentionLayers(StandardDecoderModel model)
    {
        var (marked, other) = Count(model, () => model.Forward(Tokens(40, 99)));
        Assert.Equal(0, marked);
        Assert.True(other > 0, "no multi-row attention ran (the step did not run as a graph)");
        model.ResetKVCache();
        return other;
    }

    [Fact]
    public void UnifiedPromptChunksAreMarked()
    {
        using var model = new Qwen35RuntimeModel(Trunk(), BackendType.Cpu);
        long layers = AttentionLayers(model);
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        unified.EnsureSequenceSlot("a", 128);
        var (marked, other) = Count(model, () => Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { Tokens(40) }, new[] { 0 }, new float[1][])));
        Assert.Equal(layers, marked);
        Assert.Equal(0, other);
    }

    [Fact]
    public void ShortPromptTailsAndDecodeRowsAreNot()
    {
        using var model = new Qwen35RuntimeModel(Trunk(), BackendType.Cpu);
        long layers = AttentionLayers(model);
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        unified.EnsureSequenceSlot("a", 128);
        var last = new float[1][];
        var (marked, other) = Count(model, () => Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { Tokens(10) }, new[] { 0 }, last)));
        Assert.Equal(0, marked);
        Assert.Equal(layers, other);
        (marked, other) = Count(model, () => Decode(unified, last));
        Assert.Equal(0, marked);
        Assert.Equal(0, other);

        // Decode steps advance as they do with the mark machinery off.
        using var off = new Qwen35RuntimeModel(Trunk(), BackendType.Cpu) { PromptChunkMarksDisabledForTests = true };
        var offUnified = Assert.IsAssignableFrom<IUnifiedBatchModel>(off);
        offUnified.EnsureSequenceSlot("a", 128);
        Assert.True(offUnified.TryForwardUnified(new[] { "a" }, new[] { Tokens(10) }, new[] { 0 }, last));
        Decode(offUnified, last);
        Assert.Equal(off.DecodeAdvances, model.DecodeAdvances);

        static void Decode(IUnifiedBatchModel unified, float[][] last)
        {
            for (int pos = 10; pos < 14; pos++)
                Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { 7 + pos } }, new[] { pos }, last));
        }
    }

    [Fact]
    public void MixedStepMarksOnlyThePromptSequence()
    {
        using var model = new Qwen35RuntimeModel(Trunk(), BackendType.Cpu);
        long layers = AttentionLayers(model);
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        unified.EnsureSequenceSlot("decode", 128);
        unified.EnsureSequenceSlot("prompt", 128);
        Assert.True(unified.TryForwardUnified(new[] { "decode" }, new[] { Tokens(12, 5) }, new[] { 0 }, new float[1][]));
        var (marked, other) = Count(model, () => Assert.True(unified.TryForwardUnified(new[] { "decode", "prompt" },
            new[] { new[] { 11 }, Tokens(40) }, new[] { 12, 0 }, new float[2][])));
        Assert.Equal(layers, marked);
        Assert.Equal(0, other);
    }

    /// <summary>A batched verify window, and a solo verify wider than any batched one (24 rows: the width a prompt-sized
    /// window would have), are never marked: the mark, not the row count, governs.</summary>
    [Fact]
    public void VerifyWindowsAreNeverMarked()
    {
        using var model = new Qwen35RuntimeModel(Trunk(), BackendType.Cpu);
        long layers = AttentionLayers(model);
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        var batched = Assert.IsAssignableFrom<IBatchedSpeculativeTarget>(model);
        unified.EnsureSequenceSlot("a", 128);
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { Tokens(40) }, new[] { 0 }, new float[1][]));
        int vocab = model.Config.VocabSize;
        var (marked, other) = Count(model, () =>
            Assert.True(batched.TryBatchedSpecVerifyLogits(new[] { "a" }, new[] { 40 }, 8, Tokens(8, 3), new float[8 * vocab], null)));
        Assert.Equal(0, marked);
        Assert.Equal(layers, other);
        batched.CommitBatchedSpecVerify(new[] { "a" }, new[] { 40 }, new[] { 8 });
        model.OnSequenceReleased("a");

        var spec = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        model.ResetKVCache();
        spec.SpecForwardPrompt(Tokens(40), null, new float[vocab]);
        (marked, other) = Count(model, () => spec.SpecForward(Tokens(24, 9), null, new float[24 * vocab], allLogitsRows: true));
        Assert.Equal(0, marked);
        Assert.Equal(layers, other);
    }

    [Fact]
    public void SoloPrefillIsMarked_ReforwardIsNot()
    {
        using var model = new Qwen35RuntimeModel(Trunk(), BackendType.Cpu);
        long layers = AttentionLayers(model);
        var spec = Assert.IsAssignableFrom<ISpeculativeTarget>(model);
        int vocab = model.Config.VocabSize;
        var (marked, other) = Count(model, () => spec.SpecForwardPrompt(Tokens(40), null, new float[vocab]));
        Assert.Equal(layers, marked);
        Assert.Equal(0, other);
        // The solo speculation's re-forward of kept rows (and any SpecForward): unmarked.
        (marked, other) = Count(model, () => spec.SpecForward(Tokens(40, 1), null, new float[vocab], allLogitsRows: false));
        Assert.Equal(0, marked);
        Assert.Equal(layers, other);
        // The runtime's speculative prefill reaches the model's prompt entry.
        model.ResetKVCache();
        var trunk = new LinearSpecTrunk(spec);
        (marked, _) = Count(model, () => trunk.ForwardPrompt(Tokens(40), null, new float[vocab]));
        Assert.Equal(layers, marked);
    }

    [Fact]
    public void ForwardIsNotAPrompt()
    {
        using var model = new Qwen35RuntimeModel(Trunk(), BackendType.Cpu);
        long layers = AttentionLayers(model);
        var (marked, other) = Count(model, () => model.Forward(Tokens(40)));
        Assert.Equal(0, marked);
        Assert.Equal(layers, other);
        model.ResetKVCache();
        (marked, other) = Count(model, () => model.ForwardPrompt(Tokens(40)));
        Assert.Equal(layers, marked);
        Assert.Equal(0, other);
        model.ResetKVCache();
        (marked, _) = Count(model, () => model.ForwardRefill(Tokens(600)));
        Assert.Equal(layers, marked);
        // A one-row tail of a prompt (ForwardPrompt of one token) is a decode-shaped step: unmarked.
        (marked, other) = Count(model, () => model.ForwardPrompt(new[] { 9 }));
        Assert.Equal(0, marked);
        Assert.Equal(0, other);
    }

    /// <summary>The quality probe's scored prompt chunk (ForwardPromptRowsForTests): a marked chunk returning every row's
    /// logits, the last of which is a plain prompt chunk's, and advancing the sequence as one does (the next token's logits
    /// equal after either).</summary>
    [Fact]
    public void ScoredPromptChunksAreMarkedAndAdvanceAsPromptChunks()
    {
        using var model = new Qwen35RuntimeModel(Trunk(), BackendType.Cpu);
        long layers = AttentionLayers(model);
        int vocab = model.Config.VocabSize;
        int[] prompt = Tokens(40), next = { 17 };
        float[] rows = null;
        var (marked, other) = Count(model, () => rows = model.ForwardPromptRowsForTests(prompt));
        Assert.Equal(layers, marked);
        Assert.Equal(0, other);
        Assert.Equal(prompt.Length * vocab, rows.Length);
        float[] afterScored = (float[])model.Forward(next).Clone();

        model.ResetKVCache();
        float[] last = (float[])model.ForwardPrompt(prompt).Clone();
        float[] afterPrompt = (float[])model.Forward(next).Clone();
        float[] scoredLast = rows.AsSpan((prompt.Length - 1) * vocab, vocab).ToArray();
        Assert.True(RelativeL2(scoredLast, last) <= 1e-5, $"the scored chunk's last row is {RelativeL2(scoredLast, last):E3} from a prompt chunk's logits");
        Assert.True(RelativeL2(afterScored, afterPrompt) <= 1e-5,
            $"the next token's logits after a scored chunk are {RelativeL2(afterScored, afterPrompt):E3} from those after a prompt chunk");
        Assert.Equal(ArgMax(last), ArgMax(scoredLast));
    }

    private static double RelativeL2(float[] a, float[] b)
    {
        double diff = 0, norm = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double e = (double)a[i] - b[i];
            diff += e * e;
            norm += (double)b[i] * b[i];
        }
        return Math.Sqrt(diff / Math.Max(norm, 1e-30));
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    /// <summary>Unmarked attention ops keep the ip they had before the mark (one or two values: ggml's precision and a mask
    /// row's finite keys), so a decode step's, a verify step's and a short chunk's programs keep their bytes and signatures; a
    /// prompt chunk's ops carry the mark as ip[2], and none do with the mark machinery off.</summary>
    [Fact]
    public void UnmarkedProgramsKeepTheirSignatures()
    {
        using var model = new Qwen35RuntimeModel(Trunk(), BackendType.Cpu) { DecodeAdvanceDisabled = true };
        long layers = AttentionLayers(model);
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        var batched = Assert.IsAssignableFrom<IBatchedSpeculativeTarget>(model);
        var ips = new System.Collections.Generic.List<long[]>();
        model.AttentionIpForTests = ips;
        System.Collections.Generic.List<long[]> Emitted(Action run)
        {
            ips.Clear();
            run();
            Assert.NotEmpty(ips);
            return new System.Collections.Generic.List<long[]>(ips);
        }
        static bool Unmarked(long[] ip) => ip.Length <= 2;

        unified.EnsureSequenceSlot("a", 128);
        var prompt = Emitted(() => Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { Tokens(40) }, new[] { 0 }, new float[1][])));
        Assert.Equal(layers, prompt.Count);
        Assert.All(prompt, ip => Assert.True(ip.Length == 3 && ip[0] == 10 && ip[2] == GraphAttentionRows.PromptChunk,
            $"a prompt chunk's attention ip [{string.Join(",", ip)}]"));
        Assert.All(Emitted(() => Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { 5 } }, new[] { 40 }, new float[1][]))),
            ip => Assert.True(Unmarked(ip), $"a decode step's attention ip [{string.Join(",", ip)}]"));
        Assert.All(Emitted(() => Assert.True(batched.TryBatchedSpecVerifyLogits(new[] { "a" }, new[] { 41 }, 8, Tokens(8, 3),
                new float[8 * model.Config.VocabSize], null))),
            ip => Assert.True(Unmarked(ip), $"a verify step's attention ip [{string.Join(",", ip)}]"));
        unified.EnsureSequenceSlot("b", 128);
        Assert.All(Emitted(() => Assert.True(unified.TryForwardUnified(new[] { "b" }, new[] { Tokens(12) }, new[] { 0 }, new float[1][]))),
            ip => Assert.True(Unmarked(ip), $"a 12-row chunk's attention ip [{string.Join(",", ip)}]"));

        model.PromptChunkMarksDisabledForTests = true;
        unified.EnsureSequenceSlot("c", 128);
        Assert.All(Emitted(() => Assert.True(unified.TryForwardUnified(new[] { "c" }, new[] { Tokens(40) }, new[] { 0 }, new float[1][]))),
            ip => Assert.True(Unmarked(ip), $"with the marks off, a prompt chunk's attention ip [{string.Join(",", ip)}]"));
    }

    /// <summary>Speculation over a long prompt (1,100 tokens, prefilled in one marked chunk by both arms) emits plain greedy
    /// decoding's stream: the mark changes no emission besides the op's ip.</summary>
    [Fact]
    public void SpeculationMatchesPlain_LongPrompt_OnTheManagedCpu()
    {
        string trunk = Trunk();
        int hidden = Qwen35DenseSyntheticModelBuilder.Hidden;
        string draft = Qwen4ExpSyntheticModelBuilder.WriteDFlash(Path.Combine(_dir, "dflash2.gguf"), blockSize: 5, hidden: hidden, layers: 2,
            slidingWindow: 16, dflash2: true);
        var executors = new System.Collections.Generic.List<TensorGraphExecutor>();
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() =>
            {
                var model = new Qwen35RuntimeModel(trunk, BackendType.Cpu, 1, null, 1, draft);
                executors.Add(Executor(model));
                return model;
            }, "qwen35 cpu dflash2 long prompt", maxNew: 16, prompt: Tokens(1100));
        // Both arms (the plain refill and the speculative prefill) attended the prompt as a marked chunk.
        Assert.Equal(2, executors.Count);
        Assert.All(executors, e => Assert.True(e.PromptChunkAttentions > 0, "an arm's prompt was not marked"));
    }
}
