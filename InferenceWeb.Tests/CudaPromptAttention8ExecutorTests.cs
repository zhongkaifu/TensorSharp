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
using TensorSharp.Cuda;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// The CUDA executor's 8-bit prompt attention behind its seam (CudaGraphExecutor.PromptAttention8ForTests), on an RTX Blackwell
/// image (skipped elsewhere), over a synthetic Qwen 3.5 dense trunk with an F16 cache: a marked prompt chunk takes it (the
/// logits near the 16-bit run's), verify windows, decode rows and plain forwards never do (a verify row still the decode row
/// at its position, bit for bit), speculation over 8-bit prompts emits plain greedy decoding's stream, a band without device
/// room keeps the 16-bit kernel's bits, and the path is off by default.
/// </summary>
public sealed class CudaPromptAttention8ExecutorTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("prompt-attn8").FullName;
    private readonly ITestOutputHelper _output;

    public CudaPromptAttention8ExecutorTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private const int PromptLength = 1300;

    private string _trunk;

    /// <summary>Eight layers (two of them full attention, 8 query heads over 2 KV heads of 256), an untied head.</summary>
    private string Trunk() => _trunk ??= new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "trunk.gguf"));

    private static int[] Tokens(int n, int salt = 0) => Enumerable.Range(0, n).Select(i => 3 + (i * 37 + i / 7 + salt) % 250).ToArray();

    private (Qwen35RuntimeModel Model, CudaGraphExecutor Executor) Create(bool? attn8, string draft = null)
    {
        var model = draft == null ? new Qwen35RuntimeModel(Trunk(), BackendType.Cuda) : new Qwen35RuntimeModel(Trunk(), BackendType.Cuda, 1, null, 1, draft);
        var executor = Assert.IsType<CudaGraphExecutor>(model.GraphExecutorForTests);
        executor.PromptAttention8ForTests = attn8;
        return (model, executor);
    }

    /// <summary>The trunk's attention layers: the multi-row attention ops of one unmarked prompt-sized forward.</summary>
    private static long AttentionLayers(Qwen35RuntimeModel model)
    {
        var executor = (TensorSharp.Graph.TensorGraphExecutor)model.GraphExecutorForTests;
        long before = executor.OtherMultiRowAttentions;
        model.Forward(Tokens(40, 77));
        model.ResetKVCache();
        long layers = executor.OtherMultiRowAttentions - before;
        Assert.True(layers > 0, "no multi-row attention ran as a graph");
        return layers;
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

    [CudaFact(PromptAttention8 = true)]
    public void MarkedPromptChunksTakeThe8BitKernel()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        var (model, executor) = Create(attn8: true);
        using (model)
        {
            long layers = AttentionLayers(model);
            var checks = new List<(long Call, double Rel, double Max)>();
            executor.PromptAttention8CheckForTests = (call, rel, max) => checks.Add((call, rel, max));
            float[] eight = (float[])model.ForwardPrompt(Tokens(PromptLength)).Clone();
            Assert.Equal(layers, executor.PromptAttention8Launches);
            Assert.Equal(0, executor.PromptAttention8BandRefusals);
            foreach (var c in checks) _output.WriteLine($"8-bit launch {c.Call}: rel RMS {c.Rel:E3}, max |d| {c.Max:E3} against the 16-bit kernel");
            Assert.All(checks, c => Assert.True(c.Rel <= 0.03, $"launch {c.Call}: attention rel RMS {c.Rel:E3} against the 16-bit kernel"));

            executor.PromptAttention8CheckForTests = null;
            executor.PromptAttention8ForTests = false;
            model.ResetKVCache();
            float[] sixteen = (float[])model.ForwardPrompt(Tokens(PromptLength)).Clone();
            Assert.Equal(layers, executor.PromptAttention8Launches);
            double rel = RelativeL2(eight, sixteen);
            _output.WriteLine($"{PromptLength}-token prompt, {layers} attention layers: last-row logits rel L2 {rel:E3} against the 16-bit run");
            Assert.True(rel <= 0.05, $"last-row logits rel L2 {rel:E3} against the 16-bit run");
            Assert.False(eight.AsSpan().SequenceEqual(sixteen), "the 8-bit run gave the 16-bit run's bits: it did not attend in 8 bits");
        }
    }

    /// <summary>After an 8-bit prompt, an 8-row verify window and decode rows take no 8-bit launch, and every verify row is the
    /// decode row at its position, bit for bit (the invariant the speculative contract rests on).</summary>
    [CudaFact(PromptAttention8 = true)]
    public void VerifyWindowsAndDecodeRowsKeepTheirKernels()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        var (model, executor) = Create(attn8: true);
        using (model)
        {
            long layers = AttentionLayers(model);
            var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
            var batched = Assert.IsAssignableFrom<IBatchedSpeculativeTarget>(model);
            int vocab = model.Config.VocabSize;
            int[] prompt = Tokens(PromptLength), window = Tokens(8, 5);
            var last = new float[1][];
            unified.EnsureSequenceSlot("verify", PromptLength + 64);
            Assert.True(unified.TryForwardUnified(new[] { "verify" }, new[] { prompt }, new[] { 0 }, last));
            Assert.Equal(layers, executor.PromptAttention8Launches);
            var rows = new float[8 * vocab];
            Assert.True(batched.TryBatchedSpecVerifyLogits(new[] { "verify" }, new[] { PromptLength }, 8, window, rows, null));
            batched.CommitBatchedSpecVerify(new[] { "verify" }, new[] { PromptLength }, new[] { 8 });
            Assert.Equal(layers, executor.PromptAttention8Launches);
            model.OnSequenceReleased("verify");

            unified.EnsureSequenceSlot("decode", PromptLength + 64);
            Assert.True(unified.TryForwardUnified(new[] { "decode" }, new[] { prompt }, new[] { 0 }, last));
            Assert.Equal(2 * layers, executor.PromptAttention8Launches);
            int apart = 0;
            for (int r = 0; r < 8; r++)
            {
                Assert.True(unified.TryForwardUnified(new[] { "decode" }, new[] { new[] { window[r] } }, new[] { PromptLength + r }, last));
                if (!last[0].AsSpan().SequenceEqual(rows.AsSpan(r * vocab, vocab))) apart++;
            }
            Assert.Equal(2 * layers, executor.PromptAttention8Launches);
            _output.WriteLine($"verify rows apart from the decode rows: {apart} of 8");
            Assert.Equal(0, apart);
            model.OnSequenceReleased("decode");
        }
    }

    /// <summary>Speculation with a DFlash2 drafter over a 1,300-token prompt, both arms prefilling it in one chunk on the 8-bit
    /// kernels, emits plain greedy decoding's stream.</summary>
    [CudaFact(PromptAttention8 = true)]
    public void SpeculationStaysLossless_With8BitPrompts()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string draft = Qwen4ExpSyntheticModelBuilder.WriteDFlash(Path.Combine(_dir, "dflash2.gguf"), blockSize: 5,
            hidden: Qwen35DenseSyntheticModelBuilder.Hidden, layers: 2, slidingWindow: 16, dflash2: true);
        var executors = new List<CudaGraphExecutor>();
        SpeculativeRuntimeChecks.AssertBlockDraftMatchesPlainGreedy(() =>
            {
                var (model, executor) = Create(attn8: true, draft);
                executors.Add(executor);
                return model;
            }, "qwen35 cuda dflash2 8-bit prompt", maxNew: 24, prompt: Tokens(PromptLength));
        Assert.Equal(2, executors.Count);
        foreach (CudaGraphExecutor e in executors) _output.WriteLine($"8-bit launches {e.PromptAttention8Launches}");
        Assert.All(executors, e => Assert.True(e.PromptAttention8Launches > 0, "an arm's prompt did not attend in 8 bits"));
    }

    /// <summary>A band that finds no device room leaves the chunk on the 16-bit kernel: no launch, the refusal counted, and the
    /// logits the 16-bit run's bits.</summary>
    [CudaFact(PromptAttention8 = true)]
    public void BandRefusalKeepsTheF16Kernel()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        var (model, executor) = Create(attn8: true);
        using (model)
        {
            long layers = AttentionLayers(model);
            executor.Attn8BandHeadroomForTests = long.MaxValue;
            float[] refused = (float[])model.ForwardPrompt(Tokens(PromptLength)).Clone();
            Assert.Equal(0, executor.PromptAttention8Launches);
            Assert.Equal(layers, executor.PromptAttention8BandRefusals);
            executor.PromptAttention8ForTests = false;
            model.ResetKVCache();
            float[] sixteen = (float[])model.ForwardPrompt(Tokens(PromptLength)).Clone();
            Assert.True(refused.AsSpan().SequenceEqual(sixteen), "a refused band changed the bits");
        }
    }

    /// <summary>Rows not marked as a prompt chunk (a plain Forward of as many rows) never take the 8-bit kernels.</summary>
    [CudaFact(PromptAttention8 = true)]
    public void UnmarkedRowsNeverTakeIt()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        var (model, executor) = Create(attn8: true);
        using (model)
        {
            model.Forward(Tokens(PromptLength));
            Assert.Equal(0, executor.PromptAttention8Launches);
            Assert.Equal(0, executor.PromptAttention8BandRefusals);
        }
    }

    /// <summary>Off by default: a marked prompt chunk keeps the 16-bit kernel unless the seam asks.</summary>
    [CudaFact(PromptAttention8 = true)]
    public void OffByDefault()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        var (model, executor) = Create(attn8: null);
        using (model)
        {
            model.ForwardPrompt(Tokens(PromptLength));
            Assert.Equal(0, executor.PromptAttention8Launches);
            Assert.True(((TensorSharp.Graph.TensorGraphExecutor)executor).PromptChunkAttentions > 0, "the prompt was not marked");
        }
    }
}
