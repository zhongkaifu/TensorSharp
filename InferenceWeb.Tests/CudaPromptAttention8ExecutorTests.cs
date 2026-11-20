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
/// room keeps the 16-bit kernel's bits (and takes the pool's idle blocks before it is refused), and the path is the default
/// where it runs (issue #383), --prompt-attention f16 giving back the 16-bit-run kernels' bits.
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

    /// <summary>On by default where it runs (issue #383): with nothing chosen a marked prompt chunk takes the 8-bit kernel on
    /// every attention layer, and --prompt-attention f16 gives back the 16-bit-run kernels' bits (those of the earlier
    /// default). Over a q8_0 cache the default runs the 16-bit kernel (the 8-bit kernels read an F16 cache).</summary>
    [CudaFact(PromptAttention8 = true)]
    public void OnByDefaultWhereItRuns_AndF16GivesBackTheEarlierBits()
    {
        float[] eight, sixteen, sixteenAgain;
        using (KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16))
        {
            var (model, executor) = Create(attn8: null);
            using (model)
            {
                long layers = AttentionLayers(model);
                eight = (float[])model.ForwardPrompt(Tokens(PromptLength)).Clone();
                Assert.Equal(layers, executor.PromptAttention8Launches);
                Assert.True(((TensorSharp.Graph.TensorGraphExecutor)executor).PromptChunkAttentions > 0, "the prompt was not marked");
                executor.PromptAttentionForTests = TensorSharp.Graph.PromptAttentionPath.F16;
                model.ResetKVCache();
                sixteen = (float[])model.ForwardPrompt(Tokens(PromptLength)).Clone();
                Assert.Equal(layers, executor.PromptAttention8Launches);
            }
            // The 16-bit-run kernels' bits, as a model whose executor never took the 8-bit path gives them.
            var (fresh, freshExecutor) = Create(attn8: false);
            using (fresh)
                sixteenAgain = (float[])fresh.ForwardPrompt(Tokens(PromptLength)).Clone();
            Assert.Equal(0, freshExecutor.PromptAttention8Launches);
        }
        Assert.True(sixteen.AsSpan().SequenceEqual(sixteenAgain), "--prompt-attention f16 did not give the 16-bit-run kernels' bits");
        Assert.False(eight.AsSpan().SequenceEqual(sixteen), "the default run gave the 16-bit kernels' bits: it did not attend in 8 bits");
        using (KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.Q8_0))
        {
            var (model, executor) = Create(attn8: null);
            using (model)
            {
                model.ForwardPrompt(Tokens(PromptLength));
                Assert.Equal(0, executor.PromptAttention8Launches);
            }
        }
    }

    /// <summary>
    /// Issue #383: each path <c>--prompt-attention</c> chooses (CudaGraphExecutor.PromptAttentionForTests here) runs a prompt
    /// chunk's attention on its kernel, on any CUDA image. f16 gives the 16-bit-run kernels' bits run after run; f32acc takes
    /// the kernels with F32 accumulation on every attention layer, its own bits run after run, near f16's logits and not
    /// their bits; int8, where the image carries the 8-bit kernels, takes them on every attention layer.
    /// </summary>
    [CudaFact]
    public void EachPromptAttentionPathRunsItsKernelOnThePromptChunks()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        var (model, executor) = Create(attn8: null);
        using (model)
        {
            long layers = AttentionLayers(model);
            float[] Run(TensorSharp.Graph.PromptAttentionPath path)
            {
                executor.PromptAttentionForTests = path;
                model.ResetKVCache();
                return (float[])model.ForwardPrompt(Tokens(PromptLength)).Clone();
            }
            float[] f16 = Run(TensorSharp.Graph.PromptAttentionPath.F16);
            Assert.Equal(0, executor.PromptAttentionF32AccLaunches);
            Assert.Equal(0, executor.PromptAttention8Launches);
            float[] f32acc = Run(TensorSharp.Graph.PromptAttentionPath.F32Acc);
            Assert.Equal(layers, executor.PromptAttentionF32AccLaunches);
            float[] f32accAgain = Run(TensorSharp.Graph.PromptAttentionPath.F32Acc);
            float[] f16Again = Run(TensorSharp.Graph.PromptAttentionPath.F16);
            Assert.Equal(2 * layers, executor.PromptAttentionF32AccLaunches);
            Assert.True(f16.AsSpan().SequenceEqual(f16Again), "two f16 runs differ");
            Assert.True(f32acc.AsSpan().SequenceEqual(f32accAgain), "two f32acc runs differ");
            Assert.False(f32acc.AsSpan().SequenceEqual(f16), "f32acc gave f16's bits: the setting changed nothing");
            double rel = RelativeL2(f32acc, f16);
            var flash = ((CudaAllocator)executor.Allocator).FlashPrefillKernels;
            _output.WriteLine($"{PromptLength}-token prompt, {layers} attention layers: f32acc last-row logits rel L2 {rel:E3} against f16 " +
                $"(F32 accumulation at {flash.F32AccumulationRate:F2} of the F16 rate here)");
            Assert.True(rel <= 1e-2, $"f32acc last-row logits rel L2 {rel:E3} against f16");

            if (!flash.Runs8BitPrompt(256))
            {
                _output.WriteLine("int8 not run: this CUDA image carries no 8-bit prompt kernels (an sm_120a / sm_121a build does)");
                return;
            }
            float[] int8 = Run(TensorSharp.Graph.PromptAttentionPath.Int8);
            Assert.Equal(layers, executor.PromptAttention8Launches);
            double rel8 = RelativeL2(int8, f16);
            _output.WriteLine($"int8 last-row logits rel L2 {rel8:E3} against f16");
            Assert.True(rel8 <= 0.05, $"int8 last-row logits rel L2 {rel8:E3} against f16");
            Assert.Equal(2 * layers, executor.PromptAttentionF32AccLaunches);
        }
    }

    /// <summary>The arithmetic of the band's room (the device path is CudaPromptAttention8BandRoomTests): a cached block of its
    /// size, or the driver's free memory past the band and the reserve, made where the pool's idle large blocks can make it
    /// (only the shortfall asked for), refused where even all of them could not; a reserve of long.MaxValue (the tests' refusal
    /// seam) refuses without overflowing.</summary>
    [Fact]
    public void TheBandsTrimIsItsShortfallWhereTheIdleBlocksCoverIt()
    {
        const long MB = 1L << 20;
        Assert.Equal(0, CudaGraphExecutor.BandTrimBytes(true, 0, 0, 100 * MB, 512 * MB));
        Assert.Equal(0, CudaGraphExecutor.BandTrimBytes(false, 612 * MB, 0, 100 * MB, 512 * MB));
        Assert.Equal(1 * MB, CudaGraphExecutor.BandTrimBytes(false, 611 * MB, 1 * MB, 100 * MB, 512 * MB));
        Assert.Equal(400 * MB, CudaGraphExecutor.BandTrimBytes(false, 212 * MB, 2048 * MB, 100 * MB, 512 * MB));
        Assert.Equal(-1, CudaGraphExecutor.BandTrimBytes(false, 212 * MB, 399 * MB, 100 * MB, 512 * MB));
        Assert.Equal(-1, CudaGraphExecutor.BandTrimBytes(false, 30_000 * MB, long.MaxValue / 2, 100 * MB, long.MaxValue));
        Assert.Equal(-1, CudaGraphExecutor.BandTrimBytes(false, 0, 0, 100 * MB, 512 * MB));
    }
}

/// <summary>
/// The 8-bit prompt attention's band on the device when the driver alone has no room for it (a server near the card's edge,
/// whose model released conversations into the pool's cache for a long prompt): it gives back as many of the pool's idle large
/// blocks as its room needs, not all of them, and every marked chunk still attends in 8 bits; where even all of them would
/// not make the room it is refused without giving any back. Runs alone: it reads the driver's free memory, which another
/// test's allocations or frees would move between that reading and the band's.
/// </summary>
[Collection(DeviceFillingCollection.Name)]
public sealed class CudaPromptAttention8BandRoomTests : IDisposable
{
    private const long MiB = 1L << 20;
    private const int PromptLength = 1300;
    private readonly string _dir = Directory.CreateTempSubdirectory("prompt-attn8-band").FullName;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static int[] Tokens(int n) => Enumerable.Range(0, n).Select(i => 3 + (i * 37 + i / 7) % 250).ToArray();

    [CudaFact(PromptAttention8 = true)]
    public void TheBandTakesThePoolsIdleBlocksBeforeItIsRefused()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string trunk = new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "trunk.gguf"));
        using var model = new Qwen35RuntimeModel(trunk, BackendType.Cuda);
        var executor = Assert.IsType<CudaGraphExecutor>(model.GraphExecutorForTests);
        var cuda = Assert.IsType<CudaAllocator>(executor.Allocator);
        executor.PromptAttention8ForTests = true;

        // A first prompt rents the step's buffers, so the measured ones allocate little beyond the band.
        model.ForwardPrompt(Tokens(PromptLength));
        model.ResetKVCache();
        long layers = executor.PromptAttention8Launches;
        Assert.True(layers > 0, "the prompt did not attend in 8 bits");
        Assert.Equal(0, executor.PromptAttention8BandRefusals);

        // Idle large blocks of a size the trunk never rents.
        const long Block = 130 * MiB;
        const int Blocks = 8;
        var parked = Enumerable.Range(0, Blocks).Select(_ => new Tensor(cuda, DType.Float32, Block / sizeof(float))).ToList();
        cuda.Synchronize();
        parked.ForEach(t => t.Dispose());
        Assert.True(cuda.HasCachedBlock(Block), "the parked blocks were not cached");

        // The band asks for 300 MiB more than the driver has free: the pool's idle blocks make the room.
        long band = CudaFlashPrefillKernels.Attn8BandBytes(CudaFlashPrefillKernels.Attn8BandKeys(PromptLength), kvHeads: 2);
        CudaAllocatorStats before = cuda.GetStats();
        executor.Attn8BandHeadroomForTests = cuda.FreeDeviceBytes() - band + 300 * MiB;
        model.ForwardPrompt(Tokens(PromptLength));
        model.ResetKVCache();
        CudaAllocatorStats after = cuda.GetStats();
        Assert.Equal(0, executor.PromptAttention8BandRefusals);
        Assert.Equal(2 * layers, executor.PromptAttention8Launches);
        Assert.True(after.LargeTrimFrees > before.LargeTrimFrees, "the band found room without giving an idle block back");
        Assert.True(after.LargeCachedBytes < before.LargeCachedBytes,
            $"the pool caches {before.LargeCachedBytes / MiB} -> {after.LargeCachedBytes / MiB} MiB across a 300 MiB shortfall");
        Assert.True(cuda.HasCachedBlock(Block), "the band gave back every idle block, not its shortfall");

        // More than every idle block would make: refused on every attention layer, and nothing given back.
        executor.Attn8BandHeadroomForTests = cuda.FreeDeviceBytes() + cuda.ReclaimableBytes + 1024 * MiB;
        model.ForwardPrompt(Tokens(PromptLength));
        Assert.Equal(2 * layers, executor.PromptAttention8Launches);
        Assert.Equal(layers, executor.PromptAttention8BandRefusals);
        Assert.Equal(after.LargeTrimFrees, cuda.GetStats().LargeTrimFrees);
    }
}
