// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The causal masks a shared-runtime step emits: one input a sequence, whatever the attention layers, with the values the
// layers read before; and prompt chunks' masks given as their pattern (GraphCapabilities.StructuralCausalMasks) answer as
// their host values did, bit for bit (synthetic Qwen3.5 and Qwen3.8 fixtures; the managed CPU executor through its test
// seam, the CUDA executor on a GPU).
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Runtime.Scheduling;

namespace InferenceWeb.Tests;

public sealed class StepCausalMaskTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-step-masks-" + Guid.NewGuid().ToString("N"));

    /// <summary>The fixture's layers; every fourth attends (full_attention_interval 4).</summary>
    private const int Layers = 8, AttentionLayers = Layers / 4;

    public StepCausalMaskTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Qwen35Fixture()
        => new Qwen35DenseSyntheticModelBuilder { Layers = Layers, UntiedOutput = true }.Write(Path.Combine(_dir, "q35-dense.gguf"));

    /// <summary>
    /// A step of two sequences (one decode row and another request's prompt) emits each sequence's mask once, read by every
    /// attention layer: the step had kept one mask, which each sequence's evicted, so every layer emitted both again. The
    /// logits of that step, of the prompt before it and of the two sequences' decode steps after it (advanced a position
    /// on, each sequence's mask by its own rows) are the bits the per-layer masks gave.
    /// </summary>
    [Fact]
    public void AStepOfTwoSequencesEmitsEachSequencesMaskOnce()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = Qwen35Fixture();
        var cached = RunTwoSequences(path, singleMaskCache: false);
        var perLayer = RunTwoSequences(path, singleMaskCache: true);

        Assert.Equal(2, cached.MixedStepMasks);
        Assert.Equal(2 * AttentionLayers, perLayer.MixedStepMasks);
        Assert.Equal(2, cached.DecodeStepMasks);
        Assert.Equal(2 * AttentionLayers, perLayer.DecodeStepMasks);
        Assert.True(cached.Advances > 0, "the two sequences' decode steps did not advance");
        Assert.Equal(perLayer.Logits.Count, cached.Logits.Count);
        for (int i = 0; i < cached.Logits.Count; i++)
            Assert.True(cached.Logits[i].AsSpan().SequenceEqual(perLayer.Logits[i]), $"step {i}: the per-sequence masks moved the logits");
    }

    private static (List<float[]> Logits, long MixedStepMasks, long DecodeStepMasks, int Advances) RunTwoSequences(string path,
        bool singleMaskCache)
    {
        using var model = new Qwen35RuntimeModel(path, BackendType.Cpu, 1, null, 1);
        model.SingleMaskCacheForTests = singleMaskCache;
        var unified = (IUnifiedBatchModel)model;
        foreach (string id in new[] { "a", "b" }) unified.EnsureSequenceSlot(id, 128);
        var trace = new List<float[]>();
        int[] promptA = Enumerable.Range(0, 20).Select(i => 40 + (i * 11) % 150).ToArray();
        int[] promptB = Enumerable.Range(0, 24).Select(i => 30 + (i * 7) % 180).ToArray();

        var one = new float[1][];
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { promptA }, new[] { 0 }, one));
        trace.Add((float[])one[0].Clone());
        int nextA = ArgMax(one[0]);

        // A's decode row and B's prompt in one step.
        var two = new float[2][];
        int runs = model.GraphRuns;
        long before = model.CausalMaskInputs;
        Assert.True(unified.TryForwardUnified(new[] { "a", "b" }, new[] { new[] { nextA }, promptB }, new[] { promptA.Length, 0 }, two));
        long mixed = model.CausalMaskInputs - before;
        Assert.True(model.GraphRuns > runs, "the mixed step did not run as a graph");
        trace.AddRange(two.Select(r => (float[])r.Clone()));

        // Both decode: the first step traces, the later ones advance the kept step.
        int[] starts = { promptA.Length + 1, promptB.Length };
        int[][] tokens = { new[] { ArgMax(two[0]) }, new[] { ArgMax(two[1]) } };
        long decodeMasks = -1;
        int advances = model.DecodeAdvances;
        for (int step = 0; step < 6; step++)
        {
            before = model.CausalMaskInputs;
            Assert.True(unified.TryForwardUnified(new[] { "a", "b" }, tokens, starts, two));
            if (step == 0) decodeMasks = model.CausalMaskInputs - before;
            trace.AddRange(two.Select(r => (float[])r.Clone()));
            for (int s = 0; s < 2; s++) starts[s]++;
            tokens = new[] { new[] { ArgMax(two[0]) }, new[] { ArgMax(two[1]) } };
        }
        return (trace, mixed, decodeMasks, model.DecodeAdvances - advances);
    }

    // ---- structural masks: a prompt chunk's mask as its pattern

    /// <summary>
    /// On the managed CPU executor with prompt chunks' masks forced to arrive as their pattern: a 2048-row chunk (its mask
    /// past the row-parallel fill), a step of the next 52 rows with another request's 100-row prompt, then both decoding.
    /// The executor's attention reads a prompt mask's values, which the interpreter makes from the pattern, except where its
    /// tiled host attention (ARM64: HostFlashAttention) reads a causal chunk's pattern alone, as the CUDA kernels do: the
    /// logits are the host masks' bits, and every structural mask bound was made at most once (each one, off ARM64).
    /// </summary>
    [Fact]
    public void StructuralMasksOnTheManagedCpuAnswerAsHostMasks()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = Qwen35Fixture();
        var host = RunPromptChunks(() => new Qwen35RuntimeModel(path, BackendType.Cpu, 1, null, 1), structural: false);
        var pattern = RunPromptChunks(() => new Qwen35RuntimeModel(path, BackendType.Cpu, 1, null, 1), structural: true);
        Assert.Equal(0, host.Bound);
        Assert.True(pattern.Bound >= 3, $"{pattern.Bound} structural masks bound");   // a's two chunks and b's prompt
        if (HostFlashAttention.Supported) Assert.InRange(pattern.Materialized, 0, pattern.Bound);
        else Assert.Equal(pattern.Bound, pattern.Materialized);
        Assert.Equal(host.MaskInputs, pattern.MaskInputs);
        AssertSameLogits(host.Logits, pattern.Logits, "managed cpu");
    }

    /// <summary>Qwen3.8's QSA past its dense budget (a top-k of 8: dense up to 11 cells) on the managed CPU: a 23-row prompt
    /// chunk and a 17-row one, both selecting, then decode rows. The selection's top-k mask reads the causal mask's values
    /// (made from the pattern): the same bits as the host masks.</summary>
    [Fact]
    public void StructuralMasksUnderQsaSelectionOnTheManagedCpuAnswerAsHostMasks()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-sparse.gguf"), indexerTopK: 8, qsa: true);
        var host = RunSelectingPrompt(() => new Qwen4ExpRuntimeModel(path, BackendType.Cpu), structural: false);
        var pattern = RunSelectingPrompt(() => new Qwen4ExpRuntimeModel(path, BackendType.Cpu), structural: true);
        Assert.Equal(0, host.Bound);
        Assert.Equal(2, pattern.Bound);
        Assert.Equal(pattern.Bound, pattern.Materialized);
        AssertSameLogits(host.Logits, pattern.Logits, "managed cpu, qsa selecting");
    }

    /// <summary>
    /// The CUDA executor takes prompt chunks' masks as their pattern: three chunks of one request (700, 500 and 300 rows,
    /// windows of 768, 1280 and 1536 keys), a step of its next 300 rows with another request's 100-row prompt, and both
    /// decoding (recorded and advanced). The prompt kernels read only the pattern's keys and floor, so no mask's values are
    /// made, and the logits are the host masks' bits.
    /// </summary>
    [CudaFact]
    public void StructuralMasksOnDirectCudaAnswerAsHostMasks()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = Qwen35Fixture();
        var host = RunPromptChunks(() => new Qwen35RuntimeModel(path, BackendType.Cuda, 1, null, 1), structural: false, cudaChunks: true);
        var pattern = RunPromptChunks(() => new Qwen35RuntimeModel(path, BackendType.Cuda, 1, null, 1), structural: true, cudaChunks: true);
        Assert.Equal(0, host.Bound);
        Assert.True(pattern.Bound >= 5, $"{pattern.Bound} structural masks bound");
        Assert.Equal(0, pattern.Materialized);
        Assert.Equal(host.MaskInputs, pattern.MaskInputs);
        AssertSameLogits(host.Logits, pattern.Logits, "cuda");
    }

    /// <summary>QSA past its budget on the CUDA executor (selecting prompt chunks of 23 and 17 rows, then decode rows):
    /// whatever reads the causal mask there (the selection region, or its ops' top-k mask over values made from the
    /// pattern), the logits are the host masks' bits.</summary>
    [CudaFact]
    public void StructuralMasksUnderQsaSelectionOnDirectCudaAnswerAsHostMasks()
    {
        using IDisposable dtype = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e-sparse.gguf"), indexerTopK: 8, qsa: true);
        var host = RunSelectingPrompt(() => new Qwen4ExpRuntimeModel(path, BackendType.Cuda), structural: false);
        var pattern = RunSelectingPrompt(() => new Qwen4ExpRuntimeModel(path, BackendType.Cuda), structural: true);
        Assert.Equal(0, host.Bound);
        Assert.Equal(2, pattern.Bound);
        AssertSameLogits(host.Logits, pattern.Logits, "cuda, qsa selecting");
    }

    private readonly record struct MaskRun(List<float[]> Logits, long Bound, long Materialized, long MaskInputs);

    private static TensorGraphExecutor Structural(StandardDecoderModel model, bool structural)
    {
        var executor = Assert.IsAssignableFrom<TensorGraphExecutor>(model.GraphExecutorForTests);
        if (executor is ManagedGraphExecutor managed) managed.StructuralMasksForTests = structural;
        else SetCudaStructural(executor, structural);
        Assert.Equal(structural, executor.Capabilities.StructuralCausalMasks);
        return executor;
    }

    private static void SetCudaStructural(TensorGraphExecutor executor, bool structural)
        => Assert.IsType<TensorSharp.Cuda.CudaGraphExecutor>(executor).StructuralMasksForTests = structural;

    private static MaskRun RunPromptChunks(Func<StandardDecoderModel> create, bool structural, bool cudaChunks = false)
    {
        using StandardDecoderModel model = create();
        TensorGraphExecutor executor = Structural(model, structural);
        var unified = (IUnifiedBatchModel)model;
        foreach (string id in new[] { "a", "b" }) unified.EnsureSequenceSlot(id, 2304);
        int[] a = Enumerable.Range(0, 2100).Select(i => 3 + (i * 37) % 250).ToArray();
        int[] b = Enumerable.Range(0, 100).Select(i => 5 + (i * 53) % 240).ToArray();
        // a's chunks; the last rides with b's prompt.
        int[] cuts = cudaChunks ? new[] { 700, 1200, 1500, 1800 } : new[] { 2048, 2100 };
        var trace = new List<float[]>();
        var one = new float[1][];
        var two = new float[2][];
        int at = 0;
        for (int c = 0; c < cuts.Length - 1; c++)
        {
            Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { a[at..cuts[c]] }, new[] { at }, one));
            trace.Add((float[])one[0].Clone());
            at = cuts[c];
        }
        Assert.True(unified.TryForwardUnified(new[] { "a", "b" }, new[] { a[at..cuts[^1]], b }, new[] { at, 0 }, two));
        trace.AddRange(two.Select(r => (float[])r.Clone()));
        int[] starts = { cuts[^1], b.Length };
        for (int step = 0; step < 4; step++)
        {
            int[][] tokens = { new[] { ArgMax(two[0]) }, new[] { ArgMax(two[1]) } };
            Assert.True(unified.TryForwardUnified(new[] { "a", "b" }, tokens, starts, two));
            trace.AddRange(two.Select(r => (float[])r.Clone()));
            for (int s = 0; s < 2; s++) starts[s]++;
        }
        return new MaskRun(trace, executor.StructuralMasksBound, executor.StructuralMasksMaterialized, model.CausalMaskInputs);
    }

    private static MaskRun RunSelectingPrompt(Func<StandardDecoderModel> create, bool structural)
    {
        using StandardDecoderModel model = create();
        TensorGraphExecutor executor = Structural(model, structural);
        model.ResetKVCache();
        int[] prompt = Enumerable.Range(0, 40).Select(i => 3 + (i * 37) % 240).ToArray();
        var trace = new List<float[]>();
        model.ForwardRefill(prompt[..23]);
        trace.Add((float[])model.Forward(prompt[23..]).Clone());
        foreach (int token in Enumerable.Range(0, 6).Select(i => 5 + (i * 11) % 200))
            trace.Add((float[])model.Forward(new[] { token }).Clone());
        return new MaskRun(trace, executor.StructuralMasksBound, executor.StructuralMasksMaterialized, model.CausalMaskInputs);
    }

    private static void AssertSameLogits(List<float[]> expected, List<float[]> actual, string what)
    {
        Assert.Equal(expected.Count, actual.Count);
        for (int i = 0; i < expected.Count; i++)
            Assert.True(expected[i].AsSpan().SequenceEqual(actual[i]), $"{what}: row {i} moved with the structural masks");
    }

    private static int ArgMax(float[] row)
    {
        int best = 0;
        for (int v = 1; v < row.Length; v++) if (row[v] > row[best]) best = v;
        return best;
    }
}
