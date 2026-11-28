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
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A long prompt over whole-layer stages, pipelined (StandardDecoderModel.TryRunPipelinedPrompt: wavefront t runs stage s on
/// chunk t - s, the wavefront's stages together) against the same stages one chunk after another: the same logits bit for
/// bit, after the prompt and for decode steps after it, for families with attention only, Gated DeltaNet and Mamba state,
/// short convolutions, sliding rings, a stage carry and routed experts.
/// </summary>
public sealed class PipelinedPromptTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-pipelined-prompt-" + Guid.NewGuid().ToString("N"));
    private readonly ITestOutputHelper _output;

    public PipelinedPromptTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch { }
    }

    private string Fixture(string family) => family switch
    {
        "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = 6 }.Write(Path.Combine(_dir, "mistral3.gguf")),
        "qwen3vl" => new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "qwen3vlmoe", PreTokenizer = "qwen2", IncludeMistralControlTokens = false, IncludeQkNorms = true,
            RopeBase = 10000f, LayerCount = 4, MoeExperts = 4, MoeTopK = 2,
        }.Write(Path.Combine(_dir, "qwen3vl.gguf")),
        "qwen35" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "qwen35.gguf")),
        "nemotron_h" => new NemotronSyntheticModelBuilder().Write(Path.Combine(_dir, "nemotron.gguf")),
        "gemma4" => new Gemma4DenseSyntheticModelBuilder().Write(Path.Combine(_dir, "gemma4.gguf")),
        "lfm3" => new Lfm3SyntheticModelBuilder { UntiedOutput = true }.Write(Path.Combine(_dir, "lfm3.gguf")),
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    public static TheoryData<string, int> Cases => new()
    {
        { "mistral3", 2 }, { "mistral3", 3 }, { "qwen3vl", 2 }, { "qwen35", 2 }, { "nemotron_h", 2 }, { "gemma4", 2 }, { "lfm3", 2 },
    };

    [Theory]
    [MemberData(nameof(Cases))]
    public void PipelinedPromptIsTheStagesChunkAfterChunk(string family, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        var rng = new Random(11);
        int[] prompt = Enumerable.Range(0, 70).Select(_ => rng.Next(10, 250)).ToArray();
        (List<float[]> sequential, int none) = Run(family, path, stages, prompt, pipelined: false);
        (List<float[]> pipelined, int wavefronts) = Run(family, path, stages, prompt, pipelined: true);
        Assert.Equal(0, none);
        // 69 prompt rows in 16-row chunks: five chunks, through every stage.
        Assert.Equal(5 + stages - 1, wavefronts);
        Assert.Equal(sequential.Count, pipelined.Count);
        for (int i = 0; i < sequential.Count; i++)
            Assert.True(sequential[i].AsSpan().SequenceEqual(pipelined[i]),
                $"{family} stages {stages}: logits {i} differ (max |d| {sequential[i].Zip(pipelined[i], (a, b) => Math.Abs(a - b)).Max():E2})");
        _output.WriteLine($"{family} over {stages} stages: {wavefronts} wavefronts, {sequential.Count} logit rows bit for bit");
    }

    public static TheoryData<string> GgmlFamilies => new() { "mistral3", "qwen3vl", "qwen35", "gemma4", "lfm3" };

    /// <summary>The same on two ggml_cuda GPUs, where the wavefront's stages run as one ranked run (each device's graph queued
    /// before either is waited for): the stages' bits, chunk after chunk.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [MemberData(nameof(GgmlFamilies))]
    public void PipelinedPromptOnTwoGgmlCudaGpusIsTheStagesChunkAfterChunk(string family)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        var rng = new Random(13);
        int[] prompt = Enumerable.Range(0, 70).Select(_ => rng.Next(10, 250)).ToArray();
        (List<float[]> sequential, int none) = Run(family, path, 2, prompt, pipelined: false, BackendType.GgmlCuda);
        (List<float[]> pipelined, int wavefronts) = Run(family, path, 2, prompt, pipelined: true, BackendType.GgmlCuda);
        Assert.Equal(0, none);
        Assert.Equal(6, wavefronts);
        for (int i = 0; i < sequential.Count; i++)
            Assert.True(sequential[i].AsSpan().SequenceEqual(pipelined[i]),
                $"{family} on ggml_cuda: logits {i} differ (max |d| {sequential[i].Zip(pipelined[i], (a, b) => Math.Abs(a - b)).Max():E2})");
        _output.WriteLine($"{family} on two ggml_cuda GPUs: {wavefronts} wavefronts, {sequential.Count} logit rows bit for bit");
    }

    /// <summary>The engine's route: a lone request's whole prompt as one slot step (the solo prefill) runs pipelined, then its
    /// last row as a one-row step, and holds the bits of its slot fed the same 16-row chunks one step after another.</summary>
    [Theory]
    [MemberData(nameof(Cases))]
    public void PipelinedSlotPromptIsTheStagesChunkAfterChunk(string family, int stages)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        var rng = new Random(17);
        int[] prompt = Enumerable.Range(0, 70).Select(_ => rng.Next(10, 250)).ToArray();
        (List<float[]> sequential, int none, _) = RunSlot(family, path, stages, prompt, pipelined: false);
        (List<float[]> pipelined, int prompts, int wavefronts) = RunSlot(family, path, stages, prompt, pipelined: true);
        Assert.Equal(0, none);
        Assert.Equal(1, prompts);
        Assert.Equal(5 + stages - 1, wavefronts);
        Assert.Equal(sequential.Count, pipelined.Count);
        for (int i = 0; i < sequential.Count; i++)
            Assert.True(sequential[i].AsSpan().SequenceEqual(pipelined[i]),
                $"{family} stages {stages}: slot logits {i} differ (max |d| {sequential[i].Zip(pipelined[i], (a, b) => Math.Abs(a - b)).Max():E2})");
        _output.WriteLine($"{family} over {stages} stages: the slot's prompt in {wavefronts} wavefronts, {sequential.Count} logit rows bit for bit");
    }

    /// <summary>The slot route on two ggml_cuda GPUs.</summary>
    [GgmlTheory(BackendType.GgmlCuda, Devices = 2)]
    [MemberData(nameof(GgmlFamilies))]
    public void PipelinedSlotPromptOnTwoGgmlCudaGpusIsTheStagesChunkAfterChunk(string family)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        var rng = new Random(19);
        int[] prompt = Enumerable.Range(0, 70).Select(_ => rng.Next(10, 250)).ToArray();
        (List<float[]> sequential, int none, _) = RunSlot(family, path, 2, prompt, pipelined: false, BackendType.GgmlCuda);
        (List<float[]> pipelined, int prompts, int wavefronts) = RunSlot(family, path, 2, prompt, pipelined: true, BackendType.GgmlCuda);
        Assert.Equal(0, none);
        Assert.Equal(1, prompts);
        Assert.Equal(6, wavefronts);
        for (int i = 0; i < sequential.Count; i++)
            Assert.True(sequential[i].AsSpan().SequenceEqual(pipelined[i]),
                $"{family} on ggml_cuda: slot logits {i} differ (max |d| {sequential[i].Zip(pipelined[i], (a, b) => Math.Abs(a - b)).Max():E2})");
        _output.WriteLine($"{family} on two ggml_cuda GPUs: the slot's prompt in {wavefronts} wavefronts, {sequential.Count} logit rows bit for bit");
    }

    /// <summary>A model whose lone prompts pipeline asks the scheduler for 8,192-token steps, a remainder of at most a quarter
    /// step joined to the step before it; none for a prompt of one chunk or where it does not pipeline.</summary>
    [Fact]
    public void PipelinedPromptsChooseTheirLonePromptSteps()
    {
        string path = Fixture("mistral3");
        using var model = new Mistral3Model(path, BackendType.Cpu, 1, null, 2);
        model.PromptChunkTokensForTests = 16;
        model.PipelinedPromptForTests = true;
        Assert.Equal(StandardDecoderModel.PipelinedSoloStepTokens, model.PreferredSoloPrefillTokens(0, int.MaxValue / 2));
        Assert.Equal(8192, model.PreferredSoloPrefillTokens(0, 20_000));
        Assert.Equal(10_000, model.PreferredSoloPrefillTokens(0, 10_000));
        Assert.Equal(8192, model.PreferredSoloPrefillTokens(8192, 10_241));
        Assert.Equal(0, model.PreferredSoloPrefillTokens(0, 16));
        // Once a prompt is followed (the engine speculates), the scheduler's own steps.
        using (var followed = new Mistral3Model(path, BackendType.Cpu, 1, null, 2))
        {
            followed.PromptChunkTokensForTests = 16;
            followed.PipelinedPromptForTests = true;
            Assert.Equal(8192, followed.PreferredSoloPrefillTokens(0, 20_000));
            followed.FollowPlainPrompt("r", 20_000);
            Assert.Equal(0, followed.PreferredSoloPrefillTokens(0, 20_000));
        }
        model.PipelinedPromptDisabledForTests = true;
        Assert.Equal(0, model.PreferredSoloPrefillTokens(0, 20_000));
        using var single = new Mistral3Model(path, BackendType.Cpu, 1, null, 1);
        single.PipelinedPromptForTests = true;
        Assert.Equal(0, single.PreferredSoloPrefillTokens(0, 20_000));
    }

    /// <summary>A lone prompt step the model chose for its pipeline that does not pipeline after all runs in the steps it would
    /// have had (the refill's chunks): as many steps, the bits of those steps one after another, nothing pipelined.</summary>
    [Theory]
    [InlineData("mistral3")]
    [InlineData("qwen35")]
    public void AChosenStepThatDoesNotPipelineRunsInTheStepsItWouldHaveHad(string family)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        var rng = new Random(29);
        int[] prompt = Enumerable.Range(0, 70).Select(_ => rng.Next(10, 250)).ToArray();
        (float[] Logits, int Steps) Prompt(bool chosen)
        {
            using ModelBase model = Open(family, path, BackendType.Cpu, 2);
            var sdm = (StandardDecoderModel)model;
            sdm.PromptChunkTokensForTests = 16;
            sdm.PipelinedPromptForTests = true;
            string[] ids = { "r" };
            sdm.EnsureSequenceSlot("r", 128);
            var logits = new float[1][];
            if (chosen)
            {
                Assert.Equal(prompt.Length, sdm.PreferredSoloPrefillTokens(0, prompt.Length));
                sdm.PipelinedPromptDisabledForTests = true;   // as a request's LoRA would, after the step was chosen
                Assert.True(sdm.TryForwardUnified(ids, new[] { prompt }, new[] { 0 }, logits));
            }
            else
            {
                sdm.PipelinedPromptDisabledForTests = true;
                for (int from = 0; from < prompt.Length; from += 16)
                    Assert.True(sdm.TryForwardUnified(ids, new[] { prompt[from..Math.Min(prompt.Length, from + 16)] }, new[] { from }, logits));
            }
            Assert.Equal(0, sdm.PipelinedWavefronts);
            return ((float[])logits[0].Clone(), sdm.BatchedForwardCount);
        }
        var expected = Prompt(chosen: false);
        var actual = Prompt(chosen: true);
        Assert.Equal(5, expected.Steps);
        Assert.Equal(expected.Steps, actual.Steps);
        Assert.True(expected.Logits.AsSpan().SequenceEqual(actual.Logits), $"{family}: the fallback's logits differ");
    }

    public static TheoryData<string, bool> CudaCases => new()
    {
        { "mistral3", false }, { "qwen3vl", false }, { "qwen35", false }, { "nemotron_h", false }, { "gemma4", false }, { "lfm3", false },
        { "mistral3", true }, { "qwen3vl", true }, { "qwen35", true }, { "nemotron_h", true }, { "gemma4", true }, { "lfm3", true },
    };

    /// <summary>On two direct-CUDA GPUs, where a wavefront runs its stages one after another, the last first (a run that
    /// reads nothing back returns with its device work queued): the refill and the slot route, the stages' bits chunk after
    /// chunk.</summary>
    [MultiCudaTheory(2)]
    [MemberData(nameof(CudaCases))]
    public void PipelinedPromptOnTwoCudaGpusIsTheStagesChunkAfterChunk(string family, bool slot)
    {
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
        string path = Fixture(family);
        var rng = new Random(23);
        int[] prompt = Enumerable.Range(0, 70).Select(_ => rng.Next(10, 250)).ToArray();
        List<float[]> sequential, pipelined;
        int wavefronts;
        if (slot)
        {
            (sequential, _, _) = RunSlot(family, path, 2, prompt, pipelined: false, BackendType.Cuda);
            (pipelined, int prompts, wavefronts) = RunSlot(family, path, 2, prompt, pipelined: true, BackendType.Cuda);
            Assert.Equal(1, prompts);
        }
        else
        {
            (sequential, _) = Run(family, path, 2, prompt, pipelined: false, BackendType.Cuda);
            (pipelined, wavefronts) = Run(family, path, 2, prompt, pipelined: true, BackendType.Cuda);
        }
        Assert.Equal(6, wavefronts);
        for (int i = 0; i < sequential.Count; i++)
            Assert.True(sequential[i].AsSpan().SequenceEqual(pipelined[i]),
                $"{family} on cuda ({(slot ? "slot" : "refill")}): logits {i} differ (max |d| {sequential[i].Zip(pipelined[i], (a, b) => Math.Abs(a - b)).Max():E2})");
        _output.WriteLine($"{family} on two CUDA GPUs ({(slot ? "slot" : "refill")}): {wavefronts} wavefronts, {sequential.Count} logit rows bit for bit");
    }

    private static ModelBase Open(string family, string path, BackendType backend, int stages) => family switch
    {
        "mistral3" => new Mistral3Model(path, backend, 1, null, stages),
        "qwen3vl" => new Qwen3VLModel(path, backend, 1, null, stages),
        "qwen35" => new Qwen35RuntimeModel(path, backend, 1, null, stages),
        "nemotron_h" => new NemotronRuntimeModel(path, backend, 1, null, stages),
        "gemma4" => new Gemma4RuntimeModel(path, backend, 1, null, stages),
        "lfm3" => new Lfm3RuntimeModel(path, backend, 1, null, stages),
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    /// <summary>A request's slot: its prompt as one step (pipelined) or as 16-row steps and its last row (one after another),
    /// then three greedy decode steps.</summary>
    private static (List<float[]> Logits, int Prompts, int Wavefronts) RunSlot(string family, string path, int stages, int[] prompt, bool pipelined,
        BackendType backend = BackendType.Cpu)
    {
        using ModelBase model = Open(family, path, backend, stages);
        var sdm = Assert.IsAssignableFrom<StandardDecoderModel>(model);
        Assert.True(sdm.SupportsUnifiedBatching, $"{family} serves unified steps");
        sdm.PromptChunkTokensForTests = 16;
        sdm.PipelinedPromptForTests = pipelined && backend == BackendType.Cpu;
        sdm.PipelinedPromptDisabledForTests = !pipelined;
        const string id = "r";
        string[] ids = { id };
        sdm.EnsureSequenceSlot(id, 128);
        var logits = new float[1][];
        if (pipelined)
        {
            Assert.True(sdm.TryForwardUnified(ids, new[] { prompt }, new[] { 0 }, logits));
        }
        else
        {
            int last = prompt.Length - 1;
            for (int from = 0; from < last; from += 16)
                Assert.True(sdm.TryForwardUnified(ids, new[] { prompt[from..Math.Min(last, from + 16)] }, new[] { from }, logits));
            Assert.True(sdm.TryForwardUnified(ids, new[] { new[] { prompt[last] } }, new[] { last }, logits));
        }
        var rows = new List<float[]> { (float[])logits[0].Clone() };
        for (int i = 0; i < 3; i++)
        {
            float[] row = rows[^1];
            int next = Array.IndexOf(row, row.Max());
            Assert.True(sdm.TryForwardUnified(ids, new[] { new[] { next } }, new[] { prompt.Length + i }, logits));
            rows.Add((float[])logits[0].Clone());
        }
        return (rows, sdm.PipelinedSlotPrompts, sdm.PipelinedWavefronts);
    }

    /// <summary>The refill's logits, then three greedy decode steps' (each step's input the argmax of the last).</summary>
    private static (List<float[]> Logits, int Wavefronts) Run(string family, string path, int stages, int[] prompt, bool pipelined,
        BackendType backend = BackendType.Cpu)
    {
        // The families directly (on the CPU over simulated host devices: the serving contract admits --layer-split on GPU
        // backends only).
        using ModelBase model = Open(family, path, backend, stages);
        var sdm = Assert.IsAssignableFrom<StandardDecoderModel>(model);
        Assert.Equal(stages, sdm.Mesh.Stages);
        sdm.PromptChunkTokensForTests = 16;
        sdm.PipelinedPromptForTests = pipelined && backend == BackendType.Cpu;
        sdm.PipelinedPromptDisabledForTests = !pipelined;
        var logits = new List<float[]> { (float[])model.ForwardRefill(prompt).Clone() };
        for (int i = 0; i < 3; i++)
        {
            float[] last = logits[^1];
            int next = Array.IndexOf(last, last.Max());
            logits.Add((float[])model.Forward(new[] { next }).Clone());
        }
        return (logits, sdm.PipelinedWavefronts);
    }
}
