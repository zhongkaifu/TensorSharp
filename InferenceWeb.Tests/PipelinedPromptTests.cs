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

    /// <summary>The refill's logits, then three greedy decode steps' (each step's input the argmax of the last).</summary>
    private static (List<float[]> Logits, int Wavefronts) Run(string family, string path, int stages, int[] prompt, bool pipelined,
        BackendType backend = BackendType.Cpu)
    {
        // The families directly (on the CPU over simulated host devices: the serving contract admits --layer-split on GPU
        // backends only).
        using ModelBase model = family switch
        {
            "mistral3" => new Mistral3Model(path, backend, 1, null, stages),
            "qwen3vl" => new Qwen3VLModel(path, backend, 1, null, stages),
            "qwen35" => new Qwen35RuntimeModel(path, backend, 1, null, stages),
            "nemotron_h" => new NemotronRuntimeModel(path, backend, 1, null, stages),
            "gemma4" => new Gemma4RuntimeModel(path, backend, 1, null, stages),
            "lfm3" => new Lfm3RuntimeModel(path, backend, 1, null, stages),
            _ => throw new ArgumentOutOfRangeException(nameof(family)),
        };
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
