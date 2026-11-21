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
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Models of several families (lfm3, qwen35, mistral3, gemma4, nemotron, qwen4exp, museglimmer) loaded and disposed in one
/// process, on the pinned GGML backend. Loaded and disposed in turn,
/// every load gives the logits of its first: nothing a disposed model leaves behind (a device copy, a freed block handed
/// out again) reaches the next one. A model kept loaded while others are loaded, run and disposed beside it carries on
/// exactly: each next decode step's logits are an uninterrupted run's, bit for bit, so its device copies, the plans bound
/// to them and its KV cache survived every dispose (a dispose had cleared the process's device copies: on ggml_cuda the
/// kept model's next step then crashed the process in TSGgml_GraphRun).
/// </summary>
public sealed class ModelLoadUnloadCycleTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("load-unload").FullName;
    private readonly ITestOutputHelper _output;

    public ModelLoadUnloadCycleTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly int[] Prompt = Enumerable.Range(0, 11).Select(i => 30 + (i * 13) % 160).ToArray();
    private static readonly int[] Decode = { 41, 77, 102 };

    private Dictionary<string, string> Fixtures() => new()
    {
        ["lfm3"] = new Lfm3SyntheticModelBuilder { UntiedOutput = true }.Write(Path.Combine(_dir, "lfm3.gguf")),
        ["qwen35"] = new Qwen35DenseSyntheticModelBuilder { Layers = 4, UntiedOutput = true }.Write(Path.Combine(_dir, "q35.gguf")),
        ["mistral3"] = new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = 2 }.Write(Path.Combine(_dir, "mistral3.gguf")),
        ["gemma4"] = new Gemma4DenseSyntheticModelBuilder().Write(Path.Combine(_dir, "gemma4.gguf")),
        ["nemotron"] = new NemotronSyntheticModelBuilder().Write(Path.Combine(_dir, "nemotron.gguf")),
        ["qwen4exp"] = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4e.gguf"), indexerTopK: 64, qsa: false),
        ["museglimmer"] = new MuseGlimmerSyntheticModelBuilder().Write(Path.Combine(_dir, "museglimmer.gguf")),
    };

    /// <summary>The prompt's logits and three decode steps' of a fresh cache.</summary>
    private static List<float[]> Rows(ModelBase model)
    {
        model.ResetKVCache();
        var rows = new List<float[]> { (float[])model.ForwardPrompt(Prompt).Clone() };
        foreach (int t in Decode)
            rows.Add((float[])model.Forward(new[] { t }).Clone());
        return rows;
    }

    /// <summary>A fresh cache's prompt, then the decode steps of <paramref name="tokens"/>; after each,
    /// <paramref name="between"/> (null to run uninterrupted).</summary>
    private static List<float[]> DecodeSteps(ModelBase model, int[] tokens, Action<int> between)
    {
        model.ResetKVCache();
        var rows = new List<float[]> { (float[])model.ForwardPrompt(Prompt).Clone() };
        for (int i = 0; i < tokens.Length; i++)
        {
            between?.Invoke(i);
            rows.Add((float[])model.Forward(new[] { tokens[i] }).Clone());
        }
        return rows;
    }

    private static List<float[]> LoadRunDispose(string path, BackendType backend)
    {
        using ModelBase model = ModelBase.Create(path, backend);
        return Rows(model);
    }

    private static void AssertSame(List<float[]> want, List<float[]> got, string what)
    {
        Assert.Equal(want.Count, got.Count);
        for (int r = 0; r < want.Count; r++)
            Assert.True(want[r].AsSpan().SequenceEqual(got[r]), $"{what}: row {r} differs from the first load's");
    }

    [Fact]
    public void FamiliesLoadedAndDisposedInTurnGiveTheLogitsOfTheirFirstLoad()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        Dictionary<string, string> paths = Fixtures();
        var first = paths.ToDictionary(p => p.Key, p => LoadRunDispose(p.Value, backend));
        // Two more rounds, each in another order, so every family follows several others' disposes.
        string[] order = paths.Keys.ToArray();
        for (int round = 1; round <= 2; round++)
        {
            string[] turn = round == 1 ? order.Reverse().ToArray() : order.Skip(order.Length / 2).Concat(order.Take(order.Length / 2)).ToArray();
            foreach (string family in turn)
                AssertSame(first[family], LoadRunDispose(paths[family], backend), $"{family}, round {round} on {backend}");
        }
        _output.WriteLine($"{backend}: {order.Length} families, three loads each, every load bit for bit its first");
    }

    [Fact]
    public void AModelKeptLoadedCarriesOnExactlyWhileTheOthersComeAndGo()
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        Dictionary<string, string> paths = Fixtures();
        string[] others = paths.Keys.Where(f => f != "lfm3").ToArray();
        int[] tokens = Enumerable.Range(0, others.Length + 1).Select(i => 41 + 17 * i).ToArray();
        List<float[]> uninterrupted;
        using (ModelBase alone = ModelBase.Create(paths["lfm3"], backend))
            uninterrupted = DecodeSteps(alone, tokens, null);
        using ModelBase kept = ModelBase.Create(paths["lfm3"], backend);
        // Before decode step i (i >= 1), another family is loaded, run and disposed beside the kept model, mid-sequence.
        List<float[]> carried = DecodeSteps(kept, tokens, i =>
        {
            if (i > 0) LoadRunDispose(paths[others[i - 1]], backend);
        });
        AssertSame(uninterrupted, carried, $"lfm3 kept loaded on {backend}");
        _output.WriteLine($"{backend}: lfm3 kept loaded through {others.Length} other loads and disposes mid-sequence, " +
            "every step bit for bit an uninterrupted run's");
    }

    /// <summary>The pairs whose kept model crashed the process on ggml_cuda (SIGSEGV in TSGgml_GraphRun) when the other was
    /// disposed beside it, and one that did not.</summary>
    [Theory]
    [InlineData("lfm3", "qwen35")]
    [InlineData("lfm3", "mistral3")]
    [InlineData("qwen35", "mistral3")]
    [InlineData("mistral3", "lfm3")]
    public void ALiveModelCarriesOnExactlyAfterAnotherIsDisposedBesideIt(string keptFamily, string otherFamily)
    {
        BackendType backend = TestGates.PinnedGgmlBackend;
        Dictionary<string, string> paths = Fixtures();
        int[] tokens = { 41, 77, 102 };
        List<float[]> uninterrupted;
        using (ModelBase alone = ModelBase.Create(paths[keptFamily], backend))
            uninterrupted = DecodeSteps(alone, tokens, null);
        using ModelBase kept = ModelBase.Create(paths[keptFamily], backend);
        List<float[]> carried = DecodeSteps(kept, tokens, i =>
        {
            if (i == 1) LoadRunDispose(paths[otherFamily], backend);
        });
        AssertSame(uninterrupted, carried, $"{keptFamily} with {otherFamily} disposed beside it on {backend}");
    }
}
