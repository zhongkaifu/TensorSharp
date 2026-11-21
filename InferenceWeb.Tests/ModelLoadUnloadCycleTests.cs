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
/// Models of several families loaded and disposed in turn in one process, on the pinned GGML backend, give every load the
/// logits of their first: a dispose drops the process's device copies (keyed by host address and size) and frees its
/// weights, so nothing a disposed model leaves behind (a copy, a freed block handed out again) reaches the next one.
/// (A model kept loaded while another is disposed beside it is not covered: on ggml_cuda and ggml_vulkan the dispose's
/// process-wide cache clear frees the live model's device copies under its cached plans.)
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
}
