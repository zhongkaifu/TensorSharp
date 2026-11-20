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
using System.Security.Cryptography;
using TensorSharp;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A before/after instrument for changes to the shared decoder runtime that must leave every family's arithmetic alone
/// (the Muse-Glimmer migration's runtime hooks, docs/design/muse-glimmer-unified.md stage S1): each runtime family's
/// synthetic fixture runs a fixed script on the managed backend, the pinned GGML backend and MLX, as whole-step programs and
/// on the per-op path, and the SHA-256 of every forward's logits is printed. Two builds that print the same lines computed
/// the same bytes. A Bench test: it asserts nothing about the values, only that the script runs (or fails the same way).
/// </summary>
[Trait("Category", "Bench")]
public sealed class RuntimeFamilyDigestTests(ITestOutputHelper output) : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-family-digest-" + Guid.NewGuid().ToString("N"));

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture(string family)
    {
        Directory.CreateDirectory(_dir);
        string path = Path.Combine(_dir, family + ".gguf");
        return family switch
        {
            "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = 6 }.Write(path),
            "hunyuan-dense" => new DenseDecoderSyntheticModelBuilder
            {
                Architecture = "hunyuan-dense", PreTokenizer = "hunyuan-dense", IncludeMistralControlTokens = false,
                IncludeQkNorms = true, RopeBase = 10000f, LayerCount = 6,
            }.Write(path),
            "qwen35-dense" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(path),
            "qwen35-moe" => new Qwen35MoeSyntheticModelBuilder { Layers = 8 }.Write(path),
            "lfm3-dense" => new Lfm3SyntheticModelBuilder { UntiedOutput = true }.Write(path),
            "lfm3-tied" => new Lfm3SyntheticModelBuilder().Write(path),
            "qwen4exp" => Qwen4ExpSyntheticModelBuilder.Write(path, indexerTopK: 64),
            _ => throw new ArgumentOutOfRangeException(nameof(family)),
        };
    }

    /// <summary>Prefill, decode, reset, a short prefill and decode: what every family's single-sequence path runs.</summary>
    private static List<float[]> Script(ModelBase model)
    {
        var rows = new List<float[]>();
        int[] prompt = Enumerable.Range(0, 40).Select(i => 40 + (i * 11) % 150).ToArray();
        rows.Add((float[])model.ForwardRefill(prompt).Clone());
        foreach (int token in new[] { 71, 72, 73, 74, 75, 76 })
            rows.Add((float[])model.Forward(new[] { token }).Clone());
        model.ResetKVCache();
        rows.Add((float[])model.ForwardRefill(new[] { 33, 34, 35, 36, 37, 38, 39, 40, 41 }).Clone());
        rows.Add((float[])model.Forward(new[] { 42 }).Clone());
        rows.Add((float[])model.Forward(new[] { 43 }).Clone());
        return rows;
    }

    private static string Sha(float[] values)
    {
        byte[] bytes = new byte[values.Length * sizeof(float)];
        Buffer.BlockCopy(values, 0, bytes, 0, bytes.Length);
        return Convert.ToHexStringLower(SHA256.HashData(bytes))[..16];
    }

    [Theory]
    [InlineData("mistral3")]
    [InlineData("hunyuan-dense")]
    [InlineData("qwen35-dense")]
    [InlineData("qwen35-moe")]
    [InlineData("lfm3-dense")]
    [InlineData("lfm3-tied")]
    [InlineData("qwen4exp")]
    public void PrintLogitDigests(string family)
    {
        string path = Fixture(family);
        var backends = new List<BackendType> { BackendType.Cpu, TestGates.PinnedGgmlBackend };
        if (TestGates.MlxSkip == null) backends.Add(BackendType.Mlx);
        int ran = 0;
        foreach (BackendType backend in backends)
            foreach (bool graphs in new[] { true, false })
            {
                string label = $"digest {family} {MuseGlimmerGoldens.BackendName(backend)} {(graphs ? "graph" : "per-op")}";
                try
                {
                    using var _ = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.F16);
                    using ModelBase model = ModelBase.Create(path, backend);
                    var decoder = Assert.IsAssignableFrom<StandardDecoderModel>(model);
                    decoder.GraphLoweringEnabled = graphs;
                    List<float[]> rows = Script(model);
                    output.WriteLine($"{label}: graphRuns={(decoder.GraphRuns > 0 ? "yes" : "no")} " + string.Join(" ", rows.Select(Sha)));
                    ran++;
                }
                catch (Exception ex) when (ex is NotSupportedException or InvalidOperationException or ArgumentException)
                {
                    output.WriteLine($"{label}: {ex.GetType().Name}: {ex.Message.Split('\n')[0]}");
                }
            }
        Assert.True(ran > 0, $"{family}: no backend ran the script");
    }
}
