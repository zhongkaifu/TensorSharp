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
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Batched steps whose short convolutions take their products over the step's rows and whose per-sequence results are joined
/// in a balanced tree (StandardDecoderModel.Joins.cs) give the bits of the steps they replaced (one product pair a sequence,
/// a left fold of concatenations: <c>LegacyBatchedJoinForTests</c>): 2, 3 and 16 sequences through their prompts, decode
/// steps and a step where a new sequence's prompt chunk runs beside the others' decode rows, on LFM3 (short convolutions and
/// attention), Qwen 3.5 (gated delta nets and attention) and Mistral3 (attention) fixtures, every row's logits bit for bit.
/// </summary>
public sealed class BatchedJoinBitwiseTests : IDisposable
{
    private readonly string _dir = Directory.CreateTempSubdirectory("batched-joins").FullName;
    private readonly ITestOutputHelper _output;

    public BatchedJoinBitwiseTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Fixture(string family) => family switch
    {
        "lfm3" => new Lfm3SyntheticModelBuilder { UntiedOutput = true }.Write(Path.Combine(_dir, "lfm3.gguf")),
        "lfm3-128" => new Lfm3SyntheticModelBuilder { UntiedOutput = true, HeadDimension = 128 }.Write(Path.Combine(_dir, "lfm3-128.gguf")),
        "qwen35" => new Qwen35DenseSyntheticModelBuilder { Layers = 4, UntiedOutput = true }.Write(Path.Combine(_dir, "q35.gguf")),
        "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", LayerCount = 2 }.Write(Path.Combine(_dir, "mistral3.gguf")),
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    private static int Token(int s, int t) => 40 + (t * 31 + s * 17) % 150;

    /// <summary>Every row's logits of the schedule: n prompts alone, four decode steps of the n, a step where a new sequence's
    /// prompt runs beside their decode rows, then three decode steps of all n + 1.</summary>
    private static List<float[]> Run(string path, BackendType backend, int n, bool legacy)
    {
        using ModelBase model = ModelBase.Create(path, backend);
        var runtime = Assert.IsAssignableFrom<StandardDecoderModel>(model);
        runtime.LegacyBatchedJoinForTests = legacy;
        var unified = Assert.IsAssignableFrom<IUnifiedBatchModel>(model);
        var ids = Enumerable.Range(0, n + 1).Select(s => $"s{s}").ToArray();
        var starts = new int[n + 1];
        var rows = new List<float[]>();
        int[] Prompt(int s) => Enumerable.Range(0, 9 + 3 * s).Select(i => 30 + (i * 13 + s * 7) % 160).ToArray();
        for (int s = 0; s < n; s++)
        {
            unified.EnsureSequenceSlot(ids[s], 256);
            var one = new float[1][];
            Assert.True(unified.TryForwardUnified(new[] { ids[s] }, new[] { Prompt(s) }, new[] { 0 }, one));
            rows.Add((float[])one[0].Clone());
            starts[s] = Prompt(s).Length;
        }
        unified.EnsureSequenceSlot(ids[n], 256);
        for (int t = 0; t < 8; t++)
        {
            int count = t < 4 ? n : n + 1;
            var tokens = new int[count][];
            var at = new int[count];
            for (int s = 0; s < count; s++)
            {
                tokens[s] = s == n && t == 4 ? Prompt(n) : new[] { Token(s, t) };
                at[s] = starts[s];
            }
            var logits = new float[count][];
            Assert.True(unified.TryForwardUnified(ids.Take(count).ToArray(), tokens, at, logits));
            for (int s = 0; s < count; s++)
            {
                rows.Add((float[])logits[s].Clone());
                starts[s] += tokens[s].Length;
            }
        }
        foreach (string id in ids) runtime.OnSequenceReleased(id);
        return rows;
    }

    private void AssertSameBits(string family, BackendType backend, int n)
    {
        string path = Fixture(family);
        List<float[]> legacy = Run(path, backend, n, legacy: true);
        List<float[]> joined = Run(path, backend, n, legacy: false);
        Assert.Equal(legacy.Count, joined.Count);
        for (int i = 0; i < legacy.Count; i++)
            Assert.True(legacy[i].AsSpan().SequenceEqual(joined[i]), $"{family} on {backend}, {n} sequences: row {i} differs");
        _output.WriteLine($"{family} on {backend}, {n} sequences: {legacy.Count} rows bit for bit");
    }

    [Theory]
    [InlineData("lfm3", 2)]
    [InlineData("lfm3", 3)]
    [InlineData("lfm3", 16)]
    [InlineData("qwen35", 3)]
    [InlineData("mistral3", 3)]
    public void ManagedBatchedStepsKeepTheirBits(string family, int n) => AssertSameBits(family, BackendType.Cpu, n);

    [Theory]
    [InlineData("lfm3", 2)]
    [InlineData("lfm3", 3)]
    [InlineData("lfm3", 16)]
    [InlineData("qwen35", 3)]
    [InlineData("qwen35", 16)]
    [InlineData("mistral3", 16)]
    public void GgmlBatchedStepsKeepTheirBits(string family, int n) => AssertSameBits(family, TestGates.PinnedGgmlBackend, n);

    [CudaTheory]
    [InlineData("lfm3-128", 2)]
    [InlineData("lfm3-128", 3)]
    [InlineData("lfm3-128", 16)]
    [InlineData("lfm3-128", 24)]
    [InlineData("qwen35", 16)]
    public void DirectCudaBatchedStepsKeepTheirBits(string family, int n) => AssertSameBits(family, BackendType.Cuda, n);
}
