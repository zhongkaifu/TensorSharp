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
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// A q8_0 K/V cache on direct CUDA (<c>--kv-cache-dtype q8_0</c>) keeps every attention of the families whose routes are the plain
/// attention and row writes on the CUDA executor's flash kernels: prompt chunks, short prompt tails (the window rows) and decode
/// rows run as whole-forward graphs over the q8_0 rows, and none of them takes the generic or tiled path. That path widens the
/// window to F32 and warns once (<see cref="CudaGraphExecutor.Q8GenericAttentions"/> counts it): the output stays right, so before
/// this test a regression there showed only as the stderr line and as speed. Verify windows over a q8_0 cache are
/// <see cref="Qwen35RuntimeTests.EngineSpeculationOnDirectCudaMatchesPlainOverAQ8Cache"/>'s, which asserts the same.
/// </summary>
public sealed class CudaQ8KvAttentionRoutesTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-q8-routes-" + Guid.NewGuid().ToString("N"));

    public CudaQ8KvAttentionRoutesTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
    }

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>The shared-decoder families whose attention is the plain attention alone, at the head widths the q8_0 flash kernels
    /// take: Qwen 3.5 dense (group 4) and MoE (group 8) at 256, Mistral 3 and Hunyuan dense at 128.</summary>
    private string Write(string family) => family switch
    {
        "qwen35" => new Qwen35DenseSyntheticModelBuilder { Layers = 8, UntiedOutput = true }.Write(Path.Combine(_dir, "q35.gguf")),
        "qwen35moe" => new Qwen35MoeSyntheticModelBuilder { Layers = 8, UniformWeightType = SyntheticGguf.GgmlType.BF16 }
            .Write(Path.Combine(_dir, "q35moe.gguf")),
        "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", KeyLength = 128, LayerCount = 3 }
            .Write(Path.Combine(_dir, "mistral3.gguf")),
        "hunyuan" => new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "hunyuan-dense", PreTokenizer = "hunyuan-dense", IncludeMistralControlTokens = false, IncludeQkNorms = true,
            RopeBase = 10000f, KeyLength = 128, LayerCount = 3,
        }.Write(Path.Combine(_dir, "hunyuan.gguf")),
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    [CudaTheory]
    [InlineData("qwen35")]
    [InlineData("qwen35moe")]
    [InlineData("mistral3")]
    [InlineData("hunyuan")]
    public void EveryAttentionOverAQ8CacheRunsOnTheFlashKernels(string family)
    {
        string path = Write(family);
        using IDisposable kv = KvCacheDtypeConfig.ForCurrentFlowInTests(KvCacheDtype.Q8_0);
        using var model = (StandardDecoderModel)ModelBase.Create(path, BackendType.Cuda);
        Q8Watch watch = Watch(model);
        // 300 prompt rows (the prompt kernel), a 4-row prompt tail (the window rows), then decode rows (recorded after their first).
        model.ForwardPrompt(Enumerable.Range(0, 300).Select(i => 3 + (i * 37) % 200).ToArray());
        model.ForwardPrompt(new[] { 7, 11, 13, 17 });
        for (int t = 0; t < 24; t++)
            model.Forward(new[] { 5 + (t * 13) % 190 });
        AssertFlashKernelsOnly(watch, family, _output);
    }

    /// <summary>What <see cref="AssertFlashKernelsOnly"/> reads of a model, taken where it is created: its graph executor (a disposed
    /// model forgets its executors, and asking it again would create a fresh one with nothing counted) and its graph refusal.</summary>
    internal sealed record Q8Watch(StandardDecoderModel Model, CudaGraphExecutor Executor, string Refusal);

    internal static Q8Watch Watch(StandardDecoderModel model)
        => new(model, model.GraphExecutorForTests as CudaGraphExecutor, model.GraphRefusalForTests);

    /// <summary>The model held a q8_0 cache, ran whole-forward graphs, and no attention over that cache took the CUDA executor's
    /// generic or tiled path. Readable after the model is disposed.</summary>
    internal static void AssertFlashKernelsOnly(Q8Watch watch, string label, ITestOutputHelper output = null)
    {
        StandardDecoderModel model = watch.Model;
        CudaGraphExecutor executor = watch.Executor;
        output?.WriteLine($"{label}: cache {model.KvCacheDtype.ToShortString()}, graph refusal {watch.Refusal ?? "none"}, " +
            $"graph runs {model.GraphRuns}, q8_0 generic attentions {executor?.Q8GenericAttentions.ToString() ?? "n/a"}");
        Assert.Equal(KvCacheDtype.Q8_0, model.KvCacheDtype);
        Assert.Null(watch.Refusal);
        Assert.NotNull(executor);
        Assert.True(model.GraphRuns > 0, $"{label}: no step ran as a graph");
        Assert.True(executor.Q8GenericAttentions == 0,
            $"{label}: {executor.Q8GenericAttentions} attentions over the q8_0 cache took the generic or tiled path");
    }
}
