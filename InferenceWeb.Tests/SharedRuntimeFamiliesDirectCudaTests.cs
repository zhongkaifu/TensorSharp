// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Issue #376: the direct-CUDA policies the shared decoder runtime took for Qwen3.8-27B are the runtime's, not one family's.
// Every StandardDecoderModel on a CUDA allocator gets the CUDA executor, so Mistral 3 and Hunyuan Dense too start their primary
// cache at a fresh slot whatever MAX_CONTEXT says, and run an explicit q8_0 cache on the executor's flash kernels
// (USAGE: MAX_CONTEXT and --kv-cache-dtype name them). These tests pin that reach on the two families beside the Qwen line.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using TensorSharp;
using TensorSharp.Cuda;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class SharedRuntimeFamiliesDirectCudaTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Directory.CreateTempSubdirectory("shared-runtime-families").FullName;

    public SharedRuntimeFamiliesDirectCudaTests(ITestOutputHelper output) => _output = output;

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    /// <summary>A three-layer synthetic Mistral 3 or Hunyuan Dense with <paramref name="headDim"/>-wide heads (4,096-token
    /// window).</summary>
    private string Build(string family, int headDim) => family switch
    {
        "mistral3" => new DenseDecoderSyntheticModelBuilder { Architecture = "mistral3", KeyLength = headDim, LayerCount = 3 }
            .Write(Path.Combine(_dir, $"mistral3-{headDim}.gguf")),
        "hunyuan" => new DenseDecoderSyntheticModelBuilder
        {
            Architecture = "hunyuan-dense", PreTokenizer = "hunyuan-dense", IncludeMistralControlTokens = false, IncludeQkNorms = true,
            RopeBase = 10000f, KeyLength = headDim, LayerCount = 3,
        }.Write(Path.Combine(_dir, $"hunyuan-{headDim}.gguf")),
        _ => throw new ArgumentOutOfRangeException(nameof(family)),
    };

    private static StandardDecoderModel Load(string path, KvCacheDtype dtype)
    {
        using IDisposable kv = KvCacheDtypeConfig.ForCurrentFlowInTests(dtype);
        ModelBase model = ModelBase.Create(path, BackendType.Cuda);
        if (model is StandardDecoderModel standard) return standard;
        model.Dispose();
        throw new Xunit.Sdk.XunitException($"{path} loaded as {model.GetType().Name}, not on the shared decoder runtime");
    }

    /// <summary>At an explicit window (what MAX_CONTEXT=4096 answers: the whole window) the primary cache starts at a fresh
    /// slot's 1,024 cells on direct CUDA, as the Qwen line's does.</summary>
    [CudaTheory]
    [InlineData("mistral3")]
    [InlineData("hunyuan")]
    public void ThePrimaryStartsAtAFreshSlotWhateverTheWindow(string family)
    {
        string path = Build(family, DenseDecoderSyntheticModelBuilder.HeadDim);
        int? initial = StandardDecoderModel.InitialCacheLengthForTests.Value;
        StandardDecoderModel.InitialCacheLengthForTests.Value = 4096;
        StandardDecoderModel model;
        try
        {
            model = Load(path, KvCacheDtype.F16);
        }
        finally
        {
            StandardDecoderModel.InitialCacheLengthForTests.Value = initial;
        }
        using (model)
        {
            Assert.True(model.MaxContextLength > 1024, $"a {model.MaxContextLength}-token window");
            Assert.Equal(1024, model.PrimaryCacheCapacityForTests);
        }
    }

    /// <summary>A 300-token prompt and 40 decode steps; the rows' logits, the graph runs, the q8_0 attentions that took the
    /// generic path, why the model runs no graphs, its K/V dtype.</summary>
    private static (float[][] Rows, int GraphRuns, long Generic, string Refusal, KvCacheDtype Dtype) Run(string path, bool graphs)
    {
        using StandardDecoderModel model = Load(path, KvCacheDtype.Q8_0);
        model.GraphLoweringEnabled = graphs;
        int[] prompt = Enumerable.Range(0, 300).Select(i => 3 + (i * 37) % 200).ToArray();
        var rows = new List<float[]> { (float[])model.ForwardRefill(prompt).Clone() };
        for (int t = 0; t < 40; t++)
            rows.Add((float[])model.Forward(new[] { 5 + (t * 13) % 190 }).Clone());
        long generic = model.GraphExecutorForTests is CudaGraphExecutor cuda ? cuda.Q8GenericAttentions : -1;
        return (rows.ToArray(), model.GraphRuns, generic, model.GraphRefusalForTests, model.KvCacheDtype);
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }

    /// <summary>
    /// An explicit q8_0 cache runs on the whole-forward graphs and the executor's _q8 flash kernels (no attention took the
    /// generic path), within 1% relative L2 a row of the per-op path over the same q8_0 cache (0.04% and 0.11% the worst
    /// rows of Mistral 3 and Hunyuan Dense on an RTX PRO 4500; each against F16 graphs 0.5-0.7%). Where the loaded module has no
    /// sm_80 code (the committed compute_75 PTX, an sm_7x build) the gate skips it.
    /// </summary>
    [CudaTheory(TensorCoreModule = "tensorsharp_flash_prefill")]
    [InlineData("mistral3")]
    [InlineData("hunyuan")]
    public void AQ8CacheRunsOnTheFlashKernels(string family)
    {
        string path = Build(family, 128);
        var graph = Run(path, graphs: true);
        var perOp = Run(path, graphs: false);
        Assert.Equal(KvCacheDtype.Q8_0, graph.Dtype);
        Assert.Null(graph.Refusal);
        Assert.True(graph.GraphRuns > 0, "no step ran as a graph");
        Assert.Equal(0, graph.Generic);
        Assert.Equal(KvCacheDtype.Q8_0, perOp.Dtype);
        Assert.Equal(0, perOp.GraphRuns);
        double worst = graph.Rows.Zip(perOp.Rows, RelL2).Max();
        _output.WriteLine($"{family}: {graph.GraphRuns} graph runs; worst row relative L2 against the per-op path {worst:E3}");
        Assert.True(worst <= 0.01, $"{family}: q8_0 graphs against the per-op path, worst row relative L2 {worst:E3}");
    }
}
