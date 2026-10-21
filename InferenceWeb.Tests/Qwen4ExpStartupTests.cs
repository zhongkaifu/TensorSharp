// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections;
using System.Reflection;
using TensorSharp;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Startup coverage for issue #256. The synthetic checkpoint exercises PLE, GDN,
/// and QSA through the required native token span; it does not test model quality.
/// </summary>
public sealed class Qwen4ExpStartupTests : IDisposable
{
    private static readonly int[] Prompt = Enumerable.Range(0, 24).Select(i => (i * 37 + 11) % 250).ToArray();
    private static readonly int[] DecodeTokens = [17, 203];
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-q4e-startup-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _environment = new();
    private readonly KvCacheDtype _previousDtype = KvCacheDtypeConfig.Current;
    private readonly bool _previousExplicit = KvCacheDtypeConfig.IsExplicitlySet;
    private readonly ITestOutputHelper _output;

    public Qwen4ExpStartupTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "128");
        _environment.Set("TS_PREFILL_WARMUP", "1");
        _environment.Set("TS_PREFILL_WARMUP_LEN", "24");
        _environment.ClearSpeculationVars();
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(KvCacheDtype.Q4_0)]
    [InlineData(KvCacheDtype.Q8_0)]
    [InlineData(KvCacheDtype.F16)]
    [InlineData(KvCacheDtype.F32)]
    public void RequestedKvDtype_WarmsUpAndRunsQsa_Cpu(KvCacheDtype requested)
        => AssertStartup(BackendType.GgmlCpu, requested);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(KvCacheDtype.Q4_0)]
    [InlineData(KvCacheDtype.Q8_0)]
    [InlineData(KvCacheDtype.F16)]
    [InlineData(KvCacheDtype.F32)]
    public void RequestedKvDtype_WarmsUpAndRunsQsa_Metal(KvCacheDtype requested)
        => AssertStartup(BackendType.GgmlMetal, requested);

    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(KvCacheDtype.Q4_0)]
    [InlineData(KvCacheDtype.Q8_0)]
    [InlineData(KvCacheDtype.F16)]
    [InlineData(KvCacheDtype.F32)]
    public void RequestedKvDtype_WarmsUpAndRunsQsa_Cuda(KvCacheDtype requested)
        => AssertStartup(BackendType.GgmlCuda, requested);

    [Qwen4ExpRetainedSplitFact]
    [Trait("Requires", "GgmlCuda")]
    public void RequestedQ4Cache_WarmsUpAndRunsQsa_AcrossTwoCudaDevices()
    {
        Assert.Equal(BackendType.GgmlCuda, TestGates.PinnedGgmlBackend);
        _environment.Set("TS_Q4E_LAYER_SPLIT", "3,5");
        AssertStartup(BackendType.GgmlCuda, KvCacheDtype.Q4_0, layerSplitDegree: 2);
    }

    private void AssertStartup(BackendType backend, KvCacheDtype requested, int layerSplitDegree = 1)
    {
        KvCacheDtypeConfig.Set(requested);
        using var stderr = new StringWriter();
        TextWriter previousError = Console.Error;
        Qwen4ExpModel model;
        try
        {
            Console.SetError(stderr);
            model = Load(backend, layerSplitDegree);
        }
        finally { Console.SetError(previousError); }
        using (model)
        {
            KvCacheDtype effective = requested.IsBlockQuantized() ? KvCacheDtype.F16 : requested;
            Assert.Equal(effective, model.KvCacheDtype);
            Assert.Equal(effective, KvCacheDtypeConfig.Current);
            for (int layer = 0; layer < Qwen4ExpSyntheticModelBuilder.Layers; ++layer)
                Assert.Equal(layerSplitDegree == 1 || layer < 3 ? 0 : 1, model.DeviceForLayer(layer));
            if (requested.IsBlockQuantized())
            {
                Assert.Contains(requested.ToShortString(), stderr.ToString());
                Assert.Contains("using f16 instead", stderr.ToString());
            }
            AssertCacheStorage(model, effective);

            using var warmupOutput = new StringWriter();
            TextWriter previousOutput = Console.Out;
            try
            {
                Console.SetOut(warmupOutput);
                model.WarmUpKernels();
            }
            finally { Console.SetOut(previousOutput); }
            // WarmUpKernels catches some prefill exceptions: successful return alone
            // would not prove that the complete startup path ran.
            Assert.Contains("Prefill warmup (24 tokens): completed", warmupOutput.ToString());
            Assert.DoesNotContain("warmup skipped", warmupOutput.ToString());
            Assert.Equal(0, model.CacheSeqLen);
            Assert.Equal(0, Field<int>(model, "_qsaPositionCount"));

            float[][] first = RunSequence(model);
            model.ResetKVCache();
            Assert.Equal(0, model.CacheSeqLen);
            Assert.Equal(0, Field<int>(model, "_qsaPositionCount"));
            float[][] second = RunSequence(model);
            for (int step = 0; step < first.Length; ++step)
                Assert.Equal(first[step], second[step]);
            AssertCacheStorage(model, effective);
            _output.WriteLine($"{backend}, {layerSplitDegree} device(s): {requested.ToShortString()} requested, {effective.ToShortString()} backing; decode/prefill warmup, QSA prefill, decode and reset replay passed.");
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void UnsupportedAttentionCache_ReportsLayerAndStorageBeforeNativeExecution()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        using var model = Load(BackendType.GgmlCpu);
        const int attentionLayer = 3;
        Tensor[] caches = Field<Tensor[]>(model, "_kCache");
        Tensor original = caches[attentionLayer];
        IAllocator allocator = Field<IAllocator>(model, "_allocator", typeof(ModelBase));
        using var unsupported = new Tensor(allocator, DType.Q4_0,
            Qwen4ExpSyntheticModelBuilder.KvHeads, Field<int>(model, "_kvCacheCapacity"),
            Qwen4ExpSyntheticModelBuilder.HeadDim);
        caches[attentionLayer] = unsupported;
        try
        {
            var error = Assert.Throws<InvalidOperationException>(() => model.WarmUpKernels());
            Assert.Contains("required token-span path declined", error.Message);
            Assert.Contains("attention descriptors for layer 3", error.Message);
            Assert.Contains("K=Q4_0", error.Message);
            Assert.Contains("f16", error.Message);
            Assert.Equal(0, model.CacheSeqLen);
            _output.WriteLine(error.Message);
        }
        finally { caches[attentionLayer] = original; }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeSpanRejection_PreservesNativeReasonAndForwardContext()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        using var model = Load(BackendType.GgmlCpu);
        // The native entry point validates this before building or executing a
        // graph, safely reproducing the otherwise-generic startup refusal.
        Set(model, "_seqSlotBase", int.MaxValue);
        var error = Assert.Throws<InvalidOperationException>(() => model.WarmUpKernels());
        Assert.Contains("layers [0, 8)", error.Message);
        Assert.Contains("device 0", error.Message);
        Assert.Contains("position 0", error.Message);
        Assert.Contains("tokens 1", error.Message);
        Assert.Contains("bad cache slot", error.Message);
        Assert.Equal(0, model.CacheSeqLen);
        _output.WriteLine(error.Message);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void MissingAttentionProjection_ReportsDescriptorFailure()
    {
        KvCacheDtypeConfig.Set(KvCacheDtype.F16);
        using var model = Load(BackendType.GgmlCpu);
        IDictionary weights = Field<IDictionary>(model, "_quantWeights", typeof(ModelBase));
        const string name = "blk.3.attn_q.weight";
        object weight = weights[name]!;
        Assert.NotNull(weight);
        weights.Remove(name);
        try
        {
            var error = Assert.Throws<InvalidOperationException>(() => model.WarmUpKernels());
            Assert.Contains("required token-span path declined", error.Message);
            Assert.Contains("attention descriptors for layer 3", error.Message);
            Assert.Equal(0, model.CacheSeqLen);
            _output.WriteLine(error.Message);
        }
        finally { weights.Add(name, weight); }
    }

    private Qwen4ExpModel Load(BackendType backend, int layerSplitDegree = 1)
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "fixture.gguf"));
        return Assert.IsType<Qwen4ExpModel>(ModelBase.Create(path, backend, layerSplitDegree: layerSplitDegree));
    }

    private static float[][] RunSequence(Qwen4ExpModel model)
    {
        // QSA top_k=16, ratio=4 has a nineteen-cell width; this prefill crosses it.
        var rows = new List<float[]> { (float[])model.ForwardRefill(Prompt).Clone() };
        AssertSpan(model);
        foreach (int token in DecodeTokens)
        {
            rows.Add((float[])model.Forward([token]).Clone());
            AssertSpan(model);
        }
        Assert.Equal(Prompt.Length + DecodeTokens.Length, model.CacheSeqLen);
        Assert.Equal(model.CacheSeqLen, Field<int>(model, "_qsaPositionCount"));
        foreach (float[] row in rows)
        {
            Assert.Equal(Qwen4ExpSyntheticModelBuilder.Vocab, row.Length);
            Assert.All(row, value => Assert.True(float.IsFinite(value)));
            Assert.True(row.Sum(value => (double)value * value) > 1e-6, "Degenerate logits.");
        }
        return rows.ToArray();
    }

    private static void AssertSpan(Qwen4ExpModel model)
    {
        Assert.False(Field<bool>(model, "_tokenGraphUnsupported"));
        Assert.True(Field<bool>(model, "_spanLogitsValid"));
        Assert.NotNull(Field<object>(model, "_pleArgs"));
        Qwen4ExpAttnArgs[] attention = Field<Qwen4ExpAttnArgs[]>(model, "_attnArgs");
        Qwen4ExpQsaArgs[] indexer = Field<Qwen4ExpQsaArgs[]>(model, "_qsaArgs");
        Tensor[] keys = Field<Tensor[]>(model, "_kCache");
        Tensor[] indexerKeys = Field<Tensor[]>(model, "_idxKCache");
        for (int layer = 0; layer < Qwen4ExpSyntheticModelBuilder.Layers; ++layer)
        {
            if (Qwen4ExpSyntheticModelBuilder.IsRecurrent(layer)) continue;
            Assert.Equal(model.KvCacheDtype.GgmlType(), attention[layer].KvType);
            Assert.Equal(keys[layer].Storage.ByteLength, attention[layer].KvBytes);
            Assert.Equal(model.KvCacheDtype.GgmlType(), indexer[layer].CacheType);
            Assert.Equal(indexerKeys[layer].Storage.ByteLength, indexer[layer].CacheBytes);
        }
    }

    private static void AssertCacheStorage(Qwen4ExpModel model, KvCacheDtype dtype)
    {
        Tensor[] keys = Field<Tensor[]>(model, "_kCache");
        Tensor[] values = Field<Tensor[]>(model, "_vCache");
        Tensor[] indexer = Field<Tensor[]>(model, "_idxKCache");
        int capacity = Field<int>(model, "_kvCacheCapacity");
        for (int layer = 0; layer < Qwen4ExpSyntheticModelBuilder.Layers; ++layer)
        {
            if (Qwen4ExpSyntheticModelBuilder.IsRecurrent(layer))
            {
                Assert.Null(keys[layer]);
                Assert.Null(values[layer]);
                Assert.Null(indexer[layer]);
                continue;
            }
            Check(keys[layer], Qwen4ExpSyntheticModelBuilder.KvHeads, Qwen4ExpSyntheticModelBuilder.HeadDim);
            Check(values[layer], Qwen4ExpSyntheticModelBuilder.KvHeads, Qwen4ExpSyntheticModelBuilder.HeadDim);
            Check(indexer[layer], 1, Qwen4ExpSyntheticModelBuilder.IndexerDim);
        }

        void Check(Tensor tensor, int heads, int headDim)
        {
            Assert.Equal(dtype.ToDType(), tensor.ElementType);
            Assert.Equal(new long[] { heads, capacity, headDim }, tensor.Sizes);
            Assert.Equal(dtype.ByteLengthFor((long)heads * capacity * headDim), tensor.Storage.ByteLength);
        }
    }

    private static T Field<T>(object model, string name, Type? owner = null)
        => (T)(owner ?? typeof(Qwen4ExpModel)).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private static void Set(object model, string name, object value)
        => typeof(Qwen4ExpModel).GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.SetValue(model, value);

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_previousDtype, _previousExplicit);
        _environment.Dispose();
        try { Directory.Delete(_directory, recursive: true); } catch (IOException) { }
    }
}
