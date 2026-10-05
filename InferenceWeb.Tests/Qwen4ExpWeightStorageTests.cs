// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Collections;
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using GgmlType = InferenceWeb.Tests.SyntheticGguf.GgmlType;

namespace InferenceWeb.Tests;

/// <summary>Converters may change coefficient storage without changing tensor roles.</summary>
public sealed class Qwen4ExpWeightStorageTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-q4e-weight-storage-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _environment = new();

    public Qwen4ExpWeightStorageTests()
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "128");
        _environment.Set("TS_PREFILL_WARMUP_LEN", "8");
        _environment.ClearSpeculationVars();
    }

    [Theory]
    [InlineData(BackendType.GgmlCpu)]
    [InlineData(BackendType.GgmlMetal)]
    [InlineData(BackendType.GgmlCuda)]
    [InlineData(BackendType.Cuda)]
    public void CoefficientRoles_AreIndependentOfStorageTypeAndMatrixShape(BackendType backend)
    {
        // The policy can be exercised without constructing a native backend or loading weights.
        var model = (Qwen4ExpModel)RuntimeHelpers.GetUninitializedObject(typeof(Qwen4ExpModel));
        GC.SuppressFinalize(model);
        Qwen4ExpMtpMathTests.Set(typeof(ModelBase), model, "<ExecutionPlan>k__BackingField", new BackendExecutionPlan(backend));
        string[] coefficients =
        [
            "blk.1.ple_conv1d.weight", "blk.0.ssm_conv1d.weight",
            "output_hc_norm.weight", "blk.0.hc_attn_norm.weight", "blk.0.hc_ffn_norm.weight",
            "blk.3.attn_q_norm.weight", "blk.3.attn_k_norm.weight",
            "blk.3.indexer.q_norm.weight", "blk.3.indexer.k_norm.weight",
            "blk.1.ple_norm_key.weight", "blk.1.ple_norm_query.weight", "blk.1.ple_norm_conv.weight",
            "blk.0.ssm_norm.weight", "blk.0.ssm_dt.bias", "blk.0.ssm_a", "blk.0.ffn_gate_inp_shexp.weight",
        ];
        string[] projections =
        [
            "blk.0.hc_attn_inject.weight", "blk.0.hc_ffn_inject.weight", "blk.0.ssm_alpha.weight",
            "blk.0.ssm_beta.weight", "blk.1.ple_key.weight", "blk.1.ple_value.weight",
            "blk.3.indexer.q_proj.weight", "blk.3.indexer.k_proj.weight", "token_embd.weight", "output.weight",
        ];
        foreach (GgmlTensorType type in new[] { GgmlTensorType.F16, GgmlTensorType.BF16, GgmlTensorType.Q8_0 })
        {
            bool Store(string name) => (bool)Qwen4ExpMtpMathTests.Invoke(model, "IsQuantizedLinearWeight",
                [new GgufTensorInfo { Name = name, Shape = [256, 4], Type = type }])!;
            foreach (string name in coefficients)
                Assert.False(Store(name), $"{backend} retained {name} as {type} matrix bytes.");
            foreach (string name in projections)
                Assert.True(Store(name), $"{backend} unnecessarily expanded {name} from {type}.");
        }
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(GgmlTensorType.F16, GgmlTensorType.F32, null)]
    [InlineData(GgmlTensorType.F16, GgmlTensorType.BF16, GgmlTensorType.F16)]
    [InlineData(GgmlTensorType.BF16, GgmlTensorType.F16, GgmlTensorType.BF16)]
    [InlineData(GgmlTensorType.Q8_0, GgmlTensorType.Q8_0, GgmlTensorType.Q8_0)]
    public void ConvertedCoefficients_WarmUpAndRunFusedQsaSpan_Cpu(GgmlTensorType ple, GgmlTensorType ssm, GgmlTensorType? norms)
        => RunConvertedCoefficients(BackendType.GgmlCpu, (GgmlType)ple, (GgmlType)ssm, (GgmlType?)norms);

    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(GgmlTensorType.F16, GgmlTensorType.F32, null)]
    [InlineData(GgmlTensorType.F16, GgmlTensorType.BF16, GgmlTensorType.F16)]
    [InlineData(GgmlTensorType.BF16, GgmlTensorType.F16, GgmlTensorType.BF16)]
    [InlineData(GgmlTensorType.Q8_0, GgmlTensorType.Q8_0, GgmlTensorType.Q8_0)]
    public void ConvertedCoefficients_WarmUpAndRunFusedQsaSpan_Metal(GgmlTensorType ple, GgmlTensorType ssm, GgmlTensorType? norms)
        => RunConvertedCoefficients(BackendType.GgmlMetal, (GgmlType)ple, (GgmlType)ssm, (GgmlType?)norms);

    private void RunConvertedCoefficients(BackendType backend, GgmlType ple, GgmlType ssm, GgmlType? norms)
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_directory, "fixture.gguf"),
            pleConvType: ple, ssmConvType: ssm, matrixNormType: norms);
        using var model = Assert.IsType<Qwen4ExpModel>(ModelBase.Create(path, backend));
        var weights = Field<Dictionary<string, Tensor>>(typeof(ModelBase), model, "_weights");
        var compressed = Field<IDictionary>(typeof(ModelBase), model, "_quantWeights");
        using var gguf = new GgufFile(path);
        foreach (GgufTensorInfo info in gguf.Tensors.Values.Where(info =>
            info.Name.EndsWith("conv1d.weight", StringComparison.Ordinal)
            || info.Name.Contains("norm", StringComparison.Ordinal)))
        {
            Assert.False(compressed.Contains(info.Name));
            Assert.True(weights.TryGetValue(info.Name, out Tensor? weight), $"Missing coefficient {info.Name}.");
            Assert.Equal(DType.Float32, weight!.ElementType);
            Assert.Equal(DecodeCoefficients(info, gguf.ReadTensorData(info)), weight.GetElementsAsFloat((int)info.NumElements));
        }
        Assert.True(compressed.Contains("blk.1.ple_key.weight"));
        Assert.True(compressed.Contains("blk.0.attn_qkv.weight"));

        // Startup first warms a single token. The user checkpoint failed at this exact step.
        model.WarmUpKernels();
        int[] prompt = Enumerable.Range(0, 40).Select(i => (i * 37 + 11) % 250).ToArray();
        int[] steps = [17, 203, 99, 4, 150, 61];
        float[][] Run()
        {
            var rows = new List<float[]> { (float[])model.ForwardRefill(prompt).Clone() };
            AssertSpan(model);
            foreach (int token in steps)
            {
                rows.Add((float[])model.Forward([token]).Clone());
                AssertSpan(model);
            }
            Assert.Equal(prompt.Length + steps.Length, model.CacheSeqLen);
            Assert.Equal(prompt.Length + steps.Length, Field<int>(typeof(Qwen4ExpModel), model, "_qsaPositionCount"));
            foreach (float[] row in rows)
            {
                Assert.Equal(Qwen4ExpSyntheticModelBuilder.Vocab, row.Length);
                Assert.All(row, value => Assert.True(float.IsFinite(value)));
                Assert.True(row.Sum(value => (double)value * value) > 1e-6, "Degenerate logits.");
            }
            return rows.ToArray();
        }
        // Forty tokens cross the fixture's nineteen-cell sparse-attention width.
        float[][] first = Run();
        model.ResetKVCache();
        float[][] second = Run();
        for (int i = 0; i < first.Length; i++) Assert.Equal(first[i], second[i]);
    }

    private static void AssertSpan(Qwen4ExpModel model)
    {
        Assert.False(Field<bool>(typeof(Qwen4ExpModel), model, "_tokenGraphUnsupported"));
        Assert.True(Field<bool>(typeof(Qwen4ExpModel), model, "_spanLogitsValid"));
        Assert.NotNull(Field<object>(typeof(Qwen4ExpModel), model, "_pleArgs"));
        Assert.NotNull(Field<object>(typeof(Qwen4ExpModel), model, "_qsaArgs"));
    }

    private static T Field<T>(Type owner, object model, string name)
        => (T)owner.GetField(name, BindingFlags.Instance | BindingFlags.NonPublic)!.GetValue(model)!;

    private static float[] DecodeCoefficients(GgufTensorInfo info, byte[] bytes)
    {
        var result = new float[info.NumElements];
        for (int i = 0; i < result.Length; i++)
        {
            result[i] = info.Type switch
            {
                GgmlTensorType.F32 => BitConverter.ToSingle(bytes, i * 4),
                GgmlTensorType.F16 => (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(bytes, i * 2)),
                GgmlTensorType.BF16 => BitConverter.UInt32BitsToSingle((uint)BitConverter.ToUInt16(bytes, i * 2) << 16),
                GgmlTensorType.Q8_0 => (float)BitConverter.UInt16BitsToHalf(BitConverter.ToUInt16(bytes, i / 32 * 34))
                    * unchecked((sbyte)bytes[i / 32 * 34 + 2 + i % 32]),
                _ => throw new ArgumentException($"Unexpected coefficient type {info.Type}."),
            };
        }
        return result;
    }

    public void Dispose()
    {
        _environment.Dispose();
        try { Directory.Delete(_directory, recursive: true); } catch (IOException) { }
    }
}
