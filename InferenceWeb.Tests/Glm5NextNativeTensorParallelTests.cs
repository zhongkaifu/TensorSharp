// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

// Native-loader regression coverage for glm5next tensor parallelism. The real
// GLM-5.3-Flash checkpoint is roughly 109 GB; these tests use a one-layer KDA
// fixture that retains the relevant quantization-block/head geometry.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class Glm5NextNativeTensorParallelTests : IDisposable
{
    private readonly string _dir = Path.Combine(
        Path.GetTempPath(), "ts-glm5next-tp-" + Guid.NewGuid().ToString("N"));

    public Glm5NextNativeTensorParallelTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private NativeEnvScope NativeCpuTpEnvironment()
    {
        var env = new NativeEnvScope();
        // The native executor normally requires one GPU per rank. Oversubscription
        // is its explicit test mode; with ggml_cpu both ranks share the CPU backend
        // while exercising the same source slicing and validation code.
        env.Set("TS_GLM_TP_OVERSUBSCRIBE", "1");
        env.Set("MAX_CONTEXT", "256");
        env.Set("TS_GLM_UBATCH", "4");
        env.Set("TS_GLM_THREADS", "2");
        env.Set("TS_GLM_NATIVE", null);
        env.Set("TS_GLM_TP_SHARD", null);
        return env;
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void PublishedHyphenatedArchitecture_PreservesLogitsAndRecurrentContract()
    {
        string legacy = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
            Path.Combine(_dir, "legacy.gguf"), 4, true, numLayers: 4,
            mixedAttention: true, routedExperts: true, quantizeExperts: true);
        string current = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
            Path.Combine(_dir, "current.gguf"), 4, true, numLayers: 4,
            mixedAttention: true, routedExperts: true, quantizeExperts: true, architecture: "glm5-next");
        using NativeEnvScope env = NativeCpuTpEnvironment();
        Assert.Same(ChatProtocolRegistry.For("glm5next"), ChatProtocolRegistry.For("glm5-next"));
        foreach (string native in new[] { "0", "1" })
        {
            env.Set("TS_GLM_NATIVE", native);
            using ModelBase reference = ModelBase.Create(legacy, BackendType.GgmlCpu);
            using ModelBase actual = ModelBase.Create(current, BackendType.GgmlCpu);
            Assert.Equal("glm5-next", actual.Config.Architecture);
            Assert.False(actual.SupportsKVCacheTruncation);
            int[] prompt = { 65, 66, 67 };
            float[] expected = (float[])reference.ForwardRefill(prompt).Clone();
            float[] first = (float[])actual.ForwardRefill(prompt).Clone();
            AssertLogitsClose(expected, first, 0);
            int next = ArgMax(expected);
            AssertLogitsClose(reference.Forward(new[] { next }), actual.Forward(new[] { next }), 0);
            actual.ResetKVCache();
            AssertLogitsClose(first, actual.ForwardRefill(prompt), 0);
        }
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeLoader_AcceptsGlm5NextWithTwoAlignedTpRanks()
    {
        string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
            Path.Combine(_dir, "aligned.gguf"), numHeads: 4, quantizeAttentionOutput: true);

        using NativeEnvScope env = NativeCpuTpEnvironment();
        using ModelBase single = ModelBase.Create(path, BackendType.GgmlCpu, tpDegree: 1);
        using ModelBase parallel = ModelBase.Create(path, BackendType.GgmlCpu, tpDegree: 2);

        Assert.Equal("glm5next", parallel.Config.Architecture);
        Assert.Equal(4, parallel.Config.NumHeads);

        int[] prompt = { 65, 66, 67 };
        float[] singleRefill = (float[])single.ForwardRefill(prompt).Clone();
        float[] parallelRefill = (float[])parallel.ForwardRefill(prompt).Clone();
        AssertLogitsClose(singleRefill, parallelRefill, 2e-4f);

        int next = ArgMax(singleRefill);
        float[] singleDecode = (float[])single.Forward(new[] { next }).Clone();
        float[] parallelDecode = (float[])parallel.Forward(new[] { next }).Clone();
        AssertLogitsClose(singleDecode, parallelDecode, 2e-4f);

        // KDA owns recurrent convolution and SSM state per rank. A reset must
        // clear all rank-local copies, not just rank 0's.
        parallel.ResetKVCache();
        float[] afterReset = (float[])parallel.ForwardRefill(prompt).Clone();
        AssertLogitsClose(parallelRefill, afterReset, 1e-6f);
    }

    [GgmlFact(BackendType.GgmlCpu)]
    public void NativeLoader_RejectsTpPartitionThatCutsAQuantizationHeadGroup()
    {
        string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
            Path.Combine(_dir, "unaligned.gguf"), numHeads: 2, quantizeAttentionOutput: true);

        using NativeEnvScope env = NativeCpuTpEnvironment();

        // Establish that the fixture itself is valid; only the two-rank split is
        // impossible. Q8_0 has 32-value blocks, while each KDA head is 16 wide,
        // so two heads form one indivisible group and cannot feed two ranks.
        using (ModelBase single = ModelBase.Create(path, BackendType.GgmlCpu, tpDegree: 1))
            Assert.Equal("glm5next", single.Config.Architecture);

        // A refusal, carrying the native loader's own reason rather than "see stderr":
        // the hosts turn it into one error line and exit code 2.
        var error = Assert.Throws<ModelLoadRefusedException>(
            () => ModelBase.Create(path, BackendType.GgmlCpu, tpDegree: 2));
        Assert.Contains("[glm] --tp 2 cannot split 2 heads", error.Message, StringComparison.Ordinal);
        Assert.Contains("unaligned.gguf", error.Message, StringComparison.Ordinal);
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private static void AssertLogitsClose(float[] expected, float[] actual, float tolerance)
    {
        Assert.Equal(expected.Length, actual.Length);
        Assert.Equal(ArgMax(expected), ArgMax(actual));
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(expected[i]) && float.IsFinite(actual[i]),
                $"non-finite logit at {i}: expected={expected[i]}, actual={actual[i]}");
            Assert.InRange(MathF.Abs(expected[i] - actual[i]), 0.0f, tolerance);
        }
    }
}
