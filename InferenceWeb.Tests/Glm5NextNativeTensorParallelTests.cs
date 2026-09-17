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
using System.Runtime.InteropServices;
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

    private NativeEnvironmentScope NativeCpuTpEnvironment()
    {
        var env = new NativeEnvironmentScope();
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

    [Fact]
    public void NativeLoader_AcceptsGlm5NextWithTwoAlignedTpRanks()
    {
        string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
            Path.Combine(_dir, "aligned.gguf"), numHeads: 4, quantizeAttentionOutput: true);

        using NativeEnvironmentScope env = NativeCpuTpEnvironment();
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

    [Fact]
    public void NativeLoader_RejectsTpPartitionThatCutsAQuantizationHeadGroup()
    {
        string path = GlmDsaSyntheticModelBuilder.WriteGlm5NextTpFixture(
            Path.Combine(_dir, "unaligned.gguf"), numHeads: 2, quantizeAttentionOutput: true);

        using NativeEnvironmentScope env = NativeCpuTpEnvironment();

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

    /// <summary>
    /// Environment.SetEnvironmentVariable is sufficient for managed readers,
    /// but on Unix .NET's environment table is not guaranteed to update libc's
    /// table after process startup. The native executor reads its diagnostic TP
    /// switches with getenv(), so keep both views synchronized in this test.
    /// </summary>
    private sealed class NativeEnvironmentScope : IDisposable
    {
        private readonly Dictionary<string, string?> _originals = new();

        [DllImport("libc", EntryPoint = "setenv", CharSet = CharSet.Ansi, SetLastError = true)]
        private static extern int SetEnvUnix(string name, string value, int overwrite);

        [DllImport("libc", EntryPoint = "unsetenv", CharSet = CharSet.Ansi, SetLastError = true)]
        private static extern int UnsetEnvUnix(string name);

        [DllImport("ucrtbase", EntryPoint = "_putenv_s", CharSet = CharSet.Ansi, SetLastError = true)]
        private static extern int PutEnvWindows(string name, string value);

        public void Set(string name, string? value)
        {
            if (!_originals.ContainsKey(name))
                _originals[name] = Environment.GetEnvironmentVariable(name);
            SetBoth(name, value);
        }

        public void Dispose()
        {
            foreach (var pair in _originals)
                SetBoth(pair.Key, pair.Value);
        }

        private static void SetBoth(string name, string? value)
        {
            Environment.SetEnvironmentVariable(name, value);
            int result;
            if (OperatingSystem.IsWindows())
            {
                result = PutEnvWindows(name, value ?? string.Empty);
            }
            else
            {
                result = value == null ? UnsetEnvUnix(name) : SetEnvUnix(name, value, overwrite: 1);
            }
            if (result != 0)
                throw new InvalidOperationException($"Failed to update native environment variable '{name}'.");
        }
    }
}
