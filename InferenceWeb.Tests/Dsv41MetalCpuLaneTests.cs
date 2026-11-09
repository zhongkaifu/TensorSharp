// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Linq;
using TensorSharp.GGML;
using TensorSharp.Models;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// DeepSeek V4.1 on <c>ggml_metal</c> with every routed expert on the host (<c>--cpu-moe</c>): the CPU
/// lane (TensorSharp's host decode kernel over the explicitly read RAM cache, prefill in 8-row blocks
/// with the next block read ahead, decode with the next layer's experts predicted and read ahead) must
/// answer as the ggml CPU chain it replaces (<c>TS_DSV41_HOST_LANE=0</c>), on the C#-built checkpoint
/// shaped like the 1017 (IQ1_S/IQ2_S/IQ2_XXS gate+up, five down types, F32 router).
/// </summary>
[Collection("MoeCpuOffloadConfig")]
public sealed class Dsv41MetalCpuLaneTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly EnvScope _env = new();
    private readonly NativeEnvScope _native = new();

    // A 37-token prefill (four full lane blocks and a 5-row one), decode steps, a 3-row append, decode.
    private static readonly int[][] Calls = new[]
    {
        Enumerable.Range(0, 37).Select(i => (i * 37 + 11) % Dsv41SyntheticModelBuilder.Vocab).ToArray(),
        new[] { 17 }, new[] { 203 }, new[] { 99 }, new[] { 4 },
        new[] { 150, 61, 88 },
        new[] { 12 }, new[] { 240 },
    };

    public Dsv41MetalCpuLaneTests(ITestOutputHelper output)
    {
        _output = output;
        // Read by the managed load check and by the native loader (getenv), so set on both sides.
        _env.Set("TS_DSV41_ALLOW_NON_CUDA_GPU", "1");
        _native.Set("TS_DSV41_ALLOW_NON_CUDA_GPU", "1");
        _env.Set("MAX_CONTEXT", "256");
        MoeCpuOffloadConfig.Reset();
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        _native.Dispose();
        _env.Dispose();
    }

    private float[][] Run(bool lane)
    {
        _native.Set("TS_DSV41_HOST_LANE", lane ? null : "0");
        MoeCpuOffloadConfig.SetAllLayers();
        using ModelBase model = ModelBase.Create(Dsv41SyntheticModelBuilder.Fixture(), BackendType.GgmlMetal);
        return Calls.Select(tokens => (float[])model.Forward(tokens).Clone()).ToArray();
    }

    [Dsv41MetalFact]
    public void CpuLane_AnswersAsTheGgmlChain_Metal()
    {
        float[][] chain = Run(lane: false);
        var before = GgmlBasicOps.HostExpertIoStats();
        float[][] lane = Run(lane: true);
        var after = GgmlBasicOps.HostExpertIoStats();

        // The lane really ran, from the RAM cache: its slices were looked up there.
        long lookups = after.Hits + after.Misses - before.Hits - before.Misses;
        _output.WriteLine($"RAM cache lookups during the lane run: {lookups} ({after.Hits - before.Hits} hits)");
        Assert.True(lookups > 0, "the lane run never resolved an expert through the RAM cache");

        for (int i = 0; i < Calls.Length; i++)
        {
            Assert.Equal(chain[i].Length, lane[i].Length);
            Assert.All(lane[i], v => Assert.True(float.IsFinite(v)));
            double err = RelativeError(lane[i], chain[i]);
            _output.WriteLine($"call {i} ({Calls[i].Length} token(s)): relative error {err:E3}, " +
                $"argmax {ArgMax(lane[i])} vs {ArgMax(chain[i])}");
            // Measured on an M5 Pro: 1.5e-5 for the prefill, 5e-5 to 1.9e-3 for the decode steps after it
            // (the host kernel and ggml's CPU dot round the activations differently, and the KV carries it).
            Assert.True(err <= 5e-3, $"call {i} strayed {err:E3} from the ggml chain");
            Assert.Equal(ArgMax(chain[i]), ArgMax(lane[i]));
        }
    }

    private static int ArgMax(float[] v)
    {
        int best = 0;
        for (int i = 1; i < v.Length; i++) if (v[i] > v[best]) best = i;
        return best;
    }

    private static double RelativeError(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            double d = a[i] - b[i];
            num += d * d;
            den += (double)b[i] * b[i];
        }
        return den > 0 ? Math.Sqrt(num / den) : Math.Sqrt(num);
    }
}

/// <summary>
/// A test of DeepSeek V4.1 on <c>ggml_metal</c>. On the managed side that load keeps only a ggml-cpu
/// context (the native engine creates its Metal device itself), so it runs in the process lane that pins
/// ggml-cpu, and only on macOS; anywhere else it skips visibly.
/// </summary>
[AttributeUsage(AttributeTargets.Method)]
public sealed class Dsv41MetalFactAttribute : FactAttribute
{
    public Dsv41MetalFactAttribute()
        => Skip = OperatingSystem.IsMacOS() ? TestGates.GgmlPinSkip(BackendType.GgmlCpu) : "Requires macOS (ggml_metal).";
}
