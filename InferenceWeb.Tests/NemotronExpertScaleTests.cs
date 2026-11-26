// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Nemotron NVFP4 sidecar scales on every path of the shared runtime's model (NemotronRuntimeModel). A GGUF with ".scale"
// sidecars must compute what its folded twin computes - on one device, on the GGML backends, under tensor parallelism
// (each rank's column shards keep the source's scale) and with the routed experts on the host (each route's up scale
// before the square). See NemotronSyntheticModelBuilder for why the twin is an exact oracle.
using System;
using System.Collections.Generic;
using System.IO;
using TensorSharp;
using TensorSharp.Models;
using Xunit;

namespace InferenceWeb.Tests;

[Collection("MoeCpuOffloadConfig")]   // the host-expert cases set the process-wide expert-offload policy
public sealed class NemotronExpertScaleTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-nemotron-scale-" + Guid.NewGuid().ToString("N"));
    private readonly NativeEnvScope _environment = new();
    private readonly KvCacheDtype _oldDtype = KvCacheDtypeConfig.Current;
    private readonly bool _oldExplicit = KvCacheDtypeConfig.IsExplicitlySet;

    public NemotronExpertScaleTests()
    {
        Directory.CreateDirectory(_directory);
        _environment.Set("MAX_CONTEXT", "64");
        MoeCpuOffloadConfig.Reset();
        KvCacheDtypeConfig.Set(KvCacheDtype.F32);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_oldDtype, _oldExplicit);
        MoeCpuOffloadConfig.Reset();
        _environment.Dispose();
        try { Directory.Delete(_directory, true); } catch { /* best effort */ }
    }

    private static readonly int[] LongPrompt = { 3, 17, 42, 9, 100, 7, 64, 11, 5, 88, 23, 51 };
    private static readonly int[] ShortPrompt = { 31, 8, 77, 4, 19 };

    private string Write(string name, NemotronSyntheticModelBuilder builder)
        => builder.Write(Path.Combine(_directory, name));

    private static float[] QkvScales(bool distinct) => distinct ? new[] { 0.5f, 2f, 4f } : new[] { 0.5f, 0.5f, 0.5f };

    private static NemotronRuntimeModel Create(string path, BackendType backend = BackendType.Cpu, int tp = 1)
        => new(path, backend, tp, tp > 1 ? new SimulatedDeviceGroup(tp) : null, 1, null);

    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CpuSingle_ScaledSidecarsMatchFoldedTwin(bool latent, bool distinctQkv)
    {
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Latent = latent, Scaled = true, QkvScales = QkvScales(distinctQkv) });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder { Latent = latent });

        using var reference = Create(twin);
        using var model = Create(scaled);
        AssertSameTrajectory(reference, model, 1e-3f);
    }

    /// <summary>Tensor parallelism over simulated devices: each rank's shards of a scaled projection keep its scale (a
    /// column shard's partial sum is scaled before the reduction), and the experts' per-expert scales follow their slices.</summary>
    [Theory]
    [InlineData(false, false)]
    [InlineData(false, true)]
    [InlineData(true, false)]
    [InlineData(true, true)]
    public void CpuTensorParallel_ScaledSidecarsMatchSingleAndTwin(bool latent, bool distinctQkv)
    {
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Latent = latent, Scaled = true, QkvScales = QkvScales(distinctQkv) });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder { Latent = latent });

        List<Step> single;
        using (var one = Create(scaled)) single = Trajectory(one);
        using var tp = Create(scaled, tp: 2);
        Assert.Equal(2, tp.Mesh.TensorParallel);
        // Each rank's latent partial is quantized to the latent projection's 8-bit dot type before ffn_latent_up, where one
        // device quantizes the whole sum: the same arithmetic, another rounding (up to ~0.025 in a logit here).
        AssertFollows(single, tp, latent ? 5e-2f : 2e-3f);
        using var twinTp = Create(twin, tp: 2);
        AssertFollows(Trajectory(twinTp), tp, 2e-3f);
    }

    /// <summary>Only the per-expert down scales (a file whose other sidecars are 1): the routing weights carry them.</summary>
    [Fact]
    public void CpuTensorParallel_DownScaleOnlyIsApplied()
    {
        var builder = new NemotronSyntheticModelBuilder
        {
            Scaled = true,
            ExpertUpScales = false,
            QkvScales = new[] { 1f, 1f, 1f },
            AttnOutputScale = 1f,
            SharedUpScale = 1f,
            SharedDownScale = 1f,
        };
        string scaled = Write("down-only.gguf", builder);
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder());

        using var reference = Create(twin);
        using var tp = Create(scaled, tp: 2);
        AssertSameTrajectory(reference, tp, 2e-3f);
    }

    /// <summary>The routed experts on the host (--cpu-moe): each route's up scale multiplies its up product before the
    /// square, on the managed backend and in GGML's host MoE segment, as the folded twin's experts compute it.</summary>
    [Theory]
    [InlineData(false, "managed")]
    [InlineData(true, "managed")]
    [InlineData(false, "ggml")]
    [InlineData(true, "ggml")]
    public void HostExperts_ScaledSidecarsMatchFoldedTwin(bool latent, string executor)
    {
        BackendType backend = executor == "ggml" ? TestGates.PinnedGgmlBackend : BackendType.Cpu;
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Latent = latent, Scaled = true, QkvScales = QkvScales(true) });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder { Latent = latent });

        List<Step> expected;
        using (var reference = Create(twin)) expected = Trajectory(reference);
        MoeCpuOffloadConfig.SetAllLayers();
        try
        {
            using var model = Create(scaled, backend);
            // Managed: the same products; GGML: 8-bit activations (the host kernel's dot types).
            AssertFollows(expected, model, backend == BackendType.Cpu ? 1e-3f : 5e-2f);
        }
        finally
        {
            MoeCpuOffloadConfig.Reset();
        }
    }

    [GgmlTheory(BackendType.GgmlCpu)]
    [InlineData(3)]
    [InlineData(18)]
    public void GgmlCpuSingle_ScaledSidecarsMatchFoldedTwin(int topK)
    {
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Experts = 24, TopK = topK, Scaled = true, QkvScales = QkvScales(true) });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder { Experts = 24, TopK = topK });

        using var reference = Create(twin, BackendType.GgmlCpu);
        using var model = Create(scaled, BackendType.GgmlCpu);
        AssertSameTrajectory(reference, model, 1e-3f);
    }

    /// <summary>ggml-cuda against the managed twin: at the builder's widths every latent projection and expert is a Q8_0
    /// matrix under 4 KiB, which GGML binds through its cache keys. GGML rounds activations to 8 bits (up to ~0.012 here);
    /// a wrong weight or scale moves the logits by whole units or to NaN.</summary>
    [GgmlTheory(BackendType.GgmlCuda)]
    [InlineData(false)]
    [InlineData(true)]
    public void GgmlCudaSingle_ScaledSidecarsMatchFoldedTwin(bool latent)
    {
        int expertFf = latent ? 64 : 32;
        string scaled = Write("scaled.gguf", new NemotronSyntheticModelBuilder { Latent = latent, ExpertFf = expertFf, Scaled = true });
        string twin = Write("twin.gguf", new NemotronSyntheticModelBuilder { Latent = latent, ExpertFf = expertFf });

        List<Step> expected;
        using (var reference = Create(twin))
            expected = Trajectory(reference);
        using var model = Create(scaled, BackendType.GgmlCuda);
        AssertFollows(expected, model, 5e-2f);
    }

    /// <summary>
    /// Runs the long prompt plus six greedy decode steps, then - after a reset - the short prompt plus six more, feeding
    /// both models the reference's greedy tokens and comparing every step's logits.
    /// </summary>
    internal static void AssertSameTrajectory(ModelBase reference, ModelBase model, float tolerance)
        => AssertFollows(Trajectory(reference), model, tolerance);

    /// <summary>One step of the trajectory: its input tokens and the reference's logits.</summary>
    private sealed record Step(int PromptLength, int Index, int[] Tokens, float[] Logits);

    /// <summary>The reference's side of <see cref="AssertSameTrajectory"/>, recorded so the reference can be disposed
    /// before the model under test loads.</summary>
    private static List<Step> Trajectory(ModelBase reference)
    {
        var steps = new List<Step>();
        foreach (int[] prompt in new[] { LongPrompt, ShortPrompt })
        {
            reference.ResetKVCache();
            int[] tokens = prompt;
            for (int step = 0; step < 7; step++)
            {
                float[] logits = (float[])reference.Forward(tokens).Clone();
                steps.Add(new Step(prompt.Length, step, tokens, logits));
                tokens = new[] { ArgMax(logits) };
            }
        }
        return steps;
    }

    private static void AssertFollows(List<Step> expected, ModelBase model, float tolerance)
    {
        foreach (Step step in expected)
        {
            if (step.Index == 0)
                model.ResetKVCache();
            float[] actual = model.Forward(step.Tokens);
            Close(step.Logits, actual, tolerance, $"prompt of {step.PromptLength}, step {step.Index}");
            Assert.Equal(ArgMax(step.Logits), ArgMax(actual));
        }
    }

    private static int ArgMax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; i++)
            if (values[i] > values[best]) best = i;
        return best;
    }

    private static void Close(float[] expected, float[] actual, float tolerance, string where)
    {
        Assert.Equal(expected.Length, actual.Length);
        float worst = 0;
        int at = -1;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.True(float.IsFinite(actual[i]), $"{where}: logit {i} is {actual[i]}");
            float d = MathF.Abs(expected[i] - actual[i]);
            if (d > worst) { worst = d; at = i; }
        }
        if (worst > tolerance)
            Assert.Fail($"{where}: max |dlogit| {worst:R} at {at} (expected {expected[at]:R}, actual {actual[at]:R}) exceeds {tolerance:R}");
    }
}
