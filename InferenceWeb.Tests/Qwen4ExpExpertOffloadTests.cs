// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp.GGML;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// MoE CPU offload for Qwen3.8-Flash-Next (qwen4exp): a layer whose routed experts run on the host,
/// between two slices of the token span, answers as the all-accelerator span does. Run on the tiny
/// checkpoint of <see cref="Qwen4ExpSyntheticModelBuilder"/> (8 layers, GDN and QSA, PLE at layer 1).
/// </summary>
[Collection("MoeCpuOffloadConfig")]
public sealed class Qwen4ExpExpertOffloadTests : IDisposable
{
    private readonly ITestOutputHelper _output;
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-qwen4exp-offload-" + Guid.NewGuid().ToString("N"));
    private readonly EnvScope _env = new EnvScope();

    public Qwen4ExpExpertOffloadTests(ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        _env.Set("MAX_CONTEXT", "1024");
        _env.ClearSpeculationVars();
        // Offloaded runs record a routing profile; keep it out of the user's cache.
        _env.Set("TENSORSHARP_MOE_PROFILE", null);
        _env.Set("TENSORSHARP_MOE_PROFILE_DIR", _dir);
        MoeCpuOffloadConfig.Reset();
    }

    public void Dispose()
    {
        MoeCpuOffloadConfig.Reset();
        _env.Dispose();
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private static readonly int[] Prompt = Enumerable.Range(0, 40).Select(i => (i * 37 + 11) % 250).ToArray();
    // Past the host-MoE stream threshold (128 tokens): an offloaded layer's prefill
    // runs on the accelerator with its experts streamed in, not on the host.
    private static readonly int[] LongPrompt = Enumerable.Range(0, 160).Select(i => (i * 53 + 7) % 250).ToArray();
    private static readonly int[] Steps = { 17, 203, 99, 4, 150, 61 };

    private float[][] Run(BackendType backend, Action configure) => Run(backend, configure, Prompt);

    private float[][] Run(BackendType backend, Action configure, int[] prompt, Action<ModelBase>? loaded = null,
        Action<ModelBase>? done = null)
    {
        MoeCpuOffloadConfig.Reset();
        configure();
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "qwen4exp.gguf"), 16);
        using ModelBase model = ModelBase.Create(path, backend);
        Assert.IsType<Qwen4ExpModel>(model);
        loaded?.Invoke(model);
        var rows = new float[Steps.Length + 1][];
        rows[0] = (float[])model.ForwardRefill(prompt).Clone();
        for (int i = 0; i < Steps.Length; i++)
            rows[i + 1] = (float[])model.Forward(new[] { Steps[i] }).Clone();
        done?.Invoke(model);
        return rows;
    }

    /// <summary>The first half of the layers, then every layer, on the host. Prefill (40 tokens) and
    /// decode both cross the seams; the host quantizes activations to Q8_K where Metal keeps its own
    /// kernels, so the bound is a tolerance, not equality. Measured on an M5 Pro: at most 7.4e-4
    /// relative error with half the layers offloaded and 1.03e-3 with all of them, every argmax
    /// equal.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void OffloadedExperts_AnswerAsTheAccelerator_Metal() => OffloadedExperts(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void OffloadedExperts_AnswerAsTheAccelerator_Cuda() => OffloadedExperts(BackendType.GgmlCuda);

    private void OffloadedExperts(BackendType backend)
    {
        float[][] device = Run(backend, () => { });
        float[][] half = Run(backend, () => MoeCpuOffloadConfig.SetLayers(4));
        float[][] all = Run(backend, MoeCpuOffloadConfig.SetAllLayers);
        CheckGraphReference(backend, "prefill40-device", device);
        CheckGraphReference(backend, "prefill40-host4", half);
        CheckGraphReference(backend, "prefill40-host8", all);
        AssertClose("--n-cpu-moe 4", half, device, Bound(backend, hostPrefill: true));
        AssertClose("--cpu-moe", all, device, Bound(backend, hostPrefill: true));
    }

    /// <summary>Metal multiplies with F32 activations, so the host's Q8_K quantization is the only
    /// difference (measured 1.03e-3). ggml-cuda's MMQ quantizes activations to Q8_1 per 32 values,
    /// and a span cut changes which multiplies and adds it fuses (see q4e_nodes_ffn), so two
    /// different roundings meet: measured 6.5e-3 with streamed prefill rows up to 1.5e-2 on an
    /// A40, and 2.065e-2 at decode 4 on an RTX 3080 Laptop. The latter also occurs with the
    /// original host graph (TS_HOST_MOE_DECODE=0); it is not introduced by the decode kernel.
    /// Only host prefill uses the wider CUDA bound; streamed prefill retains 2e-2.
    /// Compare that kernel against the host graph separately at 1e-6 via
    /// eng/validation/qwen4exp-offload-parity.py, retaining argmax equality in both checks.</summary>
    private static double Bound(BackendType backend, bool hostPrefill = false)
        => backend == BackendType.GgmlCuda ? (hostPrefill ? 2.5e-2 : 2e-2) : 5e-3;

    /// <summary>Accelerator expert tiers on every offloaded layer (ggml_ops_moe_hot_tier.cpp). With a
    /// window wide enough for every route, the accelerator computes every expert from the compact
    /// slots with the all-accelerator span's own kernels, so the answer matches it far inside the
    /// host bound. With only fixed slots, or a window too small for a step (evictions every token, the
    /// rest overflowing to the host), the split meets the host's Q8_K arithmetic: the host bound.
    /// Prefill (40 tokens) runs the split too; every decode step reads misses into the window.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void TieredExperts_AnswerAsTheAccelerator_Metal()
    {
        float[][] device = Run(BackendType.GgmlMetal, () => { });
        int experts = 0;
        float[][] wide = Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, Prompt, model =>
        {
            var q = Assert.IsType<Qwen4ExpModel>(model);
            experts = q.ExpertCountForTests;
            Assert.Equal(8, q.RegisterExpertTiersForTests(experts / 4, experts));
        });
        float[][] fixedOnly = Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, Prompt, model =>
            Assert.Equal(8, ((Qwen4ExpModel)model).RegisterExpertTiersForTests(experts / 2, 0)));
        float[][] narrow = Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, Prompt, model =>
            Assert.Equal(8, ((Qwen4ExpModel)model).RegisterExpertTiersForTests(1, 2)));
        AssertClose("tier, window covers every route", wide, device, 1e-4);
        AssertClose("tier, fixed slots only", fixedOnly, device, Bound(BackendType.GgmlMetal, hostPrefill: true));
        AssertClose("tier, narrow window", narrow, device, Bound(BackendType.GgmlMetal, hostPrefill: true));
    }

    /// <summary>A batched verify-width span (here a 4-token prefill) reads up to two window misses a
    /// row in the background while the accelerator computes the routes its slots already hold, then
    /// computes those after a second cut (host_moe_build_hot_part's overlap group). Only the
    /// order the routes are summed in changes, so with a window wide enough for every route the
    /// answer still matches the all-accelerator span far inside the host bound; a misread or
    /// half-written slot would not.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void TieredExperts_VerifyWidthOverlap_Metal()
    {
        int[] verifyWidth = Prompt.Take(4).ToArray();
        float[][] device = Run(BackendType.GgmlMetal, () => { }, verifyWidth);
        long deferred = 0;
        float[][] overlapped = Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, verifyWidth, model =>
        {
            var q = Assert.IsType<Qwen4ExpModel>(model);
            Assert.Equal(8, q.RegisterExpertTiersForTests(q.ExpertCountForTests / 4, q.ExpertCountForTests));
        }, _ => deferred = GgmlBasicOps.MoeHotTierStats().Deferred);
        Assert.True(deferred > 0, "no window miss was read in the background");
        AssertClose("tier, verify-width overlap", overlapped, device, 1e-4);
    }

    /// <summary>Discrete VRAM: fixed slots only (no host-writable window), the rest of every
    /// route on the host. Prefill (40 tokens) runs the batched expert kernels on the slots.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void TieredExperts_AnswerAsTheAccelerator_Cuda()
    {
        float[][] device = Run(BackendType.GgmlCuda, () => { });
        float[][] tiered = Run(BackendType.GgmlCuda, MoeCpuOffloadConfig.SetAllLayers, Prompt, model =>
        {
            var q = Assert.IsType<Qwen4ExpModel>(model);
            Assert.Equal(8, q.RegisterExpertTiersForTests(q.ExpertCountForTests / 2, 0));
        });
        AssertClose("tier, fixed slots only (CUDA)", tiered, device, Bound(BackendType.GgmlCuda, hostPrefill: true));
    }

    /// <summary>A discrete-VRAM layer plan has no host seam on its resident layers, so a profile built
    /// from the seams alone never covers the model and the next load could never tier it. While
    /// recording, the resident layers' graphs hand their router ids to the same counters: every MoE
    /// layer then counts the same routes, and the logits are those of the plan without recording.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void ResidentLayersRecordTheirRouting_Metal() => ResidentLayersRecordTheirRouting(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void ResidentLayersRecordTheirRouting_Cuda() => ResidentLayersRecordTheirRouting(BackendType.GgmlCuda);

    private void ResidentLayersRecordTheirRouting(BackendType backend)
    {
        var plainTotals = new List<ulong>();
        var totals = new List<ulong>();
        void Collect(ModelBase model, List<ulong> into)
        {
            var q = Assert.IsType<Qwen4ExpModel>(model);
            for (int l = 0; l < q.Config.NumLayers; l++)
                if (q.RoutingCountsForTests(l) is ulong[] counts)
                    into.Add(counts.Aggregate(0UL, (a, c) => a + c));
                else
                    into.Add(0);
        }
        float[][] plain = Run(backend, () => MoeCpuOffloadConfig.SetLayers(4), Prompt, done: m => Collect(m, plainTotals));
        float[][] recorded = Run(backend, () => MoeCpuOffloadConfig.SetLayers(4), Prompt,
            model => Assert.IsType<Qwen4ExpModel>(model).RecordDeviceRoutesForTests(true),
            model => Collect(model, totals));

        // Without recording only the four host layers count; with it every layer does, alike.
        Assert.Contains(plainTotals, t => t == 0);
        Assert.All(totals, t => Assert.True(t > 0));
        Assert.Single(totals.Distinct());
        Assert.Equal(plainTotals.Max(), totals[0]);
        for (int i = 0; i < plain.Length; i++)
            Assert.Equal(plain[i], recorded[i]);
    }

    /// <summary>A 160-token prefill streams each offloaded layer's experts to the accelerator. The
    /// seam then reads a result the stream path published through an asynchronous download on
    /// Metal; before the drain was added it passed on the PREVIOUS layer's expert output (relative error 2e-2
    /// at 16 tokens, a race at long prompts). Same kernels as the in-span chain, so the bound is
    /// tight.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void StreamedPrefill_AnswersAsTheAccelerator_Metal() => StreamedPrefill(BackendType.GgmlMetal);

    [GgmlFact(BackendType.GgmlCuda)]
    public void StreamedPrefill_AnswersAsTheAccelerator_Cuda() => StreamedPrefill(BackendType.GgmlCuda);

    private void StreamedPrefill(BackendType backend)
    {
        float[][] device = Run(backend, () => { }, LongPrompt);
        float[][] all = Run(backend, MoeCpuOffloadConfig.SetAllLayers, LongPrompt);
        CheckGraphReference(backend, "prefill160-device", device);
        CheckGraphReference(backend, "prefill160-host8", all);
        AssertClose("--cpu-moe, streamed prefill", all, device, Bound(backend));
    }

    // The native environment switches are cached once per process. The validation tool
    // runs fresh graph/optimized/pinned processes and supplies the graph's complete logits
    // here, so a wider CPU-versus-GPU quantization bound cannot hide a kernel regression.
    private void CheckGraphReference(BackendType backend, string scenario, float[][] logits)
    {
        string fileName = $"{backend}-{scenario}.json";
        string captureDir = Environment.GetEnvironmentVariable("TS_Q4E_OFFLOAD_CAPTURE_DIR");
        if (!string.IsNullOrEmpty(captureDir))
        {
            Directory.CreateDirectory(captureDir);
            File.WriteAllText(Path.Combine(captureDir, fileName), JsonSerializer.Serialize(logits));
        }
        string referenceDir = Environment.GetEnvironmentVariable("TS_Q4E_OFFLOAD_REFERENCE_DIR");
        if (!string.IsNullOrEmpty(referenceDir))
        {
            float[][] reference = JsonSerializer.Deserialize<float[][]>(
                File.ReadAllText(Path.Combine(referenceDir, fileName)))!;
            AssertClose($"host graph reference: {scenario}", logits, reference, 1e-6);
        }
    }

    /// <summary>TensorAgent's desktop default K/V cache is q8_0, which the qwen4exp token span cannot
    /// read and which, with QSA layers, left no path to run on: the request now loads an f16 cache
    /// instead (stderr says so) and answers as an f16 run does.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void AQuantizedKvRequest_RunsOnAnF16Cache_Metal()
    {
        KvCacheDtype restoreDtype = KvCacheDtypeConfig.Current;
        bool restoreExplicit = KvCacheDtypeConfig.IsExplicitlySet;
        try
        {
            float[][] f16 = Run(BackendType.GgmlMetal, () => KvCacheDtypeConfig.Set(KvCacheDtype.F16));
            float[][] q8 = Run(BackendType.GgmlMetal, () => KvCacheDtypeConfig.Set(KvCacheDtype.Q8_0), Prompt,
                model => Assert.Equal(KvCacheDtype.F16, model.KvCacheDtype));
            // The substitution is this model's; the operator's request stays for the next model.
            Assert.Equal(KvCacheDtype.Q8_0, KvCacheDtypeConfig.Current);
            AssertClose("--kv-cache-dtype q8_0", q8, f16, 0.0);
        }
        finally
        {
            KvCacheDtypeConfig.RestoreForTests(restoreDtype, restoreExplicit);
        }
    }

    private void AssertClose(string what, float[][] actual, float[][] expected, double bound)
    {
        Assert.Equal(expected.Length, actual.Length);
        double worst = 0;
        for (int i = 0; i < expected.Length; i++)
        {
            Assert.Equal(expected[i].Length, actual[i].Length);
            Assert.All(actual[i], v => Assert.True(float.IsFinite(v)));
            double err = RelativeError(actual[i], expected[i]);
            worst = Math.Max(worst, err);
            _output.WriteLine($"{what} {(i == 0 ? "prefill" : $"decode {i}")}: relative error {err:E3}, " +
                $"argmax {ArgMax(actual[i])} vs {ArgMax(expected[i])}");
            Assert.Equal(ArgMax(expected[i]), ArgMax(actual[i]));
        }
        Assert.True(worst <= bound, $"{what} strayed {worst:E3} from the all-accelerator span");
    }

    /// <summary>--backend mlx is refused before anything is mapped, naming the backend that runs the
    /// model; it used to die with an Int32 overflow while wrapping the 28.8 GB n-gram table.</summary>
    [Fact]
    public void Mlx_IsRefusedUpFront_WithTheBackendThatRunsIt()
    {
        var ctx = new TensorSharp.Models.Architecture.ModelCreateContext(
            Path.Combine(_dir, "never-opened.gguf"), BackendType.Mlx, probe: null);
        var ex = Assert.Throws<NotSupportedException>(() => Qwen4ExpArchitecture.Descriptor.Factory(ctx));
        Assert.Contains("--backend ggml_metal", ex.Message);
        Assert.False(File.Exists(ctx.GgufPath));
    }

    // ---------------------------------------------------------------- the placement plan

    private const long GB = 1_000_000_000L;

    /// <summary>The shipped UD-Q2_K_XL: 48 layers of 0.957 GB of experts and ~4.0 GB of everything
    /// else the span binds, on a 48 GiB M5 Pro (40.2 GB Metal working set).</summary>
    [Fact]
    public void Plan_On48GiBMac_OffloadsTheLeadingLayers()
    {
        long[] experts = Enumerable.Repeat(957_000_000L, 48).ToArray();
        int onDevice = Qwen4ExpModel.PlanDeviceExpertLayers(experts, 4 * GB, 40_200_000_000L, 51_539_607_552L);
        _output.WriteLine($"48 GiB: {onDevice} of 48 layers' experts on the accelerator");
        // Measured best on that Mac: 12-16 wired layers (see OffloadedHotFraction).
        Assert.InRange(onDevice, 12, 18);
    }

    [Fact]
    public void Plan_WhenEverythingFits_KeepsEveryLayer()
    {
        long[] experts = Enumerable.Repeat(957_000_000L, 48).ToArray();
        Assert.Equal(48, Qwen4ExpModel.PlanDeviceExpertLayers(experts, 4 * GB, 110 * GB, 137_438_953_472L));
    }

    [Fact]
    public void Plan_NeverExceedsTheWorkingSet()
    {
        long[] experts = Enumerable.Repeat(957_000_000L, 48).ToArray();
        // Plenty of RAM, a small working set: the working set is the binding limit.
        int onDevice = Qwen4ExpModel.PlanDeviceExpertLayers(experts, 4 * GB, 20 * GB, 512 * GB);
        long wired = 4 * GB + Qwen4ExpModel.DeviceScratchBytes + onDevice * 957_000_000L;
        Assert.True(wired <= 20 * GB - 20 * GB / 16, $"{onDevice} layers wire {wired} bytes");
        Assert.True(wired + 957_000_000L > 20 * GB - 20 * GB / 16, "one more layer would still have fit");
    }

    [Fact]
    public void Plan_TooSmallForAnyLayer_OffloadsEverything()
        => Assert.Equal(0, Qwen4ExpModel.PlanDeviceExpertLayers(
            Enumerable.Repeat(957_000_000L, 48).ToArray(), 4 * GB, 8 * GB, 16 * GB));

    /// <summary>The descriptor crosses to TSGgmlQwen4ExpFfnArgs by layout: 12 pointers, 10 int64,
    /// 10 int32, then cpu_moe.</summary>
    [Fact]
    public void FfnDescriptor_AppendsCpuMoeAfterTheTypes()
    {
        Assert.Equal(216, (int)Marshal.OffsetOf<Qwen4ExpFfnArgs>(nameof(Qwen4ExpFfnArgs.CpuMoe)));
        Assert.Equal(224, Marshal.SizeOf<Qwen4ExpFfnArgs>());
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
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
