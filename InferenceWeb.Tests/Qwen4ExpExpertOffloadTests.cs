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
    // Adaptive tier slots are on by default; these tests hold fixed slots to the accelerator unless they turn it on.
    private readonly NativeEnvScope _nativeEnv = new NativeEnvScope();

    public Qwen4ExpExpertOffloadTests(ITestOutputHelper output)
    {
        _output = output;
        _nativeEnv.Set("TS_MOE_TIER_ADAPT", "0");
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
        _nativeEnv.Dispose();
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
        Action<ModelBase>? done = null) => Run(backend, configure, prompt, Steps, loaded, done);

    private float[][] Run(BackendType backend, Action configure, int[] prompt, int[] steps, Action<ModelBase>? loaded,
        Action<ModelBase>? done, string? fixture = null)
    {
        MoeCpuOffloadConfig.Reset();
        configure();
        string path = fixture ?? Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "qwen4exp.gguf"), 16);
        using ModelBase model = ModelBase.Create(path, backend);
        Assert.IsType<Qwen4ExpModel>(model);
        loaded?.Invoke(model);
        var rows = new float[steps.Length + 1][];
        rows[0] = (float[])model.ForwardRefill(prompt).Clone();
        for (int i = 0; i < steps.Length; i++)
            rows[i + 1] = (float[])model.Forward(new[] { steps[i] }).Clone();
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
        // ggml's tier mechanics on its own kernels; the grouped tsm kernels are held to these in TsmExperts_*.
        _nativeEnv.Set("TS_METAL_MOE", "0");
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

    /// <summary>A tier's held routes on Metal through tsm's grouped expert kernels (Strata's, tsm/kernels) against
    /// ggml-metal's MUL_MAT_ID over the same slots, format by format: every route held (the window covers them all), so
    /// the decode steps differ only in the expert arithmetic. Prompts (40 rows) take MUL_MAT_ID either way.</summary>
    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("iq3s-iq4nl")]
    [InlineData("q2_0")]
    [InlineData("q4k-q51")]
    [InlineData("iq2xs-iq4nl")]
    public void TsmExperts_AnswerAsMulMatId_Metal(string formats)
    {
        string path = formats switch
        {
            "q2_0" => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q2_0.gguf"), 16,
                expertGateUpType: SyntheticGguf.GgmlType.Q2_0, expertDownType: SyntheticGguf.GgmlType.Q2_0),
            "q4k-q51" => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q4k.gguf"), 16, q4kxlExperts: true),
            "iq2xs-iq4nl" => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q2kxl.gguf"), 16, q2kxlExperts: true),
            _ => Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "default.gguf"), 16),
        };
        float[][] Answer(string tsm)
        {
            using var env = new NativeEnvScope();
            env.Set("TS_METAL_MOE", tsm);
            return Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, Prompt, Steps, model =>
            {
                var q = Assert.IsType<Qwen4ExpModel>(model);
                Assert.Equal(8, q.RegisterExpertTiersForTests(q.ExpertCountForTests / 4, q.ExpertCountForTests));
            }, null, path);
        }
        float[][] mulMatId = Answer("0"), grouped = Answer("1");
        // Both are float arithmetic over the same bytes in a different order: measured 1.1e-7 .. 1.1e-6 on an M5 Pro
        // (the grouped kernels alone agree with a double-precision reference to ~2e-7, TsmMoeKernelTests).
        AssertClose($"tsm vs MUL_MAT_ID ({formats})", grouped, mulMatId, 1e-5);
    }

    /// <summary>QSA keeps each layer's pooled indexer keys between steps (Q4eSeqStateEntry::aux), so a step pools only
    /// the blocks its cells complete instead of every block of the window. A block's key is the same arithmetic either
    /// way, so the logits are the same bits as pooling every block every step (TS_Q4E_QSA_POOL_CACHE=0) - over a
    /// prompt, then decode steps that cross several block boundaries. The step counter proves the cache was used.
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void QsaPooledKeys_KeptBetweenSteps_AnswerAsPoolingEveryBlock_Cpu() => QsaPooledKeys(BackendType.GgmlCpu);

    [GgmlFact(BackendType.GgmlMetal)]
    public void QsaPooledKeys_KeptBetweenSteps_AnswerAsPoolingEveryBlock_Metal() => QsaPooledKeys(BackendType.GgmlMetal);

    private void QsaPooledKeys(BackendType backend)
    {
        int[] steps = Enumerable.Range(0, 14).Select(i => (i * 61 + 5) % 250).ToArray();
        float[][] Answer(string keep)
        {
            using var env = new NativeEnvScope();
            env.Set("TS_Q4E_QSA_POOL_CACHE", keep);
            return Run(backend, () => { }, Prompt, steps, null, null);
        }
        float[][] everyBlock = Answer("0");
        long before = GgmlBasicOps.Qwen4ExpQsaIncrementalSteps();
        float[][] kept = Answer("1");
        Assert.True(GgmlBasicOps.Qwen4ExpQsaIncrementalSteps() - before >= steps.Length,
            "the decode steps did not pool incrementally");
        for (int i = 0; i < kept.Length; i++)
            Assert.True(kept[i].AsSpan().SequenceEqual(everyBlock[i]), $"step {i}: the kept pooled keys changed the logits");
    }

    /// <summary>Seams on a shared event (tsm_moe.h seams_*): the decode steps' segments are enqueued ahead and each seam
    /// is a handshake on ggml-metal's queue instead of a submit and a wait. The kernels, their order and their buffers
    /// are the same, so the logits are the same bits - with every route in a slot, and with half the experts left to
    /// the host (whose routes reach the GPU at the seam's last handshake). The pass counter proves the event path ran.
    /// </summary>
    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData(false)]
    [InlineData(true)]
    public void TsmSeams_OnASharedEvent_AnswerAsSubmitAndWait_Metal(bool hostHalf)
    {
        string path = Qwen4ExpSyntheticModelBuilder.Write(Path.Combine(_dir, "q2_0.gguf"), 16,
            expertGateUpType: SyntheticGguf.GgmlType.Q2_0, expertDownType: SyntheticGguf.GgmlType.Q2_0);
        float[][] Answer(string events)
        {
            using var env = new NativeEnvScope();
            env.Set("TS_METAL_SEAM_EVENTS", events);
            return Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, Prompt, Steps, model =>
            {
                var q = Assert.IsType<Qwen4ExpModel>(model);
                int experts = q.ExpertCountForTests;
                Assert.Equal(8, hostHalf
                    ? q.RegisterExpertTiersForTests(experts / 2, 0)
                    : q.RegisterExpertTiersForTests(experts / 4, experts));
            }, null, path);
        }
        float[][] submitAndWait = Answer("0");
        long before = GgmlBasicOps.TsmSeamPasses();
        float[][] events = Answer("1");
        Assert.True(GgmlBasicOps.TsmSeamPasses() - before >= Steps.Length,
            "the decode steps did not run on the event seams");
        Assert.Equal(submitAndWait.Length, events.Length);
        for (int i = 0; i < events.Length; i++)
            Assert.True(events[i].AsSpan().SequenceEqual(submitAndWait[i]),
                $"step {i}: the event seams changed the logits ({(hostHalf ? "half the experts on the host" : "every route held")})");
    }

    /// <summary>A prompt-width span runs its GDN layers' recurrence (conv, L2 norms, gates, the state scan, the gated
    /// norm) and its hyper-connection passes (the gated stream mean after the up projection, the scatter) on
    /// TensorSharp's Metal kernels (tsm/kernels/tsm_gdn_prompt.metal after Strata's gdn_prompt, tsm_hc_prompt.metal)
    /// between graph segments, instead of ggml's ops. The decode steps run ggml's gated delta net on the state the
    /// kernels left, so they also prove the conv history and the state are written in ggml's layout. Only the rounding
    /// differs (this fixture amplifies it): measured on an M5 Pro, at most 6.4e-5 relative error with the GDN kernels,
    /// every argmax equal. With every expert on the host the kernels share the pass with the host seams, whose Q8_K
    /// activations amplify the difference to 3.0e-4. The counters prove the kernels ran.</summary>
    [GgmlTheory(BackendType.GgmlMetal)]
    [InlineData("gdn", false)]
    [InlineData("hc", false)]
    [InlineData("gdn,hc", false)]
    [InlineData("gdn,hc", true)]
    public void PromptKernels_OnTsm_AnswerAsGgml_Metal(string kernels, bool hostExperts)
    {
        bool gdn = kernels.Contains("gdn"), hc = kernels.Contains("hc");
        float[][] Answer(bool useGdn, bool useHc)
        {
            using var env = new NativeEnvScope();
            env.Set("TS_METAL_GDN_PROMPT", useGdn ? "1" : "0");
            env.Set("TS_METAL_HC_PROMPT", useHc ? "1" : "0");
            return Run(BackendType.GgmlMetal, hostExperts ? MoeCpuOffloadConfig.SetAllLayers : () => { }, LongPrompt);
        }
        float[][] ggml = Answer(false, false);
        long gdnBefore = GgmlBasicOps.TsmGdnPromptCount(), hcBefore = GgmlBasicOps.TsmHcPromptCount();
        float[][] tsm = Answer(gdn, hc);
        long gdnLayers = GgmlBasicOps.TsmGdnPromptCount() - gdnBefore, hcPasses = GgmlBasicOps.TsmHcPromptCount() - hcBefore;
        _output.WriteLine($"tsm kernels: {gdnLayers} GDN layers, {hcPasses} hyper-connection passes");
        // The fixture's 8 layers: 6 GDN, 16 half-layers with a read and a scatter each.
        Assert.Equal(gdn ? 6 : 0, gdnLayers);
        Assert.Equal(hc ? 32 : 0, hcPasses);
        AssertClose($"tsm {kernels}", tsm, ggml, hostExperts ? 1e-3 : 2e-4);
    }

    /// <summary>A prefill-width batch streams every expert an offloaded layer routes into its staging tensor. With a
    /// tier on unified memory, the experts the tier's fixed slots hold are copied from those slots instead of read
    /// from the file (ggml_ops_moe.cpp stage_experts_from_tier); the staged bytes are the same, so the prompt answers
    /// bit for bit as the untiered offload does - every expert in a slot, and half of them (copies and reads mixed).
    /// </summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void TieredExperts_StagePrefillFromTheirSlots_Metal()
    {
        float[][] plain = Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, LongPrompt);
        foreach (int divisor in new[] { 1, 2 })
        {
            float[][] tiered = Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, LongPrompt, model =>
            {
                var q = Assert.IsType<Qwen4ExpModel>(model);
                Assert.Equal(8, q.RegisterExpertTiersForTests(q.ExpertCountForTests / divisor, 0));
            });
            Assert.True(tiered[0].AsSpan().SequenceEqual(plain[0]),
                $"fixed slots for 1/{divisor} of the experts: the {LongPrompt.Length}-token prompt's staging changed its logits");
        }
    }

    /// <summary>A prompt-width pass stages the next offloaded layer's experts on a thread while the accelerator runs
    /// the current one (ggml_ops_moe.cpp moe_stage_prefetch_start), and the layer's MoE graph binds them instead of
    /// staging its own. The staged bytes are the same, so the prompt answers bit for bit as staging each layer when it
    /// runs - every expert offloaded, and with half of them in tier slots (copied from the slots ahead). The hit counter
    /// proves the prefetched stacks were the ones used.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void PromptExpertPrefetch_AnswersAsStagingEachLayer_Metal()
    {
        foreach (bool tiered in new[] { false, true })
        {
            float[][] Answer(string prefetch)
            {
                using var env = new NativeEnvScope();
                env.Set("TS_HOST_MOE_STAGE_PREFETCH", prefetch);
                return Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, LongPrompt, model =>
                {
                    if (!tiered) return;
                    var q = Assert.IsType<Qwen4ExpModel>(model);
                    Assert.Equal(8, q.RegisterExpertTiersForTests(q.ExpertCountForTests / 2, 0));
                });
            }
            float[][] eachLayer = Answer("0");
            long before = GgmlBasicOps.MoeStagePrefetchHits();
            float[][] prefetched = Answer("1");
            Assert.True(GgmlBasicOps.MoeStagePrefetchHits() - before >= 2,
                $"the {LongPrompt.Length}-token prompt's layers did not take prefetched experts (tiered={tiered})");
            for (int i = 0; i < prefetched.Length; i++)
                Assert.True(prefetched[i].AsSpan().SequenceEqual(eachLayer[i]),
                    $"row {i}: prefetched experts changed the logits (tiered={tiered})");
        }
    }

    /// <summary>A prompt-sized gather of the on-disk PLE n-gram table reads its rows with one pread each on the native
    /// I/O team (Qwen4ExpModel.ReadPleRowsPacked, TSGgml_HostReadRows) instead of faulting each row through the mapping.
    /// The rows are the same bytes, so the prompt answers bit for bit as reading the mapping; the I/O layer's read bytes
    /// prove the explicit reads ran.</summary>
    [GgmlFact(BackendType.GgmlMetal)]
    public void PleRows_ReadExplicitly_AnswerAsTheMapping_Metal()
    {
        float[][] Answer(string explicitRead, out long readBytes)
        {
            using var env = new NativeEnvScope();
            env.Set("TS_PLE_EXPLICIT_READ", explicitRead);
            long before = GgmlBasicOps.HostExpertIoStats().ReadBytes;
            float[][] rows = Run(BackendType.GgmlMetal, MoeCpuOffloadConfig.SetAllLayers, LongPrompt, model =>
            {
                var q = Assert.IsType<Qwen4ExpModel>(model);
                Assert.True(q.EnableHostExpertIoForTests(256L << 20), "the mapping could not be registered for explicit reads");
            });
            readBytes = GgmlBasicOps.HostExpertIoStats().ReadBytes - before;
            return rows;
        }
        float[][] mapping = Answer("0", out long mappingBytes);
        float[][] explicitRows = Answer("1", out long explicitBytes);
        Assert.True(explicitBytes > mappingBytes, $"no PLE rows were read explicitly ({explicitBytes} vs {mappingBytes} bytes)");
        for (int i = 0; i < explicitRows.Length; i++)
            Assert.True(explicitRows[i].AsSpan().SequenceEqual(mapping[i]), $"row {i}: explicitly read PLE rows changed the logits");
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
        // The overlap group is MUL_MAT_ID's; the grouped tsm kernels read a window's misses before they run.
        _nativeEnv.Set("TS_METAL_MOE", "0");
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

    /// <summary>Adaptive fixed slots (<c>TS_MOE_TIER_ADAPT=1</c>, discrete VRAM): 96 decode steps over a
    /// few repeating tokens concentrate the routing on experts the initial slots (ids 0..n/4-1) mostly
    /// do not hold. Every 32 seams the hottest of them are copied into the coldest slots, so more
    /// routes land in slots than with the slots fixed, and a swapped-in slot must hold exactly that
    /// expert's bytes: the logits stay within the host bound of the all-accelerator span.</summary>
    [GgmlFact(BackendType.GgmlCuda)]
    public void AdaptiveSlots_FollowTheRoutesAndStillAnswerAsTheAccelerator_Cuda()
    {
        int[] steps = Enumerable.Range(0, 96).Select(i => 40 + (i * 7) % 5).ToArray();
        float[][] device = Run(BackendType.GgmlCuda, () => { }, Prompt, steps, null, null);
        long fixedHits = 0, adaptiveHits = 0;
        void Register(ModelBase model)
        {
            var q = Assert.IsType<Qwen4ExpModel>(model);
            Assert.Equal(8, q.RegisterExpertTiersForTests(q.ExpertCountForTests / 4, 0));
        }
        float[][] fixedRows = Run(BackendType.GgmlCuda, MoeCpuOffloadConfig.SetAllLayers, Prompt, steps, Register,
            _ => fixedHits = GgmlBasicOps.MoeHotTierStats().Hits);
        float[][] adaptive;
        using (var env = new NativeEnvScope())
        {
            env.Set("TS_MOE_TIER_ADAPT", "1");
            adaptive = Run(BackendType.GgmlCuda, MoeCpuOffloadConfig.SetAllLayers, Prompt, steps, Register,
                _ => adaptiveHits = GgmlBasicOps.MoeHotTierStats().Hits);
        }
        _output.WriteLine($"slot hits: fixed {fixedHits}, adaptive {adaptiveHits}");
        Assert.True(adaptiveHits > fixedHits, $"adaptive slots held no more routes ({adaptiveHits} vs {fixedHits})");
        // 96 steps on random weights meet near-ties (one measured at a normal 5.7e-3 step): a
        // differing argmax must be one the step's own deviation explains.
        AssertClose("tier, fixed slots (long decode)", fixedRows, device, Bound(BackendType.GgmlCuda, hostPrefill: true),
            allowExplainedTies: true);
        AssertClose("tier, adaptive slots", adaptive, device, Bound(BackendType.GgmlCuda, hostPrefill: true),
            allowExplainedTies: true);
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

    /// <param name="allowExplainedTies">Accept a differing argmax when the expected logits rank the
    /// actual choice within twice the step's largest absolute deviation of the expected maximum: a
    /// near-tie the measured error explains, not a wrong answer.</param>
    private void AssertClose(string what, float[][] actual, float[][] expected, double bound,
        bool allowExplainedTies = false)
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
            int want = ArgMax(expected[i]), got = ArgMax(actual[i]);
            if (allowExplainedTies && want != got)
            {
                double deviation = 0;
                for (int v = 0; v < expected[i].Length; v++)
                    deviation = Math.Max(deviation, Math.Abs(actual[i][v] - expected[i][v]));
                double gap = expected[i][want] - expected[i][got];
                _output.WriteLine($"  near-tie: expected gap {gap:E3}, largest deviation {deviation:E3}");
                Assert.True(gap <= 2 * deviation, $"{what}: argmax {got} vs {want} with a gap of {gap:E3}, " +
                    $"more than twice the step's deviation {deviation:E3}");
                continue;
            }
            Assert.Equal(want, got);
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
