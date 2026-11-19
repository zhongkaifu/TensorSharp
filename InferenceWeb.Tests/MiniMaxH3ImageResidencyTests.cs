// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime.Scheduling.Diffusion;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// MiniMax-H3-Image's phase residency on the devices it was validated on (stage S4 of docs/design/diffusion-runtime.md):
/// the components' working-memory estimates against the device peaks measured with the release files (the eng
/// MiniMaxH3ImageAB runs: a phase's peak device use less the context and the resident weights), and the planner's decision
/// for each validated device and file from those estimates, its components' declared bytes and the device's budget.
/// No weights are read: the release dimensions and byte counts are the headers'.
/// </summary>
public sealed class MiniMaxH3ImageResidencyTests
{
    // ---- the release files (their headers)

    /// <summary>The release DiT (hidden 5376, 50 blocks, 56 x 128 heads, FFN 14336).</summary>
    private static readonly MiniMaxH3Config Dit = new()
    {
        HiddenSize = 5376, NumLayers = 50, TokenRefinerLayers = 2, NumAttentionHeads = 56, AttentionHeadDim = 128,
        FfnHiddenSize = 14336, VideoLatentChannels = 24, PddSteps = 6, TextDim = 5120,
    };

    /// <summary>The Qwen3-VL-32B trunk MiniMax-H3 truncates to 50 layers.</summary>
    private static readonly MiniMaxH3TextEncoderConfig TextEncoder = new()
    {
        NumLayers = 50, Hidden = 5120, Heads = 64, KvHeads = 8, HeadDim = 128, Intermediate = 25600,
    };

    /// <summary>MiniMaxH3TextEncoder.DeviceWeightBytes of qwen3vl_32b_minimax_h3-Q4_K_M.gguf.</summary>
    private const long TextEncoderBytes = 15_472_691_200;

    /// <summary>MiniMaxH3ImageDenoiser.WeightBytes of the bf16 safetensors and the unsloth Q8_0 and Q4_K_M GGUFs.</summary>
    private const long DitBf16 = 40_233_055_464, DitQ8 = 22_025_181_184, DitQ4 = 12_719_540_224;

    /// <summary>MiniMaxH3ImageVaeCodec.WeightBytes of minimax_h3_image_vae_fp16.safetensors (the decoder only).</summary>
    private const long VaeBytes = 808_632_544;

    private const int VaeDim = 1024, VaeInner = 4096;
    private const long MiB = 1024 * 1024;

    /// <summary>The prompt of the measured runs (prompts.json's portrait, 59 tokens).</summary>
    private const string Portrait =
        "A photographic close-up portrait of an elderly fisherman with a weathered, deeply lined face and a short grey beard, " +
        "wearing a knitted navy wool cap, soft window light from the left, shallow depth of field, 85mm lens, detailed skin " +
        "texture and catchlights in his eyes";

    private static DiffusionWorkload Workload(int side) =>
        new(new LatentShape(24, 1, side / 16, side / 16), Array.Empty<LatentShape>(), Portrait);

    private static long DenoiserWorking(int side, bool fused, bool quantized) =>
        MiniMaxH3ImageDenoiser.WorkingBytesFor(Dit, MiniMaxH3ImageDenoiser.RowsOf(Workload(side)), fused, quantized);

    private static long ConditionerWorking => MiniMaxH3ImageConditioner.WorkingBytesFor(TextEncoder, Portrait);

    private static long VaeWorking(int side) => MiniMaxH3ImageVaeCodec.WorkingBytesFor(VaeDim, VaeInner, side);

    /// <summary>Measured working memory (MiB) and the estimate's component: a phase's peak device use less the context and
    /// the weights resident during it.</summary>
    public static IEnumerable<object[]> Measured() => new[]
    {
        // ggml_cuda, A40: device used per nvidia-smi and the tool's sampler (context 271 MiB).
        new object[] { "A40 bf16 2048 denoise", 1078L, DenoiserWorking(2048, fused: true, quantized: false) },
        new object[] { "A40 Q4_K_M 2048 denoise", 1666L, DenoiserWorking(2048, fused: true, quantized: true) },
        new object[] { "A40 Q4_K_M 4096 denoise", 5642L, DenoiserWorking(4096, fused: true, quantized: true) },
        new object[] { "A40 text encoder", 615L, ConditionerWorking },
        new object[] { "A40 bf16 2048 decode", 123L, VaeWorking(2048) },
        // ggml_metal, M5 Pro: the device's allocated size.
        new object[] { "M5 Pro Q8_0 2048 denoise", 2946L, DenoiserWorking(2048, fused: false, quantized: true) },
        new object[] { "M5 Pro Q4_K_M 512 denoise", 234L, MiniMaxH3ImageDenoiser.WorkingBytesFor(Dit,
            MiniMaxH3ImageDenoiser.RowsOf(new DiffusionWorkload(new LatentShape(24, 1, 32, 32), Array.Empty<LatentShape>(),
                "A red paper lantern glowing in a snowy courtyard at dusk, a wooden sign that reads \"冬至\", and above the door a painted shop banner that reads \"WINTER SOLSTICE TEA HOUSE\"")),
            fused: false, quantized: true) },
        new object[] { "M5 Pro Q8_0 2048 decode", 167L, VaeWorking(2048) },
    };

    /// <summary>An estimate never falls short of what was measured (the plan would then overcommit the device), and stays
    /// within 2.5x of it (so a device that holds everything is not told otherwise).</summary>
    [Theory]
    [MemberData(nameof(Measured))]
    public void EstimatesCoverTheMeasuredWorkingMemory(string cell, long measuredMiB, long estimate)
    {
        double ratio = estimate / (double)(measuredMiB * MiB);
        Assert.True(ratio >= 1.0, $"{cell}: estimate {estimate / (double)MiB:F0} MiB is below the measured {measuredMiB} MiB");
        Assert.True(ratio <= 2.5, $"{cell}: estimate {estimate / (double)MiB:F0} MiB is {ratio:F2}x the measured {measuredMiB} MiB");
    }

    // ---- the devices' budgets (GpuMemoryBudget: free less max(512 MiB, total / 16))

    private static long CudaBudget(long totalMiB, long contextMiB) => (totalMiB - contextMiB) * MiB - Math.Max(512 * MiB, totalMiB * MiB / 16);

    /// <summary>The A40 46 GB (46,068 MiB, the CUDA context 271 MiB).</summary>
    private static readonly long A40 = CudaBudget(46068, 271);

    /// <summary>An H100 NVL 94 GB (95,830 MiB), the context as on the A40.</summary>
    private static readonly long H100 = CudaBudget(95830, 271);

    /// <summary>The 48 GB M5 Pro's Metal recommended working set (40,200.90 MB), nothing allocated yet.</summary>
    private static readonly long M5Pro = 40_200_900_000L - Math.Max(512 * MiB, 40_200_900_000L / 16);

    private static ResidencyPlan Plan(long ditBytes, int side, bool fused, bool quantized, long budget,
        IReadOnlyCollection<DiffusionComponent> resident = null, IReadOnlyDictionary<DiffusionComponent, long> workspaces = null)
    {
        var phases = new List<PhaseDemand>
        {
            new(DiffusionPhase.Condition, DiffusionComponent.Conditioner, ConditionerWorking),
            new(DiffusionPhase.Denoise, DiffusionComponent.Denoiser, DenoiserWorking(side, fused, quantized)),
            new(DiffusionPhase.Decode, DiffusionComponent.Vae, VaeWorking(side)),
        };
        var weights = new Dictionary<DiffusionComponent, long>
        {
            [DiffusionComponent.Conditioner] = TextEncoderBytes,
            [DiffusionComponent.Denoiser] = ditBytes,
            [DiffusionComponent.Vae] = VaeBytes,
        };
        return ResidencyPlanner.Decide(phases, weights, resident ?? Array.Empty<DiffusionComponent>(),
            workspaces ?? new Dictionary<DiffusionComponent, long>(), budget);
    }

    private static string Releases(ResidencyPlan plan) =>
        string.Join("; ", plan.Releases.Select(r => $"{r.Before}:{r.Component}{(r.Weights ? "" : " workspace")}"));

    /// <summary>bf16 on the A40, the case the bf16 verification ran with a local patch: the text encoder goes before the
    /// denoise (15.6 GB -> 0.34 GB in 0.8 s, measured), the DiT stays through the decode.</summary>
    [Fact]
    public void Bf16OnTheA40ReleasesTheTextEncoderBeforeTheDenoise()
    {
        ResidencyPlan plan = Plan(DitBf16, 2048, fused: true, quantized: false, A40);
        Assert.False(plan.KeepsAll);
        Assert.Equal("Denoise:Conditioner", Releases(plan));
        Assert.True(plan.Fits, plan.Describe());
        // The second picture: the text encoder comes back in place of the DiT, which comes back for the denoise.
        var resident = new[] { DiffusionComponent.Denoiser, DiffusionComponent.Vae };
        var held = new Dictionary<DiffusionComponent, long>
        {
            [DiffusionComponent.Denoiser] = DenoiserWorking(2048, true, false),
            [DiffusionComponent.Vae] = VaeWorking(2048),
        };
        ResidencyPlan second = Plan(DitBf16, 2048, fused: true, quantized: false, A40, resident, held);
        Assert.Equal("Condition:Denoiser; Denoise:Conditioner", Releases(second));
    }

    /// <summary>bf16 at 4096 on the A40: the denoise alone is at the budget's edge (it ran: 44.2 GB of 46 GB, measured), and the
    /// DiT's workspace goes before the decode.</summary>
    [Fact]
    public void Bf16At4096OnTheA40AlsoDropsTheDenoisersWorkspaceBeforeTheDecode()
    {
        ResidencyPlan plan = Plan(DitBf16, 4096, fused: true, quantized: false, A40);
        Assert.StartsWith("Denoise:Conditioner", Releases(plan));
        Assert.Contains("Decode:Denoiser workspace", Releases(plan));
    }

    /// <summary>Q4_K_M keeps everything on the A40 at both sizes (as the legacy model did); Q8_0 does at 4 MP, and at 16 MP
    /// its step's 6.7 GB beside 37.5 GB of weights passes the budget's headroom, so the text encoder goes there.</summary>
    [Theory]
    [InlineData(2048)]
    [InlineData(4096)]
    public void QuantizedDitsOnTheA40KeepEverythingResidentWhereTheyFit(int side)
    {
        Assert.True(Plan(DitQ4, side, fused: true, quantized: true, A40).KeepsAll);
        ResidencyPlan q8 = Plan(DitQ8, side, fused: true, quantized: true, A40);
        if (side == 2048) Assert.True(q8.KeepsAll, q8.Describe());
        else Assert.Equal("Denoise:Conditioner", Releases(q8));
    }

    /// <summary>An H100-class card holds everything, bf16 at 16 MP included: no release, as the legacy model ran there.</summary>
    [Theory]
    [InlineData(2048)]
    [InlineData(4096)]
    public void AnH100KeepsEverythingResident(int side)
    {
        ResidencyPlan plan = Plan(DitBf16, side, fused: true, quantized: false, H100);
        Assert.True(plan.KeepsAll, plan.Describe());
    }

    /// <summary>The 48 GB Mac: Q8_0 cannot sit beside the text encoder (the legacy model was killed at 43.5 GB wired), Q4_K_M
    /// can at 4 MP but not at 16 MP, where the unfused step's activations are about 12 GB.</summary>
    [Fact]
    public void TheM5ProReleasesTheTextEncoderWhereTheDitAndItsStepDoNotFitBesideIt()
    {
        Assert.Equal("Denoise:Conditioner", Releases(Plan(DitQ8, 2048, fused: false, quantized: true, M5Pro)));
        Assert.True(Plan(DitQ4, 2048, fused: false, quantized: true, M5Pro).KeepsAll);
        Assert.Equal("Denoise:Conditioner", Releases(Plan(DitQ4, 4096, fused: false, quantized: true, M5Pro)));
        // bf16's weights alone exceed the working set: the plan says so (the run is left to the device).
        Assert.False(Plan(DitBf16, 2048, fused: false, quantized: false, M5Pro).Fits);
    }
}
