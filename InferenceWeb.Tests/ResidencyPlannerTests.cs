// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The diffusion runtime's residency decision (docs/design/diffusion-runtime.md section 3.4) at the sizes the design
// names, in GB: MiniMax-H3-Image's Q4_K_M text encoder 18.2, the bf16 DiT 40.2 (Q8_0 22.0, Q4_K_M 12.7), the VAE 1.2.
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime.Scheduling.Diffusion;
using C = TensorSharp.Models.SharedDiffusion.DiffusionComponent;

namespace InferenceWeb.Tests;

public sealed class ResidencyPlannerTests
{
    private const long GB = 1_000_000_000;

    private static Dictionary<C, long> Weights(double conditioner, double denoiser, double vae) => new()
    {
        [C.Conditioner] = (long)(conditioner * GB), [C.Denoiser] = (long)(denoiser * GB), [C.Vae] = (long)(vae * GB),
    };

    private static PhaseDemand[] TextToImage(double denoiseWork, double conditionWork = 0.5, double decodeWork = 1) => new[]
    {
        new PhaseDemand(DiffusionPhase.Condition, C.Conditioner, (long)(conditionWork * GB)),
        new PhaseDemand(DiffusionPhase.Denoise, C.Denoiser, (long)(denoiseWork * GB)),
        new PhaseDemand(DiffusionPhase.Decode, C.Vae, (long)(decodeWork * GB)),
    };

    private static readonly Dictionary<C, long> NoWork = new();

    [Fact]
    public void AnH100KeepsEveryComponentAsTodaysPipelineDoes()
    {
        // 94 GB less 6% headroom: 18.2 + 40.2 + 1.2 + activations fit.
        ResidencyPlan plan = ResidencyPlanner.Decide(TextToImage(10), Weights(18.2, 40.2, 1.2), Array.Empty<C>(), NoWork, 88 * GB);
        Assert.True(plan.KeepsAll);
        Assert.True(plan.Fits);
        Assert.Empty(plan.Releases);
        Assert.StartsWith("keep every component resident", plan.Describe());
    }

    [Fact]
    public void AnUnknownBudgetKeepsEverything()
    {
        ResidencyPlan plan = ResidencyPlanner.Decide(TextToImage(10), Weights(18.2, 40.2, 1.2), Array.Empty<C>(), NoWork, budget: null);
        Assert.True(plan.KeepsAll);
        Assert.Null(plan.Budget);
    }

    // An A40's room for the components (46 GB, the design's "tight" bf16 case: the DiT and 3 GB of 4 MP activations).
    private const long A40 = 45_500_000_000;

    [Fact]
    public void AnA40ReleasesTheTextEncoderBeforeTheBf16Denoiser()
    {
        ResidencyPlan plan = ResidencyPlanner.Decide(TextToImage(3), Weights(18.2, 40.2, 1.2), Array.Empty<C>(), NoWork, A40);
        Assert.False(plan.KeepsAll);
        Assert.True(plan.Fits);
        Assert.Equal(new[] { new ResidencyRelease(DiffusionPhase.Denoise, C.Conditioner, Weights: true) }, plan.Releases);
        Assert.True(plan.Peak <= A40);
    }

    [Fact]
    public void TheNextRequestMakesRoomForTheTextEncoderByReleasingTheDenoiserNotTheVae()
    {
        // Steady state after a request: the denoiser and the VAE resident with their workspaces.
        var work = new Dictionary<C, long> { [C.Denoiser] = 3 * GB, [C.Vae] = 1 * GB };
        ResidencyPlan plan = ResidencyPlanner.Decide(TextToImage(3), Weights(18.2, 40.2, 1.2), new[] { C.Denoiser, C.Vae }, work, A40);
        Assert.Equal(new[]
        {
            new ResidencyRelease(DiffusionPhase.Condition, C.Denoiser, Weights: true),
            new ResidencyRelease(DiffusionPhase.Denoise, C.Conditioner, Weights: true),
        }, plan.Releases);
    }

    [Fact]
    public void ThePro4500TierFitsTheQ4DenoiserBesideTheVaeButNotTheTextEncoder()
    {
        // 32 GB less 2 GB headroom: Q4_K_M DiT 12.7 + text encoder 18.2 + 3 GB of 4 MP activations does not fit.
        ResidencyPlan plan = ResidencyPlanner.Decide(TextToImage(3), Weights(18.2, 12.7, 1.2), Array.Empty<C>(), NoWork, 30 * GB);
        Assert.Equal(new[] { new ResidencyRelease(DiffusionPhase.Denoise, C.Conditioner, Weights: true) }, plan.Releases);
    }

    [Fact]
    public void AWorkspaceGoesAloneWhenThatMakesTheRoom()
    {
        // The weights fit together; a cached 12 GB denoise workspace beside the references' encode does not, nor the
        // encode's workspace beside the denoise: only compute buffers are dropped, never weights.
        var work = new Dictionary<C, long> { [C.Denoiser] = 12 * GB };
        var phases = new[]
        {
            new PhaseDemand(DiffusionPhase.EncodeReferences, C.Vae, 6 * GB),
            new PhaseDemand(DiffusionPhase.Denoise, C.Denoiser, 12 * GB),
        };
        ResidencyPlan plan = ResidencyPlanner.Decide(phases, Weights(0, 16, 1), new[] { C.Denoiser }, work, 30 * GB);
        Assert.True(plan.Fits);
        Assert.Equal(new[]
        {
            new ResidencyRelease(DiffusionPhase.EncodeReferences, C.Denoiser, Weights: false),
            new ResidencyRelease(DiffusionPhase.Denoise, C.Vae, Weights: false),
        }, plan.Releases);
        Assert.Contains("before EncodeReferences: release the denoiser's workspace", plan.Describe());
    }

    [Fact]
    public void APhaseThatAloneExceedsTheBudgetIsReportedAndKeepsAsLittleAsItCan()
    {
        ResidencyPlan plan = ResidencyPlanner.Decide(TextToImage(3), Weights(18.2, 40.2, 1.2), Array.Empty<C>(), NoWork, 30 * GB);
        Assert.False(plan.Fits);
        Assert.Contains(new ResidencyRelease(DiffusionPhase.Denoise, C.Conditioner, Weights: true), plan.Releases);
        Assert.Contains("a phase alone exceeds the budget", plan.Describe());
    }

    [Fact]
    public void ThePlannerReadsTheSpareBytesOnceAndCountsWhatTheComponentsHold()
    {
        int reads = 0;
        var log = new List<string>();
        var planner = new ResidencyPlanner(() => { reads++; return 2 * GB; }, log.Add);
        // 2 GB free plus the 40.2 + 1.2 + 4 the denoiser and VAE hold now.
        var work = new Dictionary<C, long> { [C.Denoiser] = 3 * GB, [C.Vae] = 1 * GB };
        Dictionary<C, long> weights = Weights(18.2, 40.2, 1.2);
        ResidencyPlan plan = planner.Plan(TextToImage(3), weights, new[] { C.Denoiser, C.Vae }, work, "2048x2048");
        Assert.Equal(1, reads);
        Assert.Equal(2 * GB + weights[C.Denoiser] + weights[C.Vae] + 4 * GB, plan.Budget);
        Assert.Equal(C.Denoiser, plan.Releases[0].Component);
        // The same decision for the same geometry class is logged once, whatever the free memory.
        planner.Plan(TextToImage(3), Weights(18.2, 40.2, 1.2), new[] { C.Denoiser, C.Vae }, work, "2048x2048");
        Assert.Single(log);
        planner.Plan(TextToImage(3), Weights(18.2, 40.2, 1.2), new[] { C.Denoiser, C.Vae }, work, "4096x4096");
        Assert.Equal(2, log.Count);
        Assert.StartsWith("[diffusion-residency] 4096x4096: before Condition: release the denoiser", log[1]);
    }
}
