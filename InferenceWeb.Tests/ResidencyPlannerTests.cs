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

    // ---- streamed phases

    private static PhaseDemand[] Streamed(double conditionWork, double conditionLeast, double denoiseWork, double denoiseLeast) => new[]
    {
        new PhaseDemand(DiffusionPhase.Condition, C.Conditioner, (long)(conditionWork * GB)) { StreamedWeightBytes = (long)(conditionLeast * GB) },
        new PhaseDemand(DiffusionPhase.Denoise, C.Denoiser, (long)(denoiseWork * GB)) { StreamedWeightBytes = (long)(denoiseLeast * GB) },
        new PhaseDemand(DiffusionPhase.Decode, C.Vae, 1 * GB),
    };

    /// <summary>A component that can stream does not, where releasing the others makes the room: the plan is the one it was.</summary>
    [Fact]
    public void APhaseThatFitsOnceTheOthersAreReleasedDoesNotStream()
    {
        ResidencyPlan plain = ResidencyPlanner.Decide(TextToImage(3), Weights(18.2, 40.2, 1.2), Array.Empty<C>(), NoWork, A40);
        ResidencyPlan plan = ResidencyPlanner.Decide(Streamed(0.5, 1, 3, 1), Weights(18.2, 40.2, 1.2), Array.Empty<C>(), NoWork, A40);
        Assert.Empty(plan.Streams);
        Assert.Equal(plain.Releases, plan.Releases);
        Assert.Equal(plain.Peak, plan.Peak);
        Assert.True(ResidencyPlanner.Decide(Streamed(0.5, 1, 10, 1), Weights(18.2, 40.2, 1.2), Array.Empty<C>(), NoWork, 88 * GB).KeepsAll);
    }

    /// <summary>A phase past the budget with nothing else held streams within what the room leaves of it; its component holds
    /// nothing afterwards, so the next phase has the room it had; and the refusal is measured at the fewest it streams with.</summary>
    [Fact]
    public void APhasePastTheBudgetAloneStreamsWithinTheRoomAndHoldsNothingAfter()
    {
        // 16 GB: the 15.5 GB trunk beside 5 GB of encoding, the 12.7 GB DiT beside 9.5 GB of step.
        ResidencyPlan plan = ResidencyPlanner.Decide(Streamed(5, 1.2, 9.5, 0.9), Weights(15.5, 12.7, 1.2), Array.Empty<C>(), NoWork,
            16 * GB, capacity: 17 * GB);
        Assert.True(plan.Fits, plan.Describe());
        Assert.False(plan.Refused);
        Assert.Equal(new ResidencyStream(DiffusionPhase.Condition, C.Conditioner, 11 * GB), plan.Streams[0]);
        Assert.Equal(new ResidencyStream(DiffusionPhase.Denoise, C.Denoiser, (long)(6.5 * GB)), plan.Streams[1]);
        Assert.Equal(16 * GB, plan.Peak);
        // Nothing streamed is held before the denoise and the decode: no release names the conditioner or the denoiser.
        Assert.DoesNotContain(plan.Releases, r => r.Component is C.Conditioner or C.Denoiser);
        Assert.Contains("Denoise: stream the denoiser's weights 6.5 GB at a time", plan.Describe());
        Assert.Equal(DiffusionPhase.Denoise, plan.Limiting.Phase);
        Assert.Equal((long)(0.9 * GB), plan.LimitingWeights);
        Assert.Equal((long)(10.4 * GB), plan.MinimumPeak);

        // The same without streaming is refused at the DiT's whole weights.
        ResidencyPlan whole = ResidencyPlanner.Decide(Streamed(5, 0, 9.5, 0), Weights(15.5, 12.7, 1.2), Array.Empty<C>(), NoWork,
            16 * GB, capacity: 17 * GB);
        Assert.True(whole.Refused);
        Assert.Empty(whole.Streams);
        Assert.Equal(Weights(15.5, 12.7, 1.2)[C.Denoiser] + (long)(9.5 * GB), whole.MinimumPeak);
    }

    /// <summary>A step whose working memory leaves less room than the fewest weights it streams with streams at that fewest,
    /// past the budget (the device decides), and is refused past the device's capacity.</summary>
    [Fact]
    public void AStepPastTheBudgetEvenStreamedStreamsAtItsLeastOrIsRefused()
    {
        ResidencyPlan plan = ResidencyPlanner.Decide(Streamed(1, 1, 15.5, 0.9), Weights(15.5, 12.7, 1.2), Array.Empty<C>(), NoWork,
            16 * GB, capacity: 17 * GB);
        Assert.False(plan.Fits);
        Assert.False(plan.Refused);
        Assert.Equal((long)(0.9 * GB), plan.StreamBudget(DiffusionPhase.Denoise));
        ResidencyPlan refused = ResidencyPlanner.Decide(Streamed(1, 1, 16.5, 0.9), Weights(15.5, 12.7, 1.2), Array.Empty<C>(), NoWork,
            16 * GB, capacity: 17 * GB);
        Assert.True(refused.Refused);
        Assert.Contains("refused: Denoise needs 17.4 GB", refused.Describe());
    }
}
