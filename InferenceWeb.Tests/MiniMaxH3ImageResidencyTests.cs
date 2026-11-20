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
using TensorSharp.Models.QwenImage;
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

    /// <summary>The same for the unsloth Q3_K_M, UD-Q4_K_XL, Q5_K_M and Q6_K GGUFs (each file's tensor bytes plus the vectors'
    /// F32 widening, the offset that gives Q4_K_M's and Q8_0's figures above).</summary>
    private const long DitQ3 = 10_420_676_608, DitUd = 13_605_849_088, DitQ5 = 14_597_527_552, DitQ6 = 16_594_130_944;

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

    private static long DenoiserWorking(int side, bool fused, bool quantized, bool bf16Route = true) =>
        MiniMaxH3ImageDenoiser.WorkingBytesFor(Dit, MiniMaxH3ImageDenoiser.RowsOf(Workload(side)), fused, quantized, bf16Route);

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
        // ggml_cuda, RTX A4500 20 GB (09bd25fd, defaults): the denoise's peak per nvidia-smi less the context (191 MiB) and the
        // DiT. Files with Q4_K projections take their cuBLAS BF16 route; Q6_K is all MMQ (no BF16 activation copies).
        new object[] { "A4500 Q4_K_M 4096 denoise", 5984L, DenoiserWorking(4096, fused: true, quantized: true) },
        new object[] { "A4500 Q4_K_M 4480 denoise", 7028L, DenoiserWorking(4480, fused: true, quantized: true) },
        new object[] { "A4500 Q3_K_M 4480 denoise", 6964L, DenoiserWorking(4480, fused: true, quantized: true) },
        new object[] { "A4500 UD-Q4_K_XL 4096 denoise", 5918L, DenoiserWorking(4096, fused: true, quantized: true) },
        new object[] { "A4500 Q5_K_M 3840 denoise", 5313L, DenoiserWorking(3840, fused: true, quantized: true) },
        new object[] { "A4500 Q6_K 2880 denoise", 2843L, DenoiserWorking(2880, fused: true, quantized: true, bf16Route: false) },
        new object[] { "A4500 Q6_K 3072 denoise", 3175L, DenoiserWorking(3072, fused: true, quantized: true, bf16Route: false) },
        new object[] { "A4500 Q6_K 3328 denoise", 3643L, DenoiserWorking(3328, fused: true, quantized: true, bf16Route: false) },
        // ggml_cuda, A40 (stage S5, c011de25): bf16 with 4 references, the conditioner and the VAE released before the denoise,
        // so the DiT alone is resident: 44,879,249,408 bytes peak less the context and the DiT's 40,233,055,464.
        new object[] { "A40 bf16 2048 edit, 4 references, denoise", 4160L, EditDenoiserWorking(4, fused: true, quantized: false) },
    };

    /// <summary>A 1024x1024 reference's vision copy (its size is all the estimate reads).</summary>
    private static readonly RgbImage VisionCopy = new(1024, 1024, new float[1024 * 1024 * 3]);

    /// <summary>The refs4 prompt of S5's edits (276 characters).</summary>
    private static readonly string EditPrompt = new('x', 276);

    /// <summary>An edit at 2048x2048 with <paramref name="references"/> 1024x1024 pictures (1,024 VAE rows and 1,024 merged
    /// vision tokens each), as stage S5 measured them.</summary>
    private static DiffusionWorkload EditWorkload(int references, string prompt = null) =>
        new(new LatentShape(24, 1, 128, 128), Enumerable.Repeat(new LatentShape(24, 1, 64, 64), references).ToArray(), prompt ?? EditPrompt)
        {
            Pictures = Enumerable.Range(0, references).Select(i => new PresentedImage(VisionCopy, i)).ToArray(),
        };

    private static long EditDenoiserWorking(int references, bool fused, bool quantized, bool bf16Route = true, string prompt = null) =>
        MiniMaxH3ImageDenoiser.WorkingBytesFor(Dit, MiniMaxH3ImageDenoiser.RowsOf(EditWorkload(references, prompt)), fused, quantized,
            bf16Route, edit: true);

    /// <summary>
    /// Stage S5's A40 Q4_K_M edit with 16 references: everything stayed resident, its plan said a 44.2 GB peak and its decode
    /// peaked at 45,611,155,456 bytes less the 271 MiB context (45.33 GB), 1.1 GB over. The edit term the denoiser's estimate
    /// gained raises that plan by what it adds to the step (the DiT's workspace is held through the decode), which covers
    /// the measured peak; and the estimate with it stays the text-to-image formula for a request with no references.
    /// </summary>
    [Fact]
    public void TheEditTermCoversTheSixteenReferencePeakAndLeavesTextToImageAlone()
    {
        const long planPeak = 44_150_000_000L, measured = 45_611_155_456L - 271 * MiB;
        const string refs16Prompt = "A white museum wall with sixteen framed pictures hung in a neat 4 by 4 grid, shown straight on. " +
            "From the top left, row by row, the frames hold <Picture 1>, <Picture 2>, <Picture 3>, <Picture 4>, <Picture 5>, " +
            "<Picture 6>, <Picture 7>, <Picture 8>, <Picture 9>, <Picture 10>, <Picture 11>, <Picture 12>, <Picture 13>, " +
            "<Picture 14>, <Picture 15>, <Picture 16>, each picture complete inside its own thin black frame.";
        long rows = MiniMaxH3ImageDenoiser.RowsOf(EditWorkload(16, refs16Prompt));
        long added = MiniMaxH3ImageDenoiser.WorkingBytesFor(Dit, rows, fused: true, quantized: true, edit: true)
                     - MiniMaxH3ImageDenoiser.WorkingBytesFor(Dit, rows, fused: true, quantized: true);
        Assert.True(planPeak + added >= measured, $"plan {planPeak / 1e9:F2} + {added / 1e9:F2} GB < measured {measured / 1e9:F2} GB");
        Assert.Equal(DenoiserWorking(2048, fused: true, quantized: true),
            MiniMaxH3ImageDenoiser.WorkingBytesFor(Dit, MiniMaxH3ImageDenoiser.RowsOf(Workload(2048)), fused: true, quantized: true, edit: false));
    }

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

    /// <summary>The RTX A4500 20 GB (ggml reports 20,044 MiB; the CUDA context 191 MiB).</summary>
    private static readonly long A4500 = CudaBudget(20044, 191);

    // ---- what each device can give a phase at most (free bytes with no headroom): the refusal's capacity

    private static readonly long A40Free = (46068 - 271) * MiB, H100Free = (95830 - 271) * MiB, A4500Free = (20044 - 191) * MiB;
    private const long M5ProFree = 40_200_900_000L;

    private static ResidencyPlan Plan(long ditBytes, int side, bool fused, bool quantized, long budget,
        IReadOnlyCollection<DiffusionComponent> resident = null, IReadOnlyDictionary<DiffusionComponent, long> workspaces = null,
        long? capacity = null, bool bf16Route = true)
    {
        var phases = new List<PhaseDemand>
        {
            new(DiffusionPhase.Condition, DiffusionComponent.Conditioner, ConditionerWorking),
            new(DiffusionPhase.Denoise, DiffusionComponent.Denoiser, DenoiserWorking(side, fused, quantized, bf16Route)),
            new(DiffusionPhase.Decode, DiffusionComponent.Vae, VaeWorking(side)),
        };
        var weights = new Dictionary<DiffusionComponent, long>
        {
            [DiffusionComponent.Conditioner] = TextEncoderBytes,
            [DiffusionComponent.Denoiser] = ditBytes,
            [DiffusionComponent.Vae] = VaeBytes,
        };
        return ResidencyPlanner.Decide(phases, weights, resident ?? Array.Empty<DiffusionComponent>(),
            workspaces ?? new Dictionary<DiffusionComponent, long>(), budget, capacity);
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
    /// its step's 6.7 GB beside 38.3 GB of weights still fits the denoise, but not the decode beside it: the step's plan
    /// goes before the decode (only compute buffers are lost; every weight stays).</summary>
    [Theory]
    [InlineData(2048)]
    [InlineData(4096)]
    public void QuantizedDitsOnTheA40KeepEverythingResidentWhereTheyFit(int side)
    {
        Assert.True(Plan(DitQ4, side, fused: true, quantized: true, A40).KeepsAll);
        // Q8_0 has no Q4_K projection: its products run on MMQ, without the cuBLAS route's BF16 copies.
        ResidencyPlan q8 = Plan(DitQ8, side, fused: true, quantized: true, A40, bf16Route: false);
        if (side == 2048) Assert.True(q8.KeepsAll, q8.Describe());
        else Assert.Equal("Decode:Denoiser workspace", Releases(q8));
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

    // ---- the refusal (ResidencyPlan.Refused): a request whose most demanding phase alone cannot fit the device

    /// <summary>Every request run on the RTX A4500 with the release files (09bd25fd, ggml_cuda, defaults, the portrait prompt):
    /// those that produced their picture and those that failed inside the denoise (ggml-cuda's MMQ out of memory, which ends the
    /// process, or a compute allocation that failed). The refusal must refuse exactly the failures: every picture that was made
    /// stays allowed (the closest 0.22 GB under the free memory) and every failure is refused (the closest 0.30 GB over).</summary>
    public static IEnumerable<object[]> A4500Runs() => new[]
    {
        // file, DiT bytes, side, Q4_K on cuBLAS BF16, made its picture (peak MiB)
        new object[] { "Q3_K_M", DitQ3, 4480, true, true },          // 17,093 MiB
        new object[] { "Q4_K_M", DitQ4, 4096, true, true },          // 18,305 MiB
        new object[] { "Q4_K_M", DitQ4, 4480, true, true },          // 19,349 MiB (the plan was over its budget)
        new object[] { "UD-Q4_K_XL", DitUd, 3584, true, true },      // 17,847 MiB
        new object[] { "UD-Q4_K_XL", DitUd, 4096, true, true },      // 19,085 MiB
        new object[] { "UD-Q4_K_XL", DitUd, 4480, true, false },     // MMQ out of memory in the first step
        new object[] { "Q5_K_M", DitQ5, 3584, true, true },          // 18,827 MiB
        new object[] { "Q5_K_M", DitQ5, 3840, true, true },          // 19,425 MiB
        new object[] { "Q5_K_M", DitQ5, 4096, true, false },         // MMQ out of memory in the first step
        new object[] { "Q6_K", DitQ6, 2880, false, true },           // 18,859 MiB
        new object[] { "Q6_K", DitQ6, 3072, false, true },           // 19,191 MiB
        new object[] { "Q6_K", DitQ6, 3328, false, true },           // 19,659 MiB
        new object[] { "Q6_K", DitQ6, 3584, false, false },          // out of memory in the first step
        new object[] { "Q8_0", DitQ8, 2048, false, false },          // the DiT alone does not fit: the compute allocation failed
    };

    [Theory]
    [MemberData(nameof(A4500Runs))]
    public void OnTheA4500ExactlyTheRequestsThatFailedAreRefused(string file, long ditBytes, int side, bool bf16Route, bool ran)
    {
        ResidencyPlan plan = Plan(ditBytes, side, fused: true, quantized: true, A4500, capacity: A4500Free, bf16Route: bf16Route);
        Assert.True(plan.Refused == !ran, $"{file} at {side}: {plan.Describe()} (minimum peak {plan.MinimumPeak / 1e9:F2} GB)");
        if (!ran) Assert.Equal(DiffusionPhase.Denoise, plan.Limiting.Phase);
    }

    /// <summary>The validated runs elsewhere are never refused: bf16 on the A40 at both sizes (4096x4096 ran at 42.6 GiB with its
    /// plan over the budget), the quantized DiTs on the A40, an H100 with everything, and Q8_0 on the 48 GB Mac. bf16 on that
    /// Mac is (its 40.2 GB of weights alone are the whole working set).</summary>
    [Fact]
    public void TheValidatedRunsAreNotRefused()
    {
        foreach (int side in new[] { 2048, 4096 })
        {
            Assert.False(Plan(DitBf16, side, fused: true, quantized: false, A40, capacity: A40Free).Refused, $"A40 bf16 {side}");
            Assert.False(Plan(DitQ4, side, fused: true, quantized: true, A40, capacity: A40Free).Refused, $"A40 Q4_K_M {side}");
            Assert.False(Plan(DitQ8, side, fused: true, quantized: true, A40, capacity: A40Free, bf16Route: false).Refused, $"A40 Q8_0 {side}");
            Assert.False(Plan(DitBf16, side, fused: true, quantized: false, H100, capacity: H100Free).Refused, $"H100 bf16 {side}");
        }
        Assert.False(Plan(DitQ8, 2048, fused: false, quantized: true, M5Pro, capacity: M5ProFree).Refused);
        Assert.True(Plan(DitBf16, 2048, fused: false, quantized: false, M5Pro, capacity: M5ProFree).Refused);
    }
}
