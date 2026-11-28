// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Threading;
using TensorSharp.GGML;
using TensorSharp.Models.MiniMaxH3;
using TensorSharp.Models.MiniMaxH3Image;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// Streamed phases (ResidencyPlan.Streams): a phase whose component's weights and working memory do not fit the device even
/// with every other component released holds a group of its weights at a time. MiniMax-H3-Image's text encoder streams its
/// trunk as layer groups and its denoiser a step's blocks as spans; both must compute what the whole component computes, bit
/// for bit, and the pipeline must hand each streamed phase its budget and release its component after it.
/// </summary>
public sealed class MiniMaxH3ImageStreamingTests(ITestOutputHelper output)
{
    private static readonly float[] Grid = MiniMaxH3ImageSampling.ReleaseSigmas;

    private static void AssertBits(float[] want, float[] got, string what)
    {
        Assert.Equal(want.Length, got.Length);
        for (int i = 0; i < want.Length; i++)
            if (BitConverter.SingleToInt32Bits(want[i]) != BitConverter.SingleToInt32Bits(got[i]))
                Assert.Fail($"{what}: value {i} is {got[i]:R}, the whole component's {want[i]:R}");
    }

    // ---- the text encoder's layer groups

    /// <summary>A tiny four-layer GQA trunk (the MiniMax-H3 encoder's tensor names), its tokenizer files beside it.</summary>
    private static string WriteTinyEncoder(string dir, int layers, int hidden, int vocab)
    {
        const int heads = 4, kvHeads = 2, headDim = 16, ffn = 96;
        var rng = new Random(91);
        var tensors = new List<MiniMaxH3ImageFixture.T>();
        void Add(string name, float scale, float offset, params long[] shape) =>
            tensors.Add(new MiniMaxH3ImageFixture.T(name, shape, MiniMaxH3ImageFixture.F32,
                Enumerable.Range(0, checked((int)shape.Aggregate(1L, (a, b) => a * b)))
                    .Select(_ => offset + scale * (float)(rng.NextDouble() * 2 - 1)).ToArray()));
        Add("model.embed_tokens.weight", 1f, 0f, hidden, vocab);
        for (int l = 0; l < layers; l++)
        {
            string p = $"model.layers.{l}.";
            Add(p + "input_layernorm.weight", 0.2f, 1f, hidden);
            Add(p + "post_attention_layernorm.weight", 0.2f, 1f, hidden);
            Add(p + "self_attn.q_norm.weight", 0.2f, 1f, headDim);
            Add(p + "self_attn.k_norm.weight", 0.2f, 1f, headDim);
            Add(p + "self_attn.q_proj.weight", 0.3f, 0f, hidden, heads * headDim);
            Add(p + "self_attn.k_proj.weight", 0.3f, 0f, hidden, kvHeads * headDim);
            Add(p + "self_attn.v_proj.weight", 0.3f, 0f, hidden, kvHeads * headDim);
            Add(p + "self_attn.o_proj.weight", 0.15f, 0f, heads * headDim, hidden);
            Add(p + "mlp.gate_proj.weight", 0.15f, 0f, hidden, ffn);
            Add(p + "mlp.up_proj.weight", 0.15f, 0f, hidden, ffn);
            Add(p + "mlp.down_proj.weight", 0.1f, 0f, ffn, hidden);
        }
        string path = Path.Combine(dir, "te-tiny.gguf");
        MiniMaxH3ImageFixture.Write(path, tensors);
        // One character a token (byte-level BPE with no merges): a prompt of these characters tokenizes to them.
        File.WriteAllText(Path.Combine(dir, "vocab.json"), "{" + string.Join(",", Enumerable.Range(0, vocab).Select(i => $"\"{VocabChars[i]}\":{i}")) + "}");
        File.WriteAllText(Path.Combine(dir, "merges.txt"), "#version: 0.2\n");
        return path;
    }

    private const string VocabChars = "abcdefghijklmnopqrstuvwxyz0123456789-.,!";

    /// <summary>
    /// The trunk within a device budget (MiniMaxH3TextEncoder.DeviceWeightBudget) runs as layer groups whose hidden state
    /// crosses as the F32 it is inside one call: the same bytes as the whole trunk, for a prompt and for an edit's (a picture's
    /// merged rows and its three DeepStack taps, which the first group always holds), with the edit's attention in query
    /// blocks as an edit runs it.
    /// </summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void TheTrunkInLayerGroupsIsTheWholeTrunkBitForBit()
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cpu);
        string dir = Path.Combine(Path.GetTempPath(), "ts-h3te-groups-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            const int layers = 5, hidden = 64, vocab = 40;   // VocabChars.Length
            using var te = new MiniMaxH3TextEncoder(WriteTinyEncoder(dir, layers, hidden, vocab), dir, BackendType.GgmlCpu, null);
            long layer = te.LargestLayerBytes;
            Assert.Equal(layers * layer, te.DeviceWeightBytes);
            int[] ids = Enumerable.Range(0, 29).Select(i => (i * 7 + 3) % vocab).ToArray();

            float[] whole = te.Encode(ids, null);
            Assert.Equal(new[] { layers }, te.LastLayerGroups);
            foreach ((long budget, int[] groups) in new[] { (layer, new[] { 1, 1, 1, 1, 1 }), (2 * layer + 1, new[] { 2, 2, 1 }),
                                                            (layers * layer, new[] { layers }) })
            {
                te.DeviceWeightBudget = budget;
                float[] grouped = te.Encode(ids, null);
                Assert.Equal(groups, te.LastLayerGroups);
                AssertBits(whole, grouped, $"the trunk in groups of {string.Join(",", groups)}");
            }

            // An edit's sequence: a 3x4-merged-token picture at token 4, its DeepStack taps after layers 0-2.
            var rng = new Random(5);
            float[] Rows(int n) => Enumerable.Range(0, n * hidden).Select(_ => (float)(rng.NextDouble() - 0.5)).ToArray();
            var picture = new MiniMaxH3PromptImage
            {
                TokenIndex = 4,
                Vision = new MiniMaxH3VisionOutput
                {
                    Merged = Rows(12), DeepStack = new[] { Rows(12), Rows(12), Rows(12) },
                    TokenCount = 12, GridHeight = 6, GridWidth = 8,
                },
            };
            te.DeviceWeightBudget = null;
            float[] wholeEdit = te.Encode(ids, new[] { picture }, queryChunk: 7);
            te.DeviceWeightBudget = layer;
            float[] groupedEdit = te.Encode(ids, new[] { picture }, queryChunk: 7);
            Assert.Equal(new[] { 3, 2 }, te.LastLayerGroups);
            AssertBits(wholeEdit, groupedEdit, "an edit's trunk in groups of three layers");
            output.WriteLine($"{ids.Length} tokens through {layers} layers of {layer:N0} bytes: grouped = whole, bit for bit");
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    /// <summary>MiniMax-H3-Image's conditioner hands the plan's budget to the trunk: on ggml_cpu it streams as low as one layer
    /// for a prompt (the fewest it reports), and the conditioning is the resident encode's, before and after a resident one.</summary>
    [GgmlFact(BackendType.GgmlCpu)]
    public void TheConditionerStreamsWithinThePlansBudgetToTheSameConditioning()
    {
        GgmlBasicOps.EnsureBackendAvailable(GgmlBackendType.Cpu);
        string dir = Path.Combine(Path.GetTempPath(), "ts-h3te-cond-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            const int layers = 4, hidden = 64;
            string path = WriteTinyEncoder(dir, layers, hidden, VocabChars.Length);
            using var conditioner = new MiniMaxH3ImageConditioner(path, BackendType.GgmlCpu);
            const string prompt = "a-red-lantern,in.snow!";
            var workload = new DiffusionWorkload(new LatentShape(24, 1, 8, 8), Array.Empty<LatentShape>(), prompt);
            long least = conditioner.StreamedWeightBytes(workload, DiffusionPhase.Condition);
            Assert.Equal(conditioner.WeightBytes / layers, least);
            Assert.Equal(0, conditioner.StreamedWeightBytes(workload, DiffusionPhase.Denoise));

            ConditioningSet whole = conditioner.Encode(prompt, Array.Empty<PresentedImage>(), CancellationToken.None);
            Assert.Equal(prompt.Length, whole.Rows);
            conditioner.StreamWithin(least);
            ConditioningSet streamed = conditioner.Encode(prompt, Array.Empty<PresentedImage>(), CancellationToken.None);
            AssertBits(whole.Hidden, streamed.Hidden, "the conditioning streamed a layer at a time");
            conditioner.ReleaseDevice();
            conditioner.StreamWithin(null);
            AssertBits(whole.Hidden, conditioner.Encode(prompt, Array.Empty<PresentedImage>(), CancellationToken.None).Hidden,
                "the resident conditioning after a streamed one");
        }
        finally
        {
            try { Directory.Delete(dir, recursive: true); } catch (IOException) { } catch (UnauthorizedAccessException) { }
        }
    }

    // ---- the denoiser's block spans

    /// <summary>Spans balance nothing: each is as many blocks as the budget holds beside the LoRA factors, one at least, and a
    /// budget the whole step fits is the whole step.</summary>
    [PinnedGgmlServesFact(MiniMaxH3ImageFamily.ArchitectureId)]
    public void TheDenoisersSpansAreWholeBlocksWithinTheBudget()
    {
        using var fx = new MiniMaxH3ImageFixture(MiniMaxH3ImageFixture.Q8_0);
        using var dit = new MiniMaxH3ImageDenoiser(fx.DitPath, TestGates.PinnedGgmlBackend);
        var workload = new DiffusionWorkload(new LatentShape(24, 1, 8, 8), Array.Empty<LatentShape>(), "a");
        long least = dit.StreamedWeightBytes(workload, DiffusionPhase.Denoise);
        if (TestGates.PinnedGgmlBackend is not (BackendType.GgmlCpu or BackendType.GgmlMetal))
        {
            Assert.Equal(0, least);   // ggml_cuda and ggml_vulkan keep the step's weights resident
            return;
        }
        Assert.InRange(least, 1, dit.WeightBytes - 1);
        Assert.Equal(0, dit.StreamedWeightBytes(workload, DiffusionPhase.Condition));
        DenoiserSpan[] one = dit.StreamSpans(least);
        Assert.Equal(MiniMaxH3ImageFixture.Blocks, one.Length);
        Assert.True(one[0].IsFirst && one[^1].IsLast);
        for (int i = 0; i < one.Length; i++) Assert.Equal((i, i + 1), (one[i].First, one[i].End));
        Assert.Equal(new[] { DenoiserSpan.Whole(MiniMaxH3ImageFixture.Blocks) }, dit.StreamSpans(dit.WeightBytes));
    }

    /// <summary>
    /// A streamed step (MiniMaxH3ImageDenoiser.StreamWithin): its blocks as one-shot programs over the residual stream the span
    /// before handed on, each span's weights released once it has run. Six steps of an edit with two references against the
    /// whole step's, bit for bit, with a resident step before and after (its weights go first, and come back).
    /// </summary>
    [PinnedGgmlServesFact(MiniMaxH3ImageFamily.ArchitectureId)]
    public void AStreamedStepIsTheWholeStepBitForBit()
    {
        if (TestGates.PinnedGgmlBackend is not (BackendType.GgmlCpu or BackendType.GgmlMetal)) return;
        foreach (int format in new[] { MiniMaxH3ImageFixture.Q8_0, MiniMaxH3ImageFixture.Q4_K })
        {
            using var fx = new MiniMaxH3ImageFixture(format);
            var family = MiniMaxH3ImageFamily.ForDenoiser(fx.DitPath);
            using var whole = new MiniMaxH3ImageDenoiser(fx.DitPath, TestGates.PinnedGgmlBackend);
            using var streamed = new MiniMaxH3ImageDenoiser(fx.DitPath, TestGates.PinnedGgmlBackend);
            MiniMaxH3ImageGoldens.Request rq = MiniMaxH3ImageGoldens.Request.WithReferences;

            // A resident step first: the streamed steps must hand its copies back before their first span.
            MiniMaxH3ImageGoldens.Run(family, streamed, rq, steps: 1);
            Assert.Single(streamed.LastSpans);
            var workload = new DiffusionWorkload(new LatentShape(24, 1, 8, 8), Array.Empty<LatentShape>(), "a");
            streamed.StreamWithin(streamed.StreamedWeightBytes(workload, DiffusionPhase.Denoise));
            var (want, got) = MiniMaxH3ImageGoldens.Run(family, whole, rq, also: streamed);
            Assert.Equal(MiniMaxH3ImageFixture.Blocks, streamed.LastSpans.Length);
            for (int i = 0; i < want.Count; i++) AssertBits(want[i], got[i], $"{MiniMaxH3ImageFixture.Name(format)} step {i + 1}");
            streamed.ReleaseWorkspace();
            streamed.ReleaseDevice();

            // Back to the whole step: the same velocities again.
            streamed.StreamWithin(null);
            var (_, again) = MiniMaxH3ImageGoldens.Run(family, whole, rq, also: streamed);
            Assert.Single(streamed.LastSpans);
            for (int i = 0; i < want.Count; i++) AssertBits(want[i], again[i], $"{MiniMaxH3ImageFixture.Name(format)} resident step {i + 1}");
            output.WriteLine($"{MiniMaxH3ImageFixture.Name(format)}: {want.Count} streamed steps over {MiniMaxH3ImageFixture.Blocks} spans = whole steps, bit for bit");
        }
    }

    // ---- the pipeline

    /// <summary>
    /// The pipeline hands each phase its plan's stream budget (null where it holds every weight) and releases a streamed
    /// phase's component after it, so the next request plans from what the device really holds. The request: a two-picture
    /// edit with <c>&lt;Picture 1&gt;</c> masked on a device whose budget holds neither the conditioner nor the denoiser beside
    /// its working memory; without streaming it is refused, with it it runs every step.
    /// </summary>
    [Fact]
    public void ThePipelineStreamsThePhasesItsPlanStreamsAndReleasesTheirComponents()
    {
        const long GiB = 1L << 30;
        RgbImage source = Picture(512, 512, 1);
        var mask = new RgbImage(512, 512, Enumerable.Range(0, 512 * 512 * 3).Select(i => (i / 3) % 512 < 256 ? 1f : 0f).ToArray());
        var request = new DiffusionRequest
        {
            Prompt = "Put the lamp of <Picture 2> on the table of <Picture 1>.",
            References = new[] { new DiffusionReference(source), new DiffusionReference(Picture(640, 480, 2)) },
            Mask = new EditMaskOptions { Mask = mask },
            KeepSourceSize = true,
            Seed = 11,
        };
        // 16 GiB of room (17 free): a 15 GiB conditioner beside 4 GiB of encoding, a 12 GiB denoiser beside 8 GiB of step.
        StreamingFamily Family(bool streams) => new(new MiniMaxH3ImageFamily(Grid), streams)
        {
            ConditionerWeights = 15 * GiB, ConditionerWork = 4 * GiB, DenoiserWeights = 12 * GiB, DenoiserWork = 8 * GiB,
            Least = 1 * GiB,
        };
        ResidencyPlanner Planner(List<string> log) => new(() => 16 * GiB, log.Add, freeBytes: () => 17 * GiB);

        StreamingFamily fixedFamily = Family(streams: false);
        using (var refusing = new DiffusionPipeline(fixedFamily, new DiffusionLoad { Backend = BackendType.GgmlMetal }, Planner(new()), _ => { }))
            Assert.Throws<RequestRefusedException>(() => refusing.Run(request));
        Assert.Equal(0, fixedFamily.Steps);

        StreamingFamily family = Family(streams: true);
        var log = new List<string>();
        using var pipeline = new DiffusionPipeline(family, new DiffusionLoad { Backend = BackendType.GgmlMetal }, Planner(log), log.Add);
        DiffusionResult result = pipeline.Run(request);
        foreach (string line in log.Where(l => l.Contains("residency"))) output.WriteLine(line);
        Assert.Equal(6, family.Steps);
        Assert.Equal((512, 512), (result.Image.Width, result.Image.Height));
        // The conditioner held 16 - 4 GiB of its weights at once, the denoiser 16 - 8; the VAE every weight, in both its phases.
        Assert.Equal(new long?[] { 12 * GiB }, family.ConditionerBudgets.ToArray());
        Assert.Equal(new long?[] { 8 * GiB }, family.DenoiserBudgets.ToArray());
        Assert.Equal(new long?[] { null, null }, family.VaeBudgets.ToArray());
        Assert.Equal(1, family.ConditionerReleases);
        Assert.Equal(1, family.DenoiserReleases);
        Assert.False(pipeline.IsResident(DiffusionComponent.Conditioner));
        Assert.False(pipeline.IsResident(DiffusionComponent.Denoiser));
        Assert.True(pipeline.IsResident(DiffusionComponent.Vae));
        Assert.Contains(log, l => l.Contains("Condition: stream the conditioner's weights"));
        Assert.Contains(log, l => l.Contains("Denoise: the denoiser streams its weights"));
    }

    private static RgbImage Picture(int width, int height, int seed)
    {
        var px = new float[(long)width * height * 3];
        for (long i = 0; i < px.LongLength; i++) px[i] = (byte)((i * 2654435761L + seed * 97) >> 9) / 255f;
        return new RgbImage(width, height, px);
    }

    /// <summary>MiniMax-H3-Image's family over test components that declare their weights and working memory, can stream
    /// (or not), and record the budgets the pipeline hands them and their releases.</summary>
    private sealed class StreamingFamily(MiniMaxH3ImageFamily inner, bool streams) : IDiffusionFamily
    {
        public long ConditionerWeights { get; init; }
        public long ConditionerWork { get; init; }
        public long DenoiserWeights { get; init; }
        public long DenoiserWork { get; init; }
        public long Least { get; init; }
        public bool Streams => streams;
        public int Steps;
        public int ConditionerReleases, DenoiserReleases;
        public List<long?> ConditionerBudgets { get; } = new();
        public List<long?> DenoiserBudgets { get; } = new();
        public List<long?> VaeBudgets { get; } = new();

        public string Id => inner.Id;
        public string DisplayName => inner.DisplayName;
        public DiffusionGeometry Geometry => inner.Geometry;
        public DiffusionGuidance Guidance => inner.Guidance;
        public SamplingRecipe DefaultRecipe => inner.DefaultRecipe;
        public IReferencePolicy References => inner.References;
        public IConditioner CreateConditioner(DiffusionLoad load) => new Conditioner(this);
        public IDenoiser CreateDenoiser(DiffusionLoad load) => new Denoiser(this);
        public IVaeCodec CreateVae(DiffusionLoad load) => new Vae(this);
        public INoiseSource Noise(long seed) => inner.Noise(seed);
        public DiffusionCanvas ResolveCanvas(DiffusionRequest request, EditMask mask) => inner.ResolveCanvas(request, mask);
        public DiffusionLayout Pack(ConditioningSet text, IReadOnlyList<LatentRef> references, LatentShape target, StepState step) =>
            inner.Pack(text, references, target, step);
        public float[] ToRows(LatentTensor latent) => inner.ToRows(latent);
        public LatentTensor FromRows(float[] rows, LatentShape shape) => inner.FromRows(rows, shape);
        public (float[] Weights, int ValuesPerWeight) MaskRows(float[] cellWeights, LatentShape shape) => inner.MaskRows(cellWeights, shape);

        private sealed class Conditioner(StreamingFamily f) : IConditioner
        {
            public long WeightBytes => f.ConditionerWeights;
            public long WorkingBytes(DiffusionWorkload workload) => f.ConditionerWork;
            public long StreamedWeightBytes(DiffusionWorkload workload, DiffusionPhase phase) => f.Streams ? f.Least : 0;
            public void StreamWithin(long? weightBytes) => f.ConditionerBudgets.Add(weightBytes);
            public void ReleaseDevice() => f.ConditionerReleases++;
            public void ReleaseWorkspace() { }
            public void Dispose() { }
            public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken ct)
            {
                int rows = 5 + pictures.Count * 3;
                var tags = new int[rows];
                for (int i = 0; i < rows; i++) tags[i] = i < 5 ? 1 : 0;
                return new ConditioningSet(new float[rows * 8], rows, 8, tags);
            }
        }

        private sealed class Denoiser(StreamingFamily f) : IDenoiser
        {
            public DenoiserCaps Caps { get; } = new();
            public long WeightBytes => f.DenoiserWeights;
            public long WorkingBytes(DiffusionWorkload workload) => f.DenoiserWork;
            public long StreamedWeightBytes(DiffusionWorkload workload, DiffusionPhase phase) => f.Streams ? f.Least : 0;
            public void StreamWithin(long? weightBytes) => f.DenoiserBudgets.Add(weightBytes);
            public void ReleaseDevice() => f.DenoiserReleases++;
            public void ReleaseWorkspace() { }
            public void Dispose() { }
            public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken ct)
            {
                f.Steps++;
                return new StepOutput(new float[inputs.Latent.Length]);
            }
        }

        private sealed class Vae(StreamingFamily f) : IVaeCodec
        {
            public long WeightBytes => 1L << 30;
            public long WorkingBytes(DiffusionWorkload workload) => 1L << 29;
            public void StreamWithin(long? weightBytes) => f.VaeBudgets.Add(weightBytes);
            public void ReleaseDevice() { }
            public void ReleaseWorkspace() { }
            public void Dispose() { }
            public LatentPosterior Encode(RgbImage image, CancellationToken ct) =>
                new(new LatentShape(24, 1, image.Height / 16, image.Width / 16),
                    new float[24L * (image.Height / 16) * (image.Width / 16)], new float[24L * (image.Height / 16) * (image.Width / 16)]);
            public RgbImage Decode(LatentTensor latent, CancellationToken ct) =>
                new(latent.Shape.Width * 16, latent.Shape.Height * 16, new float[(long)latent.Shape.Width * 16 * latent.Shape.Height * 16 * 3]);
            public RgbImage Preview(LatentTensor latent, int maxSide) => Decode(latent, CancellationToken.None);
        }
    }
}
