// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source
// tree.
//
// The diffusion runtime's pipeline on a fake family (no weights, no native code): the phases and their order, the
// request's noise stream and its order of draws, the Euler loop, guidance, the sampling contract, masked edits,
// previews, cancellation and residency. The fake's arithmetic is simple enough to redo here independently.
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling.Diffusion;

namespace InferenceWeb.Tests;

public sealed class DiffusionPipelineTests
{
    // ---- the fake family: 2 latent channels, 4-pixel cells, 1x1 tokens, rows cell-major (HWC)

    private sealed class FakeComponent : IResidentComponent
    {
        private readonly FakeFamily _family;
        private readonly string _name;

        public FakeComponent(FakeFamily family, string name, long weights, long working)
        {
            _family = family; _name = name; WeightBytes = weights; Working = working;
        }

        public long WeightBytes { get; }
        public long Working { get; }
        public long WorkingBytes(DiffusionWorkload workload) => Working;
        public void ReleaseDevice() => _family.Events.Add($"release {_name}");
        public void ReleaseWorkspace() => _family.Events.Add($"release {_name} workspace");
        public void Dispose() => _family.Events.Add($"dispose {_name}");
    }

    private sealed class FakeConditioner(FakeFamily family) : IConditioner
    {
        private readonly FakeComponent _c = new(family, "conditioner", family.Bytes.Conditioner, family.Bytes.ConditionWork);
        public long WeightBytes => _c.WeightBytes;
        public long WorkingBytes(DiffusionWorkload w) => _c.WorkingBytes(w);
        public void ReleaseDevice() => _c.ReleaseDevice();
        public void ReleaseWorkspace() => _c.ReleaseWorkspace();
        public void Dispose() => _c.Dispose();

        public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken ct)
        {
            family.Events.Add($"condition '{prompt}' {pictures.Count}");
            // One row of width 2: the prompt's length and the pictures' count, so branches differ.
            return new ConditioningSet(new[] { prompt.Length * 0.25f, pictures.Count }, 1, 2);
        }
    }

    private sealed class FakeDenoiser(FakeFamily family) : IDenoiser
    {
        private readonly FakeComponent _c = new(family, "denoiser", family.Bytes.Denoiser, family.Bytes.DenoiseWork);
        private float[] _buffer;
        public long WeightBytes => _c.WeightBytes;
        public long WorkingBytes(DiffusionWorkload w) => _c.WorkingBytes(w);
        public void ReleaseDevice() => _c.ReleaseDevice();
        public void ReleaseWorkspace() => _c.ReleaseWorkspace();
        public void Dispose() => _c.Dispose();
        public DenoiserCaps Caps { get; } = new();
        public List<StepState> Steps { get; } = new();

        public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken ct)
        {
            family.Events.Add($"step {inputs.Step.Index}");
            Steps.Add(inputs.Step);
            // Its own buffer, rewritten by every call, as a graph plan's outputs are.
            if (_buffer?.Length != inputs.Latent.Length) _buffer = new float[inputs.Latent.Length];
            for (int k = 0; k < _buffer.Length; k++)
                _buffer[k] = Velocity(inputs.Latent[k], inputs.Conditioning.Hidden[0], inputs.Step.Index);
            return new StepOutput(_buffer);
        }

        public static float Velocity(float x, float condition, int step) => x * 0.5f - condition + step * 0.125f;
    }

    private sealed class FakeVae(FakeFamily family) : IVaeCodec
    {
        private readonly FakeComponent _c = new(family, "vae", family.Bytes.Vae, family.Bytes.VaeWork);
        public long WeightBytes => _c.WeightBytes;
        public long WorkingBytes(DiffusionWorkload w) => _c.WorkingBytes(w);
        public void ReleaseDevice() => _c.ReleaseDevice();
        public void ReleaseWorkspace() => _c.ReleaseWorkspace();
        public void Dispose() => _c.Dispose();
        public LatentTensor LastDecoded { get; private set; }
        public int Previews { get; private set; }

        public LatentPosterior Encode(RgbImage image, CancellationToken ct)
        {
            family.Events.Add($"encode {image.Width}x{image.Height}");
            LatentShape shape = family.Geometry.LatentOf(image.Width, image.Height);
            var mean = new float[shape.Count];
            for (int c = 0; c < shape.Channels; c++)
                for (int y = 0; y < shape.Height; y++)
                    for (int x = 0; x < shape.Width; x++)
                        mean[(c * shape.Height + y) * shape.Width + x] = image.Pixels[((y * 4) * image.Width + x * 4) * 3] + c;
            return new LatentPosterior(shape, mean, Enumerable.Repeat(-2f, mean.Length).ToArray());
        }

        public RgbImage Decode(LatentTensor latent, CancellationToken ct)
        {
            family.Events.Add("decode");
            LastDecoded = latent;
            return Paint(latent);
        }

        public RgbImage Preview(LatentTensor latent, int maxSide)
        {
            Previews++;
            return Paint(latent);
        }

        // Each 4x4 pixel block is the first channel of its cell, squashed into [0, 1].
        private static RgbImage Paint(LatentTensor latent)
        {
            LatentShape s = latent.Shape;
            int w = s.Width * 4, h = s.Height * 4;
            var pixels = new float[w * h * 3];
            for (int y = 0; y < h; y++)
                for (int x = 0; x < w; x++)
                {
                    float v = 1f / (1f + MathF.Exp(-latent.Data[(y / 4) * s.Width + x / 4]));
                    for (int c = 0; c < 3; c++) pixels[(y * w + x) * 3 + c] = v;
                }
            return new RgbImage(w, h, pixels);
        }
    }

    private sealed class RecordingNoise(FakeFamily family, long seed) : INoiseSource
    {
        private readonly TorchCpuRandn _stream = new(unchecked((ulong)seed));

        public float[] Next(long count)
        {
            family.Events.Add($"noise {count}");
            return _stream.Next(count);
        }
    }

    private sealed class FakePolicy(FakeFamily family) : IReferencePolicy
    {
        public PresentedImage ForVision(RgbImage image, int index) => new(image, index);
        public RgbImage ForVae(RgbImage image, int index) => image;

        public LatentRef FromPosterior(LatentPosterior posterior, int index, INoiseSource noise)
        {
            family.Events.Add($"posterior {index}");
            LatentTensor sample = posterior.Sample(noise.Next(posterior.Shape.Count));
            return new LatentRef(index, posterior.Shape, family.ToRows(sample));
        }

        public void Augment(LatentRef reference, INoiseSource noise)
        {
            family.Events.Add($"augment {reference.Index}");
            float[] eps = noise.Next(reference.Rows.Length);
            for (int i = 0; i < eps.Length; i++) reference.Rows[i] = 0.999f * reference.Rows[i] + 0.001f * eps[i];
        }
    }

    private sealed record FakeBytes(long Conditioner, long Denoiser, long Vae, long ConditionWork, long DenoiseWork, long VaeWork);

    private sealed class FakeFamily : IDiffusionFamily
    {
        public FakeFamily()
        {
            References = new FakePolicy(this);
        }

        public List<string> Events { get; } = new();
        public FakeBytes Bytes { get; init; } = new(18, 40, 1, 1, 3, 1);
        public int Created { get; private set; }
        public FakeConditioner Conditioner { get; private set; }
        public FakeDenoiser Denoiser { get; private set; }
        public FakeVae Vae { get; private set; }

        public string Id => "fake-diffusion";
        public string DisplayName => "Fake";
        public DiffusionGeometry Geometry { get; init; } = new()
        {
            LatentChannels = 2, VaeFactor = 4, Patch = 1, SizeMultiple = 8, DefaultWidth = 32, DefaultHeight = 16, MaxReferences = 2,
        };
        public DiffusionGuidance Guidance { get; init; } = DiffusionGuidance.None;
        public SamplingRecipe DefaultRecipe { get; init; } = new()
        {
            Schedule = DiffusionSchedule.ComfySimpleOverGrid, Steps = 3, Grid = new[] { 1f, 0.7f, 0.3f, 0f }, Heads = HeadPolicy.GridInterval,
        };
        public IReferencePolicy References { get; }

        public IConditioner CreateConditioner(DiffusionLoad load) { Created++; return Conditioner = new FakeConditioner(this); }
        public IDenoiser CreateDenoiser(DiffusionLoad load) { Created++; return Denoiser = new FakeDenoiser(this); }
        public IVaeCodec CreateVae(DiffusionLoad load) { Created++; return Vae = new FakeVae(this); }
        public INoiseSource Noise(long seed) => new RecordingNoise(this, seed);

        public DiffusionCanvas ResolveCanvas(DiffusionRequest request, EditMask mask)
        {
            if (mask != null)
            {
                var (w, h) = mask.SamplingDimensions(mask.Source.Width, mask.Source.Height, Geometry.SizeMultiple);
                return new DiffusionCanvas((w, h), (mask.Source.Width, mask.Source.Height));
            }
            return DiffusionCanvas.TextToImage(request, Geometry, DisplayName);
        }

        public DiffusionLayout Pack(ConditioningSet text, IReadOnlyList<LatentRef> references, LatentShape target, StepState step)
        {
            int referenceRows = references.Sum(r => r.Rows.Length / Geometry.ValuesPerToken);
            int targetRows = target.Height * target.Width;
            return new DiffusionLayout { Rows = text.Rows + referenceRows + targetRows, TargetStart = text.Rows + referenceRows, TargetRows = targetRows };
        }

        public float[] ToRows(LatentTensor latent) => ToHwc(latent.Data, latent.Shape);

        public LatentTensor FromRows(float[] rows, LatentShape shape)
        {
            var chw = new float[shape.Count];
            int plane = shape.Height * shape.Width;
            for (int c = 0; c < shape.Channels; c++)
                for (int i = 0; i < plane; i++)
                    chw[c * plane + i] = rows[i * shape.Channels + c];
            return new LatentTensor(shape, chw);
        }

        public (float[] Weights, int ValuesPerWeight) MaskRows(float[] cellWeights, LatentShape shape) => (cellWeights, shape.Channels);

        public static float[] ToHwc(float[] chw, LatentShape shape)
        {
            var rows = new float[shape.Count];
            int plane = shape.Height * shape.Width;
            for (int c = 0; c < shape.Channels; c++)
                for (int i = 0; i < plane; i++)
                    rows[i * shape.Channels + c] = chw[c * plane + i];
            return rows;
        }
    }

    private static DiffusionPipeline Pipeline(FakeFamily family, long? spare = null, List<string> log = null)
        => new(family, new DiffusionLoad { Backend = BackendType.Cpu },
            new ResidencyPlanner(() => spare, line => log?.Add(line)), line => log?.Add(line));

    private static RgbImage Picture(int width, int height, Func<int, int, float> value)
    {
        var pixels = new float[width * height * 3];
        for (int y = 0; y < height; y++)
            for (int x = 0; x < width; x++)
                for (int c = 0; c < 3; c++) pixels[(y * width + x) * 3 + c] = value(x, y);
        return new RgbImage(width, height, pixels);
    }

    private static uint[] Bits(float[] values) => values.Select(BitConverter.SingleToUInt32Bits).ToArray();

    // ---- text to image

    [Fact]
    public void TextToImageDrawsOnlyTheTargetNoiseAndRunsTheEulerLoopOnTheRecipesSigmas()
    {
        var family = new FakeFamily();
        var progress = new List<DiffusionProgress>();
        using var pipeline = Pipeline(family);
        DiffusionResult result = pipeline.Run(new DiffusionRequest { Prompt = "a cat", Seed = 1234, Progress = progress.Add });

        // 32x16 pixels: 8x4 cells of 2 channels, drawn once from a fresh generator, as the legacy pipeline did.
        var shape = new LatentShape(2, 1, 4, 8);
        float[] x = FakeFamily.ToHwc(TorchCpuRandn.Normal(shape.Count, 1234), shape);
        float[] sigmas = DiffusionSchedules.ComfySimpleOverGrid(family.DefaultRecipe.Grid, 3);
        float condition = "a cat".Length * 0.25f;
        for (int i = 0; i < 3; i++)
        {
            float dt = sigmas[i + 1] - sigmas[i];
            for (int k = 0; k < x.Length; k++) x[k] += FakeDenoiser.Velocity(x[k], condition, i) * dt;
        }
        Assert.Equal(Bits(family.FromRows(x, shape).Data), Bits(family.Vae.LastDecoded.Data));
        Assert.Equal((32, 16), (result.Image.Width, result.Image.Height));

        Assert.Equal(new[] { "condition 'a cat' 0", "noise 64", "step 0", "step 1", "step 2", "decode" },
            family.Events.Where(e => !e.StartsWith("release")).ToArray());
        Assert.Equal(new[]
        {
            new DiffusionProgress(DiffusionPhase.Prepare), new DiffusionProgress(DiffusionPhase.Condition),
            new DiffusionProgress(DiffusionPhase.Denoise),
            new DiffusionProgress(DiffusionPhase.Denoise, 1, 3), new DiffusionProgress(DiffusionPhase.Denoise, 2, 3),
            new DiffusionProgress(DiffusionPhase.Denoise, 3, 3),
            new DiffusionProgress(DiffusionPhase.Decode), new DiffusionProgress(DiffusionPhase.Composite),
        }, progress);
        Assert.Equal(new[] { (0, 1), (1, 2), (2, 3) }, family.Denoiser.Steps.Select(s => (s.HeadStart, s.HeadStop)));
        Assert.Equal(Bits(sigmas), Bits(result.Recipe.Sigmas));
        Assert.Equal(3, result.Recipe.Steps);
        Assert.Equal(3, result.Timings.Steps.Count);
        Assert.True(result.Residency.KeepsAll);
        Assert.Null(result.Residency.Budget);
    }

    [Fact]
    public void ARunIsReproducibleAndTheSeedChangesIt()
    {
        float[] Run(long seed)
        {
            var family = new FakeFamily();
            using var pipeline = Pipeline(family);
            pipeline.Run(new DiffusionRequest { Prompt = "p", Seed = seed });
            return family.Vae.LastDecoded.Data;
        }
        Assert.Equal(Run(7), Run(7));
        Assert.NotEqual(Run(7), Run(8));
    }

    [Fact]
    public void ComponentsAreMadeOnceAndKeptBetweenRequests()
    {
        var family = new FakeFamily();
        using (var pipeline = Pipeline(family))
        {
            pipeline.Run(new DiffusionRequest { Prompt = "one" });
            pipeline.Run(new DiffusionRequest { Prompt = "two", Width = 16, Height = 8 });
            Assert.Equal(3, family.Created);
            Assert.True(pipeline.IsResident(DiffusionComponent.Denoiser));
            Assert.DoesNotContain(family.Events, e => e.StartsWith("release"));
        }
        Assert.Contains("dispose denoiser", family.Events);
        Assert.Contains("dispose conditioner", family.Events);
    }

    [Fact]
    public void BetweenStepsRunsBeforeEveryStepButTheFirst()
    {
        var family = new FakeFamily();
        using var pipeline = Pipeline(family);
        int calls = 0;
        pipeline.Run(new DiffusionRequest { Prompt = "p" }, default, () =>
        {
            calls++;
            family.Events.Add("checkpoint");
        });
        Assert.Equal(2, calls);
        Assert.Equal(new[] { "step 0", "checkpoint", "step 1", "checkpoint", "step 2" },
            family.Events.Where(e => e.StartsWith("step") || e == "checkpoint").ToArray());
    }

    // ---- refusals

    [Fact]
    public void ADistilledFamilyRefusesGuidanceAndANegativePrompt()
    {
        var family = new FakeFamily();
        using var pipeline = Pipeline(family);
        Assert.Equal("Fake is guidance-distilled: cfg must be 1 (or left unset).",
            Assert.Throws<ArgumentException>(() => pipeline.Run(new DiffusionRequest { Cfg = 2f })).Message);
        Assert.Equal("Fake takes no negative prompt.",
            Assert.Throws<ArgumentException>(() => pipeline.Run(new DiffusionRequest { NegativePrompt = "blurry" })).Message);
        Assert.Throws<ArgumentException>(() => pipeline.Run(new DiffusionRequest { Steps = -1 }));
        Assert.Throws<ArgumentException>(() => pipeline.Run(new DiffusionRequest { Width = 32 }));
        Assert.Throws<ArgumentException>(() => pipeline.Run(new DiffusionRequest { Width = 36, Height = 16 }));
        // Refused before any component is made or any noise drawn.
        Assert.Equal(0, family.Created);
        Assert.Empty(family.Events);
        // Whitespace is no negative prompt; guidance 1 is the default.
        pipeline.Run(new DiffusionRequest { Cfg = 1f, NegativePrompt = " " });
    }

    [Fact]
    public void ReferencesBeyondTheFamilysCapAreRefused()
    {
        var family = new FakeFamily();
        using var pipeline = Pipeline(family);
        var picture = new DiffusionReference(Picture(32, 16, (_, _) => .5f));
        var error = Assert.Throws<ArgumentException>(() => pipeline.Run(new DiffusionRequest { References = new[] { picture, picture, picture } }));
        Assert.Equal("Fake takes at most 2 reference images; this request has 3.", error.Message);
        var textOnly = new FakeFamily { Geometry = family.Geometry with { MaxReferences = 0 } };
        using var other = Pipeline(textOnly);
        Assert.Equal("Fake does not edit images; drop the reference images.",
            Assert.Throws<ArgumentException>(() => other.Run(new DiffusionRequest { References = new[] { picture } })).Message);
    }

    // ---- references and the stream's order of draws

    [Fact]
    public void ReferencesDrawTheirPosteriorsBeforeTheTargetNoiseAndAugmentAfterIt()
    {
        var family = new FakeFamily();
        using var pipeline = Pipeline(family);
        var progress = new List<DiffusionPhase>();
        pipeline.Run(new DiffusionRequest
        {
            Prompt = "edit",
            References = new[] { new DiffusionReference(Picture(16, 8, (x, y) => x * .01f)), new DiffusionReference(Picture(8, 16, (_, _) => .2f)) },
            Seed = 5,
            Progress = p => progress.Add(p.Phase),
        });
        Assert.Equal(new[]
        {
            "condition 'edit' 2",
            "encode 16x8", "posterior 0", "noise 16",
            "encode 8x16", "posterior 1", "noise 16",
            "noise 64",
            "augment 0", "noise 16", "augment 1", "noise 16",
            "step 0", "step 1", "step 2", "decode",
        }, family.Events.ToArray());
        Assert.Equal(new[] { DiffusionPhase.Prepare, DiffusionPhase.Condition, DiffusionPhase.EncodeReferences, DiffusionPhase.Denoise },
            progress.Distinct().Take(4));
    }

    [Fact]
    public void AnEditsTargetNoiseIsNotTheTextToImageNoiseAtTheSameSeed()
    {
        // diffusers' order: the posterior samples come first, so an edit never restarts from its source's noise.
        var t2i = new FakeFamily();
        using (var p = Pipeline(t2i)) p.Run(new DiffusionRequest { Seed = 9 });
        var edit = new FakeFamily();
        using (var p = Pipeline(edit))
            p.Run(new DiffusionRequest { Seed = 9, References = new[] { new DiffusionReference(Picture(32, 16, (_, _) => 0f)) } });
        Assert.NotEqual(t2i.Vae.LastDecoded.Data, edit.Vae.LastDecoded.Data);
    }

    // ---- guidance and contracts

    [Fact]
    public void AGuidedFamilyRunsBothBranchesAndCombinesThem()
    {
        var family = new FakeFamily { Guidance = DiffusionGuidance.Cfg };
        using var pipeline = Pipeline(family);
        pipeline.Run(new DiffusionRequest { Prompt = "sharp", NegativePrompt = "x", Cfg = 4f, Seed = 2 });
        Assert.Equal(6, family.Denoiser.Steps.Count);
        Assert.Contains("condition 'sharp' 0", family.Events);
        Assert.Contains("condition 'x' 0", family.Events);

        var shape = new LatentShape(2, 1, 4, 8);
        float[] x = FakeFamily.ToHwc(TorchCpuRandn.Normal(shape.Count, 2), shape);
        float[] sigmas = DiffusionSchedules.ComfySimpleOverGrid(family.DefaultRecipe.Grid, 3);
        float positive = "sharp".Length * 0.25f, negative = "x".Length * 0.25f;
        for (int i = 0; i < 3; i++)
        {
            float dt = sigmas[i + 1] - sigmas[i];
            for (int k = 0; k < x.Length; k++)
            {
                float c = FakeDenoiser.Velocity(x[k], positive, i), u = FakeDenoiser.Velocity(x[k], negative, i);
                x[k] += (u + 4f * (c - u)) * dt;
            }
        }
        Assert.Equal(Bits(family.FromRows(x, shape).Data), Bits(family.Vae.LastDecoded.Data));

        // Guidance 1 runs one branch, whatever the negative prompt.
        var single = new FakeFamily { Guidance = DiffusionGuidance.Cfg };
        using var other = Pipeline(single);
        other.Run(new DiffusionRequest { NegativePrompt = "x", Cfg = 1f });
        Assert.Equal(3, single.Denoiser.Steps.Count);
    }

    [Fact]
    public void ASpeedAdaptersContractFixesStepsSigmasAndHeads()
    {
        var family = new FakeFamily();
        using var pipeline = Pipeline(family);
        var contract = new SamplingContract { Name = "fast", Sigmas = new[] { 1f, 0.5f, 0f }, Heads = HeadPolicy.ByStep };
        var adapters = new AdapterSet(new[] { new DiffusionAdapter { Name = "fast", Contract = contract } });
        DiffusionResult result = pipeline.Run(new DiffusionRequest { Adapters = adapters });
        Assert.Equal(2, result.Recipe.Steps);
        Assert.Equal(new[] { 1f, 0.5f, 0f }, result.Recipe.Sigmas);
        Assert.Equal(new[] { "fast" }, result.Recipe.Adapters);
        Assert.Equal(new[] { (1f, 0.5f, 0, 1), (0.5f, 0f, 1, 2) },
            family.Denoiser.Steps.Select(s => (s.Sigma, s.Next, s.HeadStart, s.HeadStop)));
        Assert.Contains("fast samples exactly 2 steps",
            Assert.Throws<ArgumentException>(() => pipeline.Run(new DiffusionRequest { Adapters = adapters, Steps = 6 })).Message);
    }

    // ---- masked edits

    [Fact]
    public void AMaskedEditKeepsTheSourceLatentAndPixelsExactlyWhereTheMaskKeeps()
    {
        var family = new FakeFamily();
        using var pipeline = Pipeline(family);
        RgbImage source = Picture(32, 16, (x, y) => (x + 2 * y) / 64f);
        RgbImage mask = Picture(32, 16, (x, _) => x < 16 ? 1f : 0f);   // regenerate the left half, keep the right
        DiffusionResult result = pipeline.Run(new DiffusionRequest
        {
            Prompt = "repaint", Seed = 11,
            References = new[] { new DiffusionReference(source) },
            Mask = new EditMaskOptions { Mask = mask },
        });
        // The source latent is the posterior mean (the reference's own encode: same size as the canvas), never a sample.
        Assert.Single(family.Events, e => e.StartsWith("encode"));
        LatentPosterior mean = new FakeVae(family).Encode(source, default);
        LatentTensor decoded = family.Vae.LastDecoded;
        int plane = decoded.Shape.Height * decoded.Shape.Width;
        for (int c = 0; c < decoded.Shape.Channels; c++)
            for (int y = 0; y < decoded.Shape.Height; y++)
                for (int x = 0; x < decoded.Shape.Width; x++)
                {
                    int i = c * plane + y * decoded.Shape.Width + x;
                    if (x >= 4) Assert.Equal(mean.Mean[i], decoded.Data[i]);
                    else Assert.NotEqual(mean.Mean[i], decoded.Data[i]);
                }
        Assert.Equal((32, 16), (result.Image.Width, result.Image.Height));
        for (int y = 0; y < 16; y++)
            for (int x = 16; x < 32; x++)
                for (int c = 0; c < 3; c++)
                    Assert.Equal(source.Pixels[(y * 32 + x) * 3 + c], result.Image.Pixels[(y * 32 + x) * 3 + c]);
    }

    [Fact]
    public void AnEmptyMaskReturnsTheSourceWithoutMakingAComponent()
    {
        var family = new FakeFamily();
        using var pipeline = Pipeline(family);
        RgbImage source = Picture(32, 16, (x, y) => (x * y) / 512f);
        DiffusionResult result = pipeline.Run(new DiffusionRequest
        {
            References = new[] { new DiffusionReference(source) },
            Mask = new EditMaskOptions { Mask = Picture(32, 16, (_, _) => 0f) },
        });
        Assert.Equal(0, family.Created);
        Assert.Empty(family.Events);
        Assert.Null(result.Residency);
        Assert.Equal(source.Pixels, result.Image.Pixels);
        Assert.NotSame(source.Pixels, result.Image.Pixels);
    }

    [Fact]
    public void AMaskWithoutAPictureIsRefused()
    {
        using var pipeline = Pipeline(new FakeFamily());
        Assert.Throws<ArgumentException>(() => pipeline.Run(new DiffusionRequest { Mask = new EditMaskOptions { Mask = Picture(32, 16, (_, _) => 1f) } }));
        Assert.Throws<ArgumentException>(() => pipeline.Run(new DiffusionRequest { Mask = new EditMaskOptions { MaskInvert = true } }));
    }

    // ---- previews

    [Fact]
    public void PreviewsDecodeTheCleanEstimateOnQwensInterval()
    {
        var family = new FakeFamily { DefaultRecipe = new FakeFamily().DefaultRecipe with { Grid = new[] { 1f, .8f, .6f, .4f, .2f, 0f }, Steps = 5 } };
        using var pipeline = Pipeline(family);
        var previews = new List<int>();
        pipeline.Run(new DiffusionRequest
        {
            PreviewCount = 2,
            Progress = p => { if (p.IsStep && p.Preview is RgbImage) previews.Add(p.Step); },
        });
        // interval (5 + 2) / 3 = 2: steps 2 and 4 (never the last).
        Assert.Equal(new[] { 2, 4 }, previews);
        Assert.Equal(2, family.Vae.Previews);
        // Without a progress sink nothing is decoded early.
        var quiet = new FakeFamily();
        using var other = Pipeline(quiet);
        other.Run(new DiffusionRequest { PreviewCount = 3 });
        Assert.Equal(0, quiet.Vae.Previews);
    }

    // ---- cancellation

    [Fact]
    public void ACancelledRunReleasesActivationsAndKeepsWeights()
    {
        var family = new FakeFamily();
        using var pipeline = Pipeline(family);
        using var cancel = new CancellationTokenSource();
        Assert.ThrowsAny<OperationCanceledException>(() => pipeline.Run(new DiffusionRequest
        {
            Progress = p => { if (p.IsStep && p.Step == 1) cancel.Cancel(); },
        }, cancel.Token));
        Assert.Equal(new[] { "step 0" }, family.Events.Where(e => e.StartsWith("step")).ToArray());
        Assert.DoesNotContain("decode", family.Events);
        Assert.Contains("release denoiser workspace", family.Events);
        Assert.Contains("release conditioner workspace", family.Events);
        Assert.DoesNotContain("release denoiser", family.Events);
        Assert.True(pipeline.IsResident(DiffusionComponent.Denoiser));
        // The pipeline goes on serving.
        family.Events.Clear();
        pipeline.Run(new DiffusionRequest());
        Assert.Contains("decode", family.Events);
    }

    [Fact]
    public void ACancelledTokenStopsBeforeAnyComponentRuns()
    {
        var family = new FakeFamily();
        using var pipeline = Pipeline(family);
        Assert.ThrowsAny<OperationCanceledException>(() => pipeline.Run(new DiffusionRequest(), new CancellationToken(canceled: true)));
        Assert.DoesNotContain(family.Events, e => e.StartsWith("condition") || e.StartsWith("noise"));
    }

    // ---- residency

    [Fact]
    public void OnADeviceThatCannotHoldEverythingTheConditionerGoesBeforeTheDenoiserAndComesBackAfterIt()
    {
        // An A40's bf16 numbers (GB): text encoder 18, DiT 40, VAE 1; 46 free with nothing loaded.
        var family = new FakeFamily();
        long held = 0;
        var log = new List<string>();
        var planner = new ResidencyPlanner(() => 46 - held, log.Add);
        using var pipeline = new DiffusionPipeline(family, new DiffusionLoad { Backend = BackendType.GgmlCuda }, planner, log.Add);
        DiffusionResult first = pipeline.Run(new DiffusionRequest { Prompt = "one" });
        Assert.False(first.Residency.KeepsAll);
        Assert.True(first.Residency.Fits);
        int released = family.Events.IndexOf("release conditioner"), step = family.Events.IndexOf("step 0");
        Assert.True(released >= 0 && released < step, string.Join(", ", family.Events));
        Assert.False(pipeline.IsResident(DiffusionComponent.Conditioner));
        Assert.True(pipeline.IsResident(DiffusionComponent.Denoiser));

        // The next request needs the conditioner again: the denoiser (needed later) makes the room, not the VAE.
        held = 40 + 1 + 3 + 1;
        family.Events.Clear();
        DiffusionResult second = pipeline.Run(new DiffusionRequest { Prompt = "two" });
        int denoiserGone = family.Events.IndexOf("release denoiser"), condition = family.Events.IndexOf("condition 'two' 0");
        Assert.True(denoiserGone >= 0 && denoiserGone < condition, string.Join(", ", family.Events));
        Assert.DoesNotContain("release vae", family.Events);
        // Each decision is logged once: the first request's, then the steady state's.
        Assert.Equal(new[]
        {
            "[diffusion-residency] 32x16, 0 reference(s): before Denoise: release the conditioner",
            "[diffusion-residency] 32x16, 0 reference(s): before Condition: release the denoiser; before Denoise: release the conditioner",
        }, log.Where(l => l.StartsWith("[diffusion-residency]")).Select(l => l[..l.IndexOf(" (peak", StringComparison.Ordinal)]));
    }

    [Fact]
    public void OnADeviceThatHoldsEverythingNothingIsReleased()
    {
        var family = new FakeFamily();
        var log = new List<string>();
        using var pipeline = Pipeline(family, spare: 94, log: log);
        pipeline.Run(new DiffusionRequest());
        pipeline.Run(new DiffusionRequest());
        Assert.DoesNotContain(family.Events, e => e.StartsWith("release"));
        // One decision, logged once.
        Assert.Single(log, l => l.StartsWith("[diffusion-residency]"));
    }
}
