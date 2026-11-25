// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using System.Runtime.CompilerServices;
using TensorSharp.Models;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;
using static InferenceWeb.Tests.QwenImage21SamplingTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The noise a request starts from, through the family's seam (<see cref="QwenImage21Family.Noise(DiffusionRequest)"/> and
/// its canvas) and through the model's <c>Run</c> itself, over the diffusion runtime with stand-in components, stopped at
/// the first denoiser step (the latent it samples from): which pictures an edit's noise follows and the size it is drawn
/// for. The geometry reads the server default size (TS_QWEN_IMAGE_WIDTH/HEIGHT), so these run with the other classes that
/// set it, one at a time.
/// </summary>
[Collection(QwenImageDefaultSizeCollection.Name)]
public sealed class QwenImage21InitialNoiseTests : IDisposable
{
    private readonly EnvScope _env = new();

    public QwenImage21InitialNoiseTests()
    {
        _env.Set("TS_QWEN_IMAGE_WIDTH", null);
        _env.Set("TS_QWEN_IMAGE_HEIGHT", null);
        _env.Set(QwenImage21Noise.EditNoiseVariable, null);
    }

    public void Dispose() => _env.Dispose();

    private static float[] Expected(long seed, ulong stream, int latentHeight, int latentWidth) =>
        InitialLatents(seed, stream, latentHeight, latentWidth);

    /// <summary>A selection: white over the square [left, left + size) on both axes.</summary>
    private static RgbImage Square(int width, int height, int left, int size)
    {
        var pixels = new float[width * height * 3];
        for (int y = left; y < left + size; y++)
            for (int x = left; x < left + size; x++)
                pixels[(y * width + x) * 3] = pixels[(y * width + x) * 3 + 1] = pixels[(y * width + x) * 3 + 2] = 1f;
        return new RgbImage(width, height, pixels);
    }

    private static DiffusionRequest Request(RgbImage[] pictures, long seed, int width = 0, int height = 0, long area = 0,
        bool keepSourceSize = false) => new()
    {
        Prompt = "p",
        References = pictures.Select(p => new DiffusionReference(p)).ToArray(),
        Seed = seed,
        Width = width,
        Height = height,
        TargetArea = area,
        KeepSourceSize = keepSourceSize,
    };

    /// <summary>The family's first draw for <paramref name="request"/> at its sampling size, as token rows.</summary>
    private static (float[] Latents, ulong Stream, (int Width, int Height) Sampling) Draw(DiffusionRequest request, bool followReferences = true)
    {
        var family = new QwenImage21Family(BackendType.GgmlMetal, QwenImageVariant.Base) { FollowReferences = followReferences };
        RgbImage first = request.References.Count > 0 ? request.References[0].Image : null;
        var canvas = family.ResolveCanvas(request, request.Mask == null ? null : EditMask.Create(request.Mask, first));
        var (width, height) = canvas.Sampling;
        INoiseSource noise = family.Noise(request);
        float[] latents = ToTokens(noise.Next(width / 16 * (height / 16) * 64), height / 16, width / 16);
        return (latents, family.LastStream, canvas.Sampling);
    }

    [Fact]
    public void TextToImageStartsFromTheSeedsNoiseBitForBit()
    {
        var (latents, stream, sampling) = Draw(Request(Array.Empty<RgbImage>(), 3, 96, 64));

        Assert.Equal(0UL, stream);
        Assert.Equal((96, 64), sampling);
        Assert.True(SameBits(ToTokens(SeedOnlyNoise(4 * 6 * 64, 3), 4, 6), latents));
    }

    [Fact]
    public void AnEditStartsFromItsReferencesStreamAndInSeedModeFromTheSeedsNoise()
    {
        var source = Decoded(96, 64, 7);
        float[] textToImage = ToTokens(SeedOnlyNoise(4 * 6 * 64, 3), 4, 6);

        var (edit, stream, _) = Draw(Request(new[] { source }, 3, 96, 64));
        Assert.Equal(QwenImage21Noise.ReferenceStream(new[] { source }), stream);
        Assert.NotEqual(0UL, stream);
        Assert.True(SameBits(Expected(3, stream, 4, 6), edit));
        Assert.False(SameBits(textToImage, edit));

        var (seedMode, seedStream, _) = Draw(Request(new[] { source }, 3, 96, 64), followReferences: false);
        Assert.Equal(0UL, seedStream);
        Assert.True(SameBits(textToImage, seedMode));
    }

    /// <summary>
    /// An edit that keeps its source size samples at <see cref="QwenImage21Canvas.KeptSourceSampling"/> and resizes back
    /// afterwards; the noise is drawn for the sampled size. A 640 x 480 photo edited at TensorAgent's area samples at
    /// 1184 x 896 and comes back at 640 x 480.
    /// </summary>
    [Fact]
    public void AnEditThatKeepsItsSourceSizeDrawsNoiseForTheSizeItSamplesAt()
    {
        var photo = Decoded(640, 480, 9);
        var request = Request(new[] { photo }, 5, area: 1024L * 1024, keepSourceSize: true);

        DiffusionCanvas canvas = QwenImage21Canvas.Resolve(request, photo, null);
        Assert.Equal((1184, 896), canvas.Sampling);
        Assert.Equal((640, 480), canvas.Output);
        var (latents, stream, sampling) = Draw(request);
        Assert.Equal((1184, 896), sampling);
        Assert.Equal(1184 / 16 * (896 / 16) * 64, latents.Length);
        Assert.True(SameBits(Expected(5, QwenImage21Noise.ReferenceStream(new[] { photo }), 896 / 16, 1184 / 16), latents));
    }

    /// <summary>
    /// The case that retraced: a picture edited at the seed and size it was drawn at. Kept at its own size (TensorAgent and
    /// the Web UI send keepSourceSize with every edit), a 1024 x 1024 picture samples at 1024 x 1024, exactly where its
    /// text-to-image draw did, and the edit's noise is still not that draw.
    /// </summary>
    [Fact]
    public void AnEditOfAPictureAtItsOwnSeedAndSizeDoesNotStartFromItsNoise()
    {
        var picture = Decoded(1024, 1024, 11);
        var (from, _, drawnAt) = Draw(Request(Array.Empty<RgbImage>(), 0, 1024, 1024));
        var (to, _, editedAt) = Draw(Request(new[] { picture }, 0, area: 1024L * 1024, keepSourceSize: true));

        Assert.Equal(drawnAt, editedAt);
        Assert.Equal(from.Length, to.Length);
        Assert.False(SameBits(from, to));
        Assert.InRange(Math.Abs(Correlation(from, to)), 0, 0.02);
    }

    // =====================================================================================
    // through Run: the latent the runtime samples from, not only the one the family offers
    // =====================================================================================

    private sealed class StopAtTheLatents : Exception { }

    /// <summary>Stand-ins for the encoders: a text row and, for each picture, one vision slot a 32-pixel cell (as
    /// Qwen3-VL-8B gives them), and the posterior mean of zeros; the denoiser keeps the first step's latent and stops.</summary>
    private sealed class Stand : IConditioner, IVaeCodec, IDenoiser
    {
        public float[] Latents;
        public long WeightBytes => 0;
        public long WorkingBytes(DiffusionWorkload workload) => 0;
        public void ReleaseWorkspace() { }
        public void ReleaseDevice() { }
        public void Dispose() { }
        public DenoiserCaps Caps { get; } = new();

        public ConditioningSet Encode(string prompt, IReadOnlyList<PresentedImage> pictures, CancellationToken ct)
        {
            var tags = new List<int> { 0 };
            for (int i = 0; i < pictures.Count; i++)
            {
                tags.AddRange(Enumerable.Repeat(i + 1, pictures[i].Image.Width / 32 * (pictures[i].Image.Height / 32)));
                tags.Add(0);
            }
            return new ConditioningSet(new float[tags.Count * 4096], tags.Count, 4096, pictures.Count > 0 ? tags.ToArray() : null);
        }

        public LatentPosterior Encode(RgbImage image, CancellationToken ct) =>
            new(new LatentShape(64, 1, image.Height / 16, image.Width / 16), new float[image.Height / 16 * (image.Width / 16) * 64], null);

        public RgbImage Decode(LatentTensor latent, CancellationToken ct) => throw new InvalidOperationException();
        public RgbImage Preview(LatentTensor latent, int maxSide) => throw new InvalidOperationException();

        public StepOutput Step(DenoiseStepInputs inputs, AdapterSet adapters, CancellationToken ct)
        {
            Latents = (float[])inputs.Latent.Clone();
            throw new StopAtTheLatents();
        }
    }

    /// <summary>
    /// Runs <paramref name="run"/> on a Qwen-Image-2.1 model no constructor ran for: its runtime pipeline over the real
    /// family with <see cref="Stand"/> components and no device budget, stopped at the first denoiser step, and returns
    /// that step's latent and the stream the family drew it from. The model has a vision projector path, so an edit gets
    /// past its refusal, and a GPU backend, so the cpu backend's memory check does not depend on this machine; nothing
    /// native is called.
    /// </summary>
    private static (float[] Latents, ulong Stream) LatentsRunSamplesFrom(Action<QwenImage21DiffusionModel> run)
    {
        QwenImage21DiffusionModel model = QwenImage21TurboTests.ModelOf(QwenImageVariant.Base, backend: BackendType.GgmlMetal);
        var stand = new Stand();
        var family = new QwenImage21Family(BackendType.GgmlMetal, QwenImageVariant.Base)
        {
            ConditionerFactory = _ => stand,
            DenoiserFactory = _ => stand,
            VaeFactory = _ => stand,
        };
        static void Set(object target, string field, object value) =>
            (typeof(QwenImage21DiffusionModel).GetField(field, BindingFlags.Instance | BindingFlags.NonPublic)
                ?? throw new MissingFieldException(nameof(QwenImage21DiffusionModel), field)).SetValue(target, value);
        Set(model, "_mmprojPath", "mmproj-not-loaded.gguf");
        Set(model, "_family", family);
        Set(model, "_pipeline", new DiffusionPipeline(family, new DiffusionLoad { Backend = BackendType.GgmlMetal },
            new ResidencyPlanner(() => null, _ => { }), _ => { }));
        Assert.Throws<StopAtTheLatents>(() => run(model));
        return (stand.Latents ?? throw new InvalidOperationException("Run did not reach its first step."), family.LastStream);
    }

    [Fact]
    public void RunSamplesTextToImageFromTheSeedsNoise()
    {
        var (latents, stream) = LatentsRunSamplesFrom(model =>
            model.GenerateImage("a red teapot", new QwenImageParams { Seed = 3, Width = 96, Height = 64 }));

        Assert.Equal(0UL, stream);
        Assert.True(SameBits(ToTokens(SeedOnlyNoise(4 * 6 * 64, 3), 4, 6), latents));
    }

    /// <summary>
    /// The retrace's fix as the runtime applies it: an edit samples from its references' stream, and only
    /// <c>TS_QWEN21_EDIT_NOISE=seed</c> brings back the seed's text-to-image noise.
    /// </summary>
    [Fact]
    public void RunSamplesAnEditFromItsReferencesStreamAndInSeedModeFromTheSeedsNoise()
    {
        var source = Decoded(96, 64, 7);
        var p = new QwenImageParams { Seed = 3, Width = 96, Height = 64 };
        float[] textToImage = ToTokens(SeedOnlyNoise(4 * 6 * 64, 3), 4, 6);

        var (edit, stream) = LatentsRunSamplesFrom(model => model.EditImage("make it blue", source, p));
        Assert.Equal(QwenImage21Noise.ReferenceStream(new[] { source }), stream);
        Assert.True(SameBits(Expected(3, stream, 4, 6), edit));
        Assert.False(SameBits(textToImage, edit));

        _env.Set(QwenImage21Noise.EditNoiseVariable, "seed");
        var (seedMode, seedStream) = LatentsRunSamplesFrom(model => model.EditImage("make it blue", source, p));
        Assert.Equal(0UL, seedStream);
        Assert.True(SameBits(textToImage, seedMode));
    }

    [Fact]
    public void RunSamplesAKeptSourceSizeEditAtTheSizeItSamplesAt()
    {
        var photo = Decoded(640, 480, 9);
        var (latents, _) = LatentsRunSamplesFrom(model => model.EditImage("make it blue", photo,
            new QwenImageParams { Seed = 5, TargetArea = 1024L * 1024, KeepSourceSize = true }));

        Assert.True(SameBits(Expected(5, QwenImage21Noise.ReferenceStream(new[] { photo }), 896 / 16, 1184 / 16), latents));
    }

    /// <summary>
    /// A masked edit with a crop hands the encoders the selection's crop in place of the first picture. The noise still
    /// follows the pictures the caller passed, so every selection on one picture, and the unmasked edit of it, start from
    /// that picture's stream; the crop is another picture with another stream.
    /// </summary>
    [Fact]
    public void RunSamplesAMaskedEditFromTheCallersPictures()
    {
        var source = Decoded(128, 96, 4);
        var other = Decoded(32, 32, 5);
        var p = new QwenImageParams { Seed = 0, Mask = Square(128, 96, 16, 32), MaskCrop = true, MaskCropPadding = 8 };
        ulong callers = QwenImage21Noise.ReferenceStream(new[] { source, other });

        var (latents, stream) = LatentsRunSamplesFrom(model => model.EditImage("make it blue", new[] { source, other }, p));
        Assert.Equal(callers, stream);

        // The crop the encoders see is another picture, with another stream; the latent is the crop's sampling canvas.
        var request = new DiffusionRequest { References = new[] { new DiffusionReference(source), new DiffusionReference(other) },
            Mask = QwenImageEditMask.Options(p) };
        var mask = EditMask.Create(request.Mask, source);
        Assert.NotSame(source, mask.Reference);
        Assert.NotEqual(QwenImage21Noise.ReferenceStream(new[] { mask.Reference, other }), stream);
        var (width, height) = new QwenImage21Family(BackendType.GgmlMetal, QwenImageVariant.Base).ResolveCanvas(request, mask).Sampling;
        Assert.True(SameBits(Expected(0, callers, height / 16, width / 16), latents));

        var (_, unmasked) = LatentsRunSamplesFrom(model => model.EditImage("make it blue", new[] { source, other }, new QwenImageParams { Seed = 0 }));
        Assert.Equal(callers, unmasked);
        var (_, elsewhere) = LatentsRunSamplesFrom(model => model.EditImage("make it blue", new[] { source, other },
            new QwenImageParams { Seed = 0, Mask = Square(128, 96, 48, 40), MaskCrop = true, MaskCropPadding = 8 }));
        Assert.Equal(callers, elsewhere);
    }
}
