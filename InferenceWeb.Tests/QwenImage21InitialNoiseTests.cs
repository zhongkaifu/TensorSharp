// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Reflection;
using TensorSharp.Models.QwenImage;
using TensorSharp.Runtime;
using static InferenceWeb.Tests.QwenImage21SamplingTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The noise a request starts from, through the pipeline's own seam
/// (<see cref="QwenImage21Pipeline.Request"/>) and through <c>Run</c> itself, stopped where it
/// has drawn the latent it samples from (<see cref="QwenImage21Pipeline.LatentsDrawn"/>): which
/// pictures an edit's noise follows and the size it is drawn for. The geometry reads the server
/// default size (TS_QWEN_IMAGE_WIDTH/HEIGHT), so these run with the other classes that set it,
/// one at a time.
/// </summary>
[Collection(QwenImageDefaultSizeCollection.Name)]
public sealed class QwenImage21InitialNoiseTests : IDisposable
{
    private readonly EnvScope _env = new();

    public QwenImage21InitialNoiseTests()
    {
        _env.Set("TS_QWEN_IMAGE_WIDTH", null);
        _env.Set("TS_QWEN_IMAGE_HEIGHT", null);
        _env.Set(QwenImage21Sampling.EditNoiseVariable, null);
    }

    public void Dispose() => _env.Dispose();

    private static float[] Expected(long seed, ulong stream, int latentHeight, int latentWidth) =>
        QwenImage21Pipeline.ToTokens(QwenImage21Sampling.Noise(latentHeight * latentWidth * 64, seed, stream), latentHeight, latentWidth);

    /// <summary>A selection: white over the square [left, left + size) on both axes.</summary>
    private static RgbImage Square(int width, int height, int left, int size)
    {
        var pixels = new float[width * height * 3];
        for (int y = left; y < left + size; y++)
            for (int x = left; x < left + size; x++)
                pixels[(y * width + x) * 3] = pixels[(y * width + x) * 3 + 1] = pixels[(y * width + x) * 3 + 2] = 1f;
        return new RgbImage(width, height, pixels);
    }

    [Fact]
    public void TextToImageStartsFromTheSeedsNoiseBitForBit()
    {
        var request = new QwenImage21Pipeline.Request(Array.Empty<RgbImage>(), new QwenImageParams { Seed = 3, Width = 96, Height = 64 }, null, followReferences: true);

        Assert.Equal(0UL, request.NoiseStream);
        Assert.Equal((4, 6), (request.LatentHeight, request.LatentWidth));
        Assert.True(SameBits(QwenImage21Pipeline.ToTokens(SeedOnlyNoise(4 * 6 * 64, 3), 4, 6), request.InitialLatents()));
    }

    [Fact]
    public void AnEditStartsFromItsReferencesStreamAndInSeedModeFromTheSeedsNoise()
    {
        var source = Decoded(96, 64, 7);
        var p = new QwenImageParams { Seed = 3, Width = 96, Height = 64 };
        float[] textToImage = QwenImage21Pipeline.ToTokens(SeedOnlyNoise(4 * 6 * 64, 3), 4, 6);

        var edit = new QwenImage21Pipeline.Request(new[] { source }, p, null, followReferences: true);
        Assert.Equal(QwenImage21Sampling.ReferenceStream(new[] { source }), edit.NoiseStream);
        Assert.NotEqual(0UL, edit.NoiseStream);
        Assert.True(SameBits(Expected(3, edit.NoiseStream, 4, 6), edit.InitialLatents()));
        Assert.False(SameBits(textToImage, edit.InitialLatents()));

        var seedMode = new QwenImage21Pipeline.Request(new[] { source }, p, null, followReferences: false);
        Assert.Equal(0UL, seedMode.NoiseStream);
        Assert.True(SameBits(textToImage, seedMode.InitialLatents()));
    }

    /// <summary>
    /// A masked edit with a crop hands the encoders the selection's crop in place of the first
    /// picture. The noise still follows the picture the caller passed, so every selection on one
    /// picture, and the unmasked edit of it, start from that picture's stream; the crop is
    /// another picture with another stream. A later reference stays as the caller gave it.
    /// </summary>
    [Fact]
    public void AMaskedEditFollowsTheCallersPictureNotTheSelectionsCrop()
    {
        var source = Decoded(128, 96, 4);
        var other = Decoded(32, 32, 5);
        var caller = new[] { source, other };
        var p = new QwenImageParams { Seed = 0, Mask = Square(128, 96, 16, 32), MaskCrop = true, MaskCropPadding = 8 };

        var masked = new QwenImage21Pipeline.Request(caller, p, null, followReferences: true);

        Assert.Same(caller, masked.CallerInputs);
        Assert.NotSame(source, masked.Inputs[0]);
        Assert.Same(masked.Mask.Reference, masked.Inputs[0]);
        Assert.Same(other, masked.Inputs[1]);
        Assert.Same(source, caller[0]);
        Assert.Equal(QwenImage21Sampling.ReferenceStream(caller), masked.NoiseStream);
        Assert.NotEqual(QwenImage21Sampling.ReferenceStream(masked.Inputs), masked.NoiseStream);

        var unmasked = new QwenImage21Pipeline.Request(caller, new QwenImageParams { Seed = 0 }, null, followReferences: true);
        Assert.Equal(unmasked.NoiseStream, masked.NoiseStream);
        var elsewhere = new QwenImage21Pipeline.Request(caller,
            new QwenImageParams { Seed = 0, Mask = Square(128, 96, 48, 40), MaskCrop = true, MaskCropPadding = 8 }, null, followReferences: true);
        Assert.Equal(masked.NoiseStream, elsewhere.NoiseStream);

        var (width, height) = masked.Geometry.Sampling;
        Assert.True(SameBits(Expected(0, masked.NoiseStream, height / 16, width / 16), masked.InitialLatents()));
    }

    /// <summary>
    /// An edit that keeps its source size samples at <see cref="QwenImage21Pipeline.KeptSourceSampling"/>
    /// and resizes back afterwards; the noise is drawn for the sampled size. A 640 x 480 photo
    /// edited at TensorAgent's area samples at 1184 x 896 and comes back at 640 x 480.
    /// </summary>
    [Fact]
    public void AnEditThatKeepsItsSourceSizeDrawsNoiseForTheSizeItSamplesAt()
    {
        var photo = Decoded(640, 480, 9);
        var p = new QwenImageParams { Seed = 5, TargetArea = 1024L * 1024, KeepSourceSize = true };

        var request = new QwenImage21Pipeline.Request(new[] { photo }, p, null, followReferences: true);

        var geometry = QwenImage21Pipeline.ResolveGeometry(p, photo, null);
        Assert.Equal(geometry, request.Geometry);
        Assert.Equal(((1184, 896), (640, 480)), request.Geometry);
        Assert.Equal((896 / 16, 1184 / 16), (request.LatentHeight, request.LatentWidth));
        float[] latents = request.InitialLatents();
        Assert.Equal(1184 / 16 * (896 / 16) * 64, latents.Length);
        Assert.True(SameBits(Expected(5, QwenImage21Sampling.ReferenceStream(new[] { photo }), 896 / 16, 1184 / 16), latents));
    }

    /// <summary>
    /// The case that retraced: a picture edited at the seed and size it was drawn at. Kept at its
    /// own size (TensorAgent and the Web UI send keepSourceSize with every edit), a 1024 x 1024
    /// picture samples at 1024 x 1024, exactly where its text-to-image draw did, and the edit's
    /// noise is still not that draw.
    /// </summary>
    [Fact]
    public void AnEditOfAPictureAtItsOwnSeedAndSizeDoesNotStartFromItsNoise()
    {
        var picture = Decoded(1024, 1024, 11);
        var drawn = new QwenImage21Pipeline.Request(Array.Empty<RgbImage>(), new QwenImageParams { Seed = 0, Width = 1024, Height = 1024 }, null, followReferences: true);
        var edit = new QwenImage21Pipeline.Request(new[] { picture }, new QwenImageParams { Seed = 0, TargetArea = 1024L * 1024, KeepSourceSize = true }, null, followReferences: true);

        Assert.Equal(drawn.Geometry.Sampling, edit.Geometry.Sampling);
        float[] from = drawn.InitialLatents(), to = edit.InitialLatents();
        Assert.Equal(from.Length, to.Length);
        Assert.False(SameBits(from, to));
        Assert.InRange(Math.Abs(Correlation(from, to)), 0, 0.02);
    }

    // =====================================================================================
    // through Run: the latent the pipeline samples from, not only the one Request offers
    // =====================================================================================

    private sealed class StopAtTheLatents : Exception { }

    /// <summary>
    /// Runs <paramref name="run"/> on a Qwen-Image-2.1 model with no weights, stopped where
    /// <c>Run</c> has drawn the latent it samples from, and returns that latent. The model has a
    /// vision projector path, so an edit gets past its refusal, and a GPU backend, so the cpu
    /// backend's memory check does not depend on this machine; nothing native is called.
    /// </summary>
    private static float[] LatentsRunSamplesFrom(Action<QwenImageModel> run)
    {
        QwenImageModel model = QwenImage21TurboTests.ModelOf(QwenImageVariant.Base, backend: BackendType.GgmlMetal);
        (typeof(QwenImageModel).GetField("_mmprojPath", BindingFlags.Instance | BindingFlags.NonPublic)
            ?? throw new MissingFieldException(nameof(QwenImageModel), "_mmprojPath")).SetValue(model, "mmproj-not-loaded.gguf");
        float[]? drawn = null;
        QwenImage21Pipeline.LatentsDrawn.Value = latents =>
        {
            drawn = latents;
            throw new StopAtTheLatents();
        };
        try
        {
            Assert.Throws<StopAtTheLatents>(() => run(model));
        }
        finally
        {
            QwenImage21Pipeline.LatentsDrawn.Value = null;
        }
        return drawn ?? throw new InvalidOperationException("Run did not draw its latent.");
    }

    [Fact]
    public void RunSamplesTextToImageFromTheSeedsNoise()
    {
        float[] latents = LatentsRunSamplesFrom(model =>
            model.GenerateImage("a red teapot", new QwenImageParams { Seed = 3, Width = 96, Height = 64 }));

        Assert.True(SameBits(QwenImage21Pipeline.ToTokens(SeedOnlyNoise(4 * 6 * 64, 3), 4, 6), latents));
    }

    /// <summary>
    /// The retrace's fix as the pipeline applies it: an edit samples from its references' stream,
    /// and only <c>TS_QWEN21_EDIT_NOISE=seed</c> brings back the seed's text-to-image noise.
    /// </summary>
    [Fact]
    public void RunSamplesAnEditFromItsReferencesStreamAndInSeedModeFromTheSeedsNoise()
    {
        var source = Decoded(96, 64, 7);
        var p = new QwenImageParams { Seed = 3, Width = 96, Height = 64 };
        float[] textToImage = QwenImage21Pipeline.ToTokens(SeedOnlyNoise(4 * 6 * 64, 3), 4, 6);

        float[] edit = LatentsRunSamplesFrom(model => model.EditImage("make it blue", source, p));
        Assert.True(SameBits(Expected(3, QwenImage21Sampling.ReferenceStream(new[] { source }), 4, 6), edit));
        Assert.False(SameBits(textToImage, edit));

        _env.Set(QwenImage21Sampling.EditNoiseVariable, "seed");
        float[] seedMode = LatentsRunSamplesFrom(model => model.EditImage("make it blue", source, p));
        Assert.True(SameBits(textToImage, seedMode));
    }

    [Fact]
    public void RunSamplesAKeptSourceSizeEditAtTheSizeItSamplesAt()
    {
        var photo = Decoded(640, 480, 9);
        float[] latents = LatentsRunSamplesFrom(model => model.EditImage("make it blue", photo,
            new QwenImageParams { Seed = 5, TargetArea = 1024L * 1024, KeepSourceSize = true }));

        Assert.True(SameBits(Expected(5, QwenImage21Sampling.ReferenceStream(new[] { photo }), 896 / 16, 1184 / 16), latents));
    }

    [Fact]
    public void RunSamplesAMaskedEditFromTheCallersPictures()
    {
        var source = Decoded(128, 96, 4);
        var other = Decoded(32, 32, 5);
        var p = new QwenImageParams { Seed = 0, Mask = Square(128, 96, 16, 32), MaskCrop = true, MaskCropPadding = 8 };
        var (width, height) = new QwenImage21Pipeline.Request(new[] { source, other }, p, BackendType.GgmlMetal, followReferences: true).Geometry.Sampling;

        float[] latents = LatentsRunSamplesFrom(model => model.EditImage("make it blue", new[] { source, other }, p));

        Assert.True(SameBits(Expected(0, QwenImage21Sampling.ReferenceStream(new[] { source, other }), height / 16, width / 16), latents));
    }
}
