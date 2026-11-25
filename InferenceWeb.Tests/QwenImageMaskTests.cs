// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.CompilerServices;
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.QwenImage21;
using TensorSharp.Models.SharedDiffusion;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public sealed class QwenImageMaskTests
{
    private static RgbImage Image(int width, int height, Func<int, float> values, float[]? alpha = null) =>
        new(width, height, Enumerable.Range(0, width * height).SelectMany(i => Enumerable.Repeat(values(i), 3)).ToArray(), alpha);

    [Fact]
    public void GrayscaleAndExplicitAlphaHaveOppositeTransparentWhiteSemantics()
    {
        var source = Image(3, 1, _ => .25f);
        var image = Image(3, 1, i => new[] { 0f, .5f, 1f }[i], new[] { 0f, .5f, 1f });
        var grayscale = QwenImageEditMask.Create(new QwenImageParams { Mask = image }, source);
        var alpha = QwenImageEditMask.Create(new QwenImageParams { Mask = image, MaskMode = QwenImageMaskMode.Alpha }, source);
        var inverted = QwenImageEditMask.Create(new QwenImageParams { Mask = image, MaskInvert = true }, source);
        Assert.Equal(new[] { 0f, .5f, 1f }, grayscale.Weights);
        Assert.Equal(new[] { 1f, .5f, 0f }, alpha.Weights);
        Assert.Equal(alpha.Weights, inverted.Weights);
    }

    [Fact]
    public void CompositePreservesExactProtectedRgbAndAlphaAndBlendsSoftSelections()
    {
        var source = new RgbImage(3, 1, new[] { .1234567f, .4f, .9f, .2f, .4f, .6f, .1f, .3f, .5f }, new[] { 0f, .25f, .5f });
        var original = (float[])source.Pixels.Clone();
        var plan = QwenImageEditMask.Create(new QwenImageParams { Mask = Image(3, 1, i => i / 2f) }, source);
        var generated = Image(3, 1, _ => 1f, new[] { .75f, .75f, .75f });
        var output = plan.Composite(generated);
        Assert.Equal(original.Take(3), output.Pixels.Take(3));
        Assert.Equal(0f, output.Alpha[0]);
        Assert.Equal(new[] { .6f, .7f, .8f }, output.Pixels.Skip(3).Take(3));
        Assert.Equal(.5f, output.Alpha[1]);
        Assert.Equal(new[] { 1f, 1f, 1f }, output.Pixels.Skip(6));
        Assert.Equal(.75f, output.Alpha[2]);
        Assert.Equal(original, source.Pixels);
        Assert.NotSame(source.Pixels, output.Pixels);
        Assert.NotSame(source.Alpha, output.Alpha);
    }

    [Fact]
    public void TransparentGenerationRetainsOpaqueUnselectedCanvas()
    {
        var source = Image(2, 1, _ => .25f);
        var mask = QwenImageEditMask.Create(new QwenImageParams { Mask = Image(2, 1, i => i) }, source);
        var output = mask.Composite(Image(2, 1, _ => 1f, new[] { 0f, 0f }));
        Assert.Equal(new[] { 1f, 0f }, output.Alpha);
    }

    [Fact]
    public void CropUsesOneTransformForReferenceMaskLatentsAndFinalCanvas()
    {
        var source = Image(9, 7, i => i / 64f, Enumerable.Range(0, 63).Select(i => i / 64f).ToArray());
        var mask = Image(9, 7, i => i == 4 * 9 + 7 ? 1f : 0f);
        var plan = QwenImageEditMask.Create(new QwenImageParams { Mask = mask, MaskCrop = true, MaskCropPadding = 1 }, source);
        Assert.Equal((6, 3, 3, 3), (plan.X, plan.Y, plan.Width, plan.Height));
        Assert.Equal(source.Pixels[(3 * 9 + 6) * 3], plan.Reference.Pixels[0]);
        Assert.Equal(source.Alpha[3 * 9 + 6], plan.Reference.Alpha[0]);
        Assert.Equal((96, 96), plan.SamplingDimensions(288, 224));
        Assert.Equal(new[] { 0f, 0f, 0f, 0f, 1f, 0f, 0f, 0f, 0f }, plan.LatentWeights(3, 3));
        var output = plan.Composite(Image(3, 3, _ => 1f));
        Assert.Equal((9, 7), (output.Width, output.Height));
        for (int i = 0; i < 63; i++)
            for (int c = 0; c < 3; c++)
                Assert.Equal(i == 4 * 9 + 7 ? 1f : source.Pixels[i * 3 + c], output.Pixels[i * 3 + c]);
    }

    [Fact]
    public void CropAtEdgeClipsContextAndKeepsThinSelectionAfterLatentReduction()
    {
        var source = Image(64, 32, _ => .25f);
        var mask = Image(64, 32, i => i == 31 * 64 + 63 ? 1f : 0f);
        var full = QwenImageEditMask.Create(new QwenImageParams { Mask = mask }, source);
        Assert.Equal(new[] { 0f, 0f, 0f, 0f, 0f, 0f, 0f, 1f }, full.LatentWeights(4, 2));
        var cropped = QwenImageEditMask.Create(new QwenImageParams { Mask = mask, MaskCrop = true, MaskCropPadding = 16 }, source);
        Assert.Equal((47, 15, 17, 17), (cropped.X, cropped.Y, cropped.Width, cropped.Height));
        Assert.Equal((32, 32), cropped.SamplingDimensions(64, 32));
        Assert.Contains(1f, cropped.LatentWeights(2, 2));
    }

    [Fact]
    public void CropBoundsFollowInvertedAlphaSelection()
    {
        var source = Image(5, 3, _ => .25f);
        var selection = Image(5, 3, _ => 0f, Enumerable.Range(0, 15).Select(i => i == 7 ? 1f : 0f).ToArray());
        var plan = QwenImageEditMask.Create(new QwenImageParams
        {
            Mask = selection, MaskMode = QwenImageMaskMode.Alpha, MaskInvert = true,
            MaskCrop = true, MaskCropPadding = 0
        }, source);
        Assert.Equal((2, 1, 1, 1), (plan.X, plan.Y, plan.Width, plan.Height));
        Assert.Equal((32, 32), plan.SamplingDimensions(32, 32));
        Assert.All(plan.LatentWeights(2, 2), weight => Assert.Equal(1f, weight));
    }

    [Fact]
    public void CompositeResizesGeneratedCropButNeverResamplesOriginalProtectedCanvas()
    {
        var source = Image(7, 5, i => i / 37f, Enumerable.Range(0, 35).Select(i => i / 37f).ToArray());
        var plan = QwenImageEditMask.Create(new QwenImageParams
        {
            Mask = Image(7, 5, i => i == 17 ? 1f : 0f), MaskCrop = true, MaskCropPadding = 1
        }, source);
        var output = plan.Composite(Image(32, 32, _ => 1f, Enumerable.Repeat(1f, 1024).ToArray()));
        Assert.Equal((7, 5), (output.Width, output.Height));
        for (int i = 0; i < 35; i++)
        {
            Assert.Equal(i == 17 ? 1f : source.Alpha[i], output.Alpha[i]);
            for (int c = 0; c < 3; c++) Assert.Equal(i == 17 ? 1f : source.Pixels[i * 3 + c], output.Pixels[i * 3 + c]);
        }
    }

    [Fact]
    public void FeatherNeverChangesPixelsOutsideOriginalSelection()
    {
        var source = Image(9, 9, _ => .1234567f);
        var selection = Image(9, 9, i => i % 9 is >= 2 and <= 6 && i / 9 is >= 2 and <= 6 ? 1f : 0f);
        var plan = QwenImageEditMask.Create(new QwenImageParams { Mask = selection, MaskFeather = 1 }, source);
        var output = plan.Composite(Image(9, 9, _ => 1f));
        Assert.Equal(1f, plan.Weights[4 * 9 + 4]);
        Assert.InRange(plan.Weights[2 * 9 + 2], .44f, .45f);
        for (int i = 0; i < 81; i++)
            if (selection.Pixels[i * 3] == 0f)
            {
                Assert.Equal(0f, plan.Weights[i]);
                Assert.Equal(source.Pixels[i * 3], output.Pixels[i * 3]);
            }
    }

    [Theory]
    [InlineData(1, 1f / 3f)]
    [InlineData(1024, 1f / 2049f)]
    public void FeatherHandlesSingleRowAndRadiusLargerThanCanvas(int radius, float expected)
    {
        var source = Image(3, 1, _ => .25f);
        var selection = Image(3, 1, i => i == 1 ? 1f : 0f);
        var plan = QwenImageEditMask.Create(new QwenImageParams { Mask = selection, MaskFeather = radius }, source);
        Assert.Equal(new[] { 0f, expected, 0f }, plan.Weights);
    }

    [Theory]
    [InlineData(1f)]
    [InlineData(.875f)]
    [InlineData(.5f)]
    [InlineData(.02f)]
    [InlineData(0f)]
    public void ReinjectionFollowsFixedNoiseFlowAtRequestedSigma(float sigma)
    {
        var original = Enumerable.Range(0, 192).Select(i => (float)(i - 64)).ToArray();
        var noise = Enumerable.Range(0, 192).Select(i => (float)(192 - i)).ToArray();
        var latents = Enumerable.Repeat(-10f, 192).ToArray();
        QwenImageEditMask.Reinject(latents, original, noise, new[] { 0f, .5f, 1f }, sigma);
        for (int i = 0; i < latents.Length; i++)
        {
            float originalFlow = (1 - sigma) * original[i] + sigma * noise[i];
            float expected = i < 64 ? originalFlow : i < 128 ? .5f * (-10f + originalFlow) : -10f;
            Assert.Equal(expected, latents[i]);
        }
    }

    [Fact]
    public void EulerConstraintUsesNextSigmaAndFinishesAtOriginalLatent()
    {
        var source = Enumerable.Repeat(4f, 128).ToArray();
        var noise = Enumerable.Repeat(-2f, 128).ToArray();
        var state = (float[])noise.Clone();
        float[] schedule = { 1f, .75f, .25f, 0f };
        for (int step = 0; step < schedule.Length - 1; step++)
        {
            for (int i = 0; i < state.Length; i++) state[i] += (schedule[step + 1] - schedule[step]) * 10f;
            QwenImageEditMask.Reinject(state, source, noise, new[] { 0f, 1f }, schedule[step + 1]);
            Assert.Equal(4f * (1 - schedule[step + 1]) - 2f * schedule[step + 1], state[0]);
        }
        Assert.Equal(4f, state[0]);
        Assert.Equal(-12f, state[64]);
    }

    [Fact]
    public void ReinjectionAllocatesNoPerStepBuffers()
    {
        var state = new float[64 * 4]; var source = new float[state.Length]; var noise = new float[state.Length];
        float[] mask = { 0f, .5f, 1f, 0f };
        for (int i = 0; i < 20; i++) QwenImageEditMask.Reinject(state, source, noise, mask, .5f);
        long start = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) QwenImageEditMask.Reinject(state, source, noise, mask, .5f);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - start);
    }

    [Fact]
    public void EmptyMaskPipelineReturnsOwnedExactCopyWithoutLoadingModelOrRequiringProjector()
    {
        // The family has no components (no encoder, transformer, VAE or projector): any inference access would fail, so
        // running the diffusion runtime's pipeline verifies that it bypasses them.
        using var pipeline = new DiffusionPipeline(new QwenImage21Family(BackendType.GgmlMetal, QwenImageVariant.Base),
            new DiffusionLoad { Backend = BackendType.GgmlMetal }, new ResidencyPlanner(() => null, _ => { }), _ => { });
        var source = Image(3, 2, i => i / 7f, new[] { 0f, .1f, .3f, .5f, .8f, 1f });
        var output = pipeline.Run(QwenImage21DiffusionModel.RequestOf("change the color",
            new QwenImageParams { Mask = Image(3, 2, _ => 0f), Width = 32, Height = 32 }, new[] { source })).Image;
        Assert.Equal((3, 2), (output.Width, output.Height));
        Assert.Equal(source.Pixels, output.Pixels);
        Assert.Equal(source.Alpha, output.Alpha);
        Assert.NotSame(source.Pixels, output.Pixels);
    }

    [Fact]
    public void FullMaskHasUnitLatentCoverageAndNoFeatherAttenuation()
    {
        var source = Image(5, 3, _ => .25f);
        var plan = QwenImageEditMask.Create(new QwenImageParams { Mask = Image(5, 3, _ => 1f), MaskFeather = 1024, MaskCrop = true }, source);
        Assert.True(plan.IsFull);
        Assert.False(plan.IsEmpty);
        Assert.All(plan.LatentWeights(4, 2), value => Assert.Equal(1f, value));
        Assert.Same(source, plan.Reference);
    }

    [Fact]
    public void MaskValidationRejectsMissingSourceGeometryChannelsAndNonFiniteSamples()
    {
        var source = Image(2, 2, _ => 0f);
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { Mask = source }, null));
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { Mask = Image(2, 1, _ => 0f) }, source));
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { Mask = source, MaskMode = QwenImageMaskMode.Alpha }, source));
        foreach (float bad in new[] { float.NaN, float.PositiveInfinity, -.01f, 1.01f })
            Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { Mask = Image(2, 2, _ => bad) }, source));
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { Mask = source, MaskMode = (QwenImageMaskMode)99 }, source));
    }

    [Fact]
    public void UnmaskedRequestsRejectStrayProcessingOptions()
    {
        Assert.Null(QwenImageEditMask.Create(new QwenImageParams(), null));
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { MaskMode = QwenImageMaskMode.Alpha }, null));
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { MaskInvert = true }, null));
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { MaskCrop = true }, null));
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { MaskFeather = 1 }, null));
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Create(new QwenImageParams { MaskCropPadding = 32 }, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenImageEditMask.Create(new QwenImageParams { MaskFeather = -1 }, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenImageEditMask.Create(new QwenImageParams { MaskFeather = 1025 }, null));
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenImageEditMask.Create(new QwenImageParams { MaskCropPadding = 16385 }, null));
    }

    [Fact]
    public void ReinjectionRejectsBadGeometryAndSigma()
    {
        Assert.Throws<ArgumentException>(() => QwenImageEditMask.Reinject(new float[63], new float[63], new float[63], new float[1], .5f));
        foreach (float bad in new[] { -.1f, 1.1f, float.NaN })
            Assert.Throws<ArgumentOutOfRangeException>(() => QwenImageEditMask.Reinject(new float[64], new float[64], new float[64], new float[1], bad));
    }
}
