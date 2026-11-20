// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The shared EditMask beyond Qwen-Image's binding (QwenImageMaskTests keeps covering that, unchanged): Qwen's
// settings map onto the shared options exactly, reinjection takes any number of values a weight, and a mask already in
// a family's row layout (one weight a value, MiniMax-H3-Image's patchified 96-value rows) follows the same flow.
using TensorSharp.Models.QwenImage;
using TensorSharp.Models.SharedDiffusion;

namespace InferenceWeb.Tests;

public sealed class SharedDiffusionEditMaskTests
{
    private static RgbImage Image(int width, int height, Func<int, float> values, float[]? alpha = null) =>
        new(width, height, Enumerable.Range(0, width * height).SelectMany(i => Enumerable.Repeat(values(i), 3)).ToArray(), alpha);

    [Fact]
    public void QwenSettingsMapOntoTheSharedOptions()
    {
        var mask = Image(4, 4, i => i % 2);
        var p = new QwenImageParams
        {
            Mask = mask, MaskMode = QwenImageMaskMode.Alpha, MaskInvert = true, MaskFeather = 3, MaskCrop = true, MaskCropPadding = 7,
        };
        EditMaskOptions o = QwenImageEditMask.Options(p);
        Assert.Same(mask, o.Mask);
        Assert.Equal(EditMaskMode.Alpha, o.MaskMode);
        Assert.Equal((true, 3, true, 7), (o.MaskInvert, o.MaskFeather, o.MaskCrop, o.MaskCropPadding));
        // The defaults agree, so an unmasked Qwen request is an unmasked shared one.
        EditMaskOptions defaults = QwenImageEditMask.Options(new QwenImageParams());
        Assert.Equal(new EditMaskOptions(), defaults);
        Assert.Null(EditMask.Create(defaults, null));
        // A mode outside Qwen's enum stays outside the shared one and is refused alike.
        Assert.Throws<ArgumentException>(() => EditMask.Create(new EditMaskOptions { Mask = mask, MaskMode = (EditMaskMode)99 }, mask));
    }

    [Fact]
    public void SharedAndQwenMasksAgreeOnEveryProduct()
    {
        var source = Image(9, 7, i => i / 64f, Enumerable.Range(0, 63).Select(i => i / 64f).ToArray());
        var selection = Image(9, 7, i => i % 9 is >= 3 and <= 5 && i / 9 is >= 2 and <= 4 ? 1f : 0f);
        var p = new QwenImageParams { Mask = selection, MaskFeather = 1, MaskCrop = true, MaskCropPadding = 1 };
        EditMask qwen = QwenImageEditMask.Create(p, source);
        EditMask shared = EditMask.Create(new EditMaskOptions { Mask = selection, MaskFeather = 1, MaskCrop = true, MaskCropPadding = 1 }, source);
        Assert.Equal(qwen.Weights, shared.Weights);
        Assert.Equal((qwen.X, qwen.Y, qwen.Width, qwen.Height), (shared.X, shared.Y, shared.Width, shared.Height));
        Assert.Equal(qwen.LatentWeights(3, 2), shared.LatentWeights(3, 2));
        Assert.Equal(qwen.SamplingDimensions(288, 224), shared.SamplingDimensions(288, 224));
        Assert.Equal(qwen.Composite(Image(5, 5, _ => 1f)).Pixels, shared.Composite(Image(5, 5, _ => 1f)).Pixels);
    }

    [Fact]
    public void SamplingDimensionsSnapToTheFamilysMultiple()
    {
        var source = Image(100, 50, _ => .5f);
        EditMask mask = EditMask.Create(new EditMaskOptions { Mask = Image(100, 50, i => i == 0 ? 1f : 0f) }, source);
        Assert.Equal((2048, 1024), mask.SamplingDimensions(2048, 1024));
        Assert.Equal((2048, 1024), mask.SamplingDimensions(2040, 1000, multiple: 64));
        Assert.Equal((32, 32), mask.SamplingDimensions(10, 10));
    }

    [Fact]
    public void ReinjectionOverSixtyFourChannelsIsQwensBitForBit()
    {
        var random = new Random(5);
        float[] Rand(int n) => Enumerable.Range(0, n).Select(_ => (float)(random.NextDouble() * 4 - 2)).ToArray();
        float[] mask = { 0f, .25f, 1f, .5f, 0f };
        float[] source = Rand(64 * mask.Length), noise = Rand(source.Length), state = Rand(source.Length);
        float[] qwen = (float[])state.Clone(), shared = (float[])state.Clone();
        QwenImageEditMask.Reinject(qwen, source, noise, mask, .4375f);
        EditMask.Reinject(shared, source, noise, mask, 64, .4375f);
        Assert.Equal(qwen.Select(BitConverter.SingleToUInt32Bits), shared.Select(BitConverter.SingleToUInt32Bits));
    }

    [Theory]
    [InlineData(1)]
    [InlineData(3)]
    [InlineData(24)]
    [InlineData(96)]
    public void ReinjectionFollowsTheFixedNoiseFlowAtAnyWidthAWeight(int channels)
    {
        float[] mask = { 0f, .5f, 1f, .25f };
        int n = mask.Length * channels;
        float[] source = Enumerable.Range(0, n).Select(i => (float)(i - 7)).ToArray();
        float[] noise = Enumerable.Range(0, n).Select(i => (float)(n - i) / 3f).ToArray();
        float[] latents = Enumerable.Repeat(-10f, n).ToArray();
        const float sigma = .375f;
        EditMask.Reinject(latents, source, noise, mask, channels, sigma);
        for (int i = 0; i < n; i++)
        {
            float m = mask[i / channels];
            float original = (1 - sigma) * source[i] + sigma * noise[i];
            float expected = m == 0f ? original : m == 1f ? -10f : m * -10f + (1 - m) * original;
            Assert.Equal(expected, latents[i], 5);
        }
    }

    [Fact]
    public void APerValueMaskEndsAtTheSourceExactlyWhereItKeeps()
    {
        // A mask already in the rows' layout (one weight a value), as a family whose tokens interleave a cell's
        // channels hands it over: the last Euler step (sigma 0) leaves the source in every kept value.
        float[] mask = Enumerable.Range(0, 96).Select(i => i % 4 == 0 ? 0f : 1f).ToArray();
        float[] source = Enumerable.Range(0, 96).Select(i => i * .03125f - 1f).ToArray();
        float[] noise = Enumerable.Range(0, 96).Select(i => (i % 7) - 3f).ToArray();
        float[] x = (float[])noise.Clone();
        float[] sigmas = { 1f, .75f, .25f, 0f };
        for (int step = 0; step < 3; step++)
        {
            for (int i = 0; i < x.Length; i++) x[i] += (sigmas[step + 1] - sigmas[step]) * 2f;
            EditMask.Reinject(x, source, noise, mask, 1, sigmas[step + 1]);
        }
        for (int i = 0; i < x.Length; i++)
            if (mask[i] == 0f) Assert.Equal(source[i], x[i]);
            else Assert.Equal(noise[i] - 2f, x[i], 5);
    }

    [Fact]
    public void ReinjectionRefusesBadWidthsGeometryAndSigma()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => EditMask.Reinject(new float[4], new float[4], new float[4], new float[4], 0, .5f));
        Assert.Throws<ArgumentException>(() => EditMask.Reinject(new float[95], new float[95], new float[95], new float[1], 96, .5f));
        Assert.Throws<ArgumentException>(() => EditMask.Reinject(new float[96], new float[95], new float[96], new float[1], 96, .5f));
        Assert.Throws<ArgumentOutOfRangeException>(() => EditMask.Reinject(new float[24], new float[24], new float[24], new float[1], 24, 1.5f));
    }

    [Fact]
    public void ReinjectionOfAnyWidthAllocatesNothing()
    {
        var state = new float[24 * 8]; var source = new float[state.Length]; var noise = new float[state.Length];
        float[] mask = { 0f, .5f, 1f, 0f, .25f, 1f, 0f, .75f };
        for (int i = 0; i < 20; i++) EditMask.Reinject(state, source, noise, mask, 24, .5f);
        long start = GC.GetAllocatedBytesForCurrentThread();
        for (int i = 0; i < 100; i++) EditMask.Reinject(state, source, noise, mask, 24, .5f);
        Assert.Equal(0, GC.GetAllocatedBytesForCurrentThread() - start);
    }
}
