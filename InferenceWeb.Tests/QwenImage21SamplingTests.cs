using System.Runtime.InteropServices;
using TensorSharp.Models.QwenImage;

namespace InferenceWeb.Tests;

public class QwenImage21SamplingTests
{
    [Theory]
    [InlineData(1f, 0.875f)]
    [InlineData(0.75f, 0.5f)]
    [InlineData(0.5f, 0f)]
    public void PreviewRecoversCleanFlowLatentWithoutChangingSamplingState(float sigma, float nextSigma)
    {
        // For an exact linear flow, x_t = (1-t)*clean + t*noise and
        // velocity = noise-clean. Preview must recover clean after any Euler
        // interval, including an early step whose actual state is mostly noise.
        float[] clean = { -3f, 0f, 1f, 4f };
        float[] noise = { 1f, -2f, 5f, 0f };
        float[] velocity = clean.Zip(noise, (c, n) => n - c).ToArray();
        float[] updated = clean.Zip(noise, (c, n) => (1f - sigma) * c + sigma * n).ToArray();
        for (int i = 0; i < updated.Length; i++) updated[i] += (nextSigma - sigma) * velocity[i];
        float[] expectedState = (float[])updated.Clone();
        float[] expectedVelocity = (float[])velocity.Clone();

        var preview = QwenImage21Sampling.PreviewLatents(updated, velocity, nextSigma);

        Assert.Equal(clean, preview);
        Assert.Equal(expectedState, updated);
        Assert.Equal(expectedVelocity, velocity);
        Assert.NotSame(updated, preview);
        if (nextSigma == 0f) Assert.Equal(updated, preview);
        else Assert.NotEqual(updated, preview);
    }

    [Fact]
    public void PhiloxNoiseMatchesStableDiffusionCppCudaRng()
    {
        // Golden vector from unchanged sd.cpp c678dfe core/rng_philox.hpp, seed=42.
        float[] expected = { .194018871f, 2.16137385f, -.172050610f, .849060059f,
            -1.92439914f, .652985454f, -.649441063f, -.817524731f, .527964652f,
            -1.27534986f, -1.66212630f, -.303313762f, -.0925699323f, .199237078f,
            -1.12043273f, 1.85765874f };
        var actual = QwenImage21Sampling.Noise(expected.Length, 42);
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-6f);
        Assert.NotEqual(actual, QwenImage21Sampling.Noise(expected.Length, 42L + (1L << 32)));
    }

    // =====================================================================================
    // initial noise: text-to-image keeps the seed's sequence, an edit follows its references
    // =====================================================================================

    /// <summary>Text-to-image noise as it was before edits got their own stream: the oracle
    /// that stream 0 must equal bit for bit. Compared in-process, not against a pinned hash,
    /// because Log and Sin may differ by an ulp between platforms' math libraries.</summary>
    internal static float[] SeedOnlyNoise(int count, long seed)
    {
        var result = new float[count];
        const float inv32 = 2.3283064e-10f;
        const float inv32Tau = inv32 * 6.2831855f;
        for (int i = 0; i < count; i++)
        {
            uint a = 0, b = 0, c = (uint)i, d = 0;
            uint k0 = (uint)seed, k1 = (uint)((ulong)seed >> 32);
            for (int round = 0; round < 10; round++)
            {
                ulong p0 = (ulong)a * 0xD2511F53u, p1 = (ulong)c * 0xCD9E8D57u;
                (a, b, c, d) = ((uint)(p1 >> 32) ^ b ^ k0, (uint)p1, (uint)(p0 >> 32) ^ d ^ k1, (uint)p0);
                k0 = unchecked(k0 + 0x9E3779B9u);
                k1 = unchecked(k1 + 0xBB67AE85u);
            }
            float u = (float)a * inv32 + inv32 / 2;
            float v = (float)b * inv32Tau + inv32Tau / 2;
            result[i] = (float)(Math.Sqrt(-2f * Math.Log(u)) * Math.Sin(v));
        }
        return result;
    }

    internal static bool SameBits(float[] x, float[] y) =>
        MemoryMarshal.AsBytes(x.AsSpan()).SequenceEqual(MemoryMarshal.AsBytes(y.AsSpan()));

    internal static double Correlation(float[] x, float[] y)
    {
        double mx = x.Average(v => (double)v), my = y.Average(v => (double)v), sxy = 0, sxx = 0, syy = 0;
        for (int i = 0; i < x.Length; i++)
        {
            double dx = x[i] - mx, dy = y[i] - my;
            sxy += dx * dy; sxx += dx * dx; syy += dy * dy;
        }
        return sxy / Math.Sqrt(sxx * syy);
    }

    /// <summary>A decoded picture: every value is a byte over 255, as ImageIO.Decode gives.</summary>
    internal static RgbImage Decoded(int width, int height, int seed, float[]? alpha = null)
    {
        var random = new Random(seed);
        var pixels = new float[width * height * 3];
        for (int i = 0; i < pixels.Length; i++) pixels[i] = random.Next(256) / 255f;
        return new RgbImage(width, height, pixels, alpha);
    }

    private static RgbImage Ramp(int width, int height, Func<int, float> value) =>
        new(width, height, Enumerable.Range(0, width * height * 3).Select(value).ToArray());

    // A 1024x1024 latent: 64 x 64 tokens of 64 channels. 1/sqrt(n) is about 0.002, so 0.02 is
    // about ten standard deviations of the correlation of two independent draws.
    private const int LatentValues = 64 * 64 * 64;
    private const double Uncorrelated = 0.02;

    [Fact]
    public void TextToImageNoiseIsTodaysSequenceBitForBit()
    {
        foreach (int count in new[] { 64, 16 * 16 * 64, LatentValues })
            foreach (long seed in new[] { 0L, 1L, 42L, -1L, long.MaxValue, 1L << 32 })
            {
                float[] expected = SeedOnlyNoise(count, seed);
                Assert.True(SameBits(expected, QwenImage21Sampling.Noise(count, seed)), $"count {count}, seed {seed}");
                Assert.True(SameBits(expected, QwenImage21Sampling.Noise(count, seed, 0)), $"count {count}, seed {seed}");
            }
    }

    [Fact]
    public void TextToImageStartsFromTheSeedsNoiseAndSoDoesAnEditInSeedMode()
    {
        // The pipeline's own call: InitialLatents with the stream NoiseStream gives it.
        var source = Decoded(8, 8, 7);
        float[] expected = QwenImage21Pipeline.ToTokens(SeedOnlyNoise(4 * 6 * 64, 3), 4, 6);
        ulong textToImage = QwenImage21Sampling.NoiseStream(Array.Empty<RgbImage>(), followReferences: true);
        ulong seedMode = QwenImage21Sampling.NoiseStream(new[] { source }, followReferences: false);
        Assert.Equal(0UL, textToImage);
        Assert.Equal(0UL, seedMode);
        Assert.True(SameBits(expected, QwenImage21Sampling.InitialLatents(3, textToImage, 4, 6)));

        ulong edit = QwenImage21Sampling.NoiseStream(new[] { source }, followReferences: true);
        Assert.NotEqual(0UL, edit);
        Assert.Equal(QwenImage21Sampling.ReferenceStream(new[] { source }), edit);
        Assert.False(SameBits(expected, QwenImage21Sampling.InitialLatents(3, edit, 4, 6)));
    }

    /// <summary>
    /// The failure this fixes: a picture drawn at seed 0 and edited at seed 0, at the same size,
    /// began from the very noise that had become the picture. An edit's noise is now as unlike
    /// the text-to-image noise at its seed as two seeds' noise is, and still a standard normal.
    /// </summary>
    [Fact]
    public void AnEditsNoiseIsIndependentOfTheTextToImageNoiseAtItsSeed()
    {
        ulong stream = QwenImage21Sampling.ReferenceStream(new[] { Decoded(8, 8, 7) });
        float[] textToImage = QwenImage21Sampling.Noise(LatentValues, 0);
        float[] edit = QwenImage21Sampling.Noise(LatentValues, 0, stream);

        Assert.InRange(Math.Abs(Correlation(textToImage, edit)), 0, Uncorrelated);
        Assert.Equal(0, textToImage.Zip(edit).Count(pair => pair.First == pair.Second));
        double mean = edit.Average(v => (double)v);
        double variance = edit.Average(v => (v - mean) * (v - mean));
        Assert.InRange(mean, -0.01, 0.01);
        Assert.InRange(variance, 0.98, 1.02);

        // Not the text-to-image noise of a seed derived from the stream either: mixing the
        // stream into the Philox key would make an edit at seed s some picture's noise at s ^ id.
        float[] alias = QwenImage21Sampling.Noise(LatentValues, 0 ^ (long)stream);
        Assert.InRange(Math.Abs(Correlation(alias, edit)), 0, Uncorrelated);
    }

    [Fact]
    public void ChainedEditsAtOneSeedStartFromDifferentNoise()
    {
        // B is A with one region changed, as an edit's result is; editing B at A's seed must
        // not start from the noise that edited A.
        var a = Decoded(16, 16, 7);
        var changed = (float[])a.Pixels.Clone();
        for (int i = 0; i < 4 * 16 * 3; i++) changed[i] = 1f - changed[i];
        var b = new RgbImage(16, 16, changed);

        ulong streamA = QwenImage21Sampling.ReferenceStream(new[] { a });
        ulong streamB = QwenImage21Sampling.ReferenceStream(new[] { b });
        Assert.NotEqual(streamA, streamB);
        Assert.InRange(Math.Abs(Correlation(QwenImage21Sampling.Noise(LatentValues, 0, streamA),
            QwenImage21Sampling.Noise(LatentValues, 0, streamB))), 0, Uncorrelated);
    }

    [Fact]
    public void TheStreamIsStableForEqualReferences()
    {
        var first = Decoded(32, 16, 11);
        var second = new RgbImage(32, 16, (float[])first.Pixels.Clone());
        ulong stream = QwenImage21Sampling.ReferenceStream(new[] { first });
        Assert.Equal(stream, QwenImage21Sampling.ReferenceStream(new[] { second }));
        Assert.True(SameBits(QwenImage21Sampling.Noise(1024, 5, stream), QwenImage21Sampling.Noise(1024, 5, stream)));

        // The v1 encoding, pinned: changing it changes every edit's picture, so it must be a
        // deliberate new version, never an accident. Values from an independent Python
        // (hashlib + NumPy float32) evaluation of the documented encoding and of Philox.
        var ramp = Ramp(4, 2, i => i / 255f);
        var reversed = Ramp(4, 2, i => (255 - i) / 255f);
        Assert.Equal(0x9DADB36FC2CAF8D2UL, QwenImage21Sampling.ReferenceStream(new[] { ramp }));
        Assert.Equal(0xA2BB8C9373217B2AUL, QwenImage21Sampling.ReferenceStream(new[] { ramp, reversed }));
        float[] expected = { 2.4735303f, .085433364f, 1.3570157f, -1.2013689f };
        float[] actual = QwenImage21Sampling.Noise(4, 42, QwenImage21Sampling.ReferenceStream(new[] { ramp }));
        for (int i = 0; i < expected.Length; i++)
            Assert.InRange(Math.Abs(actual[i] - expected[i]), 0, 1e-6f);
    }

    [Fact]
    public void TheStreamFollowsEveryReference()
    {
        var a = Decoded(4, 4, 1);
        var b = Decoded(4, 4, 2);
        ulong stream = QwenImage21Sampling.ReferenceStream(new[] { a });

        var nudged = (float[])a.Pixels.Clone();
        nudged[5] = nudged[5] < 1f ? nudged[5] + 1 / 255f : nudged[5] - 1 / 255f;
        var translucent = Enumerable.Repeat(1f, 16).ToArray();
        translucent[3] = 254 / 255f;
        ulong[] others =
        {
            QwenImage21Sampling.ReferenceStream(new[] { new RgbImage(4, 4, nudged) }),
            QwenImage21Sampling.ReferenceStream(new[] { a, b }),
            QwenImage21Sampling.ReferenceStream(new[] { b, a }),
            QwenImage21Sampling.ReferenceStream(new[] { a, a }),
            QwenImage21Sampling.ReferenceStream(new[] { new RgbImage(2, 8, a.Pixels) }),
            QwenImage21Sampling.ReferenceStream(new[] { new RgbImage(4, 4, a.Pixels, translucent) }),
        };
        Assert.DoesNotContain(stream, others);
        Assert.Equal(others.Length, others.Distinct().Count());
    }

    [Fact]
    public void ThePixelsAreHashedAsAPngStoresThem()
    {
        // In memory, a model-made picture is VAE floats; reloaded, it is the PNG's bytes. Both
        // must give one stream, so a picture edited where it was made and after a download agree.
        var random = new Random(3);
        float[] pixels = Enumerable.Range(0, 6 * 5 * 3).Select(_ => (float)random.NextDouble()).ToArray();
        float[] alpha = Enumerable.Range(0, 6 * 5).Select(i => i % 7 == 0 ? (float)random.NextDouble() : 1f).ToArray();
        foreach (var picture in new[] { new RgbImage(6, 5, pixels), new RgbImage(6, 5, pixels, alpha) })
        {
            var reloaded = ImageIO.Decode(ImageIO.EncodePng(picture), preserveAlpha: true);
            Assert.Equal(QwenImage21Sampling.ReferenceStream(new[] { picture }), QwenImage21Sampling.ReferenceStream(new[] { reloaded }));
        }

        // Values that store as the same byte are the same picture; out-of-range values clamp.
        var exact = Ramp(3, 1, i => i * 40 / 255f);
        var near = Ramp(3, 1, i => i * 40 / 255f + 0.4f / 255f);
        Assert.Equal(QwenImage21Sampling.ReferenceStream(new[] { exact }), QwenImage21Sampling.ReferenceStream(new[] { near }));
        Assert.Equal(QwenImage21Sampling.ReferenceStream(new[] { Ramp(3, 1, _ => 1f) }),
            QwenImage21Sampling.ReferenceStream(new[] { Ramp(3, 1, _ => 1.5f) }));
        Assert.Equal(QwenImage21Sampling.ReferenceStream(new[] { Ramp(3, 1, _ => 0f) }),
            QwenImage21Sampling.ReferenceStream(new[] { Ramp(3, 1, _ => float.NaN) }));
    }

    [Fact]
    public void AnOpaqueAlphaPlaneHashesAsNoPlane()
    {
        // The VAE reads a missing alpha plane as opaque, and the hosts decode with alpha.
        var picture = Decoded(4, 4, 9);
        ulong none = QwenImage21Sampling.ReferenceStream(new[] { picture });
        Assert.Equal(none, QwenImage21Sampling.ReferenceStream(new[] { new RgbImage(4, 4, picture.Pixels, Enumerable.Repeat(1f, 16).ToArray()) }));
        // 0.999 stores as 255, so it is opaque too.
        Assert.Equal(none, QwenImage21Sampling.ReferenceStream(new[] { new RgbImage(4, 4, picture.Pixels, Enumerable.Repeat(.999f, 16).ToArray()) }));
    }

    [Theory]
    [InlineData(null, true)]
    [InlineData("", true)]
    [InlineData(" references ", true)]
    [InlineData("References", true)]
    [InlineData("seed", false)]
    [InlineData("SEED", false)]
    public void EditNoiseSettingAcceptsReferencesAndSeed(string? value, bool followsReferences)
    {
        Assert.Equal(followsReferences, QwenImage21Sampling.EditNoiseFollowsReferences(value!));
    }

    [Theory]
    [InlineData("sd")]
    [InlineData("0")]
    [InlineData("reference")]
    public void EditNoiseSettingRefusesAnythingElse(string value)
    {
        var error = Assert.Throws<ArgumentException>(() => QwenImage21Sampling.EditNoiseFollowsReferences(value));
        Assert.Contains("TS_QWEN21_EDIT_NOISE", error.Message, StringComparison.Ordinal);
        Assert.Contains("references, seed", error.Message, StringComparison.Ordinal);
    }

    [Theory]
    [InlineData(256, .9843579531, .8282203674, .6143688560, .3408319354, .0601277351)]
    [InlineData(4096, .9869637489, .8528681993, .6566662788, .3819332719, .0678805709)] // 1024²
    [InlineData(8192, .9892514348, .8756618500, .6988655925, .4275377989, .0776016116)]
    [InlineData(16384, .9926459789, .9116604924, .7724375129, .5205857754, .1022295356)] // 2048²
    public void FortyStepScheduleMatchesOfficialQwenImage21Scheduler(
        int tokens, double first, double quarter, double middle, double threeQuarter, double penultimate)
    {
        // Golden values evaluated by the unmodified NumPy time_shift and
        // stretch_shift_to_terminal methods in Diffusers 6256aa7666cedd47443adc8f82da9a10e110b09c,
        // using https://huggingface.co/Qwen/Qwen-Image-2.1/blob/main/scheduler/scheduler_config.json.
        // Covers both official anchors and extrapolation to native 2K (16384 tokens).
        var actual = QwenImage21Sampling.Sigmas(40, tokens);
        Assert.Equal(41, actual.Length);
        Assert.Equal(1f, actual[0]);
        int[] indices = { 1, 10, 20, 30, 38 };
        double[] expected = { first, quarter, middle, threeQuarter, penultimate };
        for (int i = 0; i < indices.Length; i++)
            Assert.InRange(Math.Abs(actual[indices[i]] - expected[i]), 0, 2e-7);
        Assert.Equal(.02f, actual[^2]);
        Assert.Equal(0f, actual[^1]);
    }

    [Theory]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(4)]
    [InlineData(8)]
    [InlineData(40)]
    [InlineData(100)]
    public void ResolutionSchedulesStayFiniteAndStrictlyDecrease(int steps)
    {
        foreach (int tokens in new[] { 4, 256, 4096, 8192, 16384, 65536 })
        {
            var actual = QwenImage21Sampling.Sigmas(steps, tokens);
            Assert.Equal(steps + 1, actual.Length);
            Assert.Equal(1f, actual[0]);
            Assert.Equal(0f, actual[^1]);
            for (int i = 0; i < actual.Length; i++)
            {
                Assert.True(float.IsFinite(actual[i]));
                Assert.InRange(actual[i], 0f, 1f);
                if (i > 0) Assert.True(actual[i] < actual[i - 1]);
            }
            if (steps > 1) Assert.Equal(.02f, actual[^2]);
        }
    }

    [Fact]
    public void SingleStepKeepsOneEulerUpdateWithoutDividingByZero()
    {
        Assert.Equal(new[] { 1f, 0f }, QwenImage21Sampling.Sigmas(1, 16384));
    }

    [Theory]
    [InlineData(0, 16384)]
    [InlineData(-1, 16384)]
    [InlineData(40, 0)]
    [InlineData(40, -1)]
    public void InvalidScheduleParametersAreRejected(int steps, int tokens)
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => QwenImage21Sampling.Sigmas(steps, tokens));
    }

    [Fact]
    public void LatentTransposePreservesChannelsWithoutOldPatchPacking()
    {
        float[] chw = Enumerable.Range(0, 64 * 2 * 4).Select(i => (float)i).ToArray();
        var tokens = QwenImage21Pipeline.ToTokens(chw, 2, 4);
        Assert.Equal(new float[] { 0, 8, 16, 24 }, tokens.Take(4));
        Assert.Equal(1, tokens[64]);
        Assert.Equal(chw, QwenImage21Pipeline.ToChannels(tokens, 2, 4));
    }

    [Theory]
    [InlineData(512, 0)]
    [InlineData(0, 512)]
    [InlineData(513, 512)]
    [InlineData(-32, 512)]
    public void InvalidExplicitGeometryFailsBeforeAllocating(int width, int height)
    {
        Assert.Throws<ArgumentException>(() => QwenImage21Pipeline.ResolveDimensions(
            new QwenImageParams { Width = width, Height = height }, null));
    }
}
