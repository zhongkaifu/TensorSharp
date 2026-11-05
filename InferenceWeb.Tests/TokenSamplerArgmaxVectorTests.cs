using System.Numerics;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class TokenSamplerArgmaxVectorTests
{
    private const string SimdControl = "TS_TOKEN_SAMPLER_SIMD";

    private static bool VectorPolicy(TokenSampler sampler) => (bool)typeof(TokenSampler)
        .GetField("_vectorizeArgmax", System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!
        .GetValue(sampler)!;

    [Fact]
    public void PublicConstructorDefaultsToScalarWhenControlIsUnset()
    {
        string? original = Environment.GetEnvironmentVariable(SimdControl);
        try
        {
            Environment.SetEnvironmentVariable(SimdControl, null);
            var sampler = new TokenSampler(SamplingConfig.Greedy);
            Assert.False(VectorPolicy(sampler));
            Assert.Equal(1, sampler.Sample(new[] { float.NegativeInfinity, 4f, 4f, float.NaN }));
        }
        finally { Environment.SetEnvironmentVariable(SimdControl, original); }
    }

    [Fact]
    public void PublicConstructorRequiresExactExplicitOptIn()
    {
        string? original = Environment.GetEnvironmentVariable(SimdControl);
        try
        {
            foreach (string? value in new string?[] { null, "", "0", "true", "TRUE", "01", " 1", "1 ", "yes", "2", "1" })
            {
                Environment.SetEnvironmentVariable(SimdControl, value);
                var sampler = new TokenSampler(SamplingConfig.Greedy);
                Assert.Equal(value == "1", VectorPolicy(sampler));
                Assert.Equal(0, sampler.Sample(new[] { float.NaN, float.PositiveInfinity }));
            }
        }
        finally { Environment.SetEnvironmentVariable(SimdControl, original); }
    }

    [Fact]
    public void PublicConstructorSnapshotsControlForTheSamplerLifetime()
    {
        string? original = Environment.GetEnvironmentVariable(SimdControl);
        try
        {
            Environment.SetEnvironmentVariable(SimdControl, null);
            var initialDefault = new TokenSampler(SamplingConfig.Greedy);
            Environment.SetEnvironmentVariable(SimdControl, "1");
            var explicitlyEnabled = new TokenSampler(SamplingConfig.Greedy);
            Environment.SetEnvironmentVariable(SimdControl, "0");
            var explicitlyDisabled = new TokenSampler(SamplingConfig.Greedy);
            float[] values = Enumerable.Repeat(-1f, 248320).ToArray(); values[248319] = 2;
            Assert.False(VectorPolicy(initialDefault));
            Assert.True(VectorPolicy(explicitlyEnabled));
            Assert.False(VectorPolicy(explicitlyDisabled));
            foreach (var sampler in new[] { initialDefault, explicitlyEnabled, explicitlyDisabled })
                Assert.Equal(248319, sampler.Sample(values));
            Environment.SetEnvironmentVariable(SimdControl, "1");
            Assert.False(VectorPolicy(initialDefault));
            Assert.False(VectorPolicy(explicitlyDisabled));
        }
        finally { Environment.SetEnvironmentVariable(SimdControl, original); }
    }

    // Independent, literal original ordered-compare oracle. Never calls a
    // production argmax helper or vector maximum.
    private static int Oracle(float[] values)
    {
        int index = 0;
        float maximum = values[0];
        for (int i = 1; i < values.Length; ++i)
            if (values[i] > maximum) { maximum = values[i]; index = i; }
        return index;
    }

    private static void Check(float[] values)
    {
        int[] before = Array.ConvertAll(values, BitConverter.SingleToInt32Bits);
        int expected = Oracle(values);
        Assert.Equal(expected, new TokenSampler(SamplingConfig.Greedy, false).Sample(values));
        Assert.Equal(expected, new TokenSampler(SamplingConfig.Greedy, true).Sample(values));
        Assert.Equal(before, Array.ConvertAll(values, BitConverter.SingleToInt32Bits));
    }

    [Fact]
    public void EmptyAndNullPreserveOriginalExceptions()
    {
        foreach (bool vector in new[] { false, true })
        {
            var sampler = new TokenSampler(SamplingConfig.Greedy, vector);
            Assert.Throws<IndexOutOfRangeException>(() => sampler.Sample(Array.Empty<float>()));
            Assert.Throws<NullReferenceException>(() => sampler.Sample(null!));
        }
    }

    [Fact]
    public void BoundariesAndEveryMaximumPositionRetainFirstTie()
    {
        int width = Vector<float>.Count;
        foreach (int length in new[] { 1, 2, 3, width - 1, width, width + 1, 31, 32, 33, 63, 64, 65, 127, 128, 129, 255, 256, 257 })
        for (int position = 0; position < length; position++)
        {
            float[] values = Enumerable.Repeat(-2f, length).ToArray();
            values[position] = 4;
            values[length - 1] = 4;
            Check(values);
        }
    }

    [Fact]
    public void NonFiniteSignedZeroAndSignalingNaNPayloadsRemainExact()
    {
        float[] patterns = { 0f, BitConverter.Int32BitsToSingle(int.MinValue), float.PositiveInfinity,
            float.NegativeInfinity, float.NaN, BitConverter.Int32BitsToSingle(0x7f800001),
            BitConverter.Int32BitsToSingle(unchecked((int)0xff812345)), float.Epsilon, -float.Epsilon };
        foreach (int length in new[] { 1, 2, 127, 128, 129, 513, 248320, 248323 })
        foreach (float fill in patterns)
        {
            float[] values = Enumerable.Repeat(fill, length).ToArray();
            Check(values);
            foreach (int position in new[] { 0, length / 3, length / 2, length - 1 }.Distinct())
            foreach (float special in patterns)
            {
                values[position] = special;
                Check(values);
                values[position] = fill;
            }
        }
    }

    [Fact]
    public void RandomRawFloatBitsAndFullVocabularyMatchIndependentScalar()
    {
        var random = new Random(172903);
        foreach (int length in new[] { 1, 7, 126, 127, 128, 129, 130, 1023, 1024, 1025, 16387, 248320, 248323 })
        for (int repeat = 0; repeat < 20; repeat++)
        {
            byte[] raw = new byte[length * 4]; random.NextBytes(raw);
            float[] values = new float[length]; Buffer.BlockCopy(raw, 0, values, 0, raw.Length);
            Check(values);
        }
    }

    [Fact]
    public void GreedyPenaltyPathAndOriginalLogitBitsRemainIdentical()
    {
        foreach (int length in new[] { 7, 128, 129, 248320 })
        foreach (float rep in new[] { 1f, 1.2f })
        {
            var config = new SamplingConfig { Temperature = 0, RepetitionPenalty = rep, PresencePenalty = .3f,
                FrequencyPenalty = .2f, PenaltyLastN = 5 };
            float[] logits = Enumerable.Range(0, length).Select(x => (float)(x % 17 - 8)).ToArray();
            int[] history = { 0, 2, 3, 2, length - 1, 0 };
            float[] expected = (float[])logits.Clone();
            new TokenSampler(config, false).ApplyPenalties(expected, history);
            int id = Oracle(expected);
            foreach (bool vector in new[] { false, true })
            {
                float[] copy = (float[])logits.Clone();
                Assert.Equal(id, new TokenSampler(config, vector).Sample(copy, history));
                Assert.Equal(Array.ConvertAll(logits, BitConverter.SingleToInt32Bits), Array.ConvertAll(copy, BitConverter.SingleToInt32Bits));
            }
        }
    }

    [Fact]
    public void FirstTokenAllowListStillOverridesTheGlobalMaximum()
    {
        float[] logits = new float[248320]; logits[100] = 100; logits[7] = 2;
        var config = new SamplingConfig { Temperature = 0, FirstTokenAllowList = new[] { 4, 7 }, RepetitionPenalty = 1 };
        foreach (bool vector in new[] { false, true })
        {
            var sampler = new TokenSampler(config, vector);
            Assert.Equal(7, sampler.Sample(logits));
            Assert.Equal(100, sampler.Sample(logits, new[] { 7 }));
        }
    }
}
