// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The diffusion runtime's noise stream against stored CPU torch.randn outputs: successive calls on one generator
// (Fixtures/MiniMaxH3Image/torch-randn-stream.json, written by eng/torch-randn-stream-fixture.py with torch 2.14.0 on
// macOS arm64). MiniMaxH3ImageSamplingTests keeps the single-draw checks against torch 2.14.1's x86 AVX2 build.
using System.Globalization;
using System.Text.Json;
using TensorSharp.Models.SharedDiffusion;

namespace InferenceWeb.Tests;

public sealed class TorchCpuRandnStreamTests
{
    private static readonly JsonElement[] Cases = Load();
    public static IEnumerable<object[]> CaseIndices => Cases.Select((_, i) => new object[] { i });

    // libm (this build) or SLEEF (x86 AVX2) log/sin/cos against .NET's: a few ULP, as MiniMaxH3ImageSamplingTests allows.
    private const double ValueTolerance = 2e-6;

    [Theory]
    [MemberData(nameof(CaseIndices))]
    public void SuccessiveCallsContinueOneTorchGenerator(int index)
    {
        JsonElement reference = Cases[index];
        ulong seed = ulong.Parse(reference.GetProperty("seed").GetString()!, CultureInfo.InvariantCulture);
        var stream = new TorchCpuRandn(seed);
        int call = 0;
        foreach (JsonElement expected in reference.GetProperty("calls").EnumerateArray())
        {
            long count = expected.GetProperty("count").GetInt64();
            float[] actual = stream.Next(count);
            Assert.Equal(count, actual.LongLength);
            string where = $"seed {seed}, call {call}";
            if (expected.TryGetProperty("values", out JsonElement values))
            {
                Near(values, actual, 0, where);
            }
            else
            {
                JsonElement first = expected.GetProperty("first"), last = expected.GetProperty("last");
                Near(first, actual, 0, where + " (first)");
                Near(last, actual, actual.Length - last.GetArrayLength(), where + " (last)");
                // Over hundreds of thousands of values the last-bit differences add up to well under these.
                Assert.InRange(actual.Sum(v => (double)v), expected.GetProperty("sum").GetDouble() - 0.05, expected.GetProperty("sum").GetDouble() + 0.05);
                Assert.InRange(actual.Sum(v => Math.Abs((double)v)), expected.GetProperty("absSum").GetDouble() - 0.5, expected.GetProperty("absSum").GetDouble() + 0.5);
            }
            call++;
        }
    }

    [Fact]
    public void ManualSeedKeepsTheLowThirtyTwoBits()
    {
        var a = new TorchCpuRandn(5);
        var b = new TorchCpuRandn((1UL << 32) + 5);
        Assert.Equal(a.Next(100), b.Next(100));
        Assert.Equal(a.Next(200), b.Next(200));
    }

    [Theory]
    [InlineData(16L)]
    [InlineData(360L)]
    [InlineData(24L * 128 * 128)]
    public void OneDrawFromAFreshStreamIsTheStaticDraw(long count)
    {
        Assert.Equal(TorchCpuRandn.Normal(count, 7001), new TorchCpuRandn(7001).Next(count));
    }

    [Fact]
    public void ACallDrawsItsCountAndRedrawsTheLastBlockOfSixteen()
    {
        var stream = new TorchCpuRandn(42);
        stream.Next(16);
        Assert.Equal(16, stream.WordsDrawn);
        stream.Next(360);           // 360 % 16 = 8: the last block of 16 is drawn again
        Assert.Equal(16 + 360 + 16, stream.WordsDrawn);
        stream.Next(384);
        Assert.Equal(16 + 376 + 384, stream.WordsDrawn);
    }

    [Fact]
    public void CountsBelowSixteenAreRefused()
    {
        Assert.Throws<ArgumentOutOfRangeException>(() => new TorchCpuRandn(1).Next(15));
        Assert.Throws<ArgumentOutOfRangeException>(() => TorchCpuRandn.Normal(0, 1));
    }

    private static void Near(JsonElement expected, float[] actual, int offset, string where)
    {
        int n = expected.GetArrayLength();
        for (int i = 0; i < n; i++)
        {
            double error = Math.Abs(actual[offset + i] - expected[i].GetDouble());
            Assert.True(error <= ValueTolerance, $"{where}: value {offset + i} is {actual[offset + i]:R}, torch {expected[i].GetDouble():R}");
        }
    }

    private static JsonElement[] Load()
    {
        using Stream stream = typeof(TorchCpuRandnStreamTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.MiniMaxH3Image.torch-randn-stream.json")!;
        using JsonDocument document = JsonDocument.Parse(stream);
        Assert.StartsWith("torch 2.14.0", document.RootElement.GetProperty("generator").GetString());
        return document.RootElement.GetProperty("cases").EnumerateArray().Select(value => value.Clone()).ToArray();
    }
}
