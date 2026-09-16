using System.Text;
using System.Text.Json;

namespace InferenceWeb.Tests;

public sealed class NemotronAudioInputTests
{
    [Theory]
    [InlineData(0)]
    [InlineData(1)]
    [InlineData(2)]
    [InlineData(3)]
    [InlineData(4)]
    [InlineData(5)]
    public void OfficialParakeetFrontend_AllMelValuesMatch(int caseIndex)
    {
        using var stream = typeof(NemotronAudioInputTests).Assembly.GetManifestResourceStream(
            "InferenceWeb.Tests.Fixtures.NemotronAudio.frontend_reference.json")!;
        using var json = JsonDocument.Parse(stream);
        var example = json.RootElement.GetProperty("cases")[caseIndex];
        float[] samples = example.GetProperty("samples").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        float[] expected = example.GetProperty("expected").EnumerateArray().Select(v => v.GetSingle()).ToArray();
        var result = NemotronAudioPreprocessor.ComputeParakeetMelSpectrogram(samples);
        Assert.Equal(example.GetProperty("frames").GetInt32(), result.frames);
        Assert.Equal(example.GetProperty("valid_frames").GetInt32(), result.validFrames);
        Assert.Equal(expected.Length, result.mel.Length);
        for (int i = 0; i < expected.Length; i++)
        {
            double delta = Math.Abs(result.mel[i] - expected[i]);
            // Independent torch F32 STFT/librosa Slaney-filter reference;
            // allow FFT rounding while rejecting window/filter/normalization drift.
            double tolerance = 1e-3 + 1e-4 * Math.Abs(expected[i]);
            Assert.True(double.IsFinite(delta) && delta <= tolerance,
                $"case={caseIndex} frame={i / 128} mel={i % 128} expected={expected[i]:R} actual={result.mel[i]:R} delta={delta:R} tolerance={tolerance:R}");
        }
    }

    [Theory]
    [InlineData(1, 1)]
    [InlineData(159, 1)]
    [InlineData(160, 1)]
    [InlineData(1279, 1)]
    [InlineData(1280, 2)]
    [InlineData(1281, 2)]
    [InlineData(2559, 2)]
    [InlineData(2560, 3)]
    public void CenterStftExtraFrame_IsIncludedInAudioTokenCount(int samples, int tokens)
    {
        float[] waveform = Enumerable.Range(0, samples).Select(i => MathF.Sin(i * .13f) * .2f).ToArray();
        var first = NemotronAudioPreprocessor.ComputeParakeetMelSpectrogram(waveform);
        var second = NemotronAudioPreprocessor.ComputeParakeetMelSpectrogram(waveform);
        Assert.Equal(1 + samples / 160, first.frames);
        Assert.Equal(Math.Max(1, samples / 160), first.validFrames);
        Assert.Equal(tokens, NemotronAudioEncoder.OutputLength(first.frames));
        Assert.Equal(first.mel, second.mel);
        Assert.All(first.mel, value => Assert.True(float.IsFinite(value)));
        Assert.All(first.mel.Skip(first.validFrames * 128), value => Assert.Equal(0, value));
    }

    [Fact]
    public void MediaPlan_PreservesAllClipAndImagePositionsAcrossTurns()
    {
        var history = new List<ChatMessage>
        {
            new() { Role = "user", ImagePaths = new() { "image1" }, AudioPaths = new() { "audio1", "audio2" } },
            new() { Role = "assistant", Content = "response" },
            new() { Role = "user", AudioPaths = new() { "audio3" } },
        };
        var tokens = new List<int> { 1, 27, 2, 18, 3, 27, 4, 27 };
        var plan = ModelMultimodalInjector.PlanNemotronMedia(history, tokens, 18, 27);
        Assert.Equal(new[] { (1, true, "audio1"), (3, false, "image1"), (5, true, "audio2"), (7, true, "audio3") }, plan);
        Assert.Equal(new[] { 1, 27, 2, 18, 3, 27, 4, 27 }, tokens);
        Assert.Throws<InvalidOperationException>(() => ModelMultimodalInjector.PlanNemotronMedia(history, new List<int> { 18, 27 }, 18, 27));
        Assert.Throws<InvalidOperationException>(() => ModelMultimodalInjector.PlanNemotronMedia(history, tokens, 18, -1));
    }

    [Fact]
    public void WavDecoder_PreservesInterleavedStereoOrderAndRejectsMalformedSamples()
    {
        byte[] wav = Wav(new[] { .8f, -.2f, -.6f, .4f, .2f, .6f });
        float[] mono = NemotronAudioPreprocessor.DecodeWAV(wav);
        Assert.Equal(3, mono.Length);
        Assert.InRange(Math.Abs(mono[0] - .3f), 0, 1e-7f);
        Assert.InRange(Math.Abs(mono[1] + .1f), 0, 1e-7f);
        Assert.InRange(Math.Abs(mono[2] - .4f), 0, 1e-7f);
        Assert.Throws<InvalidDataException>(() => NemotronAudioPreprocessor.DecodeWAV(wav[..^1]));
        Assert.Throws<InvalidDataException>(() => NemotronAudioPreprocessor.DecodeWAV(Wav(new[] { float.NaN, 0f })));
        Assert.Throws<InvalidDataException>(() => NemotronAudioPreprocessor.DecodeWAV(Wav(new[] { 1f })));
        Assert.Throws<InvalidDataException>(() => NemotronAudioPreprocessor.DecodeWAV(Wav(Array.Empty<float>())));
        byte[] invalidSize = (byte[])wav.Clone(); Array.Fill(invalidSize, (byte)255, 40, 4);
        Assert.Throws<InvalidDataException>(() => NemotronAudioPreprocessor.DecodeWAV(invalidSize));
        byte[] invalidRate = (byte[])wav.Clone(); Array.Clear(invalidRate, 24, 4);
        Assert.Throws<InvalidDataException>(() => NemotronAudioPreprocessor.DecodeWAV(invalidRate));
        byte[] invalidBits = (byte[])wav.Clone(); invalidBits[34] = 64;
        Assert.Throws<InvalidDataException>(() => NemotronAudioPreprocessor.DecodeWAV(invalidBits));
        Assert.Throws<ArgumentException>(() => NemotronAudioPreprocessor.ComputeParakeetMelSpectrogram(new[] { float.PositiveInfinity }));
    }

    private static byte[] Wav(float[] samples)
    {
        using var stream = new MemoryStream(); using var writer = new BinaryWriter(stream, Encoding.ASCII, true);
        writer.Write(Encoding.ASCII.GetBytes("RIFF")); writer.Write(36 + samples.Length * 4); writer.Write(Encoding.ASCII.GetBytes("WAVEfmt "));
        writer.Write(16); writer.Write((ushort)3); writer.Write((ushort)2); writer.Write(16000); writer.Write(16000 * 8);
        writer.Write((ushort)8); writer.Write((ushort)32); writer.Write(Encoding.ASCII.GetBytes("data")); writer.Write(samples.Length * 4);
        foreach (float sample in samples) writer.Write(sample); writer.Flush(); return stream.ToArray();
    }
}
