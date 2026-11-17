// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Models.Media;
using TensorSharp.Models.Media.Desktop;
using TensorSharp.Models.Video;
using Xunit;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

/// <summary>
/// LFM3-Omni's audio front end (Lfm3AudioPreprocessor): the FFT against a direct DFT, the filterbank against an
/// independent construction, the padding and row counts, the features against llama.cpp's mtmd (fixture: its encoder input
/// for eight clips), streaming, and file decoding (WAV, and through ffmpeg FLAC, M4A and an MP4's soundtrack).
/// </summary>
public sealed class Lfm3AudioFrontEndTests
{
    private readonly ITestOutputHelper _output;

    public Lfm3AudioFrontEndTests(ITestOutputHelper output) => _output = output;

    internal static string RepoRoot()
    {
        var dir = new DirectoryInfo(AppContext.BaseDirectory);
        while (dir != null && !File.Exists(Path.Combine(dir.FullName, "TensorSharp.slnx"))) dir = dir.Parent;
        Assert.True(dir != null, "repository root (TensorSharp.slnx) not found above the test binaries");
        return dir!.FullName;
    }

    internal static JsonElement Fixture()
    {
        string path = Path.Combine(RepoRoot(), "InferenceWeb.Tests", "Fixtures", "Lfm3Audio", "llamacpp-reference.json");
        using var doc = JsonDocument.Parse(File.ReadAllText(path));
        return doc.RootElement.GetProperty("clips").Clone();
    }

    internal static float[] Floats(JsonElement array) => array.EnumerateArray().Select(v => v.GetSingle()).ToArray();

    [Fact]
    public void FftPowerSpectrumMatchesADirectDft()
    {
        var rng = new Random(5);
        var frames = new List<double[]>
        {
            Enumerable.Range(0, 400).Select(_ => rng.NextDouble() * 2 - 1).ToArray(),
            Enumerable.Range(0, 400).Select(i => i == 0 ? 1.0 : 0.0).ToArray(),                     // impulse: flat spectrum
            Enumerable.Repeat(0.25, 400).ToArray(),                                                // DC only
            Enumerable.Range(0, 400).Select(i => Math.Cos(2 * Math.PI * 37 * i / 400)).ToArray(),  // on bin 37
            Enumerable.Range(0, 400).Select(i => i % 2 == 0 ? 1.0 : -1.0).ToArray(),               // Nyquist
        };
        foreach (double[] x in frames)
        {
            double[] fast = Lfm3AudioPreprocessor.PowerSpectrumOf(x);
            double scale = 0;
            for (int k = 0; k <= 200; k++)
            {
                double re = 0, im = 0;
                for (int j = 0; j < 400; j++)
                {
                    re += x[j] * Math.Cos(2 * Math.PI * k * j / 400);
                    im -= x[j] * Math.Sin(2 * Math.PI * k * j / 400);
                }
                double exact = re * re + im * im;
                scale = Math.Max(scale, exact);
                Assert.True(Math.Abs(fast[k] - exact) <= 1e-9 * Math.Max(1, scale), $"bin {k}: FFT {fast[k]}, DFT {exact}");
            }
        }
    }

    /// <summary>The filterbank against HF's construction (transformers audio_utils.mel_filter_bank, slaney scale and norm):
    /// triangles from the slopes between neighbouring mel points, clipped at zero, times 2 / (f_right - f_left).</summary>
    [Fact]
    public void FilterbankMatchesHuggingFacesConstruction()
    {
        static double HzToMel(double f) => f < 1000 ? 3 * f / 200 : 15 + Math.Log(f / 1000) * 27 / Math.Log(6.4);
        static double MelToHz(double m) => m < 15 ? 200 * m / 3 : 1000 * Math.Exp(Math.Log(6.4) / 27 * (m - 15));
        double maxMel = HzToMel(8000);
        double[] points = Enumerable.Range(0, 130).Select(i => MelToHz(maxMel * i / 129)).ToArray();
        double worst = 0;
        for (int m = 0; m < 128; m++)
        {
            for (int k = 0; k <= 200; k++)
            {
                double f = 40.0 * k;
                double down = (f - points[m]) / (points[m + 1] - points[m]);
                double up = (points[m + 2] - f) / (points[m + 2] - points[m + 1]);
                double expected = Math.Max(0, Math.Min(down, up)) * 2 / (points[m + 2] - points[m]);
                worst = Math.Max(worst, Math.Abs(Lfm3AudioPreprocessor.FilterWeight(m, k) - expected));
            }
        }
        _output.WriteLine($"filterbank vs HF construction: max |difference| {worst:G3}");
        Assert.True(worst < 1e-7, $"filterbank differs by {worst}");
    }

    [Theory]
    [InlineData(0L, 0, 0)]
    [InlineData(1L, 1, 1)]
    [InlineData(400L, 1, 1)]
    [InlineData(401L, 2, 1)]
    [InlineData(560L, 2, 1)]
    [InlineData(561L, 3, 1)]
    [InlineData(16000L, 99, 13)]
    [InlineData(160000L, 999, 125)]
    [InlineData(480000L, 2999, 375)]
    [InlineData(12345L, 76, 10)]
    public void PaddingFramesAndRows(long samples, int frames, int rows)
    {
        if (samples > 0) Assert.Equal(frames, Lfm3AudioPreprocessor.FrameCount(samples));
        Assert.Equal(rows, Lfm3AudioPreprocessor.RowCountForSamples(samples));
        Assert.Equal(frames, Lfm3AudioPreprocessor.ComputeFeatures(new float[samples], out int computed).Length / 128);
        Assert.Equal(frames, computed);
    }

    /// <summary>The features of every clip against llama.cpp's encoder input (three whole frames and each mel band's mean
    /// over the clip). llama.cpp computes the DFT as an F32 matrix product (window folded in, sin/cos tables in float); this
    /// FFT runs in double. The two agree to rounding except in bins whose power is ~1e-7 of the frame's (the leakage floor
    /// of a pure tone), where the F32 product's rounding shows after the log.</summary>
    [Theory]
    [MemberData(nameof(Lfm3AudioSignals.Cases), MemberType = typeof(Lfm3AudioSignals))]
    public void FeaturesMatchLlamaCpp(string clip)
    {
        JsonElement reference = Fixture().GetProperty(clip);
        float[] pcm = Lfm3AudioSignals.Make(clip, RepoRoot());
        Assert.Equal(reference.GetProperty("sha256").GetString(), Lfm3AudioSignals.Hash(pcm));
        float[] features = Lfm3AudioPreprocessor.ComputeFeatures(pcm, out int frames);
        Assert.Equal(reference.GetProperty("frames").GetInt32(), frames);
        Assert.Equal(reference.GetProperty("rows").GetInt32(), Lfm3AudioPreprocessor.RowCount(frames));

        int[] index = reference.GetProperty("mel_frame_index").EnumerateArray().Select(v => v.GetInt32()).ToArray();
        JsonElement[] expectedFrames = reference.GetProperty("mel_frames").EnumerateArray().ToArray();
        double worst = 0, sumSq = 0;
        int n = 0;
        for (int f = 0; f < index.Length; f++)
        {
            float[] expected = Floats(expectedFrames[f]);
            for (int m = 0; m < 128; m++)
            {
                double d = features[index[f] * 128 + m] - expected[m];
                worst = Math.Max(worst, Math.Abs(d));
                sumSq += d * d;
                n++;
            }
        }
        float[] bandMean = Floats(reference.GetProperty("mel_band_mean"));
        double worstBand = 0;
        for (int m = 0; m < 128; m++)
        {
            double s = 0;
            for (int t = 0; t < frames; t++) s += features[t * 128 + m];
            worstBand = Math.Max(worstBand, Math.Abs(s / frames - bandMean[m]));
        }
        string line = $"{clip}: {frames} frames; sampled frames max |diff| {worst:G3}, rms {Math.Sqrt(sumSq / n):G3}; band means max |diff| {worstBand:G3}";
        _output.WriteLine(line);
        Assert.True(worst <= 2e-3 && Math.Sqrt(sumSq / n) <= 2e-4 && worstBand <= 1e-3, line);
    }

    [Fact]
    public void SilenceIsTheLogFloor()
    {
        float[] features = Lfm3AudioPreprocessor.ComputeFeatures(new float[48000], out int frames);
        Assert.Equal(299, frames);
        Assert.All(features, v => Assert.Equal(-2f, v));   // (log10(1e-12) + 4) / 4
    }

    /// <summary>A stream's frames (pushes of any size, a finish that pads the tail) equal the whole clip's, bit for bit.</summary>
    [Theory]
    [InlineData("speech_6s", 1)]
    [InlineData("speech_6s", 2)]
    [InlineData("noise_odd", 3)]
    [InlineData("tiny", 4)]
    [InlineData("chirp_1s", 5)]
    public void StreamedFeaturesEqualTheWholeClips(string clip, int seed)
    {
        float[] pcm = Lfm3AudioSignals.Make(clip, RepoRoot());
        float[] whole = Lfm3AudioPreprocessor.ComputeFeatures(pcm, out int frames);
        var rng = new Random(seed);
        var stream = new Lfm3AudioFeatureStream();
        var streamed = new List<float>();
        int at = 0;
        while (at < pcm.Length)
        {
            int n = Math.Min(pcm.Length - at, rng.Next(1, 3000));
            stream.Push(pcm.AsSpan(at, n));
            at += n;
            float[] part = stream.Read(out int got);
            Assert.Equal(got * 128, part.Length);
            streamed.AddRange(part);
        }
        streamed.AddRange(stream.Finish(out _));
        Assert.Equal(frames, stream.FramesEmitted);
        Assert.Equal(whole, streamed.ToArray());
        Assert.Throws<InvalidOperationException>(() => stream.Push(new float[1]));
    }

    [Fact]
    public void LoadAudioFoldsChannelsAndResamplesTo16k()
    {
        string dir = Path.Combine(Path.GetTempPath(), "ts-lfm3-audio-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        try
        {
            // 48 kHz stereo, a 440 Hz tone on the left and its negation on the right: the mean is silence.
            const int Rate = 48000, N = 48000;
            var left = new float[N];
            for (int i = 0; i < N; i++) left[i] = (float)(0.5 * Math.Sin(2 * Math.PI * 440 * i / Rate));
            string wav = Path.Combine(dir, "stereo.wav");
            File.WriteAllBytes(wav, WavF32(Rate, left, left.Select(v => -v).ToArray()));
            float[] mono = Lfm3AudioPreprocessor.LoadAudio(wav);
            Assert.Equal(16000, mono.Length);
            Assert.All(mono, v => Assert.Equal(0f, v));
            // One channel: the band-limited resampler keeps the tone's amplitude.
            File.WriteAllBytes(wav, WavF32(Rate, left));
            mono = Lfm3AudioPreprocessor.LoadAudio(wav);
            Assert.Equal(16000, mono.Length);
            Assert.InRange(mono.Skip(1000).Take(14000).Max(), 0.49f, 0.51f);
            // 16 kHz passes through unchanged.
            float[] speech = Lfm3AudioSignals.Speech(RepoRoot());
            Assert.Equal(speech, Lfm3AudioPreprocessor.LoadAudio(Path.Combine(RepoRoot(), "InferenceWeb.Tests", "Fixtures", "JevAttachments", "ticket.wav")));
        }
        finally
        {
            Directory.Delete(dir, recursive: true);
        }
    }

    /// <summary>ffmpeg's streamed WAV: sizes it could not seek back to fill in (0 or 0xFFFFFFFF) mean "to the end".</summary>
    [Fact]
    public void StreamedWavWithUnknownSizesParses()
    {
        byte[] wav = WavF32(22050, new[] { 0.5f, -0.25f, 1f }, new[] { 0.125f, 0f, -1f });
        foreach (uint unknown in new[] { 0u, uint.MaxValue })
        {
            byte[] copy = (byte[])wav.Clone();
            BitConverter.GetBytes(unknown).CopyTo(copy, 4);
            BitConverter.GetBytes(unknown).CopyTo(copy, copy.Length - 3 * 8 - 4);
            DecodedAudio audio = DesktopMediaProvider.ParseStreamedWav(copy, "test");
            Assert.Equal(22050, audio.SampleRate);
            Assert.Equal(new[] { 0.5f, -0.25f, 1f }, audio.Channels[0]);
            Assert.Equal(new[] { 0.125f, 0f, -1f }, audio.Channels[1]);
        }
        Assert.Throws<InvalidDataException>(() => DesktopMediaProvider.ParseStreamedWav(new byte[] { 1, 2, 3 }, "test"));
    }

    /// <summary>
    /// The desktop provider's ffmpeg decoder: a FLAC of ticket.wav decodes to its exact samples, an M4A (AAC) to nearly them,
    /// and an MP4 video's soundtrack (what LFM3-Omni hears beside the frames) to the same audio, through
    /// <see cref="Lfm3AudioPreprocessor.LoadAudio"/>.
    /// </summary>
    [FfmpegFact]
    public void FfmpegDecodesFlacM4aAndAVideosSoundtrack()
    {
        string ffmpeg = MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG");
        string dir = Path.Combine(Path.GetTempPath(), "ts-lfm3-audio-" + Guid.NewGuid().ToString("N"));
        Directory.CreateDirectory(dir);
        IAudioDecoder original = MediaCodecs.Audio;
        try
        {
            MediaCodecs.Audio = DesktopMediaProvider.Instance;
            string ticket = Path.Combine(RepoRoot(), "InferenceWeb.Tests", "Fixtures", "JevAttachments", "ticket.wav");
            float[] speech = Lfm3AudioSignals.Speech(RepoRoot());
            string flac = Path.Combine(dir, "ticket.flac"), m4a = Path.Combine(dir, "ticket.m4a"), mp4 = Path.Combine(dir, "clip.mp4");
            Run(ffmpeg, "-i", ticket, "-c:a", "flac", flac);
            Run(ffmpeg, "-i", ticket, "-c:a", "aac", "-b:a", "128k", m4a);
            Run(ffmpeg, "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=4:d=6", "-i", ticket, "-map", "0:v", "-map", "1:a",
                "-c:v", "mpeg4", "-c:a", "aac", "-b:a", "128k", "-shortest", mp4);

            Assert.Equal(speech, Lfm3AudioPreprocessor.LoadAudio(flac));
            foreach (string lossy in new[] { m4a, mp4 })
            {
                float[] decoded = Lfm3AudioPreprocessor.LoadAudio(lossy);
                // AAC adds priming delay and coding noise: align by the best lag, then compare the waveforms.
                (int lag, double corr) = BestLag(speech, decoded, 4096);
                _output.WriteLine($"{Path.GetFileName(lossy)}: {decoded.Length} samples ({speech.Length} in ticket.wav), lag {lag}, correlation {corr:F4}");
                Assert.True(corr > 0.98, $"{Path.GetFileName(lossy)} correlates {corr} with ticket.wav");
                Assert.InRange(decoded.Length, speech.Length - 2048, speech.Length + 4096);
            }
            string silentVideo = Path.Combine(dir, "silent.mp4");
            Run(ffmpeg, "-f", "lavfi", "-i", "color=c=gray:s=64x64:r=4:d=1", "-c:v", "mpeg4", silentVideo);
            var ex = Assert.Throws<InvalidDataException>(() => Lfm3AudioPreprocessor.LoadAudio(silentVideo));
            Assert.Contains("no audio stream", ex.Message);
        }
        finally
        {
            MediaCodecs.Audio = original;
            Directory.Delete(dir, recursive: true);
        }
    }

    private static (int Lag, double Correlation) BestLag(float[] reference, float[] actual, int maxLag)
    {
        (int, double) best = (0, -1);
        for (int lag = 0; lag <= maxLag; lag += 1)
        {
            double ab = 0, aa = 0, bb = 0;
            for (int i = 0; i + lag < actual.Length && i < reference.Length; i += 4)
            {
                ab += reference[i] * actual[i + lag];
                aa += reference[i] * reference[i];
                bb += actual[i + lag] * actual[i + lag];
            }
            double c = ab / Math.Sqrt(aa * bb + 1e-30);
            if (c > best.Item2) best = (lag, c);
        }
        return best;
    }

    private static void Run(string ffmpeg, params string[] args)
    {
        var start = new ProcessStartInfo(ffmpeg) { UseShellExecute = false, RedirectStandardError = true };
        foreach (string a in new[] { "-nostdin", "-hide_banner", "-loglevel", "error", "-y" }.Concat(args)) start.ArgumentList.Add(a);
        using var p = Process.Start(start)!;
        string err = p.StandardError.ReadToEnd();
        p.WaitForExit();
        Assert.True(p.ExitCode == 0, $"ffmpeg {string.Join(' ', args)}: {err}");
    }

    /// <summary>A float WAV of one plane per channel.</summary>
    internal static byte[] WavF32(int rate, params float[][] channels)
    {
        int n = channels[0].Length, c = channels.Length, data = n * c * 4;
        using var ms = new MemoryStream();
        using var w = new BinaryWriter(ms);
        w.Write("RIFF"u8);
        w.Write(36 + data);
        w.Write("WAVE"u8);
        w.Write("fmt "u8);
        w.Write(16);
        w.Write((short)3);
        w.Write((short)c);
        w.Write(rate);
        w.Write(rate * c * 4);
        w.Write((short)(c * 4));
        w.Write((short)32);
        w.Write("data"u8);
        w.Write(data);
        for (int i = 0; i < n; i++)
            for (int ch = 0; ch < c; ch++) w.Write(channels[ch][i]);
        w.Flush();
        return ms.ToArray();
    }
}

/// <summary>[Fact] that needs ffmpeg (PATH or TS_FFMPEG): skips visibly without it.</summary>
public sealed class FfmpegFactAttribute : FactAttribute
{
    public FfmpegFactAttribute()
    {
        if (MediaExecutableLocator.Find("ffmpeg", "TS_FFMPEG") is not { } path || !File.Exists(path))
            Skip = "Requires ffmpeg on PATH or TS_FFMPEG (the desktop audio decoder for FLAC, M4A and video soundtracks).";
    }
}
