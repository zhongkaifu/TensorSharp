// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using TensorSharp.Graph;
using TensorSharp.Models.Graph;
using TensorSharp.Models.Lfm3;
using TensorSharp.Runtime;
using Xunit;
using Xunit.Abstractions;
using static InferenceWeb.Tests.NetworkGraphTests;

namespace InferenceWeb.Tests;

/// <summary>
/// LFM3-Omni speech output (TensorSharp.Models/Models/Lfm3/Lfm3Speech*.cs) without model files: small synthetic depthformer and
/// codec networks on every executor this machine runs (managed CPU, the tensor executor's reference decompositions = what
/// direct CUDA runs, GGML cpu and metal, MLX) against an independent double-precision host forward; the host pieces (the
/// random stream against llama.cpp's libc++ draws, the ISTFT against a direct inverse DFT, the PCM16 / WAV bytes, the slot
/// schedule, the TTS prompt) against llama.cpp's behaviour. Lfm3SpeechRealWeightsTests runs the real mmproj against
/// llama.cpp's outputs.
/// </summary>
public sealed class Lfm3SpeechOutputTests
{
    private const int Cb = Lfm3SpeechConfig.Codebooks, V = Lfm3SpeechConfig.CodebookSize;
    private readonly ITestOutputHelper _output;

    public Lfm3SpeechOutputTests(ITestOutputHelper output) => _output = output;

    internal static JsonElement Fixture(string name)
    {
        using Stream stream = typeof(Lfm3SpeechOutputTests).Assembly.GetManifestResourceStream($"InferenceWeb.Tests.Fixtures.Lfm3Speech.{name}")
            ?? throw new FileNotFoundException(name);
        return JsonDocument.Parse(stream).RootElement.Clone();
    }

    // ================================================================ host pieces

    [Fact]
    public void RandomStreamIsMt19937()
    {
        // The C++ standard's check: the 10000th output of a default-constructed std::mt19937 (seed 5489).
        var rng = new Lfm3SpeechRandom(5489);
        uint v = 0;
        for (int i = 0; i < 10000; i++) v = rng.NextUInt32();
        Assert.Equal(4123659995u, v);
    }

    /// <summary>The codebook orders and uniform draws llama.cpp's sampler takes from mt19937(seed) (std::shuffle then
    /// uniform_real_distribution, libc++), bit for bit.</summary>
    [Theory]
    [InlineData("llamacpp-reference-q8_0.json")]
    [InlineData("llamacpp-reference-f16.json")]
    public void RandomStreamMatchesLlamaCppDraws(string fixture)
    {
        JsonElement rng = Fixture(fixture).GetProperty("rng");
        var r = new Lfm3SpeechRandom(rng.GetProperty("seed").GetUInt32());
        foreach (JsonElement frame in rng.GetProperty("frames").EnumerateArray())
        {
            int[] order = Enumerable.Range(0, Cb).ToArray();
            r.Shuffle(order);
            Assert.Equal(frame.GetProperty("order").EnumerateArray().Select(e => e.GetInt32()), order);
            foreach (JsonElement u in frame.GetProperty("uniform").EnumerateArray())
                Assert.Equal(u.GetDouble(), r.NextDouble());
        }
    }

    [Fact]
    public void UniformIntRejectsOutOfRangeDraws()
    {
        // libc++: a range of 12 takes 4 low bits of each output and rejects 12..15.
        var a = new Lfm3SpeechRandom(99);
        var b = new Lfm3SpeechRandom(99);
        for (int i = 0; i < 200; i++)
        {
            long got = a.UniformInt(0, 11);
            uint u;
            do u = b.NextUInt32() & 15u; while (u >= 12);
            Assert.Equal(u, (ulong)got);
        }
    }

    /// <summary>The streaming ISTFT against a direct, non-streaming one in double precision (irfft by its definition,
    /// periodic Hann, overlap-add, the windows' squared sum, (n_fft - hop) / 2 trimmed at both ends), frame by frame and
    /// whole.</summary>
    [Fact]
    public void StreamingIstftMatchesDirectInverse()
    {
        const int N = 1280, Hop = 320, Bins = N / 2 + 1, Frames = 14;
        var rng = new Random(5);
        var logMag = new float[Frames][];
        var phase = new float[Frames][];
        for (int f = 0; f < Frames; f++)
        {
            logMag[f] = Rand(rng, Bins, -4f, 1f);
            phase[f] = Rand(rng, Bins, -3.1f, 3.1f);
        }
        var istft = new Lfm3StreamingIstft(N, Hop);
        var streamed = new List<float>();
        var counts = new List<int>();
        for (int f = 0; f < Frames; f++) counts.Add(istft.ProcessFrame(logMag[f], phase[f], streamed));
        int tail = istft.Flush(streamed);
        Assert.Equal(0, counts[0]);                           // the first 480 samples are padding: 320 of frame 0, 160 of frame 1
        Assert.Equal(Hop - 160, counts[1]);
        Assert.All(counts.Skip(2), c => Assert.Equal(Hop, c));
        Assert.Equal(480, tail);
        Assert.Equal(Frames * Hop, streamed.Count);

        // Direct: irfft of each frame, windowed, overlap-added at hop, divided by the summed squared windows.
        int length = N + (Frames - 1) * Hop;
        var y = new double[length];
        var env = new double[length];
        for (int f = 0; f < Frames; f++)
        {
            for (int n = 0; n < N; n++)
            {
                double s = 0;
                for (int k = 0; k < Bins; k++)
                {
                    double mag = Math.Exp(logMag[f][k]);
                    double re = mag * Math.Cos(phase[f][k]), im = mag * Math.Sin(phase[f][k]);
                    double w = 2 * Math.PI * k * n / N;
                    double term = re * Math.Cos(w) - im * Math.Sin(w);
                    s += k == 0 || k == Bins - 1 ? re * Math.Cos(w) : 2 * term;
                }
                double win = 0.5 * (1 - Math.Cos(2 * Math.PI * n / N));
                y[f * Hop + n] += s / N * win;
                env[f * Hop + n] += win * win;
            }
        }
        int trim = (N - Hop) / 2;
        var expected = new float[Frames * Hop];
        for (int i = 0; i < expected.Length; i++)
        {
            int at = i + trim;
            expected[i] = (float)(env[at] > 1e-8 ? y[at] / env[at] : y[at]);
        }
        AssertClose(expected, streamed.ToArray(), 2e-5, "istft");
    }

    [Fact]
    public void Pcm16AndWavMatchLlamaCppEncoding()
    {
        float[] samples = { 0f, 1f, -1f, 1.5f, -2f, 0.5f, -0.5f, 1f / 32767 * 0.5f, -1f / 32767 * 0.5f, 0.25f };
        byte[] pcm = Lfm3SpeechAudio.Pcm16(samples);
        short[] got = Enumerable.Range(0, samples.Length).Select(i => BitConverter.ToInt16(pcm, 2 * i)).ToArray();
        // lround(clamp(x, -1, 1) * 32767): halves away from zero.
        Assert.Equal(new short[] { 0, 32767, -32767, 32767, -32767, 16384, -16384, 1, -1, 8192 }, got);

        byte[] wav = Lfm3SpeechAudio.Wav(samples, 24000);
        Assert.Equal(Lfm3SpeechAudio.WavHeader((uint)pcm.Length, 24000), wav[..44]);
        Assert.Equal(pcm, wav[44..]);
        Assert.Equal(36u + (uint)pcm.Length, BitConverter.ToUInt32(wav, 4));
        Assert.Equal(48000u, BitConverter.ToUInt32(wav, 28));

        byte[] stream = Lfm3SpeechAudio.StreamingWavHeader(24000);
        Assert.Equal(uint.MaxValue, BitConverter.ToUInt32(stream, 4));
        Assert.Equal(uint.MaxValue, BitConverter.ToUInt32(stream, 40));
        Assert.Equal(wav[8..40], stream[8..40]);
    }

    [Theory]
    [InlineData("llamacpp-reference-q8_0.json")]
    [InlineData("llamacpp-reference-f16.json")]
    public void TtsPromptIsTheTemplatesRendering(string fixture)
    {
        JsonElement f = Fixture(fixture);
        Assert.Equal(f.GetProperty("prompt").GetString(),
            Lfm3SpeechPrompt.Tts(f.GetProperty("text").GetString(), f.GetProperty("lang").GetString(), f.GetProperty("voice").GetString()));
    }

    [Fact]
    public void PromptVoicesAndLanguages()
    {
        Assert.Contains("Perform TTS in British English using a male voice.", Lfm3SpeechPrompt.Tts("x", "EN-GB", "Male"));
        Assert.Contains("Perform TTS in American English using a female voice.", Lfm3SpeechPrompt.Tts("x", null, "robot"));
        Assert.Contains("Perform TTS in Klingon using a female voice.", Lfm3SpeechPrompt.Tts("x", "Klingon", null));
        Assert.EndsWith("<|im_start|>assistant\n<|audio_start|>", Lfm3SpeechPrompt.Tts("x", think: false));
        Assert.Equal("Speak in English using a male voice.", Lfm3SpeechPrompt.SpeechSystemLine("en", "MALE"));
        Assert.Equal("Speak in Japanese.", Lfm3SpeechPrompt.SpeechSystemLine("ja", "robot"));
        Assert.Equal("Speak using a female voice.", Lfm3SpeechPrompt.SpeechSystemLine(null, "female"));
        Assert.Equal("", Lfm3SpeechPrompt.SpeechSystemLine("", "robot"));
        Assert.True(Lfm3SpeechPrompt.IsVoice("Female"));
        Assert.False(Lfm3SpeechPrompt.IsVoice("alloy"));
    }

    private static readonly Lfm3SpeechTokens s_tokens = new()
    {
        AudioStart = 125012, MixedStart = 125015, TextEnd = 125014, ImEnd = 124900, Placeholders = new[] { 125031, 125032 },
    };

    [Fact]
    public void TtsTurnSpeaksUntilTheEndOfAudioFrame()
    {
        var turn = new Lfm3SpeechTurn(s_tokens, s_tokens.AudioStart, continuing: false);
        Assert.Equal(Lfm3SpeechMode.Audio, turn.Mode);
        for (int i = 0; i < 5; i++)
        {
            Assert.False(turn.NextIsText);
            Lfm3SlotAction a = turn.OnFrame(false);
            Assert.True(a.DecodeInput && a.DecodeWaveform && !a.Stop && !a.FlushCodec);
        }
        Lfm3SlotAction end = turn.OnFrame(true);
        Assert.True(end.Stop && !end.DecodeInput && !end.DecodeWaveform);
        Assert.Throws<ArgumentException>(() => new Lfm3SpeechTurn(s_tokens, 42, continuing: false));
    }

    [Fact]
    public void InterleavedTurnFollowsLlamaCppBlocks()
    {
        // A conversation's response opened with <|mixed_start|>: 6 text slots, 12 frames, 6 text, ... <|text_end|> inside a
        // text block ends the text; frames follow until the end-of-audio frame, which flushes the codec and returns to text.
        var turn = new Lfm3SpeechTurn(s_tokens, s_tokens.MixedStart, continuing: true);
        var slots = new List<char>();
        void Text(int token) { Assert.True(turn.NextIsText); slots.Add('t'); Assert.False(turn.OnText(token).Stop); }
        void Frame(bool eos = false) { Assert.False(turn.NextIsText); slots.Add(eos ? 'E' : 'a'); turn.OnFrame(eos); }
        for (int i = 0; i < 6; i++) Text(1000 + i);
        for (int i = 0; i < 12; i++) Frame();
        for (int i = 0; i < 3; i++) Text(2000 + i);
        Text(s_tokens.TextEnd);                       // ends the text in the 4th text slot of the block
        Assert.True(turn.Speaking);
        for (int i = 0; i < 20; i++) Frame();         // speech until the end-of-audio frame, no block limit any more
        Assert.Equal(Lfm3SpeechMode.Interleaved, turn.Mode);
        Lfm3SlotAction end = turn.OnFrame(true);
        Assert.True(end.FlushCodec && end.DecodeInput && !end.DecodeWaveform && !end.Stop);
        Assert.Equal(Lfm3SpeechMode.Text, turn.Mode);
        Assert.True(turn.NextIsText);
        Assert.True(turn.OnText(s_tokens.ImEnd).Stop);
        Assert.Equal(new string('t', 6) + new string('a', 12) + new string('t', 4) + new string('a', 20), new string(slots.ToArray()));
    }

    [Fact]
    public void TextTurnSwitchesOnTheModelsTokens()
    {
        var turn = new Lfm3SpeechTurn(s_tokens, 7, continuing: true);
        Assert.Equal(Lfm3SpeechMode.Text, turn.Mode);
        Assert.True(turn.OnText(s_tokens.AudioStart).DecodeInput);        // the switching token is decoded
        Assert.Equal(Lfm3SpeechMode.Audio, turn.Mode);
        turn.OnFrame(false);
        turn.OnFrame(true);                                                 // continued: back to text
        Assert.Equal(Lfm3SpeechMode.Text, turn.Mode);
        Assert.True(turn.OnText(s_tokens.TextEnd).Stop);                    // <|text_end|> outside a block ends the response
        Assert.Equal(new[] { s_tokens.AudioStart, s_tokens.MixedStart }, Lfm3SpeechTurn.TextBias(s_tokens).Keys.OrderBy(k => k));
        Assert.Equal(s_tokens.MixedStart, Lfm3SpeechTurn.OpeningToken(s_tokens, Lfm3SpeechMode.Interleaved));
        Assert.Null(Lfm3SpeechTurn.OpeningToken(s_tokens, Lfm3SpeechMode.Text));
    }

    // ================================================================ synthetic networks on every lane

    public static IEnumerable<object[]> Variants => new[] { "f16", "q8_0", "f32" }.Select(v => new object[] { v });

    [Theory]
    [MemberData(nameof(Variants))]
    public void ManagedCpu(string variant) => CheckNetworks(Lane.Managed, variant);

    [Theory]
    [MemberData(nameof(Variants))]
    public void ReferenceDecompositions(string variant) => CheckNetworks(Lane.Reference, variant);

    [GgmlTheory(BackendType.GgmlCpu)]
    [MemberData(nameof(Variants))]
    public void GgmlCpu(string variant) => CheckNetworks(Lane.GgmlCpu, variant);

    [GgmlTheory(BackendType.GgmlMetal)]
    [MemberData(nameof(Variants))]
    public void GgmlMetal(string variant) => CheckNetworks(Lane.GgmlMetal, variant);

    [MlxTheory]
    [MemberData(nameof(Variants))]
    public void Mlx(string variant) => CheckNetworks(Lane.Mlx, variant);

    [GgmlTheory(BackendType.GgmlCuda)]
    [MemberData(nameof(Variants))]
    public void GgmlCuda(string variant) => CheckNetworks(Lane.GgmlCuda, variant);

    [GgmlTheory(BackendType.GgmlVulkan)]
    [MemberData(nameof(Variants))]
    public void GgmlVulkan(string variant) => CheckNetworks(Lane.GgmlVulkan, variant);

    [CudaTheory]
    [MemberData(nameof(Variants))]
    public void DirectCuda(string variant) => CheckNetworks(Lane.Cuda, variant);

    private static readonly bool[] s_codecLayers = { true, true, false, true, false };

    internal static Lfm3SpeechConfig SyntheticConfig()
        => Lfm3SpeechConfig.ForTest(hidden: 64, depthWidth: 64, depthFfn: 96, depthLayers: 2, depthHeads: 2, depthKvHeads: 1,
            codecWidth: 64, codecFfn: 96, codecHeads: 2, codecKvHeads: 1, codecHeadDim: 32, codecConvLayer: s_codecLayers, quantizerDim: 32);

    /// <summary>Product weights in <paramref name="variant"/>'s format (f16, q8_0; f32: dense), norms and kernels F32.</summary>
    internal static void AddSyntheticWeights(NetworkWeights w, Lfm3SpeechConfig c, string variant, Random rng)
    {
        int fmt = variant switch { "f16" => GraphFormat.F16, "q8_0" => GraphFormat.Q8_0, _ => GraphFormat.F32 };
        void Product(string name, int inWidth, int outRows) => MakeProductWeight(w, name, fmt, inWidth, outRows, rng);
        void Norm(string name, int n) => w.AddF32(name, Rand(rng, n, 0.6f, 1.4f), n);
        int W = c.DepthWidth, kv = c.DepthKvHeads * c.DepthHeadDim;
        Product("a.gen.code.proj_in.weight", c.HiddenSize, Cb * W);
        w.AddF32("a.gen.code.proj_in.bias", Rand(rng, Cb * W, -0.1f, 0.1f), Cb * W);
        w.AddF32("a.gen.code.mask_embd", Rand(rng, W), W);
        Product("a.gen.code.embd.weight", W, Cb * V);
        w.AddF32("a.gen.code.head_norm.weight", Rand(rng, W * Cb, 0.6f, 1.4f), W, Cb);
        Norm("a.gen.code.output_norm.weight", W);
        for (int l = 0; l < c.DepthLayers; l++)
        {
            string p = $"a.gen.code.blk.{l}.";
            Norm(p + "ln1.weight", W);
            Norm(p + "ln2.weight", W);
            Product(p + "attn_q.weight", W, W);
            Product(p + "attn_k.weight", W, kv);
            Product(p + "attn_v.weight", W, kv);
            Norm(p + "attn_q_norm.weight", c.DepthHeadDim);
            Norm(p + "attn_k_norm.weight", c.DepthHeadDim);
            Product(p + "attn_gate.weight", W, c.DepthHeads);
            Product(p + "attn_out.weight", W, W);
            Product(p + "ffn_gate.weight", W, c.DepthFeedForward);
            Product(p + "ffn_up.weight", W, c.DepthFeedForward);
            Product(p + "ffn_down.weight", c.DepthFeedForward, W);
        }
        // Codec: the FSQ table as the converter generates it.
        var fsq = new float[4 * Lfm3SpeechConfig.EndOfAudioCode];
        int[] basis = { 1, 8, 56, 336 }, levels = { 8, 7, 6, 6 };
        for (int code = 0; code < Lfm3SpeechConfig.EndOfAudioCode; code++)
            for (int l = 0; l < 4; l++)
            {
                int half = levels[l] / 2;
                fsq[code * 4 + l] = (float)((code / basis[l] % levels[l] - half) / (double)half);
            }
        w.AddF32("a.gen.wav.quant.first.codebook.weight", fsq, 4, Lfm3SpeechConfig.EndOfAudioCode);
        w.AddF32("a.gen.wav.quant.first.out_proj.weight", Rand(rng, 48 * c.QuantizerDim, -0.3f, 0.3f), 48, c.QuantizerDim);
        w.AddF32("a.gen.wav.quant.first.out_proj.bias", Rand(rng, c.QuantizerDim, -0.1f, 0.1f), c.QuantizerDim);
        int cw = c.CodecWidth, ckv = c.CodecKvHeads * c.CodecHeadDim;
        Product("a.gen.wav.tfm.in_proj.weight", c.QuantizerDim, cw);
        w.AddF32("a.gen.wav.tfm.in_proj.bias", Rand(rng, cw, -0.1f, 0.1f), cw);
        for (int l = 0; l < c.CodecLayers; l++)
        {
            string p = $"a.gen.wav.tfm.blk.{l}.";
            foreach (string n in new[] { "ln1", "attn_post_norm", "ln2", "ffn_post_norm" }) Norm(p + n + ".weight", cw);
            if (c.CodecConvLayer[l])
            {
                Product(p + "conv_pw1.weight", cw, 3 * cw);
                w.AddF32(p + "conv_dw.weight", Rand(rng, c.CodecConvTaps * cw, -0.5f, 0.5f), c.CodecConvTaps, cw);
                Product(p + "conv_pw2.weight", cw, cw);
            }
            else
            {
                Product(p + "attn_q.weight", cw, c.CodecHeads * c.CodecHeadDim);
                Product(p + "attn_k.weight", cw, ckv);
                Product(p + "attn_v.weight", cw, ckv);
                Norm(p + "attn_q_norm.weight", c.CodecHeadDim);
                Norm(p + "attn_k_norm.weight", c.CodecHeadDim);
                Product(p + "attn_gate.weight", cw, c.CodecHeads);
                Product(p + "attn_out.weight", c.CodecHeads * c.CodecHeadDim, cw);
            }
            Product(p + "ffn_gate.weight", cw, c.CodecFeedForward);
            Product(p + "ffn_up.weight", cw, c.CodecFeedForward);
            Product(p + "ffn_down.weight", c.CodecFeedForward, cw);
        }
        Norm("a.gen.wav.tfm.output_norm.weight", cw);
        Product("a.gen.wav.tfm.out_proj.weight", cw, 2 * c.Bins);
        // Log-magnitudes around -2 (a quiet waveform), phases anywhere.
        float[] bias = Rand(rng, 2 * c.Bins, -0.2f, 0.2f);
        for (int i = 0; i < c.Bins; i++) bias[i] -= 2f;
        w.AddF32("a.gen.wav.tfm.out_proj.bias", bias, 2 * c.Bins);
    }

    /// <summary>The speech output over the lane's synthetic weights (the feedback table F16 host rows).</summary>
    internal static Lfm3SpeechOutput SyntheticOutput(LaneContext lane, string variant, int seed, out float[] feedback)
    {
        Lfm3SpeechConfig c = SyntheticConfig();
        var rng = new Random(seed);
        AddSyntheticWeights(lane.Weights, c, variant, rng);
        feedback = Rand(rng, c.HiddenSize * Cb * V, -0.1f, 0.1f).Select(v => (float)(System.Half)v).ToArray();
        var raw = new byte[feedback.Length * 2];
        for (int i = 0; i < feedback.Length; i++) BitConverter.TryWriteBytes(raw.AsSpan(2 * i), BitConverter.HalfToUInt16Bits((System.Half)feedback[i]));
        var feedbackRows = Lfm3SpeechHostRows.FromRaw(GraphFormat.F16, raw, c.HiddenSize);
        NetworkWeight table = lane.Weights.Get("a.gen.code.embd.weight");
        Lfm3SpeechHostRows codeRows = table.Quantized != null
            ? Lfm3SpeechHostRows.Borrow(table.Quantized)
            : Lfm3SpeechHostRows.FromFloats(lane.Weights.ReadF32("a.gen.code.embd.weight"), c.DepthWidth);
        return new Lfm3SpeechOutput(c, lane.Weights, feedbackRows, codeRows, lane.Graph.Executor);
    }

    private void CheckNetworks(Lane laneKind, string variant)
    {
        using var lane = new LaneContext(laneKind);
        using Lfm3SpeechOutput speech = SyntheticOutput(lane, variant, 17, out float[] feedbackTable);
        Lfm3SpeechConfig c = speech.Config;
        var reference = new HostReference(lane.Weights, c);
        // CPU block products quantize their activations to 8 bits; GGML metal / MLX products and attention round to F16, and
        // so do direct CUDA's F16-weight products (CudaQuantizedOps.RunF16Gemm narrows the activations to F16 for cuBLAS's F16
        // GEMM at every row count; the streamed bound below already counts direct CUDA among the F16-operand lanes).
        bool halfOperands = laneKind == Lane.Mlx || HalfOperandLane(laneKind) || (laneKind == Lane.Cuda && variant == "f16");
        double tol = variant == "q8_0" ? 6e-2 : halfOperands ? 1.5e-2 : variant == "f16" && laneKind == Lane.GgmlCpu ? 1e-2 : 2e-3;
        var rng = new Random(23);

        // ---- depthformer: a frame's three passes (the projection kept between them), codes revealed 4 at a time ----
        float[] hidden = Rand(rng, c.HiddenSize, -2f, 2f);
        int[] order = Enumerable.Range(0, Cb).OrderBy(_ => rng.Next()).ToArray();
        int[] codes = Enumerable.Repeat(-1, Cb).ToArray();
        var logits = new float[4 * V];
        double worst = 0;
        for (int pass = 0; pass < 3; pass++)
        {
            int[] pick = order.Skip(4 * pass).Take(4).ToArray();
            speech.Depthformer.Pass(pass == 0 ? hidden : ReadOnlySpan<float>.Empty, codes, pick, logits);
            float[][] expected = reference.Depthformer(hidden, codes);
            for (int j = 0; j < 4; j++)
            {
                AssertClose(expected[pick[j]], logits.AsSpan(j * V, V).ToArray(), tol, $"{laneKind}/{variant} depthformer pass {pass} codebook {pick[j]}");
                worst = Math.Max(worst, MaxRelative(expected[pick[j]], logits.AsSpan(j * V, V)));
            }
            foreach (int cb in pick) codes[cb] = rng.Next(V);
        }
        // Every codebook at once (the head over 12 rows).
        speech.Depthformer.Pass(hidden, codes, Enumerable.Range(0, Cb).ToArray(), new float[Cb * V]);
        Assert.Equal(speech.Depthformer.SelectedHead, lane.Weights.Get("a.gen.code.embd.weight").IsQuantized);

        // ---- the feedback row: the plain sum of the 12 codebooks' rows ----
        int[] frame = Enumerable.Range(0, Cb).Select(_ => rng.Next(V)).ToArray();
        float[] row = speech.FeedbackRow(frame);
        var expectedRow = new float[c.HiddenSize];
        for (int cb = 0; cb < Cb; cb++)
            for (int i = 0; i < c.HiddenSize; i++) expectedRow[i] += feedbackTable[((long)cb * V + frame[cb]) * c.HiddenSize + i];
        AssertClose(expectedRow, row, 1e-6, "feedback row");

        // ---- codec: 5 frames one at a time, then all at once on a fresh stream, against the host forward ----
        const int Frames = 5;
        int[] all = Enumerable.Range(0, Frames * Cb).Select(_ => rng.Next(Lfm3SpeechConfig.EndOfAudioCode)).ToArray();
        float[] expectedPcm = reference.CodecPcm(all, Frames);
        float[] streamed;
        using (Lfm3CodecStream stream = speech.CreateCodecStream())
        {
            var pcm = new List<float>();
            for (int f = 0; f < Frames; f++)
            {
                int n = stream.Decode(all.AsSpan(f * Cb, Cb), pcm);
                Assert.Equal(f == 0 ? c.SamplesPerFrame - 480 : c.SamplesPerFrame, n);
            }
            Assert.Equal(480, stream.Flush(pcm));
            streamed = pcm.ToArray();
            // After the flush the stream is cold again: the same frames give the same samples.
            float[] again = stream.Decode(all.AsSpan(0, Cb));
            AssertClose(streamed[..again.Length], again, 1e-6, $"{laneKind}/{variant} codec after flush");
            if (stream.Graph.Executor.Capabilities.Replays) Assert.Equal(2, stream.Graph.Builds);   // the cold and the warm one-frame programs
        }
        Assert.Equal(Frames * c.SamplesPerFrame, streamed.Length);
        // The log-magnitudes' errors grow through exp(); 8-bit activations (CPU block products) over five layers most.
        double codecTolerance = variant == "q8_0" ? 0.25 : tol * 2;
        AssertClose(expectedPcm, streamed, codecTolerance, $"{laneKind}/{variant} codec pcm");
        float[] whole;
        using (Lfm3CodecStream stream = speech.CreateCodecStream())
        {
            var pcm = new List<float>(stream.Decode(all));
            stream.Flush(pcm);
            whole = pcm.ToArray();
        }
        // Five frames at once are held to the host forward as one frame at a time is.
        AssertClose(expectedPcm, whole, codecTolerance, $"{laneKind}/{variant} codec pcm, frames at once");
        // A frame at a time and five at once carry the decoder's state differently and must agree. Direct CUDA's Q8_0
        // products give a row the same bits whatever the row count, so there the two are equal bit for bit (a state-carry
        // bug cannot hide inside a rounding bound), and its F16 and F32 GEMMs keep them within 1e-2 [RTX PRO 4500, 24
        // seeds: f16 1.1e-3, f32 2.8e-6; A40 f16 5.4e-4]. ggml-cuda and ggml-vulkan pick their product kernels by row
        // count (a frame is 6 rows, five are 30: ggml-cuda's mmvq/mmvf against mmq/mmf/cuBLAS), so the two run other sums
        // and operand roundings, which the codec's exp() amplifies [RTX PRO 4500, 24 seeds: ggml-cuda q8_0 up to 0.198
        // (median 7.8e-2), f16 1.10e-2, f32 1.41e-2; ggml-vulkan up to 9.6e-3; A40 ggml-cuda q8_0 6.2e-2]. Each of the two
        // is within codecTolerance of the host forward, so by the triangle inequality they are within twice that of each
        // other; that is their bound there. Elsewhere: F16-operand rounding on ggml-metal and MLX, none on the CPU lanes.
        if (laneKind == Lane.Cuda && variant == "q8_0")
            Assert.Equal(streamed, whole);
        else
        {
            double streamedTolerance = laneKind is Lane.GgmlCuda or Lane.GgmlVulkan ? 2 * codecTolerance
                : laneKind == Lane.Mlx || laneKind == Lane.Cuda || HalfOperandLane(laneKind) ? 1e-2 : 1e-4;
            AssertClose(streamed, whole, streamedTolerance, $"{laneKind}/{variant} codec whole vs streamed");
        }
        _output.WriteLine($"{laneKind}/{variant}: depthformer max error / scale {worst:G3}; codec pcm {MaxRelative(expectedPcm, streamed):G3}");

        // ---- the sampler: llama.cpp's MaskGIT over the pass logits this lane computed ----
        CheckSampler(speech, Rand(new Random(3), c.HiddenSize * 4, -2f, 2f));
    }

    private static double MaxRelative(float[] expected, ReadOnlySpan<float> actual)
    {
        double scale = 1e-12, worst = 0;
        for (int i = 0; i < expected.Length; i++) scale = Math.Max(scale, Math.Abs(expected[i]));
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs(actual[i] - expected[i]));
        return worst / scale;
    }

    /// <summary>Four frames sampled with a seed against a port of llama.cpp's sample_frame / apply_rep_penalty / sample
    /// over the same pass logits (captured): the same orders, penalties, draws and codes.</summary>
    private static void CheckSampler(Lfm3SpeechOutput speech, float[] hiddenRows)
    {
        var sampling = new Lfm3SpeechSampling { Temperature = 0.4f, RepetitionPenalty = 1.2f, RepetitionWindow = 2 };
        Lfm3FrameSampler sampler = speech.CreateSampler(321, sampling);
        var captured = new List<(int Pass, int Codebook, float[] Logits)>();
        sampler.OnPassLogits = (pass, cb, l) => captured.Add((pass, cb, l.ToArray()));
        var rng = new Lfm3SpeechRandom(321);
        var history = new List<int[]>();
        int h = speech.Config.HiddenSize;
        for (int f = 0; f < 4; f++)
        {
            captured.Clear();
            Lfm3SpeechFrame frame = sampler.Sample(hiddenRows.AsSpan(f * h, h));
            int[] order = Enumerable.Range(0, Cb).ToArray();
            rng.Shuffle(order);
            Assert.Equal(order, captured.Select(x => x.Codebook));
            Assert.Equal(order, sampler.LastOrder.ToArray());
            var codes = new int[Cb];
            foreach ((int pass, int cb, float[] l) in captured)
            {
                Assert.Equal(pass, Array.IndexOf(order, cb) / 4);
                // apply_rep_penalty over the last 2 frames, each code once
                var seen = new HashSet<int>();
                foreach (int[] past in history.Skip(Math.Max(0, history.Count - 2)))
                    if (seen.Add(past[cb])) l[past[cb]] = l[past[cb]] > 0 ? l[past[cb]] / 1.2f : l[past[cb]] * 1.2f;
                float max = l.Max();
                double[] prob = l.Select(v => Math.Exp((double)(v - max) / 0.4f)).ToArray();
                double r = rng.NextDouble() * prob.Sum();
                int pick = V - 1;
                for (int v = 0; v < V; v++)
                {
                    r -= prob[v];
                    if (r <= 0) { pick = v; break; }
                }
                codes[cb] = pick;
            }
            bool eos = codes.Contains(Lfm3SpeechConfig.EndOfAudioCode);
            Assert.Equal(eos, frame.IsEndOfAudio);
            Assert.Equal(eos ? Enumerable.Repeat(Lfm3SpeechConfig.EndOfAudioCode, Cb).ToArray() : codes, frame.Codes);
            if (!eos) history.Add(codes);
        }
        Assert.Equal(Math.Min(2, history.Count), sampler.History.Count);
    }

    /// <summary>An independent double-precision forward of both networks over the weights' host values.</summary>
    internal sealed class HostReference
    {
        private readonly NetworkWeights _w;
        private readonly Lfm3SpeechConfig _c;
        private readonly Dictionary<string, float[]> _values = new();

        public HostReference(NetworkWeights weights, Lfm3SpeechConfig config)
        {
            _w = weights;
            _c = config;
        }

        private float[] W(string name) => _values.TryGetValue(name, out float[] v) ? v : _values[name] = _w.ReadF32(name);

        /// <summary>y[o] = sum_i w[o * in + i] x[i] (+ b[o]).</summary>
        private double[] Linear(string weight, double[] x, string bias = null)
        {
            float[] w = W(weight);
            int inWidth = x.Length, outRows = w.Length / inWidth;
            float[] b = bias == null ? null : W(bias);
            var y = new double[outRows];
            for (int o = 0; o < outRows; o++)
            {
                double s = b == null ? 0 : b[o];
                for (int i = 0; i < inWidth; i++) s += (double)w[(long)o * inWidth + i] * x[i];
                y[o] = s;
            }
            return y;
        }

        private static double[] RmsNorm(double[] x, int offset, int n, float[] w, int wOffset, double eps)
        {
            double ss = 0;
            for (int i = 0; i < n; i++) ss += x[offset + i] * x[offset + i];
            double s = 1 / Math.Sqrt(ss / n + eps);
            var y = new double[n];
            for (int i = 0; i < n; i++) y[i] = x[offset + i] * s * (w == null ? 1 : w[wOffset + i]);
            return y;
        }

        private double[] Norm(double[] x, string w, double eps) => RmsNorm(x, 0, x.Length, W(w), 0, eps);

        private static void HeadNorms(double[] x, int heads, int d, float[] w, double eps)
        {
            for (int h = 0; h < heads; h++) RmsNorm(x, h * d, d, w, 0, eps).CopyTo(x, h * d);
        }

        private static void Neox(double[] x, int heads, int d, int pos, double theta)
        {
            for (int h = 0; h < heads; h++)
                for (int i = 0; i < d / 2; i++)
                {
                    // ggml computes the angle in float: pos * base^(-2i/d)
                    double angle = pos * Math.Pow(theta, -2.0 * i / d);
                    double c = Math.Cos(angle), s = Math.Sin(angle);
                    double a = x[h * d + i], b = x[h * d + i + d / 2];
                    x[h * d + i] = a * c - b * s;
                    x[h * d + i + d / 2] = a * s + b * c;
                }
        }

        private static double Silu(double v) => v / (1 + Math.Exp(-v));

        private double[] Ffn(double[] x, string p)
        {
            double[] g = Linear(p + "ffn_gate.weight", x), u = Linear(p + "ffn_up.weight", x);
            for (int i = 0; i < g.Length; i++) g[i] = Silu(g[i]) * u[i];
            return Linear(p + "ffn_down.weight", g);
        }

        /// <summary>Attention mixer over rows <paramref name="a"/> (normed inputs) at positions pos[t]; visible(q, k).</summary>
        private double[][] Attention(double[][] a, string p, int heads, int kvHeads, int d, int[] pos, double theta, double eps,
            Func<int, int, bool> visible)
        {
            int rows = a.Length, group = heads / kvHeads;
            var q = new double[rows][];
            var k = new double[rows][];
            var v = new double[rows][];
            for (int t = 0; t < rows; t++)
            {
                q[t] = Linear(p + "attn_q.weight", a[t]);
                k[t] = Linear(p + "attn_k.weight", a[t]);
                v[t] = Linear(p + "attn_v.weight", a[t]);
                HeadNorms(q[t], heads, d, W(p + "attn_q_norm.weight"), eps);
                HeadNorms(k[t], kvHeads, d, W(p + "attn_k_norm.weight"), eps);
                Neox(q[t], heads, d, pos[t], theta);
                Neox(k[t], kvHeads, d, pos[t], theta);
            }
            var output = new double[rows][];
            for (int t = 0; t < rows; t++)
            {
                var o = new double[heads * d];
                double[] gate = Linear(p + "attn_gate.weight", a[t]);
                for (int h = 0; h < heads; h++)
                {
                    int kh = h / group;
                    var score = new double[rows];
                    double max = double.NegativeInfinity;
                    for (int s = 0; s < rows; s++)
                    {
                        if (!visible(t, s)) { score[s] = double.NegativeInfinity; continue; }
                        double dot = 0;
                        for (int i = 0; i < d; i++) dot += q[t][h * d + i] * k[s][kh * d + i];
                        score[s] = dot / Math.Sqrt(d);
                        max = Math.Max(max, score[s]);
                    }
                    double sum = 0;
                    for (int s = 0; s < rows; s++) sum += score[s] = Math.Exp(score[s] - max);
                    double g = 1 / (1 + Math.Exp(-gate[h]));
                    for (int i = 0; i < d; i++)
                    {
                        double acc = 0;
                        for (int s = 0; s < rows; s++) acc += score[s] / sum * v[s][kh * d + i];
                        o[h * d + i] = acc * g;
                    }
                }
                output[t] = Linear(p + "attn_out.weight", o);
            }
            return output;
        }

        /// <summary>Every codebook's logits of one pass.</summary>
        public float[][] Depthformer(float[] hidden, int[] codes)
        {
            int width = _c.DepthWidth;
            double eps = _c.DepthEps;
            double[] proj = Linear("a.gen.code.proj_in.weight", hidden.Select(v => (double)v).ToArray(), "a.gen.code.proj_in.bias");
            float[] table = W("a.gen.code.embd.weight"), mask = W("a.gen.code.mask_embd");
            var x = new double[Cb][];
            for (int c = 0; c < Cb; c++)
            {
                x[c] = new double[width];
                for (int i = 0; i < width; i++)
                    x[c][i] = proj[c * width + i] + (codes[c] >= 0 ? table[((long)c * V + codes[c]) * width + i] : mask[i]);
            }
            int[] pos = Enumerable.Range(0, Cb).ToArray();
            for (int l = 0; l < _c.DepthLayers; l++)
            {
                string p = $"a.gen.code.blk.{l}.";
                double[][] a = x.Select(r => Norm(r, p + "ln1.weight", eps)).ToArray();
                double[][] attn = Attention(a, p, _c.DepthHeads, _c.DepthKvHeads, _c.DepthHeadDim, pos, _c.DepthRopeTheta, eps, (_, _) => true);
                for (int c = 0; c < Cb; c++)
                {
                    for (int i = 0; i < width; i++) x[c][i] += attn[c][i];
                    double[] f = Ffn(Norm(x[c], p + "ln2.weight", eps), p);
                    for (int i = 0; i < width; i++) x[c][i] += f[i];
                }
            }
            float[] headNorm = W("a.gen.code.head_norm.weight");
            var logits = new float[Cb][];
            for (int c = 0; c < Cb; c++)
            {
                double[] y = RmsNorm(Norm(x[c], "a.gen.code.output_norm.weight", eps), 0, width, headNorm, c * width, eps);
                logits[c] = new float[V];
                for (int v = 0; v < V; v++)
                {
                    double s = 0;
                    for (int i = 0; i < width; i++) s += table[((long)c * V + v) * width + i] * y[i];
                    logits[c][v] = (float)s;
                }
            }
            return logits;
        }

        /// <summary>The codec over <paramref name="frames"/> frames at once (a cold stream), then the ISTFT.</summary>
        public float[] CodecPcm(int[] codes, int frames)
        {
            int width = _c.CodecWidth, T = frames * _c.FrameStack;
            double eps = _c.CodecEps;
            float[] fsq = W("a.gen.wav.quant.first.codebook.weight");
            var x = new double[T][];
            for (int f = 0; f < frames; f++)
            {
                var q = new double[4 * Cb];
                for (int c = 0; c < Cb; c++)
                    for (int l = 0; l < 4; l++) q[c * 4 + l] = fsq[codes[f * Cb + c] * 4 + l];
                double[] h = Linear("a.gen.wav.tfm.in_proj.weight", Linear("a.gen.wav.quant.first.out_proj.weight", q, "a.gen.wav.quant.first.out_proj.bias"),
                    "a.gen.wav.tfm.in_proj.bias");
                for (int s = 0; s < _c.FrameStack; s++) x[f * _c.FrameStack + s] = (double[])h.Clone();
            }
            int[] pos = Enumerable.Range(0, T).ToArray();
            for (int l = 0; l < _c.CodecLayers; l++)
            {
                string p = $"a.gen.wav.tfm.blk.{l}.";
                double[][] a = x.Select(r => Norm(r, p + "ln1.weight", eps)).ToArray();
                double[][] mixed;
                if (_c.CodecConvLayer[l])
                {
                    float[] kernel = W(p + "conv_dw.weight");
                    int taps = _c.CodecConvTaps;
                    double[][] bcx = a.Select(r => Linear(p + "conv_pw1.weight", r)).ToArray();
                    mixed = new double[T][];
                    for (int t = 0; t < T; t++)
                    {
                        var y = new double[width];
                        for (int ch = 0; ch < width; ch++)
                        {
                            double s = 0;
                            for (int j = 0; j < taps; j++)
                            {
                                int src = t - (taps - 1) + j;   // causal: the window's zeros before the stream
                                if (src >= 0) s += bcx[src][ch] * bcx[src][2 * width + ch] * kernel[ch * taps + j];
                            }
                            y[ch] = bcx[t][width + ch] * s;
                        }
                        mixed[t] = Linear(p + "conv_pw2.weight", y);
                    }
                }
                else
                {
                    int window = _c.CodecWindow;
                    mixed = Attention(a, p, _c.CodecHeads, _c.CodecKvHeads, _c.CodecHeadDim, pos, _c.CodecRopeTheta, eps,
                        (qt, kt) => kt <= qt && kt > qt - window);
                }
                for (int t = 0; t < T; t++)
                {
                    double[] m = Norm(mixed[t], p + "attn_post_norm.weight", eps);
                    for (int i = 0; i < width; i++) x[t][i] += m[i];
                    double[] f = Norm(Ffn(Norm(x[t], p + "ln2.weight", eps), p), p + "ffn_post_norm.weight", eps);
                    for (int i = 0; i < width; i++) x[t][i] += f[i];
                }
            }
            var istft = new Lfm3StreamingIstft(_c.FftSize, _c.HopLength);
            var pcm = new List<float>();
            int bins = _c.Bins;
            for (int t = 0; t < T; t++)
            {
                double[] y = Linear("a.gen.wav.tfm.out_proj.weight", Norm(x[t], "a.gen.wav.tfm.output_norm.weight", eps), "a.gen.wav.tfm.out_proj.bias");
                float[] spec = y.Select(v => (float)v).ToArray();
                istft.ProcessFrame(spec.AsSpan(0, bins), spec.AsSpan(bins, bins), pcm);
            }
            istft.Flush(pcm);
            return pcm.ToArray();
        }
    }
}
