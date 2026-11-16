// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// The model-free parts of the audio families: Kokoro's Mandarin number reading and signal
// primitives, Whisper's transcript formats and repetition check, ACE-Step's schedule, prompts,
// mask rule and sampler, and their registration. The expected values were produced by the
// reference implementations (cn2an, torch, ComfyUI) these ports follow.
using System;
using System.Collections.Generic;
using System.Linq;
using TensorSharp.Models.AceStep;
using TensorSharp.Models.Architecture;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Kokoro;
using TensorSharp.Models.Whisper;
using Xunit;

namespace InferenceWeb.Tests;

public class AudioModelTests
{
    [Theory]
    [InlineData("我有3个苹果", "我有三个苹果")]
    [InlineData("2024年11月15日", "二零二四年十一月十五日")]
    [InlineData("100%", "百分之一百")]
    [InlineData("3.14", "三点一四")]
    [InlineData("-5", "负五")]
    [InlineData("12345678", "一千二百三十四万五千六百七十八")]
    [InlineData("10", "十")]
    [InlineData("1/3", "三分之一")]
    [InlineData("20-30分钟", "二十到三十分钟")]
    [InlineData("36℃", "三十六摄氏度")]
    [InlineData("第1名", "第一名")]
    [InlineData("1000", "一千")]
    [InlineData("1001", "一千零一")]
    [InlineData("100000", "十万")]
    [InlineData("10.05", "十点零五")]
    [InlineData("一共105元", "一共一百零五元")]
    [InlineData("abc 2 3", "abc 二 三")]
    public void ChineseNumeralsMatchCn2an(string input, string expected) =>
        Assert.Equal(expected, ChineseNumerals.Transform(input));

    [Fact]
    public void MandarinPunctuationMapsToTheModelsMarks() =>
        Assert.Equal("你好, 世界! 是吗?", KokoroChineseG2P.MapPunctuation("你好，世界！是吗？"));

    [Theory]
    [InlineData(1, 1, 0)]
    [InlineData(2, 1, 1)]
    [InlineData(1, 3, 3)]
    [InlineData(6, 1, 3)]
    public void Conv1dMatchesTheDefinition(int stride, int dilation, int padding)
    {
        var rng = new Random(1);
        int cin = 3, cout = 4, t = 37, k = 5;
        float[] x = Random(rng, cin * t), w = Random(rng, cout * cin * k), b = Random(rng, cout);
        float[] y = KokoroOps.Conv1d(x, cin, t, w, cout, k, b, stride, padding, dilation);
        int tOut = (t + 2 * padding - dilation * (k - 1) - 1) / stride + 1;
        Assert.Equal(cout * tOut, y.Length);
        for (int o = 0; o < cout; o++)
            for (int j = 0; j < tOut; j++)
            {
                double s = b[o];
                for (int c = 0; c < cin; c++)
                    for (int tap = 0; tap < k; tap++)
                    {
                        int i = j * stride + tap * dilation - padding;
                        if (i >= 0 && i < t) s += w[(o * cin + c) * k + tap] * x[c * t + i];
                    }
                Assert.Equal(s, y[o * tOut + j], 4);
            }
    }

    [Theory]
    [InlineData(2, 4, 1, 0)]
    [InlineData(10, 20, 5, 0)]
    [InlineData(3, 6, 2, 1)]
    public void ConvTranspose1dMatchesTheDefinition(int stride, int k, int padding, int outputPadding)
    {
        var rng = new Random(2);
        int cin = 3, cout = 2, t = 9;
        float[] x = Random(rng, cin * t), w = Random(rng, cin * cout * k), b = Random(rng, cout);
        float[] y = KokoroOps.ConvTranspose1d(x, cin, t, w, cout, k, b, stride, padding, outputPadding);
        int tOut = (t - 1) * stride - 2 * padding + k + outputPadding;
        var expected = new double[cout * tOut];
        for (int o = 0; o < cout; o++)
            for (int i = 0; i < tOut; i++) expected[o * tOut + i] = b[o];
        for (int c = 0; c < cin; c++)
            for (int o = 0; o < cout; o++)
                for (int j = 0; j < t; j++)
                    for (int tap = 0; tap < k; tap++)
                    {
                        int pos = j * stride - padding + tap;
                        if (pos >= 0 && pos < tOut) expected[o * tOut + pos] += w[(c * cout + o) * k + tap] * x[c * t + j];
                    }
        for (int i = 0; i < expected.Length; i++) Assert.Equal(expected[i], y[i], 4);
    }

    [Fact]
    public void InverseStftReconstructsTheSignal()
    {
        var rng = new Random(3);
        float[] x = Random(rng, 400);
        var (mag, phase, frames) = KokoroOps.Stft(x, 20, 5);
        float[] y = KokoroOps.Istft(mag, phase, frames, 20, 5);
        Assert.Equal(400, y.Length);
        for (int i = 0; i < y.Length; i++) Assert.Equal(x[i], y[i], 4);
    }

    [Fact]
    public void RepetitiveTranscriptsCompressPastTheFallbackThreshold()
    {
        string loop = string.Concat(Enumerable.Repeat("I don't know. ", 30));
        Assert.True(WhisperTranscriber.CompressionRatio(loop) > 2.4);
        Assert.True(WhisperTranscriber.CompressionRatio("This week, Nvidia announced a change at the very top.") < 2.4);
    }

    [Fact]
    public void TranscriptFormatsFollowOpenAI()
    {
        var r = new TranscriptionResult
        {
            Text = " Hello there. General Kenobi.", Language = "en", Duration = 4.5, Task = "transcribe",
            Segments = new List<TranscriptionSegment>
            {
                new() { Id = 0, Start = 0, End = 1.5, Text = " Hello there.", Tokens = Array.Empty<int>() },
                new() { Id = 1, Start = 1.5, End = 3725.125, Text = " General Kenobi.", Tokens = Array.Empty<int>() },
            },
        };
        Assert.Equal("1\n00:00:00,000 --> 00:00:01,500\nHello there.\n\n2\n00:00:01,500 --> 01:02:05,125\nGeneral Kenobi.\n\n",
            TranscriptFormats.ToSrt(r));
        Assert.StartsWith("WEBVTT\n\n00:00:00.000 --> 00:00:01.500\nHello there.", TranscriptFormats.ToVtt(r));
        Assert.Equal("{\"text\":\"Hello there. General Kenobi.\"}", TranscriptFormats.Render(r, "json"));
        Assert.Equal("srt", TranscriptFormats.FromExtension("talk.SRT"));
        Assert.Equal("verbose_json", TranscriptFormats.FromExtension("talk.json"));
    }

    [Fact]
    public void AceStepScheduleIsTheShiftedSimpleSchedule()
    {
        float[] s = AceStepModel.Sigmas(8, 3f);
        Assert.Equal(9, s.Length);
        Assert.Equal(1f, s[0], 6);
        Assert.Equal(3 * 0.875f / (1 + 2 * 0.875f), s[1], 6);
        Assert.Equal(3 * 0.125f / (1 + 2 * 0.125f), s[7], 6);
        Assert.Equal(0f, s[8]);
    }

    [Fact]
    public void AceStepPromptsMatchComfyUI()
    {
        var p = AceStepPrompts.Build("lofi, piano", "[Instrumental]", 10, 90, "C major", "4", "en");
        Assert.Contains("<|im_start|>assistant\n<think>\nbpm: 90\nduration: 10\nkeyscale: C major\ntimesignature: 4\n</think>\n\n<|im_end|>\n", p.LmPrompt);
        Assert.Contains("<|im_start|>assistant\n<think>\n\n</think>\n\n<|im_end|>\n", p.LmNegative);
        Assert.Equal("# Languages\nen\n\n# Lyric\n[Instrumental]<|endoftext|><|endoftext|>", p.Lyrics);
        Assert.EndsWith("# Metas\n- bpm: 90\n- timesignature: 4\n- keyscale: C major\n- duration: 10 seconds\n<|endoftext|>\n<|endoftext|>", p.Caption);
        var open = AceStepPrompts.Build("ambient", "", 30, 0, null, null, "en");
        Assert.Contains("<think>\nduration: 30\n</think>", open.LmPrompt);
        Assert.Contains("- bpm: N/A\n- timesignature: N/A\n- keyscale: N/A\n- duration: 30 seconds", open.Caption);
    }

    [Fact]
    public void AceStepMasksPadsLikeSdClip()
    {
        const int pad = 151643;
        Assert.Equal(new[] { true, false, false, false }, AceStepModel.ComfyMask(new[] { 5, pad, 7, pad }));
        Assert.Equal(new[] { false, false, true, true, false }, AceStepModel.ComfyMask(new[] { pad, pad, 5, 6, pad }));
    }

    [Fact]
    public void AceStepSamplerAppliesCfgAndTopP()
    {
        float[] cond = { 1f, 5f, 2f, 0f }, uncond = { 1f, 1f, 3f, 0f };
        // cfg 2: uncond + 2 (cond - uncond) = {1, 9, 1, 0}; greedy picks 1.
        Assert.Equal(1, AceStepModel.SampleCfg(cond, uncond, 2f, 0f, 1f, new Random(0)));
        // top-p 0.5 keeps only the dominant token, so any draw returns it.
        for (int seed = 0; seed < 20; seed++)
            Assert.Equal(1, AceStepModel.SampleCfg(cond, uncond, 2f, 1f, 0.5f, new Random(seed)));
    }

    [Theory]
    [InlineData("whisper")]
    [InlineData("kokoro")]
    [InlineData("ace-step-1.5")]
    public void AudioFamiliesAreRegistered(string id)
    {
        Assert.True(ModelArchitectureRegistry.TryGet(id, out var arch));
        Assert.False(arch.Serving.Autoregressive);
    }

    private static float[] Random(Random rng, int n)
    {
        var a = new float[n];
        for (int i = 0; i < n; i++) a[i] = (float)(rng.NextDouble() * 2 - 1);
        return a;
    }
}
