// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Cli;
using TensorSharp.Models;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Lfm3;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The CLI's speech flags (--tts, --speech-out, --speech-mode, --speech-temperature, and the --voice / --language they
/// share with the TTS and ASR models): one spelling each, documented on the --help page and on both usage pages (USAGE.md,
/// USAGE_zh-cn.md), parsed by the audio options, refused where they cannot apply; and a chat model that speaks reaches the
/// TTS route only with --tts.
/// </summary>
public sealed class SpeechCliFlagTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-speech-cli-" + Guid.NewGuid().ToString("N"));

    public SpeechCliFlagTests() => Directory.CreateDirectory(_dir);

    public void Dispose()
    {
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    public static TheoryData<string, string> SpeechFlags() => new()
    {
        { "--tts", null },
        { "--speech-out", "reply.wav" },
        { "--speech-mode", "audio" },
        { "--speech-temperature", "0.6" },
        { "--voice", "male" },
        { "--language", "zh" },
    };

    [Theory]
    [MemberData(nameof(SpeechFlags))]
    public void EverySpeechFlag_IsDocumentedOnTheHelpPageAndBothUsagePages(string flag, string value)
    {
        Assert.Contains(flag, CliUsage.DocumentedFlags());
        string root = Lfm3AudioFrontEndTests.RepoRoot();
        foreach (string page in new[] { "USAGE.md", "USAGE_zh-cn.md" })
            Assert.Contains(flag, File.ReadAllText(Path.Combine(root, page)), StringComparison.Ordinal);
        // The page's shape is the parser's: a value placeholder exactly where the flag takes one.
        string[] spelled = CliUsage.NormalizeOptionSpellings(value == null ? new[] { flag } : new[] { flag + "=" + value });
        Assert.Equal(value == null ? new[] { flag } : new[] { flag, value }, spelled);
    }

    [Theory]
    [MemberData(nameof(SpeechFlags))]
    public void EverySpeechFlag_IsReadByTheAudioOptions(string flag, string value)
    {
        Assert.True(AudioCliOptions.Handles(flag));
        var o = new AudioCliOptions();
        string[] args = value == null ? new[] { flag } : new[] { flag, value };
        int i = 0;
        o.Read(args, ref i);
        Assert.Equal(args.Length - 1, i);
        switch (flag)
        {
            case "--tts": Assert.True(o.Tts); break;
            case "--speech-out": Assert.Equal("reply.wav", o.SpeechOut); break;
            case "--speech-mode": Assert.Equal(SpeechReplyMode.Audio, o.SpeechMode); break;
            case "--speech-temperature": Assert.Equal(0.6f, o.SpeechTemperature); break;
            case "--voice": Assert.Equal("male", o.Voice); break;
            case "--language": Assert.Equal("zh", o.Language); break;
        }
    }

    [Theory]
    [InlineData("--speech-mode", "mixed")]
    [InlineData("--speech-temperature", "-0.1")]
    public void BadSpeechValues_AreRefusedWithTheirFlag(string flag, string value)
    {
        var o = new AudioCliOptions();
        int i = 0;
        var ex = Assert.Throws<ArgumentException>(() => o.Read(new[] { flag, value }, ref i));
        Assert.Contains(flag, ex.Message, StringComparison.Ordinal);
    }

    private Lfm3RuntimeModel Synthetic() => new(new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.F32 }
        .Write(Path.Combine(_dir, "lfm3.gguf")), BackendType.Cpu);

    private static AudioCliOptions Options(params string[] args)
    {
        var o = new AudioCliOptions();
        for (int i = 0; i < args.Length; i++) o.Read(args, ref i);
        return o;
    }

    /// <summary>--speech-out needs a model that speaks now, and a single-prompt run; --speech-mode and --speech-temperature
    /// need --speech-out (or --tts): anywhere else they would do nothing, so the run is refused, naming why.</summary>
    [Fact]
    public void SpeechFlags_AreRefusedWhereTheyCannotApply()
    {
        using Lfm3RuntimeModel model = Synthetic();
        Assert.Null(TensorSharp.Cli.Program.SpeechOutRefusal(model, Options(), false, false, false));
        Assert.Contains("does not speak", TensorSharp.Cli.Program.SpeechOutRefusal(model, Options("--speech-out", "a.wav"), false, false, false));
        Assert.Contains("--speech-out", TensorSharp.Cli.Program.SpeechOutRefusal(model, Options("--speech-mode", "audio"), false, false, false));
        Assert.Contains("--tts or --speech-out", TensorSharp.Cli.Program.SpeechOutRefusal(model, Options("--speech-temperature", "0.3"), false, false, false));
        Assert.Null(TensorSharp.Cli.Program.SpeechOutRefusal(model, Options("--tts", "--speech-temperature", "0.3"), false, false, false));

        using var lane = new NetworkGraphTests.LaneContext(NetworkGraphTests.Lane.Managed);
        model.AttachSpeechOutputForTests(SpeechStub(lane), new Lfm3SpeechTokens { AudioStart = 240, MixedStart = 241, TextEnd = 242, ImEnd = 2, AudioOut = 243 });
        Assert.Null(TensorSharp.Cli.Program.SpeechOutRefusal(model, Options("--speech-out", "a.wav"), false, false, false));
        Assert.Contains("female, male", TensorSharp.Cli.Program.SpeechOutRefusal(model, Options("--speech-out", "a.wav", "--voice", "alloy"), false, false, false));
        Assert.Contains("single-prompt", TensorSharp.Cli.Program.SpeechOutRefusal(model, Options("--speech-out", "a.wav"), true, false, false));
        Assert.Contains("single-prompt", TensorSharp.Cli.Program.SpeechOutRefusal(model, Options("--speech-out", "a.wav"), false, true, false));
    }

    /// <summary>A chat model that speaks is a TTS only with --tts: without it the CLI's audio route passes the prompt on to
    /// chat; with it and no speech decoder the run is refused naming what to load.</summary>
    [Fact]
    public void AChatModelThatSpeaks_ReachesTheTtsRouteOnlyWithTts()
    {
        using Lfm3RuntimeModel model = Synthetic();
        Assert.False(AudioCli.TryRun(model, Options(), "Hello", null, null, 0, 0));
        var ex = Assert.Throws<ArgumentException>(() => AudioCli.TryRun(model, Options("--tts"), "Hello", null, null, 0, 0));
        Assert.Contains("does not speak", ex.Message, StringComparison.Ordinal);
        var both = Assert.Throws<ArgumentException>(() => AudioCli.TryRun(model, Options("--tts", "--speech-out", "a.wav"), "Hello", null, null, 0, 0));
        Assert.Contains("--speech-out", both.Message, StringComparison.Ordinal);
    }

    private static Lfm3SpeechOutput SpeechStub(NetworkGraphTests.LaneContext lane)
    {
        Lfm3SpeechConfig c = Lfm3SpeechConfig.ForTest(hidden: Lfm3SyntheticModelBuilder.Hidden, depthWidth: 64, depthFfn: 96, depthLayers: 1,
            depthHeads: 2, depthKvHeads: 1, codecWidth: 64, codecFfn: 96, codecHeads: 2, codecKvHeads: 1, codecHeadDim: 32,
            codecConvLayer: new[] { true, false }, quantizerDim: 32);
        Lfm3SpeechOutputTests.AddSyntheticWeights(lane.Weights, c, "f32", new Random(1));
        var feedback = new float[c.HiddenSize * Lfm3SpeechConfig.Codebooks * Lfm3SpeechConfig.CodebookSize];
        return new Lfm3SpeechOutput(c, lane.Weights, Lfm3SpeechHostRows.FromFloats(feedback, c.HiddenSize),
            Lfm3SpeechHostRows.FromFloats(lane.Weights.ReadF32("a.gen.code.embd.weight"), c.DepthWidth), lane.Graph.Executor);
    }
}
