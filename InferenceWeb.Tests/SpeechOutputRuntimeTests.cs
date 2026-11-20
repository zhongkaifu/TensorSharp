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
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Models;
using TensorSharp.Models.Audio;
using TensorSharp.Models.Graph;
using TensorSharp.Models.Lfm3;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;
using static InferenceWeb.Tests.NetworkGraphTests;

namespace InferenceWeb.Tests;

/// <summary>
/// The runtime under LLM speech output, on synthetic checkpoints (the LFM3 LM of <see cref="Lfm3SyntheticModelBuilder"/>
/// and the speech networks of <see cref="Lfm3SpeechOutputTests"/> at its hidden width):
/// <list type="bullet">
/// <item>slot steps whose input is an embedding row and whose output is the final hidden row
/// (<see cref="IUnifiedBatchModel.TryForwardUnifiedRows"/>), on the graphs and the per-op layers of every Mac lane;</item>
/// <item>the engine serving a request whose slots a step driver chooses, batched with text requests, which keep their
/// replies;</item>
/// <item>the LFM3-Omni reply driver: its schedule, its PCM, its end, and the engine's speech equal to the standalone
/// synthesis.</item>
/// </list>
/// </summary>
public sealed class SpeechOutputRuntimeTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-speech-runtime-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype;
    private readonly bool _dtypeExplicit;
    private readonly Xunit.Abstractions.ITestOutputHelper _output;

    public SpeechOutputRuntimeTests(Xunit.Abstractions.ITestOutputHelper output)
    {
        _output = output;
        Directory.CreateDirectory(_dir);
        (_dtype, _dtypeExplicit) = (KvCacheDtypeConfig.Current, KvCacheDtypeConfig.IsExplicitlySet);
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private string Lm(bool tied = true) => (tied ? new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.F32 }
            : new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.F32, UntiedOutput = true })
        .Write(Path.Combine(_dir, $"lfm3-{(tied ? "tied" : "untied")}.gguf"));

    private static readonly int[] Prompt = Enumerable.Range(0, 13).Select(i => 40 + (i * 11) % 150).ToArray();

    // ------------------------------------------------------------------------------------------------ row I/O

    public static TheoryData<bool> Paths() => new() { true, false };

    /// <summary>A decode row whose input is its token's own embedding row returns the token's logits; the final hidden
    /// row the step returns, times the tied head, is those logits; a step that wants no logits returns the same row
    /// and runs no head.</summary>
    [Theory]
    [MemberData(nameof(Paths))]
    public void RowIoOnTheManagedCpu(bool graphs) => CheckRowIo(BackendType.Cpu, graphs, 1e-4);

    [Theory]
    [MemberData(nameof(Paths))]
    public void RowIoOnThePinnedGgmlBackend(bool graphs) => CheckRowIo(TestGates.PinnedGgmlBackend, graphs, 5e-3);

    [MlxFact]
    public void RowIoOnMlx() => CheckRowIo(BackendType.Mlx, true, 5e-3);

    [CudaTheory]
    [MemberData(nameof(Paths))]
    public void RowIoOnDirectCuda(bool graphs) => CheckRowIo(BackendType.Cuda, graphs, 5e-3);

    /// <summary>
    /// Head-free steps of embedding rows (speech frames: the input an embedding row, the output the final hidden row, no head)
    /// advance in place and, on direct CUDA, record and replay as text decode steps do: the hidden rows, each step's input the
    /// previous step's output rescaled (the frames' feedback chain), are the traced steps' bit for bit. With sliding-window
    /// layers the steps record below the band and run as plans past it; without them every advanced step replays.
    /// </summary>
    [CudaTheory]
    [InlineData("csfcsfsc")]
    [InlineData("cfcfcf")]
    public void HeadFreeRowStepsRecordAndReplayOnDirectCuda(string layout)
    {
        string path = new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.F32, Layout = layout }
            .Write(Path.Combine(_dir, $"lfm3-rows-{layout}.gguf"));
        int[] prompt = { 41, 52 };
        const int Steps = 24;
        (List<float[]> Rows, int Recordings, int Replays, int Advances) Run(bool advance)
        {
            using var model = new Lfm3RuntimeModel(path, BackendType.Cuda) { DecodeAdvanceDisabled = !advance };
            IUnifiedBatchModel unified = model;
            int hidden = model.Config.HiddenSize;
            unified.EnsureSequenceSlot("s", prompt.Length + Steps + 8);
            var h = new float[hidden];
            Assert.True(unified.TryForwardUnifiedRows(new[] { "s" }, new[] { prompt }, new[] { 0 }, new float[1][], new[] { false },
                new float[1][], new[] { h }));
            var rows = new List<float[]> { (float[])h.Clone() };
            for (int t = 0; t < Steps; t++)
            {
                float[] input = h.Select(v => v * 0.5f).ToArray();
                var logits = new float[1][];
                Assert.True(unified.TryForwardUnifiedRows(new[] { "s" }, new[] { new[] { 0 } }, new[] { prompt.Length + t }, new[] { input },
                    new[] { false }, logits, new[] { h }));
                Assert.Null(logits[0]);
                rows.Add((float[])h.Clone());
            }
            return (rows, model.DecodeRecordings, model.RecordedDecodeReplays, model.DecodeAdvances);
        }
        var traced = Run(advance: false);
        var advanced = Run(advance: true);
        _output.WriteLine($"{layout}: {advanced.Advances} advances, {advanced.Recordings} recordings, {advanced.Replays} replays");
        Assert.Equal(0, traced.Advances);
        Assert.True(advanced.Recordings >= 1 && advanced.Replays >= 3, $"{advanced.Recordings} recordings, {advanced.Replays} replays");
        if (!layout.Contains('s')) Assert.Equal(Steps - 2, advanced.Replays);
        for (int i = 0; i < traced.Rows.Count; i++)
            Assert.True(traced.Rows[i].AsSpan().SequenceEqual(advanced.Rows[i]), $"{layout} step {i}: the hidden row moved");
    }

    private void CheckRowIo(BackendType backend, bool graphs, double tolerance)
    {
        using var model = new Lfm3RuntimeModel(Lm(), backend);
        model.GraphLoweringEnabled = graphs;
        IUnifiedBatchModel unified = model;
        Assert.True(unified.SupportsUnifiedRowIO);
        int hidden = model.Config.HiddenSize, vocab = model.Config.VocabSize;
        float[] embedding = ReadF32(model, "token_embd.weight");
        const int next = 77;
        float[] row = embedding.AsSpan(next * hidden, hidden).ToArray();

        // Reference: the plain step with the token.
        unified.EnsureSequenceSlot("a", 64);
        var logitsA = new float[1][];
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { Prompt }, new[] { 0 }, logitsA));
        Assert.True(unified.TryForwardUnified(new[] { "a" }, new[] { new[] { next } }, new[] { Prompt.Length }, logitsA));
        float[] expected = (float[])logitsA[0].Clone();

        // The same row as an embedding row, with the final hidden row returned.
        unified.EnsureSequenceSlot("b", 64);
        var logitsB = new float[1][];
        var h = new float[hidden];
        Assert.True(unified.TryForwardUnifiedRows(new[] { "b" }, new[] { Prompt }, new[] { 0 }, new float[1][], new[] { true }, logitsB, new float[1][]));
        Assert.True(unified.TryForwardUnifiedRows(new[] { "b" }, new[] { new[] { 0 } }, new[] { Prompt.Length }, new[] { row },
            new[] { true }, logitsB, new[] { h }));
        AssertClose(expected, logitsB[0], tolerance, "logits of an embedding row vs its token");
        // The tied head over the returned row.
        var head = new float[vocab];
        for (int v = 0; v < vocab; v++)
        {
            double sum = 0;
            for (int i = 0; i < hidden; i++) sum += (double)embedding[(long)v * hidden + i] * h[i];
            head[v] = (float)sum;
        }
        AssertClose(expected, head, tolerance, "the tied head over the returned final hidden row");

        // No logits wanted: the same row, no head.
        unified.EnsureSequenceSlot("c", 64);
        var logitsC = new float[1][];
        var h2 = new float[hidden];
        Assert.True(unified.TryForwardUnifiedRows(new[] { "c" }, new[] { Prompt }, new[] { 0 }, new float[1][], new[] { false }, logitsC, new float[1][]));
        Assert.True(unified.TryForwardUnifiedRows(new[] { "c" }, new[] { new[] { 0 } }, new[] { Prompt.Length }, new[] { row },
            new[] { false }, logitsC, new[] { h2 }));
        Assert.Null(logitsC[0]);
        AssertClose(h, h2, tolerance, "the hidden row of a head-free step");

        // Batched with a plain request: each request gets its own outputs.
        unified.EnsureSequenceSlot("d", 64);
        unified.EnsureSequenceSlot("e", 64);
        var logitsDe = new float[2][];
        Assert.True(unified.TryForwardUnifiedRows(new[] { "d", "e" }, new[] { Prompt, Prompt }, new[] { 0, 0 }, new float[2][],
            new[] { true, true }, logitsDe, new float[2][]));
        var h3 = new float[hidden];
        Assert.True(unified.TryForwardUnifiedRows(new[] { "d", "e" }, new[] { new[] { 0 }, new[] { next } }, new[] { Prompt.Length, Prompt.Length },
            new[] { row, null }, new[] { false, true }, logitsDe, new[] { h3, null }));
        AssertClose(h, h3, tolerance, "a batched embedding row's hidden row");
        AssertClose(expected, logitsDe[1], tolerance, "the batched plain request's logits");
    }

    private static float[] ReadF32(ModelBase model, string name)
    {
        using var gguf = new GgufFile(model.ModelFilePaths[0]);
        GgufTensorInfo info = gguf.Tensors[name];
        var values = new float[info.NumElements];
        gguf.ReadTensorDataToFloat32(info, values, values.Length);
        return values;
    }

    private static void AssertClose(float[] expected, float[] actual, double tolerance, string what)
    {
        Assert.NotNull(actual);
        Assert.Equal(expected.Length, actual.Length);
        double scale = expected.Max(v => Math.Abs((double)v)), worst = 0;
        for (int i = 0; i < expected.Length; i++) worst = Math.Max(worst, Math.Abs((double)expected[i] - actual[i]));
        Assert.True(worst <= tolerance * Math.Max(1, scale), $"{what}: max |d| {worst:E3} (scale {scale:E3})");
    }

    // ------------------------------------------------------------------------------------------------ engine

    /// <summary>A test driver: <c>pattern</c> 't' = a text slot, 'r' = an embedding row (the hidden row it was given,
    /// rescaled), then a stop; it checks it gets the outputs it asked for.</summary>
    private sealed class PatternDriver : ISequenceStepDriver
    {
        private readonly string _pattern;
        private readonly int _placeholder;
        private int _at;
        public readonly List<string> Seen = new();

        public PatternDriver(string pattern, int placeholder)
        {
            _pattern = pattern;
            _placeholder = placeholder;
        }

        public bool NextNeedsLogits => _at < _pattern.Length && _pattern[_at] == 't';
        public bool NextNeedsHidden => _at < _pattern.Length && _pattern[_at] == 'r';

        public DriverSlot Next(float[] logits, float[] hidden, Func<int> drawToken)
        {
            if (_at >= _pattern.Length) return DriverSlot.Stop("eos");
            char kind = _pattern[_at++];
            if (kind == 't')
            {
                Assert.NotNull(logits);
                int token = drawToken();
                Seen.Add($"t{token}");
                return DriverSlot.Text(token);
            }
            Assert.NotNull(hidden);
            Seen.Add("r");
            return DriverSlot.Embedding(_placeholder, hidden.Select(v => v * 0.5f).ToArray());
        }

        public void Dispose() { }
    }

    private static SchedulerConfig Engine() => new()
    {
        MaxNumBatchedTokens = 256, MaxNumRunningSequences = 4, MaxPrefillChunkSize = 8, NumBlocks = 64, BlockSize = 16,
        EnablePrefixCaching = true,
    };

    private static async Task<(int[] Tokens, string Finish)> Serve(InferenceEngine engine, string id, int[] prompt, int max,
        ISequenceStepDriver driver = null)
    {
        InferenceRequestHandle h = engine.SubmitRequest(new SequenceState(id, prompt, max, engine.PoolStats.blockSize, SamplingConfig.Greedy)
        {
            StepDriver = driver,
        });
        var tokens = new List<int>();
        await foreach (int t in h.Tokens.ReadAllAsync()) tokens.Add(t);
        InferenceCompletion c = await h.Completion;
        return (tokens.ToArray(), c.FinishReason);
    }

    /// <summary>The engine runs a driven request beside text requests in the same steps: the driver gets the logits or
    /// the hidden row it asks for, its placeholders are its token stream (the repetition guard does not stop a run of
    /// them), it ends at its stop with its finish reason, and the text requests' greedy replies are their replies alone.</summary>
    [Theory]
    [MemberData(nameof(Paths))]
    public async Task ADrivenRequestBatchesWithTextRequests(bool graphs)
    {
        const string pattern = "trrrrrrrrrrrrtttrrrrrrrrrrrrt";
        const int placeholder = 250;
        string path = Lm(tied: false);
        int[] other = Enumerable.Range(0, 19).Select(i => 97 + (i * 5) % 26).ToArray();
        int[] alone;
        using (var model = new Lfm3RuntimeModel(path, BackendType.Cpu) { GraphLoweringEnabled = graphs })
        using (var engine = new InferenceEngine(model, Engine(), NullLogger.Instance))
            alone = (await Serve(engine, "text", other, 12).WaitAsync(TimeSpan.FromMinutes(2))).Tokens;

        using (var model = new Lfm3RuntimeModel(path, BackendType.Cpu) { GraphLoweringEnabled = graphs })
        using (var engine = new InferenceEngine(model, Engine(), NullLogger.Instance))
        {
            var driver = new PatternDriver(pattern, placeholder);
            Task<(int[] Tokens, string Finish)> driven = Serve(engine, "driven", Prompt, 100, driver);
            Task<(int[] Tokens, string Finish)> text = Serve(engine, "text", other, 12);
            var (drivenTokens, finish) = await driven.WaitAsync(TimeSpan.FromMinutes(2));
            var (textTokens, _) = await text.WaitAsync(TimeSpan.FromMinutes(2));
            Assert.Equal("eos", finish);
            Assert.Equal(pattern.Length, drivenTokens.Length);
            Assert.Equal(pattern.Length, driver.Seen.Count);
            for (int i = 0; i < pattern.Length; i++)
                if (pattern[i] == 'r') Assert.Equal(placeholder, drivenTokens[i]);
                else Assert.Equal($"t{drivenTokens[i]}", driver.Seen[i]);
            Assert.Equal(alone, textTokens);
            Assert.True(model.BatchedForwardCount > 0);
        }
    }

    /// <summary>A driven request whose step needs no logits batches with a text request that does: the head runs for the
    /// text request; alone, a run of embedding rows runs no head at all.</summary>
    [Fact]
    public async Task AStepOfEmbeddingRowsAloneRunsNoHead()
    {
        using var model = new Lfm3RuntimeModel(Lm(), BackendType.Cpu);
        using var engine = new InferenceEngine(model, Engine(), NullLogger.Instance);
        var driver = new PatternDriver("rrrrrrrrrrrrrrrrrrrrrrrr", 250);
        var (tokens, finish) = await Serve(engine, "rows", Prompt, 100, driver).WaitAsync(TimeSpan.FromMinutes(2));
        Assert.Equal("eos", finish);
        Assert.All(tokens, t => Assert.Equal(250, t));
        Assert.Equal(24, tokens.Length);
    }

    /// <summary>The engine applies the model's reply-token policy to every sampler: a token it suppresses is never drawn,
    /// for a text request and for a driven one (with the driven set).</summary>
    [Fact]
    public void SuppressedTokensAreNeverDrawn()
    {
        var logits = new float[16];
        logits[3] = 10f;
        logits[5] = 9f;
        var sampler = new TokenSampler(SamplingConfig.Greedy) { SuppressedTokens = new[] { 3 } };
        Assert.Equal(5, sampler.Sample((float[])logits.Clone(), new List<int>()));
        Assert.False(sampler.IsPlainGreedyArgmax);
        var sampled = new TokenSampler(new SamplingConfig { Temperature = 1f, TopK = 0, TopP = 1f, Seed = 3 }) { SuppressedTokens = new[] { 3, 5 } };
        for (int i = 0; i < 50; i++)
        {
            int t = sampled.Sample((float[])logits.Clone(), new List<int>());
            Assert.NotEqual(3, t);
            Assert.NotEqual(5, t);
        }
    }

    // ------------------------------------------------------------------------------------------------ the LFM3 reply driver

    private static Lfm3SpeechTokens Tokens() => new()
    {
        AudioStart = 240, MixedStart = 241, TextEnd = 242, ImEnd = 2, AudioOut = 243, Placeholders = new[] { 243, 244 },
    };

    /// <summary>The synthetic speech networks at the synthetic LM's hidden width, on <paramref name="lane"/>.</summary>
    private static Lfm3SpeechOutput Speech(LaneContext lane)
    {
        Lfm3SpeechConfig c = Lfm3SpeechConfig.ForTest(hidden: Lfm3SyntheticModelBuilder.Hidden, depthWidth: 64, depthFfn: 96, depthLayers: 2,
            depthHeads: 2, depthKvHeads: 1, codecWidth: 64, codecFfn: 96, codecHeads: 2, codecKvHeads: 1, codecHeadDim: 32,
            codecConvLayer: new[] { true, true, false, true, false }, quantizerDim: 32);
        var rng = new Random(5);
        Lfm3SpeechOutputTests.AddSyntheticWeights(lane.Weights, c, "f16", rng);
        var feedback = new float[c.HiddenSize * Lfm3SpeechConfig.Codebooks * Lfm3SpeechConfig.CodebookSize];
        for (int i = 0; i < feedback.Length; i++) feedback[i] = (float)(rng.NextDouble() * 0.2 - 0.1);
        NetworkWeight table = lane.Weights.Get("a.gen.code.embd.weight");
        Lfm3SpeechHostRows codeRows = table.Quantized is { HasHostData: true } q
            ? Lfm3SpeechHostRows.Borrow(q)
            : Lfm3SpeechHostRows.FromFloats(lane.Weights.ReadF32("a.gen.code.embd.weight"), c.DepthWidth);
        return new Lfm3SpeechOutput(c, lane.Weights, Lfm3SpeechHostRows.FromFloats(feedback, c.HiddenSize), codeRows, lane.Graph.Executor);
    }

    /// <summary>TTS: every slot is a frame made from the hidden row it is given; each non-end frame's PCM is queued (1440
    /// samples first, 1920 after) and its feedback row is the next input; the end-of-speech frame stops the reply, and the
    /// codec's 480-sample tail is queued then. After it, a slot is a stop; a disposed driver's slot is a cancellation.</summary>
    [Fact]
    public void TheTtsDriverSpeaksUntilTheEndOfSpeech()
    {
        using var lane = new LaneContext(Lane.Managed);
        using Lfm3SpeechOutput speech = Speech(lane);
        int hidden = speech.Config.HiddenSize;
        var driver = new Lfm3SpeechDriver(speech, Tokens(), Tokens().AudioStart, continuing: false, seed: 7,
            Lfm3SpeechSampling.Defaults(speech.Config), null, null, new object());
        var rng = new Random(3);
        int frames = 0;
        long samples = 0;
        var counts = new List<int>();
        DriverSlot slot = default;
        for (int i = 0; i < 400; i++)
        {
            Assert.True(driver.NextNeedsHidden);
            Assert.False(driver.NextNeedsLogits);
            float[] h = Enumerable.Range(0, hidden).Select(_ => (float)(rng.NextDouble() * 2 - 1)).ToArray();
            slot = driver.Next(null, h, () => throw new InvalidOperationException("TTS draws no text"));
            float[] pcm = driver.TakeAudio();
            samples += pcm.Length;
            if (slot.IsStop) break;
            counts.Add(pcm.Length);
            frames++;
            Assert.Equal(Tokens().AudioOut, slot.Token);
            Assert.Equal(speech.FeedbackRow(driver.FrameCodes[^1]), slot.Row);
        }
        Assert.True(slot.IsStop, "a reply of 400 random frames never drew the end-of-speech code");
        Assert.Equal("eos", slot.StopReason);
        if (frames > 0)
        {
            Assert.Equal(1440, counts[0]);
            Assert.All(counts.Skip(1), n => Assert.Equal(1920, n));
            Assert.Equal((long)frames * 1920, samples);   // the end frame flushed the 480-sample tail
        }
        Assert.Equal(frames, driver.FrameCodes.Count);
        Assert.False(driver.NextNeedsHidden);
        driver.Complete();
        Assert.Empty(driver.TakeAudio());
        Assert.True(driver.Next(null, new float[hidden], () => 0).IsStop);
        driver.Dispose();
        Assert.Equal("cancelled", driver.Next(null, new float[hidden], () => 0).StopReason);
    }

    /// <summary>An interleaved reply: 6 text slots (logits asked for, control tokens not counted as transcript), then 12
    /// frames, ...; &lt;|im_end|&gt; in a text slot ends it, and Complete queues the codec's tail of the open segment.</summary>
    [Fact]
    public void TheInterleavedDriverAlternatesTextAndSpeech()
    {
        using var lane = new LaneContext(Lane.Managed);
        using Lfm3SpeechOutput speech = Speech(lane);
        int hidden = speech.Config.HiddenSize;
        Lfm3SpeechTokens tokens = Tokens();
        var sampling = Lfm3SpeechSampling.Defaults(speech.Config);
        sampling.Temperature = 0f;   // argmax: no end-of-speech draw on these rows (checked below)
        var driver = new Lfm3SpeechDriver(speech, tokens, tokens.MixedStart, continuing: true, seed: 7, sampling, null, null, new object());
        Assert.True(driver.IsControlToken(tokens.TextEnd));
        Assert.False(driver.IsControlToken(65));
        var rng = new Random(4);
        var kinds = new List<char>();
        for (int i = 0; i < 6 + 12 + 5; i++)
        {
            if (driver.NextNeedsLogits)
            {
                DriverSlot text = driver.Next(new float[8], null, () => 65 + i);
                Assert.Equal(65 + i, text.Token);
                Assert.Null(text.Row);
                kinds.Add('t');
            }
            else
            {
                float[] h = Enumerable.Range(0, hidden).Select(_ => (float)(rng.NextDouble() * 0.5)).ToArray();
                DriverSlot frame = driver.Next(null, h, () => throw new InvalidOperationException());
                Assert.False(frame.IsStop, "an end-of-speech frame at argmax on these rows");
                kinds.Add('r');
            }
        }
        Assert.Equal(new string('t', 6) + new string('r', 12) + new string('t', 5), new string(kinds.ToArray()));
        Assert.True(driver.NextNeedsLogits);
        Assert.Equal(12 * 1920 - 480, driver.TakeAudio().Length);
        DriverSlot end = driver.Next(new float[8], null, () => tokens.ImEnd);
        Assert.True(end.IsStop);
        Assert.Equal(480, driver.TakeAudio().Length);   // the open segment's tail, at the end of the reply
        driver.Dispose();
    }

    /// <summary>
    /// A spoken reply carried back in a later request's history (<see cref="ChatMessage.SpeechFrames"/>, spliced with its
    /// raw tokens) re-renders as generation decoded it: each <c>&lt;|audio_out|&gt;</c> of the prompt gets its frame's
    /// feedback row (the row the reply decoded), one span a run keyed by its codes, so the same speech in another request
    /// has the same identity for prompt reuse; a prompt whose placeholders and frames disagree is refused.
    /// </summary>
    [Fact]
    public void ASpokenTurnReRendersWithItsFramesRows()
    {
        string path = Lm(tied: false);
        Lfm3SpeechTokens tokens = Tokens();
        using var lane = new LaneContext(Lane.Managed);
        using var model = new Lfm3RuntimeModel(path, BackendType.Cpu);
        Lfm3SpeechOutput speech = Speech(lane);
        model.AttachSpeechOutputForTests(speech, tokens);
        var rng = new Random(3);
        int[] Frame(bool end) => Enumerable.Range(0, Lfm3SpeechConfig.Codebooks)
            .Select(_ => end ? Lfm3SpeechConfig.EndOfAudioCode : rng.Next(Lfm3SpeechConfig.CodebookSize - 1)).ToArray();
        var frames = new List<int[]> { Frame(false), Frame(false), Frame(true) };
        List<ChatMessage> History(IReadOnlyList<int[]> spoken) => new()
        {
            new ChatMessage { Role = "user", Content = "hi" },
            new ChatMessage { Role = "assistant", Content = "Hello.", SpeechFrames = spoken },
            new ChatMessage { Role = "user", Content = "again" },
        };
        int o = tokens.AudioOut;
        // Interleaved: text, two frames, text, the end-of-speech frame, text.
        var prompt = new List<int> { 50, 51, tokens.MixedStart, 60, o, o, 61, o, 52 };

        List<int> placed = model.MultimodalInjector.ProcessPromptTokens(History(frames), prompt, "spoken-1");
        Assert.Equal(prompt, placed);
        IReadOnlyList<PromptMediaSpan> spans = model.MultimodalInjector.GetPreparedMediaSpans("spoken-1");
        Assert.Equal(new[] { (4, 6), (7, 8) }, spans.Select(s => (s.Start, s.End)).ToArray());
        Assert.All(spans, s => Assert.StartsWith("lfm3-speech:", s.ContentId));
        Assert.NotEqual(spans[0].ContentId, spans[1].ContentId);

        // The same speech in another request: the same identities.
        model.MultimodalInjector.ProcessPromptTokens(History(frames), prompt, "spoken-2");
        Assert.Equal(spans.Select(s => s.ContentId), model.MultimodalInjector.GetPreparedMediaSpans("spoken-2").Select(s => s.ContentId));
        // The rows are the frames' feedback rows.
        Assert.True(model.MultimodalInjector.HasPendingEmbeddings("spoken-1"));

        // A frame too few for the placeholders: refused, not given another frame's row.
        var ex = Assert.Throws<InvalidOperationException>(() =>
            model.MultimodalInjector.ProcessPromptTokens(History(frames.Take(2).ToList()), prompt, "spoken-3"));
        Assert.Contains("3 <|audio_out|> placeholder(s) for 2 speech frame(s)", ex.Message);
        model.MultimodalInjector.ClearPreparedPromptState("spoken-1");
        model.MultimodalInjector.ClearPreparedPromptState("spoken-2");
        model.MultimodalInjector.ClearPreparedPromptState("spoken-3");
    }

    /// <summary>End to end on the synthetic checkpoints: a TTS reply served by the engine (batched with a text request) is
    /// the standalone synthesis of the same seed alone, and the text request keeps its reply. Every Mac lane.</summary>
    [Fact]
    public Task EngineSpeechOnTheManagedCpu() => CheckEngineSpeech(Lane.Managed, BackendType.Cpu);

    [Fact]
    public Task EngineSpeechOnThePinnedGgmlBackend() => CheckEngineSpeech(PinnedGgmlLane, TestGates.PinnedGgmlBackend);

    [CudaFact]
    public Task EngineSpeechOnDirectCuda() => CheckEngineSpeech(Lane.Cuda, BackendType.Cuda);

    [MlxFact]
    public Task EngineSpeechOnMlx() => CheckEngineSpeech(Lane.Mlx, BackendType.Mlx);

    private async Task CheckEngineSpeech(Lane laneKind, BackendType backend)
    {
        string path = Lm(tied: false);
        int[] text = Enumerable.Range(0, 11).Select(i => 60 + (i * 7) % 40).ToArray();
        Lfm3SpeechTokens tokens = Tokens();
        int[] prompt = Prompt.Append(tokens.AudioStart).ToArray();
        using var lane = new LaneContext(laneKind);
        using var model = new Lfm3RuntimeModel(path, backend);
        model.AttachSpeechOutputForTests(Speech(lane), tokens);
        Assert.True(model.HasSpeechOutput);

        // Alone, on a slot of its own (no engine).
        float[] alone;
        int aloneFrames;
        using (SpeechReplyDriver d = model.CreateSpeechDriver(prompt, new SpeechReplyOptions { Synthesis = true, Seed = 11 }))
        {
            var pcm = new List<float>();
            model.RunDrivenRequest(prompt, d, 64, new TokenSampler(SamplingConfig.Greedy), () => pcm.AddRange(d.TakeAudio()), default, out _);
            d.Complete();
            pcm.AddRange(d.TakeAudio());
            alone = pcm.ToArray();
            aloneFrames = d.FrameCodes.Count;
        }
        int[] textAlone;
        using (var engine = new InferenceEngine(model, Engine(), NullLogger.Instance))
            textAlone = (await Serve(engine, "t0", text, 10).WaitAsync(TimeSpan.FromMinutes(2))).Tokens;

        // On the engine, beside the text request.
        using (var engine = new InferenceEngine(model, Engine(), NullLogger.Instance))
        using (SpeechReplyDriver d = model.CreateSpeechDriver(prompt, new SpeechReplyOptions { Synthesis = true, Seed = 11 }))
        {
            Task<(int[] Tokens, string Finish)> speech = Serve(engine, "s", prompt, 64, d);
            var (textTokens, _) = await Serve(engine, "t1", text, 10).WaitAsync(TimeSpan.FromMinutes(2));
            var (frames, _) = await speech.WaitAsync(TimeSpan.FromMinutes(2));
            d.Complete();
            float[] served = d.TakeAudio();
            Assert.All(frames, t => Assert.Equal(tokens.AudioOut, t));
            Assert.Equal(textAlone, textTokens);
            Assert.True(aloneFrames > 0);
            if (backend == BackendType.Cpu)
            {
                // The managed executor's rows do not depend on the batch: the same codes and PCM.
                Assert.Equal(aloneFrames, d.FrameCodes.Count);
                Assert.Equal(alone, served);
            }
            else
            {
                Assert.True(served.Length > 0);
            }
        }
    }
}
