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
using TensorSharp.Models.Lfm3;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using Xunit;

namespace InferenceWeb.Tests;

/// <summary>
/// The next turn after a spoken reply reuses the reply's cached state, on the synthetic LFM3 LM: a driven request whose
/// driver names its rows' content (<see cref="IOutputMediaDriver"/>, as LFM3-Omni's speech driver does) is keyed with a
/// media span over each run of its rows and retained when it ends, so a later prompt that carries the reply back - its
/// tokens, and the same spans over its placeholders - resumes from its end state through the whole reply. A driver that
/// does not name them is not retained; a later prompt whose rows differ stops reuse at them.
/// </summary>
public sealed class SpokenTurnPrefixReuseTests : IDisposable
{
    private readonly string _dir = Path.Combine(Path.GetTempPath(), "ts-spoken-reuse-" + Guid.NewGuid().ToString("N"));
    private readonly KvCacheDtype _dtype;
    private readonly bool _dtypeExplicit;

    public SpokenTurnPrefixReuseTests()
    {
        Directory.CreateDirectory(_dir);
        (_dtype, _dtypeExplicit) = (KvCacheDtypeConfig.Current, KvCacheDtypeConfig.IsExplicitlySet);
        KvCacheDtypeConfig.RestoreForTests(KvCacheDtype.F32, true);
    }

    public void Dispose()
    {
        KvCacheDtypeConfig.RestoreForTests(_dtype, _dtypeExplicit);
        try { Directory.Delete(_dir, recursive: true); } catch (IOException) { }
    }

    private const int Placeholder = 250;
    private const string Scope = "conversation-1";
    // Long enough that the prefix cache keeps what it makes (PrefixCacheCapabilities.MinRetainTokens).
    private static readonly int[] Prompt = Enumerable.Range(0, 40).Select(i => 40 + (i * 11) % 150).ToArray();

    /// <summary>'t' a text slot, 'r' an embedding row (a fixed row per slot), then a stop.</summary>
    private class RowDriver : ISequenceStepDriver
    {
        private readonly string _pattern;
        private int _at;

        public RowDriver(string pattern) => _pattern = pattern;

        public bool NextNeedsLogits => _at < _pattern.Length && _pattern[_at] == 't';
        public bool NextNeedsHidden => _at < _pattern.Length && _pattern[_at] == 'r';

        public DriverSlot Next(float[] logits, float[] hidden, Func<int> drawToken)
        {
            if (_at >= _pattern.Length) return DriverSlot.Stop("eos");
            int slot = _at++;
            if (_pattern[slot] == 't') return DriverSlot.Text(drawToken());
            return DriverSlot.Embedding(Placeholder, Row(slot, hidden.Length));
        }

        public void Dispose() { }
    }

    /// <summary>A <see cref="RowDriver"/> that names each run of its rows ("rows:start-end", its output positions).</summary>
    private sealed class NamingRowDriver : RowDriver, IOutputMediaDriver
    {
        public NamingRowDriver(string pattern) : base(pattern) { }

        public IReadOnlyList<PromptMediaSpan> OutputMediaSpans(IReadOnlyList<int> output, int outputStart, bool complete,
            out int keyedOutput)
        {
            var spans = RunsOf(output, outputStart, complete, out keyedOutput);
            return spans;
        }
    }

    /// <summary>The row slot <paramref name="slot"/> decodes: small fixed values, so its content is its slot number.</summary>
    private static float[] Row(int slot, int width) => Enumerable.Range(0, width).Select(i => (float)Math.Sin(slot * 0.7 + i * 0.13) * 0.1f).ToArray();

    private static List<PromptMediaSpan> RunsOf(IReadOnlyList<int> output, int outputStart, bool complete, out int keyed)
    {
        var spans = new List<PromptMediaSpan>();
        keyed = 0;
        for (int i = 0; i < output.Count;)
        {
            if (output[i] != Placeholder) { keyed = ++i; continue; }
            int start = i;
            while (i < output.Count && output[i] == Placeholder) i++;
            if (i == output.Count && !complete) break;
            spans.Add(new PromptMediaSpan(outputStart + start, outputStart + i, $"rows:{start}-{i}"));
            keyed = i;
        }
        return spans;
    }

    private static SchedulerConfig Engine() => new()
    {
        MaxNumBatchedTokens = 256, MaxNumRunningSequences = 4, MaxPrefillChunkSize = 64, NumBlocks = 64, BlockSize = 16,
        EnablePrefixCaching = true,
    };

    private string Lm() => new Lfm3SyntheticModelBuilder { WeightType = SyntheticGguf.GgmlType.F32 }
        .Write(Path.Combine(_dir, "lfm3.gguf"));

    private static async Task<(int[] Tokens, InferenceCompletion Completion)> Serve(InferenceEngine engine, string id, int[] prompt,
        int max, ISequenceStepDriver driver = null, IReadOnlyList<PromptMediaSpan> spans = null)
    {
        InferenceRequestHandle h = engine.SubmitRequest(new SequenceState(id, prompt, max, engine.PoolStats.blockSize,
            SamplingConfig.Greedy, mediaSpans: spans, cacheScope: Scope)
        {
            StepDriver = driver,
        });
        var tokens = new List<int>();
        await foreach (int t in h.Tokens.ReadAllAsync()) tokens.Add(t);
        InferenceCompletion c = await h.Completion;
        return (tokens.ToArray(), c);
    }

    /// <summary>Turn one spoken (a driven request), then turn two whose prompt carries it back: what turn two reused.</summary>
    private async Task<(int Reused, int TurnOneLength, int TurnTwoPrompt)> TwoTurns(bool namesRows, bool sameRows = true)
    {
        const string pattern = "ttrrrrtrrrt";
        using var model = new Lfm3RuntimeModel(Lm(), BackendType.Cpu);
        using var engine = new InferenceEngine(model, Engine(), NullLogger.Instance);
        RowDriver driver = namesRows ? new NamingRowDriver(pattern) : new RowDriver(pattern);
        var (output, first) = await Serve(engine, "turn-1", Prompt, 100, driver).WaitAsync(TimeSpan.FromMinutes(2));
        Assert.Equal("eos", first.FinishReason);
        Assert.Equal(pattern.Length, output.Length);

        // Turn two: turn one's prompt and reply (its placeholders under spans named by their rows), then more text.
        List<PromptMediaSpan> spans = RunsOf(output, Prompt.Length, complete: true, out _);
        if (!sameRows) spans[1] = spans[1] with { ContentId = "rows:other" };
        int[] next = Prompt.Concat(output).Concat(new[] { 61, 72, 83, 94, 105 }).ToArray();
        var (_, second) = await Serve(engine, "turn-2", next, 2, spans: spans).WaitAsync(TimeSpan.FromMinutes(2));
        return (second.PrefixCacheReusedTokens, Prompt.Length + output.Length, next.Length);
    }

    [Fact]
    public async Task TheNextTurnResumesFromASpokenReplyWhoseRowsAreNamed()
    {
        var (reused, turnOne, _) = await TwoTurns(namesRows: true);
        // Through the whole reply: its end state is retained and keyed as the next prompt keys it.
        Assert.Equal(turnOne, reused);
    }

    [Fact]
    public async Task ASpokenReplyWhoseRowsAreNotNamedIsNotResumed()
    {
        var (reused, turnOne, _) = await TwoTurns(namesRows: false);
        // At most a checkpoint inside turn one's prompt: nothing past its first row.
        Assert.True(reused <= Prompt.Length, $"reused {reused} of turn one's {turnOne}");
    }

    [Fact]
    public async Task RowsThatDifferStopReuseAtThem()
    {
        var (reused, turnOne, _) = await TwoTurns(namesRows: true, sameRows: false);
        Assert.True(reused < turnOne, $"reused {reused} of turn one's {turnOne} though its second run of rows differs");
    }

    [Fact]
    public void ASpeechDriverNamesEachRunOfItsFramesAsThePromptExpanderDoes()
    {
        int[][] frames = { Enumerable.Repeat(5, 12).ToArray(), Enumerable.Repeat(7, 12).ToArray(), Enumerable.Repeat(2016, 12).ToArray() };
        var driver = new FrameDriver(frames);
        int[] output = { 9, Placeholder, Placeholder, 10, Placeholder };
        IReadOnlyList<PromptMediaSpan> spans = driver.OutputMediaSpans(output, 100, complete: true, out int keyed);
        Assert.Equal(5, keyed);
        Assert.Equal(2, spans.Count);
        Assert.Equal(new PromptMediaSpan(101, 103, Lfm3SpeechOutput.FramesContentId(frames, 0, 2)), spans[0]);
        Assert.Equal(new PromptMediaSpan(104, 105, Lfm3SpeechOutput.FramesContentId(frames, 2, 1)), spans[1]);
        Assert.StartsWith("lfm3-speech:", spans[0].ContentId, StringComparison.Ordinal);

        // A run still open at the end of a reply in progress is not keyed yet.
        spans = driver.OutputMediaSpans(output, 100, complete: false, out keyed);
        Assert.Equal(4, keyed);
        Assert.Single(spans);
    }

    /// <summary>A speech reply driver whose fed frames are given, named as LFM3-Omni's are.</summary>
    private sealed class FrameDriver : TensorSharp.Models.Audio.SpeechReplyDriver
    {
        public FrameDriver(int[][] frames)
        {
            foreach (int[] f in frames) RecordFedFrame(f);
        }

        protected override string SpanContentId(IReadOnlyList<int[]> frames, int first, int count)
            => Lfm3SpeechOutput.FramesContentId(frames, first, count);

        public override int SampleRate => 24000;
        public override int FrameToken => Placeholder;
        public override bool IsControlToken(int token) => token == Placeholder;
        public override bool NextNeedsLogits => false;
        public override bool NextNeedsHidden => false;
        public override DriverSlot Next(float[] logits, float[] hidden, Func<int> drawToken) => DriverSlot.Stop();
        public override void Complete() { }
        public override void Dispose() { }
    }
}
