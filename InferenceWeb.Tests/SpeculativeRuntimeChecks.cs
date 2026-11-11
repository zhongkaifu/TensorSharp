// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>Speculative decoding checks shared by the runtime families.</summary>
internal static class SpeculativeRuntimeChecks
{
    /// <summary>Greedy speculation with n-gram drafts over a repetitive prompt (drafts both accepted and
    /// rejected) emits exactly the plain greedy tokens, and the verify path actually ran.</summary>
    /// <param name="tieTolerance">On a backend whose kernels round activations differently for a
    /// verify window than for one row (ggml_cuda, ggml_cpu), the streams may part only where plain
    /// greedy's top two logits are within this relative margin, and only to the runner-up; 0
    /// (exact backends) requires the streams to be equal.</param>
    public static void AssertMatchesPlainGreedy(Func<ModelBase> create, string label, int maxNew = 24, double tieTolerance = 0)
    {
        int[] prompt = Enumerable.Range(0, 12).Select(i => 40 + i % 6).Concat(Enumerable.Range(0, 9).Select(i => 40 + i % 6)).ToArray();
        List<Greedy> plain = PlainGreedy(create, prompt, maxNew);

        using ModelBase target = create();
        var spec = Assert.IsAssignableFrom<ISpeculativeTarget>(target);
        Assert.Null(spec.SpeculationRefusal);
        // Drafts from the known continuation, wrong at their third token every other time: verifies
        // are both fully and partly accepted, whatever the model's own repetition.
        var decoder = new SpeculativeDecoder(spec, new ScriptedSpeculator(prompt.Concat(plain.Select(g => g.Token)).ToArray(), target.Config.VocabSize));
        List<int> drafted = decoder.GenerateGreedy(prompt, maxNew);
        AssertSameStream(plain, drafted, tieTolerance, label);
        Assert.Equal(maxNew, drafted.Count);
        // The verify path ran, and some drafts were rejected (the prefix commit, not just a full accept).
        Assert.True(decoder.VerifySteps > 0, $"{label}: no verify ran");
        Assert.True(decoder.TokensAccepted > 0, $"{label}: no draft was accepted");
        Assert.True(decoder.TokensAccepted < decoder.TokensDrafted, $"{label}: every draft was accepted ({decoder.TokensAccepted})");
    }

    /// <summary>Greedy speculation through the model's own per-token draft head (NextN/MTP) emits
    /// exactly the plain greedy tokens: drafting, catch-up and the fused catch-up step ran.</summary>
    public static void AssertDraftHeadMatchesPlainGreedy(Func<ModelBase> create, string label, int maxNew = 24, double tieTolerance = 0)
    {
        int[] prompt = Enumerable.Range(0, 21).Select(i => 40 + (i * 7) % 23).ToArray();
        List<Greedy> plain = PlainGreedy(create, prompt, maxNew);

        using ModelBase target = create();
        var spec = Assert.IsAssignableFrom<ISpeculativeModel>(target);
        Assert.Null(spec.SpeculationRefusal);
        Assert.Equal(DraftHeadKind.PerToken, spec.DraftHeadKind);
        // Gate 0: every step drafts its whole window, whatever the (random) head's confidence.
        var speculator = new DraftHeadSpeculator(spec, target.Config.VocabSize, spec.SpecFeatureSize, 3) { MinDraftProb = 0 };
        var decoder = new SpeculativeDecoder(spec, speculator);
        List<int> drafted = decoder.GenerateGreedy(prompt, maxNew);
        AssertSameStream(plain, drafted, tieTolerance, label);
        Assert.Equal(maxNew, drafted.Count);
        Assert.True(decoder.VerifySteps > 0, $"{label}: no verify ran");
        Assert.True(decoder.TokensDrafted > 0, $"{label}: the head drafted nothing");
    }

    /// <summary>Greedy speculation through the model's block drafter (DFlash) emits exactly the plain greedy
    /// tokens: the prompt and every verify feed the drafter's feature rows, and its blocks are verified.</summary>
    public static void AssertBlockDraftMatchesPlainGreedy(Func<ModelBase> create, string label, int maxNew = 24, double tieTolerance = 0)
    {
        int[] prompt = Enumerable.Range(0, 21).Select(i => 40 + (i * 7) % 23).ToArray();
        List<Greedy> plain = PlainGreedy(create, prompt, maxNew);

        using ModelBase target = create();
        var spec = Assert.IsAssignableFrom<ISpeculativeModel>(target);
        Assert.Null(spec.SpeculationRefusal);
        Assert.Equal(DraftHeadKind.Block, spec.DraftHeadKind);
        // Gate 0 and no cost governor: every step drafts the whole block, whatever the (random) drafter's confidence.
        var speculator = new BlockDraftSpeculator(spec, spec.DraftBlockSize, spec.DraftBlockSize) { MinDraftProb = 0 };
        var decoder = new SpeculativeDecoder(spec, speculator) { AdaptiveSpeculation = false };
        List<int> drafted = decoder.GenerateGreedy(prompt, maxNew);
        AssertSameStream(plain, drafted, tieTolerance, label);
        Assert.Equal(maxNew, drafted.Count);
        Assert.True(decoder.VerifySteps > 0, $"{label}: no verify ran");
        Assert.True(decoder.TokensDrafted > 0, $"{label}: the drafter drafted nothing");
    }

    /// <summary>
    /// Speculation through the engine's unified step emits what plain decoding does, for one request and for several
    /// at once: each drafts over its own drafter state and every window is verified in one forward over the requests'
    /// slots (the batched verify must have run). Prompts repeat, so an n-gram drafter's drafts are accepted.
    /// </summary>
    public static async Task AssertEngineSpeculationMatchesPlain(Func<ModelBase> create, SpeculationOptions speculation,
        string label, int maxNew = 20)
    {
        int[][] prompts =
        {
            Enumerable.Range(0, 24).Select(i => 40 + i % 6).ToArray(),
            Enumerable.Range(0, 19).Select(i => 50 + (i * 5) % 9).ToArray(),
            Enumerable.Range(0, 27).Select(i => 44 + i % 4).ToArray(),
        };
        foreach (int concurrent in new[] { 1, prompts.Length })
        {
            int[][] chosen = prompts.Take(concurrent).ToArray();
            int[][] plain = await ServeAsync(create, chosen, null, maxNew, verifySteps: null);
            int verified = 0;
            int[][] drafted = await ServeAsync(create, chosen, speculation, maxNew, steps => verified = steps);
            for (int r = 0; r < chosen.Length; r++)
                Assert.True(plain[r].SequenceEqual(drafted[r]),
                    $"{label} x{concurrent}: request {r} gave [{string.Join(",", drafted[r])}], plain [{string.Join(",", plain[r])}]");
            Assert.True(verified > 0, $"{label} x{concurrent}: no batched verify ran");
        }
    }

    private static async Task<int[][]> ServeAsync(Func<ModelBase> create, int[][] prompts, SpeculationOptions speculation,
        int maxNew, Action<int> verifySteps)
    {
        using ModelBase model = create();
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 4,
            MaxPrefillChunkSize = 16,
            NumBlocks = 64,
            BlockSize = 16,
            EnablePrefixCaching = false,
            Speculation = speculation ?? new SpeculationOptions { Enabled = false },
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        var handles = prompts.Select((p, i) => engine.SubmitRequest(
            new SequenceState($"req{i}", p, maxNew, cfg.BlockSize, SamplingConfig.Greedy))).ToList();
        int[][] outputs = await Task.WhenAll(handles.Select(async h =>
        {
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync())
                tokens.Add(t);
            await h.Completion;
            return tokens.ToArray();
        })).WaitAsync(TimeSpan.FromMinutes(3));
        verifySteps?.Invoke(((StandardDecoderModel)model).BatchedSpecVerifySteps);
        return outputs;
    }

    private readonly record struct Greedy(int Token, int RunnerUp, double RelativeMargin);

    private static List<Greedy> PlainGreedy(Func<ModelBase> create, int[] prompt, int maxNew)
    {
        var plain = new List<Greedy>();
        using ModelBase model = create();
        float[] logits = model.ForwardRefill(prompt);
        for (int i = 0; i < maxNew; i++)
        {
            int best = ArgMax(logits), second = -1;
            for (int v = 0; v < logits.Length; v++)
                if (v != best && (second < 0 || logits[v] > logits[second])) second = v;
            plain.Add(new Greedy(best, second, (logits[best] - logits[second]) / Math.Max(Math.Abs(logits[best]), 1e-6)));
            logits = model.Forward(new[] { best });
        }
        return plain;
    }

    /// <summary>Equal streams, or (with a tie tolerance) equal up to a near-tie of plain greedy where the
    /// speculative stream took the runner-up; after that the two continue from different tokens.</summary>
    private static void AssertSameStream(List<Greedy> plain, List<int> drafted, double tieTolerance, string label)
    {
        for (int i = 0; i < Math.Min(plain.Count, drafted.Count); i++)
        {
            if (plain[i].Token == drafted[i]) continue;
            Assert.True(tieTolerance > 0 && plain[i].RelativeMargin <= tieTolerance && drafted[i] == plain[i].RunnerUp,
                $"{label}: token {i} is {drafted[i]}, plain greedy {plain[i].Token} (runner-up {plain[i].RunnerUp}, relative margin {plain[i].RelativeMargin:E2})");
            return;
        }
    }

    /// <summary>Proposes the next tokens of a known sequence, corrupting the third draft of every other
    /// proposal.</summary>
    private sealed class ScriptedSpeculator : ISpeculator
    {
        private readonly int[] _sequence;
        private readonly int _vocab;
        private int _calls;

        public ScriptedSpeculator(int[] sequence, int vocab)
        {
            _sequence = sequence;
            _vocab = vocab;
        }

        public string Name => "scripted";
        public int MaxDraftTokens => 4;
        public float MinDraftProb { get; set; }
        public float DefaultMinDraftProb => 0f;
        public bool NeedsHiddenState => false;
        public bool HandlesOwnPrefill => false;

        public int Propose(in DraftContext ctx, List<int> draftOut)
        {
            int next = ctx.Position < _sequence.Length && _sequence[ctx.Position] == ctx.LastToken ? ctx.Position + 1 : ctx.Position;
            int count = Math.Min(Math.Min(ctx.MaxTokens, MaxDraftTokens), _sequence.Length - next);
            for (int i = 0; i < count; i++) draftOut.Add(_sequence[next + i]);
            if (_calls++ % 2 == 1 && draftOut.Count >= 3)
                draftOut[2] = (draftOut[2] + 1) % _vocab;
            return draftOut.Count;
        }

        public void Commit(int[] tokens, float[]? hRows, int startPos) { }
        public void Reset() { }
        public void Dispose() { }
    }

    private static int ArgMax(float[] logits)
    {
        int best = 0;
        for (int v = 1; v < logits.Length; v++) if (logits[v] > logits[best]) best = v;
        return best;
    }
}
