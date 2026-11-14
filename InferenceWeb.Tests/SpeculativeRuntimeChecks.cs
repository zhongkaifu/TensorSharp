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
    /// <param name="reference">The model plain greedy runs on (default: <paramref name="create"/>'s).</param>
    public static void AssertMatchesPlainGreedy(Func<ModelBase> create, string label, int maxNew = 24, double tieTolerance = 0,
        Func<ModelBase> reference = null)
    {
        int[] prompt = Enumerable.Range(0, 12).Select(i => 40 + i % 6).Concat(Enumerable.Range(0, 9).Select(i => 40 + i % 6)).ToArray();
        List<Greedy> plain = PlainGreedy(reference ?? create, prompt, maxNew);

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
    public static void AssertDraftHeadMatchesPlainGreedy(Func<ModelBase> create, string label, int maxNew = 24, double tieTolerance = 0,
        Func<ModelBase> reference = null)
    {
        int[] prompt = Enumerable.Range(0, 21).Select(i => 40 + (i * 7) % 23).ToArray();
        List<Greedy> plain = PlainGreedy(reference ?? create, prompt, maxNew);

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
    public static void AssertBlockDraftMatchesPlainGreedy(Func<ModelBase> create, string label, int maxNew = 24, double tieTolerance = 0,
        Func<ModelBase> reference = null)
    {
        int[] prompt = Enumerable.Range(0, 21).Select(i => 40 + (i * 7) % 23).ToArray();
        List<Greedy> plain = PlainGreedy(reference ?? create, prompt, maxNew);

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
        int[][] prompts = EnginePrompts;
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

    private static readonly int[][] EnginePrompts =
    {
        Enumerable.Range(0, 24).Select(i => 40 + i % 6).ToArray(),
        Enumerable.Range(0, 19).Select(i => 50 + (i * 5) % 9).ToArray(),
        Enumerable.Range(0, 27).Select(i => 44 + i % 4).ToArray(),
    };

    /// <summary>Several requests served at once through the engine's unified step (plain decoding) give the same streams
    /// from <paramref name="variant"/> as from <paramref name="baseline"/>.</summary>
    public static async Task AssertEngineStreamsMatch(Func<ModelBase> baseline, Func<ModelBase> variant, string label,
        int maxNew = 20)
    {
        int[][] expected = await ServeAsync(baseline, EnginePrompts, null, maxNew, verifySteps: null);
        int[][] actual = await ServeAsync(variant, EnginePrompts, null, maxNew, verifySteps: null);
        for (int r = 0; r < EnginePrompts.Length; r++)
            Assert.True(expected[r].SequenceEqual(actual[r]),
                $"{label}: request {r} gave [{string.Join(",", actual[r])}], expected [{string.Join(",", expected[r])}]");
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
        // Every request is admitted before the first step: submitted one by one, the engine sometimes took the first prompt
        // alone, so the two runs batched their prompts differently (a prompt step rounds by how many rows it holds) and a
        // near-tie flipped between them for reasons that were not the speculation's.
        var gate = new ComputeGate();
        gate.Close();
        engine.ComputeGate = gate;
        var handles = prompts.Select((p, i) => engine.SubmitRequest(
            new SequenceState($"req{i}", p, maxNew, cfg.BlockSize, SamplingConfig.Greedy))).ToList();
        gate.Open();
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

    /// <summary>
    /// A draft head the trunk keeps in step (<see cref="IDraftHead.DraftSelfCatchUp"/>) over a prompt read on the plain path
    /// in two chunks (<see cref="IDraftHead.FollowPlainPrompt"/> before each, the carry crossing between them) and the step
    /// that captures the carry after it: the first draft is the one the speculator's own catch-up over the same rows gives
    /// (every row's features to the host and back, token k paired with row k - 1, zeros before the first), and not the one
    /// a head restarted after the prompt gives (the drafts read the prompt).
    /// </summary>
    public static void AssertFollowedPromptDraftsAsItsCatchUp(Func<ModelBase> create, string label, double tolerance)
    {
        // Chunks wider than a verify window (StandardDecoderModel.SmallChunkRows): every model's trunk then reads them alike
        // (a narrower chunk through the speculative forward keeps the recurrent state after each row, another recurrence).
        int[] prompt = Enumerable.Range(0, 45).Select(i => 40 + (i * 13) % 150).ToArray();
        int n = prompt.Length, split = 20;

        // Followed: two plain prompt chunks, then the capture step; the null-row commits the speculator makes report no gap.
        using ModelBase followedModel = create();
        var followed = Assert.IsAssignableFrom<ISpeculativeModel>(followedModel);
        Assert.True(followed.DraftSelfCatchUp, $"{label}: the head does not follow the trunk");
        int feat = followed.SpecFeatureSize, vocab = followedModel.Config.VocabSize;
        followed.FollowPlainPrompt(null);
        followedModel.ForwardPrompt(prompt[..split], morePromptFollows: true);
        followed.FollowPlainPrompt(null);
        int next = ArgMax(followedModel.ForwardPrompt(prompt[split..], morePromptFollows: false));
        followed.DraftCatchUp(prompt, null, 0);
        var carry = new float[feat];
        var logits = new float[vocab];
        followed.SpecForward(new[] { next }, carry, logits, false);
        followed.DraftCatchUp(new[] { next }, null, n);
        int after = ArgMax(logits);
        var draftFollowed = new float[vocab];
        followed.DraftStep(after, carry, n + 1, draftFollowed, new float[feat]);

        // Caught up: every prompt row's features through the host (the same chunks: a chunk's split moves the trunk's
        // rounding).
        using ModelBase caughtModel = create();
        var caught = (ISpeculativeModel)caughtModel;
        var rows = new float[n * feat];
        var first = new float[split * feat];
        caught.SpecForward(prompt[..split], first, new float[vocab], false);
        var second = new float[(n - split) * feat];
        caught.SpecForward(prompt[split..], second, new float[vocab], false);
        Array.Copy(first, rows, first.Length);
        Array.Copy(second, 0, rows, first.Length, second.Length);
        var shifted = new float[n * feat];
        Array.Copy(rows, 0, shifted, feat, (n - 1) * feat);
        caught.DraftCatchUp(prompt, shifted, 0);
        var carry2 = new float[feat];
        caught.SpecForward(new[] { next }, carry2, new float[vocab], false);
        caught.DraftCatchUp(new[] { next }, rows[((n - 1) * feat)..], n);
        var draftCaught = new float[vocab];
        caught.DraftStep(after, carry2, n + 1, draftCaught, new float[feat]);

        // Restarted after the prompt (the head before it followed): the drafts do not read the prompt.
        using ModelBase restartedModel = create();
        var restarted = (ISpeculativeModel)restartedModel;
        restartedModel.ForwardPrompt(prompt[..split], morePromptFollows: true);
        restartedModel.ForwardPrompt(prompt[split..], morePromptFollows: false);
        restarted.DraftCatchUp(prompt, null, 0);
        var carry3 = new float[feat];
        restarted.SpecForward(new[] { next }, carry3, new float[vocab], false);
        restarted.DraftCatchUp(new[] { next }, null, n);
        var draftRestarted = new float[vocab];
        restarted.DraftStep(after, carry3, n + 1, draftRestarted, new float[feat]);

        // Followed on a per-request slot (the unified step a lone request takes on a model with slots): the prompt chunks,
        // then the one-row verify that captures the carry, the drafter's commits reporting no gap.
        float[] draftUnified = null;
        using (ModelBase unifiedModel = create())
        {
            if (unifiedModel is IUnifiedBatchModel unified && unifiedModel is IBatchedSpeculativeTarget batched)
            {
                var head = (ISpeculativeModel)unifiedModel;
                const string id = "followed";
                unified.EnsureSequenceSlot(id, n + 16);
                head.FollowPlainPrompt(id);
                Assert.True(unified.TryForwardUnified(new[] { id }, new[] { prompt[..split] }, new[] { 0 }, new float[1][]));
                head.FollowPlainPrompt(id);
                var last = new float[1][];
                Assert.True(unified.TryForwardUnified(new[] { id }, new[] { prompt[split..] }, new[] { split }, last));
                Assert.Equal(next, ArgMax(last[0]));
                using (batched.BeginBatchedSpeculation())
                {
                    batched.BindDraftState(id);
                    head.DraftCatchUp(prompt, null, 0);
                }
                var carry4 = new float[feat];
                var rowLogits = new float[vocab];
                Assert.True(batched.TryBatchedSpecVerifyLogits(new[] { id }, new[] { n }, 1, new[] { next }, rowLogits, carry4));
                batched.CommitBatchedSpecVerify(new[] { id }, new[] { n }, new[] { 1 });
                Assert.Equal(after, ArgMax(rowLogits));
                draftUnified = new float[vocab];
                using (batched.BeginBatchedSpeculation())
                {
                    batched.BindDraftState(id);
                    head.DraftCatchUp(new[] { next }, null, n);
                    head.DraftStep(after, carry4, n + 1, draftUnified, new float[feat]);
                }
            }
        }

        double followedVsCaught = RelL2(draftFollowed, draftCaught), restartedVsCaught = RelL2(draftRestarted, draftCaught);
        if (draftUnified != null)
        {
            double unifiedVsCaught = RelL2(draftUnified, draftCaught);
            Assert.True(unifiedVsCaught <= tolerance,
                $"{label}: the per-request slot's followed prompt drafts {unifiedVsCaught:E2} (relative L2) from the catch-up");
        }
        Assert.True(followedVsCaught <= tolerance,
            $"{label}: the followed prompt's first draft is {followedVsCaught:E2} (relative L2) from the catch-up's");
        Assert.True(restartedVsCaught > 10 * tolerance,
            $"{label}: a restarted head drafts within {restartedVsCaught:E2} of the caught-up one: the check cannot see the prompt");
    }

    private static double RelL2(float[] a, float[] b)
    {
        double num = 0, den = 0;
        for (int i = 0; i < a.Length; i++)
        {
            num += (a[i] - b[i]) * (double)(a[i] - b[i]);
            den += b[i] * (double)b[i];
        }
        return Math.Sqrt(num / Math.Max(den, 1e-30));
    }
}
