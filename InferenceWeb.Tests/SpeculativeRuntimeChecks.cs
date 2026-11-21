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
    /// <param name="prompt">The prompt (a short default): a long one is prefilled in one chunk by both arms (the plain one's
    /// refill and the speculative decoder's), so the two attend its rows alike.</param>
    public static void AssertBlockDraftMatchesPlainGreedy(Func<ModelBase> create, string label, int maxNew = 24, double tieTolerance = 0,
        Func<ModelBase> reference = null, int[] prompt = null)
    {
        prompt ??= Enumerable.Range(0, 21).Select(i => 40 + (i * 7) % 23).ToArray();
        List<Greedy> plain = PlainGreedy(reference ?? create, prompt, maxNew);

        using ModelBase target = create();
        var spec = Assert.IsAssignableFrom<ISpeculativeModel>(target);
        Assert.Null(spec.SpeculationRefusal);
        Assert.Equal(DraftHeadKind.Block, spec.DraftHeadKind);
        // Gate 0 and no cost governor: every step drafts the whole block, whatever the (random) drafter's confidence.
        var speculator = new BlockDraftSpeculator(spec, spec.DraftBlockSize, spec.DraftBlockSize) { MinDraftProb = 0 };
        var decoder = new SpeculativeDecoder(spec, speculator) { AdaptiveSpeculation = false, PrefillChunkSize = prompt.Length };
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
    /// <paramref name="prompts"/> replaces the default short ones.
    /// </summary>
    /// <param name="nearTie">0: every stream is plain decoding's, token for token (backends whose decode-size products give a
    /// row its bits whatever the rows beside it: the managed CPU, ggml's CPU, the CUDA executor). Otherwise a stream may part
    /// from plain decoding's only where the two tokens' logits are within this share of the largest logit's magnitude - its
    /// context's logits recomputed by one prompt forward over the tokens both streams agree on: on ggml's GPU kernels a
    /// verify window's rows, plain decoding's batched rows and a prompt's rows each take other products, which round apart
    /// in the last bits, and a synthetic checkpoint's near-tied logits then pick either token.</param>
    public static async Task AssertEngineSpeculationMatchesPlain(Func<ModelBase> create, SpeculationOptions speculation,
        string label, int maxNew = 20, int[][] prompts = null, double nearTie = 0)
    {
        prompts ??= EnginePrompts;
        foreach (int concurrent in new[] { 1, prompts.Length })
        {
            int[][] chosen = prompts.Take(concurrent).ToArray();
            int[][] plain = await ServeAsync(create, chosen, null, maxNew, verifySteps: null);
            int verified = 0;
            int[][] drafted = await ServeAsync(create, chosen, speculation, maxNew, steps => verified = steps);
            for (int r = 0; r < chosen.Length; r++)
            {
                if (plain[r].SequenceEqual(drafted[r])) continue;
                string streams = $"{label} x{concurrent}: request {r} gave [{string.Join(",", drafted[r])}], plain [{string.Join(",", plain[r])}]";
                Assert.True(nearTie > 0, streams);
                int d = 0;
                while (d < plain[r].Length && d < drafted[r].Length && plain[r][d] == drafted[r][d]) d++;
                Assert.True(d < plain[r].Length && d < drafted[r].Length, streams);
                using ModelBase reference = create();
                float[] logits = reference.ForwardPrompt(chosen[r].Concat(plain[r].Take(d)).ToArray());
                float largest = logits.Max(Math.Abs);
                float apart = Math.Abs(logits[plain[r][d]] - logits[drafted[r][d]]);
                Assert.True(apart <= nearTie * largest,
                    $"{streams}: they part at token {d}, whose logits are {apart:E2} apart ({apart / largest:E2} of the largest)");
            }
            Assert.True(verified > 0, $"{label} x{concurrent}: no batched verify ran");
        }
    }

    /// <summary>
    /// The first decode step after a prompt drafts, through the engine's unified step (one request, its prompt in 16-token
    /// chunks read on the plain path, the trunk keeping the drafter in step): the drafter hands the step the carry it took
    /// as the trunk forwarded the prompt's last row (<see cref="IDraftHead.TryTakeHeldCarry"/>), where the context, armed
    /// without one, had decoded that step plainly. Two new tokens leave room for that step alone (a window of one draft; an
    /// anchor drawn after a rejected draft ends the request without a forward). The speculating streams, of those two tokens
    /// and of <paramref name="maxNew"/>, are plain decoding's: the prompt's steps are the plain ones.
    /// </summary>
    public static async Task AssertFirstDecodeStepDrafts(Func<ModelBase> create, SpeculationOptions speculation, string label,
        int maxNew = 20)
    {
        int[][] prompt = { EnginePrompts[0] };
        int[][] plain = await ServeAsync(create, prompt, null, maxNew, verifySteps: null);
        ModelBase last = null;
        var served = new List<SequenceState>();
        int[][] first = await ServeAsync(() => last = create(), prompt, speculation, 2, verifySteps: null, served);
        Assert.True(plain[0].Take(2).SequenceEqual(first[0]), $"{label}: gave [{string.Join(",", first[0])}], plain [{string.Join(",", plain[0])}]");
        SpeculationStats stats = Assert.Single(served).SpecStats;
        Assert.True(stats != null, $"{label}: speculation did not arm");
        Assert.True(stats.TokensDrafted >= 1 && stats.VerifySteps == 1 && stats.PlainSteps == 0,
            $"{label}: the first decode step drafted {stats.TokensDrafted} (verify steps {stats.VerifySteps}, plain steps {stats.PlainSteps})");
        if (last is StandardDecoderModel runtime)
            Assert.True(runtime.HeldCarriesTaken > 0, $"{label}: the first draft did not chain from the carry the drafter held");
        int[][] drafted = await ServeAsync(create, prompt, speculation, maxNew, verifySteps: null);
        Assert.True(plain[0].SequenceEqual(drafted[0]), $"{label}: gave [{string.Join(",", drafted[0])}], plain [{string.Join(",", plain[0])}]");
    }

    /// <summary>
    /// The same on the per-sequence route (a model without per-request slots): a prompt read plainly in 16-token chunks, the
    /// trunk keeping the drafter in step (<see cref="IDraftHead.FollowPlainPrompt(string, int)"/> before each chunk), then a
    /// context armed over the tokens the trunk holds (<see cref="SpeculativeExecution.SeedCommitted"/>, as the executor
    /// arms it at the first decode step): the carry the drafter held is the prompt's last feature row (within
    /// <paramref name="tolerance"/>, relative L2, of the row a capturing step returns for the same chunks), and that step
    /// drafts from it.
    /// </summary>
    public static void AssertPlainPromptCarriesIntoTheFirstDraft(Func<ModelBase> create, string label, double tolerance = 1e-2)
    {
        int[] prompt = EnginePrompts[0];
        int n = prompt.Length;
        using ModelBase model = create();
        var spec = Assert.IsAssignableFrom<ISpeculativeModel>(model);
        Assert.True(spec.DraftSelfCatchUp, $"{label}: the trunk does not keep the drafter in step");
        int feat = spec.SpecFeatureSize, vocab = model.Config.VocabSize;
        // Gate 0: the head drafts whatever its (random) confidence.
        ISpeculator speculator = spec.DraftHeadKind == DraftHeadKind.Block
            ? new BlockDraftSpeculator(spec, spec.DraftBlockSize, spec.DraftBlockSize) { MinDraftProb = 0 }
            : new DraftHeadSpeculator(spec, vocab, feat, 3) { MinDraftProb = 0 };
        using var execution = new SpeculativeExecution(spec, speculator) { AdaptiveSpeculation = false };
        float[] logits = null;
        for (int start = 0; start < n; start += 16)
        {
            int end = Math.Min(n, start + 16);
            spec.FollowPlainPrompt(null, n);
            logits = model.ForwardPrompt(prompt[start..end], morePromptFollows: end < n);
        }

        // The held carry is the prompt's last feature row: a capturing step over the same chunks on a twin returns it.
        var held = new float[feat];
        Assert.True(spec.TryTakeHeldCarry(n, held), $"{label}: the drafter holds no carry after the prompt");
        using (ModelBase twin = create())
        {
            var reference = (ISpeculativeModel)twin;
            float[] rows = null;
            for (int start = 0; start < n; start += 16)
            {
                int end = Math.Min(n, start + 16);
                rows = new float[(long)(end - start) * feat];
                reference.SpecForward(prompt[start..end], rows, new float[vocab], allLogitsRows: false);
            }
            double rel = RelL2(held, rows[^feat..]);
            Assert.True(rel <= tolerance, $"{label}: the held carry is {rel:E2} (relative L2) from the prompt's last feature row");
        }

        execution.SeedCommitted(prompt);
        SpeculativeStepOutcome outcome = execution.DecodeStep(ArgMax(logits), n, 3, ArgMax);
        Assert.True(outcome.UsedSpeculation && execution.Stats.TokensDrafted > 0,
            $"{label}: the first decode step after the prompt drafted {execution.Stats.TokensDrafted}");
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

    /// <summary>
    /// Several requests speculating through a per-token draft head at once draft in one pass a step
    /// (<see cref="IBatchedSpeculativeTarget.TryDraftChainBatch"/>): their streams are plain decoding's and those of the
    /// same requests drafting one at a time (the pass disabled), and each request drafts and keeps the drafts it does
    /// drafting alone. <paramref name="create"/> makes a model whose head chains (<see cref="IDraftHead.SupportsDraftChain"/>).
    /// </summary>
    public static async Task AssertBatchedDraftChainsDraftAsEachRequest(Func<StandardDecoderModel> create,
        SpeculationOptions speculation, string label, int maxNew = 20)
    {
        int[][] plain = await ServeAsync(create, EnginePrompts, null, maxNew, verifySteps: null);
        int chains = 0, verified = 0, alone = -1;
        var together = new List<SequenceState>();
        int[][] batched = await ServeAsync(create, EnginePrompts, speculation, maxNew, steps => verified = steps, together,
            model => chains = model.BatchedDraftChains);
        var single = new List<SequenceState>();
        int[][] perRequest = await ServeAsync(() =>
        {
            StandardDecoderModel model = create();
            model.BatchedDraftChainDisabledForTests = true;
            return model;
        }, EnginePrompts, speculation, maxNew, verifySteps: null, single, model => alone = model.BatchedDraftChains);
        for (int r = 0; r < EnginePrompts.Length; r++)
        {
            Assert.True(plain[r].SequenceEqual(batched[r]),
                $"{label}: request {r} gave [{string.Join(",", batched[r])}], plain [{string.Join(",", plain[r])}]");
            Assert.True(perRequest[r].SequenceEqual(batched[r]),
                $"{label}: request {r} gave [{string.Join(",", batched[r])}] chained together, [{string.Join(",", perRequest[r])}] alone");
        }
        Assert.True(verified > 0, $"{label}: no batched verify ran");
        Assert.True(chains > 0, $"{label}: no step chained its requests' draft heads together");
        Assert.Equal(0, alone);
        for (int r = 0; r < EnginePrompts.Length; r++)
        {
            SpeculationStats a = together[r].SpecStats, b = single[r].SpecStats;
            Assert.True(a != null && b != null, $"{label}: request {r} did not speculate");
            Assert.True(a.TokensDrafted == b.TokensDrafted && a.TokensAccepted == b.TokensAccepted,
                $"{label}: request {r} drafted {a.TokensDrafted} (accepted {a.TokensAccepted}) chained together, "
                + $"{b.TokensDrafted} ({b.TokensAccepted}) alone");
        }
    }

    /// <summary>
    /// Several requests speculating through a block drafter at once draft their blocks in one program a step
    /// (<see cref="IBatchedSpeculativeTarget.TryDraftBatch"/>) and write their commits' catch-ups in one program when the step
    /// ends: their streams are plain decoding's and those of the same requests drafting one at a time (the program disabled,
    /// as before it), and each request drafts and keeps the drafts it does drafting alone. With <paramref name="nearTie"/>
    /// above 0 (a backend whose products round by their row count: a program over three requests' blocks reads every weight
    /// for 3 x b rows where a lone block reads it for b), a stream may part from plain decoding's only at a near-tie of its
    /// context's logits (<see cref="AssertEngineSpeculationMatchesPlain"/>'s rule) and the draft counts may differ.
    /// TODO: once SpeculativeVerifyWindows (feature/muse-glimmer-unified) is on main, prove each parting from the run's own
    /// verify row as the DFlash lossless gates there do.
    /// </summary>
    public static async Task AssertBatchedBlockDraftsDraftAsEachRequest(Func<StandardDecoderModel> create,
        SpeculationOptions speculation, string label, int maxNew = 20, double nearTie = 0, BatchedRings rings = BatchedRings.Kept)
    {
        int[][] plain = await ServeAsync(create, EnginePrompts, null, maxNew, verifySteps: null);
        int blocks = 0, injects = 0, verified = 0, alone = -1, follows = 0, featureCommits = 0, ringCarries = 0;
        bool keeps = false;
        string declined = null;
        var together = new List<SequenceState>();
        int[][] batched = await ServeAsync(() =>
        {
            StandardDecoderModel model = create();
            model.VerifyFollowDisabledForTests = rings == BatchedRings.CatchUps;
            model.VerifyFollowWritesFromFeaturesForTests = rings == BatchedRings.FromFeatures;
            return model;
        }, EnginePrompts, speculation, maxNew, steps => verified = steps, together,
            model => (blocks, injects, declined, follows, featureCommits, ringCarries, keeps) = (model.BatchedBlockDrafts, model.DFlashLaneInjects,
                model.BatchedBlockDraftDeclineReason, model.VerifyFollows, model.VerifyFeatureCommits, model.RingHeldCarries,
                model.BatchedSpecVerifyKeepsDrafter));
        var single = new List<SequenceState>();
        int[][] perRequest = await ServeAsync(() =>
        {
            StandardDecoderModel model = create();
            model.BatchedBlockDraftDisabledForTests = true;
            return model;
        }, EnginePrompts, speculation, maxNew, verifySteps: null, single, model => alone = model.BatchedBlockDrafts + model.DFlashLaneInjects);
        Assert.Null(declined);
        Assert.True(verified > 0, $"{label}: no batched verify ran");
        Assert.True(blocks > 0, $"{label}: no step drafted its requests' blocks together");
        // Where the batched verify keeps the rings itself, its program (or its commit, from the features it returned to the
        // model) wrote the kept rows and the drafts took their carries from the rings; else the catch-ups went in together.
        if (keeps && rings == BatchedRings.Kept)
            Assert.True(follows > 0 && featureCommits == 0, $"{label}: {follows} verifies wrote the rings, {featureCommits} commits from features");
        else if (keeps && rings == BatchedRings.FromFeatures)
            Assert.True(featureCommits > 0 && follows == 0, $"{label}: {featureCommits} commits wrote the rings from features, {follows} verifies");
        else
            Assert.True(injects > 0, $"{label}: no step wrote its requests' catch-ups together");
        if (keeps && rings != BatchedRings.CatchUps)
            Assert.True(ringCarries > 0, $"{label}: no draft took its carry from its ring");
        Assert.Equal(0, alone);
        for (int r = 0; r < EnginePrompts.Length; r++)
        {
            AssertPartsOnlyAtANearTie(create, EnginePrompts[r], plain[r], batched[r], nearTie, $"{label}: request {r} drafted together");
            AssertPartsOnlyAtANearTie(create, EnginePrompts[r], plain[r], perRequest[r], nearTie, $"{label}: request {r} drafted alone");
            if (nearTie > 0) continue;
            SpeculationStats a = together[r].SpecStats, b = single[r].SpecStats;
            Assert.True(a != null && b != null, $"{label}: request {r} did not speculate");
            Assert.True(a.TokensDrafted == b.TokensDrafted && a.TokensAccepted == b.TokensAccepted,
                $"{label}: request {r} drafted {a.TokensDrafted} (accepted {a.TokensAccepted}) together, "
                + $"{b.TokensDrafted} ({b.TokensAccepted}) alone");
        }
    }

    /// <summary>How a batched step's commits reach the drafters' rings in <see cref="AssertBatchedBlockDraftsDraftAsEachRequest"/>.</summary>
    public enum BatchedRings
    {
        /// <summary>As the model chooses: the verify writes the windows into the rings where it keeps them.</summary>
        Kept,
        /// <summary>The verify returns its features to the model, and the commit writes the kept rows from them.</summary>
        FromFeatures,
        /// <summary>The verify returns its features to the caller, whose contexts' catch-ups go in together.</summary>
        CatchUps,
    }

    /// <summary><paramref name="actual"/> is <paramref name="plain"/>, or (with <paramref name="nearTie"/> above 0) parts from it
    /// only where the two tokens' logits are within that share of the largest logit's magnitude, the context's logits from
    /// one prompt forward over the tokens both agree on.</summary>
    private static void AssertPartsOnlyAtANearTie(Func<StandardDecoderModel> create, int[] prompt, int[] plain, int[] actual,
        double nearTie, string label)
    {
        if (plain.SequenceEqual(actual)) return;
        string streams = $"{label} gave [{string.Join(",", actual)}], plain [{string.Join(",", plain)}]";
        Assert.True(nearTie > 0, streams);
        int d = 0;
        while (d < plain.Length && d < actual.Length && plain[d] == actual[d]) d++;
        Assert.True(d < plain.Length && d < actual.Length, streams);
        using ModelBase reference = create();
        float[] logits = reference.ForwardPrompt(prompt.Concat(plain.Take(d)).ToArray());
        float largest = logits.Max(Math.Abs);
        float apart = Math.Abs(logits[plain[d]] - logits[actual[d]]);
        Assert.True(apart <= nearTie * largest,
            $"{streams}: they part at token {d}, whose logits are {apart:E2} apart ({apart / largest:E2} of the largest)");
    }

    /// <summary>
    /// The draft chains of several requests in one pass (<see cref="IBatchedSpeculativeTarget.TryDraftChainBatch"/>) are
    /// each request's own chain (<see cref="IDraftHead.DraftChain"/> on its bound slot): four requests at different
    /// positions folding 1, 3, 2 and 2 verified rows in front of their anchors, with different caps, the last one's draft
    /// layer restarted after a gap (attending only its rows past it), give the same ids (their confidences within
    /// <paramref name="tolerance"/>); a gate stops each lane where its own chain stops; a pass of the unfloored lanes alone
    /// gives their drafts again; and each lane's slot is left as its own chain leaves it: a following draft of each gives
    /// the same ids after either. A pass naming a request without a slot declines without running.
    /// </summary>
    public static void AssertBatchedDraftChainDraftsEachLane(Func<StandardDecoderModel> create, string label, double tolerance)
    {
        int[][] prompts =
        {
            Enumerable.Range(0, 17).Select(i => 40 + (i * 11) % 150).ToArray(),
            Enumerable.Range(0, 23).Select(i => 33 + (i * 7) % 90).ToArray(),
            Enumerable.Range(0, 12).Select(i => 60 + (i * 5) % 70).ToArray(),
            Enumerable.Range(0, 15).Select(i => 50 + (i * 13) % 80).ToArray(),
        };
        int[] folds = { 1, 3, 2, 2 };
        int[] caps = { 3, 2, 3, 2 };
        bool[] gapped = { false, false, false, true };
        int lanes = prompts.Length;
        int[] all = Enumerable.Range(0, lanes).ToArray(), unfloored = { 0, 1, 2 };
        string[] ids = Enumerable.Range(0, lanes).Select(r => $"lane{r}").ToArray();

        using StandardDecoderModel together = create(), alone = create();
        var lanesTogether = Prepare(together);
        var lanesAlone = Prepare(alone);
        for (int r = 0; r < lanes; r++)
        {
            Assert.Equal(lanesAlone[r].Tokens, lanesTogether[r].Tokens);
            Assert.True(together.CanBatchDraftChain(ids[r]), $"{label}: lane {r} cannot join a batched chain");
        }
        Assert.False(together.CanBatchDraftChain("absent"), $"{label}: a request without a slot joined a batched chain");
        Assert.False(Together(new[] { 0, -1 }, 0f, out _, out _, out _), $"{label}: a pass with a request without a slot ran");
        Assert.Equal(0, together.BatchedDraftChains);

        // Gate 0: every lane drafts its whole cap.
        Assert.True(Together(all, 0f, out int[][] tIds, out float[][] tProbs, out int[] tSteps), $"{label}: the pass declined");
        Alone(0f, out int[][] aIds, out float[][] aProbs, out int[] aSteps);
        AssertSame("gate 0", all, tIds, tProbs, tSteps, aIds, aProbs, aSteps);
        Assert.Equal(caps, tSteps);
        // The unfloored lanes again on their own (one floor for the step; the rows past each anchor rewritten before any
        // row reads them): the same drafts.
        Assert.True(Together(unfloored, 0f, out int[][] again, out float[][] againProbs, out int[] againSteps));
        AssertSame("unfloored again", unfloored, again, againProbs, againSteps, aIds, aProbs, aSteps);

        // A gate above lane 0's second confidence, halfway to the next one any lane drafts with (so neither run's rounding
        // moves a confidence across it): each lane stops after its first step below it, as its own chain does.
        float second = aProbs[0][1];
        float above = aProbs.SelectMany(p => p).Where(p => p > second).DefaultIfEmpty(1f).Min();
        float gate = (second + above) / 2;
        Assert.True(Together(all, gate, out tIds, out tProbs, out tSteps), $"{label}: the gated pass declined");
        Alone(gate, out aIds, out aProbs, out aSteps);
        AssertSame("gated", all, tIds, tProbs, tSteps, aIds, aProbs, aSteps);
        Assert.True(tSteps[0] <= 2, $"{label}: lane 0 drafted {tSteps[0]} steps past a gate above its second confidence");

        // Each lane's slot holds what its own chain left: a following draft over [anchor, first draft] at the anchor (the
        // verified run the next fold replays) reads the rows each pass wrote, and drafts the same.
        int feat = ((ISpeculativeModel)together).SpecFeatureSize;
        for (int r = 0; r < lanes; r++)
        {
            int[] next = { lanesTogether[r].Tokens[^1], aIds[r][0] };
            int rowsIn = lanesTogether[r].Tokens.Length;
            float[] lastRow = lanesTogether[r].Rows[((rowsIn - 1) * feat)..(rowsIn * feat)];
            float[] rows = lastRow.Concat(lastRow).ToArray();
            int anchorAt = lanesTogether[r].Start + rowsIn - 1;
            (int[] I, float[] P, int S) t = Chain(together, ids[r], next, rows, anchorAt, 2, 0f);
            (int[] I, float[] P, int S) a = Chain(alone, ids[r], next, rows, anchorAt, 2, 0f);
            Assert.True(t.I.SequenceEqual(a.I),
                $"{label}: lane {r} drafted [{string.Join(",", t.I)}] after the passes, [{string.Join(",", a.I)}] after its own chains");
            for (int k = 0; k < t.S; k++)
                Assert.True(Math.Abs(t.P[k] - a.P[k]) <= tolerance,
                    $"{label}: lane {r} step {k} confidence {t.P[k]} after the passes, {a.P[k]} after its own chains");
        }
        Assert.Equal(3, together.BatchedDraftChains);
        return;

        (int[] Tokens, float[] Rows, int Start)[] Prepare(StandardDecoderModel model)
        {
            var head = (ISpeculativeModel)model;
            int width = head.SpecFeatureSize, vocab = model.Config.VocabSize;
            var prepared = new (int[] Tokens, float[] Rows, int Start)[lanes];
            for (int r = 0; r < lanes; r++)
            {
                // The trunk holds the prompt; the head holds the rows before the lane's fold (or, after a gap, none: it attends
                // from the fold on), whose verified tokens (all but the anchor) it runs again in front of the anchor, as the
                // speculator's stash folds into its next draft.
                model.BindSequenceCache(ids[r]);
                int[] prompt = prompts[r];
                int n = prompt.Length, fold = folds[r], caught = n - fold + 1;
                var hAll = new float[(long)n * width];
                var logits = new float[vocab];
                head.SpecForward(prompt, hAll, logits, false);
                head.DraftCatchUp(prompt[..caught], gapped[r] ? null : new float[width].Concat(hAll[..((caught - 1) * width)]).ToArray(), 0);
                int[] tokens = prompt[caught..].Append(ArgMax(logits)).ToArray();
                prepared[r] = (tokens, hAll[((caught - 1) * width)..], caught);
            }
            model.RestorePrimaryCache();
            return prepared;
        }

        bool Together(int[] which, float stopBelow, out int[][] outIds, out float[][] outProbs, out int[] outSteps)
        {
            int m = which.Length;
            outIds = new int[lanes][];
            outProbs = new float[lanes][];
            outSteps = new int[lanes];
            var names = new string[m];
            var tokens = new int[m][];
            var rows = new float[m][];
            var starts = new int[m];
            var maxTokens = new int[m];
            var laneIds = new int[m][];
            var laneProbs = new float[m][];
            var laneSteps = new int[m];
            for (int j = 0; j < m; j++)
            {
                int r = Math.Max(0, which[j]);
                names[j] = which[j] < 0 ? "absent" : ids[r];
                tokens[j] = lanesTogether[r].Tokens;
                rows[j] = lanesTogether[r].Rows;
                starts[j] = lanesTogether[r].Start;
                maxTokens[j] = caps[r];
                laneIds[j] = new int[caps[r]];
                laneProbs[j] = new float[caps[r]];
            }
            if (!together.TryDraftChainBatch(names, tokens, rows, starts, maxTokens, Enumerable.Repeat(stopBelow, m).ToArray(),
                    laneIds, laneProbs, laneSteps))
                return false;
            for (int j = 0; j < m; j++)
                (outIds[which[j]], outProbs[which[j]], outSteps[which[j]]) = (laneIds[j], laneProbs[j], laneSteps[j]);
            return true;
        }

        void Alone(float stopBelow, out int[][] outIds, out float[][] outProbs, out int[] outSteps)
        {
            outIds = new int[lanes][];
            outProbs = new float[lanes][];
            outSteps = new int[lanes];
            for (int r = 0; r < lanes; r++)
                (outIds[r], outProbs[r], outSteps[r]) = Chain(alone, ids[r], lanesAlone[r].Tokens, lanesAlone[r].Rows, lanesAlone[r].Start,
                    caps[r], stopBelow);
        }

        void AssertSame(string what, int[] which, int[][] tI, float[][] tP, int[] tS, int[][] aI, float[][] aP, int[] aS)
        {
            foreach (int r in which)
            {
                Assert.True(tS[r] == aS[r], $"{label} {what}: lane {r} drafted {tS[r]} steps in the pass, {aS[r]} alone");
                Assert.True(tI[r].Take(tS[r]).SequenceEqual(aI[r].Take(aS[r])),
                    $"{label} {what}: lane {r} drafted [{string.Join(",", tI[r].Take(tS[r]))}] in the pass, [{string.Join(",", aI[r].Take(aS[r]))}] alone");
                for (int k = 0; k < tS[r]; k++)
                    Assert.True(Math.Abs(tP[r][k] - aP[r][k]) <= tolerance,
                        $"{label} {what}: lane {r} step {k} confidence {tP[r][k]} in the pass, {aP[r][k]} alone");
            }
        }
    }

    /// <summary><paramref name="requestId"/>'s draft chain on its bound slot (<see cref="IDraftHead.DraftChain"/>).</summary>
    private static (int[] Ids, float[] Probs, int Steps) Chain(StandardDecoderModel model, string requestId, int[] tokens, float[] rows,
        int start, int maxTokens, float stopBelow)
    {
        var ids = new int[maxTokens];
        var probs = new float[maxTokens];
        model.BindSequenceCache(requestId);
        try
        {
            int steps = ((IDraftHead)model).DraftChain(tokens, rows, start, maxTokens, ids, probs, stopBelow);
            return (ids, probs, steps);
        }
        finally
        {
            model.RestorePrimaryCache();
        }
    }

    /// <summary>
    /// <c>ignore_eos</c> requests speculating together run through end-of-sequence draws exactly as plain decoding does: an
    /// EOS anchor or accepted draft of one lane is verified like any token, the lanes end at their length limit, and no
    /// co-batched request fails. The batched verify used to end a lane at an EOS without forwarding it, which left that lane
    /// no logits for its next step; the step's failure then failed every request it held. Served with
    /// <paramref name="speculation"/> (n-gram: whatever it drafts) and with a drafter that proposes each request's known
    /// continuation (every other window wrong at its third draft), so EOS drafts are accepted and rejected too.
    /// <paramref name="createWithEos"/> creates the model with the given end-of-sequence id (its weights independent of
    /// it): the token both plain streams draw most often becomes the EOS, so the lanes meet it as anchors and as drafts.
    /// </summary>
    public static async Task AssertIgnoreEosSpeculationMatchesPlain(Func<int, ModelBase> createWithEos, int defaultEos,
        SpeculationOptions speculation, string label, int maxNew = 32)
    {
        int[][] prompts = EnginePrompts.Take(2).ToArray();
        static SamplingConfig IgnoreEos()
        {
            // As the request parser sets it: only the length limit ends such a request.
            SamplingConfig sampling = SamplingConfig.Greedy;
            sampling.IgnoreEos = true;
            sampling.StopRepetition = false;
            return sampling;
        }
        int[][] probe = await ServeAsync(() => createWithEos(defaultEos), prompts, null, maxNew, verifySteps: null,
            sampling: IgnoreEos);
        int eos = probe[0].Intersect(probe[1])
            .OrderByDescending(t => probe[0].Count(x => x == t) + probe[1].Count(x => x == t)).ThenBy(t => t)
            .DefaultIfEmpty(-1).First();
        Assert.True(eos >= 0, $"{label}: the two plain streams share no token to make the EOS");
        ModelBase create() => createWithEos(eos);

        int[][] full = prompts.Select((p, r) => p.Concat(probe[r]).ToArray()).ToArray();
        string oracle = "ignore-eos-oracle-" + Guid.NewGuid().ToString("N");
        SpeculatorRegistry.Register(oracle, (_, _) => new ContinuationSpeculator(full), requiresDraftHead: false);
        var drafters = new[]
        {
            (Name: speculation.SpeculatorName, Options: speculation),
            (Name: "known continuation", Options: new SpeculationOptions { Enabled = true, SpeculatorName = oracle, MaxDraftTokens = 3, MinDraftProb = 0 }),
        };
        foreach (int concurrent in new[] { prompts.Length, 1 })
        {
            int[][] chosen = prompts.Take(concurrent).ToArray();
            int[][] plain = await ServeAsync(create, chosen, null, maxNew, verifySteps: null, sampling: IgnoreEos);
            foreach (var drafter in drafters)
            {
                int verified = 0;
                var done = new List<InferenceCompletion>();
                var served = new List<SequenceState>();
                int[][] drafted = await ServeAsync(create, chosen, drafter.Options, maxNew, steps => verified = steps, served,
                    sampling: IgnoreEos, completions: done);
                for (int r = 0; r < chosen.Length; r++)
                {
                    string streams = $"{label} {drafter.Name} x{concurrent}: request {r} gave [{string.Join(",", drafted[r])}], plain [{string.Join(",", plain[r])}]";
                    Assert.True(plain[r].Length == maxNew && plain[r].Contains(eos), $"{streams}: the plain stream draws no EOS ({eos})");
                    Assert.True(plain[r].SequenceEqual(drafted[r]), streams);
                    Assert.Equal(SequenceStatus.FinishedLengthCapped, done[r].Status);
                    Assert.Equal(maxNew, done[r].OutputTokenCount);
                }
                Assert.True(verified > 0, $"{label} {drafter.Name} x{concurrent}: no batched verify ran");
                if (drafter.Options.SpeculatorName == oracle)
                {
                    // The known continuation's windows hold EOS tokens: accepted ones, and so a verify forwarded past them.
                    long accepted = served.Sum(s => s.SpecStats?.TokensAccepted ?? 0);
                    long rejected = served.Sum(s => (s.SpecStats?.TokensDrafted ?? 0) - (s.SpecStats?.TokensAccepted ?? 0));
                    Assert.True(accepted >= maxNew / 2 && rejected > 0,
                        $"{label} {drafter.Name} x{concurrent}: {accepted} drafts accepted, {rejected} rejected");
                }
            }
        }
    }

    /// <summary>Drafts the known continuation of whichever request's tokens it has committed (as the engine arms one drafter
    /// per request), wrong at the third draft of every other window.</summary>
    private sealed class ContinuationSpeculator : ISpeculator
    {
        private readonly int[][] _sequences;
        private readonly List<int> _committed = new();
        private int _calls;

        public ContinuationSpeculator(int[][] sequences) => _sequences = sequences;

        public string Name => "known-continuation";
        public int MaxDraftTokens => 3;
        public float MinDraftProb { get; set; }
        public float DefaultMinDraftProb => 0f;
        public bool NeedsHiddenState => false;
        public bool HandlesOwnPrefill => false;

        public int Propose(in DraftContext ctx, List<int> draftOut)
        {
            int position = ctx.Position, last = ctx.LastToken;
            int[]? sequence = _sequences.FirstOrDefault(q => q.Length > position && q[position] == last
                && _committed.Count >= position && q.AsSpan(0, position).SequenceEqual(System.Runtime.InteropServices.CollectionsMarshal.AsSpan(_committed)[..position]));
            if (sequence == null)
                return 0;
            int count = Math.Min(ctx.MaxTokens, sequence.Length - position - 1);
            for (int i = 0; i < count; i++)
                draftOut.Add(sequence[position + 1 + i]);
            if (_calls++ % 2 == 1 && draftOut.Count >= 3)
                draftOut[2] = sequence[position + 3] == 0 ? 1 : 0;
            return draftOut.Count;
        }

        public void Commit(int[] tokens, float[]? hRows, int startPos)
        {
            if (_committed.Count > startPos)
                _committed.RemoveRange(startPos, _committed.Count - startPos);
            while (_committed.Count < startPos)
                _committed.Add(-1);
            _committed.AddRange(tokens);
        }

        public void Reset() => _committed.Clear();
        public void Dispose() { }
    }

    private static async Task<int[][]> ServeAsync(Func<ModelBase> create, int[][] prompts, SpeculationOptions speculation,
        int maxNew, Action<int> verifySteps, List<SequenceState> served = null, Action<StandardDecoderModel> inspect = null,
        Func<SamplingConfig> sampling = null, List<InferenceCompletion> completions = null)
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
        var sequences = prompts.Select((p, i) => new SequenceState($"req{i}", p, maxNew, cfg.BlockSize,
            sampling?.Invoke() ?? SamplingConfig.Greedy)).ToList();
        served?.AddRange(sequences);
        var handles = sequences.Select(s => engine.SubmitRequest(s)).ToList();
        gate.Open();
        var finished = new InferenceCompletion[handles.Count];
        int[][] outputs = await Task.WhenAll(handles.Select(async (h, i) =>
        {
            var tokens = new List<int>();
            await foreach (int t in h.Tokens.ReadAllAsync())
                tokens.Add(t);
            finished[i] = await h.Completion;
            return tokens.ToArray();
        })).WaitAsync(TimeSpan.FromMinutes(3));
        completions?.AddRange(finished);
        verifySteps?.Invoke(((StandardDecoderModel)model).BatchedSpecVerifySteps);
        inspect?.Invoke((StandardDecoderModel)model);
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
