// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// Real weights (TS_TEST_MODEL_DIR with Muse-Glimmer-30B-UD-IQ2_XXS.gguf and dflash*.gguf; the backend from
// TS_MUSE_GLIMMER_BACKEND): DFlash on the runtime family through the engine, as served, against plain greedy decoding through
// the same engine (stage S5 of docs/design/muse-glimmer-unified.md, §4.4). The gate decided for both families: a DFlash stream
// equals plain greedy's, or its first divergence falls where plain greedy's top two tokens are under 0.15 nat apart, and
// teacher-forced past that point the plain model agrees with every later DFlash token except at such near-ties. Legacy and
// llama.cpp are not token-exact under DFlash on ggml_cuda either; every divergence checked there was such a near-tie (S0).
// A token over the line still passes when the target's own logits over the batched verify the engine ran prefer it: on
// ggml_cuda a verify step's rows round apart from one-row decode by more than the line at some positions.
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit.Abstractions;

namespace InferenceWeb.Tests;

public sealed class MuseGlimmerRuntimeSpeculationRealTests(ITestOutputHelper output)
{
    /// <summary>The near-tie bound of the decided gate: plain greedy's own log-probability gap between its token and DFlash's.</summary>
    private const double NearTieNats = 0.15;

    private sealed record Served(int[] Tokens, SpeculationStats Stats);

    /// <summary>One batched verify as the engine ran it (<see cref="InferenceEngine.SpecVerifyObserverForTests"/>): the requests,
    /// the positions of their windows' first tokens, the window's rows, the windows' tokens (request i's from i * window) and
    /// the rows each kept.</summary>
    private sealed record VerifyStep(string[] Ids, int[] Positions, int Window, int[] Tokens, int[] Kept);

    /// <summary>An arm's streams and the verify steps that drew them, for recomputing a step over the same windows.</summary>
    private sealed record Arm(int[][] Prompts, Served[] Served, List<VerifyStep> Steps);

    /// <summary>The prompts served through one engine on <paramref name="model"/>, all admitted before the first step (one
    /// step's batch, as a server under that load forms it) or one after another.</summary>
    private static async Task<Served[]> Serve(ModelBase model, IReadOnlyList<int[]> prompts, int maxNew, SpeculationOptions speculation,
        bool together, List<VerifyStep> steps = null)
    {
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 2048,
            MaxNumRunningSequences = Math.Max(1, prompts.Count),
            MaxPrefillChunkSize = 512,
            NumBlocks = 2048,
            BlockSize = 16,
            EnablePrefixCaching = false,
            Speculation = speculation ?? new SpeculationOptions { Enabled = false },
        };
        using var engine = new InferenceEngine(model, cfg, NullLogger.Instance);
        if (steps != null)
            engine.SpecVerifyObserverForTests = (ids, positions, window, tokens, kept) =>
            {
                lock (steps) steps.Add(new VerifyStep(ids, positions, window, tokens, kept));
            };
        var results = new Served[prompts.Count];
        async Task<Served> One(int i, InferenceRequestHandle handle, SequenceState seq)
        {
            var tokens = new List<int>();
            await foreach (int t in handle.Tokens.ReadAllAsync()) tokens.Add(t);
            await handle.Completion;
            return new Served(tokens.ToArray(), seq.SpecStats);
        }
        SequenceState Sequence(int i) => new($"req{i}", prompts[i], maxNew, cfg.BlockSize, SamplingConfig.Greedy);
        if (together)
        {
            var gate = new ComputeGate();
            gate.Close();
            engine.ComputeGate = gate;
            var seqs = Enumerable.Range(0, prompts.Count).Select(Sequence).ToList();
            var handles = seqs.Select(s => engine.SubmitRequest(s)).ToList();
            gate.Open();
            return await Task.WhenAll(handles.Select((h, i) => One(i, h, seqs[i]))).WaitAsync(TimeSpan.FromMinutes(20));
        }
        for (int i = 0; i < prompts.Count; i++)
        {
            SequenceState seq = Sequence(i);
            results[i] = await One(i, engine.SubmitRequest(seq), seq).WaitAsync(TimeSpan.FromMinutes(10));
        }
        return results;
    }

    /// <summary>The gate's check of request <paramref name="request"/>'s stream in <paramref name="arm"/> against
    /// <paramref name="plain"/>: equal, or every token past the first divergence is the plain model's own greedy pick over the
    /// drafted prefix except at near-ties. A token over the line passes when the target prefers it over the verify step that
    /// drew it (<see cref="PrefersOverItsVerify"/>). Returns the report line; fails the test on a token that is neither.</summary>
    private string CheckNearTie(ModelBase model, string name, int request, Arm arm, int[] plain)
    {
        int[] prompt = arm.Prompts[request], drafted = arm.Served[request].Tokens;
        int n = Math.Min(plain.Length, drafted.Length);
        int d = 0;
        while (d < n && plain[d] == drafted[d]) d++;
        if (d == n && plain.Length == drafted.Length) return $"{name}: {n} tokens, identical";
        // Teacher-force the drafted stream through plain one-row steps from the first divergence on.
        int vocab = model.Config.VocabSize;
        model.ResetKVCache();
        float[] logits = model.ForwardPrompt(prompt.Concat(drafted.Take(d)).ToArray());
        var margins = new List<string>();
        var failures = new List<string>();
        int agree = 0, verified = 0;
        string first = "";
        // Positions over the line, judged after the teacher-forced pass (the recomputation binds slots of its own).
        var overLine = new List<(int Index, string What)>();
        for (int i = d; i < drafted.Length; i++)
        {
            double[] lp = MuseGlimmerParity.LogSoftmax(logits, vocab);
            int best = MuseGlimmerParity.ArgMax(logits, vocab);
            if (i == d && d < plain.Length)
            {
                // The divergence itself: plain greedy's token against DFlash's under the plain model over their common prefix
                // (prefilled in one forward, which rounds apart from the decode that chose plain's token: either may lead).
                double split = lp[plain[d]] - lp[drafted[d]];
                first = $"; at {d} plain's {plain[d]} ({MuseGlimmerParity.Decode(model, plain[d])}) against dflash's {drafted[d]} " +
                    $"({MuseGlimmerParity.Decode(model, drafted[d])}): {split:F4} nat";
                if (Math.Abs(split) >= NearTieNats) overLine.Add((d, $"the divergence at {d}: {split:F4} nat"));
            }
            if (best == drafted[i]) agree++;
            else
            {
                double gap = lp[best] - lp[drafted[i]];
                string where = $"@{i} plain {best} ({MuseGlimmerParity.Decode(model, best)}) vs dflash {drafted[i]} " +
                    $"({MuseGlimmerParity.Decode(model, drafted[i])}): {gap:F4} nat";
                margins.Add(where);
                if (gap >= NearTieNats) overLine.Add((i, where));
            }
            if (i + 1 < drafted.Length) logits = model.Forward(new[] { drafted[i] });
        }
        foreach (var (index, what) in overLine.GroupBy(o => o.Index).Select(g => (g.Key, string.Join("; ", g.Select(o => o.What)))))
        {
            bool exact = PrefersOverItsVerify(model, arm, request, index, drafted[index], out string over);
            if (exact) verified++;
            else failures.Add(what + over);
            margins.Add($"[{(exact ? "verify-exact" : "NOT verify-exact")} @{index}{over}]");
        }
        string line = $"{name}: parts from plain at token {d} of {n}{first}; teacher-forced from it the plain model agrees at {agree} of " +
            $"{drafted.Length - d} tokens; near-ties {string.Join("; ", margins)}" +
            (verified > 0 ? $"; {verified} position(s) over the line that the target's own verify step prefers" : "");
        Assert.True(failures.Count == 0, $"{line}\n  NOT near-ties: {string.Join("; ", failures)}");
        return line;
    }

    /// <summary>
    /// The decided gate's second branch, on every lane: the target's own logits for the token at output
    /// <paramref name="index"/> of request <paramref name="request"/>, recomputed over the batched verify that drew it.
    /// <para>The step is recovered whole: its requests, the positions of their windows' first tokens, its rows and the windows'
    /// tokens. Every request in it is resumed at its window's first token, with its prompt and its stream before that token
    /// prefilled in one forward. Then the step's windows are verified together, as the engine verified them.</para>
    /// <para>When the target draws the stream's token there, the acceptance was exact against the target's verify products.
    /// The stream left plain greedy because those rows round apart from one-row decode, which ggml_cuda's do by more than the
    /// line at some positions. A lane whose verify rows are one-row decode's bits never reaches this.</para>
    /// </summary>
    private static bool PrefersOverItsVerify(ModelBase model, Arm arm, int request, int index, int token, out string note)
    {
        string id = $"req{request}";
        int position = arm.Prompts[request].Length + index;
        VerifyStep step = null;
        int lane = -1;
        lock (arm.Steps)
            for (int k = arm.Steps.Count - 1; k >= 0 && step == null; k--)
            {
                int j = Array.IndexOf(arm.Steps[k].Ids, id);
                if (j >= 0 && position > arm.Steps[k].Positions[j] && position <= arm.Steps[k].Positions[j] + arm.Steps[k].Kept[j])
                    (step, lane) = (arm.Steps[k], j);
            }
        if (step == null)
        {
            note = " (no verify step drew it)";
            return false;
        }
        var unified = (StandardDecoderModel)model;
        var batched = (IBatchedSpeculativeTarget)model;
        int vocab = model.Config.VocabSize, n = step.Ids.Length, window = step.Window;
        string[] ids = step.Ids.Select(x => $"recompute-{x}-{Guid.NewGuid():N}").ToArray();
        try
        {
            for (int j = 0; j < n; j++)
            {
                int r = int.Parse(step.Ids[j].AsSpan(3));
                int before = step.Positions[j] - arm.Prompts[r].Length;
                Assert.Equal(arm.Served[r].Tokens[before], step.Tokens[j * window]);
                int[] context = arm.Prompts[r].Concat(arm.Served[r].Tokens.Take(before)).ToArray();
                unified.EnsureSequenceSlot(ids[j], step.Positions[j] + window + 1);
                Assert.True(unified.TryForwardUnified(new[] { ids[j] }, new[] { context }, new[] { 0 }, new float[1][]));
            }
            var features = new float[(long)n * window * ((ISpeculativeTarget)model).SpecFeatureSize];
            var drawn = new int[n * window];
            float[] scratch = null, rows = null;
            using (batched.BeginBatchedSpeculation())
            {
                Assert.True(batched.TryBatchedSpecVerify(ids, step.Positions, window, step.Tokens, drawn, features));
                if (batched.SupportsBatchedSpecLogits
                    && !batched.TryBatchedSpecVerifyLogits(ids, step.Positions, window, step.Tokens, features, ref scratch,
                        (long)n * window * vocab, out rows))
                    rows = null;
            }
            int row = position - step.Positions[lane] - 1, best = drawn[lane * window + row];
            string logprobs = "";
            if (rows != null)
            {
                double[] lp = MuseGlimmerParity.LogSoftmax(rows.AsSpan((lane * window + row) * vocab, vocab).ToArray(), vocab);
                logprobs = $" ({token} {lp[token]:F4}, best {lp[best]:F4})";
            }
            note = $"; over its verify step ({n} request(s) of {window} rows, row {row}) the target draws {best}{logprobs}";
            return best == token;
        }
        finally
        {
            foreach (string x in ids) unified.OnSequenceReleased(x);
        }
    }

    /// <summary>
    /// The five llama.cpp golden prompts and the long one (4,664 tokens, past twice the window), 128 tokens each, served with
    /// DFlash at the family's defaults one request at a time and all six at once (continuous batching over the requests'
    /// slots, each drafting over its own ring), against plain greedy decoding through the same engine.
    /// </summary>
    [MuseGlimmerModelFact(NeedsDraft = true)]
    public Task DFlashThroughTheEngineIsLosslessUpToNearTies() => AssertLossless(new SpeculationOptions { Enabled = true });

    /// <summary>The same with the widest window and no confidence gate: every step drafts the drafter's whole block.</summary>
    [MuseGlimmerModelFact(NeedsDraft = true)]
    public Task UngatedWideDFlashThroughTheEngineIsLosslessUpToNearTies()
        => AssertLossless(new SpeculationOptions { Enabled = true, MaxDraftTokens = 15, MaxDraftTokensExplicit = true, MinDraftProb = 0 });

    /// <summary>The CLI's chat prompts of the speed gate (an essay, a module of code, a list; rendered through the GGUF's
    /// template, thinking off, as the CLI's --input-jsonl renders them), 512-token answers, with the ungated wide window.</summary>
    [MuseGlimmerModelFact(NeedsDraft = true)]
    public Task ChatAnswersOf512TokensAreLosslessUpToNearTies()
        => AssertLossless(new SpeculationOptions { Enabled = true, MaxDraftTokens = 7, MaxDraftTokensExplicit = true, MinDraftProb = 0 },
            chat: true);

    private static readonly (string Name, string Text)[] ChatPrompts =
    {
        ("essay", "Write a detailed, well-structured essay about the history of computing, from the abacus and mechanical calculators " +
            "to modern GPUs and large language models. Cover the key inventors, the turning points, and what each step made possible."),
        ("code", "Write a complete Python module implementing a binary search tree with insert, delete, search, minimum, maximum and " +
            "in-order, pre-order and post-order traversal. Use type hints and a docstring for every method, then add a small set of " +
            "unittest test cases."),
        ("list", "List all 50 US states in alphabetical order, one per line, in the form 'State - Capital - one notable fact about the state'."),
    };

    private async Task AssertLossless(SpeculationOptions speculation, bool chat = false)
    {
        BackendType backend = MuseGlimmerParity.Backend();
        using ModelBase model = MuseGlimmerArchitecture.CreateRuntimeModel(MuseGlimmerParity.FindModel(), backend,
            draftModelPath: MuseGlimmerParity.FindDraft());
        List<(string Name, int[] Tokens)> records = chat
            ? ChatPrompts.Select(p => (p.Name, model.Tokenizer.Encode(new TensorSharp.Runtime.GgufPromptRenderer().Render(model.Config.ChatTemplate,
                new List<TensorSharp.Runtime.ChatMessage> { new() { Role = "user", Content = p.Text } }, addGenerationPrompt: true,
                architecture: model.Config.Architecture, enableThinking: false), addSpecial: true).ToArray())).ToList()
            : MuseGlimmerParity.Load("ref_text.json").Records.Concat(MuseGlimmerParity.Load("ref_text_long.json").Records)
                .Select(r => (r.Name, r.PromptTokens)).ToList();
        int[][] prompts = records.Select(r => r.Tokens).ToArray();
        int maxNew = chat ? 512 : 128;
        Assert.True(model.HasDFlash, "the drafter did not attach");
        Served[] plain = await Serve(model, prompts, maxNew, null, together: false);
        var aloneSteps = new List<VerifyStep>();
        var togetherSteps = new List<VerifyStep>();
        Served[] alone = await Serve(model, prompts, maxNew, speculation, together: false, aloneSteps);
        Served[] together = await Serve(model, prompts, maxNew, speculation, together: true, togetherSteps);
        long drafted = 0, accepted = 0;
        foreach (var (arm, served, steps) in new[] { ("alone", alone, aloneSteps), ("together", together, togetherSteps) })
        {
            for (int r = 0; r < records.Count; r++)
            {
                output.WriteLine($"[{arm}] " + CheckNearTie(model, records[r].Name, r, new Arm(prompts, served, steps), plain[r].Tokens));
                if (served[r].Stats is { } s)
                {
                    output.WriteLine($"         drafted {s.TokensDrafted}, accepted {s.TokensAccepted}, verify steps {s.VerifySteps}, " +
                        $"plain steps {s.PlainSteps}, parked {s.ParkedSteps}");
                    drafted += s.TokensDrafted;
                    accepted += s.TokensAccepted;
                }
            }
            // Concurrent requests: each one's stream against its own sequential DFlash run too.
            if (arm == "together")
                for (int r = 0; r < records.Count; r++)
                    output.WriteLine($"[together vs alone] {(together[r].Tokens.SequenceEqual(alone[r].Tokens) ? "identical" : "differs")}: {records[r].Name}");
        }
        output.WriteLine($"{backend}: drafted {drafted}, accepted {accepted} ({100.0 * accepted / Math.Max(1, drafted):F1}%)");
        Assert.True(drafted > 0, "DFlash never drafted a token");
    }
}
