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
// A token over the line still passes when it is the argmax of the batched verify row that drew it in the run (the
// greedy-lossless property; on ggml_cuda a verify step's rows round apart from one-row decode by more than the line at some
// positions), and every test's streams agree with plain one-row decode at 97% of their positions or more.
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

    /// <summary>An arm's streams and the batched verify steps that drew them, as the engine ran them
    /// (<see cref="InferenceEngine.SpecVerifyObserverForTests"/>).</summary>
    private sealed record Arm(int[][] Prompts, Served[] Served, List<BatchedVerifyRecord> Steps);

    /// <summary>The prompts served through one engine on <paramref name="model"/>, all admitted before the first step (one
    /// step's batch, as a server under that load forms it) or one after another.</summary>
    private static async Task<Served[]> Serve(ModelBase model, IReadOnlyList<int[]> prompts, int maxNew, SpeculationOptions speculation,
        bool together, List<BatchedVerifyRecord> steps = null)
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
            engine.SpecVerifyObserverForTests = record =>
            {
                // A sampled verify's rows live only during the callback: keep each row's argmax.
                if (record.Logits is { } rows)
                    record = record with
                    {
                        Logits = null,
                        Drawn = Enumerable.Range(0, record.Ids.Length * record.Window)
                            .Select(i => MuseGlimmerParity.ArgMax(rows.AsSpan(i * record.Vocab, record.Vocab).ToArray(), record.Vocab)).ToArray(),
                    };
                lock (steps) steps.Add(record);
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
    /// drafted prefix except at near-ties. A token over the line passes when it is the argmax of the verify row that drew it
    /// (<see cref="DrawnByItsVerifyRow"/>). Returns the report line and the positions where plain one-row decode, teacher-forced
    /// on the stream, takes the stream's token; fails the test on a token that is neither.</summary>
    private (string Line, int Agree, int Total) CheckNearTie(ModelBase model, string name, int request, Arm arm, int[] plain)
    {
        int[] prompt = arm.Prompts[request], drafted = arm.Served[request].Tokens;
        int n = Math.Min(plain.Length, drafted.Length);
        int d = 0;
        while (d < n && plain[d] == drafted[d]) d++;
        if (d == n && plain.Length == drafted.Length) return ($"{name}: {n} tokens, identical", n, n);
        // Teacher-force the drafted stream through plain one-row steps from the first divergence on.
        int vocab = model.Config.VocabSize;
        model.ResetKVCache();
        float[] logits = model.ForwardPrompt(prompt.Concat(drafted.Take(d)).ToArray());
        var margins = new List<string>();
        var failures = new List<string>();
        int agree = 0, verified = 0;
        string first = "";
        // Positions over the line, judged against the run's verify rows after the teacher-forced pass.
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
            bool exact = DrawnByItsVerifyRow(arm, request, index, drafted[index], out string over);
            if (exact) verified++;
            else failures.Add(what + over);
            margins.Add($"[{(exact ? "verify-exact" : "NOT verify-exact")} @{index}{over}]");
        }
        string line = $"{name}: parts from plain at token {d} of {n}{first}; teacher-forced from it the plain model agrees at {agree} of " +
            $"{drafted.Length - d} tokens; near-ties {string.Join("; ", margins)}" +
            (verified > 0 ? $"; {verified} position(s) over the line drawn by their verify rows' argmax" : "");
        Assert.True(failures.Count == 0, $"{line}\n  NOT near-ties: {string.Join("; ", failures)}");
        return (line, d + agree, drafted.Length);
    }

    /// <summary>
    /// The decided gate's second branch, on every lane: whether the token at output <paramref name="index"/> of request
    /// <paramref name="request"/> is the argmax of the batched verify row that drew it in the run, which is the
    /// greedy-lossless property itself. The row is the engine's own: the step's draws as the trunk computed them, the device's
    /// argmax on a greedy verify. A lane whose verify rows are one-row decode's bits never reaches this. A corrupt verify
    /// forward would pass it, so the test also holds every stream's teacher-forced agreement with plain one-row decode
    /// (<see cref="VerifyWindowLog.MinAgreement"/>).
    /// </summary>
    private static bool DrawnByItsVerifyRow(Arm arm, int request, int index, int token, out string note)
    {
        string id = $"req{request}";
        int position = arm.Prompts[request].Length + index;
        lock (arm.Steps)
            for (int k = arm.Steps.Count - 1; k >= 0; k--)
            {
                BatchedVerifyRecord step = arm.Steps[k];
                int lane = Array.IndexOf(step.Ids, id);
                if (lane < 0 || position <= step.Positions[lane] || position > step.Positions[lane] + step.Kept[lane])
                    continue;
                int row = position - step.Positions[lane] - 1, drawn = step.Drawn![lane * step.Window + row];
                note = $"; the verify row that drew it ({step.Ids.Length} request(s) of {step.Window} rows, row {row}) takes {drawn}";
                return drawn == token;
            }
        note = " (no verify step drew it)";
        return false;
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
        var aloneSteps = new List<BatchedVerifyRecord>();
        var togetherSteps = new List<BatchedVerifyRecord>();
        Served[] alone = await Serve(model, prompts, maxNew, speculation, together: false, aloneSteps);
        Served[] together = await Serve(model, prompts, maxNew, speculation, together: true, togetherSteps);
        long drafted = 0, accepted = 0;
        int agreeing = 0, positions = 0;
        foreach (var (arm, served, steps) in new[] { ("alone", alone, aloneSteps), ("together", together, togetherSteps) })
        {
            for (int r = 0; r < records.Count; r++)
            {
                var (line, agree, total) = CheckNearTie(model, records[r].Name, r, new Arm(prompts, served, steps), plain[r].Tokens);
                agreeing += agree;
                positions += total;
                output.WriteLine($"[{arm}] " + line);
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
        double agreement = (double)agreeing / positions;
        output.WriteLine($"{backend}: drafted {drafted}, accepted {accepted} ({100.0 * accepted / Math.Max(1, drafted):F1}%); teacher-forced " +
            $"agreement with plain one-row decode {agreeing}/{positions} = {agreement:P2} (bound {VerifyWindowLog.MinAgreement:P0})");
        Assert.True(drafted > 0, "DFlash never drafted a token");
        Assert.True(agreement >= VerifyWindowLog.MinAgreement,
            $"the DFlash streams agree with plain one-row decode at {agreement:P2} of their positions, under {VerifyWindowLog.MinAgreement:P0}");
    }
}
