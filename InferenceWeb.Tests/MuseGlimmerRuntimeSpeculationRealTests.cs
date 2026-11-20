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

    /// <summary>The prompts served through one engine on <paramref name="model"/>, all admitted before the first step (one
    /// step's batch, as a server under that load forms it) or one after another.</summary>
    private static async Task<Served[]> Serve(ModelBase model, IReadOnlyList<int[]> prompts, int maxNew, SpeculationOptions speculation,
        bool together)
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

    /// <summary>The gate's check of <paramref name="drafted"/> against <paramref name="plain"/>: equal, or every token past the
    /// first divergence is the plain model's own greedy pick over the drafted prefix except at near-ties. Returns the report
    /// line; fails the test on a token that is not.</summary>
    private string CheckNearTie(ModelBase model, string name, int[] prompt, int[] plain, int[] drafted)
    {
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
        int agree = 0;
        string first = "";
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
                if (Math.Abs(split) >= NearTieNats) failures.Add($"the divergence at {d}: {split:F4} nat");
            }
            if (best == drafted[i]) agree++;
            else
            {
                double gap = lp[best] - lp[drafted[i]];
                string where = $"@{i} plain {best} ({MuseGlimmerParity.Decode(model, best)}) vs dflash {drafted[i]} " +
                    $"({MuseGlimmerParity.Decode(model, drafted[i])}): {gap:F4} nat";
                margins.Add(where);
                if (gap >= NearTieNats) failures.Add(where);
            }
            if (i + 1 < drafted.Length) logits = model.Forward(new[] { drafted[i] });
        }
        string line = $"{name}: parts from plain at token {d} of {n}{first}; teacher-forced from it the plain model agrees at {agree} of " +
            $"{drafted.Length - d} tokens; near-ties {string.Join("; ", margins)}";
        Assert.True(failures.Count == 0, $"{line}\n  NOT near-ties: {string.Join("; ", failures)}");
        return line;
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
        Served[] alone = await Serve(model, prompts, maxNew, speculation, together: false);
        Served[] together = await Serve(model, prompts, maxNew, speculation, together: true);
        long drafted = 0, accepted = 0;
        foreach (var (arm, served) in new[] { ("alone", alone), ("together", together) })
        {
            for (int r = 0; r < records.Count; r++)
            {
                output.WriteLine($"[{arm}] " + CheckNearTie(model, records[r].Name, prompts[r], plain[r].Tokens, served[r].Tokens));
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
