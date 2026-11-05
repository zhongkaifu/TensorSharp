// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Security.Cryptography;
using System.Text.Json;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

internal static class GreedyVerifyTrace
{
    // This is a correctness probe, not a benchmark: each opted-in verify first
    // runs the full-logit oracle, then leaves the token-only verify live for
    // Commit. Future draft/decode steps therefore consume candidate state.
    internal static int Run(string modelPath, string output, int rounds)
    {
        if (rounds < 8) throw new ArgumentException("Use at least eight rounds.");
        if (Environment.GetEnvironmentVariable("TS_Q35_DRAFT_SPLITK") != "0")
            throw new ArgumentException("Set TS_Q35_DRAFT_SPLITK=0 for a deterministic draft quality comparison.");
        string draftPath = Environment.GetEnvironmentVariable("TS_TEST_QWEN35_DRAFT")
            ?? throw new ArgumentException("Set TS_TEST_QWEN35_DRAFT.");
        Directory.CreateDirectory(output);
        void Manifest(bool complete, int captures) => File.WriteAllText(Path.Combine(output, "run.json"), JsonSerializer.Serialize(new
        {
            complete, greedy_verify = Environment.GetEnvironmentVariable("TS_Q35_GREEDY_VERIFY"),
            draft_split_k = "0", oracle_verifies = captures, rounds,
            timing_is_benchmark = false, governor_reset_each_round = true,
            note = "Full-logit oracle runs first; candidate verify runs last and is the state that commits."
        }, new JsonSerializerOptions { WriteIndented = true }));
        Manifest(false, 0);
        using var model = new Qwen35CudaModel(modelPath, draftPath);
        ISpeculativeTarget target = model;
        var trace = new List<object>();
        string Hash(float[] values) => Convert.ToHexString(SHA256.HashData(MemoryMarshal.AsBytes(values.AsSpan()))).ToLowerInvariant();
        var trunk = new OracleTrunk(target, output, (kind, values) =>
        {
            if (values.Any(v => !float.IsFinite(v))) throw new InvalidOperationException("Non-finite target logits.");
            trace.Add(new { kind, logits_sha256 = Hash(values) });
        });
        SpeculativeExecution Create()
        {
            var speculator = SpeculatorRegistry.Create(target, new SpeculationOptions
            {
                Enabled = true, MaxDraftTokens = 7, MaxDraftTokensExplicit = true, MinDraftProb = 0
            }, out string? decline) ?? throw new InvalidOperationException($"The model did not expose its learned drafter: {decline}");
            return new SpeculativeExecution(target, speculator, trunk);
        }
        string[] prompts =
        {
            "Write a Python function returning the first n Fibonacci numbers. Return code only.\n",
            "Explain why the sky appears blue during a clear day.\n",
            "Translate to Chinese: The library opens every morning except Sunday.\n"
        };
        for (int scenario = 0; scenario < prompts.Length; scenario++)
        {
            model.ResetKVCache();
            var config = SamplingConfig.Greedy;
            int forcedEnd = model.Tokenizer.Encode(".", false).First();
            if (scenario == 2) config.ThinkingBudget = new ThinkingTokenBudget(3, forcedEnd);
            var sampler = new TokenSampler(config);
            var history = new List<int>();
            var oneHot = new float[target.Config.VocabSize];
            // Use the public sampler to exercise its complete forced-thinking
            // policy without widening production API visibility for this probe.
            // A plain-greedy one-hot row represents exactly the device argmax.
            int Greedy(int raw)
            {
                if ((uint)raw >= (uint)oneHot.Length) throw new InvalidOperationException("Invalid device token.");
                oneHot[raw] = 1;
                int selected = sampler.Sample(oneHot, history);
                oneHot[raw] = 0;
                return selected;
            }
            int[] prompt = model.Tokenizer.Encode(prompts[scenario], true).ToArray();
            var committed = new List<int>(prompt);
            SpeculativeExecution exec = Create();
            try
            {
                float[] prefill = exec.PrefillStep(prompt, 0, prompt.Length);
                int anchor = sampler.Sample(prefill, history), position = prompt.Length;
                for (int round = 0; round < rounds; round++)
                {
                    // A quality comparison must not let doubled oracle work alter
                    // the timing governor's later draft/verify decisions.
                    exec.ResetStatsAndGovernor();
                    history.Add(anchor);
                    var accepted = new List<int>();
                    int Draw(float[] logits) => sampler.Sample(logits, history);
                    var result = exec.DecodeStep(anchor, position, new[] { 1, 2, 3, 7 }[round % 4], Draw,
                        adjustDraftLogits: (logits, pending) => sampler.ApplyPenalties(logits, history, pending),
                        onDraftAccepted: token => { history.Add(token); accepted.Add(token); }, drawGreedy: Greedy);
                    committed.Add(anchor);
                    committed.AddRange(accepted);
                    position += 1 + result.AcceptedCount;
                    if (target.CacheSeqLen != position) throw new InvalidOperationException("Commit position drifted.");
                    anchor = result.NextToken >= 0 ? result.NextToken : Draw(result.NextLogits
                        ?? throw new InvalidOperationException("A plain step lost its next logits."));
                    trace.Add(new { scenario, round, position, accepted = accepted.ToArray(), next_token = anchor,
                        history = history.ToArray(), used_speculation = result.UsedSpeculation });
                    if (round == 3)
                    {
                        // Drop/re-arm the executor without changing the slot; the
                        // device drafter ring must survive context lifetime changes.
                        exec.Dispose();
                        exec = Create();
                        exec.SeedCommitted(committed.ToArray());
                    }
                }
                // A later turn appends prompt rows to the retained cache, then
                // resumes speculation from the updated device ring.
                int[] suffix = model.Tokenizer.Encode("\nContinue with another example.\n", false).ToArray();
                var logits = new float[target.Config.VocabSize];
                target.SpecForward(suffix, null, logits, false);
                trunk.Record("next-turn", logits);
                committed.AddRange(suffix);
                exec.Dispose();
                exec = Create();
                exec.SeedCommitted(committed.ToArray());
                exec.ResetStatsAndGovernor();
                int nextTurnAnchor = sampler.Sample(logits, history), nextTurnPosition = target.CacheSeqLen;
                history.Add(nextTurnAnchor);
                var nextTurn = exec.DecodeStep(nextTurnAnchor, nextTurnPosition, 7,
                    row => sampler.Sample(row, history), onDraftAccepted: token => history.Add(token),
                    drawGreedy: Greedy);
                if (target.CacheSeqLen != nextTurnPosition + nextTurn.AcceptedCount + 1)
                    throw new InvalidOperationException("Next-turn commit position drifted.");
                trace.Add(new { scenario, next_turn_position = target.CacheSeqLen, next_turn_token = nextTurn.NextToken,
                    next_turn_accepted = nextTurn.AcceptedCount, history = history.ToArray() });
            }
            finally { exec.Dispose(); }
        }
        if (Environment.GetEnvironmentVariable("TS_Q35_GREEDY_VERIFY") == "1" && trunk.GreedyVerifies == 0)
            throw new InvalidOperationException("Requested greedy verify never ran.");
        File.WriteAllText(Path.Combine(output, "trace.json"), JsonSerializer.Serialize(trace, new JsonSerializerOptions { WriteIndented = true }));
        Manifest(true, trunk.GreedyVerifies);
        Console.WriteLine($"[q35-greedy-verify] complete; {trunk.GreedyVerifies} token-only verifies checked against full vocabulary logits");
        return 0;
    }

    private sealed class OracleTrunk : ISpecTrunk
    {
        private readonly ISpeculativeTarget _target;
        private readonly LinearSpecTrunk _linear;
        private readonly string _output;
        internal readonly Action<string, float[]> Record;
        internal int GreedyVerifies { get; private set; }
        internal OracleTrunk(ISpeculativeTarget target, string output, Action<string, float[]> record)
        { _target = target; _linear = new LinearSpecTrunk(target); _output = output; Record = record; }
        public void Forward(int[] tokens, float[]? hidden, float[] logits, bool allLogitsRows)
        {
            _linear.Forward(tokens, hidden, logits, allLogitsRows);
            int count = (allLogitsRows ? tokens.Length : 1) * _target.Config.VocabSize;
            Record(allLogitsRows ? "verify" : "forward", logits[..count]);
        }
        public bool TryForwardGreedyVerify(int[] tokens, float[]? hidden, int[] drawn)
        {
            if (Environment.GetEnvironmentVariable("TS_Q35_GREEDY_VERIFY") != "1") return false;
            var logits = new float[tokens.Length * _target.Config.VocabSize];
            Forward(tokens, hidden, logits, true);
            if (!_linear.TryForwardGreedyVerify(tokens, hidden, drawn))
                throw new InvalidOperationException("The requested token-only verify declined.");
            var sampler = new TokenSampler(SamplingConfig.Greedy);
            for (int row = 0; row < tokens.Length; row++)
            {
                int expected = sampler.Sample(logits.AsSpan(row * _target.Config.VocabSize, _target.Config.VocabSize).ToArray());
                if (drawn[row] != expected)
                {
                    File.WriteAllBytes(Path.Combine(_output, "failure-logits.f32"), MemoryMarshal.AsBytes(logits.AsSpan()));
                    File.WriteAllText(Path.Combine(_output, "failure.json"), JsonSerializer.Serialize(new
                    { row, expected, drawn, tokens, position = _target.CacheSeqLen }));
                    throw new InvalidOperationException($"Greedy verify row {row}: device={drawn[row]}, full-logit CPU={expected}.");
                }
            }
            GreedyVerifies++;
            return true;
        }
        public void SnapshotRecurrentState() => _linear.SnapshotRecurrentState();
        public void OnVerifyAccepted(int acceptedRows, int verifyRows) => _linear.OnVerifyAccepted(acceptedRows, verifyRows);
        public void Rollback(int position) => _linear.Rollback(position);
        public bool TryCommitVerifiedPrefix(int position) => _linear.TryCommitVerifiedPrefix(position);
    }
}
