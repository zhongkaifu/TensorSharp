// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Speculative;

namespace AgentTurnBench;

// Diagnostic only: preserve full target distributions until the first greedy
// mismatch. This distinguishes an argmax near-tie from an assumed near-tie.
// It uses the same spec prompt and public trunk protocol, without an HTTP server.
internal static class SpecParityDiagnostic
{
    public static int Run(ModelBase model, Options options)
    {
        if (model is not ISpeculativeTarget target) throw new InvalidOperationException("Model has no speculative trunk");
        if (string.IsNullOrEmpty(options.Out)) throw new ArgumentException("--spec-diagnostic requires --out");
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        string file = Corpus.CodeText(model.Tokenizer, options.SpecFile);
        var messages = new List<ChatMessage>
        {
            new() { Role = "system", Content = options.SpecMinimalSystem ? Corpus.MinimalSystemPrompt : Corpus.AgentSystemPrompt() },
            new() { Role = "user", Content = $"Here is src/Program.cs:\n```csharp\n{file}```\nRepeat the file exactly as given, then add one sentence describing what it does." },
        };
        int[] prompt = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, messages,
            model.Config.Architecture, addGenerationPrompt: true).ToArray();
        int limit = Math.Min(options.SpecNew, 256);
        if (limit < 2) throw new ArgumentException("Diagnostic needs at least two generated tokens");
        var plain = new List<int>();
        var plainLogits = new List<float[]>();
        float[] Prefill()
        {
            model.ResetKVCache();
            float[] result = null;
            for (int offset = 0; offset < prompt.Length; offset += options.Chunk)
                result = model.ForwardRefill(prompt.AsSpan(offset, Math.Min(options.Chunk, prompt.Length - offset)).ToArray());
            return result;
        }
        float[] logits = Prefill();
        for (int i = 0; i < limit; ++i)
        {
            plainLogits.Add((float[])logits.Clone());
            int token = Argmax(logits);
            plain.Add(token);
            if (model.Tokenizer.IsEos(token)) break;
            logits = model.Forward(new[] { token });
        }
        var operations = new List<object>();
        var rows = new List<object>();
        var speculative = new List<int>();
        bool mismatch = false;
        string phase = "prefill";
        int Sample(float[] values)
        {
            int index = speculative.Count;
            int actual = Argmax(values);
            if (index >= plainLogits.Count || mismatch) return actual;
            float[] expected = plainLogits[index];
            int winner = plain[index];
            double max = 0, square = 0;
            for (int i = 0; i < values.Length; ++i)
            {
                if (!float.IsFinite(values[i]) || !float.IsFinite(expected[i]))
                    throw new InvalidOperationException($"Nonfinite logits at output {index}, vocabulary {i}");
                double difference = values[i] - expected[i];
                max = Math.Max(max, Math.Abs(difference)); square += difference * difference;
            }
            int second = Second(values, actual), expectedSecond = Second(expected, winner);
            rows.Add(new { index, phase, expected = winner, actual, max_abs = max,
                rms = Math.Sqrt(square / values.Length), expected_margin = expected[winner] - expected[expectedSecond],
                actual_margin = values[actual] - values[second], expected_winner_logit = expected[winner],
                actual_winner_logit = values[actual], reference_winner_current_logit = values[winner],
                current_winner_reference_logit = expected[actual] });
            if (actual != winner)
            {
                mismatch = true;
                File.WriteAllBytes(options.Out + ".plain_logits.f32", MemoryMarshal.AsBytes(expected.AsSpan()).ToArray());
                File.WriteAllBytes(options.Out + ".spec_logits.f32", MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
                Console.WriteLine($"SPEC_MISMATCH index={index} phase={phase} expected={winner} actual={actual} max_abs={max:G8} expected_margin={expected[winner] - expected[expectedSecond]:G8} actual_margin={values[actual] - values[second]:G8}");
            }
            return actual;
        }
        logits = Prefill();
        speculative.Add(Sample(logits));
        var trunk = new TracedTrunk(target, operations);
        using var execution = new SpeculativeExecution(target, new NGramSpeculator(7), trunk);
        // Disable only timing-dependent proposal parking in this diagnostic;
        // acceptance and every emitted token still use the real greedy target.
        execution.AdaptiveSpeculation = false;
        execution.SeedCommitted(prompt);
        while (speculative.Count < plain.Count && !mismatch && !model.Tokenizer.IsEos(speculative[^1]))
        {
            phase = "verify";
            var outcome = execution.DecodeStep(speculative[^1], model.CacheSeqLen,
                Math.Min(7, plain.Count - speculative.Count - 1), Sample,
                onDraftAccepted: token => speculative.Add(token), history: speculative);
            if (outcome.UsedSpeculation) speculative.Add(outcome.NextToken);
            else { phase = "plain-after-spec"; speculative.Add(Sample(outcome.NextLogits)); }
        }
        File.WriteAllText(options.Out, JsonSerializer.Serialize(new
        {
            prompt, plain, speculative, rows, operations, mismatch,
            plain_text = model.Tokenizer.Decode(plain), speculative_text = model.Tokenizer.Decode(speculative),
            persists_accepted_kv = target.SpecVerifyPersistsAcceptedKv,
            scope = "Diagnostic with greedy raw logits and timing governor disabled; not a performance sample or scheduler parity assertion",
        }, new JsonSerializerOptions { WriteIndented = true }));
        Console.WriteLine($"SPEC_DIAGNOSTIC prompt={prompt.Length} compared={speculative.Count} mismatch={mismatch} output={options.Out}");
        return mismatch ? 1 : 0;
    }

    private static int Argmax(float[] values)
    {
        int best = 0;
        for (int i = 1; i < values.Length; ++i) if (values[i] > values[best]) best = i;
        return best;
    }
    private static int Second(float[] values, int best)
    {
        int second = best == 0 ? 1 : 0;
        for (int i = 0; i < values.Length; ++i) if (i != best && values[i] > values[second]) second = i;
        return second;
    }
    private sealed class TracedTrunk(ISpeculativeTarget model, List<object> operations) : ISpecTrunk
    {
        private readonly LinearSpecTrunk _inner = new(model);
        public bool HasCheapPlainStep => _inner.HasCheapPlainStep;
        public void Forward(int[] tokens, float[] hidden, float[] logits, bool allLogitsRows)
        {
            operations.Add(new { kind = "forward", position = model.CacheSeqLen, tokens, allLogitsRows });
            _inner.Forward(tokens, hidden, logits, allLogitsRows);
        }
        public void ForwardPlain(int token, float[] logits, bool parked)
        {
            operations.Add(new { kind = "plain", position = model.CacheSeqLen, token, parked });
            _inner.ForwardPlain(token, logits, parked);
        }
        public void SnapshotRecurrentState() => _inner.SnapshotRecurrentState();
        public void OnVerifyAccepted(int acceptedRows, int verifyRows)
        {
            operations.Add(new { kind = "accepted", acceptedRows, verifyRows });
            _inner.OnVerifyAccepted(acceptedRows, verifyRows);
        }
        public void Rollback(int position) { operations.Add(new { kind = "rollback", position }); _inner.Rollback(position); }
        public bool TryCommitVerifiedPrefix(int position)
        {
            bool kept = _inner.TryCommitVerifiedPrefix(position);
            operations.Add(new { kind = "commit-prefix", position, kept });
            return kept;
        }
    }
}
