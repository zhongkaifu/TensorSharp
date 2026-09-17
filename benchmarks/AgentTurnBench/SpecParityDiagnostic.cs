// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Runtime.InteropServices;
using System.Text.Json;
using TensorSharp;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Grammar;
using TensorSharp.Runtime.Speculative;

namespace AgentTurnBench;

// Diagnostic only: preserve full target distributions until the first greedy
// mismatch. This distinguishes an argmax near-tie from an assumed near-tie.
// It uses the same spec prompt and public trunk protocol, without an HTTP server.
//
// --spec-diagnostic-teacher-force keeps the speculative run on the plain token path
// after a mismatch, so every row of the whole generation stays a same-prefix
// comparison; the summary then counts every argmax flip with its margins and the
// logit error of each row class (verify row 0, later verify rows, plain steps after
// a verify, and whether a rollback or a verified-prefix commit preceded the row).
internal static class SpecParityDiagnostic
{
    public static int Run(ModelBase model, Options options)
    {
        if (model is not ISpeculativeTarget target) throw new InvalidOperationException("Model has no speculative trunk");
        if (string.IsNullOrEmpty(options.Out)) throw new ArgumentException("--spec-diagnostic requires --out");
        var renderer = new KVCachePromptRenderer(new GgufPromptRenderer());
        List<ChatMessage> messages;
        if (options.SpecDiagPrompt != null)
        {
            // The server decode shape: one user message, no system prompt.
            messages = new List<ChatMessage> { new() { Role = "user", Content = options.SpecDiagPrompt } };
        }
        else if (options.SpecDiagNewChat)
        {
            // AgentTurnBench's newchat chat B prompt, on the linear trunk (no checkpoint
            // clone): the same token path, so a near-tie there is a near-tie over the clone.
            string file = Corpus.CodeText(model.Tokenizer, 200);
            messages = new List<ChatMessage>
            {
                new() { Role = "system", Content = Corpus.AgentSystemPrompt() },
                new() { Role = "user", Content = $"Repeat this text exactly:\n```csharp\n{file}```" },
            };
        }
        else if (options.SpecDiagJson)
        {
            // AgentTurnBench's json scenario prompt.
            string file = Corpus.CodeText(model.Tokenizer, 300);
            messages = new List<ChatMessage>
            {
                new() { Role = "system", Content = Corpus.AgentSystemPrompt() },
                new() { Role = "user", Content = "Return a JSON object with the keys \"name\" (string), \"lines\" (integer) and \"summary\" (string) " +
                    $"describing this file. Output only the JSON object.\n```csharp\n{file}```" },
            };
        }
        else
        {
            string file = Corpus.CodeText(model.Tokenizer, options.SpecFile);
            messages = new List<ChatMessage>
            {
                new() { Role = "system", Content = options.SpecMinimalSystem ? Corpus.MinimalSystemPrompt : Corpus.AgentSystemPrompt() },
                new() { Role = "user", Content = $"Here is src/Program.cs:\n```csharp\n{file}```\nRepeat the file exactly as given, then add one sentence describing what it does." },
            };
        }
        int[] prompt = renderer.RenderToTokens(model.Tokenizer, model.Config.ChatTemplate, messages,
            model.Config.Architecture, addGenerationPrompt: true).ToArray();
        if (options.SpecDiagRowCheck) return RowCheck(model, options, prompt);
        int limit = Math.Min(options.SpecNew, 1024);
        if (limit < 2) throw new ArgumentException("Diagnostic needs at least two generated tokens");
        bool teacherForce = options.SpecDiagTeacherForce;
        int window = Math.Max(1, options.SpecDiagWindow);
        var plain = new List<int>();
        var plainLogits = new List<float[]>();
        // --spec-diagnostic-json: both runs draw through the JSON-object grammar. The
        // grammar state before every plain token is kept, so a speculative row is masked
        // with exactly the state the plain run had at that position; logit errors are
        // measured on the raw rows, argmax and margins on the masked ones.
        GrammarConstraint grammar = options.SpecDiagJson
            ? GrammarLibrary.NewConstraint(GrammarLibrary.ForJsonObject(model.Tokenizer), model.Tokenizer)
            : null;
        var grammarStates = new List<(GrammarState State, PartialUtf8 Partial)>();
        float[] Masked(float[] raw, int index)
        {
            if (grammar == null) return raw;
            grammar.Restore(grammarStates[index]);
            float[] copy = (float[])raw.Clone();
            grammar.ApplyMask(copy, grammar.IsComplete);
            return copy;
        }
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
            if (grammar != null) grammarStates.Add(grammar.Snapshot());
            int token = Argmax(Masked(logits, i));
            if (grammar != null) { grammar.Restore(grammarStates[i]); grammar.Accept(token); }
            plain.Add(token);
            if (model.Tokenizer.IsEos(token)) break;
            logits = model.Forward(new[] { token });
        }
        var operations = new List<TraceOp>();
        var rows = new List<DiagRow>();
        var speculative = new List<int>();
        bool mismatch = false;
        string phase = "prefill";
        // Where the row being drawn sits: its index inside the current verify window
        // (0 = the row of the token the window started from), the window's width, and
        // whether the trunk rolled back or committed a verified prefix since the
        // previous verify.
        int rowInWindow = -1, windowRows = 0;
        string sinceLast = "none";
        int Sample(float[] values)
        {
            int index = speculative.Count;
            if (index >= plainLogits.Count || (mismatch && !teacherForce)) return Argmax(values);
            float[] expected = plainLogits[index];
            float[] maskedValues = Masked(values, index), maskedExpected = Masked(expected, index);
            int actual = Argmax(maskedValues);
            int winner = plain[index];
            double max = 0, square = 0;
            for (int i = 0; i < values.Length; ++i)
            {
                if (!float.IsFinite(values[i]) || !float.IsFinite(expected[i]))
                    throw new InvalidOperationException($"Nonfinite logits at output {index}, vocabulary {i}");
                double difference = values[i] - expected[i];
                max = Math.Max(max, Math.Abs(difference)); square += difference * difference;
            }
            int second = Second(maskedValues, actual), expectedSecond = Second(maskedExpected, winner);
            string rowPhase = phase == "verify" ? (rowInWindow <= 0 ? "verify-row0" : "verify-rowN") : phase;
            rows.Add(new DiagRow(index, rowPhase, rowInWindow, windowRows, sinceLast, winner, actual, max,
                Math.Sqrt(square / values.Length), maskedExpected[winner] - maskedExpected[expectedSecond],
                maskedValues[actual] - maskedValues[second], expected[winner], values[actual], values[winner], expected[actual]));
            if (phase == "verify") rowInWindow++;
            if (actual != winner)
            {
                if (!mismatch)
                {
                    File.WriteAllBytes(options.Out + ".plain_logits.f32", MemoryMarshal.AsBytes(expected.AsSpan()).ToArray());
                    File.WriteAllBytes(options.Out + ".spec_logits.f32", MemoryMarshal.AsBytes(values.AsSpan()).ToArray());
                }
                mismatch = true;
                Console.WriteLine($"SPEC_MISMATCH index={index} phase={rowPhase} expected={winner} actual={actual} max_abs={max:G8} expected_margin={maskedExpected[winner] - maskedExpected[expectedSecond]:G8} actual_margin={maskedValues[actual] - maskedValues[second]:G8}");
            }
            return teacherForce ? winner : actual;
        }
        logits = Prefill();
        speculative.Add(Sample(logits));
        var trunk = new TracedTrunk(target, operations,
            onVerify: rowsInBatch => { phase = "verify"; rowInWindow = 0; windowRows = rowsInBatch; },
            onRollback: kind => sinceLast = kind);
        ISpeculator speculator = string.Equals(options.SpecDiagSpeculator, "ngram", StringComparison.OrdinalIgnoreCase)
            ? new NGramSpeculator(window)
            : SpeculatorRegistry.Create(target,
                  new SpeculationOptions
                  {
                      Enabled = true, SpeculatorName = options.SpecDiagSpeculator,
                      MaxDraftTokens = window, MaxDraftTokensExplicit = true,
                  }, out string decline)
              ?? throw new InvalidOperationException($"speculator '{options.SpecDiagSpeculator}' unavailable: {decline}");
        using var execution = new SpeculativeExecution(target, speculator, trunk);
        // Disable only timing-dependent proposal parking in this diagnostic;
        // acceptance and every emitted token still use the real greedy target.
        execution.AdaptiveSpeculation = false;
        execution.SeedCommitted(prompt);
        while (speculative.Count < plain.Count && (teacherForce || !mismatch) && !model.Tokenizer.IsEos(speculative[^1]))
        {
            phase = "plain-after-spec";
            rowInWindow = -1; windowRows = 1;
            var outcome = execution.DecodeStep(speculative[^1], model.CacheSeqLen,
                Math.Min(window, plain.Count - speculative.Count - 1), Sample,
                onDraftAccepted: token => speculative.Add(token), history: speculative);
            string after = sinceLast;
            sinceLast = "none";
            if (outcome.UsedSpeculation) { speculative.Add(outcome.NextToken); sinceLast = after; }
            else { phase = "plain-after-spec"; rowInWindow = -1; windowRows = 1; speculative.Add(Sample(outcome.NextLogits)); }
        }

        var flips = rows.Where(r => r.expected != r.actual).ToList();
        static object Summarize(IEnumerable<DiagRow> group)
        {
            var list = group.Select(r => r.max_abs).OrderBy(v => v).ToList();
            if (list.Count == 0) return new { count = 0 };
            return new
            {
                count = list.Count, max_abs_max = list[^1], max_abs_median = list[list.Count / 2],
                max_abs_p90 = list[Math.Min(list.Count - 1, (int)(list.Count * 0.9))],
            };
        }
        // A flip is inside the error when the reference's own top-two margin is no larger
        // than twice the row's measured logit error: a kernel with that error cannot be
        // relied on to pick the same token there.
        int flipsInsideError = flips.Count(r => r.expected_margin <= 2 * r.max_abs);
        int rowsAtRisk = rows.Count(r => r.expected_margin <= 2 * r.max_abs);
        var summary = new
        {
            teacher_forced = teacherForce,
            json_grammar = grammar != null,
            speculator = speculator.Describe(),
            window,
            compared_rows = rows.Count,
            flips = flips.Count,
            flip_rows = flips.Select(r => new { r.index, r.phase, r.row_in_window, r.window_rows, r.since_last, r.expected_margin, r.actual_margin, r.max_abs }).ToList(),
            flips_inside_error = flipsInsideError,
            rows_at_risk = rowsAtRisk,
            by_phase = rows.GroupBy(r => r.phase).ToDictionary(g => g.Key, g => Summarize(g)),
            by_since_last = rows.GroupBy(r => r.since_last).ToDictionary(g => g.Key, g => Summarize(g)),
            verifies = operations.Count(o => o.kind == "forward" && o.allLogitsRows),
            rollbacks = operations.Count(o => o.kind == "rollback"),
            prefix_commits = operations.Count(o => o.kind == "commit-prefix" && o.kept),
        };
        File.WriteAllText(options.Out, JsonSerializer.Serialize(new
        {
            prompt, plain, speculative, summary, rows, operations, mismatch,
            plain_text = model.Tokenizer.Decode(plain), speculative_text = model.Tokenizer.Decode(speculative),
            persists_accepted_kv = target.SpecVerifyPersistsAcceptedKv,
            scope = "Diagnostic with greedy raw logits and timing governor disabled; not a performance sample or scheduler parity assertion",
        }, new JsonSerializerOptions
        {
            WriteIndented = true,
            // A grammar can leave a single legal token, whose margin is infinite.
            NumberHandling = System.Text.Json.Serialization.JsonNumberHandling.AllowNamedFloatingPointLiterals,
        }));
        Console.WriteLine($"SPEC_DIAGNOSTIC prompt={prompt.Length} compared={speculative.Count} rows={rows.Count} mismatch={mismatch} " +
                          $"flips={flips.Count} flips_inside_error={flipsInsideError} rows_at_risk={rowsAtRisk} " +
                          $"verifies={summary.verifies} rollbacks={summary.rollbacks} prefix_commits={summary.prefix_commits} output={options.Out}");
        foreach (var kv in summary.by_phase)
            Console.WriteLine($"SPEC_DIAGNOSTIC_PHASE {kv.Key} {JsonSerializer.Serialize(kv.Value)}");
        foreach (var kv in summary.by_since_last)
            Console.WriteLine($"SPEC_DIAGNOSTIC_SINCE {kv.Key} {JsonSerializer.Serialize(kv.Value)}");
        foreach (var r in flips)
            Console.WriteLine($"SPEC_DIAGNOSTIC_FLIP index={r.index} phase={r.phase} row={r.row_in_window}/{r.window_rows} since={r.since_last} " +
                              $"expected_margin={r.expected_margin:G6} actual_margin={r.actual_margin:G6} max_abs={r.max_abs:G6}");
        return mismatch ? 1 : 0;
    }

    /// <summary>
    /// --spec-diagnostic-rowcheck: after the prompt, compute the same next-token rows
    /// through every path a speculative step can take and report their distance from
    /// the plain decode step, all over the same committed cache (the position is
    /// rewound between the calls). Separates "a multi-row forward computes a different
    /// row" from "the rows a verify committed are what later steps inherit".
    /// Only meaningful while the sliding window has not wrapped (rewinding a wrapped
    /// ring does not restore the evicted slots).
    /// </summary>
    public static int RowCheck(ModelBase model, Options options, int[] prompt)
    {
        var target = (ISpeculativeTarget)model;
        int vocab = model.Config.VocabSize;
        // Every comparison below rewinds to the prompt's end and writes up to
        // max(3, window + 1) positions again. Once a write lands past the sliding
        // window it evicts a slot that a rewound query still attends, so the rows
        // would measure the rewind rather than the kernels: refuse instead of
        // printing numbers that look like a kernel disagreement.
        int slidingWindow = model.Config.SlidingWindow;
        int lastWritten = prompt.Length + Math.Max(3, Math.Max(2, options.SpecDiagWindow + 1)) - 1;
        if (slidingWindow > 0 && lastWritten >= slidingWindow)
            throw new ArgumentException(
                $"--spec-diagnostic-rowcheck needs every position it writes under the model's {slidingWindow}-token sliding window, " +
                $"but this prompt ({prompt.Length} tokens) with --spec-diagnostic-window {options.SpecDiagWindow} reaches position {lastWritten}; " +
                "use a shorter prompt (e.g. --spec-diagnostic-prompt) or a smaller window");
        model.ResetKVCache();
        float[] last = null;
        for (int offset = 0; offset < prompt.Length; offset += options.Chunk)
            last = model.ForwardRefill(prompt.AsSpan(offset, Math.Min(options.Chunk, prompt.Length - offset)).ToArray());
        int p = target.CacheSeqLen;
        int t0 = Argmax(last);
        float[] a = (float[])model.Forward(new[] { t0 }).Clone();          // plain decode at p
        int t1 = Argmax(a);
        float[] a2 = (float[])model.Forward(new[] { t1 }).Clone();         // plain decode at p+1
        int t2 = Argmax(a2);
        float[] a3 = (float[])model.Forward(new[] { t2 }).Clone();         // plain decode at p+2
        var report = new List<string>();
        double Dist(float[] x, long xOff, float[] y)
        {
            double m = 0;
            for (int i = 0; i < vocab; i++) m = Math.Max(m, Math.Abs(x[xOff + i] - y[i]));
            return m;
        }
        void Line(string name, double d) { report.Add($"{name}={d:G6}"); Console.WriteLine($"SPEC_ROWCHECK {name} max_abs={d:G6}"); }

        target.SpecRewindCache(p);
        var one = new float[vocab];
        target.SpecForward(new[] { t0 }, null, one, allLogitsRows: false);
        Line("spec1row_vs_decode@p", Dist(one, 0, a));

        for (int rows = 2; rows <= Math.Max(2, options.SpecDiagWindow + 1); rows++)
        {
            int[] tokens = new int[rows];
            tokens[0] = t0; tokens[1] = t1;
            for (int i = 2; i < rows; i++) tokens[i] = i == 2 ? t2 : t2;   // rows past p+2 only fill the width
            target.SpecRewindCache(p);
            var all = new float[(long)rows * vocab];
            target.SpecForward(tokens, null, all, allLogitsRows: true);
            target.SpecOnVerifyAccepted(rows - 1, rows - 1);
            Line($"verify{rows}rows_row0_vs_decode@p", Dist(all, 0, a));
            Line($"verify{rows}rows_row1_vs_decode@p+1", Dist(all, vocab, a2));
            if (rows >= 3) Line($"verify{rows}rows_row2_vs_decode@p+2", Dist(all, 2L * vocab, a3));
        }

        // The kept-prefix re-forward shape: two rows, last-row logits only.
        target.SpecRewindCache(p);
        var lastRow = new float[vocab];
        target.SpecForward(new[] { t0, t1 }, null, lastRow, allLogitsRows: false);
        Line("reforward2rows_last_vs_decode@p+1", Dist(lastRow, 0, a2));

        // A verified row committed, then a plain decode reads it: the error a later
        // step inherits from one verify-written K/V row.
        target.SpecRewindCache(p);
        var both = new float[2L * vocab];
        target.SpecForward(new[] { t0, t1 }, null, both, allLogitsRows: true);
        target.SpecOnVerifyAccepted(1, 1);
        float[] after = (float[])model.Forward(new[] { t2 }).Clone();
        Line("decode_after_verify2_vs_decode@p+2", Dist(after, 0, a3));

        // A plain multi-token Forward (the prefill kernel) over the same two tokens.
        target.SpecRewindCache(p);
        float[] multi = (float[])model.Forward(new[] { t0, t1 }).Clone();
        Line("forward2tokens_last_vs_decode@p+1", Dist(multi, 0, a2));

        File.WriteAllText(options.Out, JsonSerializer.Serialize(new { prompt_tokens = prompt.Length, position = p, t0, t1, t2, report }));
        return 0;
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

    private sealed record DiagRow(int index, string phase, int row_in_window, int window_rows, string since_last,
        int expected, int actual, double max_abs, double rms, double expected_margin, double actual_margin,
        double expected_winner_logit, double actual_winner_logit, double reference_winner_current_logit,
        double current_winner_reference_logit);

    private sealed record TraceOp(string kind, int position, int[] tokens, bool allLogitsRows,
        int acceptedRows = 0, int verifyRows = 0, bool kept = false);

    private sealed class TracedTrunk(ISpeculativeTarget model, List<TraceOp> operations,
        Action<int> onVerify, Action<string> onRollback) : ISpecTrunk
    {
        private readonly LinearSpecTrunk _inner = new(model);
        public bool HasCheapPlainStep => _inner.HasCheapPlainStep;
        public void Forward(int[] tokens, float[] hidden, float[] logits, bool allLogitsRows)
        {
            operations.Add(new TraceOp("forward", model.CacheSeqLen, tokens, allLogitsRows));
            if (allLogitsRows) onVerify(tokens.Length);
            _inner.Forward(tokens, hidden, logits, allLogitsRows);
        }
        public void ForwardPlain(int token, float[] logits, bool parked)
        {
            operations.Add(new TraceOp(parked ? "plain-parked" : "plain", model.CacheSeqLen, new[] { token }, false));
            _inner.ForwardPlain(token, logits, parked);
        }
        public void SnapshotRecurrentState() => _inner.SnapshotRecurrentState();
        public void OnVerifyAccepted(int acceptedRows, int verifyRows)
        {
            operations.Add(new TraceOp("accepted", model.CacheSeqLen, null, false, acceptedRows, verifyRows));
            _inner.OnVerifyAccepted(acceptedRows, verifyRows);
        }
        public void Rollback(int position)
        {
            operations.Add(new TraceOp("rollback", position, null, false));
            onRollback("rollback-reforward");
            _inner.Rollback(position);
        }
        public bool TryCommitVerifiedPrefix(int position)
        {
            bool kept = _inner.TryCommitVerifiedPrefix(position);
            operations.Add(new TraceOp("commit-prefix", position, null, false, kept: kept));
            if (kept) onRollback("commit-prefix");
            return kept;
        }
    }
}
