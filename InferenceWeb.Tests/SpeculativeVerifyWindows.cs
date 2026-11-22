// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using TensorSharp.Runtime.Speculative;

namespace InferenceWeb.Tests;

/// <summary>
/// The verify rows a <see cref="SpeculativeDecoder"/> drew its tokens from, read from the run itself
/// (<see cref="SpeculativeDecoder.VerifyObserverForTests"/>). The top of each drawn row is kept: its log-probabilities from
/// the step's own logits, or the device's argmax where the trunk drew the rows there.
/// <para>A speculative stream's token comes from a verify row. Where a backend's verify rows round apart from its one-row decode
/// (ggml_cuda, and direct CUDA for formats without row-invariant kernels), the stream can part from plain greedy at a position
/// that is not a near-tie of the one-row logits. It is still greedy-lossless when the token is the argmax of the verify row
/// that drew it: no token was emitted that the target's verify products did not choose. A corrupt verify forward would pass
/// that check, so the tests pair it with the stream's teacher-forced agreement with plain one-row decode
/// (<see cref="MinAgreement"/>).</para>
/// </summary>
internal sealed class VerifyWindowLog
{
    /// <summary>The share of a speculative stream's positions where plain one-row decode, teacher-forced on the stream, must take
    /// the stream's token (the parity tests' argmax agreement).</summary>
    public const double MinAgreement = 0.97;

    /// <summary>The row of a verify step that drew the token at <see cref="Position"/>: the step's first position, its rows and
    /// this row's index, and the row's best tokens with their log-probabilities (descending; the log-probabilities are NaN where
    /// only the device's argmax is known).</summary>
    internal readonly record struct Row(int Position, int Start, int WindowRows, int Index, (int Id, double Logprob)[] Top);

    private const int TopKept = 8;
    private readonly Dictionary<int, Row> _rows = new();
    private readonly int _vocab;

    public VerifyWindowLog(SpeculativeDecoder decoder, int vocab)
    {
        _vocab = vocab;
        decoder.VerifyObserverForTests = Record;
    }

    public void Clear() => _rows.Clear();

    /// <summary>The verify row that drew the token at <paramref name="position"/> (an absolute position), or null where a
    /// prefill or a plain step drew it.</summary>
    public Row? Drawing(int position) => _rows.TryGetValue(position, out Row row) ? row : null;

    /// <summary>Row r of a window starting at s draws the token at s + 1 + r, for r up to the accepted drafts (the last of
    /// those rows draws the step's next token).</summary>
    private void Record(int start, SpeculativeStepOutcome outcome)
    {
        int rows = outcome.VerifiedWindow!.Length;
        for (int r = 0; r <= outcome.AcceptedCount && r < rows; r++)
        {
            (int, double)[] top;
            if (outcome.VerifiedLogits is { } logits)
                top = TopOf(logits.AsSpan(r * _vocab, _vocab));
            else if (outcome.VerifiedDraws is { } draws)
                top = new[] { (draws[r], double.NaN) };
            else
                continue;
            _rows[start + 1 + r] = new Row(start + 1 + r, start, rows, r, top);
        }
    }

    /// <summary>A row's best <see cref="TopKept"/> tokens and their log-probabilities.</summary>
    internal static (int Id, double Logprob)[] TopOf(ReadOnlySpan<float> logits)
    {
        float max = float.NegativeInfinity;
        foreach (float v in logits) max = Math.Max(max, v);
        double sum = 0;
        foreach (float v in logits) sum += Math.Exp(v - max);
        double log = max + Math.Log(sum);
        var best = new List<(int Id, double Logprob)>(TopKept + 1);
        for (int i = 0; i < logits.Length; i++)
        {
            if (best.Count == TopKept && logits[i] <= best[^1].Logprob + log) continue;
            best.Add((i, logits[i] - log));
            best.Sort((a, b) => b.Logprob.CompareTo(a.Logprob));
            if (best.Count > TopKept) best.RemoveAt(TopKept);
        }
        return best.ToArray();
    }

    /// <summary>The report of <paramref name="row"/> for the stream's <paramref name="token"/> against plain's
    /// <paramref name="plain"/>: which token the row takes and by how much.</summary>
    public static string Describe(Row row, int token, int plain)
    {
        var (best, lp) = row.Top[0];
        string margin = row.Top.Length > 1 && !double.IsNaN(lp) ? $" by {lp - row.Top[1].Logprob:F4} over {row.Top[1].Id}" : "";
        int at = Array.FindIndex(row.Top, t => t.Id == plain);
        string plainAt = double.IsNaN(lp) ? "" : at >= 0 ? $"; plain's {plain} {row.Top[at].Logprob:F4} there" : $"; plain's {plain} not in its top {row.Top.Length}";
        return $"the verify row that drew it ({row.WindowRows}-row window from position {row.Start}, row {row.Index}) takes {best}{margin}{plainAt}";
    }
}
