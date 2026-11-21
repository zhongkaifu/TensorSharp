// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
using System.Globalization;
using System.Text;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

/// <summary>
/// Where two block drafts of the same inputs part (one drafter's products rounding otherwise than the other's), whether the
/// slot's decision was a near-tie, by <see cref="SpeculativeRuntimeChecks"/>' rule: the two values that decided it within a
/// share of the largest value's magnitude. A slot decides twice: its head picks the candidates (the selector's top-k of the
/// head's logits; plain DFlash's top-1), whose boundary is the k-th logit against the (k+1)-th, and the selector's walk picks
/// one of them, its best score against the second. Read off a lone block's trace (ModelBase.DFlashTraceForTests).
/// </summary>
internal static class DFlashNearTies
{
    /// <summary>The share of the largest magnitude a deciding gap may be for a near-tie (SpeculativeRuntimeChecks' 1e-2).</summary>
    public const double NearTie = 1e-2;

    /// <summary>The slot's head boundary gap (k-th logit minus the (k+1)-th, over the largest |logit|; k = 1 without a
    /// selector) and its walk gap (best score minus the second, over the largest |score|; NaN without a selector or scores).</summary>
    public static (double Boundary, double Walk) Margins(ModelBase.DFlashSlotTrace slot)
    {
        double boundary = double.NaN, walk = double.NaN;
        if (slot.Logits != null)
        {
            int k = slot.Candidates?.Length ?? 1;
            float[] sorted = slot.Logits.OrderByDescending(v => v).ToArray();
            double largest = slot.Logits.Max(v => Math.Abs((double)v));
            if (k < sorted.Length && largest > 0) boundary = (sorted[k - 1] - sorted[k]) / largest;
        }
        if (slot.Scores is { Length: > 1 })
        {
            float[] sorted = slot.Scores.OrderByDescending(v => v).ToArray();
            double largest = slot.Scores.Max(v => Math.Abs((double)v));
            if (largest > 0) walk = (sorted[0] - sorted[1]) / largest;
        }
        return (boundary, walk);
    }

    /// <summary>Whether either of the slot's decisions was a near-tie (<see cref="NearTie"/>).</summary>
    public static bool IsNearTie((double Boundary, double Walk) m) => m.Boundary <= NearTie || m.Walk <= NearTie;

    /// <summary>The slot as both drafts saw it: each one's candidates and scores, and the head's logits around the boundary.</summary>
    public static string Describe(string nameA, ModelBase.DFlashSlotTrace a, string nameB, ModelBase.DFlashSlotTrace b)
    {
        var text = new StringBuilder();
        foreach ((string name, ModelBase.DFlashSlotTrace t) in new[] { (nameA, a), (nameB, b) })
        {
            if (t == null) continue;
            text.Append(CultureInfo.InvariantCulture, $"{name}: ");
            if (t.Candidates != null)
                text.Append("candidates ").Append(string.Join(",", t.Candidates.Zip(t.Scores ?? new float[t.Candidates.Length],
                    (c, s) => string.Create(CultureInfo.InvariantCulture, $"{c}:{s:R}"))));
            if (t.Logits != null)
            {
                int k = t.Candidates?.Length ?? 1;
                var order = Enumerable.Range(0, t.Logits.Length).OrderByDescending(i => t.Logits[i]).Take(k + 2);
                text.Append(" head top ").Append(k + 2).Append(' ').Append(string.Join(",", order.Select(i =>
                    string.Create(CultureInfo.InvariantCulture, $"{i}:{t.Logits[i]:R}"))));
            }
            (double boundary, double walk) = Margins(t);
            text.Append(CultureInfo.InvariantCulture, $" margins boundary {boundary:E2} walk {walk:E2}; ");
        }
        return text.ToString();
    }
}
