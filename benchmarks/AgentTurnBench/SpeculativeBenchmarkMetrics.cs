#nullable enable
using System;
using System.Collections.Generic;
using TensorSharp.Runtime.Speculative;

internal enum SpeculativeRunOrder { PlainFirst, SpecFirst }
internal enum DiagnosticPrefillMode { Refill, Forward }

internal static class SpeculativeDiagnosticPrefill
{
    internal static DiagnosticPrefillMode Parse(string value) => value switch
    {
        "refill" => DiagnosticPrefillMode.Refill,
        "forward" => DiagnosticPrefillMode.Forward,
        _ => throw new ArgumentException("--spec-diagnostic-prefill must be refill or forward"),
    };

    internal static float[] Run(DiagnosticPrefillMode mode, int[] tokens,
        Func<int[], float[]> forward, Func<int[], float[]> refill) => mode switch
    {
        DiagnosticPrefillMode.Refill => refill(tokens),
        DiagnosticPrefillMode.Forward => forward(tokens),
        _ => throw new ArgumentOutOfRangeException(nameof(mode)),
    };
}

internal static class SpeculativeBenchmarkOrdering
{
    internal static SpeculativeRunOrder Parse(string value) => value switch
    {
        "plain-first" => SpeculativeRunOrder.PlainFirst,
        "spec-first" => SpeculativeRunOrder.SpecFirst,
        _ => throw new ArgumentException("--spec-order must be plain-first or spec-first"),
    };

    // Both full-scenario warmup and measurement use this same ordering, with
    // exactly one plain arm and one of each selected speculative candidate.
    internal static IEnumerable<T> Arrange<T>(SpeculativeRunOrder order, T plain, IEnumerable<T> candidates)
    {
        if (order is not (SpeculativeRunOrder.PlainFirst or SpeculativeRunOrder.SpecFirst))
            throw new ArgumentOutOfRangeException(nameof(order));
        if (order == SpeculativeRunOrder.PlainFirst) yield return plain;
        foreach (T candidate in candidates) yield return candidate;
        if (order == SpeculativeRunOrder.SpecFirst) yield return plain;
    }
}

// Snapshot the public counters before the request/model can be reused. These
// values retain Stopwatch precision; the human-readable engine log rounds them.
// Governor rates are local calibration estimates, not independent control rates.
internal sealed record SpeculativePhaseTimings(double DraftMs, double VerifyMs, double SnapshotMs,
    double RollbackMs, double CatchUpMs, double PlainMs, double PlainMsPerToken, double SpecMsPerToken)
{
    internal static SpeculativePhaseTimings? From(SpeculationStats? stats) => stats == null ? null : new(
        stats.DraftMs, stats.VerifyMs, stats.SnapshotMs, stats.RollbackMs, stats.CatchUpMs, stats.PlainMs,
        stats.PlainMsPerToken, stats.SpecMsPerToken);
}
