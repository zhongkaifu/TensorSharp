// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

namespace TensorSharp.TestMatrix.Reporting;

public enum RegressionKind
{
    /// <summary>Was passing in the baseline, now fails (runtime crash, timeout, no metrics).</summary>
    NewRuntimeFailure,
    /// <summary>Was passing in the baseline, now fails the per-feature output check.</summary>
    NewCorrectnessFailure,
    /// <summary>Was passing in the baseline, throughput dropped by more than the threshold.</summary>
    ThroughputRegression,
    /// <summary>Was failing in the baseline, now passes — informational.</summary>
    NewImprovement,
    /// <summary>Cell exists in the current run but has no baseline counterpart yet.</summary>
    UntrackedNewCase,
}

public sealed record RegressionFinding(
    string CaseId,
    RegressionKind Kind,
    string Description,
    double BaselineValue,
    double CurrentValue,
    double DeltaPct);

public sealed record RegressionSummary(
    IReadOnlyList<RegressionFinding> Findings,
    string HostLabel,
    string? BaselinePath,
    DateTimeOffset? BaselineCapturedAt,
    string? BaselineCommit,
    double ThresholdPct)
{
    public int RuntimeRegressions => Findings.Count(f => f.Kind == RegressionKind.NewRuntimeFailure);
    public int CorrectnessRegressions => Findings.Count(f => f.Kind == RegressionKind.NewCorrectnessFailure);
    public int ThroughputRegressions => Findings.Count(f => f.Kind == RegressionKind.ThroughputRegression);
    public int Improvements => Findings.Count(f => f.Kind == RegressionKind.NewImprovement);
    public int Untracked => Findings.Count(f => f.Kind == RegressionKind.UntrackedNewCase);

    /// <summary>True if there is at least one regression that should fail the PR gate.</summary>
    public bool HasBlockingRegressions => RuntimeRegressions > 0 || CorrectnessRegressions > 0 || ThroughputRegressions > 0;
}

/// <summary>
/// Compares the current run's results against a baseline snapshot and
/// emits a list of regression findings ordered by severity.
/// </summary>
public static class RegressionAnalyzer
{
    public static RegressionSummary Analyze(
        IReadOnlyList<TestResult> currentResults,
        BaselineFile? baseline,
        double thresholdPct,
        string hostLabel,
        string? baselinePath)
    {
        var findings = new List<RegressionFinding>();
        if (baseline is null)
        {
            // No baseline available: every current case is untracked. We still
            // return a summary so the report can surface the absence and the
            // workflow can decide what to do (typically: do nothing in PRs,
            // capture a baseline manually).
            foreach (TestResult r in currentResults)
            {
                if (r.Skipped) continue;
                findings.Add(new RegressionFinding(
                    r.CaseId,
                    RegressionKind.UntrackedNewCase,
                    "no baseline entry for this case",
                    BaselineValue: 0,
                    CurrentValue: r.DecodeTps,
                    DeltaPct: 0));
            }
            return new RegressionSummary(findings, hostLabel, baselinePath, null, null, thresholdPct);
        }

        var byId = baseline.Entries.ToDictionary(e => e.CaseId, e => e, StringComparer.Ordinal);

        foreach (TestResult r in currentResults)
        {
            if (r.Skipped)
            {
                continue;
            }
            if (!byId.TryGetValue(r.CaseId, out BaselineEntry? b))
            {
                findings.Add(new RegressionFinding(
                    r.CaseId,
                    RegressionKind.UntrackedNewCase,
                    "no baseline entry for this case",
                    BaselineValue: 0,
                    CurrentValue: r.DecodeTps,
                    DeltaPct: 0));
                continue;
            }

            // Was passing, now broken?
            if (b.Ok && !r.Ok)
            {
                if (r.CorrectnessOk == false)
                {
                    findings.Add(new RegressionFinding(
                        r.CaseId,
                        RegressionKind.NewCorrectnessFailure,
                        r.CorrectnessError ?? "output did not match expected substrings",
                        BaselineValue: 1,
                        CurrentValue: 0,
                        DeltaPct: -100));
                }
                else
                {
                    findings.Add(new RegressionFinding(
                        r.CaseId,
                        RegressionKind.NewRuntimeFailure,
                        r.Error ?? "runtime failure",
                        BaselineValue: 1,
                        CurrentValue: 0,
                        DeltaPct: -100));
                }
                continue;
            }

            // Was failing, now passing?
            if (!b.Ok && r.Ok)
            {
                findings.Add(new RegressionFinding(
                    r.CaseId,
                    RegressionKind.NewImprovement,
                    "case previously failing, now passing",
                    BaselineValue: 0,
                    CurrentValue: 1,
                    DeltaPct: 100));
                continue;
            }

            // Both passing — check throughput regression on whichever metric
            // is meaningful for this cell. Decode TPS is the primary signal
            // for inference cases; prefill TPS for prefill-only synthetic
            // benchmarks; model_load_ms is informational only and not gated.
            if (b.Ok && r.Ok)
            {
                double baseTps;
                double curTps;
                string metric;
                if (r.DecodeTps > 0 || b.DecodeTps > 0)
                {
                    baseTps = b.DecodeTps;
                    curTps = r.DecodeTps;
                    metric = "decode TPS";
                }
                else
                {
                    baseTps = b.PrefillTps;
                    curTps = r.PrefillTps;
                    metric = "prefill TPS";
                }
                if (baseTps <= 0)
                {
                    continue;
                }
                double deltaPct = ((curTps - baseTps) / baseTps) * 100.0;
                if (deltaPct < -thresholdPct)
                {
                    findings.Add(new RegressionFinding(
                        r.CaseId,
                        RegressionKind.ThroughputRegression,
                        $"{metric} dropped {-deltaPct:F1}% (threshold {thresholdPct:F1}%)",
                        BaselineValue: baseTps,
                        CurrentValue: curTps,
                        DeltaPct: deltaPct));
                }
            }
        }

        // Severity ordering: runtime > correctness > throughput > improvement > untracked
        static int Rank(RegressionKind k) => k switch
        {
            RegressionKind.NewRuntimeFailure => 0,
            RegressionKind.NewCorrectnessFailure => 1,
            RegressionKind.ThroughputRegression => 2,
            RegressionKind.NewImprovement => 3,
            RegressionKind.UntrackedNewCase => 4,
            _ => 5,
        };
        findings.Sort((a, b) =>
        {
            int byKind = Rank(a.Kind).CompareTo(Rank(b.Kind));
            if (byKind != 0) return byKind;
            return string.CompareOrdinal(a.CaseId, b.CaseId);
        });

        return new RegressionSummary(
            findings,
            hostLabel,
            baselinePath,
            baseline.CapturedAt,
            baseline.Commit,
            thresholdPct);
    }
}
