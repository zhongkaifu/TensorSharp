// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Globalization;
using System.Text;
using TensorSharp.TestMatrix.Matrix;

namespace TensorSharp.TestMatrix.Reporting;

public sealed class MarkdownReportGenerator
{
    private const string NA = "n/a";
    private const string Fail = "fail";

    private readonly IReadOnlyList<TestResult> _results;
    private readonly IReadOnlyList<SkippedCombination> _skipped;
    private readonly DateTimeOffset _generatedAt;
    private readonly string _hostInfo;
    private readonly RegressionSummary? _regressions;

    public MarkdownReportGenerator(
        IReadOnlyList<TestResult> results,
        IReadOnlyList<SkippedCombination> skipped,
        DateTimeOffset generatedAt,
        string hostInfo,
        RegressionSummary? regressions = null)
    {
        _results = results;
        _skipped = skipped;
        _generatedAt = generatedAt;
        _hostInfo = hostInfo;
        _regressions = regressions;
    }

    public string Generate()
    {
        var sb = new StringBuilder();
        AppendHeader(sb);
        AppendSummary(sb);
        AppendRegressions(sb);
        AppendModelSections(sb);
        AppendEnvVarSensitivity(sb);
        AppendSkipped(sb);
        AppendCorrectnessFailures(sb);
        AppendRuntimeFailures(sb);
        AppendInputOutputSamples(sb);
        AppendAppendix(sb);
        return sb.ToString();
    }

    private void AppendHeader(StringBuilder sb)
    {
        sb.AppendLine("# TensorSharp Test & Benchmark Matrix");
        sb.AppendLine();
        sb.AppendLine($"Generated: `{_generatedAt:O}`");
        sb.AppendLine();
        sb.AppendLine($"Host: `{_hostInfo}`");
        sb.AppendLine();
        sb.AppendLine("Reference: [`docs/env_var_feature_matrix.md`](docs/env_var_feature_matrix.md) for the curated env var / feature matrix.");
        sb.AppendLine();
    }

    private void AppendSummary(StringBuilder sb)
    {
        int total = _results.Count;
        int ok = _results.Count(r => r.Ok);
        int runtimeFailed = _results.Count(r => !r.Ok && !r.Skipped && r.CorrectnessOk != false);
        int correctnessFailed = _results.Count(r => r.CorrectnessOk == false);
        int correctnessChecked = _results.Count(r => r.CorrectnessOk.HasValue);
        int correctnessPassed = _results.Count(r => r.CorrectnessOk == true);
        int skipped = _results.Count(r => r.Skipped) + _skipped.Count;
        double passRate = total == 0 ? 0 : (ok / (double)total) * 100;
        double correctnessRate = correctnessChecked == 0 ? 0 : (correctnessPassed / (double)correctnessChecked) * 100;

        sb.AppendLine("## Summary");
        sb.AppendLine();
        sb.AppendLine("| Metric | Value |");
        sb.AppendLine("|---|---|");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Total cases executed | {total} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Passed (runtime + correctness) | {ok} ({passRate:F1}%) |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Runtime failures | {runtimeFailed} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Correctness failures (output mismatch) | {correctnessFailed} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Correctness pass rate | {correctnessPassed} / {correctnessChecked} ({correctnessRate:F1}%) |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Skipped (capability / availability) | {skipped} |");
        sb.AppendLine();
    }

    private void AppendRegressions(StringBuilder sb)
    {
        if (_regressions is null)
        {
            return;
        }

        sb.AppendLine("## Regressions vs baseline");
        sb.AppendLine();
        if (_regressions.BaselinePath is null && _regressions.BaselineCapturedAt is null)
        {
            sb.AppendLine("_No baseline available for this host._ All cells are reported as `untracked`. Capture a baseline by running this matrix with `--update-baseline` and committing the file under `TensorSharp.TestMatrix/Baselines/`.");
            sb.AppendLine();
            return;
        }

        sb.AppendLine(CultureInfo.InvariantCulture,
            $"Baseline: `{_regressions.BaselinePath}` (host `{_regressions.HostLabel}`, captured {_regressions.BaselineCapturedAt:O}, commit `{_regressions.BaselineCommit ?? "?"}`).");
        sb.AppendLine();
        sb.AppendLine("| Severity | Count |");
        sb.AppendLine("|---|---:|");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| New runtime failure | {_regressions.RuntimeRegressions} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| New correctness failure | {_regressions.CorrectnessRegressions} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Throughput regression (> {_regressions.ThresholdPct:F1}%) | {_regressions.ThroughputRegressions} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Improvement (failing → passing) | {_regressions.Improvements} |");
        sb.AppendLine(CultureInfo.InvariantCulture, $"| Untracked (no baseline entry) | {_regressions.Untracked} |");
        sb.AppendLine();

        var blocking = _regressions.Findings
            .Where(f => f.Kind is RegressionKind.NewRuntimeFailure
                              or RegressionKind.NewCorrectnessFailure
                              or RegressionKind.ThroughputRegression)
            .ToList();
        if (blocking.Count > 0)
        {
            sb.AppendLine("**Blocking findings** (will fail `--fail-on-regression`):");
            sb.AppendLine();
            sb.AppendLine("| Case | Kind | Baseline | Current | Δ% | Description |");
            sb.AppendLine("|---|---|---:|---:|---:|---|");
            foreach (RegressionFinding f in blocking)
            {
                string baseStr = f.Kind == RegressionKind.NewRuntimeFailure || f.Kind == RegressionKind.NewCorrectnessFailure
                    ? "ok"
                    : f.BaselineValue.ToString("F1", CultureInfo.InvariantCulture);
                string curStr = f.Kind == RegressionKind.NewRuntimeFailure || f.Kind == RegressionKind.NewCorrectnessFailure
                    ? "fail"
                    : f.CurrentValue.ToString("F1", CultureInfo.InvariantCulture);
                sb.AppendLine(CultureInfo.InvariantCulture,
                    $"| `{f.CaseId}` | {KindLabel(f.Kind)} | {baseStr} | {curStr} | {f.DeltaPct:+0.0;-0.0;0.0}% | {EscapeCell(f.Description)} |");
            }
            sb.AppendLine();
        }

        var improvements = _regressions.Findings
            .Where(f => f.Kind == RegressionKind.NewImprovement).ToList();
        if (improvements.Count > 0)
        {
            sb.AppendLine($"**Improvements** ({improvements.Count}):");
            sb.AppendLine();
            foreach (RegressionFinding f in improvements.Take(20))
            {
                sb.AppendLine(CultureInfo.InvariantCulture, $"- `{f.CaseId}` — {EscapeCell(f.Description)}");
            }
            if (improvements.Count > 20)
            {
                sb.AppendLine(CultureInfo.InvariantCulture, $"- _… and {improvements.Count - 20} more_");
            }
            sb.AppendLine();
        }

        int untracked = _regressions.Untracked;
        if (untracked > 0)
        {
            sb.AppendLine($"_{untracked} cell(s) had no baseline entry. If they should be tracked, regenerate the baseline._");
            sb.AppendLine();
        }
    }

    private static string KindLabel(RegressionKind k) => k switch
    {
        RegressionKind.NewRuntimeFailure => "runtime regression",
        RegressionKind.NewCorrectnessFailure => "correctness regression",
        RegressionKind.ThroughputRegression => "throughput regression",
        RegressionKind.NewImprovement => "improvement",
        RegressionKind.UntrackedNewCase => "untracked",
        _ => k.ToString(),
    };

    private void AppendModelSections(StringBuilder sb)
    {
        var byModel = _results
            .Where(r => !r.Skipped)
            .GroupBy(r => r.Model, StringComparer.Ordinal)
            .OrderBy(g => g.Key, StringComparer.Ordinal);

        foreach (var modelGroup in byModel)
        {
            string modelId = modelGroup.Key;
            string family = modelGroup.First().ModelFamily;
            sb.AppendLine(CultureInfo.InvariantCulture, $"## Model: `{modelId}` (family `{family}`)");
            sb.AppendLine();

            // Baseline matrix: rows = features, columns = backends. Cells = prefill_tps / decode_tps.
            var baselines = modelGroup.Where(r => r.EnvVar is null).ToList();
            if (baselines.Count > 0)
            {
                sb.AppendLine("### Baseline (no env-var sweep)");
                sb.AppendLine();
                sb.AppendLine("Tokens per second (prefill / decode). `fail` = engine errored, `—` = metric not applicable.");
                sb.AppendLine();

                IEnumerable<string> backends = baselines.Select(r => r.Backend).Distinct(StringComparer.Ordinal).OrderBy(b => b, StringComparer.Ordinal);
                IEnumerable<string> features = baselines.Select(r => r.Feature).Distinct(StringComparer.Ordinal)
                    .OrderBy(f => FeatureOrderIndex(f));

                sb.Append("| Feature \\ Backend |");
                foreach (string b in backends)
                {
                    sb.Append(' ').Append(b).Append(" |");
                }
                sb.AppendLine();
                sb.Append("|---|");
                foreach (string _ in backends) sb.Append("---|");
                sb.AppendLine();

                foreach (string feat in features)
                {
                    sb.Append(CultureInfo.InvariantCulture, $"| `{feat}` |");
                    foreach (string b in backends)
                    {
                        TestResult? r = baselines.FirstOrDefault(x => x.Backend == b && x.Feature == feat);
                        sb.Append(' ').Append(FormatTpsCell(r, feat)).Append(" |");
                    }
                    sb.AppendLine();
                }
                sb.AppendLine();
            }

            // Env-var sweeps per backend.
            var sweepsByBackend = modelGroup
                .Where(r => r.EnvVar is not null)
                .GroupBy(r => r.Backend, StringComparer.Ordinal)
                .OrderBy(g => g.Key, StringComparer.Ordinal);

            foreach (var backendGroup in sweepsByBackend)
            {
                string backend = backendGroup.Key;
                sb.AppendLine(CultureInfo.InvariantCulture, $"### Env-var sweeps on `{backend}`");
                sb.AppendLine();

                var envVars = backendGroup
                    .GroupBy(r => r.EnvVar!, StringComparer.Ordinal)
                    .OrderBy(g => g.Key, StringComparer.Ordinal);

                foreach (var envGroup in envVars)
                {
                    string envName = envGroup.Key;
                    sb.AppendLine(CultureInfo.InvariantCulture, $"**`{envName}`** — decode tokens/sec per feature × value (baseline shown for delta):");
                    sb.AppendLine();

                    var values = envGroup.Select(r => r.EnvValue!).Distinct(StringComparer.Ordinal)
                        .OrderBy(v => v, StringComparer.Ordinal).ToList();
                    var features = envGroup.Select(r => r.Feature).Distinct(StringComparer.Ordinal)
                        .OrderBy(f => FeatureOrderIndex(f)).ToList();

                    sb.Append("| Feature | baseline |");
                    foreach (string v in values) sb.Append(' ').Append(v).Append(" |");
                    sb.AppendLine(" Best |");
                    sb.Append("|---|---|");
                    foreach (string _ in values) sb.Append("---|");
                    sb.AppendLine("---|");

                    foreach (string feat in features)
                    {
                        TestResult? baseline = _results.FirstOrDefault(r =>
                            r.Model == modelId && r.Backend == backend && r.Feature == feat && r.EnvVar is null);
                        sb.Append(CultureInfo.InvariantCulture, $"| `{feat}` | {FormatDecodeTps(baseline, feat)} |");
                        double best = baseline?.DecodeTps ?? 0;
                        string bestLabel = "baseline";
                        foreach (string v in values)
                        {
                            TestResult? cell = envGroup.FirstOrDefault(r =>
                                r.Feature == feat && r.EnvValue == v);
                            sb.Append(' ').Append(FormatDecodeTps(cell, feat)).Append(" |");
                            if (cell is not null && cell.Ok && cell.DecodeTps > best)
                            {
                                best = cell.DecodeTps;
                                bestLabel = $"{envName}={v}";
                            }
                        }
                        sb.Append(' ').Append(bestLabel).AppendLine(" |");
                    }
                    sb.AppendLine();
                }
            }
        }
    }

    private void AppendEnvVarSensitivity(StringBuilder sb)
    {
        var sweeps = _results.Where(r => r.EnvVar is not null && r.Ok).ToList();
        if (sweeps.Count == 0)
        {
            return;
        }

        sb.AppendLine("## Env-var sensitivity overview");
        sb.AppendLine();
        sb.AppendLine("Largest decode-throughput delta produced by each env var across the whole matrix.");
        sb.AppendLine();
        sb.AppendLine("| Env var | Cells | Max decode TPS Δ (abs) | Max decode TPS Δ (%) | Best (model, backend, feature, value) |");
        sb.AppendLine("|---|---:|---:|---:|---|");

        var byEnv = sweeps.GroupBy(r => r.EnvVar!, StringComparer.Ordinal)
            .OrderBy(g => g.Key, StringComparer.Ordinal);
        foreach (var g in byEnv)
        {
            double maxDelta = 0;
            double maxPct = 0;
            string bestLabel = "—";
            int cellCount = 0;
            foreach (TestResult cell in g)
            {
                cellCount++;
                TestResult? baseline = _results.FirstOrDefault(r =>
                    r.Model == cell.Model && r.Backend == cell.Backend && r.Feature == cell.Feature && r.EnvVar is null && r.Ok);
                if (baseline is null) continue;
                double delta = cell.DecodeTps - baseline.DecodeTps;
                if (Math.Abs(delta) > Math.Abs(maxDelta))
                {
                    maxDelta = delta;
                    maxPct = baseline.DecodeTps > 0 ? (delta / baseline.DecodeTps) * 100 : 0;
                    bestLabel = $"{cell.Model} / {cell.Backend} / {cell.Feature} / {cell.EnvValue}";
                }
            }
            sb.Append(CultureInfo.InvariantCulture, $"| `{g.Key}` | {cellCount} | {maxDelta:+0.0;-0.0;0.0} | {maxPct:+0.0;-0.0;0.0}% | {bestLabel} |");
            sb.AppendLine();
        }
        sb.AppendLine();
    }

    private void AppendSkipped(StringBuilder sb)
    {
        if (_skipped.Count == 0)
        {
            return;
        }
        sb.AppendLine("## Skipped combinations");
        sb.AppendLine();
        sb.AppendLine("| Model | Backend | Feature | Reason |");
        sb.AppendLine("|---|---|---|---|");
        foreach (var s in _skipped.OrderBy(x => x.Model.Id, StringComparer.Ordinal)
                                  .ThenBy(x => x.Backend.Id, StringComparer.Ordinal)
                                  .ThenBy(x => x.Feature.Id, StringComparer.Ordinal))
        {
            sb.AppendLine(CultureInfo.InvariantCulture,
                $"| `{s.Model.Id}` | `{s.Backend.Id}` | `{s.Feature.Id}` | {s.Reason} |");
        }
        sb.AppendLine();
    }

    private void AppendCorrectnessFailures(StringBuilder sb)
    {
        var failures = _results.Where(r => r.CorrectnessOk == false).ToList();
        if (failures.Count == 0)
        {
            return;
        }
        sb.AppendLine("## Correctness failures");
        sb.AppendLine();
        sb.AppendLine("The process ran, produced metrics, and exited cleanly — but the output did not contain all required substrings configured on the feature.");
        sb.AppendLine();
        sb.AppendLine("| Case | Expected | Output preview |");
        sb.AppendLine("|---|---|---|");
        foreach (TestResult f in failures.OrderBy(x => x.CaseId, StringComparer.Ordinal))
        {
            string expected = f.CorrectnessExpected is null
                ? "(none)"
                : string.Join(", ", f.CorrectnessExpected.Select(s => "`" + EscapeCell(s) + "`"));
            string preview = EscapeCell(f.OutputPreview);
            if (preview.Length > 200) preview = preview[..200] + "…";
            sb.AppendLine(CultureInfo.InvariantCulture,
                $"| `{f.CaseId}` | {expected} | {preview} |");
        }
        sb.AppendLine();
    }

    private void AppendRuntimeFailures(StringBuilder sb)
    {
        // Anything that failed without being a correctness failure: process
        // crash, timeout, no parseable metrics, missing input file, etc.
        var failures = _results
            .Where(r => !r.Ok && !r.Skipped && r.CorrectnessOk != false)
            .ToList();
        if (failures.Count == 0)
        {
            return;
        }
        sb.AppendLine("## Runtime failures");
        sb.AppendLine();
        sb.AppendLine("| Case | Exit | Error |");
        sb.AppendLine("|---|---:|---|");
        foreach (TestResult f in failures.OrderBy(x => x.CaseId, StringComparer.Ordinal))
        {
            string err = EscapeCell(f.Error ?? "");
            sb.AppendLine(CultureInfo.InvariantCulture,
                $"| `{f.CaseId}` | {f.ExitCode} | {err} |");
        }
        sb.AppendLine();
    }

    private static string EscapeCell(string s)
    {
        return s.Replace('|', '/').Replace('\n', ' ').Replace('\r', ' ');
    }

    private void AppendInputOutputSamples(StringBuilder sb)
    {
        // For each (model, feature) pair, pick the baseline cell on each
        // backend and show the input + output. This keeps the section
        // navigable (no env-var sweep blowup) while letting a reader audit
        // what every case actually sent in and got back.
        var samples = _results
            .Where(r => r.EnvVar is null && !r.Skipped)
            .OrderBy(r => r.Model, StringComparer.Ordinal)
            .ThenBy(r => FeatureOrderIndex(r.Feature))
            .ThenBy(r => r.Backend, StringComparer.Ordinal)
            .ToList();

        if (samples.Count == 0)
        {
            return;
        }

        sb.AppendLine("## Inputs and outputs (per case)");
        sb.AppendLine();
        sb.AppendLine("Inputs sent to the model and the model's response, one collapsible block per cell. Use this to manually verify a case behaved as expected without re-running it.");
        sb.AppendLine();

        foreach (TestResult r in samples)
        {
            string status = r.Ok ? "ok"
                : r.CorrectnessOk == false ? "correctness-fail"
                : r.Skipped ? "skip"
                : "fail";
            sb.AppendLine(CultureInfo.InvariantCulture,
                $"<details><summary><code>{r.Model} / {r.Backend} / {r.Feature}</code> — {status}</summary>");
            sb.AppendLine();

            sb.AppendLine("**Input**");
            sb.AppendLine();
            if (!string.IsNullOrEmpty(r.Input.PromptPath))
            {
                sb.AppendLine(CultureInfo.InvariantCulture, $"- Prompt file: `{r.Input.PromptPath}`");
            }
            if (!string.IsNullOrEmpty(r.Input.MediaPath))
            {
                sb.AppendLine(CultureInfo.InvariantCulture, $"- Media file: `{r.Input.MediaPath}`");
            }
            sb.AppendLine(CultureInfo.InvariantCulture, $"- Max tokens: {r.Input.MaxTokens}");
            if (r.Input.ThinkingEnabled)
            {
                sb.AppendLine("- Thinking mode: enabled (`--think`)");
            }
            if (r.Input.Env.Count > 0)
            {
                sb.AppendLine(CultureInfo.InvariantCulture,
                    $"- Environment: {string.Join(", ", r.Input.Env.Select(e => $"`{e.Key}={e.Value}`"))}");
            }
            sb.AppendLine();
            if (!string.IsNullOrEmpty(r.Input.PromptText))
            {
                sb.AppendLine("```text");
                sb.AppendLine(r.Input.PromptText);
                sb.AppendLine("```");
                sb.AppendLine();
            }
            if (!string.IsNullOrEmpty(r.Input.ToolsText))
            {
                sb.AppendLine("Tool definitions:");
                sb.AppendLine();
                sb.AppendLine("```json");
                sb.AppendLine(r.Input.ToolsText);
                sb.AppendLine("```");
                sb.AppendLine();
            }
            if (!string.IsNullOrEmpty(r.Input.MultiTurnText))
            {
                sb.AppendLine("Multi-turn transcript:");
                sb.AppendLine();
                sb.AppendLine("```jsonl");
                sb.AppendLine(r.Input.MultiTurnText);
                sb.AppendLine("```");
                sb.AppendLine();
            }

            sb.AppendLine("**Output**");
            sb.AppendLine();
            sb.AppendLine(CultureInfo.InvariantCulture, $"- Decode tokens: {r.DecodeTokens}");
            sb.AppendLine(CultureInfo.InvariantCulture, $"- Prefill TPS: {(r.PrefillTps > 0 ? r.PrefillTps.ToString("F1", CultureInfo.InvariantCulture) : "—")} / decode TPS: {(r.DecodeTps > 0 ? r.DecodeTps.ToString("F1", CultureInfo.InvariantCulture) : "—")}");
            sb.AppendLine(CultureInfo.InvariantCulture, $"- Wall time: {(r.TotalWallMs / 1000):F1}s");
            if (r.CorrectnessExpected is { Length: > 0 })
            {
                sb.AppendLine(CultureInfo.InvariantCulture,
                    $"- Expected substrings: {string.Join(", ", r.CorrectnessExpected.Select(s => "`" + EscapeCell(s) + "`"))}  → {(r.CorrectnessOk == true ? "matched" : r.CorrectnessOk == false ? "**missed**" : "n/a")}");
            }
            sb.AppendLine();
            if (!string.IsNullOrEmpty(r.Output.AssistantText))
            {
                sb.AppendLine("```text");
                sb.AppendLine(r.Output.AssistantText);
                if (r.Output.AssistantTextTruncated)
                {
                    sb.AppendLine("[... truncated; see results/<case-id>.json for the full text ...]");
                }
                sb.AppendLine("```");
            }
            else
            {
                sb.AppendLine("_(no assistant text produced — synthetic benchmark mode or the run failed before generation)_");
            }
            sb.AppendLine();
            sb.AppendLine("</details>");
            sb.AppendLine();
        }
    }

    private void AppendAppendix(StringBuilder sb)
    {
        sb.AppendLine("## Appendix: raw cell index");
        sb.AppendLine();
        sb.AppendLine("Per-case JSON files (one per row above): `results/<case-id>.json`. Each JSON file contains the full input (prompt / tools / multi-turn), the full assistant output, and all metrics.");
        sb.AppendLine();
    }

    private static string FormatTpsCell(TestResult? r, string featureId)
    {
        if (r is null) return NA;
        if (r.Skipped) return $"skip ({r.SkipReason})";
        if (!r.Ok) return Fail;
        bool prefillOnly = featureId is "pp512" or "pp2048";
        bool decodeOnly = featureId is "tg128";
        string prefill = prefillOnly || (!decodeOnly && r.PrefillTps > 0)
            ? FormatTps(r.PrefillTps)
            : "—";
        string decode = decodeOnly || (!prefillOnly && r.DecodeTps > 0)
            ? FormatTps(r.DecodeTps)
            : "—";
        if (prefillOnly) return prefill;
        if (decodeOnly) return decode;
        return $"{prefill} / {decode}";
    }

    private static string FormatDecodeTps(TestResult? r, string featureId)
    {
        if (r is null) return NA;
        if (r.Skipped) return "skip";
        if (!r.Ok) return Fail;
        if (featureId is "pp512" or "pp2048") return FormatTps(r.PrefillTps);
        return FormatTps(r.DecodeTps);
    }

    private static string FormatTps(double v)
    {
        if (v <= 0) return "—";
        return v.ToString("F1", CultureInfo.InvariantCulture);
    }

    private static int FeatureOrderIndex(string id)
    {
        for (int i = 0; i < FeatureCatalog.All.Count; i++)
        {
            if (FeatureCatalog.All[i].Id == id) return i;
        }
        return 1000;
    }
}
