// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Globalization;
using System.Text.RegularExpressions;

namespace TensorSharp.TestMatrix.Runners;

/// <summary>
/// Parses metrics out of TensorSharp.Cli structured log output. The CLI emits
/// regular lines like:
///   <c>Loaded model ... elapsedMs=12345.6</c>
///   <c>cli.inference prefill complete: tokens=512 ms=200.0 tokensPerSec=2560.0</c>
///   <c>cli.inference decode complete: tokens=128 ms=2000.0 tokensPerSec=64.0</c>
///   <c>benchmark summary: bestPrefillMs=200 bestPrefillTps=2560 bestDecodeMs=2000 bestDecodeTps=64 ...</c>
/// </summary>
public static partial class OutputParser
{
    // Numeric token used in the CLI logs: integer/float OR the literal "NaN"
    // (the CLI prints NaN for the decode TPS of a prefill-only benchmark and
    // for the prefill TPS of a decode-only one).
    private const string NumPattern = "(?:[0-9.]+|NaN)";

    [GeneratedRegex(@"Loaded model .*? elapsedMs=(?<load_ms>" + NumPattern + ")", RegexOptions.Compiled)]
    private static partial Regex LoadRegex();

    [GeneratedRegex(@"prefill complete:\s*tokens=(?<tok>\d+)\s*ms=(?<ms>" + NumPattern + @")\s*tokensPerSec=(?<tps>" + NumPattern + ")", RegexOptions.Compiled)]
    private static partial Regex PrefillRegex();

    [GeneratedRegex(@"decode complete:\s*tokens=(?<tok>\d+)\s*ms=(?<ms>" + NumPattern + @")\s*tokensPerSec=(?<tps>" + NumPattern + ")", RegexOptions.Compiled)]
    private static partial Regex DecodeRegex();

    [GeneratedRegex(@"benchmark summary:\s*bestPrefillMs=(?<pms>" + NumPattern + @")\s*bestPrefillTps=(?<pps>" + NumPattern + @")\s*bestDecodeMs=(?<dms>" + NumPattern + @")\s*bestDecodeTps=(?<dps>" + NumPattern + ")", RegexOptions.Compiled)]
    private static partial Regex BenchmarkSummaryRegex();

    public sealed record ParsedMetrics(
        double ModelLoadMs,
        int PrefillTokens,
        double PrefillMs,
        double PrefillTps,
        int DecodeTokens,
        double DecodeMs,
        double DecodeTps);

    public static ParsedMetrics Parse(string combinedStdoutStderr)
    {
        double loadMs = 0;
        int prefillTok = 0;
        double prefillMs = 0;
        double prefillTps = 0;
        int decodeTok = 0;
        double decodeMs = 0;
        double decodeTps = 0;

        Match m = LoadRegex().Match(combinedStdoutStderr);
        if (m.Success)
        {
            loadMs = ParseDouble(m.Groups["load_ms"].Value);
        }

        // Inference-mode parser: a single prefill+decode pair.
        Match pf = PrefillRegex().Match(combinedStdoutStderr);
        if (pf.Success)
        {
            prefillTok = int.Parse(pf.Groups["tok"].Value, CultureInfo.InvariantCulture);
            prefillMs = ParseDouble(pf.Groups["ms"].Value);
            prefillTps = ParseDouble(pf.Groups["tps"].Value);
        }
        Match df = DecodeRegex().Match(combinedStdoutStderr);
        if (df.Success)
        {
            decodeTok = int.Parse(df.Groups["tok"].Value, CultureInfo.InvariantCulture);
            decodeMs = ParseDouble(df.Groups["ms"].Value);
            decodeTps = ParseDouble(df.Groups["tps"].Value);
        }

        // Benchmark-mode summary overrides inference parsing.
        Match bench = BenchmarkSummaryRegex().Match(combinedStdoutStderr);
        if (bench.Success)
        {
            prefillMs = ParseDouble(bench.Groups["pms"].Value);
            prefillTps = ParseDouble(bench.Groups["pps"].Value);
            decodeMs = ParseDouble(bench.Groups["dms"].Value);
            decodeTps = ParseDouble(bench.Groups["dps"].Value);
        }

        return new ParsedMetrics(loadMs, prefillTok, prefillMs, prefillTps, decodeTok, decodeMs, decodeTps);
    }

    private static double ParseDouble(string s)
    {
        // NaN means "metric is not applicable to this run" (e.g. decode TPS on
        // a prefill-only benchmark). Treat it as 0 so consumers don't have to
        // special-case it.
        if (string.Equals(s, "NaN", StringComparison.OrdinalIgnoreCase)) return 0.0;
        return double.TryParse(s, NumberStyles.Float, CultureInfo.InvariantCulture, out double v) ? v : 0.0;
    }
}
