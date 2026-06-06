// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using System.Text;
using TensorSharp.TestMatrix.Configuration;
using TensorSharp.TestMatrix.Matrix;
using TensorSharp.TestMatrix.Reporting;

namespace TensorSharp.TestMatrix.Runners;

/// <summary>
/// Executes a <see cref="TestCase"/> by spawning TensorSharp.Cli as a
/// subprocess, capturing stdout/stderr, parsing metrics, and producing a
/// <see cref="TestResult"/>.
/// </summary>
public sealed class CliRunner
{
    private readonly MatrixConfig _config;
    private readonly string _cliExecutable;
    private readonly string _inputsRoot;

    public CliRunner(MatrixConfig config, string cliExecutable, string inputsRoot)
    {
        _config = config;
        _cliExecutable = cliExecutable;
        _inputsRoot = inputsRoot;
    }

    public TestResult Run(TestCase testCase)
    {
        var result = new TestResult
        {
            CaseId = testCase.Id,
            Model = testCase.Model.Id,
            ModelFamily = testCase.Model.Family,
            Backend = testCase.Backend.Id,
            Feature = testCase.Feature.Id,
            EnvVar = testCase.EnvVar,
            EnvValue = testCase.EnvValue,
            StartedAt = DateTimeOffset.UtcNow,
        };

        List<string> args;
        try
        {
            args = BuildArgs(testCase, result.Input);
        }
        catch (Exception ex)
        {
            result.Skipped = true;
            result.SkipReason = $"could not build args: {ex.Message}";
            return result;
        }
        foreach ((string k, string v) in testCase.ExtraEnv)
        {
            result.Input.Env[k] = v;
        }

        var psi = new ProcessStartInfo
        {
            FileName = _cliExecutable,
            UseShellExecute = false,
            RedirectStandardOutput = true,
            RedirectStandardError = true,
            CreateNoWindow = true,
        };
        foreach (string a in args)
        {
            psi.ArgumentList.Add(a);
        }
        // Wipe any inherited TS_* env vars first so the matrix value is
        // authoritative; then apply the per-case set.
        ScrubInheritedTsEnv(psi);
        foreach ((string k, string v) in testCase.ExtraEnv)
        {
            psi.Environment[k] = v;
        }

        result.Command = FormatCommand(_cliExecutable, args, testCase.ExtraEnv);

        var stdout = new StringBuilder();
        var stderr = new StringBuilder();
        var sw = Stopwatch.StartNew();
        int exit;
        try
        {
            using var p = new Process { StartInfo = psi };
            p.OutputDataReceived += (_, e) => { if (e.Data is not null) stdout.AppendLine(e.Data); };
            p.ErrorDataReceived += (_, e) => { if (e.Data is not null) stderr.AppendLine(e.Data); };
            p.Start();
            p.BeginOutputReadLine();
            p.BeginErrorReadLine();
            if (!p.WaitForExit(_config.TimeoutSeconds * 1000))
            {
                try { p.Kill(entireProcessTree: true); } catch { }
                exit = 124;
                stderr.AppendLine($"\nTIMEOUT after {_config.TimeoutSeconds}s");
            }
            else
            {
                p.WaitForExit();
                exit = p.ExitCode;
            }
        }
        catch (Exception ex)
        {
            result.Ok = false;
            result.Error = $"process start failed: {ex.Message}";
            result.ExitCode = -1;
            return result;
        }
        sw.Stop();

        result.ExitCode = exit;
        result.TotalWallMs = sw.Elapsed.TotalMilliseconds;
        result.DurationSeconds = sw.Elapsed.TotalSeconds;

        string combined = stderr + "\n" + stdout;
        string[] lines = combined.Split('\n');
        result.StdoutTail = string.Join('\n', lines.Skip(Math.Max(0, lines.Length - 40)));

        if (exit != 0)
        {
            result.Ok = false;
            result.Error = $"exit {exit}";
            return result;
        }

        OutputParser.ParsedMetrics m = OutputParser.Parse(combined);
        result.ModelLoadMs = m.ModelLoadMs;
        result.PrefillTokens = m.PrefillTokens;
        result.PrefillMs = m.PrefillMs;
        result.PrefillTps = m.PrefillTps;
        result.DecodeTokens = m.DecodeTokens;
        result.DecodeMs = m.DecodeMs;
        result.DecodeTps = m.DecodeTps;

        string fullAssistantText = ExtractAssistantText(stdout.ToString());
        result.OutputPreview = Truncate(fullAssistantText, 500);

        // Persist the full (post-filter) assistant text in result.Output so
        // users can read what came out without re-running the case. Capped at
        // 8 KiB to keep the JSON readable; for synthetic benchmarks this is
        // typically empty since --benchmark doesn't stream tokens.
        const int MaxAssistantBytes = 8 * 1024;
        if (fullAssistantText.Length > MaxAssistantBytes)
        {
            result.Output.AssistantText = fullAssistantText[..MaxAssistantBytes];
            result.Output.AssistantTextTruncated = true;
        }
        else
        {
            result.Output.AssistantText = fullAssistantText;
            result.Output.AssistantTextTruncated = false;
        }
        result.Output.DecodeTokens = m.DecodeTokens;

        bool hasMetric = m.PrefillTps > 0 || m.DecodeTps > 0 || m.ModelLoadMs > 0;
        if (!hasMetric)
        {
            result.Ok = false;
            result.Error = "no metrics parsed from output";
            return result;
        }

        ApplyCorrectnessCheck(result, testCase.Feature, fullAssistantText);
        if (result.CorrectnessOk == false)
        {
            result.Ok = false;
            result.Error = result.CorrectnessError;
            return result;
        }
        result.Ok = true;
        return result;
    }

    private static void ApplyCorrectnessCheck(TestResult result, FeatureSpec feature, string assistantText)
    {
        if (feature.ExpectedContains.Count == 0)
        {
            // No semantic check configured for this feature; leave
            // CorrectnessOk as null so the report can tell the difference
            // between "not checked" and "checked and passed".
            return;
        }

        result.CorrectnessExpected = feature.ExpectedContains.ToArray();
        var missing = new List<string>();
        foreach (string needle in feature.ExpectedContains)
        {
            if (assistantText.IndexOf(needle, StringComparison.OrdinalIgnoreCase) < 0)
            {
                missing.Add(needle);
            }
        }
        if (missing.Count == 0)
        {
            result.CorrectnessOk = true;
            return;
        }
        result.CorrectnessOk = false;
        result.CorrectnessError = $"output missing required substring(s): {string.Join(", ", missing.Select(s => "'" + s + "'"))}";
    }

    private static string Truncate(string s, int max)
    {
        if (s.Length <= max) return s;
        return s[..max] + "...";
    }

    private List<string> BuildArgs(TestCase testCase, InputCapture capture)
    {
        var args = new List<string>
        {
            "--model", testCase.Model.GgufPath,
            "--backend", testCase.Backend.Id,
            "--log-level", "info",
            "--log-file", "off",
        };

        if (!string.IsNullOrEmpty(testCase.Model.MmprojPath))
        {
            args.AddRange(new[] { "--mmproj", testCase.Model.MmprojPath });
        }

        FeatureSpec f = testCase.Feature;
        capture.MaxTokens = f.MaxTokens;
        capture.ThinkingEnabled = f.EnableThinking;

        switch (f.Kind)
        {
            case FeatureKind.SyntheticPrefill:
                args.AddRange(new[]
                {
                    "--benchmark",
                    "--bench-prefill", f.PrefillTokens.ToString(CultureInfo.InvariantCulture),
                    "--bench-decode", "0",
                    "--bench-runs", _config.BenchmarkRuns.ToString(CultureInfo.InvariantCulture),
                });
                capture.PromptText = $"(synthetic benchmark: {f.PrefillTokens} prefill tokens, no real prompt)";
                break;
            case FeatureKind.SyntheticDecode:
                args.AddRange(new[]
                {
                    "--benchmark",
                    "--bench-prefill", f.PrefillTokens.ToString(CultureInfo.InvariantCulture),
                    "--bench-decode", f.DecodeTokens.ToString(CultureInfo.InvariantCulture),
                    "--bench-runs", _config.BenchmarkRuns.ToString(CultureInfo.InvariantCulture),
                });
                capture.PromptText = $"(synthetic benchmark: {f.PrefillTokens} prefill + {f.DecodeTokens} decode, no real prompt)";
                break;
            case FeatureKind.Text:
            case FeatureKind.UploadedText:
            {
                string promptPath = ResolveInput(f.PromptFile!);
                args.AddRange(new[]
                {
                    "--input", promptPath,
                    "--max-tokens", f.MaxTokens.ToString(CultureInfo.InvariantCulture),
                    "--temperature", "0",
                    "--warmup-runs", _config.WarmupRuns.ToString(CultureInfo.InvariantCulture),
                });
                capture.PromptPath = f.PromptFile;
                capture.PromptText = SafeRead(promptPath);
                break;
            }
            case FeatureKind.MultiTurn:
            {
                string mtPath = ResolveInput(f.MultiTurnFile!);
                args.AddRange(new[]
                {
                    "--multi-turn-jsonl", mtPath,
                    "--max-tokens", f.MaxTokens.ToString(CultureInfo.InvariantCulture),
                    "--temperature", "0",
                });
                capture.PromptPath = f.MultiTurnFile;
                capture.MultiTurnText = SafeRead(mtPath);
                break;
            }
            case FeatureKind.Tools:
            {
                string promptPath = ResolveInput(f.PromptFile!);
                string toolsPath = ResolveInput(f.ToolsFile!);
                args.AddRange(new[]
                {
                    "--input", promptPath,
                    "--tools", toolsPath,
                    "--max-tokens", f.MaxTokens.ToString(CultureInfo.InvariantCulture),
                    "--temperature", "0",
                    "--warmup-runs", _config.WarmupRuns.ToString(CultureInfo.InvariantCulture),
                });
                capture.PromptPath = f.PromptFile;
                capture.PromptText = SafeRead(promptPath);
                capture.ToolsText = SafeRead(toolsPath);
                break;
            }
            case FeatureKind.Thinking:
            {
                string promptPath = ResolveInput(f.PromptFile!);
                args.AddRange(new[]
                {
                    "--input", promptPath,
                    "--think",
                    "--max-tokens", f.MaxTokens.ToString(CultureInfo.InvariantCulture),
                    "--temperature", "0",
                    "--warmup-runs", _config.WarmupRuns.ToString(CultureInfo.InvariantCulture),
                });
                capture.PromptPath = f.PromptFile;
                capture.PromptText = SafeRead(promptPath);
                break;
            }
            case FeatureKind.Image:
            {
                string promptPath = ResolveInput(f.PromptFile!);
                string mediaPath = ResolveMedia(f.MediaFile!);
                args.AddRange(new[]
                {
                    "--input", promptPath,
                    "--image", mediaPath,
                    "--max-tokens", f.MaxTokens.ToString(CultureInfo.InvariantCulture),
                    "--temperature", "0",
                    "--warmup-runs", _config.WarmupRuns.ToString(CultureInfo.InvariantCulture),
                });
                capture.PromptPath = f.PromptFile;
                capture.PromptText = SafeRead(promptPath);
                capture.MediaPath = mediaPath;
                break;
            }
            case FeatureKind.Audio:
            {
                string promptPath = ResolveInput(f.PromptFile!);
                string mediaPath = ResolveMedia(f.MediaFile!);
                args.AddRange(new[]
                {
                    "--input", promptPath,
                    "--audio", mediaPath,
                    "--max-tokens", f.MaxTokens.ToString(CultureInfo.InvariantCulture),
                    "--temperature", "0",
                    "--warmup-runs", _config.WarmupRuns.ToString(CultureInfo.InvariantCulture),
                });
                capture.PromptPath = f.PromptFile;
                capture.PromptText = SafeRead(promptPath);
                capture.MediaPath = mediaPath;
                break;
            }
            case FeatureKind.Video:
            {
                string promptPath = ResolveInput(f.PromptFile!);
                string mediaPath = ResolveMedia(f.MediaFile!);
                args.AddRange(new[]
                {
                    "--input", promptPath,
                    "--video", mediaPath,
                    "--max-tokens", f.MaxTokens.ToString(CultureInfo.InvariantCulture),
                    "--temperature", "0",
                    "--warmup-runs", _config.WarmupRuns.ToString(CultureInfo.InvariantCulture),
                });
                capture.PromptPath = f.PromptFile;
                capture.PromptText = SafeRead(promptPath);
                capture.MediaPath = mediaPath;
                break;
            }
            default:
                throw new InvalidOperationException($"unhandled feature kind: {f.Kind}");
        }

        return args;
    }

    private static string? SafeRead(string path)
    {
        const int MaxBytes = 16 * 1024;
        try
        {
            byte[] bytes = File.ReadAllBytes(path);
            if (bytes.Length <= MaxBytes)
            {
                return System.Text.Encoding.UTF8.GetString(bytes);
            }
            // For oversized prompt files (e.g. user-pointed large uploads),
            // keep the first 8 KiB and the last 4 KiB so summarizers and tail
            // markers both survive.
            string head = System.Text.Encoding.UTF8.GetString(bytes, 0, 8 * 1024);
            string tail = System.Text.Encoding.UTF8.GetString(bytes, bytes.Length - 4 * 1024, 4 * 1024);
            return head + $"\n\n[... {bytes.Length - MaxBytes} bytes elided ...]\n\n" + tail;
        }
        catch
        {
            return null;
        }
    }

    private string ResolveInput(string relPath)
    {
        string full = Path.Combine(_inputsRoot, relPath);
        if (!File.Exists(full))
        {
            throw new FileNotFoundException($"input file not found: {full}", full);
        }
        return full;
    }

    private string ResolveMedia(string relPath)
    {
        // media can live under the configured media_dir if present (so users can
        // point at shared assets without copying them into the project tree).
        if (!string.IsNullOrWhiteSpace(_config.MediaDir))
        {
            string candidate = Path.Combine(_config.MediaDir, Path.GetFileName(relPath));
            if (File.Exists(candidate))
            {
                return candidate;
            }
        }
        return ResolveInput(relPath);
    }

    private const string GeneratedOutputMarker = "=== Generated Output ===";

    private static string ExtractAssistantText(string stdout)
    {
        // TensorSharp.Cli prints the literal line "=== Generated Output ==="
        // immediately before streaming the model's generated tokens, and the
        // model output runs until the next structured-log line
        // ("HH:MM:SS info: TensorSharp.Cli[…]"). Using that delimiter pair
        // gives us exactly the assistant text — no banner, no timing dump,
        // no kernel-warmup chatter.
        //
        // For synthetic benchmark cases the CLI never emits the marker, so
        // we return empty (there is no real assistant text to capture).
        //
        // We do not truncate here; the caller is responsible for capping
        // before persistence.
        int markerIdx = stdout.IndexOf(GeneratedOutputMarker, StringComparison.Ordinal);
        if (markerIdx < 0)
        {
            return string.Empty;
        }
        int textStart = stdout.IndexOf('\n', markerIdx);
        if (textStart < 0)
        {
            return string.Empty;
        }
        textStart++;

        string[] lines = stdout[textStart..].Split('\n');
        var keep = new List<string>();
        foreach (string raw in lines)
        {
            string line = raw.TrimEnd('\r');
            // Only the structured-log prefix terminates the block here — NOT
            // the whitespace-continuation heuristic, because Gemma family
            // models emit assistant text that often starts with a leading
            // space (BPE leading-space token), which is otherwise
            // indistinguishable from a log continuation line.
            if (LogLinePrefixRegex.IsMatch(line))
            {
                break;
            }
            keep.Add(line);
        }
        // Trim trailing blank lines so consumers get a clean string.
        while (keep.Count > 0 && string.IsNullOrWhiteSpace(keep[^1]))
        {
            keep.RemoveAt(keep.Count - 1);
        }
        return string.Join('\n', keep);
    }

    private static readonly System.Text.RegularExpressions.Regex LogLinePrefixRegex =
        new(@"^(?:\d{2}:\d{2}:\d{2}\s+)?(info|warn|warning|error|critical|trace|debug|fail|crit):\s",
            System.Text.RegularExpressions.RegexOptions.Compiled
            | System.Text.RegularExpressions.RegexOptions.IgnoreCase);

    private static void ScrubInheritedTsEnv(ProcessStartInfo psi)
    {
        // Wipe TS_*, GDN_*, KV_*, MAX_CONTEXT, VIDEO_MAX_FRAMES, VIDEO_SAMPLE_FPS,
        // FUSED_*, QWEN35_* from inherited env so the matrix has a clean slate.
        var prefixes = new[] { "TS_", "GDN_", "QWEN35_", "FUSED_" };
        var keys = new[] { "KV_CACHE_DTYPE", "MAX_CONTEXT", "MAX_TOKENS", "VIDEO_MAX_FRAMES", "VIDEO_SAMPLE_FPS" };
        var toClear = new List<string>();
        foreach (System.Collections.DictionaryEntry entry in Environment.GetEnvironmentVariables())
        {
            string k = (string)entry.Key;
            if (prefixes.Any(p => k.StartsWith(p, StringComparison.Ordinal))
                || keys.Contains(k, StringComparer.Ordinal))
            {
                toClear.Add(k);
            }
        }
        foreach (string k in toClear)
        {
            psi.Environment.Remove(k);
        }
    }

    private static string FormatCommand(string exe, IReadOnlyList<string> args, IReadOnlyDictionary<string, string> env)
    {
        var sb = new StringBuilder();
        foreach ((string k, string v) in env)
        {
            sb.Append(CultureInfo.InvariantCulture, $"{k}={Quote(v)} ");
        }
        sb.Append(Quote(exe));
        foreach (string a in args)
        {
            sb.Append(' ').Append(Quote(a));
        }
        return sb.ToString();
    }

    private static string Quote(string s)
    {
        if (string.IsNullOrEmpty(s) || s.Any(c => c == ' ' || c == '\t' || c == '"'))
        {
            return "\"" + s.Replace("\"", "\\\"") + "\"";
        }
        return s;
    }
}
