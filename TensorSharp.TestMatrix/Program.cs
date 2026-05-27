// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license. See LICENSE in the repo root.

using System.Diagnostics;
using System.Globalization;
using System.Runtime.InteropServices;
using TensorSharp.TestMatrix.Configuration;
using TensorSharp.TestMatrix.Matrix;
using TensorSharp.TestMatrix.Reporting;
using TensorSharp.TestMatrix.Runners;

namespace TensorSharp.TestMatrix;

public static class Program
{
    public static int Main(string[] args)
    {
        try
        {
            var opts = CliArgs.Parse(args);
            if (opts.ShowHelp)
            {
                PrintUsage();
                return 0;
            }

            MatrixConfig cfg = MatrixConfigLoader.LoadOrDefault(opts.ConfigPath);
            ApplyOverrides(cfg, opts);

            string cliExecutable = ResolveCliExecutable(cfg, opts);
            string inputsRoot = ResolveInputsRoot(opts);

            Console.Error.WriteLine($"[testmatrix] model_dir       = {cfg.ModelDir}");
            Console.Error.WriteLine($"[testmatrix] cli_executable  = {cliExecutable}");
            Console.Error.WriteLine($"[testmatrix] inputs_root     = {inputsRoot}");
            Console.Error.WriteLine($"[testmatrix] results_dir     = {Path.GetFullPath(cfg.ResultsDir)}");
            Console.Error.WriteLine($"[testmatrix] report_path     = {Path.GetFullPath(cfg.ReportPath)}");

            IReadOnlyList<ModelSpec> models = ModelDiscovery.Discover(cfg);
            if (opts.ModelIds is { Count: > 0 } filterIds)
            {
                var allowed = new HashSet<string>(filterIds, StringComparer.OrdinalIgnoreCase);
                models = models.Where(m => allowed.Contains(m.Id) || allowed.Contains(m.Family)).ToList();
            }
            if (models.Count == 0)
            {
                Console.Error.WriteLine($"[testmatrix] no models discovered under {cfg.ModelDir} and no explicit models in config (or all filtered out by --models)");
                return 2;
            }
            Console.Error.WriteLine($"[testmatrix] discovered {models.Count} model(s):");
            foreach (ModelSpec m in models)
            {
                string mmproj = string.IsNullOrEmpty(m.MmprojPath) ? "(no mmproj)" : $"mmproj={Path.GetFileName(m.MmprojPath)}";
                Console.Error.WriteLine($"    - {m.Id} [{m.Family}] {Path.GetFileName(m.GgufPath)} {mmproj}");
            }

            IReadOnlyList<BackendInfo> backends = SelectBackends(cfg, opts);
            IReadOnlyList<FeatureSpec> features = SelectFeatures(cfg, opts);
            IReadOnlyList<EnvVarSpec> envVars = SelectEnvVars(cfg, opts);

            Console.Error.WriteLine($"[testmatrix] backends   : {string.Join(", ", backends.Select(b => b.Id))}");
            Console.Error.WriteLine($"[testmatrix] features   : {string.Join(", ", features.Select(f => f.Id))}");
            Console.Error.WriteLine($"[testmatrix] env-var sweeps : {string.Join(", ", envVars.Select(e => e.Name))}");

            var expander = new MatrixExpander(models, backends, features, envVars);
            MatrixExpansion exp = expander.Expand();
            Console.Error.WriteLine($"[testmatrix] expanded to {exp.Cases.Count} test cases ({exp.Skipped.Count} skipped before run)");

            if (opts.DryRun)
            {
                foreach (TestCase tc in exp.Cases)
                {
                    Console.WriteLine(tc.Id);
                }
                return 0;
            }

            var store = new ResultStore(cfg.ResultsDir);
            var runner = new CliRunner(cfg, cliExecutable, inputsRoot);
            int total = exp.Cases.Count;
            int idx = 0;
            int failed = 0;
            foreach (TestCase tc in exp.Cases)
            {
                idx++;
                if ((cfg.SkipExisting || opts.SkipExisting) && store.Exists(tc.Id))
                {
                    Console.Error.WriteLine($"[{idx,4}/{total}] cached   {tc.Id}");
                    continue;
                }
                Console.Error.WriteLine($"[{idx,4}/{total}] running  {tc.Id}");
                TestResult res = runner.Run(tc);
                store.Save(res);
                string status = res.Skipped ? "skip" : res.Ok ? "ok  " : "FAIL";
                string metric = res.Ok
                    ? FormatInline(res)
                    : (res.Skipped ? (res.SkipReason ?? "skip") : (res.Error ?? "fail"));
                Console.Error.WriteLine($"          {status,-4}  {metric}");
                if (!res.Ok && !res.Skipped)
                {
                    failed++;
                }
            }

            // Generate the report from the union of in-memory results and pre-existing
            // ones on disk (covers --skip-existing and resumed runs).
            IReadOnlyList<TestResult> allResults = store.LoadAll();
            string host = $"{RuntimeInformation.OSDescription} / {RuntimeInformation.OSArchitecture} / .NET {Environment.Version}";
            string hostLabel = opts.HostLabel ?? BaselineStore.DefaultHostLabel();

            double threshold = opts.RegressionThresholdPct ?? cfg.RegressionThresholdPct;
            string baselinePath = BaselineStore.ResolvePath(opts.BaselinePath, cfg.BaselinesDir, hostLabel);
            BaselineFile? baseline = BaselineStore.TryLoad(baselinePath);
            if (baseline is null)
            {
                Console.Error.WriteLine($"[testmatrix] no baseline found at {baselinePath} — regression analysis will report all cells as untracked");
            }
            else
            {
                Console.Error.WriteLine($"[testmatrix] baseline loaded: {baselinePath} ({baseline.Entries.Count} entries, captured {baseline.CapturedAt:O})");
            }
            RegressionSummary regressions = RegressionAnalyzer.Analyze(
                allResults,
                baseline,
                threshold,
                hostLabel,
                baseline is null ? null : baselinePath);

            var report = new MarkdownReportGenerator(allResults, exp.Skipped, DateTimeOffset.UtcNow, host, regressions);
            string md = report.Generate();
            string reportPath = Path.GetFullPath(cfg.ReportPath);
            Directory.CreateDirectory(Path.GetDirectoryName(reportPath) ?? ".");
            File.WriteAllText(reportPath, md);
            Console.Error.WriteLine($"[testmatrix] wrote report: {reportPath}");

            if (opts.UpdateBaseline)
            {
                BaselineFile fresh = BaselineStore.BuildFromResults(
                    allResults,
                    hostLabel,
                    BaselineStore.TryGetGitHeadShort(),
                    notes: $"Captured by --update-baseline on {DateTimeOffset.UtcNow:O}");
                BaselineStore.Write(baselinePath, fresh);
                Console.Error.WriteLine($"[testmatrix] wrote baseline ({fresh.Entries.Count} passing entries): {baselinePath}");
            }

            Console.Error.WriteLine($"[testmatrix] regressions: runtime={regressions.RuntimeRegressions} correctness={regressions.CorrectnessRegressions} throughput={regressions.ThroughputRegressions} improvements={regressions.Improvements} untracked={regressions.Untracked}");

            if (opts.FailOnRegression && regressions.HasBlockingRegressions)
            {
                Console.Error.WriteLine($"[testmatrix] blocking regressions detected; exiting non-zero (--fail-on-regression)");
                return 4;
            }
            if (failed > 0 && opts.FailOnError)
            {
                Console.Error.WriteLine($"[testmatrix] {failed} case(s) failed; exiting non-zero (--fail-on-error)");
                return 3;
            }
            return 0;
        }
        catch (Exception ex)
        {
            Console.Error.WriteLine($"[testmatrix] fatal: {ex.GetType().Name}: {ex.Message}");
            Console.Error.WriteLine(ex.StackTrace);
            return 1;
        }
    }

    private static string FormatInline(TestResult r)
    {
        if (r.PrefillTps > 0 && r.DecodeTps > 0)
        {
            return $"prefill={r.PrefillTps.ToString("F1", CultureInfo.InvariantCulture)} t/s  decode={r.DecodeTps.ToString("F1", CultureInfo.InvariantCulture)} t/s  wall={(r.TotalWallMs / 1000).ToString("F1", CultureInfo.InvariantCulture)}s";
        }
        if (r.PrefillTps > 0)
        {
            return $"prefill={r.PrefillTps.ToString("F1", CultureInfo.InvariantCulture)} t/s  wall={(r.TotalWallMs / 1000).ToString("F1", CultureInfo.InvariantCulture)}s";
        }
        if (r.DecodeTps > 0)
        {
            return $"decode={r.DecodeTps.ToString("F1", CultureInfo.InvariantCulture)} t/s  wall={(r.TotalWallMs / 1000).ToString("F1", CultureInfo.InvariantCulture)}s";
        }
        return $"wall={(r.TotalWallMs / 1000).ToString("F1", CultureInfo.InvariantCulture)}s";
    }

    private static void ApplyOverrides(MatrixConfig cfg, CliArgs opts)
    {
        if (opts.ModelDir is not null) cfg.ModelDir = opts.ModelDir;
        if (opts.ResultsDir is not null) cfg.ResultsDir = opts.ResultsDir;
        if (opts.ReportPath is not null) cfg.ReportPath = opts.ReportPath;
        if (opts.CliExecutable is not null) cfg.CliExecutable = opts.CliExecutable;
        if (opts.TimeoutSeconds is { } ts) cfg.TimeoutSeconds = ts;
        if (opts.WarmupRuns is { } w) cfg.WarmupRuns = w;
        if (opts.BenchmarkRuns is { } br) cfg.BenchmarkRuns = br;
    }

    private static IReadOnlyList<BackendInfo> SelectBackends(MatrixConfig cfg, CliArgs opts)
    {
        IEnumerable<string> ids = opts.Backends ?? (cfg.DefaultBackends.Count > 0
            ? cfg.DefaultBackends
            : BackendCatalog.All.Select(b => b.Id));
        var list = new List<BackendInfo>();
        foreach (string id in ids)
        {
            BackendInfo? b = BackendCatalog.FindById(id);
            if (b is null)
            {
                Console.Error.WriteLine($"[testmatrix] unknown backend id '{id}' — skipping");
                continue;
            }
            list.Add(b);
        }
        return list;
    }

    private static IReadOnlyList<FeatureSpec> SelectFeatures(MatrixConfig cfg, CliArgs opts)
    {
        IEnumerable<string> ids = opts.Features ?? (cfg.DefaultFeatures.Count > 0
            ? cfg.DefaultFeatures
            : FeatureCatalog.All.Select(f => f.Id));
        var list = new List<FeatureSpec>();
        foreach (string id in ids)
        {
            FeatureSpec? f = FeatureCatalog.FindById(id);
            if (f is null)
            {
                Console.Error.WriteLine($"[testmatrix] unknown feature id '{id}' — skipping");
                continue;
            }
            list.Add(f);
        }
        return list;
    }

    private static IReadOnlyList<EnvVarSpec> SelectEnvVars(MatrixConfig cfg, CliArgs opts)
    {
        if (opts.EnvVars is not null && opts.EnvVars.Count == 1 && opts.EnvVars[0] == "none")
        {
            return Array.Empty<EnvVarSpec>();
        }
        IEnumerable<string>? ids = opts.EnvVars ?? (cfg.DefaultEnvVars.Count > 0 ? cfg.DefaultEnvVars : null);
        if (ids is null)
        {
            return EnvVarMatrix.All;
        }
        var list = new List<EnvVarSpec>();
        foreach (string id in ids)
        {
            EnvVarSpec? v = EnvVarMatrix.FindByName(id);
            if (v is null)
            {
                Console.Error.WriteLine($"[testmatrix] unknown env var '{id}' — skipping");
                continue;
            }
            list.Add(v);
        }
        return list;
    }

    private static string ResolveCliExecutable(MatrixConfig cfg, CliArgs opts)
    {
        if (!string.IsNullOrWhiteSpace(opts.CliExecutable))
        {
            return Path.GetFullPath(opts.CliExecutable);
        }
        if (!string.IsNullOrWhiteSpace(cfg.CliExecutable))
        {
            return Path.GetFullPath(cfg.CliExecutable);
        }

        // Fall back to the conventional checkout layout: <repo>/TensorSharp.Cli/bin/TensorSharp.Cli
        string ext = RuntimeInformation.IsOSPlatform(OSPlatform.Windows) ? ".exe" : "";
        string baseDir = AppContext.BaseDirectory;
        // Try ../../TensorSharp.Cli/bin/TensorSharp.Cli, then ../TensorSharp.Cli/bin/...
        var candidates = new[]
        {
            Path.Combine(baseDir, "..", "..", "TensorSharp.Cli", "bin", "TensorSharp.Cli" + ext),
            Path.Combine(baseDir, "..", "TensorSharp.Cli", "bin", "TensorSharp.Cli" + ext),
            "TensorSharp.Cli" + ext,
        };
        foreach (string c in candidates)
        {
            string full = Path.GetFullPath(c);
            if (File.Exists(full))
            {
                return full;
            }
        }
        throw new FileNotFoundException(
            $"TensorSharp.Cli executable not found. Pass --cli-executable <path> or set cli_executable in the config. Tried: {string.Join(", ", candidates.Select(Path.GetFullPath))}");
    }

    private static string ResolveInputsRoot(CliArgs opts)
    {
        if (!string.IsNullOrWhiteSpace(opts.InputsRoot))
        {
            return Path.GetFullPath(opts.InputsRoot);
        }
        string baseDir = AppContext.BaseDirectory;
        string copied = Path.Combine(baseDir, "Inputs");
        if (Directory.Exists(copied))
        {
            return copied;
        }
        // Fall back to source tree
        string src = Path.GetFullPath(Path.Combine(baseDir, "..", "..", "TensorSharp.TestMatrix", "Inputs"));
        return src;
    }

    private static void PrintUsage()
    {
        Console.WriteLine("""
TensorSharp.TestMatrix — inference test / benchmark matrix runner.

Usage:
  TensorSharp.TestMatrix [options]

Options:
  --config <path>            JSON config file (default: Defaults/matrix-config.json)
  --model-dir <path>         Directory to scan for GGUF models (overrides config)
  --models <id,id,...>       Restrict to specific model ids (after discovery)
  --backends <id,id,...>     Backend ids to test (cpu, ggml_cpu, ggml_metal, ggml_cuda, cuda, mlx)
  --backend <id>             Shortcut for a single backend (alias for --backends with one entry)
  --features <id,id,...>     Feature ids to test (pp512, tg128, short_text, image, ...)
  --env-vars <name,...>      Env-var sweeps to include; pass 'none' to disable
  --inputs-root <path>       Root for prompt / media / tool / multi-turn assets
  --cli-executable <path>    TensorSharp.Cli executable to invoke
  --results-dir <path>       Where to write per-case JSON results
  --report <path>            Output markdown report path
  --timeout-seconds <n>      Per-case subprocess timeout (default 900)
  --warmup-runs <n>          Warmup runs for inference-mode cases (default 1)
  --benchmark-runs <n>       Repeat count for synthetic benchmark cases (default 3)
  --skip-existing            Skip cases whose result JSON already exists
  --dry-run                  Print case ids and exit without running
  --fail-on-error            Exit non-zero if any case failed
  --baseline <path>          Baseline JSON to compare against (default: Baselines/baseline-<host>.json)
  --update-baseline          Write the current run as the new baseline at the resolved path
  --regression-threshold-pct Throughput drop (%) flagged as a regression (default 10)
  --fail-on-regression       Exit non-zero if there are blocking regressions vs baseline
  --host-label <label>       Override the auto-detected host label (e.g. linux-cuda, macos-mlx)
  -h, --help                 Show this help
""");
    }

    private sealed class CliArgs
    {
        public string? ConfigPath;
        public string? ModelDir;
        public List<string>? ModelIds;
        public List<string>? Backends;
        public List<string>? Features;
        public List<string>? EnvVars;
        public string? InputsRoot;
        public string? CliExecutable;
        public string? ResultsDir;
        public string? ReportPath;
        public int? TimeoutSeconds;
        public int? WarmupRuns;
        public int? BenchmarkRuns;
        public bool SkipExisting;
        public bool DryRun;
        public bool FailOnError;
        public string? BaselinePath;
        public bool UpdateBaseline;
        public double? RegressionThresholdPct;
        public bool FailOnRegression;
        public string? HostLabel;
        public bool ShowHelp;

        public static CliArgs Parse(string[] args)
        {
            var o = new CliArgs();
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                switch (a)
                {
                    case "--config": o.ConfigPath = args[++i]; break;
                    case "--model-dir": o.ModelDir = args[++i]; break;
                    case "--models": o.ModelIds = SplitCsv(args[++i]); break;
                    case "--backends": o.Backends = SplitCsv(args[++i]); break;
                    case "--backend":
                        // Singular alias for one-shot dev iteration (e.g.
                        // --backend ggml_cpu). Matches TensorSharp.Cli's flag
                        // name; under the hood it just sets Backends to a
                        // one-element list.
                        o.Backends = new List<string> { args[++i].Trim() };
                        break;
                    case "--features": o.Features = SplitCsv(args[++i]); break;
                    case "--env-vars": o.EnvVars = SplitCsv(args[++i]); break;
                    case "--inputs-root": o.InputsRoot = args[++i]; break;
                    case "--cli-executable": o.CliExecutable = args[++i]; break;
                    case "--results-dir": o.ResultsDir = args[++i]; break;
                    case "--report": o.ReportPath = args[++i]; break;
                    case "--timeout-seconds": o.TimeoutSeconds = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                    case "--warmup-runs": o.WarmupRuns = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                    case "--benchmark-runs": o.BenchmarkRuns = int.Parse(args[++i], CultureInfo.InvariantCulture); break;
                    case "--skip-existing": o.SkipExisting = true; break;
                    case "--dry-run": o.DryRun = true; break;
                    case "--fail-on-error": o.FailOnError = true; break;
                    case "--baseline": o.BaselinePath = args[++i]; break;
                    case "--update-baseline": o.UpdateBaseline = true; break;
                    case "--regression-threshold-pct": o.RegressionThresholdPct = double.Parse(args[++i], CultureInfo.InvariantCulture); break;
                    case "--fail-on-regression": o.FailOnRegression = true; break;
                    case "--host-label": o.HostLabel = args[++i]; break;
                    case "-h":
                    case "--help":
                        o.ShowHelp = true;
                        break;
                    default:
                        Console.Error.WriteLine($"[testmatrix] unknown argument: {a}");
                        o.ShowHelp = true;
                        break;
                }
            }
            return o;
        }

        private static List<string> SplitCsv(string s) =>
            s.Split(',', StringSplitOptions.RemoveEmptyEntries | StringSplitOptions.TrimEntries).ToList();
    }
}
