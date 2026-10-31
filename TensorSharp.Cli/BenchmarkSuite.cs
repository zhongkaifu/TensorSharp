// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
//
// --bench-workload / --bench-suite: many benchmark rows in one process, with the model loaded once.
//
// A rented multi-GPU block used to pay a model load and a process gap for every benchmark row (in the
// DeepSeek V4.1 S9 session the four GPUs ran kernels 13.4% of the time; loads took 13.3% and the gaps
// between processes 28%). A suite runs every row through the same loop as --benchmark (the same
// stopwatches and the same "benchmark run i/N" lines), so its numbers compare with one-row processes.
// A cold row drops the model's host-mapped tables (ModelBase.HostMappedTables, DeepSeek V4.1's Engram
// tables) from the page cache on every node first, inside the process, instead of restarting it.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using Microsoft.Extensions.Logging;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace TensorSharp.Cli
{
    /// <summary>What one benchmark row runs.</summary>
    internal sealed class BenchmarkPlan
    {
        /// <summary>Synthetic prompt length; ignored when <see cref="PromptIds"/> is set.</summary>
        public int PrefillTokens { get; init; }
        /// <summary>A real prompt (text or token file) instead of the synthetic one.</summary>
        public int[] PromptIds { get; init; }
        public int DecodeTokens { get; init; }
        public int Runs { get; init; } = 1;
        public bool Chunked { get; init; }
        public bool FixedTokens { get; init; }
        public bool RandomTokens { get; init; }
        /// <summary>Seed of the random-token stream. The default is the one --bench-random-tokens has always used,
        /// so a row with the same shape replays a one-row process's ids.</summary>
        public int Seed { get; init; } = BenchWorkloadSpec.DefaultSeed;
        /// <summary>Run the untimed greedy correctness chain after a fixed/random-token row.</summary>
        public bool Check { get; init; } = true;
        /// <summary>Runs before each run (0-based), before its cache reset and outside every stopwatch.</summary>
        public Action<int> BeforeRun { get; init; }
    }

    internal sealed class BenchmarkRunSample
    {
        public int Run { get; init; }
        public double PrefillMs { get; init; }
        public double PrefillTps { get; init; }
        public double DecodeMs { get; init; }
        public double DecodeTps { get; init; }
        public double MsPerToken { get; init; }
        public double DecodeModelMs { get; init; }
        public double DecodeModelTps { get; init; }
        public double GreedySampleMs { get; init; }
        public DateTime PrefillStartUtc { get; init; }
        public DateTime DecodeStartUtc { get; init; }
        public DateTime DecodeEndUtc { get; init; }
    }

    internal sealed class BenchmarkOutcome
    {
        public int PrefillTokens { get; init; }
        public int DecodeTokens { get; init; }
        public int Runs { get; init; }
        public bool Chunked { get; init; }
        public string DecodeMode { get; init; }
        public List<BenchmarkRunSample> Samples { get; } = new List<BenchmarkRunSample>();
        public double AvgPrefillTps { get; set; }
        public double AvgDecodeTps { get; set; }
        /// <summary>"run1" or "untimed correctness"; null when no token was sampled.</summary>
        public string SampledTokensSource { get; set; }
        public int PrefillTopToken { get; set; } = -1;
        public int[] SampledDecodeTokens { get; set; }
    }

    /// <summary>When a row drops the host-mapped tables from the page cache.</summary>
    internal enum BenchEvict
    {
        /// <summary>Never: a warm row.</summary>
        None,
        /// <summary>Once, before the row's first run (a cold process's state: later runs warm up as they go).</summary>
        Row,
        /// <summary>Before every run (each run starts cold).</summary>
        Run,
    }

    /// <summary>
    /// One <c>--bench-workload</c> row: comma-separated <c>key=value</c> fields, plus the bare words <c>cold</c>,
    /// <c>warm</c> and <c>chunked</c> (also <c>cold=1</c>, <c>warm=1</c>). Keys: <c>label</c>, <c>prefill</c> (synthetic length, or a cap on a real
    /// prompt), <c>text</c> (UTF-8 file, tokenized without special tokens), <c>tokens</c> (JSON token-id array),
    /// <c>decode</c>, <c>runs</c>, <c>mode</c> (greedy | fixed | random), <c>evict</c> (none | row | run),
    /// <c>chunked</c> (0 | 1), <c>seed</c>, <c>check</c> (0 | 1). Unset fields take the process's
    /// <c>--bench-*</c> values.
    /// </summary>
    internal sealed class BenchWorkloadSpec
    {
        public const int DefaultSeed = 20261001;

        public string Label { get; set; }
        public int? Prefill { get; set; }
        public string TextPath { get; set; }
        public string TokensPath { get; set; }
        public int Decode { get; set; }
        public int Runs { get; set; }
        /// <summary>greedy, fixed or random.</summary>
        public string Mode { get; set; }
        public BenchEvict Evict { get; set; }
        public bool Chunked { get; set; }
        public int Seed { get; set; } = DefaultSeed;
        public bool Check { get; set; } = true;
        /// <summary>The text as written, for the log and the JSON.</summary>
        public string Spec { get; set; }

        public string Source => TextPath != null ? "text" : TokensPath != null ? "tokens" : "synthetic";

        /// <summary>The process-wide defaults a row starts from (<c>--bench-decode</c>, <c>--bench-runs</c>, ...).</summary>
        internal sealed class Defaults
        {
            public int Prefill { get; init; } = 32;
            public int Decode { get; init; } = 64;
            public int Runs { get; init; } = 1;
            public bool Chunked { get; init; }
            public string Mode { get; init; } = "greedy";
        }

        public static BenchWorkloadSpec Parse(string spec, Defaults defaults)
        {
            if (string.IsNullOrWhiteSpace(spec))
                throw new ArgumentException("An empty --bench-workload.");
            var w = new BenchWorkloadSpec
            {
                Spec = spec.Trim(),
                Decode = defaults.Decode,
                Runs = defaults.Runs,
                Chunked = defaults.Chunked,
                Mode = defaults.Mode,
            };
            var seen = new HashSet<string>(StringComparer.Ordinal);
            foreach (string raw in spec.Split(','))
            {
                string field = raw.Trim();
                if (field.Length == 0)
                    continue;
                int eq = field.IndexOf('=');
                string key = (eq < 0 ? field : field[..eq]).Trim().ToLowerInvariant();
                string value = eq < 0 ? null : field[(eq + 1)..].Trim();
                if (value == null)
                {
                    // Bare words.
                    (key, value) = key switch
                    {
                        "cold" => ("evict", "row"),
                        "warm" => ("evict", "none"),
                        "chunked" => ("chunked", "1"),
                        _ => throw Bad(spec, $"'{field}' is not key=value (bare words: cold, warm, chunked)"),
                    };
                }
                else if (key is "cold" or "warm")
                {
                    // The key spelling of the bare words (a JSON suite file writes "cold": true as cold=1).
                    bool set = Bool(spec, key, value);
                    (key, value) = ("evict", (key == "cold") == set ? "row" : "none");
                }
                if (!seen.Add(key))
                    throw Bad(spec, $"'{key}' is given twice");
                switch (key)
                {
                    case "label":
                        if (value.Length == 0 || value.Any(c => char.IsWhiteSpace(c) || c == '"'))
                            throw Bad(spec, "a label must be non-empty, without spaces or quotes");
                        w.Label = value;
                        break;
                    case "prefill": w.Prefill = PositiveInt(spec, key, value, min: 1); break;
                    case "text": w.TextPath = NonEmpty(spec, key, value); break;
                    case "tokens": w.TokensPath = NonEmpty(spec, key, value); break;
                    case "decode": w.Decode = PositiveInt(spec, key, value, min: 0); break;
                    case "runs": w.Runs = PositiveInt(spec, key, value, min: 1); break;
                    case "mode":
                        w.Mode = value.ToLowerInvariant() switch
                        {
                            "greedy" or "greedy-e2e" => "greedy",
                            "fixed" or "fixed-inference" => "fixed",
                            "random" or "random-inference" => "random",
                            _ => throw Bad(spec, $"mode must be greedy, fixed or random, got '{value}'"),
                        };
                        break;
                    case "evict":
                        w.Evict = value.ToLowerInvariant() switch
                        {
                            "none" or "warm" or "0" => BenchEvict.None,
                            "row" or "cold" or "1" => BenchEvict.Row,
                            "run" => BenchEvict.Run,
                            _ => throw Bad(spec, $"evict must be none, row or run, got '{value}'"),
                        };
                        break;
                    case "chunked": w.Chunked = Bool(spec, key, value); break;
                    case "seed":
                        if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int seed))
                            throw Bad(spec, $"seed must be an integer, got '{value}'");
                        w.Seed = seed;
                        break;
                    case "check": w.Check = Bool(spec, key, value); break;
                    default:
                        throw Bad(spec, $"unknown key '{key}' (label, prefill, text, tokens, decode, runs, mode, evict, chunked, seed, check)");
                }
            }
            if (w.TextPath != null && w.TokensPath != null)
                throw Bad(spec, "give text or tokens, not both");
            if (w.Source == "synthetic")
                w.Prefill ??= defaults.Prefill;
            return w;
        }

        /// <summary>Rows from a suite file: a JSON array of objects with the same keys, or one spec per line
        /// (blank lines and lines starting with '#' are skipped).</summary>
        public static List<string> ReadSuiteFile(string path)
        {
            string text = File.ReadAllText(path);
            var specs = new List<string>();
            string trimmed = text.TrimStart();
            if (trimmed.StartsWith("[", StringComparison.Ordinal))
            {
                using var doc = JsonDocument.Parse(text);
                foreach (JsonElement row in doc.RootElement.EnumerateArray())
                {
                    if (row.ValueKind != JsonValueKind.Object)
                        throw new ArgumentException($"{path}: every suite entry must be a JSON object.");
                    var parts = new List<string>();
                    foreach (JsonProperty p in row.EnumerateObject())
                    {
                        string value = p.Value.ValueKind switch
                        {
                            JsonValueKind.String => p.Value.GetString(),
                            JsonValueKind.Number => p.Value.GetRawText(),
                            JsonValueKind.True => "1",
                            JsonValueKind.False => "0",
                            _ => throw new ArgumentException($"{path}: '{p.Name}' must be a string, number or boolean."),
                        };
                        if (value.Contains(','))
                            throw new ArgumentException($"{path}: '{p.Name}' may not contain a comma.");
                        parts.Add(p.Name + "=" + value);
                    }
                    specs.Add(string.Join(",", parts));
                }
                return specs;
            }
            foreach (string line in text.Split('\n'))
            {
                string s = line.Trim();
                if (s.Length > 0 && !s.StartsWith("#", StringComparison.Ordinal))
                    specs.Add(s);
            }
            return specs;
        }

        /// <summary>Give every row a unique label: an explicit one must be unique, a missing one is derived.</summary>
        public static void AssignLabels(IList<BenchWorkloadSpec> rows)
        {
            var used = new HashSet<string>(StringComparer.Ordinal);
            foreach (var w in rows.Where(r => r.Label != null))
                if (!used.Add(w.Label))
                    throw new ArgumentException($"Two --bench-workload rows share the label '{w.Label}'.");
            foreach (var w in rows.Where(r => r.Label == null))
            {
                string stem = w.Source switch
                {
                    "text" => Path.GetFileNameWithoutExtension(w.TextPath),
                    "tokens" => Path.GetFileNameWithoutExtension(w.TokensPath),
                    _ => "pp" + w.Prefill.Value.ToString(CultureInfo.InvariantCulture),
                };
                if (w.Source != "synthetic" && w.Prefill is int cap)
                    stem += "-" + cap.ToString(CultureInfo.InvariantCulture);
                stem += "-d" + w.Decode.ToString(CultureInfo.InvariantCulture)
                    + (w.Evict == BenchEvict.None ? "-warm" : w.Evict == BenchEvict.Row ? "-cold" : "-coldeach");
                stem = new string(stem.Select(c => char.IsLetterOrDigit(c) || c is '-' or '_' or '.' ? c : '_').ToArray());
                string label = stem;
                for (int n = 2; !used.Add(label); n++)
                    label = stem + "." + n.ToString(CultureInfo.InvariantCulture);
                w.Label = label;
            }
        }

        private static int PositiveInt(string spec, string key, string value, int min)
        {
            if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int v) || v < min)
                throw Bad(spec, $"{key} must be an integer >= {min}, got '{value}'");
            return v;
        }

        private static bool Bool(string spec, string key, string value) => value.ToLowerInvariant() switch
        {
            "1" or "true" or "yes" => true,
            "0" or "false" or "no" => false,
            _ => throw Bad(spec, $"{key} must be 0 or 1, got '{value}'"),
        };

        private static string NonEmpty(string spec, string key, string value)
            => string.IsNullOrEmpty(value) ? throw Bad(spec, $"{key} needs a path") : value;

        private static ArgumentException Bad(string spec, string why)
            => new ArgumentException($"--bench-workload '{spec}': {why}.");
    }

    /// <summary>Process-wide suite settings.</summary>
    internal sealed class BenchSuiteOptions
    {
        public string JsonPath { get; init; }
        public HostPageCacheMeasure EvictMeasure { get; init; } = HostPageCacheMeasure.Sampled;
        /// <summary>Stop the suite at the first eviction that is not verified out of the page cache on every node
        /// (--bench-evict-strict): the rest of a matrix measured warm by mistake is GPU time for nothing.</summary>
        public bool EvictStrict { get; init; }
        public string Backend { get; init; }
        public string ModelPath { get; init; }

        public static HostPageCacheMeasure ParseMeasure(string value) => value?.ToLowerInvariant() switch
        {
            "none" or "off" => HostPageCacheMeasure.None,
            "sampled" or "sample" => HostPageCacheMeasure.Sampled,
            "full" => HostPageCacheMeasure.Full,
            _ => throw new ArgumentException($"--bench-evict-measure must be none, sampled or full, got '{value}'."),
        };
    }

    partial class Program
    {
        /// <summary>A suite row after its prompt has been read: the spec plus its tokens.</summary>
        internal sealed class PreparedWorkload
        {
            public BenchWorkloadSpec Spec { get; init; }
            public int[] PromptIds { get; init; }
            public string SourceSha256 { get; init; }
            public string PromptSha256 { get; init; }
        }

        /// <summary>Checks a suite before the model loads: every prompt file exists, and a cold row can be cold
        /// on this host. Saves a model load (and, on a rented block, its GPU time) when a row could never run. A
        /// refusal is an <see cref="ArgumentException"/>: the CLI prints it as one "Configuration error:" line.</summary>
        internal static void PreflightBenchmarkSuite(IReadOnlyList<BenchWorkloadSpec> rows)
        {
            foreach (var w in rows)
            {
                string path = w.TextPath ?? w.TokensPath;
                if (path != null && !File.Exists(path))
                    throw new ArgumentException($"--bench-workload {w.Label}: {path} does not exist.");
            }
            if (rows.Any(r => r.Evict != BenchEvict.None) && !HostPageCache.IsSupported)
                throw new ArgumentException(
                    "A cold --bench-workload row (evict=row|run) is not supported on this OS: " + HostPageCache.UnsupportedReason +
                    ". Run cold rows on Linux, or mark the rows warm.");
        }

        /// <summary>Reads each row's prompt with the loaded model's tokenizer and checks the lengths.</summary>
        internal static List<PreparedWorkload> PrepareBenchmarkSuite(ModelBase model, IReadOnlyList<BenchWorkloadSpec> rows)
        {
            if (rows.Any(r => r.Evict != BenchEvict.None) && model.HostMappedTables.Count == 0)
                throw new ArgumentException(
                    $"A cold --bench-workload row needs a model that reads host-mapped tables (DeepSeek V4.1's Engram); " +
                    $"{model.Config.Architecture} declares none, so evicting would not make the row cold. Mark the rows warm.");
            var prepared = new List<PreparedWorkload>();
            foreach (var w in rows)
            {
                int[] ids = null;
                string sourceSha = null;
                if (w.TextPath != null)
                {
                    byte[] bytes = File.ReadAllBytes(w.TextPath);
                    sourceSha = Convert.ToHexStringLower(SHA256.HashData(bytes));
                    ids = model.Tokenizer.Encode(Encoding.UTF8.GetString(bytes), false).ToArray();
                }
                else if (w.TokensPath != null)
                {
                    byte[] bytes = File.ReadAllBytes(w.TokensPath);
                    sourceSha = Convert.ToHexStringLower(SHA256.HashData(bytes));
                    ids = JsonSerializer.Deserialize<int[]>(bytes)
                        ?? throw new ArgumentException($"--bench-workload {w.Label}: {w.TokensPath} must hold a JSON array of token ids.");
                }
                if (ids != null)
                {
                    int vocab = model.Config.VocabSize;
                    if (ids.Any(t => t < 0 || t >= vocab))
                        throw new ArgumentException($"--bench-workload {w.Label}: a prompt token is outside the vocabulary (0..{vocab - 1}).");
                    if (w.Prefill is int cap)
                    {
                        if (cap > ids.Length)
                            throw new ArgumentException(
                                $"--bench-workload {w.Label}: prefill={cap} but {w.TextPath ?? w.TokensPath} has only {ids.Length} tokens.");
                        ids = ids[..cap];
                    }
                    if (ids.Length < 1)
                        throw new ArgumentException($"--bench-workload {w.Label}: the prompt has no tokens.");
                }
                string promptSha = ids == null ? null
                    : Convert.ToHexStringLower(SHA256.HashData(MemoryMarshalBytes(ids)));
                prepared.Add(new PreparedWorkload { Spec = w, PromptIds = ids, SourceSha256 = sourceSha, PromptSha256 = promptSha });
            }
            return prepared;
        }

        private static byte[] MemoryMarshalBytes(int[] ids)
        {
            var bytes = new byte[ids.Length * 4];
            Buffer.BlockCopy(ids, 0, bytes, 0, bytes.Length);
            return bytes;
        }

        /// <summary>
        /// Runs every row in order on the loaded model. Each row prints the --benchmark lines, framed by
        /// "bench suite workload k/K" lines; a cold row first evicts the host-mapped tables on every node
        /// ("bench evict" lines, one per node). With <see cref="BenchSuiteOptions.JsonPath"/>, a machine-readable
        /// summary is rewritten after every row, so a failed row keeps the rows before it.
        /// </summary>
        internal static void RunBenchmarkSuite(ModelBase model, IReadOnlyList<PreparedWorkload> rows, BenchSuiteOptions options)
        {
            var suiteSw = Stopwatch.StartNew();
            var root = new Dictionary<string, object>
            {
                ["schema"] = "tensorsharp.bench-suite/1",
                ["status"] = "running",
                ["started_utc"] = Iso(DateTime.UtcNow),
                ["model_file"] = options.ModelPath != null ? Path.GetFileName(options.ModelPath) : null,
                ["model_path"] = options.ModelPath,
                ["architecture"] = model.Config.Architecture,
                ["backend"] = options.Backend,
                ["context_length"] = model.MaxContextLength,
                ["kv_cache_dtype"] = model.KvCacheDtype.ToShortString(),
                ["evict_measure"] = options.EvictMeasure.ToString().ToLowerInvariant(),
                ["host_mapped_tables"] = model.HostMappedTables.Select(t => new Dictionary<string, object>
                {
                    ["name"] = t.Name, ["path"] = t.Path, ["offset"] = t.Offset, ["bytes"] = t.Bytes,
                }).ToList(),
            };
            var results = new List<object>();
            root["workloads"] = results;
            (string Label, List<object> Evictions) current = (null, null);
            _log.LogInformation(LogEventIds.CliBenchmark,
                "bench suite starting: workloads={Count} evictMeasure={Measure} hostMappedTables={Tables}",
                rows.Count, options.EvictMeasure.ToString().ToLowerInvariant(), model.HostMappedTables.Count);
            try
            {
                for (int k = 0; k < rows.Count; k++)
                {
                    PreparedWorkload row = rows[k];
                    BenchWorkloadSpec w = row.Spec;
                    _log.LogInformation(LogEventIds.CliBenchmark,
                        "bench suite workload {Index}/{Count}: label={Label} source={Source} prefill={Prefill} decode={Decode} runs={Runs} mode={Mode} evict={Evict} chunked={Chunked} seed={Seed} check={Check}",
                        k + 1, rows.Count, w.Label, w.Source, row.PromptIds?.Length ?? w.Prefill ?? 0, w.Decode, w.Runs, w.Mode,
                        w.Evict.ToString().ToLowerInvariant(), w.Chunked ? 1 : 0, w.Seed, w.Check ? 1 : 0);
                    model.ResetForwardTiming();
                    var evictions = new List<object>();
                    current = (w.Label, evictions);
                    bool? coldVerified = w.Evict == BenchEvict.None ? null : true;
                    DateTime started = DateTime.UtcNow;
                    var rowSw = Stopwatch.StartNew();
                    BenchmarkOutcome outcome = RunBenchmarkCore(model, new BenchmarkPlan
                    {
                        PrefillTokens = w.Prefill ?? 0,
                        PromptIds = row.PromptIds,
                        DecodeTokens = w.Decode,
                        Runs = w.Runs,
                        Chunked = w.Chunked,
                        FixedTokens = w.Mode == "fixed",
                        RandomTokens = w.Mode == "random",
                        Seed = w.Seed,
                        Check = w.Check,
                        BeforeRun = run =>
                        {
                            if (w.Evict == BenchEvict.None || (w.Evict == BenchEvict.Row && run > 0))
                                return;
                            HostTableEvictionResult result = model.EvictHostTablePageCache(options.EvictMeasure);
                            foreach (HostTableEvictionNode node in result.Nodes)
                                _log.LogInformation(LogEventIds.CliBenchmark, "bench evict: label={Label} beforeRun={Run} {Node}",
                                    w.Label, run + 1, node.Describe());
                            if (result.Nodes.Count < result.NodeCount)
                                _log.LogWarning(LogEventIds.CliBenchmark,
                                    "bench evict: label={Label} beforeRun={Run} only {Got} of {Nodes} nodes reported",
                                    w.Label, run + 1, result.Nodes.Count, result.NodeCount);
                            if (result.Verified != true)
                            {
                                _log.LogWarning(LogEventIds.CliBenchmark,
                                    "bench evict: label={Label} beforeRun={Run} the tables are not verified out of the page cache " +
                                    "(verified={Verified}); this row may not be cold",
                                    w.Label, run + 1, result.Verified is bool v ? (v ? "true" : "false") : "na");
                                coldVerified = result.Verified == false || coldVerified == false ? false : null;
                            }
                            evictions.Add(EvictionJson(run + 1, result));
                            if (options.EvictStrict && result.Verified != true)
                                throw new InvalidOperationException(
                                    $"--bench-evict-strict: the host-mapped tables of row '{w.Label}' are not verified out of the " +
                                    $"page cache on every node before run {run + 1}; stopping the suite. See the 'bench evict' lines.");
                        },
                    });
                    rowSw.Stop();
                    _log.LogInformation(LogEventIds.CliBenchmark,
                        "bench suite workload {Index}/{Count} done: label={Label} wallMs={WallMs:F0} coldVerified={Verified}",
                        k + 1, rows.Count, w.Label, rowSw.Elapsed.TotalMilliseconds,
                        coldVerified is bool cv ? (cv ? "true" : "false") : "na");
                    results.Add(WorkloadJson(k + 1, row, outcome, evictions, coldVerified, started, rowSw.Elapsed.TotalMilliseconds));
                    current = (null, null);
                    WriteSuiteJson(options.JsonPath, root);
                }
                root["status"] = "completed";
            }
            catch (Exception ex)
            {
                root["status"] = "failed";
                // The row that failed never reached the results: keep what it did get to (its evictions) and why.
                root["failed_workload"] = new Dictionary<string, object>
                {
                    ["label"] = current.Label,
                    ["evictions"] = current.Evictions,
                    ["error"] = ex.Message,
                };
                throw;
            }
            finally
            {
                root["ended_utc"] = Iso(DateTime.UtcNow);
                root["wall_ms"] = suiteSw.Elapsed.TotalMilliseconds;
                WriteSuiteJson(options.JsonPath, root);
            }
            _log.LogInformation(LogEventIds.CliBenchmark,
                "bench suite complete: workloads={Count} wallMs={WallMs:F0}{Json}",
                rows.Count, suiteSw.Elapsed.TotalMilliseconds, options.JsonPath != null ? " json=" + options.JsonPath : "");
        }

        private static Dictionary<string, object> WorkloadJson(int index, PreparedWorkload row, BenchmarkOutcome o,
            List<object> evictions, bool? coldVerified, DateTime started, double wallMs)
        {
            BenchWorkloadSpec w = row.Spec;
            return new Dictionary<string, object>
            {
                ["index"] = index,
                ["label"] = w.Label,
                ["spec"] = w.Spec,
                ["source"] = w.Source,
                ["path"] = w.TextPath ?? w.TokensPath,
                ["source_sha256"] = row.SourceSha256,
                ["prompt_sha256"] = row.PromptSha256,
                ["prefill_tokens"] = o.PrefillTokens,
                ["decode_tokens"] = o.DecodeTokens,
                ["runs"] = o.Runs,
                ["mode"] = o.DecodeMode,
                ["evict"] = w.Evict.ToString().ToLowerInvariant(),
                ["chunked"] = o.Chunked,
                ["seed"] = w.Seed,
                ["check"] = w.Check,
                ["started_utc"] = Iso(started),
                ["wall_ms"] = wallMs,
                ["cold_verified"] = coldVerified,
                ["evictions"] = evictions,
                ["samples"] = o.Samples.Select(s => new Dictionary<string, object>
                {
                    ["run"] = s.Run,
                    ["prefill_ms"] = Finite(s.PrefillMs),
                    ["prefill_tps"] = Finite(s.PrefillTps),
                    ["decode_ms"] = Finite(s.DecodeMs),
                    ["decode_tps"] = Finite(s.DecodeTps),
                    ["ms_per_token"] = Finite(s.MsPerToken),
                    ["decode_model_ms"] = Finite(s.DecodeModelMs),
                    ["decode_model_tps"] = Finite(s.DecodeModelTps),
                    ["greedy_sample_ms"] = Finite(s.GreedySampleMs),
                    ["prefill_start_utc"] = Iso(s.PrefillStartUtc),
                    ["decode_start_utc"] = Iso(s.DecodeStartUtc),
                    ["decode_end_utc"] = Iso(s.DecodeEndUtc),
                }).ToList(),
                ["avg_prefill_tps"] = Finite(o.AvgPrefillTps),
                ["avg_decode_tps"] = Finite(o.AvgDecodeTps),
                ["sampled_tokens"] = o.SampledTokensSource == null ? null : new Dictionary<string, object>
                {
                    ["source"] = o.SampledTokensSource,
                    ["prefill_top_token"] = o.PrefillTopToken,
                    ["decode_token_ids"] = o.SampledDecodeTokens,
                },
            };
        }

        private static Dictionary<string, object> EvictionJson(int beforeRun, HostTableEvictionResult r) => new()
        {
            ["before_run"] = beforeRun,
            ["verified"] = r.Verified,
            ["node_count"] = r.NodeCount,
            ["nodes"] = r.Nodes.Select(n => new Dictionary<string, object>
            {
                ["node"] = n.Node,
                ["supported"] = n.Supported,
                ["ranges"] = n.Ranges,
                ["bytes_mib"] = n.BytesMiB,
                ["resident_before_mib"] = n.ResidentBeforeMiB < 0 ? null : n.ResidentBeforeMiB,
                ["resident_after_mib"] = n.ResidentAfterMiB < 0 ? null : n.ResidentAfterMiB,
                ["verified"] = n.Verified,
                ["mappings_advised"] = n.MappingsAdvised,
                ["mappings_skipped"] = n.MappingsSkipped,
                ["errors"] = n.Errors,
                ["elapsed_ms"] = n.ElapsedMs,
                ["evict_ms"] = n.EvictMs,
            }).ToList(),
            ["local_errors"] = r.Local.Ranges.Where(x => x.Error != null).Select(x => x.Name + ": " + x.Error).ToList(),
            ["local_reason"] = r.Local.Reason,
        };

        private static void WriteSuiteJson(string path, Dictionary<string, object> root)
        {
            if (string.IsNullOrEmpty(path))
                return;
            string full = Path.GetFullPath(path);
            string dir = Path.GetDirectoryName(full);
            if (!string.IsNullOrEmpty(dir))
                Directory.CreateDirectory(dir);
            string temp = full + ".tmp";
            File.WriteAllText(temp, JsonSerializer.Serialize(root, new JsonSerializerOptions { WriteIndented = true }) + "\n");
            File.Move(temp, full, overwrite: true);
        }

        private static object Finite(double v) => double.IsFinite(v) ? v : null;

        private static string Iso(DateTime utc) => utc.ToString("yyyy-MM-ddTHH:mm:ss.fffZ", CultureInfo.InvariantCulture);
    }
}
