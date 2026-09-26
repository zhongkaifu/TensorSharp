// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Text.RegularExpressions;

namespace TensorSharp.Runtime
{
    /// <summary>
    /// Lets the CLI and the server read their startup options from a JSON
    /// configuration file in addition to the command line. A
    /// <c>--config &lt;path.json&gt;</c> flag names a file whose keys are the same
    /// long option names both hosts already accept (with or without the leading
    /// <c>--</c>); <see cref="Expand"/> translates each key/value into the
    /// equivalent argv tokens and splices them in <em>before</em> the caller's
    /// real command-line arguments.
    ///
    /// The command line wins over every file, and a later <c>--config</c> file wins
    /// over an earlier one. Only the winning entry of a single-valued option is
    /// kept: an entry the real command line sets itself (<c>--key value</c> or
    /// <c>--key=value</c>, any case) or a later file sets again is dropped before
    /// it is resolved, so its download entry is never fetched — <c>--mmproj none</c>
    /// beside a config that names a vision shard must not first pull gigabytes the
    /// run will not use. Each dropped entry is reported once per option, naming the
    /// files and any download that was skipped. Dropping the loser rather than
    /// relying on both hosts reading the last occurrence also keeps the result
    /// independent of how each option reader treats a repeat. Options that
    /// accumulate rather than replace (<c>--stop</c>, <c>--lora</c>, <c>--image</c>,
    /// ... — the <c>AccumulatingFlags</c> table) keep every file's values, in file
    /// order, and add the command line's after them.
    ///
    /// Three conveniences build on the basic key/value mapping:
    /// <list type="bullet">
    /// <item><b>Variables.</b> A reserved <c>"variables"</c> object defines
    /// names that any string value can reference with <c>${name}</c> — so a
    /// shared model root is written once. A reference not found among the
    /// variables falls back to an environment variable of the same name;
    /// variables may reference other variables. <c>${name:-fallback}</c> supplies
    /// a value for when the name is defined nowhere, which is how the shipped
    /// configs stay portable: they resolve a model root from
    /// <c>TENSORSHARP_MODELS</c> when it is set and from a path relative to the
    /// config file otherwise, instead of hard-coding a drive letter that means
    /// nothing on Linux or macOS.</item>
    /// <item><b>Auto-download.</b> A file option may be an object
    /// <c>{ "path": "...", "urls": ["...", "..."] }</c>. If <c>path</c> is missing
    /// on disk it is downloaded from the first working URL and saved there, so the
    /// next run reuses the local copy. See <see cref="ModelDownloader"/>.</item>
    /// </list>
    ///
    /// Example config file:
    /// <code>
    /// {
    ///   "variables": { "root": "C:\\models" },
    ///   "backend": "ggml_cuda",
    ///   "max-tokens": 4096,
    ///   "temperature": 0.7,
    ///   "stop": ["&lt;/s&gt;", "&lt;|eot|&gt;"],
    ///   "model": {
    ///     "path": "${root}/gemma-4-E4B-it-Q8_0.gguf",
    ///     "urls": [ "https://example.com/gemma-4-E4B-it-Q8_0.gguf" ]
    ///   }
    /// }
    /// </code>
    /// Value shapes: a string or number becomes <c>--key value</c>; a boolean
    /// <c>true</c> becomes the bare switch <c>--key</c> (a <c>false</c> is
    /// skipped, so use the explicit negation key — e.g. <c>"no-continuous-batching": true</c>
    /// — to turn something off); an array becomes a repeated flag
    /// (<c>--key v1 --key v2</c>); an object is a downloadable-file spec.
    /// </summary>
    public static class ConfigFileArgs
    {
        /// <summary>The flag that names a JSON configuration file.</summary>
        public const string ConfigFlag = "--config";

        // Keys with a meaning of their own rather than a flag to emit.
        private static readonly HashSet<string> ReservedKeys = new(StringComparer.OrdinalIgnoreCase)
        {
            "variables", "vars", "$schema",
        };

        // Options where every occurrence counts, so a config's values are kept even when
        // the command line (or a later file) names the same option: both hosts collect
        // every --stop, and every --skills-dir / --skill (SkillHostOptions); each --lora
        // is one plug-in and a --lora-scale / --lora-config binds to the --lora before it,
        // so dropping a config's copy would re-bind or lose one; the CLI collects every
        // --image and every reference input in order. Every other option is single-valued,
        // so only its winning entry is kept: the command line's, else the last file's.
        private static readonly HashSet<string> AccumulatingFlags = new(StringComparer.OrdinalIgnoreCase)
        {
            "--stop",
            "--skills-dir", "--skill",
            LoraCliFlags.LoraFlag, LoraCliFlags.ScaleFlag, LoraCliFlags.ConfigFlag,
            "--image", "--ref-image", "--ref-video", "--ref-audio", "--ref-video-audio",
        };

        // Legacy spellings both hosts still read as the SAME option (their parsers match
        // them in one case / one condition), so a command-line value under either
        // spelling overrides a config entry written under the other. First = canonical.
        private static readonly string[][] SameOptionSpellings =
        {
            new[] { "--video-vae", "--wan-vae" },
            new[] { "--video-text-encoder", "--video-te", "--wan-te" },
            new[] { "--video-dit2", "--wan-dit2" },
            new[] { "--continuous-batching", "--paged-batching" },
            new[] { "--no-continuous-batching", "--no-paged-batching" },
        };

        /// <summary>
        /// The spelling both hosts treat <paramref name="flag"/> as, for the legacy names
        /// they still read as another option (<c>--wan-vae</c> is <c>--video-vae</c>);
        /// any other flag comes back unchanged. Matched case-insensitively.
        /// </summary>
        public static string CanonicalOptionSpelling(string flag) => CanonicalFlag(flag);

        private static readonly JsonDocumentOptions ParseOptions = new JsonDocumentOptions
        {
            AllowTrailingCommas = true,
            CommentHandling = JsonCommentHandling.Skip,
        };

        // ${name} and ${name:-fallback}. The fallback (shell syntax) is what lets a
        // shipped config name an environment variable that most users will not have
        // set and still work unmodified — see the portability note on Substitute.
        // The fallback alternation lets a default contain its own ${...} reference
        // (one level), so "${missing:-${base}/models}" parses as name=missing with
        // default "${base}/models" rather than stopping at the inner brace.
        private static readonly Regex VariablePattern =
            new(@"\$\{([A-Za-z0-9_.\-]+)(?::-((?:[^{}]|\$\{[^{}]*\})*))?\}", RegexOptions.Compiled);

        /// <summary>
        /// Expand every <c>--config &lt;path&gt;</c> flag in <paramref name="args"/>
        /// into the argv tokens it represents and return the merged argument
        /// list: file-derived tokens first (in flag order), then every other
        /// argument in its original order. The <c>--config</c> flags themselves
        /// are removed, and so is every file entry for a single-valued option the
        /// command line or a later file sets itself (its download, if any, is not
        /// attempted). When no <c>--config</c> flag is present the input array is
        /// returned unchanged. Download progress (when a file must be fetched) and
        /// the dropped-entry notices are written to <see cref="Console.Error"/>.
        /// </summary>
        /// <exception cref="ArgumentException">A <c>--config</c> flag has no path, or a config file is malformed.</exception>
        /// <exception cref="FileNotFoundException">A named config file, or a referenced file with no download URL, does not exist.</exception>
        /// <exception cref="IOException">A referenced file could not be downloaded from any URL.</exception>
        public static string[] Expand(string[] args) =>
            Expand(args, Console.Error, interactiveProgress: !Console.IsErrorRedirected);

        /// <summary>
        /// Testable overload: routes download progress to an explicit writer and
        /// controls whether progress overwrites a single line (TTY) or emits
        /// discrete lines (log sink).
        /// </summary>
        internal static string[] Expand(string[] args, TextWriter log, bool interactiveProgress)
        {
            if (args == null || args.Length == 0)
                return args ?? Array.Empty<string>();

            var configPaths = new List<string>();
            var passThrough = new List<string>(args.Length);

            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i];

                if (string.Equals(arg, ConfigFlag, StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 >= args.Length)
                        throw new ArgumentException($"Missing value for option '{ConfigFlag}'. Expected a path to a JSON configuration file.");
                    configPaths.Add(args[++i]);
                    continue;
                }

                const string inlinePrefix = ConfigFlag + "=";
                if (arg.StartsWith(inlinePrefix, StringComparison.OrdinalIgnoreCase))
                {
                    configPaths.Add(arg.Substring(inlinePrefix.Length));
                    continue;
                }

                passThrough.Add(arg);
            }

            if (configPaths.Count == 0)
                return args;

            // A removed option on the command line is refused here too, before any file
            // the configuration names is resolved or downloaded: the hosts' own check runs
            // after this expansion, which would otherwise fetch multi-gigabyte companions
            // first and only then report a configuration error.
            RemovedCliFlags.RejectRemoved(passThrough);

            // Every file is read before any is expanded: whether an entry is the winning
            // one depends on the files AFTER it as well as on the command line, and a
            // losing entry must be dropped before it is resolved, not after its download.
            var files = new List<LoadedConfig>(configPaths.Count);
            try
            {
                foreach (string configPath in configPaths)
                    files.Add(LoadConfig(configPath));

                var context = new ExpandContext(log ?? TextWriter.Null, interactiveProgress, CommandLineOptions(passThrough));
                for (int f = 0; f < files.Count; f++)
                    context.AddFileOptions(files[f].FullPath, OptionsSetBy(files[f]));

                var merged = new List<string>(args.Length);
                for (int f = 0; f < files.Count; f++)
                    ExpandFile(files[f], f, merged, context);
                context.ReportOverrides();
                merged.AddRange(passThrough);
                return merged.ToArray();
            }
            finally
            {
                foreach (LoadedConfig file in files)
                    file.Document.Dispose();
            }
        }

        /// <summary>
        /// The options <paramref name="args"/> sets, by canonical spelling, each mapped to
        /// the spelling the command line used: every token that starts with <c>--</c>, up to
        /// an <c>=</c> when it has one — the spaced and the joined spelling, matched
        /// case-insensitively as the server's and the shared option parsers match them.
        /// </summary>
        private static Dictionary<string, string> CommandLineOptions(IReadOnlyList<string> args)
        {
            var options = new Dictionary<string, string>(StringComparer.OrdinalIgnoreCase);
            foreach (string arg in args)
            {
                if (arg == null || arg.Length <= 2 || !arg.StartsWith("--", StringComparison.Ordinal))
                    continue;
                int equals = arg.IndexOf('=');
                string name = equals >= 0 ? arg.Substring(0, equals) : arg;
                options.TryAdd(CanonicalFlag(name), name);
            }
            return options;
        }

        /// <summary>
        /// The single-valued options a file sets, by canonical spelling: every entry that
        /// contributes tokens (a <c>false</c> switch or an empty array sets nothing, so it
        /// cannot beat an earlier file's value).
        /// </summary>
        private static HashSet<string> OptionsSetBy(LoadedConfig file)
        {
            var options = new HashSet<string>(StringComparer.OrdinalIgnoreCase);
            foreach (JsonProperty property in file.Document.RootElement.EnumerateObject())
            {
                if (ReservedKeys.Contains(property.Name) || !EmitsTokens(property.Value))
                    continue;
                string flag = NormalizeFlag(file.FullPath, property.Name);
                if (!AccumulatingFlags.Contains(flag))
                    options.Add(CanonicalFlag(flag));
            }
            return options;
        }

        private static string CanonicalFlag(string flag)
        {
            foreach (string[] spellings in SameOptionSpellings)
            {
                foreach (string spelling in spellings)
                {
                    if (string.Equals(flag, spelling, StringComparison.OrdinalIgnoreCase))
                        return spellings[0];
                }
            }
            return flag;
        }

        // A false/null switch or an empty array contributes no tokens, so it neither sets
        // an option nor is worth a line when it is dropped.
        private static bool EmitsTokens(JsonElement value) => value.ValueKind switch
        {
            JsonValueKind.False or JsonValueKind.Null => false,
            JsonValueKind.Array => value.GetArrayLength() > 0,
            _ => true,
        };

        private static bool NamesADownload(JsonElement value)
        {
            if (value.ValueKind == JsonValueKind.Object)
                return true;
            if (value.ValueKind != JsonValueKind.Array)
                return false;
            foreach (JsonElement element in value.EnumerateArray())
            {
                if (element.ValueKind == JsonValueKind.Object)
                    return true;
            }
            return false;
        }

        /// <summary>One parsed <c>--config</c> file, validated but not yet expanded.</summary>
        private sealed record LoadedConfig(string FullPath, string? Directory, JsonDocument Document);

        private sealed class ExpandContext
        {
            // Canonical option -> the spelling the command line used. Null outside Expand
            // (ResolveFileEntry resolves one entry and overrides nothing).
            private readonly Dictionary<string, string>? _commandLineOptions;
            // Per file, in --config order: the path and the single-valued options it sets.
            private readonly List<(string Path, HashSet<string> Options)> _files = new();
            // Canonical option -> the entries dropped for it, in the order they were met.
            private readonly Dictionary<string, List<(string Path, bool Download)>> _dropped =
                new(StringComparer.OrdinalIgnoreCase);
            private readonly List<string> _droppedOrder = new();

            public ExpandContext(TextWriter log, bool interactiveProgress, Dictionary<string, string>? commandLineOptions = null)
            {
                Log = log;
                InteractiveProgress = interactiveProgress;
                _commandLineOptions = commandLineOptions;
            }

            public TextWriter Log { get; }
            public bool InteractiveProgress { get; }

            public void AddFileOptions(string path, HashSet<string> options) => _files.Add((path, options));

            /// <summary>
            /// True when the entry for <paramref name="flag"/> in file <paramref name="fileIndex"/>
            /// does not win: the real command line sets the single-valued option itself, or a
            /// later file sets it again. Such an entry must be neither emitted nor resolved:
            /// another value wins anyway, and resolving the entry could download a file
            /// nothing will load.
            /// </summary>
            public bool IsOverridden(string flag, int fileIndex)
            {
                if (_commandLineOptions == null || AccumulatingFlags.Contains(flag))
                    return false;
                string canonical = CanonicalFlag(flag);
                if (_commandLineOptions.ContainsKey(canonical))
                    return true;
                for (int f = fileIndex + 1; f < _files.Count; f++)
                {
                    if (_files[f].Options.Contains(canonical))
                        return true;
                }
                return false;
            }

            /// <summary>Remember a dropped entry for <see cref="ReportOverrides"/>.</summary>
            public void RecordOverride(string configPath, string flag, JsonElement value)
            {
                if (!EmitsTokens(value))
                    return;
                string canonical = CanonicalFlag(flag);
                if (!_dropped.TryGetValue(canonical, out var entries))
                {
                    entries = new List<(string Path, bool Download)>();
                    _dropped[canonical] = entries;
                    _droppedOrder.Add(canonical);
                }
                entries.Add((configPath, NamesADownload(value)));
            }

            /// <summary>
            /// One line per option whose file entries were dropped: which value wins, which
            /// files lost, and every download entry that was skipped with them.
            /// </summary>
            public void ReportOverrides()
            {
                foreach (string canonical in _droppedOrder)
                {
                    List<(string Path, bool Download)> entries = _dropped[canonical];
                    string losers = Quoted(entries.Select(e => e.Path));
                    string[] downloads = entries.Where(e => e.Download).Select(e => e.Path).ToArray();
                    string skipped = downloads.Length == 0
                        ? string.Empty
                        : downloads.Length == entries.Count && entries.Count == 1
                            ? "; its download entry is skipped, so nothing is fetched for it"
                            : $"; the download entry in {Quoted(downloads)} is skipped, so nothing is fetched for it";
                    string valueFrom = entries.Count == 1 ? "the value from " + losers + " is" : "the values from " + losers + " are";

                    if (_commandLineOptions != null && _commandLineOptions.TryGetValue(canonical, out string? typed))
                    {
                        Log.WriteLine(
                            $"[config] {typed} is set on the command line, so {valueFrom} ignored and the " +
                            $"command-line value is used instead{skipped}.");
                    }
                    else
                    {
                        string winner = Path.GetFileName(_files.Last(f => f.Options.Contains(canonical)).Path);
                        Log.WriteLine(
                            $"[config] {canonical.ToLowerInvariant()} is set again by the later '{winner}', so " +
                            $"{valueFrom} ignored and '{winner}' wins{skipped}.");
                    }
                }
            }

            private static string Quoted(IEnumerable<string> paths)
            {
                string[] names = paths.Select(p => "'" + Path.GetFileName(p) + "'").Distinct().ToArray();
                return names.Length switch
                {
                    1 => names[0],
                    2 => names[0] + " and " + names[1],
                    _ => string.Join(", ", names, 0, names.Length - 1) + " and " + names[^1],
                };
            }
        }

        private static LoadedConfig LoadConfig(string path)
        {
            if (string.IsNullOrWhiteSpace(path))
                throw new ArgumentException($"Empty value for option '{ConfigFlag}'. Expected a path to a JSON configuration file.");

            string fullPath = Path.GetFullPath(path);
            if (!File.Exists(fullPath))
                throw new FileNotFoundException($"Configuration file not found: {fullPath}", fullPath);

            string json = File.ReadAllText(fullPath);

            JsonDocument document;
            try
            {
                document = JsonDocument.Parse(json, ParseOptions);
            }
            catch (JsonException ex)
            {
                throw new ArgumentException($"Configuration file '{fullPath}' is not valid JSON: {ex.Message}", ex);
            }

            try
            {
                JsonElement root = document.RootElement;
                if (root.ValueKind != JsonValueKind.Object)
                    throw new ArgumentException($"Configuration file '{fullPath}' must contain a JSON object at its root, but found {root.ValueKind}.");

                // A removed option is refused by name before any value is resolved: the
                // same key's download spec would otherwise fetch a file nothing can use,
                // and a `false` value (which expands to nothing) would hide the key from
                // the hosts' own check entirely.
                foreach (JsonProperty property in root.EnumerateObject())
                {
                    if (!ReservedKeys.Contains(property.Name) && RemovedCliFlags.Describe(property.Name) is { } removed)
                        throw new ArgumentException($"Configuration file '{fullPath}': {removed}");
                }
            }
            catch
            {
                document.Dispose();
                throw;
            }

            return new LoadedConfig(fullPath, Path.GetDirectoryName(fullPath), document);
        }

        private static void ExpandFile(LoadedConfig file, int fileIndex, List<string> output, ExpandContext context)
        {
            JsonElement root = file.Document.RootElement;
            var variables = VariableResolver.FromConfig(file.FullPath, root);

            foreach (JsonProperty property in root.EnumerateObject())
            {
                if (ReservedKeys.Contains(property.Name))
                    continue;
                AppendProperty(file.FullPath, file.Directory, fileIndex, variables, property.Name, property.Value, output, context);
            }
        }

        private static void AppendProperty(
            string configPath,
            string? configDirectory,
            int fileIndex,
            VariableResolver variables,
            string key,
            JsonElement value,
            List<string> output,
            ExpandContext context)
        {
            string flag = NormalizeFlag(configPath, key);

            // Another value wins anyway (the command line's, or a later file's), so leave
            // the entry out -- and above all do not resolve it: a download spec would fetch
            // a file nothing loads (--mmproj none beside a config that names a
            // multi-gigabyte vision shard).
            if (context.IsOverridden(flag, fileIndex))
            {
                context.RecordOverride(configPath, flag, value);
                return;
            }

            switch (value.ValueKind)
            {
                case JsonValueKind.True:
                    // Boolean switch (e.g. "think": true -> --think).
                    output.Add(flag);
                    break;

                case JsonValueKind.False:
                case JsonValueKind.Null:
                    // A disabled/absent switch contributes nothing. To turn an
                    // option off, name its explicit negation flag instead
                    // (e.g. "no-continuous-batching": true).
                    break;

                case JsonValueKind.String:
                    output.Add(flag);
                    output.Add(variables.Substitute(value.GetString()!));
                    break;

                case JsonValueKind.Number:
                    output.Add(flag);
                    output.Add(value.GetRawText());
                    break;

                case JsonValueKind.Object:
                    // Downloadable-file spec: { "path": ..., "urls": [...] }.
                    output.Add(flag);
                    output.Add(ResolveDownloadSpec(configPath, configDirectory, variables, key, value, context));
                    break;

                case JsonValueKind.Array:
                    // A strength or config binds to the one --lora before it; an array would emit
                    // them all after the last --lora, and all but the last would be lost.
                    if (string.Equals(flag, LoraCliFlags.ScaleFlag, StringComparison.OrdinalIgnoreCase) ||
                        string.Equals(flag, LoraCliFlags.ConfigFlag, StringComparison.OrdinalIgnoreCase))
                        throw new ArgumentException(
                            $"Configuration file '{configPath}': \"{key}\" binds to the one --lora before it and cannot be an array. " +
                            "Stack LoRAs with plug-in .json files that carry their own \"scale\", or give each --lora its own --lora-scale on the command line.");
                    foreach (JsonElement element in value.EnumerateArray())
                        AppendArrayElement(configPath, configDirectory, variables, key, flag, element, output, context);
                    break;

                default:
                    throw new ArgumentException(
                        $"Configuration file '{configPath}' option '{key}' has unsupported value type {value.ValueKind}.");
            }
        }

        private static void AppendArrayElement(
            string configPath,
            string? configDirectory,
            VariableResolver variables,
            string key,
            string flag,
            JsonElement element,
            List<string> output,
            ExpandContext context)
        {
            switch (element.ValueKind)
            {
                case JsonValueKind.String:
                    output.Add(flag);
                    output.Add(variables.Substitute(element.GetString()!));
                    break;
                case JsonValueKind.Number:
                    output.Add(flag);
                    output.Add(element.GetRawText());
                    break;
                case JsonValueKind.Object:
                    output.Add(flag);
                    output.Add(ResolveDownloadSpec(configPath, configDirectory, variables, key, element, context));
                    break;
                default:
                    throw new ArgumentException(
                        $"Configuration file '{configPath}' option '{key}' has an array element of type {element.ValueKind}; only strings, numbers, and download objects are supported in arrays.");
            }
        }

        /// <summary>
        /// Resolve the file entry <paramref name="key"/> of another JSON document that follows
        /// this class's conventions (a LoRA plug-in config, say): <c>"variables"</c> and
        /// <c>${name}</c> substitution, paths relative to the document, and a download spec
        /// object fetched (and hash-checked) when the file is missing. A plain string is a path.
        /// Returns null when the document has no such key.
        /// </summary>
        /// <exception cref="ArgumentException">The document or the entry is malformed.</exception>
        /// <exception cref="FileNotFoundException">The file is missing and no URL names it.</exception>
        /// <exception cref="IOException">No URL could supply the file.</exception>
        public static string? ResolveFileEntry(string documentPath, string key)
        {
            string fullPath = Path.GetFullPath(documentPath);
            JsonDocument document;
            try
            {
                document = JsonDocument.Parse(File.ReadAllText(fullPath), ParseOptions);
            }
            catch (JsonException ex)
            {
                throw new ArgumentException($"'{fullPath}' is not valid JSON: {ex.Message}", ex);
            }
            using (document)
            {
                JsonElement root = document.RootElement;
                if (root.ValueKind != JsonValueKind.Object || !root.TryGetProperty(key, out JsonElement entry))
                    return null;
                var variables = VariableResolver.FromConfig(fullPath, root);
                string? directory = Path.GetDirectoryName(fullPath);
                if (entry.ValueKind == JsonValueKind.Object)
                    return ResolveDownloadSpec(fullPath, directory, variables, key, entry,
                        new ExpandContext(Console.Error, interactiveProgress: !Console.IsErrorRedirected));
                if (entry.ValueKind != JsonValueKind.String)
                    throw new ArgumentException($"'{fullPath}': \"{key}\" must be a path or a download object.");
                string raw = variables.Substitute(entry.GetString()!);
                return Path.IsPathRooted(raw) ? Path.GetFullPath(raw) : Path.GetFullPath(Path.Combine(directory ?? ".", raw));
            }
        }

        /// <summary>
        /// Resolve a <c>{ "path": ..., "url"|"urls": ..., "sha256": ... }</c> object
        /// into a concrete local path, downloading the file from the first working
        /// URL when it is not already present. Relative paths resolve against the
        /// config file's directory so a config and its models can travel together.
        /// </summary>
        private static string ResolveDownloadSpec(
            string configPath,
            string? configDirectory,
            VariableResolver variables,
            string key,
            JsonElement spec,
            ExpandContext context)
        {
            if (!spec.TryGetProperty("path", out JsonElement pathElement) || pathElement.ValueKind != JsonValueKind.String)
                throw new ArgumentException(
                    $"Configuration file '{configPath}' option '{key}' is an object but has no string \"path\" field. " +
                    "A downloadable-file entry looks like {{ \"path\": \"...\", \"urls\": [ \"...\" ] }}.");

            string rawPath = variables.Substitute(pathElement.GetString()!);
            string localPath = Path.IsPathRooted(rawPath)
                ? Path.GetFullPath(rawPath)
                : Path.GetFullPath(Path.Combine(configDirectory ?? Directory.GetCurrentDirectory(), rawPath));

            var urls = ReadUrls(configPath, variables, key, spec);
            string? sha256 = spec.TryGetProperty("sha256", out JsonElement shaElement) && shaElement.ValueKind == JsonValueKind.String
                ? shaElement.GetString()
                : null;

            if (File.Exists(localPath))
            {
                context.Log.WriteLine($"[model-download] {key}: using cached file at {localPath}");
                return localPath;
            }

            if (urls.Count == 0)
                throw new FileNotFoundException(
                    $"Configuration file '{configPath}' option '{key}' path not found and no download URL was provided: {localPath}",
                    localPath);

            context.Log.WriteLine($"[model-download] {key}: '{localPath}' not found locally; attempting download from {urls.Count} source(s)");
            ModelDownloader.Download(localPath, urls, sha256, key, context.Log, context.InteractiveProgress);
            return localPath;
        }

        private static List<string> ReadUrls(string configPath, VariableResolver variables, string key, JsonElement spec)
        {
            var urls = new List<string>();

            if (spec.TryGetProperty("urls", out JsonElement urlsElement))
            {
                if (urlsElement.ValueKind != JsonValueKind.Array)
                    throw new ArgumentException($"Configuration file '{configPath}' option '{key}' has a \"urls\" field that is not an array.");
                foreach (JsonElement urlElement in urlsElement.EnumerateArray())
                {
                    if (urlElement.ValueKind != JsonValueKind.String)
                        throw new ArgumentException($"Configuration file '{configPath}' option '{key}' has a non-string entry in \"urls\".");
                    urls.Add(variables.Substitute(urlElement.GetString()!));
                }
            }

            if (spec.TryGetProperty("url", out JsonElement singleUrl))
            {
                if (singleUrl.ValueKind != JsonValueKind.String)
                    throw new ArgumentException($"Configuration file '{configPath}' option '{key}' has a \"url\" field that is not a string.");
                urls.Add(variables.Substitute(singleUrl.GetString()!));
            }

            return urls;
        }

        private static string NormalizeFlag(string configPath, string key)
        {
            if (string.IsNullOrWhiteSpace(key))
                throw new ArgumentException($"Configuration file '{configPath}' contains an empty option name.");

            string trimmed = key.Trim();
            return trimmed.StartsWith("--", StringComparison.Ordinal) ? trimmed : "--" + trimmed;
        }

        /// <summary>
        /// Resolves <c>${name}</c> references in string values against the config's
        /// <c>"variables"</c> object, falling back to environment variables, and
        /// supports variables that reference other variables (with cycle detection).
        /// </summary>
        private sealed class VariableResolver
        {
            private readonly string _configPath;
            private readonly Dictionary<string, string> _raw;       // as written in the file (may contain ${...})
            private readonly Dictionary<string, string> _resolved;  // fully expanded, memoised

            private VariableResolver(string configPath, Dictionary<string, string> raw)
            {
                _configPath = configPath;
                _raw = raw;
                _resolved = new Dictionary<string, string>(StringComparer.Ordinal);
            }

            public static VariableResolver FromConfig(string configPath, JsonElement root)
            {
                var raw = new Dictionary<string, string>(StringComparer.Ordinal);
                if (TryGetReserved(root, "variables", out JsonElement vars) || TryGetReserved(root, "vars", out vars))
                {
                    if (vars.ValueKind != JsonValueKind.Object)
                        throw new ArgumentException($"Configuration file '{configPath}' \"variables\" must be a JSON object.");
                    foreach (JsonProperty v in vars.EnumerateObject())
                    {
                        raw[v.Name] = v.Value.ValueKind switch
                        {
                            JsonValueKind.String => v.Value.GetString()!,
                            JsonValueKind.Number => v.Value.GetRawText(),
                            _ => throw new ArgumentException(
                                $"Configuration file '{configPath}' variable '{v.Name}' must be a string or number, but is {v.Value.ValueKind}."),
                        };
                    }
                }
                return new VariableResolver(configPath, raw);
            }

            public string Substitute(string input)
            {
                if (string.IsNullOrEmpty(input) || input.IndexOf("${", StringComparison.Ordinal) < 0)
                    return input;
                return Substitute(input, new HashSet<string>(StringComparer.Ordinal));
            }

            /// <summary>
            /// Replace every <c>${name}</c> / <c>${name:-fallback}</c> in
            /// <paramref name="input"/>.
            ///
            /// The fallback form exists for portability. A config that hard-codes a
            /// model root cannot be shipped: <c>C:/models/x.gguf</c> is not rooted on
            /// Linux or macOS (<see cref="Path.IsPathRooted"/> knows nothing about
            /// drive letters there), so it is treated as RELATIVE and silently glued
            /// onto the config's directory. Writing
            /// <c>${TENSORSHARP_MODELS:-../models}</c> instead gives one file that
            /// works unmodified everywhere: a path relative to the config by default,
            /// overridden by one environment variable when the models live elsewhere.
            /// </summary>
            private string Substitute(string input, HashSet<string> visiting)
            {
                return VariablePattern.Replace(input, match =>
                {
                    string name = match.Groups[1].Value;
                    if (TryResolve(name, visiting, out string? value))
                        return value!;
                    if (match.Groups[2].Success)
                        return Substitute(match.Groups[2].Value, visiting);
                    throw new ArgumentException(
                        $"Configuration file '{_configPath}' references undefined variable '${{{name}}}' " +
                        "(not found among \"variables\" or environment variables). Give it a default with " +
                        $"'${{{name}:-some/value}}' if it should be optional.");
                });
            }

            private bool TryResolve(string name, HashSet<string> visiting, out string? value)
            {
                if (_resolved.TryGetValue(name, out string? cached))
                {
                    value = cached;
                    return true;
                }

                if (_raw.TryGetValue(name, out string? rawValue))
                {
                    if (!visiting.Add(name))
                        throw new ArgumentException($"Configuration file '{_configPath}' has a cyclic variable reference involving '{name}'.");
                    string result = Substitute(rawValue, visiting);
                    visiting.Remove(name);
                    _resolved[name] = result;
                    value = result;
                    return true;
                }

                // An environment variable that exists but is empty counts as unset, so
                // `set TENSORSHARP_MODELS=` falls back rather than resolving to "".
                string? env = Environment.GetEnvironmentVariable(name);
                if (!string.IsNullOrEmpty(env))
                {
                    _resolved[name] = env;
                    value = env;
                    return true;
                }

                value = null;
                return false;
            }

            private static bool TryGetReserved(JsonElement root, string name, out JsonElement value)
            {
                foreach (JsonProperty p in root.EnumerateObject())
                {
                    if (string.Equals(p.Name, name, StringComparison.OrdinalIgnoreCase))
                    {
                        value = p.Value;
                        return true;
                    }
                }
                value = default;
                return false;
            }
        }
    }
}
