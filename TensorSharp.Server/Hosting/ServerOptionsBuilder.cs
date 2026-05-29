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
using System.Globalization;
using System.IO;
using System.Linq;
using TensorSharp.Runtime;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Reads CLI arguments and environment variables and produces a fully
    /// resolved <see cref="ServerHostingOptions"/>. Pure (no I/O beyond <see cref="Path"/>
    /// helpers and probing the host for supported backends), which makes it easy
    /// to test without spinning up a web app.
    /// </summary>
    internal static class ServerOptionsBuilder
    {
        private const int DefaultWebMaxTokensFallback = 20000;
        private const int DefaultMaxTextFileChars = 8000;

        public static ServerHostingOptions Build(string[] args, string baseDirectory)
        {
            if (args == null) throw new ArgumentNullException(nameof(args));
            if (string.IsNullOrEmpty(baseDirectory)) throw new ArgumentNullException(nameof(baseDirectory));

            ParseArgs(args,
                out string configuredModel,
                out string configuredMmProj,
                out string configuredBackend,
                out int? configuredMaxTokens,
                out SamplingOverrides configuredSampling);

            if (!string.IsNullOrWhiteSpace(configuredMmProj) && string.IsNullOrWhiteSpace(configuredModel))
                throw new ArgumentException("--mmproj requires --model.");

            string startupModelPath = ResolveConfiguredModelPath(configuredModel);
            string startupMmProjPath = ResolveConfiguredMmProjPath(configuredMmProj, startupModelPath);

            string backendInput = configuredBackend ?? Environment.GetEnvironmentVariable("BACKEND");
            string requestedBackend = backendInput ?? (OperatingSystem.IsMacOS() ? "ggml_metal" : "ggml_cpu");

            var supportedBackends = BackendCatalog.GetSupportedBackends().ToArray();
            string defaultBackend = BackendCatalog.ResolveDefaultBackend(requestedBackend, supportedBackends);

            int defaultWebMaxTokens = configuredMaxTokens
                ?? (TryParsePositiveInt(Environment.GetEnvironmentVariable("MAX_TOKENS"), out int envMaxTokens)
                    ? envMaxTokens
                    : DefaultWebMaxTokensFallback);

            int maxTextFileChars = DefaultMaxTextFileChars;
            string maxTextEnv = Environment.GetEnvironmentVariable("MAX_TEXT_FILE_CHARS");
            if (!string.IsNullOrEmpty(maxTextEnv) &&
                int.TryParse(maxTextEnv, out int envMaxTextChars) &&
                envMaxTextChars > 0)
            {
                maxTextFileChars = envMaxTextChars;
            }

            string uploadDirectory = Path.Combine(baseDirectory, "uploads");
            Directory.CreateDirectory(uploadDirectory);

            string logDirectory = Environment.GetEnvironmentVariable("TENSORSHARP_LOG_DIR");
            if (string.IsNullOrWhiteSpace(logDirectory))
                logDirectory = Path.Combine(baseDirectory, "logs");

            bool fileLoggingEnabled = !string.Equals(
                Environment.GetEnvironmentVariable("TENSORSHARP_LOG_FILE"),
                "0",
                StringComparison.Ordinal);

            SamplingConfig defaultSampling = ResolveDefaultSamplingConfig(configuredSampling);

            return new ServerHostingOptions(
                startupModelPath,
                startupMmProjPath,
                defaultBackend,
                supportedBackends,
                defaultWebMaxTokens,
                maxTextFileChars,
                uploadDirectory,
                logDirectory,
                fileLoggingEnabled,
                defaultSampling);
        }

        /// <summary>Backend originally requested via <c>--backend</c> / <c>BACKEND</c> (without the OS-default fallback).</summary>
        public static string ReadConfiguredBackendInput(string[] args)
        {
            ParseArgs(args, out _, out _, out string configuredBackend, out _, out _);
            return configuredBackend ?? Environment.GetEnvironmentVariable("BACKEND");
        }

        /// <summary>
        /// Translate <c>--continuous-batching</c> / <c>--no-continuous-batching</c>
        /// into the two env vars that gate the batched path:
        /// <c>TS_SCHED_DISABLE_BATCHED</c> (scheduler — falls through to
        /// per-sequence KV-swap when set) and <c>TS_QWEN35_BATCHED</c>
        /// (model — Qwen3.5 ForwardBatch gate; default ON, set to 0 to force
        /// the per-seq fallback). Both default to ON, so operators get
        /// paged-attention continuous batching without setting any env vars
        /// and without passing any flag; <c>--continuous-batching</c> is
        /// idempotent with the default, kept for explicit operator intent.
        /// <c>--no-continuous-batching</c> forces the per-seq path for every
        /// model.
        ///
        /// Must run before <see cref="SessionKvCacheManager"/> /
        /// <see cref="InferenceEngine"/> are constructed because both
        /// <c>BatchExecutor</c> and Qwen3.5's <c>SupportsBatchedMultimodal</c>
        /// read the env vars at runtime on each step.
        /// </summary>
        public static bool ApplyContinuousBatchingCliFlag(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            bool changed = false;
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                if (string.Equals(a, "--continuous-batching", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(a, "--paged-batching", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "0");
                    Environment.SetEnvironmentVariable("TS_QWEN35_BATCHED", "1");
                    changed = true;
                    continue;
                }
                if (string.Equals(a, "--no-continuous-batching", StringComparison.OrdinalIgnoreCase)
                    || string.Equals(a, "--no-paged-batching", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable("TS_SCHED_DISABLE_BATCHED", "1");
                    Environment.SetEnvironmentVariable("TS_QWEN35_BATCHED", "0");
                    changed = true;
                    continue;
                }
                // Tune chunked-prefill granularity. Each prefill chunk runs
                // as a single ExecuteStep that holds ModelBase.GpuComputeLock
                // for the duration of its forward pass, so smaller chunks
                // give parallel decode requests more frequent turns at the
                // GPU. Default 256 (see SchedulerConfig.MaxPrefillChunkSize).
                if (TryReadOption(args, ref i, "--prefill-chunk-size", out string chunkOpt))
                {
                    if (!int.TryParse(chunkOpt, out int chunk) || chunk <= 0)
                        throw new ArgumentException($"Invalid value for --prefill-chunk-size: '{chunkOpt}'.");
                    Environment.SetEnvironmentVariable("TS_SCHED_PREFILL_CHUNK", chunk.ToString());
                    changed = true;
                    continue;
                }
            }
            return changed;
        }

        /// <summary>
        /// Translate <c>--paged-kv*</c> CLI flags into the env vars consumed by
        /// <see cref="PagedKvCacheConfig.FromEnvironment"/>. Returns true when at
        /// least one flag was applied so the caller can emit a startup-log line.
        /// Must run before <see cref="SessionKvCacheManager"/> is constructed -
        /// the manager reads the env vars at field-initializer time.
        /// </summary>
        public static bool ApplyPagedKvCacheCliFlags(string[] args)
        {
            if (args == null || args.Length == 0)
                return false;

            bool changed = false;
            for (int i = 0; i < args.Length; i++)
            {
                string a = args[i];
                if (string.Equals(a, "--paged-kv", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(a, "--paged-kv-cache", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable("TS_KV_PAGED_CACHE", "1");
                    changed = true;
                    continue;
                }
                if (string.Equals(a, "--no-paged-kv", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(a, "--no-paged-kv-cache", StringComparison.OrdinalIgnoreCase))
                {
                    Environment.SetEnvironmentVariable("TS_KV_PAGED_CACHE", "0");
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-block-size", out string blockSizeOpt))
                {
                    if (!int.TryParse(blockSizeOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int blockSize) || blockSize <= 0)
                        throw new ArgumentException($"Invalid value for --paged-kv-block-size: '{blockSizeOpt}'.");
                    Environment.SetEnvironmentVariable("TS_KV_BLOCK_SIZE", blockSize.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-ram-mb", out string ramOpt))
                {
                    if (!long.TryParse(ramOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out long ramMb) || ramMb <= 0)
                        throw new ArgumentException($"Invalid value for --paged-kv-ram-mb: '{ramOpt}'.");
                    Environment.SetEnvironmentVariable("TS_KV_CACHE_MAX_RAM_MB", ramMb.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-ssd-dir", out string ssdDirOpt))
                {
                    Environment.SetEnvironmentVariable("TS_KV_CACHE_SSD_DIR", ssdDirOpt);
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-ssd-mb", out string ssdMbOpt))
                {
                    if (!long.TryParse(ssdMbOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out long ssdMb) || ssdMb <= 0)
                        throw new ArgumentException($"Invalid value for --paged-kv-ssd-mb: '{ssdMbOpt}'.");
                    Environment.SetEnvironmentVariable("TS_KV_CACHE_MAX_SSD_MB", ssdMb.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-quant-bits", out string quantBitsOpt))
                {
                    // Accepts 0 (explicit off), 4, or 8. The paged manager
                    // gates the codec a second time on model.RequiresPerBlockCapture,
                    // so requesting 4 / 8 on a recurrent-state model still
                    // falls back to passthrough at runtime - the value here
                    // just records the operator's intent.
                    if (!int.TryParse(quantBitsOpt, NumberStyles.Integer, CultureInfo.InvariantCulture, out int quantBits))
                        throw new ArgumentException($"Invalid value for --paged-kv-quant-bits: '{quantBitsOpt}'. Expected 0 (off), 4, or 8.");
                    if (quantBits != 0 && quantBits != 4 && quantBits != 8)
                        throw new ArgumentException($"Invalid value for --paged-kv-quant-bits: {quantBits}. Expected 0 (off), 4, or 8.");
                    Environment.SetEnvironmentVariable("TS_KV_PAGED_QUANT_BITS", quantBits.ToString(CultureInfo.InvariantCulture));
                    changed = true;
                    continue;
                }
            }
            return changed;
        }

        /// <summary>
        /// Bag of nullable sampling overrides captured from the CLI. We track
        /// each field separately (as <see cref="Nullable{T}"/>) so the caller
        /// can distinguish "operator pinned this value" from "operator didn't
        /// supply any sampling flags" - that distinction matters for the
        /// CLI &gt; env var &gt; type-default precedence.
        /// </summary>
        private struct SamplingOverrides
        {
            public float? Temperature;
            public int? TopK;
            public float? TopP;
            public float? MinP;
            public float? RepetitionPenalty;
            public float? PresencePenalty;
            public float? FrequencyPenalty;
            public int? Seed;
            public List<string> StopSequences;
        }

        private static void ParseArgs(
            string[] args,
            out string configuredModel,
            out string configuredMmProj,
            out string configuredBackend,
            out int? configuredMaxTokens,
            out SamplingOverrides configuredSampling)
        {
            configuredModel = null;
            configuredMmProj = null;
            configuredBackend = null;
            configuredMaxTokens = null;
            configuredSampling = default;

            for (int i = 0; i < args.Length; i++)
            {
                if (TryReadOption(args, ref i, "--model", out string modelOption))
                {
                    configuredModel = modelOption;
                    continue;
                }

                if (TryReadOption(args, ref i, "--mmproj", out string mmProjOption))
                {
                    configuredMmProj = mmProjOption;
                    continue;
                }

                if (TryReadOption(args, ref i, "--backend", out string backendOption))
                {
                    configuredBackend = backendOption;
                    continue;
                }

                if (TryReadOption(args, ref i, "--max-tokens", out string maxTokensOption))
                {
                    if (!TryParsePositiveInt(maxTokensOption, out int parsedMaxTokens))
                        throw new ArgumentException($"Invalid value for --max-tokens: '{maxTokensOption}'.");
                    configuredMaxTokens = parsedMaxTokens;
                    continue;
                }

                if (TryReadOption(args, ref i, "--temperature", out string tempOption))
                {
                    configuredSampling.Temperature = ParseFloat("--temperature", tempOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--top-k", out string topKOption))
                {
                    configuredSampling.TopK = ParseInt("--top-k", topKOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--top-p", out string topPOption))
                {
                    configuredSampling.TopP = ParseFloat("--top-p", topPOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--min-p", out string minPOption))
                {
                    configuredSampling.MinP = ParseFloat("--min-p", minPOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--repeat-penalty", out string repPenOption))
                {
                    configuredSampling.RepetitionPenalty = ParseFloat("--repeat-penalty", repPenOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--presence-penalty", out string presPenOption))
                {
                    configuredSampling.PresencePenalty = ParseFloat("--presence-penalty", presPenOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--frequency-penalty", out string freqPenOption))
                {
                    configuredSampling.FrequencyPenalty = ParseFloat("--frequency-penalty", freqPenOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--seed", out string seedOption))
                {
                    configuredSampling.Seed = ParseInt("--seed", seedOption);
                    continue;
                }

                if (TryReadOption(args, ref i, "--stop", out string stopOption))
                {
                    // The flag is repeatable so operators can pin multiple stop
                    // sequences (e.g. `--stop "</s>" --stop "<|eot|>"`).
                    configuredSampling.StopSequences ??= new List<string>();
                    configuredSampling.StopSequences.Add(stopOption);
                    continue;
                }

                // Hidden easter-egg flag consumed by the entry point to
                // enable the animated mascot banner. Recognised here so it
                // doesn't trip the unknown-arg trap below.
                if (string.Equals(args[i], "--xzf", StringComparison.Ordinal))
                {
                    continue;
                }

                // Paged-KV flags are consumed by ApplyPagedKvCacheCliFlags(args)
                // in a separate earlier pass. They still appear in args[] when
                // ParseArgs walks the list, so recognise + skip them here to
                // keep them out of the unknown-arg trap below.
                if (string.Equals(args[i], "--paged-kv", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--paged-kv-cache", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--no-paged-kv", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--no-paged-kv-cache", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                // Continuous-batching flag is also consumed by an earlier pass
                // (ApplyContinuousBatchingCliFlag). Skip here so ParseArgs
                // doesn't trip the unknown-arg trap.
                if (string.Equals(args[i], "--continuous-batching", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--paged-batching", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--no-continuous-batching", StringComparison.OrdinalIgnoreCase) ||
                    string.Equals(args[i], "--no-paged-batching", StringComparison.OrdinalIgnoreCase))
                {
                    continue;
                }
                if (TryReadOption(args, ref i, "--paged-kv-block-size", out _)
                    || TryReadOption(args, ref i, "--paged-kv-ram-mb", out _)
                    || TryReadOption(args, ref i, "--paged-kv-ssd-dir", out _)
                    || TryReadOption(args, ref i, "--paged-kv-ssd-mb", out _)
                    || TryReadOption(args, ref i, "--paged-kv-quant-bits", out _))
                {
                    continue;
                }

                // Anything else that starts with `--` is an unknown flag and we
                // refuse to start. Previously these were silently dropped, so a
                // typo like `--mproj <path>` (instead of `--mmproj`) would launch
                // the server with no vision projector and produce image-unrelated
                // output later. Fail fast so the operator sees the typo at
                // startup, not as a confusing inference bug.
                if (args[i].StartsWith("--", StringComparison.Ordinal))
                {
                    string suggestion = SuggestFlagCorrection(args[i]);
                    string suffix = suggestion != null ? $" Did you mean '{suggestion}'?" : string.Empty;
                    throw new ArgumentException($"Unknown option '{args[i]}'.{suffix}");
                }

                // Bare positional arg (no '--' prefix) — also unsupported but
                // produce a clearer error so the operator knows it's not a
                // value attached to an above option.
                throw new ArgumentException($"Unexpected positional argument '{args[i]}'.");
            }
        }

        /// <summary>Suggest a known flag that differs from the typo by one
        /// character (insertion, deletion, or substitution) — covers `--mproj`
        /// → `--mmproj`, `--temprature` → `--temperature`, etc. Returns null
        /// when no flag is within edit distance 2.</summary>
        private static string SuggestFlagCorrection(string typo)
        {
            string[] knownFlags = new[]
            {
                "--model", "--mmproj", "--backend", "--max-tokens",
                "--temperature", "--top-k", "--top-p", "--min-p",
                "--repeat-penalty", "--presence-penalty", "--frequency-penalty",
                "--seed", "--stop",
                "--paged-kv", "--paged-kv-cache", "--no-paged-kv", "--no-paged-kv-cache",
                "--paged-kv-block-size", "--paged-kv-ram-mb",
                "--paged-kv-ssd-dir", "--paged-kv-ssd-mb", "--paged-kv-quant-bits",
                "--continuous-batching", "--no-continuous-batching",
                "--paged-batching", "--no-paged-batching",
            };
            string best = null;
            int bestDist = int.MaxValue;
            foreach (var flag in knownFlags)
            {
                int d = LevenshteinDistance(typo, flag);
                if (d < bestDist) { bestDist = d; best = flag; }
            }
            // Only suggest if it's a near-miss (≤ 2 edits) — beyond that the
            // suggestion is more confusing than helpful.
            return bestDist <= 2 ? best : null;
        }

        private static int LevenshteinDistance(string a, string b)
        {
            if (string.IsNullOrEmpty(a)) return b?.Length ?? 0;
            if (string.IsNullOrEmpty(b)) return a.Length;
            int[,] d = new int[a.Length + 1, b.Length + 1];
            for (int i = 0; i <= a.Length; i++) d[i, 0] = i;
            for (int j = 0; j <= b.Length; j++) d[0, j] = j;
            for (int i = 1; i <= a.Length; i++)
                for (int j = 1; j <= b.Length; j++)
                {
                    int cost = a[i - 1] == b[j - 1] ? 0 : 1;
                    d[i, j] = Math.Min(Math.Min(d[i - 1, j] + 1, d[i, j - 1] + 1), d[i - 1, j - 1] + cost);
                }
            return d[a.Length, b.Length];
        }

        private static float ParseFloat(string flag, string value)
        {
            if (!float.TryParse(value, NumberStyles.Float, CultureInfo.InvariantCulture, out float parsed))
                throw new ArgumentException($"Invalid value for {flag}: '{value}'.");
            return parsed;
        }

        private static int ParseInt(string flag, string value)
        {
            if (!int.TryParse(value, NumberStyles.Integer, CultureInfo.InvariantCulture, out int parsed))
                throw new ArgumentException($"Invalid value for {flag}: '{value}'.");
            return parsed;
        }

        /// <summary>
        /// Layer environment-variable fallbacks under the CLI overrides. CLI wins
        /// (CLI args are the operator's most explicit intent), then env vars,
        /// then the type's built-in <see cref="SamplingConfig"/> defaults.
        /// Returning a fresh instance instead of <c>null</c> lets adapters call
        /// <see cref="SamplingConfig.Clone"/> on it without worrying about it
        /// being missing.
        /// </summary>
        private static SamplingConfig ResolveDefaultSamplingConfig(SamplingOverrides overrides)
        {
            var resolved = new SamplingConfig();

            if (overrides.Temperature.HasValue) resolved.Temperature = overrides.Temperature.Value;
            else if (TryReadEnvFloat("TENSORSHARP_TEMPERATURE", out float envTemp)) resolved.Temperature = envTemp;

            if (overrides.TopK.HasValue) resolved.TopK = overrides.TopK.Value;
            else if (TryReadEnvInt("TENSORSHARP_TOP_K", out int envTopK)) resolved.TopK = envTopK;

            if (overrides.TopP.HasValue) resolved.TopP = overrides.TopP.Value;
            else if (TryReadEnvFloat("TENSORSHARP_TOP_P", out float envTopP)) resolved.TopP = envTopP;

            if (overrides.MinP.HasValue) resolved.MinP = overrides.MinP.Value;
            else if (TryReadEnvFloat("TENSORSHARP_MIN_P", out float envMinP)) resolved.MinP = envMinP;

            if (overrides.RepetitionPenalty.HasValue) resolved.RepetitionPenalty = overrides.RepetitionPenalty.Value;
            else if (TryReadEnvFloat("TENSORSHARP_REPEAT_PENALTY", out float envRep)) resolved.RepetitionPenalty = envRep;

            if (overrides.PresencePenalty.HasValue) resolved.PresencePenalty = overrides.PresencePenalty.Value;
            else if (TryReadEnvFloat("TENSORSHARP_PRESENCE_PENALTY", out float envPres)) resolved.PresencePenalty = envPres;

            if (overrides.FrequencyPenalty.HasValue) resolved.FrequencyPenalty = overrides.FrequencyPenalty.Value;
            else if (TryReadEnvFloat("TENSORSHARP_FREQUENCY_PENALTY", out float envFreq)) resolved.FrequencyPenalty = envFreq;

            if (overrides.Seed.HasValue) resolved.Seed = overrides.Seed.Value;
            else if (TryReadEnvInt("TENSORSHARP_SEED", out int envSeed)) resolved.Seed = envSeed;

            // Stop sequences only support CLI overrides for now: the env var
            // would need an unambiguous list separator and that's overkill.
            if (overrides.StopSequences != null)
                resolved.StopSequences = new List<string>(overrides.StopSequences);

            return resolved;
        }

        private static bool TryReadEnvFloat(string name, out float value)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(raw))
            {
                value = 0f;
                return false;
            }
            return float.TryParse(raw, NumberStyles.Float, CultureInfo.InvariantCulture, out value);
        }

        private static bool TryReadEnvInt(string name, out int value)
        {
            string raw = Environment.GetEnvironmentVariable(name);
            if (string.IsNullOrWhiteSpace(raw))
            {
                value = 0;
                return false;
            }
            return int.TryParse(raw, NumberStyles.Integer, CultureInfo.InvariantCulture, out value);
        }

        private static bool TryReadOption(string[] args, ref int index, string option, out string value)
        {
            string arg = args[index];
            if (string.Equals(arg, option, StringComparison.OrdinalIgnoreCase))
            {
                if (index + 1 >= args.Length)
                    throw new ArgumentException($"Missing value for option '{option}'.");

                value = args[++index];
                return true;
            }

            string prefix = option + "=";
            if (arg.StartsWith(prefix, StringComparison.OrdinalIgnoreCase))
            {
                value = arg.Substring(prefix.Length);
                return true;
            }

            value = null;
            return false;
        }

        private static bool TryParsePositiveInt(string value, out int parsed)
        {
            if (int.TryParse(value, out parsed) && parsed > 0)
                return true;

            parsed = 0;
            return false;
        }

        private static string ResolveConfiguredModelPath(string configuredPath)
        {
            if (string.IsNullOrWhiteSpace(configuredPath))
                return null;

            return Path.GetFullPath(configuredPath);
        }

        private static string ResolveConfiguredMmProjPath(string configuredPath, string modelPath)
        {
            if (string.IsNullOrWhiteSpace(configuredPath))
                return null;

            if (string.Equals(configuredPath, "none", StringComparison.OrdinalIgnoreCase))
                return null;

            if (Path.IsPathRooted(configuredPath) ||
                configuredPath.IndexOf(Path.DirectorySeparatorChar) >= 0 ||
                configuredPath.IndexOf(Path.AltDirectorySeparatorChar) >= 0 ||
                File.Exists(configuredPath))
            {
                return Path.GetFullPath(configuredPath);
            }

            string preferredDirectory = Path.GetDirectoryName(modelPath);
            if (string.IsNullOrWhiteSpace(preferredDirectory))
                return Path.GetFullPath(configuredPath);

            return Path.GetFullPath(Path.Combine(preferredDirectory, configuredPath));
        }
    }
}
