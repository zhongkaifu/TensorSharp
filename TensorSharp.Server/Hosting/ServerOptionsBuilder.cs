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
            }
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
