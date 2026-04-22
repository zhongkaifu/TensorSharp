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
using System.IO;
using System.Linq;

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
                out int? configuredMaxTokens);

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

            return new ServerHostingOptions(
                startupModelPath,
                startupMmProjPath,
                defaultBackend,
                supportedBackends,
                defaultWebMaxTokens,
                maxTextFileChars,
                uploadDirectory,
                logDirectory,
                fileLoggingEnabled);
        }

        /// <summary>Backend originally requested via <c>--backend</c> / <c>BACKEND</c> (without the OS-default fallback).</summary>
        public static string ReadConfiguredBackendInput(string[] args)
        {
            ParseArgs(args, out _, out _, out string configuredBackend, out _);
            return configuredBackend ?? Environment.GetEnvironmentVariable("BACKEND");
        }

        private static void ParseArgs(
            string[] args,
            out string configuredModel,
            out string configuredMmProj,
            out string configuredBackend,
            out int? configuredMaxTokens)
        {
            configuredModel = null;
            configuredMmProj = null;
            configuredBackend = null;
            configuredMaxTokens = null;

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
                }
            }
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
