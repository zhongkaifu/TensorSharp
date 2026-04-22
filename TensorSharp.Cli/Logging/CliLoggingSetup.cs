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
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Cli.Logging
{
    /// <summary>
    /// Builds an <see cref="ILoggerFactory"/> for the CLI host. The CLI is a
    /// short-lived process so we keep startup latency very low: a single console
    /// formatter plus an optional rolling file sink under the working directory.
    /// </summary>
    internal static class CliLoggingSetup
    {
        public sealed class Options
        {
            public LogLevel MinimumLevel { get; set; } = LogLevel.Information;
            public bool ConsoleEnabled { get; set; } = true;
            public bool FileEnabled { get; set; } = true;
            public string Directory { get; set; } = "logs";
            public string FilePrefix { get; set; } = "tensorsharp-cli";
        }

        /// <summary>
        /// Build a logger factory. The returned factory's lifetime is tied to the
        /// caller; callers should dispose it on exit so the file logger flushes.
        /// </summary>
        public static ILoggerFactory Build(Options options)
        {
            options ??= new Options();

            return LoggerFactory.Create(builder =>
            {
                builder.SetMinimumLevel(options.MinimumLevel);

                if (options.ConsoleEnabled)
                {
                    builder.AddSimpleConsole(simple =>
                    {
                        simple.IncludeScopes = false;
                        simple.SingleLine = true;
                        simple.TimestampFormat = "HH:mm:ss ";
                    });
                }

                if (options.FileEnabled)
                {
                    builder.AddTensorSharpFileLogger(file =>
                    {
                        file.Directory = ResolveDirectory(options.Directory);
                        file.FilePrefix = options.FilePrefix;
                        file.MinimumLevel = options.MinimumLevel;
                    });
                }
            });
        }

        /// <summary>
        /// Parse the logging-related CLI flags, leaving non-logging flags untouched
        /// so the surrounding command parser can continue to consume them. Recognised
        /// flags:
        /// <list type="bullet">
        /// <item><c>--log-level &lt;trace|debug|info|warning|error|critical|none&gt;</c></item>
        /// <item><c>--log-dir &lt;path&gt;</c></item>
        /// <item><c>--log-file &lt;0|1|off|on&gt;</c></item>
        /// <item><c>--log-console &lt;0|1|off|on&gt;</c></item>
        /// </list>
        /// Environment variables <c>TENSORSHARP_LOG_LEVEL</c> and
        /// <c>TENSORSHARP_LOG_DIR</c> provide defaults when no flag is given.
        /// </summary>
        public static Options ParseFromArgs(string[] args)
        {
            var options = new Options();

            string envLevel = Environment.GetEnvironmentVariable("TENSORSHARP_LOG_LEVEL");
            if (!string.IsNullOrWhiteSpace(envLevel) && TryParseLevel(envLevel, out var envLvl))
                options.MinimumLevel = envLvl;

            string envDir = Environment.GetEnvironmentVariable("TENSORSHARP_LOG_DIR");
            if (!string.IsNullOrWhiteSpace(envDir))
                options.Directory = envDir;

            string envFile = Environment.GetEnvironmentVariable("TENSORSHARP_LOG_FILE");
            if (!string.IsNullOrWhiteSpace(envFile))
                options.FileEnabled = ParseBool(envFile, options.FileEnabled);

            for (int i = 0; i < args.Length - 1; i++)
            {
                switch (args[i])
                {
                    case "--log-level":
                        if (TryParseLevel(args[i + 1], out var lvl))
                            options.MinimumLevel = lvl;
                        break;
                    case "--log-dir":
                        if (!string.IsNullOrWhiteSpace(args[i + 1]))
                            options.Directory = args[i + 1];
                        break;
                    case "--log-file":
                        options.FileEnabled = ParseBool(args[i + 1], options.FileEnabled);
                        break;
                    case "--log-console":
                        options.ConsoleEnabled = ParseBool(args[i + 1], options.ConsoleEnabled);
                        break;
                }
            }

            return options;
        }

        private static bool TryParseLevel(string value, out LogLevel level)
        {
            switch (value?.Trim().ToLowerInvariant())
            {
                case "trace": level = LogLevel.Trace; return true;
                case "debug": level = LogLevel.Debug; return true;
                case "info":
                case "information": level = LogLevel.Information; return true;
                case "warn":
                case "warning": level = LogLevel.Warning; return true;
                case "error": level = LogLevel.Error; return true;
                case "crit":
                case "critical": level = LogLevel.Critical; return true;
                case "off":
                case "none": level = LogLevel.None; return true;
                default: level = LogLevel.Information; return false;
            }
        }

        private static bool ParseBool(string value, bool fallback)
        {
            if (string.IsNullOrWhiteSpace(value)) return fallback;
            switch (value.Trim().ToLowerInvariant())
            {
                case "1":
                case "on":
                case "true":
                case "yes":
                    return true;
                case "0":
                case "off":
                case "false":
                case "no":
                    return false;
                default:
                    return fallback;
            }
        }

        private static string ResolveDirectory(string directory)
        {
            if (string.IsNullOrWhiteSpace(directory))
                return Path.Combine(AppContext.BaseDirectory, "logs");

            if (Path.IsPathRooted(directory))
                return directory;

            return Path.Combine(AppContext.BaseDirectory, directory);
        }
    }
}
