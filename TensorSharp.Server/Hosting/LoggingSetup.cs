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
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Logging;

namespace TensorSharp.Server.Hosting
{
    /// <summary>
    /// Wires up the application's logging providers. Console logging is left to
    /// ASP.NET Core's defaults; this only handles the optional file logger and
    /// the resolved minimum level.
    /// </summary>
    internal static class LoggingSetup
    {
        public const string AspNetCoreLoggerCategoryPrefix = "Microsoft.AspNetCore";

        public static LogLevel ResolveMinimumLevel()
        {
            return ParseLogLevel(Environment.GetEnvironmentVariable("TENSORSHARP_LOG_LEVEL")) ?? LogLevel.Information;
        }

        public static void Configure(ILoggingBuilder builder, ServerHostingOptions options, LogLevel minimumLevel)
        {
            if (builder == null) throw new ArgumentNullException(nameof(builder));
            if (options == null) throw new ArgumentNullException(nameof(options));

            // Keep ASP.NET Core request logs quiet by default while still surfacing warnings and errors.
            builder.AddFilter(AspNetCoreLoggerCategoryPrefix, LogLevel.Warning);

            if (options.FileLoggingEnabled)
            {
                builder.AddTensorSharpFileLogger(opt =>
                {
                    opt.Directory = options.LogDirectory;
                    opt.FilePrefix = "tensorsharp-server";
                    opt.MinimumLevel = minimumLevel;
                });
            }

            builder.SetMinimumLevel(minimumLevel);
        }

        private static LogLevel? ParseLogLevel(string raw)
        {
            if (string.IsNullOrWhiteSpace(raw))
                return null;
            return Enum.TryParse(raw, ignoreCase: true, out LogLevel parsed) ? parsed : (LogLevel?)null;
        }
    }
}
