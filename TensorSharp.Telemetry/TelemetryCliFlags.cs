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

namespace TensorSharp.Telemetry
{
    /// <summary>
    /// The one telemetry command-line flag, <c>--otel-endpoint &lt;url&gt;</c>, so a
    /// <c>--config</c> file can carry the collector address. It only sets
    /// <c>OTEL_EXPORTER_OTLP_ENDPOINT</c>; every other setting is environment-only, the way
    /// every collector's documentation already describes it.
    ///
    /// <para>
    /// The server refuses flags it does not recognise, so hosts consume this one before
    /// their own parsers run, the same way the code-execution flags are consumed.
    /// </para>
    /// </summary>
    public static class TelemetryCliFlags
    {
        public const string EndpointFlag = "--otel-endpoint";

        /// <summary>
        /// Remove <c>--otel-endpoint &lt;url&gt;</c> (or <c>--otel-endpoint=&lt;url&gt;</c>) from
        /// <paramref name="args"/>. The last occurrence wins, like every other single-valued
        /// option. Throws <see cref="ArgumentException"/> for a missing or relative URL.
        /// </summary>
        public static string[] Consume(string[] args, out string? endpoint)
        {
            ArgumentNullException.ThrowIfNull(args);
            endpoint = null;
            var remaining = new List<string>(args.Length);
            for (int i = 0; i < args.Length; i++)
            {
                string arg = args[i];
                if (string.Equals(arg, EndpointFlag, StringComparison.OrdinalIgnoreCase))
                {
                    if (i + 1 >= args.Length || args[i + 1].StartsWith("--", StringComparison.Ordinal))
                        throw new ArgumentException($"{EndpointFlag} needs a URL, e.g. {EndpointFlag} http://localhost:4317.");
                    endpoint = Validate(args[++i]);
                    continue;
                }
                if (arg.StartsWith(EndpointFlag + "=", StringComparison.OrdinalIgnoreCase))
                {
                    endpoint = Validate(arg[(EndpointFlag.Length + 1)..]);
                    continue;
                }
                remaining.Add(arg);
            }
            return remaining.ToArray();
        }

        /// <summary>
        /// Apply a consumed endpoint: the command line wins over the environment, as every
        /// other TensorSharp flag that maps to a variable does.
        /// </summary>
        public static void ApplyToEnvironment(string? endpoint)
        {
            if (!string.IsNullOrEmpty(endpoint))
                Environment.SetEnvironmentVariable(TelemetryOptions.OtlpEndpointEnvironmentVariable, endpoint);
        }

        private static string Validate(string value)
        {
            string trimmed = value.Trim();
            if (!Uri.TryCreate(trimmed, UriKind.Absolute, out Uri? uri) || (uri.Scheme != Uri.UriSchemeHttp && uri.Scheme != Uri.UriSchemeHttps))
                throw new ArgumentException($"{EndpointFlag} {TelemetryOptions.StripCredentials(trimmed)} is not an absolute http:// or https:// URL.");
            return trimmed;
        }
    }
}
