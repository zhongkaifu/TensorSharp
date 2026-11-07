// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.Diagnostics;
using TensorSharp.Runtime.Telemetry;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Logging;

namespace TensorSharp.Server.Host.Hosting
{
    /// <summary>
    /// The server's part of the OpenTelemetry setup that needs ASP.NET Core types, kept out
    /// of <c>TensorSharp.Telemetry</c> so the CLI, which references that project too, does
    /// not take an ASP.NET Core dependency.
    /// </summary>
    internal static class ServerTelemetry
    {
        /// <summary>Liveness probes: always left out of traces.</summary>
        internal const string HealthPath = "/health";

        /// <summary>
        /// Whether a request gets an HTTP server span. Health probes and the paths the request
        /// logger already demotes to Debug (<c>/api/queue/status</c>, which the Web UI polls on a
        /// tight loop) are left out, so they do not drown real requests in a trace backend.
        /// The <c>/health</c> and <c>/api/queue/status</c> endpoints are also left out of
        /// ASP.NET Core's <c>http.server.request.duration</c> (<c>DisableHttpMetrics()</c> where
        /// they are mapped); active-request counts still include them. A path an operator adds
        /// to <see cref="RequestLoggingOptions.LowNoisePaths"/> loses its span but keeps its
        /// duration metric.
        /// </summary>
        internal static bool ShouldTrace(HttpContext context)
        {
            string? path = context.Request.Path.Value;
            if (string.IsNullOrEmpty(path))
                return true;
            if (string.Equals(path, HealthPath, StringComparison.OrdinalIgnoreCase))
                return false;
            var options = context.RequestServices?.GetService<RequestLoggingOptions>();
            return options is null || !options.LowNoisePaths.Contains(path);
        }

        /// <summary>OpenTelemetry semantic-convention attribute the instrumentation sets from the request path.</summary>
        private const string UrlPathAttribute = "url.path";

        /// <summary>
        /// The ASP.NET Core instrumentation's <c>EnrichWithHttpRequest</c>. On a capability path
        /// (<see cref="CapabilityPaths"/>: an <c>/uploads</c> file, a code-execution artifact, a
        /// stored Responses API response) an unguessable id is all that guards an
        /// unauthenticated URL, so with content capture off the server span carries the
        /// placeholder (<c>/uploads/{file}</c>, <c>/api/code/artifacts/{run}/{path}</c>,
        /// <c>/v1/responses/{id}</c>) for <c>url.path</c> and in its name. Such requests then also
        /// group as one operation instead of one per file. Called only for recorded spans.
        /// </summary>
        internal static void Enrich(Activity activity, HttpRequest request)
        {
            if (TensorSharpTelemetry.CaptureMessageContent || CapabilityPaths.Mask(request.Path) is not { } masked)
                return;
            activity.SetTag(UrlPathAttribute, masked);
            activity.DisplayName = $"{request.Method} {masked}";
        }
    }
}
