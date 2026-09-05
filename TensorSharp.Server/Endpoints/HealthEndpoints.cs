// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System.IO;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

namespace TensorSharp.Server.Endpoints
{
    /// <summary>
    /// Serves the Web UI's <c>index.html</c> at <c>GET /</c> so the bare
    /// host:port URL opens the chat UI, keeps the plain liveness response at
    /// <c>GET /health</c> (and at <c>GET /</c> for headless deployments that
    /// ship no <c>wwwroot</c> content), plus a SPA-style fallback that serves
    /// <c>index.html</c> for any path that doesn't match a route or static file.
    /// </summary>
    public static class HealthEndpoints
    {
        private const string LivenessMessage = "TensorSharp.Server is running";

        public static IEndpointRouteBuilder MapHealthEndpoints(
            this IEndpointRouteBuilder endpoints, IWebHostEnvironment environment, bool webUiEnabled = true)
        {
            // UseDefaultFiles() cannot rewrite "/" to "/index.html" for us:
            // WebApplication runs routing ahead of the static-file middleware,
            // which then bails out because this endpoint is already selected.
            // Sending the file from the endpoint itself is what makes a bare
            // http://host:port/ open the UI instead of the liveness text.
            // With the Web UI disabled, "/" always answers the liveness text.
            endpoints.MapGet("/", async ctx =>
            {
                if (webUiEnabled && await TrySendIndexAsync(ctx, environment))
                    return;
                await Results.Ok(LivenessMessage).ExecuteAsync(ctx);
            });

            // Liveness probes that want the plain response rather than the UI
            // keep a stable route now that "/" serves index.html.
            endpoints.MapGet("/health", () => Results.Ok(LivenessMessage));

            endpoints.MapFallback(async ctx =>
            {
                if (webUiEnabled && await TrySendIndexAsync(ctx, environment))
                    return;
                // With the UI enabled, a fallback hit means index.html is
                // missing: the resolved web root goes to the server log for the
                // operator diagnosing a misdeployed wwwroot. The response stays
                // generic either way so clients don't learn host filesystem
                // paths.
                if (webUiEnabled)
                {
                    ctx.RequestServices.GetService<ILoggerFactory>()
                        ?.CreateLogger("TensorSharp.Server.Health")
                        .LogWarning(LogEventIds.HttpRequestRejected,
                            "Fallback route hit but index.html is missing. WebRootPath: {WebRootPath}",
                            environment.WebRootPath ?? "(null)");
                }
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsync(webUiEnabled ? "index.html not found." : "Not found.");
            });

            return endpoints;
        }

        private static async Task<bool> TrySendIndexAsync(HttpContext ctx, IWebHostEnvironment environment)
        {
            string root = environment.WebRootPath;
            if (string.IsNullOrEmpty(root))
                return false;

            var indexPath = Path.Combine(root, "index.html");
            if (!File.Exists(indexPath))
                return false;

            ctx.Response.ContentType = "text/html";
            await ctx.Response.SendFileAsync(indexPath);
            return true;
        }
    }
}
