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
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;

namespace TensorSharp.Server.Endpoints
{
    /// <summary>
    /// Liveness check at <c>GET /</c> plus a SPA-style fallback that serves the
    /// Web UI's <c>index.html</c> for any path that doesn't match a route or
    /// static file.
    /// </summary>
    internal static class HealthEndpoints
    {
        public static IEndpointRouteBuilder MapHealthEndpoints(this IEndpointRouteBuilder endpoints, IWebHostEnvironment environment)
        {
            endpoints.MapGet("/", () => Results.Ok("TensorSharp.Server is running"));

            endpoints.MapFallback(async ctx =>
            {
                string root = environment.WebRootPath;
                if (root != null)
                {
                    var indexPath = Path.Combine(root, "index.html");
                    if (File.Exists(indexPath))
                    {
                        ctx.Response.ContentType = "text/html";
                        await ctx.Response.SendFileAsync(indexPath);
                        return;
                    }
                }
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsync("index.html not found. WebRootPath: " + (root ?? "(null)"));
            });

            return endpoints;
        }
    }
}
