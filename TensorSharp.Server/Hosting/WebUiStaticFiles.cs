// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System;
using System.IO;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;

namespace TensorSharp.Server.Hosting;

/// <summary>Serves bundled UI code with revalidation so a host update reaches returning browsers.</summary>
public static class WebUiStaticFiles
{
    public static WebApplication UseWebUiStaticFiles(this WebApplication app, string baseDirectory)
    {
        ArgumentNullException.ThrowIfNull(app);
        ArgumentException.ThrowIfNullOrEmpty(baseDirectory);

        // Linked editor assets live in the build/publish output even when the
        // web root resolves to the project's source directory during dotnet run.
        app.MapGet("/mask-editor.js", (HttpContext context) =>
        {
            RequireRevalidation(context.Response);
            return Results.File(Path.Combine(baseDirectory, "wwwroot", "mask-editor.js"),
                "text/javascript; charset=utf-8");
        });
        app.MapGet("/mask-editor.css", (HttpContext context) =>
        {
            RequireRevalidation(context.Response);
            return Results.File(Path.Combine(baseDirectory, "wwwroot", "mask-editor.css"),
                "text/css; charset=utf-8");
        });
        app.UseDefaultFiles();
        app.UseStaticFiles(new StaticFileOptions
        {
            OnPrepareResponse = context =>
            {
                string extension = Path.GetExtension(context.File.Name);
                if (extension.Equals(".html", StringComparison.OrdinalIgnoreCase)
                    || extension.Equals(".js", StringComparison.OrdinalIgnoreCase)
                    || extension.Equals(".css", StringComparison.OrdinalIgnoreCase))
                    RequireRevalidation(context.Context.Response);
            },
        });
        return app;
    }

    internal static void RequireRevalidation(HttpResponse response) =>
        response.Headers.CacheControl = "no-cache";
}
