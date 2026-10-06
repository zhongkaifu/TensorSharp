// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.

using System.Net;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Hosting.Server;
using Microsoft.AspNetCore.Hosting.Server.Features;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

public sealed class WebUiStaticFilesTests
{
    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task UiCode_RevalidatesInPublishedAndSourceTreeLayouts(bool sourceTreeWebRoot)
    {
        string directory = Directory.CreateTempSubdirectory("ts-webui-cache-test-").FullName;
        try
        {
            string baseDirectory = Path.Combine(directory, "output");
            string contentRoot = sourceTreeWebRoot ? Path.Combine(directory, "source") : baseDirectory;
            string webRoot = Path.Combine(contentRoot, "wwwroot");
            Directory.CreateDirectory(webRoot);
            Directory.CreateDirectory(Path.Combine(baseDirectory, "wwwroot"));
            string index = "<!doctype html><title>Current UI</title>";
            await File.WriteAllTextAsync(Path.Combine(webRoot, "index.html"), index);
            await File.WriteAllTextAsync(Path.Combine(webRoot, "app.js"), "window.currentUi = true;");
            await File.WriteAllTextAsync(Path.Combine(webRoot, "app.css"), "body { color: black; }");
            await File.WriteAllBytesAsync(Path.Combine(webRoot, "icon.png"), [1, 2, 3]);
            await File.WriteAllTextAsync(Path.Combine(baseDirectory, "wwwroot", "mask-editor.js"),
                "window.currentEditor = true;");
            await File.WriteAllTextAsync(Path.Combine(baseDirectory, "wwwroot", "mask-editor.css"),
                ".mask-editor { display: block; }");

            var builder = WebApplication.CreateBuilder(new WebApplicationOptions
            {
                ContentRootPath = contentRoot,
                WebRootPath = webRoot,
                EnvironmentName = "Production",
            });
            builder.Logging.ClearProviders();
            builder.WebHost.UseUrls("http://127.0.0.1:0");
            await using var app = builder.Build();
            app.UseWebUiStaticFiles(baseDirectory);
            app.MapHealthEndpoints(app.Environment);
            await app.StartAsync();
            string address = app.Services.GetRequiredService<IServer>()
                .Features.Get<IServerAddressesFeature>()!.Addresses.Single();
            using var client = new HttpClient { BaseAddress = new Uri(address) };

            foreach (string path in new[]
            {
                "/", "/index.html", "/chat/session", "/app.js", "/app.css",
                "/mask-editor.js?v=2", "/mask-editor.css?v=2",
            })
            {
                using var response = await client.GetAsync(path);
                Assert.Equal(HttpStatusCode.OK, response.StatusCode);
                Assert.True(response.Headers.CacheControl?.NoCache, $"{path} must revalidate cached UI code");
                if (path is "/" or "/index.html" or "/chat/session")
                    Assert.Equal(index, await response.Content.ReadAsStringAsync());
                if (path.StartsWith("/mask-editor.js", StringComparison.Ordinal))
                    Assert.Equal("window.currentEditor = true;", await response.Content.ReadAsStringAsync());
            }

            // Keep conditional requests available: unchanged code can return 304,
            // but that response must still require revalidation on the next visit.
            using var original = await client.GetAsync("/app.js");
            using var request = new HttpRequestMessage(HttpMethod.Get, "/app.js");
            request.Headers.IfNoneMatch.Add(original.Headers.ETag!);
            using var unchanged = await client.SendAsync(request);
            Assert.Equal(HttpStatusCode.NotModified, unchanged.StatusCode);
            Assert.True(unchanged.Headers.CacheControl?.NoCache);

            // Revalidation must replace code after an update rather than keep
            // answering 304 for the browser's previous cached response.
            string updatedCode = "window.currentUi = 'updated after host deployment';";
            await File.WriteAllTextAsync(Path.Combine(webRoot, "app.js"), updatedCode);
            using var updatedRequest = new HttpRequestMessage(HttpMethod.Get, "/app.js");
            updatedRequest.Headers.IfNoneMatch.Add(original.Headers.ETag!);
            using var updated = await client.SendAsync(updatedRequest);
            Assert.Equal(HttpStatusCode.OK, updated.StatusCode);
            Assert.Equal(updatedCode, await updated.Content.ReadAsStringAsync());
            Assert.NotEqual(original.Headers.ETag, updated.Headers.ETag);
            Assert.True(updated.Headers.CacheControl?.NoCache);

            using var icon = await client.GetAsync("/icon.png");
            Assert.Equal(HttpStatusCode.OK, icon.StatusCode);
            Assert.Null(icon.Headers.CacheControl);
            using var health = await client.GetAsync("/health");
            Assert.Null(health.Headers.CacheControl);
        }
        finally
        {
            Directory.Delete(directory, recursive: true);
        }
    }
}
