// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.IO;
using System.Threading;
using System.Threading.Tasks;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.AspNetCore.Routing;
using Microsoft.Extensions.DependencyInjection;
using TensorSharp.Chat.Studio;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.ProtocolAdapters;
using TensorSharp.Studio;

namespace TensorSharp.Server.Endpoints;

public static class StudioEndpoints
{
    public sealed record AccessOptions(bool AllowRemote);
    public static IServiceCollection AddVideoStudio(this IServiceCollection services, string root)
    {
        services.AddSingleton(new AccessOptions(Environment.GetEnvironmentVariable("TS_STUDIO_ALLOW_REMOTE") == "1"));
        services.AddSingleton(sp =>
        {
            var chat = sp.GetRequiredService<WebUiAdapter>().Chat;
            var service = new StudioService(root, new NativeStudioGenerationAdapter(sp.GetRequiredService<ModelService>(), chat), StudioHostOptions.Create(sp.GetRequiredService<ServerHostingOptions>().UploadMaxFileBytes));
            chat.Studio = service;
            return service;
        });
        services.AddSingleton(sp => new StudioHostBoundary(sp.GetRequiredService<StudioService>(), sp.GetRequiredService<ServerHostingOptions>().UploadDirectory));
        return services;
    }

    public static IEndpointRouteBuilder MapVideoStudio(this IEndpointRouteBuilder endpoints)
    {
        var routes = endpoints.MapGroup("/api/studio");
        routes.AddEndpointFilter(async (context, next) =>
        {
            if (!ConnectionAllowed(context.HttpContext)) return Results.Json(new { error = "Video Studio is restricted to a loopback peer and loopback Host. Remote deployment requires explicit operator opt-in and authentication." }, statusCode: 403);
            return await next(context);
        });
        routes.MapPost("", async (HttpContext ctx, StudioHostBoundary studio) =>
        {
            if (!Allowed(ctx)) return Results.Json(new { error = "Studio writes require a same-origin request and X-TensorSharp-Studio: 1." }, statusCode: 403);
            if (ctx.Request.ContentLength > 1024 * 1024) return Results.StatusCode(413);
            try
            {
                using var doc = await JsonDocument.ParseAsync(new LimitedReadStream(ctx.Request.Body, 1024 * 1024), cancellationToken: ctx.RequestAborted);
                return Results.Json(await studio.DispatchAsync(doc.RootElement, ctx.RequestAborted), StudioJson.Options);
            }
            catch (Exception e) when (e is not OperationCanceledException) { return Error(e); }
        });
        routes.MapPost("/projects/{projectId}/import", async (HttpContext ctx, string projectId, StudioHostBoundary studio) =>
        {
            if (!Allowed(ctx)) return Results.StatusCode(403);
            WebUiAdapter.RaiseUploadRequestBodyLimit(ctx, studio.Service.Options.MaxImportBytes);
            try
            {
                var project = await studio.Service.ImportAssetAsync(projectId, ctx.Request.Body, ctx.Request.Query["fileName"].ToString(), ctx.RequestAborted);
                return Results.Json(project, StudioJson.Options);
            }
            catch (Exception e) when (e is not OperationCanceledException) { return Error(e); }
        });
        routes.MapPost("/projects/{projectId}/assets/{assetId}/relink", async (HttpContext ctx, string projectId, string assetId, StudioHostBoundary studio) =>
        {
            if (!Allowed(ctx)) return Results.StatusCode(403);
            WebUiAdapter.RaiseUploadRequestBodyLimit(ctx, studio.Service.Options.MaxImportBytes);
            try { return Results.Json(await studio.Service.RelinkAssetAsync(projectId, assetId, ctx.Request.Body, ctx.RequestAborted), StudioJson.Options); }
            catch (Exception e) when (e is not OperationCanceledException) { return Error(e); }
        });
        routes.MapGet("/projects/{projectId}/assets/{assetId}", (HttpContext ctx, string projectId, string assetId, StudioHostBoundary studio) =>
        {
            try
            {
                string path = studio.Service.GetAssetPath(projectId, assetId);
                ctx.Response.Headers.XContentTypeOptions = "nosniff";
                return Results.File(path, MediaType(path), enableRangeProcessing: true);
            }
            catch (Exception e) { return Error(e); }
        });
        routes.MapGet("/projects/{projectId}/jobs/{jobId}/output", (HttpContext ctx, string projectId, string jobId, StudioHostBoundary studio) =>
        {
            try
            {
                ctx.Response.Headers.XContentTypeOptions = "nosniff";
                return Results.File(studio.Service.GetJobOutputPath(projectId, jobId), "video/mp4", enableRangeProcessing: true);
            }
            catch (Exception e) { return Error(e); }
        });
        routes.MapGet("/projects/{projectId}/package", async (string projectId, StudioHostBoundary studio, CancellationToken ct) =>
        {
            try { return Results.File(await studio.Service.ExportPackageAsync(projectId, ct), "application/zip", "tensorstudio-" + projectId + ".tsstudio"); }
            catch (Exception e) when (e is not OperationCanceledException) { return Error(e); }
        });
        routes.MapPost("/packages/import", async (HttpContext ctx, StudioHostBoundary studio) =>
        {
            if (!Allowed(ctx)) return Results.StatusCode(403);
            WebUiAdapter.RaiseUploadRequestBodyLimit(ctx, studio.Service.Options.MaxImportBytes);
            try { return Results.Json(await studio.Service.ImportPackageAsync(ctx.Request.Body, ctx.RequestAborted), StudioJson.Options); }
            catch (Exception e) when (e is not OperationCanceledException) { return Error(e); }
        });
        routes.MapGet("/thumbnail/{projectId}/{assetId}", async (string projectId, string assetId, StudioHostBoundary studio, CancellationToken ct) =>
        {
            try { return Results.File(await studio.Service.GetThumbnailAsync(projectId, assetId, ct), "image/jpeg"); }
            catch (Exception e) when (e is not OperationCanceledException) { return Error(e); }
        });
        routes.MapGet("/waveform/{projectId}/{assetId}", async (string projectId, string assetId, StudioHostBoundary studio, CancellationToken ct) =>
        {
            try { return Results.Json(await studio.Service.GetWaveformAsync(projectId, assetId, ct), StudioJson.Options); }
            catch (Exception e) when (e is not OperationCanceledException) { return Error(e); }
        });
        return endpoints;
    }
    internal static bool ConnectionAllowed(HttpContext context) => StudioHostBoundary.AllowsServerConnection(context.Connection.RemoteIpAddress, context.Request.Host.Host, context.RequestServices.GetService<AccessOptions>()?.AllowRemote == true);
    private static bool Allowed(HttpContext c) => StudioHostBoundary.AllowsMutation(c.Request.Headers["X-TensorSharp-Studio"].ToString(), c.Request.Headers.Origin.ToString(), c.Request.Scheme, c.Request.Host.Value ?? "");
    private static IResult Error(Exception e) => Results.Json(new { error = e.Message, code = (e as StudioException)?.Code ?? "studio_error" }, statusCode: StudioHostBoundary.Status(e));
    internal static string MediaType(string path) => Path.GetExtension(path).ToLowerInvariant() switch
    { ".png" => "image/png", ".jpg" or ".jpeg" => "image/jpeg", ".webp" => "image/webp", ".wav" => "audio/wav", ".mp3" => "audio/mpeg", ".m4a" => "audio/mp4", ".webm" => "video/webm", ".mov" => "video/quicktime", _ => "video/mp4" };
}
