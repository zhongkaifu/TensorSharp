// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Text.Json;
using TensorSharp.Chat.Studio;
using TensorSharp.Studio;
namespace TensorAgent.Core.Hosting;

public static partial class WebUiRoutes
{
    public static void MapStudio(this LoopbackServer server, StudioService service, string uploads)
    {
        var studio = new StudioHostBoundary(service, uploads);
        server.MapPost("/api/studio", async (request, ct) =>
        {
            if (!StudioAllowed(request)) return LoopbackResponse.Json(new { error = "Studio writes require a same-origin request and X-TensorSharp-Studio: 1." }, 403);
            try
            {
                using var doc = await JsonDocument.ParseAsync(new LimitedReadStream(request.Raw.InputStream, 1024 * 1024), cancellationToken: ct);
                return LoopbackResponse.Json(await studio.DispatchAsync(doc.RootElement, ct));
            }
            catch (Exception e) when (e is not OperationCanceledException) { return StudioError(e); }
        });
        server.MapPost("/api/studio/projects/{projectId}/import", async (request, ct) =>
        {
            if (!StudioAllowed(request)) return LoopbackResponse.Json(new { error = "Studio request rejected." }, 403);
            try { return StudioJsonResponse(await service.ImportAssetAsync(request.RouteValues["projectId"], request.Raw.InputStream, request.Query("fileName") ?? "", ct)); }
            catch (Exception e) when (e is not OperationCanceledException) { return StudioError(e); }
        });
        server.MapPost("/api/studio/projects/{projectId}/assets/{assetId}/relink", async (request, ct) =>
        {
            if (!StudioAllowed(request)) return LoopbackResponse.Json(new { error = "Studio request rejected." }, 403);
            try { return StudioJsonResponse(await service.RelinkAssetAsync(request.RouteValues["projectId"], request.RouteValues["assetId"], request.Raw.InputStream, ct)); }
            catch (Exception e) when (e is not OperationCanceledException) { return StudioError(e); }
        });
        server.MapGet("/api/studio/projects/{projectId}/assets/{assetId}", (request, _) =>
        {
            try
            {
                string path = service.GetAssetPath(request.RouteValues["projectId"], request.RouteValues["assetId"]);
                return Task.FromResult<LoopbackResponse?>(LoopbackResponse.File(path, ContentTypes.For(path)));
            }
            catch (Exception e) { return Task.FromResult<LoopbackResponse?>(StudioError(e)); }
        });
        server.MapGet("/api/studio/projects/{projectId}/jobs/{jobId}/output", (request, _) =>
        {
            try { return Task.FromResult<LoopbackResponse?>(LoopbackResponse.File(service.GetJobOutputPath(request.RouteValues["projectId"], request.RouteValues["jobId"]), "video/mp4")); }
            catch (Exception e) { return Task.FromResult<LoopbackResponse?>(StudioError(e)); }
        });
        server.MapGet("/api/studio/projects/{projectId}/package", async (request, ct) =>
        {
            try { return LoopbackResponse.File(await service.ExportPackageAsync(request.RouteValues["projectId"], ct), "application/zip", attachment: true, downloadName: "tensorstudio.tsstudio"); }
            catch (Exception e) when (e is not OperationCanceledException) { return StudioError(e); }
        });
        server.MapPost("/api/studio/packages/import", async (request, ct) =>
        {
            if (!StudioAllowed(request)) return LoopbackResponse.Json(new { error = "Studio request rejected." }, 403);
            try { return StudioJsonResponse(await service.ImportPackageAsync(request.Raw.InputStream, ct)); }
            catch (Exception e) when (e is not OperationCanceledException) { return StudioError(e); }
        });
        server.MapGet("/api/studio/thumbnail/{projectId}/{assetId}", async (request, ct) =>
        {
            try { return LoopbackResponse.File(await service.GetThumbnailAsync(request.RouteValues["projectId"], request.RouteValues["assetId"], ct), "image/jpeg"); }
            catch (Exception e) when (e is not OperationCanceledException) { return StudioError(e); }
        });
        server.MapGet("/api/studio/waveform/{projectId}/{assetId}", async (request, ct) =>
        {
            try { return StudioJsonResponse(await service.GetWaveformAsync(request.RouteValues["projectId"], request.RouteValues["assetId"], ct)); }
            catch (Exception e) when (e is not OperationCanceledException) { return StudioError(e); }
        });
    }
    private static LoopbackResponse StudioJsonResponse(object value) => LoopbackResponse.Json(JsonSerializer.SerializeToElement(value, StudioJson.Options));
    private static bool StudioAllowed(LoopbackRequest request) => StudioHostBoundary.AllowsMutation(request.Raw.Headers["X-TensorSharp-Studio"] ?? "", request.Raw.Headers["Origin"] ?? "", request.Raw.Url!.Scheme, request.Raw.Url.Authority);
    private static LoopbackResponse StudioError(Exception e) => LoopbackResponse.Json(new { error = e.Message, code = (e as StudioException)?.Code ?? "studio_error" }, StudioHostBoundary.Status(e));
}
