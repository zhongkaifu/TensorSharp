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
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Chat;
using TensorSharp.Server.Endpoints;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.StreamingWriters;

namespace TensorSharp.Server.ProtocolAdapters;

/// <summary>
/// The ASP.NET Core transport for the bundled Web UI:
/// <list type="bullet">
///   <item>queue status (<c>GET /api/queue/status</c>)</item>
///   <item>session lifecycle (<c>POST /api/sessions</c>, <c>DELETE /api/sessions/{id}</c>)</item>
///   <item>model state + reload (<c>GET /api/models</c>, <c>POST /api/models/load</c>)</item>
///   <item>file upload (<c>POST /api/upload</c>)</item>
///   <item>image edit and video generation, plain and streamed</item>
///   <item>SSE chat stream (<c>POST /api/chat</c>)</item>
/// </list>
///
/// Every handler is a thin shell over <see cref="WebUiChatService"/> in
/// TensorSharp.Chat: read the body out of the <see cref="HttpContext"/>, call the
/// service, and write what comes back — a JSON payload with its status, a
/// <see cref="WebUiRequestRejectedException"/> as status + JSON, or a stream of
/// frames as <c>data:</c> events. The behaviour, the payload shapes and the
/// frame order all live in the service so the in-process hosts share them; nothing
/// here may reshape a payload.
/// </summary>
public sealed class WebUiAdapter
{
    private readonly WebUiChatService _service;
    private readonly ModelService _svc;
    private readonly InferenceQueue _queue;
    private readonly ServerHostingOptions _options;
    private readonly UploadStoragePolicy _uploads;
    private readonly ILoggerFactory _loggerFactory;

    public WebUiAdapter(
        ModelService svc,
        InferenceQueue queue,
        SessionManager sessions,
        ServerHostingOptions options,
        UploadStoragePolicy uploads,
        SkillRegistry skills,
        ICodeRunner codeRunner,
        SessionWorkspaceManager workspaces,
        CodeArtifactStore codeArtifacts,
        ILoggerFactory loggerFactory)
    {
        _service = new WebUiChatService(
            svc, sessions, options, uploads, skills, codeRunner, workspaces, codeArtifacts, loggerFactory,
            CodeArtifactEndpoints.RoutePrefix);
        _svc = svc;
        _queue = queue ?? throw new ArgumentNullException(nameof(queue));
        _options = options;
        _uploads = uploads;
        _loggerFactory = loggerFactory;
    }

    /// <summary>
    /// The chat service behind this adapter, for a host that needs to drive a turn
    /// without an HTTP request — the startup prefix-cache warm-up is the only caller.
    /// Exposed rather than duplicated because the warm-up is only worth anything if it
    /// renders the SAME prompt a real request renders, and that is guaranteed by it
    /// being the same service, not by two code paths agreeing.
    /// </summary>
    public WebUiChatService Chat => _service;

    /// <summary>A refusal, exactly as the service shaped it: its status and its JSON body.</summary>
    private static IResult Rejected(WebUiRequestRejectedException ex) =>
        Results.Json(ex.Payload, statusCode: ex.StatusCode);

    // ---- Queue ------------------------------------------------------------

    public IResult GetQueueStatus()
    {
        // The legacy queue's ticket count is what total_processed reported before
        // the engine completed anything; handed in so the field keeps its value.
        return Results.Json(_service.GetQueueStatus(_queue.GetStatus().TotalProcessed));
    }

    // ---- Sessions ---------------------------------------------------------

    public IResult CreateSession() => Results.Json(_service.CreateSession());

    public async Task<IResult> DisposeSessionAsync(string id, HttpContext ctx)
    {
        try
        {
            // Keep the legacy queue handshake for the API contract. The queue is a
            // no-op now; the ticket only feeds the total_processed count.
            using var ticket = _queue.Enqueue(ctx.RequestAborted);
            await ticket.WaitUntilReadyAsync().ConfigureAwait(false);
            return Results.Json(await _service.DisposeSessionAsync(id, ctx.RequestAborted).ConfigureAwait(false));
        }
        catch (WebUiRequestRejectedException ex)
        {
            return Rejected(ex);
        }
    }

    // ---- Models ----------------------------------------------------------

    public IResult GetModels() => Results.Json(_service.GetModels());

    public async Task<IResult> LoadModelAsync(HttpContext ctx, HttpRequest req)
    {
        var body = await JsonSerializer.DeserializeAsync<JsonElement>(req.Body).ConfigureAwait(false);
        try
        {
            using var ticket = _queue.Enqueue(ctx.RequestAborted);
            await ticket.WaitUntilReadyAsync().ConfigureAwait(false);
            return Results.Json(await _service.LoadModelAsync(body, ctx.RequestAborted).ConfigureAwait(false));
        }
        catch (WebUiRequestRejectedException ex)
        {
            return Rejected(ex);
        }
    }

    // ---- Upload ----------------------------------------------------------

    public async Task<IResult> UploadAsync(HttpRequest req)
    {
        var uploadLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Upload");
        if (!req.HasFormContentType)
        {
            uploadLogger.LogWarning(LogEventIds.UploadRejected,
                "Upload rejected: missing multipart form data");
            return Results.Json(new { error = "Expected multipart form data" }, statusCode: 400);
        }

        RaiseUploadRequestBodyLimit(req.HttpContext, _uploads.MaxFileBytes);
        var form = await req.ReadFormAsync().ConfigureAwait(false);
        var file = form.Files.Count == 0 ? null : form.Files[0];
        if (file == null)
        {
            uploadLogger.LogWarning(LogEventIds.UploadRejected,
                "Upload rejected: no file in request");
            return Results.Json(new { error = "No file uploaded" }, statusCode: 400);
        }

        try
        {
            Stream content = file.OpenReadStream();
            await using (content)
                return Results.Json(await _service.UploadAsync(content, file.FileName, file.Length, req.HttpContext.RequestAborted).ConfigureAwait(false));
        }
        catch (WebUiRequestRejectedException ex)
        {
            return Rejected(ex);
        }
    }

    /// <summary>
    /// Raise this request's body limit from the server-wide 500 MB to what the upload cap
    /// needs (<see cref="ServerOptionsBuilder.ResolveUploadRequestBodyBytes"/>). Only this
    /// route streams its file to disk, so only this route gets the larger limit; every JSON
    /// route keeps <see cref="ServerOptionsBuilder.DefaultMaxRequestBodyBytes"/>. Must run
    /// before the body is read. Returns true when the limit was raised.
    /// </summary>
    internal static bool RaiseUploadRequestBodyLimit(HttpContext context, long uploadMaxFileBytes)
    {
        var feature = context?.Features.Get<Microsoft.AspNetCore.Http.Features.IHttpMaxRequestBodySizeFeature>();
        if (feature == null || feature.IsReadOnly)
            return false;
        long limit = ServerOptionsBuilder.ResolveUploadRequestBodyBytes(uploadMaxFileBytes);
        // Null is "no limit"; never lower a limit something else already raised.
        if (feature.MaxRequestBodySize is not long current || current >= limit)
            return false;
        feature.MaxRequestBodySize = limit;
        return true;
    }

    // ---- Image editing (Qwen-Image-2.1) ----------------------------------

    /// <summary>
    /// <c>POST /api/image-edit</c> — multipart form with one or more <c>image</c> files and a
    /// <c>prompt</c> (plus optional <c>steps</c>, <c>cfg</c>, <c>seed</c>), or the same JSON
    /// the Web UI sends (<c>imagePaths</c> from <c>/api/upload</c>). Returns a downloadable
    /// URL to the generated PNG.
    /// </summary>
    public async Task<IResult> ImageEditAsync(HttpRequest req)
    {
        try
        {
            // Refused before the body is read — a wrong model or a full upload
            // directory must not cost the client a multi-megabyte upload first.
            _service.EnsureImageEditAvailable();

            if (req.HasFormContentType)
            {
                // Multipart: image file(s) + fields (direct API use). All parts named 'image'
                // (or every file part when none is) are taken in order.
                var form = await req.ReadFormAsync().ConfigureAwait(false);
                var files = form.Files.GetFiles("image");
                var fileList = files.Count > 0 ? files : form.Files;
                if (fileList.Count == 0)
                    return Results.Json(new { error = "No image uploaded (field 'image')." }, statusCode: 400);
                string prompt = form["prompt"].ToString();
                int steps = int.TryParse(form["steps"], out int s) ? s : 0;   // 0 = auto (40 for Qwen-Image-2.1)
                float cfg = float.TryParse(form["cfg"], out float c) ? c : 0f;  // 0 = auto (CFG 1.0 for Qwen-Image-2.1)
                long seed = long.TryParse(form["seed"], out long sd) ? sd : 0;
                long targetArea = long.TryParse(form["targetArea"], out long taf) && taf > 0 ? taf : 0;
                int width = int.TryParse(form["width"], out int wi) ? wi : 0;
                int height = int.TryParse(form["height"], out int he) ? he : 0;
                string negativePrompt = form.ContainsKey("negativePrompt") ? form["negativePrompt"].ToString() : " ";
                var imageBytesList = new List<byte[]>();
                foreach (var file in fileList)
                {
                    using var ms = new MemoryStream();
                    await file.CopyToAsync(ms).ConfigureAwait(false);
                    imageBytesList.Add(ms.ToArray());
                }
                return Results.Json(await _service.ImageEditAsync(
                    prompt, steps, cfg, seed, targetArea, imageBytesList, req.HttpContext.RequestAborted, width, height, negativePrompt).ConfigureAwait(false));
            }

            // JSON: { imagePaths[] or imagePath (server paths from /api/upload), prompt, steps, cfg, seed } (Web UI).
            using var body = await JsonDocument.ParseAsync(req.Body).ConfigureAwait(false);
            return Results.Json(await _service.ImageEditAsync(body.RootElement, req.HttpContext.RequestAborted).ConfigureAwait(false));
        }
        catch (WebUiRequestRejectedException ex)
        {
            return Rejected(ex);
        }
    }

    /// <summary>Test seam kept on the adapter: the upload-root confinement of image references lives in the service.</summary>
    internal Task<string> ReadUploadedImagesAsync(JsonElement root, List<byte[]> images, CancellationToken ct) =>
        _service.ReadUploadedImagesAsync(root, images, ct);

    /// <summary>
    /// <c>POST /api/image-edit/stream</c> — same JSON body as <see cref="ImageEditAsync"/> but
    /// streams Server-Sent Events so the Web UI can show live denoising progress.
    /// </summary>
    public async Task ImageEditStreamAsync(HttpContext ctx)
    {
        SseWriter.ApplyHeaders(ctx.Response);
        var ct = ctx.RequestAborted;

        // Same order as the plain route: model and quota are answered before the
        // body is read, as a { done, error } frame because the response has begun.
        try
        {
            _service.EnsureImageEditAvailable();
        }
        catch (WebUiRequestRejectedException ex)
        {
            await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = ex.Message }, ct).ConfigureAwait(false);
            return;
        }

        var (ok, body) = await ReadStreamBodyAsync(ctx).ConfigureAwait(false);
        if (!ok)
            return;
        await WriteFramesAsync(ctx, _service.ImageEditStreamAsync(body, ct), ct).ConfigureAwait(false);
    }

    /// <summary>Qwen-Image-2.1 JSON text-to-image endpoint; returns a PNG download URL.</summary>
    public async Task<IResult> ImageGenerateAsync(HttpRequest req)
    {
        try
        {
            _service.EnsureImageGenerationAvailable();
            using var body = await JsonDocument.ParseAsync(req.Body, cancellationToken: req.HttpContext.RequestAborted).ConfigureAwait(false);
            return Results.Json(await _service.ImageGenerateAsync(body.RootElement, req.HttpContext.RequestAborted).ConfigureAwait(false));
        }
        catch (WebUiRequestRejectedException ex) { return Rejected(ex); }
        catch (JsonException ex) { return Results.Json(new { error = "Bad request: " + ex.Message }, statusCode: 400); }
    }

    /// <summary>Text-to-image SSE with denoising progress and a final PNG download URL.</summary>
    public async Task ImageGenerateStreamAsync(HttpContext ctx)
    {
        SseWriter.ApplyHeaders(ctx.Response);
        var ct = ctx.RequestAborted;
        try { _service.EnsureImageGenerationAvailable(); }
        catch (WebUiRequestRejectedException ex)
        {
            await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = ex.Message }, ct).ConfigureAwait(false);
            return;
        }
        var (ok, body) = await ReadStreamBodyAsync(ctx).ConfigureAwait(false);
        if (ok) await WriteFramesAsync(ctx, _service.ImageGenerateStreamAsync(body, ct), ct).ConfigureAwait(false);
    }

    // ---- Text-to-video (Wan) ---------------------------------------------

    /// <summary>
    /// <c>POST /api/video-generate</c> — JSON <c>{ prompt, width?, height?, frames?, steps?,
    /// cfg?, seed?, fps?, flowShift?, negativePrompt? }</c>. Runs the loaded Wan text-to-video
    /// model and returns a downloadable URL to the generated MP4.
    /// </summary>
    public async Task<IResult> VideoGenerateAsync(HttpRequest req)
    {
        try
        {
            _service.EnsureVideoGenerationAvailable();
            using var body = await JsonDocument.ParseAsync(req.Body).ConfigureAwait(false);
            return Results.Json(await _service.VideoGenerateAsync(body.RootElement, req.HttpContext.RequestAborted).ConfigureAwait(false));
        }
        catch (WebUiRequestRejectedException ex)
        {
            return Rejected(ex);
        }
    }

    /// <summary>
    /// <c>POST /v1/videos/generations</c> — OpenAI-images-style envelope for Wan
    /// text-to-video: <c>{ prompt, size?: "832x480", frames?, steps?, cfg?, seed?,
    /// fps?, negative_prompt?, response_format?: "url"|"b64_json" }</c> returns
    /// <c>{ created, data: [{ url, b64_json? }], ... }</c>. The envelope is this
    /// route's own, so the checks are made here and only the generation is shared.
    /// </summary>
    public async Task<IResult> OpenAIVideoGenerationsAsync(HttpRequest req)
    {
        var logger = _loggerFactory.CreateLogger("TensorSharp.Server.VideoGenerate");
        if (_svc.Model is not TensorSharp.Models.Video.IVideoGenerationModel)
            return Results.BadRequest(new { error = new { message = "The loaded model is not a video-generation model.", type = "invalid_request_error" } });

        using var body = await JsonDocument.ParseAsync(req.Body).ConfigureAwait(false);
        var root = body.RootElement;
        string prompt = root.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
        if (string.IsNullOrWhiteSpace(prompt))
            return Results.BadRequest(new { error = new { message = "prompt is required.", type = "invalid_request_error" } });

        var p = VideoGenerationParamsParser.Parse(root, _options, out string imgError);
        if (imgError != null)
            return Results.BadRequest(new { error = new { message = imgError, type = "invalid_request_error" } });

        if (!_uploads.HasQuotaHeadroom(out string oaiVideoQuotaError))
        {
            logger.LogWarning(LogEventIds.UploadRejected, "OpenAI video generation rejected: {Reason}", oaiVideoQuotaError);
            return Results.Json(new { error = new { message = oaiVideoQuotaError, type = "server_error" } }, statusCode: 507);
        }

        if (root.TryGetProperty("size", out var sz) && sz.ValueKind == JsonValueKind.String)
        {
            var parts = (sz.GetString() ?? "").Split('x');
            if (parts.Length == 2 && int.TryParse(parts[0], out int sw_) && int.TryParse(parts[1], out int sh_))
            {
                p.Width = sw_;
                p.Height = sh_;
            }
        }
        if (root.TryGetProperty("negative_prompt", out var np2) && np2.ValueKind == JsonValueKind.String)
            p.NegativePrompt = np2.GetString() ?? string.Empty;
        bool wantB64 = root.TryGetProperty("response_format", out var rf) &&
                       rf.ValueKind == JsonValueKind.String && rf.GetString() == "b64_json";

        VideoGenerationResult result = await _service.GenerateVideoAsync(prompt, p).ConfigureAwait(false);

        if (logger.IsEnabled(LogLevel.Information))
            logger.LogInformation(LogEventIds.UploadReceived,
                "OpenAI video generation done: {F} frames -> {Url} ({Sec:F1}s)",
                result.Frames, result.Url, result.ElapsedSeconds);
        string? b64 = wantB64 ? Convert.ToBase64String(await File.ReadAllBytesAsync(result.OutputPath).ConfigureAwait(false)) : null;
        return Results.Json(new
        {
            created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            data = new[] { new { url = result.Url, b64_json = b64 } },
            audio_url = result.AudioUrl,
            width = result.Width,
            height = result.Height,
            frames = result.Frames,
            fps = result.Fps,
            seed = result.Seed,
            codec = result.Codec,
            elapsed_seconds = result.ElapsedSeconds,
        });
    }

    /// <summary>
    /// <c>POST /api/video-generate/stream</c> — same JSON body as
    /// <see cref="VideoGenerateAsync"/> but streams SSE progress events so the Web UI
    /// can show live progress.
    /// </summary>
    public async Task VideoGenerateStreamAsync(HttpContext ctx)
    {
        SseWriter.ApplyHeaders(ctx.Response);
        var ct = ctx.RequestAborted;

        try
        {
            _service.EnsureVideoGenerationAvailable();
        }
        catch (WebUiRequestRejectedException ex)
        {
            await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = ex.Message }, ct).ConfigureAwait(false);
            return;
        }

        var (ok, body) = await ReadStreamBodyAsync(ctx).ConfigureAwait(false);
        if (!ok)
            return;
        await WriteFramesAsync(ctx, _service.VideoGenerateStreamAsync(body, ct), ct).ConfigureAwait(false);
    }

    /// <summary>Test seam kept on the adapter; the warning text lives in the service.</summary>
    internal static string BuildIncompletePdfImageWarning(int extractedPages, int totalPages) =>
        WebUiChatService.BuildIncompletePdfImageWarning(extractedPages, totalPages);

    // ---- Chat (SSE) -------------------------------------------------------

    public async Task ChatStreamAsync(HttpContext ctx)
    {
        var webUiLogger = _loggerFactory.CreateLogger("TensorSharp.Server.WebUI.Chat");
        var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body).ConfigureAwait(false);

        using var ticket = _queue.Enqueue(ctx.RequestAborted);
        IAsyncEnumerator<object> frames = _service
            .ChatStreamAsync(body, ctx.RequestAborted)
            .GetAsyncEnumerator(ctx.RequestAborted);
        await using (frames.ConfigureAwait(false))
        {

            // The service refuses a request from its first MoveNextAsync, before any
            // frame; that is the only point at which a status code can still be sent.
            bool hasFrame;
            try
            {
                hasFrame = await frames.MoveNextAsync().ConfigureAwait(false);
            }
            catch (WebUiRequestRejectedException ex)
            {
                ctx.Response.StatusCode = ex.StatusCode;
                await ctx.Response.WriteAsJsonAsync(ex.Payload).ConfigureAwait(false);
                return;
            }
            if (!hasFrame)
                return;

            SseWriter.ApplyHeaders(ctx.Response);
            do
            {
                try
                {
                    await SseWriter.WriteEventAsync(ctx.Response, frames.Current, ctx.RequestAborted).ConfigureAwait(false);
                }
                catch (Exception ex)
                {
                    // The client has gone (an abort, a reset). The service sees the same
                    // token and ends its stream with done.aborted; there is nobody left
                    // to write that to, so the remaining frames are dropped here.
                    if (webUiLogger.IsEnabled(LogLevel.Debug))
                        webUiLogger.LogDebug(LogEventIds.ChatAborted,
                            "Web UI chat stream write failed; dropping the remaining frames: {Reason}", ex.Message);
                    return;
                }
            }
            while (await frames.MoveNextAsync().ConfigureAwait(false));
        }
    }

    // ---- helpers -----------------------------------------------------------

    /// <summary>
    /// The JSON body of a streaming route. A body that does not parse is answered
    /// with a <c>{ done, error: "Bad request: ..." }</c> frame — the response has
    /// already begun as an event stream, so a status code is no longer available.
    /// </summary>
    private static async Task<(bool Ok, JsonElement Body)> ReadStreamBodyAsync(HttpContext ctx)
    {
        try
        {
            using var doc = await JsonDocument.ParseAsync(ctx.Request.Body, cancellationToken: ctx.RequestAborted).ConfigureAwait(false);
            return (true, doc.RootElement.Clone());
        }
        catch (Exception ex)
        {
            await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = "Bad request: " + ex.Message }, ctx.RequestAborted).ConfigureAwait(false);
            return (false, default);
        }
    }

    private static async Task WriteFramesAsync(HttpContext ctx, IAsyncEnumerable<object> frames, CancellationToken ct)
    {
        try
        {
            await foreach (object frame in frames.WithCancellation(ct))
                await SseWriter.WriteEventAsync(ctx.Response, frame, ct).ConfigureAwait(false);
        }
        catch (OperationCanceledException)
        {
            // Client disconnected; the service observes the same token and its worker unwinds.
        }
    }
}
