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
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Text.Json;
using System.Threading;
using System.Threading.Channels;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Logging;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.StreamingWriters;

namespace TensorSharp.Server.ProtocolAdapters
{
    /// <summary>
    /// Implements the request handlers used by the bundled Web UI:
    /// <list type="bullet">
    ///   <item>queue status (<c>GET /api/queue/status</c>)</item>
    ///   <item>session lifecycle (<c>POST /api/sessions</c>, <c>DELETE /api/sessions/{id}</c>)</item>
    ///   <item>model state + reload (<c>GET /api/models</c>, <c>POST /api/models/load</c>)</item>
    ///   <item>file upload (<c>POST /api/upload</c>)</item>
    ///   <item>SSE chat stream (<c>POST /api/chat</c>)</item>
    /// </list>
    ///
    /// The adapter owns NO state of its own; everything is injected (model
    /// service, queue, session manager, configuration, loggers). That means a
    /// single instance can be reused across requests and easily faked in tests.
    /// </summary>
    internal sealed class WebUiAdapter
    {
        private readonly ModelService _svc;
        private readonly InferenceQueue _queue;
        private readonly SessionManager _sessions;
        private readonly ServerHostingOptions _options;
        private readonly ILoggerFactory _loggerFactory;

        public WebUiAdapter(
            ModelService svc,
            InferenceQueue queue,
            SessionManager sessions,
            ServerHostingOptions options,
            ILoggerFactory loggerFactory)
        {
            _svc = svc ?? throw new ArgumentNullException(nameof(svc));
            _queue = queue ?? throw new ArgumentNullException(nameof(queue));
            _sessions = sessions ?? throw new ArgumentNullException(nameof(sessions));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        }

        // ---- Queue ------------------------------------------------------------

        public IResult GetQueueStatus()
        {
            // Real concurrency now lives in the per-model inference engine, not the
            // legacy InferenceQueue (which always reports zero). Peek the engine's
            // live counters so the Web UI can show, in real time, how many requests
            // are being processed concurrently versus waiting for admission.
            // TryGetLiveStats is side-effect free and returns false before the
            // engine is built (no model loaded / no request yet), in which case
            // everything is idle.
            _svc.EngineHost.TryGetLiveStats(out int processing, out int waiting, out long totalCompleted);

            // total_processed kept for API compatibility; sourced from the engine's
            // completed count (per loaded model) rather than the legacy queue.
            long totalProcessed = totalCompleted != 0 ? totalCompleted : _queue.GetStatus().TotalProcessed;

            return Results.Ok(new
            {
                busy = processing > 0,
                // Number of requests currently being generated concurrently.
                processing,
                // Requests admitted to the engine but still waiting for a batch slot.
                pending_requests = waiting,
                total_processed = totalProcessed,
            });
        }

        // ---- Sessions ---------------------------------------------------------

        public IResult CreateSession()
        {
            var sessionsLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Sessions");
            var session = _sessions.CreateSession();
            sessionsLogger.LogInformation(LogEventIds.SessionCreated,
                "Created session via /api/sessions: {SessionId}", session.Id);
            return Results.Json(new
            {
                sessionId = session.Id,
                createdAt = session.CreatedAt.ToString("o"),
            });
        }

        public async Task<IResult> DisposeSessionAsync(string id, HttpContext ctx)
        {
            var sessionsLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Sessions");
            if (string.Equals(id, SessionManager.DefaultSessionId, StringComparison.Ordinal))
            {
                sessionsLogger.LogWarning(LogEventIds.SessionRemoved,
                    "Refused to dispose default session via API: {SessionId}", id);
                return Results.BadRequest(new { ok = false, error = "Cannot dispose the default session." });
            }

            var removed = _sessions.TryRemove(id);
            if (removed == null)
            {
                sessionsLogger.LogWarning(LogEventIds.SessionRemoved,
                    "Session not found for disposal: {SessionId}", id);
                return Results.NotFound(new { ok = false, error = $"Session '{id}' not found." });
            }

            // Keep the legacy queue handshake for the API contract. The queue is a
            // no-op now; in-flight KV state is owned by the engine, while disposing
            // the session only clears tracked chat history.
            using var ticket = _queue.Enqueue(ctx.RequestAborted);
            await ticket.WaitUntilReadyAsync();
            _svc.DisposeSession(removed);
            sessionsLogger.LogInformation(LogEventIds.SessionDisposed,
                "Disposed session via /api/sessions: {SessionId}", id);
            return Results.Json(new { ok = true, sessionId = id });
        }

        // ---- Models ----------------------------------------------------------

        public IResult GetModels()
        {
            var files = string.IsNullOrWhiteSpace(_options.StartupModelPath)
                ? new List<string>()
                : new List<string> { Path.GetFileName(_options.StartupModelPath) };
            var mmProjFiles = string.IsNullOrWhiteSpace(_options.StartupMmProjPath)
                ? new List<string>()
                : new List<string> { Path.GetFileName(_options.StartupMmProjPath) };
            return Results.Json(new
            {
                models = files,
                mmProjModels = mmProjFiles,
                loaded = _svc.LoadedModelName,
                loadedMmProj = _svc.LoadedMmProjName,
                loadedBackend = _svc.LoadedBackend,
                defaultBackend = _options.DefaultBackend,
                supportedBackends = _options.SupportedBackends,
                architecture = _svc.Architecture,
                hostedModelPath = _options.StartupModelPath,
                hostedMmProjPath = _options.StartupMmProjPath,
                defaultMaxTokens = _options.DefaultWebMaxTokens,
            });
        }

        public async Task<IResult> LoadModelAsync(HttpContext ctx, HttpRequest req)
        {
            var modelLoadLogger = _loggerFactory.CreateLogger("TensorSharp.Server.WebUI.ModelLoad");
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(req.Body);
            string modelName = body.GetProperty("model").GetString();
            string requestedBackend = body.TryGetProperty("backend", out var b) ? b.GetString() : null;
            string mmproj = body.TryGetProperty("mmproj", out var m) ? m.GetString() : null;

            modelLoadLogger.LogInformation(LogEventIds.ModelLoadStarted,
                "Web UI model load request: model={Model} backend={Backend} mmproj={MmProj}",
                modelName, requestedBackend ?? "(default)", mmproj ?? "(none)");

            if (!BackendSelector.TryResolveSupportedBackend(_options, requestedBackend, out string backend, out string backendError))
            {
                modelLoadLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "Web UI model load rejected: {Reason}", backendError);
                return Results.BadRequest(new { ok = false, error = backendError });
            }

            if (!HostedModelGuard.TryResolveHostedModelRequest(modelName, _options.StartupModelPath, out string modelPath, out string modelError))
            {
                modelLoadLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "Web UI model load rejected: {Reason}", modelError);
                return Results.BadRequest(new { ok = false, error = modelError });
            }

            if (!HostedModelGuard.TryValidateHostedMmProjRequest(mmproj, _options.StartupMmProjPath, out string mmProjError))
            {
                modelLoadLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "Web UI mmproj validation failed: {Reason}", mmProjError);
                return Results.BadRequest(new { ok = false, error = mmProjError });
            }

            using var ticket = _queue.Enqueue(ctx.RequestAborted);
            await ticket.WaitUntilReadyAsync();

            try
            {
                _svc.LoadModel(modelPath, _options.StartupMmProjPath, backend);
                return Results.Json(new
                {
                    ok = true,
                    model = _svc.LoadedModelName,
                    loadedMmProj = _svc.LoadedMmProjName,
                    architecture = _svc.Architecture,
                });
            }
            catch (Exception ex)
            {
                modelLoadLogger.LogError(LogEventIds.ModelLoadFailed, ex,
                    "Web UI model load failed: model={Model} backend={Backend}", modelName, backend);
                return Results.Json(new { ok = false, error = ex.Message }, statusCode: 500);
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
                return Results.BadRequest(new { error = "Expected multipart form data" });
            }

            var form = await req.ReadFormAsync();
            var file = form.Files.FirstOrDefault();
            if (file == null)
            {
                uploadLogger.LogWarning(LogEventIds.UploadRejected,
                    "Upload rejected: no file in request");
                return Results.BadRequest(new { error = "No file uploaded" });
            }

            string ext = Path.GetExtension(file.FileName).ToLowerInvariant();
            string safeFileName = $"{Guid.NewGuid():N}{ext}";
            string savePath = Path.Combine(_options.UploadDirectory, safeFileName);
            string uploadUrl = BuildUploadUrl(safeFileName);

            using (var stream = File.Create(savePath))
                await file.CopyToAsync(stream);

            string mediaType = ClassifyExtension(ext);

            // Include the full saved path and the classified media type so this entry
            // is self-sufficient for tracing back from the per-turn chat log
            // (which records each attachment by its saved path).
            uploadLogger.LogInformation(LogEventIds.UploadReceived,
                "Upload received: name={FileName} ext={Extension} mediaType={MediaType} bytes={Length} savedAs={SavedFile} savedPath={SavedPath}",
                file.FileName, ext, mediaType, file.Length, safeFileName, savePath);

            if (mediaType == "video")
            {
                var frames = MediaHelper.ExtractVideoFrames(savePath);
                return Results.Json(new
                {
                    ok = true,
                    path = savePath,
                    url = uploadUrl,
                    mediaType,
                    fileName = file.FileName,
                    frames = frames.Select(f => Path.GetFileName(f)).ToList(),
                    frameUrls = frames.Select(f => BuildUploadUrl(Path.GetFileName(f))).ToList(),
                    framePaths = frames,
                });
            }

            if (mediaType == "text")
            {
                string textContent = await File.ReadAllTextAsync(savePath);
                var prepared = TextUploadHelper.PrepareTextContent(
                    textContent,
                    _svc.Model?.Tokenizer,
                    _svc.Model?.MaxContextLength ?? 0,
                    _options.MaxTextFileChars);

                return Results.Json(new
                {
                    ok = true,
                    path = savePath,
                    url = uploadUrl,
                    mediaType,
                    fileName = file.FileName,
                    textContent = prepared.TextContent,
                    truncated = prepared.Truncated,
                    truncateLimit = prepared.TruncateLimit,
                    truncateUnit = prepared.TruncateUnit,
                    modelContextLimit = prepared.ModelContextLimit,
                    originalTokenCount = prepared.OriginalTokenCount,
                    returnedTokenCount = prepared.ReturnedTokenCount,
                });
            }

            return Results.Json(new { ok = true, path = savePath, url = uploadUrl, mediaType, fileName = file.FileName });
        }

        // ---- Image editing (Qwen-Image-Edit) ---------------------------------

        private static readonly object _imageEditLock = new();

        /// <summary>
        /// <c>POST /api/image-edit</c> — multipart form with an <c>image</c> file and a
        /// <c>prompt</c> (plus optional <c>steps</c>, <c>cfg</c>, <c>seed</c>). Runs the loaded
        /// Qwen-Image-Edit model and returns a downloadable URL to the generated PNG.
        /// </summary>
        public async Task<IResult> ImageEditAsync(HttpRequest req)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.ImageEdit");
            if (_svc.Model is not TensorSharp.Models.QwenImage.QwenImageModel editModel)
                return Results.BadRequest(new { error = "The loaded model is not a Qwen-Image-Edit model." });

            string prompt; int steps; float cfg; long seed; long targetArea = 0;
            byte[] imageBytes;

            if (req.HasFormContentType)
            {
                // Multipart: image file + fields (direct API use).
                var form = await req.ReadFormAsync();
                var file = form.Files.GetFile("image") ?? form.Files.FirstOrDefault();
                if (file == null)
                    return Results.BadRequest(new { error = "No image uploaded (field 'image')." });
                prompt = form["prompt"].ToString();
                steps = int.TryParse(form["steps"], out int s) ? s : 28;
                cfg = float.TryParse(form["cfg"], out float c) ? c : 4.0f;
                seed = long.TryParse(form["seed"], out long sd) ? sd : 0;
                if (long.TryParse(form["targetArea"], out long taf) && taf > 0) targetArea = taf;
                using var ms = new MemoryStream();
                await file.CopyToAsync(ms);
                imageBytes = ms.ToArray();
            }
            else
            {
                // JSON: { imagePath (server path from /api/upload), prompt, steps, cfg, seed } (Web UI).
                var body = await System.Text.Json.JsonDocument.ParseAsync(req.Body);
                var root = body.RootElement;
                string imagePath = root.TryGetProperty("imagePath", out var ip) ? ip.GetString() : null;
                prompt = root.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
                steps = root.TryGetProperty("steps", out var st) && st.TryGetInt32(out int si) ? si : 28;
                cfg = root.TryGetProperty("cfg", out var cf) && cf.TryGetSingle(out float cv) ? cv : 4.0f;
                seed = root.TryGetProperty("seed", out var se) && se.TryGetInt64(out long sv) ? sv : 0;
                if (root.TryGetProperty("targetArea", out var ta) && ta.TryGetInt64(out long tav) && tav > 0)
                    targetArea = tav;
                // Constrain reads to the upload directory.
                string full = imagePath == null ? null : Path.GetFullPath(imagePath);
                string uploadRoot = Path.GetFullPath(_options.UploadDirectory);
                if (full == null || !full.StartsWith(uploadRoot, StringComparison.OrdinalIgnoreCase) || !File.Exists(full))
                    return Results.BadRequest(new { error = "imagePath must reference a previously uploaded file." });
                imageBytes = await File.ReadAllBytesAsync(full);
            }

            string outName = $"edit-{Guid.NewGuid():N}.png";
            string outPath = Path.Combine(_options.UploadDirectory, outName);

            logger.LogInformation(LogEventIds.UploadReceived,
                "Image edit: prompt='{Prompt}' steps={Steps} cfg={Cfg} bytes={Bytes}", prompt, steps, cfg, imageBytes.Length);

            var sw = System.Diagnostics.Stopwatch.StartNew();
            (int w, int h) = await Task.Run(() =>
            {
                // The model is not thread-safe; serialize edit requests.
                lock (_imageEditLock)
                {
                    var input = TensorSharp.Models.QwenImage.ImageIO.Decode(imageBytes);
                    var p = new TensorSharp.Models.QwenImage.QwenImageParams { Steps = steps, CfgScale = cfg, Seed = seed };
                    if (targetArea > 0) p.TargetArea = targetArea;
                    var output = editModel.EditImage(prompt, input, p);
                    TensorSharp.Models.QwenImage.ImageIO.SavePng(outPath, output);
                    return (output.Width, output.Height);
                }
            });
            sw.Stop();

            string url = BuildUploadUrl(outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Image edit done: {W}x{H} -> {Url} ({Sec:F1}s)", w, h, url, sw.Elapsed.TotalSeconds);
            return Results.Json(new { ok = true, url, width = w, height = h, elapsedSeconds = sw.Elapsed.TotalSeconds });
        }

        // A live denoising frame surfaced from the edit worker to the SSE writer: a progress tick
        // (Png == null) or a decoded preview image; the terminal frame carries the final result.
        private sealed class EditFrame
        {
            public int Step, Total, Width, Height;
            public byte[] Png;        // preview PNG bytes (null = progress-only tick)
            public bool Final;        // true on the terminal frame
            public string Url;        // final image URL (Final only)
            public double Seconds;    // total elapsed (Final only)
            public string Error;      // set if the edit threw
        }

        /// <summary>
        /// <c>POST /api/image-edit/stream</c> — same JSON body as <see cref="ImageEditAsync"/> but
        /// streams Server-Sent Events so the Web UI can show live denoising progress: a
        /// <c>{ preview, step, total, image? }</c> event per step (with a decoded snapshot on
        /// throttled steps) and a final <c>{ done, url, width, height, elapsedSeconds }</c>. This
        /// keeps the user informed that the (slow) diffusion is progressing instead of looking stuck.
        /// </summary>
        public async Task ImageEditStreamAsync(HttpContext ctx)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.ImageEdit");
            SseWriter.ApplyHeaders(ctx.Response);
            var ct = ctx.RequestAborted;

            if (_svc.Model is not TensorSharp.Models.QwenImage.QwenImageModel editModel)
            {
                await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = "The loaded model is not a Qwen-Image-Edit model." }, ct);
                return;
            }

            // Parse the Web UI JSON body (mirrors the JSON branch of ImageEditAsync).
            string prompt; int steps; float cfg; long seed; long targetArea = 0; byte[] imageBytes;
            try
            {
                var body = await JsonDocument.ParseAsync(ctx.Request.Body, cancellationToken: ct);
                var root = body.RootElement;
                string imagePath = root.TryGetProperty("imagePath", out var ip) ? ip.GetString() : null;
                prompt = root.TryGetProperty("prompt", out var pr) ? pr.GetString() ?? "" : "";
                steps = root.TryGetProperty("steps", out var st) && st.TryGetInt32(out int si) ? si : 28;
                cfg = root.TryGetProperty("cfg", out var cf) && cf.TryGetSingle(out float cv) ? cv : 4.0f;
                seed = root.TryGetProperty("seed", out var se) && se.TryGetInt64(out long sv) ? sv : 0;
                if (root.TryGetProperty("targetArea", out var ta) && ta.TryGetInt64(out long tav) && tav > 0)
                    targetArea = tav;
                string full = imagePath == null ? null : Path.GetFullPath(imagePath);
                string uploadRoot = Path.GetFullPath(_options.UploadDirectory);
                if (full == null || !full.StartsWith(uploadRoot, StringComparison.OrdinalIgnoreCase) || !File.Exists(full))
                {
                    await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = "imagePath must reference a previously uploaded file." }, ct);
                    return;
                }
                imageBytes = await File.ReadAllBytesAsync(full, ct);
            }
            catch (Exception ex)
            {
                await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = "Bad request: " + ex.Message }, ct);
                return;
            }

            string outName = $"edit-{Guid.NewGuid():N}.png";
            string outPath = Path.Combine(_options.UploadDirectory, outName);
            logger.LogInformation(LogEventIds.UploadReceived,
                "Image edit (stream): prompt='{Prompt}' steps={Steps} cfg={Cfg} bytes={Bytes}", prompt, steps, cfg, imageBytes.Length);

            // The edit worker pushes frames into this channel; the SSE loop drains it. The callback
            // never blocks on the network (unbounded TryWrite) so it can't stall the denoise.
            var channel = Channel.CreateUnbounded<EditFrame>(new UnboundedChannelOptions { SingleReader = true, SingleWriter = true });
            int previewCount = Math.Clamp(steps - 1, 0, 8);

            var editTask = Task.Run(() =>
            {
                var sw = Stopwatch.StartNew();
                try
                {
                    // The model is not thread-safe; serialize edit requests (shared with ImageEditAsync).
                    lock (_imageEditLock)
                    {
                        var input = TensorSharp.Models.QwenImage.ImageIO.Decode(imageBytes);
                        var p = new TensorSharp.Models.QwenImage.QwenImageParams
                        {
                            Steps = steps,
                            CfgScale = cfg,
                            Seed = seed,
                            PreviewCount = previewCount,
                            OnStep = (step, total, preview) =>
                            {
                                if (ct.IsCancellationRequested) throw new OperationCanceledException(ct);
                                byte[] png = preview != null ? TensorSharp.Models.QwenImage.ImageIO.EncodePng(preview) : null;
                                channel.Writer.TryWrite(new EditFrame
                                {
                                    Step = step, Total = total, Png = png,
                                    Width = preview?.Width ?? 0, Height = preview?.Height ?? 0,
                                });
                            },
                        };
                        if (targetArea > 0) p.TargetArea = targetArea;
                        var output = editModel.EditImage(prompt, input, p);
                        TensorSharp.Models.QwenImage.ImageIO.SavePng(outPath, output);
                        channel.Writer.TryWrite(new EditFrame
                        {
                            Final = true, Url = BuildUploadUrl(outName),
                            Width = output.Width, Height = output.Height, Seconds = sw.Elapsed.TotalSeconds,
                        });
                    }
                }
                catch (OperationCanceledException)
                {
                    channel.Writer.TryWrite(new EditFrame { Final = true, Error = "cancelled" });
                }
                catch (Exception ex)
                {
                    logger.LogError(LogEventIds.ChatFailed, ex, "Image edit (stream) failed");
                    channel.Writer.TryWrite(new EditFrame { Final = true, Error = ex.Message });
                }
                finally { channel.Writer.Complete(); }
            }, CancellationToken.None);

            try
            {
                await foreach (var f in channel.Reader.ReadAllAsync(ct))
                {
                    if (f.Final)
                    {
                        if (f.Error == "cancelled") break;
                        if (f.Error != null)
                            await SseWriter.WriteEventAsync(ctx.Response, new { done = true, error = f.Error }, ct);
                        else
                            await SseWriter.WriteEventAsync(ctx.Response,
                                new { done = true, url = f.Url, width = f.Width, height = f.Height, elapsedSeconds = f.Seconds }, ct);
                        logger.LogInformation(LogEventIds.UploadReceived,
                            "Image edit (stream) done: {W}x{H} -> {Url} ({Sec:F1}s)", f.Width, f.Height, f.Url, f.Seconds);
                    }
                    else
                    {
                        string image = f.Png != null ? "data:image/png;base64," + Convert.ToBase64String(f.Png) : null;
                        await SseWriter.WriteEventAsync(ctx.Response,
                            new { imageEdit = true, step = f.Step, total = f.Total, image, width = f.Width, height = f.Height }, ct);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                // Client disconnected; the worker observes ct via OnStep and unwinds.
            }

            // Drain the worker so its lock/VRAM is released before the next request (it finishes
            // promptly once cancellation is seen). Swallow — any error was already streamed.
            try { await editTask; } catch { /* already reported */ }
        }

        private static string BuildUploadUrl(string fileName)
        {
            return "/uploads/" + Uri.EscapeDataString(fileName);
        }

        private static string ClassifyExtension(string ext) => ext switch
        {
            ".png" or ".jpg" or ".jpeg" or ".gif" or ".webp" or ".bmp"
                or ".heic" or ".heif" => "image",
            ".mp4" or ".mov" or ".avi" or ".mkv" or ".webm" => "video",
            ".mp3" or ".wav" or ".ogg" or ".flac" or ".m4a" => "audio",
            ".txt" or ".csv" or ".json" or ".xml" or ".md" or ".log"
                or ".py" or ".js" or ".ts" or ".cs" or ".java" or ".cpp" or ".c" or ".h"
                or ".html" or ".css" or ".yaml" or ".yml" or ".toml" or ".ini" or ".cfg"
                or ".sh" or ".bat" or ".ps1" or ".rb" or ".go" or ".rs" or ".swift"
                or ".kt" or ".sql" or ".r" or ".m" or ".tex" or ".rtf" => "text",
            _ => "unknown",
        };

        // ---- Chat (SSE) -------------------------------------------------------

        public async Task ChatStreamAsync(HttpContext ctx)
        {
            var webUiLogger = _loggerFactory.CreateLogger("TensorSharp.Server.WebUI.Chat");
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

            string requestedModel = body.TryGetProperty("model", out var modelEl) ? modelEl.GetString() : null;
            string requestedBackend = body.TryGetProperty("backend", out var beEl) ? beEl.GetString() : null;
            bool newChat = body.TryGetProperty("newChat", out var ncProp) && ncProp.GetBoolean();
            string requestedSessionId = body.TryGetProperty("sessionId", out var sidEl) ? sidEl.GetString() : null;

            if (!WebUiChatPolicy.TryValidateChatRequest(requestedModel, requestedBackend, out string selectionError))
            {
                webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat rejected: {Reason} (requestedModel={Model}, requestedBackend={Backend})",
                    selectionError, requestedModel ?? "(none)", requestedBackend ?? "(none)");
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = selectionError });
                return;
            }

            ChatSession chatSession;
            if (!string.IsNullOrWhiteSpace(requestedSessionId))
            {
                chatSession = _sessions.GetSession(requestedSessionId);
                if (chatSession == null || chatSession.IsDisposed)
                {
                    webUiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                        "/api/chat rejected: session '{SessionId}' not found or disposed", requestedSessionId);
                    ctx.Response.StatusCode = 404;
                    await ctx.Response.WriteAsJsonAsync(new { error = $"Session '{requestedSessionId}' not found or has been disposed." });
                    return;
                }
            }
            else
            {
                chatSession = _sessions.DefaultSession;
            }

            if (newChat)
            {
                webUiLogger.LogInformation(LogEventIds.SessionReset,
                    "/api/chat newChat=true; resetting session {SessionId}", chatSession.Id);
                _svc.ResetSession(chatSession);
            }

            if (!_svc.IsLoaded)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = "No model loaded" });
                return;
            }

            var messagesEl = body.GetProperty("messages");
            int maxTokens = body.TryGetProperty("maxTokens", out var mt) ? mt.GetInt32() : _options.DefaultWebMaxTokens;

            var samplingConfig = SamplingConfigParser.ParseWebUi(body, _options.DefaultSamplingConfig);
            bool uiThink = body.TryGetProperty("think", out var uiThinkProp) && uiThinkProp.GetBoolean();
            List<ToolFunction> uiTools = null;
            if (body.TryGetProperty("tools", out var uiToolsEl) && uiToolsEl.ValueKind == JsonValueKind.Array)
                uiTools = ToolFunctionParser.ParseOllama(body);

            var messages = ChatMessageParser.ParseWebUi(messagesEl);

            SseWriter.ApplyHeaders(ctx.Response);

            using var ticket = _queue.Enqueue(ctx.RequestAborted);
            while (!ticket.IsReady)
            {
                await SseWriter.WriteEventAsync(ctx.Response,
                    WebUiSseEvents.QueueProgress(ticket.Position, _queue.PendingCount),
                    ctx.RequestAborted);
                await ticket.WaitAsync(TimeSpan.FromSeconds(1));
            }

            // DiffusionGemma streams a live "denoising preview" (whole-message replace per step)
            // rather than appended tokens, so it has its own SSE loop.
            if (_svc.IsDiffusionModel)
            {
                await ChatStreamDiffusionAsync(ctx, chatSession, messages, maxTokens, webUiLogger);
                return;
            }

            var sw = Stopwatch.StartNew();
            int tokenCount = 0;
            bool alwaysNeedsParsing = OutputParserFactory.IsAlwaysRequired(_svc.Architecture);
            bool useUiParser = uiThink || (uiTools != null && uiTools.Count > 0) || alwaysNeedsParsing;

            IOutputParser uiParser = null;
            if (useUiParser)
            {
                uiParser = OutputParserFactory.Create(_svc.Architecture);
                uiParser.Init(uiThink, uiTools);
            }

            bool aborted = false;
            string inferenceError = null;
            // Captured from the metrics tuple's done item so the final SSE event can
            // report how much of this turn's prompt was served from the prior turn's
            // KV cache. Defaults to zero in case the stream is aborted before
            // generation finishes.
            int turnPromptTokens = 0;
            int turnKvReusedTokens = 0;
            try
            {
                await foreach (var (piece, done, pt, _, kvReused, _, _, _)
                    in _svc.ChatStreamWithMetricsAsync(chatSession, messages, maxTokens, ctx.RequestAborted, samplingConfig,
                        uiTools, uiThink))
                {
                    if (done)
                    {
                        turnPromptTokens = pt;
                        turnKvReusedTokens = kvReused;
                        continue;
                    }

                    if (string.IsNullOrEmpty(piece))
                        continue;

                    tokenCount++;
                    if (uiParser != null)
                    {
                        var parsed = uiParser.Add(piece, false);
                        if (!string.IsNullOrEmpty(parsed.Thinking))
                            await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Thinking(parsed.Thinking), ctx.RequestAborted);
                        if (!string.IsNullOrEmpty(parsed.Content))
                            await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Token(parsed.Content), ctx.RequestAborted);
                        if (parsed.ToolCalls != null)
                            await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.ToolCalls(parsed.ToolCalls), ctx.RequestAborted);
                    }
                    else
                    {
                        await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Token(piece), ctx.RequestAborted);
                    }
                }
            }
            catch (OperationCanceledException)
            {
                aborted = true;
                var chatLogger = ctx.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger("TensorSharp.Server.WebUI.Chat");
                chatLogger.LogWarning(LogEventIds.ChatAborted,
                    "Web UI chat aborted by client (sessionId={SessionId}, partialTokens={PartialTokens})",
                    chatSession.Id, tokenCount);
            }
            catch (Exception ex)
            {
                var chatLogger = ctx.RequestServices.GetRequiredService<ILoggerFactory>().CreateLogger("TensorSharp.Server.WebUI.Chat");
                chatLogger.LogError(LogEventIds.ChatFailed, ex,
                    "Web UI chat failed (sessionId={SessionId}, partialTokens={PartialTokens})",
                    chatSession.Id, tokenCount);
                inferenceError = ex.Message;
            }

            await FinalizeChatStreamAsync(ctx, uiParser, aborted, inferenceError, chatSession, sw, tokenCount,
                turnPromptTokens, turnKvReusedTokens);
        }

        // ---- Chat (SSE) for DiffusionGemma: live denoising preview ------------

        private async Task ChatStreamDiffusionAsync(
            HttpContext ctx, ChatSession chatSession, List<ChatMessage> messages, int maxTokens, ILogger webUiLogger)
        {
            var sw = Stopwatch.StartNew();
            bool aborted = false;
            string inferenceError = null;
            int finalTokenCount = 0;
            int turnPromptTokens = 0;
            try
            {
                await foreach (var u in _svc.DiffusionChatStreamAsync(chatSession, messages, maxTokens, ctx.RequestAborted))
                {
                    if (u.Done)
                    {
                        finalTokenCount = u.EvalTokens;
                        turnPromptTokens = u.PromptTokens;
                        continue;
                    }
                    // Both intermediate previews and the final answer use whole-message replace.
                    await SseWriter.WriteEventAsync(ctx.Response,
                        WebUiSseEvents.Replace(u.Text, u.Step, u.TotalSteps, u.IsPreview), ctx.RequestAborted);
                }
            }
            catch (OperationCanceledException)
            {
                aborted = true;
                webUiLogger.LogWarning(LogEventIds.ChatAborted,
                    "Web UI diffusion chat aborted by client (sessionId={SessionId})", chatSession.Id);
            }
            catch (Exception ex)
            {
                webUiLogger.LogError(LogEventIds.ChatFailed, ex,
                    "Web UI diffusion chat failed (sessionId={SessionId})", chatSession.Id);
                inferenceError = ex.Message;
            }

            try
            {
                sw.Stop();
                double tokPerSec = finalTokenCount > 0 ? finalTokenCount / sw.Elapsed.TotalSeconds : 0;
                await SseWriter.WriteEventAsync(ctx.Response,
                    WebUiSseEvents.Done(finalTokenCount, sw.Elapsed.TotalSeconds, tokPerSec, aborted, inferenceError,
                        chatSession.Id, turnPromptTokens, 0));
            }
            catch (Exception)
            {
                // Best-effort final flush.
            }
        }

        private static async Task FinalizeChatStreamAsync(
            HttpContext ctx, IOutputParser uiParser, bool aborted, string inferenceError,
            ChatSession chatSession, Stopwatch sw, int tokenCount, int turnPromptTokens, int turnKvReusedTokens)
        {
            try
            {
                if (uiParser != null && !aborted)
                {
                    var finalParsed = uiParser.Add("", true);
                    if (!string.IsNullOrEmpty(finalParsed.Thinking))
                        await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Thinking(finalParsed.Thinking));
                    if (!string.IsNullOrEmpty(finalParsed.Content))
                        await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.Token(finalParsed.Content));
                    if (finalParsed.ToolCalls != null)
                        await SseWriter.WriteEventAsync(ctx.Response, WebUiSseEvents.ToolCalls(finalParsed.ToolCalls));
                }

                sw.Stop();
                double tokPerSec = tokenCount > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
                await SseWriter.WriteEventAsync(ctx.Response,
                    WebUiSseEvents.Done(tokenCount, sw.Elapsed.TotalSeconds, tokPerSec, aborted, inferenceError, chatSession.Id,
                        turnPromptTokens, turnKvReusedTokens));
            }
            catch (Exception)
            {
                // Best-effort final flush; if the client has already left we silently drop the trailing frames.
            }
        }
    }
}
