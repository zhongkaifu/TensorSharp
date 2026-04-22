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
            var status = _queue.GetStatus();
            return Results.Ok(new
            {
                busy = status.Busy,
                pending_requests = status.PendingRequests,
                total_processed = status.TotalProcessed,
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

            // Fire through the inference queue so we don't race with an in-flight request
            // that is still using this session's tensors.
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
                    mediaType,
                    fileName = file.FileName,
                    frames = frames.Select(f => Path.GetFileName(f)).ToList(),
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

            return Results.Json(new { ok = true, path = savePath, mediaType, fileName = file.FileName });
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
