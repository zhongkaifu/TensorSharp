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
using System.Text;
using System.Text.Json;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Hosting;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;
using Microsoft.Extensions.Logging;

var builder = WebApplication.CreateBuilder(args);

// Keep ASP.NET Core request logs quiet by default while still surfacing warnings and errors.
builder.Logging.AddFilter("Microsoft.AspNetCore", LogLevel.Warning);

builder.WebHost.ConfigureKestrel(options =>
{
    options.Limits.MaxRequestBodySize = 500 * 1024 * 1024; // 500 MB
});

builder.Services.Configure<Microsoft.AspNetCore.Http.Features.FormOptions>(options =>
{
    options.MultipartBodyLengthLimit = 500 * 1024 * 1024; // 500 MB
});

builder.Services.AddSingleton<ModelService>();
builder.Services.AddSingleton<InferenceQueue>();

string webRoot = builder.Environment.WebRootPath;
if (string.IsNullOrEmpty(webRoot) || !Directory.Exists(webRoot))
{
    string srcWwwRoot = Path.Combine(AppContext.BaseDirectory, "..", "wwwroot");
    if (Directory.Exists(srcWwwRoot))
        webRoot = Path.GetFullPath(srcWwwRoot);
    else
        webRoot = Path.Combine(AppContext.BaseDirectory, "wwwroot");

    builder.Environment.WebRootPath = webRoot;
    Directory.CreateDirectory(webRoot);
}

var app = builder.Build();
app.UseStaticFiles();

string uploadDir = Path.Combine(AppContext.BaseDirectory, "uploads");
Directory.CreateDirectory(uploadDir);
app.UseStaticFiles(new Microsoft.AspNetCore.Builder.StaticFileOptions
{
    FileProvider = new Microsoft.Extensions.FileProviders.PhysicalFileProvider(uploadDir),
    RequestPath = "/uploads"
});

int maxTextFileChars = 8000;
string maxTextEnv = Environment.GetEnvironmentVariable("MAX_TEXT_FILE_CHARS");
if (!string.IsNullOrEmpty(maxTextEnv) && int.TryParse(maxTextEnv, out int envMax) && envMax > 0)
    maxTextFileChars = envMax;

string modelDir = Environment.GetEnvironmentVariable("MODEL_DIR")
    ?? Path.Combine(AppContext.BaseDirectory, "models");
string configuredBackend = Environment.GetEnvironmentVariable("BACKEND")
    ?? (OperatingSystem.IsMacOS() ? "ggml_metal" : "ggml_cpu");
var supportedBackends = BackendCatalog.GetSupportedBackends().ToArray();
var supportedBackendValues = new HashSet<string>(supportedBackends.Select(backend => backend.Value), StringComparer.OrdinalIgnoreCase);
string defaultBackend = BackendCatalog.ResolveDefaultBackend(configuredBackend, supportedBackends);

if (supportedBackends.Length == 0)
{
    Console.WriteLine("Warning: no supported backends detected on this machine.");
}
else
{
    Console.WriteLine($"Supported backends: {string.Join(", ", supportedBackends.Select(backend => backend.Value))}");
}

if (!string.Equals(defaultBackend, BackendCatalog.Canonicalize(configuredBackend), StringComparison.OrdinalIgnoreCase) &&
    !string.IsNullOrWhiteSpace(defaultBackend))
{
    Console.WriteLine($"Requested default backend '{configuredBackend}' is unavailable. Falling back to '{defaultBackend}'.");
}

bool TryResolveSupportedBackend(string requestedBackend, out string resolvedBackend, out string error)
{
    resolvedBackend = string.IsNullOrWhiteSpace(requestedBackend)
        ? defaultBackend
        : BackendCatalog.Canonicalize(requestedBackend);

    if (string.IsNullOrWhiteSpace(resolvedBackend) || !supportedBackendValues.Contains(resolvedBackend))
    {
        error = supportedBackends.Length == 0
            ? "No supported backend is available on this machine."
            : $"Backend '{requestedBackend ?? defaultBackend}' is not supported on this machine.";
        return false;
    }

    error = null;
    return true;
}

// ============================================================
// Internal Web UI endpoints (original)
// ============================================================

app.MapGet("/api/queue/status", (InferenceQueue queue) =>
{
    var status = queue.GetStatus();
    return Results.Ok(new
    {
        busy = status.Busy,
        pending_requests = status.PendingRequests,
        total_processed = status.TotalProcessed
    });
});

app.MapGet("/api/models", (ModelService svc) =>
{
    var files = svc.ScanModels(modelDir);
    var mmProjFiles = svc.ScanMmProjModels(modelDir);
    return Results.Json(new
    {
        models = files,
        mmProjModels = mmProjFiles,
        loaded = svc.LoadedModelName,
        loadedMmProj = svc.LoadedMmProjName,
        loadedBackend = svc.LoadedBackend,
        defaultBackend,
        supportedBackends,
        architecture = svc.Architecture,
        modelDir
    });
});

app.MapPost("/api/models/load", async (HttpContext ctx, HttpRequest req, ModelService svc, InferenceQueue queue) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(req.Body);
    string modelName = body.GetProperty("model").GetString();
    string requestedBackend = body.TryGetProperty("backend", out var b) ? b.GetString() : null;
    string mmproj = body.TryGetProperty("mmproj", out var m) ? m.GetString() : null;

    if (!TryResolveSupportedBackend(requestedBackend, out string backend, out string backendError))
        return Results.BadRequest(new { ok = false, error = backendError });

    string modelPath = Path.Combine(modelDir, modelName);
    if (!File.Exists(modelPath))
        return Results.NotFound(new { error = $"Model not found: {modelName}" });

    // mmproj handling:
    //   null/absent  -> auto-detect (ModelService default)
    //   ""/"none"    -> explicitly no mmproj (pass empty string to skip auto-detect)
    //   "filename"   -> use that specific mmproj file
    string mmProjPath;
    if (mmproj == null)
    {
        mmProjPath = null; // auto-detect
    }
    else if (string.IsNullOrWhiteSpace(mmproj) || string.Equals(mmproj, "none", StringComparison.OrdinalIgnoreCase))
    {
        mmProjPath = ""; // explicit skip
    }
    else
    {
        mmProjPath = Path.Combine(modelDir, mmproj);
    }

    using var ticket = queue.Enqueue(ctx.RequestAborted);
    await ticket.WaitUntilReadyAsync();

    try
    {
        svc.LoadModel(modelPath, mmProjPath, backend);
        return Results.Json(new
        {
            ok = true,
            model = svc.LoadedModelName,
            loadedMmProj = svc.LoadedMmProjName,
            architecture = svc.Architecture
        });
    }
    catch (Exception ex)
    {
        return Results.Json(new { ok = false, error = ex.Message }, statusCode: 500);
    }
});

app.MapPost("/api/upload", async (HttpRequest req) =>
{
    if (!req.HasFormContentType)
        return Results.BadRequest(new { error = "Expected multipart form data" });

    var form = await req.ReadFormAsync();
    var file = form.Files.FirstOrDefault();
    if (file == null)
        return Results.BadRequest(new { error = "No file uploaded" });

    string ext = Path.GetExtension(file.FileName).ToLowerInvariant();
    string safeFileName = $"{Guid.NewGuid():N}{ext}";
    string savePath = Path.Combine(uploadDir, safeFileName);

    using (var stream = File.Create(savePath))
        await file.CopyToAsync(stream);

    string mediaType = ext switch
    {
        ".png" or ".jpg" or ".jpeg" or ".gif" or ".webp" or ".bmp" => "image",
        ".mp4" or ".mov" or ".avi" or ".mkv" or ".webm" => "video",
        ".mp3" or ".wav" or ".ogg" or ".flac" or ".m4a" => "audio",
        ".txt" or ".csv" or ".json" or ".xml" or ".md" or ".log"
            or ".py" or ".js" or ".ts" or ".cs" or ".java" or ".cpp" or ".c" or ".h"
            or ".html" or ".css" or ".yaml" or ".yml" or ".toml" or ".ini" or ".cfg"
            or ".sh" or ".bat" or ".ps1" or ".rb" or ".go" or ".rs" or ".swift"
            or ".kt" or ".sql" or ".r" or ".m" or ".tex" or ".rtf" => "text",
        _ => "unknown"
    };

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
            framePaths = frames
        });
    }

    if (mediaType == "text")
    {
        string textContent = await File.ReadAllTextAsync(savePath);
        bool truncated = false;
        if (textContent.Length > maxTextFileChars)
        {
            textContent = textContent.Substring(0, maxTextFileChars);
            truncated = true;
        }
        return Results.Json(new { ok = true, path = savePath, mediaType, fileName = file.FileName, textContent, truncated });
    }

    return Results.Json(new { ok = true, path = savePath, mediaType, fileName = file.FileName });
});

app.MapPost("/api/chat", async (HttpContext ctx, ModelService svc, InferenceQueue queue) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

    string requestedModel = body.TryGetProperty("model", out var modelEl) ? modelEl.GetString() : null;
    string requestedBackend = body.TryGetProperty("backend", out var beEl) ? beEl.GetString() : null;
    bool newChat = body.TryGetProperty("newChat", out var ncProp) && ncProp.GetBoolean();

    if (!WebUiChatPolicy.TryValidateChatRequest(requestedModel, requestedBackend, out string selectionError))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = selectionError });
        return;
    }

    if (newChat)
        svc.InvalidateKVCache();

    if (!svc.IsLoaded)
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "No model loaded" });
        return;
    }

    var messagesEl = body.GetProperty("messages");
    int maxTokens = body.TryGetProperty("maxTokens", out var mt) ? mt.GetInt32() : 200;

    var samplingConfig = ParseSamplingConfig(body);
    bool uiThink = body.TryGetProperty("think", out var uiThinkProp) && uiThinkProp.GetBoolean();
    List<ToolFunction> uiTools = null;
    if (body.TryGetProperty("tools", out var uiToolsEl) && uiToolsEl.ValueKind == JsonValueKind.Array)
        uiTools = ParseOllamaTools(body);

    var messages = new List<ChatMessage>();
    foreach (var msgEl in messagesEl.EnumerateArray())
    {
        var msg = new ChatMessage
        {
            Role = msgEl.GetProperty("role").GetString(),
            Content = msgEl.GetProperty("content").GetString()
        };

        if (msgEl.TryGetProperty("imagePaths", out var imgs) && imgs.GetArrayLength() > 0)
            msg.ImagePaths = imgs.EnumerateArray().Select(e => e.GetString()).ToList();

        if (msgEl.TryGetProperty("audioPaths", out var auds) && auds.GetArrayLength() > 0)
            msg.AudioPaths = auds.EnumerateArray().Select(e => e.GetString()).ToList();

        if (msgEl.TryGetProperty("isVideo", out var iv))
            msg.IsVideo = iv.GetBoolean();

        messages.Add(msg);
    }

    ctx.Response.ContentType = "text/event-stream";
    ctx.Response.Headers["Cache-Control"] = "no-cache";
    ctx.Response.Headers["Connection"] = "keep-alive";

    using var ticket = queue.Enqueue(ctx.RequestAborted);
    while (!ticket.IsReady)
    {
        string queueData = JsonSerializer.Serialize(new { queue_position = ticket.Position, queue_pending = queue.PendingCount });
        await ctx.Response.WriteAsync($"data: {queueData}\n\n", ctx.RequestAborted);
        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        await ticket.WaitAsync(TimeSpan.FromSeconds(1));
    }

    var writer = ctx.Response.BodyWriter;
    var sw = Stopwatch.StartNew();
    int tokenCount = 0;
    bool alwaysNeedsParsing = OutputParserFactory.IsAlwaysRequired(svc.Architecture);
    bool useUiParser = uiThink || (uiTools != null && uiTools.Count > 0) || alwaysNeedsParsing;

    IOutputParser uiParser = null;
    if (useUiParser)
    {
        uiParser = OutputParserFactory.Create(svc.Architecture);
        uiParser.Init(uiThink, uiTools);
    }

    bool aborted = false;
    string inferenceError = null;
    try
    {
        await foreach (var piece in svc.ChatStreamAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig,
            uiTools, uiThink))
        {
            tokenCount++;
            if (uiParser != null)
            {
                var parsed = uiParser.Add(piece, false);
                if (!string.IsNullOrEmpty(parsed.Thinking))
                {
                    string thinkData = JsonSerializer.Serialize(new { thinking = parsed.Thinking });
                    await ctx.Response.WriteAsync($"data: {thinkData}\n\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }
                if (!string.IsNullOrEmpty(parsed.Content))
                {
                    string contentData = JsonSerializer.Serialize(new { token = parsed.Content });
                    await ctx.Response.WriteAsync($"data: {contentData}\n\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }
                if (parsed.ToolCalls != null)
                {
                    string tcData = JsonSerializer.Serialize(new
                    {
                        tool_calls = parsed.ToolCalls.ConvertAll(tc => new
                        {
                            name = tc.Name,
                            arguments = tc.Arguments
                        })
                    });
                    await ctx.Response.WriteAsync($"data: {tcData}\n\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }
            }
            else
            {
                string data = JsonSerializer.Serialize(new { token = piece });
                await ctx.Response.WriteAsync($"data: {data}\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
        }
    }
    catch (OperationCanceledException) { aborted = true; }
    catch (Exception ex)
    {
        Console.Error.WriteLine($"[Chat error] {ex.Message}");
        inferenceError = ex.Message;
    }

    try
    {
        if (uiParser != null && !aborted)
        {
            var finalParsed = uiParser.Add("", true);
            if (!string.IsNullOrEmpty(finalParsed.Thinking))
            {
                string thinkData = JsonSerializer.Serialize(new { thinking = finalParsed.Thinking });
                await ctx.Response.WriteAsync($"data: {thinkData}\n\n");
                await ctx.Response.Body.FlushAsync();
            }
            if (!string.IsNullOrEmpty(finalParsed.Content))
            {
                string contentData = JsonSerializer.Serialize(new { token = finalParsed.Content });
                await ctx.Response.WriteAsync($"data: {contentData}\n\n");
                await ctx.Response.Body.FlushAsync();
            }
            if (finalParsed.ToolCalls != null)
            {
                string tcData = JsonSerializer.Serialize(new
                {
                    tool_calls = finalParsed.ToolCalls.ConvertAll(tc => new
                    {
                        name = tc.Name,
                        arguments = tc.Arguments
                    })
                });
                await ctx.Response.WriteAsync($"data: {tcData}\n\n");
                await ctx.Response.Body.FlushAsync();
            }
        }

        sw.Stop();
        double tokPerSec = tokenCount > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
        string done = JsonSerializer.Serialize(new { done = true, tokenCount, elapsed = sw.Elapsed.TotalSeconds, tokPerSec,
            aborted, error = inferenceError });
        await ctx.Response.WriteAsync($"data: {done}\n\n");
        await ctx.Response.Body.FlushAsync();
    }
    catch (Exception) { }
});

// ============================================================
// Ollama-compatible API endpoints
// ============================================================

app.MapGet("/", () => Results.Ok("TensorSharp.Server is running"));
app.MapGet("/api/version", () => Results.Json(new { version = "0.1.0" }));

app.MapGet("/api/tags", (ModelService svc) =>
{
    var files = svc.ScanModels(modelDir);
    var models = files.Select(f =>
    {
        var fi = new FileInfo(Path.Combine(modelDir, f));
        return new Dictionary<string, object>
        {
            ["name"] = Path.GetFileNameWithoutExtension(f),
            ["model"] = f,
            ["size"] = fi.Exists ? fi.Length : 0,
            ["modified_at"] = fi.Exists ? fi.LastWriteTimeUtc.ToString("o") : ""
        };
    }).ToList();
    return Results.Json(new { models });
});

app.MapPost("/api/show", async (HttpContext ctx, ModelService svc) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);
    if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "model is required" });
        return;
    }

    string modelName = modelProp.GetString();
    string modelPath = ResolveModelPath(modelName, modelDir);
    if (modelPath == null)
    {
        ctx.Response.StatusCode = 404;
        await ctx.Response.WriteAsJsonAsync(new { error = $"model '{modelName}' not found" });
        return;
    }

    var fi = new FileInfo(modelPath);
    await ctx.Response.WriteAsJsonAsync(new
    {
        modelfile = "",
        parameters = "",
        template = "",
        details = new
        {
            format = "gguf",
            family = svc.IsLoaded && svc.LoadedModelName == Path.GetFileName(modelPath) ? svc.Architecture : "",
        },
        model_info = new
        {
            file = Path.GetFileName(modelPath),
            size = fi.Length,
        }
    });
});

app.MapPost("/api/generate", async (HttpContext ctx, ModelService svc, InferenceQueue queue) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

    if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "model is required" });
        return;
    }

    string modelName = modelProp.GetString();
    string prompt = body.TryGetProperty("prompt", out var pp) ? pp.GetString() ?? "" : "";
    bool stream = true;
    if (body.TryGetProperty("stream", out var streamProp)) stream = streamProp.GetBoolean();
    int maxTokens = 200;
    var samplingConfig = ParseOllamaOptions(body);
    if (body.TryGetProperty("options", out var opts) && opts.TryGetProperty("num_predict", out var np))
        maxTokens = np.GetInt32();

    var imagePaths = DecodeBase64Images(body, uploadDir);

    using var ticket = queue.Enqueue(ctx.RequestAborted);

    if (stream)
    {
        ctx.Response.ContentType = "application/x-ndjson";
        ctx.Response.Headers["Cache-Control"] = "no-cache";

        while (!ticket.IsReady)
        {
            var queueResp = new { model = modelName, created_at = DateTime.UtcNow.ToString("o"),
                response = "", done = false, queue_position = ticket.Position, queue_pending = queue.PendingCount };
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(queueResp) + "\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            await ticket.WaitAsync(TimeSpan.FromSeconds(1));
        }

        if (!svc.EnsureModelLoaded(modelName, modelDir, defaultBackend))
        {
            var errResp = new { model = modelName, created_at = DateTime.UtcNow.ToString("o"),
                response = "", done = true, done_reason = "error", error = $"model '{modelName}' not found" };
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(errResp) + "\n", ctx.RequestAborted);
            return;
        }

        await foreach (var (piece, done, promptTokens, evalTokens, totalNs, promptNs, evalNs)
            in svc.GenerateStreamAsync(prompt, imagePaths, maxTokens, ctx.RequestAborted, samplingConfig))
        {
            object resp;
            if (!done)
            {
                resp = new
                {
                    model = svc.LoadedModelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    response = piece,
                    done = false
                };
            }
            else
            {
                resp = new
                {
                    model = svc.LoadedModelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    response = "",
                    done = true,
                    done_reason = "stop",
                    total_duration = totalNs,
                    prompt_eval_count = promptTokens,
                    prompt_eval_duration = promptNs,
                    eval_count = evalTokens,
                    eval_duration = evalNs
                };
            }

            await ctx.Response.WriteAsync(JsonSerializer.Serialize(resp) + "\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        }
    }
    else
    {
        await ticket.WaitUntilReadyAsync();

        if (!svc.EnsureModelLoaded(modelName, modelDir, defaultBackend))
        {
            ctx.Response.StatusCode = 404;
            await ctx.Response.WriteAsJsonAsync(new { error = $"model '{modelName}' not found" });
            return;
        }

        var sb = new StringBuilder();
        int promptTokens = 0, evalTokens = 0;
        long totalNs = 0, promptNs = 0, evalNs = 0;

        await foreach (var (piece, done, pt, et, tn, pn, en)
            in svc.GenerateStreamAsync(prompt, imagePaths, maxTokens, ctx.RequestAborted, samplingConfig))
        {
            if (!done)
                sb.Append(piece);
            else
            {
                promptTokens = pt; evalTokens = et;
                totalNs = tn; promptNs = pn; evalNs = en;
            }
        }

        await ctx.Response.WriteAsJsonAsync(new
        {
            model = svc.LoadedModelName,
            created_at = DateTime.UtcNow.ToString("o"),
            response = sb.ToString(),
            done = true,
            done_reason = "stop",
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs
        });
    }
});

app.MapPost("/api/chat/ollama", async (HttpContext ctx, ModelService svc, InferenceQueue queue) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

    if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "model is required" });
        return;
    }

    string modelName = modelProp.GetString();

    if (!body.TryGetProperty("messages", out var messagesEl) || messagesEl.ValueKind != JsonValueKind.Array)
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "messages is required" });
        return;
    }

    bool stream = true;
    if (body.TryGetProperty("stream", out var streamProp)) stream = streamProp.GetBoolean();
    int maxTokens = 200;
    var samplingConfig = ParseOllamaOptions(body);
    if (body.TryGetProperty("options", out var opts) && opts.TryGetProperty("num_predict", out var np))
        maxTokens = np.GetInt32();

    var messages = ParseOllamaMessages(messagesEl, uploadDir);
    var ollamaTools = ParseOllamaTools(body);
    bool ollamaThink = body.TryGetProperty("think", out var thinkProp) && thinkProp.GetBoolean();

    using var ticket = queue.Enqueue(ctx.RequestAborted);

    if (stream)
    {
        ctx.Response.ContentType = "application/x-ndjson";
        ctx.Response.Headers["Cache-Control"] = "no-cache";

        while (!ticket.IsReady)
        {
            var queueResp = new { model = modelName, created_at = DateTime.UtcNow.ToString("o"),
                message = new { role = "assistant", content = "" }, done = false,
                queue_position = ticket.Position, queue_pending = queue.PendingCount };
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(queueResp) + "\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            await ticket.WaitAsync(TimeSpan.FromSeconds(1));
        }

        if (!svc.EnsureModelLoaded(modelName, modelDir, defaultBackend))
        {
            var errResp = new { model = modelName, created_at = DateTime.UtcNow.ToString("o"),
                message = new { role = "assistant", content = "" }, done = true,
                done_reason = "error", error = $"model '{modelName}' not found" };
            await ctx.Response.WriteAsync(JsonSerializer.Serialize(errResp) + "\n", ctx.RequestAborted);
            return;
        }

        var parser = OutputParserFactory.Create(svc.Architecture);
        parser.Init(ollamaThink, ollamaTools);
        bool useParser = ollamaThink || (ollamaTools != null && ollamaTools.Count > 0) || parser.AlwaysRequired;
        var jsonOpts = new JsonSerializerOptions { DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull };
        List<ToolCall> collectedToolCalls = null;

        await foreach (var (piece, done, promptTokens, evalTokens, totalNs, promptNs, evalNs)
            in svc.ChatStreamWithMetricsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig,
                ollamaTools, ollamaThink))
        {
            object resp;
            if (!done)
            {
                if (useParser)
                {
                    var parsed = parser.Add(piece, false);
                    if (parsed.ToolCalls != null)
                        collectedToolCalls = parsed.ToolCalls;
                    string thinkChunk = !string.IsNullOrEmpty(parsed.Thinking) ? parsed.Thinking : null;
                    string contentChunk = parsed.Content ?? "";
                    if (thinkChunk == null && contentChunk.Length == 0)
                        continue;
                    resp = new
                    {
                        model = svc.LoadedModelName,
                        created_at = DateTime.UtcNow.ToString("o"),
                        message = new { role = "assistant", content = contentChunk, thinking = thinkChunk },
                        done = false
                    };
                }
                else
                {
                    resp = new
                    {
                        model = svc.LoadedModelName,
                        created_at = DateTime.UtcNow.ToString("o"),
                        message = new { role = "assistant", content = piece },
                        done = false
                    };
                }
            }
            else
            {
                if (useParser)
                {
                    var finalParsed = parser.Add("", true);
                    if (finalParsed.ToolCalls != null)
                        collectedToolCalls = finalParsed.ToolCalls;
                    string thinkChunk = !string.IsNullOrEmpty(finalParsed.Thinking) ? finalParsed.Thinking : null;
                    string contentChunk = finalParsed.Content ?? "";

                    if (thinkChunk != null || contentChunk.Length > 0)
                    {
                        var flushResp = new
                        {
                            model = svc.LoadedModelName,
                            created_at = DateTime.UtcNow.ToString("o"),
                            message = new { role = "assistant", content = contentChunk, thinking = thinkChunk },
                            done = false
                        };
                        await ctx.Response.WriteAsync(JsonSerializer.Serialize(flushResp, jsonOpts) + "\n", ctx.RequestAborted);
                        await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                    }

                    var toolCallsJson = collectedToolCalls?.ConvertAll(tc => new
                    {
                        function = new { name = tc.Name, arguments = tc.Arguments }
                    });

                    resp = new
                    {
                        model = svc.LoadedModelName,
                        created_at = DateTime.UtcNow.ToString("o"),
                        message = new
                        {
                            role = "assistant",
                            content = "",
                            tool_calls = toolCallsJson
                        },
                        done = true,
                        done_reason = collectedToolCalls != null ? "tool_calls" : "stop",
                        total_duration = totalNs,
                        prompt_eval_count = promptTokens,
                        prompt_eval_duration = promptNs,
                        eval_count = evalTokens,
                        eval_duration = evalNs
                    };
                }
                else
                {
                    resp = new
                    {
                        model = svc.LoadedModelName,
                        created_at = DateTime.UtcNow.ToString("o"),
                        message = new { role = "assistant", content = "" },
                        done = true,
                        done_reason = "stop",
                        total_duration = totalNs,
                        prompt_eval_count = promptTokens,
                        prompt_eval_duration = promptNs,
                        eval_count = evalTokens,
                        eval_duration = evalNs
                    };
                }
            }

            await ctx.Response.WriteAsync(JsonSerializer.Serialize(resp, jsonOpts) + "\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        }
    }
    else
    {
        await ticket.WaitUntilReadyAsync();

        if (!svc.EnsureModelLoaded(modelName, modelDir, defaultBackend))
        {
            ctx.Response.StatusCode = 404;
            await ctx.Response.WriteAsJsonAsync(new { error = $"model '{modelName}' not found" });
            return;
        }

        var sb = new StringBuilder();
        int promptTokens = 0, evalTokens = 0;
        long totalNs = 0, promptNs = 0, evalNs = 0;

        await foreach (var (piece, done, pt, et, tn, pn, en)
            in svc.ChatStreamWithMetricsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig,
                ollamaTools, ollamaThink))
        {
            if (!done)
                sb.Append(piece);
            else
            {
                promptTokens = pt; evalTokens = et;
                totalNs = tn; promptNs = pn; evalNs = en;
            }
        }

        string rawOutput = sb.ToString();
        var parser2 = OutputParserFactory.Create(svc.Architecture);
        parser2.Init(ollamaThink, ollamaTools);
        bool useParser2 = ollamaThink || (ollamaTools != null && ollamaTools.Count > 0) || parser2.AlwaysRequired;
        object finalMessage;
        string doneReason = "stop";

        if (useParser2)
        {
            var parsed = parser2.Add(rawOutput, true);
            var toolCallsJson = parsed.ToolCalls?.ConvertAll(tc => new
            {
                function = new { name = tc.Name, arguments = tc.Arguments }
            });
            string thinkingOut = ollamaThink && !string.IsNullOrEmpty(parsed.Thinking) ? parsed.Thinking : null;
            finalMessage = new
            {
                role = "assistant",
                content = parsed.Content ?? "",
                thinking = thinkingOut,
                tool_calls = toolCallsJson
            };
            if (parsed.ToolCalls != null && parsed.ToolCalls.Count > 0)
                doneReason = "tool_calls";
        }
        else
        {
            finalMessage = new { role = "assistant", content = rawOutput };
        }

        await ctx.Response.WriteAsync(JsonSerializer.Serialize(new
        {
            model = svc.LoadedModelName,
            created_at = DateTime.UtcNow.ToString("o"),
            message = finalMessage,
            done = true,
            done_reason = doneReason,
            total_duration = totalNs,
            prompt_eval_count = promptTokens,
            prompt_eval_duration = promptNs,
            eval_count = evalTokens,
            eval_duration = evalNs
        }, new JsonSerializerOptions { DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull }));
    }
});

// ============================================================
// OpenAI-compatible API endpoint
// ============================================================

app.MapPost("/v1/chat/completions", async (HttpContext ctx, ModelService svc, InferenceQueue queue) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

    if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = new { message = "model is required", type = "invalid_request_error" } });
        return;
    }

    string modelName = modelProp.GetString();

    if (!body.TryGetProperty("messages", out var messagesEl) || messagesEl.ValueKind != JsonValueKind.Array)
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = new { message = "messages is required", type = "invalid_request_error" } });
        return;
    }

    bool stream = body.TryGetProperty("stream", out var streamProp) && streamProp.GetBoolean();
    int maxTokens = body.TryGetProperty("max_tokens", out var mtProp) ? mtProp.GetInt32() : 200;
    var samplingConfig = ParseOpenAIOptions(body);
    var messages = ParseOpenAIMessages(messagesEl, uploadDir);
    string requestId = $"chatcmpl-{Guid.NewGuid():N}".Substring(0, 30);

    var openaiTools = ParseOpenAITools(body);
    bool openaiThink = body.TryGetProperty("think", out var oaiThinkProp) && oaiThinkProp.GetBoolean();
    if (!OpenAIResponseFormatParser.TryParse(body, out StructuredOutputFormat responseFormat, out string responseFormatError))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = new { message = responseFormatError, type = "invalid_request_error" } });
        return;
    }

    if (responseFormat != null)
    {
        if (openaiThink)
        {
            ctx.Response.StatusCode = 400;
            await ctx.Response.WriteAsJsonAsync(new { error = new { message = "response_format cannot be combined with think=true", type = "invalid_request_error" } });
            return;
        }

        if (openaiTools != null && openaiTools.Count > 0)
        {
            ctx.Response.StatusCode = 400;
            await ctx.Response.WriteAsJsonAsync(new { error = new { message = "response_format cannot be combined with tools", type = "invalid_request_error" } });
            return;
        }

        var schemaValidation = StructuredOutputValidator.ValidateSchema(responseFormat);
        if (!schemaValidation.IsValid)
        {
            ctx.Response.StatusCode = 400;
            await ctx.Response.WriteAsJsonAsync(new
            {
                error = new
                {
                    message = schemaValidation.ErrorMessage,
                    type = "invalid_request_error",
                    details = schemaValidation.Errors
                }
            });
            return;
        }
    }

    var inferenceMessages = StructuredOutputPrompt.Apply(messages, responseFormat);

    using var ticket = queue.Enqueue(ctx.RequestAborted);

    if (stream)
    {
        bool structuredStream = responseFormat != null;
        if (!structuredStream)
        {
            ctx.Response.ContentType = "text/event-stream";
            ctx.Response.Headers["Cache-Control"] = "no-cache";

            while (!ticket.IsReady)
            {
                var queueResp = new
                {
                    id = requestId,
                    @object = "chat.completion.chunk",
                    model = modelName,
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    choices = new[] { new { index = 0, delta = new { role = "assistant", content = "" }, finish_reason = (string)null } },
                    queue_position = ticket.Position,
                    queue_pending = queue.PendingCount
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(queueResp)}\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                await ticket.WaitAsync(TimeSpan.FromSeconds(1));
            }
        }
        else
        {
            await ticket.WaitUntilReadyAsync();
        }

        if (!svc.EnsureModelLoaded(modelName, modelDir, defaultBackend))
        {
            if (structuredStream)
            {
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = $"model '{modelName}' not found", type = "invalid_request_error" } });
            }
            else
            {
                var errChunk = new
                {
                    id = requestId,
                    @object = "chat.completion.chunk",
                    model = modelName,
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    choices = new[] { new { index = 0, delta = new { content = $"Error: model '{modelName}' not found" }, finish_reason = "stop" } }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(errChunk)}\n\ndata: [DONE]\n\n", ctx.RequestAborted);
            }
            return;
        }

        bool useOaiStreamParser = openaiThink || (openaiTools != null && openaiTools.Count > 0) || OutputParserFactory.IsAlwaysRequired(svc.Architecture);
        bool bufferForStructured = structuredStream;
        var oaiStreamSb = bufferForStructured ? new StringBuilder() : null;
        IOutputParser oaiParser = null;
        List<ToolCall> oaiCollectedToolCalls = null;
        if (useOaiStreamParser && !bufferForStructured)
        {
            oaiParser = OutputParserFactory.Create(svc.Architecture);
            oaiParser.Init(openaiThink, openaiTools);
        }

        await foreach (var (piece, done, promptTokens, evalTokens, totalNs, promptNs, evalNs)
            in svc.ChatStreamWithMetricsAsync(inferenceMessages, maxTokens, ctx.RequestAborted, samplingConfig,
                openaiTools, openaiThink))
        {
            if (!done)
            {
                if (bufferForStructured)
                {
                    oaiStreamSb.Append(piece);
                    continue;
                }

                if (oaiParser != null)
                {
                    var parsed = oaiParser.Add(piece, false);
                    if (parsed.ToolCalls != null)
                        oaiCollectedToolCalls = parsed.ToolCalls;
                    string emitContent = parsed.Content ?? "";
                    if (emitContent.Length == 0 && string.IsNullOrEmpty(parsed.Thinking))
                        continue;
                    string chunkContent = emitContent.Length > 0 ? emitContent : null;
                    var chunk = new
                    {
                        id = requestId,
                        @object = "chat.completion.chunk",
                        created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        model = svc.LoadedModelName,
                        choices = new[]
                        {
                            new
                            {
                                index = 0,
                                delta = new { role = (string)null, content = chunkContent },
                                finish_reason = (string)null
                            }
                        }
                    };
                    await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(chunk)}\n\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                    continue;
                }

                var rawChunk = new
                {
                    id = requestId,
                    @object = "chat.completion.chunk",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    model = svc.LoadedModelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { role = (string)null, content = piece },
                            finish_reason = (string)null
                        }
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(rawChunk)}\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                continue;
            }

            if (bufferForStructured)
            {
                string rawContent = oaiStreamSb.ToString();

                if (useOaiStreamParser)
                {
                    var structParser = OutputParserFactory.Create(svc.Architecture);
                    structParser.Init(openaiThink, openaiTools);
                    var parsed = structParser.Add(rawContent, true);
                    rawContent = parsed.Content ?? "";
                }

                var normalized = StructuredOutputValidator.NormalizeOutput(rawContent, responseFormat);
                if (!normalized.IsValid)
                {
                    ctx.Response.StatusCode = 422;
                    await ctx.Response.WriteAsJsonAsync(new
                    {
                        error = new
                        {
                            message = normalized.ErrorMessage,
                            type = "invalid_response_error",
                            details = normalized.Errors
                        }
                    });
                    return;
                }

                ctx.Response.ContentType = "text/event-stream";
                ctx.Response.Headers["Cache-Control"] = "no-cache";

                var contentChunk = new
                {
                    id = requestId,
                    @object = "chat.completion.chunk",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    model = svc.LoadedModelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { role = "assistant", content = normalized.NormalizedContent },
                            finish_reason = (string)null
                        }
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(contentChunk)}\n\n", ctx.RequestAborted);

                var endChunk = new
                {
                    id = requestId,
                    @object = "chat.completion.chunk",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    model = svc.LoadedModelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { role = (string)null, content = (string)null },
                            finish_reason = "stop"
                        }
                    },
                    usage = new
                    {
                        prompt_tokens = promptTokens,
                        completion_tokens = evalTokens,
                        total_tokens = promptTokens + evalTokens
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(endChunk)}\n\n", ctx.RequestAborted);
                await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                continue;
            }

            if (oaiParser != null)
            {
                var finalParsed = oaiParser.Add("", true);
                if (finalParsed.ToolCalls != null)
                    oaiCollectedToolCalls = finalParsed.ToolCalls;

                if (!string.IsNullOrEmpty(finalParsed.Content))
                {
                    var flushChunk = new
                    {
                        id = requestId,
                        @object = "chat.completion.chunk",
                        created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                        model = svc.LoadedModelName,
                        choices = new[]
                        {
                            new
                            {
                                index = 0,
                                delta = new { role = (string)null, content = finalParsed.Content },
                                finish_reason = (string)null
                            }
                        }
                    };
                    await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(flushChunk)}\n\n", ctx.RequestAborted);
                    await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
                }

                string finReason = (oaiCollectedToolCalls != null && oaiCollectedToolCalls.Count > 0) ? "tool_calls" : "stop";
                var endChunk = new
                {
                    id = requestId,
                    @object = "chat.completion.chunk",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    model = svc.LoadedModelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { role = (string)null, content = (string)null },
                            finish_reason = finReason
                        }
                    },
                    usage = new
                    {
                        prompt_tokens = promptTokens,
                        completion_tokens = evalTokens,
                        total_tokens = promptTokens + evalTokens
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(endChunk)}\n\n", ctx.RequestAborted);
            }
            else
            {
                var chunk = new
                {
                    id = requestId,
                    @object = "chat.completion.chunk",
                    created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
                    model = svc.LoadedModelName,
                    choices = new[]
                    {
                        new
                        {
                            index = 0,
                            delta = new { role = (string)null, content = (string)null },
                            finish_reason = "stop"
                        }
                    },
                    usage = new
                    {
                        prompt_tokens = promptTokens,
                        completion_tokens = evalTokens,
                        total_tokens = promptTokens + evalTokens
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(chunk)}\n\n", ctx.RequestAborted);
            }

            await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        }
    }
    else
    {
        await ticket.WaitUntilReadyAsync();

        if (!svc.EnsureModelLoaded(modelName, modelDir, defaultBackend))
        {
            ctx.Response.StatusCode = 404;
            await ctx.Response.WriteAsJsonAsync(new { error = new { message = $"model '{modelName}' not found", type = "invalid_request_error" } });
            return;
        }

        var sb = new StringBuilder();
        int promptTokens = 0, evalTokens = 0;

        await foreach (var (piece, done, pt, et, tn, pn, en)
            in svc.ChatStreamWithMetricsAsync(inferenceMessages, maxTokens, ctx.RequestAborted, samplingConfig,
                openaiTools, openaiThink))
        {
            if (!done)
                sb.Append(piece);
            else { promptTokens = pt; evalTokens = et; }
        }

        string rawOutput = sb.ToString();
        bool useOaiParser = openaiThink || (openaiTools != null && openaiTools.Count > 0) || OutputParserFactory.IsAlwaysRequired(svc.Architecture);
        object responseMessage;
        string finishReason = "stop";

        if (responseFormat != null)
        {
            var normalized = StructuredOutputValidator.NormalizeOutput(rawOutput, responseFormat);
            if (!normalized.IsValid)
            {
                ctx.Response.StatusCode = 422;
                await ctx.Response.WriteAsJsonAsync(new
                {
                    error = new
                    {
                        message = normalized.ErrorMessage,
                        type = "invalid_response_error",
                        details = normalized.Errors
                    }
                });
                return;
            }

            responseMessage = new { role = "assistant", content = normalized.NormalizedContent };
        }
        else if (useOaiParser)
        {
            var parser = OutputParserFactory.Create(svc.Architecture);
            parser.Init(openaiThink, openaiTools);
            var parsed = parser.Add(rawOutput, true);

            var toolCallsList = parsed.ToolCalls?.Select((tc, idx) => new
            {
                id = $"call_{Guid.NewGuid():N}".Substring(0, 24),
                type = "function",
                function = new { name = tc.Name, arguments = JsonSerializer.Serialize(tc.Arguments) }
            }).ToArray();

            string thinkingOut = openaiThink && !string.IsNullOrEmpty(parsed.Thinking) ? parsed.Thinking : null;
            responseMessage = new
            {
                role = "assistant",
                content = parsed.Content ?? "",
                thinking = thinkingOut,
                tool_calls = toolCallsList
            };
            if (parsed.ToolCalls != null && parsed.ToolCalls.Count > 0)
                finishReason = "tool_calls";
        }
        else
        {
            responseMessage = new { role = "assistant", content = rawOutput };
        }

        await ctx.Response.WriteAsync(JsonSerializer.Serialize(new
        {
            id = requestId,
            @object = "chat.completion",
            created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            model = svc.LoadedModelName,
            choices = new object[]
            {
                new
                {
                    index = 0,
                    message = responseMessage,
                    finish_reason = finishReason
                }
            },
            usage = new
            {
                prompt_tokens = promptTokens,
                completion_tokens = evalTokens,
                total_tokens = promptTokens + evalTokens
            }
        }, new JsonSerializerOptions { DefaultIgnoreCondition = System.Text.Json.Serialization.JsonIgnoreCondition.WhenWritingNull }));
    }

});

app.MapGet("/v1/models", (ModelService svc) =>
{
    var files = svc.ScanModels(modelDir);
    var data = files.Select(f => new Dictionary<string, object>
    {
        ["id"] = Path.GetFileNameWithoutExtension(f),
        ["object"] = "model",
        ["owned_by"] = "local"
    }).ToList();
    return Results.Json(new { @object = "list", data });
});

app.MapFallback(async ctx =>
{
    string root = app.Environment.WebRootPath;
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

// ============================================================
// Helper functions
// ============================================================

static SamplingConfig ParseSamplingConfig(JsonElement body)
{
    var cfg = new SamplingConfig();
    if (body.TryGetProperty("temperature", out var temp))
        cfg.Temperature = temp.GetSingle();
    if (body.TryGetProperty("top_k", out var tk))
        cfg.TopK = tk.GetInt32();
    if (body.TryGetProperty("topK", out var tk2))
        cfg.TopK = tk2.GetInt32();
    if (body.TryGetProperty("top_p", out var tp))
        cfg.TopP = tp.GetSingle();
    if (body.TryGetProperty("topP", out var tp2))
        cfg.TopP = tp2.GetSingle();
    if (body.TryGetProperty("min_p", out var mp))
        cfg.MinP = mp.GetSingle();
    if (body.TryGetProperty("minP", out var mp2))
        cfg.MinP = mp2.GetSingle();
    if (body.TryGetProperty("repetition_penalty", out var rp))
        cfg.RepetitionPenalty = rp.GetSingle();
    if (body.TryGetProperty("repetitionPenalty", out var rp2))
        cfg.RepetitionPenalty = rp2.GetSingle();
    if (body.TryGetProperty("presence_penalty", out var pp))
        cfg.PresencePenalty = pp.GetSingle();
    if (body.TryGetProperty("presencePenalty", out var pp2))
        cfg.PresencePenalty = pp2.GetSingle();
    if (body.TryGetProperty("frequency_penalty", out var fp))
        cfg.FrequencyPenalty = fp.GetSingle();
    if (body.TryGetProperty("frequencyPenalty", out var fp2))
        cfg.FrequencyPenalty = fp2.GetSingle();
    if (body.TryGetProperty("seed", out var sd))
        cfg.Seed = sd.GetInt32();
    if (body.TryGetProperty("stop", out var stopEl) && stopEl.ValueKind == JsonValueKind.Array)
    {
        cfg.StopSequences = new List<string>();
        foreach (var s in stopEl.EnumerateArray())
            if (s.GetString() is string sv)
                cfg.StopSequences.Add(sv);
    }
    return cfg;
}

static SamplingConfig ParseOllamaOptions(JsonElement body)
{
    var cfg = new SamplingConfig();
    if (body.TryGetProperty("options", out var opts))
    {
        if (opts.TryGetProperty("temperature", out var temp)) cfg.Temperature = temp.GetSingle();
        if (opts.TryGetProperty("top_k", out var tk)) cfg.TopK = tk.GetInt32();
        if (opts.TryGetProperty("top_p", out var tp)) cfg.TopP = tp.GetSingle();
        if (opts.TryGetProperty("min_p", out var mp)) cfg.MinP = mp.GetSingle();
        if (opts.TryGetProperty("repeat_penalty", out var rp)) cfg.RepetitionPenalty = rp.GetSingle();
        if (opts.TryGetProperty("presence_penalty", out var pp)) cfg.PresencePenalty = pp.GetSingle();
        if (opts.TryGetProperty("frequency_penalty", out var fp)) cfg.FrequencyPenalty = fp.GetSingle();
        if (opts.TryGetProperty("seed", out var sd)) cfg.Seed = sd.GetInt32();
        if (opts.TryGetProperty("stop", out var stopEl) && stopEl.ValueKind == JsonValueKind.Array)
        {
            cfg.StopSequences = new List<string>();
            foreach (var s in stopEl.EnumerateArray())
                if (s.GetString() is string sv) cfg.StopSequences.Add(sv);
        }
    }
    return cfg;
}

static SamplingConfig ParseOpenAIOptions(JsonElement body)
{
    var cfg = new SamplingConfig();
    if (body.TryGetProperty("temperature", out var temp)) cfg.Temperature = temp.GetSingle();
    if (body.TryGetProperty("top_p", out var tp)) cfg.TopP = tp.GetSingle();
    if (body.TryGetProperty("presence_penalty", out var pp)) cfg.PresencePenalty = pp.GetSingle();
    if (body.TryGetProperty("frequency_penalty", out var fp)) cfg.FrequencyPenalty = fp.GetSingle();
    if (body.TryGetProperty("seed", out var sd)) cfg.Seed = sd.GetInt32();
    if (body.TryGetProperty("stop", out var stopEl))
    {
        cfg.StopSequences = new List<string>();
        if (stopEl.ValueKind == JsonValueKind.Array)
        {
            foreach (var s in stopEl.EnumerateArray())
                if (s.GetString() is string stopVal) cfg.StopSequences.Add(stopVal);
        }
        else if (stopEl.ValueKind == JsonValueKind.String && stopEl.GetString() is string singleStop)
        {
            cfg.StopSequences.Add(singleStop);
        }
    }
    return cfg;
}

static List<ChatMessage> ParseOllamaMessages(JsonElement messagesEl, string uploadDir)
{
    var messages = new List<ChatMessage>();
    foreach (var msgEl in messagesEl.EnumerateArray())
    {
        var msg = new ChatMessage
        {
            Role = msgEl.TryGetProperty("role", out var r) ? r.GetString() : "user",
            Content = msgEl.TryGetProperty("content", out var c) ? c.GetString() : ""
        };

        if (msgEl.TryGetProperty("images", out var imgs) && imgs.ValueKind == JsonValueKind.Array)
        {
            msg.ImagePaths = new List<string>();
            foreach (var imgEl in imgs.EnumerateArray())
            {
                string b64 = imgEl.GetString();
                if (!string.IsNullOrEmpty(b64))
                {
                    byte[] imgData = Convert.FromBase64String(b64);
                    string path = Path.Combine(uploadDir, $"{Guid.NewGuid():N}.png");
                    File.WriteAllBytes(path, imgData);
                    msg.ImagePaths.Add(path);
                }
            }
        }

        messages.Add(msg);
    }
    return messages;
}

static List<ChatMessage> ParseOpenAIMessages(JsonElement messagesEl, string uploadDir)
{
    var messages = new List<ChatMessage>();
    foreach (var msgEl in messagesEl.EnumerateArray())
    {
        var msg = new ChatMessage
        {
            Role = msgEl.TryGetProperty("role", out var r) ? r.GetString() : "user"
        };

        if (msgEl.TryGetProperty("content", out var contentEl))
        {
            if (contentEl.ValueKind == JsonValueKind.String)
            {
                msg.Content = contentEl.GetString();
            }
            else if (contentEl.ValueKind == JsonValueKind.Array)
            {
                var textParts = new List<string>();
                msg.ImagePaths = new List<string>();

                foreach (var part in contentEl.EnumerateArray())
                {
                    string type = part.TryGetProperty("type", out var t) ? t.GetString() : "";
                    if (type == "text" && part.TryGetProperty("text", out var txt))
                    {
                        textParts.Add(txt.GetString());
                    }
                    else if (type == "image_url" && part.TryGetProperty("image_url", out var imgUrl))
                    {
                        string url = imgUrl.TryGetProperty("url", out var u) ? u.GetString() : "";
                        if (url.StartsWith("data:"))
                        {
                            int commaIdx = url.IndexOf(',');
                            if (commaIdx > 0)
                            {
                                string b64 = url.Substring(commaIdx + 1);
                                byte[] imgData = Convert.FromBase64String(b64);
                                string path = Path.Combine(uploadDir, $"{Guid.NewGuid():N}.png");
                                File.WriteAllBytes(path, imgData);
                                msg.ImagePaths.Add(path);
                            }
                        }
                    }
                }

                msg.Content = string.Join("\n", textParts);
                if (msg.ImagePaths.Count == 0) msg.ImagePaths = null;
            }
        }

        messages.Add(msg);
    }
    return messages;
}

static List<ToolFunction> ParseOllamaTools(JsonElement body)
{
    if (!body.TryGetProperty("tools", out var toolsEl) || toolsEl.ValueKind != JsonValueKind.Array)
        return null;

    var tools = new List<ToolFunction>();
    foreach (var toolEl in toolsEl.EnumerateArray())
    {
        if (!toolEl.TryGetProperty("function", out var fnEl)) continue;
        var tf = new ToolFunction
        {
            Name = fnEl.TryGetProperty("name", out var n) ? n.GetString() : "",
            Description = fnEl.TryGetProperty("description", out var d) ? d.GetString() : ""
        };

        if (fnEl.TryGetProperty("parameters", out var paramsEl))
        {
            if (paramsEl.TryGetProperty("properties", out var propsEl) &&
                propsEl.ValueKind == JsonValueKind.Object)
            {
                tf.Parameters = new Dictionary<string, ToolParameter>();
                foreach (var prop in propsEl.EnumerateObject())
                {
                    var tp = new ToolParameter
                    {
                        Type = prop.Value.TryGetProperty("type", out var pt) ? pt.GetString() : "string",
                        Description = prop.Value.TryGetProperty("description", out var pd) ? pd.GetString() : null
                    };
                    if (prop.Value.TryGetProperty("enum", out var enumEl) && enumEl.ValueKind == JsonValueKind.Array)
                        tp.Enum = enumEl.EnumerateArray().Select(e => e.GetString()).ToList();
                    tf.Parameters[prop.Name] = tp;
                }
            }
            if (paramsEl.TryGetProperty("required", out var reqEl) && reqEl.ValueKind == JsonValueKind.Array)
                tf.Required = reqEl.EnumerateArray().Select(e => e.GetString()).ToList();
        }
        tools.Add(tf);
    }
    return tools.Count > 0 ? tools : null;
}

static List<ToolFunction> ParseOpenAITools(JsonElement body)
{
    if (!body.TryGetProperty("tools", out var toolsEl) || toolsEl.ValueKind != JsonValueKind.Array)
        return null;

    var tools = new List<ToolFunction>();
    foreach (var toolEl in toolsEl.EnumerateArray())
    {
        string type = toolEl.TryGetProperty("type", out var t) ? t.GetString() : "function";
        if (type != "function") continue;
        if (!toolEl.TryGetProperty("function", out var fnEl)) continue;

        var tf = new ToolFunction
        {
            Name = fnEl.TryGetProperty("name", out var n) ? n.GetString() : "",
            Description = fnEl.TryGetProperty("description", out var d) ? d.GetString() : ""
        };

        if (fnEl.TryGetProperty("parameters", out var paramsEl))
        {
            if (paramsEl.TryGetProperty("properties", out var propsEl) &&
                propsEl.ValueKind == JsonValueKind.Object)
            {
                tf.Parameters = new Dictionary<string, ToolParameter>();
                foreach (var prop in propsEl.EnumerateObject())
                {
                    var tp = new ToolParameter
                    {
                        Type = prop.Value.TryGetProperty("type", out var pt) ? pt.GetString() : "string",
                        Description = prop.Value.TryGetProperty("description", out var pd) ? pd.GetString() : null
                    };
                    if (prop.Value.TryGetProperty("enum", out var enumEl) && enumEl.ValueKind == JsonValueKind.Array)
                        tp.Enum = enumEl.EnumerateArray().Select(e => e.GetString()).ToList();
                    tf.Parameters[prop.Name] = tp;
                }
            }
            if (paramsEl.TryGetProperty("required", out var reqEl) && reqEl.ValueKind == JsonValueKind.Array)
                tf.Required = reqEl.EnumerateArray().Select(e => e.GetString()).ToList();
        }
        tools.Add(tf);
    }
    return tools.Count > 0 ? tools : null;
}

static List<string> DecodeBase64Images(JsonElement body, string uploadDir)
{
    if (!body.TryGetProperty("images", out var imgs) || imgs.ValueKind != JsonValueKind.Array)
        return null;

    var paths = new List<string>();
    foreach (var imgEl in imgs.EnumerateArray())
    {
        string b64 = imgEl.GetString();
        if (!string.IsNullOrEmpty(b64))
        {
            byte[] imgData = Convert.FromBase64String(b64);
            string path = Path.Combine(uploadDir, $"{Guid.NewGuid():N}.png");
            File.WriteAllBytes(path, imgData);
            paths.Add(path);
        }
    }
    return paths.Count > 0 ? paths : null;
}

static string ResolveModelPath(string modelName, string modelDir)
{
    string direct = Path.Combine(modelDir, modelName);
    if (File.Exists(direct)) return direct;

    string withExt = Path.Combine(modelDir, modelName + ".gguf");
    if (File.Exists(withExt)) return withExt;

    var match = Directory.GetFiles(modelDir, "*.gguf")
        .FirstOrDefault(f => Path.GetFileNameWithoutExtension(f)
            .Equals(modelName, StringComparison.OrdinalIgnoreCase));
    return match;
}

Console.WriteLine($"Model directory: {modelDir}");
Console.WriteLine($"Video max frames: {MediaHelper.GetConfiguredMaxVideoFrames()}");
Console.WriteLine("Starting TensorSharp.Server on http://localhost:5000");
Console.WriteLine("API endpoints:");
Console.WriteLine("  GET  /                         - Health check");
Console.WriteLine("  GET  /api/tags                  - List available models (Ollama)");
Console.WriteLine("  POST /api/show                  - Show model details (Ollama)");
Console.WriteLine("  POST /api/generate              - Generate text (Ollama)");
Console.WriteLine("  POST /api/chat/ollama           - Chat completion (Ollama)");
Console.WriteLine("  POST /v1/chat/completions       - Chat completion (OpenAI)");
Console.WriteLine("  GET  /v1/models                 - List models (OpenAI)");
Console.WriteLine("  POST /api/chat                  - Chat (Web UI SSE)");
Console.WriteLine("  POST /api/models/load           - Load model (Web UI)");
Console.WriteLine("  GET  /api/models                - List models (Web UI)");
app.Run("http://0.0.0.0:5000");



