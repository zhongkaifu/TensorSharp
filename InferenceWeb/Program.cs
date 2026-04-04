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
using InferenceEngine;
using InferenceWeb;
using Microsoft.AspNetCore.Builder;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.DependencyInjection;

var builder = WebApplication.CreateBuilder(args);
builder.Services.AddSingleton<ModelService>();

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

string modelDir = Environment.GetEnvironmentVariable("MODEL_DIR")
    ?? Path.Combine(AppContext.BaseDirectory, "models");
string defaultBackend = Environment.GetEnvironmentVariable("BACKEND") ?? "ggml_metal";

// ============================================================
// Internal Web UI endpoints (original)
// ============================================================

app.MapGet("/api/models", (ModelService svc) =>
{
    var files = svc.ScanModels(modelDir);
    return Results.Json(new
    {
        models = files,
        loaded = svc.LoadedModelName,
        architecture = svc.Architecture,
        modelDir
    });
});

app.MapPost("/api/models/load", async (HttpRequest req, ModelService svc) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(req.Body);
    string modelName = body.GetProperty("model").GetString();
    string backend = body.TryGetProperty("backend", out var b) ? b.GetString() : defaultBackend;
    string mmproj = body.TryGetProperty("mmproj", out var m) ? m.GetString() : null;

    string modelPath = Path.Combine(modelDir, modelName);
    if (!File.Exists(modelPath))
        return Results.NotFound(new { error = $"Model not found: {modelName}" });

    string mmProjPath = mmproj != null ? Path.Combine(modelDir, mmproj) : null;

    try
    {
        svc.LoadModel(modelPath, mmProjPath, backend);
        return Results.Json(new
        {
            ok = true,
            model = svc.LoadedModelName,
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

    return Results.Json(new { ok = true, path = savePath, mediaType, fileName = file.FileName });
});

app.MapPost("/api/chat", async (HttpContext ctx, ModelService svc) =>
{
    if (!svc.IsLoaded)
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "No model loaded" });
        return;
    }

    var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);
    var messagesEl = body.GetProperty("messages");
    int maxTokens = body.TryGetProperty("maxTokens", out var mt) ? mt.GetInt32() : 200;

    var samplingConfig = ParseSamplingConfig(body);

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

    var writer = ctx.Response.BodyWriter;
    var sw = Stopwatch.StartNew();
    int tokenCount = 0;

    try
    {
        await foreach (var piece in svc.ChatStreamAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig))
        {
            tokenCount++;
            string data = JsonSerializer.Serialize(new { token = piece });
            await ctx.Response.WriteAsync($"data: {data}\n\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        }
    }
    catch (OperationCanceledException) { }

    sw.Stop();
    double tokPerSec = tokenCount > 0 ? tokenCount / sw.Elapsed.TotalSeconds : 0;
    string done = JsonSerializer.Serialize(new { done = true, tokenCount, elapsed = sw.Elapsed.TotalSeconds, tokPerSec });
    await ctx.Response.WriteAsync($"data: {done}\n\n");
    await ctx.Response.Body.FlushAsync();
});

// ============================================================
// Ollama-compatible API endpoints
// ============================================================

app.MapGet("/", () => Results.Ok("InferenceWeb is running"));
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

app.MapPost("/api/generate", async (HttpContext ctx, ModelService svc) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

    if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "model is required" });
        return;
    }

    string modelName = modelProp.GetString();
    if (!svc.EnsureModelLoaded(modelName, modelDir, defaultBackend))
    {
        ctx.Response.StatusCode = 404;
        await ctx.Response.WriteAsJsonAsync(new { error = $"model '{modelName}' not found" });
        return;
    }

    string prompt = body.TryGetProperty("prompt", out var pp) ? pp.GetString() ?? "" : "";
    bool stream = true;
    if (body.TryGetProperty("stream", out var streamProp)) stream = streamProp.GetBoolean();
    int maxTokens = 200;
    var samplingConfig = ParseOllamaOptions(body);
    if (body.TryGetProperty("options", out var opts) && opts.TryGetProperty("num_predict", out var np))
        maxTokens = np.GetInt32();

    var imagePaths = DecodeBase64Images(body, uploadDir);

    if (stream)
    {
        ctx.Response.ContentType = "application/x-ndjson";
        ctx.Response.Headers["Cache-Control"] = "no-cache";

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

app.MapPost("/api/chat/ollama", async (HttpContext ctx, ModelService svc) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

    if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = "model is required" });
        return;
    }

    string modelName = modelProp.GetString();
    if (!svc.EnsureModelLoaded(modelName, modelDir, defaultBackend))
    {
        ctx.Response.StatusCode = 404;
        await ctx.Response.WriteAsJsonAsync(new { error = $"model '{modelName}' not found" });
        return;
    }

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

    if (stream)
    {
        ctx.Response.ContentType = "application/x-ndjson";
        ctx.Response.Headers["Cache-Control"] = "no-cache";

        await foreach (var (piece, done, promptTokens, evalTokens, totalNs, promptNs, evalNs)
            in svc.ChatStreamWithMetricsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig))
        {
            object resp;
            if (!done)
            {
                resp = new
                {
                    model = svc.LoadedModelName,
                    created_at = DateTime.UtcNow.ToString("o"),
                    message = new { role = "assistant", content = piece },
                    done = false
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

            await ctx.Response.WriteAsync(JsonSerializer.Serialize(resp) + "\n", ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
        }
    }
    else
    {
        var sb = new StringBuilder();
        int promptTokens = 0, evalTokens = 0;
        long totalNs = 0, promptNs = 0, evalNs = 0;

        await foreach (var (piece, done, pt, et, tn, pn, en)
            in svc.ChatStreamWithMetricsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig))
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
            message = new { role = "assistant", content = sb.ToString() },
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

// ============================================================
// OpenAI-compatible API endpoint
// ============================================================

app.MapPost("/v1/chat/completions", async (HttpContext ctx, ModelService svc) =>
{
    var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

    if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
    {
        ctx.Response.StatusCode = 400;
        await ctx.Response.WriteAsJsonAsync(new { error = new { message = "model is required", type = "invalid_request_error" } });
        return;
    }

    string modelName = modelProp.GetString();
    if (!svc.EnsureModelLoaded(modelName, modelDir, defaultBackend))
    {
        ctx.Response.StatusCode = 404;
        await ctx.Response.WriteAsJsonAsync(new { error = new { message = $"model '{modelName}' not found", type = "invalid_request_error" } });
        return;
    }

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

    if (stream)
    {
        ctx.Response.ContentType = "text/event-stream";
        ctx.Response.Headers["Cache-Control"] = "no-cache";

        await foreach (var (piece, done, promptTokens, evalTokens, totalNs, promptNs, evalNs)
            in svc.ChatStreamWithMetricsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig))
        {
            if (!done)
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
                            delta = new { role = (string)null, content = piece },
                            finish_reason = (string)null
                        }
                    }
                };
                await ctx.Response.WriteAsync($"data: {JsonSerializer.Serialize(chunk)}\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
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
                await ctx.Response.WriteAsync("data: [DONE]\n\n", ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
        }
    }
    else
    {
        var sb = new StringBuilder();
        int promptTokens = 0, evalTokens = 0;

        await foreach (var (piece, done, pt, et, tn, pn, en)
            in svc.ChatStreamWithMetricsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig))
        {
            if (!done)
                sb.Append(piece);
            else { promptTokens = pt; evalTokens = et; }
        }

        await ctx.Response.WriteAsJsonAsync(new
        {
            id = requestId,
            @object = "chat.completion",
            created = DateTimeOffset.UtcNow.ToUnixTimeSeconds(),
            model = svc.LoadedModelName,
            choices = new[]
            {
                new
                {
                    index = 0,
                    message = new { role = "assistant", content = sb.ToString() },
                    finish_reason = "stop"
                }
            },
            usage = new
            {
                prompt_tokens = promptTokens,
                completion_tokens = evalTokens,
                total_tokens = promptTokens + evalTokens
            }
        });
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
Console.WriteLine("Starting InferenceWeb on http://localhost:5000");
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

