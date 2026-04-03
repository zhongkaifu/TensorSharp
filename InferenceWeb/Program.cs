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
        await foreach (var piece in svc.ChatStreamAsync(messages, maxTokens, ctx.RequestAborted))
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

Console.WriteLine($"Model directory: {modelDir}");
Console.WriteLine("Starting InferenceWeb on http://localhost:5000");
app.Run("http://0.0.0.0:5000");

