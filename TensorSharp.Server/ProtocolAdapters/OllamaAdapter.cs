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
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
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
    /// Implements the Ollama-compatible HTTP surface:
    /// <list type="bullet">
    ///   <item><c>GET /api/tags</c> - list hosted models</item>
    ///   <item><c>POST /api/show</c> - show model details</item>
    ///   <item><c>POST /api/generate</c> - one-shot generation (NDJSON or JSON)</item>
    ///   <item><c>POST /api/chat/ollama</c> - multi-turn chat (NDJSON or JSON)</item>
    /// </list>
    /// </summary>
    internal sealed class OllamaAdapter
    {
        private readonly ModelService _svc;
        private readonly InferenceQueue _queue;
        private readonly ServerHostingOptions _options;
        private readonly ILoggerFactory _loggerFactory;

        public OllamaAdapter(
            ModelService svc,
            InferenceQueue queue,
            ServerHostingOptions options,
            ILoggerFactory loggerFactory)
        {
            _svc = svc ?? throw new ArgumentNullException(nameof(svc));
            _queue = queue ?? throw new ArgumentNullException(nameof(queue));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
        }

        // ---- Discovery -------------------------------------------------------

        public IResult GetTags()
        {
            var files = string.IsNullOrWhiteSpace(_options.StartupModelPath)
                ? Enumerable.Empty<string>()
                : new[] { _options.StartupModelPath };
            var models = files.Select(path =>
            {
                var fi = new FileInfo(path);
                string fileName = Path.GetFileName(path);
                return new Dictionary<string, object>
                {
                    ["name"] = Path.GetFileNameWithoutExtension(fileName),
                    ["model"] = fileName,
                    ["size"] = fi.Exists ? fi.Length : 0,
                    ["modified_at"] = fi.Exists ? fi.LastWriteTimeUtc.ToString("o") : "",
                };
            }).ToList();
            return Results.Json(new { models });
        }

        public async Task ShowAsync(HttpContext ctx)
        {
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);
            if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = "model is required" });
                return;
            }

            string modelName = modelProp.GetString();
            if (!HostedModelGuard.TryResolveHostedModelRequest(modelName, _options.StartupModelPath, out string modelPath, out string modelError))
            {
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsJsonAsync(new { error = modelError });
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
                    family = _svc.IsLoaded && _svc.LoadedModelName == Path.GetFileName(modelPath) ? _svc.Architecture : "",
                },
                model_info = new
                {
                    file = Path.GetFileName(modelPath),
                    size = fi.Length,
                },
            });
        }

        // ---- Generate --------------------------------------------------------

        public async Task GenerateAsync(HttpContext ctx)
        {
            var generateLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Ollama.Generate");
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

            if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
            {
                generateLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/generate rejected: missing 'model'");
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = "model is required" });
                return;
            }

            string modelName = modelProp.GetString();
            string prompt = body.TryGetProperty("prompt", out var pp) ? pp.GetString() ?? "" : "";
            bool stream = true;
            if (body.TryGetProperty("stream", out var streamProp)) stream = streamProp.GetBoolean();
            int maxTokens = 200;
            var samplingConfig = SamplingConfigParser.ParseOllama(body);
            if (body.TryGetProperty("options", out var opts) && opts.TryGetProperty("num_predict", out var np))
                maxTokens = np.GetInt32();

            var imagePaths = ChatMessageParser.DecodeBase64Images(body, _options.UploadDirectory);

            generateLogger.LogInformation(LogEventIds.ChatStarted,
                "/api/generate request: model={Model} stream={Stream} maxTokens={MaxTokens} images={ImageCount} promptChars={PromptLength} prompt=\"{Prompt}\"",
                modelName, stream, maxTokens, imagePaths?.Count ?? 0, prompt?.Length ?? 0,
                LoggingExtensions.SanitizeForLogFull(prompt));

            using var ticket = _queue.Enqueue(ctx.RequestAborted);

            if (stream)
            {
                await StreamGenerateAsync(ctx, modelName, prompt, imagePaths, maxTokens, samplingConfig, ticket);
            }
            else
            {
                await CompleteGenerateAsync(ctx, modelName, prompt, imagePaths, maxTokens, samplingConfig, ticket);
            }
        }

        private async Task StreamGenerateAsync(
            HttpContext ctx,
            string modelName,
            string prompt,
            List<string> imagePaths,
            int maxTokens,
            SamplingConfig samplingConfig,
            QueueTicket ticket)
        {
            NdJsonWriter.ApplyHeaders(ctx.Response);

            while (!ticket.IsReady)
            {
                await NdJsonWriter.WriteLineAsync(ctx.Response,
                    OllamaResponseFactory.QueueGenerateChunk(modelName, ticket.Position, _queue.PendingCount),
                    ctx.RequestAborted);
                await ticket.WaitAsync(TimeSpan.FromSeconds(1));
            }

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                await NdJsonWriter.WriteLineAsync(ctx.Response,
                    OllamaResponseFactory.GenerateError(modelName, loadError),
                    ctx.RequestAborted);
                return;
            }

            await foreach (var (piece, done, promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs)
                in _svc.GenerateStreamAsync(prompt, imagePaths, maxTokens, ctx.RequestAborted, samplingConfig))
            {
                object resp = done
                    ? OllamaResponseFactory.GenerateFinalChunk(_svc.LoadedModelName, promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs)
                    : OllamaResponseFactory.GenerateTokenChunk(_svc.LoadedModelName, piece);

                await NdJsonWriter.WriteLineAsync(ctx.Response, resp, ctx.RequestAborted);
            }
        }

        private async Task CompleteGenerateAsync(
            HttpContext ctx,
            string modelName,
            string prompt,
            List<string> imagePaths,
            int maxTokens,
            SamplingConfig samplingConfig,
            QueueTicket ticket)
        {
            await ticket.WaitUntilReadyAsync();

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsJsonAsync(new { error = loadError });
                return;
            }

            var sb = new StringBuilder();
            int promptTokens = 0, evalTokens = 0, kvReusedTokens = 0;
            long totalNs = 0, promptNs = 0, evalNs = 0;

            await foreach (var (piece, done, pt, et, kr, tn, pn, en)
                in _svc.GenerateStreamAsync(prompt, imagePaths, maxTokens, ctx.RequestAborted, samplingConfig))
            {
                if (!done)
                    sb.Append(piece);
                else
                {
                    promptTokens = pt; evalTokens = et; kvReusedTokens = kr;
                    totalNs = tn; promptNs = pn; evalNs = en;
                }
            }

            await ctx.Response.WriteAsJsonAsync(
                OllamaResponseFactory.GenerateNonStreamingResponse(
                    _svc.LoadedModelName, sb.ToString(), promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs));
        }

        // ---- Chat ------------------------------------------------------------

        public async Task ChatAsync(HttpContext ctx)
        {
            var ollamaLogger = _loggerFactory.CreateLogger("TensorSharp.Server.Ollama.Chat");
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

            if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
            {
                ollamaLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat/ollama rejected: missing 'model'");
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = "model is required" });
                return;
            }

            string modelName = modelProp.GetString();

            if (!body.TryGetProperty("messages", out var messagesEl) || messagesEl.ValueKind != JsonValueKind.Array)
            {
                ollamaLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat/ollama rejected: missing 'messages' (model={Model})", modelName);
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = "messages is required" });
                return;
            }

            bool stream = true;
            if (body.TryGetProperty("stream", out var streamProp)) stream = streamProp.GetBoolean();
            int maxTokens = 200;
            var samplingConfig = SamplingConfigParser.ParseOllama(body);
            if (body.TryGetProperty("options", out var opts) && opts.TryGetProperty("num_predict", out var np))
                maxTokens = np.GetInt32();

            var messages = ChatMessageParser.ParseOllama(messagesEl, _options.UploadDirectory);
            var ollamaTools = ToolFunctionParser.ParseOllama(body);
            bool ollamaThink = body.TryGetProperty("think", out var thinkProp) && thinkProp.GetBoolean();

            string lastOllamaUserContent = LoggingExtensions.SanitizeForLogFull(messages.LastOrDefault(m => m.Role == "user")?.Content ?? string.Empty);
            ollamaLogger.LogInformation(LogEventIds.ChatStarted,
                "/api/chat/ollama request: model={Model} stream={Stream} maxTokens={MaxTokens} messages={Messages} tools={Tools} thinking={Thinking} userInput=\"{LastUser}\"",
                modelName, stream, maxTokens, messages.Count, ollamaTools?.Count ?? 0, ollamaThink, lastOllamaUserContent);

            using var ticket = _queue.Enqueue(ctx.RequestAborted);

            if (stream)
            {
                await StreamChatAsync(ctx, modelName, messages, maxTokens, samplingConfig, ollamaTools, ollamaThink, ticket);
            }
            else
            {
                await CompleteChatAsync(ctx, modelName, messages, maxTokens, samplingConfig, ollamaTools, ollamaThink, ticket);
            }
        }

        private async Task StreamChatAsync(
            HttpContext ctx,
            string modelName,
            List<ChatMessage> messages,
            int maxTokens,
            SamplingConfig samplingConfig,
            List<ToolFunction> tools,
            bool enableThinking,
            QueueTicket ticket)
        {
            NdJsonWriter.ApplyHeaders(ctx.Response);

            while (!ticket.IsReady)
            {
                await NdJsonWriter.WriteLineAsync(ctx.Response,
                    OllamaResponseFactory.QueueChatChunk(modelName, ticket.Position, _queue.PendingCount),
                    ctx.RequestAborted);
                await ticket.WaitAsync(TimeSpan.FromSeconds(1));
            }

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                await NdJsonWriter.WriteLineAsync(ctx.Response,
                    OllamaResponseFactory.ChatErrorChunk(modelName, loadError),
                    ctx.RequestAborted);
                return;
            }

            var parser = OutputParserFactory.Create(_svc.Architecture);
            parser.Init(enableThinking, tools);
            bool useParser = enableThinking || (tools != null && tools.Count > 0) || parser.AlwaysRequired;
            List<ToolCall> collectedToolCalls = null;

            await foreach (var (piece, done, promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs)
                in _svc.ChatStreamWithMetricsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig,
                    tools, enableThinking))
            {
                if (!done)
                {
                    object resp = useParser
                        ? BuildParsedChatChunk(_svc.LoadedModelName, parser, piece, ref collectedToolCalls, out bool emit)
                        : OllamaResponseFactory.ChatRawTokenChunk(_svc.LoadedModelName, piece);

                    if (useParser && resp == null)
                        continue;

                    await NdJsonWriter.WriteLineAsync(ctx.Response, resp, ctx.RequestAborted, JsonOptions.IgnoreNulls);
                }
                else
                {
                    if (useParser)
                    {
                        var finalParsed = parser.Add("", true);
                        if (finalParsed.ToolCalls != null)
                            collectedToolCalls = finalParsed.ToolCalls;

                        if (!string.IsNullOrEmpty(finalParsed.Thinking) || !string.IsNullOrEmpty(finalParsed.Content))
                        {
                            string thinkChunk = !string.IsNullOrEmpty(finalParsed.Thinking) ? finalParsed.Thinking : null;
                            string contentChunk = finalParsed.Content ?? "";
                            await NdJsonWriter.WriteLineAsync(ctx.Response,
                                OllamaResponseFactory.ChatParsedChunk(_svc.LoadedModelName, contentChunk, thinkChunk),
                                ctx.RequestAborted, JsonOptions.IgnoreNulls);
                        }

                        await NdJsonWriter.WriteLineAsync(ctx.Response,
                            OllamaResponseFactory.ChatParsedFinalChunk(_svc.LoadedModelName, collectedToolCalls,
                                promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs),
                            ctx.RequestAborted, JsonOptions.IgnoreNulls);
                    }
                    else
                    {
                        await NdJsonWriter.WriteLineAsync(ctx.Response,
                            OllamaResponseFactory.ChatRawFinalChunk(_svc.LoadedModelName,
                                promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs),
                            ctx.RequestAborted, JsonOptions.IgnoreNulls);
                    }
                }
            }
        }

        /// <summary>
        /// Helper that runs the streaming output parser on a single token and
        /// returns either the JSON chunk to emit or null when the parser has
        /// nothing user-visible to emit yet (e.g. it's still buffering thinking
        /// markers).
        /// </summary>
        private static object BuildParsedChatChunk(
            string model,
            IOutputParser parser,
            string piece,
            ref List<ToolCall> collectedToolCalls,
            out bool emit)
        {
            var parsed = parser.Add(piece, false);
            if (parsed.ToolCalls != null)
                collectedToolCalls = parsed.ToolCalls;

            string thinkChunk = !string.IsNullOrEmpty(parsed.Thinking) ? parsed.Thinking : null;
            string contentChunk = parsed.Content ?? "";

            if (thinkChunk == null && contentChunk.Length == 0)
            {
                emit = false;
                return null;
            }

            emit = true;
            return OllamaResponseFactory.ChatParsedChunk(model, contentChunk, thinkChunk);
        }

        private async Task CompleteChatAsync(
            HttpContext ctx,
            string modelName,
            List<ChatMessage> messages,
            int maxTokens,
            SamplingConfig samplingConfig,
            List<ToolFunction> tools,
            bool enableThinking,
            QueueTicket ticket)
        {
            await ticket.WaitUntilReadyAsync();

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsJsonAsync(new { error = loadError });
                return;
            }

            var sb = new StringBuilder();
            int promptTokens = 0, evalTokens = 0, kvReusedTokens = 0;
            long totalNs = 0, promptNs = 0, evalNs = 0;

            await foreach (var (piece, done, pt, et, kr, tn, pn, en)
                in _svc.ChatStreamWithMetricsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig,
                    tools, enableThinking))
            {
                if (!done)
                    sb.Append(piece);
                else
                {
                    promptTokens = pt; evalTokens = et; kvReusedTokens = kr;
                    totalNs = tn; promptNs = pn; evalNs = en;
                }
            }

            string rawOutput = sb.ToString();
            var parser = OutputParserFactory.Create(_svc.Architecture);
            parser.Init(enableThinking, tools);
            bool useParser = enableThinking || (tools != null && tools.Count > 0) || parser.AlwaysRequired;

            object finalMessage;
            string doneReason = "stop";

            if (useParser)
            {
                var parsed = parser.Add(rawOutput, true);
                string thinkingOut = enableThinking && !string.IsNullOrEmpty(parsed.Thinking) ? parsed.Thinking : null;
                finalMessage = OllamaResponseFactory.ChatNonStreamingMessage(parsed.Content, thinkingOut, parsed.ToolCalls);
                if (parsed.ToolCalls != null && parsed.ToolCalls.Count > 0)
                    doneReason = "tool_calls";
            }
            else
            {
                finalMessage = OllamaResponseFactory.ChatPlainMessage(rawOutput);
            }

            await ctx.Response.WriteAsync(JsonSerializer.Serialize(
                OllamaResponseFactory.ChatNonStreamingResponse(_svc.LoadedModelName, finalMessage, doneReason,
                    promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs),
                JsonOptions.IgnoreNulls));
        }
    }
}
