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
    /// Implements the OpenAI-compatible chat-completions surface
    /// (<c>POST /v1/chat/completions</c> and <c>GET /v1/models</c>).
    ///
    /// Structured-output and streaming both live here because the protocol's
    /// quirks (<c>json_schema</c> needs a buffered/normalised stream, regular
    /// streams emit per-token chunks plus a final <c>[DONE]</c> sentinel,
    /// non-streaming returns a single <c>chat.completion</c>) are highly
    /// interdependent and easier to follow when kept together.
    /// </summary>
    internal sealed class OpenAIChatAdapter
    {
        private readonly ModelService _svc;
        private readonly InferenceQueue _queue;
        private readonly ServerHostingOptions _options;
        private readonly ILoggerFactory _loggerFactory;

        public OpenAIChatAdapter(
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

        public IResult ListModels()
        {
            var data = string.IsNullOrWhiteSpace(_options.StartupModelPath)
                ? new List<Dictionary<string, object>>()
                : new List<Dictionary<string, object>>
                {
                    new()
                    {
                        ["id"] = Path.GetFileNameWithoutExtension(_options.StartupModelPath),
                        ["object"] = "model",
                        ["owned_by"] = "local",
                    },
                };
            return Results.Json(new { @object = "list", data });
        }

        public async Task ChatCompletionsAsync(HttpContext ctx)
        {
            var openaiLogger = _loggerFactory.CreateLogger("TensorSharp.Server.OpenAI.ChatCompletions");
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

            if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
            {
                openaiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/v1/chat/completions rejected: missing 'model'");
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = "model is required", type = "invalid_request_error" } });
                return;
            }

            string modelName = modelProp.GetString();

            if (!body.TryGetProperty("messages", out var messagesEl) || messagesEl.ValueKind != JsonValueKind.Array)
            {
                openaiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/v1/chat/completions rejected: missing 'messages' (model={Model})", modelName);
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = "messages is required", type = "invalid_request_error" } });
                return;
            }

            bool stream = body.TryGetProperty("stream", out var streamProp) && streamProp.GetBoolean();
            int maxTokens = body.TryGetProperty("max_tokens", out var mtProp) ? mtProp.GetInt32() : 200;
            var samplingConfig = SamplingConfigParser.ParseOpenAI(body, _options.DefaultSamplingConfig);
            var messages = ChatMessageParser.ParseOpenAI(messagesEl, _options.UploadDirectory);
            string requestId = OpenAIResponseFactory.NewRequestId();

            var openaiTools = ToolFunctionParser.ParseOpenAI(body);
            bool openaiThink = body.TryGetProperty("think", out var oaiThinkProp) && oaiThinkProp.GetBoolean();

            string lastOpenAiUserContent = LoggingExtensions.SanitizeForLogFull(messages.LastOrDefault(m => m.Role == "user")?.Content ?? string.Empty);
            openaiLogger.LogInformation(LogEventIds.ChatStarted,
                "/v1/chat/completions request: id={ChatcmplId} model={Model} stream={Stream} maxTokens={MaxTokens} messages={Messages} tools={Tools} thinking={Thinking} userInput=\"{LastUser}\"",
                requestId, modelName, stream, maxTokens, messages.Count, openaiTools?.Count ?? 0, openaiThink, lastOpenAiUserContent);

            if (!OpenAIResponseFormatParser.TryParse(body, out StructuredOutputFormat responseFormat, out string responseFormatError))
            {
                openaiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/v1/chat/completions response_format invalid: {Error} (id={ChatcmplId})", responseFormatError, requestId);
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = responseFormatError, type = "invalid_request_error" } });
                return;
            }

            if (responseFormat != null && !await ValidateStructuredOutputCompatibilityAsync(ctx, responseFormat, openaiThink, openaiTools))
                return;

            var inferenceMessages = StructuredOutputPrompt.Apply(messages, responseFormat);

            using var ticket = _queue.Enqueue(ctx.RequestAborted);

            if (stream)
            {
                await StreamCompletionAsync(ctx, requestId, modelName, inferenceMessages, maxTokens,
                    samplingConfig, openaiTools, openaiThink, responseFormat, ticket);
            }
            else
            {
                await CompleteSyncAsync(ctx, requestId, modelName, inferenceMessages, maxTokens,
                    samplingConfig, openaiTools, openaiThink, responseFormat, ticket);
            }
        }

        // ---- Validation ------------------------------------------------------

        private static async Task<bool> ValidateStructuredOutputCompatibilityAsync(
            HttpContext ctx,
            StructuredOutputFormat responseFormat,
            bool openaiThink,
            List<ToolFunction> openaiTools)
        {
            if (openaiThink)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = "response_format cannot be combined with think=true", type = "invalid_request_error" } });
                return false;
            }

            if (openaiTools != null && openaiTools.Count > 0)
            {
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = "response_format cannot be combined with tools", type = "invalid_request_error" } });
                return false;
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
                        details = schemaValidation.Errors,
                    },
                });
                return false;
            }

            return true;
        }

        // ---- Streaming -------------------------------------------------------

        private async Task StreamCompletionAsync(
            HttpContext ctx,
            string requestId,
            string modelName,
            List<ChatMessage> inferenceMessages,
            int maxTokens,
            SamplingConfig samplingConfig,
            List<ToolFunction> openaiTools,
            bool openaiThink,
            StructuredOutputFormat responseFormat,
            QueueTicket ticket)
        {
            bool structuredStream = responseFormat != null;
            if (!structuredStream)
            {
                SseWriter.ApplyHeaders(ctx.Response);

                while (!ticket.IsReady)
                {
                    await SseWriter.WriteEventAsync(ctx.Response,
                        OpenAIResponseFactory.QueueChunk(requestId, modelName, ticket.Position, _queue.PendingCount),
                        ctx.RequestAborted);
                    await ticket.WaitAsync(TimeSpan.FromSeconds(1));
                }
            }
            else
            {
                await ticket.WaitUntilReadyAsync();
            }

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                if (structuredStream)
                {
                    ctx.Response.StatusCode = 404;
                    await ctx.Response.WriteAsJsonAsync(new { error = new { message = loadError, type = "invalid_request_error" } });
                }
                else
                {
                    await SseWriter.WriteEventAsync(ctx.Response,
                        OpenAIResponseFactory.ErrorContentChunk(requestId, modelName, loadError),
                        ctx.RequestAborted);
                    await SseWriter.WriteDoneSentinelAsync(ctx.Response, ctx.RequestAborted);
                }
                return;
            }

            bool useStreamParser = openaiThink || (openaiTools != null && openaiTools.Count > 0)
                || OutputParserFactory.IsAlwaysRequired(_svc.Architecture);
            bool bufferForStructured = structuredStream;
            var buffer = bufferForStructured ? new StringBuilder() : null;

            IOutputParser parser = null;
            List<ToolCall> collectedToolCalls = null;
            if (useStreamParser && !bufferForStructured)
            {
                parser = OutputParserFactory.Create(_svc.Architecture);
                parser.Init(openaiThink, openaiTools);
            }

            await foreach (var (piece, done, promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs)
                in _svc.ChatStreamWithMetricsAsync(inferenceMessages, maxTokens, ctx.RequestAborted, samplingConfig,
                    openaiTools, openaiThink))
            {
                if (!done)
                {
                    if (bufferForStructured)
                    {
                        buffer.Append(piece);
                        continue;
                    }

                    if (parser != null)
                    {
                        var parsed = parser.Add(piece, false);
                        if (parsed.ToolCalls != null)
                            collectedToolCalls = parsed.ToolCalls;
                        string emitContent = parsed.Content ?? "";
                        if (emitContent.Length == 0 && string.IsNullOrEmpty(parsed.Thinking))
                            continue;
                        string chunkContent = emitContent.Length > 0 ? emitContent : null;
                        await SseWriter.WriteEventAsync(ctx.Response,
                            OpenAIResponseFactory.ContentChunk(requestId, _svc.LoadedModelName, chunkContent),
                            ctx.RequestAborted);
                        continue;
                    }

                    await SseWriter.WriteEventAsync(ctx.Response,
                        OpenAIResponseFactory.ContentChunk(requestId, _svc.LoadedModelName, piece),
                        ctx.RequestAborted);
                    continue;
                }

                if (bufferForStructured)
                {
                    if (!await FlushStructuredCompletionAsync(ctx, requestId, responseFormat,
                            buffer.ToString(), useStreamParser, openaiThink, openaiTools,
                            promptTokens, evalTokens, kvReusedTokens))
                        return;
                    continue;
                }

                if (parser != null)
                {
                    var finalParsed = parser.Add("", true);
                    if (finalParsed.ToolCalls != null)
                        collectedToolCalls = finalParsed.ToolCalls;

                    if (!string.IsNullOrEmpty(finalParsed.Content))
                    {
                        await SseWriter.WriteEventAsync(ctx.Response,
                            OpenAIResponseFactory.ContentChunk(requestId, _svc.LoadedModelName, finalParsed.Content),
                            ctx.RequestAborted);
                    }

                    string finReason = (collectedToolCalls != null && collectedToolCalls.Count > 0) ? "tool_calls" : "stop";
                    await SseWriter.WriteEventAsync(ctx.Response,
                        OpenAIResponseFactory.EndChunk(requestId, _svc.LoadedModelName, finReason, promptTokens, evalTokens, kvReusedTokens),
                        ctx.RequestAborted);
                }
                else
                {
                    await SseWriter.WriteEventAsync(ctx.Response,
                        OpenAIResponseFactory.EndChunk(requestId, _svc.LoadedModelName, "stop", promptTokens, evalTokens, kvReusedTokens),
                        ctx.RequestAborted);
                }

                await SseWriter.WriteDoneSentinelAsync(ctx.Response, ctx.RequestAborted);
                await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            }
        }

        private async Task<bool> FlushStructuredCompletionAsync(
            HttpContext ctx,
            string requestId,
            StructuredOutputFormat responseFormat,
            string rawContent,
            bool useStreamParser,
            bool openaiThink,
            List<ToolFunction> openaiTools,
            int promptTokens,
            int evalTokens,
            int kvCacheReusedTokens)
        {
            if (useStreamParser)
            {
                var structParser = OutputParserFactory.Create(_svc.Architecture);
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
                        details = normalized.Errors,
                    },
                });
                return false;
            }

            SseWriter.ApplyHeaders(ctx.Response);

            await SseWriter.WriteEventAsync(ctx.Response,
                OpenAIResponseFactory.StructuredContentChunk(requestId, _svc.LoadedModelName, normalized.NormalizedContent),
                ctx.RequestAborted);

            await SseWriter.WriteEventAsync(ctx.Response,
                OpenAIResponseFactory.EndChunk(requestId, _svc.LoadedModelName, "stop", promptTokens, evalTokens, kvCacheReusedTokens),
                ctx.RequestAborted);

            await SseWriter.WriteDoneSentinelAsync(ctx.Response, ctx.RequestAborted);
            await ctx.Response.Body.FlushAsync(ctx.RequestAborted);
            return true;
        }

        // ---- Non-streaming ---------------------------------------------------

        private async Task CompleteSyncAsync(
            HttpContext ctx,
            string requestId,
            string modelName,
            List<ChatMessage> inferenceMessages,
            int maxTokens,
            SamplingConfig samplingConfig,
            List<ToolFunction> openaiTools,
            bool openaiThink,
            StructuredOutputFormat responseFormat,
            QueueTicket ticket)
        {
            await ticket.WaitUntilReadyAsync();

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = loadError, type = "invalid_request_error" } });
                return;
            }

            var sb = new StringBuilder();
            int promptTokens = 0, evalTokens = 0, kvReusedTokens = 0;

            await foreach (var (piece, done, pt, et, kr, _, _, _)
                in _svc.ChatStreamWithMetricsAsync(inferenceMessages, maxTokens, ctx.RequestAborted, samplingConfig,
                    openaiTools, openaiThink))
            {
                if (!done)
                    sb.Append(piece);
                else { promptTokens = pt; evalTokens = et; kvReusedTokens = kr; }
            }

            string rawOutput = sb.ToString();
            bool useParser = openaiThink || (openaiTools != null && openaiTools.Count > 0)
                || OutputParserFactory.IsAlwaysRequired(_svc.Architecture);
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
                            details = normalized.Errors,
                        },
                    });
                    return;
                }

                responseMessage = OpenAIResponseFactory.StructuredAssistantMessage(normalized.NormalizedContent);
            }
            else if (useParser)
            {
                var parser = OutputParserFactory.Create(_svc.Architecture);
                parser.Init(openaiThink, openaiTools);
                var parsed = parser.Add(rawOutput, true);

                string thinkingOut = openaiThink && !string.IsNullOrEmpty(parsed.Thinking) ? parsed.Thinking : null;
                responseMessage = OpenAIResponseFactory.ParsedAssistantMessage(parsed.Content, thinkingOut, parsed.ToolCalls);

                if (parsed.ToolCalls != null && parsed.ToolCalls.Count > 0)
                    finishReason = "tool_calls";
            }
            else
            {
                responseMessage = OpenAIResponseFactory.PlainAssistantMessage(rawOutput);
            }

            await ctx.Response.WriteAsync(JsonSerializer.Serialize(
                OpenAIResponseFactory.Completion(requestId, _svc.LoadedModelName, responseMessage,
                    finishReason, promptTokens, evalTokens, kvReusedTokens),
                JsonOptions.IgnoreNulls));
        }
    }
}
