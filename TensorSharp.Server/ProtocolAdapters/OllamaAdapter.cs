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
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;
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
    public sealed class OllamaAdapter
    {
        private readonly ModelService _svc;
        private readonly InferenceQueue _queue;
        private readonly ServerHostingOptions _options;
        private readonly UploadStoragePolicy _uploads;
        private readonly SkillRegistry _skills;
        private readonly ICodeRunner? _codeRunner;
        private readonly SessionWorkspaceManager _workspaces;
        private readonly ILoggerFactory _loggerFactory;

        public OllamaAdapter(
            ModelService svc,
            InferenceQueue queue,
            ServerHostingOptions options,
            UploadStoragePolicy uploads,
            SkillRegistry skills,
            ICodeRunner? codeRunner,
            SessionWorkspaceManager workspaces,
            ILoggerFactory loggerFactory)
        {
            _svc = svc ?? throw new ArgumentNullException(nameof(svc));
            _queue = queue ?? throw new ArgumentNullException(nameof(queue));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _uploads = uploads ?? throw new ArgumentNullException(nameof(uploads));
            _skills = skills ?? throw new ArgumentNullException(nameof(skills));
            _codeRunner = codeRunner;
            _workspaces = workspaces;
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
            var samplingConfig = SamplingConfigParser.ParseOllama(body, _options.SamplingDefaults);
            // Ollama nests the budget in "options"; -1 ("unbounded") and an
            // absent key both mean "use the server's --max-tokens".
            int maxTokens = _options.ResolveMaxTokens(
                body.TryGetProperty("options", out var opts) && opts.ValueKind == JsonValueKind.Object
                    ? SamplingConfigParser.ReadRequestedMaxTokens(opts, "num_predict")
                    : null);

            List<string> imagePaths;
            try
            {
                imagePaths = ChatMessageParser.DecodeBase64Images(body, _uploads);
            }
            catch (UploadLimitExceededException ex)
            {
                generateLogger.LogWarning(LogEventIds.UploadRejected,
                    "/api/generate attachment rejected: {Reason}", ex.Message);
                ctx.Response.StatusCode = ex.StatusCode;
                await ctx.Response.WriteAsJsonAsync(new { error = ex.Message });
                return;
            }

            generateLogger.LogInformation(LogEventIds.ChatStarted,
                "/api/generate request: model={Model} stream={Stream} maxTokens={MaxTokens} images={ImageCount} promptChars={PromptLength} prompt=\"{Prompt}\"",
                modelName, stream, maxTokens, imagePaths?.Count ?? 0, prompt?.Length ?? 0,
                LoggingExtensions.SanitizeForLog(prompt, 512));

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

            await foreach (var update
                in _svc.GenerateStreamAsync(prompt, imagePaths, maxTokens, ctx.RequestAborted, samplingConfig))
            {
                // /api/generate has no tool-call surface, so the only distinction
                // that matters here is truncated vs. finished.
                object resp = update.Done
                    ? OllamaResponseFactory.GenerateFinalChunk(_svc.LoadedModelName,
                        FinishReasonMapper.ToOllamaDoneReason(update.FinishReason, hasToolCalls: false),
                        update.PromptTokens, update.EvalTokens, update.KvCacheReusedTokens,
                        update.TotalNs, update.PromptNs, update.EvalNs)
                    : OllamaResponseFactory.GenerateTokenChunk(_svc.LoadedModelName, update.Piece);

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
            string pipelineFinishReason = null;

            await foreach (var update
                in _svc.GenerateStreamAsync(prompt, imagePaths, maxTokens, ctx.RequestAborted, samplingConfig))
            {
                if (!update.Done)
                {
                    sb.Append(update.Piece);
                }
                else
                {
                    promptTokens = update.PromptTokens; evalTokens = update.EvalTokens; kvReusedTokens = update.KvCacheReusedTokens;
                    totalNs = update.TotalNs; promptNs = update.PromptNs; evalNs = update.EvalNs;
                    pipelineFinishReason = update.FinishReason;
                }
            }

            await ctx.Response.WriteAsJsonAsync(
                OllamaResponseFactory.GenerateNonStreamingResponse(
                    _svc.LoadedModelName, sb.ToString(),
                    FinishReasonMapper.ToOllamaDoneReason(pipelineFinishReason, hasToolCalls: false),
                    promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs));
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
            var samplingConfig = SamplingConfigParser.ParseOllama(body, _options.SamplingDefaults);
            int maxTokens = _options.ResolveMaxTokens(
                body.TryGetProperty("options", out var opts) && opts.ValueKind == JsonValueKind.Object
                    ? SamplingConfigParser.ReadRequestedMaxTokens(opts, "num_predict")
                    : null);

            List<ChatMessage> messages;
            try
            {
                messages = ChatMessageParser.ParseOllama(messagesEl, _uploads);
            }
            catch (UploadLimitExceededException ex)
            {
                ollamaLogger.LogWarning(LogEventIds.UploadRejected,
                    "/api/chat/ollama attachment rejected: {Reason}", ex.Message);
                ctx.Response.StatusCode = ex.StatusCode;
                await ctx.Response.WriteAsJsonAsync(new { error = ex.Message });
                return;
            }
            var ollamaTools = ToolFunctionParser.ParseOllama(body);
            bool ollamaThink = body.TryGetProperty("think", out var thinkProp) && thinkProp.GetBoolean();
            var requestedSkills = SkillSelectionParser.Parse(body);

            string lastOllamaUserContent = LoggingExtensions.SanitizeForLog(
                messages.LastOrDefault(m => m.Role == "user")?.Content ?? string.Empty, 512);
            ollamaLogger.LogInformation(LogEventIds.ChatStarted,
                "/api/chat/ollama request: model={Model} stream={Stream} maxTokens={MaxTokens} messages={Messages} tools={Tools} skills={Skills} thinking={Thinking} userInput=\"{LastUser}\"",
                modelName, stream, maxTokens, messages.Count, ollamaTools?.Count ?? 0,
                requestedSkills?.Count ?? 0, ollamaThink, lastOllamaUserContent);

            if (requestedSkills is { Count: > 0 } && !_options.SkillsEnabled)
            {
                ollamaLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat/ollama rejected: skills requested but disabled");
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = "Agent skills are disabled on this server (--no-skills)." });
                return;
            }

            using RequestWorkspaceLease workspaceLease = RequestWorkspaceLease.Acquire(
                _workspaces, _codeRunner, _svc.Architecture);

            var skillPlan = SkillRequestPlan.Create(
                _skills, requestedSkills, SkillSelectionParser.ParseDiscovery(body), ollamaTools,
                _svc.Architecture, _svc.ContextTokens, _options, out var unknownSkills, codeRunner: _codeRunner,
                workspace: workspaceLease?.Workspace, logger: ollamaLogger);

            if (unknownSkills.Count > 0)
            {
                ollamaLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/api/chat/ollama rejected: unknown skills {Unknown}", string.Join(",", unknownSkills));
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new
                {
                    error = $"No skill called '{unknownSkills[0]}' is installed. "
                            + "GET /api/skills lists what this server has.",
                });
                return;
            }

            // This surface has no StructuredOutputPrompt.Apply to compose with, so the
            // skill block is injected directly. Same shape either way: merged into the
            // leading system message, or prepended as one.
            var ollamaTools2 = skillPlan?.Tools ?? ollamaTools;
            if (skillPlan != null)
            {
                messages = skillPlan.Apply(messages);
                ollamaLogger.LogInformation(LogEventIds.SkillSelected,
                    "/api/chat/ollama skills: selected={Selected} announced={Announced} inlined={Inlined} catalog={Catalog} tools={ToolsOffered}",
                    skillPlan.DescribeSelection(), skillPlan.Prompt.Deferred.Count,
                    skillPlan.Prompt.Inlined.Count,
                    skillPlan.Prompt.Catalog.Count, skillPlan.ToolsOffered);
            }

            using var ticket = _queue.Enqueue(ctx.RequestAborted);

            if (stream)
            {
                await StreamChatAsync(ctx, modelName, messages, maxTokens, samplingConfig, ollamaTools2, ollamaThink, ticket, skillPlan, ollamaLogger);
            }
            else
            {
                await CompleteChatAsync(ctx, modelName, messages, maxTokens, samplingConfig, ollamaTools2, ollamaThink, ticket, skillPlan, ollamaLogger);
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
            QueueTicket ticket,
            SkillRequestPlan skillPlan,
            ILogger skillLogger)
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
            // Set when the skills loop hands over already-separated pieces (see
            // SkillChatLoop). `parser` is bypassed for those and must not be flushed at
            // the end either — the loop's own parser already flushed.
            bool sawParsedUpdate = false;

            await foreach (var update
                in _svc.ChatStreamWithSkillsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig,
                    tools, enableThinking, skillPlan, skillLogger))
            {
                if (!update.Done)
                {
                    if (update.IsParsed)
                    {
                        sawParsedUpdate = true;
                        if (update.ParsedToolCalls is { Count: > 0 })
                            collectedToolCalls = new List<ToolCall>(update.ParsedToolCalls);

                        string parsedThink = string.IsNullOrEmpty(update.ThinkingPiece) ? null : update.ThinkingPiece;
                        string parsedContent = update.Piece ?? "";
                        if (parsedThink == null && parsedContent.Length == 0)
                            continue;

                        await NdJsonWriter.WriteLineAsync(ctx.Response,
                            OllamaResponseFactory.ChatParsedChunk(_svc.LoadedModelName, parsedContent, parsedThink),
                            ctx.RequestAborted, JsonOptions.IgnoreNulls);
                        continue;
                    }

                    object resp = useParser
                        ? BuildParsedChatChunk(_svc.LoadedModelName, parser, update.Piece, ref collectedToolCalls, out bool emit)
                        : OllamaResponseFactory.ChatRawTokenChunk(_svc.LoadedModelName, update.Piece);

                    if (useParser && resp == null)
                        continue;

                    await NdJsonWriter.WriteLineAsync(ctx.Response, resp, ctx.RequestAborted, JsonOptions.IgnoreNulls);
                }
                else
                {
                    if (useParser || sawParsedUpdate)
                    {
                        var finalParsed = sawParsedUpdate ? new ParsedOutput() : parser.Add("", true);
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
                            OllamaResponseFactory.ChatParsedFinalChunk(_svc.LoadedModelName,
                                FinishReasonMapper.ToOllamaDoneReason(update.FinishReason,
                                    collectedToolCalls != null && collectedToolCalls.Count > 0),
                                collectedToolCalls,
                                update.PromptTokens, update.EvalTokens, update.KvCacheReusedTokens,
                                update.TotalNs, update.PromptNs, update.EvalNs),
                            ctx.RequestAborted, JsonOptions.IgnoreNulls);
                    }
                    else
                    {
                        await NdJsonWriter.WriteLineAsync(ctx.Response,
                            OllamaResponseFactory.ChatRawFinalChunk(_svc.LoadedModelName,
                                FinishReasonMapper.ToOllamaDoneReason(update.FinishReason, hasToolCalls: false),
                                update.PromptTokens, update.EvalTokens, update.KvCacheReusedTokens,
                                update.TotalNs, update.PromptNs, update.EvalNs),
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
            QueueTicket ticket,
            SkillRequestPlan skillPlan,
            ILogger skillLogger)
        {
            await ticket.WaitUntilReadyAsync();

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsJsonAsync(new { error = loadError });
                return;
            }

            var collector = new ChatStreamCollector();
            int promptTokens = 0, evalTokens = 0, kvReusedTokens = 0;
            long totalNs = 0, promptNs = 0, evalNs = 0;
            string pipelineFinishReason = null;

            await foreach (var update
                in _svc.ChatStreamWithSkillsAsync(messages, maxTokens, ctx.RequestAborted, samplingConfig,
                    tools, enableThinking, skillPlan, skillLogger))
            {
                if (!update.Done)
                {
                    collector.Add(update);
                }
                else
                {
                    promptTokens = update.PromptTokens; evalTokens = update.EvalTokens; kvReusedTokens = update.KvCacheReusedTokens;
                    totalNs = update.TotalNs; promptNs = update.PromptNs; evalNs = update.EvalNs;
                    pipelineFinishReason = update.FinishReason;
                }
            }

            string rawOutput = collector.PlainText();
            bool useParser = enableThinking || (tools != null && tools.Count > 0)
                || OutputParserFactory.IsAlwaysRequired(_svc.Architecture);

            object finalMessage;
            bool sawToolCalls = false;

            if (useParser || collector.IsParsed)
            {
                var parsed = collector.Resolve(_svc.Architecture, enableThinking, tools);
                string thinkingOut = enableThinking && !string.IsNullOrEmpty(parsed.Thinking) ? parsed.Thinking : null;
                finalMessage = OllamaResponseFactory.ChatNonStreamingMessage(parsed.Content, thinkingOut, parsed.ToolCalls);
                sawToolCalls = parsed.ToolCalls != null && parsed.ToolCalls.Count > 0;
            }
            else
            {
                finalMessage = OllamaResponseFactory.ChatPlainMessage(rawOutput);
            }

            await ctx.Response.WriteAsync(JsonSerializer.Serialize(
                OllamaResponseFactory.ChatNonStreamingResponse(_svc.LoadedModelName, finalMessage,
                    FinishReasonMapper.ToOllamaDoneReason(pipelineFinishReason, sawToolCalls),
                    promptTokens, evalTokens, kvReusedTokens, totalNs, promptNs, evalNs),
                JsonOptions.IgnoreNulls));
        }
    }
}
