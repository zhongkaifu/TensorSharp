// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// TensorSharp is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the BSD-3-Clause License for more details.

using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Text.Json;
using System.Threading.Tasks;
using Microsoft.AspNetCore.Http;
using Microsoft.Extensions.Logging;
using TensorSharp.Models;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Logging;
using TensorSharp.AgentHost.CodeExec;
using TensorSharp.AgentHost.Skills;
using TensorSharp.Server.Hosting;
using TensorSharp.Server.Skills;
using TensorSharp.Server.RequestParsers;
using TensorSharp.Server.ResponseSerializers;
using TensorSharp.Server.Responses;
using TensorSharp.Server.StreamingWriters;

namespace TensorSharp.Server.ProtocolAdapters
{
    /// <summary>
    /// Implements the OpenAI-compatible Responses surface
    /// (<c>POST /v1/responses</c> and <c>GET /v1/responses/{id}</c>).
    ///
    /// This is a stateless MVP: <c>previous_response_id</c> conversation
    /// chaining is rejected outright (there is no cross-request session to
    /// chain against), and <c>store</c> only controls whether the completed
    /// response is cached in <see cref="IResponsesStore"/> for later retrieval
    /// by id, not whether it is used as future context. Internal code-tool rounds
    /// share one temporary request workspace, which is deleted before this method
    /// returns and never becomes cross-request state.
    /// </summary>
    public sealed class OpenAIResponsesAdapter
    {
        private readonly ModelService _svc;
        private readonly InferenceQueue _queue;
        private readonly ServerHostingOptions _options;
        private readonly UploadStoragePolicy _uploads;
        private readonly SkillRegistry _skills;
        private readonly ICodeRunner? _codeRunner;
        private readonly SessionWorkspaceManager _workspaces;
        private readonly ILoggerFactory _loggerFactory;
        private readonly IResponsesStore _store;

        public OpenAIResponsesAdapter(
            ModelService svc,
            InferenceQueue queue,
            ServerHostingOptions options,
            UploadStoragePolicy uploads,
            SkillRegistry skills,
            ICodeRunner? codeRunner,
            SessionWorkspaceManager workspaces,
            ILoggerFactory loggerFactory,
            IResponsesStore store)
        {
            _svc = svc ?? throw new ArgumentNullException(nameof(svc));
            _queue = queue ?? throw new ArgumentNullException(nameof(queue));
            _options = options ?? throw new ArgumentNullException(nameof(options));
            _uploads = uploads ?? throw new ArgumentNullException(nameof(uploads));
            _skills = skills ?? throw new ArgumentNullException(nameof(skills));
            _codeRunner = codeRunner;
            _workspaces = workspaces;
            _loggerFactory = loggerFactory ?? throw new ArgumentNullException(nameof(loggerFactory));
            _store = store ?? throw new ArgumentNullException(nameof(store));
        }

        public async Task CreateResponseAsync(HttpContext ctx)
        {
            var logger = _loggerFactory.CreateLogger("TensorSharp.Server.OpenAI.Responses");
            var body = await JsonSerializer.DeserializeAsync<JsonElement>(ctx.Request.Body);

            if (!body.TryGetProperty("model", out var modelProp) || string.IsNullOrWhiteSpace(modelProp.GetString()))
            {
                logger.LogWarning(LogEventIds.HttpRequestRejected, "/v1/responses rejected: missing 'model'");
                await WriteErrorAsync(ctx, 400, "model is required");
                return;
            }
            string modelName = modelProp.GetString();

            if (!body.TryGetProperty("input", out var inputEl) ||
                (inputEl.ValueKind != JsonValueKind.String && inputEl.ValueKind != JsonValueKind.Array))
            {
                logger.LogWarning(LogEventIds.HttpRequestRejected, "/v1/responses rejected: missing 'input' (model={Model})", modelName);
                await WriteErrorAsync(ctx, 400, "input is required and must be a string or an array");
                return;
            }

            if (body.TryGetProperty("previous_response_id", out var prevIdEl) &&
                prevIdEl.ValueKind == JsonValueKind.String && !string.IsNullOrEmpty(prevIdEl.GetString()))
            {
                await WriteErrorAsync(ctx, 400, "previous_response_id is not supported; this server is stateless per-request");
                return;
            }

            string instructions = body.TryGetProperty("instructions", out var instrProp) && instrProp.ValueKind == JsonValueKind.String
                ? instrProp.GetString()
                : null;

            bool stream = body.TryGetProperty("stream", out var streamProp) && streamProp.ValueKind == JsonValueKind.True;
            bool store = !body.TryGetProperty("store", out var storeProp) || storeProp.ValueKind != JsonValueKind.False;

            int? requestedOutputTokens = null;
            if (body.TryGetProperty("max_output_tokens", out var motProp) &&
                motProp.ValueKind != JsonValueKind.Null && motProp.ValueKind != JsonValueKind.Undefined)
            {
                if (motProp.ValueKind != JsonValueKind.Number || !motProp.TryGetInt32(out int parsedOutputTokens))
                {
                    logger.LogWarning(LogEventIds.HttpRequestRejected,
                        "/v1/responses rejected: max_output_tokens must be an integer (model={Model})", modelName);
                    await WriteErrorAsync(ctx, 400, "max_output_tokens must be an integer");
                    return;
                }
                requestedOutputTokens = parsedOutputTokens;
            }

            // No request limit means the server's --max-tokens, not a
            // hard-coded 200 that silently ignored the operator's config.
            int maxOutputTokens = _options.ResolveMaxTokens(requestedOutputTokens);

            var samplingConfig = SamplingConfigParser.ParseOpenAI(body, _options.SamplingDefaults);
            List<ChatMessage> messages;
            try
            {
                messages = ChatMessageParser.ParseResponsesInput(inputEl, instructions, _uploads, logger);
            }
            catch (UploadLimitExceededException ex)
            {
                logger.LogWarning(LogEventIds.UploadRejected,
                    "/v1/responses attachment rejected: {Reason}", ex.Message);
                await WriteErrorAsync(ctx, ex.StatusCode, ex.Message);
                return;
            }
            var tools = ToolFunctionParser.ParseOpenAIResponses(body);
            bool enableThinking = body.TryGetProperty("reasoning", out var reasoningEl) && reasoningEl.ValueKind == JsonValueKind.Object;
            var requestedSkills = SkillSelectionParser.Parse(body);

            string requestId = OpenAIResponsesFactory.NewResponseId();

            string lastUserContent = LoggingExtensions.SanitizeForLogFull(messages.LastOrDefault(m => m.Role == "user")?.Content ?? string.Empty);
            logger.LogInformation(LogEventIds.ChatStarted,
                "/v1/responses request: id={ResponseId} model={Model} stream={Stream} maxOutputTokens={MaxOutputTokens} messages={Messages} tools={Tools} skills={Skills} thinking={Thinking} userInput=\"{LastUser}\"",
                requestId, modelName, stream, maxOutputTokens, messages.Count, tools?.Count ?? 0,
                requestedSkills?.Count ?? 0, enableThinking, lastUserContent);

            if (requestedSkills is { Count: > 0 } && !_options.SkillsEnabled)
            {
                logger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/v1/responses rejected: skills requested but disabled (id={ResponseId})", requestId);
                await WriteErrorAsync(ctx, 400, "Agent skills are disabled on this server (--no-skills).");
                return;
            }

            if (!OpenAIResponseFormatParser.TryParseResponsesText(body, out StructuredOutputFormat responseFormat, out string formatError))
            {
                logger.LogWarning(LogEventIds.HttpRequestRejected, "/v1/responses text.format invalid: {Error} (id={ResponseId})", formatError, requestId);
                await WriteErrorAsync(ctx, 400, formatError);
                return;
            }

            // The compatibility check sees the CLIENT's tools only; the built-in skill
            // tools are added after it, and are suppressed entirely for a structured
            // request (see SkillRequestPlan.Create's allowTools).
            if (responseFormat != null && !await ValidateStructuredOutputCompatibilityAsync(ctx, responseFormat, enableThinking, tools))
                return;

            using RequestWorkspaceLease workspaceLease = RequestWorkspaceLease.Acquire(
                _workspaces, _codeRunner, _svc.Architecture, allowTools: responseFormat == null);

            var skillPlan = SkillRequestPlan.Create(
                _skills, requestedSkills, SkillSelectionParser.ParseDiscovery(body), tools,
                _svc.Architecture, _svc.ContextTokens, _options, out var unknownSkills,
                allowTools: responseFormat == null, codeRunner: _codeRunner,
                workspace: workspaceLease?.Workspace, logger: logger);

            if (unknownSkills.Count > 0)
            {
                logger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/v1/responses rejected: unknown skills {Unknown} (id={ResponseId})",
                    string.Join(",", unknownSkills), requestId);
                await WriteErrorAsync(ctx, 400,
                    $"No skill called '{unknownSkills[0]}' is installed. GET /v1/skills lists what this server has.");
                return;
            }

            var effectiveTools = skillPlan?.Tools ?? tools;
            var inferenceMessages = StructuredOutputPrompt.Apply(messages, responseFormat);
            if (skillPlan != null)
            {
                inferenceMessages = skillPlan.Apply(inferenceMessages);
                logger.LogInformation(LogEventIds.SkillSelected,
                    "/v1/responses skills: id={ResponseId} selected={Selected} announced={Announced} inlined={Inlined} catalog={Catalog} tools={ToolsOffered}",
                    requestId, skillPlan.DescribeSelection(), skillPlan.Prompt.Deferred.Count,
                    skillPlan.Prompt.Inlined.Count,
                    skillPlan.Prompt.Catalog.Count, skillPlan.ToolsOffered);
            }

            using var ticket = _queue.Enqueue(ctx.RequestAborted);

            if (stream)
            {
                await StreamResponseAsync(ctx, requestId, modelName, instructions, maxOutputTokens,
                    inferenceMessages, samplingConfig, effectiveTools, enableThinking, responseFormat, store, ticket,
                    skillPlan, logger);
            }
            else
            {
                await CompleteSyncAsync(ctx, requestId, modelName, instructions, maxOutputTokens,
                    inferenceMessages, samplingConfig, effectiveTools, enableThinking, responseFormat, store, ticket,
                    skillPlan, logger);
            }
        }

        public async Task GetResponseAsync(HttpContext ctx, string id)
        {
            if (_store.TryGet(id, out var stored))
            {
                ctx.Response.ContentType = "application/json";
                await ctx.Response.WriteAsync(stored.Json, ctx.RequestAborted);
                return;
            }

            await WriteErrorAsync(ctx, 404, $"No response found with id '{id}'.");
        }

        // ---- Validation ------------------------------------------------------

        private static async Task<bool> ValidateStructuredOutputCompatibilityAsync(
            HttpContext ctx,
            StructuredOutputFormat responseFormat,
            bool enableThinking,
            List<ToolFunction> tools)
        {
            if (enableThinking)
            {
                await WriteErrorAsync(ctx, 400, "text.format cannot be combined with reasoning");
                return false;
            }

            if (tools != null && tools.Count > 0)
            {
                await WriteErrorAsync(ctx, 400, "text.format cannot be combined with tools");
                return false;
            }

            var schemaValidation = StructuredOutputValidator.ValidateSchema(responseFormat);
            if (!schemaValidation.IsValid)
            {
                await WriteErrorAsync(ctx, 400, schemaValidation.ErrorMessage, schemaValidation.Errors);
                return false;
            }

            return true;
        }

        // ---- Non-streaming ---------------------------------------------------

        private async Task CompleteSyncAsync(
            HttpContext ctx,
            string requestId,
            string modelName,
            string instructions,
            int maxOutputTokens,
            List<ChatMessage> inferenceMessages,
            SamplingConfig samplingConfig,
            List<ToolFunction> tools,
            bool enableThinking,
            StructuredOutputFormat responseFormat,
            bool store,
            QueueTicket ticket,
            SkillRequestPlan skillPlan,
            ILogger skillLogger)
        {
            await ticket.WaitUntilReadyAsync();

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                await WriteErrorAsync(ctx, 404, loadError);
                return;
            }

            var collector = new ChatStreamCollector();
            int promptTokens = 0, evalTokens = 0, kvReusedTokens = 0;
            string pipelineFinishReason = null;

            await foreach (var update
                in _svc.ChatStreamWithSkillsAsync(inferenceMessages, maxOutputTokens, ctx.RequestAborted, samplingConfig, tools, enableThinking, skillPlan, skillLogger))
            {
                if (!update.Done)
                {
                    collector.Add(update);
                }
                else
                {
                    promptTokens = update.PromptTokens;
                    evalTokens = update.EvalTokens;
                    kvReusedTokens = update.KvCacheReusedTokens;
                    pipelineFinishReason = update.FinishReason;
                }
            }

            // This API has no finish_reason: a response cut short by the output-token
            // budget is reported as status "incomplete" with incomplete_details.reason
            // "max_output_tokens".
            var (status, incompleteReason) = FinishReasonMapper.ToResponsesStatus(pipelineFinishReason);
            string itemStatus = status == FinishReasonMapper.ResponsesIncomplete ? "incomplete" : "completed";

            string rawOutput = collector.PlainText();
            bool useParser = enableThinking || (tools != null && tools.Count > 0) || OutputParserFactory.IsAlwaysRequired(_svc.Architecture);
            var output = new List<object>();

            if (responseFormat != null)
            {
                var normalized = StructuredOutputValidator.NormalizeOutput(rawOutput, responseFormat);
                if (!normalized.IsValid)
                {
                    await WriteErrorAsync(ctx, 422, normalized.ErrorMessage, normalized.Errors, "invalid_response_error");
                    return;
                }
                output.Add(OpenAIResponsesFactory.OutputMessageItem(OpenAIResponsesFactory.NewMessageItemId(), normalized.NormalizedContent, itemStatus));
            }
            else if (useParser || collector.IsParsed)
            {
                var parsed = collector.Resolve(_svc.Architecture, enableThinking, tools);

                if (!string.IsNullOrEmpty(parsed.Content))
                    output.Add(OpenAIResponsesFactory.OutputMessageItem(OpenAIResponsesFactory.NewMessageItemId(), parsed.Content, itemStatus));

                if (parsed.ToolCalls != null)
                    foreach (var call in parsed.ToolCalls)
                        output.Add(OpenAIResponsesFactory.FunctionCallItem(
                            OpenAIResponsesFactory.NewFunctionCallItemId(), OpenAIResponsesFactory.NewCallId(), call));
            }
            else
            {
                output.Add(OpenAIResponsesFactory.OutputMessageItem(OpenAIResponsesFactory.NewMessageItemId(), rawOutput, itemStatus));
            }

            var response = OpenAIResponsesFactory.Response(
                requestId, _svc.LoadedModelName, status, instructions, maxOutputTokens, output,
                store, samplingConfig, promptTokens, evalTokens, kvReusedTokens,
                incompleteReason: incompleteReason);

            string json = JsonSerializer.Serialize(response, JsonOptions.IgnoreNulls);
            if (store)
                _store.Store(new StoredResponse { Id = requestId, Json = json });

            ctx.Response.ContentType = "application/json";
            await ctx.Response.WriteAsync(json);
        }

        // ---- Streaming ---------------------------------------------------------

        private async Task StreamResponseAsync(
            HttpContext ctx,
            string requestId,
            string modelName,
            string instructions,
            int maxOutputTokens,
            List<ChatMessage> inferenceMessages,
            SamplingConfig samplingConfig,
            List<ToolFunction> tools,
            bool enableThinking,
            StructuredOutputFormat responseFormat,
            bool store,
            QueueTicket ticket,
            SkillRequestPlan skillPlan,
            ILogger skillLogger)
        {
            await ticket.WaitUntilReadyAsync();
            SseWriter.ApplyHeaders(ctx.Response);

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                await SseWriter.WriteNamedEventAsync(ctx.Response, "response.failed",
                    OpenAIResponsesFactory.Failed(requestId, modelName, loadError), ctx.RequestAborted);
                return;
            }

            await SseWriter.WriteNamedEventAsync(ctx.Response, "response.created",
                OpenAIResponsesFactory.Created(requestId, _svc.LoadedModelName), ctx.RequestAborted);

            bool bufferForStructured = responseFormat != null;
            bool useParser = !bufferForStructured &&
                (enableThinking || (tools != null && tools.Count > 0) || OutputParserFactory.IsAlwaysRequired(_svc.Architecture));

            IOutputParser parser = null;
            if (useParser)
            {
                parser = OutputParserFactory.Create(_svc.Architecture);
                parser.Init(enableThinking, tools);
            }

            var buffer = bufferForStructured ? new StringBuilder() : null;
            string messageItemId = null;
            var messageText = new StringBuilder();
            var toolCalls = new List<ToolCall>();
            int outputIndex = 0;
            int promptTokens = 0, evalTokens = 0, kvReusedTokens = 0;
            string pipelineFinishReason = null;
            // Set when the skills loop hands over already-separated pieces (see
            // SkillChatLoop). `parser` is bypassed for those, so it must not be flushed
            // at the end either.
            bool sawParsedUpdate = false;

            async Task EmitDeltaAsync(string chunk)
            {
                if (string.IsNullOrEmpty(chunk))
                    return;

                if (messageItemId == null)
                {
                    messageItemId = OpenAIResponsesFactory.NewMessageItemId();
                    await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_item.added",
                        OpenAIResponsesFactory.OutputItemAdded(outputIndex, OpenAIResponsesFactory.OutputMessageItem(messageItemId, "")), ctx.RequestAborted);
                    await SseWriter.WriteNamedEventAsync(ctx.Response, "response.content_part.added",
                        OpenAIResponsesFactory.ContentPartAdded(messageItemId, outputIndex, 0), ctx.RequestAborted);
                }

                messageText.Append(chunk);
                await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_text.delta",
                    OpenAIResponsesFactory.OutputTextDelta(messageItemId, outputIndex, 0, chunk), ctx.RequestAborted);
            }

            await foreach (var update
                in _svc.ChatStreamWithSkillsAsync(inferenceMessages, maxOutputTokens, ctx.RequestAborted, samplingConfig, tools, enableThinking, skillPlan, skillLogger))
            {
                if (!update.Done)
                {
                    if (update.IsParsed)
                    {
                        // Pre-separated by the skills loop: the tool markup our parser
                        // would look for is already gone. (Reasoning is dropped here for
                        // the same reason the parsed branch below drops it — this API
                        // surfaces no reasoning item.)
                        sawParsedUpdate = true;
                        if (update.ParsedToolCalls is { Count: > 0 })
                            toolCalls.AddRange(update.ParsedToolCalls);
                        if (bufferForStructured)
                            buffer.Append(update.Piece);
                        else
                            await EmitDeltaAsync(update.Piece);
                        continue;
                    }

                    if (bufferForStructured)
                    {
                        buffer.Append(update.Piece);
                        continue;
                    }

                    if (parser != null)
                    {
                        var parsed = parser.Add(update.Piece, false);
                        if (parsed.ToolCalls != null && parsed.ToolCalls.Count > 0)
                            toolCalls = parsed.ToolCalls.ToList();
                        await EmitDeltaAsync(parsed.Content);
                        continue;
                    }

                    await EmitDeltaAsync(update.Piece);
                    continue;
                }

                promptTokens = update.PromptTokens;
                evalTokens = update.EvalTokens;
                kvReusedTokens = update.KvCacheReusedTokens;
                pipelineFinishReason = update.FinishReason;
            }

            // Same status/incomplete_details mapping the non-streaming path uses, so
            // the response.completed frame's payload matches a buffered GET of the
            // same response id.
            var (status, incompleteReason) = FinishReasonMapper.ToResponsesStatus(pipelineFinishReason);
            string itemStatus = status == FinishReasonMapper.ResponsesIncomplete ? "incomplete" : "completed";

            var output = new List<object>();

            if (bufferForStructured)
            {
                var normalized = StructuredOutputValidator.NormalizeOutput(buffer.ToString(), responseFormat);
                if (!normalized.IsValid)
                {
                    await SseWriter.WriteNamedEventAsync(ctx.Response, "response.failed",
                        OpenAIResponsesFactory.Failed(requestId, _svc.LoadedModelName, normalized.ErrorMessage), ctx.RequestAborted);
                    return;
                }

                await EmitDeltaAsync(normalized.NormalizedContent);
            }
            else if (parser != null && !sawParsedUpdate)
            {
                var finalParsed = parser.Add("", true);
                if (finalParsed.ToolCalls != null && finalParsed.ToolCalls.Count > 0)
                    toolCalls = finalParsed.ToolCalls.ToList();
                await EmitDeltaAsync(finalParsed.Content);
            }

            if (messageItemId != null)
            {
                await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_text.done",
                    OpenAIResponsesFactory.OutputTextDone(messageItemId, outputIndex, 0, messageText.ToString()), ctx.RequestAborted);
                await SseWriter.WriteNamedEventAsync(ctx.Response, "response.content_part.done",
                    OpenAIResponsesFactory.ContentPartDone(messageItemId, outputIndex, 0, messageText.ToString()), ctx.RequestAborted);

                var finishedItem = OpenAIResponsesFactory.OutputMessageItem(messageItemId, messageText.ToString(), itemStatus);
                await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_item.done",
                    OpenAIResponsesFactory.OutputItemDone(outputIndex, finishedItem), ctx.RequestAborted);
                output.Add(finishedItem);
                outputIndex++;
            }

            foreach (var call in toolCalls)
            {
                string fcItemId = OpenAIResponsesFactory.NewFunctionCallItemId();
                var fcItem = OpenAIResponsesFactory.FunctionCallItem(fcItemId, OpenAIResponsesFactory.NewCallId(), call);
                await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_item.added",
                    OpenAIResponsesFactory.OutputItemAdded(outputIndex, fcItem), ctx.RequestAborted);
                await SseWriter.WriteNamedEventAsync(ctx.Response, "response.output_item.done",
                    OpenAIResponsesFactory.OutputItemDone(outputIndex, fcItem), ctx.RequestAborted);
                output.Add(fcItem);
                outputIndex++;
            }

            var response = OpenAIResponsesFactory.Response(
                requestId, _svc.LoadedModelName, status, instructions, maxOutputTokens, output,
                store, samplingConfig, promptTokens, evalTokens, kvReusedTokens,
                incompleteReason: incompleteReason);

            if (store)
                _store.Store(new StoredResponse { Id = requestId, Json = JsonSerializer.Serialize(response, JsonOptions.IgnoreNulls) });

            await SseWriter.WriteNamedEventAsync(ctx.Response, "response.completed",
                OpenAIResponsesFactory.Completed(response), ctx.RequestAborted, JsonOptions.IgnoreNulls);
        }

        // ---- Errors ------------------------------------------------------------

        private static Task WriteErrorAsync(HttpContext ctx, int statusCode, string message, object details = null, string type = "invalid_request_error")
        {
            ctx.Response.StatusCode = statusCode;
            return ctx.Response.WriteAsJsonAsync(new { error = new { message, type, details } }, JsonOptions.IgnoreNulls);
        }
    }
}
