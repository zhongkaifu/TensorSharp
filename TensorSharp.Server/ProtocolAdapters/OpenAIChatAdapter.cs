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
    /// Implements the OpenAI-compatible chat-completions surface
    /// (<c>POST /v1/chat/completions</c> and <c>GET /v1/models</c>).
    ///
    /// Structured-output and streaming both live here because the protocol's
    /// quirks (<c>json_schema</c> needs a buffered/normalised stream, regular
    /// streams emit per-token chunks plus a final <c>[DONE]</c> sentinel,
    /// non-streaming returns a single <c>chat.completion</c>) are highly
    /// interdependent and easier to follow when kept together.
    /// </summary>
    public sealed class OpenAIChatAdapter
    {
        private readonly ModelService _svc;
        private readonly InferenceQueue _queue;
        private readonly ServerHostingOptions _options;
        private readonly UploadStoragePolicy _uploads;
        private readonly SkillRegistry _skills;
        private readonly ICodeRunner? _codeRunner;
        private readonly SessionWorkspaceManager _workspaces;
        private readonly ILoggerFactory _loggerFactory;

        public OpenAIChatAdapter(
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

            bool stream = body.TryGetProperty("stream", out var streamProp) && streamProp.ValueKind == JsonValueKind.True;
            // Absent / null / non-positive falls back to the server's
            // --max-tokens (previously a hard-coded 200 that ignored the
            // operator's configuration entirely).
            int maxTokens = _options.ResolveMaxTokens(
                SamplingConfigParser.ReadRequestedMaxTokens(body, "max_tokens", "max_completion_tokens"));
            var samplingConfig = SamplingConfigParser.ParseOpenAI(body, _options.SamplingDefaults);
            List<ChatMessage> messages;
            try
            {
                messages = ChatMessageParser.ParseOpenAI(messagesEl, _uploads, openaiLogger);
            }
            catch (UploadLimitExceededException ex)
            {
                openaiLogger.LogWarning(LogEventIds.UploadRejected,
                    "/v1/chat/completions attachment rejected: {Reason}", ex.Message);
                ctx.Response.StatusCode = ex.StatusCode;
                await ctx.Response.WriteAsJsonAsync(new
                {
                    error = new
                    {
                        message = ex.Message,
                        // 413 (file over the cap) is the client's doing; 507 (quota) is server state.
                        type = ex.StatusCode == 413 ? "invalid_request_error" : "server_error",
                    },
                });
                return;
            }
            string requestId = OpenAIResponseFactory.NewRequestId();

            var openaiTools = ToolFunctionParser.ParseOpenAI(body);
            bool openaiThink = body.TryGetProperty("think", out var oaiThinkProp) && oaiThinkProp.GetBoolean();
            var requestedSkills = SkillSelectionParser.Parse(body);

            string lastOpenAiUserContent = LoggingExtensions.SanitizeForLog(
                messages.LastOrDefault(m => m.Role == "user")?.Content ?? string.Empty, 512);
            openaiLogger.LogInformation(LogEventIds.ChatStarted,
                "/v1/chat/completions request: id={ChatcmplId} model={Model} stream={Stream} maxTokens={MaxTokens} messages={Messages} tools={Tools} skills={Skills} thinking={Thinking} userInput=\"{LastUser}\"",
                requestId, modelName, stream, maxTokens, messages.Count, openaiTools?.Count ?? 0,
                requestedSkills?.Count ?? 0, openaiThink, lastOpenAiUserContent);

            if (requestedSkills is { Count: > 0 } && !_options.SkillsEnabled)
            {
                openaiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/v1/chat/completions rejected: skills requested but disabled (id={ChatcmplId})", requestId);
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = "Agent skills are disabled on this server (--no-skills).", type = "invalid_request_error" } });
                return;
            }

            if (!OpenAIResponseFormatParser.TryParse(body, out StructuredOutputFormat responseFormat, out string responseFormatError))
            {
                openaiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/v1/chat/completions response_format invalid: {Error} (id={ChatcmplId})", responseFormatError, requestId);
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = responseFormatError, type = "invalid_request_error" } });
                return;
            }

            // The compatibility check sees the CLIENT's tools, never the built-in skill
            // tools added below: a request that combines response_format with skills is
            // legal (skills are delivered inline for it, see SkillRequestPlan.Create),
            // while one that combines response_format with its own tools is not.
            if (responseFormat != null && !await ValidateStructuredOutputCompatibilityAsync(ctx, responseFormat, openaiThink, openaiTools))
                return;

            using RequestWorkspaceLease workspaceLease = RequestWorkspaceLease.Acquire(
                _workspaces, _codeRunner, _svc.Architecture, allowTools: responseFormat == null);

            var skillPlan = SkillRequestPlan.Create(
                _skills, requestedSkills, SkillSelectionParser.ParseDiscovery(body), openaiTools,
                _svc.Architecture, _svc.ContextTokens, _options, out var unknownSkills,
                allowTools: responseFormat == null, codeRunner: _codeRunner,
                workspace: workspaceLease?.Workspace, logger: openaiLogger);

            if (unknownSkills.Count > 0)
            {
                openaiLogger.LogWarning(LogEventIds.HttpRequestRejected,
                    "/v1/chat/completions rejected: unknown skills {Unknown} (id={ChatcmplId})",
                    string.Join(",", unknownSkills), requestId);
                ctx.Response.StatusCode = 400;
                await ctx.Response.WriteAsJsonAsync(new
                {
                    error = new
                    {
                        message = $"No skill called '{unknownSkills[0]}' is installed. "
                                  + "GET /v1/skills lists what this server has.",
                        type = "invalid_request_error",
                    },
                });
                return;
            }

            var effectiveTools = skillPlan?.Tools ?? openaiTools;
            var inferenceMessages = StructuredOutputPrompt.Apply(messages, responseFormat);
            if (skillPlan != null)
            {
                inferenceMessages = skillPlan.Apply(inferenceMessages);
                openaiLogger.LogInformation(LogEventIds.SkillSelected,
                    "/v1/chat/completions skills: id={ChatcmplId} selected={Selected} announced={Announced} inlined={Inlined} catalog={Catalog} tools={ToolsOffered} promptTokens~{PromptTokens}",
                    requestId, skillPlan.DescribeSelection(), skillPlan.Prompt.Deferred.Count,
                    skillPlan.Prompt.Inlined.Count,
                    skillPlan.Prompt.Catalog.Count, skillPlan.ToolsOffered, skillPlan.Prompt.ApproximateTokens);
            }

            using var ticket = _queue.Enqueue(ctx.RequestAborted);

            if (stream)
            {
                await StreamCompletionAsync(ctx, requestId, modelName, inferenceMessages, maxTokens,
                    samplingConfig, effectiveTools, openaiThink, responseFormat, ticket, skillPlan, openaiLogger);
            }
            else
            {
                await CompleteSyncAsync(ctx, requestId, modelName, inferenceMessages, maxTokens,
                    samplingConfig, effectiveTools, openaiThink, responseFormat, ticket, skillPlan, openaiLogger);
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

        // Escape hatch to force the legacy "buffer the entire structured response
        // before sending" behavior (e.g. if a downstream client depends on a
        // single normalized json_object chunk). Off by default so json_object
        // streams incrementally.
        private static bool ForceStructuredStreamBuffer() =>
            string.Equals(Environment.GetEnvironmentVariable("TS_STRUCTURED_STREAM_BUFFER"), "1", StringComparison.Ordinal);

        // Structured output (response_format json_object / json_schema) must
        // produce a JSON object, so constrain the FIRST sampled token to a
        // '{'-opening candidate — the same effect llama.cpp gets from its JSON
        // grammar. Without it, chatty models ramble prose before the object;
        // the streaming filter suppresses that preamble, so clients saw
        // seconds of dead air before the first byte (TTFT looked like decode,
        // not prefill), and the buffered/normalized paths threw the preamble
        // away anyway. TS_JSON_FORCE_OPEN=0 disables.
        private static readonly System.Runtime.CompilerServices.ConditionalWeakTable<object, int[]>
            s_jsonOpenerTokens = new();

        /// <summary>
        /// Attach a grammar constraint for the requested structured-output
        /// format, so the decoder can only emit tokens that keep the response
        /// structurally valid.
        /// </summary>
        /// <remarks>
        /// This replaces prompt-and-hope. Previously JSON mode asked the model
        /// nicely and repaired the result afterwards, which cannot guarantee
        /// anything — a truncated or malformed object still reached the client,
        /// and a tool call built on it failed. With a grammar the invalid tokens
        /// are simply not available to sample.
        /// <para>
        /// Falls back to the old first-token nudge if a grammar cannot be built
        /// (an exotic schema, say): a request that degrades to the previous
        /// behaviour is far better than one that 500s. <c>TS_JSON_GRAMMAR=0</c>
        /// forces that fallback for A/B testing.
        /// </para>
        /// </remarks>
        private SamplingConfig WithStructuredOutputConstraint(
            SamplingConfig samplingConfig, StructuredOutputFormat responseFormat)
        {
            if (responseFormat == null || samplingConfig == null)
                return samplingConfig;
            var tok = _svc.Model?.Tokenizer;
            if (tok == null)
            {
                _loggerFactory.CreateLogger("TensorSharp.Server.OpenAI.StructuredOutput")
                    .LogWarning(
                        "response_format {Kind} was requested but the loaded model exposes no tokenizer, so the " +
                        "constraint is dropped: generating WITHOUT the JSON constraint, and the response may not " +
                        "be valid JSON.", responseFormat.Kind);
                return samplingConfig;
            }

            if (!string.Equals(Environment.GetEnvironmentVariable("TS_JSON_GRAMMAR"), "0", StringComparison.Ordinal))
            {
                try
                {
                    var cache = responseFormat.Kind == StructuredOutputKind.JsonSchema
                        ? TensorSharp.Runtime.Grammar.GrammarLibrary.ForJsonSchema(responseFormat.SchemaJson, tok)
                        : TensorSharp.Runtime.Grammar.GrammarLibrary.ForJsonObject(tok);

                    var withGrammar = samplingConfig.Clone();
                    // One constraint per request: it holds the live parse
                    // position, so sharing it across sequences would let them
                    // advance each other's parser.
                    var constraint =
                        TensorSharp.Runtime.Grammar.GrammarLibrary.NewConstraint(cache, tok);
                    // A model that reasons before it answers must be allowed to
                    // do so: enforcing the schema from token 0 forbids its own
                    // channel header and it answers the shape instead of the
                    // question (see OutputParserFactory.GrammarActivationTrigger).
                    string trigger = OutputParserFactory.GrammarActivationTrigger(_svc.Architecture);
                    if (trigger != null)
                        constraint.ActivateAfter(trigger);
                    withGrammar.Grammar = constraint;
                    return withGrammar;
                }
                catch (Exception ex)
                {
                    _loggerFactory.CreateLogger("TensorSharp.Server.OpenAI.StructuredOutput")
                        .LogWarning(ex,
                            "Could not build a grammar for {Kind}; falling back to the " +
                            "first-token constraint.", responseFormat.Kind);
                }
            }

            return WithJsonFirstTokenConstraint(samplingConfig, responseFormat);
        }

        private SamplingConfig WithJsonFirstTokenConstraint(
            SamplingConfig samplingConfig, StructuredOutputFormat responseFormat)
        {
            if (responseFormat == null || samplingConfig == null)
                return samplingConfig;
            if (string.Equals(Environment.GetEnvironmentVariable("TS_JSON_FORCE_OPEN"), "0", StringComparison.Ordinal))
                return samplingConfig;
            var tokenizer = _svc.Model?.Tokenizer;
            if (tokenizer == null)
                return samplingConfig;

            int[] openers = s_jsonOpenerTokens.GetValue(tokenizer, tk =>
            {
                var t = (TensorSharp.Runtime.ITokenizer)tk;
                var ids = new HashSet<int>();
                // Common object-opening spellings; the sampler picks the most
                // probable, so the model still chooses its preferred one.
                foreach (string opener in new[] { "{", " {", "{\"", "{\n" })
                {
                    try
                    {
                        var enc = t.Encode(opener, addSpecial: false);
                        if (enc != null && enc.Count > 0)
                            ids.Add(enc[0]);
                    }
                    catch
                    {
                        // A tokenizer that can't encode a literal is fine — just
                        // skip that spelling.
                    }
                }
                return ids.Count > 0 ? System.Linq.Enumerable.ToArray(ids) : Array.Empty<int>();
            });
            if (openers.Length == 0)
                return samplingConfig;

            var constrained = samplingConfig.Clone();
            constrained.FirstTokenAllowList = openers;
            return constrained;
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
            QueueTicket ticket,
            SkillRequestPlan skillPlan,
            ILogger skillLogger)
        {
            // Only the strict json_schema path must buffer the whole response so it
            // can be schema-normalized before anything is sent to the client. Plain
            // json_object streams incrementally like a normal completion (this is
            // what OpenAI does too) so its time-to-first-token reflects prefill
            // latency instead of the full decode. TS_STRUCTURED_STREAM_BUFFER=1
            // restores the legacy buffer-everything behavior for both kinds.
            bool bufferForStructured = responseFormat != null
                && (responseFormat.Kind == StructuredOutputKind.JsonSchema
                    || ForceStructuredStreamBuffer());

            if (!bufferForStructured)
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
                if (bufferForStructured)
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

            samplingConfig = WithStructuredOutputConstraint(samplingConfig, responseFormat);

            bool useStreamParser = openaiThink || (openaiTools != null && openaiTools.Count > 0)
                || OutputParserFactory.IsAlwaysRequired(_svc.Architecture);
            var buffer = bufferForStructured ? new StringBuilder() : null;

            IOutputParser parser = null;
            bool sawToolCall = false;
            // Set once the skills loop hands over pre-parsed pieces, so the structured
            // flush below does not run a parser over text that has already been through
            // one.
            bool preParsed = false;
            if (useStreamParser && !bufferForStructured)
            {
                parser = OutputParserFactory.Create(_svc.Architecture);
                parser.Init(openaiThink, openaiTools);
            }

            // json_object streams incrementally: strip code fences / leading prose /
            // stray tags and keep only the balanced JSON object, matching the clean
            // shape the buffered non-streaming path emits. (json_schema still buffers
            // and schema-normalizes via bufferForStructured.)
            var jsonObjectFilter = (!bufferForStructured && responseFormat != null
                && responseFormat.Kind == StructuredOutputKind.JsonObject)
                ? new StreamingJsonObjectFilter() : null;

            await foreach (var update in _svc.ChatStreamWithSkillsAsync(inferenceMessages, maxTokens,
                ctx.RequestAborted, samplingConfig, openaiTools, openaiThink, skillPlan, skillLogger))
            {
                string piece = update.Piece;
                if (!update.Done)
                {
                    if (update.IsParsed)
                    {
                        // Pre-separated by the skills loop (see SkillChatLoop): content,
                        // reasoning and tool calls arrive on their own fields, and the
                        // tool markup that our parser would look for is already gone.
                        preParsed = true;
                        if (bufferForStructured)
                        {
                            buffer.Append(piece);
                            continue;
                        }
                        if (update.ParsedToolCalls is { Count: > 0 })
                        {
                            sawToolCall = true;
                            await SseWriter.WriteEventAsync(ctx.Response,
                                OpenAIResponseFactory.ToolCallsChunk(requestId, _svc.LoadedModelName, update.ParsedToolCalls),
                                ctx.RequestAborted);
                        }
                        if (!string.IsNullOrEmpty(update.ThinkingPiece))
                            await SseWriter.WriteEventAsync(ctx.Response,
                                OpenAIResponseFactory.ReasoningContentChunk(requestId, _svc.LoadedModelName, update.ThinkingPiece),
                                ctx.RequestAborted);
                        if (!string.IsNullOrEmpty(piece))
                        {
                            string parsedContent = jsonObjectFilter != null ? jsonObjectFilter.Feed(piece) : piece;
                            if (!string.IsNullOrEmpty(parsedContent))
                                await SseWriter.WriteEventAsync(ctx.Response,
                                    OpenAIResponseFactory.ContentChunk(requestId, _svc.LoadedModelName, parsedContent),
                                    ctx.RequestAborted);
                        }
                        continue;
                    }

                    if (bufferForStructured)
                    {
                        buffer.Append(piece);
                        continue;
                    }

                    if (parser != null)
                    {
                        var parsed = parser.Add(piece, false);
                        if (parsed.ToolCalls != null && parsed.ToolCalls.Count > 0)
                        {
                            sawToolCall = true;
                            await SseWriter.WriteEventAsync(ctx.Response,
                                OpenAIResponseFactory.ToolCallsChunk(requestId, _svc.LoadedModelName, parsed.ToolCalls),
                                ctx.RequestAborted);
                        }
                        // Stream the model's reasoning ("analysis"/thinking channel)
                        // as incremental reasoning_content deltas so the first token
                        // reaches the client right after prefill. Previously this was
                        // parsed then dropped (a null-content chunk), which buffered
                        // the whole reasoning block and inflated TTFT to the full
                        // reasoning-decode time for reasoning-first models (gpt-oss).
                        if (!string.IsNullOrEmpty(parsed.Thinking))
                        {
                            await SseWriter.WriteEventAsync(ctx.Response,
                                OpenAIResponseFactory.ReasoningContentChunk(requestId, _svc.LoadedModelName, parsed.Thinking),
                                ctx.RequestAborted);
                        }
                        string emitContent = parsed.Content ?? "";
                        if (emitContent.Length == 0)
                            continue;
                        string chunkContent = emitContent;
                        if (jsonObjectFilter != null)
                        {
                            chunkContent = jsonObjectFilter.Feed(chunkContent);
                            if (chunkContent.Length == 0)
                                continue;
                        }
                        await SseWriter.WriteEventAsync(ctx.Response,
                            OpenAIResponseFactory.ContentChunk(requestId, _svc.LoadedModelName, chunkContent),
                            ctx.RequestAborted);
                        continue;
                    }

                    string passthrough = piece;
                    if (jsonObjectFilter != null)
                    {
                        passthrough = jsonObjectFilter.Feed(piece);
                        if (passthrough.Length == 0)
                            continue;
                    }
                    await SseWriter.WriteEventAsync(ctx.Response,
                        OpenAIResponseFactory.ContentChunk(requestId, _svc.LoadedModelName, passthrough),
                        ctx.RequestAborted);
                    continue;
                }

                if (bufferForStructured)
                {
                    if (!await FlushStructuredCompletionAsync(ctx, requestId, responseFormat,
                            buffer.ToString(), useStreamParser && !preParsed, openaiThink, openaiTools, update))
                        return;
                    continue;
                }

                if (parser != null)
                {
                    var finalParsed = parser.Add("", true);
                    if (finalParsed.ToolCalls != null && finalParsed.ToolCalls.Count > 0)
                    {
                        sawToolCall = true;
                        await SseWriter.WriteEventAsync(ctx.Response,
                            OpenAIResponseFactory.ToolCallsChunk(requestId, _svc.LoadedModelName, finalParsed.ToolCalls),
                            ctx.RequestAborted);
                    }

                    if (!string.IsNullOrEmpty(finalParsed.Thinking))
                        await SseWriter.WriteEventAsync(ctx.Response,
                            OpenAIResponseFactory.ReasoningContentChunk(requestId, _svc.LoadedModelName, finalParsed.Thinking),
                            ctx.RequestAborted);

                    if (!string.IsNullOrEmpty(finalParsed.Content))
                    {
                        string finalContent = jsonObjectFilter != null
                            ? jsonObjectFilter.Feed(finalParsed.Content)
                            : finalParsed.Content;
                        if (!string.IsNullOrEmpty(finalContent))
                            await SseWriter.WriteEventAsync(ctx.Response,
                                OpenAIResponseFactory.ContentChunk(requestId, _svc.LoadedModelName, finalContent),
                                ctx.RequestAborted);
                    }

                    // The final chunk must carry the same reason the non-streaming
                    // path would report; clients switch on it identically either way.
                    string finReason = FinishReasonMapper.ToOpenAIChat(update.FinishReason, sawToolCall);
                    await SseWriter.WriteEventAsync(ctx.Response,
                        OpenAIResponseFactory.EndChunk(requestId, _svc.LoadedModelName, finReason,
                            update.PromptTokens, update.EvalTokens, update.KvCacheReusedTokens),
                        ctx.RequestAborted);
                }
                else
                {
                    await SseWriter.WriteEventAsync(ctx.Response,
                        OpenAIResponseFactory.EndChunk(requestId, _svc.LoadedModelName,
                            FinishReasonMapper.ToOpenAIChat(update.FinishReason, hasToolCalls: false),
                            update.PromptTokens, update.EvalTokens, update.KvCacheReusedTokens),
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
            ChatStreamUpdate update)
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

            // Structured output never reports tool_calls — response_format and tools
            // are rejected as a pair upstream. It can still be cut off by the budget
            // though, and reaching here means the truncated JSON happened to survive
            // normalization, so "length" is what tells the client to retry with more
            // room rather than trust a document that stops early.
            await SseWriter.WriteEventAsync(ctx.Response,
                OpenAIResponseFactory.EndChunk(requestId, _svc.LoadedModelName,
                    FinishReasonMapper.ToOpenAIChat(update.FinishReason, hasToolCalls: false),
                    update.PromptTokens, update.EvalTokens, update.KvCacheReusedTokens),
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
            QueueTicket ticket,
            SkillRequestPlan skillPlan,
            ILogger skillLogger)
        {
            await ticket.WaitUntilReadyAsync();

            if (!HostedModelGuard.TryEnsureHostedModelLoaded(_svc, modelName,
                    _options.StartupModelPath, _options.StartupMmProjPath, _options.DefaultBackend, out string loadError))
            {
                ctx.Response.StatusCode = 404;
                await ctx.Response.WriteAsJsonAsync(new { error = new { message = loadError, type = "invalid_request_error" } });
                return;
            }

            samplingConfig = WithStructuredOutputConstraint(samplingConfig, responseFormat);

            var collector = new ChatStreamCollector();
            int promptTokens = 0, evalTokens = 0, kvReusedTokens = 0;
            string pipelineFinishReason = null;

            await foreach (var update in _svc.ChatStreamWithSkillsAsync(inferenceMessages, maxTokens,
                ctx.RequestAborted, samplingConfig, openaiTools, openaiThink, skillPlan, skillLogger))
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

            string rawOutput = collector.PlainText();
            bool useParser = openaiThink || (openaiTools != null && openaiTools.Count > 0)
                || OutputParserFactory.IsAlwaysRequired(_svc.Architecture);
            object responseMessage;
            bool sawToolCalls = false;

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
            else if (useParser || collector.IsParsed)
            {
                var parsed = collector.Resolve(_svc.Architecture, openaiThink, openaiTools);

                string thinkingOut = openaiThink && !string.IsNullOrEmpty(parsed.Thinking) ? parsed.Thinking : null;
                responseMessage = OpenAIResponseFactory.ParsedAssistantMessage(parsed.Content, thinkingOut, parsed.ToolCalls);

                sawToolCalls = parsed.ToolCalls != null && parsed.ToolCalls.Count > 0;
            }
            else
            {
                responseMessage = OpenAIResponseFactory.PlainAssistantMessage(rawOutput);
            }

            await ctx.Response.WriteAsync(JsonSerializer.Serialize(
                OpenAIResponseFactory.Completion(requestId, _svc.LoadedModelName, responseMessage,
                    FinishReasonMapper.ToOpenAIChat(pipelineFinishReason, sawToolCalls),
                    promptTokens, evalTokens, kvReusedTokens),
                JsonOptions.IgnoreNulls));
        }
    }
}
