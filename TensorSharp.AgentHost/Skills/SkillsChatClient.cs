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
using System.Globalization;
using System.Linq;
using System.Net.Http;
using System.Net.Http.Headers;
using System.Text;
using System.Text.Json;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Logging;

using TensorSharp.Runtime;
namespace TensorSharp.AgentHost.Skills
{
    /// <summary>Who resolves the skills for a request.</summary>
    public enum SkillDelivery
    {
        /// <summary>
        /// Probe the endpoint once for <c>/v1/skills</c> and use
        /// <see cref="Server"/> if it answers, <see cref="Local"/> otherwise.
        /// </summary>
        Auto,

        /// <summary>
        /// Send a <c>skills</c> field and let the endpoint do everything. The right
        /// choice against TensorSharp.Server: nothing is re-uploaded per request, the
        /// skill files never leave the server, and the progressive-disclosure loop runs
        /// next to the model rather than over the network.
        /// </summary>
        Server,

        /// <summary>
        /// Read the skills from a local <see cref="SkillRegistry"/>, build the prompt
        /// block here, and run the disclosure loop in this process against an ordinary
        /// chat-completions endpoint.
        ///
        /// <para>
        /// This is what makes the API useful beyond TensorSharp: any OpenAI-compatible
        /// endpoint — one that has never heard of skills — gains Agent Skills support,
        /// at the cost of one extra round trip per file the model reads.
        /// </para>
        /// </summary>
        Local,
    }

    /// <summary>How a <see cref="SkillsChatClient"/> reaches its endpoint.</summary>
    public sealed class SkillsChatClientOptions
    {
        /// <summary>
        /// The API root, with or without a trailing <c>/v1</c> —
        /// <c>http://localhost:5000</c> and <c>http://localhost:5000/v1</c> both work.
        /// </summary>
        public required string Endpoint { get; init; }

        /// <summary>Bearer token, when the endpoint wants one. TensorSharp.Server does not.</summary>
        public string? ApiKey { get; init; }

        /// <summary>The model name sent with every request unless one is set per request.</summary>
        public string? DefaultModel { get; init; }

        /// <summary>Who resolves skills. <see cref="SkillDelivery.Auto"/> decides on first use.</summary>
        public SkillDelivery Delivery { get; init; } = SkillDelivery.Auto;

        /// <summary>
        /// The skills available under <see cref="SkillDelivery.Local"/>. Ignored in
        /// <see cref="SkillDelivery.Server"/> mode, where the endpoint owns them.
        /// </summary>
        public SkillRegistry? Registry { get; init; }

        /// <summary>Prompt budgets for local delivery.</summary>
        public SkillPromptOptions PromptOptions { get; init; } = SkillPromptOptions.Default;

        /// <summary>Loop bounds for local delivery.</summary>
        public SkillAgentLoopOptions LoopOptions { get; init; } = SkillAgentLoopOptions.Default;

        /// <summary>Advertise skills the request did not select, so the model can pick one up.</summary>
        public bool Discovery { get; init; } = true;

        /// <summary>Per-request timeout. Only used when the client owns its <see cref="HttpClient"/>.</summary>
        public TimeSpan Timeout { get; init; } = TimeSpan.FromMinutes(10);
    }

    /// <summary>One chat request, with the skills it should be answered under.</summary>
    public sealed class SkillsChatRequest
    {
        /// <summary>The conversation. A leading <c>system</c> message is merged with the skill block rather than displaced.</summary>
        public List<ChatMessage> Messages { get; init; } = new();

        /// <summary>Skill names to use, as they appear in the registry.</summary>
        public List<string> Skills { get; init; } = new();

        /// <summary>The caller's own tools. Never executed by the client; returned for the caller to service.</summary>
        public List<ToolFunction>? Tools { get; init; }

        /// <summary>Overrides <see cref="SkillsChatClientOptions.DefaultModel"/>.</summary>
        public string? Model { get; init; }

        /// <summary>Generation cap. Null leaves it to the endpoint.</summary>
        public int? MaxTokens { get; init; }

        /// <summary>Sampling temperature. Null leaves it to the endpoint.</summary>
        public double? Temperature { get; init; }

        /// <summary>Nucleus sampling. Null leaves it to the endpoint.</summary>
        public double? TopP { get; init; }

        /// <summary>Ask for reasoning, on models that expose it.</summary>
        public bool Think { get; init; }

        /// <summary>
        /// Advertise skills this request did not select. Null follows the client's own
        /// <see cref="SkillsChatClientOptions.Discovery"/> setting.
        /// </summary>
        public bool? Discovery { get; init; }

        /// <summary>Convenience for the common single-turn case.</summary>
        public static SkillsChatRequest User(string prompt, params string[] skills) => new()
        {
            Messages = { new ChatMessage { Role = "user", Content = prompt } },
            Skills = skills.ToList(),
        };
    }

    /// <summary>What a completed request produced.</summary>
    /// <param name="Content">The assistant's answer.</param>
    /// <param name="Thinking">Its reasoning, when the model exposed any.</param>
    /// <param name="ToolCalls">
    /// Calls to the CALLER's own tools, which the client never executes. Non-empty
    /// means the caller must service them and send the results back in a follow-up
    /// request; the skill tools have already been answered.
    /// </param>
    /// <param name="Messages">The full transcript, including anything the disclosure loop appended.</param>
    /// <param name="SkillInvocations">Every skill file the model read, in order. Empty under server delivery.</param>
    /// <param name="FinishReason">Why generation stopped, as the endpoint reported it.</param>
    /// <param name="PromptTokens">Prompt tokens, summed over every round.</param>
    /// <param name="CompletionTokens">Generated tokens, summed over every round.</param>
    /// <param name="Rounds">Generations performed. 1 means the model answered without reading anything.</param>
    public sealed record SkillsChatResponse(
        string Content,
        string? Thinking,
        IReadOnlyList<ToolCall> ToolCalls,
        List<ChatMessage> Messages,
        IReadOnlyList<SkillToolInvocation> SkillInvocations,
        string? FinishReason,
        int PromptTokens,
        int CompletionTokens,
        int Rounds);

    /// <summary>
    /// Calls an OpenAI-compatible chat endpoint with Agent Skills.
    ///
    /// <para>
    /// This is the API a .NET developer uses to get skills into their own application.
    /// It covers the two situations that actually arise:
    /// </para>
    /// <list type="number">
    /// <item><b>Against TensorSharp.Server</b> — pass the skill names and the server
    ///   does the rest, including the progressive-disclosure loop, right next to the
    ///   model. Nothing is uploaded per request.</item>
    /// <item><b>Against any other OpenAI-compatible endpoint</b> — point the client at
    ///   a local <see cref="SkillRegistry"/> and it builds the prompt block, declares
    ///   the skill tools and runs the loop in process, so an endpoint that has never
    ///   heard of skills still behaves as though it had.</item>
    /// </list>
    /// <example>
    /// <code>
    /// var registry = new SkillRegistry(new SkillRegistryOptions { Roots = new[] { "./skills" } });
    /// using var client = new SkillsChatClient(new SkillsChatClientOptions
    /// {
    ///     Endpoint = "http://localhost:5000",
    ///     DefaultModel = "gemma-4-E4B-it-Q8_0",
    ///     Registry = registry,
    /// });
    ///
    /// var reply = await client.CompleteAsync(
    ///     SkillsChatRequest.User("Extract the tables from report.pdf", "pdf"));
    /// Console.WriteLine(reply.Content);
    /// </code>
    /// </example>
    /// </summary>
    public sealed class SkillsChatClient : IDisposable
    {
        private static readonly JsonSerializerOptions Json = new() { PropertyNamingPolicy = null };

        private readonly SkillsChatClientOptions _options;
        private readonly HttpClient _http;
        private readonly bool _ownsHttp;
        private readonly ILogger _logger;
        private readonly SemaphoreSlim _probeGate = new(1, 1);
        private SkillDelivery? _resolvedDelivery;
        private bool? _endpointHasSkillsApi;

        /// <summary>
        /// Create a client.
        /// </summary>
        /// <param name="options">Endpoint, credentials and delivery mode.</param>
        /// <param name="httpClient">
        /// An existing client to use. Supply one from <c>IHttpClientFactory</c> in a
        /// hosted application; leave it null and one is created and disposed with this
        /// instance.
        /// </param>
        /// <param name="logger">Optional.</param>
        public SkillsChatClient(SkillsChatClientOptions options, HttpClient? httpClient = null, ILogger? logger = null)
        {
            _options = options ?? throw new ArgumentNullException(nameof(options));
            if (string.IsNullOrWhiteSpace(options.Endpoint))
                throw new ArgumentException("An endpoint is required.", nameof(options));

            _logger = logger ?? NullLogger.Instance;
            _ownsHttp = httpClient == null;
            _http = httpClient ?? new HttpClient { Timeout = options.Timeout };

            if (!string.IsNullOrEmpty(options.ApiKey))
                _http.DefaultRequestHeaders.Authorization = new AuthenticationHeaderValue("Bearer", options.ApiKey);
        }

        /// <summary>The skills used for local delivery, or null when the endpoint owns them.</summary>
        public SkillRegistry? Registry => _options.Registry;

        /// <summary>
        /// Ask the endpoint which skills it has.
        /// </summary>
        /// <returns>
        /// The skills the endpoint reports, or an empty list when it does not implement
        /// the skills API. Never throws for a 404: an endpoint without skills is a
        /// normal answer to this question, not a failure.
        /// </returns>
        public async Task<IReadOnlyList<SkillDescriptor>> ListServerSkillsAsync(CancellationToken cancellationToken = default)
        {
            using HttpResponseMessage response = await _http
                .GetAsync(BuildUrl("skills"), cancellationToken).ConfigureAwait(false);

            if (!response.IsSuccessStatusCode)
                return Array.Empty<SkillDescriptor>();

            string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            return SkillDescriptor.ParseList(body);
        }

        /// <summary>
        /// Answer <paramref name="request"/>, resolving skills on whichever side
        /// <see cref="SkillsChatClientOptions.Delivery"/> selects.
        /// </summary>
        /// <exception cref="SkillsChatException">The endpoint returned an error or an unreadable body.</exception>
        public async Task<SkillsChatResponse> CompleteAsync(
            SkillsChatRequest request,
            CancellationToken cancellationToken = default)
        {
            ArgumentNullException.ThrowIfNull(request);

            SkillDelivery delivery = await ResolveDeliveryAsync(cancellationToken).ConfigureAwait(false);
            return delivery == SkillDelivery.Server
                ? await CompleteViaServerAsync(request, cancellationToken).ConfigureAwait(false)
                : await CompleteLocallyAsync(request, cancellationToken).ConfigureAwait(false);
        }

        /// <summary>
        /// Which side will resolve skills. Resolves <see cref="SkillDelivery.Auto"/> by
        /// probing once and caching the answer for the client's lifetime.
        /// </summary>
        public async Task<SkillDelivery> ResolveDeliveryAsync(CancellationToken cancellationToken = default)
        {
            if (_options.Delivery != SkillDelivery.Auto)
                return _options.Delivery;
            if (_resolvedDelivery is { } cached)
                return cached;

            await _probeGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (_resolvedDelivery is { } raced)
                    return raced;

                bool serverHasSkills;
                try
                {
                    using HttpResponseMessage response = await _http
                        .GetAsync(BuildUrl("skills"), cancellationToken).ConfigureAwait(false);
                    serverHasSkills = response.IsSuccessStatusCode;
                }
                catch (HttpRequestException)
                {
                    // Unreachable right now says nothing about whether it supports
                    // skills; local delivery is the answer that still works when the
                    // caller has a registry, and the send itself will report the outage.
                    serverHasSkills = false;
                }

                // A server that has the API but no registry of its own is no use for
                // server delivery; fall back to local if this client has skills.
                _resolvedDelivery = serverHasSkills || _options.Registry == null
                    ? SkillDelivery.Server
                    : SkillDelivery.Local;

                _logger.LogInformation(LogEventIds.SkillSelected,
                    "skills.client.delivery mode={Delivery} endpoint={Endpoint}",
                    _resolvedDelivery, _options.Endpoint);
                return _resolvedDelivery.Value;
            }
            finally
            {
                _probeGate.Release();
            }
        }

        // ---- server delivery ---------------------------------------------------

        private async Task<SkillsChatResponse> CompleteViaServerAsync(
            SkillsChatRequest request,
            CancellationToken cancellationToken)
        {
            string payload = BuildPayload(request, request.Messages, request.Tools, includeSkillsField: true);
            OpenAiReply reply = await PostAsync(payload, cancellationToken).ConfigureAwait(false);

            var messages = new List<ChatMessage>(request.Messages) { reply.ToAssistantMessage() };
            return new SkillsChatResponse(
                reply.Content,
                reply.Thinking,
                reply.ToolCalls,
                messages,
                Array.Empty<SkillToolInvocation>(),
                reply.FinishReason,
                reply.PromptTokens,
                reply.CompletionTokens,
                1);
        }

        // ---- local delivery ----------------------------------------------------

        private async Task<SkillsChatResponse> CompleteLocallyAsync(
            SkillsChatRequest request,
            CancellationToken cancellationToken)
        {
            // Against a skills-aware endpoint, local delivery has to tell the server to
            // stand down. Otherwise BOTH sides inject a catalog and both answer the
            // model's reads, the transcript this client returns is missing the fetches
            // the server made, and the reported round count is a fiction. The
            // suppression is only sent when the endpoint is known to understand it —
            // some OpenAI-compatible servers reject unrecognised request fields — which
            // is what the one-time probe below establishes.
            bool suppressServerSkills = await EndpointHasSkillsApiAsync(cancellationToken).ConfigureAwait(false);

            SkillRegistry registry = _options.Registry
                ?? throw new InvalidOperationException(
                    "Local skill delivery needs a SkillRegistry; set SkillsChatClientOptions.Registry, "
                    + "or use SkillDelivery.Server against an endpoint that provides skills itself.");

            IReadOnlyList<Skill> selected = registry.Resolve(request.Skills, out IReadOnlyList<string> unknown);
            if (unknown.Count > 0)
            {
                throw new SkillsChatException(
                    $"No skill called '{unknown[0]}' is registered. Available: "
                    + string.Join(", ", registry.Skills.Select(s => s.Id)) + ".");
            }

            bool discovery = request.Discovery ?? _options.Discovery;
            IReadOnlyList<Skill> catalog = discovery ? registry.Skills : Array.Empty<Skill>();

            SkillPlan plan = SkillPrompt.Plan(selected, catalog, _options.PromptOptions);
            List<ChatMessage> messages = SkillPrompt.Apply(request.Messages, plan);
            List<ToolFunction> tools = SkillTools.Merge(request.Tools, allowScripts: false, out _);

            var context = new SkillToolContext(plan.Reachable.ToList());
            int promptTokens = 0, completionTokens = 0;
            string? finishReason = null;

            SkillLoopResult loop = await SkillAgentLoop.RunAsync(
                messages,
                tools,
                context,
                async (turnMessages, turnTools, ct) =>
                {
                    string payload = BuildPayload(
                        request, turnMessages, turnTools,
                        includeSkillsField: false,
                        suppressServerSkills: suppressServerSkills);
                    OpenAiReply reply = await PostAsync(payload, ct).ConfigureAwait(false);
                    promptTokens += reply.PromptTokens;
                    completionTokens += reply.CompletionTokens;
                    finishReason = reply.FinishReason;
                    return new SkillTurnOutput(new ParsedOutput
                    {
                        Content = reply.Content,
                        Thinking = reply.Thinking ?? string.Empty,
                        ToolCalls = reply.ToolCalls.Count > 0 ? reply.ToolCalls.ToList() : null,
                    });
                },
                // The request's OWN tools, so a name the model invented is answered in
                // the loop rather than returned to a caller that never declared it.
                _options.LoopOptions.WithClientTools(request.Tools),
                cancellationToken).ConfigureAwait(false);

            ParsedOutput final = loop.Output.Parsed ?? new ParsedOutput();
            loop.Messages.Add(new ChatMessage
            {
                Role = "assistant",
                Content = final.Content,
                Thinking = string.IsNullOrEmpty(final.Thinking) ? null : final.Thinking,
                ToolCalls = loop.PendingClientToolCalls.Count > 0 ? loop.PendingClientToolCalls.ToList() : null,
            });

            return new SkillsChatResponse(
                final.Content,
                string.IsNullOrEmpty(final.Thinking) ? null : final.Thinking,
                loop.PendingClientToolCalls,
                loop.Messages,
                loop.Invocations,
                finishReason,
                promptTokens,
                completionTokens,
                loop.Rounds);
        }

        // ---- wire format -------------------------------------------------------

        private string BuildUrl(string path)
        {
            string root = _options.Endpoint.TrimEnd('/');
            if (!root.EndsWith("/v1", StringComparison.OrdinalIgnoreCase))
                root += "/v1";
            return root + "/" + path;
        }

        /// <summary>
        /// Whether the endpoint implements the skills API, probed once and cached.
        /// A failure to reach it answers "no": local delivery still works, and the send
        /// itself will report the outage.
        /// </summary>
        private async Task<bool> EndpointHasSkillsApiAsync(CancellationToken cancellationToken)
        {
            if (_endpointHasSkillsApi is { } cached)
                return cached;

            await _probeGate.WaitAsync(cancellationToken).ConfigureAwait(false);
            try
            {
                if (_endpointHasSkillsApi is { } raced)
                    return raced;

                try
                {
                    using HttpResponseMessage response = await _http
                        .GetAsync(BuildUrl("skills"), cancellationToken).ConfigureAwait(false);
                    _endpointHasSkillsApi = response.IsSuccessStatusCode;
                }
                catch (HttpRequestException)
                {
                    _endpointHasSkillsApi = false;
                }
                return _endpointHasSkillsApi.Value;
            }
            finally
            {
                _probeGate.Release();
            }
        }

        private string BuildPayload(
            SkillsChatRequest request,
            List<ChatMessage> messages,
            List<ToolFunction>? tools,
            bool includeSkillsField,
            bool suppressServerSkills = false)
        {
            string model = request.Model ?? _options.DefaultModel
                ?? throw new InvalidOperationException(
                    "No model was given; set SkillsChatRequest.Model or SkillsChatClientOptions.DefaultModel.");

            var buffer = new System.Buffers.ArrayBufferWriter<byte>();
            using (var writer = new Utf8JsonWriter(buffer))
            {
                writer.WriteStartObject();
                writer.WriteString("model", model);
                writer.WriteBoolean("stream", false);

                if (request.MaxTokens is { } maxTokens)
                    writer.WriteNumber("max_tokens", maxTokens);
                if (request.Temperature is { } temperature)
                    writer.WriteNumber("temperature", temperature);
                if (request.TopP is { } topP)
                    writer.WriteNumber("top_p", topP);
                if (request.Think)
                    writer.WriteBoolean("think", true);

                if (includeSkillsField && request.Skills.Count > 0)
                {
                    writer.WriteStartArray("skills");
                    foreach (string skill in request.Skills)
                        writer.WriteStringValue(skill);
                    writer.WriteEndArray();

                    if (request.Discovery is { } discovery)
                        writer.WriteBoolean("skills_discovery", discovery);
                }
                else if (suppressServerSkills)
                {
                    // An explicitly EMPTY array is how a caller opts out of a server's own
                    // skill selection, and turning discovery off stops it advertising the
                    // rest of its registry. Together they leave this client the only side
                    // putting skills in front of the model.
                    writer.WriteStartArray("skills");
                    writer.WriteEndArray();
                    writer.WriteBoolean("skills_discovery", false);
                }

                BuildMessageIds(messages, out string[][] callIds, out string?[] resultIds);
                writer.WriteStartArray("messages");
                for (int i = 0; i < messages.Count; i++)
                    WriteMessage(writer, messages[i], callIds[i], resultIds[i]);
                writer.WriteEndArray();

                if (tools is { Count: > 0 })
                {
                    writer.WriteStartArray("tools");
                    foreach (ToolFunction tool in tools)
                        WriteTool(writer, tool);
                    writer.WriteEndArray();
                }

                writer.WriteEndObject();
            }
            return Encoding.UTF8.GetString(buffer.WrittenSpan);
        }

        private static void BuildMessageIds(List<ChatMessage> messages, out string[][] callIds, out string?[] resultIds)
        {
            callIds = new string[messages.Count][];
            resultIds = new string?[messages.Count];
            var used = new HashSet<string>(StringComparer.Ordinal);
            foreach (ChatMessage message in messages)
            {
                if (!string.IsNullOrEmpty(message.ToolCallId)) used.Add(message.ToolCallId);
                foreach (ToolCall call in message.ToolCalls ?? new())
                    if (!string.IsNullOrEmpty(call.Id)) used.Add(call.Id);
            }
            int nextId = 0;
            for (int i = 0; i < messages.Count; i++)
            {
                ChatMessage message = messages[i];
                resultIds[i] = message.ToolCallId;
                callIds[i] = new string[message.ToolCalls?.Count ?? 0];
                for (int j = 0; j < callIds[i].Length; j++)
                {
                    string? id = message.ToolCalls![j].Id;
                    if (string.IsNullOrEmpty(id))
                    {
                        // Legacy callers may have no IDs. Generate wire-only IDs,
                        // reserving every explicit ID before choosing fallbacks.
                        do { id = "call_" + (nextId++).ToString(CultureInfo.InvariantCulture); }
                        while (!used.Add(id));
                    }
                    callIds[i][j] = id;
                }
            }
            for (int i = 0; i < messages.Count; i++)
            {
                if (callIds[i].Length == 0) continue;
                int end = i + 1;
                while (end < messages.Count && messages[end].Role == "tool") end++;
                var remaining = new List<string>(callIds[i]);
                // Explicit out-of-order results take precedence over legacy
                // positional association, even if an ID-less result comes first.
                for (int j = i + 1; j < end; j++)
                    if (!string.IsNullOrEmpty(resultIds[j])) remaining.Remove(resultIds[j]!);
                for (int j = i + 1; j < end; j++)
                    if (string.IsNullOrEmpty(resultIds[j]) && remaining.Count > 0)
                    {
                        resultIds[j] = remaining[0];
                        remaining.RemoveAt(0);
                    }
            }
            // An orphan ID-less result stays ID-less: no call can be inferred.
            // This helper never changes caller-owned messages or ToolCall objects.
        }

        private static void WriteMessage(Utf8JsonWriter writer, ChatMessage message, string[] callIds, string? resultId)
        {
            writer.WriteStartObject();
            writer.WriteString("role", string.IsNullOrEmpty(message.Role) ? "user" : message.Role);
            writer.WriteString("content", message.Content ?? string.Empty);
            if (!string.IsNullOrEmpty(resultId))
                writer.WriteString("tool_call_id", resultId);

            if (message.ToolCalls is { Count: > 0 })
            {
                writer.WriteStartArray("tool_calls");
                for (int i = 0; i < message.ToolCalls.Count; i++)
                {
                    ToolCall call = message.ToolCalls[i];
                    writer.WriteStartObject();
                    writer.WriteString("id", callIds[i]);
                    writer.WriteString("type", "function");
                    writer.WriteStartObject("function");
                    writer.WriteString("name", call.Name ?? string.Empty);
                    writer.WriteString("arguments", JsonSerializer.Serialize(call.Arguments ?? new(), Json));
                    writer.WriteEndObject();
                    writer.WriteEndObject();
                }
                writer.WriteEndArray();
            }
            writer.WriteEndObject();
        }

        private static void WriteTool(Utf8JsonWriter writer, ToolFunction tool)
        {
            writer.WriteStartObject();
            writer.WriteString("type", "function");
            writer.WriteStartObject("function");
            writer.WriteString("name", tool.Name ?? string.Empty);
            writer.WriteString("description", tool.Description ?? string.Empty);
            writer.WriteStartObject("parameters");
            writer.WriteString("type", "object");
            writer.WriteStartObject("properties");
            foreach (KeyValuePair<string, ToolParameter> entry in tool.Parameters ?? new())
            {
                writer.WriteStartObject(entry.Key);
                writer.WriteString("type", string.IsNullOrEmpty(entry.Value.Type) ? "string" : entry.Value.Type);
                writer.WriteString("description", entry.Value.Description ?? string.Empty);
                if (entry.Value.Enum is { Count: > 0 })
                {
                    writer.WriteStartArray("enum");
                    foreach (string value in entry.Value.Enum)
                        writer.WriteStringValue(value);
                    writer.WriteEndArray();
                }
                writer.WriteEndObject();
            }
            writer.WriteEndObject();
            writer.WriteStartArray("required");
            foreach (string required in tool.Required ?? new())
                writer.WriteStringValue(required);
            writer.WriteEndArray();
            writer.WriteEndObject();
            writer.WriteEndObject();
            writer.WriteEndObject();
        }

        private async Task<OpenAiReply> PostAsync(string payload, CancellationToken cancellationToken)
        {
            using var content = new StringContent(payload, Encoding.UTF8, "application/json");
            using HttpResponseMessage response = await _http
                .PostAsync(BuildUrl("chat/completions"), content, cancellationToken).ConfigureAwait(false);

            string body = await response.Content.ReadAsStringAsync(cancellationToken).ConfigureAwait(false);
            if (!response.IsSuccessStatusCode)
            {
                throw new SkillsChatException(
                    $"The endpoint returned {(int)response.StatusCode} {response.ReasonPhrase}: "
                    + SkillTextBudget.Truncate(body, 512));
            }

            try
            {
                return OpenAiReply.Parse(body);
            }
            catch (JsonException ex)
            {
                throw new SkillsChatException("The endpoint returned a body that is not a chat completion.", ex);
            }
        }

        /// <summary>Releases the <see cref="HttpClient"/> when this instance created it.</summary>
        public void Dispose()
        {
            _probeGate.Dispose();
            if (_ownsHttp)
                _http.Dispose();
        }

        /// <summary>One <c>chat.completion</c>, reduced to what the client needs.</summary>
        private readonly record struct OpenAiReply(
            string Content,
            string? Thinking,
            IReadOnlyList<ToolCall> ToolCalls,
            string? FinishReason,
            int PromptTokens,
            int CompletionTokens)
        {
            public ChatMessage ToAssistantMessage() => new()
            {
                Role = "assistant",
                Content = Content,
                Thinking = Thinking,
                ToolCalls = ToolCalls.Count > 0 ? ToolCalls.ToList() : null,
            };

            public static OpenAiReply Parse(string body)
            {
                using JsonDocument document = JsonDocument.Parse(body);
                JsonElement root = document.RootElement;

                string content = string.Empty;
                string? thinking = null;
                string? finishReason = null;
                var calls = new List<ToolCall>();

                if (root.TryGetProperty("choices", out JsonElement choices)
                    && choices.ValueKind == JsonValueKind.Array
                    && choices.GetArrayLength() > 0)
                {
                    JsonElement choice = choices[0];
                    if (choice.TryGetProperty("finish_reason", out JsonElement finish)
                        && finish.ValueKind == JsonValueKind.String)
                    {
                        finishReason = finish.GetString();
                    }

                    if (choice.TryGetProperty("message", out JsonElement message)
                        && message.ValueKind == JsonValueKind.Object)
                    {
                        if (message.TryGetProperty("content", out JsonElement text)
                            && text.ValueKind == JsonValueKind.String)
                        {
                            content = text.GetString() ?? string.Empty;
                        }

                        // Reasoning is spelled differently by different servers; accept
                        // every name in circulation rather than dropping it.
                        foreach (string name in new[] { "reasoning_content", "reasoning", "thinking" })
                        {
                            if (message.TryGetProperty(name, out JsonElement reasoning)
                                && reasoning.ValueKind == JsonValueKind.String)
                            {
                                thinking = reasoning.GetString();
                                break;
                            }
                        }

                        if (message.TryGetProperty("tool_calls", out JsonElement toolCalls)
                            && toolCalls.ValueKind == JsonValueKind.Array)
                        {
                            int index = 0;
                            foreach (JsonElement entry in toolCalls.EnumerateArray())
                            {
                                if (!entry.TryGetProperty("function", out JsonElement function)
                                    || function.ValueKind != JsonValueKind.Object)
                                {
                                    continue;
                                }

                                var call = new ToolCall
                                {
                                    Id = entry.TryGetProperty("id", out JsonElement id) && id.ValueKind == JsonValueKind.String
                                        ? id.GetString() : null,
                                    Name = function.TryGetProperty("name", out JsonElement name)
                                           && name.ValueKind == JsonValueKind.String
                                        ? name.GetString() ?? string.Empty
                                        : string.Empty,
                                    Index = index++,
                                };

                                if (function.TryGetProperty("arguments", out JsonElement arguments))
                                    call.Arguments = ParseArguments(arguments);
                                calls.Add(call);
                            }
                        }
                    }
                }

                int promptTokens = 0, completionTokens = 0;
                if (root.TryGetProperty("usage", out JsonElement usage) && usage.ValueKind == JsonValueKind.Object)
                {
                    if (usage.TryGetProperty("prompt_tokens", out JsonElement p) && p.TryGetInt32(out int pv))
                        promptTokens = pv;
                    if (usage.TryGetProperty("completion_tokens", out JsonElement c) && c.TryGetInt32(out int cv))
                        completionTokens = cv;
                }

                return new OpenAiReply(content, thinking, calls, finishReason, promptTokens, completionTokens);
            }

            /// <summary>
            /// Tool-call arguments are a JSON STRING containing JSON in the OpenAI wire
            /// format, but several servers send the object directly. Both are accepted:
            /// rejecting one of them would make the client work against some
            /// implementations and silently return empty arguments against others.
            /// </summary>
            private static Dictionary<string, object> ParseArguments(JsonElement arguments)
            {
                var result = new Dictionary<string, object>(StringComparer.Ordinal);

                JsonElement source = arguments;
                if (arguments.ValueKind == JsonValueKind.String)
                {
                    string text = arguments.GetString() ?? string.Empty;
                    if (string.IsNullOrWhiteSpace(text))
                        return result;
                    try
                    {
                        using JsonDocument parsed = JsonDocument.Parse(text);
                        return ToDictionary(parsed.RootElement);
                    }
                    catch (JsonException)
                    {
                        // Not JSON at all. Hand the raw text through under a single key
                        // rather than losing what the model asked for.
                        result["arguments"] = text;
                        return result;
                    }
                }

                return source.ValueKind == JsonValueKind.Object ? ToDictionary(source) : result;
            }

            private static Dictionary<string, object> ToDictionary(JsonElement element)
            {
                var result = new Dictionary<string, object>(StringComparer.Ordinal);
                if (element.ValueKind != JsonValueKind.Object)
                    return result;
                foreach (JsonProperty property in element.EnumerateObject())
                    result[property.Name] = property.Value.Clone();
                return result;
            }
        }
    }

    /// <summary>A skill as an endpoint describes it over HTTP.</summary>
    /// <param name="Name">The name used to select it.</param>
    /// <param name="Description">What it does and when to use it.</param>
    /// <param name="Files">Its bundled file paths.</param>
    /// <param name="Bytes">Its size on the server.</param>
    /// <param name="Warnings">Anything the server noticed while loading it.</param>
    public sealed record SkillDescriptor(
        string Name,
        string Description,
        IReadOnlyList<string> Files,
        long Bytes,
        IReadOnlyList<string> Warnings)
    {
        /// <summary>
        /// Read a <c>GET /v1/skills</c> body. Tolerant of both the bare-array and
        /// <c>{ "data": [...] }</c> shapes, and of missing optional members.
        /// </summary>
        public static IReadOnlyList<SkillDescriptor> ParseList(string body)
        {
            var result = new List<SkillDescriptor>();
            if (string.IsNullOrWhiteSpace(body))
                return result;

            using JsonDocument document = JsonDocument.Parse(body);
            JsonElement root = document.RootElement;

            if (root.ValueKind == JsonValueKind.Object)
            {
                if (root.TryGetProperty("data", out JsonElement data) && data.ValueKind == JsonValueKind.Array)
                    root = data;
                else if (root.TryGetProperty("skills", out JsonElement skills) && skills.ValueKind == JsonValueKind.Array)
                    root = skills;
                else
                    return result;
            }
            if (root.ValueKind != JsonValueKind.Array)
                return result;

            foreach (JsonElement entry in root.EnumerateArray())
            {
                if (entry.ValueKind != JsonValueKind.Object)
                    continue;

                string name = ReadString(entry, "name") ?? ReadString(entry, "id") ?? string.Empty;
                if (name.Length == 0)
                    continue;

                result.Add(new SkillDescriptor(
                    name,
                    ReadString(entry, "description") ?? string.Empty,
                    ReadStringArray(entry, "files"),
                    entry.TryGetProperty("bytes", out JsonElement bytes) && bytes.TryGetInt64(out long b) ? b : 0,
                    ReadStringArray(entry, "warnings")));
            }
            return result;
        }

        private static string? ReadString(JsonElement obj, string name) =>
            obj.TryGetProperty(name, out JsonElement value) && value.ValueKind == JsonValueKind.String
                ? value.GetString()
                : null;

        private static IReadOnlyList<string> ReadStringArray(JsonElement obj, string name)
        {
            if (!obj.TryGetProperty(name, out JsonElement value) || value.ValueKind != JsonValueKind.Array)
                return Array.Empty<string>();

            var result = new List<string>();
            foreach (JsonElement item in value.EnumerateArray())
            {
                if (item.ValueKind == JsonValueKind.String)
                    result.Add(item.GetString()!);
                else if (item.ValueKind == JsonValueKind.Object && ReadString(item, "path") is { } path)
                    result.Add(path);
            }
            return result;
        }
    }

    /// <summary>Thrown when a skills-aware chat call fails. The message is safe to show a user.</summary>
    public sealed class SkillsChatException : Exception
    {
        public SkillsChatException(string message) : base(message) { }

        public SkillsChatException(string message, Exception inner) : base(message, inner) { }
    }
}
