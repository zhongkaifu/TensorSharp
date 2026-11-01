using System.Runtime.CompilerServices;
using System.Text;
using Microsoft.Extensions.AI;
using TensorSharp.Runtime;
using TensorSharp.Server;
using AiMessage = Microsoft.Extensions.AI.ChatMessage;
using NativeMessage = TensorSharp.Runtime.ChatMessage;

namespace TensorSharp.MicrosoftExtensionsAISample;

/// <summary>A sample-local, text-only IChatClient over TensorSharp. One instance represents one conversation.</summary>
public sealed class TensorSharpChatClient : IChatClient
{
    static readonly HashSet<string> MetricKeys = ["tensorsharp.backend", "tensorsharp.finish_reason",
        "tensorsharp.kv_cache_reused_tokens", "tensorsharp.total_ns", "tensorsharp.prompt_ns", "tensorsharp.eval_ns"];
    static readonly ConditionalWeakTable<ModelService, SemaphoreSlim> ModelGates = new();
    readonly Func<ChatSession, List<NativeMessage>, int, CancellationToken, SamplingConfig, IAsyncEnumerable<ChatStreamUpdate>> generate;
    readonly SemaphoreSlim gate;
    readonly ModelService? model;
    readonly ChatClientMetadata metadata;
    readonly string architecture, backend;
    readonly object stateLock = new();
    readonly CancellationTokenSource lifetime = new();
    ChatSession session = new();
    List<(string Role, string Text)> completedHistory = [];
    int activeCalls;
    bool disposed;

    // The caller owns the loaded model. Conversation disposal releases only this adapter's session.
    public TensorSharpChatClient(ModelService model) : this(
        model?.LoadedModelName ?? throw new ArgumentException("Load a model before creating the chat client.", nameof(model)),
        model.Architecture, model.LoadedBackend,
        (session, history, limit, token, sampling) => model.ChatStreamWithMetricsAsync(
            session, history, limit, token, sampling, tools: null, enableThinking: false),
        ModelGates.GetValue(model, _ => new SemaphoreSlim(1, 1)), model) { }

    // A sample-local test seam; the public constructor always calls the real TensorSharp runtime.
    internal TensorSharpChatClient(string modelId, string architecture, string backend,
        Func<ChatSession, List<NativeMessage>, int, CancellationToken, SamplingConfig, IAsyncEnumerable<ChatStreamUpdate>> generate,
        SemaphoreSlim gate, ModelService? model = null)
    {
        ArgumentException.ThrowIfNullOrWhiteSpace(modelId);
        this.architecture = architecture;
        this.backend = backend;
        this.generate = generate;
        this.gate = gate;
        this.model = model;
        metadata = new ChatClientMetadata("tensorsharp", providerUri: null, defaultModelId: modelId);
    }

    public async Task<ChatResponse> GetResponseAsync(IEnumerable<AiMessage> messages, ChatOptions? options = null,
        CancellationToken cancellationToken = default)
    {
        AdditionalPropertiesDictionary? metrics = null;
        async IAsyncEnumerable<ChatResponseUpdate> CaptureAsync()
        {
            await foreach (var update in GetStreamingResponseAsync(messages, options, cancellationToken).ConfigureAwait(false))
            {
                metrics = update.AdditionalProperties ?? metrics;
                yield return update;
            }
        }
        var response = await CaptureAsync().ToChatResponseAsync(cancellationToken).ConfigureAwait(false);
        // The official aggregator combines content/usage; provider-specific update properties need explicit retention.
        response.AdditionalProperties = metrics;
        return response;
    }

    public async IAsyncEnumerable<ChatResponseUpdate> GetStreamingResponseAsync(IEnumerable<AiMessage> messages,
        ChatOptions? options = null, [EnumeratorCancellation] CancellationToken cancellationToken = default)
    {
        ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed), this);
        ArgumentNullException.ThrowIfNull(messages);
        var settings = ValidateOptions(options);
        var history = ConvertHistory(messages, options?.Instructions);
        lock (stateLock)
        {
            ObjectDisposedException.ThrowIf(disposed, this);
            activeCalls++;
        }
        using var request = CancellationTokenSource.CreateLinkedTokenSource(cancellationToken, lifetime.Token);
        IAsyncEnumerator<ChatStreamUpdate>? native = null;
        bool committed = false, started = false, gateHeld = false;
        try
        {
            await gate.WaitAsync(request.Token).ConfigureAwait(false);
            gateHeld = true;
            ObjectDisposedException.ThrowIf(disposed, this);
            if (history.Count <= completedHistory.Count || !completedHistory.Select((m, i) =>
                history[i].Role == m.Role && history[i].Content == m.Text).All(match => match))
                throw new ArgumentException("Pass the complete unchanged history plus a new user message. Use a new adapter for a different conversation.", nameof(messages));

            string id = Guid.NewGuid().ToString("N");
            var parser = OutputParserFactory.Create(architecture);
            parser.Init(enableThinking: false, tools: null);
            var answer = new StringBuilder();
            ChatStreamUpdate? terminal = null;
            started = true;
            native = generate(session, history, settings.Limit, request.Token, settings.Sampling).GetAsyncEnumerator(request.Token);
            while (await native.MoveNextAsync().ConfigureAwait(false))
            {
                var update = native.Current;
                if (!update.Done && update.RawGenerationSuffix is { } suffix) parser.SetGenerationPromptSuffix(suffix);
                string text = (update.IsParsed ? update.Piece : parser.Add(update.Piece ?? "", update.Done).Content) ?? "";
                answer.Append(text);
                if (text.Length != 0) yield return new ChatResponseUpdate(ChatRole.Assistant, text)
                { ResponseId = id, MessageId = id, ModelId = metadata.DefaultModelId };
                if (update.Done) terminal = update;
            }
            var done = NativeCompletion.Require(terminal, !string.IsNullOrWhiteSpace(answer.ToString()), request.Token);
            ChatFinishReason finish = done.FinishReason switch
            {
                "max_tokens" or "thinking_budget" or "repetition" => ChatFinishReason.Length,
                "eos" or "stop_sequence" or "stop" => ChatFinishReason.Stop,
                _ => throw new InvalidOperationException("Unsupported TensorSharp finish reason: " + done.FinishReason)
            };
            completedHistory = history.Select(m => (m.Role, m.Content ?? "")).ToList();
            completedHistory.Add(("assistant", answer.ToString()));
            committed = true;
            yield return new ChatResponseUpdate(ChatRole.Assistant, "")
            {
                ResponseId = id, MessageId = id, ModelId = metadata.DefaultModelId, FinishReason = finish,
                Contents = [new UsageContent(new UsageDetails
                {
                    InputTokenCount = done.PromptTokens, OutputTokenCount = done.EvalTokens,
                    TotalTokenCount = (long)done.PromptTokens + done.EvalTokens,
                    CachedInputTokenCount = done.KvCacheReusedTokens
                })],
                AdditionalProperties = new AdditionalPropertiesDictionary
                {
                    ["tensorsharp.backend"] = backend, ["tensorsharp.finish_reason"] = done.FinishReason,
                    ["tensorsharp.kv_cache_reused_tokens"] = done.KvCacheReusedTokens,
                    ["tensorsharp.total_ns"] = done.TotalNs, ["tensorsharp.prompt_ns"] = done.PromptNs,
                    ["tensorsharp.eval_ns"] = done.EvalNs
                }
            };
        }
        finally
        {
            try
            {
                request.Cancel();
                // Abandoning a stream must stop native generation before another caller takes the model gate.
                if (native is not null)
                {
                    if (!committed)
                    {
                        try { while (await native.MoveNextAsync().ConfigureAwait(false)) { } }
                        catch (Exception) { /* Cleanup of a cancelled/faulted iterator must not replace its original error. */ }
                    }
                    await native.DisposeAsync().ConfigureAwait(false);
                }
            }
            finally
            {
                if (gateHeld)
                {
                    lock (stateLock)
                    {
                        if (started && !committed)
                        {
                            session.Dispose();
                            if (!disposed) session = new ChatSession();
                        }
                    }
                    gate.Release();
                }
                EndCall();
            }
        }
    }

    public object? GetService(Type serviceType, object? serviceKey = null)
    {
        ArgumentNullException.ThrowIfNull(serviceType);
        ObjectDisposedException.ThrowIf(Volatile.Read(ref disposed), this);
        if (serviceKey is not null) return null;
        if (serviceType == typeof(ChatClientMetadata)) return metadata;
        if (serviceType == typeof(IChatClient) || serviceType == GetType()) return this;
        return serviceType == typeof(ModelService) ? model : null;
    }

    public void Dispose()
    {
        lock (stateLock)
        {
            if (disposed) return;
            Volatile.Write(ref disposed, true);
            activeCalls++; // Keep the cancellation source alive while dispatching callbacks outside the lock.
        }
        try { lifetime.Cancel(); }
        finally { EndCall(); }
    }

    void EndCall()
    {
        lock (stateLock)
        {
            if (--activeCalls == 0 && disposed) { session.Dispose(); lifetime.Dispose(); }
        }
    }

    (int Limit, SamplingConfig Sampling) ValidateOptions(ChatOptions? options)
    {
        if (options?.ConversationId is not null || options?.ContinuationToken is not null || options?.AllowBackgroundResponses == true)
            throw new NotSupportedException("This adapter requires full history; conversation IDs, resumable and background responses are unsupported.");
        if (options?.Tools is { Count: > 0 } || options?.ToolMode is not null || options?.AllowMultipleToolCalls is not null
            || options?.ResponseFormat is not null || options?.Reasoning is not null || options?.RawRepresentationFactory is not null
            || options?.AdditionalProperties is { Count: > 0 })
            throw new NotSupportedException("This text-only sample does not implement tools, response formats, reasoning options or provider-specific request properties.");
        if (options?.ModelId is { } requested && requested != metadata.DefaultModelId)
            throw new ArgumentException("The adapter cannot switch its loaded model. Use its ChatClientMetadata.DefaultModelId.", nameof(options));
        int limit = options?.MaxOutputTokens ?? 128;
        if (limit is <= 0 or > 65536) throw new ArgumentOutOfRangeException(nameof(options), "MaxOutputTokens must be between 1 and 65536.");
        float temperature = options?.Temperature ?? 0, topP = options?.TopP ?? 1;
        int topK = options?.TopK ?? 0;
        float frequency = options?.FrequencyPenalty ?? 0, presence = options?.PresencePenalty ?? 0;
        if (!float.IsFinite(temperature) || temperature < 0 || !float.IsFinite(topP) || topP is <= 0 or > 1 || topK < 0
            || !float.IsFinite(frequency) || !float.IsFinite(presence))
            throw new ArgumentOutOfRangeException(nameof(options), "Sampling values must be finite; temperature/top-k nonnegative and top-p in (0,1].");
        if (options?.Seed is < int.MinValue or > int.MaxValue)
            throw new ArgumentOutOfRangeException(nameof(options), "TensorSharp's sampler seed is a 32-bit integer.");
        if (options?.StopSequences?.Any(string.IsNullOrEmpty) == true)
            throw new ArgumentException("Stop sequences must not contain empty values.", nameof(options));
        return (limit, new SamplingConfig
        {
            Temperature = temperature, TopK = topK, TopP = topP, RepetitionPenalty = 1,
            FrequencyPenalty = frequency, PresencePenalty = presence, Seed = (int)(options?.Seed ?? 1),
            StopSequences = options?.StopSequences?.ToList()
        });
    }

    static List<NativeMessage> ConvertHistory(IEnumerable<AiMessage> messages, string? instructions)
    {
        var history = new List<NativeMessage>();
        if (!string.IsNullOrWhiteSpace(instructions)) history.Add(new() { Role = "system", Content = instructions });
        foreach (var message in messages)
        {
            ArgumentNullException.ThrowIfNull(message);
            if (message.Contents.Any(content => content is not TextContent) || message.AuthorName is not null
                || message.RawRepresentation is not null
                || message.AdditionalProperties?.Keys.Any(key => !MetricKeys.Contains(key)) == true)
                throw new NotSupportedException("Only unnamed text messages are supported; media, tools and named senders are not implemented.");
            string role = message.Role == ChatRole.User ? "user" : message.Role == ChatRole.Assistant ? "assistant"
                : message.Role == ChatRole.System ? "system" : throw new NotSupportedException("Only system, user and assistant roles are supported.");
            history.Add(new() { Role = role, Content = message.Text });
        }
        if (history.Count == 0 || history[^1].Role != "user" || string.IsNullOrWhiteSpace(history[^1].Content))
            throw new ArgumentException("Provide full conversation history ending with a nonempty user text message.", nameof(messages));
        return history;
    }
}
