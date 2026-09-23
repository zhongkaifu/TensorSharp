// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Cli;

/// <summary>
/// Synchronous console facade over the same inference engine used by the server.
/// The engine owns prefill, radix prefix reuse, decoding and speculative state.
///
/// <para>
/// <see cref="Generate"/> is the console's own generation: one caller at a time, in the
/// session's conversation scope. <see cref="GenerateInScopeAsync"/> is for sub-agents,
/// which generate CONCURRENTLY with it and with each other through the same engine, so
/// the engine batches their decoding: every call names its own cache scope and touches
/// no per-session state, which is what makes it safe from any thread.
/// </para>
/// </summary>
internal sealed class CliInferenceSession : IDisposable
{
    private readonly IModelArchitecture _model;
    private readonly SchedulerConfig _config;
    private readonly ILogger _logger;
    private string _scope = $"cli-{Guid.NewGuid():N}";
    private InferenceEngine _engine;

    /// <summary>
    /// Guards <see cref="_engine"/>'s creation and release. A sub-agent's first round and
    /// the parent's next round can race to create it, and two engines over one model would
    /// each drive its KV state as if it owned it.
    /// </summary>
    private readonly object _engineGate = new();

    public CliInferenceSession(IModelArchitecture model, SchedulerConfig config, ILogger logger)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _logger = logger;
    }

    /// <summary>Tokens computed by the last request; not the radix tree's total retained size.</summary>
    public int CachedTokens { get; private set; }

    /// <summary>
    /// The current conversation's cache scope. A sub-agent forked from this conversation
    /// generates in it, because its prompt IS this conversation plus one tool result, and
    /// only this scope can reach the conversation's private state.
    /// </summary>
    public string CacheScope => _scope;

    /// <summary>
    /// A cache scope of its own for one fresh sub-agent: never shared, so the agent reuses
    /// only the public prefix it has in common with its parent and keeps its own rounds'
    /// state to itself.
    /// </summary>
    public static string NewAgentScope() => $"cli-agent-{Guid.NewGuid():N}";

    /// <param name="priority">
    /// <see cref="SequenceState.Priority"/>. Zero, except in a turn that runs sub-agents:
    /// under KV pressure the engine preempts the lowest-ranked, newest sequence, which would
    /// otherwise be the parent's round submitted after its agents' — the one generation
    /// every agent's result is waiting on.
    /// </param>
    public Result Generate(IReadOnlyList<int> prompt, int maxTokens, SamplingConfig sampling,
        Func<int, bool> onToken = null, CancellationToken cancellationToken = default,
        string requestId = null, IReadOnlyList<PromptMediaSpan> mediaSpans = null,
        bool enablePrefixCache = true, int sharedPrefixTokens = 0, int priority = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(maxTokens);
        cancellationToken.ThrowIfCancellationRequested();
        InferenceEngine engine = Engine();
        var sequence = CreateSequence(engine, prompt, maxTokens, sampling, requestId, mediaSpans,
            enablePrefixCache, sharedPrefixTokens, _scope, priority);
        var watch = Stopwatch.StartNew();
        var handle = engine.SubmitRequest(sequence, cancellationToken);
        var tokens = new List<int>();
        double? firstTokenMs = null;
        bool accepting = maxTokens > 0;
        try
        {
            while (handle.Tokens.WaitToReadAsync().AsTask().GetAwaiter().GetResult())
            {
                while (handle.Tokens.TryRead(out int token))
                {
                    if (!accepting || cancellationToken.IsCancellationRequested)
                        continue;
                    firstTokenMs ??= watch.Elapsed.TotalMilliseconds;
                    tokens.Add(token);
                    if (onToken != null && !onToken(token))
                    {
                        accepting = false;
                        engine.Abort(sequence.RequestId);
                    }
                }
            }
            var completion = handle.Completion.GetAwaiter().GetResult();
            if (maxTokens == 0)
                completion = PrefillOnly(completion);
            CachedTokens = sequence.NumComputedTokens;
            return new Result(tokens, completion, sequence,
                firstTokenMs ?? watch.Elapsed.TotalMilliseconds, watch.Elapsed.TotalMilliseconds);
        }
        finally
        {
            // Even an output callback failure must finish the worker's request before
            // the caller clears media embeddings, swaps models, or starts another turn.
            if (!handle.Completion.IsCompleted)
                engine.Abort(sequence.RequestId);
            handle.Completion.GetAwaiter().GetResult();
            CachedTokens = sequence.NumComputedTokens;
        }
    }

    /// <summary>
    /// One generation in an explicit cache scope, for a sub-agent: safe to call from any
    /// thread while <see cref="Generate"/> or other calls of this method are in flight.
    ///
    /// <para>
    /// It shares the engine and nothing else. It never reads or moves the session's own
    /// scope and never writes <see cref="CachedTokens"/>, which reports the CONSOLE's last
    /// request; and it awaits the token channel rather than parking a thread on it, since
    /// several of these run at once for the whole length of a generation.
    /// </para>
    /// </summary>
    public async Task<Result> GenerateInScopeAsync(IReadOnlyList<int> prompt, int maxTokens,
        SamplingConfig sampling, string cacheScope, Func<int, bool> onToken = null,
        CancellationToken cancellationToken = default, string requestId = null,
        IReadOnlyList<PromptMediaSpan> mediaSpans = null, bool enablePrefixCache = true,
        int sharedPrefixTokens = 0, int priority = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(maxTokens);
        ArgumentException.ThrowIfNullOrEmpty(cacheScope);
        cancellationToken.ThrowIfCancellationRequested();
        InferenceEngine engine = Engine();
        var sequence = CreateSequence(engine, prompt, maxTokens, sampling, requestId, mediaSpans,
            enablePrefixCache, sharedPrefixTokens, cacheScope, priority);
        var watch = Stopwatch.StartNew();
        var handle = engine.SubmitRequest(sequence, cancellationToken);
        var tokens = new List<int>();
        double? firstTokenMs = null;
        bool accepting = maxTokens > 0;
        try
        {
            while (await handle.Tokens.WaitToReadAsync(cancellationToken).ConfigureAwait(false))
            {
                while (handle.Tokens.TryRead(out int token))
                {
                    if (!accepting)
                        continue;
                    firstTokenMs ??= watch.Elapsed.TotalMilliseconds;
                    tokens.Add(token);
                    if (onToken != null && !onToken(token))
                    {
                        accepting = false;
                        engine.Abort(sequence.RequestId);
                    }
                }
            }
            var completion = await handle.Completion.ConfigureAwait(false);
            if (maxTokens == 0)
                completion = PrefillOnly(completion);
            return new Result(tokens, completion, sequence,
                firstTokenMs ?? watch.Elapsed.TotalMilliseconds, watch.Elapsed.TotalMilliseconds);
        }
        finally
        {
            // A cancelled wait leaves the request running on the engine; it has to be
            // finished before the caller clears this request's media embeddings.
            if (!handle.Completion.IsCompleted)
                engine.Abort(sequence.RequestId);
            try
            {
                await handle.Completion.ConfigureAwait(false);
            }
            catch (Exception) when (cancellationToken.IsCancellationRequested)
            {
                // The cancellation is what propagates, not the aborted request's outcome.
            }
        }
    }

    /// <summary>Keep public system prefixes while isolating the next conversation's private tokens.</summary>
    public void StartNewConversation()
    {
        _scope = $"cli-{Guid.NewGuid():N}";
        CachedTokens = 0;
    }

    /// <summary>Discard all in-memory cache state, including public system prefixes.</summary>
    public void Reset()
    {
        ReleaseEngine();
        _model.ResetKVCache();
        StartNewConversation();
    }

    public void Dispose() => ReleaseEngine();

    private InferenceEngine Engine()
    {
        lock (_engineGate)
            return _engine ??= new InferenceEngine(_model, _config, _logger);
    }

    private void ReleaseEngine()
    {
        InferenceEngine engine;
        lock (_engineGate)
        {
            engine = _engine;
            _engine = null;
        }
        engine?.Dispose();
    }

    private static SequenceState CreateSequence(InferenceEngine engine, IReadOnlyList<int> prompt,
        int maxTokens, SamplingConfig sampling, string requestId, IReadOnlyList<PromptMediaSpan> mediaSpans,
        bool enablePrefixCache, int sharedPrefixTokens, string cacheScope, int priority)
    {
        // Preserve the CLI's zero-output prefill mode. The engine requires a positive
        // limit; its single sampled token is discarded and uses a separate sampler.
        return new SequenceState(requestId, prompt, Math.Max(1, maxTokens),
            engine.PoolStats.blockSize, maxTokens == 0 ? SamplingConfig.Greedy : sampling, mediaSpans: mediaSpans,
            cacheBreakpoints: enablePrefixCache ? null : Array.Empty<int>(),
            sharedPrefixTokens: enablePrefixCache ? sharedPrefixTokens : 0, cacheScope: cacheScope)
        {
            Priority = priority,
        };
    }

    private static InferenceCompletion PrefillOnly(InferenceCompletion completion) => new()
    {
        Status = completion.Status,
        FinishReason = completion.Status == SequenceStatus.FinishedAborted ? "aborted" : "max_tokens",
        PromptTokenCount = completion.PromptTokenCount,
        OutputTokenCount = 0,
        PrefixCacheReusedTokens = completion.PrefixCacheReusedTokens,
        FirstTokenAt = null,
        SubmittedAt = completion.SubmittedAt,
    };

    internal sealed record Result(List<int> Tokens, InferenceCompletion Completion,
        SequenceState Sequence, double PrefillMs, double TotalMs)
    {
        // PrefillMs is latency until the first delivered token, including scheduling
        // and sampling. The standalone --benchmark still measures model forwards only.
        public double DecodeMs => Math.Max(0, TotalMs - PrefillMs);
    }
}
