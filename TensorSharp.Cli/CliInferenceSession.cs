// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Cli;

/// <summary>
/// Synchronous console facade over the same inference engine used by the server.
/// The engine owns prefill, radix prefix reuse, decoding and speculative state.
/// </summary>
internal sealed class CliInferenceSession : IDisposable
{
    private readonly IModelArchitecture _model;
    private readonly SchedulerConfig _config;
    private readonly ILogger _logger;
    private string _scope = $"cli-{Guid.NewGuid():N}";
    private InferenceEngine _engine;

    public CliInferenceSession(IModelArchitecture model, SchedulerConfig config, ILogger logger)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _logger = logger;
    }

    /// <summary>Tokens computed by the last request; not the radix tree's total retained size.</summary>
    public int CachedTokens { get; private set; }

    public Result Generate(IReadOnlyList<int> prompt, int maxTokens, SamplingConfig sampling,
        Func<int, bool> onToken = null, CancellationToken cancellationToken = default,
        string requestId = null, IReadOnlyList<PromptMediaSpan> mediaSpans = null,
        bool enablePrefixCache = true, int sharedPrefixTokens = 0)
    {
        ArgumentOutOfRangeException.ThrowIfNegative(maxTokens);
        cancellationToken.ThrowIfCancellationRequested();
        _engine ??= new InferenceEngine(_model, _config, _logger);
        // Preserve the CLI's zero-output prefill mode. The engine requires a positive
        // limit; its single sampled token is discarded and uses a separate sampler.
        var sequence = new SequenceState(requestId, prompt, Math.Max(1, maxTokens),
            _engine.PoolStats.blockSize, maxTokens == 0 ? SamplingConfig.Greedy : sampling, mediaSpans: mediaSpans,
            cacheBreakpoints: enablePrefixCache ? null : Array.Empty<int>(),
            sharedPrefixTokens: enablePrefixCache ? sharedPrefixTokens : 0, cacheScope: _scope);
        var watch = Stopwatch.StartNew();
        var handle = _engine.SubmitRequest(sequence, cancellationToken);
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
                        _engine.Abort(sequence.RequestId);
                    }
                }
            }
            var completion = handle.Completion.GetAwaiter().GetResult();
            if (maxTokens == 0)
                completion = new InferenceCompletion
                {
                    Status = completion.Status,
                    FinishReason = completion.Status == SequenceStatus.FinishedAborted ? "aborted" : "max_tokens",
                    PromptTokenCount = completion.PromptTokenCount,
                    OutputTokenCount = 0,
                    PrefixCacheReusedTokens = completion.PrefixCacheReusedTokens,
                    FirstTokenAt = null,
                    SubmittedAt = completion.SubmittedAt,
                };
            CachedTokens = sequence.NumComputedTokens;
            return new Result(tokens, completion, sequence,
                firstTokenMs ?? watch.Elapsed.TotalMilliseconds, watch.Elapsed.TotalMilliseconds);
        }
        finally
        {
            // Even an output callback failure must finish the worker's request before
            // the caller clears media embeddings, swaps models, or starts another turn.
            if (!handle.Completion.IsCompleted)
                _engine.Abort(sequence.RequestId);
            handle.Completion.GetAwaiter().GetResult();
            CachedTokens = sequence.NumComputedTokens;
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
        _engine?.Dispose();
        _engine = null;
        _model.ResetKVCache();
        StartNewConversation();
    }

    public void Dispose()
    {
        _engine?.Dispose();
        _engine = null;
    }

    internal sealed record Result(List<int> Tokens, InferenceCompletion Completion,
        SequenceState Sequence, double PrefillMs, double TotalMs)
    {
        // PrefillMs is latency until the first delivered token, including scheduling
        // and sampling. The standalone --benchmark still measures model forwards only.
        public double DecodeMs => Math.Max(0, TotalMs - PrefillMs);
    }
}
