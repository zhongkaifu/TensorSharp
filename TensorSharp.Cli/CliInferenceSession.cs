// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
// Licensed under the BSD-3-Clause license in the repository root.
using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Threading;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Telemetry;

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
    // What every request's span and metrics carry, read once: the model cannot change under a session.
    private readonly string _telemetryModel;
    private readonly string _telemetryBackend;

    // error.type of a prompt the engine refused because it leaves no room in the context
    // window, the same code the server's pipeline reports for it.
    private const string RefusedContextOverflow = "context_overflow";

    // The finish reason the server's pipeline records for a stop sequence it enforced.
    private const string StopSequenceFinishReason = "stop_sequence";

    public CliInferenceSession(IModelArchitecture model, SchedulerConfig config, ILogger logger)
    {
        _model = model ?? throw new ArgumentNullException(nameof(model));
        _config = config ?? throw new ArgumentNullException(nameof(config));
        _logger = logger;
        _telemetryModel = GenAiRequestTelemetry.ModelNameFromPath(
            model is ModelBase loaded && loaded.ModelFilePaths is { Count: > 0 } files ? files[0] : null);
        _telemetryBackend = CliTelemetry.BackendOf(model);
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
        // Ask for no more than the room the prompt leaves in the context window, as the
        // server does (ChatGenerationPipeline.ClampGenerationReserve): the engine would
        // stop the reply there anyway, and an honest limit keeps its K/V reservation and
        // per-slot admission (DeepSeek V4.1) from refusing a prompt that fits. A prompt
        // that leaves no room is left for the engine to refuse with
        // ContextLengthExceededException, which the CLI reports in one line.
        int replyLimit = Math.Max(1, maxTokens);
        int context = _model.MaxContextLength;
        if (context > 0 && prompt.Count < context && context - prompt.Count < replyLimit)
        {
            replyLimit = context - prompt.Count;
            if (maxTokens > 0)
                _logger?.LogInformation(
                    "cli.inference reply limited to {Room} of {Requested} tokens: the prompt is {PromptTokens} tokens of a " +
                    "{Context}-token context window (MAX_CONTEXT)", replyLimit, maxTokens, prompt.Count, context);
        }
        var sequence = new SequenceState(requestId, prompt, replyLimit,
            _engine.PoolStats.blockSize, maxTokens == 0 ? SamplingConfig.Greedy : sampling, mediaSpans: mediaSpans,
            cacheBreakpoints: enablePrefixCache ? null : Array.Empty<int>(),
            sharedPrefixTokens: enablePrefixCache ? sharedPrefixTokens : 0, cacheScope: _scope);

        // One chat span and the request metrics per call, recorded the way the server's
        // pipeline records them: the CLI submits to an engine of its own, so nothing else
        // would. The engine runs on its worker thread, so its phases are built from the
        // sequence's timestamps once it has finished with the request (finally, below).
        using GenAiRequestTelemetry telemetry = GenAiRequestTelemetry.Start(
            TelemetryAttributes.OperationChat, _telemetryModel, _telemetryBackend);
        if (telemetry.IsRecording)
        {
            SamplingConfig requested = maxTokens == 0 ? SamplingConfig.Greedy : sampling ?? SamplingConfig.Greedy;
            telemetry.SetSampling(maxTokens, requested.Temperature, requested.TopP, requested.TopK, requested.Seed);
        }
        // The engine's id for the sequence, so its worker-thread log lines (which carry it and
        // no trace id) join to this span.
        telemetry.SetTag(TelemetryAttributes.SequenceId, sequence.RequestId);
        // Lets the engine link its optional per-step spans to this call. Never a parent
        // there: the engine's thread is not this one.
        sequence.TelemetryContext = telemetry.Context;

        var watch = Stopwatch.StartNew();
        InferenceRequestHandle handle;
        try
        {
            handle = _engine.SubmitRequest(sequence, cancellationToken);
        }
        catch (Exception ex)
        {
            RecordFailure(telemetry, ex);
            throw;
        }
        var tokens = new List<int>();
        double? firstTokenMs = null;
        bool accepting = maxTokens > 0;
        // Set when the caller's onToken ended the generation (a stop sequence, a tool call that
        // ends the turn): the engine then reports the sequence as aborted.
        bool stoppedByCaller = false;
        InferenceCompletion completion = null;
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
                        stoppedByCaller = true;
                        _engine.Abort(sequence.RequestId);
                    }
                }
            }
            completion = handle.Completion.GetAwaiter().GetResult();
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
            // A stop the CLI asked for (a stop sequence, a turn budget) is a normal finish;
            // only the caller's token counts as cancelled.
            if (cancellationToken.IsCancellationRequested)
                telemetry.MarkCancelled();
            CachedTokens = sequence.NumComputedTokens;
            return new Result(tokens, completion, sequence,
                firstTokenMs ?? watch.Elapsed.TotalMilliseconds, watch.Elapsed.TotalMilliseconds);
        }
        catch (Exception ex)
        {
            RecordFailure(telemetry, ex);
            throw;
        }
        finally
        {
            // Even an output callback failure must finish the worker's request before
            // the caller clears media embeddings, swaps models, or starts another turn.
            if (!handle.Completion.IsCompleted)
                _engine.Abort(sequence.RequestId);
            try
            {
                handle.Completion.GetAwaiter().GetResult();
                CachedTokens = sequence.NumComputedTokens;
            }
            finally
            {
                // Every exit, once the engine is done with the sequence.
                RecordEngineWork(telemetry, sequence, handle, completion, tokens.Count, stoppedByCaller);
            }
        }
    }

    /// <summary>
    /// The engine's phases and the token usage of one call, from the scheduler's timestamps.
    /// The compute times come from the engine's own completion (the zero-output rewrite above
    /// drops them); the finish reason from what the call reported, which is null when it threw,
    /// so none is claimed for a failed call.
    /// </summary>
    private static void RecordEngineWork(GenAiRequestTelemetry telemetry, SequenceState sequence,
        InferenceRequestHandle handle, InferenceCompletion reported, int deliveredTokens, bool stoppedByCaller)
    {
        DateTime endUtc = DateTime.UtcNow;
        // A stop the CLI made itself is a stop sequence, as the server's pipeline records it,
        // not the abort the engine used to carry it out.
        string finishReason = reported?.FinishReason;
        if (stoppedByCaller && finishReason == "aborted")
            finishReason = StopSequenceFinishReason;
        InferenceCompletion engine = handle.Completion.IsCompletedSuccessfully ? handle.Completion.Result : null;
        telemetry.RecordEnginePhases(sequence.SubmittedAt, sequence.FirstScheduledAt, sequence.FirstTokenAt, endUtc,
            engine?.PrefillElapsedTicks ?? 0, engine?.DecodeElapsedTicks ?? 0, sequence.PreemptionCount);
        telemetry.RecordUsage(engine?.PromptTokenCount ?? sequence.PromptTokens.Count, deliveredTokens,
            engine?.PrefixCacheReusedTokens ?? sequence.PrefixCacheReusedTokens, finishReason,
            sequence.FirstTokenAt, endUtc);
    }

    /// <summary>
    /// The outcome of a call that threw. A prompt with no room left in the context window is
    /// the operator's to fix (the CLI reports it as a configuration error), not a fault; a
    /// cancellation counts as cancelled (<see cref="GenAiRequestTelemetry.MarkFailed"/>).
    /// </summary>
    private static void RecordFailure(GenAiRequestTelemetry telemetry, Exception exception)
    {
        if (exception is ContextLengthExceededException)
            telemetry.MarkRefused(RefusedContextOverflow);
        else
            telemetry.MarkFailed(exception);
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

    /// <summary>
    /// End of the run for this engine: export the telemetry once more, then dispose it. The
    /// engine and KV-cache gauges read only live engines, so a run shorter than one export
    /// interval would otherwise never report them. Use <see cref="Discard"/> for an engine
    /// dropped in the middle of a session.
    /// </summary>
    public void Dispose()
    {
        if (_engine is null)
            return;
        Program.FlushTelemetryBeforeEngineDispose();
        Discard();
    }

    /// <summary>
    /// Dispose the engine without the flush, for a reset or a model switch in the middle of an
    /// interactive session: the session goes on, the periodic export reports the next engine,
    /// and the prompt must not wait on a network export.
    /// </summary>
    public void Discard()
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
