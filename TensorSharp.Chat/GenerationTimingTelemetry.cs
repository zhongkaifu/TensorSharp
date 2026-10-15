// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
#nullable enable
using System;
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Scheduling;

namespace TensorSharp.Server;

/// <summary>One completed generation's engine timing, correlated with its
/// HTTP logging scope. Forward durations exclude queueing and preparation;
/// shared batch time is apportioned among sequences, not request wall time.</summary>
internal static class GenerationTimingTelemetry
{
    internal readonly record struct Durations(long? PromptNs, long? EvalNs, long TotalNs);

    internal static Durations Log(ILogger logger, string requestId, string? sessionId,
        InferenceCompletion? completion, int promptTokens, int reusedTokens,
        int generatedTokens, long elapsedTicks, string finishReason)
    {
        var durations = new Durations(
            completion is null ? null : ToNanos(completion.PrefillElapsedTicks),
            completion is null ? null : ToNanos(completion.DecodeElapsedTicks),
            ToNanos(elapsedTicks));
        // Keep null for unavailable completion, including cancellation before
        // the engine published it. A measured zero remains a distinct value.
        // GeneratedTokens is the recorded pipeline output; the engine count
        // is retained separately because local stops can truncate streaming.
        logger.LogInformation(
            "generation.timings internal={InternalRequestId} session={SessionId} " +
            "promptTokens={PromptTokens} kvReused={KvReusedTokens} generatedTokens={GeneratedTokens} " +
            "engineOutputTokens={EngineOutputTokens} promptNs={PromptNs} evalNs={EvalNs} totalNs={TotalNs} " +
            "finishReason={FinishReason} engineCompletionAvailable={EngineCompletionAvailable} timingBasis={TimingBasis}",
            requestId, sessionId ?? "(none)", promptTokens, reusedTokens, generatedTokens,
            completion?.OutputTokenCount, durations.PromptNs, durations.EvalNs, durations.TotalNs,
            finishReason, completion is not null, "attributed_forward");
        return durations;
    }

    // Integer division only after scaling avoids a platform-frequency ratio
    // being rounded before it is applied to the measured duration.
    private static long ToNanos(long ticks)
        => checked((long)((Int128)ticks * 1_000_000_000L / Stopwatch.Frequency));
}
