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
using System.Collections.Concurrent;
using System.Diagnostics;
using System.Linq;
using System.Threading;
using InferenceWeb.Tests.PrefixCache.Fakes;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp;
using TensorSharp.Cli;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// The CLI's own chat spans (<see cref="CliInferenceSession"/>), observed through a real
/// ActivityListener. Each call has its own engine request id, which the span carries, so
/// spans other tests record at the same time are left out.
/// </summary>
[Collection(TelemetryGlobalStateCollection.Name)]
public sealed class CliTelemetryTests : IDisposable
{
    private readonly ConcurrentQueue<Activity> _stopped = new();
    private readonly ActivityListener _activities;

    public CliTelemetryTests()
    {
        _activities = new ActivityListener
        {
            ShouldListenTo = s => s.Name.StartsWith(TensorSharpTelemetry.NamePrefix, StringComparison.Ordinal),
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = a => _stopped.Enqueue(a),
        };
        ActivitySource.AddActivityListener(_activities);
    }

    public void Dispose() => _activities.Dispose();

    private static SchedulerConfig Config => new()
    {
        BlockSize = 16,
        NumBlocks = 64,
        MaxNumRunningSequences = 1,
        MaxNumBatchedTokens = 128,
        SoloPrefillChunkSize = 128,
        StopRepetition = false,
    };

    private Activity ChatSpan(string requestId) =>
        Assert.Single(_stopped, a => Equals(a.GetTagItem(TelemetryAttributes.SequenceId), requestId));

    // The engine runs ahead of the CLI reading its tokens, so the abort a stop asks for can
    // arrive before or after the engine finishes on its own (its token limit, here): both are
    // the same stop to the caller.
    private static readonly SequenceStatus[] StopEndings = { SequenceStatus.FinishedAborted, SequenceStatus.FinishedLengthCapped };

    [Fact]
    public void AStopTheCliMakesItselfIsRecordedAsAStopSequence()
    {
        // The CLI's onToken returns false for a stop sequence or a tool call that ends the turn;
        // the engine carries that out as an abort, but it is a normal finish, as on the server.
        using var model = OracleFakes.P();
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        string requestId = "cli-stop-" + Guid.NewGuid().ToString("N");

        var result = session.Generate(Enumerable.Range(1, 32).ToArray(), 64, SamplingConfig.Greedy,
            _ => false, timeout.Token, requestId: requestId);

        Assert.Single(result.Tokens);
        Assert.Contains(result.Completion.Status, StopEndings);
        var span = ChatSpan(requestId);
        Assert.Equal(new[] { "stop_sequence" }, Assert.IsType<string[]>(span.GetTagItem(TelemetryAttributes.GenAiResponseFinishReasons)));
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, span.GetTagItem(TelemetryAttributes.Outcome));
    }

    [Fact]
    public void AStopThatArrivesAfterTheEngineFinishedIsStillAStopSequence()
    {
        // With a one-token limit the engine finishes in the step that publishes the token, and
        // the caller stops on that token only afterwards: the engine reports its length cap,
        // which recorded max_tokens for a reply that ended where the caller stopped it.
        using var model = OracleFakes.P();
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        string requestId = "cli-late-stop-" + Guid.NewGuid().ToString("N");

        var result = session.Generate(Enumerable.Range(1, 32).ToArray(), 1, SamplingConfig.Greedy,
            _ => { Thread.Sleep(100); return false; }, timeout.Token, requestId: requestId);

        Assert.Single(result.Tokens);
        Assert.Contains(result.Completion.Status, StopEndings);
        var span = ChatSpan(requestId);
        Assert.Equal(new[] { "stop_sequence" }, Assert.IsType<string[]>(span.GetTagItem(TelemetryAttributes.GenAiResponseFinishReasons)));
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, span.GetTagItem(TelemetryAttributes.Outcome));
    }

    [Fact]
    public void ARunToItsLimitKeepsTheEnginesFinishReason()
    {
        using var model = OracleFakes.P();
        using var session = new CliInferenceSession(model, Config, NullLogger.Instance);
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(10));
        string requestId = "cli-limit-" + Guid.NewGuid().ToString("N");

        var result = session.Generate(Enumerable.Range(1, 32).ToArray(), 3, SamplingConfig.Greedy,
            cancellationToken: timeout.Token, requestId: requestId);

        var reasons = Assert.IsType<string[]>(ChatSpan(requestId).GetTagItem(TelemetryAttributes.GenAiResponseFinishReasons));
        Assert.Equal(new[] { result.Completion.FinishReason }, reasons);
        Assert.NotEqual("stop_sequence", reasons[0]);
    }
}
