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
using System.Collections.Generic;
using System.Diagnostics;
using System.Diagnostics.Metrics;
using System.Linq;
using TensorSharp.Runtime.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

[CollectionDefinition(Name, DisableParallelization = true)]
public sealed class TelemetryGlobalStateCollection
{
    /// <summary>Tests that change process-wide telemetry switches or the environment.</summary>
    public const string Name = "Telemetry global state";
}

/// <summary>
/// One generation's span tree and metrics, observed through a real ActivityListener and
/// MeterListener. Each test uses its own model name, because other tests running in
/// parallel may record on the same sources.
/// </summary>
[Collection(TelemetryGlobalStateCollection.Name)]
public sealed class GenAiRequestTelemetryTests : IDisposable
{
    private readonly ConcurrentQueue<Activity> _stopped = new();
    private readonly ConcurrentQueue<(string Instrument, double Value, Dictionary<string, object?> Tags)> _measurements = new();
    private readonly ActivityListener _activities;
    private readonly MeterListener _meters;

    public GenAiRequestTelemetryTests()
    {
        _activities = new ActivityListener
        {
            ShouldListenTo = s => s.Name.StartsWith(TensorSharpTelemetry.NamePrefix, StringComparison.Ordinal),
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = a => _stopped.Enqueue(a),
        };
        ActivitySource.AddActivityListener(_activities);

        _meters = new MeterListener
        {
            InstrumentPublished = (instrument, listener) =>
            {
                if (instrument.Meter.Name == TensorSharpTelemetry.MeterName)
                    listener.EnableMeasurementEvents(instrument);
            },
        };
        _meters.SetMeasurementEventCallback<double>((i, v, tags, _) => Record(i, v, tags));
        _meters.SetMeasurementEventCallback<long>((i, v, tags, _) => Record(i, v, tags));
        _meters.SetMeasurementEventCallback<int>((i, v, tags, _) => Record(i, v, tags));
        _meters.Start();
    }

    private void Record(Instrument instrument, double value, ReadOnlySpan<KeyValuePair<string, object?>> tags)
    {
        var map = new Dictionary<string, object?>(StringComparer.Ordinal);
        foreach (var tag in tags) map[tag.Key] = tag.Value;
        _measurements.Enqueue((instrument.Name, value, map));
    }

    public void Dispose()
    {
        _activities.Dispose();
        _meters.Dispose();
        TensorSharpTelemetry.ConfigureCaptureMessageContent(null);
    }

    private static void Near(TimeSpan expected, TimeSpan actual) =>
        Assert.InRange(Math.Abs((expected - actual).TotalMilliseconds), 0, 1.0);

    private static void Near(DateTime expected, DateTime actual) =>
        Assert.InRange(Math.Abs((expected - actual).TotalMilliseconds), 0, 1.0);

    private List<(string Instrument, double Value, Dictionary<string, object?> Tags)> Measurements(string model, string instrument) =>
        _measurements.Where(m => m.Instrument == instrument && Equals(m.Tags.GetValueOrDefault(TelemetryAttributes.GenAiRequestModel), model)).ToList();

    private static string NewModel() => "test-model-" + Guid.NewGuid().ToString("N")[..8];

    [Fact]
    public void ACompletedChatHasItsSpanPhasesUsageAndMetrics()
    {
        string model = NewModel();
        string? previousProtocol = TelemetryContext.Protocol;
        TelemetryContext.Protocol = "openai";
        DateTime submitted, scheduled, first, end;
        try
        {
            using (var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationChat, model, "ggml_cuda"))
            {
                telemetry.SetSampling(256, 0.7, 0.9, 40);
                submitted = telemetry.StartedAtUtc.AddMilliseconds(5);
                scheduled = submitted.AddMilliseconds(20);
                first = scheduled.AddMilliseconds(100);
                end = first.AddMilliseconds(900);
                telemetry.RecordEnginePhases(submitted, scheduled, first, end,
                    prefillTicks: Stopwatch.Frequency / 10, decodeTicks: Stopwatch.Frequency / 2);
                telemetry.RecordUsage(inputTokens: 120, outputTokens: 10, cachedTokens: 100, "stop", first, end);
            }
        }
        finally
        {
            TelemetryContext.Protocol = previousProtocol;
        }

        var chat = Assert.Single(_stopped, a => a.DisplayName == $"chat {model}");
        Assert.Equal(ActivityKind.Internal, chat.Kind);
        Assert.Equal("chat", chat.GetTagItem(TelemetryAttributes.GenAiOperationName));
        Assert.Equal("openai", chat.GetTagItem(TelemetryAttributes.Protocol));
        Assert.Equal(256, chat.GetTagItem(TelemetryAttributes.GenAiRequestMaxTokens));
        Assert.Equal(120, chat.GetTagItem(TelemetryAttributes.GenAiUsageInputTokens));
        Assert.Equal(10, chat.GetTagItem(TelemetryAttributes.GenAiUsageOutputTokens));
        Assert.Equal(100, chat.GetTagItem(TelemetryAttributes.GenAiUsageCacheReadInputTokens));
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, chat.GetTagItem(TelemetryAttributes.Outcome));

        var children = _stopped.Where(a => a.ParentSpanId == chat.SpanId).ToDictionary(a => a.DisplayName);
        Assert.Equal(new[] { "inference.decode", "inference.prefill", "inference.queue" }, children.Keys.OrderBy(k => k));
        Near(submitted, children["inference.queue"].StartTimeUtc);
        Near(scheduled - submitted, children["inference.queue"].Duration);
        Near(first - scheduled, children["inference.prefill"].Duration);
        Near(end - first, children["inference.decode"].Duration);
        Assert.Equal(100.0, (double)children["inference.prefill"].GetTagItem(TelemetryAttributes.ComputePrefillMs)!, 1);

        var duration = Assert.Single(Measurements(model, "gen_ai.server.request.duration"));
        Assert.Equal(TelemetryAttributes.OutcomeCompleted, duration.Tags[TelemetryAttributes.Outcome]);
        Assert.False(duration.Tags.ContainsKey(TelemetryAttributes.ErrorType));
        Assert.Single(Measurements(model, "gen_ai.server.time_to_first_token"));
        var tpot = Assert.Single(Measurements(model, "gen_ai.server.time_per_output_token"));
        Assert.Equal(0.1, tpot.Value, 3); // 900 ms over the 9 tokens after the first
        var tokens = Measurements(model, "tensorsharp.tokens");
        Assert.Equal(120, tokens.Single(t => Equals(t.Tags[TelemetryAttributes.TokenType], "input")).Value);
        Assert.Equal(10, tokens.Single(t => Equals(t.Tags[TelemetryAttributes.TokenType], "output")).Value);
        Assert.Equal(100, Assert.Single(Measurements(model, "tensorsharp.tokens.cache_read")).Value);
    }

    [Fact]
    public void APreemptedSequenceNeverGetsANegativePrefill()
    {
        string model = NewModel();
        using (var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationChat, model, null))
        {
            DateTime submitted = DateTime.UtcNow;
            // First token before the (re-)admission timestamp: the split is unknowable.
            telemetry.RecordEnginePhases(submitted, submitted.AddMilliseconds(500), submitted.AddMilliseconds(200),
                submitted.AddMilliseconds(900), preemptions: 2);
        }

        var chat = Assert.Single(_stopped, a => a.DisplayName == $"chat {model}");
        Assert.Equal(2, chat.GetTagItem(TelemetryAttributes.Preemptions));
        var children = _stopped.Where(a => a.ParentSpanId == chat.SpanId).Select(a => a.DisplayName).ToList();
        Assert.Equal(new[] { "inference.queue" }, children);
    }

    [Fact]
    public void APhaseNeverStartsBeforeItsParent()
    {
        string model = NewModel();
        DateTime early = DateTime.UtcNow.AddMilliseconds(-50); // e.g. SubmittedAt stamped before the span started
        using (var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationChat, model, null))
            telemetry.RecordEnginePhases(early, null, null, DateTime.UtcNow.AddMilliseconds(10));

        var chat = Assert.Single(_stopped, a => a.DisplayName == $"chat {model}");
        var queue = Assert.Single(_stopped, a => a.ParentSpanId == chat.SpanId);
        Assert.True(queue.StartTimeUtc >= chat.StartTimeUtc, $"queue starts {queue.StartTimeUtc:O}, parent {chat.StartTimeUtc:O}");
    }

    [Fact]
    public void ARequestAbortedBeforeSchedulingHasOnlyAQueueSpan()
    {
        string model = NewModel();
        using (var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationChat, model, null))
        {
            telemetry.RecordEnginePhases(DateTime.UtcNow, null, null, DateTime.UtcNow.AddMilliseconds(30));
            telemetry.MarkCancelled();
        }

        var chat = Assert.Single(_stopped, a => a.DisplayName == $"chat {model}");
        Assert.Equal(TelemetryAttributes.OutcomeCancelled, chat.GetTagItem(TelemetryAttributes.Outcome));
        Assert.Equal(new[] { "inference.queue" }, _stopped.Where(a => a.ParentSpanId == chat.SpanId).Select(a => a.DisplayName));
    }

    [Fact]
    public void FailureRecordsTheTypeNeverTheMessageAndTimingIsOnlyForSuccess()
    {
        string model = NewModel();
        using (var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationChat, model, null))
        {
            telemetry.RecordUsage(10, 3, 0, null, telemetry.StartedAtUtc.AddMilliseconds(50));
            telemetry.MarkFailed(new InvalidOperationException("the prompt said: my secret"));
        }

        var chat = Assert.Single(_stopped, a => a.DisplayName == $"chat {model}");
        Assert.Equal(ActivityStatusCode.Error, chat.Status);
        Assert.Equal("InvalidOperationException", chat.StatusDescription);
        Assert.Equal("System.InvalidOperationException", chat.GetTagItem(TelemetryAttributes.ErrorType));
        Assert.DoesNotContain(chat.TagObjects, t => t.Value is string s && s.Contains("secret", StringComparison.Ordinal));

        var duration = Assert.Single(Measurements(model, "gen_ai.server.request.duration"));
        Assert.Equal("System.InvalidOperationException", duration.Tags[TelemetryAttributes.ErrorType]);
        Assert.Empty(Measurements(model, "gen_ai.server.time_to_first_token"));
        Assert.Empty(Measurements(model, "gen_ai.server.time_per_output_token"));
        // Tokens were still processed.
        Assert.Equal(2, Measurements(model, "tensorsharp.tokens").Count);
    }

    [Fact]
    public void CancellationIsNotAFailure()
    {
        string model = NewModel();
        using (var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationChat, model, null))
            telemetry.MarkFailed(new OperationCanceledException());

        var chat = Assert.Single(_stopped, a => a.DisplayName == $"chat {model}");
        Assert.NotEqual(ActivityStatusCode.Error, chat.Status);
        Assert.Equal(TelemetryAttributes.OutcomeCancelled, chat.GetTagItem(TelemetryAttributes.Outcome));
    }

    [Fact]
    public void DisposeRecordsTheDurationOnce()
    {
        string model = NewModel();
        var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationEmbeddings, model, null);
        telemetry.Dispose();
        telemetry.Dispose();
        Assert.Single(Measurements(model, "gen_ai.server.request.duration"));
    }

    [Fact]
    public void ContentReachesTheSpanOnlyWithCaptureOn()
    {
        string model = NewModel();
        int built = 0;
        TensorSharpTelemetry.ConfigureCaptureMessageContent(false);
        using (var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationChat, model, null))
            telemetry.SetInputMessages(() => { built++; return "[]"; });
        Assert.Equal(0, built);
        Assert.Null(Assert.Single(_stopped, a => a.DisplayName == $"chat {model}").GetTagItem(TelemetryAttributes.GenAiInputMessages));

        string model2 = NewModel();
        TensorSharpTelemetry.ConfigureCaptureMessageContent(true);
        using (var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationChat, model2, null))
            telemetry.SetInputMessages(() => "[{\"role\":\"user\",\"content\":\"hi\"}]");
        Assert.NotNull(Assert.Single(_stopped, a => a.DisplayName == $"chat {model2}").GetTagItem(TelemetryAttributes.GenAiInputMessages));
    }

    [Fact]
    public void InternalCallsAndRoundsAreMarked()
    {
        string model = NewModel();
        string? previousOrigin = TelemetryContext.Origin;
        int? previousRound = TelemetryContext.Round;
        try
        {
            TelemetryContext.Origin = "knowledge-rerank";
            TelemetryContext.Round = 3;
            using var telemetry = GenAiRequestTelemetry.Start(TelemetryAttributes.OperationChat, model, null);
        }
        finally
        {
            TelemetryContext.Origin = previousOrigin;
            TelemetryContext.Round = previousRound;
        }

        var chat = Assert.Single(_stopped, a => a.DisplayName == $"chat {model}");
        Assert.Equal("knowledge-rerank", chat.GetTagItem(TelemetryAttributes.Origin));
        Assert.Equal(true, chat.GetTagItem(TelemetryAttributes.Internal));
        Assert.Equal(3, chat.GetTagItem(TelemetryAttributes.Round));
        Assert.Equal("knowledge-rerank", Assert.Single(Measurements(model, "gen_ai.server.request.duration")).Tags[TelemetryAttributes.Origin]);
    }

    [Fact]
    public void ModelNamesNeverCarryAPath()
    {
        // Whatever OS the test runs on: a Windows path read on Linux, where '\' is not a
        // separator, must not keep its directories either.
        Assert.Equal("Qwen3.5-9B-Q8_0", GenAiRequestTelemetry.ModelNameFromPath(@"C:\models\Qwen3.5-9B-Q8_0.gguf"));
        Assert.Equal("Qwen3.5-9B-Q8_0", GenAiRequestTelemetry.ModelNameFromPath(@"\\server\share\users\me\Qwen3.5-9B-Q8_0.gguf"));
        Assert.Equal("gemma", GenAiRequestTelemetry.ModelNameFromPath("/srv/models/gemma.gguf"));
        Assert.Equal("gemma", GenAiRequestTelemetry.ModelNameFromPath(@"C:/models\mixed/gemma.gguf"));
        Assert.Equal("unknown", GenAiRequestTelemetry.ModelNameFromPath("  "));
        Assert.Equal("unknown", GenAiRequestTelemetry.ModelNameFromPath(@"C:\models\"));
    }
}

/// <summary>Without a listener nothing is created, so a child can never become a stray root.</summary>
public class GenAiRequestTelemetryWithoutListenerTests
{
    [Fact]
    public void NothingIsRecordedWithoutAListener()
    {
        // A source nobody listens to: use a dedicated name so a listener from another test
        // (which only subscribes to TensorSharp.*) cannot interfere.
        using var source = new ActivitySource("Unlistened." + Guid.NewGuid().ToString("N"));
        Assert.Null(source.StartActivity("x"));
    }
}
