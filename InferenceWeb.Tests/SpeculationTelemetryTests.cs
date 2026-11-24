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
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Speculative;
using TensorSharp.Runtime.Telemetry;
using Xunit;

namespace InferenceWeb.Tests;

public partial class SpeculativeExecutionTests
{
    /// <summary>Every line logged, with the trace and span that were current when it was written.</summary>
    private sealed class TracedLines : ILogger
    {
        public ConcurrentQueue<(string Line, ActivityTraceId Trace, ActivitySpanId Span)> Lines { get; } = new();
        public IDisposable BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception,
            Func<TState, Exception, string> formatter) =>
            Lines.Enqueue((formatter(state, exception), Activity.Current?.TraceId ?? default, Activity.Current?.SpanId ?? default));
    }

    private static ActivityListener ListenToInference(Action<Activity> stopped) => new()
    {
        ShouldListenTo = s => s.Name == TensorSharpTelemetry.InferenceSourceName,
        Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
        ActivityStopped = stopped,
    };

    [Fact]
    public void ASpeculativeRequestHasASpeculationSpanUnderItsRequestThatHoldsItsStatsLine()
    {
        const string expected = SpeculatorRegistry.DraftHead;   // the fake's per-token head, under auto
        var request = new ActivityContext(ActivityTraceId.CreateRandom(), ActivitySpanId.CreateRandom(), ActivityTraceFlags.Recorded);
        var spans = new ConcurrentQueue<Activity>();
        using var listener = ListenToInference(a => { if (a.TraceId == request.TraceId) spans.Enqueue(a); });
        ActivitySource.AddActivityListener(listener);
        var log = new TracedLines();

        var seq = RunEngineRequest(new FakeSpeculativeModel(), promptLen: 5, maxNewTokens: 24, specEnabled: true,
            logger: log, telemetryContext: request);

        var stats = seq.SpecStats;
        Assert.True(stats.VerifySteps > 0);
        Assert.Equal(expected, stats.Speculator);
        Assert.True(stats.DraftWindow > 0);

        // One span, a child of the request's, from its first token.
        var span = Assert.Single(spans);
        Assert.Equal(SpeculationTelemetry.SpanName, span.DisplayName);
        Assert.Equal(request.SpanId, span.ParentSpanId);
        Assert.Equal(seq.FirstTokenAt, span.StartTimeUtc);
        Assert.True(span.Duration > TimeSpan.Zero);

        Assert.Equal(expected, span.GetTagItem(TelemetryAttributes.Speculator));
        Assert.Equal(stats.DraftWindow, span.GetTagItem(SpeculationTelemetry.DraftWindow));
        Assert.Equal(stats.TokensDrafted, span.GetTagItem(SpeculationTelemetry.TokensDrafted));
        Assert.Equal(stats.TokensAccepted, span.GetTagItem(SpeculationTelemetry.TokensAccepted));
        Assert.Equal(Math.Round(stats.AcceptanceRate, 4), span.GetTagItem(SpeculationTelemetry.Acceptance));
        Assert.Equal(stats.VerifySteps, span.GetTagItem(SpeculationTelemetry.VerifySteps));
        Assert.Equal(stats.PlainSteps, span.GetTagItem(SpeculationTelemetry.PlainSteps));
        Assert.Equal(stats.RollbackSteps, span.GetTagItem(SpeculationTelemetry.RollbackSteps));
        Assert.Equal((long)stats.GovernorParkedSteps, Convert.ToInt64(span.GetTagItem(SpeculationTelemetry.ParkedSteps)));
        Assert.Equal(Math.Round(stats.DraftMs, 1), span.GetTagItem(SpeculationTelemetry.DraftMs));
        Assert.Equal(Math.Round(stats.VerifyMs, 1), span.GetTagItem(SpeculationTelemetry.VerifyMs));
        Assert.Equal(Math.Round(stats.CommitMs, 1), span.GetTagItem(SpeculationTelemetry.CommitMs));
        Assert.Equal(Math.Round(stats.CatchUpMs, 1), span.GetTagItem(SpeculationTelemetry.CatchUpMs));
        Assert.Equal(Math.Round(stats.PlainMs, 1), span.GetTagItem(SpeculationTelemetry.PlainMs));
        Assert.Equal(stats.GovernorWins, span.GetTagItem(SpeculationTelemetry.GovernorWins));
        Assert.Equal(stats.GovernorLosses, span.GetTagItem(SpeculationTelemetry.GovernorLosses));

        // The stats line was written with the span current, so it is in the request's trace, and
        // the sections the eng/ scripts read are unchanged before the speculator's own.
        var (line, trace, spanId) = Assert.Single(log.Lines, l => l.Line.StartsWith("Speculative decoding stats for ", StringComparison.Ordinal));
        Assert.Equal((request.TraceId, span.SpanId), (trace, spanId));
        Assert.Matches(@"^Speculative decoding stats for \S+: drafted=\d+ accepted=\d+ acceptance=", line);
        Assert.EndsWith($"| governor plain={stats.PlainMsPerToken:F1}ms/tok spec={stats.SpecMsPerToken:F1}ms/tok "
            + $"wins={stats.GovernorWins} losses={stats.GovernorLosses} parked={stats.GovernorParkedSteps} "
            + $"| speculator name={expected} window={stats.DraftWindow}", line);
    }

    [Fact]
    public void ARequestWithoutASpanGetsNoSpeculationSpanAndStillItsStatsLine()
    {
        var roots = new ConcurrentQueue<Activity>();
        using var listener = ListenToInference(a =>
        {
            if (a.DisplayName == SpeculationTelemetry.SpanName && a.ParentSpanId == default)
                roots.Enqueue(a);
        });
        ActivitySource.AddActivityListener(listener);
        var log = new TracedLines();

        var seq = RunEngineRequest(new FakeSpeculativeModel(), promptLen: 5, maxNewTokens: 24, specEnabled: true, logger: log);

        Assert.True(seq.SpecStats.VerifySteps > 0);
        // A speculation span is never the root of a trace of its own.
        Assert.Empty(roots);
        var (_, trace, _) = Assert.Single(log.Lines, l => l.Line.StartsWith("Speculative decoding stats for ", StringComparison.Ordinal));
        Assert.Equal(default, trace);
    }

    [Fact]
    public void TheSpeculationMetricsNameTheSpeculator()
    {
        var seen = new ConcurrentQueue<(string Instrument, object Speculator)>();
        using var meters = new MeterListener
        {
            InstrumentPublished = (instrument, l) =>
            {
                if (instrument.Meter.Name == TensorSharpTelemetry.MeterName
                    && instrument.Name.StartsWith("tensorsharp.speculative.", StringComparison.Ordinal))
                    l.EnableMeasurementEvents(instrument);
            },
        };
        void Record<T>(Instrument instrument, T value, ReadOnlySpan<KeyValuePair<string, object>> tags, object state)
        {
            object speculator = null;
            foreach (var tag in tags)
                if (tag.Key == TelemetryAttributes.Speculator) speculator = tag.Value;
            seen.Enqueue((instrument.Name, speculator));
        }
        meters.SetMeasurementEventCallback<long>(Record);
        meters.SetMeasurementEventCallback<int>(Record);
        meters.Start();

        RunEngineRequest(new FakeSpeculativeModel(), promptLen: 5, maxNewTokens: 24, specEnabled: true);
        meters.Dispose();

        foreach (string instrument in new[] { "tensorsharp.speculative.tokens.drafted", "tensorsharp.speculative.tokens.accepted",
                     "tensorsharp.speculative.accepted_length" })
            Assert.Contains(seen, m => m.Instrument == instrument && Equals(m.Speculator, SpeculatorRegistry.DraftHead));
        Assert.DoesNotContain(seen, m => m.Speculator is null);
    }
}
