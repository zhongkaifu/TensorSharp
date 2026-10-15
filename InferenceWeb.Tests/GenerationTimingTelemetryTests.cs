// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Diagnostics;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Server;

namespace InferenceWeb.Tests;

public sealed class GenerationTimingTelemetryTests
{
    [Fact]
    public void CompletionPreservesForwardTimingsCountsAndRequestCorrelation()
    {
        var logger = new RecordingLogger();
        var completion = new InferenceCompletion
        {
            PrefillElapsedTicks = Stopwatch.Frequency / 4,
            DecodeElapsedTicks = Stopwatch.Frequency / 2,
            OutputTokenCount = 19,
        };
        using var scope = logger.BeginScope(new Dictionary<string, object> { ["RequestId"] = "http-7" });
        var timing = GenerationTimingTelemetry.Log(logger, "internal-9", "session-3", completion,
            120, 96, 17, Stopwatch.Frequency * 3, "stop_sequence");
        Assert.Equal(250_000_000L, timing.PromptNs);
        Assert.Equal(500_000_000L, timing.EvalNs);
        Assert.Equal(3_000_000_000L, timing.TotalNs);
        var row = Assert.Single(logger.Rows);
        Assert.Equal("http-7", row["RequestId"]);
        Assert.Equal("internal-9", row["InternalRequestId"]);
        Assert.Equal("session-3", row["SessionId"]);
        Assert.Equal(120, row["PromptTokens"]);
        Assert.Equal(96, row["KvReusedTokens"]);
        Assert.Equal(17, row["GeneratedTokens"]);
        Assert.Equal(19, row["EngineOutputTokens"]);
        Assert.Equal("stop_sequence", row["FinishReason"]);
        Assert.Equal(true, row["EngineCompletionAvailable"]);
        Assert.Equal("attributed_forward", row["TimingBasis"]);
        Assert.StartsWith("generation.timings ", (string)row["{OriginalFormat}"]);
    }

    [Fact]
    public void MissingCancelledCompletionIsUnavailableRatherThanMeasuredZero()
    {
        var logger = new RecordingLogger();
        var timing = GenerationTimingTelemetry.Log(logger, "cancelled", null, null,
            40, 0, 0, Stopwatch.Frequency, "cancelled");
        var row = Assert.Single(logger.Rows);
        Assert.Null(timing.PromptNs);
        Assert.Null(timing.EvalNs);
        Assert.Null(row["PromptNs"]);
        Assert.Null(row["EvalNs"]);
        Assert.Null(row["EngineOutputTokens"]);
        Assert.Equal(false, row["EngineCompletionAvailable"]);
        Assert.Equal("cancelled", row["FinishReason"]);
        Assert.Equal("(none)", row["SessionId"]);
        Assert.Equal(1_000_000_000L, row["TotalNs"]);
    }

    [Fact]
    public void MeasuredZeroAndCancelledPublishedCompletionRemainAvailable()
    {
        var logger = new RecordingLogger();
        var timing = GenerationTimingTelemetry.Log(logger, "finished", "session",
            new InferenceCompletion { DecodeElapsedTicks = 1, OutputTokenCount = 2 },
            20, 20, 2, Stopwatch.Frequency, "cancelled");
        var row = Assert.Single(logger.Rows);
        Assert.Equal(0L, timing.PromptNs);
        Assert.Equal((long)((Int128)1_000_000_000L / Stopwatch.Frequency), timing.EvalNs);
        Assert.Equal(true, row["EngineCompletionAvailable"]);
        Assert.Equal(2, row["EngineOutputTokens"]);
        Assert.Equal("cancelled", row["FinishReason"]);
    }

    private sealed class RecordingLogger : ILogger
    {
        public List<Dictionary<string, object?>> Rows { get; } = [];
        private Dictionary<string, object?> _scope = [];
        public IDisposable BeginScope<TState>(TState state) where TState : notnull
        {
            var before = _scope;
            _scope = ((IEnumerable<KeyValuePair<string, object>>)state).ToDictionary(p => p.Key, p => (object?)p.Value);
            return new Scope(() => _scope = before);
        }
        public bool IsEnabled(LogLevel level) => true;
        public void Log<TState>(LogLevel level, EventId id, TState state, Exception? error,
            Func<TState, Exception?, string> formatter)
        {
            Assert.Equal(LogLevel.Information, level);
            var row = new Dictionary<string, object?>(_scope);
            foreach (var pair in (IEnumerable<KeyValuePair<string, object?>>)state!) row.Add(pair.Key, pair.Value);
            Rows.Add(row);
        }
        private sealed class Scope(Action dispose) : IDisposable { public void Dispose() => dispose(); }
    }
}
