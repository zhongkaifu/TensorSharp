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
using System.Threading;
using System.Threading.Tasks;
using Microsoft.Extensions.Logging.Abstractions;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Runtime.Telemetry;

namespace InferenceWeb.Tests.Telemetry;

/// <summary>
/// What the engine and the scheduler keep for a request's trace (when it finished, how often
/// it was preempted, when it was first admitted) and the engine metrics that must survive a
/// model swap. Driven with a deterministic stub model, no weights. Each test that reads a
/// metric gives its engine an architecture name of its own, so it can run beside other tests.
/// </summary>
public sealed class EngineTelemetryTests
{
    private const string StepsCounter = "tensorsharp.engine.steps";
    private const string ForwardTimeCounter = "tensorsharp.engine.forward.time";
    private const string RunningGauge = "tensorsharp.engine.requests.running";
    private const string StepDuration = "tensorsharp.engine.step.duration";
    private const string BatchSize = "tensorsharp.engine.batch.size";

    private static readonly string[] StepKinds = { "prefill", "decode", "mixed" };

    // ---------------------------------------------------------------- completion times

    [Fact]
    public async Task ACompletionCarriesTheTimeTheEngineFinishedTheRequest()
    {
        using var engine = EngineTestSupport.NewEngine(EngineTestSupport.NewArchitecture());
        var seq = EngineTestSupport.NewSequence("finished-at", promptLen: 6, maxNew: 4);

        var completion = await EngineTestSupport.RunToCompletion(engine, seq);
        DateTime after = DateTime.UtcNow;

        Assert.Equal(SequenceStatus.FinishedLengthCapped, completion.Status);
        Assert.True(completion.FinishedAt.HasValue, "A completion from the engine has an end time.");
        DateTime finished = completion.FinishedAt!.Value;
        Assert.Equal(DateTimeKind.Utc, finished.Kind);
        Assert.InRange(finished, completion.SubmittedAt, after);
        Assert.True(completion.FirstTokenAt.HasValue);
        Assert.True(completion.FirstTokenAt!.Value <= finished, "The first token came before the end.");
    }

    [Fact]
    public async Task AnAbortedCompletionCarriesTheTimeOfTheAbort()
    {
        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        using var engine = EngineTestSupport.NewEngine(EngineTestSupport.NewArchitecture(), entered, release);
        var seq = EngineTestSupport.NewSequence("aborted-at", promptLen: 4, maxNew: 64);
        var handle = engine.SubmitRequest(seq);

        DateTime beforeAbort;
        try
        {
            Assert.True(entered.Wait(TimeSpan.FromSeconds(10)), "The request never reached the model.");
            beforeAbort = DateTime.UtcNow;
            engine.Abort(seq.RequestId);
        }
        finally
        {
            // The abort is applied between steps, after the forward in flight returns.
            release.Set();
        }

        var completion = await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
        DateTime after = DateTime.UtcNow;

        Assert.Equal(SequenceStatus.FinishedAborted, completion.Status);
        Assert.Equal("aborted", completion.FinishReason);
        Assert.True(completion.FinishedAt.HasValue, "An aborted completion has an end time.");
        Assert.Equal(DateTimeKind.Utc, completion.FinishedAt!.Value.Kind);
        Assert.InRange(completion.FinishedAt.Value, beforeAbort, after);
    }

    // ---------------------------------------------------------------- preemption

    [Fact]
    public void ASlotPreemptionIsCountedAndTheFirstAdmissionIsKept()
    {
        var pool = EngineTestSupport.NewPool(16);
        var sched = new ContinuousBatchScheduler(EngineTestSupport.SchedulerOnlyConfig(numBlocks: 16), pool, NullLogger.Instance);
        var seq = EngineTestSupport.NewSequence("slot", promptLen: 8, maxNew: 4);
        sched.Submit(seq);
        Assert.Null(seq.FirstScheduledAt);
        Assert.Equal(0, seq.PreemptionCount);

        var first = sched.Schedule();
        Assert.Contains(first.ScheduledWork, w => ReferenceEquals(w.Sequence, seq));
        Assert.True(seq.FirstScheduledAt.HasValue);
        DateTime admitted = seq.FirstScheduledAt!.Value;

        // The model had no device slot for it: it goes back to the queue.
        sched.DeferForSequenceSlot(seq, first, contextTokens: 4096);
        Assert.Equal(SequenceStatus.Preempted, seq.Status);
        Assert.Equal(1, seq.PreemptionCount);
        Assert.Contains(seq.RequestId, first.PreemptedRequestIds);
        Assert.Equal(admitted, seq.FirstScheduledAt);

        // Long enough for DateTime.UtcNow to move on, so an overwrite would show.
        Thread.Sleep(20);
        var second = sched.Schedule();
        Assert.Contains(second.ScheduledWork, w => ReferenceEquals(w.Sequence, seq));
        Assert.Equal(SequenceStatus.Running, seq.Status);
        Assert.Equal(admitted, seq.FirstScheduledAt);
        Assert.Equal(1, seq.PreemptionCount);

        sched.DeferForSequenceSlot(seq, second, contextTokens: 4096);
        Assert.Equal(2, seq.PreemptionCount);
        Assert.Equal(admitted, seq.FirstScheduledAt);
    }

    [Fact]
    public void ABlockPreemptionIsCountedOnTheVictimOnlyAndItsFirstAdmissionIsKept()
    {
        // A, B and C take 1, 1 and 2 of the 4 blocks. At the next step A needs a second
        // block and C, the newest, is preempted for it (as in ContinuousBatchSchedulerTests).
        var pool = EngineTestSupport.NewPool(4);
        var sched = new ContinuousBatchScheduler(EngineTestSupport.SchedulerOnlyConfig(numBlocks: 4, maxRunning: 3), pool, NullLogger.Instance);
        var a = EngineTestSupport.NewSequence("a", promptLen: 8, maxNew: 8);
        var b = EngineTestSupport.NewSequence("b", promptLen: 7, maxNew: 8);
        var c = EngineTestSupport.NewSequence("c", promptLen: 16, maxNew: 8);
        sched.Submit(a);
        sched.Submit(b);
        sched.Submit(c);

        var initial = sched.Schedule();
        Assert.Equal(3, initial.ScheduledWork.Count);
        foreach (var work in initial.ScheduledWork)
            work.Sequence.AdvanceComputedTokens(work.NumScheduledTokens);
        DateTime cAdmitted = c.FirstScheduledAt!.Value;

        var pressured = sched.Schedule();
        Assert.Equal(new[] { c.RequestId }, pressured.PreemptedRequestIds);
        Assert.Equal(SequenceStatus.Preempted, c.Status);
        Assert.Equal(1, c.PreemptionCount);
        Assert.Equal(0, a.PreemptionCount);
        Assert.Equal(0, b.PreemptionCount);

        // Free the pool and let C back in: still one preemption, still its first admission.
        Assert.True(sched.Abort(a.RequestId));
        Assert.True(sched.Abort(b.RequestId));
        Thread.Sleep(20);
        var readmitted = sched.Schedule();
        Assert.Contains(readmitted.ScheduledWork, w => ReferenceEquals(w.Sequence, c));
        Assert.Equal(SequenceStatus.Running, c.Status);
        Assert.Equal(cAdmitted, c.FirstScheduledAt);
        Assert.Equal(1, c.PreemptionCount);
    }

    // ---------------------------------------------------------------- engine metrics

    [Fact]
    public async Task EngineCountersStayMonotonicWhenAnEngineIsDisposedAndRebuilt()
    {
        string architecture = EngineTestSupport.NewArchitecture();
        TensorSharpMetrics.EnsureInitialized();
        var readings = new ConcurrentQueue<(string Instrument, double Value, object? Architecture)>();
        using var meters = new MeterListener
        {
            InstrumentPublished = (instrument, listener) =>
            {
                if (instrument.Meter.Name == TensorSharpTelemetry.MeterName
                    && instrument.Name is StepsCounter or ForwardTimeCounter or RunningGauge)
                    listener.EnableMeasurementEvents(instrument);
            },
        };
        meters.SetMeasurementEventCallback<long>((i, v, tags, _) =>
            readings.Enqueue((i.Name, v, EngineTestSupport.TagValue(tags, TelemetryAttributes.ModelArchitecture))));
        meters.SetMeasurementEventCallback<double>((i, v, tags, _) =>
            readings.Enqueue((i.Name, v, EngineTestSupport.TagValue(tags, TelemetryAttributes.ModelArchitecture))));
        meters.SetMeasurementEventCallback<int>((i, v, tags, _) =>
            readings.Enqueue((i.Name, v, EngineTestSupport.TagValue(tags, TelemetryAttributes.ModelArchitecture))));
        meters.Start();

        // One collection, as an exporter's would be, of this test's architecture only.
        (double? Steps, double? Forward, bool Running) Collect()
        {
            readings.Clear();
            meters.RecordObservableInstruments();
            var mine = readings.Where(r => Equals(r.Architecture, architecture)).ToList();
            double? ValueOf(string instrument)
            {
                var values = mine.Where(r => r.Instrument == instrument).Select(r => r.Value).ToList();
                Assert.True(values.Count <= 1, $"{instrument} reported {values.Count} values for one architecture");
                return values.Count == 0 ? null : values[0];
            }
            return (ValueOf(StepsCounter), ValueOf(ForwardTimeCounter), mine.Any(r => r.Instrument == RunningGauge));
        }

        var first = EngineTestSupport.NewEngine(architecture);
        long firstSteps;
        double firstForward;
        try
        {
            await EngineTestSupport.RunToCompletion(first, EngineTestSupport.NewSequence("first", promptLen: 6, maxNew: 4));
            firstSteps = first.TotalStepsRun;
            firstForward = first.TotalForwardTime.TotalSeconds;
            Assert.True(firstSteps > 0);

            var live = Collect();
            Assert.Equal(firstSteps, live.Steps);
            Assert.True(live.Running, "A live engine reports its gauges.");
        }
        finally
        {
            first.Dispose();
        }

        // The disposed engine's totals are kept; its gauges go.
        var retired = Collect();
        Assert.Equal(firstSteps, retired.Steps);
        Assert.NotNull(retired.Forward);
        Assert.Equal(firstForward, retired.Forward!.Value, 9);
        Assert.False(retired.Running, "A disposed engine stops reporting its gauges.");

        // A rebuilt engine starts from zero, and the total does not go back with it.
        var second = EngineTestSupport.NewEngine(architecture);
        try
        {
            Assert.Equal(firstSteps, Collect().Steps);
            await EngineTestSupport.RunToCompletion(second, EngineTestSupport.NewSequence("second", promptLen: 9, maxNew: 4));
            long secondSteps = second.TotalStepsRun;
            Assert.True(secondSteps > 0);
            var both = Collect();
            Assert.Equal(firstSteps + secondSteps, both.Steps);
            Assert.True(both.Forward >= retired.Forward, "The forward time never decreases.");
        }
        finally
        {
            second.Dispose();
        }

        var afterBoth = Collect();
        Assert.Equal(firstSteps + second.TotalStepsRun, afterBoth.Steps);
        Assert.Equal(firstForward + second.TotalForwardTime.TotalSeconds, afterBoth.Forward!.Value, 9);
        Assert.False(afterBoth.Running);
    }

    [Fact]
    public async Task EveryEngineStepRecordsItsDurationAndBatchSize()
    {
        string architecture = EngineTestSupport.NewArchitecture();
        var durations = new ConcurrentQueue<(double Seconds, string? Kind)>();
        var sizes = new ConcurrentQueue<(int Size, string? Kind)>();
        using var meters = new MeterListener
        {
            InstrumentPublished = (instrument, listener) =>
            {
                if (instrument.Meter.Name == TensorSharpTelemetry.MeterName && instrument.Name is StepDuration or BatchSize)
                    listener.EnableMeasurementEvents(instrument);
            },
        };
        meters.SetMeasurementEventCallback<double>((i, v, tags, _) =>
        {
            if (i.Name == StepDuration && Equals(EngineTestSupport.TagValue(tags, TelemetryAttributes.ModelArchitecture), architecture))
                durations.Enqueue((v, EngineTestSupport.TagValue(tags, TelemetryAttributes.StepKind) as string));
        });
        meters.SetMeasurementEventCallback<int>((i, v, tags, _) =>
        {
            if (i.Name == BatchSize && Equals(EngineTestSupport.TagValue(tags, TelemetryAttributes.ModelArchitecture), architecture))
                sizes.Enqueue((v, EngineTestSupport.TagValue(tags, TelemetryAttributes.StepKind) as string));
        });
        meters.Start();

        var engine = EngineTestSupport.NewEngine(architecture);
        try
        {
            await Task.WhenAll(
                EngineTestSupport.RunToCompletion(engine, EngineTestSupport.NewSequence("a", promptLen: 6, maxNew: 4)),
                EngineTestSupport.RunToCompletion(engine, EngineTestSupport.NewSequence("b", promptLen: 9, maxNew: 4)));
        }
        finally
        {
            // Joins the worker, so every step it ran has been recorded.
            engine.Dispose();
        }

        Assert.True(engine.TotalStepsRun > 0);
        Assert.Equal(engine.TotalStepsRun, durations.Count);
        Assert.Equal(engine.TotalStepsRun, sizes.Count);
        Assert.All(durations, d =>
        {
            Assert.True(d.Seconds >= 0);
            Assert.Contains(d.Kind, StepKinds);
        });
        Assert.All(sizes, s =>
        {
            Assert.InRange(s.Size, 1, 2);
            Assert.Contains(s.Kind, StepKinds);
        });
        // The first step forwards prompts only.
        Assert.Equal("prefill", durations.First().Kind);
    }

    // ---------------------------------------------------------------- serving signals

    private const string InterToken = "tensorsharp.engine.inter_token.duration";
    private const string StepPrefillTokens = "tensorsharp.engine.step.prefill_tokens";
    private const string PrefixLookups = "tensorsharp.prefix_cache.lookups";

    private static readonly object[] MissSources =
        { "none", "shared_prefix", "rewind", "breakpoints", "live_cache", "retained_state", "pages", "other" };
    private static readonly object[] HitSources = { "end_state", "primary", "pages", "batched_end_state" };

    [Fact]
    public async Task EachOutputTokenAfterASequencesFirstRecordsTheTimeSinceTheOneBefore()
    {
        string architecture = EngineTestSupport.NewArchitecture();
        using var readings = new Readings(architecture, InterToken);

        var engine = EngineTestSupport.NewEngine(architecture);
        try
        {
            await Task.WhenAll(
                EngineTestSupport.RunToCompletion(engine, EngineTestSupport.NewSequence("five", promptLen: 6, maxNew: 5)),
                EngineTestSupport.RunToCompletion(engine, EngineTestSupport.NewSequence("three", promptLen: 9, maxNew: 3)));
        }
        finally
        {
            engine.Dispose();
        }

        // 4 gaps after the first of five tokens, 2 after the first of three. A first token has
        // no gap, so no gap ends in a prompt-only step.
        var gaps = readings.Of(InterToken);
        Assert.Equal(6, gaps.Count);
        Assert.All(gaps, g =>
        {
            Assert.True(g.Value >= 0);
            Assert.Contains(g.Tags[TelemetryAttributes.StepKind], new object[] { "decode", "mixed" });
        });
    }

    [Fact]
    public async Task AGapEndedByAStepThatAlsoPrefillsAnotherPromptIsTaggedMixed()
    {
        string architecture = EngineTestSupport.NewArchitecture();
        using var readings = new Readings(architecture, InterToken);

        using var entered = new ManualResetEventSlim();
        using var release = new ManualResetEventSlim();
        // A batched model runs every sequence a step schedules, as the serving engines do.
        var engine = EngineTestSupport.NewBatchedEngine(architecture, entered, release, holdAtForward: 3);
        try
        {
            // The first request's third step (its prompt, then its first and second tokens) is
            // held; the second request arrives meanwhile, so the next step decodes the first
            // beside the second one's prompt.
            var decoding = engine.SubmitRequest(EngineTestSupport.NewSequence("decoding", promptLen: 6, maxNew: 6));
            Task<InferenceCompletion> arriving;
            try
            {
                Assert.True(entered.Wait(TimeSpan.FromSeconds(10)), "The first request never reached the model.");
                arriving = EngineTestSupport.RunToCompletion(engine, EngineTestSupport.NewSequence("arriving", promptLen: 30, maxNew: 2));
            }
            finally
            {
                release.Set();
            }
            await Task.WhenAll(decoding.Completion.WaitAsync(TimeSpan.FromSeconds(30)), arriving);
        }
        finally
        {
            engine.Dispose();
        }

        var kinds = readings.Of(InterToken).Select(g => g.Tags[TelemetryAttributes.StepKind] as string).ToList();
        string seen = string.Join(",", kinds);
        Assert.True(kinds.Contains("mixed"), $"no gap ended in a mixed step: {seen}");
        Assert.True(kinds.Contains("decode"), $"no gap ended in a decode step: {seen}");
        Assert.DoesNotContain("prefill", kinds);
    }

    [Fact]
    public async Task EveryPromptTokenAStepForwardsIsCountedOnThatStep()
    {
        string architecture = EngineTestSupport.NewArchitecture();
        using var readings = new Readings(architecture, StepPrefillTokens);

        // A lone prompt in steps of up to 32 tokens: more than one prefill step.
        var engine = EngineTestSupport.NewEngine(architecture, config: EngineTestSupport.EngineConfig(soloPrefillChunk: 32));
        try
        {
            await EngineTestSupport.RunToCompletion(engine, EngineTestSupport.NewSequence("long", promptLen: 100, maxNew: 3));
        }
        finally
        {
            engine.Dispose();
        }

        var steps = readings.Of(StepPrefillTokens);
        Assert.True(steps.Count >= 3, $"{steps.Count} prefill step(s) for a 100-token prompt in 32-token steps");
        Assert.Equal(100, steps.Sum(s => s.Value));
        Assert.All(steps, s =>
        {
            Assert.InRange(s.Value, 1, 64);
            Assert.Contains(s.Tags[TelemetryAttributes.StepKind], new object[] { "prefill", "mixed" });
        });
    }

    [Fact]
    public async Task EachAdmissionCountsOnePrefixLookupByOutcomeAndSource()
    {
        string architecture = EngineTestSupport.NewArchitecture();
        using var readings = new Readings(architecture, PrefixLookups);

        var engine = EngineTestSupport.NewPrefixCachingEngine(architecture);
        try
        {
            // The same 40-token prompt twice: the second turn finds the first one's prefix.
            var first = await EngineTestSupport.RunToCompletion(engine,
                EngineTestSupport.NewSequence("first", promptLen: 40, maxNew: 3, cacheScope: "conversation"));
            Assert.Equal(0, first.PrefixCacheReusedTokens);
            var second = await EngineTestSupport.RunToCompletion(engine,
                EngineTestSupport.NewSequence("second", promptLen: 40, maxNew: 3, cacheScope: "conversation"));
            Assert.True(second.PrefixCacheReusedTokens > 0, "The second turn reused the first one's prefix.");
        }
        finally
        {
            engine.Dispose();
        }

        var lookups = readings.Of(PrefixLookups);
        Assert.Equal(2, lookups.Count);
        Assert.All(lookups, l => Assert.Equal(1, l.Value));
        Assert.Equal("miss", lookups[0].Tags[TelemetryAttributes.PrefixCacheOutcome]);
        Assert.Contains(lookups[0].Tags[TelemetryAttributes.PrefixCacheSource], MissSources);
        Assert.Contains(lookups[1].Tags[TelemetryAttributes.PrefixCacheOutcome], new object[] { "full", "partial" });
        Assert.Contains(lookups[1].Tags[TelemetryAttributes.PrefixCacheSource], HitSources);
    }

    [Theory]
    [InlineData("radix end state", "end_state")]
    [InlineData("radix primary cache", "primary")]
    [InlineData("radix pages", "pages")]
    [InlineData("radix batched end state", "batched_end_state")]
    [InlineData(null, "unknown")]
    [InlineData("something new", "other")]
    public void APrefixSourceMapsToAFixedValue(string? source, string expected) =>
        Assert.Equal(expected, ContinuousBatchScheduler.PrefixSourceCode(source));

    [Theory]
    [InlineData(null, "none")]
    [InlineData("", "none")]
    [InlineData("no resumable radix prefix", "none")]
    [InlineData("the prompt leaves every cached prefix at token 12, inside its 40-token shared prefix", "shared_prefix")]
    [InlineData("rewinding the cached conversation is declined by the model", "rewind")]
    [InlineData("the live cache is not continued for a request with explicit cache breakpoints", "breakpoints")]
    [InlineData("the live cache belongs to another conversation", "live_cache")]
    [InlineData("the retained state is cut short by a clamp", "retained_state")]
    [InlineData("the cached pages is too short to clone", "pages")]
    [InlineData("no reason anyone wrote down yet", "other")]
    public void AMissReasonMapsToAFixedValue(string? why, string expected) =>
        Assert.Equal(expected, ContinuousBatchScheduler.PrefixDeclineCode(why));

    [Fact]
    public void AModelWithSequenceSlotsReportsItsCountersWhileItsEngineLives()
    {
        string architecture = EngineTestSupport.NewArchitecture();
        string plain = EngineTestSupport.NewArchitecture();
        using var readings = new Readings(architecture,
            "tensorsharp.slots", "tensorsharp.slots.memory", "tensorsharp.slots.rewinds",
            "tensorsharp.cuda_graph.launches", "tensorsharp.engine.forward.issue_time");
        using var plainReadings = new Readings(plain, "tensorsharp.slots", "tensorsharp.cuda_graph.launches");

        var snapshot = new ServingTelemetrySnapshot
        {
            RunningSlots = 3,
            RetainedSlots = 2,
            SlotBytes = 5L << 30,
            RewindsToCheckpoint = 7,
            RewindsWithReplay = 2,
            RewindsToZero = 1,
            SoloGraphLaunches = 40,
            BatchedGraphLaunches = 400,
            ForwardIssueSeconds = 1.5,
        };
        bool broken = false;
        using var withoutSlots = EngineTestSupport.NewEngine(plain);
        var engine = EngineTestSupport.NewServingEngine(architecture,
            () => broken ? throw new InvalidOperationException("A failing read must not break the collection.") : snapshot);
        try
        {
            var live = readings.Collect();
            double Value(string instrument, string? key = null, string? value = null) =>
                Assert.Single(live, r => r.Instrument == instrument
                    && (key == null ? r.Tags.Count == 1 : Equals(r.Tags.GetValueOrDefault(key), value))).Value;

            Assert.Equal(3, Value("tensorsharp.slots", TelemetryAttributes.SlotState, "running"));
            Assert.Equal(2, Value("tensorsharp.slots", TelemetryAttributes.SlotState, "retained"));
            Assert.Equal(5L << 30, Value("tensorsharp.slots.memory"));
            Assert.Equal(7, Value("tensorsharp.slots.rewinds", TelemetryAttributes.RewindRoute, "checkpoint"));
            Assert.Equal(2, Value("tensorsharp.slots.rewinds", TelemetryAttributes.RewindRoute, "replay"));
            Assert.Equal(1, Value("tensorsharp.slots.rewinds", TelemetryAttributes.RewindRoute, "reset"));
            Assert.Equal(0, Value("tensorsharp.slots.rewinds", TelemetryAttributes.RewindRoute, "refused"));
            Assert.Equal(40, Value("tensorsharp.cuda_graph.launches", TelemetryAttributes.GraphKind, "solo"));
            Assert.Equal(400, Value("tensorsharp.cuda_graph.launches", TelemetryAttributes.GraphKind, "batched"));
            Assert.Equal(1.5, Value("tensorsharp.engine.forward.issue_time"));

            // A model without slots of its own reports none of them.
            Assert.Empty(plainReadings.Collect());

            // A read that throws leaves that engine out of the collection, which still completes.
            broken = true;
            Assert.Empty(readings.Collect());
            broken = false;
            snapshot = snapshot with { RunningSlots = 4 };
            Assert.Equal(4, Assert.Single(readings.Collect(),
                r => r.Instrument == "tensorsharp.slots" && Equals(r.Tags[TelemetryAttributes.SlotState], "running")).Value);
        }
        finally
        {
            engine.Dispose();
        }

        // A disposed engine stops reporting them.
        Assert.Empty(readings.Collect());
    }

    /// <summary>Every measurement the named instruments make for one architecture, with its tags.</summary>
    private sealed class Readings : IDisposable
    {
        private readonly MeterListener _listener;
        private readonly string _architecture;
        private readonly ConcurrentQueue<(string Instrument, double Value, Dictionary<string, object?> Tags)> _all = new();

        public Readings(string architecture, params string[] instruments)
        {
            _architecture = architecture;
            TensorSharpMetrics.EnsureInitialized();
            _listener = new MeterListener
            {
                InstrumentPublished = (instrument, listener) =>
                {
                    if (instrument.Meter.Name == TensorSharpTelemetry.MeterName && instruments.Contains(instrument.Name))
                        listener.EnableMeasurementEvents(instrument);
                },
            };
            _listener.SetMeasurementEventCallback<double>((i, v, tags, _) => Add(i, v, tags));
            _listener.SetMeasurementEventCallback<long>((i, v, tags, _) => Add(i, v, tags));
            _listener.SetMeasurementEventCallback<int>((i, v, tags, _) => Add(i, v, tags));
            _listener.Start();
        }

        private void Add(Instrument instrument, double value, ReadOnlySpan<KeyValuePair<string, object?>> tags)
        {
            var map = new Dictionary<string, object?>(StringComparer.Ordinal);
            foreach (var tag in tags) map[tag.Key] = tag.Value;
            if (Equals(map.GetValueOrDefault(TelemetryAttributes.ModelArchitecture), _architecture))
                _all.Enqueue((instrument.Name, value, map));
        }

        public List<(string Instrument, double Value, Dictionary<string, object?> Tags)> Of(string instrument) =>
            _all.Where(r => r.Instrument == instrument).ToList();

        /// <summary>One collection of the observable instruments, as an exporter's would be.</summary>
        public List<(string Instrument, double Value, Dictionary<string, object?> Tags)> Collect()
        {
            _all.Clear();
            _listener.RecordObservableInstruments();
            return _all.ToList();
        }

        public void Dispose() => _listener.Dispose();
    }
}

/// <summary>
/// The engine's process-wide hooks: the step observer (shared rate limit) and the per-step
/// span switch. Serialized with the other tests that change process-wide telemetry state.
/// </summary>
[Collection(TelemetryGlobalStateCollection.Name)]
public sealed class EngineStepObserverTests : IDisposable
{
    public void Dispose()
    {
        TensorSharpMetrics.SetEngineStepObserver(null);
        TensorSharpTelemetry.ConfigureTraceSteps(null);
    }

    // The observer's window is a quarter of a second and shared by the whole process, so a
    // test first waits out whatever observation came before it.
    private static void WaitOutTheObserverWindow() => Thread.Sleep(300);

    [Fact]
    public void TheObserverRunsAtMostFourTimesASecondAndItsExceptionsAreSwallowed()
    {
        int calls = 0;
        TensorSharpMetrics.SetEngineStepObserver(() =>
        {
            Interlocked.Increment(ref calls);
            throw new InvalidOperationException("A broken observer must not stop the engine.");
        });
        WaitOutTheObserverWindow();

        var clock = Stopwatch.StartNew();
        for (int i = 0; i < 10_000; i++)
            TensorSharpMetrics.OnEngineStep();
        clock.Stop();

        int observed = Volatile.Read(ref calls);
        Assert.InRange(observed, 1, 1 + (int)(clock.ElapsedMilliseconds / 250));

        // With no observer the call is a no-op.
        TensorSharpMetrics.SetEngineStepObserver(null);
        WaitOutTheObserverWindow();
        TensorSharpMetrics.OnEngineStep();
        Assert.Equal(observed, Volatile.Read(ref calls));
    }

    [Fact]
    public async Task TheEngineWorkerRunsTheObserverBetweenSteps()
    {
        string architecture = EngineTestSupport.NewArchitecture();
        var threads = new ConcurrentQueue<string?>();
        TensorSharpMetrics.SetEngineStepObserver(() => threads.Enqueue(Thread.CurrentThread.Name));
        WaitOutTheObserverWindow();

        using (var engine = EngineTestSupport.NewEngine(architecture))
            await EngineTestSupport.RunToCompletion(engine, EngineTestSupport.NewSequence("observed", promptLen: 6, maxNew: 4));

        Assert.Contains($"TensorSharp.InferenceEngine[{architecture}]", threads);
    }

    [Fact]
    public async Task WithStepTracingOnEachStepIsARootSpanLinkedToItsRequests()
    {
        string architecture = EngineTestSupport.NewArchitecture();
        var steps = new ConcurrentQueue<Activity>();
        using var listener = new ActivityListener
        {
            ShouldListenTo = s => s.Name == TensorSharpTelemetry.InferenceSourceName,
            Sample = (ref ActivityCreationOptions<ActivityContext> _) => ActivitySamplingResult.AllDataAndRecorded,
            ActivityStopped = a =>
            {
                if (a.DisplayName == "inference.step" && Equals(a.GetTagItem(TelemetryAttributes.ModelArchitecture), architecture))
                    steps.Enqueue(a);
            },
        };
        ActivitySource.AddActivityListener(listener);
        TensorSharpTelemetry.ConfigureTraceSteps(true);

        var request = new ActivityContext(ActivityTraceId.CreateRandom(), ActivitySpanId.CreateRandom(), ActivityTraceFlags.Recorded);
        // The engine is built inside a request, whose context its worker thread inherits.
        using var ambient = new Activity("request that built the engine").Start();
        var engine = EngineTestSupport.NewEngine(architecture);
        try
        {
            var traced = EngineTestSupport.NewSequence("traced", promptLen: 6, maxNew: 4);
            traced.TelemetryContext = request;
            var untraced = EngineTestSupport.NewSequence("untraced", promptLen: 5, maxNew: 3);
            await Task.WhenAll(EngineTestSupport.RunToCompletion(engine, traced), EngineTestSupport.RunToCompletion(engine, untraced));
        }
        finally
        {
            engine.Dispose();
        }

        Assert.True(engine.TotalStepsRun > 0);
        Assert.Equal(engine.TotalStepsRun, steps.Count);
        Assert.All(steps, s =>
        {
            Assert.Equal(default(ActivitySpanId), s.ParentSpanId);
            Assert.NotEqual(ambient.TraceId, s.TraceId);
            Assert.NotEqual(request.TraceId, s.TraceId);
            Assert.InRange((int)s.GetTagItem(TelemetryAttributes.BatchSize)!, 1, 2);
            Assert.Contains(s.GetTagItem(TelemetryAttributes.StepKind), new object[] { "prefill", "decode", "mixed" });
            // Only the request that has a span is linked.
            Assert.True(s.Links.Count() <= 1);
        });
        Assert.Contains(steps, s => s.Links.Any(l => l.Context.TraceId == request.TraceId && l.Context.SpanId == request.SpanId));
    }
}

/// <summary>A deterministic stub model and the scheduler settings the engine tests share.</summary>
file static class EngineTestSupport
{
    public const int BlockSize = 8;
    private const int VocabSize = 16;
    private const int NumLayers = 2;
    private const int NumKVHeads = 2;
    private const int HeadDim = 4;

    public static string NewArchitecture() => "telemetry-test-" + Guid.NewGuid().ToString("N")[..8];

    public static InferenceEngine NewEngine(string architecture,
        ManualResetEventSlim? forwardEntered = null, ManualResetEventSlim? releaseForward = null,
        SchedulerConfig? config = null) =>
        new(new StubModel(architecture, peakToken: 7, forwardEntered, releaseForward), config ?? EngineConfig(), NullLogger.Instance);

    /// <summary>An engine over a batched model, which forwards every sequence a step schedules
    /// together (a single-sequence model runs one of them per step).</summary>
    public static InferenceEngine NewBatchedEngine(string architecture,
        ManualResetEventSlim forwardEntered, ManualResetEventSlim releaseForward, int holdAtForward) =>
        new(new BatchedStubModel(architecture, forwardEntered, releaseForward, holdAtForward), EngineConfig(), NullLogger.Instance);

    /// <summary>An engine whose model keeps prompt pages in the radix prefix cache, so a later
    /// request in the same conversation scope can reuse an earlier one's prefix.</summary>
    public static InferenceEngine NewPrefixCachingEngine(string architecture) =>
        new(new PrefixCachingStubModel(architecture), EngineConfig(), NullLogger.Instance);

    /// <summary>An engine whose model reports slot and graph counters, as a model with
    /// sequence slots of its own does.</summary>
    public static InferenceEngine NewServingEngine(string architecture, Func<ServingTelemetrySnapshot> snapshot) =>
        new(new ServingStubModel(architecture, snapshot), EngineConfig(), NullLogger.Instance);

    public static async Task<InferenceCompletion> RunToCompletion(InferenceEngine engine, SequenceState seq)
    {
        var handle = engine.SubmitRequest(seq);
        return await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
    }

    public static SequenceState NewSequence(string id, int promptLen, int maxNew, string? cacheScope = null) =>
        new(id + "-" + Guid.NewGuid().ToString("N")[..6], Enumerable.Range(1, promptLen).ToList(), maxNew, BlockSize,
            SamplingConfig.Default, cacheScope: cacheScope);

    public static BlockPool NewPool(int numBlocks)
    {
        long blockBytes = 2L * NumLayers * NumKVHeads * BlockSize * HeadDim * sizeof(float);
        return new BlockPool(numBlocks, BlockSize, blockBytes);
    }

    /// <summary>The engine settings ContinuousBatchSchedulerTests drives its stub with,
    /// optionally with a lone prompt's step capped.</summary>
    public static SchedulerConfig EngineConfig(int soloPrefillChunk = 0)
    {
        var cfg = new SchedulerConfig
        {
            MaxNumBatchedTokens = 256,
            MaxNumRunningSequences = 8,
            MaxPrefillChunkSize = 64,
            NumBlocks = 16,
            BlockSize = BlockSize,
            EnablePrefixCaching = true,
            DecodeQuantumTokens = BlockSize,
        };
        return soloPrefillChunk > 0 ? cfg.WithSoloPrefillChunkSize(soloPrefillChunk) : cfg;
    }

    /// <summary>A scheduler on its own, without the radix cache an engine attaches.</summary>
    public static SchedulerConfig SchedulerOnlyConfig(int numBlocks, int maxRunning = 4) => new()
    {
        MaxNumBatchedTokens = 32,
        MaxNumRunningSequences = maxRunning,
        MaxPrefillChunkSize = 8,
        SoloPrefillChunkSize = 32,
        NumBlocks = numBlocks,
        BlockSize = BlockSize,
        EnablePrefixCaching = false,
        DecodeQuantumTokens = 1,
    };

    public static object? TagValue(ReadOnlySpan<KeyValuePair<string, object?>> tags, string key)
    {
        foreach (var tag in tags)
        {
            if (tag.Key == key)
                return tag.Value;
        }
        return null;
    }

    /// <summary>
    /// Logits that always peak at one token, the full KV-snapshot contract, and a tokenizer
    /// with no EOS: a request runs to its token limit. Optionally holds every forward until
    /// released, so a test can act while a request is in flight.
    /// </summary>
    private class StubModel : IModelArchitecture
    {
        private readonly int _peak;
        private readonly ManualResetEventSlim? _forwardEntered;
        private readonly ManualResetEventSlim? _releaseForward;
        private byte[] _state = Array.Empty<byte>();
        private int _cacheSeqLen;

        public StubModel(string architecture, int peakToken, ManualResetEventSlim? forwardEntered, ManualResetEventSlim? releaseForward)
        {
            _peak = peakToken;
            _forwardEntered = forwardEntered;
            _releaseForward = releaseForward;
            KVStateFingerprint = "fp-" + architecture;
            Config = new ModelConfig { VocabSize = VocabSize, Architecture = architecture };
            Tokenizer = new StubTokenizer(VocabSize);
        }

        public ModelConfig Config { get; }
        public ITokenizer Tokenizer { get; }
        public IMultimodalInjector MultimodalInjector => null!;
        public IBackendExecutionPlan ExecutionPlan => null!;
        public bool SupportsKVCacheTruncation => true;
        public bool SupportsKVStateSnapshot => true;
        public string KVStateFingerprint { get; }

        protected void HoldIfAsked()
        {
            _forwardEntered?.Set();
            if (_releaseForward != null && !_releaseForward.Wait(TimeSpan.FromSeconds(10)))
                throw new TimeoutException("The test did not release the held forward.");
        }

        public float[] Forward(int[] tokens)
        {
            HoldIfAsked();

            int newCount = _cacheSeqLen + tokens.Length;
            long needed = ComputeKVBlockByteSize(newCount);
            if (_state.Length < needed) Array.Resize(ref _state, (int)needed);
            for (int t = 0; t < tokens.Length; t++)
            {
                long start = ComputeKVBlockByteSize(_cacheSeqLen + t);
                long end = ComputeKVBlockByteSize(_cacheSeqLen + t + 1);
                byte v = (byte)(tokens[t] & 0xFF);
                for (long i = start; i < end; i++)
                    _state[i] = v;
            }
            _cacheSeqLen += tokens.Length;

            var logits = new float[VocabSize];
            logits[_peak] = 10.0f;
            return logits;
        }

        public void ResetKVCache() => _cacheSeqLen = 0;
        public void TruncateKVCache(int tokenCount) => _cacheSeqLen = Math.Min(_cacheSeqLen, tokenCount);
        public void Dispose() { }

        public long ComputeKVBlockByteSize(int tokenCount) => 2L * NumLayers * NumKVHeads * tokenCount * HeadDim * sizeof(float);

        public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
        {
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (destination.Length != expected) return false;
            if (startToken < 0 || startToken + tokenCount > _cacheSeqLen) return false;
            long startByte = ComputeKVBlockByteSize(startToken);
            new ReadOnlySpan<byte>(_state, (int)startByte, (int)expected).CopyTo(destination);
            return true;
        }

        public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
        {
            if (destToken != _cacheSeqLen) return false;
            long expected = ComputeKVBlockByteSize(tokenCount);
            if (source.Length != expected) return false;
            long needed = ComputeKVBlockByteSize(destToken + tokenCount);
            if (_state.Length < needed) Array.Resize(ref _state, (int)needed);
            long startByte = ComputeKVBlockByteSize(destToken);
            source.CopyTo(new Span<byte>(_state, (int)startByte, (int)expected));
            _cacheSeqLen = destToken + tokenCount;
            return true;
        }

        private sealed class StubTokenizer : ITokenizer
        {
            public StubTokenizer(int vocab)
            {
                Vocab = new string[vocab];
                for (int i = 0; i < vocab; i++) Vocab[i] = i.ToString();
            }

            public string[] Vocab { get; }
            public int BosTokenId => -1;
            public int[] EosTokenIds => Array.Empty<int>();
            public int VocabSize => Vocab.Length;
            public List<int> Encode(string text, bool addSpecial = true) => new();
            public string Decode(List<int> ids) => string.Join(",", ids);
            public void AppendTokenBytes(int tokenId, List<byte> buffer)
            {
                foreach (var b in System.Text.Encoding.UTF8.GetBytes(tokenId.ToString()))
                    buffer.Add(b);
            }
            public bool IsEos(int tokenId) => false;
            public int LookupToken(string tokenStr) => -1;
        }
    }

    private sealed class BatchedStubModel : StubModel, IBatchedPagedModel
    {
        private readonly int _holdAt;
        private int _forwards;

        public BatchedStubModel(string architecture, ManualResetEventSlim forwardEntered, ManualResetEventSlim releaseForward, int holdAt)
            : base(architecture, peakToken: 7, forwardEntered, releaseForward) => _holdAt = holdAt;

        public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext ctx)
        {
            if (++_forwards == _holdAt)
                HoldIfAsked();
            var logits = new float[ctx.Sequences.Count][];
            for (int i = 0; i < logits.Length; i++)
            {
                logits[i] = new float[VocabSize];
                logits[i][7] = 10.0f;
            }
            return logits;
        }
    }

    private sealed class PrefixCachingStubModel : StubModel, IPageOnlyPrefixCacheModel
    {
        public PrefixCachingStubModel(string architecture)
            : base(architecture, peakToken: 7, forwardEntered: null, releaseForward: null) { }

        public PrefixCacheCapabilities GetPrefixCacheCapabilities() => new()
        {
            Class = FamilyClass.P,
            NamespaceFingerprint = KVStateFingerprint,
            EndState = EndStateSupport.None,
            PrimaryResident = true,
            Truncation = TruncationKind.Any,
            RewindCapTokens = 16,
            Pages = PageSupport.A1HostSlab,
        };

        public long QuerySpareBytes(ResourceClass cls) => -1;
    }

    private sealed class ServingStubModel : StubModel, IServingTelemetrySource
    {
        private readonly Func<ServingTelemetrySnapshot> _snapshot;

        public ServingStubModel(string architecture, Func<ServingTelemetrySnapshot> snapshot)
            : base(architecture, peakToken: 7, forwardEntered: null, releaseForward: null) => _snapshot = snapshot;

        public ServingTelemetrySnapshot ReadServingTelemetry() => _snapshot();
    }
}
