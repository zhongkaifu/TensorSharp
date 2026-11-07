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
        ManualResetEventSlim? forwardEntered = null, ManualResetEventSlim? releaseForward = null) =>
        new(new StubModel(architecture, peakToken: 7, forwardEntered, releaseForward), EngineConfig(), NullLogger.Instance);

    public static async Task<InferenceCompletion> RunToCompletion(InferenceEngine engine, SequenceState seq)
    {
        var handle = engine.SubmitRequest(seq);
        return await handle.Completion.WaitAsync(TimeSpan.FromSeconds(30));
    }

    public static SequenceState NewSequence(string id, int promptLen, int maxNew) =>
        new(id + "-" + Guid.NewGuid().ToString("N")[..6], Enumerable.Range(1, promptLen).ToList(), maxNew, BlockSize, SamplingConfig.Default);

    public static BlockPool NewPool(int numBlocks)
    {
        long blockBytes = 2L * NumLayers * NumKVHeads * BlockSize * HeadDim * sizeof(float);
        return new BlockPool(numBlocks, BlockSize, blockBytes);
    }

    /// <summary>The engine settings ContinuousBatchSchedulerTests drives its stub with.</summary>
    private static SchedulerConfig EngineConfig() => new()
    {
        MaxNumBatchedTokens = 256,
        MaxNumRunningSequences = 8,
        MaxPrefillChunkSize = 64,
        NumBlocks = 16,
        BlockSize = BlockSize,
        EnablePrefixCaching = true,
        DecodeQuantumTokens = BlockSize,
    };

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
    private sealed class StubModel : IModelArchitecture
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

        public float[] Forward(int[] tokens)
        {
            _forwardEntered?.Set();
            if (_releaseForward != null && !_releaseForward.Wait(TimeSpan.FromSeconds(10)))
                throw new TimeoutException("The test did not release the held forward.");

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
}
