// Copyright (c) Zhongkai Fu. All rights reserved.
// https://github.com/zhongkaifu/TensorSharp
//
// This file is part of TensorSharp.
//
// TensorSharp is licensed under the BSD-3-Clause license found in the LICENSE file in the root directory of this source tree.
//
// The speculative stats' phases, on a scripted clock: every simulated tick a step spends lands in exactly one phase.
// The batched step (the unified runtime's path for one request or several) had timed its verify from the start of the
// step, so a verify step's "verify" held its draft as well (counted again under draft), and no step counted its
// commit: issue #376 read 98.7 ms a verify step off that line. No test here measures real time.
using System;
using System.Collections.Generic;
using System.Linq;
using System.Reflection;
using Microsoft.Extensions.Logging;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Speculative;
using Xunit;

namespace InferenceWeb.Tests;

public partial class SpeculativeExecutionTests
{
    /// <summary>A clock that moves only when a fake's call says its phase took time.</summary>
    private sealed class SimulatedClock
    {
        public long Now { get; private set; } = 1_000_000_000;
        public long Read() => Now;
        public void Advance(long ticks) => Now += ticks;
    }

    private const long Draft = 1, Snapshot = 10, Verify = 100, Draw = 1_000, Accept = 10_000, Restore = 100_000,
        Rewind = 1_000_000, Reforward = 10_000_000;

    private static (FakeSpeculativeModel Model, SpeculativeExecution Exec, SimulatedClock Clock, int Pos, int Last) TimedSoloStep(
        bool persistsAcceptedKv)
    {
        var model = new FakeSpeculativeModel { PersistsAcceptedKv = persistsAcceptedKv };
        var exec = NewExec(model, maxDraftTokens: 4);
        int pos = PrefillPrompt(model, exec, promptLen: 5, out int lastToken);
        // From here on the step's calls cost simulated time; the prefill before it cost none.
        var clock = new SimulatedClock();
        exec.Timestamp = clock.Read;
        model.Clock = clock;
        (model.DraftCost, model.SnapshotCost, model.VerifyCost, model.AcceptCost) = (Draft, Snapshot, Verify, Accept);
        (model.RestoreCost, model.RewindCost, model.ForwardCost) = (Restore, Rewind, Reforward);
        return (model, exec, clock, pos, lastToken);
    }

    private static long PhaseTicks(SpeculationStats s)
        => s.DraftTicks + s.SnapshotTicks + s.VerifyTicks + s.CommitTicks + s.RollbackTicks + s.CatchUpTicks + s.PlainTicks;

    [Fact]
    public void SoloStep_WrongDraft_VerifyIsTheForward_CommitIsTheDraws_RollbackIsTheReforward()
    {
        var (model, exec, clock, pos, last) = TimedSoloStep(persistsAcceptedKv: false);
        model.DraftWrongPositions.Add(pos + 1);   // the second draft is wrong
        long t0 = clock.Now;

        var outcome = exec.DecodeStep(last, pos, kMax: 4, drawNext: logits =>
        {
            clock.Advance(Draw);
            return Argmax(logits);
        });

        Assert.Equal(1, outcome.AcceptedCount);
        var s = exec.Stats;
        Assert.Equal(4, model.DraftStepCalls);
        Assert.Equal(4 * Draft, s.DraftTicks);
        Assert.Equal(Snapshot, s.SnapshotTicks);
        Assert.Equal(Verify, s.VerifyTicks);
        // Two rows drawn (the accepted one and the correction), then the trunk told the accept count.
        Assert.Equal(2 * Draw + Accept, s.CommitTicks);
        // The trunk keeps no verified rows here: restore, rewind and the kept prefix forwarded again.
        Assert.Equal(Restore + Rewind + Reforward, s.RollbackTicks);
        Assert.Equal(1, s.RollbackSteps);
        Assert.Equal(clock.Now - t0, PhaseTicks(s));
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void SoloStep_WrongDraft_KeptInPlace_IsCommitAndNoRollbackTime()
    {
        var (model, exec, clock, pos, last) = TimedSoloStep(persistsAcceptedKv: true);
        model.DraftWrongPositions.Add(pos + 1);
        long t0 = clock.Now;

        var outcome = exec.DecodeStep(last, pos, kMax: 4, drawNext: logits =>
        {
            clock.Advance(Draw);
            return Argmax(logits);
        });

        Assert.Equal(1, outcome.AcceptedCount);
        var s = exec.Stats;
        Assert.Equal(Verify, s.VerifyTicks);
        // The verify's own rows are kept by a rewind of the position: part of the commit, not a rollback.
        Assert.Equal(2 * Draw + Accept + Rewind, s.CommitTicks);
        Assert.Equal(0, s.RollbackTicks);
        Assert.Equal(1, s.RollbackSteps);
        Assert.Equal(clock.Now - t0, PhaseTicks(s));
    }

    [Fact]
    public void SoloStep_FullAcceptance_CommitIsEveryRowsDraw()
    {
        var (model, exec, clock, pos, last) = TimedSoloStep(persistsAcceptedKv: false);
        long t0 = clock.Now;

        var outcome = exec.DecodeStep(last, pos, kMax: 4, drawNext: logits =>
        {
            clock.Advance(Draw);
            return Argmax(logits);
        });

        Assert.Equal(4, outcome.AcceptedCount);
        var s = exec.Stats;
        Assert.Equal((4 * Draft, Verify), (s.DraftTicks, s.VerifyTicks));
        Assert.Equal(5 * Draw + Accept, s.CommitTicks);   // four drafts and the bonus row
        Assert.Equal((0L, 0L), (s.RollbackTicks, s.RollbackSteps));
        Assert.Equal(clock.Now - t0, PhaseTicks(s));
    }

    [Fact]
    public void CommitBatch_VerifyStepTakesVerifyAndCommit_PlainStepIsChargedBoth()
    {
        var model = new FakeSpeculativeModel { ResumesAfterGap = true };
        var exec = NewExec(model, maxDraftTokens: 4);
        int pos = PrefillPrompt(model, exec, promptLen: 5, out int last);
        // The catch-up CommitBatch times itself reads this clock, which no call moves here: a plain step's catch-up is
        // the plain step's, so a real clock would put real time in the plain total.
        exec.Timestamp = new SimulatedClock().Read;
        var features = new float[5 * HiddenSize];
        for (int r = 0; r < 5; r++)
            Array.Fill(features, pos + r + 1, r * HiddenSize, HiddenSize);
        var drafts = new List<int> { model.ExpectedNext(pos), model.ExpectedNext(pos + 1) };

        exec.CommitBatch(last, pos, drafts, accepted: 2, features, 0, verifyTicks: 300, commitTicks: 40);
        Assert.Equal((1L, 300L, 40L, 0L), (exec.Stats.VerifySteps, exec.Stats.VerifyTicks, exec.Stats.CommitTicks, exec.Stats.PlainTicks));

        exec.CommitBatch(model.ExpectedNext(pos + 2), pos + 3, new List<int>(), accepted: 0, features, 0, verifyTicks: 200,
            commitTicks: 7);
        Assert.Equal((1L, 1L, 300L, 40L, 207L), (exec.Stats.VerifySteps, exec.Stats.PlainSteps, exec.Stats.VerifyTicks,
            exec.Stats.CommitTicks, exec.Stats.PlainTicks));
        Assert.Empty(model.ProtocolViolations);
    }

    [Fact]
    public void PlainStepCatchUp_IsThePlainStepsOnBothPaths_SoCatchUpHoldsVerifyStepsOnly()
    {
        const long catchUp = 7_000;
        // The batched second half: a verify step's catch-up is catch-up, a plain step's is the plain step's. It had been
        // catch-up too, so perVerifyMs' catch-up divided plain steps' injects by the verify steps.
        var model = new FakeSpeculativeModel { ResumesAfterGap = true };
        var exec = NewExec(model, maxDraftTokens: 4);
        int pos = PrefillPrompt(model, exec, promptLen: 5, out int last);
        var clock = new SimulatedClock();
        exec.Timestamp = clock.Read;
        (model.Clock, model.CatchUpCost) = (clock, catchUp);
        var features = new float[5 * HiddenSize];
        for (int r = 0; r < 5; r++)
            Array.Fill(features, pos + r + 1, r * HiddenSize, HiddenSize);
        var drafts = new List<int> { model.ExpectedNext(pos), model.ExpectedNext(pos + 1) };

        exec.CommitBatch(last, pos, drafts, accepted: 2, features, 0, verifyTicks: 300, commitTicks: 40);
        Assert.Equal((catchUp, 0L), (exec.Stats.CatchUpTicks, exec.Stats.PlainTicks));
        exec.CommitBatch(model.ExpectedNext(pos + 2), pos + 3, new List<int>(), accepted: 0, features, 0, verifyTicks: 200,
            commitTicks: 7);
        Assert.Equal((1L, 1L), (exec.Stats.VerifySteps, exec.Stats.PlainSteps));
        Assert.Equal((catchUp, 200L + 7L + catchUp), (exec.Stats.CatchUpTicks, exec.Stats.PlainTicks));
        Assert.Equal(exec.Stats.CatchUpMs, exec.Stats.PerVerifyStep(exec.Stats.CatchUpMs));
        Assert.Empty(model.ProtocolViolations);

        // The solo plain step: its forward and its catch-up are plain, nothing is catch-up.
        var (soloModel, solo, soloClock, soloPos, soloLast) = TimedSoloStep(persistsAcceptedKv: false);
        soloModel.CatchUpCost = catchUp;
        long t0 = soloClock.Now;
        var outcome = solo.DecodeStep(soloLast, soloPos, kMax: 0, drawNext: logits => Argmax(logits));
        Assert.False(outcome.UsedSpeculation);
        Assert.Equal((1L, Reforward + catchUp, 0L), (solo.Stats.PlainSteps, solo.Stats.PlainTicks, solo.Stats.CatchUpTicks));
        Assert.Equal(soloClock.Now - t0, PhaseTicks(solo.Stats));
        Assert.Empty(soloModel.ProtocolViolations);
    }

    [Fact]
    public void PerVerifyStep_DividesAPhaseByTheVerifySteps()
    {
        var stats = new SpeculationStats();
        Assert.Equal(0.0, stats.PerVerifyStep(120));
        stats.VerifySteps = 4;
        Assert.Equal(30.0, stats.PerVerifyStep(120));
        stats.CommitTicks = 5;
        stats.Reset();
        Assert.Equal((0L, 0L), (stats.CommitTicks, stats.VerifySteps));
    }

    /// <summary>The fake model with a per-request trunk the batched step serves (<see cref="IBatchedSpeculativeTarget"/>):
    /// a verify writes each window's draws and feature rows, a commit advances each request by its kept rows.</summary>
    private sealed class FakeBatchedSpeculativeModel : FakeSpeculativeModel, ISpeculativeTarget, IBatchedSpeculativeTarget
    {
        private readonly Dictionary<string, int> _lengths = new(StringComparer.Ordinal);
        public long CommitCost { get; set; }
        public int BatchedVerifies { get; private set; }
        public bool SpecTrunkFollowsBoundCache => true;
        public void Start(string requestId, int length) => _lengths[requestId] = length;
        public int Length(string requestId) => _lengths[requestId];
        public int BatchedSpecWindow(int sequences) => 5;
        public int FusedCacheSeqLen(string requestId) => _lengths[requestId];
        public int FusedCacheFreeRows(string requestId) => 1 << 20;
        public IDisposable BeginBatchedSpeculation() => new Scope();
        public void BindDraftState(string requestId) { }
        public void ReleaseDraftState(string requestId) { }

        public bool TryBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int window, int[] tokens, int[] drawn,
            float[] features)
        {
            BatchedVerifies++;
            Clock?.Advance(VerifyCost);
            for (int i = 0; i < requestIds.Count; i++)
                for (int r = 0; r < window; r++)
                {
                    drawn[i * window + r] = ExpectedNext(positions[i] + r);
                    Array.Fill(features, positions[i] + r + 1, (i * window + r) * HiddenSize, HiddenSize);
                }
            return true;
        }

        public void CommitBatchedSpecVerify(IReadOnlyList<string> requestIds, int[] positions, int[] kept)
        {
            Clock?.Advance(CommitCost);
            for (int i = 0; i < requestIds.Count; i++)
                _lengths[requestIds[i]] = positions[i] + kept[i];
        }

        private sealed class Scope : IDisposable
        {
            public void Dispose() { }
        }
    }

    [Fact]
    public void BatchedStep_VerifyIsTheForwardAlone_AndCommitIsItsOwnPhase()
    {
        const long verify = 100_000, commit = 10_000_000, draft = 1_000;
        var clock = new SimulatedClock();
        var model = new FakeBatchedSpeculativeModel
        {
            ResumesAfterGap = true, Clock = clock, VerifyCost = verify, CommitCost = commit, DraftCost = draft,
        };
        var options = new SpeculationOptions
        {
            Enabled = true, SpeculatorName = SpeculatorRegistry.DraftHead, MaxDraftTokens = 4, MaxDraftTokensExplicit = true,
            MinDraftProb = 0.5f,
        };
        var pool = new BlockPool(64, BlockSize, BlockSize * 4);
        var scheduler = new ContinuousBatchScheduler(new SchedulerConfig { Speculation = options }, pool);
        var executor = new BatchExecutor(model, pool, scheduler) { Timestamp = clock.Read };
        SequenceState Decoding(string id, int promptLen)
        {
            var seq = new SequenceState(id, Enumerable.Range(1, promptLen).ToList(), 64, BlockSize, SamplingConfig.Greedy);
            foreach (var block in pool.AllocateNew(8)!) seq.BlockTable.AppendBlock(block);
            seq.AdvanceComputedTokens(promptLen);
            seq.Status = SequenceStatus.Running;
            seq.PendingDeviceToken = model.ExpectedNext(promptLen - 1);
            seq.PendingDevicePosition = promptLen;
            model.Start(id, promptLen);
            return seq;
        }
        var a = Decoding("batched-a", 6);
        var b = Decoding("batched-b", 9);
        var step = typeof(BatchExecutor).GetMethod("TryBatchedSpeculativeDecode", BindingFlags.Instance | BindingFlags.NonPublic)!;
        void Step()
        {
            var work = new List<ScheduledSequenceWork>
            {
                new(a, 1, isNewAdmission: false, isPrefill: false),
                new(b, 1, isNewAdmission: false, isPrefill: false),
            };
            var results = new List<SequenceStepResult>();
            Assert.True((bool)step.Invoke(executor, new object[] { work, results })!);
            Assert.All(results, r => Assert.Null(r.Error));
        }

        // First step: armed over the committed tokens with no carried row, neither request drafts; each rides the
        // verify with its anchor and is charged its share of the forward and of the commit as a plain step.
        Step();
        Assert.Equal(1, model.BatchedVerifies);
        foreach (var seq in new[] { a, b })
        {
            var s = seq.SpecStats;
            Assert.Equal((0L, 1L, verify / 2 + commit / 2), (s.VerifySteps, s.PlainSteps, s.PlainTicks));
            Assert.Equal((0L, 0L, 0L), (s.DraftTicks, s.VerifyTicks, s.CommitTicks));
        }

        // Second step: each drafts four tokens (4,000 simulated ticks between them before the verify), all accepted.
        Step();
        Assert.Equal(2, model.BatchedVerifies);
        foreach (var seq in new[] { a, b })
        {
            var s = seq.SpecStats;
            Assert.Equal((1L, 4L, 4L), (s.VerifySteps, s.TokensDrafted, s.TokensAccepted));
            Assert.Equal(4 * draft, s.DraftTicks);
            // Its share of the verify forward, without the 8 drafts the step ran before it.
            Assert.Equal(verify / 2, s.VerifyTicks);
            Assert.Equal(commit / 2, s.CommitTicks);
            Assert.Equal(verify / 2 + commit / 2, s.PlainTicks);   // the first step's, unchanged
            Assert.Equal(s.VerifyMs, s.PerVerifyStep(s.VerifyMs));
        }
        Assert.Equal((6 + 1 + 5, 9 + 1 + 5), (model.Length("batched-a"), model.Length("batched-b")));
        Assert.Empty(model.ProtocolViolations);
    }

    private sealed class InfoLines : ILogger
    {
        public List<string> Lines { get; } = new();
        public IDisposable BeginScope<TState>(TState state) where TState : notnull => null;
        public bool IsEnabled(LogLevel logLevel) => true;
        public void Log<TState>(LogLevel logLevel, EventId eventId, TState state, Exception exception,
            Func<TState, Exception, string> formatter)
        {
            lock (Lines) Lines.Add(formatter(state, exception));
        }
    }

    [Fact]
    public void EngineRequest_StatsLine_ReportsCommitAndPerVerifyStepPhases()
    {
        var model = new FakeSpeculativeModel();
        var log = new InfoLines();
        var seq = RunEngineRequest(model, promptLen: 5, maxNewTokens: 24, specEnabled: true, logger: log);

        Assert.True(seq.SpecStats.VerifySteps > 0);
        string line;
        lock (log.Lines) line = Assert.Single(log.Lines, l => l.StartsWith("Speculative decoding stats for ", StringComparison.Ordinal));
        Assert.Matches(@"\| phaseMs draft=\d+ verify=\d+ commit=\d+ snapshot=\d+ rollback=\d+ catchUp=\d+ plain=\d+ \|", line);
        Assert.Matches(@"\| perVerifyMs draft=[\d.]+ verify=[\d.]+ commit=[\d.]+ catchUp=[\d.]+ \| governor ", line);
    }
}
