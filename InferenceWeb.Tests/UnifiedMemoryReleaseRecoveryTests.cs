// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;

namespace InferenceWeb.Tests;

public sealed class UnifiedMemoryReleaseRecoveryTests
{
    private sealed class Scope : IDisposable { public void Dispose() { } }

    private sealed class ReleaseOracle : OracleModel, IBatchedPagedModel
    {
        internal ReleaseOracle() : base(new OracleTraits
        {
            Name = "release-oracle", Class = FamilyClass.R,
            EndState = EndStateSupport.CopyAndDonate, CanCaptureCopy = true,
            AdoptPrimaryOnDisplacement = true, Truncation = TruncationKind.None,
            Pages = PageSupport.None, DeviceDirtyOnForward = true,
        }, blockSize: 8) { }

        internal int ReleaseAttempts;
        internal bool FailFirstRelease;
        void IBatchedPagedModel.OnSequenceReleased(string requestId)
        {
            Assert.True(Monitor.IsEntered(((IModelArchitecture)this).GpuComputeLock));
            ReleaseAttempts++;
            if (FailFirstRelease && ReleaseAttempts == 1)
                throw new InvalidOperationException("The physical holder is still owned.");
            base.OnSequenceReleased(requestId);
        }
    }

    private static SchedulerConfig Config(RequestMemoryAdmission admission) => new()
    {
        BlockSize = 8, NumBlocks = 8192, MaxNumRunningSequences = 3,
        MaxNumBatchedTokens = 64, MaxPrefillChunkSize = 16,
        EnablePrefixCaching = false, StopRepetition = false,
        ExecutionOptionsOverride = new()
        {
            BatchedPathDisabled = true, PerSeqFusedEnabled = true,
            BatchedFusedDecodeEnabled = false, RetainedFusedCacheBudget = 0,
            PrefixCheckpointBudget = 0, KvHolderPoolMax = 0,
        },
        MemoryAdmission = admission,
    };

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task RecoveryRunsAfterRealReleaseAndRootRefundWhileChildrenRemainOwned(bool failRecovery)
    {
        var budget = new MemoryBudget([new("device", 100)]);
        using var optionalCache = budget.Reserve([new("device", 20)]);
        using var model = new ReleaseOracle();
        BudgetReservation? retained = null;
        var requests = new[]
        {
            new SequenceState("first", [1, 2, 3, 4], 2, 8, SamplingConfig.Greedy),
            new SequenceState("second", [5, 6, 7, 8], 8, 8, SamplingConfig.Greedy),
        };
        var cohorts = new List<int>();
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 30)])
        {
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            BeforeAdmissionWithRunning = (_, _) => [new("device", 100)],
            EnterSerialExecution = request =>
            {
                if (request.RequestId == "first" && retained == null)
                {
                    retained = request.MemoryEnvelope!.TryTake([new("device", 5)])!;
                    retained.Commit();
                }
                return new Scope();
            },
            AfterSequenceMemoryRelease = running =>
            {
                Assert.True(Monitor.IsEntered(((IModelArchitecture)model).GpuComputeLock));
                Assert.Equal(cohorts.Count + 1, model.ReleaseAttempts);
                cohorts.Add(running.Count);
                Assert.All(requests.Where(r => r.Status.IsFinished()), r => Assert.Null(r.MemoryEnvelope));
                Assert.Equal(5, budget.Snapshot().Single().Committed);
                Assert.DoesNotContain(running, r => r.Status.IsFinished());
                if (failRecovery) throw new InvalidOperationException("Optional cache growth failed.");
                Assert.True(optionalCache.TryResizeTotal([new("device", running.Count == 0 ? 65 : 35)]));
            },
        };
        var gate = new ComputeGate(); gate.Close();
        using var engine = new InferenceEngine(model, Config(admission)) { ComputeGate = gate };
        try
        {
            var handles = requests.Select(r => engine.SubmitRequest(r)).ToArray();
            using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            while (engine.WaitingCount != 2) await Task.Delay(5, deadline.Token);
            gate.Open();
            await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(deadline.Token);
            Assert.Equal(new[] { 1, 0 }, cohorts);
            Assert.All(requests, r => Assert.True(r.Status.IsFinished()));
            Assert.All(requests, r => Assert.Null(r.MemoryEnvelope));
            Assert.Equal(0, engine.WaitingCount); Assert.Equal(0, engine.RunningCount);
            Assert.Equal(failRecovery ? 20 : 65, optionalCache.OwnedCharges.Single().Bytes);
            Assert.Equal(5, budget.Snapshot().Single().Committed);
            Assert.Equal(0, engine.PendingRequestCount);
            Assert.Equal(0, engine.PendingSequenceReleaseCount);
            engine.Dispose();
            Assert.Equal(2, model.ReleaseAttempts); // A maintenance failure is not a failed holder release.
        }
        finally { gate.Open(); engine.Dispose(); retained?.Dispose(); }
    }

    [Fact]
    public async Task FailedPhysicalReleaseCannotRecoverItsCreditOrRunMaintenance()
    {
        var budget = new MemoryBudget([new("device", 100)]);
        using var optionalCache = budget.Reserve([new("device", 20)]);
        using var model = new ReleaseOracle { FailFirstRelease = true };
        int recoveries = 0;
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 30)])
        {
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            BeforeAdmissionWithRunning = (_, _) => [new("device", 100)],
            EnterSerialExecution = _ => new Scope(),
            AfterSequenceMemoryRelease = _ => recoveries++,
        };
        using var engine = new InferenceEngine(model, Config(admission));
        var request = new SequenceState("failed-release", [1, 2, 3, 4], 2, 8, SamplingConfig.Greedy);
        await engine.SubmitRequest(request).Completion.WaitAsync(TimeSpan.FromSeconds(5));
        Assert.Equal(1, model.ReleaseAttempts); Assert.Equal(0, recoveries);
        Assert.Equal(1, engine.PendingSequenceReleaseCount);
        Assert.NotNull(request.MemoryEnvelope);
        Assert.Equal(30, request.MemoryEnvelope.OwnedCharges.Single().Bytes);
        Assert.Equal(50, budget.Snapshot().Single().Reserved);
        engine.Dispose(); // Retries real release; shutdown never expands an optional cache.
        Assert.Equal(2, model.ReleaseAttempts); Assert.Equal(0, recoveries);
        Assert.Equal(0, engine.PendingSequenceReleaseCount);
        Assert.Null(request.MemoryEnvelope);
        Assert.Equal(20, budget.Snapshot().Single().Reserved);
    }

    [Fact]
    public async Task UntouchedQueuedCancellationWhileGateIsClosedDoesNotAuthorizeNativeMaintenance()
    {
        var budget = new MemoryBudget([new("device", 100)]);
        using var model = new ReleaseOracle();
        int recoveries = 0;
        var admission = new RequestMemoryAdmission(budget, _ => [new("device", 30)])
        {
            UsePerSequenceHolders = true, ReestimateAfterPhysicalCheck = true,
            BeforeAdmissionWithRunning = (_, _) => [new("device", 100)],
            EnterSerialExecution = _ => new Scope(),
            AfterSequenceMemoryRelease = _ => Interlocked.Increment(ref recoveries),
        };
        var gate = new ComputeGate(); gate.Close();
        using var engine = new InferenceEngine(model, Config(admission)) { ComputeGate = gate };
        try
        {
            var queued = new SequenceState("untouched", [1, 2, 3, 4], 2, 8, SamplingConfig.Greedy);
            var handle = engine.SubmitRequest(queued);
            using var deadline = new CancellationTokenSource(TimeSpan.FromSeconds(5));
            while (engine.WaitingCount != 1) await Task.Delay(5, deadline.Token);
            engine.Abort(queued.RequestId);
            await handle.Completion.WaitAsync(deadline.Token);
            Assert.Equal(0, queued.NumComputedTokens);
            Assert.Equal(0, Volatile.Read(ref recoveries));
            Assert.False(gate.IsOpen); Assert.Null(queued.MemoryEnvelope);
            Assert.Equal(100, budget.Snapshot().Single().Available);
            gate.Open();
            var next = new SequenceState("next", [5, 6, 7, 8], 2, 8, SamplingConfig.Greedy);
            await engine.SubmitRequest(next).Completion.WaitAsync(deadline.Token);
            Assert.Equal(1, Volatile.Read(ref recoveries));
            Assert.Equal(0, engine.WaitingCount); Assert.Equal(0, engine.RunningCount);
        }
        finally { gate.Open(); }
    }
}
