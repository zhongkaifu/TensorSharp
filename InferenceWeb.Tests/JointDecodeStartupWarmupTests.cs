// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Collections.Concurrent;
using System.Runtime.CompilerServices;
using System.Text.Json;
using InferenceWeb.Tests.PrefixCache.Fakes;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Scheduling;
using TensorSharp.Runtime.Scheduling.PrefixCache;
using TensorSharp.Server;
using TensorSharp.Server.Hosting;

namespace InferenceWeb.Tests;

/// <summary>Real CPU scheduler, owner scopes and independent recurrent oracle
/// run through the startup stream helper. The fake's batch record proves the
/// batch API executed; it is not a native or trained-model qualification.</summary>
public sealed class JointDecodeStartupWarmupTests
{
    [Fact]
    public async Task DistinctOwnedCohortsExecuteWidthTwoAndThreeAndPreserveEveryRequestHistory()
    {
        using var fixture = new Fixture();
        foreach (int width in new[] { 2, 3 })
        {
            var result = await fixture.Run(width).WaitAsync(TimeSpan.FromSeconds(5));
            Assert.True(result.Warmed, result.Detail);
            Assert.Equal(width, result.Evidence.ActiveSequenceCount);
            Assert.Contains(width, fixture.Model.Widths);
            Assert.Equal(0, fixture.Engine.RunningCount); Assert.Equal(0, fixture.Engine.WaitingCount);
            Assert.Null(fixture.Engine.ComputeGate);
            Assert.Empty(fixture.Workspaces);
        }
        Assert.Equal(5, fixture.Results.Count);
        Assert.Equal(5, fixture.Released);
        foreach (var output in fixture.Results.Values)
        {
            using var cold = OracleFakes.R(8);
            var logits = cold.Forward([1, 2, 3, 4]);
            int first = ArgMax(logits), second = ArgMax(cold.Forward([first]));
            Assert.Equal(new[] { first, second }, output);
        }
        Assert.Equal(0, fixture.Budget.Snapshot().Single().Committed);
    }

    [Fact]
    public async Task SerialCompletionAndStaleEvidenceDoNotCountAsJointWarmup()
    {
        using var fixture = new Fixture();
        var first = await fixture.Run(2);
        Assert.True(first.Warmed, first.Detail);
        fixture.Model.Decline = true;
        var second = await fixture.Run(2);
        Assert.True(second.RequestsCompleted);
        Assert.False(second.NativeBatchAccepted); Assert.False(second.Warmed);
        Assert.Null(second.Evidence);
        Assert.True(second.CleanupCompleted); Assert.Empty(fixture.Workspaces);
        Assert.Contains("no fresh native", second.Detail);
    }

    [Fact]
    public async Task AsymmetricColdPrefillsReachOneJointCohortBeforeTheShortRequestCanFinish()
    {
        using var fixture = new Fixture(maxBatched: 8)
        {
            PromptFactory = id => id == "warm-1" ? new[] { 1, 2 }
                : Enumerable.Range(1, id == "warm-2" ? 39 : 17).ToArray()
        };
        var result = await fixture.Run(3).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(result.Warmed, result.Detail);
        Assert.NotEmpty(fixture.Model.Widths);
        Assert.All(fixture.Model.Widths, width => Assert.Equal(3, width));
        foreach (var (id, actual) in fixture.Results)
        {
            using var cold = OracleFakes.R(8);
            int first = ArgMax(cold.Forward(fixture.PromptFactory(id)));
            int second = ArgMax(cold.Forward([first]));
            Assert.Equal(new[] { first, second }, actual);
        }
        Assert.Null(fixture.Engine.ComputeGate); Assert.Empty(fixture.Workspaces);
        Assert.Equal(0, fixture.Budget.Snapshot().Single().Committed + fixture.Budget.Snapshot().Single().Reserved);
    }

    [Fact]
    public async Task InsufficientCohortCapacityReleasesHeldDecodersAndReportsTheWidthAsDeclined()
    {
        using var fixture = new Fixture(memoryBytes: 64);
        var result = await fixture.Run(3).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(result.RequestsCompleted, result.Detail);
        Assert.False(result.Warmed); Assert.False(result.NativeBatchAccepted);
        Assert.Contains("physical memory", result.Detail);
        Assert.DoesNotContain(3, fixture.Model.Widths);
        Assert.Equal(3, fixture.Results.Count); Assert.Empty(fixture.Workspaces);
        Assert.Null(fixture.Engine.ComputeGate);
        Assert.Equal(0, fixture.Budget.Snapshot().Single().Committed + fixture.Budget.Snapshot().Single().Reserved);
    }

    [Fact]
    public async Task AForeignLiveHandleCannotBeCapturedAsOwnedStartupWork()
    {
        using var fixture = new Fixture { InjectForeignRequest = true };
        var result = await fixture.Run(3).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(result.Warmed); Assert.False(result.NativeBatchAccepted);
        Assert.Contains("exact live request set", result.Detail);
        Assert.True(result.CleanupCompleted); Assert.Empty(fixture.Workspaces);
        Assert.Equal(0, fixture.Engine.PendingRequestCount);
        Assert.Null(fixture.Engine.ComputeGate);
    }

    [Fact]
    public async Task SequenceSlotPreemptionReleasesTheBarrierAndAllowsEveryRequestToFinish()
    {
        using var fixture = new Fixture(); fixture.Model.RefuseThirdSlotOnce = true;
        var result = await fixture.Run(3).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.True(result.RequestsCompleted, result.Detail);
        Assert.False(result.Warmed); Assert.False(result.NativeBatchAccepted);
        Assert.Contains("preempted", result.Detail);
        Assert.DoesNotContain(3, fixture.Model.Widths);
        Assert.Equal(3, fixture.Results.Count); Assert.Empty(fixture.Workspaces);
        Assert.Null(fixture.Engine.ComputeGate);
        Assert.Equal(0, fixture.Budget.Snapshot().Single().Committed + fixture.Budget.Snapshot().Single().Reserved);
    }

    [Fact]
    public async Task StreamFailureDrainsOwnedRequestsAndRestoresGateBeforeCleaningEveryWorkspace()
    {
        using var fixture = new Fixture { FailSecondStream = true };
        var result = await fixture.Run(3).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(result.Warmed); Assert.True(result.CleanupCompleted);
        Assert.Empty(fixture.Workspaces); Assert.Equal(3, fixture.Released);
        Assert.Null(fixture.Engine.ComputeGate);
        Assert.Equal(0, fixture.Engine.RunningCount + fixture.Engine.WaitingCount);
        Assert.Equal(0, fixture.Budget.Snapshot().Single().Committed);
    }

    [Fact]
    public async Task CancellationRetainsWorkspacesUntilPhysicalWorkFinishesAndThenCleansThem()
    {
        using var fixture = new Fixture(); fixture.Model.Pause = true;
        using var cancellation = new CancellationTokenSource();
        var run = fixture.Run(2, cancellation.Token, TimeSpan.FromMilliseconds(30));
        await fixture.Model.Started.Task.WaitAsync(TimeSpan.FromSeconds(3));
        cancellation.Cancel();
        var result = await run.WaitAsync(TimeSpan.FromSeconds(3));
        Assert.False(result.Warmed); Assert.False(result.CleanupCompleted);
        Assert.True(result.CleanupPending);
        Assert.Equal(2, fixture.Workspaces.Count); Assert.Equal(0, fixture.Released);
        Assert.Null(fixture.Engine.ComputeGate);
        fixture.Model.Continue.Set();
        await fixture.Cleaned.Task.WaitAsync(TimeSpan.FromSeconds(3));
        Assert.Empty(fixture.Workspaces); Assert.Equal(2, fixture.Released);
        Assert.Equal(0, fixture.Engine.RunningCount + fixture.Engine.WaitingCount);
        Assert.Equal(0, fixture.Budget.Snapshot().Single().Committed);
    }

    [Fact]
    public async Task AnExistingLifecycleGateIsNeverBorrowedOrOpened()
    {
        using var fixture = new Fixture();
        var gate = new ComputeGate(); gate.Close(); fixture.Engine.ComputeGate = gate;
        var result = await fixture.Run(2);
        Assert.False(result.Warmed); Assert.True(result.CleanupCompleted);
        Assert.Same(gate, fixture.Engine.ComputeGate); Assert.False(gate.IsOpen);
        Assert.Equal(0, fixture.Created); Assert.Empty(fixture.Workspaces);
        gate.Open();
    }

    [Fact]
    public async Task CompletedSerialStreamCleanupFenceIsAwaitedBeforeInstallingTheJointGate()
    {
        using var fixture = new Fixture();
        using var finishCleanup = new ManualResetEventSlim();
        var cleanupStarted = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var precedingCleanup = Task.Run(() =>
        {
            lock (((IModelArchitecture)fixture.Model).GpuComputeLock)
            {
                cleanupStarted.TrySetResult();
                Assert.True(finishCleanup.Wait(TimeSpan.FromSeconds(3)));
            }
        });
        await cleanupStarted.Task.WaitAsync(TimeSpan.FromSeconds(3));
        try
        {
            var run = fixture.Run(2);
            await Task.Delay(30);
            Assert.False(run.IsCompleted); Assert.Equal(0, fixture.Created);
            Assert.Null(fixture.Engine.ComputeGate);
            finishCleanup.Set(); await precedingCleanup;
            var result = await run.WaitAsync(TimeSpan.FromSeconds(3));
            Assert.True(result.Warmed, result.Detail); Assert.Contains(2, fixture.Model.Widths);
            Assert.Empty(fixture.Workspaces); Assert.Null(fixture.Engine.ComputeGate);
        }
        finally { finishCleanup.Set(); await precedingCleanup; }
    }

    [Fact]
    public async Task GateInstalledByTheLifecycleDuringCancelledWarmupRemainsClosedAndOwnedByTheLifecycle()
    {
        using var fixture = new Fixture(); fixture.Model.Pause = true;
        using var cancellation = new CancellationTokenSource();
        var run = fixture.Run(2, cancellation.Token, TimeSpan.FromMilliseconds(30));
        await fixture.Model.Started.Task.WaitAsync(TimeSpan.FromSeconds(3));
        var external = new ComputeGate(); external.Close(); fixture.Engine.ComputeGate = external;
        cancellation.Cancel();
        var result = await run.WaitAsync(TimeSpan.FromSeconds(3));
        Assert.False(result.Warmed); Assert.True(result.CleanupPending);
        Assert.Same(external, fixture.Engine.ComputeGate); Assert.False(external.IsOpen);
        Assert.Equal(0, fixture.Released); Assert.Equal(2, fixture.Workspaces.Count);
        external.Open(); fixture.Model.Continue.Set();
        await fixture.Cleaned.Task.WaitAsync(TimeSpan.FromSeconds(3));
        Assert.Empty(fixture.Workspaces); Assert.Same(external, fixture.Engine.ComputeGate);
    }

    [Theory]
    [InlineData(2, true, true, 64, "running or batch-token limit")]
    [InlineData(3, false, true, 64, "joint decode route is disabled")]
    [InlineData(3, true, false, 64, "shared allocation scope")]
    [InlineData(3, true, true, 2, "running or batch-token limit")]
    public async Task UnqualifiedActualSchedulerConfigurationSkipsBeforeCreatingOwnedSessions(
        int maxRunning, bool jointDecode, bool admission, int maxBatched, string reason)
    {
        using var fixture = new Fixture(maxRunning, jointDecode, admission, maxBatched);
        var result = await fixture.Run(3).WaitAsync(TimeSpan.FromSeconds(3));
        Assert.False(result.Warmed); Assert.True(result.CleanupCompleted);
        Assert.Contains(reason, result.Detail); Assert.Equal(0, fixture.Created);
        Assert.Null(fixture.Engine.ComputeGate); Assert.Empty(fixture.Workspaces);
    }

    [Fact]
    public async Task CancelledStalledSubmissionDrainsEveryQueuedRequestWithoutDeclaringNativeWarmup()
    {
        using var fixture = new Fixture { StallThirdStream = true };
        using var cancellation = new CancellationTokenSource(TimeSpan.FromMilliseconds(100));
        var result = await fixture.Run(3, cancellation.Token).WaitAsync(TimeSpan.FromSeconds(3));
        Assert.False(result.Warmed); Assert.False(result.NativeBatchAccepted);
        Assert.True(result.CleanupCompleted); Assert.False(result.CleanupPending);
        Assert.Empty(fixture.Workspaces); Assert.Equal(3, fixture.Released);
        Assert.Null(fixture.Engine.ComputeGate);
        Assert.Equal(0, fixture.Engine.RunningCount + fixture.Engine.WaitingCount);
        Assert.Equal(0, fixture.Budget.Snapshot().Single().Committed + fixture.Budget.Snapshot().Single().Reserved);
    }

    [Fact]
    public async Task FusedDispatchFailureIsReportedAndCannotLeaveActiveRequestsOrChargedChildren()
    {
        using var fixture = new Fixture(); fixture.Model.FailJoint = true;
        var result = await fixture.Run(3).WaitAsync(TimeSpan.FromSeconds(5));
        Assert.False(result.RequestsCompleted); Assert.False(result.NativeBatchAccepted);
        Assert.True(result.CleanupCompleted); Assert.Empty(fixture.Workspaces);
        Assert.Equal(0, fixture.Engine.RunningCount + fixture.Engine.WaitingCount);
        Assert.Equal(0, fixture.Budget.Snapshot().Single().Committed);
    }

    [Fact]
    public async Task FailedPhysicalSequenceReleaseRetainsOwnedWorkspaceUntilTheEngineSuccessfullyRetriesRelease()
    {
        using var fixture = new Fixture(); fixture.Model.ReleaseFailuresRemaining = 1;
        var result = await fixture.Run(2, drain: TimeSpan.FromMilliseconds(30)).WaitAsync(TimeSpan.FromSeconds(3));
        Assert.True(result.RequestsCompleted); Assert.True(result.NativeBatchAccepted);
        Assert.False(result.Warmed); Assert.False(result.CleanupCompleted); Assert.True(result.CleanupPending);
        Assert.Equal(0, fixture.Engine.RunningCount + fixture.Engine.WaitingCount);
        Assert.Equal(1, fixture.Engine.PendingSequenceReleaseCount);
        Assert.Equal(2, fixture.Workspaces.Count); Assert.Equal(0, fixture.Released);
        Assert.True(fixture.Budget.Snapshot().Single().Committed > 0);
        fixture.Engine.Dispose();
        await fixture.Cleaned.Task.WaitAsync(TimeSpan.FromSeconds(3));
        Assert.Equal(0, fixture.Engine.PendingSequenceReleaseCount);
        Assert.Empty(fixture.Workspaces); Assert.Equal(2, fixture.Released);
        Assert.Equal(0, fixture.Budget.Snapshot().Single().Committed + fixture.Budget.Snapshot().Single().Reserved);
    }

    private static int ArgMax(float[] values)
    { int best = 0; for (int i = 1; i < values.Length; ++i) if (values[i] > values[best]) best = i; return best; }

    private sealed class Fixture : IDisposable
    {
        internal readonly MemoryBudget Budget;
        internal readonly BatchOracle Model = new();
        internal readonly InferenceEngine Engine;
        internal readonly ConcurrentDictionary<string, string> Workspaces = new();
        internal readonly ConcurrentDictionary<string, int[]> Results = new();
        internal readonly TaskCompletionSource Cleaned = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal bool FailSecondStream, StallThirdStream, InjectForeignRequest;
        internal Func<string, int[]> PromptFactory = _ => [1, 2, 3, 4];
        internal int Created, Released;
        internal Fixture(int maxRunning = 3, bool jointDecode = true, bool admitted = true, int maxBatched = 64,
            long memoryBytes = 1024)
        {
            Budget = new([new("memory", memoryBytes)]);
            Engine = new(Model, new SchedulerConfig
            {
                BlockSize = 8, NumBlocks = 128, MaxNumRunningSequences = maxRunning, MaxNumBatchedTokens = maxBatched,
                MaxPrefillChunkSize = 8, PrefillChunkTokenLimit = 8, EnablePrefixCaching = false, StopRepetition = false,
                ExecutionOptionsOverride = new() { BatchedPathDisabled = true, PerSeqFusedEnabled = true,
                    BatchedFusedDecodeEnabled = jointDecode, RetainedFusedCacheBudget = 0, KvHolderPoolMax = 0 },
                MemoryAdmission = admitted ? new RequestMemoryAdmission(Budget, _ => [new("memory", 32)])
                {
                    UsePerSequenceHolders = true, BeforeAdmission = _ => [new("memory", 1024)],
                    EnterSerialExecution = Model.Enter, EnterBatchedExecution = _ => Model.EnterJoint(),
                } : null
            });
        }
        internal Task<JointDecodeStartupWarmup.Result> Run(int width, CancellationToken cancellation = default, TimeSpan? drain = null)
            => JointDecodeStartupWarmup.RunAsync(Engine, Frames, Create, Release, width, false, null,
                cancellation, TimeSpan.FromSeconds(4), drain ?? TimeSpan.FromSeconds(2));
        private string Create()
        {
            string id = "warm-" + Interlocked.Increment(ref Created);
            Assert.True(Workspaces.TryAdd(id, Directory.CreateTempSubdirectory("ts-joint-warm-").FullName));
            return id;
        }
        private void Release(string id)
        {
            Assert.False(Results.ContainsKey(id) && Engine.RunningCount != 0);
            Assert.True(Workspaces.TryRemove(id, out var path));
            Directory.Delete(path, recursive: true);
            if (Interlocked.Increment(ref Released) == Created) Cleaned.TrySetResult();
        }
        private async IAsyncEnumerable<object> Frames(JsonElement body, [EnumeratorCancellation] CancellationToken cancellation)
        {
            string id = body.GetProperty("sessionId").GetString()!;
            Assert.True(Workspaces.ContainsKey(id));
            Assert.Equal(2, body.GetProperty("maxTokens").GetInt32());
            Assert.False(body.TryGetProperty("tools", out _)); Assert.False(body.TryGetProperty("skills", out _));
            await Task.Yield();
            if (FailSecondStream && id == "warm-2") { yield return new { done = true, error = "injected stream refusal" }; yield break; }
            if (StallThirdStream && id == "warm-3") await Task.Delay(Timeout.Infinite, cancellation);
            var sequence = new SequenceState(id, PromptFactory(id), 2, 8, SamplingConfig.Greedy,
                userTag: new ChatSession(id));
            var handle = Engine.SubmitRequest(sequence, cancellation);
            if (InjectForeignRequest && id == "warm-3")
                Engine.SubmitRequest(new SequenceState("foreign", [1, 2, 3, 4], 2, 8, SamplingConfig.Greedy));
            await foreach (var token in handle.Tokens.ReadAllAsync(cancellation)) yield return new { token };
            await handle.Completion;
            Results[id] = sequence.OutputTokens.ToArray();
            yield return new { done = true, aborted = cancellation.IsCancellationRequested, error = sequence.Error?.Message };
        }
        public void Dispose()
        {
            Model.Continue.Set(); Engine.Dispose(); Model.Dispose();
            foreach (var pair in Workspaces) { Directory.Delete(pair.Value, true); Workspaces.TryRemove(pair.Key, out _); }
            Assert.Equal(0, Budget.Snapshot().Single().Committed + Budget.Snapshot().Single().Reserved);
        }
    }

    private sealed class BatchOracle() : OracleModel(new OracleTraits
    {
        Name = "joint-warmup", Class = FamilyClass.R, EndState = EndStateSupport.CopyAndDonate,
        CanCaptureCopy = true, DeviceDirtyOnForward = true, Truncation = TruncationKind.None, Pages = PageSupport.None,
    }, 8), IModelArchitecture, IBatchedPagedModel
    {
        private SequenceState _owner;
        private bool _joint;
        private ulong _id;
        private readonly Dictionary<string, BudgetReservation> _children = new();
        internal readonly List<int> Widths = new();
        internal readonly TaskCompletionSource Started = new(TaskCreationOptions.RunContinuationsAsynchronously);
        internal readonly ManualResetEventSlim Continue = new();
        internal bool Pause, Decline, FailJoint, RefuseThirdSlotOnce;
        internal int ReleaseFailuresRemaining;
        public FusedBatchExecutionEvidence LastAcceptedFusedBatch { get; private set; }
        internal IDisposable Enter(SequenceState sequence)
        { Assert.Null(_owner); _owner = sequence; return new ActionScope(() => _owner = null); }
        internal IDisposable EnterJoint()
        { Assert.Null(_owner); Assert.False(_joint); _joint = true; return new ActionScope(() => _joint = false); }
        public new bool BindSequenceCache(string id)
        {
            Assert.Equal(id, _owner?.RequestId);
            if (RefuseThirdSlotOnce && id == "warm-3")
            {
                RefuseThirdSlotOnce = false;
                throw new SequenceSlotUnavailableException("injected temporarily unavailable third holder");
            }
            if (!_children.ContainsKey(id))
            {
                var child = _owner!.MemoryEnvelope!.TryTake([new("memory", 8)]);
                Assert.NotNull(child); child!.Commit(); _children[id] = child;
            }
            return base.BindSequenceCache(id);
        }
        public new float[] Forward(int[] tokens)
        {
            Assert.NotNull(_owner); Started.TrySetResult();
            if (Pause && !Continue.Wait(TimeSpan.FromSeconds(4))) throw new TimeoutException("CPU oracle compute barrier");
            return base.Forward(tokens);
        }
        public bool CanBatchDecode(string id, int position) => HasFusedSequenceCache(id);
        public bool TryForwardBatchedFusedDecode(IReadOnlyList<string> ids, int[] tokens, int[] positions, float[][] logits)
        {
            if (Decline) return false;
            Assert.True(_joint); Assert.Null(_owner);
            if (FailJoint) throw new InvalidOperationException("injected joint dispatch failure");
            Widths.Add(ids.Count);
            for (int row = 0; row < ids.Count; ++row)
            {
                base.BindSequenceCache(ids[row]); Assert.Equal(positions[row], ActiveLength);
                logits[row] = base.Forward([tokens[row]]);
            }
            LastAcceptedFusedBatch = new(++_id, ids.Count, 8, 0, 1);
            return true;
        }
        public new void OnSequenceReleased(string id)
        {
            if (ReleaseFailuresRemaining > 0)
            {
                --ReleaseFailuresRemaining;
                throw new InvalidOperationException("injected physical holder release failure");
            }
            base.OnSequenceReleased(id); if (_children.Remove(id, out var child)) child.Dispose();
        }
        private sealed class ActionScope(Action dispose) : IDisposable { public void Dispose() => dispose(); }
    }
}
