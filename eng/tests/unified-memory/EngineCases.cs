// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Buffers.Binary;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

static partial class Cases
{
    public static Task SnapshotRoute()
    {
        using var model = new SnapshotEngineModel();
        var cfg = SnapshotConfig(Path.GetTempPath());
        var caps = ExecutionCapabilities.FromModel(model);
        foreach (int count in new[] { 1, 2, 4 })
        {
            var plan = ExecutionPlanner.PlanStep(caps, ExecutionOptions.Default, cfg,
                new ExecutionStepFeatures { SequenceCount = count });
            Check.Equal(ExecutionPathKind.PerSequence, plan.Candidates.Single());
        }
        Check.True(ExecutionPlanner.BuildCapabilityReport(caps, ExecutionOptions.Default, cfg)
            .Contains("live device KV, weights and model scratch are not covered"));
        model.SnapshotsSupported = false;
        Check.Throws<NotSupportedException>(() => new InferenceEngine(model, cfg));
        model.SnapshotsSupported = true;
        model.CrossSequenceSupported = false;
        Check.Throws<NotSupportedException>(() => new InferenceEngine(model, cfg));
        Check.Throws<NotSupportedException>(() => new PagedKvStorage(4, 0, cfg.KvSnapshots));
        return Task.CompletedTask;
    }

    private static SchedulerConfig SnapshotConfig(string root) => new()
    {
        BlockSize = 8, NumBlocks = 64, MaxNumRunningSequences = 4,
        MaxNumBatchedTokens = 16, MaxPrefillChunkSize = 8,
        SoloPrefillChunkSize = 8, DecodeQuantumTokens = 1,
        EnablePrefixCaching = false, StopRepetition = false,
        // The 32-byte payloads allocate 64 aligned bytes. Only one page fits
        // beside capture scratch and transfer staging, so every swap spills.
        KvSnapshots = new(3 * 64, 64 * 4096, root, 64),
    };

    public static async Task EngineSnapshotIsolation()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-engine-kv-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var model = new SnapshotEngineModel();
            using var engine = new InferenceEngine(model, SnapshotConfig(root));
            var sequences = Enumerable.Range(0, 3).Select(i => new SequenceState("swap-" + i,
                Enumerable.Range(1 + i * 41, 35).ToArray(), 17, 8, SamplingConfig.Greedy)).ToArray();
            InferenceRequestHandle[] handles;
            lock (((IModelArchitecture)model).GpuComputeLock)
                handles = sequences.Select(s => engine.SubmitRequest(s)).ToArray();
            await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(10));
            foreach (var seq in sequences)
            {
                Check.Equal(SequenceStatus.FinishedLengthCapped, seq.Status);
                Check.True(seq.OutputTokens.SequenceEqual(SnapshotEngineModel.Generate(seq.PromptTokens, 17)),
                    "Interleaved snapshot restore changed the deterministic model output");
            }
            Check.True(model.Extractions > 0 && model.Injections > 0);
            Check.True(model.FullCaptureCounts.Count > 0 && model.FullCaptureCounts.Values.All(n => n == 1),
                "An immutable full snapshot was extracted again during an ownership swap");
            Check.True(model.PartialExtractions > sequences.Length,
                "The growing partial tail must still refresh across repeated ownership swaps");
            Check.True(engine.SnapshotResidencyStats!.Value.Spills > 0);
            Check.True(engine.SnapshotResidencyStats!.Value.Loads > 0);
            Check.True(engine.SnapshotMemoryUsage!.All(p => p.Reserved + p.Committed <= p.Capacity));
            engine.Dispose();
            Check.True(engine.SnapshotMemoryUsage!.All(p => p.Reserved == 0 && p.Committed == 0));
            Check.True(!Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).Any());
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    public static Task SchedulerShrinkingBudget()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 100) });
        using var occupied = budget.Reserve(new[] { new MemoryCharge("ram", 50) });
        var cfg = new SchedulerConfig { BlockSize = 8, EnablePrefixCaching = false,
            MemoryAdmission = new(budget, seq => new[] { new MemoryCharge("ram", seq.RequestId == "large" ? 80 : 40) }) };
        var scheduler = new ContinuousBatchScheduler(cfg, new BlockPool(16, 8, 0));
        var large = MemorySequence("large"); var small = MemorySequence("small");
        scheduler.Submit(large); scheduler.Submit(small);
        Check.True(scheduler.Schedule().IsEmpty && scheduler.MemoryAdmissionBlocked);
        Check.True(budget.TrySetCapacity("ram", 50));
        var rejected = scheduler.Schedule();
        Check.Equal(SequenceStatus.FinishedError, large.Status);
        Check.True(large.Error is MemoryPressureException);
        Check.True(rejected.FinishedRequestIds.Contains(large.RequestId));
        occupied.Dispose();
        Check.Equal(small, scheduler.Schedule().ScheduledWork.Single().Sequence);
        scheduler.Abort(small.RequestId);
        scheduler.NotifyMemoryReleased(small.RequestId);
        Check.Equal(50L, budget.Snapshot().Single().Available);
        return Task.CompletedTask;
    }

    public static async Task EngineShrinkingBudget()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 100) });
        using var occupied = budget.Reserve(new[] { new MemoryCharge("ram", 50) });
        using var model = new SnapshotEngineModel();
        using var engine = new InferenceEngine(model, new SchedulerConfig { BlockSize = 8,
            EnablePrefixCaching = false,
            MemoryAdmission = new(budget, _ => new[] { new MemoryCharge("ram", 80) }) });
        var seq = MemorySequence("shrunk");
        var handle = engine.SubmitRequest(seq);
        await AwaitCondition(() => engine.WaitingCount == 1);
        Check.True(budget.TrySetCapacity("ram", 50));
        await Check.ThrowsAsync<MemoryPressureException>(() => handle.Completion.WaitAsync(TimeSpan.FromSeconds(5)));
        Check.Equal(SequenceStatus.FinishedError, seq.Status);
        Check.Equal(0, engine.WaitingCount);
        Check.Equal(0, model.Extractions);
    }

    public static async Task EngineReleaseRecovery()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 128) });
        var host = new HostMemoryBackend("ram");
        using var model = new SnapshotEngineModel { ReleaseFailuresRemaining = 2 };
        using var engine = new InferenceEngine(model, new SchedulerConfig { BlockSize = 8,
            EnablePrefixCaching = false,
            MemoryAdmission = new(budget, _ => new[] { new MemoryCharge("ram", 128) }) });
        var seq = MemorySequence("release-recovery");
        model.BeforeForward = () =>
        {
            model.BeforeForward = null;
            model.RequestAllocation = seq.MemoryEnvelope!.TryTake(host.GetAllocationCharges(64))!;
            model.RequestBuffer = host.AllocateAsync(64).GetAwaiter().GetResult();
            model.RequestAllocation.Commit();
        };
        await engine.SubmitRequest(seq).Completion.WaitAsync(TimeSpan.FromSeconds(5));
        await AwaitCondition(() => Volatile.Read(ref model.ReleaseAttempts) >= 1);

        // The first release failed after generation completed. A first disposal
        // retries the already-finished owner and also fails; neither may return
        // admission credit while the native host buffer remains alive.
        Check.Throws<AggregateException>(engine.Dispose);
        Check.Equal(2, model.ReleaseAttempts);
        Check.True(seq.MemoryEnvelope != null && model.RequestBuffer!.Pointer != 0);
        var held = budget.Snapshot().Single();
        Check.Equal(64L, held.Committed);
        Check.Equal(64L, held.Reserved);
        Check.True(budget.TryReserve(new[] { new MemoryCharge("ram", 128) }) == null);
        Check.Throws<ObjectDisposedException>(() => engine.SubmitRequest(MemorySequence("after-shutdown")));

        engine.Dispose();
        Check.Equal(3, model.ReleaseAttempts);
        Check.True(seq.MemoryEnvelope == null);
        Check.Throws<ObjectDisposedException>(() => _ = model.RequestBuffer!.Pointer);
        using var availableAgain = budget.Reserve(new[] { new MemoryCharge("ram", 128) });
        engine.Dispose();
        Check.Equal(3, model.ReleaseAttempts);
    }

    public static Task SchedulerPageReleaseRecovery()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-release-pages-" + Guid.NewGuid().ToString("N"));
        try
        {
            var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 64) });
            var cfg = new SchedulerConfig { BlockSize = 8, EnablePrefixCaching = false,
                MemoryAdmission = new(budget, _ => new[] { new MemoryCharge("ram", 64) }) };
            var pool = new BlockPool(4, 8, 32, new(192, 4 * 4096, root, 64));
            using var storage = pool.Storage;
            var scheduler = new ContinuousBatchScheduler(cfg, pool);
            var seq = new SequenceState("partial-page-release", Enumerable.Range(1, 20).ToArray(), 2, 8, SamplingConfig.Greedy);
            scheduler.Submit(seq);
            scheduler.Schedule();
            Check.Equal(3, seq.BlockTable.NumBlocks);
            foreach (var block in seq.BlockTable.Blocks)
            {
                using var bytes = storage.Acquire(block.Id, ResourceAccess.Write);
                bytes.Span.Fill((byte)(block.Id + 1));
            }
            int pinnedId = seq.BlockTable.Blocks[1].Id;
            using (var held = storage.Acquire(pinnedId))
            {
                Check.Throws<InvalidOperationException>(() => scheduler.Abort(seq.RequestId));
                Check.Equal(2, seq.BlockTable.NumBlocks); // Last page freed, middle still pinned.
                Check.Equal(pinnedId, seq.BlockTable.Blocks[1].Id);
                Check.Equal(1, pool.GetBlock(pinnedId).RefCount);
                Check.Equal(2, pool.NumFreeBlocks);
                Check.Equal(0L, budget.Snapshot().Single().Available);
                Check.Equal(SequenceStatus.Running, seq.Status);
            }
            Check.True(scheduler.Abort(seq.RequestId));
            scheduler.NotifyMemoryReleased(seq.RequestId);
            Check.Equal(4, pool.NumFreeBlocks);
            Check.Equal(0, storage.ResidencyStats!.Value.Resources);
            Check.Equal(0, seq.BlockTable.NumBlocks);
            Check.Equal(64L, budget.Snapshot().Single().Available);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        return Task.CompletedTask;
    }

    public static async Task EnginePageReleaseRecovery()
    {
        // Exercise both worker handoffs: completion after a forward, and Abort
        // from its command queue. A live page lease makes each release fail.
        foreach (bool abort in new[] { false, true })
        {
            string root = Path.Combine(Path.GetTempPath(), "ts-worker-release-" + Guid.NewGuid().ToString("N"));
            try
            {
                var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 64) });
                using var model = new SnapshotEngineModel();
                using var engine = new InferenceEngine(model, new SchedulerConfig
                {
                    BlockSize = 8, NumBlocks = 8, MaxNumRunningSequences = 1,
                    MaxNumBatchedTokens = 32, MaxPrefillChunkSize = 32, SoloPrefillChunkSize = 32,
                    EnablePrefixCaching = false, StopRepetition = false,
                    KvSnapshots = new(192, 8 * 4096, root, 64),
                    MemoryAdmission = new(budget, _ => new[] { new MemoryCharge("ram", 64) }),
                });
                var pool = (BlockPool)typeof(InferenceEngine).GetField("_pool",
                    System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic)!.GetValue(engine)!;
                var seq = new SequenceState("worker-release", Enumerable.Range(1, 20).ToArray(),
                    abort ? 32 : 1, 8, SamplingConfig.Greedy);
                KvSnapshotLease? pinned = null;
                model.BeforeForward = () =>
                {
                    model.BeforeForward = null;
                    Check.Equal(3, seq.BlockTable.NumBlocks);
                    int middle = seq.BlockTable.Blocks[1].Id;
                    using (var write = pool.Storage.Acquire(middle, ResourceAccess.Write)) write.Span.Fill(1);
                    pinned = pool.Storage.Acquire(middle);
                    if (abort) engine.Abort(seq.RequestId);
                };
                InferenceRequestHandle active;
                InferenceRequestHandle waiting;
                lock (((IModelArchitecture)model).GpuComputeLock)
                {
                    active = engine.SubmitRequest(seq);
                    waiting = engine.SubmitRequest(MemorySequence("waiting-for-worker"));
                }
                int completedForwardCalls = 0;
                long completedSteps = 0;
                try
                {
                    await Check.ThrowsAsync<InvalidOperationException>(() => active.Completion.WaitAsync(TimeSpan.FromSeconds(5)));
                    await Check.ThrowsAsync<InvalidOperationException>(() => waiting.Completion.WaitAsync(TimeSpan.FromSeconds(5)));
                    Check.Throws<ObjectDisposedException>(() => engine.SubmitRequest(MemorySequence("after-worker-failure")));
                    Check.Throws<InvalidOperationException>(engine.Dispose); // Joins worker; lease still owns the page.
                    completedForwardCalls = model.ForwardCalls;
                    Check.True(completedForwardCalls > 0);
                    completedSteps = engine.TotalStepsRun;
                    Check.True(completedSteps > 0);
                    Check.Equal(2, seq.BlockTable.NumBlocks);
                    Check.Equal(SequenceStatus.Running, seq.Status);
                    Check.True(seq.MemoryEnvelope != null);
                    Check.Equal(0L, budget.Snapshot().Single().Available);
                }
                finally { pinned?.Dispose(); }
                engine.Dispose();
                Check.Equal(0, seq.BlockTable.NumBlocks);
                Check.True(seq.MemoryEnvelope == null);
                Check.Equal(64L, budget.Snapshot().Single().Available);
                Check.Equal(0, pool.Storage.ResidencyStats!.Value.Resources);
                Check.Equal(completedForwardCalls, model.ForwardCalls);
                Check.Equal(completedSteps, engine.TotalStepsRun);
            }
            finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        }
    }

    private static async Task AwaitCondition(Func<bool> condition)
    {
        using var timeout = new CancellationTokenSource(TimeSpan.FromSeconds(5));
        while (!condition()) await Task.Delay(5, timeout.Token);
    }

    public static async Task EnginePartialDisposeRecovery()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-partial-dispose-" + Guid.NewGuid().ToString("N"));
        try
        {
            var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 128) });
            using var model = new SnapshotEngineModel();
            using var engine = new InferenceEngine(model, new SchedulerConfig
            {
                BlockSize = 8, NumBlocks = 8, MaxNumRunningSequences = 2,
                MaxNumBatchedTokens = 64, MaxPrefillChunkSize = 32,
                EnablePrefixCaching = false, KvSnapshots = new(192, 8 * 4096, root, 64),
                MemoryAdmission = new(budget, _ => new[] { new MemoryCharge("ram", 64) }),
            });
            var gate = new ComputeGate();
            gate.Close();
            engine.ComputeGate = gate;
            var first = new SequenceState("dispose-first", Enumerable.Range(1, 20).ToArray(), 2, 8, SamplingConfig.Greedy);
            var second = new SequenceState("dispose-second", Enumerable.Range(31, 20).ToArray(), 2, 8, SamplingConfig.Greedy);
            InferenceRequestHandle[] handles;
            lock (((IModelArchitecture)model).GpuComputeLock)
                handles = new[] { engine.SubmitRequest(first), engine.SubmitRequest(second) };
            await AwaitCondition(() => engine.StepsHeldByGate > 0);
            var flags = System.Reflection.BindingFlags.Instance | System.Reflection.BindingFlags.NonPublic;
            var pool = (BlockPool)typeof(InferenceEngine).GetField("_pool", flags)!.GetValue(engine)!;
            var scheduler = (ContinuousBatchScheduler)typeof(InferenceEngine).GetField("_scheduler", flags)!.GetValue(engine)!;
            // Admit both owners while the worker is parked, then fault only the
            // second owner's page release. No model forward is needed for this handoff.
            lock (((IModelArchitecture)model).GpuComputeLock) scheduler.Schedule();
            Check.Equal(2, scheduler.RunningCount);
            int middle = second.BlockTable.Blocks[1].Id;
            using (var write = pool.Storage.Acquire(middle, ResourceAccess.Write)) write.Span.Fill(2);
            using (var pinned = pool.Storage.Acquire(middle))
            {
                Check.Throws<InvalidOperationException>(engine.Dispose);
                Check.True(first.MemoryEnvelope == null, "A later owner's failure orphaned the earlier completed release");
                Check.True(second.MemoryEnvelope != null);
                Check.Equal(1, model.ReleaseAttempts);
                Check.Equal(64L, budget.Snapshot().Single().Available);
                foreach (var handle in handles)
                    await Check.ThrowsAsync<ObjectDisposedException>(() => handle.Completion);
            }
            engine.Dispose();
            Check.True(second.MemoryEnvelope == null);
            Check.Equal(2, model.ReleaseAttempts);
            Check.Equal(128L, budget.Snapshot().Single().Available);
            Check.Equal(0, pool.Storage.ResidencyStats!.Value.Resources);
            Check.Equal(0, model.ForwardCalls);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}

// Deterministic stateful model, not a GPU simulation or hardware-validation claim.
// Its logits depend on every prior token; corrupt/missing/mixed snapshots change output.
sealed class SnapshotEngineModel : IModelArchitecture, IBatchedPagedModel
{
    private readonly List<int> _history = new();
    public ModelConfig Config { get; } = new() { Architecture = "snapshot-test", VocabSize = 251 };
    public ITokenizer Tokenizer => null!;
    public IMultimodalInjector MultimodalInjector => null!;
    public IBackendExecutionPlan ExecutionPlan => null!;
    public bool SnapshotsSupported = true;
    public bool CrossSequenceSupported = true;
    public bool SupportsKVStateSnapshot => SnapshotsSupported;
    public bool SupportsCrossSequenceKvReuse => CrossSequenceSupported;
    public bool SupportsKVCacheTruncation => true;
    public bool SupportsPerSequenceFusedForward => true;
    public bool SupportsLinearKVMigration => true;
    public int Extractions;
    public int Injections;
    public int PartialExtractions;
    public Dictionary<string, int> FullCaptureCounts { get; } = new();
    public Action? BeforeForward;
    public int ForwardCalls;
    public int ReleaseFailuresRemaining;
    public int ReleaseAttempts;
    public BudgetReservation? RequestAllocation;
    public IResourceBuffer? RequestBuffer;
    public float[] Forward(int[] tokens)
    {
        ForwardCalls++;
        BeforeForward?.Invoke();
        _history.AddRange(tokens);
        var logits = new float[251];
        logits[Next(_history)] = 100;
        return logits;
    }
    private static int Next(IEnumerable<int> tokens)
    {
        uint hash = 2166136261;
        foreach (int token in tokens) hash = unchecked((hash ^ (uint)token) * 16777619);
        return (int)(hash % 251);
    }
    public static int[] Generate(IEnumerable<int> prompt, int count)
    {
        var history = prompt.ToList();
        var result = new int[count];
        for (int i = 0; i < count; i++) { result[i] = Next(history); history.Add(result[i]); }
        return result;
    }
    public void ResetKVCache() => _history.Clear();
    public void TruncateKVCache(int count) => _history.RemoveRange(count, _history.Count - count);
    public long ComputeKVBlockByteSize(int tokenCount) => tokenCount * sizeof(int);
    public bool TryExtractKVBlock(int startToken, int tokenCount, Span<byte> destination)
    {
        if (destination.Length != tokenCount * sizeof(int) || startToken + tokenCount > _history.Count) return false;
        for (int i = 0; i < tokenCount; i++)
            BinaryPrimitives.WriteInt32LittleEndian(destination.Slice(i * sizeof(int), sizeof(int)), _history[startToken + i]);
        Extractions++;
        if (tokenCount == 8)
        {
            string key = startToken + ":" + Convert.ToHexString(destination);
            FullCaptureCounts[key] = FullCaptureCounts.GetValueOrDefault(key) + 1;
        }
        else PartialExtractions++;
        return true;
    }
    public bool TryInjectKVBlock(int destToken, int tokenCount, ReadOnlySpan<byte> source)
    {
        if (destToken != _history.Count || source.Length != tokenCount * sizeof(int)) return false;
        for (int i = 0; i < tokenCount; i++)
            _history.Add(BinaryPrimitives.ReadInt32LittleEndian(source.Slice(i * sizeof(int), sizeof(int))));
        Injections++;
        return true;
    }
    public IReadOnlyList<float[]> ForwardBatch(BatchedForwardContext context)
        => throw new InvalidOperationException("Bounded snapshots bypassed by model-owned paged route");
    public bool BindSequenceCache(string requestId)
        => throw new InvalidOperationException("Bounded snapshots bypassed by fused holder route");
    public void OnSequenceReleased(string requestId)
    {
        Interlocked.Increment(ref ReleaseAttempts);
        if (ReleaseFailuresRemaining > 0)
        {
            ReleaseFailuresRemaining--;
            throw new IOException("Injected model release failure");
        }
        RequestBuffer?.Dispose();
        RequestAllocation?.Dispose();
    }
    public void Dispose() { RequestBuffer?.Dispose(); RequestAllocation?.Dispose(); }
}
