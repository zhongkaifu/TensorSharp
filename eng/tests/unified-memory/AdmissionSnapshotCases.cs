// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

static partial class Cases
{
    public static Task AtomicCapacityRefresh()
    {
        var budget = new MemoryBudget([new("ram", 128), new("gpu", 256)]);
        using var owner = budget.Reserve([new("gpu", 128)]);
        owner.Commit();
        var signal = budget.ChangeSignal;
        Check.True(!budget.TrySetCapacities([new("ram", 64), new("gpu", 64)]));
        Check.Equal(128L, budget.Snapshot().Single(p => p.Pool == "ram").Capacity);
        Check.True(!signal.IsCompleted);
        Check.True(budget.TrySetCapacities([new("ram", 128), new("gpu", 256)]));
        Check.True(!signal.IsCompleted, "An unchanged refresh woke blocked admission");
        Check.Throws<ArgumentException>(() => budget.TrySetCapacities([new("ram", 64), new("ram", 96)]));
        Check.True(budget.TrySetCapacities([new("ram", 64), new("gpu", 128)]));
        Check.True(signal.IsCompleted);
        var release = budget.ChangeSignal;
        owner.Dispose();
        Check.True(release.IsCompleted);
        using var envelope = budget.Reserve([new("ram", 64)]);
        var childSignal = budget.ChangeSignal;
        using var child = envelope.TryTake([new("ram", 32)])!;
        child.Commit(); child.Dispose();
        Check.True(!childSignal.IsCompleted, "Credit returned to a request woke unrelated waiters");
        envelope.Dispose();
        Check.True(childSignal.IsCompleted);
        return Task.CompletedTask;
    }

    public static Task SnapshotRequestEnvelope()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-envelope-kv-" + Guid.NewGuid().ToString("N"));
        var budget = new MemoryBudget([new("ram", 192), new("ssd", 3 * 4096)]);
        try
        {
            using var storage = new PagedKvStorage(8, 64, KvSnapshotOptions.FromSharedBudget(budget, "ram", "ssd", root, 64));
            using var envelope = budget.Reserve([new("ram", 64), new("ssd", 3 * 4096)]);
            for (int i = 0; i < 3; i++)
            {
                using var page = storage.Acquire(i, ResourceAccess.Write, envelope);
                page.Span.Fill((byte)(17 + i));
            }
            for (int i = 0; i < 3; i++)
            {
                using var page = storage.Acquire(i, ResourceAccess.Read, envelope);
                Check.True(page.ReadOnlySpan.ToArray().All(b => b == 17 + i));
            }
            Check.True(storage.ResidencyStats!.Value.Spills >= 3);
            Check.True(budget.Snapshot().All(p => p.Available == 0));
            // A retained prefix remains physically charged after request release.
            envelope.Dispose();
            Check.True(budget.Snapshot().All(p => p.Reserved == 0));
            Check.True(budget.Snapshot().Single(p => p.Pool == "ssd").Committed > 0);
            using (var page = storage.Acquire(2, ResourceAccess.Write)) page.Span.Fill(91);
            using (var page = storage.Acquire(0)) Check.Equal((byte)17, page.ReadOnlySpan[0]);
            using (var page = storage.Acquire(2)) Check.Equal((byte)91, page.ReadOnlySpan[0]);
            storage.Dispose();
            Check.True(budget.Snapshot().All(p => p.Available == p.Capacity));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        return Task.CompletedTask;
    }

    public static async Task PrefetchLeaseConflict()
    {
        await using var f = new Fixture();
        var key = f.Add("busy", mutable: true);
        using var writer = await f.Scheduler.AcquireAsync(key, f.Host.Location, ResourceAccess.Write);
        var pending = f.Scheduler.TryPrefetchAsync(key, f.Host.Location).AsTask();
        try
        {
            Check.True(await Task.WhenAny(pending, Task.Delay(2000)) == pending,
                "Speculative prefetch waited for the active execution lease it overlaps");
            Check.True(!await pending);
        }
        finally { writer.Dispose(); await pending.WaitAsync(TimeSpan.FromSeconds(2)); }
    }

    public static Task AdaptiveSnapshotAdmission()
    {
        var budget = new MemoryBudget([new("ram", 640), new("ssd", 32 * 4096)]);
        using var other = budget.Reserve([new("ram", 128)]);
        other.Commit();
        var snapshots = KvSnapshotOptions.FromSharedBudget(budget, "ram", "ssd", Path.GetTempPath(), 64);
        var admission = RequestMemoryAdmission.ForKvSnapshots(snapshots, 32, 8, 2);
        var longRequest = new SequenceState("long", Enumerable.Range(1, 35).ToArray(), 17, 8, SamplingConfig.Greedy);
        var peak = admission.EstimatePeak(longRequest);
        Check.Equal(192L, peak.Single(c => c.Pool == "ram").Bytes);
        Check.Equal(7 * 4096L, peak.Single(c => c.Pool == "ssd").Bytes);
        var shortRequest = new SequenceState("short", new[] { 1, 2, 3 }, 5, 8, SamplingConfig.Greedy);
        peak = admission.EstimatePeak(shortRequest);
        Check.Equal(64L, peak.Single(c => c.Pool == "ram").Bytes);
        Check.Equal(0L, peak.Single(c => c.Pool == "ssd").Bytes);
        var withWork = RequestMemoryAdmission.ForKvSnapshots(snapshots, 32, 8, 2,
            additionalPeak: _ => [new("ram", 128)]);
        peak = withWork.EstimatePeak(longRequest);
        Check.Equal(192L, peak.Where(c => c.Pool == "ram").Sum(c => c.Bytes));
        Check.Throws<ArgumentOutOfRangeException>(() => RequestMemoryAdmission.ForKvSnapshots(snapshots, 0, 8, 2));
        Check.Throws<ArgumentException>(() => RequestMemoryAdmission.ForKvSnapshots(new(192, 4096, Path.GetTempPath(), 64), 32, 8, 2));
        return Task.CompletedTask;
    }

    public static Task SnapshotEnvelopeRefusal()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-envelope-refuse-" + Guid.NewGuid().ToString("N"));
        var budget = new MemoryBudget([new("ram", 192), new("ssd", 8192)]);
        try
        {
            using var storage = new PagedKvStorage(3, 64, KvSnapshotOptions.FromSharedBudget(budget, "ram", "ssd", root, 64));
            using var envelope = budget.Reserve([new("ram", 64), new("ssd", 0)]);
            using (var page = storage.Acquire(0, ResourceAccess.Write, envelope)) page.Span.Fill(47);
            Check.Throws<MemoryPressureException>(() => storage.Acquire(1, ResourceAccess.Write, envelope).Dispose());
            using (var page = storage.Acquire(0, ResourceAccess.Read, envelope)) Check.Equal((byte)47, page.ReadOnlySpan[0]);
            Check.Equal(8192L, budget.Snapshot().Single(p => p.Pool == "ssd").Available);
            storage.Dispose(); envelope.Dispose();
            Check.True(budget.Snapshot().All(p => p.Available == p.Capacity));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        return Task.CompletedTask;
    }

    public static async Task SharedSnapshotCancellation()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-envelope-cancel-" + Guid.NewGuid().ToString("N"));
        var budget = new MemoryBudget([new("ram", 256), new("ssd", 16 * 4096)]);
        try
        {
            using var model = new SnapshotEngineModel();
            using var engine = new InferenceEngine(model, new SchedulerConfig
            {
                BlockSize = 8, NumBlocks = 64, MaxNumRunningSequences = 2,
                MaxNumBatchedTokens = 16, MaxPrefillChunkSize = 8, DecodeQuantumTokens = 1,
                EnablePrefixCaching = false, StopRepetition = false,
                KvSnapshots = KvSnapshotOptions.FromSharedBudget(budget, "ram", "ssd", root, 64),
                MemoryAdmission = new(budget, _ => [new("ram", 64), new("ssd", 8 * 4096)])
            });
            var cancelled = new SequenceState("cancel", Enumerable.Range(1, 35).ToArray(), 17, 8, SamplingConfig.Greedy);
            var survivor = new SequenceState("survivor", Enumerable.Range(41, 35).ToArray(), 17, 8, SamplingConfig.Greedy);
            model.BeforeForward = () =>
            {
                if (model.Extractions == 0) return;
                model.BeforeForward = null;
                engine.Abort(cancelled.RequestId);
            };
            InferenceRequestHandle first, second;
            lock (((IModelArchitecture)model).GpuComputeLock)
            { first = engine.SubmitRequest(cancelled); second = engine.SubmitRequest(survivor); }
            await Task.WhenAll(first.Completion, second.Completion).WaitAsync(TimeSpan.FromSeconds(10));
            Check.Equal(SequenceStatus.FinishedAborted, cancelled.Status);
            Check.True(survivor.OutputTokens.SequenceEqual(SnapshotEngineModel.Generate(survivor.PromptTokens, 17)));
            var followup = new SequenceState("followup", Enumerable.Range(81, 35).ToArray(), 17, 8, SamplingConfig.Greedy);
            await engine.SubmitRequest(followup).Completion.WaitAsync(TimeSpan.FromSeconds(10));
            Check.True(followup.OutputTokens.SequenceEqual(SnapshotEngineModel.Generate(followup.PromptTokens, 17)));
            engine.Dispose();
            Check.True(budget.Snapshot().All(p => p.Available == p.Capacity));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }

    public static async Task SnapshotOverwrite()
    {
        await using var f = new Fixture(ram: Fixture.Page);
        var first = f.Add("overwrite-a", mutable: true);
        var second = f.Add("overwrite-b", mutable: true);
        byte[] initial = Enumerable.Repeat((byte)31, Fixture.Page).ToArray();
        byte[] replacement = Enumerable.Repeat((byte)73, Fixture.Page).ToArray();
        using (var lease = await f.Scheduler.AcquireAsync(first, f.Host.Location, ResourceAccess.Write))
            await lease.WriteAsync(0, initial);
        using (var held = await f.Scheduler.AcquireAsync(second, f.Host.Location))
            await Check.ThrowsAsync<MemoryPressureException>(async () =>
                (await f.Scheduler.AcquireForOverwriteAsync(first, f.Host.Location)).Dispose());
        byte[] restored = await f.Read(first);
        Check.Bytes(initial, restored); // Failed overwrite kept authoritative SSD bytes.
        await f.Read(second); // Evict the original again.
        long copied = f.Scheduler.GetStats().TransferBytes;
        using (var lease = await f.Scheduler.AcquireForOverwriteAsync(first, f.Host.Location))
            await lease.WriteAsync(0, replacement);
        Check.Equal(copied, f.Scheduler.GetStats().TransferBytes);
        await f.Read(second);
        restored = await f.Read(first);
        Check.Bytes(replacement, restored); // New spill checksum/version round trip.
    }

    public static async Task EngineSharedSnapshotAdmission()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-engine-shared-" + Guid.NewGuid().ToString("N"));
        var budget = new MemoryBudget([new("ram", 384), new("ssd", 3 * 8 * 4096)]);
        try
        {
            using var other = budget.Reserve([new("ram", 64)]);
            other.Commit();
            using var model = new SnapshotEngineModel();
            using var engine = new InferenceEngine(model, new SchedulerConfig
            {
                BlockSize = 8, NumBlocks = 64, MaxNumRunningSequences = 3,
                MaxNumBatchedTokens = 16, MaxPrefillChunkSize = 8, DecodeQuantumTokens = 1,
                EnablePrefixCaching = true, StopRepetition = false,
                KvSnapshots = KvSnapshotOptions.FromSharedBudget(budget, "ram", "ssd", root, 64),
                MemoryAdmission = new(budget, _ => [new("ram", 64), new("ssd", 8 * 4096)])
            });
            var sequences = Enumerable.Range(0, 3).Select(i => new SequenceState("shared-" + i,
                Enumerable.Range(1 + i * 41, 35).ToArray(), 17, 8, SamplingConfig.Greedy)).ToArray();
            InferenceRequestHandle[] handles;
            lock (((IModelArchitecture)model).GpuComputeLock)
                handles = sequences.Select(s => engine.SubmitRequest(s)).ToArray();
            await Task.WhenAll(handles.Select(h => h.Completion)).WaitAsync(TimeSpan.FromSeconds(10));
            foreach (var seq in sequences)
                Check.True(seq.OutputTokens.SequenceEqual(SnapshotEngineModel.Generate(seq.PromptTokens, 17)));
            Check.True(engine.SnapshotResidencyStats!.Value.Spills > 0);
            Check.True(budget.Snapshot().All(p => p.Reserved + p.Committed <= p.Capacity));
            engine.Dispose();
            Check.Equal(64L, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
            Check.True(budget.Snapshot().All(p => p.Reserved == 0));
            other.Dispose();
            Check.True(budget.Snapshot().All(p => p.Available == p.Capacity));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
    }
}
