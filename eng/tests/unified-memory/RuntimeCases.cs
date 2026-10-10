// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp;
using TensorSharp.Memory;
using TensorSharp.Runtime;
using TensorSharp.Runtime.Paged;
using TensorSharp.Runtime.Scheduling;

static partial class Cases
{
    public static Task KvSnapshots()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-kv-test-" + Guid.NewGuid().ToString("N"));
        try
        {
            // Only one resident page, plus one capture scratch and one transfer page.
            using var storage = new PagedKvStorage(32, 4096, new(3 * 4096, 64 * 4096, root, 4096));
            for (int i = 0; i < 32; i++)
            {
                using var lease = storage.Acquire(i, ResourceAccess.Write);
                lease.Span.Fill((byte)(i + 1));
            }
            for (int round = 0; round < 2; round++)
            for (int i = 31; i >= 0; i--)
            {
                using var lease = storage.Acquire(i);
                Check.True(lease.ReadOnlySpan.ToArray().All(b => b == i + 1));
                Check.Throws<InvalidOperationException>(() => _ = lease.Span.Length);
            }
            Check.True(storage.ResidencyStats!.Value.Spills >= 31);
            Check.True(storage.MemoryUsage!.All(p => p.Reserved + p.Committed <= p.Capacity));
            Check.Throws<InvalidOperationException>(() => storage.GetSpan(0));
            using (var held = storage.Acquire(0))
            {
                Check.Throws<InvalidOperationException>(() => storage.ReleaseSlab(0));
                Check.Throws<InvalidOperationException>(storage.Dispose);
            }
            storage.ReleaseSlab(0);
            using (var fresh = storage.Acquire(0)) Check.True(fresh.ReadOnlySpan.ToArray().All(b => b == 0));
            for (int i = 0; i < 32; i++) storage.ReleaseSlab(i);
            Check.Equal(0, storage.ResidencyStats!.Value.Resources);
            storage.Dispose();
            Check.True(storage.MemoryUsage!.All(p => p.Committed == 0 && p.Reserved == 0));
            Check.Throws<ObjectDisposedException>(() => storage.Acquire(0));
            Check.True(!Directory.EnumerateFiles(root, "*", SearchOption.AllDirectories).Any());
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        return Task.CompletedTask;
    }

    public static Task KvPoolOwnership()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-kv-pool-" + Guid.NewGuid().ToString("N"));
        try
        {
            var pool = new BlockPool(4, 8, 4096, new(12288, 65536, root, 4096));
            using var storage = pool.Storage;
            var page = pool.AllocateNew(1)![0];
            using (var write = storage.Acquire(page.Id, ResourceAccess.Write)) write.Span.Fill(19);
            pool.Touch(page); // Retained prefix reference.
            pool.Free(page);
            Check.Equal(1, page.RefCount);
            using (var held = storage.Acquire(page.Id))
            {
                Check.Equal((byte)19, held.ReadOnlySpan[0]);
                Check.Throws<InvalidOperationException>(() => pool.Free(page));
                Check.Equal(1, page.RefCount);
                Check.Equal(3, pool.NumFreeBlocks);
            }
            pool.Free(page);
            Check.Equal(4, pool.NumFreeBlocks);
            Check.Equal(0, storage.ResidencyStats!.Value.Resources);
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        return Task.CompletedTask;
    }

    public static Task KvFullDisk()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-kv-full-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var storage = new PagedKvStorage(2, 4096, new(12288, 0, root, 4096));
            using (var first = storage.Acquire(0, ResourceAccess.Write)) first.Span.Fill(73);
            Check.Throws<MemoryPressureException>(() => storage.Acquire(1));
            using var preserved = storage.Acquire(0);
            Check.True(preserved.ReadOnlySpan.ToArray().All(b => b == 73));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        Check.Throws<ArgumentException>(() => new PagedKvStorage(4, 4096, new(4096, 0, root, 4096)));
        return Task.CompletedTask;
    }

    private static SequenceState MemorySequence(string id) => new(id, new[] { 1, 2 }, 2, 8, SamplingConfig.Greedy);
    public static Task SchedulerMemoryAdmission()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 100), new MemoryCharge("gpu:0", 100), new MemoryCharge("gpu:1", 60) });
        var cfg = new SchedulerConfig { BlockSize = 8, NumBlocks = 32, EnablePrefixCaching = false,
            MemoryAdmission = new(budget, _ => new[] { new MemoryCharge("ram", 40), new MemoryCharge("gpu:0", 60), new MemoryCharge("gpu:1", 40) }) };
        var pool = new BlockPool(32, 8, 0);
        var scheduler = new ContinuousBatchScheduler(cfg, pool);
        var first = MemorySequence("first"); var second = MemorySequence("second");
        scheduler.Submit(first); scheduler.Submit(second);
        var output = scheduler.Schedule();
        Check.Equal(1, output.ScheduledWork.Count);
        Check.True(first.MemoryEnvelope != null && second.MemoryEnvelope == null);
        Check.Throws<InvalidOperationException>(() => scheduler.NotifyMemoryReleased(first.RequestId));
        using var allocation = first.MemoryEnvelope!.TryTake(new[] { new MemoryCharge("ram", 24) })!;
        allocation.Commit();
        scheduler.NotifyStop(first, SequenceStatus.FinishedLengthCapped, "length", output);
        Check.True(scheduler.Schedule().IsEmpty); // Finished is not yet physically freed.
        scheduler.NotifyMemoryReleased(first.RequestId);
        Check.Equal(24L, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
        Check.Equal(1, scheduler.Schedule().ScheduledWork.Count);
        scheduler.Abort(second.RequestId);
        scheduler.NotifyMemoryReleased(second.RequestId);
        allocation.Dispose();
        Check.True(budget.Snapshot().All(p => p.Available == p.Capacity));
        // Config copies must retain both memory policies.
        Check.True(ReferenceEquals(cfg.MemoryAdmission, cfg.WithSpeculation(cfg.Speculation).MemoryAdmission));
        return Task.CompletedTask;
    }

    public static Task SchedulerMemoryRejection()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 10) });
        using var occupied = budget.Reserve(new[] { new MemoryCharge("ram", 10) });
        var cfg = new SchedulerConfig { BlockSize = 8, MemoryAdmission = new(budget,
            seq => new[] { new MemoryCharge("ram", seq.RequestId == "impossible" ? 11 : 10) }, maxQueuedRequests: 1) };
        var scheduler = new ContinuousBatchScheduler(cfg, new BlockPool(16, 8, 0));
        Check.Throws<MemoryPressureException>(() => scheduler.Submit(MemorySequence("impossible")));
        var pending = MemorySequence("pending");
        scheduler.Submit(pending);
        Check.Throws<MemoryPressureException>(() => scheduler.Submit(MemorySequence("overflow")));
        Check.True(scheduler.Schedule().IsEmpty && scheduler.MemoryAdmissionBlocked);
        scheduler.Abort(pending.RequestId);
        scheduler.NotifyMemoryReleased(pending.RequestId);
        Check.Equal(0, scheduler.WaitingCount);
        return Task.CompletedTask;
    }

    public static async Task BudgetWakeup()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 64) });
        var all = budget.Reserve(new[] { new MemoryCharge("ram", 64) });
        Task signal = budget.ChangeSignal;
        Check.True(!signal.IsCompleted);
        all.Dispose();
        await signal.WaitAsync(TimeSpan.FromSeconds(2));
        Task next = budget.ChangeSignal;
        Check.True(!next.IsCompleted);
        Check.True(budget.TrySetCapacity("ram", 128));
        await next.WaitAsync(TimeSpan.FromSeconds(2));
    }

    public static async Task MultipleLocations()
    {
        await using var f = new Fixture(gpu: Fixture.Page, ram: Fixture.Page * 2);
        var a = f.Add("multi-a"); var b = f.Add("multi-b");
        using (var leases = await f.Scheduler.AcquireReadSetAsync(new[] {
            new ResourcePlacement(a, f.Host.Location), new ResourcePlacement(a, f.Gpu.Location) }))
        {
            Check.Equal(2, leases.Leases.Count);
            var fence = new TaskCompletionSource();
            var release = leases.ReleaseAfterAsync(fence.Task).AsTask();
            Check.Throws<InvalidOperationException>(leases.Dispose);
            await Check.ThrowsAsync<MemoryPressureException>(() => f.Scheduler.AcquireReadSetAsync(new[] {
                new ResourcePlacement(b, f.Host.Location), new ResourcePlacement(b, f.Gpu.Location) }).AsTask());
            Check.Equal(2, f.Scheduler.GetStats().ActiveLeases);
            fence.SetResult();
            await release;
        }
        Check.Equal(0, f.Scheduler.GetStats().ActiveLeases);
    }
}
