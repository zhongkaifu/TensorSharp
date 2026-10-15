// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Runtime.Paged;

static partial class Cases
{
    public static Task SharedSnapshotBudget()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-shared-kv-" + Guid.NewGuid().ToString("N"));
        var budget = new MemoryBudget(new[] { new MemoryCharge("node/ram", 256), new MemoryCharge("node/ssd", 8192) });
        var options = KvSnapshotOptions.FromSharedBudget(budget, "node/ram", "node/ssd", root, 64);
        try
        {
            using var otherOwner = budget.Reserve(new[] { new MemoryCharge("node/ram", 64), new MemoryCharge("node/ssd", 4096) });
            otherOwner.Commit();
            using var storage = new PagedKvStorage(8, 64, options);
            using (var page = storage.Acquire(0, ResourceAccess.Write)) page.Span.Fill(17);
            using (var page = storage.Acquire(1, ResourceAccess.Write)) page.Span.Fill(29);
            Check.Equal(256L, budget.Snapshot().Single(p => p.Pool == "node/ram").Committed);
            Check.Equal(8192L, budget.Snapshot().Single(p => p.Pool == "node/ssd").Committed);
            Check.True(storage.ResidencyStats!.Value.Spills > 0);
            // Both owners compete for the same SSD quota. A failed spill cannot
            // discard the authoritative RAM page to make the counters fit.
            Check.Throws<MemoryPressureException>(() => storage.Acquire(2).Dispose());
            using (var page = storage.Acquire(1)) Check.True(page.ReadOnlySpan.ToArray().All(b => b == 29));
            storage.ReleaseSlab(1);
            using (var page = storage.Acquire(0)) Check.True(page.ReadOnlySpan.ToArray().All(b => b == 17));
            storage.Dispose();
            Check.Equal(64L, budget.Snapshot().Single(p => p.Pool == "node/ram").Committed);
            Check.Equal(4096L, budget.Snapshot().Single(p => p.Pool == "node/ssd").Committed);
            otherOwner.Dispose();
            Check.True(budget.Snapshot().All(p => p.Available == p.Capacity));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        return Task.CompletedTask;
    }

    public static Task SharedSnapshotOwners()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-shared-owners-" + Guid.NewGuid().ToString("N"));
        var budget = new MemoryBudget(new[] { new MemoryCharge("node/ram", 320), new MemoryCharge("node/ssd", 8192) });
        var options = KvSnapshotOptions.FromSharedBudget(budget, "node/ram", "node/ssd", root, 64);
        try
        {
            using var first = new PagedKvStorage(8, 64, options);
            using var second = new PagedKvStorage(8, 64, options);
            using (var page = first.Acquire(0, ResourceAccess.Write)) page.Span.Fill(11);
            // Each adapter may evict only its own pages, even when another
            // adapter has an unpinned resident page in the shared physical pool.
            Check.Throws<MemoryPressureException>(() => second.Acquire(0).Dispose());
            first.ReleaseSlab(0);
            using (var page = second.Acquire(0, ResourceAccess.Write)) page.Span.Fill(23);
            first.Dispose();
            Check.Equal(192L, budget.Snapshot().Single(p => p.Pool == "node/ram").Committed);
            using var unrelated = budget.Reserve(new[] { new MemoryCharge("node/ram", 128) });
            unrelated.Commit();
            using (var page = second.Acquire(0)) Check.True(page.ReadOnlySpan.ToArray().All(b => b == 23));
            second.Dispose();
            Check.Equal(128L, budget.Snapshot().Single(p => p.Pool == "node/ram").Committed);
            unrelated.Dispose();
            Check.True(budget.Snapshot().All(p => p.Available == p.Capacity));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        return Task.CompletedTask;
    }

    public static Task SharedSnapshotConstruction()
    {
        string root = Path.Combine(Path.GetTempPath(), "ts-shared-construct-" + Guid.NewGuid().ToString("N"));
        var budget = new MemoryBudget(new[] { new MemoryCharge("node/ram", 256), new MemoryCharge("node/ssd", 8192) });
        Check.Throws<ArgumentException>(() => KvSnapshotOptions.FromSharedBudget(budget, "missing", "node/ssd", root, 64));
        Check.Throws<ArgumentException>(() => KvSnapshotOptions.FromSharedBudget(budget, "node/ram", "node/ram", root, 64));
        var options = KvSnapshotOptions.FromSharedBudget(budget, "node/ram", "node/ssd", root, 64);
        try
        {
            using var occupied = budget.Reserve(new[] { new MemoryCharge("node/ram", 192) });
            occupied.Commit();
            // Staging fits but capture scratch does not. Roll back only the
            // partially constructed adapter, preserving the existing owner.
            Check.Throws<MemoryPressureException>(() => new PagedKvStorage(8, 64, options).Dispose());
            Check.Equal(192L, budget.Snapshot().Single(p => p.Pool == "node/ram").Committed);
            Check.True(budget.Snapshot().All(p => p.Reserved == 0));
            occupied.Dispose();
            Check.True(budget.TrySetCapacity("node/ram", 128));
            Check.Throws<ArgumentException>(() => new PagedKvStorage(8, 64, options).Dispose());
            Check.True(budget.Snapshot().All(p => p.Available == p.Capacity));
        }
        finally { if (Directory.Exists(root)) Directory.Delete(root, true); }
        return Task.CompletedTask;
    }
}
