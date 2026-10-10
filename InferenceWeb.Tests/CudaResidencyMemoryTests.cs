// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Cuda;
using TensorSharp.Cuda.Interop;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

[Collection(EngineEnvironmentCollection.Name)]
public sealed class CudaResidencyMemoryTests
{
    [CudaFact]
    public async Task DriverCapacityRefusalReturnsCreditAndLeavesSmallAllocationsUsable()
    {
        using var context = CudaContext.Create(0);
        CudaDriverApi.cuMemGetInfo(out _, out var total).ThrowOnError();
        // Deliberately impossible allocation, not a fill-to-OOM stress test.
        long impossible = Math.Max(1L << 40, checked((long)(ulong)total * 8));
        var budget = new MemoryBudget([new("gpu", impossible), new("ram", 4096), new("ssd", 0)]);
        string directory = Path.Combine(Path.GetTempPath(), "ts-cuda-refusal-" + Guid.NewGuid().ToString("N"));
        try
        {
            using var transfers = new BoundedTransfers(budget, "ram", 4096, 1);
            using var spill = new SsdSpillStore(budget, "ssd", directory, transfers);
            var backend = new CudaResidencyBackend(context, "gpu");
            await using (var scheduler = new TieredMemoryScheduler(budget, [backend], transfers, spill))
            {
                var key = new ResourceKey("refusal", 0, "impossible");
                scheduler.Register(new(key, impossible, ResourceKind.KvPage, Mutable: true));
                await Assert.ThrowsAsync<OutOfMemoryException>(() => scheduler.AcquireAsync(key, backend.Location).AsTask());
                Assert.Equal(1, scheduler.GetStats().PhysicalAllocationRefusals);
                Assert.Equal(0, scheduler.GetStats().ActiveLeases);
                Assert.Equal(impossible, budget.Snapshot().Single(p => p.Pool == "gpu").Available);
                Assert.Empty(scheduler.Snapshot());
                using var buffer = await backend.AllocateAsync(4096);
                byte[] zero = new byte[4096];
                Array.Fill(zero, (byte)73);
                await buffer.ReadAsync(0, zero);
                Assert.All(zero, b => Assert.Equal(0, b));
            }
        }
        finally
        {
            if (Directory.Exists(directory)) Directory.Delete(directory, true);
        }
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }
}
