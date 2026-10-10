// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp;
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;

namespace InferenceWeb.Tests;

public sealed class StreamingWeightReadCacheTests
{
    [Fact]
    public void DevicePromotionRemovesOnlyTheSameSourceFromRam()
    {
        var budget = new MemoryBudget([new("ram", 512), new("gpu", 1024)]);
        using var cache = new StreamingWeightReadCache(new(budget, "ram", ["gpu"])
            { HostCacheBytes = 512, HostCacheReserveBytes = 0 });
        var first = new Source(128); var second = new Source(128); // Value-equal, reference-distinct.
        cache.Store(first, 0, new byte[64]); cache.Store(first, 64, new byte[64]);
        cache.Store(second, 0, new byte[64]);
        cache.RemoveSource(first);
        Assert.False(cache.TryCopy(first, 0, new byte[64]));
        Assert.True(cache.TryCopy(second, 0, new byte[64]));
        Assert.Equal(64, cache.Statistics.Bytes);
        Assert.Equal(128, cache.Statistics.Evicted);
        Assert.Equal(64, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
    }
    [Fact]
    public void SequentialScanDoesNotEvictReusableRangesAndPressurePreservesOtherOwners()
    {
        var budget = new MemoryBudget([new("ram", 256), new("gpu", 1024)]);
        using var other = budget.Reserve([new("ram", 64)]); other.Commit();
        using var cache = new StreamingWeightReadCache(new(budget, "ram", ["gpu"])
            { HostCacheBytes = 128, HostCacheReserveBytes = 64 });
        var source = new Source(256);
        cache.Store(source, 0, Enumerable.Repeat((byte)1, 64).ToArray());
        cache.Store(source, 64, Enumerable.Repeat((byte)2, 64).ToArray());
        cache.Store(source, 128, new byte[64]); // Scan miss must not churn the first two ranges.
        Assert.Equal(128, cache.Statistics.Bytes);
        Assert.False(cache.TryCopy(source, 128, new byte[64]));
        var copied = new byte[64];
        Assert.True(cache.TryCopy(source, 0, copied));
        Assert.All(copied, value => Assert.Equal((byte)1, value));
        cache.LeaveAvailable(128); // Evicts the least recently used second range.
        Assert.Equal(64, cache.Statistics.Bytes);
        Assert.True(cache.TryCopy(source, 0, copied));
        Assert.False(cache.TryCopy(source, 64, copied));
        cache.Dispose();
        Assert.Equal(64, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
        Assert.Equal(128, cache.Statistics.Evicted);
    }

    [Fact]
    public void SourceIdentityRangeAndStagingOwnershipCannotAlias()
    {
        var budget = new MemoryBudget([new("ram", 1024), new("gpu", 1024)]);
        using var cache = new StreamingWeightReadCache(new(budget, "ram", ["gpu"])
            { HostCacheBytes = 512, HostCacheReserveBytes = 0 });
        var first = new Source(64); var second = new Source(64);
        byte[] input = Enumerable.Repeat((byte)7, 64).ToArray();
        cache.Store(first, 0, input); input[0] = 99;
        Assert.False(cache.TryCopy(second, 0, new byte[64])); // Sources deliberately override Equals.
        Assert.False(cache.TryCopy(first, 0, new byte[32]));
        var destination = new byte[64];
        Assert.True(cache.TryCopy(first, 0, destination));
        cache.TrimTo(0);
        Assert.All(destination, x => Assert.Equal((byte)7, x)); // Already copied staging survives eviction.
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [Fact]
    public void BudgetAndWorkspaceHoldoutRefuseRetentionWithoutFailingTheRead()
    {
        var budget = new MemoryBudget([new("ram", 128), new("gpu", 1024)]);
        using var cache = new StreamingWeightReadCache(new(budget, "ram", ["gpu"])
            { HostCacheBytes = 128, HostCacheReserveBytes = 65 });
        cache.Store(new Source(64), 0, new byte[64]);
        Assert.Equal(0, cache.Statistics.Bytes);
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Committed));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WeightStreamingOptions(budget, "ram", ["gpu"]) { HostCacheBytes = -1 });
    }

    [Fact]
    public void ReadFailureIsNeverPublishedAndCompletedReadIsReusedAfterRetry()
    {
        using var fixture = new Fixture();
        var budget = new MemoryBudget([new("ram", 2048), new("gpu", 1024)]);
        using var executor = new WeightStreamingExecutor(fixture.File,
            new(budget, "ram", ["gpu"], 136, 1, false) { HostCacheBytes = 1024, HostCacheReserveBytes = 0 });
        var source = new Source(272) { Fail = true };
        using var weight = QuantizedWeight.CreateFileBacked(source, 8, 64, 4);
        Assert.Throws<IOException>(() => Read(executor, weight));
        Assert.Equal(0, executor.Statistics.HostCacheBytes);
        source.Fail = false;
        Assert.All(Read(executor, weight), value => Assert.Equal((byte)5, value));
        Assert.Equal(272, executor.Statistics.FileBytesRead);
        Assert.All(Read(executor, weight), value => Assert.Equal((byte)5, value));
        Assert.Equal(272, executor.Statistics.FileBytesRead);
        Assert.Equal(272, executor.Statistics.HostCacheHitBytes);
        executor.TrimIdleCaches();
        Assert.Equal(192, budget.Snapshot().Single(p => p.Pool == "ram").Committed); // Required read tile remains.
        Read(executor, weight);
        Assert.Equal(544, executor.Statistics.FileBytesRead);
        executor.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [Theory]
    [InlineData(1024, 768, 512, 1024, 256)]
    [InlineData(1024, 768, 128, 1024, 128)]
    [InlineData(1024, 768, 512, 64, 64)]
    [InlineData(1024, 1024, 512, 1024, 0)]
    public void AdaptiveCacheUsesOnlyHardwareSlackAfterTheRequestPeak(long capacity, long peak, long source, long ceiling, long expected)
    {
        var plan = new InferenceMemoryPlan
        {
            SelectedCandidate = new() { Name = "stream", Placement = InferenceWeightPlacement.SsdStreaming, PrefillChunkTokens = 32 },
            Capacities = [new(AdaptiveModelSession.HostPool, capacity, capacity, 0, 0, capacity, 0)],
            PoolPeaks = [new(AdaptiveModelSession.HostPool, peak, peak, peak, peak, 0, peak)]
        };
        Assert.Equal(expected, AdaptiveModelSession.HostCacheLimit(plan, source, ceiling));
        Assert.Equal(0, AdaptiveModelSession.HostCacheLimit(plan with
        { SelectedCandidate = plan.SelectedCandidate with { Placement = InferenceWeightPlacement.Resident } }, source, ceiling));
        var devicePlan = plan with
        {
            Capacities = [new(AdaptiveModelSession.DevicePool, capacity, capacity, 0, 0, capacity, 0)],
            PoolPeaks = [new(AdaptiveModelSession.DevicePool, peak, peak, peak, peak, 0, peak)]
        };
        Assert.Equal(expected, AdaptiveModelSession.DeviceCacheLimit(devicePlan, source, ceiling));
        Assert.Equal(0, AdaptiveModelSession.DeviceCacheLimit(devicePlan with
        { SelectedCandidate = devicePlan.SelectedCandidate with { Placement = InferenceWeightPlacement.Resident } }, source, ceiling));
    }

    [Fact]
    public void CacheStartsAfterLoadAndYieldsBeforeTheNextRequestsWorkspaceIsStarved()
    {
        var plan = new InferenceMemoryPlan
        {
            SelectedCandidate = new() { Name = "stream", Placement = InferenceWeightPlacement.SsdStreaming, PrefillChunkTokens = 32 },
            Capacities = [new(AdaptiveModelSession.HostPool, 1024, 1024, 0, 0, 1024, 0)],
            PoolPeaks = [new(AdaptiveModelSession.HostPool, 1024, 512, 256, 1024, 0, 1024)]
        };
        Assert.Equal(512, AdaptiveModelSession.HostCacheLimit(plan, 2048, 2048));
        var pool = new InferenceMemoryPool(AdaptiveModelSession.HostPool, 2048, 512, 0,
            new(AdaptiveModelSession.HostPool, 1024, 0, 512), 700);
        // 512 existing bytes fit 700, but another 256 bytes are needed by the request.
        Assert.True(AdaptiveModelSession.RequiresIdleTrim(pool, 256, 128));
        Assert.False(AdaptiveModelSession.RequiresIdleTrim(pool, 256, 0));
        Assert.False(AdaptiveModelSession.RequiresIdleTrim(pool with { MaximumBudgetBytes = 768 }, 256, 128));
        Assert.True(AdaptiveModelSession.RequiresIdleTrim(pool with { MaximumBudgetBytes = 500 }, 0, 0));
        Assert.False(AdaptiveModelSession.RequiresIdleTrim(pool with { Pool = "gpu" }, 256, 128));
        var device = pool with { Pool = AdaptiveModelSession.DevicePool };
        Assert.True(AdaptiveModelSession.RequiresIdleTrim(device, 0, 0, 256, 128));
        Assert.False(AdaptiveModelSession.RequiresIdleTrim(device, 0, 0, 256, 0));
        Assert.False(AdaptiveModelSession.RequiresIdleTrim(device with { MaximumBudgetBytes = 768 }, 0, 0, 256, 128));
    }

    [GgmlFact(BackendType.GgmlCuda)]
    public void ActualCudaProjectionReusesBytesAndTrimForcesReloadWithoutChangingResults()
    {
        using var fixture = new Fixture();
        var budget = new MemoryBudget([new("ram", 4096), new("gpu", 1 << 20)]);
        using var other = budget.Reserve([new("ram", 64)]); other.Commit();
        using var model = new ProjectionModel(fixture.Path,
            new(budget, "ram", ["gpu"], 136, 1, false) { HostCacheBytes = 1024, HostCacheReserveBytes = 0 });
        var first = model.Forward([1]);
        // Stored all-one Q8 blocks use a half-precision 1/127 scale.
        float expected = 64f * 127f * (float)(System.Half)(1f / 127f);
        Assert.All(first, value => Assert.Equal(expected, value));
        long reads = model.StreamingWeightUsage.Value.FileBytesRead;
        Assert.Equal(first, model.Forward([2]));
        Assert.Equal(reads, model.StreamingWeightUsage.Value.FileBytesRead);
        Assert.True(model.StreamingWeightUsage.Value.HostCacheHits > 0);
        model.TrimIdleMemory();
        Assert.Equal(0, model.StreamingWeightUsage.Value.HostCacheBytes);
        Assert.Equal(first, model.Forward([3]));
        Assert.Equal(reads * 2, model.StreamingWeightUsage.Value.FileBytesRead);
        model.Dispose();
        Assert.Equal(64, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
        Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
    }

    private static byte[] Read(WeightStreamingExecutor executor, QuantizedWeight weight)
    {
        var bytes = new List<byte>();
        foreach (var tile in executor.ReadTiles(weight, 2))
        {
            var data = new byte[tile.Rows * 68];
            Marshal.Copy(tile.Pointer, data, 0, data.Length); bytes.AddRange(data);
        }
        return bytes.ToArray();
    }

    private sealed class Source(long length) : IResourceSource
    {
        public bool Fail { get; set; }
        public long ByteLength => length;
        public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
        {
            destination.Span.Fill(Fail ? (byte)99 : (byte)5);
            if (Fail) throw new IOException("injected partial read");
            return ValueTask.CompletedTask;
        }
        public override bool Equals(object obj) => obj is Source;
        public override int GetHashCode() => 0;
    }

    private sealed class Fixture : IDisposable
    {
        internal string Path { get; } = System.IO.Path.Combine(System.IO.Path.GetTempPath(), $"ts-weight-cache-{Guid.NewGuid():N}.gguf");
        internal GgufFile File { get; }
        internal Fixture()
        {
            SyntheticGguf.Write(Path, [new SyntheticGguf.Str { Key = "general.architecture", V = "probe" }],
                [new SyntheticGguf.Tensor { Name = "projection.weight", Dims = [64, 4],
                    Type = SyntheticGguf.GgmlType.Q8_0, Data = Enumerable.Repeat(1f, 256).ToArray() }]);
            File = new GgufFile(Path);
        }
        public void Dispose() { File.Dispose(); System.IO.File.Delete(Path); }
    }

    private sealed class ProjectionModel : ModelBase
    {
        public ProjectionModel(string path, WeightStreamingOptions options) : base(path, BackendType.GgmlCuda, weightStreaming: options)
        { try { LoadWeights(); } catch { Dispose(); throw; } }
        protected override float[] ForwardCore(int[] tokens)
        {
            using var input = CreateFloatTensor(Enumerable.Repeat(1f, 64).ToArray(), 1, 64);
            using var output = LinearForward(input, "projection.weight");
            return output.GetElementsAsFloat(4);
        }
        protected override void ResetKVCacheCore() { }
    }
}
