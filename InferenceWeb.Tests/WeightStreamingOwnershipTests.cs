// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System;
using System.IO;
using System.Linq;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class WeightStreamingOwnershipTests
{
    [Fact]
    public void FixedHostTile_ReservesBeforeAllocation_AndPreservesAnotherOwner()
    {
        var budget = Budget(192);
        var options = new WeightStreamingOptions(budget, "ram", new[] { "gpu" }, tileBytes: 65);
        using var other = budget.Reserve(new[] { new MemoryCharge("ram", 64) });
        other.Commit();
        using (var tile = new StreamingHostBuffer(options, 65))
        {
            Assert.Equal(128, tile.AllocatedBytes);
            Assert.Equal(192, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
            tile.GetSpan().Fill(0x45);
            Assert.Throws<MemoryPressureException>(() => new StreamingHostBuffer(options, 1));
            Assert.Equal(0, budget.Snapshot().Single(p => p.Pool == "ram").Reserved);
            Assert.All(tile.GetSpan().ToArray(), b => Assert.Equal((byte)0x45, b));
        }
        Assert.Equal(64, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
        other.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [Fact]
    public async System.Threading.Tasks.Task FileWeights_HaveNoRawPointer_AndReadOnlyRequestedRows()
    {
        using var fixture = new Fixture();
        var budget = Budget(256);
        var options = new WeightStreamingOptions(budget, "ram", new[] { "gpu" }, tileBytes: 136);
        var executor = new WeightStreamingExecutor(fixture.File, options);
        using var weight = executor.CreateWeight(fixture.File.Tensors["projection.weight"]);
        try
        {
            Assert.True(weight.IsStreamed);
            Assert.False(weight.HasHostData);
            Assert.Equal(IntPtr.Zero, weight.Data);
            Assert.Equal(IntPtr.Zero, weight.CacheKey);
            Assert.Throws<InvalidOperationException>(() => weight.EnsureDeviceCacheKey());
            Assert.Equal(11 * 68, weight.RawBytes);
            var rows = new byte[2 * 68];
            await weight.FileSource.ReadAsync(3 * 68, rows);
            Assert.Equal(fixture.Raw.AsSpan(3 * 68, rows.Length).ToArray(), rows);
            Assert.Equal(weight.RawBytes, executor.Statistics.FileBackedWeightBytes);
            Assert.Equal(0, executor.Statistics.FileBytesRead); // direct catalog read, no model operation
            Assert.Equal(192, executor.Statistics.PeakHostStagingBytes);
            await Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () => await weight.FileSource.ReadAsync(weight.RawBytes - 10, rows));
        }
        finally { executor.Dispose(); }
        await Assert.ThrowsAsync<ObjectDisposedException>(async () => await weight.FileSource.ReadAsync(0, new byte[1]));
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [Fact]
    public void TooSmallTile_RejectsTheMinimumLegalRow_WithoutMaterializingTheTensor()
    {
        using var fixture = new Fixture();
        var budget = Budget(64);
        var options = new WeightStreamingOptions(budget, "ram", new[] { "gpu" }, tileBytes: 64);
        using (var executor = new WeightStreamingExecutor(fixture.File, options))
        {
            Assert.Throws<MemoryPressureException>(() => executor.CreateWeight(fixture.File.Tensors["projection.weight"]));
            Assert.Equal(0, executor.Statistics.FileBackedWeightBytes);
            Assert.Equal(0, executor.Statistics.FileBytesRead);
        }
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [Fact]
    public void FileLayoutAndPoolMappings_AreCheckedWithoutNativeBackend()
    {
        var budget = Budget(128);
        Assert.Throws<ArgumentException>(() => new WeightStreamingOptions(budget, "missing", new[] { "gpu" }));
        Assert.Throws<ArgumentException>(() => new WeightStreamingOptions(budget, "ram", new[] { "gpu", "gpu" }));
        Assert.Throws<ArgumentException>(() => new WeightStreamingOptions(budget, "ram", Array.Empty<string>()));
        Assert.Throws<ArgumentOutOfRangeException>(() => new WeightStreamingOptions(budget, "ram", new[] { "gpu" }, tokenTileRows: 0));
        using var fixture = new Fixture();
        using var catalog = fixture.File.CreateMemoryCatalog("fixture");
        var (_, source) = catalog.Get("projection.weight");
        Assert.Throws<NotSupportedException>(() => QuantizedWeight.CreateFileBacked(source, 1, 64, 11));
        Assert.Throws<NotSupportedException>(() => QuantizedWeight.CreateFileBacked(source, 8, 63, 11));
        Assert.Throws<ArgumentException>(() => QuantizedWeight.CreateFileBacked(source, 8, 64, 10));
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [Fact]
    public void TilePlanner_AdaptsToSharedPressure_WithoutSplittingReductionAxis()
    {
        var budget = new MemoryBudget(new[] { new MemoryCharge("ram", 2048), new MemoryCharge("gpu", 1024) });
        var options = new WeightStreamingOptions(budget, "ram", new[] { "gpu" }, 1024, 8);
        // The fake backend sizes only its owned payload; native tests separately
        // verify the real aligned CUDA layout against allocations and computation.
        static long Payload(long k, int rows, int n) => k * n * 4 + k / 32 * 34 * rows + rows * n * 4;
        using var tile = new StreamingHostBuffer(options, 1024);
        var initial = WeightStreamingExecutor.SelectTileLayout(options, 64, 100, 8, Payload);
        Assert.Equal((6, 2), initial); // N=8/4 input alone cannot fit this GPU.
        using var other = budget.Reserve(new[] { new MemoryCharge("gpu", 680) });
        other.Commit();
        Assert.Equal((1, 1), WeightStreamingExecutor.SelectTileLayout(options, 64, 100, 8, Payload));
        Assert.True(budget.TrySetCapacity("gpu", 700));
        Assert.Throws<MemoryPressureException>(() => WeightStreamingExecutor.SelectTileLayout(options, 64, 100, 8, Payload));
        Assert.Equal(680, budget.Snapshot().Single(p => p.Pool == "gpu").Committed);
    }

    private static MemoryBudget Budget(long host) => new(new[] { new MemoryCharge("ram", host), new MemoryCharge("gpu", 1024) });

    private sealed class Fixture : IDisposable
    {
        private readonly string _path = Path.Combine(Path.GetTempPath(), "ts-stream-" + Guid.NewGuid().ToString("N") + ".gguf");
        internal readonly GgufFile File;
        internal readonly byte[] Raw;
        internal Fixture()
        {
            var tensor = new SyntheticGguf.Tensor
            {
                Name = "projection.weight", Dims = new ulong[] { 64, 11 }, Type = SyntheticGguf.GgmlType.Q8_0,
                Data = Enumerable.Range(0, 64 * 11).Select(i => (float)Math.Sin(i * 0.031)).ToArray()
            };
            Raw = tensor.Raw();
            SyntheticGguf.Write(_path, new(), new() { tensor });
            File = new GgufFile(_path);
        }
        public void Dispose() { File.Dispose(); System.IO.File.Delete(_path); }
    }
}
