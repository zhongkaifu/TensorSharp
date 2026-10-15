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

    [Theory]
    [InlineData(8, 64, 68)]
    [InlineData(1, 35, 70)]
    public async System.Threading.Tasks.Task FileWeights_HaveNoRawPointer_AndReadOnlyRequestedRows(
        int type, int width, int rowBytes)
    {
        using var fixture = new Fixture((SyntheticGguf.GgmlType)type, width);
        var budget = Budget(256);
        var options = new WeightStreamingOptions(budget, "ram", new[] { "gpu" }, tileBytes: 2 * rowBytes);
        var executor = new WeightStreamingExecutor(fixture.File, options);
        using var weight = executor.CreateWeight(fixture.File.Tensors["projection.weight"]);
        try
        {
            Assert.True(weight.IsStreamed);
            Assert.False(weight.HasHostData);
            Assert.Equal(IntPtr.Zero, weight.Data);
            Assert.Equal(IntPtr.Zero, weight.CacheKey);
            Assert.Throws<InvalidOperationException>(() => weight.EnsureDeviceCacheKey());
            Assert.Equal(rowBytes, weight.StreamingRowBytes);
            Assert.Equal(11 * rowBytes, weight.RawBytes);
            var rows = Enumerable.Repeat((byte)0xA5, 2 * rowBytes + 16).ToArray();
            await weight.FileSource.ReadAsync(3 * rowBytes, rows.AsMemory(8, 2 * rowBytes));
            Assert.Equal(fixture.Raw.AsSpan(3 * rowBytes, 2 * rowBytes).ToArray(), rows.AsSpan(8, 2 * rowBytes).ToArray());
            Assert.All(rows.AsSpan(0, 8).ToArray(), value => Assert.Equal((byte)0xA5, value));
            Assert.All(rows.AsSpan(rows.Length - 8).ToArray(), value => Assert.Equal((byte)0xA5, value));
            Assert.Equal(weight.RawBytes, executor.Statistics.FileBackedWeightBytes);
            Assert.Equal(0, executor.Statistics.FileBytesRead); // direct catalog read, no model operation
            Assert.Equal(192, executor.Statistics.PeakHostStagingBytes);
            await Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () => await weight.FileSource.ReadAsync(weight.RawBytes - 10, rows));
        }
        finally { executor.Dispose(); }
        await Assert.ThrowsAsync<ObjectDisposedException>(async () => await weight.FileSource.ReadAsync(0, new byte[1]));
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [Theory]
    [InlineData(8, 64, 64)]
    [InlineData(1, 35, 69)]
    public void TooSmallTile_RejectsTheMinimumLegalRow_WithoutMaterializingTheTensor(
        int type, int width, int tileBytes)
    {
        using var fixture = new Fixture((SyntheticGguf.GgmlType)type, width);
        var budget = Budget((tileBytes + 63) / 64 * 64);
        var options = new WeightStreamingOptions(budget, "ram", new[] { "gpu" }, tileBytes: tileBytes);
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
        Assert.Throws<NotSupportedException>(() => QuantizedWeight.CreateFileBacked(source, 0, 64, 11));
        Assert.Throws<NotSupportedException>(() => QuantizedWeight.CreateFileBacked(source, 30, 64, 11));
        Assert.Throws<NotSupportedException>(() => QuantizedWeight.CreateFileBacked(source, 8, 63, 11));
        Assert.Throws<ArgumentException>(() => QuantizedWeight.CreateFileBacked(source, 8, 64, 10));
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [Theory]
    [InlineData(1, 0, 11)]
    [InlineData(1, -1, 11)]
    [InlineData(1, 35, 0)]
    [InlineData(1, 35, -1)]
    [InlineData(1, 2147483648L, 1)]
    [InlineData(1, 1, 2147483648L)]
    [InlineData(1, long.MaxValue, 2)]
    [InlineData(8, 35, 11)]
    [InlineData(8, 64, 0)]
    [InlineData(8, 2147483648L, 1)]
    public void InvalidDimensionsRejectBeforeAFileRead(int type, long width, long rows)
    {
        var source = new MetadataOnlySource(35 * 2 * 11);
        Assert.Throws<NotSupportedException>(() => QuantizedWeight.CreateFileBacked(source, type, width, rows));
        Assert.Equal(0, source.ReadCount);
    }

    [Fact]
    public void F16RowSizeUsesWideArithmeticWithoutAllocatingItsDeclaredPayload()
    {
        long rowBytes = (long)int.MaxValue * sizeof(ushort);
        var source = new MetadataOnlySource(rowBytes);
        using var weight = QuantizedWeight.CreateFileBacked(source, 1, int.MaxValue, 1);
        Assert.Equal(4294967294L, weight.StreamingRowBytes);
        Assert.Equal(rowBytes, weight.RawBytes);
        Assert.Equal(IntPtr.Zero, weight.Data);
        Assert.Equal(0, source.ReadCount);
    }

    [Fact]
    public void F16LengthAndMatrixRankAreValidatedWithoutMaterializingWeights()
    {
        using var fixture = new Fixture(SyntheticGguf.GgmlType.F16, 35);
        using var catalog = fixture.File.CreateMemoryCatalog("f16-layout");
        var (_, source) = catalog.Get("projection.weight");
        Assert.Throws<ArgumentException>(() => QuantizedWeight.CreateFileBacked(source, 1, 35, 10));
        Assert.Throws<ArgumentException>(() => QuantizedWeight.CreateFileBacked(source, 1, 36, 11));
        Assert.Throws<ArgumentException>(() => QuantizedWeight.CreateFileBacked(
            new ResourceSlice(source, 0, source.ByteLength - 1), 1, 35, 11));
        // A valid file region can still have the wrong rank for linear/embedding.
        // The loader must reject it rather than flattening it implicitly.
        using var vectorFixture = new Fixture(SyntheticGguf.GgmlType.F16, 35, vector: true);
        var budget = Budget(128);
        using (var executor = new WeightStreamingExecutor(vectorFixture.File,
            new WeightStreamingOptions(budget, "ram", new[] { "gpu" }, tileBytes: 128)))
        {
            Assert.Throws<NotSupportedException>(() => executor.CreateWeight(vectorFixture.File.Tensors["projection.weight"]));
            Assert.Equal(0, executor.Statistics.FileBackedWeightBytes);
            Assert.Equal(0, executor.Statistics.FileBytesRead);
        }
        Assert.All(budget.Snapshot(), pool => Assert.Equal(0, pool.Reserved + pool.Committed));
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

    private sealed class MetadataOnlySource(long length) : IResourceSource
    {
        public long ByteLength => length;
        public int ReadCount { get; private set; }
        public System.Threading.Tasks.ValueTask ReadAsync(long offset, Memory<byte> destination,
            System.Threading.CancellationToken cancellationToken = default)
        {
            ReadCount++;
            throw new InvalidOperationException("Metadata validation must not read tensor payload.");
        }
    }

    private sealed class Fixture : IDisposable
    {
        private readonly string _path = Path.Combine(Path.GetTempPath(), "ts-stream-" + Guid.NewGuid().ToString("N") + ".gguf");
        internal readonly GgufFile File;
        internal readonly byte[] Raw;
        internal Fixture(SyntheticGguf.GgmlType type = SyntheticGguf.GgmlType.Q8_0, int width = 64, bool vector = false)
        {
            var tensor = new SyntheticGguf.Tensor
            {
                Name = "projection.weight", Dims = vector ? new ulong[] { (ulong)(width * 11) } : new ulong[] { (ulong)width, 11 }, Type = type,
                Data = Enumerable.Range(0, width * 11).Select(i => (float)Math.Sin(i * 0.031)).ToArray()
            };
            Raw = tensor.Raw();
            SyntheticGguf.Write(_path, new(), new() { tensor });
            File = new GgufFile(_path);
        }
        public void Dispose() { File.Dispose(); System.IO.File.Delete(_path); }
    }
}
