// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using System.Runtime.InteropServices;
using TensorSharp.Memory;
using TensorSharp.Models;
using TensorSharp.Runtime;

namespace InferenceWeb.Tests;

public class WeightReadAheadTests
{
    [Fact]
    public void ExistingStreamingConstructorAndStatisticsDeconstructionRemainAvailable()
    {
        Assert.NotNull(typeof(WeightStreamingOptions).GetConstructor([
            typeof(MemoryBudget), typeof(string), typeof(IEnumerable<string>), typeof(int), typeof(int)]));
        var statistics = new WeightStreamingStatistics(1, 2, 3, 4, 5, 6, 7, 8, 9)
        { ReadAheadOperations = 10 };
        var (weight, reads, tiles, embeddings, host, device, sessions, uploads, projections) = statistics;
        Assert.Equal(new long[] { 1, 2, 3, 4, 5, 6, 7, 8, 9 },
            new long[] { weight, reads, tiles, embeddings, host, device, sessions, uploads, projections });
        Assert.Equal(10, statistics.ReadAheadOperations);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public async Task NextReadOverlapsCurrentConsumptionAndAbortDrainsBeforeRefund(bool cacheEnabled)
    {
        using var fixture = new Fixture();
        var budget = Budget(5 << 20);
        using var executor = new WeightStreamingExecutor(fixture.File,
            new(budget, "ram", ["gpu"], tileBytes: 136)
                { HostCacheBytes = cacheEnabled ? 1024 : 0, HostCacheReserveBytes = 0 });
        var source = new DelayedSource();
        using var weight = QuantizedWeight.CreateFileBacked(source, 8, 64, 4);
        var tiles = executor.ReadTiles(weight, 2).GetEnumerator();
        Assert.True(tiles.MoveNext());
        Assert.True(source.SecondStarted.Task.IsCompleted);
        Assert.False(source.FinishSecond.Task.IsCompleted);
        var actual = new byte[136];
        Marshal.Copy(tiles.Current.Pointer, actual, 0, actual.Length);
        Assert.All(actual, b => Assert.Equal(1, b));
        Assert.Equal(cacheEnabled ? 576 : 384, budget.Snapshot().Single(p => p.Pool == "ram").Committed);
        var disposalEntered = new TaskCompletionSource(TaskCreationOptions.RunContinuationsAsynchronously);
        var disposing = Task.Run(() => { disposalEntered.SetResult(); tiles.Dispose(); });
        await disposalEntered.Task;
        Assert.False(disposing.IsCompleted);
        source.FinishSecond.SetResult();
        await disposing.WaitAsync(TimeSpan.FromSeconds(10));
        Assert.Equal(272, executor.Statistics.FileBytesRead);
        Assert.Equal(1, executor.Statistics.ReadAheadOperations);
        if (cacheEnabled)
        {
            Assert.Equal(384, executor.Statistics.HostCacheBytes);
            foreach (var tile in executor.ReadTiles(weight, 2))
            {
                Marshal.Copy(tile.Pointer, actual, 0, actual.Length);
                Assert.All(actual, b => Assert.Equal(tile.FirstRow == 0 ? (byte)1 : (byte)2, b));
            }
            Assert.Equal(272, executor.Statistics.FileBytesRead);
            Assert.Equal(272, executor.Statistics.HostCacheHitBytes);
            Assert.Equal(1, executor.Statistics.ReadAheadOperations); // Copies are not file read-ahead.
        }
        executor.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    [Theory]
    [InlineData(false, 5242880)]
    [InlineData(true, 192)]
    public void TightBudgetOrDisabledReadAheadKeepsSequentialReads(bool enabled, long hostBytes)
    {
        using var fixture = new Fixture();
        var budget = Budget(hostBytes);
        using var executor = new WeightStreamingExecutor(fixture.File,
            new(budget, "ram", ["gpu"], tileBytes: 136, tokenTileRows: 32, readAhead: enabled));
        var source = new DelayedSource();
        source.FinishSecond.SetResult();
        using var weight = QuantizedWeight.CreateFileBacked(source, 8, 64, 4);
        using var tiles = executor.ReadTiles(weight, 2).GetEnumerator();
        Assert.True(tiles.MoveNext());
        Assert.False(source.SecondStarted.Task.IsCompleted);
        Assert.True(tiles.MoveNext());
        Assert.False(tiles.MoveNext());
        Assert.Equal(0, executor.Statistics.ReadAheadOperations);
        Assert.Equal(192, executor.Statistics.PeakHostStagingBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void SpeculativeReadFailurePropagatesWhenItsTileIsConsumed(bool cacheEnabled)
    {
        using var fixture = new Fixture();
        var budget = Budget(5 << 20);
        using var executor = new WeightStreamingExecutor(fixture.File,
            new(budget, "ram", ["gpu"], tileBytes: 136)
                { HostCacheBytes = cacheEnabled ? 1024 : 0, HostCacheReserveBytes = 0 });
        var source = new DelayedSource();
        source.FinishSecond.SetException(new IOException("injected read failure"));
        using var weight = QuantizedWeight.CreateFileBacked(source, 8, 64, 4);
        using var tiles = executor.ReadTiles(weight, 2).GetEnumerator();
        Assert.True(tiles.MoveNext());
        Assert.Throws<IOException>(() => tiles.MoveNext());
        Assert.Equal(cacheEnabled ? 192 : 0, executor.Statistics.HostCacheBytes); // Only the complete first read.
        executor.Dispose();
        Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Reserved + p.Committed));
    }

    private static MemoryBudget Budget(long host) => new([new("ram", host), new("gpu", 1024)]);

    [Fact]
    public void SharedPhysicalPoolDoesNotPreemptIrreducibleDeviceWorkspaceForReadAhead()
    {
        using var fixture = new Fixture();
        var budget = new MemoryBudget([new("uma", 5 << 20)]);
        using var executor = new WeightStreamingExecutor(fixture.File,
            new(budget, "uma", ["uma"], tileBytes: 136));
        Assert.Equal(192, budget.Snapshot().Single().Committed);
        using var device = budget.Reserve([new("uma", (5 << 20) - 192)]);
        device.Commit();
        Assert.Equal(5 << 20, budget.Snapshot().Single().Committed);
    }
    private sealed class DelayedSource : IResourceSource
    {
        public long ByteLength => 272;
        public TaskCompletionSource SecondStarted { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public TaskCompletionSource FinishSecond { get; } = new(TaskCreationOptions.RunContinuationsAsynchronously);
        public async ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
        {
            if (offset != 0) { SecondStarted.TrySetResult(); await FinishSecond.Task.ConfigureAwait(false); }
            destination.Span.Fill(offset == 0 ? (byte)1 : (byte)2);
        }
    }
    private sealed class Fixture : IDisposable
    {
        private readonly string _path = Path.Combine(Path.GetTempPath(), "ts-read-ahead-" + Guid.NewGuid().ToString("N") + ".gguf");
        internal GgufFile File { get; }
        internal Fixture()
        {
            var tensor = new SyntheticGguf.Tensor
            { Name = "projection.weight", Dims = [64, 4], Type = SyntheticGguf.GgmlType.Q8_0, Data = new float[256] };
            SyntheticGguf.Write(_path, new(), new() { tensor });
            File = new GgufFile(_path);
        }
        public void Dispose() { File.Dispose(); System.IO.File.Delete(_path); }
    }
}
