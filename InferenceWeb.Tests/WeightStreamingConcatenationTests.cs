// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

public sealed class WeightStreamingConcatenationTests
{
    [Fact]
    public async Task CompositeReadsCrossNonAdjacentFileRegionsAndBorrowTheirOwner()
    {
        string path = Path.GetTempFileName();
        try
        {
            byte[] bytes = Enumerable.Range(0, 255).Select(i => (byte)i).ToArray();
            File.WriteAllBytes(path, bytes);
            using var file = new FileDataSource(path);
            using var q = QuantizedWeight.CreateFileBacked(new FileRegionSource(file, 12, 68), 8, 32, 2);
            using var k = QuantizedWeight.CreateFileBacked(new FileRegionSource(file, 180, 34), 8, 32, 1);
            using var fused = QuantizedWeight.CreateFileBackedConcatenation(q, k, k);
            Assert.Equal(4, fused.Ne1);
            Assert.Equal(136, fused.RawBytes);
            Assert.True(fused.IsStreamed);
            Assert.Equal(IntPtr.Zero, fused.Data);
            Assert.Equal(IntPtr.Zero, fused.CacheKey);
            Assert.Throws<InvalidOperationException>(() => fused.EnsureDeviceCacheKey());
            q.Dispose(); k.Dispose(); // Disposing metadata does not close the model's catalog.
            byte[] expected = bytes.AsSpan(12, 68).ToArray()
                .Concat(bytes.AsSpan(180, 34).ToArray()).Concat(bytes.AsSpan(180, 34).ToArray()).ToArray();
            byte[] read = Enumerable.Repeat((byte)0xA5, 92).ToArray();
            await fused.FileSource.ReadAsync(60, read.AsMemory(8, 72));
            Assert.Equal(expected.AsSpan(60, 72).ToArray(), read.AsSpan(8, 72).ToArray());
            Assert.All(read.Take(8).Concat(read.Skip(80)), b => Assert.Equal((byte)0xA5, b));
            fused.Dispose();
            await fused.FileSource.ReadAsync(0, new byte[1]); // No source ownership transfers.
            file.Dispose();
            await Assert.ThrowsAsync<ObjectDisposedException>(async () => await fused.FileSource.ReadAsync(0, new byte[1]));
        }
        finally { File.Delete(path); }
    }

    [Fact]
    public async Task InvalidRangesCancellationAndSourceFailuresDoNotStartLaterReads()
    {
        var first = new TrackingSource(34);
        var second = new TrackingSource(34);
        var source = new ConcatenatedWeightSource(first, second);
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () => await source.ReadAsync(-1, new byte[1]));
        await Assert.ThrowsAsync<ArgumentOutOfRangeException>(async () => await source.ReadAsync(67, new byte[2]));
        await Assert.ThrowsAsync<OperationCanceledException>(async () =>
            await source.ReadAsync(0, new byte[60], new CancellationToken(true)));
        Assert.Equal(0, first.Reads + second.Reads);
        first.Fail = true;
        await Assert.ThrowsAsync<IOException>(async () => await source.ReadAsync(30, new byte[8]));
        Assert.Equal(1, first.Reads);
        Assert.Equal(0, second.Reads);
        first.Fail = false;
        await source.ReadAsync(34, new byte[1]);
        Assert.Equal(1, first.Reads);
        Assert.Equal(1, second.Reads);
        await source.ReadAsync(68, Memory<byte>.Empty);
    }

    [Fact]
    public void IncompatibleMetadataIsRejectedWithoutReadingAndSharedScaleIsPreserved()
    {
        var source = new TrackingSource(68);
        using var q = QuantizedWeight.CreateFileBacked(source, 8, 32, 2);
        using var differentWidth = QuantizedWeight.CreateFileBacked(source, 8, 64, 1);
        using var differentType = QuantizedWeight.CreateFileBacked(source, 1, 34, 1);
        using var other = QuantizedWeight.CreateFileBacked(source, 8, 32, 2);
        Assert.Throws<ArgumentException>(() => QuantizedWeight.CreateFileBackedConcatenation(q, differentWidth));
        Assert.Throws<ArgumentException>(() => QuantizedWeight.CreateFileBackedConcatenation(q, differentType));
        other.Scale = 2;
        Assert.Throws<ArgumentException>(() => QuantizedWeight.CreateFileBackedConcatenation(q, other));
        q.Scale = 2;
        using var fused = QuantizedWeight.CreateFileBackedConcatenation(q, other);
        Assert.Equal(2, fused.Scale);
        Assert.Equal(0, source.Reads);
    }

    [Fact]
    public void VirtualFusionDoesNotDoubleCountPhysicalFileWeightBytesOrAllocateAnotherHostTile()
    {
        string path = Path.Combine(Path.GetTempPath(), "ts-composite-" + Guid.NewGuid().ToString("N") + ".gguf");
        try
        {
            SyntheticGguf.Write(path, new(), new() {
                new() { Name = "q.weight", Dims = new ulong[] { 32, 2 }, Type = SyntheticGguf.GgmlType.Q8_0, Data = new float[64] },
                new() { Name = "k.weight", Dims = new ulong[] { 32, 1 }, Type = SyntheticGguf.GgmlType.Q8_0, Data = new float[32] }
            });
            using var file = new GgufFile(path);
            var budget = new MemoryBudget(new[] { new MemoryCharge("host", 128), new MemoryCharge("gpu", 1024) });
            using (var executor = new WeightStreamingExecutor(file,
                new WeightStreamingOptions(budget, "host", new[] { "gpu" }, 128)))
            {
                using var q = executor.CreateWeight(file.Tensors["q.weight"]);
                using var k = executor.CreateWeight(file.Tensors["k.weight"]);
                var before = executor.Statistics;
                using var fused = QuantizedWeight.CreateFileBackedConcatenation(q, k, k);
                Assert.Equal(102, before.FileBackedWeightBytes);
                Assert.Equal(before, executor.Statistics);
                Assert.Equal(128, budget.Snapshot().Single(p => p.Pool == "host").Committed);
                Assert.Equal(136, fused.RawBytes); // Logical V aliases K, without counting a second physical file range.
            }
            Assert.All(budget.Snapshot(), p => Assert.Equal(0, p.Committed + p.Reserved));
        }
        finally { File.Delete(path); }
    }

    private sealed class TrackingSource(long length) : IResourceSource
    {
        public long ByteLength => length;
        public int Reads { get; private set; }
        public bool Fail { get; set; }
        public ValueTask ReadAsync(long offset, Memory<byte> destination, CancellationToken cancellationToken = default)
        {
            Reads++;
            if (Fail) throw new IOException("Injected read failure");
            destination.Span.Fill(0x37);
            return ValueTask.CompletedTask;
        }
    }
}
