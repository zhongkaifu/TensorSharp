// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp.Runtime;
using TensorSharp.Memory;

namespace InferenceWeb.Tests;

[CollectionDefinition("GGUF prefault environment", DisableParallelization = true)]
public sealed class GgufPrefaultCollection { }

[Collection("GGUF prefault environment")]
public sealed class GgufPrefaultTests : IDisposable
{
    private readonly string _directory = Path.Combine(Path.GetTempPath(), "ts-prefault-" + Guid.NewGuid());
    private readonly Dictionary<string, string?> _environment = new();
    private const int TensorBytes = 65536;

    public GgufPrefaultTests()
    {
        Directory.CreateDirectory(_directory);
        Set("TS_GGUF_PREFAULT", "1");
        Set("TS_GGUF_PREFAULT_RESIDENT", "0");
        Set("TS_GGUF_PREFAULT_THREADS", "2");
    }

    [Fact]
    public void ExcludedSparseTableDoesNotGetReadAndRemainsAvailable()
    {
        string path = Write("model.gguf", false, 0, "sparse", "dense");
        using var file = new GgufFile(path);
        file.PrefaultFileCache(info => info.Name != "sparse");
        Assert.Equal(TensorBytes, file.PrefaultReadBytes);
        Assert.Equal(0, file.PrefaultResidentBytes);
        Assert.All(file.ReadTensorData(file.Tensors["sparse"]), value => Assert.Equal((byte)17, value));
        Assert.All(file.ReadTensorData(file.Tensors["dense"]), value => Assert.Equal((byte)18, value));
    }

    [Fact]
    public void SplitCheckpointPrefaultUsesEachTensorsOwningFile()
    {
        string first = Write("split-00001-of-00002.gguf", true, 0, "first");
        Write("split-00002-of-00002.gguf", true, 1, "second");
        using var file = new GgufFile(first);
        file.PrefaultFileCache(info => info.Name == "second");
        Assert.Equal(TensorBytes, file.PrefaultReadBytes);
        Assert.All(file.ReadTensorData(file.Tensors["second"]), value => Assert.Equal((byte)18, value));
    }

    [Fact]
    public void EmptySelectionDoesNotReadTheCheckpoint()
    {
        using var file = new GgufFile(Write("empty.gguf", false, 0, "weight"));
        file.PrefaultFileCache(_ => false);
        Assert.Equal(0, file.PrefaultReadBytes);
        Assert.Equal(0, file.PrefaultResidentBytes);
    }

    [Fact]
    public void UnsupportedAuxiliaryTensorDoesNotPreventWarmingSupportedWeights()
    {
        using var file = new GgufFile(Write("auxiliary.gguf", false, 0, "auxiliary", "weight"));
        file.Tensors["auxiliary"].Type = (GgmlTensorType)int.MaxValue;
        file.PrefaultFileCache();
        Assert.Equal(TensorBytes, file.PrefaultReadBytes);
        Assert.All(file.ReadTensorData(file.Tensors["weight"]), value => Assert.Equal((byte)18, value));
    }

    [Fact]
    public void DisabledWarmupStillAllowsReadingWeights()
    {
        Set("TS_GGUF_PREFAULT", "0");
        using var file = new GgufFile(Write("disabled.gguf", false, 0, "weight"));
        file.PrefaultFileCache();
        Assert.Equal(0, file.PrefaultReadBytes);
        Assert.Equal(TensorBytes, file.ReadTensorData(file.Tensors["weight"]).Length);
    }

    [Fact]
    public void PrefaultUsesPhysicalAvailabilityIncludingItsConcurrentReadBuffers()
    {
        Assert.False(GgufFile.CanPrefaultFileCache(1, 0, 1));
        Assert.False(GgufFile.CanPrefaultFileCache(64 << 20, 128 << 20, 2));
        Assert.True(GgufFile.CanPrefaultFileCache(64 << 20, 160 << 20, 2));
        Assert.False(GgufFile.CanPrefaultFileCache(long.MaxValue, long.MaxValue, int.MaxValue));
    }

    [Fact]
    public void PrefaultReadBufferRemainsChargedUntilPhysicalFree()
    {
        using var file = new GgufFile(Write("budget.gguf", false, 0, "weight"));
        var budget = new MemoryBudget([new("ram", TensorBytes)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        file.PrefaultFileCache();
        Assert.Equal(TensorBytes, file.PrefaultReadBytes);
        Assert.Equal(TensorBytes, scope.Usage.PeakBytes);
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.Equal(0, scope.Usage.Allocations);
        Assert.Equal(0, budget.Snapshot().Single().Committed);
        Assert.Equal(0, budget.Snapshot().Single().Reserved);
    }

    [Fact]
    public void PrefaultQuotaRefusalSkipsOptionalWarmupWithoutBreakingTensorReads()
    {
        using var file = new GgufFile(Write("budget-refusal.gguf", false, 0, "weight"));
        var budget = new MemoryBudget([new("ram", 0)]);
        using var scope = new HostAllocationBudgetScope(budget, ["ram"]);
        file.PrefaultFileCache();
        Assert.Equal(0, file.PrefaultReadBytes);
        Assert.Equal(0, scope.Usage.Bytes);
        Assert.All(file.ReadTensorData(file.Tensors["weight"]), value => Assert.Equal((byte)17, value));
    }

    [Fact]
    public void ResidentCheckPreservesDataAndAccountsForEveryByte()
    {
        Set("TS_GGUF_PREFAULT_RESIDENT", "1");
        using var file = new GgufFile(Write("resident.gguf", false, 0, "a", "b"));
        file.PrefaultFileCache();
        Assert.Equal(2L * TensorBytes, file.PrefaultReadBytes + file.PrefaultResidentBytes);
        // mincore is best effort; hosts may restrict it. Data correctness is
        // required whether the optimization succeeds or falls back to reads.
        Assert.All(file.ReadTensorData(file.Tensors["b"]), value => Assert.Equal((byte)18, value));
        long read = file.PrefaultReadBytes;
        file.PrefaultFileCache();
        Assert.Equal(read, file.PrefaultReadBytes);
    }

    [Fact]
    public void ParallelReadersCrossTensorBoundariesWithoutReadingSparseGaps()
    {
        const int bytes = (6 << 20) + 128;
        using var file = new GgufFile(WriteSized("parallel.gguf", false, 0, bytes, "a", "b", "sparse", "c"));
        file.PrefaultFileCache(info => info.Name != "sparse");
        Assert.Equal(3L * bytes, file.PrefaultReadBytes);
        Assert.True(file.ReadTensorData(file.Tensors["c"]).All(value => value == 20));
        Assert.True(file.ReadTensorData(file.Tensors["sparse"]).All(value => value == 19));
    }

    private string Write(string name, bool split, int shard, params string[] tensors)
        => WriteSized(name, split, shard, TensorBytes, tensors);

    private string WriteSized(string name, bool split, int shard, int tensorBytes, params string[] tensors)
    {
        string path = Path.Combine(_directory, name);
        using var writer = new BinaryWriter(File.Create(path), Encoding.UTF8);
        writer.Write(0x46554747u);
        writer.Write(3u);
        writer.Write((ulong)tensors.Length);
        writer.Write(split ? 2ul : 0ul);
        if (split)
        {
            Text(writer, "split.count"); writer.Write((uint)GgufValueType.Uint16); writer.Write((ushort)2);
            Text(writer, "split.no"); writer.Write((uint)GgufValueType.Uint16); writer.Write((ushort)shard);
        }
        for (int i = 0; i < tensors.Length; i++)
        {
            Text(writer, tensors[i]); writer.Write(1u); writer.Write((ulong)(tensorBytes / 4));
            writer.Write((uint)GgmlTensorType.F32); writer.Write((ulong)(i * tensorBytes));
        }
        while (writer.BaseStream.Position % 32 != 0) writer.Write((byte)0);
        for (int i = 0; i < tensors.Length; i++)
            writer.Write(Enumerable.Repeat((byte)(17 + shard + i), tensorBytes).ToArray());
        return path;
    }

    private static void Text(BinaryWriter writer, string value)
    {
        byte[] bytes = Encoding.UTF8.GetBytes(value);
        writer.Write((ulong)bytes.Length); writer.Write(bytes);
    }

    private void Set(string name, string value)
    {
        _environment.TryAdd(name, Environment.GetEnvironmentVariable(name));
        Environment.SetEnvironmentVariable(name, value);
    }

    public void Dispose()
    {
        foreach (var entry in _environment) Environment.SetEnvironmentVariable(entry.Key, entry.Value);
        Directory.Delete(_directory, recursive: true);
    }
}
