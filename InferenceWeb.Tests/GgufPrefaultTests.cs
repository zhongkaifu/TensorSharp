// Copyright (c) Zhongkai Fu. All rights reserved.
// Licensed under the BSD-3-Clause license in the repository root.
using System.Text;
using TensorSharp.Runtime;
using TensorSharp.Models.MiniMaxH3;

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
        Set("TS_H3_TEXT_PREFAULT", "1");
        Set("TS_H3_PHASE", "0");
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

    [Fact]
    public void H3TextWarmupReadsOnlyConsumedProjectionsAndHonorsLayerLimit()
    {
        string[] projections=["self_attn.q_proj.weight","self_attn.k_proj.weight","self_attn.v_proj.weight",
            "self_attn.o_proj.weight","mlp.gate_proj.weight","mlp.up_proj.weight","mlp.down_proj.weight"];
        string[] names=projections.Select(p => "model.layers.0."+p).Concat([
            "model.embed_tokens.weight","visual.blocks.0.attn.qkv.weight","model.layers.1.mlp.up_proj.weight",
            "model.layers.-1.mlp.up_proj.weight","model.layers.0.unconsumed.weight","model.layers.0.input_layernorm.weight"
        ]).ToArray();
        using var file=new GgufFile(Write("h3-selected.gguf",false,0,names));
        MiniMaxH3DirectTextEncoder.PrefaultTextWeights(file,1);
        Assert.Equal(7L*TensorBytes,file.PrefaultReadBytes);
        for (int i=0; i<names.Length; i++)
            Assert.All(file.ReadTensorData(file.Tensors[names[i]]),value => Assert.Equal((byte)(17+i),value));
    }

    [Fact]
    public void H3TextWarmupSupportsSplitWeightsAndOptOut()
    {
        string first=Write("h3-00001-of-00002.gguf",true,0,"model.layers.0.self_attn.q_proj.weight");
        Write("h3-00002-of-00002.gguf",true,1,"model.layers.0.mlp.down_proj.weight","model.embed_tokens.weight");
        using var file=new GgufFile(first);
        Set("TS_H3_TEXT_PREFAULT","0");
        MiniMaxH3DirectTextEncoder.PrefaultTextWeights(file,1);
        Assert.Equal(0,file.PrefaultReadBytes);
        Set("TS_H3_TEXT_PREFAULT","1");
        MiniMaxH3DirectTextEncoder.PrefaultTextWeights(file,1);
        Assert.Equal(2L*TensorBytes,file.PrefaultReadBytes);
        Assert.All(file.ReadTensorData(file.Tensors["model.embed_tokens.weight"]),value => Assert.Equal((byte)19,value));
    }

    [Fact]
    public void H3TextWarmupReportsOnlyWhenPhaseTimingIsRequested()
    {
        using var file=new GgufFile(Write("h3-logging.gguf",false,0,"model.layers.0.mlp.up_proj.weight"));
        using var output=new StringWriter();
        TextWriter original=Console.Out;
        try
        {
            Console.SetOut(output);
            MiniMaxH3DirectTextEncoder.PrefaultTextWeights(file,1);
            Assert.Equal(string.Empty,output.ToString());
            Set("TS_H3_PHASE","1");
            MiniMaxH3DirectTextEncoder.PrefaultTextWeights(file,1);
            Assert.Contains("[h3] .. text weight prefault",output.ToString());
        }
        finally { Console.SetOut(original); }
    }

    [Fact]
    public void ACheckpointLargerThanTheBudgetStillWarmsItsEagerTensors()
    {
        // Three tensors against a two-tensor budget: before, nothing was warmed and every eagerly
        // loaded weight arrived through serial page faults (505 s for Flash Next on geesefs).
        using var file = new GgufFile(Write("constrained.gguf", false, 0,
            "blk.0.ffn_up_exps.weight", "dense", "blk.0.ffn_down_exps.weight"));
        try
        {
            GgufFile.AvailableMemoryOverrideForTests = 4L * TensorBytes;
            file.PrefaultFileCache(null, info => !info.Name.Contains("_exps"), logProgress: false);
        }
        finally { GgufFile.AvailableMemoryOverrideForTests = null; }
        Assert.Equal(TensorBytes, file.PrefaultReadBytes);
        Assert.All(file.ReadTensorData(file.Tensors["dense"]), value => Assert.Equal((byte)18, value));
        Assert.All(file.ReadTensorData(file.Tensors["blk.0.ffn_down_exps.weight"]), value => Assert.Equal((byte)19, value));
    }

    [Fact]
    public void AnOversizedCheckpointWithoutAnEagerSubsetIsNotWarmed()
    {
        using var file = new GgufFile(Write("oversized.gguf", false, 0, "a", "b", "c"));
        try
        {
            GgufFile.AvailableMemoryOverrideForTests = 4L * TensorBytes;
            file.PrefaultFileCache(null, logProgress: false);
        }
        finally { GgufFile.AvailableMemoryOverrideForTests = null; }
        Assert.Equal(0, file.PrefaultReadBytes);
    }

    [Fact]
    public void ReadHandlesAreOpenedOnceAndOutliveEveryWarm()
    {
        // geesefs drops a file's whole page cache at each open(): later warms must reuse the
        // handles opened before the first read instead of opening their own.
        using var file = new GgufFile(Write("handles.gguf", false, 0, "a", "b"));
        var handles = file.ReadHandles();
        file.PrefaultFileCache();
        Assert.True(file.TryGetTensorDataPointer(file.Tensors["b"], out IntPtr b));
        Assert.Equal(TensorBytes, file.WarmMappedRanges(new[] { (b, (long)TensorBytes) }, 4));
        Assert.Same(handles, file.ReadHandles());
        Assert.All(handles, handle => Assert.False(handle.IsClosed));
    }

    [Fact]
    public void WarmMappedRangesResolvesEachPointerToItsShardAndSkipsForeignMemory()
    {
        string first = Write("warm-00001-of-00002.gguf", true, 0, "first");
        Write("warm-00002-of-00002.gguf", true, 1, "second");
        using var file = new GgufFile(first);
        Assert.True(file.TryGetTensorDataPointer(file.Tensors["first"], out IntPtr a));
        Assert.True(file.TryGetTensorDataPointer(file.Tensors["second"], out IntPtr b));
        Assert.True(file.TryResolveMapped(b, TensorBytes, out var owner, out long offset));
        Assert.NotSame(file, owner);
        Assert.True(offset > 0);
        byte[] foreign = new byte[64];
        unsafe
        {
            fixed (byte* p = foreign)
            {
                long read = file.WarmMappedRanges(new[] { (a, (long)TensorBytes), (b + 100, 1000L), ((IntPtr)p, 64L) }, 4);
                Assert.Equal(TensorBytes + 1000L, read);
            }
        }
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
