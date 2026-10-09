// Copyright (c) Zhongkai Fu. Licensed under the BSD-3-Clause license.
using TensorSharp.Memory;
using TensorSharp.Memory.Planning;
using TensorSharp.Models;
using TensorSharp.Runtime;
using Xunit;

namespace InferenceWeb.Tests;

public sealed class DenseMemoryProfileTests
{
    private const long RuntimeAllowance = 128L << 20;
    private const long BufferPadding = 64L << 10;

    [Fact]
    public void NonMatrixHalfTensorIsDecodedAndKeepsItsFloatHostPayload()
    {
        using var f = new Fixture("gemma4", Matrix("auxiliary.weight", SyntheticGguf.GgmlType.F16, 64, 2, 2));
        var profile = f.Read();
        Assert.Equal(64 * 2 * 2 * 4, profile.ResidentHostWeightBytes);
        Assert.Equal(64 * 2 * 2 * 2, profile.FusionBytes); // temporary encoded source during load
        Assert.Equal(272 + 64 * 2 * 2 * 4, profile.Model.DenseWeights.Host);
    }

    [Theory]
    [InlineData(0, 512, 512)]
    [InlineData(1, 256, 0)]
    [InlineData(30, 256, 0)]
    [InlineData(8, 136, 0)]
    public void MatrixStorageMatchesLoaderAndSeparatesMappedBytesFromAnonymousHost(
        int type, long matrixBytes, long retainedHost)
    {
        using var f = new Fixture("gemma4", Matrix("projection.weight", (SyntheticGguf.GgmlType)type, 64, 2),
            Matrix("output_norm.weight", SyntheticGguf.GgmlType.F16, 64));
        var profile = f.Read();
        const long embeddingBytes = 64 / 32 * 34 * 4;
        Assert.Equal(embeddingBytes + matrixBytes + 128, profile.SourceWeightBytes);
        Assert.Equal(embeddingBytes + matrixBytes, profile.Model.DenseWeights.Host);
        Assert.Equal(embeddingBytes + matrixBytes + 3 * BufferPadding, profile.Model.DenseWeights.Device);
        Assert.Equal(retainedHost, profile.ResidentHostWeightBytes);
        Assert.Equal(embeddingBytes, profile.RetainedMappedWeightBytes);
        Assert.Equal(RuntimeAllowance + 256, profile.Model.Persistent.Host);
        Assert.Equal(128, profile.FusionBytes); // one vector's temporary F16 read buffer
    }

    [Fact]
    public void FloatQkvMissingValueCountsItsRetainedCopyAndOnlyOneConstructionTemporary()
    {
        using var f = new Fixture("gemma4", Matrix("blk.0.attn_q.weight", SyntheticGguf.GgmlType.F32, 64, 128),
            Matrix("blk.0.attn_k.weight", SyntheticGguf.GgmlType.F32, 64, 64));
        var profile = f.Read();
        long packed = 64 * (128 + 64 + 64) * 4;
        Assert.Equal(packed, profile.ResidentHostWeightBytes);
        Assert.Equal(packed, profile.FusionBytes);
        Assert.Equal(profile.Model.DenseWeights.Host + 64 * 64 * 4 + 3 * BufferPadding,
            profile.Model.DenseWeights.Device);
    }

    [Fact]
    public void RetainHostPolicyMovesQuantizedFusionIntoPersistentWithoutCountingItTwiceAtLoad()
    {
        using var f = new Fixture("gemma4", Matrix("blk.0.attn_q.weight", SyntheticGguf.GgmlType.Q8_0, 64, 128),
            Matrix("blk.0.attn_k.weight", SyntheticGguf.GgmlType.Q8_0, 64, 64));
        var normal = f.Read();
        var retained = f.Read(retainHost: true);
        long packed = 68 * (128 + 64 + 64);
        Assert.Equal(0, normal.ResidentHostWeightBytes);
        Assert.Equal(packed, normal.FusionBytes);
        Assert.Equal(packed, retained.ResidentHostWeightBytes);
        Assert.Equal(0, retained.FusionBytes);
        Assert.Equal(normal.Model.DenseWeights.Device, retained.Model.DenseWeights.Device);
        Assert.Equal(normal.FusionBytes, retained.ResidentHostWeightBytes + retained.FusionBytes);
    }

    [Theory]
    [InlineData("gemma4")]
    [InlineData("qwen35")]
    public void MixedGateUpPreservesOriginalFormatsWithoutRequantizationAllocations(string architecture)
    {
        using var f = new Fixture(architecture, Matrix("blk.0.ffn_gate.weight", SyntheticGguf.GgmlType.Q8_0, 64, 2),
            Matrix("blk.0.ffn_up.weight", SyntheticGguf.GgmlType.F16, 64, 2));
        var profile = f.Read();
        Assert.Equal(0, profile.ResidentHostWeightBytes);
        Assert.Equal(profile.Model.DenseWeights.Host + 3 * BufferPadding,
            profile.Model.DenseWeights.Device);
        Assert.Equal(0, profile.FusionBytes);
    }

    [Theory]
    [InlineData(false)]
    [InlineData(true)]
    public void RecurrentInputPackRetainsOnlyRawSourcesAndBoundsFloatReplacementTemporary(bool floatWeights)
    {
        var type = floatWeights ? SyntheticGguf.GgmlType.F32 : SyntheticGguf.GgmlType.F16;
        using var f = new Fixture("qwen35", Matrix("blk.0.attn_qkv.weight", type, 64, 128),
            Matrix("blk.0.attn_gate.weight", type, 64, 64), Matrix("blk.0.ssm_beta.weight", type, 64, 4),
            Matrix("blk.0.ssm_alpha.weight", type, 64, 4));
        var profile = f.Read();
        long pack = 64 * (128 + 64 + 4 + 4) * (floatWeights ? 4 : 2);
        Assert.Equal(profile.Model.DenseWeights.Host + (floatWeights ? 0 : pack) + 5 * BufferPadding,
            profile.Model.DenseWeights.Device);
        Assert.Equal(floatWeights ? pack : 0, profile.ResidentHostWeightBytes);
        Assert.Equal(pack, profile.FusionBytes);
    }

    [Fact]
    public void MixedRecurrentPackStaysSplitOnSingleRankAndDoesNotInventAFloatCopy()
    {
        using var f = new Fixture("qwen35", Matrix("blk.0.attn_qkv.weight", SyntheticGguf.GgmlType.Q8_0, 64, 128),
            Matrix("blk.0.attn_gate.weight", SyntheticGguf.GgmlType.Q8_0, 64, 64),
            Matrix("blk.0.ssm_beta.weight", SyntheticGguf.GgmlType.F32, 64, 4),
            Matrix("blk.0.ssm_alpha.weight", SyntheticGguf.GgmlType.F32, 64, 4));
        var profile = f.Read();
        Assert.Equal(profile.Model.DenseWeights.Host + 5 * BufferPadding, profile.Model.DenseWeights.Device);
        Assert.Equal(64 * 8 * 4, profile.ResidentHostWeightBytes);
        Assert.Equal(0, profile.FusionBytes);
    }

    [Fact]
    public void GemmaKvUsesActualKeyDimensionsAndCountsSharedPhysicalDonorsOnce()
    {
        using var f = new Fixture("gemma4", Matrix("blk.0.attn_k.weight", SyntheticGguf.GgmlType.F16, 64, 64),
            Matrix("blk.1.attn_k.weight", SyntheticGguf.GgmlType.F16, 64, 96));
        f.File.Metadata["gemma4.block_count"] = 4u;
        f.File.Metadata["gemma4.attention.sliding_window_pattern"] = new[] { true, false, true, false };
        f.File.Metadata["gemma4.attention.head_count_kv"] = new[] { 2, 1, 2, 1 };
        f.File.Metadata["gemma4.attention.shared_kv_layers"] = 2u;
        var profile = f.Read();
        var gpu = profile.Model.KvCaches.Where(k => k.Tier == MemoryTier.Accelerator).ToArray();
        Assert.Equal(2, gpu.Length);
        Assert.Equal(8 * 2 * 32, gpu[0].BytesPerToken);
        Assert.Equal(512, gpu[0].WindowTokens);
        Assert.Equal(8 * 96, gpu[1].BytesPerToken);
        Assert.Equal(0, gpu[1].WindowTokens);
        Assert.Equal(4, profile.Model.KvCaches.Count);
    }

    [Fact]
    public void GemmaShortContextStillAdmitsTheWholePhysicalSlidingWindowPerDonor()
    {
        using var f = new Fixture("gemma4", Matrix("blk.0.attn_k.weight", SyntheticGguf.GgmlType.F16, 64, 64));
        f.File.Metadata["gemma4.block_count"] = 2u;
        f.File.Metadata["gemma4.attention.sliding_window_pattern"] = new[] { true, true };
        f.File.Metadata["gemma4.attention.sliding_window"] = 1024u;
        f.File.Metadata["gemma4.attention.shared_kv_layers"] = 1u;
        var profile = f.Read();
        var plan = InferenceMemoryPlanner.Plan(new()
        {
            Pools = [new("ram", long.MaxValue, long.MaxValue, 0, new("ram", long.MaxValue, 0, 0)),
                new("gpu", long.MaxValue, long.MaxValue, 0, new("gpu", long.MaxValue, 0, 0))],
            HostPools = ["ram"], DevicePools = ["gpu"], Model = profile.Model,
            Workload = new(128, 128, 1, 1),
            Candidates = [new() { Name = "resident", Placement = InferenceWeightPlacement.Resident, PrefillChunkTokens = 128 }]
        });
        Assert.True(plan.Accepted);
        Assert.Equal(2, profile.Model.KvCaches.Count); // one donor, charged once per tier
        Assert.All(profile.Model.KvCaches, kv => Assert.Equal(1024, kv.AllocationBlockTokens));
        var components = plan.Components.Where(c => c.Name.StartsWith("kv-", StringComparison.Ordinal)).ToArray();
        const long completeF32Ring = 1024 * 8 * 64;
        Assert.Equal(completeF32Ring, components.Sum(c => c.Bytes.Host));
        Assert.Equal(completeF32Ring, components.Sum(c => c.Bytes.Device));
    }

    [Theory]
    [InlineData("scalar-array")]
    [InlineData("pattern-short")]
    [InlineData("heads-short")]
    [InlineData("heads-negative")]
    [InlineData("embedding-width")]
    [InlineData("expert-payload")]
    [InlineData("transcoded-format")]
    public void UnknownGeometryIsRejectedBeforeAnOptimisticPlan(string mutation)
    {
        using var f = new Fixture("gemma4");
        switch (mutation)
        {
            case "scalar-array": f.File.Metadata["gemma4.embedding_length"] = new[] { 64, 128 }; break;
            case "pattern-short": f.File.Metadata["gemma4.attention.sliding_window_pattern"] = Array.Empty<bool>(); break;
            case "heads-short": f.File.Metadata["gemma4.attention.head_count_kv"] = Array.Empty<int>(); break;
            case "heads-negative": f.File.Metadata["gemma4.attention.head_count_kv"] = new[] { -1 }; break;
            case "embedding-width": f.File.Tensors["token_embd.weight"].Shape[0] = 32; break;
            case "expert-payload": f.File.Tensors.Add("blk.0.ffn_gate_exps.weight", new()
                { Name = "blk.0.ffn_gate_exps.weight", Shape = [64, 2, 2], Type = GgmlTensorType.F32 }); break;
            case "transcoded-format": f.File.Tensors["token_embd.weight"].Type = GgmlTensorType.PQ2_0; break;
        }
        Assert.Throws<NotSupportedException>(() => f.Read());
    }

    [Theory]
    [InlineData("unknown-layer")]
    [InlineData("short-layers")]
    [InlineData("missing-time-rank")]
    [InlineData("indivisible-time-rank")]
    public void QwenDoesNotSilentlyReplaceInvalidRecurrentMetadataWithDefaults(string mutation)
    {
        using var f = new Fixture("qwen35");
        switch (mutation)
        {
            case "unknown-layer": f.File.Metadata["qwen35.layer_types"] = new[] { "mystery_attention" }; break;
            case "short-layers": f.File.Metadata["qwen35.layer_types"] = Array.Empty<string>(); break;
            case "missing-time-rank": f.File.Metadata.Remove("qwen35.ssm.time_step_rank"); break;
            case "indivisible-time-rank": f.File.Metadata["qwen35.ssm.time_step_rank"] = 3u; break;
        }
        Assert.Throws<NotSupportedException>(() => f.Read());
    }

    [Theory]
    [InlineData(256, 128, 256)]
    [InlineData(128, 256, 128)]
    [InlineData(0, 192, 192)]
    [InlineData(0, 0, 16)]
    public void QwenKvMatchesBothPhysicalCacheTensorsWhenMetadataLengthsDifferOrAreOmitted(
        int keyLength, int valueLength, int expectedHeadDimension)
    {
        using var f = new Fixture("qwen35");
        f.File.Metadata["qwen35.full_attention_interval"] = 1u;
        if (keyLength != 0) f.File.Metadata["qwen35.attention.key_length"] = (uint)keyLength;
        if (valueLength != 0) f.File.Metadata["qwen35.attention.value_length"] = (uint)valueLength;
        var profile = f.Read();
        // Qwen allocates K and V as [kvHeads, capacity, Config.HeadDim].
        // Compare with the loader's dimension precedence, not key+value or max.
        var config = new ModelConfig { HiddenSize = 64, NumHeads = 4,
            KeyLength = keyLength, ValueLength = valueLength };
        Assert.Equal(expectedHeadDimension, config.HeadDim);
        Assert.Equal(2, profile.Model.KvCaches.Count);
        foreach (MemoryTier tier in new[] { MemoryTier.Host, MemoryTier.Accelerator })
        {
            var cache = Assert.Single(profile.Model.KvCaches.Where(k => k.Tier == tier));
            Assert.Equal(2L * 2 * expectedHeadDimension * sizeof(float), cache.BytesPerToken);
            Assert.Equal(256, cache.AllocationBlockTokens);
        }
    }

    private static SyntheticGguf.Tensor Matrix(string name, SyntheticGguf.GgmlType type, params int[] shape)
        => new() { Name = name, Type = type, Dims = shape.Select(n => (ulong)n).ToArray(),
            Data = new float[shape.Aggregate(1, (a, n) => checked(a * n))] };

    private sealed class Fixture : IDisposable
    {
        private readonly string _path = Path.Combine(Path.GetTempPath(), $"ts-dense-profile-{Guid.NewGuid():N}.gguf");
        public GgufFile File { get; }
        public Fixture(string arch, params SyntheticGguf.Tensor[] tensors)
        {
            var metadata = new List<SyntheticGguf.Kv>
            {
                new SyntheticGguf.Str { Key = "general.architecture", V = arch },
                new SyntheticGguf.U32 { Key = $"{arch}.block_count", V = 1 },
                new SyntheticGguf.U32 { Key = $"{arch}.embedding_length", V = 64 },
                new SyntheticGguf.U32 { Key = $"{arch}.feed_forward_length", V = 128 },
                new SyntheticGguf.U32 { Key = $"{arch}.attention.head_count", V = 4 },
                new SyntheticGguf.U32 { Key = $"{arch}.attention.head_count_kv", V = 2 },
                new SyntheticGguf.U32 { Key = $"{arch}.ssm.inner_size", V = 64 },
                new SyntheticGguf.U32 { Key = $"{arch}.ssm.state_size", V = 16 },
                new SyntheticGguf.U32 { Key = $"{arch}.ssm.group_count", V = 2 },
                new SyntheticGguf.U32 { Key = $"{arch}.ssm.time_step_rank", V = 4 },
                new SyntheticGguf.U32 { Key = $"{arch}.ssm.conv_kernel", V = 4 }
            };
            SyntheticGguf.Write(_path, metadata,
                [Matrix("token_embd.weight", SyntheticGguf.GgmlType.Q8_0, 64, 4), .. tensors]);
            File = new GgufFile(_path);
        }
        public DenseMemoryProfile Read(bool retainHost = false)
            => DenseMemoryProfile.Read(File, new AdaptiveModelMemoryOptions(1024, 64), retainHost);
        public void Dispose() { File.Dispose(); System.IO.File.Delete(_path); }
    }
}
